# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from models.Multi_Modal_Seg.UNet import RegionModule

class RegionCo(nn.Module):
    """
    Build a RegionCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,dim=16, K=65536, m=0.999, T=0.07, region_size=16,batch_size=2,sample_k=16):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(RegionCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.sample_k = sample_k
        self.dim = dim

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = RegionModule(16,16,16,region_size)
        self.encoder_k = RegionModule(16,16,16,region_size)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("src_queue", torch.randn(K*batch_size,dim))
        # self.src_queue = nn.functional.normalize(self.src_queue, dim=2)

        self.register_buffer("src_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("trg_queue", torch.randn(K*batch_size, dim))
        # self.trg_queue = nn.functional.normalize(self.trg_queue, dim=2)

        self.register_buffer("trg_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.batch_size = batch_size

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys,modal='src'):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        # keys = keys.squeeze(1)
        ptr = int(self.src_queue_ptr) if modal == 'src' else int(self.trg_queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if modal == 'src':
            self.src_queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.src_queue_ptr[0] = ptr
        else:
            self.trg_queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.trg_queue_ptr[0] = ptr

    def mine_hard_negsample(self,anchor, keys, hard_num=100):
        sim = torch.cosine_similarity(anchor.unsqueeze(1), keys, dim=2)  # b,1,k
        result, kmax_index = torch.topk(sim.unsquueze(1), k=hard_num, dim=2)
        hard_neg = [keys[i, 0, kmax_index[i, 0, :]] for i in range(keys.shape[0])]
        hard_neg = torch.stack(hard_neg, dim=0)  # b,k,c
        return hard_neg

    def forward(self, trg_anchor,im_q, im_k,modal='src',mine_hard=False,hard_sample=100):
        """
        Input:
            trg_anchor: a batch or anchor images,b,c,h,w,z
            im_q: a batch of query images,b,c,h,w,z
            im_k: a batch of key images,b,k,c,h,w,z
        Output:
            logits, targets
        """
        # compute query features
        anchor = self.encoder_q(trg_anchor)# anchor: BxC
        q = self.encoder_k(im_q)  # queries: BxC
        # q = nn.functional.normalize(q, dim=1)
        im_k = torch.flatten(im_k,start_dim=0,end_dim=1)#b*k,c,h,w,z
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: B*k,C
            # k = nn.functional.normalize(k, dim=1)
        neg_key = self.src_queue if modal == 'src' else self.trg_queue
        neg_key = neg_key.view(self.batch_size, self.K, self.dim).clone().detach()
        if mine_hard:
            neg_key = self.mine_hard_negsample(anchor,neg_key,hard_num=hard_sample)
        all_feat = torch.cat([q.unsqueeze(1),neg_key],dim=1)
        logits = torch.cosine_similarity(anchor.unsqueeze(1), all_feat, dim=2)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k,modal)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
