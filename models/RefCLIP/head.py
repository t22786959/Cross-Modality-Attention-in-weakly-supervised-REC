# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


def getContrast(vis_emb, lan_emb):
    def process_case(lan_emb_slice, vis_emb, batchsize, target_pred):
        lan_adjancency = (lan_emb_slice @ lan_emb_slice.transpose(0, 1))    ## language  adjancency matrix (batchsize x batchsize) 
        
        lan_emb_slice = lan_emb_slice.unsqueeze(1)
        sim_map = torch.einsum('avd, bqd -> baqv', vis_emb, lan_emb_slice) 
        max_sims, indices = sim_map.topk(k=2, dim=-1, largest=True, sorted=True)   ## similarity  selection 
        max_sims = max_sims.squeeze(2)
        indices = indices.squeeze(2)
        
        # FNS loss
        diagonal_elements = indices[torch.arange(batchsize), torch.arange(batchsize), 0]   ## diagonal_elements positive samples 
        temp_list = []
        for i in range(batchsize):
            temp = vis_emb[i, diagonal_elements[i], :]
            temp_list.append(temp)
        selected_tensor = torch.stack(temp_list)
        img_adjancency = (selected_tensor@selected_tensor.transpose(0,1))    ## image adjancency matrix (batchsize x batchsize) 

        selected_tensor = selected_tensor.unsqueeze(2)
        selected_tensor = selected_tensor.unsqueeze(2)
        lan_emb_slice = lan_emb_slice.squeeze(1)
        Slogits = torch.einsum('nchw,mc->nmhw', selected_tensor, lan_emb_slice)   ## text to image similarity matrix
        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]

        # Contrastive loss
        max_sim_0, max_sim_1 = max_sims[..., 0], max_sims[..., 1]
        max_sim_0_regu = max_sim_0
        max_sim_1 = max_sim_1.masked_select(~torch.eye(batchsize).bool().to(max_sim_1.device)).contiguous().view(batchsize, batchsize - 1)
        new_logits = torch.cat([max_sim_0, max_sim_1], dim=1)
        loss = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)

        # Regularization loss
        sample_num = batchsize
        with torch.no_grad():
            weights_i2t = F.softmax(max_sim_0_regu[:batchsize, :], dim=1)
            weights_i2t.fill_diagonal_(0)
        
        images_neg_idx, text_ids_neg_idx = [], []
        for b in range(batchsize):
            if sum(weights_i2t[b]) == 0:
                images_neg_idx.append(neg_idx_topk)
                continue
            neg_idx_topk = torch.cat((torch.arange(0, b), torch.arange(b+1, batchsize)))   # regularize each sample in a batch
            images_neg_idx.append(neg_idx_topk)

        max_sim_0_regu = F.softmax(max_sim_0_regu, dim=1)
        pair_loss_regular = get_regularization_loss(max_sim_0_regu, images_neg_idx, images_neg_idx, bsz=batchsize, sample_num=sample_num, return_mask=False)
        
        loss_fns = F.l1_loss(torch.softmax((img_adjancency + lan_adjancency)/2, dim=1), torch.softmax(logits, dim=1))

        alpha = nn.Parameter(torch.tensor(0.1))
        beta = nn.Parameter(torch.tensor(0.1))
        return loss + alpha*loss_fns + beta*pair_loss_regular
    
    batchsize = vis_emb.shape[0]
    target = torch.eye(batchsize).to(vis_emb.device)
    target_pred = torch.argmax(target, dim=1)  
    loss = sum(process_case(lan_emb[:, i, :], vis_emb, batchsize, target_pred) for i in range(3)) / 3
    return loss

def getPrediction(vis_emb, lan_emb):
    sim_map = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)
    maxval, v = sim_map.max(dim=2, keepdim=True)
    predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2,v.expand(sim_map.shape), 1).bool()
    return predictions

class WeakREChead(nn.Module):
    def __init__(self, __C):
        super(WeakREChead, self).__init__()

    def forward(self, vis_fs,lan_fs):
        if self.training:
            loss = getContrast(vis_fs, lan_fs)
            return loss
        else:
            predictions = getPrediction(vis_fs, lan_fs)
            return predictions



def get_regularization_loss( score_matrix, t2i_neg_idx, i2t_neg_idx, bsz, sample_num, return_mask=False):
    score_matrix_size = bsz # score matrix i2t
    global_batch_size = score_matrix.size(1)
    mask = torch.ones(global_batch_size, global_batch_size).to(score_matrix.device)

    caption0_image0_score, caption0_imagei_score, captioni_image0_score, captioni_imagei_score = [], [], [], []
    for idx in range(score_matrix_size):
        if sample_num>0:
            idx_combine = torch.cat([t2i_neg_idx[idx], i2t_neg_idx[idx]])
        else:
            idx_combine = [t2i_neg_idx[idx], i2t_neg_idx[idx]]
        caption0_image0_score.append(score_matrix[idx, idx].unsqueeze(0).expand(len(idx_combine))) # 2 denotes both image/text-side
        caption0_imagei_score.append(score_matrix[idx_combine, idx])
        captioni_image0_score.append(score_matrix[idx, idx_combine])
        captioni_imagei_score.append(score_matrix.diag()[idx_combine])

        if return_mask:
            mask[idx, idx_combine] = 0.
            mask[idx_combine, idx] = 0.
    caption0_image0_score = torch.cat(caption0_image0_score, dim=-1).unsqueeze(1)
    caption0_imagei_score = torch.cat(caption0_imagei_score, dim=-1).unsqueeze(1)
    captioni_image0_score = torch.cat(captioni_image0_score, dim=-1).unsqueeze(1)
    captioni_imagei_score = torch.cat(captioni_imagei_score, dim=-1).unsqueeze(1)

    loss_func = pairloss_criterion( )
    pairloss = loss_func(caption0_image0_score, captioni_image0_score, caption0_imagei_score, captioni_imagei_score)
                                                     
    if return_mask:
        return pairloss, mask
    else:
        return pairloss

class pairloss_criterion(nn.Module):
    def __init__(self, text_side=False, img_side=False, group_side=False, pos_reg=False, norm=False, weight=1.0, margin=0.0):
        super(pairloss_criterion, self).__init__()
        self.text_side = text_side
        self.img_side = img_side
        self.group_side = group_side
        self.pos_reg = pos_reg
        self.norm = norm
        self.weight = weight
        self.margin = margin # smaller than margin -> loss=0.
        self.mseloss = nn.L1Loss()
        self.relu = nn.ReLU(inplace=False)

    def __call__(self, c0_i0, c1_i0, c0_i1, c1_i1):
        loss = 0.
        loss += (self.mseloss((c0_i0-c1_i0), (c1_i1-c0_i1))- self.margin).clamp(min=0.)
        loss += (self.mseloss((c0_i0-c0_i1), (c1_i1-c1_i0))- self.margin).clamp(min=0.)
        
        return loss




