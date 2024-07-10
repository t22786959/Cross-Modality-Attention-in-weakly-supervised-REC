# coding=utf-8

import torch
import torch.nn as nn

from models.language_encoder import language_encoder
from models.visual_encoder import visual_encoder
from models.RefCLIP.head import WeakREChead
from models.network_blocks import MultiScaleFusion

from models.attention_model import  GatedCrossAttentionBlock

import numpy as np
import torch.nn.functional as F
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.select_num = __C.SELECT_NUM
        self.visual_encoder = visual_encoder(__C).eval()
        self.lang_encoder = language_encoder(__C, pretrained_emb, token_size)

        self.linear_vs = nn.Linear(1024, __C.HIDDEN_SIZE)
        self.linear_ts = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.process_model = GatedCrossAttentionBlock(dim = 512)

        self.head = WeakREChead(__C)
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), hiden_planes=1024, scaled=True)
        self.class_num = __C.CLASS_NUM
        if __C.VIS_FREEZE:
            self.frozen(self.visual_encoder)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x, y):
        # Vision and Language Encoding
        with torch.no_grad():
            boxes_all, x_, boxes_sml = self.visual_encoder(x)
        y_1 = self.lang_encoder(y[:,0,:])
        y_2 = self.lang_encoder(y[:,1,:])
        y_3 = self.lang_encoder(y[:,2,:])

        # Vision Multi Scale Fusion
        s, m, l = x_
        x_input = [l, m, s]
        l_new, m_new, s_new = self.multi_scale_manner(x_input)
        x_ = [s_new, m_new, l_new]

        # Anchor Selection
        boxes_sml_new = []
        mean_i = torch.mean(boxes_sml[0], dim=2, keepdim=True)
        mean_class = mean_i
        mean_i = mean_i.squeeze(2)[:, :, 4]
        mean_class = mean_class.squeeze(2)
        

        vals, indices = mean_i.topk(k=int(self.select_num), dim=1, largest=True, sorted=True)
        bs, gridnum, anncornum, ch = boxes_sml[0].shape
        bs_, selnum = indices.shape
        box_sml_new = boxes_sml[0].masked_select(
            torch.zeros(bs, gridnum).to(boxes_sml[0].device).scatter(1, indices, 1).bool().unsqueeze(2).unsqueeze(
                3).expand(bs, gridnum, anncornum, ch)).contiguous().view(bs, selnum, anncornum, ch)
        boxes_sml_new.append(box_sml_new)

        y_1['lang_feat'] = y_1['lang_feat'].permute(0,2,1)
        y_2['lang_feat'] = y_2['lang_feat'].permute(0,2,1)
        y_3['lang_feat'] = y_3['lang_feat'].permute(0,2,1)
        lang_fea = (y_1['lang_feat'] + y_2['lang_feat'] + y_3['lang_feat'])/3
        flat_lang_fea = y_1['flat_lang_feat']
        imag_feat_2 = x_[0]

        verification_scores = self.process_model(imag_feat_2, flat_lang_fea)
        x_[0] = verification_scores

        batchsize, dim, h, w = x_[0].size()
        i_new = x_[0].view(batchsize, dim, h * w).permute(0, 2, 1)
        bs, gridnum, ch = i_new.shape
        i_new = i_new.masked_select(
            torch.zeros(bs, gridnum).to(i_new.device).scatter(1, indices, 1).
                bool().unsqueeze(2).expand(bs, gridnum,ch)).contiguous().view(bs, selnum, ch)

        x_new = i_new
        y_1_new = self.linear_ts(y_1['flat_lang_feat'].unsqueeze(1))
        y_2_new = self.linear_ts(y_2['flat_lang_feat'].unsqueeze(1))
        y_3_new = self.linear_ts(y_3['flat_lang_feat'].unsqueeze(1))

        y_new = torch.cat((y_1_new, y_2_new), dim=1)
        y_new = torch.cat((y_new, y_3_new), dim=1)
        if self.training:
            loss = self.head(x_new, y_new)
            return loss
        else:
            predictions_s = self.head(x_new, y_1_new)
            predictions_list = [predictions_s]
            box_pred = get_boxes(boxes_sml_new, predictions_list,self.class_num)
            return box_pred
        
    
def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )


def get_boxes(boxes_sml, predictionslist,class_num):
    batchsize = predictionslist[0].size()[0]
    pred = []
    for i in range(len(predictionslist)):
        mask = predictionslist[i].squeeze(1)
        masked_pred = boxes_sml[i][mask]
        refined_pred = masked_pred.view(batchsize, -1, class_num+5)
        refined_pred[:, :, 0] = refined_pred[:, :, 0] - refined_pred[:, :, 2] / 2
        refined_pred[:, :, 1] = refined_pred[:, :, 1] - refined_pred[:, :, 3] / 2
        refined_pred[:, :, 2] = refined_pred[:, :, 0] + refined_pred[:, :, 2]
        refined_pred[:, :, 3] = refined_pred[:, :, 1] + refined_pred[:, :, 3]
        pred.append(refined_pred.data)
    boxes = torch.cat(pred, 1)
    score = boxes[:, :, 4]
    max_score, ind = torch.max(score, -1)
    ind_new = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, 5)
    box_new = torch.gather(boxes, 1, ind_new)
    return box_new
