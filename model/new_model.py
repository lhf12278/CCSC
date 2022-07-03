import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
import numpy as np
from datasets.sampler import Seeds
from .backbones.vit_pytorch import DropPath,Mlp
from loss.cross_modal_loss import CrossEntropyLabel,CrossEntropyLabelSmooth,TripletLoss



class Similar_align(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()

        self.rgb_norm = norm_layer(dim)
        self.sket_norm = norm_layer(dim)


    def forward(self, x1, x2):

        rgb_feat = self.rgb_norm(x1)
        sket_feat = self.sket_norm(x2)
        similar = euclidean_dist(rgb_feat, sket_feat)
        _, index = torch.min(similar, dim=1, keepdim=True)

        return index



class Model_interra(nn.Module):

    def __init__(self,   model1, model2):
        super().__init__()

        self.model1 = model1
        self.model2 = model2

        self.similar_align = Similar_align(dim=768)

    def forward(self, x1, x2):


        B1 = x1.shape[0]
        B2 = x2.shape[0]


        for i in range(12):
            seeds = np.random.randint(0, 256, 256)

            x1 = self.model1.base.blocks[i](x1)
            x2 = self.model2.base.blocks[i](x2)

            for j in range(B1):
                rgb_feat = x1[j][1:, :]
                sket_feat = x2[j][1:, :]
                index = self.similar_align(rgb_feat, sket_feat).cpu().numpy()

                for k in range(20):
                    seed = seeds[k]
                    sim = index[seed][0]

                    h2 = x2[j][1:, :][seed]
                    t2 = h2.cpu().detach()

                    h1 = x1[j][1:, :][sim]
                    t1 = h1.cpu().detach()

                    x2[j][1:, :][seed] = t1
                    x1[j][1:, :][sim] = t2

        return x1, x2






class Align(nn.Module):
    def __init__(self,   model1, model2, classifie1,classifie3,classifie2,numclss):
        super().__init__()

        self.model1 = model1
        self.model2 = model2
        self.classifier_1 = classifie1
        self.classifier_2= classifie2
        self.classifier_3=classifie3
        self.cls = CrossEntropyLabel(4)
        self.id_loss= CrossEntropyLabelSmooth(numclss)
        self.align_loss = nn.MSELoss(reduction='mean')






    def forward(self, x1, x2,id1,id2):

        B1 = x1.shape[0]
        B2 = x2.shape[0]

        feature1 = x1[:, 1:, :]
        feature2 = x2[:, 1:, :]

        part_lenth = feature2.size(1) // 4

        part_tokens1 = x1[:, 0:1]  # stole cls_tokens impl from Phil Wang, thanks
        part_tokens2 = x2[:, 0:1]


        rgb_part_feature=[]
        sketch_part_feature=[]
        for i in range(4):
            local_feat1 = feature1[:, part_lenth*i:part_lenth*(i+1)]
            local_feat1=torch.cat((part_tokens1, local_feat1), dim=1)
            local_feat2 = feature2[:, part_lenth * i:part_lenth * (i + 1)]
            local_feat2 = torch.cat((part_tokens2, local_feat2), dim=1)
            local_feat1 = self.model1.layer2(local_feat1)
            local_feat1=local_feat1[:, 0]
            rgb_part_feature.append(local_feat1)

            local_feat2 = self.model2.layer2(local_feat2)
            local_feat2 = local_feat2[:, 0]
            sketch_part_feature.append(local_feat2)

        rgb_part_feature_bn=[]
        sketch_part_feature_bn=[]
        rgb_cls_score = []
        sketch_cls_score = []
        pids = [0,1,2,3]
        pids = torch.tensor(pids, dtype=torch.int64)
        rgb_cls=[]
        sketch_cls=[]
        align_loss=[]
        for i in range(4):
            rgb_bn_feature= self.model1.bn[i](rgb_part_feature[i])
            rgb_part_feature_bn.append(rgb_bn_feature)
            sketch_bn_feature= self.model2.bn[i](sketch_part_feature[i])
            sketch_part_feature_bn.append(sketch_bn_feature)

            rgb_cls_score_1 = self.classifier_1(rgb_part_feature_bn[i])
            rgb_cls_score.append(rgb_cls_score_1)
            sketch_cls_score_1 = self.classifier_3(sketch_part_feature_bn[i])
            sketch_cls_score.append(sketch_cls_score_1)

            rgb_cls1 = self.cls(rgb_cls_score[i], pids[i].expand(B1))
            rgb_cls.append(rgb_cls1)
            sketch_cls1 = self.cls(sketch_cls_score[i], pids[i].expand(B2))
            sketch_cls.append(sketch_cls1)
            align_loss1 = self.align_loss(rgb_part_feature[i], sketch_part_feature[i])
            align_loss.append(align_loss1)

        rgb_cls=sum(rgb_cls) / 4.0
        sketch_cls=sum(sketch_cls) / 4.0
        align_loss=sum(align_loss) / 4.0






        rgb_part_cls=[]
        sketch_part_cls=[]
        for i in range(4):
            rgb_part_score = self.classifier_2[i](rgb_part_feature_bn[i])
            cls1=self.id_loss(rgb_part_score,id1)
            rgb_part_cls.append(cls1)

            sketch_part_score = self.classifier_2[i](sketch_part_feature_bn[i])
            cls2 = self.id_loss(sketch_part_score,id2)
            sketch_part_cls.append(cls2)


        id_loss1=sum(rgb_part_cls)/4.0
        id_loss2 = sum(sketch_part_cls)/4.0

        loss =id_loss1+id_loss2+rgb_cls+sketch_cls+align_loss

        return loss









def cos_dist(x1, x2):
    """
    Args:
      x1: pytorch Variable, with shape [m, d]
      x2: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
      :param x2:
      :param x1:
    """
    m, n = x1.size(0), x2.size(0)
    x1_norm = torch.pow(x1, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    x2_norm = torch.pow(x2, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x1, x2.t())
    dist = xy_intersection/(x1_norm * x2_norm)
    dist = (1. - dist) / 2
    return dist



def euclidean_dist(x1, x2):
    """
    Args:
      x1: pytorch Variable, with shape [m, d]
      x2: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x1.size(0), x2.size(0)
    x22 = torch.pow(x2, 2).sum(1, keepdim=True).expand(m, n)
    x11 = torch.pow(x1, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = x11 + x22
    dist.addmm_(1, -2, x2, x1.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist