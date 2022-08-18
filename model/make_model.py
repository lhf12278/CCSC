import torch
import torch.nn as nn
from model.new_model import Model_interra, Align
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID
import copy



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Bn(nn.Module):

    def __init__(self, inplanes):
        super().__init__()

        self.bottleneck = nn.BatchNorm1d(inplanes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):

        x = self.bottleneck(x)
        return x

class Classlayer(nn.Module):

    def __init__(self,num_classes):
        super().__init__()

        self.classifier_ = nn.Linear(768, num_classes, bias=False)
        self.classifier_.apply(weights_init_classifier)

    def forward(self, x):

        x = self.classifier_(x)
        return x



class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))


        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)


        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        part_norm = self.base.norm

        self.layer1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(part_norm)
        )

        self.layer2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(part_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.rgb_part_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.sketch_part_token = nn.Parameter(torch.zeros(1, 1, 768))

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bn = nn.ModuleList([
         Bn(inplanes=self.in_planes) for i in range(4)])





__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,

}


def make_model(cfg, num_class):
    if cfg.MODEL.NAME == 'transformer':
        model = TrainModel(num_class,  cfg, __factory_T_type)
        print('===========building transformer===========')

    return model


class TrainModel(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super(TrainModel, self).__init__()
        self.model1 = build_transformer(num_classes,  cfg,factory)
        self.model2 = build_transformer(num_classes,  cfg,factory)
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.Model_interra = Model_interra(self.model1, self.model2)


        self.classifier_1 = nn.Linear(768, 4, bias=False)
        self.classifier_1.apply(weights_init_classifier)

        self.classifier_3 = nn.Linear(768, 4, bias=False)
        self.classifier_3.apply(weights_init_classifier)


        self.classlayer = nn.ModuleList([
            Classlayer(num_classes) for i in range(4)])

        self.align1 = Align(self.model1, self.model2, self.classifier_1, self.classifier_3, self.classlayer,num_classes)
        self.align2 = Align(self.model1, self.model2, self.classifier_3, self.classifier_1, self.classlayer,num_classes)


    def forward(self, x1, x2,id1=None,id2=None, modal=0,epoch=None):
        B1 = x1.shape[0]
        B2 = x2.shape[0]

        x1 = self.model1.base.patch_embed(x1)
        x2 = self.model2.base.patch_embed(x2)

        cls_tokens1 = self.model1.base.cls_token.expand(B1, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens2 = self.model2.base.cls_token.expand(B2, -1, -1)
        x1 = torch.cat((cls_tokens1, x1), dim=1)
        x2 = torch.cat((cls_tokens2, x2), dim=1)
        x1 = x1 + self.model1.base.pos_embed
        x2 = x2 + self.model2.base.pos_embed
        x1 = self.model1.base.pos_drop(x1)
        x2 = self.model2.base.pos_drop(x2)

        if modal == 3:
            x11 = x1
            x22 = x2

            for blk in self.model1.base.blocks[:-1]:
                x11 = blk(x11)

            for blk in self.model2.base.blocks[:-1]:
                x22 = blk(x22)

            x111=x11
            x222 =x22


            x11=self.model1.layer1(x111)
            x22=self.model2.layer1(x222)


            if epoch >50:
                self.classifier_1.weight.requires_grad = True
                self.classifier_3.weight.requires_grad = True
                align_loss1 = self.align1(x111, x222, id1, id2)

                self.classifier_1.weight.requires_grad = False
                self.classifier_3.weight.requires_grad = False
                align_loss2 = self.align2(x111, x222, id1, id2)

                align_loss = align_loss1+align_loss2

            global_feat3 = x11[:, 0]
            global_feat4 = x22[:, 0]

            feat3 = self.model1.bottleneck(global_feat3)
            feat4 = self.model2.bottleneck(global_feat4)


            self.model1.classifier.weight.requires_grad = True
            self.model2.classifier.weight.requires_grad = True

            cls_score3 = self.model1.classifier(feat3)
            cls_score4 = self.model2.classifier(feat4)

            if epoch >50:
                return cls_score3, global_feat3, cls_score4, global_feat4, align_loss
            else:
                return cls_score3, global_feat3, cls_score4, global_feat4




        if modal == 4:
            x1, x2 = self.Model_interra(x1, x2)


            x1 = self.model1.base.norm(x1)
            x2 = self.model2.base.norm(x2)

            global_feat1 = x1[:, 0]
            global_feat2 = x2[:, 0]


            feat1 = self.model1.bottleneck(global_feat1)
            feat2 = self.model2.bottleneck(global_feat2)

            self.model1.classifier.weight.requires_grad = False
            self.model2.classifier.weight.requires_grad = False

            cls_score1 = self.model1.classifier(feat1)
            cls_score2 = self.model2.classifier(feat2)

            return cls_score1, global_feat1, cls_score2, global_feat2

        elif modal == 1:
            for blk in self.model1.base.blocks[:-1]:
                x1 = blk(x1)
            x11 = x1
            x12=self.model1.layer1(x11)
            global_feat1 = x12[:,0]

            feature1 = x11[:, 1:, :]
            part_lenth = feature1.size(1) // 4
            part_tokens1 = x11[:, 0:1]

            rgb_part_feature = []
            for i in range(4):
                local_feat1 = feature1[:, part_lenth * i:part_lenth * (i + 1)]
                local_feat1 = torch.cat((part_tokens1, local_feat1), dim=1)
                local_feat1 = self.model1.layer2(local_feat1)
                local_feat1 = local_feat1[:, 0]
                rgb_part_feature.append(local_feat1)
            rgb_globe_feat = torch.cat(rgb_part_feature, dim=1)
            rgb_globe_feat = torch.cat([global_feat1, rgb_globe_feat], dim=1)
            global_feat = rgb_globe_feat

            return global_feat


        elif modal == 2:
            for blk in self.model2.base.blocks[:-1]:
                x2 = blk(x2)
            x2_11 = x2
            x2_12=self.model2.layer1(x2_11)
            global_feat2 = x2_12[:, 0]

            feature2 = x2_11[:, 1:, :]
            part_lenth = feature2.size(1) // 4
            part_tokens2 = x2_11[:, 0:1]

            sketch_part_feature = []
            for i in range(4):
                local_feat2 = feature2[:, part_lenth * i:part_lenth * (i + 1)]
                local_feat2 = torch.cat((part_tokens2, local_feat2), dim=1)

                local_feat2 = self.model2.layer2(local_feat2)
                local_feat2 = local_feat2[:, 0]

                sketch_part_feature.append(local_feat2)
            sketch_globe_feat = torch.cat(sketch_part_feature, dim=1)
            sketch_globe_feat = torch.cat([global_feat2, sketch_globe_feat], dim=1)
            global_feat = sketch_globe_feat

            return global_feat





    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))