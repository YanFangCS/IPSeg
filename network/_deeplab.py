import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.head = nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Conv2d(304, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, c, 1)
                ) for c in num_classes]
        )
        
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        
        output_feature = torch.cat( [ low_level_feature, output_feature ], dim=1 )
        
        heads = [h(output_feature) for h in self.head]
        heads = torch.cat(heads, dim=1)
        
        return heads
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)
                #nn.init.normal_(m.weight, mean=5, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()
        new_num = num_classes[:2] + [x + 2 for x in num_classes[2:]]
        new_num[1] += new_num[0]
        new_num = new_num[1:]
        # input(new_num) # [1,1,10,1,1]-->[2,12,3,3]
        self.aspp = ASPP(in_channels, aspp_dilate)
        
        self.mining_head1 = nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Conv2d(512+256,256,3,padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Upsample(size=(65, 65), mode='bilinear', align_corners=False)
                ) for c in new_num]
        )

        self.mining_head2 = nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Conv2d(256+256,256,3,padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Upsample(size=(129, 129), mode='bilinear', align_corners=False)
                ) for c in new_num]
        )

        self.mining_head3 = nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Conv2d(128+256,256,3,padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, c, 1)
                ) for c in new_num]
        )

        self.adaptive_pool=nn.AdaptiveAvgPool2d((2,2))
        self.fc=nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.scale_head=nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Linear(4096,c)
                ) for c in num_classes[2:]]
        )
        self._init_weight()

    def forward(self, feature):
        scale_heads = self.adaptive_pool(feature[3])
        scale_heads = scale_heads.view(scale_heads.size(0), -1)
        scale_heads = self.fc(scale_heads)
        s_heads = [h(scale_heads) for h in self.scale_head]
        s_heads = torch.cat(s_heads, dim=1)

        feature3 = self.aspp(feature[3]) 
        H = feature3.shape[-1]*2-1
        feature3 = F.interpolate(feature3, size=(H,H), mode='bilinear', align_corners=False) 
        feature3 = torch.cat((feature3, feature[2]), dim=1) 
        feature2 =[h(feature3) for h in self.mining_head1] 

        feature2 = [torch.cat((t, feature[1]), dim=1) for t in feature2] 

        feature1 =[]
        feature1 = [h(feature2[i]) for i, h in enumerate(self.mining_head2)] 

        feature1 = [torch.cat((t, feature[0]), dim=1) for t in feature1] 

        heads =[]
        heads = [h(feature1[i]) for i, h in enumerate(self.mining_head3)] 


        # # mask
        # for i in range(len(heads)):
        #     if i>0:
        #         max_res = heads[i].detach().max(dim=1)[1] 
        #         max_res_expanded = max_res.unsqueeze(1).expand_as(heads[i][:,2:,:,:])
        #         heads[i][:,2:,:,:] = torch.where(max_res_expanded != 0, heads[i][:,2:,:,:], torch.full_like(heads[i][:,2:,:,:], -5))


        # # soft mask
        # for i in range(len(heads)):
        #     if i>0:
        #         max_res = heads[i].detach().max(dim=1)[1] # B,129,129
        #         a=3
        #         max_res[max_res>0]=a
        #         max_res = max_res-a
        #         max_res_expanded = max_res.unsqueeze(1).expand_as(heads[i][:,2:,:,:])
        #         heads[i][:,2:,:,:] = heads[i][:,2:,:,:] + max_res_expanded

        cls=[]
        for t in heads:
            if t.shape[1]>2:
                cls.append(t[:,2:,:,:]) 
        global_output = [heads[0]] + cls
        curr_output = [heads[-1][:,:2,:,:]] + cls
        global_output = torch.cat(global_output, dim=1)
        curr_output = torch.cat(curr_output, dim=1)
        return global_output, curr_output, s_heads


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module