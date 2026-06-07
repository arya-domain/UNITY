import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from transformers import PretrainedConfig, PreTrainedModel


def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))]

    return torch.stack(grid_list, dim=-1) 

# backbone
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )
    def forward(self, x):
        return self.block(x) + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block=  nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            )
    def forward(self, x):
        return self.block(x)



class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, downscale_factor, chns=[64,128,256,256,256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(SpatialEncoder, self).__init__()
        downscale_factor = downscale_factor // 2
        in_channels = in_channels * downscale_factor**2 
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)

        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i-1], out_chns),
                                         ResBlock(out_chns),
                                         ResBlock(out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)
    
    def new_input_channels(self, in_channels):
        downscale_factor = self.unshuffle.downscale_factor
        in_channels = in_channels * downscale_factor**2 
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        # modify the first encoder layer
        out_chns = self.encoders[0][0].block[2].out_channels
        self.encoders[0] = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
    
    def forward(self, x):
        x = self.unshuffle(x)
        # print(x.shape)
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features

class ScalerBlock(nn.Module):
    def __init__(self, chns=[64,128,256,256,256]):
        super(ScalerBlock, self).__init__()
        self.chns = chns
        
        # adaptive
        self.adaptive = []
        for in_chns in list(chns):
            adaptive_layer = nn.Conv2d(in_chns, in_chns, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        
    def forward(self, x):
        conv_ftr_list = x
        feature_list = []
        for i, conv_ftr in enumerate(list(conv_ftr_list)):
            feature = self.adaptive[i](conv_ftr)
            feature_list.append(feature)

        return tuple(feature_list)


def MorphWrap(feat, offsets, att_maps, sample_k, out_ch, detach = False):
    
    att_maps = torch.repeat_interleave(att_maps, out_ch, 1)
    B,C,H,W = feat.size()
    multi_feat = torch.repeat_interleave(feat, sample_k, 0)
    if detach:
       multi_warp_feat = F.grid_sample(multi_feat, offsets.detach().permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
    else:
       multi_warp_feat = F.grid_sample(multi_feat, offsets.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

    # print(multi_warp_feat.shape, att_maps.shape)
    multi_att_warp_feat = multi_warp_feat.reshape(B,-1,H,W)*att_maps
    att_warp_feat = sum(torch.split(multi_att_warp_feat,out_ch,1))
    return att_warp_feat


class MultiScaleFlowEstimator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_filters=[128,64,32]):
        super(MultiScaleFlowEstimator, self).__init__()
        layers = []
        for i in range(len(num_filters)):
            if i==0:
                layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=num_filters[i], kernel_size=3, stride=1, padding=1))
            else:
                layers.append(torch.nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=kernel_size, stride=1, padding=kernel_size//2))
            layers.append(torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
        layers.append(torch.nn.Conv2d(in_channels=num_filters[-1], out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class Morphable_Attention_Flow_Net(nn.Module):
    def __init__(self, filter,  head_nums=1, attention_heads=8, cross_attention_dim=768, attention_dim=64):
        super(Morphable_Attention_Flow_Net, self).__init__()
        self.Self_MFEs = []
        self.Cross_MFEs = []
        self.Refine_MFEs = []
        self.k = head_nums

        self.out_chs = []
        self.norms_layers = []
        self.cross_layers = []
        for chnl in filter[::-1]:
            # cross-MFE
            Cross_MFE_layer = MultiScaleFlowEstimator(in_channels=2*chnl, out_channels=self.k*3)
            # self-MFE
            Self_MFE_layer = MultiScaleFlowEstimator(in_channels=2*chnl, out_channels=self.k*3,kernel_size=7)

            # refine-MFE
            Refine_MFE_layer = MultiScaleFlowEstimator(in_channels=2*chnl, out_channels=chnl)
            
            cross_attention = Attention(
                                        query_dim=chnl,
                                        heads=attention_heads,
                                        dim_head=attention_dim,
                                        dropout=0.0,
                                        bias=False,
                                        cross_attention_dim=cross_attention_dim,  # 2048 768
                                        upcast_attention=True,
                                        out_bias=True,
                                    )
            
            self.Self_MFEs.append(Self_MFE_layer)
            self.Cross_MFEs.append(Cross_MFE_layer)
            self.Refine_MFEs.append(Refine_MFE_layer)
            self.out_chs.append(chnl)
            self.norms_layers.append(nn.LayerNorm(chnl, elementwise_affine=True, eps=1e-05))
            self.cross_layers.append(cross_attention)

        self.Self_MFEs = nn.ModuleList(self.Self_MFEs)
        self.Cross_MFEs = nn.ModuleList(self.Cross_MFEs)
        self.Refine_MFEs = nn.ModuleList(self.Refine_MFEs)
        self.norms_layers = nn.ModuleList(self.norms_layers)
        self.cross_layers = nn.ModuleList(self.cross_layers)


    def forward(self,
                source_feats,
                reference_feats,
                prompt_embeds=None,
                return_all=True
                ):

        r"""
        Args:
            source_feats: cloth FPN features
            reference_feats: model and pose features
            return_all: bool return all intermediate try-on results in training phase
        """

        #reference branch inputs model img using self-DAFlow
        last_multi_self_offsets = None
        #source branch inputs cloth img using cross-DAFlow
        last_multi_cross_offsets = None

        if return_all:
            results_all = []

        for i in range(len(source_feats)):
            feat_source = source_feats[len(source_feats) - 1 - i]
            feat_ref = reference_feats[len(reference_feats) - 1 - i]
            B,C,H,W = feat_source.size()

            ## Pre-MorphWrap for Pyramid feature
            att_source_feat = feat_source
            att_reference_feat = feat_ref
            
            ## Cross-MFE
            input_feat =  torch.cat([att_source_feat,feat_ref],1)
            offsets_att = self.Cross_MFEs[i](input_feat)
            cross_att_maps = F.softmax(offsets_att[:,self.k*2:,:,:],dim=1)
            offsets = apply_offset(offsets_att[:,:self.k*2,:,:].reshape(-1,2,H,W))
            offsets = offsets.permute(0, 3, 1, 2)
            last_multi_cross_offsets = offsets
            
            # print(feat_source.shape, last_multi_cross_offsets.shape, cross_att_maps.shape, self.k, self.out_chs[i])
            att_source_feat = MorphWrap(feat_source, last_multi_cross_offsets, cross_att_maps, self.k, self.out_chs[i])

            ## Self-MFE
            input_feat =  torch.cat([att_source_feat,att_reference_feat],1)
            offsets_att = self.Self_MFEs[i](input_feat)
            self_att_maps = F.softmax(offsets_att[:,self.k*2:,:,:],dim=1)
            offsets = apply_offset(offsets_att[:,:self.k*2,:,:].reshape(-1,2,H,W))
            if last_multi_self_offsets is not None:
                offsets = F.grid_sample(last_multi_self_offsets, offsets, mode='bilinear', padding_mode='border')
            else:
                offsets = offsets.permute(0, 3, 1, 2)
            last_multi_self_offsets = offsets
            att_reference_feat = MorphWrap(feat_ref, last_multi_self_offsets, self_att_maps , self.k, self.out_chs[i])

            ##Refine-MFE
            input_feat =  torch.cat([att_source_feat,att_reference_feat],1)
            offsets_att = self.Refine_MFEs[i](input_feat)
            
            final_feat = self.norms_layers[i](offsets_att.permute(0,2,3,1)).permute(0,3,1,2)
            if prompt_embeds is not None:
                final_feat = self.cross_layers[i](final_feat, prompt_embeds)

            results_all.append(final_feat)

        # return results_all
        return results_all[::-1]


class UNITY_Config(PretrainedConfig):
    model_type = "UNITY"
    def __init__(self, ref_in_channel=3, source_in_channel=3, head_nums=6, downsample=8, resolution=512, variant="sd15", **kwargs):
        super().__init__(**kwargs)
        self.ref_in_channel = ref_in_channel
        self.source_in_channel = source_in_channel
        self.head_nums = head_nums
        self.downsample = downsample
        self.resolution = resolution
        self.variant = variant
        self.num_filters = [320, 640, 1280, 1280]
        if variant == "sdxl":
            self.scales = {64:32, 8:16}
            self.attention_heads = 12
            self.cross_attention_dim = 2048
            self.attention_dim = 84
        if variant == "sd15":
            self.attention_heads = 8
            self.cross_attention_dim = 768
            self.attention_dim = 64

class UNITY(PreTrainedModel):
    config_class = UNITY_Config
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_filters = config.num_filters
        self.source_encoder = SpatialEncoder(config.source_in_channel, config.downsample, self.num_filters)
        self.source_SB = ScalerBlock(self.num_filters)
        
        self.reference_encoder = SpatialEncoder(config.ref_in_channel, config.downsample, self.num_filters)
        self.reference_SB = ScalerBlock(self.num_filters)

        self.MAFNet = Morphable_Attention_Flow_Net(self.num_filters, head_nums=config.head_nums, 
                                                   attention_heads=config.attention_heads, cross_attention_dim=config.cross_attention_dim,
                                                   attention_dim=config.attention_dim
                                                   )
        
        self.post_init()
        self.resolution = (config.resolution, config.resolution)
        print(self.resolution)
        if config.variant == "sdxl":
            keys  = list(config.scales.keys())
            self.linear_proj1 = nn.Linear(keys[0]**2, config.scales[keys[0]]**2)
            self.linear_proj2 = nn.Linear(keys[1]**2, config.scales[keys[1]]**2)

    def forward(self, conditional_images, prompt_embeds=None, return_all=True):
        # print(conditional_images, conditional_images.shape)
        conditional_images = F.interpolate(conditional_images, size=self.resolution, mode='bilinear', align_corners=False).to(dtype=self.dtype)
        source_feats = self.source_SB(self.source_encoder(conditional_images)) 
        reference_feats = self.reference_SB(self.reference_encoder(conditional_images))
        result = self.MAFNet(
                             source_feats, reference_feats, prompt_embeds=prompt_embeds,
                             return_all=return_all
                            )
        if hasattr(self, 'linear_proj1'):
            shape = result[0].shape
            keys  = list(self.config.scales.keys())
            result[0] = self.linear_proj1(result[0].view(shape[0], shape[1], -1))
            result[0] = result[0].view(shape[0], shape[1], self.config.scales[keys[0]], self.config.scales[keys[0]])
            
            shape = result[-1].shape
            result[-1] = self.linear_proj2(result[-1].view(shape[0], shape[1], -1))
            result[-1] = result[-1].view(shape[0], shape[1], self.config.scales[keys[1]], self.config.scales[keys[1]])

        return result
    
    
    def set_input_channels(self, ref_in_channel, source_in_channel):
        self.source_encoder.new_input_channels(source_in_channel)
        self.reference_encoder.new_input_channels(ref_in_channel)
