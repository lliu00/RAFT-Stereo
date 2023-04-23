# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
# import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine


from core.corr import CorrBlock1D
from core.utils.utils import coords_grid, upflow8
from core.update import BasicMotionEncoder

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        assert not torch.isnan(tgt).any()
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        assert not torch.isnan(tgt).any()
        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class MultiScaleMaskedTransformerDecoder(nn.Module):

   

    def __init__(
        self,
        args,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 1
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                # weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        
        self.bins_embed = nn.Linear(hidden_dim, 1)
        
        self.corr_radius = args.corr_radius

        self.encoder = BasicMotionEncoder(args)
        
        self.convhx = nn.Conv2d(hidden_dim+in_channels, hidden_dim, 3, padding=1)
        
    def forward(self, fmap1, fmap2, mask_features, test_mode, mask = None):
        # fmap1, fmap2 is feature map of left, right img of multi-scale feature,respectively
        
        ############ initial corr_lookup op
        corr_block = CorrBlock1D
        corr_fn = corr_block(fmap1, fmap2, radius=7, num_levels=4)
        ############ initial flow 
        coords0, coords1 = self.initialize_flow(mask_features[0])

        # disable mask, it does not affect performance
        del mask

        bs, _, _, _= fmap1.shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

    
        # # prediction heads on learnable query features
        # outputs_mask = self.forward_prediction_heads(output, mask_features)

        # predictions_mask.append(outputs_mask)
        
        flow_predictions = []
        for i in range(self.num_layers):
            ######���Ż�
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  #�����Ӳ���Ҷ�Ӧ�ĵ�??ȡcorr
            flow = coords1 - coords0
            motion_features = self.encoder(flow,corr)
            corr = torch.cat([motion_features, torch.relu(mask_features[0])], dim = 1)
            corr = self.convhx(corr)
            x = [corr]
            src = []
            pos = []
            size_list = []
            for j in range(self.num_feature_levels):
                size_list.append(x[j].shape[-2:])
                pos.append(self.pe_layer(x[j], None).flatten(2))
                src.append(self.input_proj[j](x[j]).flatten(2) + self.level_embed.weight[j][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)
            #####################################
            level_index = 0
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_mask = self.forward_prediction_heads(output, mask_features[0])
            
            ######2023.4.17 ll
            bins = self.bins_embed(output).squeeze(2).permute(1,0)
            
            bins = torch.softmax(bins, dim=1)
            bins = bins / bins.sum(dim=1, keepdim=True)

            bin_widths = 100 * bins  # .shape = N, dim_out
            bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=1e-4)
            bin_edges = torch.cumsum(bin_widths, dim=1)
            centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
            n, dout = centers.size()
            centers = centers.contiguous().view(n, dout, 1, 1)
            outputs_mask = outputs_mask.softmax(dim=1)
            pred_delta_flow = -1 * torch.sum(outputs_mask * centers, dim=1, keepdim=True)
            
            # in stereo mode, project flow onto epipolar
            # F(t+1) = F(t) + \Delta(t)
            coords1[:,0] = coords1[:,0] + pred_delta_flow[:,0]

            # # We do not need to upsample or output intermediate results in test_mode
            if test_mode and i < self.num_layers-1:
                continue
            
            up_mask = None
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)
            ##################
        if test_mode:
            return coords1 - coords0, flow_up  
        return flow_predictions

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return outputs_mask

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1
    
    