import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj
import math

class PatchCrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.scale = dim ** -0.5

    def forward(self, patch_features: torch.Tensor, text_embedding: torch.Tensor):
        # patch_features: [B, N, D], text_embedding: [B, D]
        q = self.norm_q(text_embedding).unsqueeze(1)
        k = self.norm_kv(patch_features)
        v = k
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)  # [B, 1, N]
        attn = attn.transpose(1, 2)  # [B, N, 1]
        return patch_features * attn

class ProgressiveUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # PixelShuffle 2x 上采样，需要输入通道是 out_channels * (2^2) = out_channels * 4
        # 所以，卷积的输出通道是 out_channels * 4
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.norm = nn.BatchNorm2d(out_channels) # PixelShuffle 后通道数变为 out_channels
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x) # 尺寸变大2倍，通道变回 out_channels
        x = self.norm(x)
        x = self.act(x)
        return x

class ResSegmentationHead(nn.Module):
    def __init__(self, 
                 num_layers=4,            # 特征层数
                 feat_dim=768,           # ViT Patch Feature 维度 
                 text_dim=768,            # Text Feature 维度
                 embed_dim=256,           # 中间隐层维度
                 num_classes=1,           # 输出维度
                 img_size=(336, 336), 
                 patch_size=14):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        in_dim = feat_dim + text_dim 
        split_dim = embed_dim // num_layers

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, split_dim, kernel_size=1),
                nn.BatchNorm2d(split_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers)
        ])
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.up_blocks = nn.ModuleList()
        channels = embed_dim

        for _ in range(3):
            self.up_blocks.append(ProgressiveUpsampleBlock(channels, channels // 2))
            channels //= 2
            
        self.final_channels = channels
        self.pred_head = nn.Conv2d(self.final_channels, num_classes, kernel_size=1)
        
        nn.init.constant_(self.pred_head.weight, 0)
        nn.init.constant_(self.pred_head.bias, 0)

    def forward(self, patch_features_lst: list, epoch_text_feature: torch.Tensor):
        """
        patch_features_lst: List of [B, N, D_feat]
        epoch_text_feature: [B, D_text, 2] (Dim 0: Normal, Dim 1: Anomaly)
        """
        B = patch_features_lst[0].shape[0]
        N = patch_features_lst[0].shape[1]
        H_feat = int(math.sqrt(N)) # e.g., 24
        
        last_patch_feat = patch_features_lst[-1] # [B, N, D]
        
        sim_score = torch.matmul(last_patch_feat, epoch_text_feature)
        anomaly_map_prio = (sim_score[:, :, 1] + 1 - sim_score[:, :, 0]) / 2
        anomaly_map_prio = anomaly_map_prio.view(B, 1, H_feat, H_feat)
        
        if len(epoch_text_feature.shape) == 2:
            epoch_text_feature = epoch_text_feature.unsqueeze(0).expand(B, -1, -1)

        # anomaly_text_feat: [B, D_text]
        anomaly_text_feat = epoch_text_feature[:, :, 1] 
        anomaly_text_feat_spatial = anomaly_text_feat.view(B, -1, 1, 1).expand(-1, -1, H_feat, H_feat)
        
        projected_feats = []
        for i, patch_feat in enumerate(patch_features_lst):
            # [B, N, D] -> [B, D, H, H]
            feat_map = patch_feat.permute(0, 2, 1).reshape(B, -1, H_feat, H_feat)  
            cat_feat = torch.cat([feat_map, anomaly_text_feat_spatial], dim=1)
            projected_feats.append(self.projections[i](cat_feat))
            
        x = torch.cat(projected_feats, dim=1) # [B, embed_dim, H, H]
        x = self.fusion_conv(x)
        
        for block in self.up_blocks:
            x = block(x)

        residual_map = self.pred_head(x) 
        base_map_upsampled = F.interpolate(anomaly_map_prio, size=residual_map.shape[-2:], mode='bilinear', align_corners=False)
        
        final_map = base_map_upsampled + residual_map

        return final_map

class AdaptedCLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: list = [6, 12, 18, 24],
        relu: bool = True,
        use_patch_cross_attn: bool = False,
        use_segmentation_head: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.t_w = text_adapt_weight
        self.i_w = image_adapt_weight
        self.levels = levels
        self.use_patch_cross_attn = use_patch_cross_attn
        if use_segmentation_head:
            self.segmentation_head = ResSegmentationHead(num_layers=len(levels))

        layer_adapters = nn.ModuleList(
            [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
        )
        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
        )
        det_proj = SimpleProj(1024, 768, relu)
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": layer_adapters,
                "seg_proj": seg_proj,
                "det_proj": det_proj,
            }
        )
        if self.use_patch_cross_attn:
            self.image_adapter["patch_cross_attn"] = PatchCrossAttention(768)
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )
        self._init_weights_()

    def _init_weights_(self):
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def forward(self, x, text_embedding=None):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        tokens = []
        for i in range(24):
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if i < self.image_adapt_until:
                adapt_out = self.image_adapter["layer_adapters"][i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.i_w * adapt_out + (1 - self.i_w) * x
            if i + 1 in self.levels:
                tokens.append(x[1:, :, :]) # H*W, bs, D

        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        if self.use_patch_cross_attn:
            if text_embedding is None:
                raise ValueError("text_embedding must be provided when use_patch_cross_attn is True")
            # text_embedding: [B, D] or [B, D, *]; reduce if needed
            if text_embedding.dim() == 3:
                text_embedding = text_embedding.mean(dim=-1)
            seg_tokens = [
                self.image_adapter["patch_cross_attn"](t, text_embedding)
                for t in seg_tokens
            ]
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
        det_token = self.image_adapter["det_proj"](tokens[-1])
        det_token = F.normalize(det_token, dim=-1).mean(1)
        return seg_tokens, det_token

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        # x = (
            # x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            # @ self.clipmodel.text_projection
        # )
        return x
