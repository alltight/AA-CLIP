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
    """
    渐进式上采样块，使用 Conv + PixelShuffle (x2上采样)。
    确保输入通道和输出通道正确对应 PixelShuffle 的要求。
    """
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

class ViTLSegmentationHead(nn.Module):
    def __init__(self, 
                 num_layers=4,           # 你提供的掩码层数
                 feat_dim=1,          # ViT-L 的特征维度 (如果直接用 ViT 特征)
                 embed_dim=256,          # 融合后的中间特征维度
                 num_classes=2,          # 最终分割类别数
                 img_size=(336, 336),    # ViT-L-14-336px 的输入图像大小
                 patch_size=14):         # ViT 的 patch size
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        # 计算特征图的原始分辨率
        self.feat_res = img_size[0] // patch_size 

        # 1. 特征投影与融合
        # 将所有输入的通道数统一到 (embed_dim // num_layers) 
        # 拼接后总通道数为 embed_dim
        split_dim = embed_dim // num_layers
        
        self.projections = nn.ModuleList([
            nn.Conv2d(feat_dim, split_dim, kernel_size=1) 
            for _ in range(num_layers)
        ])
        
        # 融合后的平滑层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(split_dim * num_layers, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 渐进式上采样路径
        # 需要将 (img_size // patch_size) 的特征图上采样到 img_size。
        # 上采样倍数是 patch_size。
        # 例如，336 / 14 = 24，需要从 24x24 上采样到 336x336，即 14 倍。
        # 14 不是 2 的整数幂，我们需要进行一个特殊的处理。
        
        # 先进行多次 x2 上采样，尽可能接近目标
        # 24 -> 48 -> 96 -> 192 -> 384 (5次x2 上采样)
        # 4次 x2 上采样是 16倍，可以从 24 -> 384。
        # 16 = 2^4，所以需要 4 个 x2 上采样块。

        self.up_blocks = nn.ModuleList()
        current_channels = embed_dim
        # 第一次上采样：从 1/14 (24x24) -> 1/7 (48x48)
        self.up_blocks.append(ProgressiveUpsampleBlock(current_channels, embed_dim // 2))
        current_channels = embed_dim // 2
        
        # 第二次上采样：1/7 (48x48) -> 2/7 (96x96)
        self.up_blocks.append(ProgressiveUpsampleBlock(current_channels, embed_dim // 4))
        current_channels = embed_dim // 4
        
        # 第三次上采样：2/7 (96x96) -> 4/7 (192x192)
        self.up_blocks.append(ProgressiveUpsampleBlock(current_channels, embed_dim // 8))
        current_channels = embed_dim // 8
        
        # 第四次上采样：4/7 (192x192) -> 8/7 (384x384)
        # 注意：这里会稍微超过目标尺寸 336x336
        self.up_blocks.append(ProgressiveUpsampleBlock(current_channels, embed_dim // 8))
        current_channels = embed_dim // 8

        # 3. 最终预测头
        self.pred_head = nn.Conv2d(current_channels, num_classes, kernel_size=1)

    def forward(self, patch_features_lst: torch.Tensor, epoch_text_feature: torch.Tensor):
        """
        Args:
            
        """

        inputs = []
        # B, _, C = epoch_text_feature.shape
        B, N, D = patch_features_lst[0].shape
        H = int(math.sqrt(N))
        if len(epoch_text_feature.shape) == 2:
            epoch_text_feature = epoch_text_feature.unsqueeze(0).expand(B, -1, -1)
        _, _, C = epoch_text_feature.shape
        assert N == H * H, f"N = {N}, H = {H}"
        # print(f"patch_feature: {patch_features_lst[0].shape}")
        # print(f"epoch_text_feature: {epoch_text_feature.shape}")
        text_feature = epoch_text_feature.transpose(1, 2).reshape(B, -1).unsqueeze(1).expand(B, N, -1)
        for patch_features in patch_features_lst:
            sim_score = torch.matmul(patch_features, epoch_text_feature)
            sim_score = sim_score.permute(0, 2, 1).view(B, C, H, H)
            sim_score = (sim_score[:, 1] + 1 - sim_score[:, 0]) / 2
            # patch_features = torch.concat([patch_features, text_feature], dim=-1).transpose(1, 2).reshape(B, -1, H, H)
            inputs.append(sim_score.unsqueeze(1))

        # for m in masks:
            # if m.dim() == 3: # 如果是 (B, H_feat, W_feat)
                # m = m.unsqueeze(1) # 变成 (B, 1, H_feat, W_feat)
            # inputs.append(m)
            
        # 1. 投影并拼接
        projs = [layer(x) for layer, x in zip(self.projections, inputs)]
        fused = torch.cat(projs, dim=1) 
        x = self.fusion_conv(fused)

        # 2. 渐进式上采样
        for up_block in self.up_blocks:
            x = up_block(x)

        # 3. 预测
        out = self.pred_head(x)
        
        # 最终的上采样/裁剪，确保完全匹配目标尺寸 336x336
        # if out.shape[-2:] != self.img_size:
            # out = F.interpolate(out, size=self.img_size, mode='bilinear', align_corners=False)
        return out
        # return F.sigmoid(out)

class ResSegmentationHead(nn.Module):
    def __init__(self, 
                 num_layers=4,            # 特征层数
                 feat_dim=768,           # ViT Patch Feature 维度 (例如 ViT-L 为 1024)
                 text_dim=768,            # Text Feature 维度 (通常也是 ViT 维度，如 768 或 1024)
                 embed_dim=256,           # 中间隐层维度
                 num_classes=1,           # 输出维度：这里我们需要输出一个修正值 delta，通常是 1 通道
                 img_size=(336, 336), 
                 patch_size=14):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        # -------------------------------------------------------
        # 1. 特征融合分支 (Refinement Branch)
        # -------------------------------------------------------
        
        # 计算融合后的输入维度：Patch特征 + Text特征 (Anomaly部分的文本特征)
        # 我们将把 Anomaly Text Feature 拼接到每个 Patch 上
        in_dim = feat_dim + text_dim 
        split_dim = embed_dim // num_layers

        # 投影层：将 (Patch + Text) 的高维特征降维
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, split_dim, kernel_size=1),
                nn.BatchNorm2d(split_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers)
        ])
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # 渐进式上采样 (目标是从 24x24 -> 192x192 -> Resize to 336)
        self.up_blocks = nn.ModuleList()
        channels = embed_dim
        
        # 3次上采样 (x8 倍)
        for _ in range(3):
            self.up_blocks.append(ProgressiveUpsampleBlock(channels, channels // 2))
            channels //= 2
            
        self.final_channels = channels

        # -------------------------------------------------------
        # 2. 预测头 (关键修改：零初始化)
        # -------------------------------------------------------
        self.pred_head = nn.Conv2d(self.final_channels, num_classes, kernel_size=1)
        
        # 【关键点】对最后一层进行零初始化
        # 这保证了初始状态下，Res_Branch 的输出为 0
        # 整个网络退化为单纯的 Sim_Score 插值，保住 0.9 的底线
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
        
        # -------------------------------------------------------
        # Part A: 计算 Base Map (你的 Sim Score) - 强先验
        # -------------------------------------------------------
        # 取出 Normal 和 Anomaly 的文本特征
        # text_feat: [B, D_text, 2]
        
        # 我们用最后一层特征来计算最准确的 sim_score (或者你可以平均所有层)
        last_patch_feat = patch_features_lst[-1] # [B, N, D]
        
        # 计算相似度 [B, N, 2]
        sim_score = torch.matmul(last_patch_feat, epoch_text_feature)
        
        # 归一化处理 (你的逻辑) -> [B, N]
        # (Anomaly + 1 - Normal) / 2
        anomaly_map_prio = (sim_score[:, :, 1] + 1 - sim_score[:, :, 0]) / 2
        
        # Reshape to [B, 1, H, H]
        anomaly_map_prio = anomaly_map_prio.view(B, 1, H_feat, H_feat)

        # -------------------------------------------------------
        # Part B: 计算 Residual Map (特征 + 文本 -> 修正量)
        # -------------------------------------------------------
        
        if len(epoch_text_feature.shape) == 2:
            epoch_text_feature = epoch_text_feature.unsqueeze(0).expand(B, -1, -1)
        # 准备 Text Feature 用于拼接：这里我们只取 Anomaly 的文本特征作为引导
        # anomaly_text_feat: [B, D_text]
        anomaly_text_feat = epoch_text_feature[:, :, 1] 
        # 扩展到 spatial 维度: [B, D_text, H, H]
        anomaly_text_feat_spatial = anomaly_text_feat.view(B, -1, 1, 1).expand(-1, -1, H_feat, H_feat)
        
        projected_feats = []
        for i, patch_feat in enumerate(patch_features_lst):
            # [B, N, D] -> [B, D, H, H]
            feat_map = patch_feat.permute(0, 2, 1).reshape(B, -1, H_feat, H_feat)
            
            # 拼接 Patch特征 和 Text特征
            # cat_feat: [B, D_feat + D_text, H, H]
            cat_feat = torch.cat([feat_map, anomaly_text_feat_spatial], dim=1)
            
            # 投影
            projected_feats.append(self.projections[i](cat_feat))
            
        # 融合与上采样
        x = torch.cat(projected_feats, dim=1) # [B, embed_dim, H, H]
        x = self.fusion_conv(x)
        
        for block in self.up_blocks:
            x = block(x)
            
        # 预测残差 [B, 1, 192, 192]
        # 由于是零初始化，初始这里全是 0
        residual_map = self.pred_head(x) 

        # -------------------------------------------------------
        # Part C: 组合与对齐
        # -------------------------------------------------------
        
        # 1. 将 Base Map 上采样到与 Residual Map 相同的尺寸 (192x192)
        base_map_upsampled = F.interpolate(anomaly_map_prio, size=residual_map.shape[-2:], mode='bilinear', align_corners=False)
        
        # 2. 核心公式: Output = Base + Residual
        # 此时网络学习的是“如何微调 Base Map 以更贴近 GT”
        final_map = base_map_upsampled + residual_map
        
        # 3. 最终对齐到目标尺寸 (336x336)
        # if final_map.shape[-2:] != self.img_size:
            # final_map = F.interpolate(final_map, size=self.img_size, mode='bilinear', align_corners=False)
            
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
