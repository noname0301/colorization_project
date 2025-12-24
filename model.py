import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_large, ConvNeXt_Large_Weights
from arch_utils.transformer_utils import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from arch_utils.positional_encoding import PositionEmbeddingSine
from arch_utils.unet_utils import UnetBlock
from arch_utils.patchgan_utils import PatchDiscriminator
    

class DDColor(nn.Module):
    def __init__(
            self, 
            encoder_name="convnext-l", 
            num_queries=100, 
            num_scales=3, 
            nf=512,
            num_channels=3,
            do_normalize=False
    ):
        super().__init__()

        assert encoder_name == "convnext-t" or encoder_name == "convnext-l"

        self.encoder = ImageEncoder(encoder_name)
        self.encoder.eval()
        self.decoder = DualDecoder(encoder_name, num_queries, num_scales, nf)
        self.final_conv = nn.Conv2d(num_queries, num_channels, 1)

        self.do_normalize = do_normalize
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.normalize(x)

        _, features = self.encoder(x)

        coarse_input = self.decoder(features)

        out = self.final_conv(coarse_input)

        if self.do_normalize:
            out = self.denormalize(out)
        return out



class DualDecoder(nn.Module):
    def __init__(self, encoder_name, num_queries=100, num_scales=3, nf=512):
        super().__init__()
        self.nf = nf

        # For ConvNeXt, the feature channels are typically:
        # stage_0: 96/192 (tiny/large), stage_2: 192/384, stage_4: 384/768, stage_6: 768/1536
        # We'll use hardcoded values for convnext-l
        self.skip_channels = [192, 384, 768, 1536] if encoder_name == "convnext-l" else [96, 192, 384, 768]

        embed_dim = nf // 2
        self.last_shuf = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 16, 1),
            nn.PixelShuffle(4),
            nn.ReLU(True)
        )
        self.color_decoder = MultiScaleColorDecoder(in_channels=[512, 512, 256],
                                                    num_queries=num_queries,
                                                    num_scales=num_scales)

        # Build decoder layers
        in_c = self.skip_channels[-1]  # 1536 for convnext-l
        out_c = self.nf  # 512

        self.unet_block_0 = UnetBlock(in_c, self.skip_channels[-2], out_c)
        self.unet_block_1 = UnetBlock(out_c, self.skip_channels[-3], out_c)
        self.unet_block_2 = UnetBlock(out_c, self.skip_channels[-4], out_c // 2)

    def forward(self, features):
        # Extract features from the encoder output dictionary
        # Features are captured at stages 1, 3, 5, 7 (the Sequential blocks in ConvNeXt)
        skip_connections = [features[f'stage_{i}'] for i in [1, 3, 5, 7]]

        encode_feat = skip_connections[-1]
        out0 = self.unet_block_0(encode_feat, skip_connections[-2])
        out1 = self.unet_block_1(out0, skip_connections[-3])
        out2 = self.unet_block_2(out1, skip_connections[-4])
        out3 = self.last_shuf(out2)

        return self.color_decoder([out0, out1, out2], out3)


class MultiScaleColorDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=9,
        color_embed_dim=256,
        enforce_input_project=True,
        num_scales=3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = dec_layers
        self.num_feature_levels = num_scales  

        # Positional encoding layer
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        
        # Learnable query features and embeddings
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Learnable level embeddings
        self.level_embed = nn.Embedding(num_scales, hidden_dim)

        # Input projection layers
        self.input_proj = nn.ModuleList(
            [self._make_input_proj(in_ch, hidden_dim, enforce_input_project) for in_ch in in_channels]
        )

        # Transformer layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(dec_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                )
            )

        # Layer normalization for the decoder output
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        # Output embedding layer
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    def forward(self, x, img_features):
        assert len(x) == self.num_feature_levels

        src, pos = self._get_src_and_pos(x)

        bs = src[0].shape[1]

        # Prepare query embeddings (QxNxC)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
    
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                attn_mask=None,
                key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, attn_mask=None,
                key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

        decoder_output = self.decoder_norm(output).transpose(0, 1)
        color_embed = self.color_embed(decoder_output)
    
        out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

        return out

    def _make_input_proj(self, in_ch, hidden_dim, enforce):
        if in_ch != hidden_dim or enforce:
            proj = nn.Conv2d(in_ch, hidden_dim, kernel_size=1)
            nn.init.kaiming_uniform_(proj.weight, a=1)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0)
            return proj
        return nn.Sequential()

    def _get_src_and_pos(self, x):
        src, pos = [], []
        for i, feature in enumerate(x):
            pos.append(self.pe_layer(feature).flatten(2).permute(2, 0, 1))  # flatten NxCxHxW to HWxNxC
            src.append((self.input_proj[i](feature).flatten(2) + self.level_embed.weight[i][None, :, None]).permute(2, 0, 1))
        return src, pos

class ImageEncoder(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()

        assert encoder_name == "convnext-t" or encoder_name == "convnext-l"

        if encoder_name == "convnext-t":
            self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        elif encoder_name == "convnext-l":
            self.model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

        self.encoder_name = encoder_name

        self.hooks = []
        self.features = {}

        for i, stage in enumerate(self.model.features):
            if isinstance(stage, nn.Sequential):
                hook = stage.register_forward_hook(
                    self._make_hook(f'stage_{i}')
                )
                self.hooks.append(hook)

    
    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook
    
    def forward(self, x):
        self.features.clear()
        out = self.model(x)
        return out, self.features
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DDColor("convnext-t", num_queries=100, num_scales=3, nf=512, num_channels=2).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    dummy_image = torch.randn(1, 3, 256, 256)

    dummy_image = dummy_image.to(device)

    out = model(dummy_image)
    print(out.shape)

        


        