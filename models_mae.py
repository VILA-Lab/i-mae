# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# Un-Mix: https://github.com/szq0214/Un-Mix
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio,ids_shuffle,ids_restore):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if type(ids_shuffle) != type(None):
            ids_shuffle=ids_shuffle
            ids_restore = ids_restore
        else:
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_shuffle, ids_restore

    def forward_encoder(self, x, mask_ratio,ids_shuffle, ids_restore):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_shuffle, ids_restore = self.random_masking(x, mask_ratio,ids_shuffle,ids_restore)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_shuffle,ids_restore

    def forward_no_norm(self, x, mask_ratio,ids_shuffle, ids_restore):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_shuffle, ids_restore = self.random_masking(x, mask_ratio,ids_shuffle,ids_restore)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        return x,mask, ids_shuffle,ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x1 = self.decoder_embed(x)
        latent1=x1

        def dec_func(x):
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            # add pos embed
            x = x + self.decoder_pos_embed

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # predictor projection
            x = self.decoder_pred(x)

            # remove cls token
            x = x[:, 1:, :]
            return x
        x1 = dec_func(x1)
        return (x1,) , (latent1,)

    def forward_loss(self, imgs, preds, mask, weak_idx,teach_latent,latents):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        """
        if weak_idx == 0:
            target = self.patchify(imgs[1])
            loss = (preds[0] - target) ** 2

        elif weak_idx ==1:
            target = self.patchify(imgs[2])
            loss = (preds[0] - target) ** 2

        else:
            raise NotImplementedError
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        if type(teach_latent) != type(None):
            if weak_idx == 0:
                loss += torch.nn.functional.mse_loss(latents[0] ,teach_latent[0])
            if weak_idx == 1:
                loss += torch.nn.functional.mse_loss(latents[0] ,teach_latent[1])
        return loss

    def forward(self, imgs, weak_idx, mask_ratio=0.75,ids_shuffle=None,ids_restore=None, teach_latent=None):
        #samples = (mixed_images, im1,im2)
        mixed_img = imgs[0]
        latent, mask, ids_shuffle, ids_restore = self.forward_encoder(mixed_img, mask_ratio,ids_shuffle,ids_restore)
        
        preds,latents = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, preds, mask, weak_idx,teach_latent,latents)
        return loss, preds, mask



class MAEDouble(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans,
                         embed_dim, depth, num_heads,
                 decoder_embed_dim, decoder_depth, decoder_num_heads,
                 mlp_ratio, norm_layer, norm_pix_loss)

        # --------------------------------------------------------------------------
        # i-MAE decoder specifics
        self.decoder_embed1 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed2 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x1 = self.decoder_embed1(x)
        x2 = self.decoder_embed2(x)

        def dec_func(x):
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            # add pos embed
            x = x + self.decoder_pos_embed

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # predictor projection
            x = self.decoder_pred(x)

            # remove cls token
            x = x[:, 1:, :]
            return x
        x1_decoded = dec_func(x1.clone())
        x2_decoded = dec_func(x2.clone())
        return (x1_decoded,x2_decoded), (x1,x2)

    def forward_loss(self, imgs, preds, mask, weak_idx,teach_latent,latents):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        """
        if weak_idx == 0:
            target = self.patchify(imgs[1])
            loss = (preds[0] - target) ** 2

            target = self.patchify(imgs[2])
            loss += (preds[1] - target) ** 2

        else: # == 1
            target = self.patchify(imgs[2])
            loss = (preds[0] - target) ** 2

            target = self.patchify(imgs[1])
            loss += (preds[1] - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        if type(teach_latent) != type(None):
            if weak_idx == 0: #distill branch 0 to latent 0
                loss += torch.nn.functional.mse_loss(latents[0] ,teach_latent[0])
                loss += torch.nn.functional.mse_loss(latents[1] ,teach_latent[1])
            elif weak_idx == 1: #distill branch 0 to latent 1
                loss += torch.nn.functional.mse_loss(latents[0] ,teach_latent[1])
                loss += torch.nn.functional.mse_loss(latents[1] ,teach_latent[0])
        return loss

    def forward(self, imgs, weak_idx, mask_ratio=0.75,ids_shuffle=None,ids_restore=None, teach_latent=None):
        #samples = (mixed_images, im1,im2)
        mixed_img = imgs[0]
        latent, mask, ids_shuffle, ids_restore = self.forward_encoder(mixed_img, mask_ratio,ids_shuffle,ids_restore)
        
        preds,latents = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, preds, mask, weak_idx,teach_latent,latents)
        return loss, preds, mask


class MAE_Teacher(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans,
                         embed_dim, depth, num_heads,
                 decoder_embed_dim, decoder_depth, decoder_num_heads,
                 mlp_ratio, norm_layer, norm_pix_loss)

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, ids_shuffle=None, ids_restore=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MAEDouble(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def tmae_vit_large_patch16_dec512d8b(**kwargs):
    model = MAE_Teacher(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def tmae_vit_base_patch16_dec512d8b(**kwargs):
    model = MAE_Teacher(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_small_patch8_dec1928b(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=64,
        patch_size=8, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=6, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

teacher_mae_vit_base_patch16 = tmae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

teacher_mae_vit_large_patch16 = tmae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

mae_vit_small_patch8 = mae_vit_small_patch8_dec1928b
