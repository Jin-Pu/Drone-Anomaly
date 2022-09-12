import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):

        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        return out


class Decoder(nn.Module):
    def __init__(self, z_dim=512):
        super(Decoder, self).__init__()
        self.z_dim = z_dim

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample2(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=4, stride=4, padding=1,
                                         output_padding=2),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.decoder = nn.Sequential(
            # Basic(2048, 1024),
            # Upsample(1024, 1024),
            # Basic(1024, 512),
            # Upsample(512, 512),
            # Basic(512, 256),
            # Upsample(256, 256),
            Basic(256, 128),
            Upsample(128, 128),
            Basic(128, 64),
            Upsample(64, 64),
            Upsample2(64, 64),
            Gen(64, 3, 32),
        )

        self.de_dense = nn.Sequential(
            nn.Linear(768, 256 * 16 * 16),
            nn.ELU(),
        )

    def forward(self, x):
        # output = self.de_dense(x)

        output = x.view(x.size(0), 256, 16, 16)
        output = self.decoder(output)

        return output


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self,
                 image_size=(256, 256),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 num_frames=4,
                 nu=0.01,
                 feat_dim=None):
        super(VisionTransformer, self).__init__()
        h, w = image_size

        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw * num_frames
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        #Decoder
        self.decoder = Decoder()

        #Maxpooling

        self.maxpooling = nn.MaxPool1d(num_frames, stride=num_frames)

        #Sphere
        self.nu = nu
        self.radius = torch.nn.Parameter(torch.tensor([0.0]))
        self.center = torch.nn.Parameter(torch.randn(emb_dim))

        # classfier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x): # x: (b, n, c, h, w)
        x = x.float().cuda()
        for i in range(x.shape[1]):
            emb = self.embedding(x[:, i])
            emb = emb.permute(0, 2, 3, 1)
            b, h, w, c = emb.shape
            emb = emb.reshape(b, h * w, c)
            if i == 0:
                all_emb = emb
            else:
                all_emb = torch.cat((emb, all_emb), 1)  # (b, h*w*n, c)
        # emb = self.embedding(x)     # (n, c, gh, gw)
        # emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        # b, h, w, c = emb.shape
        # emb = emb.reshape(b, h * w, c)

        b, n, c = all_emb.shape

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, all_emb], dim=1)

        # transformer
        feat = self.transformer(emb)
        # feat = feat.permute(0, 2, 1)
        # enc_param = self.maxpooling(feat)
        # b, n, c = enc_param.shape
        # output = enc_param.view(b, 3, 128, 128)
        # output = torch.reshape(enc_param, (b, 3, 128, 128))
        # for i in range(b):
        #   for j in range(n):
        #     patch_vec = enc_param[i, j, :]

        # distance = torch.sum((feat[:, 0] - self.center).pow(2), 1)
        # scores = distance - self.radius.pow(2)
        # max_scores = torch.mean(torch.clamp(scores, min=0.0))/self.nu
        # loss_sphere = self.radius.pow(2) + max_scores

        #decoder
        pdb.set_trace()
        output = self.decoder(feat)

        # classifier
        # logits = self.classifier(feat[:, 0])
        return output


if __name__ == '__main__':
    model = VisionTransformer(num_layers=2)
    x = torch.randn((2, 3, 256, 256))
    out = model(x)

    state_dict = model.state_dict()

    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))

