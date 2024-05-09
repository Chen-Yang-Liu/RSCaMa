import torch
from torch import nn, einsum
import torchvision.models as models
from einops import rearrange
import clip
from transformers import MambaConfig
from model.mamba_block import CaMambaModel
# from transformers.models.mamba.modeling_mamba import MambaRMSNorm
class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            cnn = models.vgg19(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            cnn = models.inception_v3(pretrained=True, aux_logits=False)  
            modules = list(cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            cnn = models.resnet18(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            cnn = models.resnet34(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            cnn = models.resnet50(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)  
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet152': #512,1/32H,1/32W
            cnn = models.resnet152(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            cnn = models.resnext50_32x4d(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            cnn = models.resnext101_32x8d(pretrained=True)  
            modules = list(cnn.children())[:-1]

        elif 'CLIP' in self.network:
            clip_model_type = self.network.replace('CLIP-', '')
            self.clip_model, preprocess = clip.load(clip_model_type, jit=False)  #
            self.clip_model = self.clip_model.to(dtype=torch.float32)

        # self.cnn_list = nn.ModuleList(modules)
        # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, imageA, imageB):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        if "CLIP" in self.network:
            img_A = imageA.to(dtype=torch.float32)
            img_B = imageB.to(dtype=torch.float32)
            clip_emb_A, img_feat_A = self.clip_model.encode_image(img_A)
            clip_emb_B, img_feat_B = self.clip_model.encode_image(img_B)

        else:
            # feat1 = self.cnn(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
            # feat2 = self.cnn(imageB)
            feat1 = imageA
            feat2 = imageB
            feat1_list = []
            feat2_list = []
            cnn_list = list(self.cnn.children())
            for module in cnn_list:
                feat1 = module(feat1)
                feat2 = module(feat2)
                feat1_list.append(feat1)
                feat2_list.append(feat2)
            feat1_list = feat1_list[-4:]
            feat2_list = feat2_list[-4:]

        return img_feat_A, img_feat_B

    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 3 through 4
        if 'CLIP' in self.network and fine_tune:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune last 2 trans and ln_post
            children_list = list(self.clip_model.visual.transformer.resblocks.children())[-6:]
            children_list.append(self.clip_model.visual.ln_post)
            for c in children_list:
                for p in c.parameters():
                    p.requires_grad = True
        elif 'CLIP' not in self.network and fine_tune:
            for c in list(self.cnn.children())[:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune


class resblock(nn.Module):
    '''
    module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
            # nn.Conv2d(inchannel, int(outchannel / 1), kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), int(outchannel / 1), kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), outchannel, kernel_size=1),
            # nn.LayerNorm(int(outchannel / 1),dim=1)
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return self.act(out)


class AttentiveEncoder(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, n_layers, feature_size, heads, dropout=0.):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.n_layers = n_layers
        self.channels = channels
        # position embedding
        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))
        # Mamba
        config_1 = MambaConfig(num_hidden_layers=1, conv_kernel=3,hidden_size=channels)
        config_2 = MambaConfig(num_hidden_layers=1, conv_kernel=3,hidden_size=channels)
        self.CaMalayer_list = nn.ModuleList([])
        self.fuselayer_list = nn.ModuleList([])
        self.fuselayer_list_2 = nn.ModuleList([])
        self.linear_dif = nn.ModuleList([])
        self.linear_img1 = nn.ModuleList([])
        self.linear_img2 = nn.ModuleList([])
        self.Dyconv_img1_list = nn.ModuleList([])
        self.Dyconv_img2_list = nn.ModuleList([])
        embed_dim = channels
        self.Conv1_list = nn.ModuleList([])
        self.LN_list = nn.ModuleList([])
        for i in range(n_layers):
            self.CaMalayer_list.append(nn.ModuleList([
                CaMambaModel(config_1),
                CaMambaModel(config_1),
            ]))
            self.fuselayer_list.append(nn.ModuleList([
                CaMambaModel(config_2),
                CaMambaModel(config_2),
            ]))
            # self.linear_dif.append(nn.Sequential(
            #     nn.Linear(channels, channels),
            #     # nn.SiLU(),
            # ))
            # self.Dyconv_img1_list.append(Dynamic_conv(channels))
            # self.Dyconv_img2_list.append(Dynamic_conv(channels))
            # self.Dyconv_dif_list.append(Dynamic_conv(channels))
            # self.linear_img1.append(nn.Linear(2*channels, channels))
            # self.linear_img2.append(nn.Linear(2*channels, channels))
            self.Conv1_list.append(nn.Conv2d(channels * 2, embed_dim, kernel_size=1))
            self.LN_list.append(resblock(embed_dim, embed_dim))
        self.act = nn.Tanh()
        self.layerscan = CaMambaModel(config_1)
        self.LN_norm = nn.LayerNorm(channels)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.alpha2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Fusion bi-temporal feat for captioning decoder
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_pos_embedding(self, x):
        if len(x.shape) == 3: # NLD
            b = x.shape[0]
            c = x.shape[-1]
            x = x.transpose(-1, 1).view(b, c, self.h_feat, self.w_feat)
        batch, c, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        # reshape back to NLD
        x = x.view(b, c, -1).transpose(-1, 1)  # NLD (b,hw,c)
        return x

    def forward(self, img_A, img_B):
        h, w = self.h_feat, self.w_feat

        # 1. A B feature from backbone  NLD
        img_A = self.add_pos_embedding(img_A)
        img_B = self.add_pos_embedding(img_B)

        # captioning
        batch, c = img_A.shape[0], img_A.shape[-1]
        img_sa1, img_sa2 = img_A, img_B

        # Method: Mamba
        # self.CaMalayer_list.train()
        img_list = []
        N, L, D = img_sa1.shape
        for i in range(self.n_layers):
            dif = img_sa2 - img_sa1
            img_sa1 = self.CaMalayer_list[i][0](inputs_embeds=img_sa1, inputs_embeds_2=dif).last_hidden_state
            img_sa2 = self.CaMalayer_list[i][1](inputs_embeds=img_sa2, inputs_embeds_2=dif).last_hidden_state

            scan_mode = 'scan-allpixel'
            if scan_mode == 'scan-allpixel':
                img_sa1 = self.LN_norm(img_sa1)#+img_sa1_res
                img_sa2 = self.LN_norm(img_sa2)#+img_sa2_res
                img_sa1_res = img_sa1
                img_sa2_res = img_sa2
                img_fuse1 = img_sa1.permute(0, 2, 1).unsqueeze(-1) # (N,D,L,1)
                img_fuse2 = img_sa2.permute(0, 2, 1).unsqueeze(-1)
                img_fuse = torch.cat([img_fuse1, img_fuse2], dim=-1).reshape(N, D, -1) # (N,D,L*2)
                img_fuse = self.fuselayer_list[i][0](inputs_embeds=img_fuse.permute(0, 2, 1)).last_hidden_state.permute(0, 2, 1) # (N,D,L*2)
                img_fuse = img_fuse.reshape(N, D, L, -1)

                img_sa1 = img_fuse[..., 0].permute(0, 2, 1)#[...,:D] # (N,L,D)
                img_sa2 = img_fuse[..., 1].permute(0, 2, 1)#[...,:D]
                #
                img_sa1 = self.LN_norm(img_sa1) + img_sa1_res*self.alpha
                img_sa2 = self.LN_norm(img_sa2) + img_sa2_res*self.alpha

            # # bitemporal fusion
            if i == self.n_layers-1:
                img1_cap = img_sa1.transpose(-1, 1).view(batch, c, h, w)
                img2_cap = img_sa2.transpose(-1, 1).view(batch, c, h, w)
                feat_cap = torch.cat([img1_cap, img2_cap], dim=1)
                feat_cap = self.LN_list[i](self.Conv1_list[i](feat_cap))
                # feat_cap = self.Conv1_list[i](feat_cap)
                img_fuse = feat_cap.view(batch, c, -1).transpose(-1, 1)#.unsqueeze(-1) # (batch_size, L, D)
                img_fuse = self.LN_norm(img_fuse).unsqueeze(-1)
                img_list.append(img_fuse)

        # Out
        feat_cap = img_list[-1][..., 0]
        feat_cap = feat_cap.transpose(-1, 1)
        return feat_cap

if __name__ == '__main__':
    # test
    img_A = torch.randn(16, 49, 768).cuda()
    img_B = torch.randn(16, 49, 768).cuda()
    encoder = AttentiveEncoder(n_layers=3, feature_size=(7, 7, 768), heads=8).cuda()
    feat_cap = encoder(img_A, img_B)
    print(feat_cap.shape)
    print(feat_cap)
    print('Done')