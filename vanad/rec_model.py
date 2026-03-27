import torch
import os
from . import models_mae
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
from . import util
import numpy as np
from utils.GAF import GramianAngularField,GAFRecovery
from huggingface_hub import snapshot_download
import os
from pathlib import Path
import pywt
import ptwt  

MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"

class VANAD_Rec(nn.Module):

    def __init__(self, 
                 arch='mae_base', 
                 finetune_type='ln', 
                 ckpt_dir='./ckpt/', 
                 load_ckpt=True, 
                 image_method='seg', 
                 image_mask='random'
                 ):
        super(VANAD_Rec, self).__init__()

        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")

        self.vision_model = MAE_ARCH[arch][0]()

        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[arch][1]
                util.download_file(remote_url, ckpt_path)
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                self.vision_model.load_state_dict(checkpoint['model'], strict=True)
            except:
                print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")
        
        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n
        
        self.image_method = image_method
        self.image_mask = image_mask
        
        

    
    def update_config(self, seq_len, periodicity=1, norm_const=0.4, interpolation='bilinear',num_vars = None):
        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.seq_len = seq_len
        self.periodicity = periodicity
        
        if self.image_method == 'gaf':  
            self.gaf = GramianAngularField(image_size=seq_len, method='summation')
            self.recover_gaf = GAFRecovery(image_size=seq_len,method="summation")
        elif self.image_method == "stft":
            self.n_fft = 64
            self.hop_length = 8
            self.register_buffer('window', torch.hann_window(self.n_fft))
        elif self.image_method == 'wavelet':
            self.scales = range(1, 64 + 1)
            self.wavelet_name = 'cmor1.5-1.0'  # 复数 Morlet 小波

        self.pad = 0
        if self.seq_len % self.periodicity != 0:
            self.pad = self.periodicity - self.seq_len % self.periodicity


        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        self.input_resize = util.safe_resize((self.image_size, self.image_size), interpolation=interpolation)
        if self.image_method == 'seg':
            self.output_resize = util.safe_resize((self.periodicity, (self.seq_len + self.pad)//self.periodicity), interpolation=interpolation)
        elif self.image_method == 'gaf':
            self.output_resize = util.safe_resize((self.seq_len, self.seq_len), interpolation=interpolation)
        elif self.image_method == 'stft':
            self.output_resize = util.safe_resize(((self.n_fft // 2 + 1) * 2, (self.seq_len // self.hop_length  + 1)), interpolation=interpolation)
        elif self.image_method == 'wavelet':
            self.output_resize = util.safe_resize((len(self.scales), self.seq_len), interpolation=interpolation)
        elif self.image_method == 'rp':
            self.output_resize = util.safe_resize((self.seq_len, self.seq_len), interpolation=interpolation)
        elif self.image_method == "multi_var":
            self.output_resize = util.safe_resize((num_vars, self.seq_len), interpolation=interpolation)
        self.norm_const = norm_const
        
        # mask
        mask1 = torch.zeros((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask2 = torch.zeros((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        if self.image_mask == "complementary":
            # 0 1 0 1 | 1 0 1 0
            # 1 0 1 0 | 0 1 0 1
            # 0 1 0 1 | 1 0 1 0
            # 1 0 1 0 | 0 1 0 1
            for i in range(self.num_patch):
                for j in range(self.num_patch):
                    if i % 2 == 0:
                        if j % 2 == 0:
                            mask1[i, j] = 0
                            mask2[i, j] = 1
                        else:
                            mask1[i, j] = 1
                            mask2[i, j] = 0
                    else:
                        if j % 2 == 0:
                            mask1[i, j] = 1
                            mask2[i, j] = 0
                        else:
                            mask1[i, j] = 0
                            mask2[i, j] = 1
        elif self.image_mask == "complementary_row":
            # 0 1 0 1 | 1 0 1 0
            # 0 1 0 1 | 1 0 1 0
            # 0 1 0 1 | 1 0 1 0
            # 0 1 0 1 | 1 0 1 0
            for i in range(self.num_patch):
                for j in range(self.num_patch):
                    if j % 2 == 0:
                        mask1[i, j] = 0
                        mask2[i, j] = 1
                    else:
                        mask1[i, j] = 1
                        mask2[i, j] = 0
        elif self.image_mask == "complementary_col":
            # 1 1 1 1 | 0 0 0 0 
            # 0 0 0 0 | 1 1 1 1 
            # 1 1 1 1 | 0 0 0 0 
            # 0 0 0 0 | 1 1 1 1
            for i in range(self.num_patch):
                for j in range(self.num_patch):
                    if i % 2 == 0:
                        mask1[i, j] = 0
                        mask2[i, j] = 1
                    else:
                        mask1[i, j] = 1
                        mask2[i, j] = 0         
            
        self.register_buffer("mask1", mask1.float().reshape((1, -1)))
        self.register_buffer("mask2", mask2.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask1).item()
        
        
    
    def series_image(self, x, fp64=False):
        # time to image
        # 1 segmentation
        batch,s,n = x.shape
        self.batch = batch
        # 1. Normalization
        self.means = x.mean(1, keepdim=True).detach()  # [bs x 1 x nvars]
        x_enc = x - self.means
        self.stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()  # [bs x 1 x nvars]
        self.stdev /= self.norm_const
        x_enc /= self.stdev
        if self.image_method == 'seg':
            # Channel Independent
            x_enc = einops.rearrange(x, 'b s n -> b n s') # [bs x nvars x seq_len]

            # 3. Render & Alignment
            x_pad = F.pad(x_enc, (self.pad, 0), mode='replicate') # [b n s] 
            x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)
        # 2. GAF
        elif self.image_method == 'gaf':
            self.a = torch.min(x, 1, keepdim=True).values.detach()
            self.b = torch.max(x, 1, keepdim=True).values.detach()
            x_norm = x.sub(self.a).div(self.b-self.a + 1e-5)
            x_enc = x_norm.mul(self.norm_const)

            x_enc = einops.rearrange(x_enc, 'b l n -> (b n) l')
            x_2d = self.gaf.transform(x_enc) # [b*n l l]
            x_2d = einops.rearrange(x_2d, 'b h w -> b 1 h w')
        # 3. STFT (Short-Time Fourier Transform)
        elif self.image_method == 'stft':
            # 论文提到：通过滑动窗口函数 g(t) 计算 DFT [cite: 118]
            # 这里的 n_fft 对应频率轴分辨率，hop_length 决定时间轴分辨率
            x_enc = einops.rearrange(x, 'b s n -> (b n) s')
            
            # 使用 Hanning 窗口减少频谱泄漏 [cite: 122]
            stft_matrix = torch.stft(
                x_enc, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                window=self.window.to(x_enc.device),
                return_complex=True
            )
            
            # 论文指出：使用平方幅值 |f(w, τ)|^2 来绘制热力图 
            energy = torch.abs(stft_matrix).pow(2)
            real,imag = torch.real(stft_matrix), torch.imag(stft_matrix)
            x_2d = torch.stack([real, imag], dim=2) # [Batch*Vars, Freq, 3, Time]
            x_2d = einops.rearrange(x_2d, 'b f c t -> b (f c) t') # [Batch*Vars, Freq*3, Time]
            # 通常进行 log 缩放以增强视觉特征，并调整为 [Batch*Vars, 1, Freq, Time]
            # x_2d = torch.log1p(x_2d) 
            x_2d = x_2d.unsqueeze(1)
        # 4. WAVL (Continuous Wavelet Transform)
        elif self.image_method == 'wavelet':
            
            # 1. 准备数据：[Batch*Vars, Seq_len]
            x_enc = einops.rearrange(x, 'b s n -> (b n) s')
            cwt_coeffs, _ = ptwt.cwt(x_enc, self.scales, self.wavelet_name)
            x_2d = torch.abs(cwt_coeffs).to(x.dtype)  
            x_2d = einops.rearrange(x_2d, 's b l -> b 1 s l')
        # 5. RP (Recurrence Plot)
        elif self.image_method == 'rp':
            # 论文 §3.5: RP 将 UTS 编码为捕捉周期模式的图像 
            # 参数说明：m 为相空间维度，tau 为时间延迟 [cite: 160]
            # l = T - (m - 1) * tau [cite: 161]
            x_enc = einops.rearrange(x, 'b s n -> (b n) s')
            
            # 时间延迟嵌入 (Time Delay Embedding)
            # 模仿论文公式 (4): Vi = [x_t, x_{t+tau}, ..., x_{t+(m-1)tau}] [cite: 160]
            # 使用 unfold 快速构建相空间向量
            # m 和 tau 建议在 __init__ 中定义为类属性
            m = 3
            tau = 1
            
            # vectors 维度: [Batch*Vars, l, m]
            vectors = x_enc.unfold(dimension=-1, size=(m-1)*tau + 1, step=1)
            vectors = vectors[:, :, ::tau] 
            
            # 计算成对距离 (Pairwise Distances)
            # 论文公式 (5): RP_ij = Theta(epsilon - ||vi - vj||) [cite: 162]
            # 提示：论文提到为了避免信息丢失，可以省略阈值处理，直接使用距离 
            
            # 计算欧氏距离矩阵: ||vi - vj||
            # 使用 torch.cdist 高效计算 [Batch*Vars, l, l]
            dist_matrix = torch.cdist(vectors, vectors, p=2)
            
            # 如果需要严格遵守论文公式 (5) 的二值化[cite: 164]:
            if hasattr(self, 'epsilon') and self.epsilon is not None:
                x_2d = (dist_matrix < self.epsilon).float()
            else:
                # 论文提到使用连续值图像可以保留更多信息 [cite: 171]
                x_2d = dist_matrix
                
            x_2d = x_2d.unsqueeze(1) # [Batch*Vars, 1, l, l]
        elif self.image_method == 'multi_var': 
            # Channel Independent
            x_enc = einops.rearrange(x, 'b s n -> b n s') # [bs x nvars x seq_len]

            # 3. Render & Alignment
            x_pad = F.pad(x_enc, (self.pad, 0), mode='replicate') # [b n s] 
            x_2d = einops.rearrange(x_pad, 'b n s -> b 1 n s')
        return x_2d
    
    def image_series(self, image_reconstructed,denormalize):
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # color image to grey
        y_segmentations = self.output_resize(y_grey) # resize back
        if self.image_method == 'seg':     
            y = einops.rearrange(
                y_segmentations, 
                '(b n) 1 f p -> b (p f) n', 
                b=self.batch, f=self.periodicity
            ) # flatten
            y = y[:, :self.seq_len, :]
            # 6. Denormalization
            # y = y * (self.stdev.repeat(1, self.seq_len, 1))
            # y = y + (self.means.repeat(1, self.seq_len, 1))
        elif self.image_method == 'gaf':
            y_img = einops.rearrange(
                y_segmentations, 
                'b 1 h w -> b h w', 
            )
            y_recover = self.recover_gaf.recover(y_img)
            y = einops.rearrange(
                y_recover, 
                '(b n) l -> b l n', 
                b=self.batch
            )
            y = self.recover_gaf.recover_with_known_range(y, self.a, self.b)
        elif self.image_method == 'stft':
            y = y_segmentations.squeeze(1) # [BN, Freq*3, Time]
            real,imag = y[:, 0::2, :], y[:, 1::2, :]
            stft_matrix = torch.complex(real, imag)
            y_recover = torch.istft(
                stft_matrix, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                window=self.window.to(y.device),
                length=self.seq_len,
            )
            y = einops.rearrange(
                y_recover, 
                '(b n) l -> b l n', 
                b=self.batch
            )
        # 5. RP (Recurrence Plot) Recovery
        elif self.image_method == 'rp':
            # 论文 §5 备注：RP 恢复比较困难，因为其记录的是点对点的相似性 
            # 一种近似恢复策略：由于 RP 矩阵的主对角线 (i=j) 永远是 0 (距离最小)
            # 我们通常通过分析距离矩阵的某一列或加权平均来推取相对数值
            
            # 去掉通道维: [BN, l, l]
            y_img = y_segmentations.squeeze(1)
            
            # 近似方案：取每一行/列的均值，这反映了该时间点相对于整体序列的“平均距离特征”
            # 或者取第一行 y_img[:, 0, :]，代表所有点相对于起始点的距离
            y_recover = torch.mean(y_img, dim=-1) # [BN, l]
            
            # 将 Batch*Vars 展开回原始维度 [B, l, N]
            y = einops.rearrange(
                y_recover, 
                '(b n) l -> b l n', 
                b=self.batch
            )
            
            # 注意：RP 转换时通常不依赖 A/B 范围归一化 [cite: 267]
            # 如果你在预处理时手动记录了原序列的 mean/std，此处应进行反归一化
            # if hasattr(self, 'stdev') and hasattr(self, 'means'):
            #     y = y * self.stdev + self.means
        elif self.image_method == 'wavelet':
            # y_segmentations 维度: [BN, 1, n_scales, seq_len]
            # 论文指出：y 轴代表尺度(频率)，x 轴代表时间 [cite: 132]
            
            # 1. 去掉通道维，得到 [BN, n_scales, seq_len]
            coeffs_mag = y_segmentations.squeeze(1) 
            
            # 2. 能量叠加近似恢复：
            # 在所有尺度上求均值。这反映了每个时间步在所有观测频率下的综合强度。
            y_recover = torch.mean(coeffs_mag, dim=1) # [BN, seq_len]
            
            # 3. 重排回 [B, L, N]
            y = einops.rearrange(
                y_recover, 
                '(b n) l -> b l n', 
                b=self.batch
            )
            
            # 4. 按照论文 §5 的建议进行反归一化 [cite: 261, 264]
            # if hasattr(self, 'stdev') and hasattr(self, 'means'):
            #     y = y * self.stdev + self.means       
        elif self.image_method == 'multi_var': 
            y = einops.rearrange(
                y_segmentations, 
                'b 1 n s -> b s n', 
            ) # flatten
        
        # 6. Denormalization
        if denormalize:
            y_dec = y * (self.stdev.repeat(1, y.shape[1], 1))
            y_dec = y_dec + (self.means.repeat(1, y.shape[1], 1))
        else:
            y_dec = y
        return y_dec, y
        
    def forward(self, x, fp64=False,denormalize=False):
        # Forecasting using visual model.
        # x: look-back window, size: [bs x context_len x nvars]
        # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
        # return: forecasting window, size: [bs x pred_len x nvars]

        # 1. Normalization
        # 2. to 2d
        x_2d = self.series_image(x, fp64)

        # 3. Render & Alignment
        x_resize = self.input_resize(x_2d)
        
        image_input = einops.repeat(x_resize, 'b 1 h w -> b c h w', c=3)

        if self.image_mask == 'random':
            # 这里直接在 patch 层面聚合，逻辑更清晰
            combined_y = torch.zeros((image_input.shape[0], 196, 768), device=x.device)
            count_mask = torch.zeros((image_input.shape[0], 196, 1), device=x.device)
            for _ in range(4):
                _, y1, mask1 = self.vision_model(image_input)
                y1 = y1 * (mask1.unsqueeze(-1).repeat(1, 1, y1.shape[-1]))
                combined_y += y1
                count_mask += mask1.unsqueeze(-1)
            final_y = combined_y / torch.clamp(count_mask, min=1e-6) 
            #
            image_reconstructed = self.vision_model.unpatchify(final_y)
        elif self.image_mask == 'complementary' or self.image_mask == 'complementary_row' or self.image_mask == 'complementary_col':
            # 4. Reconstruction
            # mask1 reconstruction
            _, y1, mask1 = self.vision_model(
                image_input, 
                mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask1, '1 l -> n l', n=image_input.shape[0])
            )
            y1 = y1 * (mask1.unsqueeze(-1).repeat(1, 1, y1.shape[-1]))
            image_reconstructed1 = self.vision_model.unpatchify(y1) # [(bs x nvars) x 3 x h x w]
            
            # mask2 reconstruction
            _, y2, mask2 = self.vision_model(
                image_input, 
                mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask2, '1 l -> n l', n=image_input.shape[0])
            )
            y2 = y2 * (mask2.unsqueeze(-1).repeat(1, 1, y2.shape[-1]))
            image_reconstructed2 = self.vision_model.unpatchify(y2) # [(bs x nvars) x 3 x h x w]\
                
            image_reconstructed = image_reconstructed1 + image_reconstructed2
        # 5. Forecasting
        y_dec, y = self.image_series(image_reconstructed,denormalize)
        return y_dec, y