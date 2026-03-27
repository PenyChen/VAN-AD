

import torch
import math
import numpy as np
from scipy.optimize import least_squares

class GramianAngularField:

    def __init__(self, image_size=1, method='summation', sample_range=(0, 1)):
        self.image_size = image_size
        self.method = method
        self.sample_range = sample_range

    def transform(self, X):
        """
        X: torch.Tensor, shape (n_samples, n_timestamps)
        Returns: torch.Tensor, shape (n_samples, image_size, image_size)
        """
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        n_samples, n_timestamps = X.shape
        image_size = self._check_params(n_timestamps)

        # PAA (Piecewise Aggregate Approximation)
        if image_size < n_timestamps:
            X_paa = self._paa(X, image_size)
        else:
            X_paa = X

        # MinMax scaling
        if self.sample_range is not None:
            min_val, max_val = self.sample_range
            X_paa = (X_paa - X_paa.min(dim=1, keepdim=True)[0]) / (
                X_paa.max(dim=1, keepdim=True)[0] - X_paa.min(dim=1, keepdim=True)[0] + 1e-8
            )
            X_paa = X_paa * (max_val - min_val) + min_val

        # Angular encoding
        X_cos = X_paa
        X_sin = torch.sqrt(torch.clamp(1 - X_cos ** 2, min=0.0))

        # GAF computation
        if self.method in ['s', 'summation']:
            X_gaf = self._gasf(X_cos, X_sin)
        else:
            X_gaf = self._gadf(X_cos, X_sin)

        return X_gaf

    def _paa(self, X, output_size):
        """Piecewise Aggregate Approximation (PAA)"""
        n_samples, n_timestamps = X.shape
        seg_size = n_timestamps / output_size
        X_paa = torch.zeros((n_samples, output_size), dtype=X.dtype, device=X.device)
        for i in range(output_size):
            start = int(math.floor(i * seg_size))
            end = int(math.floor((i + 1) * seg_size))
            if end > start:
                X_paa[:, i] = X[:, start:end].mean(dim=1)
            else:
                X_paa[:, i] = X[:, start]
        return X_paa

    def _gasf(self, X_cos, X_sin):
        """Gramian Angular Summation Field"""
        # X_cos: (n_samples, image_size)
        outer_cos = torch.einsum('bi,bj->bij', X_cos, X_cos)
        outer_sin = torch.einsum('bi,bj->bij', X_sin, X_sin)
        return outer_cos - outer_sin

    def _gadf(self, X_cos, X_sin):
        """Gramian Angular Difference Field"""
        outer_sin_cos = torch.einsum('bi,bj->bij', X_sin, X_cos)
        outer_cos_sin = torch.einsum('bi,bj->bij', X_cos, X_sin)
        return outer_sin_cos - outer_cos_sin

    def _check_params(self, n_timestamps):
        if not isinstance(self.image_size, int):
            raise TypeError("'image_size' must be an integer.")
        if self.image_size < 1 or self.image_size > n_timestamps:
            raise ValueError(
                "image_size must be >= 1 and <= n_timestamps (got {}).".format(self.image_size)
            )
        if self.method not in ['s', 'd', 'summation', 'difference']:
            raise ValueError("'method' must be 'summation', 's', 'difference' or 'd'.")
        return self.image_size



class GAFRecovery:
    
    def __init__(self, image_size=1, method='summation', original_length=None):
        self.image_size = image_size
        self.method = method
        self.original_length = original_length if original_length else image_size
    
    def recover(self, X_gaf):
        """
        参数:
        X_gaf : torch.Tensor, shape (n_samples, image_size, image_size) 或 (image_size, image_size)
        
        返回:
        X_recovered : torch.Tensor, shape (n_samples, original_length)
        """
        if not torch.is_tensor(X_gaf):
            X_gaf = torch.tensor(X_gaf, dtype=torch.float32)
        
        # 处理输入维度
        if X_gaf.dim() == 2:
            # 单样本，添加批次维度
            X_gaf = X_gaf.unsqueeze(0)
        
        n_samples, h, w = X_gaf.shape
        if h != w or h != self.image_size:
            raise ValueError(f"输入GAF图像尺寸应为({self.image_size}, {self.image_size})，但得到({h}, {w})")
        
        if self.method in ['s', 'summation']:
            X_recovered = self._recover_from_gasf(X_gaf)
        else: 
            pass
        
        # 逆PAA恢复原始长度
        if self.original_length > self.image_size:
            X_recovered = self._inverse_paa(X_recovered, self.original_length)
        
        return X_recovered
    
    def _recover_from_gasf(self, X_gasf):
        """
        从GASF图像恢复序列
        对于GASF: GASF[i,j] = cos(φ_i + φ_j) = cos(φ_i)cos(φ_j) - sin(φ_i)sin(φ_j)
        """
        n_samples = X_gasf.shape[0]
        
        # 方法1: 直接从对角线恢复（最直接的方法）
        # 对角线元素: GASF[i,i] = cos(2φ_i)
        diag = torch.diagonal(X_gasf, dim1=1, dim2=2)
        
        # 计算cos(2φ_i)
        cos_2phi = torch.clamp(diag, -1.0, 1.0)
        
        # 计算φ_i
        # φ_i = 0.5 * arccos(cos_2phi)
        phi = 0.5 * torch.acos(cos_2phi)
        
        # 恢复标准化序列: x_i = cos(φ_i)
        X_norm = torch.cos(phi)
        
        return X_norm
    
    def _inverse_paa(self, X_paa, original_length):
        """
        逆PAA变换，将降采样序列恢复为原始长度
        简单线性插值方法
        """
        n_samples, paa_length = X_paa.shape
        
        # 创建输出张量
        X_full = torch.zeros((n_samples, original_length), dtype=X_paa.dtype, device=X_paa.device)
        
        # 计算插值点
        for i in range(n_samples):
            # 使用线性插值
            x_paa = X_paa[i].detach().cpu().numpy()
            x_full = np.interp(
                np.linspace(0, paa_length - 1, original_length),
                np.arange(paa_length),
                x_paa
            )
            X_full[i] = torch.tensor(x_full, dtype=X_paa.dtype, device=X_paa.device)
        
        return X_full
    
    def recover_with_known_range(self, x, original_min, original_max):
        """
        已知原始数据范围的恢复（更准确）
        x : b l 
        """
        a = torch.min(x,1,keepdim=True).values.detach()
        b = torch.max(x,1,keepdim=True).values.detach()
        x = x.sub(a).div(b - a)
        X_recovered = x * (original_max - original_min) + original_min
        return X_recovered


# 使用示例
def example_usage():
    # 创建示例序列
    original_length = 100
    t = torch.linspace(0, 10, original_length)
    X_original = torch.sin(t) + torch.cos(t) # 形状: (1, 100)
    X_original = X_original.unsqueeze(0)
    a = X_original.min()
    b = X_original.max()
    
    method = 'summation'
    gaf = GramianAngularField(image_size=100, method=method)
    X_gaf = gaf.transform(X_original)  # 形状: (1, 50, 50)
    
    print(f"原始序列形状: {X_original.shape}")
    print(f"GAF图像形状: {X_gaf.shape}")
    
    recovery = GAFRecovery(
        image_size=100, 
        method=method, 
    )
    X_recovered = recovery.recover(X_gaf)
    X_recovered = recovery.recover_with_known_range(X_recovered, a, b)
    print(f"恢复序列形状: {X_recovered.shape}")
    
    mse = torch.mean((X_original - X_recovered) ** 2).item()
    print(f"恢复MSE: {mse:.6f}")
    
    return X_original, X_recovered, X_gaf



if __name__ == "__main__":
    # 测试恢复
    X_original, X_recovered, X_gaf = example_usage()
    import matplotlib.pyplot as plt
    plt.plot(X_original.squeeze().detach().numpy(), label='Original')
    plt.plot(X_recovered.squeeze().detach().numpy(), label='Recovered')
    plt.legend()
    plt.savefig('gasf_recovered.png')