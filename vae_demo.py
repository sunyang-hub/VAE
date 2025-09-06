import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class TrainSet(Dataset):
    def __init__(self, X, Y):
        # 定义数据z
        self.X, self.Y = X, Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


class VAE2D(nn.Module):
    def __init__(self, latent_dim=64, in_channels=1):
        super(VAE2D, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        
        # 编码器 - 2D卷积层
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # 计算潜在变量的均值和方差
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)  # 针对64x64输入
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # 解码器 - 从潜在空间重建图像
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 输出范围[0,1]
        )

    def encode(self, x):
        # 编码过程：输入 -> 特征图 -> 均值和方差
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        # 重参数化技巧：从N(mu, var)中采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 解码过程：潜在变量 -> 重建图像
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 8, 8)  # 重塑为特征图
        return self.decoder(x)

    def forward(self, x):
        # 完整前向传播
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """VAE损失函数：重建损失 + KL散度"""
    # 重建损失（MSE或交叉熵）
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度：衡量与标准正态分布的差异
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_div


def main():
    # 创建2D图像示例数据 (batch_size, channels, height, width)
    X_tensor = torch.rand((1024, 1, 64, 64))  # 随机2D图像数据
    Y_tensor = X_tensor.clone()  # VAE是自编码器，输入即目标
    
    mydataset = TrainSet(X_tensor, Y_tensor) #[1024, 1, 64, 64])
    train_loader = DataLoader(mydataset, batch_size=32, shuffle=True)

    # 初始化2D VAE模型
    vae = VAE2D(latent_dim=64, in_channels=1)  # network
    print(vae)

    # 优化器
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # 训练循环
    for epoch in range(10):
        total_loss = 0
        for i, (X, y) in enumerate(train_loader):
            # 前向传播
            recon_x, mu, logvar = vae(X)
            loss = vae_loss(recon_x, X, mu, logvar)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f'epoch={epoch}, i={i}, loss={loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch} complete, Average Loss: {avg_loss:.4f}')


if __name__ == '__main__':
    main()
    