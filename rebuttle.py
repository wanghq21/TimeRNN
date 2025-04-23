import torch
import matplotlib.pyplot as plt

# 示例数据：100个时间步，10个通道
seqlen, channel = 100, 100
data = torch.randn(seqlen, channel)

# 计算相关系数矩阵
# 转置后，每一行为一个变量，每一列为一个观测值
data_for_corr = data.T  # shape: (10, 100)
corr_matrix = torch.corrcoef(data_for_corr)

# 将 tensor 转为 numpy 数组
corr_np = corr_matrix.cpu().numpy()

# 绘制热力图
plt.figure(figsize=(8, 6))
im = plt.imshow(corr_np, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title('Correlation Matrix Heatmap')
plt.xlabel('Channel')
plt.ylabel('Channel')
plt.xticks(range(channel))
plt.yticks(range(channel))
# plt.show()
plt.savefig('correlation.pdf')


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 示例数据：假设我们有 10 个通道，100 个时间步的数据
channel, seqlen = 500, 100
data = torch.randn(channel, seqlen)  # 随机生成示例数据

# 将数据转为 numpy 数组
data_np = data.numpy()

# 使用 t-SNE 降维到 2D
tsne = TSNE(n_components=2, perplexity=100, random_state=42)
data_tsne = tsne.fit_transform(data_np)

# 可视化降维后的数据
plt.figure(figsize=(8, 6))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c='b', label='Channels')
plt.title('t-SNE Visualization of Channels')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
# plt.show()
plt.savefig('tsne.pdf')
