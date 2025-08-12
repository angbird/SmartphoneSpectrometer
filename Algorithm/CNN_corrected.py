import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# ========== CNN 模型定义 ==========
class CNNRegressor(nn.Module):
    def __init__(self, input_len=370, output_dim=500):
        super(CNNRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(50),
            nn.Flatten(),
            nn.Linear(64 * 50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# class CNNRegressor(nn.Module):
#     def __init__(self, output_dim=500):
#         super(CNNRegressor, self).__init__()
#
#         # Expanded convolutional blocks with batch normalization
#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=7, padding=3),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool1d(2)
#         )
#
#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(32, 64, kernel_size=5, padding=2),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool1d(2)
#         )
#
#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool1d(2)
#         )
#
#         # Attention mechanism
#         self.attention = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.Tanh(),
#             nn.Linear(64, 128),
#             nn.Softmax(dim=1)
#         )
#
#         # Adaptive pooling remains
#         self.adaptive_pool = nn.AdaptiveAvgPool1d(50)
#
#         # Expanded fully connected layers with dropout
#         self.fc = nn.Sequential(
#             nn.Linear(128 * 50, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, output_dim)
#         )
#
#         # Residual connection
#         self.residual = nn.Sequential(
#             nn.Conv1d(1, 128, kernel_size=1),
#             nn.MaxPool1d(8)
#         )
#
#     def forward(self, x):
#         # Original path
#         residual = self.residual(x)
#
#         x = self.conv_block1(x)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#
#         # Attention mechanism
#         attn_weights = self.attention(x.transpose(1, 2)).transpose(1, 2)
#         x = x * attn_weights
#
#         # Add residual connection
#         x = x + residual
#
#         # Continue with pooling and FC
#         x = self.adaptive_pool(x)
#         x = x.flatten(1)
#         x = self.fc(x)
#
#         return x


# class ResidualBlock(nn.Module):
#     def __init__(self, channels, kernel_size=3, padding=1):
#         super().__init__()
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
#         self.bn2 = nn.BatchNorm1d(channels)
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += residual
#         out = self.relu(out)
#         return out
#
# class CNNRegressor(nn.Module):
#     def __init__(self, input_len=370, output_dim=500):
#         super().__init__()
#         self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.resblock1 = ResidualBlock(32)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
#         self.bn2 = nn.BatchNorm1d(64)
#
#         self.resblock2 = ResidualBlock(64)
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
#
#         self.adaptive_pool = nn.AdaptiveAvgPool1d(50)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(128 * 50, 512)
#         self.dropout = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(512, output_dim)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.resblock1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#
#         x = self.resblock2(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#
#         x = self.adaptive_pool(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x


# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv1d(channels, channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(channels, channels, kernel_size=3, padding=1)
#         )
#
#     def forward(self, x):
#         return x + self.block(x)
#
# class CNNRegressor(nn.Module):
#     def __init__(self, input_len=370, output_dim=370):
#         super(CNNRegressor, self).__init__()
#
#         self.feature_extractor = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=7, padding=3),  # output: [B, 32, L]
#             nn.ReLU(),
#             ResidualBlock(32),                           # 加一层残差
#             nn.MaxPool1d(2),                             # output: [B, 32, L/2]
#
#             nn.Conv1d(32, 64, kernel_size=5, padding=2),
#             nn.ReLU(),
#             ResidualBlock(64),
#             nn.MaxPool1d(2),                             # output: [B, 64, L/4]
#
#             nn.Conv1d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(128),
#             nn.AdaptiveAvgPool1d(25),                    # 压缩成固定长度
#         )
#
#         self.regressor = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 25, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim)
#         )
#
#     def forward(self, x):
#         features = self.feature_extractor(x)  # x: [B, 1, 370]
#         output = self.regressor(features)
#         return output

# ========== Dataset 类 ==========
class SpectrumDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X[:, None, :], dtype=torch.float32)  # (N, 1, 1080)
        self.Y = torch.tensor(Y, dtype=torch.float32)              # (N, 1080)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

standard_df = pd.read_csv('./PhonesData/消除后处理_调整光源强度/xiaomi14/allData.csv', header=None)
target_df = pd.read_csv('./PhonesData/消除后处理_调整光源强度/redmik70/allData.csv', header=None)
standard_test_df = pd.read_csv("./PhonesData/消除相机后处理/xiaomi14/test/tp0.4.csv", header=None)
target_test_df = pd.read_csv("./PhonesData/消除相机后处理/redmik70/test/tp0.4.csv", header=None)

standard_test_df = standard_test_df[standard_test_df.iloc[:, 0] == "Intensity"]
target_test_df = target_test_df[target_test_df.iloc[:, 0] == "Intensity"]

a = 200
b = 700
# 2. 提取像素范围
standard_cal = standard_df.values[:, a:b]
target_cal = target_df.values[:, a:b]
standard_test = standard_test_df.values[:, a:b]
target_test = target_test_df.values[:, a:b]
wavelengths = np.arange(a, b)

standard_cal = standard_cal.astype(np.float32)
target_cal = target_cal.astype(np.float32)
standard_test = standard_test.astype(np.float32)
target_test = target_test.astype(np.float32)

eps = 1e-8
# standard_cal = standard_cal / (standard_cal[-1, :] + eps)
# target_cal = target_cal / (target_cal[-1, :] + eps)

# ========== 划分训练 / 测试 ==========
# standard_cal, standard_test, target_cal, target_test = train_test_split(standard_data, target_data, test_size=0.2, random_state=42)

print(f"校正集: {len(standard_cal)}, 测试集: {len(standard_test)}")

# ========== 准备训练 ==========
train_dataset = SpectrumDataset(target_cal, standard_cal)
test_dataset = SpectrumDataset(target_test, standard_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 训练模型 ==========
epochs = 300
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

# ========== 模型预测 ==========
model.eval()
corrected_test = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch).cpu().numpy()
        corrected_test.append(pred)
corrected_test = np.vstack(corrected_test)

# ========== 评估指标 ==========
mse_before = mean_squared_error(standard_test, target_test)
mse_after = mean_squared_error(standard_test, corrected_test)

mae_before = mean_absolute_error(standard_test, target_test)
mae_after = mean_absolute_error(standard_test, corrected_test)

r_before = [pearsonr(standard_test[i], target_test[i])[0] for i in range(len(standard_test))]
r_after = [pearsonr(standard_test[i], corrected_test[i])[0] for i in range(len(standard_test))]

print(f"MSE（校正前）:  {mse_before:.4f}")
print(f"MSE（校正后）:  {mse_after:.4f}")
print(f"MAE（校正前）:  {mae_before:.4f}")
print(f"MAE（校正后）:  {mae_after:.4f}")
print(f"平均 Pearson（校正前）: {np.mean(r_before):.4f}")
print(f"平均 Pearson（校正后）: {np.mean(r_after):.4f}")

# ========== 可视化 ==========
plt.figure(figsize=(12, 6))
# 只给第一条线加图例标签，其他的设 label=None 避免重复
for i in range(standard_test.shape[0]):
    label = 'Standard' if i == 0 else None
    plt.plot(wavelengths, standard_test[i], color='black', label=label)

for i in range(target_test.shape[0]):
    label = 'Target (Before Correction)' if i == 0 else None
    plt.plot(wavelengths, target_test[i], color='red', label=label)

for i in range(corrected_test.shape[0]):
    label = 'Target (After Correction)' if i == 0 else None
    plt.plot(wavelengths, corrected_test[i], color='green', label=label)

plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
# plt.title('Spectral Correction Result')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

torch.save(model.state_dict(), "cnn_corrected_spec.pth")
print("模型参数保存成功！")
