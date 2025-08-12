import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def snv(X):
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_snv = (X - X_mean) / X_std
    return X_snv


def scale_to_minus_one_to_one(x):
    min_val = np.min(x)
    max_val = np.max(x)
    scaled_x = 2 * ((x - min_val) / (max_val - min_val)) - 1
    return scaled_x


def gasf(x):
    x = scale_to_minus_one_to_one(x)
    phi = np.arccos(x)
    n = len(x)
    gasf_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gasf_matrix[i, j] = np.cos(phi[i] + phi[j])
    return gasf_matrix


def gadf(x):
    x = scale_to_minus_one_to_one(x)
    phi = np.arcsin(x)
    n = len(x)
    gadf_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gadf_matrix[i, j] = np.sin(phi[i] - phi[j])
    return gadf_matrix


# 双输入 2D CNN 模型
class DualInput2DCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(DualInput2DCNN, self).__init__()
        # GASF 分支
        self.conv1_gasf = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1_gasf = nn.ReLU()
        self.pool1_gasf = nn.MaxPool2d(kernel_size=2)
        self.conv2_gasf = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2_gasf = nn.ReLU()
        self.pool2_gasf = nn.MaxPool2d(kernel_size=2)

        # GADF 分支
        self.conv1_gadf = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1_gadf = nn.ReLU()
        self.pool1_gadf = nn.MaxPool2d(kernel_size=2)
        self.conv2_gadf = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2_gadf = nn.ReLU()
        self.pool2_gadf = nn.MaxPool2d(kernel_size=2)

        # 合并后的全连接层
        self.fc1 = nn.Linear(2 * 32 * (100 // 4) * (100 // 4), 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_gasf, x_gadf):
        # GASF 分支前向传播
        x_gasf = self.pool1_gasf(self.relu1_gasf(self.conv1_gasf(x_gasf)))
        x_gasf = self.pool2_gasf(self.relu2_gasf(self.conv2_gasf(x_gasf)))
        x_gasf = x_gasf.view(-1, 32 * (100 // 4) * (100 // 4))

        # GADF 分支前向传播
        x_gadf = self.pool1_gadf(self.relu1_gadf(self.conv1_gadf(x_gadf)))
        x_gadf = self.pool2_gadf(self.relu2_gadf(self.conv2_gadf(x_gadf)))
        x_gadf = x_gadf.view(-1, 32 * (100 // 4) * (100 // 4))

        # 合并两个分支的输出
        x = torch.cat((x_gasf, x_gadf), dim=1)

        # 全连接层前向传播
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 读取数据
df = pd.read_csv('tp_data.csv', encoding='ISO-8859-1')
# y = df['Name'].values
y = df['RefTP'].values
X = df['TP_Spec'].apply(lambda x: np.array(list(map(float, x.split())))[350:450]).values
X = np.vstack(X)

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler_dual_2dcnn.pkl')

# 转换为 GASF 和 GADF
X_gasf = np.array([gasf(x) for x in X])
X_gasf = X_gasf[:, np.newaxis, :, :]
X_gadf = np.array([gadf(x) for x in X])
X_gadf = X_gadf[:, np.newaxis, :, :]

# 划分训练集、验证集和测试集
X_gasf_train, X_gasf_temp, y_train, y_temp = train_test_split(X_gasf, y, test_size=0.3)
X_gasf_val, X_gasf_test, y_val, y_test = train_test_split(X_gasf_temp, y_temp, test_size=0.5)

X_gadf_train, X_gadf_temp, _, _ = train_test_split(X_gadf, y, test_size=0.3)
X_gadf_val, X_gadf_test, _, _ = train_test_split(X_gadf_temp, y_temp, test_size=0.5)

X_gasf_train = torch.tensor(X_gasf_train, dtype=torch.float32)
X_gasf_val = torch.tensor(X_gasf_val, dtype=torch.float32)
X_gasf_test = torch.tensor(X_gasf_test, dtype=torch.float32)

X_gadf_train = torch.tensor(X_gadf_train, dtype=torch.float32)
X_gadf_val = torch.tensor(X_gadf_val, dtype=torch.float32)
X_gadf_test = torch.tensor(X_gadf_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建 DataLoader
train_dataset = torch.utils.data.TensorDataset(X_gasf_train, X_gadf_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_gasf_val, X_gadf_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_gasf_test, X_gadf_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 实例化模型、损失函数和优化器
model = DualInput2DCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
best_model_wts = None
best_val_loss = float('inf')

train_losses = []
val_losses = []
train_r2s = []
val_r2s = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_predictions = []
    train_labels = []
    for inputs_gasf, inputs_gadf, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs_gasf, inputs_gadf)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs_gasf.size(0)
        train_predictions.extend(outputs.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    train_predictions = np.array(train_predictions)
    train_labels = np.array(train_labels)
    valid_indices = ~np.isnan(train_predictions).any(axis=1) & ~np.isnan(train_labels).any(axis=1)
    train_predictions = train_predictions[valid_indices]
    train_labels = train_labels[valid_indices]
    train_r2 = r2_score(train_labels, train_predictions)
    train_r2s.append(train_r2)

    # 验证模型
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for inputs_gasf, inputs_gadf, labels in val_loader:
            outputs = model(inputs_gasf, inputs_gadf)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs_gasf.size(0)
            val_predictions.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    val_predictions = np.array(val_predictions)
    val_labels = np.array(val_labels)
    valid_indices = ~np.isnan(val_predictions).any(axis=1) & ~np.isnan(val_labels).any(axis=1)
    val_predictions = val_predictions[valid_indices]
    val_labels = val_labels[valid_indices]
    val_r2 = r2_score(val_labels, val_predictions)
    val_r2s.append(val_r2)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = model.state_dict()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}')

# 加载最佳模型
model.load_state_dict(best_model_wts)

# 评估模型
model.eval()
with torch.no_grad():
    predictions = model(X_gasf_test, X_gadf_test)
    print(predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f'Test MSE: {mse:.4f}')
    print("R^2:", r2)

# 保存数据到 CSV 文件
data = {
    'Epoch': range(1, num_epochs + 1),
    'Train Loss': train_losses,
    'Validation Loss': val_losses,
    'Train R2': train_r2s,
    'Validation R2': val_r2s
}
df_data = pd.DataFrame(data)
df_data.to_csv('training_metrics_2dcnn_fusion_tp.csv', index=False)

# 绘制损失图
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 绘图
plt.scatter(y_test, predictions, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('true value')
plt.ylabel('predict value')
plt.grid()
plt.show()

torch.save(model.state_dict(), 'model_2dcnn_fusion_tp.pth')