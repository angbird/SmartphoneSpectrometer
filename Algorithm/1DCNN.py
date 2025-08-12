import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def snv(X):
    X_mean = np.mean(X, axis=1, keepdims=True)  # 计算每个样本的均值
    X_std = np.std(X, axis=1, keepdims=True)  # 计算每个样本的标准差
    X_snv = (X - X_mean) / X_std  # 对每个样本进行 SNV 处理
    return X_snv


# 定义 1D - CNN 网络
class OneDCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(OneDCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(32 * (100 // 4), 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(-1, 32 * (100 // 4))
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 读取数据
df = pd.read_csv('tp_data_8pi.csv', encoding='ISO - 8859 - 1')  # 你可以根据情况选择适当的编码格式

# 提取标签和特征
# y = df['RefTP'].values
y = df['Name'].values
X = df['TP_Spec'].apply(lambda x: np.array(list(map(float, x.split())))[350:450]).values
X = np.vstack(X)

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, 'scaler_1dcnn.pkl')  # 保存标准化器到文件

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)  # 70%训练，30%临时
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)  # 15%验证，15%测试

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建 DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 实例化模型、损失函数和优化器
model = OneDCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
best_model_wts = None  # 存储训练过程中表现最好的模型权重
best_val_loss = float('inf')  # 存储验证集上观察到的最低损失值,初始值设置为正无穷大

# 记录训练损失和验证损失
train_losses = []
val_losses = []
train_r2s = []
val_r2s = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_predictions = []
    train_labels = []
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_predictions.extend(outputs.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # 计算训练集的 R2
    train_r2 = r2_score(train_labels, train_predictions)
    train_r2s.append(train_r2)

    # 验证模型
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_predictions.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # 计算验证集的 R2
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
    predictions = model(X_test)
    print(predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f'Test MSE: {mse:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test MAE: {mae:.4f}')
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
# df_data.to_csv('training_metrics_1dcnn_an.csv', index=False)

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
# plt.title('RF result')
plt.grid()
plt.show()

# torch.save(model.state_dict(), 'model_1dcnn_an.pth')  # 保存模型的权重和参数