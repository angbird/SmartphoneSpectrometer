import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
# import SPA
import numpy as np
import pandas as pd
import torch
from kymatio import Scattering1D
import yuchuli

def snv(X):
    X_mean = np.mean(X, axis=1, keepdims=True)  # 计算每个样本的均值
    X_std = np.std(X, axis=1, keepdims=True)    # 计算每个样本的标准差
    X_snv = (X - X_mean) / X_std                 # 对每个样本进行 SNV 处理
    return X_snv

def scattering_transform(X, J=4, Q=8):
    scattering = Scattering1D(J=J, shape=(X.shape[1],), Q=Q)
    X_scattered = []

    for x in X:
        Sx = scattering(x)
        Sx = Sx[:1]
        # Sx = Sx[:23]
        # Sx = np.delete(Sx, order0[0], axis=0)  #移除零阶散射系数
        # print(Sx.shape)
        X_scattered.append(Sx.flatten())

    # 将所有展平后的结果转换为 numpy 数组，形状为 (n_samples, n_features)
    X_scattered = np.array(X_scattered)
    return X_scattered

# 定义残差块（Residual Block）,是ResNet模型的核心组成部分,它的设计允许网络在每个块中跳过一个或多个层
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

# 定义ResNet网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.fc(x)
        return x

# 读取数据
df = pd.read_csv('tp_data_8pi.csv', encoding='ISO-8859-1')  # 你可以根据情况选择适当的编码格式

# 提取标签和特征
y = df['Name'].values
X = df['TP_Spec'].apply(lambda x: np.array(list(map(float, x.split())))[350:450]).values
X = np.vstack(X)

# X_SG = SPA.sg_filter(X, 9, 3, 0)
# X = snv(X)

# X_msc, mean_spectrum, Fit = SPA.msc(X_SG)
# X_scattered = scattering_transform(X_snv)

# X_combined = np.concatenate((X_snv, X_scattered), axis=1)  # axis=1 表示按列拼接

# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)  # 70%训练，30%临时
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)  # 15%验证，15%测试


# X_train, X_remain, y_train, y_remain = yuchuli.spxy(X, y, test_size=0.3)
# X_val, X_test, y_val, y_test = yuchuli.spxy(X_remain, y_remain, test_size=0.5)

# 使用SPA选择特征
# num_selected_variables = 100  # 你希望选择的特征数量
# selected_wavelengths = SPA.successive_projections_algorithm(X_train, num_selected_variables)
#
#
# X_train = X_train[:, selected_wavelengths]
# X_val = X_val[:, selected_wavelengths]
# X_test = X_test[:, selected_wavelengths]

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)
# X_test = scaler.transform(X_test)
# X_val = scaler.transform(X_val)

joblib.dump(scaler, 'scaler_1dresnet.pkl')  # 保存标准化器到文件

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)  # 70%训练，30%临时
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)  # 15%验证，15%测试

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 划分训练集、验证集和测试集
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70%训练，30%临时
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15%验证，15%测试

# 创建DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# # 实例化模型、损失函数和优化器
model = ResNet(ResidualBlock, [2, 2, 2, 2])  # 使用简化的ResNet-18
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 200
best_model_wts = None  #存储训练过程中表现最好的模型权重
best_val_loss = float('inf') #储验证集上观察到的最低损失值,初始值设置为正无穷大

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
        outputs = model(inputs.unsqueeze(1))  # 增加1个通道维度，将输入数据形状转换为（批次大小，1，序列长度）
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        # train_predictions.extend(outputs.cpu().numpy())
        train_predictions.extend(outputs.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # 计算训练集的R2
    train_r2 = r2_score(train_labels, train_predictions)
    train_r2s.append(train_r2)

    # 验证模型
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_predictions.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # 计算验证集的R2
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
    predictions = model(X_test.unsqueeze(1))
    print(predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f'Test MSE: {mse:.4f}')
    print("R^2:", r2)


# 保存数据到CSV文件
data = {
    'Epoch': range(1, num_epochs + 1),
    'Train Loss': train_losses,
    'Validation Loss': val_losses,
    'Train R2': train_r2s,
    'Validation R2': val_r2s
}
df_data = pd.DataFrame(data)
df_data.to_csv('training_metrics_1d_resnet_tp2.csv', index=False)

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
plt.xlabel('ture value')
plt.ylabel('predict value')
# plt.title('RF result')
plt.grid()
plt.show()

# torch.save(model.state_dict(), 'model_1dresnet.pth')  # 保存模型的权重和参数




























# df = pd.read_csv('selected_data_101112131415.csv', encoding='ISO-8859-1')
#
#
# y = df['RefTP'].values
# X = df['TP_Spec'].apply(lambda x: np.array(list(map(float, x.split())))[350:450]).values
# X = np.vstack(X)
#
# scaler = joblib.load('scaler_1dresnet.pkl')
# model.load_state_dict(torch.load('model_1dresnet.pth'))
#
# # 切换到评估模式
# model.eval()  # 设置模型为评估模式，关闭 Dropout 等训练时才启用的层
# # X = yuchuli.scattering_transform(X)
# # X = yuchuli.snv(X)
# X = scaler.transform(X)
#
# X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
# y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
#
#
# with torch.no_grad():  # 关闭梯度计算
#     y_pred_test = model(X)  # 测试集上的预测值
#
#
# print("Predicted shape:", y_pred_test)
# print("True shape:", y.shape)
#
#
# # 计算均方误差 (MSE)
# mse_test = mean_squared_error(y, y_pred_test)
# print(f"Test MSE: {mse_test:.4f}")
#
# # 计算决定系数 (R²)
# r2_test = r2_score(y, y_pred_test)
# print(f"Test R²: {r2_test:.4f}")
#
# # 可视化预测结果与真实值的比较
# plt.scatter(y, y_pred_test, color='blue')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)
# plt.xlabel('ture value')
# plt.ylabel('predict value')
# plt.title('RF result')
# plt.grid()
# plt.show()

