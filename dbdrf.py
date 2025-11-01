import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 创建输出目录
os.makedirs('DBDRFOutput', exist_ok=True)

# 定义DNN模型
class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNN, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 数据预处理
def load_data(file_path):
    return pd.read_csv(file_path)

# 加载数据集
print("Loading datasets...")
dbd_train_data = load_data('DBDDataset/DBD Train set.CSV')
dbd_test_data = load_data('DBDDataset/DBD Test set 10000Hz 2450V.CSV')
rf_train_data = load_data('RFDataset/RF Train set.CSV')
rf_test_data = load_data('RFDataset/RF Test set 112.5mA.CSV')

# 准备DBD训练数据
X_dbd_train = dbd_train_data[['Voltage']].values
y_dbd_train = dbd_train_data[['Current density']].values

# 准备DBD测试数据
X_dbd_test = dbd_test_data[['Voltage']].values
y_dbd_test = dbd_test_data[['Current density']].values

# 准备RF训练数据
X_rf_train = rf_train_data[['Current density']].values
y_rf_train = rf_train_data[['Voltage']].values

# 准备RF测试数据
X_rf_test = rf_test_data[['Current density']].values
y_rf_test = rf_test_data[['Voltage']].values

# 数据标准化
print("Scaling data...")
scaler_X_dbd = MinMaxScaler()
scaler_y_dbd = MinMaxScaler()
scaler_X_rf = MinMaxScaler()
scaler_y_rf = MinMaxScaler()

X_dbd_train_scaled = scaler_X_dbd.fit_transform(X_dbd_train)
y_dbd_train_scaled = scaler_y_dbd.fit_transform(y_dbd_train)
X_dbd_test_scaled = scaler_X_dbd.transform(X_dbd_test)
y_dbd_test_scaled = scaler_y_dbd.transform(y_dbd_test)

X_rf_train_scaled = scaler_X_rf.fit_transform(X_rf_train)
y_rf_train_scaled = scaler_y_rf.fit_transform(y_rf_train)
X_rf_test_scaled = scaler_X_rf.transform(X_rf_test)
y_rf_test_scaled = scaler_y_rf.transform(y_rf_test)

# 转换为PyTorch张量
X_dbd_train_tensor = torch.tensor(X_dbd_train_scaled, dtype=torch.float32)
y_dbd_train_tensor = torch.tensor(y_dbd_train_scaled, dtype=torch.float32)
X_dbd_test_tensor = torch.tensor(X_dbd_test_scaled, dtype=torch.float32)
y_dbd_test_tensor = torch.tensor(y_dbd_test_scaled, dtype=torch.float32)

X_rf_train_tensor = torch.tensor(X_rf_train_scaled, dtype=torch.float32)
y_rf_train_tensor = torch.tensor(y_rf_train_scaled, dtype=torch.float32)
X_rf_test_tensor = torch.tensor(X_rf_test_scaled, dtype=torch.float32)
y_rf_test_tensor = torch.tensor(y_rf_test_scaled, dtype=torch.float32)

# 创建模型
print("Creating models...")
dbd_model = DNN(input_size=1, hidden_sizes=[64, 64, 32], output_size=1)
rf_model = DNN(input_size=1, hidden_sizes=[64, 64, 32], output_size=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
dbd_optimizer = optim.Adam(dbd_model.parameters(), lr=0.001)
rf_optimizer = optim.Adam(rf_model.parameters(), lr=0.001)

# 训练DBD模型
print("Training DBD model...")
dbd_losses = []
dbd_model.train()
for epoch in range(1000):
    dbd_optimizer.zero_grad()
    outputs = dbd_model(X_dbd_train_tensor)
    loss = criterion(outputs, y_dbd_train_tensor)
    loss.backward()
    dbd_optimizer.step()
    dbd_losses.append(loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'DBD Model - Epoch [{epoch+1}/1000], Loss: {loss.item():.6f}')

# 训练RF模型
print("Training RF model...")
rf_losses = []
rf_model.train()
for epoch in range(1000):
    rf_optimizer.zero_grad()
    outputs = rf_model(X_rf_train_tensor)
    loss = criterion(outputs, y_rf_train_tensor)
    loss.backward()
    rf_optimizer.step()
    rf_losses.append(loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'RF Model - Epoch [{epoch+1}/1000], Loss: {loss.item():.6f}')

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(dbd_losses, label='DBD Model Loss')
plt.plot(rf_losses, label='RF Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('DBDRFOutput/training_loss_curves.png')
plt.close()

# 评估模型
print("Evaluating models...")
dbd_model.eval()
rf_model.eval()

with torch.no_grad():
    dbd_test_pred_scaled = dbd_model(X_dbd_test_tensor)
    rf_test_pred_scaled = rf_model(X_rf_test_tensor)
    
    # 反标准化预测结果
    dbd_test_pred = scaler_y_dbd.inverse_transform(dbd_test_pred_scaled.numpy())
    rf_test_pred = scaler_y_rf.inverse_transform(rf_test_pred_scaled.numpy())

# 绘制四张图
print("Generating plots...")

# 左上图：双纵轴时域图
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_xlabel('Time / μs')
ax1.set_ylabel('Current density / mA·cm⁻²', color='tab:blue')
line1 = ax1.plot(dbd_test_data['Cycles'] * 1e6, dbd_test_data['Current density'] * 1000, 
                 label='Actual Current Density', color='tab:blue', linewidth=1)
line2 = ax1.plot(dbd_test_data['Cycles'] * 1e6, dbd_test_pred.flatten() * 1000, 
                 label='Predicted Current Density', color='tab:cyan', linestyle='--', linewidth=1)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('Voltage / kV', color='tab:red')
line3 = ax2.plot(dbd_test_data['Cycles'] * 1e6, dbd_test_data['Voltage'] / 1000, 
                 label='Voltage', color='tab:red', linewidth=1)
ax2.tick_params(axis='y', labelcolor='tab:red')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')
plt.title('Time Domain Plot')
plt.tight_layout()
plt.savefig('DBDRFOutput/time_domain_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 右上图：电流密度随电压变化
plt.figure(figsize=(12, 8))
plt.plot(dbd_test_data['Voltage'], dbd_test_data['Current density'] * 1000, 
         label='Actual', marker='o', markersize=2, linestyle='-', linewidth=1)
plt.plot(dbd_test_data['Voltage'], dbd_test_pred.flatten() * 1000, 
         label='Predicted', marker='x', markersize=2, linestyle='-', linewidth=1)
plt.xlabel('Voltage / V')
plt.ylabel('Current density / mA·cm⁻²')
plt.title('Current Density vs Voltage')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('DBDRFOutput/current_density_vs_voltage.png', dpi=300, bbox_inches='tight')
plt.close()

# 左下图：电压随周期变化
plt.figure(figsize=(12, 8))
# 分离不同模式的数据点
rf_test_current = rf_test_data['Current density'].iloc[0]
# 假设前50%为alpha模式，后50%为gamma模式
split_point = len(rf_test_data) // 2
alpha_cycles = rf_test_data['Cycles'][:split_point]
alpha_voltage = rf_test_data['Voltage'][:split_point]
gamma_cycles = rf_test_data['Cycles'][split_point:]
gamma_voltage = rf_test_data['Voltage'][split_point:]

alpha_pred_voltage = rf_test_pred.flatten()[:split_point]
gamma_pred_voltage = rf_test_pred.flatten()[split_point:]

plt.plot(alpha_cycles, alpha_voltage, label='Alpha Mode (Actual)', marker='o', markersize=2, linewidth=1)
plt.plot(gamma_cycles, gamma_voltage, label='Gamma Mode (Actual)', marker='s', markersize=2, linewidth=1)
plt.plot(alpha_cycles, alpha_pred_voltage, label='Alpha Mode (Predicted)', marker='x', markersize=2, linewidth=1, linestyle='--')
plt.plot(gamma_cycles, gamma_pred_voltage, label='Gamma Mode (Predicted)', marker='^', markersize=2, linewidth=1, linestyle='--')
plt.xlabel('Cycles')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs Cycles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('DBDRFOutput/voltage_vs_cycles.png', dpi=300, bbox_inches='tight')
plt.close()

# 右下图：RMS值关系图
# 计算多个RMS值用于绘图
# 对于DBD数据，我们按周期计算RMS值
dbd_cycles = dbd_test_data['Cycles']
dbd_voltage = dbd_test_data['Voltage']
dbd_current_density = dbd_test_data['Current density']
dbd_pred_current_density = dbd_test_pred.flatten()

# 按周期分组计算RMS值
dbd_unique_cycles = np.unique(np.round(dbd_cycles, 4))
dbd_rms_current_list = []
dbd_rms_voltage_list = []
dbd_rms_pred_current_list = []

for cycle in dbd_unique_cycles:
    mask = np.round(dbd_cycles, 4) == cycle
    cycle_current = dbd_current_density[mask]
    cycle_voltage = dbd_voltage[mask]
    cycle_pred_current = dbd_pred_current_density[mask]
    
    dbd_rms_current_list.append(np.sqrt(np.mean(cycle_current**2)) * 1000)  # 转换为mA/cm²
    dbd_rms_voltage_list.append(np.sqrt(np.mean(cycle_voltage**2)))
    dbd_rms_pred_current_list.append(np.sqrt(np.mean(cycle_pred_current**2)) * 1000)  # 转换为mA/cm²

# 对于RF数据，同样按周期计算RMS值
rf_cycles = rf_test_data['Cycles']
rf_current_density = rf_test_data['Current density']
rf_voltage = rf_test_data['Voltage']
rf_pred_voltage = rf_test_pred.flatten()

rf_unique_cycles = np.unique(np.round(rf_cycles, 4))
rf_rms_current_list = []
rf_rms_voltage_list = []
rf_rms_pred_voltage_list = []

for cycle in rf_unique_cycles:
    mask = np.round(rf_cycles, 4) == cycle
    cycle_current = rf_current_density[mask]
    cycle_voltage = rf_voltage[mask]
    cycle_pred_voltage = rf_pred_voltage[mask]
    
    rf_rms_current_list.append(np.sqrt(np.mean(cycle_current**2)))
    rf_rms_voltage_list.append(np.sqrt(np.mean(cycle_voltage**2)))
    rf_rms_pred_voltage_list.append(np.sqrt(np.mean(cycle_pred_voltage**2)))

plt.figure(figsize=(12, 8))
# 绘制DBD RMS值
plt.scatter(dbd_rms_current_list, dbd_rms_voltage_list, 
            c='blue', label='DBD Actual', alpha=0.7, s=20)
plt.scatter(dbd_rms_pred_current_list, dbd_rms_voltage_list, 
            c='cyan', label='DBD Predicted', alpha=0.7, s=20, marker='^')

# 绘制RF RMS值
plt.scatter(rf_rms_current_list, rf_rms_voltage_list, 
            c='red', label='RF Actual', alpha=0.7, s=20)
plt.scatter(rf_rms_current_list, rf_rms_pred_voltage_list, 
            c='orange', label='RF Predicted', alpha=0.7, s=20, marker='^')

plt.xlabel('RMS total current density (mA/cm²)')
plt.ylabel('RMS voltage (V)')
plt.title('RMS Values Relationship')

# 标注关键转折点 (使用示例值)
gas_breakdown_point = (10, 1500)
alpha_gamma_transition_point = (50, 2000)
plt.scatter(*gas_breakdown_point, color='purple', marker='*', s=300, label='Gas Breakdown', zorder=5)
plt.scatter(*alpha_gamma_transition_point, color='green', marker='*', s=300, label='Alpha-Gamma Transition', zorder=5)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('DBDRFOutput/rms_values_relationship.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots have been saved to the 'DBDRFOutput' directory.")
