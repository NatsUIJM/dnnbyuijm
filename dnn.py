import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# 创建保存图像的目录
os.makedirs('plots', exist_ok=True)

# 读取数据
phi_data = pd.read_csv('phi_700.csv', header=None)
rho_data = pd.read_csv('rho_700.csv', header=None)

# 数据预处理
X = rho_data.values  # 电荷密度作为输入
y = phi_data.values  # 电势作为输出

# 划分训练集和测试集 (2400训练, 600测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=600, random_state=42)

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 构建DNN模型
model = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128),  # 隐藏层结构
    activation='relu',                   # 激活函数
    solver='adam',                       # 优化器
    max_iter=500,                        # 最大迭代次数
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

# 训练模型
print("Starting model training...")
model.fit(X_train_scaled, y_train_scaled)
print("Model training completed")

# 在测试集上进行预测
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 计算损失 (均方误差)
test_loss = mean_squared_error(y_test, y_pred)
print(f"Test set loss (MSE): {test_loss}")

# 计算每个测试样本的误差
errors = np.mean((y_test - y_pred) ** 2, axis=1)
print(f"Prediction error statistics:")
print(f"Min error: {np.min(errors)}")
print(f"Max error: {np.max(errors)}")
print(f"Mean error: {np.mean(errors)}")
print(f"Error std: {np.std(errors)}")

# 保存预测结果和误差
np.savetxt('predicted_potential.csv', y_pred, delimiter=',')
np.savetxt('prediction_errors.csv', errors, delimiter=',')

print("Predictions saved to 'predicted_potential.csv'")
print("Prediction errors saved to 'prediction_errors.csv'")

# 创建坐标轴 (假设空间维度和时间维度)
# 假设每行数据有100个空间点，共3000个时间步
n_space_points = phi_data.shape[1]
n_time_steps = phi_data.shape[0]

# 创建空间和时间坐标
x = np.linspace(0, 10, n_space_points)  # 假设空间范围是0-10 cm
t = np.linspace(0, 3000, n_time_steps)  # 假设时间范围是0-3000 μs

# 创建网格
X_grid, T_grid = np.meshgrid(x, t)

# 绘制电势(phi)的等高线图
plt.figure(figsize=(10, 8))
plt.contourf(X_grid, T_grid, phi_data.values, levels=50, cmap='rainbow')
plt.colorbar(label='Electric Potential (phi)')
plt.xlabel('Position (cm)')
plt.ylabel('Time (μs)')
plt.title('Electric Potential Distribution')
plt.savefig('plots/phi_contour.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制电荷密度(rho)的等高线图
plt.figure(figsize=(10, 8))
plt.contourf(X_grid, T_grid, rho_data.values, levels=50, cmap='rainbow')
plt.colorbar(label='Charge Density (rho)')
plt.xlabel('Position (cm)')
plt.ylabel('Time (μs)')
plt.title('Charge Density Distribution')
plt.savefig('plots/rho_contour.png', dpi=300, bbox_inches='tight')
plt.close()

print("Contour plots saved to 'plots/phi_contour.png' and 'plots/rho_contour.png'")
