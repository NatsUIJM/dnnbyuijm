import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import os

# 创建输出目录
output_dir = "output-pinns"
os.makedirs(output_dir, exist_ok=True)

def load_data_in_chunks(filename, chunk_size=100):
    """分块读取数据以节省内存"""
    data_chunks = []
    with open(filename, 'r') as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line)
            if (i + 1) % chunk_size == 0:
                chunk = np.loadtxt(lines, delimiter=',')
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)
                data_chunks.append(chunk)
                lines = []
        
        # 处理剩余的行
        if lines:
            chunk = np.loadtxt(lines, delimiter=',')
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            data_chunks.append(chunk)
    
    return np.vstack(data_chunks)

# 分块读取数据
print("正在读取 phi_700.csv...")
phi_data = load_data_in_chunks("phi_700.csv")
print(f"phi_data 形状: {phi_data.shape}")

print("正在读取 rho_700.csv...")
rho_data = load_data_in_chunks("rho_700.csv")
print(f"rho_data 形状: {rho_data.shape}")

# 使用较小的分辨率以减少内存使用
num_timesteps, num_x_points = phi_data.shape
sample_factor = max(1, num_timesteps // 1000)  # 最多保留1000行
phi_data_sampled = phi_data[::sample_factor, :]
rho_data_sampled = rho_data[::sample_factor, :]
print(f"采样后数据形状: {phi_data_sampled.shape}")

# 构造 x 和 t 的坐标轴
x = np.linspace(0, 1, phi_data_sampled.shape[1])  # 假设长度为 1 cm
t = np.linspace(0, 3, phi_data_sampled.shape[0])  # 假设时间为 3 μs

# 创建插值函数
phi_interpolator = RectBivariateSpline(t, x, phi_data_sampled)
rho_interpolator = RectBivariateSpline(t, x, rho_data_sampled)

# 计算电场 E = -dφ/dx
def compute_e_field(t_vals, x_vals):
    return -phi_interpolator(t_vals, x_vals, dx=1)

# 生成用于绘图的网格（使用适中的分辨率）
x_plot = np.linspace(0, 1, 500)
t_plot = np.linspace(0, 3, 500)
X_plot, T_plot = np.meshgrid(x_plot, t_plot)

# 分块计算以避免内存问题
def compute_field_in_chunks(interpolator, T, X, dx=0):
    result = np.empty_like(T)
    chunk_size = 50  # 每次处理50行
    
    for i in range(0, T.shape[0], chunk_size):
        end_i = min(i + chunk_size, T.shape[0])
        result[i:end_i, :] = interpolator(T[i:end_i, :], X[i:end_i, :], dx=dx, grid=False)
    
    return result

print("正在计算电势场...")
phi_plot = compute_field_in_chunks(phi_interpolator, T_plot, X_plot)
print("正在计算电荷密度场...")
rho_plot = compute_field_in_chunks(rho_interpolator, T_plot, X_plot)
print("正在计算电场...")
E_plot = compute_field_in_chunks(phi_interpolator, T_plot, X_plot, dx=1)
E_plot = -E_plot  # E = -dφ/dx

# 绘图函数
def plot_field(X, T, Z, title, filename):
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, T, Z, levels=100, cmap='viridis')
    plt.colorbar(contour, label=title)
    plt.xlabel("x (cm)")
    plt.ylabel("t (μs)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# 绘制并保存三个场的等高线图
print("正在生成图表...")
plot_field(X_plot, T_plot, rho_plot, "Charge Density ρ(x,t)", "rho-PINNS.png")
plot_field(X_plot, T_plot, phi_plot, "Electric Potential φ(x,t)", "phi-PINNS.png")
plot_field(X_plot, T_plot, E_plot, "Electric Field E(x,t)", "E-PINNS.png")

print("绘图完成，文件已保存到 output 文件夹。")
