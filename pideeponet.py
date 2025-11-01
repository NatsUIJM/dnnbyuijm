import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import time

# Create output directory
os.makedirs('PI-DeepONet', exist_ok=True)

# Load data
print("Loading data...")
rho_data = np.loadtxt('rho_700.csv', delimiter=',')
phi_data = np.loadtxt('phi_700.csv', delimiter=',')

print(f"rho_data shape: {rho_data.shape}")
print(f"phi_data shape: {phi_data.shape}")
print(f"rho_data range: [{rho_data.min():.2e}, {rho_data.max():.2e}]")
print(f"phi_data range: [{phi_data.min():.2f}, {phi_data.max():.2f}]")

# Parameters
n_samples, n_points = rho_data.shape
d = 0.07  # 7cm = 0.07m
total_time = 3.0  # 假设总时间为3微秒

# 数据标准化
rho_mean = rho_data.mean()
rho_std = rho_data.std()
phi_mean = phi_data.mean()
phi_std = phi_data.std()

rho_data_normalized = (rho_data - rho_mean) / rho_std
phi_data_normalized = (phi_data - phi_mean) / phi_std

print(f"Normalized rho range: [{rho_data_normalized.min():.2f}, {rho_data_normalized.max():.2f}]")
print(f"Normalized phi range: [{phi_data_normalized.min():.2f}, {phi_data_normalized.max():.2f}]")

# Convert to tensors
rho_tensor = torch.FloatTensor(rho_data_normalized)
phi_tensor = torch.FloatTensor(phi_data_normalized)

# Split data
rho_train, rho_test, phi_train, phi_test = train_test_split(
    rho_tensor, phi_tensor, test_size=0.2, random_state=42
)

print(f"Training samples: {rho_train.shape[0]}")
print(f"Test samples: {rho_test.shape[0]}")

# DeepONet components with better initialization
class BranchNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BranchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # Xavier初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class TrunkNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TrunkNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # Xavier初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class PI_DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net, output_dim=1):
        super(PI_DeepONet, self).__init__()
        self.branch = branch_net
        self.trunk = trunk_net
        self.output_dim = output_dim
        
    def forward(self, rho, x_coord):
        b = self.branch(rho)
        t = self.trunk(x_coord)
        # Dot product
        out = torch.sum(b * t, dim=1, keepdim=True)
        return out

# Initialize networks
branch_net = BranchNet(n_points, 128)
trunk_net = TrunkNet(1, 128)
model = PI_DeepONet(branch_net, trunk_net, output_dim=1)

# 使用学习率调度器 (移除verbose参数)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

# Training function with gradient clipping
def train_model(model, rho_train, phi_train, epochs=300):
    batch_size = 64
    
    # Create spatial coordinate points (normalized to [0,1])
    x_points = torch.linspace(0, 1, n_points).view(-1, 1)
    
    # Progress bar
    epoch_pbar = tqdm(range(epochs), desc='Training Progress')
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle training data
        perm = torch.randperm(rho_train.size(0))
        rho_shuffled = rho_train[perm]
        phi_shuffled = phi_train[perm]
        
        for i in range(0, rho_train.size(0), batch_size):
            rho_batch = rho_shuffled[i:i+batch_size]
            phi_batch = phi_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Data loss
            x_batch = x_points.repeat(rho_batch.shape[0], 1).view(-1, 1)
            rho_repeated = rho_batch.repeat_interleave(n_points, dim=0)
            phi_pred = model(rho_repeated, x_batch)
            phi_true = phi_batch.view(-1, 1)
            data_loss = torch.mean((phi_pred - phi_true) ** 2)
            
            # Backward pass with gradient clipping
            data_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += data_loss.item()
            n_batches += 1
            
        # Average loss
        avg_loss = epoch_loss / n_batches
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Update progress bar
        epoch_pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'PI-DeepONet/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
    # Load best model
    model.load_state_dict(torch.load('PI-DeepONet/best_model.pth'))
    print("Training completed!")

# Train the model
print("Training PI-DeepONet...")
train_start_time = time.time()
train_model(model, rho_train, phi_train, epochs=300)
train_end_time = time.time()
print(f"Training completed in {train_end_time - train_start_time:.2f} seconds")

# Evaluation and visualization
print("Generating predictions...")
model.eval()

# Generate predictions for test set
n_viz_samples = min(500, len(rho_test))  # Limit for visualization
rho_viz = rho_test[:n_viz_samples]
phi_viz_true = phi_test[:n_viz_samples]

# Generate predictions
x_plot = torch.linspace(0, 1, n_points).view(-1, 1)  # Normalized coordinates
phi_viz_pred = []

pred_start_time = time.time()
with torch.no_grad():
    for i in tqdm(range(n_viz_samples), desc='Generating predictions'):
        rho_sample = rho_viz[i:i+1]
        rho_repeated = rho_sample.repeat(n_points, 1)
        phi_pred = model(rho_repeated, x_plot)
        phi_viz_pred.append(phi_pred.numpy().flatten())
pred_end_time = time.time()
print(f"Prediction completed in {pred_end_time - pred_start_time:.4f} seconds")

# Convert to numpy arrays and denormalize
phi_viz_pred = np.array(phi_viz_pred) * phi_std + phi_mean
phi_viz_true = phi_viz_true.numpy() * phi_std + phi_mean

# Create time axis for visualization
t_viz = np.linspace(0, total_time * n_viz_samples / len(rho_test), n_viz_samples)

# Plot 1: Training data (charge density) - Heatmap
plt.figure(figsize=(12, 8))
# Use subset of data for better visualization
subset_indices = np.linspace(0, len(rho_data)-1, min(500, len(rho_data))).astype(int)
plt.imshow(rho_data[subset_indices, :].T, aspect='auto', origin='lower', 
           extent=[0, total_time * len(subset_indices) / len(rho_data), 0, d*100], 
           cmap='viridis')
plt.colorbar(label='Charge Density ρ(x,t) (C/m³)')
plt.xlabel('Time t (μs)')
plt.ylabel('Position x (cm)')
plt.title('Training Data: ρ(x,t) by Fluid Model')
plt.tight_layout()
plt.savefig('PI-DeepONet/rho_training.png', dpi=300)
plt.close()

# Plot 2: True solution (potential from fluid model) - Heatmap
plt.figure(figsize=(12, 8))
plt.imshow(phi_viz_true.T, aspect='auto', origin='lower', 
           extent=[0, total_time * n_viz_samples / len(rho_test), 0, d*100], 
           cmap='plasma')
plt.colorbar(label='Potential φ(x,t) (V)')
plt.xlabel('Time t (μs)')
plt.ylabel('Position x (cm)')
plt.title('Ground Truth: φ(x,t) by Fluid Model')
plt.tight_layout()
plt.savefig('PI-DeepONet/phi_ground_truth.png', dpi=300)
plt.close()

# Plot 3: PI-DeepONet prediction - Heatmap
plt.figure(figsize=(12, 8))
plt.imshow(phi_viz_pred.T, aspect='auto', origin='lower', 
           extent=[0, total_time * n_viz_samples / len(rho_test), 0, d*100], 
           cmap='plasma')
plt.colorbar(label='Potential φ(x,t) (V)')
plt.xlabel('Time t (μs)')
plt.ylabel('Position x (cm)')
plt.title('Prediction: φ(x,t) by PI-DeepONet')
plt.tight_layout()
plt.savefig('PI-DeepONet/phi_prediction.png', dpi=300)
plt.close()

# Plot 4: L2 error - Heatmap
error = np.abs(phi_viz_pred - phi_viz_true)
plt.figure(figsize=(12, 8))
plt.imshow(error.T, aspect='auto', origin='lower', 
           extent=[0, total_time * n_viz_samples / len(rho_test), 0, d*100], 
           cmap='Reds')
plt.colorbar(label='L2 Error (V)')
plt.xlabel('Time t (μs)')
plt.ylabel('Position x (cm)')
plt.title('Prediction Error: φ(x,t) L2 Error')
plt.tight_layout()
plt.savefig('PI-DeepONet/phi_error.png', dpi=300)
plt.close()

print("Results saved to PI-DeepONet folder")
