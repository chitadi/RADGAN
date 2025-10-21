import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import safe_open_yaml, stringify

CONFIG_FILE = "/src/config/train_baseline.yaml"

config = safe_open_yaml(CONFIG_FILE)
version = 10

model_name = config["model"]
model_params = config["model_params"]
model_params_str = stringify(model_params, delimiter="_")
save_dir = f"logs/{model_name}_{model_params_str}/version_{version}"
os.makedirs(save_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(f'logs/{model_name}_{model_params_str}/version_{version}/metrics.csv')

# Create figure and axis
plt.figure(figsize=(10, 6))

# Get validation losses (one per epoch)
val_data = df[df['val/loss'].notna()]
completed_epochs = val_data['epoch'].unique()

# Calculate mean training loss only for completed epochs
train_losses = []
for epoch in completed_epochs:
    epoch_data = df[df['epoch'] == epoch]
    train_loss = epoch_data['train/loss'].mean()
    train_losses.append(train_loss)

val_losses = val_data['val/loss'].values
epochs = range(len(completed_epochs))

# Plot
plt.plot(epochs, train_losses, label='Training Loss (mean per epoch)', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='s')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(f'{save_dir}/loss_plot.png')
plt.close()