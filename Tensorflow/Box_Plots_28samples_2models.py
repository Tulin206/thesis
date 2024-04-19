import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
labels = ['ResNet18', 'Pre-Trained-ResNet18']

# resnet18_auc = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_balance = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
# resnet18_training = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_validation = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]

# pre_trained_resnet18_auc = [0.5, 0.0, 0.0, 0.0, 0.0, 1.0]
pre_trained_resnet18_balance = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
# pre_trained_resnet18_training = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
pre_trained_resnet18_validation = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]

# Check and ensure all arrays have the same length
# assert len(resnet18_auc) == len(resnet18_balance) == len(resnet18_training) == len(mobilenetv1_auc) == len(mobilenetv1_balance) == len(mobilenetv1_training), "Arrays must have the same length"

# # Combine data into a DataFrame
# data = pd.DataFrame({
#     'Model': ['ResNet18'] * len(resnet18_balance) + ['ResNet18'] * len(resnet18_training) +
#              ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_balance) + ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_training),
#     'Metric': ['Balance'] * len(resnet18_balance) + ['Training'] * len(resnet18_training) +
#               ['Balance'] * len(pre_trained_resnet18_balance) + ['Training'] * len(pre_trained_resnet18_training),
#     'Value': resnet18_balance + resnet18_training +
#              pre_trained_resnet18_balance + pre_trained_resnet18_training
# })

# Combine data into a DataFrame
data = pd.DataFrame({
    'Model': ['ResNet18'] * len(resnet18_balance) + ['ResNet18'] * len(resnet18_validation) +
             ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_balance) + ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_validation),
    'Metric': ['Balanced accuracy'] * len(resnet18_balance) + ['validation accuracy'] * len(resnet18_validation) +
              ['Balanced accuracy'] * len(pre_trained_resnet18_balance) + ['validation accuracy'] * len(pre_trained_resnet18_validation),
    'Value': resnet18_balance + resnet18_validation +
             pre_trained_resnet18_balance + pre_trained_resnet18_validation
})

# Create box plots using Seaborn
plt.figure(figsize=(12, 8))
box_plot = sns.boxplot(x='Model', y='Value', hue='Metric', data=data, palette="Set3", width=0.9)

# Add swarm plot
sns.swarmplot(x='Model', y='Value', data=data, size=5, alpha=0.75, palette="Set3", dodge=True, hue='Metric', legend=False)

# Explicitly plot mean values
mean_values = data.groupby(['Model', 'Metric'])['Value'].mean().reset_index()

# # Calculate x-coordinate for each mean value
# for i, model in enumerate(labels):
#     for metric in ['Balance', 'Training']:
#         mean_value = mean_values[(mean_values['Model'] == model) & (mean_values['Metric'] == metric)]['Value'].values[0]
#         if metric == 'AUC':
#           hue_offset = i - 0.27
#         elif metric == 'Balance':
#           hue_offset = i
#         else:
#           hue_offset = i + 0.27
#
#         plt.text(hue_offset, mean_value, f'{mean_value:.2f}', ha="center", va="bottom")
#
# plt.title('Box Plots of AUC, Balance, and Training for ResNet18 and MobileNetV1')
# plt.ylabel('Value')
# plt.show()

# Set font size for labels and ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Model', fontsize=16)
plt.ylabel('Value', fontsize=16)

# Rotate x-axis labels for better readability
plt.xticks(rotation=10)

# Add gridlines for better visualization
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

# Adjust legend position and font size
plt.legend(loc='lower center', fontsize=16, bbox_to_anchor=(0.5, -0.25), ncol=3)  # move legend outside with 3 columns
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

