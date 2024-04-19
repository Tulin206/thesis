import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
labels = ['ResNet18_8-fold-CV', 'ResNet18_11-fold-CV']

resnet18_auc_8 = [0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_balance_8 = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_training_8 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_validation_8 = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

resnet18_auc_11 = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_balance_11 = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_training_11 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_validation_11 = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def get_training_dataframe():
    # Combine data into a DataFrame TRAINING
    return pd.DataFrame({
        'Model': ['ResNet18_8-fold-CV'] * len(resnet18_auc_8) + ['ResNet18_8-fold-CV'] * len(resnet18_balance_8) + ['ResNet18_8-fold-CV'] * len(resnet18_training_8) +
                 ['ResNet18_11-fold-CV'] * len(resnet18_auc_11) + ['ResNet18_11-fold-CV'] * len(resnet18_balance_11) + ['ResNet18_11-fold-CV'] * len(resnet18_training_11),
        'Metric': ['AUC Score'] * len(resnet18_auc_8) + ['Balanced Accuracy'] * len(resnet18_balance_8) + ['Training Accuracy'] * len(resnet18_training_8) +
                  ['AUC Score'] * len(resnet18_auc_11) + ['Balanced Accuracy'] * len(resnet18_balance_11) + ['Training Accuracy'] * len(resnet18_training_11),
        'Value': resnet18_auc_8 + resnet18_balance_8 + resnet18_training_8 +
                 resnet18_auc_11 + resnet18_balance_11 + resnet18_training_11
    })


def get_validation_dataframe():
    # Combine data into a DataFrame VALIDATION
    return pd.DataFrame({
        'Model': ['ResNet18_8-fold-CV'] * len(resnet18_auc_8) + ['ResNet18_8-fold-CV'] * len(resnet18_balance_8) + ['ResNet18_8-fold-CV'] * len(resnet18_validation_8) +
                 ['ResNet18_11-fold-CV'] * len(resnet18_auc_11) + ['ResNet18_11-fold-CV'] * len(resnet18_balance_11) + ['ResNet18_11-fold-CV'] * len(resnet18_validation_11),
        'Metric': ['AUC Score'] * len(resnet18_auc_8) + ['Balanced Accuracy'] * len(resnet18_balance_8) + ['validation Accuracy'] * len(resnet18_validation_8) +
                  ['AUC Score'] * len(resnet18_auc_11) + ['Balanced Accuracy'] * len(resnet18_balance_11) + ['validation Accuracy'] * len(resnet18_validation_11),
        'Value': resnet18_auc_8 + resnet18_balance_8 + resnet18_validation_8 +
                 resnet18_auc_11 + resnet18_balance_11 + resnet18_validation_11
    })


# data = get_training_dataframe()
data = get_validation_dataframe()

# Create box plots using Seaborn
plt.figure(figsize=(12, 8))
box_plot = sns.boxplot(x='Model', y='Value', hue='Metric', data=data, palette="Set3", width=0.9)

# Add swarm plot
sns.swarmplot(x='Model', y='Value', data=data, size=5, alpha=0.75, palette="Set3", dodge=True, hue='Metric', legend=False)


def mean_value_plotting(data, labels, plt):
    # Explicitly plot mean values
    mean_values = data.groupby(['Model', 'Metric'])['Value'].mean().reset_index()
    # Calculate x-coordinate for each mean value
    for i, model in enumerate(labels):
        # for metric in ['AUC Score', 'Balanced Accuracy', 'Training Accuracy']:
        for metric in ['AUC Score', 'Balanced Accuracy', 'Validation Accuracy']:
            mean_value = mean_values[(mean_values['Model'] == model) & (mean_values['Metric'] == metric)]['Value'].values[0]
            if metric == 'AUC Score':
                hue_offset = i - 0.27
            elif metric == 'Balanced Accuracy':
                hue_offset = i
            else:
                hue_offset = i + 0.27

            plt.text(hue_offset, mean_value, f'{mean_value:.2f}', ha="center", va="bottom")


# plt.title('AUC, Balance, and Training for ResNet18 and MobileNetV1 (8 samples)')
# plt.title('AUC, Balance, and Validation for ResNet18 and MobileNetV1 (8 samples)')


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
