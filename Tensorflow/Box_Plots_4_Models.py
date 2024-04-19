import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
labels = ['ResNet18', 'Pre-Trained-ResNet18', 'MobileNetV1', 'LogisticRegression']

resnet18_auc = [0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_balance = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
resnet18_training = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
resnet18_validation = [0.65, 0.65, 1.0, 1.0, 1.0, 1.0]

pre_trained_resnet18_auc = [0.5, 0.0, 0.0, 0.0, 0.0, 1.0]
pre_trained_resnet18_balance = [0.25, 0.25, 0.5, 0.0, 0.0, 0.5]
pre_trained_resnet18_training = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
pre_trained_resnet18_validation = [0.32, 0.32, 0.62, 0.1, 0.1, 0.5]

mobilenetv1_auc = [0.5, 0.5, 1.0, 0.5, 1.0, 1.0]
mobilenetv1_balance = [0.75, 0.75, 0.75, 0.25, 0.5, 1.0]
mobilenetv1_training = [0.6, 1.0, 1.0, 1.0, 1.0, 1.0]
mobilenetv1_validation = [0.3, 0.65, 0.3, 0.3, 0.5, 1.0]

LogisticRegression_auc = [0.5, 0.5, 1.0, 0.5, 0.5, 0.5]
LogisticRegression_balance = [0.5, 0.5, 1.0, 0.5, 0.5, 0.5]
LogisticRegression_training = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
LogisticRegression_validation = [0.65, 0.65, 1.0, 0.65, 0.5, 0.5]


def get_training_dataframe():
    # Combine data into a DataFrame TRAINING
    return pd.DataFrame({
        'Model': ['ResNet18'] * len(resnet18_auc) + ['ResNet18'] * len(resnet18_balance) + ['ResNet18'] * len(resnet18_training) +
                 ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_auc) + ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_balance) + ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_training) +
                 ['MobileNetV1'] * len(mobilenetv1_auc) + ['MobileNetV1'] * len(mobilenetv1_balance) + ['MobileNetV1'] * len(mobilenetv1_training) +
                 ['LogisticRegression'] * len(LogisticRegression_auc) + ['LogisticRegression'] * len(LogisticRegression_balance) + ['LogisticRegression'] * len(LogisticRegression_training),
        'Metric': ['AUC Score'] * len(resnet18_auc) + ['Balanced Accuracy'] * len(resnet18_balance) + ['Training Accuracy'] * len(resnet18_training) +
                  ['AUC Score'] * len(pre_trained_resnet18_auc) + ['Balanced Accuracy'] * len(pre_trained_resnet18_balance) + ['Training Accuracy'] * len(pre_trained_resnet18_training) +
                  ['AUC Score'] * len(mobilenetv1_auc) + ['Balanced Accuracy'] * len(mobilenetv1_balance) + ['Training Accuracy'] * len(mobilenetv1_training) +
                  ['AUC Score'] * len(LogisticRegression_auc) + ['Balanced Accuracy'] * len(LogisticRegression_balance) + ['Training Accuracy'] * len(LogisticRegression_training),
        'Value': resnet18_auc + resnet18_balance + resnet18_training +
                 pre_trained_resnet18_auc + pre_trained_resnet18_balance + pre_trained_resnet18_training +
                 mobilenetv1_auc + mobilenetv1_balance + mobilenetv1_training +
                 LogisticRegression_auc + LogisticRegression_balance + LogisticRegression_training
    })


def get_validation_dataframe():
    # Combine data into a DataFrame VALIDATION
    return pd.DataFrame({
        'Model': ['ResNet18'] * len(resnet18_auc) + ['ResNet18'] * len(resnet18_balance) + ['ResNet18'] * len(resnet18_validation) +
                 ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_auc) + ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_balance) + ['Pre-Trained-ResNet18'] * len(pre_trained_resnet18_validation) +
                 ['MobileNetV1'] * len(mobilenetv1_auc) + ['MobileNetV1'] * len(mobilenetv1_balance) + ['MobileNetV1'] * len(mobilenetv1_validation) +
                 ['LogisticRegression'] * len(LogisticRegression_auc) + ['LogisticRegression'] * len(LogisticRegression_balance) + ['LogisticRegression'] * len(LogisticRegression_validation),
        'Metric': ['AUC Score'] * len(resnet18_auc) + ['Balanced Accuracy'] * len(resnet18_balance) + ['Validation Accuracy'] * len(resnet18_validation) +
                  ['AUC Score'] * len(pre_trained_resnet18_auc) + ['Balanced Accuracy'] * len(pre_trained_resnet18_balance) + ['Validation Accuracy'] * len(pre_trained_resnet18_validation) +
                  ['AUC Score'] * len(mobilenetv1_auc) + ['Balanced Accuracy'] * len(mobilenetv1_balance) + ['Validation Accuracy'] * len(mobilenetv1_validation) +
                  ['AUC Score'] * len(LogisticRegression_auc) + ['Balanced Accuracy'] * len(LogisticRegression_balance) + ['Validation Accuracy'] * len(LogisticRegression_validation),
        'Value': resnet18_auc + resnet18_balance + resnet18_validation +
                 pre_trained_resnet18_auc + pre_trained_resnet18_balance + pre_trained_resnet18_validation +
                 mobilenetv1_auc + mobilenetv1_balance + mobilenetv1_validation +
                 LogisticRegression_auc + LogisticRegression_balance + LogisticRegression_validation
    })


# data = get_training_dataframe()
data = get_validation_dataframe()

# Create box plots using Seaborn
plt.figure(figsize=(12, 8))
box_plot = sns.boxplot(x='Model', y='Value', hue='Metric', data=data, palette="Set3", width=0.9, dodge=True, gap=0.1)

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


# plt.title('AUC, Balance, and Training for ResNet18, Pre-Trained-ResNet18, MobileNetV1, and LogisticRegression (6 samples)')
# plt.title('AUC, Balance, and Validation for ResNet18, Pre-Trained-ResNet18, MobileNetV1, and LogisticRegression (6 samples)')


# Set font size for labels and ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Value', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=10)

# Add gridlines for better visualization
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

# Adjust legend position and font size
plt.legend(loc='lower right', fontsize=12)

plt.show()
