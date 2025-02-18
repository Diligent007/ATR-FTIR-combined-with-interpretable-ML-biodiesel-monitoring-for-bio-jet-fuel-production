import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the global font size and family for all text elements in the plot
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_context("notebook", font_scale=1.5)  # Adjusts the size of the labels, ticks, and title

# Load the SHAP values data from the CSV file
shap_data = pd.read_csv('shap_values.csv')

# Calculate the mean absolute SHAP value for each feature
shap_summary = shap_data.abs().mean().sort_values(ascending=False).reset_index()
shap_summary.columns = ['Feature', 'Mean Absolute SHAP Value']
shap_summary['Mean Absolute SHAP Value'] *= 1000  # Scale the values by 1000

# Plot the summary as a bar chart
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Mean Absolute SHAP Value', y='Feature', data=shap_summary, palette='viridis')
plt.title('Mean Absolute SHAP Values per Feature', fontsize=16)  # Set title font size
plt.xlabel('Mean Absolute SHAP Value', fontsize=14)  # Set x-axis label font size
plt.ylabel('Features', fontsize=14)  # Set y-axis label font size

# Optional: Customize tick size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplots_adjust(left=0.2)  # 增加下边界的距离
# Save the plot as a jpg image file
plt.savefig('mean_absolute_shap_values-3.jpg', format='jpg', dpi=150)
plt.close()  # Close the figure to prevent it from displaying in the notebook/output

# Save the mean absolute SHAP value summary to a CSV file
shap_summary.to_csv('mean_absolute_shap_values-3.csv', index=False)
