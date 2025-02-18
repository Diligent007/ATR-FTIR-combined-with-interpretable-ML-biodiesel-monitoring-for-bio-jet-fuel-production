import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15  # 设置基本字体大小

# 加载数据
x_data = pd.read_excel("Inputs-C194.xlsx")
y_data = pd.read_excel("Outputs-C194.xlsx")

# 确保数据为数值类型
x_data = x_data.astype(float)
y_data = y_data.astype(float)

# 尝试使用 joblib 加载保存的融合模型
try:
    ensemble_bytePair = joblib.load('融合模型.pkl')
except Exception as e:
    print("加载模型时发生错误：", str(e))
    ensemble_bytePair = None

if ensemble_bytePair:
    # 使用 Shap 计算解释值
    explainer = shap.KernelExplainer(ensemble_bytePair.predict, shap.sample(x_data, 100))
    shap_values = explainer.shap_values(x_data)

    columns = ["530", "720", "881", "1020", "1080", "1157",
    "1233", "1363", "1458", "1741", "2852", "2923", "3007", "3318"]

    # 将 SHAP values 转换为 DataFrame
    shap_df = pd.DataFrame(shap_values, columns=columns)
    shap_df *= 0.01  # 根据需求调整 SHAP 值
    shap_df.to_csv('shap_values.csv', index=False)  # 保存到 CSV 文件
    print("SHAP values have been calculated and saved to 'shap_values.csv'.")

    # 绘制 SHAP 图
    plt.figure(figsize=(15, 10))
    shap_values *= 1  # 根据需求调整 SHAP 值用于图形显示
    shap.summary_plot(shap_values, x_data, feature_names=columns, show=False)

    plt.xlabel('SHAP value (impact on model output)')
    plt.ylabel('Features')
    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig('C171-SHAP.png', bbox_inches='tight')  # 保存图像
    plt.show()

else:
    print("由于模型未加载成功，无法计算 SHAP values。")
