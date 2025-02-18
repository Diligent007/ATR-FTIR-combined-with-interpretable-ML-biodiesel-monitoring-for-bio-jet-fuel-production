
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

# 加载训练好的模型
model = joblib.load('RF_model.pkl')

# 加载输入特征数据
inputs_df = pd.read_excel('C192-Inputs.xlsx')

# 处理NaN值和无穷大值
inputs_df.fillna(inputs_df.median(), inplace=True)  # 使用中位数填充NaN值
inputs_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 将无限值替换为NaN
inputs_df.dropna(inplace=True)  # 删除仍有NaN的行


# 数据归一化
scaler = MinMaxScaler()
inputs_df_scaled = pd.DataFrame(scaler.fit_transform(inputs_df), columns=inputs_df.columns)

# 设置Time和Temperature的索引
N_index = 2  # 第9列，基于0开始的索引
VM_index = 6  # 第10列，基于0开始的索引


# 创建时间和温度的网格数据
N_range = np.linspace(inputs_df_scaled.iloc[:, N_index].min(), inputs_df_scaled.iloc[:, N_index].max(), 50)
VM_range = np.linspace(inputs_df_scaled.iloc[:, VM_index].min(), inputs_df_scaled.iloc[:, VM_index].max(), 50)
N_grid, VM_grid = np.meshgrid(N_range, VM_range)

# 准备预测数据集，正确设置复制次数
num_repeats = len(N_grid.ravel())
pred_data = inputs_df_scaled.loc[np.zeros(num_repeats, dtype=int)].copy()  # 复制第一行足够次数
pred_data.iloc[:, N_index] = np.tile(N_grid.ravel(), 1)  # 设置Time值
pred_data.iloc[:, VM_index] = np.tile(VM_grid.ravel(), 1)  # 设置Temperature值

# 使用模型进行预测
predictions = model.predict(pred_data).reshape(N_grid.shape)

# 绘制3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制表面图
surf = ax.plot_surface(N_grid, VM_grid, predictions, cmap=plt.cm.viridis, edgecolor='none')


# 设置坐标轴标签，加粗并指定字体大小
ax.set_xlabel('881', fontsize=12, fontweight='bold')
ax.set_ylabel('1233', fontsize=12, fontweight='bold')
ax.set_zlabel('Predicted Adsorption', fontsize=12, fontweight='bold')
ax.set_title('3D Partial Dependence Plot', fontsize=14, fontweight='bold')

# 保存图形到PNG文件
plt.savefig('3D_Partial_Dependence_Plot-try.png')

# 显示图形
plt.show()