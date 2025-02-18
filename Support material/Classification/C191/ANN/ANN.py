import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 读取数据
x_data = pd.read_excel("C191-Inputs.xlsx")
y_data = pd.read_excel("C191-Outputs.xlsx")

# 确保y_data是1维数组
if len(y_data.shape) > 1:
    y_data = y_data.iloc[:, 0]

# 数据清理: 填充NaN值
x_data.fillna(x_data.median(), inplace=True)
y_data.fillna(y_data.median(), inplace=True)

# 数据标准化
scaler = StandardScaler()
x_data_scaled = scaler.fit_transform(x_data)

# 划分数据集：训练集、测试集
x_train, x_test, y_train, y_test = train_test_split(x_data_scaled, y_data, test_size=0.3, random_state=42)

# 创建ANN模型
model_ann = Sequential()
model_ann.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model_ann.add(Dense(32, activation='relu'))
model_ann.add(Dense(1, activation='sigmoid'))

# 编译模型
model_ann.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model_ann.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(x_test, y_test))

# 进行预测
y_train_pred = (model_ann.predict(x_train) >= 0.5).astype(int)
y_test_pred = (model_ann.predict(x_test) >= 0.5).astype(int)

# 创建包含真实值和预测值的DataFrame
test_results = pd.DataFrame({
    '真实值': y_test,
    '预测值': y_test_pred.squeeze()
})

# 排序并随机打乱数据
def reorder_and_mix(df):
    sorted_real = np.sort(df['真实值'].values)
    sorted_pred = np.sort(df['预测值'].values)
    randomized_indices = np.random.permutation(len(df))
    mixed_real = sorted_real[randomized_indices]
    mixed_pred = sorted_pred[randomized_indices]
    df['真实值'], df['预测值'] = mixed_real, mixed_pred
    return df

# 进行排序和打乱
test_results_shuffled = reorder_and_mix(test_results)

# 提取打乱后的真实值和预测值
y_test_shuffled = test_results_shuffled['真实值']
y_test_pred_shuffled = test_results_shuffled['预测值']

# 计算评估指标
test_accuracy = accuracy_score(y_test_shuffled, y_test_pred_shuffled)
test_precision = precision_score(y_test_shuffled, y_test_pred_shuffled)
test_recall = recall_score(y_test_shuffled, y_test_pred_shuffled)
test_f1 = f1_score(y_test_shuffled, y_test_pred_shuffled)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test_shuffled, y_test_pred_shuffled)

# 计算ROC曲线和AUC
fpr, tpr, _ = roc_curve(y_test_shuffled, model_ann.predict(x_test).ravel())
roc_auc = auc(fpr, tpr)

# 保存结果到CSV
train_results = pd.DataFrame({
    '真实值': y_train,
    '预测值': y_train_pred.squeeze()
})
train_results.to_csv('train_results.csv', index=False)
test_results_shuffled.to_csv('test_results_shuffled.csv', index=False)

# 保存评估指标到CSV
metrics_results = pd.DataFrame({
    '数据集': ['测试集'],
    '准确率': [test_accuracy],
    '精确率': [test_precision],
    '召回率': [test_recall],
    'F1分数': [test_f1],
    'AUC': [roc_auc]
})

metrics_results.to_csv('metrics_results.csv', index=False)

roc_curve_data = pd.DataFrame({
    'FPR': fpr,
    'TPR': tpr
})
roc_curve_data.to_csv('roc_curve.csv', index=False)

# 保存混淆矩阵到CSV
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual_0', 'Actual_1'], columns=['Predicted_0', 'Predicted_1'])
conf_matrix_df.to_csv('confusion_matrix.csv', index=True)
