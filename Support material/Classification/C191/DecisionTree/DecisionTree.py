import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

# 读取数据
x_data = pd.read_excel("C191-Inputs.xlsx")
y_data = pd.read_excel("C191-Outputs.xlsx")

# 确保y_data是1维数组
if y_data.ndim > 1:
    y_data = y_data.iloc[:, 0]

# 数据清理: 填充NaN值
x_data.fillna(x_data.median(), inplace=True)
y_data.fillna(y_data.median(), inplace=True)

# 数据标准化
scaler = StandardScaler()
x_data_scaled = scaler.fit_transform(x_data)

# 划分数据集：训练集、验证集、测试集
x_train, x_temp, y_train, y_temp = train_test_split(x_data_scaled, y_data.values.ravel(), test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 创建决策树分类模型
model_dt = DecisionTreeClassifier(random_state=42)

# 使用5折交叉验证来评估模型
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(model_dt, x_train, y_train, cv=cv, scoring='accuracy')

# 输出交叉验证结果
cv_results = pd.DataFrame({
    'Fold': range(1, 6),
    'Accuracy': accuracy_scores
})
cv_results.to_csv('cv_results.csv', index=False)

# 在整个训练集上训练模型并进行预测
model_dt.fit(x_train, y_train)
y_train_pred = model_dt.predict(x_train)
y_test_pred = model_dt.predict(x_test)

# 创建包含真实值和预测值的DataFrame
test_results = pd.DataFrame({
    '真实值': y_test,
    '预测值': y_test_pred
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
fpr, tpr, _ = roc_curve(y_test_shuffled, model_dt.predict_proba(x_test)[:, 1])
roc_auc = auc(fpr, tpr)

# 保存结果到CSV
train_results = pd.DataFrame({
    '真实值': y_train,
    '预测值': y_train_pred
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
