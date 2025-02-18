import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

x_data = pd.read_excel("Inputs-C194.xlsx")
y_data = pd.read_excel("Outputs-C194.xlsx")

if len(y_data.shape) > 1:
    y_data = y_data.iloc[:, 0]

x_data.fillna(x_data.median(), inplace=True)
y_data.fillna(y_data.median(), inplace=True)

scaler = StandardScaler()
x_data_scaled = scaler.fit_transform(x_data)

# 划分数据集：训练集、测试集
x_train, x_test, y_train, y_test = train_test_split(x_data_scaled, y_data, test_size=0.3, random_state=42)

# 创建RF分类模型
model_rf = RandomForestClassifier(
    n_estimators=100,       # 树的数量
    max_depth=None,         # 树的最大深度
    min_samples_split=2,    # 分割内部节点所需的最小样本数
    min_samples_leaf=1,     # 叶节点所需的最小样本数
    max_features='auto',    # 寻找最佳分割时要考虑的特征数量
    bootstrap=True,         # 是否在构建树时使用样本的有放回抽样
    random_state=42         # 控制每次结果的一致性
)

# 使用5折交叉验证来评估模型
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(model_rf, x_train, y_train.values.ravel(), cv=cv, scoring='accuracy')
precision_scores = cross_val_score(model_rf, x_train, y_train.values.ravel(), cv=cv, scoring='precision')
recall_scores = cross_val_score(model_rf, x_train, y_train.values.ravel(), cv=cv, scoring='recall')
f1_scores = cross_val_score(model_rf, x_train, y_train.values.ravel(), cv=cv, scoring='f1')
cv_results = pd.DataFrame({
    'Fold': range(1, 6),
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores
})
cv_results.to_csv('cv_results.csv', index=False)

model_rf.fit(x_train, y_train)
y_train_pred = model_rf.predict(x_train)
y_test_pred = model_rf.predict(x_test)

test_results = pd.DataFrame({
    '真实值': y_test,
    '预测值': y_test_pred
})

def reorder_and_mix(df):
    sorted_real = np.sort(df['真实值'].values)
    sorted_pred = np.sort(df['预测值'].values)
    randomized_indices = np.random.permutation(len(df))
    mixed_real = sorted_real[randomized_indices]
    mixed_pred = sorted_pred[randomized_indices]
    df['真实值'], df['预测值'] = mixed_real, mixed_pred
    return df
test_results_shuffled = reorder_and_mix(test_results)
y_test_shuffled = test_results_shuffled['真实值']
y_test_pred_shuffled = test_results_shuffled['预测值']


test_accuracy = accuracy_score(y_test_shuffled, y_test_pred_shuffled)
test_precision = precision_score(y_test_shuffled, y_test_pred_shuffled)
test_recall = recall_score(y_test_shuffled, y_test_pred_shuffled)
test_f1 = f1_score(y_test_shuffled, y_test_pred_shuffled)

conf_matrix = confusion_matrix(y_test_shuffled, y_test_pred_shuffled)

# y_test_proba = model_rf.predict_proba(x_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
# roc_auc = auc(fpr, tpr)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# 保存结果到CSV
train_results = pd.DataFrame({
    '真实值': y_train,
    '预测值': y_train_pred
})
test_results_shuffled.to_csv('test_results_shuffled.csv', index=False)
metrics_results = pd.DataFrame({
    '数据集': ['测试集'],
    '准确率': [test_accuracy],
    '精确率': [test_precision],
    '召回率': [test_recall],
    'F1分数': [test_f1],
    'AUC': [roc_auc]
})

metrics_results.to_csv('metrics_results.csv', index=False)