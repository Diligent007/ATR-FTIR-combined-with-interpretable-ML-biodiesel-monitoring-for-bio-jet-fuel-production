import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

x_data = pd.read_excel("Inputs-C194.xlsx")
y_data = pd.read_excel("Outputs-C194.xlsx")

if len(y_data.shape) > 1:
    y_data = y_data.iloc[:, 0]

x_data.fillna(x_data.median(), inplace=True)
y_data.fillna(y_data.median(), inplace=True)

scaler = StandardScaler()
x_data_scaled = scaler.fit_transform(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data_scaled, y_data, test_size=0.3, random_state=42)

model_ann = Sequential()
model_ann.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
model_ann.add(Dense(32, activation='relu'))
model_ann.add(Dense(1, activation='sigmoid'))

model_ann.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model_ann.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(x_test, y_test))

y_train_pred = (model_ann.predict(x_train) >= 0.5).astype(int)
y_test_pred = (model_ann.predict(x_test) >= 0.5).astype(int)

test_results = pd.DataFrame({
    '真实值': y_test,
    '预测值': y_test_pred.squeeze()
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

fpr, tpr, _ = roc_curve(y_test_shuffled, model_ann.predict(x_test).ravel())
roc_auc = auc(fpr, tpr)

train_results = pd.DataFrame({
    '真实值': y_train,
    '预测值': y_train_pred.squeeze()
})
train_results.to_csv('train_results.csv', index=False)
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