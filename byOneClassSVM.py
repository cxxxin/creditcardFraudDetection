from tkinter.messagebox import NO
import pandas as pd
import numpy as np 
from sklearn import utils  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split  
from sklearn import svm
from sklearn import metrics  
from sklearn.utils import shuffle
import seaborn as sns
from scipy.stats import zscore
from scipy import stats
from numpy import percentile


# load dataset
Normal = pd.read_csv("Normal_for_SVM.csv", low_memory=False)
Fraud = pd.read_csv("Fraud_for_SVM.csv",low_memory=False)
# df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
Normal = Normal.drop(['V5','V8','V9','V13','V15','V20','V21','V22','V23','V24','V25','V26','V27','V28'], axis =1)
Fraud = Fraud.drop(['V5','V8','V9','V13','V15','V20','V21','V22','V23','V24','V25','V26','V27','V28'], axis =1)
features = Normal.columns.values # features是特征的名字

# 与顺序无关
# df.drop('Time',axis=1)	

# 合并数据集
df = pd.concat([Normal,Fraud],axis=0)

# 归一化
for feature in features:
	if feature == 'Fraud' or feature == 'Normal':
		continue
	mean, std = df[feature].mean(), df[feature].std()
	Normal.loc[:, feature] = (Normal[feature] - mean) / std  # 标准化
	Fraud.loc[:, feature] = (Fraud[feature] - mean) / std
	# min = df.loc[:, feature].min()
	# max = df.loc[:, feature].max()
	# Normal.loc[:, feature] = (Normal.loc[:, feature]-  min) / (max - min) 
	# Fraud.loc[:, feature] = (Fraud.loc[:, feature]-  min) / (max - min) 


# 从欺诈交易样本中采样75%的样本作为训练样本
X_train = Normal.sample(frac=0.75)

# 从正常交易样本中采样75%的样本作为训练样本
# X_train = pd.concat([X_train, Normal.sample(frac = 0.75)], axis = 0)

#X_test contains all the transaction not in X_train.
X_test = Normal.loc[~Normal.index.isin(X_train.index)]
X_test = pd.concat([X_test,Fraud],axis=0)

#Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(X_test)

#Add our target features to y_train and y_test.
y_train = X_train.Normal

# y_train = pd.concat([y_train, X_train.Normal], axis=1)

y_test = X_test.Normal
# y_test = pd.concat([y_test, X_test.Normal], axis=1)

# 从特征集中删除类别标签
X_train = X_train.drop(['Fraud','Normal'], axis = 1)
X_test = X_test.drop(['Fraud','Normal'], axis = 1)


inputX = X_train.values
inputY = y_train.values

inputX_test = X_test.values
inputY_test = y_test.values

# nu = 0.1
nu = len(Fraud)/(len(Fraud)+len(Normal))
model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=0.00005)  
model.fit(inputX) 


preds = model.predict(inputX)  
targs = inputY

print("accuracy: ", metrics.accuracy_score(targs, preds))  
print("precision: ", metrics.precision_score(targs, preds))  
print("recall: ", metrics.recall_score(targs, preds))  
print("f1: ", metrics.f1_score(targs, preds))  
# print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))



preds = model.predict(inputX_test)  
targs = inputY_test

print("accuracy: ", metrics.accuracy_score(targs, preds))  
print("precision: ", metrics.precision_score(targs, preds))  
print("recall: ", metrics.recall_score(targs, preds))  
print("f1: ", metrics.f1_score(targs, preds))  
print("area under curve (auc): ", metrics.roc_auc_score(targs, preds)) 
confusion_matrix = confusion_matrix(targs,preds)
print(confusion_matrix)  #输出分类混淆矩阵

# 绘制混淆矩阵
classes = ['Fraud transactions','Normal transactions']
conf_df = pd.DataFrame(confusion_matrix, index=classes ,columns=classes)  #将矩阵转化为 DataFrame

# sns.set_context({"figure.figsize":(8,8)})
conf_fig = sns.heatmap(data=conf_df, square=True, annot=True, fmt="d", cmap="Pastel2_r",center=300,
linewidths=0.3,cbar_kws={"orientation":"vertical"})  #绘制 heatmap
plt.xlabel('prediction',fontsize=14, color='k') #x轴label的文本和字体大小
plt.ylabel('true label',fontsize=14, color='k') #y轴label的文本和字体大小
plt.show()
