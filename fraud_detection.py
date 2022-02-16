import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
#import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns


df= pd.read_csv("creditcard.csv")

# 预处理
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1) # 为什么要drop掉这些列？
# 替换数据 v1的值小于-3则替换为1，否则为0
# df['V1_'] = df.V1.map(lambda x: 1 if x < -3 else 0)
# df['V2_'] = df.V2.map(lambda x: 1 if x > 2.5 else 0)
# df['V3_'] = df.V3.map(lambda x: 1 if x < -4 else 0)
# df['V4_'] = df.V4.map(lambda x: 1 if x > 2.5 else 0)
# df['V5_'] = df.V5.map(lambda x: 1 if x < -4.5 else 0)
# df['V6_'] = df.V6.map(lambda x: 1 if x < -2.5 else 0)
# df['V7_'] = df.V7.map(lambda x: 1 if x < -3 else 0)
# df['V9_'] = df.V9.map(lambda x: 1 if x < -2 else 0)
# df['V10_'] = df.V10.map(lambda x: 1 if x < -2.5 else 0)
# df['V11_'] = df.V11.map(lambda x: 1 if x > 2 else 0)
# df['V12_'] = df.V12.map(lambda x: 1 if x < -2 else 0)
# df['V14_'] = df.V14.map(lambda x: 1 if x < -2.5 else 0)
# df['V16_'] = df.V16.map(lambda x: 1 if x < -2 else 0)
# df['V17_'] = df.V17.map(lambda x: 1 if x < -2 else 0)
# df['V18_'] = df.V18.map(lambda x: 1 if x < -2 else 0)
# df['V19_'] = df.V19.map(lambda x: 1 if x > 1.5 else 0)
# df['V21_'] = df.V21.map(lambda x: 1 if x > 0.6 else 0)

# 选定类别为0的表项，normal的值设为1，即认为是正常的交易
df.loc[df.Class == 0, 'Normal'] = 1
df.loc[df.Class == 1, 'Normal'] = 0

# 将class更名为fraud，即是否为欺诈交易
df = df.rename(columns={'Class': 'Fraud'})

# 分离出正常的样本和欺诈的样本
Fraud = df[df.Fraud == 1]
Normal = df[df.Normal == 1]

# 从欺诈交易样本中采样75%的样本作为训练样本
X_train = Fraud.sample(frac=0.75)
count_Frauds = len(X_train)

# 从正常交易样本中采样75%的样本作为训练样本
# X_train = pd.concat([X_train, Normal.sample(frac = 0.75)], axis = 0)
X_train = pd.concat([X_train, Normal[:count_Frauds]],axis=0)

#X_test contains all the transaction not in X_train.
X_test = df.loc[~df.index.isin(X_train.index)]

#Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(X_test)

#Add our target features to y_train and y_test.
y_train = X_train.Fraud
y_train = pd.concat([y_train, X_train.Normal], axis=1)

y_test = X_test.Fraud
y_test = pd.concat([y_test, X_test.Normal], axis=1)

# 从特征集中删除类别标签
X_train = X_train.drop(['Fraud','Normal'], axis = 1)
X_test = X_test.drop(['Fraud','Normal'], axis = 1)

# 训练集样本总数与其中的欺诈交易样本数量的比例
ratio = len(X_train)/count_Frauds 

# 对于欺诈交易样本放大标签值？
# y_train.Fraud *= ratio
# y_test.Fraud *= ratio

#Names of all of the features in X_train.
features = X_train.columns.values # features是特征的名字
#mean= df[feature].mean()
#print (features)
#Transform each feature in features so that it has a mean of 0 and standard deviation of 1; 
#this helps with training the neural network.

# 计算所有样本的均值和方差 进行标准化
for feature in features:
	mean, std = df[feature].mean(), df[feature].std()
	#print mean
	#print std
	X_train.loc[:, feature] = (X_train[feature] - mean) / std  # 标准化
	X_test.loc[:, feature] = (X_test[feature] - mean) / std

inputX = X_train.values
inputY = y_train.values
inputX_test = X_test.values
inputY_test = y_test.values

input_nodes = 19

#Multiplier maintains a fixed ratio of nodes between each layer.
mulitplier = 1.5 

#Number of nodes in each hidden layer
hidden_nodes1 = 15
hidden_nodes2 = round(hidden_nodes1 * mulitplier)
hidden_nodes3 = round(hidden_nodes2 * mulitplier)
hidden_nodes2= int(hidden_nodes2)
hidden_nodes3= int(hidden_nodes3)
#Percent of nodes to keep during dropout.
pkeep = 0.9

#input
x = tf.placeholder(tf.float32,shape=(None, input_nodes))

#layer 1
W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev = 0.1)) # 权重
b1 = tf.Variable(tf.zeros([hidden_nodes1])) # bias
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1) # y=wx+b

#layer 2
W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev = 0.1))
b2 = tf.Variable(tf.zeros([hidden_nodes2]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

#layer 3
W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev = 0.1)) 
b3 = tf.Variable(tf.zeros([hidden_nodes3]),)
y3 = tf.nn.sigmoid(tf.matmul(y2, W3) + b3)
y3 = tf.nn.dropout(y3, pkeep)
                                                        
#layer 4
W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, 2], stddev = 0.1)) 
b4 = tf.Variable(tf.zeros([2]))
y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)

#output
y = y4
y_ = tf.placeholder(tf.float32, [None, 2])

#Parameters
training_epochs = 100 #should be 2000, but the kernels dies from running for more than 1200 seconds.
display_step = 20
n_samples = y_train.size

batch = tf.Variable(0)

learning_rate = tf.train.exponential_decay(
  0.01,              #Base learning rate.
  batch,             #Current index into the dataset.
  len(inputX),       #Decay step.																																																																																																																																																																																																			
  0.95,              #Decay rate.
  staircase=False)

#Cost function: Cross Entropy
cost = -tf.reduce_sum(y_ * tf.log(y))

#We will optimize our model via AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Correct prediction if the most likely value (Fraud or Normal) from softmax equals the target value.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cm = tf.confusion_matrix(tf.argmax(y_,1), tf.argmax(y,1),2)


#Initialize variables and tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

accuracy_summary = [] #Record accuracy values for plot
cost_summary = [] #Record cost values for plot

for i in range(training_epochs):  
    sess.run([optimizer], feed_dict={x: inputX, y_: inputY})

    # Display logs per epoch step
    if (i) % display_step == 0:
        train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={x: inputX, y_: inputY})
        print ('Training step:', i,
               'Accuracy =', '{:.5f}'.format(train_accuracy), 
               'Cost = ', '{:.5f}'.format(newCost))
        accuracy_summary.append(train_accuracy)
        cost_summary.append(newCost)
        loss_ = sess.run(cost, feed_dict={x: inputX, y_: inputY})
        print(i, "loss: {:.4f}".format(loss_))
        
print ('Optimization Finished!')
training_accuracy = sess.run(accuracy, feed_dict={x: inputX, y_: inputY})
print ('Training Accuracy=', training_accuracy)
training_cm = sess.run(cm, feed_dict={x:inputX, y_:inputY})
print(training_cm)
testing_accuracy = sess.run(accuracy, feed_dict={x: inputX_test, y_: inputY_test})
print ('Testing Accuracy=', testing_accuracy)
testing_cm = sess.run(cm, feed_dict={x:inputX_test, y_:inputY_test})
print(testing_cm)

# #Plot accuracy and cost summary
# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))

# ax1.plot(accuracy_summary)
# ax1.set_title('Accuracy')

# ax2.plot(cost_summary)
# ax2.set_title('Cost')

# plt.xlabel('Epochs (x50)')
# #plt.show()