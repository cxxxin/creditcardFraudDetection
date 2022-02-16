import imp
import pandas as pd
import numpy as np

"""
【函数说明】获取箱体图特征
【输入】 input_list 输入数据列表
【输出】 out_list：列表的特征[下限，Q1,Q2,Q3,上限] 和 Error_Point_num：异常值数量
【版本】 V1.0.0
【日期】 2019 10 16
"""
def BoxFeature(input_list):
    # 获取箱体图特征
    percentile = np.percentile(input_list, (25, 50, 75), interpolation='linear')
    #以下为箱线图的五个特征值
    Q1 = percentile[0]#上四分位数
    Q2 = percentile[1]
    Q3 = percentile[2]#下四分位数
    IQR = Q3 - Q1#四分位距
    ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
    llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
    # llim = 0 if llim < 0 else llim
    # out_list = [llim,Q1,Q2,Q3,ulim]
    # 统计异常点个数
    # 正常数据列表
    right_list = []
    Error_Point_num = 0
    value_total = 0
    average_num = 0
    for item in input_list:
        if item < llim or item > ulim:
            Error_Point_num += 1
        else:
            right_list.append(item)
            value_total += item
            average_num += 1
    average_value =  value_total/average_num
    # 特征值保留一位小数
    out_list = [average_value,min(right_list), Q1, Q2, Q3, max(right_list)]
    # print(out_list)
    # out_list = Save1point(out_list)
    return min(right_list),max(right_list)


def preprocess():
    # load dataset
    df = pd.read_csv("creditcard.csv", low_memory=False)
    # df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
    # df = df.drop(['V5','V8','V9','V13','V15','V20','V21','V22','V23','V24','V25','V26','V27','V28'], axis =1)

    # 与顺序无关
    # df.drop('Time',axis=1)	

    # 选定类别为0的表项，normal的值设为1，即认为是正常的交易
    df.loc[df.Class == 0, 'Normal'] = 1
    df.loc[df.Class == 1, 'Normal'] = -1

    # 将class更名为fraud，即是否为欺诈交易
    df = df.rename(columns={'Class': 'Fraud'})
    features = df.columns.values # features是特征的名字

    # 分离出正常的样本和欺诈的样本
    Fraud = df[df.Fraud == 1]
    Normal = df[df.Normal == 1]

    # preprocessing: 处理正样本中的离群点 
    for feature in features:
        if feature == 'Fraud' or feature == 'Normal' or feature == 'Time':
            continue
        # calculate the maximum and minimum value of non-outliers
        X = Normal.loc[:,feature].values
        ymin,ymax = BoxFeature(X)

        # replace outliers
        Normal[feature] = Normal[feature].mask(Normal[feature]>ymax, ymax)
        Normal[feature] = Normal[feature].mask(Normal[feature]<ymin, ymin)
        # Normal[feature] = Normal.loc[:,feature].map(lambda x: ymax if x > ymax else x)
        # Normal[feature] = Normal.loc[:,feature].map(lambda x: ymin if x < ymin else x)    


    #将DataFrame存储为csv,index表示是否显示行名，default=True
    Normal.to_csv("Normal_for_SVM.csv",index=False,sep=',')
    Fraud.to_csv("Fraud_for_SVM.csv",index=False,sep=',')

if __name__ == "__main__":
    preprocess()