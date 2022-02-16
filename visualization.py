from decimal import Subnormal
from pyexpat import features
# from random import shuffle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import plotly
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import scipy.stats
import matplotlib.ticker as mtick
from scipy.stats import norm


def hist():
    # load dataset
    df = pd.read_csv("creditcard.csv")
    # df.drop('Time',axis=1)

    # 分离出正常的样本和欺诈的样本
    Fraud = df[df.Class == 1]
    Normal = df[df.Class == 0]

    features = df.columns.values


    fig = plt.figure(figsize=(2,2))
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=1, hspace=0.8)
    index = 1

    features=['Time','Amount']
    for i in range(2):
        indexCol = features[i]
        # indexCol = f'V{index+i}'
        index = i
        subFraud = Fraud.loc[:,indexCol].values
        subNormal = Normal.loc[:,indexCol].values

        ax = fig.add_subplot(2,2,index*2+1) # 2rows 2columns
        n, bins, patches = ax.hist(subFraud, density=True, bins=30, color='yellowgreen')
        mu = np.mean(subFraud)
        sigma = np.std(subFraud)
        y = norm.pdf(bins, mu, sigma)
        ax.plot(bins, y, '--', color ='black', alpha = 0.7) 
        ax.set_title(f'{indexCol}-Fraud',fontsize=14)
        ax.tick_params(labelsize=8) #刻度字体大小8
        ax = fig.add_subplot(2,2,index*2+2)
        n, bins, patches = ax.hist(subNormal, density=True, bins=30, color='pink')
        mu = np.mean(subNormal)
        sigma = np.std(subNormal)
        y = norm.pdf(bins, mu, sigma)
        ax.plot(bins, y, '--', color ='black',alpha = 0.7) 
        ax.tick_params(labelsize=8) #刻度字体大小8
        ax.set_title(f'{indexCol}-Normal',fontsize=14)
        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        # plt.tight_layout()

    plt.legend()
    plt.show()

def boxplot():
    
    # load dataset
    df = pd.read_csv("creditcard.csv")
    features = df.columns.values

    for i in range(len(features)-1):
        
        indexCol = features[i]
        fraud = df[df.Class == 1]
        fraud = fraud.rename(columns={indexCol:"Fraud"})
        fraud = fraud.loc[:,"Fraud"]
        normal = df[df.Class == 0]
        normal = normal.rename(columns={indexCol:"Normal"})
        normal = normal.loc[:,"Normal"]

        fig = plt.figure(num=i)
        # manager = plt.get_current_fig_manager()
        # manager.resize(*manager.window.maxsize())

        ax1 = fig.add_subplot(1, 2, 1)
        # sns.boxplot(y=fraud, ax=ax1, showmeans=True, color="powderblue")
        # current_palette = sns.color_palette("Paired")
        sns.boxplot(y=fraud, ax=ax1, showmeans=True, width=0.7,
                medianprops = {'linestyle':'--','color':'#ab1239'},#设置中位数线线型及颜色
                meanprops = {'marker':'D','markerfacecolor':'white'},#设置均值属性
                boxprops={'color':'#ab1239','facecolor':'#c0737a','alpha':0.7},# color = dusty rose
                flierprops={"marker": "o", "markerfacecolor": "#6c7a0e", "markersize": 3})
        maxY = fraud.max(axis=0)
        minY = fraud.min(axis=0)

        ax2 = fig.add_subplot(1, 2, 2)
        # sns.boxplot(y=normal, ax=ax2, showmeans=True, color="darkseagreen")
        # current_palette =sns.hls_palette(8, l=.6, s=.8) # 通过palette设置调色板
        # current_palette = sns.color_palette("Paired")
        ax_temp = sns.boxplot(y=normal, ax=ax2, showmeans=True,width=0.7,#箱图显示均值，
                medianprops = {'linestyle':'--','color':'#6f7632'},#设置中位数线线型及颜色
                meanprops = {'marker':'D','markerfacecolor':'white'},#设置均值属性
                boxprops={'color':'#6f7632','facecolor':'#769958','alpha':0.7},# color = moss
                flierprops={"marker": "o", "markerfacecolor": "#6c7a0e", "markersize": 3}) 
        maxY = normal.max(axis=0) if normal.max(axis=0) > maxY else maxY
        minY = normal.min(axis=0) if normal.min(axis=0) < minY else minY

        ax1.set(ylim=(minY, maxY))
        ax1.set(title=f'{indexCol} - Fraud')
        ax1.set(ylabel=None)

        ax2.set(ylim=(minY, maxY))
        ax2.set(yticklabels=[])
        ax2.set(title=f'{indexCol} - Normal')
        ax2.set(ylabel=None)

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.subplots_adjust(wspace=0.7, hspace=0.4)
        # plt.savefig(f'./figures/{indexCol}.png', bbox_inches='tight', dpi=300)
        # plt.legend()
        # plt.show()
        plt.close()

def violin():
    # load dataset
    df = pd.read_csv("creditcard.csv")

    fig = make_subplots(rows=6, cols=5, subplot_titles=list(df.columns))  

    for row_num in range(1, 7):
        start_list = 0 + ((row_num-1) * 5)
        end_list = 5 + ((row_num-1) * 5)
        for idx, feature in enumerate(list(df.columns)[start_list: end_list]):
            fig.add_trace(go.Violin(x=df["Class"][df["Class"] == 1],
                                    y=df[feature][df["Class"] == 1],
                                    legendgroup="Fraud", scalegroup="Fraud", name="Fraud",
                                    line_color="blue"),
                        row=row_num, col=(idx+1))
            fig.add_trace(go.Violin(x=df["Class"][df["Class"] == 0][::100], # 1 in 100 otherwise will crash...
                                    y=df[feature][df["Class"] == 0][::100], # 1 in 100 otherwise will crash...
                                    legendgroup="Not Fraud", scalegroup="Not Fraud", name="Not Fraud",
                                    line_color='orange'),
                        row=row_num, col=(idx+1))

    fig.update_traces(meanline_visible=True)
    fig.update_layout(showlegend=False, violingap=0, height=1500,
                    title_text="Distributions of Fraudulent vs Not Fraudulent Transactions along each PC and Amount")
    # fig.show()
    fig.write_html('first_figure.html')

def linear_correlations():
    df = pd.read_csv("creditcard.csv")
    # print(df.corr())
    # sns.pairplot(df[:1000], hue ='Class', markers=["o","d"])
    # cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
    # sns.heatmap(df[::100].corr(), linewidths = 0.05, cmap="YlGnBu", center=None)
    # sns.heatmap(df[::100].corr(), linewidths = 0.05, cmap="vlag", center=None)
    sns.heatmap(df[::100].corr(), linewidths = 0.05, cmap="Pastel1", center=None)
    plt.show()
    

def similarity():
    df = pd.read_csv("creditcard.csv")
    features = df.columns.values
    

    for f in features:   
        Fraud = df[df.Class==1].loc[:,f].values
        Normal = df[df.Class==0].loc[:,f]
        subNormal = shuffle(Normal)
        subNormal = subNormal[:len(Fraud)].values

        # M = (subNormal+Fraud)/2
        # print(0.5*scipy.stats.entropy(subNormal, M, base=2)+0.5*scipy.stats.entropy(Fraud, M, base=2))
        print(f+":"+str(scipy.stats.wasserstein_distance(subNormal,Fraud)))

def classification():
    df = pd.read_csv("creditcard.csv")
    features = df.columns.values

    for f in features:
        sns.set(style="whitegrid",color_codes=True)
        ax = sns.stripplot(x="Class",y=f,data=df,palette=["#769958","#c0737a"],alpha=0.4,size=8)
        plt.savefig(f'./classification/{f}.png', bbox_inches='tight', dpi=300)
        plt.show()

if __name__ == "__main__":
    # violin()
    # boxplot()
    # linear_correlations()
    similarity()
    # hist()
    # classification()