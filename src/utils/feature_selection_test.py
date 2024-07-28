import numpy as np
import itertools
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from skbio.stats.composition import clr
# from lefse import run_lefse
# from test_load_data import test_load_data
# from sklearn.linear_model import Lasso, LassoCV, Ridge

"""
def calc_lda_scores(x, y):
    # Calculate LDA scores for each feature using LEfSE.
    # LEfSE expects input data in a specific format, so we transform the data accordingly
    data = pd.DataFrame(x)
    data['class'] = y
    data.to_csv('data.txt', sep='\t', index=False)

    # Run LEfSE
    def run_lefse(input_file, output_file):
    command = f"run_lefse {input_file} {output_file}"
    os.system(command)

def parse_lefse_results(output_file):
    results = pd.read_csv(output_file, sep='\t', header=None)
    results.columns = ["Feature", "LDA Score", "p-value"]
    return results

    # Read the LEfSE output and sort features by LDA score
    lda_scores = pd.read_csv('lefse_output.txt', sep='\t')
    lda_scores = lda_scores.sort_values(by='lda_score', ascending=False)

    return lda_scores


def feature_select(x, y, n_features):

    # Select top features based on LDA scores and reduce dimensionality using RandomForest.

    lda_scores = calc_lda_scores(x, y)

    # Select top features based on LDA scores
    top_features = lda_scores['feature'][:n_features]

    # Reduce dimensionality using RandomForest
    x_selected = x[:, top_features]

    return x_selected, y, top_features

"""

"""
def feature_select(x, y, k):

    def calc_mrmr(x, y, k):
        n_feature = x.shape[1]
        f_x_y = np.zeros(n_feature)
        mi = np.zeros((n_feature, n_feature))
        for i in range(n_feature):
            f_x_y[i] = np.abs(np.corrcoef(x[:, i], y)[0, 1])
            for j in range(n_feature):
                if i == j:
                    mi[i, j] = 0
                else:
                    mi[i, j] = calc_mi(x[:, i], x[:, j])
        s = set()
        for i in range(k):
            max_f_x_y = -float('inf')
            max_f_index = None
            for j in range(n_feature):
                if j not in s:
                    f = f_x_y[j]
                    if len(s) == 0:
                        max_f_s = 0
                    else:
                        max_f_s = np.max(np.array([mi[j, index] for index in s]))
                    if f - max_f_s > max_f_x_y:
                        max_f_x_y = f - max_f_s
                        max_f_index = j
            if max_f_index is None:
                break
            s.add(max_f_index)

        return np.array(list(s))


    def calc_mi(x, y):
       
        
        hist_x = np.histogram(x, 10)[0]
        hist_y = np.histogram(y, 10)[0]
        hist_x_y = np.histogram2d(x, y, 10)[0]
        p_x = hist_x / len(x)
        p_y = hist_y / len(y)
        p_x_y = (hist_x_y + 1e-8) / len(x)
        mask = np.where(p_x_y == 0, 1, 0)
        # p_x_y[mask] = 1
        p_x[mask[:, 0]] = 1
        p_y[mask[0]] = 1

        # p_x_y[p_x_y == 0] = 1
        mi = np.sum(p_x_y * np.log((p_x_y + 1e-8) / ((p_x.reshape(-1, 1) + 1e-8) * (p_y + 1e-8))))
        return mi

    feature = calc_mrmr(x, y, k)
    return feature

    """

"""
def feature_select(x, y, k):
    model = Ridge(alpha=0.0110113135)  
    model.fit(x, y)
    coef = np.abs(model.coef_) 
    index = np.argsort(coef)[-k:] 
    return index

"""
"""
def feature_select(x, y, k, n_feature):
    model = LassoCV(cv=5, max_iter=1000, tol=0.0001)  # 设置Lasso模型，并进行交叉验证
    model.fit(x, y)
    coef = np.abs(model.coef_)  
    index = np.argsort(coef)[-k:]  
    return index[:n_feature]
#
# x, y, dataid = test_load_data()
# feature, model = select_feature(x, y, 300)
# print("model alpa is ", model.alpha_)
# print(feature)

"""

def calc_f3(x,y):
    labels=np.unique(y)
    indexs={}
    c_mins={}
    c_maxs={}
    for label in labels:
        index=np.where(y==label)[0]
        indexs[label]=index
        c_min=np.min(x[index])
        c_max=np.max(x[index])
        c_mins[label]=c_min
        c_maxs[label]=c_max
    label_combin=list(itertools.combinations(labels,2))
    f3=0.0

    if not label_combin:
        return f3

    for combination in label_combin:
        # sample_num=len(indexs[combination[0]])+len(indexs[combination[1]])
        # print(sample_num)
        # print(combination)
        # print(sample_num)
        c1_max,c1_min=c_maxs[combination[0]],c_mins[combination[0]]
        c2_max,c2_min=c_maxs[combination[1]],c_mins[combination[1]]
        # print(c1_max,c1_min,c2_max,c2_min)
        if c1_max<c2_min or c2_max<c1_min:
            f3+=1
        else:
            interval=(max(c1_min,c2_min),min(c1_max,c2_max))
            sample=np.hstack((x[indexs[combination[0]]],x[indexs[combination[1]]]))
            # print(sample.shape[0])
            n_overlay=0
            for k in range(sample.shape[0]):
                if sample[k]>=interval[0] and sample[k]<=interval[1]:
                    n_overlay+=1
            if sample.shape[0] > 0:
                f3+=1-n_overlay/sample.shape[0]
            else:
                f3 += 0
    f3/=len(label_combin)
    return f3
def feature_select(x,y,k):
# def feature_select(x,y,k, n_feature): # cross-study validation
    n_feature=x.shape[1]
    f3s=[0.0 for i in range(n_feature)]
    for i in range(n_feature):
        if len(np.unique(x[:,i]))==1:
            f3s[i]=0
        elif len(np.unique(x[:,i]))==2:
            f3s[i]=1
        else:
            f3s[i]=calc_f3(x[:,i],y)
    index=np.argsort(f3s)
    index=index[-k:]
    # return x[:,index],y

    return index[:n_feature]

