from gcForest import gcForest
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from evaluation import accuracy,f1_binary,f1_macro,f1_micro, auc_scores

#Addition of phylogenetic tree structure data
from ConcatenatTestLoad import test_load_data
# from test_load_data import test_load_data
from feature_selection_test import feature_select
from sklearn.metrics import average_precision_score,matthews_corrcoef,f1_score,recall_score,confusion_matrix,classification_report,roc_auc_score,auc,precision_recall_curve,accuracy_score,classification_report, roc_curve
from imblearn.metrics import geometric_mean_score


def get_config():
    config={}
    config["random_state"]=None
    config["max_layers"]=100
    config["early_stop_rounds"]=1
    config["if_stacking"]=False
    config["if_save_model"]=False
    config["train_evaluation"]=f1_binary    # f1_binary,f1_macro,f1_micro,accuracy
    config["estimator_configs"]=[]
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":100,"n_jobs":-1})
    config["output_layer_config"]=0
    return config
def get_config1():
    config={}
    config["random_state"]=None
    config["max_layers"]=100
    config["early_stop_rounds"]=1
    config["if_stacking"]=False
    config["if_save_model"]=False
    config["train_evaluation"]=f1_micro
    config["estimator_configs"]=[]
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":50,"n_jobs":-1})
    config["output_layer_config"]=[]
    return config
def get_config2():
    config={}
    config["random_state"]=None
    config["max_layers"]=100
    config["early_stop_rounds"]=1
    config["if_stacking"]=False
    config["if_save_model"]=False
    config["train_evaluation"]=f1_binary
    config["estimator_configs"]=[]
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"RandomForestClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":50,"n_jobs":-1})
    config["estimator_configs"].append({"n_fold":5,"type":"ExtraTreesClassifier","n_estimators":50,"n_jobs":-1})
    config["output_layer_config"]=[]
    return config

if __name__=="__main__":



    # x_train_ibd, y_train_ibd, _, n_feature_2_ibd = test_load_data(dataid=1)  # 加载IBD数据集
    # x_val_ijazuz, y_val_ijazuz, _, _ = test_load_data(dataid=9)  # 加载IjazUz数据集
    x, y, dataid, n_feature_2 =test_load_data(1)

    skf=RepeatedStratifiedKFold(n_splits=5,random_state=123,n_repeats=1)
    accuracys=[]
    aucs=[]
    f1s=[]
    auprs=[]
    mccs=[]
    recalls=[]
    gmeans=[]
    i = 1
    for train_id,test_id in skf.split(x,y):
        print("============{}-th cross validation============".format(i))
        
        mate_data = x[:, n_feature_2:-1]
        phylogen = x[:, 0:n_feature_2]

        mate_data_train, mate_data_test = mate_data[train_id], mate_data[test_id]
        phylogen_train, phylogen_test = phylogen[train_id], phylogen[test_id]
        print("mate_data_train", mate_data_train.shape)
        print("mate_data_test", mate_data_test.shape)
        y_train, y_test = y[train_id], y[test_id]

        mate_index = feature_select(mate_data, y_train,mate_data.shape[0])
        phylogen_index = feature_select(phylogen, y_train, phylogen.shape[0])


        mate_data_train = mate_data_train[:, mate_index]
        phylogen_train = phylogen_train[:, phylogen_index]
        mate_data_test = mate_data_test[:, mate_index]
        phylogen_test = phylogen_test[:, phylogen_index]


        # train gcForse for modality 1
        gc1 = gcForest(get_config1())
        gc1.fit(mate_data_train, y_train)
        mate_data_train_features = gc1.predict_proba(mate_data_train)  # shape: (n_samples, n_features1)
        mate_data_test_features = gc1.predict_proba(mate_data_test)  # shape: (n_samples, n_features1)
        # train gcForest for modality 2
        gc2 = gcForest(get_config2())
        gc2.fit(phylogen_train, y_train)
        phylogen_train_features = gc2.predict_proba(phylogen_train)  # shape: (n_samples, n_features2)
        phylogen_test_features = gc2.predict_proba(phylogen_test)  # shape: (n_samples, n_features2)



        # concatenate features from both modalities
        x_train_features = np.concatenate((mate_data_train_features, phylogen_train_features ,mate_data_train, phylogen_train),axis=1)  # shape: (n_samples, n_features1 + n_features2)
    
        print("x_train_feature shape:", x_train_features.shape)
        x_test_features = np.concatenate((mate_data_test_features, phylogen_test_features,mate_data_test, phylogen_test), axis=1)  # shape: (n_samples, n_features1 + n_features2)
        


        config = get_config()
        gc = gcForest(config)
        gc.fit(x_train_features, y_train)
        
        y_pred = gc.predict(x_test_features)
        y_pred_prob = gc.predict_proba(x_test_features)
        y_score = y_pred_prob[:, 1]
        y_score = []
        for item in y_pred_prob:
            y_score.append(item[1])
        y_score = np.array(y_score)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score, pos_label=2)
        aupr = auc(recall,precision)

        # compute model output tpr and fpr, and save as file
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred,average='binary')
        recall = recall_score(y_test, y_pred,average="binary")
        mcc = matthews_corrcoef(y_test, y_pred)
        gmean= geometric_mean_score(y_test, y_pred,average='binary')
        auc_s = auc_scores(y_test, y_pred_prob[:, 1])
          
        f1s.append(f1)
        accuracys.append(accuracy)
        aucs.append(auc_s)
        auprs.append(aupr)
        recalls.append(recall)
        mccs.append(mcc)
        gmeans.append(gmean)

        with open('tpr_fpr.txt', 'a') as f:
            f.write(f"========={i}-th cross validation=========\n")
            for j in range(len(fpr)):
                f.write(f"FPR: {fpr[j]}\tTPR: {tpr[j]}\n")

        i += 1

        
    print("============training finished============")
    f1s=np.array(f1s)
    accs=np.array(accuracys)
    auprs=np.array(auprs)
    recalls=np.array(recalls)
    mccs=np.array(mccs)
    gmeans=np.array(gmeans)

    def print_mean_std(metric_name, values):
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric_name}: {mean:.4f} ± {std:.4f}")


    print("Data:", dataid)
    print_mean_std("accuracy", accs)
    print_mean_std("auc", aucs)
    print_mean_std("f1", f1s)
    print_mean_std("aupr", auprs)
    print_mean_std("recall", recalls)
    print_mean_std("mcc", mccs)
    print_mean_std("gmean", gmeans)





       







