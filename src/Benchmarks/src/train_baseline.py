import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os.path import abspath
import numpy as np
import pandas as pd
from utils.generate_network import generate_network
from utils.prepare_data import prepare_data
from utils.benchemarks_io import get_config, save_params, load_params
from utils.benchemarks_io import get_stat, get_stat_dict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc, confusion_matrix
from models.CNN1D import CNN1D
from models.MLPNN import MLPNN
from models.RF import RF
from models.SVM import SVM
from models.LASSO import LASSO
from models.DF import DF
import warnings
from datetime import datetime
import webbrowser
import subprocess
import json

config = get_config()
warnings.filterwarnings("ignore")
# np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=sys.maxsize)

def train_baseline():
    # Read in Config File
    config = get_config()
    filt_thresh = config.get('Evaluation', 'FilterThresh')
    dataset = config.get('Evaluation', 'DataSet')

    time_stamp = int(np.round(datetime.timestamp(datetime.now()), 0))
    ts = str(filt_thresh) + "_" + str(time_stamp)

    result_path = "../results/" + dataset + "/" + ts
    print("Saving results to %s" % (result_path))
    try:
        os.mkdir("../results/")
    except OSError:
        pass

    try:
        os.mkdir("../results/" + dataset)
    except OSError:
        pass

    try:
        os.mkdir(result_path)
    except OSError:
        print("Creation of the result subdirectory failed...")
        print("Exiting...")
        sys.exit()

    try:
        os.mkdir(result_path + "/prediction_evaluation")
    except OSError:
        print("Creation of the prediction evaluation subdirectory failed...")
        print("Exiting...")
        sys.exit()

    try:
        os.mkdir(result_path + "/feature_evaluation")
    except OSError:
        print("Creation of the feature evaluation subdirectory failed...")
        print("Exiting...")
        sys.exit()

    num_runs = int(config.get('Evaluation', 'NumberRuns'))
    num_test = int(config.get('Evaluation', 'NumberTestSplits'))

    print("\nStarting Benchmark on %s..." % (dataset))
    path = "../z_data/" + dataset

    my_maps, raw_x, tree_x, raw_features, tree_features, labels, label_set, g, feature_df = prepare_data(path, config)


    num_class = len(np.unique(labels))
    if num_class == 2:
        metric = "AUC"
    else:
        metric = "MCC"

    seed = np.random.randint(100)
    np.random.seed(seed)
    np.random.shuffle(my_maps)
    np.random.seed(seed)
    np.random.shuffle(raw_x)
    np.random.seed(seed)
    np.random.shuffle(tree_x)
    np.random.seed(seed)
    np.random.shuffle(labels)

    n_values = np.max(labels) + 1
    labels_oh = np.eye(n_values)[labels]

    tree_row = my_maps.shape[1]
    tree_col = my_maps.shape[2]

    print("There are %d classes...%s" % (num_class, ", ".join(label_set)))
    cv_list = ["Run_" + str(x) + "_CV_" + str(y) for x in range(num_runs) for y in range(num_test)]
    seeds = np.random.randint(1000, size=num_runs)

    # 使用丰度谱数据，对比基准模型：CNN1D，MLP,RF,SVM,Lasso
    # Read in data and generate tree maps
    cnn1d_stat_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1", "AUPR", "Accuracy"], columns=cv_list)
    mlpnn_stat_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1", "AUPR", "Accuracy"], columns=cv_list)
    rf_stat_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1", "AUPR", "Accuracy"], columns=cv_list)
    gc_stat_df = pd.DataFrame(index = ["AUC", "MCC", "Precision", "Recall", "F1", "AUPR", "Accuracy"], columns=cv_list)
    svm_stat_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1", "AUPR", "Accuracy"], columns=cv_list)
    lasso_stat_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1", "AUPR", "Accuracy"], columns=cv_list)
    

    run = 0
    for seed in seeds:
        skf = StratifiedKFold(n_splits=num_test, shuffle=True, random_state=seed)
        fold = 0
        # models train and test ,fit models
        for train_index, test_index in skf.split(my_maps, labels):
            train_x, test_x = raw_x[train_index, :], raw_x[test_index, :]
            train_y_oh, test_y_oh = labels_oh[train_index, :], labels_oh[test_index, :]
            train_y, test_y = labels[train_index], labels[test_index]

            train_x = np.log(train_x + 1)
            test_x = np.log(test_x + 1)

            c_prob = [0] * len(np.unique(labels))
            train_weights = []

            for l in np.unique(labels):
                a = float(len(labels))
                b = 2.0 * float((np.sum(labels == l)))
                c_prob[int(l)] = a / b

            c_prob = np.array(c_prob).reshape(-1)

            for l in np.argmax(train_y_oh, 1):
                train_weights.append(c_prob[int(l)])
            train_weights = np.array(train_weights)

            scaler = MinMaxScaler().fit(train_x)
            train_x = np.clip(scaler.transform(train_x), 0, 1)
            test_x = np.clip(scaler.transform(test_x), 0, 1)

            train_oh = [train_x, train_y_oh]
            test_oh = [test_x, test_y_oh]

            train = [train_x, train_y]
            test = [test_x, test_y]

            cnn1D_model = CNN1D(train_x.shape[1], num_class, config)
            mlpnn_model = MLPNN(train_x.shape[1], num_class, config)
            # gc_model = DF(config)
            rf_model = RF(config)
            svm_model = SVM(config, label_set)
            lasso_model = LASSO(config, label_set)

            if fold + run == 0:
                print("CNN-1D")
                print(cnn1D_model.model.summary())
                print("\n\nMLPNN")
                print(mlpnn_model.model.summary())
                print("\n\n Run\tFold\tRF %s\t\tSVM %s\t\tLASSO %s\tMLPNN %s\tCNN-1D %s" % (metric, metric,
                                                                                            metric, metric, metric))

            cnn1D_model.train(train_oh, train_weights)
            preds, cnn1d_stats = cnn1D_model.test(test_oh)
            if num_class == 2:
                cnn1d_stat_df.loc["AUC"]["Run_" + str(run) + "_CV_" + str(fold)] = cnn1d_stats["AUC"]
            cnn1d_stat_df.loc["MCC"]["Run_" + str(run) + "_CV_" + str(fold)] = cnn1d_stats["MCC"]
            cnn1d_stat_df.loc["Precision"]["Run_" + str(run) + "_CV_" + str(fold)] = cnn1d_stats["Precision"]
            cnn1d_stat_df.loc["Recall"]["Run_" + str(run) + "_CV_" + str(fold)] = cnn1d_stats["Recall"]
            cnn1d_stat_df.loc["F1"]["Run_" + str(run) + "_CV_" + str(fold)] = cnn1d_stats["F1"]
            cnn1d_stat_df.loc["AUPR"]["Run_" + str(run) + "_CV_" + str(fold)] = cnn1d_stats["AUPR"]
            cnn1d_stat_df.loc["Accuracy"]["Run_" + str(run) + "_CV_" + str(fold)] = cnn1d_stats["Accuracy"]

            mlpnn_model.train(train_oh, train_weights)
            preds, mlpnn_stats = mlpnn_model.test(test_oh)
            if num_class == 2:
                mlpnn_stat_df.loc["AUC"]["Run_" + str(run) + "_CV_" + str(fold)] = mlpnn_stats["AUC"]
            mlpnn_stat_df.loc["MCC"]["Run_" + str(run) + "_CV_" + str(fold)] = mlpnn_stats["MCC"]
            mlpnn_stat_df.loc["Precision"]["Run_" + str(run) + "_CV_" + str(fold)] = mlpnn_stats["Precision"]
            mlpnn_stat_df.loc["Recall"]["Run_" + str(run) + "_CV_" + str(fold)] = mlpnn_stats["Recall"]
            mlpnn_stat_df.loc["F1"]["Run_" + str(run) + "_CV_" + str(fold)] = mlpnn_stats["F1"]
            mlpnn_stat_df.loc["AUPR"]["Run_" + str(run) + "_CV_" + str(fold)] = mlpnn_stats["AUPR"]
            mlpnn_stat_df.loc["Accuracy"]["Run_" + str(run) + "_CV_" + str(fold)] = mlpnn_stats["Accuracy"]

            rf_model.train(train)
            preds, rf_stats = rf_model.test(test)
            if num_class == 2:
                rf_stat_df.loc["AUC"]["Run_" + str(run) + "_CV_" + str(fold)] = rf_stats["AUC"]
            rf_stat_df.loc["MCC"]["Run_" + str(run) + "_CV_" + str(fold)] = rf_stats["MCC"]
            rf_stat_df.loc["Precision"]["Run_" + str(run) + "_CV_" + str(fold)] = rf_stats["Precision"]
            rf_stat_df.loc["Recall"]["Run_" + str(run) + "_CV_" + str(fold)] = rf_stats["Recall"]
            rf_stat_df.loc["F1"]["Run_" + str(run) + "_CV_" + str(fold)] = rf_stats["F1"]
            rf_stat_df.loc["AUPR"]["Run_" + str(run) + "_CV_" + str(fold)] = rf_stats["AUPR"]
            rf_stat_df.loc["Accuracy"]["Run_" + str(run) + "_CV_" + str(fold)] = rf_stats["Accuracy"]

            # gc_model.train(train)
            # preds, gc_stats = gc_model.test(test)
            # if num_class == 2:
            #     gc_stat_df.loc["AUC"]["Run_" + str(run) + "_CV_" + str(fold)] = gc_stats["AUC"]
            # gc_stat_df.loc["MCC"]["Run_" + str(run) + "_CV_" + str(fold)] = gc_stats["MCC"]
            # gc_stat_df.loc["Precision"]["Run_" + str(run) + "_CV_" + str(fold)] = gc_stats["Precision"]
            # gc_stat_df.loc["Recall"]["Run_" + str(run) + "_CV_" + str(fold)] = gc_stats["Recall"]
            # gc_stat_df.loc["F1"]["Run_" + str(run) + "_CV_" + str(fold)] = gc_stats["F1"]
            # gc_stat_df.loc["AUPR"]["Run_" + str(run) + "_CV_" + str(fold)] = gc_stats["AUPR"]
            # gc_stat_df.loc["Accuracy"]["Run_" + str(run) + "_CV_" + str(fold)] = gc_stats["Accuracy"]

            svm_model.train(train)
            preds, svm_stats = svm_model.test(test)
            if num_class == 2:
                svm_stat_df.loc["AUC"]["Run_" + str(run) + "_CV_" + str(fold)] = svm_stats["AUC"]
            svm_stat_df.loc["MCC"]["Run_" + str(run) + "_CV_" + str(fold)] = svm_stats["MCC"]
            svm_stat_df.loc["Precision"]["Run_" + str(run) + "_CV_" + str(fold)] = svm_stats["Precision"]
            svm_stat_df.loc["Recall"]["Run_" + str(run) + "_CV_" + str(fold)] = svm_stats["Recall"]
            svm_stat_df.loc["F1"]["Run_" + str(run) + "_CV_" + str(fold)] = svm_stats["F1"]
            svm_stat_df.loc["AUPR"]["Run_" + str(run) + "_CV_" + str(fold)] = svm_stats["AUPR"]
            svm_stat_df.loc["Accuracy"]["Run_" + str(run) + "_CV_" + str(fold)] = svm_stats["Accuracy"]

            lasso_model.train(train)
            preds, lasso_stats = lasso_model.test(test)
            if num_class == 2:
                lasso_stat_df.loc["AUC"]["Run_" + str(run) + "_CV_" + str(fold)] = lasso_stats["AUC"]
            lasso_stat_df.loc["MCC"]["Run_" + str(run) + "_CV_" + str(fold)] = lasso_stats["MCC"]
            lasso_stat_df.loc["Precision"]["Run_" + str(run) + "_CV_" + str(fold)] = lasso_stats["Precision"]
            lasso_stat_df.loc["Recall"]["Run_" + str(run) + "_CV_" + str(fold)] = lasso_stats["Recall"]
            lasso_stat_df.loc["F1"]["Run_" + str(run) + "_CV_" + str(fold)] = lasso_stats["F1"]
            lasso_stat_df.loc["AUPR"]["Run_" + str(run) + "_CV_" + str(fold)] = lasso_stats["AUPR"]
            lasso_stat_df.loc["Accuracy"]["Run_" + str(run) + "_CV_" + str(fold)] = lasso_stats["Accuracy"]

            # if metric == "AUC":
            #     print("# %d\t%d\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t%.3f" % (run, fold, rf_stats["AUC"], svm_stats["AUC"],
            #                                                               lasso_stats["AUC"], mlpnn_stats["AUC"],
            #                                                               cnn1d_stats["AUC"], gc_stats["AUC"]))
            # if metric == "MCC":
            #     print("# %d\t%d\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t%.3f" % (run, fold, rf_stats["MCC"], svm_stats["MCC"],
            #                                                               lasso_stats["MCC"], mlpnn_stats["MCC"],
            #                                                               cnn1d_stats["MCC"], gc_stats["AUC"]))
                
            if metric == "AUC":
                print("# %d\t%d\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t" % (run, fold, rf_stats["AUC"], svm_stats["AUC"],
                                                                          lasso_stats["AUC"], mlpnn_stats["AUC"],
                                                                          cnn1d_stats["AUC"]))
            if metric == "MCC":
                print("# %d\t%d\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t" % (run, fold, rf_stats["MCC"], svm_stats["MCC"],
                                                                          lasso_stats["MCC"], mlpnn_stats["MCC"],
                                                                          cnn1d_stats["MCC"]))

            cnn1D_model.destroy()
            mlpnn_model.destroy()
            del (rf_model)
            del (svm_model)
            del (lasso_model)
            # del(gc_model)

            fold += 1
        run += 1

    cnn1d_stat_df.to_csv(result_path + "/cnn1d_raw_evaluation.csv")
    mlpnn_stat_df.to_csv(result_path + "/mlpnn_raw_evaluation.csv")
    lasso_stat_df.to_csv(result_path + "/lasso_raw_evaluation.csv")
    svm_stat_df.to_csv(result_path + "/svm_raw_evaluation.csv")
    rf_stat_df.to_csv(result_path + "/rf_raw_evaluation.csv")
    # gc_stat_df.to_csv(result_path + "/df_raw_evaluation.csv")

    benchmark_df = pd.DataFrame(index=["AUC", "MCC", "Precision", "Recall", "F1", "AUPR", "Accuracy"],
                                columns=["RF", "SVM", "LASSO", "MLPNN", "CNN1D", "DF"])
    benchmark_df["RF"] = rf_stat_df.mean(1)
    benchmark_df["SVM"] = svm_stat_df.mean(1)
    benchmark_df["LASSO"] = lasso_stat_df.mean(1)
    benchmark_df["MLPNN"] = mlpnn_stat_df.mean(1)
    benchmark_df["CNN1D"] = cnn1d_stat_df.mean(1)
    # benchmark_df["DF"] = gc_stat_df.mean(1)
    benchmark_df.to_csv(result_path + "/benchmark_raw.csv")
    benchmark_df

if __name__ == "__main__":
    train_baseline()