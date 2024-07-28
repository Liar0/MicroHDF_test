import sys
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from utils.benchemarks_io import get_stat_dict, get_stat
from sklearn.metrics import roc_curve
from models.GCForest import *
from models.GCForest import gcForest
# from gcforest.gcforest import GCForest
from sklearn.model_selection import StratifiedKFold


class DF():
    def __init__(self, config):
        self.num_trees = int(config.get('Benchmark', 'NumberTrees'))
        self.num_cascade = int(config.get('Benchmark', 'NumCascadeRF'))
        self.num_valid_models = int(config.get('Benchmark', 'ValidationModels'))
        self.window = int(config.get('Benchmark', 'WindowSize'))

        self.cascade_forest = gcForest(shape_1X=self.window, window=[self.window],
                                       n_mgsRFtree=self.num_trees, stride=1,
                                       cascade_test_size=0.2, n_cascadeRF=self.num_cascade,
                                       n_cascadeRFtree=101, cascade_layer=5,
                                       min_samples_mgs=0.1, min_samples_cascade=0.05,
                                       tolerance=0.0, n_jobs=-1)
    #     cascade_layer=np.inf
    # gcf = gcForest(cascade_forest={"random_state": 1, "n_mgsRFtree": 4,
    #                                "n_cascadeRF": 2, "n_jobs": -1, "verbose": 1},
    #                gc_fly={"window": 2})
    gcf = gcForest(shape_1X=2, window=100, stride=1, cascade_test_size=0.2,
                   n_mgsRFtree=4, n_cascadeRF=2, min_samples_mgs=0.1,
                   min_samples_cascade=0.05, tolerance=0.0, n_jobs=-1)
    def train(self, train, seed=42):

        x, y = train
        skf = StratifiedKFold(n_splits=self.num_valid_models, shuffle=True)

        if len(np.unique(y)) == 2:
            metric = "AUC"
        else:
            metric = "MCC"
        best_score = -1

        for percent in [0.25, 0.5, 0.75, 1.0]:
            run_score = -1
            run_probs = []
            for train_index, valid_index in skf.split(x, y):
                train_x, valid_x = x[train_index], x[valid_index]
                train_y, valid_y = y[train_index], y[valid_index]

                # self.cascade_forest.fit_transform(train_x, train_y)
                self.cascade_forest.fit(train_x, train_y)
                prob_predict = self.cascade_forest.predict_proba(valid_x)
                run_probs = list(run_probs) + list(prob_predict)
            run_score = get_stat(y, run_probs, metric)

            # if run_score > best_score:
            #     best_num_features = self.cascade_forest.max_features_
            #     best_score = run_score
            #     self.best_num_features = best_num_features

        return

    def test(self, test):
        x, y = test
        prob_predict = self.cascade_forest.predict_proba(x)
        preds = np.argmax(prob_predict, axis=-1)
        stat = get_stat_dict(y, prob_predict)
        return preds, stat
