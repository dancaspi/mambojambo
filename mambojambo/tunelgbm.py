import math
import pickle

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#import lgbm_param_grids



def tuneOneParam(grid,param_name):
    gbm = GridSearchCV(estimator, param_grid=param_grid,
                             cv=10 if folds is None else folds, scoring=scoring, verbose=10, fit_params={})  # n_jobs=3#,
    #todo: complete
    return
    # gbm.fit(X_train, y_train.values)
    # bestidx = getBestParamsFromGrid(gbm)
    # ret = gbm.cv_results_['params'][bestidx][param_name]
    # return ret ## value of the best parameter considering mean test score + std


def getBestParamsFromGrid(gridresults):
    test_score=gridresults.cv_results_['mean_test_score']
    test_std=gridresults.cv_results_['std_test_score']
    bestidx=np.argmax(test_score-test_std)
    return gridresults.cv_results_['params'][bestidx]




def binary_error_feval(preds,train_data):
    labels = train_data.get_label()
    return 'error', np.mean(labels != (preds > 0.5)), False

def average_precision_feval(preds,train_data):
    labels = train_data.get_label()
    return 'error', average_precision_score(y_true=labels,y_score=preds), False


def tuneLgbm(X_train,y_train,estimator,scoring,folds=5):
    hist_size=4096
    numLeafRange = [4,8,16,32]
    n_estimators = 100
    param_grid = {
        'histogram_pool_size':[hist_size],
        'learning_rate': [0.02],
        'n_estimators': [n_estimators],
        'num_leaves': numLeafRange,
        'boosting_type': ['gbdt'],
        # 'objective': ['binary'],
        'random_state': [777],
        "min_data_in_leaf": [1],

        'feature_fraction': [1],
        'bagging_fraction': [1.0],
        'subsample': [1.0],
        'max_bin': [100],
        'is_unbalance': [True],
    }

    gbm = GridSearchCV(estimator, param_grid=param_grid,
                             cv=folds, scoring=scoring, verbose=10, fit_params={})  # n_jobs=3#,

    gbm.fit(X_train, y_train)
    nleaves = getBestParamsFromGrid(gbm)['num_leaves'] ## chosen number of leaves
    ## now tune other factors
    param_grid['num_leaves']= [nleaves]
    param_grid['feature_fraction'] = [1, 0.9, 0.95, 0.8,0.7]

    gbm = GridSearchCV(estimator, param_grid=param_grid,
                       cv=folds, scoring=scoring, verbose=10, fit_params={})  # n_jobs=3#,

    gbm.fit(X_train, y_train)


    feature_fraction = getBestParamsFromGrid(gbm)['feature_fraction']  ## chosen number of leaves
    param_grid['feature_fraction']=[feature_fraction]
    param_grid['min_data_in_leaf']=[1,5,15,25,50]

    gbm = GridSearchCV(estimator, param_grid=param_grid,
                             cv=folds, scoring=scoring, verbose=10, fit_params={})  # n_jobs=3#,

    gbm.fit(X_train, y_train)
    min_data_in_leaf = getBestParamsFromGrid(gbm)['min_data_in_leaf']
    param_grid['min_data_in_leaf'] = [min_data_in_leaf]

    param_grid['subsample']=[1,0.9,0.95,0.85]#,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3]
    gbm = GridSearchCV(estimator, param_grid=param_grid,
                            cv=folds, scoring=scoring, verbose=10, fit_params={})  # n_jobs=3#,

    gbm.fit(X_train, y_train)

    subsample = getBestParamsFromGrid(gbm)['subsample'] ## chosen number of leaves
    param_grid['subsample']=[subsample]


    lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',
                                       objective='binary',
                                       learning_rates=lambda iter: 0.02 * (0.99 ** iter)
                                       )

    lgb_train = lgb.Dataset(X_train, np.array(y_train).reshape(y_train.shape[0], ), free_raw_data=False)
    param_grid.pop('n_estimators')
    validation_summary = lgb.cv(params=param_grid,
                                nfold= None if isinstance(folds,list) else folds,
                                folds = folds if isinstance(folds,list) else None,
                                num_boost_round=4096,
                                train_set=lgb_train,
                                early_stopping_rounds=300,
                                metrics=["auc"], ## need to add pr-auc to lgbm, for now will use this
                                #feval=,
                                verbose_eval=10
                                )

    optimal_num_trees = len(validation_summary["auc-mean"])
    param_grid['n_estimators']=optimal_num_trees
    return param_grid


def randomTuneLgbm(X_train,y_train,estimator,scoring,folds=None,boosting_type='gbdt',objective='binary',outdir=None,ngrid=20):
    if outdir is None:
        outdir = 'lgbmstats_'+boosting_type+'/'
    param_grid = None
    if boosting_type == 'gbdt':
        param_grid = lgbm_param_grids.gbdt_param_grid

    elif boosting_type == 'rf':
        param_grid = lgbm_param_grids.rf_param_grid
        param_grid['feature_fraction'] = [1 / math.sqrt(X_train.shape[1])]
        param_grid['n_estimators'] = [1000]

    assert (param_grid is not None)
    param_grid['objective'] = [objective]

    gbm = RandomizedSearchCV(estimator, param_distributions=param_grid,
                             n_iter=ngrid, cv=10 if folds is None else folds, scoring=scoring, verbose=10,
                             fit_params={})  # n_jobs=3#,

    gbm.fit(X_train, y_train)
    # gbm.fit(X_train, np.array(y_train).reshape(y_train.shape[0],))#,{ 'group':np.array([X_train.shape[0]])})

    f = open(outdir + "CVStatsByAuc.txt", "w")
    gridScores = gbm.cv_results_.copy()
    adjustedScores = list(gridScores['mean_test_score'] - gridScores['std_test_score'])
    sortedScores = adjustedScores.copy()
    sortedScores.sort(reverse=True)
    bestModelIdx = adjustedScores.index(sortedScores[0])  # best model adjusted to stdev

    unadjustedScores = list(gridScores['mean_test_score'])
    sortedScores = unadjustedScores.copy()
    sortedScores.sort(reverse=True)
    bestNonAdjustedModelIdx = unadjustedScores.index(unadjustedScores[0])  # best model adjusted to stdev

    f.write(
        "mean test score for best model (adjusted to stdev):%0.5f, stdev:%0.5f\n---------------------------------------------------------------------------\n" % (
            adjustedScores[bestModelIdx], gridScores['std_test_score'][bestModelIdx]))
    modelIdx = list(gridScores['rank_test_score'])
    modelIdx.reverse()
    for idx in modelIdx:
        idx = idx - 1  ### rank is one index, data saved zero-indexes
        # print(idx)
        f.write(str(gridScores['params'][idx]) + "\n"
                +
                "cv auc: " + str(gridScores['mean_test_score'][idx]) + "\n"
                +
                "cv auc std: " + str(gridScores['std_test_score'][idx]) + "\n"
                +
                "*********************************\n")

    f.close()

    params = gbm.best_params_.copy()
    pickle.dump(params, open(outdir + "bestParams.pkl", "wb"))
    return params

