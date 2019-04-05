import os
import pickle
import lightgbm as lgb
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
import tunelgbm

class LgbmModelWrapper():

    def predict(self,x):
        x=self.preprocessingFunc(x)
        return self.raw_model.predict(x)

    def preprocessingFunc(self,x):
        return np.round(x, 7)

    def __init__(self,model):
        self.raw_model = model



def save_model_stats(model,features,boosting_type='gbdt',outdir=None):

    outdir = 'lgbmstats_'+boosting_type+'/' if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)
    print('Calculate feature importances...')
    # feature importances
    feature_importance = list(model.feature_importance(importance_type='split'))
    imp = {features[i]: int(feature_importance[i]) for i in range(len(features))}
    imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    f = open(outdir+"feature_importance_split.txt", "w")
    for ff in imp:
        f.write(ff[0] + "," + str(ff[1]) + "\n")
    f.close()

    feature_importance = list(model.feature_importance(importance_type='gain'))
    imp = {features[i]: int(feature_importance[i]) for i in range(len(features))}
    imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    f = open(outdir+"feature_importance_gain.txt", "w")
    for ff in imp:
        f.write(ff[0] + "," + str(ff[1]) + "\n")
    f.close()

    aa = pd.read_csv(outdir+"feature_importance_gain.txt", header=None)

    aa.head(n=50).plot(x=0, y=1, kind="barh", figsize=(10, 10))
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.savefig(outdir+'feature_importance_gain.png')

    aa = pd.read_csv(outdir+"feature_importance_split.txt", header=None)
    aa.head(n=50).plot(x=0, y=1, kind="barh", figsize=(10, 10))
    plt.tight_layout()
    plt.ylabel('Feature')
    plt.savefig(outdir+'feature_importance_split.png')



def train(data=None, X_train=None, y_train=None, X_test=None,y_test=None, boosting_type='gbdt', objective='binary', outdir='Lgbm/',
          ngrid=20, folds=2, retune=True, randomGridSearch=False, crossValScore=False):

    #outdirpostfix = '_' + boosting_type if outdirpostfix is None else outdirpostfix
    #outdir = 'lgbmstats' + outdirpostfix+'/'

    random.seed(a=11)
    np.random.seed(11)  ## before random sampling, fix a seed for replicability
    os.makedirs(outdir, exist_ok=True)

    if data is not None:
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

    #X_train = X_train[sorted(list(X_train))]
    X_train = np.round(X_train, 4)
    try:
        #X_test = X_test[sorted(list(X_test))]
        X_test = np.round(X_test, 4)
    except:
        print('No X_test provided')

    estimators = {
        'lambdarank': lgb.LGBMRanker(silent=True, objective='lambdarank', score='ndcg'),
        'binary': lgb.LGBMClassifier(silent=True, objective='binary'),
        'regression': lgb.LGBMRegressor(silent=True, objective='regression'),

    }

    scorings = {
        'lambdarank': 'ndcg',
        'binary': 'roc_auc',
        'regression': 'neg_mean_squared_error'
    }

    estimator = estimators.get(objective)

    scoring = scorings.get(objective)

    if retune:

        if randomGridSearch:
            params = tunelgbm.randomTuneLgbm(X_train, y_train, estimator, scoring, folds=folds, boosting_type=boosting_type,
                                             objective=objective, outdir=outdir, ngrid=ngrid)
            pickle.dump(params, open(outdir + "bestParams.pkl", "wb"))
        else:
            params = tunelgbm.tuneLgbm(X_train, y_train, estimator, scoring,folds=folds)
            pickle.dump(params, open(outdir + "bestParams.pkl", "wb"))

    params = pickle.load(open(outdir+"bestParams.pkl","rb"))


    skestimator = estimators.get(objective)
    skestimator.__init__(**params) ## initialize with the found parameters, for oof predictions
    skestimator.fit(X_train,y_train)
    oofpreds = None
    if folds is not None and crossValScore:
        oofpreds = cross_val_predict(skestimator, X_train, y_train, cv=folds, method='predict_proba')[:,1]
    n_estimators=params.pop('n_estimators')

    lgb_train = lgb.Dataset(X_train, np.array(y_train).reshape(y_train.shape[0], ), free_raw_data=False)
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=n_estimators
                      )

    model.save_model(outdir+"model.txt")

    model = lgb.Booster(model_file=outdir+'model.txt') ## this will be used in production, so we always use this here
    save_model_stats(model, list(X_train), boosting_type=boosting_type, outdir=outdir)
    lgbmModelWrapper = LgbmModelWrapper(model)

    ## if we get X_test also make predictions and return them
    if X_test is not None:
        y_pred = lgbmModelWrapper.predict(X_test)
        y_pred = pd.DataFrame(y_pred,index=X_test.index)
        print(len(y_pred))
    else:
        y_pred = None

    oofpreds = pd.Series(oofpreds, index=X_train.index)


    #y_pred_train = pd.Series(model.predict(X_train[sorted(list(X_train))]), index=X_train.index)

    ret =  {'model': lgbmModelWrapper,
            'rawModel': model,
            'y_pred': y_pred.loc[:,0], ## remove the empty header we have there, it annoys using this as indexer
            'y_test': y_test if X_test is not None else None,
            'y_test_binary':y_test,
            'model_params': pickle.load(open(outdir+"bestParams.pkl", "rb")),
            'oofpreds': oofpreds,
            'y_train': y_train,
            #'y_pred_train': y_pred_train,
            'skestimator': skestimator
            }

    return ret


