'''
Source : https://ruivieira.dev/counterfactual-fairness.html
'''
import numpy as np
from sklearn import metrics
import pymc3 as pm
import torch

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBRegressor
except:
    print('no xgboost')


def MCMC(data, A, samples=1000):
    N = len(data)
    a = np.array(data[A])
    K = len(A)
    # a = A
    # K = A.shape[1]
    model = pm.Model()

    with model:
        # Priors
        k = pm.Normal("k", mu=0, sigma=1, shape=(1, N))
        gpa0 = pm.Normal("gpa0", mu=0, sigma=1)
        lsat0 = pm.Normal("lsat0", mu=0, sigma=1)
        w_k_gpa = pm.Normal("w_k_gpa", mu=0, sigma=1)
        w_k_lsat = pm.Normal("w_k_lsat", mu=0, sigma=1)
        w_k_zfya = pm.Normal("w_k_zfya", mu=0, sigma=1)

        w_a_gpa = pm.Normal("w_a_gpa", mu=np.zeros(K), sigma=np.ones(K), shape=K)
        w_a_lsat = pm.Normal("w_a_lsat", mu=np.zeros(K), sigma=np.ones(K), shape=K)
        w_a_zfya = pm.Normal("w_a_zfya", mu=np.zeros(K), sigma=np.ones(K), shape=K)

        sigma_gpa_2 = pm.InverseGamma("sigma_gpa_2", alpha=1, beta=1)

        mu = gpa0 + (w_k_gpa * k) + pm.math.dot(a, w_a_gpa)

        # Observed data
        gpa = pm.Normal(
            "gpa",
            mu=mu,
            sigma=pm.math.sqrt(sigma_gpa_2),
            observed=list(data["UGPA"]),
            shape=(1, N),
        )
        lsat = pm.Poisson(
            "lsat",
            pm.math.exp(lsat0 + w_k_lsat * k + pm.math.dot(a, w_a_lsat)),
            observed=list(data["LSAT"]),
            shape=(1, N),
        )
        zfya = pm.Normal(
            "zfya",
            mu=w_k_zfya * k + pm.math.dot(a, w_a_zfya),
            sigma=1,
            observed=list(data["ZFYA"]),
            shape=(1, N),
        )

        step = pm.Metropolis()
        trace = pm.sample(samples, step, progressbar=True)

    return trace


def supervised_model_training(x_train, y_train, model_name):
    """Train supervised learning models and report the results.

    Args:
      - x_train, y_train: training dataset
      - x_test, y_test: testing dataset
      - model_name: supervised model name such as logisticregression

    Returns:
      - auc: Area Under ROC Curve
      - apr: Average Precision and Recall
    """
    if model_name == 'linearregression':
        model = LinearRegression()
        model.fit(x_train, y_train)
    if model_name == 'Ridge':
        model = Ridge(alpha=0.01)
        model.fit(x_train, y_train)
    if model_name == 'logisticregression':
        model = LogisticRegression()
        model.fit(x_train, y_train)
    elif model_name == 'randomforest':
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
    elif model_name == 'randomforestReg':
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
    elif model_name == 'gaussiannb':
        model = GaussianNB()
        model.fit(x_train, y_train)
    elif model_name == 'bernoullinb':
        model = BernoulliNB()
        model.fit(x_train, y_train)
    elif model_name == 'multinb':
        model = MultinomialNB()
        model.fit(x_train, y_train)
    elif model_name == 'svmlin':
        model = svm.LinearSVC()
        model.fit(x_train, y_train)
    elif model_name == 'gbm':
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train)
    elif model_name == 'Extra Trees':
        model = ExtraTreesClassifier(n_estimators=20)
        model.fit(x_train, y_train)
    elif model_name == 'LDA':
        model = LinearDiscriminantAnalysis()
        model.fit(x_train, y_train)
    elif model_name == 'Passive Aggressive':
        model = PassiveAggressiveClassifier()
        model.fit(x_train, y_train)
    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier()
        model.fit(x_train, y_train)
    elif model_name == 'Bagging':
        model = BaggingClassifier()
        model.fit(x_train, y_train)
    elif model_name == 'xgb':
        model = XGBRegressor()
        model.fit(np.asarray(x_train), y_train)

    return model


def model_test(model, x_test, y_test, model_name):
    if model_name == 'svmlin' or model_name == 'Passive Aggressive':
        predict = model.decision_function(x_test)
    elif model_name == 'xgb':
        predict = model.predict(np.asarray(x_test))
    else:
        predict = model.predict_proba(x_test)[:, 1]

    # AUC / AUPRC Computation
    acc = metrics.accuracy_score(y_test, predict > 0.5)
    auc = metrics.roc_auc_score(y_test, predict)
    apr = metrics.average_precision_score(y_test, predict)

    return acc, auc, apr


# Batch of CF sensitive attribute for current batch of A

def law_cf_sensitive(A):
    batch = A.shape[0]

    perm = torch.randint(1, 7, (batch,))
    for i in range(batch):
        A[i, :] = torch.roll(A[i, :], shifts=int(perm[i]))

    print(A[0, :])
    return A


def cf_sample(a, k, norm):
    k = np.random.normal(k, 0.2)
    gpa0 = 1.7868
    lsat0 = 1.54263
    w_k_gpa = 0.06966
    w_k_lsat = 0.07639
    w_k_zfya = -0.062212

    w_a_gpa = np.array([-1.02179471, 1.35733018, 0.92978805, 1.30527247, 0.88830321, 2.19190978,
                        0.82551243, 1.40430065, 0.08848209, 0.0673493])
    w_a_lsat = np.array([-2.25080326, -0.73727328, -0.82029424, -0.74885257, -0.94447789, -1.01934862,
                         -0.26419691, -0.70994488, 2.75623623, 2.79232874])
    w_a_zfya = np.array([-0.96818263, -0.11085893, -0.86371096, -0.12763885, -0.06871432, -0.39684417,
                         -0.63694664, 0.3595703, -0.1071931, -0.03830721])
    sigma_gpa_2 = 0.2576

    mu_cf = gpa0 + (w_k_gpa * k) + np.dot(a, w_a_gpa)
    gpa_cf = np.random.normal(mu_cf, np.sqrt(sigma_gpa_2))
    lsat_cf = np.random.poisson(np.exp(lsat0 + w_k_lsat * k + np.dot(a, w_a_lsat)))

    s_cf = np.concatenate((a[0:8], np.array(gpa_cf).reshape(1), np.array(lsat_cf).reshape(1), a[8:]))
    s_cf[8:10] = (s_cf[8:10] - norm[1]) / norm[2]
    zfya_cf = np.random.normal(w_k_zfya * k + np.dot(a, w_a_zfya), 1)

    return s_cf.reshape(1, -1), zfya_cf


