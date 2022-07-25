""" Linear classifiers for population data
"""


    # for n in range(numsplits):

    #     # === XVAL split
    #     from sklearn.model_selection import train_test_split
    #     Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size)


# def model_fit(X, y, version):
#   """ Fit model to training data X and y, with methods 



    # X, y = make_classification(random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                      random_state=0)


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# print(scaler.mean_)
# print(scaler.mean_.shape)

# scaler.transform(X)
import numpy as np


def _model_score(X, y, version="lin_svm", model_params={}, do_train_test_split=False, 
    niter=10, mean_score=False):
    """ Quickly fit and score model
    """
    list_scores = []
    for n in range(niter):
        mod, score = _model_fit(X, y, version, model_params, do_train_test_split)
        list_scores.append(score)
    if mean_score:
        return np.mean(list_scores)
    else:
        return list_scores


def _model_fit(X, y, version="lin_svm", model_params={}, do_train_test_split=False):
    """ 
    PARAMS:
    - X, (nsamp, nfeat), features
    - y, (nsamp, ), labels
    """
    import warnings
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    if version=="lin_svm":
        # Linear SVM
        mod = Pipeline([('scaler', StandardScaler()), (version, LinearSVC(**model_params))])
    else:
        assert False, "code it"

    # if modver=="logistic":
    #     reg = linear_model.LogisticRegression(solver="lbfgs", multi_class="multinomial")

    # - regression model
    # reg = linear_model.LinearRegression()

    # elif modver=="ridge":
    #     reg = linear_model.RidgeCV(alphas = np.logspace(-6, 6, 13))
    # elif modver=="ridgeclass":
    #     reg = linear_model.RidgeClassifierCV(alphas = np.logspace(-6, 6, 13))

    with warnings.catch_warnings():
        warnings.simplefilter('error') # to break if fails to converge
        if do_train_test_split:
            from sklearn.model_selection import train_test_split
            test_size=0.1
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size)
            mod.fit(Xtrain, ytrain)
            score = mod.score(Xtest, ytest)
        else:
            mod.fit(X, y)
            score = mod.score(X, y)

    return mod, score
