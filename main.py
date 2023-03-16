# -----------------------------------------------------------------------------
#   PYTHON SCRIPT ASSESSING THE INFLUENCE OF RANDOM STATES ON CLASSIFICATIONS
# -----------------------------------------------------------------------------

# (C) Nicolas Rost, 2023


# ------------------------------- Packages ------------------------------------
import numpy as np
import pandas as pd
from skopt.space import Real
from skopt import BayesSearchCV
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


# ---------------------------- Initializations --------------------------------
n_samples = 5000 # number of samples
n_features = 50 # number of features
n_seeds = 500 # number of random states
steps = 10 # steps for informative features
folds = 5 # CV folds
n_params = 500 # number of parameter combinations

# for loops
random_states = range(n_seeds)
informativeness = np.arange(0, n_features, steps) 
informativeness[0] += 2 # 0 informative features lead to error

# parameter space
parameters = {'m__C': Real(0.001, 100, prior = 'log-uniform'),
              'm__l1_ratio': Real(0, 1, prior = 'uniform')}

# scaler
scl = StandardScaler()

# results dict
res = pd.DataFrame({'informativeness': np.repeat(informativeness, len(random_states)),
                    'random_state': np.tile(random_states, len(informativeness)),
                    'BAC': None})


# ---------------------------- Start Analyses ---------------------------------
for inf in informativeness:
    
    # create and split data
    X, y = make_classification(n_samples = n_samples,
                               n_features = n_features,
                               n_informative = inf,
                               n_classes = 2,
                               random_state = 0)
    if inf == 2:
        X = shuffle(X, random_state = 0) # shuffle to create 0 informative features
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    for rs in random_states:
        
        # model and fitting
        m = LogisticRegression(penalty = 'elasticnet', 
                               solver = 'saga',
                               max_iter = 10000,
                               random_state = rs)
        pipe = Pipeline([('scl', scl),
                         ('m', m)])
        clf = BayesSearchCV(pipe, 
                            parameters,
                            cv = folds,
                            n_iter = n_params,
                            scoring = 'balanced_accuracy', 
                            return_train_score = True, 
                            random_state = rs,
                            n_jobs = -1)
        clf.fit(Xtrain, ytrain)
        
        # evaluation
        ypred = clf.predict(Xtest)
        bac = balanced_accuracy_score(ytest, ypred)
        res.loc[(res['informativeness'] == inf) & (res['random_state'] == rs), 'BAC'] = bac

        print(f"Loop with {inf} informative features and random state {rs} done.")

res['informativeness'] = res['informativeness'].replace({2:0})

# save results
res.to_csv('classification_results_bac.csv', index = False)
