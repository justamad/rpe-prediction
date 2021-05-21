from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import pandas as pd

# Read in train and test data
X = pd.read_csv("x.csv", sep=";").to_numpy()
y_labels = pd.read_csv("y.csv", sep=";")
y = y_labels['rpe'].to_numpy()
groups = y_labels['groups'].to_numpy()

logo = LeaveOneGroupOut()

models = [('svr', SVR())]

tuned_parameters = [{'svr__kernel': ['linear'],  # , 'rbf'],
                     'svr__gamma': [1e-3],  # , 1e-4],
                     'svr__C': [1e0]}]  # , 1e1 , 1e2, 1e3]}]

# Main Loop: iterate over learning models
for learner_name, learner in models:
    pipe = Pipeline(steps=[('scaler', StandardScaler()), (learner_name, learner)])

    search = GridSearchCV(estimator=pipe,
                          param_grid=tuned_parameters,
                          cv=logo.get_n_splits(groups=groups),
                          n_jobs=-1,
                          verbose=10)

    print(search)
    search.fit(X, y)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(f"{learner_name}_results.csv", sep=';')
