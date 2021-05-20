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

# Define a pipeline to search for the best combination of PCA truncation and classifier regularization.
# Set the tolerance to a large value to make the example faster
tuned_parameters = [{'svr__kernel': ['rbf'],
                     'svr__gamma': [1e-3, 1e-4],
                     'svr__C': [1e0, 1e1, 1e2, 1e3]}]

pipe = Pipeline(steps=[('scaler', StandardScaler()), ('svr', SVR())])

search = GridSearchCV(estimator=pipe,
                      param_grid=tuned_parameters,
                      cv=logo.get_n_splits(groups=groups),
                      n_jobs=-1,
                      verbose=10)

print(search)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
