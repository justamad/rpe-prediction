from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def feature_elimination_xgboost(X_train: pd.DataFrame, y_train: pd.DataFrame,
                                X_test: pd.DataFrame, y_test: pd.DataFrame,
                                win_size: int, overlap: float, nr_features: int, path: str = None):
    model = XGBRegressor()
    model.fit(X_train, y_train)

    accuracy_train = mean_squared_error(y_train, model.predict(X_train))
    accuracy_test = mean_squared_error(y_test, model.predict(X_test))
    print(f"Accuracy Train={accuracy_train:.3f}, Test={accuracy_test:.3f}")
    thresholds = np.sort(np.unique(model.feature_importances_))

    # result = {}
    # for thresh in thresholds:
    #     selection = SelectFromModel(model, threshold=thresh, prefit=True)
    #     select_X_train = selection.transform(X_train)
    #
    #     # Train model
    #     selection_model = XGBRegressor()
    #     selection_model.fit(select_X_train, y_train)
    #
    #     # Evaluate model
    #     select_X_test = selection.transform(X_test)
    #     predictions = selection_model.predict(select_X_test)
    #     accuracy = mean_squared_error(y_test, predictions)
    #     print(f"Thresh={thresh:.6f}, n={select_X_train.shape[1]}, MSE: {accuracy:.4f}")
    #     result[select_X_train.shape[1]] = accuracy
    #
    # plt.plot(result.keys(), result.values())
    # plt.xlabel("Number of features")
    # plt.ylabel("MSE")
    # plt.title(f"Winsize={win_size}, Overlap={overlap}")
    # plt.tight_layout()
    #
    # if path is None:
    #     plt.show()
    # else:
    #     plt.savefig(join(path, f"feature_elimination_win_size_{win_size}_overlap={overlap}.png"))

    # plt.close()
    # plt.cla()
    # plt.clf()

    df = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=["Importance"]).sort_values(
        by='Importance', ascending=False)
    df.index.names = ["Feature"]
    df.to_csv(join(path, f"importance_winsize_{win_size}_overlap_{overlap}.csv"), sep=';')
    threshold = sorted(model.feature_importances_)[-(nr_features + 1)]
    mask = model.feature_importances_ > threshold
    return X_train.loc[:, mask].copy(), X_test.loc[:, mask].copy()

# def feature_elimination_rfe(X_train, y_train, X_test, y_test, nr_features: int):
#     estimator = XGBRegressor()
#     rfe = RFECV(estimator,
#                 min_features_to_select=nr_features,
#                 step=0.1,
#                 n_jobs=-1,
#                 verbose=10,
#                 scoring='neg_mean_squared_error',
#                 cv=logo.get_n_splits(groups=y_train['group']))
#
#     rfe.fit(X_train, y_train['rpe'])
#
#     # Save RFECV results for later
#     rfe_df = pd.DataFrame(rfe.ranking_, index=X_train.columns, columns=['Rank']).sort_values(by='Rank', ascending=True)
#     rfe_df.index.names = ["Feature"]
#     rfe_df.to_csv(join(out_path, f"features_win_{window_size}_overlap_{overlap}.csv"), sep=';')
