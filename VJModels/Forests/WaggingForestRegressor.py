from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.exceptions import DataConversionWarning

from datetime import datetime

import pandas as pd
import warnings
import math

# Suppress the warning
warnings.filterwarnings("ignore", category=DataConversionWarning)


class WaggingForestRegressor:
    def __init__(self, verbose=True, test_size=0.8, n_estimators_by_step=10, n_models=10, score_exponent=2, importance_func=None, n_models_to_keep='all', random_state=42):
        self.verbose = verbose
        self.test_size = test_size
        self.n_estimators_by_step = n_estimators_by_step
        self.n_models = n_models
        self.score_exponent = score_exponent
        self.importance_func = importance_func
        self.n_models_to_keep = n_models_to_keep
        self.random_state = random_state
        self.trees = []
        self.scores = []

    def log(self, *args):
        if self.verbose:
            print(*args)

    def fit(self, X, y):
        X_train = pd.DataFrame()
        y_train = pd.Series(dtype=int)
        
        X_test, y_test = X.copy(), y.copy()
        count = 0
        
        for _ in range (self.n_models):            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
            tree = RandomForestRegressor(n_estimators=self.n_estimators_by_step, random_state=self.random_state)
            tree.fit(X_train, y_train)
            
            score = tree.score(X_test, y_test)
            self.scores.append(score)
            self.trees.append(tree)
            
            self.log(f"Score {count}: {score:.4f}")            
            count += 1

        self.log(f"Trained {len(self.trees)} trees.")
        self.prune_models()

    def prune_models(self):
        if self.n_models_to_keep == 'all':
            return

        # Combine scores and trees into tuples
        scored_trees = list(zip(self.scores, self.trees))
        sorted_scored_trees = sorted(scored_trees, key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_trees = zip(*sorted_scored_trees)
        
        # Keep only the top n_models_to_keep scores and trees
        self.scores = sorted_scores[:self.n_models_to_keep]
        self.trees = sorted_trees[:self.n_models_to_keep]

    def get_importance(self, i):
        n = len(self.trees)
        score = self.scores[i]
        if self.importance_func is None:
            return score ** self.score_exponent
        else:
            return self.importance_func(i)

    def predict(self, X):
        predictions_list = [tree.predict(X) for tree in self.trees]
        importance_list = [self.get_importance(i) for i in range(len(self.trees))]

        # Calculate the weighted average of predictions
        results = [sum(importance * prediction for importance, prediction in zip(importance_list, preds)) / sum(importance_list) for preds in zip(*predictions_list)]
        
        return results
    

if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from datetime import datetime

    # Load California Housing dataset
    cali_housing = fetch_california_housing()
    df = pd.DataFrame(cali_housing.data, columns=cali_housing.feature_names)
    df['target'] = cali_housing.target

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    results = []

    test_size_options = [0.5, 0.8, 0.9]
    n_estimators_by_step_options = [10, 20]
    score_exponent_options = [0.5, 2, 3, 10]

    for test_size in test_size_options:
        for n_estimators_by_step in n_estimators_by_step_options:
            for score_exponent in score_exponent_options:
                # Wagging Forest
                start_time = datetime.now()
                inc_forest = WaggingForestRegressor(
                    verbose=False,
                    test_size=test_size,
                    n_estimators_by_step=n_estimators_by_step,
                    score_exponent=score_exponent,
                )
                inc_forest.fit(X_train, y_train)
                fit_time = (datetime.now() - start_time).total_seconds()

                start_time = datetime.now()
                y_train_pred = inc_forest.predict(X_train)
                y_test_pred = inc_forest.predict(X_test)
                predict_time = (datetime.now() - start_time).total_seconds()

                partial = {
                    "model": "WForest",
                    "testsize": test_size,
                    "estim by step": n_estimators_by_step,
                    "exponent": score_exponent,
                    "fitTime": fit_time,
                    "predictTime": predict_time,
                    "train MSE": mean_squared_error(y_train, y_train_pred),
                    "test MSE": mean_squared_error(y_test, y_test_pred),
                    "test r2": r2_score(y_test, y_test_pred)
                }
                print("partial", partial)

                results.append(partial)

    # Random Forest (as a baseline)
    random_forest = RandomForestRegressor(n_estimators=200, random_state=42)

    start_time = datetime.now()
    random_forest.fit(X_train, y_train)
    fit_time = (datetime.now() - start_time).total_seconds()

    start_time = datetime.now()
    y_train_pred = random_forest.predict(X_train)
    y_test_pred = random_forest.predict(X_test)
    predict_time = (datetime.now() - start_time).total_seconds()

    results.append({
        "model": "RandomForest",
        "testsize": None,
        "estim by step": None,
        "exponent": None,
        "fitTime": fit_time,
        "predictTime": predict_time,
        "train MSE": mean_squared_error(y_train, y_train_pred),
        "test MSE": mean_squared_error(y_test, y_test_pred),
        "test r2": r2_score(y_test, y_test_pred)
    })

    # Convert results to DataFrame and print
    results_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)  # Mostra todas as colunas
    pd.set_option('display.max_rows', None)     # Mostra todas as linhas

    print(results_df)