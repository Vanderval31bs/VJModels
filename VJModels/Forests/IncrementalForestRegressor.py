from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import math

class IncrementalForestRegressor:
    def __init__(self, verbose=True, test_size=0.8, n_estimators_by_step=10, max_n_forests=30, min_test_size=25, max_score_diff=0.05, importance_func='cubic', test_score_goal=1.0, random_state=42):
        self.verbose = verbose
        self.test_size = test_size
        self.n_estimators_by_step = n_estimators_by_step
        self.max_n_forests = max_n_forests
        self.min_test_size = min_test_size
        self.max_score_diff = max_score_diff
        self.importance_func = importance_func
        self.test_score_goal = test_score_goal
        self.random_state = random_state
        self.trees = []
        self.scores = []

    def log(self, *args):
        if self.verbose:
            print(*args)

    def fit_one_tree(self, X_train_prev, X_test_prev, y_train_prev, y_test_prev):
        X_test_prev_A, X_test_prev_B, y_test_prev_A, y_test_prev_B = train_test_split(
            X_test_prev, y_test_prev, test_size=self.test_size, random_state=self.random_state)
        
        X_train = pd.concat([X_train_prev, X_test_prev_A], ignore_index=True)
        y_train = pd.concat([y_train_prev, y_test_prev_A], ignore_index=True)

        tree = RandomForestRegressor(n_estimators=self.n_estimators_by_step, random_state=self.random_state)
        tree.fit(X_train, y_train)
        
        return tree, X_train, X_test_prev_B, y_train, y_test_prev_B
    
    def score_diff_is_below_max(self, iteration):
        if iteration < 2 or len(self.scores) <= iteration - 1:
            return True
        else:
            return math.fabs(max(self.scores[:-1]) - self.scores[-1]) < self.max_score_diff

    def fit(self, X, y):
        X_train = pd.DataFrame()
        y_train = pd.Series(dtype=float)
        
        X_test, y_test = X.copy(), y.copy()
        count = 0

        while len(X_test) >= self.min_test_size and self.score_diff_is_below_max(count) and len(self.trees) < self.max_n_forests:
            tree, X_train, X_test, y_train, y_test = self.fit_one_tree(X_train, X_test, y_train, y_test)
            
            score = tree.score(X_test, y_test)
            self.scores.append(score)
            self.trees.append(tree)
            
            self.log(f"Score {count}: {score:.4f}")
            if score >= self.test_score_goal:
                break
            
            count += 1

        self.log(f"Trained {len(self.trees)} trees.")

    def get_importance(self, i):
        n = len(self.trees)
        score = self.scores[i]
        importance_func_map = {
            'cubic': lambda: (n - i) * score ** 3,
            'square': lambda: (n - i) * score ** 2,
            'linear': lambda: (n - i) * score
        }
        return importance_func_map.get(self.importance_func, lambda: self.importance_func(i, n, score))()

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

    test_size_options = [0.5]
    n_estimators_by_step_options = [10]
    importance_func_options = ['cubic']
    min_test_size_options = [25]
    max_score_diff_options = [0.1]

    for test_size in test_size_options:
        for n_estimators_by_step in n_estimators_by_step_options:
            for importance_func in importance_func_options:
                for min_test_size in min_test_size_options:
                    for max_score_diff in max_score_diff_options:
                        # Incremental Forest
                        start_time = datetime.now()
                        inc_forest = IncrementalForestRegressor(
                            verbose=False,
                            test_size=test_size,
                            n_estimators_by_step=n_estimators_by_step,
                            importance_func=importance_func,
                            min_test_size=min_test_size,
                            max_score_diff=max_score_diff
                        )
                        inc_forest.fit(X_train, y_train)
                        fit_time = (datetime.now() - start_time).total_seconds()

                        start_time = datetime.now()
                        y_train_pred = inc_forest.predict(X_train)
                        y_test_pred = inc_forest.predict(X_test)
                        predict_time = (datetime.now() - start_time).total_seconds()

                        partial = {
                            "model": "IForest",
                            "testsize": test_size,
                            "estim by step": n_estimators_by_step,
                            "imp func": importance_func,
                            "min testsize": min_test_size,
                            "maxscore diff": max_score_diff,
                            "fitTime": fit_time,
                            "predictTime": predict_time,
                            "train MSE": mean_squared_error(y_train, y_train_pred),
                            "test MSE": mean_squared_error(y_test, y_test_pred)
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
        "imp func": None,
        "min testsize": None,
        "maxscore diff": None,
        "fitTime": fit_time,
        "predictTime": predict_time,
        "train MSE": mean_squared_error(y_train, y_train_pred),
        "test MSE": mean_squared_error(y_test, y_test_pred)
    })

    # Convert results to DataFrame and print
    results_df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)     # Show all rows

    print(results_df)