from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import DataConversionWarning

from datetime import datetime

import pandas as pd
import warnings
import math

# Suppress the warning
warnings.filterwarnings("ignore", category=DataConversionWarning)


class IncrementalForestClassifier:
    def __init__(self, verbose=True, test_size=0.8, n_estimators_by_step=10, max_n_forests=30, min_test_size=10, max_score_diff=0.05, importance_func='square', test_score_goal=1, random_state=42):
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
        
        # Avoid repeated concatenations
        X_train = pd.concat([X_train_prev, X_test_prev_A], ignore_index=True)
        y_train = pd.concat([y_train_prev, y_test_prev_A], ignore_index=True)

        tree = RandomForestClassifier(n_estimators=self.n_estimators_by_step, random_state=self.random_state)
        tree.fit(X_train, y_train)
        
        return tree, X_train, X_test_prev_B, y_train, y_test_prev_B
    
    def score_diff_is_below_max(self, iteration):
        if iteration < 2 or len(self.scores) <= iteration - 1:
            return True
        else:
            return math.fabs(max(self.scores[:-1]) - self.scores[-1]) < self.max_score_diff

    def fit(self, X, y):
        X_train = pd.DataFrame()
        y_train = pd.Series(dtype=int)
        
        X_test, y_test = X.copy(), y.copy()
        count = 0
        
        while len(X_test) >= self.min_test_size and self.score_diff_is_below_max(count) and len(self.trees) < self.max_n_forests:
            tree, X_train, X_test, y_train, y_test = self.fit_one_tree(X_train, X_test, y_train, y_test)
            
            score = accuracy_score(y_test, tree.predict(X_test))
            self.scores.append(score)
            self.trees.append(tree)
            
            self.log(f"Score {count}: {score:.4f}")
            if (score >= self.test_score_goal):
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
        return int(importance_func_map.get(self.importance_func, lambda: self.importance_func(i, n, score))())

    def predict(self, X):
        n_samples = len(X)
        results = [0] * n_samples

        predictions_list = [tree.predict(X) for tree in self.trees]
        importance_list = [self.get_importance(i) for i in range(len(self.trees))]

        for i in range(n_samples):
            results[i] = sum(importance if prediction[i] == 1 else -importance
                             for prediction, importance in zip(predictions_list, importance_list))

        return [1 if result > 0 else 0 for result in results]
    

if __name__ == "__main__":
    df = pd.read_csv('cardio.csv', sep=";")

    X = df.drop(columns=['cardio'])
    y = df['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    results = []

    # Parameter grids
    test_size_options = [0.8]
    n_estimators_by_step_options = [10]
    importance_func_options = ['cubic', 'square']
    min_test_size_options = [10, 25]
    max_score_diff_options = [0.05, 0.01]

    for test_size in test_size_options:
        for n_estimators_by_step in n_estimators_by_step_options:
            for importance_func in importance_func_options:
                for min_test_size in min_test_size_options:
                    for max_score_diff in max_score_diff_options:
                        # Incremental Forest
                        start_time = datetime.now()
                        inc_forest = IncrementalForestClassifier(
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
                            "Model": "Incremental Forest",
                            "Test Size": test_size,
                            "N Estimators By Step": n_estimators_by_step,
                            "Importance Func": importance_func,
                            "min_test_size": min_test_size,
                            "max_score_diff": max_score_diff,
                            "Fit Time (s)": fit_time,
                            "Predict Time (s)": predict_time,
                            "Train Accuracy": accuracy_score(y_train, y_train_pred),
                            "Test Accuracy": accuracy_score(y_test, y_test_pred)
                        }
                        print("partial", partial)

                        results.append(partial)

    # Random Forest (as a baseline)
    random_forest = RandomForestClassifier(n_estimators=200, random_state=42)

    start_time = datetime.now()
    random_forest.fit(X_train, y_train)
    fit_time = (datetime.now() - start_time).total_seconds()

    start_time = datetime.now()
    y_train_pred = random_forest.predict(X_train)
    y_test_pred = random_forest.predict(X_test)
    predict_time = (datetime.now() - start_time).total_seconds()

    results.append({
        "Model": "Random Forest",
        "Test Size": None,
        "Importance Func": None,
        "min_test_size": None,
        "max_score_diff": None,
        "Fit Time (s)": fit_time,
        "Predict Time (s)": predict_time,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred)
    })

    # Convert results to DataFrame and print
    results_df = pd.DataFrame(results)
    print(results_df)