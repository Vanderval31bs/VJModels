# 🧪 VJModels

A collection of my experimental machine learning models. These models are part of my personal exploration in the field, so they might not be fully refined, but they contain some interesting ideas. Feel free to check them out! You can also install the package via [pip](https://pypi.org/project/VJModels/) and incorporate the models into your own projects.


```bash
pip install VJModels
```

**Example usage:**

```python
from VJModels.Forests import IncrementalForestClassifier

# X_train, y_train, X_test should be your datasets

inc_forest = IncrementalForestClassifier()
inc_forest.fit(X_train, y_train)
y_test_pred = inc_forest.predict(X_test)
```

# Summary

1. Forests
   - [1.1 WSagging](#wsagging)
   - [1.2 Incremental](#incremental)

# Forests

## WSagging

**WSagging** is a term I coined, standing for **Weighted Score Averaging**. The idea is quite simple. Suppose you have `X` features and `y` targets. You select a number of times the algorithm will run (`n_models` parameter in the class constructor). At each iteration, you randomly split the dataset `X` into two different datasets: `X_train` and `X_validation`. The model is trained on the `X_train` dataset and scored on `X_validation` against `y_validation`. Both the trained model and its score are saved.

Later, during the prediction phase, you average the predictions based on the scores from the validation set to obtain a new prediction for your data. Specifically, the classifier and regressor work as follows:

### Classifier

```python
def predict(self, X):
    n_samples = len(X)
    results = [0] * n_samples

    predictions_list = [tree.predict(X) for tree in self.trees]
    importance_list = [self.get_importance(i) for i in range(len(self.trees))]

    for i in range(n_samples):
        results[i] = sum(importance if prediction[i] == 1 else -importance
                         for prediction, importance in zip(predictions_list, importance_list))

    return [1 if result > 0 else 0 for result in results]
```

- **`predictions_list`**: This contains the predictions made by all trees in the forest for each sample.
- **`importance_list`**: This holds the importance score for each tree, reflecting how much weight each tree's prediction carries.
- **Weighted Sum Calculation**:
  - For each sample, calculate the weighted sum of predictions from all trees.
  - Add the importance score for positive predictions (`1`) and subtract the importance score for negative predictions (`0`).
- **Final Classification**:
  - If the resulting weighted sum is positive, classify the sample as `1`.
  - If the weighted sum is zero or negative, classify the sample as `0`.

  ### Regressor

```python
def predict(self, X):
    predictions_list = [tree.predict(X) for tree in self.trees]
    importance_list = [self.get_importance(i) for i in range(len(self.trees))]

    results = [sum(importance * prediction for importance, prediction in zip(importance_list, preds)) / sum(importance_list) for preds in zip(*predictions_list)]
    
    return results
```

 **`predictions_list`**: This contains the predictions made by all trees in the forest for each sample.
- **`importance_list`**: This holds the importance score for each tree, indicating the weight of each tree's predictions.
- **Weighted Average Calculation**:
  - For each sample, compute the weighted average of predictions from all trees.
  - Multiply each prediction by its corresponding tree's importance score.
  - Sum these weighted predictions and divide by the total sum of importance scores to obtain the final result.
- **Final Prediction**:
  - The result is a weighted average of the predictions, where the importance scores determine the contribution of each tree’s prediction to the final outcome.

## Incremental

The IncrementalForests algorithm builds upon WSagging by incorporating an incremental approach. In the first iteration, the dataset `X` and targets `y` are split into `train_0` and `validation_0`. A model is trained on `train_0`, scored on `validation_0`, and both the model and score are saved.

In the next iteration, `validation_0` is further split into `train_1` and `validation_1`. The new training set `train_0` is combined with `train_1`, and a new model is trained on this merged dataset. This model is evaluated on `validation_1`, and the model and score are saved.

This process continues until one of the stopping criteria is met: the validation set becomes too small, the score drops by a predefined margin, the maximum score is achieved, or the number of trees reaches the maximum limit.

During the prediction phase, like in WSagging, predictions are averaged based on validation scores to obtain final predictions. The prediction algorithm is similar to WSagging but uses a different importance formula: `(n - i) * scores[i] ** exponent`, where `n` is the total number of models trained and `scores[i]` is the score of the ith model. This formula gives more weight to models trained earlier in the process.


  