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
