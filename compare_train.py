# compare_train.py is for helping with select a setting. 
# Is it the C in this case
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


dataset = load_iris()

# Reproduce the original split
# Keep the test data out of this comparison
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data,
    dataset.target,
    test_size=0.2,
    random_state=42,
    stratify=dataset.target,
)

folds = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for c in [1.0, 0.1]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=1000, 
            random_state=42,
        ),
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=folds,
        scoring="accuracy",
    )

    print(f"\nC: {c}")
    print("Fold accuracies:", [round(float(s), 3) for s in scores])
    print(f"Mean accuracy: {scores.mean():.3f}")
    print(f"Standard deviation: {scores.std():.3f}")