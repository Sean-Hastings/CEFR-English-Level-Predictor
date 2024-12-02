from __future__ import annotations
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

module_path = os.path.abspath(os.path.join("."))
if module_path not in sys.path:
    sys.path.append(module_path)


import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from cefr_predictor.preprocessing import generate_features

RANDOM_SEED = 1

label_encoder = None


def train(model):
    global X_train, X_test, y_train, y_test
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    print(f"Training {model['name']}.")

    for i, (i_train, i_test) in enumerate(skf.split(X, Y)):
        X_train, X_test = X[i_train], X[i_test]
        y_train, y_test = Y[i_train], Y[i_test]
        pipeline = build_pipeline(model["model"])
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        scores.append(score)
        print(f"Fold {i}: {score}")
    
    print(f"Average score: {sum(scores) / len(scores)}")

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y
    # )

    # print(f"Training {model['name']}.")
    # pipeline = build_pipeline(model["model"])
    # pipeline.fit(X_train, y_train)
    # print(pipeline.score(X_test, y_test)) # TODO: use this to calc accuracy in dist_compare. It will have to be a new entry in the data_dict within the helper function that is then accessed from outside like the others
    # save_model(pipeline, model["name"])


def build_pipeline(model):
    """Creates a pipeline with feature extraction, feature scaling, and a predictor."""
    return Pipeline(
        steps=[
            ("generate features", FunctionTransformer(generate_features)),
            ("scale features", StandardScaler()),
            ("model", model),
        ],
        verbose=True,
    )


def load_data(path_to_data):
    data = pd.read_csv(path_to_data)
    X = data.text.tolist()
    y = encode_labels(data.label)
    return X, y


def encode_labels(labels):
    global label_encoder
    if not label_encoder:
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
    return label_encoder.transform(labels)


def save_model(model, name):
    name = name.lower().replace(" ", "_")
    file_name = f"cefr_predictor/models/{name}.joblib"
    print(f"Saving {file_name}")
    dump(model, file_name)


models = [
    {
        "name": "XGBoost",
        "model": XGBClassifier(
            objective="multi:softprob",
            random_state=RANDOM_SEED,
            use_label_encoder=False,
        ),
    },
    # {
    #     "name": "Logistic Regression",
    #     "model": LogisticRegression(random_state=RANDOM_SEED),
    # },
    # {
    #     "name": "Random Forest",
    #     "model": RandomForestClassifier(random_state=RANDOM_SEED),
    # },
    # {"name": "SVC", "model": SVC(random_state=RANDOM_SEED, probability=True)},
]

X, Y = load_data("data/cefr_leveled_texts.csv")
X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_SEED, stratify=Y
)

# X_train, y_train = load_data("data/train.csv")
# X_test, y_test = load_data("data/test.csv")


if __name__ == "__main__":
    for model in models:
        for RANDOM_SEED in range(10):
            print(f"RANDOM SEED: {RANDOM_SEED}")
            train(model)
