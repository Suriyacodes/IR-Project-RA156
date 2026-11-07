from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WHAT_IF = ROOT / "what_if_experiment"
RESULTS.mkdir(exist_ok=True)
WHAT_IF.mkdir(exist_ok=True)

# Binary classes built from the same 20 Newsgroups corpus:
# 1 = technology-related computer groups
# 0 = non-technology comparison groups
TECH_CATEGORIES = [
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
]
NON_TECH_CATEGORIES = [
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.space",
    "talk.politics.misc",
    "misc.forsale",
]
ALL_CATEGORIES = TECH_CATEGORIES + NON_TECH_CATEGORIES
CLASS_NAMES = ["Non-Technology", "Technology"]


def make_pipeline(classifier):
    # Vectorizer is inside the pipeline so it is fitted on training text only.
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])


def load_data():
    print("Downloading/loading 20 Newsgroups data (first run needs internet)...")
    train = fetch_20newsgroups(
        subset="train",
        categories=ALL_CATEGORIES,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes")
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=ALL_CATEGORIES,
        shuffle=True,
        random_state=42,
        remove=("headers", "footers", "quotes")
    )

    def to_binary(targets, target_names):
        # The category names are alphabetical in the dataset's target_names.
        return [
            1 if target_names[int(label)] in TECH_CATEGORIES else 0
            for label in targets
        ]

    y_train = to_binary(train.target, train.target_names)
    y_test = to_binary(test.target, test.target_names)
    return train.data, y_train, test.data, y_test


def metrics_row(name, y_true, predictions, train_seconds, predict_seconds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "algorithm": name,
        "accuracy_percent": round(accuracy_score(y_true, predictions) * 100, 2),
        "precision_percent": round(precision * 100, 2),
        "recall_percent": round(recall * 100, 2),
        "f1_score_percent": round(f1 * 100, 2),
        "training_time_seconds": round(train_seconds, 3),
        "prediction_time_seconds": round(predict_seconds, 3),
    }


def save_confusion_matrix(name, y_true, predictions):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=CLASS_NAMES,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160)
    plt.close(fig)


def save_metric_graphs(results):
    metrics = [
        ("accuracy_percent", "Accuracy (%)", "accuracy_comparison.png"),
        ("precision_percent", "Precision (%)", "precision_comparison.png"),
        ("recall_percent", "Recall (%)", "recall_comparison.png"),
        ("f1_score_percent", "F1-score (%)", "f1_score_comparison.png"),
    ]
    for column, label, filename in metrics:
        ordered = results.sort_values(column)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(ordered["algorithm"], ordered[column])
        ax.set_xlabel(label)
        ax.set_xlim(0, 100)
        ax.set_title(f"Algorithm Comparison — {label}")
        fig.tight_layout()
        fig.savefig(RESULTS / filename, dpi=160)
        plt.close(fig)


def run_what_if(X_train, y_train):
    """
    Parameter what-if study for Logistic Regression.
    Tune C on a validation split from the official training set.
    The official test set is not used to choose C.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )
    rows = []
    for c_value in [0.1, 1.0, 10.0]:
        model = make_pipeline(
            LogisticRegression(C=c_value, max_iter=2000, random_state=42)
        )
        start = time.perf_counter()
        model.fit(X_fit, y_fit)
        train_seconds = time.perf_counter() - start
        predictions = model.predict(X_val)
        rows.append({
            "logistic_regression_C": c_value,
            "validation_accuracy_percent": round(accuracy_score(y_val, predictions) * 100, 2),
            "validation_f1_percent": round(
                precision_recall_fscore_support(
                    y_val, predictions, average="binary", zero_division=0
                )[2] * 100, 2
            ),
            "training_time_seconds": round(train_seconds, 3)
        })

    df = pd.DataFrame(rows).sort_values(
        ["validation_f1_percent", "validation_accuracy_percent"],
        ascending=False
    )
    df.to_csv(WHAT_IF / "logistic_regression_C_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        df["logistic_regression_C"].astype(str),
        df["validation_accuracy_percent"],
        marker="o",
        label="Validation accuracy"
    )
    ax.set_xlabel("Logistic Regression C")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("What-if Experiment: Effect of C")
    ax.legend()
    fig.tight_layout()
    fig.savefig(WHAT_IF / "C_parameter_accuracy.png", dpi=160)
    plt.close(fig)

    best_c = float(df.iloc[0]["logistic_regression_C"])
    print("\nWHAT-IF EXPERIMENT (validation data only)")
    print(df.to_string(index=False))
    print(f"Selected C based on validation F1/accuracy: {best_c}")
    return best_c


def main():
    X_train, y_train, X_test, y_test = load_data()

    print("\nBinary class definitions:")
    print("Class 1 (Technology):", ", ".join(TECH_CATEGORIES))
    print("Class 0 (Non-Technology):", ", ".join(NON_TECH_CATEGORIES))
    print(f"Training documents: {len(X_train)}")
    print(f"Test documents: {len(X_test)}")
    print(f"Training class counts: technology={sum(y_train)}, non-technology={len(y_train)-sum(y_train)}")
    print(f"Test class counts: technology={sum(y_test)}, non-technology={len(y_test)-sum(y_test)}")

    # What-if experiment uses a validation split from training data only.
    best_c = run_what_if(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(C=best_c, max_iter=2000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
        "SVM": LinearSVC(C=1.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    rows = []
    for name, classifier in models.items():
        model = make_pipeline(classifier)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(X_test)
        predict_seconds = time.perf_counter() - start

        row = metrics_row(name, y_test, predictions, train_seconds, predict_seconds)
        rows.append(row)
        save_confusion_matrix(name, y_test, predictions)

        print("\n" + "=" * 62)
        print(name)
        print("=" * 62)
        print(f"Accuracy : {row['accuracy_percent']:.2f}%")
        print(f"Precision: {row['precision_percent']:.2f}%")
        print(f"Recall   : {row['recall_percent']:.2f}%")
        print(f"F1-score : {row['f1_score_percent']:.2f}%")
        print(f"Training time: {train_seconds:.3f} seconds")
        print(classification_report(
            y_test, predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        ))

    results = pd.DataFrame(rows).sort_values(
        ["f1_score_percent", "accuracy_percent"], ascending=False
    ).reset_index(drop=True)
    results.insert(0, "rank_by_f1_score", range(1, len(results) + 1))
    results.to_csv(RESULTS / "algorithm_comparison.csv", index=False)
    save_metric_graphs(results)

    print("\nFINAL RANKING (primary ranking metric: weighted class F1 for positive class)")
    print(results.to_string(index=False))
    print("\nFinished. Check results/ and what_if_experiment/ folders.")
    print("Use only these measured outputs in your report.")


if __name__ == "__main__":
    main()
