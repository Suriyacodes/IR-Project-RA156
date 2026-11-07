# Technology IR Final Project — Binary Text Classification

## Project title
**Comparative Performance Analysis of Machine Learning Algorithms for Binary Text Classification Using TF-IDF**

## Objective
Build a reproducible binary text-classification experiment using a single labelled text corpus and the same preprocessing/features and held-out test set for four algorithms:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (LinearSVC)
4. Multinomial Naive Bayes

Evaluate accuracy, precision, recall, F1-score, training time and prediction time; compare confusion matrices; run a what-if parameter experiment; and identify the best model from measured results.

## Dataset and two class definitions
The project uses selected categories from the 20 Newsgroups corpus, available through scikit-learn.

- **Technology (positive class / label 1):** `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`.
- **Non-Technology (negative class / label 0):** `rec.sport.baseball`, `rec.sport.hockey`, `sci.space`, `talk.politics.misc`, `misc.forsale`.

These ten source categories are grouped into exactly two target classes. This definition is operational for this assignment: the selected computer-related newsgroups are grouped as Technology, and the selected comparison topics are grouped as Non-Technology.

## Methodology
1. Load the official training and test subsets from the same corpus.
2. Remove headers, footers and quoted text using the dataset loader.
3. Convert the ten original topic labels into two binary labels.
4. Apply TF-IDF with lowercasing, English stop-word removal, unigrams/bigrams and a 30,000-feature cap.
5. Put TF-IDF inside each scikit-learn Pipeline to ensure it is fitted on training text only and avoid test-data leakage.
6. Train and evaluate all four classifiers on the same official test set.
7. Calculate accuracy, positive-class precision, positive-class recall and positive-class F1-score.
8. Save confusion matrices and metric graphs.
9. Run the what-if experiment for Logistic Regression `C` values 0.1, 1.0 and 10.0. Choose the setting using a validation split from training data; the official test set is not used to choose `C`.

## What-if experiment
`what_if_experiment/logistic_regression_C_comparison.csv` compares validation accuracy and F1 for three C values. `C` controls inverse regularization strength: smaller C means stronger regularization; larger C means weaker regularization. The chosen C is then used for the final Logistic Regression test evaluation.

## Run on macOS (VS Code terminal)
From this project folder:

```bash
python3 --version
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If you already created a virtual environment, activate it with `source venv/bin/activate` and continue with installation/run.

## Run on Windows
```bash
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

The first run downloads 20 Newsgroups data and therefore requires internet access.

## Generated files
`results/`:
- `index.html` — detailed results page with charts, measured metrics, and reproduction commands
- `algorithm_comparison.csv`
- `accuracy_comparison.png`
- `precision_comparison.png`
- `recall_comparison.png`
- `f1_score_comparison.png`
- four confusion matrix PNGs

`what_if_experiment/`:
- `logistic_regression_C_comparison.csv`
- `C_parameter_accuracy.png`

## Important interpretation notes
- All four algorithms use the same train/test documents and TF-IDF setup.
- The TF-IDF transformer is trained only on the training portion within each pipeline.
- The what-if parameter is selected using a validation split from training data, not the held-out test set.
- Precision, recall and F1 are reported for the positive class, Technology (label 1). The classification report printed in the terminal also provides per-class and macro/weighted summaries.
- Runtime depends on the computer and software environment.
- Do not invent or copy example metric values. Run the program and report the actual outputs from your machine.
- This is a text classification experiment with an IR-style TF-IDF representation. It is not a full search engine/ranking system.

## Suggested report sections
1. Abstract
2. Introduction
3. Problem statement
4. Objectives
5. Dataset description and class definitions
6. Preprocessing and TF-IDF feature extraction
7. Algorithms: Logistic Regression, KNN, SVM, Naive Bayes
8. Experimental setup and evaluation metrics
9. Results and comparative analysis
10. Confusion matrix analysis
11. What-if experiment and discussion
12. Best model recommendation
13. Conclusion
14. References
