
import numpy as np
from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn import utils

def test_random_forest_classifier_fit():
    # 14 instance interview dataset (test case 1)
    # header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    model = MyRandomForestClassifier()
    model.fit(X_train_interview, y_train_interview)

    # assert that the correct number of trees (M) were selected

    # assert that the bootstrapping was successful (each tree trained on a different sample set)

    # assert that each tree used a different subset of features (F)

def test_random_forest_classifier_predict():
    pass