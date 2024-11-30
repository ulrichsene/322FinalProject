
import numpy as np
from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn import utils

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

def test_random_forest_classifier_fit():
    model = MyRandomForestClassifier(N=5, M=3, F=2)
    model.fit(X_train_interview, y_train_interview)
    desk_m = 3

    assert model.M == desk_m
    assert model.F != model.F
    assert model.F != model.F
    assert model.F != model.F
    # check that the ensemble is correct m = m, f = f, check that each pair of the two trees in the ensemble are different
# hard code m = 3
    # assert that the correct number of trees (M) were selected

    # assert that the bootstrapping was successful (each tree trained on a different sample set)

    # assert that each tree used a different subset of features (F)

def test_random_forest_classifier_predict():
    model = MyRandomForestClassifier(N =5, M =3, F = 2)
    model.fit(X_train_interview, y_train_interview)

    # test cases:
    test_case1 = ["Junior", "Java", "yes", "no"] # true
    test_case2 = ["Junior", "Java", "yes", "yes"] # false

    desk_prediction= "True"

    predictions = model.predict(test_case1)

    assert predictions == desk_prediction

    # n = 10, compute bagging samples, produces 10 lists, each list has [bootstrap sample (trees training set), out of bag sample (used for testing) ]