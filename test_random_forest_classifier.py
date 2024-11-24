
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

    model = MyRandomForestClassifier(N=5, M=3, F=2)
    model.fit(X_train_interview, y_train_interview)

    # assert that the correct number of trees (M) were selected

    # assert that the bootstrapping was successful (each tree trained on a different sample set)

    # assert that each tree used a different subset of features (F)

def test_random_forest_classifier_predict():
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

    model = MyRandomForestClassifier(N =5, M =3, F = 2)
    model.fit(X_train_interview, y_train_interview)

    # test cases:
    test_case1 = [["Junior", "Java", "yes", "no"]]
    test_case2 = [["Junior", "Java", "yes", "yes"]]

    tree_predictions1 = []
    tree_predictions2 = []

    for tree in model.trees:
        prediction1 = tree.predict(test_case1)
        tree_predictions1.append(prediction1[0])

        prediction2 = tree.predict(test_case2)
        tree_predictions2.append(prediction2[0])
    
    # need majority voting logic for first test case
    vote_count1 = {"True": 0, "False": 0}
    for prediction in tree_predictions1:
        vote_count1[prediction] +=1
    
    majority_vote1 = max(vote_count1, key = vote_count1.get)

    # same thing here for test case 2

    # determine expected predictions based on majority voting
    expected_prediction1 = majority_vote1

    # call predict and assert that it equals the majority voting results
    assert model.predict(test_case1) == expected_prediction1