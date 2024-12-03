
import numpy as np
import random
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
    # N = number of trees, M = features, F = features per split
    # seed random generator here:
    random.seed(0)
    model = MyRandomForestClassifier(N=5, M=3, F=2)
    model.fit(X_train_interview, y_train_interview)

    for idx, dt in enumerate(model.trees, start=1):
        print(f"Tree {idx}:")
        print(dt.tree)
        print("-" * 40)
    
    # check the correct number of trees (N) were created in the ensemble
    assert len(model.trees) == model.M
    
    # check that M and F are set correctly
    assert model.M == 3
    assert model.F == 2
    
    # # assert that each tree is trained on a different bootstrapped dataset
    # bootstrapped_samples = [tree for tree in model.trees]
    # assert all(len(set(bootstrapped_samples[i]) & set(bootstrapped_samples[j])) < len(bootstrapped_samples[i])
    #            for i in range(len(bootstrapped_samples)) for j in range(i + 1, len(bootstrapped_samples)))
    
    # verify that each tree in the ensemble is using a different subset of F
    # feature_subsets = [tree[1] for tree in model.trees]

    # check if all feature subsets are unique
    # assert all(feature_subsets[i] != feature_subsets[j]
    #            for i in range(len(feature_subsets)) for j in range(i + 1, len(feature_subsets)))

    # check all trees in the ensemble are unique
    tree_signatures = [str(tree) for tree in model.trees]
    assert len(tree_signatures) == len(set(tree_signatures))

def test_random_forest_classifier_predict():
    model = MyRandomForestClassifier(N =5, M =3, F = 2)
    model.fit(X_train_interview, y_train_interview)

    # test cases:
    test_case1 = ["Junior", "Java", "yes", "no"] # true
    test_case2 = ["Junior", "Java", "yes", "yes"] # false

    prediction1 = model.predict([test_case1])
    print(prediction1)
    prediction2 = model.predict([test_case2])
    print(prediction2)

    # tree 1 from fit:
    ['Attribute', 1, ['Value', 'no', ['Leaf', 'False', 5, 14]], ['Value', 'yes', ['Leaf', 'True', 9, 14]]]

    # tree 2 from fit:
    ['Attribute', 0, ['Value', 'Junior', ['Leaf', 'True', 5, 14]], ['Value', 'Mid', ['Leaf', 'True', 7, 14]], ['Value', 'Senior', ['Attribute', 1, ['Value', 'no', ['Leaf', 'False', 1, 2]], ['Value', 'yes', ['Leaf', 'True', 1, 2]]]]]

    # tree 3 from fit:
    ['Attribute', 1, ['Value', 'Java', ['Leaf', 'False', 2, 14]], ['Value', 'Python', ['Attribute', 0, ['Value', 'no', ['Leaf', 'False', 14, 3]], ['Value', 'yes', ['Leaf', 'False', 14, 3]]]], ['Value', 'R', ['Attribute', 0, ['Value', 'no', ['Leaf', 'True', 2, 6]], ['Value', 'yes', ['Leaf', 'False', 14, 4]]]]]

    assert prediction1 == ["True"]
    assert prediction2 == ["False"]
