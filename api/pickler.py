import pickle
from mysklearn.myclassifiers import MyNaiveBayesClassifier
from mysklearn import utils

# load and prepare the data
X, y = utils.prepare_mixed_data()

# create and train the naive bayes classifier
model = MyNaiveBayesClassifier()
model.fit(X, y)

# save the trained model using pickle
with open("naive_bayes_model.p", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model saved successfully.")
