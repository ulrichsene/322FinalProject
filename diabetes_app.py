import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_model():
    # load the trained naive bayes model from the file
    with open("knn_model.p", "rb") as model_file:
        model = pickle.load(model_file)
    return model

@app.route("/")
def index():
    return "<h1>Welcome to the Diabetes Classifier API</h1>", 200

@app.route("/predict")
def predict():
    age = request.args.get("age")
    a1c_level = request.args.get("a1c_level")
    glucose_level = request.args.get("glucose_level")
    gender = request.args.get("gender")
    hypertension = request.args.get("hypertension")
    heart_disease = request.args.get("heart_disease")

    instance = [[age, a1c_level, glucose_level, gender, hypertension, heart_disease]]

    # load the naive bayes model
    model = load_model()

    # should make the prediction using the naive bayes model's predict method
    pred = model.predict(instance)[0]
    print("pred is: ", pred)

    if pred is not None:
        return jsonify({"prediction (1 = diabetes, 0 = no diabetes)": pred}), 200
    # if something went wrong
    return "Error making a prediction", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5001, debug = False)