import pickle

# Load the model
with open('naive_bayes_model.p', 'rb') as model_file:
    model = pickle.load(model_file)


test_data = [
    [22.0, 5.5, 110.0, 0, 0.0, 0.0],  # Example 1: female, normal glucose
    [22.0, 5.5, 210.0, 0, 0.0, 0.0],  # Example 2: female, high glucose
    [22.0, 5.5, 210.0, 1, 1.0, 1.0],  # Example 3: male, high glucose, hypertension, heart disease
]

for instance in test_data:
    prediction = model.predict([instance])[0]
    print(f"Input: {instance} -> Prediction: {prediction}")
