import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title("Titanic Survival Prediction App")

    # User inputs
    pclass = st.selectbox('Passenger Class', [1, 2, 3], index=0)
    sex = st.selectbox('Sex', ['male', 'female'], index=0)
    age = st.slider('Age', 0, 100, 25)
    sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 10, 0)
    parch = st.slider('Number of Parents/Children Aboard', 0, 10, 0)
    fare = st.number_input('Fare', min_value=0.0, value=50.0)
    embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'], index=0)

    # Create a DataFrame from user inputs
    input_features = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })

    # One-hot encoding
    input_features = pd.get_dummies(input_features, columns=['Sex', 'Embarked'], drop_first=True)

    # Load scaler and model
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Ensure input features have all necessary columns
    feature_names = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    for col in feature_names:
        if col not in input_features.columns:
            input_features[col] = 0

    # Ensure the order of columns matches the scaler and model
    input_features = input_features[feature_names]

    # Scale features
    scaled_features = scaler.transform(input_features)

    # Make predictions
    prediction = model.predict(scaled_features)
    prediction_prob = model.predict_proba(scaled_features)

    # Display the results
    st.write(f"Prediction (1: Survived, 0: Did not survive): {int(prediction[0])}")
    st.write(f"Prediction Probability: {prediction_prob[0][int(prediction[0])]}")

if __name__ == "__main__":
    main()


















