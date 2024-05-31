import streamlit as st
import pandas as pd
import pickle

# Load the scaler and feature names
with open('scaler.pkl', 'rb') as f:
    scaler, scaler_feature_names = pickle.load(f)

# Load the trained logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_survival_probability(input_features):
    # Ensure that the input features have the same order of features as during training
    input_features = input_features[scaler_feature_names]

    # Transform the input features using the loaded scaler
    scaled_features = scaler.transform(input_features)

    # Make prediction
    prediction_proba = model.predict_proba(scaled_features)[0][1]
    return prediction_proba

def main():
    st.title("Titanic Survival Predictor")

    # Input features
    st.sidebar.header("Enter Passenger Details")
    pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3], index=2)
    age = st.sidebar.slider("Age", 0, 100, 30)
    sib_sp = st.sidebar.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.sidebar.slider("Number of Parents/Children Aboard", 0, 6, 0)
    fare = st.sidebar.slider("Fare", 0, 600, 30)
    sex_male = st.sidebar.radio("Gender", ["Male", "Female"], index=0)
    embarked = st.sidebar.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"], index=2)

    # Convert categorical variables to numerical
    sex_male = 1 if sex_male == "Male" else 0
    embarked_c = 1 if embarked == "Cherbourg" else 0
    embarked_q = 1 if embarked == "Queenstown" else 0
    embarked_s = 1 if embarked == "Southampton" else 0

    # Predict survival probability
    input_data = {
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sib_sp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_male': [sex_male],
        'Embarked_C': [embarked_c],
        'Embarked_Q': [embarked_q],
        'Embarked_S': [embarked_s]
    }
    input_df = pd.DataFrame(input_data)
    prediction_proba = predict_survival_probability(input_df)

    # Display prediction
    st.subheader("Prediction")
    st.write(f"The predicted survival probability is: {prediction_proba:.2f}")

if __name__ == "__main__":
    main()















