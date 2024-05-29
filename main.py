import streamlit as st
import pandas as pd
import os

# Use absolute path for debugging
file_path = "C:/Users/disha/Downloads/Logistic Regression/Logistic Regression/Titanic_train.csv"

# Debugging: Print the current working directory and list files
st.write("Current working directory:", os.getcwd())
st.write("Files in the directory:", os.listdir())

# Try to read the CSV file
try:
    predictions_df = pd.read_csv(file_path)
    st.write("File read successfully!")
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

def main():
    # Set the title of the app
    st.title('Titanic Survival Predictor')

    # Display the predictions DataFrame
    st.subheader('Predictions DataFrame')
    st.write(predictions_df)

    # Display basic statistics about the predictions
    st.subheader('Predictions Statistics')
    st.write("Number of predictions:", len(predictions_df))
    st.write("Number of survivors:", predictions_df['Survived'].sum())
    st.write("Number of non-survivors:", len(predictions_df) - predictions_df['Survived'].sum())

if __name__ == '__main__':
    main()





