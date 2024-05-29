import streamlit as st
import pandas as pd

# Use relative path for deployment
file_path = r"C:\Users\disha\Downloads\Logistic Regression\Logistic Regression\Titanic_train.csv"
predictions_df = pd.read_csv(file_path)

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


