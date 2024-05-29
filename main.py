import pandas as pd
import streamlit as st
predictions_df = pd.read_csv(r"C:\Users\disha\Downloads\Logistic Regression\Logistic Regression\Titanic_train.csv")

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