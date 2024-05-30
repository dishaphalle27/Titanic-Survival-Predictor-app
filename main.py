import streamlit as st
import pandas as pd
import os

def main():
    # Path to the uploaded file
    file_path = "Titanic_train.csv"

    # Try to read the CSV file
    try:
        predictions_df = pd.read_csv(file_path)
        st.write("File read successfully!")
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except pd.errors.EmptyDataError as e:
        st.error(f"File is empty: {e}")
        st.stop()
    except pd.errors.ParserError as e:
        st.error(f"Error parsing file: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

    # Ensure that predictions_df was read successfully
    if predictions_df is not None:
        # Set the title of the app
        st.title('Titanic Survival Predictor')

        # Display the predictions DataFrame
        st.subheader('Predictions DataFrame')
        st.write(predictions_df)

        # Display basic statistics about the predictions
        st.subheader('Predictions Statistics')
        st.write("Number of predictions:", len(predictions_df))
        if 'Survived' in predictions_df.columns:
            st.write("Number of survivors:", predictions_df['Survived'].sum())
            st.write("Number of non-survivors:", len(predictions_df) - predictions_df['Survived'].sum())
        else:
            st.write("The 'Survived' column is not in the DataFrame.")

if __name__ == '__main__':
    main()









