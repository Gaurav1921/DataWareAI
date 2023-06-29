import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model

df = None


with st.sidebar:
    st.image("Screenshot (165).png")
    st.title("DataWareAI: Streamlined ML Automation")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an autmated ML pipeline using Streamlit, Pandas Profiling, and PyCaret. Hope you like it!")


if os.path.exists('user_data.csv'):
    df = pd.read_csv('user_data.csv', index_col = None)

if choice == 'Upload':
    st.title("Upload you Data for Modelling")
    file = st.file_uploader("Upload you Dataset here")
    if file is not None:
        df = pd.read_csv(file, index_col = None)   # reading the file
        df.to_csv('user_data.csv', index = None)   # saving file as user_data.csv
        st.dataframe(df)         # to render it on the screen

if choice == 'Profiling':
    st.title('Automated Exploratory Data Analysis')
    if df is not None:
        profile_report = df.profile_report()  # give us the profile report
        st_profile_report(profile_report)      # to render the profile report
    else:
        st.warning("Please upload a dataset first to perform profiling.")


if choice == 'ML':
    st.title('Fitting Machine Learning Models')
    target = st.selectbox("Select your Target", df.columns)
    if st.button('Train the ML model'):

        setup(df, target=target)
        setup_df = pull()
        st.info('This is the ML experminent settings')

        best_model = compare_models()
        compare_df = pull()
        st.info('This is the ML model')

        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')


if choice == 'Download':
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download the File', f, 'trained_model.pkl')
        # (button text, what to downaload, what to name the new file)
