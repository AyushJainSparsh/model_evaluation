import streamlit as st
import pandas as pd

def app():
    st.title(":red[MODELS]")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write(data.head())
            features = st.multiselect('Select Independent Features',data.columns)
            target = st.selectbox('Select the Dependent Feature',data.columns)

            if st.button('Show EDA'):
                st.write(data.describe())
        except Exception as e:
            st.error(f"Error {e}")

app()