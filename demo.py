import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Data Upload and Preparation
st.title('Best ML Model Finder')
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    features = st.multiselect('Select the features', data.columns)
    target = st.selectbox('Select the target variable', data.columns)
    
    if features and target:
        # Step 2: Model Training
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save the model to a pickle file
        pickle_filename = 'trained_model.pkl'
        with open(pickle_filename, 'wb') as file:
            pickle.dump(model, file)
        
        # Step 3: Provide Download Link
        st.write("Model trained successfully!")
        
        with open(pickle_filename, 'rb') as file:
            st.download_button(
                label="Download trained model",
                data=file,
                file_name=pickle_filename,
                mime="application/octet-stream"
            )

