import streamlit as st
import pandas as pd

import univariate.regression as reg
import univariate.classification as classifier

import multivariate.regression as multi_Reg
import multivariate.classification as multi_Classifier
import pickle
import model

from sklearn.model_selection import train_test_split

def app():
    st.title(":red[MODELS]")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        category = st.selectbox('Select Problem Category',['Regression' , 'Classification'])

        features = st.multiselect('Select Independent Features',data.columns)
        target = st.multiselect('Select Dependent Feature',data.columns)

        if st.button('Show EDA'):
            st.write(data.describe())

        if features is not None and target is not None :
            if st.button('Train'):
                X_train, X_test, y_train, y_test = train_test_split(data[features] , data[target] , test_size= 0.2 , random_state= 50 , train_size=0.8)

                pickle_filename = 'trained_model.pkl'
                
                if len(target) == 1:

                    if category == 'Regression':
                        result = reg.Regression(X_train, X_test, y_train, y_test)
                        st.write(result)
                        with open(pickle_filename, 'wb') as file:
                            pickle.dump(model.Regression_best_model(result, X_train , y_train), file)
                    
                    else :
                        result = classifier.Classification(X_train, X_test, y_train, y_test)
                        with open(pickle_filename, 'wb') as file:
                            pickle.dump(model.Classification_best_model(result, X_train , y_train), file)

                if len(target) > 1 :

                    if category == 'Regression':
                        result = multi_Reg.Regression(X_train, X_test, y_train, y_test)
                        st.write(result)
                        with open(pickle_filename, 'wb') as file:
                            pickle.dump(model.Multi_Regression_best_model(result, X_train , y_train), file)
                    
                    else :
                        result = multi_Classifier.Classification(X_train, X_test, y_train, y_test)
                        with open(pickle_filename, 'wb') as file:
                            pickle.dump(model.Multi_Classification_best_model(result, X_train , y_train), file)

                # Step 3: Provide Download Link
                st.write("Model trained successfully!")
                
                with open(pickle_filename, 'rb') as file:
                    st.download_button(
                        label="Download trained model",
                        data=file,
                        file_name=pickle_filename,
                        mime="application/octet-stream"
                    )
