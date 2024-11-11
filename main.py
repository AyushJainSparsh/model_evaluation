import streamlit as st 
from streamlit_option_menu import option_menu
import models

class main :
    def __init__(self):
        self.apps = []
    def add_apps(self, title ,function):
        self.apps.append({
            'title' : title,
            'function' : function
        })
    def run():
        with st.sidebar :
            app = option_menu(
                menu_title='ML MODEL SELECTION',
                options=['ML MODELS'] , 
                default_index=0 ,
                menu_icon= 'robot' , 
                icons = ['motherboard'])
            
        if app == 'ML MODEL':
            models.app()
    run()