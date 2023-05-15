import streamlit as st
import psycopg2
import time
import datetime
import pandas as pd
import numpy as np
from name_selector import name_function
from open_AI_new_workout import GPT_Coach
from DecisionTreeRegressor_variable_prescriber import ai_prescription_support
from autoencoder_exercise_selector import exercise_selector
from coach_functions import show


def coach_page():
    home_text=st.markdown(
    """
    # Welcome to the Coaching Webpage!
    As a coach, we understand your need for a platform that facilitates seamless and efficient management of your clients' training data. 
    
    ### 👈 Select your client's name from the drop down menu to the left to begin!

    Our webpage is designed to assist coaches in various aspects of their job, providing you with advanced tools and features. Here's what you can do:

    ## 1. Upload Your Clients' Training Data
    Say goodbye to misplaced files and disorganized data. Our platform allows you to upload your clients' training data, making it easily accessible at any time. This ensures that important information is always at your fingertips, ready to be utilized when needed.

    ## 2. Leverage OpenAI API for Generated Workouts
    This platform is integrated with OpenAI API, enabling you to make API calls to generate workouts based on your clients' data. This feature allows you to provide personalized workout routines, tailored to meet each client's unique needs and fitness levels.

    ## 3. Utilize a dedicated Transformer AI Model for Exercise Suggestions
    Designing exercise programs can be cumbersome and repetitive. Our platform is equipped with a Transformer AI model, trained on thousands of exercises. This cutting-edge AI suggests similar exercises and programs, making exercise prescription easier and clients happier.

    ## 4. Get Personalized Exercise Variable Recommendations
    This platform provides personalized exercise variable recommendations, developed from your clients' training data. This means each workout, each exercise, and each set is optimized to your client's specific requirements, enhancing workout efficacy and coaching success.

    - Any questions? Got some ideas for me? Hate the website? Shoot me an email! jaredaperez1995@gmail.com

   

"""
)



def app_2():
    st.session_state['conn'] = psycopg2.connect(**st.secrets.psycopg2_credentials)
    conn = st.session_state['conn']

    if 'name' not in st.session_state:
        st.session_state['name'] = None

    st.session_state['name']=name_function()
    name=st.session_state['name']
    # st.write(st.session_state['name'])

    if not name:
        coach_page()
    else:
        st.header(f'You are in Coach Mode for {name}. Please Select an Option Below')
        show(st.session_state['name'], st.session_state['conn'])
        with st.sidebar:
            if st.button('Change name'):
                st.session_state['change_name'] = True
                st.experimental_rerun()

        with st.sidebar:
            if st.button('Home'):
                for key in list(st.session_state.keys()):
                    if key != 'name':
                        del st.session_state[key]
                st.experimental_rerun()

        cursor = conn.cursor()
        cursor.execute("SELECT exercise FROM exercises")
        existing_exercises = [row[0] for row in cursor.fetchall()]


     # Define the "New Workout" button
        col1,col2= st.columns([1,12])
        # Define the HTML code for the label
        new_label = '<span style="color:red; font-weight:bold">*New*</span>'
        with col2:
            new_workout=st.button('Generate New Workout')
            if 'new_workout' not in st.session_state:
                st.session_state['new_workout']= False
            if new_workout:
                GPT_Coach(name)
        with col1:
            # Add the label to the second column
            st.markdown(new_label, unsafe_allow_html=True)

        #Autoencoder button
        beta_label_2 = '<span style="color:orange; font-weight:bold">*In Testing*</span>'
        col1,col2= st.columns([1,10])
        with col1:
            st.markdown(beta_label_2, unsafe_allow_html=True)
        with col2:    
            autoencoder_predictions=st.button('AI Prescriber')
            if 'autoencoder' not in st.session_state:
                st.session_state['autoencoder']= False
            if autoencoder_predictions or st.session_state.autoencoder:
                st.session_state['autoencoder']=True
                exercise_selector(conn, name)



app_2()