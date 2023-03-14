import numpy as np
import pandas as pd
import psycopg2
import random
import glob
import os
import openpyxl
import re
import streamlit as st
from clean_workouts import clean_workouts
from grab_all_workouts import grab_all_workouts
from intensity_classification import intensity_classification
from from_drive_test import grab_workbook_from_drive

def show_new_workout():
    st.write("Feature coming soon! :)")

def load_workouts(name):
    workbook=grab_workbook_from_drive(name)
    files = grab_all_workouts(workbook)
    files = clean_workouts(files)
    files = intensity_classification(files)

    return files


def show_existing_workout(files, selected_intensity):
    intensities = list(set(df.Intensity for df in files))
    filtered_workouts = [df for df in files if df.Intensity == selected_intensity]
    workout = random.choice(filtered_workouts)
    return workout
        
    

def app():
    st.set_page_config(page_title="Cabral Fitness Exercise Prescriptions")

    # Define the home page
    st.header("Welcome!")

    # Define the "New Workout" button
    new_workout=st.button('New Workout')
    if new_workout:
        show_new_workout()

    # Define the "Existing Workout" button
    existing_workout_button = st.button('Existing Workout')

    if 'new_workout' not in st.session_state:
        st.session_state['new_workout']= False

    if 'existing_workout' not in st.session_state:
        st.session_state['existing_workout']=False

    st.write(st.session_state)

    
    if existing_workout_button or st.session_state.existing_workout:
        st.session_state.existing_workout=True
        with st.form(key='existing_workout_form'):
            name = st.text_input('Enter Your First Name:')
            intensities=['Moderate-Heavy', 'Moderate-Light', 'Light', 'Moderate', 'Heavy']
            intensity = st.selectbox('Select intensity level:', intensities)            
            submit_button = st.form_submit_button(label='Produce Workout')
            if submit_button:
                files=load_workouts(name)
                workout=show_existing_workout(files, intensity)        
                st.text("Don't like the workout? Just hit the button again!")
                st.dataframe(workout)

        


app()






