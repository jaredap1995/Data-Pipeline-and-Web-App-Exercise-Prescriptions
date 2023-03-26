import psycopg2
import random
import streamlit as st
from clean_workouts import clean_workouts
from grab_all_workouts import grab_all_workouts
from intensity_classification import intensity_classification
from grab_workbook_drive import grab_workbook_from_drive
from record_workout import record_workout 
from track_weight_changes import track_weight_changes
import time
import datetime
from track_workouts import track_workouts
import subprocess

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
    
        
        # Start the Cloud SQL proxy when the application starts
        #proxy_process = subprocess.Popen(
        #['./cloud-sql-proxy', '--address', st.secrets.proxy_credentials.address, '--port', st.secrets.proxy_credentials.port, st.secrets.proxy_credentials.name])
    #@st.cache_resource
    #def init_connection():
        #return psycopg2.connect(**st.secrets.psycopg2_credentials)

    conn = psycopg2.connect(**st.secrets.psycopg2_credentials)


    cursor = conn.cursor()
    cursor.execute("SELECT exercise FROM exercises")
    existing_exercises = [row[0] for row in cursor.fetchall()]

    #Trackign session state changes
    #st.write(st.session_state)

    # Define the home page
    st.header("""Welcome! Please Select an Option Below! 
    
         As this is still a work in progress there may be 
        instances when the application is not working properly or doing something weird. 
        Please tell me about it!""")

    # Define the "New Workout" button
    new_workout=st.button('Produce New Workout')
    if 'new_workout' not in st.session_state:
        st.session_state['new_workout']= False
    if new_workout:
        show_new_workout()

    if 'exercise_selection' not in st.session_state:
        st.session_state['exercise_selection']=False

    #Record a Workout Functionality
    record_a_workout = st.button('Record a Workout')
    if 'record_a_workout' not in st.session_state:
        st.session_state['record_a_workout']=False
    if record_a_workout or st.session_state.record_a_workout:
        st.session_state['record_a_workout'] = True
        with st.form(key='record_a_workout_form'):
            name = st.text_input("Name")
            selected_exercises = st.multiselect('Select Exercises (Begin Typing in The Exercise You Performed):', existing_exercises)
            new_exercise = st.text_input("Don't See The exercise you want? Enter your own!")
            submit_new_exercise=st.form_submit_button('Submit New Exercise')
            if submit_new_exercise:
                cursor.execute("INSERT INTO exercises (exercise) VALUES (%s)", (new_exercise,))
                conn.commit()
                st.success(f"Exercise '{new_exercise}' has been added to the database. Please wait")
                time.sleep(1) # wait for 2 seconds before rerunning the script
                st.experimental_rerun()
            submit_button = st.form_submit_button(label='Record Sets, Reps, and Weight')
            if submit_button or st.session_state.exercise_selection:
                st.session_state['exercise_selection']=True
                sets = []
                reps = []
                weight = []
                for exercise in selected_exercises:
                    sets.append(st.text_input(f'{exercise} - Sets: '))
                    reps.append(st.text_input(f'{exercise} - Reps: '))
                    weight.append(st.text_input(f'{exercise} - Weight: '))
                rep_submit_button = st.form_submit_button(label='Record Workout')
                if rep_submit_button:
                    record_workout(conn, name, selected_exercises, reps, sets, weight)
                    # Confirmation message
                    st.success("Data has been submitted.")
                    conn.close()

    #Track Weight Functionality
    track_progress_button = st.button('Track Progress in Exercise')
    if 'track_progress' not in st.session_state:
        st.session_state['track_progress']=False
    if track_progress_button or st.session_state.track_progress:
        st.session_state['track_progress']=True
        with st.form(key='Track_weight_changes'):
            name=st.text_input('Name')
            selected_exercises = st.multiselect('Select Exercises (Begin Typing in The Exercise You Performed):', existing_exercises)
            submit_button=st.form_submit_button(label='Track Progress in Selected Exercises')
            if submit_button:
                track_weight_changes(conn, name, selected_exercises)


    #Track workouts over period of time functionality
    track_workouts_button = st.button('Retrieve a Workout')
    if 'track_workouts' not in st.session_state:
        st.session_state['track_workouts']=False
    if track_workouts_button or st.session_state.track_workouts:
        st.session_state['track_workouts']=True
        with st.form(key='Track_workouts_over_time'):
            name= st.text_input('Enter Your First Name:')
            start_date = st.date_input('Start date', value=datetime.date.today() - datetime.timedelta(days=7))
            end_date = st.date_input('End date', value=datetime.date.today())

            
            if start_date <= end_date:
                st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
            else:
                st.error('Error: date must fall after start date.')
            range_submission=st.form_submit_button('Produce workouts over designated period')
            if range_submission:
                track_workouts(conn, name, start_date, end_date)



    # Produce Archived Spreadsheet Workout Functionality
    existing_workout_button = st.button('Produce Archived Workout')
    if 'existing_workout' not in st.session_state:
        st.session_state['existing_workout']=False
    if existing_workout_button or st.session_state.existing_workout:
        st.session_state['existing_workout']=True
        with st.form(key='archived_workout_form'):
            name = st.text_input('Enter Your First Name:')
            name=name.lower().capitalize()
            intensities=['Moderate-Heavy', 'Moderate-Light', 'Light', 'Moderate', 'Heavy']
            intensity = st.selectbox('Select intensity level:', intensities)            
            submit_button = st.form_submit_button(label='Produce Workout')
            if submit_button:
                files=load_workouts(name)
                workout=show_existing_workout(files, intensity)        
                st.text("Don't like the workout? Just hit the button again!")
                st.dataframe(workout)

        


app()






