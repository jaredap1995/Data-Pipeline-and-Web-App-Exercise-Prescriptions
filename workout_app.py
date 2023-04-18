import psycopg2
import random
import pandas as pd
import numpy as np
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
from coach_center import coach, create_a_block, deload, increasing_load
from retrieve_prescriptions import retrieve_block
from testing_coach_and_prescriptions import prescribe_block
from track_block_progress import check_if_workout_performed, show_progress_in_block
from update_block import update_workout_in_block
from in_progress_functions import test, update_in_progress_workout, check_if_in_progress_exists

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

def home_page():
    st.write("# Welcome!")
    home_text=st.markdown(
    """
    Thank you for supporting me by using this application! 😃
    As this site is constantly evolving I appreciate your patience as I navigate this adventure and 
    attmept to provide you with the best service possible!

    ### 👈 Select your name from the drop down menu to the left to begin!

    ### Want to know about the features this provides?
    - Get a custom exercise program tailored to your health and fitness goals that is updated every 4 weeks as you progress!
    - Track your workouts across your training program and stay on top of your progress with intuitive visualizations to see how far you've come.
    - Track your Volume-Load across workouts, weeks, and months to monitor injury risk and optimize performance!
    - Get additional supplemenatry workouts anytime, anywhere, at no extra cost. 
    - Any questions? Got some ideas for me? Hate the website? Shoot me an email! jaredaperez1995@gmail.com
"""
)
    return home_text

def name_function():
    #Trackign session state changes
    #st.write(st.session_state)

    conn = psycopg2.connect(**st.secrets.psycopg2_credentials)
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM client;")
    clients = [row[0] for row in cursor.fetchall()]

    if 'new_name' not in st.session_state:
        st.session_state['new_name']=False
    if 'new_submit' not in st.session_state:
        st.session_state['new_submit']=False

    # Set up the sidebar for user input
    with st.sidebar:
        st.success("Please Select your Name from the Dropdown Menu. Or if you are a new client, enter your name!")
        name = st.multiselect('Please select your name', clients, key='client_name')
        if name:
            name=name[0]
        new_client = st.button('New client?')
        if new_client or st.session_state.new_name:
            st.session_state['new_name']=True
            new_name = st.text_input('Please enter your first name:', value='')
            if st.button('Submit') or st.session_state.new_submit:
                st.session_state['new_submit']=True
                if new_name != '':
                    st.session_state['new_name'] = new_name
                    st.experimental_set_query_params(name=new_name)
                    cursor.execute("INSERT INTO client(name) VALUES (%s)", (new_name,))
                    conn.commit()
                    name=new_name
                else: 
                    st.warning('Please Enter Your Name')
        else:
            if st.button('Submit'):
                st.experimental_set_query_params(name=name)
                st.experimental_rerun()

        conn.close()

        return name
        
    

def app():
    st.set_page_config(page_title="Exercise Tracking", layout='wide')

    #Initialize session state
    if 'exercise_selection' not in st.session_state:
            st.session_state['exercise_selection']=False

    if 'actual_workouts_2' not in st.session_state: 
        st.session_state['actual_workouts_2']=False

    if 'update_workout' not in st.session_state:
        st.session_state['update_workout']=False
    
    if 'Show_Block_Progress' not in st.session_state:
        st.session_state['Show_Block_Progress']=False

    if 'next_workout' not in st.session_state:
        st.session_state['next_workout']=False
    if 'whole_block' not in st.session_state:
        st.session_state['whole_block']=False

    if 'continued' not in st.session_state:
        st.session_state['continued'] = False



    name=None
    name=name_function()
    if not name:
        home_page()
    else:
        conn = psycopg2.connect(**st.secrets.psycopg2_credentials)


        cursor = conn.cursor()
        cursor.execute("SELECT exercise FROM exercises")
        existing_exercises = [row[0] for row in cursor.fetchall()]

        # Define the home page
        st.header(f'Hello {name}! Welcome to your Dashboard. Please Select an Option Below')

        if st.session_state.Show_Block_Progress==False:
            continued_workout=check_if_in_progress_exists(conn, name)
            if continued_workout is not None:   
                st.session_state['continued'] = True
                st.stop()
            
        show_progress_in_a_block = st.button('Track and Record Training Across Current Block')
        if show_progress_in_a_block or st.session_state.Show_Block_Progress:
            st.session_state['Show_Block_Progress']=True
            show_progress_in_block(conn, name)
            st.stop()


        #Record a Workout Functionality
        record_a_workout = st.button('Record a Workout Outside of Your Current Block')
        if 'record_a_workout' not in st.session_state:
            st.session_state['record_a_workout']=False
        if record_a_workout or st.session_state.record_a_workout:
            st.session_state['record_a_workout'] = True
            with st.form(key='record_a_workout_form'):
                name = name
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
                    df=pd.DataFrame({'Exercise': selected_exercises,
                                    'Sets': np.zeros(len(selected_exercises)),
                                    'Reps': np.zeros(len(selected_exercises)),
                                    'Weight': np.zeros(len(selected_exercises))})
                    edited_df = st.experimental_data_editor(df)
                rep_submit_button = st.form_submit_button(label='Record Workout')
                if rep_submit_button:
                    record_workout(conn, name, selected_exercises, edited_df['Reps'], edited_df['Sets'], edited_df['Weight'])
                    # Confirmation message
                    st.success("Data has been submitted.")
                    time.sleep(1)
                    st.experimental_rerun()

        #Track Weight Functionality
        track_progress_button = st.button("View Weight Progress", help="""Track your weightlifting progress by selecting an exercise 
                                                            and seeing the weights you lifted over time.
                                                            """)
        if 'track_progress' not in st.session_state:
            st.session_state['track_progress']=False
        if track_progress_button or st.session_state.track_progress:
            st.session_state['track_progress']=True
            with st.form(key='Track_weight_changes'):
                name=name
                selected_exercises = st.multiselect('Select Exercises (Begin Typing in The Exercise You Performed):', existing_exercises)
                submit_button=st.form_submit_button(label='Track Progress in Selected Exercises')
                if submit_button:
                    track_weight_changes(conn, name, selected_exercises)


        #Track workouts over period of time functionality
        track_workouts_button = st.button('See all Workouts', help="See all the workouts you have performed over a period of time")
        if 'track_workouts' not in st.session_state:
            st.session_state['track_workouts']=False
        if track_workouts_button or st.session_state.track_workouts:
            st.session_state['track_workouts']=True
            with st.form(key='Track_workouts_over_time'):
                name= name
                start_date = st.date_input('Start date', value=datetime.date.today() - datetime.timedelta(days=7))
                end_date = st.date_input('End date', value=datetime.date.today())

                
                if start_date <= end_date:
                    st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
                else:
                    st.error('Error: date must fall after start date.')
                range_submission=st.form_submit_button('Produce workouts over designated period')
                if range_submission:
                    track_workouts(conn, name, start_date, end_date)

        #Coach Center Functionality
        with st.sidebar:
            prescribe_block_button = st.button('Coach Center')
            if 'coach_center' not in st.session_state:
                st.session_state['coach_center']=False
            if prescribe_block_button or st.session_state.coach_center:
                name=name
                st.session_state['coach_center']=True
                prescribe_block(conn, name)


        # Produce Archived Spreadsheet Workout Functionality
        existing_workout_button = st.button('Produce Archived Workout', help="Get a random workout from legacy system")
        if 'existing_workout' not in st.session_state:
            st.session_state['existing_workout']=False
        if existing_workout_button or st.session_state.existing_workout:
            st.session_state['existing_workout']=True
            with st.form(key='archived_workout_form'):
                name = name
                name=name.lower().capitalize()
                intensities=['Moderate-Heavy', 'Moderate-Light', 'Light', 'Moderate', 'Heavy']
                intensity = st.selectbox('Select intensity level:', intensities)            
                submit_button = st.form_submit_button(label='Produce Workout')
                if submit_button:
                    files=load_workouts(name)
                    workout=show_existing_workout(files, intensity)        
                    st.text("Don't like the workout? Just hit the button again!")
                    st.dataframe(workout)


        # Define the "New Workout" button
        new_workout=st.button('Produce New Workout')
        if 'new_workout' not in st.session_state:
            st.session_state['new_workout']= False
        if new_workout:
            show_new_workout()

            
        conn.close()
# if __name__ == "main":
app()











