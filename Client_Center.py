import psycopg2
import random
import pandas as pd
import numpy as np
import streamlit as st
from excel_file_parsing_modules.clean_workouts import clean_workouts
from excel_file_parsing_modules.grab_all_workouts import grab_all_workouts
from excel_file_parsing_modules.intensity_classification import intensity_classification
from excel_file_parsing_modules.grab_workbook_drive import grab_workbook_from_drive
from psycopg2_database_files.record_workout import record_workout 
from miscellaneous.track_weight_changes import track_weight_changes
import time
import datetime
from miscellaneous.track_workouts import track_workouts
from miscellaneous.track_block_progress import show_progress_in_block
from psycopg2_database_files.in_progress_functions import test, update_in_progress_workout, check_if_in_progress_exists
from miscellaneous.visualization_functions import pull_visuals
from name_selector import name_function
from miscellaneous.demo_video import demo_video_function

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
    home_text_1=st.markdown(
    """
    Thank you for visiting my application! 😃
    As this site is constantly evolving I appreciate your patience as I rapidly iterate and 
    attempt to provide the best service possible!

    ### 👈 If you are a client select your name from the drop down menu to the left to begin!

    ### For a demo of the application, click the button below! 👇""")

    demo = demo_video_function()

    home_text_2=st.markdown("""### Want to know about the features this application provides?

    Get a custom exercise program tailored to your health and fitness goals that is updated every 4 weeks as you progress!
    Track your workouts across your training program and stay on top of your progress with intuitive visualizations to see how far you've come.
    Expedite Training Prescriptions with a custom built encoder-decoder AI model configured with Attention Mechanism to suggest new exercises based on your previous workouts.
    Track your Volume-Load across workouts, weeks, and months to monitor injury risk and optimize performance!
    Get additional supplemenatry workouts anytime, anywhere, at no extra cost. 
    Any questions? Got some ideas for me? Hate the website? Shoot me an email! jaredaperez1995@gmail.com
""")

     

    

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

    if "df_value" not in st.session_state:
        st.session_state.df_value = None

    if 'single_exercise_visual' not in st.session_state:
        st.session_state['single_exercise_visual']=False

    if 'all_exercises_visual' not in st.session_state:
        st.session_state['all_exercises_visual']=False

    if 'change_name' not in st.session_state:
        st.session_state['change_name']=False

    if 'name' not in st.session_state:
        st.session_state['name']=None
    
    st.session_state['name']=name_function()
    name = st.session_state['name']
    if not name:
        home_page()
    else:
        with psycopg2.connect(**st.secrets.psycopg2_credentials) as conn:
            st.session_state['conn']=conn

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

            # Define the home page
            st.header(f'Hello {name}! Welcome to your Dashboard. Please Select an Option Below')


            
            if st.session_state.Show_Block_Progress==False:
                continued_workout=check_if_in_progress_exists(conn, name)
                # st.write(continued_workout)
                if continued_workout is not None:   
                    #st.session_state['continued'] = True
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
                st.stop()


            #Visualize Training Functionality
            visualize_training_button = st.button('Visualize Training', help="""Visualize your training over the previous trianing block""")
            if 'visualize_training' not in st.session_state:
                st.session_state['visualize_training']=False
            if visualize_training_button or st.session_state.visualize_training:
                st.session_state['visualize_training']=True
                pull_visuals(conn, name)
                st.stop()


            #Track Weight Functionality
            track_progress_button = st.button("View Weight Progress", 
                help="""Track your weightlifting progress by selecting an exercise 
                and seeing the weights you lifted over time.""")
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
                st.stop()


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
                st.stop()

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
                st.stop()

                
            # conn.close()
    # if __name__ == "main":
app()











