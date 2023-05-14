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
from track_block_progress import show_progress_in_block
from in_progress_functions import test, update_in_progress_workout, check_if_in_progress_exists
from visualization_functions import pull_visuals
from open_AI_new_workout import GPT_Coach
from DecisionTreeRegressor_variable_prescriber import ai_prescription_support
from autoencoder_exercise_selector import exercise_selector

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
    Thank you for visiting my application! 😃
    As this site is constantly evolving I appreciate your patience as I rapidly iterate and 
    attempt to provide the best service possible!

    ### 👈 If you are a client select your name from the drop down menu to the left to begin!

    ### Want to know about the features this application provides?
    - Get a custom exercise program tailored to your health and fitness goals that is updated every 4 weeks as you progress!
    - Track your workouts across your training program and stay on top of your progress with intuitive visualizations to see how far you've come.
    - Expedite Training Prescriptions with a custom built encoder-decoder AI model configured with Attention Mechanism to suggest new exercises based on your previous workouts.
    - Track your Volume-Load across workouts, weeks, and months to monitor injury risk and optimize performance!
    - Get additional supplemenatry workouts anytime, anywhere, at no extra cost. 
    - Any questions? Got some ideas for me? Hate the website? Shoot me an email! jaredaperez1995@gmail.com

   

"""
)
     
   # st.button('Demo')
    return home_text

def name_function():

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

    if "df_value" not in st.session_state:
        st.session_state.df_value = None

    if 'single_exercise_visual' not in st.session_state:
        st.session_state['single_exercise_visual']=False

    if 'all_exercises_visual' not in st.session_state:
        st.session_state['all_exercises_visual']=False


    if 'name' not in st.session_state:
        st.session_state['name'] = name_function()

    name = st.session_state['name']
    st.write(st.session_state['name'])
    if not name:
        home_page()
    else:
        st.session_state['conn'] = psycopg2.connect(**st.secrets.psycopg2_credentials)
        conn = st.session_state['conn']


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


        #Visualize Training Functionality
        col1,col2= st.columns([2,9])
        new_label = '<span style="color:red; font-weight:bold">*New*</span>'
        with col1:
            visualize_training_button = st.button('Visualize Training', help="""Visualize your training over the previous trianing block""")
            if 'visualize_training' not in st.session_state:
                st.session_state['visualize_training']=False
            if visualize_training_button or st.session_state.visualize_training:
                st.session_state['visualize_training']=True
                pull_visuals(conn, name)
        with col2:
            st.markdown(new_label, unsafe_allow_html=True)


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
        col1,col2= st.columns([2,7])
        # Define the HTML code for the label
        new_label = '<span style="color:red; font-weight:bold">*New*</span>'
        with col1:
            new_workout=st.button('Generate New Workout')
            if 'new_workout' not in st.session_state:
                st.session_state['new_workout']= False
            if new_workout:
                GPT_Coach(name)
        with col2:
            # Add the label to the second column
            st.markdown(new_label, unsafe_allow_html=True)

        # Define the "Regressor" button
        beta_label = '<span style="color:orange; font-weight:bold">*In Testing*</span>'
        # col1,col2= st.columns([2,7])
        # with col1:
        st.markdown(beta_label, unsafe_allow_html=True)
        ai_predictions=st.button('Test DTR')
        if 'regressor' not in st.session_state:
            st.session_state['regressor']= False
        if ai_predictions or st.session_state.regressor:
            st.session_state['regressor']=True
            ai_prescription_support(existing_exercises, conn)
    # with col2:
        # Add the label to the second column


        #Autoencoder button
        beta_label_2 = '<span style="color:orange; font-weight:bold">*In Testing*</span>'
        st.markdown(beta_label_2, unsafe_allow_html=True)
        autoencoder_predictions=st.button('Test Autoencoder')
        if 'autoencoder' not in st.session_state:
            st.session_state['autoencoder']= False
        if autoencoder_predictions or st.session_state.autoencoder:
            st.session_state['autoencoder']=True
            exercise_selector(conn)
            
        # conn.close()
# if __name__ == "main":
app()










