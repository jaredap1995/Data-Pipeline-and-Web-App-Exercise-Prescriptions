import streamlit as st
import psycopg2
import time
import datetime
import pandas as pd
import numpy as np
from record_prescriptions import record_block
from name_selector import name_function
from open_AI_new_workout import GPT_Coach
from DecisionTreeRegressor_variable_prescriber import ai_prescription_support
from autoencoder_exercise_selector import exercise_selector


def increasing_load(workouts, operation, weeks):
    copies = []
    if len(workouts)!=1:
        for workout in workouts:
            for i in range(1, weeks+1):
                df = workout.copy()
                if operation == 'Intensity':
                    df['Weight'] += i * 5
                   # df['Relative Load']+= i*5
                elif operation == 'Volume':
                    df['Sets'] += i
                   # df['Relative Load']+= i*5
                elif operation == 'Endurance':
                    df['Reps'] += i * 4
                   # df['Relative Load']+= i*5
                else:
                    pass
                copies.append(df)
        reordered_list = []
        for i in range(weeks):
            correct_order = copies[i::weeks]  
            reordered_list.extend(correct_order)
        copies=reordered_list
    else:
        for i in range(1, weeks+1):
            df = workouts[0].copy()
            if operation == 'Intensity':
                df['Weight'] += i * 5
               # df['Relative Load']+= i*5
            elif operation == 'Volume':
                 df['Sets'] += i
                # df['Relative Load']+= i*5
            elif operation == 'Endurance':
                df['Reps'] += i * 4
               # df['Relative Load']+= i*5
            else:
                pass
            copies.append(df)
    return copies


def deload(workouts, operation, weeks):
    copies = []
    if len(workouts)!=1:
        for workout in workouts:
            for i in range(1, weeks+1):
                df = workout.copy()
                if operation == 'Intensity':
                    df['Weight'] = (df['Weight'] - (i*5)).apply(lambda x: max(1, x))
                   # df['Relative Load'] = (df['Relative Load'] -5).apply(lambda x: max(1, x))
                elif operation == 'Volume':
                    df['Sets'] = (df['Sets']).apply(lambda x: max(1, x))
                  #  df['Relative Load'] = (df['Relative Load'] -5).apply(lambda x: max(1, x))
                elif operation == 'Endurance':
                    df['Reps'] = (df['Reps'] -2).apply(lambda x: max(1, x))
                  #  df['Relative Load'] = (df['Relative Load'] -5).apply(lambda x: max(1, x))
                # else:
                #     pass
                copies.append(df)
    else:
        for i in range(1, weeks+1):
            df = workouts[0].copy()
            if operation == 'Intensity':
                df['Weight'] = (df['Weight'] - (i*5)).apply(lambda x: max(1, x))
               # df['Relative Load'] = (df['Relative Load'] -5).apply(lambda x: max(1, x))
            elif operation == 'Volume':
                df['Sets'] = (df['Sets']).apply(lambda x: max(1, x))
                #df['Relative Load'] = (df['Relative Load'] -5).apply(lambda x: max(1, x))
            elif operation == 'Endurance':
                df['Reps'] = (df['Reps'] -2).apply(lambda x: max(1, x))
               # df['Relative Load'] = (df['Relative Load'] -5).apply(lambda x: max(1, x))
            # else:
            #     pass
            copies.append(df)
    return copies




def create_a_block(workouts, operation, total_weeks):
    heavier_weeks=increasing_load(workouts, operation=operation, weeks=total_weeks-2)
    deload_week=deload(workouts, operation=operation, weeks=total_weeks-(total_weeks-1))
    workouts.extend(heavier_weeks)
    workouts.extend(deload_week)
    return workouts


def coach(conn, name):
    
    cursor=conn.cursor()

    cursor.execute("SELECT exercise FROM exercises")
    existing_exercises = [row[0] for row in cursor.fetchall()]

    if 'exercise_selection' not in st.session_state:
        st.session_state['exercise_selection']=False

    if 'workouts' not in st.session_state:
        st.session_state['workouts']=False

    
    block=None
    name=name
    length_of_block=None                   

                
    if 'prescriptions' not in st.session_state:
        st.session_state['prescriptions']=[]     

    if 'prescription_recording' not in st.session_state:
        st.session_state['prescription_recording']=False


    record_a_workout = st.button('Prescribe Block')
    if 'record_a_workout_2' not in st.session_state:
        st.session_state['record_a_workout_2']=False
    if record_a_workout or st.session_state.record_a_workout_2:
        st.session_state['record_a_workout_2'] = True
        with st.form(key='record_a_workout_form 2'):
            name = name
            selected_exercises = st.multiselect('Select Exercises (Begin Typing in The Exercise You Performed):', existing_exercises)
            new_exercise = st.text_input("Don't See The exercise you want? Enter your own!")
            submit_new_exercise=st.form_submit_button('Submit New Exercise 2')
            if submit_new_exercise:
                cursor.execute("INSERT INTO exercises (exercise) VALUES (%s)", (new_exercise,))
                conn.commit()
                st.success(f"Exercise '{new_exercise}' has been added to the database. Please wait")
                # time.sleep(1) # wait for 2 seconds before rerunning the script
                # st.experimental_rerun()
            submit_button = st.form_submit_button(label='Record Sets, Reps, and Weight 2')
            if submit_button or st.session_state.exercise_selection:
                st.session_state['exercise_selection']=True
                df=pd.DataFrame({'Exercise': selected_exercises,
                                'Sets': np.zeros(len(selected_exercises)),
                                'Reps': np.zeros(len(selected_exercises)),
                                'Weight': np.zeros(len(selected_exercises))})
                edited_df = st.experimental_data_editor(df, num_rows='dynamic')
                rep_submit_button_2 = st.form_submit_button(label='Record Workout')
                if rep_submit_button_2 or st.session_state.prescription_recording:
                    st.session_state['prescription_recording']=True
                    st.success('Prescription submitted. Adding to list')
                    prescription=edited_df
                    st.session_state['prescriptions'].append(prescription)
                    st.session_state.exercise_selection = False
                    st.session_state.prescription_recording = False
                    time.sleep(1)
                    st.experimental_rerun()
        for df in st.session_state['prescriptions']:
            st.dataframe(df)
        with st.form('prescribe_length_and_goal'):
            goals=['Intensity', 'Volume', 'Endurance']
            goal=st.selectbox('Select goal', goals)
            length_of_block=st.slider("Select number of weeks:", 2, 10)
            block_button=st.form_submit_button(label='Program Block')
            if block_button:
                workout=st.session_state['prescriptions']
                if len(workout) ==1:
                    my_list=st.session_state['prescriptions'][0]
                    my_list=[my_list]
                else:
                    my_list=st.session_state['prescriptions']
                workouts_per_week = len(my_list)
                block=create_a_block(my_list, operation=goal, total_weeks=length_of_block)
                week_num = 1
                if len(block) > 0:
                    st.write(f"Week {week_num}")
                    st.dataframe(block[0])
                    for i in range(1, len(block)):
                        if i % workouts_per_week == 0:
                            week_num += 1
                            st.write(f"Week {week_num}")
                        st.dataframe(block[i])
                st.session_state['prescriptions']=[]

    
    return block, name, length_of_block


def show(name, conn):

    name=name
    st.session_state['coach_center']=True
    prescribe_block(conn, name)
    

def prescribe_block(conn, name):

    if 'block' not in st.session_state:
        st.session_state['block']=False

    #st.write('Testing Coach Center and Record Prescriptions Together')
    st.session_state['block'], name, length_of_block=coach(conn, name)
    # submit_button=st.button('Record Block in Database')
    if st.session_state.block:
        record_block(conn, name, st.session_state['block'], length_of_block)
        st.success('Block Uploaded to Database Successfully. Rerunning to update state.')
        time.sleep(4)
        st.session_state['block']=[]
        st.experimental_rerun()

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

        # Define the "Regressor" button
        # beta_label = '<span style="color:orange; font-weight:bold">*In Testing*</span>'
        # col1,col2= st.columns([1,10])
        # with col1:
        #     st.markdown(beta_label, unsafe_allow_html=True)
        # with col2:
        #     ai_predictions=st.button('Test DTR')
        #     if 'regressor' not in st.session_state:
        #         st.session_state['regressor']= False
        #     if ai_predictions or st.session_state.regressor:
        #         st.session_state['regressor']=True
        #         ai_prescription_support(existing_exercises, conn)

        #Autoencoder button
        beta_label_2 = '<span style="color:orange; font-weight:bold">*In Testing*</span>'
        col1,col2= st.columns([1,10])
        with col1:
            st.markdown(beta_label_2, unsafe_allow_html=True)
        with col2:    
            autoencoder_predictions=st.button('Test Autoencoder')
            if 'autoencoder' not in st.session_state:
                st.session_state['autoencoder']= False
            if autoencoder_predictions or st.session_state.autoencoder:
                st.session_state['autoencoder']=True
                exercise_selector(conn)



app_2()