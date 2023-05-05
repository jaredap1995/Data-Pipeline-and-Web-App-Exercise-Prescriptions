import streamlit as st
import psycopg2
import time
from record_workout import record_workout
import datetime
from track_workouts import track_workouts
import pandas as pd
import numpy as np
from testing_coach_and_prescriptions import prescribe_block
from record_prescriptions import record_block


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
                time.sleep(1) # wait for 2 seconds before rerunning the script
                st.experimental_rerun()
            submit_button = st.form_submit_button(label='Record Sets, Reps, and Weight 2')
            if submit_button or st.session_state.exercise_selection:
                st.session_state['exercise_selection']=True
                df=pd.DataFrame({'Exercise': selected_exercises,
                                'Sets': np.zeros(len(selected_exercises)),
                                'Reps': np.zeros(len(selected_exercises)),
                                'Weight': np.zeros(len(selected_exercises))})
                edited_df = st.experimental_data_editor(df)
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
    prescribe_block_button = st.button('Coach Center')
    if 'coach_center' not in st.session_state:
        st.session_state['coach_center']=False
    if prescribe_block_button or st.session_state.coach_center:
        name=name
        st.session_state['coach_center']=True
        prescribe_block(conn, name)
    
    return name

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

show(st.session_state['name'], st.session_state['conn'])


