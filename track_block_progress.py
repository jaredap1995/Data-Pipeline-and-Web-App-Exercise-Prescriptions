import pandas as pd
import streamlit as st
import numpy as np
import psycopg2
from collections import defaultdict
from update_block import update_workout_in_block
import time

def check_if_workout_performed(conn, block_id, workout_number):


    exercises_df = pd.read_sql("SELECT * FROM exercises", conn)
    exercises_df.set_index("id", inplace=True)
    exercises_dict = exercises_df["exercise"].to_dict()

    cursor=conn.cursor()
    cursor.execute(f"""SELECT DISTINCT
        p.workout_number AS workout_number,
        p.exercise_id AS prescribed_exercise,
        p.sets AS prescribed_sets,
        p.reps AS prescribed_reps,
        p.weight AS prescribed_weight,
        we.exercise_id AS performed_exercise,
        we.sets AS actual_sets,
        we.reps AS actual_reps,
        we.weight AS actual_weight
    FROM 
        prescriptions p
        JOIN workout_exercises we ON p.block_id = we.block_id 
        JOIN actual_to_prescription atp ON p.block_id = atp.block_id 
            AND we.workout_id = atp.session_id 
            AND p.workout_number = atp.workout_number
    WHERE 
        p.block_id = {block_id}
            AND atp.block_id = {block_id}
            AND atp.workout_number = {workout_number};
    """)
    table=cursor.fetchall()

    # Check if any rows were returned
    if len(table) > 0:
        table_headers=['workout_number','prescribed_exercise', 'prescribed_sets', 'prescribed_reps', 'prescribed_weight', 
                'performed_exercise', 'actual_sets', 'actual_reps', 'actual_weight']
        table=pd.DataFrame(table, columns=table_headers)
        workout_number_s=table['workout_number']

        prescribed_df=table.loc[:,:'prescribed_weight'].drop_duplicates().reset_index(drop=True) #.drop(columns='workout_number')
        performed_df=table.loc[:,'performed_exercise':].drop_duplicates().reset_index(drop=True)
        performed_df['Workout Number']=workout_number_s

        prescribed_df['prescribed_exercise']=prescribed_df['prescribed_exercise'].map(exercises_dict)
        performed_df['performed_exercise']=performed_df['performed_exercise'].map(exercises_dict)

        performed_order=['Workout Number', 'performed_exercise','actual_sets', 'actual_reps', 'actual_weight']
        performed_df=performed_df[performed_order]
        performed_df.columns=['Workout Number','Exercise', 'Sets', 'Reps', 'Weight']
        # prescribed_df.columns=['Workout Number','Exercise', 'Sets', 'Reps', 'Weight']

        return performed_df, prescribed_df, workout_number
    else:
        return None
        

def show_progress_in_block(conn, name):

    if name == '':
        st.warning('Please enter your name.')
        return

    with st.form(key='Show Block Progress'):
        # Create a placeholder for the "Show Whole Block" button

        cursor=conn.cursor()

        cursor.execute(f"""SELECT id FROM blocks 
        WHERE client_id = (SELECT id FROM client WHERE name = '{name}')
        """)

        block_ids=cursor.fetchall()
        block_id=[i[0] for i in block_ids][-1]

        cursor.execute(f"""select * from prescriptions WHERE block_id={block_id};""")

        prescriptions=cursor.fetchall()
        prescriptions
        prescriptions=np.asarray(prescriptions)

        cursor.execute(f"""select workouts_per_week from blocks where id ={block_id}""")
        num_workouts=cursor.fetchone()[0]

        block = pd.DataFrame(prescriptions[:, 2:], columns=['Workout Number', 'Exercise', 'Sets', 'Reps', 'Weight'])
        block.set_index('Workout Number', inplace=True)
        unique_workout_nums = block.index.unique()
        unique_ex_ids=block['Exercise'].unique()

        exercises=[]
        for i in unique_ex_ids:
            cursor.execute(f"""
            SELECT exercise from exercises WHERE id={i}
            """)
            exercise=cursor.fetchone()[0]
            pair=(i, exercise)
            exercises.append(pair)
            
        mapper=defaultdict(dict)
        for i in exercises:
            mapper[i[0]]=i[1]
        block['Exercise']=block['Exercise'].map(mapper)

        dfs=[block.loc[block.index==i] for i in unique_workout_nums]
        
        actuals=[]
        prescribed=[]
        for i in range(0,len(dfs)):
            try:
                perf,pres,num=check_if_workout_performed(conn=conn, block_id=block_id, workout_number=i)
                actuals.append(perf)
                prescribed.append(pres)
            except:
                pass
            
        # unique_exercises = set()
        # unique_dfs = set()

        # for df in dfs:
        #     exercise_values = df['Exercise'].unique()
        #     if frozenset(exercise_values) not in unique_dfs:
        #         # this DataFrame has unique exercises not seen before, add them to the set
        #         unique_dfs.add(frozenset(exercise_values))
        #         unique_exercises.update(exercise_values)
        
        dfs=[i.reset_index() for i in dfs]

        new_dfs = []
        performed_workout_numbers=[]
        for j, x in zip(actuals, dfs):
            performed_workout_number = j['Workout Number'].values[0]
            performed_workout_numbers.append(performed_workout_number)
            prescribed_workout_number = x['Workout Number'].values[0]
            #if performed_workout_number != prescribed_workout_number:
            new_dfs.append(j)

        performed_workout_numbers.reverse()
        for i in performed_workout_numbers:
            dfs.pop(i)

        new_dfs.extend(dfs)

        performed_workout_numbers.reverse()
        
        # Define parameters
        workouts_per_week = num_workouts
        total_prescribed_workouts = len(new_dfs)
        total_weeks = total_prescribed_workouts // workouts_per_week
        step = [i*workouts_per_week for i in range(total_weeks)]

        # Get list of indices for each week
        week_indices = [list(range(start, start+workouts_per_week)) for start in step]

        edited_df=None

        show_next_workout = st.form_submit_button('See Next Workout')
        if show_next_workout or st.session_state.next_workout:
            st.session_state['next_workout']=True
            next_workout_index = None
            for index, df in enumerate(new_dfs):
                number = df['Workout Number'].unique()[0]
                if number not in performed_workout_numbers:
                    next_workout_index = index
                    break

            if next_workout_index is not None:
                df = new_dfs[next_workout_index]
                number = df['Workout Number'].unique()[0]
                st.markdown(f"<h1 style='text-align: left;'>Next Workout: Workout {number+1}--------------------------------</h1>", unsafe_allow_html=True)
                workout_number_column=df['Workout Number']
                df=df.drop(columns='Workout Number')
                edited_df=st.experimental_data_editor(df, key=number, num_rows='dynamic')
                store_performed_workout=st.form_submit_button(f'Submit Workout Number {number+1}')
                if store_performed_workout:
                    edited_df['Workout Number']=workout_number_column
                    update_workout_in_block(name, conn, edited_df, dfs)
                    st.success('Workout Submitted Successfully')
            else:
                st.write('You have completed all of your prescribed workouts. Great job!')


        whole_block = st.form_submit_button('Show Whole Block')
        if whole_block or st.session_state.whole_block:
            st.session_state['whole_block']=True
            #Split The screen for visualizations
            col1, col2 = st.columns(2)

            # Print workouts for each week
            with col1:
                for week, indices in enumerate(week_indices):
                    st.markdown(f"<h1 style='text-align: left;'>Week {week+1}--------------------------------</h1>", unsafe_allow_html=True)
                    for index in indices:
                        if index < total_prescribed_workouts:
                            df = new_dfs[index]
                            number = df['Workout Number'].unique()[0]
                            performed = number in performed_workout_numbers
                            if performed:
                                st.markdown(f"<h3 style='font-size: 20px; font-style: italic;'>You performed workout number {number+1}, Good Work!</h3>", unsafe_allow_html=True)
                                performed_index = performed_workout_numbers.index(number)
                                performed_df = actuals[performed_index]
                                visualized_df=performed_df[['Exercise', 'Sets', 'Reps', 'Weight']]
                                #performed_df=performed_df[["Exercise", 'Sets', 'Reps', "Weight"]]

                                """Show dataframe without workout number, save column, then append back on after modification"""

                                st.dataframe(visualized_df.style.set_properties(**{'background-color': 'lightgreen'}))
                            else:
                                st.markdown(f"<h3 style='font-size: 20px; font-style: italic;'>You have not yet performed workout number {number+1}.</h3>", unsafe_allow_html=True)
                                workout_number_column=df['Workout Number']
                                df=df.drop(columns='Workout Number')
                                try: #try/except block for instance of where user hits both buttons, the keys for the current workout (appearing first) and the same wokrout later on are the same
                                    edited_df=st.experimental_data_editor(df, key=number, num_rows='dynamic')
                                    store_performed_workout=st.form_submit_button(f'Submit Workout Number {number+1}')
                                except:
                                    pass
                                if store_performed_workout:
                                    edited_df['Workout Number']=workout_number_column
                                    update_workout_in_block(name, conn, edited_df, dfs)
                                    st.write('Workout Submitted Successfully')
                                    

            with col2:
                text = 'Visualization Features Coming Soon!'
                formatted_text = '<br>'.join(text.split())
                st.markdown(f"<h2 style='font-size: 36px; text-align: center;'>{formatted_text}</h2>", unsafe_allow_html=True)


    return dfs, actuals, new_dfs, edited_df
