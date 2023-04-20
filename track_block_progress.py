import pandas as pd
import streamlit as st
import numpy as np
import psycopg2
from collections import defaultdict
from update_block import update_workout_in_block
import time
from collections import OrderedDict
import datetime
from in_progress_functions import test, check_if_in_progress_exists, update_in_progress_workout



def ordering_function_for_performed_workouts(num_workouts, num_weeks, dfs, actuals):

    num_unique_arrangements = [tuple(df['Exercise']) for df in dfs[:num_workouts]]
    #unique_arrangements = list(OrderedDict.fromkeys(num_unique_arrangements))

    orders = [pd.Series(order) for order in num_unique_arrangements]


    # define the indices of the dataframes that should use each ordering
    order_indices = {
        i: [j for j in range(i, num_workouts*num_weeks, num_workouts)] for i in range(num_workouts)
    }


    for i, df in enumerate(actuals):
        # determine which ordering to apply based on the index
        for order_idx, df_indices in order_indices.items():
            if i in df_indices:
                order = orders[order_idx]
                break
        else:
            raise ValueError(f"No ordering found for dataframe {i}")

        # get the unique exercises in the dataframe
        unique_exercises = df['Exercise'].unique()

        # check if any exercise in the dataframe is not in the categorical
        not_in_categorical = set(unique_exercises) - set(order)

        order=list(order)
        # loop through the exercises that are not in the categorical and replace NaN values with the original exercise name
        for exercise in not_in_categorical:
            original_exercise = df.loc[df['Exercise'] == exercise, 'Exercise'].iloc[0]
            order.append(original_exercise)


        order=pd.Series(order)
        # apply the ordering to the "Exercise" column of the dataframe
        df['Exercise'] = pd.Categorical(df['Exercise'], categories=order, ordered=True)
        df.sort_values('Exercise', inplace=True)
        
        
    return actuals

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

    # with st.form(key='Show Block Progress'):
    # Create a placeholder for the "Show Whole Block" button

    cursor=conn.cursor()

    cursor.execute(f"""SELECT id FROM blocks 
    WHERE client_id = (SELECT id FROM client WHERE name = '{name}')
    """)

    block_ids=cursor.fetchall()
    try:
        block_id=[i[0] for i in block_ids][-1]
    except:
        st.warning('No Block Found! Tell me to write one for you!')
        return

    cursor.execute(f"""select * from prescriptions WHERE block_id={block_id};""")

    prescriptions=cursor.fetchall()
    prescriptions
    prescriptions=np.asarray(prescriptions)

    cursor.execute(f"""select workouts_per_week from blocks where id ={block_id}""")
    num_workouts=cursor.fetchone()[0]

    block = pd.DataFrame(prescriptions[:, 2:7], columns=['Workout Number', 'Exercise', 'Sets', 'Reps', 'Weight'])
    block.set_index('Workout Number', inplace=True)
    unique_workout_nums = block.index.unique()
    unique_ex_ids=block['Exercise'].unique()

        # exercises=[]
        # for i in unique_ex_ids:
        #     cursor.execute(f"""
        #     SELECT exercise from exercises WHERE id={i}
        #     """)
        #     exercise=cursor.fetchone()[0]
        #     pair=(i, exercise)
        #     exercises.append(pair)
            
        # mapper=defaultdict(dict)
        # for i in exercises:
        #     mapper[i[0]]=i[1]
        # block['Exercise']=block['Exercise'].map(mapper)

    exercises_df = pd.read_sql("SELECT * FROM exercises", conn)
    exercises_df.set_index("id", inplace=True)
    exercises_dict = exercises_df["exercise"].to_dict()

    dfs=[block.loc[block.index==i] for i in unique_workout_nums]
    
    dfs = [df.assign(Exercise=df['Exercise'].map(exercises_dict)) for df in dfs]

    actuals=[]
    prescribed=[]
    for i in range(0,len(dfs)):
        try: #Needs to be a try except because some workouts may not have been performed yet and will throw error if you pass a wrokout number that hasn't been performed
            perf,pres,num=check_if_workout_performed(conn=conn, block_id=block_id, workout_number=i)
            actuals.append(perf)
            prescribed.append(pres)
        except:
            pass

    num_weeks = len(dfs) // num_workouts

    actuals=ordering_function_for_performed_workouts(num_weeks=num_weeks, num_workouts=num_workouts, dfs=dfs, actuals=actuals)
    
    dfs=[i.reset_index() for i in dfs]

    if len(actuals) == len(dfs):
        st.success('You have completed all of your prescribed workouts. Great job!')

    new_dfs = []
    performed_workout_numbers=[]
    for j, x in zip(actuals, dfs):
        performed_workout_number = j['Workout Number'].values[0]
        performed_workout_numbers.append(performed_workout_number)
        prescribed_workout_number = x['Workout Number'].values[0]
        #if performed_workout_number != prescribed_workout_number:
        new_dfs.append(j)

    #Ordering original_dfs because occasionally it gets pulled from database in random order causing error later on
    dfs=[i.reset_index(drop=True) for i in dfs]
    big_df = pd.concat(dfs)
    df_sorted = big_df.sort_values(by=['Workout Number'])
    dfs = [df_sorted.loc[df_sorted['Workout Number'] == i] for i in df_sorted['Workout Number'].unique()]
    dfs=[i.sort_index() for i in dfs]


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



    show_next_workout = st.button('See Next Workout')
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
            edited_df=st.experimental_data_editor(df, key=f"editor{number}", num_rows='dynamic')
            notes=st.text_input('Notes', key=f"notes_{number}")
            store_performed_workout=st.button(f'Submit Workout Number {number+1}')
            if store_performed_workout:
                edited_df['Workout Number']=workout_number_column
                update_workout_in_block(name, conn, edited_df, dfs, notes)
                st.success('Workout Submitted Successfully')
        else:
            st.success('You have completed all of your prescribed workouts. Great job!')


    #Code to organize the new_dfs according to the order of the prescribed workouts rather than performed workouts
    new_dfs=[i.reset_index(drop=True) for i in new_dfs]
    big_df = pd.concat(new_dfs)
    df_sorted = big_df.sort_values(by=['Workout Number'])
    new_dfs = [df_sorted.loc[df_sorted['Workout Number'] == i] for i in df_sorted['Workout Number'].unique()]
    new_dfs=[i.sort_index() for i in new_dfs]

    whole_block = st.button('Show Whole Block')
    if whole_block or st.session_state.whole_block:
        st.session_state['whole_block']=True
        #Split The screen for visualizations
        # col1, col2 = st.columns(2)
    
    
        #Print workouts for each week
        for week, indices in enumerate(week_indices):
            st.markdown(f"<h1 style='text-align: left;'>Week {week+1}--------------------------------</h1>", unsafe_allow_html=True)
            for index in indices:
                if index < total_prescribed_workouts:
                    df = new_dfs[index]
                    number = df['Workout Number'].unique()[0]
                    performed = number in performed_workout_numbers
                    if performed:
                        st.markdown(f"<h3 style='font-size: 20px; font-style: italic;'>You performed workout number {number+1}, Good Work!</h3>", unsafe_allow_html=True)
                        performed_df = new_dfs[index]
                        visualized_df=performed_df[['Exercise', 'Sets', 'Reps', 'Weight']]
                        #performed_df=performed_df[["Exercise", 'Sets', 'Reps', "Weight"]]
                        visualized_df=visualized_df.reset_index(drop=True)
                        st.dataframe(visualized_df.style.set_properties(**{'background-color': 'lightgreen'}))
                    else:
                        st.markdown(f"<h3 style='font-size: 20px; font-style: italic;'>You have not yet performed workout number {number+1}.</h3>", unsafe_allow_html=True)
                        workout_number_column=df['Workout Number']
                        df=df.drop(columns='Workout Number')
                        df=df.reset_index(drop=True)
                        #try/except block for instance of where user hits both buttons, the keys for the current workout (appearing first) and the same wokrout later on are the same
                        #This try/except block also controls if a subject adds a new exercise but hasnt yet hit the 'done' checkbox
                        try:
                            #df['Done']=[False for _ in range(len(df.index))]
                            notes=None
                            if f'edited_df{number}' not in st.session_state:
                                st.session_state[f'edited_df{number}'] = df
                            edited_df = st.experimental_data_editor(st.session_state[f'edited_df{number}'], num_rows='dynamic', on_change=update_in_progress_workout, args=(conn, st.session_state[f'edited_df{number}'], name, workout_number_column[0], notes))
                            
                            notes=st.text_input('Workout Notes', key=f"notes_{number}")
                            #update_in_progress_workout(conn, edited_df, name, workout_number_column[0], notes)
                            store_performed_workout=st.button(f'Submit Workout Number {number+1}')
                        except st.errors.DuplicateWidgetID:
                            pass
                        except ValueError as e:
                            if str(e) == "Cannot mask with non-boolean array containing NA / NaN values":
                                st.error("Make sure to hit the checkbox after entering a new exercise")
                                st.stop()
                        if store_performed_workout:
                            edited_df['Workout Number']=workout_number_column
                            result=update_workout_in_block(name, conn, edited_df, dfs, notes)
                            if result is not None:
                                st.success('Workout Submitted Successfully')
                                time.sleep(1)
                                st.experimental_rerun()
                                

    #         # text = 'Visualization Features Coming Soon!'
    #         # formatted_text = '<br>'.join(text.split())
    #         # st.markdown(f"<h2 style='font-size: 36px; text-align: center;'>{formatted_text}</h2>", unsafe_allow_html=True)


    return dfs, actuals, new_dfs, edited_df

