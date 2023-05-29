import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from collections import defaultdict
import time
from psycopg2_database_files.update_actuals import update_workout

def retrieve_block(conn, name):

    # if 'update_workout' not in st.session_state:
    #     st.session_state['update_workout']=False


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
        
    unique_exercises = set()
    unique_dfs = set()

    for df in dfs:
        exercise_values = df['Exercise'].unique()
        if frozenset(exercise_values) not in unique_dfs:
            # this DataFrame has unique exercises not seen before, add them to the set
            unique_dfs.add(frozenset(exercise_values))
            unique_exercises.update(exercise_values)
    
    workouts_per_week=len(unique_dfs)
    week_num = 1
    if len(unique_dfs) > 0:
        st.write(f"Week {week_num}")
        #st.dataframe(dfs[0])
        dfs=[i.reset_index() for i in dfs]
        edited_1=st.experimental_data_editor(dfs[0], key="df_0", num_rows='dynamic')
        actual_workouts_2=[edited_1]
        for i in range(1, len(dfs)):
            if i % workouts_per_week == 0:
                week_num += 1
                st.write(f"Week {week_num}")
            #st.dataframe(dfs[i])
            edited_2=st.experimental_data_editor(dfs[i], key=f"df_{i}", num_rows='dynamic')
            actual_workouts_2.append(edited_2)       #st.session_state[f'df_{i}'])
        #actual_workouts_2.extend(edited_1)                 #st.session_state['df_0'])

    return actual_workouts_2, dfs














