
import streamlit as st
import psycopg2
import datetime
import pandas as pd
import numpy as np

def update_workout(name, conn, actual_workouts_2, dfs):

    exercises_per_workout=[len(i) for i in actual_workouts_2]

    # if any(ex == 1 for ex in exercises_per_workout):
    #      st.error("""You appear to have a workout with only a single exercise. Please add an additional row with None in all columns if this is true""")
    # else:
    #      pass

    modified_workouts = []
    for df_p, df_a in zip(dfs, actual_workouts_2):
        if not (df_p.equals(df_a)):
            if 'Workout Number' in df_p.columns:
                modified_workouts.append(df_p['Workout Number'].unique()[0])
            else:
                modified_workouts.append('Unknown Workout Number')



    # modified_workouts = [df_p['Workout Number'].unique()[0] for df_p, df_a in zip(dfs, actual_workouts_2) if not (df_p.values == df_a.values).all()]
    # st.write(modified_workouts)

    if len(modified_workouts)>1:
         st. error("""Please Only Modify The Workout You Performed Today. If you need to modify mutliple workouts from your block
         please modify one workout, then refresh the page and modify the next.""")
         return
    else:
         pass
    # Check if client exists in database

    if not modified_workouts:
         st.success("Way to Follow your program exactly! Great Work!")
         #Rest of my code that submits into workout exercises exactly what they did....i'll figure that out later.
         return actual_workouts_2, actual_workouts_2[0] #This is a placeholder until later
    else:
         pass

    WOD_=int(modified_workouts[0])
    cursor = conn.cursor()
    cursor.execute("SELECT EXISTS(SELECT 1 FROM client WHERE name=%s)", (f"{name}",))
    exists = cursor.fetchone()[0]

    if exists:
        cursor.execute("SELECT id FROM client WHERE name=%s", (name,))
        client_id = cursor.fetchone()[0]

        cursor.execute(f"""SELECT id FROM blocks WHERE client_id = %s
        """, (client_id,))

        block_ids=cursor.fetchall()
        block_id=[i[0] for i in block_ids][-1]
        
    #     # num_workouts=cursor.fetchall()
    #     # num_workouts=[i[0] for i in num_workouts]
    #     # num_workouts=pd.Series(num_workouts).unique()

    #     # matched_workouts = num_workouts[~np.isin(num_workouts, WOD_)]
        
        cursor.execute(
            "INSERT INTO sessions (session_date, client_id) VALUES (%s, %s) RETURNING id",
            (datetime.datetime.now(), client_id))
        
        conn.commit()
        session_id=cursor.fetchone()[0]

        exercise_ids = []
        WOD=actual_workouts_2[WOD_]
        sel_ex=WOD['Exercise']
        for ex in sel_ex:
            cursor.execute("SELECT EXISTS(SELECT 1 FROM exercises WHERE exercise=%s);",(ex,))
            exists = cursor.fetchone()[0]
            if not exists:
                cursor.execute("INSERT INTO exercises (exercise) VALUES (%s);", (ex,))
            cursor.execute("SELECT id FROM exercises WHERE exercise = %s;", (ex,))
            exercise_id = cursor.fetchone()[0]
            exercise_ids.append(exercise_id)

        conn.commit()

        exercise_ids=pd.Series(exercise_ids)
        WOD['ex_id']=exercise_ids
        actual_workouts_2[WOD_]=WOD

        for e_i, s,r,w in zip(WOD['ex_id'], WOD['Sets'], WOD['Reps'], WOD['Weight']):
                #st.write(block_id, session_id, e_i, s, r, w)
                try:
                    cursor.execute("""INSERT INTO workout_exercises (block_id, workout_id, exercise_id, sets, reps, weight)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (block_id, session_id, e_i, s, r, w))
                except:
                    st.error("Please Fill in all Values for Sets, Reps, and Weight")
                    return
                     
                
        cursor.execute("""INSERT INTO actual_to_prescription(block_id, workout_number, session_id) VALUES (%s, %s, %s) """, (block_id, WOD_, session_id))
                
        conn.commit()

    return WOD
        


        # # actual_workouts_2=[i.groupby(['Workout Number', 'Exercise']).sum() for i in actual_workouts_2]
        # # dfs=[i.groupby(['Workout Number', 'Exercise']).sum() for i in dfs]



