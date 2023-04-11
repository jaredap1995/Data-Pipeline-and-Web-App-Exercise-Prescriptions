
import streamlit as st
import psycopg2
import datetime
import pandas as pd
import numpy as np

def update_workout_in_block(name, conn, edited_workout, dfs, notes):

    exercises_per_workout=[len(i) for i in edited_workout]

    # if any(ex == 1 for ex in exercises_per_workout):
    #      st.error("""You appear to have a workout with only a single exercise. Please add an additional row with None in all columns if this is true""")
    # else:
    #      pass

    edited_workout=[edited_workout]
    modified_workouts = []
    for df_p, df_a in zip(dfs, edited_workout):
        if not (df_p.equals(df_a)):
            if 'Workout Number' in df_p.columns:
                modified_workouts.append(df_a['Workout Number'].unique()[0])
            else:
                modified_workouts.append('Unknown Workout Number')
        if (df_p.equals(df_a)):
            st.success("Way to Follow your program exactly! Great Work!")
            if 'Workout Number' in df_p.columns:
                modified_workouts.append(df_a['Workout Number'].unique()[0])



    # modified_workouts = [df_p['Workout Number'].unique()[0] for df_p, df_a in zip(dfs, actual_workouts_2) if not (df_p.values == df_a.values).all()]
    # st.write(modified_workouts)

    if len(modified_workouts)>1:
         st. error("""Please Only Modify The Workout You Performed Today. If you need to modify mutliple workouts from your block
         please modify one workout, then refresh the page and modify the next.""")
         return
    else:
         pass
    # Check if client exists in database
    # st.write(df_p)


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
            "INSERT INTO sessions (session_date, client_id, notes) VALUES (%s, %s, %s) RETURNING id",
            (datetime.datetime.now(), client_id, notes))
        
        session_id=cursor.fetchone()[0]

        perf_exercise_ids = []
        WOD=edited_workout[0]
        # prescribed_workout=dfs[WOD_]
        performed_ex=WOD['Exercise']
        for perf_ex in performed_ex:
            cursor.execute("SELECT EXISTS(SELECT 1 FROM exercises WHERE exercise=%s);",(perf_ex,))
            exists = cursor.fetchone()[0]
            if not exists:
                cursor.execute("INSERT INTO exercises (exercise) VALUES (%s);", (perf_ex,))
            cursor.execute("SELECT id FROM exercises WHERE exercise = %s;", (perf_ex,))
            perf_exercise_id = cursor.fetchone()[0]
            perf_exercise_ids.append(perf_exercise_id)

        exercise_ids=pd.Series(perf_exercise_ids)
        WOD=WOD.reset_index(drop=True)
        WOD['ex_id']=exercise_ids
        #dfs[WOD_]=WOD
        
        for e_i, s,r,w in zip(WOD['ex_id'], WOD['Sets'], WOD['Reps'], WOD['Weight']):
                #st.write(block_id, session_id, e_i, s, r, w)
                try:
                    cursor.execute("""INSERT INTO workout_exercises (block_id, workout_id, exercise_id, sets, reps, weight)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (block_id, session_id, e_i, s, r, w))
                except:
                    st.error("There seems to be an error in your Sets, Reps, and Weight. Please check your input and try again.")
                    return
                    
        try:     
            cursor.execute("""INSERT INTO actual_to_prescription(block_id, workout_number, session_id) VALUES (%s, %s, %s) """, (block_id, WOD_, session_id))
        except:
            st.error("""There appears to be an error with the workout being submitted. If you have recently hit submit it is likely because the workout is already submitted. 
                    If you have not submitted a workout recently and everything looks correct, reach out directly to your coach.""")
            return
        conn.commit()

    return WOD
        


        # # actual_workouts_2=[i.groupby(['Workout Number', 'Exercise']).sum() for i in actual_workouts_2]
        # # dfs=[i.groupby(['Workout Number', 'Exercise']).sum() for i in dfs]




