import streamlit as st
import pandas as pd
import psycopg2
import datetime
import time
from update_block import update_workout_in_block
from collections import defaultdict
import numpy as np




def test(conn, in_progress, name, workout_number, notes):
    cursor=conn.cursor()


    cursor.execute("SELECT id FROM client WHERE name=%s", (name,))
    client_id = cursor.fetchone()[0]

    cursor.execute(f"""SELECT id FROM blocks WHERE client_id = %s
    """, (client_id,))

    block_ids=cursor.fetchall()
    block_id=[i[0] for i in block_ids][-1]

    perf_exercise_ids=[]
    performed_ex=in_progress['Exercise']
    for perf_ex in performed_ex:
        cursor.execute("SELECT EXISTS(SELECT 1 FROM exercises WHERE exercise=%s);",(perf_ex,))
        exists = cursor.fetchone()[0]
        if not exists:
            cursor.execute("INSERT INTO exercises (exercise) VALUES (%s);", (perf_ex,))
        cursor.execute("SELECT id FROM exercises WHERE exercise = %s;", (perf_ex,))
        perf_exercise_id = cursor.fetchone()[0]
        perf_exercise_ids.append(perf_exercise_id)

    workout_number=int(workout_number)

    cursor.execute("""DELETE FROM in_progress WHERE workout_number=%s AND client_id=%s""", (workout_number, client_id))

    for ex_id,sets,reps,weight in zip(perf_exercise_ids,in_progress['Sets'],in_progress['Reps'],in_progress['Weight']):
        try:
            cursor.execute("""
            INSERT INTO in_progress (workout_number, exercise_id, sets, reps, weight, block_id, client_id) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)""", 
            (workout_number, ex_id, sets, reps, weight, block_id, client_id))
        except psycopg2.errors.NumericValueOutOfRange:
            st.error("""Please make sure all values are integers before you store workout.""")
            st.stop()
    
    conn.commit()

    return in_progress


#In progress==edited_df
def update_in_progress_workout(conn, in_progress, name, workout_number, notes=None):
    # st.write(in_progress)
    #st.write(workout_number)
    st.dataframe(in_progress)
    st.session_state['df_value']=in_progress
    #st.write(st.session_state['df_value'].style.set_properties(**{'background-color': 'lightgreen'}))

    while True:
        if in_progress.shape[0] > 0:
            in_progress=test(conn, in_progress, name, workout_number, notes)
            #st.write(ex_ids)
            break
        else:
            # st.write("No exercises selected")
            break
    
    return in_progress



def check_if_in_progress_exists(conn, name):
    cursor=conn.cursor()
    cursor.execute("SELECT id FROM client WHERE name=%s", (name,))
    client_id = cursor.fetchone()[0]

    cursor.execute("SELECT * FROM in_progress WHERE client_id = %s", (client_id,))
    rows=cursor.fetchall()
    if rows:
            st.write("## Unfinished workout in progress! Would you like to continue?")
            # continue_workout = st.button("Continue Workout")
            # if continue_workout or st.session_state.continued:
            st.session_state['continued'] = True
            exercises_df = pd.read_sql("SELECT * FROM exercises", conn)
            exercises_df.set_index("id", inplace=True)
            exercises_dict = exercises_df["exercise"].to_dict()

            #Getting block_id for query
            cursor.execute(f"""SELECT id FROM blocks WHERE client_id = %s
            """, (client_id,))

            block_ids=cursor.fetchall()
            block_id=[i[0] for i in block_ids][-1]

            df=pd.DataFrame(rows)
            workout_number=df.iloc[0,1]
            df=df.iloc[:, 2:6]
            df.columns=['Exercise', 'Sets', 'Reps', 'Weight']
            df = df.assign(Exercise=df['Exercise'].map(exercises_dict))
            if st.session_state['df_value'] is None:
                st.session_state['df_value']=df
            continued_workout=st.experimental_data_editor(df, num_rows='dynamic')
            st.session_state['df_value']=continued_workout
            # try:
                # if not continued_workout.equals(st.session_state["df_value"]):
                 # st.session_state['continued_workout'] = True
                    
            update_in_progress_workout(conn, continued_workout, name, workout_number)
            # except ValueError as e:
            #     if str(e) == "Cannot mask with non-boolean array containing NA / NaN values":
            #         st.error("Make sure to hit the checkbox after entering a new exercise")
            #         st.stop()

            
            #Grabbing dfs for submission
            cursor.execute("""select * from prescriptions WHERE block_id=%s;""", (block_id,))

            prescriptions=cursor.fetchall()
            prescriptions
            prescriptions=np.asarray(prescriptions)

            cursor.execute("""select workouts_per_week from blocks where id =%s;""", (block_id,))
            num_workouts=cursor.fetchone()[0]

            block = pd.DataFrame(prescriptions[:, 2:7], columns=['Workout Number', 'Exercise', 'Sets', 'Reps', 'Weight'])
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


            store_workout=st.button("Store Workout")
            reset_workout=st.button("Reset Workout")


            st.write(continued_workout.style.set_properties(**{'background-color': 'red'}))
            if store_workout:
                continued_workout['Workout Number']=workout_number
                result=update_workout_in_block(name, conn, continued_workout, dfs)
                if result is not None:
                    st.success('Workout Submitted Successfully')
                    time.sleep(1)
                    st.experimental_rerun()
                
            if reset_workout:
                #A function that will delete the in_progress row and start over
                cursor.execute("""DELETE FROM in_progress WHERE client_id=%s""", (client_id,))
                conn.commit()
                st.experimental_rerun()


            return continued_workout
    else:
        return None




