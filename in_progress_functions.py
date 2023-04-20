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

    # cursor.execute("""
    #     CREATE UNIQUE INDEX IF NOT EXISTS idx_exercise_sets_reps_weight 
    #     ON in_progress (exercise_id, sets, reps, weight);
    # """)
    
    for ex_id,sets,reps,weight in zip(perf_exercise_ids,in_progress['Sets'],in_progress['Reps'],in_progress['Weight']):
        # try:
            cursor.execute("""
            INSERT INTO in_progress (workout_number, exercise_id, sets, reps, weight, block_id, client_id) 
            VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT 
            (workout_number, exercise_id, sets, reps, weight, block_id, client_id);""", 
            (workout_number, ex_id, sets, reps, weight, block_id, client_id))
        # except psycopg2.errors.NumericValueOutOfRange:
        #     st.error("""Please make sure all values are integers before you store workout.""")
        #     st.stop()
    
    conn.commit()

    return perf_exercise_ids, in_progress

def set_state():
    st.session_state.continued = True


#In progress==edited_df
def update_in_progress_workout(conn, in_progress, name, workout_number, notes=None):
    #in_progress = in_progress.loc[in_progress["Done"]][["Exercise", "Sets", "Reps", "Weight"]]
    # st.write(in_progress)
    st.write(workout_number)

    while True:
        if in_progress.shape[0] > 0:
            ex_ids, in_progress=test(conn, in_progress, name, workout_number, notes)
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
            df_in_progress = df.assign(Exercise=df['Exercise'].map(exercises_dict))

            #Find what is left to do (original prescription)
            cursor.execute("""SELECT * FROM prescriptions WHERE block_id IN 
                            (SELECT %s FROM in_progress) AND client_id IN 
                            (SELECT %s FROM in_progress)
                            AND workout_number IN (SELECT %s FROM in_progress);""", 
                            (block_id, client_id,int(workout_number)))
            
            df_original=pd.DataFrame(cursor.fetchall())
            df_original=df_original.iloc[:, 3:7]
            df_original.columns=['Exercise', 'Sets', 'Reps', 'Weight']
            df_original=df_original.assign(Exercise=df_original['Exercise'].map(exercises_dict))

            #Concatenate what has been done, with what is left to do

            df=pd.merge(df_in_progress, df_original, on=['Exercise'], how='outer', suffixes=("_in_progress", "_original"))
            df.index=df['Exercise']
            df=df.drop(columns='Exercise')
            original=df.iloc[:len(df_in_progress),:3]
            remaining=df.iloc[len(df_in_progress):,3:]
            original.columns=['Sets', 'Reps', 'Weight']
            remaining.columns=['Sets', 'Reps', 'Weight']
            df_2=pd.concat([original, remaining], axis=0)
            df_2=df_2.reset_index()
            #df_2['Done']=[False for _ in range(len(df.index))]
            continued_workout=st.experimental_data_editor(df_2, num_rows='dynamic')
            try:
                if not (continued_workout.equals(df_2)):
                    diff_rows = continued_workout[continued_workout != df_2].dropna(how='all').index
                    diff_rows=continued_workout.loc[diff_rows]
                    update_in_progress_workout(conn, diff_rows, name, workout_number)
            except ValueError as e:
                if str(e) == "Cannot mask with non-boolean array containing NA / NaN values":
                    st.error("Make sure to hit the checkbox after entering a new exercise")
                    st.stop()

            
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

            if store_workout:
                continued_workout['Workout Number']=workout_number
                result=update_workout_in_block(name, conn, continued_workout, dfs)
                if result is not None:
                    st.success('Workout Submitted Successfully')
                    time.sleep(1)
                    st.experimental_rerun()
                
            if reset_workout:
                ##A function that will delete the in_progress row and start over
                cursor.execute("""DELETE FROM in_progress WHERE client_id=%s""", (client_id,))
                conn.commit()
                st.experimental_rerun()


            return continued_workout
    else:
        return None




