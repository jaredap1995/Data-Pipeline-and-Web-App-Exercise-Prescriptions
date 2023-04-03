import psycopg2
import streamlit as st
import pandas as pd
import datetime
import time


def record_block(conn, name, block, length_of_block):

    # Check if client exists in database
    cursor = conn.cursor()
    cursor.execute("SELECT EXISTS(SELECT 1 FROM client WHERE name=%s)", (f"{name}",))
    exists = cursor.fetchone()[0]

    #Calculate workouts per week
    total_workouts=len(block)
    workouts_per_week=total_workouts/length_of_block

    if exists:
        cursor.execute("SELECT id FROM client WHERE name=%s", (name,))
        client_id = cursor.fetchone()[0]

        cursor.execute("INSERT INTO blocks (client_id, workouts, workouts_per_week) \
            VALUES (%s, %s) RETURNING id", 
            (client_id, len(block), workouts_per_week))
        
        block_id=cursor.fetchone()[0]
        conn.commit()

     

        #Go over every workout in block and return the exercise_ids from the database for each workout
        exercise_ids = []
        for df in block:
            sel_ex=df['Exercise']
            b_ex=[]
            for ex in sel_ex:
                cursor.execute("SELECT id FROM exercises WHERE exercise = %s", (ex,))
                exercise_id = cursor.fetchone()[0]
                b_ex.append(exercise_id)
            exercise_ids.append(b_ex)
    
        #Attach the Exercise ids to the block
        for ex_id,df in zip(exercise_ids, block):
            s=pd.Series(ex_id)
            df['ex_id']=ex_id

#Use the columns and the new ex_id column to enter all the data into the prescirptions table for each client
        for idx_df, df in enumerate(block):
            for e_i, s,r,w in zip(df['ex_id'], df['Sets'], df['Reps'], df['Weight']):
                cursor.execute("""INSERT INTO prescriptions (block_id, workout_number, exercise_id, sets, reps, weight)
            VALUES (%s, %s, %s, %s, %s, %s)""",
            (block_id, idx_df, e_i, s, r, w))
                
        conn.commit()

    else:
        # If client doesn't exist, create new client table and insert data
        cursor.execute(
            "INSERT INTO client (name) VALUES (%s) RETURNING id",
            (f"{name}",)
        )

        conn.commit()
        client_id = cursor.fetchone()[0]

        cursor.execute("INSERT INTO blocks (client_id, workouts, workouts_per_week) \
            VALUES (%s, %s) RETURNING id", 
            (client_id, len(block), workouts_per_week))
        
        block_id=cursor.fetchone()[0]
        conn.commit()

     

        #Go over every workout in block and return the exercise_ids from the database for each workout
        exercise_ids = []
        for df in block:
            sel_ex=df['Exercise']
            b_ex=[]
            for ex in sel_ex:
                cursor.execute("SELECT id FROM exercises WHERE exercise = %s", (ex,))
                exercise_id = cursor.fetchone()[0]
                b_ex.append(exercise_id)
            exercise_ids.append(b_ex)
    
        #Attach the Exercise ids to the block
        for ex_id,df in zip(exercise_ids, block):
            s=pd.Series(ex_id)
            df['ex_id']=ex_id

        #Use the columns and the new ex_id column to enter all the data into the prescirptions table for each client
        for idx_df, df in enumerate(block):
            for e_i, s,r,w in zip(df['ex_id'], df['Sets'], df['Reps'], df['Weight']):
                cursor.execute("""INSERT INTO prescriptions (block_id, workout_number, exercise_id, sets, reps, weight)
            VALUES (%s, %s, %s, %s, %s, %s)""",
            (block_id, idx_df, e_i, s, r, w))
                
        conn.commit()
    
