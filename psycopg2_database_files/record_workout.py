import psycopg2
import streamlit as st
import pandas as pd
import datetime
import time

def record_workout(conn, name, selected_exercises, reps, sets, weights):

    # Check if client exists in database
    cursor = conn.cursor()
    cursor.execute("SELECT EXISTS(SELECT 1 FROM client WHERE name=%s)", (f"{name}",))
    exists = cursor.fetchone()[0]

    if exists:
        cursor.execute("SELECT id FROM client WHERE name=%s", (name,))
        client_id = cursor.fetchone()[0]
        # If client exists, insert data into existing client table
        cursor.execute(
            "INSERT INTO sessions (session_date, client_id) \
            VALUES (%s, %s) RETURNING id",
            (datetime.datetime.now(), client_id)
    )
    
        conn.commit()
        session_id=cursor.fetchone()[0]
        exercise_ids = []
        for ex in selected_exercises:
            cursor.execute("SELECT id FROM exercises WHERE exercise = %s", (ex,))
            exercise_id = cursor.fetchone()[0]
            exercise_ids.append(exercise_id)

        exercise_order = {ex: i for i, ex in enumerate(selected_exercises)}
        exercise_info = []
        for ex_id, ex, s, r, w in zip(exercise_ids, selected_exercises, sets, reps, weights):
            order = exercise_order[ex]
            exercise_info.append((ex_id, s, r, w, order))
            exercise_info.sort(key=lambda x: x[4])


        # Insert the workout and exercise ids into the workout_exercises table
        for ex_id, s, r, w, _ in exercise_info:
            cursor.execute(f"""INSERT INTO workout_exercises (workout_id, exercise_id, reps, sets, weight, client_id)
            VALUES ({session_id}, {ex_id}, {r}, {s}, {w}, {client_id})""")


        conn.commit()
    else:
        # If client doesn't exist, create new client table and insert data
        cursor.execute(
            "INSERT INTO client (name) VALUES (%s) RETURNING id",
            (f"{name}",)
        )

        conn.commit()
        client_id = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO sessions (session_date, client_id) VALUES (%s, %s) RETURNING id",
            (datetime.datetime.now(), client_id)
        )

        conn.commit()

        session_id=cursor.fetchone()[0]
        exercise_ids = []
        for ex in selected_exercises:
            cursor.execute("SELECT id FROM exercises WHERE exercise = %s", (ex,))
            exercise_id = cursor.fetchone()[0]
            exercise_ids.append(exercise_id)

        # Insert the workout and exercise ids into the workout_exercises table
        for ex_id,rep,set,weight in zip(exercise_ids, reps, sets, weights):
            cursor.execute(f"""INSERT INTO workout_exercises (workout_id, exercise_id, reps, sets, weight)
                       VALUES ({session_id}, {ex_id}, {rep}, {set}, {weight})""")
        conn.commit()