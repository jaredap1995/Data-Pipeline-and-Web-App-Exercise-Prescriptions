import psycopg2
import streamlit as st
import pandas as pd





def track_weight_changes(conn, name, selected_exercises):
    cursor=conn.cursor()

    if name and selected_exercises:
        cursor.execute(f"SELECT id from client where name ='{name}'")
        client_id=cursor.fetchone()[0]

        dfs=[]
        for exercise in selected_exercises:
            cursor.execute(f"""
            SELECT s.session_date, we.weight
            FROM sessions s
            JOIN workout_exercises we ON s.id = we.workout_id
            JOIN exercises e ON we.exercise_id = e.id
            WHERE s.client_id = {client_id} AND e.exercise = '{exercise}'
            ORDER BY s.session_date ASC;
            """
            )

            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=[exercise, "weight"])
            dfs.append(df)
        st.text(f'See Your weight progression for each exercise over time')
        for df in dfs:
            st.dataframe(df)

    else:
        st.text('Please enter your name and exercises of choice')

