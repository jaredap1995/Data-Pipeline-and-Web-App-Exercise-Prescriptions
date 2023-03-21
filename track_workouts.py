import psycopg2
import pandas as pd
import streamlit as st


#Define Helper Sorting Function
def make_sorter(l):
        """
        Create a dict from the list to map to 0..len(l)
        Returns a mapper to map a series to this custom sort order
        """
        sort_order = {k:v for k,v in zip(l, range(len(l)))}
        return lambda s: s.map(lambda x: sort_order[x])



def track_workouts (conn, name, start_date, end_date):
    
    cursor=conn.cursor()

    cursor.execute(f"""SELECT id FROM sessions 
    WHERE client_id = (SELECT id FROM client WHERE name = '{name}')
    AND session_date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY session_date ASC;
    """)


    workout_ids=cursor.fetchall()
    workout_ids=[i[0] for i in workout_ids]


    exercise_orders=[]
    for i in workout_ids:
        cursor.execute(f"""SELECT exercise_id from workout_exercises WHERE workout_id={i};
        """)
        exercise_order=cursor.fetchall()
        exercise_order=[i[0] for i in exercise_order]
        exercise_orders.append(exercise_order)


    # Execute the query
    dfs=[]
    for workout, order in zip(workout_ids, exercise_orders):
        cursor.execute(f"""
        SELECT workout_exercises.exercise_id, workout_exercises.workout_id, exercises.exercise, workout_exercises.sets, workout_exercises.reps, workout_exercises.weight
        FROM workout_exercises
        JOIN exercises ON workout_exercises."exercise_id" = exercises.id
        WHERE workout_exercises.workout_id = {workout};
        """)
        rows = cursor.fetchall()

        # Create a Pandas DataFrame from the query results...Need to modify this so rows in dataframe correspond to the order in which they are prescribed
        df = pd.DataFrame(rows, columns=["exercise_id", "workout_id", "exercise", "sets", "reps", "weight"])
        df=df.sort_values('exercise_id', key=make_sorter(order))
        dfs.append(df)

    for df in dfs:
        st.dataframe(df)

    conn.close()




