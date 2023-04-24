import numpy as np
import pandas as pd
import psycopg2
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import itertools
import math
import streamlit as st
from track_block_progress import ordering_function_for_performed_workouts, check_if_workout_performed


def grid_numbers(rows, cols):
    grid = list(itertools.product(range(1, rows+1), range(1, cols+1)))
    return grid 

def get_subplot_rows_cols(num_rows):
    """
    Returns the number of rows and columns for subplots based on the number of rows in the dataframe.
    Assumes the subplots should be arranged in a square grid with equal number of rows and columns if possible.
    """
    num_plots = num_rows
    sqrt_num_plots = math.sqrt(num_plots)
    num_rows = math.ceil(sqrt_num_plots)
    num_cols = math.ceil(num_plots / num_rows)
    return num_rows, num_cols
    

def grab_workouts_for_visualization(conn, name):

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

    cursor.execute(f"""select workouts_per_week from blocks where id ={block_id}""")
    num_workouts=cursor.fetchone()[0]

    block = pd.DataFrame(prescriptions[:, 2:7], columns=['Workout Number', 'Exercise', 'Sets', 'Reps', 'Weight'])
    block.set_index('Workout Number', inplace=True)
    unique_workout_nums = block.index.unique()
    unique_ex_ids=block['Exercise'].unique()

    exercises_df = pd.read_sql("SELECT * FROM exercises", conn)
    exercises_df.set_index("id", inplace=True)
    exercises_dict = exercises_df["exercise"].to_dict()

    dfs=[block.loc[block.index==i] for i in unique_workout_nums]

    dfs = [df.assign(Exercise=df['Exercise'].map(exercises_dict)) for df in dfs]

    actuals=[]
    prescribed=[]
    for i in range(0,len(dfs)):
        try:
            perf,pres,num=check_if_workout_performed(conn=conn, block_id=block_id, workout_number=i)
            actuals.append(perf)
            prescribed.append(pres)
        except:
            pass
        
    num_weeks = len(dfs) // num_workouts
        
    actuals=ordering_function_for_performed_workouts(num_weeks=num_weeks, num_workouts=num_workouts, dfs=dfs, actuals=actuals)


    return actuals, dfs, num_weeks, num_workouts


def weight_charts_per_workout(prescribed_df, actual_df, columns, num_weeks, num_workouts_per_week):
    bar_charts = []
    for col1, col2 in zip(columns, columns):
        bar_charts.append(go.Bar(
            name=f'{col1} prescribed',
            x=prescribed_df.index,
            y=prescribed_df[col1],
            text=prescribed_df[col1], 
        textposition='inside',  # Set text position to inside the bars
        textangle=0,  # Set text angle to 0 degrees
        textfont=dict(size=12))
    )
        
        bar_charts.append(go.Bar(
            name=f'{col2} actual',
            x=actual_df.index,
            y=actual_df[col2],
            text=actual_df[col2], 
        textposition='inside',  # Set text position to inside the bars
        textangle=0,  # Set text angle to 0 degrees
        textfont=dict(size=12))
    )

    num_rows, num_cols=get_subplot_rows_cols(len(actual_df))   
    
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f'Week {i}' for i in range(1,num_weeks+1)])

    grid=grid_numbers(num_rows, num_cols)
    
    iterations=range(1, num_weeks*num_workouts_per_week+1, 2)
    zipped=zip(grid,iterations)
    
    
    for i,x in zipped: # Update the range to start from 1
        data = bar_charts[x-1:x+num_workouts_per_week-1]  # Update the indexing to start from 1
        for bar_chart in data:
            fig.add_trace(bar_chart, 
                          row=i[0], 
                          col=i[1])


    fig.update_layout(
        title=dict(
            text='First Workout of Week Weight Comparison',
            x=0,  
            y=0.95,
            font=dict(size=35)   
        ),
        barmode='group',
        xaxis_ticktext=columns,
        showlegend=True,
        legend=dict(title='Prescribed and Actual Weight',)
    )

    return fig


def weight_charts_per_exercise(df, df_actual, num_weeks, num_workouts_per_week):
    # create a list of bar charts for each series in both dataframes
    bar_charts = []
    for exercise_1, exercise_2 in zip(df.index, df_actual.index):
        bar_charts.append(go.Bar(
            name=f'{exercise_1} prescribed',
            x=df.columns,
            y=df.loc[exercise_1],
            text=df.loc[exercise_1], 
            textposition='inside',  # Set text position to inside the bars
            textangle=0,  # Set text angle to 0 degrees
        textfont=dict(size=12),
        legendgroup='Prescribed')
        )

        bar_charts.append(go.Bar(
            name=f'{exercise_2} performed',
            x=df_actual.columns,
            y=df_actual.loc[exercise_2],
            text=df_actual.loc[exercise_2], 
            textposition='inside',  # Set text position to inside the bars
            textangle=0,  # Set text angle to 0 degrees
            textfont=dict(size=12),
        legendgroup='Actual')
        )

    num_rows, num_cols=get_subplot_rows_cols(len(df_actual))   
    
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f'{i}' for i in df.index])

    grid=grid_numbers(num_rows, num_cols)
    
    iterations=range(1, num_weeks*num_workouts_per_week+1, 2)
    zipped=zip(grid,iterations)
    
    for i,x in zipped: # Update the range to start from 1
        data = bar_charts[x-1:x+num_workouts_per_week-1]  # Update the indexing to start from 1
        for bar_chart in data:
            fig.add_trace(bar_chart, row=i[0], col=i[1])
    
    
    fig.update_layout(
    title=dict(
        text='Weight Comparison Across Block Per Exercise',
        x=0,  # Center title horizontally
        y=0.95,   # Adjust vertical position of title
        font=dict(size=35)),
    barmode='group',
    xaxis_ticktext=list(df.columns),
    showlegend=True,
    legend=dict(title='Legend Title', groupclick='togglegroup'),)

    # Update legend titles
    for i in range(num_weeks):
        fig.update_traces(
            selector=dict(name=f'{df.index[i]} prescribed'),  # Use legendgroup instead of name
            name=f'Weight Prescribed')  # Convert index value to string

        fig.update_traces(
            showlegend=True,
            selector=dict(name=f'{df_actual.index[i]} performed' ),
            name=f'Weight Performed')  # Convert index value to string

    return fig



def pull_visuals (conn, name):

    actuals, dfs, num_weeks, num_workouts=grab_workouts_for_visualization(conn=conn, name=name)

    workkouts_per_week=num_workouts/num_weeks


    weight_p=[i[['Exercise', 'Weight']].reset_index(drop=True) for i in dfs]
    weight_a=[i[['Exercise','Weight']] for i in actuals]

    index_1 = ['Exercise', 'Weight'] * 9
    index_2 = [f"Workout Number {i}" for i in range(len(actuals))]
    weight_actuals_df = pd.concat(weight_a, axis=1, ignore_index=True)
    st.dataframe(weight_actuals_df)
    

    weight_actuals_df.columns=index_1
    st.dataframe(weight_actuals_df)
    st.stop()
    
    
    weight_prescribed_df=pd.concat(weight_p, axis=1, ignore_index=True)
    col_names_p = [f"Workout Number {i}" for i in range(len(dfs))]
    weight_prescribed_df.columns=col_names_p

    st.dataframe(weight_prescribed_df)
    st.dataframe(weight_actuals_df)
    st.stop()

    #Actual weights
    first_workout_of_week_actual_weight=weight_actuals_df.iloc[:,::2].dropna(how='all')
    """Will come back and change the number one below, needs to be iteration for however many workouts per week-1::workouts_per_week"""

    first_workout_of_week_indexes=actuals[::2]
    df_with_longest_index_1 = max(first_workout_of_week_indexes, key=lambda df: len(df.index))
    first_workout_of_week_actual_weight.index=df_with_longest_index_1['Exercise']

    second_workout_of_week_actual_weight=weight_actuals_df.iloc[:,1::2].dropna(how='all')
    second_workout_of_week_indexes=actuals[1::2]
    df_with_longest_index_2 = max(second_workout_of_week_indexes, key=lambda df: len(df.index))
    second_workout_of_week_actual_weight.index=df_with_longest_index_2['Exercise']

    st.dataframe(first_workout_of_week_actual_weight)
    st.dataframe(second_workout_of_week_actual_weight)
    st.stop()

    #Prescribed weights
    first_workout_of_week_prescribed_weight=weight_prescribed_df.iloc[:,::2]
    second_workout_of_week_prescribed_weight=weight_prescribed_df.iloc[:,1::2]
    first_workout_of_week_prescribed_weight.index=dfs[0]['Exercise']
    second_workout_of_week_prescribed_weight.index=dfs[1]['Exercise']

    # st.dataframe(first_workout_of_week_prescribed_weight)
    # st.dataframe(second_workout_of_week_prescribed_weight)
    # st.dataframe(first_workout_of_week_actual_weight)
    # st.dataframe(second_workout_of_week_actual_weight)


    analysis_dfs=[first_workout_of_week_actual_weight, second_workout_of_week_actual_weight, first_workout_of_week_prescribed_weight, second_workout_of_week_prescribed_weight]

    for i in analysis_dfs:
        i.columns=['Week 1', 'Week 2', 'Week 3', 'Week 4']

    view_block=st.button('View Block')
    if view_block:
        fig_workouts=weight_charts_per_workout(first_workout_of_week_prescribed_weight, first_workout_of_week_actual_weight, 
                               first_workout_of_week_actual_weight.columns, num_weeks, num_workouts)
        fig_exercises=weight_charts_per_exercise(first_workout_of_week_prescribed_weight, 
                                                 first_workout_of_week_actual_weight, num_weeks, num_workouts)
        st.plotly_chart(fig_workouts, use_container_width=True)
        st.plotly_chart(fig_exercises, use_container_width=True)

