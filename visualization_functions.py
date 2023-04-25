import numpy as np
import pandas as pd
import psycopg2
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import itertools
import math
import re
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
    workouts_per_week=cursor.fetchone()[0]

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
        
    num_weeks = len(dfs) // workouts_per_week
        
    actuals=ordering_function_for_performed_workouts(num_weeks=num_weeks, num_workouts=workouts_per_week, dfs=dfs, actuals=actuals)


    return actuals, dfs, num_weeks, workouts_per_week


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


def weight_charts_per_exercise(df, df_actual):

    actual_exercises=df_actual.index.values
    prescribed_exercises=df.index.values

    combined_array = np.concatenate((prescribed_exercises, actual_exercises), axis=0)
    combined_array=np.unique(combined_array)
    combined_array

    bar_charts_paired = []
    bar_charts_unpaired=[]
    unpaired=[]
    paired=[]
    for exercise_1 in combined_array:
        if exercise_1 in df.index.values and exercise_1 in df_actual.index.values:
            paired.append(exercise_1)
            bar_charts_paired.append(go.Bar(
                name=f'{exercise_1} prescribed',
                x=df.columns,
                y=df.loc[exercise_1],
                text=df.loc[exercise_1], 
                textposition='inside',  # Set text position to inside the bars
                textangle=0,  # Set text angle to 0 degrees
                textfont=dict(size=12),
                showlegend=(exercise_1 == paired[0]),
                legendgroup='Prescribed')
            )

            bar_charts_paired.append(go.Bar(
                name=f'{exercise_1} performed',
                x=df_actual.columns,
                y=df_actual.loc[exercise_1],
                text=df_actual.loc[exercise_1], 
                textposition='inside',  # Set text position to inside the bars
                textangle=0,  # Set text angle to 0 degrees
                textfont=dict(size=12),
                showlegend=False,
                legendgroup='Actual')
            )
        else:
            unpaired.append(exercise_1)

    for ex in unpaired:
        if ex not in df.index.values:
                bar_charts_unpaired.append(go.Bar(
                name=f'{ex} performed',
                x=df_actual.columns,
                y=df_actual.loc[ex],
                text=df_actual.loc[ex], 
                textposition='inside',  # Set text position to inside the bars
                textangle=0,  # Set text angle to 0 degrees
                textfont=dict(size=12),
                showlegend=(ex == unpaired[0]),
                legendgroup='Actual')
            )
        else:
            bar_charts_unpaired.append(go.Bar(
                name=f'{ex} prescribed',
                x=df.columns,
                y=df.loc[ex],
                text=df.loc[ex], 
                textposition='inside',  # Set text position to inside the bars
                textangle=0,  # Set text angle to 0 degrees
                textfont=dict(size=12),
                showlegend=(ex == unpaired[0]),
                legendgroup='Prescribed')
            )

    #Creating header for subplots with regex because passing index of the dataframe caused ordering issues     
    names=[i['name'] for i in bar_charts_paired]
    names=names[::2]
    names_2=[i['name'] for i in bar_charts_unpaired]
    names.extend(names_2)
    regex_pattern = r' \b(performed|prescribed)\b.*$'
    output_list = [re.sub(regex_pattern, '', string) for string in names]

    num_rows, num_cols=get_subplot_rows_cols(len(df_actual))   

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=output_list)

    grid=grid_numbers(num_rows, num_cols)


    step=2
    for i in range(0, len(bar_charts_paired), step):
        bar_chart1 = bar_charts_paired[i]
        bar_chart2 = bar_charts_paired[i+1] if i+1 < len(bar_charts_paired) else None
        x = grid[i//step]  # Calculate the corresponding grid cell for the paired bar charts
        fig.add_trace(bar_chart1, row=x[0], col=x[1])
        if bar_chart2:
            fig.add_trace(bar_chart2, row=x[0], col=x[1])

    for i in range(len(bar_charts_unpaired)):
        bar_chart1 = bar_charts_unpaired[i]
        idx = grid.index(x)+1  # Calculate the corresponding grid cell for the paired bar charts
        fig.add_trace(bar_chart1, row=grid[idx][0], col=grid[idx][1])
        x=grid[idx]


    fig.update_layout(
    title=dict(
        text='Weight Comparison Across Block Per Exercise',
        x=0.5,  # Center title horizontally
        y=0.95   # Adjust vertical position of title
    ),
    barmode='group',
    xaxis_ticktext=list(df.columns),
    showlegend=True,
    legend=dict(title='Groups', groupclick='togglegroup',
               tracegroupgap=15, font=dict(size=15)),
               clickmode='event+select'
    )

    
    fig.update_traces(showlegend=True,
        selector=dict(name=f'{paired[0]} prescribed'),
        name=f'Weight Prescribed',)

    fig.update_traces(
        showlegend=True,
        selector=dict(name=f'{unpaired[0]} performed'), #Not a perfect strategy but should work for most instances
        name=f'Weight Performed',)


    return fig

#Helper Function
def link_workout_number_to_weeks(num_workouts_per_week, num_weeks):
    total_workouts = num_weeks * num_workouts_per_week
    numbers = list(range(0, total_workouts))
    output = [(tuple(numbers[i:i+num_workouts_per_week]), i // num_workouts_per_week) for i in range(0, total_workouts, num_workouts_per_week)]

    return output


def pull_visuals (conn, name):

    actuals, dfs, num_weeks, workkouts_per_week=grab_workouts_for_visualization(conn=conn, name=name)

    num_workouts = len(dfs)
    output=link_workout_number_to_weeks(num_workouts_per_week=workkouts_per_week, num_weeks=num_weeks)

    dfs = [i.reset_index() for i in dfs]
    weight_p=[i[['Workout Number','Exercise', 'Weight']] for i in dfs]
    weight_a = [i[['Workout Number','Exercise','Weight']] for i in actuals]

    #####Organize Actuals into Dataframes for wieght####
    concat_df=pd.concat(weight_a, axis=0)

    workout_specific_dfs=[]
    #Seperate Workouts based on workouts per week
    for i in range(workkouts_per_week):
        workout_numbers = [i + (j * workkouts_per_week) for j in range(num_weeks)]
        filtered_df = concat_df[concat_df['Workout Number'].isin(workout_numbers)]
        workout_specific_dfs.append(filtered_df)

    grouped_actuals=[]
    for df in workout_specific_dfs:
        weight=df.groupby(['Workout Number', 'Exercise']).sum()
        weight=weight.unstack(level=0)
        weight=weight.droplevel(0, axis=1)
        grouped_actuals.append(weight)
        
    result = []
    values_to_match = [item for tpl in output for item in tpl[0]]
    for tpl, df in zip(output, grouped_actuals):
        weeks=[]
        for col in df.columns:
            if int(col) in values_to_match:
                result_value = [x[1] for x in output if int(col) in x[0]][0]
                weeks.append(result_value)
        result.append(weeks)

        
    for df,weeks in zip(grouped_actuals,result):
        df_columns=[]
        for week in weeks:
            column=f'Week {week+1}'
            df_columns.append(column)
        df.columns=df_columns
    
    ####Organize Prescribed into Dataframes for wieght###
    concat_df=pd.concat(weight_p, axis=0)

    workout_specific_dfs=[]
    for i in range(workkouts_per_week):
        workout_numbers = [i + (j * workkouts_per_week) for j in range(num_weeks)]
        filtered_df = concat_df[concat_df['Workout Number'].isin(workout_numbers)]
        workout_specific_dfs.append(filtered_df)
    
    
    grouped_prescribed=[]
    for df in workout_specific_dfs:
        weight=df.groupby(['Workout Number', 'Exercise']).sum()
        weight=weight.unstack(level=0)
        weight=weight.droplevel(0, axis=1)
        grouped_prescribed.append(weight)
        
    result = []
    values_to_match = [item for tpl in output for item in tpl[0]]
    for tpl, df in zip(output, grouped_prescribed):
        weeks=[]
        for col in df.columns:
            if int(col) in values_to_match:
                result_value = [x[1] for x in output if int(col) in x[0]][0]
                weeks.append(result_value)
        result.append(weeks)

        
    for df,weeks in zip(grouped_prescribed,result):
        df_columns=[]
        for week in weeks:
            column=f'Week {week+1}'
            df_columns.append(column)
        df.columns=df_columns


    fig_exercises=weight_charts_per_exercise(grouped_prescribed[0], grouped_actuals[0])
    st.plotly_chart(fig_exercises)



    st.stop()
    merged_df = second_workout_actual_list[0]
    suffixes = [f'_{i}' for i in range(len(second_workout_actual_list) - 1)]
    for i, df in enumerate(second_workout_actual_list[1:]):
        merged_df = pd.merge(merged_df, df, on='Exercise', how='outer', suffixes=('', suffixes[i]))

    merged_df.index = merged_df['Exercise']
    merged_df = merged_df.drop(columns='Exercise')

    # Add NaN column for missing weeks
    if len(merged_df.columns) < num_weeks:
        missing_weeks = num_weeks - len(merged_df.columns)
        for i in range(missing_weeks):
            merged_df[f'Week {len(merged_df.columns) + 1}'] = pd.Series([np.nan] * len(merged_df), name=f'Week {len(merged_df.columns) + 1}')

    merged_df.columns = [f'Week {i}' for i in range(1, len(merged_df.columns) + 1)]
    st.dataframe(merged_df)

    

    #Weight Actuals Indexing
    # index_1 = ['Exercise', 'Weight'] * len(actuals)
    # index_2 = [f"Workout Number {i}" for i in range(len(actuals))]
    # weight_actuals_df = pd.concat(weight_a, axis=1, ignore_index=True)
    # weight_actuals_df.columns=pd.Index([f'actual_{index_1[i]}_{i}' for i in range(len(index_1))])
    
    # #Weight Prescribed Indexing
    # index_1 = ['Exercise', 'Weight'] * len(dfs)
    # index_2 = [f"Workout Number {i}" for i in range(len(dfs))]
    # weight_prescribed_df=pd.concat(weight_p, axis=1, ignore_index=True)
    # weight_prescribed_df.columns=pd.Index([f'prescribed_{index_1[i]}_{i}' for i in range(len(index_1))])

    # st.dataframe(weight_prescribed_df)
    # st.dataframe(weight_actuals_df)

    st.stop()
    #Actual weights

    first_workout_of_week_actual_weight=weight_actuals_df.iloc[:,1::4].dropna(how='all')
    """Will come back and change the number one below, needs to be iteration for however many workouts per week-1::workouts_per_week"""

    first_workout_of_week_indexes=actuals[::2]
    df_with_longest_index_1 = max(first_workout_of_week_indexes, key=lambda df: len(df.index))
    first_workout_of_week_actual_weight.index=df_with_longest_index_1['Exercise']

    st.dataframe(first_workout_of_week_actual_weight)
    st.stop()

    second_workout_of_week_actual_weight=weight_actuals_df.iloc[:,2::4].dropna(how='all')
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

