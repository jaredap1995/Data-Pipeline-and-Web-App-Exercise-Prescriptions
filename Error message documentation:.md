Error message documentation:

# ZeroDivisionError: Occurs when some users (Yelena, Shawn, Nat) go to workout 3 (but not all workout 3's are affected, Misha/Nick is fine) under the visualize training functionality under "View A workout In most Recent Block"
        Traceback (most recent call last):

        File "/home/appuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 565, in _run_script

            exec(code, module.__dict__)

        File "/app/data-pipeline-and-web-app-exercise-prescriptions/workout_app.py", line 288, in <module>

            app()

        File "/app/data-pipeline-and-web-app-exercise-prescriptions/workout_app.py", line 220, in app

            pull_visuals(conn, name)

        File "/app/data-pipeline-and-web-app-exercise-prescriptions/visualization_functions.py", line 378, in pull_visuals

            fig, not_used = weight_chart_per_block(grouped_prescribed[workout_index[0]], grouped_actuals[workout_index[0]])

        File "/app/data-pipeline-and-web-app-exercise-prescriptions/visualization_functions.py", line 157, in weight_chart_per_block

            num_rows, num_cols=get_subplot_rows_cols(len(df_actual))

        File "/app/data-pipeline-and-web-app-exercise-prescriptions/visualization_functions.py", line 26, in get_subplot_rows_cols

            num_cols = math.ceil(num_plots / num_rows)

        ZeroDivisionError: division by zero

# IndexError: List Index out of range in Visualization Functions under 'View a Workout in your most recent block. Presumably occurs when the workout has not been performed yet?

Traceback (most recent call last):

  File "/home/appuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 565, in _run_script

    exec(code, module.__dict__)

  File "/app/data-pipeline-and-web-app-exercise-prescriptions/workout_app.py", line 288, in <module>

    app()

  File "/app/data-pipeline-and-web-app-exercise-prescriptions/workout_app.py", line 220, in app

    pull_visuals(conn, name)

  File "/app/data-pipeline-and-web-app-exercise-prescriptions/visualization_functions.py", line 378, in pull_visuals

    fig, not_used = weight_chart_per_block(grouped_prescribed[workout_index[0]], grouped_actuals[workout_index[0]])

  File "/app/data-pipeline-and-web-app-exercise-prescriptions/visualization_functions.py", line 176, in weight_chart_per_block

    fig.add_trace(bar_chart1, row=grid[idx][0], col=grid[idx][1])

IndexError: list index out of range