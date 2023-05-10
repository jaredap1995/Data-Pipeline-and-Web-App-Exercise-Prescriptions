import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib


def load_data(conn):
    cursor=conn.cursor()
    cursor.execute("""
        SELECT 
            c.name AS client_name, 
            e.exercise AS exercise_name, 
            td.weight, 
            td.sets, 
            td.reps
        FROM training_data td
        LEFT JOIN client c ON td.client_id = c.id
        LEFT JOIN exercises e ON td.exercise_id = e.id;
        """)
    all_workout_data_json = cursor.fetchall()
    # with open('all_workout_data.json') as f:
    #     all_workout_data_json = json.load(f)
    return all_workout_data_json

def passes_filters(exercise):

    ## Preprocessing
    char_to_remove = [str(i) for i in range(10)]
    specials_to_remove = ['*', '/', '**', '***', '(', '[', ')', ']', '&']
    yelenas_to_remove = ['did', 'doing', 'didnt', 'need', 'could', 'couldnt', '?']
    emphasis_to_remove = ['emphasis', 'Emphasis', 'Focus', 'Condition', 'recon', 'recondition', 'Recondition', 'wedding']

    if any(char in exercise for char in char_to_remove):
        return False
    if any(char in exercise for char in specials_to_remove):
        return False
    if any(char in exercise for char in yelenas_to_remove):
        return False
    if any(char in exercise for char in emphasis_to_remove):
        return False
    return True


def create_inputs_outputs(workout_data_json):
    inputs=[]
    outputs=[]
    workouts=[]
    for client in workout_data_json:
        for workout in workout_data_json[client]:
            workouts.append(workout)
            for i in range(len(workout)):
                try:
                    name = workout_data_json[client][workout][i]["exercise"]
                    weight = workout_data_json[client][workout][i]["weight"]
                    sets = workout_data_json[client][workout][i]["sets"]
                    reps = workout_data_json[client][workout][i]["reps"]
                    inputs.append(name)
                    outputs.append([weight, sets, reps])
                except:
                    pass
                
    inputs=np.asarray(inputs)
    outputs=np.asarray(outputs)

    inputs=np.delete(inputs,8884)
    outputs=np.delete(outputs,8884, axis=0)

    inputs=np.delete(inputs,15176)
    outputs=np.delete(outputs,15176, axis=0)

    return inputs, outputs


def convert_to_int(arr):
    try:
        return np.array([int(float(elem)) for elem in arr], dtype=int)
    except ValueError:
        return None
    

def unique_pairs(inputs, outputs):
    unique_pairs = {}

    for input_value, output_value in zip(inputs, outputs):
        key = (input_value, tuple(output_value)) # tuple because lists are not hashable

        if key not in unique_pairs:
            unique_pairs[key] = output_value

    unique_inputs = [key[0] for key in unique_pairs]
    unique_outputs = [unique_pairs[key] for key in unique_pairs]
    return unique_inputs, unique_outputs

def filter_data(inputs, outputs):
    #Filtering out exercises that don't pass the string filters
    new_inputs = []
    new_outputs = []
    for i, exercise in enumerate(inputs):
        if passes_filters(exercise):
            new_inputs.append(exercise)
            new_outputs.append(outputs[i])

    inputs = new_inputs
    outputs = new_outputs

    # Making sure all values are integers
    converted_outputs = []
    new_inputs = []

    for i, arr in enumerate(outputs):
        converted_arr = convert_to_int(arr)
        if converted_arr is not None:
            converted_outputs.append(converted_arr)
            new_inputs.append(inputs[i])

    inputs = new_inputs
    outputs = converted_outputs

    return inputs, outputs


def tokenize_and_pad(unique_inputs, unique_outputs):
    # Tokenize inputs
    tokenizer=tf.keras.preprocessing.text.Tokenizer
    pad_sequences=tf.keras.preprocessing.sequence.pad_sequences

    input_tokenizer = tokenizer(char_level=False, filters='', lower=False)
    input_tokenizer.fit_on_texts(unique_inputs)
    input_sequences = input_tokenizer.texts_to_sequences(unique_inputs)

    # Pad sequences
    input_seq_padded = pad_sequences(input_sequences, padding='pre')

    #Changing outputs to 2D Array
    outputs=np.asarray(unique_outputs)
    outputs

    return input_tokenizer, pad_sequences, input_seq_padded, outputs

def split_fit_model(input_seq_padded, outputs):
    # Splitting data
    X=input_seq_padded
    y=outputs
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Create and fit the model
    regressor = DecisionTreeRegressor()
    regressor.fit(input_seq_padded, outputs)

    return regressor, X_test, y_test


def ai_prescription_support(exercises, conn):
    #set session state variable
    if 'regressor_ai' not in st.session_state:
        st.session_state['regressor_ai'] = False

    # Run the functions
    workout_data = load_data(conn)
    workout_data=np.asarray(workout_data)
    # inputs, outputs = create_inputs_outputs(workout_data_json)
    inputs, outputs = filter_data(workout_data[:,1], workout_data[:,2:])
    # unique_inputs, unique_outputs = unique_pairs(inputs, outputs)
    input_tokenizer, pad_sequences, input_seq_padded, outputs = tokenize_and_pad(inputs, outputs)
    regressor, X_test, y_test = split_fit_model(input_seq_padded, outputs)
    workout=st.multiselect("Select Exercises", exercises)
    submit=st.button("Submit", key='regressor_test_submit')
    if submit or st.session_state['regressor_ai']:
        st.session_state['regressor_ai'] = True
        token_exercise=input_tokenizer.texts_to_sequences(workout)
        token_exercise=np.asarray(token_exercise)
        token_exercise=pad_sequences(token_exercise, maxlen=6, padding='pre')

        test=['SA Squat Thrusters']
        test=input_tokenizer.texts_to_sequences(test)
        test=np.asarray(test)
        test=pad_sequences(test, maxlen=6, padding='pre')
        st.write(test)
        # try:
        predicted_output = regressor.predict(token_exercise)
        test_output = regressor.predict(test)
        loaded_regressor = joblib.load('DTR_exercise_variables.joblib')
        test_loaded_output = loaded_regressor.predict(test)
        st.write(test_output)
        st.write(test_loaded_output)
        
        #     loaded_output = loaded_regressor.predict(token_exercise)
        # except ValueError as e:
        #     if "minimum of 1 is required" in str(e):
        #         st.error("No Exercises Selected")
        #         st.stop()
        #     else:
        #         raise e
        # predicted_output=predicted_output.astype(int)
        # loaded_output=loaded_output.astype(int)
        # df=pd.DataFrame(predicted_output, columns=['Weight', 'Sets', 'Reps'])
        # df2=pd.DataFrame(loaded_output, columns=['Weight', 'Sets', 'Reps'])
        # workout=pd.Series(workout, name='Exercise')
        # df=pd.concat([workout, df], axis=1)
        # df2=pd.concat([workout, df2], axis=1)
        # st.experimental_data_editor(df)
        # st.experimental_data_editor(df2, key='loaded_regressor')
