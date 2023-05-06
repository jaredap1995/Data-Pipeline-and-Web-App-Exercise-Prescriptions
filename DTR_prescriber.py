import streamlit as st
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data():
    with open('all_workout_data.json') as f:
        all_workout_data_json = json.load(f)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and fit the model
    regressor = DecisionTreeRegressor()
    regressor.fit(input_seq_padded, outputs)

    return regressor, X_test, y_test


def ai_prescription_support(exercises):
    #set session state variable
    if 'regressor_test_submit' not in st.session_state:
        st.session_state['regressor_test_submit'] = False

    # Run the functions
    workout_data_json = load_data()
    inputs, outputs = create_inputs_outputs(workout_data_json)
    inputs, outputs = filter_data(inputs, outputs)
    unique_inputs, unique_outputs = unique_pairs(inputs, outputs)
    input_tokenizer, pad_sequences, input_seq_padded, outputs = tokenize_and_pad(unique_inputs, unique_outputs)
    regressor, X_test, y_test = split_fit_model(input_seq_padded, outputs)
    workout=st.multiselect("Select Exercises", exercises)
    submit=st.button("Submit", key='regressor_test_submit')
    if submit or st.session_state['regressor_test_submit']:
        st.session_state['regressor_test_submit'] = True
        token_exercise=input_tokenizer.texts_to_sequences(workout)
        token_exercise=np.asarray(token_exercise)
        token_exercise=pad_sequences(token_exercise, maxlen=6, padding='pre')
        predicted_output = regressor.predict(token_exercise)
        df=pd.DataFrame(predicted_output, columns=['Weight', 'Sets', 'Reps'])
        workout=pd.Series(workout, name='Exercise')
        df=pd.concat([workout, df], axis=1)
        st.experimental_data_editor(df)
