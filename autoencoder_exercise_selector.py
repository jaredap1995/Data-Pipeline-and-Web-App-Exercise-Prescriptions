import streamlit as st
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from gensim.test.utils import datapath
from gensim import utils
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from coach_functions import create_a_block
from record_prescriptions import record_block


def convert_to_int(arr):
    try:
        return np.array([int(float(elem)) for elem in arr], dtype=int)
    except ValueError:
        return None
    
def get_intensity_range(exercise_options_df, intensity):
    if intensity.lower() == 'light':
        quantile_value = exercise_options_df['VL'].quantile(1/4)
        filtered_options = exercise_options_df[exercise_options_df['VL'] <= quantile_value].sort_values(by='VL')
        return filtered_options['VL']
    elif intensity.lower() == 'moderate':
        quantile_value_lower = exercise_options_df['VL'].quantile(1/4)
        quantile_value_upper = exercise_options_df['VL'].quantile(3/4)
        filtered_options = exercise_options_df[(exercise_options_df['VL'] > quantile_value_lower) & 
                                               (exercise_options_df['VL'] <= quantile_value_upper)].sort_values(by='VL')
        return filtered_options['VL']
    elif intensity.lower() == 'heavy':
        quantile_value = exercise_options_df['VL'].quantile(3/4)
        filtered_options = exercise_options_df[exercise_options_df['VL'] > quantile_value].sort_values(by='VL')
        return filtered_options['VL']
    else:
        raise ValueError("Invalid intensity value")
    

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


def load_prepare_data(conn):
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
    all_workout_data = cursor.fetchall()
    all_workout_data = np.asarray(all_workout_data)
    inputs=all_workout_data[:,1]
    outputs=all_workout_data[:,2:]
    inputs, outputs = filter_data(inputs, outputs)

    df=pd.DataFrame(inputs)
    variables=pd.DataFrame(outputs)
    variables.columns=['Weight', 'Sets', 'Reps']
    df=pd.concat([df, variables], axis=1)

    df['VL'] = df['Weight'].mul(df['Sets']).mul(df['Reps'])
    df.columns=['Exercise', 'Weight', 'Sets', 'Reps', 'VL']

    volume_loads=df['VL']
    exercises=df['Exercise']
    scaled_VL=MinMaxScaler().fit_transform(df['VL'].to_numpy().reshape(-1,1))

    non_101=[]
    for idx, ex in enumerate(exercises):
        if ex.startswith("I'") or ex.startswith("I,"):
            non_101.append(idx)


    tokenizer=tf.keras.preprocessing.text.Tokenizer
    pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
    input_tokenizer = tokenizer(char_level=False, filters='', lower=False)
    input_tokenizer.fit_on_texts(exercises)

    exercises[non_101]='eyes, whys, and tees'

    return df, volume_loads, exercises, scaled_VL, input_tokenizer, pad_sequences

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for document in self.documents:
            # assume there's one document per element in the list, tokens separated by whitespace
            yield utils.simple_preprocess(document)

def corpus_build(exercises):
    tokens=[token for token in MyCorpus(exercises)]
    corpus_instance = MyCorpus(exercises)
    model = gensim.models.Word2Vec(vector_size=100, window=5, min_count=1, workers=4)

    # Build vocabulary from the corpus
    model.build_vocab(corpus_instance)

    # Train the model on the corpus
    model.train(corpus_instance, total_examples=model.corpus_count, epochs=10)

    exercise_vectors = []
    for exercise in tokens:
        exercise_vector = np.mean([model.wv[word] for word in exercise], axis=0)
        exercise_vectors.append(exercise_vector)

    return exercise_vectors

def sanitizie_inputs(exercise_vectors, scaled_VL):
    # Combine exercise vectors with volume loads
    input_data = []
    for exercise_vector, volume_load_normalized in zip(exercise_vectors, scaled_VL):
        combined = np.hstack((exercise_vector, volume_load_normalized))
        input_data.append(combined)
    # input_data = np.array(input_data)

    input_data=np.array(input_data, dtype=np.float32)


    return input_data

def load_model_make_predictions(input_data):
    input_size = input_data[0].shape[0]
    encoding_dim = 32

    # Provided for reference...model already trained and called upon in app.py
    input_layer = tf.keras.layers.Input(shape=(input_size,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = tf.keras.layers.Dense(input_size, activation='sigmoid')(encoded)

    
    # autoencoder = Model(input_layer, decoded)
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    # autoencoder.fit(input_data, input_data, epochs=100, batch_size=32, shuffle=True)

    autoencoder= tf.keras.models.load_model('autoencoder_exercise_selector.h5')
    encoder=tf.keras.models.Model(input_layer, encoded)
    encoded_exercises = encoder.predict(input_data)
    similarity_matrix = cosine_similarity(encoded_exercises)

    return similarity_matrix

def find_similar_exercises(exercise_index, exercises, similarity_matrix, top_n):
    similarity_scores = similarity_matrix[exercise_index]

    # Sort the similarity scores and get the indices
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # Exclude exercises with the same string value as the original exercise
    original_exercise = exercises[exercise_index]
    unique_indices = [idx for idx in sorted_indices if exercises[idx] != original_exercise]

    # Find top_n unique exercises
    top_n_indices = []
    unique_exercises = set()
    for idx in unique_indices:
        if exercises[idx] not in unique_exercises:
            unique_exercises.add(exercises[idx])
            top_n_indices.append(idx)
            if len(top_n_indices) >= top_n:
                break

    return top_n_indices

def exercise_selector(conn, name):

    if 'modifications' not in st.session_state:
        st.session_state.modifications = None

    if 'exercise_selector' not in st.session_state:
        st.session_state.exercise_selector = False

    if 'modified_df' not in st.session_state:
        st.session_state.modified_df = False

    if 'workout_dfs' not in st.session_state:
        st.session_state.workout_dfs = []
    workout_dfs = st.session_state.workout_dfs

    if 'prescribe' not in st.session_state:
        st.session_state.prescribe = False

    if 'view_ai_prescriptions' not in st.session_state:
        st.session_state.view_ai_prescriptions = False

    df, volume_loads, exercises, scaled_VL, input_tokenizer, pad_sequences = load_prepare_data(conn)

    exercise_vectors = corpus_build(exercises)
    input_data = sanitizie_inputs(exercise_vectors, scaled_VL)
    similarity_matrix = load_model_make_predictions(input_data)
    with st.form(key='ai_predictor'):
        original_exercise=st.multiselect('Select exercises', exercises.unique())
        workout_length=st.slider('Select number of exercises per workout', 1, 15)
        num_workouts=st.slider('Select number of workouts', 1, 5)
        intensities=['Light', 'Moderate', 'Heavy']
        intensity=st.selectbox('Select intensity', intensities)
        provide_suggestions=st.form_submit_button('Provide suggestions')
        if provide_suggestions: 
            if num_workouts != len(original_exercise):
                st.error('Number of workouts must equal number of exercises')
                st.stop()
            try:
                st.session_state['workout_dfs']=[]
                workout_dfs=st.session_state.workout_dfs
                for ex in original_exercise:
                    exercise_options=df[df['Exercise']==ex]
                    VL_range=get_intensity_range(exercise_options, intensity)
                    exercise_index=random.choice(VL_range.index)
                    similar_exercise_indices = find_similar_exercises(exercise_index, exercises, similarity_matrix, top_n=workout_length)
                    semantic_vl_exercises_list=exercises[similar_exercise_indices]
                # Load the trained model from a file and tokeinze for regression

                    loaded_regressor = joblib.load('DTR_exercise_variables.joblib')
                    token_exercise=input_tokenizer.texts_to_sequences(semantic_vl_exercises_list)
                    token_exercise=np.asarray(token_exercise)
                    token_exercise=pad_sequences(token_exercise, maxlen=6, padding='pre')

                    # Make predictions
                    predicted_output = loaded_regressor.predict(token_exercise)
                    predicted_output=predicted_output.astype(int)    
                    cursor=conn.cursor()

                    #Quick sanitization
                    old_string = 'eyes, whys, and tees'
                    new_string = 'IYTs'
                    semantic_vl_exercises_list = [new_string if x == old_string else x for x in semantic_vl_exercises_list]
                
                    for idx, exercise in enumerate(semantic_vl_exercises_list):
                        # Convert numpy int64s to Python ints
                        weight = int(predicted_output[idx, 0])
                        sets = int(predicted_output[idx, 1])
                        reps = int(predicted_output[idx, 2])
                        cursor.execute("SELECT EXISTS(SELECT 1 FROM exercises WHERE exercise=%s);",(exercise,))
                        exists = cursor.fetchone()[0]
                        if not exists:
                            cursor.execute("INSERT INTO exercises (exercise) VALUES (%s);", (exercise,))
                        
                        # Insert statement with subquery for exercise id
                        cursor.execute('''
                            INSERT INTO predictions (exercise_id, client_id, weight, sets, reps, original_exercise_for_predictions) 
                            VALUES ((SELECT id FROM exercises WHERE exercise = %s LIMIT 1), 
                            (SELECT id FROM client WHERE name = %s), %s, %s, %s, 
                            (SELECT id FROM exercises WHERE exercise = %s LIMIT 1))
                            ''', (exercise, st.session_state['name'], weight, sets, reps, original_exercise[0]))
                    conn.commit()
                    df_produced=pd.DataFrame({'Exercise': semantic_vl_exercises_list,
                            'Weight': predicted_output[:,0],
                            'Sets': predicted_output[:,1],
                            'Reps': predicted_output[:,2]})
                    workout_dfs.append(df_produced)
                # st.session_state['modifications']=df
                # modifications=st.session_state['modifications']
            except IndexError as e:
                if "list index" in str(e):
                    st.error("Please select an exercise")
                    st.stop()
                else:
                    raise e
    

    if st.session_state['workout_dfs'] or st.session_state.modified_df:
        st.session_state.modified_df = True
        modifications_list = []
        for df in workout_dfs:
            modifications = st.experimental_data_editor(df)
            modifications_list.append(modifications)
        if st.button('Submit Modifications'):
            cursor=conn.cursor()
            for modifications in modifications_list:
                for idx, row in modifications.iterrows():
                    cursor.execute("SELECT EXISTS(SELECT 1 FROM exercises WHERE exercise=%s);",(row['Exercise'],))
                    exists = cursor.fetchone()[0]
                    if not exists:
                        cursor.execute("INSERT INTO exercises (exercise) VALUES (%s);", (row['Exercise'],))
                    cursor.execute('''
                        INSERT INTO training_data (exercise_id, client_id, weight, sets, reps) 
                        VALUES ((SELECT id FROM exercises WHERE exercise = %s LIMIT 1), 
                        (SELECT id FROM client WHERE name = %s), %s, %s, %s)
                        ''', (row['Exercise'], st.session_state['name'], row['Weight'], row['Sets'], row['Reps']))
            conn.commit()
            st.success('Training Data Updated')
        if st.button('Prescribe') or st.session_state.prescribe:
            st.session_state.prescribe = True
            operations=['Volume', 'Intensity', 'Endurance']
            operation=st.selectbox('Select operation', operations)
            num_weeks=st.slider('Select number of weeks', 1, 8)
            prescriptions=create_a_block(modifications_list, operation, num_weeks)
            if st.button('View Block') or st.session_state.view_ai_prescriptions:
                st.session_state.view_ai_prescriptions = True
                for w in prescriptions:
                    st.dataframe(w)
                if st.button('Submit Block'): 
                    record_block(conn, name, prescriptions, num_weeks)
                    st.success('Block Uploaded to Database Successfully. Rerunning to update state.')
                    time.sleep(4)
                    st.experimental_rerun()







