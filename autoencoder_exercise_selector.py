import streamlit as st
import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras.models import Model
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

    int_var=[]
    variables=all_workout_data[:,2:]
    for i, arr in enumerate(variables):
        converted_arr = convert_to_int(arr)
        if converted_arr is not None:
            int_var.append(converted_arr)

    df=pd.DataFrame(all_workout_data)
    df=df.drop(columns=[2,3,4])

    variables=pd.DataFrame(int_var)
    variables.columns=['Weight', 'Sets', 'Reps']
    df=pd.concat([df, variables], axis=1)

    df['VL'] = df['Weight'].mul(df['Sets']).mul(df['Reps'])
    df.columns=['Name', 'Exercise', 'Weight', 'Sets', 'Reps', 'VL']

    volume_loads=df['VL']
    exercises=df['Exercise']
    scaled_VL=MinMaxScaler().fit_transform(df['VL'].to_numpy().reshape(-1,1))

    #indexes of I'Y'T's exercises that mess up embeddings later on. Found in accompanying Jupyter and manually removed
    non_101_shape=[1401,1466,1490,1589,1608,1730,1777,1822,1846,1894,1899,1907,1992,2035,
                    2044,2094,2101,2211]
    
    exercises[non_101_shape]='eyes, whys, and tees'

    return df, volume_loads, exercises, scaled_VL

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
    input_data = np.array(input_data)

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

    # Return the indices of the top n most similar unique exercises
    return top_n_indices

def exercise_selector(conn):
    if 'exercise_selector' not in st.session_state:
        st.session_state.exercise_selector = False

    df, volume_loads, exercises, scaled_VL = load_prepare_data(conn)

    tokenizer=tf.keras.preprocessing.text.Tokenizer
    pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
    input_tokenizer = tokenizer(char_level=False, filters='', lower=False)
    input_tokenizer.fit_on_texts(exercises)


    exercise_vectors = corpus_build(exercises)
    input_data = sanitizie_inputs(exercise_vectors, scaled_VL)
    similarity_matrix = load_model_make_predictions(input_data)
    exercise=st.multiselect('Select exercises', exercises.unique())
    workout_length=st.slider('Select number of exercises', 1, 15)
    intensities=['Light', 'Moderate', 'Heavy']
    intensity=st.selectbox('Select intensity', intensities)
    provide_suggestions=st.button('Provide suggestions')
    if provide_suggestions or st.session_state.exercise_selector:
        st.session_state.exercise_selector = True
        try:
            exercise_options=df[df['Exercise']==exercise[0]]
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
            df=pd.DataFrame({'Exercise': semantic_vl_exercises_list,
                    'Weight': predicted_output[:,0],
                    'Sets': predicted_output[:,1],
                    'Reps': predicted_output[:,2]})
            st.experimental_data_editor(df)
        except IndexError as e:
            if "list index" in str(e):
                st.error("Please select an exercise")
                st.stop()
            else:
                raise e




