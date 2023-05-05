import pandas as pd
import openai
import streamlit as st
import json
import tiktoken
import re
import pandas as pd
import random
import psycopg2

def find_matching_workout(user_name, workout_data, goal=None, muscle_focus=None):
    # You can implement any logic to match the user input to the corresponding JSON object
    return workout_data.get(user_name, {})

encoding = tiktoken.get_encoding("p50k_base")

def num_tokens_from_string(string: str, encoding_name="p50k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokenized_text=encoding.encode(string)
    num_tokens = len(encoding.encode(string))
    return num_tokens, tokenized_text


def truncate_prompt(prompt, max_tokens):
    tokens, tokenized_text = num_tokens_from_string(prompt)
    if tokens > max_tokens:
        tokenized_text = tokenized_text[tokens - max_tokens:] 
        prompt=encoding.decode(tokenized_text)
        return prompt
    else:
        return prompt


def format_prompt(workout_data, goal=None, muscle_focus=None):
    all_workouts = []
    for workout in workout_data.values():
        all_workouts.extend(workout)

    workout_str = "\n".join([f"- {item['exercise']} (Weight: {item['weight']} lbs, Sets: {item['sets']}, Reps: {item['reps']})" for item in all_workouts if 'exercise' in item])
    prompt = workout_str
    return prompt


def parse_output(output):
    pattern = r"""(\d+\.?\d*)kg for (\d+) sets of (\d+) reps ([\w\s\/]+)(?= with|$)"""
    matches = re.findall(pattern, output)
    
    exercises = []
    for match in matches:
        weight, sets, reps, exercise = match
        exercises.append({
            'exercise': exercise.strip(),
            'weight': float(weight),
            'sets': int(sets),
            'reps': int(reps)
        })
    
    return pd.DataFrame(exercises)

##Added context to the 'system' prompt instead of integrating context into the 'user' prompt

def generate_text(prompt, model="gpt-3.5-turbo", max_tokens=700, n=1, stop=None, temperature=0.5):
    
    context = "You are the best personal trainer in the world.\
    I will provide you with the historic data in the format: Exercise name (weight, sets, reps),\
    and your job is to create a single workout based on that.\
    Your response should only be the plan in HTML table format with the exercise, \
    weight, sets and reps in each column.\
    The workout should contain exactly 7 exercises."
    
    
    response = openai.ChatCompletion.create(
        messages= [
            {"role": "system", "content":context},
            {"role": "user", "content": prompt}
        ],
        model=model,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temperature
    )

    return response['choices'][0]['message']['content']

def GPT_Coach(name):
    # Load JSON data from file
    with open("all_workout_data.json", "r") as f:
        workout_data = json.load(f)


    openai.organization = st.secrets.open_AI_credentials.org
    openai.api_key = st.secrets.open_AI_credentials.api_key

    try:    
        #name=st.multiselect('Select a name',names)
        # if name is not None:
        #     name=name[0]
        st.write("Generating your workout from previous workouts. This may take a few seconds. Please be patient :)")
        res = generate_text(truncate_prompt(format_prompt(find_matching_workout(name, workout_data)),2500))
        try:
            df=pd.read_html(res)[0]
            st.dataframe(df)
        except:
            st.warning("""Hmmm, something seems to have gone wrong, if the output below does not make sense,
            please unselect and reselect your name. If the problem persists, please contact the developer.""")
            st.write(res)
    except IndexError:
        pass
