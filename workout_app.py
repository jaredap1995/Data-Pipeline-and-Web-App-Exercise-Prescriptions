import numpy as np
import pandas as pd
import psycopg2
import random
import glob
import os
import openpyxl
import re
import streamlit as st
from clean_workouts import clean_workouts
from grab_all_workouts import grab_all_workouts

name=st.text_input("Enter Your First Name: ")
show_workout = st.button("Show Workout")

if show_workout:
    if name: 
        files = grab_all_workouts(name)
        files = clean_workouts(files)
        workout = random.choice(files)
        st.text("Don't like the workout? Just hit the button again!")
        st.dataframe(workout)
        

    else:
        st.warning("Please enter your name")




