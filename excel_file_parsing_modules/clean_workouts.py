import numpy as np
import pandas as pd
import re


def clean_workouts(list_of_dataframes):
    cleaned_dataframes = []
    for df in list_of_dataframes:
        if df['Weight'].dtype == 'float64':
            cleaned_dataframes.append(df)
            list_of_dataframes = cleaned_dataframes

            
    for df in list_of_dataframes:
        if df['Prescription'].apply(lambda x: isinstance(x, int)).any():
            df['Prescription'] = df['Prescription'].astype(str)

    prescriptions=[i['Prescription'] for i in list_of_dataframes]
    prescriptions=[i.fillna('1x') for i in prescriptions]
    
    pattern = r'^(\d+)x(\d+)?(?:\s*sec|\s*\(\s*\w*\s*\)|\s*min|\s*@|)?'
    outputs = []
    for arr in prescriptions:
        parts_full = []
        for s in arr:
            try:
                parts = re.findall(pattern, s)
                parts_full.append(parts)
            except:
                pass
        outputs.append(parts_full)
    
    outputs_fixed = [item for item in outputs if item != [[]]]
    for idx,i in enumerate(outputs_fixed):
        if any(not sub for sub in i):
            outputs_fixed[idx] = [[('1', '')] if not sub else sub for sub in i]
        
    sets_reps=[np.array(i).reshape(-1,2) for i in outputs_fixed]
    sets_reps=[np.where(i=='','1',i).astype(dtype='int') for i in sets_reps]
    
    volume=[i[:,0]*i[:,1] for i in sets_reps]
    
    for df,arr in zip(list_of_dataframes,volume):
        try:
            df['Volume-Load']=df['Weight']*arr
        except:
            pass
        
    for df in list_of_dataframes:
        try:
            df['Total'] = pd.Series([0] * len(df))
            df.loc[len(df) - 1, 'Total'] = df['Volume-Load'].sum()
        except:
            pass
        
    return list_of_dataframes
