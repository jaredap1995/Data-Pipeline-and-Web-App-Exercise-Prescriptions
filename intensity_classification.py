import pandas as pd
import numpy as np


def intensity_classification(files):
    # Get the manual_cuts for the client in question to be able to classify intensity
    totals = [i['Total'].sum() for i in files]
    totals = pd.Series(totals).sort_values()
    bins = pd.qcut(totals, [0,.2,.4,.6,.8,1])
    manual_cuts = [totals[bins == category] for category in bins.cat.categories]
    
    for i, df in enumerate(files):
        VL = df['Total'].sum()
        if VL <= manual_cuts[0].max():
            intensity = 'Light'
        elif VL <= manual_cuts[1].max() and VL >= manual_cuts[0].max():
            intensity = 'Moderate-Light'
        elif VL <= manual_cuts[2].max() and VL >= manual_cuts[1].max():
            intensity = 'Moderate' 
        elif VL <= manual_cuts[3].max() and VL >= manual_cuts[2].max():
            intensity = 'Moderate-Heavy'
        else:
            intensity = 'Heavy'
        # Add an intensity label as an attribute to the dataframe
        setattr(df, 'Intensity', intensity)
    return files
