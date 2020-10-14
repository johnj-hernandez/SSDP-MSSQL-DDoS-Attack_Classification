def remove_columns(df, columns):
    """ The remove_columns method removes a set of columns from the dataframe
       
       Args: 
           df (dataframe)
           columns (list)
       Returns:
           dataframe: New dataframe without the selected columns
    """
    for column in columns:
        del df[column]
    return df.copy()

def strip_columns(df):
    """The remove_columns method removes leading and trailing whitespaces from all column names
       
       Args: 
           df (dataframe)
       
       Returns:
           dataframe: Dataframe with new column names
    """
    
    headers=[]
    for header in (df.columns):
        headers.append(header.strip())

    df.columns=headers
    
    return df.copy()

import numpy as np
def delete_inf_rows(df, columns):
    
    for column in columns:
        df = df.drop(df[df[column]==np.inf].index)
    
    return df.copy()


from imblearn.under_sampling import OneSidedSelection
def undersampling(X, y,sampling_strategy='auto',n_neighbors = 1):
    sampler = OneSidedSelection(n_jobs=36,sampling_strategy = sampling_strategy, n_neighbors = n_neighbors )
    X_us, y_us = sampler.fit_sample(X, y)
    
    return X_us.copy(), y_us.copy()


from imblearn.over_sampling import SMOTE 
def oversampling(X, y,sampling_strategy='auto',random_state=42):
    
    sm = SMOTE(n_jobs=36,k_neighbors=4,random_state=random_state, sampling_strategy=sampling_strategy)
    X_os, y_os = sm.fit_resample(X, y)
    
    return X_os.copy(), y_os.copy()