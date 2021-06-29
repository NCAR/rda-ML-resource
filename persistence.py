import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date

save_path = '/glade/work/jdubeau/model-saves/'

def save_trained_model(name, 
                       model, model_df,
                       X_features,
                       X_train, y_train_full, 
                       X_val, y_val_full, 
                       X_test, y_test_full,
                       save_date=True):
    '''Parameters: name, model, model_df,
       X_train, y_train_full,
       X_val, y_val_full,
       X_test, y_test_full'''
    
    folder_path = save_path + name
    if save_date:
        day = date.today()
        time = datetime.now().strftime("%H:%M")
        folder_path = folder_path + str(day)+ '-' +str(time) + '/'

    os.makedirs(os.path.dirname(folder_path), exist_ok=True)
    
    pickle.dump(model, open(folder_path+'model.pkl', 'wb'))
    pickle.dump(model_df, open(folder_path+'model_df.pkl', 'wb'))
    pickle.dump(X_features, open(folder_path+'X_features.pkl', 'wb'))
    pickle.dump(X_train, open(folder_path+'X_train.pkl', 'wb'))
    pickle.dump(y_train_full, open(folder_path+'y_train_full.pkl', 'wb'))
    pickle.dump(X_val, open(folder_path+'X_val.pkl', 'wb'))
    pickle.dump(y_val_full, open(folder_path+'y_val_full.pkl', 'wb'))
    pickle.dump(X_test, open(folder_path+'X_test.pkl', 'wb'))
    pickle.dump(y_test_full, open(folder_path+'y_test_full.pkl', 'wb'))