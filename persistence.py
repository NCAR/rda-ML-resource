import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date

class model_saver:
    def __init__(self, name=None, notes=None,
                 model=None, model_df=None,
                 y_categories=None,
                 X_features=None,
                 X_train=None, y_train_full=None, 
                 X_val=None, y_val_full=None, 
                 X_test=None, y_test_full=None,
                 save_date=True,
                 mode=None,
                 folder_path=None):
        self.name = name
        self.notes = notes
        self.model = model
        self.model_df = model_df
        self.y_categories = y_categories
        self.X_features = X_features
        self.X_train = X_train
        self.y_train_full = y_train_full
        self.X_val = X_val
        self.y_val_full = y_val_full
        self.X_test = X_test
        self.y_test_full = y_test_full
        self.save_date = save_date
        if mode is None:
            mode='sklearn'
        self.mode = mode
        if folder_path is None:
            folder_path= '/glade/work/jdubeau/model-saves/'
        self.folder_path = folder_path
        
    def save(self):
        save_path = self.folder_path + self.name
        if self.save_date:
            day = date.today()
            time = datetime.now().strftime("%H:%M")
            save_path = save_path + str(day)+ '-' +str(time) + '/'
        else:
            save_path = save_path + '/'

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if self.mode == 'sklearn':
            pickle.dump(self.model, open(save_path+'model.pkl', 'wb'))
        elif self.mode == 'tf':
            self.model.save(save_path)
        else:
            raise ValueError("Mode must be either sklearn or tf.")
        pickle.dump(self.notes,
                open(save_path+'notes.pkl', 'wb'))
        pickle.dump(self.model_df,
                    open(save_path+'model_df.pkl', 'wb'))
        pickle.dump(self.y_categories,
                    open(save_path+'y_categories.pkl', 'wb'))
        pickle.dump(self.X_features,
                    open(save_path+'X_features.pkl', 'wb'))
        pickle.dump(self.X_train,
                    open(save_path+'X_train.pkl', 'wb'))
        pickle.dump(self.y_train_full,
                    open(save_path+'y_train_full.pkl', 'wb'))
        pickle.dump(self.X_val,
                    open(save_path+'X_val.pkl', 'wb'))
        pickle.dump(self.y_val_full,
                    open(save_path+'y_val_full.pkl', 'wb'))
        pickle.dump(self.X_test,
                    open(save_path+'X_test.pkl', 'wb'))
        pickle.dump(self.y_test_full,
                    open(save_path+'y_test_full.pkl', 'wb'))
    
    def load(self, save_path, mode='sklearn'):
        if mode == 'sklearn':
            self.model = pickle.load(open(save_path+'model.pkl','rb'))
        elif mode == 'tf':
            self.model = tf.keras.models.load_model(save_path)
        else:
            raise ValueError("Mode must be either sklearn or tf.")
        self.notes = pickle.load(open(save_path+'notes.pkl', 'rb'))
        self.model_df = pickle.load(open(save_path+'model_df.pkl', 'rb'))
        self.y_categories = pickle.load(open(save_path+'y_categories.pkl', 'rb'))
        self.X_features = pickle.load(open(save_path+'X_features.pkl', 'rb'))
        self.X_train = pickle.load(open(save_path+'X_train.pkl', 'rb'))
        self.y_train_full = pickle.load(open(save_path+'y_train_full.pkl', 'rb'))
        self.X_val = pickle.load(open(save_path+'X_val.pkl', 'rb'))
        self.y_val_full = pickle.load(open(save_path+'y_val_full.pkl', 'rb'))
        self.X_test = pickle.load(open(save_path+'X_test.pkl', 'rb'))
        self.y_test_full = pickle.load(open(save_path+'y_test_full.pkl', 'rb'))
    
    
    def get_min(self):
        '''Returns the minimum amount of data necessary to make predictions
        with the model: the model itself, the list of X features, and the
        X training data (so that input data can be scaled the same way as
        the X training data was).
        '''
        return self.model, self.y_categories, self.X_features, self.X_train
        
    def get_all(self):
        return self.notes, \
               self.model, self.model_df, \
               self.y_categories, \
               self.X_features, self.X_train, \
               self.y_train_full, self.X_val, \
               self.y_val_full, self.X_test, \
               self.y_test_full  
               