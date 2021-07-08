# Full machine learning process, start to finish:
# Takes in a list of requests, and uses the current ML model to predict 
# the time and memory they will use.
import cProfile
import pstats
from pstats import SortKey
import pandas as pd
import numpy as np
from preprocessing import add_new_features
from preprocessing import handle_missing
from preprocessing import scale_other
from persistence import model_saver
import math
from math import exp
import mysql.connector
import json
import sys

pd.options.display.max_columns = None
pd.options.display.max_rows = 75

def read_args():
    rindex = sys.argv[1]
    request_type = sys.argv[2]
    
    return rindex, request_type

def get_rinfo(rindex):
    credentials = json.load(open('dsrqst-creds.json'))
    host_name = 'rda-db.ucar.edu'
    db_name = 'dssdb'
    
    conn = mysql.connector.connect(user=credentials['user'],
                                  password=credentials['password'],
                                  host=host_name,
                                  database=db_name)
    curs = conn.cursor()
    
    curs.execute("SELECT rinfo FROM dsrqst WHERE rindex = %s",
                 (rindex,))
    rinfo = curs.fetchone()[0]
    
    conn.close()
    return rinfo

def handle_missing_rinfo_val(feature, val):
    """Decides the appropriate 'null value' for when a feature is
    not present in an rinfo string.
    """

    if feature == 'gui':
        return False
    elif feature in ['startdate', 'enddate']:
        return pd.NaT
    else:
        return float('nan')
    

def format_rinfo_val(rinfo, feature, val):
    """Formats a value found in an rinfo string so that it can be entered
    correctly into the pandas dataframe.
    
    Example input: 
    rinfo = '...;elon=76.4;...'
    feature = 'elon'
    val = '76.4'

    Example output: 76.4
    """

    if val == '':
        return handle_missing_rinfo_val(feature, val)

    try:
        if feature in ['elon', 'wlon', 'nlat', 'slat']:
            return float(val)
        elif feature in ['startdate', 'enddate']:
            return pd.to_datetime(val, errors='coerce')
        elif feature in ['gindex', 'tindex']:
            return float(val)
        elif feature == 'gui':
            return True if val == 'yes' else False
        else:
            return val
    except:
        failed_parse.write(f"Could not parse {feature} from {val}. \n")
        failed_parse.write(f"rinfo string: {rinfo} \n")
        return val

    
def get_val_from_rinfo(rinfo, feature, alternate_names):
    """Finds the value of the given feature in the given rinfo string.
    If the feature is not present in rinfo, calls handle_missing_rinfo_val.
    If the feature is present, uses format_rinfo_val to convert the value 
    to the appropriate data type before returning.
    
    Example input: 
    rinfo = '...;elon=76.4;...' 
    feature = 'elon'

    Example output: 76.4
    """
    
    rinfo = rinfo.replace('%3D', '=')
    
    if ';' in rinfo:
        sep = ';'
    else:
        sep = '&'

    val = ''
    for alternate_name in alternate_names[feature]:
        if rinfo.lower().find(alternate_name) != -1:
            start_ind = rinfo.lower().find(alternate_name) + len(alternate_name) + 1
            end_ind = rinfo.find(sep, start_ind)
            if end_ind != -1:
                val = rinfo[start_ind:end_ind]
            else:
                val = rinfo[start_ind:]
                
    val = format_rinfo_val(rinfo, feature, val)
    return val


def valid_rinfo(rinfo):
    """Decides whether an rinfo string is 'valid.'
    In practice, just serves to filter out a few problematic rinfo strings
    (14 out of the original 59803, or 0.023%).
    """
    if '\\n' in rinfo:
        return False
    elif '76,78,81,83,85,88,90,92,94,96grid_definition' in rinfo:
        return False
    elif 'startDate' in rinfo:
        return False
    else:
        return True
    

def parse_lats_lons(val):
    """Takes a 'lats' or 'lons' value and returns two floats representing the 
    southern/western coordinate and the northern/eastern coordinate.
    Example input: '60 S 80 N'
    Example output: (-60.0, 80.0)
    """
    val = val.replace(',', '')
    substrings = val.split(' ')

    try:
        first_coord = float(substrings[0])
        second_coord = float(substrings[2])
    except:
        print(f"Error expanding lats/lons. Value = {val}")
        return (float('nan'), float('nan'))

    if substrings[1] == 'W' or substrings[1] == 'S':
        first_coord = -1*first_coord
    if substrings[3] == 'W' or substrings[3] == 'S':
        second_coord = -1*second_coord

    return (first_coord, second_coord)


def update_lat_lon(feature, row):
    """Used to update 'slat', 'nlat', 'wlon', or 'elon' by 
    getting the values from 'lats' or 'lons' in the same row.
    Example input: ('nlat', <row containing 'lats=45 S 50 N'>)
    Example output: 50.0
    """
    if row['lats'] != row['lats']:
        return row[feature]
    else:
        if feature == 'slat':
            return parse_lats_lons(row['lats'])[0]
        elif feature == 'nlat':
            return parse_lats_lons(row['lats'])[1]
        elif feature == 'wlon':
            return parse_lats_lons(row['lons'])[0]
        else:
            return parse_lats_lons(row['lons'])[1]
        

def parse_dates(feature, dates):
    """Deduces a start date or end date from whatever was in
    the 'dates' column. The entered feature must be either 
    'startdate' or 'enddate'.
    Example input: ('enddate', '2019-01-01 00:00 2019-12-31 18:00')
    Example output: 2019-12-31 18:00 (pandas datetime object)
    """
    dates_split = dates.split(' ')
        
    if len(dates_split) == 4:
        # Typical case: dates=2019-01-01 00:00 2019-12-31 18:00
        if feature == 'startdate':
            date = dates_split[0] + ' ' + dates_split[1]
        else:
            date = dates_split[2] + ' ' + dates_split[3]
    else:
        # Typical cases: either dates=1806-01-01 1900-12-31
        # or dates=197005 201412
        if feature == 'startdate':
            date = dates_split[0]
        else:
            date = dates_split[1]
            
        if '-' not in dates_split[0]:
            date = date[:4] + '-' + date[4:]
                
    return pd.to_datetime(date, errors='coerce')


def update_dates(feature, row):
    """Used to update 'startdate', 'enddate', or 'dates_init' by
    getting the values from the 'dates' column in the same row.
    feature must be either 'startdate', 'enddate', or 'dates_init'.
    """
    dates = row['dates']
    
    if feature == 'dates_init':
        return True if dates == 'init' else False

    if row[feature] == row[feature]:
        return row[feature]
    else:
        if row['dates'] == row['dates'] and row['dates'] != 'init':
            return parse_dates(feature, dates)
        else:
            return pd.NaT
        

def custom_predict(X, model):
    probas = model.predict_proba(X)
    preds = []
    for entry in probas:
        first_guess = np.argmax(entry)
        first_proba = np.amax(entry)
        
        entry[first_guess] = 0
        second_guess = np.argmax(entry)
        second_proba = np.amax(entry)
        
        entry[second_guess] = 0
        third_guess = np.argmax(entry)
        third_proba = np.amax(entry)
        
        if first_proba > 0.7:
            preds.append(first_guess)
        else:
            preds.append(max([first_guess, second_guess, third_guess]))
    return np.array(preds)


def scale_predictions(preds, mem_categories, init_val=3.0, ten_pct_pt=10000):
    '''Scales predictions.
    
       First interprets the predicted category as a memory
       value.
       
       Then multiplies by a scaling value which begins
       at the init_val and decays for higher memory values,
       reaching a value of 1.1 (thus scaling by 10%) at the 
       given value called ten_pct_pt. 
       '''
    
    a = init_val - 1
    b = (1/ten_pct_pt)*math.log(0.1/a)
    
    sc_predictions = []
    for cat in preds:
        mem_val = mem_categories[cat]
        mem_val *= 1 + a*exp(b*mem_val)
        mem_val = round(mem_val)
        sc_predictions.append(mem_val)
    return sc_predictions


def add_rinfo_features(df):
    features_filename = "rinfo-features.txt"
    features_file = open(features_filename)
    all_rinfo_features = features_file.read().split('\n')

    special_features = ['grid_definition', 'params', 'format']
    normal_features = [feat for feat in all_rinfo_features
                       if feat not in special_features]

    special_alt_names = {'grid_definition': ['grid_definition', 'grid-definition'],
                         'parameters': ['parameters', 'params', 'parms'],
                         'format': ['format', 'ofmt', 'fmt'],
                         'tindex': ['tindex', 'gindex']}

    alternate_names = {feat: [feat] for feat in normal_features}
    alternate_names.update(special_alt_names)
    
    df = df[df.rinfo.notnull()]
    df['valid_rinfo'] = df.apply(lambda row: valid_rinfo(row['rinfo']), axis = 1)
    df = df[df.valid_rinfo]
    
    failed_parse = open('failed-parse.txt', 'w')

    for feature in all_rinfo_features:
        df[feature] = df.apply(lambda row: get_val_from_rinfo(row['rinfo'], feature, alternate_names), axis = 1)

    failed_parse.close()
    
    for feature in ['slat', 'nlat', 'wlon', 'elon']:
        df[feature] = df.apply(lambda row: update_lat_lon(feature, row), axis = 1)
        
    date_features = ['dates_init', 'startdate', 'enddate']
    for feature in date_features:
        df[feature] = df.apply(lambda row: update_dates(feature, row), axis = 1)
        
    df = df.rename(columns={'enddate': 'rqst_end_at',
                            'startdate': 'rqst_start_at'})
    return df


def predict_from_df(df, model_save_path, training_mode='final'):
    save_path = model_save_path
    ms = model_saver()
    ms.load(save_path)
    notes, \
        model, model_df, \
        X_features, X_train, \
        y_train_full, X_val, \
        y_val_full, X_test, \
        y_test_full = ms.get_all()

    new_features = [feat for feat in X_features if feat not in list(df.columns)]

    df = add_new_features(df, new_features=new_features)
    df = handle_missing(df)

    X = df[X_features]

    X = scale_other(X, X_train)

    mem_bin_cutoffs = [50, 100, 200, 500, 1000, 2000, 10000, 20000, 50000, 100000]
    mem_categories = {i:mem_bin_cutoffs[i] for i in range(len(mem_bin_cutoffs))}

    pred_categories = custom_predict(X, model)
    predictions = scale_predictions(pred_categories, mem_categories)
    return predictions


def main():
    model_save_path = '/glade/work/jdubeau/model-saves/class_forest_final2021-07-01-18:05/'
    rindex, request_type = read_args()
    rinfo = get_rinfo(rindex)
    print(f"rinfo string: {rinfo}")
    df = pd.DataFrame([[request_type, rinfo]], columns=['request_type', 'rinfo'])
    
    #df = pd.DataFrame(test_input, columns=['request_type', 'rinfo'])
    df = add_rinfo_features(df)
    
    print(f"Predicted memory: {predict_from_df(df, model_save_path)[0]}MB")
    

if __name__ == "__main__":
    cProfile.run('main()', 'program-stats')
    p = pstats.Stats('program-stats')
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
    
    
    
'''test_input = [['PP',
      'gui=yes;dsnum=084.1;startdate=2015-01-15 00:00;enddate=2020-11-16 12:00;parameters=3!7-0.2-1:0.0.4,3!7-0.2-1:0.0.5;ofmt=netCDF;nlat=28.75;slat=28.75;wlon=77.25;elon=77.25'],
     ['PP',
      'gui=yes;dsnum=084.1;startdate=2015-01-15 00:00;enddate=2020-11-16 12:00;parameters=3!7-0.2-1:0.0.4,3!7-0.2-1:0.0.5;ofmt=netCDF;nlat=28.75;slat=28.75;wlon=77.25;elon=77.25'],
     ['SP',
      'gui=yes;dsnum=084.1;startdate=2016-01-01 00:00;enddate=2016-12-31 18:00;parameters=3!7-0.2-1:0.3.193,3!7-0.2-1:0.2.10,3!7-0.2-1:0.19.1,3!7-0.2-1:0.0.21,3!7-0.2-1:0.7.193,3!7-0.2-1:0.1.193,3!7-0.2-1:0.1.194,3!7-0.2-1:0.1.192,3!7-0.2-1:0.1.195,3!7-0.2-1:0.6.6,3!7-0.2-1:0.1.22,3!7-0.2-1:0.6.193,3!7-0.2-1:0.7.6,3!7-0.2-1:0.7.7,3!7-0.2-1:0.1.10,3!7-0.2-1:0.1.196,3!7-0.2-1:0.0.6,3!7-0.2-1:0.5.192,3!7-0.2-1:0.4.192,3!7-0.2-1:2.3.203,3!7-0.2-1:0.3.5,3!7-0.2-1:2.0.193,3!7-0.2-1:2.4.2,3!7-0.2-1:0.3.3,3!7-0.2-1:10.2.0,3!7-0.2-1:0.19.234,3!7-0.2-1:2.0.0,3!7-0.2-1:0.0.10,3!7-0.2-1:0.0.4,3!7-0.2-1:0.3.195,3!7-0.2-1:0.0.5,3!7-0.2-1:0.2.17,3!7-0.2-1:0.2.18,3!7-0.2-1:0.3.192,3!7-0.2-1:0.14.192,3!7-0.2-1:0.1.39,3!7-0.2-1:0.3.196,3!7-0.2-1:0.1.200,3!7-0.2-1:0.0.2,3!7-0.2-1:0.1.3,3!7-0.2-1:0.1.7,3!7-0.2-1:0.3.0,3!7-0.2-1:0.3.200,3!7-0.2-1:0.3.1,3!7-0.2-1:0.1.1,3!7-0.2-1:0.0.11,3!7-0.2-1:0.1.11,3!7-0.2-1:2.0.2,3!7-0.2-1:0.1.0,3!7-0.2-1:0.7.8,3!7-0.2-1:0.6.201,3!7-0.2-1:0.7.192,3!7-0.2-1:0.0.0,3!7-0.2-1:0.6.1,3!7-0.2-1:0.14.0,3!7-0.2-1:0.1.8,3!7-0.2-1:0.2.194,3!7-0.2-1:0.2.2,3!7-0.2-1:0.5.193,3!7-0.2-1:0.4.193,3!7-0.2-1:0.2.195,3!7-0.2-1:0.2.3,3!7-0.2-1:0.2.224,3!7-0.2-1:0.2.192,3!7-0.2-1:0.2.8,3!7-0.2-1:2.0.192,3!7-0.2-1:0.1.13,3!7-0.2-1:2.0.5,3!7-0.2-1:2.0.201,3!7-0.2-1:0.2.22,3!7-0.2-1:0.3.194'],
     ['SP',
      'gui=yes;dsnum=628.0;startdate=1957-12-31 21:00;enddate=2020-11-01 00:00;parameters=1!34-241.200:11,1!34-241.200:33,1!34-241.200:34;product=1;grid_definition=;ofmt=netCDF'],
     ['BR',
      'gui=yes;dsnum=093.1;startdate=1979-01-01 00:00;enddate=2011-01-01 00:00;parameters=3!7-4.2-1:10.1.2,3!7-4.2-1:10.1.3;grid_definition=63;ofmt=netCDF;nlat=5;slat=3;wlon=7;elon=10'],
     ['BR',
      'dsnum=094.0;startdate=2020-06-23 00:00;enddate=2020-07-30 00:00;parameters=3conda.2-1:0.0.0,3.2-1:0.0.0;product=1;grid_definition=81;nlat=59;slat=39;wlon=-53;elon=13;ofmt=netCDF']]'''