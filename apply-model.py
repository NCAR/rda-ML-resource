# Full machine learning process, start to finish:
# Takes in an rindex number and request type, and uses the current best 
# ML models to predict the time and the request will use.
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

def read_args():
    '''Reads command-line arguments and returns them, after translating the
    request type from 'P', 'R', or 'S' to 'PP', 'BR', or 'SP' respectively.
    
    Checks that the given rindex is an integer and the given request type
    is one of P, R, S.
    '''
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} <rindex> <rtype>")

    rtype_dict = {'P':'PP', 'R':'BR', 'S':'SP'}
    
    try:
        rindex = int(sys.argv[1])
        request_type = rtype_dict[sys.argv[2]]
    except:
        raise ValueError("Invalid rindex or request type. " 
                         + "rindex must be an integer, request type must be "
                         + "one of 'P', 'R', 'S'. ")
    
    return rindex, request_type

def get_rinfo(rindex):
    '''Gets the rinfo string associated to the given rindex by connecting
    to the dsrqst table and performing a mysql query.
    
    Parameters
    ----------
    rindex : int
        rindex number to look up.
    '''
    credentials_path = './dsrqst-creds.json'
    try:
        credentials = json.load(open(credentials_path))
    except:
        raise FileNotFoundError(f"Could not find dsrqst credentials file at {credentials_path}")
        
    host_name = 'rda-db.ucar.edu'
    db_name = 'dssdb'
    
    conn = mysql.connector.connect(user=credentials['user'],
                                   password=credentials['password'],
                                   host=host_name,
                                   database=db_name)
    curs = conn.cursor()
    curs.execute("SELECT rinfo FROM dsrqst WHERE rindex = %s",
                 (rindex,))
    
    try:
        rinfo = curs.fetchone()[0]
    except:
        raise ValueError(f"Found no rinfo string matching rindex {rindex}.")
    
    conn.close()
    return rinfo

def handle_missing_rinfo_val(feature):
    """Decides the appropriate 'null value' for when a feature is
    not present in an rinfo string. This function only handles missing
    basic rinfo values, like 'elon' and 'wlon' -- for missing composite
    values like 'rqst_area_rect', see handle_missing in preprocessing.py.
    
    Parameters
    ----------
    feature : str
        Name of feature to decide null value for.
    """

    if feature == 'gui':
        return False
    elif feature in ['startdate', 'enddate']:
        return pd.NaT
    else:
        return float('nan')
    

def format_rinfo_val(rinfo, feature, val):
    """Converts a value found in an rinfo string to the correct data type
    so that it can be entered into the pandas dataframe.
    
    Example input: 
    rinfo = '...;elon=76.4;...'
    feature = 'elon'
    val = '76.4'

    Example output: 76.4 (type float)
    
    Parameters
    ----------
    rinfo : str
        Full rinfo string. (Used to print error message when conversion fails.)
    feature : str
        Name of feature from rinfo string.
    val : str
        Value of given feature (as a string) to be converted.
    """

    if val == '':
        return handle_missing_rinfo_val(feature)

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

    
def get_rinfo_val(rinfo, feature, rinfo_feature_names):
    """Finds the value of the given feature in the given rinfo string.
    If the feature is not present in rinfo, calls handle_missing_rinfo_val.
    If the feature is present, uses format_rinfo_val to convert the value 
    to the appropriate data type before returning.
    
    Example input: 
    rinfo = '...;elon=76.4;...' 
    feature = 'elon'

    Example output: '76.4'
    
    Parameters
    ----------
    rinfo : str
        Full rinfo string.
    feature : str
        Name of feature from rinfo string.
    alternate_names : dict
        Dictionary relating each feature name to a list of alternate
        names for the same feature.
    """
    
    rinfo = rinfo.replace('%3D', '=')
    
    if ';' in rinfo:
        sep = ';'
    else:
        sep = '&'

    val = ''
    for name in rinfo_feature_names[feature]:
        if rinfo.lower().find(name) != -1:
            start_ind = rinfo.lower().find(name) + len(name) + 1
            end_ind = rinfo.find(sep, start_ind)
            if end_ind != -1:
                val = rinfo[start_ind:end_ind]
            else:
                val = rinfo[start_ind:]
                
    val = format_rinfo_val(rinfo, feature, val)
    return val


    def valid_rinfo(rinfo):
    """Currently unused.
    Decides whether an rinfo string is 'valid.'
    In practice, just serves to filter out a few problematic rinfo strings.
    
    Parameters
    ----------
    rinfo : str
        Full rinfo string.
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
    
    Parameters
    ----------
    val : str
        String representing 'lats' or 'lons' value.
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
    
    Example input: 
    feature = 'nlat'
    row = <row in which row['lats'] = '45 S 50 N'>
    Example output: 50.0
    
    Parameters
    ----------
    feature : str
        Name of feature to be updated (slat, nlat, wlon, or elon).
    row : pandas.core.series.Series
        Row of dataframe to be updated.
    """
    # First check to see if there is a non-null value for 
    # 'lats' or 'lons' in the row. The two always come together,
    # so if suffices to just check for one of them.
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
    
    Example input: 
    feature = 'enddate'
    dates = '2019-01-01 00:00 2019-12-31 18:00'
    Example output: 2019-12-31 18:00 (pandas datetime object)
    
    Parameters
    ----------
    feature : str
        Name of feature to be parsed out of 'dates' column.
    dates : str
        Content of 'dates' column.
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
    
    Parameters
    ----------
    feature : str
        Name of feature to be updated.
    row : pandas.core.series.Series
        Row to be updated.
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
        

def custom_predict(X, model, threshold=0.7):
    '''Custom method of making predictions based on the model's probability 
    estimates. Finds the three most likely guesses and selects the highest
    class, unless the model is very sure about its first guess. Returns
    an array of predicted classes.
    
    Parameters
    ----------
    X : numpy.ndarray of shape (n_entries, n_features)
        2-D array of entries to predict.
    model : sklearn.ensemble._forest.RandomForestClassifier 
            or other sklearn classifier
        Fitted model to get predictions from.
    '''
    probas = model.predict_proba(X)
    preds = []
    for entry in probas:
        first_guess = np.argmax(entry)
        first_proba = np.amax(entry)
        
        entry[first_guess] = 0
        second_guess = np.argmax(entry)
        
        entry[second_guess] = 0
        third_guess = np.argmax(entry)
        
        if first_proba > threshold:
            preds.append(first_guess)
        else:
            preds.append(max([first_guess, second_guess, third_guess]))

    return np.array(preds)


def translate_predictions(preds, categories_dict, init_val=3.0, ten_pct_pt=10000):
    '''Translates predictions from categories into actual values.
    
    First interprets the predicted category as a value using the
    given dictionary.

    Then multiplies that amount by a scaling factor which begins
    at the given init_val and decays for higher memory values,
    reaching a value of 1.1 (thus scaling by 10%) at the 
    given value called ten_pct_pt. 
       
    Parameters
    ----------
    preds : numpy.ndarray of shape (n_entries,)
        Array of predicted categories.
    categories_dict : dict
        Dictionary to translate categories into values.
    init_val : float (default 3.0)
        Initial scaling value. Values close to zero will be scaled up by
        approximately this amount.
    ten_pct_pt : int (default 10000)
        Ten percent point. Values near the ten percent point will be 
        scaled up by approximately 10%.
    '''
    a = init_val - 1
    b = (1/ten_pct_pt)*math.log(0.1/a)
    
    scaled_preds = []
    for category in preds:
        val = categories_dict[category]
        val *= 1 + a*exp(b*val)
        val = round(val)
        scaled_preds.append(val)

    return scaled_preds

def get_rinfo_feature_names():
    '''Determines which features will be parsed from the rinfo string,
    and what other possible names those features could have. Returns 
    a dictionary mapping each feature name to a list of possible
    feature names representing the same value.
    '''
    all_rinfo_features = ['dates', 'dsnum', 'elon', 'enddate',
                          'format', 'grid_definition', 'gui',
                          'lats', 'level', 'lons', 'nlat',
                          'parameters', 'product', 'slat', 'startdate',
                          'station', 'tindex', 'wlon']

    special_alt_names = {'grid_definition': ['grid_definition', 'grid-definition'],
                         'parameters': ['parameters', 'params', 'parms'],
                         'format': ['format', 'ofmt', 'fmt'],
                         'tindex': ['tindex', 'gindex']}
    
    normal_features = [feat for feat in all_rinfo_features
                       if feat not in special_alt_names]

    rinfo_feature_names = {feat: [feat] for feat in normal_features}
    rinfo_feature_names.update(special_alt_names)
    
    return rinfo_feature_names


def add_rinfo_features(df):
    '''Takes a dataframe with an rinfo column and adds all features that
    can be parsed out of the column. 
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe with rinfo column.
    '''
    rinfo_feature_names = get_rinfo_feature_names()
    
    failed_parse = open('failed-parse.txt', 'w')

    for feature in rinfo_feature_names:
        df[feature] = df.apply(lambda row: 
                               get_rinfo_val(row['rinfo'], 
                                             feature, 
                                             rinfo_feature_names),
                               axis = 1)

    failed_parse.close()
    return df

def combine_redundant_features(df):    
    '''Separates the information from columns 'lats'/'lons' 
    and 'dates' into the columns ('slat', 'nlat', 'wlon', 'elon') 
    and ('startdate', 'enddate', 'dates_init'), respectively.
    Drops the now-unnecessary 'lats', 'lons', and 'dates' columns and 
    renames 'startdate' and 'enddate' to 'rqst_start_at' and 'rqst_end_at'.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe to process.
    '''
    for feature in ['slat', 'nlat', 'wlon', 'elon']:
        df[feature] = df.apply(lambda row: update_lat_lon(feature, row), axis = 1)

    for feature in ['dates_init', 'startdate', 'enddate']:
        df[feature] = df.apply(lambda row: update_dates(feature, row), axis = 1)
    
    df.drop(labels=['lats', 'lons', 'dates'], axis=1, inplace=True)

    df.rename(columns={'enddate': 'rqst_end_at',
                       'startdate': 'rqst_start_at'},
              inplace=True)
    return df

def process_with_model(df, X_features, X_train):
    '''Preprocess the data according to how the model was made.
    I.e. add any features the model uses that aren't already there,
    and scale the data the same way the model's input data was 
    scaled in training. Returns a numpy.ndarray with the processed
    data, ready for the model to predict.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe to process.
    X_features : list
        List of feature names the model uses.
    X_train: pandas.core.frame.DataFrame
        Training set the model used (unscaled).
    '''
    new_features = [feat for feat in X_features if feat not in df.columns]
    df = add_new_features(df, new_features)
    
    df = handle_missing(df)

    X = df[X_features]
    X = scale_other(X, X_train)
    
    return X

def predict_with_model(X, model, target):
    '''Use the model to predict either the wall time or the used
    memory for the given (already processed) input data.
    
    Parameters
    ----------
    X : pandas.core.frame.DataFrame
        Processed input data.
    model : sklearn.ensemble._forest.RandomForestClassifier 
            or other sklearn classifier
        Fitted model to get predictions from.
    target : str
        Either 'mem' or 'time'.
    '''
    if target == 'mem':
        categories_dict = {0:50, 1:100, 2:200,
                           3:500, 4:1000, 5:2000,
                           6:10000, 7:20000, 8:50000}
    else:
        categories_dict = {0:60, 1:120, 2:300, 
                           3:600, 4:1800, 5:3600,
                           6:14400, 7:43200}

    pred_categories = custom_predict(X, model)
    predictions = translate_predictions(pred_categories, categories_dict)
    return predictions

def format_predictions(mem, time):
    '''Formats predictions into a PBS formatted resource request string.
    
    Parameters
    ----------
    mem : int
        Number of megabytes predicted.
    time : int
        Number of seconds predicted.
    '''
    mem_string = str(mem)+'mb'
    
    time_string = str(pd.to_timedelta(time, unit='S'))[-8:]
    
    output_string = f"-l walltime={time_string},select=1:mem={mem_string}"
    
    return output_string

def main():
    mem_model_save_path = '/glade/work/jdubeau/model-saves/class_forest_final2021-07-01-18:05/'
    time_model_save_path = '/glade/work/jdubeau/model-saves/time_forest2021-07-07-13:48/'
    save_paths = {'mem': mem_model_save_path, 'time': time_model_save_path}
    
    rindex, request_type = read_args()
    rinfo = get_rinfo(rindex)
    input_df = pd.DataFrame([[request_type, rinfo]], columns=['request_type', 'rinfo'])

    input_df = add_rinfo_features(input_df)
    input_df = combine_redundant_features(input_df)
    
    predictions = []
    
    for target in ['mem', 'time']:
        ms = model_saver()
        ms.load(save_paths[target])
        model, X_features, X_train = ms.get_min()
        X = process_with_model(input_df, X_features, X_train)
        pred = predict_with_model(X, model, target)
        predictions.append(pred)
    
    predictions = np.ravel(predictions)
    print(format_predictions(predictions[0], predictions[1]))
    

if __name__ == "__main__":
    try:
        #cProfile.run('main()', 'program-stats')
        main()
    except:
        print(sys.exc_info())
    #p = pstats.Stats('program-stats')
    #p.strip_dirs().sort_stats(SortKey.TIME).print_stats(20)
