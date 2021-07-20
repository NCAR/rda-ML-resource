import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(training_path):
    '''Load training data from the given path. 
    
    Note: simply calling pd.read_json would cause 'dsnum' to be interpreted 
    as a float. Therefore we have to indicate that 'dsnum' should be interpreted
    as a string.
    
    Parameters
    ----------
    training_path (str): Path to training data stored as a JSON.
        
    Returns
    ----------
        (pandas.core.frame.DataFrame): Dataframe from given file.
    '''
    
    data_types = {'dsnum':'object'}

    df = pd.read_json(training_path, dtype = data_types)
    return df


def walltime_feat(row):
    '''Helper function for adding 'wall_time' column. 
    
    Parameters
    ----------
    row (pandas.core.series.Series): Row of dataframe to be updated.
        
    Returns
    ----------
        (float): Wall time value. Can be float('nan').
    '''
    end = row['job_end_at']
    start = row['job_start_at']
    return (end - start) / np.timedelta64(1, 's')


def timespan_feat(row):
    '''Helper function for adding 'rqst_timespan' column. 
    
    Parameters
    ----------
    row (pandas.core.series.Series): Row of dataframe to be updated.
        
    Returns
    ----------
        (float): Request timespan value. Can be float('nan').
    '''
    return (row['rqst_end_at'] - row['rqst_start_at']) / np.timedelta64(1, 'D')


def list_length_feat(feature, row, sep = ','):
    '''Helper function for adding 'grid_definition', 'level_num',
    'product_num', and 'station_num' columns.  They are all essentially 
    keeping track of the length of a list.
    
    Parameters
    ----------
    row (pandas.core.series.Series): Row of dataframe to be updated.
        
    Returns
    ----------
        (int): Length of list. Can be None.
    '''
    if not pd.isnull(row[feature]):
        return len(row[feature].split(sep))
    else:
        return None


def params_num_feat(row):
    '''Helper function for adding 'params_num' column. Slightly 
    different from the other list length features because the separator
    could be two different characters.
    
    Parameters
    ----------
    row (pandas.core.series.Series): Row of dataframe to be updated.
        
    Returns
    ----------
        (int): Number of parameters. Can be None.
    '''
    if not pd.isnull(row['parameters']):
        sep = ',' if ',' in row['parameters'] else ' '
        return list_length_feat('parameters', row, sep = sep)
    else:
        return None


def rqst_area_feat(row):
    '''Helper function for adding 'rqst_area_rect' column. 
    
    Parameters
    ----------
    row (pandas.core.series.Series): Row of dataframe to be updated.
        
    Returns
    ----------
        (float): Request area value. Can be float('nan').
    '''
    return abs((row['nlat'] - row['slat']) * (row['elon'] - row['wlon']))


def rtype_feat(feature, row):
    '''Helper function for adding 'PP', 'SP', and 'BR' columns. This
    function will be called three times, one for each column.
    
    Parameters
    ----------
    row (pandas.core.series.Series): Row of dataframe to be updated.
        
    Returns
    ----------
        (bool): Whether the row represents an entry of the given request
            type. Can be None.
    '''
    if not pd.isnull(row['request_type']):
        return True if row['request_type'] == feature else False
    else:
        return None


def converted_feat(row):
    '''Helper function for adding 'converted' column. 
    
    Parameters
    ----------
    row (pandas.core.series.Series): Row of dataframe to be updated.
        
    Returns
    ----------
        (bool): Whether an output format was specified in the request
            (which would mean that the data will be converted).
    '''
    return False if pd.isnull(row['format']) else True


def dsnum_feat(feature, row):
    '''Helper function for adding 'ds' columns, e.g. 'ds084.1'.
    
    Parameters
    ----------
    row (pandas.core.series.Series): Row of dataframe to be updated.
        
    Returns
    ----------
        (bool): Whether the dataset matches the column name. Can be None.
    '''
    if not pd.isnull(row['dsnum']):
        return row['dsnum'] == feature[2:]
    else:
        return None


def add_feature(feature, df):
    '''Adds a new feature (column) to the given dataframe. New feature must
    be one of ['wall_time', 'rqst_timespan', 'rqst_area_rect', 'grid_def_num',
    'level_num', 'product_num', 'station_num', 'params_num', 'PP', 'SP', 'BR',
    'converted'] or a feature starting with 'ds' as in 'ds084.1'.
    
    Parameters
    ----------
    feature (str): Name of new feature to add.
    df (pandas.core.frame.DataFrame): Dataframe to add column to.
        
    Returns
    ----------
        (pandas.core.frame.DataFrame): Dataframe with new column added.
    '''
    if feature == 'wall_time':
        df[feature] = df.apply(lambda row: walltime_feat(row), axis=1)
    elif feature == 'rqst_timespan':
        df[feature] = df.apply(lambda row: timespan_feat(row), axis=1)
    elif feature == 'rqst_area_rect':
        df[feature] = df.apply(lambda row: rqst_area_feat(row), axis=1)
    elif feature == 'grid_def_num':
        df[feature] = df.apply(lambda row:
                               list_length_feat('grid_definition', row),
                               axis=1)
    elif feature in ['level_num', 'product_num']:
        df[feature] = df.apply(lambda row:
                               list_length_feat(feature[:-4], row), axis=1)
    elif feature == 'station_num':
        df[feature] = df.apply(lambda row:
                               list_length_feat(feature[:-4], row, sep=' '),
                               axis=1)
    elif feature == 'params_num':
        df[feature] = df.apply(lambda row: params_num_feat(row), axis=1)
    
    elif feature in ['PP', 'SP', 'BR']:
        df[feature] = df.apply(lambda row: rtype_feat(feature, row), axis=1)
    elif feature == 'converted':
        df[feature] = df.apply(lambda row: converted_feat(row), axis=1)
    elif feature.startswith('ds'):
        df[feature] = df.apply(lambda row: dsnum_feat(feature, row), axis=1)
        
    return df


def add_new_features(df, new_features=None):
    '''Adds all new features from the given list to the given dataframe.
    If no list of features is provided, uses a default list of features to
    add, along with new columns for the five most popular dsid's in the
    given dataframe.
    
    Parameters
    ----------
    df (pandas.core.frame.DataFrame): Dataframe to add column to.
    new_features (list): List of features to add.
        
    Returns
    ----------
        (pandas.core.frame.DataFrame): Dataframe with new features added.
    '''
    if new_features == None:
        new_features = ['wall_time', 'rqst_timespan', 'grid_def_num', 'level_num',
                        'product_num', 'station_num', 'params_num', 
                        'rqst_area_rect', 'PP', 'SP', 'BR', 'converted']
    
        most_common = df['dsnum'].value_counts()[:5].index.tolist()
        for common_id in most_common:
            new_features.append('ds' + common_id)

    for feature in new_features:
        df = add_feature(feature, df)
    
    return df


def default_value(feature):
    '''Decides a default value for the given feature, to replace missing or
    null values.
    
    Parameters
    ----------
    feature (str): Name of feature.
        
    Returns
    ----------
        (int or bool): Default value for given feature.
    '''
    if feature == 'rqst_timespan':
        return 36500
    elif feature == 'rqst_area_rect':
        return 129600
    elif feature.endswith('_num'):
        return 0
    elif feature == 'converted':
        return False
    elif feature.startswith('ds'):
        return False
    elif feature in ['PP', 'BR']:
        return False
    elif feature in ['SP']:
        return True
    else:
        raise ValueError(f"Feature {feature} has no default value set.")


def fill_missing(df, features):
    '''Fill all null values in the specified columns with the default values
    provided by default_value().
    
    Parameters
    ----------
    df (pandas.core.frame.DataFrame): Dataframe to add column to.
    features (list): List of features to fill null values for.
        
    Returns
    ----------
        (pandas.core.frame.DataFrame): Dataframe with new features added.
    '''
    for feature in features: 
        df[feature] = df.apply(lambda row: default_value(feature) if pd.isnull(row[feature]) \
                                                                  else row[feature],
                               axis=1)
    return df


def categorize(row, target, categories_dict):
    '''Helper function for adding a category column (either 'mem_category'
    or 'time_category'). Returns the category that the given row falls
    into.
    
    Parameters
    ----------
    row (pandas.core.series.Series): Row of dataframe.
    target (str): Either 'mem' or 'time'.
    categories_dict (dict): Dictionary of upper bounds for each category, so
        for example the dictionary {0:50, 1:100} indicates that an entry is 
        category 0 if the corresponding value is <50, category 1 if the 
        corresponding value is >=50 and <100.
        
    Returns
    ----------
        (int or None): Category of the given row.
    '''
    if target == 'mem': 
        val = row['used_mem']
    elif target == 'time':
        val = row['wall_time']
    else:
        raise ValueError("Target for categorization must be 'mem' or 'time'.")
    
    num_classes = len(categories_dict)
    
    if val < categories_dict[0]:
        return 0
    for i in range(1, num_classes):
        if val >= categories_dict[i-1] and val < categories_dict[i]:
            return i
    return None


def make_category_col(df, target, categories_dict):
    '''Makes a column indicating which category each row belongs to in
    terms of memory usage or wall time.
    
    Parameters
    ----------
    df (pandas.core.frame.DataFrame): Dataframe to add column to.
    target (str): Either 'mem' or 'time'.
    categories_dict (dict): Dictionary of upper bounds for each category, so
        for example the dictionary {0:50, 1:100} indicates that an entry is 
        category 0 if the corresponding value is <50, category 1 if the 
        corresponding value is >=50 and <100.
    
    Returns
    ----------
        (pandas.core.frame.DataFrame): Dataframe with new column added.
    '''
    if target == 'mem':
        new_column_name = 'mem_category'
    elif target == 'time':
        new_column_name = 'time_category'
    else:
        raise ValueError("Target for categorization must be 'mem' or 'time'.")

    df[new_column_name] = df.apply(lambda row: categorize(row,
                                                          target,
                                                          categories_dict),
                                   axis=1)
    return df


def get_df(training_path='/glade/work/jdubeau/job-metrics-training.json'):
    '''Loads the stored training data and preprocesses it for training.
    
    Parameters
    ----------
    training_path (str): Path to training dataset.
    
    Returns
    ----------
        (pandas.core.frame.DataFrame): Processed dataframe ready for training.
    '''
    df = load_data()
    df = add_new_features(df)
    df = fill_missing(df)
    return df


def scale(X_train, X_val, X_test):
    '''Scales a training, validation, and testing set according to the
    statistics of the training set. Performs the default standardization, i.e.
    replaces x with (x - u)/s, where u is the mean and s is the standard
    deviation.
    
    Parameters
    ----------
    X_train (pandas.core.frame.DataFrame): Training data.
    X_val (pandas.core.frame.DataFrame): Validation data.
    X_test (pandas.core.frame.DataFrame): Testing data.
    
    Returns
    ----------
        (tuple): Three numpy.ndarray's with the normalized training,
            validation, and testing data respectively.
    '''
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_val_norm, X_test_norm

    
def scale_other(X_other, X_scale):
    '''Scales one data set according to the statistics of the other, i.e.
    replaces x in X_other with (x - u)/s, where u and s are the mean and 
    standard deviation of X_scale.
    
    Parameters
    ----------
    X_other (pandas.core.frame.DataFrame): Data to be scaled.
    X_scale (pandas.core.frame.DataFrame): Data to use for scaling.
    
    Returns
    ----------
        (numpy.ndarray): Scaled data.
    '''
    scaler = StandardScaler()
    scaler.fit(X_scale)
    return scaler.transform(X_other)
