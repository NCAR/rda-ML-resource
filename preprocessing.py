import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    training_path = '/glade/work/jdubeau/job-metrics-training.json'
    data_types = {'dsnum':'object'}

    df = pd.read_json(training_path, dtype = data_types)
    return df


def walltime_feat(row):
    end = row['job_end_at']
    start = row['job_start_at']
    return (end - start) / np.timedelta64(1, 's')


def timespan_feat(row):
    return (row['rqst_end_at'] - row['rqst_start_at']) / np.timedelta64(1, 'D')


def list_length_feat(feature, row, sep = ','):
    if not pd.isnull(row[feature]):
        return len(row[feature].split(sep))
    else:
        return None


def params_num_feat(row):
    if not pd.isnull(row['parameters']):
        sep = ',' if ',' in row['parameters'] else ' '
        return list_length_feat('parameters', row, sep = sep)
    else:
        return None


def rqst_area_feat(row):
    return abs((row['nlat'] - row['slat']) * (row['elon'] - row['wlon']))


def rtype_feat(feature, row):
    if not pd.isnull(row['request_type']):
        return True if row['request_type'] == feature else False
    else:
        return None


def converted_feat(row):
    return False if pd.isnull(row['format']) else True


def dsnum_feat(feature, row):
    if not pd.isnull(row['dsnum']):
        return row['dsnum'] == feature[2:]
    else:
        return None


def add_feature(feature, df):
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
    


def fill_missing(df, features):
    for feature in features: 
        df[feature] = df.apply(lambda row: default_value(feature) if pd.isnull(row[feature]) \
                                                                  else row[feature],
                               axis=1)
    return df

def categorize(row, target, categories_dict):
    if target == 'mem': 
        val = row['used_mem']
    elif target == 'time':
        val = row['wall_time']
    else:
        raise ValueError
    
    num_classes = len(categories_dict)
    
    if val < categories_dict[0]:
        return 0
    for i in range(1, num_classes):
        if val >= categories_dict[i-1] and val < categories_dict[i]:
            return i
    return None

def make_category_col(df, target, categories_dict):
    if target == 'mem':
        new_column_name = 'mem_category'
    elif target == 'time':
        new_column_name = 'time_category'
    else:
        raise ValueError    
    df[new_column_name] = df.apply(lambda row: categorize(row,
                                                          target,
                                                          categories_dict),
                                   axis=1)
    return df

def get_df():
    df = load_data()
    df = add_new_features(df)
    df = fill_missing(df)
    return df

def scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_val_norm, X_test_norm

    
def scale_other(X_other, X_scale):
    scaler = StandardScaler()
    scaler.fit(X_scale)
    return scaler.transform(X_other)

