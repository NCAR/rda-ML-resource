import json
import logging

def main():
    config_file = open("model-config.json", 'w')
    
    settings_dict = {}
    
    settings_dict['model_paths'] = \
        {'time': '/glade/work/jdubeau/model-saves/time_forest_final2021-07-12-14:03/',
         'mem': '/glade/work/jdubeau/model-saves/class_forest_final2021-07-12-14:34/',
         'time_regr': '/glade/work/jdubeau/model-saves/time_regr_log_gboost_final2021-07-14-15:34/'}
    
    
    settings_dict['mem_scaling'] = {'init_val': 2.31, 
                                    'ten_pct_pt': 1000}
    settings_dict['time_scaling'] = {'init_val':6.00, 
                                     'ten_pct_pt': 30000}
    settings_dict['time_regr_scaling'] = {'init_val': 1.5, 
                                          'ten_pct_pt': 1000}
    
    settings_dict['logging'] = {'file_path': '/glade/u/home/jdubeau/github/rda-ML-resource/apply-model.log',
                                'level': logging.INFO}
    
    json.dump(settings_dict, config_file, indent=4)
    config_file.close()

if __name__ == "__main__":
    main()
