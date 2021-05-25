# Goal: pull job metrics data from stratus and assemble it into a pandas dataframe.
# Then save the dataframe to work directory.
import boto3
import pandas as pd

url_name = 'https://stratus.ucar.edu'
bucket_name = 'rda-data'
file_prefix = 'web/jobMetrics/mem_metrics'

print('Connecting to {}...'.format(url_name))
# Note: creating a boto3 client uses credentials file from ~/.aws
client = boto3.client(endpoint_url=url_name, service_name='s3')

print('Loading files from bucket {} with prefix {}'.format(bucket_name, file_prefix))
bucket_contents = client.list_objects_v2(Bucket = bucket_name, Prefix = file_prefix)['Contents']
object_keys = [c['Key'] for c in bucket_contents]

print('Found {} files matching prefix. Retrieving them...'.format(len(bucket_contents)))
objects = [client.get_object(Bucket = bucket_name, Key = k) for k in object_keys]
object_bodies = [ob['Body'] for ob in objects]

print('Reading files...')
all_data = [body.read() for body in object_bodies]

print('Parsing JSON data from files and loading into dataframes...')
all_dfs = []
for i in range(len(all_data)):
	current_data = all_data[i] 
	try:	
		all_dfs.append(pd.read_json(current_data, orient='index'))	
	except:
		print('Warning: could not parse JSON data from {}'.format(object_keys[i]))
	
print('Finished parsing JSON data ({}/{} successful)'.format(len(all_dfs), len(all_data)))

print('Combining dataframes...')
merged_df = pd.concat(all_dfs, axis = 0)

print('Combined dataframe created with the following info:')
print(merged_df.info())

print('Saving dataframe to work directory...')
path = '/glade/work/jdubeau/job-metrics-data.pkl'
merged_df.to_pickle(path)

print('Done.')
