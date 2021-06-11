# Goal: pull job metrics data from stratus and assemble it into a pandas dataframe.
# Then save the dataframe to work directory.
import boto3
import pandas as pd

url_name = 'https://stratus.ucar.edu'
bucket_name = 'rda-data'
file_prefix = 'web/jobMetrics/mem_metrics'

print(f"Connecting to {url_name}...")
# Note: creating a boto3 client uses credentials file from ~/.aws
client = boto3.client(endpoint_url=url_name, service_name='s3')

print(f"Loading files from bucket {bucket_name} with prefix {file_prefix}")
bucket_contents = client.list_objects_v2(Bucket = bucket_name, Prefix = file_prefix)['Contents']
object_keys = [c['Key'] for c in bucket_contents]

print(f"Found {len(bucket_contents)} files matching prefix. Retrieving them...")
objects = [client.get_object(Bucket = bucket_name, Key = k) for k in object_keys]
object_bodies = [ob['Body'] for ob in objects]

print("Reading files...")
all_data = [body.read() for body in object_bodies]

print("Parsing JSON data from files and loading into dataframes...")
all_dfs = []
for i in range(len(all_data)):
	current_data = all_data[i] 
	try:	
		all_dfs.append(pd.read_json(current_data, orient='index'))	
	except:
		print(f"Warning: could not parse JSON data from {object_keys[i]}")
	
print(f"Finished parsing JSON data ({len(all_dfs)}/{len(all_data)} successful)")

print("Combining dataframes...")
merged_df = pd.concat(all_dfs, axis = 0)

print("Dropping duplicate rows...")
merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

print("Dataframe created with the following info:")
merged_df.info()

print("Saving dataframe to work directory...")
path = '/glade/work/jdubeau/job-metrics-data.json'

merged_df.to_json(path)

print('Done.')
