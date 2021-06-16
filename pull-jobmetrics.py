# Goal: pull job metrics data from stratus and assemble it into
# a pandas dataframe.
# Then save the dataframe to work directory.
import boto3
import pandas as pd
import logging

url_name = 'https://stratus.ucar.edu'
bucket_name = 'rda-data'
file_prefix = 'web/jobMetrics/mem_metrics'

df_path = '/glade/work/jdubeau/job-metrics-data.json'

logging.basicConfig(filename='pull-jobmetrics-log.txt', level=logging.INFO)


def grab_bucket_contents(url, bucket, prefix):
    """ Connects to the given url and grabs the all the files in the
    given bucket which match the given prefix. Returns a list of strings
    (the filenames) and a list of botocore.response.StreamingBody objects
    which can later be read to get the actual contents of each file.
    """
    # Note: creating a boto3 client uses credentials file from ~/.aws
    client = boto3.client(endpoint_url=url, service_name='s3')

    bucket_contents = client.list_objects_v2(Bucket=bucket,
                                             Prefix=prefix)['Contents']
    obj_keys = [c['Key'] for c in bucket_contents]

    # Here bucket_contents is a list of dictionaries, one for each file in
    # the bucket matching the prefix. For example, one element of
    # bucket_contents could be:

    # {'Key': 'web/jobMetrics/mem_metrics_20-09-16:1600.json',
    # 'LastModified': ...,
    # 'ETag': ...,
    # 'Size': ...,
    # 'StorageClass': ...}

    # Therefore object_keys is the list of filenames we're interested in.
    # We use these filenames to retrieve the files themselves.


    objects = [client.get_object(Bucket=bucket, Key=k) for k in obj_keys]
    obj_bodies = [ob['Body'] for ob in objects]

    # Here is an example of a member of objects:
    # {'ResponseMetadata': ...,
    # 'AcceptRanges': ...,
    # 'LastModified': ...,
    # 'ContentLength': ...,
    # 'ETag': ...,
    # 'ContentType': ...,
    # 'Metadata': ...,
    # 'Body': <botocore.response.StreamingBody object at 0x7fa9f81b6be0>}

    return obj_keys, obj_bodies


def read_and_store(obj_keys, obj_bodies):
    """ Reads the StreamingBody objects contained in object_bodies and
    saves each of them to a pandas dataframe. Returns a list of all the
    dataframes created.
    """
    all_data = [body.read() for body in obj_bodies]

    all_dfs = []
    for i in range(len(all_data)):
        current_data = all_data[i]
        try:
            all_dfs.append(pd.read_json(current_data, orient='index'))
        except:
            logging.warning(f"Could not parse JSON data from {obj_keys[i]}")

    logging.info(f"Finished parsing JSON data \
                 ({len(all_dfs)}/{len(all_data)} successful)")
    return all_dfs


def combine_and_save(dfs_list):
    """
    Combines the dataframes in dfs_list into one, removes any duplicate
    rows, and saves the combined dataframe to a file (using the path
    given by df_path).
    """
    merged_df = pd.concat(dfs_list, axis=0)
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    merged_df.to_json(df_path)


logging.info(f"Connecting to {url_name} and loading files from {bucket_name}")
keys, bodies = grab_bucket_contents(url_name, bucket_name, file_prefix)
logging.info(f"Found {len(keys)} files matching prefix.")

logging.info("Reading and storing...")
all_dfs = read_and_store(keys, bodies)

logging.info(f"Combining into a single dataframe and saving to {df_path}")
combine_and_save(all_dfs)
