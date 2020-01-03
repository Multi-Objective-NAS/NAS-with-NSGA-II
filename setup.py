import subprocess
# Download the raw data (only 108 epoch data points, for full dataset,
# uncomment the second line for nasbench_full.tfrecord).

#subprocess.call('curl -O https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord', shell = True)
subprocess.call('curl -O https://storage.googleapis.com/nasbench/nasbench_full.tfrecord', shell = True)

# Clone and install the code and dependencies.subprocess.call('git clone https://github.com/google-research/nasbench', shell = True )
subprocess.call('pip install ./nasbench', shell = True)

# Initialize the NASBench object which parses the raw data into memory (this
# should only be run once as it takes up to a few minutes).
