from nasbench import api


# Use nasbench_full.tfrecord for full dataset (run download command above).
nasbench = api.NASBench('nasbench_only108.tfrecord')

# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2  # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2  # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]  # Binary adjacency matrix
OBJECTIVES = ['acc', 'time']  # Objectives of sorting.
OPT = [1, -1]  # 1 : Minimizing is objective ; -1 : Maximizing is objective.
INFINITE = 1000000000
Prob_size = [3.0178527546528822e-05, 0.00033196380301181704,
             0.002243270547625309, 0.010844150898386024,
             0.04008714409097245, 0.11806845927120294,
             0.28517127751455396, 0.5776012094404183, 1.0]
whole_hash_list = [hash_val for hash_val in nasbench.hash_iterator()]
