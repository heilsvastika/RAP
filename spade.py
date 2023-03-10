# extract sequential patterns from dataset via SPADE algorithm
import pandas as pd
from collections import defaultdict
import numpy as np

# load the dataset
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    # user_train = {}
    # user_valid = {}
    # user_test = {}
    # assume user/item index starting from 1
    f = open('%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    data = []
    for k, v in User.items():
        data.append(v)

    return data, usernum, itemnum

# Define the SPADE algorithm, return a patterns_dict including the sequential patterns whose length is equivalent to 1~length
def SPADE(sequences, support, length) -> {}:
    # Initialize the frequent patterns dictionary
    frequent_patterns = defaultdict(int)
    patterns_dict = {}

    if length >= 1:
        # Generate the frequent 1-sequences
        frequent_1_sequences = defaultdict(int)
        for sequence in sequences:
            for item in sequence:
                frequent_1_sequences[item] += 1
        patterns_dict['1'] = frequent_1_sequences

    # Generate the frequent 2-sequences
    if length >= 2:
        frequent_2_sequences = defaultdict(int)
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                for j in range(i + 1, len(sequence)):
                    if sequence[i] != sequence[j]:
                        frequent_2_sequences[(sequence[i], sequence[j])] += 1
        patterns_dict['2'] = frequent_2_sequences

    if length >= 3:
        # Generate the frequent k-sequences
        for k in range(3, length + 1):
            frequent_k_sequences = defaultdict(int)
            for sequence in sequences:
                for i in range(len(sequence) - k + 1):
                    # the sequence whose length is equivalent to k
                    subsequence = sequence[i:i + k - 1]
                    for j in range(i + k - 1, len(sequence)):
                        if sequence[j] not in subsequence:
                            frequent_k_sequences[tuple(subsequence + [sequence[j], ])] += 1

            # Prune the infrequent k-sequences
            frequent_k_sequences = {k_sequence: count for k_sequence, count in frequent_k_sequences.items() if
                                    count >= support}

            # Add the frequent k-sequences to the frequent patterns dictionary
            for k_sequence in frequent_k_sequences:
                frequent_patterns[k_sequence] += frequent_k_sequences[k_sequence]
            patterns_dict[str(k)] = frequent_k_sequences

    # Prune the infrequent patterns
    frequent_patterns = {pattern: count for pattern, count in frequent_patterns.items() if count >= support}

    # Filter the frequent patterns to only include those with length 5
    frequent_patterns = {pattern: count for pattern, count in frequent_patterns.items() if len(pattern) == 5}

    return patterns_dict

# generate the pattern_matrix according to the patterns_dict
def gen_ptn_mx(patterns_dict):
    ptn_mx_dict = {}
    if patterns_dict:
        # generate a n*n matrix, where n=itemnum, elements in the matrix have default values
        for i in range(1, len(patterns_dict)):
            # the n*n matrix store the times coexistence of item ei and ej
            ptn_mx = np.full((itemnum, itemnum), 0, dtype=np.uint8)
            num = int(i) + 1
            for u, v in patterns_dict[str(num)].items():
                start = u[0] - 1
                end = u[-1] - 1
                ptn_mx[start][end] = v
            ptn_mx_dict[str(i + 1)] = ptn_mx
    return ptn_mx_dict
