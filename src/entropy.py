import math
from collections import Counter
from collections import defaultdict

# 1. returns the calculated entropy of single attribute
def partition_entropy_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

# 2. partition data by attribute's specific trait (key)
# groups is a set of subsets (key=trait, value=data)
def partition_by(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute]
        groups[key].append(input)
    return groups

# 3. subset is for one attribute's trait
# which hold key as trait and value as data
def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    
    return sum(data_entropy(subset) * len(subset) / total_count
                for subset in subsets)

# 4. return entropy for a subset of a certain attribute
def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

# 5. return probability of a class in a subset
def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

# 6. return the entropy for single trait
def entropy(class_probabilities):
    return sum(-p * math.log(p, 2)
                for p in class_probabilities
                if p)