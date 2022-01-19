import random

# choose random samples from origin dataset
# in the same size
def bootstrap_sample(data):
    return [random.choice(data) for _ in data]