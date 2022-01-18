import entropy
from functools import partial

def build_tree(inputs, split_candidates=None):
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues

    if num_trues == 0 : return False
    if num_falses == 0 : return True

    if not split_candidates:
        return num_trues >= num_falses
    
    best_attribute = min(split_candidates,
                            key=partial(entropy.partition_entropy_by, inputs))
    
    partitions = entropy.partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                        if a != best_attribute]

    subtrees = {attribute_value : build_tree(subset, new_candidates)
                for attribute_value, subset in partitions.items()}
    
    subtrees[None] = num_trues > num_falses

    return (best_attribute, subtrees)


def classify(tree, input):
    if tree in [True, False]:
        return tree
    
    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)

    if subtree_key not in subtree_dict:
        subtree_key = None
    
    subtree = subtree_dict[subtree_key]
    return classify(subtree, input)

def print_tree(tree, n):
    attribute, subtree_dict = tree

    print(str(n) + '. ' + str(attribute), end='')
    if type(subtree_dict) == dict:
        print(' - ', end='')
        for key in subtree_dict:
            print(key, end=', ')
        print()
        
        for key in subtree_dict:
            if type(subtree_dict[key]) == tuple:
                print()
                print(str(key) + ' >')
                print_tree(subtree_dict[key], n+1)
            elif type(subtree_dict[key]) == dict:
                print(str(key) + ": " + str(subtree_dict[key]), end=', ')
            else:
                print(str(key) + ": " + str(subtree_dict[key]))
        print()