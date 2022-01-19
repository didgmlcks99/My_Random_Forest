import bootstrap
import decision_tree
import recorder
from collections import Counter

def build_random_forest(input, max_depth=1000000000, n_trees=100, num_split_candidates=2):
    i = 1
    random_forest = []
    for _ in range(n_trees):
        bootstrapped_data = bootstrap.bootstrap_sample(input)
        tree = decision_tree.build_tree(bootstrapped_data, 0, max_depth, num_split_candidates, None)
        random_forest.append(tree)
        recorder.write_forest(tree)
        print('built random tree #' + str(i) + ' ' + str(i) + "/" + str(n_trees) +  ' ' + "{:.2f}".format((i/n_trees)*100) + '%')
        i += 1

    return random_forest

def forest_classify(trees, input):
    votes = [decision_tree.classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]

def print_random_forest(trees):
    cnt = 1
    for tree in trees:
        # decision_tree.print_tree(tree, 0)
        print(str(cnt) + '. ' + str(tree))
        cnt += 1
        print('--------------------------------------------------------')