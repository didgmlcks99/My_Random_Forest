import random_forest

# default settings
num_trees = 3
num_split_candidates = 2

keys = ['level', 'lang', 'tweets', 'phd']

inputs = [({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False),
            ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'}, False),
            ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
            ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
            ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
            ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, False),
            ({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, True),
            ({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
            ({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
            ({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
            ({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
            ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, True),
            ({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'}, True),
            ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False),]

trees = random_forest.build_random_forest(inputs, num_trees, num_split_candidates)
# random_forest.print_random_forest(trees)
res = random_forest.forest_classify(trees, {'level':'Junior', 'lang':'Java', 'tweets':'yes', 'phd':'no'})
print(res)
# print(decision_tree.classify(tree, {'level':'Junior', 'lang':'Java', 'tweets':'yes', 'phd':'no'}))