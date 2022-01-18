import decision_tree

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

tree = decision_tree.build_tree(inputs)
decision_tree.print_tree(tree, 0)

print(decision_tree.classify(tree, {'level':'Junior', 'lang':'Java', 'tweets':'yes', 'phd':'no'}))