from collections import defaultdict

groups = defaultdict(list)

groups['a'] = 1
groups['b'] = 2

print(groups['a'])
print(groups['b'])
print(groups['c'])

val = 0
subtrees = {val : (i, j)
                for i, j in [(1, 2), (3, 4)]}
print(subtrees)