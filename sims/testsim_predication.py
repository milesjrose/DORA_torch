simType='sym_file'
symProps = [
# ---- DRIVER (analog 0) ----
# Objects share semantics with recipient objects; predicates are distinct.
{'name': 'bigApple', 'RBs': [
    {'pred_name': 'big', 'pred_sem': ['size1', 'size2', 'size3'], 'higher_order': False,
     'object_name': 'apple', 'object_sem': ['red', 'round', 'fruit'], 'P': 'non_exist'}
], 'set': 'driver', 'analog': 0},

{'name': 'tallTree', 'RBs': [
    {'pred_name': 'tall', 'pred_sem': ['height1', 'height2', 'height3'], 'higher_order': False,
     'object_name': 'tree', 'object_sem': ['green', 'leafy', 'plant'], 'P': 'non_exist'}
], 'set': 'driver', 'analog': 0},

# ---- RECIPIENT (analog 1) ----
# Objects share semantics with driver objects (red/round, green/leafy);
# predicates have no semantic overlap with driver predicates.
{'name': 'niceBall', 'RBs': [
    {'pred_name': 'nice', 'pred_sem': ['nice1', 'nice2', 'nice3'], 'higher_order': False,
     'object_name': 'ball', 'object_sem': ['red', 'round', 'toy'], 'P': 'non_exist'}
], 'set': 'recipient', 'analog': 1},

{'name': 'goodBush', 'RBs': [
    {'pred_name': 'good', 'pred_sem': ['good1', 'good2', 'good3'], 'higher_order': False,
     'object_name': 'bush', 'object_sem': ['green', 'leafy', 'shrub'], 'P': 'non_exist'}
], 'set': 'recipient', 'analog': 1},
]
