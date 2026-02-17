simType='sym_file'
symProps = [
# ---- DRIVER (analog 0) ---- 
# "loves(Mary, Tom)" with lover and beloved predicates sharing semantics with memory
{'name': 'lovesMaryTom', 'RBs': [
    {'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False,
     'object_name': 'Mary', 'object_sem': ['mary1', 'mary2', 'mary3'], 'P': 'non_exist'},
    {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False,
     'object_name': 'Tom', 'object_sem': ['tom1', 'tom2', 'tom3'], 'P': 'non_exist'}
], 'set': 'driver', 'analog': 0},

# ---- MEMORY (analog 1) ---- high overlap with driver (shares lover/beloved semantics)
# "loves(John, Kathy)" 
{'name': 'lovesJohnKathy', 'RBs': [
    {'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False,
     'object_name': 'John', 'object_sem': ['john1', 'john2', 'john3'], 'P': 'non_exist'},
    {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False,
     'object_name': 'Kathy', 'object_sem': ['kathy1', 'kathy2', 'kathy3'], 'P': 'non_exist'}
], 'set': 'memory', 'analog': 1},

# ---- MEMORY (analog 2) ---- low overlap with driver (no shared semantics)
# "hates(Bob, Alice)" 
{'name': 'hatesBobAlice', 'RBs': [
    {'pred_name': 'hater', 'pred_sem': ['hater1', 'hater2', 'hater3'], 'higher_order': False,
     'object_name': 'Bob', 'object_sem': ['bob1', 'bob2', 'bob3'], 'P': 'non_exist'},
    {'pred_name': 'hated', 'pred_sem': ['hated1', 'hated2', 'hated3'], 'higher_order': False,
     'object_name': 'Alice', 'object_sem': ['alice1', 'alice2', 'alice3'], 'P': 'non_exist'}
], 'set': 'memory', 'analog': 2}
]

