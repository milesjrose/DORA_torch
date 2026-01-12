from DORA_bridge.new_network_state_generator import NewNetworkStateGenerator
from DORA_bridge.utils import compare_states
from DORA_bridge.test_data_generator import TestDataGenerator

ogen = TestDataGenerator()
ngen = NewNetworkStateGenerator()

ogen.load_sim("sims/testsim15.py")
ngen.load_sim("sims/testsim15.py")

o_state = ogen.get_state()
n_state = ngen.get_state()

ogen.print_tokens()

print("\n\n----------------[ META DATA ]-----------------")
print(" key : old value, new value")
for key in o_state["metadata"]:
    print(f"{key}: {o_state['metadata'][key]}, {n_state['metadata'][key]}\n")


old_pos = o_state["tokens"]["POs"]
new_pos = n_state["tokens"]["POs"]
print("----------------[ POS ]-----------------")
print("\nold:")
for po in old_pos:
    print(po["name"])

print("\nnew:")
for po in new_pos:
    print(po["name"])


print("\n\n----------------[ RESULTS ]-----------------")
compare_states(o_state, n_state)