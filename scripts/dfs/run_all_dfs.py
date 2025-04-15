import subprocess

# List of methods to analyze (commented-out alternatives are skipped)
methods = [
    "do_retrieval",
    "do_map",
    "do_predication",
    "do_rel_form",
    "do_rel_gen",
    "do_schematization",
    "do_compression",
    "do_entropy_ops_between",
    "do_entropy_ops_within"
]

# Target file that contains the basicRunDORA.runDORA class
target_file = "basicRunDORA.py"

# Path to the DFS call graph generator script
dfs_script = "generate_dfs_call_graph.py"

for method in methods:
    print(f"Generating DFS call graph for method: {method}")
    # Construct the command: python generate_dfs_call_graph.py basicRunDORA.py <method>
    command = ["python", dfs_script, target_file, method]
    
    # Run the command and wait for it to finish.
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print stdout and stderr for logging purposes.
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print("-" * 50)