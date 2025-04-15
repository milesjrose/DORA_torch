import ast
import sys
import networkx as nx
import matplotlib.pyplot as plt

class FunctionCallGraph(ast.NodeVisitor):
    def __init__(self):
        # Directed graph: nodes are function names; edges are calls (caller -> callee)
        self.graph = nx.DiGraph()
        self.current_func = None

    def visit_FunctionDef(self, node):
        # Record the function definition (assume all functions here are methods of basicRunDORA.runDORA)
        self.current_func = node.name
        self.graph.add_node(node.name)
        self.generic_visit(node)
        self.current_func = None

    def visit_Call(self, node):
        # Only record calls if we're inside a function definition.
        caller = self.current_func
        if caller:
            # Case 1: Attribute call (e.g., self.foo(), basicRunDORA.runDORA.bar())
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                self.graph.add_edge(caller, method_name)
            # Case 2: Name call (e.g., indexMemory(...))
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
                self.graph.add_edge(caller, func_name)
        self.generic_visit(node)

def analyze_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filename)
    analyzer = FunctionCallGraph()
    analyzer.visit(tree)
    return analyzer.graph

def extract_dfs_tree(full_graph, start_function):
    if start_function not in full_graph.nodes:
        print(f"Starting function '{start_function}' not found in the graph.")
        sys.exit(1)
    # Generate a DFS tree (which is cycle-free)
    dfs_tree = nx.dfs_tree(full_graph, source=start_function)
    return dfs_tree

def safe_topological_order(graph):
    try:
        # The DFS tree is acyclic so we can topologically sort it.
        ordered = list(nx.topological_sort(graph))
        return ordered
    except nx.NetworkXUnfeasible:
        print("❌ The DFS tree contains cycles, which should not happen.")
        return []

def draw_graph(graph, output_image):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=0.5, iterations=50)
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=800)
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=20)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family="sans-serif")
    plt.title("DFS Call Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Graph saved to {output_image}")

def write_log(log_filename, dfs_tree, topo_order):
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("DFS Call Graph:\n")
        for node in dfs_tree.nodes:
            callees = list(dfs_tree.successors(node))
            f.write(f"{node} -> {callees}\n")
        f.write("\nSafe Topological Order (refactoring order):\n")
        for i, func in enumerate(reversed(topo_order), 1):
            f.write(f"{i:2d}. {func}\n")
    print(f"Log saved to {log_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_dfs_call_graph.py <python_file> <start_function>")
        sys.exit(1)

    filename = sys.argv[1]
    start_function = sys.argv[2]

    # Build the full call graph from the target file.
    full_graph = analyze_file(filename)
    # Extract the DFS tree starting from the given function (cycle-free).
    dfs_tree = extract_dfs_tree(full_graph, start_function)
    # Compute a safe topological order for refactoring (leaves first).
    topo_order = safe_topological_order(dfs_tree)

    # Define output file names based on the starting function.
    png_filename = f"./dfs_graphs/dfs_{start_function}.png"
    log_filename = f"./dfs_graphs/dfs_{start_function}.log"

    # Write the DFS tree and topological order to the log file.
    write_log(log_filename, dfs_tree, topo_order)
    # Draw and save the graph visualization.
    draw_graph(dfs_tree, png_filename)
