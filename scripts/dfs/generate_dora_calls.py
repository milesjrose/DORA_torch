import ast
import sys

class RunDORACallAnalyzer(ast.NodeVisitor):
    def __init__(self):
        # Variables that are known to hold an instance of basicRunDORA.runDORA
        self.runDORA_instances = set()
        # List of calls to methods on runDORA instances or directly on basicRunDORA.runDORA
        self.calls = []

    def visit_Assign(self, node):
        # Look for assignments like: var = basicRunDORA.runDORA(...)
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if self._is_runDORA_constructor(func):
                # Mark every target variable name as an instance
                for target in node.targets:
                    # Handle simple assignment names: var = ...
                    if isinstance(target, ast.Name):
                        self.runDORA_instances.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check if the call is made on an instance or on the class directly.
        if isinstance(node.func, ast.Attribute):
            # Check for instance calls: variable.method(...)
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                if var_name in self.runDORA_instances:
                    self.calls.append((node.lineno, f"{var_name}.{node.func.attr}"))
            # Check for direct calls on the class: basicRunDORA.runDORA.method(...)
            elif isinstance(node.func.value, ast.Attribute):
                # This checks for patterns like basicRunDORA.runDORA.method(...)
                attr = node.func.value
                if (isinstance(attr.value, ast.Name) and 
                    attr.value.id == "basicRunDORA" and 
                    attr.attr == "runDORA"):
                    self.calls.append((node.lineno, f"basicRunDORA.runDORA.{node.func.attr}"))
        self.generic_visit(node)

    def _is_runDORA_constructor(self, func):
        """
        Returns True if the function call node represents a call to basicRunDORA.runDORA
        """
        # Pattern: basicRunDORA.runDORA(...)
        if isinstance(func, ast.Attribute):
            if (isinstance(func.value, ast.Name) and
                func.value.id == "basicRunDORA" and
                func.attr == "runDORA"):
                return True
        # (If you support additional patterns, add them here)
        return False

def analyze_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filename)

    analyzer = RunDORACallAnalyzer()
    analyzer.visit(tree)

    # Output the calls found
    if analyzer.calls:
        print("Calls to methods of basicRunDORA.runDORA:")
        for lineno, call in sorted(analyzer.calls, key=lambda x: x[0]):
            print(f"Line {lineno}: {call}")
    else:
        print("No calls to methods of basicRunDORA.runDORA were found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_runDORA_calls.py <filename>")
    else:
        analyze_file(sys.argv[1])
