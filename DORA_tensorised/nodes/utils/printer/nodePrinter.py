# nodes/utils/printer/nodePrinter.py
# Provides a class for printing the nodes/connections to console or a file.

import torch

from ...enums import *

from .print_table import tablePrinter

class nodePrinter(object):
    """
    Used to print nodes/connections to console or a file.
    
    Attributes:
        print_to_console (bool): Whether to print to the console.
        log_file (str): The file to print to.
        default_feats (list): Types to print from tokens by default
    """
    def __init__(self,print_to_console: bool = True, print_to_file: bool = True, file_path: str = None):
        """
        Initialize the node printer.

        Args:
            print_to_console (bool): Whether to print to the console.
            print_to_file (bool): Whether to print to a file.
            file_path (str) (Optional): The path to the file to print to. (Defaults to printer/output.log)
        """
        self.default_feats = [TF.ID, TF.TYPE, TF.SET, TF.ANALOG, TF.PRED, TF.DELETED]
        self.print_to_console = print_to_console
        self.print_to_file = print_to_file
        if self.print_to_file:
            if file_path is None:
                self.file_path = "DORA_tensorised/nodes/utils/printer/output/output.log"
            else:
                self.file_path = file_path
        else:
            self.file_path = None

    def print_set(self, tensor, feature_types = None, mask = None, label_values = True, label_names = True, headers = None, print_cons = True, cons_headers = None, links_headers = None):
        """
        Print the token tensor for a given set.
        Args:
            tensor (Token_Tensor): The set object to print
            feature_types (list): List of features to print. (IE: TF.SET, TF.ID, etc.)
            mask (torch.Tensor): Mask of subtensor to print.
            label_values (bool): Whether to convert features floats to their enum names. (IE: TF.TYPE == 0.0 -> TYPE(0.0).name)
            label_names (bool): Whether to include the names for each node. (IE: ID==0 -> tensor.names[0])
            headers (list): The headers to print, defaults to "Set: {set.name} Tokens" if left as None.
            print_cons (bool): Whether to print the connections tensor.
            cons_headers (list): The connections headers to print, defaults to "Set: {set.name} Connections" if left as None.
            links_headers (list): The links headers to print, defaults to "Set: {set.name} Links" if left as None.
        """
        token_set = tensor.token_set
        if headers is None:
            headers = [f"{token_set.name} Tokens:"]
            if cons_headers is None:
                cons_headers = [f"{token_set.name} Connections:"]
            if links_headers is None:
                links_headers = [f"{token_set.name} Links:"]
        
        if feature_types is None:
            feature_types = self.default_feats
        if mask is not None:
            tokens = tensor.nodes[mask]
            if print_cons:
                cons = tensor.connections[mask, :]
                links = tensor.links.sets[token_set]
        else:
            tokens = tensor.nodes
            if print_cons:
                cons = tensor.connections
                links = tensor.links.sets[token_set]
        if label_names:
            names = tensor.names
        else:
            names = None
        
        if self.print_tk_tensor(tokens, types=feature_types, label_values=label_values, names=names, headers=headers): # Only print connctions etc, if there are nodes in tensor
            if print_cons:
                self.print_con_tensor(cons, mask=mask, names=names, headers=cons_headers)
                sems = tensor.links.semantics
                self.print_links_tensor(links, sems, mask=mask, names=names, headers=links_headers)
        
    def print_tk_tensor(self, tensor: torch.Tensor, types = None, label_values = True, names = None, headers = None):
        """
        Print the given tensor of tokens.
        Args:
            tensor (torch.Tensor): The tensor to print.
            types (list): List of features to print. (IE: TF.SET, TF.ID, etc.)
            label_values (bool): Whether to convert feature floats to their enum names. (IE: TF.TYPE == 0.0 -> TYPE(0.0).name)
            names (dict): The names of the tokens. If None, the names will not be printed.
            headers (list): The headers to print, defaults to "Tokens Tensor:" if left as None.
        """
        if headers is None:
            headers = [f"Tokens Tensor:"]
        
        if types is None:
            columns = [TF.name for TF in TF]
        else:
            columns = [TF.name for TF in types]
        token_tensor = tensor[:,types]
        rows = token_tensor.tolist()
        if len(rows) == 0:
            printer = tablePrinter(columns, rows, headers, self.file_path, self.print_to_console)
            printer.print_header()
            print("NO TOKENS FOUND")
            return False
        if label_values:
            rows, success = self.label_values(rows, types, names)
            if names is not None:
                if success:
                    columns.insert(1, "Name")
        printer = tablePrinter(columns, rows, headers, self.file_path, self.print_to_console)
        printer.print_table()
        return True
    
    def print_con_tensor(self, tensor: torch.Tensor, mask = None, names = None, headers = None):
        """
        Print the given connections tensor.
        Args:
            tensor (torch.Tensor): The tensor to print.
            mask (torch.Tensor): The mask to apply to the tensor.
            names (dict): The names of the tokens. If None, the names will not be printed.
            headers (list): The headers to print, defaults to "Connections:" if left as None.
        """
        if names is not None:
            pass
        if headers is None:
            headers = ["Connections:"]
        if mask is not None:
            pass

        rows = tensor.tolist()
        for r, row in enumerate(rows):
            for i in range(len(row)):
                if row[i] == 1:
                    row[i] = "x"
                else:
                    row[i] = " "
            try:
                row.insert(0, names[r])
            except:
                row.insert(0, r)
        
        columns = ["P-> C:"]
        if len(rows)>0:
            for i in range(len(rows[0])-1):
                try:
                    columns.append(names[i])
                except:
                    columns.append(i)
        else:
            rows = [["Empty"]]
        
        printer = tablePrinter(columns, rows, headers, self.file_path, self.print_to_console)
        printer.print_table(split=True)
    
    def print_weight_tensor(self, tensor: torch.Tensor, mask = None, names = None, headers = None):
        """
        Print the given weight tensor.
        Args:
            tensor (torch.Tensor): The tensor to print.
            mask (torch.Tensor): The mask to apply to the tensor.
            names (dict): The names of the tokens. If None, the names will not be printed.
            headers (list): The headers to print, defaults to "Weight:" if left as None.
        """
        if names is not None:
            pass
        if headers is None:
            headers = ["Weight:"]
        if mask is not None:
            pass

        rows = tensor.tolist()
        for r, row in enumerate(rows):
            for i in range(len(row)):
                row[i] = f"{row[i]:.2f}"
            try:
                row.insert(0, names[r])
            except:
                row.insert(0, r)
        
        columns = ["P-> C:"]
        if len(rows)>0:
            for i in range(len(rows[0])-1):
                try:
                    columns.append(names[i])
                except:
                    columns.append(i)
        else:
            rows = [["Empty"]]
        
        printer = tablePrinter(columns, rows, headers, self.file_path, self.print_to_console)
        printer.print_table(split=True)
    
    def print_links_tensor(self, tensor: torch.Tensor, semantics, mask = None, names = None, headers = None):
        """
        Print the given links tensor.
        Args:
            tensor (torch.Tensor): The tensor to print.
            mask (torch.Tensor): The mask to apply to the tensor.
            names (dict): The names of the tokens. If None, the names will not be printed.
            headers (list): The headers to print, defaults to "Links:" if left as None.
        """
        sem_names = None
        if names is not None:
            sem_names = semantics.names
        if headers is None:
            headers = ["Links:"]
        
        if mask is not None:
            pass

        rows = tensor.tolist()
        for r, row in enumerate(rows):
            for i in range(len(row)):
                if row[i] == 1:
                    row[i] = "x"
                else:
                    row[i] = " "
            try:
                row.insert(0, names[r])
            except:
                row.insert(0, r)
        
        columns = ["P-> C:"]
        if len(rows)>0:
            for i in range(len(rows[0])-1):
                try:
                    columns.append(sem_names[i])
                except:
                    columns.append(i)
        else:
            rows=[["Empty"]]

        printer = tablePrinter(columns, rows, headers, self.file_path, self.print_to_console)
        printer.print_table(split=True)

    def print_tokens(self, set: Set, token_ids: list[int], types: list[TF]):
        """
        Print the given tokens.
        Args:
            set (Set): The set to print.
            token_ids (list): The ids of the tokens to print.
            types (list): The types of the features to print. (IE: TF.SET, TF.ID, etc.)
        """
        columns = [str(type.name) for type in types]
        tokens = self.nodes.sets[set].nodes
        token_tensors = tokens[token_ids, :]
        token_tensors = token_tensors[:,types]
        rows = token_tensors.tolist()
        if len(rows) == 0:
            print("No tokens found")
            return
        
        headers = [f"Set:{set.name}"] + ["Tokens:"] +[f"{token_ids}"]
        printer = tablePrinter(columns, rows, headers, self.file_path, self.print_to_console)
        printer.print_table()
    
    def label_values(self, row:list[float], types:list[TF], names:dict[int, str]):
        """
        Label the values and names of the given row.
        Args:
            row (list): The row to label.
            types (list): The types of the features to label. (IE: TF.SET, TF.ID, etc.)
            names (dict): The names of the tokens. If None, the names will not be added to the row.
        """
        labels =  {
            # Labeled values:
            TF.TYPE: Type,
            TF.SET: Set,
            TF.MODE: Mode,
            #  Bool values:
            TF.INFERRED: B,
            TF.RETRIEVED: B,
            TF.COPY_FOR_DR: B,
            TF.COPIED_DR_INDEX: B,
            TF.SIM_MADE: B,
            TF.DELETED: B,
            TF.PRED: B,
        }
        
        # Label values
        for t, type in enumerate(types):
            for i, value in enumerate(row):
                if value[t] == null:
                    row[i][t] = "Null"
                else:
                    try:
                        row[i][t] = labels[type](value[t]).name
                    except KeyError:
                        pass
                success = True
    
        # Add name column to row
        if names is not None:
            try:
                if types[0] == TF.ID:
                    for i in range(len(row)):
                        try:
                            row[i].insert(1, names[row[i][0]])
                        except:
                            row[i].insert(1, row[i][0])
            except:
                success = False

        return row, success
