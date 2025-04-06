from .nodeEnums import *
import os
import torch

class nodePrinter(object):
    """
    This class is used to print the nodes and their tensors to the console or a file.
    
    Attributes:
        print_to_console (bool): Whether to print to the console.
        log_file (str): The file to print to.
        default_feats (list): Types to print from tokens by default
    """
    def __init__(self,print_to_console: bool = True, log_file: str = None):
        """
        Initialize the node printer.

        Args:
            print_to_console (bool): Whether to print to the console.
            log_file (str): The file to print to.
        """
        self.default_feats = [TF.ID, TF.TYPE, TF.SET, TF.ANALOG, TF.PRED, TF.DELETED]
        self.print_to_console = print_to_console
        self.log_file = log_file

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
            printer = tablePrinter(columns, rows, headers, self.log_file, self.print_to_console)
            printer.print_header()
            print("NO TOKENS FOUND")
            return False
        if label_values:
            rows, success = self.label_values(rows, types, names)
            if names is not None:
                if success:
                    columns.insert(1, "Name")
        printer = tablePrinter(columns, rows, headers, self.log_file, self.print_to_console)
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
        
        printer = tablePrinter(columns, rows, headers, self.log_file, self.print_to_console)
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

        printer = tablePrinter(columns, rows, headers, self.log_file, self.print_to_console)
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
        printer = tablePrinter(columns, rows, headers, self.log_file, self.print_to_console)
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
                        row[i].insert(1, names[row[i][0]])
            except:
                success = False

        return row, success

# ====================[ PRINTER ENUMS ]=====================
class lineTypes(IntEnum):
    """
    Enum for the type of line to print.
    """
    TOP = 0
    MIDDLE = 1
    BOTTOM = 2
    SPLIT = 3

class C(IntEnum):
    """
    Enum for the characters to print.
    """
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3
    HORIZONTAL = 4
    VERTICAL = 5
    CROSS = 6
    HORIZONTAL_DOWN = 7
    HORIZONTAL_UP = 8
    VERTICAL_LEFT = 9
    VERTICAL_RIGHT = 10
# =================[ TABLE PRINTER CLASS ]==================
class tablePrinter(object):
    """
    Print a table of data.

    Attributes:
        columns (list): The columns of the table.
        rows (list): The rows of the table.
        headers (list): The headers of the table.
        log_file (str): The file to log to. Only logs if provided.
        print_to_console (bool): Whether to print to the console.
    """
    def __init__(self, columns: list[str], rows: list[list[str]], headers: list[str], log_file: str = None, print_to_console: bool = True):
        """
        Initialize the table printer. Must either set log_file or print_to_console, or both.
        Args:
            columns (list): The columns of the table.
            rows (list): The rows of the table.
            headers (list): The headers of the table.
            log_file (str): The file to log to. Only logs if provided.
            print_to_console (bool): Whether to print to the console.
        """
        self.col_widths = None
        self.header_widths = None
        self.dp = 2
        self.padding = 4
        self.columns = self.format(columns)
        self.rows = self.format_rows(rows)
        self.headers = self.format(headers)
        self.log_file = log_file
        self.print_to_console = print_to_console
        self.chars = {
            "header" : ["╒","╕","╘","╛","═","│","╪", "╤", "╧", "╣", "╠"],
            "table"  : ["┌","┐","└","┘","─","│","┼", "┬", "┴", "┤", "├"]
        }

        self.line_chars = {
            # LINETYPE : [LEFTC, RIGHTC, FILLC, SPLITC]
            lineTypes.TOP: [C.TOP_LEFT, C.TOP_RIGHT, C.HORIZONTAL, C.HORIZONTAL_DOWN],
            lineTypes.MIDDLE: [C.VERTICAL, C.VERTICAL, C.HORIZONTAL, C.VERTICAL],
            lineTypes.BOTTOM: [C.BOTTOM_LEFT, C.BOTTOM_RIGHT, C.HORIZONTAL, C.HORIZONTAL_UP],
            lineTypes.SPLIT: [C.VERTICAL_RIGHT, C.VERTICAL_LEFT, C.HORIZONTAL, C.CROSS],
        }
    
    def format(self, data: list[str]):
        """
        Format the given data into strings.
        Args:
            data (list): The data to format.
        """
        for i, value in enumerate(data):
            if isinstance(value, float):
                data[i] = f"{value:.{self.dp}f}"
            else:
                data[i] = str(value)
        return data
    
    def format_rows(self, rows: list[list[str]]):
        """
        Format the given rows data into strings.
        Args:
            rows (list): The rows to format.
        """
        for i, row in enumerate(rows):
            rows[i] = self.format(row)
        return rows

    def print_header(self, char_set= "header"):
        """
        Print the header.
        Args:
            char_set (str): The character set to use.
        """
        if self.header_widths is None:
            self.calc_header_width()
        self.get_line(lineTypes.TOP, char_set, self.header_widths)
        self.get_line(lineTypes.MIDDLE, char_set, self.header_widths, format_data=self.headers)
        self.get_line(lineTypes.BOTTOM, char_set, self.header_widths)
    
    def print_column_names(self, split= True, char_set="table"):
        """
        Print the column names.
        Args:
            split (bool): Whether to add a split line after column names, otherwise add a bottom line. Default is True.
            char_set (str): The character set to use.
        """
        if self.col_widths is None:
            self.calc_col_widths()
        self.get_line(lineTypes.TOP, char_set, self.col_widths)
        self.get_line(lineTypes.MIDDLE, char_set, self.col_widths, format_data=self.columns)
        if split:
            self.get_line(lineTypes.SPLIT, char_set, self.col_widths)
        else:
            self.get_line(lineTypes.BOTTOM, char_set, self.col_widths)
    
    def print_rows(self, print_top=False, print_bottom=True, char_set="table", split=False):
        """
        Print all rows of data in the table.
        Args:
            print_top (bool): Whether to print a top line. Default is False.
            print_bottom (bool): Whether to print a bottom line. Default is True.
            char_set (str): The character set to use.
            split (bool): Whether to add a split line after each row. Default is False.
        """
        if self.col_widths is None:
            self.calc_col_widths()
        if print_top:
            self.get_line(lineTypes.TOP   , char_set, self.col_widths)
        for i, row in enumerate(self.rows):
            self.get_line(lineTypes.MIDDLE, char_set, self.col_widths, format_data=row)
            if split and i != len(self.rows) - 1:
                self.get_line(lineTypes.SPLIT, char_set, self.col_widths)
        if print_bottom:
            self.get_line(lineTypes.BOTTOM, char_set, self.col_widths)
    
    def print_table(self, header=True, column_names=True, header_char_set="header", column_char_set="table", row_char_set="table", split=False):
        """
        Print the header, column names, and rows of data in the table.
        Args:
            header (bool): Whether to print the header. Default is True.
            column_names (bool): Whether to print the column names. Default is True.
            header_char_set (str): The character set to use for the header. Default is "header".
            column_char_set (str): The character set to use for the column names. Default is "table".
            row_char_set (str): The character set to use for the rows. Default is "table".
            split (bool): Whether to add a split line after each row. Default is False.
        """
        if self.col_widths is None:
            self.calc_col_widths()
        if header:
            self.print_header(char_set=header_char_set)
        if column_names:
            self.print_column_names(char_set=column_char_set)
        self.print_rows(char_set=row_char_set, split=split, print_top=not(column_names))

    def calc_col_widths(self):
        """
        Calculate the widths of the columns, based on longest content in each column.
        """
        self.col_widths = [0] * len(self.columns)
        self.check_row_column_lengths()
        for i, column in enumerate(self.columns):
            self.col_widths[i] = len(column)                # Get width of column name
            for row in self.rows:                           # Check each row's
                content = row[i]                            # content of column i
                if isinstance(content, float):              # If float, format to dp
                    content_str = f"{content:.{self.dp}f}"
                else:
                    content_str = str(content)              # If not float, convert to string
                content_width = len(content_str)            # Get width of content
                if content_width > self.col_widths[i]:      # If content is wider than column, update column width
                    self.col_widths[i] = content_width
            self.col_widths[i] += self.padding              # Add padding

    def check_row_column_lengths(self):
        """
        Check that the number of columns in each row matches the number of columns in the table.
        """
        col_len = len(self.columns)
        for row in self.rows:
            if len(row) != col_len:
                for r in self.rows:
                    print(r, len(r))
                print(self.columns, len(self.columns))
                raise ValueError(f"Row/Column length mismatch row_len:{len(row)} != col_len:{col_len} at row {row, enumerate(row)}")
    
    def calc_header_width(self):
        """
        Calculate the widths of the header strings.
        """
        self.header_widths = [0] * len(self.headers)
        for i, header in enumerate(self.headers):
            self.header_widths[i] = len(header) + self.padding

    def get_line(self, line_type: lineTypes, char_set: str, widths, format_data = None):
        """
        Get the line of the given type. 
        If self.print_to_console is True, print the line to the console. 
        If self.log_file is not None, write the line to the file.
        
        Args:
            line_type (lineTypes): The type of line to get.
            char_set (str): The character set to use.
            widths (list): The widths of the columns.
            format_data (list): The data to format.
        """
        if format_data is None:
            line = self.get_line_no_data(line_type, char_set, widths)
        else:
            line = self.get_line_with_data(line_type, char_set, widths, format_data)
        if self.log_file is not None:
            with self.open_file(self.log_file) as f:
                f.write(str(line) + "\n")
        if self.print_to_console:
            print(line)
    
    def get_line_with_data(self, line_type: lineTypes, char_set: str, widths, format_data):
        """
        Return a string for the given row in the table.
        Args:
            line_type (lineTypes): The type of line to get.
            char_set (str): The character set to use.
            widths (list): The widths of the columns.
            format_data (list): The data to format.
        
        Returns:
            str: The line of the given type, with data.
        """
        leftc, fillc, splitc, rightc = self.get_line_chars(line_type, char_set)
        left = self.get_col_string(leftc, fillc, widths[0], format_data[0])
        col_strings = []
        for col in range(1, len(widths)):
            col_strings.append(self.get_col_string(splitc, fillc, widths[col], format_data[col]))
        return left + "".join(col_strings) + rightc

    def get_line_no_data(self, line_type: lineTypes, char_set: str, widths):
        """
        Return a string for the given line type.
        Args:
            line_type (lineTypes): The type of line to get.
            char_set (str): The character set to use.
            widths (list): The widths of the columns.
        
        Returns:
            str: Line of the given type.
        """
        leftc, fillc, splitc, rightc = self.get_line_chars(line_type, char_set)
        left = self.get_col_string(leftc, fillc, widths[0])
        col_strings = []
        for col in range(1, len(widths)):
            col_strings.append(self.get_col_string(splitc, fillc, widths[col]))
        return left + "".join(col_strings) + rightc
     
    def get_col_string(self, startc, fillc, width, format_data = None):
        """
        Return a string for the given column.
        Args:
            startc (str): The character to start the column with.
            fillc (str): The character to fill the column with.
            width (int): The width of the column.
            format_data (str): The data to format.
        
        Returns:
            str: String for the given column.
        """
        if format_data is None:
            string = startc + (fillc * width)                  # No data, so just print fill chars
        else:
            string = startc + f"{format_data:^{width}}"        # Center the data, with the column width
        return string

    def get_line_chars(self, line_type: lineTypes, char_set: str):
        """
        Get the characters for the given line type and character set.
        """
        chars = self.chars[char_set]
        char_list = self.line_chars[line_type]
        leftc = chars[char_list[0]]
        rightc = chars[char_list[1]]
        fillc = chars[char_list[2]]
        splitc = chars[char_list[3]]
        return leftc, fillc, splitc, rightc

    def open_file(self, filename):
        """
        Open the given file.
        If the file exists, open it in append mode.
        If the file does not exist, create it and open it in write mode.

        Args:
            filename (str): The name of the file to open.
            
        Returns:
            file: The file object.
        """
        if os.path.exists(filename):
            f = open(filename, "a", encoding='utf-8')
        else:
            f = open(filename, "x", encoding='utf-8')
        return f
    
