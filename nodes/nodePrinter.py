from nodes import *
from nodeEnums import *
import os

class nodePrinter(object):
    def __init__(self, nodes: Nodes, print_to_console: bool = True, log_file: str = None):
        self.nodes = nodes
        self.print_to_console = print_to_console
        self.log_file = log_file

    def print_token_tensor(self, set: Set, types = None, mask = None, label_values = True, label_names = True, headers = None, print_cons = True, cons_headers = None, links_headers = None):
        tensor = self.nodes.sets[set]
        if headers is None:
            headers = [f"{set.name} Tokens:"]
            if cons_headers is None:
                cons_headers = [f"{set.name} Connections:"]
            if links_headers is None:
                links_headers = [f"{set.name} Links:"]
        
        if types is None:
            types = [TF.ID, TF.TYPE, TF.SET, TF.ANALOG, TF.PRED]
        if mask is not None:
            tokens = tensor.nodes[mask]
            if print_cons:
                cons = tensor.connections[mask, :]
                links = tensor.links.sets[set]
        else:
            tokens = tensor.nodes
            if print_cons:
                cons = tensor.connections
                links = tensor.links.sets[set]
        if label_names:
            names = tensor.names
        else:
            names = None
        
        self.print_tk_tensor(tokens, types=types, label_values=label_values, names=names, headers=headers)
        if print_cons:
            self.print_con_tensor(cons, mask=mask, names=names, headers=cons_headers)
            self.print_links_tensor(links, mask=mask, names=names, headers=links_headers)
        
        

    def print_tk_tensor(self, tensor: torch.Tensor, types = None, label_values = True, names = None, headers = None):
        if headers is None:
            headers = [f"Tokens Tensor:"]
        
        if types is None:
            columns = [TF.name for TF in TF]
        else:
            columns = [TF.name for TF in types]
        token_tensor = tensor[:,types]
        rows = token_tensor.tolist()
        if len(rows) == 0:
            print("No tokens found")
            return
        if label_values:
            rows, success = self.label_values(rows, types, names)
            if names is not None:
                if success:
                    columns.insert(1, "Name")
        printer = tablePrinter(columns, rows, headers, self.log_file, self.print_to_console)
        printer.print_table()
    
    def print_con_tensor(self, tensor: torch.Tensor, mask = None, names = None, headers = None):
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
            if names is not None:
                row.insert(0, names[r])
            else:
                row.insert(0, r)
        
        columns = ["P-> C:"]
        for i in range(len(rows[0])-1):
            if names is not None:
                columns.append(names[i])
            else:
                columns.append(i)
        
        printer = tablePrinter(columns, rows, headers, self.log_file, self.print_to_console)
        printer.print_table(split=True)
    
    def print_links_tensor(self, tensor: torch.Tensor, mask = None, names = None, headers = None):
        sem_names = None
        if names is not None:
            sem_names = self.nodes.semantics.names
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
            if names is not None:
                row.insert(0, names[r])
            else:
                row.insert(0, r)
        
        columns = ["P-> C:"]
        for i in range(len(rows[0])-1):
            if names is not None:
                columns.append(sem_names[i])
            else:
                columns.append(i)

        printer = tablePrinter(columns, rows, headers, self.log_file, self.print_to_console)
        printer.print_table(split=True)
        
            

    def print_tokens(self, set: Set, token_ids: list[int], types: list[TF]):
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
    TOP = 0
    MIDDLE = 1
    BOTTOM = 2
    SPLIT = 3

class C(IntEnum):
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
    def __init__(self, columns: list[str], rows: list[list[str]], headers: list[str], log_file: str = None, print_to_console: bool = True):
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
        for i, value in enumerate(data):
            if isinstance(value, float):
                data[i] = f"{value:.{self.dp}f}"
            else:
                data[i] = str(value)
        return data

    
    def format_rows(self, rows: list[list[str]]):
        for i, row in enumerate(rows):
            rows[i] = self.format(row)
        return rows

    def print_header(self, char_set= "header"):
        if self.header_widths is None:
            self.calc_header_width()
        self.get_line(lineTypes.TOP, char_set, self.header_widths)
        self.get_line(lineTypes.MIDDLE, char_set, self.header_widths, format_data=self.headers)
        self.get_line(lineTypes.BOTTOM, char_set, self.header_widths)
    
    def print_column_names(self, split= True, char_set="table"):
        if self.col_widths is None:
            self.calc_col_widths()
        self.get_line(lineTypes.TOP, char_set, self.col_widths)
        self.get_line(lineTypes.MIDDLE, char_set, self.col_widths, format_data=self.columns)
        if split:
            self.get_line(lineTypes.SPLIT, char_set, self.col_widths)
        else:
            self.get_line(lineTypes.BOTTOM, char_set, self.col_widths)
    
    def print_rows(self, print_top=False, print_bottom=True, char_set="table", split=False):
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
        if self.col_widths is None:
            self.calc_col_widths()
        if header:
            self.print_header(char_set=header_char_set)
        if column_names:
            self.print_column_names(char_set=column_char_set)
        self.print_rows(char_set=row_char_set, split=split, print_top=not(column_names))

    def calc_col_widths(self):
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
        col_len = len(self.columns)
        for row in self.rows:
            if len(row) != col_len:
                for r in self.rows:
                    print(r, len(r))
                print(self.columns, len(self.columns))
                raise ValueError(f"Row/Column length mismatch row_len:{len(row)} != col_len:{col_len} at row {row, enumerate(row)}")
    
    def calc_header_width(self):
        self.header_widths = [0] * len(self.headers)
        for i, header in enumerate(self.headers):
            self.header_widths[i] = len(header) + self.padding

    def get_line(self, line_type: lineTypes, char_set: str, widths, format_data = None):
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
        leftc, fillc, splitc, rightc = self.get_line_chars(line_type, char_set)
        left = self.get_col_string(leftc, fillc, widths[0], format_data[0])
        col_strings = []
        for col in range(1, len(widths)):
            col_strings.append(self.get_col_string(splitc, fillc, widths[col], format_data[col]))
        return left + "".join(col_strings) + rightc

    def get_line_no_data(self, line_type: lineTypes, char_set: str, widths):
        leftc, fillc, splitc, rightc = self.get_line_chars(line_type, char_set)
        left = self.get_col_string(leftc, fillc, widths[0])
        col_strings = []
        for col in range(1, len(widths)):
            col_strings.append(self.get_col_string(splitc, fillc, widths[col]))
        return left + "".join(col_strings) + rightc
     
    def get_col_string(self, startc, fillc, width, format_data = None):
        if format_data is None:
            string = startc + (fillc * width)                  # No data, so just print fill chars
        else:
            string = startc + f"{format_data:^{width}}"        # Center the data, with the column width
        return string

    def get_line_chars(self, line_type: lineTypes, char_set: str):
        chars = self.chars[char_set]
        char_list = self.line_chars[line_type]
        leftc = chars[char_list[0]]
        rightc = chars[char_list[1]]
        fillc = chars[char_list[2]]
        splitc = chars[char_list[3]]
        return leftc, fillc, splitc, rightc

    def open_file(self, filename):
        if os.path.exists(filename):
            f = open(filename, "a", encoding='utf-8')
        else:
            f = open(filename, "x", encoding='utf-8')
        return f
    
