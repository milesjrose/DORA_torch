from nodes import *


class nodePrinter(object):
    def __init__(self, nodes: Nodes):
        self.nodes = nodes
    
    def print_tk_tensor(self, tensor: torch.Tensor, types = None):
        headers = f"{tensor.shape[0]}x{tensor.shape[1]} Tensor:"
        if types is None:
            columns = [TF.name for TF in TF]
        else:
            columns = [TF.name for TF in types]
        token_tensor = tensor[:,types]
        rows = token_tensor.tolist()
        if len(rows) == 0:
            print("No tokens found")
            return
        headers = ["Tokens:"]
        printer = tablePrinter(columns, rows, headers)
        printer.print_table()
    
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
        printer = tablePrinter(columns, rows, headers)
        printer.print_table()

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
    def __init__(self, columns: list[str], rows: list[list[str]], headers: list[str]):
        self.columns = columns
        self.rows = rows
        self.headers = headers
        self.col_widths = None
        self.header_widths = None
        self.dp = 2
        self.padding = 4

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

    def print_header(self, char_set= "header"):
        if self.header_widths is None:
            self.calc_header_width()
        print(self.get_line(lineTypes.TOP, char_set, self.header_widths))
        print(self.get_line(lineTypes.MIDDLE, char_set, self.header_widths, format_data=self.headers))
        print(self.get_line(lineTypes.BOTTOM, char_set, self.header_widths))
    
    def print_column_names(self, split= True, char_set="table"):
        if self.col_widths is None:
            self.calc_col_widths()
        print(self.get_line(lineTypes.TOP, char_set, self.col_widths))
        print(self.get_line(lineTypes.MIDDLE, char_set, self.col_widths, format_data=self.columns))
        if split:
            print(self.get_line(lineTypes.SPLIT, char_set, self.col_widths))
        else:
            print(self.get_line(lineTypes.BOTTOM, char_set, self.col_widths))
    
    def print_rows(self, print_top=False, print_bottom=True, char_set="table"):
        if self.col_widths is None:
            self.calc_col_widths()
        if print_top:
            print(self.get_line(lineTypes.TOP   , char_set, self.col_widths))
        for row in self.rows:
            print(self.get_line(lineTypes.MIDDLE, char_set, self.col_widths, format_data=row))
        if print_bottom:
            print(self.get_line(lineTypes.BOTTOM, char_set, self.col_widths))
    
    def print_table(self, header=True, column_names=True, header_char_set="header", column_char_set="table", row_char_set="table"):
        if self.col_widths is None:
            self.calc_col_widths()
        if header:
            self.print_header(char_set=header_char_set)
        if column_names:
            self.print_column_names(char_set=column_char_set)
            self.print_rows(char_set=row_char_set)
        else:
            self.print_rows(char_set=row_char_set)

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
                raise ValueError("All rows must have the same number of columns")
    
    def calc_header_width(self):
        self.header_widths = [0] * len(self.headers)
        for i, header in enumerate(self.headers):
            self.header_widths[i] = len(header) + self.padding

    def get_line(self, line_type: lineTypes, char_set: str, widths, format_data = None):
        if format_data is None:
            return self.get_line_no_data(line_type, char_set, widths)
        else:
            return self.get_line_with_data(line_type, char_set, widths, format_data)
    
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

