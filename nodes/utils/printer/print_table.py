# nodes/utils/printer/print_table.py
# Provides a class for printing tables.

from enum import IntEnum
import os

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
    
