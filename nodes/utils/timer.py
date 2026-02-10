from time import monotonic
from logging import getLogger
from .printer.print_table import TablePrinter
from math import log10
logger = getLogger("TIMER")

class Timer:
    def __init__(self):
        self.timers: list[float] = []
        self.comments: list[str] = []
    
    def start(self, log=True):
        """ 
        Start timer

        Args:
            log (bool, optional): Whether to log the start. Defaults to True.
        """
        time = monotonic()
        self.timers.append([time])
        if log:
            logger.debug(f"Timer started")
    
    def click(self, comment:str = "",log=True):
        """ 
        Add new interval to last timer. Returns last interval.
        
        Args:
            log (bool, optional): Whether to log the click. Defaults to True.
        
        Returns:
            float: The last interval.
        """
        time = monotonic()
        current_timer = self.timers[-1]
        current_timer.append(time)
        time_interval = current_timer[-1] - current_timer[-2]
        log_comment = f"Time interval"
        if comment != "":
            self.comments.append(comment)
            log_comment += f"{comment}"
        if log:
            logger.debug(f"{log_comment}: {time_interval}")
        return time_interval

    def stop(self, comment:str = "", log=True):
        """ Add last interval, and return all intervals.
        
        Args:
            log (bool, optional): Whether to log the stop. Defaults to True.
        
        Returns:
            list[float]: The intervals since start.
        """
        time = monotonic()
        if self.timers == []:
            logger.warning("No timers to stop")
            return []
        times = self.timers[-1]
        times.append(time)
        if len(times) < 2:
            logger.warning("Not enough intervals to stop")
            return []
        self.comments.append(comment)
        intervals = []
        for i in range(len(times) - 1):
            intervals.append(times[i+1] - times[i])
        if log:
            logger.debug(f"Intervals: {intervals}")
        return intervals
    
    def get_comments(self):
        """ Get comments for all timers. """
        return self.comments
    
    def init(self):
        """ clear all timers and comments. """
        self.timers = []
        self.comments = []
    
    def get_times(self, log=False):
        """ Get raw times from all timers.
    
        Args:
            log (bool, optional): Whether to log the get times. Defaults to False.
        
        Returns:
            list[list[float]]: The raw times from all timers.
        """
        if log:
            logger.debug(f"Times: {self.timers}")
        return self.timers
    
    def get_intervals(self, log=False, combine=True):
        """ Return intervals between clicks for all timers.
        
        Args:
            log (bool, optional): Whether to log the get intervals. Defaults to False.
        
        Returns:
            list[list[float]]: The intervals between clicks for all timers.
        """
        output = []
        for times in self.timers:
            if combine:
                for i in range(len(times) - 1):
                    output.append(times[i+1] - times[i])
            else:
                intervals = []
                for i in range(len(times) - 1):
                    intervals.append(times[i+1] - times[i])
                output.append(intervals)
        if log:
            logger.debug(f"Intervals: {output}")
        return output

def intervals_table(timers: list[list[float]], comments: list[str]=None):
    """ Print a table of intervals. """
    rows = []
    timer_len = None
    max_interval = 0
    min_interval = float('inf')
    # Check that all timers have the same number of intervals, and get mix/max intervals lengths
    for row in timers:
        max_interval = max(max_interval, max(row))
        min_interval = min(min_interval, min(row))
        if timer_len is None:
            timer_len = len(row)
        elif timer_len != len(row):
            logger.error(f"Number of intervals mismatch: {timer_len} != {len(row)}, will remove extra intervals")
    # Decide on formating
    round_power = True
    factor = 100 / max_interval
    power = -int(log10(factor))
    tags = {
        0: "s",
        -3: "ms",
        -6: "μs",
        -9: "ns",
    }
    closest = min(tags.keys(), key=lambda x: abs(x - power))
    diff = power - closest
    if round_power:
        power = closest
        diff = 0
        factor = 10**(-power)
    tag = tags[closest]
    if diff != 0:
        tag = f"{10**diff} {tag}"
    # Format rows
    for i, row in enumerate(timers):
        formatted_row = [i]
        for int_idx, interval in enumerate(row):
            factored = interval * factor
            formatted_row.append(f"{factored:.2f}")
        rows.append(formatted_row)
    # Add comparison rows if two timers
    if len(timers) == 2:
        diff_row = ["diff"]
        factor_row = ["factor"]
        for i in range(timer_len):
            min_iv = min(timers[0][i], timers[1][i])
            max_iv = max(timers[0][i], timers[1][i])
            diff_row.append(f"{(max_iv - min_iv)*factor:.2f}")
            factor_row.append(f"{max_iv / min_iv:.2f}")
        rows.append(diff_row)
        rows.append(factor_row)
    # Format columns
    columns = ["Timer"]
    if comments is not None:
        if len(comments) == timer_len:
            columns.extend(comments)
        else:
            logger.error(f"Number of comments mismatch: {len(comments)} != {len(timers)}, will use empty comments")
            comments = None
    if comments is None:
        columns.extend([f"{i}" for i in range(timer_len)])
    # Print table
    headers=[f"Timer intervals"]
    if power != 0:
        headers.append(f"10e{power}")
        headers.append(f"{tag}")
    else:
        headers.append("s")
    
    table = TablePrinter(
        columns=columns,
        rows=rows,
        headers=headers
    )
    table.print_table()