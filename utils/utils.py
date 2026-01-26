"""Simple litte helpers
"""

import numpy as np
import pandas as pd
import re

from typing import Dict, Any, List
from datetime import datetime



def np_timedelta64_to_seconds(td: np.timedelta64) -> int:
    """
    Converts an arbitrary timedelta64 to seconds

    Args:
        td (np.timedelta64): Timedelta object.

    Returns:
        int: Timedelta object in seconds.

    """
    td = np.timedelta64(td, "s")
    return td.astype("int")


def np_timedelta64_to_milli_seconds(td: np.timedelta64) -> int:
    """Converts an arbitrary timedelta64 to milliseconds

    Args:
        td (np.timedelta64): Timedelta object.

    Returns:
        int: Timedelta object in milliseconds.

    """

    # make sure timedelta is based on milliseconds
    td = np.timedelta64(td, "ms")
    return td.astype("int")


def np_timedelta64_to_days(td: np.timedelta64) -> int:
    """Converts an arbitrary timedelta64 to days

    Args:
        td (np.timedelta64): Timedelta object.

    Returns:
        int: Timedelta object in days.

    """
    # make sure timedelta is based on seconds
    td = np.timedelta64(td, "s")
    secs = td.astype("int")
    return secs / 60 / 60 / 24


def np_datetime64_to_sql_string(dt: np.datetime64) -> str:
    """
    Convert timedelta object to string to be used in an sql query

    Args:
        dt (np.datetime64): Timedelta object.

    Returns:
        str: Str to be used in sql query.

    """
    return f"{dt}".replace("T", " ")


def np_datetime64_to_file_name(dt: np.datetime64) -> str:
    """
    Convert timedelta object to filename

    Args:
        dt (np.datetime64): Timedelta object.

    Returns:
        str: Filename.

    """
    # make sure dt is based on seconds
    dt = np.datetime64(dt, "s")
    return dt.astype(str).replace("-", "_").replace("T", "__").replace(":", "_")


def np_datetime64_to_header_date_string(dt: np.datetime64) -> str:
    """
    Convert timedelta to header date string (to ms and replace T by space)

    Args:
        dt (np.datetime64): Timedelta object.

    Returns:
        str: Header data string.

    """
    dt = np.datetime64(dt, "ms")
    return f"{dt}".replace("T", " ")


def seconds_to_readable(seconds: int):
    """
    Helper function to convert seconds to a readable timestamp

    Args:
        seconds (int): Seconds.

    Returns:
        str: Readable timestamp string.

    """

    h = seconds // 3600
    seconds %= 3600
    m = seconds // 60
    s = seconds % 60
    # return f'{h:02}:{m:02}:{s:02}'
    return f"{h}h {m:02}m {s:02}s"



def verbose_print(verbose: bool, msg: Any) -> None:
    """
    Verbose printing

    Args:
        verbose (bool): Verbose flag.
        msg (Any): Message to be printed.

    Returns:
        None

    """
    if verbose:
        print(msg)


def timed_print(msg: Any) -> None:
    """
    Prints out msg together with current time

    Args:
        msg ():

    Returns:

    """
    print(f"{np.datetime64('now')} {msg}")


def nan_to_neg(v: float) -> float:
    """
    Convert nan values to -1

    Args:
        v (float): Value to convert eventually.

    Returns:
        float: Converted value.

    """
    # ass
    if np.isnan(v):
        return -1
    return v


def save_display(df: pd.DataFrame) -> None:
    """
    Display a DataFrame when in a jupyter notebook, else print it.

    Args:
        df (pd.DataFrame): DataFrame.

    Returns:
        None

    """
    try:
        display(df)
    except NameError:
        print(df)


def find_replace_multi_ordered(s: str, dictionary: Dict[str, str]):
    # sort keys by length, in reverse order
    for item in sorted(dictionary.keys(), key=len, reverse=True):
        s = re.sub(item, dictionary[item], s)
    return s


def now_string_secs() -> str:
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def now_string_millisecs() -> str:
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f")


def check_seizure_windows(windows: List[int]) -> None:
    """Checks if given windows list is within the valid range. Throws RuntimeError if not.

    Args:
        windows ():

    Returns:

    """

    if np.min(windows) < -18:
        raise RuntimeError("Minimum window you can use is -18.")

    if np.max(windows) > 0:
        raise RuntimeError("Maximum window you can use is 0.")


def bool_verbose_print(flag: bool, verbose: bool, message: str) -> None:
    """Prints the message only if flag is False and verbose is True

    Args:
        flag ():
        verbose ():
        message ():

    Returns:

    """

    if not flag and verbose:
        print(message)
