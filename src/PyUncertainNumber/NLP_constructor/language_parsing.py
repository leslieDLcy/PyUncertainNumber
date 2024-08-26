import re
import math
from decimal import Decimal
from PyUncertainNumber.pba.interval import PM
from PyUncertainNumber.pba.interval import Interval as I
import numpy as np


def is_number(n):
    """check if a string is a number"""

    try:
        float(n)  # Type-casting the string to `float`.
        # If string is not a valid `float`,
        # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True


def count_sigfigs(numstr: str) -> int:
    """Count the number of significant figures in a number string"""

    return len(Decimal(numstr).as_tuple().digits)


def count_sig_digits_bias(number):
    """to count the bias for the getting the significant digits after the decimal point

    note:
        - to exclude the sig digits before the decimal point
    """
    a, b = math.modf(number)
    return len(str(int(b)))


def findWholeWord(w):
    """Find a whole word in a string

    note:
        - this returns the matched word, but not directly a boolean
    """

    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search


def whole_word_detect(word, string):
    """Detect if a whole word is in a string, return y or n"""

    if word in string.split():
        print("success")
    else:
        print("Not found")


def hedge_interpret(text):
    """interpret linguistic hedge words into UncertainNumber objects"""

    splitted_list = text.split()

    # parse the numeric value denoted as x
    x = [s for s in splitted_list if is_number(s)][0]

    # decide if the number is a float or an integer
    if "." in x:
        x = float(x)
    else:
        x = int(x)

    # parse the decimal place 'd'
    d = count_sigfigs(str(x))
    bias_num = count_sig_digits_bias(x)
    d = d - bias_num

    # parse the keyword
    try:
        kwd = [s for s in splitted_list if not is_number(s)]
        kwd = " ".join(kwd)
    except:
        kwd = ""

    match kwd:
        case "exactly":
            return PM(x, 10 ** (-(d + 1)))
        case "":
            return PM(x, 0.5 * 10 ** (-d))
        case "about":
            return PM(x, 2 * 10 ** (-d))
        case "around":
            return PM(x, 10 * 10 ** (-d))
        case "count":
            return PM(x, np.sqrt(np.abs(x)))
        case "almost":
            return I(x - 0.5 * (10 ** (-d)), x)
        case "over":
            return I(x, x + 0.5 * (10 ** (-d)))
        case "below":
            return I(x - 2 * (10 ** (-d)), x)
        case "above":
            return I(x, x + 2 * (10 ** (-d)))
        case "at most":
            return I(-np.inf, x)
        # TODO conditonal based on unit and common sense....
        # TODO optional negative or not
        case "at least":
            return I(x, np.inf)
        case "order":
            return I(x / 2, 5 * x)
        case "between":
            return f"why not directly use an interval object?"
        case _:
            return "not a hedge word"
