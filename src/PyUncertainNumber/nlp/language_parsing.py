from __future__ import annotations
from typing import TYPE_CHECKING
import re
import math
from decimal import Decimal
from ..pba.interval import PM
from ..pba.interval import Interval as I
import numpy as np
from ..characterisation.utils import PlusMinus_parser, parser4, percentage_finder, percentage_converter, initial_list_checking, bad_list_checking
from ..pba.params import Params

if TYPE_CHECKING:
    from ..pba.pbox_base import Pbox


def hedge_interpret(hedge: str, return_type='interval') -> I | Pbox:
    """interpret linguistic hedge words into UncertainNumber objects"""

    assert isinstance(hedge, str), "hedge must be a string"
    splitted_list = hedge.split()

    # parse the numeric value denoted as x
    x = [s for s in splitted_list if is_number(s)][0]

    # decide if the number is a float or an integer
    if "." in x:
        x = float(x)
    else:
        x = int(x)

    # we get the number at this step

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

    if return_type == 'interval':
        # return the interval object
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
    elif return_type == 'pbox':
        pass
    else:
        raise ValueError("return_type must be either 'interval' or 'pbox'")


def parse_interval_expression(expression):
    """Parse the expression to interpret and return an Interval-type Uncertain Number object

        args:
            expression (str): the flexible string desired by Scott to instantiate a Uncertain Number

        caveat:
            the expression needs to have space between the values and the operators, such as '[15 +- 10%]'
        return:
            an Interval object
    """

    ### type 1 ###
    # initial check if string-rep of list
    if initial_list_checking(expression):
        an_int = initial_list_checking(expression)
        if len(an_int) == 1:
            return PM(an_int[0], hw=Params.hw)
        elif len(an_int) > 1:
            return I(*an_int)
    ### type 2 ###
    elif bad_list_checking(expression):
        if PlusMinus_parser(expression) & (not percentage_finder(expression)):
            parsed_list = parser4(expression)
            return PM(*parsed_list)
        elif PlusMinus_parser(expression) & percentage_finder(expression):
            # parse the percentage first
            mid_range = percentage_converter(expression)
            parsed_mid_value = parser4(expression)[0]

            # if we take the percentage literally
            # return PM(parsed_mid_value, hw=mid_range)
            # if we take the percentage based on the context
            return PM(parsed_mid_value, hw=parsed_mid_value * mid_range)
    else:
        return "not a valid expression"

# * ---------------------moduels  --------------------- *#


def decipher_zrf(num, d):
    """ decipher the value of z, r, and f

    args:
        num (float | int): a number parsed from the string
        d (int): the decimal place of the last significant digit in the exemplar number
    #TODO d can be inferred from the number itself
    """
    def is_last_digit_five(number):
        # Convert the number to a string and check its last character
        return str(number)[-1] == '5'

    z = math.log(num, 10)
    r = -1 * d
    f = is_last_digit_five(num)
    return z, r, f


def is_number(n):
    """check if a string is a number 
    note:
        - If string is not a valid `float`,
        - it'll raise `ValueError` exception
    """

    try:
        float(n)  # Type-casting the string to `float`.
    except ValueError:
        return False
    return True


def count_sigfigs(numstr: str) -> int:
    """Count the number of significant figures in a number string"""

    return len(Decimal(numstr).as_tuple().digits)


def count_sig_digits_bias(number):
    """to count the bias for the getting the significant digits after the decimal point

        note:
            to exclude the sig digits before the decimal point
    """
    a, b = math.modf(number)
    return len(str(int(b)))


def findWholeWord(w):
    """Find a whole word in a string

        note:
            this returns the matched word, but not directly a boolean
    """

    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search


def whole_word_detect(word, string):
    """Detect if a whole word is in a string, return y or n"""

    if word in string.split():
        print("success")
    else:
        print("Not found")
