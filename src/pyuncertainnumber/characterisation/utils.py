from scipy.stats import ecdf
import os
import pathlib
import re
import ast
import json
import dataclasses
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt
import scipy.stats as sps


# TODO create a defending mechanism for parsing '[15+-10%]' as only '[15 +- 10%]' works now


def save2Vis(figname):
    """a shortcut function to save plot to visualization dir

    Note
    ----

    We simply assume that every repo will have a 'visulizations'
    dir under the root directory
    """

    axe = plt.gca()
    plt.savefig(f"{figname}.png", format="png", dpi=300, bbox_inches="tight")


def tranform_ecdf(s, display=False, **kwargs):
    """plot the CDF return the quantile

    args:
        s: sample
    """
    sth = ecdf(s)
    if display:
        fig, ax = plt.subplots()
        # ax.plot(x_support, p_values, color='g')
        ax.step(
            sth.cdf.quantiles,
            sth.cdf.probabilities,
            color="red",
            zorder=10,
            where="post",
            **kwargs,
        )
        return sth.cdf.quantiles, ax
    else:
        return sth.cdf.quantiles


def pl_pcdf(
    dist: type[sps.rv_continuous | sps.rv_discrete], ax=None, title=None, **kwargs
):
    from ..pba.params import Params

    """plot CDF from parametric distribution objects"""

    if ax is None:
        fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)

    x_values = dist.ppf(Params.p_values)

    ax.plot(x_values, Params.p_values, **kwargs)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\Pr(X \leq x)$")
    return ax


def pl_ecdf(s, ax=None, return_value=False, **kwargs):
    """plot the empirical CDF given samples

    args:
        s (array-like): sample which can be either raw data
            or deviates as a representation of dist construct
    """
    sth = ecdf(s)
    if ax is None:
        fig, ax = plt.subplots()
    # ax.plot(x_support, p_values, color='g')
    ax.step(sth.cdf.quantiles, sth.cdf.probabilities, where="post", **kwargs)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\Pr(X \leq x)$")
    if not return_value:
        return ax
    else:
        return ax, sth.cdf.quantiles, sth.cdf.probabilities


def to_database(dict_list, db_name, col_name):

    json_rep = json.dumps(dict_list, cls=UNEncoder)
    dict_rep = json.loads(json_rep)

    with MongoClient() as client:
        db = client[db_name]

        # collection
        a_collection = db[col_name]

        # insert documents
        new_result = a_collection.insert_many(dict_rep)
        return new_result


def cd_root_dir(depth=0):
    # change directory to the path of the root directory of the project

    ref_path = os.path.abspath("")
    ref_path = pathlib.Path(ref_path).resolve().parents[depth]
    os.chdir(ref_path)
    print("current directory:", os.getcwd())


def initial_list_checking(text):
    """detects if a string representation of a list"""

    try:
        return ast.literal_eval(text)
    except:
        # print(error)
        # print("Not a list-like string representation")
        pass


def bad_list_checking(text):
    """detects if a syntactically wrong specification of a list"""

    flag = text.startswith("[") & text.endswith("]")
    # if flag:
    #     print("Wrong spec of a list repre")
    # else:
    #     print("Not even a list")
    return flag


def PlusMinus_parser(txt):

    flag = "+-" in txt
    if flag:
        # print("Contains '+-' ergo initiate using mid range style")
        return True
        # txt_list = list(txt)
    # return txt_list


# def deciper_num_from_string():


def parser4(text):

    # do an extra step of scraping the '[' and ']'
    if bad_list_checking(text):
        subtexts = text.strip("[]").split()
    else:
        subtexts = text.split()
    return [int(s) for s in subtexts if s.isdigit()]


def percentage_finder(txt):
    pctg = re.findall("\d*%", txt)
    # return pctg
    if pctg:
        return True
    else:
        return False


def percentage_converter(txt):
    """convert a percentage into a float number

    note:
        force only 1 percentage
    """
    # return re.findall(r'(\d+(\.\d+)?%)', txt)

    pctg = re.findall("\d*%", txt)
    return float(pctg[0].strip("%")) / 100


class EnhancedJSONEncoder(json.JSONEncoder):
    """a template for jsonify general (dataclass) object

    #TODO Interval object in not json serializable
    """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


class PBAEncoder(json.JSONEncoder):
    """a bespoke JSON encoder for the PBA object"""

    def default(self, o):
        # if any(isinstance(value, np.ndarray) for value in o.__dict__.values()):
        # TODO to use __slot__ later on to save disk space
        removed_dict = o.__dict__.copy()
        entries_to_remove(remove_entries=["left", "right"], the_dict=removed_dict)
        return removed_dict


class UNEncoder(json.JSONEncoder):
    """a bespoke JSON encoder for the UncertainNumber object

    note:
        - Currently I'm treating the JSON data represent of a UN object
        the same as the __repr__ method. But this can be changed later on to
        show more explicitly the strucutre of pbox or distribution
        # TODO prettify the JSON output to be explicit
        e.g. 'essence': 'interval', 'interval_initialisation': [2, 3] to shown as 'interval' with lower end and upper end
        distribution to shown as the type and parameters; e.g. 'distribution': 'normal', 'parameters': [2, 3]
    """

    def default(self, o):
        # if any(isinstance(value, np.ndarray) for value in o.__dict__.values()):
        # TODO to use __slot__ later on to save disk space
        copy_dict = o.__dict__.copy()

        return get_concise_repr(copy_dict)


def get_concise_repr(a_dict):
    # remove None fields
    Noneremoved_dict = {k: v for k, v in a_dict.items() if v is not None}

    # remove some unwanted fields (keys)
    entries_to_remove(
        remove_entries=[
            "distribution_parameters",
            "pbox_parameters",
            "bounds",
            "masses",
            "_unit",
            "_physical_quantity",
            "_skip_construct_init",
            "_UnitsRep",
            "_math_object",
            "deter_value_rep",
            "_Q",
            "p_flag",
        ],
        the_dict=Noneremoved_dict,
    )
    return Noneremoved_dict


def array2list(a_dict):
    """convert an array from a dictionary into a list"""

    return {
        k: arr.tolist() if isinstance(arr, np.ndarray) else arr
        for k, arr in a_dict.items()
    }


def entries_to_remove(remove_entries, the_dict):

    for key in remove_entries:
        if key in the_dict:
            del the_dict[key]
