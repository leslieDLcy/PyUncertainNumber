# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:51:46 2024

I have created the first UP method: the vertex propagation.  It works for a list of intervals as input .  It produces three .csv files as output. It will not fail if  there the function yields NA.

@author: ioanna, Leslie
"""
import csv
import numpy as np
import tqdm

# import plotly.express as ps
import itertools
import pandas as pd
#<<<<<<< HEAD
from .utils  import header_results
#=======
from pathlib import Path
from .utils import post_process


def header_results(OUTPUT, INPUT):
    header = []

    for i in range(0, len(OUTPUT[1])):
        output = "y" + str(i)
        header.append(output)
    for i in range(0, len(INPUT[1])):
        Input = "x" + str(i)
        header.append(Input)

    return header


def Upper_lower_values_with_input(df_OUTPUT_INPUT, i):
    # Removes possible NAN values from a given output QoI.
    # Estimates the min and max for each output QoI from the remaining non NAN values.
    # Creates a dataframe with the output-input for the min and max of each Qof.
    ## Issues: if more than one combinations produce min/max it canot be recorded.
    ## if some variables are responsible for one QoI and not hte others it cannot be accounted for
    y = df_OUTPUT_INPUT[df_OUTPUT_INPUT.columns[i]]

    # create a data.frame with one Qof and input
    x = df_OUTPUT_INPUT.filter(regex="x")

    filtered_df = pd.concat([y, x], axis=1)

    filtered_df_OUTPUT_INPUT = filtered_df[filtered_df.iloc[:, 0].notnull()]
    yint = [
        min(filtered_df_OUTPUT_INPUT.iloc[:, 0]),
        max(filtered_df_OUTPUT_INPUT.iloc[:, 0]),
    ]

    dat_ymin = filtered_df_OUTPUT_INPUT[filtered_df_OUTPUT_INPUT.iloc[:, 0] == yint[0]]
    dat_ymax = filtered_df_OUTPUT_INPUT[filtered_df_OUTPUT_INPUT.iloc[:, 0] == yint[1]]

    y_dat_com = pd.concat([dat_ymin, dat_ymax])
    y_dat_com = y_dat_com.drop_duplicates(subset=[y_dat_com.columns[0]], keep="first")
    y_dat_com = y_dat_com.rename(columns={y_dat_com.columns[0]: "y"})

    y_dat_com.insert(0, "Name", df_OUTPUT_INPUT.columns[i])
    y_dat_com.insert(1, "fun", ["min", "max"])

    return y_dat_com


def NA_values_with_input(df_OUTPUT_INPUT, header, res_path):
    # creates a data.frame to store the input values which yield NAN of an output QoI
    # Store the data.frame into a file
    df_NA = df_OUTPUT_INPUT[df_OUTPUT_INPUT.isna().any(axis=1)]
    # The input values ae rounded to ensure equality
    df_NA = df_NA.apply(np.round, args=[4])
    df_NA_unique = df_NA.drop_duplicates(keep="first", ignore_index=True)

    create_csv(res_path, "NAlog.csv", df_NA_unique, header)

    return df_NA_unique


# def create_results_folder(newpath):
#      if not os.path.exists(newpath):
#         os.makedirs(newpath)
#      return newpath


def create_csv(res_path, filename, data, header):
    # creates a csv file to store output of a UP method in .csv format
    try:
        # Attempt to open the file
        file_path = res_path / filename
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data.values.tolist())
    except FileNotFoundError:
        print("The file does not exist.")

    return filename


def create_folder(base_path, method):
    # creates a folder where all the output files of each Up method will be stored
    # named after hte method used.
    base_path = Path(base_path)

    res_path = base_path / method
    res_path.mkdir(parents=True, exist_ok=True)

    return res_path

#>>>>>>> bc1b301fa7bef3748ecd22d0cb3067a1a9b38fb4

def listit(t):
    # transforms tuple of QoI values to sublists.
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

##### Vertex propagation ####
def vertexMethod(x, fun):
    """the vertex method for uncertainty propagation

    args:
        - intervals: list of lists, the intervals for each input variable
        - x: np.ndarray, the transformed nd.array for each input variable such as (5, 2)
        - fun: function, the performance function to be propagated

    signature:
        vertexMethod(intervals: List[List[float]], fun: Callable) -> pd.DataFrame

    note:
        - the augument `intervals` will specify the list of input intervals while the
        - arguement `fun` will specify the function mapping of variables to be propagated.

    return:
        - pd.DataFrame:

    example:
        x = [[1, 2], [3, 4], [5, 6]]
        fun = lambda x: x[0] + x[1] + x[2]
        y = vertexMethod(x, fun)

    # TODO post-process the df into UN objects ...
    # units ...
    # fields: None...
    """

    # x = np.array(intervals).reshape(-1, 2)  # Just in case in shape such as (5, 2)

    """  
        note: below suggests the original Ioanna's implementation which takes a list of intervals (
        indeed Python list), ergo I'll need the code block below to take such list of lists. But later on,
        I modify and directly take a numpy array of shape (n, 2) where n is the number of input variables.
        )

        # with below in place: 
            x = np.array(intervals).reshape(-1, 2)  # Just in case in shape such as (5, 2)

        # then 
            #     obj_vars = {instance.symbol: instance for instance in cls.instances}
            # intervals_ = [
            #     obj_vars[k].interval_initialisation for k in obj_vars if k in vars
            # ]
            # df = vM(intervals_, func)
            # return df    
    
    """

    # x.shape -> (5, 2) Interval object....
    # TODO a short-cut herein to get x

    total = 2 ** x.shape[0]

    y = []
    i = 0
    for c in tqdm.tqdm(itertools.product(*x)):
        y.append(fun(c))
        i += 1
        # print(f" Completed evaluation #{i} of total {total}...")

    y = listit(y)
    ## Create a data.frame with output-input ####
    # Input
    INPUT = [list(ele) for ele in list(itertools.product(*x))]
    # Output
    if len(y[0]) == 1:
        OUTPUT = []
        # Iterate over a sequence of numbers from 0 to 3
        for i in range(len(y)):
            # In each iteration, add an empty list to the main list
            OUTPUT.append([y[i]])
    else:
        OUTPUT = y

    # Merge output input in a single sublist
    OUTPUT_INPUT = [sub1 + sub2 for sub1, sub2 in zip(OUTPUT, INPUT)]
    header = header_results(OUTPUT, INPUT)

    # construct output-input data.frame
    df_OUTPUT_INPUT = pd.DataFrame(OUTPUT_INPUT)
    df_OUTPUT_INPUT.columns = header
    return df_OUTPUT_INPUT
#<<<<<<< HEAD

#=======
#>>>>>>> bc1b301fa7bef3748ecd22d0cb3067a1a9b38fb4
