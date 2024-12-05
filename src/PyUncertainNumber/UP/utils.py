from pathlib import Path
import csv 
import pandas as pd
import numpy as np


def header_results(all_output, all_input, method = None):
    """
    Determine generic header for output and input.

    Args:
        all_output (np.ndarray): A NumPy array containing the output values.
        all_input (np.ndarray): A NumPy array containing the input values.

    Returns:
        list: A list of strings representing the header for the combined DataFrame.
    """
    if all_output is None:
        header_y = []
    else:
        if all_output.ndim == 1:
            len_y = 1
        else:
            len_y = all_output.shape[1]
        header_y = ["y" + str(i) for i in range(len_y)]

    if method in "cauchy": 
        m = all_input.shape[1] - 1  # Exclude 'K' from the count
        header_x = ["x" + str(i) for i in range(m)] + ["K"]  # Add 'K' at the end
    else:
        m = all_input.shape[1]
        header_x = ["x" + str(i) for i in range(m)]

    header = header_y + header_x
    return header

def post_processing(all_input: np.ndarray, all_output: np.ndarray = None, method = None, res_path=None):

    """Post-processes the results of an uncertainty propagation (UP) method.

    This function takes the input and output values from a UP method, combines them into a 
    pandas DataFrame, and optionally saves the raw data to a CSV file. It also checks for 
    NaN values in the output and logs them with their corresponding input values if found.
    If all_output is None, it creates a DataFrame with only the input data.

    Args:
        all_input (np.ndarray): A NumPy array containing the input values used in the UP method.
        all_output (np.ndarray, optional): A NumPy array containing the corresponding output 
                                            values from the UP method. Defaults to None.
        res_path (str, optional): The path to the directory where the results will be saved. 
                                    Defaults to None.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing the combined output and input data 
                        (if all_output is provided). If all_output is None, it returns a 
                        DataFrame with only the input data.
    """

    if all_output is None:
        print("No function was evaluated. Only input is available")

        df_input = pd.DataFrame(all_input)

        # Handle Cauchy input with 'x' and 'K' (moved outside the if block)
        # if method in ( "endpoints_cauchy"):  
        #     x_fields = pd.DataFrame(all_input)  # Get all 'x' field names
        #     print("x_fields", x_fields)
        #     x_data = all_input[x_fields]  # Select only the 'x' fields
        #     df_input = pd.DataFrame(x_data)
        # else:
        #     df_input = pd.DataFrame(all_input)

        header = header_results(all_output=None, all_input=all_input, method = method)  
        df_input.columns = header
        df_output_input = df_input  

    else:
        # Transform np.array input-output into pandas data.frame 
        df_input = pd.DataFrame(all_input)
        df_output = pd.DataFrame(all_output)

        # Create a single output input data.frame
        df_output_input = pd.concat([df_output, df_input], axis=1)

        # determine generic header for output and input
        header = header_results(all_output, all_input, method)
        df_output_input.columns = header

    # Return .csv with raw data only if asked ###
    if res_path is not None:
        create_csv(res_path, "Raw_data.csv", df_output_input)

    # Check for NaN values ONLY if all_output is provided
    if all_output is not None:  
        df_NA = df_output_input[df_output_input.isna().any(axis=1)]
        if len(df_NA) != 0:
            # The input values are rounded to ensure equality
            df_NA = df_NA.apply(np.round, args=[4])
            df_NA_unique = df_NA.drop_duplicates(keep="first", ignore_index=True)

            create_csv(res_path, "NAlog.csv", df_NA_unique)
        else:
            print("There are no NA values produced by the input")

    return df_output_input


def create_folder(base_path, method):
    """Creates a folder named after the called UP method where the results files are stored

    args:
        - base_path: The base path
        - method: the name of the called method

    signature:
        create_folder(base_path: string, method: string ) -> path.folder

    note:
        - the augument `base_path` will specify the location of the created results folder.
        - the argument `method` will provide the name for the results folder. 

    return:
        -  A folder in a prespecified path
 
    example:
        base_path = "C:/Users/DAWS2_code/UP"
        method = "vertex"
        y = create_folder(base_path, method)
    """    
    base_path = Path(base_path)

    res_path = base_path / method
    res_path.mkdir(parents=True, exist_ok=True)

    return res_path

def create_csv(res_path, filename, data):
    """Creates a .csv file and sotres it in a pre-specified folder with results generated by a UP method

    args:
        - res_path: A folder in a prespecified path named after the called UP method
        - filename: the name of the file
        - data: a pandas.dataframe with results from UP method

    signature:
        create_csv(res_path = path, filename = filename, data = pandas.dataframe) -> path.filename

    note:
        - the augument `res_path` will specify the folder where the .csv file will be created.
        - argument `file` will provide the name of hte .csv file.
        - argument `data` will provide data in terms of pandas.dataframe.

    return:
        -  A .csv file in a prespecified folder 
 
    example:
        base_path = "C:/Users/DAWS2_code/UP/vertex"
        filename = 'min_max_values'
       df = pd.DataFrame(
         {"Name" : ["y0", "y0"],
          "fun"  : ["min","max"]
          "y0" : [4, 6]}, index = [1, 2, 3])
        header = ['Name', 'fun', 'values']
        y = create_csv(res_path, filename, df)
    """  
    try:
        # Attempt to open the file
        file_path = res_path / filename
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data.columns)
            writer.writerows(data.values.tolist())
    except FileNotFoundError:
        print("The file does not exist.")

    return filename

def save_results(data, method, res_path, fun=None):
    if fun is None:
        Results = post_processing(data['x'], all_output=None, method=method, res_path=res_path)
    else:
        Results = post_processing(data['x'], data.get('f'), method, res_path)
            
    return Results