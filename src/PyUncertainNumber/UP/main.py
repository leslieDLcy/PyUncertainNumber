import numpy as np

# import plotly.express as ps
from .vertex import vertexMethod, BB_results, create_folder


def UP(base_path, *, intervals, fun, method=vertexMethod):
    """the general template function to do uncertainty propagation (UP)

    args:
        - base_path: str, the path where the results will be saved
        - intervals: list of lists, the intervals for each input variable
        - fun: function, the performance function to be propagated
    """

    res_path = create_folder(base_path, method)
    df_OUTPUT_INPUT = method(intervals, fun)

    Results = BB_results(df_OUTPUT_INPUT, res_path)
    print(Results)
    return Results


def main():
    """implementation of the vertex method on the cantilever beam example"""

    y = [0.145, 0.155]  # in m
    beam_length = [9.95, 10.05]  # in m
    I = [0.0003861591, 0.0005213425]  # in m^4
    F = [11, 37]  # kN
    E = [200, 220]  # GPa

    intervals = [y, beam_length, I, F, E]

    # Define function
    fun = lambda x: x[0] * x[1] / (x[2] * np.log(x[3]))

    # perhaps we need to add a new input to the function with the names of the input output in line with UN code
    # check the code with the xfoil code
    # a = vertexMethod(intervals, fun)
    # df_OUTPUT_INPUT = subintervalMethod(intervals, fun, n=2)

    method = "vertex"
    # base_path = "C:/Users/Ioanna/OneDrive - The University of Liverpool/DAWS2_code/UP/Results_vertex"
    a = UP(base_path, method, intervals, fun)


if __name__ == "__main__":
    main()
