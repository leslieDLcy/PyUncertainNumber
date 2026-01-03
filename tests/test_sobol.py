from pyuncertainnumber import UncertainNumber as UN
from pyuncertainnumber.sensitivity.sobol_analysis import sobol_analysis


def test_sobol_analysis():
    # 1. Define uncertain inputs
    inputs = [
        UN(
            name="x1",
            essence="distribution",
            distribution_parameters=["uniform", (0, 1)],
        ),
        UN(
            name="x2",
            essence="distribution",
            distribution_parameters=["gaussian", (0.5, 0.1)],
        ),
    ]

    # 2. Define a model function (non-vectorized)
    def my_model(x):
        out = x[:, 0] + x[:, 1]
        return out

    # 3. Run the analysis
    results = sobol_analysis(
        inputs,
        my_model,
        plot_results=False,
        calc_second_order=True,
        print_to_console=True,
        N=512,
    )

    # 4. Print results for the first output
    print(results)


# def myFunctionWithTwoOutputs(x):
#     """Non-vectorized example function."""
#     input1, input2, input3, input4, input5 = x[0], x[1], x[2], x[3], x[4]
#     output1 = input1 + input2 + input3 + input4 + input5
#     output2 = input1 * input2 * input3 * input4 * input5
#     return output1#, output2

# x_inputs = [
#     UN(name='x0', essence='distribution', distribution_parameters=["gaussian", (1, 0.1)]),
#     UN(name='x1', essence='distribution', distribution_parameters=["gaussian", (2, 0.2)]),
#     UN(name='x2', essence='distribution', distribution_parameters=["gaussian", (3, 0.3)]),
#     UN(name='x3', essence='distribution', distribution_parameters=["gaussian", (4, 0.4)]),
#     UN(name='x4', essence='distribution', distribution_parameters=["gaussian", (5, 0.5)]),
# ]

# # Call the analysis function.
# results = sobol_analysis(
#     x_inputs,
#     myFunctionWithTwoOutputs,
#     output_names=['Sum'], #'Product'
#     N=2048
# )
