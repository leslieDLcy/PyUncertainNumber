from typing import Callable
from rich.progress import track
import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from matplotlib import pyplot as plt


def sobol_analysis(
    x: list, f: Callable, output_names=None, plot_results: bool = True, **kwargs
):
    """Performs a Sobol sensitivity analysis using the SALibe library.

    Args:
        x (list): A list of ``UncertainNumber`` objects defining the input variables.
        f (Callable): The model function to be analyzed.
        output_names (list, optional): A list of strings to name the model outputs.
                                       Defaults to None.
        plot_results (bool, optional): If True, generates and displays plots of the
                                       results. Defaults to True.
        **kwargs: Additional keyword arguments to pass to SALib's ``sample`` and
                  ``analyze`` functions. For example, ``N`` (the number of samples)
                  must be a power of 2 (e.g., 1024, 2048) and defaults to 1024.

    Returns:
        dict: A dictionary where keys are the output names and values are the
              corresponding SALib analysis result objects.

    .. tip::
        This function is designed to work with a list of ``UncertainNumber`` objects
        that are defined as distributions. It evaluates the sensitivity of a model
        function ``f``, which can have single or multiple outputs, to these inputs.

        The code performs the following steps:
            1.  **Problem Definition**: A SALib `problem` dictionary is constructed from the
                input list `X` of UncertainNumber objects. This dictionary defines the
                variables, their names, distributions, and parameters.

            2.  **Vectorization Auto-Detection**: The function automatically probes the
                model function `f` to determine if it is "vectorized" (i.e., can
                process all samples at once with NumPy) or "non-vectorized" (processes
                one sample at a time). This allows for flexible use without needing to
                modify the model function.

            3.  **Direct Sampling**: Input samples are generated using `SALib.sample.sobol.sample`.
                This function is passed the problem definition directly.
                **Important Note**: When using this direct method, SALib performs a
                linear scaling of the Sobol sequence from the [0, 1] domain to the
                `bounds` provided. It does *not* perform a distributional transformation
                (e.g., using the inverse CDF). For a 'gaussian' distribution defined
                with bounds `[mean, std_dev]`, the generated samples will be uniformly
                distributed between `mean` and `std_dev`, not normally distributed.

            4.  **Model Evaluation**: The model `f` is evaluated using the generated samples.
                The evaluation is performed efficiently if the model is vectorized, or
                iteratively (with a progress bar) if it is not.

            5.  **Output Handling**: The function gracefully handles models that return
                single or multiple outputs, normalizing them for analysis.

            6.  **Sobol Analysis**: For each model output, `SALib.analyze.sobol.analyze`
                is called to compute the first-order (S1), second-order (S2), and
                total-order (ST) sensitivity indices.

            7.  **Plotting**: If `plot_results` is True, bar charts of the S1 and ST
                indices are generated for each output.

    Example:
        >>> from pyuncertain import UncertainNumber as UN
        >>>
        >>> # 1. Define uncertain inputs
        >>> inputs = [
        ...     UN(name='x1', essence='distribution', distribution_parameters=["uniform", (0, 1)]),
        ...     UN(name='x2', essence='distribution', distribution_parameters=["gaussian", (0.5, 0.1)])
        ... ]
        >>>
        >>> # 2. Define a model function (non-vectorized)
        >>> def my_model(x):
        ...     return x[0]**2 + x[1]
        >>>
        >>> # 3. Run the analysis
        >>> results = sobol_analysis(inputs, my_model, print_to_console = True, plot_results: bool = True, N=512)
        >>>
        >>> # 4. Print results for the first output
        >>> results['Output_1']
        >>>

    """
    try:
        # === STAGE 1: Build the problem dictionary ===
        dist_map = {"gaussian": "norm", "uniform": "unif", "triangular": "triang"}
        names, bounds, dists = [], [], []

        for un in x:
            var_name = un.name
            dist_name, dist_params = un.distribution_parameters
            if dist_name not in dist_map:
                raise ValueError(f"Distribution '{dist_name}' is not supported.")
            names.append(var_name)
            bounds.append(list(dist_params))
            dists.append(dist_map[dist_name])

        problem = {"num_vars": len(x), "names": names, "bounds": bounds, "dists": dists}

        # === STAGE 2: Auto-Detect Vectorization ===
        is_vectorized = False
        try:
            probe_input = np.zeros((2, problem["num_vars"]))
            f(probe_input)
            print(f(probe_input))
            is_vectorized = True
            print("INFO: Auto-detected vectorized model.")
        except Exception:
            print(
                "INFO: Auto-detected non-vectorized model. Using sample-by-sample evaluation."
            )

        # === STAGE 3: Sample and Evaluate Model ===
        KNOWN_SAMPLING_ARGS = [
            "N",
            "calc_second_order",
            "scramble",
            "skip_values",
            "seed",
        ]
        sample_args, analyze_args = {}, {}
        for key, value in kwargs.items():
            if key in KNOWN_SAMPLING_ARGS:
                sample_args[key] = value
            else:
                analyze_args[key] = value

        sample_args.setdefault("N", 1024)
        sample_args.setdefault("calc_second_order", True)
        analyze_args.setdefault("print_to_console", True)

        param_values = sobol_sample.sample(problem, **sample_args)

        if is_vectorized:
            Y = f(param_values)
        else:
            Y = [
                f(row)
                for row in track(param_values, description="Processing samples...")
            ]
            print("Y", Y)

        # === STAGE 4: Normalize Output and Analyze ===
        if isinstance(Y, (list, tuple)) and not np.isscalar(Y[0]):
            Y_multi_output = (
                np.array(Y) if isinstance(Y[0], (list, tuple)) else np.column_stack(Y)
            )
        elif isinstance(Y, list):
            Y_multi_output = np.array(Y).reshape(-1, 1)
        elif Y.ndim == 1:
            Y_multi_output = Y.reshape(-1, 1)
        elif Y.ndim == 2:
            Y_multi_output = Y
        else:
            raise ValueError(f"Model output has an unsupported shape: {Y.shape}")

        num_outputs = Y_multi_output.shape[1]
        if output_names is None:
            output_names = [f"Output_{i+1}" for i in range(num_outputs)]
        elif len(output_names) != num_outputs:
            raise ValueError(
                f"Provided {len(output_names)} names, but model has {num_outputs} outputs."
            )

        all_results = {}
        for i, name in enumerate(output_names):
            print(f"\nResults for '{name}':")
            Si = sobol_analyze.analyze(problem, Y_multi_output[:, i], **analyze_args)
            all_results[name] = Si

            # all_results[name]
            if plot_results:

                all_results[name].plot()
                plt.title(f"Sobol Indices for: {name}")
                plt.xlabel("Parameters")
                plt.show()

        return all_results

    except Exception as e:
        print(f"\n>>> SCRIPT FAILED <<<")
        import traceback

        traceback.print_exc()
        return None
