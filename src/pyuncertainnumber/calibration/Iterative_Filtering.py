import numpy as np
import random as rd


def iterative_filter(
    func: callable,
    variables: dict,
    nsamples: int,
    thresholds: list,
    conditions: dict,
    folder: str,
    max_repetitions: int = 3,
    index: list = None,
    controls: list = None,
):
    """Performs the iterative filtering calibration method on a given function with uncertain inputs.

    args:

        func (callable): the function being analysed and used to get the outputs.

        variables (dict): a dictionary that contains the names and bounds of all variables within the function.

        nsamples (int): the number of samples taken for each variable.

        thresholds (list): the set of threshold filter values that the conditions will need to meet.

        conditions (dict): a dictionary of 'correct' values that the outputs of interest will need to meet.

        folder (str): where all outputs of this function will be stored in.

        max_repetitions (int): how many repetitions are allowed at a singular threshold value before stopping the process.

        index (list): a list of indexes referring to the list of variables that are being filtered.

        controls (list): a list of indexes referring to the variables whose values are unique to their condition and must be generated separately.
    """

    if (
        index == None
    ):  # If i is None, that means that no variables are selected for filtering which does not make sense and so returns an error
        raise ValueError(
            "No variables selected for filtering, use the index to list variables to be filtered."
        )

    for s in index:
        with open(f"./{folder}/bounds_all_min.txt", "a") as a:
            a.write(str(variables[conditions["names"][0]][s][0]) + ",")
        with open(f"./{folder}/bounds_all_max.txt", "a") as a:
            a.write(str(variables[conditions["names"][0]][s][1]) + ",")
    with open(f"./{folder}/bounds_all_min.txt", "a") as a:
        a.write(f"str(0)" + "," + f">{thresholds[0]}" + "\n")
    with open(f"./{folder}/bounds_all_max.txt", "a") as a:
        a.write(f"str(0)" + "," + f">{thresholds[0]}" + "\n")

    Iteration = 1
    max_Iteration = len(thresholds[1:])
    repetitions = 0
    total_repetitions = 0
    nconditions = len(conditions["names"])

    while Iteration <= max_Iteration:
        print(Iteration)  # optional, provides visual counter of progress
        X_input = np.empty([nsamples * nconditions, len(variables["names"])])
        for j in range(0, nsamples * nconditions, 2):
            for k in range(nconditions):
                for l in range(len(variables["names"])):
                    if l not in controls and k > 0:
                        X_input[j + k, l] = X_input[j, l]
                    elif (
                        variables["number_type"][l] == "int"
                    ):  # Used for NASA UQ Example to get an integer for the seed.
                        X_input[j + k, l] = rd.randint(
                            variables[conditions["names"][k]][l][0],
                            variables[conditions["names"][k]][l][1],
                        )
                    else:
                        X_input[j + k, l] = rd.uniform(
                            variables[conditions["names"][k]][l][0],
                            variables[conditions["names"][k]][l][1],
                        )

        # for j in range(nconditions): #Storing each condition's inputs separately (optional)
        #     X_input_set = X_input[j::nconditions]
        #     input_file_path = f'./{folder}/input{variables['names'][j]}_Level{Iteration}_{repetitions}.txt'
        #     np.savetxt(input_file_path, X_input_set, delimiter=',')

        input_file_path = f"./{folder}/input_Level{Iteration}_{repetitions}.txt"
        np.savetxt(
            input_file_path, X_input, delimiter=","
        )  # stores the full input matrix in one file (required for NASA UQ example)

        Y_out = func(
            input_file_path, nsamples, nconditions
        )  # Must be edited as per user input requirements, output must be 'Y_out'.

        if (
            repetitions == 0
        ):  # When a new Iteration is reached, resets the accepted sample matrix and count.
            filtered_samples = np.empty([nsamples, 9])
            count = 0
        for i in range(nsamples):  # Looping through all samples
            test_threshold = thresholds[Iteration]
            test_Level = Iteration
            point_added = False
            while (
                point_added is not True
            ):  # While loop finds which Iteration the sample belongs to.
                threshold_set = np.empty([len(conditions["values"]), nconditions])
                for j in range(len(conditions["values"])):
                    threshold_set[j] = (
                        abs(Y_out[j, i, :] - conditions["values"][j]) <= test_threshold
                    )

                if threshold_set[:, :].all() == 1:
                    if (
                        test_Level == Iteration
                    ):  # If sample is accepted at the same Iteration that it starts, it is added to the matrix for updating the Xe bounds
                        filtered_samples[count] = X_input[nconditions * i]
                        count += 1
                    point_added = True
                    for s in index:
                        with open(f"./{folder}/samples.csv", "a") as a:
                            a.write(str(X_input[nconditions * i, s]) + ",")
                        with open(f"./{folder}/Iteration{test_Level}.csv", "a") as a:
                            a.write(str(X_input[nconditions * i, s]) + ",")
                        with open(f"./{folder}/samples.csv", "a") as a:
                            a.write(f"{test_Level}" + "," + "\n")
                        with open(f"./{folder}/Iteration{test_Level}.csv", "a") as a:
                            a.write(f"{test_Level}" + "," + "\n")
                else:
                    test_Level = test_Level - 1
                    test_threshold = thresholds[
                        test_Level
                    ]  # Moving the radius down a Iteration (can be changed out to an array of radii instead)
                    if (
                        test_Level <= 0
                    ):  # If sample is not within the start radius for all controls, the while loop is broken.
                        for s in index:
                            with open(f"./{folder}/samples.csv", "a") as a:
                                a.write(str(X_input[nconditions * i, s]) + ",")
                            with open(
                                f"./{folder}/Iteration{test_Level}.csv", "a"
                            ) as a:
                                a.write(str(X_input[nconditions * i, s]) + ",")
                            with open(f"./{folder}/samples.csv", "a") as a:
                                a.write(f"{test_Level}" + "," + "\n")
                            with open(
                                f"./{folder}/Iteration{test_Level}.csv", "a"
                            ) as a:
                                a.write(f"{test_Level}" + "," + "\n")
                        break

        print(
            count
        )  # Optional, allows for a quick check of how many samples were 'accepted' at a given iteration while process is ongoing.
        if (
            count > 1
        ):  # If more than one sample is accepted at the current Iteration, new bounds are computed and Iteration goes up.
            filtered_samples.resize(count, len(variables["names"]))

            for i in range(nconditions):
                for s in index:
                    variables[conditions["names"][i]][s] = np.min(
                        filtered_samples[:, s]
                    ), np.max(filtered_samples[:, s])
                    if i == 0:
                        print(
                            f"New Bounds for {variables['names'][s]}: {np.min(filtered_samples[:,s]), np.max(filtered_samples[:,s])}"
                        )
            for s in index:
                with open(f"./{folder}/bounds_all_min.txt", "a") as a:
                    a.write(str(variables[conditions["names"][i]][s][0]) + ", ")
                with open(f"./{folder}/bounds_all_max.txt", "a") as a:
                    a.write(str(variables[conditions["names"][i]][s][1]) + ", ")
            with open(f"./{folder}/bounds_all_min.txt", "a") as a:
                a.write(
                    f" str({Iteration})"
                    + ", "
                    + f"{thresholds[Iteration]}"
                    + ", "
                    + "\n"
                )
            with open(f"./{folder}/bounds_all_max.txt", "a") as a:
                a.write(
                    f"str({Iteration})"
                    + ", "
                    + f"{thresholds[Iteration]}"
                    + ", "
                    + "\n"
                )

            total_repetitions += (
                repetitions  # Add repetitions from current Iteration to total
            )
            repetitions = 0  # Reset repetitions when new bounds are found.
            Iteration += (
                1  # Increase Iteration (and by extension, decrease radius being tested)
            )
            if (
                Iteration > max_Iteration
            ):  # If max_Iteration is reached, no more radii are available so while loop is broken.
                print(
                    f"All control points intersect within the radius {thresholds[max_Iteration]} with {max_Iteration} levels of reduction and {max_Iteration+total_repetitions} iterations of {nsamples} samples."
                )
                break

        else:  # Otherwise the process gets repeated and the number of repetitions at the current Iteration is increased.
            if (
                repetitions >= max_repetitions
            ):  # If Iteration is repeated max_repetitions amount of times, while loop is broken.
                total_repetitions += repetitions
                print(
                    f"All control points lack viable points below radius: {thresholds[Iteration]} at Iteration {Iteration} with {Iteration+total_repetitions} iterations of {nsamples} samples."
                )
                break
            repetitions += 1

    return
