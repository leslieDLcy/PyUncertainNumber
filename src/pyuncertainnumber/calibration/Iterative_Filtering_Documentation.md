# Iterative Filtering

This notebook is designed to explain the iterative filtering function, its processes, and its requirements to work.
The purpose of this process is to obtain an approximate estimate of the true value of a set of variables by using a set of output values related to the 'true' output.

The function relies mainly on the use of two dictionaries: one to detail the variables used in the function and their initial values; and another to detail the filtering conditions being used for the function.

Throughout this document, the 2025 NASA UQ Challenge model will be used as the example to show how the filtering process works.

## Variables dictionary

This dictionary lists all of the variables required in the function or model being analysed, along with their number types and their initial values. The default names for the keys are 'names' for the name of each variable and 'number_type' for the number type of each variable (most typically 'int' or 'float'). The keys for the variable values associated with each condition are typically the name of the condition.
These values are also theoretically compatible with interval arithmetic, allowing for their usage if the function being analysed is also compatible.

As an example, for the NASA UQ model the dictionary for the variables with only a 'base' and 'first query' condition would look like this:

```python
variables = {
    'names': ['a1','a2','e1','e2','e3','c1','c2','c3','s'],
    'number_type': ['float','float','float','float','float','float','float','float','int'],
    'b':  [[0,1],[0,1],[0,1],[0,1],[0,1],[0.533,0.533],[0.666,0.666],[0.5,0.5],[0,1000000]],
    's1': [[0,1],[0,1],[0,1],[0,1],[0,1],[0.052632,0.052632],[0.421053,0.421053],[0.631579,0.631579],[0,1000000]]
}
```

where 'b' is the name used for the 'base' condition and so it is also the name of the key for the associated values.

## Conditions dictionary

This dictionary lists all of the conditions used for the function or model. For the NASA UQ example this dictionary has the following form:

```python
conditions = {
    'names': ['b','s1'], #condition vector names (for saving files and getting how many conditions are used)
    'values':   [[2.031260556,1.746109960],  #y4y5 ratio values
                [2.408021879,3.429627891],  #y4y6 ratio values
                [1.185481534,1.964153444]]  #y5y6 ratio values
}
```
where we have each of the filtering conditions being used with their names in the 'names' key (then used to refer to the set of variable values to be used), along with the condition values in the 'values' key.
For this example, we use the epistemic ratios between the output variables y4, y5, and y6 as the filter conditions.
While the number of filtering values may vary depending on the function being analysed, this is not a problem as long the same number of filtering values can be obtained from the output, as this process allows for any number of filter values.

## Initial conditions

This process has a number of different necessary initial conditions which are required for correct functionality. Looking at the function definition:

```python
def iterative_filter(func,variables,nsamples,thresholds,conditions,folder,max_repetitions=3,index=None,controls=None):
```
we have nine initial conditions that must be specified.

In order of appearance, these are:
func - the function being analysed and used to get the outputs. $\\$
variables - a dictionary that contains the names and bounds of all variables within the function. $\\$
nsamples - the number of samples taken for each variable. $\\$
thresholds - the set of threshold filter values that the conditions will need to meet. $\\$
conditions - a dictionary of 'correct' values that the outputs of interest will need to meet. $\\$
folder - where all outputs of this function will be stored in. $\\$
max_repetitions - how many repetitions are allowed at a singular threshold value before stopping the process. $\\$
index - a list of indexes referring to the list of variables that are being filtered. $\\$
controls - a list of indexes referring to the variables whose values are unique to their condition and must be generated separately. $\\$

While the index condition has a default of None, this is intentional to bring up an error message if no variables are selected for filtering using the index.

For the set of thresholds, the first threshold is repeated to avoid the case where the Iteration variable goes to $-1$ and in turn takes the final threshold when a set of values does not meet any threshold. For example:
```python
thresholds = [0.5,0.5,0.3,0.2,0.1,0.05]
```
the threshold list for the NASA UQ example repeats the $0.5$ threshold.

At the start of the process, the function stores the initial values and highest threshold into a set of two text documents: one for the minimum values and one for the maximum values.
This is done through the following iterative process:
```python
for s in index:
    with open(f'./{folder}/bounds_all_min.txt', "a") as a:
        a.write(str(variables[conditions['names'][0]][s][0])+',')
    with open(f'./{folder}/bounds_all_max.txt', "a") as a:
        a.write(str(variables[conditions['names'][0]][s][1])+',')
with open(f'./{folder}/bounds_all_min.txt', "a") as a:
    a.write(f'str(0)'+ ',' + f'>{thresholds[0]}'+'\n')
with open(f'./{folder}/bounds_all_max.txt', "a") as a:
    a.write(f'str(0)'+ ',' + f'>{thresholds[0]}'+'\n')
```
which allows for any number of variables to be written.

The other initial conditions are the max_Level, which dictates the number of filter iterations that the process will run for, and the number of repetitions and total repetitions if the process needs to be resumed after it was paused. We also set the number of conditions to the nconditions variable by using the number of condition names from the conditions dictionary.

```python
Iteration = 1
max_Iteration = len(thresholds[1:])
repetitions = 0
total_repetitions = 0
nconditions = len(conditions['names'])
```

## Generating Samples

For each iteration we have to generate samples to test against the current threshold.
We do this by first creating an empty numpy array and then iteratively filling the array. To obtain identical samples across each condition, we have an if condition that duplicates the values of non-control variables when we have more than one condition and a sample was already generated. Then we generate either an integer or float sample depending on the number type of the variable. By default the sample distribution is uniform, so if the user requires a different distribution that must be changed by the user.
```python
X_input = np.empty([nsamples*nconditions,len(variables['names'])])
for j in range(0,nsamples*nconditions,2):
    for k in range(nconditions):
        for l in range(len(variables['names'])):
            if l not in controls and k > 0:
                X_input[j+k,l] = X_input[j,l]
            elif variables['number_type'][l] == 'int': #Used for NASA UQ Example to get an integer for the seed.
                X_input[j+k,l] = rd.randint(variables[conditions['names'][k]][l][0], variables[conditions['names'][k]][l][1])
            else:
                X_input[j+k,l] = rd.uniform(variables[conditions['names'][k]][l][0], variables[conditions['names'][k]][l][1])
```

## Executing a function and data collection

After the input array is generated, we have to run the function and collect the data necessary for the filtering process.
This is function specific and therefore requires a separate function to be created by the user that will do this step of the function.
The only requirement is that the output of the user-generated function is an array with name 'Y_out' where each filter condition output is separately indexable and in the same order as the conditions from the 'conditions' dictionary.

Using the NASA UQ example, for this step we have:
```python
Y_out = func(input_file_path, nsamples, nconditions)
```
where our written function takes in three inputs from earlier in the process.

This function handles the execution and data collection separately, allowing for generalisation of the filter process and greater user flexibility.

```python
def NASA_UQ_Function(input_file_path, nsamples, nconditions):
    # This section needs to be written as a function that executes the function or model being used
    exe_path = os.path.abspath(Path(__file__).parent / "local_model_windows.exe")  # Path to the executable in the current folder
    command = [exe_path, input_file_path]
    print('Simulation executable has been called.')
    result = subprocess.run(command, capture_output=True, text=True)
    output_file_path = 'Y_out.csv'
    df = pd.read_csv(output_file_path, header=None)
    print(f'Simulation output data loaded from {output_file_path}')

    sample_indices = df[6].unique()  # Extract unique sample indices
    num_samples = len(sample_indices)
    df = df.drop(columns=[6])  # Drop the sample index column
    # df.to_csv(f'./{folder}/Y_out_Level{Iteration}_{repetitions}.csv',header=False,index=False) #stores the full output file (optional)
    Y_out = df.to_numpy().reshape(num_samples, 60, 6).transpose(1, 2, 0)

    epsilon = 1e-14 #Small constant to avoid division by zero
    y4y5 = (Y_out[10,3,:]/(Y_out[10,4,:]+epsilon)) #Get the simulation y ratios for all samples.
    y4y6 = (Y_out[10,3,:]/(Y_out[10,5,:]+epsilon))
    y5y6 = (Y_out[10,4,:]/(Y_out[10,5,:]+epsilon))
    
    y4y5 = np.reshape(y4y5,[nsamples,nconditions]) #Reshape each y threshold matrix into one where all conditions are in one line.
    y4y6 = np.reshape(y4y6,[nsamples,nconditions])
    y5y6 = np.reshape(y5y6,[nsamples,nconditions])

    y_out = np.empty([3,nsamples,nconditions])
    y_out[0] = y4y5
    y_out[1] = y4y6
    y_out[2] = y5y6

    return y_out
```
From this function we return an array of shape (3,nsamples,nconditions) which we can index by condition and sample used to match against the necessary filter condition.

## Filter Process

Now that we have the required data, we can start the filter process. First, we create an empty array for samples that pass the current threshold along with a counter of the 'accepted' samples. Both of these variables are reset to empty and 0 respectively when a new threshold iteration is started.
All of the samples, which includes all of the conditions for that sample, are looped through until each sample is 'accepted' into a threshold set, or is found to not fit into any threshold.
```python
if repetitions == 0: #When a new Iteration is reached, resets the accepted sample matrix and count.
    filtered_samples = np.empty([nsamples,9])
    count = 0
for i in range(nsamples): #Looping through all samples
    test_threshold = thresholds[Iteration]
    test_Level = Iteration
    point_added = False
    while point_added is not True: #While loop finds which Iteration the sample belongs to.
        threshold_set = np.empty([len(conditions['values']),nconditions])
        for j in range(len(conditions['values'])):
            threshold_set[j] = abs(Y_out[j,i,:] - conditions['values'][j]) <= test_threshold

        if threshold_set[:,:].all() == 1:
            if test_Level == Iteration: #If sample is accepted at the same Iteration that it starts, it is added to the matrix for updating the Xe bounds
                filtered_samples[count] = X_input[nconditions*i]
                count += 1
            point_added = True
            for s in index:
                with open(f'./{folder}/samples.csv', "a") as a:                 a.write(str(X_input[nconditions*i,s])+',')
                with open(f'./{folder}/Iteration{test_Level}.csv', "a") as a:   a.write(str(X_input[nconditions*i,s])+',')
                with open(f'./{folder}/samples.csv', "a") as a:                 a.write(f'{test_Level}' + ',' + '\n')
                with open(f'./{folder}/Iteration{test_Level}.csv', "a") as a:   a.write(f'{test_Level}' + ',' + '\n')
        else:
            test_Level = test_Level - 1
            test_threshold = thresholds[test_Level] #Moving the radius down a Iteration (can be changed out to an array of radii instead)
            if test_Level <= 0: #If sample is not within the start radius for all controls, the while loop is broken.
                for s in index:
                    with open(f'./{folder}/samples.csv', "a") as a:                 a.write(str(X_input[nconditions*i,s])+',')
                    with open(f'./{folder}/Iteration{test_Level}.csv', "a") as a:   a.write(str(X_input[nconditions*i,s])+',')
                    with open(f'./{folder}/samples.csv', "a") as a:                 a.write(f'{test_Level}' + ',' + '\n')
                    with open(f'./{folder}/Iteration{test_Level}.csv', "a") as a:   a.write(f'{test_Level}' + ',' + '\n')
                break
```

## Updating the bounds for the filtered variables

After the filter process is completed, if more than 1 (this can be changed if necessary) sample has been 'accepted' during the process, we can update the bounds for the filtered variables and move to the next iteration in the filter process.

However, if only 1 or no samples were accepted, we repeat the current iteration unless this iteration has already been repeated enough times equal to the 'max_repetitions' condition, which breaks the filter process early. This break condition being triggered signifies that the filter process is unable to find enough samples that meet the threshold condition to be able to update the possible bounds for the variables being filtered.
```python
print(count)
if count > 1: #If more than one sample is accepted at the current Iteration, new bounds are computed and Iteration goes up.
    filtered_samples.resize(count,len(variables['names']))

    for i in range(nconditions):
        for s in index:
            variables[conditions['names'][i]][s] = np.min(filtered_samples[:,s]), np.max(filtered_samples[:,s])
            if i == 0:
                print(f'New Bounds for {variables['names'][s]}: {np.min(filtered_samples[:,s]), np.max(filtered_samples[:,s])}')
    for s in index:
        with open(f'./{folder}/bounds_all_min.txt', "a") as a:
            a.write(str(variables[conditions['names'][i]][s][0])+', ')
        with open(f'./{folder}/bounds_all_max.txt', "a") as a:
            a.write(str(variables[conditions['names'][i]][s][1])+', ')
    with open(f'./{folder}/bounds_all_min.txt', "a") as a:
        a.write(f' str({Iteration})' + ', ' + f'{thresholds[Iteration]}' + ', ' +'\n')
    with open(f'./{folder}/bounds_all_max.txt', "a") as a:
        a.write(f'str({Iteration})' + ', ' + f'{thresholds[Iteration]}' + ', ' +'\n')

    total_repetitions += repetitions  # Add repetitions from current Iteration to total
    repetitions = 0 #Reset repetitions when new bounds are found.
    Iteration += 1 #Increase Iteration (and by extension, decrease radius being tested)
    if Iteration > max_Iteration: #If max_Iteration is reached, no more radii are available so while loop is broken.
        print(f'All control points intersect within the radius {thresholds[max_Iteration]} with {max_Iteration} levels of reduction and {max_Iteration+total_repetitions} iterations of {nsamples} samples.')
        break
        
else: #Otherwise the process gets repeated and the number of repetitions at the current Iteration is increased.
    if repetitions >= max_repetitions: #If Iteration is repeated max_repetitions amount of times, while loop is broken.
        total_repetitions += repetitions
        print(f'All control points lack viable points below radius: {thresholds[Iteration]} at Iteration {Iteration} with {Iteration+total_repetitions} iterations of {nsamples} samples.')
        break
    repetitions += 1
```