# Installation

## Virtual environment

Set up your Python3 virtual environment to safely install the dependencies. For details, refer to this [guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for `venv` or this [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) if you prefer `Conda`. For [Spyder](https://www.spyder-ide.org) users, [specify the path to the  the environment interpreter](https://youtu.be/3ELzEG5_haU?si=FZoQ7qtQra-Iro_T) will get you started.

On MacOS/Linux:

```shell
python3 -m venv myenv 

source myenv/bin/activate 
```

```{attention}
**Requirement:**
- It requires Python >=3.11
```

## Install using pip

`PyUncertainNumber` can be installed from [PyPI](https://pypi.org/project/pyuncertainnumber/). Upon activation of your virtual environment, use your terminal:

```shell
pip install pyuncertainnumber
```

Verify the installation using the snippet below:

```shell
python3 -c "import pyuncertainnumber as pun; print(pun.norm([0,12],[1,4]))"
```

## Dependencies

Refer to the `requirements.txt` file from the [GitHub repository](https://github.com/leslieDLcy/PyUncertainNumber/) to see the requirements. Note these will be automatically installed during the `pip` install process of the package. There is no need to manually install them.


## Contributions

If you feel like make contributions to the developement, which is highly encouraged, have a fork on GitHub, `git` clone it, `cd` to the project root directory and install it in editable mode. Once you are happy, `pull request` your changes.

```shell
pip install -e .
```

## Questions

Raise an issue on the GitHub page if you run into problems.
