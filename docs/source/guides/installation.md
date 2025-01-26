# Installation

## Virtual environment

Set up your Python3 virtual environment to safely install the dependencies. For details, refer to this [guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for `venv` or this [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) if you prefer `Conda`.

On MacOS/Linux:

```shell
python3 -m venv myenv 

source myenv/bin/activate 
```

```{attention}
**Requirement:**
- It requires Python >=3.10 
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

## Questions

Raise an issue on the GitHub page if you run into problems.
