# Virtual environments

Operating system (Ubuntu 22.04.5 LTS):

```shell
lsb_release -a

# No LSB modules are available.
# Distributor ID: Ubuntu
# Description:    Ubuntu 22.04.5 LTS
# Release:        22.04
# Codename:       jammy
```

# For Python

I assume that Python 3 is already installed on your operating system. If not, please install it first.

Below is an example from [How to Install Python on Linux](https://www.geeksforgeeks.org/how-to-install-python-on-linux/)

```shell
# Check if Python is installed
# Open your terminal window using the shortcut key Ctrl + Alt + T, then execute:
python3 --version # should see something like "Python 3.10.12" if installed

# If Python is not installed, install it
sudo apt-get update
sudo apt-get install python3
sudo apt install python3-pip
sudo apt install python3-dev python3-venv build-essential
```

## General 

This environment was used for data processing and analysis.

```shell
# Change the repository path according to your file system:
cd /scratch/$USER/smartphone-insect-detect

# Creates folder 'envs' only if it doesn't exist
mkdir -p envs

# Create a new virtual environment for data processing and analysis.
# Do not execute the commands below if the virtual environment was already set
# and the packages were already installed.
python3 -m venv envs/general
# Note, if you need to delete an environment, you can do so by deleting the folder:
# rm -rf path/to/environment
# Example:
# rm -rf envs/general

# Activate the virtual environment to install packages
source ./envs/general/bin/activate 

pip install -r envs/requirements_general.txt

# This is needed to make the virtual environment visible to Jupyter Notebook 
# (e.g. in Visual Studio Code)
python3 -m ipykernel install --user --name=general --display-name="Python 3 (general)"

deactivate
```

In case when the virtual environment is not visible to a jupyter notebook via VS Code, you can manually add the path to the kernel:

- Open the command palette (Ctrl + Shift + P)
- Type "Python: Select Interpreter"
- Select "Enter interpreter path"
- Enter the path to the Python executable in the virtual environment, e.g. `/scratch/$USER/smartphone-insect-detect/envs/general/bin/python`


## YOLOv5

### Example on our GPU workstation
```shell
cd /scratch/$USER/smartphone-insect-detect
python3 -m venv envs/yolov5

# Clone the repository
cd detectors
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Check out the commit that was used in the project
git checkout 050c72cbba9ca24dcff103fca54caf21d7e8b527

# Double check that the repository state is at that commit
git log -1 --format="%H %cd"
# The output should be:
# 050c72cbba9ca24dcff103fca54caf21d7e8b527 Thu Jan 18 17:53:34 2024 +0100

# Activate the virtual environment
source ../../envs/yolov5/bin/activate

# Install the dependencies from the time of that commit
pip install -r requirements.txt

# And to make the virtual environment visible to Jupyter Notebook
python3 -m ipykernel install --user --name=yolov5

deactivate
```

### Example on the Uni Leipzig GPU cluster

Note that this required a certain precompiled software package maintained by the system administrators.
Jupyter Notebooks were not used here.

```shell
# SSH-ed into the cluster, then:
cd ~/smartphone-insect-detect

# Unload all modules to have a clean start
module purge
# Load toolchain maintained by the system administrators
module load Python/3.10.8-GCCcore-12.2.0

python -m venv envs/yolov5
source ./envs/yolov5/bin/activate 

pip3 install torch torchvision 
# NOTE: command taken from https://pytorch.org/get-started/locally/
# torchaudio is listed there, but not on yolov5/requirements.txt, and not needed

pip install -r ./detectors/yolov5/requirements.txt

deactivate
```

## YOLOv7

### Example on our GPU workstation

```shell
cd /scratch/$USER/smartphone-insect-detect
python3 -m venv envs/yolov7

# Clone the repository
cd detectors
git clone https://github.com/WongKinYiu/yolov7
cd yolov7

# Check out the commit that was used in the project
git checkout a207844b1ce82d204ab36d87d496728d3d2348e7

# Double check that the repository state is at that commit
git log -1 --format="%H %cd"
# The output should be:
# a207844b1ce82d204ab36d87d496728d3d2348e7 Fri Nov 3 10:05:01 2023 +0800

# Activate the virtual environment
source ../../envs/yolov7/bin/activate

# Install the dependencies from the time of that commit
pip install -r requirements.txt

# And to make the virtual environment visible to Jupyter Notebook
python3 -m ipykernel install --user --name=yolov7

deactivate
```

## SAHI

SAHI - [Slicing Aided Hyper Inference](https://docs.ultralytics.com/guides/sahi-tiled-inference/)

On the GPU workstation:

```shell
# Change the repository path according to your file system:
cd /scratch/$USER/smartphone-insect-detect

# Do not execute the commands below if the virtual environment was already set
# and the packages were already installed.
python3 -m venv envs/sahi

# Activate the virtual environment to install packages
source ./envs/sahi/bin/activate 

pip install -r envs/requirements_sahi.txt

# This is needed to make the virtual environment visible to Jupyter Notebook 
# (e.g. in Visual Studio Code)
python3 -m ipykernel install --user --name=sahi --display-name="Python 3 (sahi)"

deactivate
```

Note that due to updates in `numpy`, you could get an error like this:

> ValueError: Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead. If the context of this error is of the form `arr[nonzero(cond)]`, just use `arr[cond]`. 

Therefore, for back compatibility with `pycocotools`, this environment solved the issue for me:

```shell
# Change the repository path according to your file system:
cd /scratch/$USER/smartphone-insect-detect

# Do not execute the commands below if the virtual environment was already set
# and the packages were already installed.
python3 -m venv envs/pycocotools

# Activate the virtual environment to install packages
source ./envs/pycocotools/bin/activate 

# https://pypi.org/project/pycocotools/#history
pip install pycocotools==2.0.7 # Aug 14 2023
# https://pypi.org/project/numpy/#history
pip install numpy==1.25.2 # Jul 31 2023
# https://pypi.org/project/pandas/#history
pip install pandas==2.0.3 # Jun 29 2023
# https://pypi.org/project/GitPython/#history
pip install gitpython==3.1.32 # Jul 10 2023

deactivate
```

# For R

[RStudio](https://posit.co/downloads/) and [RStudio Server](https://posit.co/download/rstudio-server/) where used for this project together with [R for statistical computing and graphics](https://www.r-project.org/).

The environment (project library) for R was managed with the [renv](https://github.com/rstudio/renv) package (renv_1.1.4).

```r
install.packages("renv")
renv::init()
```

This created the folder 'renv' and the 'renv.lock' file under this repository. 
The 'renv.lock' file contains details about each R package used and their dependencies including the R version used.

To reproduce the R environment, in an R session, run `renv::restore()`. 
This command will read the 'renv.lock' file and install all the packages specified in it, using the exact versions.

`renv::snapshot()` was used to update the `renv.lock` file.
