# NDVI Differencing Tool (Python)

## What it is

This is a tool created while learning Python for GIS applications. The script finds the change between two dates in the Normalised Difference Vegetation Index (NDVI) within a given area, and produces a map of the results.

---

## Installation

### 1. Prepare for install

The following instructions assume you will be using **git** and **conda**. Directions for installing git can be found [here](https://git-scm.com/downloads "Git - Downloads"), and for Anaconda [here](https://docs.anaconda.com/anaconda/install/ "Installation - Anaconda documentation").

### 2. Clone the repository with git

Open **Git Bash** ( **Start** > **Git** > **Git Bash** ) and navigate to the directory you want to use for the repository. Execute the following: `git clone https://github.com/ejleighton/ejl-pyproj.git`. After some internet wizardry you should have a copy of the repository in a folder named **ejl-pyproj**.

Alternatively, if you do not wish to use git, click the green *"Code"* button above and select *"Download ZIP"*. Once complete, unzip the folder into your desired directory.

### 3. Create a conda environment

The folder installed in step 2 contains an environment.yml file which can be used to reproduce the required environment to run the script.

Using Anaconda Navigator the environment can easily be imported by going to the **Environments** panel and clicking the **Import** button at the bottom, then navigating to the environment.yml file from the import wizard.

To create the environment from the command line (Windows users should run **Anaconda Prompt** from **Start**), first navigate to the directory you unpacked/cloned the repository to and run: `conda env create -f environment.yml`.

This will install the following packages and their dependencies:

- python 3.8.8
- cartopy 0.18.0
- rasterio 1.2.0
- geopandas 0.9.0
- scikit-image 0.18.1

## Using the tool

### 1. Configuration

To run the script using your own data you will need to create a myconfig.py file. Example configs are provided in the repository to run with the test data. Details on how to set up the config file are provided in the user guide.

### 2. Run the script!

Open a command prompt (Windows: **Anaconda Prompt**), navigate to the repository directory, and execute the following: `python ndvidiff.py`.
