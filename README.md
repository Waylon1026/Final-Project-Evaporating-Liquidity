Evaporating Liquidity
==================

# About this project

In this project, we replicate tables from the paper "Evaporating Liquidity" by Stefan Nagel using the Principals of Reproducible Analytical Pipelines (RAPs) learned in the class. 

Our replication is automated from end-to-end using Pydoit, formatted using the project template (blank_project) provided by professor Bejarano, which is based on the Cookiecutter Data Science template.

# Task List

- LaTeX document: Zhiyuan Liu, Junhan Fu
- Jupyter notebook: Sifei Zhao, Ruilong Guo
- Table 1: Sifei Zhao, Ruilong Guo
    - Replicate
    - Reproduce
    - Unit tests
    - .env, requirements.txt
    - make commits to repo; pull requests
    - docstring of python file and python function
- Table 2: Zhiyuan Liu, Junhan Fu
    - Replicate
    - Reproduce
    - Unit tests
    - .env, requirements.txt
    - make commits to repo; pull requests
    - docstring of python file and python function
  
- Other summary statistics tables and charts outside replication: Ruilong Guo
- Data cleaning (Tidy data set): Everyone
- PyDoit (dodo.py): Sifei Zhao


# General Directory Structure

 - The `assets` folder is used for things like hand-drawn figures or other pictures that were not generated from code. These things cannot be easily recreated if they are deleted.

 - The `output` folder, on the other hand, contains tables and figures that are generated from code. The entire folder should be able to be deleted, because the code can be run again, which would again generate all of the contents.

 - I'm using the `doit` Python module as a task runner. It works like `make` and the associated `Makefile`s. To rerun the code, install `doit` (https://pydoit.org/) and execute the command `doit` from the `src` directory. Note that doit is very flexible and can be used to run code commands from the command prompt, thus making it suitable for projects that use scripts written in multiple different programming languages.

 - I'm using the `.env` file as a container for absolute paths that are private to each collaborator in the project. You can also use it for private credentials, if needed. It should not be tracked in Git.


# Data and Output Storage

I'll often use a separate folder for storing data. I usually write code that will pull the data and save it to a directory in the data folder called "pulled"  to let the reader know that anything in the "pulled" folder could hypothetically be deleted and recreated by rerunning the PyDoit command (the pulls are in the dodo.py file).

I'll usually store manually created data in the "assets" folder if the data is small enough. Because of the risk of manually data getting changed or lost, I prefer to keep it under version control if I can.

Output is stored in the "output" directory. This includes tables, charts, and rendered notebooks. When the output is small enough, I'll keep this under version control. I like this because I can keep track of how tables change as my analysis progresses, for example.

Of course, the data directory and output directory can be kept elsewhere on the machine. To make this easy, I always include the ability to customize these locations by defining the path to these directories in environment variables, which I intend to be defined in the `.env` file, though they can also simply be defined on the command line or elsewhere. The `config.py` is reponsible for loading these environment variables and doing some like preprocessing on them. The `config.py` file is the entry point for all other scripts to these definitions. That is, all code that references these variables and others are loading by importing `config`.