"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based
"""
import sys
sys.path.insert(1, './src/')


import config
from pathlib import Path
from doit.tools import run_once
import platform

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --to python ./src/{notebook}.ipynb --output _{notebook}.py --output-dir {build_dir}"
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
# fmt: on


def get_os():
    os_name = platform.system()
    if os_name == "Windows":
        return "windows"
    elif os_name == "Darwin":
        return "nix"
    elif os_name == "Linux":
        return "nix"
    else:
        return "unknown"
    
os_type = get_os()


def copy_notebook_to_folder(notebook_stem, origin_folder, destination_folder):
    origin_path = Path(origin_folder) / f"{notebook_stem}.ipynb"
    destination_path = Path(destination_folder) / f"_{notebook_stem}.ipynb"
    if os_type == "nix":
        command =  f"cp {origin_path} {destination_path}"
    else:
        command = f"copy  {origin_path} {destination_path}"
    return command


def task_pull_CRSP_Stock():
    """
    Pull CRSP data from WRDS and save to disk
    """
    file_dep = [
        "./src/config.py", 
        "./src/load_CRSP_stock.py",
        ]
    targets = [
        Path(DATA_DIR) / "pulled" / file for file in 
        [
            ## src/load_CRSP_stock.py
            "CRSP_stock.parquet", 
        ]
    ]

    return {
        "actions": [
            "ipython src/config.py",
            "ipython src/load_CRSP_stock.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
        "verbosity": 2, # Print everything immediately. This is important in
        # case WRDS asks for credentials.
    }


def task_pull_FF_industry():
    """
    Pull 48 industry portfolio data from the Fama/French Data Library 
    and save to disk
    """
    file_dep = ["./src/config.py", "./src/load_FF_industry.py"]
    file_output = ["FF_portfolios_value_weighted.parquet", 
                   "FF_portfolios_equal_weighted.parquet"]
    targets = [DATA_DIR / "pulled" / file for file in file_output]

    return {
        "actions": [
            "ipython src/config.py",
            "ipython ./src/load_FF_industry.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_pull_vix():
    """
    Pull vix data from FRED and save to disk
    """
    file_dep = ["./src/config.py", "./src/load_vix.py"]
    file_output = ["vix.parquet"]
    targets = [DATA_DIR / "pulled" / file for file in file_output]

    return {
        "actions": [
            "ipython src/config.py",
            "ipython ./src/load_vix.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_clean_CRSP_stock():
    """
    Clean CRSP data and save to disk
    """
    file_dep = ["./src/config.py", "./src/clean_CRSP_stock.py"]
    file_output = ["CRSP_closing_price.parquet", "CRSP_midpoint.parquet"]
    targets = [DATA_DIR / "pulled" / file for file in file_output]

    return {
        "actions": [
            "ipython src/config.py",
            "ipython ./src/clean_CRSP_stock.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }


def task_replicate_table_1():
    """ 
    Construct reversal strategy,
    replicate Table 1: Summary Statistics of Reversal Strategy Returns
    """
    file_dep = ["./src/config.py", "./src/calc_reversal_strategy.py"]
    file_output = ["reversal_return_2010.parquet", "reversal_return_2023.parquet", 
                   "Table_1A.parquet", "Table_1B.parquet", 
                   "Table_1A.parquet_reproduce", "Table_1B_reproduce.parquet"]
    targets = [DATA_DIR / "derived" / file for file in file_output]

    return {
        "actions": [
            "ipython src/config.py",
            "ipython ./src/calc_reversal_strategy.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }

def task_replicate_table_2():
    """ 
    Replicate Table 2: Predicting Reversal Strategy Returns with VIX
    """
    file_dep = ["./src/config.py", "./src/regression_hac.py"]
    targets = [DATA_DIR / "derived/Table_2.parquet", OUTPUT_DIR / "Table_2.tex"]

    return {
        "actions": [
            "ipython src/config.py",
            "ipython ./src/regression_hac.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }

def task_additional_analysis():
    """ 
    Generate additional analysis table and figure
    """
    file_dep = ["./src/config.py", "./src/additional_analysis.py"]
    file_output = ["Additional_Table.tex", "reversal_strategy_and_vix.png"]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "ipython src/config.py",
            "ipython ./src/additional_analysis.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }



def task_convert_notebooks_to_scripts():
    """Preps the notebooks for presentation format.
    Execute notebooks with summary stats and plots and remove metadata.
    """
    build_dir = Path(OUTPUT_DIR)
    build_dir.mkdir(parents=True, exist_ok=True)

    notebooks = [
        "notebook.ipynb",
    ]
    file_dep = [Path("./src") / file for file in notebooks]
    stems = [notebook.split(".")[0] for notebook in notebooks]
    targets = [build_dir / f"_{stem}.py" for stem in stems]

    actions = [
        # *[jupyter_execute_notebook(notebook) for notebook in notebooks_to_run],
        # *[jupyter_to_html(notebook) for notebook in notebooks_to_run],
        *[jupyter_clear_output(notebook) for notebook in stems],
        *[jupyter_to_python(notebook, build_dir) for notebook in stems],
    ]
    return {
        "actions": actions,
        "targets": targets,
        "task_dep": [],
        "file_dep": file_dep,
        "clean": True,
    }


def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks with summary stats and plots and remove metadata.
    """
    notebooks = [
        "notebook.ipynb",
    ]
    stems = [notebook.split(".")[0] for notebook in notebooks]

    file_dep = [
        # 'load_other_data.py',
        *[Path(OUTPUT_DIR) / f"_{stem}.py" for stem in stems],
    ]

    targets = [
        ## Notebooks converted to HTML
        *[OUTPUT_DIR / f"{stem}.html" for stem in stems],
    ]

    actions = [
        *[jupyter_execute_notebook(notebook) for notebook in stems],
        *[jupyter_to_html(notebook) for notebook in stems],
        *[copy_notebook_to_folder(notebook, Path("./src"), OUTPUT_DIR) for notebook in stems],
        *[copy_notebook_to_folder(notebook, Path("./src"), "./docs") for notebook in stems],
        *[jupyter_clear_output(notebook) for notebook in stems],
        # *[jupyter_to_python(notebook, build_dir) for notebook in notebooks_to_run],
    ]
    return {
        "actions": actions,
        "targets": targets,
        "task_dep": [],
        "file_dep": file_dep,
        "clean": True,
    }




def task_compile_latex_docs():
    """compile the latex documents to pdfs."""
    file_dep = [
        "./reports/report.tex",
    ]
    file_output = [
        "./reports/report.pdf",
    ]
    targets = [file for file in file_output]

    return {
        "actions": [
            "latexmk -xelatex -cd ./reports/report.tex",  # Compile
            "latexmk -xelatex -c -cd ./reports/report.tex",  # Clean
            # "latexmk -CA -cd ../reports/",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }