## Prerequisites

- Python 3.9 (make sure to use this version)
- `pip` (Python package manager)

## Setting up the environment

### Step 1: Clone the repository

Start by cloning the repository to your local machine.

### Step 2 (optional): Create a virtual environment with Python 3.9

It is strongly **recommended** to create a virtual environment to isolate your project dependencies. Make sure you have Python 3.9 installed, as this is the required version for this project.

```
python3.9 -m venv venv
```

### Step 3: Activate the virtual environment

```
source venv/bin/activate
```

### Step 4: Install the required dependencies

After activating the virtual environment, install the project dependencies by running:

```
pip install -r requirements.txt
```

This will install all the necessary libraries specified in the `requirements.txt` file.

### Step 5: Run `setup.py` to create the necessary directories

To properly set up the directory structure, run the `setup.py` script:

```
python3 setup.py
```

This will automatically create the necessary directories for the project.

After running the script, you need to populate the directories with the following files:

- **'queries' directory**: Place the following files here:

  - `robust04.qrels`
  - `robust04.topics`

- **'tipster' directory**: Insert the following file:

  - `inquery.stoplist`

- **'tipster/corpora_unzipped' directory**: Load your Tipster collection here. There is no specific naming or structure required for the files. Disk 4 and 5 are sufficient for now.

Make sure you have these files in the specified directories for the project to run correctly.

### Step 6: Run the project

Now that the environment is set up, you can start using the project.

First of all start elasticsearch and kibana containers with:

```
docker compose up -d
```

Then everything is set up and the code can be run with:

```
python3 main.py
```

The project allows you to run the script with multiple command-line arguments for different operations. You can use the following options:

- **`-i` or `--index`**:  
  Use this option to index files. This will start the process of indexing the documents.

- **`-d` or `--delete-index`**:  
  Use this option if you want to delete the existing index before re-indexing the files.

- **`-e` or `--evaluate`**:  
  This option allows you to evaluate the performance of the search engine. It will run the trec_eval process along with a tailor made evaluation to check relevant documents for each qrel.

- **`-v` or `--verbose`**:  
  Use this option to print step-by-step results during the execution of the script. It provides detailed logs of each action being performed.

- **`-s` or `--simulate`**:  
  This option simulates a random query and retrieves only a subset of the relevant documents for that query.
