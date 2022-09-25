Patrick Miller

repository location: https://github.com/patrickdmiller/switrs_fhs_analysis


## Preparing Data
### SWITRS:
Either download the out.csv from my public google drive: [link](https://drive.google.com/drive/u/0/folders/1YWLXP5TTw7KsqrkTNv2vmQvlzHxbdx0T) or download the sqlite file and run extract_all_serious_injury.py from the ```/source``` directory. pass ```--help``` flag for details on data transformation script. Warning: the sqlite file is 10GB

Data sourced from [alexgude](https://alexgude.com/blog/switrs-sqlite-hosted-dataset/)

### FHS:
download from [kaggle](https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data), no transformation scripts are required. 

## Running Experiments

All analysis was performed using Jupter Notebooks. The notebooks with inline are all included in the source directory. 

Both experiment Notebooks rely on a data Notebook that includes utility functions to undersample/oversample and filter data. 

raw python versions of the notebooks can be found in ```/source/python_raw```

### SWITRS :
* switrs experiments are in *switrs_experiments* Notebook. The Notebook imports *switrs_data* Notebook and is used throughout. 

* *switrs_data* will require you to edit the location of out.csv you either downloaded or generated (see preparing data section above)

* Please note, this notebook requires sklearnex but can be commented out to use vanilla sklearn.



### FHS :
* fhs experiments are in *fhs_experiments* Notebook. The Notebook imports *fhs_data* Notebook.

* *fhs_data* will require you to edit the location of out.csv you downloaded from kaggle. 

### Other Charts:

* learning_curves are built in the experiment Notebooks but hyperparameter tuning charts were built in the *charts* Notebook based on output data from the experiment Notebooks. 