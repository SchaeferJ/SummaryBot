# Multilingual document summarization and topic clustering

### About
This repository contains the code accompanying the project _Document summarisation and topic clustering in a multi-lingual environment_
submitted as an individual project for the Mannheim Master in Data Science.

### Instructions
The code is written in Python and R, so make sure to have them both
installed. For Python I strongly recommend using the Anaconda distribution
as it comes with many poupular data science libraries pre-nstalled. Get it
[here](https://www.anaconda.com/).

Project-wide configurations (like model URLs, padding lengths, etc.) are stored
in the `config.yml` file and can be freely changed to fit your needs.

When running the instructions below for the first time, the script will offer you to download
the required embeddings. Depending on which and how many languages you summarize, this
can be several (>30) gigabyte of data, so make sure you are connected to an unmetered 
broadband internet connection.

####1: Install dependencies
If you are using pip the required python packages can be installed by opening 
a terminal in the current directory and entering 
 `pip install -r requirements.txt `
 
To install the required R packages, open R and enter
```r
install.packages(c("httr","rvest","dplyr","stringr","R.utils","jsonlite"))
```
**NOTE:** If installing the R packages fails on Linux you probabably have to install
system-wide dependencies. Look through the terminal output of R to identify the missing
dependencies and install them with your system's package manager.

####2: Generate the raw data
Go to the `./webscraper/` directory and run all `XXX_Scraper.R` scripts to retrieve
the raw articles. In the meantime you can download the processed CNN dataset from [here](https://github.com/abisee/cnn-dailymail)
as well as the BBC news summary dataset from [here](https://www.kaggle.com/pariza/bbc-news-summary). Extract
both datasets into the `./webscraper/` directory. Once the scrapers are complete, run
`Dataset_Generator.R` and move the resulting files `Topic_Data.csv` and `Raw_Data.csv` to the
main directory.

####3: Generate Test- and Training-Data
Go back to the main directory and run

`python3 Make_Training_Data.py`

####4: Train the CRSum Model
You now have all the data you need to train the CRSum model. You can either write
your own code or use the example provided in the jupyter notebook 

`Train_CRSum.ipynb`

####5: Run the experiments
Run the experiments provided in the jupyter  notebooks or
write your own ones. All Python code is commented and documented and provides
a minimal working example at the end of the script to make adapting the code as easy
as possible.