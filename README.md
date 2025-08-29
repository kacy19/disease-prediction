# disease-prediction

# Disease Prediction 

A simple toolkit for predicting types of chest pain using machine learning models.  
Includes data preprocessing, model training, evaluation, visualization, and unit tests.


## Project Structure

Disease Prediction 1/
├── data/
│   └── heart\_dataset.csv        # Dataset file
├── scripts/
│   ├── preprocessing.py         # Data loading and preprocessing
│   ├── models.py                # ML models definitions
│   ├── evaluation.py            # Model evaluation & metrics
│   ├── visualize.py             # Visualization functions (ROC, Confusion matrix)
│   └── run\_pipeline.py          # Main script to run training and evaluation
├── results/                     # Output folder for saved plots
├── tests/
│   └── test\_preprocessing.py    # Unit tests for preprocessing
└── README.md                    # This file


## Setup Instructions

1. Clone the repository
   bash
   git clone <your-repo-url>
   cd Disease-Prediction-Toolkit


2. Create and activate a Python environment (optional but recommended)

   bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
  

3. Install required dependencies

   bash
   pip install -r requirements.txt

   If `requirements.txt` is not available, install manually:

   bash
   pip install pandas scikit-learn matplotlib seaborn

4. Place the dataset
   Ensure `heart_dataset.csv` is placed inside the `data/` folder.

## Usage

### Run full pipeline

To preprocess data, train models, and evaluate:

bash
python scripts/run_pipeline.py

### Run unit tests

To test data preprocessing:

bash
python -m unittest discover tests

## Scripts Overview

* scripts/preprocessing.py: Loads and preprocesses data.
* scripts/models.py: Defines ML models.
* scripts/evaluation.py: Evaluates models and saves metrics and confusion matrices.
* scripts/visualize.py: Contains visualization utilities.
* scripts/run_pipeline.py: Runs the full training and evaluation pipeline.
* tests/test_preprocessing.py: Unit tests for preprocessing.

## Results

Evaluation plots will be saved in the `results/` directory after running the pipeline.

