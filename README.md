# Model #1
For our first model, we implemented a Linear Regression model to predict the Earnings-to-Price (E/P) ratio of companies based on key financial metrics such as Return on Assets (ROA), Return on Equity (ROE), Debt-to-Equity ratio, and Sales Growth. The workflow includes data preprocessing, model training, evaluation, and visualization of results.


# Directory and Relevant Files

### `/data/`
Contains all datasets used in the project.

- **`/data/raw/`**
  - **`compustat_data.csv`** — Original raw dataset containing company-level financial metrics extracted from Compustat.

- **`/data/processed/ - Midterm dataset`**
  - **`train.parquet`** — Preprocessed training dataset used for model fitting.  
  - **`test.parquet`** — Preprocessed testing dataset used for model evaluation.
 
- **`/data/finalParquet_leakfree/ - Final dataset`**
  - **`train.parquet`** — Final leakfree preprocessed training dataset used for model fitting.  
  - **`test.parquet`** — Final leakfree preprocessed testing dataset used for model evaluation.

---

### `/models/`
Contains Python scripts related to model implementation and visualization.

- **`/models/LinearRegression/LinearRegressionNew.py`** — Trains and evaluates a Linear Regression model on the processed data, outputting performance metrics (MAE, RMSE, R²).  
- **`/models/LinearRegression/visualization_new.ipynb`** — Generates visualizations such as actual vs. predicted plots, residual plots, and performance metric comparisons.
- **`/models/GB/gradientBoosting.py`** — Trains and evaluates a Gradient Boosting model on the processed data to capture non-linear patterns, outputting performance metrics (MAE, RMSE, R²).
- **`/models/GB/GBvisualization.ipynb`** — Generates visualizations for the Gradient Boosting model, including feature importance ranking and error distribution plots.
- **`/models/RF/RandomForest.py`** — Trains and evaluates a Random Forest ensemble model on the processed data, outputting performance metrics (MAE, RMSE, R²).
- **`/models/RF/RFvisualization.ipynb`** — Generates visualizations for the Random Forest model, including actual vs. predicted plots and feature contribution analysis.
---

### `/preprocessing.py - Midterm`
Handles data cleaning, feature selection, and transformation of the raw dataset into train and test parquet files.

### `/scripts/final_preprocessing_leakfree.py - Final`
Implements the complete, leak-free data pipeline. Unlike previous versions, this script strictly separates train and test sets before calculating any statistics.


# Conda Environment Setup

To set up the Python environment for this project:

1. Create the environment from the provided file:

   ```zsh
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```zsh
   conda activate <env_name>
   ```

   Replace `<env_name>` with the name specified in `environment.yml` (see the first line of the file).

**Do not push the `.conda/` folder to GitHub.**
It is ignored via `.gitignore`. Share your environment using `environment.yml` instead.

# Data Preprocessing Pipeline

Run the preprocessing script from the project root once the raw Compustat export is in `data/raw/compustat_data.csv`:

```zsh
python preprocessing.py
```

The script cleans the raw fundamentals data, engineers ratios, performs a date-based train/test split, scales features, and writes two Parquet files to `data/processed/`:

- `train.parquet`
- `test.parquet`

You should see a short summary with the save locations and row counts after it completes. Rerun the script any time new raw data arrives.

# Inspect Processed Parquet Files

To quickly inspect the contents of the generated Parquet files, use the helper script:

```zsh
python scripts/view_parquet.py
```

It prints the first few rows and shape of both train and test splits, which is useful for sanity checks before modeling.

# ML Project Proposal Website
