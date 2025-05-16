# KuaiRec 2.0 Recommender System

This repository implements a two-stage recommender system pipeline for the KuaiRec 2.0 dataset, comparing a baseline collaborative filtering model with a hybrid model that incorporates side features.

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Setup

1. Clone the repository
2. Set up a virtual environment (recommended)
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset files should be organized in the following structure:

```
sys-recommenders/
  KuaiRec2.0/
    data/
      small_matrix.csv         # Interaction data for prototyping (~4.7M rows)
      big_matrix.csv           # Complete interaction data (~12.5M rows)
      item_categories.csv      # Item category data
      user_features.csv        # User feature data
```

## Running the Pipeline

To run the pipeline with the small matrix (for prototyping):

```bash
python main.py --matrix small_matrix.csv
```

To run with the big matrix (for final evaluation):

```bash
python main.py --matrix big_matrix.csv
```

### Command Line Arguments

- `--matrix`: Specifies which interaction matrix to use (default: `small_matrix.csv`)
- `--epochs`: Number of training epochs (default: `100`)
- `--eval_every`: Evaluate model every N epochs (default: `10`)
- `--data_dir`: Custom data directory path (optional, overrides default path)

## Project Structure

- `loaddata.py`: Data loading utilities
- `preprocess.py`: Data preprocessing functions
- `evaluation.py`: Evaluation metrics implementation
- `main.py`: Main pipeline orchestration
- `report.md`: Report template for summarizing findings

## Pipeline Steps

1. Load and explore data
2. Derive implicit labels (positive if watch_ratio ≥ 0.8)
3. Filter users and items with ≥ 3 positive interactions
4. Split data into train/test sets (leave-N-out strategy)
5. Train and evaluate baseline model (LightFM with BPR loss)
6. Train and evaluate hybrid model (LightFM with side features)
7. Compare results and plot learning curves

## Output

The pipeline produces:
- Learning curve plots saved as PNG files
- Detailed metrics in the console output
- Results can be used to update the `report.md` file
