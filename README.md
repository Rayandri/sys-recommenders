# KuaiRec 2.0 Recommender System

Ce projet implémente un système de recommandation à deux étapes pour le dataset KuaiRec 2.0, comparant un modèle de filtrage collaboratif de référence avec un modèle hybride qui incorpore des caractéristiques secondaires.

## Installation

```bash
# Créer un environnement virtuel Python
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Exécution optimisée

Pour une exécution optimisée, plusieurs options sont disponibles:

### Mode rapide (échantillonnage et évaluations réduites)

```bash
python main.py --fast --epochs 20 --eval_every 5 --test_neg_ratio 10
```

### Configuration des threads

```bash
# Spécifier manuellement le nombre de threads (utile sur serveurs avec beaucoup de cœurs)
python main.py --threads 8 --epochs 50
```

### Mode standard avec paramètres optimisés

```bash
python main.py --epochs 50 --eval_every 10 --test_neg_ratio 20
```

### Mode complet (lent, mais résultats plus précis)

```bash
python main.py --epochs 100 --test_neg_ratio 99
```

## Résolution des problèmes courants

- **Erreur de mémoire**: Réduire `test_neg_ratio` (défaut: 20) et utiliser le mode `--fast`
- **Performance lente**: Augmenter le nombre de threads, réduire `eval_every` et utiliser le mode `--fast`
- **Précision insuffisante**: Augmenter `epochs` et `test_neg_ratio`, désactiver le mode `--fast`

## Exécution sur différents matériels

### CPU multi-cœurs
- Augmenter les threads selon les capacités du CPU (4-8 threads généralement optimal)
- Utiliser `--threads X` où X est ~80% des cœurs disponibles

### Serveur de calcul
- Utiliser toute la mémoire disponible pour des ratios de test plus élevés
- Augmenter les threads selon les capacités du serveur

### Machine virtuelle / Cloud
- Choisir une instance avec plus de vCPUs
- Ajouter swap si la mémoire est limitée
- Réduire `test_neg_ratio` pour économiser la mémoire

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
