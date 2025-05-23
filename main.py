import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import lightfm
import os
import json

from loaddata import load_interaction_data, load_item_categories, load_user_features, print_dataset_info
from preprocess import (
    derive_implicit_labels, filter_interactions, create_user_item_maps,
    leave_n_out_split, prepare_item_features, prepare_user_features
)
from evaluation import evaluate_model, plot_learning_curves

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
NUM_THREADS = mp.cpu_count()

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = SCRIPT_DIR / "KuaiRec2.0" / "data"

def train_baseline_model(train_data, test_data, epochs=100, eval_every=10, patience=5):
    """
    Train a baseline LightFM model using only user-item interactions
    
    Args:
        train_data: Dictionary with training data
        test_data: Dictionary with test data
        epochs: Number of training epochs
        eval_every: Evaluate every N epochs
        patience: Number of evaluations with no improvement after which training will stop
        
    Returns:
        Trained model and evaluation metrics
    """
    print("\n" + "="*50)
    print("Training Baseline Model (LightFM with WARP loss)")
    print("="*50)
    
    model = lightfm.LightFM(
        loss="warp",
        no_components=256,
        learning_rate=0.03,
        user_alpha=0.0005,
        item_alpha=0.0005,
        max_sampled=150,
        random_state=RANDOM_SEED
    )
    
    train_metrics = []
    test_metrics = []
    epochs_list = []
    
    start_time = time.time()
    
    best_score = -float('inf')
    best_epoch = 0
    best_model = None
    no_improvement = 0
    
    progress_bar = tqdm(range(1, epochs + 1), desc="Training baseline model")
    for epoch in progress_bar:
        model.fit_partial(
            train_data['train_interactions'],
            epochs=1,
            num_threads=NUM_THREADS
        )
        
        if epoch % eval_every == 0 or epoch == epochs:
            progress_bar.set_description(f"Evaluating at epoch {epoch}")
            
            train_metric = evaluate_model(
                model, 
                train_data['train_df'], 
                train_data['n_users'], 
                train_data['n_items']
            )
            
            test_metric = evaluate_model(
                model, 
                test_data['test_df'], 
                test_data['n_users'], 
                test_data['n_items']
            )
            
            train_metrics.append(train_metric)
            test_metrics.append(test_metric)
            epochs_list.append(epoch)
            
            current_score = test_metric['f1@5']
            
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_model = model.get_params()
                no_improvement = 0
            else:
                no_improvement += 1
                
            progress_bar.set_postfix({
                'train_f1@5': f"{train_metric['f1@5']:.4f}",
                'test_f1@5': f"{test_metric['f1@5']:.4f}",
                'best_f1@5': f"{best_score:.4f}",
                'no_improv': no_improvement
            })
            
            if no_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch was {best_epoch} with f1@5 = {best_score:.4f}")
                break
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")
    
    if best_model is not None and no_improvement >= patience:
        print("Restoring best model...")
        model = lightfm.LightFM(**best_model)
    
    print("\nFinal metrics:")
    print(f"  Train: {train_metrics[-1]}")
    print(f"  Test:  {test_metrics[-1]}")
    
    return model, train_metrics, test_metrics, epochs_list, training_time

def train_hybrid_model(train_data, test_data, user_features, item_features, epochs=100, eval_every=10, patience=5):
    """
    Train a hybrid LightFM model using user-item interactions and side features
    
    Args:
        train_data: Dictionary with training data
        test_data: Dictionary with test data
        user_features: User features matrix
        item_features: Item features matrix
        epochs: Number of training epochs
        eval_every: Evaluate every N epochs
        patience: Number of evaluations with no improvement after which training will stop
        
    Returns:
        Trained model and evaluation metrics
    """
    print("\n" + "="*50)
    print("Training Hybrid Model (LightFM with user and item features)")
    print("="*50)
    
    model = lightfm.LightFM(
        loss="warp",
        no_components=256,
        learning_rate=0.03,
        item_alpha=0.0005,
        user_alpha=0.0005,
        max_sampled=150,
        random_state=RANDOM_SEED
    )
    
    train_metrics = []
    test_metrics = []
    epochs_list = []
    
    start_time = time.time()
    
    best_score = -float('inf')
    best_epoch = 0
    best_model = None
    no_improvement = 0
    
    progress_bar = tqdm(range(1, epochs + 1), desc="Training hybrid model")
    for epoch in progress_bar:
        model.fit_partial(
            train_data['train_interactions'],
            user_features=user_features,
            item_features=item_features,
            epochs=1,
            num_threads=NUM_THREADS
        )
        
        if epoch % eval_every == 0 or epoch == epochs:
            progress_bar.set_description(f"Evaluating at epoch {epoch}")
            
            train_metric = evaluate_model(
                model, 
                train_data['train_df'], 
                train_data['n_users'], 
                train_data['n_items'],
                user_features,
                item_features
            )
            
            test_metric = evaluate_model(
                model, 
                test_data['test_df'], 
                test_data['n_users'], 
                test_data['n_items'],
                user_features,
                item_features
            )
            
            train_metrics.append(train_metric)
            test_metrics.append(test_metric)
            epochs_list.append(epoch)
            
            current_score = test_metric['f1@5']
            
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_model = model.get_params()
                no_improvement = 0
            else:
                no_improvement += 1
                
            progress_bar.set_postfix({
                'train_f1@5': f"{train_metric['f1@5']:.4f}",
                'test_f1@5': f"{test_metric['f1@5']:.4f}",
                'best_f1@5': f"{best_score:.4f}",
                'no_improv': no_improvement
            })
            
            if no_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch was {best_epoch} with f1@5 = {best_score:.4f}")
                break
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")
    
    if best_model is not None and no_improvement >= patience:
        print("Restoring best model...")
        model = lightfm.LightFM(**best_model)
    
    print("\nFinal metrics:")
    print(f"  Train: {train_metrics[-1]}")
    print(f"  Test:  {test_metrics[-1]}")
    
    return model, train_metrics, test_metrics, epochs_list, training_time

def run_pipeline(matrix_file, item_categories_file, user_features_file, epochs=300, eval_every=10, 
                patience=8, test_neg_ratio=49, fast_mode=False, model='all'):
    """
    Run the complete recommender system pipeline
    
    Args:
        matrix_file: Interaction data file
        item_categories_file: Item categories file
        user_features_file: User features file
        epochs: Number of training epochs
        eval_every: Evaluate every N epochs
        patience: Number of evaluations with no improvement for early stopping
        test_neg_ratio: Negative sampling ratio for test set (default 49, lower for faster execution)
        fast_mode: If True, use a smaller dataset and skip some evaluations
        model: Which model(s) to train: baseline, hybrid, or all
    """
    print(f"Using {NUM_THREADS} CPU threads")
    print(f"Data directory: {DATA_DIR}")
    
    if fast_mode:
        print("FAST MODE ENABLED: Using reduced dataset and evaluations")
    
    matrix_path = DATA_DIR / matrix_file
    item_categories_path = DATA_DIR / item_categories_file
    user_features_path = DATA_DIR / user_features_file
    
    if not matrix_path.exists():
        raise FileNotFoundError(f"Interaction data file not found: {matrix_path}")
    if not item_categories_path.exists():
        raise FileNotFoundError(f"Item categories file not found: {item_categories_path}")
    if not user_features_path.exists():
        raise FileNotFoundError(f"User features file not found: {user_features_path}")
    
    print(f"\nLoading data from {matrix_file}...")
    interactions_df = load_interaction_data(matrix_path)
    
    print(f"Loading item categories...")
    item_categories_df = load_item_categories(item_categories_path)
    
    print(f"Loading user features...")
    user_features_df = load_user_features(user_features_path)
    
    print(f"\nInteraction matrix shape: {interactions_df.shape}")
    print(f"Item categories shape: {item_categories_df.shape}")
    print(f"User features shape: {user_features_df.shape}")
    
    print("\nDeriving implicit labels (watch_ratio >= 0.7)...")
    interactions_df = derive_implicit_labels(interactions_df)
    positive_ratio = interactions_df['label'].mean()
    print(f"Positive interactions ratio: {positive_ratio:.4f}")
    
    print("\nFiltering users and items with >= 3 positive interactions...")
    filtered_df, valid_users, valid_items = filter_interactions(interactions_df)
    
    # Échantillonner les données en mode rapide
    if fast_mode and len(valid_users) > 500:
        sampled_users = np.random.choice(valid_users, 500, replace=False)
        filtered_df = filtered_df[filtered_df['user_id'].isin(sampled_users)]
        valid_users = sampled_users
        print(f"Fast mode: Sampled down to {len(valid_users)} users")
    
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = create_user_item_maps(valid_users, valid_items)
    
    print("\nSplitting data (leave-n-out)...")
    split_data = leave_n_out_split(
        filtered_df, 
        user_to_idx, 
        item_to_idx, 
        test_ratio=0.2, 
        neg_ratio=8, 
        test_neg_ratio=test_neg_ratio, 
        random_state=RANDOM_SEED
    )
    
    baseline_model = hybrid_model = None
    baseline_train_metrics = baseline_test_metrics = baseline_epochs = baseline_time = None
    hybrid_train_metrics = hybrid_test_metrics = hybrid_epochs = hybrid_time = None
    metric_names = ['precision@5', 'recall@5', 'f1@5', 'recall@10', 'ndcg@10', 'item_coverage@10', 'f1@20', 'f1@50']

    if model in ('baseline', 'all'):
        baseline_model, baseline_train_metrics, baseline_test_metrics, baseline_epochs, baseline_time = train_baseline_model(
            split_data, 
            split_data, 
            epochs=epochs, 
            eval_every=eval_every,
            patience=patience
        )
        print("\nPlotting learning curves for baseline model...")
        plot_learning_curves(
            baseline_train_metrics, 
            baseline_test_metrics, 
            metric_names, 
            baseline_epochs, 
            'Baseline Model'
        )

    if model in ('hybrid', 'all'):
        print("\nPreparing item and user features...")
        item_features_mat = prepare_item_features(item_categories_df, item_to_idx)
        user_features_mat = prepare_user_features(user_features_df, user_to_idx)
        hybrid_model, hybrid_train_metrics, hybrid_test_metrics, hybrid_epochs, hybrid_time = train_hybrid_model(
            split_data, 
            split_data, 
            user_features_mat, 
            item_features_mat, 
            epochs=epochs, 
            eval_every=eval_every,
            patience=patience
        )
        print("\nPlotting learning curves for hybrid model...")
        plot_learning_curves(
            hybrid_train_metrics, 
            hybrid_test_metrics, 
            metric_names, 
            hybrid_epochs, 
            'Hybrid Model'
        )

    if model == 'all':
        final_baseline = baseline_test_metrics[-1]
        final_hybrid = hybrid_test_metrics[-1]
        print("\n" + "="*50)
        print("Final Results Comparison")
        print("="*50)
        for k in [5, 10, 20, 50]:
            metric = f'f1@{k}'
            if metric in final_baseline and metric in final_hybrid:
                print(f"{metric}:")
                print(f"  Baseline: {final_baseline[metric]:.4f}")
                print(f"  Hybrid:   {final_hybrid[metric]:.4f}")
                improvement = (final_hybrid[metric] - final_baseline[metric]) / final_baseline[metric] * 100
                print(f"  Improvement: {improvement:.2f}%")
        print("\nPrecision at different cutoffs:")
        for k in [5, 10, 20, 50]:
            metric = f'precision@{k}'
            if metric in final_baseline and metric in final_hybrid:
                print(f"{metric}:")
                print(f"  Baseline: {final_baseline[metric]:.4f}")
                print(f"  Hybrid:   {final_hybrid[metric]:.4f}")
        print("\nRecall at different cutoffs:")
        for k in [5, 10, 20, 50]:
            metric = f'recall@{k}'
            if metric in final_baseline and metric in final_hybrid:
                print(f"{metric}:")
                print(f"  Baseline: {final_baseline[metric]:.4f}")
                print(f"  Hybrid:   {final_hybrid[metric]:.4f}")
        print("\nNDCG at different cutoffs:")
        for k in [5, 10, 20, 50]:
            metric = f'ndcg@{k}'
            if metric in final_baseline and metric in final_hybrid:
                print(f"{metric}:")
                print(f"  Baseline: {final_baseline[metric]:.4f}")
                print(f"  Hybrid:   {final_hybrid[metric]:.4f}")
        if 'item_coverage@10' in final_baseline:
            print("\nOther metrics:")
            print(f"Item Coverage@10:")
            print(f"  Baseline: {final_baseline['item_coverage@10']:.4f}")
            print(f"  Hybrid:   {final_hybrid['item_coverage@10']:.4f}")
        if 'diversity@10' in final_baseline:
            print(f"Diversity@10:")
            print(f"  Baseline: {final_baseline['diversity@10']:.4f}")
            print(f"  Hybrid:   {final_hybrid['diversity@10']:.4f}")
        print("\nTraining times:")
        print(f"  Baseline: {baseline_time:.2f} seconds")
        print(f"  Hybrid:   {hybrid_time:.2f} seconds")
    elif model == 'baseline':
        final_baseline = baseline_test_metrics[-1]
        print("\nFinal Baseline Model Metrics:")
        for k in [5, 10, 20, 50]:
            metric = f'f1@{k}'
            if metric in final_baseline:
                print(f"{metric}: {final_baseline[metric]:.4f}")
        print(f"Training time: {baseline_time:.2f} seconds")
    elif model == 'hybrid':
        final_hybrid = hybrid_test_metrics[-1]
        print("\nFinal Hybrid Model Metrics:")
        for k in [5, 10, 20, 50]:
            metric = f'f1@{k}'
            if metric in final_hybrid:
                print(f"{metric}: {final_hybrid[metric]:.4f}")
        print(f"Training time: {hybrid_time:.2f} seconds")
    # Prepare results dictionary
    results_dict = {
        'matrix_file': matrix_file,
        'baseline_metrics': baseline_test_metrics[-1] if baseline_test_metrics else None,
        'hybrid_metrics': hybrid_test_metrics[-1] if hybrid_test_metrics else None,
        'baseline_time': baseline_time,
        'hybrid_time': hybrid_time,
        'epochs': epochs,
        'test_neg_ratio': test_neg_ratio,
        'timestamp': time.strftime("%Y%m%d-%H%M%S")
    }
    
    # Save results to JSON file
    results_file = f"results_{matrix_file.replace('.csv', '')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    print(f"\nResults saved to {results_file}")
    
    return {
        'baseline_model': baseline_model,
        'hybrid_model': hybrid_model,
        'baseline_metrics': baseline_test_metrics[-1] if baseline_test_metrics else None,
        'hybrid_metrics': hybrid_test_metrics[-1] if hybrid_test_metrics else None,
        'baseline_time': baseline_time,
        'hybrid_time': hybrid_time,
        'results_file': results_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the recommender system pipeline')
    parser.add_argument('--matrix', type=str, default='small_matrix.csv', 
                        help='Interaction matrix file (small_matrix.csv or big_matrix.csv)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate every N epochs')
    parser.add_argument('--patience', type=int, default=8, 
                        help='Number of evaluations with no improvement before early stopping')
    parser.add_argument('--data_dir', type=str, default=None, 
                        help='Custom data directory path (overrides default)')
    parser.add_argument('--test_neg_ratio', type=int, default=49, 
                        help='Negative sampling ratio for test set (lower for faster execution)')
    parser.add_argument('--fast', action='store_true', 
                        help='Run in fast mode with reduced dataset and evaluations')
    parser.add_argument('--threads', type=int, default=None, 
                        help='Number of threads to use (default: auto-detected)')
    parser.add_argument('--model', type=str, default='all', choices=['baseline', 'hybrid', 'all'],
                        help='Which model(s) to train: baseline, hybrid, or all')
    
    args = parser.parse_args()
    
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        print(f"Using custom data directory: {DATA_DIR}")
    
    if args.threads:
        NUM_THREADS = args.threads
        print(f"Using {NUM_THREADS} threads as specified")
    
    results = run_pipeline(
        args.matrix,
        'item_categories.csv',
        'user_features.csv',
        epochs=args.epochs,
        eval_every=args.eval_every,
        patience=args.patience,
        test_neg_ratio=args.test_neg_ratio,
        fast_mode=args.fast,
        model=args.model
    ) 
