import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def precision_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=5):
    """
    Calculate precision@k
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        k: Cutoff for precision calculation
        
    Returns:
        Precision@k score
    """
    precisions = []
    
    # Group test data by user
    user_groups = test_df.groupby('user_idx')
    
    for user_idx, group in user_groups:
        # Get positive items for this user
        pos_items = group[group['label'] == 1]['item_idx'].values
        
        if len(pos_items) == 0:
            continue
            
        # Get all items for this user
        all_items = group['item_idx'].values
        
        # Predict scores for all items
        scores = model.predict(
            user_ids=np.repeat(user_idx, len(all_items)),
            item_ids=all_items,
            user_features=user_features,
            item_features=item_features
        )
        
        # Get top k items based on predicted scores
        top_k_items = all_items[np.argsort(-scores)[:k]]
        
        # Calculate precision for this user
        precision = len(np.intersect1d(top_k_items, pos_items)) / k
        precisions.append(precision)
    
    return np.mean(precisions)

def recall_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=5):
    """
    Calculate recall@k
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        k: Cutoff for recall calculation
        
    Returns:
        Recall@k score
    """
    recalls = []
    
    # Group test data by user
    user_groups = test_df.groupby('user_idx')
    
    for user_idx, group in user_groups:
        # Get positive items for this user
        pos_items = group[group['label'] == 1]['item_idx'].values
        
        if len(pos_items) == 0:
            continue
            
        # Get all items for this user
        all_items = group['item_idx'].values
        
        # Predict scores for all items
        scores = model.predict(
            user_ids=np.repeat(user_idx, len(all_items)),
            item_ids=all_items,
            user_features=user_features,
            item_features=item_features
        )
        
        # Get top k items based on predicted scores
        top_k_items = all_items[np.argsort(-scores)[:k]]
        
        # Calculate recall for this user
        recall = len(np.intersect1d(top_k_items, pos_items)) / len(pos_items)
        recalls.append(recall)
    
    return np.mean(recalls)

def ndcg_at_k(model, test_df, n_users, n_items, user_features=None, item_features=None, k=10):
    """
    Calculate NDCG@k
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        k: Cutoff for NDCG calculation
        
    Returns:
        NDCG@k score
    """
    ndcgs = []
    
    # Group test data by user
    user_groups = test_df.groupby('user_idx')
    
    for user_idx, group in user_groups:
        # Get items and labels for this user
        items = group['item_idx'].values
        labels = group['label'].values
        
        if sum(labels) == 0:
            continue
            
        # Predict scores for all items
        scores = model.predict(
            user_ids=np.repeat(user_idx, len(items)),
            item_ids=items,
            user_features=user_features,
            item_features=item_features
        )
        
        # Sort by scores
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]
        
        # Truncate to top k
        sorted_labels = sorted_labels[:k]
        
        # Calculate DCG
        dcg = np.sum((2**sorted_labels - 1) / np.log2(np.arange(2, len(sorted_labels) + 2)))
        
        # Calculate ideal DCG
        ideal_labels = np.sort(labels)[::-1][:k]
        idcg = np.sum((2**ideal_labels - 1) / np.log2(np.arange(2, len(ideal_labels) + 2)))
        
        # Calculate NDCG
        if idcg > 0:
            ndcg = dcg / idcg
            ndcgs.append(ndcg)
    
    return np.mean(ndcgs)

def evaluate_model(model, test_df, n_users, n_items, user_features=None, item_features=None):
    """
    Evaluate a model on multiple metrics
    
    Args:
        model: Trained LightFM model
        test_df: Test DataFrame with user_idx, item_idx, label
        n_users: Number of users
        n_items: Number of items
        user_features: User features (optional)
        item_features: Item features (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'precision@5': precision_at_k(model, test_df, n_users, n_items, user_features, item_features, k=5),
        'recall@5': recall_at_k(model, test_df, n_users, n_items, user_features, item_features, k=5),
        'recall@10': recall_at_k(model, test_df, n_users, n_items, user_features, item_features, k=10),
        'ndcg@10': ndcg_at_k(model, test_df, n_users, n_items, user_features, item_features, k=10)
    }
    
    return metrics

def plot_learning_curves(train_metrics, test_metrics, metric_names, epochs, model_name):
    """
    Plot learning curves for multiple metrics
    
    Args:
        train_metrics: List of training metrics
        test_metrics: List of test metrics
        metric_names: List of metric names
        epochs: List of epoch numbers
        model_name: Name of the model for plot title
    """
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(15, 5))
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        ax.plot(epochs, [m[metric] for m in train_metrics], 'b-', label='Train')
        ax.plot(epochs, [m[metric] for m in test_metrics], 'r-', label='Test')
        ax.set_title(f'{metric} - {model_name}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_learning_curves.png')
    plt.show() 
