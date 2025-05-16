import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from collections import defaultdict

def derive_implicit_labels(df, watch_ratio_threshold=0.8):
    """
    Derive implicit feedback labels based on watch ratio
    
    Args:
        df: DataFrame with interaction data
        watch_ratio_threshold: Threshold for positive interactions
        
    Returns:
        DataFrame with added label column
    """
    df['label'] = (df['watch_ratio'] >= watch_ratio_threshold).astype(int)
    return df

def filter_interactions(df, min_positive_interactions=3):
    """
    Filter users and items with at least min_positive_interactions
    
    Args:
        df: DataFrame with interaction data and labels
        min_positive_interactions: Minimum positive interactions required
        
    Returns:
        Filtered DataFrame
    """
    # Count positive interactions per user and item
    user_pos_counts = df[df['label'] == 1]['user_id'].value_counts()
    item_pos_counts = df[df['label'] == 1]['video_id'].value_counts()
    
    # Filter users and items with enough positive interactions
    valid_users = user_pos_counts[user_pos_counts >= min_positive_interactions].index
    valid_items = item_pos_counts[item_pos_counts >= min_positive_interactions].index
    
    # Apply the filter
    filtered_df = df[df['user_id'].isin(valid_users) & df['video_id'].isin(valid_items)]
    
    print(f"Original interactions: {len(df)}")
    print(f"Filtered interactions: {len(filtered_df)}")
    print(f"Unique users: {len(valid_users)}")
    print(f"Unique items: {len(valid_items)}")
    
    return filtered_df, valid_users, valid_items

def create_user_item_maps(users, items):
    """
    Create mappings between IDs and indices
    
    Args:
        users: List of user IDs
        items: List of item IDs
        
    Returns:
        Tuple of mappings (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
    """
    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for i, u in enumerate(users)}
    item_to_idx = {i: j for j, i in enumerate(items)}
    idx_to_item = {j: i for j, i in enumerate(items)}
    
    return user_to_idx, idx_to_user, item_to_idx, idx_to_item

def leave_n_out_split(df, user_to_idx, item_to_idx, test_ratio=0.2, neg_ratio=4, test_neg_ratio=99, random_state=42):
    """
    Split data using leave-n-out strategy
    
    Args:
        df: DataFrame with interaction data
        user_to_idx: Mapping from user IDs to indices
        item_to_idx: Mapping from item IDs to indices
        test_ratio: Proportion of positive interactions to use for testing
        neg_ratio: Number of negative samples per positive in training
        test_neg_ratio: Number of negative samples per positive in testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with train/test matrices and data
    """
    np.random.seed(random_state)
    
    # Create dictionaries to store positives and all interactions per user
    user_positives = defaultdict(list)
    user_all_items = defaultdict(list)
    
    # Fill dictionaries
    for _, row in df.iterrows():
        user_idx = user_to_idx[row['user_id']]
        item_idx = item_to_idx[row['video_id']]
        
        user_all_items[user_idx].append(item_idx)
        if row['label'] == 1:
            user_positives[user_idx].append(item_idx)
    
    # Initialize data structures for train/test split
    train_data = []
    test_data = []
    
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    
    # For each user
    for user_idx in range(n_users):
        positives = user_positives[user_idx]
        
        if not positives:
            continue
        
        # Split user's positives into train and test
        train_pos, test_pos = train_test_split(
            positives, test_size=test_ratio, random_state=random_state + user_idx
        )
        
        # Add train positives
        for item_idx in train_pos:
            train_data.append((user_idx, item_idx, 1))
        
        # Sample train negatives (4 per positive)
        all_items = set(range(n_items))
        known_items = set(user_all_items[user_idx])
        unknown_items = list(all_items - known_items)
        
        # Sample train negatives
        if unknown_items:
            n_train_neg = len(train_pos) * neg_ratio
            if len(unknown_items) > n_train_neg:
                train_neg = np.random.choice(unknown_items, size=n_train_neg, replace=False)
            else:
                train_neg = unknown_items
                
            for item_idx in train_neg:
                train_data.append((user_idx, item_idx, 0))
        
        # Add test positives and sample test negatives
        for item_idx in test_pos:
            test_data.append((user_idx, item_idx, 1))
            
        # Sample test negatives (99 per positive)
        if unknown_items:
            n_test_neg = len(test_pos) * test_neg_ratio
            
            # If we need more negatives than available, sample with replacement
            if len(unknown_items) > n_test_neg:
                test_neg = np.random.choice(unknown_items, size=n_test_neg, replace=False)
            else:
                test_neg = np.random.choice(unknown_items, size=n_test_neg, replace=True)
                
            for item_idx in test_neg:
                test_data.append((user_idx, item_idx, 0))
    
    # Convert to DataFrames
    train_df = pd.DataFrame(train_data, columns=['user_idx', 'item_idx', 'label'])
    test_df = pd.DataFrame(test_data, columns=['user_idx', 'item_idx', 'label'])
    
    # Create sparse matrices for LightFM
    train_mat = sparse.coo_matrix(
        (np.ones(len(train_df[train_df['label'] == 1])), 
         (train_df[train_df['label'] == 1]['user_idx'], train_df[train_df['label'] == 1]['item_idx'])),
        shape=(n_users, n_items)
    )
    
    test_mat = sparse.coo_matrix(
        (np.ones(len(test_df[test_df['label'] == 1])), 
         (test_df[test_df['label'] == 1]['user_idx'], test_df[test_df['label'] == 1]['item_idx'])),
        shape=(n_users, n_items)
    )
    
    print(f"Training interactions: {len(train_df)}")
    print(f"Testing interactions: {len(test_df)}")
    
    return {
        'train_interactions': train_mat,
        'test_interactions': test_mat,
        'train_df': train_df,
        'test_df': test_df,
        'n_users': n_users,
        'n_items': n_items
    }

def prepare_item_features(item_categories_df, item_to_idx):
    """
    Prepare item features for LightFM
    
    Args:
        item_categories_df: DataFrame with item categories
        item_to_idx: Mapping from item IDs to indices
        
    Returns:
        Sparse matrix with item features
    """
    # One-hot encode item categories
    from sklearn.preprocessing import OneHotEncoder
    
    # Extract the categories as a list
    item_cats = item_categories_df.copy()
    
    # Keep only the items that are in our filtered dataset
    item_cats = item_cats[item_cats['video_id'].isin(item_to_idx.keys())]
    
    # Create item feature matrix
    item_features = []
    item_indices = []
    feature_indices = []
    
    # Create a mapping from category to feature index
    all_categories = set()
    for cats in item_cats['categories']:
        categories = cats.split(';')
        all_categories.update(categories)
    
    cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
    n_features = len(cat_to_idx)
    
    # Populate the feature matrix
    for _, row in item_cats.iterrows():
        if row['video_id'] not in item_to_idx:
            continue
            
        item_idx = item_to_idx[row['video_id']]
        categories = row['categories'].split(';')
        
        for cat in categories:
            item_indices.append(item_idx)
            feature_indices.append(cat_to_idx[cat])
            item_features.append(1.0)
    
    # Create sparse matrix
    item_features_mat = sparse.coo_matrix(
        (item_features, (item_indices, feature_indices)),
        shape=(len(item_to_idx), n_features)
    )
    
    return item_features_mat

def prepare_user_features(user_features_df, user_to_idx):
    """
    Prepare user features for LightFM
    
    Args:
        user_features_df: DataFrame with user features
        user_to_idx: Mapping from user IDs to indices
        
    Returns:
        Sparse matrix with user features
    """
    # Select the columns to one-hot encode
    cols_to_encode = ['user_active_degree', 'is_live_streamer', 'follow_user_num_range']
    
    # Keep only the users that are in our filtered dataset
    user_feat = user_features_df.copy()
    user_feat = user_feat[user_feat['user_id'].isin(user_to_idx.keys())]
    
    # Initialize lists for COO matrix
    user_indices = []
    feature_indices = []
    feature_values = []
    
    # Create mappings for categorical features
    feature_map = {}
    current_idx = 0
    
    for col in cols_to_encode:
        unique_values = user_feat[col].unique()
        feature_map[col] = {val: idx + current_idx for idx, val in enumerate(unique_values)}
        current_idx += len(unique_values)
    
    n_features = current_idx
    
    # Fill in the feature matrix
    for _, row in user_feat.iterrows():
        if row['user_id'] not in user_to_idx:
            continue
            
        user_idx = user_to_idx[row['user_id']]
        
        for col in cols_to_encode:
            if pd.notna(row[col]):  # Check for NaN values
                feat_idx = feature_map[col][row[col]]
                user_indices.append(user_idx)
                feature_indices.append(feat_idx)
                feature_values.append(1.0)
    
    # Create sparse matrix
    user_features_mat = sparse.coo_matrix(
        (feature_values, (user_indices, feature_indices)),
        shape=(len(user_to_idx), n_features)
    )
    
    return user_features_mat 
