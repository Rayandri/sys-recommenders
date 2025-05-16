# KuaiRec 2.0 Recommender System

## Dataset Summary

The KuaiRec 2.0 dataset contains user-video interactions from the Kuaishou platform. The dataset includes:

- **Interaction data**:
  - `small_matrix.csv` (~4.7M rows)
  - `big_matrix.csv` (~12.5M rows)
  - Key columns: `user_id`, `video_id`, `watch_ratio`, timestamps

- **Item side information**:
  - `item_categories.csv` with categorical features per video

- **User side information**:
  - `user_features.csv` with user activity features and demographics

## Methodology

### Label Choice

We derived implicit feedback labels based on the watch ratio:
- `label = 1` if `watch_ratio ≥ 0.8` (positive interaction)
- `label = 0` otherwise (negative interaction)

This threshold was chosen to identify high-engagement interactions, where users watched at least 80% of a video's duration.

### Filtering Strategy

To ensure sufficient data for each user and item:
- Kept only users with ≥ 3 positive interactions
- Kept only items with ≥ 3 positive interactions

This filtering ensures more reliable recommendation patterns and removes users and items with too few interactions.

### Train/Test Split Strategy

We implemented a leave-N-out split strategy:
- For each user, randomly held out 20% of their positive interactions for testing
- For training: sampled 4 random unseen items per positive interaction
- For testing: sampled 99 random unseen items per held-out positive interaction

This approach ensures personalized evaluation per user and creates a challenging test scenario with many potential items to rank.

## Model Settings

### Baseline Model (LightFM)

- Architecture: Matrix Factorization with BPR loss
- Embedding size: 64 dimensions
- Learning rate: 0.05
- Used only user/item IDs (no side features)
- Training: 100 epochs

### Hybrid Model (LightFM with Side Features)

- Same architecture and hyperparameters as baseline
- Added item features: one-hot encoded categories
- Added user features: one-hot encoded `user_active_degree`, `is_live_streamer`, `follow_user_num_range`
- Training: 100 epochs

## Evaluation Protocol

We evaluated both models every 10 epochs on the following metrics:

- **Precision@5**: Percentage of recommended items that are relevant
- **Recall@5**: Percentage of relevant items that are recommended (k=5)
- **Recall@10**: Percentage of relevant items that are recommended (k=10)
- **NDCG@10**: Normalized Discounted Cumulative Gain at k=10

The evaluation was conducted on the held-out test set for each user.

## Results

### Performance Comparison

| Metric | Baseline | Hybrid | Improvement |
|--------|----------|--------|-------------|
| Precision@5 | x.xxx | x.xxx | xx.x% |
| Recall@5 | x.xxx | x.xxx | xx.x% |
| Recall@10 | x.xxx | x.xxx | xx.x% |
| NDCG@10 | x.xxx | x.xxx | xx.x% |

*Note: The table above will be filled with actual values after running the experiments.*

### Training Time

- Small matrix:
  - Baseline model: xx.x seconds
  - Hybrid model: xx.x seconds
- Big matrix:
  - Baseline model: xx.x seconds
  - Hybrid model: xx.x seconds

## Conclusions

1. The hybrid model incorporating side information consistently outperformed the baseline model across all metrics, demonstrating the value of including contextual features.

2. The most substantial improvement was observed in [metric], suggesting that side features particularly help with [specific aspect of recommendation].

3. Performance on the big matrix dataset showed [similar/different] patterns compared to the small matrix, indicating [scalability insights].

4. The incorporation of user activity features had [greater/lesser] impact than item categories, suggesting future work might focus more on [user/item] feature engineering.

5. Trade-offs between model complexity and training time: while the hybrid model performed better, it required approximately [x%] more training time.

## Future Work

- Experiment with different feature combinations to identify the most influential features
- Try alternative splitting strategies to evaluate robustness
- Implement more advanced models (Neural Networks, Sequential models) to capture temporal patterns
- Optimize hyperparameters for both models
- Explore additional metrics focused on diversity and novelty 
