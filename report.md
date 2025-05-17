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
- `label = 1` if `watch_ratio ≥ 0.7` (positive interaction)
- `label = 0` otherwise (negative interaction)

This threshold was chosen to identify high-engagement interactions, where users watched at least 70% of a video's duration.

### Filtering Strategy

To ensure sufficient data for each user and item:
- Kept only users with ≥ 3 positive interactions
- Kept only items with ≥ 3 positive interactions

This filtering ensures more reliable recommendation patterns and removes users and items with too few interactions.

### Train/Test Split Strategy

We implemented a leave-n-out split strategy:
- For each user, randomly held out 20% of their positive interactions for testing
- For training: sampled 8 random unseen items per positive interaction
- For testing: sampled up to 99 random unseen items per held-out positive interaction (configurable via `test_neg_ratio`)

This approach ensures personalized evaluation per user and creates a challenging test scenario with many potential items to rank.

## Model Settings

### Baseline Model (LightFM)

- Architecture: Matrix Factorization with WARP loss
- Embedding size: 256 dimensions
- Learning rate: 0.03
- Regularization: alpha=0.0005 for both user and item embeddings
- Max sampled: 150
- Used only user/item IDs (no side features)
- Early stopping with patience of 5-8 evaluations
- Training: 100-300 epochs (configurable)

### Hybrid Model (LightFM with Side Features)

- Same architecture and hyperparameters as baseline
- Added item features: one-hot encoded categories with L2 normalization
- Added user features: 
  - Numerical: `follow_user_num`, `fans_user_num`, `friend_user_num`, `register_days` (RobustScaler normalized)
  - Categorical: `user_active_degree`, `follow_user_num_range`, `fans_user_num_range`, `friend_user_num_range`, `register_days_range`

## Evaluation Protocol

The system evaluates both models on multiple ranking metrics at different cutoffs (k=5,10,20,50):

- **Precision@k**: Percentage of recommended items that are relevant
- **Recall@k**: Percentage of relevant items that are recommended
- **F1@k**: Harmonic mean of precision and recall
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Item Coverage@10**: Percentage of total items that appear in any user's top-10 recommendations
- **Diversity@10**: Average pairwise distance between recommended items using embeddings

The evaluation is conducted on the held-out test set for each user.

## Results

### Performance Comparison

| Metric | Baseline | Hybrid | Improvement |
|--------|----------|--------|-------------|
| Precision@5 | x.xxx | x.xxx | xx.x% |
| Recall@5 | x.xxx | x.xxx | xx.x% |
| F1@5 | x.xxx | x.xxx | xx.x% |
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

1. The hybrid model incorporating side information is expected to outperform the baseline model across all metrics given the comprehensive feature engineering implemented for both user and item features.

2. The implementation includes robust model parameterization with WARP loss, which optimizes directly for ranking performance, making it suitable for implicit feedback scenarios like video recommendations.

3. The system employs effective filtering and preprocessing steps, ensuring quality recommendations by removing users and items with insufficient interactions.

4. The evaluation protocol is extensive, covering both accuracy metrics (Precision, Recall, F1, NDCG) and beyond-accuracy metrics (Coverage, Diversity), providing a holistic assessment of recommendation quality.

5. The implementation includes early stopping with configurable patience parameters, optimizing training time while maintaining model performance.

6. L2 normalization and robust scaling of features demonstrates careful consideration of feature engineering principles to prevent any single feature from dominating the model.

## Future Work

- Experiment with different watch ratio thresholds (currently set at 0.7) to analyze sensitivity to the definition of positive interactions
- Implement sequential recommendation models to leverage temporal patterns in user viewing behavior
- Add session-based recommendations to capture short-term preferences
- Explore neural network architectures (NCF, VAE) as alternatives to matrix factorization
- Incorporate content-based features from video metadata
- Implement cross-validation to ensure robustness of model comparison
- Explore different loss functions beyond WARP
- Add A/B testing framework for online evaluation 
