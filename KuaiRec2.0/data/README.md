# KuaiRec2.0 Data Directory

## Git LFS

This directory contains data files that are tracked using Git Large File Storage (Git LFS).
The CSV files in this directory are stored using Git LFS to avoid storing large binary files directly in the Git repository.

## Setup Instructions

To work with these files, you need to install and configure Git LFS:

1. Install Git LFS:
   ```bash
   sudo apt-get install git-lfs  # Debian/Ubuntu
   # OR
   brew install git-lfs          # macOS
   # OR
   yum install git-lfs           # CentOS/RHEL
   ```

2. Initialize Git LFS:
   ```bash
   git lfs install
   ```

3. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

4. If the LFS files are not automatically pulled:
   ```bash
   git lfs pull
   ```

## Available Data Files

- `small_matrix.csv`: Small interaction matrix for testing
- `big_matrix.csv`: Complete interaction matrix
- `item_categories.csv`: Categories for items
- `user_features.csv`: User feature data
- `item_daily_features.csv`: Item features
- `social_network.csv`: Social network data
- `kuairec_caption_category.csv`: Caption categories

## Working with LFS Files

- To check which files are tracked by LFS:
  ```bash
  git lfs ls-files
  ```

- To ensure you have the latest version of LFS files:
  ```bash
  git lfs fetch --all
  git lfs checkout
  ```

For more information about Git LFS, visit: https://git-lfs.github.com/ 
