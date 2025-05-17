#!/usr/bin/env python3

import json
import os
import glob
import re
from pathlib import Path

def get_latest_results_file():
    """Find the most recent results JSON file"""
    results_files = glob.glob("results_*.json")
    if not results_files:
        print("No results files found!")
        return None
        
    # Sort by modification time, newest first
    latest_file = max(results_files, key=os.path.getmtime)
    print(f"Using latest results file: {latest_file}")
    return latest_file

def format_metrics_table(baseline_metrics, hybrid_metrics):
    """Format a comparison table of metrics"""
    table_rows = []
    table_rows.append("| Metric | Baseline | Hybrid | Improvement |")
    table_rows.append("|--------|----------|--------|-------------|")
    
    # Priority metrics to include
    priority_metrics = [
        'precision@5', 'recall@5', 'f1@5', 
        'recall@10', 'ndcg@10'
    ]
    
    for metric in priority_metrics:
        if metric in baseline_metrics and metric in hybrid_metrics:
            baseline_value = baseline_metrics[metric]
            hybrid_value = hybrid_metrics[metric]
            
            # Calculate improvement
            improvement = ((hybrid_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0
            
            row = f"| {metric} | {baseline_value:.4f} | {hybrid_value:.4f} | {improvement:.1f}% |"
            table_rows.append(row)
    
    return "\n".join(table_rows)

def format_training_times(results, matrix_file):
    """Format the training times section"""
    baseline_time = results.get('baseline_time', 'N/A')
    hybrid_time = results.get('hybrid_time', 'N/A')
    
    if baseline_time != 'N/A':
        baseline_time = f"{baseline_time:.1f} seconds"
    if hybrid_time != 'N/A':
        hybrid_time = f"{hybrid_time:.1f} seconds"
    
    matrix_type = "small" if "small" in matrix_file else "big"
    
    times = []
    times.append(f"- {matrix_type} matrix:")
    times.append(f"  - Baseline model: {baseline_time}")
    times.append(f"  - Hybrid model: {hybrid_time}")
    
    return "\n".join(times)

def update_report(results_file, report_file='report.md'):
    """Update the report.md file with results from the JSON file"""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    if not os.path.exists(report_file):
        print(f"Report file not found: {report_file}")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Load report
    with open(report_file, 'r') as f:
        report_content = f.read()
    
    # Extract baseline and hybrid metrics
    baseline_metrics = results.get('baseline_metrics', {})
    hybrid_metrics = results.get('hybrid_metrics', {})
    matrix_file = results.get('matrix_file', 'unknown_matrix.csv')
    
    if not baseline_metrics or not hybrid_metrics:
        print("Missing baseline or hybrid metrics in results file")
        return
    
    # Format metrics table
    metrics_table = format_metrics_table(baseline_metrics, hybrid_metrics)
    
    # Format training times
    training_times = format_training_times(results, matrix_file)
    
    # Replace placeholder table in report
    table_pattern = r'\| Metric \| Baseline \| Hybrid \| Improvement \|(.*?)\*Note: .*?\*'
    updated_report = re.sub(table_pattern, metrics_table + '\n\n', report_content, flags=re.DOTALL)
    
    # Replace training times
    times_pattern = r'### Training Time\n\n(.*?)(?=\n\n## Conclusions)'
    updated_report = re.sub(times_pattern, f"### Training Time\n\n{training_times}", updated_report, flags=re.DOTALL)
    
    # Write updated report
    with open(report_file, 'w') as f:
        f.write(updated_report)
    
    print(f"Report updated successfully: {report_file}")

if __name__ == "__main__":
    latest_results = get_latest_results_file()
    if latest_results:
        update_report(latest_results) 
