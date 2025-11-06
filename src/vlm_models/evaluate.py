"""
VLM Model Performance Evaluation Script

This script evaluates the performance of different VLM models on fraud detection
by analyzing the results stored in the output/vlm_analysis folder.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import numpy as np


class VLMEvaluator:
    """Evaluates VLM model performance on fraud detection task."""
    
    def __init__(self, base_path: str = "output/vlm_analysis"):
        self.base_path = Path(base_path)
        self.models = []
        self.results = {}
        
    def load_analyses(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Load all analysis results from the output directory."""
        data = {}
        
        # Get all model names
        fraud_path = self.base_path / "fraud"
        non_fraud_path = self.base_path / "non_fraud"
        
        if fraud_path.exists():
            self.models = [d.name for d in fraud_path.iterdir() if d.is_dir()]
        
        print(f"Found models: {self.models}")
        
        for model in self.models:
            data[model] = {
                'fraud': [],
                'non_fraud': []
            }
            
            # Load fraud analyses
            fraud_model_path = fraud_path / model
            if fraud_model_path.exists():
                for image_dir in fraud_model_path.iterdir():
                    if image_dir.is_dir():
                        analysis_file = image_dir / "_analysis.json"
                        if analysis_file.exists():
                            with open(analysis_file, 'r') as f:
                                analysis = json.load(f)
                                analysis['true_label'] = 1  # Fraud
                                analysis['image_name'] = image_dir.name
                                data[model]['fraud'].append(analysis)
            
            # Load non-fraud analyses
            non_fraud_model_path = non_fraud_path / model
            if non_fraud_model_path.exists():
                for image_dir in non_fraud_model_path.iterdir():
                    if image_dir.is_dir():
                        analysis_file = image_dir / "_analysis.json"
                        if analysis_file.exists():
                            with open(analysis_file, 'r') as f:
                                analysis = json.load(f)
                                analysis['true_label'] = 0  # Non-fraud
                                analysis['image_name'] = image_dir.name
                                data[model]['non_fraud'].append(analysis)
        
        return data
    
    def extract_predictions(self, data: Dict) -> pd.DataFrame:
        """Extract predictions and true labels into a DataFrame."""
        rows = []
        
        for model, categories in data.items():
            for category, analyses in categories.items():
                for analysis in analyses:
                    row = {
                        'model': model,
                        'image_name': analysis['image_name'],
                        'true_label': analysis['true_label'],
                        'predicted_fraud': analysis['analysis']['is_fraudulent'],
                        'confidence': analysis['analysis']['overall_confidence'],
                        'fraud_score': analysis['analysis'].get('fraud_likelihood_score', 0),
                        'processing_time': analysis.get('processing_time', 0)
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        # Convert predicted_fraud to binary
        df['predicted_label'] = df['predicted_fraud'].astype(int)
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate performance metrics for each model."""
        metrics = {}
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            y_true = model_df['true_label']
            y_pred = model_df['predicted_label']
            y_scores = model_df['fraud_score'] / 100.0  # Normalize to 0-1
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics[model] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'total_samples': len(model_df),
                'avg_processing_time': model_df['processing_time'].mean()
            }
            
            # Calculate AUC-ROC if possible
            try:
                metrics[model]['auc_roc'] = roc_auc_score(y_true, y_scores)
            except:
                metrics[model]['auc_roc'] = None
        
        return metrics
    
    def create_visualizations(self, df: pd.DataFrame, metrics: Dict, output_dir: str = "output/evaluation-cheques"):
        """Create comprehensive visualization comparing model performance."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # 1. Performance Metrics Comparison
        self._plot_metrics_comparison(metrics, output_path)
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices(df, output_path)
        
        # 3. ROC Curves (if scores available)
        self._plot_roc_curves(df, output_path)
        
        # 4. Processing Time Comparison
        self._plot_processing_times(metrics, output_path)
        
        # 5. Fraud Score Distribution
        self._plot_fraud_score_distribution(df, output_path)
        
        # 6. Detailed Performance Heatmap
        self._plot_performance_heatmap(metrics, output_path)
        
        print(f"\nVisualizations saved to: {output_path}")
    
    def _plot_metrics_comparison(self, metrics: Dict, output_path: Path):
        """Plot comparison of key metrics across models."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('VLM Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for idx, (metric_name, label) in enumerate(zip(metric_names, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            values = [metrics[model][metric_name] for model in models]
            
            bars = ax.bar(models, values, alpha=0.7, edgecolor='black')
            ax.set_ylabel(label, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.set_title(f'{label} by Model', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
            
            # Highlight best performer
            best_idx = values.index(max(values))
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('darkgoldenrod')
            bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, df: pd.DataFrame, output_path: Path):
        """Plot confusion matrices for all models."""
        models = df['model'].unique()
        n_models = len(models)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Confusion Matrices by Model', fontsize=16, fontweight='bold')
        
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, model in enumerate(models):
            model_df = df[df['model'] == model]
            y_true = model_df['true_label']
            y_pred = model_df['predicted_label']
            
            cm = confusion_matrix(y_true, y_pred)
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Fraud', 'Fraud'],
                       yticklabels=['Non-Fraud', 'Fraud'],
                       ax=ax, cbar_kws={'label': 'Count'})
            ax.set_title(f'{model}', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, df: pd.DataFrame, output_path: Path):
        """Plot ROC curves for all models."""
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(12, 8))
        
        models = df['model'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for idx, model in enumerate(models):
            model_df = df[df['model'] == model]
            y_true = model_df['true_label']
            y_scores = model_df['fraud_score'] / 100.0
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[idx], lw=2, 
                    label=f'{model} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_processing_times(self, metrics: Dict, output_path: Path):
        """Plot processing time comparison."""
        plt.figure(figsize=(12, 6))
        
        models = list(metrics.keys())
        times = [metrics[model]['avg_processing_time'] for model in models]
        
        bars = plt.bar(models, times, alpha=0.7, edgecolor='black')
        plt.ylabel('Average Processing Time (seconds)', fontsize=12, fontweight='bold')
        plt.title('Processing Time by Model', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=10)
        
        # Highlight fastest
        fastest_idx = times.index(min(times))
        bars[fastest_idx].set_color('lightgreen')
        bars[fastest_idx].set_edgecolor('darkgreen')
        bars[fastest_idx].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(output_path / 'processing_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fraud_score_distribution(self, df: pd.DataFrame, output_path: Path):
        """Plot fraud score distributions by model and true label."""
        models = df['model'].unique()
        
        fig, axes = plt.subplots(len(models), 1, figsize=(12, 4*len(models)))
        if len(models) == 1:
            axes = [axes]
        
        fig.suptitle('Fraud Score Distributions', fontsize=16, fontweight='bold')
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            model_df = df[df['model'] == model]
            
            fraud_scores = model_df[model_df['true_label'] == 1]['fraud_score']
            non_fraud_scores = model_df[model_df['true_label'] == 0]['fraud_score']
            
            ax.hist(non_fraud_scores, bins=20, alpha=0.5, label='True Non-Fraud', 
                   color='green', edgecolor='black')
            ax.hist(fraud_scores, bins=20, alpha=0.5, label='True Fraud', 
                   color='red', edgecolor='black')
            
            ax.set_xlabel('Fraud Likelihood Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{model}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper center')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'fraud_score_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self, metrics: Dict, output_path: Path):
        """Plot heatmap of all performance metrics."""
        # Prepare data
        models = list(metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 
                       'specificity', 'auc_roc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 
                        'Specificity', 'AUC-ROC']
        
        data = []
        for model in models:
            row = []
            for metric in metric_names:
                value = metrics[model].get(metric)
                row.append(value if value is not None else 0)
            data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   xticklabels=metric_labels, yticklabels=models,
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                   linewidths=0.5, linecolor='gray')
        plt.title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Metric', fontsize=12, fontweight='bold')
        plt.ylabel('Model', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, metrics: Dict, df: pd.DataFrame, output_dir: str = "output/evaluation"):
        """Save detailed results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_serializable = {}
        for model, model_metrics in metrics.items():
            metrics_serializable[model] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in model_metrics.items()
            }
        
        with open(output_path / 'metrics_summary.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        # Save detailed predictions
        df.to_csv(output_path / 'detailed_predictions.csv', index=False)
        
        # Create summary report
        self._create_summary_report(metrics, output_path)
        
        print(f"\nResults saved to: {output_path}")
    
    def _create_summary_report(self, metrics: Dict, output_path: Path):
        """Create a text summary report."""
        with open(output_path / 'evaluation_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VLM FRAUD DETECTION - MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall ranking
            f.write("OVERALL RANKING (by F1 Score):\n")
            f.write("-" * 80 + "\n")
            sorted_models = sorted(metrics.items(), 
                                 key=lambda x: x[1]['f1_score'], 
                                 reverse=True)
            
            for rank, (model, model_metrics) in enumerate(sorted_models, 1):
                f.write(f"{rank}. {model:25s} - F1: {model_metrics['f1_score']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Detailed metrics for each model
            for model, model_metrics in sorted_models:
                f.write(f"MODEL: {model}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Accuracy:           {model_metrics['accuracy']:.4f}\n")
                f.write(f"  Precision:          {model_metrics['precision']:.4f}\n")
                f.write(f"  Recall:             {model_metrics['recall']:.4f}\n")
                f.write(f"  F1 Score:           {model_metrics['f1_score']:.4f}\n")
                f.write(f"  Specificity:        {model_metrics['specificity']:.4f}\n")
                if model_metrics['auc_roc']:
                    f.write(f"  AUC-ROC:            {model_metrics['auc_roc']:.4f}\n")
                f.write(f"\n")
                f.write(f"  True Positives:     {model_metrics['true_positives']}\n")
                f.write(f"  False Positives:    {model_metrics['false_positives']}\n")
                f.write(f"  True Negatives:     {model_metrics['true_negatives']}\n")
                f.write(f"  False Negatives:    {model_metrics['false_negatives']}\n")
                f.write(f"\n")
                f.write(f"  Total Samples:      {model_metrics['total_samples']}\n")
                f.write(f"  Avg Processing Time: {model_metrics['avg_processing_time']:.2f}s\n")
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Best in category
            f.write("BEST IN CATEGORY:\n")
            f.write("-" * 80 + "\n")
            
            categories = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1 Score', 'f1_score'),
                ('Specificity', 'specificity'),
                ('Fastest Processing', 'avg_processing_time')
            ]
            
            for label, metric in categories:
                if metric == 'avg_processing_time':
                    best_model = min(metrics.items(), key=lambda x: x[1][metric])
                else:
                    best_model = max(metrics.items(), key=lambda x: x[1][metric])
                
                f.write(f"  {label:25s}: {best_model[0]} ({best_model[1][metric]:.4f})\n")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        print("=" * 80)
        print("VLM FRAUD DETECTION - MODEL EVALUATION")
        print("=" * 80)
        
        # Load data
        print("\n[1/5] Loading analysis results...")
        data = self.load_analyses()
        
        # Extract predictions
        print("[2/5] Extracting predictions...")
        df = self.extract_predictions(data)
        print(f"  Total predictions: {len(df)}")
        print(f"  Models evaluated: {df['model'].nunique()}")
        
        # Calculate metrics
        print("[3/5] Calculating performance metrics...")
        metrics = self.calculate_metrics(df)
        
        # Create visualizations
        print("[4/5] Creating visualizations...")
        self.create_visualizations(df, metrics)
        
        # Save results
        print("[5/5] Saving results...")
        self.save_results(metrics, df)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        
        # Print quick summary
        print("\nQUICK SUMMARY:")
        print("-" * 80)
        sorted_models = sorted(metrics.items(), 
                             key=lambda x: x[1]['f1_score'], 
                             reverse=True)
        
        print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 80)
        for model, m in sorted_models:
            print(f"{model:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} {m['f1_score']:>10.4f}")
        
        print("\n" + "=" * 80)
        print(f"Best Overall Model (F1): {sorted_models[0][0]}")
        print(f"F1 Score: {sorted_models[0][1]['f1_score']:.4f}")
        print("=" * 80)


def main():
    """Main execution function."""
    # Initialize evaluator
    evaluator = VLMEvaluator(base_path="output/vlm_analysis_cheques")
    
    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
