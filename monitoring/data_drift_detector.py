import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import ks_2samp
from azureml.core import Workspace, Dataset
from azureml.datadrift import DataDriftDetector, AlertConfiguration
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDriftMonitor:
    def __init__(self, workspace=None):
        self.workspace = workspace
        self.drift_threshold = 0.1  # Configurable threshold
        
    def load_baseline_data(self, dataset_name=None, file_path=None):
        """Load baseline/training data"""
        if dataset_name and self.workspace:
            dataset = Dataset.get_by_name(self.workspace, dataset_name)
            df = dataset.to_pandas_dataframe()
            logger.info(f"Loaded baseline dataset: {dataset_name}")
        elif file_path:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded baseline data from: {file_path}")
        else:
            raise ValueError("Either dataset_name or file_path must be provided")
        
        return df
    
    def load_current_data(self, dataset_name=None, file_path=None):
        """Load current/production data"""
        if dataset_name and self.workspace:
            dataset = Dataset.get_by_name(self.workspace, dataset_name)
            df = dataset.to_pandas_dataframe()
            logger.info(f"Loaded current dataset: {dataset_name}")
        elif file_path:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded current data from: {file_path}")
        else:
            raise ValueError("Either dataset_name or file_path must be provided")
        
        return df
    
    def detect_statistical_drift(self, baseline_data, current_data):
        """Detect drift using statistical tests"""
        drift_results = {}
        
        # Get numeric columns
        numeric_columns = baseline_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in current_data.columns:
                baseline_values = baseline_data[column].dropna()
                current_values = current_data[column].dropna()
                
                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = ks_2samp(baseline_values, current_values)
                
                # Mann-Whitney U test
                try:
                    mw_statistic, mw_p_value = stats.mannwhitneyu(baseline_values, current_values)
                except ValueError:
                    mw_statistic, mw_p_value = np.nan, np.nan
                
                # Population Stability Index (PSI)
                psi_score = self.calculate_psi(baseline_values, current_values)
                
                drift_results[column] = {
                    'ks_statistic': ks_statistic,
                    'ks_p_value': ks_p_value,
                    'mw_statistic': mw_statistic,
                    'mw_p_value': mw_p_value,
                    'psi_score': psi_score,
                    'drift_detected': ks_p_value < 0.05 or psi_score > 0.2,
                    'baseline_mean': baseline_values.mean(),
                    'current_mean': current_values.mean(),
                    'baseline_std': baseline_values.std(),
                    'current_std': current_values.std()
                }
        
        return drift_results
    
    def calculate_psi(self, baseline, current, bins=10):
        """Calculate Population Stability Index"""
        try:
            # Create bins based on baseline data
            _, bin_edges = np.histogram(baseline, bins=bins)
            
            # Calculate expected (baseline) and actual (current) frequencies
            baseline_freq, _ = np.histogram(baseline, bins=bin_edges)
            current_freq, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to percentages
            baseline_pct = baseline_freq / len(baseline)
            current_pct = current_freq / len(current)
            
            # Avoid division by zero
            baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
            current_pct = np.where(current_pct == 0, 0.0001, current_pct)
            
            # Calculate PSI
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            
            return psi
        except Exception as e:
            logger.warning(f"Error calculating PSI: {str(e)}")
            return np.nan
    
    def detect_categorical_drift(self, baseline_data, current_data):
        """Detect drift in categorical variables"""
        categorical_results = {}
        
        # Get categorical columns
        categorical_columns = baseline_data.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            if column in current_data.columns:
                baseline_counts = baseline_data[column].value_counts(normalize=True)
                current_counts = current_data[column].value_counts(normalize=True)
                
                # Align indices
                all_categories = set(baseline_counts.index) | set(current_counts.index)
                baseline_aligned = baseline_counts.reindex(all_categories, fill_value=0)
                current_aligned = current_counts.reindex(all_categories, fill_value=0)
                
                # Chi-square test
                try:
                    chi2_stat, chi2_p_value = stats.chisquare(
                        current_aligned * len(current_data),
                        baseline_aligned * len(current_data)
                    )
                except ValueError:
                    chi2_stat, chi2_p_value = np.nan, np.nan
                
                # Calculate categorical PSI
                psi_cat = self.calculate_categorical_psi(baseline_aligned, current_aligned)
                
                categorical_results[column] = {
                    'chi2_statistic': chi2_stat,
                    'chi2_p_value': chi2_p_value,
                    'psi_score': psi_cat,
                    'drift_detected': chi2_p_value < 0.05 or psi_cat > 0.2,
                    'baseline_categories': len(baseline_counts),
                    'current_categories': len(current_counts),
                    'new_categories': len(set(current_counts.index) - set(baseline_counts.index)),
                    'missing_categories': len(set(baseline_counts.index) - set(current_counts.index))
                }
        
        return categorical_results
    
    def calculate_categorical_psi(self, baseline_pct, current_pct):
        """Calculate PSI for categorical variables"""
        try:
            # Avoid division by zero
            baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
            current_pct = np.where(current_pct == 0, 0.0001, current_pct)
            
            # Calculate PSI
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            
            return psi
        except Exception as e:
            logger.warning(f"Error calculating categorical PSI: {str(e)}")
            return np.nan
    
    def generate_drift_report(self, drift_results, categorical_results, output_dir='outputs/drift_reports'):
        """Generate comprehensive drift report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine results
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'numeric_features': drift_results,
            'categorical_features': categorical_results,
            'summary': {
                'total_features_analyzed': len(drift_results) + len(categorical_results),
                'features_with_drift': sum([1 for r in drift_results.values() if r['drift_detected']]) + 
                                     sum([1 for r in categorical_results.values() if r['drift_detected']]),
                'drift_percentage': 0
            }
        }
        
        # Calculate drift percentage
        total_features = all_results['summary']['total_features_analyzed']
        if total_features > 0:
            all_results['summary']['drift_percentage'] = (
                all_results['summary']['features_with_drift'] / total_features * 100
            )
        
        # Save JSON report
        report_path = os.path.join(output_dir, f'drift_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Drift report saved to: {report_path}")
        
        # Generate visualizations
        self.create_drift_visualizations(drift_results, categorical_results, output_dir)
        
        return all_results
    
    def create_drift_visualizations(self, drift_results, categorical_results, output_dir):
        """Create visualizations for drift analysis"""
        # PSI scores plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Numeric features PSI
        if drift_results:
            numeric_features = list(drift_results.keys())
            psi_scores = [drift_results[f]['psi_score'] for f in numeric_features]
            
            axes[0, 0].bar(numeric_features, psi_scores)
            axes[0, 0].axhline(y=0.2, color='r', linestyle='--', label='Drift Threshold')
            axes[0, 0].set_title('Numeric Features - PSI Scores')
            axes[0, 0].set_ylabel('PSI Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend()
        
        # Categorical features PSI
        if categorical_results:
            cat_features = list(categorical_results.keys())
            cat_psi_scores = [categorical_results[f]['psi_score'] for f in cat_features]
            
            axes[0, 1].bar(cat_features, cat_psi_scores)
            axes[0, 1].axhline(y=0.2, color='r', linestyle='--', label='Drift Threshold')
            axes[0, 1].set_title('Categorical Features - PSI Scores')
            axes[0, 1].set_ylabel('PSI Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend()
        
        # P-values heatmap for numeric features
        if drift_results:
            p_values_data = []
            for feature in numeric_features:
                p_values_data.append([
                    drift_results[feature]['ks_p_value'],
                    drift_results[feature]['mw_p_value']
                ])
            
            p_values_df = pd.DataFrame(p_values_data, 
                                     index=numeric_features, 
                                     columns=['KS Test', 'Mann-Whitney'])
            
            sns.heatmap(p_values_df, annot=True, cmap='RdYlBu_r', ax=axes[1, 0])
            axes[1, 0].set_title('Statistical Test P-values')
        
        # Summary drift status
        all_features = list(drift_results.keys()) + list(categorical_results.keys())
        drift_status = []
        
        for feature in drift_results.keys():
            drift_status.append('Drift Detected' if drift_results[feature]['drift_detected'] else 'No Drift')
        
        for feature in categorical_results.keys():
            drift_status.append('Drift Detected' if categorical_results[feature]['drift_detected'] else 'No Drift')
        
        if all_features:
            drift_counts = pd.Series(drift_status).value_counts()
            axes[1, 1].pie(drift_counts.values, labels=drift_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Overall Drift Status')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'drift_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.close()
        
        logger.info(f"Drift visualizations saved to: {output_dir}")

def main():
    """Main drift detection pipeline"""
    try:
        # Initialize workspace
        try:
            ws = Workspace.from_config()
            logger.info("Connected to Azure ML workspace")
        except Exception as e:
            logger.warning(f"Could not connect to Azure ML workspace: {str(e)}")
            ws = None
        
        # Initialize drift monitor
        drift_monitor = DataDriftMonitor(workspace=ws)
        
        # Load data (customize these paths/dataset names)
        baseline_data = drift_monitor.load_baseline_data(file_path='data/baseline_data.csv')
        current_data = drift_monitor.load_current_data(file_path='data/current_data.csv')
        
        # Detect drift
        logger.info("Detecting statistical drift...")
        drift_results = drift_monitor.detect_statistical_drift(baseline_data, current_data)
        
        logger.info("Detecting categorical drift...")
        categorical_results = drift_monitor.detect_categorical_drift(baseline_data, current_data)
        
        # Generate report
        logger.info("Generating drift report...")
        report_results = drift_monitor.generate_drift_report(drift_results, categorical_results)
        
        # Log summary
        summary = report_results['summary']
        logger.info(f"Drift Analysis Summary:")
        logger.info(f"- Total features analyzed: {summary['total_features_analyzed']}")
        logger.info(f"- Features with drift: {summary['features_with_drift']}")
        logger.info(f"- Drift percentage: {summary['drift_percentage']:.1f}%")
        
        # Set exit code based on drift detection
        if summary['drift_percentage'] > 20:  # Threshold for significant drift
            logger.warning("Significant data drift detected!")
            exit(1)
        else:
            logger.info("Data drift analysis completed successfully")
            exit(0)
            
    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()