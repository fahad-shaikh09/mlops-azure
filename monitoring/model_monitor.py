import os
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timedelta
from azureml.core import Workspace, Webservice
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPerformanceMonitor:
    def __init__(self, workspace=None):
        self.workspace = workspace
        self.performance_thresholds = {
            'accuracy': 0.7,
            'precision': 0.6,
            'recall': 0.6,
            'f1_score': 0.6
        }
    
    def get_deployment_logs(self, service_name, num_logs=1000):
        """Get deployment logs from Azure ML service"""
        if not self.workspace:
            logger.warning("Workspace not available, skipping log retrieval")
            return None
        
        try:
            service = Webservice(workspace=self.workspace, name=service_name)
            logs = service.get_logs(num_lines=num_logs)
            logger.info(f"Retrieved {len(logs.split('\\n'))} log lines from {service_name}")
            return logs
        except Exception as e:
            logger.error(f"Failed to retrieve logs: {str(e)}")
            return None
    
    def extract_prediction_data(self, logs):
        """Extract prediction data from service logs"""
        # This is a simplified example - in practice, you'd parse actual log format
        predictions_data = []
        
        if logs:
            log_lines = logs.split('\\n')
            for line in log_lines:
                # Look for prediction patterns in logs
                if 'prediction' in line.lower() and 'result' in line.lower():
                    try:
                        # Extract timestamp, input, and prediction from log line
                        # This would depend on your actual log format
                        timestamp = datetime.now()  # Placeholder
                        prediction_info = {
                            'timestamp': timestamp,
                            'prediction': 1,  # Placeholder
                            'confidence': 0.8,  # Placeholder
                            'response_time': 0.1  # Placeholder
                        }
                        predictions_data.append(prediction_info)
                    except Exception as e:
                        logger.warning(f"Failed to parse log line: {str(e)}")
        
        return predictions_data
    
    def load_ground_truth_data(self, file_path=None):
        """Load ground truth data for performance evaluation"""
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logger.info(f"Loaded ground truth data: {len(df)} samples")
            return df
        else:
            logger.warning("Ground truth data not available")
            return None
    
    def calculate_performance_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate model performance metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC if probabilities are available and binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def analyze_prediction_patterns(self, predictions_data):
        """Analyze patterns in model predictions"""
        if not predictions_data:
            return {}
        
        df = pd.DataFrame(predictions_data)
        
        analysis = {
            'total_predictions': len(df),
            'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else None,
            'avg_response_time': df['response_time'].mean() if 'response_time' in df.columns else None,
            'prediction_distribution': df['prediction'].value_counts().to_dict() if 'prediction' in df.columns else {},
            'low_confidence_predictions': len(df[df['confidence'] < 0.5]) if 'confidence' in df.columns else 0
        }
        
        return analysis
    
    def detect_performance_degradation(self, current_metrics, baseline_metrics):
        """Detect performance degradation compared to baseline"""
        degradation_detected = {}
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                degradation_pct = (baseline_value - current_value) / baseline_value * 100
                
                # Check if degradation exceeds threshold (e.g., 10% drop)
                degradation_detected[metric_name] = {
                    'current_value': current_value,
                    'baseline_value': baseline_value,
                    'degradation_pct': degradation_pct,
                    'degradation_detected': degradation_pct > 10,
                    'below_threshold': current_value < self.performance_thresholds.get(metric_name, 0)
                }
        
        return degradation_detected
    
    def monitor_resource_usage(self, service_name):
        """Monitor resource usage of the deployed model"""
        # This would typically integrate with Azure Monitor or similar
        # For now, return placeholder data
        resource_metrics = {
            'cpu_usage': np.random.uniform(20, 80),  # Placeholder
            'memory_usage': np.random.uniform(30, 90),  # Placeholder
            'request_rate': np.random.uniform(10, 100),  # Placeholder
            'error_rate': np.random.uniform(0, 5),  # Placeholder
            'avg_response_time': np.random.uniform(0.1, 2.0)  # Placeholder
        }
        
        logger.info(f"Resource metrics for {service_name}: {resource_metrics}")
        return resource_metrics
    
    def generate_performance_report(self, metrics, pattern_analysis, degradation_analysis, 
                                  resource_metrics, output_dir='outputs/performance_reports'):
        """Generate comprehensive performance report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': metrics,
            'pattern_analysis': pattern_analysis,
            'degradation_analysis': degradation_analysis,
            'resource_metrics': resource_metrics,
            'alerts': []
        }
        
        # Generate alerts based on analysis
        for metric_name, degradation_info in degradation_analysis.items():
            if degradation_info.get('degradation_detected', False):
                report['alerts'].append({
                    'type': 'performance_degradation',
                    'metric': metric_name,
                    'message': f"{metric_name} degraded by {degradation_info['degradation_pct']:.1f}%"
                })
            
            if degradation_info.get('below_threshold', False):
                report['alerts'].append({
                    'type': 'threshold_violation',
                    'metric': metric_name,
                    'message': f"{metric_name} below threshold: {degradation_info['current_value']:.3f}"
                })
        
        # Check resource usage alerts
        if resource_metrics.get('cpu_usage', 0) > 80:
            report['alerts'].append({
                'type': 'resource_usage',
                'metric': 'cpu_usage',
                'message': f"High CPU usage: {resource_metrics['cpu_usage']:.1f}%"
            })
        
        if resource_metrics.get('error_rate', 0) > 3:
            report['alerts'].append({
                'type': 'error_rate',
                'metric': 'error_rate',
                'message': f"High error rate: {resource_metrics['error_rate']:.1f}%"
            })
        
        # Save report
        report_path = os.path.join(output_dir, f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to: {report_path}")
        
        # Generate visualizations
        self.create_performance_visualizations(report, output_dir)
        
        return report
    
    def create_performance_visualizations(self, report, output_dir):
        """Create visualizations for performance monitoring"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics chart
        if report['performance_metrics']:
            metrics_names = list(report['performance_metrics'].keys())
            metrics_values = list(report['performance_metrics'].values())
            
            axes[0, 0].bar(metrics_names, metrics_values)
            axes[0, 0].set_title('Current Performance Metrics')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add threshold lines
            for i, metric_name in enumerate(metrics_names):
                if metric_name in self.performance_thresholds:
                    axes[0, 0].axhline(y=self.performance_thresholds[metric_name], 
                                     color='r', linestyle='--', alpha=0.7)
        
        # Resource usage chart
        if report['resource_metrics']:
            resource_names = list(report['resource_metrics'].keys())
            resource_values = list(report['resource_metrics'].values())
            
            axes[0, 1].bar(resource_names, resource_values)
            axes[0, 1].set_title('Resource Usage Metrics')
            axes[0, 1].set_ylabel('Usage %')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Degradation analysis
        if report['degradation_analysis']:
            degradation_data = []
            for metric_name, degradation_info in report['degradation_analysis'].items():
                degradation_data.append({
                    'metric': metric_name,
                    'current': degradation_info['current_value'],
                    'baseline': degradation_info['baseline_value']
                })
            
            if degradation_data:
                df_degradation = pd.DataFrame(degradation_data)
                x = np.arange(len(df_degradation))
                width = 0.35
                
                axes[1, 0].bar(x - width/2, df_degradation['baseline'], width, label='Baseline')
                axes[1, 0].bar(x + width/2, df_degradation['current'], width, label='Current')
                axes[1, 0].set_title('Performance Comparison')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(df_degradation['metric'], rotation=45)
                axes[1, 0].legend()
        
        # Alerts summary
        if report['alerts']:
            alert_types = [alert['type'] for alert in report['alerts']]
            alert_counts = pd.Series(alert_types).value_counts()
            
            axes[1, 1].pie(alert_counts.values, labels=alert_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Alerts Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Alerts', ha='center', va='center', fontsize=16)
            axes[1, 1].set_title('Alerts Status')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.close()
        
        logger.info(f"Performance visualizations saved to: {output_dir}")

def main():
    """Main model monitoring pipeline"""
    try:
        # Initialize workspace
        try:
            ws = Workspace.from_config()
            logger.info("Connected to Azure ML workspace")
        except Exception as e:
            logger.warning(f"Could not connect to Azure ML workspace: {str(e)}")
            ws = None
        
        # Initialize monitor
        monitor = ModelPerformanceMonitor(workspace=ws)
        
        # Monitor specific service (customize service name)
        service_name = os.getenv('SERVICE_NAME', 'model-endpoint')
        
        # Get deployment logs
        logs = monitor.get_deployment_logs(service_name)
        
        # Extract prediction data
        predictions_data = monitor.extract_prediction_data(logs)
        
        # Analyze prediction patterns
        pattern_analysis = monitor.analyze_prediction_patterns(predictions_data)
        logger.info(f"Pattern analysis: {pattern_analysis}")
        
        # Load baseline metrics (customize path)
        baseline_metrics_path = 'outputs/baseline_metrics.json'
        if os.path.exists(baseline_metrics_path):
            with open(baseline_metrics_path, 'r') as f:
                baseline_metrics = json.load(f)
        else:
            # Use placeholder baseline metrics
            baseline_metrics = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.80,
                'f1_score': 0.81
            }
            logger.warning("Using placeholder baseline metrics")
        
        # Calculate current performance (would need ground truth data)
        current_metrics = {
            'accuracy': np.random.uniform(0.7, 0.9),  # Placeholder
            'precision': np.random.uniform(0.6, 0.9),  # Placeholder
            'recall': np.random.uniform(0.6, 0.9),  # Placeholder
            'f1_score': np.random.uniform(0.6, 0.9)  # Placeholder
        }
        
        # Detect performance degradation
        degradation_analysis = monitor.detect_performance_degradation(current_metrics, baseline_metrics)
        
        # Monitor resource usage
        resource_metrics = monitor.monitor_resource_usage(service_name)
        
        # Generate report
        report = monitor.generate_performance_report(
            current_metrics, pattern_analysis, degradation_analysis, resource_metrics
        )
        
        # Log summary
        logger.info(f"Performance monitoring completed:")
        logger.info(f"- Alerts generated: {len(report['alerts'])}")
        logger.info(f"- Performance metrics: {current_metrics}")
        
        # Set exit code based on alerts
        critical_alerts = [alert for alert in report['alerts'] 
                         if alert['type'] in ['performance_degradation', 'threshold_violation']]
        
        if critical_alerts:
            logger.warning(f"Critical alerts detected: {len(critical_alerts)}")
            exit(1)
        else:
            logger.info("Model performance monitoring completed successfully")
            exit(0)
            
    except Exception as e:
        logger.error(f"Model monitoring failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()