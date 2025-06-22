import logging
from datetime import datetime
import json
import os
from typing import Dict, List
import pandas as pd
from collections import defaultdict
import asyncio
import aiofiles

class FakeNewsMonitor:
    """Monitor system performance and usage"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(f"{log_dir}/fake_news_monitor.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Performance metrics
        self.metrics = defaultdict(list)
        
    async def log_prediction(self, text: str, prediction: Dict, user_id: str = None):
        """Log prediction asynchronously"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'text_length': len(text),
            'prediction': prediction['prediction'],
            'confidence': prediction['confidence'],
            'processing_time': prediction.get('processing_time', 0)
        }
        
        # Write to log file
        log_file = f"{self.log_dir}/predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        async with aiofiles.open(log_file, 'a') as f:
            await f.write(json.dumps(log_entry) + '\n')
        
        # Update metrics
        self.metrics['daily_predictions'].append(log_entry)
        
        # Log to standard logger
        self.logger.info(f"Prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.2f})")
    
    def get_system_metrics(self) -> Dict:
        """Get current system performance metrics"""
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_daily_report(self) -> Dict:
        """Generate daily usage and performance report"""
        today = datetime.now().strftime('%Y%m%d')
        log_file = f"{self.log_dir}/predictions_{today}.jsonl"
        
        if not os.path.exists(log_file):
            return {'error': 'No data for today'}
        
        # Read predictions
        predictions = []
        with open(log_file, 'r') as f:
            for line in f:
                predictions.append(json.loads(line))
        
        df = pd.DataFrame(predictions)
        
        # Calculate statistics
        report = {
            'date': today,
            'total_predictions': len(df),
            'fake_news_count': (df['prediction'] == 'Fake').sum(),
            'real_news_count': (df['prediction'] == 'Real').sum(),
            'average_confidence': df['confidence'].mean(),
            'average_processing_time': df['processing_time'].mean(),
            'peak_hour': df.groupby(pd.to_datetime(df['timestamp']).dt.hour).size().idxmax()
        }
        
        return report
    
    def alert_suspicious_activity(self, pattern: str, threshold: int = 10):
        """Alert on suspicious usage patterns"""
        # Check for repeated submissions
        recent_predictions = self.metrics['daily_predictions'][-100:]
        
        # Count similar texts
        text_counts = defaultdict(int)
        for pred in recent_predictions:
            text_hash = hash(pred.get('text_length', 0))
            text_counts[text_hash] += 1
        
        # Check for threshold
        for text_hash, count in text_counts.items():
            if count > threshold:
                self.logger.warning(f"Suspicious activity detected: {count} similar texts submitted")
                return True
        
        return False