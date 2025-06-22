import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import json

class AdvancedFakeNewsAnalyzer:
    """Advanced features for fake news detection"""
    
    def __init__(self, classifier, explainer):
        self.classifier = classifier
        self.explainer = explainer
        self.analysis_cache = {}
        
    def analyze_credibility_score(self, text: str) -> Dict:
        """Calculate detailed credibility score with multiple factors"""
        
        # Get base prediction
        prediction = self.classifier.predict(text)
        
        # Analyze various factors
        factors = {
            'model_confidence': prediction['confidence'],
            'text_complexity': self._analyze_text_complexity(text),
            'emotional_language': self._detect_emotional_language(text),
            'fact_checkability': self._assess_fact_checkability(text),
            'source_credibility': self._estimate_source_credibility(text)
        }
        
        # Calculate weighted credibility score
        weights = {
            'model_confidence': 0.4,
            'text_complexity': 0.15,
            'emotional_language': 0.15,
            'fact_checkability': 0.15,
            'source_credibility': 0.15
        }
        
        credibility_score = sum(
            factors[factor] * weights[factor] 
            for factor in factors
        )
        
        return {
            'overall_score': credibility_score,
            'factors': factors,
            'recommendation': self._get_credibility_recommendation(credibility_score),
            'detailed_analysis': self._generate_detailed_analysis(factors)
        }
    
    def _analyze_text_complexity(self, text: str) -> float:
        """Analyze text complexity and readability"""
        words = text.split()
        sentences = text.split('.')
        
        # Simple readability metrics
        avg_word_length = np.mean([len(word) for word in words])
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Normalize to 0-1 scale (higher = more credible)
        complexity_score = 1 - (1 / (1 + np.exp(-0.1 * (avg_sentence_length - 20))))
        
        return float(complexity_score)
    
    def _detect_emotional_language(self, text: str) -> float:
        """Detect emotional or sensational language"""
        emotional_words = [
            'shocking', 'unbelievable', 'amazing', 'terrible', 'horrible',
            'breaking', 'urgent', 'exclusive', 'sensational', 'outrageous'
        ]
        
        text_lower = text.lower()
        emotional_count = sum(word in text_lower for word in emotional_words)
        
        # Lower score for more emotional language
        emotional_score = 1 / (1 + emotional_count * 0.1)
        
        return float(emotional_score)
    
    def _assess_fact_checkability(self, text: str) -> float:
        """Assess how fact-checkable the claims are"""
        # Look for specific claims, dates, numbers, quotes
        import re
        
        has_numbers = bool(re.findall(r'\d+', text))
        has_quotes = '"' in text or "'" in text
        has_dates = bool(re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', text))
        
        checkability_score = (has_numbers * 0.3 + has_quotes * 0.4 + has_dates * 0.3)
        
        return float(checkability_score)
    
    def _estimate_source_credibility(self, text: str) -> float:
        """Estimate source credibility based on citations and references"""
        credibility_indicators = [
            'according to', 'study shows', 'research indicates',
            'experts say', 'data reveals', 'survey found'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(indicator in text_lower for indicator in credibility_indicators)
        
        credibility_score = min(1.0, indicator_count * 0.2)
        
        return float(credibility_score)
    
    def _get_credibility_recommendation(self, score: float) -> str:
        """Get recommendation based on credibility score"""
        if score >= 0.8:
            return "Highly Credible - Safe to share"
        elif score >= 0.6:
            return "Moderately Credible - Verify key claims"
        elif score >= 0.4:
            return "Questionable - Fact-check before sharing"
        else:
            return "Low Credibility - High risk of misinformation"
    
    def _generate_detailed_analysis(self, factors: Dict) -> List[str]:
        """Generate detailed analysis points"""
        analysis = []
        
        if factors['emotional_language'] < 0.5:
            analysis.append("⚠️ High use of emotional or sensational language detected")
        
        if factors['text_complexity'] < 0.3:
            analysis.append("📝 Text structure appears overly simplistic")
        
        if factors['fact_checkability'] < 0.3:
            analysis.append("❓ Lacks specific, verifiable claims")
        
        if factors['source_credibility'] < 0.3:
            analysis.append("📰 No credible sources or citations found")
        
        if factors['model_confidence'] > 0.9:
            analysis.append("🤖 AI model shows very high confidence in classification")
        
        return analysis
    
    def track_misinformation_trends(self, articles: List[Dict]) -> Dict:
        """Track misinformation trends over time"""
        
        # Group by date and classification
        df = pd.DataFrame(articles)
        df['date'] = pd.to_datetime(df['timestamp'])
        df['day'] = df['date'].dt.date
        
        # Calculate daily statistics
        daily_stats = df.groupby(['day', 'prediction']).size().unstack(fill_value=0)
        
        # Calculate trends
        fake_trend = daily_stats.get('Fake', pd.Series()).pct_change().mean()
        
        return {
            'daily_counts': daily_stats.to_dict(),
            'fake_news_trend': float(fake_trend) if not pd.isna(fake_trend) else 0,
            'total_analyzed': len(articles),
            'fake_percentage': (df['prediction'] == 'Fake').mean() * 100
        }
    
    def generate_fact_check_report(self, text: str, url: str = None) -> Dict:
        """Generate comprehensive fact-check report"""
        
        # Generate unique ID for the article
        article_id = hashlib.md5(text.encode()).hexdigest()[:10]
        
        # Check cache
        if article_id in self.analysis_cache:
            return self.analysis_cache[article_id]
        
        # Perform analysis
        prediction = self.classifier.predict(text)
        credibility = self.analyze_credibility_score(text)
        lime_exp = self.explainer.explain_with_lime(text, num_features=5)
        
        report = {
            'report_id': article_id,
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'classification': {
                'prediction': prediction['prediction'],
                'confidence': prediction['confidence'],
                'probabilities': prediction['probabilities']
            },
            'credibility_analysis': credibility,
            'key_indicators': lime_exp['feature_importance'][:5],
            'recommendations': self._generate_recommendations(prediction, credibility),
            'similar_articles': self._find_similar_articles(text),
            'fact_check_resources': self._get_fact_check_resources()
        }
        
        # Cache the report
        self.analysis_cache[article_id] = report
        
        return report
    
    def _generate_recommendations(self, prediction: Dict, credibility: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        if prediction['prediction'] == 'Fake':
            recommendations.append("🚫 Do not share this article without verification")
            recommendations.append("🔍 Check official sources for this information")
            recommendations.append("⚠️ Report this content if found on social media")
        
        if credibility['overall_score'] < 0.5:
            recommendations.append("📋 Cross-reference claims with trusted news sources")
            recommendations.append("🏛️ Check government or institutional websites for official information")
        
        if credibility['factors']['emotional_language'] < 0.5:
            recommendations.append("😤 Be aware of emotional manipulation tactics")
            recommendations.append("🧘 Take a moment to think critically before reacting")
        
        return recommendations
    
def _find_similar_articles(self, text: str, top_k: int = 3) -> List[Dict]:
        """Find similar articles (placeholder - would need article database)"""
        # In production, this would search a database of previously analyzed articles
        return [
            {
                'title': 'Similar article example',
                'similarity_score': 0.85,
                'classification': 'Fake',
                'date': '2024-01-15'
            }
        ]
    
def _get_fact_check_resources(self) -> List[Dict]:
        """Get relevant fact-checking resources"""
        return [
            {'name': 'Snopes', 'url': 'https://www.snopes.com'},
            {'name': 'FactCheck.org', 'url': 'https://www.factcheck.org'},
            {'name': 'PolitiFact', 'url': 'https://www.politifact.com'},
            {'name': 'Full Fact', 'url': 'https://fullfact.org'}
        ]
    
def create_performance_dashboard(self, test_data: pd.DataFrame) -> Dict:
        """Create performance metrics dashboard data"""
        
        # Get predictions for test data
        predictions = []
        true_labels = test_data['label'].tolist()
        
        for text in test_data['text']:
            pred = self.classifier.predict(text)
            predictions.append(1 if pred['prediction'] == 'Fake' else 0)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1_score': f1_score(true_labels, predictions)
        }
        
        # Get prediction probabilities for ROC curve
        probs = []
        for text in test_data['text']:
            pred = self.classifier.predict(text)
            probs.append(pred['probabilities']['Fake'])
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(true_labels, probs)
        
        return {
            'metrics': metrics,
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc},
            'pr_curve': {'precision': precision.tolist(), 'recall': recall.tolist()}
        }
        