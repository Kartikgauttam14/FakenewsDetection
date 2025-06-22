import unittest
import torch
import pandas as pd
from model import FakeNewsClassifier
from explainer import FakeNewsExplainer
from utils import preprocess_text
from config import Config

class TestFakeNewsDetector(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
                """Set up test fixtures"""
    cls.config = Config()
    cls.classifier = FakeNewsClassifier(model_name=cls.config.MODEL_NAME)
    cls.explainer = FakeNewsExplainer(cls.classifier, cls.classifier.tokenizer)
    
        # Test samples
    cls.test_samples = {
            'real': [
                "The President announced new economic policies today in a press conference at the White House.",
                "Scientists have published a peer-reviewed study on climate change in Nature journal.",
                "The stock market closed higher today following positive earnings reports from major tech companies."
            ],
            'fake': [
                "BREAKING: Aliens have officially made contact with Earth governments, sources confirm!",
                "This one weird trick will make you lose 50 pounds in just 2 days! Doctors hate this!",
                "Government admits to putting mind control chips in vaccines to track citizens."
            ]
        }
    
    def test_model_prediction(self):
        """Test basic model predictions"""
        for text in self.test_samples['real'][:1]:
            result = self.classifier.predict(text)
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertIn('probabilities', result)
            self.assertIn(result['prediction'], ['Real', 'Fake'])
            self.assertTrue(0 <= result['confidence'] <= 1)
    
    def test_lime_explanation(self):
        """Test LIME explanation generation"""
        text = self.test_samples['real'][0]
        explanation = self.explainer.explain_with_lime(text, num_features=5)
        
        self.assertIn('prediction', explanation)
        self.assertIn('feature_importance', explanation)
        self.assertIsInstance(explanation['feature_importance'], list)
        self.assertTrue(len(explanation['feature_importance']) > 0)
    
    def test_attention_explanation(self):
        """Test attention-based explanation"""
        text = self.test_samples['fake'][0]
        explanation = self.explainer.explain_with_shap(text, num_features=5)
        
        self.assertIn('prediction', explanation)
        self.assertIn('important_words', explanation)
        self.assertIn('attention_weights', explanation)
        self.assertIsInstance(explanation['important_words'], list)
    
    def test_preprocessing(self):
        """Test text preprocessing"""
        test_text = "Check out this link: https://example.com <p>HTML content</p>   Extra    spaces"
        cleaned = preprocess_text(test_text)
        
        self.assertNotIn('https://', cleaned)
        self.assertNotIn('<p>', cleaned)
        self.assertEqual(len(cleaned.split()), 4)  # "Check out this link HTML content Extra spaces"
    
    def test_batch_prediction(self):
        """Test batch predictions"""
        texts = self.test_samples['real'] + self.test_samples['fake']
        
        for text in texts:
            result = self.classifier.predict(text)
            self.assertIsInstance(result, dict)
            self.assertTrue(result['probabilities']['Real'] + result['probabilities']['Fake'] > 0.99)

class TestModelPerformance(unittest.TestCase):
    """Test model performance metrics"""
    
    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.classifier = FakeNewsClassifier(model_name=cls.config.MODEL_NAME)
    
    def test_inference_speed(self):
        """Test inference speed"""
        import time
        
        text = "This is a test news article to measure inference speed."
        
        start_time = time.time()
        _ = self.classifier.predict(text)
        inference_time = time.time() - start_time
        
        # Should complete within 2 seconds even on CPU
        self.assertLess(inference_time, 2.0)
    
    def test_memory_usage(self):
        """Test memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make multiple predictions
        for _ in range(10):
            text = "Test article content" * 50
            _ = self.classifier.predict(text)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB)
        self.assertLess(memory_increase, 500)

if __name__ == '__main__':
    unittest.main()