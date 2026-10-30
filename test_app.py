"""
Automated Integration and Unit Test Suite for Smart Farming Crop Recommendation System
Tests routes, model predictions, chatbot responses, and REST APIs.
"""

import sys
import unittest

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app import app


class CropRecommendationAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_health_endpoint(self):
        """Verify API health endpoint status and model loading."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))
        self.assertEqual(data.get('supported_crops'), 22)
        print("[PASS] test_01_health_endpoint")

    def test_02_home_get(self):
        """Verify homepage renders successfully with 200 OK."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Smart Farming', response.data)
        self.assertIn(b'Soil &', response.data)
        print("[PASS] test_02_home_get")

    def test_03_home_post_prediction(self):
        """Verify HTML form POST prediction for Rice conditions."""
        payload = {
            'nitrogen': '90',
            'phosphorus': '42',
            'potassium': '43',
            'temperature': '20.88',
            'humidity': '82.0',
            'ph': '6.5',
            'rainfall': '202.9'
        }
        response = self.client.post('/', data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Recommended Crop', response.data)
        self.assertIn(b'Rice', response.data)
        print("[PASS] test_03_home_post_prediction (Rice)")

    def test_04_home_post_coffee(self):
        """Verify HTML form POST prediction for Coffee conditions."""
        payload = {
            'nitrogen': '107',
            'phosphorus': '21',
            'potassium': '31',
            'temperature': '26.5',
            'humidity': '58.1',
            'ph': '6.8',
            'rainfall': '158.4'
        }
        response = self.client.post('/', data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Coffee', response.data)
        print("[PASS] test_04_home_post_coffee (Coffee)")

    def test_05_chat_endpoint(self):
        """Verify chatbot /chat JSON API response."""
        response = self.client.post('/chat', json={'message': 'Hello, what does nitrogen do?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('reply', data)
        self.assertIn('Nitrogen', data['reply'])
        print("[PASS] test_05_chat_endpoint")

    def test_06_api_predict(self):
        """Verify programmatic REST API /api/predict."""
        payload = {
            'nitrogen': 24,
            'phosphorus': 128,
            'potassium': 196,
            'temperature': 22.3,
            'humidity': 92.3,
            'ph': 5.9,
            'rainfall': 110.7
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('recommended_crop'), 'apple')
        self.assertGreaterEqual(data.get('confidence_percent'), 50.0)
        print(f"[PASS] test_06_api_predict (Apple, Confidence: {data.get('confidence_percent')}%)")

    def test_07_api_crops(self):
        """Verify /api/crops returns all 22 crops metadata."""
        response = self.client.get('/api/crops')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('total_crops'), 22)
        print("[PASS] test_07_api_crops")


if __name__ == '__main__':
    unittest.main()
