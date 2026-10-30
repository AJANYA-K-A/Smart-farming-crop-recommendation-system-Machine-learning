"""
Automated Integration and Unit Test Suite for Smart Farming Crop Recommendation System
Tests authentication, login/logout flow, dashboard, model predictions, chatbot, and REST APIs.
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
        app.config['SECRET_KEY'] = 'test-secret-key'
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

    def test_02_index_redirects_to_login(self):
        """Verify root URL / redirects to /login for unauthenticated users."""
        response = self.client.get('/', follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers.get('Location', ''))
        print("[PASS] test_02_index_redirects_to_login")

    def test_03_login_page_renders(self):
        """Verify /login renders login portal successfully."""
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CropIQ', response.data)
        self.assertIn(b'Welcome Back', response.data)
        print("[PASS] test_03_login_page_renders")

    def test_04_login_post_and_dashboard(self):
        """Verify user login POST creates session and accesses /dashboard."""
        with self.client:
            response = self.client.post('/login', data={
                'email': 'farmer@smartfarm.org',
                'password': 'farming123'
            }, follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'CropIQ Dashboard', response.data)
            self.assertIn(b'Farmer John', response.data)
            print("[PASS] test_04_login_post_and_dashboard")

    def test_05_logout(self):
        """Verify /logout clears session and renders separate logout page."""
        with self.client:
            self.client.post('/login', data={'email': 'farmer@smartfarm.org', 'password': '123'})
            response = self.client.get('/logout')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'You Have Been Signed Out', response.data)
            self.assertIn(b'Sign In Again', response.data)
            print("[PASS] test_05_logout (Separate Logout Page)")

    def test_05b_register(self):
        """Verify /register GET and POST registration flow."""
        response = self.client.get('/register')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Join CropIQ', response.data)

        with self.client:
            post_res = self.client.post('/register', data={
                'name': 'Agronomist Maria',
                'farm_name': 'Sunny Acres',
                'email': 'maria@agro.org',
                'role': 'Lead Agronomist',
                'password': 'safePassword123'
            }, follow_redirects=True)
            self.assertEqual(post_res.status_code, 200)
            self.assertIn(b'CropIQ Dashboard', post_res.data)
            self.assertIn(b'Model Accuracy', post_res.data)
            print("[PASS] test_05b_register (Separate Register Page)")

    def test_06_chat_endpoint(self):
        """Verify chatbot /chat JSON API response."""
        response = self.client.post('/chat', json={'message': 'What does Nitrogen do for plants?'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('reply', data)
        self.assertIn('Nitrogen', data['reply'])
        print("[PASS] test_06_chat_endpoint")

    def test_07_api_predict_rice(self):
        """Verify programmatic REST API /api/predict for Rice."""
        payload = {
            'nitrogen': 90,
            'phosphorus': 42,
            'potassium': 43,
            'temperature': 20.88,
            'humidity': 82.0,
            'ph': 6.5,
            'rainfall': 202.9
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('recommended_crop'), 'rice')
        self.assertGreaterEqual(data.get('confidence_percent'), 50.0)
        print(f"[PASS] test_07_api_predict_rice (Confidence: {data.get('confidence_percent')}%)")

    def test_08_api_predict_apple(self):
        """Verify programmatic REST API /api/predict for Apple."""
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
        print("[PASS] test_08_api_predict_apple")

    def test_09_api_crops(self):
        """Verify /api/crops returns all 22 crops metadata."""
        response = self.client.get('/api/crops')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('total_crops'), 22)
        print("[PASS] test_09_api_crops")


if __name__ == '__main__':
    unittest.main()
