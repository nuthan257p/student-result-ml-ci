import unittest
from fastapi.testclient import TestClient
from app import app

class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_high_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 85,
                "assignment_marks": 88,
                "previous_score": 80
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "PASS")

    def test_low_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 55,
                "internal_marks": 30,
                "assignment_marks": 40,
                "previous_score": 35
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction"], "FAIL")

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 85
            }
        )
        self.assertEqual(response.status_code, 422)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("student_api_model_loaded", response.text)
        self.assertIn("student_api_http_requests_total", response.text)

if __name__ == "__main__":
    unittest.main()
