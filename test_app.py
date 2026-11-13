"""
test_app.py
Comprehensive test suite verifying all routes, database operations,
ML predictions, and PDF report downloads.
"""

import unittest
import time
from app import app
from database import get_db_connection


class TestStudentPredictionSystem(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()

    def test_01_landing_page(self):
        """Test landing page loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AI-Based Student", response.data)
        self.assertIn(b"EduPredict AI", response.data)

    def test_02_demo_login(self):
        """Test demo student login with seeded credentials."""
        response = self.client.post('/login', data={
            'identifier': 'rahul.student@college.edu',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome back", response.data)
        self.assertIn(b"Rahul Sharma", response.data)

    def test_03_dashboard_authenticated(self):
        """Test dashboard view displays all metrics and cards."""
        with self.client:
            self.client.post('/login', data={
                'identifier': 'rahul.student@college.edu',
                'password': 'password123'
            })
            response = self.client.get('/dashboard')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Total Attendance", response.data)
            self.assertIn(b"Avg Internal Marks", response.data)
            self.assertIn(b"Predicted Final Mark", response.data)

    def test_04_attendance_page(self):
        """Test attendance tracking page."""
        with self.client:
            self.client.post('/login', data={
                'identifier': 'rahul.student@college.edu',
                'password': 'password123'
            })
            response = self.client.get('/attendance')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Attendance Tracking", response.data)
            self.assertIn(b"CS601", response.data)
            self.assertIn(b"Safe", response.data)

    def test_05_internals_page(self):
        """Test internal assessment marks page."""
        with self.client:
            self.client.post('/login', data={
                'identifier': 'rahul.student@college.edu',
                'password': 'password123'
            })
            response = self.client.get('/internals')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Internal Assessment Marks", response.data)
            self.assertIn(b"Design &amp; Analysis of Algorithms", response.data)

    def test_06_prediction_flow(self):
        """Test AI-based final marks prediction with Random Forest model."""
        with self.client:
            self.client.post('/login', data={
                'identifier': 'rahul.student@college.edu',
                'password': 'password123'
            })
            # Submit prediction input
            response = self.client.post('/predict', data={
                'attendance': '82.5',
                'study_hours': '4.5',
                'internal_marks': '80.0',
                'assignment_marks': '85.0',
                'previous_semester_marks': '78.0'
            }, follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"AI Prediction Complete!", response.data)
            self.assertIn(b"Expected Final Exam Marks", response.data)

    def test_07_recommendations_page(self):
        """Test personalized improvement recommendations page."""
        with self.client:
            self.client.post('/login', data={
                'identifier': 'rahul.student@college.edu',
                'password': 'password123'
            })
            response = self.client.get('/recommendations')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Personalized Improvement Recommendations", response.data)
            self.assertIn(b"Daily Study Timetable", response.data)

    def test_08_reports_and_pdf_download(self):
        """Test progress reports page and official PDF report generation."""
        with self.client:
            self.client.post('/login', data={
                'identifier': 'rahul.student@college.edu',
                'password': 'password123'
            })
            # Check reports page
            response = self.client.get('/reports')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Academic Progress", response.data)

            # Test PDF download endpoint
            pdf_response = self.client.get('/download-report-pdf')
            self.assertEqual(pdf_response.status_code, 200)
            self.assertEqual(pdf_response.content_type, 'application/pdf')
            self.assertTrue(len(pdf_response.data) > 1000)
            self.assertTrue(pdf_response.data.startswith(b'%PDF'))

    def test_09_api_analytics(self):
        """Test JSON analytics API for Chart.js."""
        with self.client:
            self.client.post('/login', data={
                'identifier': 'rahul.student@college.edu',
                'password': 'password123'
            })
            response = self.client.get('/api/student-analytics')
            self.assertEqual(response.status_code, 200)
            json_data = response.get_json()
            self.assertIn('attendance', json_data)
            self.assertIn('marks', json_data)
            self.assertIn('predictions', json_data)

    def test_10_registration_new_student(self):
        """Test registering a brand new student and verifying isolated data."""
        unique_suffix = int(time.time())
        email = f"student_{unique_suffix}@college.edu"
        roll = f"CS-{unique_suffix % 10000}"

        response = self.client.post('/register', data={
            'name': 'Priya Patel',
            'student_id': roll,
            'email': email,
            'password': 'password456',
            'department': 'Artificial Intelligence & Data Science',
            'semester': '6'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome, Priya Patel", response.data)

        # Cleanup test registration
        conn = get_db_connection()
        conn.execute("DELETE FROM students WHERE email = ?", (email,))
        conn.commit()
        conn.close()


if __name__ == '__main__':
    unittest.main()
