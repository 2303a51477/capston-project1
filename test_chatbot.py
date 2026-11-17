"""
test_chatbot.py
Comprehensive automated test suite for AI Bot Assistant:
- Route authentication and permissions
- React app container delivery
- Chatbot status and context grounding APIs
- Conversation creation, listing, messaging, clearing, and deletion
- Gemini service formatting and fallback handling
- Persistent storage engine (MongoDB / SQLite fallback)
"""

import unittest
import json
from app import app
import chat_storage
import gemini_service


class TestAIChatbotAssistant(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()

    def _login(self):
        """Helper to authenticate demo student."""
        return self.client.post('/login', data={
            'identifier': 'rahul.student@college.edu',
            'password': 'password123'
        }, follow_redirects=True)

    def test_01_chatbot_page_requires_auth(self):
        """Verify unauthenticated access redirects to login."""
        response = self.client.get('/chatbot')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers.get('Location', ''))

    def test_02_chatbot_page_authenticated(self):
        """Verify authenticated student can access the React chatbot page."""
        with self.client:
            self._login()
            response = self.client.get('/chatbot')
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"chatbot-app-root", response.data)
            self.assertIn(b"chatbot.jsx", response.data)
            self.assertIn(b"react.production.min.js", response.data)
            self.assertIn(b"AI Bot Assistant", response.data)

    def test_03_api_chat_status(self):
        """Verify AI system status endpoint returns model, student, and storage details."""
        with self.client:
            self._login()
            response = self.client.get('/api/chat/status')
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data['success'])
            self.assertIn('storage', data)
            self.assertIn('student', data)
            self.assertEqual(data['student']['student_id'], 'CS2026-042')
            self.assertIn('model', data)

    def test_04_api_student_context(self):
        """Verify live student academic grounding snapshot returned via API."""
        with self.client:
            self._login()
            response = self.client.get('/api/chat/student-context')
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data['success'])
            ctx = data['context']
            self.assertEqual(ctx['student']['name'], 'Rahul Sharma')
            self.assertIn('aggregates', ctx)
            self.assertIn('overall_attendance_pct', ctx['aggregates'])
            self.assertIn('attendance', ctx)
            self.assertIn('marks', ctx)
            self.assertTrue(len(ctx['attendance']) > 0)

    def test_05_conversation_crud_lifecycle(self):
        """Verify conversation thread creation, retrieval, and deletion."""
        with self.client:
            self._login()

            # 1. Create a new conversation
            res_new = self.client.post('/api/chat/conversations/new', json={
                'title': 'Test Algorithms Thread'
            })
            self.assertEqual(res_new.status_code, 200)
            new_data = res_new.get_json()
            self.assertTrue(new_data['success'])
            conv_id = new_data['conversation']['id']
            self.assertEqual(new_data['conversation']['title'], 'Test Algorithms Thread')

            # 2. Verify it appears in conversations list
            res_list = self.client.get('/api/chat/conversations')
            self.assertEqual(res_list.status_code, 200)
            list_data = res_list.get_json()
            self.assertTrue(any(c['id'] == conv_id for c in list_data['conversations']))

            # 3. Retrieve messages (should be empty initially)
            res_msgs = self.client.get(f'/api/chat/conversations/{conv_id}/messages')
            self.assertEqual(res_msgs.status_code, 200)
            msgs_data = res_msgs.get_json()
            self.assertEqual(len(msgs_data['messages']), 0)

            # 4. Delete the conversation
            res_del = self.client.delete(f'/api/chat/conversations/{conv_id}')
            self.assertEqual(res_del.status_code, 200)
            del_data = res_del.get_json()
            self.assertTrue(del_data['success'])

            # 5. Verify it's no longer present
            res_msgs_after = self.client.get(f'/api/chat/conversations/{conv_id}/messages')
            self.assertEqual(res_msgs_after.status_code, 404)

    def test_06_send_chat_message(self):
        """Verify sending a message stores user prompt and generates an AI response."""
        with self.client:
            self._login()

            # Create new thread
            res_new = self.client.post('/api/chat/conversations/new', json={'title': 'Python Question'})
            conv_id = res_new.get_json()['conversation']['id']

            # Post message to API
            res_msg = self.client.post('/api/chat/message', json={
                'conversation_id': conv_id,
                'message': 'How does binary search work in Python?'
            })
            self.assertEqual(res_msg.status_code, 200)
            msg_data = res_msg.get_json()
            self.assertIn('message', msg_data)
            self.assertEqual(msg_data['conversation_id'], conv_id)
            self.assertTrue(len(msg_data['message']) > 0)

            # Verify both messages (user & AI) are saved in storage
            res_msgs = self.client.get(f'/api/chat/conversations/{conv_id}/messages')
            msgs = res_msgs.get_json()['messages']
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]['role'], 'user')
            self.assertEqual(msgs[0]['content'], 'How does binary search work in Python?')
            self.assertEqual(msgs[1]['role'], 'model')

            # Test clear messages
            res_clear = self.client.post(f'/api/chat/conversations/{conv_id}/clear')
            self.assertEqual(res_clear.status_code, 200)
            res_msgs_after = self.client.get(f'/api/chat/conversations/{conv_id}/messages')
            self.assertEqual(len(res_msgs_after.get_json()['messages']), 0)

            # Cleanup
            self.client.delete(f'/api/chat/conversations/{conv_id}')

    def test_07_gemini_context_formatting(self):
        """Verify academic context formatting accurately extracts real student data."""
        context = gemini_service.format_student_academic_context(1)
        self.assertIn("Rahul Sharma", context)
        self.assertIn("CS2026-042", context)
        self.assertIn("RandomForestRegressor", context)
        self.assertIn("COURSE-WISE ATTENDANCE STATUS", context)
        self.assertIn("COURSE-WISE INTERNAL ASSESSMENT MARKS", context)
        self.assertIn("AI PREDICTION ENGINE FACTS", context)

    def test_08_gemini_smart_title(self):
        """Verify title generator produces readable titles."""
        t1 = gemini_service.generate_smart_title("Can you explain how quicksort works in Java?")
        self.assertTrue(len(t1.split()) <= 6)
        t2 = gemini_service.generate_smart_title("What is my attendance?")
        self.assertIn("attendance", t2.lower())

    def test_09_gemini_live_generation_flow_with_mock(self):
        """Verify full response generation pipeline with Gemini AI mock."""
        from unittest.mock import patch, MagicMock

        mock_response = MagicMock()
        mock_response.text = (
            "### Binary Search in Python\n\n"
            "Binary search operates on sorted arrays with **O(log n)** time complexity.\n\n"
            "```python\ndef binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n```\n\n"
            "Line-by-line explanation:\n"
            "1. `low, high = 0, len(arr) - 1`: Initializes search boundary.\n"
            "2. `mid = (low + high) // 2`: Finds middle index."
        )

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response

        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat

        with patch('gemini_service.is_api_configured', return_value=True):
            with patch('google.generativeai.GenerativeModel', return_value=mock_model):
                res = gemini_service.generate_gemini_response(
                    user_message="Write a binary search algorithm in Python",
                    conversation_history=[],
                    student_id=1
                )
                self.assertTrue(res['success'])
                self.assertIn("binary_search", res['content'])
                self.assertIn("```python", res['content'])
                self.assertIn("O(log n)", res['content'])
                self.assertEqual(res['model'], gemini_service.CANDIDATE_MODELS[0])


if __name__ == '__main__':
    unittest.main()
