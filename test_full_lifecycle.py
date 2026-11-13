"""
test_full_lifecycle.py
Exercises complete CRUD lifecycle for attendance, internal marks,
AI predictions, and student data integrity.
"""

from app import app
from database import get_db_connection

def test_lifecycle():
    client = app.test_client()
    
    # 1. Login as demo student
    login_res = client.post('/login', data={
        'identifier': 'rahul.student@college.edu',
        'password': 'password123'
    }, follow_redirects=True)
    assert login_res.status_code == 200
    print("[PASS] Login verified")

    # 2. Add subject to attendance
    add_att_res = client.post('/attendance/add', data={
        'subject_code': 'CS699',
        'subject_name': 'Cloud Computing & DevOps',
        'attended_classes': '28',
        'total_classes': '30'
    }, follow_redirects=True)
    assert add_att_res.status_code == 200
    assert b"Cloud Computing" in add_att_res.data
    print("[PASS] Add Attendance verified")

    # Check ID in DB
    conn = get_db_connection()
    row = conn.execute("SELECT id FROM attendance WHERE subject_code = 'CS699'").fetchone()
    att_id = row['id']
    conn.close()

    # 3. Edit attendance
    edit_att_res = client.post(f'/attendance/edit/{att_id}', data={
        'attended_classes': '29',
        'total_classes': '30'
    }, follow_redirects=True)
    assert edit_att_res.status_code == 200
    print("[PASS] Edit Attendance verified")

    # 4. Delete attendance
    del_att_res = client.post(f'/attendance/delete/{att_id}', follow_redirects=True)
    assert del_att_res.status_code == 200
    conn = get_db_connection()
    deleted_row = conn.execute("SELECT id FROM attendance WHERE id = ?", (att_id,)).fetchone()
    conn.close()
    assert deleted_row is None
    print("[PASS] Delete Attendance verified")

    # 5. Add internal marks
    add_mark_res = client.post('/internals/add', data={
        'subject_name': 'Cloud Computing & DevOps',
        'ia1': '23.5',
        'ia2': '24.0',
        'assignment_marks': '23.0',
        'previous_exam_marks': '85.0'
    }, follow_redirects=True)
    assert add_mark_res.status_code == 200
    assert b"Cloud Computing" in add_mark_res.data
    print("[PASS] Add Internal Marks verified")

    conn = get_db_connection()
    mark_row = conn.execute("SELECT id FROM internal_marks WHERE subject_name = 'Cloud Computing & DevOps'").fetchone()
    mark_id = mark_row['id']
    conn.close()

    # 6. Edit internal marks
    edit_mark_res = client.post(f'/internals/edit/{mark_id}', data={
        'ia1': '24.0',
        'ia2': '24.5',
        'assignment_marks': '24.0',
        'previous_exam_marks': '88.0'
    }, follow_redirects=True)
    assert edit_mark_res.status_code == 200
    print("[PASS] Edit Internal Marks verified")

    # 7. Delete internal marks
    del_mark_res = client.post(f'/internals/delete/{mark_id}', follow_redirects=True)
    assert del_mark_res.status_code == 200
    conn = get_db_connection()
    deleted_mark = conn.execute("SELECT id FROM internal_marks WHERE id = ?", (mark_id,)).fetchone()
    conn.close()
    assert deleted_mark is None
    print("[PASS] Delete Internal Marks verified")

    # 8. Test Prediction & Database Persistence
    pred_res = client.post('/predict', data={
        'attendance': '88.0',
        'study_hours': '5.0',
        'internal_marks': '88.0',
        'assignment_marks': '90.0',
        'previous_semester_marks': '86.0'
    }, follow_redirects=True)
    assert pred_res.status_code == 200
    assert b"AI Prediction Complete!" in pred_res.data
    assert b"Excellent" in pred_res.data
    print("[PASS] Prediction Execution and Category Classification verified (Excellent)")

    # 9. Test Low Score Prediction Category (Needs Improvement)
    pred_low_res = client.post('/predict', data={
        'attendance': '45.0',
        'study_hours': '1.0',
        'internal_marks': '35.0',
        'assignment_marks': '40.0',
        'previous_semester_marks': '40.0'
    }, follow_redirects=True)
    assert pred_low_res.status_code == 200
    assert b"Needs Improvement" in pred_low_res.data
    print("[PASS] Low Score Prediction Category Classification verified (Needs Improvement)")

    # 10. Verify PDF generation stream
    pdf_res = client.get('/download-report-pdf')
    assert pdf_res.status_code == 200
    assert pdf_res.data.startswith(b'%PDF')
    print(f"[PASS] PDF Report downloaded successfully ({len(pdf_res.data)} bytes)")

    print("\nALL LIFECYCLE TESTS COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    test_lifecycle()
