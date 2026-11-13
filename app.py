"""
app.py
Main Flask application for AI-Based Student Performance Prediction System.
Connects frontend templates with SQLite database and Scikit-learn Random Forest ML model.
"""

import os
import json
import functools
import numpy as np
import joblib
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session, jsonify, send_file
)
from werkzeug.security import generate_password_hash, check_password_hash

from database import get_db_connection, init_db, seed_demo_data
from recommendation_engine import generate_recommendations
from pdf_generator import generate_student_pdf_report
from train_model import MODEL_PATH, METRICS_PATH, train_and_evaluate

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'ai-student-perf-pred-secure-key-2026')

# Initialize DB on start
init_db()
seed_demo_data()

# Load or Train Machine Learning Model
ml_model = None
ml_metrics = {}


def load_or_train_ml():
    global ml_model, ml_metrics
    if not os.path.exists(MODEL_PATH):
        print("ML model file not found. Running training pipeline...")
        ml_model, ml_metrics = train_and_evaluate()
    else:
        try:
            ml_model = joblib.load(MODEL_PATH)
            print("Successfully loaded trained Random Forest model from disk.")
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                    ml_metrics = json.load(f)
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
            ml_model, ml_metrics = train_and_evaluate()


load_or_train_ml()


# Authentication Decorator
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if 'student_db_id' not in session:
            flash("Please sign in to access your student dashboard.", "warning")
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view


def get_current_student():
    """Helper to fetch logged-in student's DB record."""
    if 'student_db_id' not in session:
        return None
    conn = get_db_connection()
    student = conn.execute("SELECT * FROM students WHERE id = ?", (session['student_db_id'],)).fetchone()
    conn.close()
    return student


def compute_student_aggregates(student_id):
    """
    Computes cumulative statistics for a student:
    - total attended classes, total conducted, overall attendance %
    - average internal assessment score (scaled to 100)
    - latest prediction record
    - performance category
    """
    conn = get_db_connection()
    att_rows = conn.execute("SELECT * FROM attendance WHERE student_id = ?", (student_id,)).fetchall()
    mark_rows = conn.execute("SELECT * FROM internal_marks WHERE student_id = ?", (student_id,)).fetchall()
    latest_pred = conn.execute(
        "SELECT * FROM predictions WHERE student_id = ? ORDER BY created_at DESC LIMIT 1",
        (student_id,)
    ).fetchone()
    conn.close()

    total_att = sum(r['attended_classes'] for r in att_rows) if att_rows else 0
    total_cls = sum(r['total_classes'] for r in att_rows) if att_rows else 0
    overall_att_pct = round((total_att / total_cls * 100), 1) if total_cls > 0 else 0.0

    avg_internal = 0.0
    if mark_rows:
        total_scaled = sum(
            ((r['ia1'] + r['ia2'] + r['assignment_marks']) / 75.0 * 100) for r in mark_rows
        )
        avg_internal = round(total_scaled / len(mark_rows), 1)

    predicted_marks = latest_pred['predicted_marks'] if latest_pred else None
    category = latest_pred['category'] if latest_pred else None

    # Derive fallback category if no prediction yet
    if not category:
        if avg_internal >= 85:
            category = "Excellent"
        elif avg_internal >= 70:
            category = "Good"
        elif avg_internal >= 50:
            category = "Average"
        elif avg_internal > 0:
            category = "Needs Improvement"
        else:
            category = "Not Evaluated"

    return {
        'total_attended': total_att,
        'total_classes': total_cls,
        'overall_attendance_pct': overall_att_pct,
        'avg_internal': avg_internal,
        'predicted_marks': predicted_marks,
        'category': category,
        'latest_prediction': latest_pred,
        'attendance_count': len(att_rows),
        'marks_count': len(mark_rows)
    }


# ==========================================
# PUBLIC ROUTES: LANDING & AUTH
# ==========================================

@app.route('/')
def index():
    """Page 1: Landing Page with project logo, welcome, explanation, and CTA."""
    student = get_current_student()
    return render_template('index.html', student=student, metrics=ml_metrics)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Page 2: Student Registration."""
    if 'student_db_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        student_id = request.form.get('student_id', '').strip().upper()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        department = request.form.get('department', 'Computer Science & Engineering').strip()
        semester = request.form.get('semester', 5, type=int)

        if not name or not student_id or not email or not password:
            flash("All required fields must be filled out.", "danger")
            return render_template('register.html')

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check existing student_id or email
        existing = cursor.execute(
            "SELECT id FROM students WHERE email = ? OR student_id = ?",
            (email, student_id)
        ).fetchone()

        if existing:
            conn.close()
            flash("An account with this Email or Student ID already exists. Please login.", "warning")
            return redirect(url_for('login'))

        pass_hash = generate_password_hash(password)
        cursor.execute("""
            INSERT INTO students (name, student_id, email, password_hash, department, semester)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, student_id, email, pass_hash, department, semester))
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()

        # Auto login after registration
        session['student_db_id'] = new_id
        session['student_name'] = name
        session['student_roll'] = student_id
        flash(f"Welcome, {name}! Your student account was created successfully.", "success")
        return redirect(url_for('dashboard'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Page 2: Student Login."""
    if 'student_db_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()  # email or student_id
        password = request.form.get('password', '')

        if not identifier or not password:
            flash("Please enter both your Student ID / Email and password.", "danger")
            return render_template('login.html')

        conn = get_db_connection()
        student = conn.execute("""
            SELECT * FROM students 
            WHERE lower(email) = lower(?) OR upper(student_id) = upper(?)
        """, (identifier, identifier)).fetchone()
        conn.close()

        if student and check_password_hash(student['password_hash'], password):
            session['student_db_id'] = student['id']
            session['student_name'] = student['name']
            session['student_roll'] = student['student_id']
            flash(f"Welcome back, {student['name']}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid Student ID/Email or password. Please try again.", "danger")

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logs out current student and clears session."""
    session.clear()
    flash("You have been securely logged out.", "info")
    return redirect(url_for('login'))


# ==========================================
# STUDENT DASHBOARD (PAGE 3)
# ==========================================

@app.route('/dashboard')
@login_required
def dashboard():
    student = get_current_student()
    aggregates = compute_student_aggregates(student['id'])

    conn = get_db_connection()
    attendance_records = conn.execute(
        "SELECT * FROM attendance WHERE student_id = ? ORDER BY id ASC",
        (student['id'],)
    ).fetchall()

    internal_records = conn.execute(
        "SELECT * FROM internal_marks WHERE student_id = ? ORDER BY id ASC",
        (student['id'],)
    ).fetchall()

    recent_predictions = conn.execute(
        "SELECT * FROM predictions WHERE student_id = ? ORDER BY created_at DESC LIMIT 5",
        (student['id'],)
    ).fetchall()
    conn.close()

    # Generate quick recommendations for the dashboard snippet
    recs = generate_recommendations(
        attendance_pct=aggregates['overall_attendance_pct'],
        study_hours=3.5,
        internal_avg=aggregates['avg_internal'],
        assignment_avg=aggregates['avg_internal'],
        predicted_marks=aggregates['predicted_marks'],
        category=aggregates['category']
    )

    return render_template(
        'dashboard.html',
        student=student,
        aggregates=aggregates,
        attendance_records=attendance_records,
        internal_records=internal_records,
        recent_predictions=recent_predictions,
        top_recommendation=recs[0] if recs else None
    )


# ==========================================
# ATTENDANCE TRACKING (PAGE 4)
# ==========================================

@app.route('/attendance')
@login_required
def attendance():
    student = get_current_student()
    conn = get_db_connection()
    records = conn.execute(
        "SELECT * FROM attendance WHERE student_id = ? ORDER BY subject_code ASC",
        (student['id'],)
    ).fetchall()
    conn.close()

    # Calculate statistics & per-subject metrics
    enriched = []
    total_att = 0
    total_cls = 0
    low_attendance_warnings = []

    for r in records:
        att = r['attended_classes']
        tot = r['total_classes']
        pct = round((att / tot * 100), 1) if tot > 0 else 0.0
        total_att += att
        total_cls += tot

        # Determine indicator status: Green (>= 75%), Yellow (65%-74%), Red (< 65%)
        if pct >= 75.0:
            badge_class = 'success'
            status_text = 'Safe'
        elif pct >= 65.0:
            badge_class = 'warning'
            status_text = 'Borderline Warning'
        else:
            badge_class = 'danger'
            status_text = 'Critical Shortage'

        # Calculate needed classes to reach 75% OR classes that can be safely missed
        classes_needed = 0
        classes_can_miss = 0
        if tot > 0:
            if pct < 75.0:
                diff = (0.75 * tot) - att
                classes_needed = max(1, int(np.ceil(diff / 0.25)))
                low_attendance_warnings.append({
                    'subject': r['subject_name'],
                    'code': r['subject_code'],
                    'current_pct': pct,
                    'needed': classes_needed,
                    'status': status_text,
                    'badge': badge_class
                })
            else:
                max_total = int(np.floor(att / 0.75))
                classes_can_miss = max(0, max_total - tot)

        enriched.append({
            'id': r['id'],
            'subject_code': r['subject_code'],
            'subject_name': r['subject_name'],
            'attended': att,
            'total': tot,
            'percentage': pct,
            'badge': badge_class,
            'status': status_text,
            'classes_needed': classes_needed,
            'classes_can_miss': classes_can_miss
        })

    overall_pct = round((total_att / total_cls * 100), 1) if total_cls > 0 else 0.0

    return render_template(
        'attendance.html',
        student=student,
        records=enriched,
        total_attended=total_att,
        total_classes=total_cls,
        overall_pct=overall_pct,
        low_attendance_warnings=low_attendance_warnings
    )


@app.route('/attendance/add', methods=['POST'])
@login_required
def add_attendance():
    student = get_current_student()
    subject_code = request.form.get('subject_code', '').strip().upper()
    subject_name = request.form.get('subject_name', '').strip()
    attended = request.form.get('attended_classes', 0, type=int)
    total = request.form.get('total_classes', 0, type=int)

    if not subject_code or not subject_name or total < 0 or attended < 0:
        flash("Please enter valid subject code, name, and class counts.", "danger")
        return redirect(url_for('attendance'))

    if attended > total:
        flash("Attended classes cannot exceed total classes conducted.", "danger")
        return redirect(url_for('attendance'))

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO attendance (student_id, subject_code, subject_name, attended_classes, total_classes)
        VALUES (?, ?, ?, ?, ?)
    """, (student['id'], subject_code, subject_name, attended, total))
    conn.commit()
    conn.close()

    flash(f"Subject '{subject_name}' added to attendance tracking.", "success")
    return redirect(url_for('attendance'))


@app.route('/attendance/edit/<int:record_id>', methods=['POST'])
@login_required
def edit_attendance(record_id):
    student = get_current_student()
    attended = request.form.get('attended_classes', 0, type=int)
    total = request.form.get('total_classes', 0, type=int)

    if attended < 0 or total < 0 or attended > total:
        flash("Invalid class numbers provided.", "danger")
        return redirect(url_for('attendance'))

    conn = get_db_connection()
    conn.execute("""
        UPDATE attendance 
        SET attended_classes = ?, total_classes = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND student_id = ?
    """, (attended, total, record_id, student['id']))
    conn.commit()
    conn.close()

    flash("Attendance record updated successfully.", "success")
    return redirect(url_for('attendance'))


@app.route('/attendance/quick-log/<int:record_id>/<string:action>', methods=['POST'])
@login_required
def quick_log_attendance(record_id, action):
    """Allows 1-click attendance increment directly from subject cards."""
    student = get_current_student()
    conn = get_db_connection()
    rec = conn.execute("SELECT * FROM attendance WHERE id = ? AND student_id = ?", (record_id, student['id'])).fetchone()

    if not rec:
        conn.close()
        flash("Subject record not found.", "danger")
        return redirect(url_for('attendance'))

    att = rec['attended_classes']
    tot = rec['total_classes']

    if action == 'present':
        att += 1
        tot += 1
        msg = f"Marked PRESENT for {rec['subject_name']}."
    elif action == 'absent':
        tot += 1
        msg = f"Marked ABSENT for {rec['subject_name']}."
    else:
        conn.close()
        return redirect(url_for('attendance'))

    conn.execute("""
        UPDATE attendance 
        SET attended_classes = ?, total_classes = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND student_id = ?
    """, (att, tot, record_id, student['id']))
    conn.commit()
    conn.close()

    flash(msg, "success")
    return redirect(url_for('attendance'))


@app.route('/attendance/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_attendance(record_id):
    student = get_current_student()
    conn = get_db_connection()
    conn.execute("DELETE FROM attendance WHERE id = ? AND student_id = ?", (record_id, student['id']))
    conn.commit()
    conn.close()
    flash("Attendance record deleted.", "info")
    return redirect(url_for('attendance'))


# ==========================================
# INTERNAL ASSESSMENT MARKS (PAGE 5)
# ==========================================

@app.route('/internals')
@login_required
def internals():
    student = get_current_student()
    conn = get_db_connection()
    records = conn.execute(
        "SELECT * FROM internal_marks WHERE student_id = ? ORDER BY id ASC",
        (student['id'],)
    ).fetchall()
    conn.close()

    enriched = []
    total_ia1 = 0
    total_ia2 = 0
    total_assign = 0
    total_scaled_pct = 0

    for r in records:
        ia1 = r['ia1']
        ia2 = r['ia2']
        assign = r['assignment_marks']
        prev = r['previous_exam_marks']
        # IA1 (25) + IA2 (25) + Assign (25) = 75 total
        tot = ia1 + ia2 + assign
        scaled_pct = round((tot / 75.0 * 100), 1)

        total_ia1 += ia1
        total_ia2 += ia2
        total_assign += assign
        total_scaled_pct += scaled_pct

        enriched.append({
            'id': r['id'],
            'subject_name': r['subject_name'],
            'ia1': ia1,
            'ia2': ia2,
            'assignment_marks': assign,
            'previous_exam_marks': prev,
            'total_internal': round(tot, 1),
            'scaled_pct': scaled_pct
        })

    count = len(enriched)
    avg_pct = round(total_scaled_pct / count, 1) if count > 0 else 0.0

    return render_template(
        'internals.html',
        student=student,
        records=enriched,
        avg_pct=avg_pct,
        count=count
    )


@app.route('/internals/add', methods=['POST'])
@login_required
def add_internal():
    student = get_current_student()
    subject_name = request.form.get('subject_name', '').strip()
    ia1 = request.form.get('ia1', 0, type=float)
    ia2 = request.form.get('ia2', 0, type=float)
    assignment = request.form.get('assignment_marks', 0, type=float)
    previous = request.form.get('previous_exam_marks', 0, type=float)

    if not subject_name:
        flash("Subject name is required.", "danger")
        return redirect(url_for('internals'))

    if not (0 <= ia1 <= 25 and 0 <= ia2 <= 25 and 0 <= assignment <= 25 and 0 <= previous <= 100):
        flash("Please enter marks within standard limits (IA1: 0-25, IA2: 0-25, Assignment: 0-25, Previous Exam: 0-100).", "danger")
        return redirect(url_for('internals'))

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO internal_marks (student_id, subject_name, ia1, ia2, assignment_marks, previous_exam_marks)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (student['id'], subject_name, ia1, ia2, assignment, previous))
    conn.commit()
    conn.close()

    flash(f"Internal marks for '{subject_name}' recorded successfully.", "success")
    return redirect(url_for('internals'))


@app.route('/internals/edit/<int:record_id>', methods=['POST'])
@login_required
def edit_internal(record_id):
    student = get_current_student()
    ia1 = request.form.get('ia1', 0, type=float)
    ia2 = request.form.get('ia2', 0, type=float)
    assignment = request.form.get('assignment_marks', 0, type=float)
    previous = request.form.get('previous_exam_marks', 0, type=float)

    conn = get_db_connection()
    conn.execute("""
        UPDATE internal_marks 
        SET ia1 = ?, ia2 = ?, assignment_marks = ?, previous_exam_marks = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND student_id = ?
    """, (ia1, ia2, assignment, previous, record_id, student['id']))
    conn.commit()
    conn.close()

    flash("Internal marks updated successfully.", "success")
    return redirect(url_for('internals'))


@app.route('/internals/delete/<int:record_id>', methods=['POST'])
@login_required
def delete_internal(record_id):
    student = get_current_student()
    conn = get_db_connection()
    conn.execute("DELETE FROM internal_marks WHERE id = ? AND student_id = ?", (record_id, student['id']))
    conn.commit()
    conn.close()
    flash("Internal marks record removed.", "info")
    return redirect(url_for('internals'))


# ==========================================
# PERFORMANCE PREDICTION (PAGE 6)
# ==========================================

def calculate_performance_category(score):
    """
    Performance categories per specification:
    - 85–100: Excellent
    - 70–84: Good
    - 50–69: Average
    - Below 50: Needs Improvement
    """
    if score >= 85.0:
        return "Excellent", "success"
    elif score >= 70.0:
        return "Good", "primary"
    elif score >= 50.0:
        return "Average", "warning"
    else:
        return "Needs Improvement", "danger"


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    student = get_current_student()
    aggregates = compute_student_aggregates(student['id'])

    prediction_result = None

    if request.method == 'POST':
        try:
            subject_name = request.form.get('subject_name', 'Overall Examination').strip()
            attendance = float(request.form.get('attendance', 0))
            study_hours = float(request.form.get('study_hours', 0))
            internal_marks = float(request.form.get('internal_marks', 0))
            assignment_marks = float(request.form.get('assignment_marks', 0))
            previous_semester_marks = float(request.form.get('previous_semester_marks', 0))

            # Input validation
            if not (0 <= attendance <= 100):
                flash("Attendance must be between 0% and 100%.", "danger")
                return redirect(url_for('predict'))
            if not (0 <= study_hours <= 24):
                flash("Study hours must be between 0 and 24 hours.", "danger")
                return redirect(url_for('predict'))
            if not (0 <= internal_marks <= 100):
                flash("Internal marks must be between 0 and 100.", "danger")
                return redirect(url_for('predict'))
            if not (0 <= assignment_marks <= 100):
                flash("Assignment marks must be between 0 and 100.", "danger")
                return redirect(url_for('predict'))
            if not (0 <= previous_semester_marks <= 100):
                flash("Previous semester marks must be between 0 and 100.", "danger")
                return redirect(url_for('predict'))

            # Execute actual Random Forest ML prediction with feature names
            import pandas as pd
            input_df = pd.DataFrame([{
                'attendance': attendance,
                'study_hours': study_hours,
                'internal_marks': internal_marks,
                'assignment_marks': assignment_marks,
                'previous_semester_marks': previous_semester_marks
            }])

            raw_pred = ml_model.predict(input_df)[0]
            predicted_marks = round(float(np.clip(raw_pred, 0.0, 100.0)), 1)
            category, badge_color = calculate_performance_category(predicted_marks)

            # Store in SQLite database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    student_id, subject_name, attendance, study_hours, internal_marks,
                    assignment_marks, previous_semester_marks, predicted_marks, category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                student['id'], subject_name, attendance, study_hours, internal_marks,
                assignment_marks, previous_semester_marks, predicted_marks, category
            ))
            conn.commit()
            pred_id = cursor.lastrowid
            conn.close()

            prediction_result = {
                'id': pred_id,
                'subject_name': subject_name,
                'predicted_marks': predicted_marks,
                'category': category,
                'badge_color': badge_color,
                'inputs': {
                    'attendance': attendance,
                    'study_hours': study_hours,
                    'internal_marks': internal_marks,
                    'assignment_marks': assignment_marks,
                    'previous_semester_marks': previous_semester_marks
                }
            }

            flash(f"AI Prediction for {subject_name} Complete! Expected Final Marks: {predicted_marks}/100 ({category})", "success")

        except Exception as e:
            flash(f"Prediction failed: {str(e)}", "danger")

    # Fetch past predictions, subject attendance, and marks for this student
    conn = get_db_connection()
    past_predictions = conn.execute(
        "SELECT * FROM predictions WHERE student_id = ? ORDER BY created_at DESC LIMIT 10",
        (student['id'],)
    ).fetchall()

    att_rows = conn.execute(
        "SELECT * FROM attendance WHERE student_id = ? ORDER BY subject_code ASC",
        (student['id'],)
    ).fetchall()

    marks_rows = conn.execute(
        "SELECT * FROM internal_marks WHERE student_id = ? ORDER BY id ASC",
        (student['id'],)
    ).fetchall()
    conn.close()

    # Build Subject-Wise Attendance & AI Prediction Breakdown
    subject_predictions = []
    import pandas as pd
    for att in att_rows:
        tot_cls = att['total_classes']
        att_cls = att['attended_classes']
        att_pct = round((att_cls / tot_cls * 100), 1) if tot_cls > 0 else 0.0

        # Match marks record by subject name
        matched_mark = next((
            m for m in marks_rows
            if m['subject_name'].strip().lower() in att['subject_name'].strip().lower()
            or att['subject_name'].strip().lower() in m['subject_name'].strip().lower()
        ), None)

        if matched_mark:
            tot_internal = matched_mark['ia1'] + matched_mark['ia2'] + matched_mark['assignment_marks']
            scaled_internal = round((tot_internal / 75.0 * 100), 1)
            scaled_assign = round((matched_mark['assignment_marks'] / 25.0 * 100), 1)
            prev_mark = matched_mark['previous_exam_marks']
        else:
            scaled_internal = aggregates['avg_internal']
            scaled_assign = 80.0
            prev_mark = 72.0

        # AI forecast for this specific subject
        sub_df = pd.DataFrame([{
            'attendance': att_pct,
            'study_hours': 4.0,
            'internal_marks': scaled_internal,
            'assignment_marks': scaled_assign,
            'previous_semester_marks': prev_mark
        }])
        raw_sub = ml_model.predict(sub_df)[0]
        sub_pred_score = round(float(np.clip(raw_sub, 0.0, 100.0)), 1)
        sub_cat, sub_badge = calculate_performance_category(sub_pred_score)

        subject_predictions.append({
            'code': att['subject_code'],
            'name': att['subject_name'],
            'attendance_pct': att_pct,
            'attended': att_cls,
            'total': tot_cls,
            'internal_pct': scaled_internal,
            'assignment_pct': scaled_assign,
            'prev_exam': prev_mark,
            'predicted_score': sub_pred_score,
            'category': sub_cat,
            'badge_color': sub_badge
        })

    return render_template(
        'predict.html',
        student=student,
        aggregates=aggregates,
        result=prediction_result,
        past_predictions=past_predictions,
        subject_predictions=subject_predictions,
        metrics=ml_metrics
    )


# ==========================================
# IMPROVEMENT RECOMMENDATIONS (PAGE 7)
# ==========================================

@app.route('/recommendations')
@login_required
def recommendations():
    student = get_current_student()
    aggregates = compute_student_aggregates(student['id'])

    latest = aggregates['latest_prediction']
    study_hours = latest['study_hours'] if latest else 3.5
    assign_marks = latest['assignment_marks'] if latest else 75.0

    recs = generate_recommendations(
        attendance_pct=aggregates['overall_attendance_pct'],
        study_hours=study_hours,
        internal_avg=aggregates['avg_internal'],
        assignment_avg=assign_marks,
        predicted_marks=aggregates['predicted_marks'],
        category=aggregates['category']
    )

    return render_template(
        'recommendations.html',
        student=student,
        aggregates=aggregates,
        recommendations=recs,
        study_hours=study_hours
    )


# ==========================================
# PROGRESS REPORTS (PAGE 8) & PDF EXPORT
# ==========================================

@app.route('/reports')
@login_required
def reports():
    student = get_current_student()
    aggregates = compute_student_aggregates(student['id'])

    conn = get_db_connection()
    attendance_records = conn.execute(
        "SELECT * FROM attendance WHERE student_id = ? ORDER BY subject_code ASC",
        (student['id'],)
    ).fetchall()

    internal_records = conn.execute(
        "SELECT * FROM internal_marks WHERE student_id = ? ORDER BY id ASC",
        (student['id'],)
    ).fetchall()

    prediction_history = conn.execute(
        "SELECT * FROM predictions WHERE student_id = ? ORDER BY created_at ASC",
        (student['id'],)
    ).fetchall()
    conn.close()

    recs = generate_recommendations(
        attendance_pct=aggregates['overall_attendance_pct'],
        study_hours=3.5,
        internal_avg=aggregates['avg_internal'],
        assignment_avg=80.0,
        predicted_marks=aggregates['predicted_marks'],
        category=aggregates['category']
    )

    return render_template(
        'reports.html',
        student=student,
        aggregates=aggregates,
        attendance_records=attendance_records,
        internal_records=internal_records,
        prediction_history=prediction_history,
        recommendations=recs
    )


@app.route('/download-report-pdf')
@login_required
def download_report_pdf():
    """Generates an official institutional PDF report and sends as attachment."""
    student = get_current_student()
    aggregates = compute_student_aggregates(student['id'])

    conn = get_db_connection()
    attendance_records = conn.execute(
        "SELECT * FROM attendance WHERE student_id = ? ORDER BY subject_code ASC",
        (student['id'],)
    ).fetchall()

    internal_records = conn.execute(
        "SELECT * FROM internal_marks WHERE student_id = ? ORDER BY id ASC",
        (student['id'],)
    ).fetchall()
    conn.close()

    recs = generate_recommendations(
        attendance_pct=aggregates['overall_attendance_pct'],
        study_hours=3.5,
        internal_avg=aggregates['avg_internal'],
        assignment_avg=78.0,
        predicted_marks=aggregates['predicted_marks'],
        category=aggregates['category']
    )

    pdf_buffer = generate_student_pdf_report(
        student=student,
        attendance_records=attendance_records,
        internals_records=internal_records,
        latest_prediction=aggregates['latest_prediction'],
        recommendations=recs
    )

    filename = f"Performance_Report_{student['student_id']}.pdf"
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )


# ==========================================
# REST API FOR CHART.JS & DEMO SEEDING
# ==========================================

@app.route('/api/student-analytics')
@login_required
def api_student_analytics():
    """Returns JSON analytics for responsive Chart.js rendering."""
    student = get_current_student()
    conn = get_db_connection()
    att_rows = conn.execute(
        "SELECT subject_code, subject_name, attended_classes, total_classes FROM attendance WHERE student_id = ?",
        (student['id'],)
    ).fetchall()

    marks_rows = conn.execute(
        "SELECT subject_name, ia1, ia2, assignment_marks, previous_exam_marks FROM internal_marks WHERE student_id = ?",
        (student['id'],)
    ).fetchall()

    preds_rows = conn.execute(
        "SELECT predicted_marks, category, created_at FROM predictions WHERE student_id = ? ORDER BY created_at ASC",
        (student['id'],)
    ).fetchall()
    conn.close()

    attendance_data = {
        'labels': [r['subject_code'] for r in att_rows],
        'names': [r['subject_name'] for r in att_rows],
        'percentages': [
            round((r['attended_classes'] / r['total_classes'] * 100), 1) if r['total_classes'] > 0 else 0
            for r in att_rows
        ]
    }

    marks_data = {
        'labels': [r['subject_name'] for r in marks_rows],
        'ia1': [r['ia1'] for r in marks_rows],
        'ia2': [r['ia2'] for r in marks_rows],
        'assignments': [r['assignment_marks'] for r in marks_rows],
        'totals': [round(r['ia1'] + r['ia2'] + r['assignment_marks'], 1) for r in marks_rows]
    }

    preds_data = {
        'timestamps': [p['created_at'][:10] for p in preds_rows],
        'scores': [p['predicted_marks'] for p in preds_rows],
        'categories': [p['category'] for p in preds_rows]
    }

    return jsonify({
        'attendance': attendance_data,
        'marks': marks_data,
        'predictions': preds_data
    })


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """AJAX endpoint for real-time what-if simulations without page reload."""
    try:
        data = request.get_json() or {}
        attendance = float(data.get('attendance', 75.0))
        study_hours = float(data.get('study_hours', 3.5))
        internal_marks = float(data.get('internal_marks', 70.0))
        assignment_marks = float(data.get('assignment_marks', 75.0))
        previous_semester_marks = float(data.get('previous_semester_marks', 70.0))

        import pandas as pd
        input_df = pd.DataFrame([{
            'attendance': attendance,
            'study_hours': study_hours,
            'internal_marks': internal_marks,
            'assignment_marks': assignment_marks,
            'previous_semester_marks': previous_semester_marks
        }])

        raw_pred = ml_model.predict(input_df)[0]
        predicted_marks = round(float(np.clip(raw_pred, 0.0, 100.0)), 1)
        category, badge_color = calculate_performance_category(predicted_marks)

        return jsonify({
            'success': True,
            'predicted_marks': predicted_marks,
            'category': category,
            'badge_color': badge_color
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/download-dataset')
@login_required
def download_dataset():
    """Provides download access to the 1,500 record student performance CSV."""
    from train_model import DATA_PATH
    if os.path.exists(DATA_PATH):
        return send_file(DATA_PATH, as_attachment=True, download_name='student_performance_dataset.csv', mimetype='text/csv')
    return "Dataset file not found", 404


@app.route('/seed-demo-records', methods=['POST'])
@login_required
def seed_demo_records():
    """Allows any newly registered student to auto-populate sample coursework for quick testing."""
    student = get_current_student()
    conn = get_db_connection()

    # Clear previous if empty
    conn.execute("DELETE FROM attendance WHERE student_id = ?", (student['id'],))
    conn.execute("DELETE FROM internal_marks WHERE student_id = ?", (student['id'],))

    sample_att = [
        ("CS601", "Design & Analysis of Algorithms", 39, 42),
        ("CS602", "Operating Systems & Virtualization", 33, 40),
        ("CS603", "Database Management Systems", 27, 38),   # ~71%
        ("CS604", "Computer Communication Networks", 22, 38), # ~57.9%
        ("CS605", "Machine Learning & AI", 37, 40)
    ]
    for code, name, a, t in sample_att:
        conn.execute(
            "INSERT INTO attendance (student_id, subject_code, subject_name, attended_classes, total_classes) VALUES (?, ?, ?, ?, ?)",
            (student['id'], code, name, a, t)
        )

    sample_internals = [
        ("Design & Analysis of Algorithms", 22.0, 23.5, 24.0, 84.0),
        ("Operating Systems & Virtualization", 20.0, 21.0, 21.5, 76.0),
        ("Database Management Systems", 17.0, 18.0, 19.0, 70.0),
        ("Computer Communication Networks", 15.0, 16.5, 17.5, 64.0),
        ("Machine Learning & AI", 24.0, 24.5, 25.0, 90.0)
    ]
    for name, ia1, ia2, assign, prev in sample_internals:
        conn.execute(
            "INSERT INTO internal_marks (student_id, subject_name, ia1, ia2, assignment_marks, previous_exam_marks) VALUES (?, ?, ?, ?, ?, ?)",
            (student['id'], name, ia1, ia2, assign, prev)
        )

    conn.commit()
    conn.close()

    flash("Sample coursework & attendance records populated successfully!", "success")
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
