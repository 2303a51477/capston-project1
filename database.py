"""
database.py
Database management and SQLite schema for AI-Based Student Performance Prediction System.
Handles database initialization, schema migrations, and sample data seeding.
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), 'database.sqlite')


def get_db_connection():
    """Returns a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Initializes SQLite database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Students Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        student_id TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        department TEXT DEFAULT 'Computer Science & Engineering',
        semester INTEGER DEFAULT 5,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. Attendance Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        subject_code TEXT NOT NULL,
        subject_name TEXT NOT NULL,
        attended_classes INTEGER NOT NULL DEFAULT 0,
        total_classes INTEGER NOT NULL DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
    )
    """)

    # 3. Internal Assessment Marks Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS internal_marks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        subject_name TEXT NOT NULL,
        ia1 REAL NOT NULL DEFAULT 0.0,
        ia2 REAL NOT NULL DEFAULT 0.0,
        assignment_marks REAL NOT NULL DEFAULT 0.0,
        previous_exam_marks REAL NOT NULL DEFAULT 0.0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
    )
    """)

    # 4. Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        subject_name TEXT DEFAULT 'Overall Examination',
        attendance REAL NOT NULL,
        study_hours REAL NOT NULL,
        internal_marks REAL NOT NULL,
        assignment_marks REAL NOT NULL,
        previous_semester_marks REAL NOT NULL,
        predicted_marks REAL NOT NULL,
        category TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
    )
    """)

    # Ensure subject_name column exists for existing databases
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [row['name'] for row in cursor.fetchall()]
    if 'subject_name' not in columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN subject_name TEXT DEFAULT 'Overall Examination'")
        print("Migrated predictions table: added subject_name column.")

    conn.commit()
    conn.close()
    print("Database tables initialized successfully.")


def seed_demo_data():
    """Seeds a demonstration student account with realistic records if not present."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM students WHERE email = 'rahul.student@college.edu'")
    existing = cursor.fetchone()

    if not existing:
        demo_pass_hash = generate_password_hash("password123")
        cursor.execute("""
            INSERT INTO students (name, student_id, email, password_hash, department, semester)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "Rahul Sharma",
            "CS2026-042",
            "rahul.student@college.edu",
            demo_pass_hash,
            "Computer Science & Engineering",
            6
        ))
        student_id = cursor.lastrowid

        # Seed realistic attendance records
        sample_attendance = [
            ("CS601", "Design & Analysis of Algorithms", 38, 42),
            ("CS602", "Operating Systems & Virtualization", 32, 40),
            ("CS603", "Database Management Systems", 26, 38),   # 68.4% - Yellow warning
            ("CS604", "Computer Communication Networks", 22, 38), # 57.9% - Red critical
            ("CS605", "Machine Learning & AI", 36, 40),
        ]
        for code, name, attended, total in sample_attendance:
            cursor.execute("""
                INSERT INTO attendance (student_id, subject_code, subject_name, attended_classes, total_classes)
                VALUES (?, ?, ?, ?, ?)
            """, (student_id, code, name, attended, total))

        # Seed internal assessment records (IA1 out of 25, IA2 out of 25, Assignment out of 25, Prev Exam out of 100)
        sample_internals = [
            ("Design & Analysis of Algorithms", 22.5, 23.0, 23.5, 82.0),
            ("Operating Systems & Virtualization", 19.0, 21.0, 22.0, 75.0),
            ("Database Management Systems", 16.5, 17.5, 19.0, 68.0),
            ("Computer Communication Networks", 14.0, 16.0, 18.0, 62.0),
            ("Machine Learning & AI", 23.5, 24.5, 24.0, 89.0),
        ]
        for name, ia1, ia2, assign, prev in sample_internals:
            cursor.execute("""
                INSERT INTO internal_marks (student_id, subject_name, ia1, ia2, assignment_marks, previous_exam_marks)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (student_id, name, ia1, ia2, assign, prev))

        # Seed initial realistic prediction history
        sample_predictions = [
            (78.5, 3.5, 74.0, 80.0, 72.0, 75.4, "Good", "2026-10-15 10:30:00"),
            (80.2, 4.0, 78.5, 84.0, 74.0, 78.8, "Good", "2026-10-28 14:15:00"),
            (82.0, 4.5, 82.0, 86.5, 75.2, 82.1, "Good", "2026-11-04 16:45:00")
        ]
        for att, hours, imarks, amarks, pmarks, pscore, cat, dt in sample_predictions:
            cursor.execute("""
                INSERT INTO predictions (student_id, attendance, study_hours, internal_marks, assignment_marks, previous_semester_marks, predicted_marks, category, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (student_id, att, hours, imarks, amarks, pmarks, pscore, cat, dt))

        conn.commit()
        print("Demo student data seeded successfully: rahul.student@college.edu / password123")
    else:
        print("Demo student already exists.")

    conn.close()


if __name__ == '__main__':
    init_db()
    seed_demo_data()

