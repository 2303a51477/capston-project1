# AI-Based Student Performance Prediction System

An end-to-end, modern, and user-friendly web application designed to help college students understand, monitor, and improve their academic performance using Artificial Intelligence and Machine Learning.

---

## 📌 Project Overview

College students often find it challenging to know whether their daily study habits, attendance percentage, and internal marks will result in a good final examination grade before it is too late.

The **AI-Based Student Performance Prediction System** bridges this gap:
- Students can log their subject-wise attendance and internal assessment marks.
- A trained **Random Forest Regression Machine Learning model** forecasts their expected final exam marks (out of 100) and categorizes their performance (*Excellent*, *Good*, *Average*, *Needs Improvement*).
- An intelligent recommendation engine provides actionable advice in plain language (e.g. minimum consecutive classes to attend to reach 75%, focused study timetable, practice question papers).
- Students can generate and download an official university-grade academic performance report as a **PDF**.

---

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3.3, Bootstrap Icons, Chart.js 4.4 |
| **Backend** | Python 3.11+, Flask 3.1, Werkzeug Security |
| **Machine Learning** | Scikit-learn (RandomForestRegressor), Pandas, NumPy, Joblib |
| **Database** | SQLite3 (normalized relational schema with foreign keys) |
| **PDF Generation** | ReportLab 5.1 |

---

## 📂 Project Folder Structure

```text
capston1 project/
├── app.py                      # Main Flask application entry point and route handlers
├── database.py                 # SQLite database initialization & seed utility
├── train_model.py              # ML pipeline: dataset generation, model training & evaluation
├── recommendation_engine.py    # Personalized improvement suggestions & study timetable logic
├── pdf_generator.py            # Official institutional PDF report generator (ReportLab)
├── requirements.txt            # Python dependencies
├── database.sqlite             # SQLite database file
├── README.md                   # Complete documentation & execution guide
├── test_app.py                 # Unit & integration test suite (10 test cases)
├── test_full_lifecycle.py      # End-to-end CRUD and ML verification script
├── data/
│   └── student_performance.csv # CSV dataset (1,500 records)
├── model/
│   ├── student_model.joblib    # Trained Random Forest Regression model
│   └── model_metrics.json      # Model performance metrics (MAE, MSE, R²)
├── static/
│   ├── css/
│   │   └── style.css           # Clean white & light-blue college dashboard theme
│   ├── js/
│   │   ├── main.js             # Mobile drawer toggle, autofill & target calculators
│   │   └── charts.js           # Chart.js renderers (doughnut, bar, line, radar)
│   └── img/
│       ├── logo.svg            # Modern academic crest logo
│       └── student_hero.svg    # Vector student illustration for landing page
└── templates/
    ├── base.html               # Master layout with responsive sidebar & navbar
    ├── index.html              # Page 1: Home / Landing page
    ├── login.html              # Page 2: Student Login
    ├── register.html           # Page 2: Student Registration
    ├── dashboard.html          # Page 3: Student Dashboard
    ├── attendance.html         # Page 4: Attendance Tracking (75% indicator)
    ├── internals.html          # Page 5: Internal Assessment Marks
    ├── predict.html            # Page 6: AI Performance Prediction
    ├── recommendations.html    # Page 7: Personalized Improvement Suggestions
    └── reports.html            # Page 8: Progress Reports & PDF Export
```

---

## 🚀 Website Pages & Features

### 1. Home / Landing Page (`/`)
- Modern clean hero banner with custom student vector illustrations.
- Key feature overview cards and 3-step explanation of how the system works.
- Direct links to **Get Started**, **Student Login**, and **Registration**.

### 2. Student Registration & Login (`/register`, `/login`)
- Clean forms for Student Name, Student ID (Roll No), Email, Password, Department, and Semester.
- Passwords securely hashed with `werkzeug.security.generate_password_hash`.
- One-click **Demo Login** autofill button for instant testing.

### 3. Student Dashboard (`/dashboard`)
- Personalized welcome card with student name and department info.
- 4 High-impact metric cards:
  - **Total Attendance Percentage** (with status indicator)
  - **Average Internal Marks** (scaled out of 100)
  - **Predicted Final Exam Marks** (out of 100)
  - **Performance Status Category** (*Excellent*, *Good*, *Average*, *Needs Improvement*)
- Interactive **Chart.js** attendance doughnut and subject internal marks breakdown.
- Recent AI prediction history table and improvement spotlight.

### 4. Attendance Tracking (`/attendance`)
- Subject-wise table displaying attended classes, total classes, and calculated percentage.
- **Color-coded attendance indicators**:
  - 🟢 **Green (Safe):** 75.0% and above
  - 🟡 **Yellow (Borderline Warning):** 65.0% – 74.9%
  - 🔴 **Red (Critical Shortage):** Below 65.0%
- **Attendance Recovery Calculator**: Computes exactly how many consecutive classes a student must attend to reach the mandatory 75% threshold.
- Add, Edit, and Delete subject attendance records with instant recalculation.

### 5. Internal Assessment Marks (`/internals`)
- Record continuous evaluation marks:
  - Internal Assessment 1 (IA1, max 25)
  - Internal Assessment 2 (IA2, max 25)
  - Assignment Marks (max 25)
  - Previous Examination Marks (max 100)
- Automatic calculation of cumulative internal totals (out of 75) and scaled percentage (out of 100).
- Add, Edit, and Delete internal marks per subject.

### 6. AI Performance Prediction (`/predict`)
- Predictive input form:
  - Attendance percentage (0–100%)
  - Study hours per day (0–24 hrs)
  - Internal assessment marks (0–100)
  - Assignment marks (0–100)
  - Previous semester marks (0–100)
- **"Autofill from My Records"** button: Pulls the student's live average attendance and internal marks automatically into the form.
- Real-time prediction via Flask backend using the trained **Random Forest Regressor**.
- **Performance Categories**:
  - **85 – 100:** *Excellent* (Green)
  - **70 – 84:** *Good* (Blue)
  - **50 – 69:** *Average* (Yellow)
  - **Below 50:** *Needs Improvement* (Red)
- Prediction results saved to SQLite database with timestamp history.

### 7. Improvement Recommendations (`/recommendations`)
- Dynamic, customized guidance across 5 areas:
  - **Attendance Strategy:** Specific warning if <65%, recovery plan if 65–74%, or commendation if ≥75%.
  - **Study Habit Optimization:** Tailored guidance based on daily study hours.
  - **Internal Marks Focus:** Revision tips and practice suggestions for low-scoring areas.
  - **Assignment Timeliness:** Quality and submission advice.
  - **Exam Strategy:** Distinction roadmap and past paper problem-solving schedules.
- **Suggested Daily Study Timetable** (morning review, afternoon lab, evening problem solving, and night coursework review).

### 8. Progress Reports (`/reports`)
- Comprehensive overview of all academic dimensions.
- **Prediction Trajectory Line Chart** (Chart.js) showing score evolution over time.
- **Subject Mastery Radar Chart** (Chart.js) illustrating strengths and weaknesses.
- **Download Official PDF Report** button: Generates an institutional PDF report with tables and advisor signature blocks using ReportLab.

---

## 🤖 Machine Learning Model Details

- **Algorithm:** Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
- **Dataset:** `data/student_performance.csv` (1,500 records)
- **Features Used:**
  1. `attendance` (attendance percentage)
  2. `study_hours` (daily self-study hours)
  3. `internal_marks` (scaled internal marks)
  4. `assignment_marks` (assignment score)
  5. `previous_semester_marks` (previous semester percentage)
- **Target Variable:** `final_marks` (final exam score out of 100)
- **Train/Test Split:** 80% Training (1,200 samples), 20% Testing (300 samples)
- **Evaluation Results on Test Set:**
  - **Mean Absolute Error (MAE):** `3.05` marks
  - **Mean Squared Error (MSE):** `13.99`
  - **Root Mean Squared Error (RMSE):** `3.74` marks
  - **R² Score:** `0.880` (88.0% variance explained)
- **Feature Importances:**
  - `internal_marks`: ~63.0%
  - `assignment_marks`: ~18.1%
  - `attendance`: ~8.1%
  - `study_hours`: ~7.8%
  - `previous_semester_marks`: ~3.1%

---

## 🔑 Pre-Seeded Demo Account

For testing, a demo account with realistic sample data is pre-seeded in SQLite:

- **Email:** `rahul.student@college.edu`
- **Student ID:** `CS2026-042`
- **Password:** `password123`

*(You can also click the **"Fill"** button on the Login page to autofill these credentials instantly!)*

---

## 💻 Step-by-Step Instructions to Run Locally

### Step 1: Open Terminal in the Project Directory
```powershell
cd "c:\Users\DANDU RAHUL\OneDrive\Desktop\Documents\Desktop\capston1 project"
```

### Step 2: (Optional) Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: (Optional) Retrain Model or Re-initialize Database
*The model and database are already generated and included, but you can retrain anytime:*
```powershell
python train_model.py
python database.py
```

### Step 5: Start the Flask Application
```powershell
python app.py
```

### Step 6: Open the Application in Your Browser
Navigate to:
```text
http://127.0.0.1:5000/
```

---

## 🧪 Running Automated Tests

To run the complete automated test suite:

```powershell
python test_app.py
```

To run the end-to-end CRUD and ML lifecycle test:

```powershell
python test_full_lifecycle.py
```

