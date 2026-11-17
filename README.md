# AI-Based Student Performance Prediction System with AI Bot Assistant

An end-to-end, modern, and user-friendly web application designed to help college students monitor, predict, and elevate their academic performance using Machine Learning and a powerful **Google Gemini AI Bot Assistant** (ChatGPT-style academic & technical copilot).

---

## 📌 Project Overview

College students often find it challenging to know whether their daily study habits, attendance percentage, and internal marks will result in a good final examination grade before it is too late.

The **AI-Based Student Performance Prediction System** bridges this gap:
- Students log subject-wise attendance and continuous internal assessment marks.
- A trained **Random Forest Regression Machine Learning model** forecasts expected final exam marks (out of 100) and categorizes performance (*Excellent*, *Good*, *Average*, *Needs Improvement*).
- An intelligent recommendation engine provides actionable advice (e.g. minimum consecutive classes to attend to reach 75%, focused study timetable).
- Students can download an official institutional academic performance report as a **PDF**.
- **NEW! AI Bot Assistant (ChatGPT-style Copilot)**: Integrated with Google Gemini Generative AI, React 18, and MongoDB/SQLite persistent storage. Answers **any** general question (programming, technical CS subjects, interview prep, mathematics, reasoning, general knowledge) AND provides personalized guidance grounded in the student's actual attendance records and ML predictions.

---

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| **Frontend** | React 18, HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3.3, Bootstrap Icons, Chart.js 4.4, Marked.js, Highlight.js |
| **Backend** | Python 3.11+, Flask 3.1, Werkzeug Security, Python-Dotenv |
| **Generative AI** | Google Gemini API (`google-generativeai`), Gemini 1.5 Flash / Gemini 2.0 Flash / Gemini 1.5 Pro |
| **Machine Learning** | Scikit-learn (RandomForestRegressor), Pandas, NumPy, Joblib |
| **Databases** | **MongoDB** (PyMongo for chat conversations & history) + **SQLite3** (Relational student coursework data & persistent fallback) |
| **PDF Generation** | ReportLab 5.1 |

---

## 📂 Project Folder Structure

```text
capston1 project/
├── .env                        # Environment configuration (Gemini API key, MongoDB URI)
├── .env.example                # Template for environment configuration
├── app.py                      # Main Flask application entry point, route handlers & Chat API
├── database.py                 # SQLite database initialization & seed utility
├── chat_storage.py             # Persistent chat engine: MongoDB with SQLite fallback
├── gemini_service.py           # Google Gemini AI integration, grounding & prompt engineering
├── train_model.py              # ML pipeline: dataset generation, model training & evaluation
├── recommendation_engine.py    # Personalized improvement suggestions & study timetable logic
├── pdf_generator.py            # Official institutional PDF report generator (ReportLab)
├── requirements.txt            # Python dependencies (includes google-generativeai, pymongo, dotenv)
├── database.sqlite             # SQLite database file
├── README.md                   # Complete documentation & execution guide
├── test_app.py                 # Core application test suite (10 test cases)
├── test_full_lifecycle.py      # End-to-end CRUD and ML verification script
├── test_chatbot.py             # Dedicated AI Bot Assistant test suite (9 test cases)
├── data/
│   └── student_performance.csv # CSV dataset (1,500 records)
├── model/
│   ├── student_model.joblib    # Trained Random Forest Regression model
│   └── model_metrics.json      # Model performance metrics (MAE, MSE, R²)
├── static/
│   ├── css/
│   │   ├── style.css           # Clean white & light-blue college dashboard theme
│   │   └── chatbot.css         # Modern ChatGPT-like interface styling
│   ├── js/
│   │   ├── main.js             # Mobile drawer toggle, autofill & target calculators
│   │   ├── charts.js           # Chart.js renderers (doughnut, bar, line, radar)
│   │   └── chatbot.jsx         # React 18 ChatGPT conversational interface
│   ├── vendor/                 # Local offline assets (React, ReactDOM, Babel, Marked, Highlight.js)
│   └── img/
│       ├── logo.svg            # Modern academic crest logo
│       └── student_hero.svg    # Vector student illustration for landing page
└── templates/
    ├── base.html               # Master layout with responsive sidebar, navbar & AI floating pill
    ├── index.html              # Page 1: Home / Landing page
    ├── login.html              # Page 2: Student Login
    ├── register.html           # Page 2: Student Registration
    ├── dashboard.html          # Page 3: Student Dashboard with AI Copilot Launch Showcase
    ├── attendance.html         # Page 4: Attendance Tracking (75% indicator)
    ├── internals.html          # Page 5: Internal Assessment Marks
    ├── predict.html            # Page 6: AI Performance Prediction
    ├── recommendations.html    # Page 7: Personalized Improvement Suggestions
    ├── reports.html            # Page 8: Progress Reports & PDF Export
    └── chatbot.html            # Page 9: Dedicated ChatGPT-like AI Bot Assistant
```

---

## 🤖 AI Bot Assistant (ChatGPT-Style Copilot)

The AI Bot Assistant is an omni-capable Generative AI conversational assistant accessible at `/chatbot`, featured directly on the dashboard, and reachable from anywhere via the floating `Ask AI Copilot` pill.

### Key Capabilities:
1. **General Purpose Knowledge**:
   - **Programming**: Python, Java, C, C++, JavaScript, TypeScript, SQL, Go, Rust, Data Structures (Trees, Graphs, Queues, Heaps), Algorithms (Dynamic Programming, Sorting, Graph Traversals).
   - **Technical CS Disciplines**: Artificial Intelligence, Machine Learning, Deep Learning, DBMS (SQL, Normalization, ACID transactions, Indexing), Operating Systems (Processes, Threads, Deadlocks, Memory Management, CPU Scheduling), Computer Networks (OSI layers, TCP/UDP, IP routing, HTTP/HTTPS), System Design.
   - **Mathematics & Logical Reasoning**: Calculus, Linear Algebra, Probability, Statistics, Discrete Mathematics, Aptitude Puzzles.
   - **Interview Preparation**: Technical coding problems, CS fundamental screening, HR behavioral questions (STAR technique).
   - **Project Architecture**: Can explain this web system's Random Forest Regressor, dataset features, and Flask/React stack.

2. **Grounding on Live Student Data**:
   - When asked about marks, attendance, or risks, it accesses the authenticated student's **actual** SQLite records.
   - Accurately explains predicted final exam marks using feature importance weights (Internal Marks 63%, Assignments 18%, Attendance 8%, Study Hours 8%).
   - Flags courses below 75% attendance or below 50% internal marks and calculates exact consecutive classes needed to recover.
   - Generates personalized daily and weekly study timetables tailored to the student's actual weak courses.
   - **Never invents or hallucinates student data**.

3. **ChatGPT-Like React.js Interface**:
   - **Sidebar**: Conversation threads, new chat button, storage status pill (MongoDB / SQLite), quick dashboard navigation.
   - **Welcome Hero**: 4 instant prompt cards for quick start.
   - **Interactive Chat Canvas**: AI and user message bubbles with avatars and timestamps.
   - **Markdown & Syntax Highlighting**: Full GitHub-flavored markdown parsing via `marked.js` and code syntax highlighting with dark theme via `highlight.js`.
   - **Copy Actions**: One-click "Copy Code" button on each code block and "Copy Response" button on each AI reply.
   - **Typing Animation**: Animated bouncing gradient dots while Gemini is generating.
   - **Clear & Delete**: Clear messages within a thread or delete conversation threads.
   - **Live Academic Context Modal**: Inspect the exact live academic facts provided to Gemini.
   - **Mobile-Responsive**: Collapsible sidebar drawer for mobile devices.

4. **Persistent Dual-Engine Storage**:
   - **MongoDB**: Automatically connects to local MongoDB or MongoDB Atlas via `MONGO_URI`. Stores chat sessions and message histories in `conversations` and `messages` collections.
   - **SQLite Fallback**: If MongoDB is not running or unreachable, the system automatically uses SQLite tables (`chat_conversations` and `chat_messages`) so the app never crashes and chat histories are always persisted.

---

## ⚙️ Configuration & Environment Setup

### 1. Configure Google Gemini API Key
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Open or create the `.env` file in the project root:
```env
# Google Gemini API Key
GEMINI_API_KEY=AIzaSy...your_actual_key_here

# Preferred Gemini Model (gemini-1.5-flash recommended)
GEMINI_MODEL=gemini-1.5-flash

# MongoDB Connection URI (optional - falls back to SQLite if offline)
MONGO_URI=mongodb://localhost:27017/student_ai_chatbot

# Flask Secret Key
SECRET_KEY=ai-student-perf-pred-secure-key-2026
```
*(If no API key is set, the chatbot gracefully displays a helpful setup prompt without throwing unhandled exceptions).*

### 2. (Optional) MongoDB Setup
- If you have MongoDB installed locally or use MongoDB Atlas:
  - Start MongoDB: `mongod` or start the MongoDB service.
  - Or set your Atlas connection string in `.env`:
    `MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/student_ai_chatbot`
- If MongoDB is not running, the application **automatically falls back to SQLite** with full persistence.

---

## 🚀 Website Pages & Features

### 1. Home / Landing Page (`/`)
- Modern clean hero banner with custom student vector illustrations.
- Overview cards and 3-step explanation of how the system works.

### 2. Student Registration & Login (`/register`, `/login`)
- Clean forms for Name, Student ID (Roll No), Email, Password, Department, and Semester.
- Passwords securely hashed with `werkzeug.security.generate_password_hash`.
- One-click **Demo Login** autofill button.

### 3. Student Dashboard (`/dashboard`)
- High-impact metric cards: Total Attendance, Avg Internal Marks, Predicted Final Marks, Performance Category.
- **NEW: AI Bot Assistant Showcase Card**: Instant prompt shortcuts into Gemini AI.
- Chart.js attendance doughnut and subject internal marks breakdown.

### 4. Attendance Tracking (`/attendance`)
- Color-coded attendance indicators (🟢 Safe ≥75%, 🟡 Warning 65–74%, 🔴 Critical <65%).
- Attendance Recovery Calculator: Computes exact classes needed to reach 75%.
- Quick Log attendance increment buttons (Present / Absent).

### 5. Internal Assessment Marks (`/internals`)
- Record continuous evaluation marks (IA1, IA2, Assignments, Previous Exam).
- Automatic calculation of cumulative totals and scaled percentages.

### 6. AI Performance Prediction (`/predict`)
- Predictive form with "Autofill from My Records" button.
- Real-time prediction using trained Scikit-learn **Random Forest Regressor**.
- Performance classification: Excellent (85–100), Good (70–84), Average (50–69), Needs Improvement (<50).

### 7. Improvement Recommendations (`/recommendations`)
- Dynamic advice across Attendance Strategy, Study Habit Optimization, Internal Marks Focus, and Suggested Daily Study Timetable.

### 8. Progress Reports (`/reports`)
- Score trajectory line chart, Subject mastery radar chart, and official institutional **PDF Report Export** via ReportLab.

### 9. AI Bot Assistant (`/chatbot`)
- Dedicated full-page ChatGPT-like workspace powered by React 18 and Google Gemini.
- Multi-turn conversation retention, code highlighting, copy buttons, and student academic grounding.

---

## 🤖 Machine Learning Model Details

- **Algorithm:** Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`, 150 trees)
- **Dataset:** `data/student_performance.csv` (1,500 records)
- **Features Used:**
  1. `attendance` (attendance percentage)
  2. `study_hours` (daily self-study hours)
  3. `internal_marks` (scaled internal marks)
  4. `assignment_marks` (assignment score)
  5. `previous_semester_marks` (previous semester percentage)
- **Target Variable:** `final_marks` (final exam score out of 100)
- **Evaluation Results on Test Set:**
  - **MAE:** `3.05` marks
  - **MSE:** `13.99`
  - **RMSE:** `3.74` marks
  - **R² Score:** `0.880` (88.0% variance explained)
- **Feature Importances:**
  - `internal_marks`: ~63.0%
  - `assignment_marks`: ~18.1%
  - `attendance`: ~8.1%
  - `study_hours`: ~7.8%
  - `previous_semester_marks`: ~3.1%

---

## 🔑 Pre-Seeded Demo Account

- **Email:** `rahul.student@college.edu`
- **Student ID:** `CS2026-042`
- **Password:** `password123`
*(Or click the "Fill" button on the Login page)*

---

## 💻 Step-by-Step Instructions to Run Locally

### Step 1: Open Terminal in the Project Directory
```powershell
cd "c:\Users\DANDU RAHUL\OneDrive\Desktop\Documents\Desktop\capston1\capston1 project"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Configure Environment
Copy `.env.example` to `.env` and insert your Gemini API key:
```powershell
copy .env.example .env
```
Edit `.env` and set:
```text
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here
```

### Step 4: Initialize Database & Seed Records
```powershell
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
Log in and click **AI Bot Assistant** in the sidebar or top bar!

---

## 🧪 Running Automated Tests

To run the AI Bot Assistant test suite:
```powershell
python test_chatbot.py
```

To run the core application route test suite:
```powershell
python test_app.py
```

To run the end-to-end CRUD and ML lifecycle test:
```powershell
python test_full_lifecycle.py
```
