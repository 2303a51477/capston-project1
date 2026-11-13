"""
recommendation_engine.py
Personalized academic improvement recommendation generator based on student metrics.
Translates statistical predictive analytics into plain, empowering advice for college students.
"""

import math


def generate_recommendations(attendance_pct, study_hours, internal_avg, assignment_avg, predicted_marks=None, category=None):
    """
    Analyzes student metrics and generates an array of structured, actionable recommendation cards.
    Each item has:
      - title: Short bold title
      - category: 'Attendance' | 'Study Habits' | 'Internal Marks' | 'Assignments' | 'Exam Strategy'
      - level: 'critical' | 'warning' | 'info' | 'success'
      - icon: Bootstrap icon name
      - advice: Clear, student-friendly explanation
      - action_plan: Concrete steps to execute
    """
    recs = []

    # 1. ATTENDANCE ANALYSIS
    if attendance_pct < 65.0:
        recs.append({
            'title': 'Critical Attendance Alert (< 65%)',
            'category': 'Attendance',
            'level': 'danger',
            'icon': 'bi-exclamation-triangle-fill',
            'advice': f"Your overall attendance is currently {attendance_pct:.1f}%, which is below the mandatory 75% college examination requirement.",
            'action_plan': "Attend all remaining lectures without missing any classes. Meet your faculty advisor immediately to explore approved remedial or attendance makeup options."
        })
    elif attendance_pct < 75.0:
        recs.append({
            'title': 'Attendance Improvement Needed (65% – 74%)',
            'category': 'Attendance',
            'level': 'warning',
            'icon': 'bi-clock-history',
            'advice': f"Your attendance is at {attendance_pct:.1f}%. You are close to the 75% threshold, but an unexpected absence could place you in the condonation or debarred list.",
            'action_plan': "Prioritize attending the next 8–10 consecutive classes in every low-attendance subject to comfortably enter the green zone (75%+)."
        })
    else:
        recs.append({
            'title': 'Excellent Attendance Record (≥ 75%)',
            'category': 'Attendance',
            'level': 'success',
            'icon': 'bi-check-circle-fill',
            'advice': f"Your attendance is healthy at {attendance_pct:.1f}%. Maintaining regular class presence gives you a significant advantage in grasping core exam topics.",
            'action_plan': "Continue attending regularly, participate actively in doubts sessions, and maintain your buffer above 80%."
        })

    # 2. STUDY HOURS & TIME MANAGEMENT
    if study_hours < 2.5:
        recs.append({
            'title': 'Increase Daily Focused Study Time',
            'category': 'Study Habits',
            'level': 'warning',
            'icon': 'bi-hourglass-split',
            'advice': f"You currently study approximately {study_hours:.1f} hours per day. College syllabi generally require at least 3 to 4 hours of self-study outside of classroom lectures.",
            'action_plan': "Adopt a daily study routine: 2 focused sessions of 50 minutes each in the evening, plus 40 minutes for revision of that day's lecture notes."
        })
    elif study_hours <= 4.5:
        recs.append({
            'title': 'Optimize Study Routine with Active Recall',
            'category': 'Study Habits',
            'level': 'info',
            'icon': 'bi-lightning-charge-fill',
            'advice': f"You maintain a steady {study_hours:.1f} hours/day study habit. You can significantly boost retention without adding extra hours by switching from passive reading to active problem solving.",
            'action_plan': "Use the Pomodoro technique (25 min study / 5 min break) and test yourself with flashcards and closed-book concept summaries."
        })
    else:
        recs.append({
            'title': 'High Study Commitment — Avoid Burnout',
            'category': 'Study Habits',
            'level': 'success',
            'icon': 'bi-stars',
            'advice': f"You dedicate {study_hours:.1f} hours/day to self-study, which is commendable. Make sure your study time is balanced with rest and quality sleep.",
            'action_plan': "Ensure 7+ hours of sleep, take regular screen breaks, and focus on high-weightage chapters and past university question patterns."
        })

    # 3. INTERNAL ASSESSMENT PERFORMANCE
    if internal_avg < 50.0:
        recs.append({
            'title': 'Targeted Revision for Internal Assessments',
            'category': 'Internal Marks',
            'level': 'danger',
            'icon': 'bi-journal-x',
            'advice': f"Your average internal score is {internal_avg:.1f}%. Internal marks constitute up to 30-40% of your total semester grade and heavily influence final exam readiness.",
            'action_plan': "Solve textbook end-of-chapter problems, clarify doubts with professors during office hours, and review solved question papers from the last 3 years."
        })
    elif internal_avg < 70.0:
        recs.append({
            'title': 'Elevate Internal Marks to 75%+',
            'category': 'Internal Marks',
            'level': 'warning',
            'icon': 'bi-graph-up-arrow',
            'advice': f"Your internal score stands at {internal_avg:.1f}%. With structured preparation on numerical problems and structured answers, you can cross 75%.",
            'action_plan': "Practice writing neat step-by-step solutions with labeled diagrams. Review test papers to identify topics where you lost marks."
        })
    else:
        recs.append({
            'title': 'Strong Internal Assessment Performance',
            'category': 'Internal Marks',
            'level': 'success',
            'icon': 'bi-award-fill',
            'advice': f"Superb internal performance ({internal_avg:.1f}%)! You have built a solid foundation across all continuous evaluation components.",
            'action_plan': "Form peer study groups to teach challenging concepts and test your knowledge against tough case studies or university distinction problems."
        })

    # 4. ASSIGNMENT COMPLETION & QUALITY
    if assignment_avg < 60.0:
        recs.append({
            'title': 'Submit Assignments On Time with Proper Structure',
            'category': 'Assignments',
            'level': 'warning',
            'icon': 'bi-file-earmark-text-fill',
            'advice': f"Your assignment average is {assignment_avg:.1f}%. Assignments are guaranteed marks that boost your cumulative internal score if completed thoroughly.",
            'action_plan': "Start assignments 3 days before the deadline. Adhere strictly to the grading rubric, format codes cleanly, and cite diagrams properly."
        })
    else:
        recs.append({
            'title': 'Reliable Assignment Consistency',
            'category': 'Assignments',
            'level': 'info',
            'icon': 'bi-check2-all',
            'advice': f"Your assignment score is solid ({assignment_avg:.1f}%). High assignment quality reflects discipline and deep conceptual comprehension.",
            'action_plan': "Keep up timely submissions and use assignments as self-testing tools for the final university examinations."
        })

    # 5. OVERALL PREDICTED PERFORMANCE ROADMAP
    if predicted_marks is not None:
        if predicted_marks >= 85.0:
            recs.append({
                'title': 'Honors & Distinction Trajectory',
                'category': 'Exam Strategy',
                'level': 'success',
                'icon': 'bi-trophy-fill',
                'advice': f"The AI model predicts a final score of {predicted_marks:.1f}% (Excellent category). You are on track for academic distinction!",
                'action_plan': "Maintain consistency across all subjects, practice time-boxed mock exam papers, and explore research papers or competitive certifications."
            })
        elif predicted_marks >= 70.0:
            recs.append({
                'title': 'Solid Standing — Push for Distinction (85%+)',
                'category': 'Exam Strategy',
                'level': 'info',
                'icon': 'bi-speedometer2',
                'advice': f"Predicted score: {predicted_marks:.1f}% (Good category). You are performing well, and small optimizations can boost you to the top tier.",
                'action_plan': "Identify your lowest-scoring subject, increase daily study time by 30-45 minutes for that subject, and master recurring 10-mark questions."
            })
        elif predicted_marks >= 50.0:
            recs.append({
                'title': 'Targeted Push to Reach 70% First Class',
                'category': 'Exam Strategy',
                'level': 'warning',
                'icon': 'bi-arrow-up-circle-fill',
                'advice': f"Predicted score: {predicted_marks:.1f}% (Average category). While passing is secure, your score can be improved significantly before final exams.",
                'action_plan': "Focus 60% of your revision time on high-weightage modules. Create 1-page summary formula sheets and solve past 5 years' question papers."
            })
        else:
            recs.append({
                'title': 'Immediate Academic Intervention Needed',
                'category': 'Exam Strategy',
                'level': 'danger',
                'icon': 'bi-shield-fill-exclamation',
                'advice': f"Predicted score: {predicted_marks:.1f}% (Needs Improvement). There is a risk of scoring below passing threshold if immediate changes are not made.",
                'action_plan': "Implement a 30-day turnaround plan: reach 80% attendance, study 4 hours daily, complete every pending assignment, and seek personal tutoring."
            })

    return recs

