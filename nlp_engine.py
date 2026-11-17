"""
nlp_engine.py
Natural Language Processing (NLP) Engine for Question Understanding & Intent Identification.

Features:
- Multi-class Intent Classification using linguistic rules, semantic keywords, and regex.
- Named Entity Recognition (NER) for courses, academic metrics, and programming languages.
- Question Type Identification (factual, analytical, prescriptive, conceptual, procedural).
- Sentiment and Academic Risk signal detection.
"""

from typing import Dict, Any, List, Optional
import re


class IntentType:
    STUDENT_ATTENDANCE = "STUDENT_ATTENDANCE"
    STUDENT_INTERNALS = "STUDENT_INTERNALS"
    STUDENT_ASSIGNMENTS = "STUDENT_ASSIGNMENTS"
    STUDENT_PREDICTION = "STUDENT_PREDICTION"
    STUDENT_WEAK_STRONG = "STUDENT_WEAK_STRONG"
    STUDENT_PROFILE = "STUDENT_PROFILE"
    STUDENT_STUDY_PLAN = "STUDENT_STUDY_PLAN"
    PROGRAMMING_DSA = "PROGRAMMING_DSA"
    COURSE_CONCEPT = "COURSE_CONCEPT"
    CAREER_GUIDANCE = "CAREER_GUIDANCE"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"


class NLPEngine:
    """Processes user natural language queries and extracts structural metadata."""

    COURSE_SYNONYMS = {
        'ccn': 'Computer Communication Networks',
        'network': 'Computer Communication Networks',
        'networks': 'Computer Communication Networks',
        'networking': 'Computer Communication Networks',
        'cs604': 'Computer Communication Networks',
        'dbms': 'Database Management Systems',
        'database': 'Database Management Systems',
        'databases': 'Database Management Systems',
        'sql': 'Database Management Systems',
        'cs603': 'Database Management Systems',
        'os': 'Operating Systems & Virtualization',
        'operating system': 'Operating Systems & Virtualization',
        'operating systems': 'Operating Systems & Virtualization',
        'cs602': 'Operating Systems & Virtualization',
        'algorithms': 'Design & Analysis of Algorithms',
        'algorithm': 'Design & Analysis of Algorithms',
        'algo': 'Design & Analysis of Algorithms',
        'dsa': 'Design & Analysis of Algorithms',
        'cs601': 'Design & Analysis of Algorithms',
        'ml': 'Machine Learning & AI',
        'machine learning': 'Machine Learning & AI',
        'ai': 'Machine Learning & AI',
        'cs605': 'Machine Learning & AI',
        'cloud': 'Cloud Computing & DevOps',
        'devops': 'Cloud Computing & DevOps',
        'cs699': 'Cloud Computing & DevOps',
        'python': 'Python Programming',
        'java': 'Java Programming',
        'c++': 'C++ Programming'
    }

    INTENT_PATTERNS = [
        (
            IntentType.STUDENT_PROFILE,
            [
                r"\b(about myself|about my self|about me|who am i|my profile|my details|total about him|about him|tell me about him|my record|my data|student info|everything about me|student dossier)\b"
            ]
        ),
        (
            IntentType.STUDENT_ATTENDANCE,
            [
                r"\b(attendance|attended|total classes|absent|present|classes attended|shortage|bunk|75%|condonation)\b"
            ]
        ),
        (
            IntentType.STUDENT_INTERNALS,
            [
                r"\b(internal|internals|ia1|ia2|ia-1|ia-2|midterm|mid terms|internal marks|continuous assessment)\b"
            ]
        ),
        (
            IntentType.STUDENT_ASSIGNMENTS,
            [
                r"\b(assignment|assignments|project marks|lab marks|practical marks)\b"
            ]
        ),
        (
            IntentType.STUDENT_WEAK_STRONG,
            [
                r"\b(weak subject|weak subjects|weak area|weak areas|strong subject|strong subjects|strengths|weaknesses|failing|struggling|where am i weak|where do i need to improve)\b"
            ]
        ),
        (
            IntentType.STUDENT_STUDY_PLAN,
            [
                r"\b(study plan|timetable|time table|schedule|routine|daily routine|revision plan|how to study|study hours)\b"
            ]
        ),
        (
            IntentType.STUDENT_PREDICTION,
            [
                r"\b(predicted|prediction|forecast|expected mark|expected marks|final exam score|predicted marks|will i pass|distinction score)\b"
            ]
        ),
        (
            IntentType.PROGRAMMING_DSA,
            [
                r"\b(quicksort|quick sort|mergesort|merge sort|binary search|linear search|bubble sort|insertion sort|linked list|doubly linked list|binary tree|bst|binary search tree|stack|queue|heap|graph|bfs|dfs|dijkstra|dynamic programming|dp|recursion|time complexity|space complexity|big o|leetcode|write code|implement in)\b"
            ]
        ),
        (
            IntentType.CAREER_GUIDANCE,
            [
                r"\b(career|carrer|job|jobs|placement|placements|internship|roles|software engineer|sde|web developer|data scientist|ai engineer|resume|interview preparation)\b"
            ]
        ),
        (
            IntentType.COURSE_CONCEPT,
            [
                r"\b(deadlock|coffman|process scheduling|virtual memory|paging|thrashing|semaphore|mutex|acid|normalization|1nf|2nf|3nf|bcnf|indexing|b-tree|b\+ tree|osi layer|osi layers|tcp|udp|three way handshake|ip address|subnetting|routing|random forest|decision tree|overfitting|underfitting)\b"
            ]
        )
    ]

    @classmethod
    def analyze_query(cls, user_message: str) -> Dict[str, Any]:
        """
        Parses user message and extracts linguistic features, intents, and entities.
        """
        text = user_message.strip()
        lower_text = text.lower()

        # 1. Identify primary and secondary intents
        primary_intent = IntentType.GENERAL_KNOWLEDGE
        matched_intents = []

        for intent, patterns in cls.INTENT_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, lower_text):
                    matched_intents.append(intent)
                    break

        if matched_intents:
            primary_intent = matched_intents[0]

        # 2. Extract course entities
        detected_courses = []
        for key, full_name in cls.COURSE_SYNONYMS.items():
            if re.search(rf"\b{re.escape(key)}\b", lower_text):
                if full_name not in detected_courses:
                    detected_courses.append(full_name)

        # 3. Detect question category
        if any(w in lower_text for w in ["how to", "steps to", "how do i", "how can i"]):
            q_type = "procedural"
        elif any(w in lower_text for w in ["why", "explain", "what is", "define", "concept"]):
            q_type = "conceptual"
        elif any(w in lower_text for w in ["what are my", "how many", "show my", "what is my", "tell me my"]):
            q_type = "analytical_personal"
        elif any(w in lower_text for w in ["write", "code", "implement", "program"]):
            q_type = "coding"
        else:
            q_type = "general"

        # 4. Check whether query targets authenticated student data
        is_student_query = primary_intent in [
            IntentType.STUDENT_PROFILE,
            IntentType.STUDENT_ATTENDANCE,
            IntentType.STUDENT_INTERNALS,
            IntentType.STUDENT_ASSIGNMENTS,
            IntentType.STUDENT_WEAK_STRONG,
            IntentType.STUDENT_STUDY_PLAN,
            IntentType.STUDENT_PREDICTION
        ] or bool(re.search(r"\b(my|i|myself|me|am i)\b", lower_text))

        return {
            "original_query": text,
            "primary_intent": primary_intent,
            "all_intents": matched_intents,
            "detected_courses": detected_courses,
            "question_type": q_type,
            "is_student_query": is_student_query
        }
