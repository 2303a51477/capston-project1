"""
conversation_memory.py
Contextual Multi-Turn Conversation Memory Engine for AI Study Copilot.

Features:
- Maintains chronological turn history per conversation thread.
- Tracks active topic and entity states (e.g., current subject, metric).
- Performs pronoun and anaphora resolution for follow-up questions.
- Formats sliding-window history for LLM and RAG prompt injection.
"""

from typing import List, Dict, Any, Optional
import re


class ConversationMemory:
    """Manages conversational state and contextual memory across turns."""

    KNOWN_SUBJECTS = {
        'ccn': 'Computer Communication Networks',
        'networks': 'Computer Communication Networks',
        'networking': 'Computer Communication Networks',
        'cs604': 'Computer Communication Networks',
        'dbms': 'Database Management Systems',
        'database': 'Database Management Systems',
        'sql': 'Database Management Systems',
        'cs603': 'Database Management Systems',
        'os': 'Operating Systems & Virtualization',
        'operating system': 'Operating Systems & Virtualization',
        'cs602': 'Operating Systems & Virtualization',
        'algorithms': 'Design & Analysis of Algorithms',
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

    KNOWN_METRICS = {
        'attendance': 'attendance',
        'classes': 'attendance',
        'absent': 'attendance',
        'present': 'attendance',
        'shortage': 'attendance',
        'internal': 'internal_marks',
        'ia1': 'internal_marks',
        'ia2': 'internal_marks',
        'midterm': 'internal_marks',
        'assignment': 'assignment_marks',
        'assignments': 'assignment_marks',
        'predicted': 'prediction',
        'prediction': 'prediction',
        'forecast': 'prediction',
        'marks': 'marks',
        'study hours': 'study_hours',
        'weak': 'weak_areas',
        'strong': 'strong_areas',
        'plan': 'study_plan',
        'timetable': 'study_plan'
    }

    @classmethod
    def resolve_contextual_query(
        cls,
        current_query: str,
        past_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyzes the current query in light of previous conversational turns:
        - Resolves pronouns ('it', 'that subject', 'this course', 'my marks in that')
        - Identifies active subject and metric entities
        - Constructs an expanded search query for RAG semantic search.
        """
        clean_q = current_query.strip().lower()
        active_subject = None
        active_metric = None

        # 1. Check if current query explicitly specifies subject or metric
        for key, subj in cls.KNOWN_SUBJECTS.items():
            if re.search(rf"\b{re.escape(key)}\b", clean_q):
                active_subject = subj
                break

        for key, met in cls.KNOWN_METRICS.items():
            if re.search(rf"\b{re.escape(key)}\b", clean_q):
                active_metric = met
                break

        # 2. If explicit pronouns or elliptical follow-ups detected, look back in past messages
        pronoun_triggers = [
            r"\bit\b", r"\bthat\b", r"\bthis\b", r"\bthem\b", r"\bthose\b",
            r"\bthat subject\b", r"\bthat course\b", r"\bin that\b", r"\bfor that\b",
            r"\bto fix it\b", r"\babout it\b", r"\bimprove it\b", r"\bhow to fix\b"
        ]
        has_pronoun = any(re.search(p, clean_q) for p in pronoun_triggers)

        if has_pronoun and past_messages:
            # Look back through the last 4 messages in reverse
            for msg in reversed(past_messages[-4:]):
                content = msg.get("content", "").lower()
                if not active_subject:
                    for key, subj in cls.KNOWN_SUBJECTS.items():
                        if re.search(rf"\b{re.escape(key)}\b", content):
                            active_subject = subj
                            break
                if not active_metric:
                    for key, met in cls.KNOWN_METRICS.items():
                        if re.search(rf"\b{re.escape(key)}\b", content):
                            active_metric = met
                            break
                if active_subject and active_metric:
                    break

        # 3. Build expanded query for semantic retrieval
        expanded_query = current_query
        if active_subject and active_subject.lower() not in clean_q:
            expanded_query += f" {active_subject}"
        if active_metric and active_metric.lower() not in clean_q:
            expanded_query += f" {active_metric}"

        return {
            "original_query": current_query,
            "expanded_query": expanded_query,
            "active_subject": active_subject,
            "active_metric": active_metric,
            "is_followup": bool(has_pronoun)
        }

    @classmethod
    def get_sliding_window_context(
        cls,
        past_messages: List[Dict[str, Any]],
        max_turns: int = 6
    ) -> List[Dict[str, str]]:
        """Extracts the recent conversational history in standardized format."""
        window = []
        recent = past_messages[-max_turns:] if len(past_messages) > max_turns else past_messages
        for msg in recent:
            role = "user" if msg.get("role") == "user" else "assistant"
            content = msg.get("content", "").strip()
            if content:
                window.append({"role": role, "content": content})
        return window
