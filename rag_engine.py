"""
rag_engine.py
Retrieval-Augmented Generation (RAG) & Semantic Search Engine.

Features:
- Educational Knowledge Base: Core curriculum concepts across Computer Science courses
  (Algorithms, OS, DBMS, Networks, ML/AI, Cloud/DevOps, and College Academic Regulations).
- Dynamic Student Academic Grounding: Real-time chunking of verified database records.
- Semantic Vector Search: TF-IDF vectorization and Cosine Similarity ranking via Scikit-Learn.
- Top-K Context Retrieval for LLM augmentation and local zero-hallucination synthesis.
- Honest handling of out-of-domain / unknown queries.
"""

from typing import List, Dict, Any, Optional
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================================
# STATIC EDUCATIONAL KNOWLEDGE BASE (CURRICULUM & REGULATIONS)
# =====================================================================

EDUCATIONAL_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    # 1. ALGORITHMS & DSA
    {
        "id": "algo_quicksort",
        "category": "Algorithms",
        "course_code": "CS601",
        "title": "QuickSort Algorithm: Divide-and-Conquer Sorting",
        "content": (
            "QuickSort is a Divide-and-Conquer algorithm. It selects a pivot element and partitions the array "
            "so elements smaller than the pivot go to the left and larger elements go to the right. "
            "Time Complexity: Best Case O(n log n), Average Case O(n log n), Worst Case O(n^2) when pivot is poorly chosen. "
            "Space Complexity: O(log n) auxiliary stack space for recursion. "
            "Python implementation: def quicksort(arr): if len(arr)<=1: return arr; pivot=arr[len(arr)//2]; "
            "return quicksort([x for x in arr if x<pivot]) + [x for x in arr if x==pivot] + quicksort([x for x in arr if x>pivot]). "
            "Java uses in-place Lomuto or Hoare partitioning."
        )
    },
    {
        "id": "algo_mergesort",
        "category": "Algorithms",
        "course_code": "CS601",
        "title": "MergeSort Algorithm: Stable Divide-and-Conquer Sorting",
        "content": (
            "MergeSort divides an array into two halves recursively until arrays of size 1 are obtained, "
            "then merges the sorted halves. "
            "Time Complexity: Guaranteed O(n log n) in all cases (Best, Average, Worst). "
            "Space Complexity: O(n) auxiliary memory for temporary merge buffers. "
            "Stable sorting algorithm, ideal for linked lists and external sorting where disk access dominates."
        )
    },
    {
        "id": "algo_binary_search",
        "category": "Algorithms",
        "course_code": "CS601",
        "title": "Binary Search Algorithm on Sorted Collections",
        "content": (
            "Binary Search finds a target value within a sorted array by repeatedly dividing the search space in half. "
            "Time Complexity: O(log n). Space Complexity: O(1) iterative, O(log n) recursive call stack. "
            "Midpoint calculation: mid = low + (high - low) // 2 prevents integer overflow. "
            "Prerequisite: Array must be sorted in ascending or descending order prior to search."
        )
    },
    {
        "id": "algo_linked_list",
        "category": "Data Structures",
        "course_code": "CS601",
        "title": "Linked Lists: Singly and Doubly Linked Structures",
        "content": (
            "A Linked List is a linear data structure where elements (nodes) contain data and pointer(s) to next/previous nodes. "
            "Insertion at head: O(1). Insertion at tail: O(n) or O(1) with tail pointer. Search/Access: O(n). Deletion: O(1) given node pointer. "
            "Reversal algorithm: Initialize prev = None, current = head; while current: next_node = current.next, "
            "current.next = prev, prev = current, current = next_node; return prev."
        )
    },
    {
        "id": "algo_bst",
        "category": "Data Structures",
        "course_code": "CS601",
        "title": "Binary Search Tree (BST) Properties and Operations",
        "content": (
            "In a Binary Search Tree (BST), for every node: all values in the left subtree are smaller, "
            "and all values in the right subtree are greater. "
            "Search, Insert, Delete: Average O(log n), Worst O(n) when tree degenerates into a linear linked list. "
            "In-Order Traversal (Left, Root, Right) always produces nodes in strictly sorted ascending order."
        )
    },
    {
        "id": "algo_stack_queue",
        "category": "Data Structures",
        "course_code": "CS601",
        "title": "Stack (LIFO) and Queue (FIFO) Data Structures",
        "content": (
            "Stack operates on Last-In, First-Out (LIFO) principle. Key operations: push(x), pop(), peek(), isEmpty() all in O(1) time. "
            "Applications: function call stack, undo operations, parenthesis matching, depth-first search (DFS). "
            "Queue operates on First-In, First-Out (FIFO) principle. Key operations: enqueue(x), dequeue(), peek() all in O(1) time. "
            "Applications: CPU job scheduling, printer queues, breadth-first search (BFS)."
        )
    },

    # 2. OPERATING SYSTEMS
    {
        "id": "os_deadlocks",
        "category": "Operating Systems",
        "course_code": "CS602",
        "title": "Deadlocks and Coffman Conditions in Operating Systems",
        "content": (
            "A Deadlock is a permanent blocking condition where processes hold resources and wait for resources held by each other. "
            "Four necessary Coffman conditions: "
            "1. Mutual Exclusion (resources non-shareable), "
            "2. Hold and Wait (process holds resource while requesting another), "
            "3. No Preemption (resources cannot be forcibly taken), "
            "4. Circular Wait (closed cycle of waiting processes P0->P1->...->P0). "
            "Handling: Prevention (invalidate one condition), Avoidance (Banker's Algorithm), Detection and Recovery (resource preemption)."
        )
    },
    {
        "id": "os_virtual_memory",
        "category": "Operating Systems",
        "course_code": "CS602",
        "title": "Virtual Memory, Paging, and Page Replacement",
        "content": (
            "Virtual memory separates logical address space from physical memory using Paging and Translation Lookaside Buffers (TLBs). "
            "Page Fault occurs when a referenced page is not in physical RAM, requiring disk swap. "
            "Page replacement algorithms: FIFO, LRU (Least Recently Used), Optimal (Belady's). "
            "Thrashing occurs when high paging activity causes CPU utilization to collapse."
        )
    },
    {
        "id": "os_process_scheduling",
        "category": "Operating Systems",
        "course_code": "CS602",
        "title": "CPU Scheduling Algorithms: FCFS, SJF, Round Robin, Priority",
        "content": (
            "CPU Scheduling determines which process in ready queue gets CPU execution. "
            "FCFS (First-Come, First-Served): non-preemptive, suffers from convoy effect. "
            "SJF (Shortest Job First): optimal average waiting time, but risks starvation of long jobs. "
            "Round Robin: preemptive with fixed time quantum Q, fair and interactive. "
            "Priority Scheduling: executes highest priority first; aging prevents starvation."
        )
    },

    # 3. DATABASE MANAGEMENT SYSTEMS
    {
        "id": "dbms_acid",
        "category": "DBMS",
        "course_code": "CS603",
        "title": "ACID Properties of Database Transactions",
        "content": (
            "A database transaction is a logical unit of work that must satisfy ACID properties: "
            "Atomicity: 'All or nothing' execution; rolls back on failure. "
            "Consistency: Preserves all integrity constraints and database invariants. "
            "Isolation: Concurrent execution yields identical state as serial execution (prevents dirty reads, phantom reads). "
            "Durability: Committed transactions permanently survive system crashes via write-ahead logging (WAL)."
        )
    },
    {
        "id": "dbms_indexing_normalization",
        "category": "DBMS",
        "course_code": "CS603",
        "title": "Database Normalization and B-Tree Indexing",
        "content": (
            "Normalization reduces data redundancy and update anomalies: 1NF (atomic attributes), "
            "2NF (no partial functional dependencies), 3NF (no transitive dependencies), BCNF (every determinant is superkey). "
            "Indexing: B-Trees and B+ Trees provide O(log n) search, insertion, and range queries. "
            "B+ Trees store all actual record pointers in leaf nodes linked sequentially, optimizing range scans (BETWEEN, >, <)."
        )
    },

    # 4. COMPUTER NETWORKS
    {
        "id": "networks_osi_tcp",
        "category": "Computer Networks",
        "course_code": "CS604",
        "title": "OSI 7-Layer Architecture and TCP vs UDP",
        "content": (
            "OSI 7 Layers: 7. Application (HTTP, DNS), 6. Presentation (SSL/TLS), 5. Session, "
            "4. Transport (TCP, UDP), 3. Network (IP, ICMP, Routing), 2. Data Link (Ethernet, MAC, ARP), 1. Physical (Bits). "
            "TCP vs UDP: TCP is connection-oriented (3-way handshake SYN, SYN-ACK, ACK), reliable (retransmissions), "
            "in-order delivery with flow/congestion control. UDP is connectionless, lightweight (8-byte header), "
            "best-effort delivery suited for VoIP, video streaming, and DNS."
        )
    },

    # 5. MACHINE LEARNING & AI
    {
        "id": "ml_random_forest",
        "category": "Machine Learning",
        "course_code": "CS605",
        "title": "Random Forest Regressor & Performance Prediction Architecture",
        "content": (
            "Random Forest is an ensemble learning method constructing multiple decision trees (150 trees in this system) "
            "using bootstrap aggregating (bagging) and feature subsampling. "
            "System Performance Metrics: R² Score = 0.880 (explains 88% variance), Mean Absolute Error (MAE) = 3.05 marks, "
            "Root Mean Squared Error (RMSE) = 3.74 marks on 1,500 historical student records. "
            "Feature Importance: Continuous Internal Assessment (63.0%), Assignments (18.1%), Attendance (8.1%), "
            "Daily Study Hours (7.8%), Previous Semester Marks (3.1%)."
        )
    },

    # 6. COLLEGE ACADEMIC REGULATIONS
    {
        "id": "regulations_attendance_policy",
        "category": "College Regulations",
        "course_code": "ACADEMIC_RULES",
        "title": "Mandatory 75% Attendance Regulation & Shortage Condonation",
        "content": (
            "College Academic Regulation Rule 4.2: Every student must maintain a minimum of 75.0% attendance in each registered course. "
            "Attendance between 65.0% and 74.9% constitutes 'Warning Shortage' and requires formal condonation on valid medical grounds. "
            "Attendance below 65.0% constitutes 'Critical Shortage' resulting in exam debarment. "
            "Recovery Formula: Minimum consecutive classes needed = ceil((0.75 * Total_Classes - Attended) / 0.25)."
        )
    },

    # 7. CAREER PATHWAYS
    {
        "id": "career_sde_ai",
        "category": "Career Guidance",
        "course_code": "CAREER",
        "title": "Software Engineering (SDE) and AI Engineering Career Tracks",
        "content": (
            "Software Development Engineer (SDE) Track: Focus on Data Structures & Algorithms (CS601), DBMS (CS603), "
            "and Operating Systems (CS602). Core skills: Java/Python, System Design, REST APIs, Git. "
            "AI/ML Engineer Track: Focus on Machine Learning & AI (CS605), Algorithms (CS601). Core skills: Scikit-learn, "
            "PyTorch/TensorFlow, Pandas, Statistics, LLMs, Predictive Modeling. "
            "Cloud/DevOps Track: Focus on CCN (CS604), OS (CS602), Cloud (CS699). Skills: Linux, Docker, Kubernetes, AWS/GCP."
        )
    }
]


class RAGEngine:
    """Orchestrates Semantic Search, Context Chunking, and Knowledge Retrieval."""

    @classmethod
    def chunk_student_profile(cls, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Converts live student database records into granular, searchable RAG chunks."""
        chunks = []
        s = profile['student']
        att = profile['attendance']
        sh = profile['study_hours']
        ints = profile['internals']
        preds = profile['predictions']

        # Chunk 1: Overall Profile & Identity
        chunks.append({
            "id": f"student_{s['id']}_profile",
            "category": "Student Profile",
            "course_code": "STUDENT",
            "title": f"Student Profile: {s['name']} ({s['student_id']})",
            "content": (
                f"Student Name: {s['name']}. Roll Number ID: {s['student_id']}. "
                f"Department: {s['department']}, Semester: {s['semester']}, Email: {s['email']}. "
                f"Overall Attendance: {att['overall_pct']}% ({att['total_attended']}/{att['total_classes']} classes). "
                f"Average Internal Assessment: {ints['avg_internal_pct']}%. "
                f"Daily Self-Study Hours: {sh['current_daily_hours']} hours/day. "
                f"Predicted Final Exam Marks: {preds['predicted_marks']}/100, Performance Tier: {preds['category']}, "
                f"Academic Risk Level: {preds['risk_level']}."
            )
        })

        # Chunk 2: Attendance by Subject
        for c in att['courses']:
            rec_note = f"Needs {c['classes_needed']} consecutive classes to reach 75%." if c['classes_needed'] > 0 else "Safe attendance."
            chunks.append({
                "id": f"student_{s['id']}_att_{c['code']}",
                "category": "Student Attendance",
                "course_code": c['code'],
                "title": f"Attendance in {c['name']} ({c['code']})",
                "content": (
                    f"Student: {s['name']}. Course: {c['name']} ({c['code']}). "
                    f"Attended {c['attended']} out of {c['total']} classes ({c['pct']}%). "
                    f"Attendance Status: {c['status']}. {rec_note}"
                )
            })

        # Chunk 3: Course Internal Marks & Assignments
        for c in profile['courses']:
            chunks.append({
                "id": f"student_{s['id']}_marks_{c['code']}",
                "category": "Student Internal Marks",
                "course_code": c['code'],
                "title": f"Internal Marks & Assignments in {c['name']} ({c['code']})",
                "content": (
                    f"Student: {s['name']}. Course: {c['name']} ({c['code']}). "
                    f"Internal Test 1 (IA1): {c['ia1']}/25. Internal Test 2 (IA2): {c['ia2']}/25. "
                    f"Assignment Score: {c['assignment']}/25. Total Internal Marks: {c['total_internal']}/75 "
                    f"({c['scaled_internal']}%). Previous Semester Marks: {c['previous_marks']}/100. "
                    f"Attendance: {c['attendance_pct']}%."
                )
            })

        # Chunk 4: Predictions & Feature Weights
        chunks.append({
            "id": f"student_{s['id']}_prediction",
            "category": "Student Predictions",
            "course_code": "PREDICTIONS",
            "title": f"Machine Learning Prediction & Risk Breakdown for {s['name']}",
            "content": (
                f"Student: {s['name']}. Predicted Final Examination Score: {preds['predicted_marks']}/100. "
                f"Performance Classification: {preds['category']}. Academic Risk Level: {preds['risk_level']}. "
                f"Model: {preds['model_name']}, Accuracy: {preds['model_accuracy']}. "
                f"Feature Importance Weights: Internal Assessment Marks: 63.0%, Assignments: 18.1%, "
                f"Classroom Attendance: 8.1%, Daily Study Hours: 7.8%, Previous Semester: 3.1%."
            )
        })

        # Chunk 5: Weak Subjects & Strong Subjects
        strong_names = ", ".join([c['name'] for c in profile['strong_subjects']]) or "None"
        weak_details = "; ".join([f"{w['course']['name']} ({', '.join(w['reasons'])})" for w in profile['weak_subjects']]) or "None"
        chunks.append({
            "id": f"student_{s['id']}_strengths_weaknesses",
            "category": "Student Weak & Strong Subjects",
            "course_code": "ACADEMIC_ANALYSIS",
            "title": f"Strong and Weak Academic Subjects for {s['name']}",
            "content": (
                f"Student: {s['name']}. Strong Subjects (High Attendance and High Internals): {strong_names}. "
                f"Weak Subjects & Critical Areas Requiring Remediation: {weak_details}."
            )
        })

        return chunks

    @classmethod
    def semantic_search(
        cls,
        query: str,
        student_profile: Optional[Dict[str, Any]] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Executes TF-IDF + Cosine Similarity semantic search across both
        the Educational Knowledge Base and live Student Academic Record chunks.
        """
        corpus = list(EDUCATIONAL_KNOWLEDGE_BASE)
        if student_profile:
            student_chunks = cls.chunk_student_profile(student_profile)
            corpus.extend(student_chunks)

        if not corpus or not query.strip():
            return []

        doc_texts = [f"{d.get('title', '')} {d.get('content', '')}" for d in corpus]

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words='english'
            )
            tfidf_matrix = vectorizer.fit_transform(doc_texts)
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

            # Dynamic relevance adjustment: if query is personal, prioritize student chunks;
            # if query is technical/conceptual, boost educational curriculum chunks
            is_personal = bool(re.search(r"\b(my|i|me|myself|am i|profile)\b", query.lower()))
            adjusted_scores = []
            for i, score in enumerate(similarities):
                is_student_doc = corpus[i].get("category", "").startswith("Student ")
                boosted = float(score)
                if not is_personal and not is_student_doc:
                    boosted *= 1.45
                elif is_personal and is_student_doc:
                    boosted *= 1.30
                adjusted_scores.append((boosted, i))

            adjusted_scores.sort(key=lambda x: x[0], reverse=True)

            results = []
            for boosted_score, idx in adjusted_scores[:top_k]:
                if boosted_score > 0.05:
                    doc = dict(corpus[idx])
                    doc["similarity_score"] = round(boosted_score, 4)
                    results.append(doc)

            return results
        except Exception as e:
            print(f"[RAGEngine] Semantic search error: {e}")
            return []

    @classmethod
    def synthesize_local_response(
        cls,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        nlp_analysis: Dict[str, Any],
        student_profile: Optional[Dict[str, Any]]
    ) -> str:
        """
        Synthesizes an intelligent, structured, truthful answer grounded in the retrieved
        RAG chunks and NLP metadata. Never invents student data or hallucinates facts.
        """
        intent = nlp_analysis.get("primary_intent", "GENERAL_KNOWLEDGE")
        q = query.lower().strip()

        # -------------------------------------------------------------
        # 0. CONVERSATIONAL GREETINGS & CLOSINGS ("HI", "HELLO", "THANKS")
        # -------------------------------------------------------------
        if intent == "CONVERSATIONAL_GREETING":
            if student_profile:
                s = student_profile['student']
                att = student_profile['attendance']
                preds = student_profile['predictions']
                weak_names = ", ".join([w['course']['name'] for w in student_profile['weak_subjects']])
                weak_line = f"- **Attention Needed:** {weak_names}" if weak_names else "- **Academic Status:** All courses currently in safe standing!"

                return f"""### 👋 Hello {s['name']}! Welcome to EduPredict AI Study Copilot

I'm connected directly to your verified academic profile (**{s['student_id']}**, {s['department']}, Semester {s['semester']}) and ready to help!

#### 📊 Your Live Academic Snapshot:
- **Overall Attendance:** **{att['overall_pct']}%** ({att['total_attended']}/{att['total_classes']} classes)
- **Predicted Final Marks:** **{preds['predicted_marks']}/100** (*{preds['category']}*)
{weak_line}

---

#### 💬 How Can I Help You Today?
You can ask me anything like:
- *"What are my weak subjects and how do I improve them?"*
- *"How many classes do I need to attend in CCN to reach 75%?"*
- *"Generate a personalized daily study timetable for me."*
- *"Explain QuickSort, MergeSort, Linked Lists, or Binary Search Trees with code."*
- *"Explain ACID properties in DBMS or Deadlocks in Operating Systems."*
- *"What are the best career options for my subjects?"*"""
            else:
                return """### 👋 Hello! Welcome to EduPredict AI Study Copilot

I am your AI academic mentor for the **AI-Based Student Performance Prediction System**.

Ask me about:
- **Academic Analytics:** Attendance tracking, internal marks, predicted scores, and personalized study plans.
- **Computer Science & DSA:** QuickSort, Binary Search Trees, Linked Lists, DBMS, Operating Systems, Networks, and Machine Learning.
- **Career Guidance:** Software Engineering, AI/ML, and Cloud/DevOps roadmaps."""

        if intent == "CONVERSATIONAL_CLOSING":
            sname = student_profile['student']['name'] if student_profile else "there"
            return f"""### 😊 You're Welcome, {sname}!

Glad I could help! Whenever you want to check your attendance recovery math, review internal scores, practice DSA coding problems, or update your study plan, just send a message. Keep up the great work! 🚀"""

        # -------------------------------------------------------------
        # 0B. PROJECT & ML MODEL ARCHITECTURE
        # -------------------------------------------------------------
        if intent == "PROJECT_ARCHITECTURE":
            return """### 🏗️ AI-Based Student Performance Prediction System — Architecture

This capstone application combines **Machine Learning**, **Natural Language Processing (NLP)**, **Retrieval-Augmented Generation (RAG)**, and **Generative AI** in a full-stack web portal:

---

#### 1. 🧠 Machine Learning Prediction Engine (`train_model.py`)
- **Algorithm:** Scikit-learn **Random Forest Regressor** (150 Decision Trees trained on 1,500 historical student records).
- **Validated Accuracy:** **R² = 0.880** (explains 88% of final exam score variance), **MAE = 3.05 marks**, **RMSE = 3.74 marks**.
- **Feature Importance Weights:**
  - Continuous Internal Marks: **63.0%** (Primary driver)
  - Assignment Scores: **18.1%**
  - Classroom Attendance: **8.1%**
  - Daily Study Hours: **7.8%**
  - Previous Semester Marks: **3.1%**

#### 2. 🤖 5-Layer AI Chatbot Architecture
1. **Conversation Memory (`conversation_memory.py`):** Tracks multi-turn context and resolves follow-up pronouns (*"fix it"*, *"that subject"*).
2. **NLP Engine (`nlp_engine.py`):** Classifies user intents and extracts course & metric entities.
3. **Semantic Search & RAG (`rag_engine.py`):** Uses TF-IDF + Cosine Similarity over live SQLite student records and CS curriculum materials.
4. **LLM Integration (`gemini_service.py`):** Dispatches RAG-grounded prompts to **Google Gemini** (`gemini-3.5-flash-lite`) when configured, with honest local RAG synthesis when offline.
5. **Dual Storage (`chat_storage.py`):** Supports MongoDB with zero-downtime SQLite persistence.

#### 3. 💻 Full-Stack Technology Stack
- **Backend:** Python Flask + Gunicorn (deployed on Render)
- **Frontend:** React 18 (Chatbot UI), Bootstrap 5, Chart.js analytics, Marked.js + Highlight.js
- **Reporting:** Automated downloadable PDF progress reports via ReportLab"""

        # -------------------------------------------------------------
        # 1. DSA / PROGRAMMING SYNTHESIS
        # -------------------------------------------------------------
        if intent == "PROGRAMMING_DSA" or any(k in q for k in ["quicksort", "binary search", "mergesort", "linked list", "bst", "binary tree", "stack", "queue"]):
            # A. QuickSort
            if "quicksort" in q or "quick sort" in q:
                return """### ⚡ QuickSort Algorithm: Deep-Dive & Complete Code

**QuickSort** is an efficient, comparison-based **Divide-and-Conquer** sorting algorithm. It selects an element as a 'pivot' and partitions the array around the pivot such that smaller elements move to the left and larger elements move to the right.

---

#### 🐍 Python Implementation
```python
def quicksort(arr):
    # Base case: arrays with 0 or 1 element are already sorted
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]  # Choose middle element as pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Recursively sort partitions and concatenate
    return quicksort(left) + middle + quicksort(right)

# Example Usage:
numbers = [38, 27, 43, 3, 9, 82, 10]
print("Sorted Array:", quicksort(numbers))
```

---

#### ☕ Java Implementation (In-Place Partitioning)
```java
public class QuickSort {
    public static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);  // Before pivot
            quickSort(arr, pi + 1, high); // After pivot
        }
    }

    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high]; // Last element as pivot
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i]; arr[i] = arr[j]; arr[j] = temp;
            }
        }
        int temp = arr[i + 1]; arr[i + 1] = arr[high]; arr[high] = temp;
        return i + 1;
    }
}
```

---

#### ⏱️ Algorithmic Complexity Breakdown:
- **Best Case:** O(n log n) (Balanced partitions)
- **Average Case:** O(n log n)
- **Worst Case:** O(n²) (Occurs when pivot is smallest/largest, e.g. already sorted list with naive end-pivot)
- **Auxiliary Space:** O(log n) call stack space."""

            # B. Binary Search (excluding Binary Search Tree)
            if "binary search" in q and "tree" not in q:
                return """### 🔍 Binary Search Algorithm: Complete Guide

**Binary Search** is an optimal search algorithm designed for **sorted collections**. It cuts the search space in half with every comparison, achieving logarithmic time complexity.

---

#### 🐍 Python Implementation
```python
def binary_search(arr, target):
    \"\"\"Returns index of target in sorted arr, or -1 if not found.\"\"\"
    low, high = 0, len(arr) - 1

    while low <= high:
        # Prevents overflow in bounded languages: low + (high - low) // 2
        mid = (low + high) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1

# Example:
data = [3, 9, 10, 27, 38, 43, 82]
print("Index of 43:", binary_search(data, 43))  # Output: 5
```

---

#### ☕ Java Implementation
```java
public class BinarySearch {
    public static int search(int[] arr, int target) {
        int low = 0, high = arr.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }
}
```

---

#### ⏱️ Complexity:
- **Time Complexity:** O(log n)
- **Space Complexity:** O(1) iterative, O(log n) recursive."""

            # C. MergeSort
            if "mergesort" in q or "merge sort" in q:
                return """### 🔀 MergeSort Algorithm: Divide-and-Conquer

**MergeSort** guarantees **O(n log n)** worst-case time complexity by recursively halving the list and merging sorted sub-arrays.

---

#### 🐍 Python Implementation
```python
def mergesort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

---

#### ⏱️ Complexity:
- **Time Complexity:** O(n log n) in all cases (Best, Average, Worst).
- **Space Complexity:** O(n) auxiliary storage."""

            # D. Linked List
            if "linked list" in q or "linkedlist" in q:
                return """### 🔗 Singly Linked List: Operations & Reversal

A **Singly Linked List** consists of nodes where each node stores a data value and a pointer (`next`) to the subsequent node.

---

#### 🐍 Python Implementation
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def reverse(self):
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def display(self):
        elements = []
        curr = self.head
        while curr:
            elements.append(str(curr.data))
            curr = curr.next
        print(" -> ".join(elements) + " -> None")
```

---

#### ⏱️ Complexity:
- **Insert at Head:** O(1)
- **Insert at Tail:** O(n) (or O(1) with tail pointer)
- **Reversal:** O(n) time, O(1) space."""

            # E. Binary Search Tree (BST) in C++ and Python
            if "bst" in q or "binary search tree" in q or "binary tree" in q:
                return """### 🌳 Binary Search Tree (BST): C++ & Python Implementation

A **Binary Search Tree (BST)** maintains the invariant: for every node, all keys in the **left subtree are smaller** and all keys in the **right subtree are larger**. An **In-Order Traversal** always visits nodes in sorted ascending order.

---

#### ⚙️ C++ Implementation
```cpp
#include <iostream>
using namespace std;

struct Node {
    int key;
    Node* left;
    Node* right;
    Node(int val) : key(val), left(nullptr), right(nullptr) {}
};

// Insert a key into the BST
Node* insert(Node* root, int key) {
    if (root == nullptr) return new Node(key);
    if (key < root->key)
        root->left = insert(root->left, key);
    else if (key > root->key)
        root->right = insert(root->right, key);
    return root;
}

// Search for a key in the BST
bool search(Node* root, int key) {
    if (root == nullptr) return false;
    if (root->key == key) return true;
    return (key < root->key) ? search(root->left, key) : search(root->right, key);
}

// In-Order Traversal (prints sorted order)
void inorder(Node* root) {
    if (root == nullptr) return;
    inorder(root->left);
    cout << root->key << " ";
    inorder(root->right);
}

int main() {
    Node* root = nullptr;
    int values[] = {50, 30, 70, 20, 40, 60, 80};
    for (int v : values) root = insert(root, v);

    cout << "In-Order Traversal: ";
    inorder(root); // Output: 20 30 40 50 60 70 80
    cout << endl << "Search 40: " << (search(root, 40) ? "Found" : "Not Found") << endl;
    return 0;
}
```

---

#### 🐍 Python Implementation
```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

def insert(root, key):
    if root is None:
        return Node(key)
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
    return root

def inorder(root):
    if root:
        inorder(root.left)
        print(root.key, end=" ")
        inorder(root.right)
```

---

#### ⏱️ Complexity Analysis:
- **Search / Insert / Delete (Average):** O(log n) for balanced trees
- **Worst Case (Skewed Tree):** O(n) — solved using self-balancing AVL or Red-Black Trees
- **Space Complexity:** O(h) recursion stack where h is tree height."""

            # F. Stack & Queue
            if "stack" in q or "queue" in q:
                return """### 📚 Stack (LIFO) & Queue (FIFO) Data Structures

- **Stack (Last-In, First-Out):** Elements are pushed and popped from the **top** (`O(1)`). Used in recursion call stacks, undo features, and DFS.
- **Queue (First-In, First-Out):** Elements enter at the **rear** (`enqueue`) and leave from the **front** (`dequeue`) (`O(1)`). Used in CPU scheduling and BFS.

---

#### 🐍 Python Implementation
```python
from collections import deque

# Stack using Python list
stack = []
stack.append(10)   # push -> O(1)
stack.append(20)
top = stack.pop()  # pop  -> 20, O(1)

# Queue using collections.deque
queue = deque()
queue.append("Job1")      # enqueue -> O(1)
queue.append("Job2")
first = queue.popleft()   # dequeue -> "Job1", O(1)
```"""

        # -------------------------------------------------------------
        # 1B. CAREER GUIDANCE & ROADMAPS
        # -------------------------------------------------------------
        if intent == "CAREER_GUIDANCE":
            sname = student_profile['student']['name'] if student_profile else "Student"
            dept = student_profile['student']['department'] if student_profile else "Computer Science & Engineering"
            return f"""### 🚀 Personalized Career Pathways for {sname} ({dept})

Based on your core coursework, here are the highest-growth engineering tracks and how your subjects map directly to them:

1. **🤖 AI / Machine Learning Engineer**
   - *Key Subjects:* Machine Learning & AI (CS605), Algorithms (CS601)
   - *Core Skills:* Python, Scikit-learn, PyTorch/TensorFlow, RAG & LLMs, Statistics.
2. **💻 Software Development Engineer (Full-Stack / Backend SDE)**
   - *Key Subjects:* Design & Analysis of Algorithms (CS601), DBMS (CS603), Operating Systems (CS602)
   - *Core Skills:* DSA (Trees, Graphs, DP), Java/Python/C++, React.js, Flask/Node.js, SQL/NoSQL.
3. **☁️ Cloud & DevOps / Site Reliability Engineer (SRE)**
   - *Key Subjects:* Cloud Computing & DevOps (CS699), Computer Communication Networks (CS604), OS (CS602)
   - *Core Skills:* Linux internals, Docker, Kubernetes, AWS/GCP, CI/CD, TCP/IP Networking.
4. **🗄️ Data & Database Engineer**
   - *Key Subjects:* DBMS (CS603), Machine Learning (CS605)
   - *Core Skills:* Advanced SQL, Indexing (B+ Trees), MongoDB, ETL Pipelines.

#### 💡 Placement Action Steps:
- Practice 2 DSA problems daily (Arrays, Trees, Graphs, Dynamic Programming).
- Highlight your **AI-Based Student Performance Prediction System** capstone (Random Forest ML + Flask + React + Gemini RAG) prominently on your resume!"""

        # -------------------------------------------------------------
        # 2. STUDENT STUDY PLAN / TIMETABLE
        # -------------------------------------------------------------
        if intent == "STUDENT_STUDY_PLAN" and student_profile:
            s = student_profile['student']
            weak_names = [w['course']['name'] for w in student_profile['weak_subjects']]
            weak_bullet = ", ".join(weak_names) if weak_names else "all core subjects evenly"

            return f"""### 🗓️ Personalized Daily & Weekly Academic Study Plan
**Specially calibrated for {s['name']} ({s['department']}, Semester {s['semester']}):**

---

#### ⏰ Recommended Daily Routine (Monday – Friday)
| Time Window | Focus Area | Recommended Activity |
|---|---|---|
| **07:00 AM – 08:00 AM** | 🌅 Morning Concept Window | Formula memorization, theoretical definitions, network layers |
| **09:00 AM – 04:30 PM** | 🏫 College Lectures & Labs | 100% active attendance (crucial for courses under 75%) |
| **05:30 PM – 07:00 PM** | 💻 Weak Subject Mastery | Focused problem solving on **{weak_bullet}** |
| **07:30 PM – 08:30 PM** | 📝 Continuous Assignments | Complete lab records and upcoming assignment submissions |
| **09:30 PM – 10:30 PM** | 🔄 Past Paper Consolidation | Solve 2 university distinction questions under timed conditions |

---

#### 📅 Weekly Subject Allocation Matrix:
- **Priority 1 (High Focus - 4.5 hrs/week):** Courses currently in attendance or internal shortage ({weak_bullet})
- **Priority 2 (Core Application - 3.5 hrs/week):** Design & Analysis of Algorithms & Machine Learning
- **Priority 3 (Consolidation - 2.5 hrs/week):** Operating Systems & Database Management Systems

#### 🎯 Strategic Study Method: The 50/10 Pomodoro
Study for 50 uninterrupted minutes with notifications disabled, followed by a 10-minute break. This prevents cognitive overload and reinforces long-term exam retention."""

        # -------------------------------------------------------------
        # 3. STUDENT PERFORMANCE INQUIRIES (GROUNDED IN PROFILE)
        # -------------------------------------------------------------
        if student_profile and (nlp_analysis.get("is_student_query") or intent.startswith("STUDENT_")):
            from gemini_service import format_student_performance_report
            return format_student_performance_report(student_profile, query)

        # -------------------------------------------------------------
        # 4. RAG KNOWLEDGE BASE CONCEPT SYNTHESIS
        # -------------------------------------------------------------
        if retrieved_chunks:
            top_chunk = retrieved_chunks[0]
            related_points = []
            for c in retrieved_chunks[:3]:
                related_points.append(f"#### 📌 {c['title']} ({c.get('course_code', 'General')})\n{c['content']}")

            chunks_body = "\n\n---\n\n".join(related_points)

            return f"""### 📚 Coursework & Concept Knowledge (Retrieved via RAG)

{chunks_body}

---
💡 **Examination Tip:** In semester exams, structure your answers with:
1. Formal definitions and key principles.
2. Architecture / schematic diagrams or code snippets.
3. Time/Space complexity and real-world industrial use cases."""

        # -------------------------------------------------------------
        # 5. HONEST HANDLING OF UNKNOWN / OUT-OF-DOMAIN QUESTIONS
        # -------------------------------------------------------------
        clean_q = query.strip()
        return f"""### 🤖 AI Study Copilot Notice
I could not find information regarding *"{clean_q}"* in your student academic records or college curriculum knowledge base.

As an educational assistant for the **AI-Based Student Performance Prediction System**, I specialize in:
- **Your Academic Performance:** Attendance analysis, internal marks, predicted final exam marks, weak and strong areas.
- **Computer Science Coursework:** Algorithms & DSA, Operating Systems, DBMS, Networks, Machine Learning, Cloud Computing.
- **Study Timetables & Placements:** Personalized revision schedules and engineering career roadmaps.

---
💡 **Next Step:** 
- Ask me about your attendance, predicted marks, weak subjects, or any CS engineering topic!
- To enable open-ended general web knowledge using live Generative AI, please configure a free Google Gemini API Key via **⚙️ API Key**."""
