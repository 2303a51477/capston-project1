"""
chat_storage.py
Persistent storage engine for AI Chatbot Assistant.
Supports MongoDB (via pymongo) with seamless automated fallback to SQLite
when MongoDB is offline or unconfigured, ensuring 100% zero-crash operation.
"""

import os
import uuid
import datetime
import sqlite3
from typing import List, Dict, Optional, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/student_ai_chatbot")
DB_PATH = os.path.join(os.path.dirname(__file__), 'database.sqlite')

_mongo_client = None
_mongo_db = None
_storage_backend = None  # 'mongodb' or 'sqlite'
_mongo_error = None


def get_sqlite_conn():
    """Returns an SQLite connection configured with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_sqlite_tables():
    """Initializes chat tables in SQLite as fallback."""
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_conversations (
        id TEXT PRIMARY KEY,
        student_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata TEXT DEFAULT '{}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE
    )
    """)
    conn.commit()
    conn.close()


def init_storage(force_recheck: bool = False):
    """
    Attempts to initialize MongoDB connection.
    If MongoDB is not reachable or fails within 1.5 seconds, falls back to SQLite.
    """
    global _mongo_client, _mongo_db, _storage_backend, _mongo_error

    if _storage_backend is not None and not force_recheck:
        return _storage_backend

    # Ensure SQLite tables exist regardless (as safety net)
    _init_sqlite_tables()

    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/student_ai_chatbot")
    try:
        import pymongo
        client = pymongo.MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=1500,
            connectTimeoutMS=1500
        )
        # Test connection with ping
        client.admin.command('ping')
        
        # Determine database name
        db_name = "student_ai_chatbot"
        if "/" in mongo_uri.split("?")[0]:
            parts = mongo_uri.split("?")[0].rstrip("/").split("/")
            if len(parts) > 3 and parts[-1]:
                db_name = parts[-1]

        _mongo_client = client
        _mongo_db = client[db_name]
        _storage_backend = 'mongodb'
        _mongo_error = None

        # Create indexes for optimal querying
        _mongo_db.conversations.create_index([("student_id", 1), ("updated_at", -1)])
        _mongo_db.messages.create_index([("conversation_id", 1), ("created_at", 1)])
        print(f"[ChatStorage] Connected to MongoDB successfully (database: '{db_name}').")

    except Exception as e:
        _storage_backend = 'sqlite'
        _mongo_error = str(e)
        print(f"[ChatStorage] MongoDB offline or unreachable ({e}). Using persistent SQLite storage fallback.")

    return _storage_backend


def get_storage_status() -> Dict[str, Any]:
    """Returns current active storage backend details."""
    init_storage()
    return {
        "backend": _storage_backend,
        "is_mongodb": _storage_backend == 'mongodb',
        "mongo_uri_configured": bool(os.environ.get("MONGO_URI")),
        "mongo_error": _mongo_error if _storage_backend == 'sqlite' else None,
        "description": "MongoDB (Active)" if _storage_backend == 'mongodb' else "SQLite Local Database (Persistent Fallback)"
    }


def create_conversation(student_id: int, title: str = "New Conversation") -> Dict[str, Any]:
    """Creates a new conversation thread for a student."""
    init_storage()
    conv_id = f"conv_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.datetime.utcnow().isoformat()

    if _storage_backend == 'mongodb':
        doc = {
            "conversation_id": conv_id,
            "student_id": int(student_id),
            "title": title.strip() or "New Conversation",
            "created_at": now_iso,
            "updated_at": now_iso
        }
        _mongo_db.conversations.insert_one(doc)
        return {
            "id": conv_id,
            "student_id": student_id,
            "title": doc["title"],
            "created_at": now_iso,
            "updated_at": now_iso,
            "message_count": 0
        }
    else:
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_conversations (id, student_id, title, created_at, updated_at)
            VALUES (?, ?, ?, datetime('now'), datetime('now'))
        """, (conv_id, student_id, title.strip() or "New Conversation"))
        conn.commit()
        conn.close()
        return {
            "id": conv_id,
            "student_id": student_id,
            "title": title.strip() or "New Conversation",
            "created_at": now_iso,
            "updated_at": now_iso,
            "message_count": 0
        }


def get_conversations(student_id: int) -> List[Dict[str, Any]]:
    """Returns all conversations for a student, ordered by most recently updated."""
    init_storage()
    conversations = []

    if _storage_backend == 'mongodb':
        cursor = _mongo_db.conversations.find({"student_id": int(student_id)}).sort("updated_at", -1)
        for doc in cursor:
            cid = doc.get("conversation_id") or str(doc.get("_id"))
            msg_count = _mongo_db.messages.count_documents({"conversation_id": cid})
            # Fetch latest message snippet if any
            last_msg = _mongo_db.messages.find_one({"conversation_id": cid}, sort=[("created_at", -1)])
            last_preview = last_msg["content"][:60] + "..." if last_msg and len(last_msg["content"]) > 60 else (last_msg["content"] if last_msg else "")

            conversations.append({
                "id": cid,
                "student_id": doc["student_id"],
                "title": doc.get("title", "Conversation"),
                "created_at": str(doc.get("created_at", "")),
                "updated_at": str(doc.get("updated_at", "")),
                "message_count": msg_count,
                "last_preview": last_preview
            })
    else:
        conn = get_sqlite_conn()
        rows = conn.execute("""
            SELECT c.*, 
                   COUNT(m.id) as message_count,
                   (SELECT content FROM chat_messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_msg
            FROM chat_conversations c
            LEFT JOIN chat_messages m ON m.conversation_id = c.id
            WHERE c.student_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """, (student_id,)).fetchall()
        conn.close()

        for r in rows:
            last_msg = r['last_msg'] or ""
            preview = last_msg[:60] + "..." if len(last_msg) > 60 else last_msg
            conversations.append({
                "id": r['id'],
                "student_id": r['student_id'],
                "title": r['title'],
                "created_at": str(r['created_at']),
                "updated_at": str(r['updated_at']),
                "message_count": r['message_count'],
                "last_preview": preview
            })

    return conversations


def get_conversation(student_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves metadata of a single conversation owned by the student."""
    init_storage()

    if _storage_backend == 'mongodb':
        doc = _mongo_db.conversations.find_one({
            "conversation_id": conversation_id,
            "student_id": int(student_id)
        })
        if not doc:
            return None
        return {
            "id": doc.get("conversation_id"),
            "student_id": doc["student_id"],
            "title": doc.get("title", "Conversation"),
            "created_at": str(doc.get("created_at", "")),
            "updated_at": str(doc.get("updated_at", ""))
        }
    else:
        conn = get_sqlite_conn()
        row = conn.execute(
            "SELECT * FROM chat_conversations WHERE id = ? AND student_id = ?",
            (conversation_id, student_id)
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "id": row['id'],
            "student_id": row['student_id'],
            "title": row['title'],
            "created_at": str(row['created_at']),
            "updated_at": str(row['updated_at'])
        }


def update_conversation_title(student_id: int, conversation_id: str, title: str) -> bool:
    """Updates the title of a conversation."""
    init_storage()
    clean_title = title.strip()[:100] or "Conversation"
    now_iso = datetime.datetime.utcnow().isoformat()

    if _storage_backend == 'mongodb':
        res = _mongo_db.conversations.update_one(
            {"conversation_id": conversation_id, "student_id": int(student_id)},
            {"$set": {"title": clean_title, "updated_at": now_iso}}
        )
        return res.modified_count > 0
    else:
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE chat_conversations 
            SET title = ?, updated_at = datetime('now')
            WHERE id = ? AND student_id = ?
        """, (clean_title, conversation_id, student_id))
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated


def delete_conversation(student_id: int, conversation_id: str) -> bool:
    """Deletes a conversation and its messages."""
    init_storage()

    if _storage_backend == 'mongodb':
        res = _mongo_db.conversations.delete_one({
            "conversation_id": conversation_id,
            "student_id": int(student_id)
        })
        if res.deleted_count > 0:
            _mongo_db.messages.delete_many({"conversation_id": conversation_id})
            return True
        return False
    else:
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM chat_conversations WHERE id = ? AND student_id = ?", (conversation_id, student_id))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted


def clear_conversation_messages(student_id: int, conversation_id: str) -> bool:
    """Clears all messages within a conversation while keeping the thread intact."""
    init_storage()

    # Verify ownership
    conv = get_conversation(student_id, conversation_id)
    if not conv:
        return False

    now_iso = datetime.datetime.utcnow().isoformat()
    if _storage_backend == 'mongodb':
        _mongo_db.messages.delete_many({"conversation_id": conversation_id})
        _mongo_db.conversations.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"updated_at": now_iso}}
        )
        return True
    else:
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("UPDATE chat_conversations SET updated_at = datetime('now') WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()
        return True


def add_message(conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Appends a new message ('user' or 'model') to a conversation.
    Updates the conversation's updated_at timestamp.
    """
    init_storage()
    now_iso = datetime.datetime.utcnow().isoformat()
    meta = metadata or {}

    if _storage_backend == 'mongodb':
        doc = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "metadata": meta,
            "created_at": now_iso
        }
        res = _mongo_db.messages.insert_one(doc)
        _mongo_db.conversations.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"updated_at": now_iso}}
        )
        return {
            "id": str(res.inserted_id),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "metadata": meta,
            "created_at": now_iso
        }
    else:
        import json
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_messages (conversation_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        """, (conversation_id, role, content, json.dumps(meta)))
        msg_id = cursor.lastrowid
        cursor.execute("""
            UPDATE chat_conversations 
            SET updated_at = datetime('now') 
            WHERE id = ?
        """, (conversation_id,))
        conn.commit()
        conn.close()

        return {
            "id": msg_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "metadata": meta,
            "created_at": now_iso
        }


def get_messages(conversation_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Returns message history for a conversation in chronological order."""
    init_storage()
    messages = []

    if _storage_backend == 'mongodb':
        cursor = _mongo_db.messages.find({"conversation_id": conversation_id}).sort("created_at", 1).limit(limit)
        for doc in cursor:
            messages.append({
                "id": str(doc.get("_id")),
                "conversation_id": doc["conversation_id"],
                "role": doc["role"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "created_at": str(doc.get("created_at", ""))
            })
    else:
        import json
        conn = get_sqlite_conn()
        rows = conn.execute("""
            SELECT * FROM chat_messages 
            WHERE conversation_id = ? 
            ORDER BY id ASC 
            LIMIT ?
        """, (conversation_id, limit)).fetchall()
        conn.close()

        for r in rows:
            meta = {}
            if r['metadata']:
                try:
                    meta = json.loads(r['metadata'])
                except Exception:
                    meta = {}
            messages.append({
                "id": r['id'],
                "conversation_id": r['conversation_id'],
                "role": r['role'],
                "content": r['content'],
                "metadata": meta,
                "created_at": str(r['created_at'])
            })

    return messages
