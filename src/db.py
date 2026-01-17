import sqlite3
import uuid
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_NAME = os.path.join(BASE_DIR, "chat_history.db")

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    """Create Table If Not Exist"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 1: List of Chat Sessions (Title & Time)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TIMESTAMP
        )
    """)
    
    # Table 2: Message Content (Role: user/assistant & Content)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """)
    
    conn.commit()
    conn.close()

def create_session(title="New Chat"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (title, created_at) VALUES (?, ?)",
        (title, datetime.now())
    )
    # Retrieve the ID that was just created by SQLite.
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def add_message(session_id, role, content):
    """save chat to database"""
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now())
    )
    conn.commit()
    conn.close()

def get_messages(session_id):
    """fetch all chat from specific session"""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at", 
        (session_id,)
    )
    messages = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return messages

def get_all_sessions():
    """Retrieve the list of all chats for the Sidebar"""
    conn = get_connection()
    cursor = conn.execute("SELECT id, title FROM sessions ORDER BY created_at DESC")
    sessions = [{"id": row[0], "title": row[1]} for row in cursor.fetchall()]
    conn.close()
    return sessions

def get_session_title(session_id):
    """Retrieve Title based on ID"""
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return row[0] if row else "Untitled Chat"
    except Exception:
        return "Session Error"
    finally:
        conn.close()

def update_session_title(session_id, new_title):
    """Update Chat Title"""
    conn = get_connection()
    conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
    conn.commit()
    conn.close()

def delete_session(session_id):
    conn = get_connection()
    # 1. delete messages
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    # 2. delete session
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

