import streamlit as st
from dotenv import dotenv_values
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote_plus
import re
import ssl
import certifi

from dotenv import dotenv_values
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError
from argon2 import PasswordHasher, exceptions as argon2_exceptions


# --------------------------------------------------
# 1. MongoDB Atlas connection
# --------------------------------------------------

ENV_PATH = Path(__file__).resolve().parent / ".env"
config = dotenv_values(str(ENV_PATH), encoding="utf-8-sig")

# Read credentials from Streamlit Cloud Secrets first,
# then fall back to the local .env file.
try:
    username = st.secrets.get("MONGODB_USERNAME") or config.get("MONGODB_USERNAME")
    password = st.secrets.get("MONGODB_PASSWORD") or config.get("MONGODB_PASSWORD")
except Exception:
    username = config.get("MONGODB_USERNAME")
    password = config.get("MONGODB_PASSWORD")

if not username or not password:
    raise ValueError(
        "MongoDB credentials are missing. Configure MONGODB_USERNAME "
        "and MONGODB_PASSWORD in Streamlit Secrets or your local .env file."
    )

uri = (
    f"mongodb+srv://{quote_plus(username)}:{quote_plus(password)}"
    "@intellivault.neba1su.mongodb.net/intellivault"
    "?retryWrites=true&w=majority"
)

# Explicit TLS options fix the TLSV1_ALERT_INTERNAL_ERROR on Python 3.11
# with MongoDB Atlas. certifi provides the correct CA bundle on Windows.
client = MongoClient(
    uri,
    serverSelectionTimeoutMS=20000,
    connectTimeoutMS=20000,
    socketTimeoutMS=20000,
    tls=True,
    tlsCAFile=certifi.where(),  # <-- key fix for Windows SSL errors
    retryWrites=True,
    retryReads=True,
)

db = client["intellivault"]

users = db["users"]
pdf_history = db["pdf_history"]
qa_history = db["qa_history"]

password_hasher = PasswordHasher()


# --------------------------------------------------
# 2. Database indexes — created lazily, never crash app
# --------------------------------------------------

_indexes_created = False


def ensure_indexes():
    """
    Create indexes on first use.
    Called by each public function so startup never blocks.
    Silently skips if MongoDB is unreachable.
    """
    global _indexes_created
    if _indexes_created:
        return
    try:
        users.create_index(
            [("email", ASCENDING)],
            unique=True,
        )
        pdf_history.create_index(
            [("user_id", ASCENDING), ("file_hash", ASCENDING)],
            unique=True,
        )
        qa_history.create_index(
            [("user_id", ASCENDING), ("created_at", DESCENDING)],
        )
        _indexes_created = True
    except Exception:
        # Don't crash — indexes will be retried on next call
        pass


# --------------------------------------------------
# 3. Test MongoDB connection
# --------------------------------------------------

def test_connection():
    client.admin.command("ping")
    return True


# --------------------------------------------------
# 4. Register a new user
# --------------------------------------------------

def register_user(email, password):
    ensure_indexes()
    email = (email or "").strip().lower()

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return False, "Enter a valid email address."

    if not password or len(password) < 12:
        return False, "Password must contain at least 12 characters."

    if len(password) > 128:
        return False, "Password must not exceed 128 characters."

    password_hash = password_hasher.hash(password)

    try:
        result = users.insert_one({
            "email": email,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc),
        })

        return True, str(result.inserted_id)

    except DuplicateKeyError:
        return False, "An account with this email already exists."


# --------------------------------------------------
# 5. Log in an existing user
# --------------------------------------------------

def login_user(email, password):
    ensure_indexes()
    email = (email or "").strip().lower()

    user = users.find_one({"email": email})

    if not user:
        return None

    try:
        password_hasher.verify(
            user["password_hash"],
            password or "",
        )

    except (
        argon2_exceptions.VerifyMismatchError,
        argon2_exceptions.VerificationError,
        argon2_exceptions.InvalidHashError,
    ):
        return None

    if password_hasher.check_needs_rehash(user["password_hash"]):
        users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "password_hash": password_hasher.hash(password)
                }
            },
        )

    return {
        "user_id": str(user["_id"]),
        "email": user["email"],
    }


# --------------------------------------------------
# 6. Save uploaded document history
# --------------------------------------------------

def save_pdf_history(user_id, filename, file_hash):
    pdf_history.update_one(
        {
            "user_id": str(user_id),
            "file_hash": file_hash,
        },
        {
            "$set": {
                "filename": filename,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {
                "created_at": datetime.now(timezone.utc),
            },
        },
        upsert=True,
    )


# --------------------------------------------------
# 7. Retrieve one user's document history
# --------------------------------------------------

def get_pdf_history(user_id):
    records = pdf_history.find(
        {"user_id": str(user_id)}
    ).sort("created_at", DESCENDING)

    return [
        {
            "filename": item.get("filename", ""),
            "file_hash": item.get("file_hash", ""),
            "created_at": item.get("created_at"),
        }
        for item in records
    ]


# --------------------------------------------------
# 8. Save a question and its answer
# --------------------------------------------------

def save_qa_history(
    user_id,
    question,
    answer,
    filename=None,
    mode=None,
):
    qa_history.insert_one({
        "user_id": str(user_id),
        "question": question,
        "answer": answer,
        "filename": filename,
        "mode": mode,
        "created_at": datetime.now(timezone.utc),
    })


# --------------------------------------------------
# 9. Retrieve one user's chat history
# --------------------------------------------------

def get_qa_history(user_id, limit=100):
    records = qa_history.find(
        {"user_id": str(user_id)}
    ).sort("created_at", DESCENDING).limit(limit)

    history = list(records)
    history.reverse()

    return [
        {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "filename": item.get("filename"),
            "mode": item.get("mode"),
            "created_at": item.get("created_at"),
        }
        for item in history
    ]
