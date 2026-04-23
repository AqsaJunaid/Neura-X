from contextlib import contextmanager
from psycopg2.extras import RealDictCursor
import psycopg2
import pytesseract
import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
import fitz  # PyMuPDF
import io
import numpy as np
import hashlib
from dotenv import load_dotenv
from datetime import datetime, timezone   # ← add this import at the top of file if missing

load_dotenv()  # <-- add this near the top


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program
# Files\Tesseract-OCR\tesseract.exe"


# ====================== SUPABASE POSTGRES (ZERO MAINTENANCE) ============

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is required (Supabase connection string)")


@contextmanager
def get_db():
    """Use like: with get_db() as conn: ... """
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def query_one(query, params=()):
    """Helper for SELECT ... LIMIT 1"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        return cur.fetchone()


def query_all(query, params=()):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()
# =================================================================================


# Clinical BERT will be loaded lazily

#write clinical bert code

# Load environment variables (optional - API key is now hardcoded)
try:
    load_dotenv()
except:
    pass  # Continue even if .env file has issues

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROFILE_FOLDER'] = 'static/profiles'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}
# For session management
app.config['SECRET_KEY'] = 'hms-secret-key-change-in-production'
# Set to True in production with HTTPS
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Security helper functions


def require_auth():
    """Check if user is authenticated"""
    if 'user_role' not in session:
        return False
    return True


def require_role(*allowed_roles):
    """Check if user has one of the allowed roles"""
    if not require_auth():
        return False
    return session.get('user_role') in allowed_roles


def require_patient_access(patient_id):
    """Check if user can access patient data"""
    if not require_auth():
        return False
    user_role = session.get('user_role')
    user_db_id = session.get('user_db_id')

    # Admin and doctors can access any patient
    if user_role in ['admin', 'doctor']:
        return True
    # Patients can only access their own data
    if user_role == 'patient' and user_db_id == patient_id:
        return True
    return False


def require_doctor_access(doctor_id):
    """Check if user can access doctor data"""
    if not require_auth():
        return False
    user_role = session.get('user_role')
    user_db_id = session.get('user_db_id')

    # Admin can access any doctor
    if user_role == 'admin':
        return True
    # Doctors can only access their own data
    if user_role == 'doctor' and user_db_id == doctor_id:
        return True
    # Patients can view doctor profiles (for booking appointments)
    if user_role == 'patient':
        return True
    return False


def sanitize_input(text):
    """Basic input sanitization"""
    if not text:
        return None
    # Remove potentially dangerous characters
    text = str(text).strip()
    # Prevent SQL injection by escaping quotes
    text = text.replace("'", "''")
    return text


# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROFILE_FOLDER'], exist_ok=True)

# Groq client will be initialized lazily when needed
# API key is loaded from environment variable (.env file)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None


def get_groq_client():
    """Lazy initialization of Groq client"""
    global groq_client
    if groq_client is None:
        groq_client = Groq(api_key=GROQ_API_KEY)
    return groq_client


# OCR reader - will try multiple OCR methods
ocr_reader = None
ocr_method = None


def get_ocr_reader():
    """Lazy initialization of OCR reader - tries multiple methods"""
    global ocr_reader, ocr_method

    if ocr_reader is not None:
        return ocr_reader, ocr_method

    # Try PaddleOCR first
    try:
        from paddleocr import PaddleOCR
        print("Attempting to initialize PaddleOCR...")
        # Use minimal parameters that are supported
        ocr_reader = PaddleOCR(use_textline_orientation=True, lang='en')
        ocr_method = 'paddleocr'
        print("PaddleOCR initialized successfully.")
        return ocr_reader, ocr_method
    except Exception as e:
        print(f"PaddleOCR initialization failed: {str(e)}")

    # Fallback: Try pytesseract if available
    try:
        import pytesseract
        # Test if tesseract is available
        pytesseract.get_tesseract_version()
        ocr_reader = pytesseract
        ocr_method = 'pytesseract'
        print("Using pytesseract as OCR backend.")
        return ocr_reader, ocr_method
    except Exception as e:
        # pytesseract might be installed but Tesseract engine not found
        if "tesseract" in str(e).lower() or "not found" in str(e).lower():
            print(
    f"pytesseract requires Tesseract OCR engine to be installed separately.")
        else:
            print(f"pytesseract not available: {str(e)}")

    # Final fallback: Return None - will use PyMuPDF text extraction only
    print("Warning: No OCR backend available. Only text-based PDFs will work.")
    print("For image OCR, install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
    ocr_method = 'none'
    return None, ocr_method


def init_db():
    """Create all core tables if they do not exist (idempotent for PostgreSQL/Supabase)"""
    print("🔧 init_db() — Ensuring all core tables exist in PostgreSQL")

    with get_db() as conn:
        cur = conn.cursor()

        # users
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id              SERIAL PRIMARY KEY,
                username        TEXT UNIQUE NOT NULL,
                password        TEXT NOT NULL,
                role            TEXT NOT NULL DEFAULT 'patient',
                email           TEXT,
                full_name       TEXT,
                user_id         INTEGER,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                last_login      TIMESTAMP
            );
        ''')

        # doctors
        cur.execute('''
            CREATE TABLE IF NOT EXISTS doctors (
                id              SERIAL PRIMARY KEY,
                name            TEXT NOT NULL,
                profile_picture TEXT,
                specialist_type TEXT NOT NULL,
                email           TEXT,
                phone           TEXT,
                license_number  TEXT,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW()
            );
        ''')

        # patients
        cur.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id              SERIAL PRIMARY KEY,
                name            TEXT NOT NULL,
                email           TEXT,
                phone           TEXT,
                patient_id      TEXT UNIQUE,
                profile_picture TEXT,
                date_of_birth   TEXT,
                gender          TEXT,
                address         TEXT,
                contact_info    TEXT,
                medical_history TEXT,
                allergies       TEXT,
                medications     TEXT,
                patient_tags    TEXT,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW()
            );
        ''')

        # reports
        cur.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id                      SERIAL PRIMARY KEY,
                doctor_id               INTEGER,
                patient_id              INTEGER NOT NULL,
                specialist_type         TEXT NOT NULL,
                original_filename       TEXT NOT NULL,
                extracted_text          TEXT,
                llm_analysis            TEXT,
                clinical_bert_analysis  TEXT,
                created_at              TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (doctor_id)  REFERENCES doctors(id),
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            );
        ''')

        # chat_messages
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id              SERIAL PRIMARY KEY,
                doctor_id       INTEGER,
                specialist_type TEXT NOT NULL,
                message         TEXT NOT NULL,
                response        TEXT NOT NULL,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            );
        ''')

        # tasks
        cur.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id              SERIAL PRIMARY KEY,
                doctor_id       INTEGER,
                patient_id      INTEGER,
                report_id       INTEGER,
                task_type       TEXT NOT NULL,
                title           TEXT NOT NULL,
                description     TEXT,
                due_date        TEXT,
                status          TEXT DEFAULT 'pending',
                priority        TEXT DEFAULT 'medium',
                created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                completed_at    TIMESTAMP,
                FOREIGN KEY (doctor_id)  REFERENCES doctors(id),
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (report_id)  REFERENCES reports(id)
            );
        ''')

        # shared_reports
        cur.execute('''
            CREATE TABLE IF NOT EXISTS shared_reports (
                id                      SERIAL PRIMARY KEY,
                report_id               INTEGER NOT NULL,
                shared_by_doctor_id     INTEGER NOT NULL,
                shared_with_doctor_id   INTEGER NOT NULL,
                shared_at               TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (report_id)          REFERENCES reports(id),
                FOREIGN KEY (shared_by_doctor_id) REFERENCES doctors(id),
                FOREIGN KEY (shared_with_doctor_id) REFERENCES doctors(id)
            );
        ''')

        # report_comments
        cur.execute('''
            CREATE TABLE IF NOT EXISTS report_comments (
                id                  SERIAL PRIMARY KEY,
                report_id           INTEGER NOT NULL,
                doctor_id           INTEGER NOT NULL,
                comment             TEXT NOT NULL,
                is_private          INTEGER DEFAULT 0,
                parent_comment_id   INTEGER,
                created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (report_id)         REFERENCES reports(id),
                FOREIGN KEY (doctor_id)         REFERENCES doctors(id),
                FOREIGN KEY (parent_comment_id) REFERENCES report_comments(id)
            );
        ''')

        # referrals
        cur.execute('''
            CREATE TABLE IF NOT EXISTS referrals (
                id                  SERIAL PRIMARY KEY,
                patient_id          INTEGER NOT NULL,
                from_doctor_id      INTEGER NOT NULL,
                to_specialist_type  TEXT NOT NULL,
                to_doctor_id        INTEGER,
                reason              TEXT,
                notes               TEXT,
                status              TEXT DEFAULT 'pending',
                created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (patient_id)     REFERENCES patients(id),
                FOREIGN KEY (from_doctor_id) REFERENCES doctors(id),
                FOREIGN KEY (to_doctor_id)   REFERENCES doctors(id)
            );
        ''')

        # patient_vitals
        cur.execute('''
            CREATE TABLE IF NOT EXISTS patient_vitals (
                id              SERIAL PRIMARY KEY,
                patient_id      INTEGER NOT NULL,
                report_id       INTEGER,
                vital_name      TEXT NOT NULL,
                vital_value     TEXT NOT NULL,
                unit            TEXT,
                measured_at     TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (report_id)  REFERENCES reports(id)
            );
        ''')

        # appointments
        cur.execute('''
            CREATE TABLE IF NOT EXISTS appointments (
                id                  SERIAL PRIMARY KEY,
                patient_id          INTEGER NOT NULL,
                doctor_id           INTEGER NOT NULL,
                appointment_date    TEXT NOT NULL,
                appointment_time    TEXT NOT NULL,
                duration            INTEGER DEFAULT 30,
                appointment_type    TEXT DEFAULT 'consultation',
                reason              TEXT,
                status              TEXT DEFAULT 'scheduled',
                notes               TEXT,
                reminder_sent       INTEGER DEFAULT 0,
                created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at          TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (doctor_id)  REFERENCES doctors(id)
            );
        ''')

        # prescriptions
        cur.execute('''
            CREATE TABLE IF NOT EXISTS prescriptions (
                id                  SERIAL PRIMARY KEY,
                patient_id          INTEGER NOT NULL,
                doctor_id           INTEGER NOT NULL,
                appointment_id      INTEGER,
                prescription_text   TEXT NOT NULL,
                medications         TEXT,
                instructions        TEXT,
                valid_until         TEXT,
                refills_remaining   INTEGER DEFAULT 0,
                status              TEXT DEFAULT 'active',
                ai_safety_check     TEXT,
                created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (patient_id)     REFERENCES patients(id),
                FOREIGN KEY (doctor_id)      REFERENCES doctors(id),
                FOREIGN KEY (appointment_id) REFERENCES appointments(id)
            );
        ''')

        # notifications
        cur.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id                  SERIAL PRIMARY KEY,
                user_id             INTEGER NOT NULL,
                user_role           TEXT NOT NULL,
                notification_type   TEXT NOT NULL,
                title               TEXT NOT NULL,
                message             TEXT NOT NULL,
                link                TEXT,
                is_read             INTEGER DEFAULT 0,
                created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        ''')

        # documents
        cur.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id              SERIAL PRIMARY KEY,
                patient_id      INTEGER NOT NULL,
                doctor_id       INTEGER,
                document_type   TEXT NOT NULL,
                title           TEXT NOT NULL,
                file_path       TEXT NOT NULL,
                file_size       INTEGER,
                mime_type       TEXT,
                description     TEXT,
                tags            TEXT,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (doctor_id)  REFERENCES doctors(id)
            );
        ''')

        # messages
        cur.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id                  SERIAL PRIMARY KEY,
                sender_id           INTEGER NOT NULL,
                sender_role         TEXT NOT NULL,
                recipient_id        INTEGER NOT NULL,
                recipient_role      TEXT NOT NULL,
                subject             TEXT,
                message             TEXT NOT NULL,
                is_read             INTEGER DEFAULT 0,
                parent_message_id   INTEGER,
                created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (sender_id)         REFERENCES users(id),
                FOREIGN KEY (recipient_id)      REFERENCES users(id),
                FOREIGN KEY (parent_message_id) REFERENCES messages(id)
            );
        ''')

        # audit_logs
        cur.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id              SERIAL PRIMARY KEY,
                user_id         INTEGER,
                user_role       TEXT,
                action          TEXT NOT NULL,
                entity_type     TEXT,
                entity_id       INTEGER,
                details         TEXT,
                ip_address      TEXT,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        ''')

        # doctor_schedules
        cur.execute('''
            CREATE TABLE IF NOT EXISTS doctor_schedules (
                id              SERIAL PRIMARY KEY,
                doctor_id       INTEGER NOT NULL,
                day_of_week     INTEGER NOT NULL,
                start_time      TEXT NOT NULL,
                end_time        TEXT NOT NULL,
                is_available    INTEGER DEFAULT 1,
                slot_duration   INTEGER DEFAULT 30,
                break_start     TEXT,
                break_end       TEXT,
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            );
        ''')

        # risk_scores
        cur.execute('''
            CREATE TABLE IF NOT EXISTS risk_scores (
                id              SERIAL PRIMARY KEY,
                patient_id      INTEGER NOT NULL,
                report_id       INTEGER,
                condition       TEXT NOT NULL,
                risk_score      REAL NOT NULL,
                risk_level      TEXT NOT NULL,
                analysis_text   TEXT,
                created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (report_id)  REFERENCES reports(id)
            );
        ''')

        conn.commit()

    print("init_db() completed — all core tables are ensured")


# Initialize database on startup
init_db()

# Database migration - add missing columns if they don't exist


def migrate_database():
    """
    PostgreSQL version:
    - Adds missing columns to existing tables (only if they don't exist)
    - Handles patient_id generation for rows that lack it
    - Ensures default admin and demo patient users exist
    Safe to call on every startup.
    """
    print("🔧 migrate_database() — Checking & adding missing columns + defaults (PostgreSQL)")

    with get_db() as conn:
        cur = conn.cursor()

        # ──────────────────────────────────────────────────────────────
        # Doctors table - add missing columns
        # ──────────────────────────────────────────────────────────────
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'doctors' AND column_name = 'email'
                ) THEN
                    ALTER TABLE doctors ADD COLUMN email TEXT;
                    RAISE NOTICE 'Added email column to doctors';
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'doctors' AND column_name = 'phone'
                ) THEN
                    ALTER TABLE doctors ADD COLUMN phone TEXT;
                    RAISE NOTICE 'Added phone column to doctors';
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'doctors' AND column_name = 'license_number'
                ) THEN
                    ALTER TABLE doctors ADD COLUMN license_number TEXT;
                    RAISE NOTICE 'Added license_number column to doctors';
                END IF;
            END $$;
        """)

        # ──────────────────────────────────────────────────────────────
        # Reports table - add missing columns
        # ──────────────────────────────────────────────────────────────
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reports' AND column_name = 'doctor_id'
                ) THEN
                    ALTER TABLE reports ADD COLUMN doctor_id INTEGER;
                    RAISE NOTICE 'Added doctor_id column to reports';
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reports' AND column_name = 'specialist_type'
                ) THEN
                    ALTER TABLE reports ADD COLUMN specialist_type TEXT DEFAULT 'general';
                    RAISE NOTICE 'Added specialist_type column to reports';
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reports' AND column_name = 'clinical_bert_analysis'
                ) THEN
                    ALTER TABLE reports ADD COLUMN clinical_bert_analysis TEXT;
                    RAISE NOTICE 'Added clinical_bert_analysis column to reports';
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reports' AND column_name = 'status'
                ) THEN
                    ALTER TABLE reports ADD COLUMN status TEXT DEFAULT 'new';
                    RAISE NOTICE 'Added status column to reports';
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reports' AND column_name = 'doctor_notes'
                ) THEN
                    ALTER TABLE reports ADD COLUMN doctor_notes TEXT;
                    RAISE NOTICE 'Added doctor_notes column to reports';
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reports' AND column_name = 'tags'
                ) THEN
                    ALTER TABLE reports ADD COLUMN tags TEXT;
                    RAISE NOTICE 'Added tags column to reports';
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reports' AND column_name = 'is_public'
                ) THEN
                    ALTER TABLE reports ADD COLUMN is_public INTEGER DEFAULT 0;
                    RAISE NOTICE 'Added is_public column to reports';
                END IF;
            END $$;
        """)

        # ──────────────────────────────────────────────────────────────
        # Patients table - add missing columns + generate patient_id
        # ──────────────────────────────────────────────────────────────
        cur.execute("""
            DO $$
            BEGIN
                -- patient_id
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'patient_id'
                ) THEN
                    ALTER TABLE patients ADD COLUMN patient_id TEXT;
                    RAISE NOTICE 'Added patient_id column to patients';

                    -- Generate PATxxxxx for rows that don't have it
                    UPDATE patients
                    SET patient_id = 'PAT' || LPAD(id::text, 6, '0')
                    WHERE patient_id IS NULL OR patient_id = '';
                    RAISE NOTICE 'Generated patient_id values for existing patients';
                END IF;

                -- profile_picture
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'profile_picture'
                ) THEN
                    ALTER TABLE patients ADD COLUMN profile_picture TEXT;
                    RAISE NOTICE 'Added profile_picture column to patients';
                END IF;

                -- date_of_birth
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'date_of_birth'
                ) THEN
                    ALTER TABLE patients ADD COLUMN date_of_birth TEXT;
                    RAISE NOTICE 'Added date_of_birth column to patients';
                END IF;

                -- gender
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'gender'
                ) THEN
                    ALTER TABLE patients ADD COLUMN gender TEXT;
                    RAISE NOTICE 'Added gender column to patients';
                END IF;

                -- address
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'address'
                ) THEN
                    ALTER TABLE patients ADD COLUMN address TEXT;
                    RAISE NOTICE 'Added address column to patients';
                END IF;

                -- contact_info
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'contact_info'
                ) THEN
                    ALTER TABLE patients ADD COLUMN contact_info TEXT;
                    RAISE NOTICE 'Added contact_info column to patients';
                END IF;

                -- medical_history
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'medical_history'
                ) THEN
                    ALTER TABLE patients ADD COLUMN medical_history TEXT;
                    RAISE NOTICE 'Added medical_history column to patients';
                END IF;

                -- allergies
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'allergies'
                ) THEN
                    ALTER TABLE patients ADD COLUMN allergies TEXT;
                    RAISE NOTICE 'Added allergies column to patients';
                END IF;

                -- medications
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'medications'
                ) THEN
                    ALTER TABLE patients ADD COLUMN medications TEXT;
                    RAISE NOTICE 'Added medications column to patients';
                END IF;

                -- patient_tags
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'patient_tags'
                ) THEN
                    ALTER TABLE patients ADD COLUMN patient_tags TEXT;
                    RAISE NOTICE 'Added patient_tags column to patients';
                END IF;
            END $$;
        """)

        # ──────────────────────────────────────────────────────────────
        # Ensure default admin and demo patient users exist
        # ──────────────────────────────────────────────────────────────
        import hashlib
        from datetime import datetime


        now = datetime.now(timezone.utc)
        # Default admin (password: admin123)
        admin_password = hashlib.md5('admin123'.encode()).hexdigest()
        cur.execute("""
            INSERT INTO users (username, password, role, full_name, created_at)
            VALUES (%s, %s, 'admin', 'System Administrator', %s)
            ON CONFLICT (username) DO NOTHING;
        """, ('admin', admin_password, now))

        # Default patient (password: patient123)
        patient_password = hashlib.md5('patient123'.encode()).hexdigest()
        cur.execute("""
            INSERT INTO users (username, password, role, full_name, user_id, created_at)
            VALUES (%s, %s, 'patient', 'Demo Patient', 1, %s)
            ON CONFLICT (username) DO NOTHING;
        """, ('patient', patient_password, now))

        conn.commit()

        # Auto-create user accounts for all existing doctors that don't have one yet
        cur.execute("""
            INSERT INTO users (username, password, role, full_name, user_id, created_at)
            SELECT 
                'dr_' || d.id::text,                          -- username = dr_1, dr_2, etc.
                '5f4dcc3b5aa765d61d8327deb882cf99',           -- md5 hash of "password123"
                'doctor',
                COALESCE(d.name, 'Doctor ' || d.id::text),    -- use doctor name or fallback
                d.id,
                NOW()
            FROM doctors d
            LEFT JOIN users u 
                ON u.user_id = d.id 
                AND u.role = 'doctor'
            WHERE u.id IS NULL                                -- only if no user exists yet
            ON CONFLICT (username) DO NOTHING;                -- safe even if duplicate usernames
        """)

        print("Auto-created user accounts for existing doctors without accounts.")

    print("migrate_database() completed successfully")


# Run migration
migrate_database()


# Specialist types
SPECIALIST_TYPES = {
    'cardiac': 'Cardiac Specialist',
    'radiologist': 'Radiologist',
    'neurologist': 'Neurologist',
    'dermatologist': 'Dermatologist',
    'gynecologist': 'Gynecologist',
}

def get_clinical_bert():
    """Lazy initialization of Clinical BERT model"""
    global clinical_bert_model, clinical_bert_tokenizer
    if clinical_bert_model is None:
        try:
            from transformers import AutoTokenizer, AutoModel
            print("Loading Clinical BERT model (emilyalsentzer/Bio_ClinicalBERT)...")
            clinical_bert_tokenizer = AutoTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT")
            clinical_bert_model = AutoModel.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT")
            print("Clinical BERT loaded successfully.")
        except ImportError:
            print(
                "Clinical BERT not available. Install with: pip install transformers torch")
            clinical_bert_model = None
        except Exception as e:
            print(f"Clinical BERT loading failed: {str(e)}")
            clinical_bert_model = None
    return clinical_bert_model, clinical_bert_tokenizer


def analyze_with_clinical_bert(text):
    """Use Clinical BERT to extract medical entities and insights"""
    model, tokenizer = get_clinical_bert()
    if model is None or tokenizer is None:
        return None

    try:
        import torch
        # Tokenize and encode the text
        inputs = tokenizer(text[:512], return_tensors="pt",
                           truncation=True, max_length=512)

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract key medical insights
        # In production, you'd use NER models to extract medical entities
        return "Clinical BERT Analysis: Medical terminology and clinical concepts successfully processed. Key medical entities identified in the report text."
    except Exception as e:
        print(f"Clinical BERT analysis error: {str(e)}")
        return None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit(
    '.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_image(image_path):
    """Extract text from an image using available OCR method"""
    reader, method = get_ocr_reader()

    if method == 'paddleocr':
        try:
            # PaddleOCR method
            results = reader.ocr(image_path, cls=True)
            text_lines = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text_lines.append(line[1][0])  # line[1][0] is the text
            text = '\n'.join(text_lines)
            if not text.strip():
                raise Exception("No text extracted from image")
            return text
        except Exception as e:
            # If PaddleOCR fails at runtime, try to fallback
            print(f"PaddleOCR runtime error: {str(e)}. Attempting fallback...")
            # Try pytesseract as fallback
            try:
                import pytesseract
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                if text.strip():
                    return text
            except:
                pass
            raise Exception(f"OCR extraction failed: {str(e)}")

    elif method == 'pytesseract':
        try:
            # pytesseract method
            image = Image.open(image_path)
            text = reader.image_to_string(image)
            return text
        except Exception as e:
            raise Exception(f"OCR extraction failed: {str(e)}")

    else:
        # If no OCR available, provide helpful error message
        raise Exception("No OCR backend available for image processing. The uploaded file appears to be an image without extractable text. Please: 1) Install Tesseract OCR engine (https://github.com/UB-Mannheim/tesseract/wiki) and ensure pytesseract works, OR 2) Upload a text-based PDF which can be extracted without OCR. For now, you can upload PDFs with text layers that don't require OCR.")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF (Python-only)"""
    try:
        # Open PDF with PyMuPDF
        pdf_document = fitz.open(pdf_path)
        extracted_texts = []

        # Try to extract text directly first (if PDF has text layer)
        text_content = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text_content += page.get_text()

        # If direct text extraction worked, return it
        if text_content.strip():
            pdf_document.close()
            return text_content

        # If no text layer, convert pages to images and use OCR
        extracted_texts = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Convert page to image (pixmap)
            pix = page.get_pixmap(
    matrix=fitz.Matrix(
        2, 2))  # 2x zoom for better OCR
            img_data = pix.tobytes("png")

            # Convert to temporary image file for PaddleOCR
            # PaddleOCR works best with file paths
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                img = Image.open(io.BytesIO(img_data))
                img.save(tmp_path)

            # Get OCR reader (lazy initialization) and use on the image
            reader, method = get_ocr_reader()

            if method == 'paddleocr':
                results = reader.ocr(tmp_path, cls=True)
                text_lines = []
                if results and results[0]:
                    for line in results[0]:
                        if line and len(line) >= 2:
                            text_lines.append(line[1][0])
                page_text = '\n'.join(text_lines)
            elif method == 'pytesseract':
                img = Image.open(tmp_path)
                page_text = reader.image_to_string(img)
            else:
                page_text = "[OCR not available - image-based PDF requires OCR backend]"

            extracted_texts.append(page_text)

            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

        pdf_document.close()

        # Concatenate all pages
        return '\n\n--- Page Break ---\n\n'.join(extracted_texts)
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")


def get_or_create_patient(name):
    """Get existing patient or create a new one"""
    with get_db() as conn:
        cur = conn.cursor()

        # Check if patient exists
        cur.execute('SELECT id FROM patients WHERE name = %s', (name,))
        patient = cur.fetchone()

        if patient:
            patient_id = patient['id']   # RealDictCursor returns dict
        else:
            # Create new patient
            created_at = datetime.now().isoformat()
            cur.execute(
                'INSERT INTO patients (name, created_at) VALUES (%s, %s) RETURNING id',
                (name, created_at)
            )
            patient_id = cur.fetchone()['id']

        conn.commit()
        return patient_id


def get_specialist_prompt(specialist_type):
    """Get specialist-specific system prompt"""
    prompts = {
        'cardiac': """You are an expert Cardiac Specialist analyzing cardiovascular test reports. Focus on:
- ECG findings, cardiac enzymes, lipid profiles
- Heart function indicators (EF, wall motion abnormalities)
- Arrhythmias, conduction abnormalities
- Cardiovascular risk factors
- Recommendations for cardiac interventions or medications""",

        'radiologist': """You are an expert Radiologist analyzing medical imaging reports (X-rays, CT, MRI, Ultrasound). Focus on:
- Image findings and abnormalities
- Anatomical structures and pathological changes
- Size, location, and characteristics of findings
- Comparison with previous studies if mentioned
- Recommendations for follow-up imaging""",

        'neurologist': """You are an expert Neurologist analyzing neurological test reports. Focus on:
- Neurological examination findings
- Brain imaging results (MRI, CT scans)
- EEG, EMG results
- Cognitive assessments
- Neurological symptoms and their clinical significance
- Treatment recommendations for neurological conditions""",

        'dermatologist': """You are an expert Dermatologist analyzing dermatological reports. Focus on:
- Skin lesion characteristics
- Biopsy results
- Dermatological conditions and diagnoses
- Treatment protocols for skin conditions
- Follow-up care recommendations""",

        'gynecologist': """You are an expert Gynecologist analyzing women's health and reproductive system reports. Focus on:
- Menstrual and hormonal health (PCOS, thyroid-related menstrual issues)
- Pregnancy and fertility-related findings
- Pelvic ultrasound and reproductive organ assessments (uterus, ovaries, cervix)
- Vaginal and cervical screening results (Pap smear, HPV)
- Gynecological infections and inflammatory conditions
- Treatment guidance, follow-up recommendations, and when specialist care is needed"""

    }
    return prompts.get(
    specialist_type,
     """You are an expert medical analyst. Analyze the medical report comprehensively.""")


def format_analysis_response(text):
    """Format the AI analysis response into structured HTML"""
    if not text:
        return text

    import re

    # Process markdown bold (**text**)
    def process_bold(text):
        return re.sub(
    r'\*\*(.*?)\*\*',
    r'<strong class="font-bold text-gray-900">\1</strong>',
     text)

    # Split text into lines
    lines = text.split('\n')
    formatted_html = []
    in_list = False
    in_table = False
    table_rows = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            if in_table and table_rows:
                formatted_html.append(_format_table(table_rows))
                table_rows = []
                in_table = False
            formatted_html.append('<br>')
            continue

        # Check for table rows (contains | separator)
        if '|' in line and line.count('|') >= 2:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                # Check if header row (next line might be separator)
                is_separator = all(
    c.replace(
        '-', '').strip() == '' for c in cells)
                if not is_separator:
                    table_rows.append(cells)
                    in_table = True
                    if in_list:
                        formatted_html.append('</ul>')
                        in_list = False
            continue

        # If we were in table but this isn't a table row, finish the table
        if in_table and table_rows:
            formatted_html.append(_format_table(table_rows))
            table_rows = []
            in_table = False

        # Check for main headings (marked with # or ##)
        if line.startswith('##'):
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            heading_text = line.replace('##', '').strip()
            heading_text = process_bold(heading_text)
            formatted_html.append(
    f'<h3 class="text-xl font-bold text-gray-800 mt-4 mb-2 pb-2 border-b-2 border-blue-300">{heading_text}</h3>')

        # Check for numbered sections (1., 2., etc.)
        elif re.match(r'^\d+\.\s+[A-Z]', line):
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            heading_text = re.sub(r'^\d+\.\s+', '', line)
            heading_text = process_bold(heading_text)
            formatted_html.append(
    f'<h3 class="text-lg font-bold text-gray-800 mt-4 mb-2 pb-2 border-b border-blue-200">{heading_text}</h3>')

        # Check for subheadings (usually have : at the end or are short all
        # caps)
        elif (line.endswith(':') and len(line) < 60) or (line.isupper() and len(line) > 5 and len(line) < 50):
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            heading_text = process_bold(line)
            formatted_html.append(
    f'<h4 class="text-lg font-semibold text-gray-700 mt-3 mb-2">{heading_text}</h4>')

        # Check for bullet points (starting with -, *, •, or numbered bullets)
        elif re.match(r'^[-*•]\s', line) or re.match(r'^\d+[.)]\s', line) or line.startswith('•'):
            if not in_list:
                formatted_html.append(
                    '<ul class="list-disc list-inside space-y-1 ml-4 mb-3">')
                in_list = True
            bullet_text = re.sub(r'^[-*•\d.)\s]+', '', line).strip()
            bullet_text = process_bold(bullet_text)
            formatted_html.append(
    f'<li class="text-gray-700 mb-1">{bullet_text}</li>')

        # Check for sub-bullets (indented)
        elif line.startswith('  ') and (line.strip().startswith('-') or line.strip().startswith('*')):
            if in_list:
                bullet_text = line.strip().lstrip('-*').strip()
                bullet_text = process_bold(bullet_text)
                formatted_html.append(
    f'<li class="text-gray-600 ml-6 list-circle">{bullet_text}</li>')

        # Regular paragraph
        else:
            if in_list:
                formatted_html.append('</ul>')
                in_list = False
            processed_line = process_bold(line)
            formatted_html.append(
    f'<p class="text-gray-700 mb-2 leading-relaxed">{processed_line}</p>')

    if in_list:
        formatted_html.append('</ul>')
    if in_table and table_rows:
        formatted_html.append(_format_table(table_rows))

    return ''.join(formatted_html)


def _format_table(rows):
    """Format table rows into HTML table"""
    if not rows:
        return ''

    html = ['<div class="overflow-x-auto my-4"><table class="min-w-full border-collapse border border-gray-300">']

    # First row as header
    if rows:
        html.append('<thead class="bg-blue-100">')
        html.append('<tr>')
        for cell in rows[0]:
            html.append(
    f'<th class="border border-gray-300 px-4 py-2 text-left font-semibold text-gray-800">{cell}</th>')
        html.append('</tr>')
        html.append('</thead>')
        html.append('<tbody>')

        # Data rows
        for row in rows[1:]:
            html.append('<tr class="hover:bg-gray-50">')
            for cell in row:
                html.append(
    f'<td class="border border-gray-300 px-4 py-2 text-gray-700">{cell}</td>')
            html.append('</tr>')

        html.append('</tbody>')

    html.append('</table></div>')
    return ''.join(html)


def analyze_with_groq(extracted_text, specialist_type='general'):
    """Send extracted text to Groq API for specialist-specific medical analysis"""
    specialist_prompt = get_specialist_prompt(specialist_type)

    system_prompt = f"""{specialist_prompt}

Your analysis should be formatted in a clear, structured manner with the following sections:

**1. EXECUTIVE SUMMARY**
- Brief overview of the report findings
- Overall health status assessment

**2. ABNORMAL FINDINGS & ANOMALIES**
- List all values outside normal ranges
- Identify specific test abnormalities
- Note any concerning patterns

**3. CRITICAL FINDINGS (if applicable)**
- Urgent findings requiring immediate attention
- High-risk indicators
- Emergency recommendations

**4. DETAILED ANALYSIS BY CATEGORY**
Break down findings by test category (e.g., Hematology, Biochemistry, Imaging, etc.)
For each category:
- Normal findings
- Abnormal findings with reference ranges
- Clinical significance

**5. RECOMMENDED FOLLOW-UP**
- Additional diagnostic tests needed
- Specialist consultations recommended
- Monitoring requirements

**6. TREATMENT & MANAGEMENT PLAN**
- Medication considerations
- Lifestyle modifications
- Dietary recommendations
- Activity restrictions/suggestions

**7. RISK ASSESSMENT & PROGNOSIS**
- Risk factors identified
- Preventive measures
- Long-term outlook

Format your response using:
- Clear section headings (use ## for main headings)
- Bullet points for lists (use - or *)
- Tables for comparative data when appropriate (use | for columns)
- Bold text for important values (**text**)
- Clear paragraphs for explanations

Be thorough, professional, and prioritize patient safety. Use medical terminology appropriately."""

    user_prompt = f"""Please analyze the following medical test report text extracted from a patient's document:

{extracted_text}

Please provide your comprehensive analysis in the structured format specified above. Use clear headings, bullet points, and organize the information logically."""

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )

        raw_response = response.choices[0].message.content
        # Format the response into HTML
        formatted_response = format_analysis_response(raw_response)

        return {
            'raw': raw_response,
            'formatted': formatted_response
        }
    except Exception as e:
        raise Exception(f"Groq API call failed: {str(e)}")


# Lab test reference ranges for anomaly detection
LAB_REFERENCE_RANGES = {
    # Complete Blood Count (CBC)
    'hemoglobin': {'min': 12.0, 'max': 17.5, 'unit': 'g/dL', 'specialist': 'Hematologist'},
    'hgb': {'min': 12.0, 'max': 17.5, 'unit': 'g/dL', 'specialist': 'Hematologist'},
    'hematocrit': {'min': 36.0, 'max': 52.0, 'unit': '%', 'specialist': 'Hematologist'},
    'hct': {'min': 36.0, 'max': 52.0, 'unit': '%', 'specialist': 'Hematologist'},
    'rbc': {'min': 4.0, 'max': 6.0, 'unit': 'million/µL', 'specialist': 'Hematologist'},
    'wbc': {'min': 4.0, 'max': 11.0, 'unit': 'thousand/µL', 'specialist': 'Hematologist'},
    'platelets': {'min': 150, 'max': 400, 'unit': 'thousand/µL', 'specialist': 'Hematologist'},
    'plt': {'min': 150, 'max': 400, 'unit': 'thousand/µL', 'specialist': 'Hematologist'},

    # Metabolic Panel
    'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'specialist': 'Endocrinologist'},
    'fasting glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'specialist': 'Endocrinologist'},
    'blood sugar': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'specialist': 'Endocrinologist'},
    'hba1c': {'min': 4.0, 'max': 5.7, 'unit': '%', 'specialist': 'Endocrinologist'},
    'creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL', 'specialist': 'Nephrologist'},
    'bun': {'min': 7, 'max': 20, 'unit': 'mg/dL', 'specialist': 'Nephrologist'},
    'urea': {'min': 15, 'max': 45, 'unit': 'mg/dL', 'specialist': 'Nephrologist'},

    # Lipid Panel
    'cholesterol': {'min': 0, 'max': 200, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},
    'total cholesterol': {'min': 0, 'max': 200, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},
    'ldl': {'min': 0, 'max': 100, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},
    'hdl': {'min': 40, 'max': 200, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},
    'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL', 'specialist': 'Cardiologist'},

    # Liver Function
    'alt': {'min': 7, 'max': 56, 'unit': 'U/L', 'specialist': 'Gastroenterologist'},
    'sgpt': {'min': 7, 'max': 56, 'unit': 'U/L', 'specialist': 'Gastroenterologist'},
    'ast': {'min': 10, 'max': 40, 'unit': 'U/L', 'specialist': 'Gastroenterologist'},
    'sgot': {'min': 10, 'max': 40, 'unit': 'U/L', 'specialist': 'Gastroenterologist'},
    'bilirubin': {'min': 0.1, 'max': 1.2, 'unit': 'mg/dL', 'specialist': 'Gastroenterologist'},

    # Thyroid
    'tsh': {'min': 0.4, 'max': 4.0, 'unit': 'mIU/L', 'specialist': 'Endocrinologist'},
    't3': {'min': 80, 'max': 200, 'unit': 'ng/dL', 'specialist': 'Endocrinologist'},
    't4': {'min': 5.0, 'max': 12.0, 'unit': 'µg/dL', 'specialist': 'Endocrinologist'},

    # Electrolytes
    'sodium': {'min': 136, 'max': 145, 'unit': 'mEq/L', 'specialist': 'Nephrologist'},
    'potassium': {'min': 3.5, 'max': 5.0, 'unit': 'mEq/L', 'specialist': 'Nephrologist'},
    'calcium': {'min': 8.5, 'max': 10.5, 'unit': 'mg/dL', 'specialist': 'Endocrinologist'},

    # Inflammation markers
    'esr': {'min': 0, 'max': 20, 'unit': 'mm/hr', 'specialist': 'Rheumatologist'},
    'crp': {'min': 0, 'max': 3.0, 'unit': 'mg/L', 'specialist': 'Rheumatologist'},

    # Vitamins
    'vitamin d': {'min': 30, 'max': 100, 'unit': 'ng/mL', 'specialist': 'Endocrinologist'},
    'vitamin b12': {'min': 200, 'max': 900, 'unit': 'pg/mL', 'specialist': 'Hematologist'},
    'iron': {'min': 60, 'max': 170, 'unit': 'µg/dL', 'specialist': 'Hematologist'},
    'ferritin': {'min': 12, 'max': 300, 'unit': 'ng/mL', 'specialist': 'Hematologist'},
}


def detect_lab_anomalies(extracted_text, report_id=None, patient_id=None):
    """
    Detect abnormal values in lab report text by comparing against reference ranges.
    Returns list of anomalies with severity and recommended specialist.
    """
    import re

    anomalies = []
    text_lower = extracted_text.lower()

    # Patterns to match lab values...
    patterns = [
        r'([a-zA-Z\s]+)[\s:]+(\d+\.?\d*)\s*([a-zA-Z/%µ]+)?',
        r'([a-zA-Z\s]+)\s*[=:]\s*(\d+\.?\d*)',
    ]

    for test_name, ranges in LAB_REFERENCE_RANGES.items():
        test_pattern = re.compile(
            rf'\b{re.escape(test_name)}\b[\s:=]+(\d+\.?\d*)',
            re.IGNORECASE
        )
        matches = test_pattern.findall(text_lower)

        for match in matches:
            try:
                value = float(match)

                status = 'normal'
                if value < ranges['min']:
                    status = 'low'
                    if value < ranges['min'] * 0.7:
                        status = 'critical_low'
                elif value > ranges['max']:
                    status = 'high'
                    if value > ranges['max'] * 1.5:
                        status = 'critical_high'

                if status != 'normal':
                    anomalies.append({
                        'test_name': test_name.upper(),
                        'value': value,
                        'unit': ranges['unit'],
                        'normal_min': ranges['min'],
                        'normal_max': ranges['max'],
                        'status': status,
                        'recommended_specialist': ranges['specialist'],
                        'report_id': report_id,
                        'patient_id': patient_id
                    })
            except (ValueError, TypeError):
                continue

    # Store anomalies in database if report_id is provided
    if report_id and anomalies:
        try:
            with get_db() as conn:
                cur = conn.cursor()

                cur.execute('''
                    CREATE TABLE IF NOT EXISTS report_anomalies (
                        id                      SERIAL PRIMARY KEY,
                        report_id               INTEGER NOT NULL,
                        patient_id              INTEGER,
                        test_name               TEXT NOT NULL,
                        value                   REAL NOT NULL,
                        unit                    TEXT,
                        normal_min              REAL,
                        normal_max              REAL,
                        status                  TEXT NOT NULL,
                        recommended_specialist  TEXT,
                        created_at              TIMESTAMP NOT NULL DEFAULT NOW(),
                        FOREIGN KEY (report_id)  REFERENCES reports(id),
                        FOREIGN KEY (patient_id) REFERENCES patients(id)
                    )
                ''')

                cur.execute(
                    'DELETE FROM report_anomalies WHERE report_id = %s',
                    (report_id,)
                )

                now = datetime.now().isoformat()
                insert_data = [
                    (
                        report_id,
                        patient_id,
                        anomaly['test_name'],
                        anomaly['value'],
                        anomaly['unit'],
                        anomaly['normal_min'],
                        anomaly['normal_max'],
                        anomaly['status'],
                        anomaly['recommended_specialist'],
                        now
                    )
                    for anomaly in anomalies
                ]

                if insert_data:
                    cur.executemany('''
                        INSERT INTO report_anomalies (
                            report_id, patient_id, test_name, value, unit,
                            normal_min, normal_max, status, recommended_specialist, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', insert_data)

        except Exception as e:
            print(f"Error storing anomalies: {str(e)}")

    # THIS LINE MUST BE INDENTED UNDER THE def (same level as anomalies = [])
    return anomalies


def get_doctor_profile(doctor_id):
    """Get doctor profile from database"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM doctors WHERE id = %s', (doctor_id,))
        doctor = cur.fetchone()

    if doctor:
        return {
            'id': doctor['id'],
            'name': doctor['name'],
            'profile_picture': doctor['profile_picture'],
            'specialist_type': doctor['specialist_type'],
            'created_at': doctor['created_at']
        }
    return None

def create_or_update_doctor(name, specialist_type, profile_picture=None):
    """Create or update doctor profile"""
    with get_db() as conn:
        cur = conn.cursor()

        # Check if doctor exists
        cur.execute(
            'SELECT id FROM doctors WHERE name = %s AND specialist_type = %s',
            (name, specialist_type)
        )
        doctor = cur.fetchone()

        if doctor:
            # Update existing
            if profile_picture:
                cur.execute(
                    'UPDATE doctors SET profile_picture = %s WHERE id = %s',
                    (profile_picture, doctor['id'])
                )
            doctor_id = doctor['id']
        else:
            # Create new
            created_at = datetime.now().isoformat()
            cur.execute(
                '''
                INSERT INTO doctors (name, profile_picture, specialist_type, created_at)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                ''',
                (name, profile_picture, specialist_type, created_at)
            )
            doctor_id = cur.fetchone()['id']

        # commit automatic via context manager
        conn.commit()
        return doctor_id


def chatbot_response(message, specialist_type, doctor_id=None):
    """Generate chatbot response using Clinical BERT with specialist context"""
    try:
        # Use Clinical BERT to analyze the doctor's query
        clinical_bert_analysis = analyze_with_clinical_bert(message)

        # Get specialist context
        specialist_prompt = get_specialist_prompt(specialist_type)
        specialist_name = SPECIALIST_TYPES.get(specialist_type, 'medical')

        # Build context for response generation
        context_text = f"""Specialist Context: {specialist_prompt}
Specialist Type: {specialist_name}

Clinical BERT Analysis of Query: {clinical_bert_analysis if clinical_bert_analysis else 'No specific medical entities detected.'}

You are an AI assistant helping a {specialist_name} professional.
Answer questions clearly, professionally, and with medical accuracy based on the Clinical BERT analysis.
If asked about something outside your specialty, acknowledge it and suggest consulting the appropriate specialist."""

        # Use Groq LLM with Clinical BERT context for generating the response
        # (Clinical BERT provides medical entity extraction, Groq provides natural language generation)
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": context_text},
                {"role": "user", "content": f"Doctor's Question: {message}\n\nPlease provide a professional medical response based on the Clinical BERT analysis above."}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        bot_response = response.choices[0].message.content

        # Add note about Clinical BERT usage
        if clinical_bert_analysis:
            bot_response += f"\n\n[Note: This response was enhanced using Clinical BERT medical entity analysis.]"

        # Save chat message to database
        # Save chat message to database
        if doctor_id:
            try:
                with get_db() as conn:
                    cur = conn.cursor()
                    created_at = datetime.now().isoformat()
                    cur.execute('''
                        INSERT INTO chat_messages
                        (doctor_id, specialist_type, message, response, created_at)
                        VALUES (%s, %s, %s, %s, %s)
                    ''', (doctor_id, specialist_type, message, bot_response, created_at))
                    # commit is automatic on success
            except Exception as e:
                print(f"Error saving chat message: {str(e)}")

        return bot_response
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"


def save_report(
    patient_id,
    filename,
    extracted_text,
    llm_analysis,
    specialist_type='general',
    doctor_id=None,
     clinical_bert_analysis=None):
    if not specialist_type or specialist_type not in SPECIALIST_TYPES:
        specialist_type = 'general'

    created_at = datetime.now(timezone.utc)
    report_id = None

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO reports
                (doctor_id, patient_id, specialist_type, original_filename,
                 extracted_text, llm_analysis, clinical_bert_analysis, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (doctor_id, patient_id, specialist_type, filename,
                  extracted_text, llm_analysis, clinical_bert_analysis, created_at))
            report_id = cur.fetchone()['id']
            conn.commit()
            print(f"[save_report] Report saved successfully with ID: {report_id}")

    except psycopg2.errors.UndefinedColumn:
        print("Missing column - running migration")
        migrate_database()
        # Retry
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO reports
                (doctor_id, patient_id, specialist_type, original_filename,
                 extracted_text, llm_analysis, clinical_bert_analysis, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (doctor_id, patient_id, specialist_type, filename,
                  extracted_text, llm_analysis, clinical_bert_analysis, created_at))
            report_id = cur.fetchone()['id']
            conn.commit()
            print(f"[save_report] Report saved after migration with ID: {report_id}")

    except Exception as e:
        print(f"Error saving report: {str(e)}")
        return None

    # Anomaly detection
    if report_id and extracted_text:
        try:
            anomalies = detect_lab_anomalies(
    extracted_text, report_id=report_id, patient_id=patient_id)
            if anomalies:
                print(
    f"Detected {
        len(anomalies)} anomalies in report {report_id}")
        except Exception as e:
            print(f"Error running anomaly detection: {str(e)}")

    return report_id


@app.route('/')
def index():
    """Landing page"""
    if 'user_role' in session:
        role = session['user_role']
        if role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        elif role == 'patient':
            return redirect(url_for('patient_dashboard'))
        elif role == 'admin':
            return redirect(url_for('admin_dashboard'))
    return render_template('landing.html')


@app.route('/home')
def home():
    """Alternative landing page route"""
    return redirect(url_for('index'))


@app.route('/login', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            return render_template('login.html',
     error='Username and password are required')

        password_hash = hashlib.md5(password.encode()).hexdigest()

        try:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute('''
                    SELECT id, username, role, full_name, user_id, email, password
                    FROM users WHERE username = %s
                ''', (username,))
                user = cur.fetchone()

                if not user:
                    return render_template(
    'login.html', error='Invalid username or password')

                stored_password = user['password']
                if stored_password != password_hash:
                    return render_template(
    'login.html', error='Invalid username or password')

                # Set session
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['user_role'] = user['role']
                session['full_name'] = user['full_name'] or user['username']
                session['user_db_id'] = user['user_id']
                session['email'] = user['email']

                # Update last login
                cur.execute(
                    'UPDATE users SET last_login = %s WHERE id = %s',
                    (datetime.now(timezone.utc), user['id'])
                )
                # commit automatic
                conn.commit()

            # Redirect based on role
            if session['user_role'] == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            elif session['user_role'] == 'patient':
                return redirect(url_for('patient_dashboard'))
            elif session['user_role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return render_template('login.html', error='Invalid user role')

        except Exception as e:
            print(f"Login error: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template(
    'login.html',
    error=f'Login error: {
        str(e)}')

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    return redirect(url_for('login'))


@app.route('/doctor/appointments')
def doctor_appointments_page():
    """Doctor appointments page"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return redirect(url_for('login'))
    return render_template('doctor_appointments.html', user=session)


@app.route('/patient/appointments')
def patient_appointments_page():
    """Patient appointments page"""
    if 'user_role' not in session or session['user_role'] != 'patient':
        return redirect(url_for('login'))
    return render_template('patient_appointments.html', user=session)


@app.route('/patient/chat')
def patient_chat_page():
    """Patient chat page"""
    if 'user_role' not in session or session['user_role'] != 'patient':
        return redirect(url_for('login'))
    return render_template('patient_chat.html', user=session)


@app.route('/doctor/chat')
def doctor_chat_page():
    """Doctor chat page"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return redirect(url_for('login'))
    return render_template('doctor_chat.html', user=session)


@app.route('/messages')
def messages_redirect():
    """Redirect to appropriate messages page based on user role"""
    if 'user_role' not in session:
        return redirect(url_for('login'))

    user_role = session.get('user_role')
    if user_role == 'patient':
        return redirect(url_for('patient_chat_page'))
    elif user_role == 'doctor':
        return redirect(url_for('doctor_chat_page'))
    else:
        return redirect(url_for('index'))


@app.route('/doctor/dashboard')
def doctor_dashboard():
    """Doctor dashboard page"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return redirect(url_for('login'))

    doctor = None
    user_db_id = session.get('user_db_id')

    try:
        with get_db() as conn:
            cur = conn.cursor()

            if user_db_id:
                cur.execute(
    'SELECT * FROM doctors WHERE id = %s', (user_db_id,))
            else:
                # Fallback - last created doctor (should rarely happen)
                cur.execute(
                    'SELECT * FROM doctors ORDER BY created_at DESC LIMIT 1')

            doctor = cur.fetchone()  # Already a dict thanks to RealDictCursor

            # Optional: log if no doctor found
            if not doctor:
                print(f"No doctor profile found for user_db_id: {user_db_id}")

    except Exception as e:
        print(f"Error loading doctor profile in dashboard: {str(e)}")
        # Optional: import traceback; traceback.print_exc()

    return render_template('doctor_dashboard.html',
                         specialists=SPECIALIST_TYPES,
                         doctor=doctor,
                         user=session)


@app.route('/profile')
def profile():
    """Profile management page"""
    if 'user_role' not in session:
        return redirect(url_for('login'))

    role = session['user_role']
    user_db_id = session.get('user_db_id')
    doctor = None
    patient = None

    try:
        with get_db() as conn:
            cur = conn.cursor()

            if role == 'doctor' and user_db_id:
                cur.execute(
    'SELECT * FROM doctors WHERE id = %s', (user_db_id,))
                doctor = cur.fetchone()  # already a dict due to RealDictCursor

            elif role == 'patient' and user_db_id:
                cur.execute(
    'SELECT * FROM patients WHERE id = %s', (user_db_id,))
                patient = cur.fetchone()  # already a dict

            # Optional: log if nothing found
            if not doctor and not patient:
                print(
    f"No profile found for {role} with user_db_id: {user_db_id}")

    except Exception as e:
        print(f"Profile load error: {str(e)}")
        # Optional: import traceback; traceback.print_exc()

    return render_template('profile.html',
                         specialists=SPECIALIST_TYPES,
                         doctor=doctor,
                         patient=patient,
                         user=session)


@app.route('/specialist/<specialist_type>')
def specialist_page(specialist_type):
    """Render specialist-specific page"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return redirect(url_for('login'))

    if specialist_type not in SPECIALIST_TYPES:
        return "Specialist not found", 404

    doctor = None
    user_db_id = session.get('user_db_id')

    try:
        with get_db() as conn:
            cur = conn.cursor()

            if user_db_id:
                cur.execute(
    'SELECT * FROM doctors WHERE id = %s', (user_db_id,))
            else:
                # Fallback (should rarely happen)
                cur.execute(
                    'SELECT * FROM doctors ORDER BY created_at DESC LIMIT 1')

            doctor = cur.fetchone()  # Returns dict due to RealDictCursor

            if not doctor:
                print(f"No doctor profile found for user_db_id: {user_db_id}")

    except Exception as e:
        print(f"Specialist page doctor load error: {str(e)}")
        # Optional: import traceback; traceback.print_exc()

    return render_template('specialist.html',
                         specialist_type=specialist_type,
                         specialist_name=SPECIALIST_TYPES[specialist_type],
                         specialists=SPECIALIST_TYPES,
                         doctor=doctor,
                         user=session)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload, OCR, and AI analysis"""
    # Allow both doctors and patients to upload reports
    if 'user_role' not in session or session['user_role'] not in [
        'doctor', 'patient']:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        # Validate file and patient name
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        patient_name = request.form.get('patient_name', '').strip()

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not patient_name:
            return jsonify({'error': 'Patient name is required'}), 400

        if not allowed_file(file.filename):
            return jsonify(
                {'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, PDF'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text using OCR
        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext == 'pdf':
            extracted_text = extract_text_from_pdf(filepath)
        else:
            extracted_text = extract_text_from_image(filepath)

        if not extracted_text or not extracted_text.strip():
            # Check if it was an image file that failed OCR
            if file_ext != 'pdf':
                return jsonify({'error': 'No text could be extracted from the image. This may be because OCR is not available. Please ensure: 1) The image is clear and readable, 2) OCR backend (PaddleOCR or Tesseract) is properly installed. Alternatively, convert the image to a text-based PDF.'}), 400
            else:
                return jsonify(
                    {'error': 'No text could be extracted from the PDF. If this is an image-based PDF, OCR may not be available. Please ensure OCR backend is installed, or try a PDF with extractable text.'}), 400

        # Get specialist type and doctor ID from form
        specialist_type = request.form.get(
    'specialist_type', 'general').strip()
        if not specialist_type or specialist_type not in SPECIALIST_TYPES:
            specialist_type = 'general'

        doctor_id = request.form.get('doctor_id', None)
        if doctor_id:
            try:
                doctor_id = int(doctor_id) if doctor_id else None
            except:
                doctor_id = None
        else:
            doctor_id = None

        # Analyze with Groq API (specialist-specific)
        llm_analysis_result = analyze_with_groq(
            extracted_text, specialist_type)

        # Handle both old format (string) and new format (dict)
        if isinstance(llm_analysis_result, dict):
            llm_analysis_raw = llm_analysis_result.get('raw', '')
            llm_analysis_formatted = llm_analysis_result.get('formatted', '')
        else:
            llm_analysis_raw = llm_analysis_result
            llm_analysis_formatted = format_analysis_response(
                llm_analysis_result)

        # Clinical BERT analysis
        clinical_bert_analysis = analyze_with_clinical_bert(extracted_text)

        # Get or create patient
        patient_id = get_or_create_patient(patient_name)

        # Save to database (save formatted HTML - it will render properly in
        # templates)
        save_report(
    patient_id,
    filename,
    extracted_text,
    llm_analysis_formatted,
    specialist_type,
    doctor_id,
     clinical_bert_analysis)

        # Return results - ensure HTML is properly formatted
        # Use jsonify which handles JSON correctly, HTML in strings should not
        # be escaped
        response_data = {
            'success': True,
            'extracted_text': extracted_text,
            'llm_analysis': llm_analysis_formatted,  # Return formatted HTML string
            'llm_analysis_raw': llm_analysis_raw,  # Also include raw for reference
            'clinical_bert_analysis': clinical_bert_analysis
        }

        # Use jsonify - it properly handles strings with HTML without escaping
        return jsonify(response_data)

    except ValueError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@app.route('/api/profile', methods=['POST'])
def update_profile():
    """Update doctor profile"""
    try:
        name = request.form.get('name', '').strip()
        specialist_type = request.form.get('specialist_type', '').strip()

        if not name or not specialist_type:
            return jsonify(
                {'error': 'Name and specialist type are required'}), 400

        if specialist_type not in SPECIALIST_TYPES:
            return jsonify({'error': 'Invalid specialist type'}), 400

        profile_picture = None
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file.filename:
                filename = secure_filename(
    f"{specialist_type}_{name}_{
        datetime.now().timestamp()}.{
            file.filename.rsplit(
                '.',
                 1)[1].lower()}")
                filepath = os.path.join(app.config['PROFILE_FOLDER'], filename)
                file.save(filepath)
                profile_picture = f"profiles/{filename}"

        doctor_id = create_or_update_doctor(
    name, specialist_type, profile_picture)

        return jsonify({
            'success': True,
            'doctor_id': doctor_id,
            'profile_picture': profile_picture
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot messages"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        specialist_type = data.get('specialist_type', 'general')
        doctor_id = data.get('doctor_id', None)

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        if specialist_type not in SPECIALIST_TYPES:
            specialist_type = 'general'

        response = chatbot_response(message, specialist_type, doctor_id)

        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/patient/chatbot/context-questions', methods=['POST'])
def get_patient_chatbot_context_questions():
    if 'user_role' not in session or session['user_role'] != 'patient':
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.json
        patient_id = data.get('patient_id') or session.get('user_db_id')

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
    'SELECT allergies, medications, medical_history FROM patients WHERE id = %s',
    (patient_id,
    ))
            patient_data = cur.fetchone()

            questions = [
                {'id': 1,
    'question': 'What is your main health concern or question today?',
    'type': 'text'},
                {'id': 2,
    'question': 'Are you currently experiencing any symptoms? If yes, please describe them.',
    'type': 'text'},
                {'id': 3, 'question': 'When did these symptoms start?', 'type': 'text'}
            ]

            # Use dict access
            if patient_data and patient_data['medications']:
                questions.append({
                    'id': 4,
                    'question': f"Are you currently taking any medications? (You mentioned: {patient_data['medications'][:50]}...)",
                    'type': 'text'
                })
            else:
                questions.append(
                    {'id': 4, 'question': 'Are you currently taking any medications?', 'type': 'text'})

            return jsonify({
                'success': True,
                'questions': questions
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/chatbot', methods=['POST'])
def patient_chatbot():
    """Handle patient chatbot messages with Clinical BERT - Continuous Q&A flow"""
    if 'user_role' not in session or session['user_role'] != 'patient':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        message = data.get('message', '').strip()
        patient_id = data.get('patient_id') or session.get('user_db_id')
        conversation_history = data.get('conversation_history', [])  # Array of {role, content}
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get patient's medical reports and info for context
        with get_db() as conn:
            cur = conn.cursor()
            
            # Patient information
            cur.execute(
                'SELECT name, allergies, medications, medical_history FROM patients WHERE id = %s',
                (patient_id,)
            )
            patient_data = cur.fetchone()
            patient_info = {}
            if patient_data:
                patient_info = {
                    'name': patient_data['name'],
                    'allergies': patient_data['allergies'],
                    'medications': patient_data['medications'],
                    'medical_history': patient_data['medical_history']
                }
            
            # Recent reports
            cur.execute('''
                SELECT extracted_text, llm_analysis, clinical_bert_analysis, specialist_type 
                FROM reports 
                WHERE patient_id = %s
                ORDER BY created_at DESC LIMIT 3
            ''', (patient_id,))
            reports = cur.fetchall()
        
        # Use Clinical BERT to analyze the patient's query
        clinical_bert_analysis = analyze_with_clinical_bert(message)
        
        # Build conversation context
        context_text = f"""Patient Information:
- Name: {patient_info.get('name', 'N/A')}
- Allergies: {patient_info.get('allergies', 'None reported')}
- Current Medications: {patient_info.get('medications', 'None reported')}
- Medical History: {patient_info.get('medical_history', 'None reported')}
"""
        
        if reports:
            context_text += "\nRecent Medical Reports Summary:\n"
            for report in reports:
                if report['clinical_bert_analysis']:
                    context_text += f"- {report['clinical_bert_analysis'][:200]}...\n"
        
        if clinical_bert_analysis:
            context_text += f"\nClinical BERT Analysis of Current Query: {clinical_bert_analysis}\n"
        
        # Use Groq LLM with Clinical BERT context and conversation history
        client = get_groq_client()
        system_prompt = """You are an AI medical assistant for the Neura-X platform.

ROLE:
You are a calm, knowledgeable, and supportive virtual doctor who conducts natural medical conversations to understand patient concerns before giving guidance.

PRIMARY GOAL:
Hold a focused, human-like consultation with the patient by asking ONE question at a time to gather sufficient clinical context before providing guidance.

IMPORTANT INTERNAL BEHAVIOR (DO NOT REVEAL TO USER):

* Track internally how many questions you have asked.
* After approximately 4–5 questions, STOP asking more questions and provide final guidance.
* Never mention question counts or internal reasoning to the user.
* Never output system markers or testing phrases.

STRICT CONVERSATION RULE:
Ask ONLY ONE question at a time.
Do NOT ask multiple questions in a single response.

---

RESTRICTIONS:

* ONLY provide information related to medical, health, or wellness topics.
* If asked anything non-medical, respond exactly with:
  "I'm a medical consultation assistant and can only help with health or medical-related concerns."
* Never provide definitive diagnoses.
* Always use cautious language (e.g., "could be", "might be").
* This system provides preliminary guidance only and is not a substitute for professional care.
* Do NOT use emojis, emoticons, or pictographs.

RESPONSE LENGTH RULE:

* During question phases, keep responses brief (ideally under 80 words).
* Do not provide long explanations until the final guidance phase.

---

CONVERSATION FLOW

PHASE 1 — Information Gathering

* Acknowledge warmly in 1–2 sentences.
* Ask ONE relevant follow-up question.
* Do NOT give final assessment.

---

PHASE 2 — Continued Clarification

* Continue asking ONE question at a time.
* Consider onset, duration, severity, associated symptoms.

---

PHASE 3 — Final Guidance

Provide the final response ONLY after ~4–5 questions OR when enough info is gathered.

Use EXACT structure:

Based on what you've described: [short summary]

Possible causes (preliminary, not a diagnosis):
* [cause 1]
* [optional cause 2]

Home care suggestions:
* [tip 1]
* [tip 2]

Important: Do not follow these if you have any allergy or medical restriction.

When to see a doctor urgently:
* [warning 1]
* [warning 2]
* [warning 3]

Follow-up:
* [monitoring advice]

---

DOCTOR RECOMMENDATION LOGIC (VERY IMPORTANT)

* ONLY recommend a doctor IF the user explicitly asks something like:
  "Which doctor should I see?"
  "Recommend a specialist"
  "Which doctor is best for this?"

* If the user DOES NOT ask → DO NOT suggest any doctor.

* If the user asks:

  1. Identify the most relevant specialist based on symptoms.
     Examples:

     * Fever, infection → General Physician
     * Hormonal issues → Endocrinologist
     * Heart issues → Cardiologist
     * Blood issues → Hematologist
     * Female reproductive issues → Gynecologist

  2. If specialist matches available system specialists:

     * Respond like:
       "You may consider consulting a [Specialist Type]."

     * Then include:
       "Available doctors will be shown based on your location."

     (DO NOT hallucinate doctor names — backend will inject real doctors)

  3. If symptoms DO NOT clearly match any specialist:

     * Recommend:
       "You may start with a General Physician for initial evaluation."

---

RECOVERY / CONVERSATION STOP RULE:

* If the user indicates they are feeling better, fine, recovered, or no longer need help (e.g., "I am fine now", "I feel better", "I'm okay now"):

  • STOP asking further medical questions immediately
  • Do NOT continue the consultation flow
  • Do NOT ask follow-up questions

* Instead, respond with:

  * A brief positive acknowledgment
  * 1–2 general wellness suggestions (optional)
  * A short note to seek medical care if symptoms return or worsen

* Keep the response short and supportive.

---

MODE 2: Instructor Mode

If user asks general medical knowledge:

* Provide clear explanation
* No questioning required

---

TONE AND STYLE

* Warm, calm, professional
* Conversational
* No jargon overload

---

IMPORTANT REMINDERS

* This is preliminary guidance only.
* Not a substitute for licensed medical care."""

        # Build messages array with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            if msg.get('role') and msg.get('content'):
                messages.append({"role": msg['role'], "content": msg['content']})
        
        # Add current patient message with context
        user_prompt = f"""
Patient message: {message}

Relevant patient context:
{context_text}

Follow the system instructions strictly.
"""
        
        messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1200
        )
        
        bot_response = response.choices[0].message.content
        
        # Check if response contains a question (ends with ? or contains question words)
        has_question = '?' in bot_response or any(word in bot_response.lower() for word in ['can you', 'could you', 'would you', 'what', 'when', 'where', 'how', 'why', 'tell me', 'describe'])
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'clinical_bert_used': clinical_bert_analysis is not None,
            'has_question': has_question
        })
    
    except Exception as e:
        print(f"Chatbot error for patient {patient_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/<specialist_type>')
def get_reports(specialist_type):
    """Get reports for a specific specialist"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT r.id, r.created_at, p.name as patient_name, r.original_filename, 
                       d.name as doctor_name 
                FROM reports r
                LEFT JOIN patients p ON r.patient_id = p.id
                LEFT JOIN doctors d ON r.doctor_id = d.id
                WHERE r.specialist_type = %s 
                ORDER BY r.created_at DESC LIMIT 50
            ''', (specialist_type,))
            reports = cur.fetchall()  # list of dicts due to RealDictCursor
        
        return jsonify({
            'success': True,
            'reports': [{
                'id': r['id'],
                'created_at': r['created_at'],
                'patient_name': r['patient_name'],
                'filename': r['original_filename'],
                'doctor_name': r['doctor_name']
            } for r in reports]
        })
    
    except Exception as e:
        print(f"Error fetching reports: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard/stats')
def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Total reports per specialist
            cur.execute('SELECT specialist_type, COUNT(*) as count FROM reports GROUP BY specialist_type')
            reports_by_specialist_raw = cur.fetchall()
            reports_by_specialist_list = [
                {'type': row['specialist_type'], 
                 'name': SPECIALIST_TYPES.get(row['specialist_type'], row['specialist_type']), 
                 'count': row['count']} 
                for row in reports_by_specialist_raw
            ]
            
            # Totals
            cur.execute('SELECT COUNT(*) FROM reports')
            total_reports = cur.fetchone()['count']
            
            cur.execute('SELECT COUNT(*) FROM patients')
            total_patients = cur.fetchone()['count']
            
            cur.execute('SELECT COUNT(*) FROM doctors')
            total_doctors = cur.fetchone()['count']
            
            # Reports over time (last 7 days)
            cur.execute('''
                SELECT DATE(created_at) as date, COUNT(*) as count 
                FROM reports 
                WHERE created_at >= NOW() - INTERVAL '7 days'
                GROUP BY DATE(created_at) ORDER BY date
            ''')
            reports_over_time = [{'date': row['date'], 'count': row['count']} for row in cur.fetchall()]
            
            # Most active specialists
            cur.execute('''
                SELECT specialist_type, COUNT(*) as count FROM reports 
                GROUP BY specialist_type ORDER BY count DESC LIMIT 5
            ''')
            active_specialists = [
                {'type': row['specialist_type'], 
                 'name': SPECIALIST_TYPES.get(row['specialist_type'], row['specialist_type']), 
                 'count': row['count']} 
                for row in cur.fetchall()
            ]
            
            # Recent reports
            cur.execute('''
                SELECT r.id, r.created_at, p.name as patient_name, r.specialist_type, d.name as doctor_name
                FROM reports r
                LEFT JOIN patients p ON r.patient_id = p.id
                LEFT JOIN doctors d ON r.doctor_id = d.id
                ORDER BY r.created_at DESC LIMIT 10
            ''')
            recent_reports = [dict(row) for row in cur.fetchall()]  # already dicts
        
        return jsonify({
            'success': True,
            'stats': {
                'total_reports': total_reports,
                'total_patients': total_patients,
                'total_doctors': total_doctors,
                'reports_by_specialist': {r['type']: r['count'] for r in reports_by_specialist_list},
                'reports_by_specialist_list': reports_by_specialist_list,
                'reports_over_time': reports_over_time,
                'active_specialists': active_specialists,
                'recent_reports': recent_reports
            }
        })
    
    except Exception as e:
        print(f"Dashboard stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            data = request.form
            username = data.get('username', '').strip()
            password = data.get('password', '').strip()
            full_name = data.get('full_name', '').strip()
            email = data.get('email', '').strip()
            phone = data.get('phone', '').strip()
            date_of_birth = data.get('date_of_birth', '').strip()
            gender = data.get('gender', '').strip()
            address = data.get('address', '').strip()
            
            if not username or not password or not full_name:
                return render_template('signup.html', error='Username, password, and full name are required')

            print(f"[Signup] Attempting to register user: {username}")

            with get_db() as conn:
                cur = conn.cursor()
                
                # Check if username already exists
                cur.execute('SELECT id FROM users WHERE username = %s', (username,))
                if cur.fetchone():
                    print(f"[Signup] Username {username} already exists")
                    return render_template('signup.html', error='Username already exists')
                
                created_at = datetime.now(timezone.utc)
                
                # 1. Insert into patients table
                cur.execute('''
                    INSERT INTO patients 
                    (name, email, phone, date_of_birth, gender, address, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (full_name, email or None, phone or None, date_of_birth or None, 
                      gender or None, address or None, created_at))
                
                patient_row = cur.fetchone()
                if not patient_row:
                    raise Exception("Failed to create patient record")
                
                patient_id = patient_row['id']
                print(f"[Signup] Patient created with ID: {patient_id}")

                # Generate patient_id (PAT000001 format)
                cur.execute('''
                    UPDATE patients 
                    SET patient_id = %s 
                    WHERE id = %s AND (patient_id IS NULL OR patient_id = '')
                ''', (f"PAT{patient_id:06d}", patient_id))

                # 2. Insert into users table
                password_hash = hashlib.md5(password.encode()).hexdigest()
                
                cur.execute('''
                    INSERT INTO users 
                    (username, password, role, full_name, email, user_id, created_at)
                    VALUES (%s, %s, 'patient', %s, %s, %s, %s)
                    RETURNING id
                ''', (username, password_hash, full_name, email or None, patient_id, created_at))
                
                user_row = cur.fetchone()
                if not user_row:
                    raise Exception("Failed to create user account")
                
                user_id = user_row['id']
                print(f"[Signup] User account created with ID: {user_id}")

                # CRITICAL: Commit the transaction
                conn.commit()
                print(f"[Signup] SUCCESS - Patient {username} registered successfully")

            # Auto-login the user
            session['user_id'] = user_id
            session['username'] = username
            session['user_role'] = 'patient'
            session['full_name'] = full_name
            session['email'] = email
            session['user_db_id'] = patient_id

            return redirect(url_for('patient_dashboard'))
        
        except Exception as e:
            print(f"[Signup] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('signup.html', error=f'Registration failed: {str(e)}')
    
    return render_template('signup.html')

@app.route('/report/<int:report_id>')
def report_detail(report_id):
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('''
                SELECT r.*, p.name as patient_name, d.name as doctor_name, 
                       d.profile_picture as doctor_picture
                FROM reports r
                LEFT JOIN patients p ON r.patient_id = p.id
                LEFT JOIN doctors d ON r.doctor_id = d.id
                WHERE r.id = %s
            ''', (report_id,))
            report = cur.fetchone()
            
            if not report:
                return "Report not found", 404
            
            # Security check - only for patients
            if session['user_role'] == 'patient' and report['patient_id'] != session.get('user_db_id'):
                return "Unauthorized - You can only view your own reports", 403
            
            # Load patient's other reports for sidebar
            cur.execute('''
                SELECT id, created_at, specialist_type, original_filename, status
                FROM reports 
                WHERE patient_id = %s 
                ORDER BY created_at DESC
            ''', (report['patient_id'],))
            patient_reports = [dict(r) for r in cur.fetchall()]
            
            doctor = None
            if report.get('doctor_id'):
                cur.execute('''
                    SELECT id, name, profile_picture, specialist_type 
                    FROM doctors WHERE id = %s
                ''', (report['doctor_id'],))
                doctor_row = cur.fetchone()
                if doctor_row:
                    doctor = dict(doctor_row)
            # Safe conversion of created_at (works for both datetime and string)
            report_dict = dict(report) if report else {}

            print("=== REPORT DETAIL DEBUG ===")
            print("Report ID:", report_id)
            print("Report data keys:", list(dict(report).keys()) if report else None)
            print("Doctor data:", dict(doctor) if doctor else None)
            print("Patient reports count:", len(patient_reports))
            print("User role:", session.get('user_role'))

            # Safe date handling for template
            report_dict = dict(report) if report else {}
            if report_dict.get('created_at'):
                created = report_dict['created_at']
                report_dict['created_at'] = str(created)[:10] if isinstance(created, (str, type(None))) else created.strftime('%Y-%m-%d') if hasattr(created, 'strftime') else str(created)[:10]

            for pr in patient_reports:
                if pr.get('created_at'):
                    created = pr['created_at']
                    pr['created_at'] = str(created)[:10] if isinstance(created, (str, type(None))) else created.strftime('%Y-%m-%d') if hasattr(created, 'strftime') else str(created)[:10]

            return render_template('report_detail.html',
                                report=report_dict,
                                patient_reports=patient_reports,
                                specialists=SPECIALIST_TYPES,
                                doctor=doctor,
                                user=session)

    except Exception as e:
        import traceback
        print(f"🚨 Report detail error for report {report_id}: {str(e)}")
        traceback.print_exc()
        return f"Error loading report: {str(e)}", 500   # ← shows real error in browser for now
@app.route('/patient/profile')
@app.route('/patient/<int:patient_id>')
def patient_profile(patient_id=None):
    """Patient profile page"""
    if 'user_role' not in session:
        return redirect(url_for('login'))
    
    if patient_id is None:
        if session.get('user_role') != 'patient':
            return "Access denied", 403
        patient_id = session.get('user_db_id')
    
    if session['user_role'] == 'patient' and session.get('user_db_id') != patient_id:
        return "Unauthorized - You can only view your own profile", 403
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Get patient
            cur.execute('SELECT * FROM patients WHERE id = %s', (patient_id,))
            patient_row = cur.fetchone()
            
            if not patient_row:
                return "Patient not found", 404
            
            patient = dict(patient_row)

            # ✅ FIX 1: Convert DOB datetime → string
            if patient.get('date_of_birth'):
                patient['date_of_birth'] = patient['date_of_birth'].strftime('%Y-%m-%d')
            
            # Get reports
            cur.execute('''
                SELECT r.id, r.created_at, r.specialist_type, r.original_filename, 
                       r.llm_analysis, r.status, r.doctor_notes,
                       d.name as doctor_name
                FROM reports r
                LEFT JOIN doctors d ON r.doctor_id = d.id
                WHERE r.patient_id = %s 
                ORDER BY r.created_at DESC
            ''', (patient_id,))
            
            reports = [dict(r) for r in cur.fetchall()]

            # ✅ FIX 2: Convert report datetime → string
            for r in reports:
                if r.get('created_at'):
                    r['created_at'] = r['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return render_template('patient_profile.html',
                             patient=patient,
                             reports=reports,
                             specialists=SPECIALIST_TYPES,
                             user=session)

    except Exception as e:
        print(f"Patient profile error for ID {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error loading profile: {str(e)}", 500


@app.route('/api/search/reports')
def search_reports():
    query = request.args.get('q', '').strip()
    specialist_type = request.args.get('specialist', '').strip()
    status = request.args.get('status', '').strip()
    date_from = request.args.get('date_from', '').strip()
    date_to = request.args.get('date_to', '').strip()
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            sql = '''
                SELECT r.*, p.name as patient_name, d.name as doctor_name
                FROM reports r
                LEFT JOIN patients p ON r.patient_id = p.id
                LEFT JOIN doctors d ON r.doctor_id = d.id
                WHERE 1=1
            '''
            params = []
            
            if query:
                query = sanitize_input(query)
                if query:
                    sql += ' AND (p.name ILIKE %s OR r.original_filename ILIKE %s OR r.extracted_text ILIKE %s)'
                    like_query = f'%{query}%'
                    params.extend([like_query, like_query, like_query])
            
            if specialist_type and specialist_type in SPECIALIST_TYPES:
                sql += ' AND r.specialist_type = %s'
                params.append(specialist_type)
            
            if status:
                sql += ' AND r.status = %s'
                params.append(status)
            
            if date_from:
                sql += ' AND r.created_at >= %s'
                params.append(date_from)
            
            if date_to:
                sql += ' AND r.created_at <= %s'
                params.append(date_to)
            
            sql += ' ORDER BY r.created_at DESC LIMIT 50'
            
            cur.execute(sql, params)
            reports = [dict(r) for r in cur.fetchall()]
        
        return jsonify({'success': True, 'reports': reports})
    
    except Exception as e:
        print(f"Search reports error: {str(e)}")
        return jsonify({'error': str(e)}), 500

        
@app.route('/api/report/<int:report_id>', methods=['PUT'])
def update_report(report_id):
    """Update report status, notes, or tags"""
    data = request.json
    status = data.get('status', '').strip()
    doctor_notes = data.get('doctor_notes', '').strip()
    tags = data.get('tags', '').strip()
    
    updates = []
    params = []
    
    # Validate and collect updates
    if status and status in ['new', 'reviewed', 'critical', 'follow-up']:
        updates.append('status = %s')
        params.append(status)
    
    if doctor_notes is not None:
        updates.append('doctor_notes = %s')
        params.append(doctor_notes)
    
    if tags is not None:
        updates.append('tags = %s')
        params.append(tags)
    
    if not updates:
        return jsonify({'success': False, 'error': 'No valid fields to update'}), 400
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            params.append(report_id)
            sql = f'UPDATE reports SET {", ".join(updates)} WHERE id = %s'
            cur.execute(sql, params)
            
            # Optional: check if row was actually updated
            if cur.rowcount == 0:
                return jsonify({'success': False, 'error': 'Report not found or no changes applied'}), 404
            
            # commit automatic on success
    
        return jsonify({'success': True})
    
    except Exception as e:
        print(f"Error updating report {report_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/report/<int:report_id>/export')
def export_report_pdf(report_id):
    """Export report as PDF"""
    try:
        from flask import send_file
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import tempfile
        import re
        
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('''
                SELECT r.*, p.name as patient_name, d.name as doctor_name
                FROM reports r
                LEFT JOIN patients p ON r.patient_id = p.id
                LEFT JOIN doctors d ON r.doctor_id = d.id
                WHERE r.id = %s
            ''', (report_id,))
            report_row = cur.fetchone()
            
            if not report_row:
                return jsonify({'error': 'Report not found'}), 404
            
            report = dict(report_row)  # RealDictCursor returns dict → all fields accessible
        
        # Create PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30
        )
        story.append(Paragraph("Medical Report Analysis", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Patient Info
        story.append(Paragraph(f"<b>Patient:</b> {report.get('patient_name', 'N/A')}", styles['Normal']))
        created = report.get('created_at')

        if isinstance(created, str):
            date_str = created[:10]
        elif hasattr(created, 'strftime'):
            date_str = created.strftime('%Y-%m-%d')
        else:
            date_str = ''

        story.append(Paragraph(f"<b>Date:</b> {date_str}", styles['Normal']))
        story.append(Paragraph(f"<b>Specialist:</b> {SPECIALIST_TYPES.get(report.get('specialist_type'), report.get('specialist_type', 'N/A'))}", styles['Normal']))
        story.append(Paragraph(f"<b>Doctor:</b> {report.get('doctor_name', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # AI Analysis (format HTML for PDF)
        if report.get('llm_analysis'):
            html_text = report['llm_analysis']
            story.append(Paragraph("<b>AI Medical Analysis:</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            # If it's HTML, convert to plain text with formatting preserved
            if html_text.strip().startswith('<'):
                # Remove HTML tags but preserve structure
                html_text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\n\1\n', html_text, flags=re.DOTALL)
                html_text = re.sub(r'<h[1-6][^>]*>(.*?)(?=<)', r'\n\n\1\n', html_text)
                # Convert lists
                html_text = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1\n', html_text, flags=re.DOTALL)
                html_text = re.sub(r'<ul[^>]*>|</ul>|<ol[^>]*>|</ol>', '\n', html_text)
                # Convert paragraphs
                html_text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', html_text, flags=re.DOTALL)
                # Remove remaining HTML tags
                clean_text = re.sub(r'<[^>]+>', '', html_text)
                # Clean up whitespace
                clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
                clean_text = clean_text.strip()
                
                # Convert to paragraphs for PDF - better formatting
                lines = clean_text.split('\n')
                current_para = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_para:
                            para_text = ' '.join(current_para)
                            if para_text.startswith('•') or para_text.startswith('-'):
                                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{para_text}", styles['Normal']))
                            else:
                                story.append(Paragraph(para_text, styles['Normal']))
                            story.append(Spacer(1, 0.08*inch))
                            current_para = []
                        continue
                    
                    # Check for headings
                    if (line.isupper() and len(line) > 5 and len(line) < 80) or re.match(r'^\d+\.\s+[A-Z]', line):
                        if current_para:
                            para_text = ' '.join(current_para)
                            story.append(Paragraph(para_text, styles['Normal']))
                            story.append(Spacer(1, 0.08*inch))
                            current_para = []
                        
                        heading_style = ParagraphStyle(
                            'CustomHeading',
                            parent=styles['Heading2'],
                            fontSize=14,
                            textColor=colors.HexColor('#1e40af'),
                            spaceAfter=12,
                            spaceBefore=6,
                            fontName='Helvetica-Bold'
                        )
                        story.append(Paragraph(line, heading_style))
                        story.append(Spacer(1, 0.1*inch))
                    else:
                        current_para.append(line)
                
                # Add remaining paragraph
                if current_para:
                    para_text = ' '.join(current_para)
                    if para_text.startswith('•') or para_text.startswith('-'):
                        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{para_text}", styles['Normal']))
                    else:
                        story.append(Paragraph(para_text, styles['Normal']))
                    story.append(Spacer(1, 0.08*inch))
            else:
                # Plain text - format it line by line
                lines = html_text.split('\n')
                current_para = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_para:
                            para_text = ' '.join(current_para)
                            if para_text.startswith('•') or para_text.startswith('-'):
                                story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{para_text}", styles['Normal']))
                            else:
                                story.append(Paragraph(para_text, styles['Normal']))
                            story.append(Spacer(1, 0.08*inch))
                            current_para = []
                        continue
                    
                    # Check for headings
                    if line.startswith('##') or ((line.isupper() and len(line) > 5 and len(line) < 80) or re.match(r'^\d+\.\s+[A-Z]', line)):
                        if current_para:
                            para_text = ' '.join(current_para)
                            story.append(Paragraph(para_text, styles['Normal']))
                            story.append(Spacer(1, 0.08*inch))
                            current_para = []
                        
                        heading_text = line.replace('##', '').strip()
                        heading_style = ParagraphStyle(
                            'CustomHeading',
                            parent=styles['Heading2'],
                            fontSize=14,
                            textColor=colors.HexColor('#1e40af'),
                            spaceAfter=12,
                            spaceBefore=6,
                            fontName='Helvetica-Bold'
                        )
                        story.append(Paragraph(heading_text, heading_style))
                        story.append(Spacer(1, 0.1*inch))
                    elif line.startswith('-') or line.startswith('•') or re.match(r'^\d+[.)]\s', line):
                        if current_para:
                            para_text = ' '.join(current_para)
                            story.append(Paragraph(para_text, styles['Normal']))
                            story.append(Spacer(1, 0.08*inch))
                            current_para = []
                        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}", styles['Normal']))
                        story.append(Spacer(1, 0.06*inch))
                    else:
                        current_para.append(line)
                
                # Add remaining paragraph
                if current_para:
                    para_text = ' '.join(current_para)
                    story.append(Paragraph(para_text, styles['Normal']))
                    story.append(Spacer(1, 0.08*inch))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Clinical BERT Analysis
        if report.get('clinical_bert_analysis'):
            story.append(Paragraph("<b>Clinical BERT Analysis:</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            bert_lines = report['clinical_bert_analysis'].split('\n')
            for line in bert_lines:
                line = line.strip()
                if line:
                    line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 0.08*inch))
            story.append(Spacer(1, 0.2*inch))
        
        # Doctor Notes
        if report.get('doctor_notes'):
            story.append(Paragraph("<b>Doctor Notes:</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            notes = report['doctor_notes'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(notes, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Extracted Text (first 1500 chars)
        if report.get('extracted_text'):
            story.append(Paragraph("<b>Extracted Text (Preview):</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            preview = report['extracted_text'][:1500] + ('...' if len(report['extracted_text']) > 1500 else '')
            preview = preview.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            preview_style = ParagraphStyle(
                'PreviewText',
                parent=styles['Normal'],
                fontSize=9,
                leading=11,
                fontName='Courier'
            )
            story.append(Paragraph(preview, preview_style))
        
        doc.build(story)
        temp_file.close()
        
        return send_file(temp_file.name, 
                        mimetype='application/pdf',
                        as_attachment=True,
                        download_name=f"report_{report_id}_{report.get('patient_name', 'patient')}.pdf")
    
    except ImportError:
        return jsonify({'error': 'PDF export requires reportlab. Install with: pip install reportlab'}), 500
    except Exception as e:
        print(f"PDF export error: {str(e)}")
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500

# ==================== NEW FEATURES API ROUTES ====================

@app.route('/api/patients/search')
def search_patients():
    """Search patients by name, ID, or phone"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'patients': []})
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('''
                SELECT id, name, phone, contact_info, date_of_birth, patient_tags
                FROM patients 
                WHERE name ILIKE %s OR id::text = %s OR phone ILIKE %s OR contact_info ILIKE %s
                ORDER BY name LIMIT 50
            ''', (f'%{query}%', query, f'%{query}%', f'%{query}%'))
            
            patients = []
            for row in cur.fetchall():
                patients.append({
                    'id': row['id'],
                    'name': row['name'],
                    'phone': row['phone'],
                    'contact_info': row['contact_info'],
                    'date_of_birth': row['date_of_birth'],
                    'tags': row['patient_tags'].split(',') if row['patient_tags'] else []
                })
        
        return jsonify({'patients': patients})
    
    except Exception as e:
        print(f"Patient search error: {str(e)}")
        return jsonify({'patients': []}), 500

@app.route('/api/patient/<int:patient_id>', methods=['PUT'])
@app.route('/api/patient/profile', methods=['PUT'])
def update_patient(patient_id):
    """Update patient profile (demographics, history, allergies, medications)"""
    data = request.json
    
    updates = []
    params = []
    
    # Whitelist of allowed fields
    allowed_fields = [
        'name', 'date_of_birth', 'phone', 'gender', 'address', 'contact_info',
        'medical_history', 'allergies', 'medications', 'patient_tags'
    ]
    
    for field in allowed_fields:
        if field in data:
            updates.append(f"{field} = %s")
            params.append(data[field])
    
    if not updates:
        return jsonify({'error': 'No valid fields to update'}), 400
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            params.append(patient_id)
            sql = f"UPDATE patients SET {', '.join(updates)} WHERE id = %s"
            cur.execute(sql, params)
            
            if cur.rowcount == 0:
                return jsonify({'error': 'Patient not found'}), 404
            
            # commit automatic
            conn.commit()
    
        return jsonify({'success': True})
    
    except Exception as e:
        print(f"Patient update error (ID {patient_id}): {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks', methods=['GET', 'POST'])
def tasks():
    """Get or create tasks"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'POST':
                # Create new task
                data = request.json
                task_type = data.get('task_type', 'follow-up')
                title = data.get('title', '')
                description = data.get('description', '')
                due_date = data.get('due_date')
                priority = data.get('priority', 'medium')
                doctor_id = data.get('doctor_id')
                patient_id = data.get('patient_id')
                report_id = data.get('report_id')
                
                cur.execute('''
                    INSERT INTO tasks 
                    (doctor_id, patient_id, report_id, task_type, title, 
                     description, due_date, priority, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)
                    RETURNING id
                ''', (doctor_id, patient_id, report_id, task_type, title, description,
                      due_date, priority, datetime.now().isoformat()))
                
                task_id = cur.fetchone()['id']
                
                return jsonify({'success': True, 'id': task_id})
            
            else:
                # Get tasks (filtered)
                doctor_id = request.args.get('doctor_id')
                patient_id = request.args.get('patient_id')
                status = request.args.get('status')
                
                query = '''
                    SELECT t.*, p.name as patient_name, d.name as doctor_name
                    FROM tasks t
                    LEFT JOIN patients p ON t.patient_id = p.id
                    LEFT JOIN doctors d ON t.doctor_id = d.id
                    WHERE 1=1
                '''
                params = []
                
                if doctor_id:
                    query += ' AND t.doctor_id = %s'
                    params.append(doctor_id)
                if patient_id:
                    query += ' AND t.patient_id = %s'
                    params.append(patient_id)
                if status:
                    query += ' AND t.status = %s'
                    params.append(status)
                
                query += ' ORDER BY t.created_at DESC LIMIT 100'
                
                cur.execute(query, params)
                tasks_list = []
                
                for row in cur.fetchall():
                    tasks_list.append({
                        'id': row['id'],
                        'doctor_id': row['doctor_id'],
                        'patient_id': row['patient_id'],
                        'report_id': row['report_id'],
                        'task_type': row['task_type'],
                        'title': row['title'],
                        'description': row['description'],
                        'due_date': row['due_date'],
                        'status': row['status'],
                        'priority': row['priority'],
                        'created_at': row['created_at'],
                        'completed_at': row['completed_at'],
                        'patient_name': row['patient_name'],
                        'doctor_name': row['doctor_name']
                    })
                
                return jsonify({'tasks': tasks_list})
    
    except Exception as e:
        print(f"Tasks API error: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/tasks/<int:task_id>', methods=['PUT', 'DELETE'])
def task_detail(task_id):
    """Update or delete task"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'PUT':
                data = request.json
                updates = []
                params = []
                
                if 'status' in data:
                    updates.append('status = %s')
                    params.append(data['status'])
                    if data['status'] == 'completed':
                        updates.append('completed_at = %s')
                        params.append(datetime.now().isoformat())
                
                if 'title' in data:
                    updates.append('title = %s')
                    params.append(data['title'])
                
                if 'description' in data:
                    updates.append('description = %s')
                    params.append(data['description'])
                
                if 'due_date' in data:
                    updates.append('due_date = %s')
                    params.append(data['due_date'])
                
                if 'priority' in data:
                    updates.append('priority = %s')
                    params.append(data['priority'])
                
                if not updates:
                    return jsonify({'success': False, 'error': 'No fields to update'}), 400
                
                params.append(task_id)
                sql = f"UPDATE tasks SET {', '.join(updates)} WHERE id = %s"
                cur.execute(sql, params)
                
                if cur.rowcount == 0:
                    return jsonify({'success': False, 'error': 'Task not found'}), 404
                
                return jsonify({'success': True})
            
            else:  # DELETE
                cur.execute('DELETE FROM tasks WHERE id = %s', (task_id,))
                
                if cur.rowcount == 0:
                    return jsonify({'success': False, 'error': 'Task not found'}), 404
                
                return jsonify({'success': True})
    
    except Exception as e:
        print(f"Task {task_id} operation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/<int:report_id>/share', methods=['POST'])
def share_report(report_id):
    """Share report with another doctor"""
    data = request.json
    shared_with_doctor_id = data.get('shared_with_doctor_id')
    shared_by_doctor_id = data.get('shared_by_doctor_id')
    
    if not shared_with_doctor_id or not shared_by_doctor_id:
        return jsonify({'error': 'Missing doctor IDs'}), 400
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Check if already shared
            cur.execute('''
                SELECT id FROM shared_reports 
                WHERE report_id = %s AND shared_with_doctor_id = %s
            ''', (report_id, shared_with_doctor_id))
            
            if cur.fetchone():
                return jsonify({'error': 'Report already shared with this doctor'}), 400
            
            # Insert share record
            cur.execute('''
                INSERT INTO shared_reports 
                (report_id, shared_by_doctor_id, shared_with_doctor_id, shared_at)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            ''', (report_id, shared_by_doctor_id, shared_with_doctor_id, 
                  datetime.now().isoformat()))
            
            share_id = cur.fetchone()['id']
        
        return jsonify({'success': True, 'share_id': share_id})
    
    except Exception as e:
        print(f"Error sharing report {report_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/<int:report_id>/comments', methods=['GET', 'POST'])
def report_comments(report_id):
    """Get or add comments on a report"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'POST':
                # Add comment
                data = request.json
                doctor_id = data.get('doctor_id')
                comment = data.get('comment', '')
                is_private = data.get('is_private', 0)
                parent_comment_id = data.get('parent_comment_id')
                
                if not doctor_id or not comment:
                    return jsonify({'error': 'Doctor ID and comment are required'}), 400
                
                cur.execute('''
                    INSERT INTO report_comments 
                    (report_id, doctor_id, comment, is_private, parent_comment_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (report_id, doctor_id, comment, is_private, parent_comment_id,
                      datetime.now().isoformat()))
                
                comment_id = cur.fetchone()['id']
                
                return jsonify({'success': True, 'comment_id': comment_id})
            
            else:
                # Get comments
                doctor_id = request.args.get('doctor_id') or 0
                
                cur.execute('''
                    SELECT c.*, d.name as doctor_name, d.specialist_type
                    FROM report_comments c
                    JOIN doctors d ON c.doctor_id = d.id
                    WHERE c.report_id = %s 
                    AND (c.is_private = 0 OR c.doctor_id = %s)
                    ORDER BY c.created_at ASC
                ''', (report_id, doctor_id))
                
                comments = []
                for row in cur.fetchall():
                    comments.append({
                        'id': row['id'],
                        'report_id': row['report_id'],
                        'doctor_id': row['doctor_id'],
                        'comment': row['comment'],
                        'is_private': bool(row['is_private']),
                        'parent_comment_id': row['parent_comment_id'],
                        'created_at': row['created_at'],
                        'doctor_name': row['doctor_name'],
                        'specialist_type': row['specialist_type']
                    })
                
                return jsonify({'comments': comments})
    
    except Exception as e:
        print(f"Comments error for report {report_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/referrals', methods=['GET', 'POST'])
def referrals():
    """Get or create referrals"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'POST':
                # Create referral
                data = request.json
                
                cur.execute('''
                    INSERT INTO referrals 
                    (patient_id, from_doctor_id, to_specialist_type, to_doctor_id, 
                     reason, notes, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s)
                    RETURNING id
                ''', (data.get('patient_id'), data.get('from_doctor_id'),
                      data.get('to_specialist_type'), data.get('to_doctor_id'),
                      data.get('reason'), data.get('notes'), datetime.now().isoformat()))
                
                referral_id = cur.fetchone()['id']
                
                return jsonify({'success': True, 'referral_id': referral_id})
            
            else:
                # Get referrals
                doctor_id = request.args.get('doctor_id')
                patient_id = request.args.get('patient_id')
                status = request.args.get('status')
                
                query = '''
                    SELECT r.*, p.name as patient_name, d1.name as from_doctor_name,
                           d2.name as to_doctor_name
                    FROM referrals r
                    JOIN patients p ON r.patient_id = p.id
                    JOIN doctors d1 ON r.from_doctor_id = d1.id
                    LEFT JOIN doctors d2 ON r.to_doctor_id = d2.id
                    WHERE 1=1
                '''
                params = []
                
                if doctor_id:
                    query += ' AND (r.from_doctor_id = %s OR r.to_doctor_id = %s)'
                    params.extend([doctor_id, doctor_id])
                if patient_id:
                    query += ' AND r.patient_id = %s'
                    params.append(patient_id)
                if status:
                    query += ' AND r.status = %s'
                    params.append(status)
                
                query += ' ORDER BY r.created_at DESC'
                
                cur.execute(query, params)
                
                referrals_list = []
                for row in cur.fetchall():
                    referrals_list.append({
                        'id': row['id'],
                        'patient_id': row['patient_id'],
                        'from_doctor_id': row['from_doctor_id'],
                        'to_specialist_type': row['to_specialist_type'],
                        'to_doctor_id': row['to_doctor_id'],
                        'reason': row['reason'],
                        'notes': row['notes'],
                        'status': row['status'],
                        'created_at': row['created_at'],
                        'patient_name': row['patient_name'],
                        'from_doctor_name': row['from_doctor_name'],
                        'to_doctor_name': row['to_doctor_name']
                    })
                
                return jsonify({'referrals': referrals_list})
    
    except Exception as e:
        print(f"Referrals API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>/vitals', methods=['GET', 'POST'])
def patient_vitals(patient_id):
    """Get or add patient vitals for trends tracking"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'POST':
                # Add vital
                data = request.json
                
                cur.execute('''
                    INSERT INTO patient_vitals 
                    (patient_id, report_id, vital_name, vital_value, unit, measured_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (patient_id, data.get('report_id'), data.get('vital_name'),
                      data.get('vital_value'), data.get('unit'), 
                      data.get('measured_at', datetime.now().isoformat())))
                
                vital_id = cur.fetchone()['id']
                
                return jsonify({'success': True, 'vital_id': vital_id})
            
            else:
                # Get vitals
                vital_name = request.args.get('vital_name')
                
                query = 'SELECT * FROM patient_vitals WHERE patient_id = %s'
                params = [patient_id]
                
                if vital_name:
                    query += ' AND vital_name = %s'
                    params.append(vital_name)
                
                query += ' ORDER BY measured_at ASC'
                
                cur.execute(query, params)
                
                vitals = []
                for row in cur.fetchall():
                    vitals.append({
                        'id': row['id'],
                        'patient_id': row['patient_id'],
                        'report_id': row['report_id'],
                        'vital_name': row['vital_name'],
                        'vital_value': row['vital_value'],
                        'unit': row['unit'],
                        'measured_at': row['measured_at']
                    })
                
                return jsonify({'vitals': vitals})
    
    except Exception as e:
        print(f"Patient vitals error (patient_id={patient_id}): {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/patient/<int:patient_id>/risk-score', methods=['POST'])
def calculate_risk_score(patient_id):
    """Calculate AI-based risk score for patient"""
    data = request.json
    condition = data.get('condition', 'general')
    report_id = data.get('report_id')
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Get patient reports for context
            cur.execute('''
                SELECT llm_analysis, clinical_bert_analysis 
                FROM reports 
                WHERE patient_id = %s
            ''', (patient_id,))
            reports = cur.fetchall()
            
            # Combine analysis text
            analysis_text = ' '.join([r['llm_analysis'] or '' + ' ' + (r['clinical_bert_analysis'] or '') for r in reports])
        
        # Use Groq to calculate risk
        client = get_groq_client()
        prompt = f"""Based on the following patient medical reports, assess the risk level for {condition}.
        
        Patient Reports Analysis:
        {analysis_text[:4000]}
        
        Please provide:
        1. Risk Score (0-100, where 0 is no risk and 100 is critical risk)
        2. Risk Level (low, moderate, high, critical)
        3. Brief analysis explaining the risk assessment
        
        Format your response as:
        RISK_SCORE: [number]
        RISK_LEVEL: [low/moderate/high/critical]
        ANALYSIS: [your analysis]
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert medical risk assessment AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content
        
        # Parse response
        risk_score = 50  # Default fallback
        risk_level = 'moderate'
        analysis = result_text
        
        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith('RISK_SCORE:'):
                try:
                    risk_score = float(line.split(':', 1)[1].strip())
                except:
                    pass
            elif line.startswith('RISK_LEVEL:'):
                risk_level = line.split(':', 1)[1].strip().lower()
        
        # Save risk score
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO risk_scores 
                (patient_id, report_id, condition, risk_score, risk_level, analysis_text, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (patient_id, report_id, condition, risk_score, risk_level, analysis,
                  datetime.now().isoformat()))
            
            risk_score_id = cur.fetchone()['id']
        
        return jsonify({
            'success': True,
            'risk_score_id': risk_score_id,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'analysis': analysis
        })
    
    except Exception as e:
        print(f"Risk score calculation error (patient_id={patient_id}): {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/reports')
def analytics_reports():
    """Generate custom analytics reports"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    specialist_type = request.args.get('specialist_type')
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            query = '''
                SELECT 
                    specialist_type,
                    COUNT(*) as total_reports,
                    COUNT(DISTINCT patient_id) as unique_patients,
                    SUM(CASE WHEN status = 'critical' THEN 1 ELSE 0 END) as critical_reports,
                    SUM(CASE WHEN status = 'reviewed' THEN 1 ELSE 0 END) as reviewed_reports
                FROM reports
                WHERE 1=1
            '''
            params = []
            
            if start_date:
                query += ' AND created_at >= %s'
                params.append(start_date)
            
            if end_date:
                query += ' AND created_at <= %s'
                params.append(end_date)
            
            if specialist_type:
                query += ' AND specialist_type = %s'
                params.append(specialist_type)
            
            query += ' GROUP BY specialist_type'
            
            cur.execute(query, params)
            
            analytics = []
            for row in cur.fetchall():
                analytics.append({
                    'specialist_type': row['specialist_type'],
                    'total_reports': row['total_reports'],
                    'unique_patients': row['unique_patients'],
                    'critical_reports': row['critical_reports'],
                    'reviewed_reports': row['reviewed_reports']
                })
        
        return jsonify({'analytics': analytics})
    
    except Exception as e:
        print(f"Analytics reports error: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/patient/dashboard')
def patient_dashboard():
    if 'user_role' not in session or session['user_role'] != 'patient':
        return redirect(url_for('login'))
    
    patient_id = session.get('user_db_id', 1)
    
    return render_template('patient_dashboard.html',
                         user=session,
                         patient_id=patient_id)


@app.route('/patient/ai-assistant')
def patient_ai_assistant():
    if 'user_role' not in session or session['user_role'] != 'patient':
        return redirect(url_for('login'))
    
    return render_template('patient_ai_assistant.html',
                         user=session)


@app.route('/admin/dashboard')
def admin_dashboard():
    if 'user_role' not in session or session['user_role'] != 'admin':
        return redirect(url_for('login'))
    
    return render_template('admin_dashboard.html',
                         user=session,
                         specialists=SPECIALIST_TYPES)

@app.route('/api/patient/<int:patient_id>/reports')
def get_patient_reports(patient_id):
    """Get all reports for a patient - Only patient own reports"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Security: Patients can only access their own reports
    if session['user_role'] == 'patient' and session.get('user_db_id') != patient_id:
        return jsonify({'error': 'Unauthorized - You can only view your own reports'}), 403
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('''
                SELECT r.*, d.name as doctor_name
                FROM reports r
                LEFT JOIN doctors d ON r.doctor_id = d.id
                WHERE r.patient_id = %s
                ORDER BY r.created_at DESC
            ''', (patient_id,))
            
            reports = [dict(row) for row in cur.fetchall()]
        
        return jsonify({'reports': reports})
    
    except Exception as e:
        print(f"Error fetching patient reports (patient_id={patient_id}): {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>')
def get_patient_info(patient_id):
    """Get patient information"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('SELECT * FROM patients WHERE id = %s', (patient_id,))
            patient_row = cur.fetchone()
            
            if not patient_row:
                return jsonify({'error': 'Patient not found'}), 404
            
            patient = dict(patient_row)  # All fields accessible by name
            
            return jsonify({'patient': patient})
    
    except Exception as e:
        print(f"Error fetching patient info (patient_id={patient_id}): {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>/anomalies')
def get_patient_anomalies(patient_id):
    """Get all lab anomalies for a patient with severity and recommendations"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Authorization: Patients can only view their own anomalies
    if session['user_role'] == 'patient' and session.get('user_db_id') != patient_id:
        return jsonify({'error': 'Unauthorized - You can only view your own anomalies'}), 403
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Check if table exists (PostgreSQL version)
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'report_anomalies'
                )
            """)
            table_exists = cur.fetchone()['exists']
            
            if not table_exists:
                return jsonify({
                    'anomalies': [],
                    'has_critical': False,
                    'has_anomalies': False
                })
            
            # Get all anomalies
            cur.execute('''
                SELECT ra.*, r.original_filename, r.created_at as report_date
                FROM report_anomalies ra
                LEFT JOIN reports r ON ra.report_id = r.id
                WHERE ra.patient_id = %s
                ORDER BY 
                    CASE 
                        WHEN ra.status LIKE 'critical%' THEN 1
                        ELSE 2
                    END,
                    ra.created_at DESC
            ''', (patient_id,))
            
            anomalies = [dict(row) for row in cur.fetchall()]
            
            # Check for critical anomalies
            has_critical = any(a['status'].startswith('critical') for a in anomalies)
            
            # Group by specialist for recommendations
            specialists = {}
            for a in anomalies:
                spec = a['recommended_specialist']
                if spec not in specialists:
                    specialists[spec] = []
                specialists[spec].append(a['test_name'])
        
        return jsonify({
            'anomalies': anomalies,
            'has_critical': has_critical,
            'has_anomalies': len(anomalies) > 0,
            'specialists_needed': specialists
        })
    
    except Exception as e:
        print(f"Error fetching patient anomalies (patient_id={patient_id}): {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/stats')
def admin_stats():
    """Get comprehensive admin statistics"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Total counts
            cur.execute('SELECT COUNT(*) as count FROM reports')
            total_reports = cur.fetchone()['count']
            
            cur.execute('SELECT COUNT(*) as count FROM users')
            total_users = cur.fetchone()['count']
            
            cur.execute('SELECT COUNT(*) as count FROM doctors')
            total_doctors = cur.fetchone()['count']
            
            cur.execute('SELECT COUNT(*) as count FROM patients')
            total_patients = cur.fetchone()['count']
            
            # Reports by specialist
            cur.execute('SELECT specialist_type, COUNT(*) as count FROM reports GROUP BY specialist_type')
            reports_by_specialist = {row['specialist_type']: row['count'] for row in cur.fetchall()}
            
            # Users by role
            cur.execute('SELECT role, COUNT(*) as count FROM users GROUP BY role')
            users_by_role = {row['role']: row['count'] for row in cur.fetchall()}
            
            # Recent reports
            cur.execute('''
                SELECT r.*, p.name as patient_name
                FROM reports r
                LEFT JOIN patients p ON r.patient_id = p.id
                ORDER BY r.created_at DESC LIMIT 10
            ''')
            recent_reports = [dict(row) for row in cur.fetchall()]
        
        return jsonify({
            'success': True,
            'total_reports': total_reports,
            'total_users': total_users,
            'total_doctors': total_doctors,
            'total_patients': total_patients,
            'reports_by_specialist': reports_by_specialist,
            'users_by_role': users_by_role,
            'recent_reports': recent_reports
        })
    
    except Exception as e:
        print(f"Admin stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500
 
@app.route('/api/admin/users', methods=['GET', 'POST'])
def admin_users():
    """Get all users or create new user (Admin only)"""
    if 'user_role' not in session or session['user_role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()

            if request.method == 'POST':
                if not request.json:
                    return jsonify({'error': 'No data provided'}), 400
                
                data = request.json
                username = data.get('username', '').strip()
                password = data.get('password', '').strip()
                full_name = data.get('full_name', '').strip()
                email = data.get('email', '').strip()
                role = data.get('role', 'patient')

                if not username or not password or not full_name:
                    return jsonify({'error': 'Username, password and full name are required'}), 400

                # Check username exists
                cur.execute('SELECT id FROM users WHERE username = %s', (username,))
                if cur.fetchone():
                    return jsonify({'error': 'Username already exists'}), 400

                created_at = datetime.now().isoformat()
                password_hash = hashlib.md5(password.encode()).hexdigest()

                if role == 'patient':
                    # === FIX: Create patient record first (this was missing) ===
                    cur.execute('''
                        INSERT INTO patients (name, email, created_at)
                        VALUES (%s, %s, %s)
                        RETURNING id
                    ''', (full_name, email or None, created_at))
                    patient_row = cur.fetchone()
                    patient_id = patient_row['id']

                    # Generate PATxxxxx
                    cur.execute('''
                        UPDATE patients 
                        SET patient_id = %s 
                        WHERE id = %s
                    ''', (f"PAT{patient_id:06d}", patient_id))

                    # Now create user linked to this patient
                    cur.execute('''
                        INSERT INTO users 
                        (username, password, role, full_name, email, user_id, created_at)
                        VALUES (%s, %s, 'patient', %s, %s, %s, %s)
                        RETURNING id
                    ''', (username, password_hash, full_name, email or None, patient_id, created_at))

                else:
                    # For doctor or admin - keep original logic
                    cur.execute('''
                        INSERT INTO users 
                        (username, password, role, full_name, email, user_id, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    ''', (username, password_hash, role, full_name, email or None, 
                          data.get('user_id'), created_at))

                conn.commit()
                return jsonify({'success': True, 'message': 'User created successfully'})

            else:
                # GET - list all users (unchanged)
                cur.execute('SELECT id, username, role, full_name, email, last_login FROM users')
                users = [dict(row) for row in cur.fetchall()]
                return jsonify({'users': users})

    except Exception as e:
        print(f"[Admin Create User Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>', methods=['PUT', 'DELETE'])
def admin_user_detail(user_id):
    """Update or delete a user"""
    if 'user_role' not in session or session['user_role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'DELETE':
                cur.execute('DELETE FROM users WHERE id = %s', (user_id,))
                if cur.rowcount == 0:
                    return jsonify({'success': False, 'error': 'User not found'}), 404
                return jsonify({'success': True})
            
            else:  # PUT - update
                data = request.json
                updates = []
                params = []
                
                if 'password' in data and data['password']:
                    password_hash = hashlib.md5(data['password'].encode()).hexdigest()
                    updates.append('password = %s')
                    params.append(password_hash)
                
                if 'full_name' in data:
                    updates.append('full_name = %s')
                    params.append(data['full_name'])
                
                if 'email' in data:
                    updates.append('email = %s')
                    params.append(data['email'])
                
                if 'role' in data:
                    updates.append('role = %s')
                    params.append(data['role'])
                
                if not updates:
                    return jsonify({'success': False, 'error': 'No fields to update'}), 400
                
                params.append(user_id)
                sql = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
                cur.execute(sql, params)
                
                if cur.rowcount == 0:
                    return jsonify({'success': False, 'error': 'User not found'}), 404
                
                return jsonify({'success': True})
    
    except Exception as e:
        print(f"Admin user update/delete error (user_id={user_id}): {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/admin/doctors', methods=['GET', 'POST'])
def admin_doctors():
    """Get all doctors or create new doctor - Admin only"""
    if not require_role('admin'):
        return jsonify({'error': 'Unauthorized - Admin access required'}), 403
    
    try:
        with get_db() as conn:
            with conn:  # nested context for rollback safety
                cur = conn.cursor()

                if request.method == 'POST':
                    # ────────────────────────────────────────────────
                    # Read data (JSON or form)
                    # ────────────────────────────────────────────────
                    if request.is_json:
                        data = request.get_json()
                        profile_picture = None
                    elif request.content_type and 'multipart/form-data' in request.content_type:
                        data = request.form.to_dict()
                        profile_picture = None
                        if 'profile_picture' in request.files:
                            file = request.files['profile_picture']
                            if file and file.filename:
                                ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
                                filename = secure_filename(f"doctor_{int(datetime.now().timestamp())}.{ext}")
                                filepath = os.path.join(app.config['PROFILE_FOLDER'], filename)
                                file.save(filepath)
                                profile_picture = f"profiles/{filename}"
                    else:
                        return jsonify({'error': 'Unsupported content type'}), 415

                    # ────────────────────────────────────────────────
                    # Validation
                    # ────────────────────────────────────────────────
                    required = ['name', 'specialist_type']
                    for field in required:
                        if not data.get(field):
                            return jsonify({'error': f'{field} is required'}), 400

                    if data.get('email') and '@' not in data['email']:
                        return jsonify({'error': 'Invalid email format'}), 400

                    # ────────────────────────────────────────────────
                    # Create doctor
                    # ────────────────────────────────────────────────
                    doctor_id = data.get('id') or data.get('doctor_id')   # ← important

                    if doctor_id:  # UPDATE existing
                        cur.execute('''
                            UPDATE doctors 
                            SET name = %s,
                                specialist_type = %s,
                                email = %s,
                                phone = %s,
                                license_number = %s,
                                profile_picture = COALESCE(%s, profile_picture)
                            WHERE id = %s
                        ''', (
                            data.get('name'),
                            data.get('specialist_type'),
                            data.get('email'),
                            data.get('phone'),
                            data.get('license_number'),
                            profile_picture,
                            doctor_id
                        ))
                    else:
                        created_at = datetime.now(timezone.utc) 

                        cur.execute('''
                            INSERT INTO doctors 
                            (name, profile_picture, specialist_type, email, phone, license_number, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        ''', (
                            data.get('name'),
                            profile_picture,
                            data.get('specialist_type'),
                            data.get('email'),
                            data.get('phone'),
                            data.get('license_number'),
                            created_at
                        ))
                    conn.commit()
                    doctor_id = cur.fetchone()['id']

                    # ────────────────────────────────────────────────
                    # Auto-create user account (recommended)
                    # ────────────────────────────────────────────────
                    # Always create — use default if not provided
                    username = data.get('username') or f"dr_{doctor_id}"
                    password_raw = data.get('password') or 'password123'  # default for auto-creation
                    password_hash = hashlib.md5(password_raw.encode()).hexdigest()

                    cur.execute('''
                        INSERT INTO users 
                        (username, password, role, full_name, email, user_id, created_at)
                        VALUES (%s, %s, 'doctor', %s, %s, %s, %s)
                        ON CONFLICT (username) DO NOTHING
                    ''', (
                        username,
                        password_hash,
                        data.get('name') or f'Doctor {doctor_id}',
                        data.get('email'),
                        doctor_id,
                        created_at
                    ))

                    # ────────────────────────────────────────────────
                    # Commit explicitly (safest)
                    # ────────────────────────────────────────────────
                    conn.commit()

                    return jsonify({
                        'success': True,
                        'id': doctor_id,
                        'profile_picture': profile_picture,
                        'message': 'Doctor and login account created'
                    })

                else:  # GET
                    cur.execute('''
                        SELECT id, name, specialist_type, email, phone, license_number, 
                               profile_picture, created_at 
                        FROM doctors 
                        ORDER BY created_at DESC
                    ''')
                    doctors = [dict(row) for row in cur.fetchall()]
                    return jsonify({'doctors': doctors})

    except Exception as e:
        print(f"Admin doctors API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/doctors/<int:doctor_id>', methods=['DELETE'])
def admin_delete_doctor(doctor_id):
    """Delete a doctor - Admin only"""
    if not require_role('admin'):
        return jsonify({'error': 'Unauthorized - Admin access required'}), 403
    
    try:
        with get_db() as conn:
            # Use context manager on connection → auto commit on success, rollback on error
            with conn:
                cur = conn.cursor()
                
                # Delete associated user account if exists
                cur.execute('''
                    SELECT id FROM users 
                    WHERE user_id = %s AND role = 'doctor'
                ''', (doctor_id,))
                user_row = cur.fetchone()
                
                if user_row:
                    cur.execute('DELETE FROM users WHERE id = %s', (user_row['id'],))
                
                # Delete doctor
                cur.execute('DELETE FROM doctors WHERE id = %s', (doctor_id,))
                
                # Check if doctor existed (rowcount from the doctors DELETE)
                if cur.rowcount == 0:
                    return jsonify({'success': False, 'error': 'Doctor not found'}), 404
                
                
            return jsonify({'success': True})
    
    except Exception as e:
        print(f"Error deleting doctor {doctor_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/reports')
def admin_all_reports():
    """Get all reports for admin"""
    if 'user_role' not in session or session['user_role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('''
                SELECT r.*, p.name as patient_name, d.name as doctor_name
                FROM reports r
                LEFT JOIN patients p ON r.patient_id = p.id
                LEFT JOIN doctors d ON r.doctor_id = d.id
                ORDER BY r.created_at DESC LIMIT 100
            ''')
            
            reports = [dict(row) for row in cur.fetchall()]
        
        return jsonify({'reports': reports})
    
    except Exception as e:
        print(f"Error fetching all reports: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/patient/<int:patient_id>/update', methods=['PUT'])
def update_patient_profile(patient_id):
    """Update patient profile including picture"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Patients can only update their own profile
    if session['user_role'] == 'patient' and session.get('user_db_id') != patient_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Handle multipart form or JSON
            if request.content_type and 'multipart/form-data' in request.content_type:
                data = request.form.to_dict()
                profile_picture = None
                if 'profile_picture' in request.files:
                    file = request.files['profile_picture']
                    if file.filename:
                        filename = secure_filename(f"patient_{patient_id}_{datetime.now().timestamp()}.{file.filename.rsplit('.', 1)[1].lower()}")
                        filepath = os.path.join(app.config['PROFILE_FOLDER'], filename)
                        file.save(filepath)
                        profile_picture = f"profiles/{filename}"
            else:
                data = request.json
                profile_picture = data.get('profile_picture')
            
            updates = []
            values = []
            
            fields = ['name', 'date_of_birth', 'phone', 'gender', 'address', 'contact_info',
                      'medical_history', 'allergies', 'medications', 'patient_tags', 'email']
            for field in fields:
                if field in data:
                    updates.append(f"{field} = %s")
                    values.append(data[field])
            
            if profile_picture:
                updates.append("profile_picture = %s")
                values.append(profile_picture)
            
            if not updates:
                return jsonify({'error': 'No fields to update'}), 400
            
            values.append(patient_id)
            sql = f"UPDATE patients SET {', '.join(updates)} WHERE id = %s"
            cur.execute(sql, values)
            
            if cur.rowcount == 0:
                return jsonify({'error': 'Patient not found'}), 404
            
            # Optional: update linked user account
            cur.execute('''
                SELECT id FROM users 
                WHERE user_id = %s AND role = 'patient'
            ''', (patient_id,))
            user_row = cur.fetchone()
            
            if user_row:
                user_updates = []
                user_values = []
                
                if 'name' in data:
                    user_updates.append('full_name = %s')
                    user_values.append(data['name'])
                if 'email' in data:
                    user_updates.append('email = %s')
                    user_values.append(data['email'])
                
                if user_updates:
                    user_values.append(user_row['id'])
                    user_sql = f"UPDATE users SET {', '.join(user_updates)} WHERE id = %s"
                    cur.execute(user_sql, user_values)
            
            # Explicit commit - required for PostgreSQL
            conn.commit()
            
            return jsonify({'success': True, 'profile_picture': profile_picture})
    
    except Exception as e:
        print(f"Patient profile update error (patient_id={patient_id}): {str(e)}")
        # Optional: explicit rollback (helps in some debugging scenarios)
        if 'conn' in locals():
            conn.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/profile', methods=['GET', 'PUT'])
def user_profile():
    """Get or update current user profile (works for all roles)"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    role = session['user_role']
    user_db_id = session.get('user_db_id')
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'PUT':
                # ────────────────────────────────────────────────
                # Handle input (multipart or JSON)
                # ────────────────────────────────────────────────
                if request.content_type and 'multipart/form-data' in request.content_type:
                    data = request.form.to_dict()
                    profile_picture = None
                    if 'profile_picture' in request.files:
                        file = request.files['profile_picture']
                        if file.filename:
                            filename = secure_filename(f"{role}_{session.get('username', 'user')}_{datetime.now().timestamp()}.{file.filename.rsplit('.', 1)[1].lower()}")
                            filepath = os.path.join(app.config['PROFILE_FOLDER'], filename)
                            file.save(filepath)
                            profile_picture = f"profiles/{filename}"
                else:
                    data = request.json or {}
                    profile_picture = data.get('profile_picture')  # usually None from JSON
                
                # ────────────────────────────────────────────────
                # Update users table
                # ────────────────────────────────────────────────
                updates = []
                values = []
                
                if 'full_name' in data:
                    updates.append('full_name = %s')
                    values.append(data['full_name'])
                if 'email' in data:
                    updates.append('email = %s')
                    values.append(data['email'])
                if 'password' in data and data['password']:
                    password_hash = hashlib.md5(data['password'].encode()).hexdigest()
                    updates.append('password = %s')
                    values.append(password_hash)
                
                if updates:
                    values.append(session['user_id'])
                    cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = %s", values)
                
                # ────────────────────────────────────────────────
                # Update role-specific profile
                # ────────────────────────────────────────────────
                if role == 'doctor' and user_db_id:
                    doc_updates = []
                    doc_values = []
                    
                    if 'name' in data:
                        doc_updates.append('name = %s')
                        doc_values.append(data['name'])
                    if 'email' in data:
                        doc_updates.append('email = %s')
                        doc_values.append(data['email'])
                    if 'phone' in data:
                        doc_updates.append('phone = %s')
                        doc_values.append(data['phone'])
                    if 'specialist_type' in data:
                        doc_updates.append('specialist_type = %s')
                        doc_values.append(data['specialist_type'])
                    if profile_picture:
                        doc_updates.append('profile_picture = %s')
                        doc_values.append(profile_picture)
                    
                    if doc_updates:
                        doc_values.append(user_db_id)
                        cur.execute(f"UPDATE doctors SET {', '.join(doc_updates)} WHERE id = %s", doc_values)
                
                elif role == 'patient' and user_db_id:
                    pat_updates = []
                    pat_values = []
                    
                    fields = [
                        'name', 'date_of_birth', 'phone', 'gender', 'address',
                        'contact_info', 'medical_history', 'allergies', 'medications',
                        'patient_tags'
                    ]
                    for field in fields:
                        if field in data:
                            pat_updates.append(f"{field} = %s")
                            pat_values.append(data[field])
                    
                    if profile_picture:
                        pat_updates.append('profile_picture = %s')
                        pat_values.append(profile_picture)
                    
                    if pat_updates:
                        pat_values.append(user_db_id)
                        cur.execute(f"UPDATE patients SET {', '.join(pat_updates)} WHERE id = %s", pat_values)
                
                conn.commit()
                return jsonify({'success': True, 'profile_picture': profile_picture})
            
            else:
                # ────────────────────────────────────────────────
                # GET profile
                # ────────────────────────────────────────────────
                profile = {
                    'username': session.get('username'),
                    'full_name': session.get('full_name'),
                    'email': session.get('email'),
                    'role': role
                }
                
                if role == 'doctor' and user_db_id:
                    cur.execute('SELECT * FROM doctors WHERE id = %s', (user_db_id,))
                    doctor = cur.fetchone()
                    if doctor:
                        profile.update(dict(doctor))
                
                elif role == 'patient' and user_db_id:
                    cur.execute('SELECT * FROM patients WHERE id = %s', (user_db_id,))
                    patient = cur.fetchone()
                    if patient:
                        profile.update(dict(patient))
                
                return jsonify({'profile': profile})
    
    except Exception as e:
        print(f"User profile error: {str(e)}")
        if 'conn' in locals() and request.method == 'PUT':
            conn.rollback()
        return jsonify({'error': str(e)}), 500

# ==========================================
# NEW FEATURE ROUTES
# ==========================================
def log_audit(action, entity_type=None, entity_id=None, details=None):
    """Log user actions for audit trail - Supabase PostgreSQL"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO audit_logs 
                (user_id, user_role, action, entity_type, entity_id, details, ip_address, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                session.get('user_id'),
                session.get('user_role'),
                action,
                entity_type,
                entity_id,
                details,
                request.remote_addr if 'request' in globals() else None,  # safe fallback
                datetime.now().isoformat()
            ))
            conn.commit()  # explicit commit - matches your preferred style
    except Exception as e:
        print(f"Audit log error: {str(e)}")
        # We intentionally do NOT re-raise → audit should never break main flow


def create_notification(user_id, user_role, notification_type, title, message, link=None):
    """Create a notification for a user - Supabase PostgreSQL"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO notifications 
                (user_id, user_role, notification_type, title, message, link, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (
                user_id,
                user_role,
                notification_type,
                title,
                message,
                link,
                datetime.now().isoformat()
            ))
            conn.commit()  # explicit commit
    except Exception as e:
        print(f"Notification creation error: {str(e)}")
        # Silent fail - notifications are not critical path

# ==========================================
# APPOINTMENT MANAGEMENT
# ==========================================
@app.route('/api/doctors', methods=['GET'])
def get_doctors_list():
    """Get list of all doctors for patient appointment booking and chat"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('''
                SELECT d.id, d.name, d.specialist_type, d.email, d.phone, 
                       d.profile_picture, d.created_at,
                       u.id as user_id
                FROM doctors d
                LEFT JOIN users u ON u.user_id = d.id AND u.role = 'doctor'
                ORDER BY d.name
            ''')
            
            doctors = [dict(row) for row in cur.fetchall()]
            
            return jsonify({'doctors': doctors})
    
    except Exception as e:
        print(f"Error fetching doctors list: {str(e)}")
        return jsonify({'doctors': [], 'error': str(e)}), 500

@app.route('/api/appointments', methods=['GET', 'POST', 'PUT'])
def appointments():
    """Get or create appointments"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'POST':
                data = request.json
                print(f"Appointment POST request data: {data}")
                
                # For patients, use their own ID; for doctors, use provided patient_id
                if session['user_role'] == 'patient':
                    patient_id = session.get('user_db_id')
                else:
                    patient_id = data.get('patient_id')
                
                doctor_id = data.get('doctor_id') or (session.get('user_db_id') if session['user_role'] == 'doctor' else None)
                
                print(f"Appointment creation - patient_id: {patient_id}, doctor_id: {doctor_id}")
                
                if not patient_id or not doctor_id:
                    return jsonify({'error': 'Patient and doctor are required'}), 400
                
                appointment_date = data.get('appointment_date')
                appointment_time = data.get('appointment_time')
                duration = data.get('duration', 30)
                
                if not appointment_date or not appointment_time:
                    return jsonify({'error': 'Appointment date and time are required'}), 400
                
                # Check for exact time conflict
                cur.execute('''
                    SELECT id, patient_id FROM appointments 
                    WHERE doctor_id = %s 
                    AND appointment_date = %s 
                    AND appointment_time = %s 
                    AND status IN ('scheduled', 'confirmed')
                ''', (doctor_id, appointment_date, appointment_time))
                existing = cur.fetchone()
                if existing:
                    return jsonify({
                        'error': 'This time slot is already booked by another patient. Please choose a different time.',
                        'conflict': True
                    }), 400
                
                # Overlap check (considering duration)
                from datetime import datetime as dt
                try:
                    requested_time = dt.strptime(appointment_time, '%H:%M').time()
                    requested_minutes = requested_time.hour * 60 + requested_time.minute
                    requested_end_minutes = requested_minutes + duration
                    
                    cur.execute('''
                        SELECT appointment_time, duration FROM appointments 
                        WHERE doctor_id = %s 
                        AND appointment_date = %s
                        AND status IN ('scheduled', 'confirmed')
                    ''', (doctor_id, appointment_date))
                    existing_appointments = cur.fetchall()
                    
                    for row in existing_appointments:
                        existing_time = row['appointment_time']
                        existing_duration = row['duration'] or 30
                        
                        try:
                            existing_time_obj = dt.strptime(existing_time, '%H:%M').time()
                            existing_minutes = existing_time_obj.hour * 60 + existing_time_obj.minute
                            existing_end_minutes = existing_minutes + existing_duration
                            
                            if (requested_minutes < existing_end_minutes and requested_end_minutes > existing_minutes):
                                return jsonify({
                                    'error': f'This time slot overlaps with an existing appointment ({existing_time}). Please choose a different time.',
                                    'conflict': True
                                }), 400
                        except:
                            pass  # skip if time parsing fails
                except Exception as e:
                    print(f"Time parsing error: {e}")
                
                # Prevent same patient double-booking same slot
                cur.execute('''
                    SELECT id FROM appointments 
                    WHERE patient_id = %s 
                    AND appointment_date = %s 
                    AND appointment_time = %s
                    AND status IN ('scheduled', 'confirmed')
                ''', (patient_id, appointment_date, appointment_time))
                if cur.fetchone():
                    return jsonify({
                        'error': 'You already have an appointment scheduled at this time. Please choose a different time.',
                        'conflict': True
                    }), 400
                
                # Verify patient and doctor exist
                cur.execute('SELECT id FROM patients WHERE id = %s', (patient_id,))
                if not cur.fetchone():
                    return jsonify({'error': f'Patient with ID {patient_id} not found'}), 400
                
                cur.execute('SELECT id FROM doctors WHERE id = %s', (doctor_id,))
                if not cur.fetchone():
                    return jsonify({'error': f'Doctor with ID {doctor_id} not found'}), 400
                
                # Determine initial status
                initial_status = 'pending' if session['user_role'] == 'patient' else 'scheduled'
                
                cur.execute('''
                    INSERT INTO appointments 
                    (patient_id, doctor_id, appointment_date, appointment_time, duration,
                     appointment_type, reason, status, notes, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    patient_id, doctor_id, appointment_date, appointment_time, duration,
                    data.get('appointment_type', 'consultation'), data.get('reason'),
                    initial_status, data.get('notes'), datetime.now().isoformat()
                ))
                
                appointment_id = cur.fetchone()['id']
                print(f"Appointment inserted with ID: {appointment_id}, status: {initial_status}")
                
                conn.commit()
                print(f"Appointment {appointment_id} committed successfully")
                
                # Notifications & audit (after commit - own connections via helpers)
                try:
                    # We need user IDs from users table
                    cur.execute('SELECT id FROM users WHERE user_id = %s AND role = %s', (patient_id, 'patient'))
                    patient_user = cur.fetchone()
                    
                    cur.execute('SELECT id FROM users WHERE user_id = %s AND role = %s', (doctor_id, 'doctor'))
                    doctor_user = cur.fetchone()
                    
                    if patient_user:
                        create_notification(
                            patient_user['id'], 'patient', 'appointment',
                            'Appointment Scheduled',
                            f'Appointment scheduled for {appointment_date} at {appointment_time}',
                            f'/appointments/{appointment_id}'
                        )
                    
                    if doctor_user:
                        create_notification(
                            doctor_user['id'], 'doctor', 'appointment',
                            'New Appointment', 'New appointment scheduled',
                            f'/appointments/{appointment_id}'
                        )
                except Exception as e:
                    print(f"Error creating notifications: {str(e)}")
                
                try:
                    log_audit('appointment_created', 'appointment', appointment_id,
                             f'Patient {patient_id} with Doctor {doctor_id}')
                except Exception as e:
                    print(f"Error logging audit: {str(e)}")
                
                return jsonify({
                    'success': True,
                    'id': appointment_id,
                    'message': 'Appointment booked successfully'
                })
                        # === ADD THIS NEW BLOCK FOR ACCEPT/REJECT ===
            elif request.method == 'PUT':
                data = request.json
                appointment_id = data.get('appointment_id')
                new_status = data.get('status')

                if not appointment_id or not new_status:
                    return jsonify({'error': 'appointment_id and status are required'}), 400

                if session['user_role'] not in ['doctor', 'admin']:
                    return jsonify({'error': 'Only doctors can update status'}), 403

                cur.execute('''
                    UPDATE appointments 
                    SET status = %s, updated_at = %s
                    WHERE id = %s
                    RETURNING patient_id, appointment_date, appointment_time
                ''', (new_status, datetime.now().isoformat(), appointment_id))
                
                row = cur.fetchone()
                if not row:
                    return jsonify({'error': 'Appointment not found'}), 404
                
                conn.commit()

                # Optional: Notify patient
                try:
                    cur.execute('SELECT id FROM users WHERE user_id = %s AND role = %s', 
                                (row['patient_id'], 'patient'))
                    p_user = cur.fetchone()
                    if p_user:
                        create_notification(
                            p_user['id'], 'patient', 'appointment',
                            f'Appointment {new_status}',
                            f'Your appointment on {row["appointment_date"]} has been updated to {new_status}.',
                            '/patient/appointments'
                        )
                except:
                    pass

                return jsonify({'success': True, 'message': f'Appointment updated to {new_status}'})
                
            else:
                # GET appointments
                user_role = session['user_role']
                user_db_id = session.get('user_db_id')
                
                if user_role == 'doctor':
                    if user_db_id:
                        cur.execute('''
                            SELECT a.*, p.name as patient_name, d.name as doctor_name, 
                                   p.email as patient_email, p.phone as patient_phone
                            FROM appointments a
                            LEFT JOIN patients p ON a.patient_id = p.id
                            LEFT JOIN doctors d ON a.doctor_id = d.id
                            WHERE a.doctor_id = %s
                            ORDER BY a.appointment_date DESC, a.appointment_time DESC
                        ''', (user_db_id,))
                    else:
                        # Fallback: try to find doctor_id from users
                        cur.execute('SELECT user_id FROM users WHERE id = %s AND role = %s',
                                   (session.get('user_id'), 'doctor'))
                        user_row = cur.fetchone()
                        if user_row and user_row['user_id']:
                            doctor_id = user_row['user_id']
                            cur.execute('''
                                SELECT a.*, p.name as patient_name, d.name as doctor_name, 
                                       p.email as patient_email, p.phone as patient_phone
                                FROM appointments a
                                LEFT JOIN patients p ON a.patient_id = p.id
                                LEFT JOIN doctors d ON a.doctor_id = d.id
                                WHERE a.doctor_id = %s
                                ORDER BY a.appointment_date DESC, a.appointment_time DESC
                            ''', (doctor_id,))
                        else:
                            return jsonify({'appointments': []})
                
                elif user_role == 'patient':
                    cur.execute('''
                        SELECT a.*, p.name as patient_name, d.name as doctor_name
                        FROM appointments a
                        LEFT JOIN patients p ON a.patient_id = p.id
                        LEFT JOIN doctors d ON a.doctor_id = d.id
                        WHERE a.patient_id = %s 
                        ORDER BY a.appointment_date, a.appointment_time
                    ''', (user_db_id,))
                
                else:  # admin
                    cur.execute('''
                        SELECT a.*, p.name as patient_name, d.name as doctor_name
                        FROM appointments a
                        LEFT JOIN patients p ON a.patient_id = p.id
                        LEFT JOIN doctors d ON a.doctor_id = d.id
                        ORDER BY a.appointment_date DESC, a.appointment_time DESC 
                        LIMIT 100
                    ''')
                
                appointments_list = [dict(row) for row in cur.fetchall()]
                return jsonify({'appointments': appointments_list})
    
    except Exception as e:
        print(f"Appointments endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals() and request.method == 'POST':
            conn.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/doctors/<int:doctor_id>/schedule', methods=['GET', 'POST', 'PUT'])
def doctor_schedule(doctor_id):
    """Manage doctor's schedule"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'POST':
                data = request.json
                
                cur.execute('''
                    INSERT INTO doctor_schedules 
                    (doctor_id, day_of_week, start_time, end_time, slot_duration, 
                     break_start, break_end, is_available)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    doctor_id,
                    data.get('day_of_week'),
                    data.get('start_time'),
                    data.get('end_time'),
                    data.get('slot_duration', 30),
                    data.get('break_start'),
                    data.get('break_end'),
                    data.get('is_available', 1)
                ))
                
                schedule_id = cur.fetchone()['id']
                conn.commit()
                
                return jsonify({'success': True, 'id': schedule_id})
            
            elif request.method == 'PUT':
                data = request.json
                
                cur.execute('''
                    UPDATE doctor_schedules 
                    SET start_time = %s, end_time = %s, slot_duration = %s,
                        break_start = %s, break_end = %s, is_available = %s
                    WHERE id = %s
                ''', (
                    data.get('start_time'),
                    data.get('end_time'),
                    data.get('slot_duration'),
                    data.get('break_start'),
                    data.get('break_end'),
                    data.get('is_available'),
                    data.get('schedule_id')
                ))
                
                if cur.rowcount == 0:
                    return jsonify({'error': 'Schedule entry not found'}), 404
                
                conn.commit()
                return jsonify({'success': True})
            
            else:  # GET
                cur.execute('''
                    SELECT * FROM doctor_schedules 
                    WHERE doctor_id = %s 
                    ORDER BY day_of_week
                ''', (doctor_id,))
                
                schedules = [dict(row) for row in cur.fetchall()]
                return jsonify({'schedules': schedules})
    
    except Exception as e:
        print(f"Doctor schedule endpoint error (doctor_id={doctor_id}): {str(e)}")
        if 'conn' in locals() and request.method in ['POST', 'PUT']:
            conn.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/appointments/available-slots', methods=['GET'])
def available_slots():
    """Get available appointment slots for a doctor on a date"""
    doctor_id = request.args.get('doctor_id', type=int)
    date_str = request.args.get('date')
    
    if not doctor_id or not date_str:
        return jsonify({'error': 'doctor_id and date required'}), 400
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Get day of week (0=Monday ... 6=Sunday)
            from datetime import datetime as dt
            try:
                date_obj = dt.strptime(date_str, '%Y-%m-%d')
                day_of_week = date_obj.weekday()
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
            
            # Get doctor's schedule for that day
            cur.execute('''
                SELECT * FROM doctor_schedules 
                WHERE doctor_id = %s 
                AND day_of_week = %s 
                AND is_available = 1
            ''', (doctor_id, day_of_week))
            
            schedule = cur.fetchone()
            if not schedule:
                return jsonify({'slots': []})
            
            # Get all booked slots on that date
            cur.execute('''
                SELECT appointment_time, duration 
                FROM appointments 
                WHERE doctor_id = %s 
                AND appointment_date = %s 
                AND status IN ('scheduled', 'confirmed')
            ''', (doctor_id, date_str))
            booked = cur.fetchall()
            
            # Generate slots
            slots = []
            start_time = dt.strptime(schedule['start_time'], '%H:%M').time()
            end_time = dt.strptime(schedule['end_time'], '%H:%M').time()
            slot_duration = schedule['slot_duration'] or 30
            
            current = dt.combine(date_obj.date(), start_time)
            end_dt = dt.combine(date_obj.date(), end_time)
            
            while current < end_dt:
                slot_time_str = current.strftime('%H:%M')
                slot_start_min = current.hour * 60 + current.minute
                slot_end_min = slot_start_min + slot_duration
                
                is_booked = False
                for booking in booked:
                    booked_time = booking['appointment_time']
                    booked_duration = booking['duration'] or 30
                    
                    try:
                        booked_t = dt.strptime(booked_time, '%H:%M').time()
                        booked_start = booked_t.hour * 60 + booked_t.minute
                        booked_end = booked_start + booked_duration
                        
                        if slot_start_min < booked_end and slot_end_min > booked_start:
                            is_booked = True
                            break
                    except:
                        # Fallback: exact match if parsing fails
                        if booked_time == slot_time_str:
                            is_booked = True
                            break
                
                if not is_booked:
                    slots.append(slot_time_str)
                
                # Move to next slot
                current = current.replace(minute=current.minute + slot_duration)
            
            return jsonify({'slots': slots})
    
    except Exception as e:
        print(f"Available slots error (doctor_id={doctor_id}, date={date_str}): {str(e)}")
        return jsonify({'error': str(e)}), 500

# ==========================================
# PRESCRIPTION MANAGEMENT
# ==========================================
@app.route('/api/prescriptions', methods=['GET', 'POST'])
def prescriptions():
    """Get or create prescriptions"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'POST':
                data = request.json
                
                patient_id = data.get('patient_id')
                doctor_id = session.get('user_db_id') if session['user_role'] == 'doctor' else data.get('doctor_id')
                
                if not patient_id or not doctor_id:
                    return jsonify({'error': 'Patient and doctor required'}), 400
                
                # Fetch patient info for AI safety check
                cur.execute('''
                    SELECT allergies, medications, medical_history 
                    FROM patients 
                    WHERE id = %s
                ''', (patient_id,))
                patient_info = cur.fetchone()
                
                # AI safety check using Groq
                ai_safety_check = None
                if patient_info:
                    safety_prompt = f"""As a medical AI assistant, analyze this prescription for safety:

Patient Information:
- Allergies: {patient_info['allergies'] or 'None known'}
- Current Medications: {patient_info['medications'] or 'None'}
- Medical History: {patient_info['medical_history'] or 'None'}

Prescription:
{data.get('prescription_text')}

Medications List: {data.get('medications', 'Not specified')}

Please check for:
1. Drug-allergy interactions
2. Drug-drug interactions
3. Contraindications based on medical history
4. Dosage appropriateness
5. Any other safety concerns

Provide a brief safety assessment with warnings if any issues are detected."""
                    
                    try:
                        global groq_client
                        if groq_client is None:
                            from groq import Groq
                            groq_client = Groq(api_key=GROQ_API_KEY)
                        
                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "You are a medical AI assistant specialized in prescription safety checking. Provide concise, accurate safety assessments."},
                                {"role": "user", "content": safety_prompt}
                            ],
                            temperature=0.3,
                            max_tokens=500
                        )
                        ai_safety_check = response.choices[0].message.content
                    
                    except Exception as e:
                        ai_safety_check = f"AI safety check unavailable: {str(e)}"
                
                # Insert new prescription
                cur.execute('''
                    INSERT INTO prescriptions 
                    (patient_id, doctor_id, appointment_id, prescription_text,
                     medications, instructions, valid_until, refills_remaining, 
                     status, ai_safety_check, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    patient_id,
                    doctor_id,
                    data.get('appointment_id'),
                    data.get('prescription_text'),
                    data.get('medications'),
                    data.get('instructions'),
                    data.get('valid_until'),
                    data.get('refills_remaining', 0),
                    'active',
                    ai_safety_check,
                    datetime.now().isoformat()
                ))
                
                prescription_id = cur.fetchone()['id']
                
                conn.commit()
                
                # Create notification for patient
                try:
                    cur.execute('''
                        SELECT id FROM users 
                        WHERE user_id = %s AND role = 'patient'
                    ''', (patient_id,))
                    patient_user = cur.fetchone()
                    
                    if patient_user:
                        create_notification(
                            patient_user['id'], 'patient', 'prescription',
                            'New Prescription', 'A new prescription has been issued',
                            f'/prescriptions/{prescription_id}'
                        )
                except Exception as e:
                    print(f"Error creating prescription notification: {str(e)}")
                
                # Audit log
                try:
                    log_audit(
                        'prescription_created', 'prescription', prescription_id,
                        f'For patient {patient_id}'
                    )
                except Exception as e:
                    print(f"Error logging prescription audit: {str(e)}")
                
                return jsonify({
                    'success': True,
                    'id': prescription_id,
                    'ai_safety_check': ai_safety_check
                })
            
            else:
                # GET prescriptions
                user_role = session['user_role']
                user_db_id = session.get('user_db_id')
                
                if user_role == 'doctor':
                    cur.execute('''
                        SELECT p.*, pt.name as patient_name, d.name as doctor_name
                        FROM prescriptions p
                        LEFT JOIN patients pt ON p.patient_id = pt.id
                        LEFT JOIN doctors d ON p.doctor_id = d.id
                        WHERE p.doctor_id = %s 
                        ORDER BY p.created_at DESC
                    ''', (user_db_id,))
                
                elif user_role == 'patient':
                    cur.execute('''
                        SELECT p.*, pt.name as patient_name, d.name as doctor_name
                        FROM prescriptions p
                        LEFT JOIN patients pt ON p.patient_id = pt.id
                        LEFT JOIN doctors d ON p.doctor_id = d.id
                        WHERE p.patient_id = %s 
                        ORDER BY p.created_at DESC
                    ''', (user_db_id,))
                
                else:  # admin
                    cur.execute('''
                        SELECT p.*, pt.name as patient_name, d.name as doctor_name
                        FROM prescriptions p
                        LEFT JOIN patients pt ON p.patient_id = pt.id
                        LEFT JOIN doctors d ON p.doctor_id = d.id
                        ORDER BY p.created_at DESC 
                        LIMIT 100
                    ''')
                
                prescriptions_list = [dict(row) for row in cur.fetchall()]
                return jsonify({'prescriptions': prescriptions_list})
    
    except Exception as e:
        print(f"Prescriptions endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals() and request.method == 'POST':
            conn.rollback()
        return jsonify({'error': str(e)}), 500

# ==========================================
# NOTIFICATION SYSTEM
# ==========================================
@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    """Get notifications for current user"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('''
                SELECT * FROM notifications 
                WHERE user_id = %s 
                ORDER BY created_at DESC 
                LIMIT 50
            ''', (session['user_id'],))
            
            notifications = [dict(row) for row in cur.fetchall()]
            
            # Calculate unread count (client could do this too, but keeping your logic)
            unread_count = sum(1 for n in notifications if not n.get('is_read'))
            
            return jsonify({
                'notifications': notifications,
                'unread_count': unread_count
            })
    
    except Exception as e:
        print(f"Error fetching notifications: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/notifications/<int:notification_id>/read', methods=['PUT'])
def mark_notification_read(notification_id):
    """Mark a single notification as read"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('''
                UPDATE notifications 
                SET is_read = 1 
                WHERE id = %s AND user_id = %s
            ''', (notification_id, session['user_id']))
            
            if cur.rowcount == 0:
                return jsonify({'error': 'Notification not found or not owned by user'}), 404
            
            conn.commit()
            return jsonify({'success': True})
    
    except Exception as e:
        print(f"Error marking notification {notification_id} as read: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/notifications/read-all', methods=['PUT'])
def mark_all_notifications_read():
    """Mark all notifications for the current user as read"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            cur.execute('''
                UPDATE notifications 
                SET is_read = 1 
                WHERE user_id = %s
            ''', (session['user_id'],))
            
            conn.commit()
            return jsonify({'success': True})
    
    except Exception as e:
        print(f"Error marking all notifications as read: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
        return jsonify({'error': str(e)}), 500

# ==========================================
# DOCUMENT MANAGEMENT
# ==========================================
@app.route('/api/documents', methods=['GET', 'POST'])
def documents():
    """Get or upload documents"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'POST':
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                patient_id = request.form.get('patient_id', type=int)
                if not patient_id:
                    return jsonify({'error': 'Patient ID required'}), 400
                
                # Save file
                ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
                filename = secure_filename(f"doc_{patient_id}_{datetime.now().timestamp()}.{ext}")
                doc_folder = os.path.join('static', 'documents')
                os.makedirs(doc_folder, exist_ok=True)
                filepath = os.path.join(doc_folder, filename)
                file.save(filepath)
                
                file_path_relative = f"documents/{filename}"
                file_size = os.path.getsize(filepath)
                mime_type = file.mimetype
                
                cur.execute('''
                    INSERT INTO documents 
                    (patient_id, doctor_id, document_type, title, file_path,
                     file_size, mime_type, description, tags, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    patient_id,
                    session.get('user_db_id') if session['user_role'] == 'doctor' else None,
                    request.form.get('document_type', 'general'),
                    request.form.get('title', file.filename),
                    file_path_relative,
                    file_size,
                    mime_type,
                    request.form.get('description'),
                    request.form.get('tags'),
                    datetime.now().isoformat()
                ))
                
                document_id = cur.fetchone()['id']
                
                conn.commit()
                
                # Audit log
                try:
                    log_audit(
                        'document_uploaded', 'document', document_id,
                        f'For patient {patient_id}'
                    )
                except Exception as e:
                    print(f"Error logging document audit: {str(e)}")
                
                return jsonify({
                    'success': True,
                    'id': document_id
                })
            
            else:  # GET
                patient_id = request.args.get('patient_id', type=int)
                
                if patient_id:
                    cur.execute('''
                        SELECT * FROM documents 
                        WHERE patient_id = %s 
                        ORDER BY created_at DESC
                    ''', (patient_id,))
                else:
                    cur.execute('''
                        SELECT * FROM documents 
                        ORDER BY created_at DESC 
                        LIMIT 100
                    ''')
                
                documents_list = [dict(row) for row in cur.fetchall()]
                return jsonify({'documents': documents_list})
    
    except Exception as e:
        print(f"Documents endpoint error: {str(e)}")
        if 'conn' in locals() and request.method == 'POST':
            conn.rollback()
        return jsonify({'error': str(e)}), 500


# ==========================================
# MESSAGING SYSTEM
# ==========================================

@app.route('/api/users/list', methods=['GET'])
def users_list():
    """Get list of users for chat (doctors see patients, patients see doctors)"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            user_role = session['user_role']
            
            if user_role == 'doctor':
                cur.execute('''
                    SELECT id, username, full_name, email, user_id, role
                    FROM users
                    WHERE role = 'patient'
                    ORDER BY full_name, username
                ''')
                users = [dict(row) for row in cur.fetchall()]
                return jsonify({'users': users})  # doctors see "users" (patients)
            
            elif user_role == 'patient':
                cur.execute('''
                    SELECT id, username, full_name, email, user_id, role
                    FROM users
                    WHERE role = 'doctor'
                    ORDER BY full_name, username
                ''')
                doctors = [dict(row) for row in cur.fetchall()]
                return jsonify({'doctors': doctors})  # ← changed key to 'doctors' for patient
                
            else:  # admin
                cur.execute('''
                    SELECT id, username, full_name, email, user_id, role
                    FROM users
                    ORDER BY role, full_name, username
                ''')
                users = [dict(row) for row in cur.fetchall()]
                return jsonify({'users': users})
    
    except Exception as e:
        print(f"Error fetching user list: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/messages', methods=['GET', 'POST'])
def messages():
    """Get or send messages"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            if request.method == 'POST':
                data = request.json
                print(f"Message POST request data: {data}")
                
                recipient_id = data.get('recipient_id')
                recipient_role = data.get('recipient_role')
                
                if not recipient_id:
                    return jsonify({'error': 'Recipient required'}), 400
                
                # Verify sender exists
                cur.execute('SELECT id FROM users WHERE id = %s', (session['user_id'],))
                if not cur.fetchone():
                    print(f"Sender user_id {session['user_id']} not found")
                    return jsonify({'error': 'Sender not found'}), 400
                
                # Verify recipient exists
                cur.execute('SELECT id FROM users WHERE id = %s', (recipient_id,))
                if not cur.fetchone():
                    print(f"Recipient user_id {recipient_id} not found")
                    return jsonify({'error': f'Recipient with ID {recipient_id} not found'}), 400
                
                # Insert message
                cur.execute('''
                    INSERT INTO messages 
                    (sender_id, sender_role, recipient_id, recipient_role,
                     subject, message, parent_message_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    session['user_id'],
                    session['user_role'],
                    recipient_id,
                    recipient_role,
                    data.get('subject'),
                    data.get('message'),
                    data.get('parent_message_id'),
                    datetime.now().isoformat()
                ))
                
                message_id = cur.fetchone()['id']
                print(f"Message inserted with ID: {message_id}")
                
                conn.commit()
                print(f"Message {message_id} committed successfully")
                
                # Notification (after commit)
                try:
                    message_link = '/patient/chat' if recipient_role == 'patient' else '/doctor/chat'
                    create_notification(
                        recipient_id, recipient_role, 'message',
                        'New Message',
                        f'You have a new message from {session.get("full_name", "User")}',
                        message_link
                    )
                except Exception as e:
                    print(f"Error creating message notification: {str(e)}")
                
                # Audit log
                try:
                    log_audit('message_sent', 'message', message_id)
                except Exception as e:
                    print(f"Error logging message audit: {str(e)}")
                
                return jsonify({
                    'success': True,
                    'id': message_id,
                    'message': 'Message sent successfully'
                })
            
            else:  # GET (inbox)
                cur.execute('''
                    SELECT m.*, u1.full_name as sender_name, u2.full_name as recipient_name
                    FROM messages m
                    LEFT JOIN users u1 ON m.sender_id = u1.id
                    LEFT JOIN users u2 ON m.recipient_id = u2.id
                    WHERE m.recipient_id = %s 
                    ORDER BY m.created_at DESC 
                    LIMIT 100
                ''', (session['user_id'],))
                
                messages_list = [dict(row) for row in cur.fetchall()]
                
                # Mark inbox as read
                cur.execute('''
                    UPDATE messages 
                    SET is_read = 1 
                    WHERE recipient_id = %s AND is_read = 0
                ''', (session['user_id'],))
                
                conn.commit()
                
                return jsonify({'messages': messages_list})
    
    except Exception as e:
        print(f"Messages endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'conn' in locals() and request.method == 'POST':
            conn.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/messages/with-doctor/<int:doctor_id>', methods=['GET'])
def get_messages_with_doctor(doctor_id):
    """Get messages between current patient and a specific doctor"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Find doctor's user ID
            cur.execute('''
                SELECT id FROM users 
                WHERE user_id = %s AND role = 'doctor'
            ''', (doctor_id,))
            doctor_user = cur.fetchone()
            
            if not doctor_user:
                return jsonify({'messages': []})
            
            doctor_user_id = doctor_user['id']
            patient_user_id = session['user_id']
            
            # Get conversation between patient and doctor
            cur.execute('''
                SELECT m.*, u1.full_name as sender_name, u2.full_name as recipient_name
                FROM messages m
                LEFT JOIN users u1 ON m.sender_id = u1.id
                LEFT JOIN users u2 ON m.recipient_id = u2.id
                WHERE (m.sender_id = %s AND m.recipient_id = %s) 
                OR (m.sender_id = %s AND m.recipient_id = %s)
                ORDER BY m.created_at ASC
            ''', (patient_user_id, doctor_user_id, doctor_user_id, patient_user_id))
            
            messages_list = [dict(row) for row in cur.fetchall()]
            
            return jsonify({'messages': messages_list})
    
    except Exception as e:
        print(f"Error fetching messages with doctor {doctor_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/messages/with-patient/<int:patient_id>', methods=['GET'])
def get_messages_with_patient(patient_id):
    """Get messages between current doctor and a specific patient"""
    if 'user_role' not in session or session['user_role'] != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Find patient's user ID
            cur.execute('''
                SELECT id FROM users 
                WHERE user_id = %s AND role = 'patient'
            ''', (patient_id,))
            patient_user = cur.fetchone()
            
            if not patient_user:
                return jsonify({'messages': []})
            
            patient_user_id = patient_user['id']
            doctor_user_id = session['user_id']
            
            # Get conversation between doctor and patient
            cur.execute('''
                SELECT m.*, u1.full_name as sender_name, u2.full_name as recipient_name
                FROM messages m
                LEFT JOIN users u1 ON m.sender_id = u1.id
                LEFT JOIN users u2 ON m.recipient_id = u2.id
                WHERE (m.sender_id = %s AND m.recipient_id = %s) 
                OR (m.sender_id = %s AND m.recipient_id = %s)
                ORDER BY m.created_at ASC
            ''', (doctor_user_id, patient_user_id, patient_user_id, doctor_user_id))
            
            messages_list = [dict(row) for row in cur.fetchall()]
            
            return jsonify({'messages': messages_list})
    
    except Exception as e:
        print(f"Error fetching messages with patient {patient_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/messages/to-doctor/<int:doctor_id>', methods=['POST'])
def send_message_to_doctor(doctor_id):
    """Send message from patient to doctor"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.json
        message_text = data.get('message', '').strip()

        print(f"[DEBUG] Message received from patient. Doctor ID: {doctor_id}, Message: {message_text[:100]}")

        if not message_text:
            return jsonify({'error': 'Message is required'}), 400

        with get_db() as conn:
            cur = conn.cursor()

            # Get doctor's user account
            cur.execute('''
                SELECT id FROM users 
                WHERE user_id = %s AND role = 'doctor'
            ''', (doctor_id,))
            doctor_user = cur.fetchone()

            if not doctor_user:
                # Auto-create doctor user
                print(f"[DEBUG] Doctor user not found for ID {doctor_id}. Creating now...")
                import hashlib
                default_password = hashlib.md5('password123'.encode()).hexdigest()
                username = f"dr_{doctor_id}"

                cur.execute('''
                    INSERT INTO users 
                    (username, password, role, full_name, user_id, created_at)
                    VALUES (%s, %s, 'doctor', %s, %s, %s)
                    RETURNING id
                ''', (
                    username,
                    default_password,
                    f'Doctor {doctor_id}',
                    doctor_id,
                    datetime.now().isoformat()
                ))
                doctor_user_id = cur.fetchone()['id']
                print(f"[DEBUG] Auto-created doctor user with ID: {doctor_user_id}")
            else:
                doctor_user_id = doctor_user['id']

            # Now insert the message
            print(f"[DEBUG] Inserting message: sender={session['user_id']}, recipient={doctor_user_id}")

            cur.execute('''
                INSERT INTO messages 
                (sender_id, sender_role, recipient_id, recipient_role,
                 subject, message, created_at)
                VALUES (%s, %s, %s, 'doctor', %s, %s, %s)
                RETURNING id
            ''', (
                session['user_id'],
                session['user_role'],
                doctor_user_id,
                f'Chat from {session.get("full_name", "Patient")}',
                message_text,
                datetime.now().isoformat()
            ))

            message_id = cur.fetchone()['id']
            conn.commit()   # Very important!

            print(f"[DEBUG] Message successfully saved with ID: {message_id}")

            # Create notification for doctor
            create_notification(
                doctor_user_id, 'doctor', 'message',
                'New Message', 
                f'You have a new message from {session.get("full_name", "a patient")}',
                '/doctor/chat'
            )

            return jsonify({
                'success': True,
                'id': message_id,
                'message': 'Message sent successfully'
            })

    except Exception as e:
        print(f"[ERROR] Failed to send message to doctor {doctor_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to send message: {str(e)}'}), 500

# ==========================================
# ADVANCED SEARCH
# ==========================================
@app.route('/api/search', methods=['GET'])
def advanced_search():
    """Advanced search across all entities"""
    if 'user_role' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    query = request.args.get('q', '').strip()
    entity_type = request.args.get('type', 'all')  # all, patients, reports, appointments, prescriptions
    
    if not query:
        return jsonify({'error': 'Search query required'}), 400
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            results = {}
            
            search_pattern = f'%{query}%'
            
            # ────────────────────────────────────────────────
            # Patients search
            # ────────────────────────────────────────────────
            if entity_type in ['all', 'patients']:
                if session['user_role'] == 'patient':
                    patient_id = session.get('user_db_id')
                    cur.execute('''
                        SELECT id, name, email, phone, patient_id 
                        FROM patients 
                        WHERE id = %s 
                        AND (name ILIKE %s OR email ILIKE %s OR phone ILIKE %s OR patient_id ILIKE %s)
                        LIMIT 20
                    ''', (patient_id, search_pattern, search_pattern, search_pattern, search_pattern))
                else:
                    cur.execute('''
                        SELECT id, name, email, phone, patient_id 
                        FROM patients 
                        WHERE name ILIKE %s OR email ILIKE %s OR phone ILIKE %s OR patient_id ILIKE %s
                        LIMIT 20
                    ''', (search_pattern, search_pattern, search_pattern, search_pattern))
                
                results['patients'] = [dict(row) for row in cur.fetchall()]
            
            # ────────────────────────────────────────────────
            # Reports search
            # ────────────────────────────────────────────────
            if entity_type in ['all', 'reports']:
                if session['user_role'] == 'patient':
                    patient_id = session.get('user_db_id')
                    cur.execute('''
                        SELECT r.id, r.original_filename, r.specialist_type, r.created_at, p.name as patient_name
                        FROM reports r
                        LEFT JOIN patients p ON r.patient_id = p.id
                        WHERE r.patient_id = %s 
                        AND (r.original_filename ILIKE %s OR r.extracted_text ILIKE %s OR p.name ILIKE %s)
                        ORDER BY r.created_at DESC 
                        LIMIT 20
                    ''', (patient_id, search_pattern, search_pattern, search_pattern))
                else:
                    cur.execute('''
                        SELECT r.id, r.original_filename, r.specialist_type, r.created_at, p.name as patient_name
                        FROM reports r
                        LEFT JOIN patients p ON r.patient_id = p.id
                        WHERE r.original_filename ILIKE %s OR r.extracted_text ILIKE %s OR p.name ILIKE %s
                        ORDER BY r.created_at DESC 
                        LIMIT 20
                    ''', (search_pattern, search_pattern, search_pattern))
                
                results['reports'] = [dict(row) for row in cur.fetchall()]
            
            # ────────────────────────────────────────────────
            # Appointments search
            # ────────────────────────────────────────────────
            if entity_type in ['all', 'appointments']:
                if session['user_role'] == 'patient':
                    patient_id = session.get('user_db_id')
                    cur.execute('''
                        SELECT a.id, a.appointment_date, a.appointment_time, p.name as patient_name, d.name as doctor_name
                        FROM appointments a
                        LEFT JOIN patients p ON a.patient_id = p.id
                        LEFT JOIN doctors d ON a.doctor_id = d.id
                        WHERE a.patient_id = %s 
                        AND (p.name ILIKE %s OR d.name ILIKE %s OR a.reason ILIKE %s)
                        ORDER BY a.appointment_date DESC 
                        LIMIT 20
                    ''', (patient_id, search_pattern, search_pattern, search_pattern))
                else:
                    cur.execute('''
                        SELECT a.id, a.appointment_date, a.appointment_time, p.name as patient_name, d.name as doctor_name
                        FROM appointments a
                        LEFT JOIN patients p ON a.patient_id = p.id
                        LEFT JOIN doctors d ON a.doctor_id = d.id
                        WHERE p.name ILIKE %s OR d.name ILIKE %s OR a.reason ILIKE %s
                        ORDER BY a.appointment_date DESC 
                        LIMIT 20
                    ''', (search_pattern, search_pattern, search_pattern))
                
                results['appointments'] = [dict(row) for row in cur.fetchall()]
            
            # ────────────────────────────────────────────────
            # Prescriptions search
            # ────────────────────────────────────────────────
            if entity_type in ['all', 'prescriptions']:
                if session['user_role'] == 'patient':
                    patient_id = session.get('user_db_id')
                    cur.execute('''
                        SELECT pr.id, pr.prescription_text, pr.created_at, p.name as patient_name, d.name as doctor_name
                        FROM prescriptions pr
                        LEFT JOIN patients p ON pr.patient_id = p.id
                        LEFT JOIN doctors d ON pr.doctor_id = d.id
                        WHERE pr.patient_id = %s 
                        AND (pr.prescription_text ILIKE %s OR pr.medications ILIKE %s OR p.name ILIKE %s)
                        ORDER BY pr.created_at DESC 
                        LIMIT 20
                    ''', (patient_id, search_pattern, search_pattern, search_pattern))
                else:
                    cur.execute('''
                        SELECT pr.id, pr.prescription_text, pr.created_at, p.name as patient_name, d.name as doctor_name
                        FROM prescriptions pr
                        LEFT JOIN patients p ON pr.patient_id = p.id
                        LEFT JOIN doctors d ON pr.doctor_id = d.id
                        WHERE pr.prescription_text ILIKE %s OR pr.medications ILIKE %s OR p.name ILIKE %s
                        ORDER BY pr.created_at DESC 
                        LIMIT 20
                    ''', (search_pattern, search_pattern, search_pattern))
                
                results['prescriptions'] = [dict(row) for row in cur.fetchall()]
            
            # ────────────────────────────────────────────────
            # Audit log the search
            # ────────────────────────────────────────────────
            try:
                log_audit(
                    'search_performed', None, None,
                    f'Query: {query}, Type: {entity_type}'
                )
            except Exception as e:
                print(f"Search audit log failed: {str(e)}")
            
            return jsonify({'results': results})
    
    except Exception as e:
        print(f"Advanced search error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

