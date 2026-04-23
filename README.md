# Neura-X - Advanced Hospital Management System (HMS)

A comprehensive, AI-powered Hospital Management System built with Flask that streamlines hospital operations, enhances patient care, and provides intelligent medical report analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Database Schema](#database-schema)
- [API Endpoints](#api-endpoints)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

ObrixLabs HMS is a full-featured hospital management system designed to modernize healthcare operations. It combines traditional hospital management features with cutting-edge AI capabilities for medical report analysis, patient care optimization, and clinical decision support.

### Key Highlights

- **AI-Powered Analysis**: Advanced medical report analysis using GROQ LLM and Clinical BERT models
- **Multi-Specialist Support**: Supports 5 specialist types (Cardiac, Radiologist, Neurologist, Dermatologist, Oncologist)
- **Comprehensive Patient Management**: Complete patient profiles with medical history, vitals tracking, and timeline
- **Secure Communication**: Internal messaging system for doctors, patients, and staff
- **Appointment Scheduling**: Automated appointment booking and management
- **Real-time Analytics**: Dashboard with charts, statistics, and custom reports
- **HIPAA Compliant**: Security-focused design with audit trails

## ✨ Features

### Core Features

1. **AI Medical Report Analysis**
   - Automatic text extraction from PDF and image files
   - Dual AI analysis: GROQ LLM + Clinical BERT
   - Anomaly detection and treatment recommendations
   - Specialist-specific analysis prompts

2. **Patient Management**
   - Comprehensive patient profiles
   - Medical history tracking
   - Allergies and medications management
   - Patient vitals tracking and trends
   - Patient tags and categorization
   - Profile pictures and demographics

3. **Doctor Management**
   - Multi-specialist support
   - Doctor profiles with credentials
   - Specialist-specific dashboards
   - Report sharing between doctors
   - Task management system

4. **Appointment System**
   - Online appointment booking
   - Doctor schedule management
   - Available slot detection
   - Appointment reminders
   - Status tracking (scheduled, completed, cancelled)

5. **Prescription Management**
   - Digital prescription creation
   - Medication tracking
   - AI safety checks
   - Refill management
   - Prescription history

6. **Communication & Collaboration**
   - Internal messaging system
   - Report comments and annotations
   - Patient referrals between specialists
   - Task assignments
   - Notification system

7. **Analytics & Reporting**
   - Real-time dashboard statistics
   - Reports by specialist charts
   - Time-series analysis
   - Custom analytics reports
   - Export capabilities (PDF)

8. **Document Management**
   - Secure file uploads
   - Document categorization
   - Tag-based organization
   - File type support (PDF, PNG, JPG, JPEG)

9. **Admin Panel**
   - User management
   - System statistics
   - Doctor management
   - Report oversight
   - Audit logs

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Layer                          │
│  (HTML Templates + Tailwind CSS + JavaScript)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Flask Application                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Routes     │  │   Business   │  │   Database   │      │
│  │  (58+ APIs)  │  │    Logic     │  │   Layer      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│   Supabase   │ │   GROQ     │ │  Clinical  │
│  Database    │ │    LLM     │ │    BERT    │
└──────────────┘ └────────────┘ └────────────┘
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐
│   OCR        │ │  File      │
│  (PaddleOCR/ │ │  Storage   │
│  Tesseract)  │ │            │
└──────────────┘ └────────────┘
```

### Request Flow

1. **User Request** → Flask Route Handler
2. **Authentication** → Session-based authentication
3. **Business Logic** → Process request, validate data
4. **Database Operations** → SQLite queries
5. **AI Processing** (if needed) → GROQ/Clinical BERT
6. **Response** → JSON or HTML template

### User Roles

- **Admin**: Full system access, user management, analytics
- **Doctor**: Patient management, report analysis, appointments, prescriptions
- **Patient**: View reports, book appointments, view prescriptions, messaging

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask (Python)
- **Database**: SQLite3
- **AI/ML**:
  - GROQ LLM API (for medical report analysis)
  - Clinical BERT (emilyalsentzer/Bio_ClinicalBERT)
- **OCR**: 
  - PaddleOCR (primary)
  - pytesseract (fallback)
- **PDF Processing**: PyMuPDF (fitz)
- **Image Processing**: PIL/Pillow

### Frontend
- **Styling**: Tailwind CSS (CDN)
- **Charts**: Chart.js
- **Icons**: Heroicons (SVG)
- **JavaScript**: Vanilla JS (ES6+)

### Additional Libraries
- `python-dotenv`: Environment variable management
- `werkzeug`: File uploads and security
- `reportlab`: PDF generation

## 📊 Database Schema

### Core Tables

#### `users`
User authentication and basic information.

```
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'patient',
    email TEXT,
    full_name TEXT,
    user_id INTEGER,
    created_at TEXT NOT NULL,
    last_login TEXT
)
```

#### `doctors`
Doctor profiles and specialist information.

```
CREATE TABLE doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    profile_picture TEXT,
    specialist_type TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    license_number TEXT,
    created_at TEXT NOT NULL
)
```

#### `patients`
Comprehensive patient information.

```
CREATE TABLE patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    patient_id TEXT UNIQUE,
    profile_picture TEXT,
    date_of_birth TEXT,
    gender TEXT,
    address TEXT,
    contact_info TEXT,
    medical_history TEXT,
    allergies TEXT,
    medications TEXT,
    patient_tags TEXT,
    created_at TEXT NOT NULL
)
```

#### `reports`
Medical reports with AI analysis.

```
CREATE TABLE reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER,
    patient_id INTEGER NOT NULL,
    specialist_type TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    extracted_text TEXT,
    llm_analysis TEXT,
    clinical_bert_analysis TEXT,
    status TEXT DEFAULT "new",
    doctor_notes TEXT,
    tags TEXT,
    is_public INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (doctor_id) REFERENCES doctors (id),
    FOREIGN KEY (patient_id) REFERENCES patients (id)
)
```

#### `appointments`
Appointment scheduling and management.

```
CREATE TABLE appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    doctor_id INTEGER NOT NULL,
    appointment_date TEXT NOT NULL,
    appointment_time TEXT NOT NULL,
    duration INTEGER DEFAULT 30,
    appointment_type TEXT DEFAULT "consultation",
    reason TEXT,
    status TEXT DEFAULT "scheduled",
    notes TEXT,
    reminder_sent INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients (id),
    FOREIGN KEY (doctor_id) REFERENCES doctors (id)
)
```

#### `prescriptions`
Digital prescription management.

```
CREATE TABLE prescriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    doctor_id INTEGER NOT NULL,
    appointment_id INTEGER,
    prescription_text TEXT NOT NULL,
    medications TEXT,
    instructions TEXT,
    valid_until TEXT,
    refills_remaining INTEGER DEFAULT 0,
    status TEXT DEFAULT "active",
    ai_safety_check TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients (id),
    FOREIGN KEY (doctor_id) REFERENCES doctors (id),
    FOREIGN KEY (appointment_id) REFERENCES appointments (id)
)
```

#### `messages`
Internal messaging system.

```
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id INTEGER NOT NULL,
    sender_role TEXT NOT NULL,
    recipient_id INTEGER NOT NULL,
    recipient_role TEXT NOT NULL,
    subject TEXT,
    message TEXT NOT NULL,
    is_read INTEGER DEFAULT 0,
    parent_message_id INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (sender_id) REFERENCES users (id),
    FOREIGN KEY (recipient_id) REFERENCES users (id),
    FOREIGN KEY (parent_message_id) REFERENCES messages (id)
)
```

#### `tasks`
Task management for doctors.

```
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER,
    patient_id INTEGER,
    report_id INTEGER,
    task_type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    due_date TEXT,
    status TEXT DEFAULT "pending",
    priority TEXT DEFAULT "medium",
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (doctor_id) REFERENCES doctors (id),
    FOREIGN KEY (patient_id) REFERENCES patients (id),
    FOREIGN KEY (report_id) REFERENCES reports (id)
)
```

#### `referrals`
Patient referrals between specialists.

```
CREATE TABLE referrals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    from_doctor_id INTEGER NOT NULL,
    to_specialist_type TEXT NOT NULL,
    to_doctor_id INTEGER,
    reason TEXT,
    notes TEXT,
    status TEXT DEFAULT "pending",
    created_at TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients (id),
    FOREIGN KEY (from_doctor_id) REFERENCES doctors (id),
    FOREIGN KEY (to_doctor_id) REFERENCES doctors (id)
)
```

#### `documents`
Document management system.

```
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    doctor_id INTEGER,
    document_type TEXT NOT NULL,
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    mime_type TEXT,
    description TEXT,
    tags TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients (id),
    FOREIGN KEY (doctor_id) REFERENCES doctors (id)
)
```

#### `notifications`
System notifications.

```
CREATE TABLE notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    user_role TEXT NOT NULL,
    notification_type TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    link TEXT,
    is_read INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
```

#### `patient_vitals`
Patient vital signs tracking.

```
CREATE TABLE patient_vitals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    report_id INTEGER,
    vital_name TEXT NOT NULL,
    vital_value TEXT NOT NULL,
    unit TEXT,
    measured_at TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients (id),
    FOREIGN KEY (report_id) REFERENCES reports (id)
)
```

#### `chat_messages`
AI chatbot conversation history.

```
CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id INTEGER,
    specialist_type TEXT NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (doctor_id) REFERENCES doctors (id)
)
```

#### Additional Tables
- `shared_reports`: Report sharing between doctors
- `report_comments`: Comments and annotations on reports
- `audit_logs`: System audit trail

## 🔌 API Endpoints

### Authentication & User Management

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Landing page | No |
| GET | `/login` | Login page | No |
| POST | `/login` | User login | No |
| GET | `/signup` | Signup page | No |
| POST | `/signup` | User registration | No |
| GET | `/logout` | User logout | Yes |
| GET | `/profile` | User profile page | Yes |
| GET | `/api/user/profile` | Get user profile | Yes |
| PUT | `/api/user/profile` | Update user profile | Yes |

### Dashboard & Statistics

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/doctor/dashboard` | Doctor dashboard | Doctor |
| GET | `/patient/dashboard` | Patient dashboard | Patient |
| GET | `/admin/dashboard` | Admin dashboard | Admin |
| GET | `/api/dashboard/stats` | Dashboard statistics | Yes |
| GET | `/api/admin/stats` | Admin statistics | Admin |

### Reports Management

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/specialist/<specialist_type>` | Specialist reports page | Doctor |
| POST | `/upload` | Upload medical report | Doctor |
| GET | `/report/<report_id>` | View report details | Yes |
| GET | `/api/reports/<specialist_type>` | Get reports by specialist | Doctor |
| GET | `/api/report/<report_id>` | Get report data | Yes |
| PUT | `/api/report/<report_id>` | Update report | Doctor |
| GET | `/api/report/<report_id>/export` | Export report as PDF | Yes |
| POST | `/api/reports/<report_id>/share` | Share report with doctor | Doctor |
| GET | `/api/search/reports` | Search reports | Doctor |
| GET | `/api/analytics/reports` | Generate analytics | Doctor |

### Patient Management

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/patient/<patient_id>` | Patient profile page | Yes |
| GET | `/api/patients/search` | Search patients | Doctor |
| GET | `/api/patient/<patient_id>` | Get patient data | Yes |
| PUT | `/api/patient/<patient_id>` | Update patient | Doctor |
| PUT | `/api/patient/<patient_id>/update` | Update patient profile | Doctor |
| GET | `/api/patient/<patient_id>/reports` | Get patient reports | Yes |
| GET | `/api/patient/<patient_id>/vitals` | Get patient vitals | Doctor |
| POST | `/api/patient/<patient_id>/vitals` | Add patient vitals | Doctor |
| POST | `/api/patient/<patient_id>/risk-score` | Calculate risk score | Doctor |

### Doctor Management

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/doctors` | List all doctors | Yes |
| GET | `/api/doctors/<doctor_id>/schedule` | Get doctor schedule | Yes |
| POST | `/api/doctors/<doctor_id>/schedule` | Create schedule slot | Doctor |
| PUT | `/api/doctors/<doctor_id>/schedule` | Update schedule | Doctor |

### Appointments

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/doctor/appointments` | Doctor appointments page | Doctor |
| GET | `/patient/appointments` | Patient appointments page | Patient |
| GET | `/api/appointments` | List appointments | Yes |
| POST | `/api/appointments` | Create appointment | Yes |
| PUT | `/api/appointments/<appointment_id>` | Update appointment | Yes |
| DELETE | `/api/appointments/<appointment_id>` | Cancel appointment | Yes |
| GET | `/api/appointments/available-slots` | Get available slots | Yes |

### Prescriptions

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/prescriptions` | List prescriptions | Yes |
| POST | `/api/prescriptions` | Create prescription | Doctor |

### Tasks Management

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/tasks` | List tasks | Doctor |
| POST | `/api/tasks` | Create task | Doctor |
| PUT | `/api/tasks/<task_id>` | Update task | Doctor |
| DELETE | `/api/tasks/<task_id>` | Delete task | Doctor |

### Referrals

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/referrals` | List referrals | Doctor |
| POST | `/api/referrals` | Create referral | Doctor |

### Messaging

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/messages` | List messages | Yes |
| POST | `/api/messages` | Send message | Yes |
| GET | `/api/messages/with-doctor/<doctor_id>` | Messages with doctor | Yes |
| GET | `/api/messages/with-patient/<patient_id>` | Messages with patient | Doctor |

### Chatbot

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/doctor/chat` | Doctor chat page | Doctor |
| GET | `/patient/chat` | Patient chat page | Patient |
| POST | `/api/chat` | Send chat message | Yes |

### Documents

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/documents` | List documents | Yes |
| POST | `/api/documents` | Upload document | Yes |

### Notifications

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/notifications` | List notifications | Yes |
| PUT | `/api/notifications/<notification_id>/read` | Mark as read | Yes |
| PUT | `/api/notifications/read-all` | Mark all as read | Yes |

### Admin Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/admin/users` | List users | Admin |
| POST | `/api/admin/users` | Create user | Admin |
| PUT | `/api/admin/users/<user_id>` | Update user | Admin |
| DELETE | `/api/admin/users/<user_id>` | Delete user | Admin |
| GET | `/api/admin/doctors` | List doctors | Admin |
| GET | `/api/admin/reports` | List all reports | Admin |

### Search

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/search` | Global search | Yes |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Tesseract OCR (optional, for OCR fallback)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "FCP - AISB - Copy"
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install flask
pip install groq
pip install python-dotenv
pip install pillow
pip install PyMuPDF
pip install numpy

# Optional: For OCR functionality
pip install paddleocr
# OR
pip install pytesseract

# Optional: For Clinical BERT
pip install transformers torch

# Optional: For PDF export
pip install reportlab
```

### Step 4: Install Tesseract OCR (Optional)

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### Step 5: Create Required Directories

```bash
mkdir uploads
mkdir static/profiles
mkdir static/documents
```

### Step 6: Run Database Migration

The database will be automatically initialized on first run. The application includes automatic migration logic.

### Step 7: Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory (optional, as API key is hardcoded):

```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
```

### Application Configuration

Key configuration settings in `app.py`:

```python
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROFILE_FOLDER'] = 'static/profiles'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['SECRET_KEY'] = 'hms-secret-key-change-in-production'
```

### Specialist Types

Currently supported specialist types:

```python
SPECIALIST_TYPES = {
    'cardiac': 'Cardiac Specialist',
    'radiologist': 'Radiologist',
    'neurologist': 'Neurologist',
    'dermatologist': 'Dermatologist',
    'oncologist': 'Oncologist'
}
```

## 📖 Usage

### Initial Setup

1. **Start the application**: `python app.py`
2. **Access the landing page**: Navigate to `http://localhost:5000`
3. **Create an admin account**: Use the signup page or create directly in database
4. **Create doctor accounts**: Admin can create doctor accounts via admin panel
5. **Create patient accounts**: Patients can self-register or admin can create

### Uploading Medical Reports

1. Login as a doctor
2. Navigate to your specialist dashboard (e.g., `/specialist/cardiac`)
3. Click "Upload Report"
4. Select patient and upload PDF/image file
5. System will automatically:
   - Extract text using OCR
   - Analyze with GROQ LLM
   - Analyze with Clinical BERT (if available)
   - Save to database

### Managing Appointments

1. **Doctor**: Set available time slots in schedule
2. **Patient**: Browse available slots and book appointment
3. **Both**: View and manage appointments from dashboard

### Using the Chatbot

1. Navigate to `/doctor/chat` or `/patient/chat`
2. Select specialist type (for doctors)
3. Ask medical questions
4. AI provides contextual responses based on specialist type

### Generating Analytics

1. Login as doctor or admin
2. Navigate to dashboard
3. Click "Analytics" button
4. Select date range and specialist type
5. Generate custom reports

## 📁 Project Structure

```
FCP - AISB - Copy/
│
├── app.py                 # Main Flask application (4233 lines)
├── README.md             # This file
│
├── templates/            # HTML templates
│   ├── landing.html      # Landing page
│   ├── login.html        # Login page
│   ├── signup.html       # Signup page
│   ├── dashboard.html    # Doctor dashboard
│   ├── patient_dashboard.html
│   ├── admin_dashboard.html
│   ├── profile.html       # User profile
│   ├── specialist.html   # Specialist reports page
│   ├── report.html       # Report view page
│   ├── patient.html      # Patient profile page
│   └── ...
│
├── static/               # Static files
│   ├── style.css         # Custom CSS
│   ├── profiles/         # Profile pictures
│   └── documents/        # Document storage
│
└── uploads/             # Uploaded medical reports
```

## 🔒 Security

### Security Features

1. **Session Management**: Flask sessions with secret key
2. **Password Hashing**: Passwords stored securely (implement hashing in production)
3. **File Upload Validation**: File type and size restrictions
4. **SQL Injection Prevention**: Parameterized queries
5. **XSS Protection**: Template escaping
6. **Role-Based Access Control**: Route protection by user role
7. **Audit Logging**: System activity tracking

### Security Recommendations

⚠️ **Important**: Before deploying to production:

1. Change `SECRET_KEY` in `app.py`
2. Implement proper password hashing (bcrypt, argon2)
3. Use environment variables for sensitive data
4. Enable HTTPS
5. Implement rate limiting
6. Add CSRF protection
7. Regular security audits
8. Database encryption at rest
9. Implement MFA (Multi-Factor Authentication)
10. Regular backup strategy

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GROQ for LLM API
- Hugging Face for Clinical BERT model
- PaddleOCR team for OCR capabilities
- Flask community for the excellent framework
- Tailwind CSS for the beautiful UI framework

## 📧 Support

For support, email support@obrixlabs.com or create an issue in the repository.

## 🔮 Future Enhancements

- [ ] Real-time notifications (WebSocket)
- [ ] Mobile app (React Native)
- [ ] Telemedicine integration
- [ ] Electronic Health Records (EHR) integration
- [ ] Advanced AI diagnostics
- [ ] Multi-language support
- [ ] Cloud storage integration
- [ ] Advanced analytics with ML predictions
- [ ] Integration with medical devices
- [ ] Blockchain for medical records

---


