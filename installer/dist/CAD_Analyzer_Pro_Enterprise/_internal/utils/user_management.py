
import os
import sqlite3
import hashlib
import uuid
from datetime import datetime, timedelta
import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import secrets
import json
from typing import Dict, List, Optional, Any

class UserManager:
    """Professional user management system with authentication and authorization"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize user database with proper schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                company TEXT,
                department TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                email_verified BOOLEAN DEFAULT 0,
                profile_data TEXT,
                preferences TEXT
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                project_data TEXT,
                is_public BOOLEAN DEFAULT 0,
                tags TEXT,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (owner_id) REFERENCES users (id)
            )
        ''')
        
        # Project collaborators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_collaborators (
                project_id TEXT,
                user_id TEXT,
                role TEXT DEFAULT 'viewer',
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                permissions TEXT,
                PRIMARY KEY (project_id, user_id),
                FOREIGN KEY (project_id) REFERENCES projects (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                action TEXT,
                resource_type TEXT,
                resource_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                details TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Create default admin user if none exists
        self.create_default_admin()
    
    def create_default_admin(self):
        """Create default admin user"""
        if not self.user_exists("admin"):
            self.create_user(
                username="admin",
                email="admin@floorplan.app",
                password="admin123",
                role="admin",
                company="System",
                department="Administration"
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_value = password_hash.split(':')
            computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return secrets.compare_digest(hash_value, computed_hash.hex())
        except:
            return False
    
    def create_user(self, username: str, email: str, password: str, 
                   role: str = "user", company: str = "", department: str = "") -> Dict[str, Any]:
        """Create new user account"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            user_id = str(uuid.uuid4())
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (id, username, email, password_hash, role, company, department)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, username, email, password_hash, role, company, department))
            
            conn.commit()
            
            # Log user creation
            self.log_action(user_id, "user_created", "user", user_id, f"User {username} created")
            
            return {
                'success': True,
                'user_id': user_id,
                'message': 'User created successfully'
            }
            
        except sqlite3.IntegrityError as e:
            return {
                'success': False,
                'message': 'Username or email already exists'
            }
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, password_hash, role, company, department, is_active
            FROM users WHERE username = ? OR email = ?
        ''', (username, username))
        
        user = cursor.fetchone()
        
        if user and user[7] and self.verify_password(password, user[3]):
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user[0],))
            conn.commit()
            
            # Create session
            session_id = self.create_session(user[0])
            
            # Log login
            self.log_action(user[0], "user_login", "session", session_id, "User logged in")
            
            conn.close()
            
            return {
                'success': True,
                'user': {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2],
                    'role': user[4],
                    'company': user[5],
                    'department': user[6]
                },
                'session_id': session_id
            }
        
        conn.close()
        return {
            'success': False,
            'message': 'Invalid credentials'
        }
    
    def create_session(self, user_id: str) -> str:
        """Create user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=24)
        
        cursor.execute('''
            INSERT INTO sessions (session_id, user_id, expires_at)
            VALUES (?, ?, ?)
        ''', (session_id, user_id, expires_at))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.user_id, u.username, u.email, u.role, u.company, u.department
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_id = ? AND s.expires_at > CURRENT_TIMESTAMP AND u.is_active = 1
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'username': result[1],
                'email': result[2],
                'role': result[3],
                'company': result[4],
                'department': result[5]
            }
        
        return None
    
    def logout_user(self, session_id: str):
        """Logout user and invalidate session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()
    
    def user_exists(self, username: str) -> bool:
        """Check if user exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT 1 FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    def get_user_permissions(self, user_id: str, resource_type: str = None) -> List[str]:
        """Get user permissions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT role FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return []
        
        role = user[0]
        
        # Define role-based permissions
        permissions = {
            'admin': ['create', 'read', 'update', 'delete', 'manage_users', 'system_settings'],
            'manager': ['create', 'read', 'update', 'delete', 'manage_projects'],
            'architect': ['create', 'read', 'update', 'export'],
            'user': ['create', 'read', 'update'],
            'viewer': ['read']
        }
        
        conn.close()
        return permissions.get(role, ['read'])
    
    def log_action(self, user_id: str, action: str, resource_type: str, 
                  resource_id: str, details: str = ""):
        """Log user action for audit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log (user_id, action, resource_type, resource_id, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, action, resource_type, resource_id, details))
        
        conn.commit()
        conn.close()
    
    def get_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """Get projects for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.id, p.name, p.description, p.created_at, p.updated_at, p.status, p.tags,
                   CASE WHEN p.owner_id = ? THEN 'owner'
                        ELSE COALESCE(pc.role, 'none') END as role
            FROM projects p
            LEFT JOIN project_collaborators pc ON p.id = pc.project_id AND pc.user_id = ?
            WHERE p.owner_id = ? OR pc.user_id = ? OR p.is_public = 1
            ORDER BY p.updated_at DESC
        ''', (user_id, user_id, user_id, user_id))
        
        projects = []
        for row in cursor.fetchall():
            projects.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'created_at': row[3],
                'updated_at': row[4],
                'status': row[5],
                'tags': row[6].split(',') if row[6] else [],
                'role': row[7]
            })
        
        conn.close()
        return projects
    
    def create_project(self, user_id: str, name: str, description: str = "", 
                      project_data: Dict[str, Any] = None) -> str:
        """Create new project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        project_id = str(uuid.uuid4())
        project_data_json = json.dumps(project_data) if project_data else "{}"
        
        cursor.execute('''
            INSERT INTO projects (id, name, description, owner_id, project_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (project_id, name, description, user_id, project_data_json))
        
        conn.commit()
        conn.close()
        
        # Log project creation
        self.log_action(user_id, "project_created", "project", project_id, f"Project {name} created")
        
        return project_id
    
    def save_project(self, project_id: str, user_id: str, project_data: Dict[str, Any]) -> bool:
        """Save project data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            project_data_json = json.dumps(project_data)
            
            cursor.execute('''
                UPDATE projects 
                SET project_data = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND (owner_id = ? OR id IN (
                    SELECT project_id FROM project_collaborators 
                    WHERE user_id = ? AND role IN ('editor', 'manager')
                ))
            ''', (project_data_json, project_id, user_id, user_id))
            
            success = cursor.rowcount > 0
            conn.commit()
            
            if success:
                self.log_action(user_id, "project_saved", "project", project_id, "Project data saved")
            
            return success
            
        except Exception:
            return False
        finally:
            conn.close()
    
    def load_project(self, project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Load project data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.project_data, p.name, p.description, p.owner_id
            FROM projects p
            LEFT JOIN project_collaborators pc ON p.id = pc.project_id AND pc.user_id = ?
            WHERE p.id = ? AND (p.owner_id = ? OR pc.user_id = ? OR p.is_public = 1)
        ''', (user_id, project_id, user_id, user_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.log_action(user_id, "project_loaded", "project", project_id, "Project data loaded")
            
            return {
                'project_data': json.loads(result[0]) if result[0] else {},
                'name': result[1],
                'description': result[2],
                'is_owner': result[3] == user_id
            }
        
        return None

class EmailService:
    """Email service for notifications and reports"""
    
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = os.getenv("EMAIL_USERNAME", "")
        self.password = os.getenv("EMAIL_PASSWORD", "")
    
    def send_report_email(self, to_email: str, subject: str, report_data: Dict[str, Any], 
                         attachment_path: str = None) -> bool:
        """Send analysis report via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Create HTML email body
            html_body = self._create_report_email_body(report_data)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Add attachment if provided
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as f:
                    attachment = MIMEBase('application', 'octet-stream')
                    attachment.set_payload(f.read())
                
                encoders.encode_base64(attachment)
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}'
                )
                msg.attach(attachment)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
                server.send_message(msg)
                server.quit()
                return True
            
            return False
            
        except Exception as e:
            print(f"Email sending failed: {str(e)}")
            return False
    
    def _create_report_email_body(self, report_data: Dict[str, Any]) -> str:
        """Create HTML email body for report"""
        metrics = report_data.get('metrics', {})
        
        html = f"""
        <html>
        <body>
            <h2>Floor Plan Analysis Report</h2>
            <p>Your floor plan analysis has been completed successfully.</p>
            
            <h3>Key Metrics:</h3>
            <ul>
                <li>Total Analysis Points: {metrics.get('analysis_points', 0)}</li>
                <li>Optimization Score: {metrics.get('optimization_score', 0):.1f}%</li>
                <li>Space Utilization: {metrics.get('space_utilization', 0):.1f}%</li>
            </ul>
            
            <h3>Recommendations:</h3>
            <ul>
        """
        
        for rec in report_data.get('recommendations', [])[:5]:
            html += f"<li>{rec.get('title', '')}: {rec.get('description', '')}</li>"
        
        html += """
            </ul>
            
            <p>Please find the detailed report attached.</p>
            
            <p>Best regards,<br>
            Professional Floor Plan Analyzer Team</p>
        </body>
        </html>
        """
        
        return html

# Initialize global user manager
user_manager = UserManager()
email_service = EmailService()
