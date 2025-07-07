
import json
import base64
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
import os
import zipfile
import tempfile
import threading
import time
from pathlib import Path

class CloudStorageService:
    """Cloud storage integration for floor plan projects"""
    
    def __init__(self, storage_type: str = "local"):
        self.storage_type = storage_type
        self.local_storage_path = "cloud_storage"
        self.ensure_storage_directory()
        
        # Cloud service configurations (would use environment variables)
        self.aws_config = {
            'access_key': os.getenv('AWS_ACCESS_KEY', ''),
            'secret_key': os.getenv('AWS_SECRET_KEY', ''),
            'bucket_name': os.getenv('AWS_BUCKET_NAME', 'floorplan-storage'),
            'region': os.getenv('AWS_REGION', 'us-east-1')
        }
        
        self.google_config = {
            'credentials_path': os.getenv('GOOGLE_CREDENTIALS_PATH', ''),
            'bucket_name': os.getenv('GOOGLE_BUCKET_NAME', 'floorplan-storage')
        }
    
    def ensure_storage_directory(self):
        """Ensure local storage directory exists"""
        os.makedirs(self.local_storage_path, exist_ok=True)
    
    def upload_project(self, project_id: str, project_data: Dict[str, Any], 
                      user_id: str) -> Dict[str, Any]:
        """Upload project to cloud storage"""
        try:
            # Prepare project package
            package_data = self._create_project_package(project_data)
            
            if self.storage_type == "local":
                return self._upload_to_local_storage(project_id, package_data, user_id)
            elif self.storage_type == "aws":
                return self._upload_to_aws(project_id, package_data, user_id)
            elif self.storage_type == "google":
                return self._upload_to_google_cloud(project_id, package_data, user_id)
            else:
                return {'success': False, 'error': 'Unsupported storage type'}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def download_project(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Download project from cloud storage"""
        try:
            if self.storage_type == "local":
                return self._download_from_local_storage(project_id, user_id)
            elif self.storage_type == "aws":
                return self._download_from_aws(project_id, user_id)
            elif self.storage_type == "google":
                return self._download_from_google_cloud(project_id, user_id)
            else:
                return {'success': False, 'error': 'Unsupported storage type'}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_project_package(self, project_data: Dict[str, Any]) -> bytes:
        """Create compressed project package"""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save project data as JSON
            project_file = os.path.join(temp_dir, 'project.json')
            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=2, default=str)
            
            # Create metadata file
            metadata = {
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'format': 'floor_plan_analyzer_project'
            }
            metadata_file = os.path.join(temp_dir, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create ZIP package
            zip_path = os.path.join(temp_dir, 'project.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(project_file, 'project.json')
                zipf.write(metadata_file, 'metadata.json')
            
            # Read ZIP as bytes
            with open(zip_path, 'rb') as f:
                return f.read()
    
    def _upload_to_local_storage(self, project_id: str, package_data: bytes, 
                                user_id: str) -> Dict[str, Any]:
        """Upload to local storage"""
        user_dir = os.path.join(self.local_storage_path, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        file_path = os.path.join(user_dir, f"{project_id}.zip")
        
        with open(file_path, 'wb') as f:
            f.write(package_data)
        
        return {
            'success': True,
            'storage_path': file_path,
            'size_bytes': len(package_data),
            'checksum': hashlib.md5(package_data).hexdigest()
        }
    
    def _download_from_local_storage(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Download from local storage"""
        file_path = os.path.join(self.local_storage_path, user_id, f"{project_id}.zip")
        
        if not os.path.exists(file_path):
            return {'success': False, 'error': 'Project not found'}
        
        with open(file_path, 'rb') as f:
            package_data = f.read()
        
        # Extract project data
        project_data = self._extract_project_package(package_data)
        
        return {
            'success': True,
            'project_data': project_data,
            'size_bytes': len(package_data)
        }
    
    def _extract_project_package(self, package_data: bytes) -> Dict[str, Any]:
        """Extract project data from package"""
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, 'project.zip')
            
            with open(zip_path, 'wb') as f:
                f.write(package_data)
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            project_file = os.path.join(temp_dir, 'project.json')
            with open(project_file, 'r') as f:
                return json.load(f)
    
    def _upload_to_aws(self, project_id: str, package_data: bytes, user_id: str) -> Dict[str, Any]:
        """Upload to AWS S3 (simulation - would use boto3)"""
        # This is a simulation - real implementation would use boto3
        return {
            'success': True,
            'storage_path': f"s3://{self.aws_config['bucket_name']}/{user_id}/{project_id}.zip",
            'size_bytes': len(package_data),
            'provider': 'aws_s3'
        }
    
    def _upload_to_google_cloud(self, project_id: str, package_data: bytes, 
                               user_id: str) -> Dict[str, Any]:
        """Upload to Google Cloud Storage (simulation)"""
        # This is a simulation - real implementation would use google-cloud-storage
        return {
            'success': True,
            'storage_path': f"gs://{self.google_config['bucket_name']}/{user_id}/{project_id}.zip",
            'size_bytes': len(package_data),
            'provider': 'google_cloud'
        }

class RealtimeCollaboration:
    """Real-time collaboration features for team projects"""
    
    def __init__(self, db_path: str = "collaboration.db"):
        self.db_path = db_path
        self.init_collaboration_db()
        self.active_sessions = {}
        self.change_listeners = {}
    
    def init_collaboration_db(self):
        """Initialize collaboration database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Collaboration sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collaboration_sessions (
                session_id TEXT PRIMARY KEY,
                project_id TEXT,
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Session participants table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_participants (
                session_id TEXT,
                user_id TEXT,
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role TEXT DEFAULT 'viewer',
                cursor_position TEXT,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, user_id)
            )
        ''')
        
        # Real-time changes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                change_type TEXT,
                change_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                applied BOOLEAN DEFAULT 0
            )
        ''')
        
        # Comments and annotations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_comments (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                user_id TEXT,
                comment_text TEXT,
                position_x REAL,
                position_y REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT 0,
                parent_comment_id TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_collaboration_session(self, project_id: str, user_id: str, 
                                   duration_hours: int = 8) -> str:
        """Create new collaboration session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=duration_hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO collaboration_sessions (session_id, project_id, created_by, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (session_id, project_id, user_id, expires_at))
        
        # Add creator as participant
        cursor.execute('''
            INSERT INTO session_participants (session_id, user_id, role)
            VALUES (?, ?, 'owner')
        ''', (session_id, user_id))
        
        conn.commit()
        conn.close()
        
        # Initialize session in memory
        self.active_sessions[session_id] = {
            'project_id': project_id,
            'participants': {user_id: {'role': 'owner', 'cursor': None}},
            'changes_queue': []
        }
        
        return session_id
    
    def join_collaboration_session(self, session_id: str, user_id: str, 
                                 role: str = 'viewer') -> Dict[str, Any]:
        """Join existing collaboration session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if session exists and is active
        cursor.execute('''
            SELECT project_id, expires_at FROM collaboration_sessions
            WHERE session_id = ? AND is_active = 1 AND expires_at > CURRENT_TIMESTAMP
        ''', (session_id,))
        
        session = cursor.fetchone()
        if not session:
            conn.close()
            return {'success': False, 'error': 'Session not found or expired'}
        
        # Add participant
        cursor.execute('''
            INSERT OR REPLACE INTO session_participants (session_id, user_id, role, last_activity)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (session_id, user_id, role))
        
        conn.commit()
        conn.close()
        
        # Update in-memory session
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                'project_id': session[0],
                'participants': {},
                'changes_queue': []
            }
        
        self.active_sessions[session_id]['participants'][user_id] = {
            'role': role,
            'cursor': None,
            'joined_at': datetime.now()
        }
        
        return {
            'success': True,
            'project_id': session[0],
            'participants': list(self.active_sessions[session_id]['participants'].keys())
        }
    
    def broadcast_change(self, session_id: str, user_id: str, change_type: str, 
                        change_data: Dict[str, Any]):
        """Broadcast change to all session participants"""
        if session_id not in self.active_sessions:
            return
        
        change = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'change_type': change_type,
            'change_data': change_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO realtime_changes (session_id, user_id, change_type, change_data)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_id, change_type, json.dumps(change_data)))
        
        conn.commit()
        conn.close()
        
        # Add to in-memory queue
        self.active_sessions[session_id]['changes_queue'].append(change)
        
        # Notify listeners (in real app, would use WebSockets)
        if session_id in self.change_listeners:
            for callback in self.change_listeners[session_id]:
                try:
                    callback(change)
                except:
                    pass
    
    def get_session_changes(self, session_id: str, since_timestamp: str = None) -> List[Dict[str, Any]]:
        """Get changes for session since timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if since_timestamp:
            cursor.execute('''
                SELECT user_id, change_type, change_data, timestamp
                FROM realtime_changes
                WHERE session_id = ? AND timestamp > ?
                ORDER BY timestamp
            ''', (session_id, since_timestamp))
        else:
            cursor.execute('''
                SELECT user_id, change_type, change_data, timestamp
                FROM realtime_changes
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 50
            ''', (session_id,))
        
        changes = []
        for row in cursor.fetchall():
            changes.append({
                'user_id': row[0],
                'change_type': row[1],
                'change_data': json.loads(row[2]),
                'timestamp': row[3]
            })
        
        conn.close()
        return changes
    
    def add_comment(self, project_id: str, user_id: str, comment_text: str, 
                   position_x: float, position_y: float, 
                   parent_comment_id: str = None) -> str:
        """Add comment/annotation to project"""
        comment_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO project_comments (id, project_id, user_id, comment_text, 
                                        position_x, position_y, parent_comment_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (comment_id, project_id, user_id, comment_text, 
              position_x, position_y, parent_comment_id))
        
        conn.commit()
        conn.close()
        
        return comment_id
    
    def get_project_comments(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all comments for project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.user_id, u.username, c.comment_text, 
                   c.position_x, c.position_y, c.created_at, c.resolved, c.parent_comment_id
            FROM project_comments c
            LEFT JOIN users u ON c.user_id = u.id
            WHERE c.project_id = ?
            ORDER BY c.created_at
        ''', (project_id,))
        
        comments = []
        for row in cursor.fetchall():
            comments.append({
                'id': row[0],
                'user_id': row[1],
                'username': row[2] or 'Unknown',
                'comment_text': row[3],
                'position_x': row[4],
                'position_y': row[5],
                'created_at': row[6],
                'resolved': bool(row[7]),
                'parent_comment_id': row[8]
            })
        
        conn.close()
        return comments
    
    def update_cursor_position(self, session_id: str, user_id: str, 
                             cursor_x: float, cursor_y: float):
        """Update user cursor position for real-time tracking"""
        if session_id in self.active_sessions and user_id in self.active_sessions[session_id]['participants']:
            self.active_sessions[session_id]['participants'][user_id]['cursor'] = {
                'x': cursor_x,
                'y': cursor_y,
                'timestamp': datetime.now().isoformat()
            }
            
            # Broadcast cursor update
            self.broadcast_change(session_id, user_id, 'cursor_update', {
                'x': cursor_x,
                'y': cursor_y
            })

class VersionControl:
    """Version control system for floor plan projects"""
    
    def __init__(self, db_path: str = "version_control.db"):
        self.db_path = db_path
        self.init_version_db()
    
    def init_version_db(self):
        """Initialize version control database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Project versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_versions (
                version_id TEXT PRIMARY KEY,
                project_id TEXT,
                version_number INTEGER,
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                commit_message TEXT,
                project_data TEXT,
                changes_summary TEXT,
                is_major_version BOOLEAN DEFAULT 0
            )
        ''')
        
        # Version diffs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS version_diffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_version_id TEXT,
                to_version_id TEXT,
                diff_type TEXT,
                diff_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_version(self, project_id: str, user_id: str, project_data: Dict[str, Any], 
                      commit_message: str, is_major: bool = False) -> str:
        """Create new version of project"""
        version_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get next version number
        cursor.execute('''
            SELECT COALESCE(MAX(version_number), 0) + 1 
            FROM project_versions 
            WHERE project_id = ?
        ''', (project_id,))
        
        version_number = cursor.fetchone()[0]
        
        # Calculate changes summary
        changes_summary = self._calculate_changes_summary(project_id, project_data)
        
        cursor.execute('''
            INSERT INTO project_versions (version_id, project_id, version_number, 
                                        created_by, commit_message, project_data, 
                                        changes_summary, is_major_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (version_id, project_id, version_number, user_id, commit_message, 
              json.dumps(project_data), json.dumps(changes_summary), is_major))
        
        conn.commit()
        conn.close()
        
        return version_id
    
    def get_version_history(self, project_id: str) -> List[Dict[str, Any]]:
        """Get version history for project"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pv.version_id, pv.version_number, pv.created_by, u.username,
                   pv.created_at, pv.commit_message, pv.changes_summary, pv.is_major_version
            FROM project_versions pv
            LEFT JOIN users u ON pv.created_by = u.id
            WHERE pv.project_id = ?
            ORDER BY pv.version_number DESC
        ''', (project_id,))
        
        versions = []
        for row in cursor.fetchall():
            changes_summary = json.loads(row[6]) if row[6] else {}
            versions.append({
                'version_id': row[0],
                'version_number': row[1],
                'created_by': row[2],
                'username': row[3] or 'Unknown',
                'created_at': row[4],
                'commit_message': row[5],
                'changes_summary': changes_summary,
                'is_major_version': bool(row[7])
            })
        
        conn.close()
        return versions
    
    def _calculate_changes_summary(self, project_id: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary of changes from previous version"""
        # Get previous version
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT project_data FROM project_versions
            WHERE project_id = ?
            ORDER BY version_number DESC
            LIMIT 1
        ''', (project_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return {'type': 'initial_version', 'changes': []}
        
        previous_data = json.loads(result[0])
        
        # Simple change detection
        changes = []
        
        # Check îlot changes
        prev_ilots = len(previous_data.get('ilot_results', []))
        curr_ilots = len(current_data.get('ilot_results', []))
        
        if curr_ilots != prev_ilots:
            changes.append(f"Îlots changed: {prev_ilots} → {curr_ilots}")
        
        # Check corridor changes
        prev_corridors = len(previous_data.get('corridor_results', []))
        curr_corridors = len(current_data.get('corridor_results', []))
        
        if curr_corridors != prev_corridors:
            changes.append(f"Corridors changed: {prev_corridors} → {curr_corridors}")
        
        return {
            'type': 'update',
            'changes': changes,
            'ilot_delta': curr_ilots - prev_ilots,
            'corridor_delta': curr_corridors - prev_corridors
        }

# Initialize global services
cloud_service = CloudStorageService()
collaboration_service = RealtimeCollaboration()
version_control = VersionControl()
