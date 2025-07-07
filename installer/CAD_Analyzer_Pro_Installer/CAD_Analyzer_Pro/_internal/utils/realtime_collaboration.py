import asyncio
import websockets
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid
import redis
from dataclasses import dataclass, asdict
import threading
import queue
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

@dataclass
class CollaborationEvent:
    """Collaboration event data structure"""
    event_id: str
    event_type: str  # 'ilot_add', 'ilot_move', 'ilot_delete', 'cursor_move', 'selection_change'
    user_id: str
    timestamp: float
    data: Dict[str, Any]
    project_id: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CollaborationEvent':
        return cls(**json.loads(json_str))


@dataclass
class User:
    """Collaboration user"""
    user_id: str
    name: str
    email: str
    role: str  # 'viewer', 'editor', 'admin'
    color: str  # User's cursor/selection color
    last_active: float
    cursor_position: Optional[Dict[str, float]] = None
    selected_items: List[str] = None
    
    def __post_init__(self):
        if self.selected_items is None:
            self.selected_items = []


class CollaborationServer:
    """WebSocket server for real-time collaboration"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.projects: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, User] = {}
        self.project_locks: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Redis for persistence and scaling
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_enabled = True
            logger.info("Redis connected for collaboration persistence")
        except:
            self.redis_client = None
            self.redis_enabled = False
            logger.warning("Redis not available, using in-memory storage")
        
        # Event history
        self.event_history: Dict[str, List[CollaborationEvent]] = defaultdict(list)
        self.max_history_per_project = 1000
    
    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting collaboration server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle new client connection"""
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        
        try:
            logger.info(f"New client connected: {client_id}")
            
            # Authentication
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            user = await self._authenticate_user(auth_data)
            if not user:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Authentication failed'
                }))
                return
            
            self.users[client_id] = user
            
            # Send initial state
            await self._send_initial_state(websocket, user, auth_data.get('project_id'))
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {str(e)}")
        finally:
            # Clean up
            if client_id in self.clients:
                del self.clients[client_id]
            if client_id in self.users:
                await self._broadcast_user_left(self.users[client_id])
                del self.users[client_id]
    
    async def _authenticate_user(self, auth_data: Dict[str, Any]) -> Optional[User]:
        """Authenticate user"""
        # In production, verify JWT token or API key
        token = auth_data.get('token')
        
        # For now, create user from auth data
        user = User(
            user_id=auth_data.get('user_id', str(uuid.uuid4())),
            name=auth_data.get('name', 'Anonymous'),
            email=auth_data.get('email', ''),
            role=auth_data.get('role', 'editor'),
            color=auth_data.get('color', self._generate_user_color()),
            last_active=time.time()
        )
        
        return user
    
    def _generate_user_color(self) -> str:
        """Generate unique user color"""
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#48DBFB', '#0ABDE3', '#006BA6', '#6C5CE7', '#A29BFE'
        ]
        used_colors = {user.color for user in self.users.values()}
        available_colors = [c for c in colors if c not in used_colors]
        
        if available_colors:
            return available_colors[0]
        else:
            # Generate random color if all predefined colors are used
            import random
            return f'#{random.randint(0, 0xFFFFFF):06x}'
    
    async def _send_initial_state(self, websocket: websockets.WebSocketServerProtocol,
                                 user: User, project_id: str):
        """Send initial project state to new user"""
        # Get project data
        project_data = await self._get_project_data(project_id)
        
        # Get active users
        active_users = [
            {
                'user_id': u.user_id,
                'name': u.name,
                'color': u.color,
                'cursor_position': u.cursor_position,
                'selected_items': u.selected_items
            }
            for u in self.users.values()
            if u.user_id != user.user_id
        ]
        
        # Get recent events
        recent_events = self.event_history.get(project_id, [])[-50:]
        
        # Send initial state
        await websocket.send(json.dumps({
            'type': 'initial_state',
            'project_id': project_id,
            'project_data': project_data,
            'active_users': active_users,
            'recent_events': [asdict(e) for e in recent_events],
            'user_info': {
                'user_id': user.user_id,
                'color': user.color,
                'role': user.role
            }
        }))
        
        # Broadcast new user joined
        await self._broadcast_user_joined(user, project_id)
    
    async def _get_project_data(self, project_id: str) -> Dict[str, Any]:
        """Get project data from storage"""
        if self.redis_enabled:
            data = self.redis_client.get(f'project:{project_id}')
            if data:
                return json.loads(data)
        
        return self.projects.get(project_id, {
            'ilots': [],
            'corridors': [],
            'analysis_results': {},
            'configuration': {}
        })
    
    async def _save_project_data(self, project_id: str, data: Dict[str, Any]):
        """Save project data to storage"""
        self.projects[project_id] = data
        
        if self.redis_enabled:
            self.redis_client.set(
                f'project:{project_id}',
                json.dumps(data),
                ex=86400  # Expire after 24 hours
            )
    
    async def _handle_message(self, client_id: str, message: str):
        """Handle message from client"""
        try:
            data = json.loads(message)
            user = self.users.get(client_id)
            
            if not user:
                return
            
            # Update user activity
            user.last_active = time.time()
            
            # Create event
            event = CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=data['type'],
                user_id=user.user_id,
                timestamp=time.time(),
                data=data.get('data', {}),
                project_id=data.get('project_id', '')
            )
            
            # Handle different event types
            if event.event_type == 'cursor_move':
                await self._handle_cursor_move(user, event)
            elif event.event_type == 'ilot_add':
                await self._handle_ilot_add(user, event)
            elif event.event_type == 'ilot_move':
                await self._handle_ilot_move(user, event)
            elif event.event_type == 'ilot_delete':
                await self._handle_ilot_delete(user, event)
            elif event.event_type == 'selection_change':
                await self._handle_selection_change(user, event)
            elif event.event_type == 'lock_request':
                await self._handle_lock_request(user, event)
            elif event.event_type == 'lock_release':
                await self._handle_lock_release(user, event)
            elif event.event_type == 'chat_message':
                await self._handle_chat_message(user, event)
            
            # Store event in history
            self._store_event(event)
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client {client_id}")
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
    
    async def _handle_cursor_move(self, user: User, event: CollaborationEvent):
        """Handle cursor movement"""
        user.cursor_position = event.data.get('position')
        
        # Broadcast to other users
        await self._broadcast_to_project(event.project_id, {
            'type': 'cursor_update',
            'user_id': user.user_id,
            'position': user.cursor_position,
            'color': user.color
        }, exclude_user=user.user_id)
    
    async def _handle_ilot_add(self, user: User, event: CollaborationEvent):
        """Handle îlot addition"""
        if user.role not in ['editor', 'admin']:
            await self._send_error(user, "Insufficient permissions")
            return
        
        # Update project data
        project_data = await self._get_project_data(event.project_id)
        ilot = event.data.get('ilot')
        
        if ilot:
            project_data.setdefault('ilots', []).append(ilot)
            await self._save_project_data(event.project_id, project_data)
            
            # Broadcast to all users
            await self._broadcast_to_project(event.project_id, {
                'type': 'ilot_added',
                'ilot': ilot,
                'user_id': user.user_id,
                'user_name': user.name
            })
    
    async def _handle_ilot_move(self, user: User, event: CollaborationEvent):
        """Handle îlot movement"""
        if user.role not in ['editor', 'admin']:
            await self._send_error(user, "Insufficient permissions")
            return
        
        ilot_id = event.data.get('ilot_id')
        new_position = event.data.get('position')
        
        # Check if îlot is locked by another user
        lock_user = self.project_locks[event.project_id].get(ilot_id)
        if lock_user and lock_user != user.user_id:
            await self._send_error(user, f"Îlot is locked by another user")
            return
        
        # Update project data
        project_data = await self._get_project_data(event.project_id)
        for ilot in project_data.get('ilots', []):
            if ilot.get('id') == ilot_id:
                ilot['position'] = new_position
                break
        
        await self._save_project_data(event.project_id, project_data)
        
        # Broadcast to all users
        await self._broadcast_to_project(event.project_id, {
            'type': 'ilot_moved',
            'ilot_id': ilot_id,
            'position': new_position,
            'user_id': user.user_id,
            'user_name': user.name
        })
    
    async def _handle_ilot_delete(self, user: User, event: CollaborationEvent):
        """Handle îlot deletion"""
        if user.role not in ['editor', 'admin']:
            await self._send_error(user, "Insufficient permissions")
            return
        
        ilot_id = event.data.get('ilot_id')
        
        # Update project data
        project_data = await self._get_project_data(event.project_id)
        project_data['ilots'] = [
            ilot for ilot in project_data.get('ilots', [])
            if ilot.get('id') != ilot_id
        ]
        
        await self._save_project_data(event.project_id, project_data)
        
        # Broadcast to all users
        await self._broadcast_to_project(event.project_id, {
            'type': 'ilot_deleted',
            'ilot_id': ilot_id,
            'user_id': user.user_id,
            'user_name': user.name
        })
    
    async def _handle_selection_change(self, user: User, event: CollaborationEvent):
        """Handle selection change"""
        user.selected_items = event.data.get('selected_items', [])
        
        # Broadcast to other users
        await self._broadcast_to_project(event.project_id, {
            'type': 'selection_update',
            'user_id': user.user_id,
            'selected_items': user.selected_items,
            'color': user.color
        }, exclude_user=user.user_id)
    
    async def _handle_lock_request(self, user: User, event: CollaborationEvent):
        """Handle lock request for an item"""
        item_id = event.data.get('item_id')
        project_locks = self.project_locks[event.project_id]
        
        # Check if already locked
        if item_id in project_locks:
            lock_user = project_locks[item_id]
            if lock_user != user.user_id:
                await self._send_to_user(user, {
                    'type': 'lock_denied',
                    'item_id': item_id,
                    'locked_by': lock_user
                })
                return
        
        # Grant lock
        project_locks[item_id] = user.user_id
        
        # Broadcast lock status
        await self._broadcast_to_project(event.project_id, {
            'type': 'item_locked',
            'item_id': item_id,
            'user_id': user.user_id,
            'user_name': user.name
        })
    
    async def _handle_lock_release(self, user: User, event: CollaborationEvent):
        """Handle lock release"""
        item_id = event.data.get('item_id')
        project_locks = self.project_locks[event.project_id]
        
        if project_locks.get(item_id) == user.user_id:
            del project_locks[item_id]
            
            # Broadcast lock release
            await self._broadcast_to_project(event.project_id, {
                'type': 'item_unlocked',
                'item_id': item_id,
                'user_id': user.user_id
            })
    
    async def _handle_chat_message(self, user: User, event: CollaborationEvent):
        """Handle chat message"""
        message = event.data.get('message', '')
        
        # Broadcast chat message
        await self._broadcast_to_project(event.project_id, {
            'type': 'chat_message',
            'user_id': user.user_id,
            'user_name': user.name,
            'message': message,
            'timestamp': event.timestamp,
            'color': user.color
        })
    
    async def _broadcast_user_joined(self, user: User, project_id: str):
        """Broadcast when user joins"""
        await self._broadcast_to_project(project_id, {
            'type': 'user_joined',
            'user': {
                'user_id': user.user_id,
                'name': user.name,
                'color': user.color,
                'role': user.role
            }
        }, exclude_user=user.user_id)
    
    async def _broadcast_user_left(self, user: User):
        """Broadcast when user leaves"""
        # Release all locks held by user
        for project_id, locks in self.project_locks.items():
            items_to_unlock = [item_id for item_id, lock_user in locks.items() 
                             if lock_user == user.user_id]
            for item_id in items_to_unlock:
                del locks[item_id]
                await self._broadcast_to_project(project_id, {
                    'type': 'item_unlocked',
                    'item_id': item_id,
                    'user_id': user.user_id
                })
        
        # Broadcast user left to all projects
        for project_id in self.projects.keys():
            await self._broadcast_to_project(project_id, {
                'type': 'user_left',
                'user_id': user.user_id
            })
    
    async def _broadcast_to_project(self, project_id: str, message: Dict[str, Any],
                                  exclude_user: Optional[str] = None):
        """Broadcast message to all users in a project"""
        message['project_id'] = project_id
        message_json = json.dumps(message)
        
        # Find users in the project
        tasks = []
        for client_id, websocket in self.clients.items():
            user = self.users.get(client_id)
            if user and user.user_id != exclude_user:
                tasks.append(websocket.send(message_json))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_user(self, user: User, message: Dict[str, Any]):
        """Send message to specific user"""
        client_id = next((cid for cid, u in self.users.items() if u == user), None)
        if client_id and client_id in self.clients:
            await self.clients[client_id].send(json.dumps(message))
    
    async def _send_error(self, user: User, error_message: str):
        """Send error message to user"""
        await self._send_to_user(user, {
            'type': 'error',
            'message': error_message
        })
    
    def _store_event(self, event: CollaborationEvent):
        """Store event in history"""
        history = self.event_history[event.project_id]
        history.append(event)
        
        # Limit history size
        if len(history) > self.max_history_per_project:
            history.pop(0)
        
        # Store in Redis if available
        if self.redis_enabled:
            self.redis_client.lpush(
                f'events:{event.project_id}',
                event.to_json()
            )
            self.redis_client.ltrim(f'events:{event.project_id}', 0, self.max_history_per_project)


class CollaborationClient:
    """Client for connecting to collaboration server"""
    
    def __init__(self, server_url: str, user_info: Dict[str, Any], project_id: str):
        self.server_url = server_url
        self.user_info = user_info
        self.project_id = project_id
        self.websocket = None
        self.connected = False
        self.event_handlers = {}
        self.message_queue = queue.Queue()
        self.worker_thread = None
    
    async def connect(self):
        """Connect to collaboration server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.connected = True
            
            # Send authentication
            auth_data = {
                'token': self.user_info.get('token', ''),
                'user_id': self.user_info.get('user_id'),
                'name': self.user_info.get('name'),
                'email': self.user_info.get('email'),
                'role': self.user_info.get('role', 'editor'),
                'project_id': self.project_id
            }
            await self.websocket.send(json.dumps(auth_data))
            
            # Start message handler
            asyncio.create_task(self._handle_messages())
            
            # Start worker thread for sending messages
            self.worker_thread = threading.Thread(target=self._send_worker)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            
            logger.info("Connected to collaboration server")
            
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from server"""
        self.connected = False
        if self.websocket:
            await self.websocket.close()
    
    def on(self, event_type: str, handler: callable):
        """Register event handler"""
        self.event_handlers[event_type] = handler
    
    def send_cursor_position(self, x: float, y: float):
        """Send cursor position update"""
        self._queue_message({
            'type': 'cursor_move',
            'project_id': self.project_id,
            'data': {
                'position': {'x': x, 'y': y}
            }
        })
    
    def send_ilot_add(self, ilot: Dict[str, Any]):
        """Send îlot addition"""
        self._queue_message({
            'type': 'ilot_add',
            'project_id': self.project_id,
            'data': {
                'ilot': ilot
            }
        })
    
    def send_ilot_move(self, ilot_id: str, position: Dict[str, float]):
        """Send îlot movement"""
        self._queue_message({
            'type': 'ilot_move',
            'project_id': self.project_id,
            'data': {
                'ilot_id': ilot_id,
                'position': position
            }
        })
    
    def send_ilot_delete(self, ilot_id: str):
        """Send îlot deletion"""
        self._queue_message({
            'type': 'ilot_delete',
            'project_id': self.project_id,
            'data': {
                'ilot_id': ilot_id
            }
        })
    
    def send_selection_change(self, selected_items: List[str]):
        """Send selection change"""
        self._queue_message({
            'type': 'selection_change',
            'project_id': self.project_id,
            'data': {
                'selected_items': selected_items
            }
        })
    
    def request_lock(self, item_id: str):
        """Request lock on an item"""
        self._queue_message({
            'type': 'lock_request',
            'project_id': self.project_id,
            'data': {
                'item_id': item_id
            }
        })
    
    def release_lock(self, item_id: str):
        """Release lock on an item"""
        self._queue_message({
            'type': 'lock_release',
            'project_id': self.project_id,
            'data': {
                'item_id': item_id
            }
        })
    
    def send_chat_message(self, message: str):
        """Send chat message"""
        self._queue_message({
            'type': 'chat_message',
            'project_id': self.project_id,
            'data': {
                'message': message
            }
        })
    
    def _queue_message(self, message: Dict[str, Any]):
        """Queue message for sending"""
        if self.connected:
            self.message_queue.put(json.dumps(message))
    
    def _send_worker(self):
        """Worker thread for sending messages"""
        while self.connected:
            try:
                message = self.message_queue.get(timeout=0.1)
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(message),
                    asyncio.get_event_loop()
                )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
    
    async def _handle_messages(self):
        """Handle incoming messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                event_type = data.get('type')
                
                # Call registered handler
                if event_type in self.event_handlers:
                    handler = self.event_handlers[event_type]
                    await self._call_handler(handler, data)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error handling messages: {str(e)}")
    
    async def _call_handler(self, handler: callable, data: Dict[str, Any]):
        """Call event handler"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(data)
            else:
                handler(data)
        except Exception as e:
            logger.error(f"Error in event handler: {str(e)}")


def start_collaboration_server(host: str = '0.0.0.0', port: int = 8765):
    """Start the collaboration server"""
    server = CollaborationServer(host, port)
    asyncio.run(server.start())