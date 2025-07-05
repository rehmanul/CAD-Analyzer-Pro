"""
Production PostgreSQL Database Integration
Full production-ready database management for DWG Analyzer Pro
"""

import os
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProductionDatabaseManager:
    """Production PostgreSQL database manager"""
    
    def __init__(self):
        # Use Render PostgreSQL database URL
        self.database_url = "postgresql://de_de:PUPB8V0s2b3bvNZUblolz7d6UM9bcBzb@dpg-d1h53rffte5s739b1i40-a.oregon-postgres.render.com/dwg_analyzer_pro"
        self.connection_pool = None
        self.init_connection_pool()
        self.init_production_schema()
    
    def init_connection_pool(self):
        """Initialize connection pool for production use"""
        try:
            # Configure SSL settings for Render PostgreSQL
            self.connection_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=20,
                dsn=self.database_url,
                sslmode='require'
            )
            print("✅ PostgreSQL connection pool initialized")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.putconn(conn)
    
    def init_production_schema(self):
        """Initialize full production database schema"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Projects table - main project storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    file_path TEXT,
                    file_type VARCHAR(10),
                    file_hash VARCHAR(64),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id UUID,
                    status VARCHAR(50) DEFAULT 'active',
                    metadata JSONB
                )
            """)
            
            # Floor plans table - DXF/DWG data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS floor_plans (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    original_entities JSONB NOT NULL,
                    processed_zones JSONB,
                    walls JSONB,
                    restricted_areas JSONB,
                    entrances JSONB,
                    bounds JSONB,
                    scale_factor REAL DEFAULT 1.0,
                    units VARCHAR(20) DEFAULT 'meters',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Ilot configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ilot_configurations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    size_0_1_percent INTEGER DEFAULT 10,
                    size_1_3_percent INTEGER DEFAULT 25,
                    size_3_5_percent INTEGER DEFAULT 30,
                    size_5_10_percent INTEGER DEFAULT 35,
                    total_area REAL,
                    min_spacing REAL DEFAULT 1.0,
                    wall_clearance REAL DEFAULT 0.5,
                    entrance_clearance REAL DEFAULT 2.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Ilot placements table - actual îlot positions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ilot_placements (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    configuration_id UUID REFERENCES ilot_configurations(id),
                    ilot_number INTEGER NOT NULL,
                    x_position REAL NOT NULL,
                    y_position REAL NOT NULL,
                    width REAL NOT NULL,
                    height REAL NOT NULL,
                    area REAL NOT NULL,
                    size_category VARCHAR(20),
                    rotation REAL DEFAULT 0,
                    zone_id VARCHAR(50),
                    accessibility_score REAL,
                    placement_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Corridors table - corridor networks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS corridors (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    corridor_type VARCHAR(50) NOT NULL,
                    start_point JSONB NOT NULL,
                    end_point JSONB NOT NULL,
                    width REAL NOT NULL,
                    path_points JSONB,
                    connects_ilots JSONB,
                    is_mandatory BOOLEAN DEFAULT FALSE,
                    accessibility_compliant BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    space_utilization REAL,
                    coverage_percentage REAL,
                    efficiency_score REAL,
                    accessibility_score REAL,
                    circulation_efficiency REAL,
                    safety_compliance REAL,
                    total_ilots INTEGER,
                    total_corridors INTEGER,
                    optimization_method VARCHAR(100),
                    analysis_duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    user_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Optimization logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    algorithm VARCHAR(100),
                    parameters JSONB,
                    initial_score REAL,
                    final_score REAL,
                    iterations INTEGER,
                    execution_time REAL,
                    convergence_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_floor_plans_project_id ON floor_plans(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ilot_placements_project_id ON ilot_placements(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_corridors_project_id ON corridors(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_results_project_id ON analysis_results(project_id)")
            
            conn.commit()
            print("✅ Production database schema initialized")
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Schema initialization failed: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def create_project(self, name: str, description: str = "", user_id: str = None, 
                      metadata: Dict = None) -> str:
        """Create new project"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            project_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO projects (id, name, description, user_id, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (project_id, name, description, user_id, json.dumps(metadata or {})))
            
            result = cursor.fetchone()
            conn.commit()
            return str(result[0])
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.return_connection(conn)
    
    def store_floor_plan(self, project_id: str, entities: List[Dict], 
                        zones: Dict = None, walls: List = None,
                        restricted_areas: List = None, entrances: List = None,
                        bounds: Dict = None, scale_factor: float = 1.0) -> str:
        """Store floor plan data"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO floor_plans 
                (project_id, original_entities, processed_zones, walls, 
                 restricted_areas, entrances, bounds, scale_factor)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                project_id,
                json.dumps(entities),
                json.dumps(zones or {}),
                json.dumps(walls or []),
                json.dumps(restricted_areas or []),
                json.dumps(entrances or []),
                json.dumps(bounds or {}),
                scale_factor
            ))
            
            result = cursor.fetchone()
            conn.commit()
            return str(result[0])
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.return_connection(conn)
    
    def store_ilot_configuration(self, project_id: str, config: Dict) -> str:
        """Store îlot configuration"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ilot_configurations 
                (project_id, size_0_1_percent, size_1_3_percent, size_3_5_percent, 
                 size_5_10_percent, total_area, min_spacing, wall_clearance, entrance_clearance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                project_id,
                config.get('size_0_1_percent', 10),
                config.get('size_1_3_percent', 25),
                config.get('size_3_5_percent', 30),
                config.get('size_5_10_percent', 35),
                config.get('total_area', 0),
                config.get('min_spacing', 1.0),
                config.get('wall_clearance', 0.5),
                config.get('entrance_clearance', 2.0)
            ))
            
            result = cursor.fetchone()
            conn.commit()
            return str(result[0])
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.return_connection(conn)
    
    def store_ilot_placements(self, project_id: str, configuration_id: str, 
                            ilots: List[Dict]) -> List[str]:
        """Store îlot placements"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            placement_ids = []
            
            for i, ilot in enumerate(ilots):
                cursor.execute("""
                    INSERT INTO ilot_placements 
                    (project_id, configuration_id, ilot_number, x_position, y_position,
                     width, height, area, size_category, rotation, zone_id,
                     accessibility_score, placement_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    project_id, configuration_id, i + 1,
                    ilot['x'], ilot['y'], ilot['width'], ilot['height'],
                    ilot['area'], ilot['size_category'], ilot.get('rotation', 0),
                    ilot.get('zone_id', ''), ilot.get('accessibility_score', 0),
                    ilot.get('placement_score', 0)
                ))
                
                result = cursor.fetchone()
                placement_ids.append(str(result[0]))
            
            conn.commit()
            return placement_ids
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.return_connection(conn)
    
    def store_corridors(self, project_id: str, corridors: List[Dict]) -> List[str]:
        """Store corridor data"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            corridor_ids = []
            
            for corridor in corridors:
                cursor.execute("""
                    INSERT INTO corridors 
                    (project_id, corridor_type, start_point, end_point, width,
                     path_points, connects_ilots, is_mandatory, accessibility_compliant)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    project_id,
                    corridor['type'],
                    json.dumps(corridor['start_point']),
                    json.dumps(corridor['end_point']),
                    corridor['width'],
                    json.dumps(corridor.get('path_points', [])),
                    json.dumps(corridor.get('connects_ilots', [])),
                    corridor.get('is_mandatory', False),
                    corridor.get('accessibility_compliant', True)
                ))
                
                result = cursor.fetchone()
                corridor_ids.append(str(result[0]))
            
            conn.commit()
            return corridor_ids
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.return_connection(conn)
    
    def store_analysis_results(self, project_id: str, results: Dict) -> str:
        """Store analysis results"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analysis_results 
                (project_id, space_utilization, coverage_percentage, efficiency_score,
                 accessibility_score, circulation_efficiency, safety_compliance,
                 total_ilots, total_corridors, optimization_method, analysis_duration)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                project_id,
                results.get('space_utilization', 0),
                results.get('coverage_percentage', 0),
                results.get('efficiency_score', 0),
                results.get('accessibility_score', 0),
                results.get('circulation_efficiency', 0),
                results.get('safety_compliance', 0),
                results.get('total_ilots', 0),
                results.get('total_corridors', 0),
                results.get('optimization_method', ''),
                results.get('analysis_duration', 0)
            ))
            
            result = cursor.fetchone()
            conn.commit()
            return str(result[0])
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.return_connection(conn)
    
    def get_project_data(self, project_id: str) -> Dict:
        """Get complete project data"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get project info
            cursor.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
            project = cursor.fetchone()
            
            if not project:
                return {}
            
            # Get floor plan
            cursor.execute("SELECT * FROM floor_plans WHERE project_id = %s", (project_id,))
            floor_plan = cursor.fetchone()
            
            # Get îlot configuration
            cursor.execute("SELECT * FROM ilot_configurations WHERE project_id = %s", (project_id,))
            config = cursor.fetchone()
            
            # Get îlot placements
            cursor.execute("SELECT * FROM ilot_placements WHERE project_id = %s ORDER BY ilot_number", (project_id,))
            ilots = cursor.fetchall()
            
            # Get corridors
            cursor.execute("SELECT * FROM corridors WHERE project_id = %s", (project_id,))
            corridors = cursor.fetchall()
            
            # Get analysis results
            cursor.execute("SELECT * FROM analysis_results WHERE project_id = %s ORDER BY created_at DESC LIMIT 1", (project_id,))
            analysis = cursor.fetchone()
            
            return {
                'project': dict(project) if project else {},
                'floor_plan': dict(floor_plan) if floor_plan else {},
                'configuration': dict(config) if config else {},
                'ilots': [dict(ilot) for ilot in ilots],
                'corridors': [dict(corridor) for corridor in corridors],
                'analysis': dict(analysis) if analysis else {}
            }
            
        except Exception as e:
            raise e
        finally:
            self.return_connection(conn)
    
    def get_all_projects(self, user_id: str = None) -> List[Dict]:
        """Get all projects for user"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if user_id:
                cursor.execute("""
                    SELECT p.*, COUNT(ip.id) as ilot_count, COUNT(c.id) as corridor_count
                    FROM projects p
                    LEFT JOIN ilot_placements ip ON p.id = ip.project_id
                    LEFT JOIN corridors c ON p.id = c.project_id
                    WHERE p.user_id = %s OR p.user_id IS NULL
                    GROUP BY p.id
                    ORDER BY p.updated_at DESC
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT p.*, COUNT(ip.id) as ilot_count, COUNT(c.id) as corridor_count
                    FROM projects p
                    LEFT JOIN ilot_placements ip ON p.id = ip.project_id
                    LEFT JOIN corridors c ON p.id = c.project_id
                    GROUP BY p.id
                    ORDER BY p.updated_at DESC
                """)
            
            return [dict(project) for project in cursor.fetchall()]
            
        except Exception as e:
            raise e
        finally:
            self.return_connection(conn)
    
    def delete_project(self, project_id: str) -> bool:
        """Delete project and all related data"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE id = %s", (project_id,))
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.return_connection(conn)

# Global instance
production_db = ProductionDatabaseManager()