"""
Render Database Configuration
Production-ready PostgreSQL setup for Render deployment
"""

import os
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

logger = logging.getLogger(__name__)

class RenderDatabase:
    """Database manager for Render deployment"""
    
    def __init__(self):
        self.pool = None
        self.engine = None
        self.SessionLocal = None
        self.setup_database()
    
    def setup_database(self):
        """Setup database connection for Render"""
        try:
            # Get database URL from Render environment
            database_url = os.environ.get('DATABASE_URL')
            
            if database_url:
                # Use provided DATABASE_URL
                self.engine = create_engine(
                    database_url,
                    pool_size=20,
                    max_overflow=30,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
            else:
                # Fallback to individual components
                db_config = {
                    'host': os.environ.get('DATABASE_HOST', 'localhost'),
                    'port': int(os.environ.get('DATABASE_PORT', '5432')),
                    'database': os.environ.get('DATABASE_NAME', 'cad_analyzer_prod'),
                    'username': os.environ.get('DATABASE_USER', 'cad_analyzer_user'),
                    'password': os.environ.get('DATABASE_PASSWORD', ''),
                }
                
                connection_string = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
                
                self.engine = create_engine(
                    connection_string,
                    pool_size=20,
                    max_overflow=30,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Initialize schema
            self.init_schema()
            
            logger.info("✅ Render database connection established")
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {str(e)}")
            # Use SQLite fallback for development
            self.setup_fallback_database()
    
    def setup_fallback_database(self):
        """Setup SQLite fallback database"""
        try:
            self.engine = create_engine(
                'sqlite:///cad_analyzer_fallback.db',
                pool_pre_ping=True,
                echo=False
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            self.init_schema()
            logger.info("✅ Fallback SQLite database initialized")
            
        except Exception as e:
            logger.error(f"❌ Fallback database setup failed: {str(e)}")
    
    def init_schema(self):
        """Initialize database schema"""
        try:
            with self.engine.connect() as conn:
                # Create projects table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        file_name VARCHAR(255),
                        file_type VARCHAR(50),
                        analysis_data JSONB,
                        ilot_data JSONB,
                        corridor_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create analysis_cache table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS analysis_cache (
                        id SERIAL PRIMARY KEY,
                        cache_key VARCHAR(255) UNIQUE NOT NULL,
                        cache_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """))
                
                # Create users table (optional)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        email VARCHAR(255) UNIQUE,
                        name VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.commit()
                logger.info("✅ Database schema initialized")
                
        except Exception as e:
            logger.error(f"❌ Schema initialization failed: {str(e)}")
    
    def get_session(self):
        """Get database session"""
        if self.SessionLocal:
            return self.SessionLocal()
        return None
    
    def execute_query(self, query, params=None):
        """Execute raw SQL query"""
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                conn.commit()
                return result
        except Exception as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            return None
    
    def health_check(self):
        """Check database health"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"❌ Database health check failed: {str(e)}")
            return False

# Global database instance
render_db = RenderDatabase()