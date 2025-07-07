import hashlib
import json
import streamlit as st
from utils.production_database import production_db
from typing import Dict, Any, Optional

class CacheManager:
    def __init__(self):
        self.memory_cache = {}
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    @st.cache_data(ttl=3600)
    def get_cached_result(_self, cache_key: str) -> Optional[Dict]:
        """Get cached result from memory or database"""
        # Check memory cache first
        if cache_key in _self.memory_cache:
            return _self.memory_cache[cache_key]
        
        # Check database cache
        try:
            conn = production_db.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT result_data FROM processing_cache WHERE cache_key = %s",
                    (cache_key,)
                )
                result = cursor.fetchone()
                production_db.return_connection(conn)
                
                if result:
                    cached_data = json.loads(result[0])
                    _self.memory_cache[cache_key] = cached_data
                    return cached_data
        except:
            pass
        
        return None
    
    def store_cached_result(self, cache_key: str, result: Dict):
        """Store result in cache"""
        # Store in memory
        self.memory_cache[cache_key] = result
        
        # Store in database
        try:
            conn = production_db.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO processing_cache (cache_key, result_data, created_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (cache_key) DO UPDATE SET
                    result_data = EXCLUDED.result_data,
                    updated_at = NOW()
                """, (cache_key, json.dumps(result)))
                conn.commit()
                production_db.return_connection(conn)
        except:
            pass
    
    def clear_cache(self):
        """Clear all caches"""
        self.memory_cache.clear()
        st.cache_data.clear()

# Global cache manager
cache_manager = CacheManager()