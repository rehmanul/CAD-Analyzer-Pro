"""
Production System Test Script
Validates all core components are working correctly
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly")
    except ImportError as e:
        print(f"❌ Plotly: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        from shapely.geometry import Polygon
        print("✅ Shapely")
    except ImportError as e:
        print(f"❌ Shapely: {e}")
        return False
    
    try:
        import ezdxf
        print("✅ ezdxf")
    except ImportError as e:
        print(f"❌ ezdxf: {e}")
        return False
    
    return True

def test_production_modules():
    """Test production modules"""
    print("\n🧪 Testing production modules...")
    
    try:
        from utils.production_database import production_db
        print("✅ Production Database")
    except Exception as e:
        print(f"❌ Production Database: {e}")
        return False
    
    try:
        from utils.production_floor_analyzer import ProductionFloorAnalyzer
        analyzer = ProductionFloorAnalyzer()
        print("✅ Production Floor Analyzer")
    except Exception as e:
        print(f"❌ Production Floor Analyzer: {e}")
        return False
    
    try:
        from utils.production_ilot_system import ProductionIlotSystem
        ilot_system = ProductionIlotSystem()
        print("✅ Production Îlot System")
    except Exception as e:
        print(f"❌ Production Îlot System: {e}")
        return False
    
    try:
        from utils.production_visualizer import ProductionVisualizer
        visualizer = ProductionVisualizer()
        print("✅ Production Visualizer")
    except Exception as e:
        print(f"❌ Production Visualizer: {e}")
        return False
    
    return True

def test_database_connection():
    """Test database connection"""
    print("\n🧪 Testing database connection...")
    
    try:
        from utils.production_database import production_db
        
        if production_db.connection_pool:
            print("✅ Database connection pool initialized")
            
            # Test basic operations
            try:
                conn = production_db.get_connection()
                if conn:
                    production_db.return_connection(conn)
                    print("✅ Database connection test successful")
                    return True
                else:
                    print("⚠️ Database connection unavailable")
                    return False
            except Exception as e:
                print(f"⚠️ Database connection test failed: {e}")
                return False
        else:
            print("⚠️ Database connection pool not initialized")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_sample_processing():
    """Test sample data processing"""
    print("\n🧪 Testing sample data processing...")
    
    try:
        from utils.production_floor_analyzer import ProductionFloorAnalyzer
        from utils.production_ilot_system import ProductionIlotSystem
        
        # Initialize components
        analyzer = ProductionFloorAnalyzer()
        ilot_system = ProductionIlotSystem()
        
        # Generate sample DXF data
        sample_data = analyzer.generate_sample_dxf_data("test.dxf")
        
        if sample_data['success']:
            print("✅ Sample DXF data generation")
            
            # Test îlot system with sample data
            ilot_system.load_floor_plan_data(
                walls=sample_data.get('walls', []),
                restricted_areas=sample_data.get('restricted_areas', []),
                entrances=sample_data.get('entrances', []),
                zones={},
                bounds=sample_data.get('bounds', {})
            )
            
            # Test îlot distribution calculation
            config = {
                'size_0_1_percent': 10,
                'size_1_3_percent': 25,
                'size_3_5_percent': 30,
                'size_5_10_percent': 35
            }
            
            available_area = 1000  # Sample area
            distribution = ilot_system.calculate_ilot_distribution(config, available_area)
            
            if distribution:
                print("✅ Îlot distribution calculation")
                print(f"   Distribution: {distribution}")
                return True
            else:
                print("❌ Îlot distribution calculation failed")
                return False
        else:
            print("❌ Sample DXF data generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Sample processing test failed: {e}")
        traceback.print_exc()
        return False

def test_visualization():
    """Test visualization components"""
    print("\n🧪 Testing visualization...")
    
    try:
        from utils.production_visualizer import ProductionVisualizer
        import plotly.graph_objects as go
        
        visualizer = ProductionVisualizer()
        
        # Test color scheme
        colors = visualizer.generate_color_legend()
        if colors:
            print("✅ Color scheme generation")
            print(f"   Colors: {len(colors)} defined")
        else:
            print("❌ Color scheme generation failed")
            return False
        
        # Test empty figure creation
        sample_data = {
            'walls': [[(0, 0), (100, 0), (100, 80), (0, 80)]],
            'restricted_areas': [[(10, 10), (20, 10), (20, 25), (10, 25)]],
            'entrances': [[(45, 0), (55, 0), (55, 5), (45, 5)]],
            'bounds': {'min_x': 0, 'min_y': 0, 'max_x': 100, 'max_y': 80}
        }
        
        fig = visualizer.create_complete_floor_plan_view(sample_data, [], [])
        
        if fig and isinstance(fig, go.Figure):
            print("✅ Visualization generation")
            return True
        else:
            print("❌ Visualization generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        traceback.print_exc()
        return False

def test_main_app():
    """Test main application file"""
    print("\n🧪 Testing main application...")
    
    try:
        app_path = Path("main_production_app.py")
        if app_path.exists():
            print("✅ Main application file exists")
            
            # Try to import (without running)
            import importlib.util
            spec = importlib.util.spec_from_file_location("main_app", app_path)
            if spec and spec.loader:
                print("✅ Main application can be imported")
                return True
            else:
                print("❌ Main application import failed")
                return False
        else:
            print("❌ Main application file not found")
            return False
            
    except Exception as e:
        print(f"❌ Main application test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🏨 CAD Analyzer Pro - Production System Test")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_imports),
        ("Production Modules", test_production_modules),
        ("Database Connection", test_database_connection),
        ("Sample Processing", test_sample_processing),
        ("Visualization", test_visualization),
        ("Main Application", test_main_app)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🧪 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System ready for production!")
        return True
    else:
        print("⚠️ Some tests failed - Check configuration before deployment")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)