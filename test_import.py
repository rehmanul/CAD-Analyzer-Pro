
#!/usr/bin/env python3
"""
Test script to check if main_production_app.py can be imported
"""

import sys
import traceback

def test_import():
    """Test importing the main application"""
    try:
        print("Testing import of main_production_app...")
        
        # Try to import the module
        import main_production_app
        print("✅ Module imported successfully")
        
        # Try to get the class
        if hasattr(main_production_app, 'ProductionCADAnalyzer'):
            print("✅ ProductionCADAnalyzer class found")
            
            # Try to instantiate
            app = main_production_app.ProductionCADAnalyzer()
            print("✅ Class instantiated successfully")
            
            return True
        else:
            print("❌ ProductionCADAnalyzer class not found in module")
            print(f"Available attributes: {dir(main_production_app)}")
            return False
            
    except SyntaxError as e:
        print(f"❌ Syntax error in main_production_app.py:")
        print(f"Line {e.lineno}: {e.text}")
        print(f"Error: {e.msg}")
        return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1)
