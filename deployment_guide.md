# Deployment Guide - CAD Analyzer Pro

## For Streamlit Share Deployment

### Files needed:
1. `streamlit_app.py` - Main entry point (already configured)
2. `requirements_clean.txt` - Clean requirements without duplicates
3. `packages.txt` - System dependencies (if needed)

### Requirements.txt for Streamlit Share:
```
streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
shapely>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
ezdxf>=1.0.0
PyMuPDF>=1.23.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
reportlab>=4.0.0
matplotlib>=3.7.0
networkx>=3.0
```

### Key Changes Made:
1. ✅ Removed `psutil` dependency - made optional with graceful fallback
2. ✅ Removed `psycopg2-binary` - not needed for basic deployment
3. ✅ Removed `sqlalchemy` - not needed for basic deployment
4. ✅ Fixed all import errors and syntax issues
5. ✅ Added proper error handling for missing dependencies

## For Render.com Deployment

### Advantages of Render.com:
- Full control over system packages
- Better support for complex dependencies
- Can include psutil and database libraries
- More suitable for production applications
- Better performance and reliability

### Files needed:
1. `app.py` - Main entry point
2. `requirements.txt` - Full requirements (current file)
3. `render.yaml` - Render configuration

### Render.yaml configuration:
```yaml
services:
  - type: web
    name: cad-analyzer-pro
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: STREAMLIT_SERVER_HEADLESS
        value: true
      - key: STREAMLIT_SERVER_PORT
        value: $PORT
```

## Recommendation

For this enterprise-grade application with complex dependencies, **Render.com is the better choice** because:

1. **Full dependency support** - Can install psutil, psycopg2-binary, and all system packages
2. **Better performance** - More resources and better infrastructure
3. **Database support** - Can connect to PostgreSQL databases
4. **Production-ready** - Better for enterprise applications
5. **System packages** - Can install OpenCV and other system dependencies

Streamlit Share is better for simple demos, but this application needs the full capabilities.