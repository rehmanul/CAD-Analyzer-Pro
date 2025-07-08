#!/bin/bash
# Render deployment script for CAD Analyzer Pro

echo "🚀 Starting Render deployment preparation..."

# Copy render-specific requirements
cp requirements_render.txt requirements.txt

# Set environment variables for Render
export PYTHONPATH="/opt/render/project/src"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_SERVER_PORT="10000"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p .streamlit
mkdir -p utils
mkdir -p attached_assets

# Set proper permissions
chmod -R 755 .
chmod +x streamlit_app.py

echo "✅ Render deployment preparation complete!"
echo "🌐 Ready to deploy to Render.com"
echo ""
echo "Next steps:"
echo "1. Create new Web Service on Render"
echo "2. Connect your GitHub repository"
echo "3. Set build command: pip install -r requirements.txt"
echo "4. Set start command: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0"
echo "5. Add environment variables from render.yaml"
echo "6. Deploy!"