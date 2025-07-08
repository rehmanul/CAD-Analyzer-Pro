# 🚀 Render Deployment Guide for CAD Analyzer Pro

This guide will help you deploy your professional Floor Plan Analyzer to Render.com.

## 📋 Prerequisites

- GitHub account with your code repository
- Render.com account (free tier available)
- PostgreSQL database (can be created on Render)

## 🔧 Deployment Steps

### 1. Repository Setup

Ensure your repository contains these files:
- `render.yaml` - Render service configuration
- `requirements_render.txt` - Optimized dependencies
- `Dockerfile` - Container configuration
- `app_render.py` - Render-optimized entry point
- `render_config.py` - Production configuration

### 2. Create Web Service on Render

1. **Login to Render.com**
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository**
4. **Configure the service:**

```yaml
Name: cad-analyzer-pro
Environment: Python 3
Region: Oregon (or your preferred region)
Branch: main
Build Command: pip install -r requirements_render.txt
Start Command: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

### 3. Environment Variables

Add these environment variables in Render dashboard:

```bash
# Application Settings
PYTHONPATH=/opt/render/project/src
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=$PORT
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Performance Settings
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
STREAMLIT_SERVER_ENABLE_CORS=false

# Theme Settings
STREAMLIT_THEME_BASE=light
STREAMLIT_THEME_PRIMARY_COLOR=#0066cc

# Application Features
PSUTIL_AVAILABLE=true
CACHE_ENABLED=true
MAX_FILE_SIZE=200
```

### 4. Database Setup (Optional)

If you want PostgreSQL database:

1. **Create PostgreSQL service on Render:**
   - Name: `cad-analyzer-db`
   - Database: `cad_analyzer_prod`
   - User: `cad_analyzer_user`

2. **Add database environment variables:**
   ```bash
   DATABASE_URL=<provided-by-render>
   DATABASE_HOST=<provided-by-render>
   DATABASE_PORT=5432
   DATABASE_NAME=cad_analyzer_prod
   DATABASE_USER=cad_analyzer_user
   DATABASE_PASSWORD=<provided-by-render>
   ```

### 5. Advanced Configuration

For production optimization, add:

```bash
# Security
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
STREAMLIT_CLIENT_TOOLBAR_MODE=minimal

# Performance
STREAMLIT_RUNNER_FAST_RERUNS=true
STREAMLIT_RUNNER_INSTALL_TRACER=false

# Logging
STREAMLIT_LOGGER_LEVEL=info
RENDER_ENV=production
```

## 🎯 Deployment Process

1. **Push your code to GitHub**
2. **Render will automatically build and deploy**
3. **Wait for deployment to complete** (usually 5-10 minutes)
4. **Access your app at the provided Render URL**

## 🔍 Monitoring & Debugging

### Logs
- Check deployment logs in Render dashboard
- Monitor runtime logs for errors
- Use Render's built-in monitoring tools

### Performance
- Monitor memory usage (1GB free tier limit)
- Check response times
- Optimize file upload sizes if needed

### Common Issues
1. **Build fails**: Check requirements_render.txt
2. **Memory issues**: Optimize image processing
3. **Database connection**: Verify DATABASE_URL

## 🌟 Production Features

Your deployed app will have:
- ✅ Professional domain (your-app.onrender.com)
- ✅ SSL certificate (HTTPS)
- ✅ Automatic deployments from GitHub
- ✅ Health monitoring
- ✅ Log aggregation
- ✅ Database backup (PostgreSQL)

## 📊 Cost Estimation

**Free Tier Limits:**
- 750 hours/month runtime
- 1GB RAM
- 100GB bandwidth
- PostgreSQL: 1GB storage

**Paid Plans Start at $7/month:**
- Always-on service
- More RAM/CPU
- Priority support
- Custom domains

## 🎉 Success!

Once deployed, your CAD Analyzer Pro will be accessible at:
`https://your-app-name.onrender.com`

Share this URL with clients for professional floor plan analysis!

## 📞 Support

For deployment issues:
- Check Render documentation
- Review application logs
- Verify environment variables
- Test with sample DXF files

Your professional Floor Plan Analyzer is now ready for production use!