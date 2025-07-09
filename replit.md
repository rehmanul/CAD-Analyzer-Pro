# CAD Analyzer Pro

## Overview

CAD Analyzer Pro is a professional hotel floor plan analyzer built with Streamlit that provides intelligent îlot placement and corridor generation. The application processes DXF/DWG files to extract architectural elements and generates optimal furniture placement with corridor networks. It features advanced visualization capabilities using Plotly and supports PostgreSQL database integration.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit as the primary web framework
- **Visualization**: Plotly for interactive charts and floor plan displays
- **UI Components**: Custom sidebar controls for îlot size distribution and configuration
- **Rendering**: Multiple specialized renderers (SVG, production-grade, architectural)

### Backend Architecture
- **Processing Pipeline**: Modular architecture with specialized processors for different file types
- **File Processing**: Support for DXF, DWG, PDF, and image formats
- **Analysis Engine**: Ultra-high performance analyzer with parallel processing capabilities
- **Spatial Indexing**: Advanced spatial indexing for geometric queries and optimization

### Data Processing Components
- **DXF Processing**: Multiple processors (OptimizedDXFProcessor, RealDXFProcessor, FastDXFProcessor) for different performance requirements
- **Floor Plan Extraction**: Intelligent detection of main floor plan from multi-view files
- **Îlot Placement**: Smart placement algorithms with size distribution and spatial optimization
- **Corridor Generation**: Optimized corridor network generation with pathfinding algorithms

## Key Components

### File Processing
- **UltraHighPerformanceAnalyzer**: Main analysis engine with parallel processing
- **DXF Processors**: Multiple specialized processors for different scenarios
- **Floor Plan Extractors**: Extract main floor plan from complex multi-view files
- **Smart Floor Plan Detector**: Intelligent detection of architectural elements

### Visualization System
- **Production Renderers**: Professional-grade SVG and floor plan renderers
- **Architectural Visualizers**: Create clean architectural drawings matching reference standards
- **Client Expected Visualizer**: Generates exact matches to client expected output
- **Reference Style Visualizers**: Match specific client visual requirements

### Placement Algorithms
- **Smart Îlot Placer**: Intelligent placement using room detection
- **Optimized Îlot Placer**: High-performance placement with spatial indexing
- **Simple Îlot Placer**: Reliable fallback placement system
- **Corridor Generator**: Creates optimal corridor networks between îlots

### Database Integration
- **SQLAlchemy ORM**: Database abstraction layer
- **PostgreSQL Support**: Production database for Render deployment
- **Analysis Storage**: Persistent storage of analysis results and metrics
- **Performance Tracking**: Store processing times and optimization data

## Data Flow

1. **File Upload**: User uploads DXF/DWG files through Streamlit interface
2. **File Processing**: Ultra-high performance analyzer processes files in parallel
3. **Architectural Extraction**: Extract walls, doors, restricted areas, and entrances
4. **Floor Plan Detection**: Identify main floor plan from multi-view files
5. **Îlot Placement**: Generate optimal furniture placement based on size distribution
6. **Corridor Generation**: Create corridor networks connecting îlots
7. **Visualization**: Render professional floor plans with interactive features
8. **Export**: Generate JSON summaries and downloadable results

## External Dependencies

### Core Libraries
- **Streamlit**: Web framework for the application interface
- **Plotly**: Interactive visualization and charting
- **ezdxf**: DXF file processing and manipulation
- **PyMuPDF**: PDF and image processing
- **Shapely**: Geometric operations and spatial analysis
- **OpenCV**: Image processing for floor plan analysis
- **NumPy/SciPy**: Numerical computations and optimization
- **NetworkX**: Graph algorithms for corridor generation

### Database and Storage
- **PostgreSQL**: Production database (psycopg2-binary)
- **SQLAlchemy**: Database ORM and abstraction
- **python-dotenv**: Environment variable management

### Performance and Optimization
- **Concurrent Processing**: ThreadPoolExecutor for parallel operations
- **Spatial Indexing**: STRtree for fast geometric queries
- **Memory Management**: Optimized for 512MB Render limits
- **Caching**: Multiple levels of caching for performance

## Deployment Strategy

### Render.com Configuration
- **Environment**: Production deployment on Render
- **Memory Limits**: Optimized for 512MB memory constraints
- **Database**: PostgreSQL integration with connection pooling
- **File Handling**: Temporary file management for DXF processing
- **Performance**: Optimized requirements for fast startup

### Configuration Files
- **render_config.py**: Production configuration with memory limits
- **requirements_render.txt**: Memory-optimized dependencies
- **Database Setup**: Automatic schema creation and migration

### Performance Optimizations
- **Parallel Processing**: Multi-threaded file processing
- **Memory Management**: Efficient memory usage for large files
- **Caching**: Multiple cache layers for repeated operations
- **Timeout Protection**: Prevents hanging on large files

### Monitoring and Logging
- **Performance Metrics**: Track processing times and memory usage
- **Error Handling**: Comprehensive error handling with fallbacks
- **Data Validation**: Ensures authentic data processing without mock data
- **Production Logging**: Structured logging for debugging and monitoring