
# CAD Analyzer Pro

Professional hotel floor plan analyzer with îlot placement and corridor generation.

## Features

- Floor plan analysis (DXF/DWG support)
- Intelligent îlot placement with size distribution
- Corridor generation between facing rows
- Professional visualizations with Plotly
- Export functionality (JSON/Summary)
- PostgreSQL database support

## Usage

1. Upload a floor plan file (DXF/DWG recommended)
2. Configure îlot size distribution in sidebar
3. Generate îlot placement
4. Generate corridor network
5. Export results

## Running

```bash
streamlit run streamlit_app.py --server.port 5000 --server.address 0.0.0.0
```

## Deployment

Configured for Render.com deployment with PostgreSQL database.
