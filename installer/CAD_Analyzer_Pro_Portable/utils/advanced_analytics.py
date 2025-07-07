
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import sqlite3
from datetime import datetime, timedelta
import json
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    """Advanced analytics and business intelligence for floor plan data"""
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self.init_analytics_db()
    
    def init_analytics_db(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Analytics events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                project_id TEXT,
                event_type TEXT,
                event_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_unit TEXT,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                algorithm_version TEXT
            )
        ''')
        
        # Usage statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_statistics (
                date TEXT PRIMARY KEY,
                total_users INTEGER DEFAULT 0,
                active_users INTEGER DEFAULT 0,
                total_projects INTEGER DEFAULT 0,
                new_projects INTEGER DEFAULT 0,
                total_analyses INTEGER DEFAULT 0,
                avg_analysis_time REAL DEFAULT 0,
                popular_features TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def track_event(self, user_id: str, project_id: str, event_type: str, 
                   event_data: Dict[str, Any], session_id: str = None):
        """Track analytics event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analytics_events (user_id, project_id, event_type, event_data, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, project_id, event_type, json.dumps(event_data), session_id))
        
        conn.commit()
        conn.close()
    
    def calculate_project_kpis(self, project_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Key Performance Indicators for a project"""
        analysis_results = project_data.get('analysis_results', {})
        ilot_results = project_data.get('ilot_results', [])
        corridor_results = project_data.get('corridor_results', [])
        
        # Space utilization metrics
        spatial_metrics = analysis_results.get('spatial_metrics', {})
        total_area = spatial_metrics.get('total_area', 0)
        usable_area = spatial_metrics.get('usable_area', 0)
        
        # Îlot efficiency metrics
        total_ilots = len(ilot_results)
        ilot_area = sum(ilot.get('area', 0) for ilot in ilot_results)
        avg_ilot_size = ilot_area / total_ilots if total_ilots > 0 else 0
        
        # Corridor efficiency metrics
        total_corridors = len(corridor_results)
        corridor_length = sum(c.get('length', 0) for c in corridor_results)
        avg_corridor_width = np.mean([c.get('width', 0) for c in corridor_results]) if corridor_results else 0
        
        # Calculate KPIs
        kpis = {
            'space_utilization_ratio': (usable_area / total_area * 100) if total_area > 0 else 0,
            'ilot_density': (total_ilots / total_area * 100) if total_area > 0 else 0,
            'circulation_efficiency': (corridor_length / total_area) if total_area > 0 else 0,
            'average_ilot_size': avg_ilot_size,
            'corridor_coverage': (corridor_length / (total_area ** 0.5)) if total_area > 0 else 0,
            'accessibility_score': spatial_metrics.get('accessibility_score', 0),
            'optimization_score': self._calculate_optimization_score(analysis_results, ilot_results),
            'cost_efficiency': self._estimate_cost_efficiency(project_data),
            'sustainability_score': self._calculate_sustainability_score(project_data),
            'compliance_score': self._calculate_compliance_score(analysis_results)
        }
        
        # Store metrics in database
        self._store_performance_metrics(project_data.get('project_id'), kpis)
        
        return kpis
    
    def _calculate_optimization_score(self, analysis_results: Dict[str, Any], 
                                    ilot_results: List[Dict[str, Any]]) -> float:
        """Calculate overall optimization score"""
        weights = {
            'space_utilization': 0.25,
            'accessibility': 0.20,
            'circulation': 0.20,
            'safety': 0.15,
            'flexibility': 0.10,
            'aesthetics': 0.10
        }
        
        scores = {
            'space_utilization': min(100, analysis_results.get('spatial_metrics', {}).get('space_efficiency', 0) * 1.2),
            'accessibility': analysis_results.get('spatial_metrics', {}).get('accessibility_score', 0),
            'circulation': analysis_results.get('spatial_metrics', {}).get('circulation_efficiency', 0),
            'safety': 85.0,  # Base safety score - would be calculated from actual safety analysis
            'flexibility': self._calculate_flexibility_score(ilot_results),
            'aesthetics': self._calculate_aesthetics_score(ilot_results)
        }
        
        return sum(scores[key] * weights[key] for key in weights.keys())
    
    def _calculate_flexibility_score(self, ilot_results: List[Dict[str, Any]]) -> float:
        """Calculate layout flexibility score"""
        if not ilot_results:
            return 0
        
        # Analyze size distribution for flexibility
        size_categories = defaultdict(int)
        for ilot in ilot_results:
            category = ilot.get('size_category', 'medium')
            size_categories[category] += 1
        
        # Good flexibility has balanced distribution
        total_ilots = len(ilot_results)
        if total_ilots == 0:
            return 0
        
        # Calculate distribution entropy (higher = more flexible)
        entropy = 0
        for count in size_categories.values():
            p = count / total_ilots
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize to 0-100 scale
        max_entropy = np.log2(len(size_categories)) if size_categories else 1
        return (entropy / max_entropy * 100) if max_entropy > 0 else 0
    
    def _calculate_aesthetics_score(self, ilot_results: List[Dict[str, Any]]) -> float:
        """Calculate aesthetic arrangement score"""
        if len(ilot_results) < 2:
            return 50
        
        # Calculate spatial distribution uniformity
        positions = [(ilot['position']['x'], ilot['position']['y']) for ilot in ilot_results]
        
        # Calculate average distances between îlots
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                             (positions[i][1] - positions[j][1])**2)
                distances.append(dist)
        
        if not distances:
            return 50
        
        # Good aesthetics = consistent spacing
        std_dev = np.std(distances)
        mean_dist = np.mean(distances)
        
        # Lower coefficient of variation = better aesthetics
        cv = std_dev / mean_dist if mean_dist > 0 else 1
        aesthetics_score = max(0, 100 * (1 - cv))
        
        return min(100, aesthetics_score)
    
    def _estimate_cost_efficiency(self, project_data: Dict[str, Any]) -> float:
        """Estimate cost efficiency score"""
        analysis_results = project_data.get('analysis_results', {})
        ilot_results = project_data.get('ilot_results', [])
        
        # Simple cost model based on space utilization and material efficiency
        spatial_metrics = analysis_results.get('spatial_metrics', {})
        space_efficiency = spatial_metrics.get('space_efficiency', 0)
        
        # More îlots in smaller space = higher cost efficiency
        total_area = spatial_metrics.get('total_area', 0)
        ilot_count = len(ilot_results)
        
        if total_area > 0:
            density_score = min(100, (ilot_count / total_area) * 1000)
            cost_efficiency = (space_efficiency * 0.6 + density_score * 0.4)
        else:
            cost_efficiency = space_efficiency
        
        return min(100, cost_efficiency)
    
    def _calculate_sustainability_score(self, project_data: Dict[str, Any]) -> float:
        """Calculate sustainability score"""
        analysis_results = project_data.get('analysis_results', {})
        
        # Factors affecting sustainability
        natural_light_access = 75  # Would be calculated from window analysis
        energy_efficiency = 80     # Would be calculated from HVAC optimization
        material_efficiency = analysis_results.get('spatial_metrics', {}).get('space_efficiency', 0)
        
        # Weighted sustainability score
        sustainability = (natural_light_access * 0.3 + 
                         energy_efficiency * 0.4 + 
                         material_efficiency * 0.3)
        
        return min(100, sustainability)
    
    def _calculate_compliance_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate building code compliance score"""
        spatial_metrics = analysis_results.get('spatial_metrics', {})
        
        # Check various compliance factors
        accessibility_compliance = spatial_metrics.get('accessibility_score', 0)
        safety_compliance = 85  # Would be calculated from actual safety analysis
        fire_code_compliance = 90  # Would be calculated from egress analysis
        
        # Weighted compliance score
        compliance = (accessibility_compliance * 0.4 + 
                     safety_compliance * 0.3 + 
                     fire_code_compliance * 0.3)
        
        return min(100, compliance)
    
    def _store_performance_metrics(self, project_id: str, metrics: Dict[str, float]):
        """Store performance metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in metrics.items():
            cursor.execute('''
                INSERT INTO performance_metrics (project_id, metric_name, metric_value, metric_unit, algorithm_version)
                VALUES (?, ?, ?, ?, ?)
            ''', (project_id, metric_name, metric_value, 'percentage', '1.0'))
        
        conn.commit()
        conn.close()
    
    def generate_benchmark_report(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark comparison report"""
        current_kpis = self.calculate_project_kpis(project_data)
        
        # Get industry benchmarks (simulated - would come from real data)
        industry_benchmarks = {
            'space_utilization_ratio': {'average': 75, 'excellent': 85, 'poor': 60},
            'ilot_density': {'average': 12, 'excellent': 15, 'poor': 8},
            'circulation_efficiency': {'average': 0.15, 'excellent': 0.12, 'poor': 0.20},
            'accessibility_score': {'average': 80, 'excellent': 95, 'poor': 65},
            'optimization_score': {'average': 75, 'excellent': 90, 'poor': 60},
            'cost_efficiency': {'average': 70, 'excellent': 85, 'poor': 55},
            'sustainability_score': {'average': 72, 'excellent': 88, 'poor': 55},
            'compliance_score': {'average': 85, 'excellent': 95, 'poor': 70}
        }
        
        # Calculate performance vs benchmarks
        benchmark_analysis = {}
        for metric, value in current_kpis.items():
            if metric in industry_benchmarks:
                benchmark = industry_benchmarks[metric]
                
                if value >= benchmark['excellent']:
                    performance = 'Excellent'
                    percentile = 90
                elif value >= benchmark['average']:
                    performance = 'Above Average'
                    percentile = 70
                elif value >= benchmark['poor']:
                    performance = 'Average'
                    percentile = 50
                else:
                    performance = 'Below Average'
                    percentile = 25
                
                benchmark_analysis[metric] = {
                    'current_value': value,
                    'industry_average': benchmark['average'],
                    'performance_level': performance,
                    'percentile': percentile,
                    'improvement_potential': max(0, benchmark['excellent'] - value)
                }
        
        return {
            'current_kpis': current_kpis,
            'benchmark_analysis': benchmark_analysis,
            'overall_score': np.mean(list(current_kpis.values())),
            'top_strengths': self._identify_top_metrics(benchmark_analysis, top=True),
            'improvement_areas': self._identify_top_metrics(benchmark_analysis, top=False),
            'recommendations': self._generate_benchmark_recommendations(benchmark_analysis)
        }
    
    def _identify_top_metrics(self, benchmark_analysis: Dict[str, Any], top: bool = True) -> List[str]:
        """Identify top performing or improvement areas"""
        metric_scores = []
        for metric, data in benchmark_analysis.items():
            score = data['percentile']
            metric_scores.append((metric, score))
        
        # Sort by score
        metric_scores.sort(key=lambda x: x[1], reverse=top)
        
        return [metric for metric, score in metric_scores[:3]]
    
    def _generate_benchmark_recommendations(self, benchmark_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on benchmark analysis"""
        recommendations = []
        
        for metric, data in benchmark_analysis.items():
            if data['performance_level'] in ['Below Average', 'Average']:
                if metric == 'space_utilization_ratio':
                    recommendations.append({
                        'metric': metric,
                        'recommendation': 'Consider optimizing îlot placement to increase space utilization',
                        'priority': 'High' if data['percentile'] < 30 else 'Medium'
                    })
                elif metric == 'accessibility_score':
                    recommendations.append({
                        'metric': metric,
                        'recommendation': 'Improve corridor widths and entrance accessibility',
                        'priority': 'High' if data['percentile'] < 30 else 'Medium'
                    })
                elif metric == 'circulation_efficiency':
                    recommendations.append({
                        'metric': metric,
                        'recommendation': 'Optimize corridor network to reduce travel distances',
                        'priority': 'Medium'
                    })
        
        return recommendations
    
    def create_performance_dashboard(self, project_data: Dict[str, Any]) -> go.Figure:
        """Create comprehensive performance dashboard"""
        benchmark_report = self.generate_benchmark_report(project_data)
        current_kpis = benchmark_report['current_kpis']
        benchmark_analysis = benchmark_report['benchmark_analysis']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'KPI Overview', 'Benchmark Comparison',
                'Performance Trends', 'Risk Assessment',
                'Cost Analysis', 'Sustainability Metrics'
            ],
            specs=[
                [{"type": "bar"}, {"type": "radar"}],
                [{"type": "scatter"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. KPI Overview (Bar chart)
        kpi_names = list(current_kpis.keys())
        kpi_values = list(current_kpis.values())
        
        fig.add_trace(
            go.Bar(
                x=kpi_names,
                y=kpi_values,
                name="Current KPIs",
                marker_color='steelblue'
            ),
            row=1, col=1
        )
        
        # 2. Benchmark Comparison (Radar chart)
        radar_metrics = []
        current_values = []
        benchmark_values = []
        
        for metric, data in benchmark_analysis.items():
            radar_metrics.append(metric.replace('_', ' ').title())
            current_values.append(data['current_value'])
            benchmark_values.append(data['industry_average'])
        
        fig.add_trace(
            go.Scatterpolar(
                r=current_values,
                theta=radar_metrics,
                fill='toself',
                name='Current Performance',
                line_color='blue'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatterpolar(
                r=benchmark_values,
                theta=radar_metrics,
                fill='toself',
                name='Industry Average',
                line_color='red',
                opacity=0.6
            ),
            row=1, col=2
        )
        
        # 3. Performance Trends (Mock data - would be real historical data)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        performance_trend = [70, 72, 75, 78, 80, current_kpis.get('optimization_score', 75)]
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=performance_trend,
                mode='lines+markers',
                name='Performance Trend',
                line_color='green'
            ),
            row=2, col=1
        )
        
        # 4. Risk Assessment (Gauge)
        risk_score = 100 - current_kpis.get('compliance_score', 85)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Risk Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if risk_score > 30 else "orange" if risk_score > 15 else "green"},
                    'steps': [
                        {'range': [0, 15], 'color': "lightgreen"},
                        {'range': [15, 30], 'color': "yellow"},
                        {'range': [30, 100], 'color': "lightcoral"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        # 5. Cost Analysis
        cost_categories = ['Space Efficiency', 'Material Usage', 'Labor Optimization', 'Energy Efficiency']
        cost_savings = [15, 12, 8, 10]  # Percentage savings
        
        fig.add_trace(
            go.Bar(
                x=cost_categories,
                y=cost_savings,
                name="Cost Savings %",
                marker_color='lightblue'
            ),
            row=3, col=1
        )
        
        # 6. Sustainability Metrics
        sustainability_categories = ['Energy', 'Materials', 'Space', 'Accessibility']
        sustainability_scores = [
            current_kpis.get('sustainability_score', 75),
            current_kpis.get('cost_efficiency', 70),
            current_kpis.get('space_utilization_ratio', 75),
            current_kpis.get('accessibility_score', 80)
        ]
        
        fig.add_trace(
            go.Bar(
                x=sustainability_categories,
                y=sustainability_scores,
                name="Sustainability Scores",
                marker_color='lightgreen'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Professional Floor Plan Performance Dashboard",
            showlegend=True
        )
        
        return fig
    
    def export_analytics_report(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export comprehensive analytics report"""
        benchmark_report = self.generate_benchmark_report(project_data)
        
        report = {
            'executive_summary': {
                'overall_score': benchmark_report['overall_score'],
                'performance_level': self._get_performance_level(benchmark_report['overall_score']),
                'key_strengths': benchmark_report['top_strengths'],
                'improvement_areas': benchmark_report['improvement_areas']
            },
            'detailed_metrics': benchmark_report['current_kpis'],
            'benchmark_comparison': benchmark_report['benchmark_analysis'],
            'recommendations': benchmark_report['recommendations'],
            'cost_benefit_analysis': self._generate_cost_benefit_analysis(project_data),
            'roi_projection': self._calculate_roi_projection(benchmark_report),
            'implementation_roadmap': self._create_implementation_roadmap(benchmark_report)
        }
        
        return report
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level description"""
        if score >= 85:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 65:
            return "Average"
        else:
            return "Needs Improvement"
    
    def _generate_cost_benefit_analysis(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cost-benefit analysis"""
        analysis_results = project_data.get('analysis_results', {})
        spatial_metrics = analysis_results.get('spatial_metrics', {})
        
        # Estimated costs and benefits (would be calculated from real data)
        implementation_cost = 50000  # Base implementation cost
        space_efficiency = spatial_metrics.get('space_efficiency', 0)
        
        # Calculate benefits
        space_savings = space_efficiency * 1000  # $1000 per efficiency point
        operational_savings = space_efficiency * 500  # Annual operational savings
        productivity_gains = space_efficiency * 750  # Productivity improvements
        
        total_benefits = space_savings + operational_savings + productivity_gains
        roi = ((total_benefits - implementation_cost) / implementation_cost * 100) if implementation_cost > 0 else 0
        
        return {
            'implementation_cost': implementation_cost,
            'annual_benefits': {
                'space_savings': space_savings,
                'operational_savings': operational_savings,
                'productivity_gains': productivity_gains,
                'total': total_benefits
            },
            'roi_percentage': roi,
            'payback_period_months': (implementation_cost / (total_benefits / 12)) if total_benefits > 0 else 0
        }
    
    def _calculate_roi_projection(self, benchmark_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate return on investment projection"""
        current_score = benchmark_report['overall_score']
        potential_score = 90  # Target improvement score
        
        improvement_potential = potential_score - current_score
        estimated_value_increase = improvement_potential * 2000  # $2000 per point improvement
        
        return {
            'current_performance_value': current_score * 2000,
            'potential_performance_value': potential_score * 2000,
            'value_increase_potential': estimated_value_increase,
            'confidence_level': 85
        }
    
    def _create_implementation_roadmap(self, benchmark_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create implementation roadmap"""
        recommendations = benchmark_report['recommendations']
        
        roadmap = []
        for i, rec in enumerate(recommendations):
            phase = f"Phase {i + 1}"
            duration = "2-4 weeks" if rec['priority'] == 'High' else "4-6 weeks"
            
            roadmap.append({
                'phase': phase,
                'focus_area': rec['metric'],
                'activities': [rec['recommendation']],
                'duration': duration,
                'priority': rec['priority'],
                'expected_impact': 'High' if rec['priority'] == 'High' else 'Medium'
            })
        
        return roadmap

# Initialize global analytics instance
analytics_service = AdvancedAnalytics()
