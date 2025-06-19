import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import numpy as np
import math

# Page configuration
st.set_page_config(
    page_title="KPI Performance Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for attractive UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main > div {
        padding-top: 1rem;
    }

    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.8);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .total-score-container {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .total-score {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .total-score.excellent {
        background: linear-gradient(45deg, #00C851, #007E33);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .total-score.good {
        background: linear-gradient(45deg, #2BBBAD, #1C7973);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .total-score.warning {
        background: linear-gradient(45deg, #FF8800, #CC6600);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .total-score.critical {
        background: linear-gradient(45deg, #FF4444, #CC0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .score-label {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.5rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        padding: 2rem 0;
    }

    .kpi-card {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }

    .kpi-card.critical {
        background: linear-gradient(135deg, rgba(255,68,68,0.1), rgba(255,68,68,0.05));
        border: 2px solid rgba(255,68,68,0.3);
        animation: pulse-red 2s infinite;
    }

    .kpi-card.good {
        background: linear-gradient(135deg, rgba(0,200,81,0.1), rgba(0,200,81,0.05));
        border: 2px solid rgba(0,200,81,0.3);
    }

    @keyframes pulse-red {
        0%, 100% { 
            box-shadow: 0 10px 30px rgba(255,68,68,0.2);
            border-color: rgba(255,68,68,0.3);
        }
        50% { 
            box-shadow: 0 15px 40px rgba(255,68,68,0.4);
            border-color: rgba(255,68,68,0.6);
        }
    }

    .kpi-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 1rem;
        text-align: center;
    }

    .circle-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }

    .details-card {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.1);
    }

    .metric-label {
        font-weight: 600;
        color: #34495e;
    }

    .metric-value {
        color: #FFFFFF;
        font-weight: 500;
    }

    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-good {
        background: linear-gradient(45deg, #00C851, #007E33);
        color: white;
    }

    .status-critical {
        background: linear-gradient(45deg, #FF4444, #CC0000);
        color: white;
    }

    .sidebar .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }

    .analytics-container {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }

    .section-header {
        color: white;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Enhanced Plotly styling */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


def load_csv_data(uploaded_file):
    """Load CSV data and clean it"""
    try:
        # Read CSV with proper handling of extra columns
        df = pd.read_csv(uploaded_file)
        # Clean the data - remove rows where KPI is NaN or empty
        df = df.dropna(subset=['KPI'])
        df = df[df['KPI'].str.strip() != '']
        # Remove the 'Total' row if it exists
        df = df[df['KPI'] != 'Total']
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        # Handle the 'Achieved KPI' column (it might be named differently)
        if 'Achieved KPI' in df.columns:
            df['Achieved'] = df['Achieved KPI']
        # Clean the Remakrs column (handle typo)
        if 'Remakrs' in df.columns:
            df['Remarks'] = df['Remakrs']
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return None


def load_default_data():
    """Load default KPI data"""
    data = {
        'KPI': [
            'SOPs Training', 'Inventory Turnover', 'Planning Forecast Accuracy (Event Based)(A)',
            'Monthly Purchase Local OTIF(B)', 'Monthly Purchase Import OTIF(B)',
            'Logistics Cost', 'Cost Saving', '5S Implementation', 'Kaizen',
            'ATS Compliance', 'Sops Compliance'
        ],
        'Formula': [
            'No of training delivered/no of training planned',
            'COGS Value/Average Closing Inventory',
            'Actual Production/Planned Production',
            'Actual Purchase Local (No.)/Planned Purchase Local (No.)*100',
            'Actual Import (No of Shipments)/Planned Import (No of Shipments)*100',
            'Actual/Budgeted',
            'Budgeted- Actual',
            'Compliance of 5S',
            'Actual/Total',
            'Total done task/Total Task',
            'Non compliance of SOPs/total no of SOPs audited*100'
        ],
        'Measuring unit': ['Num', 'Num', '%', '%', '%', 'PKR', '%', '%', 'Num', '%', '%'],
        'Weightage': ['10%', '35%', '30%', '4%', '4%', '4%', '3%', '3%', '3%', '2%', '2%'],
        'Target': [10, 0.5, '95%', '95%', '100%', 1065000, '5%', '100%', 1, '100%', '100%'],
        'Actual': [10, 0.11, '31%', '91%', '10%', 891571, '14%', '88%', 1, '99%', '87%'],
        'Achieved': ['10%', '8%', '9.8%', '3.8%', '0.4%', '4.8%', '3.0%', '2.6%', '3.0%', '2.0%', '1.7%'],
        'Remarks': [
            'less than 10 will be red, above and equal to 10 will be green.',
            'Less than 35% will be red , above and equal to 35%',
            'Less than 30% will be red , above and equal to 30%',
            'Less than 4% will be red , above and equal to 4% will be green',
            'Less than 4% will be red , above and equal to 4% will be green',
            'Less than 4% will be red , above and equal to 4% will be green',
            'Less than 3% will be red , above and equal to 3% will be green',
            'less than 3% will be green , above and equal to 3% will be red.',
            'less than 3% will be green , above and equal to 3% will be red.',
            'less than 2% will be green , above and equal to 2% will be red.',
            'less than 2% will be green , above and equal to 2% will be red.'
        ]
    }
    return pd.DataFrame(data)


def parse_percentage(value):
    """Convert percentage string to float"""
    if pd.isna(value):
        return 0
    if isinstance(value, str) and '%' in value:
        return float(value.replace('%', '').strip())
    try:
        return float(value)
    except:
        return 0


def determine_kpi_status(kpi_name, achieved_value, weightage_value):
    """Determine if KPI should be red or green based on business rules"""
    achieved = parse_percentage(achieved_value)
    weightage = parse_percentage(weightage_value)

    # Special cases where logic is inverted (less is better)
    inverted_kpis = ['5S Implementation', 'Kaizen', 'ATS Compliance', 'Sops Compliance']

    if kpi_name in inverted_kpis:
        return achieved < weightage  # Good if achieved is less than weightage
    else:
        return achieved >= weightage  # Good if achieved is greater than or equal to weightage


def create_beautiful_circle(kpi_name, achieved_value, weightage_value, is_good):
    """Create a beautiful animated KPI circle using Plotly"""
    achieved_num = parse_percentage(achieved_value)
    color_good = '#00C851'
    color_bad = '#FF4444'

    # Create the main circle with gradient effect
    fig = go.Figure()

    # Background circle (larger, subtle)
    fig.add_shape(
        type="circle",
        x0=-0.1, y0=-0.1, x1=1.1, y1=1.1,
        fillcolor='rgba(255,255,255,0.1)',
        line=dict(color='rgba(255,255,255,0.2)', width=2),
        opacity=0.3
    )

    # Main circle with gradient effect
    main_color = color_good if is_good else color_bad
    fig.add_shape(
        type="circle",
        x0=0, y0=0, x1=1, y1=1,
        fillcolor=main_color,
        line=dict(color=main_color, width=3),
        opacity=0.9 if is_good else 0.8
    )

    # Inner highlight circle for 3D effect
    fig.add_shape(
        type="circle",
        x0=0.15, y0=0.15, x1=0.85, y1=0.85,
        fillcolor='rgba(255,255,255,0.2)',
        line=dict(color='rgba(255,255,255,0.3)', width=1),
        opacity=0.6
    )

    # Achievement percentage as a progress ring
    theta = np.linspace(0, 2 * np.pi * min(achieved_num / 100, 1), 100)
    x_ring = 0.5 + 0.35 * np.cos(theta)
    y_ring = 0.5 + 0.35 * np.sin(theta)

    if len(theta) > 1:
        fig.add_trace(go.Scatter(
            x=x_ring, y=y_ring,
            mode='lines',
            line=dict(color='rgba(255,255,255,0.8)', width=8),
            showlegend=False
        ))

    # Center text - KPI value
    fig.add_annotation(
        x=0.5, y=0.55,
        text=f"<b>{achieved_value}</b>",
        showarrow=False,
        font=dict(size=28, color="white", family="Inter"),
        align="center"
    )

    # Status indicator
    status_text = "✓" if is_good else "⚠"
    status_color = "rgba(255,255,255,0.9)" if is_good else "rgba(255,255,255,1)"
    fig.add_annotation(
        x=0.5, y=0.4,
        text=f"<b>{status_text}</b>",
        showarrow=False,
        font=dict(size=20, color=status_color, family="Inter"),
        align="center"
    )

    # Add subtle animation effect for bad performance
    if not is_good:
        # Add pulsing effect with multiple circles
        for i, opacity in enumerate([0.1, 0.05, 0.02]):
            fig.add_shape(
                type="circle",
                x0=-0.05 * i, y0=-0.05 * i, x1=1 + 0.05 * i, y1=1 + 0.05 * i,
                fillcolor='rgba(0,0,0,0)',  # Changed from 'transparent' to 'rgba(0,0,0,0)'
                line=dict(color=color_bad, width=2 - i * 0.5),
                opacity=opacity
            )

    fig.update_layout(
        width=200, height=200,
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.2, 1.2]),
        yaxis=dict(visible=False, range=[-0.2, 1.2]),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter")
    )

    return fig


def get_score_class(score):
    """Determine score class based on total percentage"""
    if score >= 85:
        return "excellent"
    elif score >= 70:
        return "good"
    elif score >= 50:
        return "warning"
    else:
        return "critical"


def main():
    # Header
    st.markdown("""
        <style>
            .main-header {
                color: #FFFFFF;
                background-color: rgba(0, 0, 0, 0); /* Fully transparent background */
                font-size: 36px;
                font-weight: bold;
                text-align: center;
                margin-top: 20px;
            }
        </style>
        <h1 class="main-header">🎯 KPI Performance Dashboard</h1>
    """, unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time monitoring of key performance indicators</p>', unsafe_allow_html=True)

    # Sidebar for file upload
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file with KPI data",
            type=['csv'],
            help="Upload a CSV file with your KPI data"
        )

        st.markdown("### 📋 Expected Columns:")
        st.markdown("""
        - **KPI**: Name of the KPI
        - **Formula**: Calculation method
        - **Measuring unit**: Unit of measurement
        - **Weightage**: Weight percentage
        - **Target**: Target value
        - **Actual**: Actual achieved value
        - **Achieved**: Achievement percentage
        - **Remarks**: Performance criteria
        """)

        if uploaded_file:
            st.success("✅ File uploaded successfully!")
        else:
            st.info("📊 Using sample data")

    # Load data
    if uploaded_file is not None:
        df = load_csv_data(uploaded_file)
        if df is None:
            df = load_default_data()
            st.error("❌ Error loading CSV. Using default data.")
    else:
        df = load_default_data()

    # Validate required columns
    required_columns = ['KPI', 'Weightage', 'Achieved', 'Remarks']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return

    # Calculate total score
    total_achieved = sum(parse_percentage(val) for val in df['Achieved'] if pd.notna(val))
    score_class = get_score_class(total_achieved)

    # Display total score with enhanced styling
    st.markdown(f"""
    <div class="total-score-container">
        <div class="total-score {score_class}">
            {total_achieved:.1f}%
        </div>
        <div class="score-label">Overall Performance Score</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Performance Overview
    st.markdown('<h2 class="section-header">📊 KPI Performance Overview</h2>', unsafe_allow_html=True)

    # Initialize session state
    if 'selected_kpi' not in st.session_state:
        st.session_state.selected_kpi = None

    # Create KPI cards in a responsive grid
    cols_per_row = 3
    num_kpis = len(df)

    for i in range(0, num_kpis, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < num_kpis:
                row = df.iloc[i + j]
                kpi_name = row['KPI']
                achieved_value = row['Achieved']
                weightage_value = row['Weightage']
                is_good = determine_kpi_status(kpi_name, achieved_value, weightage_value)
                card_class = "good" if is_good else "critical"

                with col:
                    # Create KPI card
                    if st.button(f"📈 {kpi_name}", key=f"kpi_{i + j}", use_container_width=True):
                        st.session_state.selected_kpi = i + j

                    # Create beautiful circle
                    fig = create_beautiful_circle(kpi_name, achieved_value, weightage_value, is_good)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i + j}")

                    # KPI info below circle
                    status_badge = "status-good" if is_good else "status-critical"
                    status_text = "On Track" if is_good else "Needs Attention"

                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 1rem;">
                        <div style="font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;">
                            {kpi_name}
                        </div>
                        <span class="status-badge {status_badge}">{status_text}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # Display detailed information for selected KPI
    if st.session_state.selected_kpi is not None:
        selected_row = df.iloc[st.session_state.selected_kpi]
        st.markdown("---")
        st.markdown(f'<h2 class="section-header">📋 Detailed Analysis: {selected_row["KPI"]}</h2>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="details-card">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">📊 KPI Metrics</h3>
            """, unsafe_allow_html=True)

            metrics = [
                ("Formula", selected_row.get('Formula', 'N/A')),
                ("Measuring Unit", selected_row.get('Measuring unit', 'N/A')),
                ("Weightage", selected_row.get('Weightage', 'N/A')),
                ("Target", selected_row.get('Target', 'N/A')),
                ("Actual", selected_row.get('Actual', 'N/A')),
                ("Achieved", selected_row.get('Achieved', 'N/A'))
            ]

            for label, value in metrics:
                st.markdown(f"""
                <div class="metric-row">
                    <span class="metric-label" style="color: #FFFFFF">{label}:</span>
                    <span class="metric-value">{value}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="details-card">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">💬 Performance Analysis</h3>
            """, unsafe_allow_html=True)

            # Performance status
            is_good = determine_kpi_status(selected_row['KPI'], selected_row['Achieved'], selected_row['Weightage'])
            status_badge = "status-good" if is_good else "status-critical"
            status_text = "🟢 EXCELLENT PERFORMANCE" if is_good else "🔴 REQUIRES IMMEDIATE ATTENTION"

            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <span class="status-badge {status_badge}">{status_text}</span>
                </div>
                <div style="background: #FFFFFF; color: #000000; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <strong>Remarks:</strong><br>
                    {selected_row.get('Remarks', 'No remarks available')}
                </div>
            """, unsafe_allow_html=True)

        # Close button
        if st.button("❌ Close Detailed View", use_container_width=True):
            st.session_state.selected_kpi = None
            st.rerun()

    # # Analytics Section
    # st.markdown("---")
    # st.markdown('<h2 class="section-header">📈 Performance Analytics</h2>', unsafe_allow_html=True)
    #
    # col1, col2 = st.columns(2)
    #
    # with col1:
    #     # Performance distribution pie chart
    #     good_kpis = sum(1 for i, row in df.iterrows()
    #                     if determine_kpi_status(row['KPI'], row['Achieved'], row['Weightage']))
    #     total_kpis = len(df)
    #
    #     fig_pie = go.Figure(data=[go.Pie(
    #         labels=['On Track', 'Needs Attention'],
    #         values=[good_kpis, total_kpis - good_kpis],
    #         colors=['#00C851', '#FF4444'],
    #         hole=0.6,
    #         textinfo='label+percent',
    #         textfont=dict(size=14, family="Inter"),
    #         marker=dict(line=dict(color='white', width=3))
    #     )])
    #
    #     fig_pie.update_layout(
    #         title=dict(
    #             text="<b>KPI Performance Distribution</b>",
    #             font=dict(size=18, family="Inter", color="white"),
    #             x=0.5
    #         ),
    #         paper_bgcolor='rgba(0,0,0,0)',
    #         plot_bgcolor='rgba(0,0,0,0)',
    #         font=dict(color="white", family="Inter"),
    #         margin=dict(l=20, r=20, t=60, b=20)
    #     )
    #
    #     # Add center text
    #     fig_pie.add_annotation(
    #         text=f"<b>{good_kpis}/{total_kpis}<br>KPIs</b>",
    #         x=0.5, y=0.5,
    #         font=dict(size=20, color="white", family="Inter"),
    #         showarrow=False
    #     )
    #
    #     st.plotly_chart(fig_pie, use_container_width=True)
    #
    # with col2:
    #     # Achievement vs Weightage comparison
    #     kpi_names = [name[:15] + "..." if len(name) > 15 else name for name in df['KPI']]
    #     achieved_values = [parse_percentage(val) for val in df['Achieved']]
    #     weightage_values = [parse_percentage(val) for val in df['Weightage']]
    #
    #     fig_bar = go.Figure()
    #
    #     fig_bar.add_trace(go.Bar(
    #         name='Achieved',
    #         x=kpi_names,
    #         y=achieved_values,
    #         marker=dict(
    #             color='#00C851',
    #             line=dict(color='white', width=1)
    #         ),
    #         text=[f'{val:.1f}%' for val in achieved_values],
    #         textposition='outside'
    #     ))
    #
    #     fig_bar.add_trace(go.Bar(
    #         name='Weightage',
    #         x=kpi_names,
    #         y=weightage_values,
    #         marker=dict(
    #             color='#FF4444',
    #             line=dict(color='white', width=1)
    #         ),
    #         text=[f'{val:.1f}%' for val in weightage_values],
    #         textposition='outside'
    #     ))
    #
    #     fig_bar.update_layout(
    #         title=dict(
    #             text="<b>Achievement vs Target Weightage</b>",
    #             font=dict(size=18, family="Inter", color="white"),
    #             x=0.5
    #         ),
    #         xaxis=dict(
    #             title="KPIs",
    #             titlefont=dict(color="white", family="Inter", size=14),
    #             tickfont=dict(color="white", family="Inter", size=10),
    #             tickangle=-45
    #         ),
    #         yaxis=dict(
    #             title="Percentage (%)",
    #             titlefont=dict(color="white", family="Inter", size=14),
    #             tickfont=dict(color="white", family="Inter", size=12)
    #         ),
    #         barmode='group',
    #         paper_bgcolor='rgba(0,0,0,0)',
    #         plot_bgcolor='rgba(0,0,0,0)',
    #         font=dict(color="white", family="Inter"),
    #         legend=dict(
    #             font=dict(color="white", family="Inter"),
    #             bgcolor='rgba(255,255,255,0.1)'
    #         ),
    #         margin=dict(l=20, r=20, t=60, b=100)
    #     )
    #
    #     st.plotly_chart(fig_bar, use_container_width=True)
    #
    #     # Performance trend analysis
    # st.markdown("---")
    # st.markdown('<h2 class="section-header">🎯 Quick Performance Summary</h2>', unsafe_allow_html=True)
    #
    # # Create summary metrics
    # col1, col2, col3, col4 = st.columns(4)
    #
    # with col1:
    #     st.markdown(f"""
    #                 <div class="analytics-container" style="text-align: center;">
    #                     <h3 style="color: white; margin-bottom: 0.5rem;">Total KPIs</h3>
    #                     <div style="font-size: 2.5rem; font-weight: bold; color: #FFD700;">
    #                         {len(df)}
    #                     </div>
    #                 </div>
    #                 """, unsafe_allow_html=True)
    #
    # with col2:
    #     good_count = sum(1 for i, row in df.iterrows()
    #                      if determine_kpi_status(row['KPI'], row['Achieved'], row['Weightage']))
    #     st.markdown(f"""
    #                 <div class="analytics-container" style="text-align: center;">
    #                     <h3 style="color: white; margin-bottom: 0.5rem;">On Track</h3>
    #                     <div style="font-size: 2.5rem; font-weight: bold; color: #00C851;">
    #                         {good_count}
    #                     </div>
    #                 </div>
    #                 """, unsafe_allow_html=True)
    #
    # with col3:
    #     critical_count = len(df) - good_count
    #     st.markdown(f"""
    #                 <div class="analytics-container" style="text-align: center;">
    #                     <h3 style="color: white; margin-bottom: 0.5rem;">Critical</h3>
    #                     <div style="font-size: 2.5rem; font-weight: bold; color: #FF4444;">
    #                         {critical_count}
    #                     </div>
    #                 </div>
    #                 """, unsafe_allow_html=True)
    #
    # with col4:
    #     success_rate = (good_count / len(df)) * 100 if len(df) > 0 else 0
    #     st.markdown(f"""
    #                 <div class="analytics-container" style="text-align: center;">
    #                     <h3 style="color: white; margin-bottom: 0.5rem;">Success Rate</h3>
    #                     <div style="font-size: 2.5rem; font-weight: bold; color: #2BBBAD;">
    #                         {success_rate:.1f}%
    #                     </div>
    #                 </div>
    #                 """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
                <div style="text-align: center; color: rgba(255,255,255,0.7); padding: 2rem;">
                    <p>🎯 KPI Dashboard - Real-time Performance Monitoring</p>
                    <p style="font-size: 0.9rem;">Upload your CSV file to see live KPI data | Built with Streamlit & Plotly</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
