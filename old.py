import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="Portfolio Construction & Risk Attribution Engine", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Theme toggle in header
col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    st.title("Portfolio Construction & Risk Attribution Engine")
with col3:
    if st.button("Toggle Theme"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

# Apply theme-specific CSS
if st.session_state.theme == 'light':
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #F5F7FA;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #F5F7FA;
        border-radius: 5px;
        color: #4a5568;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    div[data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #1a1d23;
        color: #e4e7eb;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background-color: #252932;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border-left: 4px solid #667eea;
        color: #e4e7eb;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #252932;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1a1d23;
        border-radius: 5px;
        color: #9ca3af;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    div[data-testid="stDataFrame"] {
        background-color: #252932;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for data
if 'stocks_df' not in st.session_state:
    st.session_state.stocks_df = pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT', 'AMGN', 'JPM', 'XOM'],
        'Sector': ['Technology', 'Technology', 'Healthcare', 'Financials', 'Energy'],
        'Expected Return (%)': [12.0, 10.0, 9.0, 11.0, 8.0],
        'Volatility (%)': [25.0, 20.0, 18.0, 22.0, 28.0],
        'Current Weight (%)': [20.0, 20.0, 20.0, 20.0, 20.0]
    })

# Correlation matrix helper
def get_correlation_matrix(tickers):
    n = len(tickers)
    corr_matrix = np.eye(n)
    
    # Default correlations based on sectors
    for i in range(n):
        for j in range(i+1, n):
            if st.session_state.stocks_df.loc[i, 'Sector'] == st.session_state.stocks_df.loc[j, 'Sector']:
                corr_matrix[i, j] = corr_matrix[j, i] = 0.70
            else:
                corr_matrix[i, j] = corr_matrix[j, i] = 0.35
    
    return corr_matrix

# Portfolio optimization function
def optimize_portfolio(returns, volatilities, corr_matrix, constraints_dict):
    n = len(returns)
    
    # Convert to numpy arrays
    returns = np.array(returns) / 100
    volatilities = np.array(volatilities) / 100
    
    # Covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    # Objective: Maximize Sharpe Ratio (minimize negative Sharpe)
    def negative_sharpe(weights):
        portfolio_return = np.sum(returns * weights)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - constraints_dict['risk_free_rate']/100) / portfolio_vol
        return -sharpe
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
    ]
    
    # Bounds: each weight between 0 and max_position
    bounds = tuple((0, constraints_dict['max_position']/100) for _ in range(n))
    
    # Initial guess: equal weights
    x0 = np.ones(n) / n
    
    # Optimize
    result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x * 100  # Return as percentages

# Calculate portfolio metrics
def calculate_portfolio_metrics(weights, returns, volatilities, corr_matrix, risk_free_rate):
    weights = np.array(weights) / 100
    returns = np.array(returns) / 100
    volatilities = np.array(volatilities) / 100
    
    cov_matrix = np.outer(volatilities, volatilities) * corr_matrix
    
    portfolio_return = np.sum(returns * weights)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate/100) / portfolio_vol
    
    # Marginal contribution to risk
    mctr = np.dot(cov_matrix, weights) / portfolio_vol
    
    # Risk contribution
    risk_contrib = weights * mctr
    
    return {
        'return': portfolio_return * 100,
        'volatility': portfolio_vol * 100,
        'sharpe': sharpe_ratio,
        'mctr': mctr * 100,
        'risk_contrib': risk_contrib * 100
    }

# Bottom navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Portfolio Setup", "Optimization", "Risk Attribution", "Stress Testing", "Settings"])

with tab1:
    st.header("Portfolio Setup")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Stock Candidates")
        
        # Editable dataframe
        edited_df = st.data_editor(
            st.session_state.stocks_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Expected Return (%)": st.column_config.NumberColumn(
                    "Expected Return (%)",
                    min_value=0,
                    max_value=100,
                    step=0.5,
                    format="%.1f%%"
                ),
                "Volatility (%)": st.column_config.NumberColumn(
                    "Volatility (%)",
                    min_value=0,
                    max_value=100,
                    step=0.5,
                    format="%.1f%%"
                ),
                "Current Weight (%)": st.column_config.NumberColumn(
                    "Current Weight (%)",
                    min_value=0,
                    max_value=100,
                    step=1.0,
                    format="%.1f%%"
                )
            }
        )
        
        st.session_state.stocks_df = edited_df
    
    with col2:
        st.subheader("Current Allocation")
        
        if not st.session_state.stocks_df.empty:
            fig = go.Figure(data=[go.Pie(
                labels=st.session_state.stocks_df['Ticker'],
                values=st.session_state.stocks_df['Current Weight (%)'],
                hole=0.4
            )])
            
            fig.update_layout(
                height=300,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#e4e7eb' if st.session_state.theme == 'dark' else '#1a1d23')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats
        st.metric("Total Positions", len(st.session_state.stocks_df))
        st.metric("Sectors", st.session_state.stocks_df['Sector'].nunique())

with tab2:
    st.header("Portfolio Optimization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_position = st.number_input("Max Position Size (%)", min_value=1, max_value=100, value=30, step=1)
    
    with col2:
        max_sector = st.number_input("Max Sector Exposure (%)", min_value=1, max_value=100, value=40, step=1)
    
    with col3:
        target_vol = st.number_input("Target Volatility (%)", min_value=1, max_value=50, value=18, step=1)
    
    with col4:
        risk_free = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
    
    st.markdown("---")
    
    if st.button("Run Optimization", type="primary", use_container_width=True):
        if len(st.session_state.stocks_df) > 0:
            constraints_dict = {
                'max_position': max_position,
                'max_sector': max_sector,
                'target_volatility': target_vol,
                'risk_free_rate': risk_free
            }
            
            corr_matrix = get_correlation_matrix(st.session_state.stocks_df['Ticker'].tolist())
            
            optimized_weights = optimize_portfolio(
                st.session_state.stocks_df['Expected Return (%)'].tolist(),
                st.session_state.stocks_df['Volatility (%)'].tolist(),
                corr_matrix,
                constraints_dict
            )
            
            st.session_state.optimized_weights = optimized_weights
            st.session_state.corr_matrix = corr_matrix
            st.session_state.constraints = constraints_dict
            
            st.success("Optimization Complete")
    
    if 'optimized_weights' in st.session_state:
        st.subheader("Optimization Results")
        
        # Create comparison dataframe
        comparison_df = st.session_state.stocks_df.copy()
        comparison_df['Optimized Weight (%)'] = st.session_state.optimized_weights
        comparison_df['Weight Change (%)'] = comparison_df['Optimized Weight (%)'] - comparison_df['Current Weight (%)']
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.dataframe(
                comparison_df[['Ticker', 'Sector', 'Current Weight (%)', 'Optimized Weight (%)', 'Weight Change (%)']],
                use_container_width=True
            )
        
        with col2:
            # Calculate metrics for both portfolios
            current_metrics = calculate_portfolio_metrics(
                comparison_df['Current Weight (%)'].tolist(),
                comparison_df['Expected Return (%)'].tolist(),
                comparison_df['Volatility (%)'].tolist(),
                st.session_state.corr_matrix,
                st.session_state.constraints['risk_free_rate']
            )
            
            optimized_metrics = calculate_portfolio_metrics(
                comparison_df['Optimized Weight (%)'].tolist(),
                comparison_df['Expected Return (%)'].tolist(),
                comparison_df['Volatility (%)'].tolist(),
                st.session_state.corr_matrix,
                st.session_state.constraints['risk_free_rate']
            )
            
            metrics_comparison = pd.DataFrame({
                'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
                'Current': [f"{current_metrics['return']:.2f}%", f"{current_metrics['volatility']:.2f}%", f"{current_metrics['sharpe']:.3f}"],
                'Optimized': [f"{optimized_metrics['return']:.2f}%", f"{optimized_metrics['volatility']:.2f}%", f"{optimized_metrics['sharpe']:.3f}"]
            })
            
            st.dataframe(metrics_comparison, use_container_width=True, hide_index=True)
        
        # Visualization
        st.subheader("Weight Comparison")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=comparison_df['Ticker'],
            y=comparison_df['Current Weight (%)'],
            marker_color='#94a3b8'
        ))
        
        fig.add_trace(go.Bar(
            name='Optimized',
            x=comparison_df['Ticker'],
            y=comparison_df['Optimized Weight (%)'],
            marker_color='#667eea'
        ))
        
        fig.update_layout(
            barmode='group',
            height=400,
            xaxis_title='Stock',
            yaxis_title='Weight (%)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#e4e7eb' if st.session_state.theme == 'dark' else '#1a1d23'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Risk Attribution")
    
    if 'optimized_weights' in st.session_state:
        comparison_df = st.session_state.stocks_df.copy()
        comparison_df['Optimized Weight (%)'] = st.session_state.optimized_weights
        
        optimized_metrics = calculate_portfolio_metrics(
            comparison_df['Optimized Weight (%)'].tolist(),
            comparison_df['Expected Return (%)'].tolist(),
            comparison_df['Volatility (%)'].tolist(),
            st.session_state.corr_matrix,
            st.session_state.constraints['risk_free_rate']
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Portfolio Return", f"{optimized_metrics['return']:.2f}%")
        
        with col2:
            st.metric("Portfolio Volatility", f"{optimized_metrics['volatility']:.2f}%")
        
        with col3:
            st.metric("Sharpe Ratio", f"{optimized_metrics['sharpe']:.3f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Contribution by Stock")
            
            risk_df = pd.DataFrame({
                'Stock': comparison_df['Ticker'],
                'Risk Contribution (%)': optimized_metrics['risk_contrib']
            })
            
            fig = go.Figure(go.Bar(
                x=risk_df['Stock'],
                y=risk_df['Risk Contribution (%)'],
                marker_color='#ef4444',
                text=risk_df['Risk Contribution (%)'].round(2),
                textposition='auto'
            ))
            
            fig.update_layout(
                height=350,
                xaxis_title='Stock',
                yaxis_title='Risk Contribution (%)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#e4e7eb' if st.session_state.theme == 'dark' else '#1a1d23')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sector Exposure")
            
            sector_exposure = comparison_df.groupby('Sector')['Optimized Weight (%)'].sum().reset_index()
            
            fig = go.Figure(data=[go.Pie(
                labels=sector_exposure['Sector'],
                values=sector_exposure['Optimized Weight (%)'],
                hole=0.4
            )])
            
            fig.update_layout(
                height=350,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#e4e7eb' if st.session_state.theme == 'dark' else '#1a1d23')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Risk Analysis")
        
        risk_detail_df = pd.DataFrame({
            'Stock': comparison_df['Ticker'],
            'Weight (%)': comparison_df['Optimized Weight (%)'],
            'Volatility (%)': comparison_df['Volatility (%)'],
            'MCTR (%)': optimized_metrics['mctr'],
            'Risk Contribution (%)': optimized_metrics['risk_contrib']
        })
        
        st.dataframe(risk_detail_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("Run optimization first to see risk attribution analysis")

with tab4:
    st.header("Stress Testing")
    
    if 'optimized_weights' in st.session_state:
        st.subheader("Define Scenarios")
        
        scenario_name = st.text_input("Scenario Name", value="Tech Selloff")
        
        st.write("Define sector shocks (%):")
        
        sectors = st.session_state.stocks_df['Sector'].unique()
        sector_shocks = {}
        
        cols = st.columns(len(sectors))
        for i, sector in enumerate(sectors):
            with cols[i]:
                sector_shocks[sector] = st.number_input(
                    sector,
                    min_value=-50,
                    max_value=50,
                    value=-10 if sector == 'Technology' else 0,
                    step=1
                )
        
        if st.button("Run Stress Test", use_container_width=True):
            comparison_df = st.session_state.stocks_df.copy()
            comparison_df['Optimized Weight (%)'] = st.session_state.optimized_weights
            
            # Apply shocks
            comparison_df['Shock (%)'] = comparison_df['Sector'].map(sector_shocks)
            comparison_df['Shocked Return (%)'] = comparison_df['Expected Return (%)'] + comparison_df['Shock (%)']
            
            # Calculate portfolio impact
            portfolio_impact = np.sum(comparison_df['Optimized Weight (%)'].values * comparison_df['Shock (%)'].values) / 100
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    "Portfolio Impact",
                    f"{portfolio_impact:.2f}%",
                    delta=f"{portfolio_impact:.2f}%",
                    delta_color="inverse"
                )
                
                st.dataframe(
                    comparison_df[['Ticker', 'Sector', 'Optimized Weight (%)', 'Shock (%)', 'Shocked Return (%)']],
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.subheader("Impact by Stock")
                
                comparison_df['Stock Impact (%)'] = comparison_df['Optimized Weight (%)'] * comparison_df['Shock (%)'] / 100
                
                fig = go.Figure()
                
                colors = ['#ef4444' if x < 0 else '#10b981' for x in comparison_df['Stock Impact (%)']]
                
                fig.add_trace(go.Bar(
                    x=comparison_df['Ticker'],
                    y=comparison_df['Stock Impact (%)'],
                    marker_color=colors,
                    text=comparison_df['Stock Impact (%)'].round(2),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title='Stock',
                    yaxis_title='Impact on Portfolio (%)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color='#e4e7eb' if st.session_state.theme == 'dark' else '#1a1d23')
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Run optimization first to perform stress testing")

with tab5:
    st.header("Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Display Preferences")
        
        st.write(f"Current Theme: **{st.session_state.theme.title()}**")
        st.write("Use the 'Toggle Theme' button in the header to switch themes")
        
        st.markdown("---")
        
        st.subheader("Correlation Matrix")
        
        if len(st.session_state.stocks_df) > 0:
            corr_matrix = get_correlation_matrix(st.session_state.stocks_df['Ticker'].tolist())
            corr_df = pd.DataFrame(
                corr_matrix,
                index=st.session_state.stocks_df['Ticker'],
                columns=st.session_state.stocks_df['Ticker']
            )
            
            st.dataframe(corr_df.style.format("{:.2f}"), use_container_width=True)
    
    with col2:
        st.subheader("Export Data")
        
        if 'optimized_weights' in st.session_state:
            comparison_df = st.session_state.stocks_df.copy()
            comparison_df['Optimized Weight (%)'] = st.session_state.optimized_weights
            
            csv = comparison_df.to_csv(index=False)
            
            st.download_button(
                label="Download Portfolio Data",
                data=csv,
                file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        st.subheader("About")
        st.write("Portfolio Construction & Risk Attribution Engine")
        st.write("Version 1.0")
        st.write("Optimizes portfolio weights using mean-variance optimization and provides comprehensive risk attribution analysis")