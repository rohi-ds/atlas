"""
INSTITUTIONAL PORTFOLIO CONSTRUCTION & REBALANCING PLATFORM
============================================================
Professional-grade portfolio optimization and management system
Built for asset managers, advisors, and institutional investors

Requirements:
pip install streamlit pandas numpy scipy yfinance plotly scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION & CONSTANTS
# ===========================

st.set_page_config(
    page_title="atlas.",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Bloomberg-inspired dark theme
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Roboto Mono', 'Courier New', monospace !important;
    }
    
    .stApp {
        background-color: #0a0e27;
        color: #e8e8e8;
        font-family: 'Roboto Mono', 'Courier New', monospace;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #1a1f3a 0%, #0f1229 100%);
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #00d4ff;
        margin: 10px 0;
        font-family: 'Roboto Mono', 'Courier New', monospace;
    }
    
    .terminal-header {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        padding: 15px;
        border-radius: 6px;
        font-family: 'Roboto Mono', 'Courier New', monospace;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 20px;
        text-align: center;
        color: #0a0e27;
    }
    
    .section-header {
        background: #1a1f3a;
        padding: 12px;
        border-left: 3px solid #00d4ff;
        margin: 15px 0;
        font-weight: 600;
        font-size: 16px;
        font-family: 'Roboto Mono', 'Courier New', monospace;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00d4ff;
        font-family: 'Roboto Mono', 'Courier New', monospace;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
        font-size: 14px;
        font-family: 'Roboto Mono', 'Courier New', monospace;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f1229;
        padding: 10px;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f3a;
        color: #9ca3af;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 600;
        font-family: 'Roboto Mono', 'Courier New', monospace;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff;
        color: #0a0e27;
    }
    
    .dataframe {
        font-family: 'Roboto Mono', 'Courier New', monospace;
        font-size: 12px;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto Mono', 'Courier New', monospace !important;
    }
    
    .stButton button {
        font-family: 'Roboto Mono', 'Courier New', monospace;
        font-weight: 600;
    }
    
    .stSelectbox, .stMultiSelect, .stTextInput, .stTextArea, .stNumberInput {
        font-family: 'Roboto Mono', 'Courier New', monospace;
    }
    
    label {
        font-family: 'Roboto Mono', 'Courier New', monospace !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Asset Universe - Real, Tradable Securities
ASSET_UNIVERSE = {
    'US_LARGE_CAP': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'JPM', 'V', 'JNJ', 'PG'],
    'US_MID_CAP': ['MDY', 'IJH', 'VO'],
    'US_SMALL_CAP': ['IWM', 'VB', 'IJR'],
    'INTL_DEVELOPED': ['EFA', 'VEA', 'IEFA'],
    'EMERGING_MARKETS': ['EEM', 'VWO', 'IEMG'],
    'US_BONDS_LONG': ['TLT', 'VGLT', 'EDV'],
    'US_BONDS_INTERMEDIATE': ['IEF', 'VGIT', 'IEI'],
    'US_BONDS_SHORT': ['SHY', 'VGSH', 'SCHO'],
    'CORPORATE_BONDS': ['LQD', 'VCIT', 'VCLT'],
    'HIGH_YIELD': ['HYG', 'JNK', 'SJNK'],
    'TIPS': ['TIP', 'VTIP', 'SCHP'],
    'AGGREGATE_BONDS': ['AGG', 'BND', 'IUSB'],
    'REITS': ['VNQ', 'IYR', 'SCHH'],
    'COMMODITIES': ['DBC', 'GSG', 'PDBC'],
    'GOLD': ['GLD', 'IAU', 'SGOL'],
    'ALTERNATIVES': ['ABRYX', 'QLEIX', 'QSPIX']  # Liquid alt proxies
}

# Flatten for easy selection
ALL_TICKERS = [ticker for category in ASSET_UNIVERSE.values() for ticker in category]

# Risk-free rate proxy (use 3-month T-Bill)
RISK_FREE_RATE = 0.045  # 4.5% annual, adjust based on current market

# ===========================
# DATA INGESTION MODULE
# ===========================

@st.cache_data(ttl=3600)
def fetch_market_data(tickers, start_date, end_date):
    """
    Fetch historical price data from Yahoo Finance
    Returns: DataFrame with adjusted close prices (or close if adj close unavailable)
    """
    try:
        # Download all data
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if raw_data.empty:
            st.error("No data returned. Please check ticker symbols and date range.")
            return pd.DataFrame()
        
        # Handle single vs multiple tickers
        if len(tickers) == 1:
            # Single ticker - data is not MultiIndex
            if 'Adj Close' in raw_data.columns:
                data = raw_data[['Adj Close']].copy()
            elif 'Close' in raw_data.columns:
                data = raw_data[['Close']].copy()
            else:
                st.error(f"No price data found for {tickers[0]}")
                return pd.DataFrame()
            data.columns = [tickers[0]]
        else:
            # Multiple tickers - data has MultiIndex columns
            if isinstance(raw_data.columns, pd.MultiIndex):
                # Try Adj Close first
                if 'Adj Close' in raw_data.columns.get_level_values(0):
                    data = raw_data['Adj Close'].copy()
                elif 'Close' in raw_data.columns.get_level_values(0):
                    data = raw_data['Close'].copy()
                else:
                    st.error("No price data found in downloaded data")
                    return pd.DataFrame()
            else:
                # Fallback for unexpected structure
                st.error("Unexpected data structure from yfinance")
                return pd.DataFrame()
        
        # Drop rows with any NaN values to ensure clean data
        initial_rows = len(data)
        data = data.dropna()
        dropped_rows = initial_rows - len(data)
        
        if dropped_rows > 0:
            st.info(f"Dropped {dropped_rows} rows with missing data")
        
        if data.empty:
            st.error("All data contained missing values after cleaning")
            return pd.DataFrame()
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def calculate_returns(price_data):
    """Calculate daily log returns"""
    return np.log(price_data / price_data.shift(1)).dropna()

def calculate_statistics(returns, annualization_factor=252):
    """
    Calculate key portfolio statistics
    annualization_factor: 252 for daily data, 12 for monthly
    """
    mean_returns = returns.mean() * annualization_factor
    cov_matrix = returns.cov() * annualization_factor
    corr_matrix = returns.corr()
    volatility = returns.std() * np.sqrt(annualization_factor)
    
    return {
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'corr_matrix': corr_matrix,
        'volatility': volatility
    }

# ===========================
# PORTFOLIO OPTIMIZATION ENGINE
# ===========================

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate portfolio return, volatility, and Sharpe ratio
    """
    returns = np.sum(mean_returns * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (returns - risk_free_rate) / volatility if volatility > 0 else 0
    return returns, volatility, sharpe

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    """Objective function for maximizing Sharpe ratio"""
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def portfolio_volatility(weights, cov_matrix):
    """Calculate portfolio volatility"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def optimize_portfolio(mean_returns, cov_matrix, objective='max_sharpe', 
                       constraints=None, risk_free_rate=RISK_FREE_RATE):
    """
    Core optimization function supporting multiple objectives
    
    Objectives:
    - max_sharpe: Maximum Sharpe ratio
    - min_vol: Minimum volatility
    - risk_parity: Equal risk contribution
    - target_return: Achieve specific return with minimum risk
    - target_risk: Achieve specific risk with maximum return
    """
    n_assets = len(mean_returns)
    
    # Default constraints
    if constraints is None:
        constraints = {
            'min_weight': 0.0,
            'max_weight': 0.5,
            'max_asset_class_weight': 1.0
        }
    
    # Bounds for each asset
    bounds = tuple((constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets))
    
    # Constraint: weights sum to 1
    constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Initial guess (equal weight)
    init_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize based on objective
    if objective == 'max_sharpe':
        result = minimize(
            negative_sharpe,
            init_weights,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
    
    elif objective == 'min_vol':
        result = minimize(
            portfolio_volatility,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
    
    elif objective == 'risk_parity':
        # Risk parity: equal risk contribution from each asset
        def risk_parity_objective(weights, cov_matrix):
            portfolio_vol = portfolio_volatility(weights, cov_matrix)
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return np.sum((risk_contrib - risk_contrib.mean()) ** 2)
        
        result = minimize(
            risk_parity_objective,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
    
    elif objective == 'balanced':
        # 60/40-style but optimized: target moderate risk with good return
        target_vol = 0.10  # 10% annual volatility
        
        def balanced_objective(weights, mean_returns, cov_matrix, target_vol):
            ret, vol, _ = portfolio_performance(weights, mean_returns, cov_matrix)
            # Minimize deviation from target volatility while maximizing return
            return -(ret / 0.10) + 100 * (vol - target_vol) ** 2
        
        result = minimize(
            balanced_objective,
            init_weights,
            args=(mean_returns, cov_matrix, target_vol),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
    
    elif objective == 'defensive':
        # Minimize volatility with constraint on minimum return
        min_return = 0.04  # 4% minimum return
        constraints_list.append({'type': 'ineq', 'fun': lambda x: np.sum(mean_returns * x) - min_return})
        
        result = minimize(
            portfolio_volatility,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
    
    elif objective == 'growth':
        # Maximum return with volatility constraint
        max_vol = 0.18  # 18% max volatility
        constraints_list.append({'type': 'ineq', 'fun': lambda x: max_vol - portfolio_volatility(x, cov_matrix)})
        
        result = minimize(
            lambda x: -np.sum(mean_returns * x),
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
    
    elif objective == 'income':
        # Maximize return with moderate risk constraint
        max_vol = 0.12  # 12% max volatility
        constraints_list.append({'type': 'ineq', 'fun': lambda x: max_vol - portfolio_volatility(x, cov_matrix)})
        
        result = minimize(
            lambda x: -np.sum(mean_returns * x),
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
    
    return result.x if result.success else init_weights

def generate_efficient_frontier(mean_returns, cov_matrix, n_portfolios=50):
    """
    Generate efficient frontier data
    Returns portfolios along the frontier
    """
    min_vol_weights = optimize_portfolio(mean_returns, cov_matrix, objective='min_vol')
    _, min_vol, _ = portfolio_performance(min_vol_weights, mean_returns, cov_matrix)
    
    max_ret_weights = optimize_portfolio(mean_returns, cov_matrix, objective='growth')
    max_ret, _, _ = portfolio_performance(max_ret_weights, mean_returns, cov_matrix)
    
    target_returns = np.linspace(min_vol * 0.8, max_ret, n_portfolios)
    
    frontier_portfolios = []
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target}
        ]
        
        n_assets = len(mean_returns)
        bounds = tuple((0, 0.5) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_volatility,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            ret, vol, sharpe = portfolio_performance(result.x, mean_returns, cov_matrix)
            frontier_portfolios.append({'return': ret, 'volatility': vol, 'sharpe': sharpe})
    
    return pd.DataFrame(frontier_portfolios)

def calculate_risk_contribution(weights, cov_matrix):
    """
    Calculate marginal and absolute risk contribution by asset
    """
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
    risk_contrib = weights * marginal_contrib
    pct_contrib = risk_contrib / risk_contrib.sum()
    
    return pd.DataFrame({
        'Marginal Risk': marginal_contrib,
        'Risk Contribution': risk_contrib,
        'Risk Contribution %': pct_contrib * 100
    })

# ===========================
# REBALANCING ENGINE
# ===========================

def calculate_rebalancing_trades(current_weights, target_weights, portfolio_value, 
                                 transaction_cost_bps=10):
    """
    Calculate trades needed to rebalance from current to target weights
    
    Parameters:
    - current_weights: dict or Series of current allocations
    - target_weights: dict or Series of target allocations
    - portfolio_value: total portfolio value
    - transaction_cost_bps: transaction cost in basis points
    
    Returns: DataFrame with trade details
    """
    if isinstance(current_weights, dict):
        current_weights = pd.Series(current_weights)
    if isinstance(target_weights, dict):
        target_weights = pd.Series(target_weights)
    
    # Align indices
    all_assets = current_weights.index.union(target_weights.index)
    current_weights = current_weights.reindex(all_assets, fill_value=0)
    target_weights = target_weights.reindex(all_assets, fill_value=0)
    
    # Calculate changes
    weight_change = target_weights - current_weights
    current_value = current_weights * portfolio_value
    target_value = target_weights * portfolio_value
    trade_value = target_value - current_value
    
    # Transaction costs
    transaction_costs = np.abs(trade_value) * (transaction_cost_bps / 10000)
    
    trades_df = pd.DataFrame({
        'Current Weight %': current_weights * 100,
        'Target Weight %': target_weights * 100,
        'Weight Change %': weight_change * 100,
        'Current Value': current_value,
        'Target Value': target_value,
        'Trade Value': trade_value,
        'Action': ['BUY' if tv > 0 else 'SELL' if tv < 0 else 'HOLD' for tv in trade_value],
        'Transaction Cost': transaction_costs
    })
    
    # Calculate turnover
    turnover = np.sum(np.abs(weight_change)) / 2
    total_transaction_cost = transaction_costs.sum()
    
    return trades_df, turnover, total_transaction_cost

# ===========================
# VISUALIZATION FUNCTIONS
# ===========================

def plot_efficient_frontier(frontier_df, current_portfolio=None):
    """Create interactive efficient frontier plot"""
    fig = go.Figure()
    
    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=frontier_df['volatility'] * 100,
        y=frontier_df['return'] * 100,
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=6, color='#00d4ff'),
        hovertemplate='<b>Volatility:</b> %{x:.2f}%<br><b>Return:</b> %{y:.2f}%<br><b>Sharpe:</b> %{customdata:.2f}<extra></extra>',
        customdata=frontier_df['sharpe']
    ))
    
    # Current portfolio if provided
    if current_portfolio:
        fig.add_trace(go.Scatter(
            x=[current_portfolio['volatility'] * 100],
            y=[current_portfolio['return'] * 100],
            mode='markers',
            name='Current Portfolio',
            marker=dict(size=15, color='#ff6b35', symbol='star'),
            hovertemplate='<b>Current Portfolio</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Efficient Frontier Analysis',
        xaxis_title='Volatility (Annual %)',
        yaxis_title='Expected Return (Annual %)',
        template='plotly_dark',
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#0f1229',
        font=dict(family='Courier New', size=12, color='#e8e8e8'),
        hovermode='closest',
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

def plot_correlation_matrix(corr_matrix):
    """Create correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title='Asset Correlation Matrix',
        template='plotly_dark',
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#0f1229',
        font=dict(family='Courier New', size=10, color='#e8e8e8'),
        height=600
    )
    
    return fig

def plot_portfolio_allocation(weights, tickers):
    """Create portfolio allocation pie chart"""
    weights_pct = weights * 100
    
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weights_pct,
        hole=0.4,
        marker=dict(colors=px.colors.sequential.Turbo),
        textinfo='label+percent',
        textfont=dict(size=12, color='#0a0e27')
    )])
    
    fig.update_layout(
        title='Portfolio Allocation',
        template='plotly_dark',
        paper_bgcolor='#0a0e27',
        font=dict(family='Courier New', size=12, color='#e8e8e8'),
        showlegend=False
    )
    
    return fig

def plot_risk_contribution(risk_contrib_df, tickers):
    """Plot risk contribution by asset"""
    fig = go.Figure(data=[
        go.Bar(
            x=tickers,
            y=risk_contrib_df['Risk Contribution %'].values,
            marker=dict(color='#00d4ff'),
            text=np.round(risk_contrib_df['Risk Contribution %'].values, 1),
            texttemplate='%{text}%',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Risk Contribution by Asset',
        xaxis_title='Asset',
        yaxis_title='Risk Contribution (%)',
        template='plotly_dark',
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#0f1229',
        font=dict(family='Courier New', size=12, color='#e8e8e8')
    )
    
    return fig

def plot_historical_performance(price_data, weights, tickers):
    """Plot historical portfolio performance"""
    returns = calculate_returns(price_data)
    
    # Ensure weights and returns are aligned
    weights_series = pd.Series(weights, index=tickers)
    
    # Only use tickers that exist in both returns and weights
    common_tickers = returns.columns.intersection(weights_series.index)
    
    if len(common_tickers) == 0:
        st.error("No common tickers between price data and portfolio weights")
        return go.Figure()
    
    # Align data
    aligned_returns = returns[common_tickers]
    aligned_weights = weights_series[common_tickers]
    
    # Normalize weights to sum to 1
    aligned_weights = aligned_weights / aligned_weights.sum()
    
    # Calculate portfolio returns
    portfolio_returns = (aligned_returns * aligned_weights.values).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=(cumulative_returns.values - 1) * 100,
        mode='lines',
        name='Portfolio',
        line=dict(color='#00d4ff', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    fig.update_layout(
        title='Historical Portfolio Performance (Cumulative Returns)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#0f1229',
        font=dict(family='Roboto Mono', size=12, color='#e8e8e8'),
        hovermode='x unified'
    )
    
    return fig

# ===========================
# MAIN APPLICATION
# ===========================

def main():
    # Header
    st.markdown('<div class="terminal-header">ATLAS PORTFOLIO CONSTRUCTION PLATFORM</div>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'optimized_portfolios' not in st.session_state:
        st.session_state.optimized_portfolios = {}
    if 'current_portfolio' not in st.session_state:
        st.session_state.current_portfolio = None
    if 'selected_tickers' not in st.session_state:
        # Initialize with default tickers
        default_tickers = []
        for asset_class in ['US_LARGE_CAP', 'US_BONDS_INTERMEDIATE', 'GOLD']:
            default_tickers.extend(ASSET_UNIVERSE[asset_class])
        st.session_state.selected_tickers = list(set(default_tickers))
    
    # Main navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "PORTFOLIO CONSTRUCTION",
        "OPTIMIZATION ENGINE",
        "REBALANCING MODULE",
        "ANALYTICS & REPORTING"
    ])
    
    # ===========================
    # TAB 1: PORTFOLIO CONSTRUCTION
    # ===========================
    with tab1:
        st.markdown('<div class="section-header">ASSET UNIVERSE SELECTION</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Select Assets")
            
            # Asset class filters
            asset_classes = st.multiselect(
                "Asset Classes",
                list(ASSET_UNIVERSE.keys()),
                default=['US_LARGE_CAP', 'US_BONDS_INTERMEDIATE', 'GOLD'],
                key="asset_classes_select"
            )
            
            # Build selected ticker list from asset classes
            class_based_tickers = []
            for asset_class in asset_classes:
                class_based_tickers.extend(ASSET_UNIVERSE[asset_class])
            
            # Manual ticker selection from predefined universe
            manual_tickers = st.multiselect(
                "Additional Tickers (From Universe)",
                [t for t in ALL_TICKERS if t not in class_based_tickers],
                default=[],
                key="manual_tickers_select"
            )
            
            # Combine and update session state
            combined_tickers = list(set(class_based_tickers + manual_tickers))
            
            # Merge with any custom tickers already in session state
            if 'custom_tickers' not in st.session_state:
                st.session_state.custom_tickers = []
            
            all_tickers = list(set(combined_tickers + st.session_state.custom_tickers))
            st.session_state.selected_tickers = all_tickers
            
            st.info(f"Total Assets Selected: {len(st.session_state.selected_tickers)}")
            
        with col2:
            st.markdown("### Data Parameters")
            
            lookback_period = st.selectbox(
                "Historical Lookback",
                ["1 Year", "3 Years", "5 Years", "10 Years"],
                index=2
            )
            
            lookback_map = {
                "1 Year": 365,
                "3 Years": 365 * 3,
                "5 Years": 365 * 5,
                "10 Years": 365 * 10
            }
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_map[lookback_period])
            
            st.markdown(f"**Start:** {start_date.strftime('%Y-%m-%d')}")
            st.markdown(f"**End:** {end_date.strftime('%Y-%m-%d')}")
            
            fetch_data_btn = st.button("FETCH MARKET DATA", type="primary", use_container_width=True)
        
        # Custom ticker management section
        st.markdown('<div class="section-header">CUSTOM TICKER MANAGEMENT</div>', unsafe_allow_html=True)
        
        col_custom1, col_custom2 = st.columns([3, 1])
        
        with col_custom1:
            st.markdown("### Add or Remove Specific Tickers")
            
            # Display current selection
            if st.session_state.selected_tickers:
                st.markdown("**Current Universe:**")
                ticker_display = ", ".join(sorted(st.session_state.selected_tickers))
                st.text_area("Selected Tickers", ticker_display, height=100, disabled=True, key="current_tickers_display")
            
            # Add custom tickers
            custom_ticker_input = st.text_input(
                "Add Custom Tickers (comma-separated, e.g., TSLA, BRK.B, ^GSPC)",
                placeholder="TSLA, BRK.B, QQQ",
                key="custom_ticker_input"
            )
            
            col_add, col_remove = st.columns(2)
            
            with col_add:
                if st.button("ADD TICKERS", use_container_width=True, key="add_tickers_btn"):
                    if custom_ticker_input:
                        new_tickers = [t.strip().upper() for t in custom_ticker_input.split(',') if t.strip()]
                        
                        # Validate tickers before adding
                        validated_tickers = []
                        invalid_tickers = []
                        
                        with st.spinner("Validating tickers..."):
                            for ticker in new_tickers:
                                try:
                                    # Quick validation check
                                    test_data = yf.Ticker(ticker)
                                    info = test_data.info
                                    
                                    # Check if ticker has valid data
                                    if info and ('regularMarketPrice' in info or 'previousClose' in info):
                                        validated_tickers.append(ticker)
                                    else:
                                        invalid_tickers.append(ticker)
                                except:
                                    invalid_tickers.append(ticker)
                        
                        if validated_tickers:
                            # Add to custom tickers list
                            st.session_state.custom_tickers.extend(validated_tickers)
                            st.session_state.custom_tickers = list(set(st.session_state.custom_tickers))
                            
                            # Update selected tickers
                            st.session_state.selected_tickers.extend(validated_tickers)
                            st.session_state.selected_tickers = list(set(st.session_state.selected_tickers))
                            
                            st.success(f"✓ Added: {', '.join(validated_tickers)}")
                            st.rerun()
                        
                        if invalid_tickers:
                            st.error(f"✗ Invalid or not found: {', '.join(invalid_tickers)}")
                            st.info("Please check ticker symbols. Ensure they exist on Yahoo Finance.")
            
            with col_remove:
                # Remove tickers
                tickers_to_remove = st.multiselect(
                    "Select Tickers to Remove",
                    st.session_state.selected_tickers,
                    key="remove_tickers_select"
                )
                
                if st.button("REMOVE SELECTED", use_container_width=True, key="remove_tickers_btn"):
                    if tickers_to_remove:
                        for ticker in tickers_to_remove:
                            if ticker in st.session_state.selected_tickers:
                                st.session_state.selected_tickers.remove(ticker)
                            if ticker in st.session_state.custom_tickers:
                                st.session_state.custom_tickers.remove(ticker)
                        st.success(f"✓ Removed: {', '.join(tickers_to_remove)}")
                        st.rerun()
        
        with col_custom2:
            st.markdown("### Quick Actions")
            
            if st.button("CLEAR ALL", use_container_width=True, key="clear_all_btn"):
                st.session_state.selected_tickers = []
                st.session_state.custom_tickers = []
                st.success("✓ All tickers cleared")
                st.rerun()
            
            if st.button("RESET TO DEFAULT", use_container_width=True, key="reset_default_btn"):
                default_tickers = []
                for asset_class in ['US_LARGE_CAP', 'US_BONDS_INTERMEDIATE', 'GOLD']:
                    default_tickers.extend(ASSET_UNIVERSE[asset_class])
                st.session_state.selected_tickers = list(set(default_tickers))
                st.session_state.custom_tickers = []
                st.success("✓ Reset to default selection")
                st.rerun()
            
            st.markdown("---")
            st.markdown(f"**Total Tickers:** {len(st.session_state.selected_tickers)}")
            
            if len(st.session_state.selected_tickers) < 2:
                st.warning("⚠️ Add at least 2 assets for portfolio optimization")
            elif len(st.session_state.selected_tickers) > 50:
                st.warning("⚠️ Large universe may slow optimization")
        
        if fetch_data_btn and len(st.session_state.selected_tickers) > 0:
            with st.spinner("Fetching market data..."):
                price_data = fetch_market_data(st.session_state.selected_tickers, start_date, end_date)
                
                if not price_data.empty:
                    # Check which tickers actually returned data
                    successful_tickers = list(price_data.columns)
                    failed_tickers = [t for t in st.session_state.selected_tickers if t not in successful_tickers]
                    
                    if failed_tickers:
                        st.warning(f"⚠️ No data found for: {', '.join(failed_tickers)}")
                        st.info("These tickers were removed from the analysis. Check ticker symbols or date range.")
                    
                    # Update selected tickers to only successful ones
                    st.session_state.selected_tickers = successful_tickers
                    
                    st.session_state.price_data = price_data
                    
                    # Calculate statistics
                    returns = calculate_returns(price_data)
                    stats = calculate_statistics(returns)
                    st.session_state.stats = stats
                    
                    st.success(f"✓ Data fetched successfully: {len(price_data)} observations across {len(successful_tickers)} assets")
                    
                    # Display summary statistics
                    st.markdown('<div class="section-header">ASSET STATISTICS</div>', unsafe_allow_html=True)
                    
                    summary_df = pd.DataFrame({
                        'Expected Return (%)': stats['mean_returns'] * 100,
                        'Volatility (%)': stats['volatility'] * 100,
                        'Sharpe Ratio': (stats['mean_returns'] - RISK_FREE_RATE) / stats['volatility']
                    }).round(2)
                    
                    st.dataframe(summary_df, use_container_width=True)
                else:
                    st.error("❌ Failed to fetch data. Please check your ticker selections and try again.")
        
        elif fetch_data_btn and len(st.session_state.selected_tickers) == 0:
            st.error("❌ Please select at least one asset before fetching data")
    
    # ===========================
    # TAB 2: OPTIMIZATION ENGINE
    # ===========================
    with tab2:
        if 'stats' not in st.session_state:
            st.warning("Please fetch market data in Portfolio Construction tab first")
        else:
            st.markdown('<div class="section-header">PORTFOLIO OPTIMIZATION ENGINE</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Optimization Parameters")
                
                objective = st.selectbox(
                    "Portfolio Objective",
                    [
                        "max_sharpe",
                        "min_vol",
                        "risk_parity",
                        "balanced",
                        "defensive",
                        "growth",
                        "income"
                    ],
                    format_func=lambda x: {
                        "max_sharpe": "Maximum Sharpe Ratio",
                        "min_vol": "Minimum Volatility",
                        "risk_parity": "Risk Parity",
                        "balanced": "Balanced (60/40 Style)",
                        "defensive": "Defensive",
                        "growth": "Growth",
                        "income": "Income Focused"
                    }[x]
                )
                
                st.markdown("### Constraints")
                
                min_weight = st.slider("Minimum Asset Weight (%)", 0, 20, 0) / 100
                max_weight = st.slider("Maximum Asset Weight (%)", 10, 100, 50) / 100
                
                constraints = {
                    'min_weight': min_weight,
                    'max_weight': max_weight
                }
                
                optimize_btn = st.button("RUN OPTIMIZATION", type="primary", use_container_width=True)
            
            with col2:
                if optimize_btn:
                    with st.spinner("Optimizing portfolio..."):
                        mean_returns = st.session_state.stats['mean_returns']
                        cov_matrix = st.session_state.stats['cov_matrix']
                        
                        # Optimize
                        optimal_weights = optimize_portfolio(
                            mean_returns,
                            cov_matrix,
                            objective=objective,
                            constraints=constraints
                        )
                        
                        # Calculate performance
                        port_return, port_vol, port_sharpe = portfolio_performance(
                            optimal_weights,
                            mean_returns,
                            cov_matrix
                        )
                        
                        # Store in session state
                        st.session_state.optimized_portfolios[objective] = {
                            'weights': optimal_weights,
                            'tickers': st.session_state.selected_tickers,
                            'return': port_return,
                            'volatility': port_vol,
                            'sharpe': port_sharpe
                        }
                        
                        st.session_state.current_portfolio = st.session_state.optimized_portfolios[objective]
                        
                        st.success("✓ Optimization complete")
                        
                        # Display results
                        st.markdown("### Portfolio Metrics")
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        metric_col1.metric("Expected Return", f"{port_return*100:.2f}%")
                        metric_col2.metric("Volatility", f"{port_vol*100:.2f}%")
                        metric_col3.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
                        
                        # Allocation table
                        st.markdown("### Optimal Allocation")
                        
                        weights_df = pd.DataFrame({
                            'Ticker': st.session_state.selected_tickers,
                            'Weight (%)': optimal_weights * 100,
                            'Expected Return (%)': mean_returns.values * 100,
                            'Volatility (%)': st.session_state.stats['volatility'].values * 100
                        }).sort_values('Weight (%)', ascending=False)
                        
                        weights_df = weights_df[weights_df['Weight (%)'] > 0.01]  # Filter out tiny weights
                        
                        st.dataframe(weights_df.round(2), use_container_width=True, hide_index=True)
                        
                        # Visualization
                        st.markdown("### Portfolio Allocation")
                        st.plotly_chart(
                            plot_portfolio_allocation(optimal_weights, st.session_state.selected_tickers),
                            use_container_width=True
                        )
                        
                        # Risk contribution
                        risk_contrib = calculate_risk_contribution(optimal_weights, cov_matrix)
                        risk_contrib.index = st.session_state.selected_tickers
                        
                        st.markdown("### Risk Contribution Analysis")
                        st.plotly_chart(
                            plot_risk_contribution(risk_contrib, st.session_state.selected_tickers),
                            use_container_width=True
                        )
            
            # Efficient Frontier
            if 'stats' in st.session_state:
                st.markdown('<div class="section-header">EFFICIENT FRONTIER</div>', unsafe_allow_html=True)
                
                if st.button("GENERATE EFFICIENT FRONTIER"):
                    with st.spinner("Generating efficient frontier..."):
                        frontier_df = generate_efficient_frontier(
                            st.session_state.stats['mean_returns'],
                            st.session_state.stats['cov_matrix']
                        )
                        
                        current_port = None
                        if st.session_state.current_portfolio:
                            current_port = {
                                'return': st.session_state.current_portfolio['return'],
                                'volatility': st.session_state.current_portfolio['volatility']
                            }
                        
                        st.plotly_chart(
                            plot_efficient_frontier(frontier_df, current_port),
                            use_container_width=True
                        )
    
    # ===========================
    # TAB 3: REBALANCING MODULE
    # ===========================
    with tab3:
        st.markdown('<div class="section-header">PORTFOLIO REBALANCING ENGINE</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Current Portfolio")
            
            portfolio_source = st.radio(
                "Portfolio Source",
                ["Use Optimized Portfolio", "Manual Input"],
                horizontal=True
            )
            
            if portfolio_source == "Use Optimized Portfolio":
                if st.session_state.optimized_portfolios:
                    selected_portfolio = st.selectbox(
                        "Select Portfolio",
                        list(st.session_state.optimized_portfolios.keys()),
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                    
                    current_weights = pd.Series(
                        st.session_state.optimized_portfolios[selected_portfolio]['weights'],
                        index=st.session_state.optimized_portfolios[selected_portfolio]['tickers']
                    )
                    
                    st.dataframe(
                        pd.DataFrame({
                            'Ticker': current_weights.index,
                            'Weight (%)': current_weights.values * 100
                        }).round(2),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No optimized portfolios available. Please optimize a portfolio first.")
                    current_weights = None
            
            else:
                st.markdown("Enter current portfolio weights:")
                
                manual_tickers = st.text_area(
                    "Tickers (comma-separated)",
                    "AAPL, MSFT, TLT, GLD"
                ).split(',')
                manual_tickers = [t.strip() for t in manual_tickers if t.strip()]
                
                manual_weights = st.text_area(
                    "Weights in % (comma-separated, must sum to 100)",
                    "40, 30, 20, 10"
                ).split(',')
                manual_weights = [float(w.strip())/100 for w in manual_weights if w.strip()]
                
                if len(manual_tickers) == len(manual_weights):
                    if abs(sum(manual_weights) - 1.0) < 0.01:
                        current_weights = pd.Series(manual_weights, index=manual_tickers)
                        st.success("✓ Portfolio loaded")
                    else:
                        st.error(f"Weights sum to {sum(manual_weights)*100:.1f}%, must sum to 100%")
                        current_weights = None
                else:
                    st.error("Number of tickers and weights must match")
                    current_weights = None
        
        with col2:
            st.markdown("### Rebalancing Parameters")
            
            rebalance_trigger = st.selectbox(
                "Rebalancing Trigger",
                [
                    "New Optimization",
                    "Changed Risk Tolerance",
                    "Changed Return Expectations",
                    "Market Shock Adjustment",
                    "Capital Addition/Withdrawal"
                ]
            )
            
            portfolio_value = st.number_input(
                "Current Portfolio Value ($)",
                min_value=1000,
                value=1000000,
                step=10000
            )
            
            transaction_cost_bps = st.slider(
                "Transaction Cost (basis points)",
                0, 50, 10
            )
            
            if rebalance_trigger == "New Optimization":
                new_objective = st.selectbox(
                    "New Portfolio Objective",
                    [
                        "max_sharpe",
                        "min_vol",
                        "risk_parity",
                        "balanced",
                        "defensive",
                        "growth",
                        "income"
                    ],
                    format_func=lambda x: x.replace('_', ' ').title()
                )
            
            elif rebalance_trigger == "Changed Risk Tolerance":
                new_risk_tolerance = st.slider(
                    "New Maximum Volatility (%)",
                    5, 25, 12
                )
            
            elif rebalance_trigger == "Changed Return Expectations":
                return_adjustment = st.slider(
                    "Return Expectation Adjustment (%)",
                    -5.0, 5.0, 0.0, 0.1
                )
            
            rebalance_btn = st.button("CALCULATE REBALANCING", type="primary", use_container_width=True)
        
        if rebalance_btn and current_weights is not None:
            with st.spinner("Calculating rebalancing trades..."):
                # Determine target weights based on trigger
                if rebalance_trigger == "New Optimization" and 'stats' in st.session_state:
                    # Re-optimize with new objective
                    target_weights_array = optimize_portfolio(
                        st.session_state.stats['mean_returns'],
                        st.session_state.stats['cov_matrix'],
                        objective=new_objective
                    )
                    target_weights = pd.Series(
                        target_weights_array,
                        index=st.session_state.selected_tickers
                    )
                
                elif rebalance_trigger == "Changed Risk Tolerance" and 'stats' in st.session_state:
                    # Optimize with new risk constraint
                    target_weights_array = optimize_portfolio(
                        st.session_state.stats['mean_returns'],
                        st.session_state.stats['cov_matrix'],
                        objective='balanced'
                    )
                    target_weights = pd.Series(
                        target_weights_array,
                        index=st.session_state.selected_tickers
                    )
                
                else:
                    # For other triggers, use existing target or current weights
                    target_weights = current_weights
                
                # Calculate rebalancing trades
                trades_df, turnover, total_cost = calculate_rebalancing_trades(
                    current_weights,
                    target_weights,
                    portfolio_value,
                    transaction_cost_bps
                )
                
                # Display results
                st.markdown('<div class="section-header">REBALANCING ANALYSIS</div>', unsafe_allow_html=True)
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Portfolio Turnover", f"{turnover*100:.2f}%")
                metric_col2.metric("Total Transaction Cost", f"${total_cost:,.2f}")
                metric_col3.metric("Cost as % of AUM", f"{(total_cost/portfolio_value)*100:.3f}%")
                
                st.markdown("### Trade List")
                
                trades_display = trades_df[trades_df['Action'] != 'HOLD'].copy()
                trades_display = trades_display.sort_values('Trade Value', key=abs, ascending=False)
                
                st.dataframe(trades_display.round(2), use_container_width=True)
                
                # Allocation comparison
                st.markdown("### Allocation Comparison")
                
                comparison_df = pd.DataFrame({
                    'Asset': trades_df.index,
                    'Current %': trades_df['Current Weight %'],
                    'Target %': trades_df['Target Weight %'],
                    'Change %': trades_df['Weight Change %']
                })
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Current',
                    x=comparison_df['Asset'],
                    y=comparison_df['Current %'],
                    marker_color='#ff6b35'
                ))
                
                fig.add_trace(go.Bar(
                    name='Target',
                    x=comparison_df['Asset'],
                    y=comparison_df['Target %'],
                    marker_color='#00d4ff'
                ))
                
                fig.update_layout(
                    title='Current vs Target Allocation',
                    barmode='group',
                    xaxis_title='Asset',
                    yaxis_title='Weight (%)',
                    template='plotly_dark',
                    paper_bgcolor='#0a0e27',
                    plot_bgcolor='#0f1229',
                    font=dict(family='Courier New', size=12, color='#e8e8e8')
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # ===========================
    # TAB 4: ANALYTICS & REPORTING
    # ===========================
    with tab4:
        st.markdown('<div class="section-header">PORTFOLIO ANALYTICS & REPORTING</div>', unsafe_allow_html=True)
        
        if 'price_data' not in st.session_state:
            st.warning("Please fetch market data first")
        else:
            # Correlation matrix
            st.markdown("### Asset Correlation Matrix")
            st.plotly_chart(
                plot_correlation_matrix(st.session_state.stats['corr_matrix']),
                use_container_width=True
            )
            
            # Historical performance
            if st.session_state.current_portfolio:
                st.markdown("### Historical Performance")
                
                st.plotly_chart(
                    plot_historical_performance(
                        st.session_state.price_data,
                        st.session_state.current_portfolio['weights'],
                        st.session_state.current_portfolio['tickers']
                    ),
                    use_container_width=True
                )
                
                # Drawdown analysis
                st.markdown("### Drawdown Analysis")
                
                returns = calculate_returns(st.session_state.price_data)
                portfolio_returns = (returns * st.session_state.current_portfolio['weights']).sum(axis=1)
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values * 100,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='#ff6b35', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 53, 0.2)'
                ))
                
                fig.update_layout(
                    title='Portfolio Drawdown Over Time',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    template='plotly_dark',
                    paper_bgcolor='#0a0e27',
                    plot_bgcolor='#0f1229',
                    font=dict(family='Courier New', size=12, color='#e8e8e8'),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance statistics
                st.markdown("### Performance Statistics")
                
                max_drawdown = drawdown.min()
                annual_return = portfolio_returns.mean() * 252
                annual_vol = portfolio_returns.std() * np.sqrt(252)
                sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0
                
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                stat_col1.metric("Annual Return", f"{annual_return*100:.2f}%")
                stat_col2.metric("Annual Volatility", f"{annual_vol*100:.2f}%")
                stat_col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                stat_col4.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")

if __name__ == "__main__":
    main()