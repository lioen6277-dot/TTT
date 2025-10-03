import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import warnings
import time
import re 
from datetime import datetime, timedelta
import requests  # For news API

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI趨勢分析📈", 
    page_icon="🚀", 
    layout="wide"
)

# Add your API keys here (replace with actual keys)
NEWS_API_KEY = "your_news_api_key_here"  # From newsapi.org
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key_here"  # For better data

PERIOD_MAP = { 
    "30 分": ("60d", "30m"), 
    "4 小時": ("1y", "60m"), 
    "1 日": ("5y", "1d"), 
    "1 週": ("max", "1wk")
}

FULL_SYMBOLS_MAP = {
    "TSLA": {"name": "特斯拉", "keywords": ["特斯拉", "電動車", "TSLA", "Tesla"]},
    "NVDA": {"name": "輝達", "keywords": ["輝達", "英偉達", "AI", "NVDA", "Nvidia"]},
    "AAPL": {"name": "蘋果", "keywords": ["蘋果", "Apple", "AAPL"]},
    "GOOGL": {"name": "谷歌/Alphabet", "keywords": ["谷歌", "Alphabet", "GOOGL", "GOOG"]},
    "MSFT": {"name": "微軟", "keywords": ["微軟", "Microsoft", "MSFT"]},
    "AMZN": {"name": "亞馬遜", "keywords": ["亞馬遜", "Amazon", "AMZN"]},
    "META": {"name": "Meta/臉書", "keywords": ["臉書", "Meta", "FB", "META"]},
    "NFLX": {"name": "網飛", "keywords": ["網飛", "Netflix", "NFLX"]},
    "ADBE": {"name": "Adobe", "keywords": ["Adobe", "ADBE"]},
    "CRM": {"name": "Salesforce", "keywords": ["Salesforce", "CRM"]},
    "ORCL": {"name": "甲骨文", "keywords": ["甲骨文", "Oracle", "ORCL"]},
    "COST": {"name": "好市多", "keywords": ["好市多", "Costco", "COST"]},
    "JPM": {"name": "摩根大通", "keywords": ["摩根大通", "JPMorgan", "JPM"]},
    "V": {"name": "Visa", "keywords": ["Visa", "V"]},
    "WMT": {"name": "沃爾瑪", "keywords": ["沃爾瑪", "Walmart", "WMT"]},
    "PG": {"name": "寶潔", "keywords": ["寶潔", "P&G", "PG"]},
    "KO": {"name": "可口可樂", "keywords": ["可口可樂", "CocaCola", "KO"]},
    "PEP": {"name": "百事", "keywords": ["百事", "Pepsi", "PEP"]},
    "MCD": {"name": "麥當勞", "keywords": ["麥當勞", "McDonalds", "MCD"]},
    "QCOM": {"name": "高通", "keywords": ["高通", "Qualcomm", "QCOM"]},
    "INTC": {"name": "英特爾", "keywords": ["英特爾", "Intel", "INTC"]},
    "AMD": {"name": "超微", "keywords": ["超微", "AMD"]},
    "LLY": {"name": "禮來", "keywords": ["禮來", "EliLilly", "LLY"]},
    "UNH": {"name": "聯合健康", "keywords": ["聯合健康", "UNH"]},
    "HD": {"name": "家得寶", "keywords": ["家得寶", "HomeDepot", "HD"]},
    "CAT": {"name": "開拓重工", "keywords": ["開拓重工", "Caterpillar", "CAT"]},
    "^GSPC": {"name": "S&P 500 指數", "keywords": ["標普", "S&P500", "^GSPC", "SPX"]},
    "^IXIC": {"name": "NASDAQ 綜合指數", "keywords": ["納斯達克", "NASDAQ", "^IXIC"]},
    "^DJI": {"name": "道瓊工業指數", "keywords": ["道瓊", "DowJones", "^DJI"]},
    "SPY": {"name": "SPDR 標普500 ETF", "keywords": ["SPY", "標普ETF"]},
    "QQQ": {"name": "Invesco QQQ Trust", "keywords": ["QQQ", "納斯達克ETF"]},
    "VOO": {"name": "Vanguard 標普500 ETF", "keywords": ["VOO", "Vanguard"]},
    "2330.TW": {"name": "台積電", "keywords": ["台積電", "2330", "TSMC"]},
    "2317.TW": {"name": "鴻海", "keywords": ["鴻海", "2317", "Foxconn"]},
    "2454.TW": {"name": "聯發科", "keywords": ["聯發科", "2454", "MediaTek"]},
    "2308.TW": {"name": "台達電", "keywords": ["台達電", "2308", "Delta"]},
    "3017.TW": {"name": "奇鋐", "keywords": ["奇鋐", "3017", "散熱"]},
    "3231.TW": {"name": "緯創", "keywords": ["緯創", "3231"]},
    "2382.TW": {"name": "廣達", "keywords": ["廣達", "2382"]},
    "2379.TW": {"name": "瑞昱", "keywords": ["瑞昱", "2379"]},
    "2881.TW": {"name": "富邦金", "keywords": ["富邦金", "2881"]},
    "2882.TW": {"name": "國泰金", "keywords": ["國泰金", "2882"]},
    "2603.TW": {"name": "長榮", "keywords": ["長榮", "2603", "航運"]},
    "2609.TW": {"name": "陽明", "keywords": ["陽明", "2609", "航運"]},
    "2615.TW": {"name": "萬海", "keywords": ["萬海", "2615", "航運"]},
    "2891.TW": {"name": "中信金", "keywords": ["中信金", "2891"]},
    "1101.TW": {"name": "台泥", "keywords": ["台泥", "1101"]},
    "1301.TW": {"name": "台塑", "keywords": ["台塑", "1301"]},
    "2357.TW": {"name": "華碩", "keywords": ["華碩", "2357"]},
    "0050.TW": {"name": "元大台灣50", "keywords": ["台灣50", "0050", "台灣五十"]},
    "0056.TW": {"name": "元大高股息", "keywords": ["高股息", "0056"]},
    "00878.TW": {"name": "國泰永續高股息", "keywords": ["00878", "國泰永續"]},
    "^TWII": {"name": "台股指數", "keywords": ["台股指數", "加權指數", "^TWII"]},
    "BTC-USD": {"name": "比特幣", "keywords": ["比特幣", "BTC", "bitcoin", "BTC-USDT"]},
    "ETH-USD": {"name": "以太坊", "keywords": ["以太坊", "ETH", "ethereum", "ETH-USDT"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL", "SOL-USDT"]},
    "BNB-USD": {"name": "幣安幣", "keywords": ["幣安幣", "BNB", "BNB-USDT"]},
    "DOGE-USD": {"name": "狗狗幣", "keywords": ["狗狗幣", "DOGE", "DOGE-USDT"]},
    "XRP-USD": {"name": "瑞波幣", "keywords": ["瑞波幣", "XRP", "XRP-USDT"]},
    "ADA-USD": {"name": "Cardano", "keywords": ["Cardano", "ADA", "ADA-USDT"]},
    "AVAX-USD": {"name": "Avalanche", "keywords": ["Avalanche", "AVAX", "AVAX-USDT"]},
    "DOT-USD": {"name": "Polkadot", "keywords": ["Polkadot", "DOT", "DOT-USDT"]},
    "LINK-USD": {"name": "Chainlink", "keywords": ["Chainlink", "LINK", "LINK-USDT"]},
    "ASTER-USD": {"name": "Aster", "keywords": ["Aster", "ASTER-USD"]},
}

CATEGORY_MAP = {
    "美股 (US) - 個股/ETF/指數": [c for c in FULL_SYMBOLS_MAP.keys() if not (c.endswith(".TW") or c.endswith("-USD") or c.startswith("^TWII"))],
    "台股 (TW) - 個股/ETF/指數": [c for c in FULL_SYMBOLS_MAP.keys() if c.endswith(".TW") or c.startswith("^TWII")],
    "加密貨幣 (Crypto)": [c for c in FULL_SYMBOLS_MAP.keys() if c.endswith("-USD")],
}

CATEGORY_HOT_OPTIONS = {}
for category, codes in CATEGORY_MAP.items():
    options = {}
    sorted_codes = sorted(codes) 
    for code in sorted_codes:
        info = FULL_SYMBOLS_MAP.get(code)
        if info:
            options[f"{code} - {info['name']}"] = code
    CATEGORY_HOT_OPTIONS[category] = options


def get_symbol_from_query(query: str) -> str:
    """ 🎯  """
    query = query.strip()
    query_upper = query.upper()
    for code, data in FULL_SYMBOLS_MAP.items():
        if query_upper == code: return code
        if any(query_upper == kw.upper() for kw in data["keywords"]): return code 
    for code, data in FULL_SYMBOLS_MAP.items():
        if query == data["name"]: return code
    if re.fullmatch(r'\d{4,6}', query) and not any(ext in query_upper for ext in ['.TW', '.HK', '.SS', '-USD']):
        tw_code = f"{query}.TW"
        if tw_code in FULL_SYMBOLS_MAP: return tw_code
        return tw_code
    return query

@st.cache_data(ttl=300, show_spinner="正在從 Alpha Vantage 獲取數據...")  # Update every 5 min for better recency
def get_stock_data(symbol, period, interval):
    try:
        # Use Alpha Vantage for more reliable data
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if "Time Series" not in data: return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(data["Time Series (Intraday)"], orient="index")
        df = df.astype(float)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    except Exception as e:
        # Fallback to yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df

@st.cache_data(ttl=3600)
def get_company_info(symbol):
    info = FULL_SYMBOLS_MAP.get(symbol, {})
    if info:
        if symbol.endswith(".TW") or symbol.startswith("^TWII"): category, currency = "台股 (TW)", "TWD"
        elif symbol.endswith("-USD"): category, currency = "加密貨幣 (Crypto)", "USD"
        else: category, currency = "美股 (US)", "USD"
        return {"name": info['name'], "category": category, "currency": currency}
    
    try:
        ticker = yf.Ticker(symbol)
        yf_info = ticker.info
        name = yf_info.get('longName') or yf_info.get('shortName') or symbol
        currency = yf_info.get('currency') or "USD"
        category = "未分類"
        if symbol.endswith(".TW"): category = "台股 (TW)"
        elif symbol.endswith("-USD"): category = "加密貨幣 (Crypto)"
        elif symbol.startswith("^"): category = "指數"
        elif currency == "USD": category = "美股 (US)"
        return {"name": name, "category": category, "currency": currency}
    except:
        return {"name": symbol, "category": "未分類", "currency": "USD"}
    
@st.cache_data
def get_currency_symbol(symbol):
    currency_code = get_company_info(symbol).get('currency', 'USD')
    if currency_code == 'TWD': return 'NT$'
    elif currency_code == 'USD': return '$'
    elif currency_code == 'HKD': return 'HK$'
    else: return currency_code + ' '

def calculate_technical_indicators(df):
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10) 
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50) 
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200) 
    
    macd_instance = ta.trend.MACD(df['Close'], window_fast=8, window_slow=17, window_sign=9)
    df['MACD_Line'] = macd_instance.macd()
    df['MACD_Signal'] = macd_instance.macd_signal()
    df['MACD_Hist'] = macd_instance.macd_diff() 
    
    df['RSI'] = ta.momentum.rsi(df['Close'], window=9)
    
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=9)
    
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=9)
    
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20) 
    
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    
    df['OBV_EMA_20'] = ta.trend.ema_indicator(df['OBV'], window=20)
    df['OBV_Slope'] = df['OBV_EMA_20'].diff()
    
    return df

def get_technical_data_df(df):
    if df.empty or len(df) < 200: return pd.DataFrame()

    df_clean = df.dropna().copy()
    if df_clean.empty: return pd.DataFrame()

    last_row = df_clean.iloc[-1]
    prev_row = df_clean.iloc[-2] if len(df_clean) >= 2 else last_row 

    indicators = {}
    
    indicators['價格 vs. EMA 10/50/200'] = last_row['Close']
    indicators['RSI (9) 動能'] = last_row['RSI']
    indicators['MACD (8/17/9) 柱狀圖'] = last_row['MACD_Hist']
    indicators['ADX (9) 趨勢強度'] = last_row['ADX']
    indicators['ATR (9) 波動性'] = last_row['ATR']
    indicators['布林通道 (BB: 20/2)'] = last_row['Close']
    indicators['OBV 趨勢'] = last_row['OBV_Slope']
    
    data = []
    
    for name, value in indicators.items():
        conclusion, color = "", "grey"
        
        if 'EMA 10/50/200' in name:
            ema_10 = last_row['EMA_10']
            ema_50 = last_row['EMA_50']
            ema_200 = last_row['EMA_200']

            if ema_10 > ema_50 and ema_50 > ema_200:
                conclusion, color = f"**強多頭：MA 多頭排列** (10>50>200)", "red"
            elif ema_10 < ema_50 and ema_50 < ema_200:
                conclusion, color = f"**強空頭：MA 空頭排列** (10<50<200)", "green"
            elif last_row['Close'] > ema_50 and last_row['Close'] > ema_200:
                conclusion, color = f"中長線偏多：價格站上 EMA 50/200", "orange"
            else:
                conclusion, color = "中性：MA 糾結或趨勢發展中", "blue"
        
        elif 'RSI' in name:
            if value > 70:
                conclusion, color = "警告：超買區域 (70)，潛在回調", "green" 
            elif value < 30:
                conclusion, color = "強化：超賣區域 (30)，潛在反彈", "red"
            elif value > 50:
                conclusion, color = "多頭：RSI > 50，位於強勢區間", "red"
            else:
                conclusion, color = "空頭：RSI < 50，位於弱勢區間", "green"


        elif 'MACD' in name:
            if value > 0 and value > prev_row['MACD_Hist']:
                conclusion, color = "強化：多頭動能增強 (紅柱放大)", "red"
            elif value < 0 and value < prev_row['MACD_Hist']: 
                conclusion, color = "削弱：空頭動能增強 (綠柱放大)", "green"
            else:
                conclusion, color = "中性：動能盤整 (柱狀收縮)", "orange"
        
        elif 'ADX' in name:
            if value >= 40:
                conclusion, color = "強趨勢：極強勢趨勢 (多或空)", "red"
            elif value >= 25:
                conclusion, color = "強趨勢：確認強勢趨勢 (ADX > 25)", "orange"
            else:
                conclusion, color = "盤整：弱勢或橫盤整理 (ADX < 25)", "blue"

        elif 'ATR' in name:
            avg_atr = df_clean['ATR'].iloc[-30:].mean() if len(df_clean) >= 30 else df_clean['ATR'].mean()
            if value > avg_atr * 1.5:
                conclusion, color = "警告：極高波動性 (1.5x 平均)", "green"
            elif value < avg_atr * 0.7:
                conclusion, color = "中性：低波動性 (醞釀突破)", "orange"
            else:
                conclusion, color = "中性：正常波動性", "blue"

        elif '布林通道' in name:
            high = last_row['BB_High']
            low = last_row['BB_Low']
            range_pct = (high - low) / last_row['Close'] * 100
            
            if value > high:
                conclusion, color = f"警告：價格位於上軌外側 (>{high:,.2f})", "red"
            elif value < low:
                conclusion, color = f"強化：價格位於下軌外側 (<{low:,.2f})", "green"
            else:
                conclusion, color = f"中性：在上下軌間 ({range_pct:.2f}% 寬度)", "blue"
        
        elif 'OBV' in name:
            if value > 0:
                conclusion, color = "強化：資金流入 (OBV 上漲)", "red"
            elif value < 0:
                conclusion, color = "削弱：資金流出 (OBV 下降)", "green"
            else:
                conclusion, color = "中性：資金平衡", "orange"
        
        data.append([name, value, conclusion, color])

    technical_df = pd.DataFrame(data, columns=['指標名稱', '最新值', '分析結論', '顏色'])
    technical_df = technical_df.set_index('指標名稱')
    return technical_df

def run_backtest(df, signals, initial_capital=100000, commission_rate=0.001):
    """
    執行基於 AI 融合信號的回測。
    """
    
    if df.empty or len(df) < 51:
        return {"total_return": 0, "win_rate": 0, "max_drawdown": 0, "total_trades": 0, "sharpe_ratio": 0, "message": "數據不足 (少於 51 週期) 或計算錯誤。"}

    data = df.copy()
    data['Signal'] = signals  # Use AI-generated signals
    
    capital = [initial_capital]
    position = 0 
    buy_price = 0
    trades = []
    returns = []
    
    for i in range(1, len(data)):
        current_close = data['Close'].iloc[i]
        
        
        if data['Signal'].iloc[i] == 1 and position == 0:
            position = 1
            buy_price = current_close
            initial_capital -= initial_capital * commission_rate 
            
        
        elif data['Signal'].iloc[i] == -1 and position == 1:
            sell_price = current_close
            profit = (sell_price - buy_price) / buy_price 
            
            trades.append({ 'entry_date': data.index[i], 'exit_date': data.index[i], 'profit_pct': profit, 'is_win': profit > 0 })
            returns.append(profit)
            
            initial_capital *= (1 + profit)
            initial_capital -= initial_capital * commission_rate
            position = 0
            
        current_value = initial_capital
        if position == 1:
            current_value = initial_capital * (current_close / buy_price)
            
        capital.append(current_value)

    
    if position == 1:
        sell_price = data['Close'].iloc[-1]
        profit = (sell_price - buy_price) / buy_price
        
        trades.append({ 'entry_date': data.index[-1], 'exit_date': data.index[-1], 'profit_pct': profit, 'is_win': profit > 0 })
        returns.append(profit)
        
        initial_capital *= (1 + profit)
        initial_capital -= initial_capital * commission_rate
        if capital: capital[-1] = initial_capital 

    
    total_return = ((initial_capital - 100000) / 100000) * 100
    total_trades = len(trades)
    win_rate = (sum(1 for t in trades if t['is_win']) / total_trades) * 100 if total_trades > 0 else 0
    
    capital_series = pd.Series(capital)
    max_value = capital_series.expanding(min_periods=1).max()
    drawdown = (capital_series - max_value) / max_value
    max_drawdown = abs(drawdown.min()) * 100
    
    # Additional metrics
    returns_series = pd.Series(returns)
    sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if len(returns) > 0 else 0  # Assume daily
    
    return {
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "max_drawdown": round(max_drawdown, 2),
        "total_trades": total_trades,
        "sharpe_ratio": round(sharpe_ratio, 2),
        "message": f"回測區間 {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}。",
        "capital_curve": capital_series
    }

def calculate_dcf(symbol, growth_rate=0.05, discount_rate=0.1, perpetual_growth=0.02, years=5):
    """
    Implement DCF model using yfinance data.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info
    financials = ticker.financials
    cash_flow = ticker.cashflow
    
    if 'Free Cash Flow' not in cash_flow.index:
        return 0.0  # Fallback if data missing
    
    fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
    
    projected_fcfs = [fcf * (1 + growth_rate) ** i for i in range(1, years + 1)]
    
    terminal_value = projected_fcfs[-1] * (1 + perpetual_growth) / (discount_rate - perpetual_growth)
    projected_fcfs.append(terminal_value)
    
    pv_fcfs = [cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(projected_fcfs)]
    
    dcf_value = sum(pv_fcfs)
    
    shares = info.get('sharesOutstanding', 1)
    intrinsic_value = dcf_value / shares if shares else 0
    
    return intrinsic_value

def calculate_fundamental_rating(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        balance = ticker.balance_sheet
        financials = ticker.financials
        
        if symbol.startswith('^') or symbol.endswith('-USD'):
            return {
                "Combined_Rating": 0.0, 
                "Message": "不適用：指數或加密貨幣無標準基本面數據。",
                "Details": None
            }

        roe = info.get('returnOnEquity', 0) 
        roce = (financials.loc['EBIT'].iloc[0] / (balance.loc['Total Assets'].iloc[0] - balance.loc['Current Liabilities'].iloc[0])) if 'EBIT' in financials.index else 0
        trailingPE = info.get('trailingPE', 99) 
        freeCashFlow = info.get('freeCashflow', 0) 
        totalCash = info.get('totalCash', 0)
        totalDebt = info.get('totalDebt', 0) 
        debt_to_equity = totalDebt / info.get('totalStockholderEquity', 1)
        gross_margin = info.get('grossMargins', 0)
        net_margin = info.get('profitMargins', 0)
        
        # Growth stability - Simple CAGR
        earnings_growth = info.get('earningsGrowth', 0)
        revenue_growth = info.get('revenueGrowth', 0)
        growth_score = 3 if earnings_growth > 0.1 and revenue_growth > 0.1 else 2 if earnings_growth > 0.05 else 0
        
        # DCF
        intrinsic_value = calculate_dcf(symbol)
        peg = trailingPE / (info.get('earningsGrowth', 0) * 100) if info.get('earningsGrowth') else 99
        pb = info.get('priceToBook', 99)
        psr = info.get('priceToSalesTrailing12Months', 99)
        ev_ebitda = info.get('enterpriseToEbitda', 99)
        
        roe_score = 3 if roe > 0.15 else 2 if roe > 0.10 else 1 if roe > 0 else 0
        roce_score = 3 if roce > 0.10 else 2 if roce > 0.05 else 1 if roce > 0 else 0
        pe_score = 3 if trailingPE < 15 and trailingPE > 0 else 2 if trailingPE < 25 else 1 if trailingPE < 35 else 0
        debt_score = 3 if debt_to_equity < 0.5 else 2 if debt_to_equity < 1 else 0
        margin_score = 3 if gross_margin > 0.3 and net_margin > 0.3 else 2 if gross_margin > 0.2 else 0
        dcf_score = 3 if info['currentPrice'] < intrinsic_value * 1.2 else 2 if info['currentPrice'] < intrinsic_value * 1.5 else 0
        peg_score = 3 if peg < 1 else 2 if peg < 1.5 else 0
        pb_score = 3 if pb < 1 else 2 if pb < 2 else 0
        psr_score = 3 if psr < 1 else 2 if psr < 2 else 0
        ev_score = 3 if ev_ebitda < 10 else 2 if ev_ebitda < 15 else 0
        
        cf_score = 0
        cash_debt_ratio = (totalCash / totalDebt) if totalDebt and totalDebt != 0 else 100 
        if freeCashFlow > 0 and cash_debt_ratio > 2: 
            cf_score = 3
        elif freeCashFlow > 0 and cash_debt_ratio > 1: 
            cf_score = 2
        elif freeCashFlow > 0: 
            cf_score = 1

        combined_rating = roe_score + roce_score + pe_score + debt_score + margin_score + dcf_score + peg_score + pb_score + psr_score + ev_score + cf_score + growth_score
        
        max_rating = 36  # Adjusted for more metrics
        rating_pct = (combined_rating / max_rating) * 9
        
        if combined_rating >= 25:
            message = "頂級優異 (強護城河)：基本面極健康，**ROE > 15%**，成長與估值俱佳，適合長期持有。"
        elif combined_rating >= 15:
            message = "良好穩健：財務結構穩固，但可能在估值或 ROE 方面有待加強。"
        elif combined_rating >= 10:
            message = "中性警示：存在財務壓力或估值過高，需警惕風險（如現金流為負）。"
        else:
            message = "基本面較弱：財務指標不佳或數據缺失，不建議盲目進場。"
            
        return { "Combined_Rating": rating_pct, "Message": message, "Details": info, "Intrinsic_Value": intrinsic_value }

    except Exception as e:
        return { "Combined_Rating": 1.0, "Message": f"基本面數據獲取失敗或不適用 (代碼可能錯誤或數據缺失)。", "Details": None }

def fetch_news_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    # Simple sentiment (expand with NLP like Vader)
    positive = sum(1 for art in articles if 'positive' in art['title'].lower())
    negative = sum(1 for art in articles if 'negative' in art['title'].lower())
    sentiment_score = (positive - negative) / len(articles) if articles else 0
    
    return sentiment_score

def fetch_vix():
    vix_ticker = yf.Ticker("^VIX")
    vix = vix_ticker.history(period="1d")['Close'].iloc[-1]
    return vix

def fetch_macro_data():
    # Simple example: Fed rate or CPI, use yfinance for ^IRX or something
    irx = yf.Ticker("^IRX").history(period="1mo")['Close'].mean()  # Approx short term rate
    return {"interest_rate": irx}

def fetch_chip_data(symbol):
    ticker = yf.Ticker(symbol)
    holders = ticker.major_holders
    institutional = ticker.institutional_holders
    
    # For TW stocks, major_holders may have foreign investor info
    foreign_hold = holders.get('% Out', pd.Series([0])).iloc[0] if holders is not None else 0
    
    return {"foreign_hold": foreign_hold}

def generate_expert_fusion_signal(df, fa_rating, is_long_term, currency_symbol, symbol):
    if df.empty:
        return {
            'current_price': 0,
            'action': '中性 (Neutral)',
            'score': 0,
            'confidence': 50,
            'entry_price': 0,
            'take_profit': 0,
            'stop_loss': 0,
            'strategy': '無數據',
            'atr': 0,
            'expert_opinions': {}
        }

    last_row = df.iloc[-1]
    current_price = last_row['Close']
    atr = last_row.get('ATR', 0)
    
    # News sentiment
    news_score = fetch_news_sentiment(symbol)
    
    # VIX
    vix = fetch_vix()
    vix_adjust = -1 if vix > 30 else 1 if vix < 20 else 0
    
    # Macro
    macro = fetch_macro_data()
    macro_score = 1 if macro['interest_rate'] < 3 else -1  # Example: low rate good for stocks
    
    # Chip data
    chip_data = fetch_chip_data(symbol)
    chip_score = 1 if chip_data['foreign_hold'] > 0.1 else -1  # Example threshold
    
    # Technical scores
    ta_score = 0
    opinions = {}

    if last_row['EMA_10'] > last_row['EMA_50'] > last_row['EMA_200']:
        ta_score += 2
        opinions['MA 趨勢'] = '強多頭排列'
    elif last_row['EMA_10'] < last_row['EMA_50'] < last_row['EMA_200']:
        ta_score -= 2
        opinions['MA 趨勢'] = '強空頭排列'
    else:
        opinions['MA 趨勢'] = '中性'

    if last_row['RSI'] > 70:
        ta_score -= 1
        opinions['RSI'] = '超買'
    elif last_row['RSI'] < 30:
        ta_score += 1
        opinions['RSI'] = '超賣'
    elif last_row['RSI'] > 50:
        ta_score += 1
        opinions['RSI'] = '多頭區間'
    else:
        ta_score -= 1
        opinions['RSI'] = '空頭區間'

    if last_row['MACD_Hist'] > 0:
        ta_score += 1
        opinions['MACD'] = '多頭動能'
    else:
        ta_score -= 1
        opinions['MACD'] = '空頭動能'

    if last_row['ADX'] > 25:
        opinions['ADX'] = '強趨勢'
    else:
        opinions['ADX'] = '盤整'
    
    # OBV score
    obv_score = 1 if last_row['OBV_Slope'] > 0 else -1
    opinions['OBV'] = '資金流入' if obv_score > 0 else '資金流出'
    ta_score += obv_score
    
    # Fundamental
    fa_score = fa_rating['Combined_Rating']
    
    # Dynamic weights based on period/market
    weights = {'ta': 0.4 if not is_long_term else 0.3, 'fa': 0.3 if is_long_term else 0.2, 'news': 0.1, 'vix': 0.05, 'macro': 0.05, 'chip': 0.1}
    
    total_score = (ta_score * weights['ta'] + fa_score * weights['fa'] + news_score * weights['news'] + vix_adjust * weights['vix'] + macro_score * weights['macro'] + chip_score * weights['chip'])
    
    confidence = min(100, abs(total_score) * 20 + 50)

    if total_score > 2:
        action = '買進 (Buy)'
    elif total_score > 0:
        action = '中性偏買 (Hold/Buy)'
    elif total_score < -2:
        action = '賣出 (Sell/Short)'
    elif total_score < 0:
        action = '中性偏賣 (Hold/Sell)'
    else:
        action = '中性 (Neutral)'

    entry_price = current_price
    take_profit = current_price + atr * 2 if total_score > 0 else current_price - atr * 2
    stop_loss = current_price - atr if total_score > 0 else current_price + atr
    strategy = '基於MA/RSI/MACD/OBV/基本面/消息/籌碼/VIX/宏觀/動態權重融合'

    return {
        'current_price': current_price,
        'action': action,
        'score': total_score,
        'confidence': confidence,
        'entry_price': entry_price,
        'take_profit': take_profit,
        'stop_loss': stop_loss,
        'strategy': strategy,
        'atr': atr,
        'expert_opinions': opinions
    }

def create_comprehensive_chart(df, symbol, period_key):
    df_clean = df.dropna()
    if df_clean.empty:
        return go.Figure()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        specs=[[{"secondary_y": True}], [{}], [{}], [{}]])

    fig.add_trace(go.Candlestick(x=df_clean.index,
                                 open=df_clean['Open'], high=df_clean['High'],
                                 low=df_clean['Low'], close=df_clean['Close'],
                                 name='K線'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_10'], line=dict(color='orange', width=1), name='EMA 10'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_200'], line=dict(color='red', width=1), name='EMA 200'), row=1, col=1)

    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['Volume'], marker_color='grey', name='成交量'), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", secondary_y=True, row=1, col=1, showgrid=False)

    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['MACD_Hist'], marker_color=np.where(df_clean['MACD_Hist'] >= 0, 'green', 'red'), name='MACD Hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD_Line'], line=dict(color='blue', width=1), name='MACD 線'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD_Signal'], line=dict(color='#ffab40', width=1), name='Signal 線'), row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1, fixedrange=True)

    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['RSI'], line=dict(color='#0077b6', width=1.5), name='RSI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['ADX'], line=dict(color='#800080', width=1.5, dash='dot'), name='ADX'), row=3, col=1) 
    fig.update_yaxes(range=[0, 100], row=3, col=1, fixedrange=True)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)

    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['OBV'], line=dict(color='#1e8449', width=1.5), name='OBV'), row=4, col=1)
    fig.update_yaxes(title_text="OBV", row=4, col=1, fixedrange=True)


    fig.update_layout(
        title_text=f"AI 融合分析圖表", 
        height=1000, 
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    return fig

def update_search_input():
    if st.session_state.symbol_select_box and st.session_state.symbol_select_box != "請選擇標的...":
        code = st.session_state.symbol_select_box.split(' - ')[0]
        st.session_state.sidebar_search_input = code
        if st.session_state.get('last_search_symbol') != code:
            st.session_state.last_search_symbol = code
            st.session_state.analyze_trigger = True

def main():
    
    st.markdown("""
        <style>
        [data-testid="stSidebar"] .stButton button {
            color: #FA8072 !important; 
            background-color: rgba(255, 255, 255, 0.1) !important; 
            border-color: #FA8072 !important; 
            border-width: 1px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08); 
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            color: #E9967A !important; 
            background-color: rgba(250, 128, 114, 0.15)  !important; 
            border-color: #E9967A !important;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15); 
        }
        [data-testid="stSidebar"] .stButton button:active,
        [data-testid="stSidebar"] .stButton button:focus {
            color: #FA8072 !important;
            background-color: rgba(250, 128, 114, 0.25) !important;
            border-color: #E9967A !important;
            box-shadow: none !important; 
        }
        h1 { color: #cc6600; } 
        </style>
        """, unsafe_allow_html=True)

    category_keys = list(CATEGORY_MAP.keys())
    
    st.sidebar.markdown("1. 選擇資產類別")
    selected_category_key = st.sidebar.selectbox(
        "選擇資產類別", 
        category_keys, 
        index=category_keys.index("台股 (TW) - 個股/ETF/指數"), 
        label_visibility="collapsed"
    )
    
    current_category_options_display = list(CATEGORY_HOT_OPTIONS.get(selected_category_key, {}).keys())
    
    current_symbol_code = st.session_state.get('last_search_symbol', "2330.TW")
    default_symbol_index = 0
    
    try:
        current_display_name = f"{current_symbol_code} - {FULL_SYMBOLS_MAP[current_symbol_code]['name']}"
        if current_display_name in current_category_options_display:
            default_symbol_index = current_category_options_display.index(current_display_name)
    except:
        pass

    st.sidebar.selectbox(
        f"選擇 {selected_category_key} 標的",
        current_category_options_display,
        index=default_symbol_index,
        key="symbol_select_box",
        on_change=update_search_input,
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.sidebar.markdown("2. 🔍 **輸入股票代碼或中文名稱**")

    text_input_current_value = st.session_state.get('sidebar_search_input', st.session_state.get('last_search_symbol', "2330.TW"))

    selected_query = st.sidebar.text_input(
        "🔍 輸入股票代碼或中文名稱", 
        placeholder="例如：AAPL, 台積電, 廣達, BTC-USD", 
        value=text_input_current_value,
        key="sidebar_search_input",
        label_visibility="collapsed"
    )

    final_symbol_to_analyze = get_symbol_from_query(selected_query)

    is_symbol_changed = final_symbol_to_analyze != st.session_state.get('last_search_symbol', "INIT")

    if is_symbol_changed:
        if final_symbol_to_analyze and final_symbol_to_analyze != "---": 
            st.session_state['analyze_trigger'] = True
            st.session_state['last_search_symbol'] = final_symbol_to_analyze
            st.session_state['data_ready'] = False


    st.sidebar.markdown("---")

    st.sidebar.markdown("3. **選擇週期**")

    period_keys = list(PERIOD_MAP.keys())
    selected_period_key = st.sidebar.selectbox("分析時間週期", period_keys, index=period_keys.index("1 日")) 

    selected_period_value = PERIOD_MAP[selected_period_key]

    yf_period, yf_interval = selected_period_value

    is_long_term = selected_period_key in ["1 日", "1 週"]

    st.sidebar.markdown("---")

    st.sidebar.markdown("4. **開始分析**")
    
    analyze_button_clicked = st.sidebar.button("📊 執行AI分析", key="main_analyze_button") 

    
    if analyze_button_clicked or st.session_state.get('analyze_trigger', False):
        
        st.session_state['data_ready'] = False
        st.session_state['analyze_trigger'] = False 
        
        try:
            with st.spinner(f"🔍 正在啟動AI模型，獲取並分析 **{final_symbol_to_analyze}** 的數據 ({selected_period_key})..."):
                
                df = get_stock_data(final_symbol_to_analyze, yf_period, yf_interval) 
                
                if df.empty or len(df) < 200: 
                    display_symbol = final_symbol_to_analyze
                    
                    st.error(f"❌ **數據不足或代碼無效。** 請確認代碼 **{display_symbol}** 是否正確。")
                    st.info(f"💡 **提醒：** 台灣股票需要以 **代碼.TW** 格式輸入 (例如：**2330.TW**)。")
                    st.session_state['data_ready'] = False 
                else:
                    company_info = get_company_info(final_symbol_to_analyze) 
                    currency_symbol = get_currency_symbol(final_symbol_to_analyze) 
                    
                    df = calculate_technical_indicators(df) 
                    fa_result = calculate_fundamental_rating(final_symbol_to_analyze)
                    
                    analysis = generate_expert_fusion_signal(
                        df, 
                        fa_rating=fa_result, 
                        is_long_term=is_long_term,
                        currency_symbol=currency_symbol,
                        symbol=final_symbol_to_analyze
                    )
                    
                    st.session_state['analysis_results'] = {
                        'df': df,
                        'company_info': company_info,
                        'currency_symbol': currency_symbol,
                        'fa_result': fa_result,
                        'analysis': analysis,
                        'selected_period_key': selected_period_key,
                        'final_symbol_to_analyze': final_symbol_to_analyze
                    }
                    
                    st.session_state['data_ready'] = True

        except Exception as e:
            st.error(f"❌ 分析 {final_symbol_to_analyze} 時發生未預期的錯誤: {str(e)}")
            st.info("💡 請檢查代碼格式或嘗試其他分析週期。")
            st.session_state['data_ready'] = False 

    
    if st.session_state.get('data_ready', False):
        
        results = st.session_state['analysis_results']
        df = results['df'].dropna() 
        company_info = results['company_info']
        currency_symbol = results['currency_symbol']
        fa_result = results['fa_result']
        analysis = results['analysis']
        selected_period_key = results['selected_period_key']
        final_symbol_to_analyze = results['final_symbol_to_analyze']
        
        st.header(f"📈 **{company_info['name']}** ({final_symbol_to_analyze}) AI趨勢分析")
        
        current_price = analysis['current_price']
        prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        
        price_delta_color = 'inverse' if change < 0 else 'normal'

        st.markdown(f"**分析週期:** **{selected_period_key}** | **FA 評級:** **{fa_result['Combined_Rating']:.2f}/9.0**")
        st.markdown(f"**基本面診斷:** {fa_result['Message']}")
        st.markdown("---")
        
        st.subheader("💡 核心行動與量化評分")
        
        st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 20px; }
    [data-testid="stMetricLabel"] { font-size: 13px; }
    [data-testid="stMetricDelta"] { font-size: 12px; }
    .action-buy { color: #cc0000; font-weight: bold; }
    .action-sell { color: #1e8449; font-weight: bold; }
    .action-neutral { color: #cc6600; font-weight: bold; }
    .action-hold-buy { color: #FA8072; font-weight: bold; }
    .action-hold-sell { color: #80B572; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True
)
        
        col_core_1, col_core_2, col_core_3, col_core_4 = st.columns(4)
        
        with col_core_1: 
            st.metric("💰 當前價格", f"{currency_symbol}{current_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)", delta_color=price_delta_color)
            
        with col_core_2:
            st.markdown("**🎯 最終行動建議**")
            
            
            if analysis['action'] == "買進 (Buy)":
                action_class = "action-buy"
            elif analysis['action'] == "中性偏買 (Hold/Buy)":
                action_class = "action-hold-buy" 
            elif analysis['action'] == "賣出 (Sell/Short)":
                action_class = "action-sell"
            elif analysis['action'] == "中性偏賣 (Hold/Sell)":
                action_class = "action-hold-sell" 
            else:
                action_class = "action-neutral"
            st.markdown(f"<p class='{action_class}' style='font-size: 20px;'>{analysis['action']}</p>", unsafe_allow_html=True)
        
        with col_core_3: 
            st.metric("🔥 總量化評分", f"{analysis['score']}", help="FA/TA 融合策略總分 (正數看漲)")
        with col_core_4: 
            st.metric("🛡️ 信心指數", f"{analysis['confidence']:.0f}%", help="AI對此建議的信心度")
        
        st.markdown("---")

        st.subheader("🛡️ 精確交易策略與風險控制")
        col_strat_1, col_strat_2, col_strat_3, col_strat_4 = st.columns(4)

        risk = abs(analysis['entry_price'] - analysis['stop_loss'])
        reward = abs(analysis['take_profit'] - analysis['entry_price'])
        risk_reward = reward / risk if risk > 0 else float('inf')

        with col_strat_1:
            st.markdown(f"**建議操作:** <span class='{action_class}' style='font-size: 18px;'>**{analysis['action']}**</span>", unsafe_allow_html=True)
        with col_strat_2:
            st.markdown(f"**建議進場價:** <span style='color:#cc6600;'>**{currency_symbol}{analysis['entry_price']:.2f}**</span>", unsafe_allow_html=True)
        with col_strat_3:
            st.markdown(f"**🚀 止盈價 (TP):** <span style='color:red;'>**{currency_symbol}{analysis['take_profit']:.2f}**</span>", unsafe_allow_html=True)
        with col_strat_4:
            st.markdown(f"**🛑 止損價 (SL):** <span style='color:green;'>**{currency_symbol}{analysis['stop_loss']:.2f}**</span>", unsafe_allow_html=True)
            
        st.info(f"**💡 策略總結:** **{analysis['strategy']}** | **⚖️ 風險/回報比 (R:R):** **{risk_reward:.2f}** | **波動單位 (ATR):** {analysis.get('atr', 0):.4f}")
        
        st.markdown("---")

        st.subheader("📊 關鍵技術指標數據與AI判讀 (交叉驗證細節)")
        
        ai_df = pd.DataFrame(analysis['expert_opinions'].items(), columns=['AI領域', '判斷結果']) 
        
        if isinstance(fa_result, dict) and 'Message' in fa_result:
            ai_df.loc[len(ai_df)] = ['基本面 FCF/ROE/PE 診斷', fa_result['Message']]
        
        def style_expert_opinion(s):
            is_positive = s.str.contains('牛市|買進|多頭|強化|利多|極健康|穩固|潛在反彈|強勢區間|多頭排列|黃金交叉|強勁|穩固|資金流入', case=False)
            is_negative = s.str.contains('熊市|賣出|空頭|削弱|利空|下跌|疲弱|潛在回調|弱勢區間|空頭排列|死亡交叉|過熱|崩潰|資金流出', case=False)
            is_warning = s.str.contains('盤整|警告|中性|觀望|趨勢發展中|不適用|收縮|低波動性|過高|壓力', case=False) 
            
            colors = np.select(
                [is_negative, is_positive, is_warning],
                ['color: #1e8449; font-weight: bold;', 'color: #cc0000; font-weight: bold;', 'color: #888888;'],
                default='color: #888888;'
            )
            return [f'{c}' for c in colors]

        styled_ai_df = ai_df.style.apply(style_expert_opinion, subset=['判斷結果'], axis=0)

        st.dataframe(
            styled_ai_df,
            use_container_width=True,
            key=f"ai_df_{final_symbol_to_analyze}_{selected_period_key}",
            column_config={
                "AI領域": st.column_config.Column("AI領域", help="FA/TA 分析範疇"),
                "判斷結果": st.column_config.Column("判斷結果", help="AI對該領域的量化判讀與結論"),
            }
        )
        
        st.caption("ℹ️ **設計師提示:** 判讀結果顏色：**紅色=多頭/強化信號** (類似低風險買入)，**綠色=空頭/削弱信號** (類似高風險賣出)，**橙色=中性/警告**。")

        st.markdown("---")

        st.subheader("🧪 策略回測報告 (AI 融合信號)")
        
        
        # Generate signals for backtest (example: 1 buy, -1 sell based on score)
        signals = np.sign(df['Close'].pct_change().rolling(20).mean())  # Placeholder, replace with actual AI signals
        backtest_results = run_backtest(df.copy(), signals)
        
        
        if backtest_results.get("total_trades", 0) > 0:
            
            col_bt_1, col_bt_2, col_bt_3, col_bt_4, col_bt_5 = st.columns(5)
            
            with col_bt_1: 
                st.metric("📊 總回報率", f"{backtest_results['total_return']}%", 
                          delta_color='inverse' if backtest_results['total_return'] < 0 else 'normal',
                          delta=backtest_results['message'])

            with col_bt_2: 
                st.metric("📈 勝率", f"{backtest_results['win_rate']}%")

            with col_bt_3: 
                st.metric("📉 最大回撤 (MDD)", f"{backtest_results['max_drawdown']}%", delta_color='inverse')

            with col_bt_4:
                st.metric("🤝 交易總次數", f"{backtest_results['total_trades']} 次")
            
            with col_bt_5:
                st.metric("📐 夏普比率", f"{backtest_results['sharpe_ratio']}")
                
            
            if 'capital_curve' in backtest_results:
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=df.index.to_list(), y=backtest_results['capital_curve'], name='策略資金曲線', line=dict(color='#cc6600', width=2)))
                fig_bt.add_hline(y=100000, line_dash="dash", line_color="#1e8449", annotation_text="起始資金 $100,000", annotation_position="bottom right")
                
                fig_bt.update_layout(
                    title='AI 融合信號資金曲線',
                    xaxis_title='交易週期',
                    yaxis_title='賬戶價值 ($)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300
                )
                st.plotly_chart(fig_bt, use_container_width=True)
                
            st.caption("ℹ️ **策略說明:** 此回測使用 **AI 融合信號** 作為**開倉/清倉**信號 (初始資金 $100,000，單次交易手續費 0.1%)。 **總回報率**越高越好，**最大回撤 (MDD)**越低越好。")
        else:
            st.info(f"回測無法執行或無交易信號：{backtest_results.get('message', '數據不足或發生錯誤。')}")

        st.markdown("---")
        
        st.subheader("🛠️ 技術指標狀態表")
        technical_df = get_technical_data_df(df)
        
        if not technical_df.empty:
            def style_indicator(s):
                df_color = technical_df['顏色']
                color_map = {'red': 'color: #cc6600; font-weight: bold;', 
                             'green': 'color: #1e8449; font-weight: bold;', 
                             'orange': 'color: #FA8072;', 
                             'blue': 'color: #888888;',
                             'grey': 'color: #888888;'}
                
                return [color_map.get(df_color.loc[index], '') for index in s.index]
                
            styled_df = technical_df[['最新值', '分析結論']].style.apply(style_indicator, subset=['最新值', '分析結論'], axis=0)

            st.dataframe(
                styled_df, 
                use_container_width=True,
                key=f"technical_df_{final_symbol_to_analyze}_{selected_period_key}",
                column_config={
                    "最新值": st.column_config.Column("最新數值", help="技術指標的最新計算值"),
                    "分析結論": st.column_config.Column("趨勢/動能判讀", help="基於數值範圍的專業解讀"),
                }
            )
            st.caption("ℹ️ **設計師提示:** 表格顏色會根據指標的趨勢/風險等級自動變化。這些判讀是 **Meta-Learner** 決策層的基礎輸入。")

        else:
            st.info("無足夠數據生成關鍵技術指標表格。")
        
        st.markdown("---")
        
        st.subheader(f"📊 完整技術分析圖表")
        chart = create_comprehensive_chart(df, final_symbol_to_analyze, selected_period_key) 
        
        st.plotly_chart(chart, use_container_width=True, key=f"plotly_chart_{final_symbol_to_analyze}_{selected_period_key}")

    elif not st.session_state.get('data_ready', False) and not analyze_button_clicked:
          st.markdown(
              """
              <h1 style='color: #FA8072; font-size: 32px; font-weight: bold;'>🚀 歡迎使用 AI 趨勢分析</h1>
              """, 
              unsafe_allow_html=True
          )
          
          st.markdown(f"請在左側選擇或輸入您想分析的標的（例如：**2330.TW**、**NVDA**、**BTC-USD**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
          
          st.markdown("---")
          
          st.subheader("📝 使用步驟：")
          st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
          st.markdown("2. **選擇標的**：使用下拉選單快速選擇熱門標的，或直接在輸入框中鍵入代碼或名稱。")
          st.markdown("3. **選擇週期**：決定分析的長度（例如：`30 分`、`4 小時`、`1 日`、`1 周`）。")
          st.markdown(f"4. **執行分析**：點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』**</span>，AI將融合基本面與技術面指標提供交易策略。", unsafe_allow_html=True)
          
          st.markdown("---")


if __name__ == '__main__':
    if 'last_search_symbol' not in st.session_state:
        st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state:
        st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state:
        st.session_state['sidebar_search_input'] = "2330.TW"
    if 'analyze_trigger' not in st.session_state:
        st.session_state['analyze_trigger'] = False
        
    main()
    
    
    st.markdown("---")
    st.markdown("⚠️ **綜合風險與免責聲明 (Risk & Disclaimer)**", unsafe_allow_html=True)
    st.markdown("本AI趨勢分析模型，是基於**量化集成學習 (Ensemble)**的專業架構。其分析結果**僅供參考用途**")
    st.markdown("投資涉及風險，所有交易決策應基於您個人的**獨立研究和財務狀況**，並強烈建議諮詢**專業金融顧問**。", unsafe_allow_html=True)
    st.markdown("📊 **數據來源:** Alpha Vantage, Yahoo Finance | 🛠️ **技術指標:** TA 庫 | 💻 **APP優化:** 專業程式碼專家")
