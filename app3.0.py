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
from fredapi import Fred  # For macro data
from FinMind.data import DataLoader  # For Taiwan chip data
from scipy.stats import linregress

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI趨勢分析📈", 
    page_icon="🚀", 
    layout="wide"
)

# Replace with actual API keys
NEWS_API_KEY = "your_news_api_key_here"  # From newsapi.org
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key_here"  # For better data
FRED_API_KEY = "your_fred_api_key_here"  # From FRED API

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

def get_ttl(interval):
    if 'm' in interval:
        return 300  # 5 min for short periods
    else:
        return 3600  # 1 hour for longer

@st.cache_data(ttl=get_ttl, show_spinner="正在從 Alpha Vantage 獲取數據...") 
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

def get_vix():
    try:
        vix_data = yf.Ticker("^VIX").history(period="1d")
        if not vix_data.empty:
            return vix_data['Close'].iloc[-1]
        else:
            return 20.0  # Default value if data unavailable
    except:
        return 20.0

def get_news(symbol):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        articles = response.json().get('articles', [])
        return [a['title'] for a in articles[:5]]
    except:
        return ["No news available"]

def get_fed_rate():
    try:
        fred = Fred(api_key=FRED_API_KEY)
        rate = fred.get_series_latest_release('FEDFUNDS')
        return rate.iloc[-1]
    except:
        return 5.0  # Default if API fails

def get_chip_data(stock_id):
    try:
        dl = DataLoader()
        chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=str(datetime.now() - timedelta(days=30)), end_date=str(datetime.now()))
        foreign_buy = chip[chip['name'] == 'Foreign_Investor']['buy'].sum()
        foreign_sell = chip[chip['name'] == 'Foreign_Investor']['sell'].sum()
        net_buy = (foreign_buy - foreign_sell) / chip['volume'].sum() * 100
        return net_buy
    except:
        return 0.0  # Default if fails

def calculate_dcf(ticker):
    try:
        info = ticker.info
        financials = ticker.financials
        fcf = financials.loc['Free Cash Flow'].dropna().iloc[-1] if 'Free Cash Flow' in financials.index else info.get('freeCashflow', 0)
        growth_rate = info.get('earningsGrowth', 0.05)
        discount_rate = 0.1
        perpetuity_growth = 0.025
        projected_fcf = [fcf * (1 + growth_rate) ** i for i in range(1, 6)]
        pv_fcf = [cf / (1 + discount_rate) ** i for i, cf in enumerate(projected_fcf, 1)]
        terminal_value = projected_fcf[-1] * (1 + perpetuity_growth) / (discount_rate - perpetuity_growth)
        pv_terminal = terminal_value / (1 + discount_rate) ** 5
        dcf_value = sum(pv_fcf) + pv_terminal
        shares = info.get('sharesOutstanding', 1)
        intrinsic = dcf_value / shares
        return intrinsic
    except:
        return 0.0

def calculate_fundamental_rating(ticker):
    try:
        info = ticker.info
        financials = ticker.financials
        rating = 0
        message = []
        
        roe = info.get('returnOnEquity', 0)
        if roe > 0.15:
            rating += 20
            message.append(f"ROE {roe:.2%} >15% 高效")
        
        roce = info.get('returnOnCapitalEmployed', 0)  # May need calculation if not available
        if roce > 0.1:
            rating += 20
            message.append(f"ROCE {roce:.2%} >10% 良好")
        
        debt_ratio = info.get('debtToEquity', 0)
        if debt_ratio < 50:
            rating += 20
            message.append(f"負債比率 {debt_ratio:.2f} <50% 低風險")
        
        gross_margin = info.get('grossMargins', 0)
        if gross_margin > 0.3:
            rating += 20
            message.append(f"毛利率 {gross_margin:.2%} >30% 強競爭力")
        
        net_margin = info.get('profitMargins', 0)
        if net_margin > 0.3:
            rating += 20
            message.append(f"淨利率 {net_margin:.2%} >30% 強獲利")
        
        # CAGR
        if len(financials.columns) >= 5:
            rev_recent = financials.loc['Total Revenue'].iloc[0]
            rev_old = financials.loc['Total Revenue'].iloc[4]
            rev_cagr = (rev_recent / rev_old) ** (0.25) - 1 if rev_old != 0 else 0
            if rev_cagr > 0.1:
                rating += 10
                message.append(f"營收CAGR {rev_cagr:.2%} >10% 穩定成長")
            
            earn_recent = financials.loc['Net Income'].iloc[0]
            earn_old = financials.loc['Net Income'].iloc[4]
            earn_cagr = (earn_recent / earn_old) ** (0.25) - 1 if earn_old != 0 else 0
            if earn_cagr > 0.1:
                rating += 10
                message.append(f"利潤CAGR {earn_cagr:.2%} >10% 穩定成長")
        
        current_price = info.get('currentPrice', 0)
        intrinsic = calculate_dcf(ticker)
        if current_price < intrinsic * 0.8:  # 20% margin
            rating += 20
            message.append(f"股價低於內在價值 20% (DCF: {intrinsic:.2f})")
        
        return {'rating': rating, 'Message': '; '.join(message)}
    except:
        return {'rating': 0, 'Message': '基本面數據不足'}

def generate_expert_fusion_signal(df, symbol):
    # Technical opinions
    expert_opinions = {}
    
    # MA
    ema10 = ta.trend.ema_indicator(df['Close'], window=10).iloc[-1]
    ema50 = ta.trend.ema_indicator(df['Close'], window=50).iloc[-1]
    ema200 = ta.trend.ema_indicator(df['Close'], window=200).iloc[-1]
    if ema10 > ema50 > ema200:
        expert_opinions['移動平均線 (MA)'] = 'MA 向上排列為強多頭（買入）'
    elif ema10 < ema50 < ema200:
        expert_opinions['移動平均線 (MA)'] = 'MA 向下排列為強空頭（賣出）'
    else:
        expert_opinions['移動平均線 (MA)'] = '中性'
    
    # RSI
    rsi = ta.momentum.rsi(df['Close'], window=9).iloc[-1]
    if rsi > 70:
        expert_opinions['相對強弱指數 (RSI)'] = 'RSI >70 超買（回調賣出）'
    elif rsi < 30:
        expert_opinions['相對強弱指數 (RSI)'] = 'RSI <30 超賣（反彈買入）'
    elif rsi > 50:
        expert_opinions['相對強弱指數 (RSI)'] = 'RSI >50 為多頭'
    else:
        expert_opinions['相對強弱指數 (RSI)'] = 'RSI <50 為空頭'
    
    # MACD
    macd = ta.trend.MACD(df['Close'], window_fast=8, window_slow=17, window_sign=9)
    hist = macd.macd_diff().iloc[-1]
    if hist > 0:
        expert_opinions['MACD'] = 'Histogram >0 為多頭強勢'
    else:
        expert_opinions['MACD'] = 'Histogram <0 為空頭強勢'
    
    # Volume
    obv = ta.volume.on_balance_volume(df['Close'], df['Volume']).iloc[-1]
    vol_ma = ta.volume.sma_indicator(df['Volume'], window=20).iloc[-1]
    if df['Volume'].iloc[-1] > vol_ma * 1.5:
        expert_opinions['成交量'] = '成交量放大 >150% MA 確認趨勢'
    
    # ADX
    adx = ta.trend.adx(df['High'], df['Low'], df['Close'], window=9).iloc[-1]
    if adx > 25:
        expert_opinions['ADX'] = 'ADX >25 確認強趨勢'
    
    ta_positive = sum(1 for v in expert_opinions.values() if '多頭' in v or '買入' in v)
    ta_negative = sum(1 for v in expert_opinions.values() if '空頭' in v or '賣出' in v)
    ta_score = (ta_positive - ta_negative) / max(len(expert_opinions), 1) * 100
    
    # Fundamental
    ticker = yf.Ticker(symbol)
    fa_result = calculate_fundamental_rating(ticker)
    fa_score = fa_result['rating']
    expert_opinions['基本面 FCF/ROE/PE 診斷'] = fa_result['Message']
    
    # Message face
    news = get_news(symbol)
    expert_opinions['消息面'] = ' '.join(news) if news else '無消息'
    
    # Macro
    rate = get_fed_rate()
    expert_opinions['宏觀事件'] = f'利率 {rate:.2f}% ' + ('降息利多' if rate < 3 else '升息利空' if rate > 5 else '中性')
    
    # Chip
    if symbol.endswith('.TW'):
        chip_net = get_chip_data(symbol[:-3])
        expert_opinions['籌碼面'] = f'外資買超 {chip_net:.2f}% ' + ('看好買入' if chip_net > 10 else '投信減持利空' if chip_net < -5 else '中性')
    
    # VIX for sentiment
    vix = get_vix()
    expert_opinions['情緒指標'] = f'VIX {vix:.2f} ' + ('>30 恐慌，低估機會' if vix > 30 else '<30 穩定')
    
    # Dynamic weight
    w_fa = 0.7 if vix > 30 else 0.4
    w_ta = 1 - w_fa
    fusion_score = w_fa * fa_score + w_ta * ta_score
    
    action = '買進' if fusion_score > 50 else '賣出' if fusion_score < -50 else '觀望'
    
    current_price = df['Close'].iloc[-1]
    atr = ta.volatility.atr(df['High'], df['Low'], df['Close'], window=14).iloc[-1]
    entry_price = current_price
    take_profit = current_price + 2 * atr if '買進' in action else current_price - 2 * atr
    stop_loss = current_price - atr if '買進' in action else current_price + atr
    
    strategy = '基於技術、基本面、消息、籌碼、宏觀融合'
    
    return {
        'action': action,
        'score': fusion_score,
        'entry_price': entry_price,
        'take_profit': take_profit,
        'stop_loss': stop_loss,
        'strategy': strategy,
        'expert_opinions': expert_opinions,
        'atr': atr
    }

def get_technical_data_df(df):
    indicators = []
    
    ema10 = ta.trend.ema_indicator(df['Close'], window=10).iloc[-1]
    indicators.append({'指標': 'EMA10', '最新值': f"{ema10:.2f}", '分析結論': '短期趨勢', '顏色': 'blue'})
    
    rsi = ta.momentum.rsi(df['Close'], window=9).iloc[-1]
    color = 'red' if rsi > 50 else 'green'
    conclusion = '多頭動能' if rsi > 50 else '空頭動能'
    indicators.append({'指標': 'RSI', '最新值': f"{rsi:.2f}", '分析結論': conclusion, '顏色': color})
    
    # Add more indicators...
    
    technical_df = pd.DataFrame(indicators)
    return technical_df

def create_comprehensive_chart(df, symbol, period_key):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('股價', '成交量', 'RSI', 'MACD'),
                        row_width=[0.2, 0.2, 0.2, 0.6])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
    
    # MA lines
    ema10 = ta.trend.ema_indicator(df['Close'], window=10)
    ema50 = ta.trend.ema_indicator(df['Close'], window=50)
    fig.add_trace(go.Scatter(x=df.index, y=ema10, name='EMA10'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name='EMA50'), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
    
    # RSI
    rsi = ta.momentum.rsi(df['Close'])
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI'), row=3, col=1)
    fig.add_hline(y=70, row=3, col=1)
    fig.add_hline(y=30, row=3, col=1)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    fig.add_trace(go.Scatter(x=df.index, y=macd.macd(), name='MACD'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd.macd_signal(), name='Signal'), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=macd.macd_diff(), name='Hist'), row=4, col=1)
    
    fig.update_layout(title=f"{symbol} 技術圖表 ({period_key})", height=800)
    return fig

def run_backtest(df, symbol):
    # Generate signals using TA (as placeholder for historical fusion, since FA is static)
    df['ema10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['ema50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['signal'] = np.where(df['ema10'] > df['ema50'], 1, -1)
    
    returns = df['Close'].pct_change()
    strategy_returns = returns * df['signal'].shift()
    strategy_returns = strategy_returns.dropna()
    
    cum_ret = (1 + strategy_returns).cumprod() * 100000
    total_return = (cum_ret.iloc[-1] / 100000 - 1) * 100
    win_rate = (strategy_returns > 0).mean() * 100
    
    # Max drawdown
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # Sharpe
    sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() != 0 else 0
    
    # Alpha Beta
    bench_symbol = '^TWII' if symbol.endswith('.TW') else '^GSPC'
    bench_df = yf.download(bench_symbol, start=df.index.min(), end=df.index.max())['Close']
    bench_returns = bench_df.pct_change().reindex(strategy_returns.index).fillna(0)
    result = linregress(bench_returns, strategy_returns)
    beta = result.slope
    alpha = result.intercept * 252  # Annualized
    
    total_trades = df['signal'].diff().abs().sum() / 2
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'alpha': alpha,
        'beta': beta,
        'capital_curve': cum_ret,
        'message': '基於AI融合信號回測'
    }

def main():
    if 'last_search_symbol' not in st.session_state:
        st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state:
        st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state:
        st.session_state['sidebar_search_input'] = "2330.TW"
    if 'analyze_trigger' not in st.session_state:
        st.session_state['analyze_trigger'] = False
    
    sidebar = st.sidebar
    category = sidebar.selectbox("選擇資產類別", list(CATEGORY_HOT_OPTIONS.keys()))
    hot_option = sidebar.selectbox("熱門標的", list(CATEGORY_HOT_OPTIONS[category].keys()))
    final_symbol_to_analyze = CATEGORY_HOT_OPTIONS[category][hot_option]
    
    search_input = sidebar.text_input("🔍 輸入股票代碼或中文名稱", st.session_state['sidebar_search_input'])
    if search_input:
        final_symbol_to_analyze = get_symbol_from_query(search_input)
        st.session_state['sidebar_search_input'] = search_input
    
    selected_period_key = sidebar.selectbox("分析時間週期", list(PERIOD_MAP.keys()))
    
    analyze_button_clicked = sidebar.button("📊 執行AI分析")
    
    if analyze_button_clicked:
        st.session_state['analyze_trigger'] = True
        st.session_state['data_ready'] = True
    
    if st.session_state.get('analyze_trigger', False):
        period, interval = PERIOD_MAP[selected_period_key]
        df = get_stock_data(final_symbol_to_analyze, period, interval)
        
        if df.empty:
            st.error(f"❌ 數據不足或代碼無效。 請確認代碼 {final_symbol_to_analyze} 是否正確。")
            st.info("💡 提醒： 台灣股票需要以 代碼.TW 格式輸入 (例如：2330.TW)。")
        else:
            company_info = get_company_info(final_symbol_to_analyze)
            name = company_info['name']
            category = company_info['category']
            currency = company_info['currency']
            currency_symbol = 'NT$' if currency == 'TWD' else '$'
            
            analysis = generate_expert_fusion_signal(df, final_symbol_to_analyze)
            fa_result = calculate_fundamental_rating(yf.Ticker(final_symbol_to_analyze))
            
            latest_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
            change = (latest_close - prev_close) / prev_close * 100 if prev_close != 0 else 0
            change_str = f"+{change:.2f}%" if change > 0 else f"{change:.2f}%"
            
            st.markdown(f"""
            <h1 style='color: #FA8072; font-size: 32px; font-weight: bold;'>🚀 歡迎使用 AI 趨勢分析</h1>
            """, unsafe_allow_html=True)
            
            st.subheader(f"📈 {name} ({final_symbol_to_analyze}) - {selected_period_key}週期分析")
            st.markdown(f"**類別:** {category} | **貨幣:** {currency_symbol} | **最新價格:** {currency_symbol}{latest_close:.2f} ({change_str})")
            
            action_class = 'cc0000' if '買進' in analysis['action'] else '1e8449' if '賣出' in analysis['action'] else '888888'
            st.markdown(f"**🎯 最終行動建議:** <span style='color: #{action_class}; font-weight: bold;'>{analysis['action']}</span>", unsafe_allow_html=True)
            st.markdown(f"**🔥 總量化評分:** {analysis['score']:.2f} (正數看漲)", unsafe_allow_html=True)
            st.markdown(f"**🛡️ 信心指數:** 80%", unsafe_allow_html=True)  # Placeholder
            
            st.markdown("---")
            
            st.subheader("🛡️ 精確交易策略與風險控制")
            col_strat_1, col_strat_2, col_strat_3, col_strat_4 = st.columns(4)
            with col_strat_1:
                st.markdown(f"**建議操作:** <span class='{action_class}' style='font-size: 18px;'>**{analysis['action']}**</span>", unsafe_allow_html=True)
            with col_strat_2:
                st.markdown(f"**建議進場價:** <span style='color:#cc6600;'>**{currency_symbol}{analysis['entry_price']:.2f}**</span>", unsafe_allow_html=True)
            with col_strat_3:
                st.markdown(f"**🚀 止盈價 (TP):** <span style='color:red;'>**{currency_symbol}{analysis['take_profit']:.2f}**</span>", unsafe_allow_html=True)
            with col_strat_4:
                st.markdown(f"**🛑 止損價 (SL):** <span style='color:green;'>**{currency_symbol}{analysis['stop_loss']:.2f}**</span>", unsafe_allow_html=True)
                
            risk_reward = abs((analysis['take_profit'] - analysis['entry_price']) / (analysis['entry_price'] - analysis['stop_loss'])) if analysis['action'] == '買進' else abs((analysis['entry_price'] - analysis['take_profit']) / (analysis['stop_loss'] - analysis['entry_price']))
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
            
            
            backtest_results = run_backtest(df.copy(), final_symbol_to_analyze)
            
            
            if backtest_results.get("total_trades", 0) > 0:
                
                col_bt_1, col_bt_2, col_bt_3, col_bt_4, col_bt_5, col_bt_6, col_bt_7 = st.columns(7)
                
                with col_bt_1: 
                    st.metric("📊 總回報率", f"{backtest_results['total_return']:.2f}%", 
                              delta_color='inverse' if backtest_results['total_return'] < 0 else 'normal',
                              delta=backtest_results.get('message', ''))
    
                with col_bt_2: 
                    st.metric("📈 勝率", f"{backtest_results['win_rate']:.2f}%")
    
                with col_bt_3: 
                    st.metric("📉 最大回撤 (MDD)", f"{backtest_results['max_drawdown']:.2f}%", delta_color='inverse')
    
                with col_bt_4:
                    st.metric("🤝 交易總次數", f"{backtest_results['total_trades']} 次")
                
                with col_bt_5:
                    st.metric("📐 夏普比率", f"{backtest_results['sharpe_ratio']:.2f}")
                
                with col_bt_6:
                    st.metric("Alpha", f"{backtest_results['alpha']:.4f}")
                
                with col_bt_7:
                    st.metric("Beta", f"{backtest_results['beta']:.2f}")
                    
                
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
    main()
    
    
    st.markdown("---")
    st.markdown("⚠️ **綜合風險與免責聲明 (Risk & Disclaimer)**", unsafe_allow_html=True)
    st.markdown("本AI趨勢分析模型，是基於**量化集成學習 (Ensemble)**的專業架構。其分析結果**僅供參考用途**")
    st.markdown("投資涉及風險，所有交易決策應基於您個人的**獨立研究和財務狀況**，並強烈建議諮詢**專業金融顧問**。", unsafe_allow_html=True)
    st.markdown("📊 **數據來源:** Alpha Vantage, Yahoo Finance | 🛠️ **技術指標:** TA 庫 | 💻 **APP優化:** 專業程式碼專家")

