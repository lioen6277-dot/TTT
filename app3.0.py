import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import warnings
import re

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 頁面配置與全局設定 (綜合 2.0 & 3.0)
# ==============================================================================

st.set_page_config(
    page_title="AI趨勢分析📈",
    page_icon="🚀",
    layout="wide"
)

# 週期映射 (採用 3.0 的穩定版本)
PERIOD_MAP = {
    "30 分": ("60d", "30m"),
    "4 小時": ("1y", "90m"),  # 3.0 優化：yf 的 4h 常失敗，改用 90m 更穩定
    "1 日": ("5y", "1d"),
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】(採用 2.0 的完整清單，並更新命名格式)
FULL_SYMBOLS_MAP = {
    # ----------------------------------------------------
    # A. 美股核心 (US Stocks)
    # ----------------------------------------------------
    "TSLA": {"name": "特斯拉 (Tesla)", "keywords": ["特斯拉", "電動車", "TSLA", "Tesla"]},
    "NVDA": {"name": "輝達 (Nvidia)", "keywords": ["輝達", "英偉達", "AI", "NVDA", "Nvidia"]},
    "AAPL": {"name": "蘋果 (Apple)", "keywords": ["蘋果", "Apple", "AAPL"]},
    "GOOGL": {"name": "谷歌/Alphabet", "keywords": ["谷歌", "Alphabet", "GOOGL", "GOOG"]},
    "MSFT": {"name": "微軟 (Microsoft)", "keywords": ["微軟", "Microsoft", "MSFT"]},
    "AMZN": {"name": "亞馬遜 (Amazon)", "keywords": ["亞馬遜", "Amazon", "AMZN"]},
    "META": {"name": "Meta/臉書", "keywords": ["臉書", "Meta", "FB", "META"]},
    "NFLX": {"name": "網飛 (Netflix)", "keywords": ["網飛", "Netflix", "NFLX"]},
    "ADBE": {"name": "Adobe", "keywords": ["Adobe", "ADBE"]},
    "CRM": {"name": "Salesforce", "keywords": ["Salesforce", "CRM"]},
    "ORCL": {"name": "甲骨文 (Oracle)", "keywords": ["甲骨文", "Oracle", "ORCL"]},
    "COST": {"name": "好市多 (Costco)", "keywords": ["好市多", "Costco", "COST"]},
    "JPM": {"name": "摩根大通 (JPMorgan)", "keywords": ["摩根大通", "JPMorgan", "JPM"]},
    "V": {"name": "Visa", "keywords": ["Visa", "V"]},
    "WMT": {"name": "沃爾瑪 (Walmart)", "keywords": ["沃爾瑪", "Walmart", "WMT"]},
    "PG": {"name": "寶潔 (P&G)", "keywords": ["寶潔", "P&G", "PG"]},
    "KO": {"name": "可口可樂 (CocaCola)", "keywords": ["可口可樂", "CocaCola", "KO"]},
    "PEP": {"name": "百事 (Pepsi)", "keywords": ["百事", "Pepsi", "PEP"]},
    "MCD": {"name": "麥當勞 (McDonalds)", "keywords": ["麥當勞", "McDonalds", "MCD"]},
    "QCOM": {"name": "高通 (Qualcomm)", "keywords": ["高通", "Qualcomm", "QCOM"]},
    "INTC": {"name": "英特爾 (Intel)", "keywords": ["英特爾", "Intel", "INTC"]},
    "AMD": {"name": "超微 (AMD)", "keywords": ["超微", "AMD"]},
    "LLY": {"name": "禮來 (Eli Lilly)", "keywords": ["禮來", "EliLilly", "LLY"]},
    "UNH": {"name": "聯合健康 (UnitedHealth)", "keywords": ["聯合健康", "UNH"]},
    "HD": {"name": "家得寶 (Home Depot)", "keywords": ["家得寶", "HomeDepot", "HD"]},
    "CAT": {"name": "開拓重工 (Caterpillar)", "keywords": ["開拓重工", "Caterpillar", "CAT"]},
    # B. 美股指數/ETF (US Indices/ETFs)
    "^GSPC": {"name": "S&P 500 指數", "keywords": ["標普", "S&P500", "^GSPC", "SPX"]},
    "^IXIC": {"name": "NASDAQ 綜合指數", "keywords": ["納斯達克", "NASDAQ", "^IXIC"]},
    "^DJI": {"name": "道瓊工業指數", "keywords": ["道瓊", "DowJones", "^DJI"]},
    "SPY": {"name": "SPDR 標普500 ETF", "keywords": ["SPY", "標普ETF"]},
    "QQQ": {"name": "Invesco QQQ Trust", "keywords": ["QQQ", "納斯達克ETF"]},
    "VOO": {"name": "Vanguard 標普500 ETF", "keywords": ["VOO", "Vanguard"]},
    # ----------------------------------------------------
    # C. 台灣市場 (TW Stocks/ETFs/Indices)
    # ----------------------------------------------------
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
    # ----------------------------------------------------
    # D. 加密貨幣 (Crypto)
    # ----------------------------------------------------
    "BTC-USD": {"name": "比特幣 (Bitcoin)", "keywords": ["比特幣", "BTC", "bitcoin", "BTC-USDT"]},
    "ETH-USD": {"name": "以太坊 (Ethereum)", "keywords": ["以太坊", "ETH", "ethereum", "ETH-USDT"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL", "SOL-USDT"]},
    "BNB-USD": {"name": "幣安幣 (BNB)", "keywords": ["幣安幣", "BNB", "BNB-USDT"]},
    "DOGE-USD": {"name": "狗狗幣 (Dogecoin)", "keywords": ["狗狗幣", "DOGE", "DOGE-USDT"]},
    "XRP-USD": {"name": "瑞波幣 (XRP)", "keywords": ["瑞波幣", "XRP", "XRP-USDT"]},
    "ADA-USD": {"name": "Cardano", "keywords": ["Cardano", "ADA", "ADA-USDT"]},
    "AVAX-USD": {"name": "Avalanche", "keywords": ["Avalanche", "AVAX", "AVAX-USDT"]},
    "DOT-USD": {"name": "Polkadot", "keywords": ["Polkadot", "DOT", "DOT-USDT"]},
    "LINK-USD": {"name": "Chainlink", "keywords": ["Chainlink", "LINK", "LINK-USDT"]},
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

# ==============================================================================
# 2. 輔助函式定義 (數據獲取與預處理)
# ==============================================================================

def get_symbol_from_query(query: str) -> str:
    """ 🎯 綜合 2.0 & 3.0 的代碼解析函數 """
    query = query.strip()
    query_upper = query.upper()
    for code, data in FULL_SYMBOLS_MAP.items():
        if query_upper == code: return code
        if any(query_upper == kw.upper() for kw in data["keywords"]): return code
    for code, data in FULL_SYMBOLS_MAP.items():
        if query == data["name"]: return code
    if re.fullmatch(r'\d{4,6}', query) and not any(ext in query_upper for ext in ['.TW', '.HK', '.SS', '-USD']):
        tw_code = f"{query}.TW"
        return tw_code
    return query

@st.cache_data(ttl=3600, show_spinner="正在從 Yahoo Finance 獲取數據...")
def get_stock_data(symbol, period, interval):
    """ 採用 3.0 的數據獲取邏輯，更穩定 """
    try:
        ticker = yf.Ticker(symbol)
        # 3.0 優化: auto_adjust=True 自動調整股價
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        df.columns = [col.capitalize() for col in df.columns]
        df.index.name = 'Date'
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[~df.index.duplicated(keep='first')]
        # 3.0 優化: 移除 iloc[:-1] 以保留最新數據點
        if df.empty: return pd.DataFrame()
        return df
    except Exception as e:
        return pd.DataFrame()

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
        return {"name": name, "category": "未分類", "currency": currency}
    except Exception:
        return {"name": symbol, "category": "未分類", "currency": "USD"}

@st.cache_data(ttl=3600)
def get_currency_symbol(symbol):
    currency_code = get_company_info(symbol).get('currency', 'USD')
    if currency_code == 'TWD': return 'NT$'
    elif currency_code == 'USD': return '$'
    else: return currency_code + ' '

# ==============================================================================
# 3. 核心分析函式 (綜合 2.0 & 3.0)
# ==============================================================================

def calculate_technical_indicators(df):
    """ 採用 3.0 的技術指標計算，增加動態窗口、OBV、Volume MA """
    data_len = len(df)
    win_200 = min(data_len, 200)
    win_50 = min(data_len, 50)
    win_20 = min(data_len, 20)
    win_17 = min(data_len, 17)
    win_10 = min(data_len, 10)
    win_9 = min(data_len, 9)

    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=win_10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=win_50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=win_200)

    macd_instance = ta.trend.MACD(df['Close'], window_fast=8, window_slow=win_17, window_sign=win_9)
    df['MACD_Line'] = macd_instance.macd()
    df['MACD_Signal'] = macd_instance.macd_signal()
    df['MACD_Hist'] = macd_instance.macd_diff() # 3.0 使用 MACD_Hist

    df['RSI'] = ta.momentum.rsi(df['Close'], window=win_9)
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'], window=win_20, window_dev=2)
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'], window=win_20, window_dev=2)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=win_9)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=win_9)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=win_20)

    # 3.0 新增: 成交量指標
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Volume_MA_20'] = df['Volume'].rolling(window=win_20).mean()

    return df

@st.cache_data(ttl=3600)
def get_chips_and_news_analysis(symbol):
    """ 3.0 新增功能: 獲取籌碼面和消息面數據 """
    try:
        ticker = yf.Ticker(symbol)
        inst_holders = ticker.institutional_holders
        inst_hold_pct = 0
        if inst_holders is not None and not inst_holders.empty:
            inst_hold_pct = inst_holders.iloc[0, 0] if isinstance(inst_holders.iloc[0, 0], (int, float)) else 0
        news = ticker.news
        news_summary = "近期無相關新聞"
        if news:
            headlines = [f"- **{item.get('type', '新聞')}**: {item['title']}" for item in news[:5]]
            news_summary = "\n".join(headlines)
        return {
            "inst_hold_pct": inst_hold_pct,
            "news_summary": news_summary
        }
    except Exception:
        return { "inst_hold_pct": 0, "news_summary": "無法獲取新聞數據" }

@st.cache_data(ttl=3600)
def calculate_advanced_fundamental_rating(symbol):
    """ 採用 3.0 的先進基本面評分模型 (總分7分) """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if info.get('quoteType') in ['INDEX', 'CRYPTOCURRENCY', 'ETF']:
            return {"score": 0, "summary": "指數、加密貨幣或ETF不適用基本面分析。", "details": {}}

        score, details = 0, {}
        roe = info.get('returnOnEquity')
        if roe and roe > 0.15: score += 2; details['✅ ROE > 15%'] = f"{roe:.2%}"
        else: details['❌ ROE < 15%'] = f"{roe:.2%}" if roe is not None else "N/A"

        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None and debt_to_equity < 50: score += 2; details['✅ 負債權益比 < 50%'] = f"{debt_to_equity/100:.2%}"
        else: details['❌ 負債權益比 > 50%'] = f"{debt_to_equity/100:.2%}" if debt_to_equity is not None else "N/A"

        revenue_growth = info.get('revenueGrowth')
        if revenue_growth and revenue_growth > 0.1: score += 1; details['✅ 營收年增 > 10%'] = f"{revenue_growth:.2%}"
        else: details['❌ 營收年增 < 10%'] = f"{revenue_growth:.2%}" if revenue_growth is not None else "N/A"

        pe = info.get('trailingPE')
        peg = info.get('pegRatio')
        if pe is not None and 0 < pe < 15: score += 1; details['✅ 本益比(P/E) < 15'] = f"{pe:.2f}"
        else: details['⚠️ 本益比(P/E) > 15'] = f"{pe:.2f}" if pe else "N/A"
        if peg is not None and 0 < peg < 1: score += 1; details['✅ PEG < 1'] = f"{peg:.2f}"
        else: details['⚠️ PEG > 1'] = f"{peg:.2f}" if peg else "N/A"

        if score >= 6: summary = "頂級優異：公司在獲利、財務、成長性上表現強勁，且估值合理。"
        elif score >= 4: summary = "良好穩健：公司基本面穩固，但在某些方面有待加強。"
        else: summary = "中性警示：需留意公司的財務風險、獲利能力不足或估值偏高的問題。"
        return {"score": score, "summary": summary, "details": details}
    except Exception:
        return {"score": 0, "summary": "無法獲取或計算基本面數據。", "details": {}}

def generate_ai_fusion_signal(df, fa_rating, chips_news_data, is_long_term, currency_symbol):
    """ 採用 3.0 的 AI 四維融合訊號生成器 (技術+基本+籌碼+成交量) """
    df_signal = df.dropna(subset=['Close', 'EMA_10', 'EMA_50', 'MACD_Hist', 'RSI', 'ATR']).copy()
    if df_signal.empty or len(df_signal) < 2:
        return { 'action': '數據不足', 'score': 0, 'confidence': 50, 'strategy': '無法評估', 'entry_price': 0, 'take_profit': 0, 'stop_loss': 0, 'current_price': df['Close'].iloc[-1] if not df.empty else 0, 'ai_opinions': {}, 'atr': 0 }

    last_row, prev_row = df_signal.iloc[-1], df_signal.iloc[-2]
    current_price, atr = last_row['Close'], last_row.get('ATR', 0)
    ai_opinions = {}
    
    WEIGHTS = {'LongTerm': {'TA': 0.8, 'FA': 1.6, 'Chips': 1.2, 'Volume': 0.4}, 'ShortTerm': {'TA': 1.6, 'FA': 0.8, 'Chips': 0.4, 'Volume': 1.2}}
    weights = WEIGHTS['LongTerm'] if is_long_term else WEIGHTS['ShortTerm']
    
    ta_score = 0
    if last_row['EMA_10'] > last_row['EMA_50'] > last_row.get('EMA_200', float('inf')): ta_score += 2; ai_opinions['MA 趨勢'] = '✅ 強多頭排列 (10>50>200)'
    elif last_row['EMA_10'] < last_row['EMA_50'] < last_row.get('EMA_200', float('-inf')): ta_score -= 2; ai_opinions['MA 趨勢'] = '❌ 強空頭排列 (10<50<200)'
    else: ai_opinions['MA 趨勢'] = '⚠️ 中性盤整'

    if last_row['RSI'] > 70: ta_score -= 1; ai_opinions['RSI 動能'] = '⚠️ 超買區域 (>70)，潛在回調'
    elif last_row['RSI'] < 30: ta_score += 1; ai_opinions['RSI 動能'] = '✅ 超賣區域 (<30)，潛在反彈'
    elif last_row['RSI'] > 50: ta_score += 1; ai_opinions['RSI 動能'] = '✅ 多頭區間 (>50)'
    else: ta_score -= 1; ai_opinions['RSI 動能'] = '❌ 空頭區間 (<50)'

    if last_row['MACD_Hist'] > 0 and last_row['MACD_Hist'] > prev_row['MACD_Hist']: ta_score += 2; ai_opinions['MACD 動能'] = '✅ 多頭動能增強 (柱狀圖>0)'
    elif last_row['MACD_Hist'] < 0 and last_row['MACD_Hist'] < prev_row['MACD_Hist']: ta_score -= 2; ai_opinions['MACD 動能'] = '❌ 空頭動能增強 (柱狀圖<0)'
    else: ai_opinions['MACD 動能'] = '⚠️ 動能盤整'
    
    ta_multiplier = 1.3 if last_row['ADX'] > 25 else 0.8
    ai_opinions['ADX 趨勢強度'] = f"✅ 強趨勢確認 (>{last_row['ADX']:.1f})" if last_row['ADX'] > 25 else f"⚠️ 盤整趨勢 (<{last_row['ADX']:.1f})"
    ta_score *= ta_multiplier
    
    fa_score = ((fa_rating.get('score', 0) / 7.0) * 6.0) - 3.0
    
    chips_score, volume_score = 0, 0
    inst_hold_pct = chips_news_data.get('inst_hold_pct', 0) * 100
    if inst_hold_pct > 70: chips_score = 1.5; ai_opinions['籌碼分析'] = f'✅ 法人高度集中 ({inst_hold_pct:.1f}%)'
    elif inst_hold_pct > 40: chips_score = 0.5; ai_opinions['籌碼分析'] = f'✅ 法人持股穩定 ({inst_hold_pct:.1f}%)'
    else: chips_score = -0.5; ai_opinions['籌碼分析'] = f'⚠️ 籌碼較分散 ({inst_hold_pct:.1f}%)'

    is_high_volume = last_row['Volume'] > (last_row.get('Volume_MA_20', 0) * 1.5)
    if is_high_volume and last_row['Close'] > prev_row['Close']: volume_score = 1.5; ai_opinions['成交量分析'] = '✅ 價漲量增，趨勢強勁'
    elif is_high_volume and last_row['Close'] < prev_row['Close']: volume_score = -1.5; ai_opinions['成交量分析'] = '❌ 價跌量增，空頭壓力'
    else: ai_opinions['成交量分析'] = '⚠️ 量能中性或萎縮'
    
    total_score = (ta_score * weights['TA'] + fa_score * weights['FA'] + chips_score * weights['Chips'] + volume_score * weights['Volume']) / 5.0
    confidence = min(100, max(40, abs(total_score) * 15 + 40))

    if total_score > 3.5: action = '買進 (Strong Buy)'
    elif total_score > 1.5: action = '中性偏買 (Hold/Buy)'
    elif total_score < -3.5: action = '賣出 (Strong Sell/Short)'
    elif total_score < -1.5: action = '中性偏賣 (Hold/Sell)'
    else: action = '中性 (Neutral)'

    entry_price = current_price
    take_profit = current_price + atr * 2.0 if total_score > 0 else current_price - atr * 2.0
    stop_loss = current_price - atr * 1.0 if total_score > 0 else current_price + atr * 1.0
    strategy = f'基於TA/FA/籌碼/量能的四維融合模型 (長期模式: {is_long_term})'

    return {
        'current_price': current_price, 'action': action, 'score': total_score, 'confidence': confidence,
        'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss,
        'strategy': strategy, 'atr': atr, 'ai_opinions': ai_opinions
    }

def get_technical_data_df(df):
    """ 採用 3.0 的技術指標狀態表生成器，更穩健 """
    if df.empty or len(df) < 51: return pd.DataFrame()
    df_clean = df.dropna().copy()
    if df_clean.empty: return pd.DataFrame()
    last_row = df_clean.iloc[-1]
    prev_row = df_clean.iloc[-2] if len(df_clean) >= 2 else last_row
    
    indicators = {
        '價格 vs. EMA 10/50/200': last_row.get('Close', np.nan), 'RSI (9) 動能': last_row.get('RSI', np.nan),
        'MACD (8/17/9) 柱狀圖': last_row.get('MACD_Hist', np.nan), 'ADX (9) 趨勢強度': last_row.get('ADX', np.nan),
        'ATR (9) 波動性': last_row.get('ATR', np.nan), '布林通道 (BB: 20/2)': last_row.get('Close', np.nan)
    }
    data = []
    for name, value in indicators.items():
        conclusion, color = "", "grey"
        if pd.isna(value):
            data.append([name, "N/A", "數據不足", "blue"]); continue
        if 'EMA' in name:
            emas = [last_row.get(f'EMA_{w}', np.nan) for w in [10, 50, 200]]
            if not any(pd.isna(e) for e in emas) and emas[0] > emas[1] > emas[2]: conclusion, color = "**強多頭排列**", "red"
            elif not any(pd.isna(e) for e in emas) and emas[0] < emas[1] < emas[2]: conclusion, color = "**強空頭排列**", "green"
            else: conclusion, color = "中性盤整", "blue"
        elif 'RSI' in name:
            if value > 70: conclusion, color = "警告：超買", "green"
            elif value < 30: conclusion, color = "強化：超賣", "red"
            elif value > 50: conclusion, color = "多頭", "red"
            else: conclusion, color = "空頭", "green"
        elif 'MACD' in name:
            if value > 0 and value > prev_row.get('MACD_Hist', 0): conclusion, color = "強化：多頭動能增強", "red"
            elif value < 0 and value < prev_row.get('MACD_Hist', 0): conclusion, color = "削弱：空頭動能增強", "green"
            else: conclusion, color = "中性：動能盤整", "orange"
        elif 'ADX' in name:
            conclusion, color = ("強趨勢", "orange") if value >= 25 else ("盤整", "blue")
        elif 'ATR' in name:
            conclusion, color = "中性：正常波動", "blue"
        elif '布林通道' in name:
            if value > last_row.get('BB_High', float('inf')): conclusion, color = "警告：觸及上軌", "red"
            elif value < last_row.get('BB_Low', float('-inf')): conclusion, color = "強化：觸及下軌", "green"
            else: conclusion, color = "中性：軌道內", "blue"
        data.append([name, f"{value:,.4f}", conclusion, color])
    return pd.DataFrame(data, columns=['指標名稱', '最新值', '分析結論', '顏色']).set_index('指標名稱')

def run_backtest(df, initial_capital=100000, commission_rate=0.001):
    """ 採用 3.0 的回測邏輯，對數據處理更嚴謹 """
    if df.empty or len(df.dropna(subset=['SMA_20', 'EMA_50'])) < 51:
        return {"total_return": 0, "win_rate": 0, "max_drawdown": 0, "total_trades": 0, "message": "數據不足無法回測"}
    data = df.dropna(subset=['SMA_20', 'EMA_50']).copy()
    
    buy_signal = (data['SMA_20'] > data['EMA_50']) & (data['SMA_20'].shift(1) <= data['EMA_50'].shift(1))
    sell_signal = (data['SMA_20'] < data['EMA_50']) & (data['SMA_20'].shift(1) >= data['EMA_50'].shift(1))
    data['Signal'] = 0
    data.loc[buy_signal, 'Signal'] = 1
    data.loc[sell_signal, 'Signal'] = -1
    
    position, capital, trades, buy_price, capital_curve = 0, initial_capital, [], 0, []
    for i in range(len(data)):
        current_capital = capital * (data['Close'].iloc[i] / buy_price) if position == 1 else capital
        capital_curve.append(current_capital)
        if data['Signal'].iloc[i] == 1 and position == 0:
            position, buy_price = 1, data['Close'].iloc[i]
            capital = current_capital * (1 - commission_rate)
        elif data['Signal'].iloc[i] == -1 and position == 1:
            profit = (data['Close'].iloc[i] - buy_price) / buy_price
            trades.append(1 if profit > 0 else 0)
            capital = current_capital * (1 - commission_rate) * (1 + profit)
            position, buy_price = 0, 0
    if position == 1:
        profit = (data['Close'].iloc[-1] - buy_price) / buy_price
        trades.append(1 if profit > 0 else 0)
        capital_curve[-1] = capital_curve[-1] * (1 + profit)
        capital = capital_curve[-1]

    total_return = (capital / initial_capital - 1) * 100
    win_rate = (sum(trades) / len(trades)) * 100 if trades else 0
    capital_s = pd.Series(capital_curve, index=data.index[:len(capital_curve)])
    max_drawdown = (capital_s / capital_s.cummax() - 1).min() * 100 if not capital_s.empty else 0
    
    return {
        "total_return": round(total_return, 2), "win_rate": round(win_rate, 2),
        "max_drawdown": round(abs(max_drawdown), 2), "total_trades": len(trades),
        "message": f"回測區間 {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}",
        "capital_curve": capital_s
    }

def create_comprehensive_chart(df, symbol, period_key):
    """ 採用 3.0 的四合一圖表 (價格+成交量, MACD, RSI/ADX, OBV) """
    df_clean = df.dropna().copy()
    if df_clean.empty: return go.Figure()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        specs=[[{"secondary_y": True}], [{}], [{}], [{}]])

    fig.add_trace(go.Candlestick(x=df_clean.index, open=df_clean['Open'], high=df_clean['High'], low=df_clean['Low'], close=df_clean['Close'], name='K線'), row=1, col=1)
    if 'EMA_10' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_10'], line=dict(color='orange', width=1), name='EMA 10'), row=1, col=1)
    if 'EMA_50' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_50'], line=dict(color='cyan', width=1.5), name='EMA 50'), row=1, col=1)
    if 'EMA_200' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_200'], line=dict(color='purple', width=2, dash='dot'), name='EMA 200'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['Volume'], marker_color='grey', name='成交量', opacity=0.3), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="價格", row=1, col=1); fig.update_yaxes(title_text="成交量", secondary_y=True, row=1, col=1, showgrid=False)

    macd_colors = np.where(df_clean.get('MACD_Hist', pd.Series()) >= 0, '#cc0000', '#1e8449')
    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean.get('MACD_Hist', pd.Series()), marker_color=macd_colors, name='MACD Hist'), row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1, zeroline=True)

    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.get('RSI', pd.Series()), line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.get('ADX', pd.Series()), line=dict(color='#cc6600', width=1.5, dash='dot'), name='ADX'), row=3, col=1)
    fig.update_yaxes(title_text="RSI/ADX", range=[0, 100], row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)

    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.get('OBV', pd.Series()), line=dict(color='green', width=1.5), name='OBV'), row=4, col=1)
    fig.update_yaxes(title_text="OBV", row=4, col=1)

    fig.update_layout(title_text=f"AI 融合分析圖表 - {symbol} ({period_key})", height=900, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ==============================================================================
# 4. Streamlit 主應用程式邏輯
# ==============================================================================
def main():
    # 綜合 2.0 & 3.0 的 CSS 風格
    st.markdown("""
        <style>
        [data-testid="stSidebar"] .stButton button {
            color: #FA8072 !important; background-color: rgba(255, 255, 255, 0.1) !important;
            border-color: #FA8072 !important; border-width: 1px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-radius: 8px;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            color: #E9967A !important; background-color: rgba(250, 128, 114, 0.15)  !important;
        }
        h1, h2, h3 { color: #cc6600; }
        [data-testid="stMetricValue"] { font-size: 20px; }
        .action-buy { color: #cc0000; font-weight: bold; } .action-sell { color: #1e8449; font-weight: bold; }
        .action-neutral { color: #cc6600; font-weight: bold; } .action-hold-buy { color: #FA8072; font-weight: bold; }
        .action-hold-sell { color: #80B572; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

    # --- 側邊欄 UI (採用 3.0 簡潔佈局) ---
    st.sidebar.title("🚀 AI 趨勢分析")
    st.sidebar.markdown("---")
    
    selected_category = st.sidebar.selectbox('1. 選擇資產類別', list(CATEGORY_HOT_OPTIONS.keys()), key='category_selector')
    hot_options_map = CATEGORY_HOT_OPTIONS.get(selected_category, {})

    def sync_text_input_from_selection():
        selected_code = hot_options_map.get(st.session_state.hot_target_selector, "")
        st.session_state.sidebar_search_input = selected_code

    st.sidebar.selectbox('2. 選擇熱門標的', list(hot_options_map.keys()), key='hot_target_selector', on_change=sync_text_input_from_selection)
    
    search_input = st.sidebar.text_input('...或手動輸入代碼/名稱:', key='sidebar_search_input')
    st.sidebar.markdown("---")
    
    selected_period_key = st.sidebar.selectbox('3. 選擇分析週期', list(PERIOD_MAP.keys()), index=2)
    is_long_term = st.sidebar.checkbox('長期投資者模式', value=False, help="勾選後將更側重基本面和籌碼面分析")
    st.sidebar.markdown("---")
    
    analyze_button_clicked = st.sidebar.button('📊 執行AI分析', use_container_width=True)

    # --- 主分析流程 (採用 3.0 邏輯) ---
    if analyze_button_clicked:
        final_symbol = get_symbol_from_query(search_input or hot_options_map.get(st.session_state.hot_target_selector))
        if not final_symbol:
            st.error("請輸入或選擇一個有效的股票代碼。")
            return

        with st.spinner(f"🔍 正在啟動AI模型，分析 **{final_symbol}** 的數據..."):
            yf_period, yf_interval = PERIOD_MAP[selected_period_key]
            df = get_stock_data(final_symbol, yf_period, yf_interval)
            
            if df.empty or len(df) < 51:
                st.error(f"❌ **數據不足或代碼無效：** {final_symbol}。回測與核心指標至少需要51個數據點。")
                st.session_state['data_ready'] = False
            else:
                df_with_ta = calculate_technical_indicators(df)
                st.session_state['analysis_results'] = {
                    'df': df_with_ta,
                    'company_info': get_company_info(final_symbol),
                    'currency_symbol': get_currency_symbol(final_symbol),
                    'fa_result': calculate_advanced_fundamental_rating(final_symbol),
                    'chips_news_data': get_chips_and_news_analysis(final_symbol),
                    'selected_period_key': selected_period_key,
                    'final_symbol_to_analyze': final_symbol,
                    'is_long_term': is_long_term
                }
                st.session_state['data_ready'] = True
    
    # --- 結果呈現區 (採用 3.0 的新順序與佈局) ---
    if st.session_state.get('data_ready', False):
        res = st.session_state['analysis_results']
        analysis = generate_ai_fusion_signal(res['df'], res['fa_result'], res['chips_news_data'], res['is_long_term'], res['currency_symbol'])
        
        st.header(f"📈 **{res['company_info']['name']}** ({res['final_symbol_to_analyze']}) AI趨勢分析")
        st.markdown(f"**分析週期:** {res['selected_period_key']} | **基本面(FA)評級:** **{res['fa_result'].get('score', 0):.1f}/7.0**")
        st.markdown(f"**基本面診斷:** {res['fa_result'].get('summary', 'N/A')}")
        st.markdown("---")
        
        st.subheader("💡 核心行動與量化評分")
        price = analysis['current_price']
        prev_close = res['df']['Close'].iloc[-2] if len(res['df']) > 1 else price
        change, change_pct = price - prev_close, (price - prev_close) / prev_close * 100 if prev_close else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 當前價格", f"{res['currency_symbol']}{price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)", delta_color='inverse' if change < 0 else 'normal')
        
        action_class = "action-neutral"
        if "買進" in analysis['action']: action_class = "action-buy" if "偏" not in analysis['action'] else "action-hold-buy"
        elif "賣出" in analysis['action']: action_class = "action-sell" if "偏" not in analysis['action'] else "action-hold-sell"
        col2.markdown(f"**🎯 最終行動建議**\n<p class='{action_class}' style='font-size: 20px;'>{analysis['action']}</p>", unsafe_allow_html=True)
        
        col3.metric("🔥 總量化評分", f"{analysis['score']:.2f}", help="四維融合模型總分")
        col4.metric("🛡️ 信心指數", f"{analysis['confidence']:.0f}%", help="AI對此建議的信心度")
        st.markdown("---")
        
        st.subheader("🛡️ 交易策略參考 (基於 ATR 風險/報酬)")
        col_risk_1, col_risk_2, col_risk_3 = st.columns(3)
        col_risk_1.metric("🛒 建議入場價", f"{res['currency_symbol']}{analysis['entry_price']:,.2f}")
        col_risk_2.metric("🟢 建議止盈 (2x ATR)", f"{res['currency_symbol']}{analysis['take_profit']:,.2f}")
        col_risk_3.metric("🔴 建議止損 (1x ATR)", f"{res['currency_symbol']}{analysis['stop_loss']:,.2f}")
        st.caption(f"波動性 (ATR): {res['currency_symbol']}{analysis['atr']:,.2f}。採用 2:1 風報比策略。")
        st.markdown("---")

        st.subheader("📊 AI 四維分析判讀")
        opinions_data = list(analysis['ai_opinions'].items())
        if 'details' in res['fa_result']:
            for key, val in res['fa_result']['details'].items(): opinions_data.append([f"基本面 - {key}", str(val)])
        ai_df = pd.DataFrame(opinions_data, columns=['AI分析維度', '判斷結果'])
        st.dataframe(ai_df.style.apply(lambda s: ['color: #1e8449' if any(w in x for w in ['❌', '空頭', '削弱']) else 'color: #cc0000' if any(w in x for w in ['✅', '多頭', '強化']) else 'color: #cc6600' for x in s], subset=['判斷結果']), use_container_width=True)
        st.markdown("---")

        st.subheader("🧪 策略回測報告 (SMA 20/EMA 50 交叉)")
        backtest_results = run_backtest(res['df'].copy())
        if backtest_results.get("total_trades", 0) > 0:
            col_bt_1, col_bt_2, col_bt_3, col_bt_4 = st.columns(4)
            col_bt_1.metric("📊 總回報率", f"{backtest_results['total_return']}%", delta=backtest_results['message'])
            col_bt_2.metric("📈 勝率", f"{backtest_results['win_rate']}%")
            col_bt_3.metric("📉 最大回撤", f"{backtest_results['max_drawdown']}%")
            col_bt_4.metric("🤝 交易次數", f"{backtest_results['total_trades']} 次")
            
            if 'capital_curve' in backtest_results and not backtest_results['capital_curve'].empty:
                fig_bt = go.Figure(go.Scatter(x=backtest_results['capital_curve'].index, y=backtest_results['capital_curve'], name='策略資金曲線', line=dict(color='#cc6600')))
                fig_bt.update_layout(title='SMA 20/EMA 50 交叉策略資金曲線', height=300)
                st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.warning(f"回測無法執行：{backtest_results.get('message', '發生錯誤')}")
        st.markdown("---")

        st.subheader(f"📊 完整技術分析圖表")
        st.plotly_chart(create_comprehensive_chart(res['df'], res['final_symbol_to_analyze'], res['selected_period_key']), use_container_width=True)

    else:
        st.markdown("<h1 style='color: #FA8072;'>🚀 歡迎使用 AI 趨勢分析</h1>", unsafe_allow_html=True)
        st.markdown("請在左側選擇或輸入您想分析的標的，然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕。", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇或輸入標的**：使用下拉選單或直接鍵入代碼。")
        st.markdown("3. **選擇分析週期**：決定分析的時間長度。")
        st.markdown("4. **(可選) 長期投資者模式**：勾選後將提高基本面與籌碼面的分析權重。")
        st.markdown("5. **執行分析**：點擊按鈕，AI將融合四維度指標提供交易策略。")

    st.markdown("---")
    st.caption("⚠️ **免責聲明:** 本AI分析結果僅供參考，不構成任何投資建議。所有交易決策應基於您的獨立研究，投資涉及風險。")
    st.caption("📊 **數據來源:** Yahoo Finance | 🛠️ **技術指標:** TA 庫 | 💻 **APP優化:** 專業程式碼專家")

if __name__ == '__main__':
    # 初始化 session state
    if 'data_ready' not in st.session_state: st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state: st.session_state['sidebar_search_input'] = "NVDA"
    main()
