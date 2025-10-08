# AI Trend Analysis Tool (Expert Consolidated Version)

import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 頁面配置與全局設定 (Page Configuration & Global Settings)
# ==============================================================================

st.set_page_config(
    page_title="AI趨勢分析📈",
    page_icon="🚀",
    layout="wide"
)

# 週期映射 (Period Mapping): (YFinance Period, YFinance Interval)
PERIOD_MAP = {
    "30 分": ("60d", "30m"),
    "4 小時": ("1y", "90m"), # yfinance 4h interval often fails, use 90m instead
    "1 日": ("5y", "1d"),
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】(擴充整合版)
# An extensive and consolidated list of assets
FULL_SYMBOLS_MAP = {
    # 美股/ETF/指數 (US Stocks/ETFs/Indices)
    "AAPL": {"name": "蘋果 (Apple)", "keywords": ["蘋果", "Apple", "AAPL"]},
    "MSFT": {"name": "微軟 (Microsoft)", "keywords": ["微軟", "Microsoft", "MSFT", "雲端", "AI"]},
    "GOOGL": {"name": "谷歌/Alphabet A股 (Google A)", "keywords": ["谷歌", "Alphabet", "GOOGL", "GOOG"]},
    "AMZN": {"name": "亞馬遜 (Amazon)", "keywords": ["亞馬遜", "Amazon", "AMZN", "電商"]},
    "NVDA": {"name": "輝達 (Nvidia)", "keywords": ["輝達", "英偉達", "AI", "NVDA", "Nvidia", "GPU", "半導體"]},
    "TSLA": {"name": "特斯拉 (Tesla)", "keywords": ["特斯拉", "電動車", "TSLA", "Tesla"]},
    "META": {"name": "Meta/臉書 (Facebook)", "keywords": ["臉書", "Meta", "FB", "META", "Facebook"]},
    "BRK-B": {"name": "波克夏海瑟威 B (Berkshire Hathaway)", "keywords": ["波克夏", "巴菲特", "BRKB", "保險", "投資"]},
    "JPM": {"name": "摩根大通 (JPMorgan Chase)", "keywords": ["摩根大通", "JPMorgan", "JPM", "金融股"]},
    "JNJ": {"name": "嬌生 (Johnson & Johnson)", "keywords": ["嬌生", "Johnson&Johnson", "JNJ", "醫療保健"]},
    "V": {"name": "Visa", "keywords": ["Visa", "V"]},
    "PG": {"name": "寶潔 (Procter & Gamble)", "keywords": ["寶潔", "P&G", "PG"]},
    "HD": {"name": "家得寶 (Home Depot)", "keywords": ["家得寶", "HomeDepot", "HD"]},
    "MA": {"name": "萬事達卡 (Mastercard)", "keywords": ["萬事達卡", "Mastercard", "MA", "支付"]},
    "UNH": {"name": "聯合健康 (UnitedHealth Group)", "keywords": ["聯合健康", "UNH", "醫療保健"]},
    "BAC": {"name": "美國銀行 (Bank of America)", "keywords": ["美國銀行", "BankOfAmerica", "BAC", "金融股"]},
    "ADBE": {"name": "Adobe", "keywords": ["Adobe", "ADBE"]},
    "CRM": {"name": "Salesforce", "keywords": ["Salesforce", "CRM", "雲端", "SaaS"]},
    "NFLX": {"name": "網飛 (Netflix)", "keywords": ["網飛", "Netflix", "NFLX"]},
    "KO": {"name": "可口可樂 (Coca-Cola)", "keywords": ["可口可樂", "CocaCola", "KO"]},
    "COST": {"name": "好市多 (Costco)", "keywords": ["好市多", "Costco", "COST"]},
    "AMD": {"name": "超微 (Advanced Micro Devices)", "keywords": ["超微", "AMD", "半導體"]},
    "INTC": {"name": "英特爾 (Intel)", "keywords": ["英特爾", "Intel", "INTC", "半導體"]},
    "QCOM": {"name": "高通 (Qualcomm)", "keywords": ["高通", "Qualcomm", "QCOM", "半導體"]},
    "PYPL": {"name": "PayPal", "keywords": ["PayPal", "PYPL", "金融科技", "Fintech"]},
    "^GSPC": {"name": "S&P 500 指數", "keywords": ["標普", "S&P500", "^GSPC", "SPX", "指數"]},
    "^IXIC": {"name": "NASDAQ 綜合指數", "keywords": ["納斯達克", "NASDAQ", "^IXIC", "指數", "科技股"]},
    "^DJI": {"name": "道瓊工業指數 (Dow Jones Industrial Average)", "keywords": ["道瓊", "DowJones", "^DJI", "指數"]},
    "SPY": {"name": "SPDR 標普500 ETF", "keywords": ["SPY", "標普ETF"]},
    "QQQ": {"name": "Invesco QQQ Trust", "keywords": ["QQQ", "納斯達克ETF", "科技股ETF"]},
    "VOO": {"name": "Vanguard 標普500 ETF", "keywords": ["VOO", "Vanguard"]},
    "VTI": {"name": "Vanguard整體股市ETF (Total Market)", "keywords": ["VTI", "整體股市", "TotalMarket"]},
    "ARKK": {"name": "方舟創新ETF (ARK Innovation)", "keywords": ["ARKK", "CathieWood", "創新ETF", "木頭姐"]},
    "GLD": {"name": "SPDR黃金ETF (Gold ETF)", "keywords": ["GLD", "黃金ETF", "避險資產"]},
    "TLT": {"name": "iShares 20年期以上公債ETF (Treasury Bond)", "keywords": ["TLT", "美債", "公債ETF"]},
    "SMH": {"name": "VanEck Vectors半導體ETF", "keywords": ["SMH", "半導體ETF", "晶片股"]},
    "^VIX": {"name": "恐慌指數 (VIX)", "keywords": ["VIX", "恐慌指數", "波動率指數"]},

    # 台股/ETF/指數 (Taiwan Stocks/ETFs/Indices)
    "2330.TW": {"name": "台積電", "keywords": ["台積電", "2330", "TSMC", "晶圓", "半導體"]},
    "2317.TW": {"name": "鴻海", "keywords": ["鴻海", "2317", "Foxconn"]},
    "2454.TW": {"name": "聯發科", "keywords": ["聯發科", "2454", "MediaTek"]},
    "2308.TW": {"name": "台達電", "keywords": ["台達電", "2308", "Delta"]},
    "2382.TW": {"name": "廣達", "keywords": ["廣達", "2382", "AI伺服器"]},
    "3231.TW": {"name": "緯創", "keywords": ["緯創", "3231", "AI伺服器"]},
    "3017.TW": {"name": "奇鋐", "keywords": ["奇鋐", "3017", "散熱"]},
    "6669.TW": {"name": "緯穎", "keywords": ["緯穎", "6669", "AI伺服器", "資料中心"]},
    "2881.TW": {"name": "富邦金", "keywords": ["富邦金", "2881", "金融股"]},
    "2882.TW": {"name": "國泰金", "keywords": ["國泰金", "2882", "金融股"]},
    "2891.TW": {"name": "中信金", "keywords": ["中信金", "2891", "金融股"]},
    "2603.TW": {"name": "長榮", "keywords": ["長榮", "2603", "航運"]},
    "2609.TW": {"name": "陽明", "keywords": ["陽明", "2609", "航運"]},
    "2615.TW": {"name": "萬海", "keywords": ["萬海", "2615", "航運"]},
    "0050.TW": {"name": "元大台灣50", "keywords": ["台灣50", "0050", "台灣五十", "ETF"]},
    "0056.TW": {"name": "元大高股息", "keywords": ["高股息", "0056", "ETF"]},
    "00878.TW": {"name": "國泰永續高股息", "keywords": ["00878", "國泰永續", "ETF"]},
    "00929.TW": {"name": "復華台灣科技優息", "keywords": ["00929", "科技優息", "月配息", "ETF"]},
    "00940.TW": {"name": "元大臺灣價值高息", "keywords": ["00940", "臺灣價值高息", "ETF"]},
    "^TWII": {"name": "台股加權指數", "keywords": ["台股指數", "加權指數", "^TWII", "指數"]},

    # 加密貨幣 (Cryptocurrencies)
    "BTC-USD": {"name": "比特幣 (Bitcoin)", "keywords": ["比特幣", "BTC", "bitcoin", "BTC-USDT", "加密貨幣之王"]},
    "ETH-USD": {"name": "以太坊 (Ethereum)", "keywords": ["以太坊", "ETH", "ethereum", "ETH-USDT", "智能合約"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL", "SOL-USDT"]},
    "BNB-USD": {"name": "幣安幣 (Binance Coin)", "keywords": ["幣安幣", "BNB", "BNB-USDT", "交易所幣"]},
    "XRP-USD": {"name": "瑞波幣 (Ripple)", "keywords": ["瑞波幣", "XRP", "XRP-USDT"]},
    "DOGE-USD": {"name": "狗狗幣 (Dogecoin)", "keywords": ["狗狗幣", "DOGE", "DOGE-USDT", "迷因幣"]},
    "ADA-USD": {"name": "Cardano", "keywords": ["Cardano", "ADA", "ADA-USDT"]},
    "AVAX-USD": {"name": "Avalanche", "keywords": ["Avalanche", "AVAX", "AVAX-USDT"]},
    "LINK-USD": {"name": "Chainlink", "keywords": ["Chainlink", "LINK", "LINK-USDT", "預言機"]},
    "DOT-USD": {"name": "Polkadot", "keywords": ["Polkadot", "DOT", "DOT-USDT"]},
}

# 建立第二層選擇器映射 (Category Mapping for Selectors)
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
# 2. 核心數據與分析函式 (Core Data & Analysis Functions)
# ==============================================================================

def get_symbol_from_query(query: str) -> str:
    """ 智能代碼解析函數，支援代碼、名稱、關鍵字搜尋 """
    query = query.strip()
    query_upper = query.upper()

    # 優先匹配代碼和關鍵字
    for code, data in FULL_SYMBOLS_MAP.items():
        if query_upper == code.upper(): return code
        if any(query_upper == kw.upper() for kw in data["keywords"]): return code

    # 其次匹配完整名稱
    for code, data in FULL_SYMBOLS_MAP.items():
        if query == data["name"]: return code

    # 處理純數字輸入（推斷為台股）
    if re.fullmatch(r'\d{4,6}', query) and not any(ext in query_upper for ext in ['.TW', '.HK', '.SS', '-USD']):
        tw_code = f"{query}.TW"
        return tw_code
    return query

@st.cache_data(ttl=300, show_spinner="🔄 正在從 Yahoo Finance 獲取最新市場數據...")
def get_stock_data(symbol, period, interval):
    """
    獲取股票數據，並進行標準化清洗。
    - 使用 auto_adjust=True 自動處理股息和股票分割。
    - 移除最後一根可能未收盤的K線，確保指標計算穩定性。
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df.empty: return pd.DataFrame()

        df.columns = [col.capitalize() for col in df.columns]
        df.index.name = 'Date'
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[~df.index.duplicated(keep='first')]

        if len(df) > 1:
            df = df.iloc[:-1]

        if df.empty: return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_company_info(symbol):
    """ 從自定義映射或 yfinance 獲取公司基本資訊 """
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
    except Exception:
        return {"name": symbol, "category": "未分類", "currency": "USD"}

@st.cache_data
def get_currency_symbol(symbol):
    """ 獲取對應貨幣符號 """
    currency_code = get_company_info(symbol).get('currency', 'USD')
    if currency_code == 'TWD': return 'NT$'
    elif currency_code == 'USD': return '$'
    else: return currency_code + ' '

def calculate_technical_indicators(df):
    """
    依據《程式碼基本原則》中的進階設定計算所有需要的技術指標。
    - MA: EMA 10, 50, 200
    - MACD: 快線8, 慢線17, 信號線9
    - RSI: 週期9
    - ATR: 週期9
    - ADX: 週期9
    - Volume: OBV, Volume MA 20
    """
    # 趨勢指標 (Trend Indicators)
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=9)

    # 動能指標 (Momentum Indicators)
    macd_instance = ta.trend.MACD(df['Close'], window_fast=8, window_slow=17, window_sign=9)
    df['MACD_Line'] = macd_instance.macd()
    df['MACD_Signal'] = macd_instance.macd_signal()
    df['MACD_Hist'] = macd_instance.macd_diff()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=9)
    
    # 波動率指標 (Volatility Indicators)
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=9)

    # 成交量指標 (Volume Indicators)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    # 回測用指標 (Backtesting Indicator)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    
    return df

@st.cache_data(ttl=3600)
def get_chips_and_news_analysis(symbol):
    """ 獲取籌碼面 (法人持股) 和消息面 (最新新聞) 數據 """
    try:
        ticker = yf.Ticker(symbol)
        inst_hold_pct = 0
        if ticker.institutional_holders is not None and not ticker.institutional_holders.empty:
            value = ticker.institutional_holders.iloc[0, 0]
            if isinstance(value, (int, float)): inst_hold_pct = value

        news = ticker.news
        news_summary = "近期無相關新聞"
        if news:
            headlines = [f"- {item['title']}" for item in news[:3]]
            news_summary = "\n".join(headlines)

        return {"inst_hold_pct": inst_hold_pct, "news_summary": news_summary}
    except Exception:
        return {"inst_hold_pct": 0, "news_summary": "無法獲取新聞數據"}

@st.cache_data(ttl=3600)
def calculate_fundamental_rating(symbol):
    """
    完全依據《程式碼基本原則》計算基本面評分。
    - 獲利能力 (ROE > 15%)
    - 財務健康 (負債權益比 < 50%)
    - 成長性 (營收年增 > 10%)
    - 估值 (P/E < 15, PEG < 1)
    """
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

        pe, peg = info.get('trailingPE'), info.get('pegRatio')
        if pe and 0 < pe < 15: score += 1; details['✅ 本益比(P/E) < 15'] = f"{pe:.2f}"
        else: details['⚠️ 本益比(P/E) > 15'] = f"{pe:.2f}" if pe else "N/A"
        if peg and 0 < peg < 1: score += 1; details['✅ PEG < 1'] = f"{peg:.2f}"
        else: details['⚠️ PEG > 1'] = f"{peg:.2f}" if peg else "N/A"

        if score >= 6: summary = "頂級優異：公司在獲利、財務、成長性上表現強勁，且估值合理。"
        elif score >= 4: summary = "良好穩健：公司基本面穩固，但在某些方面有待加強。"
        else: summary = "中性警示：需留意公司的財務風險、獲利能力不足或估值偏高的問題。"

        return {"score": score, "summary": summary, "details": details}
    except Exception:
        return {"score": 0, "summary": "無法獲取或計算基本面數據。", "details": {}}

def generate_ai_fusion_signal(df, fa_rating, chips_news_data, is_long_term, currency_symbol):
    """
    AI四維融合訊號生成器，並整合精確交易策略與風險控制。
    - 權重動態調整 (長期 vs 短期)
    - 風險回報比 (R:R) 設為 2:1
    """
    if df.empty or len(df) < 2:
        return {'action': '數據不足', 'score': 0, 'confidence': 50, 'strategy': '無法評估', 'entry_price': 0, 'take_profit': 0, 'stop_loss': 0, 'current_price': 0, 'ai_opinions': {}, 'atr': 0}

    last_row = df.iloc[-1]
    current_price = last_row['Close']
    atr = last_row.get('ATR', 0)
    ai_opinions = {}
    
    # 1. 技術面評分 (TA Score)
    ta_score = 0
    if last_row['EMA_10'] > last_row['EMA_50'] > last_row['EMA_200']: ta_score += 2; ai_opinions['MA 趨勢'] = '✅ 強多頭排列 (10>50>200)'
    elif last_row['EMA_10'] < last_row['EMA_50'] < last_row['EMA_200']: ta_score -= 2; ai_opinions['MA 趨勢'] = '❌ 強空頭排列 (10<50<200)'
    else: ai_opinions['MA 趨勢'] = '⚠️ 中性盤整'

    if last_row['RSI'] > 70: ta_score -= 1; ai_opinions['RSI 動能'] = '⚠️ 超買區域 (>70)，潛在回調'
    elif last_row['RSI'] < 30: ta_score += 1; ai_opinions['RSI 動能'] = '✅ 超賣區域 (<30)，潛在反彈'
    elif last_row['RSI'] > 50: ta_score += 1; ai_opinions['RSI 動能'] = '✅ 多頭區間 (>50)'
    else: ta_score -= 1; ai_opinions['RSI 動能'] = '❌ 空頭區間 (<50)'

    if last_row['MACD_Hist'] > 0: ta_score += 1; ai_opinions['MACD 動能'] = '✅ 多頭動能 (柱狀圖>0)'
    else: ta_score -= 1; ai_opinions['MACD 動能'] = '❌ 空頭動能 (柱狀圖<0)'
    
    if last_row['ADX'] > 25: ta_score *= 1.2; ai_opinions['ADX 趨勢強度'] = f'✅ 強趨勢確認 ({last_row["ADX"]:.1f})'
    else: ta_score *= 0.8; ai_opinions['ADX 趨勢強度'] = f'⚠️ 盤整趨勢 ({last_row["ADX"]:.1f})'

    # 2. 基本面評分 (FA Score)
    fa_score = ((fa_rating.get('score', 0) / 7.0) - 0.5) * 6.0

    # 3. 籌碼與成交量評分 (Chips & Volume Score)
    chips_score, volume_score = 0, 0
    inst_hold_pct = chips_news_data.get('inst_hold_pct', 0) * 100
    if inst_hold_pct > 70: chips_score = 1.0; ai_opinions['籌碼分析'] = f'✅ 法人高度集中 ({inst_hold_pct:.1f}%)'
    elif inst_hold_pct > 40: chips_score = 0.5; ai_opinions['籌碼分析'] = f'✅ 法人持股穩定 ({inst_hold_pct:.1f}%)'
    else: ai_opinions['籌碼分析'] = f'⚠️ 籌碼較分散 ({inst_hold_pct:.1f}%)'
        
    if last_row['Close'] > last_row['Open'] and last_row['Volume'] > last_row['Volume_MA_20']: volume_score = 1.0; ai_opinions['成交量分析'] = '✅ 價漲量增，多頭健康'
    elif last_row['Close'] < last_row['Open'] and last_row['Volume'] > last_row['Volume_MA_20']: volume_score = -1.0; ai_opinions['成交量分析'] = '❌ 價跌量增，空頭壓力'
    else: ai_opinions['成交量分析'] = '⚠️ 量能萎縮或價量背離'
    
    # 4. 融合總分 (Fusion Score)
    if is_long_term: total_score = ta_score * 0.4 + fa_score * 1.2 + chips_score * 1.2 + volume_score * 0.2
    else: total_score = ta_score * 1.2 + fa_score * 0.6 + chips_score * 0.6 + volume_score * 0.6

    # 5. 判斷行動與信心指數 (Action & Confidence)
    confidence = min(100, max(0, 50 + (total_score * 5)))
    if total_score > 4: action = '買進 (Buy)'
    elif total_score > 1.5: action = '中性偏買 (Hold/Buy)'
    elif total_score < -4: action = '賣出 (Sell/Short)'
    elif total_score < -1.5: action = '中性偏賣 (Hold/Sell)'
    else: action = '中性 (Neutral)'

    # 6. 精確交易策略 (Precise Trading Strategy)
    risk_multiple, reward_ratio = 1.5, 2.0
    entry_buffer = atr * 0.2
    price_format = ".4f" if current_price < 100 and not currency_symbol == 'NT$' else ".2f"
    
    if "買進" in action or "偏買" in action:
        entry = current_price - entry_buffer
        stop_loss = entry - (atr * risk_multiple)
        take_profit = entry + (atr * risk_multiple * reward_ratio)
        strategy_desc = f"基於{action}信號，建議在 **{currency_symbol}{entry:{price_format}} (± {entry_buffer:,.4f})** 範圍內尋找支撐或等待回調進場。"
    elif "賣出" in action or "偏賣" in action:
        entry = current_price + entry_buffer
        stop_loss = entry + (atr * risk_multiple)
        take_profit = entry - (atr * risk_multiple * reward_ratio)
        strategy_desc = f"基於{action}信號，建議在 **{currency_symbol}{entry:{price_format}} (± {entry_buffer:,.4f})** 範圍內尋找阻力或等待反彈後進場。"
    else:
        entry, stop_loss, take_profit = current_price, current_price - atr, current_price + atr
        strategy_desc = "市場信號混亂，建議等待趨勢明朗或在區間內操作。"

    return {
        'current_price': current_price, 'action': action, 'score': total_score, 'confidence': confidence,
        'entry_price': entry, 'take_profit': take_profit, 'stop_loss': stop_loss,
        'strategy': strategy_desc, 'atr': atr, 'ai_opinions': ai_opinions
    }

def get_technical_data_df(df):
    """ 生成技術指標狀態表 """
    if df.empty or len(df) < 200: return pd.DataFrame()
    df_clean = df.dropna().copy()
    if df_clean.empty: return pd.DataFrame()
    last_row, prev_row = df_clean.iloc[-1], df_clean.iloc[-2]
    
    indicators = {'價格 vs. EMA 10/50/200': last_row['Close'], 'RSI (9) 動能': last_row['RSI'], 'MACD (8/17/9) 柱狀圖': last_row['MACD_Hist'], 'ADX (9) 趨勢強度': last_row['ADX'], 'ATR (9) 波動性': last_row['ATR'], '布林通道 (BB: 20/2)': last_row['Close']}
    data = []
    
    for name, value in indicators.items():
        conclusion, color = "", "grey"
        if 'EMA' in name:
            if last_row['EMA_10'] > last_row['EMA_50'] > last_row['EMA_200']: conclusion, color = f"**強多頭排列**", "red"
            elif last_row['EMA_10'] < last_row['EMA_50'] < last_row['EMA_200']: conclusion, color = f"**強空頭排列**", "green"
            elif last_row['Close'] > last_row['EMA_50']: conclusion, color = f"中長線偏多", "orange"
            else: conclusion, color = "中性：趨勢發展中", "blue"
        elif 'RSI' in name:
            if value > 70: conclusion, color = "警告：超買", "green"
            elif value < 30: conclusion, color = "強化：超賣", "red"
            elif value > 50: conclusion, color = "多頭：RSI > 50", "red"
            else: conclusion, color = "空頭：RSI < 50", "green"
        elif 'MACD' in name:
            if value > 0 and value > prev_row['MACD_Hist']: conclusion, color = "強化：多頭動能增強", "red"
            elif value < 0 and value < prev_row['MACD_Hist']: conclusion, color = "削弱：空頭動能增強", "green"
            else: conclusion, color = "中性：動能盤整", "orange"
        elif 'ADX' in name:
            if value >= 25: conclusion, color = "強趨勢", "orange"
            else: conclusion, color = "盤整", "blue"
        elif '布林通道' in name:
            if value > last_row['BB_High']: conclusion, color = "警告：價格觸及上軌", "green"
            elif value < last_row['BB_Low']: conclusion, color = "強化：價格觸及下軌", "red"
            else: conclusion, color = "中性：通道內運行", "blue"
        elif 'ATR' in name:
            conclusion, color = "中性：正常波動", "blue"
            
        data.append([name, f"{value:,.4f}", conclusion, color])
    return pd.DataFrame(data, columns=['指標名稱', '最新值', '分析結論', '顏色']).set_index('指標名稱')

# ==============================================================================
# 3. 繪圖與回測函式 (Charting & Backtesting Functions)
# ==============================================================================
def create_comprehensive_chart(df, symbol, period_key):
    """ 創建包含價格、成交量、MACD、RSI/ADX、OBV的完整技術圖表 """
    df_clean = df.dropna()
    if df_clean.empty: return go.Figure()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.15, 0.15, 0.15], specs=[[{"secondary_y": True}], [{}], [{}], [{}]])
    # 圖 1: K線、均線、成交量
    fig.add_trace(go.Candlestick(x=df_clean.index, open=df_clean['Open'], high=df_clean['High'], low=df_clean['Low'], close=df_clean['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_10'], line=dict(color='orange', width=1), name='EMA 10'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_50'], line=dict(color='cyan', width=1.5), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_200'], line=dict(color='purple', width=2, dash='dot'), name='EMA 200'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['Volume'], marker_color='grey', name='成交量', opacity=0.3), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="價格", row=1, col=1); fig.update_yaxes(title_text="成交量", secondary_y=True, row=1, col=1, showgrid=False)
    
    # 圖 2: MACD
    macd_colors = np.where(df_clean['MACD_Hist'] >= 0, '#cc0000', '#1e8449')
    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['MACD_Hist'], marker_color=macd_colors, name='MACD Hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD_Line'], line=dict(color='blue', width=1), name='DIF'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD_Signal'], line=dict(color='orange', width=1), name='DEA'), row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1, zeroline=True)
    
    # 圖 3: RSI / ADX
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['ADX'], line=dict(color='#cc6600', width=1.5, dash='dot'), name='ADX'), row=3, col=1)
    fig.update_yaxes(title_text="RSI/ADX", range=[0, 100], row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5); fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
    
    # 圖 4: OBV
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['OBV'], line=dict(color='green', width=1.5), name='OBV'), row=4, col=1)
    fig.update_yaxes(title_text="OBV", row=4, col=1, zeroline=True)
    
    fig.update_layout(title_text=f"AI 融合分析圖表 - {symbol} ({period_key})", height=850, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def run_backtest(df, initial_capital=100000, commission_rate=0.001):
    """ 執行基於 SMA 20 / EMA 50 交叉的簡單回測 """
    if df.empty or len(df) < 51: return {"total_return": 0, "win_rate": 0, "max_drawdown": 0, "total_trades": 0, "message": "數據不足"}
    data = df.copy()
    data['Signal'] = 0
    buy_signal = (data['SMA_20'] > data['EMA_50']) & (data['SMA_20'].shift(1) <= data['EMA_50'].shift(1))
    sell_signal = (data['SMA_20'] < data['EMA_50']) & (data['SMA_20'].shift(1) >= data['EMA_50'].shift(1))
    data.loc[buy_signal, 'Signal'] = 1; data.loc[sell_signal, 'Signal'] = -1
    
    position, capital, trades, buy_price, capital_curve = 0, initial_capital, [], 0, []

    for i in range(len(data)):
        current_capital = capital * (data['Close'].iloc[i] / buy_price) if position == 1 else capital
        capital_curve.append(current_capital)

        if data['Signal'].iloc[i] == 1 and position == 0:
            position, buy_price, capital = 1, data['Close'].iloc[i], current_capital * (1 - commission_rate)
        elif data['Signal'].iloc[i] == -1 and position == 1:
            profit = (data['Close'].iloc[i] - buy_price) / buy_price
            trades.append(1 if profit > 0 else 0)
            capital, position, buy_price = current_capital * (1 - commission_rate), 0, 0

    if position == 1:
        profit = (data['Close'].iloc[-1] - buy_price) / buy_price
        trades.append(1 if profit > 0 else 0); capital *= (1 + profit)

    total_return = (capital / 100000 - 1) * 100
    win_rate = (sum(trades) / len(trades)) * 100 if trades else 0
    capital_s = pd.Series(capital_curve, index=data.index)
    max_drawdown = (capital_s / capital_s.cummax() - 1).min() * 100 if not capital_s.empty else 0
    
    return { "total_return": round(total_return, 2), "win_rate": round(win_rate, 2), "max_drawdown": round(abs(max_drawdown), 2), "total_trades": len(trades), "message": f"回測區間 {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}。", "capital_curve": capital_s }

# ==============================================================================
# 4. Streamlit 主應用程式邏輯 (Main Streamlit App Logic)
# ==============================================================================
def main():
    # --- 自定義 CSS ---
    st.markdown("""
        <style>
        [data-testid="stSidebar"] .stButton button {
            background-color: rgba(255, 255, 255, 0.1); border-color: #FA8072; color: #FA8072;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: rgba(250, 128, 114, 0.15); border-color: #E9967A; color: #E9967A;
        }
        [data-testid="stMetricValue"] { font-size: 20px; } 
        [data-testid="stMetricLabel"] { font-size: 14px; } 
        .action-buy {color: #cc0000; font-weight: bold;} 
        .action-sell {color: #1e8449; font-weight: bold;} 
        .action-neutral {color: #cc6600; font-weight: bold;} 
        .action-hold-buy {color: #FA8072; font-weight: bold;} 
        .action-hold-sell {color: #80B572; font-weight: bold;}
        </style>""", unsafe_allow_html=True)

    # --- 側邊欄 UI ---
    st.sidebar.title("🚀 AI 趨勢分析")
    st.sidebar.markdown("---")
    
    selected_category = st.sidebar.selectbox('1. 選擇資產類別', list(CATEGORY_HOT_OPTIONS.keys()), index=1, key='category_selector')
    hot_options_map = CATEGORY_HOT_OPTIONS.get(selected_category, {})
    
    default_index = 0
    if '2330.TW' in hot_options_map.values(): default_index = list(hot_options_map.values()).index('2330.TW')

    def sync_text_input_from_selection():
        try:
            symbol_code = hot_options_map[st.session_state.hot_target_selector]
            st.session_state.sidebar_search_input = symbol_code
        except Exception: pass
    
    st.sidebar.selectbox('2. 選擇熱門標的 (或手動輸入)', list(hot_options_map.keys()), index=default_index, key='hot_target_selector', on_change=sync_text_input_from_selection)
    st.sidebar.text_input('...或在這裡手動輸入代碼/名稱:', key='sidebar_search_input')
    st.sidebar.markdown("---")
    selected_period_key = st.sidebar.selectbox('3. 選擇分析週期', list(PERIOD_MAP.keys()), index=2)
    is_long_term = st.sidebar.checkbox('長期投資者模式', value=False, help="勾選後將更側重基本面和籌碼面")
    st.sidebar.markdown("---")
    analyze_button_clicked = st.sidebar.button('📊 執行AI分析', use_container_width=True)

    # --- 主分析流程 ---
    if analyze_button_clicked:
        final_symbol = get_symbol_from_query(st.session_state.sidebar_search_input)
        
        with st.spinner(f"🔍 正在啟動AI模型，分析 **{final_symbol}** 的數據..."):
            yf_period, yf_interval = PERIOD_MAP[selected_period_key]
            df = get_stock_data(final_symbol, yf_period, yf_interval)
            
            if df.empty or len(df) < 51:
                st.error(f"❌ **數據不足或代碼無效：** {final_symbol}。請檢查代碼或更換週期（至少需要51個數據點）。")
                st.session_state['data_ready'] = False
            else:
                st.session_state['analysis_results'] = {
                    'df': calculate_technical_indicators(df),
                    'company_info': get_company_info(final_symbol),
                    'currency_symbol': get_currency_symbol(final_symbol),
                    'fa_result': calculate_fundamental_rating(final_symbol),
                    'chips_news_data': get_chips_and_news_analysis(final_symbol),
                    'selected_period_key': selected_period_key,
                    'final_symbol_to_analyze': final_symbol,
                    'is_long_term': is_long_term
                }
                st.session_state['data_ready'] = True
    
    # --- 結果呈現區 (依照指定順序) ---
    if st.session_state.get('data_ready', False):
        res = st.session_state['analysis_results']
        df, fa_result = res['df'].dropna(), res['fa_result']
        
        analysis = generate_ai_fusion_signal(df, fa_result, res['chips_news_data'], res['is_long_term'], res['currency_symbol'])
        
        # 1. 標題
        st.header(f"📈 {res['company_info']['name']} ({res['final_symbol_to_analyze']}) AI趨勢分析")
        
        # 2. 核心行動與量化評分
        st.subheader("💡 核心行動與量化評分")
        price, prev_close = analysis['current_price'], df['Close'].iloc[-2]
        change, change_pct = price - prev_close, (price - prev_close) / prev_close * 100
        price_format = ".4f" if price < 100 and not res['currency_symbol'] == 'NT$' else ".2f"

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 當前價格", f"{res['currency_symbol']}{price:{price_format}}", f"{change:+.{price_format}} ({change_pct:+.2f}%)", delta_color='inverse' if change < 0 else 'normal')
        
        if "買進" in analysis['action']: action_class = "action-buy" if "偏" not in analysis['action'] else "action-hold-buy"
        elif "賣出" in analysis['action']: action_class = "action-sell" if "偏" not in analysis['action'] else "action-hold-sell"
        else: action_class = "action-neutral"
        col2.markdown(f"**🎯 最終行動建議**\n<p class='{action_class}' style='font-size: 20px;'>{analysis['action']}</p>", unsafe_allow_html=True)
        
        col3.metric("🔥 總量化評分", f"{analysis['score']:.2f}", help="四維融合模型總分，正數看漲")
        st.markdown("---")

        # 3. 精確交易策略與風險控制
        st.subheader("🛡️ 精確交易策略與風險控制")
        strat_col1, strat_col2, strat_col3, strat_col4 = st.columns(4)
        strat_col1.markdown(f"**建議操作:** <br><p class='{action_class}' style='font-size: 18px;'>{analysis['action']}</p>", unsafe_allow_html=True)
        strat_col2.metric("建議進場價:", f"{res['currency_symbol']}{analysis['entry_price']:{price_format}}")
        strat_col3.metric("🚀 止盈價 (TP):", f"{res['currency_symbol']}{analysis['take_profit']:{price_format}}")
        strat_col4.metric("🛑 止損價 (SL):", f"{res['currency_symbol']}{analysis['stop_loss']:{price_format}}")
        
        risk = abs(analysis['entry_price'] - analysis['stop_loss'])
        reward = abs(analysis['take_profit'] - analysis['entry_price'])
        risk_reward = reward / risk if risk > 0 else 0
        
        st.info(f"**💡 策略總結:** {analysis['strategy']} | **⚖️ 風險/回報比 (R:R):** {risk_reward:.2f} | **波動單位 (ATR):** {analysis.get('atr', 0):.4f}")
        st.markdown("---")
        
        # 4. 關鍵技術指標數據 (AI判讀細節)
        st.subheader("📊 關鍵技術指標數據")
        opinions_data = list(analysis['ai_opinions'].items())
        if 'details' in fa_result:
            for key, val in fa_result['details'].items(): opinions_data.append([f"基本面 - {key}", str(val)])
        
        ai_df = pd.DataFrame(opinions_data, columns=['AI分析維度', '判斷結果'])
        st.dataframe(ai_df.style.apply(lambda s: ['color: #1e8449; font-weight: bold;' if '❌' in x or '空頭' in x else 'color: #cc0000; font-weight: bold;' if '✅' in x or '多頭' in x else 'color: #cc6600;' if '⚠️' in x else '' for x in s], subset=['判斷結果']), use_container_width=True)
        st.markdown("---")

        # 5. 技術指標狀態表
        st.subheader("🛠️ 技術指標狀態表")
        technical_df = get_technical_data_df(df)
        if not technical_df.empty:
            def style_indicator(s):
                color_map = {'red': 'color: #cc0000; font-weight: bold;', 'green': 'color: #1e8449; font-weight: bold;', 'orange': 'color: #cc6600;', 'blue': '#888888'}
                return [color_map.get(technical_df['顏色'].loc[idx], '') for idx in s.index]
            styled_df = technical_df[['最新值', '分析結論']].style.apply(style_indicator, axis=0)
            st.dataframe(styled_df, use_container_width=True)
        st.markdown("---")

        # 6. 策略回測報告
        st.subheader("🧪 策略回測報告 (SMA 20/EMA 50 交叉)")
        backtest_results = run_backtest(df.copy())
        if backtest_results.get("total_trades", 0) > 0:
            bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
            bt_col1.metric("📊 總回報率", f"{backtest_results['total_return']}%", delta=backtest_results['message'], delta_color='off')
            bt_col2.metric("📈 勝率", f"{backtest_results['win_rate']}%")
            bt_col3.metric("📉 最大回撤 (MDD)", f"{backtest_results['max_drawdown']}%")
            bt_col4.metric("🤝 交易總次數", f"{backtest_results['total_trades']} 次")
            
            if 'capital_curve' in backtest_results and not backtest_results['capital_curve'].empty:
                fig_bt = go.Figure(go.Scatter(x=backtest_results['capital_curve'].index, y=backtest_results['capital_curve'], name='策略資金曲線', line=dict(color='#cc6600', width=2)))
                fig_bt.update_layout(title='SMA 20/EMA 50 交叉策略資金曲線', xaxis_title='日期', yaxis_title='賬戶價值 ($)', height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.warning(f"回測無法執行或無交易信號：{backtest_results.get('message', '發生錯誤')}")
        st.markdown("---")

        # 7. 完整技術分析圖表
        st.subheader(f"📊 完整技術分析圖表")
        st.plotly_chart(create_comprehensive_chart(df, res['final_symbol_to_analyze'], res['selected_period_key']), use_container_width=True)
        
        with st.expander("📰 點此查看近期相關新聞"):
            st.markdown(res['chips_news_data'].get('news_summary', 'N/A').replace("\n", "\n\n"))

    # --- 歡迎頁面 ---
    elif not st.session_state.get('data_ready', False):
        st.markdown("<h1 style='color: #FA8072; font-size: 32px; font-weight: bold;'>🚀 歡迎使用 AI 趨勢分析</h1>", unsafe_allow_html=True)
        st.markdown(f"請在左側選擇或輸入您想分析的標的（例如：**2330.TW**、**NVDA**、**BTC-USD**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇標的**：使用下拉選單快速選擇熱門標的，或直接在輸入框中鍵入代碼或名稱。")
        st.markdown("3. **選擇週期**：決定分析的長度（例如：`30 分` (短期)、`1 日` (中長線)）。")
        st.markdown("4. **執行分析**：點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span>，AI將融合多維度指標提供精準交易策略。", unsafe_allow_html=True)
        
if __name__ == '__main__':
    if 'data_ready' not in st.session_state: st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state: st.session_state['sidebar_search_input'] = "2330.TW"
    
    main()
    
    # 8. 綜合風險與免責聲明
    st.markdown("---")
    st.markdown("⚠️ **綜合風險與免責聲明 (Risk & Disclaimer)**")
    st.caption("本分析模型包含多位AI的量化觀點，但僅供教育與參考用途。投資涉及風險，所有交易決策應基於您個人的獨立研究和財務狀況，並建議諮詢專業金融顧問。")
    st.markdown("📊 **數據來源:** Yahoo Finance | **技術指標:** TA 庫 | **APP優化:** 專業程式碼專家")
