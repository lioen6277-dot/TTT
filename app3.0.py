import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re

# 忽略 yfinance 的警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 頁面配置與全局設定
# ==============================================================================

st.set_page_config(
    page_title="AI趨勢分析 v7.0📈",
    page_icon="🚀",
    layout="wide"
)

# 週期映射：(YFinance Period, YFinance Interval)
PERIOD_MAP = {
    "30 分": ("60d", "30m"),
    "4 小時": ("1y", "90m"), # yfinance 4h interval often fails, use 90m instead
    "1 日": ("5y", "1d"),
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】(擴充版)
FULL_SYMBOLS_MAP = {
    # 美股/ETF/指數
    "ACN": {"name": "Accenture (埃森哲)", "keywords": ["Accenture", "ACN", "諮詢", "科技服務"]},
    "ADBE": {"name": "Adobe", "keywords": ["Adobe", "ADBE"]},
    "AAPL": {"name": "蘋果 (Apple)", "keywords": ["Apple", "AAPL"]},
    "NVDA": {"name": "輝達 (NVIDIA)", "keywords": ["NVIDIA", "NVDA"]},
    "TSLA": {"name": "特斯拉 (Tesla)", "keywords": ["Tesla", "TSLA"]},
    "GOOG": {"name": "谷歌 (Alphabet Class C)", "keywords": ["Google", "GOOG"]},
    "MSFT": {"name": "微軟 (Microsoft)", "keywords": ["Microsoft", "MSFT"]},
    "QQQ": {"name": "納斯達克100指數ETF", "keywords": ["NASDAQ 100", "QQQ"]},
    "^GSPC": {"name": "標普500指數", "keywords": ["S&P 500", "GSPC"]},
    
    # 台股 (TW) - 個股/ETF/指數
    "2330.TW": {"name": "台積電", "keywords": ["台積電", "TSMC"]},
    "2303.TW": {"name": "聯電", "keywords": ["聯電"]},
    "0050.TW": {"name": "元大台灣50 ETF", "keywords": ["台灣50", "0050"]},
    "2454.TW": {"name": "聯發科", "keywords": ["聯發科"]},
    "^TWII": {"name": "台灣加權指數", "keywords": ["台指", "加權指數"]},

    # 加密貨幣 (Crypto)
    "BTC-USD": {"name": "比特幣", "keywords": ["Bitcoin", "BTC"]},
    "ETH-USD": {"name": "以太坊", "keywords": ["Ethereum", "ETH"]},
}

# 根據 FULL_SYMBOLS_MAP 生成熱門選項
CATEGORY_HOT_OPTIONS = {}
for category, codes in {
    "美股 (US) - 個股/ETF/指數": ["AAPL", "NVDA", "MSFT", "QQQ", "^GSPC"],
    "台股 (TW) - 個股/ETF/指數": ["2330.TW", "2454.TW", "0050.TW", "^TWII"],
    "加密貨幣 (Crypto)": ["BTC-USD", "ETH-USD"]
}.items():
    options = {}
    sorted_codes = sorted(codes)
    for code in sorted_codes:
        info = FULL_SYMBOLS_MAP.get(code)
        if info:
            options[f"{code} - {info['name']}"] = code
    CATEGORY_HOT_OPTIONS[category] = options

# ==============================================================================
# 2. 數據獲取與預處理
# ==============================================================================

def get_symbol_from_query(query: str) -> str:
    """從使用者輸入的代碼或名稱解析出標準代碼"""
    query = query.strip()
    query_upper = query.upper()

    for code, data in FULL_SYMBOLS_MAP.items():
        if query_upper == code.upper(): return code
        if any(query_upper == kw.upper() for kw in data["keywords"]): return code

    for code, data in FULL_SYMBOLS_MAP.items():
        if query == data["name"]: return code

    # 猜測台灣股票代碼 (4-6位數字 + .TW)
    if re.fullmatch(r'\d{4,6}', query) and not any(ext in query_upper for ext in ['.TW', '.HK', '.SS', '-USD']):
        tw_code = f"{query}.TW"
        return tw_code
    return query

@st.cache_data(ttl=3600, show_spinner="正在從 Yahoo Finance 獲取數據...")
def get_stock_data(symbol, period, interval):
    """獲取歷史股票數據"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        df.columns = [col.capitalize() for col in df.columns]
        df.index.name = 'Date'
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[~df.index.duplicated(keep='first')]
        # 移除最後一行（可能是不完整的當前週期數據）
        if len(df) > 1: df = df.iloc[:-1] 
        if df.empty: return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_company_info(symbol):
    """獲取公司名稱、類別和貨幣"""
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

@st.cache_data(ttl=3600)
def get_currency_symbol(symbol):
    """獲取貨幣符號"""
    currency_code = get_company_info(symbol).get('currency', 'USD')
    if currency_code == 'TWD': return 'NT$'
    elif currency_code == 'USD': return '$'
    else: return currency_code + ' '

# ==============================================================================
# 3. V7.0 技術/基本面/融合訊號計算
# ==============================================================================

# 技術指標計算 (v7.0 參數: EMA/MACD/RSI/ADX)
def calculate_technical_indicators(df):
    """計算 v7.0 所需的技術指標"""
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)

    # MACD v7.0 參數: 快8, 慢17, 信號9
    macd_instance = ta.trend.MACD(df['Close'], window_fast=8, window_slow=17, window_sign=9)
    df['MACD_Line'] = macd_instance.macd()
    df['MACD_Signal'] = macd_instance.macd_signal()
    df['MACD_Hist'] = macd_instance.macd_diff()

    # RSI/ADX/ATR v7.0 參數: 9期
    df['RSI'] = ta.momentum.rsi(df['Close'], window=9)
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=9)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=9)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    return df

@st.cache_data(ttl=3600)
def get_chips_and_news_analysis(symbol):
    """ 獲取籌碼面和消息面數據 """
    try:
        ticker = yf.Ticker(symbol)
        inst_holders = ticker.institutional_holders
        inst_hold_pct = 0
        if inst_holders is not None and not inst_holders.empty:
            # 嘗試獲取機構持股比例 (通常在第一欄)
            inst_hold_pct = inst_holders.iloc[0, 0] if isinstance(inst_holders.iloc[0, 0], (int, float)) else 0
        
        news = ticker.news
        news_summary = "近期無相關新聞"
        if news:
            # 獲取前 5 條新聞
            headlines = [f"- **{item.get('type', '新聞')}**: {item['title']}" for item in news[:5]]
            news_summary = "\n".join(headlines)
        
        return {
            "inst_hold_pct": inst_hold_pct,
            "news_summary": news_summary
        }
    except Exception:
        return {
            "inst_hold_pct": 0,
            "news_summary": "無法獲取新聞數據"
        }

@st.cache_data(ttl=3600)
def calculate_advanced_fundamental_rating(symbol):
    """
    v7.0 基本面核心判斷標準 (總分7分)
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if info.get('quoteType') in ['INDEX', 'CRYPTOCURRENCY', 'ETF']:
            return {"score": 0, "summary": "指數、加密貨幣或ETF不適用基本面分析。", "details": {}}

        score = 0
        details = {}

        # 1. 獲利能力 (ROE > 15%) - 權重2分
        roe = info.get('returnOnEquity')
        if roe is not None and roe > 0.15:
            score += 2; details['✅ ROE > 15%'] = f"{roe:.2%}"
        else:
            details['❌ ROE < 15%'] = f"{roe:.2%}" if roe is not None else "N/A"

        # 2. 財務健康 (負債權益比 < 50%) - 權重2分
        debt_to_equity = info.get('debtToEquity')
        # debtToEquity 來自 yf.info 已經是百分比 (例如 100 表示 100%)
        if debt_to_equity is not None and debt_to_equity < 50: 
            score += 2; details['✅ 負債權益比 < 50%'] = f"{debt_to_equity/100:.2%}"
        else:
            details['❌ 負債權益比 > 50%'] = f"{debt_to_equity/100:.2%}" if debt_to_equity is not None else "N/A"

        # 3. 成長性 (營收年增 > 10%) - 權重1分
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth is not None and revenue_growth > 0.1:
            score += 1; details['✅ 營收年增 > 10%'] = f"{revenue_growth:.2%}"
        else:
            details['❌ 營收年增 < 10%'] = f"{revenue_growth:.2%}" if revenue_growth is not None else "N/A"

        # 4. 估值 (P/E < 15, PEG < 1) - 權重2分 (各1分)
        pe = info.get('trailingPE')
        peg = info.get('pegRatio')
        
        if pe is not None and 0 < pe < 15: 
            score += 1; details['✅ 本益比(P/E) < 15'] = f"{pe:.2f}"
        else:
            details['⚠️ 本益比(P/E) > 15'] = f"{pe:.2f}" if pe else "N/A"

        if peg is not None and 0 < peg < 1: 
            score += 1; details['✅ PEG < 1'] = f"{peg:.2f}"
        else:
            details['⚠️ PEG > 1'] = f"{peg:.2f}" if peg else "N/A"
        
        # 總結
        if score >= 6: summary = "頂級優異：公司在獲利、財務、成長性上表現強勁，且估值合理。"
        elif score >= 4: summary = "良好穩健：公司基本面穩固，但在某些方面（如估值或成長性）有待加強。"
        else: summary = "中性警示：需留意公司的財務風險、獲利能力不足或估值偏高的問題。"

        return {"score": score, "summary": summary, "details": details}

    except Exception:
        return {"score": 0, "summary": "無法獲取或計算基本面數據。", "details": {}}

def generate_ai_fusion_signal(df, fa_rating, chips_news_data, is_long_term, currency_symbol):
    """
    v7.0 AI四維融合訊號生成器 (技術+基本+籌碼+成交量)
    """
    if df.empty or len(df) < 2:
        return { 'action': '數據不足', 'score': 0, 'confidence': 50, 'strategy': '無法評估', 'entry_price': 0, 'take_profit': 0, 'stop_loss': 0, 'current_price': 0, 'ai_opinions': {}, 'atr': 0 }

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    current_price = last_row['Close']
    atr = last_row.get('ATR', 0)
    ai_opinions = {}
    
    # 權重參數 (v7.0 精神：長期重基本面/籌碼，短期重技術面/成交量)
    # 總權重皆為 4.0
    WEIGHTS = {
        'LongTerm': {'TA': 0.8, 'FA': 1.6, 'Chips': 1.2, 'Volume': 0.4},
        'ShortTerm': {'TA': 1.6, 'FA': 0.8, 'Chips': 0.4, 'Volume': 1.2}
    }
    
    weights = WEIGHTS['LongTerm'] if is_long_term else WEIGHTS['ShortTerm']
    
    # --- 1. 技術面評分 (TA Score, Max: +6, Min: -6) ---
    ta_score = 0
    # MA 趨勢：強多頭/強空頭 (+/- 2分)
    if last_row['EMA_10'] > last_row['EMA_50'] > last_row['EMA_200']: ta_score += 2; ai_opinions['MA 趨勢'] = '✅ 強多頭排列 (10>50>200)'
    elif last_row['EMA_10'] < last_row['EMA_50'] < last_row['EMA_200']: ta_score -= 2; ai_opinions['MA 趨勢'] = '❌ 強空頭排列 (10<50<200)'
    else: ai_opinions['MA 趨勢'] = '⚠️ 中性盤整'

    # RSI 動能： (+/- 1分)
    if last_row['RSI'] > 70: ta_score -= 1; ai_opinions['RSI 動能'] = '⚠️ 超買區域 (>70)，潛在回調'
    elif last_row['RSI'] < 30: ta_score += 1; ai_opinions['RSI 動能'] = '✅ 超賣區域 (<30)，潛在反彈'
    elif last_row['RSI'] > 50: ta_score += 1; ai_opinions['RSI 動能'] = '✅ 多頭區間 (>50)'
    else: ta_score -= 1; ai_opinions['RSI 動能'] = '❌ 空頭區間 (<50)'

    # MACD 動能： (+/- 2分 for momentum, +/- 1 for crossover)
    if last_row['MACD_Hist'] > 0 and last_row['MACD_Hist'] > prev_row['MACD_Hist']: ta_score += 2; ai_opinions['MACD 動能'] = '✅ 多頭動能增強 (柱狀圖>0)'
    elif last_row['MACD_Hist'] < 0 and last_row['MACD_Hist'] < prev_row['MACD_Hist']: ta_score -= 2; ai_opinions['MACD 動能'] = '❌ 空頭動能增強 (柱狀圖<0)'
    elif last_row['MACD_Line'] > last_row['MACD_Signal'] and prev_row['MACD_Line'] <= prev_row['MACD_Signal']: ta_score += 1; ai_opinions['MACD 動能'] = '✅ MACD金叉，動能轉強'
    elif last_row['MACD_Line'] < last_row['MACD_Signal'] and prev_row['MACD_Line'] >= prev_row['MACD_Signal']: ta_score -= 1; ai_opinions['MACD 動能'] = '❌ MACD死叉，動能轉弱'
    else: ai_opinions['MACD 動能'] = '⚠️ 動能盤整'
    
    # ADX 趨勢強度作為乘數 (ADX > 25 強趨勢: 放大 1.3 倍)
    if last_row['ADX'] > 25: ta_multiplier = 1.3; ai_opinions['ADX 趨勢強度'] = '✅ 強趨勢確認 (>25)'
    else: ta_multiplier = 0.8; ai_opinions['ADX 趨勢強度'] = '⚠️ 盤整趨勢 (<25)'
        
    ta_score *= ta_multiplier
    
    # --- 2. 基本面評分 (FA Score) ---
    # 將 7 分制轉換為 -3 到 +3 的評分
    fa_score_val = fa_rating.get('score', 0)
    fa_score = ((fa_score_val / 7.0) * 6.0) - 3.0
    
    # --- 3. 籌碼與成交量評分 (Chips & Volume Score, Max +/- 1.5) ---
    chips_score, volume_score = 0, 0
    inst_hold_pct = chips_news_data.get('inst_hold_pct', 0) * 100
    
    # Chips v7.0 Logic
    if inst_hold_pct > 70: 
        chips_score = 1.5
        ai_opinions['籌碼分析'] = f'✅ 法人高度集中 ({inst_hold_pct:.1f}%)'
    elif inst_hold_pct > 40: 
        chips_score = 0.5
        ai_opinions['籌碼分析'] = f'✅ 法人持股穩定 ({inst_hold_pct:.1f}%)'
    # 新增 v7.0 規則：基本面佳 (>=4分) 但籌碼數據缺失或極低 (<1%)
    elif fa_score_val >= 4 and inst_hold_pct < 1: 
        chips_score = -1.5 
        ai_opinions['籌碼分析'] = '❌ 基本面佳但法人數據缺失/極低，高流動性風險警示'
    else: 
        chips_score = -0.5
        ai_opinions['籌碼分析'] = f'⚠️ 籌碼較分散 ({inst_hold_pct:.1f}%)'
        
    is_high_volume = last_row['Volume'] > (last_row.get('Volume_MA_20', 0) * 1.5) # V > 150% MA
    is_low_volume = last_row['Volume'] < (last_row.get('Volume_MA_20', 0) * 0.5) # V < 50% MA

    # Volume v7.0 Logic
    if is_high_volume and last_row['Close'] > prev_row['Close']: 
        volume_score = 1.5
        ai_opinions['成交量分析'] = '✅ 價漲量爆 (>150% MA)，趨勢強勁'
    elif is_high_volume and last_row['Close'] < prev_row['Close']: 
        volume_score = -1.5
        ai_opinions['成交量分析'] = '❌ 價跌量爆 (>150% MA)，空頭壓力'
    elif is_low_volume: 
        volume_score = -0.5
        ai_opinions['成交量分析'] = '⚠️ 量能萎縮 (<50% MA)，趨勢無力'
    else: 
        ai_opinions['成交量分析'] = '⚠️ 量能中性或價量背離'
    
    # --- 4. 融合總分 ---
    divisor = 4.0 # 權重總和
    total_score = (ta_score * weights['TA'] + fa_score * weights['FA'] + chips_score * weights['Chips'] + volume_score * weights['Volume']) / divisor
    
    # 信心指數 (將總分絕對值放大到 40%~100% 範圍)
    confidence = min(100, max(40, abs(total_score) * 15 + 40))

    if total_score > 3.5: action = '買進 (Strong Buy)'
    elif total_score > 1.5: action = '中性偏買 (Hold/Buy)'
    elif total_score < -3.5: action = '賣出 (Strong Sell/Short)'
    elif total_score < -1.5: action = '中性偏賣 (Hold/Sell)'
    else: action = '中性 (Neutral)'

    entry_price = current_price
    # 採用 2:1 風報比，止盈 2x ATR，止損 1x ATR
    take_profit = current_price + atr * 2.0 if total_score > 0 else current_price - atr * 2.0
    stop_loss = current_price - atr * 1.0 if total_score > 0 else current_price + atr * 1.0
    strategy = f'基於TA/FA/籌碼/量能的四維融合模型 (長期模式: {is_long_term})'

    return {
        'current_price': current_price, 'action': action, 'score': total_score, 'confidence': confidence,
        'entry_price': entry_price, 'take_profit': take_profit, 'stop_loss': stop_loss,
        'strategy': strategy, 'atr': atr, 'ai_opinions': ai_opinions
    }

def get_technical_data_df(df):
    """計算並彙整技術指標狀態，並將顏色編碼作為單獨一欄。"""
    if df.empty or len(df) < 200: return pd.DataFrame()
    df_clean = df.dropna().copy()
    if df_clean.empty: return pd.DataFrame()
    last_row = df_clean.iloc[-1]
    prev_row = df_clean.iloc[-2] if len(df_clean) >= 2 else last_row
    
    indicators = {'價格 vs. EMA 10/50/200': last_row['Close'], 'RSI (9) 動能': last_row['RSI'], 'MACD (8/17/9) 柱狀圖': last_row['MACD_Hist'], 'ADX (9) 趨勢強度': last_row['ADX'], 'ATR (9) 波動性': last_row['ATR'], '布林通道 (BB: 20/2)': last_row['Close']}
    data = []
    
    for name, value in indicators.items():
        conclusion, color = "", "grey"
        if 'EMA' in name:
            ema_10, ema_50, ema_200 = last_row['EMA_10'], last_row['EMA_50'], last_row['EMA_200']
            if ema_10 > ema_50 and ema_50 > ema_200: conclusion, color = f"**強多頭：MA 多頭排列**", "red"
            elif ema_10 < ema_50 and ema_50 < ema_200: conclusion, color = f"**強空頭：MA 空頭排列**", "green"
            elif last_row['Close'] > ema_50: conclusion, color = f"中長線偏多", "orange"
            else: conclusion, color = "中性：趨勢發展中", "blue"
        elif 'RSI' in name:
            if value > 70: conclusion, color = "警告：超買區域", "green"
            elif value < 30: conclusion, color = "強化：超賣區域", "red"
            elif value > 50: conclusion, color = "多頭：RSI > 50", "red"
            else: conclusion, color = "空頭：RSI < 50", "green"
        elif 'MACD' in name:
            if value > 0 and value > prev_row['MACD_Hist']: conclusion, color = "強化：多頭動能增強", "red"
            elif value < 0 and value < prev_row['MACD_Hist']: conclusion, color = "削弱：空頭動能增強", "green"
            else: conclusion, color = "中性：動能盤整", "orange"
        elif 'ADX' in name:
            if value >= 25: conclusion, color = "強趨勢：確認趨勢", "orange"
            else: conclusion, color = "盤整：弱勢或橫盤", "blue"
        elif 'ATR' in name:
            avg_atr = df_clean['ATR'].iloc[-30:].mean() if len(df_clean) >= 30 else df_clean['ATR'].mean()
            if value > avg_atr * 1.5: conclusion, color = "警告：極高波動性", "green"
            else: conclusion, color = "中性：正常波動", "blue"
        elif '布林通道' in name:
            if value > last_row['BB_High']: conclusion, color = "警告：價格位於上軌外側 (超買)", "red"
            elif value < last_row['BB_Low']: conclusion, color = "強化：價格位於下軌外側 (超賣)", "green"
            else: conclusion, color = "中性：在上下軌間", "blue"
        data.append([name, value, conclusion, color])
    # 返回帶有 '顏色' 欄位的 DataFrame
    return pd.DataFrame(data, columns=['指標名稱', '最新值', '分析結論', '顏色']).set_index('指標名稱')

# 繪圖函數
def create_comprehensive_chart(df, symbol, period_key):
    """生成綜合 K 線圖、MACD、RSI/ADX 和 OBV 圖"""
    df_clean = df.dropna()
    if df_clean.empty: return go.Figure()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2], specs=[[{"secondary_y": True}], [{}], [{}], [{}]])
    
    # 1. K線 & EMA & 成交量
    fig.add_trace(go.Candlestick(x=df_clean.index, open=df_clean['Open'], high=df_clean['High'], low=df_clean['Low'], close=df_clean['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_10'], line=dict(color='orange', width=1), name='EMA 10'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_50'], line=dict(color='cyan', width=1.5), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_200'], line=dict(color='purple', width=2, dash='dot'), name='EMA 200'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['Volume'], marker_color='grey', name='成交量', opacity=0.3), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="價格", row=1, col=1); fig.update_yaxes(title_text="成交量", secondary_y=True, row=1, col=1, showgrid=False)
    
    # 2. MACD
    macd_colors = np.where(df_clean['MACD_Hist'] >= 0, '#cc0000', '#1e8449')
    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['MACD_Hist'], marker_color=macd_colors, name='MACD Hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD_Line'], line=dict(color='blue', width=1), name='MACD 線'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD_Signal'], line=dict(color='orange', width=1), name='Signal 線'), row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1, zeroline=True)
    
    # 3. RSI/ADX
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['ADX'], line=dict(color='#cc6600', width=1.5, dash='dot'), name='ADX'), row=3, col=1)
    fig.update_yaxes(title_text="RSI/ADX", range=[0, 100], row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5); fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
    
    # 4. OBV
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['OBV'], line=dict(color='green', width=1.5), name='OBV'), row=4, col=1)
    fig.update_yaxes(title_text="OBV", row=4, col=1)
    
    fig.update_layout(title_text=f"AI 融合分析圖表 - {symbol} ({period_key})", height=900, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# 回測函數 (SMA 20/EMA 50 交叉)
def run_backtest(df, initial_capital=100000, commission_rate=0.001):
    """運行簡單的均線交叉回測策略"""
    if df.empty or len(df) < 51: return {"total_return": 0, "win_rate": 0, "max_drawdown": 0, "total_trades": 0, "message": "數據不足"}
    data = df.copy()
    data['Signal'] = 0
    # 策略: SMA_20/EMA_50 交叉
    buy_signal = (data['SMA_20'] > data['EMA_50']) & (data['SMA_20'].shift(1) <= data['EMA_50'].shift(1))
    sell_signal = (data['SMA_20'] < data['EMA_50']) & (data['SMA_20'].shift(1) >= data['EMA_50'].shift(1))
    data.loc[buy_signal, 'Signal'] = 1; data.loc[sell_signal, 'Signal'] = -1
    
    position, capital, trades, buy_price = 0, initial_capital, [], 0
    capital_curve = []

    for i in range(len(data)):
        current_capital = capital
        if position == 1:
            current_capital = capital * (data['Close'].iloc[i] / buy_price)
        capital_curve.append(current_capital)

        if data['Signal'].iloc[i] == 1 and position == 0:
            # 開倉 (買入)
            position = 1; buy_price = data['Close'].iloc[i]; capital = current_capital * (1 - commission_rate)
        elif data['Signal'].iloc[i] == -1 and position == 1:
            # 平倉 (賣出)
            profit = (data['Close'].iloc[i] - buy_price) / buy_price
            trades.append(1 if profit > 0 else 0)
            capital = current_capital * (1 - commission_rate) * (1 + profit) 
            position = 0
            buy_price = 0

    # 結算最後一筆交易
    if position == 1:
        profit = (data['Close'].iloc[-1] - buy_price) / buy_price
        trades.append(1 if profit > 0 else 0); capital = capital_curve[-1] * (1 + profit) 

    total_return = (capital / initial_capital - 1) * 100
    win_rate = (sum(trades) / len(trades)) * 100 if trades else 0
    
    capital_s = pd.Series(capital_curve, index=data.index)
    max_drawdown = (capital_s / capital_s.cummax() - 1).min() * 100
    
    return { "total_return": round(total_return, 2), "win_rate": round(win_rate, 2), "max_drawdown": round(abs(max_drawdown), 2), "total_trades": len(trades), "message": f"回測區間 {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}。", "capital_curve": capital_s }

# ==============================================================================
# 4. Streamlit 主應用程式邏輯
# ==============================================================================

def main():
    # --- 側邊欄 UI ---
    st.sidebar.title("🚀 AI 趨勢分析 v7.0")
    st.sidebar.markdown("---")
    
    selected_category = st.sidebar.selectbox(
        '1. 選擇資產類別', 
        list(CATEGORY_HOT_OPTIONS.keys()), 
        index=1,
        key='category_selector'
    )
    
    hot_options_map = CATEGORY_HOT_OPTIONS.get(selected_category, {})
    st.sidebar.markdown("---")

    # 設置台積電 (2330.TW) 為台股類別的預設值
    default_index = 0
    if selected_category == '台股 (TW) - 個股/ETF/指數' and '2330.TW - 台積電' in hot_options_map.keys():
        try: default_index = list(hot_options_map.keys()).index('2330.TW - 台積電')
        except ValueError: default_index = 0

    selected_hot_option_key = st.sidebar.selectbox(
        '2. 選擇熱門標的 (或手動輸入)', 
        list(hot_options_map.keys()), 
        index=default_index,
        key='hot_target_selector',
        on_change=sync_text_input_from_selection
    )
    
    initial_search_input = st.session_state.get('sidebar_search_input', "2330.TW")
    
    search_input = st.sidebar.text_input(
        '...或在這裡手動輸入代碼/名稱:', 
        value=initial_search_input,
        key='sidebar_search_input'
    )
    
    st.sidebar.markdown("---")
    # 預設選中 '1 日' (Index 2)
    selected_period_key = st.sidebar.selectbox('3. 選擇分析週期', list(PERIOD_MAP.keys()), index=2)
    st.sidebar.markdown("---")
    is_long_term = st.sidebar.checkbox('長期投資者模式', value=False, help="勾選後將更側重基本面和籌碼面，採用 v7.0 長期權重模型。")
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
                df_with_ta = calculate_technical_indicators(df)
                if df_with_ta.dropna(subset=['Close', 'EMA_10', 'RSI', 'MACD_Hist']).empty:
                     st.error("❌ **數據處理失敗：** 核心技術指標計算結果為空。請嘗試更換週期或標的。")
                     st.session_state['data_ready'] = False
                     return
                     
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
    
    # --- 結果呈現區 ---
    if st.session_state.get('data_ready', False):
        res = st.session_state['analysis_results']
        df_clean = res['df'].dropna(subset=['Close', 'EMA_10', 'RSI', 'MACD_Hist'])

        analysis = generate_ai_fusion_signal(
            df_clean, res['fa_result'], res['chips_news_data'], res['is_long_term'], res['currency_symbol']
        )
        
        st.header(f"📈 **{res['company_info']['name']}** ({res['final_symbol_to_analyze']}) AI四維融合趨勢分析")
        price = analysis['current_price']

        if df_clean.shape[0] >= 2:
            prev_close = df_clean['Close'].iloc[-2]
            change, change_pct = price - prev_close, (price - prev_close) / prev_close * 100
            delta_label = f"{change:+.2f} ({change_pct:+.2f}%)"
            delta_color = 'inverse' if change < 0 else 'normal'
        else:
            delta_label = "N/A"
            delta_color = 'off'

        st.markdown(f"**分析週期:** **{res['selected_period_key']}** | **基本面(FA)評級:** **{res['fa_result'].get('score', 0):.1f}/7.0**")
        st.markdown(f"**基本面診斷:** {res['fa_result'].get('summary', 'N/A')}")
        st.markdown("---")
        
        st.subheader("💡 核心行動與量化評分")
        st.markdown("""<style>[data-testid="stMetricValue"] { font-size: 20px; } [data-testid="stMetricLabel"] { font-size: 13px; } .action-buy {color: #cc0000; font-weight: bold;} .action-sell {color: #1e8449; font-weight: bold;} .action-neutral {color: #cc6600; font-weight: bold;} .action-hold-buy {color: #FA8072; font-weight: bold;} .action-hold-sell {color: #80B572; font-weight: bold;}</style>""", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 當前價格", f"{res['currency_symbol']}{price:,.2f}", delta_label, delta_color=delta_color)
        
        if "買進" in analysis['action']: action_class = "action-buy" if "偏" not in analysis['action'] else "action-hold-buy"
        elif "賣出" in analysis['action']: action_class = "action-sell" if "偏" not in analysis['action'] else "action-hold-sell"
        else: action_class = "action-neutral"
        col2.markdown(f"**🎯 最終行動建議**\n<p class='{action_class}' style='font-size: 20px;'>{analysis['action']}</p>", unsafe_allow_html=True)
        
        col3.metric("🔥 總量化評分", f"{analysis['score']:.2f}", help=f"四維融合模型總分 (長期模式:{res['is_long_term']})")
        col4.metric("🛡️ 信心指數", f"{analysis['confidence']:.0f}%", help="AI對此建議的信心度")
        
        st.markdown("---")
        st.subheader("🛡️ 交易策略參考 (基於 ATR 風險/報酬)")
        col_risk_1, col_risk_2, col_risk_3 = st.columns(3)
        col_risk_1.metric("🛒 建議入場價", f"{res['currency_symbol']}{analysis['entry_price']:,.2f}")
        col_risk_2.metric("🟢 建議止盈 (2x ATR)", f"{res['currency_symbol']}{analysis['take_profit']:,.2f}")
        col_risk_3.metric("🔴 建議止損 (1x ATR)", f"{res['currency_symbol']}{analysis['stop_loss']:,.2f}")
        st.caption(f"波動性 (ATR): {res['currency_symbol']}{analysis['atr']:,.2f}。採用 2:1 風報比策略。")
        st.markdown("---")
        
        st.subheader("📊 AI判讀細節 (交叉驗證)")
        opinions_data = list(analysis['ai_opinions'].items())
        if 'details' in res['fa_result']:
            # 將基本面細節加入判讀
            for key, val in res['fa_result']['details'].items(): opinions_data.append([f"基本面 - {key}", str(val)])
        
        ai_df = pd.DataFrame(opinions_data, columns=['AI分析維度', '判斷結果'])
        
        # 修正 Styler ValueError: 應用顏色樣式
        def style_analysis(s):
            is_negative = ('❌' in s) or ('空頭' in s) or ('削弱' in s) or ('< 50%' in s) or ('< 15%' in s) or ('< 10%' in s) or ('< 50' in s) or ('超買' in s) or ('> 15' in s) or ('> 1' in s)
            is_positive = ('✅' in s) or ('多頭' in s) or ('強化' in s) or ('金叉' in s)
            
            if is_negative:
                return ['color: #1e8449']
            elif is_positive:
                return ['color: #cc0000']
            else:
                return ['color: #cc6600']

        styled_ai_df = ai_df.style.apply(lambda s: style_analysis(s['判斷結果']), axis=1, subset=['判斷結果'])
        st.dataframe(styled_ai_df, use_container_width=True)
        st.markdown("---")
        
        st.subheader("🧪 策略回測報告 (SMA 20/EMA 50 交叉)")
        backtest_results = run_backtest(df_clean.copy())
        if backtest_results.get("total_trades", 0) > 0:
            col_bt_1, col_bt_2, col_bt_3, col_bt_4 = st.columns(4)
            col_bt_1.metric("📊 總回報率", f"{backtest_results['total_return']}%", delta=backtest_results['message'])
            col_bt_2.metric("📈 勝率", f"{backtest_results['win_rate']}%")
            col_bt_3.metric("📉 最大回撤 (MDD)", f"{backtest_results['max_drawdown']}%")
            col_bt_4.metric("🤝 交易總次數", f"{backtest_results['total_trades']} 次")
            
            if 'capital_curve' in backtest_results and not backtest_results['capital_curve'].empty:
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=backtest_results['capital_curve'].index, y=backtest_results['capital_curve'], name='策略資金曲線', line=dict(color='#cc6600', width=2)))
                fig_bt.update_layout(title='SMA 20/EMA 50 交叉策略資金曲線', xaxis_title='日期', yaxis_title='賬戶價值 ($)', height=300)
                st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.warning(f"回測無法執行或無交易信號：{backtest_results.get('message', '發生錯誤')}")
        st.markdown("---")

        st.subheader("🛠️ 技術指標狀態表")
        technical_df = get_technical_data_df(df_clean)
        
        # 修正 Pandas Styler ValueError, 改用 row-wise 應用 (axis=1)
        if not technical_df.empty:
            def style_row(row):
                """根據 '顏色' 欄位的值，應用 CSS 樣式到 '分析結論' 欄位。"""
                color_map = {'red': '#cc0000', 'green': '#1e8449', 'orange': '#cc6600', 'blue': '#888888', 'grey': '#888888'}
                color_code = color_map.get(row['顏色'], '')
                css_style = f"color: {color_code}; font-weight: bold;" if color_code else ""
                
                # 必須返回與顯示 DataFrame 列數相同的列表。顯示的列為 ['最新值', '分析結論']
                styles = [''] * 2
                styles[1] = css_style # 樣式應用到 '分析結論' 欄位 (索引 1)
                
                return styles
            
            # 準備要顯示的 DataFrame (排除 '顏色' 列)
            df_display = technical_df[['最新值', '分析結論']]

            # 應用 row-wise 樣式 (axis=1)
            styled_df = df_display.style.apply(style_row, axis=1)
            
            st.dataframe(styled_df, use_container_width=True)
        st.markdown("---")

        st.subheader(f"📊 完整技術分析圖表")
        st.plotly_chart(create_comprehensive_chart(df_clean, res['final_symbol_to_analyze'], res['selected_period_key']), use_container_width=True)
        
        with st.expander("📰 點此查看近期相關新聞"):
            st.markdown(res['chips_news_data'].get('news_summary', 'N/A').replace("\n", "\n\n"))

    # --- 歡迎頁面 (v7.0 更新) ---
    elif not st.session_state.get('data_ready', False):
        st.markdown("<h1 style='color: #FA8072; font-size: 32px; font-weight: bold;'>🚀 歡迎使用 AI 趨勢分析 (v7.0 優化版)</h1>", unsafe_allow_html=True)
        st.markdown(f"本系統整合 **20+ AI 代理視角**（量化分析師、風險管理AI、金融軟體工程師等），提供 **四維融合模型** 的精準趨勢判斷。")
        st.markdown(f"請在左側選擇或輸入您想分析的標的（例如：**2330.TW**、**NVDA**、**BTC-USD**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📝 V7.0 使用步驟：")
        st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇標的**：使用下拉選單快速選擇熱門標的，或直接在輸入框中鍵入代碼或名稱。")
        st.markdown("3. **選擇週期**：決定分析的長度（例如：`30 分` (短期)、`1 日` (中長線)）。")
        st.markdown("4. **選擇模式**：勾選**『長期投資者模式』**可將權重傾向於基本面與籌碼面。")
        st.markdown("5. **執行分析**：點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span>，AI將融合基本面與技術面指標提供交易策略。", unsafe_allow_html=True)
        
# --- 應用程式進入點與免責聲明 ---
def sync_text_input_from_selection():
    """當下拉選單變動時，觸發此函式，更新文字輸入框的值。"""
    try:
        selected_category = st.session_state.category_selector
        selected_hot_key = st.session_state.hot_target_selector
        symbol_code = CATEGORY_HOT_OPTIONS[selected_category][selected_hot_key]
        st.session_state.sidebar_search_input = symbol_code
    except Exception:
        pass

if __name__ == '__main__':
    if 'data_ready' not in st.session_state: st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state: st.session_state['sidebar_search_input'] = "2330.TW"
    
    main()
    
    st.markdown("---")
    st.markdown("⚠️ **免責聲明**")
    st.caption("本分析模型包含多位AI（如：量化分析師、風險管理AI、金融軟體工程師、區塊鏈開發者）的量化觀點，但僅供教育與參考用途。投資涉及風險，所有交易決策應基於您個人的獨立研究和財務狀況，並建議諮詢專業金融顧問。")
    st.markdown("📊 **數據基礎與技術支持:** 多源交叉驗證 (Yahoo Finance, Goodinfo等) | **技術指標:** TA 庫 | **APP優化:** 專業程式碼專家+雲端AI部署")


