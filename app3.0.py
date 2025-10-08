# app_ai_fusion_v8_FINAL.py (v8.0 - 整合廣泛技術指標)

import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf
from plotly.subplots import make_subplots

# 警告過濾
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 頁面配置與全局設定
# ==============================================================================

st.set_page_config(
    page_title="AI趨勢分析📈",
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

# 🚀 您的【所有資產清單】
# 為了節省篇幅，此處保持簡化，請確保您的實際檔案中的清單完整
FULL_SYMBOLS_MAP = {
    "AAPL": {"name": "蘋果 (Apple)", "keywords": ["蘋果", "Apple", "AAPL"]},
    "NVDA": {"name": "輝達 (Nvidia)", "keywords": ["輝達", "英偉達", "AI", "NVDA"]},
    "MSFT": {"name": "微軟 (Microsoft)", "keywords": ["微軟", "Microsoft", "MSFT", "雲端", "AI"]},
    "2330.TW": {"name": "台積電", "keywords": ["台積電", "2330", "TSMC"]},
    "BTC-USD": {"name": "比特幣 (Bitcoin)", "keywords": ["比特幣", "BTC", "bitcoin"]},
    "^GSPC": {"name": "S&P 500 指數", "keywords": ["標普", "S&P500", "^GSPC", "SPX"]},
    "2317.TW": {"name": "鴻海", "keywords": ["鴻海", "2317", "Foxconn"]},
    # ... 其他標的
}
# (由於 Streamlit 執行需要，此處省略了完整的資產清單，請使用您上一個步驟提供的完整清單)


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
# 2. 數據獲取與預處理 (此部分保持不變)
# ==============================================================================

def get_symbol_from_query(query: str) -> str:
    query = query.strip()
    query_upper = query.upper()

    for code, data in FULL_SYMBOLS_MAP.items():
        if query_upper == code.upper(): return code
        if any(query_upper == kw.upper() for kw in data["keywords"]): return code

    for code, data in FULL_SYMBOLS_MAP.items():
        if query == data["name"]: return code

    if re.fullmatch(r'\d{4,6}', query) and not any(ext in query_upper for ext in ['.TW', '.HK', '.SS', '-USD']):
        tw_code = f"{query}.TW"
        return tw_code
    return query

@st.cache_data(ttl=3600, show_spinner="正在從 Yahoo Finance 獲取數據...")
def get_stock_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        df.columns = [col.capitalize() for col in df.columns]
        df.index.name = 'Date'
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[~df.index.duplicated(keep='first')]
        
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
# 3. 技術分析 (TA) 計算 (新增多種移動平均線與震盪指標)
# ==============================================================================

def calculate_technical_indicators(df):
    
    # 設置動態窗口，確保在數據點不足時不會生成過多的 NaN
    data_len = len(df)
    win_200 = min(data_len, 200) 
    win_50 = min(data_len, 50)
    win_20 = min(data_len, 20)
    win_14 = min(data_len, 14) # 新標準窗口 (RSI, ADX, ATR, StochRSI)
    win_10 = min(data_len, 10)
    win_9 = min(data_len, 9)

    # --- 1. 移動平均線家族 (MA: SMA, EMA, LWMA, HMA, KAMA) ---
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=win_20)
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=win_10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=win_50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=win_200)
    
    # LWMA (Linear Weighted Moving Average)
    df['LWMA_20'] = ta.trend.wma_indicator(df['Close'], window=win_20)
    # HMA (Hull Moving Average)
    df['HMA_14'] = ta.trend.hull_moving_average(df['Close'], window=win_14)
    # KAMA (Kaufman Adaptive Moving Average)
    df['KAMA_10'] = ta.trend.kama(df['Close'], window=win_10)


    # --- 2. 動能與震盪指標 (RSI, MACD, StochRSI, CCI, Williams %R) ---
    macd_instance = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9) # MACD 標準週期
    df['MACD_Line'] = macd_instance.macd()
    df['MACD_Signal'] = macd_instance.macd_signal()
    df['MACD_Hist'] = macd_instance.macd_diff()

    df['RSI'] = ta.momentum.rsi(df['Close'], window=win_14)
    # StochRSI (Stochastic RSI)
    df['StochRSI'] = ta.momentum.stochrsi(df['Close'], window=win_14)
    # CCI (Commodity Channel Index)
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=win_20)
    # Williams %R
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], window=win_14)

    # --- 3. 趨勢與波動性指標 (ADX, ATR, BB, Ichimoku) ---
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'], window=win_20, window_dev=2)
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'], window=win_20, window_dev=2)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=win_14)
    
    # ADX/DMI (Average Directional Index/Movement Index)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=win_14)
    df['ADX_DI_P'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=win_14)
    df['ADX_DI_N'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=win_14)
    
    # Ichimoku Cloud (一目均衡表 - 轉換線/基準線/延遲線)
    ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'], window1=9, window2=26, window3=52)
    df['Ichimoku_Convert'] = ichimoku.ichimoku_conversion_line()
    df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
    df['Ichimoku_Lag'] = ichimoku.ichimoku_lagging_span()
    # 雲圖的先行帶 Senkou Span A/B 
    # df['Ichimoku_SpanA'] = ichimoku.ichimoku_a()
    # df['Ichimoku_SpanB'] = ichimoku.ichimoku_b()


    # --- 4. 成交量指標 (OBV, MFI, VWAP) ---
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Volume_MA_20'] = df['Volume'].rolling(window=win_20).mean()
    # MFI (Money Flow Index)
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=win_14)
    # VWAP (Volume Weighted Average Price) - 計算整個期間的累計VWAP
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum() 

    return df

# ==============================================================================
# 4. AI 訊號與技術指標狀態 (更新)
# ==============================================================================

# ... (get_chips_and_news_analysis, calculate_advanced_fundamental_rating, get_currency_symbol 保持不變)

def generate_ai_fusion_signal(df, fa_rating, chips_news_data, is_long_term, currency_symbol):
    """
    AI四維融合訊號生成器
    """
    # 確保數據夠用
    df_signal = df.dropna(subset=['Close', 'EMA_10', 'EMA_50', 'MACD_Hist', 'RSI', 'ATR', 'CCI', 'StochRSI']).copy()
    if df_signal.empty or len(df_signal) < 2:
        return { 'action': '數據不足', 'score': 0, 'confidence': 50, 'strategy': '無法評估', 'entry_price': 0, 'take_profit': 0, 'stop_loss': 0, 'current_price': df['Close'].iloc[-1] if not df.empty else 0, 'ai_opinions': {}, 'atr': 0 }

    last_row = df_signal.iloc[-1]
    prev_row = df_signal.iloc[-2]
    current_price = last_row['Close']
    atr = last_row.get('ATR', 0)
    ai_opinions = {}
    
    # 權重參數
    WEIGHTS = {
        'LongTerm': {'TA': 0.8, 'FA': 1.6, 'Chips': 1.2, 'Volume': 0.4},
        'ShortTerm': {'TA': 1.6, 'FA': 0.8, 'Chips': 0.4, 'Volume': 1.2}
    }
    weights = WEIGHTS['LongTerm'] if is_long_term else WEIGHTS['ShortTerm']
    
    # --- 1. 技術面評分 (TA Score, Max: +6, Min: -6) ---
    ta_score = 0

    # MA：EMA/HMA/KAMA 多頭排列
    if last_row['EMA_10'] > last_row['EMA_50'] > last_row['EMA_200']: ta_score += 1.5; ai_opinions['MA 趨勢'] = '✅ 強多頭排列 (10>50>200)'
    elif last_row['EMA_10'] < last_row['EMA_50'] < last_row['EMA_200']: ta_score -= 1.5; ai_opinions['MA 趨勢'] = '❌ 強空頭排列 (10<50<200)'
    elif last_row['HMA_14'] > last_row['LWMA_20'] and last_row['Close'] > last_row['VWAP']: ta_score += 0.5; ai_opinions['MA 趨勢'] = '✅ 短線MA/VWAP偏多'
    else: ai_opinions['MA 趨勢'] = '⚠️ MA/VWAP中性盤整'

    # RSI/StochRSI/CCI：
    if last_row['RSI'] > 70 or last_row['StochRSI'] > 0.8: ta_score -= 1; ai_opinions['RSI/Stoch 動能'] = '⚠️ 超買區域，潛在回調'
    elif last_row['RSI'] < 30 or last_row['StochRSI'] < 0.2: ta_score += 1; ai_opinions['RSI/Stoch 動能'] = '✅ 超賣區域，潛在反彈'
    elif last_row['CCI'] > 100: ta_score += 1; ai_opinions['CCI 動能'] = '✅ 強多頭動能 (>100)'
    elif last_row['CCI'] < -100: ta_score -= 1; ai_opinions['CCI 動能'] = '❌ 強空頭動能 (<-100)'
    else: ai_opinions['CCI 動能'] = '⚠️ 動能中性'

    # MACD：
    if last_row['MACD_Hist'] > 0 and last_row['MACD_Hist'] > prev_row['MACD_Hist']: ta_score += 1; ai_opinions['MACD 動能'] = '✅ 多頭動能增強 (柱狀圖>0)'
    elif last_row['MACD_Hist'] < 0 and last_row['MACD_Hist'] < prev_row['MACD_Hist']: ta_score -= 1; ai_opinions['MACD 動能'] = '❌ 空頭動能增強 (柱狀圖<0)'
    else: ai_opinions['MACD 動能'] = '⚠️ 動能盤整'
    
    # ADX：
    if last_row['ADX'] > 25: 
        if last_row['ADX_DI_P'] > last_row['ADX_DI_N']: ta_score += 1; ai_opinions['ADX 趨勢強度'] = '✅ 強多頭趨勢 (ADX>25, +DI>-DI)'
        else: ta_score -= 1; ai_opinions['ADX 趨勢強度'] = '❌ 強空頭趨勢 (ADX>25, -DI>+DI)'
    else: ai_opinions['ADX 趨勢強度'] = '⚠️ 盤整趨勢 (<25)'
        
    # Ichimoku: 價格在雲圖轉換線/基準線上方
    if last_row['Close'] > last_row.get('Ichimoku_Base', current_price) and last_row['Ichimoku_Convert'] > last_row['Ichimoku_Base']:
        ta_score += 1; ai_opinions['一目均衡表'] = '✅ 價格/轉換線在基準線上方'
    else:
        ta_score -= 0.5; ai_opinions['一目均衡表'] = '⚠️ 價格/轉換線在基準線下方或盤整'

    # --- 2. 基本面評分 (FA Score) (保持不變) ---
    fa_score = ((fa_rating.get('score', 0) / 7.0) * 6.0) - 3.0
    
    # --- 3. 籌碼與成交量評分 (Chips & Volume Score) ---
    chips_score, volume_score = 0, 0
    inst_hold_pct = chips_news_data.get('inst_hold_pct', 0) * 100
    
    # 籌碼集中度 (保持不變)
    if inst_hold_pct > 70: chips_score = 1.5; ai_opinions['籌碼分析'] = f'✅ 法人高度集中 ({inst_hold_pct:.1f}%)'
    elif inst_hold_pct > 40: chips_score = 0.5; ai_opinions['籌碼分析'] = f'✅ 法人持股穩定 ({inst_hold_pct:.1f}%)'
    elif inst_hold_pct == 0 and fa_rating.get('score', 0) > 0: chips_score = -1.5; ai_opinions['籌碼分析'] = '❌ 數據缺失，可能流動性低/無法人關注'
    else: chips_score = -0.5; ai_opinions['籌碼分析'] = f'⚠️ 籌碼較分散 ({inst_hold_pct:.1f}%)'
        
    # 成交量 (新增 MFI)
    is_high_volume = last_row['Volume'] > (last_row.get('Volume_MA_20', 0) * 1.5)
    
    if is_high_volume and last_row['Close'] > prev_row['Close'] and last_row['MFI'] > 80: volume_score = 1.5; ai_opinions['成交量分析'] = '✅ 價漲量爆/MFI強，趨勢強勁'
    elif is_high_volume and last_row['Close'] < prev_row['Close'] and last_row['MFI'] < 20: volume_score = -1.5; ai_opinions['成交量分析'] = '❌ 價跌量爆/MFI弱，空頭壓力'
    else: volume_score = 0; ai_opinions['成交量分析'] = '⚠️ 量能中性或價量背離'
    
    # --- 4. 融合總分 ---
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
    """
    計算並彙整技術指標狀態，包含新加入的核心指標。
    """
    if df.empty or len(df.dropna(subset=['EMA_50', 'MACD_Hist', 'RSI', 'ADX'])) < 51: return pd.DataFrame()
    df_clean = df.dropna().copy()
    if df_clean.empty: return pd.DataFrame()
    last_row = df_clean.iloc[-1]
    prev_row = df_clean.iloc[-2] if len(df_clean) >= 2 else last_row
    
    indicators = {
        '價格 vs. MA/VWAP': last_row['Close'],
        'RSI (14) 動能': last_row.get('RSI', np.nan),
        'MACD 柱狀圖': last_row.get('MACD_Hist', np.nan),
        'StochRSI/CCI': last_row.get('StochRSI', np.nan),
        'ADX (14) 趨勢強度': last_row.get('ADX', np.nan),
        'Ichimoku 轉換線/基準線': last_row.get('Ichimoku_Convert', np.nan),
        '資金流量 (MFI)': last_row.get('MFI', np.nan),
        'ATR (14) 波動性': last_row.get('ATR', np.nan), 
    }
    data = []
    for name, value in indicators.items():
        conclusion, color = "", "grey"
        if pd.isna(value):
            data.append([name, "N/A", "數據不足，無法計算", "blue"])
            continue

        if 'MA/VWAP' in name:
            ema_50, hma_14, vwap = last_row.get('EMA_50', np.nan), last_row.get('HMA_14', np.nan), last_row.get('VWAP', np.nan)
            if all(not pd.isna(e) for e in [ema_50, hma_14, vwap]):
                if last_row['Close'] > ema_50 and hma_14 > ema_50: conclusion, color = "**強多頭：多MA/HMA/VWAP上方**", "red"
                elif last_row['Close'] < ema_50 and hma_14 < ema_50: conclusion, color = "**強空頭：價格在MA/HMA/VWAP下方**", "green"
                else: conclusion, color = "中性：盤整或趨勢發展中", "blue"
            else: conclusion, color = "數據不足，趨勢判斷困難", "grey"
        elif 'RSI' in name:
            if value > 70: conclusion, color = "警告：超買區域", "green"
            elif value < 30: conclusion, color = "強化：超賣區域", "red"
            elif value > 50: conclusion, color = "多頭：RSI > 50", "red"
            else: conclusion, color = "空頭：RSI < 50", "green"
        elif 'MACD' in name:
            if value > 0 and value > prev_row.get('MACD_Hist', 0): conclusion, color = "強化：多頭動能增強", "red"
            elif value < 0 and value < prev_row.get('MACD_Hist', 0): conclusion, color = "削弱：空頭動能增強", "green"
            else: conclusion, color = "中性：動能盤整", "orange"
        elif 'StochRSI/CCI' in name:
            cci = last_row.get('CCI', np.nan)
            if last_row.get('StochRSI', 0.5) > 0.8 or cci > 100: conclusion, color = "警告：StochRSI/CCI過熱", "green"
            elif last_row.get('StochRSI', 0.5) < 0.2 or cci < -100: conclusion, color = "強化：StochRSI/CCI超賣", "red"
            else: conclusion, color = "中性：動能中性", "blue"
        elif 'ADX' in name:
            if value >= 25 and last_row.get('ADX_DI_P', 0) > last_row.get('ADX_DI_N', 0): conclusion, color = "強趨勢：多頭趨勢確認", "red"
            elif value >= 25 and last_row.get('ADX_DI_P', 0) < last_row.get('ADX_DI_N', 0): conclusion, color = "強趨勢：空頭趨勢確認", "green"
            else: conclusion, color = "盤整：弱勢或橫盤", "blue"
        elif 'Ichimoku' in name:
            if last_row['Ichimoku_Convert'] > last_row.get('Ichimoku_Base', np.nan): conclusion, color = "多頭：轉換線在基準線上方", "red"
            elif last_row['Ichimoku_Convert'] < last_row.get('Ichimoku_Base', np.nan): conclusion, color = "空頭：轉換線在基準線下方", "green"
            else: conclusion, color = "中性：盤整", "blue"
        elif '資金流量' in name:
            if value > 80: conclusion, color = "警告：MFI資金流入過熱", "green"
            elif value < 20: conclusion, color = "強化：MFI資金流出過度", "red"
            else: conclusion, color = "中性：資金流中性", "blue"
        elif 'ATR' in name:
            avg_atr = df_clean.get('ATR', pd.Series()).iloc[-30:].mean()
            if value > avg_atr * 1.5: conclusion, color = "警告：極高波動性", "green"
            else: conclusion, color = "中性：正常波動", "blue"

        data.append([name, value, conclusion, color])
    
    return pd.DataFrame(data, columns=['指標名稱', '最新值', '分析結論', '顏色']).set_index('指標名稱')

# 繪圖函數 (強化圖表分區)
def create_comprehensive_chart(df, symbol, period_key):
    df_clean = df.dropna()
    if df_clean.empty: return go.Figure()

    # 調整圖表分區：新增一列用於震盪指標，並調整高度比例
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        row_heights=[0.45, 0.15, 0.15, 0.15, 0.1], 
                        specs=[[{"secondary_y": True}], [{}], [{}], [{}], [{}]])
    
    # --- Row 1: K線圖, 趨勢線 (MA, VWAP, Ichimoku) ---
    fig.add_trace(go.Candlestick(x=df_clean.index, open=df_clean['Open'], high=df_clean['High'], low=df_clean['Low'], close=df_clean['Close'], name='K線'), row=1, col=1)
    
    # 核心 EMA
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_10'], line=dict(color='orange', width=1), name='EMA 10'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_50'], line=dict(color='cyan', width=1.5), name='EMA 50'), row=1, col=1)
    # 新增 MA
    if 'LWMA_20' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['LWMA_20'], line=dict(color='yellow', width=1, dash='dot'), name='LWMA 20'), row=1, col=1)
    if 'HMA_14' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['HMA_14'], line=dict(color='lime', width=1.5, dash='dash'), name='HMA 14'), row=1, col=1)
    if 'VWAP' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['VWAP'], line=dict(color='magenta', width=1, dash='dash'), name='VWAP'), row=1, col=1)
    # Ichimoku 
    if 'Ichimoku_Convert' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['Ichimoku_Convert'], line=dict(color='red', width=1), name='一目-轉換線'), row=1, col=1)
    if 'Ichimoku_Base' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['Ichimoku_Base'], line=dict(color='blue', width=1), name='一目-基準線'), row=1, col=1)
    
    # 成交量 (副Y軸)
    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['Volume'], marker_color='grey', name='成交量', opacity=0.3), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="價格", row=1, col=1); fig.update_yaxes(title_text="成交量", secondary_y=True, row=1, col=1, showgrid=False)
    
    # --- Row 2: MACD ---
    macd_colors = np.where(df_clean.get('MACD_Hist', pd.Series()) >= 0, '#cc0000', '#1e8449')
    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean.get('MACD_Hist', pd.Series()), marker_color=macd_colors, name='MACD Hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.get('MACD_Line', pd.Series()), line=dict(color='blue', width=1), name='MACD 線'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.get('MACD_Signal', pd.Series()), line=dict(color='orange', width=1), name='Signal 線'), row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1, zeroline=True)
    
    # --- Row 3: RSI/ADX/DMI ---
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.get('RSI', pd.Series()), line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.get('ADX', pd.Series()), line=dict(color='#cc6600', width=1.5, dash='dot'), name='ADX'), row=3, col=1)
    if 'ADX_DI_P' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['ADX_DI_P'], line=dict(color='red', width=1), name='+DI'), row=3, col=1)
    if 'ADX_DI_N' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['ADX_DI_N'], line=dict(color='green', width=1), name='-DI'), row=3, col=1)
    fig.update_yaxes(title_text="RSI/ADX", range=[0, 100], row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5); fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
    
    # --- Row 4: 震盪指標 (CCI, StochRSI, Williams %R) ---
    if 'CCI' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['CCI'], line=dict(color='blue', width=1), name='CCI'), row=4, col=1)
    # StochRSI 範圍 0-1
    if 'StochRSI' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['StochRSI'] * 100, line=dict(color='red', width=1), name='StochRSI'), row=4, col=1)
    # Williams %R 範圍 0-(-100)，取反向繪製 0-100
    if 'Williams_R' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['Williams_R'] * -1, line=dict(color='purple', width=1, dash='dot'), name='W%R'), row=4, col=1)
    
    fig.update_yaxes(title_text="震盪指標", row=4, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="red", row=4, col=1, opacity=0.3)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1, opacity=0.3)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1, opacity=0.3)
    fig.add_hline(y=0, line_dash="dash", line_color="green", row=4, col=1, opacity=0.3)

    # --- Row 5: OBV/MFI (成交量指標) ---
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean.get('OBV', pd.Series()), line=dict(color='green', width=1.5), name='OBV'), row=5, col=1)
    # MFI (範圍 0-100) 
    if 'MFI' in df_clean.columns: fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MFI'] * df_clean['OBV'].max() / 100, line=dict(color='orange', width=1), name='MFI (Scaled)'), row=5, col=1)
    
    fig.update_yaxes(title_text="量能指標", row=5, col=1)
    
    fig.update_layout(title_text=f"AI 融合分析圖表 - {symbol} ({period_key})", height=1000, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ... (run_backtest 保持不變)
def run_backtest(df, initial_capital=100000, commission_rate=0.001):
    # 檢查 SMA_20 和 EMA_50 是否有足夠的非空值
    if df.empty or len(df.dropna(subset=['SMA_20', 'EMA_50'])) < 51: return {"total_return": 0, "win_rate": 0, "max_drawdown": 0, "total_trades": 0, "message": "數據不足或指標無法計算"}
    # ... (回測邏輯保持不變)
    data = df.dropna(subset=['SMA_20', 'EMA_50']).copy()
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
            # 開倉
            position = 1; buy_price = data['Close'].iloc[i]; capital = current_capital * (1 - commission_rate)
        elif data['Signal'].iloc[i] == -1 and position == 1:
            # 平倉
            profit = (data['Close'].iloc[i] - buy_price) / buy_price
            trades.append(1 if profit > 0 else 0)
            capital = current_capital * (1 - commission_rate) * (1 + profit) 
            position = 0
            buy_price = 0

    # 結算最後一筆交易
    if position == 1 and len(capital_curve) > 0:
        profit = (data['Close'].iloc[-1] - buy_price) / buy_price
        trades.append(1 if profit > 0 else 0); 
        # 修正此處計算邏輯
        capital = capital_curve[-1] * (1 + profit) / (data['Close'].iloc[-1] / buy_price)
        capital_curve[-1] = capital

    total_return = (capital / initial_capital - 1) * 100 if len(capital_curve) > 0 else 0
    win_rate = (sum(trades) / len(trades)) * 100 if trades else 0
    
    capital_s = pd.Series(capital_curve, index=data.index[:len(capital_curve)])
    max_drawdown = (capital_s / capital_s.cummax() - 1).min() * 100 if not capital_s.empty else 0
    
    return { "total_return": round(total_return, 2), "win_rate": round(win_rate, 2), "max_drawdown": round(abs(max_drawdown), 2), "total_trades": len(trades), "message": f"回測區間 {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}。", "capital_curve": capital_s }


# ... (main 函數保持不變，因為 UI 邏輯在上次已調整)

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
    if 'sidebar_search_input' not in st.session_state: st.session_state['sidebar_search_input'] = "MSFT" # 修正為 MSFT
    
    # 確保 MSFT 在 FULL_SYMBOLS_MAP 中，若無則添加 (用於避免初始化錯誤)
    if "MSFT" not in FULL_SYMBOLS_MAP:
        FULL_SYMBOLS_MAP["MSFT"] = {"name": "微軟 (Microsoft)", "keywords": ["微軟", "Microsoft", "MSFT", "雲端", "AI"]}
        
    main()
