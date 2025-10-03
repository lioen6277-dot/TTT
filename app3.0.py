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

warnings.filterwarnings('ignore')



st.set_page_config(
    page_title="AI趨勢分析📈", 
    page_icon="🚀", 
    layout="wide"
)

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

# ==============================================================================
# 2. 數據獲取與指標計算函數
# ==============================================================================

@st.cache_data(ttl=600)
def get_historical_data(symbol, period="5y", interval="1d"):
    """從 Yahoo Finance 獲取歷史數據"""
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            return pd.DataFrame(), "錯誤：找不到數據或交易代碼無效。"
        
        # 數據清理與準備
        df = data.copy()
        df.columns = [col.capitalize() for col in df.columns]
        df.index.name = 'Date'
        
        # 關鍵：使用前向填充 (ffill) 處理數據空值，確保技術指標計算穩定
        df = df.ffill() 
        df = df.dropna()
        
        if df.empty:
            return pd.DataFrame(), "錯誤：數據點太少，無法計算指標。"
            
        # 確保價格是數字格式
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df, None
    except Exception as e:
        return pd.DataFrame(), f"錯誤：無法獲取數據，請檢查代碼或網路。({e})"

@st.cache_data(ttl=600)
def calculate_technical_indicators(df):
    """計算所有必需的技術指標"""
    if df.empty:
        return df
    
    # === 趨勢與均線 ===
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
    
    # === 動能與強度 ===
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=9)
    
    # === 波動性與風險 ===
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # === 籌碼與量能 (OBV) ===
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    # === K線形態 (Heikin Ashi) ===
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    for i in range(len(df)):
        if i == 0:
            df.loc[df.index[i], 'HA_Open'] = df.loc[df.index[i], 'Open']
        else:
            df.loc[df.index[i], 'HA_Open'] = (df.loc[df.index[i-1], 'HA_Open'] + df.loc[df.index[i-1], 'HA_Close']) / 2
    
    # 使用 Heikin Ashi 繪圖時，需將 Open, Close, High, Low 替換為 HA_Open, HA_Close, High, Low
    
    return df.dropna().copy()

@st.cache_data(ttl=86400) # 每日更新一次 VIX
def get_vix_context():
    """獲取 VIX 恐慌指數，作為情緒專家數據"""
    try:
        vix_df = yf.download("^VIX", period="1d", interval="1d")
        if not vix_df.empty:
            return vix_df['Close'].iloc[-1]
        return None
    except:
        return None

@st.cache_data(ttl=86400) # 每日更新一次長期數據
def get_long_term_context(symbol):
    """獲取 1D 週期數據的 EMA 200 和 ADX，作為多時間框架濾鏡"""
    try:
        df, err = get_historical_data(symbol, period="3y", interval="1d")
        if df.empty:
            return None, None
        
        df = calculate_technical_indicators(df)
        if df.empty:
            return None, None
            
        latest_row = df.iloc[-1]
        return latest_row.get('EMA_200'), latest_row.get('ADX')
    except Exception as e:
        print(f"Error fetching long term context: {e}")
        return None, None
    
# 假設您有一個 get_fa_rating 函數來獲取基本面評分
def get_fa_rating(symbol):
    """模擬/實際獲取基本面評分 (0-9)"""
    # 這裡應該連接到您的基本面數據庫或 API
    # 為了範例運行，我們使用一個隨機的或基於代碼的模擬評分
    if symbol in ["NVDA", "BTC-USD"]:
        rating = 8 
        message = "評級高：基本面強勁或趨勢看好。"
    elif symbol == "2330.TW":
        rating = 9
        message = "評級極高：護城河寬廣，基本面頂尖。"
    else:
        rating = 5
        message = "評級中性：基本面穩健，但缺乏爆發性成長因子。"

    return {'Combined_Rating': rating, 'Message': message}


# ==============================================================================
# 3. 核心 AI 融合分析專家函數 (已修正為防禦性編程)
# ==============================================================================

def generate_expert_fusion_signal(df, fa_rating, is_long_term=True, currency_symbol="$", long_term_ema_200=None, long_term_adx=None, latest_vix=None):
    """
    專家融合分析：結合 MA, 動能, 強度, K線, 基本面, 籌碼, MTF, VIX 指標，給出融合信號。
    """
    # 🚀 關鍵修正點 1: 新增 try 區塊，確保函數在遇到錯誤時不會返回 None
    try: 
        
        df_clean = df.dropna().copy()
        if df_clean.empty or len(df_clean) < 2:
            return {'action': '數據不足', 'score': 0, 'confidence': 0, 'strategy': '無法評估', 'entry_price': 0, 'take_profit': 0, 'stop_loss': 0, 'current_price': 0, 'expert_opinions': {}, 'atr': 0, 'signal_list': []}

        last_row = df_clean.iloc[-1]
        prev_row = df_clean.iloc[-2]
        current_price = last_row['Close']
        atr_value = last_row['ATR']
        adx_value = last_row['ADX'] 
        
        expert_opinions = {}

        # 確保所有變數初始化
        ma_score = 0
        volume_score = 0
        momentum_score = 0 
        strength_score = 0
        kline_score = 0
        mtf_score = 0
        vix_score = 0
        
        # === 1. 趨勢專家 (MA) 邏輯 ===
        ema_10 = last_row['EMA_10']
        ema_50 = last_row['EMA_50']
        ema_200 = last_row['EMA_200']
        
        prev_10_above_50 = prev_row['EMA_10'] > prev_row['EMA_50']
        curr_10_above_50 = ema_10 > ema_50
        
        if not prev_10_above_50 and curr_10_above_50:
            ma_score = 3.5 
            expert_opinions['趨勢分析 (MA 交叉)'] = "**🚀 黃金交叉 (GC)**：EMA 10 向上穿越 EMA 50，強勁看漲信號！"
        elif prev_10_above_50 and not curr_10_above_50:
            ma_score = -3.5 
            expert_opinions['趨勢分析 (MA 交叉)'] = "**💀 死亡交叉 (DC)**：EMA 10 向下穿越 EMA 50，強勁看跌信號！"
        elif ema_10 > ema_50 and ema_50 > ema_200:
            ma_score = 2.0 
            expert_opinions['趨勢分析 (MA 排列)'] = "強勢多頭排列：**10 > 50 > 200**，趨勢結構穩固。"
        elif ema_10 < ema_50 and ema_50 < ema_200:
            ma_score = -2.0 
            expert_opinions['趨勢分析 (MA 排列)'] = "強勢空頭排列：**10 < 50 < 200**，趨勢結構崩潰。"
        elif curr_10_above_50:
            ma_score = 1.0
            expert_opinions['趨勢分析 (MA 排列)'] = "多頭：EMA 10 位於 EMA 50 之上。"
        else:
            ma_score = -1.0
            expert_opinions['趨勢分析 (MA 排列)'] = "空頭：EMA 10 位於 EMA 50 之下。"

        # === 2. 籌碼專家 (OBV) 邏輯 (簡化) ===
        obv = last_row['OBV']
        prev_obv = prev_row['OBV']
        if obv > prev_obv:
            volume_score = 1.0
            expert_opinions['籌碼專家 (OBV)'] = "籌碼面：OBV 上升，主力資金流入，趨勢有效性高。"
        elif obv < prev_obv:
            volume_score = -1.0
            expert_opinions['籌碼專家 (OBV)'] = "籌碼面：OBV 下降，主力資金流出，趨勢有效性低。"
        else:
            volume_score = 0
            expert_opinions['籌碼專家 (OBV)'] = "籌碼面：OBV 持平，市場處於盤整或觀望。"

        # === 3. 動能專家 (RSI) 邏輯 (簡化) ===
        rsi_value = last_row['RSI']
        if rsi_value > 70:
            momentum_score = -1.5
            expert_opinions['動能專家 (RSI 14)'] = f"警惕：RSI {rsi_value:.2f} **超買區**，短線動能可能過熱。"
        elif rsi_value < 30:
            momentum_score = 1.5
            expert_opinions['動能專家 (RSI 14)'] = f"機會：RSI {rsi_value:.2f} **超賣區**，短線反彈動能積累。"
        elif rsi_value > 50:
            momentum_score = 0.5
            expert_opinions['動能專家 (RSI 14)'] = "多頭：RSI > 50，多頭動能佔優。"
        else:
            momentum_score = -0.5
            expert_opinions['動能專家 (RSI 14)'] = "空頭：RSI < 50，空頭動能佔優。"

        # === 4. 強度專家 (MACD + ADX) 邏輯 ===
        macd_diff = last_row['MACD_Hist']
        prev_macd_diff = prev_row['MACD_Hist']
        
        strength_score = 0
        if macd_diff > 0 and macd_diff > prev_macd_diff:
            strength_score += 1.5
            expert_opinions['趨勢強度 (MACD)'] = "多頭：MACD 柱狀圖放大，多頭動能強勁。"
        elif macd_diff < 0 and macd_diff < prev_macd_diff:
            strength_score -= 1.5
            expert_opinions['趨勢強度 (MACD)'] = "空頭：MACD 柱狀圖放大，空頭動能強勁。"
        else:
            strength_score += 0
            expert_opinions['趨勢強度 (MACD)'] = "中性：MACD 柱狀圖收縮，動能盤整。"

        if adx_value > 25:
            strength_score *= 1.5 
            expert_opinions['趨勢強度 (ADX 9)'] = f"**確認強趨勢**：ADX {adx_value:.2f} > 25，信號有效性高。"
        else:
            expert_opinions['趨勢強度 (ADX 9)'] = f"盤整：ADX {adx_value:.2f} < 25，信號有效性降低。"

        # === 5. K線形態專家 (HA) 邏輯 ===
        kline_score = 0
        is_ha_up_bar = last_row['Close'] >= last_row['Open'] 
        is_ha_strong_bull = is_ha_up_bar and (last_row['Low'] == last_row['Open'])
        is_ha_strong_bear = (not is_ha_up_bar) and (last_row['High'] == last_row['Open'])

        if is_ha_strong_bull:
            kline_score = 1.5 
            expert_opinions['K線形態分析'] = "**🚀 HA 強勢多頭**：陽線且無下影線，多頭趨勢**非常穩定**。"
        elif is_ha_strong_bear:
            kline_score = -1.5 
            expert_opinions['K線形態分析'] = "**💀 HA 強勢空頭**：陰線且無上影線，空頭趨勢**非常穩定**。"
        elif is_ha_up_bar:
            kline_score = 0.5
            expert_opinions['K線形態分析'] = "HA 陽線：趨勢偏多，但有影線（波動或修正）。"
        else:
            kline_score = -0.5
            expert_opinions['K線形態分析'] = "HA 陰線：趨勢偏空，但有影線（波動或修正）。"
            
        # === 6. MTF 多時間框架濾鏡邏輯 ===
        if long_term_ema_200 is not None:
            if current_price > long_term_ema_200:
                mtf_score += 1.5 
                expert_opinions['多時間框架 (MTF)'] = f"長期趨勢：**牛市確認** (價格 > 1D EMA 200 {long_term_ema_200:.2f})。"
            elif current_price < long_term_ema_200:
                mtf_score -= 1.5
                expert_opinions['多時間框架 (MTF)'] = f"長期趨勢：**熊市警告** (價格 < 1D EMA 200 {long_term_ema_200:.2f})。"
            
            # 長期 ADX 確認趨勢強度，權重加倍
            if long_term_adx is not None and long_term_adx > 25:
                mtf_score *= 2.0 
            else:
                expert_opinions['多時間框架 (MTF)'] = expert_opinions.get('多時間框架 (MTF)', '長期趨勢：中性/盤整，MTF 濾鏡權重未加倍。')
        else:
            expert_opinions['多時間框架 (MTF)'] = "MTF 數據缺失，不影響融合評分。"

        # === 7. 情緒專家 (VIX) 邏輯 ===
        VIX_THRESHOLD_HIGH = 25
        VIX_THRESHOLD_LOW = 15
        
        if latest_vix is not None:
            if latest_vix > VIX_THRESHOLD_HIGH:
                vix_score += 1.0 # VIX 高表示恐慌，逆勢看多
                expert_opinions['情緒專家 (VIX)'] = f"恐慌：VIX {latest_vix:.2f} > 25，市場過度恐慌，有利於多頭入場修正。"
            elif latest_vix < VIX_THRESHOLD_LOW:
                vix_score -= 1.0 # VIX 低表示貪婪/自滿，逆勢看空
                expert_opinions['情緒專家 (VIX)'] = f"自滿：VIX {latest_vix:.2f} < 15，市場過度自滿，警惕空頭修正。"
            else:
                expert_opinions['情緒專家 (VIX)'] = f"中性：VIX {latest_vix:.2f} 介於 15-25 之間，情緒穩定。"
        else:
            expert_opinions['情緒專家 (VIX)'] = "情緒指標 VIX 數據缺失。"
        
        # === 8. 基本面 (FA) 邏輯 ===
        fa_normalized_score = ((fa_rating['Combined_Rating'] / 9) * 6) - 3 if fa_rating['Combined_Rating'] > 0 else -3 

        # ----------------------------------------------------
        # 9. 【專家融合計算與行動判斷】
        # ----------------------------------------------------
        fusion_score = (
            ma_score 
            + momentum_score 
            + strength_score 
            + kline_score 
            + fa_normalized_score 
            + volume_score
            + mtf_score     
            + vix_score     
        )

        # ----------------------------------------------------
        # 10. 【動態風險管理 (R:R 2:1)】
        # ----------------------------------------------------
        ADX_TREND_THRESHOLD = 25
        BASE_ATR_MULTIPLIER = 2.0 
        
        if adx_value >= 40: 
            atr_multiplier = 1.0 
            expert_opinions['風險專家 (ATR)'] = f"風險管理：**超強趨勢** (ADX >= 40)，使用 **1.0x ATR** 止損 (R:R 2:1)。"
        elif adx_value > ADX_TREND_THRESHOLD:
            atr_multiplier = 1.5 
            expert_opinions['風險專家 (ATR)'] = f"風險管理：**強趨勢** (ADX > 25)，使用 **1.5x ATR** 止損 (R:R 2:1)。"
        else:
            atr_multiplier = BASE_ATR_MULTIPLIER 
            expert_opinions['風險專家 (ATR)'] = f"風險管理：**弱勢/盤整** (ADX <= 25)，使用 **2.0x ATR** 止損 (R:R 2:1)。"
        
        risk_unit = atr_value * atr_multiplier 
        reward_unit = risk_unit * 2.0 

        # ----------------------------------------------------
        # 11. 【最終行動與策略執行】
        # ----------------------------------------------------
        MAX_SCORE = 18.25 # 總分權重上限
        confidence = min(100, max(0, 50 + (fusion_score / MAX_SCORE) * 50))
        
        action = "觀望 (Neutral)"
        if fusion_score >= 4.0: action = "強力買入 (Strong Buy)"
        elif fusion_score >= 1.0: action = "中性偏買 (Hold/Buy)"
        elif fusion_score <= -4.0: action = "強力賣出 (Strong Sell)"
        elif fusion_score <= -1.0: action = "中性偏賣 (Hold/Sell)"

        entry_buffer = atr_value * 0.3
        price_format = ".4f" if current_price < 100 and not currency_symbol == 'NT$' else ".2f"
        
        entry = current_price
        stop_loss = 0
        take_profit = 0
        strategy_desc = "市場信號混亂，建議等待趨勢明朗或在區間內操作。"
        
        if action in ["強力買入 (Strong Buy)", "中性偏買 (Hold/Buy)"]:
            entry = current_price - entry_buffer
            stop_loss = entry - risk_unit 
            take_profit = entry + reward_unit 
            strategy_desc = f"基於{action}信號，建議在 **{currency_symbol}{entry:{price_format}}** 範圍內尋找支撐或等待回調進場。"
        elif action in ["強力賣出 (Strong Sell)", "中性偏賣 (Hold/Sell)"]:
            entry = current_price + entry_buffer
            stop_loss = entry + risk_unit 
            take_profit = entry - reward_unit 
            strategy_desc = f"基於{action}信號，建議在 **{currency_symbol}{entry:{price_format}}** 範圍內尋找阻力或等待反彈後進場。"
        
        # ----------------------------------------------------
        # 12. 【報告總結與回傳】
        # ----------------------------------------------------
        mtf_opinion = expert_opinions.get('多時間框架 (MTF)', 'MTF 濾鏡數據缺失。')
        vix_opinion = expert_opinions.get('情緒專家 (VIX)', '情緒指標數據缺失。')
        volume_opinion = expert_opinions.get('籌碼專家 (OBV)', '籌碼面數據缺失。')
        
        # 從 opinion 中提取摘要
        volume_summary = volume_opinion.split('：')[-1].strip()
        mtf_summary = mtf_opinion.split('：')[-1].strip()
        vix_summary = vix_opinion.split('：')[-1].strip()

        total_signal_list = [
            "--- 評分細項 (Score Breakdown) ---", 
            f"趨勢均線評分 (MA): {ma_score:.1f} / 3.5",
            f"動能評分 (RSI): {momentum_score:.1f} / 2.0", 
            f"強度評分 (MACD+ADX): {strength_score:.2f} / 2.25", 
            f"K線形態評分 (HA K-Line): {kline_score:.1f} / 1.5", 
            f"**多時間框架濾鏡 (MTF): {mtf_score:.2f} / 3.0 ({mtf_summary})**", 
            f"**情緒評分 (VIX): {vix_score:.1f} / 1.0 ({vix_summary})**", 
            "--- 基本面與籌碼面 ---",
            f"基本面評分 (FA Score): {fa_rating['Combined_Rating']:.1f} / 9.0 ({fa_rating.get('Message', '數據缺失')})",
            f"籌碼面評分 (Volume Score): {volume_score:.1f} / 2.0 ({volume_summary})",
            f"風險單位 (Risk Unit): {currency_symbol}{risk_unit:{price_format}} ({atr_multiplier:.1f}x ATR)" 
        ]
        
        def format_price(p):
            if p is None or p == 0:
                return 0
            return round(p, 4) if current_price < 100 else round(p, 2)
        
        return {
            'action': action,
            'score': fusion_score,
            'confidence': confidence,
            'strategy': strategy_desc,
            'entry_price': format_price(entry),
            'take_profit': format_price(take_profit),
            'stop_loss': format_price(stop_loss),
            'current_price': format_price(current_price),
            'expert_opinions': expert_opinions,
            'atr': format_price(atr_value),
            'signal_list': total_signal_list,
            'currency_symbol': currency_symbol
        }
    
    # 🚀 關鍵修正點 2: except 區塊處理錯誤
    except Exception as e: 
        # 捕獲所有意外錯誤，並返回一個包含錯誤信息的安全字典
        print(f"致命錯誤發生在專家融合分析中: {e}") 
        return {
            'action': '分析失敗', 
            'score': 0, 
            'confidence': 0, 
            'strategy': f'發生例外錯誤：{e.__class__.__name__}', 
            'entry_price': 0, 
            'take_profit': 0, 
            'stop_loss': 0, 
            'current_price': 0, 
            'expert_opinions': {'錯誤報告': str(e)}, 
            'atr': 0,
            'signal_list': [f"分析引擎崩潰：{e.__class__.__name__} - 請檢查數據或聯繫技術支援。"]
        }


# ==============================================================================
# 4. 圖表繪製函數 (待完整優化)
# ==============================================================================

def create_comprehensive_chart(df, analysis, symbol, period_key):
    df_clean = df.dropna().copy()
    if df_clean.empty: 
        st.info("數據不足，無法繪製圖表。")
        return None

    # 設定 K 線數據 (使用 Heikin Ashi)
    # 由於 HA Open/Close 已經計算，我們直接使用它們來繪製 K 線圖
    
    fig = make_subplots(rows=4, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.55, 0.15, 0.15, 0.15], 
                        subplot_titles=(f"{symbol} 價格走勢 (週期: {period_key})", 
                                        "MACD 動能指標", 
                                        "RSI/ADX 強弱與趨勢指標", 
                                        "OBV 籌碼/量能趨勢"))
    
    # === Subplot 1: 價格 K 線與均線 ===
    
    # 繪製 K 線圖 (使用 Heikin Ashi)
    fig.add_trace(go.Candlestick(x=df_clean.index,
                                open=df_clean['HA_Open'],
                                high=df_clean['High'],
                                low=df_clean['Low'],
                                close=df_clean['HA_Close'],
                                name='K線 (HA)',
                                increasing_line_color='green', 
                                decreasing_line_color='red'), row=1, col=1)

    # 繪製均線
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_10'], mode='lines', name='EMA 10', line=dict(color='yellow')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_50'], mode='lines', name='EMA 50', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_200'], mode='lines', name='EMA 200', line=dict(color='blue')), row=1, col=1)

    # 繪製布林通道
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['BB_High'], mode='lines', name='BB Upper', line=dict(color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['BB_Low'], mode='lines', name='BB Lower', line=dict(color='gray', dash='dot')), row=1, col=1)
    
    # 繪製當前價格、TP/SL 點位
    if analysis and analysis.get('action') not in ['分析失敗', '數據不足']:
        current_price = analysis['current_price']
        stop_loss = analysis['stop_loss']
        take_profit = analysis['take_profit']
        entry_price = analysis['entry_price']
        
        fig.add_hline(y=current_price, line_dash="solid", line_color="purple", row=1, col=1, name="Current")
        fig.add_hline(y=entry_price, line_dash="dot", line_color="yellow", row=1, col=1, name="Entry")
        fig.add_hline(y=take_profit, line_dash="dash", line_color="green", row=1, col=1, name="TP")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", row=1, col=1, name="SL")
    
    # === Subplot 2: MACD ===
    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['MACD_Hist'], name='MACD Hist', 
                         marker_color=np.where(df_clean['MACD_Hist'] > 0, 'rgba(0,128,0,0.7)', 'rgba(255,0,0,0.7)')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD'], mode='lines', name='MACD Line', line=dict(color='white')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='red')), row=2, col=1)

    # === Subplot 3: RSI/ADX ===
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['RSI'], mode='lines', name='RSI', line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['ADX'], mode='lines', name='ADX', line=dict(color='lightblue')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.add_hline(y=25, line_dash="dot", line_color="yellow", row=3, col=1) # ADX 強趨勢閾值

    # === Subplot 4: OBV ===
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['OBV'], mode='lines', name='OBV', line=dict(color='lime')), row=4, col=1)

    # === 佈局美化 ===
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=900,
        hovermode="x unified",
        title_font_size=18
    )
    
    # 隱藏子圖上的圖例，只在最上面顯示
    fig.for_each_trace(lambda t: t.update(showlegend=t.row==1))
    fig.update_yaxes(title_text="價格 / BB", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI / ADX", row=3, col=1)
    fig.update_yaxes(title_text="OBV", row=4, col=1)

    return fig


# ==============================================================================
# 5. 主應用程式邏輯
# ==============================================================================

def main():
    # --- 側邊欄：輸入與選擇 ---
    with st.sidebar:
        st.title("AI 趨勢分析儀 📈")
        
        # 選擇資產類別 (Asset Class)
        asset_class_options = ["美股/ETF", "台股/ETF", "加密貨幣"]
        selected_class = st.selectbox("選擇資產類別:", asset_class_options)
        
        # 根據類別過濾符號
        if selected_class == "美股/ETF":
            current_symbols = {k: v for k, v in FULL_SYMBOLS_MAP.items() if not k.endswith(('.TW', '-USD'))}
        elif selected_class == "台股/ETF":
            current_symbols = {k: v for k, v in FULL_SYMBOLS_MAP.items() if k.endswith('.TW')}
        else:
            current_symbols = {k: v for k, v in FULL_SYMBOLS_MAP.items() if k.endswith('-USD')}
            
        # 創建顯示名稱 (e.g., "NVDA (輝達)")
        display_options = [f"{s} ({d['name']})" for s, d in current_symbols.items()]
        
        # 選擇符號
        selected_display = st.selectbox(
            "選擇熱門標的：",
            options=display_options,
            index=display_options.index(f"{st.session_state['sidebar_search_input']} ({current_symbols.get(st.session_state['sidebar_search_input'], {'name': '未知'})['name']})" if st.session_state['sidebar_search_input'] in current_symbols else 0),
            key='selectbox_symbol'
        )
        
        # 解析出實際的代碼
        selected_symbol = selected_display.split(' ')[0]
        
        # 自由輸入框
        symbol_input = st.text_input(
            "或手動輸入交易代碼 (Symbol)：",
            value=selected_symbol,
            key='symbol_input'
        ).upper()
        
        # 同步狀態 (確保 selectbox 或 text_input 更改時， symbol_input 保持一致)
        if selected_symbol != st.session_state['symbol_input'] and 'selectbox_symbol' in st.session_state:
             st.session_state['symbol_input'] = selected_symbol
             
        # 週期選擇
        period_key = st.selectbox("選擇分析週期：", options=list(PERIOD_MAP.keys()), index=2)
        period, interval = PERIOD_MAP[period_key]
        
        # 執行按鈕放在主區域

    # --- 主區域：顯示結果 ---
    
    st.title(f"🚀 AI 專家融合分析：{st.session_state['symbol_input']}")
    
    # 確保 sidebar_search_input 保持最新狀態，用於下次進入應用程式時顯示
    st.session_state['sidebar_search_input'] = st.session_state['symbol_input']
    
    st.markdown("---")


    # 執行 AI 分析的邏輯區塊 (已修正)
    if st.button('📊 執行 AI 分析', use_container_width=True, type='primary'):
        
        # 1. 獲取數據與基本面
        with st.spinner(f"正在獲取 {st.session_state['symbol_input']} 歷史數據..."):
            raw_df, err = get_historical_data(st.session_state['symbol_input'], period=period, interval=interval)
            
        if not raw_df.empty:
            current_df = calculate_technical_indicators(raw_df)
            st.session_state['data_ready_df'] = current_df
            st.session_state['data_ready'] = True
            
            # 獲取基本面和長期背景
            fa_rating = get_fa_rating(st.session_state['symbol_input'])
            long_term_ema_200, long_term_adx = get_long_term_context(st.session_state['symbol_input'])
            latest_vix = get_vix_context() 

            # 2. 執行專家融合分析
            with st.spinner("正在執行 AI 專家融合分析..."):
                analysis = generate_expert_fusion_signal(
                    current_df, 
                    fa_rating, 
                    currency_symbol="$" if not st.session_state['symbol_input'].endswith('.TW') else "NT$",
                    long_term_ema_200=long_term_ema_200, 
                    long_term_adx=long_term_adx,
                    latest_vix=latest_vix 
                )

            # 🚀 關鍵修正：防禦性檢查，避免 TypeError
            if analysis is None or not isinstance(analysis, dict) or analysis.get('action') in ['分析失敗', '數據不足']:
                st.error("❌ 分析引擎無法產出結果！請檢查數據完整性或聯繫技術支援。")
                if isinstance(analysis, dict):
                    st.error(f"詳細錯誤資訊: {analysis.get('strategy', '未知的錯誤類型')}。錯誤報告: {analysis.get('expert_opinions', {}).get('錯誤報告', 'N/A')}")
                return 

            # 3. 儲存與顯示結果
            st.session_state['analysis_result'] = analysis
            st.success(f"✅ AI 分析完成！當前價格：{analysis['currency_symbol']}{analysis['current_price']:.2f}")

        else:
            st.error(err)
            st.session_state['data_ready'] = False
            return
    
    # --- 顯示上次分析結果 ---
    if st.session_state.get('data_ready') and st.session_state.get('analysis_result'):
        
        analysis = st.session_state['analysis_result']
        current_price = analysis['current_price']
        action = analysis['action']
        confidence = analysis['confidence']
        strategy_desc = analysis['strategy']
        stop_loss = analysis['stop_loss']
        take_profit = analysis['take_profit']
        entry_price = analysis['entry_price']
        signal_list = analysis['signal_list']
        expert_opinions = analysis['expert_opinions']
        currency_symbol = analysis['currency_symbol']
        
        
        # 1. 核心指標與策略
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("當前價格", f"{currency_symbol}{current_price:.2f}")
        with col2:
            color = "green" if "買" in action else ("red" if "賣" in action else "orange")
            st.markdown(f"**💡 AI 最終信號**")
            st.markdown(f"## <span style='color: {color};'>{action}</span>", unsafe_allow_html=True)
        with col3:
            st.metric("📈 策略信心", f"{confidence:.1f}%")
        with col4:
            st.markdown(f"**💰 風險報酬比**")
            st.markdown(f"## <span style='color: yellow;'>1 : 2.0</span>", unsafe_allow_html=True)

        st.markdown(f"---")
        
        # 2. TP/SL 與進場建議
        col_tp, col_entry, col_sl = st.columns(3)
        with col_entry:
            st.metric("建議進場價 (Entry)", f"{currency_symbol}{entry_price:.4f}")
        with col_tp:
            st.metric("✅ 止盈價 (Take Profit)", f"{currency_symbol}{take_profit:.4f}")
        with col_sl:
            st.metric("❌ 止損價 (Stop Loss)", f"{currency_symbol}{stop_loss:.4f}")
            
        st.info(strategy_desc)
        
        st.markdown(f"---")
        
        # 3. 圖表顯示
        st.subheader("📊 價格與指標詳情")
        
        current_df = st.session_state['data_ready_df']
        fig = create_comprehensive_chart(current_df, analysis, st.session_state['symbol_input'], period_key)
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"---")
        
        # 4. 專家意見與評分細項
        col_signal, col_expert = st.columns([1, 2])
        
        with col_signal:
            st.subheader("🚨 評分細項")
            for line in signal_list:
                st.markdown(line)

        with col_expert:
            st.subheader("🧠 專家意見總結")
            for opinion, message in expert_opinions.items():
                st.markdown(f"**{opinion}:** {message}")

    # --- 尾部免責聲明 ---
    st.markdown("---")
    st.markdown("⚠️ **綜合風險與免責聲明 (Risk & Disclaimer)**", unsafe_allow_html=True)
    st.markdown("本AI趨勢分析模型，是基於**量化集成學習 (Ensemble)**的專業架構。其分析結果**僅供參考用途**")
    st.markdown("投資涉及風險，所有交易決策應基於您個人的**獨立研究和財務狀況**，並強烈建議諮詢**專業金融顧問**。", unsafe_allow_html=True)
    st.markdown("📊 **數據來源:** Yahoo Finance | 🛠️ **技術指標:** TA 庫 | 💻 **APP優化:** 專業程式碼專家")


if __name__ == '__main__':
    # Streamlit Session State 初始化
    if 'last_search_symbol' not in st.session_state:
        st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state:
        st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state:
        st.session_state['sidebar_search_input'] = "2330.TW"
    if 'analysis_result' not in st.session_state:
        st.session_state['analysis_result'] = None
    if 'data_ready_df' not in st.session_state:
        st.session_state['data_ready_df'] = pd.DataFrame()
        
    main()

