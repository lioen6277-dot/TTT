import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import warnings
import time
from datetime import datetime, timedelta

# 抑制 ta 庫可能產生的 SettingWithCopyWarning
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 頁面配置與全局設定
# ==============================================================================

st.set_page_config(
    page_title="AI趨勢分析", 
    page_icon="📈", 
    layout="wide"
)

# 週期映射：(YFinance Period, YFinance Interval)
PERIOD_MAP = { 
    "30 分": ("60d", "30m"),  # 短線日內交易
    "4 小時": ("1y", "60m"),  # 短期波段
    "1 日": ("5y", "1d"),     # 中期投資
    "1 週": ("max", "1wk")    # 長期趨勢
}

# 🚀 您的【所有資產清單】
FULL_SYMBOLS_MAP = {
    # A. 美股核心 (US Stocks)
    "TSLA": {"name": "特斯拉", "keywords": ["電動車", "TSLA", "Tesla"]},
    "NVDA": {"name": "輝達", "keywords": ["AI", "NVDA", "Nvidia"]},
    "AAPL": {"name": "蘋果", "keywords": ["iPhone", "AAPL", "Apple"]},
    "MSFT": {"name": "微軟", "keywords": ["雲端", "MSFT", "Microsoft"]},
    "GOOG": {"name": "Google", "keywords": ["Alphabet", "GOOG"]},
    "AMZN": {"name": "亞馬遜", "keywords": ["電商", "AMZN", "Amazon"]},
    # B. 台股核心 (Taiwan Stocks)
    "2330.TW": {"name": "台積電", "keywords": ["晶圓", "2330", "TSMC"]},
    "2454.TW": {"name": "聯發科", "keywords": ["IC設計", "2454"]},
    "2317.TW": {"name": "鴻海", "keywords": ["電子代工", "2317"]},
    # C. 加密貨幣 (Crypto)
    "BTC-USD": {"name": "比特幣", "keywords": ["BTC", "Bitcoin"]},
    "ETH-USD": {"name": "以太幣", "keywords": ["ETH", "Ethereum"]},
}

# 將 FULL_SYMBOLS_MAP 拆分為子類別，用於快速選擇
US_STOCKS_MAP = {k: v for k, v in FULL_SYMBOLS_MAP.items() if not k.endswith(".TW") and not k.endswith("-USD")}
TW_STOCKS_MAP = {k: v for k, v in FULL_SYMBOLS_MAP.items() if k.endswith(".TW")}
CRYPTO_MAP = {k: v for k, v in FULL_SYMBOLS_MAP.items() if k.endswith("-USD")}

# ==============================================================================
# 2. 資料獲取與數據清洗 (提升穩定性與精準度)
# ==============================================================================

def get_data(symbol, period_key, progress_bar, max_retries=3):
    """
    從 yfinance 獲取歷史股價資料，包含指數退避重試機制及數據清洗。
    """
    period, interval = PERIOD_MAP.get(period_key, ("5y", "1d"))
    
    for attempt in range(max_retries):
        progress_bar.progress(20 + (attempt * 10), text=f"📥 正在獲取 {symbol} ({period_key}) 歷史數據... (嘗試 {attempt + 1}/{max_retries})")
        time.sleep(0.3)
        
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False, timeout=15)
            
            if data.empty:
                st.error(f"⚠️ **獲取 {symbol} 數據失敗:** 找不到該代碼的資料，請檢查代碼是否正確。")
                progress_bar.empty()
                return None
            
            # --- 數據清洗與標準化 (最終修正，確保 1D 數據結構) ---
            
            # 1. 處理 MultiIndex 結構 (日內數據常見問題，扁平化欄位)
            if isinstance(data.columns, pd.MultiIndex):
                # 提取最內層的欄位名稱 (例如從 (Ticker, OHLCV) 中提取 OHLCV)
                data.columns = [col[-1] if isinstance(col, tuple) else col for col in data.columns]
                
            # 2. 統一欄位命名格式
            data.columns = [c.replace('Adj Close', 'Adj_Close').replace(' ', '_').capitalize() for c in data.columns]
            
            # 3. 確保關鍵欄位存在並為 1D Series (修正所有維度錯誤)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

            for col in required_cols:
                if col in data.columns:
                    # 關鍵修正：將數據轉換為底層 NumPy 陣列並強制扁平化 (`.values.flatten()`)
                    # 然後重新構建為一個乾淨的 1D pandas Series，dtype 設為 float。
                    # 這樣就徹底解決了 'Data must be 1-dimensional' 的 ValueError。
                    data[col] = pd.Series(
                        data[col].values.flatten(), 
                        index=data.index, 
                        dtype=float 
                    )
                else:
                    st.warning(f"⚠️ 數據缺少關鍵欄位: {col}")


            # 4. 處理缺失值 (使用前一個有效值填充，然後移除剩餘的 NaN)
            data.fillna(method='ffill', inplace=True)
            data.dropna(inplace=True) 
            
            # 5. 處理 Volume 為零的異常數據 (可能為數據錯誤或停牌日)
            data = data[data['Volume'] > 0]
            
            # 6. 確保數據量足夠
            if len(data) < 20: # 至少需要20天計算短期均線
                st.error(f"⚠️ **數據量過少:** {symbol} 在所選週期 ({period_key}) 僅有 {len(data)} 筆數據，無法進行有效分析。請選擇更長的週期。")
                progress_bar.empty()
                return None
            
            data['Symbol'] = symbol
            progress_bar.progress(80, text=f"✅ 數據清洗與獲取成功！")
            time.sleep(0.3)
            return data

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                st.warning(f"網路或API錯誤，正在重試... (等待 {wait_time} 秒)")
                time.sleep(wait_time)
            else:
                st.error(f"❌ **獲取 {symbol} 數據時發生嚴重錯誤:** 已達最大重試次數。錯誤訊息: {e}")
                progress_bar.empty()
                return None
    return None

# ==============================================================================
# 3. 趨勢分析與指標計算 (強化技術與籌碼判斷標準)
# ==============================================================================

def calculate_technical_indicators(data):
    """計算技術指標 (SMA, RSI, MACD, Bollinger Bands, Volume SMA)"""
    if data is None or data.empty:
        return None

    df = data.copy()
    
    # 趨勢指標
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean() # 增加長期趨勢判斷
    
    # 動量指標: RSI
    # 由於數據已在 get_data 中被強制轉換為 1D Series，這裡將順利運行
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # 動量指標: MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # 波動性指標: Bollinger Bands (布林帶)
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    # 成交量指標: 20日成交量均線 (模擬籌碼/資金流的活躍度)
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    # 關鍵：確保數據能進行所有計算
    required_data_points = max(200, 50, 20) + 1 # 取最大窗口 + 1
    if len(df) < required_data_points:
         # 對於日線以上的週期，若數據不足 200，則不使用 SMA 200
         if len(df) < 200:
             df.drop(columns=['SMA_200'], errors='ignore', inplace=True) 

    return df.dropna()

def perform_ai_analysis(symbol, data, period_key):
    """
    模擬 AI 頂級專家 (專業操盤手/量化分析師) 的四維度分析輸出。
    
    融入了價格趨勢、動量、成交量(籌碼)、波動性等綜合標準。
    """
    
    st.subheader(f"目標標的：{symbol} ({period_key} 週期)")
    last_date = data.index[-1].strftime('%Y-%m-%d')
    last_close = data['Close'].iloc[-1]
    st.markdown(f"#### 最新收盤日: {last_date} | 最新收盤價: **${last_close:.2f}**")
    st.markdown("---")
    
    # --- 提取最新數據 ---
    sma_20 = data['SMA_20'].iloc[-1]
    sma_50 = data['SMA_50'].iloc[-1]
    sma_200 = data['SMA_200'].iloc[-1] if 'SMA_200' in data.columns else None
    
    rsi = data['RSI'].iloc[-1]
    macd_hist = data['MACD_Hist'].iloc[-1]
    
    last_volume = data['Volume'].iloc[-1]
    volume_sma_20 = data['Volume_SMA_20'].iloc[-1]
    
    bb_high = data['BB_High'].iloc[-1]
    bb_low = data['BB_Low'].iloc[-1]
    
    # 計算漲跌幅 (與前一日或前一週期比較)
    prev_close = data['Close'].iloc[-2]
    change_pct = ((last_close - prev_close) / prev_close) * 100
    
    # --- 1. 綜合趨勢評估 (模擬 AI 核心判斷) ---
    
    score = 0
    analysis_points = []
    
    # A. 價格與趨勢結構 (佔比 40%)
    trend_analysis = ""
    
    # 嚴格遵循技術分析的多頭排列標準
    if last_close > sma_20 and sma_20 > sma_50 and (sma_200 is None or sma_50 > sma_200):
        score += 4
        trend_analysis = "🚀 **強勢多頭結構**: 短中長期均線形成完美多頭排列，價格趨勢極為強勁。"
    elif last_close > sma_50 and last_close > sma_20:
        score += 2
        trend_analysis = "🟢 **多頭主導**: 股價站穩短期與中期均線之上，趨勢向上確立。"
    elif last_close < sma_20 and last_close > sma_50:
        score += 0
        trend_analysis = "🟡 **回檔整理**: 短期均線壓力大，但中期趨勢仍保持向上支撐。"
    else:
        score -= 3
        trend_analysis = "🔴 **空頭趨勢**: 股價跌破中短期均線，趨勢向下，賣壓沉重。"

    analysis_points.append(f"**價格結構 ({change_pct:+.2f}%)**: {trend_analysis}")

    # B. 動量指標 (佔比 30%)
    if macd_hist > 0 and rsi < 70:
        score += 3
        analysis_points.append(f"✅ **動量加強**: MACD 柱狀圖為正，且 RSI ({rsi:.1f}) 位於健康強勢區，動能持續積累。")
    elif macd_hist < 0 and rsi > 30:
        score -= 2
        analysis_points.append(f"❌ **動量衰退**: MACD 柱狀圖為負，動能減弱，RSI ({rsi:.1f}) 顯示市場缺乏買盤意願。")
    elif rsi >= 70:
        score -= 1
        analysis_points.append(f"⚠️ **動量過熱**: RSI 進入超買區 ({rsi:.1f})，需警惕短期獲利了結壓力。")
    else:
        score += 1
        analysis_points.append(f"💡 **動量中性**: 動能雖弱，但 RSI 位於超賣邊緣 ({rsi:.1f})，醞釀技術性反彈可能。")

    # C. 成交量與波動性 (模擬籌碼/資金流 - 佔比 30%)
    
    # 籌碼/資金流分析 (量價配合)
    volume_factor = 0
    if last_volume > volume_sma_20 * 1.5: # 顯著放量
        if change_pct > 0:
            volume_factor = 2
            analysis_points.append("💰 **資金追捧 (價漲量增)**: 成交量顯著放大 (+50%以上) 配合價格上漲，確認多頭積極建倉，為極佳的籌碼訊號。")
        else:
            volume_factor = -2
            analysis_points.append("💥 **恐慌性賣壓 (價跌量增)**: 成交量顯著放大 (+50%以上) 配合價格下跌，顯示大筆資金恐慌性流出。")
    elif last_volume < volume_sma_20 * 0.5: # 顯著縮量
        if change_pct > 0:
            volume_factor = -1
            analysis_points.append("📉 **上漲無量**: 價格上漲但成交量極度萎縮，追高意願不足，趨勢穩定性存疑。")
        else:
            volume_factor = 1
            analysis_points.append("🛡️ **下跌縮量**: 價格下跌但成交量極度萎縮，顯示拋壓減輕，可能即將築底反轉。")
            
    score += volume_factor

    # 波動性/風險評估 (布林帶)
    bb_range = bb_high - bb_low
    current_dist_to_mid = last_close - data['BB_Mid'].iloc[-1]
    
    if last_close >= bb_high * 0.99:
        score -= 1
        analysis_points.append("🛡️ **波動性警示**: 價格貼近布林帶上軌，短期超買，波動性風險高，不宜追高。")
    elif last_close <= bb_low * 1.01:
        score += 1
        analysis_points.append("🎯 **波動性機會**: 價格貼近布林帶下軌，短期超賣，有強力支撐或反轉機會。")
        
    # --- 2. 總結判斷與交易策略 ---
    
    st.markdown("#### 1. AI 綜合趨勢評估 (多維度整合)")
    
    final_score = score
    if final_score >= 5:
        tech_advice = "🟢 **【極度看好】重倉持有或積極建倉**"
        tech_summary = "綜合評估結果：趨勢、動量、資金流（籌碼）三維度均表現出**共振性的強勁多頭訊號**。價格結構完美，量價配合健康。建議積極參與，並將風險控制在第一道關鍵支撐位（例如 SMA 20）。"
    elif final_score >= 2:
        tech_advice = "🟡 **【觀望偏多】可逢低買入或持股待漲**"
        tech_summary = "綜合評估結果：整體趨勢向上，但部分指標（如 MACD 或成交量）略有分歧。建議等待價格回調至重要均線（如 SMA 50）時分批建倉，保持謹慎樂觀。"
    elif final_score >= 0:
        tech_advice = "⚪ **【中性觀望】等待訊號明確**"
        tech_summary = "綜合評估結果：市場處於盤整或多空交戰區。價格在均線之間震盪，動能指標中性。建議等待突破重要壓力或確認關鍵支撐後再決定方向。"
    else:
        tech_advice = "🔴 **【賣出或避險】空倉觀望或考慮賣出**"
        tech_summary = "綜合評估結果：趨勢結構已遭破壞，且伴隨籌碼鬆動或動能衰竭。賣壓主導市場。建議避免建倉，並考慮減持或避險。直到價格重新站上短期均線。"

    st.info(f"**最終交易建議：{tech_advice}**")
    st.markdown(tech_summary)
    
    st.markdown("##### 核心分析觀點：")
    for point in analysis_points:
        st.markdown(f"- {point}")
    
    st.markdown("---")
    
    # --- 3. 圖表展示 (增加布林帶) ---
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08, 
                        row_heights=[0.5, 0.15, 0.15, 0.2]) 
    
    # K線圖與均線/布林帶
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'],
                                 name='K線'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], line=dict(color='pink', width=0.5, dash='dot'), name='布林上軌'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], line=dict(color='cyan', width=0.5, dash='dot'), name='布林下軌'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], line=dict(color='blue', width=1), name='SMA 20 (短期趨勢)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='orange', width=1), name='SMA 50 (中期趨勢)'), row=1, col=1)
    if sma_200 is not None:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], line=dict(color='red', width=1, dash='dot'), name='SMA 200 (長期趨勢)'), row=1, col=1)
    
    # MACD 圖
    colors_macd = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='MACD 柱狀圖', marker_color=colors_macd), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], line=dict(color='blue', width=1), name='MACD 線'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], line=dict(color='red', width=1), name='Signal 線'), row=2, col=1)
    
    # RSI 圖
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='purple', width=1), name='RSI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data), line=dict(color='red', width=0.5, dash='dash'), name='超買(70)', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data), line=dict(color='green', width=0.5, dash='dash'), name='超賣(30)', showlegend=False), row=3, col=1)

    # Volume 圖 (籌碼/資金流)
    colors_volume = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='成交量', marker_color=colors_volume), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Volume_SMA_20'], line=dict(color='gray', width=1, dash='dot'), name='Volume SMA 20'), row=4, col=1)
    
    # 更新佈局
    fig.update_layout(
        title=f'${symbol}$ - 價格、動量、籌碼與波動性多維度趨勢',
        xaxis_rangeslider_visible=False,
        height=900,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="價格 / 布林帶", row=1, col=1)
    fig.update_yaxes(title_text="MACD (動量)", row=2, col=1)
    fig.update_yaxes(title_text="RSI (動量)", row=3, col=1)
    fig.update_yaxes(title_text="成交量 (資金流)", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 4. Streamlit 介面與邏輯 (狀態同步與精簡化)
# ==============================================================================

def update_search_input_from_select():
    """
    通用回調函式：根據當前選擇的資產類別更新 'sidebar_search_input' 的值。
    """
    asset_class = st.session_state.get('current_asset_class', '台股')
    
    if asset_class == "美股":
        selected_key = st.session_state.quick_select_us
    elif asset_class == "台股":
        selected_key = st.session_state.quick_select_tw
    else:
        selected_key = st.session_state.quick_select_crypto
        
    st.session_state['sidebar_search_input'] = selected_key


def sidebar_ui():
    """側邊欄 UI 元素和狀態管理"""
    st.sidebar.header("📈 參數設定")
    
    # 1. 初始化 Session State (確保最小化初始化邏輯)
    if 'last_search_symbol' not in st.session_state:
        st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state:
        st.session_state['data_ready'] = False
    # 關鍵：初始化時，sidebar_search_input 應與 last_search_symbol 保持一致
    if 'sidebar_search_input' not in st.session_state:
        st.session_state['sidebar_search_input'] = st.session_state['last_search_symbol']
    if 'current_asset_class' not in st.session_state:
        st.session_state['current_asset_class'] = "台股" # 預設台股

    # 2. 選擇資產類別 (使用更精簡的判斷邏輯)
    current_input = st.session_state['sidebar_search_input'].upper().strip()
    
    def get_default_asset_index(input_symbol):
        if input_symbol in US_STOCKS_MAP or (not input_symbol.endswith(".TW") and not input_symbol.endswith("-USD") and input_symbol in FULL_SYMBOLS_MAP):
            return 0  # 美股/其他
        elif input_symbol.endswith(".TW") or (input_symbol not in FULL_SYMBOLS_MAP and 'TW' in input_symbol):
            return 1  # 台股
        elif input_symbol.endswith("-USD"):
            return 2  # 加密貨幣
        return 1 # 默認台股 (如果找不到，給台灣用戶較多可能使用台股)

    default_asset_index = get_default_asset_index(current_input)
    
    asset_class = st.sidebar.radio(
        "選擇資產類別", 
        ["美股", "台股", "加密貨幣"], 
        key="asset_class_radio",
        index=default_asset_index,
    )
    st.session_state['current_asset_class'] = asset_class


    # 3. 快速選擇標的 (動態顯示)
    st.sidebar.markdown("---")
    
    if asset_class == "美股":
        map_to_use = US_STOCKS_MAP
        key_select = "quick_select_us"
    elif asset_class == "台股":
        map_to_use = TW_STOCKS_MAP
        key_select = "quick_select_tw"
    else:
        map_to_use = CRYPTO_MAP
        key_select = "quick_select_crypto"

    options_keys = list(map_to_use.keys())
    
    # 設置快速選擇的預設索引：如果當前手動輸入的值在選項中，則選中它
    try:
        default_index = options_keys.index(current_input)
    except ValueError:
        default_index = 0
    
    st.sidebar.selectbox(
        f"快速選擇標的 ({asset_class})",
        options=options_keys,
        index=default_index,
        key=key_select,
        on_change=update_search_input_from_select # 統一使用回調函式
    )

    # 4. 手動輸入代碼 (狀態同步的關鍵)
    # 這裡直接使用 'sidebar_search_input' 作為 key，讓其值與 Session State 保持雙向綁定。
    st.sidebar.text_input(
        "或手動輸入代碼 (例如: GOOG, 2330.TW, BTC-USD)：",
        key="sidebar_search_input" # 保持與快速選擇的回調函式使用的 Session State 鍵一致
    )

    # 5. 選擇分析週期
    period_selection = st.sidebar.radio(
        "選擇分析週期", 
        list(PERIOD_MAP.keys()),
        key="period_selection"
    )
    
    st.sidebar.markdown("---")
    
    # 6. 執行按鈕
    if st.sidebar.button('📊 執行AI分析', type="primary", use_container_width=True):
        st.session_state['data_ready'] = True
        # 確保在執行時，將最新的輸入框值賦予 last_search_symbol
        st.session_state['last_search_symbol'] = st.session_state['sidebar_search_input'].upper().strip() 
    
    return st.session_state['last_search_symbol'], st.session_state['period_selection']

def main():
    """主應用程式邏輯"""
    
    symbol, period_key = sidebar_ui()
    
    st.title("📈 AI 頂級專家 四維度趨勢分析平台")
    
    # 初始化進度條
    progress_bar = st.empty()
    
    if st.session_state.get('data_ready', False) and st.session_state['last_search_symbol']:
        
        progress_bar.progress(0, text="📊 開始 AI 分析流程...")

        # 獲取資料
        data = get_data(st.session_state['last_search_symbol'], period_key, progress_bar)
        
        progress_bar.progress(100, text="✅ 資料獲取與處理完成。")
        time.sleep(0.5)
        progress_bar.empty()

        if data is not None:
            # 計算指標
            analyzed_data = calculate_technical_indicators(data)
            
            if analyzed_data is not None and len(analyzed_data) > 0:
                # 執行專業分析
                perform_ai_analysis(st.session_state['last_search_symbol'], analyzed_data, period_key)
            else:
                 # 這裡的錯誤處理已在 get_data 和 calculate_technical_indicators 中優化
                 st.error(f"⚠️ **分析資料不足或處理失敗**：請檢查所選週期是否包含足夠的歷史數據。")

            # 重設 data_ready 狀態以允許下次執行
            st.session_state['data_ready'] = False 
        
    else:
        progress_bar.empty()
        # 初始歡迎畫面
        st.markdown("---")
        st.markdown(f"## 歡迎使用 AI 頂級專家 四維度趨勢分析平台")
        st.markdown(f"本平台結合 **價格結構 (趨勢)**、**動量**、**波動性** 與 **成交量 (籌碼)** 四大維度，為您提供最嚴謹的趨勢判斷。", unsafe_allow_html=True)
        st.markdown(f"請在左側欄設定標的代碼（例如 **{', '.join(list(US_STOCKS_MAP.keys())[:2])}**、**{', '.join(list(TW_STOCKS_MAP.keys())[:1])}**、**{', '.join(list(CRYPTO_MAP.keys())[:1])}**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📝 平台核心價值：")
        st.markdown("1. **數據精準**：具備重試與數據清洗機制，確保輸入分析的數據可靠性。")
        st.markdown("2. **多維度分析**：納入傳統技術分析外的**成交量/資金流**判斷，模擬專業操盤手的**籌碼面**考量。")
        st.markdown("3. **交易策略**：AI建議與趨勢結構（多頭排列、空頭趨勢）嚴格對應，提供明確的交易操作指引。")
        
        st.markdown("---")


if __name__ == '__main__':
    # 確保 Session State 在應用程式啟動時就被初始化
    if 'last_search_symbol' not in st.session_state:
        st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state:
        st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state:
        st.session_state['sidebar_search_input'] = st.session_state['last_search_symbol']
    if 'current_asset_class' not in st.session_state:
        st.session_state['current_asset_class'] = "台股" 
    
    main()
