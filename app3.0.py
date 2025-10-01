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
    "30 分": ("60d", "30m"), 
    "4 小時": ("1y", "60m"), 
    "1 日": ("5y", "1d"), 
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】
FULL_SYMBOLS_MAP = {
    # ----------------------------------------------------
    # A. 美股核心 (US Stocks) - 個股
    # ----------------------------------------------------
    "TSLA": {"name": "特斯拉", "keywords": ["特斯拉", "電動車", "TSLA", "Tesla"]},
    "NVDA": {"name": "輝達", "keywords": ["輝達", "英偉達", "AI", "NVDA", "Nvidia"]},
    "AAPL": {"name": "蘋果", "keywords": ["蘋果", "iPhone", "AAPL", "Apple"]},
    "MSFT": {"name": "微軟", "keywords": ["微軟", "雲端", "MSFT", "Microsoft"]},
    "GOOG": {"name": "Google", "keywords": ["谷歌", "Alphabet", "GOOG"]},
    "AMZN": {"name": "亞馬遜", "keywords": ["亞馬遜", "電商", "AMZN", "Amazon"]},
    # ----------------------------------------------------
    # B. 台股核心 (Taiwan Stocks) - 個股 (使用 .TW 後綴)
    # ----------------------------------------------------
    "2330.TW": {"name": "台積電", "keywords": ["台積電", "晶圓", "2330"]},
    "2454.TW": {"name": "聯發科", "keywords": ["聯發科", "IC設計", "2454"]},
    "2317.TW": {"name": "鴻海", "keywords": ["鴻海", "電子代工", "2317"]},
    # ----------------------------------------------------
    # C. 加密貨幣 (Crypto) - (使用 -USD 後綴)
    # ----------------------------------------------------
    "BTC-USD": {"name": "比特幣", "keywords": ["比特幣", "BTC", "Bitcoin"]},
    "ETH-USD": {"name": "以太幣", "keywords": ["以太幣", "ETH", "Ethereum"]},
}

# 將 FULL_SYMBOLS_MAP 拆分為子類別，用於快速選擇
US_STOCKS_MAP = {k: v for k, v in FULL_SYMBOLS_MAP.items() if not k.endswith(".TW") and not k.endswith("-USD")}
TW_STOCKS_MAP = {k: v for k, v in FULL_SYMBOLS_MAP.items() if k.endswith(".TW")}
CRYPTO_MAP = {k: v for k, v in FULL_SYMBOLS_MAP.items() if k.endswith("-USD")}

# ==============================================================================
# 2. 資料獲取函式 (增加重試機制以提高穩定性)
# ==============================================================================

def get_data(symbol, period_key, progress_bar, max_retries=3):
    """
    從 yfinance 獲取歷史股價資料，包含指數退避重試機制。
    
    Args:
        symbol (str): 股票代碼。
        period_key (str): 選擇的分析週期鍵 (e.g., "1 日")。
        progress_bar (st.progress): Streamlit 進度條物件。
        max_retries (int): 最大重試次數。
        
    Returns:
        pd.DataFrame or None: 歷史股價資料。
    """
    period, interval = PERIOD_MAP.get(period_key, ("5y", "1d"))
    
    for attempt in range(max_retries):
        progress_bar.progress(20 + (attempt * 10), text=f"📥 正在獲取 {symbol} ({period_key}) 歷史數據... (嘗試 {attempt + 1}/{max_retries})")
        time.sleep(0.5)
        
        try:
            # 確保使用正確的 yfinance 參數 (已移除 show_errors)
            data = yf.download(symbol, period=period, interval=interval, progress=False, timeout=10)
            
            if data.empty:
                # 代碼無效或無數據，不需重試
                st.error(f"⚠️ **獲取 {symbol} 數據失敗:** 找不到該代碼的資料，請檢查代碼是否正確。")
                progress_bar.empty()
                return None
            
            # 數據成功獲取
            progress_bar.progress(80, text=f"✅ 數據獲取成功！")
            time.sleep(0.5)
            
            data.columns = [c.capitalize() for c in data.columns]
            data = data.rename(columns={'Adj close': 'Adj_Close'})
            data['Symbol'] = symbol
            return data

        except Exception as e:
            if attempt < max_retries - 1:
                # 執行指數退避
                wait_time = 2 ** attempt  # 1, 2, 4 seconds
                st.warning(f"網路或API錯誤，正在重試... (等待 {wait_time} 秒)")
                time.sleep(wait_time)
            else:
                st.error(f"❌ **獲取 {symbol} 數據時發生嚴重錯誤:** 已達最大重試次數。錯誤訊息: {e}")
                progress_bar.empty()
                return None
    return None

# ==============================================================================
# 3. 趨勢分析與指標計算 (新增成交量指標)
# ==============================================================================

def calculate_technical_indicators(data):
    """計算技術指標 (RSI, MACD, Bollinger Bands, Volume SMA)"""
    if data is None or data.empty:
        return None

    df = data.copy()
    
    # 趨勢指標
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 動量指標: RSI
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
    
    # 成交量指標: 20日成交量均線
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    return df.dropna()

def perform_ai_analysis(symbol, data):
    """
    模擬 AI 頂級專家的分析輸出
    
    結合了價格趨勢 (SMA), 動量 (RSI/MACD), 和 成交量 (Volume) 的簡化四維度分析
    """
    
    st.subheader(f"目標標的：{symbol} ({data.index[-1].strftime('%Y-%m-%d')} 收盤價：${data['Close'].iloc[-1]:.2f})")
    st.markdown("---")
    
    # --- 提取最新數據 ---
    last_close = data['Close'].iloc[-1]
    sma_20 = data['SMA_20'].iloc[-1]
    sma_50 = data['SMA_50'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    macd_hist = data['MACD_Hist'].iloc[-1]
    last_volume = data['Volume'].iloc[-1]
    volume_sma_20 = data['Volume_SMA_20'].iloc[-1]
    
    # --- 1. 綜合趨勢評估 (模擬 AI 核心判斷) ---
    
    score = 0
    analysis_points = []
    
    # A. 價格趨勢 (佔比最高)
    if last_close > sma_20:
        score += 3
        analysis_points.append("📈 **價格趨勢 (SMA)**: 股價位於 20 日移動平均線之上，短期多頭結構確立。")
    elif last_close < sma_20 and last_close > sma_50:
        score += 1
        analysis_points.append("⚠️ **價格趨勢 (SMA)**: 股價跌破短期 (SMA 20) 支撐，但仍守住中期 (SMA 50) 關鍵支撐。")
    else:
        score -= 2
        analysis_points.append("🔻 **價格趨勢 (SMA)**: 股價跌破短期與中期均線，空頭壓力較大。")

    # B. 動量指標 (RSI)
    if rsi >= 70:
        score -= 1
        analysis_points.append(f"🔴 **動量 (RSI)**: RSI 進入超買區 ({rsi:.1f})，短線有回調風險。")
    elif rsi <= 30:
        score += 1
        analysis_points.append(f"🟢 **動量 (RSI)**: RSI 進入超賣區 ({rsi:.1f})，可能醞釀技術性反彈。")
    else:
        score += 0.5
        analysis_points.append(f"🟡 **動量 (RSI)**: RSI 處於中性健康區 ({rsi:.1f})，動能均衡。")

    # C. MACD
    if macd_hist > 0:
        score += 2
        analysis_points.append("✅ **動量 (MACD)**: MACD 柱狀圖為正，動能持續增強，處於金叉後的擴張階段。")
    else:
        score -= 1
        analysis_points.append("❌ **動量 (MACD)**: MACD 柱狀圖為負，動能減弱或處於死叉後的收縮階段。")

    # D. 成交量 (Volume) - 模擬籌碼/資金流
    if last_volume > volume_sma_20 and last_close > data['Close'].iloc[-2]:
        score += 2
        analysis_points.append("💰 **成交量/資金流**: 股價上漲配合成交量顯著放大，顯示**資金追捧積極**。")
    elif last_volume > volume_sma_20 and last_close < data['Close'].iloc[-2]:
        score -= 1
        analysis_points.append("💧 **成交量/資金流**: 股價下跌但成交量放大，顯示**市場恐慌性賣壓或資金流出**。")
    else:
        analysis_points.append("⚪ **成交量/資金流**: 成交量處於正常水準，市場情緒穩定。")

    # --- 2. 總結判斷與交易策略 ---
    
    if score >= 4:
        tech_advice = "🟢 **強烈買入/重倉持有**"
        tech_summary = "綜合評估結果：趨勢、動量、資金流向均表現強勁，屬於極佳的多頭配置。建議積極持有，並將短期停損點設在 20 日均線附近。"
    elif score >= 1:
        tech_advice = "🟡 **買入/持續觀察**"
        tech_summary = "綜合評估結果：趨勢結構良好，但部分動量指標（如 RSI）或資金流略有分歧。建議輕倉佈局或等待回調至關鍵支撐位 (如 SMA 50) 時再加倉。"
    else:
        tech_advice = "🔴 **觀望/賣出**"
        tech_summary = "綜合評估結果：趨勢結構已遭破壞，且動量/資金流顯示賣壓主導。建議避免建倉，或等待價格重新站上短期均線後再行考慮。"

    # --- 輸出結果 ---
    st.markdown("#### 1. AI 綜合趨勢評估 (四維度解析)")
    st.info(f"**最終交易建議：{tech_advice}**")
    st.markdown(tech_summary)
    
    st.markdown("##### 核心分析觀點：")
    for point in analysis_points:
        st.markdown(f"- {point}")
    
    st.markdown("---")
    
    # --- 3. 圖表展示 ---
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08, # 調整垂直間距
                        row_heights=[0.5, 0.15, 0.15, 0.2]) # 增加 Volume Row
    
    # K線圖與均線
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'],
                                 name='K線'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], line=dict(color='blue', width=1), name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    
    # MACD 圖
    colors_macd = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='MACD 柱狀圖', marker_color=colors_macd), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], line=dict(color='blue', width=1), name='MACD 線'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], line=dict(color='red', width=1), name='Signal 線'), row=2, col=1)
    
    # RSI 圖
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='purple', width=1), name='RSI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data), line=dict(color='red', width=0.5, dash='dash'), name='超買(70)', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data), line=dict(color='green', width=0.5, dash='dash'), name='超賣(30)', showlegend=False), row=3, col=1)

    # Volume 圖 (新增的第四維度)
    colors_volume = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='成交量', marker_color=colors_volume), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Volume_SMA_20'], line=dict(color='gray', width=1, dash='dot'), name='Volume SMA 20'), row=4, col=1)
    
    # 更新佈局
    fig.update_layout(
        title=f'{symbol} - 價格、動量與成交量趨勢分析',
        xaxis_rangeslider_visible=False,
        height=900,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="成交量", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 4. Streamlit 介面與邏輯 (UI狀態優化)
# ==============================================================================

# Helper function to update the manual search input state
def update_search_input_us():
    """Callback function for US stock quick select."""
    st.session_state['sidebar_search_input'] = st.session_state.quick_select_us

def update_search_input_tw():
    """Callback function for TW stock quick select."""
    st.session_state['sidebar_search_input'] = st.session_state.quick_select_tw

def update_search_input_crypto():
    """Callback function for Crypto quick select."""
    st.session_state['sidebar_search_input'] = st.session_state.quick_select_crypto
    
def sidebar_ui():
    """側邊欄 UI 元素和狀態管理"""
    st.sidebar.header("📈 參數設定")
    
    # 初始化 Session State - 確保所有鍵都存在
    if 'last_search_symbol' not in st.session_state:
        st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state:
        st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state:
        st.session_state['sidebar_search_input'] = st.session_state['last_search_symbol']
    if 'current_asset_class' not in st.session_state:
        st.session_state['current_asset_class'] = "台股" # 根據預設值初始化

    # 根據當前輸入代碼判斷應選中的資產類別索引 (用於 Radio Button)
    current_input = st.session_state['sidebar_search_input'].upper().strip()
    if current_input in US_STOCKS_MAP or not (current_input.endswith(".TW") or current_input.endswith("-USD")):
        default_asset_index = 0  # 美股/其他
    elif current_input in TW_STOCKS_MAP or current_input.endswith(".TW"):
        default_asset_index = 1  # 台股
    elif current_input in CRYPTO_MAP or current_input.endswith("-USD"):
        default_asset_index = 2  # 加密貨幣
    else:
        # Fallback to current state if symbol is custom/unknown
        if st.session_state['current_asset_class'] == "美股":
            default_asset_index = 0
        elif st.session_state['current_asset_class'] == "台股":
            default_asset_index = 1
        else:
            default_asset_index = 2
        
    # 1. 選擇資產類別
    asset_class = st.sidebar.radio(
        "選擇資產類別", 
        ["美股", "台股", "加密貨幣"], 
        key="asset_class_radio",
        index=default_asset_index
    )
    # 更新當前資產類別到 Session State
    st.session_state['current_asset_class'] = asset_class


    # 2. 快速選擇標的 (動態顯示)
    st.sidebar.markdown("---")
    
    if asset_class == "美股":
        map_to_use = US_STOCKS_MAP
        update_func = update_search_input_us
        key_select = "quick_select_us"
    elif asset_class == "台股":
        map_to_use = TW_STOCKS_MAP
        update_func = update_search_input_tw
        key_select = "quick_select_tw"
    else:
        map_to_use = CRYPTO_MAP
        update_func = update_search_input_crypto
        key_select = "quick_select_crypto"

    options_keys = list(map_to_use.keys())
    
    # 根據當前輸入值設置快速選擇的預設索引
    try:
        default_index = options_keys.index(current_input)
    except ValueError:
        # 如果當前輸入不在快速選擇列表中，則選中第一個選項
        default_index = 0
    
    st.sidebar.selectbox(
        f"快速選擇標的 ({asset_class})",
        options=options_keys,
        index=default_index,
        key=key_select,
        on_change=update_func 
    )

    # 3. 手動輸入代碼 (使用 Session State 控制 value)
    st.session_state['sidebar_search_input'] = st.sidebar.text_input(
        "或手動輸入代碼 (例如: GOOG, 2330.TW, BTC-USD)：",
        value=st.session_state['sidebar_search_input'], # 綁定 Session State
        key="manual_search_input_final_key"
    )

    # 4. 選擇分析週期
    period_selection = st.sidebar.radio(
        "選擇分析週期", 
        list(PERIOD_MAP.keys()),
        key="period_selection"
    )
    
    st.sidebar.markdown("---")
    
    # 執行按鈕
    if st.sidebar.button('📊 執行AI分析', type="primary", use_container_width=True):
        st.session_state['data_ready'] = True
        st.session_state['last_search_symbol'] = st.session_state['sidebar_search_input'].upper().strip() # 確定最終使用的代碼
    
    return st.session_state['last_search_symbol'], st.session_state['period_selection']

def main():
    """主應用程式邏輯"""
    
    symbol, period_key = sidebar_ui()
    
    st.title("📈 AI 頂級專家 四維度趨勢分析平台")
    
    # 設置頂部標題和參數顯示
    st.markdown(f"#### 目標標的：{symbol} ({period_key} 週期)")
    
    # 初始化進度條
    progress_bar = st.empty()
    progress_bar.progress(0, text="等待執行分析...")
    
    if st.session_state.get('data_ready', False) and st.session_state['last_search_symbol']:
        # 獲取資料
        data = get_data(st.session_state['last_search_symbol'], period_key, progress_bar)
        
        progress_bar.progress(100, text="🚀 分析完成，正在生成報告...")
        time.sleep(0.5)
        progress_bar.empty()

        if data is not None:
            # 計算指標並執行分析
            analyzed_data = calculate_technical_indicators(data)
            
            # 確保有足夠的數據進行計算（例如 SMA 200 需要 200 筆數據）
            if analyzed_data is not None and len(analyzed_data) > 0:
                perform_ai_analysis(st.session_state['last_search_symbol'], analyzed_data)
            else:
                 st.error(f"⚠️ **分析資料不足:** {st.session_state['last_search_symbol']} 在所選週期 ({period_key}) 內沒有足夠的數據來計算所有指標 (例如 SMA 200)，請嘗試更長的週期。")

            # 重設 data_ready 狀態以允許下次執行
            st.session_state['data_ready'] = False 
        
    else:
        progress_bar.empty()
        # 初始歡迎畫面
        st.markdown("---")
        st.markdown(f"## 歡迎使用 AI 頂級專家 四維度趨勢分析平台")
        st.markdown(f"請在左側欄設定標的代碼（例如 **{', '.join(US_STOCKS_MAP.keys())}**、**{', '.join(TW_STOCKS_MAP.keys())}**、**{', '.join(CRYPTO_MAP.keys())}**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇標的**：使用下拉選單快速選擇熱門標的，或直接在輸入框中鍵入代碼或名稱。")
        st.markdown("3. **選擇週期**：決定分析的長度（例如：`30 分`、`4 小時`、`1 日`、`1 周`）。")
        st.markdown(f"4. **執行分析**：點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』**</span>，AI將融合基本面與技術面指標提供交易策略。", unsafe_allow_html=True)
        
        st.markdown("---")


if __name__ == '__main__':
    main()
