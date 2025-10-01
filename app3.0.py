import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import warnings
import json
import time
import re 
from datetime import datetime, timedelta

# 為了在 Canvas 環境中模擬 API 金鑰
# 實際執行時，Canvas 會自動注入 API 金鑰
# 此處保留空白字串
API_KEY = ""
GEMINI_MODEL_TEXT = "gemini-2.5-flash-preview-05-20" 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_TEXT}:generateContent?key={API_KEY}"

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
# 確保 4 小時使用 60m 的 interval，並使用 1y period 獲取足夠數據
PERIOD_MAP = { 
    "30 分": ("60d", "30m"), 
    "4 小時": ("1y", "60m"), 
    "1 日": ("5y", "1d"), 
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】 (簡化版)
FULL_SYMBOLS_MAP = {
    "TSLA": {"name": "特斯拉", "keywords": ["特斯拉", "電動車", "TSLA", "Tesla"]},
    "NVDA": {"name": "輝達", "keywords": ["輝達", "英偉達", "AI", "NVDA", "Nvidia"]},
    "AAPL": {"name": "蘋果", "keywords": ["蘋果", "手機", "AAPL", "Apple"]},
    "GOOG": {"name": "Google", "keywords": ["谷歌", "Alphabet", "GOOG"]},
    "2330.TW": {"name": "台積電", "keywords": ["台積電", "晶圓", "2330", "TSMC", "TW"]},
    "0050.TW": {"name": "元大台灣50", "keywords": ["0050", "ETF", "台灣50", "TW"]},
    "BTC-USD": {"name": "比特幣", "keywords": ["比特幣", "加密貨幣", "BTC"]},
    "ETH-USD": {"name": "以太坊", "keywords": ["以太坊", "加密貨幣", "ETH"]},
}

# ==============================================================================
# 2. 數據獲取與處理
# ==============================================================================

@st.cache_data(ttl=600) # 緩存數據 10 分鐘
def get_yfinance_data(symbol, period, interval):
    """
    從 yfinance 獲取歷史價格數據。
    """
    try:
        # 檢查是否為台股代碼，如果是則加上 .TW 後綴 (如果用戶忘記)
        if symbol.isdigit() and not symbol.endswith(('.TW', '.US', '-USD')):
            symbol_to_fetch = f"{symbol}.TW"
        else:
            symbol_to_fetch = symbol.upper()

        data = yf.download(
            symbol_to_fetch, 
            period=period, 
            interval=interval, 
            progress=False, 
            show_errors=False
        )
        
        if data.empty:
            st.error(f"⚠️ 找不到代碼 `{symbol_to_fetch}` 的數據，請檢查代碼是否正確。")
            return None, symbol_to_fetch
        
        # 重新命名列，以符合 ta 庫的標準，並確保都是 float
        data.columns = [col.capitalize() for col in data.columns]
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        
        # 移除任何重複的索引 (可能在某些時間區間出現)
        data = data[~data.index.duplicated(keep='first')]
        
        return data, symbol_to_fetch

    except Exception as e:
        st.error(f"獲取 `{symbol}` 數據時發生錯誤: {e}")
        return None, symbol.upper()

def calculate_technical_indicators(df):
    """
    計算並添加技術指標到 DataFrame。
    **修復方案: 確保傳遞給 ta 庫的 'Close'/'Volume' 是單一 Pandas Series (1D 結構)**
    """
    if df is None or df.empty:
        return None
    
    try:
        # 1. 數據清理：替換無限值為 NaN
        # 這一行是您 Traceback 中顯示的執行位置。
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 確保我們操作的是 Pandas Series，防止傳遞多維數據給 ta 庫
        close_series = df['Close']
        high_series = df['High']
        low_series = df['Low']
        volume_series = df['Volume']
        
        # === 趨勢指標 (Trend Indicators) ===
        # Simple Moving Average (SMA)
        df['SMA_5'] = ta.trend.sma_indicator(close=close_series, window=5, fillna=True)
        df['SMA_20'] = ta.trend.sma_indicator(close=close_series, window=20, fillna=True)
        df['SMA_60'] = ta.trend.sma_indicator(close=close_series, window=60, fillna=True)

        # Moving Average Convergence Divergence (MACD)
        macd = ta.trend.MACD(close=close_series, window_fast=12, window_slow=26, window_sign=9, fillna=True)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # === 動量指標 (Momentum Indicators) ===
        # Relative Strength Index (RSI)
        df['RSI'] = ta.momentum.rsi(close=close_series, window=14, fillna=True)
        
        # Stochastic Oscillator (STOCH)
        stoch = ta.momentum.StochasticOscillator(high=high_series, low=low_series, close=close_series, window=14, smooth_window=3, fillna=True)
        df['STOCH_K'] = stoch.stoch()
        df['STOCH_D'] = stoch.stoch_signal()
        
        # === 波動性指標 (Volatility Indicators) ===
        # Bollinger Bands (BB)
        bb = ta.volatility.BollingerBands(close=close_series, window=20, window_dev=2, fillna=True)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        
        # === 交易量指標 (Volume Indicators) ===
        # On-Balance Volume (OBV)
        df['OBV'] = ta.volume.on_balance_volume(close=close_series, volume=volume_series, fillna=True)
        
        # 填充可能產生的 NaN 值 (如 SMA/BB 的初期值)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        return df
    
    except Exception as e:
        # 如果在計算 TA 時發生錯誤，打印錯誤並返回 None
        st.error(f"技術指標計算失敗，請檢查數據結構: {e}")
        # print(f"Error in TA calculation: {e}")
        return None

# ==============================================================================
# 3. 圖表繪製
# ==============================================================================

def plot_candlestick(df, symbol, interval):
    """
    繪製 K 線圖、MACD、RSI、成交量等綜合圖表。
    """
    if df is None or df.empty:
        return None

    # 確保只取有數據的行
    df = df.dropna()

    fig = make_subplots(
        rows=4, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=(
            f'**{symbol} K 線圖與趨勢指標 ({interval})**', 
            'MACD (趨勢動量)', 
            'RSI (相對強弱指標)', 
            '成交量與 OBV'
        )
    )

    # 1. K 線圖 (Row 1)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='K線',
        increasing_line_color='red', 
        decreasing_line_color='green'
    ), row=1, col=1)

    # SMA 趨勢線
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_5'], line=dict(color='#8B008B', width=1), name='SMA 5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='#FFD700', width=1), name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_60'], line=dict(color='#4169E1', width=1), name='SMA 60'), row=1, col=1)

    # 布林通道 (BB)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='rgba(135, 206, 235, 0.5)', width=1), name='BB 上軌'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='rgba(135, 206, 235, 0.5)', width=1), name='BB 下軌', fill='tonexty', fillcolor='rgba(135, 206, 235, 0.1)'), row=1, col=1)


    # 2. MACD (Row 2)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#FFA500', width=1.5), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#4682B4', width=1.5), name='Signal'), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=['#FF4500' if v >= 0 else '#008000' for v in df['MACD_Hist']], name='Histogram'), row=2, col=1)


    # 3. RSI (Row 3)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#DC143C', width=2), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FA8072", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#3CB371", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # 4. 成交量與 OBV (Row 4)
    colors = ['red' if df['Open'][i] < df['Close'][i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='成交量'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='#1E90FF', width=1.5), name='OBV', yaxis='y2'), row=4, col=1)
    
    # 設置佈局
    fig.update_layout(
        title_text=f"**{symbol} 四維度技術分析總覽**",
        xaxis_rangeslider_visible=False,
        height=900,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified"
    )

    # 更新 Row 1 的 Y 軸
    fig.update_yaxes(title_text="價格", row=1, col=1)

    # 更新 Row 4 的 Y 軸以添加第二個軸（給 OBV）
    fig.update_layout(
        yaxis4=dict(
            title="OBV",
            overlaying="y",
            side="right"
        )
    )

    # 隱藏範圍滑塊
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    # 清理不必要的圖例
    for i in [1, 2, 3, 4]:
        fig.update_xaxes(showgrid=True, row=i, col=1)
        fig.update_yaxes(showgrid=True, row=i, col=1)
        
    fig.update_traces(showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 4. AI 分析與 LLM 互動 (模擬)
# ==============================================================================

async def retry_fetch(url, payload, retries=3, delay=1.0):
    """Implement exponential backoff retry for fetch operations."""
    headers = {'Content-Type': 'application/json'}
    for i in range(retries):
        try:
            response = await st.runtime.scriptrunner.add_script_run_ctx(
                fetch
            )(url, method='POST', headers=headers, body=json.dumps(payload))
            if response.status == 200:
                return await response.json()
            # If not 200, wait and retry
            await time.sleep(delay * (2 ** i)) 
        except Exception as e:
            # print(f"Attempt {i+1} failed: {e}")
            await time.sleep(delay * (2 ** i))
    raise Exception(f"Failed to fetch content after {retries} retries.")


def generate_analysis_payload(symbol, interval, latest_data_summary, indicators_summary):
    """
    構建 LLM 的 API 請求 payload。
    """
    
    system_prompt = (
        "您是一位頂級的量化分析師，擁有 20 年的交易經驗。您的任務是結合基本面（通過 Google Search 獲取）、"
        "技術面和市場情緒，為用戶提供一個全面且精準的四維度趨勢分析。 "
        "請根據提供的數據摘要和技術指標，以專業、簡潔、有邏輯的方式進行分析，並給出明確的交易建議（買入/賣出/觀望）。"
        "請使用繁體中文輸出。分析內容應包含：1. 技術面簡述 2. 宏觀/基本面（基於外部搜索） 3. 交易建議與風險提示。"
    )

    user_query = (
        f"請針對標的 {symbol} (週期: {interval}) 進行四維度趨勢分析。\n\n"
        f"【最新數據摘要】:\n{latest_data_summary}\n\n"
        f"【技術指標趨勢摘要】:\n{indicators_summary}\n\n"
        f"請務必利用 Google Search 工具，獲取 {symbol} 的最新財報或重大新聞，並將其融入分析報告中。"
    )

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        # 啟用 Google Search 進行基本面和消息面分析
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    return payload


async def generate_ai_analysis(symbol, interval, df):
    """
    調用 Gemini API 進行趨勢分析，並處理結果和引用來源。
    """
    if df is None or df.empty:
        return "無法生成分析：數據為空。"

    st.subheader(f"🧠 AI 頂級專家分析報告 - {symbol} ({interval})")
    
    # 創建數據摘要
    latest_row = df.iloc[-1]
    
    latest_data_summary = f"""
    - 最新收盤價 (Close): {latest_row['Close']:.2f}
    - 20 日均價 (SMA 20): {latest_row['SMA_20']:.2f}
    - 20 日布林通道上軌 (BB High): {latest_row['BB_High']:.2f}
    - RSI (14): {latest_row['RSI']:.2f} (一般認為 <30 超賣, >70 超買)
    - MACD 柱狀圖 (Hist): {latest_row['MACD_Hist']:.2f} (一般認為 >0 動能強勁)
    """

    # 趨勢判斷邏輯 (簡化為 LLM 提供上下文)
    trend_data = {
        'SMA5_Above_SMA20': latest_row['SMA_5'] > latest_row['SMA_20'],
        'MACD_Positive': latest_row['MACD_Hist'] > 0,
        'RSI_Level': '超賣 (<30)' if latest_row['RSI'] < 30 else ('超買 (>70)' if latest_row['RSI'] > 70 else '中性 (30-70)'),
        'Price_Near_BB_High': latest_row['Close'] >= latest_row['BB_High']
    }
    
    indicators_summary = f"""
    - 短期均線 (SMA 5) {'高於' if trend_data['SMA5_Above_SMA20'] else '低於'} 中期均線 (SMA 20)，顯示短期趨勢 {'看漲' if trend_data['SMA5_Above_SMA20'] else '看跌'}。
    - MACD 柱狀圖 {'為正值' if trend_data['MACD_Positive'] else '為負值'}，動量顯示 {'多頭佔優' if trend_data['MACD_Positive'] else '空頭佔優'}。
    - RSI 位於 {trend_data['RSI_Level']} 區域。
    - 價格 {'正在觸及或突破' if trend_data['Price_Near_BB_High'] else '位於'} 布林通道上軌。
    """

    payload = generate_analysis_payload(symbol, interval, latest_data_summary, indicators_summary)
    
    try:
        # 使用 Streamlit 內建的運行時上下文執行 fetch
        with st.spinner("🤖 AI 正在整合基本面、技術面和宏觀消息進行深度分析..."):
            response_json = await retry_fetch(GEMINI_API_URL, payload)

        if response_json and response_json.get('candidates'):
            candidate = response_json['candidates'][0]
            text = candidate['content']['parts'][0]['text']
            
            # 提取引用來源
            sources = []
            grounding_metadata = candidate.get('groundingMetadata')
            if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                sources = [
                    {
                        'uri': attr['web']['uri'],
                        'title': attr['web']['title']
                    }
                    for attr in grounding_metadata['groundingAttributions']
                    if attr.get('web') and attr['web'].get('uri') and attr['web'].get('title')
                ]
            
            st.markdown(text)
            
            if sources:
                st.markdown("---")
                st.subheader("📚 資訊來源 (消息面/基本面)")
                source_markdown = ""
                for i, source in enumerate(sources):
                    source_markdown += f"- **[{source['title']}]({source['uri']})**\n"
                st.markdown(source_markdown)
        else:
            st.warning("AI 分析服務暫時無法回應。")
            
    except Exception as e:
        st.error(f"AI 分析服務調用失敗: {e}")

# ==============================================================================
# 5. Streamlit 主程式
# ==============================================================================

def get_symbol_name(symbol):
    """根據代碼獲取中文名稱"""
    for data in FULL_SYMBOLS_MAP.values():
        if data['keywords'] and symbol.upper() in [k.upper() for k in data['keywords']]:
            return data['name']
    return symbol


def main():
    # 初始化 Session State
    if 'last_search_symbol' not in st.session_state:
        st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state:
        st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state:
        st.session_state['sidebar_search_input'] = ""
    if 'current_symbol' not in st.session_state:
        st.session_state['current_symbol'] = "2330.TW"
    if 'current_interval' not in st.session_state:
        st.session_state['current_interval'] = "1 日"

    # 側邊欄 (Sidebar) 設置
    st.sidebar.title("📈 參數設定")

    # 1. 資產類別選擇 (簡化，主要影響推薦列表，實際仍靠代碼)
    asset_class = st.sidebar.selectbox(
        "選擇資產類別",
        ["台股", "美股", "加密貨幣"],
        index=0 if st.session_state['current_symbol'].endswith('.TW') else (1 if not st.session_state['current_symbol'].endswith('-USD') else 2)
    )

    # 2. 快速選擇標的 (簡化，僅供參考)
    default_symbols = {
        "美股": ["TSLA", "NVDA", "AAPL", "GOOG"],
        "台股": ["2330.TW", "0050.TW"],
        "加密貨幣": ["BTC-USD", "ETH-USD"]
    }.get(asset_class, [])

    quick_select = st.sidebar.selectbox(
        f"快速選擇標的 ({asset_class})",
        [""] + default_symbols,
        index=0
    )

    # 3. 手動輸入代碼 (核心輸入)
    default_input = st.session_state['sidebar_search_input'] if st.session_state['sidebar_search_input'] else st.session_state['current_symbol']
    
    manual_input = st.sidebar.text_input(
        "或手動輸入代碼 (例如: GOOG, 2330.TW, BTC-USD)",
        value=default_input
    )
    
    # 處理輸入邏輯
    search_symbol = quick_select if quick_select else manual_input.upper()
    
    # 4. 分析週期選擇
    interval_options = list(PERIOD_MAP.keys())
    selected_interval = st.sidebar.selectbox(
        "選擇分析週期",
        interval_options,
        index=interval_options.index(st.session_state['current_interval'])
    )
    
    # 更新 session state
    st.session_state['sidebar_search_input'] = manual_input
    st.session_state['current_symbol'] = search_symbol
    st.session_state['current_interval'] = selected_interval

    # 5. 執行按鈕
    if st.sidebar.button("📊 執行AI分析", key="run_analysis", type="primary"):
        if not search_symbol:
            st.sidebar.error("請輸入或選擇標的代碼。")
        else:
            st.session_state['data_ready'] = False
            st.session_state['last_search_symbol'] = search_symbol
            st.rerun() # 重新運行以獲取數據並顯示結果

    # --- 主要內容區 ---
    st.title("📈 AI 頂級專家 四維度趨勢分析平台")

    # 獲取和處理數據
    symbol_to_process = st.session_state['last_search_symbol']
    interval_to_process = st.session_state['current_interval']
    
    if symbol_to_process:
        period, interval_yf = PERIOD_MAP[interval_to_process]
        
        # 顯示目標標的資訊 (與您的 Traceback 輸出格式一致)
        st.markdown(f"目標標的：**{symbol_to_process}** ({interval_to_process} 週期)")

        # 獲取數據
        df_raw, fetched_symbol = get_yfinance_data(symbol_to_process, period, interval_yf)
        
        if df_raw is not None and not df_raw.empty:
            # 計算技術指標 (這裡是修正了 ValueError 的核心函數)
            df_with_ta = calculate_technical_indicators(df_raw.copy())
            
            if df_with_ta is not None:
                st.session_state['data_ready'] = True
                st.session_state['processed_df'] = df_with_ta
                st.session_state['fetched_symbol'] = fetched_symbol
            else:
                st.session_state['data_ready'] = False
        else:
             st.session_state['data_ready'] = False


    # 顯示結果
    if st.session_state['data_ready'] and 'processed_df' in st.session_state:
        df_display = st.session_state['processed_df']
        fetched_symbol = st.session_state['fetched_symbol']

        # 1. 繪製圖表
        plot_candlestick(df_display, get_symbol_name(fetched_symbol), interval_to_process)

        # 2. 執行 AI 分析 (使用 await 調用非同步函數)
        st.markdown("---")
        st.header("🤖 AI 分析結果")
        st.markdown("")
        st.session_state['ai_analysis_placeholder'] = st.empty()
        st.session_state['ai_analysis_placeholder'].markdown("---")
        st.session_state['ai_analysis_placeholder'].text("點擊『📊 執行AI分析』按鈕後，AI報告將顯示在此處...")

        # 使用 Streamlit 的非同步執行器來運行 AI 分析
        st.run_in_thread(generate_ai_analysis(fetched_symbol, interval_to_process, df_display))

        st.markdown("---")
        st.subheader("📊 原始數據表 (含技術指標)")
        st.dataframe(df_display.tail(30)) # 顯示最新 30 筆數據
        
    else:
        st.markdown("---")
        st.subheader("歡迎使用 AI 頂級專家 四維度趨勢分析平台")
        st.markdown(f"請在左側欄設定標的代碼（例如 **TSLA**, **2330.TW**, **BTC-USD**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇標的**：使用下拉選單快速選擇熱門標的，或直接在輸入框中鍵入代碼或名稱。")
        st.markdown("3. **選擇週期**：決定分析的長度（例如：`30 分`、`4 小時`、`1 日`、`1 周`）。")
        st.markdown(f"4. **執行分析**：點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』**</span>，AI將融合基本面與技術面指標提供交易策略。", unsafe_allow_html=True)
        
        st.markdown("---")


if __name__ == '__main__':
    main()
