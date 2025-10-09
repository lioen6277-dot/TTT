import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 頁面配置與全局設定
# ==============================================================================

st.set_page_config(
    page_title="AI趨勢分析📈 (v4.0 - 多模型止損)",
    page_icon="🤖",
    layout="wide"
)

# 週期映射：(YFinance Period, YFinance Interval)
PERIOD_MAP = { 
    "30 分": ("60d", "30m"), 
    "4 小時": ("1y", "90m"), # 修正為 90m 提高穩定性
    "1 日": ("5y", "1d"), 
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】(請根據您的 app3.0.py 保持此處的完整列表)
FULL_SYMBOLS_MAP = {
    # ----------------------------------------------------
    # A. 美股核心 (US Stocks) - 個股
    # ----------------------------------------------------
    "TSLA": {"name": "特斯拉", "keywords": ["特斯拉", "電動車", "TSLA", "Tesla"]},
    "NVDA": {"name": "輝達", "keywords": ["輝達", "英偉達", "AI", "NVDA", "Nvidia"]},
    "AMD": {"name": "超微 (Advanced Micro Devices)", "keywords": ["超微", "AMD", "半導體"]},
    "AAPL": {"name": "蘋果 (Apple)", "keywords": ["蘋果", "Apple", "AAPL"]},
    "GOOGL": {"name": "谷歌 (Google)", "keywords": ["谷歌", "Alphabet", "GOOGL"]},
    "AMZN": {"name": "亞馬遜", "keywords": ["亞馬遜", "Amazon", "AMZN"]},
    "MSFT": {"name": "微軟", "keywords": ["微軟", "Microsoft", "MSFT"]},
    "META": {"name": "臉書", "keywords": ["臉書", "Meta", "META"]},
    "JPM": {"name": "摩根大通", "keywords": ["摩根大通", "JPM", "金融股"]},
    "BABA": {"name": "阿里巴巴", "keywords": ["阿里巴巴", "BABA", "中概股"]},
    "ADBE": {"name": "Adobe", "keywords": ["Adobe", "ADBE"]},
    "ACN": {"name": "Accenture (埃森哲)", "keywords": ["Accenture", "ACN", "諮詢", "科技服務"]},
    # ----------------------------------------------------
    # B. 美股核心 (US Stocks) - ETF/指數
    # ----------------------------------------------------
    "SPY": {"name": "標普500 ETF", "keywords": ["標普500", "SPY", "S&P 500"]},
    "QQQ": {"name": "納斯達克100 ETF", "keywords": ["納斯達克", "QQQ", "Nasdaq"]},
    "VGT": {"name": "Vanguard資訊科技ETF", "keywords": ["VGT", "科技ETF", "資訊科技"]},
    "^VIX": {"name": "恐慌指數 (VIX)", "keywords": ["VIX", "恐慌指數", "波動率指數"]},
    # ----------------------------------------------------
    # C. 台股核心 (TW Stocks)
    # ----------------------------------------------------
    "2330.TW": {"name": "台積電 (TSMC)", "keywords": ["台積電", "2330", "TSMC", "半導體"]},
    "2454.TW": {"name": "聯發科 (MediaTek)", "keywords": ["聯發科", "2454", "MediaTek"]},
    "2317.TW": {"name": "鴻海 (Foxconn)", "keywords": ["鴻海", "2317", "Foxconn"]},
    "0050.TW": {"name": "元大台灣50", "keywords": ["台灣50", "0050", "ETF"]},
    "0056.TW": {"name": "元大高股息", "keywords": ["高股息", "0056", "ETF"]},
    # ----------------------------------------------------
    # D. 加密貨幣 (Crypto)
    # ----------------------------------------------------
    "BTC-USD": {"name": "比特幣", "keywords": ["比特幣", "BTC", "加密貨幣"]},
    "ETH-USD": {"name": "以太幣", "keywords": ["以太幣", "ETH", "加密貨幣"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL", "加密貨幣"]},
    # ----------------------------------------------------
}


# ==============================================================================
# 2. 數據獲取與指標計算 (新增 BB, PSAR)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_data(symbol, period_tuple):
    """從 Yahoo Finance 獲取數據"""
    period, interval = period_tuple
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # 排除最後一根不完整的 K 線
        if interval not in ['1d', '1wk']:
            df = df.iloc[:-1] 
        return df
    except Exception as e:
        st.error(f"獲取數據失敗: {e}")
        return None

def calculate_technical_indicators(df):
    """
    計算核心技術指標，新增 ATR, BB, PSAR
    """
    if df is None or df.empty: return pd.DataFrame()

    # --- 核心趨勢指標 ---
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)

    # --- 動量與超買超賣指標 ---
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD_Line'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # 🔥 V4.0 風險管理指標
    # 1. ATR (Average True Range)
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # 2. 布林通道 (Bollinger Bands)
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2.0)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    # 3. 拋物線 SAR (Parabolic SAR)
    # 使用預設參數 (0.02, 0.2)，適合趨勢追蹤
    psar = ta.trend.PSAR(df['High'], df['Low'], df['Close'], step=0.02, max_step=0.2)
    df['PSAR_Up'] = psar.psar_up()
    df['PSAR_Down'] = psar.psar_down()

    return df.dropna()


# ==============================================================================
# 3. 核心：動態止損/止盈水平計算
# ==============================================================================

def calculate_stop_levels(entry_data, risk_model, sl_multiplier, tp_multiplier):
    """
    根據選定的風險模型和入場數據，計算止損 (SL) 和止盈 (TP) 價格水平。
    """
    entry_price = entry_data['Close']
    
    if entry_data.isnull().any():
        return None, None
        
    # --- ATR 動態模型 (基礎波動率調整) ---
    if risk_model == 'ATR 動態模型':
        # 多頭: SL = 入場價 - SL_Multiplier * ATR, TP = 入場價 + TP_Multiplier * ATR
        stop_loss = entry_price - sl_multiplier * entry_data['ATR']
        take_profit = entry_price + tp_multiplier * entry_data['ATR']
        
    # --- 布林通道 (BB) 模型 (通道邊界) ---
    elif risk_model == '布林通道 (BB) 模型':
        # 多頭: SL = 中軌 - SL_Multiplier * ATR (增加緩衝避免中軌假跌破)
        # TP = 上軌 (BB_High)
        stop_loss = entry_data['BB_Mid'] - sl_multiplier * entry_data['ATR'] 
        take_profit = entry_data['BB_High'] 
        
    # --- 拋物線 SAR (PSAR) 追蹤模型 (動態追蹤) ---
    elif risk_model == '拋物線 SAR (PSAR) 追蹤模型':
        # SL = 當前 PSAR 點位 (在回測中動態更新)
        # TP 設為一個寬鬆的水平 (例如 6xATR) 讓 PSAR 追蹤止損發揮最大作用
        stop_loss = entry_data['PSAR_Up'] # 入場時的 PSAR 上軌
        take_profit = entry_price + 6.0 * entry_data['ATR']
        
    else:
        # 預設回到 ATR 模型
        stop_loss = entry_price - sl_multiplier * entry_data['ATR']
        take_profit = entry_price + tp_multiplier * entry_data['ATR']

    # 確保多頭策略下 SL < Entry < TP
    if stop_loss >= entry_price:
        stop_loss = entry_price * 0.95 # 如果計算結果無效，給予 5% 的保護
    if take_profit <= entry_price:
         take_profit = entry_price * 1.05 # 如果計算結果無效，給予 5% 的利潤空間

    return stop_loss, take_profit


# ==============================================================================
# 4. 趨勢信號與回測邏輯 (核心優化部分)
# ==============================================================================

def generate_trend_signal(df):
    """SMA 20/EMA 50 交叉信號 (維持 V3.0 的核心策略)"""
    df['Signal'] = 0
    
    # 買入信號：SMA 20 上穿 EMA 50
    df.loc[(df['SMA_20'].shift(1) <= df['EMA_50'].shift(1)) & (df['SMA_20'] > df['EMA_50']), 'Signal'] = 1
    
    # 賣出信號：SMA 20 下穿 EMA 50
    df.loc[(df['SMA_20'].shift(1) >= df['EMA_50'].shift(1)) & (df['SMA_20'] < df['EMA_50']), 'Signal'] = -1
    
    # 移除重複信號
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
    df['EntryExit'] = df['Position'].diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df.loc[df['EntryExit'] == 0, 'Signal'] = 0 

    return df

def backtest_strategy_with_risk_management(df, capital=100000, risk_model='ATR 動態模型', sl_multiplier=1.5, tp_multiplier=3.0):
    """
    🔥 V4.0 核心優化：執行回測策略，根據 `risk_model` 應用動態止損止盈。
    """
    df = df.copy().dropna(subset=['Signal', 'ATR', 'BB_High', 'PSAR_Up', 'PSAR_Down', 'BB_Mid']) 

    trades = []
    current_position = None
    entry_price = 0
    entry_index = None
    # 入場時計算的固定 SL/TP 水平
    stop_loss_price_initial = 0  
    take_profit_price = 0 

    for i in range(len(df)):
        current_data = df.iloc[i]
        current_close = current_data['Close']
        current_high = current_data['High']
        current_low = current_data['Low']
        
        # --- 1. 動態 SL 更新 (僅適用於 PSAR 模型) ---
        stop_loss_price_tracking = stop_loss_price_initial # 預設為初始 SL
        if current_position == 'Buy' and risk_model == '拋物線 SAR (PSAR) 追蹤模型':
            # PSAR 追蹤止損點會隨著價格上漲而不斷抬高
            # 我們使用前一根 K 線的 PSAR 點位作為當前的止損價
            if i > 0 and not pd.isna(df.iloc[i-1]['PSAR_Up']):
                stop_loss_price_tracking = max(stop_loss_price_initial, df.iloc[i-1]['PSAR_Up'])
            # 如果 PSAR 點位已經翻轉為 PSAR_Down，則視為信號反轉平倉 (在下面的平倉邏輯處理)
        
        # --- 2. 平倉邏輯 (優先檢查 SL/TP/PSAR 反轉) ---
        if current_position == 'Buy':
            
            # 2a. 止損檢查：使用動態或初始 SL
            if current_low <= stop_loss_price_tracking:
                # 止損平倉：假設以止損價平倉
                profit = (stop_loss_price_tracking - entry_price) / entry_price * 100 
                trades.append({'entry_date': entry_index, 'exit_date': df.index[i], 'type': 'Buy', 'profit': profit, 'status': 'SL', 'price': stop_loss_price_tracking, 'atr_sl': stop_loss_price_tracking, 'atr_tp': take_profit_price})
                current_position = None
                continue
            
            # 2b. 止盈檢查
            elif current_high >= take_profit_price and take_profit_price != 0:
                # 止盈平倉：假設以止盈價平倉
                profit = (take_profit_price - entry_price) / entry_price * 100 
                trades.append({'entry_date': entry_index, 'exit_date': df.index[i], 'type': 'Buy', 'profit': profit, 'status': 'TP', 'price': take_profit_price, 'atr_sl': stop_loss_price_tracking, 'atr_tp': take_profit_price})
                current_position = None
                continue
            
            # 2c. 反向信號平倉 (MA 交叉反轉)
            elif current_data['Signal'] == -1:
                profit = (current_close - entry_price) / entry_price * 100
                trades.append({'entry_date': entry_index, 'exit_date': df.index[i], 'type': 'Buy', 'profit': profit, 'status': 'Signal_Close', 'price': current_close, 'atr_sl': stop_loss_price_tracking, 'atr_tp': take_profit_price})
                current_position = None
                continue


        # --- 3. 開倉邏輯 ---
        if current_position is None and current_data['Signal'] == 1:
            
            entry_data = current_data
            
            # 根據模型計算初始 SL/TP
            sl_level, tp_level = calculate_stop_levels(entry_data, risk_model, sl_multiplier, tp_multiplier)
            
            if sl_level is not None and tp_level is not None:
                entry_price = current_close
                entry_index = df.index[i]
                stop_loss_price_initial = sl_level
                take_profit_price = tp_level
                current_position = 'Buy'
                
        # 4. 處理未平倉部位 (Open Position)
        if current_position == 'Buy' and i == len(df) - 1:
            last_close = df['Close'].iloc[-1]
            
            # 結算時 PSAR 追蹤止損的最終點位
            if risk_model == '拋物線 SAR (PSAR) 追蹤模型':
                 # 使用當前計算的追蹤止損價 (或最後一個 PSAR 點)
                 stop_loss_price_tracking = df['PSAR_Up'].iloc[-1] if not pd.isna(df['PSAR_Up'].iloc[-1]) else stop_loss_price_initial
            
            profit = (last_close - entry_price) / entry_price * 100
                
            trades.append({'entry_date': entry_index, 'exit_date': df.index[-1], 'type': current_position, 'profit': profit, 'status': 'Open', 'price': last_close, 'atr_sl': stop_loss_price_tracking, 'atr_tp': take_profit_price})

    # 統計回測結果
    trades_df = pd.DataFrame(trades)
    closed_trades_df = trades_df[trades_df['status'] != 'Open'].copy()
    
    capital_curve = pd.Series([capital], index=[df.index[0]])
    if not closed_trades_df.empty:
        closed_trades_df['return_factor'] = 1 + closed_trades_df['profit'] / 100
        closed_trades_df.set_index('exit_date', inplace=True)
        # 處理同一 K 線平倉多次導致的 index 重複問題
        closed_trades_df = closed_trades_df[~closed_trades_df.index.duplicated(keep='last')]
        
        temp_curve = closed_trades_df['return_factor'].cumprod() * capital
        capital_curve = pd.concat([capital_curve, temp_curve]).sort_index()
        # 確保曲線從 $100,000 開始，且不重複
        capital_curve = capital_curve[~capital_curve.index.duplicated(keep='first')]

    total_trades = len(trades_df)
    total_closed_trades = len(closed_trades_df)
    
    # 計算結果
    if total_closed_trades > 0 and len(capital_curve) > 1:
        win_trades = len(closed_trades_df[closed_trades_df['profit'] > 0])
        total_return = (capital_curve.iloc[-1] / capital_curve.iloc[0] - 1) * 100
        win_rate = (win_trades / total_closed_trades) * 100
        max_drawdown = ((capital_curve.cummax() - capital_curve) / capital_curve.cummax()).max() * 100
    else:
        # 如果沒有足夠的交易或曲線數據，設置為 0
        total_return = 0
        win_rate = 0
        max_drawdown = 0
        
    
    return {
        'total_return': round(total_return, 2),
        'win_rate': round(win_rate, 2),
        'max_drawdown': round(max_drawdown, 2),
        'total_trades': total_trades,
        'trades_summary': trades_df, 
        'capital_curve': capital_curve,
        'message': f"模型: {risk_model}"
    }


# ==============================================================================
# 5. 圖表繪製與 Streamlit 介面 (新增 PSAR 繪製)
# ==============================================================================

def calculate_score(df, symbol):
    # (維持 V3.0 的簡化趨勢評分邏輯)
    score = 0
    if df['SMA_20'].iloc[-1] > df['EMA_50'].iloc[-1]:
        score += 25 
    else:
        score -= 25 
        
    if df['RSI'].iloc[-1] < 30:
        score += 10 
    elif df['RSI'].iloc[-1] > 70:
        score -= 10 
        
    trend_analysis = "強勢多頭" if score > 30 else ("弱勢空頭" if score < -30 else "震盪盤整")
    
    return {
        'trend_analysis': trend_analysis,
        'current_score': score,
        'recommendation': "建議買入" if score > 30 else ("建議賣出" if score < -30 else "觀望")
    }

def create_comprehensive_chart(df, symbol, period):
    """繪製綜合 K 線圖，加入 BB/PSAR"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.6, 0.2, 0.2])

    # K線圖 (Row 1)
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='K線',
                                 increasing_line_color='#FF4500', 
                                 decreasing_line_color='#1E90FF'), row=1, col=1)

    # 均線 (Row 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='purple', width=1)), row=1, col=1)
    
    # 布林通道 (BB) - 可作為輔助支撐/阻力 (Row 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], mode='lines', name='BB 上軌', line=dict(color='lime', width=0.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], mode='lines', name='BB 下軌', line=dict(color='lime', width=0.5, dash='dash')), row=1, col=1)

    # 拋物線 SAR (PSAR) - 作為動態追蹤止損點 (Row 1)
    # 僅繪製 PSAR_Up (多頭模式下的追蹤止損點)
    fig.add_trace(go.Scatter(x=df.index, y=df['PSAR_Up'], mode='markers', name='PSAR 追蹤點', 
                             marker=dict(color='cyan', size=3, symbol='circle')), row=1, col=1)


    # MACD (Row 2)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Line'], mode='lines', name='MACD', line=dict(color='#FFD700')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='#008000')), row=2, col=1)

    # RSI (Row 3)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#DC143C')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[70] * len(df), mode='lines', name='RSI 70', line=dict(color='gray', dash='dash', width=0.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[30] * len(df), mode='lines', name='RSI 30', line=dict(color='gray', dash='dash', width=0.5)), row=3, col=1)

    fig.update_layout(title=f'{symbol} - {period} 綜合技術分析圖表 (含BB/PSAR)',
                      xaxis_rangeslider_visible=False, 
                      height=800, 
                      template="plotly_dark")
    fig.update_yaxes(title_text="價格 / 通道", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    return fig

# ==============================================================================
# 6. 主應用程式邏輯
# ==============================================================================

def app():
    
    # --- 側邊欄配置 (新增 SL/TP 模型選擇) ---
    st.sidebar.markdown("<h1 style='color: #FA8072; font-size: 24px; font-weight: bold;'>🎯 風險管理參數 (v4.0)</h1>", unsafe_allow_html=True)
    
    # 🔥 核心：選擇止盈止損模型
    risk_model = st.sidebar.selectbox("🛡️ 選擇止盈止損模型", 
                                      ['ATR 動態模型', '布林通道 (BB) 模型', '拋物線 SAR (PSAR) 追蹤模型'], 
                                      key='risk_model')
    
    st.sidebar.markdown("---")
    
    # 根據模型顯示對應的參數
    sl_multiplier = 1.5
    tp_multiplier = 3.0
    
    if risk_model == 'ATR 動態模型':
        sl_multiplier = st.sidebar.slider("📉 止損倍數 (SL xATR)", 0.5, 3.0, 1.5, 0.1)
        tp_multiplier = st.sidebar.slider("📈 止盈倍數 (TP xATR)", 1.0, 6.0, 3.0, 0.5)
        rr_ratio = round(tp_multiplier/sl_multiplier, 1) if sl_multiplier != 0 else np.inf
        st.sidebar.markdown(f"**ℹ️ 風險報酬比 (R:R)**: **1:{rr_ratio}**", unsafe_allow_html=True)
    
    elif risk_model == '布林通道 (BB) 模型':
        st.sidebar.markdown("**止盈**：上軌 (2σ) / **止損**：中軌 ± 波動率緩衝")
        sl_multiplier = st.sidebar.slider("📉 SL 波動率緩衝 (xATR)", 0.5, 3.0, 1.5, 0.1, help="此數值用於計算中軌下方的 SL 緩衝")
        tp_multiplier = 2.0 # TP 默認基於上軌，但此參數仍用於 ATR 波動率計算
    
    elif risk_model == '拋物線 SAR (PSAR) 追蹤模型':
        st.sidebar.markdown("**止損**：動態追蹤 PSAR 點位 (已固定參數)")
        st.sidebar.markdown("**止盈**：設為寬鬆值，專注於追蹤止損**")
        sl_multiplier = 1.5 # 保持參數存在，用於計算初始 SL
        tp_multiplier = 6.0 # 專注於追蹤止損，TP 設一個寬鬆值 (6xATR)
        
    st.sidebar.markdown("---")
    
    # (資產選擇與週期選擇邏輯... 與 V3.0 相同)
    
    st.sidebar.markdown("<h1 style='color: #FA8072; font-size: 24px; font-weight: bold;'>⚙️ 數據配置</h1>", unsafe_allow_html=True)
    
    category = st.sidebar.selectbox("選擇資產類別", ["美股/ETF/指數", "台股/ETF/指數", "加密貨幣"], key='category_selector')
    
    # 簡易篩選邏輯 (與 V3.0 保持一致)
    def filter_symbols(symbol_map, category):
        if category == "美股/ETF/指數":
            return {k: v for k, v in symbol_map.items() if not re.match(r'^\d+\.TW$', k) and '-USD' not in k}
        elif category == "台股/ETF/指數":
            return {k: v for k, v in symbol_map.items() if '.TW' in k}
        elif category == "加密貨幣":
            return {k: v for k, v in symbol_map.items() if '-USD' in k}
        return symbol_map
        
    filtered_map = filter_symbols(FULL_SYMBOLS_MAP, category)

    hot_keys = list(filtered_map.keys())
    hot_key_options = {k: f"{k} ({v['name']})" for k, v in filtered_map.items()}
    hot_key_list = [""] + [hot_key_options[k] for k in hot_keys]

    selected_option = st.sidebar.selectbox("或 選擇熱門標的", hot_key_list, index=0)

    selected_symbol_from_list = ""
    if selected_option and selected_option != "":
        match = re.search(r"^(.*?)\s\(", selected_option)
        selected_symbol_from_list = match.group(1) if match else selected_option.split(" (")[0]
        st.session_state['sidebar_search_input'] = selected_symbol_from_list 

    symbol_input = st.sidebar.text_input("或 輸入標的代碼/名稱", value=st.session_state.get('last_search_symbol', "2330.TW"), key='search_input')
    
    symbol = selected_symbol_from_list if selected_symbol_from_list else symbol_input
    
    period = st.sidebar.selectbox("選擇分析週期", list(PERIOD_MAP.keys()))
    period_tuple = PERIOD_MAP[period]
    
    # 執行按鈕
    if st.sidebar.button("📊 執行AI分析", key='analyze_button'):
        st.session_state['last_search_symbol'] = symbol
        st.session_state['data_ready'] = False
        
        with st.spinner(f"正在分析 {symbol} 的趨勢數據..."):
            df = get_data(symbol, period_tuple)
            if df is not None:
                df = calculate_technical_indicators(df)
                df = generate_trend_signal(df)
                
                # 執行 V4.0 回測策略
                bt = backtest_strategy_with_risk_management(
                    df, 
                    risk_model=risk_model, 
                    sl_multiplier=sl_multiplier, 
                    tp_multiplier=tp_multiplier
                )
                score_res = calculate_score(df, symbol)
                
                st.session_state['analysis_result'] = {
                    'df': df,
                    'symbol': symbol,
                    'period': period,
                    'backtest': bt,
                    'score': score_res,
                }
                st.session_state['data_ready'] = True
            else:
                st.error(f"無法獲取 {symbol} 的數據。請檢查代碼是否正確。")

    # --- 輸出結果區 ---
    if st.session_state.get('data_ready', False):
        res = st.session_state['analysis_result']
        bt = res['backtest']
        score_res = res['score']
        
        st.markdown(f"<h2 style='color: #4CAF50;'>🤖 {res['symbol']} ({res['period']}) AI趨勢分析報告 (v4.0)</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # 趨勢分析總結
        col1, col2 = st.columns([1, 2])
        col1.metric("🔍 趨勢判斷", score_res['trend_analysis'], delta=f"評分: {score_res['current_score']}", delta_color='off')
        col2.warning(f"💡 **AI 策略建議**：{score_res['recommendation']} (請結合下方 {risk_model} 回測結果進行決策)", icon="💡")

        st.markdown("---")
        
        # 回測結果
        st.subheader(f"🛡️ 動態風險管理回測結果 ({bt['message']})")
        if bt.get("total_trades", 0) > 0:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("📊 總回報率", f"{bt['total_return']}%", delta=bt['message'], delta_color='off')
            b2.metric("📈 勝率", f"{bt['win_rate']}%")
            b3.metric("📉 最大回撤", f"{bt['max_drawdown']}%")
            b4.metric("🤝 交易次數", f"{bt['total_trades']} 次")
            
            # 資金曲線圖
            if 'capital_curve' in bt and not bt['capital_curve'].empty and len(bt['capital_curve']) > 1:
                fig = go.Figure(go.Scatter(x=bt['capital_curve'].index, y=bt['capital_curve'], name='資金曲線', line=dict(color='#00FFFF')))
                fig.update_layout(title=f'{risk_model} 策略資金曲線', height=300, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            # 交易明細
            st.markdown("<h4 style='color: #FF6347;'>📃 交易紀錄與 SL/TP 追蹤</h4>", unsafe_allow_html=True)
            trades_df = bt['trades_summary'][['entry_date', 'exit_date', 'type', 'profit', 'status', 'atr_sl', 'atr_tp']].copy()
            trades_df['profit'] = trades_df['profit'].round(2).astype(str) + '%'
            
            trades_df['status'] = trades_df['status'].apply(lambda x: 
                f"✅ 止盈 ({x})" if x == 'TP' else (f"❌ 止損 ({x})" if x == 'SL' else (f"⚠️ 反轉平倉 ({x})" if x == 'Signal_Close' else f"⏳ 未平倉 ({x})"))
            )
            
            st.dataframe(trades_df, use_container_width=True, hide_index=True,
                column_config={
                    "entry_date": st.column_config.DatetimeColumn("入場時間", format="YYYY-MM-DD HH:mm"),
                    "exit_date": st.column_config.DatetimeColumn("出場時間/最新時間", format="YYYY-MM-DD HH:mm"),
                    "type": "方向",
                    "profit": "盈虧",
                    "status": "狀態",
                    "atr_sl": "SL 價格 (動態追蹤)",
                    "atr_tp": "TP 價格",
                }
            )

        else: 
            st.warning(f"回測無法執行或無交易：{bt.get('message', '錯誤')}")
            
        st.markdown("---")
        st.subheader(f"📊 完整技術分析圖表")
        st.plotly_chart(create_comprehensive_chart(res['df'], res['symbol'], res['period']), use_container_width=True)
    
    # --- 歡迎頁面 ---
    elif not st.session_state.get('data_ready', False):
        st.markdown("<h1 style='color: #FA8072; font-size: 32px; font-weight: bold;'>🚀 歡迎使用 AI 趨勢分析 (v4.0)</h1>", unsafe_allow_html=True)
        st.markdown(f"**🔥 新增功能**：多模型止損/止盈，您可以選擇 **ATR**、**布林通道 (BB)** 或 **PSAR** 進行回測。", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇止盈止損模型**：在左側欄選擇您希望測試的風險管理模型。")
        st.markdown("2. **調整模型參數**：根據選擇的模型，調整 `SL/TP 倍數` 或 `SL 波動率緩衝`。")
        st.markdown("3. **選擇資產/週期**：選擇您想分析的標的和時間週期。")
        st.markdown(f"4. **執行分析**：點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』**</span>，系統將使用您選擇的風險模型進行回測。", unsafe_allow_html=True)
        st.markdown("---")


if __name__ == '__main__':
    # Streamlit Session State 初始化，確保變數存在
    if 'last_search_symbol' not in st.session_state:
        st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state:
        st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state:
        st.session_state['sidebar_search_input'] = "2330.TW"
        
    app()
