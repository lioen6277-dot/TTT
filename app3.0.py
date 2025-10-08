import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import warnings
import re 
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 頁面配置與全局設定
# ==============================================================================

st.set_page_config(
    page_title="AI多因子量化趨勢分析 (v8.0)", 
    page_icon="📈", 
    layout="wide"
)

# 週期映射：(YFinance Period, YFinance Interval) - 採用最穩定的 Interval 配置
PERIOD_MAP = { 
    "30 分": ("60d", "30m"), 
    "4 小時": ("1y", "90m"), # 程式碼專家優化：使用 90m 替代 60m/4h 以提高 yfinance 穩定性
    "1 日": ("5y", "1d"), 
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】(簡化展示，實戰中可擴充)
FULL_SYMBOLS_MAP = {
    "TSLA": {"name": "特斯拉", "keywords": ["電動車"]},
    "NVDA": {"name": "輝達", "keywords": ["AI"]},
    "AAPL": {"name": "蘋果", "keywords": ["科技"]},
    "2330.TW": {"name": "台積電", "keywords": ["半導體"]},
    "0050.TW": {"name": "元大台灣50 ETF", "keywords": ["ETF"]},
    "BTC-USD": {"name": "比特幣", "keywords": ["加密貨幣"]},
}

# ==============================================================================
# 2. 數據獲取與預處理
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner="⏳ 正在從 Yahoo Finance 獲取數據...")
def get_stock_data(symbol, period, interval):
    """獲取股票/資產的歷史數據，並檢查數據完整性。"""
    try:
        # 使用 max interval 時，yfinance 不支援 period 參數
        if period == "max":
            data = yf.download(symbol, interval=interval, progress=False)
        else:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if data.empty:
            st.error(f"❌ 找不到代碼 {symbol} 的數據，請檢查輸入是否正確。")
            return None
        
        # 修正欄位名稱，統一為大寫
        data.columns = [col.capitalize() for col in data.columns]
        
        return data.dropna()
    except Exception as e:
        st.error(f"❌ 數據獲取失敗: {e}")
        return None

# ==============================================================================
# 3. 技術分析：實作進階數值設定
# ==============================================================================

def calculate_technical_indicators(df):
    """
    計算並添加進階技術分析指標，採用使用者設定的進階參數。
    """
    
    # --- 移動平均線 (MA) - 進階設定 ---
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    
    # --- 相對強弱指數 (RSI) - 進階設定：週期 9 期 + 濾鏡 ---
    df['RSI'] = ta.momentum.rsi(df['Close'], window=9)
    df['RSI_SMA'] = df['RSI'].rolling(window=3).mean() # 新增 SMA 濾鏡減噪音

    # --- MACD - 進階設定：8, 17, 9 ---
    macd = ta.trend.MACD(df['Close'], window_fast=8, window_slow=17, window_sign=9)
    df['MACD_Line'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff() 

    # --- 趨勢強度指標 (ADX) - 進階設定：週期 9 期 ---
    adx = ta.trend.ADX(df['High'], df['Low'], df['Close'], window=9)
    df['ADX'] = adx.adx()
    
    # --- 成交量 (Volume) - 進階設定：OBV + 20 期 MA ---
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    return df.dropna()

# ==============================================================================
# 4. 多因子量化評分系統 (核心邏輯)
# ==============================================================================

def fetch_simulated_external_data(symbol, current_price):
    """
    【重要：此為模擬數據】
    模擬獲取外部基本面、價值、消息、籌碼數據。
    實戰應用時，此處需替換為真實的 API 數據獲取。
    """
    # 根據資產類型模擬不同的數據傾向
    is_growth_stock = "NVDA" in symbol or "TSLA" in symbol
    
    data = {
        # 股價估值的判斷標準 (Value)
        "PE_Ratio": 35.0 if is_growth_stock else 12.0, 
        "PB_Ratio": 5.5 if is_growth_stock else 0.9,
        "Expected_Growth_Rate": 0.25 if is_growth_stock else 0.08, # 預期成長率
        # 基本面的判斷標準 (Fundamental)
        "ROE": 0.22 if is_growth_stock else 0.16, # 高成長股 ROE 更高
        "EPS_Growth_Rate": 0.30 if is_growth_stock else 0.12,
        "Debt_to_Equity": 0.3 if is_growth_stock else 0.45,
        "Current_Ratio": 2.5,
        # 籌碼面的判斷標準 (Chips)
        "Foreign_Investor_Buy_Ratio": 0.15 if "TW" in symbol else 0.0, # 假設外資買超
        "Margin_Trading_Change": 0.05, 
        # 消息面的判斷標準 (News/Sentiment)
        "Sentiment_Score": 0.8, # 正面情緒
        "News_Event": "財報超預期" if is_growth_stock else "穩定派息公告",
    }
    
    # 計算 PEG (P/E / 預期成長率 * 100)
    data["PEG_Ratio"] = data["PE_Ratio"] / (data["Expected_Growth_Rate"] * 100) if data["Expected_Growth_Rate"] > 0 else 99.0
    
    return data

def calculate_trend_score(df, external_data):
    """
    根據六大面向標準計算總量化評分。
    評分範圍: [-10, 10]
    """
    scores = {'Value': 0, 'Technical': 0, 'Fundamental': 0, 'News': 0, 'Chips': 0, 'Total': 0}
    
    if df.empty or df.shape[0] < 200:
        return scores, "數據不足，無法進行精準量化分析。", "Neutral", {}
    
    last = df.iloc[-1]
    
    # ----------------------------------------------------
    # I. 技術分析評分 (Technical Score) - 權重高 (±5)
    # ----------------------------------------------------
    tech_score = 0
    tech_details = []

    # 1. MA (10, 50, 200 EMA)
    is_golden_cross = (df['EMA_10'].iloc[-2] < df['EMA_50'].iloc[-2]) and (last['EMA_10'] > last['EMA_50'])
    is_up_arrangement = last['EMA_10'] > last['EMA_50'] > last['EMA_200']
    
    if is_up_arrangement: tech_score += 2; tech_details.append("MA: 強多頭向上排列 (+2)")
    elif is_golden_cross: tech_score += 1.5; tech_details.append("MA: 10/50 金叉確認 (+1.5)")
    elif last['EMA_10'] < last['EMA_50'] < last['EMA_200']: tech_score -= 2; tech_details.append("MA: 強空頭向下排列 (-2)")
    elif (df['EMA_10'].iloc[-2] > df['EMA_50'].iloc[-2]) and (last['EMA_10'] < last['EMA_50']): tech_score -= 1.5; tech_details.append("MA: 10/50 死叉確認 (-1.5)")
    
    # 2. RSI (9)
    if last['RSI_SMA'] > 70: tech_score -= 1; tech_details.append(f"RSI(9): >70 超買 (-1)")
    elif last['RSI_SMA'] > 50: tech_score += 1; tech_details.append("RSI(9): >50 多頭確認 (+1)")
    elif last['RSI_SMA'] < 30: tech_score += 1.5; tech_details.append(f"RSI(9): <30 超賣 (+1.5)")
    elif last['RSI_SMA'] < 50: tech_score -= 1; tech_details.append("RSI(9): <50 空頭確認 (-1)")

    # 3. MACD (8, 17, 9)
    if last['MACD_Hist'] > 0 and last['MACD_Line'] > 0: tech_score += 1.5; tech_details.append("MACD: Histogram >0 且零線以上 (+1.5)")
    elif last['MACD_Hist'] < 0 and last['MACD_Line'] < 0: tech_score -= 1.5; tech_details.append("MACD: Histogram <0 且零線以下 (-1.5)")
    
    # 4. ADX (9) - 趨勢強度確認 (ADX > 25)
    if last['ADX'] > 25:
        if tech_score > 0: tech_score += 1; tech_details.append(f"ADX(9): >25 強勢多頭趨勢 (+1)")
        elif tech_score < 0: tech_score -= 1; tech_details.append(f"ADX(9): >25 強勢空頭趨勢 (-1)")

    # 5. 成交量 (Volume) - 量價配合
    last_volume_ratio = last['Volume'] / last['Volume_MA_20']
    if last_volume_ratio > 1.5:
        if last['Close'] > df['Close'].iloc[-2]: tech_score += 1.5; tech_details.append("Volume: 價格上漲 + 量能放大 (+1.5)")
        elif last['Close'] < df['Close'].iloc[-2]: tech_score -= 1.5; tech_details.append("Volume: 價格下跌 + 量能放大 (-1.5)")
        
    scores['Technical'] = round(tech_score, 2)
    
    # ----------------------------------------------------
    # II. 價值/估值評分 (Value Score) - 權重中 (±3)
    # ----------------------------------------------------
    value_score = 0
    value_details = []

    # P/E Ratio (< 15 低估)
    if external_data['PE_Ratio'] < 15: value_score += 1.5; value_details.append(f"P/E: {external_data['PE_Ratio']:.2f} (<15 低估) (+1.5)")
    # PEG Ratio (PEG < 1 低估/高成長)
    if external_data['PEG_Ratio'] < 1: value_score += 1.5; value_details.append(f"PEG: {external_data['PEG_Ratio']:.2f} (<1 高成長/低估) (+1.5)")
    elif external_data['PEG_Ratio'] > 2: value_score -= 1; value_details.append(f"PEG: {external_data['PEG_Ratio']:.2f} (成長被高估) (-1)")
    
    scores['Value'] = round(value_score, 2)

    # ----------------------------------------------------
    # III. 基本面評分 (Fundamental Score) - 權重中 (±3)
    # ----------------------------------------------------
    fundamental_score = 0
    fundamental_details = []

    # 獲利能力 (ROE > 15%)
    if external_data['ROE'] > 0.15: fundamental_score += 1.5; fundamental_details.append(f"ROE: {external_data['ROE']*100:.1f}% (>15% 高效資本) (+1.5)")
    # 成長與效率 (EPS Growth > 10%)
    if external_data['EPS_Growth_Rate'] > 0.10: fundamental_score += 1; fundamental_details.append(f"EPS Growth: {external_data['EPS_Growth_Rate']*100:.1f}% (>10% 穩定成長) (+1)")
    # 財務健康 (Debt/Equity < 0.5)
    if external_data['Debt_to_Equity'] < 0.5: fundamental_score += 0.5; fundamental_details.append(f"Health: D/E <0.5 (低風險) (+0.5)")
    
    scores['Fundamental'] = round(fundamental_score, 2)

    # ----------------------------------------------------
    # IV. 消息面評分 (News Score) - 權重低 (±2)
    # ----------------------------------------------------
    news_score = 0
    news_details = []
    
    # 情緒指標 (Sentiment > 0.7 正面)
    if external_data['Sentiment_Score'] > 0.7: news_score += 1.5; news_details.append("Sentiment: >0.7 (市場情緒非常樂觀) (+1.5)")
    elif "超預期" in external_data['News_Event']: news_score += 0.5; news_details.append("Event: 業績超預期 (正面公告) (+0.5)")
    elif external_data['Sentiment_Score'] < 0.3: news_score -= 1.5; news_details.append("Sentiment: <0.3 (市場情緒悲觀) (-1.5)")

    scores['News'] = round(news_score, 2)

    # ----------------------------------------------------
    # V. 籌碼面評分 (Chips Score) - 權重低 (±2)
    # ----------------------------------------------------
    chips_score = 0
    chips_details = []

    # 外資買超 > 10%
    if external_data['Foreign_Investor_Buy_Ratio'] > 0.10: chips_score += 1.5; chips_details.append(f"Foreign Buy: >10% (看好買入) (+1.5)")
    # 融資增加 + 價格上漲
    if external_data['Margin_Trading_Change'] > 0.05 and last['Close'] > df['Close'].iloc[-2]: chips_score += 0.5; chips_details.append("Margin: 融資增加且股價上漲 (+0.5)")
    
    scores['Chips'] = round(chips_score, 2)

    # ----------------------------------------------------
    # VI. 總量化評分與趨勢判斷
    # ----------------------------------------------------
    total_score = sum(scores.values())
    scores['Total'] = round(total_score, 2)
    
    # 最終判斷邏輯
    if total_score >= 4.0:
        trend_judgement, action = "強烈多頭趨勢", "Strong Buy"
    elif total_score >= 1.5:
        trend_judgement, action = "偏多頭趨勢", "Buy"
    elif total_score <= -4.0:
        trend_judgement, action = "強烈空頭趨勢", "Strong Sell"
    elif total_score <= -1.5:
        trend_judgement, action = "偏空頭趨勢", "Sell"
    else:
        trend_judgement, action = "盤整或中性趨勢", "Hold/Wait"

    final_details = {k: v for k, v in zip(['Technical', 'Value', 'Fundamental', 'News', 'Chips'], 
                                          [tech_details, value_details, fundamental_details, news_details, chips_details])}

    return scores, trend_judgement, action, final_details


def generate_ai_analysis(df, symbol, period, external_data, scores, judgement, action, details):
    """根據數據和量化評分，生成 AI 趨勢分析報告。"""
    if df.empty: return "無法生成分析報告，數據獲取失敗或數據不足。"
    current_price = df['Close'].iloc[-1]
    adx_val = df['ADX'].iloc[-1]
    adx_color = "red" if adx_val > 40 else ("orange" if adx_val > 25 else "gray")
    color_map = {"Strong Buy": "green", "Buy": "lightgreen", "Sell": "red", "Strong Sell": "darkred", "Hold/Wait": "gray"}
    action_color = color_map.get(action, "gray")
    
    # 格式化 Technical Details 輸出
    tech_summary = [f"<li>**{d.split(':')[0]}**:{d.split(':')[1].replace('(','**(').replace(')','**)')}</li>" for d in details['Technical']]
    
    markdown_output = f"""
<div style="border: 2px solid #FA8072; padding: 15px; border-radius: 10px; background-color: #333333;">
<h3 style='color: white; border-bottom: 2px solid #FA8072; padding-bottom: 5px;'>🤖 AI 多因子趨勢分析報告：{symbol} ({period} 週期)</h3>

<div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
    <p style='font-size: 18px; color: white;'>
        **當前價格 (Close)**: <span style='color: #FFD700; font-weight: bold;'>${current_price:,.2f}</span>
    </p>
    <p style='font-size: 18px; color: white;'>
        **總量化評分**: <span style='color: {"green" if scores["Total"] >= 0 else "red"}; font-weight: bold;'>{scores["Total"]:+.2f} / 10.0</span>
    </p>
</div>

<p style='font-size: 24px; text-align: center; padding: 10px; border: 1px solid {action_color}; border-radius: 5px; background-color: #{action_color}33;'>
    **📈 最終趨勢判斷**: <span style='color: {action_color}; font-weight: bold;'>{judgement}</span>
    <br>
    **🎯 建議操作**: <span style='color: {action_color}; font-weight: bold;'>{action}</span>
</p>
</div>

---

#### 📊 多因子評分細節 (Total Score: {scores["Total"]:+.2f})

<div style="display: flex; justify-content: space-around; text-align: center;">
    <div style="width: 18%;">
        <h4 style="color: #6495ED;">技術分析 (Technical)</h4>
        <p style="font-size: 20px; font-weight: bold; color: {'green' if scores['Technical'] >= 0 else 'red'};">{scores['Technical']:+.2f}</p>
    </div>
    <div style="width: 18%;">
        <h4 style="color: #FFC0CB;">價值估值 (Value)</h4>
        <p style="font-size: 20px; font-weight: bold; color: {'green' if scores['Value'] >= 0 else 'red'};">{scores['Value']:+.2f}</p>
    </div>
    <div style="width: 18%;">
        <h4 style="color: #F08080;">基本面 (Fundamental)</h4>
        <p style="font-size: 20px; font-weight: bold; color: {'green' if scores['Fundamental'] >= 0 else 'red'};">{scores['Fundamental']:+.2f}</p>
    </div>
    <div style="width: 18%;">
        <h4 style="color: #FFD700;">消息情緒 (News)</h4>
        <p style="font-size: 20px; font-weight: bold; color: {'green' if scores['News'] >= 0 else 'red'};">{scores['News']:+.2f}</p>
    </div>
    <div style="width: 18%;">
        <h4 style="color: #98FB98;">籌碼資金 (Chips)</h4>
        <p style="font-size: 20px; font-weight: bold; color: {'green' if scores['Chips'] >= 0 else 'red'};">{scores['Chips']:+.2f}</p>
    </div>
</div>

---

### 📝 因子分析報告

<details>
<summary style="font-size: 20px; color: #6495ED;">**I. 技術分析 (Technical Score: {scores['Technical']:+.2f})**</summary>
<div style="padding-left: 15px; border-left: 3px solid #6495ED;">
**趨勢強度 ADX(9)**: <span style='color: {adx_color}; font-weight: bold;'>{adx_val:.2f}</span> ({'強趨勢' if adx_val > 25 else '弱勢或盤整'})
<ul>
""" + "".join(tech_summary) + """
</ul>
</div>
</details>

<details>
<summary style="font-size: 20px; color: #FFC0CB;">**II. 股價估值 (Value Score: {scores['Value']:+.2f})**</summary>
<div style="padding-left: 15px; border-left: 3px solid #FFC0CB;">
**主要數據**: P/E={external_data['PE_Ratio']:.2f}, P/B={external_data.get('PB_Ratio', 'N/A'):.2f}, PEG={external_data['PEG_Ratio']:.2f}
<ul>
""" + "".join([f"<li>{item}</li>" for item in details['Value']]) + """
</ul>
</div>
</details>

<details>
<summary style="font-size: 20px; color: #F08080;">**III. 基本面 (Fundamental Score: {scores['Fundamental']:+.2f})**</summary>
<div style="padding-left: 15px; border-left: 3px solid #F08080;">
**主要數據**: ROE={external_data['ROE']*100:.1f}%, EPS Growth={external_data['EPS_Growth_Rate']*100:.1f}%, Debt/Equity={external_data['Debt_to_Equity']:.2f}
<ul>
""" + "".join([f"<li>{item}</li>" for item in details['Fundamental']]) + """
</ul>
</div>
</details>

<details>
<summary style="font-size: 20px; color: #FFD700;">**IV. 消息情緒 (News Score: {scores['News']:+.2f})**</summary>
<div style="padding-left: 15px; border-left: 3px solid #FFD700;">
**主要事件**: {external_data['News_Event']}
<ul>
""" + "".join([f"<li>{item}</li>" for item in details['News']]) + """
</ul>
</div>
</details>

<details>
<summary style="font-size: 20px; color: #98FB98;">**V. 籌碼資金 (Chips Score: {scores['Chips']:+.2f})**</summary>
<div style="padding-left: 15px; border-left: 3px solid #98FB98;">
**主要數據**: 外資買超={external_data['Foreign_Investor_Buy_Ratio']*100:.1f}%, 融資變化={external_data['Margin_Trading_Change']*100:.1f}%
<ul>
""" + "".join([f"<li>{item}</li>" for item in details['Chips']]) + """
</ul>
</div>
</details>

---
<p style='color: gray; font-size: 14px;'>**免責聲明**: 本報告基於量化模型計算，其中**估值/基本面/消息/籌碼數據為模擬值**。投資決策請以實際數據為準，並自行承擔風險。</p>
"""
    return markdown_output

# ==============================================================================
# 5. K線圖與指標繪製
# ==============================================================================

def plot_stock_data(df, symbol):
    """繪製 K 線圖，並加入 MA 和技術指標 (RSI, MACD, Volume/OBV/ADX)。"""
    
    # 創建子圖：1個價格圖，3個技術指標圖
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.55, 0.15, 0.15, 0.15]
    )

    # --- Row 1: K線圖 & MA ---
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K-Line'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_10'], mode='lines', name='EMA 10', line=dict(color='yellow', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='pink', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200', line=dict(color='orange', width=2)), row=1, col=1)

    # --- Row 2: RSI ---
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI(9)', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_SMA'], mode='lines', name='RSI SMA(3)', line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    # --- Row 3: MACD ---
    histogram_colors = ['red' if val >= 0 else 'green' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist', marker_color=histogram_colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Line'], mode='lines', name='MACD (8,17)', line=dict(color='yellow', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal (9)', line=dict(color='red', width=1.5)), row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    # --- Row 4: 成交量 & OBV & ADX (雙 Y 軸) ---
    volume_colors = ['red' if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else 'green' for i in range(1, len(df))]
    volume_colors.insert(0, 'gray')
    
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=volume_colors), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA_20'], mode='lines', name='Vol MA 20', line=dict(color='pink', width=1)), row=4, col=1)
    
    # ADX (第二個 Y 軸)
    fig.update_layout(yaxis4=dict(title='ADX', side='right', overlaying='y4', position=0.98, showgrid=False))
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX(9)', line=dict(color='orange', width=2)), row=4, col=1, yaxis='y4')
    
    # OBV (第三個 Y 軸)
    fig.update_layout(yaxis5=dict(title='OBV', side='right', overlaying='y4', position=1.0, showgrid=False, range=[df['OBV'].min(), df['OBV'].max()]))
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', name='OBV', line=dict(color='cyan', width=1)), row=4, col=1, yaxis='y5')
    
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    fig.update_layout(title=f'{symbol} K線圖與進階技術指標', xaxis_rangeslider_visible=False, hovermode="x unified", height=900, template="plotly_dark")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    return fig

# ==============================================================================
# 6. Streamlit 應用程式主體
# ==============================================================================

def filter_symbols(category):
    """根據類別篩選符號列表。"""
    if category == "美股": return [s for s in FULL_SYMBOLS_MAP if not any(k in s for k in ['.TW', 'USD'])]
    elif category == "台股": return [s for s in FULL_SYMBOLS_MAP if '.TW' in s]
    elif category == "加密貨幣": return [s for s in FULL_SYMBOLS_MAP if 'USD' in s]
    return list(FULL_SYMBOLS_MAP.keys())

def sync_text_input_from_selection():
    """同步下拉選單與文字輸入框。"""
    try:
        selected_hot_key = st.session_state.hot_symbol_selector
        if selected_hot_key and selected_hot_key != "--- 選擇熱門標的 ---":
            symbol_match = re.search(r'\((\S+)\)', selected_hot_key)
            st.session_state['search_input_box'] = symbol_match.group(1) if symbol_match else selected_hot_key
    except Exception:
        pass


def main():
    st.sidebar.title("🔍 參數設定")
    
    # --- 側邊欄控制項 ---
    category_options = ["美股", "台股", "加密貨幣"]
    if 'category_selector' not in st.session_state: st.session_state['category_selector'] = category_options[0]

    st.sidebar.selectbox("選擇資產類別:", category_options, key='category_selector', on_change=sync_text_input_from_selection)

    hot_symbols = filter_symbols(st.session_state['category_selector'])
    hot_symbol_options = ["--- 選擇熱門標的 ---"] + [
        f"{FULL_SYMBOLS_MAP[s]['name']} ({s})" if 'name' in FULL_SYMBOLS_MAP[s] else s
        for s in hot_symbols
    ]

    st.sidebar.selectbox("快速選擇熱門標的:", hot_symbol_options, key='hot_symbol_selector', on_change=sync_text_input_from_selection)

    if 'search_input_box' not in st.session_state: st.session_state['search_input_box'] = ""
    
    symbol_input = st.sidebar.text_input(
        "直接輸入代碼或名稱 (例如: NVDA, 2330.TW)", 
        value=st.session_state['search_input_box'], 
        key='search_input_box'
    ).upper().strip() 

    period_label_map = {k: f"{k} ({'超短期' if k=='30 分' else '波段' if k=='4 小時' else '中長線' if k=='1 日' else '長期'})" for k in PERIOD_MAP}
    selected_period_key = st.sidebar.selectbox("選擇分析週期:", list(PERIOD_MAP.keys()), format_func=lambda x: period_label_map.get(x, x))

    if st.sidebar.button("📊 執行AI多因子分析"):
        if not symbol_input:
            st.error("請輸入資產代碼或選擇熱門標的！"); st.session_state['data_ready'] = False; return
        
        st.session_state['last_search_symbol'] = symbol_input
        st.session_state['data_ready'] = False

        with st.spinner(f"⏳ 正在分析 {symbol_input} 的數據..."):
            
            period_val, interval_val = PERIOD_MAP[selected_period_key]
            df_raw = get_stock_data(symbol_input, period_val, interval_val)
            
            if df_raw is not None and not df_raw.empty:
                df_tech = calculate_technical_indicators(df_raw.copy())
                current_price = df_tech['Close'].iloc[-1]
                external_data = fetch_simulated_external_data(symbol_input, current_price)
                scores, judgement, action, details = calculate_trend_score(df_tech, external_data)
                analysis_report = generate_ai_analysis(df_tech.tail(1), symbol_input, selected_period_key, external_data, scores, judgement, action, details)
                chart = plot_stock_data(df_tech.tail(500), symbol_input)

                st.session_state['analysis_report'] = analysis_report
                st.session_state['chart'] = chart
                st.session_state['data_ready'] = True
            else:
                 st.session_state['data_ready'] = False
                 st.error(f"❌ 無法獲取或計算 {symbol_input} 的完整數據。")


    # --- 主頁面顯示區 ---
    st.title("📈 AI 多因子量化趨勢分析系統")
    
    if st.session_state.get('data_ready', False):
        st.markdown(st.session_state['analysis_report'], unsafe_allow_html=True)
        st.plotly_chart(st.session_state['chart'], use_container_width=True)
        
        with st.expander("🛠️ 點此查看原始數據 (含所有指標)"):
            st.dataframe(df_tech.tail(10), use_container_width=True) # 顯示最近 10 筆數據

    elif not st.session_state.get('data_ready', False):
        st.markdown("<h1 style='color: #FA8072; font-size: 32px; font-weight: bold;'>🚀 歡迎使用 AI 多因子量化趨勢分析系統</h1>", unsafe_allow_html=True)
        st.markdown(f"本系統集成了**價值、技術、基本面、消息面、籌碼面**等多因子量化評分邏輯，旨在提供穩定且精準的趨勢分析。請在左側選擇或輸入您想分析的標的（例如：**2330.TW**、**NVDA**、**BTC-USD**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI多因子分析』</span> 按鈕開始。", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇標的**：使用下拉選單快速選擇熱門標的，或直接在輸入框中鍵入代碼或名稱。")
        st.markdown("3. **選擇週期**：決定分析的長度。")
        st.markdown(f"4. **執行分析**：點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI多因子分析』**</span>，AI將融合多重指標提供交易策略。", unsafe_allow_html=True)

if __name__ == '__main__':
    # Streamlit Session State 初始化，避免 Key Error
    if 'last_search_symbol' not in st.session_state: st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state: st.session_state['data_ready'] = False
    if 'search_input_box' not in st.session_state: st.session_state['search_input_box'] = ""
    
    main()
