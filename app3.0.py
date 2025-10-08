import re
import time
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf
from plotly.subplots import make_subplots

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
    "4 小時": ("1y", "60m"),
    "1 日": ("5y", "1d"),
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】(FULL_SYMBOLS_MAP)
FULL_SYMBOLS_MAP = {
    # ----------------------------------------------------
    # 美股/ETF/指數 (US Stocks/ETFs/Indices) - 排序依據: 英文名稱 (name 欄位)
    # ----------------------------------------------------
    "ACN": {"name": "Accenture (埃森哲)", "keywords": ["Accenture", "ACN", "諮詢", "科技服務"]},
    "ADBE": {"name": "Adobe", "keywords": ["Adobe", "ADBE"]},
    "AAPL": {"name": "蘋果 (Apple)", "keywords": ["蘋果", "Apple", "AAPL"]},
    "AMD": {"name": "超微 (Advanced Micro Devices)", "keywords": ["超微", "AMD", "半導體"]},
    "AMZN": {"name": "亞馬遜 (Amazon)", "keywords": ["亞馬遜", "Amazon", "AMZN", "電商"]},
    "ARKG": {"name": "方舟基因體革命ETF (ARK Genomic)", "keywords": ["ARKG", "基因科技", "生物科技ETF"]},
    "ARKK": {"name": "方舟創新ETF (ARK Innovation)", "keywords": ["ARKK", "CathieWood", "創新ETF", "木頭姐"]},
    "BA": {"name": "波音 (Boeing)", "keywords": ["波音", "Boeing", "BA", "工業股", "航太"]},
    "BAC": {"name": "美國銀行 (Bank of America)", "keywords": ["美國銀行", "BankOfAmerica", "BAC", "金融股"]},
    "BND": {"name": "Vanguard總體債券市場ETF", "keywords": ["BND", "總體債券", "債券ETF"]},
    "BRK-B": {"name": "波克夏海瑟威 B (Berkshire Hathaway)", "keywords": ["波克夏", "巴菲特", "BRKB", "保險", "投資"]},
    "CAT": {"name": "開拓重工 (Caterpillar)", "keywords": ["開拓重工", "Caterpillar", "CAT"]},
    "CVX": {"name": "雪佛龍 (Chevron)", "keywords": ["雪佛龍", "Chevron", "CVX", "能源股", "石油"]},
    "KO": {"name": "可口可樂 (Coca-Cola)", "keywords": ["可口可樂", "CocaCola", "KO"]},
    "COST": {"name": "好市多 (Costco)", "keywords": ["好市多", "Costco", "COST"]},
    "CRM": {"name": "Salesforce", "keywords": ["Salesforce", "CRM", "雲端", "SaaS"]},
    "DE": {"name": "迪爾公司 (Deere & Co.)", "keywords": ["迪爾", "Deere", "DE", "農業機械"]},
    "DIA": {"name": "SPDR 道瓊工業ETF (Dow Jones ETF)", "keywords": ["DIA", "道瓊ETF"]},
    "DIS": {"name": "迪士尼 (Disney)", "keywords": ["迪士尼", "Disney", "DIS", "媒體", "娛樂"]},
    "^DJI": {"name": "道瓊工業指數 (Dow Jones Industrial Average)", "keywords": ["道瓊", "DowJones", "^DJI", "指數"]},
    "DXY": {"name": "美元指數 (Dollar Index)", "keywords": ["美元指數", "DXY", "外匯", "USD"]},
    "EEM": {"name": "iShares 新興市場ETF (Emerging Markets)", "keywords": ["EEM", "新興市場", "新興市場ETF"]},
    "XOM": {"name": "埃克森美孚 (ExxonMobil)", "keywords": ["埃克森美孚", "ExxonMobil", "XOM", "能源股"]},
    "^FTSE": {"name": "富時100指數 (FTSE 100)", "keywords": ["富時", "倫敦股市", "^FTSE", "指數"]},
    "FUTY": {"name": "富時公用事業ETF (Utilities ETF)", "keywords": ["FUTY", "公用事業", "防禦股"]},
    "^GDAXI": {"name": "德國DAX指數", "keywords": ["DAX", "德國股市", "^GDAXI", "指數"]},
    "GLD": {"name": "SPDR黃金ETF (Gold ETF)", "keywords": ["GLD", "黃金ETF", "避險資產"]},
    "GOOG": {"name": "谷歌/Alphabet C股 (Google C)", "keywords": ["谷歌C", "Alphabet C", "GOOG"]},
    "GOOGL": {"name": "谷歌/Alphabet A股 (Google A)", "keywords": ["谷歌", "Alphabet", "GOOGL", "GOOG"]},
    "^GSPC": {"name": "S&P 500 指數", "keywords": ["標普", "S&P500", "^GSPC", "SPX", "指數"]},
    "GS": {"name": "高盛集團 (Goldman Sachs)", "keywords": ["高盛", "GoldmanSachs", "GS", "投行", "金融股"]},
    "HD": {"name": "家得寶 (Home Depot)", "keywords": ["家得寶", "HomeDepot", "HD"]},
    "INTC": {"name": "英特爾 (Intel)", "keywords": ["英特爾", "Intel", "INTC", "半導體"]},
    "IJR": {"name": "iShares 核心標普小型股ETF (Small Cap)", "keywords": ["IJR", "小型股ETF", "Russell2000"]},
    "IYR": {"name": "iShares 美國房地產ETF (Real Estate)", "keywords": ["IYR", "房地產ETF", "REITs"]},
    "JNJ": {"name": "嬌生 (Johnson & Johnson)", "keywords": ["嬌生", "Johnson&Johnson", "JNJ", "醫療保健"]},
    "JPM": {"name": "摩根大通 (JPMorgan Chase)", "keywords": ["摩根大通", "JPMorgan", "JPM", "金融股"]},
    "LLY": {"name": "禮來 (Eli Lilly)", "keywords": ["禮來", "EliLilly", "LLY", "製藥"]},
    "LMT": {"name": "洛克希德·馬丁 (Lockheed Martin)", "keywords": ["洛克希德馬丁", "LMT", "軍工", "國防"]},
    "LULU": {"name": "Lululemon", "keywords": ["Lululemon", "LULU", "運動服飾", "消費股"]},
    "MA": {"name": "萬事達卡 (Mastercard)", "keywords": ["萬事達卡", "Mastercard", "MA", "支付"]},
    "MCD": {"name": "麥當勞 (McDonald's)", "keywords": ["麥當勞", "McDonalds", "MCD"]},
    "META": {"name": "Meta/臉書 (Facebook)", "keywords": ["臉書", "Meta", "FB", "META", "Facebook"]},
    "MGM": {"name": "美高梅國際酒店集團 (MGM Resorts)", "keywords": ["美高梅", "MGM", "娛樂", "博彩"]},
    "MSFT": {"name": "微軟 (Microsoft)", "keywords": ["微軟", "Microsoft", "MSFT", "雲端", "AI"]},
    "MS": {"name": "摩根士丹利 (Morgan Stanley)", "keywords": ["摩根士丹利", "MorganStanley", "MS", "投行"]},
    "MRNA": {"name": "莫德納 (Moderna)", "keywords": ["莫德納", "Moderna", "MRNA", "生物科技", "疫苗"]},
    "MSCI": {"name": "MSCI ACWI ETF", "keywords": ["MSCI", "全球股票ETF"]},
    "^IXIC": {"name": "NASDAQ 綜合指數", "keywords": ["納斯達克", "NASDAQ", "^IXIC", "指數", "科技股"]},
    "^N225": {"name": "日經225指數 (Nikkei 225)", "keywords": ["日經", "Nikkei", "^N225", "日本股市", "指數"]},
    "NFLX": {"name": "網飛 (Netflix)", "keywords": ["網飛", "Netflix", "NFLX"]},
    "NKE": {"name": "耐克 (Nike)", "keywords": ["耐克", "Nike", "NKE", "運動用品"]},
    "NOW": {"name": "ServiceNow", "keywords": ["ServiceNow", "NOW", "SaaS", "企業軟體"]},
    "NVDA": {"name": "輝達 (Nvidia)", "keywords": ["輝達", "英偉達", "AI", "NVDA", "Nvidia", "GPU", "半導體"]},
    "ORCL": {"name": "甲骨文 (Oracle)", "keywords": ["甲骨文", "Oracle", "ORCL"]},
    "PEP": {"name": "百事 (PepsiCo)", "keywords": ["百事", "Pepsi", "PEP"]},
    "PFE": {"name": "輝瑞 (Pfizer)", "keywords": ["輝瑞", "Pfizer", "PFE", "製藥", "疫苗"]},
    "PG": {"name": "寶潔 (Procter & Gamble)", "keywords": ["寶潔", "P&G", "PG"]},
    "PYPL": {"name": "PayPal", "keywords": ["PayPal", "PYPL", "金融科技", "Fintech"]},
    "QCOM": {"name": "高通 (Qualcomm)", "keywords": ["高通", "Qualcomm", "QCOM", "半導體"]},
    "QQQM": {"name": "Invesco NASDAQ 100 ETF (低費率)", "keywords": ["QQQM", "納斯達克ETF", "科技股ETF"]},
    "QQQ": {"name": "Invesco QQQ Trust", "keywords": ["QQQ", "納斯達克ETF", "科技股ETF"]},
    "RTX": {"name": "雷神技術 (Raytheon Technologies)", "keywords": ["雷神", "Raytheon", "RTX", "軍工", "航太國防"]},
    "SCHD": {"name": "Schwab美國高股息ETF (High Dividend)", "keywords": ["SCHD", "高股息ETF", "美股派息"]},
    "SBUX": {"name": "星巴克 (Starbucks)", "keywords": ["星巴克", "Starbucks", "SBUX", "消費股"]},
    "SIRI": {"name": "Sirius XM", "keywords": ["SiriusXM", "SIRI", "媒體", "廣播"]},
    "SMH": {"name": "VanEck Vectors半導體ETF", "keywords": ["SMH", "半導體ETF", "晶片股"]},
    "SPY": {"name": "SPDR 標普500 ETF", "keywords": ["SPY", "標普ETF"]},
    "TLT": {"name": "iShares 20年期以上公債ETF (Treasury Bond)", "keywords": ["TLT", "美債", "公債ETF"]},
    "TSLA": {"name": "特斯拉 (Tesla)", "keywords": ["特斯拉", "電動車", "TSLA", "Tesla"]},
    "UNH": {"name": "聯合健康 (UnitedHealth Group)", "keywords": ["聯合健康", "UNH", "醫療保健"]},
    "USO": {"name": "美國石油基金ETF (Oil Fund)", "keywords": ["USO", "石油ETF", "原油"]},
    "V": {"name": "Visa", "keywords": ["Visa", "V"]},
    "VGT": {"name": "Vanguard資訊科技ETF (Tech ETF)", "keywords": ["VGT", "科技ETF", "資訊科技"]},
    "^VIX": {"name": "恐慌指數 (VIX)", "keywords": ["VIX", "恐慌指數", "波動率指數"]},
    "VNQ": {"name": "Vanguard房地產ETF (Real Estate)", "keywords": ["VNQ", "房地產ETF", "REITs"]},
    "VOO": {"name": "Vanguard 標普500 ETF", "keywords": ["VOO", "Vanguard"]},
    "VTI": {"name": "Vanguard整體股市ETF (Total Market)", "keywords": ["VTI", "整體股市", "TotalMarket"]},
    "VZ": {"name": "威瑞森 (Verizon)", "keywords": ["威瑞森", "Verizon", "VZ", "電信股"]},
    "WBA": {"name": "沃爾格林 (Walgreens Boots Alliance)", "keywords": ["沃爾格林", "Walgreens", "WBA", "藥品零售"]},
    "WFC": {"name": "富國銀行 (Wells Fargo)", "keywords": ["富國銀行", "WellsFargo", "WFC", "金融股"]},
    "WMT": {"name": "沃爾瑪 (Walmart)", "keywords": ["沃爾瑪", "Walmart", "WMT"]},

    # ----------------------------------------------------
    # 台股/ETF/指數 (TW Stocks/ETFs/Indices) - 排序依據: 代碼數字 (Key)
    # ----------------------------------------------------
    "0050.TW": {"name": "元大台灣50", "keywords": ["台灣50", "0050", "台灣五十", "ETF"]},
    "0051.TW": {"name": "元大中型100", "keywords": ["中型100", "0051", "ETF"]},
    "0055.TW": {"name": "元大MSCI金融", "keywords": ["元大金融", "0055", "金融股ETF"]},
    "0056.TW": {"name": "元大高股息", "keywords": ["高股息", "0056", "ETF"]},
    "006208.TW": {"name": "富邦台50", "keywords": ["富邦台50", "006208", "台灣五十ETF"]},
    "00679B.TW": {"name": "元大美債20年", "keywords": ["00679B", "美債ETF", "債券ETF"]},
    "00687B.TW": {"name": "國泰20年美債", "keywords": ["00687B", "美債ETF", "債券ETF"]},
    "00713.TW": {"name": "元大台灣高息低波", "keywords": ["00713", "高息低波", "ETF"]},
    "00878.TW": {"name": "國泰永續高股息", "keywords": ["00878", "國泰永續", "ETF"]},
    "00888.TW": {"name": "永豐台灣ESG", "keywords": ["00888", "ESG", "ETF"]},
    "00891.TW": {"name": "富邦特選高股息30", "keywords": ["00891", "高股息30", "ETF"]},
    "00919.TW": {"name": "群益台灣精選高股息", "keywords": ["00919", "群益高股息", "ETF"]},
    "00929.TW": {"name": "復華台灣科技優息", "keywords": ["00929", "科技優息", "月配息", "ETF"]},
    "00939.TW": {"name": "統一台灣高息動能", "keywords": ["00939", "高息動능", "ETF"]},
    "00940.TW": {"name": "元大臺灣價值高息", "keywords": ["00940", "臺灣價值高息", "ETF"]},
    "1101.TW": {"name": "台泥", "keywords": ["台泥", "1101"]},
    "1216.TW": {"name": "統一", "keywords": ["統一", "1216", "食品股", "集團股"]},
    "1301.TW": {"name": "台塑", "keywords": ["台塑", "1301", "塑化股"]},
    "1303.TW": {"name": "南亞", "keywords": ["南亞", "1303", "台塑集團"]},
    "1504.TW": {"name": "東元", "keywords": ["東元", "1504", "電機", "重電"]},
    "1710.TW": {"name": "東聯", "keywords": ["東聯", "1710", "塑化", "遠東集團"]},
    "2002.TW": {"name": "中鋼", "keywords": ["中鋼", "2002", "鋼鐵"]},
    "2201.TW": {"name": "裕隆", "keywords": ["裕隆", "2201", "汽車", "電動車"]},
    "2301.TW": {"name": "光寶科", "keywords": ["光寶科", "2301", "電源供應器", "光電"]},
    "2303.TW": {"name": "聯電", "keywords": ["聯電", "2303", "UMC", "晶圓", "半導體"]},
    "2308.TW": {"name": "台達電", "keywords": ["台達電", "2308", "Delta"]},
    "2317.TW": {"name": "鴻海", "keywords": ["鴻海", "2317", "Foxconn"]},
    "2327.TW": {"name": "國巨", "keywords": ["國巨", "2327", "被動元件"]},
    "2330.TW": {"name": "台積電", "keywords": ["台積電", "2330", "TSMC", "晶圓", "半導體"]},
    "2344.TW": {"name": "華邦電", "keywords": ["華邦電", "2344", "DRAM", "Flash", "記憶體"]},
    "2345.TW": {"name": "智邦", "keywords": ["智邦", "2345", "網通設備", "交換器"]},
    "2353.TW": {"name": "宏碁", "keywords": ["宏碁", "2353", "Acer", "PC"]},
    "2357.TW": {"name": "華碩", "keywords": ["華碩", "2357"]},
    "2379.TW": {"name": "瑞昱", "keywords": ["瑞昱", "2379", "RTL"]},
    "2382.TW": {"name": "廣達", "keywords": ["廣達", "2382", "AI伺服器"]},
    "2408.TW": {"name": "南亞科", "keywords": ["南亞科", "2408", "DRAM"]},
    "2409.TW": {"name": "友達", "keywords": ["友達", "2409", "面板股", "顯示器"]},
    "2454.TW": {"name": "聯發科", "keywords": ["聯發科", "2454", "MediaTek"]},
    "2455.TW": {"name": "全新", "keywords": ["全新", "2455", "砷化鎵", "PA"]},
    "2474.TW": {"name": "可成", "keywords": ["可成", "2474", "金屬機殼"]},
    "2498.TW": {"name": "宏達電", "keywords": ["宏達電", "2498", "HTC", "VR", "元宇宙"]},
    "2603.TW": {"name": "長榮", "keywords": ["長榮", "2603", "航運"]},
    "2609.TW": {"name": "陽明", "keywords": ["陽明", "2609", "航運"]},
    "2615.TW": {"name": "萬海", "keywords": ["萬海", "2615", "航運"]},
    "2834.TW": {"name": "臺企銀", "keywords": ["臺企銀", "2834", "金融股", "公股"]},
    "2880.TW": {"name": "華南金", "keywords": ["華南金", "2880", "金融股"]},
    "2881.TW": {"name": "富邦金", "keywords": ["富邦金", "2881", "金融股"]},
    "2882.TW": {"name": "國泰金", "keywords": ["國泰金", "2882", "金融股"]},
    "2884.TW": {"name": "玉山金", "keywords": ["玉山金", "2884", "金融股"]},
    "2886.TW": {"name": "兆豐金", "keywords": ["兆豐金", "2886", "金融股"]},
    "2890.TW": {"name": "永豐金", "keywords": ["永豐金", "2890", "金融股"]},
    "2891.TW": {"name": "中信金", "keywords": ["中信金", "2891", "金融股"]},
    "2892.TW": {"name": "第一金", "keywords": ["第一金", "2892", "金融股", "公股銀行"]},
    "3008.TW": {"name": "大立光", "keywords": ["大立光", "3008", "光學鏡頭"]},
    "3017.TW": {"name": "奇鋐", "keywords": ["奇鋐", "3017", "散熱"]},
    "3037.TW": {"name": "欣興", "keywords": ["欣興", "3037", "ABF載板", "PCB"]},
    "3231.TW": {"name": "緯創", "keywords": ["緯創", "3231", "AI伺服器"]},
    "3711.TW": {"name": "日月光投控", "keywords": ["日月光", "3711", "封裝測試", "半導體後段"]},
    "4938.TW": {"name": "和碩", "keywords": ["和碩", "4938", "代工", "電子組裝"]},
    "5880.TW": {"name": "合庫金", "keywords": ["合庫金", "5880", "金融股"]},
    "6239.TW": {"name": "力積電", "keywords": ["力積電", "6239", "DRAM", "晶圓代工"]},
    "6415.TW": {"name": "M31 (創意電子)", "keywords": ["M31", "創意電子", "6415", "IP"]},
    "6669.TW": {"name": "緯穎", "keywords": ["緯穎", "6669", "AI伺服器", "資料中心"]},
    "^TWII": {"name": "台股指數", "keywords": ["台股指數", "加權指數", "^TWII", "指數"]},

    # ----------------------------------------------------
    # 加密貨幣 (Cryptocurrency) - 排序依據: 代碼英文 (Key)
    # ----------------------------------------------------
    "AAVE-USD": {"name": "Aave", "keywords": ["Aave", "AAVE", "DeFi", "借貸協議"]},
    "ADA-USD": {"name": "Cardano", "keywords": ["Cardano", "ADA", "ADA-USDT"]},
    "ALGO-USD": {"name": "Algorand", "keywords": ["Algorand", "ALGO", "公鏈"]},
    "APT-USD": {"name": "Aptos", "keywords": ["Aptos", "APT", "Layer1", "公鏈"]},
    "ARB-USD": {"name": "Arbitrum", "keywords": ["Arbitrum", "ARB", "Layer2", "擴容"]},
    "ATOM-USD": {"name": "Cosmos", "keywords": ["Cosmos", "ATOM", "跨鏈"]},
    "AVAX-USD": {"name": "Avalanche", "keywords": ["Avalanche", "AVAX", "AVAX-USDT"]},
    "AXS-USD": {"name": "Axie Infinity", "keywords": ["Axie", "AXS", "GameFi", "遊戲"]},
    "BCH-USD": {"name": "比特幣現金 (Bitcoin Cash)", "keywords": ["比特幣現金", "BCH"]},
    "BNB-USD": {"name": "幣安幣 (Binance Coin)", "keywords": ["幣安幣", "BNB", "BNB-USDT", "交易所幣"]},
    "BTC-USD": {"name": "比特幣 (Bitcoin)", "keywords": ["比特幣", "BTC", "bitcoin", "BTC-USDT", "加密貨幣之王"]},
    "DAI-USD": {"name": "Dai", "keywords": ["Dai", "DAI", "穩定幣", "MakerDAO"]},
    "DOGE-USD": {"name": "狗狗幣 (Dogecoin)", "keywords": ["狗狗幣", "DOGE", "DOGE-USDT", "迷因幣"]},
    "DOT-USD": {"name": "Polkadot", "keywords": ["Polkadot", "DOT", "DOT-USDT"]},
    "ETC-USD": {"name": "以太坊經典 (Ethereum Classic)", "keywords": ["以太坊經典", "ETC", "EthereumClassic"]},
    "ETH-USD": {"name": "以太坊 (Ethereum)", "keywords": ["以太坊", "ETH", "ethereum", "ETH-USDT"]},
    "FIL-USD": {"name": "Filecoin", "keywords": ["Filecoin", "FIL", "分散式儲存"]},
    "FTM-USD": {"name": "Fantom", "keywords": ["Fantom", "FTM", "公鏈"]},
    "ICP-USD": {"name": "Internet Computer", "keywords": ["InternetComputer", "ICP", "Dfinity"]},
    "LTC-USD": {"name": "萊特幣 (Litecoin)", "keywords": ["萊特幣", "LTC", "Litecoin"]},
    "MANA-USD": {"name": "Decentraland", "keywords": ["Decentraland", "MANA", "元宇宙"]},
    "MATIC-USD": {"name": "Polygon", "keywords": ["Polygon", "MATIC", "Layer2"]},
    "OP-USD": {"name": "Optimism", "keywords": ["Optimism", "OP", "Layer2"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL", "SOL-USDT"]},
    "TRX-USD": {"name": "Tron", "keywords": ["Tron", "TRX"]},
    "UNI-USD": {"name": "Uniswap", "keywords": ["Uniswap", "UNI", "DEX", "去中心化交易所"]},
    "USDC-USD": {"name": "USD Coin", "keywords": ["USDC", "穩定幣"]},
    "USDT-USD": {"name": "Tether", "keywords": ["USDT", "穩定幣"]},
    "XLM-USD": {"name": "Stellar", "keywords": ["Stellar", "XLM"]},
    "XRP-USD": {"name": "Ripple", "keywords": ["Ripple", "XRP"]},
}


# ==============================================================================
# 2. 標的搜尋與數據獲取函數
# ==============================================================================

def search_symbol(query):
    """根據輸入的代碼或名稱關鍵字，模糊搜尋匹配的標的"""
    query = query.lower().strip()
    if not query:
        return []

    # 1. 精確代碼匹配
    if query.upper() in FULL_SYMBOLS_MAP:
        return [{"code": query.upper(), "name": FULL_SYMBOLS_MAP[query.upper()]['name']}]

    results = []
    # 2. 模糊匹配 (代碼或名稱)
    for code, info in FULL_SYMBOLS_MAP.items():
        if query in code.lower() or query in info['name'].lower() or any(query in kw.lower() for kw in info['keywords']):
            results.append({"code": code, "name": info['name']})

    # 限制最多返回 10 個結果
    return results[:10]

@st.cache_data(ttl=3600)
def download_data(symbol, period, interval):
    """從yfinance下載數據"""
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        # 移除任何具有 NaN 值的列
        data.dropna(inplace=True)
        return data
    except Exception:
        return None

# ==============================================================================
# 3. 技術分析指標計算
# ==============================================================================

def calculate_technical_indicators(data):
    """計算技術指標：EMA, MACD, RSI, ATR, KDJ, ADX"""
    df = data.copy()

    # 1. 指數移動平均線 (EMA)
    df['EMA_10'] = ta.trend.EMAIndicator(df['Close'], window=10, fillna=True).ema_indicator()
    df['EMA_30'] = ta.trend.EMAIndicator(df['Close'], window=30, fillna=True).ema_indicator()
    df['EMA_60'] = ta.trend.EMAIndicator(df['Close'], window=60, fillna=True).ema_indicator()

    # 2. MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9, fillna=True)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # 3. RSI (Relative Strength Index)
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14, fillna=True).rsi()

    # 4. ATR (Average True Range) - 用於波動性/風險評估
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14, fillna=True).average_true_range()

    # 5. KDJ (Stochastic Oscillator)
    kdj = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=9, smooth_window=3, fillna=True)
    df['K_9_3'] = kdj.stoch()
    df['D_9_3'] = kdj.stoch_signal()

    # 6. ADX (Average Directional Index) - 用於判斷趨勢強度
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14, fillna=True).adx()

    # 僅保留需要的列，並移除任何可能因計算而產生的 NaN
    df.dropna(inplace=True)

    # 僅返回最後 200 筆數據用於繪圖和分析，以優化效能
    return df.iloc[-200:].copy()

# ==============================================================================
# 4. 圖表繪製
# ==============================================================================

def plot_analysis_chart(df, symbol_name, period_interval):
    """繪製包含K線圖、MACD和RSI的組合圖表"""
    
    # 建立三個子圖：K線、MACD、RSI
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2])

    # --- 第一個子圖：K線圖與均線 ---
    
    # K線圖 (Candlestick)
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='K線',
                                 increasing_line_color='#FF4B4B', decreasing_line_color='#26C281'),
                  row=1, col=1)

    # EMA 均線
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_10'], line=dict(color='orange', width=1), name='EMA 10'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], line=dict(color='blue', width=1), name='EMA 30'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_60'], line=dict(color='purple', width=1), name='EMA 60'), row=1, col=1)

    # 設置標題和佈局
    fig.update_layout(
        title_text=f"**{symbol_name} ({df.index.max().strftime('%Y-%m-%d %H:%M')}) - {period_interval} K線圖**",
        xaxis_rangeslider_visible=False, # 隱藏底部的時間軸滑桿
        height=700,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- 第二個子圖：MACD ---
    
    # MACD 柱體
    colors = ['#FF4B4B' if val >= 0 else '#26C281' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD 柱體', marker_color=colors), row=2, col=1)
    
    # MACD 線
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#0000FF', width=1), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#FFA500', width=1), name='Signal'), row=2, col=1)
    
    fig.update_yaxes(title_text="MACD", row=2, col=1, fixedrange=True)
    fig.update_layout(hovermode="x unified")


    # --- 第三個子圖：RSI ---
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='magenta', width=1), name='RSI'), row=3, col=1)
    
    # 超買/超賣水平線
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100], fixedrange=True)
    
    return fig

# ==============================================================================
# 5. 消息面、籌碼面與基本面分析 (新增與重大修改)
# ==============================================================================

@st.cache_data(ttl=3600)
def get_chips_and_news_analysis(symbol):
    """獲取籌碼面和消息面數據 (新增)"""
    try:
        ticker = yf.Ticker(symbol)
        
        # 籌碼面分析
        # yfinance 提供的持股數據 (Major Holders, Institutional Holders)
        major_holders = ticker.major_holders
        institutional_holders = ticker.institutional_holders
        
        chips_summary = "無法獲取籌碼數據 (可能為指數/ETF或美股非核心數據)"
        if major_holders is not None and not major_holders.empty:
            # 確保不是全部為 NaN 或 None
            if major_holders.shape[0] > 0 and major_holders.iloc[0, 0]:
                insider_hold = major_holders.iloc[0, 0]
                chips_summary = f"內部大股東持股: **{insider_hold}**"
        
        if institutional_holders is not None and not institutional_holders.empty:
            if institutional_holders.shape[0] > 0 and institutional_holders.iloc[0, 0]:
                inst_hold = institutional_holders.iloc[0, 0]
                # 若 chips_summary 仍是預設值，則直接賦值；否則追加
                if "無法獲取" in chips_summary:
                     chips_summary = f"法人機構持股: **{inst_hold}**"
                else:
                     chips_summary += f"，法人機構持股: **{inst_hold}**"
        
        # 排除預設值但標的是個股的情況，避免誤判
        if "無法獲取" in chips_summary and ticker.info.get('quoteType') in ['EQUITY']:
            chips_summary = "數據獲取失敗或yfinance無相關數據。"


        # 消息面分析
        news = ticker.news
        news_summary = "近期無相關新聞"
        if news:
            # 取前三條
            headlines = [f"- {item['title']}" for item in news[:3]] 
            news_summary = "\n".join(headlines)

        return {
            "chips_summary": chips_summary,
            "news_summary": news_summary
        }
    except Exception:
        return {
            "chips_summary": "無法獲取籌碼數據",
            "news_summary": "無法獲取新聞數據"
        }

@st.cache_data(ttl=3600)
def calculate_fundamental_rating(symbol): 
    """根據進階原則計算基本面評分 (最高7分) - 替換舊版"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # 對非個股標的跳過基本面分析
        # 檢查 quoteType 排除指數/ETF/加密貨幣
        is_stock = info.get('quoteType') in ['EQUITY']
        
        # 檢查關鍵財務數據是否存在
        has_key_data = info.get('returnOnEquity') is not None
        
        if not is_stock or not has_key_data:
            return {"score": 0, "summary": "非個股標的或無法獲取關鍵財務數據，不適用基本面分析。", "details": {}}

        score = 0
        details = {}
        
        # 獲利能力 (ROE) - 權重高 (2分)
        roe = info.get('returnOnEquity')
        if roe and roe > 0.15:
            score += 2
            details['✅ 股東權益報酬率(ROE) > 15%'] = f"{roe:.2%}"
        else:
            details['❌ 股東權益報酬率(ROE) < 15%'] = f"{roe:.2%}" if roe is not None else "N/A"

        # 財務健康 (Debt to Equity) - 權重高 (2分)
        debt_to_equity = info.get('debtToEquity')
        # yfinance 的 D/E 是比率 * 100，檢查 D/E < 0.5 (即 debtToEquity < 50)
        if debt_to_equity is not None and debt_to_equity < 50: 
            score += 2
            details['✅ 負債權益比 < 50%'] = f"{debt_to_equity/100:.2f}"
        else:
            details['❌ 負債權益比 > 50%'] = f"{debt_to_equity/100:.2f}" if debt_to_equity is not None else "N/A"
            
        # 成長性 (Revenue Growth) - (1分)
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth is not None and revenue_growth > 0.1:
            score += 1
            details['✅ 營收年增 > 10%'] = f"{revenue_growth:.2%}"
        else:
            details['❌ 營收年增 < 10%'] = f"{revenue_growth:.2%}" if revenue_growth is not None else "N/A"

        # 估值 (PE) - (1分)
        pe = info.get('trailingPE')
        if pe is not None and pe > 0 and pe < 15:
            score += 1
            details['✅ 本益比(P/E) < 15'] = f"{pe:.2f}"
        elif pe is not None and pe > 0:
            details['⚠️ 本益比(P/E) > 15'] = f"{pe:.2f}"
        else:
            details['⚠️ 本益比(P/E)'] = "N/A"
        
        # 估值 (PEG) - (1分)
        peg = info.get('pegRatio')
        if peg is not None and peg > 0 and peg < 1.0:
            score += 1
            details['✅ PEG < 1'] = f"{peg:.2f}"
        elif peg is not None and peg > 0:
            details['⚠️ PEG > 1'] = f"{peg:.2f}"
        else:
            details['⚠️ PEG'] = "N/A"
        
        # 綜合評語
        if score >= 6:
            summary = "優秀：公司在獲利能力、財務健康和成長性上表現強勁，且估值合理。"
        elif score >= 4:
            summary = "良好：公司基本面穩健，但在某些方面（如估值或成長性）有待加強。"
        else:
            summary = "中性/警示：需留意公司的財務風險、獲利能力不足或估值偏高的問題。"

        return {"score": score, "summary": summary, "details": details}

    except Exception:
        return {"score": 0, "summary": "無法獲取或計算基本面數據。", "details": {}}


# ==============================================================================
# 6. AI融合訊號產生 (修改函式簽名與邏輯)
# ==============================================================================

def generate_expert_fusion_signal(data, fa_rating, chips_news_analysis, symbol_info, period_interval):
    """
    結合技術面、基本面、籌碼面、消息面數據，產生AI融合交易訊號。
    回傳: (final_score, signal, confidence, analysis_comment, chips_news_comment, all_signals)
    *** 簽名修改：新增 chips_news_analysis 參數 ***
    """
    # ----------------------------------------------------
    # 1. 技術面分數計算 (Technical Score, 權重最高) - 範圍約 -5 到 5
    # ----------------------------------------------------
    tech_score = 0
    signal_list = []

    # Moving Averages (EMA 10/30/60) - 趨勢方向
    ema10 = data['EMA_10'].iloc[-1]
    ema30 = data['EMA_30'].iloc[-1]
    ema60 = data['EMA_60'].iloc[-1]
    last_close = data['Close'].iloc[-1]

    # 多頭排列：10 > 30 > 60
    if ema10 > ema30 and ema30 > ema60 and last_close > ema10:
        tech_score += 2.5
        signal_list.append("✅ **趨勢：** 多頭排列，股價站上所有均線，趨勢強勁。")
    # 空頭排列：10 < 30 < 60
    elif ema10 < ema30 and ema30 < ema60 and last_close < ema10:
        tech_score -= 2.5
        signal_list.append("❌ **趨勢：** 空頭排列，股價跌破所有均線，趨勢轉弱。")
    else:
        signal_list.append("⚠️ **趨勢：** 均線糾結或混沌，市場方向不明。")
        
    # RSI (Relative Strength Index) - 超買/超賣
    rsi = data['RSI_14'].iloc[-1]
    if rsi >= 70:
        tech_score -= 1.0
        signal_list.append("⚠️ **動能：** RSI > 70，進入超買區，短線修正壓力增高。")
    elif rsi <= 30:
        tech_score += 1.0
        signal_list.append("✅ **動能：** RSI < 30，進入超賣區，具備潛在反彈動能。")
    else:
        # 接近50是中性，RSI越高，分數越高 (微調，正負 0.5 分影響)
        tech_score += (rsi - 50) / 40 

    # MACD - 動能轉折
    macd_hist = data['MACD_Hist'].iloc[-1]
    macd_prev_hist = data['MACD_Hist'].iloc[-2] if len(data) >= 2 else 0

    if macd_hist > 0 and macd_hist > macd_prev_hist:
        tech_score += 1.0
        signal_list.append("✅ **動能：** MACD柱體為正且持續增長，多頭動能強。")
    elif macd_hist < 0 and macd_hist < macd_prev_hist:
        tech_score -= 1.0
        signal_list.append("❌ **動能：** MACD柱體為負且持續增長，空頭動能強。")
    elif macd_hist > macd_prev_hist:
        tech_score += 0.5
    elif macd_hist < macd_prev_hist:
        tech_score -= 0.5
    
    # KDJ/KD - 短期超買超賣
    k_val = data['K_9_3'].iloc[-1]
    d_val = data['D_9_3'].iloc[-1]
    
    if k_val < 20 and d_val < 20 and k_val > d_val:
        tech_score += 0.5
        signal_list.append("✅ **短線：** KD處於低檔黃金交叉，短線反彈機率高。")
    elif k_val > 80 and d_val > 80 and k_val < d_val:
        tech_score -= 0.5
        signal_list.append("❌ **短線：** KD處於高檔死亡交叉，短線修正機率高。")
        
    # ----------------------------------------------------
    # 2. 融合基本面、籌碼面與消息面分數 (加權機制)
    # ----------------------------------------------------
    
    # a. 基本面加權 (Max 7分 -> 權重加分 Max 3分)
    fa_score = fa_rating['score']
    fa_scaled_score = 0
    
    # 只有分數 > 0 且適用基本面分析時才加權
    if fa_score > 0:
        fa_scaled_score = min(fa_score / 7 * 3.0, 3.0) # 最大加權 3 分
        tech_score += fa_scaled_score
        
    # b. 籌碼面/消息面加權 (Max 1分)
    chips_news_boost = 0
    chips_news_comment = ""
    
    chips_summary = chips_news_analysis['chips_summary']
    news_summary = chips_news_analysis['news_summary']

    # 籌碼面：若有機構/大股東持股資訊，略為正面 (非預設值即視為有數據)
    if not ("無法獲取" in chips_summary or "數據獲取失敗" in chips_summary):
        chips_news_boost += 0.5
        chips_news_comment += "籌碼面有基本持股數據支持，略為正面。 "
    
    # 消息面：若有新聞，則提示留意
    if "近期無相關新聞" not in news_summary and news_summary:
        chips_news_boost += 0.5
        # 只取第一行新聞標題作為提示
        first_headline = news_summary.splitlines()[0].replace('-', '').strip()
        chips_news_comment += f"⚠️ 留意到近期消息面活動：{first_headline}..."
        
    final_score = tech_score + chips_news_boost

    # ----------------------------------------------------
    # 3. 判斷最終訊號與信心度
    # ----------------------------------------------------

    # 調整分析週期對信心的影響：週期越長，信號越可靠
    period_boost = 0
    if "週" in period_interval:
        period_boost = 0.5 
    elif "日" in period_interval:
        period_boost = 0.2
        
    # 最終訊號判斷
    if final_score >= 5.0:
        signal = "極度看漲 (Strong Buy)"
        confidence = min(100, 85 + period_boost * 10)
    elif final_score >= 3.0:
        signal = "看漲 (Buy)"
        confidence = min(100, 70 + period_boost * 10)
    elif final_score <= -5.0:
        signal = "極度看跌 (Strong Sell)"
        confidence = min(100, 85 + period_boost * 10)
    elif final_score <= -3.0:
        signal = "看跌 (Sell)"
        confidence = min(100, 70 + period_boost * 10)
    else:
        signal = "中性觀望 (Hold)"
        confidence = 50 + period_boost * 10
        
    # 最終總結語
    base_tech_score = tech_score - fa_scaled_score - chips_news_boost
    analysis_comment = f"技術面總分 **{base_tech_score:.2f}**，基本面加權 **+{fa_scaled_score:.2f}**，籌碼/消息加權 **+{chips_news_boost:.2f}**，最終融合總分 **{final_score:.2f}**。"
    
    # 合併所有的判斷依據
    all_signals = "\n".join(signal_list)
    
    return final_score, signal, confidence, analysis_comment, chips_news_comment, all_signals

# ==============================================================================
# 7. 主程式 (main) 與 Streamlit 介面 (修改呼叫與介面呈現)
# ==============================================================================

def main():
    # Streamlit Sidebar
    st.sidebar.title("🔍 標的選擇與參數")
    
    # 判斷標的類別
    asset_class_options = ["台股", "美股/ETF", "加密貨幣/指數"]
    asset_class = st.sidebar.selectbox("1. 選擇資產類別", asset_class_options, index=1)
    
    # 篩選符號列表
    if asset_class == "台股":
        filtered_symbols = {k: v for k, v in FULL_SYMBOLS_MAP.items() if k.endswith('.TW') or re.match(r'^\d{4}\.TW$', k)}
        default_symbol_code = "2330.TW"
    elif asset_class == "美股/ETF":
        # 排除台股和加密貨幣/指數
        filtered_symbols = {k: v for k, v in FULL_SYMBOLS_MAP.items() if not k.endswith('.TW') and not re.search(r'-USD|\^', k)}
        default_symbol_code = "NVDA"
    else: # 加密貨幣/指數
        filtered_symbols = {k: v for k, v in FULL_SYMBOLS_MAP.items() if re.search(r'-USD|\^', k)}
        default_symbol_code = "BTC-USD"

    # 預設選項轉換為 "名稱 (代碼)" 格式
    sorted_options = sorted([f"{v['name']} ({k})" for k, v in filtered_symbols.items()])
    
    # 找到預設代碼在列表中的位置
    default_index = next((i for i, opt in enumerate(sorted_options) if default_symbol_code in opt), 0)

    # 標的選擇
    st.sidebar.markdown("---")
    st.session_state['sidebar_search_input'] = st.sidebar.text_input(
        "2. 搜尋代碼或名稱 (e.g. 2330, TSLA)",
        key='symbol_search_input'
    )
    
    # 動態建議
    search_results = search_symbol(st.session_state['sidebar_search_input'])
    
    # 下拉選單用於選擇
    options_with_default = [f"請選擇或輸入標的 ({len(filtered_symbols)}個熱門資產)",] + sorted_options
    selected_option = st.sidebar.selectbox(
        "或從熱門資產快速選擇",
        options_with_default,
        index=default_index + 1 if default_index >= 0 else 0, # +1 避開預設提示
        key='symbol_selectbox'
    )
    
    # 確定最終選擇的代碼
    selected_symbol = ""
    if selected_option and "(" in selected_option and ")" in selected_option:
        # 從 "名稱 (代碼)" 中解析出代碼
        selected_symbol = selected_option.split('(')[-1].replace(')', '').strip()
    elif st.session_state['sidebar_search_input']:
        # 使用者直接輸入代碼/名稱
        if search_results:
            selected_symbol = search_results[0]['code']
        else:
            selected_symbol = st.session_state['sidebar_search_input'].upper().strip()
    
    # 週期選擇
    st.sidebar.markdown("---")
    period_interval_map = {
        "30 分": "短期趨勢 (60天, 30分鐘K)",
        "4 小時": "波段趨勢 (1年, 4小時K)",
        "1 日": "中長線趨勢 (5年, 1日K)",
        "1 週": "長期趨勢 (最大, 1週K)"
    }
    st.session_state['selected_period'] = st.sidebar.radio(
        "3. 選擇分析週期", 
        list(PERIOD_MAP.keys()),
        index=2, # 預設為 1 日 (中長線)
        format_func=lambda x: f"{x} ({period_interval_map[x].split(' ')[0]})"
    )
    
    # 執行按鈕
    st.sidebar.markdown("---")
    
    # 確保當前分析的代碼存在
    if selected_symbol:
        st.session_state['last_search_symbol'] = selected_symbol
        symbol_info = FULL_SYMBOLS_MAP.get(selected_symbol, {"name": selected_symbol, "keywords": []})
        st.sidebar.info(f"當前選擇：**{symbol_info['name']} ({selected_symbol})**")
    else:
        st.sidebar.warning("請選擇或輸入有效的標的代碼。")


    if st.sidebar.button('📊 執行AI分析'):
        if not selected_symbol:
            st.error("請選擇或輸入有效的標的代碼才能執行分析。")
            st.session_state['data_ready'] = False
            return
            
        period, interval = PERIOD_MAP[st.session_state['selected_period']]
        symbol_info = FULL_SYMBOLS_MAP.get(selected_symbol, {"name": selected_symbol, "keywords": []})

        st.info(f"正在下載 {symbol_info['name']} ({selected_symbol})，週期：{st.session_state['selected_period']} 的數據...")
        
        with st.spinner('🚀 正在執行AI智能分析...'):
            # 1. 數據下載
            data = download_data(selected_symbol, period, interval)

            if data is None:
                st.error(f"無法下載 {selected_symbol} 的數據。請檢查代碼是否正確或yfinance是否有數據。")
                st.session_state['data_ready'] = False
                return

            # 2. 技術指標計算
            data = calculate_technical_indicators(data)
            
            # 3. 基本面、消息面、籌碼面分析 (新增呼叫)
            fa_rating = calculate_fundamental_rating(selected_symbol)
            chips_news_analysis = get_chips_and_news_analysis(selected_symbol) # *** 新增呼叫 ***

            # 4. 專家融合訊號
            try:
                # *** 傳入 chips_news_analysis 參數 ***
                final_score, signal, confidence, analysis_comment, chips_news_comment, all_signals = generate_expert_fusion_signal(
                    data, 
                    fa_rating, 
                    chips_news_analysis, # *** 新增參數 ***
                    symbol_info, 
                    st.session_state['selected_period']
                )

                # 儲存結果
                st.session_state['analysis_data'] = {
                    "symbol_name": symbol_info['name'],
                    "symbol_code": selected_symbol,
                    "data_for_plot": data.copy(),
                    "fa_rating": fa_rating,
                    "chips_news_analysis": chips_news_analysis, # *** 新增儲存 ***
                    "final_score": final_score,
                    "signal": signal,
                    "confidence": confidence,
                    "analysis_comment": analysis_comment,
                    "chips_news_comment": chips_news_comment, # *** 新增儲存 ***
                    "all_signals": all_signals,
                    "period": st.session_state['selected_period']
                }
                st.session_state['data_ready'] = True
                st.success("分析完成！請查看下方的AI專家報告。")

            except Exception as e:
                st.error(f"分析失敗：無法生成融合訊號。錯誤：{e}")
                st.session_state['data_ready'] = False
            
    # --- 結果呈現 ---
    if st.session_state['data_ready']:
        analysis_data = st.session_state['analysis_data']
        symbol_name = analysis_data['symbol_name']
        symbol_code = analysis_data['symbol_code']
        period_str = analysis_data['period']
        
        st.title(f"📈 {symbol_name} ({symbol_code}) - {period_str} 趨勢分析報告")
        st.markdown("---")
        
        # 繪圖
        fig = plot_analysis_chart(analysis_data['data_for_plot'], symbol_name, period_str)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")

        # 🔥 技術面判斷結果 (Technical Analysis Summary)
        st.subheader(f"🤖 AI 技術面判讀 ({period_str})")
        st.markdown(analysis_data['all_signals'])
        
        st.markdown("---")

        # ----------------------------------------------------
        # 🔥 進階基本面評分 (Fundamental Rating) - 修改呈現方式
        # ----------------------------------------------------
        fa_data = analysis_data['fa_rating']
        st.subheader("💡 進階基本面評分 (7項檢查)")
        
        if fa_data['score'] == 0 and "非個股標的" in fa_data['summary']:
            st.warning(f"當前標的為指數/ETF/加密貨幣，不適用基本面分析。")
        else:
            # 顏色強調分數
            color = "#4CAF50" if fa_data['score'] >= 4 else ("#FFA500" if fa_data['score'] >= 2 else "#FF4B4B")
            st.markdown(f"**綜合評分：** <span style='font-size: 24px; color: {color}; font-weight: bold;'>{fa_data['score']}/7 分</span>", unsafe_allow_html=True)
            st.markdown(f"**評語：** {fa_data['summary']}")
            
            # 呈現詳細檢查項目
            with st.expander("📝 點擊查看詳細檢查項目"):
                fa_details_list = [f"* {k}: **{v}**" for k, v in fa_data['details'].items()]
                st.markdown("\n".join(fa_details_list))

        st.markdown("---")
        
        # ----------------------------------------------------
        # 🔥 籌碼面與消息面分析 (Chips & News Analysis) - 新增區塊
        # ----------------------------------------------------
        chips_news_data = analysis_data['chips_news_analysis']
        st.subheader("📰 籌碼面與消息面分析 (Chips & News)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**籌碼面 (Chips)**：\n> {chips_news_data['chips_summary']}")
        with col2:
            st.markdown(f"**消息面 (News)**：\n> {chips_news_data['news_summary']}")
        
        st.markdown("---")
        
        # ----------------------------------------------------
        # 🔥 最終AI專家融合訊號 (Fusion Signal)
        # ----------------------------------------------------
        
        signal_color = ""
        if "看漲" in analysis_data['signal'] or "Buy" in analysis_data['signal']:
            signal_color = "#FF4B4B"  # 紅色代表多頭
        elif "看跌" in analysis_data['signal'] or "Sell" in analysis_data['signal']:
            signal_color = "#26C281"  # 綠色代表空頭
        else:
            signal_color = "#808080"
            
        st.subheader("🚀 最終AI專家融合訊號 (Fusion Signal)")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"<p style='font-size: 30px; color: {signal_color}; font-weight: bold;'>{analysis_data['signal']}</p>", unsafe_allow_html=True)
        with col2:
            st.metric(label="預測信心度", value=f"{analysis_data['confidence']:.1f}%")

        # 融合總結
        st.markdown(f"**融合專家總結：** {analysis_data['analysis_comment']}")
        
        # 新增籌碼/消息影響提示
        if analysis_data['chips_news_comment']:
             st.info(f"**額外考量：** {analysis_data['chips_news_comment']}")
             
    else:
        # 歡迎畫面
        st.title(f"<h1 style='text-align: center; color: #FA8072;'>🚀 歡迎使用 AI 趨勢分析</h1>", unsafe_allow_html=True)
        st.markdown(f"請在左側選擇或輸入您想分析的標的（例如：**2330.TW**、**NVDA**、**BTC-USD**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇標的**：使用下拉選單快速選擇熱門標的，或直接在輸入框中鍵入代碼或名稱。")
        st.markdown("3. **選擇週期**：決定分析的長度（例如：`30 分`、`4 小時`、`1 日`、`1 周`）。")
        st.markdown(f"4. **執行分析**：點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』**</span>，AI將融合基本面與技術面指標提供交易策略。", unsafe_allow_html=True)
        
        st.markdown("---")


if __name__ == '__main__':
    # Streamlit Session State 初始化，確保變數存在
    if 'last_search_symbol' not in st.session_state:
        st.session_state['last_search_symbol'] = "2330.TW"
    if 'data_ready' not in st.session_state:
        st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state:
        st.session_state['sidebar_search_input'] = ""
    if 'selected_period' not in st.session_state:
        st.session_state['selected_period'] = "1 日"
        
    main()
