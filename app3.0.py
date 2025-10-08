# app_final_pro_v3.py (基於 app3.0.py 和 v7.0 文件優化)

import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta
import yfinance as yf
from plotly.subplots import make_subplots

# 警告過濾 (保持與原文件一致)
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 頁面配置與全局設定 (保持與原文件一致)
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

# 🚀 您的【所有資產清單】(擴充版 - 保持與原文件一致)
FULL_SYMBOLS_MAP = {
    # 美股/ETF/指數
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
    "^IXIC": {"name": "NASDAQ 綜合指數", "keywords": ["納斯達크", "NASDAQ", "^IXIC", "指數", "科技股"]},
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
    "QQQM": {"name": "Invesco NASDAQ 100 ETF (低費率)", "keywords": ["QQQM", "納斯達크ETF", "科技股ETF"]},
    "QQQ": {"name": "Invesco QQQ Trust", "keywords": ["QQQ", "納斯達크ETF", "科技股ETF"]},
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
    # 台股/ETF/指數
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
    "00939.TW": {"name": "統一台灣高息動能", "keywords": ["00939", "高息動能", "ETF"]},
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
    "6415.TW": {"name": "創意", "keywords": ["M31", "創意電子", "6415", "IP"]},
    "6669.TW": {"name": "緯穎", "keywords": ["緯穎", "6669", "AI伺服器", "資料中心"]},
    "^TWII": {"name": "台股指數", "keywords": ["台股指數", "加權指數", "^TWII", "指數"]},
    # 加密貨幣
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
    "ETH-USD": {"name": "以太坊 (Ethereum)", "keywords": ["以太坊", "ETH", "ethereum", "ETH-USDT", "智能合約"]},
    "FIL-USD": {"name": "Filecoin", "keywords": ["Filecoin", "FIL", "去中心化儲存"]},
    "FTM-USD": {"name": "Fantom", "keywords": ["Fantom", "FTM", "公鏈"]},
    "HBAR-USD": {"name": "Hedera", "keywords": ["Hedera", "HBAR", "分散式帳本"]},
    "ICP-USD": {"name": "Internet Computer", "keywords": ["ICP", "網際網路電腦"]},
    "IMX-USD": {"name": "ImmutableX", "keywords": ["ImmutableX", "IMX", "GameFi", "NFT L2"]},
    "INJ-USD": {"name": "Injective Protocol", "keywords": ["Injective", "INJ", "DeFi", "去中心化交易"]},
    "LDO-USD": {"name": "Lido DAO", "keywords": ["Lido", "LDO", "ETH質押", "DeFi"]},
    "LINK-USD": {"name": "Chainlink", "keywords": ["Chainlink", "LINK", "LINK-USDT", "預言機"]},
    "LTC-USD": {"name": "萊特幣 (Litecoin)", "keywords": ["萊特幣", "LTC", "數位白銀"]},
    "LUNA1-USD": {"name": "Terra 2.0 (LUNA)", "keywords": ["LUNA", "Terra 2.0"]},
    "MANA-USD": {"name": "Decentraland", "keywords": ["Decentraland", "MANA", "元宇宙", "虛擬土地"]},
    "MATIC-USD": {"name": "Polygon", "keywords": ["Polygon", "MATIC", "Layer2", "側鏈"]},
    "MKR-USD": {"name": "Maker", "keywords": ["Maker", "MKR", "DAI發行", "DeFi"]},
    "NEAR-USD": {"name": "Near Protocol", "keywords": ["Near", "NEAR", "公鏈"]},
    "OP-USD": {"name": "Optimism", "keywords": ["Optimism", "OP", "Layer2", "擴容"]},
    "SAND-USD": {"name": "The Sandbox", "keywords": ["TheSandbox", "SAND", "元宇宙", "NFT"]},
    "SHIB-USD": {"name": "柴犬幣 (Shiba Inu)", "keywords": ["柴犬幣", "SHIB", "迷因幣", "Shiba"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL", "SOL-USDT"]},
    "SUI-USD": {"name": "Sui", "keywords": ["Sui", "SUI", "Layer1", "公鏈"]},
    "TIA-USD": {"name": "Celestia", "keywords": ["Celestia", "TIA", "模組化區塊鏈"]},
    "TRX-USD": {"name": "Tron", "keywords": ["波場", "TRX", "Tron"]},
    "UNI-USD": {"name": "Uniswap", "keywords": ["Uniswap", "UNI", "去中心化交易所", "DEX"]},
    "USDC-USD": {"name": "USD Coin", "keywords": ["USDC", "穩定幣", "美元幣"]},
    "USDT-USD": {"name": "泰達幣 (Tether)", "keywords": ["泰達幣", "USDT", "穩定幣", "Tether"]},
    "VET-USD": {"name": "VeChain", "keywords": ["VeChain", "VET", "供應鏈"]},
    "WLD-USD": {"name": "Worldcoin", "keywords": ["Worldcoin", "WLD", "AI", "身份驗證"]},
    "XMR-USD": {"name": "門羅幣 (Monero)", "keywords": ["門羅幣", "Monero", "XMR", "隱私幣"]},
    "XRP-USD": {"name": "瑞波幣 (Ripple)", "keywords": ["瑞波幣", "XRP", "XRP-USDT"]},
    "XTZ-USD": {"name": "Tezos", "keywords": ["Tezos", "XTZ", "公鏈"]},
    "ZEC-USD": {"name": "大零幣 (ZCash)", "keywords": ["大零幣", "ZCash", "ZEC", "隱私幣"]},
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
        # 修正：使用 auto_adjust=True 獲取調整後的價格
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df.empty: return pd.DataFrame()

        df.columns = [col.capitalize() for col in df.columns]
        df.index.name = 'Date'
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        df = df[~df.index.duplicated(keep='first')]

        # 移除最後一筆可能不完整的數據
        if len(df) > 1:
            df = df.iloc[:-1]

        if df.empty: return pd.DataFrame()
        return df
    except Exception as e:
        return pd.DataFrame()

# 保持其他數據輔助函數不變 (get_company_info, get_currency_symbol)
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
        return {"name": name, "category": category, "currency": currency}
    except Exception:
        return {"name": symbol, "category": "未分類", "currency": "USD"}

@st.cache_data(ttl=3600)
def get_currency_symbol(symbol):
    currency_code = get_company_info(symbol).get('currency', 'USD')
    if currency_code == 'TWD': return 'NT$'
    elif currency_code == 'USD': return '$'
    else: return currency_code + ' '

# 技術指標計算 (保持與原文件一致)def get_technical_data_df(df):
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
            avg_atr = df_clean['ATR'].iloc[-30:].mean()
            if value > avg_atr * 1.5: conclusion, color = "警告：極高波動性", "green"
            else: conclusion, color = "中性：正常波動", "blue"
        elif '布林通道' in name:
            if value > last_row['BB_High']: conclusion, color = "警告：價格位於上軌外側 (超買)", "red"
            elif value < last_row['BB_Low']: conclusion, color = "強化：價格位於下軌外側 (超賣)", "green"
            else: conclusion, color = "中性：在上下軌間", "blue"
        data.append([name, value, conclusion, color])
    # 注意: DataFrame 包含 '顏色' 欄位，這在後面樣式應用時會用到
    return pd.DataFrame(data, columns=['指標名稱', '最新值', '分析結論', '顏色']).set_index('指標名稱')

# ... (create_comprehensive_chart, run_backtest 函數保持不變)

# ==============================================================================
# 4. Streamlit 主應用程式邏輯 (修正樣式應用部分)
# ==============================================================================

def main():
    # ... (側邊欄 UI 保持不變)

    # --- 結果呈現區 ---
    if st.session_state.get('data_ready', False):
        res = st.session_state['analysis_results']
        df_clean = res['df'].dropna(subset=['Close', 'EMA_10', 'RSI', 'MACD_Hist'])
        if df_clean.empty:
            st.error("❌ **數據處理失敗：** 核心技術指標計算結果為空。請嘗試更換週期或標的。")
            st.session_state['data_ready'] = False
            return

        # ... (AI 融合訊號、核心行動、交易策略、AI判讀細節等部分保持不變)
        
        st.subheader("🛠️ 技術指標狀態表")
        technical_df = get_technical_data_df(df_clean)
        if not technical_df.empty:
            
            # === FIX: 修正 Pandas Styler ValueError, 改用 row-wise 應用 (axis=1) ===
            def style_row(row):
                """根據 '顏色' 欄位的值，應用 CSS 樣式到 '分析結論' 欄位。"""
                color_map = {'red': '#cc0000', 'green': '#1e8449', 'orange': '#cc6600', 'blue': '#888888'}
                # 獲取顏色代碼
                color_code = color_map.get(row['顏色'], '')
                
                # 完整
