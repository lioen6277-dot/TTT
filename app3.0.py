# app_v4.0.py

import re
import warnings
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

# 🚀 您的【所有資產清單】
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
    "SMH": {"name": "Van-eck Vectors半導體ETF", "keywords": ["SMH", "半導體ETF", "晶片股"]},
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

# 建立第二層選擇器映射
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
# 2. 核心數據與分析函式
# ==============================================================================

def sync_text_input_from_selection():
    try:
        selected_category = st.session_state.category_selector
        selected_hot_key = st.session_state.hot_target_selector
        symbol_code = CATEGORY_HOT_OPTIONS[selected_category][selected_hot_key]
        st.session_state.sidebar_search_input = symbol_code
    except Exception:
        pass

def get_symbol_from_query(query: str) -> str:
    query = query.strip().upper()
    for code, data in FULL_SYMBOLS_MAP.items():
        if query == code.upper(): return code
        if any(query == kw.upper() for kw in data["keywords"]): return code
    for code, data in FULL_SYMBOLS_MAP.items():
        if query == data["name"].upper(): return code
    if re.fullmatch(r'\d{4,6}', query) and not any(ext in query for ext in ['.TW', '.HK', '.SS', '-USD']):
        return f"{query}.TW"
    return query

@st.cache_data(ttl=3600, show_spinner="正在從 Yahoo Finance 獲取數據...")
def get_stock_data(symbol, period, interval):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        df.columns = [col.capitalize() for col in df.columns]
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        if len(df) > 1: df = df.iloc[:-1]
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def calculate_technical_indicators(df):
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    macd = ta.trend.MACD(df['Close'], window_fast=8, window_slow=17, window_sign=9)
    df['MACD_Line'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
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
def get_comprehensive_analysis_data(symbol):
    """獲取並整合估值、基本面、籌碼和消息面數據"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        analysis_data = {
            'valuation': {}, 'fundamentals': {}, 'chips': {}, 'news': "暫無相關新聞"
        }

        if info.get('quoteType') in ['INDEX', 'CRYPTOCURRENCY', 'ETF']:
            analysis_data['fundamentals']['summary'] = "指數、加密貨幣或ETF不適用基本面/估值分析。"
            return analysis_data

        # 估值分析
        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        peg = info.get('pegRatio')
        ps = info.get('priceToSalesTrailing12Months')
        ev_ebitda = info.get('enterpriseToEbitda')

        if pe and 0 < pe < 15: analysis_data['valuation']['P/E Ratio'] = (f"{pe:.2f}", "可能低估", "red")
        else: analysis_data['valuation']['P/E Ratio'] = (f"{pe:.2f}" if pe else "N/A", "正常或偏高", "orange")
        if pb and 0 < pb < 1: analysis_data['valuation']['P/B Ratio'] = (f"{pb:.2f}", "可能低估", "red")
        else: analysis_data['valuation']['P/B Ratio'] = (f"{pb:.2f}" if pb else "N/A", "正常或偏高", "orange")
        if peg and 0 < peg < 1: analysis_data['valuation']['PEG Ratio'] = (f"{peg:.2f}", "成長潛力佳", "red")
        else: analysis_data['valuation']['PEG Ratio'] = (f"{peg:.2f}" if peg else "N/A", "正常或偏高", "orange")
        if ps and 0 < ps < 1: analysis_data['valuation']['P/S Ratio'] = (f"{ps:.2f}", "可能低估", "red")
        else: analysis_data['valuation']['P/S Ratio'] = (f"{ps:.2f}" if ps else "N/A", "正常或偏高", "orange")
        if ev_ebitda and 0 < ev_ebitda < 10: analysis_data['valuation']['EV/EBITDA'] = (f"{ev_ebitda:.2f}", "可能低估", "red")
        else: analysis_data['valuation']['EV/EBITDA'] = (f"{ev_ebitda:.2f}" if ev_ebitda else "N/A", "正常或偏高", "orange")

        # 基本面分析
        roe = info.get('returnOnEquity', 0)
        debt_equity = info.get('debtToEquity')
        if roe > 0.15: analysis_data['fundamentals']['ROE'] = (f"{roe:.2%}", "資本運用效率高", "red")
        else: analysis_data['fundamentals']['ROE'] = (f"{roe:.2%}", "資本效率有待加強", "orange")
        if debt_equity is not None and debt_equity < 50: analysis_data['fundamentals']['負債權益比'] = (f"{debt_equity/100:.2%}", "財務狀況健康", "red")
        else: analysis_data['fundamentals']['負債權益比'] = (f"{debt_equity/100:.2%}" if debt_equity else "N/A", "財務風險較高", "green")
        analysis_data['fundamentals']['summary'] = "基本面數據已整合至下方判讀表。"

        # 籌碼面分析
        inst_own = info.get('heldPercentInstitutions', 0)
        insider_own = info.get('heldPercentInsiders', 0)
        analysis_data['chips']['法人持股比例'] = (f"{inst_own:.2%}", "籌碼穩定" if inst_own > 0.5 else "籌碼較分散", "red" if inst_own > 0.5 else "orange")
        analysis_data['chips']['內部人持股比例'] = (f"{insider_own:.2%}", "公司派信心高" if insider_own > 0.2 else "中性", "red" if insider_own > 0.2 else "orange")

        # 消息面分析
        news = ticker.news
        if news:
            analysis_data['news'] = "\n".join([f"- {item['title']}" for item in news[:3]])

        return analysis_data
    except Exception:
        return {'valuation': {}, 'fundamentals': {"summary": "無法獲取相關數據"}, 'chips': {}, 'news': "無法獲取新聞"}

def generate_ai_fusion_signal(df, other_data):
    if df.empty or len(df) < 20: return {'action': '數據不足', 'score': 0, 'confidence': 0, 'ai_interpretations': []}
    
    last, prev = df.iloc[-1], df.iloc[-2]
    score, interpretations = 0, []

    # --- 技術面判讀 ---
    if last['EMA_10'] > last['EMA_50'] > last['EMA_200']: score += 2; interpretations.append({'AI領域': '趨勢分析 (MA排列)', '判斷結果': '強多頭排列，趨勢向上', '顏色': 'red'})
    elif last['EMA_10'] < last['EMA_50'] < last['EMA_200']: score -= 2; interpretations.append({'AI領域': '趨勢分析 (MA排列)', '判斷結果': '強空頭排列，趨勢向下', '顏色': 'green'})
    else: interpretations.append({'AI領域': '趨勢分析 (MA排列)', '判斷結果': '盤整或趨勢不明', '顏色': 'orange'})

    if last['RSI'] < 30: score += 1.5; interpretations.append({'AI領域': '動能分析 (RSI 9)', '判斷結果': f"{last['RSI']:.2f} 進入超賣區，潛在反彈機會", '顏色': 'red'})
    elif last['RSI'] > 70: score -= 1.5; interpretations.append({'AI領域': '動能分析 (RSI 9)', '判斷結果': f"{last['RSI']:.2f} 進入超買區，潛在回調風險", '顏色': 'green'})
    else: interpretations.append({'AI領域': '動能分析 (RSI 9)', '判斷結果': f"{last['RSI']:.2f} 位於中性區間", '顏色': 'orange'})
    
    macd_cross_up = prev['MACD_Line'] < prev['MACD_Signal'] and last['MACD_Line'] > last['MACD_Signal']
    if macd_cross_up: score += 1.5; interpretations.append({'AI領域': '趨勢強度 (MACD)', '判斷結果': '黃金交叉，多頭動能增強', '顏色': 'red'})
    else: interpretations.append({'AI領域': '趨勢強度 (MACD)', '判斷結果': '柱狀圖 ' + ('放大' if last['MACD_Hist'] > 0 else '收斂'), '顏色': 'red' if last['MACD_Hist'] > 0 else 'green'})

    if last['ADX'] > 25: score *= 1.1; interpretations.append({'AI領域': '趨勢強度 (ADX 9)', '判斷結果': f"{last['ADX']:.2f} 強趨勢確認", '顏色': 'orange'})
    else: interpretations.append({'AI領域': '趨勢強度 (ADX 9)', '判斷結果': f"{last['ADX']:.2f} 盤整趨勢", '顏色': 'orange'})

    is_bullish_engulfing = prev['Close'] < prev['Open'] and last['Close'] > last['Open'] and last['Close'] > prev['Open'] and last['Open'] < prev['Close']
    if is_bullish_engulfing: score += 2; interpretations.append({'AI領域': 'K線形態分析', '判斷結果': '出現看漲吞噬型態', '顏色': 'red'})
    else: interpretations.append({'AI領域': 'K線形態分析', '判斷結果': '無明顯反轉信號', '顏色': 'orange'})
    
    # --- 整合其他維度判讀 ---
    for k, v in other_data['valuation'].items(): interpretations.append({'AI領域': f'估值分析 ({k})', '判斷結果': f"{v[0]} ({v[1]})", '顏色': v[2]})
    for k, v in other_data['fundamentals'].items(): 
        if k != 'summary': interpretations.append({'AI領域': f'基本面 ({k})', '判斷結果': f"{v[0]} ({v[1]})", '顏色': v[2]})
    for k, v in other_data['chips'].items(): interpretations.append({'AI領域': f'籌碼面 ({k})', '判斷結果': f"{v[0]} ({v[1]})", '顏色': v[2]})
    
    # --- 最終裁決 ---
    action = '中性 (Neutral)'
    if score > 4: action = '買進 (Buy)'
    elif score > 1.5: action = '中性偏買 (Hold/Buy)'
    elif score < -4: action = '賣出 (Sell/Short)'
    elif score < -1.5: action = '中性偏賣 (Hold/Sell)'
    
    confidence = min(100, abs(score) * 12 + 40)

    return {
        'current_price': last['Close'], 'action': action, 'score': score,
        'confidence': confidence, 'entry_price': last['Close'], 
        'take_profit': last['Close'] + last['ATR'] * 2 if score > 0 else last['Close'] - last['ATR'] * 2,
        'stop_loss': last['Close'] - last['ATR'] if score > 0 else last['Close'] + last['ATR'],
        'atr': last['ATR'], 'ai_interpretations': interpretations
    }

def get_technical_data_df(df):
    """擴充後的技術指標狀態表"""
    if df.empty or len(df) < 50: return pd.DataFrame()
    last = df.iloc[-1]
    data = []
    
    # 價格 vs EMA
    if last['Close'] > last['EMA_10'] > last['EMA_50']: conclusion, color = "多頭趨勢", "red"
    elif last['Close'] < last['EMA_10'] < last['EMA_50']: conclusion, color = "空頭趨勢", "green"
    else: conclusion, color = "中性盤整", "orange"
    data.append(['價格 vs. EMA 10/50/200', last['Close'], conclusion, color])

    # RSI
    if last['RSI'] < 30: conclusion, color = "強化：超賣", "red"
    elif last['RSI'] > 70: conclusion, color = "削弱：超買", "green"
    else: conclusion, color = "中性", "orange"
    data.append(['RSI (9) 動能', last['RSI'], conclusion, color])

    # MACD
    conclusion, color = ("多頭動能", "red") if last['MACD_Hist'] > 0 else ("空頭動能", "green")
    data.append(['MACD (8/17/9) 柱狀圖', last['MACD_Hist'], conclusion, color])
    
    # ADX
    conclusion, color = ("強趨勢", "red") if last['ADX'] >= 25 else ("盤整", "orange")
    data.append(['ADX (9) 趨勢強度', last['ADX'], conclusion, color])

    # ATR
    data.append(['ATR (9) 波動性', last['ATR'], "正常波動", "orange"])

    # 布林通道
    data.append(['布林通道 (BB: 20/2)', last['Close'], "區間震盪", "orange"])
    
    # OBV
    if last['OBV'] > df['OBV'].iloc[-2]: conclusion, color = "資金流入", "red"
    else: conclusion, color = "資金流出", "green"
    data.append(['OBV 能量潮', last['OBV'], conclusion, color])
    
    # Volume
    if last['Volume'] > last['Volume_MA_20']: conclusion, color = "放量", "red" if last['Close'] > last['Open'] else "green"
    else: conclusion, color = "縮量", "orange"
    data.append(['成交量 vs 20期均量', last['Volume'], conclusion, color])

    return pd.DataFrame(data, columns=['指標名稱', '最新數值', '趨勢/動能判讀', '顏色']).set_index('指標名稱')

def create_comprehensive_chart(df):
    if df.empty: return go.Figure()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2], specs=[[{"secondary_y": True}], [{}], [{}]])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    for w, c in [(10, 'orange'), (50, 'blue'), (200, 'red')]:
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{w}'], line=dict(color=c, width=1.5), name=f'EMA {w}'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='grey', name='成交量'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=np.where(df['MACD_Hist'] >= 0, 'green', 'red'), name='MACD Hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#0077b6', width=1.5), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def run_backtest(df, initial_capital=100000, commission_rate=0.001):
    if df.empty or len(df) < 51: return {"message": "數據不足無法回測"}
    data = df.copy()
    data['Signal'] = 0
    buy_signal = (data['SMA_20'] > data['EMA_50']) & (data['SMA_20'].shift(1) <= data['EMA_50'].shift(1))
    sell_signal = (data['SMA_20'] < data['EMA_50']) & (data['SMA_20'].shift(1) >= data['EMA_50'].shift(1))
    data.loc[buy_signal, 'Signal'] = 1
    data.loc[sell_signal, 'Signal'] = -1
    
    position, capital, trades, buy_price = 0, initial_capital, [], 0
    capital_curve = [initial_capital]
    
    for i in range(1, len(data)):
        current_capital_base = capital if position == 0 else capital_curve[-1]
        
        if data['Signal'].iloc[i] == 1 and position == 0:
            position = 1
            buy_price = data['Close'].iloc[i]
            capital = current_capital_base * (1 - commission_rate)
            capital_curve.append(capital)
        elif data['Signal'].iloc[i] == -1 and position == 1:
            profit = (data['Close'].iloc[i] - buy_price) / buy_price
            trades.append(1 if profit > 0 else 0)
            capital = capital * (1 + profit) * (1 - commission_rate)
            position = 0
            capital_curve.append(capital)
        else:
            if position == 1:
                equity = capital * (data['Close'].iloc[i] / buy_price)
                capital_curve.append(equity)
            else:
                capital_curve.append(capital)

    capital_s = pd.Series(capital_curve, index=data.index)
    total_return = (capital_s.iloc[-1] / initial_capital - 1) * 100
    win_rate = (sum(trades) / len(trades)) * 100 if trades else 0
    max_drawdown = (capital_s / capital_s.cummax() - 1).min() * 100
    
    return {
        "total_return": round(total_return, 2), "win_rate": round(win_rate, 2),
        "max_drawdown": round(abs(max_drawdown), 2), "total_trades": len(trades),
        "message": f"回測區間 {data.index[0].strftime('%Y-%m-%d')} 到 {data.index[-1].strftime('%Y-%m-%d')}",
        "capital_curve": capital_s
    }

# ==============================================================================
# 4. Streamlit 主應用程式邏輯
# ==============================================================================

def main():
    st.sidebar.title("🚀 AI 趨勢分析")
    st.sidebar.markdown("---")
    
    selected_category = st.sidebar.selectbox('1. 選擇資產類別', list(CATEGORY_HOT_OPTIONS.keys()), index=1, key='category_selector')
    hot_options_map = CATEGORY_HOT_OPTIONS.get(selected_category, {})
    
    selected_hot_option_key = st.sidebar.selectbox('2. 選擇熱門標的', list(hot_options_map.keys()), index=list(hot_options_map.values()).index('2330.TW') if '2330.TW' in hot_options_map.values() else 0, key='hot_target_selector', on_change=sync_text_input_from_selection)
    
    st.sidebar.text_input('...或手動輸入代碼/名稱:', key='sidebar_search_input')
    st.sidebar.selectbox('3. 選擇分析週期', list(PERIOD_MAP.keys()), index=2, key='period_selector')
    st.sidebar.markdown("---")

    if st.sidebar.button('📊 執行AI分析', use_container_width=True):
        final_symbol = get_symbol_from_query(st.session_state.sidebar_search_input)
        with st.spinner(f"🔍 正在啟動AI模型，分析 **{final_symbol}**..."):
            yf_period, yf_interval = PERIOD_MAP[st.session_state.period_selector]
            df = get_stock_data(final_symbol, yf_period, yf_interval)
            
            if df.empty or len(df) < 50:
                st.error(f"❌ **數據不足或代碼無效：** {final_symbol}。請檢查代碼或更換週期。")
                st.session_state.data_ready = False
            else:
                st.session_state.analysis_results = {
                    'df': calculate_technical_indicators(df),
                    'info': get_company_info(final_symbol),
                    'currency': get_currency_symbol(final_symbol),
                    'other_data': get_comprehensive_analysis_data(final_symbol),
                    'period': st.session_state.period_selector,
                    'symbol': final_symbol
                }
                st.session_state.data_ready = True

    if st.session_state.get('data_ready', False):
        res = st.session_state.analysis_results
        df, other_data, info = res['df'].dropna(), res['other_data'], res['info']
        analysis = generate_ai_fusion_signal(df, other_data)
        price, prev_close = analysis['current_price'], df['Close'].iloc[-2]
        change, pct_change = price - prev_close, (price - prev_close) / prev_close * 100

        st.header(f"📈 {info['name']} ({res['symbol']}) AI趨勢分析")
        st.markdown(f"**分析週期:** {res['period']} | **基本面診斷:** {other_data['fundamentals'].get('summary', 'N/A')}")
        st.markdown("---")

        st.subheader("💡 核心行動與量化評分")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 當前價格", f"{res['currency']}{price:,.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
        action_color = "red" if "買" in analysis['action'] else "green" if "賣" in analysis['action'] else "orange"
        col2.markdown(f"**🎯 最終行動建議**<p style='font-size: 20px; color: {action_color}; font-weight: bold;'>{analysis['action']}</p>", unsafe_allow_html=True)
        col3.metric("🔥 AI 綜合評分", f"{analysis['score']:.2f}", help="綜合技術指標的量化評分。")
        col4.metric("🛡️ 信心指數", f"{analysis['confidence']:.0f}%", help="AI模型對當前判斷的確定性程度。")
        st.markdown("---")
        
        st.subheader("📊 關鍵指標數據與AI判讀 (綜合評估)")
        interpret_df = pd.DataFrame(analysis['ai_interpretations'])
        
        def style_interpretations(df):
            def get_color(color_name):
                return f'color: {color_name};'
            return df.style.apply(lambda row: [get_color(row['顏色'])] * len(row), axis=1, subset=['判斷結果'])

        st.dataframe(style_interpretations(interpret_df), use_container_width=True)
        st.markdown("---")

        st.subheader("🛠️ 技術指標狀態總表")
        tech_df = get_technical_data_df(df)
        st.dataframe(tech_df.style.apply(lambda row: [f"color: {row['顏色']}"]*len(row), axis=1, subset=['趨勢/動能判讀']), use_container_width=True)
        st.markdown("---")
        
        with st.expander("📰 點此查看近期相關新聞"):
            st.markdown(other_data['news'])
        st.markdown("---")
        
        st.subheader("🧪 策略回測報告 (SMA 20/EMA 50 交叉)")
        backtest = run_backtest(df.copy())
        if "capital_curve" in backtest:
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("📊 總回報率", f"{backtest['total_return']}%", delta=backtest['message'])
            b_col2.metric("📈 勝率", f"{backtest['win_rate']}%")
            b_col3.metric("📉 最大回撤 (MDD)", f"{backtest['max_drawdown']}%")
            b_col4.metric("🤝 交易總次數", f"{backtest['total_trades']} 次")
            fig_bt = go.Figure(go.Scatter(x=backtest['capital_curve'].index, y=backtest['capital_curve'], name='策略資金曲線', line_color='orange'))
            st.plotly_chart(fig_bt.update_layout(title='策略資金曲線'), use_container_width=True)
        else:
            st.info(f"回測無法執行：{backtest.get('message', '未知錯誤')}")
        st.markdown("---")

        st.subheader("📊 完整技術分析圖表")
        st.plotly_chart(create_comprehensive_chart(df), use_container_width=True)

    else:
        st.markdown("<h1 style='color: #FA8072;'>🚀 歡迎使用 AI 趨勢分析</h1>", unsafe_allow_html=True)
        st.markdown("請在左側選擇或輸入您想分析的標的（例如：**2330.TW**、**NVDA**、**BTC-USD**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇資產類別**: 在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇標的**: 使用下拉選單或在輸入框中鍵入代碼/名稱。")
        st.markdown("3. **選擇週期**: 決定分析的時間級別。")
        st.markdown("4. **執行分析**: 點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span>，AI將提供融合分析結果。")

    st.markdown("---")
    st.markdown("⚠️ **免責聲明**")
    st.caption("本分析模型包含多位AI的量化觀點，但僅供教育與參考用途。投資涉及風險，所有交易決策應基於您個人的獨立研究和財務狀況，並建議諮詢專業金融顧問。")
    st.markdown("📊 **數據來源:** Yahoo Finance | **技術指標:** TA 庫 | **APP優化:** AI架構師")

if __name__ == '__main__':
    if 'data_ready' not in st.session_state: st.session_state.data_ready = False
    if 'sidebar_search_input' not in st.session_state: st.session_state.sidebar_search_input = "2330.TW"
    main()
