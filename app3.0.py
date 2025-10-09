# app_ultimate_version.py

import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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

# 週期映射
PERIOD_MAP = {
    "30 分": ("60d", "30m"),
    "4 小時": ("1y", "60m"),
    "1 日": ("5y", "1d"),
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】(與您提供的一致，此處省略以節省空間)
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
    query = query.strip()
    query_upper = query.upper()
    for code, data in FULL_SYMBOLS_MAP.items():
        if query_upper == code.upper(): return code
        if any(query_upper == kw.upper() for kw in data["keywords"]): return code
    for code, data in FULL_SYMBOLS_MAP.items():
        if query == data["name"]: return code
    if re.fullmatch(r'\d{4,6}', query) and not any(ext in query_upper for ext in ['.TW', '.HK', '.SS', '-USD']):
        return f"{query}.TW"
    return query

@st.cache_data(ttl=300, show_spinner="正在從 Yahoo Finance 獲取最新市場數據...")
def get_stock_data(symbol, period, interval):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        df.columns = [col.capitalize() for col in df.columns]
        df.index.name = 'Date'
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[~df.index.duplicated(keep='first')]
        if len(df) > 1: df = df.iloc[:-1]
        return df if not df.empty else pd.DataFrame()
    except Exception:
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
        quote_type = yf_info.get('quoteType', '')
        
        if quote_type == 'CRYPTOCURRENCY': category = "加密貨幣 (Crypto)"
        elif quote_type == 'INDEX': category = "指數"
        elif symbol.endswith(".TW"): category = "台股 (TW)"
        else: category = "美股 (US)"
        return {"name": name, "category": category, "currency": currency}
    except Exception:
        return {"name": symbol, "category": "未分類", "currency": "USD"}

@st.cache_data
def get_currency_symbol(symbol):
    currency_code = get_company_info(symbol).get('currency', 'USD')
    return 'NT$' if currency_code == 'TWD' else '$' if currency_code == 'USD' else currency_code + ' '


# ==============================================================================
# 3. 專業級 TP/SL 策略函式 (已啟用)
# ==============================================================================

def support_resistance(df, lookback=60):
    df['Support'] = df['Low'].rolling(window=lookback).min()
    df['Resistance'] = df['High'].rolling(window=lookback).max()
    df['Volume_Filter'] = df['Volume'] > df['Volume'].rolling(50).mean() * 1.3
    df['SL'] = df['Support'] * 0.98
    df['TP'] = df['Resistance'] * 1.02
    return df

def bollinger_bands_strategy(df, period=50, dev=2.5):
    df['SMA'] = df['Close'].rolling(window=period).mean()
    df['STD'] = df['Close'].rolling(window=period).std()
    df['Upper'] = df['SMA'] + (df['STD'] * dev)
    df['Lower'] = df['SMA'] - (df['STD'] * dev)
    df['RSI'] = pandas_rsi(df['Close'], 14)
    df['Volume_Filter'] = df['Volume'] > df['Volume'].rolling(50).mean() * 1.2
    
    # 買進條件: RSI超賣且爆量； 賣出條件: RSI超買且爆量
    buy_condition = (df['RSI'] < 30) & df['Volume_Filter']
    sell_condition = (df['RSI'] > 70) & df['Volume_Filter']
    
    # 預設SL/TP為NaN
    df['SL'] = np.nan
    df['TP'] = np.nan
    
    # 當前趨勢判斷 (基於SMA)
    if df['Close'].iloc[-1] > df['SMA'].iloc[-1]: # 多頭趨勢
        df['SL'] = df['Lower']
        df['TP'] = df['Upper']
    else: # 空頭趨勢
        df['SL'] = df['Upper']
        df['TP'] = df['Lower']
    return df

def atr_stop(df, period=21, multiplier_sl=2.5, multiplier_tp=5):
    df['ATR'] = pandas_atr(df, period=period)
    df['ADX'] = pandas_adx(df, period=14)
    df['Trend_Filter'] = df['ADX'] > 25
    
    # 根據趨勢設定SL/TP
    df['SL'] = np.where(df['Trend_Filter'], df['Close'] - (df['ATR'] * multiplier_sl), np.nan)
    df['TP'] = np.where(df['Trend_Filter'], df['Close'] + (df['ATR'] * multiplier_tp), np.nan)
    return df

def donchian_channel(df, period=50):
    df['Upper'] = df['High'].rolling(window=period).max()
    df['Lower'] = df['Low'].rolling(window=period).min()
    df['MACD_Line'], _, _ = pandas_macd(df['Close'])
    df['Volume_Filter'] = df['Volume'] > df['Volume'].rolling(50).mean() * 1.3
    
    buy_signal = (df['MACD_Line'] > 0) & df['Volume_Filter']
    sell_signal = (df['MACD_Line'] < 0) & df['Volume_Filter']
    
    df['SL'] = np.where(buy_signal, df['Lower'], np.nan)
    df['TP'] = np.where(buy_signal, df['Upper'], np.nan)
    return df

def keltner_channel(df, period=30, atr_multiplier=2.5, atr_period=14):
    df['EMA'] = df['Close'].ewm(span=period, adjust=False).mean()
    df['ATR'] = pandas_atr(df, period=atr_period)
    df['Upper'] = df['EMA'] + (df['ATR'] * atr_multiplier)
    df['Lower'] = df['EMA'] - (df['ATR'] * atr_multiplier)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Filter'] = df['OBV'] > df['OBV'].shift(1)
    
    # 多頭趨勢下，下軌為支撐(SL)，上軌為目標(TP)
    df['SL'] = np.where(df['OBV_Filter'], df['Lower'], np.nan)
    df['TP'] = np.where(df['OBV_Filter'], df['Upper'], np.nan)
    return df

def ichimoku_cloud(df):
    high_9 = df['High'].rolling(9).max()
    low_9 = df['High'].rolling(9).min()
    df['Tenkan'] = (high_9 + low_9) / 2
    
    high_26 = df['High'].rolling(26).max()
    low_26 = df['Low'].rolling(26).min()
    df['Kijun'] = (high_26 + low_26) / 2
    
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    
    high_52 = df['High'].rolling(52).max()
    low_52 = df['Low'].rolling(52).min()
    df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
    
    # 價格在雲之上，雲層為支撐區；反之為壓力區
    price = df['Close']
    if price.iloc[-1] > df['Senkou_A'].iloc[-1] and price.iloc[-1] > df['Senkou_B'].iloc[-1]:
        df['SL'] = df[['Senkou_A', 'Senkou_B']].min(axis=1)
        df['TP'] = price + (price - df['SL']) * 2 # 簡單的目標價
    else:
        df['TP'] = df[['Senkou_A', 'Senkou_B']].min(axis=1)
        df['SL'] = df[['Senkou_A', 'Senkou_B']].max(axis=1)
    return df

def ma_crossover(df, fast=20, slow=50):
    df['Fast_MA'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['Slow_MA'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Filter'] = df['OBV'] > df['OBV'].shift(1)

    # 黃金交叉且OBV向上，慢線為支撐；死亡交叉則快線為壓力
    is_golden_cross = df['Fast_MA'] > df['Slow_MA']
    
    df['SL'] = np.where(is_golden_cross & df['OBV_Filter'], df['Slow_MA'], df['Fast_MA'])
    df['TP'] = np.where(is_golden_cross & df['OBV_Filter'], df['Fast_MA'] * 1.05, df['Slow_MA'] * 0.95) # 示例目標
    return df

def vwap_strategy(df):
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['Cum_Vol_Price'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
    df['VWAP'] = df['Cum_Vol_Price'] / df['Cum_Vol']
    
    # VWAP 作為動態支撐/阻力
    df['SL'] = df['VWAP'] * 0.98
    df['TP'] = df['VWAP'] * 1.02
    return df

def trailing_stop(df, atr_period=14, atr_multiplier=3):
    df['ATR'] = pandas_atr(df, period=atr_period)
    df['SL'] = df['Close'] - (df['ATR'] * atr_multiplier)
    df['TP'] = df['Close'] + (df['ATR'] * 2 * atr_multiplier) # R:R=2:1
    return df

def chandelier_exit(df, period=22, atr_multiplier=3.5):
    df['ATR'] = pandas_atr(df, period=14)
    df['High_Max'] = df['High'].rolling(window=period).max()
    df['Low_Min'] = df['Low'].rolling(window=period).min()
    
    # Chandelier Exit (Long and Short)
    df['SL_Long'] = df['High_Max'] - df['ATR'] * atr_multiplier
    df['TP_Short'] = df['Low_Min'] + df['ATR'] * atr_multiplier
    
    # 根據當前價格位置決定使用多頭或空頭止損
    if df['Close'].iloc[-1] > df['Close'].iloc[-period]: # 簡易上升趨勢
        df['SL'] = df['SL_Long']
        df['TP'] = df['Close'].iloc[-1] + (df['Close'].iloc[-1] - df['SL_Long'].iloc[-1]) * 2
    else:
        df['SL'] = df['TP_Short'] # 在空頭趨勢中， chandelier exit (short) 可作為止損
        df['TP'] = df['Close'].iloc[-1] - (df['TP_Short'].iloc[-1] - df['Close'].iloc[-1]) * 2
    return df

def supertrend(df, period=14, multiplier=3.5):
    df['ATR'] = pandas_atr(df, period=period)
    df['Upper_Band'] = ((df['High'] + df['Low']) / 2) + (multiplier * df['ATR'])
    df['Lower_Band'] = ((df['High'] + df['Low']) / 2) - (multiplier * df['ATR'])
    
    df['Supertrend'] = df['Lower_Band'] # 預設
    for i in range(1, len(df)):
        if df['Close'].iloc[i-1] <= df['Supertrend'].iloc[i-1]:
            df.loc[df.index[i], 'Supertrend'] = min(df['Upper_Band'].iloc[i], df['Supertrend'].iloc[i-1])
        else:
            df.loc[df.index[i], 'Supertrend'] = max(df['Lower_Band'].iloc[i], df['Supertrend'].iloc[i-1])
            
    if df['Close'].iloc[-1] > df['Supertrend'].iloc[-1]: # 上升趨勢
        df['SL'] = df['Supertrend']
        df['TP'] = df['Close'] + (df['Close'] - df['Supertrend']) * 2
    else: # 下降趨勢
        df['SL'] = df['Supertrend']
        df['TP'] = df['Close'] - (df['Supertrend'] - df['Close']) * 2
    return df

def pivot_points(df):
    df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['S1'] = (2 * df['Pivot']) - df['High'].shift(1)
    df['R1'] = (2 * df['Pivot']) - df['Low'].shift(1)
    df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
    df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
    
    price = df['Close']
    if price.iloc[-1] > df['Pivot'].iloc[-1]: # 價格在樞軸點之上
        df['SL'] = df['S1']
        df['TP'] = df['R1']
    else: # 價格在樞軸點之下
        df['SL'] = df['R1']
        df['TP'] = df['S1']
    return df

# 策略字典
STRATEGY_FUNCTIONS = {
    "支撐與阻力 (Support & Resistance)": support_resistance,
    "布林通道策略 (Bollinger Bands)": bollinger_bands_strategy,
    "ATR 停損 (ATR Stop)": atr_stop,
    "唐奇安通道 (Donchian Channel)": donchian_channel,
    "肯特納通道 (Keltner Channel)": keltner_channel,
    "一目均衡表 (Ichimoku Cloud)": ichimoku_cloud,
    "均線交叉 (MA Crossover)": ma_crossover,
    "VWAP 策略 (VWAP Strategy)": vwap_strategy,
    "移動止損 (Trailing Stop)": trailing_stop,
    "吊燈停損 (Chandelier Exit)": chandelier_exit,
    "超級趨勢 (Supertrend)": supertrend,
    "樞軸點 (Pivot Points)": pivot_points,
}

# ==============================================================================
# 4. 核心分析與指標計算 (無需 TA-Lib)
# ==============================================================================

def pandas_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def pandas_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.ewm(alpha=1/period, adjust=False).mean()

def pandas_adx(df, period=14):
    atr = pandas_atr(df, period)
    up_move = df['High'].diff()
    down_move = df['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), -down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.ewm(alpha=1/period, adjust=False).mean()
    
def pandas_macd(close, fast=8, slow=17, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def calculate_technical_indicators(df):
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    df['MACD_Line'], df['MACD_Signal'], df['MACD_Hist'] = pandas_macd(df['Close'])
    df['RSI'] = pandas_rsi(df['Close'], period=9)
    
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_High'] = sma20 + (std20 * 2)
    df['BB_Low'] = sma20 - (std20 * 2)
    
    df['ATR'] = pandas_atr(df, period=9)
    df['ADX'] = pandas_adx(df, period=9)
    
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['OBV_MA_20'] = df['OBV'].rolling(window=20).mean()
    return df

@st.cache_data(ttl=3600)
def get_chips_and_news_analysis(symbol):
    try:
        ticker = yf.Ticker(symbol)
        inst_holders = ticker.institutional_holders
        inst_hold_pct = 0
        if inst_holders is not None and not inst_holders.empty:
            value = inst_holders.iloc[0, 2] # '% of Shares Held by Institutions' column
            inst_hold_pct = float(str(value).replace('%','')) / 100 if isinstance(value, str) else float(value)

        news = ticker.news
        news_summary = ""
        if news:
            headlines = [f"- {item['title']}" for item in news[:5]]
            news_summary = "\n".join(headlines)

        return {"inst_hold_pct": inst_hold_pct, "news_summary": news_summary}
    except Exception:
        return {"inst_hold_pct": 0, "news_summary": "無法獲取新聞。"}

@st.cache_data(ttl=3600)
def calculate_advanced_fundamental_rating(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if info.get('quoteType') in ['INDEX', 'CRYPTOCURRENCY', 'ETF']:
            return {"score": 0, "summary": "不適用基本面分析。", "details": {}}
        score, details = 0, {}
        roe = info.get('returnOnEquity')
        if roe and roe > 0.15: score += 2; details['ROE > 15%'] = f"✅ {roe:.2%}"
        else: details['ROE < 15%'] = f"❌ {roe:.2%}" if roe is not None else "N/A"
        
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None and debt_to_equity < 50: score += 2; details['負債權益比 < 50'] = f"✅ {debt_to_equity:.2f}"
        else: details['負債權益比 > 50'] = f"❌ {debt_to_equity:.2f}" if debt_to_equity is not None else "N/A"
        
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth and revenue_growth > 0.1: score += 1; details['營收年增 > 10%'] = f"✅ {revenue_growth:.2%}"
        else: details['營收年增 < 10%'] = f"❌ {revenue_growth:.2%}" if revenue_growth is not None else "N/A"

        pe = info.get('trailingPE')
        if pe and 0 < pe < 15: score += 1; details['P/E < 15'] = f"✅ {pe:.2f}"
        else: details['P/E > 15'] = f"⚠️ {pe:.2f}" if pe else "N/A"
        
        peg = info.get('pegRatio')
        if peg and 0 < peg < 1: score += 1; details['PEG < 1'] = f"✅ {peg:.2f}"
        else: details['PEG > 1'] = f"⚠️ {peg:.2f}" if peg else "N/A"
        
        summary = "頂級優異" if score >= 5 else "良好穩健" if score >= 3 else "中性警示"
        return {"score": score, "summary": summary, "details": details}
    except Exception:
        return {"score": 0, "summary": "無法獲取數據。", "details": {}}

def generate_ai_fusion_signal(df, fa_rating, chips_news_data):
    df_clean = df.dropna()
    if df_clean.empty or len(df_clean) < 2: return {'action': '數據不足', 'score': 0, 'confidence': 0}
    
    last, prev = df_clean.iloc[-1], df_clean.iloc[-2]
    opinions = {}

    ta_score = 0
    if last['EMA_10'] > last['EMA_50'] > last['EMA_200']: ta_score += 2; opinions['趨勢分析 (MA)'] = '✅ 強多頭排列'
    elif last['EMA_10'] < last['EMA_50'] < last['EMA_200']: ta_score -= 2; opinions['趨勢分析 (MA)'] = '❌ 強空頭排列'
    else: opinions['趨勢分析 (MA)'] = '⚠️ 中性盤整'
    
    if last['RSI'] > 70: ta_score -= 1.5; opinions['動能分析 (RSI)'] = '❌ 超買區域'
    elif last['RSI'] < 30: ta_score += 1.5; opinions['動能分析 (RSI)'] = '✅ 超賣區域'
    elif last['RSI'] > 50: ta_score += 1; opinions['動能分析 (RSI)'] = '✅ 多頭區間'
    else: ta_score -= 1; opinions['動能分析 (RSI)'] = '❌ 空頭區間'

    if last['MACD_Hist'] > 0 and last['MACD_Hist'] > prev['MACD_Hist']: ta_score += 1.5; opinions['動能分析 (MACD)'] = '✅ 多頭動能增強'
    elif last['MACD_Hist'] < 0 and last['MACD_Hist'] < prev['MACD_Hist']: ta_score -= 1.5; opinions['動能分析 (MACD)'] = '❌ 空頭動能增強'
    else: opinions['動能分析 (MACD)'] = '⚠️ 動能盤整'
        
    if last['ADX'] > 25: ta_score *= 1.2; opinions['趨勢強度 (ADX)'] = f'✅ 強趨勢確認'
    else: ta_score *= 0.8; opinions['趨勢強度 (ADX)'] = f'⚠️ 盤整趨勢'

    fa_score = ((fa_rating.get('score', 0) / 7.0) - 0.5) * 8.0 # Max score is 7
    chips_score = (chips_news_data.get('inst_hold_pct', 0) - 0.4) * 4
    
    volume_score = 0
    if not pd.isna(last['OBV_MA_20']) and last['OBV'] > last['OBV_MA_20']: volume_score += 1; opinions['成交量 (OBV)'] = '✅ OBV 在均線之上'
    else: volume_score -=1; opinions['成交量 (OBV)'] = '❌ OBV 在均線之下'
    
    total_score = ta_score * 0.5 + fa_score * 0.25 + (chips_score + volume_score) * 0.25
    confidence = min(100, 50 + abs(total_score) * 8)
    
    action = '中性/觀望'
    if total_score > 4: action = '買進 (Buy)'
    elif total_score > 1.5: action = '中性偏買 (Hold/Buy)'
    elif total_score < -4: action = '賣出 (Sell/Short)'
    elif total_score < -1.5: action = '中性偏賣 (Hold/Sell)'
    
    return {'action': action, 'score': total_score, 'confidence': confidence, 'ai_opinions': opinions}

# ==============================================================================
# 5. 回測與圖表繪製
# ==============================================================================
def run_backtest(df, initial_capital=100000):
    try:
        # 策略: SMA 20 / EMA 50 交叉
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        df['position'] = np.where(df['SMA_20'] > df['EMA_50'], 1, -1)
        df['returns'] = df['Close'].pct_change()
        df['strategy_returns'] = df['returns'] * df['position'].shift(1)
        
        # 計算指標
        cumulative_returns = (1 + df['strategy_returns']).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        
        trades = df['position'].diff().ne(0) & df['position'].shift().ne(0)
        total_trades = trades.sum()
        if total_trades == 0:
            return {"total_trades": 0, "message": "無交易信號"}
            
        trade_returns = df['strategy_returns'][trades]
        win_rate = (trade_returns > 0).sum() / total_trades * 100 if total_trades > 0 else 0
        
        # 最大回撤
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        capital_curve = initial_capital * cumulative_returns
        
        return {
            "total_return": f"{total_return:.2f}",
            "win_rate": f"{win_rate:.2f}",
            "max_drawdown": f"{max_drawdown:.2f}",
            "total_trades": total_trades,
            "capital_curve": capital_curve,
            "message": "相對初始資金的回報"
        }
    except Exception as e:
        return {"total_trades": 0, "message": f"回測錯誤: {e}"}

def create_comprehensive_chart(df, symbol, period_key):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2])

    # K線主圖
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_10'], mode='lines', name='EMA 10', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='blue', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200', line=dict(color='red', width=2)), row=1, col=1)

    # MACD副圖
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Line'], name='MACD Line', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal Line', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=np.where(df['MACD_Hist'] > 0, 'green', 'red')), row=2, col=1)

    # RSI副圖
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.2, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.2, row=3, col=1)

    fig.update_layout(
        title=f'{symbol} 技術分析圖 ({period_key})',
        xaxis_rangeslider_visible=False,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    return fig

# ==============================================================================
# 6. UI 呈現與主邏輯
# ==============================================================================
def main():
    if 'run_analysis' not in st.session_state: st.session_state['run_analysis'] = False

    st.sidebar.title("🚀 AI 趨勢分析")
    st.sidebar.markdown("---")
    
    selected_category = st.sidebar.selectbox('1. 選擇資產類別', list(CATEGORY_HOT_OPTIONS.keys()), index=2, key='category_selector')
    hot_options_map = CATEGORY_HOT_OPTIONS.get(selected_category, {})
    default_index = 0
    if selected_category == '加密貨幣 (Crypto)' and 'SOL-USD - Solana' in hot_options_map:
        default_index = list(hot_options_map.keys()).index('SOL-USD - Solana')
    
    st.sidebar.selectbox('2. 選擇熱門標的', list(hot_options_map.keys()), index=default_index, key='hot_target_selector', on_change=sync_text_input_from_selection)
    st.sidebar.text_input('...或手動輸入代碼/名稱:', st.session_state.get('sidebar_search_input', 'SOL-USD'), key='sidebar_search_input')
    
    st.sidebar.markdown("---")
    selected_period_key = st.sidebar.selectbox('3. 選擇分析週期', list(PERIOD_MAP.keys()), index=2)
    selected_strategy_name = st.sidebar.selectbox('4. 選擇交易策略', list(STRATEGY_FUNCTIONS.keys()), index=0)
    st.sidebar.markdown("---")
    
    if st.sidebar.button('📊 執行AI分析', use_container_width=True):
        st.session_state['run_analysis'] = True
        st.session_state['symbol_to_analyze'] = get_symbol_from_query(st.session_state.sidebar_search_input)
        st.session_state['period_key'] = selected_period_key
        st.session_state['strategy_name'] = selected_strategy_name

    if st.session_state.get('run_analysis', False):
        final_symbol = st.session_state['symbol_to_analyze']
        period_key = st.session_state['period_key']
        strategy_name = st.session_state['strategy_name']
        period, interval = PERIOD_MAP[period_key]

        with st.spinner(f"🔍 正在啟動AI模型，分析 **{final_symbol}**..."):
            df_raw = get_stock_data(final_symbol, period, interval)
            
            if df_raw.empty or len(df_raw) < 52:
                st.error(f"❌ **數據不足或代碼無效：** {final_symbol}。AI模型至少需要52個數據點才能進行精準分析。")
            else:
                info = get_company_info(final_symbol)
                fa_rating = calculate_advanced_fundamental_rating(final_symbol)
                chips_data = get_chips_and_news_analysis(final_symbol)
                
                # 流程：1. 基礎指標 -> 2. AI融合信號 -> 3. 獨立策略TP/SL
                df_tech = calculate_technical_indicators(df_raw.copy())
                analysis = generate_ai_fusion_signal(df_tech, fa_rating, chips_data)
                
                strategy_func = STRATEGY_FUNCTIONS[strategy_name]
                df_strategy = strategy_func(df_tech.copy())
                
                strategy_sl = df_strategy.iloc[-1].get('SL', np.nan)
                strategy_tp = df_strategy.iloc[-1].get('TP', np.nan)
                
                st.header(f"📈 {info['name']} ({final_symbol}) AI趨勢分析報告")

                # --- 根據標的類型顯示不同標頭 ---
                if info.get('category') in ["加密貨幣 (Crypto)", "指數"]:
                    st.markdown(f"**分析週期:** {period_key} | **標的類型:** {info.get('category')} (不適用基本面分析)")
                else:
                    st.markdown(f"**分析週期:** {period_key} | **FA評級:** **{fa_rating.get('score',0):.1f}/7.0** | **診斷:** {fa_rating.get('summary','N/A')}")
                st.markdown("---")
                
                st.subheader("💡 核心行動與量化評分")
                price = df_raw['Close'].iloc[-1]
                prev_close = df_raw['Close'].iloc[-2]
                change, pct = price - prev_close, (price - prev_close) / prev_close * 100
                currency_symbol = get_currency_symbol(final_symbol)
                
                c1, c2, c3, c4 = st.columns(4)
                pf = ".4f" if price < 100 and currency_symbol != 'NT$' else ".2f"
                c1.metric("💰 當前價格", f"{currency_symbol}{price:{pf}}", f"{change:{pf}} ({pct:+.2f}%)")
                c2.metric("🎯 AI 行動建議", analysis['action'])
                c3.metric("🔥 AI 總量化評分", f"{analysis['score']:.2f}")
                c4.metric("🛡️ AI 信心指數", f"{analysis['confidence']:.0f}%")
                
                st.markdown("---")
                st.subheader(f"🛡️ 精確交易策略 ({strategy_name})")
                s1, s2, s3 = st.columns(3)
                s1.metric("建議進場價:", f"{currency_symbol}{price:{pf}}")
                s2.metric("🚀 止盈價 (TP):", f"{currency_symbol}{strategy_tp:{pf}}" if pd.notna(strategy_tp) else "N/A")
                s3.metric("🛑 止損價 (SL):", f"{currency_symbol}{strategy_sl:{pf}}" if pd.notna(strategy_sl) else "N/A")
                
                st.markdown("---")
                st.subheader("📊 AI判讀細節")
                opinions = list(analysis['ai_opinions'].items())
                if fa_rating['details']:
                    for k, v in fa_rating['details'].items(): opinions.append([f"基本面 - {k}", str(v)])
                st.dataframe(pd.DataFrame(opinions, columns=['分析維度', '判斷結果']), use_container_width=True)
                
                st.markdown("---")
                st.subheader("🧪 策略回測報告 (SMA 20/EMA 50 交叉)")
                bt = run_backtest(df_raw.copy())
                if bt.get("total_trades", 0) > 0:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("📊 總回報率", f"{bt['total_return']}%", delta=bt['message'], delta_color='off')
                    b2.metric("📈 勝率", f"{bt['win_rate']}%")
                    b3.metric("📉 最大回撤", f"{bt['max_drawdown']}%")
                    b4.metric("🤝 交易次數", f"{bt['total_trades']} 次")
                    if 'capital_curve' in bt and not bt['capital_curve'].empty:
                        fig = go.Figure(go.Scatter(x=bt['capital_curve'].index, y=bt['capital_curve'], name='資金曲線'))
                        fig.update_layout(title='SMA 20/EMA 50 交叉策略資金曲線', height=300)
                        st.plotly_chart(fig, use_container_width=True)
                else: st.warning(f"回測無法執行：{bt.get('message', '錯誤')}")
                
                st.markdown("---")
                st.subheader(f"📊 完整技術分析圖表")
                st.plotly_chart(create_comprehensive_chart(df_tech, final_symbol, period_key), use_container_width=True)
                with st.expander("📰 點此查看近期相關新聞"): st.markdown(chips_data['news_summary'].replace("\n", "\n\n"))

    else:
        st.markdown("<h1 style='color: #FA8072;'>🚀 歡迎使用 AI 趨勢分析</h1>", unsafe_allow_html=True)
        st.markdown(f"請在左側選擇或輸入您想分析的標的（例如：**2330.TW**、**NVDA**、**BTC-USD**），然後點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇標的**：使用下拉選單快速選擇熱門標的，或直接在輸入框中鍵入代碼或名稱。")
        st.markdown("3. **選擇週期**：決定分析的長度（例如：`30 分` (短期)、`1 日` (中長線)）。")
        st.markdown("4. **選擇交易策略**：選擇一種您偏好的策略來計算精確的止盈止損點。")
        st.markdown(f"5. **執行分析**：點擊 <span style='color: #FA8072; font-weight: bold;'>『📊 執行AI分析』</span>，AI將融合綜合指標與您指定的策略提供完整報告。", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    st.markdown("---")
    st.markdown("⚠️ **免責聲明**")
    st.caption("本分析模型包含AI的量化觀點，並結合多種技術分析策略，但僅供教育與參考用途。投資涉及風險，所有交易決策應基於您個人的獨立研究和財務狀況，並建議諮詢專業金融顧問。")
    st.markdown("📊 **數據來源:** Yahoo Finance")
