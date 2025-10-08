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
    # ... (此處省略完整列表，與您提供的原始碼相同) ...
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
        df = ticker.history(period=period, interval=interval)

        if df.empty: return None, "數據獲取失敗或該代碼無數據。"

        df.columns = [col.capitalize() for col in df.columns]
        df.index.name = 'Date'
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        df = df[~df.index.duplicated(keep='first')]

        if len(df) < 50: # 確保有足夠數據進行基本分析
             return None, f"數據長度不足 (少於50筆)，無法進行有效分析。"

        # 移除最後一筆可能尚未收盤的即時數據，確保分析一致性
        df = df.iloc[:-1]
        
        return df, "數據獲取成功"
    except Exception as e:
        return None, f"獲取數據時發生錯誤: {e}"

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
    except Exception as e:
        return {"name": symbol, "category": "未分類", "currency": "USD"}

@st.cache_data
def get_currency_symbol(symbol):
    currency_code = get_company_info(symbol).get('currency', 'USD')
    if currency_code == 'TWD': return 'NT$'
    elif currency_code == 'USD': return '$'
    elif currency_code == 'HKD': return 'HK$'
    else: return currency_code + ' '

def calculate_technical_indicators(df):
    # EMA 10, 50, 200 (符合原則)
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)

    # MACD 8, 17, 9 (符合原則)
    macd_instance = ta.trend.MACD(df['Close'], window_fast=8, window_slow=17, window_sign=9)
    df['MACD_Line'] = macd_instance.macd()
    df['MACD_Signal'] = macd_instance.macd_signal()
    df['MACD_Hist'] = macd_instance.macd_diff()

    # RSI 9 (符合原則)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=9)
    
    # ADX 9 (符合原則)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=9)

    # 成交量 MA 20 (符合原則)
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    
    # OBV (符合原則)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

    # 其他輔助指標
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)

    return df

# *** 新增 *** @st.cache_data(ttl=3600)
def get_chips_and_news_analysis(symbol):
    """獲取籌碼面和消息面數據"""
    try:
        ticker = yf.Ticker(symbol)
        
        # 籌碼面分析
        major_holders = ticker.major_holders
        institutional_holders = ticker.institutional_holders
        
        chips_summary = "暫無籌碼數據"
        if major_holders is not None and not major_holders.empty:
            insider_hold = major_holders.iloc[0, 0]
            chips_summary = f"內部大股東持股: {insider_hold}"
        if institutional_holders is not None and not institutional_holders.empty:
            inst_hold = institutional_holders.iloc[0, 0]
            chips_summary += f"，法人機構持股: {inst_hold}"
        
        # 消息面分析
        news = ticker.news
        news_summary = "近期無相關新聞"
        if news:
            headlines = [f"- {item['title']}" for item in news[:3]] # 取前三條
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

# *** 重大修改 ***
@st.cache_data(ttl=3600)
def calculate_advanced_fundamental_rating(symbol):
    """根據進階原則計算基本面評分"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if info.get('quoteType') in ['INDEX', 'CRYPTOCURRENCY', 'ETF']:
            return {"score": 0, "summary": "指數、加密貨幣或ETF不適用基本面分析。", "details": {}}

        score = 0
        details = {}
        
        # 獲利能力
        roe = info.get('returnOnEquity')
        if roe and roe > 0.15:
            score += 2
            details['✅ ROE > 15%'] = f"{roe:.2%}"
        else:
            details['❌ ROE < 15%'] = f"{roe:.2%}" if roe else "N/A"

        # 財務健康
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity and debt_to_equity < 50: # yfinance debtToEquity is ratio*100
            score += 2
            details['✅ 負債權益比 < 50%'] = f"{debt_to_equity/100:.2%}"
        else:
            details['❌ 負債權益比 > 50%'] = f"{debt_to_equity/100:.2%}" if debt_to_equity else "N/A"
            
        # 成長性
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth and revenue_growth > 0.1:
            score += 1
            details['✅ 營收年增 > 10%'] = f"{revenue_growth:.2%}"
        else:
            details['❌ 營收年增 < 10%'] = f"{revenue_growth:.2%}" if revenue_growth else "N/A"

        # 估值
        pe = info.get('trailingPE')
        peg = info.get('pegRatio')
        if pe and pe < 15:
            score += 1
            details['✅ 本益比(P/E) < 15'] = f"{pe:.2f}"
        else:
            details['⚠️ 本益比(P/E) > 15'] = f"{pe:.2f}" if pe else "N/A"
        
        if peg and peg < 1:
            score += 1
            details['✅ PEG < 1'] = f"{peg:.2f}"
        else:
            details['⚠️ PEG > 1'] = f"{peg:.2f}" if peg else "N/A"
        
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


# *** 重大修改 ***
def generate_ai_synthesis_report(df, fa_results, chips_news_results):
    """生成AI綜合分析報告"""
    report = {
        "final_action": "中性 (Neutral)",
        "confidence": 50,
        "summary": "AI正在分析數據...",
        "technical_opinion": "",
        "fundamental_opinion": fa_results['summary'],
        "chips_opinion": chips_news_results['chips_summary'],
        "news_opinion": chips_news_results['news_summary'],
        "entry_price": df.iloc[-1]['Close'],
        "take_profit": df.iloc[-1]['Close'] + df.iloc[-1]['ATR'],
        "stop_loss": df.iloc[-1]['Close'] - df.iloc[-1]['ATR'],
    }
    
    last = df.iloc[-1]
    bullish_signals = 0
    bearish_signals = 0
    
    # 1. 技術面分析
    tech_points = []
    # MA 趨勢
    if last['EMA_10'] > last['EMA_50'] and last['EMA_50'] > last['EMA_200']:
        bullish_signals += 2
        tech_points.append("MA呈強多頭排列 (10>50>200)")
    elif last['EMA_10'] < last['EMA_50'] and last['EMA_50'] < last['EMA_200']:
        bearish_signals += 2
        tech_points.append("MA呈強空頭排列 (10<50<200)")
    elif last['Close'] > last['EMA_200']:
        bullish_signals += 1
        tech_points.append("價格位於200期EMA之上，長線偏多")
    else:
        bearish_signals += 1
        tech_points.append("價格位於200期EMA之下，長線偏空")
    # RSI
    if last['RSI'] > 50:
        bullish_signals += 1
        tech_points.append(f"RSI({last['RSI']:.1f}) > 50 處於多頭區間")
    else:
        bearish_signals += 1
        tech_points.append(f"RSI({last['RSI']:.1f}) < 50 處於空頭區間")
    # MACD
    if last['MACD_Hist'] > 0:
        bullish_signals += 1
        tech_points.append("MACD柱狀體 > 0，多頭動能持續")
    else:
        bearish_signals += 1
        tech_points.append("MACD柱狀體 < 0，空頭動能持續")
    # ADX
    if last['ADX'] > 25:
        tech_points.append(f"ADX({last['ADX']:.1f}) > 25，趨勢明確")
    else:
        tech_points.append(f"ADX({last['ADX']:.1f}) < 25，處於盤整")

    report['technical_opinion'] = " | ".join(tech_points)
    
    # 2. 基本面分數
    if fa_results['score'] >= 4:
        bullish_signals += 2
    elif fa_results['score'] < 2:
        bearish_signals += 1

    # 綜合判斷
    total_signals = bullish_signals + bearish_signals
    confidence = int((abs(bullish_signals - bearish_signals) / total_signals) * 50 + 50) if total_signals > 0 else 50
    
    summary_parts = []
    if bullish_signals > bearish_signals + 1:
        action = "買進 (Buy)"
        summary_parts.append("多個指標顯示多頭趨勢。")
        if "MA呈強多頭排列" in report['technical_opinion']: summary_parts.append("技術面呈現強勢的多頭排列。")
        if fa_results['score'] >= 4: summary_parts.append("基本面穩健，提供長線支撐。")
        
    elif bearish_signals > bullish_signals + 1:
        action = "賣出 (Sell)"
        summary_parts.append("多個指標顯示空頭趨勢。")
        if "MA呈強空頭排列" in report['technical_opinion']: summary_parts.append("技術面呈現明確的空頭排列。")
        if fa_results['score'] < 2: summary_parts.append("基本面存在潛在風險。")

    else:
        action = "中性 (Neutral)"
        summary_parts.append("市場方向不明朗，多空信號交錯。")
        if last['ADX'] < 25: summary_parts.append("ADX顯示目前可能處於盤整區間，建議觀望。")
    
    report.update({
        "final_action": action,
        "confidence": confidence,
        "summary": " ".join(summary_parts),
        "take_profit": last['Close'] + (last['ATR'] * 2),
        "stop_loss": last['Close'] - (last['ATR'] * 1.5)
    })
    
    return report


def create_comprehensive_chart(df, symbol, period_key):
    df_clean = df.dropna()
    if df_clean.empty:
        return go.Figure()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        specs=[[{"secondary_y": True}], [{}], [{}], [{}]])

    fig.add_trace(go.Candlestick(x=df_clean.index,
                                 open=df_clean['Open'], high=df_clean['High'],
                                 low=df_clean['Low'], close=df_clean['Close'],
                                 name='K線'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_10'], line=dict(color='orange', width=1), name='EMA 10'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['EMA_200'], line=dict(color='red', width=2), name='EMA 200'), row=1, col=1)

    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['Volume'], marker_color='grey', name='成交量'), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['Volume_MA_20'], line=dict(color='purple', width=1, dash='dot'), name='成交量 MA 20'), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", secondary_y=True, row=1, col=1, showgrid=False)

    fig.add_trace(go.Bar(x=df_clean.index, y=df_clean['MACD_Hist'], marker_color=np.where(df_clean['MACD_Hist'] >= 0, 'green', 'red'), name='MACD Hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD_Line'], line=dict(color='blue', width=1), name='MACD 線'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['MACD_Signal'], line=dict(color='#ffab40', width=1), name='Signal 線'), row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1, fixedrange=True)

    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['RSI'], line=dict(color='#0077b6', width=1.5), name='RSI'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['ADX'], line=dict(color='#800080', width=1.5, dash='dot'), name='ADX'), row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1, fixedrange=True)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)

    fig.add_trace(go.Scatter(x=df_clean.index, y=df_clean['OBV'], line=dict(color='#1e8449', width=1.5), name='OBV'), row=4, col=1)
    fig.update_yaxes(title_text="OBV", row=4, col=1, fixedrange=True)

    fig.update_layout(
        title_text=f"AI 融合分析圖表",
        height=900,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def main():
    st.sidebar.title("🚀 AI 趨勢分析")
    st.sidebar.markdown("---")

    selected_category = st.sidebar.selectbox(
        '1. 選擇資產類別',
        list(CATEGORY_HOT_OPTIONS.keys()),
        index=1
    )

    hot_options_map = CATEGORY_HOT_OPTIONS.get(selected_category, {})
    
    st.sidebar.markdown("---")
    
    selected_hot_option_key = st.sidebar.selectbox(
        '2. 選擇熱門標的 (或手動輸入)',
        list(hot_options_map.keys()),
        index=list(hot_options_map.values()).index('2330.TW') if '2330.TW' in hot_options_map.values() else 0,
        help="從列表中選擇一個熱門標的，或直接在下方框中輸入"
    )

    search_input = st.sidebar.text_input(
        '...或在這裡手動輸入代碼/名稱:',
        value=st.session_state.get('sidebar_search_input', '2330.TW')
    )
    
    st.sidebar.markdown("---")

    period_options = list(PERIOD_MAP.keys())
    selected_period_key = st.sidebar.selectbox(
        '3. 選擇分析週期',
        period_options,
        index=2, # Default to '1 日'
        help="30 分 (短期)、1 日 (中長線)"
    )
    
    st.sidebar.markdown("---")

    analyze_button_clicked = st.sidebar.button('📊 執行AI分析', use_container_width=True)

    if analyze_button_clicked:
        st.session_state['analyze_trigger'] = True
        
        if search_input and search_input.strip():
            st.session_state['sidebar_search_input'] = search_input
            final_symbol_to_analyze = get_symbol_from_query(search_input)
        else:
            final_symbol_to_analyze = hot_options_map.get(selected_hot_option_key, '2330.TW')
            st.session_state['sidebar_search_input'] = final_symbol_to_analyze
            
        st.session_state['last_search_symbol'] = final_symbol_to_analyze
        
        yf_period, yf_interval = PERIOD_MAP[selected_period_key]

        with st.spinner(f"🔍 AI正在啟動，分析 **{final_symbol_to_analyze}** 的多維度數據..."):
            df, msg = get_stock_data(final_symbol_to_analyze, yf_period, yf_interval)
            
            if df is None:
                st.error(f"❌ **數據錯誤:** {msg}")
                st.session_state['data_ready'] = False
            else:
                company_info = get_company_info(final_symbol_to_analyze)
                currency_symbol = get_currency_symbol(final_symbol_to_analyze)
                
                # 執行所有分析
                df = calculate_technical_indicators(df)
                fa_results = calculate_advanced_fundamental_rating(final_symbol_to_analyze)
                chips_news_results = get_chips_and_news_analysis(final_symbol_to_analyze)
                
                report = generate_ai_synthesis_report(df, fa_results, chips_news_results)
                
                st.session_state['analysis_results'] = {
                    'df': df,
                    'company_info': company_info,
                    'currency_symbol': currency_symbol,
                    'fa_results': fa_results,
                    'report': report,
                    'selected_period_key': selected_period_key,
                    'final_symbol_to_analyze': final_symbol_to_analyze
                }
                st.session_state['data_ready'] = True

    if st.session_state.get('data_ready', False):
        results = st.session_state['analysis_results']
        df = results['df'].dropna()
        company_info = results['company_info']
        currency_symbol = results['currency_symbol']
        fa_results = results['fa_results']
        report = results['report']
        selected_period_key = results['selected_period_key']
        final_symbol_to_analyze = results['final_symbol_to_analyze']
        
        st.header(f"📈 **{company_info['name']}** ({final_symbol_to_analyze}) AI趨勢分析")
        
        last_row = df.iloc[-1]
        prev_close = df.iloc[-2]['Close']
        change = last_row['Close'] - prev_close
        change_pct = (change / prev_close) * 100
        
        st.markdown(
            """
            <style>
            [data-testid="stMetricValue"] { font-size: 24px; }
            [data-testid="stMetricLabel"] { font-size: 15px; }
            .action-buy { color: #D32F2F; font-weight: bold; }
            .action-sell { color: #388E3C; font-weight: bold; }
            .action-neutral { color: #F57C00; font-weight: bold; }
            </style>
            """, unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("當前價格", f"{currency_symbol}{last_row['Close']:,.4f}", f"{change:,.2f} ({change_pct:+.2f}%)")
        
        action_class = f"action-{report['final_action'].split(' ')[0].lower()}"
        col2.markdown(f"**🎯 AI綜合建議**\n<p class='{action_class}' style='font-size: 24px;'>{report['final_action']}</p>", unsafe_allow_html=True)
        
        col3.metric("信心指數", f"{report['confidence']}%", help="AI根據多空信號強度計算的信心度")
        
        st.info(f"**📝 AI分析摘要:** {report['summary']}")
        st.markdown("---")

        # *** 新增 ***
        st.subheader("💡 核心交易策略")
        col1, col2, col3 = st.columns(3)
        col1.metric("➡️ 建議進場價", f"~ {currency_symbol}{report['entry_price']:,.2f}")
        col2.metric("🚀 止盈目標", f"~ {currency_symbol}{report['take_profit']:,.2f}")
        col3.metric("🛑 止損參考", f"~ {currency_symbol}{report['stop_loss']:,.2f}")
        st.markdown("---")
        
        # *** 新增 ***
        st.subheader("🔬 四大面向分析詳情")
        
        with st.expander("技術面分析 (Technical)"):
            st.write(report['technical_opinion'])
        
        with st.expander("基本面分析 (Fundamental)"):
            st.write(report['fundamental_opinion'])
            if fa_results['details']:
                for key, value in fa_results['details'].items():
                    st.text(f"  - {key}: {value}")

        with st.expander("籌碼面分析 (Chips)"):
            st.write(report['chips_opinion'])
        
        with st.expander("消息面分析 (News)"):
            st.write(report['news_opinion'])

        st.markdown("---")
        st.subheader(f"📊 完整技術分析圖表")
        chart = create_comprehensive_chart(df, final_symbol_to_analyze, selected_period_key)
        st.plotly_chart(chart, use_container_width=True, key=f"plotly_chart_{final_symbol_to_analyze}")

    elif not st.session_state.get('data_ready', False):
        # *** 修改 ***
        st.markdown("<h1 style='color: #FF8C00; font-size: 32px; font-weight: bold;'>🚀 歡迎使用 AI 趨勢分析</h1>", unsafe_allow_html=True) # 淡橘色色調
        st.markdown(f"請在左側選擇或輸入您想分析的標的（例如：**2330.TW**、**NVDA**、**BTC-USD**），然後點擊 <span style='color: #FF8C00; font-weight: bold;'>『📊 執行AI分析』</span> 按鈕開始。", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("📝 使用步驟：")
        st.markdown("1. **選擇資產類別**：在左側欄選擇 `美股`、`台股` 或 `加密貨幣`。")
        st.markdown("2. **選擇標的**：使用下拉選單快速選擇熱門標的，或直接在輸入框中鍵入代碼或名稱。")
        st.markdown("3. **選擇週期**：決定分析的長度（例如：`30 分` (短期)、`1 日` (中長線)）。")
        st.markdown(f"4. **執行分析**：點擊 <span style='color: #FF8C00; font-weight: bold;'>『📊 執行AI分析』</span>，AI將融合基本面與技術面指標提供交易策略。", unsafe_allow_html=True)
        st.markdown("---")

if __name__ == '__main__':
    if 'data_ready' not in st.session_state: st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state: st.session_state['sidebar_search_input'] = "2330.TW"
        
    main()
    
    # *** 修改 ***
    st.markdown("---")
    st.markdown("⚠️ **免責聲明**")
    st.markdown("本分析模型包含多位AI的量化觀點，但僅供教育與參考用途。投資涉及風險，所有交易決策應基於您個人的獨立研究和財務狀況，並建議諮詢專業金融顧問。")
    st.markdown("📊 **數據來源:** Yahoo Finance | **技術指標:** TA 庫 | **APP優化:** AI程式碼架構師")
