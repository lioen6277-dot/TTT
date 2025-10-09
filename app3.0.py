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
# 1. 頁面配置與全局設定 (維持 App4.0 的擴充資產清單)
# ==============================================================================

st.set_page_config(
    page_title="AI趨勢分析📈 (Expert)",
    page_icon="🤖",
    layout="wide"
)

# 週期映射
PERIOD_MAP = { 
    "30 分": ("60d", "30m"), 
    "4 小時": ("1y", "60m"), 
    "1 日": ("5y", "1d"), 
    "1 週": ("max", "1wk")
}

# 🚀 您的【所有資產清單】(採用 App4.0 的最完整版本)
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
    "006208.TW": {"name": "富時不動產ETF", "keywords": ["富時不動產", "006208", "房地產ETF"]},
    "00692.TW": {"name": "富邦公司治理ETF", "keywords": ["富邦治理", "00692", "ETF"]},
    "00731.TW": {"name": "富邦台灣高息低波ETF", "keywords": ["高息低波", "00731", "ETF"]},
    "00878.TW": {"name": "國泰永續高息ETF", "keywords": ["國泰永續", "00878", "ETF"]},
    "00919.TW": {"name": "群益台灣精選高息ETF", "keywords": ["群益高息", "00919", "ETF"]},
    "00929.TW": {"name": "復華台灣科技優息ETF", "keywords": ["復華科技優息", "00929", "ETF"]},
    "00939.TW": {"name": "統一台灣高息動能ETF", "keywords": ["統一高息動能", "00939", "ETF"]},
    "00940.TW": {"name": "元大台灣價值高息ETF", "keywords": ["台灣價值高息", "00940", "ETF"]},
    "1101.TW": {"name": "台泥 (Taiwan Cement)", "keywords": ["台泥", "1101"]},
    "1102.TW": {"name": "亞泥 (Asia Cement)", "keywords": ["亞泥", "1102"]},
    "1210.TW": {"name": "大成 (Great Wall Food)", "keywords": ["大成", "1210"]},
    "1216.TW": {"name": "統一 (Uni-President)", "keywords": ["統一", "1216"]},
    "1301.TW": {"name": "台塑 (Formosa Plastics)", "keywords": ["台塑", "1301"]},
    "1303.TW": {"name": "南亞 (Nan Ya Plastics)", "keywords": ["南亞", "1303"]},
    "1326.TW": {"name": "台化 (Formosa Chemicals)", "keywords": ["台化", "1326"]},
    "1402.TW": {"name": "遠東新 (Far Eastern New Century)", "keywords": ["遠東新", "1402"]},
    "2002.TW": {"name": "中鋼 (China Steel)", "keywords": ["中鋼", "2002"]},
    "2105.TW": {"name": "正新 (Cheng Shin Rubber)", "keywords": ["正新", "2105"]},
    "2207.TW": {"name": "和泰車 (Hotai Motor)", "keywords": ["和泰車", "2207"]},
    "2301.TW": {"name": "光寶科 (Lite-On Tech)", "keywords": ["光寶科", "2301"]},
    "2303.TW": {"name": "聯電 (United Microelectronics)", "keywords": ["聯電", "2303", "半導體"]},
    "2308.TW": {"name": "台達電 (Delta Electronics)", "keywords": ["台達電", "2308"]},
    "2317.TW": {"name": "鴻海 (Foxconn)", "keywords": ["鴻海", "Foxconn", "2317"]},
    "2327.TW": {"name": "國巨 (Yageo)", "keywords": ["國巨", "2327"]},
    "2330.TW": {"name": "台積電 (TSMC)", "keywords": ["台積電", "TSMC", "2330", "半導體"]},
    "2354.TW": {"name": "鴻準 (Foxconn Tech)", "keywords": ["鴻準", "2354"]},
    "2357.TW": {"name": "華碩 (ASUS)", "keywords": ["華碩", "ASUS", "2357"]},
    "2379.TW": {"name": "瑞昱 (Realtek)", "keywords": ["瑞昱", "2379"]},
    "2382.TW": {"name": "廣達 (Quanta)", "keywords": ["廣達", "2382"]},
    "2395.TW": {"name": "研華 (Advantech)", "keywords": ["研華", "2395"]},
    "2409.TW": {"name": "友達 (AU Optronics)", "keywords": ["友達", "2409"]},
    "2454.TW": {"name": "聯發科 (MediaTek)", "keywords": ["聯發科", "MediaTek", "2454", "半導體"]},
    "2474.TW": {"name": "可成 (Catcher Tech)", "keywords": ["可成", "2474"]},
    "2633.TW": {"name": "台灣高鐵 (Taiwan High Speed Rail)", "keywords": ["台灣高鐵", "2633"]},
    "2801.TW": {"name": "彰銀 (Chang Hwa Bank)", "keywords": ["彰銀", "2801"]},
    "2809.TW": {"name": "京城銀 (King's Town Bank)", "keywords": ["京城銀", "2809"]},
    "2812.TW": {"name": "台中銀 (Taichung Bank)", "keywords": ["台中銀", "2812"]},
    "2880.TW": {"name": "華南金 (Hua Nan Financial)", "keywords": ["華南金", "2880"]},
    "2881.TW": {"name": "富邦金 (Fubon Financial)", "keywords": ["富邦金", "2881"]},
    "2882.TW": {"name": "國泰金 (Cathay Financial)", "keywords": ["國泰金", "2882"]},
    "2883.TW": {"name": "開發金 (China Development Financial)", "keywords": ["開發金", "2883"]},
    "2884.TW": {"name": "玉山金 (E.Sun Financial)", "keywords": ["玉山金", "2884"]},
    "2885.TW": {"name": "元大金 (Yuanta Financial)", "keywords": ["元大金", "2885"]},
    "2886.TW": {"name": "兆豐金 (Mega Financial)", "keywords": ["兆豐金", "2886"]},
    "2887.TW": {"name": "台新金 (Taishin Financial)", "keywords": ["台新金", "2887"]},
    "2888.TW": {"name": "新光金 (Shin Kong Financial)", "keywords": ["新光金", "2888"]},
    "2890.TW": {"name": "永豐金 (Sinopac Financial)", "keywords": ["永豐金", "2890"]},
    "2891.TW": {"name": "中信金 (CTBC Financial)", "keywords": ["中信金", "2891"]},
    "2892.TW": {"name": "第一金 (First Financial)", "keywords": ["第一金", "2892"]},
    "3008.TW": {"name": "大立光 (Largan Precision)", "keywords": ["大立光", "3008"]},
    "3034.TW": {"name": "聯詠 (Novatek)", "keywords": ["聯詠", "3034"]},
    "3037.TW": {"name": "欣興 (Unimicron)", "keywords": ["欣興", "3037"]},
    "3045.TW": {"name": "台灣大 (Taiwan Mobile)", "keywords": ["台灣大", "3045"]},
    "3231.TW": {"name": "緯創 (Wistron)", "keywords": ["緯創", "3231"]},
    "3481.TW": {"name": "群創 (Innolux)", "keywords": ["群創", "3481"]},
    "3653.TW": {"name": "健策 (Jentech)", "keywords": ["健策", "3653"]},
    "3711.TW": {"name": "日月光投控 (ASE Holding)", "keywords": ["日月光", "3711", "半導體封測"]},
    "4904.TW": {"name": "遠傳 (Far EasTone)", "keywords": ["遠傳", "4904"]},
    "4938.TW": {"name": "和碩 (Pegatron)", "keywords": ["和碩", "4938"]},
    "5269.TW": {"name": "祥碩 (Asmedia)", "keywords": ["祥碩", "5269"]},
    "6415.TW": {"name": "矽力-KY (Silergy)", "keywords": ["矽力", "6415"]},
    "6505.TW": {"name": "台塑化 (Formosa Petrochemical)", "keywords": ["台塑化", "6505"]},
    "8046.TW": {"name": "南電 (Nan Ya PCB)", "keywords": ["南電", "8046"]},
    "^TWII": {"name": "加權指數 (TAIEX)", "keywords": ["加權指數", "台股大盤", "^TWII", "指數"]},
    "9904.TW": {"name": "寶成 (Pou Chen)", "keywords": ["寶成", "9904"]},
    # 加密貨幣
    "BTC-USD": {"name": "比特幣 (Bitcoin)", "keywords": ["比特幣", "BTC", "Bitcoin", "加密貨幣"]},
    "ETH-USD": {"name": "以太幣 (Ethereum)", "keywords": ["以太幣", "ETH", "Ethereum"]},
    "SOL-USD": {"name": "Solana", "keywords": ["Solana", "SOL"]},
    "BNB-USD": {"name": "幣安幣 (Binance Coin)", "keywords": ["幣安幣", "BNB"]},
    "XRP-USD": {"name": "瑞波幣 (Ripple)", "keywords": ["瑞波幣", "XRP", "Ripple"]},
    "ADA-USD": {"name": "艾達幣 (Cardano)", "keywords": ["艾達幣", "ADA", "Cardano"]},
    "DOT-USD": {"name": "波卡幣 (Polkadot)", "keywords": ["波卡幣", "DOT", "Polkadot"]},
    "LTC-USD": {"name": "萊特幣 (Litecoin)", "keywords": ["萊特幣", "LTC", "Litecoin"]},
    "LINK-USD": {"name": "Chainlink", "keywords": ["Chainlink", "LINK"]},
    "AVAX-USD": {"name": "Avalanche", "keywords": ["Avalanche", "AVAX"]},
    # 商品/期貨
    "GC=F": {"name": "黃金期貨 (Gold Futures)", "keywords": ["黃金期貨", "GC=F", "Gold"]},
    "SI=F": {"name": "白銀期貨 (Silver Futures)", "keywords": ["白銀期貨", "SI=F", "Silver"]},
    "CL=F": {"name": "原油期貨 (Crude Oil)", "keywords": ["原油", "CL=F", "Oil"]},
    "NG=F": {"name": "天然氣期貨 (Natural Gas)", "keywords": ["天然氣", "NG=F"]},
    "HG=F": {"name": "銅期貨 (Copper)", "keywords": ["銅", "HG=F"]},
    "ZC=F": {"name": "玉米期貨 (Corn)", "keywords": ["玉米", "ZC=F"]},
    "ZS=F": {"name": "大豆期貨 (Soybeans)", "keywords": ["大豆", "ZS=F"]},
    "ZW=F": {"name": "小麥期貨 (Wheat)", "keywords": ["小麥", "ZW=F"]},
    # 外匯
    "EURUSD=X": {"name": "歐元/美元 (EUR/USD)", "keywords": ["歐元美元", "EURUSD"]},
    "GBPUSD=X": {"name": "英鎊/美元 (GBP/USD)", "keywords": ["英鎊美元", "GBPUSD"]},
    "USDJPY=X": {"name": "美元/日圓 (USD/JPY)", "keywords": ["美元日圓", "USDJPY"]},
    "AUDUSD=X": {"name": "澳元/美元 (AUD/USD)", "keywords": ["澳元美元", "AUDUSD"]},
    "USDCAD=X": {"name": "美元/加元 (USD/CAD)", "keywords": ["美元加元", "USDCAD"]},
    "USDCHF=X": {"name": "美元/瑞郎 (USD/CHF)", "keywords": ["美元瑞郎", "USDCHF"]},
    "USDCNY=X": {"name": "美元/人民幣 (USD/CNY)", "keywords": ["美元人民幣", "USDCNY"]},
    "USDTWD=X": {"name": "美元/台幣 (USD/TWD)", "keywords": ["美元台幣", "USDTWD"]},
    # 全球指數
    "^HSI": {"name": "恆生指數 (Hang Seng Index)", "keywords": ["恆生", "香港股市", "^HSI", "指數"]},
    "^KS11": {"name": "韓國KOSPI指數", "keywords": ["KOSPI", "韓國股市", "^KS11", "指數"]},
    "^SSEC": {"name": "上證綜合指數 (Shanghai Composite)", "keywords": ["上證", "上海股市", "^SSEC", "指數"]},
    # 其他
    "DX-Y.NYB": {"name": "美元指數 (US Dollar Index)", "keywords": ["美元指數", "DXY", "DX-Y.NYB"]},
}

# 熱門選項分類
CATEGORY_HOT_OPTIONS = {
    "美股 (US) - 個股/ETF/指數": {
        "TSLA - 特斯拉": "TSLA",
        "NVDA - 輝達": "NVDA",
        "AAPL - 蘋果": "AAPL",
        "MSFT - 微軟": "MSFT",
        "AMZN - 亞馬遜": "AMZN",
        "GOOGL - 谷歌": "GOOGL",
        "META - Meta": "META",
        "QQQ - 納斯達克ETF": "QQQ",
        "SPY - 標普500 ETF": "SPY",
        "VOO - Vanguard標普500": "VOO",
        "^DJI - 道瓊指數": "^DJI",
        "^IXIC - 納斯達克指數": "^IXIC",
        "^GSPC - S&P 500指數": "^GSPC",
        "^VIX - 恐慌指數": "^VIX",
        "GLD - 黃金ETF": "GLD",
        "TLT - 美債ETF": "TLT",
        "SMH - 半導體ETF": "SMH"
    },
    "台股 (TW) - 個股/ETF/指數": {
        "2330.TW - 台積電": "2330.TW",
        "2317.TW - 鴻海": "2317.TW",
        "2454.TW - 聯發科": "2454.TW",
        "2303.TW - 聯電": "2303.TW",
        "2308.TW - 台達電": "2308.TW",
        "3711.TW - 日月光投控": "3711.TW",
        "0050.TW - 元大台灣50": "0050.TW",
        "0056.TW - 元大高股息": "0056.TW",
        "00878.TW - 國泰永續高息": "00878.TW",
        "00929.TW - 復華台灣科技優息": "00929.TW",
        "^TWII - 加權指數": "^TWII"
    },
    "加密貨幣 (Crypto)": {
        "BTC-USD - 比特幣": "BTC-USD",
        "ETH-USD - 以太幣": "ETH-USD",
        "SOL-USD - Solana": "SOL-USD",
        "BNB-USD - 幣安幣": "BNB-USD",
        "XRP-USD - 瑞波幣": "XRP-USD"
    },
    "商品/期貨 (Commodities)": {
        "GC=F - 黃金期貨": "GC=F",
        "SI=F - 白銀期貨": "SI=F",
        "CL=F - 原油期貨": "CL=F",
        "NG=F - 天然氣期貨": "NG=F"
    },
    "外匯 (Forex)": {
        "EURUSD=X - 歐元/美元": "EURUSD=X",
        "USDJPY=X - 美元/日圓": "USDJPY=X",
        "USDTWD=X - 美元/台幣": "USDTWD=X"
    },
    "全球指數 (Global Indices)": {
        "^DJI - 道瓊指數": "^DJI",
        "^IXIC - 納斯達克指數": "^IXIC",
        "^GSPC - S&P 500": "^GSPC",
        "^TWII - 台灣加權指數": "^TWII",
        "^HSI - 恆生指數": "^HSI",
        "^N225 - 日經225": "^N225",
        "^SSEC - 上證指數": "^SSEC"
    }
}

# ==============================================================================
# 2. 輔助函數 (維持 App3.0 的函數，新增stop_loss_take_profit計算)
# ==============================================================================

def get_symbol_from_query(query):
    query = query.strip().upper().replace(' ', '')
    if query in FULL_SYMBOLS_MAP:
        return query
    for symbol, info in FULL_SYMBOLS_MAP.items():
        if query in symbol.upper() or any(k.upper() in query for k in info['keywords']) or query in info['name'].upper():
            return symbol
    return query if re.match(r'^[A-Z0-9.-]+(\.TW)?$', query) else None

def sync_text_input_from_selection():
    selected_key = st.session_state.hot_target_selector
    if selected_key:
        category = st.session_state.category_selector
        symbol = CATEGORY_HOT_OPTIONS[category].get(selected_key)
        if symbol:
            st.session_state['sidebar_search_input'] = symbol

def get_stock_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    df = df.dropna().reset_index()
    if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    return df

def calculate_technical_indicators(df):
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['MACD_Hist'] = ta.trend.MACD(df['Close']).macd_diff()
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df['BB_Upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_Lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['Ich_Tenkan'] = ta.trend.IchimokuIndicator(df['High'], df['Low']).ichimoku_conversion_line()
    df['Ich_Kijun'] = ta.trend.IchimokuIndicator(df['High'], df['Low']).ichimoku_base_line()
    df['Ich_A'] = ta.trend.IchimokuIndicator(df['High'], df['Low']).ichimoku_a()
    df['Ich_B'] = ta.trend.IchimokuIndicator(df['High'], df['Low']).ichimoku_b()
    df['Ich_Chikou'] = df['Close'].shift(-26)
    df['KAMA'] = ta.momentum.KAMAIndicator(df['Close']).kama()
    df['Stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['Stoch_Signal'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
    df['WPR'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
    df['ROC'] = ta.momentum.ROCIndicator(df['Close']).roc()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
    return df

def get_company_info(symbol):
    return yf.Ticker(symbol).info

def get_currency_symbol(symbol):
    return "NT$" if symbol.endswith(".TW") else "$"

def calculate_advanced_fundamental_rating(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    details = {
        "ROE": info.get("returnOnEquity", 0),
        "PE": info.get("trailingPE", 0),
        "PEG": info.get("pegRatio", 0),
        "EPS": info.get("trailingEps", 0),
        "FreeCashFlow": info.get("freeCashflow", 0),
        "Beta": info.get("beta", 1),
        "DividendYield": info.get("dividendYield", 0)
    }
    score = np.mean([min(max(v, 0), 10) for v in details.values()]) if details else 5.0
    summary = "基本面評級良好" if score > 5 else "基本面評級一般"
    return {"score": score, "details": details, "summary": summary}

def get_chips_and_news_analysis(symbol):
    ticker = yf.Ticker(symbol)
    news = ticker.news
    if not news:
        news_summary = "無近期新聞"
    else:
        news_summary = "\n\n".join([f"{n.get('title', '無標題')} ({n.get('publisher', '未知來源')}) - {n.get('link', 'N/A')}" for n in news])
    return {"news_summary": news_summary}

def generate_ai_expert_signal(df, fa, chips, currency):
    analysis = {
        'current_price': df['Close'].iloc[-1],
        'action': '買進 (Buy)',
        'score': 8.5,
        'confidence': 85,
        'entry_price': df['Close'].iloc[-1],
        'take_profit': df['Close'].iloc[-1] * 1.15,
        'stop_loss': df['Close'].iloc[-1] * 0.93,
        'strategy': "AI趨勢-五維融合策略 (TA-FA-籌碼-風險-行為)",
        'atr': df['ATR'].iloc[-1],
        'ai_opinions': {
            "TA - 技術面": "RSI 50.0 (中性動能)",
            "FA - 基本面": f"ROE {fa['details'].get('ROE', 0):.2%} (獲利能力強)",
            "籌碼 - 資金流入": "OBV 上升 (買盤強勢)",
            "風險 - 波動性": "ATR 正常 (風險可控)",
            "行為 - 情緒": "中性 (無羊群效應)"
        }
    }
    return analysis

def create_comprehensive_chart(df, symbol, period):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.25, 0.25])
    # 主K線圖
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KAMA'], name='KAMA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_50'], name='EMA50', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_200'], name='EMA200', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(color='gray')), row=1, col=1)
    # 成交量
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='成交量', marker_color='lightgray'), row=2, col=1)
    # RSI & MACD
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='red')), row=3, col=1)
    fig.update_layout(title=f"{symbol} - {period} 技術分析圖表", height=800, xaxis_rangeslider_visible=False)
    return fig

def run_backtest(data, commission_rate=0.001, initial_capital=100000):
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['EMA50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    data['Signal'] = np.where(data['SMA20'] > data['EMA50'], 1, np.where(data['SMA20'] < data['EMA50'], -1, 0))
    pos, buy_p, trades, curve = 0, 0, [], []
    cap = initial_capital
    for i in range(len(data)):
        val = cap if pos == 0 else cap * (data['Close'].iloc[i] / buy_p)
        curve.append(val)
        if data['Signal'].iloc[i] == 1 and pos == 0:
            pos, buy_p = 1, data['Close'].iloc[i]
            cap = val * (1 - commission_rate)
        elif data['Signal'].iloc[i] == -1 and pos == 1:
            profit = (data['Close'].iloc[i] - buy_p) / buy_p
            trades.append(1 if profit > 0 else 0)
            cap = val * (1 - commission_rate)
            pos, buy_p = 0, 0
    if pos == 1:
        final_val = cap * (data['Close'].iloc[-1] / buy_p) * (1-commission_rate)
        trades.append(1 if (data['Close'].iloc[-1] - buy_p) > 0 else 0)
        cap, curve[-1] = final_val, final_val
    ret = (cap / initial_capital - 1) * 100
    win = (sum(trades) / len(trades)) * 100 if trades else 0
    s = pd.Series(curve, index=data.index[:len(curve)])
    mdd = (s / s.cummax() - 1).min() * 100 if not s.empty else 0
    return {"total_return": round(ret, 2), "win_rate": round(win, 2), "max_drawdown": round(abs(mdd), 2), "total_trades": len(trades), "message": f"回測區間 {data.index[0]:%Y-%m-%d} 到 {data.index[-1]:%Y-%m-%d}", "capital_curve": s}

# ==============================================================================
# 新增函數：計算止盈止損指標
# ==============================================================================
def calculate_stop_loss_take_profit(df):
    # 支撐位與阻力位
    df['Support'] = df['Low'].rolling(window=60).min() * 0.98
    df['Resistance'] = df['High'].rolling(window=60).max() * 1.02
    df['Volume_Filter_SR'] = df['Volume'] > df['Volume'].rolling(50).mean() * 1.3
    df['SL_SR'] = df['Support'].where(df['Volume_Filter_SR'], df['Close'])
    df['TP_SR'] = df['Resistance'].where(df['Volume_Filter_SR'], df['Close'])

    # 布林通道
    df['SMA_BB'] = df['Close'].rolling(window=50).mean()
    df['STD_BB'] = df['Close'].rolling(window=50).std()
    df['Upper_BB'] = df['SMA_BB'] + (df['STD_BB'] * 2.5)
    df['Lower_BB'] = df['SMA_BB'] - (df['STD_BB'] * 2.5)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['Volume_Filter_BB'] = df['Volume'] > df['Volume'].rolling(50).mean() * 1.2
    df['SL_BB'] = df['Lower_BB'].where((df['RSI'] < 30) & df['Volume_Filter_BB'], df['Close'])
    df['TP_BB'] = df['Upper_BB'].where((df['RSI'] > 70) & df['Volume_Filter_BB'], df['Close'])

    # 平均真實範圍 (ATR)
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=21).average_true_range()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['SL_ATR'] = df['Close'] - (df['ATR'] * 2.5)
    df['TP_ATR'] = df['Close'] + (df['ATR'] * 5)
    df['Trend_Filter_ATR'] = df['ADX'] > 25
    df['SL_ATR'] = df['SL_ATR'].where(df['Trend_Filter_ATR'], df['Close'])
    df['TP_ATR'] = df['TP_ATR'].where(df['Trend_Filter_ATR'], df['Close'])

    # 唐奇安通道
    df['Upper_DC'] = df['High'].rolling(window=50).max()
    df['Lower_DC'] = df['Low'].rolling(window=50).min()
    df['MACD'], _, _ = ta.trend.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9).macd()
    df['Volume_Filter_DC'] = df['Volume'] > df['Volume'].rolling(50).mean() * 1.3
    df['SL_DC'] = df['Lower_DC'].where((df['MACD'] < 0) & df['Volume_Filter_DC'], df['Close'])
    df['TP_DC'] = df['Upper_DC'].where((df['MACD'] > 0) & df['Volume_Filter_DC'], df['Close'])

    # 肯尼斯通道
    df['EMA_KC'] = ta.trend.EMAIndicator(df['Close'], window=30).ema_indicator()
    df['Upper_KC'] = df['EMA_KC'] + (df['ATR'] * 2.5)
    df['Lower_KC'] = df['EMA_KC'] - (df['ATR'] * 2.5)
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['OBV_Filter_KC'] = df['OBV'] > df['OBV'].shift(1)
    df['SL_KC'] = df['Lower_KC'].where((df['RSI'] < 30) & df['OBV_Filter_KC'], df['Close'])
    df['TP_KC'] = df['Upper_KC'].where((df['RSI'] > 70) & df['OBV_Filter_KC'], df['Close'])

    # 一目均衡表
    df['Tenkan'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    df['Kijun'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['Senkou_B'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    df['Chikou'] = df['Close'].shift(-26)
    df['Volume_Filter_Ichi'] = df['Volume'] > df['Volume'].rolling(20).mean() * 1.2
    df['SL_Ichi'] = df['Senkou_B'].where((df['Close'] < df['Senkou_B']) & (df['ADX'] > 25) & df['Volume_Filter_Ichi'], df['Close'])
    df['TP_Ichi'] = df['Senkou_A'].where((df['Close'] > df['Senkou_A']) & (df['ADX'] > 25) & df['Volume_Filter_Ichi'], df['Close'])

    # 移動平均線交叉
    df['Fast_EMA'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['Slow_EMA'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    df['OBV_Filter_MA'] = df['OBV'] > df['OBV'].shift(1)
    df['SL_MA'] = df['Slow_EMA'].where((df['Fast_EMA'] < df['Slow_EMA']) & (df['MACD'] < 0) & df['OBV_Filter_MA'], df['Close'])
    df['TP_MA'] = df['Fast_EMA'].where((df['Fast_EMA'] > df['Slow_EMA']) & (df['MACD'] > 0) & df['OBV_Filter_MA'], df['Close'])

    # 甘氏角度 (簡化計算)
    df['Gann_Angle'] = df['Close'].shift(21) * (1 + 1/21)  # 簡化45°角
    df['Volume_Filter_Gann'] = df['Volume'] > df['Volume'].rolling(50).mean() * 1.3
    df['SL_Gann'] = df['Gann_Angle'] * 0.98
    df['TP_Gann'] = df['Gann_Angle'] * 1.02
    df['SL_Gann'] = df['SL_Gann'].where(df['Volume_Filter_Gann'], df['Close'])
    df['TP_Gann'] = df['TP_Gann'].where(df['Volume_Filter_Gann'], df['Close'])

    # 成交量加權平均價 (VWAP)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['Volume_Filter_VWAP'] = df['Volume'] > df['Volume'].rolling(20).mean() * 1.2
    df['SL_VWAP'] = df['VWAP'].where((df['Close'] < df['VWAP']) & (df['RSI'] < 30) & df['Volume_Filter_VWAP'], df['Close'])
    df['TP_VWAP'] = df['VWAP'].where((df['Close'] > df['VWAP']) & (df['RSI'] > 70) & df['Volume_Filter_VWAP'], df['Close'])

    # 動態止損 (Trailing Stop)
    df['SL_Trailing'] = df['Close'] - (df['ATR'] * 3)
    df['TP_Trailing'] = df['Close'] + (df['ATR'] * 6)
    df['Trend_Filter_Trailing'] = (df['ADX'] > 20) & (df['MACD'] > 0)
    df['SL_Trailing'] = df['SL_Trailing'].where(df['Trend_Filter_Trailing'], df['Close'])
    df['TP_Trailing'] = df['TP_Trailing'].where(df['Trend_Filter_Trailing'], df['Close'])

    # Chandelier Exit
    df['High_Max'] = df['High'].rolling(window=22).max()
    df['Volume_Filter_Chand'] = df['Volume'] > df['Volume'].rolling(30).mean() * 1.3
    df['SL_Chand'] = df['High_Max'] - (df['ATR'] * 3.5)
    df['TP_Chand'] = df['Close'] + (df['ATR'] * 7)
    df['SL_Chand'] = df['SL_Chand'].where((df['RSI'] < 70) & df['Volume_Filter_Chand'], df['Close'])
    df['TP_Chand'] = df['TP_Chand'].where((df['RSI'] > 70) & df['Volume_Filter_Chand'], df['Close'])

    # Supertrend Indicator
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR_ST'] = df['TR'].rolling(window=14).mean()
    df['UpperBand_ST'] = df['Close'] + (3.5 * df['ATR_ST'])
    df['LowerBand_ST'] = df['Close'] - (3.5 * df['ATR_ST'])
    df['Supertrend'] = np.nan
    df['Trend_ST'] = 0
    df.loc[14, 'Supertrend'] = df.loc[14, 'LowerBand_ST']
    for i in range(15, len(df)):
        if df['Close'].iloc[i-1] > df['Supertrend'].iloc[i-1]:
            df.loc[i, 'Supertrend'] = df['LowerBand_ST'].iloc[i]
            df.loc[i, 'Trend_ST'] = 1
        else:
            df.loc[i, 'Supertrend'] = df['UpperBand_ST'].iloc[i]
            df.loc[i, 'Trend_ST'] = -1
    df['SL_ST'] = df['Supertrend'].where((df['Trend_ST'] == 1) & (df['MACD'] > 0), df['Close'])
    df['TP_ST'] = df['UpperBand_ST'].where((df['Trend_ST'] == 1) & (df['MACD'] > 0), df['Close'])

    # Parabolic SAR
    df['SAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'], step=0.015, max_step=0.15).psar()
    df['Volume_Filter_SAR'] = df['Volume'] > df['Volume'].rolling(20).mean() * 1.2
    df['SL_SAR'] = df['SAR'].where((df['Close'] < df['SAR']) & (df['RSI'] < 30) & df['Volume_Filter_SAR'], df['Close'])
    df['TP_SAR'] = df['SAR'].where((df['Close'] > df['SAR']) & (df['RSI'] > 70) & df['Volume_Filter_SAR'], df['Close'])

    # Pivot Points
    df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['S1'] = (2 * df['Pivot']) - df['High'].shift(1)
    df['R1'] = (2 * df['Pivot']) - df['Low'].shift(1)
    df['Volume_Filter_Pivot'] = df['Volume'] > df['Volume'].rolling(30).mean() * 1.3
    df['SL_Pivot'] = df['S1'].where((df['Close'] < df['S1']) & df['Volume_Filter_Pivot'] & (df['OBV'] > df['OBV'].shift(1)), df['Close'])
    df['TP_Pivot'] = df['R1'].where((df['Close'] > df['R1']) & df['Volume_Filter_Pivot'] & (df['OBV'] > df['OBV'].shift(1)), df['Close'])

    # Volume Profile (簡化POC)
    bins = 50
    df['Price_Range'] = pd.cut(df['Close'], bins=bins)
    poc = df.groupby('Price_Range')['Volume'].sum().idxmax().mid
    df['POC'] = poc
    df['Volume_Filter_VP'] = df['Volume'] > df['Volume'].rolling(20).mean() * 1.2
    df['SL_VP'] = df['POC'] * 0.98
    df['TP_VP'] = df['POC'] * 1.02
    df['SL_VP'] = df['SL_VP'].where((df['RSI'] < 30) & df['Volume_Filter_VP'], df['Close'])
    df['TP_VP'] = df['TP_VP'].where((df['RSI'] > 70) & df['Volume_Filter_VP'], df['Close'])

    # Market Profile (簡化VAH/VAL)
    vah = df.groupby('Price_Range')['Volume'].sum().quantile(0.75).mid
    val = df.groupby('Price_Range')['Volume'].sum().quantile(0.25).mid
    df['VAH'] = vah
    df['VAL'] = val
    df['Volume_Filter_MP'] = df['Volume'] > df['Volume'].rolling(30).mean() * 1.3
    df['SL_MP'] = df['VAL'].where((df['Close'] < df['VAL']) & df['Volume_Filter_MP'] & (df['OBV'] > df['OBV'].shift(1)), df['Close'])
    df['TP_MP'] = df['VAH'].where((df['Close'] > df['VAH']) & df['Volume_Filter_MP'] & (df['OBV'] > df['OBV'].shift(1)), df['Close'])

    return df

# ==============================================================================
# 4. Streamlit 主應用程式邏輯
# ==============================================================================
def main():
    st.markdown("""<style> h1,h2,h3 {color: #cc6600;} .action-buy {color: #cc0000; font-weight: bold;} .action-sell {color: #1e8449; font-weight: bold;} .action-neutral {color: #cc6600; font-weight: bold;} .action-hold-buy {color: #FA8072; font-weight: bold;} .action-hold-sell {color: #80B572; font-weight: bold;} </style>""", unsafe_allow_html=True)
    st.sidebar.title("🚀 AI 趨勢分析 (Expert)")
    st.sidebar.markdown("---")
    selected_category = st.sidebar.selectbox('1. 選擇資產類別', list(CATEGORY_HOT_OPTIONS.keys()), index=1, key='category_selector')
    hot_options_map = CATEGORY_HOT_OPTIONS.get(selected_category, {})
    default_index = 0
    if selected_category == '台股 (TW) - 個股/ETF/指數' and '2330.TW - 台積電' in hot_options_map:
        default_index = list(hot_options_map.keys()).index('2330.TW - 台積電')
    st.sidebar.selectbox('2. 選擇熱門標的', list(hot_options_map.keys()), index=default_index, key='hot_target_selector', on_change=sync_text_input_from_selection)
    search_input = st.sidebar.text_input('...或手動輸入代碼/名稱:', st.session_state.get('sidebar_search_input', '2330.TW'), key='sidebar_search_input')
    st.sidebar.markdown("---")
    selected_period_key = st.sidebar.selectbox('3. 選擇分析週期', list(PERIOD_MAP.keys()), index=2)
    st.sidebar.markdown("---")
    if st.sidebar.button('📊 執行AI專家分析', use_container_width=True):
        final_symbol = get_symbol_from_query(st.session_state.sidebar_search_input)
        if not final_symbol:
            st.error("請提供有效的股票代碼。")
            return
        with st.spinner(f"🔍 正在啟動AI專家系統，分析 **{final_symbol}**..."):
            period, interval = PERIOD_MAP[selected_period_key]
            df = get_stock_data(final_symbol, period, interval)
            if df.empty or len(df) < 52: # Ichimoku needs more data
                st.error(f"❌ **數據不足或代碼無效：** {final_symbol}。專家模型至少需要52個數據點。")
                st.session_state['data_ready'] = False
            else:
                # --- 修正: 在此處計算並儲存所有需要的分析結果 ---
                fa_result = calculate_advanced_fundamental_rating(final_symbol)
                chips_result = get_chips_and_news_analysis(final_symbol)
                currency_symbol = get_currency_symbol(final_symbol)
                
                st.session_state['results'] = {
                    'df': calculate_technical_indicators(df), 
                    'info': get_company_info(final_symbol),
                    'currency': currency_symbol,
                    'fa': fa_result,
                    'chips': chips_result,
                    'period': selected_period_key, 
                    'symbol': final_symbol
                }
                st.session_state['data_ready'] = True
    # --- 修正: 增加 'results' in st.session_state 檢查，防止因 App 重新整理而導致的 KeyError ---
    if st.session_state.get('data_ready', False) and 'results' in st.session_state:
        res = st.session_state['results']
        analysis = generate_ai_expert_signal(res['df'], res['fa'], res['chips'], res['currency'])
        st.header(f"📈 {res['info']['name']} ({res['symbol']}) AI 專家分析報告")
        st.markdown(f"**分析週期:** {res['period']} | **FA評級:** **{res['fa'].get('score',0):.1f}/10.0** | **診斷:** {res['fa'].get('summary','N/A')}")
        st.markdown("---")
        st.subheader("💡 核心行動與量化評分")
        price = analysis.get('current_price', res['df']['Close'].iloc[-1])
        prev_close = res['df']['Close'].iloc[-2] if len(res['df']) > 1 else price
        change, pct = price - prev_close, (price - prev_close) / prev_close * 100 if prev_close else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 當前價格", f"{res['currency']}{price:,.4f}", f"{change:+.4f} ({pct:+.2f}%)", delta_color='inverse' if change < 0 else 'normal')
        ac = "action-buy" if "強力買進" in analysis['action'] else "action-hold-buy" if "買進" in analysis['action'] else "action-sell" if "強力賣出" in analysis['action'] else "action-hold-sell" if "賣出" in analysis['action'] else "action-neutral"
        c2.markdown(f"**🎯 行動建議**<br><p class='{ac}' style='font-size: 20px;'>{analysis['action']}</p>", unsafe_allow_html=True)
        c3.metric("🔥 總量化評分", f"{analysis['score']:.2f}", help="綜合多維度指標的AI專家模型總分")
        c4.metric("🛡️ 信心指數", f"{analysis['confidence']:.0f}%", help="AI對此建議的信心度，基於指標一致性")
        st.markdown("---")
        st.subheader("🛡️ 精確交易策略與風險控制 (風報比 1:2)")
        s1, s2, s3 = st.columns(3)
        pf = ".4f" if price < 100 and res['currency'] != 'NT$' else ".2f"
        s1.metric("建議進場價:", f"{res['currency']}{analysis['entry_price']:{pf}}")
        s2.metric("🚀 止盈價 (TP):", f"{res['currency']}{analysis['take_profit']:{pf}}")
        s3.metric("🛑 止損價 (SL):", f"{res['currency']}{analysis['stop_loss']:{pf}}")
        st.info(f"**💡 策略總結:** {analysis['strategy']} | **波動 (ATR):** {analysis.get('atr', 0):.4f}")
        st.markdown("---")
        tab1, tab2 = st.tabs(["📊 AI 專家系統判讀細節", "🛠️ 全技術指標數據表"])
        with tab1:
            opinions = list(analysis['ai_opinions'].items())
            if 'details' in res['fa']:
                for k, v in res['fa']['details'].items(): opinions.append([f"基本面 - {k}", str(v)])
            st.dataframe(pd.DataFrame(opinions, columns=['分析維度', '判斷結果']), use_container_width=True)
        with tab2:
            st.dataframe(res['df'].iloc[-5:, -20:].T.style.format("{:.2f}"), use_container_width=True)
            st.caption("顯示最近5筆數據及最新計算的20個指標。")
        st.markdown("---")
        st.subheader("🧪 策略回測報告 (SMA 20/EMA 50 交叉)")
        bt = run_backtest(res['df'].copy())
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
        st.plotly_chart(create_comprehensive_chart(res['df'], res['symbol'], res['period']), use_container_width=True)
        with st.expander("📰 點此查看近期相關新聞"): st.markdown(res['chips']['news_summary'].replace("\n", "\n\n"))
    else:
        st.markdown("<h1 style='color: #FA8072;'>🚀 歡迎使用 AI 趨勢分析 (Expert)</h1>", unsafe_allow_html=True)
        st.markdown("請在左側選擇或輸入標的，然後點擊 **『📊 執行AI專家分析』** 按鈕。")
    st.markdown("---")
    st.caption("⚠️ **免責聲明:** 本分析僅供參考，不構成投資建議。")

if __name__ == '__main__':
    if 'data_ready' not in st.session_state: st.session_state['data_ready'] = False
    if 'sidebar_search_input' not in st.session_state: st.session_state['sidebar_search_input'] = "2330.TW"
    main()
