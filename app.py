import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np

# --- 1. 頁面設定 (極致黑金風格) ---
st.set_page_config(page_title="Felix 華爾街操盤室", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    h1 { color: #d4af37; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .stMetric { background-color: #1a1a1a; border: 1px solid #333; padding: 15px; border-radius: 5px; }
    .stMetric label { color: #888; }
    .stMetric div[data-testid="stMetricValue"] { color: #fff; font-weight: bold; }
    .stButton>button { 
        width: 100%; 
        background: linear-gradient(90deg, #d4af37 0%, #f2c94c 100%); 
        color: black; font-weight: 900; border: none; height: 55px; font-size: 18px;
    }
    .stButton>button:hover { box-shadow: 0 0 15px #d4af37; color: white; }
    .stInfo, .stSuccess, .stError { border-radius: 0px; border-left: 5px solid #d4af37; background-color: #111; color: #ddd; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 超級股票字典 (名稱 <-> 代碼 互轉) ---
# 這是你要求的「輸入名稱出代碼，輸入數字出名稱」的核心
STOCK_MAPPING = {
    # 港股 (HK)
    "騰訊": "0700.HK", "TENCENT": "0700.HK", "700": "0700.HK",
    "阿里": "9988.HK", "BABA HK": "9988.HK", "9988": "9988.HK", "ALIBABA": "9988.HK",
    "美團": "3690.HK", "MEITUAN": "3690.HK", "3690": "3690.HK",
    "小米": "1810.HK", "XIAOMI": "1810.HK", "1810": "1810.HK",
    "比亞迪": "1211.HK", "BYD": "1211.HK", "1211": "1211.HK",
    "匯豐": "0005.HK", "HSBC": "0005.HK", "5": "0005.HK", "0005": "0005.HK",
    "港交所": "0388.HK", "HKEX": "0388.HK", "388": "0388.HK",
    "快手": "1024.HK", "KUAISHOU": "1024.HK", "1024": "1024.HK",
    "中芯": "0981.HK", "SMIC": "0981.HK", "981": "0981.HK",
    
    # 美股 (US)
    "NVDA": "NVDA", "輝達": "NVDA", "NVIDIA": "NVDA",
    "TSLA": "TSLA", "特斯拉": "TSLA", "TESLA": "TSLA",
    "AAPL": "AAPL", "蘋果": "AAPL", "APPLE": "AAPL",
    "MSFT": "MSFT", "微軟": "MSFT", "MICROSOFT": "MSFT",
    "GOOG": "GOOG", "谷歌": "GOOG", "GOOGLE": "GOOG",
    "AMD": "AMD", "超微": "AMD",
    "MSTR": "MSTR", "微策略": "MSTR",
    "COIN": "COIN", "COINBASE": "COIN",
    "PLTR": "PLTR", "PALANTIR": "PLTR",
    "SMCI": "SMCI", "美超微": "SMCI",
    "TQQQ": "TQQQ", "SOXL": "SOXL"
}

# --- 3. 智能代碼解析器 (Smart Resolver) ---
def resolve_ticker(user_input):
    clean_input = user_input.strip().upper()
    
    # 策略 1: 直接查字典 (最準)
    if clean_input in STOCK_MAPPING:
        return STOCK_MAPPING[clean_input]
    
    # 策略 2: 處理純數字 (港股邏輯)
    # 如果輸入 "700"，自動補成 "0700.HK"
    if clean_input.isdigit():
        return f"{int(clean_input):04d}.HK"
        
    # 策略 3: 假設是用戶輸入的美股代碼 (如 ORCL, INTC)
    return clean_input

# --- 4. 數據抓取 ---
def get_data_v18(ticker):
    try:
        # 使用最新的 yfinance 抓取
        stock = yf.Ticker(ticker)
        
        # 強制抓取 2 年數據 (計算年線用)
        df = stock.history(period="2y")
        
        if df.empty:
            return None, None
            
        # 嘗試獲取名稱，若失敗則用代碼代替
        try:
            name = stock.info.get('longName', ticker)
        except:
            name = ticker
            
        return df, name
    except Exception as e:
        return None, None

# --- 5. 華爾街指標運算 ---
def calculate_indicators(df):
    if len(df) < 50: return df # 數據過少不運算
    
    # 均線
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean() # 牛熊線
    
    # 布林
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (2 * df['STD20'])
    df['Lower'] = df['MA20'] - (2 * df['STD20'])
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (波動率) - 用於計算精確止損
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

# --- 6. AI 策略大腦 (更嚴謹的 Prompt) ---
def ask_gemini_pro(api_key, ticker_name, ticker_code, df, style):
    last = df.iloc[-1]
    
    # 黃金分割
    high_2y = df['High'].max()
    low_2y = df['Low'].min()
    fibo_0618 = high_2y - (high_2y - low_2y) * 0.618
    
    # 計算乖離率
    bias_200 = ((last['Close'] - last['MA200']) / last['MA200']) * 100 if not pd.isna(last['MA200']) else 0
    
    data_summary = f"""
    【標的】{ticker_name} ({ticker_code})
    【當前數據】
    - 現價：{last['Close']:.2f}
    - MA20(月線)：{last['MA20']:.2f}
    - MA60(季線)：{last['MA60']:.2f}
    - MA200(年線)：{last['MA200']:.2f} (判斷牛熊關鍵)
    - 年線乖離率：{bias_200:.2f}%
    - RSI(14)：{last['RSI']:.2f}
    - ATR(波動值)：{last['ATR']:.2f}
    - 布林上軌：{last['Upper']:.2f} / 下軌：{last['Lower']:.2f}
    - 黃金分割 0.618 回調位：{fibo_0618:.2f}
    """
    
    persona = ""
    if style == "保守價值型":
        persona = "你是巴菲特風格的價值投資者。你極度保守，只在股價回測年線(MA200)或嚴重超賣(RSI<30)時才考慮買入。若股價在高位，請建議觀望或賣出。"
    elif style == "趨勢動能型":
        persona = "你是順勢交易者。喜歡在股價站上均線且突破時買入。若跌破均線則果斷止損。"
    else:
        persona = "你是短線當沖客。利用布林通道上下軌進行逆勢操作。"

    prompt = f"""
    {persona}
    請分析以下數據：
    {data_summary}
    
    請給我一份【極度精確】的交易計畫 (繁體中文)，嚴禁模稜兩可：
    
    1. 🎯 **【機構精算買入價】**：
       - 請不要給範圍！給我一個具體數字。
       - 計算邏輯：請參考 MA200、布林下軌或黃金分割位。
       - 如果現在不適合買，請給出「等待回調至 XXX」的建議。
       
    2. 🎯 **【機構精算賣出價】**：
       - 給出一個具體數字。
       - 計算邏輯：參考布林上軌或前波高點。
       
    3. 🛡️ **【專業止損點】**：
       - 請利用 ATR 計算 (例如：買入價 - 2*ATR)。
       - 這能顯示你的專業度。
       
    4. 🧠 **【趨勢總結】**：
       - 一句話判斷目前是「多頭回調」還是「空頭下跌」？
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        return f"AI 連線錯誤: {response.text}"
    except Exception as e:
        return f"程式錯誤: {str(e)}"

# --- 主畫面 ---
st.title("🏛️ Felix 華爾街機構操盤室 V18.2")
st.caption("Auto-Symbol Recognition | Institutional Algorithms | Gemini 2.5")

# Key 檢測
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ 未檢測到 API Key")
    api_key = st.sidebar.text_input("請輸入 API Key", type="password")
else:
    st.sidebar.success("✅ 華爾街專線：已連線")

# --- 側邊欄：單一智能輸入框 ---
st.sidebar.header("♟️ 戰略設定")

# 這是你要的：只有一個輸入框，但很聰明
user_input = st.sidebar.text_input("輸入股票 (支援：700, 騰訊, NVDA, Tesla)", value="700")

# 風格選擇
invest_style = st.sidebar.selectbox("投資風格", ["趨勢動能型", "保守價值型", "激進短線型"])

if st.sidebar.button("🔥 啟動機構級 AI 分析"):
    if not api_key:
        st.error("請輸入 API Key")
    else:
        # 1. 智能解析代碼
        target_ticker = resolve_ticker(user_input)
        
        st.info(f"🔍 正在搜尋：{user_input} ➝ 識別代碼：{target_ticker}")

        with st.spinner(f'正在調用 {target_ticker} 兩年大數據與 AI 運算...'):
            # 2. 抓取數據
            df, stock_name = get_data_v18(target_ticker)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                # --- 顯示股票名稱與代碼 ---
                st.subheader(f"📊 {stock_name} ({target_ticker})")
                
                # --- 頂部指標 ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
                
                ma200_val = f"{last['MA200']:.2f}" if not pd.isna(last['MA200']) else "N/A"
                c2.metric("牛熊線 (MA200)", ma200_val)
                c3.metric("RSI 強弱", f"{last['RSI']:.2f}")
                c4.metric("ATR 波動", f"{last['ATR']:.2f}")

                # --- 內容分頁 ---
                tab1, tab2 = st.tabs(["🧠 AI 戰略報告", "📈 趨勢K線圖"])
                
                with tab1:
                    st.markdown(f"### 🤖 Gemini 2.5 分析報告 ({invest_style})")
                    # 傳入名稱和代碼給 AI，讓它知道自己在分析誰
                    ai_reply = ask_gemini_pro(api_key, stock_name, target_ticker, df, invest_style)
                    st.info(ai_reply)

                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                    if not pd.isna(last['MA20']): fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1), name='月線'))
                    if not pd.isna(last['MA60']): fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='cyan', width=1), name='季線'))
                    if not pd.isna(last['MA200']): fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='purple', width=2), name='年線'))
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"❌ 依然找不到數據：{target_ticker}")
                st.warning("可能原因：\n1. 股票代碼輸入錯誤。\n2. Yahoo Finance 暫時無法連線。\n3. 請確認 requirements.txt 已更新至 yfinance>=0.2.40")
