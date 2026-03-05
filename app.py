import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np

# --- 1. 頁面設定 (固定標題) ---
st.set_page_config(page_title="Felix AI 股票分析員", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    h1 { color: #FFA500; font-family: 'Helvetica Neue', sans-serif; font-weight: 900; letter-spacing: 1px; }
    .stMetric { background-color: #111; border: 1px solid #444; padding: 15px; border-radius: 8px; }
    .stMetric label { color: #888; font-size: 14px; }
    .stMetric div[data-testid="stMetricValue"] { color: #eee; font-size: 26px; font-weight: bold; }
    .stButton>button { 
        width: 100%; background: linear-gradient(90deg, #FFD700, #DAA520); 
        color: black; font-weight: bold; border: none; height: 60px; font-size: 20px;
    }
    .stButton>button:hover { transform: scale(1.02); transition: 0.2s; box-shadow: 0 0 15px #FFD700; }
    .stInfo { background-color: #0d1117; color: #c9d1d9; border-left: 5px solid #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智能代碼偵探 ---
def smart_ticker_search(user_input):
    clean = user_input.strip().upper()
    if clean.isdigit(): return f"{int(clean):04d}.HK"
    if clean.isalpha() and len(clean) <= 5: return clean
    return clean

# --- 3. 數據抓取 (輕量化 1年數據) ---
def get_data_native(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        # 自動容錯
        if df.empty:
            if ticker.endswith(".HK"):
                alt = ticker.replace(".HK", "")
                stock = yf.Ticker(alt)
                df = stock.history(period="1y")
                if not df.empty: ticker = alt
            elif ticker.isdigit():
                alt = f"{int(ticker):04d}.HK"
                stock = yf.Ticker(alt)
                df = stock.history(period="1y")
                if not df.empty: ticker = alt
        
        if df.empty: return None, None, ticker

        name = ticker
        try:
            info_name = stock.info.get('longName')
            if info_name: name = info_name
        except: pass
            
        return df, name, ticker
    except:
        return None, None, ticker

# --- 4. 指標運算 ---
def calculate_indicators(df):
    if len(df) < 20: return df
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 布林通道
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (2 * df['STD20'])
    df['Lower'] = df['MA20'] - (2 * df['STD20'])
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

# --- 5. AI 核心 (Native REST API - Gemini 2.5 Flash) ---
def ask_gemini_native(api_key, name, ticker, df, style, custom_question):
    last = df.iloc[-1]
    
    ref_low = f"{last['Lower']:.2f}"
    ref_high = f"{last['Upper']:.2f}"
    
    data_summary = f"""
    代碼：{ticker} (請自行校正公司名稱)
    現價：{last['Close']:.2f}
    RSI：{last['RSI']:.2f}
    布林下軌(支撐)：{ref_low}
    布林上軌(壓力)：{ref_high}
    """
    
    user_q = ""
    if custom_question:
        user_q = f"用戶提問：{custom_question} (請回答)"
    
    prompt = f"""
    角色：華爾街分析師 ({style})。
    數據：
    {data_summary}
    
    任務：
    1. 確認公司名稱。
    2. 判斷多空趨勢。
    3. 給出買入價 (低於現價，參考支撐)。
    4. 給出賣出價 (高於現價，參考壓力)。
    5. 設定止損價。
    {user_q}
    請用繁體中文列點回答，精簡有力。
    """

    # --- 關鍵修改：完全模仿 server.js 的連線方式 ---
    # 使用 gemini-2.5-flash (你提供的 server.js 裡的型號)
    # 如果 2.5 失敗，備用 gemini-2.0-flash
    
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash"]
    
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    for model in models_to_try:
        try:
            # 這就是 server.js 裡的 fetch 網址結構
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            elif response.status_code == 429:
                return "🚨 API 額度不足 (429)。請稍後再試。"
            elif response.status_code == 404:
                continue # 嘗試下一個模型
            else:
                continue
                
        except Exception as e:
            continue

    return "❌ 連線失敗：無法連接 Gemini 2.5 或 2.0。請檢查 Key 是否有效。"

# --- 主畫面 ---
st.title("🏛️ Felix AI 股票分析員")
st.caption("V27.0 Native Core | Gemini 2.5 Flash | Server.js Replica")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ 請輸入金鑰")
    api_key = st.sidebar.text_input("API Key", type="password")
else:
    st.sidebar.success("✅ 系統連線：正常")

st.sidebar.header("♟️ 分析設定")
user_input = st.sidebar.text_input("輸入代碼 (如 0697, NVDA)", value="0697")
style = st.sidebar.selectbox("風格", ["趨勢 (Momentum)", "價值 (Value)", "逆勢 (Contrarian)"])

st.sidebar.markdown("---")
custom_question = st.sidebar.text_area("提問 (選填)", height=80)

if st.sidebar.button("🔥 開始分析"):
    if not api_key:
        st.error("Missing API Key")
    else:
        ticker_search = smart_ticker_search(user_input)
        st.info(f"🔍 搜尋中：{ticker_search}")

        with st.spinner('AI 運算中 (Gemini 2.5)...'):
            df, name, ticker = get_data_native(ticker_search)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                st.subheader(f"📊 {ticker}")
                st.caption(f"原始名稱：{name}")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
                c2.metric("MA20", f"{last['MA20']:.2f}")
                c3.metric("RSI", f"{last['RSI']:.2f}")
                c4.metric("ATR", f"{last['ATR']:.2f}")

                tab1, tab2 = st.tabs(["🧠 分析報告", "📈 技術圖"])
                
                with tab1:
                    st.markdown(f"### 🎯 {style} 策略")
                    ai_reply = ask_gemini_native(api_key, name, ticker, df, style, custom_question)
                    
                    if "連線失敗" in ai_reply or "額度不足" in ai_reply:
                        st.error(ai_reply)
                    else:
                        st.info(ai_reply)

                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1), name='MA20'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='布林上'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='布林下'))
                    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"❌ 找不到數據：{ticker_search}")
