import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import json

# --- 頁面設定 ---
st.set_page_config(page_title="V15.0 AI 股票分析 (原生 API 版)", layout="wide")

# --- 1. 股票代碼資料庫 ---
STOCK_DB = {
    "快手": "1024.HK", "01024": "1024.HK", "1024": "1024.HK",
    "騰訊": "0700.HK", "700": "0700.HK",
    "阿里": "9988.HK", "美團": "3690.HK",
    "小米": "1810.HK", "比亞迪": "1211.HK",
    "匯豐": "0005.HK", "港交所": "0388.HK",
    "NVDA": "NVDA", "TSLA": "TSLA", "AAPL": "AAPL", "AMD": "AMD",
    "MSTR": "MSTR", "COIN": "COIN", "SMCI": "SMCI", "GOOG": "GOOG"
}

# --- 2. 智能數據獲取 ---
def smart_get_data(user_input, time_frame):
    user_input = user_input.strip().upper()
    target_ticker = user_input
    for key, val in STOCK_DB.items():
        if key in user_input:
            target_ticker = val
            break
    if user_input.isdigit():
        target_ticker = f"{str(int(user_input)).zfill(4)}.HK"

    period = "6mo"; interval = "1d"
    if time_frame == "1 Day (即時)": period = "1d"; interval = "5m"
    elif time_frame == "5 Days": period = "5d"; interval = "15m"
    elif time_frame == "1 Month": period = "1mo"; interval = "1d"
    
    try:
        stock = yf.Ticker(target_ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty: return None, None, target_ticker
        try: info = stock.info
        except: info = {}
        return df, info, target_ticker
    except:
        return None, None, target_ticker

# --- 3. 指標運算 ---
def calculate_indicators(df):
    if len(df) < 20: return df
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (2 * df['STD20'])
    df['Lower'] = df['MA20'] - (2 * df['STD20'])
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

# --- 4. AI 核心 (改用原生 Requests，模仿你的 server.js) ---
def ask_gemini_direct(api_key, ticker, df, time_frame):
    recent_data = df.tail(10).to_string()
    last = df.iloc[-1]
    
    prompt = f"""
    角色：華爾街量化分析師。
    標的：{ticker} ({time_frame})
    數據：現價 {last['Close']:.2f}, RSI {last['RSI']:.2f}, 布林上 {last['Upper']:.2f}/下 {last['Lower']:.2f}
    近10筆走勢：
    {recent_data}
    
    請回答：
    1. 【精準買入價】：(給數字)
    2. 【精準賣出價】：(給數字)
    3. 【理由】：(簡短分析)
    """

    # 這是你的 server.js 使用的邏輯，我們用 Python 重寫一次
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            # 解析 Google 回傳的 JSON 結構
            try:
                return result['candidates'][0]['content']['parts'][0]['text']
            except:
                return f"解析錯誤: {result}"
        else:
            return f"連線錯誤 (Status {response.status_code}): {response.text}"
            
    except Exception as e:
        return f"程式錯誤: {str(e)}"

# --- 主畫面 ---
st.title("⚡ V15.0 AI 原生架構版 (與 Node.js 同核)")

# 自動嘗試讀取兩個常見的變數名稱
api_key = os.environ.get("GEMINI_API_KEY") # 優先讀取這個，跟你舊網站一樣
if not api_key:
    api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    masked = api_key[:5] + "..." + api_key[-4:]
    st.sidebar.success(f"✅ 已讀取 Key: {masked}")
else:
    st.sidebar.error("❌ 未讀取到 Key，請確認 Render 環境變數設定為 GEMINI_API_KEY")
    api_key = st.sidebar.text_input("或手動輸入 Key", type="password")

st.sidebar.header("🔍 設定")
stock_input = st.sidebar.text_input("輸入代號", value="NVDA")
time_options = ["1 Day (即時)", "5 Days", "1 Month", "3 Months"]
selected_time = st.sidebar.selectbox("⏱️ 週期", time_options, index=0)

if st.sidebar.button("🚀 啟動 AI (Gemini 2.5 Flash)"):
    if not api_key:
        st.error("請輸入 Key")
    else:
        with st.spinner('連線 Google Gemini 2.5 Flash 中...'):
            df, info, ticker = smart_get_data(stock_input, selected_time)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                
                st.subheader(f"🏷️ {ticker} 看板")
                c1, c2 = st.columns(2)
                c1.metric("💰 最新價", f"{last['Close']:.2f}", f"{(last['Close'] - prev['Close']):.2f}")
                c2.metric("📊 RSI", f"{last['RSI']:.2f}")
                
                # 呼叫 AI (使用模擬 Node.js 的方式)
                ai_result = ask_gemini_direct(api_key, ticker, df, selected_time)
                
                st.markdown("---")
                if "連線錯誤" in ai_result or "程式錯誤" in ai_result:
                    st.error(ai_result)
                else:
                    st.info(f"🤖 **Gemini 2.5 Flash 分析結果：**\n\n{ai_result}")

                # 圖表
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                if len(df) > 20:
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='均線'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='壓力'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='支撐'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("找不到股票")
