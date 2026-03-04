import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import os
import re

# --- 頁面設定 ---
st.set_page_config(page_title="V12.0 AI 精準定價系統", layout="wide")

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

    # 時間週期設定
    period = "6mo"
    interval = "1d"
    
    if time_frame == "1 Day (即時)":
        period = "1d"; interval = "5m"
    elif time_frame == "5 Days":
        period = "5d"; interval = "15m"
    elif time_frame == "1 Month":
        period = "1mo"; interval = "1d"
    elif time_frame == "3 Months":
        period = "3mo"; interval = "1d"
    elif time_frame == "6 Months":
        period = "6mo"; interval = "1d"
    elif time_frame == "1 Year":
        period = "1y"; interval = "1d"

    try:
        stock = yf.Ticker(target_ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty: return None, None, target_ticker
        try: info = stock.info
        except: info = {}
        return df, info, target_ticker
    except:
        return None, None, target_ticker

# --- 3. 技術指標運算 ---
def calculate_indicators(df):
    if len(df) < 20: return df # 數據不足不運算
    
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

# --- 4. AI 核心：精準定價運算 ---
def ask_gemini_pricing(api_key, ticker, df, time_frame):
    """
    請求 AI 給出精確的數值，並解析結果
    """
    genai.configure(api_key=api_key)
    recent_data = df.tail(10).to_string()
    last = df.iloc[-1]
    
    # 這是給 AI 的嚴格指令，要求它做數學運算
    prompt = f"""
    角色：你是一位華爾街頂級量化交易員。
    任務：分析以下股票數據，計算出【最精準】的買入價與賣出價。
    
    標的：{ticker} (週期: {time_frame})
    
    【最新技術指標】
    - 現價：{last['Close']:.2f}
    - RSI(14)：{last['RSI']:.2f}
    - MACD：{last['MACD']:.3f}
    - 布林上軌：{last['Upper']:.2f}
    - 布林下軌：{last['Lower']:.2f}
    - 20日均線：{last['MA20']:.2f}
    
    【近10筆詳細數據】
    {recent_data}
    
    請用繁體中文回答，必須包含以下三個部分：
    1. **【AI 精算價格】**：
       - 請直接給出一個具體的「最佳買入價格」數字。
       - 請直接給出一個具體的「最佳賣出/止盈價格」數字。
       (不要給範圍，請根據支撐壓力算出一個最可能的精確值)
       
    2. **【精簡分析】**：一句話解釋為什麼選這兩個價格。
    
    3. **【詳細邏輯】**：詳細解釋你的運算邏輯（為什麼這個價格是強力支撐或壓力）。
    """
    
    models = ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro']
    
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content(prompt)
            return response.text
        except: continue
    return "❌ AI 連線失敗"

# --- 主畫面 ---
st.title("⚡ V12.0 AI 精準定價系統")

# 環境變數 Key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    # 嘗試從 Streamlit Secrets 讀取 (進階用法)
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.sidebar.text_input("輸入 Gemini API Key", type="password")

st.sidebar.header("🔍 操盤設定")
stock_input = st.sidebar.text_input("輸入代號", value="NVDA")
time_options = ["1 Day (即時)", "5 Days", "1 Month", "3 Months", "6 Months", "1 Year"]
selected_time = st.sidebar.selectbox("⏱️ 選擇時間週期", time_options, index=0) # 預設改為即時

if st.sidebar.button("🚀 AI 智能運算 (獲取精準價格)"):
    if not api_key:
        st.error("❌ 請輸入 API Key！")
    else:
        with st.spinner('AI 正在進行大數據運算，尋找最佳買賣點...'):
            df, info, ticker = smart_get_data(stock_input, selected_time)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                
                # 1. 顯示即時報價
                st.subheader(f"🏷️ {ticker} 即時看板")
                c1, c2, c3 = st.columns(3)
                c1.metric("💰 最新成交價", f"{last['Close']:.2f}", f"{(last['Close'] - prev['Close']):.2f}")
                c2.metric("📊 RSI 動能", f"{last['RSI']:.2f}")
                
                # 2. 呼叫 AI 進行精算
                ai_result = ask_gemini_pricing(api_key, ticker, df, selected_time)
                
                if "AI 連線失敗" in ai_result:
                    st.error("AI 連線失敗，請檢查 Key。")
                else:
                    # 3. 顯示 AI 算出來的結果
                    st.markdown("---")
                    st.markdown("### 🤖 AI 精算結果 (Gemini Computed)")
                    
                    # 使用 Info 框顯示 AI 的完整回答
                    st.info(ai_result)
                    
                    # 4. 顯示圖表
                    st.markdown("---")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                    
                    if len(df) > 20:
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='均線'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='壓力'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='支撐'))
                    
                    fig.update_layout(title=f"{ticker} - {selected_time} 走勢圖", height=500, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"找不到代號：{stock_input}")
