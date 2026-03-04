import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import os

# --- 頁面設定 ---
st.set_page_config(page_title="V13.0 AI 精算除錯版", layout="wide")

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
    elif time_frame == "3 Months": period = "3mo"; interval = "1d"
    elif time_frame == "6 Months": period = "6mo"; interval = "1d"
    elif time_frame == "1 Year": period = "1y"; interval = "1d"

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

# --- 4. AI 核心 (除錯模式) ---
def ask_gemini_debug(api_key, ticker, df, time_frame):
    # 設定 API Key
    genai.configure(api_key=api_key)
    recent_data = df.tail(10).to_string()
    last = df.iloc[-1]
    
    prompt = f"""
    角色：華爾街量化交易員。
    任務：分析 {ticker} (週期: {time_frame})。
    數據：現價 {last['Close']:.2f}, RSI {last['RSI']:.2f}, 上軌 {last['Upper']:.2f}, 下軌 {last['Lower']:.2f}。
    
    請回答：
    1. 【AI 精算買入價】：具體數字。
    2. 【AI 精算賣出價】：具體數字。
    3. 【理由】：簡短理由。
    """
    
    # 嘗試的模型列表
    models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    
    errors = [] # 收集錯誤訊息

    for m in models:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content(prompt)
            return response.text, "Success" # 成功回傳
        except Exception as e:
            # 捕捉具體錯誤原因
            error_msg = str(e)
            errors.append(f"模型 {m} 失敗: {error_msg}")
            continue
    
    # 如果全部失敗，回傳詳細錯誤日誌
    return "\n".join(errors), "Fail"

# --- 主畫面 ---
st.title("⚡ V13.0 AI 精算除錯版")

# 嘗試從環境變數抓 Key
api_key = os.environ.get("GOOGLE_API_KEY")

# 狀態顯示區
if api_key:
    # 隱藏部分 Key 顯示，確認是否抓對
    masked_key = api_key[:5] + "*" * 10 + api_key[-4:]
    st.sidebar.success(f"✅ 環境變數已載入\n({masked_key})")
else:
    st.sidebar.warning("⚠️ 未偵測到環境變數，請手動輸入")
    api_key = st.sidebar.text_input("輸入 Gemini API Key", type="password")

st.sidebar.header("🔍 操盤設定")
stock_input = st.sidebar.text_input("輸入代號", value="NVDA")
time_options = ["1 Day (即時)", "5 Days", "1 Month", "3 Months", "6 Months", "1 Year"]
selected_time = st.sidebar.selectbox("⏱️ 選擇時間週期", time_options, index=0)

if st.sidebar.button("🚀 AI 智能運算"):
    if not api_key:
        st.error("❌ 請輸入 API Key！")
    else:
        with st.spinner('AI 正在連線...'):
            df, info, ticker = smart_get_data(stock_input, selected_time)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                
                st.subheader(f"🏷️ {ticker} 即時看板")
                c1, c2 = st.columns(2)
                c1.metric("💰 最新成交價", f"{last['Close']:.2f}", f"{(last['Close'] - prev['Close']):.2f}")
                c2.metric("📊 RSI 動能", f"{last['RSI']:.2f}")
                
                # --- AI 運算 ---
                ai_result, status = ask_gemini_debug(api_key, ticker, df, selected_time)
                
                st.markdown("---")
                if status == "Success":
                    st.success("✅ AI 連線成功！")
                    st.info(ai_result)
                else:
                    st.error("❌ AI 連線全數失敗，請截圖以下錯誤訊息給我：")
                    st.code(ai_result) # 這裡會顯示 Google 拒絕的真實原因

                # --- 圖表 ---
                st.markdown("---")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                if len(df) > 20:
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='均線'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='壓力'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='支撐'))
                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"找不到代號：{stock_input}")
