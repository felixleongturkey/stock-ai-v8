import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import os

# --- 頁面設定 ---
st.set_page_config(page_title="V10.0 Gemini 3.1 旗艦分析師", layout="wide")

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

def smart_get_data(user_input, period="6mo"):
    user_input = user_input.strip().upper()
    target_ticker = user_input
    for key, val in STOCK_DB.items():
        if key in user_input:
            target_ticker = val
            break
    if user_input.isdigit():
        target_ticker = f"{str(int(user_input)).zfill(4)}.HK"

    try:
        stock = yf.Ticker(target_ticker)
        df = stock.history(period=period)
        if df.empty: return None, None, target_ticker
        try: info = stock.info
        except: info = {}
        return df, info, target_ticker
    except:
        return None, None, target_ticker

# --- 2. 技術指標運算 ---
def calculate_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
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

# --- 3. AI 核心 (自動偵測最新模型) ---
def get_available_models(api_key):
    """查詢 Google 帳號目前能用的最新模型列表"""
    genai.configure(api_key=api_key)
    available = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available.append(m.name)
    except:
        pass
    return available

def ask_gemini(api_key, ticker, df):
    genai.configure(api_key=api_key)
    recent_data = df.tail(5).to_string()
    last = df.iloc[-1]
    
    prompt = f"""
    角色：你是一位華爾街頂級分析師，精通技術分析與宏觀經濟。
    標的：{ticker}
    
    【關鍵數據】
    - 現價：{last['Close']:.2f}
    - RSI(14)：{last['RSI']:.2f}
    - MACD：{last['MACD']:.3f} (Signal: {last['Signal']:.3f})
    - 布林通道：上 {last['Upper']:.2f} / 下 {last['Lower']:.2f}
    - 均線：月線 {last['MA20']:.2f} / 季線 {last['MA60']:.2f}
    
    【近5日數據】
    {recent_data}
    
    請用【繁體中文】與【專業條列式】回答：
    1. **趨勢診斷**：目前多空結構判斷。
    2. **買賣訊號**：RSI 與通道的具體操作建議。
    3. **策略規劃**：進場點、止損點、獲利點建議。
    """
    
# --- 優先順序清單 (修正版：加入更多穩定模型) ---
    priority_models = [
        'gemini-2.0-flash-exp', # 嘗試最新的
        'gemini-1.5-pro',       # 最強穩定版
        'gemini-1.5-flash',     # 速度最快版 (保底)
        'gemini-pro'            # 舊版 (最後手段)
    ]
    
    used_model = "Unknown"
    response_text = ""
    
    # 嘗試連線
    for model_name in priority_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            response_text = response.text
            used_model = model_name
            break # 成功了就跳出迴圈
        except:
            continue # 失敗就試下一個

    if response_text == "":
        return "❌ 無法連接任何 AI 模型，請檢查 API Key 或額度。", "None"
        
    return response_text, used_model

# --- 主畫面 ---
st.title("🤖 V10.0 Gemini 3.1 旗艦分析師")

# --- 自動環境變數偵測 ---
env_key = os.environ.get("GOOGLE_API_KEY")

if env_key:
    api_key = env_key
    st.sidebar.success("🔒 安全連線：已使用系統加密 Key")
else:
    st.sidebar.warning("⚠️ 未偵測到環境變數")
    api_key = st.sidebar.text_input("請輸入 Gemini API Key", type="password")

st.sidebar.header("🔍 股票搜尋")
stock_input = st.sidebar.text_input("輸入代號", value="NVDA")

if st.sidebar.button("🚀 啟動 AI 分析"):
    if not api_key:
        st.error("❌ 請輸入 API Key！")
    else:
        with st.spinner('AI 正在切換至最強模型進行分析...'):
            df, info, ticker = smart_get_data(stock_input)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                
                st.subheader(f"📊 {ticker} 深度報告")
                c1, c2, c3 = st.columns(3)
                c1.metric("最新價", f"{last['Close']:.2f}")
                c2.metric("RSI", f"{last['RSI']:.2f}")
                
                ai_reply, used_model = ask_gemini(api_key, ticker, df)
                
                # 顯示使用的模型版本 (讓你知道是不是用到 3.1)
                st.markdown("---")
                st.caption(f"🧠 AI 核心版本：`{used_model}`")
                st.info(ai_reply)
                
                # 畫圖
                st.markdown("---")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='月線'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='壓力'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='支撐'))
                fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"找不到代號：{stock_input}")

