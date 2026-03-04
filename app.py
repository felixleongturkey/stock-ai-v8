import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os

# --- 1. 頁面設定 (專業金融風格) ---
st.set_page_config(page_title="Felix 股價 AI 助手", layout="wide", initial_sidebar_state="expanded")

# 自定義 CSS 讓介面更像彭博終端機 (Bloomberg Terminal)
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #303030; }
    .stButton>button { width: 100%; background-color: #00d2be; color: black; font-weight: bold; border-radius: 8px; height: 50px; }
    .stButton>button:hover { background-color: #00b8a5; }
    .stSuccess { background-color: #1e2130; color: #00d2be; border: 1px solid #00d2be; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 股票代碼資料庫 ---
STOCK_DB = {
    "快手": "1024.HK", "01024": "1024.HK", "1024": "1024.HK",
    "騰訊": "0700.HK", "700": "0700.HK",
    "阿里": "9988.HK", "美團": "3690.HK",
    "小米": "1810.HK", "比亞迪": "1211.HK",
    "匯豐": "0005.HK", "港交所": "0388.HK",
    "NVDA": "NVDA", "TSLA": "TSLA", "AAPL": "AAPL", "AMD": "AMD",
    "MSTR": "MSTR", "COIN": "COIN", "SMCI": "SMCI", "GOOG": "GOOG"
}

# --- 3. 智能數據獲取 ---
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

# --- 4. 專業指標運算 ---
def calculate_indicators(df):
    if len(df) < 20: return df
    
    # 均線系統
    df['MA20'] = df['Close'].rolling(window=20).mean() # 月線 (短期成本)
    df['MA60'] = df['Close'].rolling(window=60).mean() # 季線 (生命線)
    
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

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 成交量變異 (Volume Ratio)
    df['VolMA'] = df['Volume'].rolling(window=20).mean()
    df['VolRatio'] = df['Volume'] / df['VolMA']
    
    return df

# --- 5. AI 核心 (Prompt 大升級) ---
def ask_gemini_pro(api_key, ticker, df, time_frame):
    recent_data = df.tail(10).to_string()
    last = df.iloc[-1]
    
    # 這是讓 AI 變聰明的關鍵指令
    prompt = f"""
    角色：你是華爾街頂級避險基金的首席交易員，擅長「左側交易」(在支撐位掛單)。
    任務：分析 {ticker} ({time_frame})，給出極具操作價值的買賣點。
    
    【關鍵技術數據】
    - 現價：{last['Close']:.2f}
    - RSI(14)：{last['RSI']:.2f} (30以下超賣，70以上超買)
    - 布林下軌(支撐)：{last['Lower']:.2f}
    - 布林上軌(壓力)：{last['Upper']:.2f}
    - 月線(MA20)：{last['MA20']:.2f}
    - 季線(MA60 - 生命線)：{last['MA60']:.2f} (若無數據則忽略)
    
    【近10筆走勢】
    {recent_data}
    
    請回答以下三點 (必須用繁體中文，格式要漂亮)：
    
    1. 🎯 **【精準買入價位】**：
       - 請不要只寫現價！
       - 請找出下方的「強力支撐位」(例如布林下軌或季線附近)。
       - 告訴我一個具體的掛單價格。
       
    2. 🎯 **【精準賣出價位】**：
       - 請找出上方的「重大壓力位」(例如布林上軌或整數關卡)。
       - 告訴我一個具體的止盈價格。
       
    3. 🧠 **【AI 操盤邏輯】**：
       - 簡短解釋為什麼選這兩個價格？(例如：RSI背離、回測季線有撐...)
       - 目前趨勢是多頭還是空頭？
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"連線錯誤: {response.text}"
    except Exception as e:
        return f"程式錯誤: {str(e)}"

# --- 主畫面 ---
st.title("🤖 Felix 股價 AI 助手")

# 環境變數讀取
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if api_key:
    # 優化 1: 不顯示 Key，只顯示安全連線
    st.sidebar.success("✅ 已安全連線 Google Gemini")
else:
    st.sidebar.error("❌ 未偵測到 Key")
    api_key = st.sidebar.text_input("請輸入 API Key", type="password")

st.sidebar.header("🔍 股票搜尋")
stock_input = st.sidebar.text_input("輸入代號", value="NVDA")
time_options = ["1 Day (即時)", "5 Days", "1 Month", "3 Months", "6 Months"]
selected_time = st.sidebar.selectbox("⏱️ 分析週期", time_options, index=0)

# 優化 2: 按鈕文字修改
if st.sidebar.button("🚀 啟動 AI 即時分析"):
    if not api_key:
        st.error("請輸入 API Key")
    else:
        with st.spinner('Felix AI 正在計算最佳買賣點...'):
            df, info, ticker = smart_get_data(stock_input, selected_time)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                change = last['Close'] - prev['Close']
                pct_change = (change / prev['Close']) * 100
                
                # --- 專業看板區 ---
                st.subheader(f"📊 {ticker} 即時盤面")
                
                # 優化 4: 增加多點專業數據
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("💰 最新成交", f"{last['Close']:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
                c2.metric("🌊 RSI 動能", f"{last['RSI']:.2f}")
                c3.metric("📈 趨勢 (MACD)", "多頭" if last['MACD'] > last['Signal'] else "空頭")
                
                # 成交量異動判斷
                vol_status = "🔥 爆量" if last.get('VolRatio', 1) > 1.5 else "⚖️ 正常"
                c4.metric("📊 籌碼量能", vol_status)

                # --- AI 分析結果 ---
                ai_result = ask_gemini_pro(api_key, ticker, df, selected_time)
                
                st.markdown("---")
                st.markdown("### 🧠 Felix AI 交易策略報告")
                
                # 使用原生 HTML 渲染更漂亮的卡片
                st.info(ai_result)

                # --- 互動圖表 ---
                st.markdown("### 📉 K線技術圖表")
                fig = go.Figure()
                
                # K線
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                
                # 只有資料足夠才畫均線
                if len(df) > 20:
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='月線 (MA20)'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='壓力 (Upper)'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='支撐 (Lower)'))
                
                # 佈局設定
                fig.update_layout(
                    height=550, 
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark", # 深色模式
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"找不到代號：{stock_input}")
