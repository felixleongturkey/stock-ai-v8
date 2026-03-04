import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np

# --- 1. 頁面設定 (極致專業暗黑風) ---
st.set_page_config(page_title="Felix 華爾街機構操盤室", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    /* 頂部標題 */
    h1 { color: #d4af37; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    /* 側邊欄 */
    .css-1d391kg { background-color: #111111; }
    /* 指標卡片 */
    .stMetric { background-color: #1a1a1a; border: 1px solid #333; padding: 15px; border-radius: 5px; }
    .stMetric label { color: #888; }
    .stMetric div[data-testid="stMetricValue"] { color: #fff; font-weight: bold; }
    /* 按鈕優化 */
    .stButton>button { 
        width: 100%; 
        background: linear-gradient(90deg, #d4af37 0%, #f2c94c 100%); 
        color: black; 
        font-weight: 900; 
        border: none; 
        height: 55px; 
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover { box-shadow: 0 0 15px #d4af37; color: white; }
    /* 訊息框 */
    .stInfo, .stSuccess, .stError { border-radius: 0px; border-left: 5px solid #d4af37; background-color: #111; color: #ddd; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 資料庫 ---
STOCK_DB = {
    "NVDA (輝達)": "NVDA", "TSLA (特斯拉)": "TSLA", "AAPL (蘋果)": "AAPL", "AMD": "AMD",
    "MSFT (微軟)": "MSFT", "GOOG (谷歌)": "GOOG", "AMZN (亞馬遜)": "AMZN",
    "MSTR (微策略)": "MSTR", "COIN (Coinbase)": "COIN", "SMCI": "SMCI",
    "TQQQ (三倍那斯達克)": "TQQQ", "SOXL (三倍半導體)": "SOXL",
    "騰訊 (0700)": "0700.HK", "阿里 (9988)": "9988.HK", "美團 (3690)": "3690.HK",
    "小米 (1810)": "1810.HK", "比亞迪 (1211)": "1211.HK", "匯豐 (0005)": "0005.HK", "港交所 (0388)": "0388.HK"
}

# --- 3. 專業數據獲取 (抓取 2 年數據) ---
def smart_get_data(ticker_key):
    ticker = STOCK_DB.get(ticker_key, ticker_key)
    # 如果使用者手動輸入數字 (港股)
    if ticker.isdigit(): ticker = f"{str(int(ticker)).zfill(4)}.HK"
    
    try:
        stock = yf.Ticker(ticker)
        # 抓取 2 年數據，這是計算 MA200 (牛熊線) 的基礎
        df = stock.history(period="2y", interval="1d")
        if df.empty: return None, None, ticker
        return df, stock.info, ticker
    except:
        return None, None, ticker

# --- 4. 華爾街級指標運算 ---
def calculate_pro_indicators(df):
    if len(df) < 200: return df # 數據不足 200 天無法算牛熊線
    
    # 1. 均線系統 (MA System)
    df['MA20'] = df['Close'].rolling(window=20).mean()  # 月線 (短期)
    df['MA60'] = df['Close'].rolling(window=60).mean()  # 季線 (中期)
    df['MA120'] = df['Close'].rolling(window=120).mean() # 半年線 (中長)
    df['MA200'] = df['Close'].rolling(window=200).mean() # 年線 (長期牛熊分界)
    
    # 2. 布林通道 (Bollinger Bands)
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (2 * df['STD20'])
    df['Lower'] = df['MA20'] - (2 * df['STD20'])
    
    # 3. RSI (強弱指標)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4. MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 5. ATR (真實波動幅度 - 用於計算止損)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

# --- 5. AI 策略大腦 (Persona Injection) ---
def ask_gemini_strategy(api_key, ticker, df, style):
    # 準備近 30 天數據 + 關鍵長線指標
    last = df.iloc[-1]
    
    # 計算黃金分割率 (Fibonacci)
    recent_high = df['High'].tail(120).max() # 半年高點
    recent_low = df['Low'].tail(120).min()   # 半年低點
    fibo_0382 = recent_high - (recent_high - recent_low) * 0.382
    fibo_0618 = recent_high - (recent_high - recent_low) * 0.618
    
    data_summary = f"""
    【市場狀態 snapshot】
    - 現價：{last['Close']:.2f}
    - MA20 (月線)：{last['MA20']:.2f}
    - MA60 (季線)：{last['MA60']:.2f}
    - MA200 (牛熊線)：{last['MA200']:.2f} (判斷長期多空)
    - RSI：{last['RSI']:.2f}
    - ATR (波動值)：{last['ATR']:.2f}
    - 半年高點：{recent_high:.2f} / 半年低點：{recent_low:.2f}
    - 黃金分割 0.618 回調位：{fibo_0618:.2f}
    """
    
    # 根據用戶選擇的風格，切換 AI 的「大腦」
    if style == "保守價值型 (Value)":
        persona = "你是一位像巴菲特 (Warren Buffett) 的價值投資者。你極度厭惡風險，只喜歡在股價低於價值、且回測長期均線 (MA200) 或黃金分割 0.618 時才出手。如果股價過高，你會建議空手。"
    elif style == "趨勢動能型 (Momentum)":
        persona = "你是一位像馬克·米奈爾維尼 (Mark Minervini) 的趨勢交易者。你喜歡『右側交易』，只在股價站上 MA20/MA60 且強勢突破時買入。如果股價在 MA200 之下，你會判定為空頭，建議做空或觀望。"
    else: # 激進短線
        persona = "你是一位高頻當沖交易員，擅長利用布林通道逆勢操作。尋找短線乖離過大的反彈機會。"

    prompt = f"""
    {persona}
    
    請分析標的：{ticker}。
    這是我提供的 2 年期大數據運算結果：
    {data_summary}
    
    請給我一份【專業機構級】的交易計畫 (繁體中文)，包含：
    
    1. 🧠 **【深度趨勢解讀】**：
       - 目前股價相對於 MA200 (牛熊線) 的位置代表什麼意義？
       - 這是牛市回調，還是熊市反彈？
       
    2. 💎 **【機構精算買點 (Entry)】**：
       - 不要給範圍，給出一個「狙擊價格」。
       - 理由必須結合 MA200、黃金分割 0.618 或 ATR 支撐。
       
    3. 🚀 **【目標獲利點 (Target)】**：
       - 第一目標價 (保守) 與 第二目標價 (貪婪)。
       - 理由 (例如前波壓力或整數關卡)。
       
    4. 🛡️ **【嚴格止損點 (Stop Loss)】**：
       - 根據 ATR 計算 (例如進場價 - 2倍 ATR)。
       - 告訴我跌破哪個價位代表趨勢完全壞掉。
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        return "AI 連線逾時，請重試。"
    except Exception as e:
        return f"錯誤: {str(e)}"

# --- 主畫面邏輯 ---
st.title("🏛️ Felix 華爾街機構操盤室 V18")
st.caption("Powered by Gemini 2.5 Pro | 2-Year Historical Data | Institutional Algorithms")

# 環境變數
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.sidebar.error("⚠️ 未檢測到金鑰")
    api_key = st.sidebar.text_input("請輸入 API Key", type="password")
else:
    st.sidebar.success("✅ 華爾街連線通道：已建立")

# --- 側邊欄設定 ---
st.sidebar.header("♟️ 戰略設定")
selected_ticker_key = st.sidebar.selectbox("選擇標的", list(STOCK_DB.keys()))
invest_style = st.sidebar.selectbox("選擇你的投資風格 (AI 人格)", ["趨勢動能型 (Momentum)", "保守價值型 (Value)", "激進短線型 (Day Trade)"])
st.sidebar.info(f"當前模式：{invest_style}\nAI 將模擬該流派大師的思維進行運算。")

if st.sidebar.button("🔥 啟動機構級 AI 分析"):
    if not api_key:
        st.error("請輸入 API Key")
    else:
        with st.spinner('正在調用 2 年歷史數據，計算 MA200 牛熊分界與黃金分割率...'):
            df, info, ticker = smart_get_data(selected_ticker_key)
            
            if df is not None:
                df = calculate_pro_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                # --- 頂部關鍵指標 ---
                c1, c2, c3, c4 = st.columns(4)
                change_color = "off" if last['Close'] < prev['Close'] else "normal"
                c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
                c2.metric("MA200 (牛熊線)", f"{last['MA200']:.2f}", delta_color="off")
                c3.metric("RSI (14)", f"{last['RSI']:.2f}")
                c4.metric("ATR (波動值)", f"{last['ATR']:.2f}")

                # --- 核心分頁 ---
                tab1, tab2, tab3 = st.tabs(["🧠 AI 戰略報告", "📊 技術儀表板", "📈 趨勢K線圖"])
                
                with tab1:
                    st.markdown(f"### 🤖 Gemini 2.5 Flash 首席分析師報告 ({invest_style})")
                    ai_reply = ask_gemini_strategy(api_key, ticker, df, invest_style)
                    st.info(ai_reply)
                    
                with tab2:
                    st.markdown("### 🛠️ 市場結構分析")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("#### 均線多空排列")
                        if last['Close'] > last['MA200']:
                            st.success("🔥 股價位於年線 (MA200) 之上：長線多頭格局")
                        else:
                            st.error("❄️ 股價位於年線 (MA200) 之下：長線空頭格局")
                            
                        if last['MA20'] > last['MA60']:
                            st.success("✅ 短期均線向上：動能強勢")
                        else:
                            st.warning("⚠️ 短期均線向下：動能轉弱")
                            
                    with col_b:
                        st.write("#### 乖離率與風險")
                        bias_200 = ((last['Close'] - last['MA200']) / last['MA200']) * 100
                        st.metric("年線乖離率 (BIAS)", f"{bias_200:.2f}%")
                        if bias_200 > 30: st.warning("⚠️ 乖離過大，小心回調")
                        elif bias_200 < -20: st.info("💎 負乖離過大，超跌機會")
                        else: st.write("✅ 乖離正常")

                with tab3:
                    fig = go.Figure()
                    # K線
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                    # 關鍵均線
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1), name='月線 (20MA)'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='cyan', width=1), name='季線 (60MA)'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='purple', width=2), name='年線 (200MA)'))
                    
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"找不到數據：{ticker}")
