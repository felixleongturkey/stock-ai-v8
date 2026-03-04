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
    h1 { color: #d4af37; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    .css-1d391kg { background-color: #111111; }
    .stMetric { background-color: #1a1a1a; border: 1px solid #333; padding: 15px; border-radius: 5px; }
    .stMetric label { color: #888; }
    .stMetric div[data-testid="stMetricValue"] { color: #fff; font-weight: bold; }
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
    .stInfo, .stSuccess, .stError { border-radius: 0px; border-left: 5px solid #d4af37; background-color: #111; color: #ddd; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 常用股票快速選單 (僅供參考，不強制) ---
# 這個字典主要用於快速填入，但使用者可以修改
COMMON_STOCKS = {
    "NVDA (輝達)": "NVDA", "TSLA (特斯拉)": "TSLA", "AAPL (蘋果)": "AAPL", 
    "AMD": "AMD", "MSFT (微軟)": "MSFT", "GOOG (谷歌)": "GOOG",
    "騰訊 (0700)": "0700.HK", "阿里 (9988)": "9988.HK", "美團 (3690)": "3690.HK",
    "小米 (1810)": "1810.HK", "比亞迪 (1211)": "1211.HK", "匯豐 (0005)": "0005.HK"
}

# --- 3. 數據獲取邏輯 (修復版) ---
def smart_get_data(user_input):
    # 1. 清理輸入
    ticker = user_input.strip().upper()
    
    # 2. 如果使用者選的是下拉選單的中文名稱 (例如 "騰訊 (0700)")，嘗試從字典轉換
    if ticker in COMMON_STOCKS:
        ticker = COMMON_STOCKS[ticker]
    
    # 3. 處理港股代碼 (如果輸入的是纯数字)
    # 例如輸入 "700" -> "0700.HK", 輸入 "0700" -> "0700.HK"
    if ticker.isdigit():
        ticker = f"{int(ticker):04d}.HK"
        
    try:
        stock = yf.Ticker(ticker)
        # 嘗試抓取數據
        df = stock.history(period="2y", interval="1d")
        
        if df.empty:
            return None, None, ticker
            
        return df, stock.info, ticker
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, ticker

# --- 4. 華爾街級指標運算 ---
def calculate_pro_indicators(df):
    # 如果數據不足 200 天，還是盡量算，避免報錯
    min_periods = 1 
    
    # 1. 均線系統
    df['MA20'] = df['Close'].rolling(window=20, min_periods=min_periods).mean()
    df['MA60'] = df['Close'].rolling(window=60, min_periods=min_periods).mean()
    df['MA120'] = df['Close'].rolling(window=120, min_periods=min_periods).mean()
    df['MA200'] = df['Close'].rolling(window=200, min_periods=min_periods).mean()
    
    # 2. 布林通道
    df['STD20'] = df['Close'].rolling(window=20, min_periods=min_periods).std()
    df['Upper'] = df['MA20'] + (2 * df['STD20'])
    df['Lower'] = df['MA20'] - (2 * df['STD20'])
    
    # 3. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=min_periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=min_periods).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4. MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 5. ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14, min_periods=min_periods).mean()
    
    return df

# --- 5. AI 策略大腦 ---
def ask_gemini_strategy(api_key, ticker, df, style):
    last = df.iloc[-1]
    
    # 黃金分割計算 (需有足夠數據)
    if len(df) > 120:
        recent_high = df['High'].tail(120).max()
        recent_low = df['Low'].tail(120).min()
    else:
        recent_high = df['High'].max()
        recent_low = df['Low'].min()
        
    fibo_0618 = recent_high - (recent_high - recent_low) * 0.618
    
    data_summary = f"""
    【市場數據 snapshot】
    - 現價：{last['Close']:.2f}
    - MA20：{last['MA20']:.2f}
    - MA60：{last['MA60']:.2f}
    - MA200 (牛熊線)：{last['MA200']:.2f}
    - RSI：{last['RSI']:.2f}
    - ATR：{last['ATR']:.2f}
    - 區間高點：{recent_high:.2f} / 低點：{recent_low:.2f}
    - 0.618 回調位：{fibo_0618:.2f}
    """
    
    if style == "保守價值型 (Value)":
        persona = "你是一位像巴菲特 (Warren Buffett) 的價值投資者。極度厭惡風險，喜歡在回測長期均線 (MA200) 或黃金分割位時低接。"
    elif style == "趨勢動能型 (Momentum)":
        persona = "你是一位像馬克·米奈爾維尼 (Mark Minervini) 的趨勢交易者。喜歡右側交易，突破買入，跌破均線止損。"
    else:
        persona = "你是一位高頻當沖交易員，擅長利用布林通道逆勢操作。"

    prompt = f"""
    {persona}
    請分析標的：{ticker}。
    數據：
    {data_summary}
    
    請給我一份【專業機構級】的交易計畫 (繁體中文)，包含：
    1. 🧠 **深度趨勢解讀**：(MA200位置意義)
    2. 💎 **機構精算買點**：(給出明確狙擊價格)
    3. 🚀 **目標獲利點**：(第一/第二目標)
    4. 🛡️ **嚴格止損點**：(基於ATR計算)
    """

    # 使用與你 Node.js 相同的原生 requests 方式
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

# --- 主畫面邏輯 ---
st.title("🏛️ Felix 華爾街機構操盤室 V18.1")
st.caption("Powered by Gemini 2.5 | 2-Year Data | Institutional Algorithms")

# 環境變數讀取 (支援 GEMINI_API_KEY 或 GOOGLE_API_KEY)
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.sidebar.error("⚠️ 未檢測到金鑰")
    api_key = st.sidebar.text_input("請輸入 API Key", type="password")
else:
    st.sidebar.success("✅ 華爾街連線通道：已建立")

# --- 側邊欄設定 (改回文字輸入框) ---
st.sidebar.header("♟️ 戰略設定")

# 1. 提供一個下拉選單作為「快速選擇」
quick_select = st.sidebar.selectbox(
    "快速選擇熱門股 (或直接在下方輸入)", 
    ["自定義輸入"] + list(COMMON_STOCKS.keys())
)

# 2. 核心輸入框：預設值根據上面的選擇變動
default_input = "NVDA"
if quick_select != "自定義輸入":
    default_input = COMMON_STOCKS[quick_select]

# 這裡讓使用者可以自由輸入任何代碼
stock_input = st.sidebar.text_input("輸入股票代碼 (如 0700, TSLA, AAPL)", value=default_input)

invest_style = st.sidebar.selectbox("選擇你的投資風格", ["趨勢動能型 (Momentum)", "保守價值型 (Value)", "激進短線型 (Day Trade)"])

if st.sidebar.button("🔥 啟動機構級 AI 分析"):
    if not api_key:
        st.error("請輸入 API Key")
    else:
        with st.spinner('正在調用 2 年歷史數據，計算 MA200 牛熊分界與黃金分割率...'):
            # 使用輸入框的內容去抓數據
            df, info, ticker = smart_get_data(stock_input)
            
            if df is not None:
                df = calculate_pro_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                # --- 頂部指標 ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
                
                # 容錯處理：如果數據不足算不出 MA200，顯示 N/A
                ma200_val = f"{last['MA200']:.2f}" if not pd.isna(last['MA200']) else "N/A (上市未滿一年)"
                c2.metric("MA200 (牛熊線)", ma200_val)
                
                c3.metric("RSI (14)", f"{last['RSI']:.2f}")
                c4.metric("ATR (波動)", f"{last['ATR']:.2f}")

                # --- 核心分頁 ---
                tab1, tab2, tab3 = st.tabs(["🧠 AI 戰略報告", "📊 技術儀表板", "📈 趨勢K線圖"])
                
                with tab1:
                    st.markdown(f"### 🤖 Gemini 2.5 首席分析師報告 ({invest_style})")
                    ai_reply = ask_gemini_strategy(api_key, ticker, df, invest_style)
                    st.info(ai_reply)
                    
                with tab2:
                    st.markdown("### 🛠️ 市場結構分析")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("#### 均線多空排列")
                        # 確保 MA200 不是 NaN 才能比較
                        if not pd.isna(last['MA200']):
                            if last['Close'] > last['MA200']:
                                st.success("🔥 股價 > 年線：長線多頭")
                            else:
                                st.error("❄️ 股價 < 年線：長線空頭")
                        else:
                            st.info("⚠️ 數據不足以判斷年線")
                            
                    with col_b:
                        st.write("#### 乖離率")
                        if not pd.isna(last['MA200']):
                            bias = ((last['Close'] - last['MA200']) / last['MA200']) * 100
                            st.metric("年線乖離率", f"{bias:.2f}%")
                        else:
                            st.metric("年線乖離率", "N/A")

                with tab3:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                    
                    # 畫均線 (有值才畫)
                    if not pd.isna(last['MA20']): fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1), name='月線'))
                    if not pd.isna(last['MA60']): fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='cyan', width=1), name='季線'))
                    if not pd.isna(last['MA200']): fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='purple', width=2), name='年線'))
                    
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                # 這裡會顯示到底為什麼找不到
                st.error(f"❌ 找不到數據：{ticker}")
                st.warning("請確認代碼是否正確。港股請輸入 0700, 美股請輸入 NVDA。")
