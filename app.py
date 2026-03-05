import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np

# --- 1. 頁面設定 (華爾街黑金風格) ---
st.set_page_config(page_title="Felix 華爾街 AI 終端", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #000000; }
    h1 { color: #FFA500; font-family: 'Arial', sans-serif; font-weight: 900; letter-spacing: 1px; }
    .stMetric { background-color: #111; border: 1px solid #444; padding: 10px; }
    .stMetric label { color: #888; font-size: 14px; }
    .stMetric div[data-testid="stMetricValue"] { color: #eee; font-size: 24px; font-weight: bold; }
    .stButton>button { 
        width: 100%; background: linear-gradient(45deg, #FFD700, #DAA520); 
        color: black; font-weight: bold; border: none; height: 60px; font-size: 20px;
    }
    .stButton>button:hover { transform: scale(1.02); transition: 0.3s; }
    .stInfo { background-color: #0d1117; color: #c9d1d9; border-left: 5px solid #58a6ff; }
    /* 自定義輸入框樣式 */
    .stTextArea textarea { background-color: #111; color: #fff; border: 1px solid #444; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智能代碼偵探 ---
def smart_ticker_search(user_input):
    clean = user_input.strip().upper()
    # 常用對照表
    aliases = {
        "騰訊": "0700.HK", "700": "0700.HK",
        "阿里": "9988.HK", "9988": "9988.HK",
        "美團": "3690.HK", "3690": "3690.HK",
        "小米": "1810.HK", "1810": "1810.HK",
        "比亞迪": "1211.HK", "1211": "1211.HK",
        "匯豐": "0005.HK", "5": "0005.HK", "0005": "0005.HK",
        "港交所": "0388.HK", "388": "0388.HK",
        "首程": "0697.HK", "首程控股": "0697.HK", "0697": "0697.HK",
        "NVDA": "NVDA", "TSLA": "TSLA", "AAPL": "AAPL", "AMD": "AMD"
    }
    if clean in aliases: return aliases[clean]
    if clean.isdigit(): return f"{int(clean):04d}.HK"
    return clean

# --- 3. 數據抓取引擎 (增強名稱獲取) ---
def get_data_v23(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        
        # 自動容錯
        if df.empty:
            if not ticker.endswith(".HK") and ticker.isdigit():
                ticker = f"{int(ticker):04d}.HK"
                stock = yf.Ticker(ticker)
                df = stock.history(period="2y")
            elif ticker.endswith(".HK"):
                ticker = ticker.replace(".HK", "")
                stock = yf.Ticker(ticker)
                df = stock.history(period="2y")
        
        if df.empty: return None, None, ticker

        # 嘗試多種方式獲取正確名稱
        name = None
        try: name = stock.info.get('longName')
        except: pass
        
        if not name:
            try: name = stock.info.get('shortName')
            except: pass
            
        # 如果 Yahoo 真的抓不到名字，就回傳代碼，讓 AI 去辨識
        if not name: name = ticker 
            
        return df, name, ticker
    except:
        return None, None, ticker

# --- 4. 專業指標運算 ---
def calculate_indicators(df):
    if len(df) < 5: return df
    
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['Close'].rolling(window=60, min_periods=1).mean()
    
    if len(df) >= 150:
        df['MA200'] = df['Close'].rolling(window=200, min_periods=150).mean()
    else:
        df['MA200'] = np.nan

    df['STD20'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['Upper'] = df['MA20'] + (2 * df['STD20'])
    df['Lower'] = df['MA20'] - (2 * df['STD20'])
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14, min_periods=1).mean()
    
    return df

# --- 5. AI 大腦 (新增：自定義提問 + 名稱校正) ---
def ask_gemini_v23(api_key, name, ticker, df, style, custom_question):
    last = df.iloc[-1]
    has_ma200 = not pd.isna(last['MA200'])
    
    ref_support = f"{last['Lower']:.2f} (布林下軌)"
    ref_resistance = f"{last['Upper']:.2f} (布林上軌)"
    
    trend_note = f"MA200: {last['MA200']:.2f}" if has_ma200 else "新股模式"

    data_summary = f"""
    【標的代碼】{ticker}
    【Yahoo提供的名稱】{name} (若名稱為代碼，請AI自行辨識正確公司名稱)
    【即時數據】
    - 現價：{last['Close']:.2f}
    - 參考低吸位：{ref_support}
    - 參考止盈位：{ref_resistance}
    - ATR：{last['ATR']:.2f}
    - RSI：{last['RSI']:.2f}
    """
    
    # 處理用戶自定義問題
    user_q_prompt = ""
    if custom_question:
        user_q_prompt = f"""
        🙋‍♂️ **【用戶專屬提問】**：
        用戶問：「{custom_question}」
        請在報告最後，專門開一個段落，針對這個問題進行專業解答。
        """
    
    prompt = f"""
    角色：你是華爾街頂級操盤手，風格是「{style}」。
    
    ⚠️ **重要指令：公司名稱校正**
    請先確認代碼 {ticker} 的正確公司名稱 (例如 0697.HK 應為 首程控股)。
    如果 Yahoo 提供的名稱有誤，請以你資料庫中的正確名稱為準進行分析，不要產生幻覺。
    
    數據面板：
    {data_summary}
    {trend_note}
    
    請撰寫【狙擊手交易計畫】(繁體中文)：
    
    1. 🏢 **【標的確認】**：請明確寫出你正在分析的公司名稱。
    2. 🧠 **【趨勢研判】**：簡短判斷多空。
    3. 🔵 **【狙擊買入價 (Buy Limit)】**：給出具體數字，必須低於現價 (回調買入)。
    4. 🔴 **【狙擊賣出價 (Sell Limit)】**：給出具體數字。
    5. 🛡️ **【止損防線】**：跌破哪裡逃命。
    
    {user_q_prompt}
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        elif response.status_code == 429:
            return "🚨 **系統忙碌 (429)**：API 使用過於頻繁，請稍等一分鐘再試。"
        else:
            return f"連線錯誤 ({response.status_code}): {response.text}"
    except Exception as e:
        return f"系統錯誤: {str(e)}"

# --- 主畫面 ---
st.title("🏛️ Felix 華爾街 AI 終端 V23")
st.caption("Custom Q&A | Anti-Hallucination | Gemini 1.5 Flash")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ 請輸入金鑰")
    api_key = st.sidebar.text_input("API Key", type="password")
else:
    st.sidebar.success("✅ 華爾街專線：Connected")

st.sidebar.header("♟️ 戰略中樞")
user_input = st.sidebar.text_input("輸入股票代碼 (如 0697, NVDA)", value="0697")
style = st.sidebar.selectbox("切換操盤風格", ["趨勢狙擊 (Momentum)", "價值投資 (Value)", "逆勢交易 (Contrarian)"])

# --- 新增：自定義問題輸入框 ---
st.sidebar.markdown("---")
st.sidebar.write("🙋‍♂️ **向 Felix 提問 (選填)**")
custom_question = st.sidebar.text_area(
    "輸入你想問的問題...", 
    placeholder="例如：這隻股票適合存股嗎？現在進場風險大嗎？",
    height=100
)

if st.sidebar.button("🔥 啟動狙擊運算"):
    if not api_key:
        st.error("Missing API Key")
    else:
        ticker_search = smart_ticker_search(user_input)
        st.info(f"🔍 鎖定標的：{ticker_search} ...")

        with st.spinner('AI 正在校對公司名稱並進行深度分析...'):
            df, name, ticker = get_data_v23(ticker_search)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                is_new_stock = pd.isna(last['MA200'])
                
                # 在標題只顯示代碼，避免 Yahoo 錯誤名稱誤導，正確名稱由 AI 報告提供
                st.subheader(f"📊 股票代碼：{ticker}")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
                ma200_val = f"{last['MA200']:.2f}" if not is_new_stock else "N/A"
                c2.metric("牛熊線 (MA200)", ma200_val)
                c3.metric("RSI 強弱", f"{last['RSI']:.2f}")
                c4.metric("ATR 波動率", f"{last['ATR']:.2f}")

                tab1, tab2 = st.tabs(["🧠 AI 交易報告 (含問答)", "📈 專業技術圖"])
                
                with tab1:
                    st.markdown(f"### 🎯 {style} 策略報告")
                    # 傳入自定義問題
                    ai_reply = ask_gemini_v23(api_key, name, ticker, df, style, custom_question)
                    
                    if "系統忙碌" in ai_reply:
                        st.error(ai_reply)
                    else:
                        st.info(ai_reply)

                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                    if not pd.isna(last['MA20']): fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1), name='MA20'))
                    if not pd.isna(last['MA60']): fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='cyan', width=1), name='MA60'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='布林上軌'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='布林下軌'))
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"❌ 找不到數據：{ticker_search}")
