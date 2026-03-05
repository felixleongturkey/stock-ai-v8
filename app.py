import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np

# --- 1. 頁面設定 (商業級黑金風格) ---
st.set_page_config(page_title="Felix 華爾街 AI 終端", layout="wide", initial_sidebar_state="expanded")

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
    .stSuccess { background-color: #051e05; color: #4caf50; border-left: 5px solid #4caf50; }
    /* 加強文字顯示 */
    p, li { font-size: 16px; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智能代碼偵探 (通用型，不寫死) ---
def smart_ticker_search(user_input):
    clean = user_input.strip().upper()
    
    # 針對純數字輸入，優先判定為港股 (這是最常見的輸入習慣)
    if clean.isdigit():
        return f"{int(clean):04d}.HK"
    
    # 常見美股直接回傳
    if clean.isalpha() and len(clean) <= 5:
        return clean
        
    return clean

# --- 3. 數據抓取引擎 (只負責抓數據，不負責猜名字) ---
def get_data_v25(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 強制抓取 2 年數據
        df = stock.history(period="2y")
        
        # 自動容錯：如果 .HK 抓不到，嘗試去掉 .HK (針對美股被誤判的情況)
        if df.empty:
            if ticker.endswith(".HK"):
                alt_ticker = ticker.replace(".HK", "")
                stock = yf.Ticker(alt_ticker)
                df = stock.history(period="2y")
                if not df.empty: ticker = alt_ticker # 修正代碼
        
        # 再抓不到，嘗試補 .HK (針對港股被誤判的情況)
        if df.empty:
            if not ticker.endswith(".HK") and ticker.isdigit():
                alt_ticker = f"{int(ticker):04d}.HK"
                stock = yf.Ticker(alt_ticker)
                df = stock.history(period="2y")
                if not df.empty: ticker = alt_ticker

        if df.empty: return None, None, ticker

        # 嘗試獲取名稱，但我們不再完全信任它
        raw_name = None
        try: raw_name = stock.info.get('longName')
        except: pass
        if not raw_name:
            try: raw_name = stock.info.get('shortName')
            except: pass
            
        return df, raw_name, ticker
    except:
        return None, None, ticker

# --- 4. 專業指標運算 ---
def calculate_indicators(df):
    if len(df) < 5: return df
    
    # 均線
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['Close'].rolling(window=60, min_periods=1).mean()
    
    if len(df) >= 150:
        df['MA200'] = df['Close'].rolling(window=200, min_periods=150).mean()
    else:
        df['MA200'] = np.nan

    # 布林通道
    df['STD20'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['Upper'] = df['MA20'] + (2 * df['STD20'])
    df['Lower'] = df['MA20'] - (2 * df['STD20'])
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (波動率)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14, min_periods=1).mean()
    
    return df

# --- 5. AI 核心 (包含身份核對層 & 模型輪盤) ---
def ask_gemini_universal(api_key, raw_name, ticker, df, style, custom_question):
    last = df.iloc[-1]
    has_ma200 = not pd.isna(last['MA200'])
    
    # 計算參考價位
    ref_support = f"{last['Lower']:.2f}"
    ref_resistance = f"{last['Upper']:.2f}"
    
    # 如果 raw_name 是 None，標記為未知，強迫 AI 識別
    company_name_input = raw_name if raw_name else "未知 (請根據代碼識別)"

    data_summary = f"""
    【系統輸入代碼】{ticker}
    【系統抓取名稱】{company_name_input} (⚠️警告：若此名稱與代碼不符，請忽略並自行更正)
    【即時技術數據】
    - 最新收盤價：{last['Close']:.2f} (請用此價格核對公司身份)
    - RSI (14)：{last['RSI']:.2f}
    - ATR (波動率)：{last['ATR']:.2f}
    - 布林下軌 (支撐參考)：{ref_support}
    - 布林上軌 (壓力參考)：{ref_resistance}
    - MA200 (牛熊線)：{f"{last['MA200']:.2f}" if has_ma200 else "N/A"}
    """
    
    user_q_prompt = ""
    if custom_question:
        user_q_prompt = f"""
        🙋‍♂️ **【用戶專屬提問】**：
        用戶問：「{custom_question}」
        請在報告最後，新增一個標題為「🗣️ Felix 答客問」的段落，詳細回答此問題。
        """
    
    # Prompt 優化：加入「身份核對」步驟
    prompt = f"""
    角色：你是華爾街頂級避險基金經理，風格是「{style}」。
    任務：撰寫一份【機構級投資交易計畫】。
    
    🚨 **STEP 1: 身份核對 (Critical)**
    你收到的代碼是 {ticker}，價格是 {last['Close']:.2f}。
    請先確認這是哪一家公司？(例如 0697.HK 應為 首程控股，而非華電)。
    如果系統提供的名稱有誤，請直接更正，並以更正後的公司為準進行分析。
    
    🚨 **STEP 2: 交易計畫 (繁體中文)**
    請包含以下章節，嚴禁模稜兩可：
    
    1. 🏢 **【標的確認】**：
       - 明確寫出你分析的公司名稱、代碼、以及當前股價。
       
    2. 🧠 **【趨勢研判】**：
       - 目前是多頭、空頭還是盤整？(參考 MA200 與 RSI)
       
    3. 🔵 **【精準買入價 (Buy Limit)】**：
       - 給出一個具體的「掛單價格」。
       - 邏輯：必須低於現價 (等待回調)，可參考布林下軌 {ref_support}。
       
    4. 🔴 **【精準賣出價 (Sell Limit)】**：
       - 給出一個具體的「止盈價格」。
       - 邏輯：參考布林上軌 {ref_resistance} 或前波高點。
       
    5. 🛡️ **【風控止損 (Stop Loss)】**：
       - 跌破哪裡必須離場？(建議參考 ATR)
       
    {user_q_prompt}
    """

    # 模型輪盤：解決 404/429 問題
    models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    for model in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()['candidates'][0]['content']['parts'][0]['text']
                return result, model # 成功回傳
            elif response.status_code == 429:
                return "🚨 系統忙碌 (429)：額度不足，請稍後再試。", "None"
            
            # 如果是 404/500/503，繼續嘗試下一個模型
            continue
            
        except:
            continue
            
    return "❌ 所有 AI 伺服器皆無回應，請檢查 API Key 或網路。", "None"

# --- 主畫面 ---
st.title("🏛️ Felix 華爾街 AI 終端 V25")
st.caption("AI Identity Check | Auto-Correction | Institutional Grade")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ 請輸入金鑰")
    api_key = st.sidebar.text_input("API Key", type="password")
else:
    st.sidebar.success("✅ 華爾街專線：Connected")

st.sidebar.header("♟️ 戰略中樞")
user_input = st.sidebar.text_input("輸入股票代碼 (如 0697, NVDA)", value="0697")
style = st.sidebar.selectbox("切換操盤風格", ["趨勢狙擊 (Momentum)", "價值投資 (Value)", "逆勢交易 (Contrarian)"])

st.sidebar.markdown("---")
st.sidebar.write("🙋‍♂️ **向 Felix 提問 (選填)**")
custom_question = st.sidebar.text_area("輸入你的問題...", height=100)

if st.sidebar.button("🔥 啟動全自動分析"):
    if not api_key:
        st.error("Missing API Key")
    else:
        ticker_search = smart_ticker_search(user_input)
        st.info(f"🔍 正在鎖定代碼：{ticker_search} ...")

        with st.spinner('AI 正在進行「身份核對」與「數據運算」...'):
            df, raw_name, ticker = get_data_v25(ticker_search)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                # --- 數據看板 ---
                st.subheader(f"📊 代碼：{ticker}")
                st.caption(f"原始回傳名稱：{raw_name if raw_name else 'N/A (交由 AI 識別)'}")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
                
                ma200_display = f"{last['MA200']:.2f}" if not pd.isna(last['MA200']) else "N/A"
                c2.metric("牛熊線 (MA200)", ma200_display)
                c3.metric("RSI 強弱", f"{last['RSI']:.2f}")
                c4.metric("ATR 波動率", f"{last['ATR']:.2f}")

                tab1, tab2 = st.tabs(["🧠 機構交易報告", "📈 技術圖表"])
                
                with tab1:
                    st.markdown(f"### 🎯 {style} 策略報告")
                    
                    # 執行 AI 分析
                    ai_reply, used_model = ask_gemini_universal(api_key, raw_name, ticker, df, style, custom_question)
                    
                    if "系統忙碌" in ai_reply or "無回應" in ai_reply:
                        st.error(ai_reply)
                    else:
                        st.success(f"✅ 分析完成 (核心引擎：{used_model})")
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
                st.warning("請確認代碼是否正確。")
