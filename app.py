import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np

# --- 1. 頁面設定 (專業深色版) ---
st.set_page_config(page_title="Felix AI 股票分析員", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    h1 { color: #00d2ff; font-family: 'Helvetica Neue', sans-serif; font-weight: 900; }
    .stMetric { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    .stMetric label { color: #888; }
    .stMetric div[data-testid="stMetricValue"] { color: #fff; font-size: 26px; font-weight: bold; }
    .stButton>button { 
        width: 100%; background: linear-gradient(90deg, #0072ff, #00c6ff); 
        color: white; font-weight: bold; border: none; height: 60px; font-size: 18px; border-radius: 8px;
    }
    .stButton>button:hover { box-shadow: 0 0 15px rgba(0, 198, 255, 0.5); }
    .stTextArea textarea { background-color: #111; color: #fff; border: 1px solid #444; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智能代碼與名稱修正 (最高權限字典) ---
def smart_ticker_search(user_input):
    clean = user_input.strip().upper()
    if clean.isdigit(): return f"{int(clean):04d}.HK"
    if clean.isalpha() and len(clean) <= 5: return clean
    return clean

def get_fixed_name(ticker):
    # 這裡強制修正名稱，解決 Yahoo 抓錯問題
    FIXED_NAMES = {
        "0697.HK": "首程控股 (Shoucheng Holdings)",
        "0694.HK": "北京首都機場 (Beijing Capital Airport)",
        "0700.HK": "騰訊控股 (Tencent)",
        "9988.HK": "阿里巴巴 (Alibaba)",
        "3690.HK": "美團 (Meituan)",
        "1810.HK": "小米集團 (Xiaomi)",
        "1211.HK": "比亞迪股份 (BYD)",
        "0005.HK": "匯豐控股 (HSBC)",
        "0388.HK": "香港交易所 (HKEX)",
        "NVDA": "NVIDIA",
        "TSLA": "Tesla",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "AMD": "AMD"
    }
    return FIXED_NAMES.get(ticker, None)

# --- 3. 數據抓取 ---
def get_data_v30(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        # 容錯：自動切換代碼後綴
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

        # 優先使用強制修正的名稱
        fixed_name = get_fixed_name(ticker)
        if fixed_name:
            name = fixed_name
        else:
            try: name = stock.info.get('longName', ticker)
            except: name = ticker
            
        return df, name, ticker
    except:
        return None, None, ticker

# --- 4. 專業數學運算 (計算買賣點給 AI 用) ---
def calculate_levels(df):
    if len(df) < 50: return df
    
    # 均線
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
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

    # --- 關鍵：計算「有憑有據」的價格 ---
    last = df.iloc[-1]
    
    # 1. 樞軸點 (Pivot Support/Resistance)
    P = (last['High'] + last['Low'] + last['Close']) / 3
    S1 = (2 * P) - last['High'] # 第一支撐
    R1 = (2 * P) - last['Low']  # 第一壓力
    
    # 2. 黃金分割 (Fibonacci)
    recent_high = df['High'].tail(120).max()
    recent_low = df['Low'].tail(120).min()
    fibo_0618 = recent_high - (recent_high - recent_low) * 0.618
    
    # 3. ATR (止損距離)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]

    return df, S1, R1, fibo_0618, atr

# --- 5. AI 核心 (Gemini 2.5 Flash - 模仿 Node.js 連線) ---
def ask_gemini_v30(api_key, name, ticker, df, style, custom_q, S1, R1, fibo, atr):
    last = df.iloc[-1]
    
    # 準備精準數據
    buy_target = min(S1, fibo, last['Lower']) # 取三個支撐中最低的 (保守買點)
    sell_target = max(R1, last['Upper'])      # 取壓力中最高的 (最大獲利)
    
    data_summary = f"""
    【標的】{name} ({ticker}) - 名稱已強制校正
    【現價】{last['Close']:.2f}
    【RSI】{last['RSI']:.2f}
    【ATR波動】{atr:.2f}
    
    【系統計算的關鍵位階 (請直接參考)】
    - 樞軸支撐 (S1)：{S1:.2f}
    - 黃金分割 (0.618)：{fibo:.2f}
    - 布林下軌：{last['Lower']:.2f}
    --> 建議買入參考價：{buy_target:.2f} 左右
    
    - 樞軸壓力 (R1)：{R1:.2f}
    - 布林上軌：{last['Upper']:.2f}
    --> 建議賣出參考價：{sell_target:.2f} 左右
    """
    
    user_prompt = ""
    if custom_q:
        user_prompt = f"\n🙋‍♂️ **用戶特別提問**：{custom_q}\n(請在報告最後單獨回答此問題)"
    
    prompt = f"""
    角色：你是華爾街資深交易員，風格為 {style}。
    任務：根據提供的數學位階，給出精準的交易指令。
    
    ⚠️ **最高指令**：
    1. **拒絕廢話**：不要說「現價附近買入」。
    2. **使用數據**：買入價必須參考上方的【建議買入參考價】。
    3. **解釋原因**：告訴用戶為什麼是這個價格 (例如：這是黃金分割與布林通道的重疊支撐)。
    
    請撰寫繁體中文報告：
    
    1. 🎯 **市場趨勢判斷**
    2. 🔵 **精準買入掛單 (Buy Limit)**：給出具體價格，並解釋支撐來源。
    3. 🔴 **精準獲利掛單 (Sell Limit)**：給出具體價格，並解釋壓力來源。
    4. 🛡️ **止損設定 (Stop Loss)**：跌破哪裡要跑？(參考 ATR)
    {user_prompt}
    """

    # 鎖定 gemini-2.5-flash (這是你在 server.js 驗證過能用的型號)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        elif response.status_code == 404:
            # 如果 2.5 真的不行，自動降級到 1.5 (雙重保險)
            url_backup = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            response_backup = requests.post(url_backup, headers=headers, json=data)
            if response_backup.status_code == 200:
                return response_backup.json()['candidates'][0]['content']['parts'][0]['text'] + "\n\n(註：使用 1.5 Flash 備用線路)"
            return f"❌ 連線失敗 (404)。請確認 Key 是否支援該模型。"
        elif response.status_code == 429:
            return "🚨 API 額度已滿 (429)。請稍後再試。"
        else:
            return f"連線錯誤 ({response.status_code})"
    except Exception as e:
        return f"系統錯誤: {str(e)}"

# --- 主畫面 ---
st.title("🏛️ Felix AI 股票分析員")
st.caption("Powered by Gemini 2.5 Flash | Professional Math Models")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ 請輸入 API Key")
    api_key = st.sidebar.text_input("Key", type="password")
else:
    st.sidebar.success("✅ 華爾街專線：Connected")

st.sidebar.header("♟️ 操盤設定")
user_input = st.sidebar.text_input("輸入代碼 (如 0697, NVDA)", value="0697")
style = st.sidebar.selectbox("分析風格", ["趨勢波段 (Trend)", "價值回歸 (Value)", "短線當沖 (Day Trade)"])

# --- 這裡是你要求的：加回自定義問題 ---
st.sidebar.markdown("---")
st.sidebar.write("🙋‍♂️ **向 Felix 提問 (選填)**")
custom_question = st.sidebar.text_area("例如：這隻股票適合存股嗎？", height=100)

if st.sidebar.button("🔥 啟動分析"):
    if not api_key:
        st.error("請輸入 API Key")
    else:
        ticker_search = smart_ticker_search(user_input)
        st.info(f"🔍 鎖定代碼：{ticker_search} ...")

        with st.spinner('正在進行 樞軸點(Pivot) 與 黃金分割 運算...'):
            df, name, ticker = get_data_v30(ticker_search)
            
            if df is not None:
                # 算出精準價位
                df, S1, R1, fibo, atr = calculate_levels(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                st.subheader(f"📊 {name} ({ticker})")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
                c2.metric("樞軸支撐 (S1)", f"{S1:.2f}")
                c3.metric("黃金分割 (0.618)", f"{fibo:.2f}")
                c4.metric("布林上軌 (壓力)", f"{last['Upper']:.2f}")

                tab1, tab2 = st.tabs(["🧠 AI 策略報告", "📈 技術分析圖"])
                
                with tab1:
                    # 呼叫 AI (帶入算好的 S1, R1, Fibo)
                    report = ask_gemini_v30(api_key, name, ticker, df, style, custom_question, S1, R1, fibo, atr)
                    
                    if "❌" in report or "🚨" in report:
                        st.error(report)
                    else:
                        st.info(report)

                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='布林上軌'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='布林下軌'))
                    # 畫出樞軸點支撐
                    fig.add_hline(y=S1, line_dash="dash", line_color="orange", annotation_text="Pivot S1 (支撐)")
                    
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"❌ 找不到數據：{ticker_search}")
                st.warning("請確認代碼。")
