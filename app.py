import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np

# --- 1. 頁面設定 (2026 未來風格) ---
st.set_page_config(page_title="Felix AI 股票分析員", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    h1 { 
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif; font-weight: 900; 
    }
    .stMetric { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 10px; }
    .stMetric label { color: #888; }
    .stMetric div[data-testid="stMetricValue"] { color: #fff; font-size: 26px; font-weight: bold; }
    .stButton>button { 
        width: 100%; background: linear-gradient(90deg, #00d2ff, #3a7bd5); 
        color: white; font-weight: bold; border: none; height: 60px; font-size: 18px; border-radius: 8px;
    }
    .stButton>button:hover { box-shadow: 0 0 20px rgba(0, 210, 255, 0.5); transform: scale(1.01); }
    .stInfo { background-color: #0f172a; border-left: 5px solid #0ea5e9; }
    .stSuccess { background-color: #022c22; border-left: 5px solid #34d399; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智能代碼與名稱修正 (最高權限) ---
def smart_ticker_search(user_input):
    clean = user_input.strip().upper()
    if clean.isdigit(): return f"{int(clean):04d}.HK"
    if clean.isalpha() and len(clean) <= 5: return clean
    return clean

def get_corrected_name(ticker):
    # 這裡是你要求的「精準名稱」對照表
    # 無論 Yahoo 回傳什麼，這裡的名稱優先級最高
    OVERRIDES = {
        "0697.HK": "首程控股 (Shoucheng Holdings)",
        "0694.HK": "北京首都機場 (Beijing Capital Airport)",
        "0696.HK": "中國民航信息 (TravelSky)",
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
    return OVERRIDES.get(ticker, None)

# --- 3. 數據抓取 ---
def get_data_v29(ticker):
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
        corrected_name = get_corrected_name(ticker)
        if corrected_name:
            name = corrected_name
        else:
            try: name = stock.info.get('longName', ticker)
            except: name = ticker
            
        return df, name, ticker
    except:
        return None, None, ticker

# --- 4. 專業數學指標 (Pivot Points & Fibonacci) ---
def calculate_pro_math(df):
    if len(df) < 50: return df
    
    # 基礎指標
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

    # --- 關鍵：樞軸點 (Pivot Points) - 機構常用的支撐壓力 ---
    last = df.iloc[-1]
    P = (last['High'] + last['Low'] + last['Close']) / 3
    R1 = (2 * P) - last['Low']  # 第一壓力
    S1 = (2 * P) - last['High'] # 第一支撐
    
    # --- 關鍵：黃金分割 (Fibonacci Retracement) - 近半年 ---
    recent_high = df['High'].tail(120).max()
    recent_low = df['Low'].tail(120).min()
    fibo_0618 = recent_high - (recent_high - recent_low) * 0.618
    fibo_0382 = recent_high - (recent_high - recent_low) * 0.382

    # 將這些數值存入 df 最後一行，方便傳給 AI
    df['Pivot_S1'] = S1
    df['Pivot_R1'] = R1
    df['Fibo_Support'] = fibo_0618
    df['Fibo_Resist'] = fibo_0382
    
    return df

# --- 5. AI 核心 (支援 Gemini 3.0 Flash) ---
def ask_gemini_v29(api_key, name, ticker, df, style):
    last = df.iloc[-1]
    
    # 準備「精準價位」給 AI，不讓它亂猜
    # 買入參考：取 Pivot S1 或 布林下軌 中較低者 (安全邊際)
    buy_ref_1 = f"{last['Pivot_S1']:.2f} (樞軸支撐)"
    buy_ref_2 = f"{last['Lower']:.2f} (布林下軌)"
    
    # 賣出參考：取 Pivot R1 或 布林上軌
    sell_ref_1 = f"{last['Pivot_R1']:.2f} (樞軸壓力)"
    sell_ref_2 = f"{last['Upper']:.2f} (布林上軌)"
    
    ma200_status = "N/A"
    if not pd.isna(last['MA200']):
        ma200_status = f"{last['MA200']:.2f} ({'股價在牛熊線上' if last['Close'] > last['MA200'] else '股價在牛熊線下'})"

    data_summary = f"""
    【標的】{name} ({ticker})
    【機構級數據面板】
    - 現價：{last['Close']:.2f}
    - 牛熊分界 (MA200)：{ma200_status}
    - RSI 動能：{last['RSI']:.2f}
    
    【關鍵數學位階 (請依此設定價格)】
    - 強力支撐區 (建議買點)：{buy_ref_1} 或 {buy_ref_2}
    - 強力壓力區 (建議賣點)：{sell_ref_1} 或 {sell_ref_2}
    """
    
    prompt = f"""
    角色：你是一位使用 Gemini 3.0 模型的頂級量化交易員。
    風格：{style}
    
    任務：分析 {name} ({ticker})，並依據數學模型給出「精確」的交易指令。
    
    ⚠️ **最高指令**：
    1. **名稱確認**：你分析的是 {name}，絕對不要搞錯。
    2. **拒絕平庸**：不要給「現價減一點」這種無腦建議。請參考上方的【樞軸支撐】或【布林下軌】給出掛單價。
    3. **專業術語**：請使用「回測支撐」、「突破壓力」、「RSI背離」等專業用語。
    
    請輸出報告 (繁體中文)：
    
    ### 1. 🎯 市場解讀 (Gemini 3 Analysis)
    (簡述目前趨勢是多頭回調，還是空頭下跌？MA200 的意義？)
    
    ### 2. 🔵 機構級買入掛單 (Buy Limit)
    - **建議價格**：(請給出具體數字，參考上方支撐區)
    - **邏輯**：(例如：股價回測樞軸點 S1，且 RSI 超賣)
    
    ### 3. 🔴 機構級止盈掛單 (Sell Limit)
    - **建議價格**：(請給出具體數字，參考上方壓力區)
    - **邏輯**：(例如：觸及布林上軌壓力)
    
    ### 4. 🛡️ 止損防線
    - **價格**：(跌破哪裡代表趨勢反轉？)
    """

    # --- 模型選擇 (Model Strategy) ---
    # 優先嘗試 Gemini 3.0 (如果 API 已支援)，否則降級到 2.0/1.5
    models_to_try = [
        "gemini-3.0-flash",       # 用戶要求的最新型號
        "gemini-2.0-flash-exp",   # 目前最強的實驗版
        "gemini-1.5-flash"        # 最穩定的備用版
    ]
    
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    for model in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result_text = response.json()['candidates'][0]['content']['parts'][0]['text']
                return result_text, model
            elif response.status_code == 404:
                continue # 該型號在 API 可能尚未開通，嘗試下一個
            elif response.status_code == 429:
                continue # 額度滿，嘗試下一個
                
        except:
            continue

    return "❌ 無法連接任何 AI 模型 (Gemini 3/2/1.5 皆無回應)。請檢查 API Key 額度。", "None"

# --- 主畫面 ---
st.title("🏛️ Felix AI 股票分析員")
st.caption("Powered by Gemini 3.0 Flash | Pivot Points | Institutional Grade")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ 請輸入 API Key")
    api_key = st.sidebar.text_input("Key", type="password")
else:
    st.sidebar.success("✅ 華爾街專線：已連線")

st.sidebar.header("♟️ 操盤設定")
user_input = st.sidebar.text_input("輸入代碼 (如 0697, NVDA)", value="0697")
style = st.sidebar.selectbox("分析風格", ["趨勢波段 (Trend)", "價值回歸 (Value)", "短線當沖 (Day Trade)"])

if st.sidebar.button("🔥 啟動 Gemini 3 分析"):
    if not api_key:
        st.error("請輸入 API Key")
    else:
        ticker_search = smart_ticker_search(user_input)
        st.info(f"🔍 鎖定代碼：{ticker_search} ...")

        with st.spinner('正在進行 樞軸點(Pivot) 與 黃金分割 運算...'):
            df, name, ticker = get_data_v29(ticker_search)
            
            if df is not None:
                df = calculate_pro_math(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                st.subheader(f"📊 {name} ({ticker})")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
                c2.metric("樞軸支撐 (S1)", f"{last['Pivot_S1']:.2f}")
                c3.metric("布林下軌 (低吸)", f"{last['Lower']:.2f}")
                c4.metric("布林上軌 (壓力)", f"{last['Upper']:.2f}")

                tab1, tab2 = st.tabs(["🧠 Gemini 3 策略報告", "📈 技術分析圖"])
                
                with tab1:
                    report, model_used = ask_gemini_v29(api_key, name, ticker, df, style)
                    
                    if "❌" in report:
                        st.error(report)
                    else:
                        st.success(f"✅ 分析完成 (核心引擎：{model_used})")
                        st.markdown(report)

                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Pivot_S1'], line=dict(color='orange', width=1, dash='dash'), name='樞軸支撐 (S1)'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='布林上軌'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='布林下軌'))
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"❌ 找不到數據：{ticker_search}")
                st.warning("請確認代碼。")
