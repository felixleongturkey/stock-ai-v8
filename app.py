import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np
import time

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Felix AI 股票分析員", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1 { 
        background: linear-gradient(to right, #58a6ff, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; }
    .stMetric label { color: #8b949e; }
    .stMetric div[data-testid="stMetricValue"] { color: #fff; font-size: 26px; font-weight: bold; }
    
    .stButton>button { 
        width: 100%; background: linear-gradient(90deg, #1f6feb, #11998e); 
        color: white; font-weight: bold; border: none; height: 55px; font-size: 18px; border-radius: 8px;
    }
    .stButton>button:hover { transform: scale(1.01); box-shadow: 0 0 15px rgba(31, 111, 235, 0.5); }
    
    div[data-testid="stSidebar"] .stButton:nth-of-type(2) button {
        background: linear-gradient(90deg, #8957e5, #b392f0); color: white;
    }
    
    .stTextArea textarea { background-color: #0d1117; color: #fff; border: 1px solid #30363d; }
    .stSuccess { background-color: #064e3b; border-left: 5px solid #34d399; }
    .stError { background-color: #450a0a; border-left: 5px solid #f85149; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 模擬富途搜尋 (Smart Ticker Resolver v2) ---
def resolve_ticker(user_input):
    clean = user_input.strip().upper()
    
    # 擴充字典：模擬富途牛牛的關聯詞搜尋
    MAPPING = {
        # 美股巨頭
        "GOOGLE": "GOOG", "GOOGL": "GOOG", "ALPHABET": "GOOG",
        "TESLA": "TSLA", "TSLA": "TSLA",
        "APPLE": "AAPL", "AAPL": "AAPL",
        "MICROSOFT": "MSFT", "MSFT": "MSFT",
        "NVIDIA": "NVDA", "NVDA": "NVDA",
        "AMAZON": "AMZN", "AMZN": "AMZN",
        "META": "META", "FACEBOOK": "META",
        "NETFLIX": "NFLX",
        "AMD": "AMD", "INTEL": "INTC", "TSM": "TSM", "TSMC": "TSM",
        "COIN": "COIN", "COINBASE": "COIN",
        "MSTR": "MSTR", "MICROSTRATEGY": "MSTR",
        "SMCI": "SMCI", "SUPERMICRO": "SMCI",
        "PLTR": "PLTR", "PALANTIR": "PLTR",
        
        # 港股熱門
        "TENCENT": "0700.HK", "騰訊": "0700.HK", "700": "0700.HK",
        "ALIBABA": "9988.HK", "BABA": "9988.HK", "阿里": "9988.HK", "9988": "9988.HK",
        "MEITUAN": "3690.HK", "美團": "3690.HK", "3690": "3690.HK",
        "XIAOMI": "1810.HK", "小米": "1810.HK", "1810": "1810.HK",
        "BYD": "1211.HK", "比亞迪": "1211.HK", "1211": "1211.HK",
        "HSBC": "0005.HK", "匯豐": "0005.HK", "5": "0005.HK", "0005": "0005.HK",
        "HKEX": "0388.HK", "港交所": "0388.HK", "388": "0388.HK",
        "SHOUCHENG": "0697.HK", "首程": "0697.HK", "首程控股": "0697.HK", "0697": "0697.HK",
        "TRAVELSKY": "0696.HK", "中航信": "0696.HK",
        "AIRPORT": "0694.HK", "北京機場": "0694.HK"
    }
    
    if clean in MAPPING:
        return MAPPING[clean]
    
    # 智能判斷：純數字預設為港股
    if clean.isdigit(): 
        return f"{int(clean):04d}.HK"
    
    # 智能判斷：純字母預設為美股
    if clean.isalpha() and len(clean) <= 5:
        return clean
        
    return clean

# --- 3. 數據抓取 (加強容錯) ---
def get_data_v34(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 抓 1 年數據即可，減少 API 負擔，加快速度
        df = stock.history(period="1y")
        
        # 容錯：找不到就換後綴
        if df.empty:
            if ticker.endswith(".HK"):
                alt = ticker.replace(".HK", "")
                stock = yf.Ticker(alt)
                df = stock.history(period="1y")
                if not df.empty: ticker = alt
            elif ticker.isdigit(): # 這是關鍵：有些庫存檔是 0700.HK
                alt = f"{int(ticker):04d}.HK"
                stock = yf.Ticker(alt)
                df = stock.history(period="1y")
                if not df.empty: ticker = alt
        
        if df.empty: return None, None, ticker

        # 獲取名稱 (如果 Yahoo 抓不到，就用代碼本身，不要回傳 None)
        name = ticker
        try: 
            info_name = stock.info.get('longName')
            if info_name: name = info_name
        except: pass
            
        return df, name, ticker
    except:
        return None, None, ticker

# --- 4. 狼性數學運算 (Wolf Math) ---
def calculate_wolf_levels(df):
    if len(df) < 50: return df, 0, 0, 0, 0, 0, "數據不足"
    
    # 指標
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean() # 狼性指標
    
    # 布林
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (2 * df['STD20'])
    df['Lower'] = df['MA20'] - (2 * df['STD20'])
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    last = df.iloc[-1]
    
    # 趨勢判斷
    trend = "盤整震盪"
    if last['Close'] > last['MA60']: trend = "多頭趨勢"
    if last['Close'] > last['MA60'] and last['MA20'] > last['MA60']: trend = "強勢多頭 🔥"
    if last['Close'] < last['MA60']: trend = "空頭趨勢 ❄️"

    # --- 狼性買點邏輯 ---
    # 1. 積極買點：EMA20 (強勢股回檔不破月線)
    aggressive_buy = last['EMA20']
    
    # 2. 保守買點：波段 0.618 或 布林下軌
    recent_high = df['High'].tail(60).max()
    recent_low = df['Low'].tail(60).min()
    fibo_support = recent_high - (recent_high - recent_low) * 0.382
    conservative_buy = max(last['Lower'], fibo_support)
    
    # 決策：如果是強勢多頭，AI 推薦積極買點；否則推薦保守買點
    final_buy = aggressive_buy if "強勢" in trend else conservative_buy

    # 賣點：布林上軌 或 前高
    final_sell = max(last['Upper'], recent_high)
    
    # ATR 止損
    high_low = df['High'] - df['Low']
    atr = high_low.rolling(14).mean().iloc[-1]

    return df, final_buy, final_sell, atr, last['MA200'], last['EMA20'], trend

# --- 5. AI 連線核心 (含自動重試機制 - Anti-429) ---
def call_gemini_retry(api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            
            elif response.status_code == 429:
                # 遇到 429 錯誤，等待後重試 (Exponential Backoff)
                wait_time = 2 ** (attempt + 1) # 2s, 4s, 8s
                time.sleep(wait_time)
                continue # 重試
                
            else:
                return f"連線錯誤 Code: {response.status_code}"
                
        except Exception as e:
            return f"系統錯誤: {str(e)}"
            
    return "🚨 系統忙碌 (429)：已重試 3 次仍無回應，請稍後再試。"

def ask_gemini_wolf(api_key, name, ticker, df, style, buy_ref, sell_ref, atr, ma200, ema20, trend):
    last = df.iloc[-1]
    
    # 如果 Yahoo 沒抓到名字，直接用 ticker 代替，避免 AI 混亂
    display_name = name if name else ticker
    
    data_summary = f"""
    【標的】{display_name} ({ticker})
    【現價】{last['Close']:.2f}
    【趨勢】{trend} (MA200: {ma200:.2f})
    【動能】RSI: {last['RSI']:.2f}
    
    【狼性交易邊界】
    - 建議買入：{buy_ref:.2f} (若為強勢股，此為 EMA20 支撐)
    - 建議賣出：{sell_ref:.2f}
    - ATR波動：{atr:.2f}
    """
    
    prompt = f"""
    角色：華爾街頂級交易員 (20年經驗)，風格：{style}。
    任務：根據數據給出明確、有自信的交易指令。
    
    ⚠️ **指令**：
    1. **確認標的**：你分析的是 {display_name}。如果名字看起來是代碼，請自行判斷公司。
    2. **狼性思維**：如果是「強勢多頭」，請建議在 {buy_ref:.2f} (EMA20) 附近積極佈局，不要叫人等崩盤。
    3. **拒絕模稜兩可**：直接給出價格和理由。
    
    請撰寫分析報告 (繁體中文)：
    1. 🎯 **市場解讀**：(一句話判斷多空強度)
    2. 🔵 **狙擊買入 (Entry)**：
       - 價格：{buy_ref:.2f}
       - 理由：(例如：回測 EMA20 強力支撐，趨勢未破)
    3. 🔴 **獲利了結 (Exit)**：價格 {sell_ref:.2f}。
    4. 🛡️ **止損防線**：價格 {buy_ref - atr*1.5:.2f}。
    """
    return call_gemini_retry(api_key, prompt)

def ask_gemini_qa(api_key, name, ticker, df, question):
    last = df.iloc[-1]
    prompt = f"角色：專業交易員 Felix。標的：{name} ({ticker}) 現價 {last['Close']}。用戶問：{question}。請簡潔回答。"
    return call_gemini_retry(api_key, prompt)

# --- 主畫面 ---
st.title("🏛️ Felix AI 股票分析員")
st.caption("V34.0 Anti-Fragile | Retry Logic | Smart Search")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ 請輸入 API Key")
    api_key = st.sidebar.text_input("Key", type="password")
else:
    st.sidebar.success("✅ 華爾街專線：Ready")

st.sidebar.header("♟️ 操盤面板")
user_input = st.sidebar.text_input("輸入代碼 (可輸入 GOOGLE, NVDA, 700)", value="GOOGLE")
style = st.sidebar.selectbox("風格", ["趨勢波段 (Trend)", "價值投資 (Value)", "短線當沖 (Day Trade)"])

st.sidebar.markdown("---")
# 功能按鈕區
btn_analyze = st.sidebar.button("🔥 全面分析")

st.sidebar.markdown("---")
qa_input = st.sidebar.text_area("提問 (如：財報後會跌嗎？)", height=80)
btn_ask = st.sidebar.button("💬 提問解答")

if btn_analyze:
    if not api_key: st.error("No API Key")
    else:
        # 1. 智能搜尋 (模擬富途)
        real_ticker = resolve_ticker(user_input)
        st.info(f"🔍 識別代碼：{user_input} ➝ {real_ticker}")
        
        # 2. 抓取數據
        df, name, ticker = get_data_v34(real_ticker)
        
        if df is not None:
            # 3. 狼性運算
            df, buy_ref, sell_ref, atr, ma200, ema20, trend = calculate_wolf_levels(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            st.subheader(f"📊 {name} ({ticker})")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
            c2.metric("趨勢強度", trend)
            c3.metric("建議買點", f"{buy_ref:.2f}")
            c4.metric("強支撐 (EMA20)", f"{ema20:.2f}")

            tab1, tab2 = st.tabs(["🧠 AI 狼性策略", "📈 專業圖表"])
            
            with tab1:
                # 呼叫 AI (含自動重試)
                report = ask_gemini_wolf(api_key, name, ticker, df, style, buy_ref, sell_ref, atr, ma200, ema20, trend)
                if "429" in report:
                    st.warning(report)
                else:
                    st.info(report)
                
            with tab2:
                # 預設顯示 1 年圖表
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='orange', width=1), name='EMA20'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red', width=1, dash='dot'), name='布林上'))
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green', width=1, dash='dot'), name='布林下'))
                fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"❌ 找不到數據：{real_ticker} (請確認代碼)")

if btn_ask:
    if not api_key: st.error("No API Key")
    else:
        real_ticker = resolve_ticker(user_input)
        df, name, ticker = get_data_v34(real_ticker)
        if df is not None:
            st.info(f"🤖 思考中：{qa_input}")
            ans = ask_gemini_qa(api_key, name, ticker, df, qa_input)
            st.success(ans)
        else:
            st.error("找不到數據")
