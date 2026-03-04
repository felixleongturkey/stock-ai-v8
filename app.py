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
    .stWarning { background-color: #1e1e00; color: #ffeb3b; border-left: 5px solid #ffeb3b; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智能代碼偵探 ---
def smart_ticker_search(user_input):
    clean = user_input.strip().upper()
    aliases = {
        "騰訊": "0700.HK", "700": "0700.HK",
        "阿里": "9988.HK", "9988": "9988.HK",
        "美團": "3690.HK", "3690": "3690.HK",
        "小米": "1810.HK", "1810": "1810.HK",
        "比亞迪": "1211.HK", "1211": "1211.HK",
        "匯豐": "0005.HK", "5": "0005.HK", "0005": "0005.HK",
        "港交所": "0388.HK", "388": "0388.HK",
        "台積電": "2330.TW", "2330": "2330.TW",
        "NVDA": "NVDA", "TSLA": "TSLA", "AAPL": "AAPL", "AMD": "AMD",
        "SMCI": "SMCI", "MSTR": "MSTR"
    }
    if clean in aliases: return aliases[clean]
    if clean.isdigit(): return f"{int(clean):04d}.HK"
    return clean

# --- 3. 數據抓取引擎 ---
def get_data_v20(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 強制抓取 2 年，確保有足夠數據算 MA200
        df = stock.history(period="2y")
        
        # 自動容錯機制
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

        try: name = stock.info.get('longName', ticker)
        except: name = ticker
            
        return df, name, ticker
    except:
        return None, None, ticker

# --- 4. 專業指標運算 (防呆機制) ---
def calculate_indicators(df):
    if len(df) < 5: return df # 數據過少直接回傳
    
    # 計算均線 (使用 min_periods=1 防止新股報錯)
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA60'] = df['Close'].rolling(window=60, min_periods=1).mean()
    
    # MA200 必須嚴格，不足 150 天就不算，避免誤導
    if len(df) >= 150:
        df['MA200'] = df['Close'].rolling(window=200, min_periods=150).mean()
    else:
        df['MA200'] = np.nan # 新股標記為 NaN

    # 布林
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

# --- 5. AI 大腦 (自適應新舊股邏輯) ---
def ask_gemini_adaptive(api_key, name, ticker, df, style):
    last = df.iloc[-1]
    
    # 判斷是新股還是老股
    has_ma200 = not pd.isna(last['MA200'])
    
    # 準備數據摘要
    ma200_info = f"{last['MA200']:.2f}" if has_ma200 else "N/A (上市不足200天)"
    
    # 根據是否有 MA200，動態改變給 AI 的指令 (Prompt Engineering)
    if has_ma200:
        # 老股指令：強調牛熊線
        trend_instruction = f"這是一隻成熟股票。請重點分析股價與【MA200 (牛熊線) {ma200_info}】的關係。若股價在 MA200 上方為多頭，下方為空頭。"
    else:
        # 新股指令：強調新股爆發力與短期均線
        trend_instruction = "🚨 注意：這是一隻【次新股/半新股】(上市不足200天)。MA200 參考價值低。請重點分析【MA60 (季線)】以及【歷史新高/新低】的價格發現過程。重點關注籌碼沉澱與短期爆發力。"

    data_summary = f"""
    【標的】{name} ({ticker})
    【技術數據】
    - 現價：{last['Close']:.2f}
    - MA20 (月線)：{last['MA20']:.2f}
    - MA60 (季線)：{last['MA60']:.2f}
    - MA200 (年線)：{ma200_info}
    - RSI(14)：{last['RSI']:.2f}
    - ATR(波動)：{last['ATR']:.2f}
    - 布林上軌：{last['Upper']:.2f} / 下軌：{last['Lower']:.2f}
    """
    
    prompt = f"""
    角色：你是華爾街的避險基金經理，風格是「{style}」。
    
    任務：分析以下股票。
    {trend_instruction}
    
    數據面板：
    {data_summary}
    
    請撰寫【機構級交易計畫】(繁體中文)：
    
    1. 🧠 **【深度邏輯分析】**：
       - 基本面一句話點評 (請動用你的知識庫)。
       - 技術面趨勢判斷 (根據上面指定的均線重點)。
       
    2. 🎯 **【精準買入點 (Buy Limit)】**：
       - 給出具體數字。
       - { "如果目前是新股，請尋找回測 MA20 或 MA60 的支撐點。" if not has_ma200 else "參考 MA200 或黃金分割回調位。" }
       
    3. 🚀 **【獲利目標 (Take Profit)】**：
       - 設定第一/第二目標價。
       
    4. 🛡️ **【動態止損 (Stop Loss)】**：
       - 請使用 ATR ({last['ATR']:.2f}) 來設定止損寬度 (例如 進場價 - 2*ATR)。
       - 這是專業風控的標誌。
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        return f"AI 連線錯誤: {response.text}"
    except Exception as e:
        return f"系統錯誤: {str(e)}"

# --- 主畫面 ---
st.title("🏛️ Felix 華爾街 AI 終端 V20")
st.caption("Adaptive Algorithms | New Stock Detection | Gemini 2.5")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ 請輸入金鑰")
    api_key = st.sidebar.text_input("API Key", type="password")
else:
    st.sidebar.success("✅ 華爾街專線：Connected")

st.sidebar.header("♟️ 戰略中樞")
user_input = st.sidebar.text_input("輸入股票代碼 (如 700, NVDA, SMCI)", value="NVDA")
style = st.sidebar.selectbox("切換操盤風格", ["趨勢狙擊 (Momentum)", "價值投資 (Value)", "逆勢交易 (Contrarian)"])

if st.sidebar.button("🔥 啟動深度運算"):
    if not api_key:
        st.error("Missing API Key")
    else:
        ticker_search = smart_ticker_search(user_input)
        st.info(f"🔍 鎖定標的：{ticker_search} ...")

        with st.spinner('正在從交易所獲取大數據並進行「上市時間」檢測...'):
            df, name, ticker = get_data_v20(ticker_search)
            
            if df is not None:
                df = calculate_indicators(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                # --- 智能檢測：是否為新股 ---
                is_new_stock = pd.isna(last['MA200'])
                
                st.subheader(f"📊 {name} ({ticker})")
                
                # --- 數據看板 ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
                
                # 動態顯示 MA200 狀態
                if is_new_stock:
                    c2.metric("牛熊線 (MA200)", "N/A (新股)")
                else:
                    c2.metric("牛熊線 (MA200)", f"{last['MA200']:.2f}")
                    
                c3.metric("RSI 強弱", f"{last['RSI']:.2f}")
                c4.metric("ATR 波動率", f"{last['ATR']:.2f}")

                if is_new_stock:
                    st.warning("⚠️ 檢測到此為【次新股/半新股】(上市數據不足200天)。AI 將自動切換為「新股爆發模式」，重點分析短期動能。")

                # --- 分頁 ---
                tab1, tab2 = st.tabs(["🧠 AI 機構報告", "📈 專業技術圖"])
                
                with tab1:
                    st.markdown(f"### 📝 {style} 策略報告")
                    ai_reply = ask_gemini_adaptive(api_key, name, ticker, df, style)
                    st.info(ai_reply)

                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
                    if not pd.isna(last['MA20']): fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='yellow', width=1), name='MA20'))
                    if not pd.isna(last['MA60']): fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='cyan', width=1), name='MA60'))
                    
                    # 只有老股才畫 MA200
                    if not is_new_stock: 
                        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='purple', width=2), name='MA200 (牛熊)'))
                        
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"❌ 找不到數據：{ticker_search}")
