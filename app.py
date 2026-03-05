import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np
import datetime

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Felix AI 股票分析員", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    h1 { color: #00d2ff; font-family: 'Helvetica Neue', sans-serif; font-weight: 900; }
    .stMetric { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    .stMetric label { color: #888; }
    .stMetric div[data-testid="stMetricValue"] { color: #fff; font-size: 26px; font-weight: bold; }
    
    /* 按鈕美化 */
    .stButton>button { 
        width: 100%; background: linear-gradient(90deg, #0072ff, #00c6ff); 
        color: white; font-weight: bold; border: none; height: 50px; font-size: 16px; border-radius: 8px;
    }
    .stButton>button:hover { box-shadow: 0 0 15px rgba(0, 198, 255, 0.5); }
    
    /* 提問按鈕獨立顏色 */
    div[data-testid="stSidebar"] .stButton:nth-of-type(2) button {
        background: linear-gradient(90deg, #11998e, #38ef7d); color: black;
    }
    
    .stTextArea textarea { background-color: #111; color: #fff; border: 1px solid #444; }
    
    /* 圖表按鈕區塊微調 */
    div[data-testid="stHorizontalBlock"] button {
        background: #222; border: 1px solid #444; color: #ddd; height: 40px; font-size: 14px;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background: #444; color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智能代碼與名稱修正 ---
def smart_ticker_search(user_input):
    clean = user_input.strip().upper()
    if clean.isdigit(): return f"{int(clean):04d}.HK"
    if clean.isalpha() and len(clean) <= 5: return clean
    return clean

def get_fixed_name(ticker):
    # 強制修正表
    FIXED_NAMES = {
        "0697.HK": "首程控股 (Shoucheng Holdings)",
        "0694.HK": "北京首都機場", "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴",
        "3690.HK": "美團", "1810.HK": "小米集團", "1211.HK": "比亞迪股份",
        "0005.HK": "匯豐控股", "0388.HK": "香港交易所", "NVDA": "NVIDIA",
        "TSLA": "Tesla", "AAPL": "Apple", "MSFT": "Microsoft", "AMD": "AMD"
    }
    return FIXED_NAMES.get(ticker, None)

# --- 3. 數據抓取 (分為「AI分析用」與「圖表用」) ---

# A. 獲取 AI 分析用的大數據 (強制抓取 2年以上)
def get_data_for_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 抓取 5年數據，確保 AI 能看到長期趨勢 (牛熊線、歷史高低)
        df = stock.history(period="5y")
        
        # 容錯機制
        if df.empty:
            if ticker.endswith(".HK"):
                alt = ticker.replace(".HK", "")
                stock = yf.Ticker(alt)
                df = stock.history(period="5y")
                if not df.empty: ticker = alt
            elif ticker.isdigit():
                alt = f"{int(ticker):04d}.HK"
                stock = yf.Ticker(alt)
                df = stock.history(period="5y")
                if not df.empty: ticker = alt
        
        if df.empty: return None, None, ticker

        fixed_name = get_fixed_name(ticker)
        name = fixed_name if fixed_name else stock.info.get('longName', ticker)
            
        return df, name, ticker
    except:
        return None, None, ticker

# B. 獲取圖表用的數據 (根據按鈕切換)
def get_data_for_chart(ticker, timeframe):
    stock = yf.Ticker(ticker)
    
    # 根據用戶選擇，抓取不同精細度的數據
    if timeframe == "1日":
        return stock.history(period="1d", interval="5m")
    elif timeframe == "5日":
        return stock.history(period="5d", interval="15m")
    elif timeframe == "1個月":
        return stock.history(period="1mo", interval="1d")
    elif timeframe == "3個月":
        return stock.history(period="3mo", interval="1d")
    elif timeframe == "6個月":
        return stock.history(period="6mo", interval="1d")
    elif timeframe == "1年":
        return stock.history(period="1y", interval="1d")
    elif timeframe == "2年":
        return stock.history(period="2y", interval="1d")
    else: # 全部
        return stock.history(period="max", interval="1d")

# --- 4. 專業數學運算 (優化版：尋找「最近」的支撐) ---
def calculate_levels_pro(df):
    if len(df) < 50: return df, 0, 0, 0, 0, 0
    
    # 計算長期均線 (因為我們現在有 5年數據了)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean() # 牛熊分界
    
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

    last = df.iloc[-1]
    
    # --- 關鍵修正：智能支撐尋找演算法 ---
    # 我們不找 5 年前的低點，我們找「近半年」的波段
    recent_df = df.tail(120) # 近半年
    recent_high = recent_df['High'].max()
    recent_low = recent_df['Low'].min()
    
    # 黃金分割 0.382 & 0.5 (強勢回調位)
    fibo_0382 = recent_high - (recent_high - recent_low) * 0.382
    fibo_0500 = recent_high - (recent_high - recent_low) * 0.500
    
    # 樞軸點 (Pivot)
    P = (last['High'] + last['Low'] + last['Close']) / 3
    S1 = (2 * P) - last['High']
    R1 = (2 * P) - last['Low']

    # ATR 止損
    high_low = df['High'] - df['Low']
    true_range = np.max(pd.concat([high_low, np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1), axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]

    # 決定「最佳買入參考價」
    # 邏輯：在 S1, Fibo 0.382, MA60, 布林下軌 中，找出「最接近現價但低於現價」的那個，作為支撐
    candidates = [S1, fibo_0382, last['Lower'], last['MA60']]
    # 過濾掉太低的離譜價格，只保留合理的支撐 (例如現價的 85% 以上)
    valid_candidates = [x for x in candidates if x < last['Close'] and x > last['Close'] * 0.8]
    
    if valid_candidates:
        buy_ref = max(valid_candidates) # 取最靠近現價的支撐
    else:
        buy_ref = last['MA20'] # 如果都沒有，回歸月線支撐

    # 決定「最佳賣出參考價」
    sell_candidates = [R1, last['Upper'], recent_high]
    valid_sells = [x for x in sell_candidates if x > last['Close']]
    
    if valid_sells:
        sell_ref = min(valid_sells) # 取最靠近現價的壓力
    else:
        sell_ref = last['Close'] * 1.05 # 如果都破新高了，設 5% 獲利

    return df, buy_ref, sell_ref, fibo_0382, atr, last['MA200']

# --- 5. AI 核心 (Gemini 2.5 Flash) ---
def call_gemini(api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200: return response.json()['candidates'][0]['content']['parts'][0]['text']
        return f"API Error: {response.status_code}"
    except Exception as e: return str(e)

def ask_gemini_strategy(api_key, name, ticker, df, style, buy_ref, sell_ref, atr, ma200):
    last = df.iloc[-1]
    
    # 判斷長期趨勢
    trend_long = "多頭格局 (股價 > 年線)" if not pd.isna(ma200) and last['Close'] > ma200 else "空頭格局 (股價 < 年線)"
    if pd.isna(ma200): trend_long = "新股 (無年線數據)"

    data_summary = f"""
    【標的】{name} ({ticker})
    【分析區間】已分析該股過去 2~5 年的完整走勢
    【現價】{last['Close']:.2f}
    【長期趨勢】{trend_long} (MA200: {ma200:.2f})
    【RSI】{last['RSI']:.2f}
    
    【系統運算的精準位階 (請嚴格執行)】
    - 精算買入點：{buy_ref:.2f} (依據：近期波段支撐/均線)
    - 精算賣出點：{sell_ref:.2f} (依據：波段壓力/前高)
    - ATR波動值：{atr:.2f}
    """
    
    prompt = f"""
    角色：華爾街資深基金經理，風格：{style}。
    任務：利用大數據分析結果，給出精準交易建議。
    
    ⚠️ **最高指令**：
    1. **準確度**：買入價請直接使用我計算好的 {buy_ref:.2f}，不要隨意更改。
    2. **專業度**：請解釋為什麼這個位置重要 (例如：這是波段回調的黃金支撐)。
    3. **拒絕廢話**：直接給出數字和理由。
    
    請撰寫報告：
    1. 🎯 **宏觀趨勢**：結合 MA200 判斷目前是大牛市還是熊市反彈？
    2. 🔵 **精準買入掛單**：價格 {buy_ref:.2f}，理由？
    3. 🔴 **精準獲利掛單**：價格 {sell_ref:.2f}，理由？
    4. 🛡️ **止損防線**：設定在買入價下方 {atr*1.5:.2f} (1.5倍 ATR)。
    """
    return call_gemini(api_key, prompt)

def ask_gemini_qa(api_key, name, ticker, df, question):
    last = df.iloc[-1]
    prompt = f"""
    角色：Felix AI 首席分析師。
    標的：{name} ({ticker})，現價：{last['Close']:.2f}。
    用戶提問：「{question}」
    請針對此股票的技術面與基本面，專業回答用戶問題。
    """
    return call_gemini(api_key, prompt)

# --- 主畫面 ---
st.title("🏛️ Felix AI 股票分析員")
st.caption("V32.0 Big Data | Multi-Timeframe Chart | Gemini 2.5")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("⚠️ 請輸入 API Key")
    api_key = st.sidebar.text_input("Key", type="password")
else:
    st.sidebar.success("✅ 華爾街專線：Connected")

st.sidebar.header("♟️ 操盤設定")
user_input = st.sidebar.text_input("輸入代碼 (如 0697, NVDA)", value="NVDA")
style = st.sidebar.selectbox("分析風格", ["趨勢波段 (Trend)", "價值回歸 (Value)", "短線當沖 (Day Trade)"])

st.sidebar.markdown("---")
# 雙按鈕功能區
col1, col2 = st.sidebar.columns(2)
btn_analyze = st.sidebar.button("🔥 全面分析")

st.sidebar.markdown("---")
qa_input = st.sidebar.text_area("提問 (例如：這隻能存股嗎？)", height=80)
btn_ask = st.sidebar.button("💬 提問解答")

# --- 邏輯處理 ---
if btn_analyze:
    if not api_key: st.error("請輸入 Key")
    else:
        ticker_search = smart_ticker_search(user_input)
        st.info(f"🔍 正在調取 {ticker_search} 近 5 年歷史大數據...")
        
        # 1. 抓取 AI 分析用的長線數據 (Daily, 5y)
        df_analysis, name, ticker = get_data_for_analysis(ticker_search)
        
        if df_analysis is not None:
            # 2. 計算精準位階
            df_analysis, buy_ref, sell_ref, fibo, atr, ma200 = calculate_levels_pro(df_analysis)
            last = df_analysis.iloc[-1]
            prev = df_analysis.iloc[-2]

            st.subheader(f"📊 {name} ({ticker})")
            
            # 頂部數據卡片
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
            c2.metric("智能買點", f"{buy_ref:.2f}")
            c3.metric("智能賣點", f"{sell_ref:.2f}")
            c4.metric("長期趨勢 (MA200)", "多頭" if last['Close'] > ma200 else "空頭")

            # 分頁
            tab1, tab2 = st.tabs(["🧠 AI 深度報告", "📈 專業互動圖表"])
            
            with tab1:
                report = ask_gemini_strategy(api_key, name, ticker, df_analysis, style, buy_ref, sell_ref, atr, ma200)
                st.info(report)
            
            with tab2:
                # 圖表控制區
                st.write("⏱️ **選擇圖表週期：**")
                # 使用 columns 讓按鈕排一排
                t1, t2, t3, t4, t5, t6, t7 = st.columns(7)
                timeframe = "1年" # 預設
                
                # 這裡的按鈕邏輯：Streamlit 按鈕按下會重整，所以通常要配合 Session State
                # 但為了保持代碼簡潔，我們這裡做一個簡單的 selectbox 或者 radio 可能更好，但你要求按鈕。
                # 為了穩定性，我們用 Radio 但做成橫向 (pills) 樣式
                tf_selected = st.radio("選擇時間範圍", ["1日", "5日", "1個月", "6個月", "1年", "2年", "全部"], index=4, horizontal=True)
                
                # 根據選擇抓取對應數據畫圖
                chart_df = get_data_for_chart(ticker, tf_selected)
                
                if chart_df is not None and not chart_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='K線'))
                    
                    # 只有在日線級別以上才畫均線，避免分時圖混亂
                    if tf_selected not in ["1日", "5日"]:
                         fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Close'].rolling(20).mean(), line=dict(color='yellow', width=1), name='MA20'))
                    
                    fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False, title=f"{ticker} - {tf_selected} 走勢")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("該週期暫無數據")

        else:
            st.error("找不到股票數據")

if btn_ask:
    if not api_key or not qa_input: st.error("請輸入 Key 和 問題")
    else:
        ticker_search = smart_ticker_search(user_input)
        # 提問不需要抓 5年，抓 1年夠了
        df_qa, name, ticker = get_data_for_analysis(ticker_search) 
        if df_qa is not None:
            st.info(f"🤖 Felix 正在思考您的問題：{qa_input}")
            ans = ask_gemini_qa(api_key, name, ticker, df_qa, qa_input)
            st.markdown(f"### 💬 回答：")
            st.success(ans)
        else:
            st.error("找不到數據")
