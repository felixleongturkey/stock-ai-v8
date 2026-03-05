import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import numpy as np

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Felix AI 股票分析員", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #c9d1d9; }
    h1 { 
        background: linear-gradient(to right, #00f260, #0575e6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; }
    .stMetric label { color: #8b949e; }
    .stMetric div[data-testid="stMetricValue"] { color: #fff; font-size: 26px; font-weight: bold; }
    
    /* 主按鈕：狼性綠 */
    .stButton>button { 
        width: 100%; background: linear-gradient(90deg, #11998e, #38ef7d); 
        color: black; font-weight: bold; border: none; height: 55px; font-size: 18px; border-radius: 8px;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 15px #38ef7d; }
    
    /* 提問按鈕：科技藍 */
    div[data-testid="stSidebar"] .stButton:nth-of-type(2) button {
        background: linear-gradient(90deg, #2193b0, #6dd5ed); color: black;
    }
    
    .stTextArea textarea { background-color: #0d1117; color: #fff; border: 1px solid #30363d; }
    .stRadio div { color: #fff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 超級代碼翻譯機 (解決找不到 GOOGLE 的問題) ---
def resolve_ticker(user_input):
    clean = user_input.strip().upper()
    
    # 常見名稱對照表 (User 輸入名稱 -> 轉換為正確代碼)
    MAPPING = {
        "GOOGLE": "GOOG", "GOOGL": "GOOG", "ALPHABET": "GOOG",
        "TESLA": "TSLA", "TSLA": "TSLA",
        "APPLE": "AAPL", "AAPL": "AAPL",
        "MICROSOFT": "MSFT", "MSFT": "MSFT",
        "NVIDIA": "NVDA", "NVDA": "NVDA",
        "AMAZON": "AMZN", "AMZN": "AMZN",
        "META": "META", "FACEBOOK": "META",
        "NETFLIX": "NFLX",
        "AMD": "AMD", "INTEL": "INTC", "TSM": "TSM", "TSMC": "TSM",
        "COINBASE": "COIN", "MSTR": "MSTR",
        "TENCENT": "0700.HK", "騰訊": "0700.HK", "700": "0700.HK",
        "ALIBABA": "9988.HK", "BABA": "9988.HK", "阿里": "9988.HK", "9988": "9988.HK",
        "MEITUAN": "3690.HK", "美團": "3690.HK", "3690": "3690.HK",
        "XIAOMI": "1810.HK", "小米": "1810.HK", "1810": "1810.HK",
        "BYD": "1211.HK", "比亞迪": "1211.HK", "1211": "1211.HK",
        "HSBC": "0005.HK", "匯豐": "0005.HK", "5": "0005.HK",
        "HKEX": "0388.HK", "港交所": "0388.HK", "388": "0388.HK",
        "SHOUCHENG": "0697.HK", "首程": "0697.HK", "0697": "0697.HK",
        "SMCI": "SMCI", "PLTR": "PLTR"
    }
    
    if clean in MAPPING:
        return MAPPING[clean]
    
    # 港股邏輯
    if clean.isdigit(): 
        return f"{int(clean):04d}.HK"
    
    # 預設直接回傳
    return clean

# --- 3. 數據抓取 ---
def get_data_v33(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 抓取 3 年數據，足夠計算長期均線，也不會太慢
        df = stock.history(period="3y")
        
        # 容錯：如果找不到，嘗試切換後綴
        if df.empty:
            if ticker.endswith(".HK"):
                alt = ticker.replace(".HK", "")
                stock = yf.Ticker(alt)
                df = stock.history(period="3y")
                if not df.empty: ticker = alt
            elif ticker.isdigit():
                alt = f"{int(ticker):04d}.HK"
                stock = yf.Ticker(alt)
                df = stock.history(period="3y")
                if not df.empty: ticker = alt
        
        if df.empty: return None, None, ticker

        # 獲取名稱 (如果字典裡有就用字典的，沒有就抓 Yahoo 的)
        name = ticker
        try: 
            name = stock.info.get('longName', ticker)
        except: pass
            
        return df, name, ticker
    except:
        return None, None, ticker

def get_chart_data(ticker, period_str):
    # 處理圖表週期的轉換
    p_map = {
        "1天": ("1d", "5m"), "5天": ("5d", "15m"), 
        "1週": ("5d", "30m"), "1月": ("1mo", "1d"), 
        "3月": ("3mo", "1d"), "6月": ("6mo", "1d"), 
        "1年": ("1y", "1d"), "3年": ("3y", "1d"), "全部": ("max", "1d")
    }
    p, i = p_map.get(period_str, ("1y", "1d"))
    return yf.Ticker(ticker).history(period=p, interval=i)

# --- 4. 狼性數學運算 (Aggressive Math) ---
def calculate_wolf_levels(df):
    if len(df) < 60: return df, 0, 0, 0, 0, 0, "盤整"
    
    # 均線
    df['MA20'] = df['Close'].rolling(window=20).mean() # 月線
    df['MA60'] = df['Close'].rolling(window=60).mean() # 季線
    df['MA200'] = df['Close'].rolling(window=200).mean() # 牛熊線
    
    # EMA (指數移動平均) - 反應更快，適合強勢股
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # 布林
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (2 * df['STD20'])
    df['Lower'] = df['MA20'] - (2 * df['STD20'])
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / (-df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))

    last = df.iloc[-1]
    
    # 判斷趨勢強度
    trend_status = "盤整"
    if last['Close'] > last['MA60'] and last['MA20'] > last['MA60']:
        trend_status = "強勢多頭 🔥"
    elif last['Close'] < last['MA60']:
        trend_status = "弱勢空頭 ❄️"

    # --- 關鍵：決定買入點 (不再只看地板) ---
    
    # 1. 積極買點 (Aggressive)：針對 NVDA 這種強勢股
    # 如果是強勢多頭，回踩 EMA20 或 MA20 就是買點，不用等布林下軌
    aggressive_buy = last['EMA20']
    
    # 2. 保守買點 (Conservative)：針對震盪股
    # 尋找近期波段的 0.618 或 布林下軌
    recent_high = df['High'].tail(60).max()
    recent_low = df['Low'].tail(60).min()
    fibo_support = recent_high - (recent_high - recent_low) * 0.382 # 強勢回調只回 0.382
    conservative_buy = max(last['Lower'], fibo_support) # 取較高的支撐
    
    # 智能選擇：如果是強勢多頭，AI 傾向推薦積極買點
    final_buy_ref = aggressive_buy if "強勢" in trend_status else conservative_buy

    # --- 決定賣出點 ---
    # 積極獲利：布林上軌
    # 瘋狂獲利：布林上軌 + 1 ATR
    ranges = df['High'] - df['Low']
    atr = ranges.rolling(14).mean().iloc[-1]
    final_sell_ref = max(last['Upper'], last['Close'] + 2*atr)

    return df, final_buy_ref, final_sell_ref, atr, last['MA200'], last['EMA20'], trend_status

# --- 5. AI 核心 (狼性 Prompt) ---
def call_gemini(api_key, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200: return response.json()['candidates'][0]['content']['parts'][0]['text']
        return f"Error {response.status_code}"
    except Exception as e: return str(e)

def ask_gemini_wolf(api_key, name, ticker, df, style, buy_ref, sell_ref, atr, ma200, ema20, trend):
    last = df.iloc[-1]
    
    data_summary = f"""
    【標的】{name} ({ticker})
    【現價】{last['Close']:.2f}
    【市場狀態】{trend} (MA200: {ma200:.2f})
    【動能指標】EMA20(短線生命線): {ema20:.2f}, RSI: {last['RSI']:.2f}
    
    【系統計算的交易邊界】
    - 建議買入價：{buy_ref:.2f} (依據：{trend} 下的動態支撐)
    - 建議賣出價：{sell_ref:.2f} (依據：壓力位延伸)
    - ATR波動：{atr:.2f}
    """
    
    prompt = f"""
    角色：華爾街擁有 20 年經驗的頂級交易員。風格：{style}。
    心態：自信、果斷、專業。不要使用「可能、也許」這種不確定的詞彙。
    
    任務：給出明確的交易指令。
    
    ⚠️ **針對用戶投訴的回應**：
    用戶之前抱怨買點太保守 (太低買不到)。
    請注意：如果現在是「強勢多頭」，請大膽建議在 EMA20 ({ema20:.2f}) 或 MA20 附近介入，**不要** 叫用戶等到崩盤才買。
    
    請撰寫分析報告 (繁體中文)：
    1. 🎯 **市場解讀**：直接判斷現在是主升段還是盤整？
    2. 🔵 **狙擊買入 (Entry)**：
       - 價格：{buy_ref:.2f} (請引用此數值)
       - 理由：(強調為什麼這裡進場風險回報比最好？如果是強勢股，強調「強勢回調不破月線」的邏輯)
    3. 🔴 **獲利了結 (Exit)**：價格 {sell_ref:.2f}。
    4. 🛡️ **鐵血止損 (Stop)**：價格 {buy_ref - atr*1.5:.2f}。
    """
    return call_gemini(api_key, prompt)

def ask_gemini_qa(api_key, name, ticker, df, question):
    last = df.iloc[-1]
    prompt = f"角色：專業交易員 Felix。標的：{name} ({ticker}) 現價 {last['Close']}。用戶問：{question}。請簡潔專業回答。"
    return call_gemini(api_key, prompt)

# --- 主畫面 ---
st.title("🏛️ Felix AI 股票分析員")
st.caption("V33.0 Wolf Edition | Smart Search | Aggressive Algo")

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
col1, col2 = st.sidebar.columns(2)
btn_analyze = st.sidebar.button("🔥 全面分析")

st.sidebar.markdown("---")
qa_input = st.sidebar.text_area("提問 (如：現在能追嗎？)", height=80)
btn_ask = st.sidebar.button("💬 提問解答")

if btn_analyze:
    if not api_key: st.error("No API Key")
    else:
        # 1. 智能代碼轉換
        real_ticker = resolve_ticker(user_input)
        st.info(f"🔍 識別代碼：{user_input} ➝ {real_ticker}")
        
        # 2. 抓取數據 (3年)
        df, name, ticker = get_data_v33(real_ticker)
        
        if df is not None:
            # 3. 狼性運算
            df, buy_ref, sell_ref, atr, ma200, ema20, trend = calculate_wolf_levels(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            st.subheader(f"📊 {name} ({ticker})")
            
            # 數據卡片
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{last['Close']:.2f}", f"{(last['Close']-prev['Close']):.2f}")
            c2.metric("趨勢狀態", trend)
            c3.metric("積極買點", f"{buy_ref:.2f}")
            c4.metric("EMA20 (強支撐)", f"{ema20:.2f}")

            tab1, tab2 = st.tabs(["🧠 AI 狼性策略", "📈 專業圖表"])
            
            with tab1:
                report = ask_gemini_wolf(api_key, name, ticker, df, style, buy_ref, sell_ref, atr, ma200, ema20, trend)
                st.info(report)
                
            with tab2:
                # 圖表週期選擇
                t_options = ["1天", "5天", "1週", "1月", "3月", "6月", "1年", "3年", "全部"]
                t_sel = st.radio("週期：", t_options, index=6, horizontal=True)
                
                chart_df = get_chart_data(ticker, t_sel)
                if not chart_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='K線'))
                    
                    # 只在日線級別畫均線
                    if t_sel not in ["1天", "5天", "1週"]:
                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Close'].ewm(span=20).mean(), line=dict(color='orange', width=1), name='EMA20'))
                        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Close'].rolling(60).mean(), line=dict(color='cyan', width=1), name='MA60'))
                    
                    fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False, title=f"{ticker} ({t_sel})")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"❌ 找不到數據：{real_ticker} (請確認代碼)")

if btn_ask:
    if not api_key: st.error("No API Key")
    else:
        real_ticker = resolve_ticker(user_input)
        df, name, ticker = get_data_v33(real_ticker)
        if df is not None:
            st.info(f"🤖 思考中：{qa_input}")
            ans = ask_gemini_qa(api_key, name, ticker, df, qa_input)
            st.success(ans)
        else:
            st.error("找不到數據")
