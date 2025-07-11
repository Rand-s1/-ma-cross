import requests
import ta
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
import logging
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 页面配置
st.set_page_config(page_title="双MA交叉扫描器", page_icon="📈", layout="wide")

# 简化CSS
st.markdown("""
<style>
.big-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# 配置
class Config:
    ENDPOINTS = ["https://api.bitget.com"]
    PRODUCT_TYPE = "usdt-futures"
    LIMIT = 400
    MAX_WORKERS = 10
    TIMEFRAMES = {"5分钟": "5m", "15分钟": "15m", "30分钟": "30m", "1小时": "1H", "4小时": "4H", "1天": "1D"}
    MA_OPTIONS = [10, 20, 55, 70, 150, 200, 350]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sidebar():
    """侧边栏设置"""
    with st.sidebar:
        st.markdown("### ⚙️ 扫描设置")
        
        timeframe = Config.TIMEFRAMES[st.selectbox("时间框架", list(Config.TIMEFRAMES.keys()), index=3)]
        
        col1, col2 = st.columns(2)
        with col1:
            ma_fast = st.selectbox("快线", Config.MA_OPTIONS, index=1)
        with col2:
            ma_slow = st.selectbox("慢线", Config.MA_OPTIONS, index=3)
        
        signal_types = st.multiselect("信号类型", ["金叉信号", "死叉信号"], default=["金叉信号", "死叉信号"])
        cross_confirm_bars = st.slider("确认周期", 1, 5, 2)
        
        st.markdown("### 🔍 过滤设置")
        enable_filter = st.checkbox("启用K线位置过滤", value=True)
        
        golden_filter = death_filter = "无要求"
        if enable_filter:
            if "金叉信号" in signal_types:
                golden_filter = st.selectbox("金叉过滤", [
                    "无要求", f"K线在MA{ma_fast}上方", f"K线在MA{ma_slow}上方", "K线在双MA上方"
                ], index=3)
            if "死叉信号" in signal_types:
                death_filter = st.selectbox("死叉过滤", [
                    "无要求", f"K线在MA{ma_fast}下方", f"K线在MA{ma_slow}下方", "K线在双MA下方"
                ], index=3)
        
        min_volume = st.number_input("最小成交量", value=0.0)
        
        return timeframe, ma_fast, ma_slow, signal_types, cross_confirm_bars, enable_filter, golden_filter, death_filter, min_volume

def get_api_data(base: str, endpoint: str, params: dict):
    """统一API请求"""
    try:
        r = requests.get(f"{base}{endpoint}", params=params, timeout=10)
        j = r.json()
        return j["data"] if j.get("code") == "00000" else None
    except Exception as e:
        logger.error(f"API请求失败: {e}")
        return None

def get_working_endpoint() -> str:
    """获取可用端点"""
    for ep in Config.ENDPOINTS:
        if get_api_data(ep, "/api/v2/mix/market/candles", {
            "symbol": "BTCUSDT", "granularity": "4H", "limit": 1, "productType": Config.PRODUCT_TYPE
        }):
            return ep
    raise RuntimeError("无可用端点")

def get_symbols_and_tickers(base: str):
    """获取交易对和价格数据"""
    # 获取交易对
    symbols_data = get_api_data(base, "/api/v2/mix/market/contracts", {"productType": Config.PRODUCT_TYPE})
    symbols = [c["symbol"] for c in symbols_data] if symbols_data else []
    
    # 获取价格数据
    tickers_data = get_api_data(base, "/api/v2/mix/market/tickers", {"productType": Config.PRODUCT_TYPE})
    tickers = {}
    if tickers_data:
        for item in tickers_data:
            symbol = item.get("symbol", "")
            if symbol:
                tickers[symbol] = {
                    "change24h": float(item.get("change24h", 0)) * 100,
                    "volume": float(item.get("baseVolume", 0)),
                    "price": float(item.get("close", 0))
                }
    
    return symbols, tickers

def fetch_candles(base: str, symbol: str, granularity: str) -> pd.DataFrame:
    """获取K线数据"""
    data = get_api_data(base, "/api/v2/mix/market/candles", {
        "symbol": symbol, "granularity": granularity, "limit": Config.LIMIT, "productType": Config.PRODUCT_TYPE
    })
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume_base", "volume_quote"])
    df[["open", "high", "low", "close", "volume_base", "volume_quote"]] = df[
        ["open", "high", "low", "close", "volume_base", "volume_quote"]].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    return df.sort_values("ts").reset_index(drop=True)

def calculate_ma_signals(df: pd.DataFrame, ma_fast: int, ma_slow: int, cross_confirm_bars: int, 
                        enable_filter: bool, golden_filter: str, death_filter: str) -> Tuple[Optional[dict], int]:
    """计算MA交叉信号"""
    candle_count = len(df)
    min_needed = max(ma_fast, ma_slow) + 10
    
    if candle_count < min_needed:
        return None, candle_count
    
    close = pd.Series(df["close"]).reset_index(drop=True)
    ma_fast_line = close.rolling(window=ma_fast).mean()
    ma_slow_line = close.rolling(window=ma_slow).mean()
    
    if ma_fast_line.isna().all() or ma_slow_line.isna().all():
        return None, candle_count
    
    signal_info = {
        "ma_fast_current": ma_fast_line.iloc[-1],
        "ma_slow_current": ma_slow_line.iloc[-1],
        "price_current": close.iloc[-1],
        "golden_cross": False,
        "death_cross": False,
        "cross_bars_ago": None,
        "ma_fast_period": ma_fast,
        "ma_slow_period": ma_slow
    }
    
    # 计算价格位置
    signal_info["price_above_fast"] = signal_info["price_current"] > signal_info["ma_fast_current"]
    signal_info["price_above_slow"] = signal_info["price_current"] > signal_info["ma_slow_current"]
    signal_info["ma_distance"] = abs(signal_info["ma_fast_current"] - signal_info["ma_slow_current"]) / signal_info["ma_slow_current"] * 100
    
    # 检测交叉
    for i in range(1, min(cross_confirm_bars + 1, len(ma_fast_line))):
        if pd.isna(ma_fast_line.iloc[-(i+1)]) or pd.isna(ma_slow_line.iloc[-(i+1)]):
            continue
        
        # 金叉
        if (ma_fast_line.iloc[-(i+1)] <= ma_slow_line.iloc[-(i+1)] and ma_fast_line.iloc[-i] > ma_slow_line.iloc[-i]):
            signal_info["golden_cross"] = True
            signal_info["cross_bars_ago"] = i
            break
        # 死叉
        elif (ma_fast_line.iloc[-(i+1)] >= ma_slow_line.iloc[-(i+1)] and ma_fast_line.iloc[-i] < ma_slow_line.iloc[-i]):
            signal_info["death_cross"] = True
            signal_info["cross_bars_ago"] = i
            break
    
    # 应用过滤
    if enable_filter:
        if signal_info["golden_cross"]:
            if golden_filter == f"K线在MA{ma_fast}上方" and not signal_info["price_above_fast"]:
                signal_info["golden_cross"] = False
            elif golden_filter == f"K线在MA{ma_slow}上方" and not signal_info["price_above_slow"]:
                signal_info["golden_cross"] = False
            elif golden_filter == "K线在双MA上方" and not (signal_info["price_above_fast"] and signal_info["price_above_slow"]):
                signal_info["golden_cross"] = False
        
        if signal_info["death_cross"]:
            if death_filter == f"K线在MA{ma_fast}下方" and signal_info["price_above_fast"]:
                signal_info["death_cross"] = False
            elif death_filter == f"K线在MA{ma_slow}下方" and signal_info["price_above_slow"]:
                signal_info["death_cross"] = False
            elif death_filter == "K线在双MA下方" and (signal_info["price_above_fast"] or signal_info["price_above_slow"]):
                signal_info["death_cross"] = False
    
    return signal_info, candle_count

def scan_symbols(base: str, symbols: List[str], tickers: dict, timeframe: str, ma_fast: int, ma_slow: int, 
                signal_types: List[str], cross_confirm_bars: int, enable_filter: bool, 
                golden_filter: str, death_filter: str, min_volume: float):
    """扫描所有交易对"""
    results = []
    
    # 并行获取K线数据
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_candles, base, symbol, timeframe): symbol for symbol in symbols}
        
        progress_bar = st.progress(0)
        processed = 0
        
        for future in as_completed(futures):
            symbol = futures[future]
            df = future.result()
            processed += 1
            progress_bar.progress(processed / len(symbols))
            
            if df.empty:
                continue
            
            signal_info, candle_count = calculate_ma_signals(
                df, ma_fast, ma_slow, cross_confirm_bars, enable_filter, golden_filter, death_filter
            )
            
            if not signal_info:
                continue
            
            ticker_data = tickers.get(symbol, {"change24h": 0, "volume": 0, "price": 0})
            
            if ticker_data["volume"] < min_volume:
                continue
            
            # 检查信号
            for signal_type, has_signal in [("金叉", signal_info["golden_cross"]), ("死叉", signal_info["death_cross"])]:
                if has_signal and f"{signal_type}信号" in signal_types:
                    results.append({
                        "symbol": symbol,
                        "signal_type": signal_type,
                        "change (%)": round(ticker_data["change24h"], 2),
                        "cross_bars_ago": signal_info["cross_bars_ago"],
                        "price_above_fast": signal_info["price_above_fast"],
                        "price_above_slow": signal_info["price_above_slow"],
                        "ma_distance": round(signal_info["ma_distance"], 2),
                        "ma_fast_period": ma_fast,
                        "ma_slow_period": ma_slow
                    })
        
        progress_bar.empty()
    
    return results

def format_results(results: List[dict]) -> pd.DataFrame:
    """格式化结果"""
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    def add_icon(row):
        change = row["change (%)"]
        signal = row["signal_type"]
        icon = "🚀🟢" if change > 5 else "📈🟢" if change > 0 else "📉🟢" if signal == "金叉" else "💥🔴" if change < -5 else "📉🔴" if change < 0 else "📈🔴"
        return f"{icon} {row['symbol']}"
    
    def get_direction(row):
        arrow = "↗️" if row["signal_type"] == "金叉" else "↘️"
        return f"MA{row['ma_fast_period']} {arrow} MA{row['ma_slow_period']}"
    
    def get_position(row):
        if row["price_above_fast"] and row["price_above_slow"]:
            return "双线上方"
        elif not row["price_above_fast"] and not row["price_above_slow"]:
            return "双线下方"
        elif row["price_above_fast"]:
            return f"MA{row['ma_fast_period']}上方"
        else:
            return f"MA{row['ma_slow_period']}上方"
    
    df_formatted = df.copy()
    df_formatted["交易对"] = df.apply(add_icon, axis=1)
    df_formatted["穿越方向"] = df.apply(get_direction, axis=1)
    df_formatted["24h涨跌"] = df_formatted["change (%)"].apply(lambda x: f"{x:+.2f}%")
    df_formatted["交叉时间"] = df_formatted["cross_bars_ago"].apply(lambda x: f"{x}根K线前")
    df_formatted["价格位置"] = df.apply(get_position, axis=1)
    df_formatted["MA距离"] = df_formatted["ma_distance"].apply(lambda x: f"{x:.2f}%")
    
    return df_formatted[["交易对", "穿越方向", "24h涨跌", "交叉时间", "价格位置", "MA距离"]]

def main():
    # 页面头部
    st.markdown('<h1 class="big-title">📈 双MA交叉扫描器</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 侧边栏
    timeframe, ma_fast, ma_slow, signal_types, cross_confirm_bars, enable_filter, golden_filter, death_filter, min_volume = create_sidebar()
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        scan_button = st.button("🚀 开始扫描", use_container_width=True)
        
        with st.expander("📋 当前设置"):
            st.write(f"**时间框架**: {timeframe}")
            st.write(f"**MA设置**: {ma_fast} × {ma_slow}")
            st.write(f"**信号类型**: {', '.join(signal_types)}")
    
    with col1:
        if not scan_button:
            st.markdown(f"""
            ### 🎯 使用指南
            
            **双MA交叉扫描器** - 检测移动平均线交叉信号
            
            #### 📊 当前设置：
            - **快线**: MA{ma_fast} | **慢线**: MA{ma_slow}
            - **时间级别**: {timeframe}
            
            #### 🎯 交易信号：
            - 🟢 **金叉**: MA{ma_fast} ↗️ MA{ma_slow} (看涨)
            - 🔴 **死叉**: MA{ma_fast} ↘️ MA{ma_slow} (看跌)
            
            #### 🚀 开始使用：
            点击右侧"开始扫描"按钮开始分析
            """)
            return
    
    if scan_button:
        if not signal_types:
            st.error("❌ 请选择至少一种信号类型")
            return
        
        if ma_fast >= ma_slow:
            st.error("❌ 快线周期应小于慢线周期")
            return
        
        try:
            # 获取数据
            with st.spinner("🔗 连接API..."):
                base = get_working_endpoint()
            
            with st.spinner("📋 获取市场数据..."):
                symbols, tickers = get_symbols_and_tickers(base)
                st.success(f"✅ 找到 {len(symbols)} 个交易对")
            
            # 扫描
            with st.spinner("🔍 扫描交叉信号..."):
                results = scan_symbols(base, symbols, tickers, timeframe, ma_fast, ma_slow, 
                                     signal_types, cross_confirm_bars, enable_filter, 
                                     golden_filter, death_filter, min_volume)
            
            st.success(f"✅ 扫描完成! 找到 {len(results)} 个信号")
            
            # 显示结果
            if results:
                golden = [r for r in results if r["signal_type"] == "金叉"]
                death = [r for r in results if r["signal_type"] == "死叉"]
                
                if golden and "金叉信号" in signal_types:
                    st.markdown(f"### 🟢 金叉信号 ({len(golden)}个)")
                    formatted_golden = format_results(golden)
                    st.dataframe(formatted_golden, use_container_width=True, hide_index=True)
                
                if death and "死叉信号" in signal_types:
                    st.markdown(f"### 🔴 死叉信号 ({len(death)}个)")
                    formatted_death = format_results(death)
                    st.dataframe(formatted_death, use_container_width=True, hide_index=True)
            else:
                st.info("🤔 当前没有符合条件的交叉信号")
                
        except Exception as e:
            st.error(f"❌ 扫描失败: {str(e)}")

if __name__ == "__main__":
    main()
