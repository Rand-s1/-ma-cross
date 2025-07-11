import requests
import ta
import pandas as pd
import streamlit as st
from datetime import datetime
import logging
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 页面配置
st.set_page_config(page_title="双MA交叉扫描器", page_icon="📈", layout="wide")

# 配置
class Config:
    ENDPOINTS = ["https://api.bitget.com"]
    PRODUCT_TYPE = "usdt-futures"
    LIMIT = 200  # 减少到200根K线，提高准确性
    MAX_WORKERS = 8
    TIMEFRAMES = {"5分钟": "5m", "15分钟": "15m", "30分钟": "30m", "1小时": "1H", "4小时": "4H", "1天": "1D"}
    MA_OPTIONS = [10, 20, 55, 70, 150, 200, 350]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sidebar():
    """侧边栏设置"""
    with st.sidebar:
        st.markdown("### ⚙️ 扫描设置")
        
        timeframe = Config.TIMEFRAMES[st.selectbox("时间框架", list(Config.TIMEFRAMES.keys()), index=1)]  # 默认15分钟
        
        col1, col2 = st.columns(2)
        with col1:
            ma_fast = st.selectbox("快线", Config.MA_OPTIONS, index=0)  # 默认10
        with col2:
            ma_slow = st.selectbox("慢线", Config.MA_OPTIONS, index=1)  # 默认20
        
        signal_types = st.multiselect("信号类型", ["金叉信号", "死叉信号"], default=["金叉信号", "死叉信号"])
        cross_confirm_bars = st.slider("确认周期", 1, 3, 1)  # 改为默认1，减少误判
        
        st.markdown("### 🔍 过滤设置")
        enable_filter = st.checkbox("启用K线位置过滤", value=False)  # 默认关闭，先看原始信号
        
        golden_filter = death_filter = "无要求"
        if enable_filter:
            if "金叉信号" in signal_types:
                golden_filter = st.selectbox("金叉过滤", [
                    "无要求", f"K线在MA{ma_fast}上方", f"K线在MA{ma_slow}上方", "K线在双MA上方"
                ], index=0)
            if "死叉信号" in signal_types:
                death_filter = st.selectbox("死叉过滤", [
                    "无要求", f"K线在MA{ma_fast}下方", f"K线在MA{ma_slow}下方", "K线在双MA下方"
                ], index=0)
        
        min_volume = st.number_input("最小成交量", value=0.0)
        show_debug = st.checkbox("显示调试信息", value=False)
        
        return timeframe, ma_fast, ma_slow, signal_types, cross_confirm_bars, enable_filter, golden_filter, death_filter, min_volume, show_debug

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
            "symbol": "BTCUSDT", "granularity": "15m", "limit": 5, "productType": Config.PRODUCT_TYPE
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
    
    # 确保按时间正序排列
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def calculate_ma_signals(df: pd.DataFrame, ma_fast: int, ma_slow: int, cross_confirm_bars: int, 
                        enable_filter: bool, golden_filter: str, death_filter: str, show_debug: bool = False) -> Tuple[Optional[dict], int]:
    """计算MA交叉信号 - 修正版本"""
    candle_count = len(df)
    min_needed = max(ma_fast, ma_slow) + 5  # 减少最小需求
    
    if candle_count < min_needed:
        return None, candle_count
    
    # 计算MA线
    close = df["close"].astype(float)
    ma_fast_line = close.rolling(window=ma_fast, min_periods=ma_fast).mean()
    ma_slow_line = close.rolling(window=ma_slow, min_periods=ma_slow).mean()
    
    # 检查是否有足够的有效数据
    valid_data_count = (~(ma_fast_line.isna() | ma_slow_line.isna())).sum()
    if valid_data_count < 10:  # 至少需要10个有效数据点
        return None, candle_count
    
    # 获取最新的有效数据
    current_idx = len(df) - 1
    while current_idx >= 0 and (pd.isna(ma_fast_line.iloc[current_idx]) or pd.isna(ma_slow_line.iloc[current_idx])):
        current_idx -= 1
    
    if current_idx < ma_slow:  # 确保有足够的历史数据
        return None, candle_count
    
    signal_info = {
        "ma_fast_current": ma_fast_line.iloc[current_idx],
        "ma_slow_current": ma_slow_line.iloc[current_idx],
        "price_current": close.iloc[current_idx],
        "golden_cross": False,
        "death_cross": False,
        "cross_bars_ago": None,
        "ma_fast_period": ma_fast,
        "ma_slow_period": ma_slow,
        "current_time": df["ts"].iloc[current_idx]
    }
    
    # 计算价格位置
    signal_info["price_above_fast"] = signal_info["price_current"] > signal_info["ma_fast_current"]
    signal_info["price_above_slow"] = signal_info["price_current"] > signal_info["ma_slow_current"]
    signal_info["ma_distance"] = abs(signal_info["ma_fast_current"] - signal_info["ma_slow_current"]) / signal_info["ma_slow_current"] * 100
    
    # 检测交叉 - 修正逻辑
    cross_detected = False
    for i in range(1, min(cross_confirm_bars + 1, current_idx)):
        prev_idx = current_idx - i
        curr_idx = current_idx - i + 1
        
        # 确保两个点的MA都有效
        if (pd.isna(ma_fast_line.iloc[prev_idx]) or pd.isna(ma_slow_line.iloc[prev_idx]) or
            pd.isna(ma_fast_line.iloc[curr_idx]) or pd.isna(ma_slow_line.iloc[curr_idx])):
            continue
        
        ma_fast_prev = ma_fast_line.iloc[prev_idx]
        ma_slow_prev = ma_slow_line.iloc[prev_idx]
        ma_fast_curr = ma_fast_line.iloc[curr_idx]
        ma_slow_curr = ma_slow_line.iloc[curr_idx]
        
        # 金叉检测：快线从下方穿越到上方
        if ma_fast_prev <= ma_slow_prev and ma_fast_curr > ma_slow_curr:
            signal_info["golden_cross"] = True
            signal_info["cross_bars_ago"] = i
            cross_detected = True
            if show_debug:
                st.write(f"🟢 {df['symbol'].iloc[0] if 'symbol' in df.columns else 'Unknown'} 金叉检测: {i}根K线前")
                st.write(f"   前一根: MA{ma_fast}={ma_fast_prev:.4f}, MA{ma_slow}={ma_slow_prev:.4f}")
                st.write(f"   当前根: MA{ma_fast}={ma_fast_curr:.4f}, MA{ma_slow}={ma_slow_curr:.4f}")
            break
        
        # 死叉检测：快线从上方穿越到下方
        elif ma_fast_prev >= ma_slow_prev and ma_fast_curr < ma_slow_curr:
            signal_info["death_cross"] = True
            signal_info["cross_bars_ago"] = i
            cross_detected = True
            if show_debug:
                st.write(f"🔴 {df['symbol'].iloc[0] if 'symbol' in df.columns else 'Unknown'} 死叉检测: {i}根K线前")
                st.write(f"   前一根: MA{ma_fast}={ma_fast_prev:.4f}, MA{ma_slow}={ma_slow_prev:.4f}")
                st.write(f"   当前根: MA{ma_fast}={ma_fast_curr:.4f}, MA{ma_slow}={ma_slow_curr:.4f}")
            break
    
    # 如果没有检测到交叉，返回None
    if not cross_detected:
        return None, candle_count
    
    # 应用过滤条件
    if enable_filter:
        if signal_info["golden_cross"]:
            if golden_filter == f"K线在MA{ma_fast}上方" and not signal_info["price_above_fast"]:
                return None, candle_count
            elif golden_filter == f"K线在MA{ma_slow}上方" and not signal_info["price_above_slow"]:
                return None, candle_count
            elif golden_filter == "K线在双MA上方" and not (signal_info["price_above_fast"] and signal_info["price_above_slow"]):
                return None, candle_count
        
        if signal_info["death_cross"]:
            if death_filter == f"K线在MA{ma_fast}下方" and signal_info["price_above_fast"]:
                return None, candle_count
            elif death_filter == f"K线在MA{ma_slow}下方" and signal_info["price_above_slow"]:
                return None, candle_count
            elif death_filter == "K线在双MA下方" and (signal_info["price_above_fast"] or signal_info["price_above_slow"]):
                return None, candle_count
    
    return signal_info, candle_count

def scan_symbols(base: str, symbols: List[str], tickers: dict, timeframe: str, ma_fast: int, ma_slow: int, 
                signal_types: List[str], cross_confirm_bars: int, enable_filter: bool, 
                golden_filter: str, death_filter: str, min_volume: float, show_debug: bool):
    """扫描所有交易对"""
    results = []
    debug_info = []
    
    # 限制扫描数量用于调试
    if show_debug:
        symbols = symbols[:20]  # 调试时只扫描前20个
    
    # 并行获取K线数据
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_candles, base, symbol, timeframe): symbol for symbol in symbols}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed = 0
        
        for future in as_completed(futures):
            symbol = futures[future]
            df = future.result()
            processed += 1
            progress_bar.progress(processed / len(symbols))
            status_text.text(f"正在处理: {symbol} ({processed}/{len(symbols)})")
            
            if df.empty:
                if show_debug:
                    debug_info.append(f"❌ {symbol}: 无K线数据")
                continue
            
            # 添加symbol信息到df中用于调试
            df['symbol'] = symbol
            
            signal_info, candle_count = calculate_ma_signals(
                df, ma_fast, ma_slow, cross_confirm_bars, enable_filter, golden_filter, death_filter, show_debug
            )
            
            if show_debug:
                if signal_info:
                    debug_info.append(f"✅ {symbol}: 检测到{'金叉' if signal_info['golden_cross'] else '死叉'}信号")
                else:
                    debug_info.append(f"⚪ {symbol}: 无交叉信号 (K线数: {candle_count})")
            
            if not signal_info:
                continue
            
            ticker_data = tickers.get(symbol, {"change24h": 0, "volume": 0, "price": 0})
            
            if ticker_data["volume"] < min_volume:
                continue
            
            # 检查信号类型
            if signal_info["golden_cross"] and "金叉信号" in signal_types:
                results.append({
                    "symbol": symbol,
                    "signal_type": "金叉",
                    "change (%)": round(ticker_data["change24h"], 2),
                    "cross_bars_ago": signal_info["cross_bars_ago"],
                    "price_above_fast": signal_info["price_above_fast"],
                    "price_above_slow": signal_info["price_above_slow"],
                    "ma_distance": round(signal_info["ma_distance"], 2),
                    "ma_fast_period": ma_fast,
                    "ma_slow_period": ma_slow,
                    "current_time": signal_info["current_time"]
                })
            
            if signal_info["death_cross"] and "死叉信号" in signal_types:
                results.append({
                    "symbol": symbol,
                    "signal_type": "死叉",
                    "change (%)": round(ticker_data["change24h"], 2),
                    "cross_bars_ago": signal_info["cross_bars_ago"],
                    "price_above_fast": signal_info["price_above_fast"],
                    "price_above_slow": signal_info["price_above_slow"],
                    "ma_distance": round(signal_info["ma_distance"], 2),
                    "ma_fast_period": ma_fast,
                    "ma_slow_period": ma_slow,
                    "current_time": signal_info["current_time"]
                })
        
        progress_bar.empty()
        status_text.empty()
    
    return results, debug_info

def format_results(results: List[dict]) -> pd.DataFrame:
    """格式化结果"""
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    def add_icon(row):
        change = row["change (%)"]
        signal = row["signal_type"]
        if signal == "金叉":
            icon = "🚀🟢" if change > 5 else "📈🟢" if change > 0 else "📉🟢"
        else:
            icon = "💥🔴" if change < -5 else "📉🔴" if change < 0 else "📈🔴"
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
    df_formatted["检测时间"] = df_formatted["current_time"].apply(lambda x: x.strftime("%m-%d %H:%M"))
    
    return df_formatted[["交易对", "穿越方向", "24h涨跌", "交叉时间", "价格位置", "MA距离", "检测时间"]]

def main():
    # 页面头部
    st.markdown("# 📈 双MA交叉扫描器 (调试版)")
    st.markdown("---")
    
    # 侧边栏
    timeframe, ma_fast, ma_slow, signal_types, cross_confirm_bars, enable_filter, golden_filter, death_filter, min_volume, show_debug = create_sidebar()
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        scan_button = st.button("🚀 开始扫描", use_container_width=True)
        
        with st.expander("📋 当前设置"):
            st.write(f"**时间框架**: {timeframe}")
            st.write(f"**MA设置**: {ma_fast} × {ma_slow}")
            st.write(f"**信号类型**: {', '.join(signal_types)}")
            st.write(f"**确认周期**: {cross_confirm_bars}")
            if show_debug:
                st.write("**调试模式**: 开启")
    
    with col1:
        if not scan_button:
            st.markdown(f"""
            ### 🎯 使用指南 (调试版)
            
            **当前设置**: MA{ma_fast} × MA{ma_slow}, {timeframe}
            
            #### 🔧 调试功能：
            - 启用"显示调试信息"查看详细检测过程
            - 调试模式下只扫描前20个交易对
            - 显示每个币种的检测状态
            
            #### 🎯 交叉检测逻辑：
            - **金叉**: 快线从下方穿越慢线到上方
            - **死叉**: 快线从上方穿越慢线到下方
            - **确认周期**: 在最近N根K线内发生的交叉
            
            点击"开始扫描"测试检测准确性
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
            start_time = time.time()
            results, debug_info = scan_symbols(base, symbols, tickers, timeframe, ma_fast, ma_slow, 
                                             signal_types, cross_confirm_bars, enable_filter, 
                                             golden_filter, death_filter, min_volume, show_debug)
            scan_time = time.time() - start_time
            
            st.success(f"✅ 扫描完成! 找到 {len(results)} 个信号 (耗时 {scan_time:.1f}秒)")
            
            # 显示调试信息
            if show_debug and debug_info:
                with st.expander("🔍 调试信息"):
                    for info in debug_info[:50]:  # 只显示前50条
                        st.text(info)
            
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
            st.exception(e)

if __name__ == "__main__":
    main()
