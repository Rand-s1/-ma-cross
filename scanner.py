import requests
import ta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置页面配置
st.set_page_config(
    page_title="鹅的MA交叉扫描器 Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主要背景和主题 */
    .main {
        padding-top: 2rem;
    }
    
    /* 标题样式 */
    .big-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4ecdc4;
        margin: 1rem 0;
    }
    
    /* 按钮样式 */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* 数据表格样式 */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 警告和信息框样式 */
    .stAlert {
        border-radius: 10px;
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置常量
class Config:
    ENDPOINTS = ["https://api.bitget.com"]
    PRODUCT_TYPE = "usdt-futures"
    LIMIT = 500  # 增加到500根K线以支持更长周期的MA
    SLEEP_BETWEEN_REQUESTS = 0.5
    MAX_WORKERS = 10
    MIN_CANDLES_RELIABLE = 50
    
    # UI配置
    TIMEFRAMES = {
        "5分钟": "5m",
        "15分钟": "15m",
        "30分钟": "30m",
        "1小时": "1H",
        "4小时": "4H", 
        "1天": "1D"
    }
    
    # MA周期选项
    MA_PERIODS = [10, 20, 55, 70, 150, 200, 350]

def create_header():
    """创建页面头部"""
    st.markdown('<h1 class="big-title">📊 鹅的MA交叉扫描器 Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🚀 Bitget USDT永续合约 - 移动平均线交叉信号扫描</p>', unsafe_allow_html=True)
    
    # 添加分隔线
    st.markdown("---")

def create_sidebar():
    """创建侧边栏"""
    with st.sidebar:
        st.markdown("### ⚙️ MA交叉扫描设置")
        
        # 时间框架选择
        timeframe_display = st.selectbox(
            "📊 时间框架",
            options=list(Config.TIMEFRAMES.keys()),
            index=3,  # 默认1小时
            help="选择K线时间周期"
        )
        timeframe = Config.TIMEFRAMES[timeframe_display]
        
        st.markdown("### 📈 MA线设置")
        
        # MA周期选择
        col1, col2 = st.columns(2)
        with col1:
            ma_fast = st.selectbox(
                "快线周期", 
                options=Config.MA_PERIODS,
                index=1,  # 默认20
                help="选择快速移动平均线周期"
            )
        with col2:
            ma_slow = st.selectbox(
                "慢线周期", 
                options=Config.MA_PERIODS,
                index=5,  # 默认200
                help="选择慢速移动平均线周期"
            )
        
        # 验证MA设置
        if ma_fast >= ma_slow:
            st.error("⚠️ 快线周期必须小于慢线周期！")
            return None, None, None, None, None, None
        
        st.markdown("### 🎯 交叉信号设置")
        
        # 交叉类型选择
        cross_type = st.selectbox(
            "交叉类型",
            options=["所有交叉", "金叉(向上)", "死叉(向下)"],
            index=0,
            help="选择要扫描的交叉类型"
        )
        
        # 高级设置
        with st.expander("🔧 高级设置"):
            show_charts = st.checkbox("显示图表分析", value=True)
            min_volume = st.number_input("最小成交量过滤", value=0.0, help="过滤低成交量币种")
            cross_within_bars = st.number_input("交叉发生在最近N根K线内", min_value=1, max_value=10, value=3, help="只显示最近N根K线内发生的交叉")
            
        return timeframe, ma_fast, ma_slow, cross_type, show_charts, min_volume, cross_within_bars

def ping_endpoint(endpoint: str) -> bool:
    """测试端点是否可用"""
    url = f"{endpoint}/api/v2/mix/market/candles"
    params = {
        "symbol": "BTCUSDT",
        "granularity": "1H",
        "limit": 1,
        "productType": Config.PRODUCT_TYPE,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        return r.status_code == 200 and r.json().get("code") == "00000"
    except:
        return False

def get_working_endpoint() -> str:
    """获取可用端点"""
    for ep in Config.ENDPOINTS:
        for _ in range(3):
            if ping_endpoint(ep):
                return ep
            time.sleep(1)
    raise RuntimeError("无可用端点，请检查网络连接")

def get_usdt_symbols(base: str) -> List[str]:
    """获取USDT永续合约交易对"""
    url = f"{base}/api/v2/mix/market/contracts"
    params = {"productType": Config.PRODUCT_TYPE}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        if j.get("code") != "00000":
            raise RuntimeError(f"获取交易对失败: {j}")
        symbols = [c["symbol"] for c in j["data"]]
        logger.info(f"找到 {len(symbols)} 个USDT永续合约")
        return symbols
    except Exception as e:
        logger.error(f"获取交易对错误: {e}")
        raise

def fetch_candles(base: str, symbol: str, granularity: str) -> pd.DataFrame:
    """获取K线数据"""
    url = f"{base}/api/v2/mix/market/candles"
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": Config.LIMIT,
        "productType": Config.PRODUCT_TYPE,
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        j = r.json()
        if j.get("code") != "00000":
            return pd.DataFrame()
            
        cols = ["ts", "open", "high", "low", "close", "volume_base", "volume_quote"]
        df = pd.DataFrame(j["data"], columns=cols)
        df[["open", "high", "low", "close", "volume_base", "volume_quote"]] = df[
            ["open", "high", "low", "close", "volume_base", "volume_quote"]
        ].astype(float)
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        logger.error(f"{symbol} K线获取失败: {e}")
        return pd.DataFrame()

def fetch_all_tickers(base: str) -> Dict[str, dict]:
    """批量获取ticker数据"""
    url = f"{base}/api/v2/mix/market/tickers"
    params = {"productType": Config.PRODUCT_TYPE}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        
        if j.get("code") != "00000":
            logger.error(f"API返回错误: {j}")
            return {}
            
        if not isinstance(j.get("data"), list):
            logger.error(f"API数据格式错误: {type(j.get('data'))}")
            return {}
        
        tickers = {}
        for item in j["data"]:
            try:
                symbol = item.get("symbol", "")
                if not symbol:
                    continue
                
                # 兼容不同的字段名
                change24h = 0.0
                if "change24h" in item:
                    change24h = float(item["change24h"]) * 100
                elif "chgUtc" in item:
                    change24h = float(item["chgUtc"]) * 100
                elif "changeUtc24h" in item:
                    change24h = float(item["changeUtc24h"]) * 100
                
                # 成交量字段
                volume = 0.0
                if "baseVolume" in item:
                    volume = float(item["baseVolume"])
                elif "baseVol" in item:
                    volume = float(item["baseVol"])
                elif "vol24h" in item:
                    volume = float(item["vol24h"])
                
                # 价格字段
                price = 0.0
                if "close" in item:
                    price = float(item["close"])
                elif "last" in item:
                    price = float(item["last"])
                elif "lastPr" in item:
                    price = float(item["lastPr"])
                
                tickers[symbol] = {
                    "change24h": change24h,
                    "volume": volume,
                    "price": price
                }
                
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"处理ticker数据失败 {item.get('symbol', 'unknown')}: {e}")
                continue
        
        logger.info(f"成功获取 {len(tickers)} 个ticker数据")
        return tickers
        
    except Exception as e:
        logger.error(f"获取ticker数据失败: {e}")
        return {}

def detect_ma_crossover(df: pd.DataFrame, ma_fast: int, ma_slow: int, cross_within_bars: int = 3) -> Tuple[Optional[str], Optional[int], dict]:
    """检测MA交叉信号"""
    try:
        if len(df) < max(ma_fast, ma_slow) + 10:
            return None, None, {}
        
        close_series = df["close"].astype(float)
        
        # 计算MA线
        ma_fast_series = ta.trend.sma_indicator(close_series, window=ma_fast)
        ma_slow_series = ta.trend.sma_indicator(close_series, window=ma_slow)
        
        # 检测交叉
        crossover_up = (ma_fast_series > ma_slow_series) & (ma_fast_series.shift(1) <= ma_slow_series.shift(1))
        crossover_down = (ma_fast_series < ma_slow_series) & (ma_fast_series.shift(1) >= ma_slow_series.shift(1))
        
        # 查找最近的交叉点
        recent_cross_up = crossover_up.tail(cross_within_bars).any()
        recent_cross_down = crossover_down.tail(cross_within_bars).any()
        
        cross_type = None
        bars_since_cross = None
        
        if recent_cross_up:
            cross_idx = crossover_up.tail(cross_within_bars).idxmax()
            if crossover_up.iloc[cross_idx]:
                cross_type = "金叉"
                bars_since_cross = len(df) - 1 - cross_idx
        
        if recent_cross_down:
            cross_idx = crossover_down.tail(cross_within_bars).idxmax()
            if crossover_down.iloc[cross_idx]:
                if cross_type is None or (len(df) - 1 - cross_idx) < bars_since_cross:
                    cross_type = "死叉"
                    bars_since_cross = len(df) - 1 - cross_idx
        
        # 计算额外指标
        metrics = {
            "ma_fast_current": ma_fast_series.iloc[-1],
            "ma_slow_current": ma_slow_series.iloc[-1],
            "ma_distance": ((ma_fast_series.iloc[-1] - ma_slow_series.iloc[-1]) / ma_slow_series.iloc[-1]) * 100,
            "price_vs_ma_fast": ((close_series.iloc[-1] - ma_fast_series.iloc[-1]) / ma_fast_series.iloc[-1]) * 100,
            "price_vs_ma_slow": ((close_series.iloc[-1] - ma_slow_series.iloc[-1]) / ma_slow_series.iloc[-1]) * 100,
            "current_price": close_series.iloc[-1]
        }
        
        return cross_type, bars_since_cross, metrics
        
    except Exception as e:
        logger.error(f"MA交叉检测错误: {e}")
        return None, None, {}

def fetch_candles_wrapper(args) -> tuple:
    """并行获取K线数据的包装函数"""
    base, symbol, granularity = args
    df = fetch_candles(base, symbol, granularity)
    if not df.empty:
        df["symbol"] = symbol
    return symbol, df

def create_statistics_cards(results: List[dict], total_symbols: int, ma_fast: int, ma_slow: int):
    """创建统计信息卡片"""
    golden_cross = len([r for r in results if r["cross_type"] == "金叉"])
    death_cross = len([r for r in results if r["cross_type"] == "死叉"])
    gainers = len([r for r in results if r["change (%)"] > 0])
    
    # 使用metrics显示，一行4个指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 总扫描数",
            value=f"{total_symbols}",
            help="扫描的交易对总数"
        )
        
    with col2:
        st.metric(
            label=f"🟢 金叉信号",
            value=f"{golden_cross}",
            help=f"MA{ma_fast}向上穿越MA{ma_slow}的币种数量"
        )
        
    with col3:
        st.metric(
            label=f"🔴 死叉信号", 
            value=f"{death_cross}",
            help=f"MA{ma_fast}向下穿越MA{ma_slow}的币种数量"
        )
        
    with col4:
        st.metric(
            label="📈 上涨币种",
            value=f"{gainers}",
            help="24h涨幅 > 0的币种数量"
        )

def create_ma_distance_chart(results: List[dict], ma_fast: int, ma_slow: int):
    """创建MA距离分布图表"""
    if not results:
        return None
        
    df = pd.DataFrame(results)
    
    # MA距离分布直方图
    fig = px.histogram(
        df, 
        x="ma_distance (%)", 
        nbins=30,
        title=f"MA{ma_fast} 与 MA{ma_slow} 距离分布",
        labels={"ma_distance (%)": f"MA{ma_fast} 相对 MA{ma_slow} 的距离 (%)", "count": "币种数量"},
        color_discrete_sequence=["#4ecdc4"]
    )
    
    # 添加零线
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="零线")
    
    fig.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def create_cross_scatter_plot(results: List[dict]):
    """创建交叉信号散点图"""
    if not results:
        return None
        
    df = pd.DataFrame(results)
    
    fig = px.scatter(
        df,
        x="ma_distance (%)",
        y="change (%)",
        color="cross_type",
        title="MA交叉信号 vs 24小时涨跌幅",
        labels={"ma_distance (%)": "MA距离 (%)", "change (%)": "24h涨跌幅 (%)"},
        hover_data=["symbol", "bars_since_cross"],
        color_discrete_map={
            "金叉": "#51cf66",
            "死叉": "#ff6b6b"
        }
    )
    
    # 添加分割线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="涨跌分界线")
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="MA位置分界线")
    
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    
    return fig

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """格式化数据框显示"""
    if df.empty:
        return df
        
    # 添加信号图标
    def add_signal_icon(row):
        cross_type = row["cross_type"]
        change = row["change (%)"]
        
        if cross_type == "金叉":
            if change > 0:
                icon = "🚀"  # 金叉且上涨
            else:
                icon = "🟢"  # 金叉但下跌
        else:  # 死叉
            if change < 0:
                icon = "💥"  # 死叉且下跌
            else:
                icon = "🔴"  # 死叉但上涨
                
        return f"{icon} {row['symbol']}"
    
    df_formatted = df.copy()
    df_formatted["交易对"] = df.apply(add_signal_icon, axis=1)
    df_formatted["交叉类型"] = df_formatted["cross_type"]
    df_formatted["几根K线前"] = df_formatted["bars_since_cross"].apply(lambda x: f"{x}根前")
    df_formatted["24h涨跌"] = df_formatted["change (%)"].apply(lambda x: f"{x:+.2f}%")
    df_formatted["MA距离"] = df_formatted["ma_distance (%)"].apply(lambda x: f"{x:+.2f}%")
    df_formatted["当前价格"] = df_formatted["current_price"].apply(lambda x: f"{x:.4f}")
    
    return df_formatted[["交易对", "交叉类型", "几根K线前", "24h涨跌", "MA距离", "当前价格"]]

def scan_symbols(base: str, symbols: List[str], granularity: str, ma_fast: int, ma_slow: int, cross_type: str, cross_within_bars: int, min_volume: float = 0) -> Tuple[List[dict], dict]:
    """扫描交易对的MA交叉信号"""
    start_time = time.time()
    results = []
    
    # 获取ticker数据
    with st.spinner("📊 正在获取市场数据..."):
        tickers = fetch_all_tickers(base)
        if not tickers:
            st.warning("⚠️ 无法获取完整的市场数据，将使用默认值")
            tickers = {}
    
    # 进度条容器
    progress_container = st.empty()
    status_container = st.empty()
    
    # 并行获取K线数据
    candle_data = {}
    total_symbols = len(symbols)
    processed = 0
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_candles_wrapper, (base, symbol, granularity)) for symbol in symbols]
        
        for future in as_completed(futures):
            symbol, df = future.result()
            processed += 1
            
            if not df.empty:
                candle_data[symbol] = df
                
            # 更新进度
            progress = processed / total_symbols
            progress_container.progress(progress, text=f"🔄 获取K线数据: {processed}/{total_symbols}")
            status_container.info(f"⏱️ 正在处理: {symbol}")
    
    # 清除进度显示
    progress_container.empty()
    status_container.empty()
    
    # 处理数据
    with st.spinner("🧮 正在检测MA交叉信号..."):
        insufficient_data = []
        
        for symbol in symbols:
            try:
                if symbol not in candle_data:
                    continue
                    
                df = candle_data[symbol]
                detected_cross, bars_since, metrics = detect_ma_crossover(df, ma_fast, ma_slow, cross_within_bars)
                
                if detected_cross is None:
                    insufficient_data.append(symbol)
                    continue
                
                # 使用默认值如果ticker数据不可用
                ticker_data = tickers.get(symbol, {
                    "change24h": 0, 
                    "volume": 0, 
                    "price": 0
                })
                
                # 应用成交量过滤
                if ticker_data["volume"] < min_volume:
                    continue
                
                # 检查交叉类型条件
                if cross_type == "金叉(向上)" and detected_cross != "金叉":
                    continue
                elif cross_type == "死叉(向下)" and detected_cross != "死叉":
                    continue
                
                results.append({
                    "symbol": symbol,
                    "cross_type": detected_cross,
                    "bars_since_cross": bars_since,
                    "change (%)": round(ticker_data["change24h"], 2),
                    "ma_distance (%)": round(metrics.get("ma_distance", 0), 2),
                    "current_price": metrics.get("current_price", ticker_data["price"]),
                    "volume": ticker_data["volume"],
                    "ma_fast_current": metrics.get("ma_fast_current", 0),
                    "ma_slow_current": metrics.get("ma_slow_current", 0)
                })
                    
            except Exception as e:
                logger.warning(f"{symbol} 处理失败: {e}")
                continue
    
    # 扫描统计
    scan_stats = {
        "scan_time": time.time() - start_time,
        "total_symbols": total_symbols,
        "processed_symbols": len(candle_data),
        "insufficient_data": len(insufficient_data),
        "results_count": len(results)
    }
    
    return results, scan_stats

def main():
    # 创建页面头部
    create_header()
    
    # 创建侧边栏并获取参数
    sidebar_result = create_sidebar()
    if sidebar_result[0] is None:  # 参数验证失败
        return
    
    timeframe, ma_fast, ma_slow, cross_type, show_charts, min_volume, cross_within_bars = sidebar_result
    
    # 主要内容区域
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # 扫描按钮
        if st.button("🚀 开始扫描", key="scan_button", help="点击开始扫描MA交叉信号"):
            scan_pressed = True
        else:
            scan_pressed = False
            
        # 显示当前设置
        with st.expander("📋 当前设置", expanded=True):
            st.write(f"⏰ **时间框架**: {timeframe}")
            st.write(f"📈 **快线**: MA{ma_fast}")
            st.write(f"📉 **慢线**: MA{ma_slow}")
            st.write(f"🎯 **交叉类型**: {cross_type}")
            st.write(f"⏱️ **时间窗口**: {cross_within_bars}根K线内")
            if min_volume > 0:
                st.write(f"📊 **最小成交量**: {min_volume:,.0f}")
    
    with col1:
        if not scan_pressed:
            # 显示使用说明
            st.markdown(f"""
            ### 🎯 MA交叉扫描器使用指南
            
            **MA交叉扫描器**是一个专业的移动平均线交叉信号检测工具，帮助您快速找到交易机会：
            
            #### 📊 功能特点：
            - 🔄 **实时扫描**: 并行处理所有USDT永续合约
            - 📈 **多时间框架**: 支持5m、15m、30m、1H、4H、1D级别
            - 🎨 **可视化分析**: 直观的交叉信号图表
            - 📁 **数据导出**: 支持CSV格式下载
            - ⚡ **高性能**: 多线程处理，扫描速度快
            
            #### 🎯 交叉信号说明：
            - 🟢 **金叉信号**: 快线(MA{ma_fast})向上穿越慢线(MA{ma_slow}) - 潜在买入信号
            - 🔴 **死叉信号**: 快线(MA{ma_fast})向下穿越慢线(MA{ma_slow}) - 潜在卖出信号
            
            #### 📋 可选MA周期：
            **{', '.join([f'MA{p}' for p in Config.MA_PERIODS])}**
            
            #### 🚀 开始使用：
            1. 在左侧选择您的MA周期组合
            2. 设置时间框架和交叉类型
            3. 点击"开始扫描"按钮
            4. 等待扫描完成并查看结果
            """)
            return
    
    if scan_pressed:
        try:
            # 获取API端点
            with st.spinner("🔗 连接到Bitget API..."):
                base = get_working_endpoint()
                st.success("✅ API连接成功")
            
            # 获取交易对
            with st.spinner("📋 获取交易对列表..."):
                symbols = get_usdt_symbols(base)
                st.success(f"✅ 找到 {len(symbols)} 个USDT永续合约")
            
            # 执行扫描
            results, scan_stats = scan_symbols(base, symbols, timeframe, ma_fast, ma_slow, cross_type, cross_within_bars, min_volume)
            
            # 显示扫描统计
            st.success(f"✅ 扫描完成! 耗时 {scan_stats['scan_time']:.1f} 秒")
            
            if scan_stats['insufficient_data'] > 0:
                st.info(f"ℹ️ 有 {scan_stats['insufficient_data']} 个币种数据不足，已跳过")
            
            # 分类结果
            golden_crosses = sorted([r for r in results if r["cross_type"] == "金叉"], key=lambda x: x["bars_since_cross"])
            death_crosses = sorted([r for r in results if r["cross_type"] == "死叉"], key=lambda x: x["bars_since_cross"])
            
            # 显示统计卡片
            create_statistics_cards(results, scan_stats['total_symbols'], ma_fast, ma_slow)
            
            # 显示结果表格
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 金叉信号
            if cross_type in ["所有交叉", "金叉(向上)"] and golden_crosses:
                st.markdown(f"### 🟢 金叉信号 (MA{ma_fast} ↗️ MA{ma_slow}) - {timeframe}")
                golden_df = pd.DataFrame(golden_crosses)
                formatted_golden = format_dataframe(golden_df)
                st.dataframe(formatted_golden, use_container_width=True, hide_index=True)
                
                # 下载按钮
                csv_data = golden_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载金叉信号 CSV",
                    data=csv_data,
                    file_name=f"golden_cross_MA{ma_fast}_{ma_slow}_{timeframe}_{current_time.replace(' ', '_').replace(':', '-')}.csv",
                    mime="text/csv",
                    key="download_golden"
                )
            
            # 死叉信号
            if cross_type in ["所有交叉", "死叉(向下)"] and death_crosses:
                st.markdown(f"### 🔴 死叉信号 (MA{ma_fast} ↘️ MA{ma_slow}) - {timeframe}")
                death_df = pd.DataFrame(death_crosses)
                formatted_death = format_dataframe(death_df)
                st.dataframe(formatted_death, use_container_width=True, hide_index=True)
                
                # 下载按钮
                csv_data = death_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载死叉信号 CSV", 
                    data=csv_data,
                    file_name=f"death_cross_MA{ma_fast}_{ma_slow}_{timeframe}_{current_time.replace(' ', '_').replace(':', '-')}.csv",
                    mime="text/csv",
                    key="download_death"
                )
            
            # 如果没有找到信号
            if not results:
                st.info(f"🤔 在最近{cross_within_bars}根K线内未找到符合条件的MA交叉信号")
            
            # 📊 图表分析
            if show_charts and results:
                st.markdown("---")
                st.markdown("### 📊 数据分析")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    distance_chart = create_ma_distance_chart(results, ma_fast, ma_slow)
                    if distance_chart:
                        st.plotly_chart(distance_chart, use_container_width=True)
                
                with chart_col2:
                    scatter_chart = create_cross_scatter_plot(results)
                    if scatter_chart:
                        st.plotly_chart(scatter_chart, use_container_width=True)
                
            # 扫描信息
            with st.expander("ℹ️ 扫描详情"):
                st.write(f"**扫描时间**: {current_time}")
                st.write(f"**处理时间**: {scan_stats['scan_time']:.2f} 秒")
                st.write(f"**MA设置**: MA{ma_fast} × MA{ma_slow}")
                st.write(f"**时间框架**: {timeframe}")
                st.write(f"**交叉类型**: {cross_type}")
                st.write(f"**时间窗口**: {cross_within_bars}根K线内")
                st.write(f"**总交易对数**: {scan_stats['total_symbols']}")
                st.write(f"**成功处理**: {scan_stats['processed_symbols']}")
                st.write(f"**找到信号**: {scan_stats['results_count']}")
                st.write(f"**数据不足**: {scan_stats['insufficient_data']}")
                
        except Exception as e:
            st.error(f"❌ 扫描过程中发生错误: {str(e)}")
            logger.error(f"扫描错误: {e}")

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📊 MA交叉扫描器 Pro - 移动平均线交叉信号检测</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
