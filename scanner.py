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
    page_title="鹅的双MA交叉扫描器",
    page_icon="📈",
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
    LIMIT = 400  # 足够计算MA350
    SLEEP_BETWEEN_REQUESTS = 0.5
    MAX_WORKERS = 10
    
    # UI配置
    TIMEFRAMES = {
        "15分钟": "15m",
        "30分钟": "30m", 
        "1小时": "1H",
        "4小时": "4H", 
        "1天": "1D"
    }
    
    # MA周期选项
    MA_OPTIONS = [20, 70, 150, 200, 350]

def create_header():
    """创建页面头部"""
    st.markdown('<h1 class="big-title">📈 鹅的双MA交叉扫描器</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🚀 自定义双移动平均线交叉信号扫描</p>', unsafe_allow_html=True)
    
    # 添加分隔线
    st.markdown("---")

def create_sidebar():
    """创建侧边栏 - 增强版本"""
    with st.sidebar:
        st.markdown("### ⚙️ 扫描设置")
        
        # 时间框架选择
        timeframe_display = st.selectbox(
            "📊 时间框架",
            options=list(Config.TIMEFRAMES.keys()),
            index=2,  # 默认1小时
            help="选择K线时间周期"
        )
        timeframe = Config.TIMEFRAMES[timeframe_display]
        
        st.markdown("### 📈 MA线设置")
        
        # MA线选择
        col1, col2 = st.columns(2)
        with col1:
            ma_fast = st.selectbox(
                "快线周期",
                options=Config.MA_OPTIONS,
                index=0,  # 默认20
                help="选择快速移动平均线周期（通常较小）"
            )
        with col2:
            ma_slow = st.selectbox(
                "慢线周期",
                options=Config.MA_OPTIONS,
                index=1,  # 默认70
                help="选择慢速移动平均线周期（通常较大）"
            )
        
        # 检查MA设置是否合理
        if ma_fast >= ma_slow:
            st.warning("⚠️ 建议快线周期小于慢线周期")
        
        st.markdown("### 🎯 交叉信号设置")
        
        # 信号类型选择
        signal_types = st.multiselect(
            "选择扫描信号",
            options=["金叉信号", "死叉信号"],
            default=["金叉信号", "死叉信号"],
            help=f"金叉：MA{ma_fast}上穿MA{ma_slow}做多；死叉：MA{ma_fast}下穿MA{ma_slow}做空"
        )
        
        # 交叉确认周期
        cross_confirm_bars = st.slider(
            "交叉确认周期",
            min_value=1,
            max_value=5,
            value=2,
            help="最近N根K线内发生的交叉才显示"
        )
        
        # 新增：K线位置过滤
        st.markdown("### 🔍 K线位置过滤")
        
        enable_price_filter = st.checkbox(
            "启用K线位置过滤", 
            value=True, 
            help="根据K线相对MA的位置过滤信号，提高信号质量"
        )
        
        if enable_price_filter:
            # 金叉K线位置要求
            if "金叉信号" in signal_types:
                golden_price_filter = st.selectbox(
                    "🟢 金叉K线位置要求",
                    options=[
                        "无要求",
                        f"K线在MA{ma_fast}上方",
                        f"K线在MA{ma_slow}上方", 
                        "K线在双MA上方",
                        f"K线在MA{ma_fast}上方且距离<3%",
                        f"K线在MA{ma_slow}上方且距离<3%"
                    ],
                    index=3,  # 默认双MA上方
                    help="金叉信号的K线位置要求"
                )
            else:
                golden_price_filter = "无要求"
            
            # 死叉K线位置要求  
            if "死叉信号" in signal_types:
                death_price_filter = st.selectbox(
                    "🔴 死叉K线位置要求",
                    options=[
                        "无要求",
                        f"K线在MA{ma_fast}下方",
                        f"K线在MA{ma_slow}下方",
                        "K线在双MA下方",
                        f"K线在MA{ma_fast}下方且距离<3%",
                        f"K线在MA{ma_slow}下方且距离<3%"
                    ],
                    index=3,  # 默认双MA下方
                    help="死叉信号的K线位置要求"
                )
            else:
                death_price_filter = "无要求"
        else:
            golden_price_filter = "无要求"
            death_price_filter = "无要求"
        
        # 高级设置
        with st.expander("🔧 高级设置"):
            show_charts = st.checkbox("显示图表分析", value=True)
            min_volume = st.number_input("最小成交量过滤", value=0.0, help="过滤低成交量币种")
            
        return timeframe, ma_fast, ma_slow, signal_types, cross_confirm_bars, show_charts, min_volume, enable_price_filter, golden_price_filter, death_price_filter

def ping_endpoint(endpoint: str) -> bool:
    """测试端点是否可用"""
    url = f"{endpoint}/api/v2/mix/market/candles"
    params = {
        "symbol": "BTCUSDT",
        "granularity": "4H",
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
        
        logger.info(f"Ticker API响应: code={j.get('code')}, msg={j.get('msg')}")
        
        if j.get("code") != "00000":
            logger.error(f"API返回错误: {j}")
            return {}
            
        if not isinstance(j.get("data"), list):
            logger.error(f"API数据格式错误: {type(j.get('data'))}")
            return {}
        
        tickers = {}
        for item in j["data"]:
            try:
                if len(tickers) == 0:
                    logger.info(f"Ticker数据结构示例: {list(item.keys())}")
                
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
        
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求失败: {e}")
        return {}
    except Exception as e:
        logger.error(f"获取ticker数据失败: {e}")
        return {}

def calculate_dual_ma_signals(df: pd.DataFrame, ma_fast: int, ma_slow: int, cross_confirm_bars: int = 2, enable_price_filter: bool = False, golden_price_filter: str = "无要求", death_price_filter: str = "无要求") -> Tuple[Optional[dict], int]:
    """计算双MA交叉信号 - 增强价格过滤版本"""
    try:
        candle_count = len(df)
        
        # 动态计算最小K线数量要求
        min_candles_needed = max(ma_fast, ma_slow) + 10
        
        if candle_count < min_candles_needed:
            return None, candle_count
            
        # 计算技术指标
        close_series = pd.Series(df["close"].astype(float)).reset_index(drop=True)
        
        # 计算双MA线
        ma_fast_line = close_series.rolling(window=ma_fast).mean()
        ma_slow_line = close_series.rolling(window=ma_slow).mean()
        
        # 检查数据有效性
        if ma_fast_line.isna().all() or ma_slow_line.isna().all():
            return None, candle_count
        
        # 检测交叉信号
        signal_info = {
            "ma_fast_current": ma_fast_line.iloc[-1],
            "ma_slow_current": ma_slow_line.iloc[-1],
            "ma_fast_period": ma_fast,
            "ma_slow_period": ma_slow,
            "price_current": close_series.iloc[-1],
            "golden_cross": False,
            "death_cross": False,
            "cross_bars_ago": None,
            "ma_distance": 0,
            "price_above_fast": False,
            "price_above_slow": False,
            "price_deviation_fast": 0,
            "price_deviation_slow": 0
        }
        
        # 计算MA线距离（百分比）
        if signal_info["ma_slow_current"] > 0:
            signal_info["ma_distance"] = abs(signal_info["ma_fast_current"] - signal_info["ma_slow_current"]) / signal_info["ma_slow_current"] * 100
        
        # 计算价格相对位置和偏离度
        signal_info["price_above_fast"] = signal_info["price_current"] > signal_info["ma_fast_current"]
        signal_info["price_above_slow"] = signal_info["price_current"] > signal_info["ma_slow_current"]
        
        # 计算价格偏离MA的百分比
        if signal_info["ma_fast_current"] > 0:
            signal_info["price_deviation_fast"] = abs(signal_info["price_current"] - signal_info["ma_fast_current"]) / signal_info["ma_fast_current"] * 100
        if signal_info["ma_slow_current"] > 0:
            signal_info["price_deviation_slow"] = abs(signal_info["price_current"] - signal_info["ma_slow_current"]) / signal_info["ma_slow_current"] * 100
        
        # 在最近的几根K线内检测交叉
        for i in range(1, min(cross_confirm_bars + 1, len(ma_fast_line))):
            if pd.isna(ma_fast_line.iloc[-(i+1)]) or pd.isna(ma_slow_line.iloc[-(i+1)]) or pd.isna(ma_fast_line.iloc[-i]) or pd.isna(ma_slow_line.iloc[-i]):
                continue
                
            # 金叉检测：快线从下方穿过慢线向上
            if (ma_fast_line.iloc[-(i+1)] <= ma_slow_line.iloc[-(i+1)] and ma_fast_line.iloc[-i] > ma_slow_line.iloc[-i]):
                signal_info["golden_cross"] = True
                signal_info["cross_bars_ago"] = i
                break
                
            # 死叉检测：快线从上方穿过慢线向下  
            elif (ma_fast_line.iloc[-(i+1)] >= ma_slow_line.iloc[-(i+1)] and ma_fast_line.iloc[-i] < ma_slow_line.iloc[-i]):
                signal_info["death_cross"] = True
                signal_info["cross_bars_ago"] = i
                break
        
        # 新增：K线位置过滤
        if enable_price_filter:
            # 金叉K线位置过滤
            if signal_info["golden_cross"]:
                if golden_price_filter == f"K线在MA{ma_fast}上方":
                    if not signal_info["price_above_fast"]:
                        signal_info["golden_cross"] = False
                elif golden_price_filter == f"K线在MA{ma_slow}上方":
                    if not signal_info["price_above_slow"]:
                        signal_info["golden_cross"] = False
                elif golden_price_filter == "K线在双MA上方":
                    if not (signal_info["price_above_fast"] and signal_info["price_above_slow"]):
                        signal_info["golden_cross"] = False
                elif golden_price_filter == f"K线在MA{ma_fast}上方且距离<3%":
                    if not signal_info["price_above_fast"] or signal_info["price_deviation_fast"] > 3:
                        signal_info["golden_cross"] = False
                elif golden_price_filter == f"K线在MA{ma_slow}上方且距离<3%":
                    if not signal_info["price_above_slow"] or signal_info["price_deviation_slow"] > 3:
                        signal_info["golden_cross"] = False
            
            # 死叉K线位置过滤
            if signal_info["death_cross"]:
                if death_price_filter == f"K线在MA{ma_fast}下方":
                    if signal_info["price_above_fast"]:
                        signal_info["death_cross"] = False
                elif death_price_filter == f"K线在MA{ma_slow}下方":
                    if signal_info["price_above_slow"]:
                        signal_info["death_cross"] = False
                elif death_price_filter == "K线在双MA下方":
                    if signal_info["price_above_fast"] or signal_info["price_above_slow"]:
                        signal_info["death_cross"] = False
                elif death_price_filter == f"K线在MA{ma_fast}下方且距离<3%":
                    if signal_info["price_above_fast"] or signal_info["price_deviation_fast"] > 3:
                        signal_info["death_cross"] = False
                elif death_price_filter == f"K线在MA{ma_slow}下方且距离<3%":
                    if signal_info["price_above_slow"] or signal_info["price_deviation_slow"] > 3:
                        signal_info["death_cross"] = False
        
        return signal_info, candle_count
        
    except Exception as e:
        logger.error(f"双MA交叉计算错误: {e}")
        return None, 0

def fetch_candles_wrapper(args) -> tuple:
    """并行获取K线数据的包装函数"""
    base, symbol, granularity = args
    df = fetch_candles(base, symbol, granularity)
    if not df.empty:
        df["symbol"] = symbol
    return symbol, df

def create_statistics_cards(results: List[dict], total_symbols: int):
    """创建统计信息"""
    golden_crosses = len([r for r in results if r["signal_type"] == "金叉"])
    death_crosses = len([r for r in results if r["signal_type"] == "死叉"])
    gainers = len([r for r in results if r["change (%)"] > 0])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 总扫描数",
            value=f"{total_symbols}",
            help="扫描的交易对总数"
        )
        
    with col2:
        st.metric(
            label="🟢 金叉信号",
            value=f"{golden_crosses}",
            help="快线上穿慢线的做多信号"
        )
        
    with col3:
        st.metric(
            label="🔴 死叉信号", 
            value=f"{death_crosses}",
            help="快线下穿慢线的做空信号"
        )
        
    with col4:
        st.metric(
            label="📈 上涨币种",
            value=f"{gainers}",
            help="24h涨幅 > 0的币种数量"
        )

def create_signal_distribution_chart(results: List[dict]):
    """创建信号分布图表"""
    if not results:
        return None
        
    df = pd.DataFrame(results)
    
    # 信号类型分布饼图
    signal_counts = df['signal_type'].value_counts()
    
    fig = px.pie(
        values=signal_counts.values,
        names=signal_counts.index,
        title="交叉信号分布",
        color_discrete_map={
            "金叉": "#51cf66",
            "死叉": "#ff6b6b"
        }
    )
    
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    
    return fig

def create_ma_distance_chart(results: List[dict]):
    """创建MA距离分布图"""
    if not results:
        return None
        
    df = pd.DataFrame(results)
    
    fig = px.histogram(
        df,
        x="ma_distance",
        color="signal_type",
        title="MA线距离分布 (%)",
        labels={"ma_distance": "MA线距离 (%)", "count": "信号数量"},
        color_discrete_map={
            "金叉": "#51cf66",
            "死叉": "#ff6b6b"
        },
        nbins=20
    )
    
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    
    return fig

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """格式化数据框显示 - 增强版本"""
    if df.empty:
        return df
        
    # 添加信号图标
    def add_signal_icon(row):
        change = row["change (%)"]
        signal = row["signal_type"]
        
        if signal == "金叉":
            if change > 5:
                icon = "🚀🟢"
            elif change > 0:
                icon = "📈🟢"
            else:
                icon = "📉🟢"
        else:  # 死叉
            if change < -5:
                icon = "💥🔴"
            elif change < 0:
                icon = "📉🔴"
            else:
                icon = "📈🔴"
                
        return f"{icon} {row['symbol']}"
    
    def get_cross_direction(row):
        """获取具体的穿越方向"""
        fast_period = row["ma_fast_period"]
        slow_period = row["ma_slow_period"]
        signal = row["signal_type"]
        
        if signal == "金叉":
            return f"MA{fast_period} ↗️ MA{slow_period}"
        else:
            return f"MA{fast_period} ↘️ MA{slow_period}"
    
    def format_price_position(row):
        if row["price_above_fast"] and row["price_above_slow"]:
            return "双线上方"
        elif not row["price_above_fast"] and not row["price_above_slow"]:
            return "双线下方"
        elif row["price_above_fast"]:
            return f"MA{row['ma_fast_period']}上方"
        else:
            return f"MA{row['ma_slow_period']}上方"
    
    df_formatted = df.copy()
    df_formatted["交易对"] = df.apply(add_signal_icon, axis=1)
    df_formatted["穿越方向"] = df.apply(get_cross_direction, axis=1)
    df_formatted["24h涨跌"] = df_formatted["change (%)"].apply(lambda x: f"{x:+.2f}%")
    df_formatted["交叉时间"] = df_formatted["cross_bars_ago"].apply(lambda x: f"{x}根K线前")
    df_formatted["价格位置"] = df.apply(format_price_position, axis=1)
    df_formatted["MA距离"] = df_formatted["ma_distance"].apply(lambda x: f"{x:.2f}%")
    df_formatted["K线数"] = df_formatted["k_lines"]
    df_formatted["备注"] = df_formatted["note"]
    
    return df_formatted[["交易对", "穿越方向", "24h涨跌", "交叉时间", "价格位置", "MA距离", "K线数", "备注"]]

def scan_symbols(base: str, symbols: List[str], granularity: str, ma_fast: int, ma_slow: int, signal_types: List[str], cross_confirm_bars: int, min_volume: float = 0, enable_price_filter: bool = False, golden_price_filter: str = "无要求", death_price_filter: str = "无要求") -> Tuple[List[dict], dict]:
    """扫描交易对的双MA交叉信号 - 增强价格过滤版本"""
    start_time = time.time()
    results = []
    
    # 动态计算最小K线数量要求
    min_candles_needed = max(ma_fast, ma_slow) + 10
    
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
    with st.spinner("🧮 正在计算交叉信号..."):
        insufficient_data = []
        
        for symbol in symbols:
            try:
                if symbol not in candle_data:
                    continue
                    
                df = candle_data[symbol]
                signal_info, candle_count = calculate_dual_ma_signals(df, ma_fast, ma_slow, cross_confirm_bars, enable_price_filter, golden_price_filter, death_price_filter)
                
                if signal_info is None:
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
                
                # 检查是否有符合条件的交叉信号
                has_golden_cross = signal_info["golden_cross"]
                has_death_cross = signal_info["death_cross"]
                
                if has_golden_cross and "金叉信号" in signal_types:
                    note = ""
                    if candle_count < min_candles_needed + 20:
                        note = f"数据较少({candle_count}根，建议≥{min_candles_needed + 20}根)"
                    
                    # 添加过滤条件说明
                    if enable_price_filter and golden_price_filter != "无要求":
                        if note:
                            note += f" | {golden_price_filter}"
                        else:
                            note = f"过滤:{golden_price_filter}"
                    
                    results.append({
                        "symbol": symbol,
                        "signal_type": "金叉",
                        "change (%)": round(ticker_data["change24h"], 2),
                        "cross_bars_ago": signal_info["cross_bars_ago"],
                        "price_above_fast": signal_info["price_above_fast"],
                        "price_above_slow": signal_info["price_above_slow"],
                        "ma_distance": round(signal_info["ma_distance"], 2),
                        "k_lines": candle_count,
                        "note": note,
                        "volume": ticker_data["volume"],
                        "price": ticker_data["price"],
                        "ma_fast_value": signal_info["ma_fast_current"],
                        "ma_slow_value": signal_info["ma_slow_current"],
                        "ma_fast_period": ma_fast,
                        "ma_slow_period": ma_slow,
                        "price_deviation_fast": round(signal_info["price_deviation_fast"], 2),
                        "price_deviation_slow": round(signal_info["price_deviation_slow"], 2)
                    })
                    
                elif has_death_cross and "死叉信号" in signal_types:
                    note = ""
                    if candle_count < min_candles_needed + 20:
                        note = f"数据较少({candle_count}根，建议≥{min_candles_needed + 20}根)"
                    
                    # 添加过滤条件说明
                    if enable_price_filter and death_price_filter != "无要求":
                        if note:
                            note += f" | {death_price_filter}"
                        else:
                            note = f"过滤:{death_price_filter}"
                    
                    results.append({
                        "symbol": symbol,
                        "signal_type": "死叉",
                        "change (%)": round(ticker_data["change24h"], 2),
                        "cross_bars_ago": signal_info["cross_bars_ago"],
                        "price_above_fast": signal_info["price_above_fast"],
                        "price_above_slow": signal_info["price_above_slow"],
                        "ma_distance": round(signal_info["ma_distance"], 2),
                        "k_lines": candle_count,
                        "note": note,
                        "volume": ticker_data["volume"],
                        "price": ticker_data["price"],
                        "ma_fast_value": signal_info["ma_fast_current"],
                        "ma_slow_value": signal_info["ma_slow_current"],
                        "ma_fast_period": ma_fast,
                        "ma_slow_period": ma_slow,
                        "price_deviation_fast": round(signal_info["price_deviation_fast"], 2),
                        "price_deviation_slow": round(signal_info["price_deviation_slow"], 2)
                    })
                    
            except Exception as e:
                logger.warning(f"{symbol} 处理失败: {e}")
                continue
    
    scan_stats = {
        "scan_time": time.time() - start_time,
        "total_symbols": total_symbols,
        "processed_symbols": len(candle_data),
        "insufficient_data": len(insufficient_data),
        "results_count": len(results),
        "min_candles_needed": min_candles_needed
    }
    
    return results, scan_stats

def main():
    # 创建页面头部
    create_header()
    
    # 创建侧边栏并获取参数 - 新增参数
    timeframe, ma_fast, ma_slow, signal_types, cross_confirm_bars, show_charts, min_volume, enable_price_filter, golden_price_filter, death_price_filter = create_sidebar()
    
    # 主要内容区域
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # 扫描按钮
        if st.button("🚀 开始扫描", key="scan_button", help=f"点击开始扫描MA{ma_fast}×MA{ma_slow}交叉信号"):
            scan_pressed = True
        else:
            scan_pressed = False
            
        # 显示当前设置
        with st.expander("📋 当前设置", expanded=True):
            st.write(f"⏰ **时间框架**: {timeframe}")
            st.write(f"📈 **快线**: MA{ma_fast}")
            st.write(f"📈 **慢线**: MA{ma_slow}")
            st.write(f"🎯 **扫描信号**: {', '.join(signal_types)}")
            st.write(f"⏱️ **确认周期**: {cross_confirm_bars}根K线")
            if min_volume > 0:
                st.write(f"📊 **最小成交量**: {min_volume:,.0f}")
            if enable_price_filter:
                if golden_price_filter != "无要求":
                    st.write(f"🟢 **金叉过滤**: {golden_price_filter}")
                if death_price_filter != "无要求":
                    st.write(f"🔴 **死叉过滤**: {death_price_filter}")
    
    with col1:
        if not scan_pressed:
            # 显示使用说明
            st.markdown(f"""
            ### 🎯 使用指南
            
            **双MA交叉扫描器**专门用于检测两条移动平均线的交叉信号：
            
            #### 📊 核心指标：
            - 📈 **快线MA{ma_fast}**: {ma_fast}日移动平均线，反映短期趋势
            - 📈 **慢线MA{ma_slow}**: {ma_slow}日移动平均线，反映长期趋势
            - ⚡ **交叉检测**: 实时监控两线交叉状态
            
            #### 🎯 交易信号：
            - 🟢 **金叉信号**: MA{ma_fast} ↗️ MA{ma_slow} = 看涨趋势开始，考虑做多
            - 🔴 **死叉信号**: MA{ma_fast} ↘️ MA{ma_slow} = 看跌趋势开始，考虑做空
            
            #### 🔍 K线位置过滤（重要功能）：
            - **金叉过滤**: 可要求K线在MA{ma_fast}或MA{ma_slow}上方，提高做多信号质量
            - **死叉过滤**: 可要求K线在MA{ma_fast}或MA{ma_slow}下方，提高做空信号质量
            - **距离限制**: 可限制K线与MA的距离，避免过度偏离的信号
            - **推荐设置**: 金叉要求"K线在双MA上方"，死叉要求"K线在双MA下方"
            
            #### 💡 针对你的策略优化：
            - **做空策略**: 选择"MA20下穿MA70"，设置"K线在MA20上方"过滤
            - **做多策略**: 选择"MA20上穿MA70"，设置"K线在MA70上方"过滤
            - **高质量信号**: 启用位置过滤后，信号数量会减少，但质量大幅提升
            
            #### 🚀 开始使用：
            1. 选择时间框架和双MA周期（推荐MA20×MA70）
            2. 启用K线位置过滤，设置合适的过滤条件
            3. 点击"开始扫描"并等待结果
            4. 查看"穿越方向"列明确了解交叉方向
            
            #### ⚠️ 风险提示：
            - MA交叉信号存在滞后性，适合趋势跟踪
            - 震荡行情中可能产生频繁假信号
            - 建议结合其他指标综合分析
            - 请设置合理的止损和资金管理策略
            """)
            return
    
    if scan_pressed:
        if not signal_types:
            st.error("❌ 请至少选择一种信号类型进行扫描")
            return
            
        if ma_fast >= ma_slow:
            st.error("❌ 快线周期应该小于慢线周期")
            return
            
        try:
            # 获取API端点
            with st.spinner("🔗 连接到Bitget API..."):
                base = get_working_endpoint()
                st.success("✅ API连接成功")
            
            # 获取交易对
            with st.spinner("📋 获取交易对列表..."):
                symbols = get_usdt_symbols(base)
                st.success(f"✅ 找到 {len(symbols)} 个USDT永续合约")
            
            # 执行扫描 - 添加新参数
            results, scan_stats = scan_symbols(base, symbols, timeframe, ma_fast, ma_slow, signal_types, cross_confirm_bars, min_volume, enable_price_filter, golden_price_filter, death_price_filter)
            
            # 显示扫描统计
            st.success(f"✅ 扫描完成! 耗时 {scan_stats['scan_time']:.1f} 秒")
            
            if scan_stats['insufficient_data'] > 0:
                st.info(f"ℹ️ 有 {scan_stats['insufficient_data']} 个币种数据不足(需要{scan_stats['min_candles_needed']}+根K线)，已跳过")
            
            # 分类结果
            golden_crosses = sorted([r for r in results if r["signal_type"] == "金叉"], key=lambda x: x["change (%)"], reverse=True)
            death_crosses = sorted([r for r in results if r["signal_type"] == "死叉"], key=lambda x: x["change (%)"])
            
            # 显示统计卡片
            create_statistics_cards(results, scan_stats['total_symbols'])
            
            # 显示结果表格
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 金叉信号
            if "金叉信号" in signal_types:
                st.markdown(f"### 🟢 金叉信号 (MA{ma_fast} 上穿 MA{ma_slow} - {timeframe})")
                if golden_crosses:
                    golden_df = pd.DataFrame(golden_crosses)
                    formatted_golden = format_dataframe(golden_df)
                    
                    # 添加说明
                    filter_text = f" | 过滤条件: {golden_price_filter}" if enable_price_filter and golden_price_filter != "无要求" else ""
                    st.info(f"💡 **交易建议**: MA{ma_fast}上穿MA{ma_slow}，看涨信号，可考虑做多{filter_text}")
                    
                    st.dataframe(formatted_golden, use_container_width=True, hide_index=True)
                    
                    # 下载按钮
                    csv_data = golden_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载金叉信号 CSV",
                        data=csv_data,
                        file_name=f"golden_cross_ma{ma_fast}_ma{ma_slow}_{timeframe}_{current_time.replace(' ', '_').replace(':', '-')}.csv",
                        mime="text/csv",
                        key="download_golden"
                    )
                else:
                    st.info("🤔 当前没有符合条件的金叉信号")
            
            # 死叉信号
            if "死叉信号" in signal_types:
                st.markdown(f"### 🔴 死叉信号 (MA{ma_fast} 下穿 MA{ma_slow} - {timeframe})")
                if death_crosses:
                    death_df = pd.DataFrame(death_crosses)
                    formatted_death = format_dataframe(death_df)
                    
                    # 添加说明
                    filter_text = f" | 过滤条件: {death_price_filter}" if enable_price_filter and death_price_filter != "无要求" else ""
                    st.info(f"💡 **交易建议**: MA{ma_fast}下穿MA{ma_slow}，看跌信号，可考虑做空{filter_text}")
                    
                    st.dataframe(formatted_death, use_container_width=True, hide_index=True)
                    
                    # 下载按钮
                    csv_data = death_df.to_csv(index=False)
                    st.download_button(
                        label="📥 下载死叉信号 CSV", 
                        data=csv_data,
                        file_name=f"death_cross_ma{ma_fast}_ma{ma_slow}_{timeframe}_{current_time.replace(' ', '_').replace(':', '-')}.csv",
                        mime="text/csv",
                        key="download_death"
                    )
                else:
                    st.info("🤔 当前没有符合条件的死叉信号")
            
            # 图表分析
            if show_charts and results:
                st.markdown("---")
                st.markdown("### 📊 信号分析")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    signal_chart = create_signal_distribution_chart(results)
                    if signal_chart:
                        st.plotly_chart(signal_chart, use_container_width=True)
                
                with chart_col2:
                    distance_chart = create_ma_distance_chart(results)
                    if distance_chart:
                        st.plotly_chart(distance_chart, use_container_width=True)
                
            # 扫描信息
            with st.expander("ℹ️ 扫描详情"):
                st.write(f"**扫描时间**: {current_time}")
                st.write(f"**处理时间**: {scan_stats['scan_time']:.2f} 秒")
                st.write(f"**总交易对数**: {scan_stats['total_symbols']}")
                st.write(f"**成功处理**: {scan_stats['processed_symbols']}")
                st.write(f"**符合条件**: {scan_stats['results_count']}")
                st.write(f"**数据不足**: {scan_stats['insufficient_data']}")
                st.write(f"**扫描信号**: {', '.join(signal_types)}")
                st.write(f"**确认周期**: {cross_confirm_bars}根K线")
                st.write(f"**MA设置**: 快线MA{ma_fast} × 慢线MA{ma_slow}")
                if enable_price_filter:
                    if golden_price_filter != "无要求":
                        st.write(f"**金叉过滤**: {golden_price_filter}")
                    if death_price_filter != "无要求":
                        st.write(f"**死叉过滤**: {death_price_filter}")
                
        except Exception as e:
            st.error(f"❌ 扫描过程中发生错误: {str(e)}")
            logger.error(f"扫描错误: {e}")

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📈 双MA交叉信号扫描器 Pro</p>
        <p>🎯 精准捕捉趋势转换机会 | 🔍 智能K线位置过滤</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
