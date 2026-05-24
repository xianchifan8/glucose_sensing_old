"""
数据库数据加载模块 - 从 SQLite .db 文件加载用户的频谱数据和血糖标签

支持新的 DB 格式数据源：
1. 从 spectrum .db 文件读取 spectrum 表获取频谱数据
2. 从 glucose .db 文件读取 blood_sugar 表获取血糖标签（优先）
3. 或从配套的 .json 文件读取血糖数据（备用）
4. 从 spectrum .db 文件读取辅助传感器数据（BME680, PPG, T117, ICM）

时间处理原则 (IMPORTANT):
====================
1. 数据库存储: 所有数据库中的时间戳 (ts/create_time) 均为 UTC Unix timestamp
   - spectrum 表的 ts
   - blood_sugar 表的 create_time
   - 所有传感器表 (cheez_ppg, env_bme, env_temp, imu_icm) 的 ts
   
2. 配置文件: experiments_config.json 中的 start_time/end_time 使用北京时间字符串
   - 格式: "YYYY-MM-DD HH:MM:SS" (北京时间 UTC+8)
   - 加载时会自动转换为 UTC timestamp 用于数据库查询
   
3. 时间转换:
   - beijing_to_utc_timestamp(): 配置的北京时间 -> UTC timestamp (查询数据库)
   - utc_timestamp_to_beijing(): UTC timestamp -> 北京时间 (显示给用户)

Spectrum DB 数据结构:
- spectrum 表: ts (UTC Unix timestamp), data (binary blob, 1001个uint32值)
- cheez_ppg 表: ts (UTC), raw, avg, filter, peak, hr, hrv
- env_bme 表: ts (UTC), t (温度), p (气压), h (湿度)
- env_temp 表: ts (UTC), t (温度)
- imu_icm 表: ts (UTC), ax, ay, az, gx, gy, gz

Glucose DB 数据结构 (blood_sugar 表):
- create_time: Unix timestamp (UTC)
- blood_sugar: 血糖值 (mmol/L)

配套 JSON 文件格式（备用）：
{"times": ["YYYY-MM-DD HH:MM:SS" (北京时间), ...], "username": "...", "values": [...]}
"""

import sqlite3
import json
import struct
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator, Akima1DInterpolator
from pathlib import Path

from .data_loader import (
    AuxSensorData,
    GlucoseDatasetLoader,
    apply_data_fusion,
)


def _db_get_rss_mb() -> float:
    """读取当前进程RSS内存(MB)。"""
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def _db_debug_mem(stage: str):
    rss = _db_get_rss_mb()
    if rss >= 0:
        print(f"  [DEBUG][MEM][DB] {stage} | RSS={rss:.2f} MB")
    else:
        print(f"  [DEBUG][MEM][DB] {stage} | RSS=unknown")


@dataclass
class DBExperimentConfig:
    """数据库实验配置
    
    时间字段说明:
    - start_time/end_time: 北京时间字符串 "YYYY-MM-DD HH:MM:SS" (UTC+8)
    - 加载数据时会自动转换为 UTC timestamp 用于数据库查询
    
    血糖数据来源优先级: glucose_db_path > json_path
    至少需要提供其中一个血糖数据来源
    
    支持单个db文件或多个db文件（实验中断导致的多个文件）：
    - db_path: 可以是单个db文件路径，或包含多个db文件的文件夹路径
    - db_files: 可选，显式指定多个db文件列表
    """
    db_path: str        # spectrum .db 文件路径，或包含多个db文件的文件夹路径
    start_time: str     # 起始时间 (北京时间 UTC+8，格式: "YYYY-MM-DD HH:MM:SS")
    end_time: str       # 结束时间 (北京时间 UTC+8，格式: "YYYY-MM-DD HH:MM:SS")
    experiment_name: str  # 实验名称（用于标识）
    json_path: str = ''  # glucose JSON 文件路径 (可选，glucose_db_path 不存在时使用)
    glucose_db_path: Optional[str] = None  # glucose .db 文件路径 (优先于 json_path)
    db_files: Optional[List[str]] = field(default=None)  # 多个db文件列表（实验中断时使用，可选）
    
    def get_db_paths(self, base_dir: Optional[Path] = None) -> List[str]:
        """获取所有db文件路径列表
        
        Args:
            base_dir: 基础目录，用于解析相对路径
        """
        if self.db_files:
            # 如果指定了db_files列表，直接使用（可能需要解析相对路径）
            if base_dir:
                return [str(base_dir / f) if not Path(f).is_absolute() else f for f in self.db_files]
            return self.db_files
        
        # 解析db_path
        db_path = Path(self.db_path)
        if base_dir and not db_path.is_absolute():
            db_path = base_dir / db_path
            
        if db_path.is_dir():
            # 如果db_path是文件夹，查找其中所有.db文件
            db_files = sorted(db_path.glob("*.db"))
            return [str(f) for f in db_files]
        else:
            # 单个db文件
            return [str(db_path)]


class DBDataLoader:
    """
    数据库格式数据加载器
    
    功能：
    1. 从 .db 文件读取 spectrum 数据
    2. 从配套 .json 文件读取血糖数据
    3. 支持时间范围过滤
    4. 自动处理时间戳转换（北京时间 <-> Unix timestamp）
    """
    
    def __init__(self, db_root: str = "./Dataset/Tao_db"):
        """
        Args:
            db_root: 数据库文件根目录
        """
        self.db_root = Path(db_root)
    
    @staticmethod
    def beijing_to_utc_timestamp(beijing_time_str: str) -> float:
        """
        将北京时间字符串转换为 UTC Unix timestamp
        
        Args:
            beijing_time_str: 北京时间字符串，格式 "YYYY-MM-DD HH:MM:SS"
        
        Returns:
            UTC Unix timestamp (秒)
        """
        # 解析北京时间
        dt = pd.to_datetime(beijing_time_str)
        # 减去8小时得到UTC
        utc_dt = dt - pd.Timedelta(hours=8)
        # 转换为Unix timestamp
        return utc_dt.timestamp()
    
    @staticmethod
    def utc_timestamp_to_beijing(ts: float) -> datetime:
        """
        将 UTC Unix timestamp 转换为北京时间 datetime
        
        Args:
            ts: UTC Unix timestamp (秒)
        
        Returns:
            北京时间 datetime 对象
        """
        return pd.to_datetime(ts, unit='s') + pd.Timedelta(hours=8)
    
    # def get_db_time_range(self, db_path: Path) -> Tuple[float, float]:
    #     """
    #     获取数据库中 spectrum 数据的时间范围
        
    #     Args:
    #         db_path: 数据库文件路径
        
    #     Returns:
    #         (min_ts, max_ts) - UTC Unix timestamp
            
    #     Note:
    #         数据库中的 ts 字段存储的是 UTC Unix timestamp
    #     """
    #     conn = sqlite3.connect(str(db_path))
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT MIN(ts), MAX(ts) FROM spectrum")
    #     min_ts, max_ts = cursor.fetchone()
    #     conn.close()
    #     return min_ts, max_ts
    
    def get_db_time_range(self, db_path: Path) -> Tuple[float, float]:
        # 1. 立即转换为绝对路径并打印，确认程序到底在看哪
        absolute_path = Path(db_path).resolve()
        print(f"正在获取数据库时间范围，查询路径: {absolute_path}")
        
        if absolute_path.is_dir():
            print(f"警告：这是一个文件夹，不是文件！文件夹内的内容有：{os.listdir(absolute_path)}")
            # 如果是文件夹，你可能需要拼接文件名，例如：
            # absolute_path = absolute_path / "spectrum.db" 
            raise IsADirectoryError(f"路径 {absolute_path} 是一个文件夹，请指向具体的 .db 文件")
        
        # if not absolute_path.exists():
        #     print("警告：文件不存在")
        
        # 2. 检查文件是否存在
        if not absolute_path.exists():
            raise FileNotFoundError(f"致命错误：数据库文件不存在！\n查询路径: {absolute_path}")
        
        # 3. 检查文件大小 (如果是0字节，肯定没表)
        if absolute_path.stat().st_size == 0:
            raise ValueError(f"致命错误：数据库文件是空的 (0字节)！\n文件路径: {absolute_path}")

        # 4. 使用 URI 模式连接（只读模式），如果文件有问题会报错而不是新建
        try:
            # uri=True 配合 mode=rw 可以防止自动创建新文件
            conn = sqlite3.connect(f"file:{absolute_path}?mode=rw", uri=True)
            cursor = conn.cursor()

            # 检查所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if "spectrum" not in tables:
                conn.close()
                raise ValueError(f"数据库中找不到 'spectrum' 表。绝对路径: {absolute_path}\n可用表有: {tables}")

            cursor.execute("SELECT MIN(ts), MAX(ts) FROM spectrum")
            result = cursor.fetchone()
            conn.close()
            
            if result[0] is None:
                return 0.0, 0.0
                
            return result
            
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"无法打开数据库: {e}\n路径: {absolute_path}")
        
    def parse_spectrum_blob(self, data: bytes) -> np.ndarray:
        """
        解析 spectrum 数据 blob
        
        Args:
            data: 二进制数据 (1001个uint32小端序)
        
        Returns:
            1001维的numpy数组
        """
        if data is None:
            return None
        
        if isinstance(data, bytes):
            # 每4个字节代表一个uint32值
            return np.frombuffer(data, dtype=np.uint32)
        elif isinstance(data, str):
            # 如果是hex字符串
            try:
                byte_data = bytes.fromhex(data)
                return np.frombuffer(byte_data, dtype=np.uint32)
            except:
                return None
        return None
    
    def load_spectrum_from_db(
        self, 
        db_path: Path, 
        start_ts_utc: Optional[float] = None,
        end_ts_utc: Optional[float] = None
    ) -> pd.DataFrame:
        """
        从数据库加载 spectrum 数据
        
        Args:
            db_path: 数据库文件路径
            start_ts_utc: 起始时间戳 (UTC Unix timestamp)
            end_ts_utc: 结束时间戳 (UTC Unix timestamp)
        
        Returns:
            DataFrame，包含 timestamp (UTC Unix timestamp) 和 power_0 到 power_N 列
            
        Note:
            - 数据库 spectrum 表的 ts 字段为 UTC Unix timestamp
            - 返回的 timestamp 列也是 UTC Unix timestamp
        """
        conn = sqlite3.connect(str(db_path))
        
        # 构建查询条件 - 使用 UTC timestamp 过滤
        if start_ts_utc is not None and end_ts_utc is not None:
            query = f"SELECT ts, data FROM spectrum WHERE ts >= {start_ts_utc} AND ts <= {end_ts_utc}"
        else:
            query = "SELECT ts, data FROM spectrum"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        # 解析 spectrum blob 数据
        spectrums = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            spectrum = self.parse_spectrum_blob(row['data'])
            if spectrum is not None:
                spectrums.append(spectrum)
                valid_indices.append(idx)
        
        if not spectrums:
            return pd.DataFrame()
        
        # 获取实际的spectrum长度
        actual_length = len(spectrums[0])
        
        # 创建power列名
        power_cols = [f'power_{i}' for i in range(actual_length)]
        
        # 构建结果DataFrame - timestamp 列为 UTC Unix timestamp
        result_df = pd.DataFrame(spectrums, columns=power_cols)
        result_df.insert(0, 'timestamp', df.loc[valid_indices, 'ts'].values)
        
        # 按时间戳排序
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)
        
        return result_df
    
    def load_spectrum_from_multiple_dbs(
        self,
        db_paths: List[str],
        start_ts_utc: Optional[float] = None,
        end_ts_utc: Optional[float] = None,
        include_db_source: bool = False
    ) -> pd.DataFrame:
        """
        从多个数据库文件加载并合并 spectrum 数据
        
        用于处理实验中断导致的多个db文件的情况
        
        Args:
            db_paths: 数据库文件路径列表
            start_ts_utc: 起始时间戳 (UTC Unix timestamp)
            end_ts_utc: 结束时间戳 (UTC Unix timestamp)
        
        Returns:
            合并后的DataFrame，包含 timestamp (UTC) 列和 power_* 列，按时间戳排序
            
        Note:
            所有db文件的 ts 字段均为 UTC Unix timestamp
        """
        all_dfs = []
        
        for db_path in db_paths:
            db_path = Path(db_path)
            if not db_path.exists():
                print(f"    ⚠ 跳过不存在的文件: {db_path.name}")
                continue
                
            df = self.load_spectrum_from_db(db_path, start_ts_utc, end_ts_utc)
            if not df.empty:
                if include_db_source:
                    df['__db_file'] = str(db_path)
                all_dfs.append(df)
                print(f"    ✓ {db_path.name}: {len(df)} 条记录")
        
        if not all_dfs:
            return pd.DataFrame()
        
        # 合并所有DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # 按时间戳排序并去重（以防万一有重复数据）
        combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        return combined_df
    
    def load_aux_sensors_from_multiple_dbs(
        self,
        db_paths: List[str],
        start_ts_utc: Optional[float] = None,
        end_ts_utc: Optional[float] = None
    ) -> AuxSensorData:
        """
        从多个数据库文件加载并合并辅助传感器数据
        
        Args:
            db_paths: 数据库文件路径列表
            start_ts_utc: 起始时间戳 (UTC Unix timestamp)
            end_ts_utc: 结束时间戳 (UTC Unix timestamp)
        
        Returns:
            合并后的 AuxSensorData 对象，所有 timestamp 列为 UTC Unix timestamp
            
        Note:
            所有传感器表的 ts 字段均为 UTC Unix timestamp
        """
        all_bme = []
        all_ppg = []
        all_t117 = []
        all_icm = []
        
        for db_path in db_paths:
            db_path = Path(db_path)
            if not db_path.exists():
                continue
                
            aux_data = self.load_aux_sensors_from_db(db_path, start_ts_utc, end_ts_utc)
            
            if not aux_data.bme680.empty:
                all_bme.append(aux_data.bme680)
            if not aux_data.ppg.empty:
                all_ppg.append(aux_data.ppg)
            if not aux_data.t117.empty:
                all_t117.append(aux_data.t117)
            if not aux_data.icm.empty:
                all_icm.append(aux_data.icm)
        
        # 合并各传感器数据
        def merge_dfs(dfs):
            if not dfs:
                return pd.DataFrame()
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            return combined
        
        return AuxSensorData(
            bme680=merge_dfs(all_bme),
            ppg=merge_dfs(all_ppg),
            t117=merge_dfs(all_t117),
            icm=merge_dfs(all_icm)
        )
    
    def get_multiple_db_time_range(self, db_paths: List[str]) -> Tuple[float, float]:
        """
        获取多个数据库中 spectrum 数据的时间范围
        
        Args:
            db_paths: 数据库文件路径列表
        
        Returns:
            (min_ts, max_ts) - UTC Unix timestamp，所有db文件的总时间范围
            
        Note:
            数据库中的 ts 字段存储的是 UTC Unix timestamp
        """
        min_ts = float('inf')
        max_ts = float('-inf')
        
        for db_path in db_paths:
            db_path = Path(db_path)
            if not db_path.exists():
                continue
            try:
                db_min, db_max = self.get_db_time_range(db_path)
                min_ts = min(min_ts, db_min)
                max_ts = max(max_ts, db_max)
            except Exception as e:
                print(f"    ⚠ 无法读取时间范围 {db_path.name}: {e}")
                continue
        
        if min_ts == float('inf') or max_ts == float('-inf'):
            raise ValueError("无法获取任何db文件的时间范围")
        
        return min_ts, max_ts

    def load_glucose_from_json(
        self,
        json_path: Path,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None
    ) -> pd.DataFrame:
        """
        从 JSON 文件加载血糖数据
        
        JSON 格式: {"times": [...], "username": "...", "values": [...]}
        times 格式: "YYYY-MM-DD HH:MM:SS" (北京时间)
        
        Args:
            json_path: JSON 文件路径
            start_ts: 起始时间戳 (UTC)
            end_ts: 结束时间戳 (UTC)
        
        Returns:
            DataFrame，包含 time_str, glucose, ts(UTC), datetime(北京时间)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if not isinstance(json_data, dict) or 'times' not in json_data or 'values' not in json_data:
            raise ValueError(f"JSON格式不正确。需要包含 'times' 和 'values' 字段。实际: {json_data.keys() if isinstance(json_data, dict) else type(json_data)}")
        
        # 创建DataFrame
        df = pd.DataFrame({
            'time_str': json_data['times'],
            'glucose': json_data['values']
        })
        
        # 将北京时间字符串转换为 UTC Unix timestamp
        # times 格式: "2025-10-22 15:00:45" (北京时间)
        df['datetime'] = pd.to_datetime(df['time_str'])  # 解析为 datetime (仍为北京时间)
        df['ts'] = (df['datetime'] - pd.Timedelta(hours=8)).astype('int64') // 10**9  # 转为 UTC timestamp
        
        # 使用 UTC timestamp 过滤时间范围
        if start_ts is not None and end_ts is not None:
            df = df[(df['ts'] >= start_ts) & (df['ts'] <= end_ts)]
        
        return df.reset_index(drop=True)
    
    def load_glucose_from_db(
        self,
        glucose_db_path: Path,
        start_ts_utc: Optional[float] = None,
        end_ts_utc: Optional[float] = None
    ) -> pd.DataFrame:
        """
        从血糖数据库文件加载血糖数据 (blood_sugar 表)
        
        blood_sugar 表结构:
        - create_time: Unix timestamp (UTC)
        - blood_sugar: 血糖值 (mmol/L)
        
        Args:
            glucose_db_path: 血糖数据库文件路径
            start_ts_utc: 起始时间戳 (UTC Unix timestamp)
            end_ts_utc: 结束时间戳 (UTC Unix timestamp)
        
        Returns:
            DataFrame，包含 ts(UTC Unix timestamp), glucose, datetime(北京时间)
            
        Note:
            - 数据库 blood_sugar 表的 create_time 字段为 UTC Unix timestamp
            - 返回的 ts 列为 UTC Unix timestamp
            - datetime 列为转换后的北京时间（仅用于显示）
        """
        conn = sqlite3.connect(str(glucose_db_path))
        
        try:
            # 检查表是否存在
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='blood_sugar';")
            if not cursor.fetchone():
                raise ValueError(f"血糖数据库中未找到 blood_sugar 表: {glucose_db_path}")
            
            # 构建查询条件 - 使用 UTC timestamp 过滤
            if start_ts_utc is not None and end_ts_utc is not None:
                query = f"SELECT create_time as ts, blood_sugar as glucose FROM blood_sugar WHERE create_time >= {start_ts_utc} AND create_time <= {end_ts_utc} ORDER BY create_time"
            else:
                query = "SELECT create_time as ts, blood_sugar as glucose FROM blood_sugar ORDER BY create_time"
            
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                return pd.DataFrame()
            
            # 添加北京时间列（UTC timestamp + 8小时）
            df['datetime'] = pd.to_datetime(df['ts'], unit='s') + pd.Timedelta(hours=8)
            
            return df.reset_index(drop=True)
            
        finally:
            conn.close()
    
    def load_aux_sensors_from_db(
        self,
        db_path: Path,
        start_ts_utc: Optional[float] = None,
        end_ts_utc: Optional[float] = None
    ) -> AuxSensorData:
        """
        从数据库加载辅助传感器数据
        
        Args:
            db_path: 数据库文件路径
            start_ts_utc: 起始时间戳 (UTC Unix timestamp)
            end_ts_utc: 结束时间戳 (UTC Unix timestamp)
        
        Returns:
            AuxSensorData 对象，包含 bme680, ppg, t117, icm 四种传感器数据
            所有 DataFrame 的 timestamp 列为 UTC Unix timestamp
            
        Note:
            所有传感器表的 ts 字段均为 UTC Unix timestamp
        """
        conn = sqlite3.connect(str(db_path))
        
        # 构建时间过滤条件 - 使用 UTC timestamp 过滤
        time_filter = ""
        if start_ts_utc is not None and end_ts_utc is not None:
            time_filter = f" WHERE ts >= {start_ts_utc} AND ts <= {end_ts_utc}"
        
        # 读取 BME680 数据 (env_bme 表)
        try:
            df_bme = pd.read_sql_query(
                f"SELECT ts as timestamp, t as temperature_c, p as pressure_hpa, h as humidity_pct FROM env_bme{time_filter}",
                conn
            )
        except Exception as e:
            print(f"  ⚠ 读取 env_bme 失败: {e}")
            df_bme = pd.DataFrame()
        
        # 读取 PPG 数据 (cheez_ppg 表)
        try:
            df_ppg = pd.read_sql_query(
                f"SELECT ts as timestamp, raw, avg, filter, peak, hr, hrv FROM cheez_ppg{time_filter}",
                conn
            )
        except Exception as e:
            try:
                df_ppg = pd.read_sql_query(
                    f"SELECT ts as timestamp, raw FROM cheez_ppg{time_filter}",
                    conn
                )
                print(f"  ⚠ cheez_ppg只读取raw列（扩展列不可用）: {e}")
            except Exception as inner_e:
                print(f"  ⚠ 读取 cheez_ppg 失败: {inner_e}")
                df_ppg = pd.DataFrame()
        
        # 读取 T117 数据 (env_temp 表)
        try:
            df_t117 = pd.read_sql_query(
                f"SELECT ts as timestamp, t as temperature_c FROM env_temp{time_filter}",
                conn
            )
        except Exception as e:
            print(f"  ⚠ 读取 env_temp 失败: {e}")
            df_t117 = pd.DataFrame()
        
        # 读取 ICM 数据 (imu_icm 表)
        try:
            df_icm = pd.read_sql_query(
                f"SELECT ts as timestamp, ax as acc_x, ay as acc_y, az as acc_z, gx as gyr_x, gy as gyr_y, gz as gyr_z FROM imu_icm{time_filter}",
                conn
            )
        except Exception as e:
            print(f"  ⚠ 读取 imu_icm 失败: {e}")
            df_icm = pd.DataFrame()
        
        conn.close()
        
        # 排序
        if not df_bme.empty:
            df_bme = df_bme.sort_values('timestamp').reset_index(drop=True)
        if not df_ppg.empty:
            df_ppg = df_ppg.sort_values('timestamp').reset_index(drop=True)
        if not df_t117.empty:
            df_t117 = df_t117.sort_values('timestamp').reset_index(drop=True)
        if not df_icm.empty:
            df_icm = df_icm.sort_values('timestamp').reset_index(drop=True)
        
        return AuxSensorData(df_bme, df_ppg, df_t117, df_icm)
    
    def align_spectrum_glucose(
        self,
        spectrum_df: pd.DataFrame,
        glucose_df: pd.DataFrame,
        interpolation_method: str = 'pchip'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对齐频谱数据和血糖数据，并插值
        
        Args:
            spectrum_df: 频谱数据 DataFrame (包含 timestamp 和 power_* 列)
            glucose_df: 血糖数据 DataFrame (包含 ts 和 glucose 列)
            interpolation_method: 插值方法
        
        Returns:
            (spectrum_features, glucose_labels, spectrum_timestamps)
        """
        # 提取功率列作为特征
        power_cols = [col for col in spectrum_df.columns if col.startswith('power_')]
        spectrum_features = spectrum_df[power_cols].to_numpy()
        spectrum_timestamps = spectrum_df['timestamp'].to_numpy()
        
        # 获取血糖时间和值
        glucose_times = glucose_df['ts'].values
        glucose_values = glucose_df['glucose'].values
        
        # 排序并去重
        sort_idx = np.argsort(glucose_times)
        glucose_times = glucose_times[sort_idx]
        glucose_values = glucose_values[sort_idx]
        
        # 去重
        unique_mask = np.concatenate([[True], np.diff(glucose_times) > 0])
        glucose_times = glucose_times[unique_mask]
        glucose_values = glucose_values[unique_mask]
        
        n_glucose_points = len(glucose_times)
        
        if n_glucose_points < 2:
            raise ValueError(f"血糖数据点不足（需要至少2个点进行插值，当前只有{n_glucose_points}个点）")
        
        # 计算平均采样间隔
        if n_glucose_points > 1:
            glucose_intervals = np.diff(glucose_times)
            avg_glucose_interval = np.mean(glucose_intervals)
            print(f"    血糖数据: {n_glucose_points}个采样点, 平均间隔: {avg_glucose_interval/60:.1f}分钟")
        
        # 选择插值方法
        if interpolation_method == 'pchip' and n_glucose_points >= 3:
            glucose_interp = PchipInterpolator(glucose_times, glucose_values, extrapolate=True)
            glucose_labels = glucose_interp(spectrum_timestamps)
            print(f"    ✓ 使用PCHIP插值")
        elif interpolation_method == 'akima' and n_glucose_points >= 5:
            glucose_interp = Akima1DInterpolator(glucose_times, glucose_values)
            glucose_labels = glucose_interp(spectrum_timestamps, extrapolate=True)
            print(f"    ✓ 使用Akima插值")
        elif interpolation_method == 'cubic' and n_glucose_points >= 4:
            glucose_interp = interp1d(
                glucose_times, glucose_values,
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            glucose_labels = glucose_interp(spectrum_timestamps)
            print(f"    ✓ 使用三次样条插值")
        else:
            glucose_interp = interp1d(
                glucose_times, glucose_values,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            glucose_labels = glucose_interp(spectrum_timestamps)
            print(f"    ✓ 使用线性插值")
        
        return spectrum_features, glucose_labels, spectrum_timestamps


class DBGlucoseDatasetLoader(GlucoseDatasetLoader):
    """
    数据库格式的血糖数据集加载器
    
    继承自 GlucoseDatasetLoader，重写关键方法以支持 DB 格式
    """
    
    def __init__(self, dataset_root: str = "./Dataset", db_config_file: str = None):
        """
        Args:
            dataset_root: Dataset 根目录路径
            db_config_file: DB 配置文件路径（可选，包含实验时间范围等配置）
        """
        super().__init__(dataset_root)
        self.db_loader = DBDataLoader(dataset_root)
        self.db_config_file = db_config_file
        self._experiments_config = None

    def _resolve_db_root(self, user_name: Optional[str] = None) -> Path:
        """
        解析当前用户的 DB 根目录

        兼容两种目录结构：
        1) dataset_root 已经指向用户目录 (e.g., Dataset/Tao_db)
        2) dataset_root 是总目录，用户目录在其下 (e.g., Dataset/Tao_db)
        """
        base_root = self.db_loader.db_root
        if user_name:
            candidate = base_root / user_name
            if candidate.exists() and candidate.is_dir():
                return candidate
        return base_root

    @staticmethod
    def _resolve_db_file_path(path_str: str) -> Path:
        """
        解析并兜底DB文件路径。

        兼容配置中常见两种写法：
        1) /root/DEV_xxx/DEV_xxx.db
        2) /root/DEV_xxx.db  (实际文件在同名子目录下)
        """
        candidate = Path(path_str).with_suffix('.db')
        if candidate.exists():
            return candidate

        # 兜底: /a/b/DEV_xxx.db -> /a/b/DEV_xxx/DEV_xxx.db
        fallback = candidate.parent / candidate.stem / candidate.name
        if fallback.exists():
            print(f"  ⚠ 路径自动修正: {candidate} -> {fallback}")
            return fallback

        return candidate

    def validate_user_db_files(
        self,
        user_name: str,
        experiment_filter: Optional[List[str]] = None
    ) -> List[str]:
        """
        轻量级校验：仅检查用户实验配置中的DB文件是否存在。

        Returns:
            缺失文件列表（已解析为绝对路径字符串）
        """
        db_root = self._resolve_db_root(user_name)
        experiments = self.load_experiments_config(db_root=db_root)

        if experiment_filter:
            filtered = []
            for exp in experiments:
                db_path_obj = Path(exp.db_path)
                folder_name = db_path_obj.name if db_path_obj.is_dir() else db_path_obj.parent.name
                if exp.experiment_name in experiment_filter or folder_name in experiment_filter:
                    filtered.append(exp)
            experiments = filtered

        missing = []
        for exp in experiments:
            db_paths = exp.get_db_paths()
            for p in db_paths:
                resolved = self._resolve_db_file_path(p)
                if not resolved.exists():
                    missing.append(str(resolved.resolve()))

        return missing
    
    def load_experiments_config(
        self,
        config_path: Optional[str] = None,
        db_root: Optional[Path] = None
    ) -> List[DBExperimentConfig]:
        """
        加载实验配置
        
        配置文件格式 (JSON):
        [
            {
                "db_path": "DEV_002_qiangtao_260121/DEV_002_qiangtao_260121.db",
                "glucose_db_path": "qiangtao_glucose.db",
                "start_time": "2026-01-21 17:03:00",
                "end_time": "2026-01-21 20:21:00",
                "experiment_name": "260121_1703_Tao"
            },
            {
                "db_path": "DEV_002_qiangtao_260123/DEV_002_qiangtao_260123.db",
                "json_path": "qiangtao.json",
                "start_time": "2026-01-23 12:57:00",
                "end_time": "2026-01-23 19:37:00",
                "experiment_name": "260123_1257_Tao"
            }
        ]
        
        字段说明:
        - db_path: 频谱数据库路径 (必填)
        - glucose_db_path: 血糖数据库路径 (可选，优先使用)
        - json_path: 血糖JSON文件路径 (可选，glucose_db_path不存在时使用)
        - start_time/end_time: 实验时间范围 (北京时间)
        - experiment_name: 实验名称
        
        支持的目录结构：
        1. 配置文件在 db_root 根目录: Dataset/Tao_db/experiments_config.json
        2. 配置文件在子目录中: Dataset/Tao_db/DEV_xxx/experiments_config.json
        """
        config_path = config_path or self.db_config_file
        db_root = db_root or self.db_loader.db_root
        
        # 如果没有指定配置文件，尝试查找默认配置文件
        if config_path is None:
            # 首先在根目录查找
            default_config = db_root / "experiments_config.json"
            if default_config.exists():
                config_path = str(default_config)
                print(f"  ✓ 使用配置文件: {default_config}")
            else:
                # 在子目录中查找配置文件
                for subdir in db_root.iterdir():
                    if subdir.is_dir():
                        subdir_config = subdir / "experiments_config.json"
                        if subdir_config.exists():
                            config_path = str(subdir_config)
                            print(f"  ✓ 使用配置文件: {subdir_config}")
                            break
        
        if config_path and Path(config_path).exists():
            config_file_path = Path(config_path)
            # 配置文件所在的目录（用于解析相对路径）
            config_dir = config_file_path.parent
            
            with open(config_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
            
            # 处理相对路径：将相对路径转换为绝对路径（相对于配置文件所在目录）
            processed_configs = []
            for cfg in configs:
                # 如果 db_path 是相对路径，转换为绝对路径（相对于配置文件目录）
                if not Path(cfg['db_path']).is_absolute():
                    cfg['db_path'] = str(config_dir / cfg['db_path'])
                # 如果 json_path 是相对路径，转换为绝对路径（相对于配置文件目录）
                if 'json_path' in cfg and cfg['json_path'] and not Path(cfg['json_path']).is_absolute():
                    cfg['json_path'] = str(config_dir / cfg['json_path'])
                # 如果 glucose_db_path 是相对路径，转换为绝对路径（相对于配置文件目录）
                if 'glucose_db_path' in cfg and cfg['glucose_db_path'] and not Path(cfg['glucose_db_path']).is_absolute():
                    cfg['glucose_db_path'] = str(config_dir / cfg['glucose_db_path'])
                # 如果 db_files 存在，处理列表中每个路径
                if 'db_files' in cfg and cfg['db_files']:
                    cfg['db_files'] = [
                        str(config_dir / f) if not Path(f).is_absolute() else f 
                        for f in cfg['db_files']
                    ]
                # 确保 json_path 有默认值（如果未提供）
                cfg.setdefault('json_path', '')
                cfg.setdefault('glucose_db_path', None)
                cfg.setdefault('db_files', None)
                processed_configs.append(DBExperimentConfig(**cfg))
            
            return processed_configs
        
        # 如果没有配置文件，尝试自动发现
        return self._auto_discover_experiments(db_root=db_root)
    
    def _auto_discover_experiments(
        self,
        db_root: Optional[Path] = None,
        user_name: Optional[str] = None
    ) -> List[DBExperimentConfig]:
        """
        自动发现实验配置
        
        查找 db_root 目录及其子目录下的所有 spectrum .db 文件和对应的血糖数据
        支持两种血糖数据来源:
        1. glucose .db 文件 (blood_sugar 表) - 优先
        2. .json 文件 (times + values)
        """
        experiments = []
        db_root = db_root or self.db_loader.db_root
        
        if not db_root.exists():
            print(f"⚠ DB目录不存在: {db_root}")
            return experiments
        
        # 排除的文件名（不是频谱数据）
        excluded_json_names = {'experiments_config.json', 'config.json'}
        # 血糖数据库文件名关键词（用于识别 glucose db）
        glucose_db_keywords = ['glucose', 'blood_sugar', 'cgm']
        
        # 收集所有要搜索的目录（根目录 + 子目录）
        search_dirs = [db_root]
        for item in db_root.iterdir():
            if item.is_dir():
                search_dirs.append(item)
        
        # 先收集所有可能的 glucose db 文件
        glucose_db_candidates = []
        for search_dir in search_dirs:
            for db_file in search_dir.glob("*.db"):
                # 检查是否是 glucose db (通过文件名或检查是否有 blood_sugar 表)
                is_glucose_db = any(kw in db_file.stem.lower() for kw in glucose_db_keywords)
                if not is_glucose_db:
                    # 检查是否有 blood_sugar 表
                    try:
                        conn = sqlite3.connect(str(db_file))
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='blood_sugar';")
                        if cursor.fetchone():
                            is_glucose_db = True
                        conn.close()
                    except:
                        pass
                if is_glucose_db:
                    glucose_db_candidates.append(db_file)
        
        # 在每个目录中查找 spectrum .db 文件
        for search_dir in search_dirs:
            for db_file in search_dir.glob("*.db"):
                # 跳过 glucose db 文件
                if db_file in glucose_db_candidates:
                    continue
                
                # 检查是否有 spectrum 表（确认是频谱数据库）
                try:
                    conn = sqlite3.connect(str(db_file))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='spectrum';")
                    if not cursor.fetchone():
                        conn.close()
                        continue  # 不是频谱数据库，跳过
                    conn.close()
                except:
                    continue
                
                # 获取数据库时间范围
                try:
                    min_ts, max_ts = self.db_loader.get_db_time_range(db_file)
                    
                    # 转换为北京时间字符串
                    start_time = self.db_loader.utc_timestamp_to_beijing(min_ts).strftime("%Y-%m-%d %H:%M:%S")
                    end_time = self.db_loader.utc_timestamp_to_beijing(max_ts).strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 优先查找 glucose .db 文件
                    glucose_db_path = None
                    json_path = None
                    
                    # 1. 在同目录和父目录查找 glucose db
                    search_glucose_dirs = [search_dir, db_root]
                    for gdir in search_glucose_dirs:
                        for gdb in glucose_db_candidates:
                            if gdb.parent == gdir:
                                glucose_db_path = gdb
                                break
                        if glucose_db_path:
                            break
                    
                    # 2. 如果没有找到 glucose db，查找 JSON 文件
                    if glucose_db_path is None:
                        json_candidates = [
                            search_dir / f"{db_file.stem}.json",
                            search_dir / f"{db_file.stem}_glucose.json",
                        ]
                        
                        # 也尝试查找目录下的任何 .json 文件（排除配置文件）
                        for json_file in search_dir.glob("*.json"):
                            if json_file.name.lower() not in excluded_json_names and json_file not in json_candidates:
                                json_candidates.append(json_file)
                        
                        for candidate in json_candidates:
                            if candidate.name.lower() in excluded_json_names:
                                continue
                            if candidate.exists():
                                json_path = candidate
                                break
                    
                    # 必须有血糖数据来源
                    if glucose_db_path is None and json_path is None:
                        print(f"⚠ 未找到 {db_file.name} 对应的血糖数据文件")
                        continue
                    
                    # 生成实验名称（基于数据库文件名和时间）
                    dt = self.db_loader.utc_timestamp_to_beijing(min_ts)
                    user_tag = user_name or db_root.name
                    experiment_name = f"{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}_{user_tag}"
                    
                    experiments.append(DBExperimentConfig(
                        db_path=str(db_file),
                        json_path=str(json_path) if json_path else '',
                        start_time=start_time,
                        end_time=end_time,
                        experiment_name=experiment_name,
                        glucose_db_path=str(glucose_db_path) if glucose_db_path else None
                    ))
                    
                    print(f"  ✓ 发现实验: {experiment_name}")
                    print(f"    Spectrum DB: {db_file.name}")
                    if glucose_db_path:
                        print(f"    Glucose DB: {glucose_db_path.name}")
                    else:
                        print(f"    Glucose JSON: {json_path.name}")
                    print(f"    时间范围: {start_time} - {end_time}")
                    
                except Exception as e:
                    print(f"⚠ 处理 {db_file.name} 时出错: {e}")
                    continue
        
        return experiments
    
    def load_db_experiment(
        self,
        experiment_config: DBExperimentConfig,
        interpolation_method: str = 'pchip',
        data_fusion: bool = False,
        fusion_method: str = 'concat',
        fusion_stage: str = 'early',
        downsample: bool = False,
        downsample_interval: float = 1.0,
        smooth_config: Optional[dict] = None,
        icm_mode: str = 'raw',
        fusion_config: Optional[dict] = None,
        user_name: Optional[str] = None,
        config_dir: Optional[Path] = None,
        track_db_file_source: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
        """
        加载单个 DB 格式实验的数据
        
        支持多个db文件（实验中断情况）
        
        Args:
            experiment_config: 实验配置
            config_dir: 配置文件所在目录，用于解析相对路径
            其他参数与 load_user_experiment 相同
        
        Returns:
            (features, labels, timestamps, metadata) 或 None
        """
        try:
            print(f"\n加载实验: {experiment_config.experiment_name}")
            _db_debug_mem(f"exp={experiment_config.experiment_name} start")
            
            # 获取所有db文件路径
            db_paths = experiment_config.get_db_paths(base_dir=config_dir)
            db_paths = [str(self._resolve_db_file_path(p)) for p in db_paths]

            missing_files = [p for p in db_paths if not Path(p).exists()]
            if missing_files:
                missing_text = "\n".join([f"    - {Path(p).resolve()}" for p in missing_files])
                raise FileNotFoundError(
                    "实验配置中的数据库文件不存在，已中止加载以避免继续占用内存:\n"
                    f"  实验: {experiment_config.experiment_name}\n"
                    f"  缺失文件:\n{missing_text}"
                )
            
            print(f"  - 解析到 {len(db_paths)} 个DB文件")
            print(db_paths)
            
            if len(db_paths) > 1:
                print(f"  - DB文件 ({len(db_paths)}个，实验分段):")
                for p in db_paths:
                    print(f"    • {Path(p).name}")
            else:
                print(f"  - DB文件: {Path(db_paths[0]).name}")
            print(f"  - 配置时间范围: {experiment_config.start_time} - {experiment_config.end_time}")
            
            # 转换配置的北京时间为 UTC timestamp (用于数据库查询)
            config_start_ts = self.db_loader.beijing_to_utc_timestamp(experiment_config.start_time)
            config_end_ts = self.db_loader.beijing_to_utc_timestamp(experiment_config.end_time)
            
            # 获取所有数据库中 spectrum 数据的 UTC 时间范围
            if len(db_paths) > 1:
                db_min_ts, db_max_ts = self.db_loader.get_multiple_db_time_range(db_paths)
            else:
                db_min_ts, db_max_ts = self.db_loader.get_db_time_range(Path(db_paths[0]))
            
            # 转换为北京时间显示给用户
            db_start_beijing = self.db_loader.utc_timestamp_to_beijing(db_min_ts).strftime("%Y-%m-%d %H:%M:%S")
            db_end_beijing = self.db_loader.utc_timestamp_to_beijing(db_max_ts).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  - 数据库时间范围: {db_start_beijing} - {db_end_beijing}")
            
            # 取配置和数据库时间范围的交集 (UTC timestamp)
            start_ts_utc = max(config_start_ts, db_min_ts)
            end_ts_utc = min(config_end_ts, db_max_ts)
            
            # 检查交集是否有效
            if start_ts_utc >= end_ts_utc:
                print(f"  ✗ 配置时间范围与数据库时间范围无交集")
                return None
            
            # 显示实际使用的时间范围 (转换为北京时间)
            actual_start_beijing = self.db_loader.utc_timestamp_to_beijing(start_ts_utc).strftime("%Y-%m-%d %H:%M:%S")
            actual_end_beijing = self.db_loader.utc_timestamp_to_beijing(end_ts_utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  - 实际使用范围: {actual_start_beijing} - {actual_end_beijing}")
            
            # 1. 加载频谱数据（支持多db文件合并） - 使用 UTC timestamp 查询
            print("  - 加载频谱数据...")
            if len(db_paths) > 1:
                spectrum_df = self.db_loader.load_spectrum_from_multiple_dbs(
                    db_paths, start_ts_utc, end_ts_utc,
                    include_db_source=track_db_file_source
                )
            else:
                spectrum_df = self.db_loader.load_spectrum_from_db(Path(db_paths[0]), start_ts_utc, end_ts_utc)
                if track_db_file_source:
                    spectrum_df['__db_file'] = str(Path(db_paths[0]))
            
            if spectrum_df.empty:
                print(f"  ✗ 未找到频谱数据")
                return None
            
            print(f"    ✓ 共 {len(spectrum_df)} 条频谱记录")
            _db_debug_mem(f"exp={experiment_config.experiment_name} after spectrum load")
            
            # 2. 加载血糖数据 (优先从 glucose_db_path 读取，否则从 json_path 读取)
            # 使用 UTC timestamp 过滤
            print("  - 加载血糖数据...")
            glucose_db_path = experiment_config.glucose_db_path
            json_path = experiment_config.json_path
            
            # 解析相对路径
            if glucose_db_path and config_dir and not Path(glucose_db_path).is_absolute():
                glucose_db_path = str(config_dir / glucose_db_path)
            if json_path and config_dir and not Path(json_path).is_absolute():
                json_path = str(config_dir / json_path)
            
            if glucose_db_path and Path(glucose_db_path).exists():
                print(f"    从DB加载: {Path(glucose_db_path).name}")
                glucose_df = self.db_loader.load_glucose_from_db(
                    Path(glucose_db_path),
                    start_ts_utc, end_ts_utc
                )
            elif json_path and Path(json_path).exists():
                print(f"    从JSON加载: {Path(json_path).name}")
                glucose_df = self.db_loader.load_glucose_from_json(
                    Path(json_path),
                    start_ts_utc, end_ts_utc
                )
            else:
                print(f"  ✗ 未配置血糖数据来源（需要 glucose_db_path 或 json_path）")
                return None
            
            if glucose_df.empty:
                print(f"  ✗ 未找到血糖数据")
                return None
            
            print(f"    ✓ {len(glucose_df)} 条血糖记录")
            _db_debug_mem(f"exp={experiment_config.experiment_name} after glucose load")
            
            # 3. 时间对齐和插值
            print("  - 对齐时间戳并插值...")
            spectrum_features, labels, timestamps = self.db_loader.align_spectrum_glucose(
                spectrum_df,
                glucose_df,
                interpolation_method
            )
            _db_debug_mem(f"exp={experiment_config.experiment_name} after align/interp")
            
            # 3.5 数据平滑处理（可选）
            if smooth_config and smooth_config.get('smooth_data', False):
                print("  - 应用数据平滑处理...")
                from .smoothing import apply_smoothing_pipeline
                
                time_smooth_method = smooth_config.get('time_smooth_method', 'none')
                spectrum_smooth_method = smooth_config.get('spectrum_smooth_method', 'none')
                
                time_smooth_params = {}
                if time_smooth_method in ['moving_average', 'median', 'savgol']:
                    time_smooth_params['window_size'] = smooth_config.get('time_smooth_window', 5)
                if time_smooth_method == 'savgol':
                    time_smooth_params['poly_order'] = 3
                if time_smooth_method == 'gaussian':
                    time_smooth_params['sigma'] = smooth_config.get('time_smooth_sigma', 2.0)
                if time_smooth_method == 'ema':
                    time_smooth_params['alpha'] = smooth_config.get('time_smooth_alpha', 0.3)
                
                spectrum_smooth_params = {}
                if spectrum_smooth_method in ['moving_average', 'median', 'savgol']:
                    spectrum_smooth_params['window_size'] = smooth_config.get('spectrum_smooth_window', 5)
                if spectrum_smooth_method == 'savgol':
                    spectrum_smooth_params['poly_order'] = 3
                if spectrum_smooth_method == 'gaussian':
                    spectrum_smooth_params['sigma'] = smooth_config.get('spectrum_smooth_sigma', 2.0)
                
                spectrum_features, smooth_info = apply_smoothing_pipeline(
                    spectrum_features,
                    time_smooth_method=time_smooth_method,
                    time_smooth_params=time_smooth_params,
                    spectrum_smooth_method=spectrum_smooth_method,
                    spectrum_smooth_params=spectrum_smooth_params,
                    detect_outliers=smooth_config.get('detect_outliers', False),
                    outlier_method=smooth_config.get('outlier_method', 'zscore'),
                    outlier_threshold=smooth_config.get('outlier_threshold', 3.0),
                    handle_outliers=smooth_config.get('handle_outliers', False),
                    outlier_handle_method=smooth_config.get('outlier_handle_method', 'interpolate')
                )
                
                if time_smooth_method != 'none':
                    print(f"    ✓ 时间轴平滑: {time_smooth_method}")
                if spectrum_smooth_method != 'none':
                    print(f"    ✓ 频谱轴平滑: {spectrum_smooth_method}")
            
            # 3.6 下采样（可选）
            if downsample:
                print(f"  - 下采样数据（间隔: {downsample_interval}秒）...")
                original_count = len(timestamps)
                
                start_time = timestamps[0]
                end_time = timestamps[-1]
                new_timestamps = np.arange(start_time, end_time, downsample_interval)
                
                downsample_indices = []
                for target_time in new_timestamps:
                    idx = np.argmin(np.abs(timestamps - target_time))
                    downsample_indices.append(idx)
                
                downsample_indices = sorted(list(set(downsample_indices)))
                
                spectrum_features = spectrum_features[downsample_indices]
                labels = labels[downsample_indices]
                timestamps = timestamps[downsample_indices]
                
                print(f"    ✓ 下采样完成: {original_count} -> {len(timestamps)} 样本")
            
            # 4. 辅助传感器数据融合（可选） - 使用 UTC timestamp 查询
            aux_feature_names = []
            if data_fusion:
                print("  - 加载辅助传感器数据...")
                if len(db_paths) > 1:
                    aux_data = self.db_loader.load_aux_sensors_from_multiple_dbs(db_paths, start_ts_utc, end_ts_utc)
                else:
                    aux_data = self.db_loader.load_aux_sensors_from_db(Path(db_paths[0]), start_ts_utc, end_ts_utc)
                
                aux_counts = {
                    'BME680': len(aux_data.bme680),
                    'PPG': len(aux_data.ppg),
                    'T117': len(aux_data.t117),
                    'ICM': len(aux_data.icm)
                }
                available_sensors = [k for k, v in aux_counts.items() if v > 0]
                
                if available_sensors:
                    print(f"    ✓ 找到传感器: {', '.join(available_sensors)}")
                    for sensor, count in aux_counts.items():
                        if count > 0:
                            print(f"      - {sensor}: {count} 条记录")
                    
                    # 插值到频谱时间戳
                    aux_feature_mode = 'raw'
                    if fusion_config:
                        aux_feature_mode = fusion_config.get('aux_feature_mode', 'raw')
                    aux_features, aux_feature_names = self.interpolate_aux_to_spectrum(
                        aux_data, timestamps,
                        normalize=True,
                        interpolation_method=interpolation_method,
                        icm_mode=icm_mode,
                        aux_feature_mode=aux_feature_mode
                    )
                    
                    # 覆盖aux sensor数值（消融实验）
                    if fusion_config and fusion_config.get('aux_override', False):
                        aux_override_value = fusion_config.get('aux_override_value', 0.0)
                        print(f"    ⚠ 覆盖aux sensor数值为: {aux_override_value}")
                        aux_features = np.full_like(aux_features, aux_override_value)
                    
                    if aux_features.shape[1] > 0:
                        if fusion_stage == 'early':
                            features = apply_data_fusion(spectrum_features, aux_features, fusion_method)
                            print(f"    ✓ Early Fusion - 融合后特征维度: {features.shape[1]}")
                        else:
                            features = np.concatenate([spectrum_features, aux_features], axis=1)
                            print(f"    ✓ Late Fusion - 数据层不融合，保持分离")
                    else:
                        features = spectrum_features
                        aux_features = None
                        print(f"    ⚠ 未能提取辅助传感器特征")
                else:
                    features = spectrum_features
                    aux_features = None
                    print(f"    ⚠ 未找到辅助传感器数据")
            else:
                features = spectrum_features
                aux_features = None
            
            print(f"    ✓ 最终特征shape: {features.shape}, 标签shape: {labels.shape}")
            print(f"    ✓ 血糖范围: {labels.min():.2f} - {labels.max():.2f} mmol/L")
            _db_debug_mem(f"exp={experiment_config.experiment_name} before return")
            
            # 5. 元数据
            # 生成完整日期时间，避免仅MM.DD导致跨年排序歧义
            dt = self.db_loader.utc_timestamp_to_beijing(timestamps[0])
            start_time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            start_time_display = f"{dt.month:02d}.{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}"
            
            metadata = {
                'experiment_dir': experiment_config.experiment_name,
                'experiment_name': experiment_config.experiment_name,
                'user_name': user_name,
                'start_time': start_time_str,
                'start_time_display': start_time_display,
                'n_samples': len(features),
                'n_features': features.shape[1],
                'n_spectrum_features': spectrum_features.shape[1],
                'n_aux_features': len(aux_feature_names) if data_fusion and aux_features is not None else 0,
                'aux_feature_names': aux_feature_names if data_fusion and aux_features is not None else [],
                'data_fusion': data_fusion,
                'fusion_stage': fusion_stage,
                'glucose_mean': float(np.mean(labels)),
                'glucose_std': float(np.std(labels)),
                'glucose_min': float(np.min(labels)),
                'glucose_max': float(np.max(labels)),
                'data_source': 'db',  # 标记数据来源
            }

            if track_db_file_source and '__db_file' in spectrum_df.columns:
                sample_db_files = spectrum_df['__db_file'].astype(str).to_numpy()
                unique_db_files = []
                db_file_to_id = {}
                sample_db_file_ids = np.empty(len(sample_db_files), dtype=np.int32)
                for i, db_file in enumerate(sample_db_files):
                    if db_file not in db_file_to_id:
                        db_file_to_id[db_file] = len(unique_db_files)
                        unique_db_files.append(db_file)
                    sample_db_file_ids[i] = db_file_to_id[db_file]

                metadata['db_file_paths'] = unique_db_files
                metadata['db_file_names'] = [Path(p).name for p in unique_db_files]
                metadata['sample_db_file_ids'] = sample_db_file_ids

            # 主动释放中间DataFrame引用，降低分实验循环中的瞬时内存峰值
            del spectrum_df, glucose_df
            
            return features, labels, timestamps, metadata
            
        except FileNotFoundError:
            # 文件缺失属于配置级致命错误：向上抛出，避免继续加载导致额外内存占用
            raise
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_user_all_experiments(
        self,
        user_name: str = "Tao",
        interpolation_method: str = 'pchip',
        experiment_filter: Optional[List[str]] = None,
        data_fusion: bool = False,
        fusion_method: str = 'concat',
        fusion_stage: str = 'early',
        downsample: bool = False,
        downsample_interval: float = 1.0,
        smooth_config: Optional[dict] = None,
        icm_mode: str = 'raw',
        fusion_config: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict], np.ndarray]:
        """
        加载所有 DB 格式实验数据
        
        返回格式与 GlucoseDatasetLoader.load_user_all_experiments 完全兼容
        """
        print(f"\n{'='*70}")
        db_root = self._resolve_db_root(user_name)
        print(f"加载 DB 格式数据 (用户: {user_name})")
        print(f"DB 根目录: {db_root}")
        if data_fusion:
            stage_str = '数据层(Early)' if fusion_stage == 'early' else '特征层(Late)'
            print(f"✓ 启用辅助传感器数据融合（阶段: {stage_str}, 方法: {fusion_method}）")
        if downsample:
            print(f"✓ 启用数据下采样（间隔: {downsample_interval}秒）")
        if smooth_config and smooth_config.get('smooth_data', False):
            print("✓ 启用数据平滑处理")
        print(f"{'='*70}")
        
        # 加载实验配置
        experiments = self.load_experiments_config(db_root=db_root)
        
        # 调试信息：显示加载的实验
        if experiments:
            print(f"  从配置文件加载了 {len(experiments)} 个实验:")
            for exp in experiments:
                # 从 db_path 提取文件夹名作为备用标识
                db_path_obj = Path(exp.db_path)
                if db_path_obj.is_dir():
                    folder_name = db_path_obj.name
                else:
                    folder_name = db_path_obj.parent.name
                print(f"    - {exp.experiment_name} (文件夹: {folder_name})")
        
        if experiment_filter:
            print(f"  应用实验过滤器: {experiment_filter}")
            original_count = len(experiments)
            
            # 支持两种匹配方式：
            # 1. 匹配 experiment_name（如：260121_1703_Tao）
            # 2. 匹配 db_path 中的文件夹名（如：DEV_002_qiangtao_260121）
            filtered_experiments = []
            for exp in experiments:
                # 从 db_path 提取文件夹名
                db_path_obj = Path(exp.db_path)
                if db_path_obj.is_dir():
                    folder_name = db_path_obj.name
                else:
                    folder_name = db_path_obj.parent.name
                
                # 检查是否匹配（实验名或文件夹名）
                if exp.experiment_name in experiment_filter or folder_name in experiment_filter:
                    filtered_experiments.append(exp)
            
            experiments = filtered_experiments
            print(f"  过滤后: {len(experiments)} / {original_count} 个实验")
        
        track_db_file_source = bool(fusion_config.get('track_db_file_source', False)) if fusion_config else False

        print(f"找到 {len(experiments)} 个实验")
        
        all_features_list = []
        all_labels_list = []
        all_timestamps_list = []
        metadata_list = []
        cumulative_samples = 0
        cumulative_feature_bytes = 0
        
        for exp_config in experiments:
            result = self.load_db_experiment(
                exp_config,
                interpolation_method=interpolation_method,
                data_fusion=data_fusion,
                fusion_method=fusion_method,
                fusion_stage=fusion_stage,
                downsample=downsample,
                downsample_interval=downsample_interval,
                smooth_config=smooth_config,
                icm_mode=icm_mode,
                fusion_config=fusion_config,
                user_name=user_name,
                track_db_file_source=track_db_file_source
            )
            
            if result is not None:
                features, labels, timestamps, metadata = result
                # 统一转float32，尽量避免后续额外拷贝
                all_features_list.append(features.astype(np.float32, copy=False))
                all_labels_list.append(labels)
                all_timestamps_list.append(timestamps)
                metadata_list.append(metadata)

                cumulative_samples += len(features)
                cumulative_feature_bytes += int(all_features_list[-1].nbytes)
                print(
                    f"  [DEBUG] 已加载实验 {metadata.get('experiment_name', 'unknown')} | "
                    f"累计样本={cumulative_samples:,} | "
                    f"累计特征内存={cumulative_feature_bytes / (1024 ** 2):.2f} MB"
                )
        
        if not all_features_list:
            raise ValueError(f"未能加载任何 DB 格式实验数据")
        
        # 合并所有实验（features已在append时转换为float32）
        all_features = np.vstack(all_features_list)
        all_labels = np.concatenate(all_labels_list).astype(np.float32)
        all_timestamps = np.concatenate(all_timestamps_list)
        
        # 创建实验分组索引
        experiment_indices = []
        for exp_idx, features in enumerate(all_features_list):
            experiment_indices.extend([exp_idx] * len(features))
        experiment_indices = np.array(experiment_indices, dtype=np.int32)
        
        # 释放原始列表内存
        del all_features_list, all_labels_list, all_timestamps_list
        
        print(f"\n{'='*70}")
        print(f"✓ 成功加载 {len(metadata_list)} 个实验")
        print(f"  总样本数: {len(all_features)}")
        print(f"  特征维度: {all_features.shape[1]}")
        print(f"  数据大小: {all_features.nbytes / 1e9:.2f} GB")
        if fusion_stage == 'late' and data_fusion:
            n_spectrum = metadata_list[0].get('n_spectrum_features', 1001)
            n_aux = metadata_list[0].get('n_aux_features', 0)
            print(f"  Late Fusion: spectrum={n_spectrum}, aux={n_aux}")
        print(f"  血糖范围: {all_labels.min():.2f} - {all_labels.max():.2f} mmol/L")
        print(f"  血糖均值: {all_labels.mean():.2f} ± {all_labels.std():.2f} mmol/L")
        print(f"{'='*70}\n")
        
        return all_features, all_labels, all_timestamps, metadata_list, experiment_indices
