import akshare as ak
import pandas as pd
import time
from datetime import datetime, timedelta
import os


class FinancialReportFetcher:
    def __init__(self, cache_dir="./financial_reports_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cached_report(self, report_date):
        """检查是否有缓存的数据"""
        cache_file = os.path.join(self.cache_dir, f"{report_date}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, dtype={'股票代码': str})
                print(f"从缓存加载 {report_date} 的数据")
                return df
            except Exception as e:
                print(f"读取缓存文件失败: {e}")
        return None
    
    def cache_report(self, report_date, df):
        """缓存数据到本地"""
        if df.empty:
            return
        
        cache_file = os.path.join(self.cache_dir, f"{report_date}.csv")
        try:
            # 强制覆盖旧的缓存文件，保存最新完整数据
            df.to_csv(cache_file, index=False, encoding='utf-8-sig')
            print(f"已缓存/更新 {report_date} 的数据")
        except Exception as e:
            print(f"缓存数据失败: {e}")
    
    def get_quarter_end_date(self, year, quarter):
        """根据年份和季度获取正确的季度末日期"""
        if quarter == 1:
            return f"{year}0331"  # 3月31日
        elif quarter == 2:
            return f"{year}0630"  # 6月30日
        elif quarter == 3:
            return f"{year}0930"  # 9月30日
        elif quarter == 4:
            return f"{year}1231"  # 12月31日
        return None
    
    # 【改动 1】：增加 force_update 参数
    def fetch_report(self, report_date, max_retries=3, force_update=False):
        """获取单个报告期的数据"""
        # 如果不强制更新，才去检查缓存
        if not force_update:
            cached_data = self.get_cached_report(report_date)
            if cached_data is not None:
                return cached_data
        else:
            print(f"[{report_date} 属于近一年数据，将忽略缓存，强制拉取最新数据...]")
        
        # 如果没有缓存或强制更新，则从接口获取
        for attempt in range(max_retries):
            try:
                print(f"正在获取 {report_date} 的业绩报表数据...")
                df = ak.stock_yjbb_em(date=report_date)
                
                if not df.empty:
                    # 添加报告日期列
                    df['报告日期'] = report_date
                    
                    # 标准化股票代码格式
                    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
                    
                    # 缓存数据（会自动覆盖旧文件）
                    self.cache_report(report_date, df)
                    
                    print(f"成功获取 {report_date} 的数据，共 {len(df)} 条记录")
                    return df
                else:
                    print(f"{report_date} 的数据为空")
                    return pd.DataFrame()
                    
            except Exception as e:
                print(f"获取 {report_date} 数据失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    # 等待一段时间后重试
                    time.sleep(2)
                else:
                    print(f"无法获取 {report_date} 的数据")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_all_reports(self, start_date="20100331", save_path=None):
        """获取所有报告期的数据"""
        all_reports = pd.DataFrame()
        
        start_year = int(start_date[:4])
        start_month = int(start_date[4:6])
        
        if start_month <= 3: start_quarter = 1
        elif start_month <= 6: start_quarter = 2
        elif start_month <= 9: start_quarter = 3
        else: start_quarter = 4
        
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        
        if current_month <= 3: current_quarter = 1
        elif current_month <= 6: current_quarter = 2
        elif current_month <= 9: current_quarter = 3
        else: current_quarter = 4

        # 【改动 2】：计算出一年前的日期字符串阈值（YYYYMMDD格式）
        # 比如今天是 2026-03-08，一年前就是 2025-03-08
        one_year_ago = current_date - timedelta(days=365)
        one_year_ago_str = one_year_ago.strftime("%Y%m%d")
        
        # 遍历所有可能的季度
        for year in range(start_year, current_year + 1):
            if year == start_year:
                quarters = range(start_quarter, 5)
            elif year == current_year:
                quarters = range(1, current_quarter + 1)
            else:
                quarters = range(1, 5)
            
            for quarter in quarters:
                report_date = self.get_quarter_end_date(year, quarter)
                
                # 判断当前财报日期是否在近一年内
                # 因为格式都是 YYYYMMDD，可以直接进行字符串比较
                is_recent_year = report_date >= one_year_ago_str
                
                # 获取数据，传入 force_update 状态
                df = self.fetch_report(report_date, force_update=is_recent_year)
                
                if not df.empty:
                    # 添加到总数据中
                    all_reports = pd.concat([all_reports, df], ignore_index=True)
        
        # 数据去重
        all_reports = all_reports.drop_duplicates()
        
        # 保存数据
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            all_reports.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"所有数据已合并并保存到: {save_path}")
        
        return all_reports


fetcher = FinancialReportFetcher(cache_dir="./financial_reports_cache")
all_data = fetcher.get_all_reports(
    start_date="20100331", 
    save_path="./financial_reports_all.csv"
)
print(f"共获取 {len(all_data)} 条业绩报表记录")


# 货币供应量
# 接口: macro_china_supply_of_money
# 目标地址: http://finance.sina.com.cn/mac/#fininfo-1-0-31-1
# 描述: 新浪财经-中国宏观经济数据-货币供应量
# 限量: 单次返回所有历史数据
macro_china_supply_of_money_df = ak.macro_china_supply_of_money()
print(macro_china_supply_of_money_df)
macro_china_supply_of_money_df.to_excel("macro_china_supply_of_money_df.xlsx",sheet_name='Sheet1', index=False)
time.sleep(2)

# 创新高和新低的股票数量——修改为自己算
# 接口: stock_a_high_low_statistics
# 目标地址: https://www.legulegu.com/stockdata/high-low-statistics
# 描述: 不同市场的创新高和新低的股票数量
# 限量: 单次获取指定 market 的近两年的历史数据
# stock_a_high_low_statistics_df = ak.stock_a_high_low_statistics(symbol="all")
# print(stock_a_high_low_statistics_df)
# breadth_df = pd.read_excel(
#     'stock_a_high_low_statistics_df.xlsx',
#     dtype={
#         'close': 'float32',
#         'high20': 'float32',
#         'low20': 'float32',
#         'high60': 'float32',
#         'low60': 'float32',
#         'high120': 'float32',
#         'low120': 'float32'
#     },
#     parse_dates=['date']
# )
# stock_a_high_low_statistics_df = stock_a_high_low_statistics_df.append(breadth_df, ignore_index=True)
# stock_a_high_low_statistics_df = stock_a_high_low_statistics_df.drop_duplicates()
# stock_a_high_low_statistics_df.to_excel("stock_a_high_low_statistics_df.xlsx",sheet_name='Sheet1', index=False)
# time.sleep(10)

# 大盘拥挤度——接口开会员，自己算
# 接口: stock_a_congestion_lg
# 目标地址: https://legulegu.com/stockdata/ashares-congestion
# 描述: 乐咕乐股-大盘拥挤度
# 限量: 单次获取近 4 年的历史数据
# stock_a_congestion_lg_df = ak.stock_a_congestion_lg()
# print(stock_a_congestion_lg_df)
# stock_a_congestion_lg_df.to_excel("stock_a_congestion_lg_df.xlsx",sheet_name='Sheet1', index=False)
# time.sleep(10)

# 历史行情数据-通用——Remote end closed connection without response，改为自己抓取，gemini知道东财的数据接口和参数，神奇
# 接口: index_zh_a_hist
# 目标地址: http://quote.eastmoney.com/center/hszs.html
# 描述: 东方财富网-中国股票指数-行情数据
# 限量: 单次返回具体指数指定 period 从 start_date 到 end_date 的之间的近期数据
# 000985 中证全指

# os.environ['HTTP_PROXY'] = "socks5h://127.0.0.1:7890"
# os.environ['HTTPS_PROXY'] = "socks5h://127.0.0.1:7890"
# # 获取当前日期和10天前的日期
# current_date = datetime.now().strftime("%Y%m%d")
# ten_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

# # 文件路径
# file_path = "zzqz_df.xlsx"

# # 检查文件是否存在
# if os.path.exists(file_path):
#     # 读取现有文件数据
#     existing_df = pd.read_excel(file_path)
#     print("已读取现有文件数据，数据行数:", len(existing_df))
# else:
#     # 如果文件不存在，创建空的DataFrame
#     existing_df = pd.DataFrame()
#     print("文件不存在，将创建新文件")

# # 获取近10天的数据
# print(f"获取从 {ten_days_ago} 到 {current_date} 的数据...")
# try:
#     new_data_df = ak.index_zh_a_hist(symbol="000985", period="daily", 
#                                     start_date=ten_days_ago, end_date=current_date)
#     print(f"获取到 {len(new_data_df)} 行新数据")
    
#     # 合并数据（如果现有数据不为空）
#     if not existing_df.empty:
#         combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
#     else:
#         combined_df = new_data_df
    
#     # 去重（假设'日期'列是唯一标识）
#     # 注意：根据实际数据列名调整，可能需要使用'date'或其他列名
#     if '日期' in combined_df.columns:
#         # 按日期降序排序，保留最新的数据
#         combined_df = combined_df.sort_values('日期', ascending=False)
#         # 去重，保留第一条（最新的）
#         combined_df = combined_df.drop_duplicates(subset=['日期'], keep='first')
#         # 按日期升序排序
#         combined_df = combined_df.sort_values('日期', ascending=True)
#     elif 'date' in combined_df.columns:
#         # 如果列名是'date'
#         combined_df = combined_df.sort_values('date', ascending=False)
#         combined_df = combined_df.drop_duplicates(subset=['date'], keep='first')
#         combined_df = combined_df.sort_values('date', ascending=True)
#     else:
#         # 如果没有找到日期列，尝试找到第一列作为日期列
#         date_col = combined_df.columns[0]
#         combined_df = combined_df.sort_values(date_col, ascending=False)
#         combined_df = combined_df.drop_duplicates(subset=[date_col], keep='first')
#         combined_df = combined_df.sort_values(date_col, ascending=True)
#         print(f"使用列 '{date_col}' 进行去重")
    
#     print(f"去重后总数据行数: {len(combined_df)}")
    
#     # 覆盖保存到Excel文件
#     combined_df.to_excel(file_path, sheet_name='Sheet1', index=False)
#     print(f"数据已保存到 {file_path}")
    
#     # 显示最新几行数据
#     print("\n最新数据预览:")
#     print(combined_df.tail())
    
# except Exception as e:
#     print(f"获取数据时出错: {e}")


# import requests——普通接口抓取方式也被东财封了，改用唤起浏览器
# import pandas as pd
# import os
# import time
# import logging
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# from curl_cffi import requests


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def update_zzqz_excel(file_path="zzqz_df.xlsx"):
#     # 1. 抓取配置
#     symbol = "1.000985"
#     end_date = time.strftime("%Y%m%d")
#     start_date = (pd.Timestamp.now() - pd.Timedelta(days=20)).strftime("%Y%m%d")
    
#     # 2. 基础 URL 和 Headers（参考浏览器）
#     base_headers = {
#         "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
#         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#         "Accept-Language": "zh-CN,zh;q=0.9",
#         "Accept-Encoding": "gzip, deflate, br",
#         "Connection": "keep-alive",
#         "Upgrade-Insecure-Requests": "1",
#     }
    
#     # 3. 创建 Session，配置重试
#     session = requests.Session(impersonate="chrome120")

#     session.proxies = {"http": None, "https": None}  # 禁用代理
#     session.headers.update(base_headers)
    
#     retry_strategy = Retry(
#         total=3,
#         connect=3,
#         read=2,
#         backoff_factor=1,
#         status_forcelist=[500, 502, 503, 504],
#         allowed_methods=["GET"],
#         raise_on_status=False
#     )
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     session.mount("http://", adapter)
#     session.mount("https://", adapter)
    
#     try:
#         # 4. 第一步：访问主页，获取 Cookie 和会话
#         logger.info("正在访问东方财富主页以建立会话...")
#         home_url = "https://quote.eastmoney.com/center/hszs.html"  # 中证全指页面
#         home_resp = session.get(home_url, timeout=10)
#         home_resp.raise_for_status()
#         logger.info(f"主页访问成功，状态码: {home_resp.status_code}")
        
#         # 可选：打印获取到的 Cookie（调试用）
#         # logger.info(f"当前 Cookie: {session.cookies.get_dict()}")
        
#         # 5. 第二步：请求数据 API
#         api_url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
#         api_params = {
#             "secid": symbol,
#             "fields1": "f1,f2,f3,f4,f5,f6",
#             "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
#             "klt": "101",
#             "fqt": "0",
#             "beg": start_date,
#             "end": end_date,
#             "lmt": "100",
#             # 不添加 _ 时间戳，避免干扰
#         }
#         # API 请求可能需要特定的 Referer，我们在 headers 中动态设置
#         api_headers = {
#             "Referer": "https://quote.eastmoney.com/center/hszs.html",
#             "X-Requested-With": "XMLHttpRequest",  # 模拟 AJAX 请求
#         }
#         logger.info(f"正在抓取 {start_date} 至 {end_date} 的最新数据...")
#         response = session.get(api_url, params=api_params, headers=api_headers, timeout=15)
#         response.raise_for_status()
#         data = response.json()
        
#         if not data or 'data' not in data or not data['data']:
#             logger.warning("API 返回数据为空")
#             return
        
#         # 6. 解析数据
#         klines = data['data']['klines']
#         new_rows = [item.split(',') for item in klines]
#         df_new = pd.DataFrame(new_rows, columns=[
#             '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'
#         ])
#         df_new['日期'] = pd.to_datetime(df_new['日期']).dt.strftime('%Y-%m-%d')
        
#         # 7. 读取本地文件并合并（与之前相同）
#         if os.path.exists(file_path):
#             df_old = pd.read_excel(file_path)
#             df_old['日期'] = pd.to_datetime(df_old['日期']).dt.strftime('%Y-%m-%d')
#             logger.info(f"成功读取本地文件，当前记录数: {len(df_old)}")
#         else:
#             df_old = pd.DataFrame()
#             logger.info("本地文件不存在，将创建新文件。")
        
#         df_combined = pd.concat([df_old, df_new], ignore_index=True)
#         df_combined.drop_duplicates(subset=['日期'], keep='last', inplace=True)
#         df_combined = df_combined.sort_values(by='日期', ascending=True)
#         df_combined.to_excel(file_path, index=False)
        
#         logger.info(f"更新完成！去重后总记录数: {len(df_combined)}")
#         logger.info(f"最新一行记录: \n{df_combined.iloc[-1:]}")
    
#     except requests.exceptions.RequestException as e:
#         logger.error(f"请求失败: {e}")
#     except Exception as e:
#         logger.error(f"操作失败: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     update_zzqz_excel("zzqz_df.xlsx")

import os
import pandas as pd
from playwright.sync_api import sync_playwright
import json
import time

def fetch_with_auth_persistence(file_path="zzqz_df.xlsx", auth_file="auth.json"):
    api_url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        "secid=1.000985&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        "&klt=101&fqt=1&end=20500101&lmt=60"
    )
    portal_url = "https://quote.eastmoney.com/zs000985.html"

    with sync_playwright() as p:
        # 启动浏览器
        browser = p.chromium.launch(headless=False)
        
        # 加载状态或创建新上下文
        if os.path.exists(auth_file):
            print("🔑 正在加载验证状态 (auth.json)...")
            context = browser.new_context(storage_state=auth_file)
        else:
            print("🆕 准备进行人工验证...")
            context = browser.new_context()

        page = context.new_page()

        # --- 手动隐身逻辑：抹除自动化特征 ---
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => False
            });
        """)

        try:
            # 1. 访问详情页预热
            print(f"🌍 正在打开详情页，请在弹出的窗口中手动完成【拼图验证】并【关闭广告】...")
            page.goto(portal_url)
            
            # 给充足的时间让你操作
            print("⏳ 窗口将等待 25 秒，请抓紧时间完成验证...")
            page.wait_for_timeout(25000) 

            # 2. 访问数据接口
            print("📡 尝试抓取数据...")
            page.goto(api_url)
            
            # 确保 body 加载
            page.wait_for_load_state("networkidle")
            content = page.inner_text("body")

            if "klines" in content:
                print("✅ 抓取成功！保存验证状态...")
                context.storage_state(path=auth_file)
                
                # 解析数据
                json_data = json.loads(content)
                klines = json_data['data']['klines']
                rows = [line.split(',') for line in klines]
                df_new = pd.DataFrame(rows, columns=['日期','开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌幅','涨跌额','换手率'])
                
                # 类型转换
                for col in df_new.columns[1:]:
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')

                df_new['日期'] = pd.to_datetime(df_new['日期']).dt.strftime('%Y-%m-%d')
                
                # 合并旧文件
                if os.path.exists(file_path):
                    df_old = pd.read_excel(file_path)
                    df_old['日期'] = pd.to_datetime(df_old['日期']).dt.strftime('%Y-%m-%d')
                    df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates('日期', keep='last')
                else:
                    df_combined = df_new
                
                df_combined.sort_values('日期').to_excel(file_path, index=False)
                print("-" * 30)
                print(f"✨ 补全成功！最新数据：")
                print(df_combined.tail(3)[['日期', '收盘', '成交额', '换手率']])
            else:
                print("❌ 失败：未识别到数据。可能是拼图验证未通过或超时。")
                print("当前内容：", content[:100])

        except Exception as e:
            print(f"❌ 运行报错: {e}")
        finally:
            time.sleep(2)
            browser.close()

fetch_with_auth_persistence()

import sqlite3
import os
import baostock as bs
import pandas as pd
from datetime import datetime

def update_cache_meta_code(cache_dir='./stock_data_cache'):
    """
    定期运行脚本：读取 BaoStock 全量股票，将 code 写入/更新至原有的 cache_meta 表中。
    采用 SQLite UPSERT 语法，仅更新 code 映射，保留已有的历史缓存进度数据（min_date, max_date 等）。
    """
    db_path = os.path.join(cache_dir, 'stock_data.db')
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[{datetime.now()}] 开始登录 BaoStock...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录 BaoStock 失败: {lg.error_msg}")
        return

    print("开始获取全量证券基本资料...")
    rs = bs.query_stock_basic()
    if rs.error_code != '0':
        print(f"获取证券资料失败: {rs.error_msg}")
        bs.logout()
        return

    # 提取结果集
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    
    bs.logout()
    print("BaoStock 注销成功。")

    if not data_list:
        print("未获取到任何证券数据。")
        return

    df = pd.DataFrame(data_list, columns=rs.fields)
    
    # 过滤出正在上市的股票（type='1' 代表股票，status='1' 代表上市中）
    df_stocks = df[(df['type'] == '1') & (df['status'] == '1')].copy()

    if df_stocks.empty:
        print("过滤后没有有效的股票数据。")
        return

    # 从 code (如 sh.600000) 拆分出 symbol (如 600000)
    df_stocks['symbol'] = df_stocks['code'].apply(lambda x: x.split('.')[1] if '.' in x else x)
    
    # 准备写入的数据列表 [(symbol, code), ...]
    write_data = df_stocks[['symbol', 'code']].values.tolist()

    print(f"开始更新本地 cache_meta 表: {db_path}...")
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 1. 确保 cache_meta 表存在（若不存在则建表，包含 code 列）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_meta (
                    symbol TEXT PRIMARY KEY,
                    code TEXT,
                    last_updated TEXT,
                    min_date TEXT,
                    max_date TEXT
                )
            ''')
            
            # 2. 兼容性升级：如果老数据库中存在 cache_meta 表但缺少 code 列，则动态追加 code 列
            cursor.execute("PRAGMA table_info(cache_meta)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'code' not in columns:
                print("检测到旧版 cache_meta 缺少 'code' 列，正在为您升级表结构...")
                cursor.execute("ALTER TABLE cache_meta ADD COLUMN code TEXT")
                conn.commit()

            # 3. 使用 UPSERT 语法进行更新：
            # 若 symbol 不存在则插入，若已存在则仅更新 code，不影响已有的 last_updated, min_date, max_date 等缓存进度
            upsert_query = '''
                INSERT INTO cache_meta (symbol, code)
                VALUES (?, ?)
                ON CONFLICT(symbol) DO UPDATE SET code = excluded.code
            '''
            cursor.executemany(upsert_query, write_data)
            conn.commit()
            
            print(f"[{datetime.now()}] 成功更新了 {len(write_data)} 只上市股票的映射关系至 cache_meta。")
            
    except Exception as e:
        print(f"更新数据库失败: {e}")
update_cache_meta_code()

import co_compute
market_context = co_compute.sync_market_context_file(cache_dir='./stock_data_cache')
print(market_context)