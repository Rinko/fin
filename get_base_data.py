"""
每日数据获取脚本 - 可交互入口

包含 5 个数据任务：
  1. financial : 财务报告（东方财富业绩报表，akshare）
  2. macro     : 货币供应量（新浪财经，akshare）
  3. zzqz      : 中证全指 000985（东方财富，playwright 拼图验证）
  4. market    : 大盘环境快照（广度/拥挤度/5维因子，同步 parquet）
  5. daily     : 每日行情更新（BaoStock -> SQLite 缓存）

用法：
  python get_base_data.py                # 交互式菜单选择
  python get_base_data.py --task all     # 依次执行全部任务
  python get_base_data.py --task financial --task macro   # 执行指定任务
  python get_base_data.py --task daily --date 2026-08-04  # 指定日期补账
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd

import akshare as ak

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 任务 1：财务报告
# ============================================================
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
            df.to_csv(cache_file, index=False, encoding='utf-8-sig')
            print(f"已缓存/更新 {report_date} 的数据")
        except Exception as e:
            print(f"缓存数据失败: {e}")

    def get_quarter_end_date(self, year, quarter):
        """根据年份和季度获取正确的季度末日期"""
        if quarter == 1:
            return f"{year}0331"
        elif quarter == 2:
            return f"{year}0630"
        elif quarter == 3:
            return f"{year}0930"
        elif quarter == 4:
            return f"{year}1231"
        return None

    def fetch_report(self, report_date, max_retries=3, force_update=False):
        """获取单个报告期的数据，近一年数据强制拉取最新"""
        if not force_update:
            cached_data = self.get_cached_report(report_date)
            if cached_data is not None:
                return cached_data
        else:
            print(f"[{report_date} 属于近一年数据，将忽略缓存，强制拉取最新数据...]")

        for attempt in range(max_retries):
            try:
                print(f"正在获取 {report_date} 的业绩报表数据...")
                df = ak.stock_yjbb_em(date=report_date)

                if not df.empty:
                    df['报告日期'] = report_date
                    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
                    self.cache_report(report_date, df)
                    print(f"成功获取 {report_date} 的数据，共 {len(df)} 条记录")
                    return df
                else:
                    print(f"{report_date} 的数据为空")
                    return pd.DataFrame()

            except Exception as e:
                print(f"获取 {report_date} 数据失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"无法获取 {report_date} 的数据")
                    return pd.DataFrame()

        return pd.DataFrame()

    def get_all_reports(self, start_date="20100331", save_path=None):
        """获取所有报告期的数据，合并去重后保存"""
        all_reports = pd.DataFrame()

        start_year = int(start_date[:4])
        start_month = int(start_date[4:6])
        start_quarter = 1 if start_month <= 3 else (2 if start_month <= 6 else (3 if start_month <= 9 else 4))

        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        current_quarter = 1 if current_month <= 3 else (2 if current_month <= 6 else (3 if current_month <= 9 else 4))

        one_year_ago_str = (current_date - timedelta(days=365)).strftime("%Y%m%d")
        today_str = current_date.strftime("%Y%m%d")

        for year in range(start_year, current_year + 1):
            if year == start_year:
                quarters = range(start_quarter, 5)
            elif year == current_year:
                quarters = range(1, current_quarter + 1)
            else:
                quarters = range(1, 5)

            for quarter in quarters:
                report_date = self.get_quarter_end_date(year, quarter)

                # 跳过尚未到季度末的未来报告期（如8月时Q3的0930未到）
                if report_date > today_str:
                    continue

                is_recent_year = report_date >= one_year_ago_str
                df = self.fetch_report(report_date, force_update=is_recent_year)
                if not df.empty:
                    all_reports = pd.concat([all_reports, df], ignore_index=True)

        all_reports = all_reports.drop_duplicates()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            all_reports.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"所有数据已合并并保存到: {save_path}")

        return all_reports


def task_financial():
    """任务1：财务报告全量同步"""
    logger.info("任务1: 同步财务报告...")
    fetcher = FinancialReportFetcher(cache_dir="./financial_reports_cache")
    all_data = fetcher.get_all_reports(
        start_date="20100331",
        save_path="./financial_reports_all.csv"
    )
    logger.info(f"任务1完成: 共 {len(all_data)} 条业绩报表记录")
    return {'rows': len(all_data), 'file': 'financial_reports_all.csv'}


# ============================================================
# 任务 2：货币供应量
# ============================================================
def task_macro():
    """任务2：货币供应量（新浪财经宏观数据，单次返回全部历史）"""
    logger.info("任务2: 同步货币供应量...")
    df = ak.macro_china_supply_of_money()
    df.to_excel("macro_china_supply_of_money_df.xlsx", sheet_name='Sheet1', index=False)
    logger.info(f"任务2完成: 共 {len(df)} 条记录，已保存 macro_china_supply_of_money_df.xlsx")
    return {'rows': len(df), 'file': 'macro_china_supply_of_money_df.xlsx'}


# ============================================================
# 任务 3：中证全指 000985（东财 playwright 抓取）
# ============================================================
def _find_chrome():
    """定位可复用的系统 Chrome/Chromium，避免下载 Playwright 自带浏览器。"""
    candidates = [
        os.environ.get('PLAYWRIGHT_CHROME'),
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/Applications/Chromium.app/Contents/MacOS/Chromium',
        '/usr/bin/google-chrome',
        '/usr/bin/google-chrome-stable',
        '/usr/bin/chromium',
        '/usr/bin/chromium-browser',
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _save_zzqz_klines(content, file_path):
    """从东财 API 响应解析中证全指 K 线并增量写入 xlsx。成功返回结果 dict, 否则 None。"""
    if "klines" not in content:
        return None
    import json
    json_data = json.loads(content)
    klines = json_data['data']['klines']
    rows = [line.split(',') for line in klines]
    df_new = pd.DataFrame(rows, columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'])

    for col in df_new.columns[1:]:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
    df_new['日期'] = pd.to_datetime(df_new['日期']).dt.strftime('%Y-%m-%d')

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
    logger.info(f"任务3完成: {file_path} 共 {len(df_combined)} 行")
    return {'rows': len(df_combined), 'file': file_path}


def task_zzqz(file_path="zzqz_df.xlsx", auth_file="auth.json"):
    """
    任务3：增量更新中证全指 000985 行情到 zzqz_df.xlsx。

    抓取策略：
      1. 有头浏览器 + 已有 auth.json：先打开详情页暖场（触发验证通过），
         再直接抓 API（验证状态有效时免人工拼图）。
      2. 抓取失败时，等待人工完成拼图验证一次，成功后把验证状态持久化到
         auth.json，下次优先复用。
    """
    from playwright.sync_api import sync_playwright

    logger.info("任务3: 更新中证全指 000985...")
    api_url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        "secid=1.000985&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        "&klt=101&fqt=1&end=20500101&lmt=60"
    )
    portal_url = "https://quote.eastmoney.com/zs000985.html"
    chrome = _find_chrome()
    if chrome:
        logger.info(f"复用系统浏览器: {chrome}")

    webdriver_script = """
        Object.defineProperty(navigator, 'webdriver', {
            get: () => False
        });
    """

    launch_kwargs = {'headless': False}
    if chrome:
        launch_kwargs['executable_path'] = chrome

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_kwargs)

        if os.path.exists(auth_file):
            print("🔑 正在加载验证状态 (auth.json)...")
            context = browser.new_context(storage_state=auth_file)
        else:
            print("🆕 准备进行人工验证...")
            context = browser.new_context()

        page = context.new_page()
        page.add_init_script(webdriver_script)

        try:
            # --- 步骤 1: portal 暖场, 触发验证通过 (auth 有效时免人工) ---
            print(f"🌍 正在打开详情页 (若需人工验证请在窗口中完成拼图)...")
            page.goto(portal_url, timeout=30000)
            page.wait_for_timeout(8000)

            # --- 步骤 2: 抓取 API 数据 (失败自动重试一次) ---
            print("📡 尝试抓取数据...")
            success = False
            for attempt in range(1, 3):
                try:
                    page.goto(api_url, timeout=20000)
                    page.wait_for_load_state("networkidle")
                    content = page.inner_text("body")
                    result = _save_zzqz_klines(content, file_path)
                    if result is not None:
                        success = True
                        print(f"✅ 抓取成功！")
                        context.storage_state(path=auth_file)
                        return result
                    print(f"⚠️ 第{attempt}次未识别到数据，可能验证未通过...")
                except Exception as e:
                    print(f"⚠️ 第{attempt}次抓取失败({type(e).__name__})...")
                if attempt < 2:
                    print("🔁 等待 3 秒后自动重试...")
                    page.wait_for_timeout(3000)

            # --- 步骤 3: 等待人工拼图验证后重试 ---
            print("⏳ 请在弹出的窗口中手动完成【拼图验证】并【关闭广告】，等待 25 秒...")
            page.wait_for_timeout(25000)
            print("📡 重新尝试抓取数据...")
            page.goto(api_url, timeout=20000)
            page.wait_for_load_state("networkidle")
            content = page.inner_text("body")

            result = _save_zzqz_klines(content, file_path)
            if result is not None:
                print("✅ 抓取成功！保存验证状态...")
                context.storage_state(path=auth_file)
                return result
            else:
                print("❌ 失败：未识别到数据。可能是拼图验证未通过或超时。")
                print("当前内容：", content[:100])
                return {'rows': 0, 'error': '未识别到数据（验证未通过/超时）'}

        except Exception as e:
            print(f"❌ 运行报错: {e}")
            return {'rows': 0, 'error': str(e)}
        finally:
            time.sleep(2)
            browser.close()


# ============================================================
# 任务 4：大盘环境快照（广度 + 拥挤度 + 5维因子）
# ============================================================
def task_market():
    """任务4：同步大盘环境快照到 stock_data_cache/market_context_cache.parquet"""
    logger.info("任务4: 同步大盘环境快照 (Breadth, Congestion, Mkt Factors)...")
    import co_compute
    market_context = co_compute.sync_market_context_file(cache_dir='./stock_data_cache')
    logger.info(f"任务4完成: stock_data_cache/market_context_cache.parquet 共 {len(market_context)} 行")
    return {'rows': len(market_context), 'file': 'stock_data_cache/market_context_cache.parquet'}


# ============================================================
# 任务 5：每日行情更新（BaoStock -> SQLite）
# ============================================================
def task_daily(date_str=None):
    """任务5：更新每日行情数据到本地 SQLite 缓存（缺位自动逐日补齐至最近交易日）"""
    from stock_fetcher_bao import BaostockCodeFetcher
    from local_data_cache import LocalDataCache

    logger.info("任务5: 更新每日行情数据...")
    fetcher = BaostockCodeFetcher()
    cache = LocalDataCache(code_fetcher=fetcher)

    cache.update_daily_market_data(date_str)
    logger.info("任务5完成: 行情同步结束")

    # 5.1 行业数据同步 (申万一级): 日K增量 + 成分映射刷新 (新股纳入)
    # 加在 daily 内, 无需额外定期任务
    try:
        from industry_data import sync_industry_data
        industry_status = sync_industry_data()
        logger.info(f"行业数据同步完成: 日K {industry_status['daily_rows']}行, 成分 {industry_status['components']}只")
    except Exception as e:
        logger.error(f"行业数据同步失败(不影响行情主流程): {e}")

    return {'rows': None, 'date': date_str or 'smart', 'file': 'stock_data_cache/*.db'}


# ============================================================
# 任务调度
# ============================================================
TASKS = {
    'financial': task_financial,
    'macro': task_macro,
    'zzqz': task_zzqz,
    'market': task_market,
    'daily': task_daily,
}

TASK_DESC = {
    'financial': '财务报告（东财业绩报表，近一年强制刷新）',
    'macro': '货币供应量（新浪财经宏观）',
    'zzqz': '中证全指 000985（东财，需拼图验证）',
    'market': '大盘环境快照（广度/拥挤度/5维因子 -> parquet）',
    'daily': '每日行情更新（BaoStock -> SQLite）',
}

SEQUENTIAL = ['financial', 'macro', 'zzqz', 'daily', 'market']

SUMMARY_LOG_FILE = 'data_sync_history.csv'


def run_task(name, args):
    """执行单个任务，返回 (status, detail)。status in {ok, error}"""
    try:
        if name == 'daily':
            result = TASKS[name](args.date)
        else:
            result = TASKS[name]()
        return 'ok', result or {}
    except Exception as e:
        logger.error(f"任务 {name} 失败: {e}")
        return 'error', {'error': str(e)}


def save_summary(results):
    """把本轮结果追加到 data_sync_history.csv"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rows = []
    for name, (status, detail) in results.items():
        rows.append({
            '时间': timestamp,
            '任务': name,
            '状态': status,
            '记录数': detail.get('rows', '') if isinstance(detail, dict) else '',
            '输出': detail.get('file', '') if isinstance(detail, dict) else '',
            '备注': detail.get('error', '') if isinstance(detail, dict) else '',
        })
    df = pd.DataFrame(rows)
    if os.path.exists(SUMMARY_LOG_FILE):
        old = pd.read_csv(SUMMARY_LOG_FILE)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(SUMMARY_LOG_FILE, index=False, encoding='utf-8-sig')


def print_summary(results):
    """统一汇总输出本轮结果"""
    print("\n" + "=" * 56)
    print("本轮任务结果汇总")
    print("-" * 56)
    for name, (status, detail) in results.items():
        mark = '✅' if status == 'ok' else '❌'
        if status == 'ok':
            extra = f" (记录数={detail.get('rows')})" if 'rows' in detail and detail.get('rows') is not None else ""
            if detail.get('file'):
                extra += f" -> {detail['file']}"
            print(f"  {mark} {name:10s}{extra}")
        else:
            print(f"  {mark} {name:10s} - {detail.get('error')}")
    failed = [k for k, (s, _) in results.items() if s != 'ok']
    if failed:
        print("-" * 56)
        print(f"⚠️  失败任务: {', '.join(failed)}")
    print("=" * 56)


def run_batch(names, args, persist=True):
    """执行任务集合并统一汇总输出；返回失败任务列表"""
    results = {}
    for name in names:
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"\n[{ts}] 执行任务 {name} ...")
        results[name] = run_task(name, args)

    if persist:
        save_summary(results)
    print_summary(results)
    return [k for k, (s, _) in results.items() if s != 'ok']


def interactive_menu():
    """无参数时的交互式菜单"""
    print("=" * 50)
    print("每日数据获取工具")
    print("=" * 50)
    print("可用任务:")
    for key, desc in TASK_DESC.items():
        print(f"  {key:10s} - {desc}")
    print("  all        - 依次执行全部任务")
    print("  quit       - 退出")
    print("-" * 50)

    while True:
        choice = input("请选择任务: ").strip().lower()
        if choice == 'quit':
            print("已退出")
            break
        if choice == 'all':
            failed = run_batch(SEQUENTIAL, argparse.Namespace(date=None))
            sys.exit(1 if failed else 0)
        if choice in TASKS:
            failed = run_batch([choice], argparse.Namespace(date=None))
        else:
            print(f"未知任务: {choice}")


def main():
    parser = argparse.ArgumentParser(description="每日数据获取工具")
    parser.add_argument('--task', action='append', choices=list(TASKS.keys()) + ['all'],
                        help="要执行的任务，可多次指定；不指定则进入交互菜单")
    parser.add_argument('--date', default=None,
                        help="daily 任务指定日期 YYYY-MM-DD（默认今天，用于历史补账）")
    args = parser.parse_args()

    if not args.task:
        interactive_menu()
        return

    if 'all' in args.task:
        sys.exit(1 if run_batch(SEQUENTIAL, args) else 0)

    names = [t for t in args.task if t != 'all']
    sys.exit(1 if run_batch(names, args) else 0)


if __name__ == "__main__":
    main()