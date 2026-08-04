# daily_run.py
from datetime import datetime
from baostock_fetcher import BaostockCodeFetcher
from local_data_cache import LocalDataCache

def main():
    # 1. 实例化 Fetcher 与 Cache
    fetcher = BaostockCodeFetcher()
    cache = LocalDataCache(code_fetcher=fetcher)
    
    # 2. 登录 BaoStock（重要：运行全市场更新前必须保持登录）
    print("正在建立 BaoStock 线上连接...")
    lg = fetcher.login()
    if lg.error_code != '0':
        print(f"BaoStock 登录失败: {lg.error_msg}")
        return

    try:
        # 3. 默认获取当天日期的数据
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        # 也可以手动指定某天进行历史补账，例如：
        # today_str = "2026-08-03"
        
        cache.update_daily_market_data(today_str)
        
    except Exception as e:
        print(f"任务运行期间发生未捕获异常: {e}")
        
    finally:
        # 4. 登出系统以释放服务器连接
        print("正在断开 BaoStock 线上连接...")
        fetcher.logout()

if __name__ == "__main__":
    main()