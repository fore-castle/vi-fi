import requests
import re

# 代理配置
proxies = {
    "http": "http://agent.baidu.com:8891",
    "https": "http://agent.baidu.com:8891",
}

def get_confirm_token(response):
    # 尝试从 cookie 中获取 token
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    # 如果 cookie 中没有，尝试从页面源码中找确认链接（针对某些特定情况）
    return None

def download_test():
    file_id = '1K-QRtbxsJnnSi2GL1N3IdjkOFtjnDqYP'
    base_url = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    # 第一次请求：尝试获取确认页面
    print("正在获取确认令牌...")
    res = session.get(base_url, params={'id': file_id}, proxies=proxies, verify=False, stream=True)
    
    token = get_confirm_token(res)
    
    params = {'id': file_id}
    if token:
        print(f"找到 Token: {token[:10]}...")
        params['confirm'] = token
    
    # 第二次请求：正式下载
    print("正在开始正式流式下载...")
    with session.get(base_url, params=params, proxies=proxies, verify=False, stream=True) as r:
        r.raise_for_status()
        
        # 检查返回的内容类型，如果是 text/html，说明还是没跳过警告
        content_type = r.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print("警告：下载的依然是网页内容，请检查文件权限或 Token 逻辑。")
        
        with open("test_real_data.part", 'wb') as f:
            count = 0
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    count += 1
                    print(f"\r实测下载真实数据: {count} MB", end="")
                    if count >= 5: # 测试 5MB 即可
                        break
    print("\n测试完成。")

if __name__ == "__main__":
    download_test()