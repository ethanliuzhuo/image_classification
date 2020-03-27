# coding=utf-8

"""根据搜索词下载百度图片"""
import re
import sys
import urllib
import requests
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

base_file_path =  os.path.join(os.getcwd(), 'test')

def getPage(keyword, page, n):
    page = page * n
    keyword = urllib.parse.quote(keyword, safe='/')
    url_begin = "http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word="
    url = url_begin + keyword + "&pn=" + str(page) + "&gsm=" + str(hex(page)) + "&ct=&ic=0&lm=-1&width=0&height=0"

    return url

def get_onepage_urls(onepageurl):
    try:
        html = requests.get(onepageurl).text
    except Exception as e:
        print(e)
        pic_urls = []
        return pic_urls
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    return pic_urls

def down_pic(keyword, pic_urls):
    """给出图片链接列表, 下载所有图片"""
    full_file_paths = os.path.join(base_file_path, keyword)
    if not os.path.exists(full_file_paths):
        os.makedirs(full_file_paths)
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            string = keyword + '_' + str(i + 1) + '.jpg'
            full_file_path = os.path.join(full_file_paths, string)
            with open(full_file_path, 'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue



def start_thread(count):
    """
    启动线程池执行下载任务
    :return:
    """
    with ThreadPoolExecutor(max_workers=count) as t:
        key_word = pd.read_csv('./data/label.csv')
        for keyword in key_word['物品']:
        #keyword = 'Rolls-Royce'  # 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样
            page_begin = 0
            page_number = 0
            image_number = 0
            all_pic_urls = []
            while 1:
                if page_begin > image_number:
                    break
                print("第%d次请求数据", [page_begin])
                url = getPage(keyword, page_begin, page_number)
                onepage_urls = get_onepage_urls(url)
                page_begin += 1
                all_pic_urls.extend(onepage_urls)
            # 一页60图
            # down_pic(keyword,list(set(all_pic_urls))[:5])
            # with ThreadPoolExecutor(max_workers=count) as t:
                t.submit(down_pic, keyword ,list(set(all_pic_urls))[:])


if __name__ == '__main__':
    # key_word = pd.read_csv('D:\\data\\classification2\\label.csv')
    # for keyword in key_word['物品'][0][0]:
    # #keyword = 'Rolls-Royce'  # 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样
    #     page_begin = 0
    #     page_number = 0
    #     image_number = 0
    #     all_pic_urls = []
    #     while 1:
    #         if page_begin > image_number:
    #             break
    #         print("第%d次请求数据", [page_begin])
    #         url = getPage(keyword, page_begin, page_number)
    #         onepage_urls = get_onepage_urls(url)
    #         page_begin += 1
    #         all_pic_urls.extend(onepage_urls)
    #     # 一页60图
    #     down_pic(keyword,list(set(all_pic_urls))[:5])
    print("程序执行开始")
    print("======================================")
    print("温馨提示： 输入内容必须为大于的0的数字才行！")
    print("======================================")
    count = int(input("请输入您需要启动的线程数： "))
    start_thread(count)
    print("程序执行结束")