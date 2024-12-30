"""爬虫配置文件"""

# 知网搜索API URL
BASE_URL = "https://kns.cnki.net/kns8s/brief/grid"

# 请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "Referer": "https://kns.cnki.net/"
}

# 查询JSON模板
QUERY_JSON_TEMPLATE = {
    "Platform": "",
    "Resource": "CROSSDB",
    "Classid": "WD0FTY92",
    "Products": "",
    "QNode": {
        "QGroup": [
            {
                "Key": "Subject",
                "Title": "",
                "Logic": 0,
                "Items": [],
                "ChildItems": [
                    {
                        "Key": ".extend-indent-labels>.colorful-lable",
                        "Title": "",
                        "Logic": 0,
                        "Items": [
                            {
                                "Key": ".extend-indent-labels>.colorful-lable",
                                "Title": "增强出版",
                                "Logic": 0,
                                "Field": "NPM",
                                "Operator": "DEFAULT",
                                "Value": "ZQ",
                                "Value2": ""
                            }
                        ],
                        "ChildItems": []
                    }
                ]
            },
            {
                "Key": "ControlGroup",
                "Title": "",
                "Logic": 0,
                "Items": [],
                "ChildItems": [
                    {
                        "Key": "span[value=PT]",
                        "Title": "",
                        "Logic": 0,
                        "Items": [
                            {
                                "Key": "span[value=PT]",
                                "Title": "发表时间",
                                "Logic": 0,
                                "Field": "PT",
                                "Operator": 7,
                                "Value": "{start_date}",
                                "Value2": "{end_date}"
                            }
                        ],
                        "ChildItems": []
                    }
                ]
            },
            {
                "Key": "NaviParam",
                "Title": "",
                "Logic": 0,
                "Items": [
                    {
                        "Key": "naviScope",
                        "Title": "文献分类：信息科技",
                        "Logic": 0,
                        "Field": "CCL",
                        "Operator": "DEFAULT",
                        "Value": "I?",
                        "Value2": "",
                        "ExtendType": 2
                    }
                ],
                "ChildItems": []
            }
        ]
    },
    "ExScope": "1",
    "SearchType": 1,
    "Rlang": "CHINESE",
    "KuaKuCode": "YSTT4HG0,LSTPFY1C,JUP3MUPD,MPMFIG1A,EMRPGLPA,WQ0UVIAA,BLZOG7CK,PWFIRAGL,NN3FJMUV,NLBO1Z6R",
    "SearchFrom": 4
}

# 基础搜索参数
BASE_SEARCH_PARAMS = {
    "boolSearch": "false",
    "pageSize": "20",
    "sortField": "PT",
    "sortType": "desc",
    "dstyle": "listmode",
    "boolSortSearch": "false",
    "sentenceSearch": "false",
    "productStr": "YSTT4HG0,LSTPFY1C,RMJLXHZ3,JQIRZIYA,JUP3MUPD,1UR4K4HZ,BPBAFJ5S,R79MZMCB,MPMFIG1A,EMRPGLPA,J708GVCE,ML4DRIDX,WQ0UVIAA,NB3BWEHK,XVLO76FD,HR1YT1Z9,BLZOG7CK,PWFIRAGL,NN3FJMUV,NLBO1Z6R",
    "aside": ""
}

# 请求配置
REQUEST_CONFIG = {
    "timeout": 30,  # 请求超时时间（秒）
    "max_retries": 3,  # 最大重试次数
    "retry_delay": 2,  # 重试间隔（秒）
    "page_delay": 1  # 翻页间隔（秒）
}

# 数据保存配置
DATA_CONFIG = {
    "output_dir": "data/raw",  # 输出目录
    "filename_template": "thesis_titles_{year}.csv",  # 文件名模板
    "encoding": "utf-8"  # 文件编码
}

# 数据字段配置
DATA_FIELDS = {
    "required_columns": ["id", "title", "publish_date", "author", "major"],  # 必需的列
    "default_major": "信息科技"  # 默认专业分类
}

# 正则表达式配置
REGEX_PATTERNS = {
    "title_clean": r"[^\u4e00-\u9fa5a-zA-Z0-9\s]"  # 标题清洗正则表达式
}
