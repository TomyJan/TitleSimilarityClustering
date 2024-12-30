"""知网论文爬虫实现"""
import os
from typing import Optional, Dict
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime
from .config import (
    BASE_URL,
    HEADERS,
    QUERY_JSON_TEMPLATE,
    BASE_SEARCH_PARAMS,
    REQUEST_CONFIG,
    DATA_CONFIG,
    DATA_FIELDS,
    REGEX_PATTERNS
)

logger = logging.getLogger(__name__)

class CNKICrawler:
    """知网论文爬虫类"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.base_url = BASE_URL
        
    def _get_search_params(self, start_date: str, end_date: str, page: int = 1) -> Dict:
        """构造搜索参数"""
        # 复制查询JSON模板并替换日期
        query_json = json.loads(json.dumps(QUERY_JSON_TEMPLATE).replace(
            "{start_date}", start_date).replace("{end_date}", end_date))
        
        # 复制基础搜索参数并添加页码
        params = BASE_SEARCH_PARAMS.copy()
        params.update({
            "QueryJson": json.dumps(query_json, ensure_ascii=False),
            "pageNum": str(page),
            "searchFrom": f"资源范围：总库; 增强出版,中英文扩展; 时间范围：发表时间：{start_date}到{end_date};更新时间：不限; 文献分类：信息科技"
        })
        
        return params

    def _make_request(self, url: str, params: Dict) -> Optional[requests.Response]:
        """发送请求并处理错误"""
        for i in range(REQUEST_CONFIG["max_retries"]):
            try:
                response = self.session.post(url, data=params, timeout=REQUEST_CONFIG["timeout"])
                response.raise_for_status()
                
                # 检查是否被重定向到登录页面
                if "login" in response.url.lower():
                    logger.error("需要登录知网账号")
                    return None
                    
                return response
                
            except requests.RequestException as e:
                logger.error(f"请求失败 (尝试 {i+1}/{REQUEST_CONFIG['max_retries']}): {str(e)}")
                if i < REQUEST_CONFIG["max_retries"] - 1:
                    time.sleep(REQUEST_CONFIG["retry_delay"] * (2 ** i))  # 指数退避
                continue
                
        return None

    def _clean_title(self, title: str) -> str:
        """清理标题文本"""
        # 移除特殊字符
        title = re.sub(REGEX_PATTERNS["title_clean"], '', title)
        # 移除多余空格
        return ' '.join(title.split())

    def _extract_paper_info(self, tr_element: BeautifulSoup, index: int) -> Optional[Dict]:
        """从表格行中提取论文信息"""
        try:
            # 提取标题
            title_element = tr_element.select_one("td.name a.fz14")
            if not title_element:
                return None
                
            title = self._clean_title(title_element.text.strip())
            if not title:  # 标题为空则跳过
                return None
            
            # 提取作者
            author_elements = tr_element.select("td.author a")
            if not author_elements:  # 作者为空则跳过
                return None
            
            author = author_elements[0].text.strip()
            if not author:  # 第一作者为空则跳过
                return None
                
            # 提取发布日期
            date_element = tr_element.select_one("td.date")
            if not date_element:
                return None
                
            publish_date = date_element.text.strip()
            if not publish_date:  # 日期为空则跳过
                return None
            
            # 验证日期格式
            try:
                datetime.strptime(publish_date, "%Y-%m-%d")
            except ValueError:
                try:
                    # 尝试解析带时间的格式
                    datetime.strptime(publish_date, "%Y-%m-%d %H:%M")
                    publish_date = publish_date.split()[0]  # 只保留日期部分
                except ValueError:
                    logger.warning(f"无效的日期格式: {publish_date}")
                    return None
            
            return {
                "id": index,
                "title": title,
                "publish_date": publish_date,
                "author": author,
                "major": DATA_FIELDS["default_major"]
            }
        except Exception as e:
            logger.error(f"提取论文信息时出错: {str(e)}")
            return None

    def _get_total_count(self, html_content: str) -> int:
        """从HTML中提取总条数"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            count_element = soup.select_one("span.pagerTitleCell em")
            if count_element:
                return int(count_element.text.strip())
            else:
                logger.error("未找到总条数元素")
                # 保存HTML内容以供调试
                with open("debug_response.html", "w", encoding="utf-8") as f:
                    f.write(html_content)
        except Exception as e:
            logger.error(f"提取总条数时出错: {str(e)}")
        return 0

    def crawl_papers(self, start_date: str, end_date: str) -> bool:
        """爬取指定时间范围内的论文数据"""
        logger.info(f"开始爬取 {start_date} 到 {end_date} 的数据")
        try:
            # 获取年份并构造输出路径
            year = int(start_date.split('-')[0])
            output_path = os.path.join(
                DATA_CONFIG["output_dir"],
                DATA_CONFIG["filename_template"].format(year=year)
            )
            
            # 获取第一页并解析总页数
            params = self._get_search_params(start_date, end_date)
            logger.info("正在发送搜索请求...")
            
            response = self._make_request(self.base_url, params)
            if not response:
                return False
                
            logger.info("成功获取响应，正在解析数据...")
            total_count = self._get_total_count(response.text)
            if total_count == 0:
                logger.warning("未找到符合条件的论文")
                return False
                
            total_pages = (total_count + 19) // 20  # 向上取整
            logger.info(f"共找到 {total_count} 条结果，{total_pages} 页")
            
            all_papers = []
            current_index = 1
            
            # 解析第一页数据
            soup = BeautifulSoup(response.text, 'html.parser')
            for tr in soup.select("table.result-table-list tbody tr"):
                paper_info = self._extract_paper_info(tr, current_index)
                if paper_info:
                    all_papers.append(paper_info)
                    current_index += 1
            
            # 获取剩余页面
            for page in range(2, total_pages + 1):
                logger.info(f"正在爬取第 {page}/{total_pages} 页...")
                params = self._get_search_params(start_date, end_date, page)
                response = self._make_request(self.base_url, params)
                
                if not response:
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                for tr in soup.select("table.result-table-list tbody tr"):
                    paper_info = self._extract_paper_info(tr, current_index)
                    if paper_info:
                        all_papers.append(paper_info)
                        current_index += 1
                        
                logger.info(f"成功爬取第 {page}/{total_pages} 页数据")
                time.sleep(REQUEST_CONFIG["page_delay"])
            
            # 保存数据
            if all_papers:
                df = pd.DataFrame(all_papers)
                # 确保所有必需的列都存在
                if not all(col in df.columns for col in DATA_FIELDS["required_columns"]):
                    logger.error("数据格式不完整，缺少必需的列")
                    return False
                    
                # 确保没有空值
                if df.isnull().any().any():
                    logger.error("数据中存在空值")
                    return False
                    
                # 创建输出目录并保存数据
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df.to_csv(output_path, index=False, encoding=DATA_CONFIG["encoding"])
                logger.info(f"成功保存 {len(all_papers)} 条记录到 {output_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"爬取过程中出错: {str(e)}")
            return False
