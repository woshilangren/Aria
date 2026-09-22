"""时间解析统一入口（R09a）——纯标准库，不 import 任何上层模块。

三个事实（为什么必须收拢，而不是各处裸调 fromisoformat）：
1. 库里的历史时间戳是 naive 本地时间（datetime.now() 时代的产物），naive 与
   aware 混算直接 TypeError——这是时间路径上最常见的炸点；
2. Python 3.10 的 fromisoformat 不认 "Z" 后缀（3.11 才支持），而 Z 在上游
   数据与日志里很常见；
3. 截断时间串（比如 [:19]）会丢掉 "+08:00" 偏移：同一时刻的 Z / +08:00 /
   -05:00 三种写法必须算出同一个答案，截断做不到。

约定（R09a）：
- parse_to_aware：能解析就返回 aware 时刻；naive 输入按**历史本地时间**语义
  补本地时区（绝不假装是 UTC）；坏值/空串/None 返回 None，由调用方降级；
- safe_delta_seconds：两个时刻的受保护减法（任何一方解析失败返回 None），
  负值（未来时间戳）由调用方按用途钳制或降级。
"""

from datetime import datetime


def parse_to_aware(value):
    """把 ISO 时间串（或 datetime）解析成 aware 时刻；解析不了返回 None。

    - "Z"/"z" 后缀替换成 "+00:00"（3.10 兼容）；
    - naive 输入按本地时区补全——历史数据是本地钟面时间，语义就是"当时的
      本地几点"，补本地时区后与 aware 数据可比；
    - 全库不改写旧时间：转换只发生在读取路径上。
    """
    if isinstance(value, datetime):
        dt = value
    else:
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.astimezone()  # naive：按本地时区解释（历史本地时间语义）
    return dt


def safe_delta_seconds(later, earlier):
    """(later - earlier) 的秒数；任何一方解析失败返回 None（调用方降级）。

    两侧都先过 parse_to_aware：aware-aware 减法跨时区自动折算，永远不会
    TypeError；"减法必须在受保护范围"由这里统一兜住。
    """
    a = parse_to_aware(later)
    b = parse_to_aware(earlier)
    if a is None or b is None:
        return None
    return (a - b).total_seconds()
