"""外部服务三件套：天气、搜索、画图。

天气用 Open-Meteo（免费、不要 key），搜索用 ddgs（DuckDuckGo，也不要 key），
画图走百炼的异步任务接口（提交任务+轮询取图，需要 key；与主 LLM 的 key 相互独立）。
"""

import uuid
from datetime import datetime

import dashscope
import httpx
from ddgs import DDGS

from config.settings import get_settings
from tools.misc import has_key


# Open-Meteo 返回的是天气代码（weathercode），这里挑常见的翻成中文
_WEATHER_CODES = {
    0: "晴",
    1: "基本晴",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "有雾",
    51: "毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "阵雨",
    95: "雷阵雨",
    96: "雷阵雨伴冰雹",
}


class WeatherTool:
    """查天气：城市名 -> 当天温度、天气情况、湿度。"""

    def query(self, city: str) -> dict:
        # 先把城市名换成经纬度
        with httpx.Client(timeout=15) as client:
            geo = client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "zh"},
            )
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                raise ValueError(f"查不到这个城市：{city}")
            spot = results[0]
            # 再拿经纬度查当天天气
            weather = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": spot["latitude"],
                    "longitude": spot["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,weather_code",
                },
            )
            weather.raise_for_status()
            cur = weather.json().get("current", {})
        code = cur.get("weather_code", -1)
        return {
            "city": spot.get("name", city),
            "temp": cur.get("temperature_2m"),
            "condition": _WEATHER_CODES.get(code, "未知"),
            "humidity": cur.get("relative_humidity_2m"),
        }


class ExternalSearchTool:
    """联网搜索：关键词 -> 几条搜索结果（标题、摘要、链接）。"""

    def search(self, query: str, top_k: int = 3) -> list:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=top_k))
        results = []
        for row in rows:
            results.append(
                {
                    "title": row.get("title", ""),
                    "snippet": row.get("body", ""),
                    "url": row.get("href", ""),
                }
            )
        return results


class ImageGenTool:
    """文字生图：走百炼的异步任务接口（先提交任务，再轮询到出图）。

    qwen-image-3.0-pro 不支持同步一次出图，流程是 async_call 提交 ->
    wait 轮询 -> 拿到图片 URL -> 下载落盘。图存到 storage/images 下，
    返回值里带路径，写回流程会登记到图片库。
    """

    def generate(self, prompt: str) -> dict:
        cfg = get_settings()
        key = cfg.image_gen_api_key or cfg.llm_api_key
        if not has_key(key):
            raise RuntimeError("画图服务没配置（IMAGE_GEN_API_KEY）")
        # 提交任务：注意 dashscope 的 size 用星号分隔（1024*1024），不是 x
        rsp = dashscope.ImageSynthesis.async_call(
            api_key=key,
            model=cfg.image_gen_model,
            prompt=prompt,
            n=1,
            size="1024*1024",
        )
        if rsp.status_code != 200:
            raise RuntimeError(f"画图任务提交失败：{rsp.status_code} {rsp.message}")
        # wait 内部自带轮询，出图或失败才返回
        rsp = dashscope.ImageSynthesis.wait(task_id=rsp.output.task_id, api_key=key)
        if rsp.status_code != 200 or rsp.status != "SUCCEEDED":
            raise RuntimeError(f"画图任务没成功：{rsp.status} {rsp.message}")
        url = rsp.output.results[0].url
        # 结果给的是图片链接，下载回来存本地
        image_dir = cfg.data_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
        with httpx.Client(timeout=120) as client:
            png = client.get(url).content
        path.write_bytes(png)
        return {"path": str(path), "prompt": prompt}
