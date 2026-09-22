"""外部服务三件套：天气、搜索、画图。

天气用 Open-Meteo（免费、不要 key），搜索用 ddgs（DuckDuckGo，也不要 key），
画图走百炼的异步任务接口（提交任务+轮询取图，需要 key；与主 LLM 的 key 相互独立）。
"""

import uuid
from datetime import datetime

import dashscope
import httpx
from ddgs import DDGS

from config.settings import get_settings, timeout_seconds
from shared.types import ExternalServiceError
from tools.misc import has_key

# config.json 的 timeouts 段缺失/写坏时的兜底上限（秒），与 config/settings.py 的
# defaults 保持一致
_IMAGE_TIMEOUT_DEFAULT = 300.0


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
        # R27c：网络边界——httpx 的连接/超时/HTTP 状态错误统一转窄载体；
        # "查不到城市"是业务空结果（ValueError，调用方按 no_result 兜底），
        # 不与网络故障混为一谈；响应体缺字段属于本地装配，原样上抛不计数
        try:
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
        except ValueError:
            raise
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "weather", "request_failed", retryable=True, detail=str(exc)
            ) from exc
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
        # R27c：网络边界——ddgs 的网络/限流异常统一转窄载体；
        # 结果行清洗在边界之外（本地装配，出错原样上抛）
        try:
            with DDGS() as ddgs:
                rows = list(ddgs.text(query, max_results=top_k))
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError(
                "search", "request_failed", retryable=True, detail=str(exc)
            ) from exc
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
    wait 轮询 -> 拿到图片 URL -> 下载落盘。图存到 `DATA_DIR/images` 下
    （跟着 get_settings().data_dir 走，不是写死的 storage/images），
    返回值里带路径，写回流程会登记到图片库。
    """

    def generate(self, prompt: str) -> dict:
        cfg = get_settings()
        key = cfg.image_gen_api_key or cfg.llm_api_key
        if not has_key(key):
            raise ExternalServiceError("image", "not_configured", retryable=False,
                                       detail="IMAGE_GEN_API_KEY 没配")
        # 提交任务：注意 dashscope 的 size 用星号分隔（1024*1024），不是 x
        # R27c：提交/轮询都是付费的远端任务边界，失败统一转窄载体并带上 SDK 错误码
        try:
            rsp = dashscope.ImageSynthesis.async_call(
                api_key=key,
                model=cfg.image_gen_model,
                prompt=prompt,
                n=1,
                size="1024*1024",
            )
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError(
                "image", "submit_failed", retryable=True, detail=str(exc)
            ) from exc
        if rsp.status_code != 200:
            raise ExternalServiceError(
                "image", "submit_failed", retryable=True,
                detail=f"画图任务提交失败：{rsp.status_code} {rsp.message}",
            )
        # wait 内部自带轮询，出图或失败才返回。
        #
        # J3：必须显式传 wait_timeout——SDK 的默认值是 -1，也就是"无限轮询"。
        # 这是全项目唯一一处能把 asyncio.to_thread 默认线程池的线程永久占住的调用，
        # 攒够几个就是整个服务卡死（连 SQLite 写回都排队）。超时后 SDK 不抛异常，
        # 而是返回 status_code=408 / code=WaitTaskTimeout，正好落进下面的失败分支。
        #
        # 顺带修掉这一行上两个"必炸"的缺陷（改这行才发现，都属于同一条调用）：
        # 1. 形参名是 `task`，不是 `task_id`。原来写 `wait(task_id=...)`，task_id 掉进
        #    **kwargs、位置参数 task 没人给 → TypeError: missing a required argument: 'task'。
        #    最坏的地方是它炸在 async_call **之后**：任务已经提交、钱已经花了，图却永远拿不到。
        # 2. `rsp.status` 这个属性不存在（ImageSynthesisResponse 上只有 output.task_status），
        #    读它直接 AttributeError —— 连"画图失败"的报错信息都构造不出来。
        wait_timeout = int(timeout_seconds("image_seconds", _IMAGE_TIMEOUT_DEFAULT))
        rsp = dashscope.ImageSynthesis.wait(
            task=rsp.output.task_id,
            api_key=key,
            wait_timeout=wait_timeout,
        )
        if rsp.status_code != 200:
            # 408/WaitTaskTimeout 走这条：把 SDK 给的 code 一起带上，别只说"没成功"
            raise ExternalServiceError(
                "image", "task_failed", retryable=True,
                detail=f"画图任务没成功：{rsp.status_code} {rsp.code} {rsp.message}",
            )
        task_status = getattr(rsp.output, "task_status", "") if rsp.output is not None else ""
        if task_status != "SUCCEEDED":
            raise ExternalServiceError(
                "image", "task_failed", retryable=True,
                detail=f"画图任务没成功：{task_status} {rsp.message}",
            )
        url = rsp.output.results[0].url
        # 结果给的是图片链接，下载回来存本地
        image_dir = cfg.data_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
        try:
            with httpx.Client(timeout=120) as client:
                png = client.get(url).content
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "image", "download_failed", retryable=True, detail=str(exc)
            ) from exc
        path.write_bytes(png)
        return {"path": str(path), "prompt": prompt}
