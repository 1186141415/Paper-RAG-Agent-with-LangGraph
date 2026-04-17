import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import (
    ZHIPU_API_KEY,
    MCP_SEARCH_URL,
    MCP_SEARCH_RECENCY,
    MCP_SEARCH_CONTENT_SIZE,
    MCP_SEARCH_LOCATION
)

from app.logger_config import setup_logger

logger = setup_logger()


async def _call_zhipu_web_search(
        query: str,
        recency: str = MCP_SEARCH_RECENCY,
        content_size: str = MCP_SEARCH_CONTENT_SIZE,
        location: str = MCP_SEARCH_LOCATION,
):
    api_key = ZHIPU_API_KEY
    if not api_key:
        raise ValueError("ZHIPU_API_KEY not found in .env")

    client = MultiServerMCPClient(
        {
            "zhipu_search": {
                "transport": "http",
                "url": MCP_SEARCH_URL,
                "headers": {
                    "Authorization": f"Bearer {api_key}"
                },
            }
        }
    )

    tools = await client.get_tools()
    search_tool = next((tool for tool in tools if tool.name == "web_search_prime"), None)

    if search_tool is None:
        raise RuntimeError("web_search_prime not found in MCP tools")

    result = await search_tool.ainvoke(
        {
            "search_query": query,
            "search_recency_filter": recency,
            "content_size": content_size,
            "location": location,
        }
    )
    return result


def _parse_mcp_search_result(raw_result) -> list[dict]:
    if not raw_result:
        return []

    raw_text = ""

    if isinstance(raw_result, list) and len(raw_result) > 0:
        first_item = raw_result[0]
        if isinstance(first_item, dict):
            raw_text = first_item.get("text", "")
        else:
            raw_text = getattr(first_item, "text", "")
    else:
        raw_text = str(raw_result)

    data = raw_text

    # 这个 MCP 返回里，text 可能是 “字符串里的 JSON”
    # 所以这里做最多两次 json.loads
    for _ in range(2):
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                break

    if isinstance(data, list):
        return data

    return []


def web_search_tool(query: str) -> str:
    logger.info(f"[web_search_tool] query: {query}")

    try:
        raw_result = asyncio.run(
            _call_zhipu_web_search(
                query=query,
                recency="oneMonth",
                content_size="medium",
                location="us",
            )
        )

        items = _parse_mcp_search_result(raw_result)

        if not items:
            logger.warning("[web_search_tool] parsed result is empty, fallback to raw text")
            return str(raw_result)

        lines = []
        for idx, item in enumerate(items[:5], start=1):
            title = item.get("title", "No title")
            link = item.get("link", "")
            content = item.get("content", "")

            lines.append(
                f"[{idx}] {title}\n"
                f"{content}\n"
                f"{link}"
            )

        final_text = "\n\n".join(lines)
        logger.info("[web_search_tool] search finished successfully")
        return final_text

    except Exception as e:
        logger.exception(f"[web_search_tool] search failed")
        return f'Web search failed: {str(e)}'
