import os
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()


def get_current_time(city: str) -> str:
    """获取指定城市的当前时间"""
    from zoneinfo import ZoneInfo
    from datetime import datetime
    timezones = {"北京": "Asia/Shanghai", "上海": "Asia/Shanghai", "东京": "Asia/Tokyo"}
    tz_name = timezones.get(city, "UTC")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    return f"{city} 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})"


agent = create_agent(
    model="openai:deepseek-chat",
    tools=[get_current_time],
    system_prompt="你是一个助手。用中文回答。",
)

print("=== Debug: invoke ===")
try:
    result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
    print(f"result type: {type(result)}")
    print(f"result keys: {result.keys() if hasattr(result, 'keys') else 'N/A'}")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content = ""
        if hasattr(msg, "content") and msg.content:
            content = str(msg.content)[:200]
        elif hasattr(msg, "tool_calls"):
            content = f"tool_calls={msg.tool_calls}"
        print(f"  [{i}] {msg_type}: {content}")
    print()
except Exception as e:
    print(f"invoke error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug: stream ===")
try:
    for i, event in enumerate(agent.stream({"messages": [{"role": "user", "content": "1+1等于几？"}]})):
        print(f"  event[{i}] type={type(event).__name__}")
        if hasattr(event, "content"):
            print(f"    content: {repr(event.content[:100]) if event.content else 'empty'}")
        if hasattr(event, "tool_calls"):
            print(f"    tool_calls: {event.tool_calls}")
        if isinstance(event, dict):
            print(f"    dict keys: {list(event.keys())[:10]}")
            for k, v in event.items():
                print(f"    {k}: {type(v).__name__} = {repr(v)[:150]}")
except Exception as e:
    print(f"stream error: {e}")
    import traceback
    traceback.print_exc()