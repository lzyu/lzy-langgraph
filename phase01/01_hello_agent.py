import os
import sys
import io
from dotenv import load_dotenv

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANGSMITH_TRACING"] = "false"

from langchain.agents import create_agent

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

load_dotenv()


def get_current_time(city: str) -> str:
    """获取指定城市的当前时间"""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    timezones = {
        "北京": "Asia/Shanghai",
        "上海": "Asia/Shanghai",
        "东京": "Asia/Tokyo",
        "纽约": "America/New_York",
        "伦敦": "Europe/London",
        "旧金山": "America/Los_Angeles",
    }
    tz_name = timezones.get(city, "UTC")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    print(city, tz_name, now)
    return f"{city} 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})"


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def main():
    agent = create_agent(
        model="openai:deepseek-chat",
        tools=[get_current_time, calculate],
        system_prompt="你是一个友好的助手，可以查询时间和做简单计算。用中文回答。",
    )

    print("=== 第一个 LangChain v1.x Agent ===")
    print("输入 'quit' 退出\n")


    messages = []

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "quit":
            print("再见！")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})


        for event in agent.stream({"messages": messages}):
            for node_name, state_update in event.items():
                new_messages = state_update.get("messages", [])
                for msg in new_messages:
                    if msg.type == "ai" and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"AI: [调用工具: {tc['name']}({tc['args']})]")
                    elif msg.type == "tool":
                        print(f"  [工具结果: {msg.content}]")
                    elif msg.type == "ai" and msg.content:
                        print(f"AI: {msg.content}")
                        messages.append({"role": "assistant", "content": msg.content})

        print()


if __name__ == "__main__":
    main()
