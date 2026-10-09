# 第一步： 导入需要的包
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


load_dotenv()

@tool
def get_current_utc_time() -> str:
    """Get the current time in UTC."""
    # 第一种查看工具调用的方式
    # print(">>> 工具被调用了！获取当前UTC时间")
    return datetime.now(timezone.utc).isoformat()


def build_agent():
    # 使用百炼的 qwen-plus 模型，并配置兼容模式参数
    model = ChatOpenAI(
        model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7
    )
    return create_agent(model, tools=[get_current_utc_time])


def main() -> None:
    agent = build_agent()
    question = input("You: ").strip()
    if not question:
        return

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # 第二种查看工具调用的方式
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content}")
    print(f"Assistant: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()