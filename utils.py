from langchain.chains import ConversationChain
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
import os
from langchain.memory import ConversationBufferMemory


def get_chat_response(prompt, memory, openai_api_key):
    # model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    model = ChatOpenAI(
        model="deepseek-chat",  # 指定 DeepSeek 模型
        openai_api_key=openai_api_key,
        openai_api_base="https://api.deepseek.com/v1"  # DeepSeek API 端点
    )
    chain = ConversationChain(llm=model, memory=memory)

    # 直接调用 LLM 生成回复
    response = model.invoke(prompt)
    return response.content  # DeepSeek 返回的是 Message 对象


# memory = ConversationBufferMemory(return_messages=True)

# print(get_chat_response("牛顿提出过哪些知名的定律？", memory, openai_api_key))
# print(get_chat_response("我上一个问题是什么？", memory, openai_api_key))
