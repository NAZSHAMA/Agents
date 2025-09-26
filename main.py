from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
madle =OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    client=external_client,
)

config = RunConfig(
    model=madle,
    model_provider = external_client,
    tracing_disabled=True,
)

agent=Agent(
    name="Frontend expert",
    instructions="You are a helpful frontend expert.",
)
@cl.on_chat_start
async def handle_start():
    cl.user_sassion.set("history", [])
    await cl.Message(
        content="Hello! I am a frontend expert. How can I assist you today?"
    ).send()

result= Runner.run_sync(
    agent,
    input="What is the best way to implement a responsive design in web development?",
    config=config,
)

print(result.final_output)

@cl.on_message
async def handle_message(mesage :cl.Message):

    history = cl.user_session.get("history")

    history.append({"role": "user", "content": mesage.content})
    result = await Runner.run(
        agent,
        input= history, 
        run_config=config,
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()
