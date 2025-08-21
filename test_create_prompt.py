import os
from dotenv import load_dotenv
from galileo import Message, MessageRole
from galileo.prompts import create_prompt

load_dotenv()

project_name = os.getenv("GALILEO_PROJECT")
unique_prompt_name = "test_prompt_2"

def main():
    print(f"Creating prompt template with name {unique_prompt_name} and project {project_name}")
    prompt_template = create_prompt(
        name=unique_prompt_name,
        template=[
            Message(role=MessageRole.system, content="You are a helpful assistant that can answer questions."),
            Message(role=MessageRole.user, content="What is the capital of France?")
        ]
    )
    print(f"Prompt template created: {prompt_template}")

if __name__ == "__main__":
    main()
