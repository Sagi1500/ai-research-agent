from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool
from datetime import datetime


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
    # add getters for the fields if needed
    def get_topic(self):
        return self.topic
    
    def get_summary(self):
        return self.summary
    
    def get_sources(self):
        return self.sources
    
    def get_tools_used(self):
        return self.tools_used


def save_to_txt(data: ResearchResponse, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n"
    
    # load the json data into a dictionary
    # print("Data to save: ", data)
    # print("Data type: ", type(data)) 
    
    # extract the topic, summary, sources and used tools
    topic = data.get_topic()
    summary = data.get_summary()
    sources = data.get_sources()
    tools_used = data.get_tools_used()
    
    # format the text
    formatted_text += f"Topic: {topic}\n\n"
    formatted_text += f"Summary: {summary}\n\n"
    formatted_text += f"Sources: {', '.join(sources)}\n\n"
    formatted_text += f"Tools Used: {', '.join(tools_used)}\n\n"
    

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"    

llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
    
    save_to_txt(
        data=structured_response,
        filename="research_output.txt"
    )
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)