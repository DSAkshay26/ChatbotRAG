import os
from typing import Optional

from dotenv import dotenv_values, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings  # , ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pydantic import BaseModel

# from constants import text
# from langchain_core.documents import Document

# Integrating Langsmith with the system for troubleshooting and debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Chatbot Doc Mapping"

# loading env variables
config = dotenv_values(".env")
load_dotenv()

# Initiating Fast api
app = FastAPI(title="RAG FUSION MAPPING SEMANTICS")

# handling CORS
origins = [
    "http://localhost:5174",
    "http://localhost:5174/",
    "http://localhost:8000",
    "https://a32e-163-47-141-203.ngrok-free.app",
    "https://66793bd3a58cfa8f07013835--harmonious-marigold-987178.netlify.app",
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Defining Request model
class RequestModel(BaseModel):
    query: str
    context: Optional[list[str]] = None
    session_id: str


# creating vector db with pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# declaring embedding
embeddings = OpenAIEmbeddings()
# declaring index name for vector db which will be used for retrieval
index_name = "jellyfish-vectordb"


# retrieving context
def context_retriever(query):
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    docs = vectorstore.similarity_search(query, k=1)
    if len(docs) != 0:
        content = docs[0].metadata["original_content"]
    else:
        content = "Frame a professional answer which shows the positive image of the company and should be relevant to the query"
    return content


user_message = """
<|Storyline|>
You are Jelly, a Chatbot of Jellyfish Technologies Website.\
Throughout your interactions, you aim to ask relevant question to understand their requirements and highlight how Jellyfish Technologies can meet those needs.\
Your primary goal is to guide users to contact the company through the contact form or contact details to discuss their needs and how Jellyfish Technologies can help them.\
Each response should be a step towards encouraging the user to get in touch with the company.\
Your mission is to ensure every visitor is impressed with Jellyfish Technologies and eager to take advantage of your services.\
Start by greeting the user warmly, then proceed to ask questions that help identify their needs. Highlight the benefits of Jellyfish Technologies' services and guide the user to the contact form to make a deal.\
Here are some key points to include in your responses:
- Highlight Jellyfish Technologies' expertise and certifications.
- Mention the company's global presence and successful projects.
- Emphasize the importance of getting a tailored solution from the sales team.
- Always direct the user to the contact form for further assistance.

<|Instructions|>
Always give professional and formal answers.\
Strictly provide relevant urls with every repsonse.\
Don't try to connect the user to us, on your own. Always make the user contact us on their own through contact form or provided contact details of jft.\
Don't ask too much specifications about the query simply ask three to four follow up then direct them to contact us page without anything else in the response.\
If any question is unproffesional or irrevelant to the benefits of the company like song, bomb threat, illegal activities, reply "Your question does not align with professional standards. If you have any inquiries related to Jellyfish Technologies, please feel free to ask. I am happy to help."\
Make sure you always provide a positive image of Jellyfish, do not provide unnecessary details.\
Only use the context provided below, to provide an answer in about 70 words kind of summary without missing any important information present in the context. Don't write according to the context. Stick to the role.\
If you don't know the answer, just say that you are still learning and improving yourself. \
Strictly don't provide response in markdown\

<|Context|>
CTO of JFT: Amit Kumar Pandey, CEO of the Company: Gaurav Chauhan, COO of Jellyfish: Neeraj Kumar. \ 
Global Presence/Branches/Addresses/Locations : 59, West Broadway #200 Salt Lake City Utah-84101, United States: US office Location, URL: https://www.google.com/maps/search/jellyfish+technologies+salt+lake/@40.761821,-116.5041917,7z?entry=ttu, D-5, Third Floor, Logix Infotech, Sector-59, Noida-201301, India: Location, URL: https://www.google.com/maps/place/Jellyfish+Technologies+%7C+Software+Development+Company/@28.6080993,77.3696763,17z/data=!3m1!4b1!4m6!3m5!1s0x390ceff2a400bb77:0xf4123d7195e9427a!8m2!3d28.6080993!4d77.3722512!16s%2Fg%2F11bxg37vwc?entry=ttu\
Contact form: https://www.jellyfishtechnologies.com/contact-us/\
Awards/Recognition/Certifications/Greeting bagged by Jellyfish/ reasons to choose jellyfish: 5/5 verified rating on Goodfirms, Top Developers on Clutch, Salesforce Certified Developer, Best Company by Goodfirms, Great Place to Work Certified\
Contact Details: Email: enquiry@jellyfishtechnologies.com, hr@jellyfishtechnologies.com, Phone: +1-760-514-0178, linekdin: https://www.linkedin.com/company/teamjft/mycompany/ 
{context}\
Strictly Answer in less than 70 words\
Strictly provide all the relevant urls with every response.\
Strictly frame your responses in such a way that it diverts the user to the contact form page along with the relevant url associated with the information.\
"""

human_message_template = """
<|Question|>
{question}
"""

# creating prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", user_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# declaring llm
llm = ChatGroq(
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192",
)

# llm = ChatOpenAI(temperature=0.0, api_key=os.getenv('OPENAI_API_KEY'), model='gpt-4o-mini')


# funtion for history retrieval using redis
def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=os.getenv("REDIS_URL"))


@app.get("/")
async def greet():
    return "Hi, welcome!"


@app.post("/query")
async def answer_query(req: RequestModel):
    try:

        rag_chain = prompt | llm | StrOutputParser()
        with_message_history = RunnableWithMessageHistory(
            rag_chain,
            get_message_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        final_response = await with_message_history.ainvoke(
            {"context": context_retriever(req.query), "input": req.query},
            config={"configurable": {"session_id": req.session_id}},
        )
        return final_response

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
