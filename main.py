
import json
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

llm=ChatOllama(model="llama3.2:3b",temperature=0)
search=DuckDuckGoSearchRun()

@tool
def web_search(query:str)->str:
    """Search the web."""
    return search.run(query)

def invoke(system,user):
    msgs=[
        ("system",system),
        ("human",user)
    ]
    return llm.invoke(msgs).content

ROUTER_PROMPT="""
You are a router.
Return ONLY JSON:
{"agent":"research"} OR
{"agent":"coder"} OR
{"agent":"reviewer"}
Use coder for coding requests.
Use research for latest info.
Use reviewer for review/debug requests.
"""

def router(question):
    try:
        out=invoke(ROUTER_PROMPT,question)
        return json.loads(out)["agent"]
    except:
        q=question.lower()
        if any(x in q for x in ["code","python","java","c#","api","sql"]):
            return "coder"
        if any(x in q for x in ["review","bug","fix"]):
            return "reviewer"
        return "research"

def research_agent(question):
    web=web_search.invoke(question)
    prompt=f"""
You are a research agent.
Summarize this information for a developer.

Question:
{question}

Search:
{web}
"""
    return invoke("Research only.",prompt)

def coder_agent(question,research=""):
    prompt=f"""
You are a senior software engineer.

Task:
{question}

Research:
{research}

Generate clean code with explanation.
"""
    return invoke("Write production quality code.",prompt)

def reviewer_agent(code):
    prompt=f"""
Review the following code.

Return:
1. Issues
2. Improvements
3. Corrected version if needed.

{code}
"""
    return invoke("You are a strict reviewer.",prompt)

def workflow(question):
    agent=router(question)
    print(f"\nSelected Agent: {agent}\n")

    if agent=="research":
        res=research_agent(question)
        print(res)
        return

    if agent=="reviewer":
        review=reviewer_agent(question)
        print(review)
        return

    research=research_agent(question)
    print("="*60)
    print("RESEARCH")
    print("="*60)
    print(research)

    code=coder_agent(question,research)
    print("\n"+"="*60)
    print("CODE")
    print("="*60)
    print(code)

    review=reviewer_agent(code)
    print("\n"+"="*60)
    print("REVIEW")
    print("="*60)
    print(review)

if __name__=="__main__":
    print("Mini AI Team (Ollama)")
    while True:
        q=input("\nAsk (or exit): ")
        if q.lower()=="exit":
            break
        workflow(q)
