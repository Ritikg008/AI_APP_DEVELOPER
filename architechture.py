from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
import langchain


langchain.debug = True
langchain.verbose = True
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Optional, Literal
from operator import add
from langchain_core.messages import HumanMessage, SystemMessage

from MYAGENT.agent_prompts import *
from MYAGENT.langgraph_states import *
from MYAGENT.tools import write_file, read_file, get_current_directory, list_files

load_dotenv()

langchain.debug = True
langchain.verbose = True

llm = ChatGroq(model="openai/gpt-oss-120b")


# Define proper TypedDict state
class AgentState(TypedDict):
    user_prompt: str
    plan: Optional[Plan]
    task_plan: Optional[TaskPlan]
    coder_state: Optional[CoderState]
    status: Optional[str]
    messages: Annotated[list, add]


def planner_agent(state: AgentState) -> dict:
    """Converts user prompt into a structured Plan."""
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    return {"plan": resp}


def architect_agent(state: AgentState) -> dict:
    """Creates TaskPlan from Plan."""
    plan: Plan = state["plan"]
    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json())
    )
    if resp is None:
        raise ValueError("Architect did not return a valid response.")

    resp.plan = plan
    print(resp.model_dump_json())
    return {"task_plan": resp}


def coder_agent(state: AgentState) -> dict:
    """LangGraph tool-using coder agent - manual implementation."""
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        print("\n✅ All tasks completed!")
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    print(f"\n🔨 Working on: {current_task.filepath}")
    print(f"📝 Task: {current_task.task_description[:100]}...")
    
    try:
        existing_content = read_file.invoke({"filepath": current_task.filepath})
    except Exception as e:
        existing_content = f"File not found or error: {e}"
        print(f"⚠️  File doesn't exist yet: {current_task.filepath}")

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    coder_tools = [read_file, write_file, list_files, get_current_directory]
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(coder_tools)
    
    # Build sub-graph for ReAct pattern
    coder_graph = StateGraph(MessagesState)
    
    def call_model(state: MessagesState):
        """Call the LLM with tools."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: MessagesState) -> Literal["tools", "end"]:
        """Check if we should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        # If there are no tool calls, we're done
        if not last_message.tool_calls:
            return "end"
        return "tools"
    
    # Create tool node
    tool_node = ToolNode(coder_tools)
    
    # Build the coder sub-graph
    coder_graph.add_node("agent", call_model)
    coder_graph.add_node("tools", tool_node)
    
    coder_graph.add_edge(START, "agent")
    coder_graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    coder_graph.add_edge("tools", "agent")
    
    coder_agent_executor = coder_graph.compile()
    
    # Invoke the coder agent
    coder_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    react_result = coder_agent_executor.invoke(
        {"messages": coder_messages},
        config={"recursion_limit": 50}
    )
    
    print(f"✅ Completed: {current_task.filepath}")
    print(f"   Messages exchanged: {len(react_result.get('messages', []))}")

    coder_state.current_step_idx += 1
    return {
        "coder_state": coder_state,
        "messages": react_result.get("messages", [])
    }


def should_continue_coding(state: AgentState) -> str:
    """Conditional edge function for coder loop."""
    if state.get("status") == "DONE":
        return "end"
    return "continue"


# Create main graph with proper state type
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

# Add edges
graph.add_edge(START, "planner")
graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

# Add conditional edges
graph.add_conditional_edges(
    "coder",
    should_continue_coding,
    {
        "continue": "coder",
        "end": END
    }
)

# Compile the graph
agent = graph.compile()


if __name__ == "__main__":
    user_input = input("Enter your project prompt: ")
    
    print("\n" + "="*50)
    print("🚀 Starting Multi-Agent System")
    print("="*50)
    
    result = agent.invoke(
        {
            "user_prompt": user_input,
            "plan": None,
            "task_plan": None,
            "coder_state": None,
            "status": None,
            "messages": []
        },
        config={"recursion_limit": 100}
    )
    
    print("\n" + "="*50)
    print("✅ Final State")
    print("="*50)
    print(f"Status: {result.get('status')}")
    
    if result.get('task_plan'):
        print(f"\n📦 Project: {result['task_plan'].plan.name}")
        print(f"📝 Description: {result['task_plan'].plan.description}")
        print(f"\n📁 Files Created:")
        for step in result['task_plan'].implementation_steps:
            print(f"   • {step.filepath}")
    
    # List actual files created
    import os
    print(f"\n💾 Actual files in directory:")
    for file in os.listdir('.'):
        if os.path.isfile(file) and not file.startswith('.'):
            print(f"   • {file}")