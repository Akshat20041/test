import os
import re
import json
import subprocess
from typing import TypedDict, Annotated
import streamlit as st

import importlib, subprocess, sys
if importlib.util.find_spec("dotenv") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv==1.0.1"])
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ------------------------- Setup -------------------------
load_dotenv()
st.set_page_config(page_title="Function-Based Unit Test Generator", layout="wide")
st.title("🧪 Function-Based Pytest Test Generator with LangGraph")

# --- LangChain / Groq ---
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
llm_generator = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
llm_critic = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.1)
llm_reporter = ChatGroq(model="qwen/qwen3-32b", temperature=0.3)

# --------------------- State Definition -------------------
class AgentState(TypedDict):
    readme_content: str
    detected_functions: list
    num_functions: int
    iteration_results: list
    test_code: str
    pytest_output: str
    pytest_stderr: str
    return_code: int
    report: dict
    iteration: int
    max_iterations: int
    feedback: str
    status: str
    final_message: str
    history: list

# --------------------- Helper Functions -------------------
def extract_functions_from_readme(readme: str) -> list:
    """Extract function names from README using multiple patterns."""
    functions = []
    
    # Pattern 1: def function_name( in code blocks
    pattern1 = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    functions.extend(re.findall(pattern1, readme))
    
    # Pattern 2: function_name(self) in method signatures
    pattern2 = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(self'
    functions.extend(re.findall(pattern2, readme))
    
    # Pattern 3: `function_name()` or `function_name(args)`
    pattern3 = r'`([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)`'
    functions.extend(re.findall(pattern3, readme))
    
    # Pattern 4: ### function_name(args) or ### function_name - headers with function calls
    pattern4 = r'###\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    functions.extend(re.findall(pattern4, readme))
    
    # Pattern 5: function_name(args) at start of line (not in code blocks)
    pattern5 = r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)(?:\s*$|\s*[-:])'
    functions.extend(re.findall(pattern5, readme, re.MULTILINE))
    
    # Pattern 6: **function_name(args)** in bold
    pattern6 = r'\*\*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\*\*'
    functions.extend(re.findall(pattern6, readme))
    
    # Pattern 7: - function_name(args) in lists
    pattern7 = r'[-•]\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    functions.extend(re.findall(pattern7, readme))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_functions = []
    for func in functions:
        # Skip private methods and common non-function words
        if func not in seen and not func.startswith('_') and func.lower() not in ['module', 'key', 'class', 'object', 'property', 'input', 'output', 'returns', 'return']:
            seen.add(func)
            unique_functions.append(func)
    
    return unique_functions[:15]  # Max 20 functions

def extract_code(raw: str) -> str:
    """Clean the LLM output and extract pure Python code."""
    if not raw:
        return ""
    
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"```(?:python)?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    raw = raw.strip()
    
    match = re.search(r"<PYTEST_FILE>([\s\S]*?)</PYTEST_FILE>", raw, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return raw.strip()

def run_pytest_json(test_file: str, timeout_sec: int = 60):
    """Run pytest and parse JSON report."""
    cmd = ["pytest", test_file, "--disable-warnings", "--maxfail=20", "--json-report", "-q"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        report = None
        if os.path.exists(".report.json"):
            with open(".report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
        return (proc.returncode, report, proc.stdout, proc.stderr)
    except Exception as e:
        return (-1, None, "", str(e))

# --------------------- Agent Nodes -------------------

def function_detector_node(state: AgentState) -> AgentState:
    """Detect functions from README."""
    functions = extract_functions_from_readme(state["readme_content"])
    
    state["detected_functions"] = functions
    state["num_functions"] = len(functions)
    state["history"].append({
        "iteration": state["iteration"],
        "agent": "detector",
        "action": f"Detected {len(functions)} functions: {', '.join(functions)}"
    })
    
    return state

def generator_node(state: AgentState) -> AgentState:
    """Generator Agent - Creates test code with one test per function."""
    
    feedback_text = state.get("feedback", "")
    previous_code_text = ""
    
    if state.get("test_code"):
        # Only show preview of previous code
        previous_code_text = f"\nPREVIOUS CODE (first 1000 chars):\n{state['test_code'][:1000]}"
    
    if not feedback_text:
        feedback_text = "Generate comprehensive unit tests based on the README."
    
    # Create concise function list
    function_list = ", ".join(state["detected_functions"])
    
    # Truncate README if too long
    readme_preview = state['readme_content'][:3000]
    
    prompt = f"""
Expert Python test generator.

FUNCTIONS ({state['num_functions']}): {function_list}

REQUIREMENTS:
1. ONE test per function
2. Naming: test1_functionName, test2_functionName, etc.
3. Total: {state['num_functions']} tests
4. Self-contained file (no external imports)
5. Add imports: pytest, typing.Callable, Union, Tuple
6. Implement WORKING placeholder functions (not just raise NotImplementedError)

FEEDBACK: {feedback_text}

README (preview):
{readme_preview}
{previous_code_text}

Generate test_generated.py with:
- Imports at top
- WORKING placeholder class/function definitions with actual logic
- {state['num_functions']} test functions that will PASS

CRITICAL: Placeholder implementations must be functional and make tests PASS, not raise errors.

Return ONLY code in:
<PYTEST_FILE>
code here
</PYTEST_FILE>

✅ Self-contained, executable Python
✅ EXACTLY {state['num_functions']} tests
✅ Working implementations that make tests pass
❌ No markdown, no explanations
❌ No NotImplementedError or pass statements
"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm_generator.invoke(messages)
    test_code = extract_code(response.content)
    
    state["test_code"] = test_code
    state["history"].append({
        "iteration": state["iteration"],
        "agent": "generator",
        "action": f"Generated {state['num_functions']} test functions"
    })
    
    return state

def execution_node(state: AgentState) -> AgentState:
    """Execution Engine - Runs pytest."""
    
    with open("test_generated.py", "w", encoding="utf-8") as f:
        f.write(state["test_code"])
    
    return_code, report, stdout, stderr = run_pytest_json("test_generated.py", 90)
    
    state["return_code"] = return_code
    state["report"] = report or {}
    state["pytest_output"] = stdout
    state["pytest_stderr"] = stderr
    state["history"].append({
        "iteration": state["iteration"],
        "agent": "executor",
        "action": f"Executed tests - Return code: {return_code}"
    })
    
    summary = report.get("summary", {}) if report else {}
    state["iteration_results"].append({
        "iteration": state["iteration"],
        "collected": summary.get("collected", 0),
        "passed": summary.get("passed", 0),
        "failed": summary.get("failed", 0),
        "errors": summary.get("errors", 0)
    })
    
    return state

def critic_node(state: AgentState) -> AgentState:
    """Reflection Agent - Analyzes results and decides next step."""
    
    summary = state["report"].get("summary", {})
    collected = summary.get("collected", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    errors = summary.get("errors", 0)
    
    # CHECK IF ALL TESTS PASSED IMMEDIATELY - Early exit
    if collected == state['num_functions'] and passed == collected and failed == 0 and errors == 0:
        state["status"] = "success"
        state["feedback"] = "All tests passed successfully"
        state["history"].append({
            "iteration": state["iteration"],
            "agent": "critic",
            "action": f"Analysis: success - {passed}/{collected} tests passed"
        })
        return state
    
    # Truncate long outputs
    pytest_output = state["pytest_output"][:2000] if state["pytest_output"] else ""
    pytest_stderr = state["pytest_stderr"][:1000] if state["pytest_stderr"] else ""
    test_code_preview = state["test_code"][:1500] if state["test_code"] else ""
    
    prompt = f"""
Analyze pytest results and decide next action.

EXPECTED: {state['num_functions']} tests for functions: {', '.join(state['detected_functions'])}

RESULTS:
- Collected: {collected} (Expected: {state['num_functions']})
- Passed: {passed}
- Failed: {failed}
- Errors: {errors}
- Return Code: {state['return_code']}

ITERATION: {state['iteration']} of {state['max_iterations']}

TEST CODE (preview):
{test_code_preview}

OUTPUT:
{pytest_output}

STDERR:
{pytest_stderr}

Respond with ONE JSON format:

1. All tests passed and count correct:
{{"status": "success", "message": "Tests passed"}}

2. Wrong test count:
{{"status": "incomplete", "feedback": "Need exactly {state['num_functions']} tests"}}

3. Test code has errors (fix placeholder implementations to make tests pass):
{{"status": "test_error", "feedback": "Specific fix needed - implement working placeholder functions"}}

4. Source code issue (only if test code looks correct):
{{"status": "source_error", "message": "Source code problem"}}

Return ONLY valid JSON.
"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm_critic.invoke(messages)
    
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"status": "test_error", "feedback": "Could not parse critic response"}
    except:
        result = {"status": "test_error", "feedback": "Error parsing critic response"}
    
    state["status"] = result.get("status", "unknown")
    state["feedback"] = result.get("feedback", result.get("message", ""))
    state["history"].append({
        "iteration": state["iteration"],
        "agent": "critic",
        "action": f"Analysis: {state['status']}"
    })

    if state["status"] in ["test_error", "incomplete"]:
        state["iteration"] += 1

    return state

def reporter_node(state: AgentState) -> AgentState:
    """Reporting Agent - Formats final output."""
    
    summary = state.get("report", {}).get("summary", {})
    collected = summary.get("collected", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    
    # Truncate outputs if too long
    pytest_output = state.get('pytest_output', '')[:1000]
    pytest_stderr = state.get('pytest_stderr', '')[:1000]
    
    prompt = f"""
Generate a concise final report for the user.

STATUS: {state['status']}
ITERATIONS: {state['iteration']}
FUNCTIONS TESTED: {state['num_functions']}
RESULTS: {passed}/{collected} passed, {failed} failed

Expected: {state['num_functions']} tests
Detected functions: {', '.join(state['detected_functions'][:10])}

{"✅ SUCCESS - All tests passed!" if state['status'] == 'success' else ""}
{"⚠️ INCOMPLETE - Wrong number of tests or still failing" if state['status'] in ['incomplete', 'max_iterations'] else ""}
{"❌ SOURCE ERROR - User's code has bugs" if state['status'] == 'source_error' else ""}

{f"Errors: {pytest_stderr[:500]}" if state['status'] == 'source_error' else ""}

Provide a brief, clear summary for the user explaining what happened.

IMPORTANT: Do NOT include any <think> tags or reasoning in your response. 
Provide ONLY the final report text without any internal thoughts or analysis.
"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm_reporter.invoke(messages)
    
    # Remove any <think> tags from the response
    clean_response = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL | re.IGNORECASE)
    clean_response = clean_response.strip()
    
    state["final_message"] = clean_response
    state["history"].append({
        "iteration": state["iteration"],
        "agent": "reporter",
        "action": "Generated final report"
    })
    
    return state

# --------------------- Routing Logic -------------------

def should_continue(state: AgentState) -> str:
    """Decide whether to continue iteration or end."""
    
    status = state.get("status", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if status == "success":
        return "reporter"
    
    if status == "source_error":
        return "reporter"
    
    if iteration > max_iterations:
        state["status"] = "max_iterations"
        state["feedback"] = f"Reached maximum iterations ({max_iterations})"
        return "reporter"
    
    if status in ["test_error", "incomplete"]:
        return "generate"
    
    return "reporter"

# --------------------- Build Graph -------------------

def build_graph():
    """Build the LangGraph workflow."""
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("detect", function_detector_node)
    workflow.add_node("generate", generator_node)
    workflow.add_node("execute", execution_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reporter", reporter_node)
    
    workflow.set_entry_point("detect")
    workflow.add_edge("detect", "generate")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "generate": "generate",
            "reporter": "reporter"
        }
    )
    workflow.add_edge("reporter", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# --------------------- Streamlit UI ----------------------

st.markdown("Upload a README, and the AI will automatically detect functions and generate one test per function.")

with st.sidebar:
    st.header("⚙️ Configuration")
    max_iterations = st.slider("Max iterations", 1, 5, 3)
    st.info("System will generate one test per detected function (max 20 functions).")
    st.markdown("### Naming Convention")
    st.code("test1_functionName\ntest2_functionName\ntest3_functionName")

uploaded_file = st.file_uploader("Upload README.md", type=["md", "txt"])

if uploaded_file:
    readme_content = uploaded_file.read().decode("utf-8", errors="ignore")
    
    with st.expander("📄 README Preview", expanded=False):
        st.code(readme_content, language="markdown")
    
    if st.button("🚀 Generate Function-Based Tests", type="primary"):
        
        initial_state = {
            "readme_content": readme_content,
            "detected_functions": [],
            "num_functions": 0,
            "test_code": "",
            "iteration_results": [],
            "pytest_output": "",
            "pytest_stderr": "",
            "return_code": -1,
            "report": {},
            "iteration": 1,
            "max_iterations": max_iterations,
            "feedback": "",
            "status": "",
            "final_message": "",
            "history": []
        }
        
        app = build_graph()
        config = {"configurable": {"thread_id": "test_generation_1"}}
        
        progress_container = st.container()
        
        with st.spinner("Running agentic workflow..."):
            final_state = None
            for state in app.stream(initial_state, config):
                final_state = state
                
                if list(state.keys())[0] in ["detect", "generate", "execute", "critic"]:
                    node_name = list(state.keys())[0]
                    node_state = list(state.values())[0]
                    
                    with progress_container:
                        if node_name == "detect":
                            funcs = node_state.get('detected_functions', [])
                            st.info(f"🔍 Detected {len(funcs)} functions: {', '.join(funcs[:5])}{'...' if len(funcs) > 5 else ''}")
                        elif node_name == "generate":
                            st.info(f"🤖 Iteration {node_state.get('iteration', 1)}: Generating {node_state.get('num_functions', 0)} tests...")
                        elif node_name == "execute":
                            st.info(f"⚙️ Iteration {node_state.get('iteration', 1)}: Executing tests...")
                        elif node_name == "critic":
                            status = node_state.get('status', '')
                            if status == "success":
                                st.success(f"✅ Iteration {node_state.get('iteration', 1)}: All tests passed!")
                            elif status in ["test_error", "incomplete"]:
                                st.warning(f"🔄 Iteration {node_state.get('iteration', 1)}: Refining tests...")
                            elif status == "source_error":
                                st.error(f"❌ Iteration {node_state.get('iteration', 1)}: Source code issue detected")
        
        if final_state:
            final_state = list(final_state.values())[0]
            
            st.divider()
            st.subheader("📊 Workflow Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Functions Detected", final_state["num_functions"])
            col2.metric("Total Iterations", final_state["iteration"])
            col3.metric("Final Status", final_state["status"].replace("_", " ").title())
            
            if final_state.get("report"):
                summary = final_state["report"].get("summary", {})
                col4.metric("Tests Passed", f"{summary.get('passed', 0)}/{summary.get('collected', 0)}")
            
            with st.expander("🔍 Detected Functions", expanded=True):
                for i, func in enumerate(final_state["detected_functions"], 1):
                    st.write(f"{i}. `test{i}_{func}()` → Tests `{func}()`")
            
            st.divider()
            st.subheader("📈 Progress Across Iterations")
            for result in final_state["iteration_results"]:
                col1, col2, col3, col4 = st.columns(4)
                col1.write(f"**Iteration {result['iteration']}**")
                col2.metric("Collected", result['collected'])
                col3.metric("Passed", result['passed'], delta=result['passed'] - (final_state["iteration_results"][result['iteration']-2]['passed'] if result['iteration'] > 1 else 0))
                col4.metric("Failed", result['failed'])
            
            st.divider()
            st.subheader("📝 Generated Test Code")
            st.code(final_state["test_code"], language="python")
            
            st.subheader("📋 Final Report")
            st.markdown(final_state["final_message"])
            
            if final_state.get("report"):
                with st.expander("🔍 Detailed Test Results"):
                    tests = final_state["report"].get("tests", [])
                    for t in tests:
                        nodeid = t.get("nodeid", "")
                        outcome = t.get("outcome", "")
                        if outcome == "passed":
                            st.success(f"✅ {nodeid}")
                        elif outcome == "failed":
                            st.error(f"❌ {nodeid}")
                            if t.get("longrepr"):
                                st.code(t["longrepr"], language="bash")
            
            with st.expander("📜 Execution History"):
                for entry in final_state["history"]:
                    st.write(f"**Iteration {entry['iteration']}** - {entry['agent'].title()}: {entry['action']}")
            
            st.download_button(
                label="⬇️ Download Test File",
                data=final_state["test_code"],
                file_name="test_generated.py",
                mime="text/x-python"
            )
else:
    st.info("⬆️ Upload a README.md file to start generating function-based tests.")
    
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### Function-Based Test Generation with LangGraph
        
        This system automatically detects functions and generates exactly one test per function:
        
        1. **Function Detector** - Extracts function names from README (max 15)
        2. **Generator Agent** - Creates one test per function with naming: test1_funcName, test2_funcName, etc.
        3. **Execution Engine** - Runs pytest and captures results
        4. **Critic Agent** - Validates test count and results, provides feedback
        5. **Reporter Agent** - Formats final output
        
        **Key Features:**
        - ✅ Automatic function detection from README
        - ✅ One test per function (max 20 functions)
        - ✅ Standardized naming: test(i)_functionName
        - ✅ Iterative refinement on test failures
        - ✅ Full execution history tracking
        - ✅ Working placeholder implementations
        
        **Function Detection Patterns:**
        - Code blocks: `def function_name(`
        - Backticks: `` `function_name()` ``
        - Headers: `### function_name(args)`
        - Bold text: `**function_name()**`
        - Start of line: `function_name(args)`
        """)
