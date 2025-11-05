import os
import re
import json
import subprocess
from typing import TypedDict, Annotated
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ------------------------- Setup -------------------------
load_dotenv()
st.set_page_config(page_title="Unit Test Generator with Feedback Loop", layout="wide")
st.title("🧪 Pytest Test Generator with Feedback Loop")

# --- LangChain / Groq ---
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
llm_generator = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)
llm_critic = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.1)
llm_fixer = ChatGroq(model="qwen/qwen3-32b", temperature=0.2)
llm_reporter = ChatGroq(model="qwen/qwen3-32b", temperature=0.3)

# --------------------- State Definition -------------------
class AgentState(TypedDict):
    readme_content: str
    user_functions: str
    detected_functions: list
    num_functions: int
    iteration_results: list
    test_code: str
    combined_code: str
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
    framework: str
    previous_errors: list

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
    
    # Pattern 8: Flask routes like GET /items - function_name()
    pattern8 = r'[-•]\s*`[A-Z]+\s+/[^`]*`.*?[-–—]\s*`?([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    functions.extend(re.findall(pattern8, readme))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_functions = []
    for func in functions:
        # Skip private methods and common non-function words
        if func not in seen and not func.startswith('_') and func.lower() not in ['module', 'key', 'class', 'object', 'property', 'input', 'output', 'returns', 'return']:
            seen.add(func)
            unique_functions.append(func)
    
    return unique_functions[:20]  # Max 20 functions

def extract_functions_from_python_file(python_code: str) -> list:
    """Extract function names directly from Python code using AST."""
    import ast
    
    functions = []
    
    try:
        tree = ast.parse(python_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods
                if not node.name.startswith('_'):
                    functions.append(node.name)
        
        return functions[:20]  # Max 20 functions
        
    except Exception as e:
        # Fallback to empty list if parsing fails
        return []

def detect_framework(user_code: str) -> str:
    """Detect if code uses Flask, Django, FastAPI, etc."""
    code_lower = user_code.lower()
    
    if 'from flask import' in code_lower or 'import flask' in code_lower:
        return 'flask'
    elif 'from django' in code_lower or 'import django' in code_lower:
        return 'django'
    elif 'from fastapi import' in code_lower or 'import fastapi' in code_lower:
        return 'fastapi'
    else:
        return 'generic'

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

def extract_user_functions(user_code: str, detected_function_names: list) -> str:
    """Extract only the detected functions/classes from user's code."""
    import ast
    
    try:
        tree = ast.parse(user_code)
        extracted_items = []
        extracted_names = set()
        
        for node in ast.walk(tree):
            # Extract standalone functions
            if isinstance(node, ast.FunctionDef):
                if node.name in detected_function_names and node.name not in extracted_names:
                    func_lines = user_code.split('\n')[node.lineno-1:node.end_lineno]
                    extracted_items.append('\n'.join(func_lines))
                    extracted_names.add(node.name)
            
            # Extract entire class if any method matches
            elif isinstance(node, ast.ClassDef):
                class_has_target_method = False
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name in detected_function_names:
                        class_has_target_method = True
                        break
                
                if class_has_target_method and node.name not in extracted_names:
                    class_lines = user_code.split('\n')[node.lineno-1:node.end_lineno]
                    extracted_items.append('\n'.join(class_lines))
                    extracted_names.add(node.name)
        
        if extracted_items:
            result = '\n\n'.join(extracted_items)
            # Keep essential imports that don't reference external packages
            essential_imports = []
            for line in user_code.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    # Only keep standard library imports
                    if not any(pkg in line for pkg in ['requests', 'urllib3', 'chardet', 'idna']):
                        essential_imports.append(line)
            
            if essential_imports:
                return '\n'.join(essential_imports) + '\n\n' + result
            return result
        else:
            return user_code
            
    except Exception as e:
        return user_code
    
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
    """Detect functions from README and Python file."""
    
    # Try to detect from README first
    functions_from_readme = extract_functions_from_readme(state["readme_content"])
    
    # Also extract from the actual Python code
    functions_from_code = extract_functions_from_python_file(state["user_functions"])
    
    # Detect framework
    framework = detect_framework(state["user_functions"])
    state["framework"] = framework
    
    # Combine both sources, prioritizing README (if it has functions)
    if functions_from_readme:
        functions = functions_from_readme
    elif functions_from_code:
        functions = functions_from_code
        state["history"].append({
            "iteration": state["iteration"],
            "agent": "detector",
            "action": f"⚠️ No functions in README, extracted {len(functions)} from Python code"
        })
    else:
        functions = []
        state["history"].append({
            "iteration": state["iteration"],
            "agent": "detector",
            "action": "❌ No functions detected in README or code"
        })
    
    state["detected_functions"] = functions
    state["num_functions"] = len(functions)
    state["history"].append({
        "iteration": state["iteration"],
        "agent": "detector",
        "action": f"Detected {len(functions)} functions ({framework} framework): {', '.join(functions)}"
    })
    
    return state

def test_generator_node(state: AgentState) -> AgentState:
    """Generator Agent - Creates test code based on README and user functions."""
    
    feedback_text = state.get("feedback", "")
    previous_test_preview = ""
    
    if state.get("test_code"):
        previous_test_preview = f"\n\nPREVIOUS TEST CODE (first 800 chars):\n{state['test_code'][:800]}"
    
    if not feedback_text:
        feedback_text = "Generate comprehensive unit tests based on the README and provided functions."
    
    framework = state.get("framework", "generic")
    
    function_list = ", ".join(state["detected_functions"])
    readme_preview = state['readme_content'][:2500]
    user_functions_preview = state['user_functions'][:2500]
    
    # Framework-specific instructions
    framework_instructions = ""
    if framework == 'flask':
        framework_instructions = """
FLASK-SPECIFIC REQUIREMENTS:
- Use the `client` fixture (automatically provided) for all route tests
- Test routes using: response = client.get('/route') or client.post('/route', json={...})
- Access response data with: response.get_json() or response.data
- Check status codes with: response.status_code
- DO NOT call route handlers directly like hello() - use client.get('/') instead
- ALL test functions MUST have 'client' as a parameter
- Example:
  def test_hello(client):
      response = client.get('/')
      assert response.status_code == 200
      assert b"Hello, Flask!" in response.data
"""
    elif framework == 'fastapi':
        framework_instructions = """
FASTAPI-SPECIFIC REQUIREMENTS:
- Use TestClient from fastapi.testclient
- Create client fixture if needed
- Test endpoints using: response = client.get('/route')
- Check response.status_code and response.json()
"""
    
    prompt = f"""
You are an expert Python test generator. Generate comprehensive unit tests.

DETECTED FUNCTIONS ({state['num_functions']}): {function_list}

FRAMEWORK DETECTED: {framework.upper()}
{framework_instructions}

CRITICAL REQUIREMENTS:
1. Generate EXACTLY ONE test function per detected function
2. Test naming convention: test_originalfunctionname (e.g., test_hello, test_get_items)
3. Total tests: {state['num_functions']}
4. Each test should be independent and self-contained
5. For {framework} framework, use appropriate fixtures and patterns

FEEDBACK FROM PREVIOUS ITERATION: {feedback_text}

README (preview):
{readme_preview}

USER'S FUNCTIONS (preview):
{user_functions_preview}
{previous_test_preview}

Generate test code with:
- Standard library imports (pytest, json, typing)
- {state['num_functions']} test functions (one per detected function)
- Proper use of fixtures for {framework} framework
- Clear assertions that validate behavior
- Docstrings explaining what each test validates

Return ONLY code wrapped in:
<PYTEST_FILE>
# test_generated.py
import pytest
import json
from typing import Dict, List

def test_hello(client):
    \"\"\"Test hello route returns greeting.\"\"\"
    response = client.get('/')
    assert response.status_code == 200
    assert b"Hello, Flask!" in response.data

def test_get_items(client):
    \"\"\"Test get_items route returns items list.\"\"\"
    response = client.get('/items')
    assert response.status_code == 200
    data = response.get_json()
    assert 'items' in data

# ... more tests (total: {state['num_functions']})
</PYTEST_FILE>

✅ EXACTLY {state['num_functions']} test functions
✅ Naming: test_originalfunctionname
✅ Use {framework} patterns correctly
✅ Include fixtures as needed (client for Flask)
✅ Each test must be independent and complete
❌ No markdown, no explanations outside tags
"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm_generator.invoke(messages)
    test_code = extract_code(response.content)
    
    state["test_code"] = test_code
    state["history"].append({
        "iteration": state["iteration"],
        "agent": "generator",
        "action": f"Generated {state['num_functions']} test functions for {framework}"
    })
    
    return state

def combiner_node(state: AgentState) -> AgentState:
    """Combine user functions with generated tests."""
    
    framework = state.get("framework", "generic")
    
    # Extract only the detected functions from user's code
    filtered_functions = extract_user_functions(
        state['user_functions'], 
        state['detected_functions']
    )
    
    # For Flask apps, we need special setup
    if framework == 'flask':
        # Remove any imports from filtered functions to avoid conflicts
        import re
        filtered_functions = re.sub(r'^import\s+.*$', '', filtered_functions, flags=re.MULTILINE)
        filtered_functions = re.sub(r'^from\s+.*import\s+.*$', '', filtered_functions, flags=re.MULTILINE)
        
        combined = f"""# Combined test file with Flask app and tests
import pytest
from flask import Flask

# Create Flask app for testing
app = Flask(__name__)

# Import request after app is created
from flask import request

# User's route handlers and data
{filtered_functions}

# Test fixtures
@pytest.fixture
def client():
    \"\"\"Create a test client for the Flask app.\"\"\"
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# ==================== TESTS ====================

{state['test_code']}
"""
    else:
        # Generic combination for non-Flask code
        # Remove any imports from filtered functions to avoid conflicts
        import re
        filtered_functions = re.sub(r'^import\s+.*$', '', filtered_functions, flags=re.MULTILINE)
        filtered_functions = re.sub(r'^from\s+.*import\s+.*$', '', filtered_functions, flags=re.MULTILINE)
        
        # Clean up the test code - remove problematic imports
        test_code_cleaned = state['test_code']
        # Remove imports that reference packages (like requests.models)
        test_code_cleaned = re.sub(r'^from\s+[\w.]+\s+import\s+.*$', '', test_code_cleaned, flags=re.MULTILINE)
        test_code_cleaned = re.sub(r'^import\s+[\w.]+.*$', lambda m: m.group(0) if not '.' in m.group(0).split()[1] else '', test_code_cleaned, flags=re.MULTILINE)
        
        combined = f"""# Combined test file with user functions and tests

# User's function implementations
{filtered_functions}

# ==================== TESTS ====================

{test_code_cleaned}
"""
    
    state["combined_code"] = combined
    state["history"].append({
        "iteration": state["iteration"],
        "agent": "combiner",
        "action": f"Combined {len(state['detected_functions'])} functions ({framework} detected)"
    })
    
    return state

def execution_node(state: AgentState) -> AgentState:
    """Execution Engine - Runs pytest on combined code."""
    
    with open("test_combined.py", "w", encoding="utf-8") as f:
        f.write(state["combined_code"])
    
    return_code, report, stdout, stderr = run_pytest_json("test_combined.py", 90)
    
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
    """Critic Agent - Analyzes results and decides next step."""
    
    summary = state["report"].get("summary", {})
    collected = summary.get("collected", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    errors = summary.get("errors", 0)
    
    # SUCCESS: All tests passed
    if collected > 0 and passed == collected and failed == 0 and errors == 0:
        state["status"] = "success"
        state["feedback"] = "All tests passed successfully"
        state["history"].append({
            "iteration": state["iteration"],
            "agent": "critic",
            "action": f"✅ SUCCESS - {passed}/{collected} tests passed"
        })
        return state
    
    # Extract failed test details
    failed_tests = []
    if state["report"].get("tests"):
        for test in state["report"]["tests"]:
            if test.get("outcome") in ["failed", "error"]:
                failed_tests.append({
                    "name": test.get("nodeid", ""),
                    "error": test.get("longrepr", "")[:400]
                })
    
    pytest_output = state["pytest_output"][:1200] if state["pytest_output"] else ""
    pytest_stderr = state["pytest_stderr"][:800] if state["pytest_stderr"] else ""
    
    framework = state.get("framework", "generic")
    
    prompt = f"""
Analyze pytest results and provide SPECIFIC, ACTIONABLE feedback.

FRAMEWORK: {framework.upper()}

RESULTS:
- Collected: {collected} (Expected: {state['num_functions']})
- Passed: {passed}
- Failed: {failed}
- Errors: {errors}
- Iteration: {state['iteration']} of {state['max_iterations']}

FAILED TESTS (detailed):
{json.dumps(failed_tests[:3], indent=2)}

PYTEST OUTPUT (last 1200 chars):
{pytest_output}

STDERR:
{pytest_stderr}

YOUR TASK: Analyze failures and provide SPECIFIC fixes.

Common Flask test issues:
1. Missing client fixture in test signature - ALL tests need (client) parameter
2. Calling routes directly instead of using client.get()/post()
3. Wrong assertion on response format (use response.data or response.get_json())
4. Not checking response.status_code
5. Shared state between tests (items list not cleared)

Common generic test issues:
1. Incorrect function calls or parameters
2. Wrong expected values in assertions
3. Missing imports or fixtures
4. Type mismatches in assertions

RESPONSE FORMAT (JSON only):

If all tests passed:
{{"status": "success", "message": "All tests passed"}}

If tests failed with SPECIFIC issues:
{{"status": "needs_fix", "feedback": "SPECIFIC ACTIONABLE FIXES: 1) test_get_item: Use client.get('/items/0') not get_item(0). 2) test_add_item: Use client.post('/items', json={{...}}) not add_item(). 3) All tests need 'client' fixture parameter."}}

If wrong number collected:
{{"status": "needs_fix", "feedback": "Expected {state['num_functions']} tests but collected {collected}. Regenerate with correct count and ensure all tests are properly named."}}

If max iterations reached:
{{"status": "max_iterations", "message": "Maximum iterations reached"}}

Be VERY SPECIFIC about what's wrong and how to fix it. Include test names and exact changes needed. Return ONLY valid JSON.
"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm_critic.invoke(messages)
    
    try:
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"status": "needs_fix", "feedback": "Could not parse critic response"}
    except:
        result = {"status": "needs_fix", "feedback": "Error parsing critic response"}
    
    state["status"] = result.get("status", "unknown")
    state["feedback"] = result.get("feedback", result.get("message", ""))
    state["history"].append({
        "iteration": state["iteration"],
        "agent": "critic",
        "action": f"Analysis: {state['status']} - {passed}/{collected} passed"
    })

    if state["status"] == "needs_fix":
        state["iteration"] += 1

    return state

def reporter_node(state: AgentState) -> AgentState:
    """Reporting Agent - Formats final output."""
    
    summary = state.get("report", {}).get("summary", {})
    collected = summary.get("collected", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    
    prompt = f"""
Generate a concise final report.

STATUS: {state['status']}
ITERATIONS: {state['iteration']}
FRAMEWORK: {state.get('framework', 'generic').upper()}
FUNCTIONS: {state['num_functions']}
RESULTS: {passed}/{collected} passed, {failed} failed

Detected functions: {', '.join(state['detected_functions'][:10])}

{"✅ SUCCESS - All tests passed!" if state['status'] == 'success' else ""}
{"⚠️ INCOMPLETE - Still have failing tests after max iterations" if state['status'] == 'max_iterations' else ""}
{"⚠️ STALLED - No improvement in recent iterations" if state['status'] == 'stalled' else ""}

Provide a brief, clear summary explaining the results.

IMPORTANT: NO <think> tags. Only the final report text.
"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm_reporter.invoke(messages)
    
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
    
    # Stop if no improvement for 2 consecutive iterations
    if len(state["iteration_results"]) >= 3:
        last_three = state["iteration_results"][-3:]
        if last_three[0]["passed"] == last_three[1]["passed"] == last_three[2]["passed"]:
            state["status"] = "stalled"
            state["feedback"] = "No improvement in last 2 iterations"
            return "reporter"
    
    if iteration > max_iterations:
        state["status"] = "max_iterations"
        state["feedback"] = f"Reached maximum iterations ({max_iterations})"
        return "reporter"
    
    if status == "needs_fix":
        return "generate"
    
    return "reporter"

# --------------------- Build Graph -------------------

def build_graph():
    """Build the LangGraph workflow."""
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("detect", function_detector_node)
    workflow.add_node("generate", test_generator_node)
    workflow.add_node("combine", combiner_node)
    workflow.add_node("execute", execution_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reporter", reporter_node)
    
    workflow.set_entry_point("detect")
    workflow.add_edge("detect", "generate")
    workflow.add_edge("generate", "combine")
    workflow.add_edge("combine", "execute")
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

st.markdown("""
Upload a README and provide your function implementations. The system will:
1. Detect functions from README and identify framework (Flask, FastAPI, Django, etc.)
2. Generate tests (one per function) with framework-specific patterns
3. Run tests against your functions
4. **Feedback loop**: Fix any failing tests automatically until all pass
""")

with st.sidebar:
    st.header("⚙️ Configuration")
    max_iterations = st.slider("Max Fix Iterations", 1, 7, 3)
    st.info("System generates one test per function with naming: test_functionname")
    st.success("✨ NEW: Automatic Flask/FastAPI/Django detection")
    st.markdown("### Process")
    st.markdown("""
    1. Upload README
    2. Provide functions
    3. Click Generate
    4. Feedback loop runs
    5. Framework-aware tests
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 README File")
    uploaded_file = st.file_uploader("Upload README.md", type=["md", "txt"])
    
    readme_content = None
    if uploaded_file:
        readme_content = uploaded_file.read().decode("utf-8", errors="ignore")
        with st.expander("Preview README", expanded=False):
            st.code(readme_content[:1000] + "..." if len(readme_content) > 1000 else readme_content, language="markdown")

with col2:
    st.subheader("💻 Your Functions")
    uploaded_functions_file = st.file_uploader("Upload functions.py", type=["py"])
    
    user_functions = None
    if uploaded_functions_file:
        user_functions = uploaded_functions_file.read().decode("utf-8", errors="ignore")
        with st.expander("Preview Functions", expanded=False):
            st.code(user_functions[:1000] + "..." if len(user_functions) > 1000 else user_functions, language="python")

st.divider()

if readme_content and user_functions and st.button("🚀 Generate Tests & Run Feedback Loop", type="primary", use_container_width=True):
    
    test_functions_readme = extract_functions_from_readme(readme_content)
    test_functions_code = extract_functions_from_python_file(user_functions)
    detected_framework = detect_framework(user_functions)
    
    if not test_functions_readme and not test_functions_code:
        st.error("""
        ❌ **No functions detected!**
        
        Make sure your README includes function signatures like:
        - `function_name()` in backticks
        - `def function_name(` in code blocks
        - Headers like `### function_name(args)`
        
        OR your Python file contains actual function definitions.
        """)
        st.stop()
    
    # Show preview of detected functions
    all_detected = test_functions_readme if test_functions_readme else test_functions_code
    st.info(f"✅ Pre-check: Found {len(all_detected)} functions in {detected_framework.upper()} app: {', '.join(all_detected[:5])}{'...' if len(all_detected) > 5 else ''}")
    
    initial_state = {
        "readme_content": readme_content,
        "user_functions": user_functions,
        "detected_functions": [],
        "num_functions": 0,
        "test_code": "",
        "combined_code": "",
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
        "history": [],
        "framework": "generic",
        "previous_errors": []
    }
    
    app = build_graph()
    config = {"configurable": {"thread_id": "test_generation_workflow"}}
    
    progress_container = st.container()
    
    with st.spinner("🔄 Running workflow with feedback loop..."):
        final_state = None
        for state in app.stream(initial_state, config):
            final_state = state
            
            if list(state.keys())[0] in ["detect", "generate", "combine", "execute", "critic"]:
                node_name = list(state.keys())[0]
                node_state = list(state.values())[0]
                
                with progress_container:
                    if node_name == "detect":
                        funcs = node_state.get('detected_functions', [])
                        fw = node_state.get('framework', 'generic')
                        st.info(f"🔍 Detected {len(funcs)} functions in {fw.upper()} app: {', '.join(funcs[:8])}{'...' if len(funcs) > 8 else ''}")
                    elif node_name == "generate":
                        iter_num = node_state.get('iteration', 1)
                        fw = node_state.get('framework', 'generic')
                        if iter_num == 1:
                            st.info(f"🤖 Iteration {iter_num}: Generating {node_state.get('num_functions', 0)} {fw}-aware tests...")
                        else:
                            st.info(f"🔧 Iteration {iter_num}: Fixing tests based on feedback...")
                    elif node_name == "combine":
                        fw = node_state.get('framework', 'generic')
                        st.info(f"🔗 Iteration {node_state.get('iteration', 1)}: Combining {fw} functions with tests...")
                    elif node_name == "execute":
                        st.info(f"⚙️ Iteration {node_state.get('iteration', 1)}: Running pytest...")
                    elif node_name == "critic":
                        status = node_state.get('status', '')
                        iter_num = node_state.get('iteration', 1)
                        summary = node_state.get('report', {}).get('summary', {})
                        passed = summary.get('passed', 0)
                        collected = summary.get('collected', 0)
                        
                        if status == "success":
                            st.success(f"✅ Iteration {iter_num}: All tests passed! ({passed}/{collected})")
                        elif status == "needs_fix":
                            st.warning(f"🔄 Iteration {iter_num}: {passed}/{collected} tests passed - Fixing...")
                        elif status == "stalled":
                            st.warning(f"⚠️ Iteration {iter_num}: No improvement detected - stopping")
    
    if final_state:
        final_state = list(final_state.values())[0]
        
        st.divider()
        st.subheader("📊 Workflow Results")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Functions Detected", final_state["num_functions"])
        col2.metric("Total Iterations", final_state["iteration"])
        
        status_emoji = "✅" if final_state["status"] == "success" else "⚠️"
        col3.metric("Final Status", f"{status_emoji} {final_state['status'].replace('_', ' ').title()}")
        
        if final_state.get("report"):
            summary = final_state["report"].get("summary", {})
            col4.metric("Tests Passed", f"{summary.get('passed', 0)}/{summary.get('collected', 0)}")
        
        # Show framework badge
        framework = final_state.get('framework', 'generic')
        if framework != 'generic':
            st.info(f"🎯 Framework: **{framework.upper()}** - Tests generated with {framework}-specific patterns")
        
        with st.expander("🔍 Function → Test Mapping", expanded=True):
            for i, func in enumerate(final_state["detected_functions"], 1):
                st.write(f"{i}. `test_{func}()` ← Tests → `{func}()`")
        
        st.divider()
        st.subheader("📈 Progress Across Iterations")
        
        for result in final_state["iteration_results"]:
            col1, col2, col3, col4 = st.columns(4)
            col1.write(f"**Iteration {result['iteration']}**")
            col2.metric("Collected", result['collected'])
            
            # Show delta for passed tests
            delta_passed = None
            if result['iteration'] > 1:
                prev_passed = final_state["iteration_results"][result['iteration']-2]['passed']
                delta_passed = result['passed'] - prev_passed
            
            col3.metric("Passed", result['passed'], delta=delta_passed)
            col4.metric("Failed", result['failed'])
        
        st.divider()
        st.subheader("📝 Final Test Code")
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
                            st.code(t["longrepr"][:600], language="bash")
                    elif outcome == "error":
                        st.error(f"💥 {nodeid} (Error)")
                        if t.get("longrepr"):
                            st.code(t["longrepr"][:600], language="bash")
        
        with st.expander("📜 Execution History"):
            for entry in final_state["history"]:
                agent_emoji = {
                    "detector": "🔍",
                    "generator": "🤖",
                    "combiner": "🔗",
                    "executor": "⚙️",
                    "critic": "🔬",
                    "reporter": "📋"
                }.get(entry['agent'], "•")
                st.write(f"{agent_emoji} **Iteration {entry['iteration']}** - {entry['agent'].title()}: {entry['action']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ Download Test File",
                data=final_state["test_code"],
                file_name="test_generated.py",
                mime="text/x-python",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="⬇️ Download Combined File (Functions + Tests)",
                data=final_state["combined_code"],
                file_name="test_combined.py",
                mime="text/x-python",
                use_container_width=True
            )

else:
    if not readme_content:
        st.info("⬆️ Please upload a README.md file")
    if not user_functions:
        st.info("💻 Please provide your functions.py file")

with st.expander("ℹ️ How it works"):
    st.markdown("""
    ### Automated Test Generation with Feedback Loop
    
    **Single-Phase Process:**
    1. **Upload README** - Contains function documentation
    2. **Provide Functions** - Your actual Python function implementations
    3. **Click Generate** - System starts the workflow
    
    **Automated Workflow:**
    1. 🔍 **Detector** - Extracts function names from README (max 20) and detects framework
    2. 🤖 **Generator** - Creates one test per function with framework-specific patterns
    3. 🔗 **Combiner** - Combines your functions with generated tests (adds Flask fixtures if needed)
    4. ⚙️ **Executor** - Runs pytest on combined code
    5. 🔬 **Critic** - Analyzes results with framework-aware feedback
    6. 🔄 **Feedback Loop** - If tests fail, regenerates tests with specific fixes
    7. 📋 **Reporter** - Generates final report when done
    
    **Key Features:**
    - ✅ One test per function with clear naming (test_functionname)
    - ✅ Automatic framework detection (Flask, FastAPI, Django)
    - ✅ Framework-specific test patterns (client fixtures, proper route testing)
    - ✅ Intelligent feedback loop fixes failing tests
    - ✅ Early stopping when no improvement detected
    - ✅ Your functions remain unchanged - only tests are modified
    - ✅ Full iteration tracking and detailed results
    - ✅ Download final tests and combined file
    
    **Framework Support:**
    - **Flask**: Automatically adds client fixture and uses proper route testing
    - **FastAPI**: Uses TestClient for endpoint testing
    - **Django**: Framework-aware test patterns
    - **Generic**: Standard pytest patterns for regular functions
    
    **Function Detection Patterns:**
    - Code blocks: `def function_name(`
    - Backticks: `` `function_name()` ``
    - Headers: `### function_name(args)`
    - Bold: `**function_name()**`
    - Lists: `- function_name(args)`
    - Flask routes: `GET /route - function_name()`
    """)

st.divider()
st.caption("🧪 Powered by LangGraph + Groq AI Models | Framework-Aware Test Generation with Automated Feedback Loop")
