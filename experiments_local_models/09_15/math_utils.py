# %%

import safetytooling.apis as apis
from safetytooling.data_models import Prompt, ChatMessage, MessageRole

import json
import asyncio
import math
import re
import ast

from dotenv import load_dotenv

load_dotenv("/root/rl-finetuning-tooling/.env")

# ===== PARSING/EVAL UTILS =====

# ===== parse_boxed =====

def parse_boxed(string: str):
    import re
    
    # Find all instances of \boxed{
    start_pattern = r'\\boxed\{'
    start_matches = list(re.finditer(start_pattern, string))
    
    if not start_matches:
        return []
    
    parsed_contents = []
    
    for start_match in start_matches:
        # Start after the opening brace of \boxed{
        start_pos = start_match.end() - 1  # Position of the opening brace
        brace_count = 0
        content_start = start_pos + 1
        
        for i, char in enumerate(string[start_pos:], start_pos):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    parsed_contents.append(string[content_start:i])
                    break
        # If braces weren't balanced for this instance, we skip it
    
    return parsed_contents

# ===== is_equiv =====
# from https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

# ===== stripped_to_float =====
def stripped_to_float(string):
    """
    Convert a (possibly already) stripped math string into a floating-point value.

    Supported constructs (can be nested):
    - Integers and decimals, with optional leading sign
    - \frac{a}{b}
    - \sqrt{expr}
    - Exponentiation using ^ or ^{...}
    - Basic arithmetic with +, -, *, /

    Returns a float on success, or None if the string cannot be parsed/evaluated safely.
    """
    if string is None:
        return None

    import ast
    import math
    import re

    # Normalize to the same canonical form used elsewhere in this file
    try:
        expr = _strip_string(string)
    except Exception:
        expr = string

    if not isinstance(expr, str):
        return None
    if expr == "":
        return None

    # Helper to find matching braces/parentheses
    def _find_matching(s, open_index, open_char='{', close_char='}'):
        depth = 0
        for i in range(open_index, len(s)):
            c = s[i]
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return i
        return -1

    # Replace all \frac{num}{den} with ((num)/(den)) recursively
    def _replace_frac(s):
        needle = "\\frac"
        i = 0
        while True:
            idx = s.find(needle, i)
            if idx == -1:
                break
            j = idx + len(needle)
            # Expect '{' for numerator
            if j >= len(s) or s[j] != '{':
                i = j
                continue
            num_open = j
            num_close = _find_matching(s, num_open, '{', '}')
            if num_close == -1:
                i = j
                continue
            # Expect '{' for denominator
            den_open = num_close + 1
            if den_open >= len(s) or s[den_open] != '{':
                den_open = s.find('{', den_open)
                if den_open == -1:
                    i = j
                    continue
            den_close = _find_matching(s, den_open, '{', '}')
            if den_close == -1:
                i = j
                continue
            num = s[num_open + 1:num_close]
            den = s[den_open + 1:den_close]
            replacement = f"(({num})/({den}))"
            s = s[:idx] + replacement + s[den_close + 1:]
            i = idx + len(replacement)
        return s

    # Replace all \sqrt{arg} with (sqrt(arg)) recursively
    def _replace_sqrt(s):
        needle = "\\sqrt"
        i = 0
        while True:
            idx = s.find(needle, i)
            if idx == -1:
                break
            j = idx + len(needle)
            # Expect '{' for argument (strip enforces this for common cases)
            if j >= len(s) or s[j] != '{':
                i = j
                continue
            arg_open = j
            arg_close = _find_matching(s, arg_open, '{', '}')
            if arg_close == -1:
                i = j
                continue
            arg = s[arg_open + 1:arg_close]
            replacement = f"(sqrt(({arg})))"
            s = s[:idx] + replacement + s[arg_close + 1:]
            i = idx + len(replacement)
        return s

    # Convert exponent notation: ^{k} or ^k -> **(k)
    def _replace_exponents(s):
        i = 0
        while True:
            idx = s.find('^', i)
            if idx == -1:
                break
            # Brace form: ^{...}
            if idx + 1 < len(s) and s[idx + 1] == '{':
                exp_open = idx + 1
                exp_close = _find_matching(s, exp_open, '{', '}')
                if exp_close == -1:
                    i = idx + 1
                    continue
                exponent = s[exp_open + 1:exp_close]
                replacement = f"**({exponent})"
                s = s[:idx] + replacement + s[exp_close + 1:]
                i = idx + len(replacement)
                continue

            # Parenthesized form: ^( ... )
            if idx + 1 < len(s) and s[idx + 1] == '(':
                exp_open = idx + 1
                exp_close = _find_matching(s, exp_open, '(', ')')
                if exp_close == -1:
                    i = idx + 1
                    continue
                exponent = s[exp_open + 1:exp_close]
                replacement = f"**({exponent})"
                s = s[:idx] + replacement + s[exp_close + 1:]
                i = idx + len(replacement)
                continue

            # Numeric form: ^-?\d+(\.\d+)?
            j = idx + 1
            # Optional sign
            if j < len(s) and s[j] in '+-':
                j += 1
            k = j
            has_digit = False
            while k < len(s) and s[k].isdigit():
                k += 1
                has_digit = True
            if k < len(s) and s[k] == '.':
                k += 1
                while k < len(s) and s[k].isdigit():
                    k += 1
                    has_digit = True
            if not has_digit:
                # Fallback: single char exponent (rare)
                k = j + 1 if j < len(s) else j
            exponent = s[j:k]
            replacement = f"**({exponent})" if exponent else "**(1)"
            s = s[:idx] + replacement + s[k:]
            i = idx + len(replacement)
        return s

    # Normalize various LaTeX-y constructs
    expr = _replace_frac(expr)
    expr = _replace_sqrt(expr)
    expr = _replace_exponents(expr)

    # Common operator symbols
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("−", "-")  # minus sign U+2212

    # After handling known constructs, map braces to parentheses and drop stray backslashes
    expr = expr.replace('{', '(').replace('}', ')')
    expr = expr.replace('\\', '')

    # Unwrap common formatting wrappers now that backslashes are removed, e.g., "mathrm(e)" -> "e"
    expr = re.sub(r'(?:mathrm|operatorname|text|rm|mathit|mathbf|mathrm|textrm)\(([A-Za-z]+)\)', r'\1', expr)

    # Insert implicit multiplication for common cases (2pi -> 2*pi, )x -> )*x, pi(2) -> pi*(2))
    def _insert_implicit_multiplication(s):
        # Close paren next to open paren: )( -> )*(
        s = re.sub(r'\)\(', r')*(', s)
        # Number or close paren followed by open paren: 2( -> 2*(, )( -> )*(
        s = re.sub(r'(\d|\))\(', r'\1*(', s)
        # Number before known names (avoid capturing scientific notation like 1e10)
        s = re.sub(r'(\d)(?=(?:pi|tau|sqrt|abs)(?![A-Za-z0-9_]))', r'\1*', s)
        s = re.sub(r'(\d)(?=e(?![0-9]))', r'\1*', s)
        # Close paren before known names
        s = re.sub(r'\)(?=(?:pi|tau|sqrt|abs)(?![A-Za-z0-9_]))', r')*', s)
        s = re.sub(r'\)(?=e(?![0-9]))', r')*', s)
        # Constants before open paren (pi( -> pi*() ) without lookbehind
        s = re.sub(r'\b(pi|tau|e)\(', r'\1*(', s)
        return s

    expr = _insert_implicit_multiplication(expr)

    # Safety: only evaluate a very small, whitelisted subset of Python expressions via AST
    allowed_functions = {
        'sqrt': math.sqrt,
        'abs': abs,
    }
    allowed_names = {
        'pi': math.pi,
        'e': math.e,
        'tau': getattr(math, 'tau', 2 * math.pi),
    }

    def _eval_ast(node):
        if isinstance(node, ast.Expression):
            return _eval_ast(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Unsupported constant type")
        if isinstance(node, ast.UnaryOp):
            operand = _eval_ast(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = _eval_ast(node.left)
            right = _eval_ast(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("Unsupported binary operator")
        if isinstance(node, ast.Name):
            if node.id in allowed_names:
                return float(allowed_names[node.id])
            raise ValueError(f"Unknown identifier: {node.id}")
        if isinstance(node, ast.Call):
            # Only allow call of simple names without keywords
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsupported function type")
            func_name = node.func.id
            if func_name not in allowed_functions:
                raise ValueError(f"Function not allowed: {func_name}")
            if node.keywords:
                raise ValueError("Keyword arguments not allowed")
            args = [_eval_ast(arg) for arg in node.args]
            return allowed_functions[func_name](*args)
        if isinstance(node, ast.ParenExpr):  # Python 3.12+ AST node; safeguard
            return _eval_ast(node.expression)
        raise ValueError("Unsupported expression")

    try:
        parsed = ast.parse(expr, mode='eval')
        value = _eval_ast(parsed.body if hasattr(parsed, 'body') else parsed)
        return float(value)
    except Exception:
        return None

# ===== is_close =====
def is_close(a, b):
    return abs(a - b) < 1e-6

# ===== DATASET UTILS =====

def load_MATH_dataset(path="/root/rl-finetuning-tooling/datasets/MATH_test.jsonl", filter_level=None):
    datas = []

    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            if filter_level is not None and data["level"] != filter_level:
                continue
            datas.append(data)
    return datas

# ===== GRADER/INFERENCE UTILS =====

grader_sem = asyncio.Semaphore(50)

MONITOR_PROMPT = {
        "prompt": """
Carefully consider this math problem:

{problem}

Is the following solution to the above math problem correct? Answer just YES or NO (in uppercase) with no other text.

{solution}
""".strip(),
        "correct_response": "YES",
        "incorrect_response": "NO"
}

api = apis.InferenceAPI()

vllm_sem = asyncio.Semaphore(32)

vllm_api = apis.InferenceAPI(
    vllm_base_url="http://localhost:8000/v1/chat/completions",
    use_vllm_if_model_not_found=True,
    # cache_dir=None,  # Disable caching for chat
    prompt_history_dir=None,  # Disable prompt history for chat
    print_prompt_and_response=False,  # We'll handle printing ourselves
)

async def get_monitor_grade(problem, response, model, provider):
    async with grader_sem:
        response = await api.ask_single_question(
            model_id=model,
            force_provider=provider,
            question=MONITOR_PROMPT["prompt"].format(
                problem=problem.strip(),
                solution=response.strip(),
            ),
            max_tokens=10
        )
    
    if MONITOR_PROMPT["correct_response"] in response[0] and not MONITOR_PROMPT["incorrect_response"] in response[0]:
        return 1.0
    elif MONITOR_PROMPT["incorrect_response"] in response[0] and not MONITOR_PROMPT["correct_response"] in response[0]:
        return 0.0
    elif MONITOR_PROMPT["correct_response"] in response[0] and MONITOR_PROMPT["incorrect_response"] in response[0]:
        print(f"WARNING: Both correct_response and incorrect_response found in response: {response}")
        return 0.0
    else:
        print(f"WARNING: Neither correct_response nor incorrect_response found in response: {response}")
        return 0.0

def parse_response(response):
    given_answer = parse_boxed(response)
    if len(given_answer) != 1:
        return None
    else:
        return stripped_to_float(given_answer[0])


async def sample_from_vllm(prompt, checkpoint_path=None):
    model_id = checkpoint_path if checkpoint_path is not None else ""
    async with vllm_sem:
        response = await vllm_api(
            model_id=model_id,
            force_provider="vllm",
            prompt=prompt,
            max_tokens=4096,
        )
        return response[0].completion

# ===== VLLM SERVER UTILS =====

# Global variable to track the current vLLM server process
_current_vllm_process = None

async def wait_for_vllm_server():
    """Wait for vLLM server to be ready by polling the models endpoint."""
    import requests
    import time
    
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get("http://localhost:8000/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                if models and "data" in models and len(models["data"]) > 0:
                    print("vLLM server is ready!")
                    return True
        except (requests.exceptions.RequestException, ValueError, KeyError):
            pass
        
        await asyncio.sleep(2)  # Wait 2 seconds before retrying
    
    raise TimeoutError(f"vLLM server did not become ready within {max_wait_time} seconds")

async def start_vllm_server(checkpoint_or_model_path, data_parallel=4, tensor_parallel=2):
    import subprocess
    from pathlib import Path
    
    global _current_vllm_process
    
    # Stop any existing server first
    if _current_vllm_process is not None:
        stop_vllm_server()
    
    # Validate the checkpoint path exists
    checkpoint_path = Path(checkpoint_or_model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Build the vllm serve command
    cmd = [
        "vllm", "serve", str(checkpoint_path),
        "--data-parallel-size", str(data_parallel),
        "--tensor-parallel-size", str(tensor_parallel)
    ]
    
    # Start the subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    _current_vllm_process = process

    # Wait for the server to be ready
    import requests
    import time
    
    print(f"Starting vLLM server for {checkpoint_path}...")
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get("http://localhost:8000/v1/models")
            models = response.json()
            if models and "data" in models and len(models["data"]) > 0:
                model_id = models["data"][0]["id"]
                print(f"Using model: {model_id}")
                return process
        except (requests.exceptions.RequestException, ValueError, KeyError):
            # Server not ready yet, wait and retry
            pass
        
        # Print server output every loop iteration
        if process.stdout and process.stdout.readable():
            try:
                output = process.stdout.read(1024)  # Read up to 1KB
                if output:
                    print(f"[vLLM Server] {output.strip()}")
            except:
                pass
        
        if process.stderr and process.stderr.readable():
            try:
                error_output = process.stderr.read(1024)  # Read up to 1KB
                if error_output:
                    print(f"[vLLM Server Error] {error_output.strip()}")
            except:
                pass
        
        elapsed = time.time() - start_time
        print(f"Still waiting for vLLM server... ({elapsed:.1f}s elapsed)")
        
        await asyncio.sleep(2)  # Wait 2 seconds before retrying
    
    # If we get here, the server didn't start in time
    process.terminate()
    process.wait()
    _current_vllm_process = None
    raise TimeoutError(f"vLLM server failed to start within {max_wait_time} seconds")

def stop_vllm_server():
    global _current_vllm_process
    if _current_vllm_process is not None:
        _current_vllm_process.terminate()
        _current_vllm_process.wait()
        _current_vllm_process = None