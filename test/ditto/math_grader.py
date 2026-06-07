import re
from typing import Optional

import sympy
from sympy.parsing import sympy_parser


BAD_SUBSTRINGS = ["^{", "^(", "=", "in", "U", "[", "]", "{", "}", "+-"]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]{}"


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = str(answer).strip()
    if answer.startswith("$"):
        answer = answer[1:]
    if answer.endswith("$"):
        answer = answer[:-1]
    try:
        match = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if match is not None:
            answer = match.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr:
                continue
            if substr[0] == "{":
                new_str += substr
                continue
            if len(substr) < 2:
                return string
            a = substr[0]
            b = substr[1]
            if b != "{":
                post_substr = substr[2:] if len(substr) > 2 else ""
                new_str += "{" + a + "}{" + b + "}" + post_substr
            else:
                post_substr = substr[2:] if len(substr) > 2 else ""
                new_str += "{" + a + "}" + b + post_substr
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _extract_braced_expr(expr: str, start: int):
    if start >= len(expr) or expr[start] != "{":
        return None, start
    depth = 0
    for idx in range(start, len(expr)):
        if expr[idx] == "{":
            depth += 1
        elif expr[idx] == "}":
            depth -= 1
            if depth == 0:
                return expr[start + 1:idx], idx + 1
    return expr[start + 1:], len(expr)


def _read_latex_token(expr: str, start: int):
    while start < len(expr) and expr[start].isspace():
        start += 1
    if start >= len(expr):
        return "", start
    if expr[start] == "{":
        return _extract_braced_expr(expr, start)
    if expr[start:start + 5] == "\\sqrt":
        return _replace_latex_functions(expr[start:]), len(expr)
    if expr[start] == "\\":
        match = re.match(r"\\[A-Za-z]+", expr[start:])
        if match:
            return match.group(0), start + len(match.group(0))
    return expr[start], start + 1


def _replace_latex_functions(expr: str):
    expr = expr.replace("\\left", "")
    expr = expr.replace("\\right", "")
    expr = expr.replace("\\{", "{")
    expr = expr.replace("\\}", "}")

    out = []
    idx = 0
    while idx < len(expr):
        if expr.startswith("\\frac", idx):
            idx += len("\\frac")
            numerator, idx = _read_latex_token(expr, idx)
            denominator, idx = _read_latex_token(expr, idx)
            out.append(
                f"(({_replace_latex_functions(numerator)})/({_replace_latex_functions(denominator)}))"
            )
            continue
        if expr.startswith("\\sqrt", idx):
            idx += len("\\sqrt")
            radicand, idx = _read_latex_token(expr, idx)
            out.append(f"sqrt({_replace_latex_functions(radicand)})")
            continue
        out.append(expr[idx])
        idx += 1

    expr = "".join(out)
    replacements = {
        "\\pi": "pi",
        "\\infty": "oo",
        "\\cdot": "*",
        "\\times": "*",
        "\\cup": "U",
        "\\pm": "+-",
        "\\in": "in",
    }
    for src, dst in replacements.items():
        expr = expr.replace(src, dst)
    return expr


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _strip_properly_formatted_commas(expr: str):
    matcher = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = matcher.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return expr


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str):
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _normalize(expr: str) -> Optional[str]:
    if expr is None:
        return None

    match = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if match is not None:
        expr = match.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")
    expr = re.sub(r"(\d)\s*\\frac", r"\1+\\frac", expr)

    for unit in [
            "degree",
            "degrees",
            "cm",
            "cent",
            "cents",
            "centimeter",
            "meter",
            "mile",
            "second",
            "minute",
            "hour",
            "day",
            "week",
            "month",
            "year",
            "foot",
            "feet",
            "inch",
            "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))

    if "\\" in expr:
        expr = _replace_latex_functions(expr)

    expr = re.sub(r"- *", "-", expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.lower()

    membership_match = re.match(r"^[a-z]{1,2}in(.+)$", expr)
    if membership_match:
        expr = membership_match.group(1)

    if len(expr.split("=")) == 2 and len(expr.split("=")[0]) <= 2:
        expr = expr.split("=")[1]

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    expr = expr.replace("pi", "")
    expr = expr.replace("oo", "")
    letters_in_expr = {char for char in expr if char.isalpha()}
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application, )
        ),
    )


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            return simplified == 0
    except Exception:
        pass
    return False


def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)
    ):
        return [elem.strip() for elem in expr[1:-1].split(",")]
    return [expr]


def grade_answer(given_answer: Optional[str], ground_truth: str) -> bool:
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None or given_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        return False
    if len(ground_truth_elems) != len(given_elems):
        return False

    for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
        if _is_frac(ground_truth_elem) and _is_frac(given_elem):
            is_correct = ground_truth_elem == given_elem
        elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
            is_correct = False
        else:
            is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
        if not is_correct:
            return False

    return True
