"""Conservative local numeric-equality checks. No labels, eval, or symbolic algebra.

Unknown is not incorrect; a false equality does not imply a wrong final answer.
Only numeric components of explicit '=' chains are compared. We do not fill in
variables, remove units inside expressions, or infer missing equations from prose.
"""
import ast
from fractions import Fraction
import re

import numpy as np

FEATURE_NAMES = ('log_checked', 'log_inconsistent', 'inconsistent_fraction', 'no_checked_equations')
MATH = re.compile(r'\$\$(.*?)\$\$|(?<!\\)\$([^\n$]*?)(?<!\\)\$|\\\[(.*?)\\\]|\\\((.*?)\\\)|<<(.*?)>>', re.S)


def _braced(s, start):
    while start < len(s) and s[start].isspace():
        start += 1
    if start >= len(s) or s[start] != '{':
        raise ValueError('Missing brace')
    depth = 1
    end = start + 1
    while end < len(s) and depth:
        depth += (s[end] == '{') - (s[end] == '}')
        end += 1
    if depth:
        raise ValueError('Unclosed brace')
    return s[start+1:end-1], end


def _latex(s, depth=0):
    if depth > 16:
        raise ValueError('Too deeply nested')
    # Expand only recognized wrappers; keep unsupported commands as barriers.
    pattern = re.compile(r'\\(dfrac|tfrac|frac|boxed|mathbf|mathrm|textbf|text)\b')
    out, last = [], 0
    for_match = pattern.search(s)
    while for_match:
        m = for_match
        out.append(s[last:m.start()])
        first, end = _braced(s, m.end())
        if m[1] in ('frac', 'dfrac', 'tfrac'):
            second, end = _braced(s, end)
            out.append('((' + _latex(first, depth+1) + ')/(' + _latex(second, depth+1) + '))')
        elif m[1] == 'text':
            out.append('@')  # units/prose are never silently deleted internally
        else:
            out.append('(' + _latex(first, depth+1) + ')')
        last = end
        for_match = pattern.search(s, last)
    out.append(s[last:])
    return ''.join(out)


def normalize(s):
    s = _latex(s)
    for old, new in [('\\times', '*'), ('\\cdot', '*'), ('\\div', '/'), ('×', '*'), ('·', '*'), ('÷', '/'), ('−', '-'), ('–', '-')]:
        s = s.replace(old, new)
    for old in ('\\left', '\\right', '\\,', '\\;', '\\!', '\\quad', '\\qquad'):
        s = s.replace(old, ' ')
    s = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', s)
    s = re.sub(r'([0-9]+(?:\.[0-9]+)?)\s*\\?%', r'(\1/100)', s)
    s = s.replace('^', '**').replace('{', '(').replace('}', ')')
    return s


def numeric_value(s):
    """Interpret a bounded arithmetic AST using exact rational arithmetic."""
    s = s.strip()
    if not s or len(s) > 256 or not re.fullmatch(r'[\d.\s()+*/-]+', s):
        raise ValueError('Not a complete numeric expression')
    root = ast.parse(s, mode='eval')
    if len(list(ast.walk(root))) > 96:
        raise ValueError('Expression too large')

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(s, node)
            if not re.fullmatch(r'\d+(?:\.\d*)?|\.\d+', token):
                raise ValueError('Unsupported literal')
            value = Fraction(token)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): value = a+b
            elif isinstance(node.op, ast.Sub): value = a-b
            elif isinstance(node.op, ast.Mult): value = a*b
            elif isinstance(node.op, ast.Div): value = a/b
            elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 8: value = a**int(b)
            else: raise ValueError('Unsupported operator')
        else:
            raise ValueError('Unsupported syntax')
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 256:
            raise ValueError('Number too large')
        return value
    return visit(root)


def _component(s, right=False):
    # A trailing LaTeX prose/unit annotation is allowed after a complete result.
    # A leading/internal annotation prevents checking that expression.
    if right and '@' in s:
        prefix, suffix = s.split('@', 1)
        if not suffix.strip(' @\\\t\r '):
            s = prefix
    s = s.strip().strip('$`').strip()
    if s.endswith('.') and not s.endswith('..'):
        s = s[:-1].rstrip()
    return s


def inspect_text(text):
    if len(text) > 100000:
        raise ValueError('Unexpectedly large candidate')
    chunks = []
    blank = list(text)
    for m in MATH.finditer(text):
        chunks.append(next(g for g in m.groups() if g is not None))
        # Retain line boundaries and barriers: deleting spans must never join
        # unrelated currency values/equations left on either side of a span.
        blank[m.start():m.end()] = ['\n' if c == '\n' else '#' for c in m.group()]
    # Remaining full lines can contain plain-text or <<calculator>> equalities.
    chunks.extend(''.join(blank).splitlines())
    checks = []
    for chunk in chunks:
        if '=' not in chunk or any(x in chunk for x in ('≈', '\\approx', '\\sim', '...', '…', '%')):
            continue
        # Quotient-with-remainder and rounding prose are not equality assertions.
        if re.search(r'\b(remainder|round(?:ed|ing)?|approximately)\b', chunk, re.I):
            continue
        try:
            normalized = normalize(chunk)
        except (ValueError, RecursionError):
            continue
        for line in normalized.splitlines():
            components = line.split('=')
            for left, right in zip(components, components[1:]):
                lhs, rhs = _component(left), _component(right, right=True)
                try:
                    a, b = numeric_value(lhs), numeric_value(rhs)
                except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
                    continue
                # Explicit decimal results may be rounded to their displayed precision.
                # Integer and fraction results require exact equality.
                stripped_rhs = rhs.strip('() ')
                decimal = re.fullmatch(r'[-+]?\d+\.(\d+)', stripped_rhs)
                tolerance = Fraction(1, 2 * 10**len(decimal[1])) if decimal else Fraction(0)
                checks.append({'left': lhs, 'right': rhs, 'computed_left': str(a), 'computed_right': str(b),
                               'tolerance': str(tolerance), 'inconsistent': abs(a-b) > tolerance})
    return checks


def features_for_text(text):
    checks = inspect_text(text)
    n, bad = len(checks), sum(c['inconsistent'] for c in checks)
    return [np.log1p(n), np.log1p(bad), bad / n if n else 0.0, float(n == 0)]


def make_arithmetic_features(bank):
    return np.asarray([[features_for_text(c['text']) for c in row['candidates']] for row in bank], dtype=np.float64)
