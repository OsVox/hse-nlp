"""Conservative preprocessing of existing targets; never adds examples.

Only comments and empty physical lines outside strings are candidates. Keep a
change only if parsing and compilation succeed and the full Python AST (apart
from source locations) is identical. This does not promise equivalence for code
that reads its own source or relies on traceback/line-number inspection.
"""
import ast
import io
import tokenize


def cleaned_batches(batches):
    """Replace each supplied target once; preserve examples and input order."""
    for batch in batches:
        yield [dict(row, code=strip_nonexecuting_text(row['code'])[0]) for row in batch]


def strip_nonexecuting_text(code: str) -> tuple[str, dict]:
    info = dict(comments=0, blank_lines=0, changed=False, rejected=False)
    try:
        original_tree = ast.parse(code)
        compile(original_tree, '<original>', 'exec')
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        lines = code.splitlines(keepends=True)
        protected = set()
        for token in tokens:
            if token.type == tokenize.STRING:
                protected.update(range(token.start[0], token.end[0]+1))
        # Work backwards so column offsets still address the original source.
        for token in reversed(tokens):
            if token.type == tokenize.COMMENT:
                row, start = token.start
                _, end = token.end
                lines[row-1] = lines[row-1][:start] + lines[row-1][end:]
                info['comments'] += 1
        kept = []
        for row, line in enumerate(lines, 1):
            # Preserve explicit line continuations, including an empty next line.
            continuation = row > 1 and lines[row-2].rstrip('\r\n').endswith('\\')
            if not line.strip() and row not in protected and not continuation:
                info['blank_lines'] += 1
            else:
                kept.append(line)
        cleaned = ''.join(kept)
        cleaned_tree = ast.parse(cleaned)
        compile(cleaned_tree, '<cleaned>', 'exec')
        if ast.dump(original_tree, include_attributes=False) != ast.dump(cleaned_tree, include_attributes=False):
            raise ValueError('Changed Python AST')
    except (SyntaxError, ValueError, TypeError, tokenize.TokenError, IndentationError):
        info['rejected'] = True
        return code, info
    info['changed'] = cleaned != code
    return cleaned, info
