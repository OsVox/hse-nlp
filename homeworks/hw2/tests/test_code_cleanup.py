import ast

import pytest

from homeworks.hw2.code_cleanup import strip_nonexecuting_text, cleaned_batches


@pytest.mark.parametrize('code', [
    '# explanation\ndef solve():\n    # calculate height\n    return 2 * 54 / 12  # inches\n',
    'def solve():\n    s = "# not a comment"\n    return len(s)\n',
    'def solve():\n    s = """a\n\n# retained string\nb"""\n    return len(s)\n',
    'def solve():\n    """Keep the docstring."""\n    return 9\n',
    'def solve():\n    return (\n        9 # value\n\n    )\n',
    'def solve():\n    x = 9; s = "é#✓" # Unicode comment\n    return x\n',
    'def solve():\n    x = 9 \\\n\n    return x\n',
])
def test_preserves_ast_and_execution(code):
    cleaned, info = strip_nonexecuting_text(code)
    assert not info['rejected']
    assert ast.dump(ast.parse(cleaned)) == ast.dump(ast.parse(code))
    original, result = {}, {}
    exec(code, original)
    exec(cleaned, result)
    assert original['solve']() == result['solve']()
    assert original['solve'].__doc__ == result['solve'].__doc__
    assert strip_nonexecuting_text(cleaned)[0] == cleaned


def test_comment_removal_and_invalid_program_fallback():
    code = '# note\n\ndef solve():\n    return 9 # inches\n'
    cleaned, info = strip_nonexecuting_text(code)
    assert info['comments'] == 2 and info['changed']
    assert cleaned == 'def solve():\n    return 9 \n'
    invalid = 'def solve(:\n# broken\n'
    assert strip_nonexecuting_text(invalid)[0] == invalid
    assert strip_nonexecuting_text(invalid)[1]['rejected']


def test_training_stream_retains_examples_without_mutating_originals():
    row = {'question': 'height?', 'code': '# note\ndef solve():\n    return 9\n'}
    result = list(cleaned_batches([[row], [row]]))
    assert len(result) == 2 and all(len(batch) == 1 for batch in result)
    assert result[0][0]['question'] == row['question']
    assert result[0][0]['code'] == 'def solve():\n    return 9\n'
    assert row['code'].startswith('# note')
