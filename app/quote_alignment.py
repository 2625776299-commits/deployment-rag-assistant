"""Locate quotes in original passages; tolerate only single-backtick inline markup.

No punctuation, whitespace, code-block, case or semantic normalization. Location
does not prove that the cited passage supports the model's conclusion.
"""
import re

_INLINE = re.compile(r'(?<![\\`])`([^`\r\n]+)`(?!`)')
_FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')


def _projection(text):
    chars, positions = [], []
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = _FENCE.match(line.rstrip('\r\n'))
        protected = fence is not None or marker is not None
        if marker:
            run, tail = marker.groups()
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not tail.strip():
                fence = None
        cursor = 0
        matches = [] if protected else list(_INLINE.finditer(line))
        for match in matches:
            for i in range(cursor, match.start()):
                chars.append(line[i]); positions.append((offset+i, offset+i+1))
            start, end = match.span(1)
            for i in range(start, end):
                chars.append(line[i])
                positions.append((offset+match.start() if i==start else offset+i,
                                  offset+match.end() if i==end-1 else offset+i+1))
            cursor = match.end()
        for i in range(cursor, len(line)):
            chars.append(line[i]); positions.append((offset+i, offset+i+1))
        offset += len(line)
    return ''.join(chars), positions


def align_quote(content, quote):
    if not isinstance(content,str) or not isinstance(quote,str) or not quote.strip():
        return None
    exact = content.find(quote)
    if exact >= 0:
        return {'mode':'exact', 'source_start':exact, 'source_end':exact+len(quote),
                'source_quote':quote, 'model_quote':quote}
    text, positions = _projection(content)
    needle, _ = _projection(quote)
    if not needle:
        return None
    start = text.find(needle)
    if start < 0 or text.find(needle,start+1)>=0:
        return None
    source_start = positions[start][0]
    source_end = positions[start+len(needle)-1][1]
    source_quote = content[source_start:source_end]
    if _projection(source_quote)[0] != needle:
        return None
    return {'mode':'inline_code_markup', 'source_start':source_start, 'source_end':source_end,
            'source_quote':source_quote, 'model_quote':quote}


def assessment_quote_alignments(proposal, candidates):
    by_id = {c['chunk_id']:c['content'] for c in candidates}
    alignments = []
    for assessment in proposal.get('assessments',[]):
        for ref in assessment['evidence']:
            match = align_quote(by_id.get(ref['chunk_id']),ref['quote'])
            if match is None:
                raise ValueError('support_quote_not_in_passage')
            alignments.append({'requirement_id':assessment['requirement_id'],
                               'chunk_id':ref['chunk_id'], **match})
    return alignments
