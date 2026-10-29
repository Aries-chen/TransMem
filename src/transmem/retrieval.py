"""Algorithm 2, with a single candidate pool and continuation rule."""
from dataclasses import dataclass
import numpy as np

def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0

@dataclass
class Path:
    edges: tuple
    endpoint: str
    state: np.ndarray
    score: float

def other_endpoint(edge, endpoint):
    if edge['source'] == endpoint:
        return edge['target']
    if edge['target'] == endpoint:
        return edge['source']
    return None

def expand(query_vector, anchors, transitions, config):
    """Follow incident transitions to their other endpoint without reversing facts.

    Terminated paths retain their evidence. A beam limits active paths globally;
    repeated records or facts are rejected per path, and records are assembled once.
    """
    anchor_nodes = {a[k] for a in anchors for k in ('source', 'target')}
    seeds = []
    for edge in transitions:
        # Source-first matches the legacy choice when both endpoints are anchors.
        anchor = next((node for node in (edge['source'], edge['target'])
                       if node in anchor_nodes), None)
        if anchor is not None:
            score = cosine(query_vector, edge['embedding'])
            if score >= config.continuation_threshold:
                seeds.append((score, edge, other_endpoint(edge, anchor)))
    seeds.sort(key=lambda item: (-item[0], item[1]['id']))
    active = [Path((edge,), endpoint,
                   config.gamma*np.asarray(query_vector)+(1-config.gamma)*np.asarray(edge['embedding']), score)
              for score, edge, endpoint in seeds[:config.beam_size]]
    finished = []
    for _ in range(2, config.max_hops + 1):
        candidates = []
        for path in active:
            extensions = []
            used_ids = {edge['id'] for edge in path.edges}
            used_facts = {edge['fact'] for edge in path.edges}
            for edge in transitions:
                endpoint = other_endpoint(edge, path.endpoint)
                if (endpoint is None or edge['id'] in used_ids
                        or not edge['fact'] or edge['fact'] in used_facts):
                    continue
                score = cosine(path.state, edge['embedding'])
                if score < config.continuation_threshold or score-path.score < -config.drop_tolerance:
                    continue
                extensions.append(Path(path.edges+(edge,), endpoint,
                    config.gamma*path.state+(1-config.gamma)*np.asarray(edge['embedding']), score))
            if not extensions:
                finished.append(path)
            candidates.extend(extensions)
        candidates.sort(key=lambda p: (-p.score, tuple(e['id'] for e in p.edges)))
        active = candidates[:config.beam_size]
        if not active:
            break
    finished.extend(active)
    records = {}
    for path in finished:
        for edge in path.edges:
            records.setdefault(edge['id'], edge)
    return list(records.values()), [[e['id'] for e in p.edges] for p in finished]

def assemble(anchors, transitions, tokenizer, token_budget):
    lines = []
    for kind, records in [('FACT', anchors), ('TRANSITION', transitions)]:
        for record in records:
            status = 'PAST' if record.get('invalid_at') else 'CURRENT'
            line = f"[{kind}; {status}; {record.get('valid_at') or 'time unknown'}] {record['fact']}"
            trial = '\n'.join(lines+[line])
            if len(tokenizer.encode(trial)) <= token_budget:
                lines.append(line)
    return '\n'.join(lines)
