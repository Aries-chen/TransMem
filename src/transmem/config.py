from dataclasses import asdict, dataclass

@dataclass
class Config:
    subset: str = '32k'
    backbone: str = 'qwen3-8b'
    anchor_topk: int = 50
    gamma: float = 0.8
    continuation_threshold: float = 0.4
    drop_tolerance: float = 0.1
    beam_size: int = 3
    max_hops: int = 6
    macro_threshold: float = 0.65
    micro_threshold: float = 0.25
    maturity_messages: int = 15
    micro_entity_cap: int = 50
    buffer_size: int = 5
    history_topk: int = 10
    grounding_threshold: float = 0.5
    summary_interval: int = 10
    context_token_budget: int = 8192
    graphiti_invalidation_limit: int = 3
    graphiti_node_dedupe_candidate_limit: int = 8
    graphiti_disable_edge_invalidation_search: int = 0
    graphiti_relevant_schema_limit: int = 3

    def validate(self):
        for name in ('anchor_topk', 'beam_size', 'max_hops', 'maturity_messages',
                     'micro_entity_cap', 'buffer_size', 'history_topk',
                     'summary_interval', 'context_token_budget'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        for name in ('gamma', 'continuation_threshold', 'drop_tolerance',
                     'macro_threshold', 'micro_threshold', 'grounding_threshold'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        return self

    def as_dict(self):
        return asdict(self)

def paper_config(subset, backbone):
    larger = backbone == 'qwen3.5-27b'
    return Config(subset=subset, backbone=backbone,
                  anchor_topk=(110 if larger else 90) if subset == '128k' else (90 if larger else 50),
                  gamma=0.6 if larger else 0.8)
