import unittest
import numpy as np
from transmem.config import Config, paper_config
from transmem.retrieval import expand, assemble

def edge(i, source, target, vector, kind='intra'):
    return {'id':i,'source':source,'target':target,'embedding':vector,'fact':i,'kind':kind}

class RetrievalTests(unittest.TestCase):
    def test_first_hop_filters_before_beam(self):
        records=[edge('bad','a','b',[-1,0]),edge('good','a','c',[1,0])]
        found,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config(beam_size=1))
        self.assertEqual([r['id'] for r in found],['good'])

    def test_state_updates_after_first_hop(self):
        records=[edge('first','a','b',[0.6,0.8]),edge('second','b','c',[0,1])]
        c=Config(gamma=0,continuation_threshold=.5,drop_tolerance=1)
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,c)
        self.assertEqual(paths,[['first','second']])
        c.gamma=1
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,c)
        self.assertEqual(paths,[['first']])

    def test_edge_type_does_not_allocate_separate_beams(self):
        records=[edge('intra','a','b',[.8,.6]),edge('inter','a','c',[1,0],'inter')]
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config(beam_size=1))
        self.assertEqual(paths,[['inter']])

    def test_both_directions_and_hop_limit(self):
        records=[edge('first','a','b',[1,0]),edge('reverse','x','b',[1,0]),
                 edge('cycle','b','a',[1,0]),edge('second','b','c',[1,0])]
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config(max_hops=1))
        # The cycle also touches an anchor; first-hop adjacency is intentionally undirected.
        self.assertTrue(all(len(p)==1 for p in paths))
        records=records[:2]+records[3:]
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config())
        self.assertEqual(paths,[['first','reverse'],['first','second']])

    def test_incoming_seed_moves_to_source_and_preserves_records(self):
        records=[edge('0-incoming','a','b',[1,0]),edge('1-next','x','a',[1,0])]
        original=[dict(record) for record in records]
        found,paths=expand([1,0],[{'source':'b','target':'z'}],records,Config(beam_size=1))
        self.assertEqual(paths,[['0-incoming','1-next']])
        self.assertEqual(records,original)
        self.assertEqual(found,original)

    def test_later_incoming_hops_move_to_other_endpoint(self):
        records=[edge('0-first','a','b',[1,0]),edge('1-incoming','x','b',[1,0]),
                 edge('2-incoming','y','x',[1,0])]
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config(beam_size=1))
        self.assertEqual(paths,[['0-first','1-incoming','2-incoming']])

    def test_two_anchor_endpoints_seed_one_path_source_first(self):
        records=[edge('0-first','a','b',[1,0]),edge('1-next','x','b',[1,0])]
        _,paths=expand([1,0],[{'source':'a','target':'b'}],records,Config(beam_size=1))
        self.assertEqual(paths,[['0-first','1-next']])
        _,paths=expand([1,0],[{'source':'a','target':'b'}],records[:1],Config())
        self.assertEqual(paths,[['0-first']])

    def test_stopped_path_retains_evidence(self):
        records=[edge('first','a','b',[1,0]),edge('drop','b','c',[.6,.8])]
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config(drop_tolerance=.1))
        self.assertEqual(paths,[['first']])

    def test_beam_pruned_parent_is_not_saved_as_stopped(self):
        records=[edge('0-first','a','b',[1,0]),edge('1-first','z','c',[.8,.6]),
                 edge('2-next','b','d',[1,0]),edge('3-next','b','e',[.9,.43589]),
                 edge('4-next','c','f',[.8,.6])]
        found,paths=expand([1,0],[{'source':'a','target':'z'}],records,
                           Config(beam_size=2,max_hops=2,gamma=1,drop_tolerance=1))
        self.assertEqual(paths,[['0-first','2-next'],['0-first','3-next']])
        self.assertNotIn('1-first',[record['id'] for record in found])

    def test_self_loop_seed_can_continue_without_reusing_it(self):
        records=[edge('0-loop','a','a',[1,0]),edge('1-next','a','b',[1,0])]
        found,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config(beam_size=1))
        self.assertEqual(paths,[['0-loop','1-next']])
        self.assertEqual(len(found),2)

    def test_self_loop_after_first_hop_is_retained(self):
        records=[edge('0-first','a','b',[1,0]),edge('1-loop','b','b',[1,0]),
                 edge('2-next','b','c',[1,0])]
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config(beam_size=1))
        self.assertEqual(paths,[['0-first','1-loop','2-next']])

    def test_node_revisit_does_not_repeat_records(self):
        records=[edge('0-first','a','b',[1,0]),edge('1-return','b','a',[1,0])]
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config(beam_size=1))
        self.assertEqual(paths,[['0-first','1-return']])

    def test_same_fact_with_different_id_is_not_repeated(self):
        records=[edge('0-first','a','b',[1,0]),edge('1-copy','b','b',[1,0])]
        records[1]['fact']=records[0]['fact']
        _,paths=expand([1,0],[{'source':'a','target':'z'}],records,Config(beam_size=1))
        self.assertEqual(paths,[['0-first']])

    def test_final_configs(self):
        self.assertEqual([paper_config(s,b).anchor_topk for s in ('32k','128k')
            for b in ('qwen3-8b','qwen3.5-27b')],[50,90,90,110])
        self.assertEqual(paper_config('128k','qwen3.5-27b').gamma,.6)
        for subset in ('32k', '128k'):
            for backbone in ('qwen3-8b', 'qwen3.5-27b'):
                config = paper_config(subset, backbone)
                self.assertEqual(config.graphiti_invalidation_limit, 3)
                self.assertEqual(config.graphiti_node_dedupe_candidate_limit, 8)

if __name__=='__main__':
    unittest.main()
