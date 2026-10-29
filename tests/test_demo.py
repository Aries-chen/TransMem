import json
from pathlib import Path
import unittest


class DemoTests(unittest.TestCase):
    def test_two_questions_straddle_update(self):
        folder = Path(__file__).resolve().parents[1] / 'examples/demo'
        history = json.loads((folder / 'histories.jsonl').read_text(encoding='utf-8'))['messages']
        questions = [json.loads(line) for line in (folder / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(history), 4)
        self.assertEqual([q['history_end'] for q in questions], [2, 3])
        self.assertEqual([q['answer'] for q in questions], ['a', 'b'])
        self.assertEqual(history[-1]['timestamp'], '2026-12-06T15:30:00+00:00')
        earlier = ' '.join(m['content'] for m in history[:3]).lower()
        self.assertNotIn('cycling', earlier)
        self.assertNotIn('closed for repairs', earlier)
        self.assertIn('closed for repairs', history[3]['content'])
        self.assertIn('Cycling is my new favorite', history[3]['content'])
        self.assertEqual(questions[0]['query'], questions[1]['query'])
        self.assertEqual(questions[0]['options'], questions[1]['options'])
