import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from graphiti_core.llm_client.config import LLMConfig
from pydantic import BaseModel
from transmem.clients import KGClient
from transmem.config import Config
from transmem.memory import Memory, Topic


class Payload(BaseModel):
    summary: str


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, finish='stop'):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish, message=SimpleNamespace(content='I prefer group hikes.'))], usage=None)
        api = Mock()
        api.with_options.return_value = api
        api.chat.completions.create = AsyncMock(return_value=response)
        client = KGClient(LLMConfig(model='test', base_url='http://localhost:8000/v1',
                                    max_tokens=8192), client=api, disable_thinking=True)
        return client, api

    async def test_summary_has_separate_bound_and_no_schema(self):
        client, api = self.make_client()
        self.assertEqual(client.max_tokens, 8192)
        self.assertFalse(hasattr(client, 'total_prompt_tokens'))
        self.assertFalse(hasattr(client, 'token_scope'))
        self.assertEqual(await client.summarize('Summarize.'), 'I prefer group hikes.')
        kwargs = api.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs['max_tokens'], 1024)
        self.assertNotIn('response_format', kwargs)
        api.with_options.assert_called_once_with(timeout=90, max_retries=0)

    async def test_truncated_summary_rejected(self):
        client, _ = self.make_client('length')
        with self.assertRaises(ValueError):
            await client.summarize('Summarize.')

    async def test_explicit_graphiti_budget_is_capped(self):
        client, api = self.make_client()
        api.chat.completions.create.return_value.choices[0].message.content = '{"summary":"valid"}'
        await client._generate_response([], Payload, max_tokens=16384)
        self.assertEqual(api.chat.completions.create.call_args.kwargs['max_tokens'], 8192)

    async def test_summary_failure_preserves_state(self):
        memory = Memory.__new__(Memory)
        memory.graph = SimpleNamespace(driver=SimpleNamespace(
            execute_query=AsyncMock(return_value=([{'fact':'a fact'}], None, None))))
        memory.client = SimpleNamespace(summarize=AsyncMock(side_effect=TimeoutError))
        memory.vector = AsyncMock()
        topic = Topic('test_topic_0', 'Previous summary', [1., 0.], pending=3)
        await memory.refresh(topic)
        self.assertEqual((topic.summary, topic.embedding, topic.pending),
                         ('Previous summary', [1., 0.], 3))
        memory.vector.assert_not_called()

    async def test_json_repair_retained(self):
        client, _ = self.make_client()
        self.assertEqual(client._parse_structured_result('{"summary": "valid",}', Payload),
                         {'summary':'valid'})

    async def test_same_endpoint_transition_is_stored_with_grounding_gate(self):
        for vector, expected in [([1., 0.], 1), ([0., 1.], 0)]:
            with self.subTest(vector=vector):
                memory = Memory.__new__(Memory)
                memory.config = Config()
                memory.namespace = 'test'
                memory.topics = [Topic('test_topic_0', 'Coffee', [1., 0.])]
                memory.buffer = [{'timestamp':'2026-12-06', 'content':'I now enjoy coffee.'}]
                memory.created_transitions = 0
                memory.normal_facts = AsyncMock(return_value=[])
                memory.entities = AsyncMock(return_value=[{
                    'id':'coffee', 'gid':'test_topic_0', 'embedding':[1., 0.]}])
                memory.vector = AsyncMock(return_value=vector)
                memory.client = SimpleNamespace(generate_response=AsyncMock(return_value={
                    'shifts':[{'event_type':'Preference Change', 'old_state':'Dislikes coffee',
                               'new_state':'Enjoys coffee', 'trigger':'Tried a milder roast',
                               'valid_at':'2026-12-06'}]}))
                write = AsyncMock()
                memory.graph = SimpleNamespace(driver=SimpleNamespace(execute_query=write))
                await memory.detect()
                self.assertEqual(memory.created_transitions, expected)
                self.assertEqual(write.await_count, expected)
                if expected:
                    self.assertEqual(write.call_args.kwargs['source'], 'coffee')
                    self.assertEqual(write.call_args.kwargs['target'], 'coffee')
                self.assertEqual(memory.buffer, [])
