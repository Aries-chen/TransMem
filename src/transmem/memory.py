import logging
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import spacy
import httpx
from openai import AsyncOpenAI
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.prompts.models import Message
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from .clients import KGClient, GENERIC_SCHEMA, ShiftEventList
from .prompts import _SHIFT_DETECT_PROMPT, EVO_ROLLING_TOPIC_PROMPT
from .retrieval import cosine, expand

log = logging.getLogger(__name__)

class NoReranker(CrossEncoderClient):
    async def rank(self, query, passages):
        raise RuntimeError('TransMem uses RRF, not an LLM reranker')

@dataclass
class Topic:
    gid: str
    summary: str
    embedding: list
    messages: int = 0
    pending: int = 0

class Memory:
    def __init__(self, config, namespace, settings):
        self.config, self.namespace = config, namespace
        self.topics, self.buffer = [], []
        self.events = []
        self.created_transitions = 0
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError as error:
            raise RuntimeError('Install the router model: python -m spacy download en_core_web_sm') from error
        self.client = KGClient(LLMConfig(api_key=settings['api_key'], model=settings['model'],
            small_model=settings['model'], base_url=settings['llm_url'], temperature=0,
            max_tokens=8192), disable_thinking=True,
            client=AsyncOpenAI(api_key=settings['api_key'],base_url=settings['llm_url'], max_retries=1,
                http_client=httpx.AsyncClient(trust_env=settings['trust_env'],timeout=300)))
        self.embedder = OpenAIEmbedder(OpenAIEmbedderConfig(api_key=settings['embedding_api_key'],
            embedding_model=settings['embedding_model'], embedding_dim=1024,
            base_url=settings['embedding_url']),
            client=AsyncOpenAI(api_key=settings['embedding_api_key'],base_url=settings['embedding_url'],
                http_client=httpx.AsyncClient(trust_env=settings['trust_env'],timeout=120)))
        self.graph = Graphiti(settings['neo4j_uri'], settings['neo4j_user'], settings['neo4j_password'],
                              llm_client=self.client, embedder=self.embedder, cross_encoder=NoReranker())

    @property
    def gids(self):
        return [t.gid for t in self.topics]

    async def initialize(self):
        await self.graph.build_indices_and_constraints()

    async def vector(self, text):
        return await self.embedder.create(input_data=text)

    def keywords(self, text):
        doc = self.nlp(text)
        return {x.lemma_.lower().strip() for x in list(doc.noun_chunks)+list(doc.ents)
                if len(x.lemma_.strip()) > 1}

    async def entities(self, gids):
        rows, _, _ = await self.graph.driver.execute_query(
            'MATCH (n:Entity) WHERE n.group_id IN $gids '
            'RETURN n.uuid AS id,n.name AS name,n.name_embedding AS embedding,n.group_id AS gid '
            'ORDER BY n.created_at, n.uuid', gids=gids)
        return [dict(r) for r in rows]

    async def normal_facts(self, query, k):
        if not self.gids:
            return []
        recipe = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        recipe.limit = k
        result = await self.graph.search_(query, config=recipe, group_ids=self.gids)
        # Transition records use a separate relationship type, so they cannot
        # consume normal-fact search positions before the RRF cutoff.
        return [{'id':e.uuid, 'source':e.source_node_uuid, 'target':e.target_node_uuid,
                 'fact':e.fact, 'valid_at':str(e.valid_at) if e.valid_at else None,
                 'invalid_at':str(e.invalid_at) if e.invalid_at else None} for e in result.edges]

    async def ingest(self, message, index):
        text = message['content']
        timestamp = datetime.fromisoformat(message['timestamp'])
        vector = await self.vector(text)
        best = max(self.topics, key=lambda t: cosine(vector,t.embedding), default=None)
        macro = cosine(vector,best.embedding) if best else -1
        micro = 0.0
        if best is None or macro < self.config.macro_threshold:
            best = Topic(f'{self.namespace}_topic_{len(self.topics)}', text, vector)
            self.topics.append(best)
            route = 'BIRTH'
        else:
            entities = (await self.entities([best.gid]))[-self.config.micro_entity_cap:]
            known = set()
            for entity in entities:
                doc = self.nlp(entity['name'].lower())
                known.add(doc.text)
                known.update(t.lemma_ for t in doc if not t.is_stop and len(t.lemma_) > 1)
            keywords = self.keywords(text)
            micro = sum(any(k in e or e in k for e in known) for k in keywords)/max(1,len(keywords))
            route = ('EVOLVE' if best.messages >= self.config.maturity_messages and
                     micro < self.config.micro_threshold else 'DEEPENING')
        await self.graph.add_episode(name=f'message-{index}', episode_body=f"{message.get('role','user')}: {text}",
            source_description='Chronological user interaction', reference_time=timestamp,
            source=EpisodeType.message, group_id=best.gid, entity_types=GENERIC_SCHEMA,
            update_communities=False, skip_node_summaries=True)
        best.messages += 1
        self.events.append({'message':index,'route':route,'topic':best.gid,'macro':macro,'micro':micro})
        log.info('message=%s route=%s macro=%.3f micro=%.3f',index,route,macro,micro)
        if route == 'DEEPENING':
            best.pending += 1
        if route == 'EVOLVE':
            self.buffer.append(message)
            if len(self.buffer) >= self.config.buffer_size:
                await self.detect()
                await self.refresh(best)
        elif best.pending >= self.config.summary_interval:
            await self.refresh(best)

    async def refresh(self, topic):
        rows,_,_ = await self.graph.driver.execute_query(
            'MATCH ()-[e:RELATES_TO]->() WHERE e.group_id=$gid '
            'RETURN e.fact AS fact ORDER BY e.created_at DESC LIMIT 30', gid=topic.gid)
        prompt = EVO_ROLLING_TOPIC_PROMPT.format(topic=topic.gid.split('_topic_')[-1],
            prev_summary=topic.summary, new_facts='\n'.join(r['fact'] for r in rows))
        try:
            summary = await self.client.summarize(prompt)
            embedding = await self.vector(summary)
        except Exception:
            log.warning('Topic summary refresh failed; preserving previous summary and embedding', exc_info=True)
            return
        topic.summary = summary
        topic.embedding = embedding
        topic.pending = 0

    async def detect(self):
        activity = '\n'.join(f"[{m['timestamp']}] {m['content']}" for m in self.buffer)
        facts = await self.normal_facts(activity, self.config.history_topk)
        evidence = '\n'.join(f"[{f['valid_at']}] {f['fact']}" for f in facts)
        prompt = _SHIFT_DETECT_PROMPT.format(sorted_facts=f'HISTORICAL FACTS:\n{evidence}\nNEW ACTIVITY:\n{activity}')
        response = await self.client.generate_response([Message(role='user',content=prompt)], ShiftEventList)
        entities = [e for e in await self.entities(self.gids) if e['embedding']]
        for record in response.get('shifts',[]):
            old, new = await self.vector(record['old_state']), await self.vector(record['new_state'])
            if not entities:
                continue
            source = max(entities,key=lambda e: cosine(old,e['embedding']))
            target = max(entities,key=lambda e: cosine(new,e['embedding']))
            if min(cosine(old,source['embedding']),cosine(new,target['embedding'])) < self.config.grounding_threshold:
                continue
            fact = f"{record['event_type']}: {record['old_state']} -> {record['new_state']} (Trigger: {record['trigger']})"
            properties = dict(record, uuid=str(uuid4()), fact=fact, fact_embedding=await self.vector(fact),
                kind='intra' if source['gid']==target['gid'] else 'inter', namespace=self.namespace)
            await self.graph.driver.execute_query(
                'MATCH (s:Entity {uuid:$source}),(t:Entity {uuid:$target}) '
                'CREATE (s)-[e:TRANSITION]->(t) SET e=$properties',
                source=source['id'],target=target['id'],properties=properties)
            self.created_transitions += 1
        log.info('detector: %d candidates, %d stored transitions total',len(response.get('shifts',[])),self.created_transitions)
        self.buffer.clear()

    async def retrieve(self, query):
        anchors = await self.normal_facts(query,self.config.anchor_topk)
        rows,_,_ = await self.graph.driver.execute_query(
            'MATCH (s:Entity)-[e:TRANSITION]->(t:Entity) WHERE e.namespace=$namespace '
            'RETURN e.uuid AS id,s.uuid AS source,t.uuid AS target,e.fact AS fact,'
            'e.fact_embedding AS embedding,e.valid_at AS valid_at,e.kind AS kind', namespace=self.namespace)
        transitions, paths = expand(await self.vector(query),anchors,[dict(r) for r in rows],self.config)
        return anchors,transitions,paths

    async def close(self):
        try:
            if self.gids:
                await self.graph.driver.execute_query(
                    'MATCH (n) WHERE n.group_id IN $gids DETACH DELETE n',gids=self.gids)
        finally:
            await self.graph.close()
            await self.embedder.client.close()
            await self.client.client.close()
