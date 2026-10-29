"""One command: build memory, answer questions, report accuracy, clean run data."""
import argparse
import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from .config import paper_config

ROOT = Path(__file__).resolve().parents[2]

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',choices=['32k','128k'],default='32k')
    p.add_argument('--backbone',choices=['qwen3-8b','qwen3.5-27b'],default='qwen3-8b')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/personamem')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    p.add_argument('--demo',action='store_true')
    p.add_argument('--max-users',type=int)
    p.add_argument('--model',help='Served model ID; defaults to backbone name')
    p.add_argument('--anchor-topk',type=int)
    p.add_argument('--gamma',type=float)
    p.add_argument('--continuation-threshold',type=float)
    p.add_argument('--drop-tolerance',type=float)
    p.add_argument('--context-token-budget',type=int)
    p.add_argument('--graphiti-invalidation-limit',type=int)
    p.add_argument('--graphiti-node-dedupe-candidate-limit',type=int)
    return p

def required(name):
    value=os.environ.get(name)
    if not value:
        raise ValueError(f'Set {name} in .env (see .env.example)')
    return value

async def run(args, config, settings, out):
    # Cost settings must be set before importing Graphiti modules.
    from .memory import Memory
    from .retrieval import assemble
    import tiktoken
    from openai import AsyncOpenAI
    import httpx
    tokenizer=tiktoken.get_encoding('cl100k_base')
    data=ROOT/'examples/demo' if args.demo else args.data_dir/args.subset
    histories={h['user_id']:h['messages'] for h in map(json.loads,(data/'histories.jsonl').read_text(encoding='utf-8').splitlines())}
    by_user=defaultdict(list)
    for question in map(json.loads,(data/'questions.jsonl').read_text(encoding='utf-8').splitlines()):
        by_user[question['user_id']].append(question)
    if args.max_users is not None and args.max_users<1:
        raise ValueError('--max-users must be positive')
    users=sorted(by_user)
    if args.max_users:
        users=users[:args.max_users]
    results=[]
    total_transitions=0
    async with AsyncOpenAI(base_url=settings['llm_url'],api_key=settings['api_key'],
            http_client=httpx.AsyncClient(trust_env=settings['trust_env'],timeout=180)) as qa:
        for uid in users:
            memory=Memory(config,'tm_'+uuid4().hex,settings)
            try:
                await memory.initialize()
                cursor=0
                for question in sorted(by_user[uid],key=lambda q:(q['history_end'],q['id'])):
                    end=question['history_end']
                    messages=histories[uid]
                    if not 0<=end<len(messages):
                        raise ValueError('Invalid question history boundary')
                    while cursor<=end:
                        await memory.ingest(messages[cursor],cursor)
                        cursor+=1
                    anchors, transitions, paths=await memory.retrieve(question['query'])
                    context=assemble(anchors,transitions,tokenizer,config.context_token_budget)
                    prompt=(f"Relevant historical information:\n{context}\n\nQuestion: {question['query']}\n"
                        +'\n'.join(question['options'])
                        +'\nFind the most appropriate model response and only give your final answer as '
                        +'<final_answer>(a)</final_answer>, <final_answer>(b)</final_answer>, '
                        +'<final_answer>(c)</final_answer>, or <final_answer>(d)</final_answer>. Do not provide any explanation.')
                    response=await qa.chat.completions.create(model=settings['model'],temperature=0,
                        messages=[{'role':'user','content':prompt}],max_tokens=256,
                        extra_body={'chat_template_kwargs':{'enable_thinking':False}})
                    answer=response.choices[0].message.content or ''
                    match=re.search(r'<final_answer>\s*\(?([a-d])\)?\s*</final_answer>',answer,re.I)
                    prediction=match.group(1).lower() if match else None
                    result={'question_id':question['id'],'user_id':uid,'category':question['category'],
                        'gold':question['answer'],'prediction':prediction,'correct':prediction==question['answer'],
                        'response':answer,'anchors':anchors,
                        'transitions':[{k:v for k,v in t.items() if k!='embedding'} for t in transitions],
                        'paths':paths,'context_tokens':len(tokenizer.encode(context))}
                    results.append(result)
                    with (out/'predictions.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(result)+'\n')
                    logging.info('question=%s prediction=%s gold=%s transitions=%d',question['id'],prediction,question['answer'],len(transitions))
                total_transitions+=memory.created_transitions
                with (out/'construction.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'user_id':uid,'events':memory.events,
                        'stored_transitions':memory.created_transitions})+'\n')
            finally:
                await memory.close()
    grouped=defaultdict(list)
    for result in results:
        grouped[result['category']].append(result['correct'])
    metrics={'questions':len(results),'accuracy':sum(r['correct'] for r in results)/len(results) if results else None,
        'by_category':{k:{'count':len(v),'accuracy':sum(v)/len(v)} for k,v in grouped.items()},
        'stored_transitions':total_transitions,'demo':args.demo}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metrics,indent=2))
    if args.demo and (total_transitions==0 or not any(r['transitions'] for r in results)):
        raise RuntimeError('Demo did not exercise grounded transition retrieval; inspect construction and predictions')

def main():
    load_dotenv(ROOT/'.env',override=False)
    args=parser().parse_args()
    config=paper_config(args.subset,args.backbone)
    if args.demo:
        # Synthetic integration demo, explicitly distinct from paper settings.
        config.macro_threshold=0.25
        config.micro_threshold=0.95
        config.maturity_messages=3
        config.buffer_size=1
        config.summary_interval=3
    for key in config.as_dict():
        value=getattr(args,key,None)
        if value is not None:
            setattr(config,key,value)
    config.validate()
    for key,value in config.as_dict().items():
        if key.startswith('graphiti_'):
            os.environ[key.upper()]=str(value)
    os.environ['GRAPHITI_TELEMETRY_ENABLED']='false'
    settings={'llm_url':required('LLM_BASE_URL'),'embedding_url':required('EMBEDDING_BASE_URL'),
        'model':args.model or args.backbone,'api_key':os.getenv('OPENAI_API_KEY','EMPTY'),
        'embedding_model':os.getenv('EMBEDDING_MODEL','bge-m3'),
        'embedding_api_key':os.getenv('EMBEDDING_API_KEY','EMPTY'),
        'neo4j_uri':required('NEO4J_URI'),'neo4j_user':os.getenv('NEO4J_USER','neo4j'),
        'neo4j_password':required('NEO4J_PASSWORD'),
        'trust_env':os.getenv('TRANSMEM_TRUST_ENV','false').lower()=='true'}
    out=args.output_dir/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log', encoding='utf-8')])
    (out/'config.json').write_text(json.dumps(config.as_dict(),indent=2)+'\n', encoding='utf-8')
    print('Output:',out,flush=True)
    asyncio.run(run(args,config,settings,out))

if __name__=='__main__':
    main()
