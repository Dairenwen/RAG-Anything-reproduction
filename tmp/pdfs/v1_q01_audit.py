"""Offline evidence audit. Reads the KB and saved Q01; never calls a model."""
from pathlib import Path
import base64, collections, hashlib, json, re, xml.etree.ElementTree as ET, zlib
from lightrag.utils import TiktokenTokenizer
from lightrag.prompt import PROMPTS

ROOT = Path('/Users/drw/PycharmProjects/RAG-Anything-reproduction')
KB = ROOT / 'storage/handson-qwen4096-v1'
RUN = ROOT / 'runs/20261011_110345_7173b7cc'
OUT = ROOT / 'tmp/pdfs/v1-q01'
OUT.mkdir(parents=True, exist_ok=True)
def load(p): return json.loads(p.read_text())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
chunks = load(KB/'kv_store_text_chunks.json')
vectors = {k:load(KB/f'vdb_{k}.json') for k in ['entities','relationships','chunks']}
q = load(RUN/'cv1-q01-result.json')
manifest = load(KB/'pilot_manifest.json')
run_manifest = load(RUN/'run-manifest.json')
ns={'g':'http://graphml.graphdrawing.org/xmlns'}
tree=ET.parse(KB/'graph_chunk_entity_relation.graphml').getroot()
keys={k.attrib['id']:k.attrib['attr.name'] for k in tree.findall('g:key',ns)}
g=tree.find('g:graph',ns)
def attrs(x): return {keys[d.attrib['key']]:d.text or '' for d in x.findall('g:data',ns)}
nodes={n.attrib['id']:attrs(n) for n in g.findall('g:node',ns)}
edges=[{'src_id':e.attrib['source'],'tgt_id':e.attrib['target'],**attrs(e)} for e in g.findall('g:edge',ns)]
edge_map={tuple(sorted((e['src_id'],e['tgt_id']))):e for e in edges}
degree=collections.Counter()
for e in edges: degree.update((e['src_id'],e['tgt_id']))
ev={k:[e for e in q['events'] if e['kind']==k] for k in ['text_request','rerank_request','actual_retrieval_prompt','vision_request']}
content_map={sha(c['content']):cid for cid,c in chunks.items()}
ranks=[]
for i,r in enumerate(ev['rerank_request'][-1]['ranked_candidates'],1):
 cid=content_map[r['content_sha256']]
 c=chunks[cid]
 ranks.append({'rank':i,'chunk_id':cid,'type':c.get('original_type','text'),'page':c['page_idx']+1,'page_end':c.get('page_idx_end',c['page_idx'])+1,'modal_entity_name':c.get('modal_entity_name',''),'tokens':c['tokens'],**r})
final_ids=[c['chunk_id'] for c in q['retrieval']['data']['chunks']]
for r in ranks: r['in_final']=r['chunk_id'] in final_ids
prompt=ev['actual_retrieval_prompt'][0]['prompt']
prompt_json=[]
for line in prompt.splitlines():
 try:
  v=json.loads(line)
  if isinstance(v,dict): prompt_json.append(v)
 except (ValueError,TypeError): pass
prompt_chunks=[v for v in prompt_json if 'content' in v and 'reference_id' in v]
assert [x['content'] for x in prompt_chunks]==[x['content'] for x in q['retrieval']['data']['chunks']]
image_pattern=r'Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))'
image_paths=re.findall(image_pattern,prompt,re.I)
payloads=[x['image_url']['payload_sha256'] for m in ev['vision_request'][0]['messages'] if isinstance(m['content'],list) for x in m['content'] if x.get('type')=='image_url']
images=[]
for i,path in enumerate(image_paths):
 p=Path(path); data=p.read_bytes(); mime='image/jpeg' if p.suffix.lower() in ['.jpg','.jpeg'] else 'image/'+p.suffix[1:]
 payload='data:'+mime+';base64,'+base64.b64encode(data).decode()
 cid=next(cid for cid in final_ids if path in chunks[cid]['content'])
 images.append({'index':i+1,'stored_path':path,'actual_path':str(p.resolve()),'chunk_id':cid,'type':chunks[cid]['original_type'],'page':chunks[cid]['page_idx']+1,'anchor':chunks[cid]['modal_entity_name'],'payload_sha256':sha(payload),'payload_matches':sha(payload)==payloads[i]})
assert all(x['payload_matches'] for x in images)
assert len(images)==8
vdb_ent={v['entity_name']:v for v in vectors['entities']['data']}
vdb_rel={tuple(sorted((v['src_id'],v['tgt_id']))):v for v in vectors['relationships']['data']}
anchor_names={c['modal_entity_name'] for c in chunks.values() if c.get('is_multimodal')}
standard=[]; summary=[]; other=[]
for name,v in vdb_ent.items():
 n=nodes[name]
 if v['content']==name+'\n'+n['description']: standard.append(name)
 elif v['content']==n['description']: summary.append(name)
 else: other.append(name)
rel_content_bad=[]
for pair,v in vdb_rel.items():
 e=edge_map[pair]
 expected=f"{e['keywords']}\t{v['src_id']}\n{v['tgt_id']}\n{e['description']}"
 if v['content']!=expected: rel_content_bad.append(pair)
mixed_nodes=[];mixed_edges=[]
for name,n in nodes.items():
 types={chunks[cid].get('original_type','text') for cid in n['source_id'].split('<SEP>') if cid in chunks}
 if 'text' in types and len(types)>1: mixed_nodes.append(name)
for pair,e in edge_map.items():
 types={chunks[cid].get('original_type','text') for cid in e['source_id'].split('<SEP>') if cid in chunks}
 if 'text' in types and len(types)>1: mixed_edges.append(pair)
miss_sources=[cid for x in list(nodes.values())+edges for cid in x['source_id'].split('<SEP>') if cid not in chunks]
formula_ids={}
for kind,data in vectors.items():
 bad=[]
 for r in data['data']:
  if kind=='entities': expected='ent-'+hashlib.md5(r['entity_name'].encode()).hexdigest()
  elif kind=='relationships': expected='rel-'+hashlib.md5((r['src_id']+r['tgt_id']).encode()).hexdigest()
  else: expected='chunk-'+hashlib.md5(r['content'].encode()).hexdigest()
  if r['__id__']!=expected: bad.append(r['__id__'])
 formula_ids[kind]={'verified':len(data['data'])-len(bad),'bad':bad,'first_vector_dimensions':len(zlib.decompress(base64.b64decode(data['data'][0]['vector'])))//2}
entity_chunks=load(KB/'kv_store_entity_chunks.json');relation_chunks=load(KB/'kv_store_relation_chunks.json')
fullent=load(KB/'kv_store_full_entities.json');fullrel=load(KB/'kv_store_full_relations.json')
main_chunk='chunk-2d4727f37ea1b30518f13dc89ffc7ea1'
anchor='RAG-Anything Framework Architecture Diagram (image)'
local_entity=q['retrieval']['data']['entities'][0]['entity_name']
example_pair=tuple(sorted(('RAG-Anything','bge-reranker-v2-m3')))
sample_names=[local_entity,anchor,'RAG-Anything','bge-reranker-v2-m3','Text Encoder','Embedding Table']
sample_nodes={n:{'graph':nodes[n],'vector':{k:v for k,v in vdb_ent[n].items() if k!='vector'},'degree':degree[n]} for n in sample_names if n in nodes}
sample_edges=[e for e in edges if tuple(sorted((e['src_id'],e['tgt_id']))) in [example_pair,tuple(sorted((anchor,'RAG-Anything'))),tuple(sorted((local_entity,'Text Encoder')))]]
primary_text=chunks[main_chunk]['content']
pos=primary_text.find('Implementation Details')
if pos<0:pos=primary_text.find('text-embedding')
out={
 'question':q['question'],'answer':q['answer'],'historical_diagnostic':q['diagnostic_retrieval_is_separate_from_generation'],
 'versions':manifest.get('packages'),'index_identity':manifest['index_identity'],'query_settings':run_manifest['settings'],
 'metadata':q['retrieval']['metadata'],'keyword_response':json.loads(ev['text_request'][-1]['response']),
 'counts':{'nodes':len(nodes),'edges':len(edges),'anchors':len(anchor_names),'belongs_to_edges':sum('belongs_to' in e['keywords'] for e in edges),'chunks':len(chunks),'chunk_types':dict(collections.Counter(c.get('original_type','text') for c in chunks.values())),'final_types':dict(collections.Counter(chunks[c].get('original_type','text') for c in final_ids)),'mixed_nodes':len(mixed_nodes),'mixed_edges':len(mixed_edges),'entity_source_store':len(entity_chunks),'relation_source_store':len(relation_chunks),'full_entity_names':len(next(iter(fullent.values()))['entity_names']),'full_relation_pairs':len(next(iter(fullrel.values()))['relation_pairs'])},
 'checks':{'id_formulas':formula_ids,'entity_standard_content':len(standard),'entity_summary_only_content':len(summary),'entity_content_other':other,'relationship_content_bad':rel_content_bad,'missing_graph_source_chunks':miss_sources,'same_prompt_chunks':True,'same_keyword_twice':ev['text_request'][0]['response']==ev['text_request'][1]['response'],'same_rerank_order_twice':[x['content_sha256'] for x in ev['rerank_request'][0]['ranked_candidates']]==[x['content_sha256'] for x in ev['rerank_request'][1]['ranked_candidates']]},
 'sample_nodes':sample_nodes,'sample_edges':sample_edges,'sample_relationship_vector':{k:v for k,v in vdb_rel[example_pair].items() if k!='vector'},
 'main_chunk_id':main_chunk,'main_chunk':chunks[main_chunk],'main_excerpt':primary_text[max(0,pos-200):pos+1800],
 'architecture_chunk_id':sample_nodes[anchor]['graph']['source_id'],'architecture_chunk':chunks[sample_nodes[anchor]['graph']['source_id']],
 'ranked_candidates':ranks,'final_chunk_ids':final_ids,'images':images,
 'retrieved_entities':q['retrieval']['data']['entities'],'retrieved_relations':q['retrieval']['data']['relationships'],
 'result_fields':list(q),'raw_prompt_characters':len(prompt),'events':{k:len(v) for k,v in ev.items()},
 'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [KB/'graph_chunk_entity_relation.graphml',KB/'kv_store_text_chunks.json',KB/'pilot_manifest.json',RUN/'cv1-q01-result.json',ROOT/'reproduce.py']}
}
tokenizer=TiktokenTokenizer('gpt-4o-mini')
entities_ctx=[x for x in prompt_json if 'entity' in x]
relations_ctx=[x for x in prompt_json if 'entity1' in x]
kg_without_chunks=PROMPTS['kg_query_context'].format(
 entities_str='\n'.join(json.dumps(x,ensure_ascii=False) for x in entities_ctx),
 relations_str='\n'.join(json.dumps(x,ensure_ascii=False) for x in relations_ctx),
 text_chunks_str='',reference_list_str='')
sys_without_context=PROMPTS['rag_response'].format(context_data='',response_type='OneSentence',user_prompt='')
costs={'system_prompt_overhead':len(tokenizer.encode(sys_without_context)),
 'graph_context':len(tokenizer.encode(kg_without_chunks)),
 'query':len(tokenizer.encode(q['question'])),'reserved':200,'max_total_tokens':24000}
costs['available_chunk_tokens']=24000-sum(costs[k] for k in ['system_prompt_overhead','graph_context','query','reserved'])
running=0;kept=[]
for r in ranks[:20]:
 cid=r['chunk_id']; c=chunks[cid]
 doc={'content':c['content'],'file_path':c['file_path'],'chunk_id':cid,'rerank_score':r['relevance_score']}
 cost=len(tokenizer.encode(json.dumps(doc,ensure_ascii=False)))
 running+=cost
 r['budget_tokens']=cost;r['cumulative_budget_tokens']=running
 if running<=costs['available_chunk_tokens']:kept.append(cid)
 else:break
assert kept==final_ids,(len(kept),len(final_ids),costs)
out['token_budget_replay']=costs|{'verified_final_chunk_ids':True,'kept_chunk_tokens':sum(r['budget_tokens'] for r in ranks[:15]),'first_rejected_rank':16,'first_rejected_cumulative_tokens':ranks[15]['cumulative_budget_tokens'],'raw_prompt_tokens':len(tokenizer.encode(prompt))}
(OUT/'evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:out[k] for k in ['counts','checks','token_budget_replay']},ensure_ascii=False,indent=2))
