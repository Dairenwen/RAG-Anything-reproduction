"""Build a self-contained HTML manual and a verified PDF from offline evidence."""
from pathlib import Path
import base64, html, json, math, re, textwrap
from xml.sax.saxutils import escape
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether

ROOT=Path('/Users/drw/PycharmProjects/RAG-Anything-reproduction')
WORK=ROOT/'tmp/pdfs/v1-q01'; DEST=ROOT/'reports'
DEST.mkdir(exist_ok=True)
D=json.loads((WORK/'evidence.json').read_text())
KB=ROOT/'storage/handson-qwen4096-v1'; SP=ROOT/'.venv/lib/python3.12/site-packages'
chunks=json.loads((KB/'kv_store_text_chunks.json').read_text())
aliases={cid:f'C{i:02d}' for i,cid in enumerate(chunks,1)}
def cid_alias(cid): return aliases[cid]
def sources(s): return ', '.join(aliases.get(x,x) for x in s.split('<SEP>'))
F_REG='/System/Library/Fonts/STHeiti Light.ttc';F_BOLD='/System/Library/Fonts/STHeiti Medium.ttc'
pdfmetrics.registerFont(TTFont('CN',F_REG));pdfmetrics.registerFont(TTFont('CNB',F_BOLD))
pdfmetrics.registerFont(TTFont('Code','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
FONT=ImageFont.truetype(F_REG,29);FONT_B=ImageFont.truetype(F_BOLD,32)

def wrap_pixels(text,width,font):
 lines=[]
 for raw in text.split('\n'):
  line=''
  for ch in raw:
   if font.getlength(line+ch)>width and line: lines.append(line);line=ch
   else:line+=ch
  lines.append(line)
 return lines

def diagram(name,boxes,links,title,size=(1500,1100)):
 im=Image.new('RGB',size,'white'); dr=ImageDraw.Draw(im)
 dr.text((42,24),title,font=FONT_B,fill='#162b45')
 positions={key:(x,y,w,h) for key,x,y,w,h,label,color in boxes}
 def center(key):
  x,y,w,h=positions[key];return (x+w/2,y+h/2)
 for a,b,label in links:
  x1,y1=center(a);x2,y2=center(b);pa=positions[a];pb=positions[b]
  dx=x2-x1;dy=y2-y1
  if abs(dy)>=abs(dx)*0.7:
   start=(x1,pa[1]+pa[3] if dy>0 else pa[1]);end=(x2,pb[1] if dy>0 else pb[1]+pb[3])
  else:
   start=(pa[0]+pa[2] if dx>0 else pa[0],y1);end=(pb[0] if dx>0 else pb[0]+pb[2],y2)
  dr.line([start,end],fill='#60788d',width=4)
  ang=math.atan2(end[1]-start[1],end[0]-start[0]);l=16
  dr.polygon([end,(end[0]-l*math.cos(ang-.45),end[1]-l*math.sin(ang-.45)),(end[0]-l*math.cos(ang+.45),end[1]-l*math.sin(ang+.45))],fill='#60788d')
  if label:
   mid=((start[0]+end[0])/2,(start[1]+end[1])/2)
   ft=ImageFont.truetype(F_REG,22); tw=ft.getlength(label)
   dr.rectangle((mid[0]-tw/2-4,mid[1]-17,mid[0]+tw/2+4,mid[1]+14),fill='white')
   dr.text((mid[0]-tw/2,mid[1]-14),label,font=ft,fill='#40566b')
 for key,x,y,w,h,label,color in boxes:
  dr.rounded_rectangle((x,y,x+w,y+h),radius=17,fill=color,outline='#648298',width=2)
  lines=wrap_pixels(label,w-28,FONT); leading=37; yy=y+(h-leading*len(lines))/2
  for line in lines:
   dr.text((x+(w-FONT.getlength(line))/2,yy),line,font=FONT,fill='#152535');yy+=leading
 path=WORK/f'{name}.png';im.save(path);return path

storage_fig=diagram('storage',[
 ('t',40,110,650,130,'正文解析文本\n10 个文本 chunk','#edf3fa'),
 ('m',810,110,650,130,'图片 表格 公式\n27 个多模态单元','#e9f3ef'),
 ('proxy',810,290,650,150,'类型处理器生成描述与摘要\n模板保留原图路径或 LaTeX','#e9f3ef'),
 ('chunk',40,500,650,170,'统一 chunk 池 共 37 块\nKV 保存内容  chunk VDB 保存向量','#edf3fa'),
 ('ext',810,500,650,170,'LightRAG 抽取实体与关系\n多模态另建锚点与 belongs_to','#e9f3ef'),
 ('merge',410,740,680,130,'按实体名与无序端点对合并\n来源 ID 合并  描述合并或摘要','#fff3df'),
 ('graph',40,950,450,130,'融合 GraphML\n563 节点 998 边','#edf3fa'),
 ('ev',525,950,450,130,'实体向量索引\n563 条记录','#edf3fa'),
 ('rv',1010,950,450,130,'关系向量索引\n998 条记录','#edf3fa'),
 ],[('t','chunk',''),('m','proxy',''),('proxy','chunk','模板文本'),('chunk','ext','content'),('proxy','ext','锚点摘要'),('ext','merge',''),('merge','graph',''),('merge','ev','编码实体内容'),('merge','rv','编码关系内容')], '图 1 建库输出按知识对象组织',size=(1500,1140))

retrieval_fig=diagram('retrieval',[
 ('q',390,100,720,130,'Q01 用户问题\nLightRAG 调用 DeepSeek 提取关键词','#edf3fa'),
 ('enc',180,300,1140,150,'qwen3-embedding 一次批量编码\n原问题  低层关键词字符串  高层关键词字符串','#e9f3ef'),
 ('ll',40,540,450,150,'低层向量 → 实体 VDB\ntop_k=20\n读取节点及相邻关系','#edf3fa'),
 ('hl',525,540,450,150,'高层向量 → 关系 VDB\ntop_k=20\n读取边及两端实体','#edf3fa'),
 ('qq',1010,540,450,150,'原问题向量 → chunk VDB\nchunk_top_k=20\n直接获得候选块内容','#edf3fa'),
 ('gm',150,790,820,150,'local 与 global 轮流合并并去重\nQ01 合并后 48 实体 48 关系\n实体与关系分别做 token 裁剪','#fff3df'),
 ('next',1010,790,450,150,'直接向量 chunk 候选\n送入后续统一块池','#edf3fa'),
 ],[('q','enc',''),('enc','ll','低层向量'),('enc','hl','高层向量'),('enc','qq','问题向量'),('ll','gm',''),('hl','gm',''),('qq','next','')], '图 4 Q01 召回与图谱扩展',size=(1500,1000))

synthesis_fig=diagram('synthesis',[
 ('g',40,110,690,160,'保留的实体与关系\nsource_id → 来源 chunk ID\n默认 VECTOR 选择  KV 读取内容','#edf3fa'),
 ('v',820,110,640,160,'直接向量召回的 chunk\n来自 vdb_chunks.json','#edf3fa'),
 ('mix',350,350,800,150,'三路块轮流合并并去重\nQ01 得到 35 个候选 chunk\n预先计算 chunk 预算 17796 token','#fff3df'),
 ('r',40,560,660,160,'BGE 对 问题＋块文本 评分\n35 块 → 排序取前 20\nmin_rerank_score=0','#e9f3ef'),
 ('b',820,560,640,160,'按 17796 token 预算裁剪\n保留前 15 块\n第 16 块令累计预算超限','#fff3df'),
 ('p',40,820,660,160,'文字证据 P(q)\n48 实体 48 关系 15 块\n同一 PDF 对应 reference_id=1','#edf3fa'),
 ('img',820,820,640,160,'匹配最终提示词的 Image Path\n读取 6 图片＋2 表格原图\n按位置插入多模态 messages','#e9f3ef'),
 ('ans',350,1060,800,120,'DeepSeek 视觉模型生成答案\ntext-embedding-3-large  3072 维','#edf3fa'),
 ],[('g','mix',''),('v','mix',''),('mix','r',''),('r','b',''),('b','p',''),('p','img',''),('img','ans','图文证据')], '图 5 Q01 来源回查到答案生成',size=(1500,1250))

graph_fig=diagram('graph-sample',[
 ('r',40,130,610,155,'RAG-Anything\n普通融合实体  来源横跨正文与多模态','#edf3fa'),
 ('b',850,130,610,155,'bge-reranker-v2-m3\n普通实体  来源 C05 正文','#edf3fa'),
 ('a',40,510,610,200,'架构图锚点实体\nRAG-Anything Framework\nArchitecture Diagram (image)\n来源 C12 图片描述块','#e9f3ef'),
 ('c5',850,510,610,200,'C05 原论文第 3.1 节正文\nchunk-2d4727f37ea1b30518f13dc89ffc7ea1\n含模型名与 3072 维','#fff3df'),
 ('c12',40,900,610,150,'C12 多模态 chunk\n含 Image Path 与增强描述','#fff3df'),
 ('img',850,900,610,150,'parsed 中的架构图文件\n像素留到答案生成阶段读取','#e9f3ef'),
 ],[('r','b','语义关系 weight=1'),('r','a','belongs_to weight=10'),('b','c5','source_id'),('a','c12','source_id'),('c12','img','Image Path')], '图 2 真实记录组成的字段连接示例',size=(1500,1120))

blocks=[];sections=[]
def h(title):
 n=len(sections)+1;sid=f's{n}';sections.append((sid,title));blocks.append(('h',title,sid))
def p(s):blocks.append(('p',s))
def bullets(xs):blocks.append(('list',xs))
def table(headers,rows,widths=None):blocks.append(('table',headers,rows,widths))
def code(s):blocks.append(('code',s.strip()))
def fig(path,caption,height=None):blocks.append(('fig',Path(path),caption,height))
def page():blocks.append(('page',))
def note(s):blocks.append(('note',s))
def j(obj):return json.dumps(obj,ensure_ascii=False,indent=2)

TITLE='RAG Anything v1 知识库与第一题检索流程详解'
INTRO='GraphML 保存融合后的实体节点与关系边；三个向量索引分别面向实体、关系和 chunk，均接收正文与多模态文字表示。查询先通过向量索引定位知识对象，再按名字或端点读取图谱，通过 source_id 回查 chunk，最后从最终提示词的 Image Path 恢复原图。本手册把这条链路对应到 v1 实际记录、字段、官方函数与 Q01 运行证据。'

h('1 六个问题的直接回答')
p(INTRO)
table(['问题','回答'],[
 ('1 实体 关系 边','实体是 node；关系就是连接两个实体的 edge。关系描述、关键词、weight 和 source_id 都是边属性，没有再建立第三类“关系节点”。图为无向普通图，同一实体对的关系片段会合并到同一条边。'),
 ('2 向量生成与融合','实体、关系在建库抽取和合并后调用各自 VDB 的 upsert，底层编码 content。两个 VDB 按对象类型分库，不按正文或多模态来源分库；多模态描述还要抽取实体关系、建立锚点和 belongs_to，再与正文知识融合。'),
 ('3 内容回查','把节点或边的 source_id 按 <SEP> 拆成 chunk ID，再按 ID 读取 kv_store_text_chunks.json 的 content。这里保存的内容可为正文，也可为带图像路径、表格结构或 LaTeX 的多模态模板文本。'),
 ('4 谁调用谁','LightRAG 调用你配置的 DeepSeek 回调生成关键词，DeepSeek 不负责调用 LightRAG。这是 LightRAG 内部机制。RAG-Anything 负责外层多模态入库和查询后原图恢复。'),
 ('5 关键词匹配什么','低层关键词字符串的向量匹配实体索引；高层关键词字符串的向量匹配关系索引；原问题向量匹配 chunk 索引。先做向量相似度搜索，再按命中记录读取图谱并扩展。'),
 ('6 第一题实证','Q01 找到 48 实体和 48 关系，合并 35 个候选块，BGE 取前 20，预算最终保留 15 块，恢复 8 张图片。最高重排块包含第 3.1 节的 text-embedding-3-large 和 3072 维。')],[90,414])
note('证据版本必须分开：Q01 文件来自 2026 年 10 月 11 日的历史完整测试，当时存在额外诊断检索；当前 reproduce.py 已移除该步骤。本文用历史记录核验实际证据，用当前代码解释单次正常查询，不把旧耗时与调用数当成新代码实测结果。')

h('2 版本 模型与证据位置')
p('项目根目录为 '+str(ROOT)+'。下文的文件简称均相对于此目录。C01 到 C37 只是本手册给 37 个 chunk 添加的阅读编号，官方库里没有这些简称；附录给出完整 ID。')
table(['范围','模型或版本','职责'],[
 ('RAG-Anything','固定提交 8664e8b318a3ed651fe62dfcfddfb1f1d633d5be','多模态解析 入库 原图恢复'),
 ('LightRAG','lightrag-hku 1.4.16','实体关系抽取 合并 图谱与向量检索'),
 ('MinerU','3.4.1','从 PDF 解析文字 图片 表格 公式'),
 ('v1 建库文字模型',D['index_identity']['text_model'],'实体关系抽取 描述合并等文字任务'),
 ('v1 建库视觉模型',D['index_identity']['vision_model'],'图片语义描述等视觉任务'),
 ('建库及查询编码器','qwen3-embedding:latest 4096 维','将文字内容编码为数值向量'),
 ('Q01 查询文字及视觉模型','deepseek-flash','提取关键词与最终图文回答'),
 ('Q01 重排器','BAAI/bge-reranker-v2-m3','对问题与候选块文本评分')],[93,224,187])
p('v1 的建库身份见 storage/handson-qwen4096-v1/pilot_manifest.json 的 index_identity。Q01 的模型和参数见 runs/20261011_110345_7173b7cc/run-manifest.json 的 settings。原论文第 3.1 节写的是 text-embedding-3-large、3072 维，不能用本地编码器的 4096 维代替论文实验设置。[F01 F02 F03]')
table(['Q01 参数','实际值'],[(k,str(D['query_settings'][k])) for k in ['top_k','chunk_top_k','max_entity_tokens','max_relation_tokens','max_total_tokens','enable_rerank','rerank_max_length','rerank_batch_size']]+[('mode','mix'),('vlm_requested','true'),('cosine_better_than_threshold','0.2 项目配置'),('min_rerank_score','0.0 项目配置')],[240,264])
p('本次文档审计没有调用 DeepSeek、Qwen、BGE，也没有重新建库。数值来自已保存记录，token 裁剪用同版本官方 tokenizer 与模板离线重算。对历史记录没有保存的向量相似度、分支原始命中列表，不伪造数值。')

h('3 库文件分工与连接字段')
p('文件之间依靠对象名称、端点对和 chunk ID 连接。GraphML 负责结构与属性，VDB 负责向量召回，KV 负责按 ID 读取内容。原图文件仍在 parsed 中；向量 JSON 与 GraphML 都不代替原图字节。下面分别给出建库关系图和文件清单。')
fig(storage_fig,'图 1 正文与多模态知识进入同一套对象索引。向量数据库编码的是文字 content；GraphML 不保存这些向量。',380)
table(['文件','实际数量','内容与检索作用'],[
 ('graph_chunk_entity_relation.graphml','563 节点 998 边','融合后的实体与关系；查询回读对象属性及邻接结构'),
 ('vdb_entities.json','563 记录','实体名称 描述内容 来源 ID 数值向量'),
 ('vdb_relationships.json','998 记录','关系端点 关键词 描述内容 来源 ID 数值向量'),
 ('vdb_chunks.json','37 记录','正文及多模态模板文本的向量和内容'),
 ('kv_store_text_chunks.json','37 记录','以 chunk ID 为键的完整块内容和元数据'),
 ('kv_store_entity_chunks.json','536 记录','实体到完整来源 chunk ID 列表的维护索引'),
 ('kv_store_relation_chunks.json','998 记录','无序端点对到来源 chunk ID 列表的维护索引'),
 ('kv_store_full_docs.json','1 文档','原文档正文文本；正常混合检索主要取 chunk，不直接整篇读取'),
 ('kv_store_full_entities.json','320 名称','按文档保存实体清单，不是完整图谱的替代品'),
 ('kv_store_full_relations.json','786 端点对','按文档保存关系清单，不是检索主索引'),
 ('kv_store_doc_status.json','1 文档','处理状态 chunk 列表及多模态完成标记'),
 ('kv_store_llm_response_cache.json','建库抽取及摘要缓存','支持建库缓存和维护；当前配置关闭普通查询 LLM 缓存'),
 ('pilot_manifest.json','项目自定义','建库身份 版本 文件摘要等审计信息'),
 ('parsed/.../auto/images/*.jpg','原始视觉切块','真正的图片字节；通过 chunk content 中的 Image Path 找到')],[177,92,235])
p('实体来源维护索引比图节点少 27 条，正好对应 27 个多模态锚点：锚点由 RAG-Anything 单独写入图谱与实体 VDB，已有 source_id，未必走普通实体来源索引维护路径。因此回查应以命中图谱对象的 source_id 为准。[C02 C04 C06]')
p('full_entities 与 full_relations 当前只有部分清单，而 GraphML 与 VDB 具有完整对象。官方合并函数按本次处理结果写入文档清单，这类清单不能作为判断全图节点是否存在的唯一证据。正常查询走 GraphML、对象 VDB 和 text_chunks，不能用这两份清单替代。[C06]')

h('4 GraphML 的实体 关系与边')
p('XML 的 graph 声明 edgedefault="undirected"，底层 NetworkXStorage 使用 nx.Graph。563 个 node 代表实体，998 个 edge 代表实体对之间的关系。没有“节点、关系、边”三种彼此独立的数据集合，关系就存储在边上。[F04 C01]')
table(['实体字段','含义','例子'],[
 ('node 的 id','图内查找键，通常就是实体名','RAG-Anything'),
 ('entity_id','保存到属性中的实体名；不是 ent- 开头向量 ID','RAG-Anything'),
 ('entity_type','抽取或锚点指定的类型，可能为 UNKNOWN','concept image table equation'),
 ('description','合并后的描述或多模态锚点摘要','模型生成文字，不是原文真实性保证'),
 ('source_id','支撑该实体的来源块，多个 ID 以 <SEP> 连接','chunk-...<SEP>chunk-...'),
 ('file_path','来源文档路径，多个文档也可合并','data/RAG-Anything_arXiv_2025.pdf'),
 ('created_at','写入或更新时的 Unix 时间','不应理解为文档发布时间'),
 ('truncate','来源列表限制的记录标记；v1 中为空','不是描述文本自动被裁剪的长度')],[94,253,157])
table(['关系字段','含义'],[
 ('edge source 与 target','两端实体名。图是无向的，XML 顺序不构成图遍历方向'),
 ('description','关系的自然语言说明；可合并多个来源的关系片段'),
 ('keywords','关系语义标签，以逗号分隔；不是当前用户问题提取的高层关键词'),
 ('weight','关系片段权重合并的结果；官方合并会求和，不是余弦相似度，也不是 BGE 分数'),
 ('source_id file_path','该关系的证据块和来源文档'),
 ('created_at truncate','关系更新时刻与来源列表限制标记'),
 ('rank','GraphML 没有此固定字段。local 检索时动态计算：节点 rank 为度数，边 rank 为两端度数之和')],[130,374])
p('同一实体对只能保留一条普通无向边；多个谓词或多个模态发现的关系，会合并到该边的 description、keywords、weight、source_id。它不是允许每个谓词独立成边的 RDF 三元组库，也不是 MultiDiGraph。belongs_to 虽有方向含义，图中仍以无向边存储，方向含义主要留在描述文字中。')
edge=D['sample_edges'][0]
code('<graph edgedefault="undirected">\n  <node id="bge-reranker-v2-m3">\n    <data key="d0">bge-reranker-v2-m3</data>\n    <data key="d1">artifact</data>\n    <!-- d2 description, d3 source_id, d4 file_path 等 -->\n  </node>\n  <edge source="RAG-Anything" target="bge-reranker-v2-m3">\n    <data key="d7">1.0</data>\n    <data key="d8">bge-reranker-v2-m3 is used for reranking in RAG-Anything.</data>\n    <data key="d9">reranking,tool usage</data>\n    <data key="d10">chunk-2d4727f37ea1b30518f13dc89ffc7ea1</data>\n  </edge>\n</graph>')
p('上面的 XML 为真实字段摘录，省略路径和时间字段。d0 到 d13 的实际含义由文件开头的 key 声明指定，不能只看 d 编号猜字段。实体 bge-reranker-v2-m3 的实际类型以库字段为准。')
fig(graph_fig,'图 2 箭头表示字段回查或描述中的语义连接；GraphML 中的两条实体关系实际都是无向边。C05 与 C12 通过同名 RAG-Anything 实体进入融合图。',380)
p('v1 共有 27 个多模态锚点和 406 条 keywords 包含 belongs_to 的边。按 source_id 反查来源类型，有 45 个实体同时由正文和至少一种多模态块支撑，另有 15 条关系同时具有正文和多模态来源。这些数字是实际融合的证据；不是简单把两个文件并排保存。')

h('5 多模态描述如何成为知识图谱')
p('生成图片描述只完成第一阶段，不会自动等于已经建好知识图谱。官方批处理路径还要把描述变成 chunk，建立模态锚点，用 LightRAG 从 chunk content 抽取实体关系，增加归属关系，最后合并到图谱并写向量索引。[C02 C03 C04 C05]')
table(['阶段','实际处理','写入或产生的对象'],[
 ('正文入库','MinerU 正文文本经 LightRAG 按 1200 token 切分，重叠 100；本库 10 块','text_chunks chunk VDB；实体关系抽取结果'),
 ('多模态描述','图片处理器读取图片及上下文，生成增强描述、锚点名称、类型、摘要；表格和公式使用各自处理器','description 与 entity_info'),
 ('模板化','图片含路径 图注 邻近正文与分析；表格含路径 HTML 结构与分析；公式含 LaTeX 与分析','27 个统一格式的 chunk'),
 ('写块','text_chunks 保存完整模板；chunks_vdb 对同一 content 编码','KV 与 chunk 向量记录'),
 ('锚点','单独将 modal_entity_name 及摘要写成图节点和实体向量记录','27 个 image table equation 锚点'),
 ('内容抽取','调用 LightRAG.extract_entities；含初始抽取及配置允许的补充抽取','普通实体与关系的描述 来源块 ID'),
 ('归属关联','把该块中抽取的实体与模态锚点连接，keywords 为 belongs_to,part_of,contained_in，初始 weight=10','多模态特有的归属关系'),
 ('融合写入','LightRAG.merge_nodes_and_edges 将同名节点和相同无序端点对合并','一个 GraphML 实体 VDB 关系 VDB')],[77,245,182])
p('“双图构建”在论文中描述正文语义图与跨模态关联图的互补构建。在当前落地文件中，它们最终融合为同一份 GraphML，以及按实体、关系、chunk 分开的三个 VDB，并没有单独保存 text_graph.graphml 与 multimodal_graph.graphml。')
archimg=Path(D['images'][2]['actual_path'])
fig(archimg,'图 3 原论文架构图，也是 Q01 恢复的第 3 张原图。图中的概念性 Text VDB 与 Multi-modal VDB 不能直接当作本项目两个 VDB 文件的分工。',310)
p('图片描述能够把图像内容送入文字抽取器，但无法保证视觉内容被无损表示。描述漏掉的图例、坐标、空间关系、数值仍可能影响召回；只有命中的块保留到最终提示词并具有有效路径，原图才会进入答案生成。公式块的当前模板没有 Image Path，即使解析阶段保留过截图，本入口也不会自动恢复公式截图。')

h('6 实体与关系向量何时生成 如何融合')
p('向量真正生成的位置在 NanoVectorDBStorage.upsert：取每条待写记录的 content，调用 embedding_func(batch, context="document")，将结果与记录逐条对应。实体和关系的 upsert 由抽取合并完成后的函数发起。Qwen 编码数值向量，不创建图边；图节点边的建立由结构化抽取与合并逻辑完成。[C06 C07 C08]')
table(['索引','送给编码器的 content','v1 实测对应'],[
 ('普通实体','entity_name + "\\n" + 合并后的 description','536 条，逐条与图节点字段匹配'),
 ('多模态锚点实体','entity_info.summary，缺失时使用描述','27 条，当前 content 等于锚点节点 description；这一专门路径不统一加实体名'),
 ('关系','keywords + "\\t" + src_id + "\\n" + tgt_id + "\\n" + description','998 条，逐条与图边及端点一致'),
 ('chunk','完整 chunk.content','37 条，包括 10 正文 15 图片 6 表格 6 公式')],[94,252,158])
p('不存在“vdb_entities 只编码原文、vdb_relationships 只编码多模态描述”的分工。原文块和多模态模板都可产生普通实体与关系；两类来源共同写入各自的对象索引。两个索引都不是只装原始段落向量的数据库。')
code('实体向量 ID = "ent-" + md5(entity_name)\n关系向量 ID = "rel-" + md5(src_id + tgt_id)\n  本版本先按字符串排序两端，再计算这个 ID\nchunk ID = "chunk-" + md5(chunk.content)\n  md5 的输入为 UTF-8 文本；不能改写换行或空格后期待 ID 不变')
p('以上 ID 公式分别核验了 563、998、37 条记录，全部匹配。关系 ID 的哈希输入没有分隔符；来源维护索引的关系键则通常用已排序端点加 <SEP>，二者不是同一种键。实体在 VDB 中由 ent- ID 定位，在图中由 entity_name 定位；不能拿 ent- ID 直接当图节点名。')
rv=D['sample_relationship_vector']
code(j({k:rv[k] for k in ['__id__','src_id','tgt_id','source_id','content']}))
p('合并按节点名进行；同名实体的新旧类型取计数优势类型，来源 ID 合并，描述去重后连接或按阈值由建库文字模型生成摘要。关系按无序端点对合并，关键词去重，weight 求和，来源合并，描述连接或摘要。图和 VDB 都更新同一对象，所以融合后的描述会被重新编码，并覆盖该对象既有向量记录，而不是把原文向量和图片描述向量做简单加法。[C06 C07]')
p('同名共享是融合键，但这不等于完整的实体消歧。不同拼写如 RAG-Anything 与 RAG-Anything Framework 可能各自保留；同名却不同义也可能被合并。图片锚点能提供归属连接，不能自动解决所有别名或歧义。')
table(['VDB JSON 字段','含义'],[
 ('embedding_dim','向量维数 4096'),('data','逐条记录的数组，保存 __id__、对象键、content、source_id、file_path 等元数据'),
 ('data[i].vector','单记录向量的 Float16 字节经 zlib 压缩再 Base64 编码，可按 ID 取回用于来源块选择'),
 ('matrix','与 data 行顺序一致的归一化 Float32 检索矩阵，经 Base64 保存；不是普通 JSON 数字数组'),
 ('__created_at__','向量记录写入时刻，可能不同于图节点或边的 created_at'),
 ('distance','查询返回的相似度字段。本后端实际为 cosine 相似度，越大越相关，不是越小越好')],[135,369])
p('三个库的首条 vector 解码后都是 4096 个 Float16 元素，matrix 的行数与 data 数量一致。实际余弦搜索通过归一化矩阵与归一化查询向量的点积得到分数。仅向量维度一样不足以替换编码模型，查询应与建库保持相同编码空间。[C08 C09]')

h('7 kv_store_text_chunks 的实际内容回查')
p('此 JSON 顶层键就是 chunk ID。查到 source_id 后拆分、选择来源块、调用 text_chunks.get_by_ids，返回相应对象，再读取 content。该“回查”是按键读取，不是让 LLM 再凭记忆生成内容，也不是再解析 PDF。[C13 C14]')
sample={k:v for k,v in D['main_chunk'].items() if k!='content'}
code(j({D['main_chunk_id']:sample}))
table(['chunk 字段','含义'],[
 ('content','真正参与检索重排与文字证据组装的块文本'),('tokens','建库时对 content 的 token 计数；不是包含全部 JSON 元数据的预算成本'),
 ('chunk_order_index','该文档内块次序，从 0 开始；不是召回排名或 PDF 页码'),('full_doc_id','doc-rag-anything-handson-v1，用于归属文档'),
 ('file_path','PDF 来源引用路径；每个块都有来源，不表示每个块都有原图'),('page_idx page_idx_end','从 0 开始的页码范围；图片页索引也从 0 开始'),
 ('is_multimodal original_type','多模态标记及 image table equation 类型；普通正文不具有这些键'),
 ('modal_entity_name','该多模态块对应锚点的实体名'),('llm_cache_list','该块实体关系抽取缓存的键'),
 ('create_time update_time _id','KV 创建 更新时刻及键的副本')],[139,365])
code('from pathlib import Path\nimport json\nroot = Path("/Users/drw/PycharmProjects/RAG-Anything-reproduction")\nstore = json.loads((root / "storage/handson-qwen4096-v1/kv_store_text_chunks.json").read_text())\nsource_id = "chunk-2d4727f37ea1b30518f13dc89ffc7ea1"\nfor chunk_id in source_id.split("<SEP>"):\n    chunk = store.get(chunk_id)\n    if chunk is None:\n        print("missing", chunk_id)\n        continue\n    print(chunk_id, "PDF page", chunk["page_idx"] + 1)\n    print(chunk["content"])')
p('直接向量检索的另一条路径不同：_get_vector_context 从 vdb_chunks 查询结果中直接拿到 content；实体和关系的来源回查则按选中的 ID 从 KV 获取内容。二者在本库对应同一套 chunk ID。')
p('C05 对应正文块的关键原文如下。它是检索证据，而不是从本机 .env、pilot_manifest 或模型名称中推导出的答案。[F02 F05]')
code('For the retrieval pipeline, we employ the text-embedding-3-large model with 3072-dimensional embeddings. We use the bge-reranker-v2-m3 model for reranking.')
p('C05 的 page_idx=5、page_idx_end=8，KV 因此标注其覆盖 PDF 第 6 至 9 页；第 3.1 节的实现设置位于该块内。块允许跨页，页范围是解析与回填得到的定位信息，不能把单一 page_idx 误认为所有内容都严格只在一页。')

h('8 关键词调用关系与第一题输入')
p('调用方向为 RAG-Anything → LightRAG → llm_model_func → DeepSeek → 返回 JSON。高低层关键词的生成与解析是 LightRAG 内部机制，本项目只提供 DeepSeek 回调，不在 RAG-Anything 内重新设计关键词算法。RAG-Anything 增加的是多模态入库、外层 VLM 查询和原图恢复。[C10 C20]')
p('Q01 的原问题为：'+D['question'])
code(j(D['keyword_response']))
p('低层列表拼接成 "Section 3.1, Embedding model, Vector dimension"，高层列表拼接成 "Original paper, Embedding model, Vector dimension"；加上原问题，共三段文本批量送给 qwen3-embedding。关键词不是一个词一个词分别生成六个查询向量。关键词为英文是因为本次配置的 summary_language=English，而原问题仍以原始中文编码。[C10 C11]')
p('生成的高层关键词也不是拿去精确匹配 GraphML 的 keywords 字段。关系 VDB 的 content 包括边的关键词、两端名称和描述；高层关键词向量与这个复合文本的存量向量做相似度搜索。')

h('9 官方 mix 检索的每一步')
fig(retrieval_fig,'图 4 三条候选来源画成分支以说明数据流；当前实现依次 await 这些检索，不代表这三路一定同时执行。',345)
bullets([
 '实体路径：用低层关键词向量查询 vdb_entities，按相似度和阈值取最多 top_k 条。通过命中记录的 entity_name 回查图节点，读取描述 source_id 等属性，计算节点度数。再读取这些实体的相邻边，按无序端点去重，按边 rank 与 weight 的二元组降序排序。rank 为两端度数之和，优先比较 rank，其次 weight；它不是 rank+weight 的数值相加。',
 '关系路径：用高层关键词向量查询 vdb_relationships，最多 top_k 条；通过 src_id 与 tgt_id 读取图边，保留向量检索顺序，再补充两端实体并去重。这里不会按边 weight 重新替代关系向量排序。',
 'chunk 路径：用原问题向量查询 vdb_chunks，数量为 chunk_top_k；若此参数为空则退回 top_k。结果本身带有 content 和文件引用。',
 '图证据合并：local 与 global 的实体轮流合并，按 entity_name 去重；关系轮流合并，按排序后的端点对去重。邻接扩展和端点补充能使合并数超过最初的 top_k。',
 '图证据裁剪：实体与关系分别按 max_entity_tokens、max_relation_tokens 保留前缀。只使用保留下来的对象回查来源块；裁掉的对象无法再从这个图路径贡献来源块。图对象本身不进入 BGE。'
])
p('实体召回是“查询低层向量 → VDB 中的实体内容向量”，关系召回是“查询高层向量 → VDB 中的关系复合内容向量”。命中后，图检索本身是按实体名或端点对获取属性、邻接关系；GraphML 不在 XML 文字里做向量计算。')
code('cosine(q, x) = dot(q, x) / (norm(q) * norm(x))\n检索结果先按 cosine 降序，再应用本项目 0.2 阈值\n实体 top_k 与关系 top_k 是各自初始召回上限\n它们不限制后续图扩展后的总实体数和总关系数')
fig(synthesis_fig,'图 5 Q01 的来源块选择 重排 预算及视觉恢复。数值来自历史生成路径记录，预算已按官方模板离线核验。',420)
bullets([
 '实体来源：拆分 source_id；统计各来源块的出现次数，去重。默认 kg_chunk_pick_method=VECTOR，取原问题向量与这些来源块已有向量的余弦相似度排序。选择上限为 int(related_chunk_number * 有来源的实体数 / 2)，默认 related_chunk_number=5，实际还受候选总数限制。',
 '关系来源：先排除实体路径已选出的块，再统计去重和选择。VECTOR 的上限公式相同，但使用去重后仍有来源块的关系数。',
 '来源读取：选出块 ID 后，批量从 kv_store_text_chunks 读取 content。来源选择复用原问题向量和已有 chunk 向量，不重新编码全部来源块。',
 '来源选择的回退：没有编码器、无法得到完整来源向量、或选择结果为空时，外层会用 WEIGHT 加权轮询。pick_by_vector_similarity 的内部异常处理还可能直接截取候选 ID 列表，不能把任何异常都说成一定使用 WEIGHT。',
 '块池合并：按直接向量块、实体来源块、关系来源块的顺序轮流取块，按 chunk_id 去重。不是把三路分数进行统一加权求和。',
 '统一块处理：先算剩余 token 预算，再让 BGE 评分排序，回调取前 chunk_top_k；其后检查最小重排分数阈值、再次限制数量，最后按 token 成本保留前缀。',
 '文字证据：实体和关系的描述与最终块内容按模板序列化；按来源文档分配 reference_id；随后 RAG-Anything 从这个最终提示词恢复图片。'
])
p('相关实现见 _perform_kg_search、_apply_token_truncation、_merge_all_chunks、_build_context_str 和 process_chunks_unified。[C11 C12 C13 C14 C15 C16]')

h('10 第一题与真实记录逐项连接')
table(['阶段','Q01 实际证据','对应字段'],[
 ('输入','cv1-q01，mix，视觉增强 重排开启','question mode vlm_requested rerank_enabled'),
 ('关键词','高层 3 个 低层 3 个','metadata.keywords 与 text_request.response'),
 ('图候选','合并找到 48 实体 48 关系','processing_info.total_entities_found / total_relations_found'),
 ('图裁剪','保留 48 实体 48 关系','entities_after_truncation / relations_after_truncation'),
 ('块候选','三路合并去重后 35 块','merged_chunks_count 与 rerank_request.candidate_count'),
 ('重排','35 块评分，top_n=20','生成路径的第二个 rerank_request'),
 ('预算','前 20 中最终保留前 15','final_chunks_count 与 actual_retrieval_prompt'),
 ('视觉内容','6 image 块加 2 table 块，共 8 张','vision_request.image_count 与图片 payload 摘要'),
 ('回答','text-embedding-3-large，3072 维','answer 与 vision_request.response')],[80,230,194])
p('历史记录先保存了一轮诊断 retrieval，再进行生成路径检索。两轮关键词响应相同，35 候选的重排顺序相同；进一步核对实际生成 raw_prompt 中的 15 块，与 retrieval.data.chunks 的内容和顺序完全相同。因此本文可以用保存的 retrieval 字段展示 Q01 的最终块，但这是逐项核验后的结论，不能把所有双检索任务都默认视为证据完全一致。')
p('当前 reproduce.py 通过旁路观察 LightRAG.aquery_llm 返回值记录同次 raw_data；没有额外调用 aquery_data。正常有图时仅取提示词再调用一次视觉模型。历史 Q01 的 rerank_calls=2、call_count=5 与耗时保留历史含义；不是新代码的测量值。[C20]')
p('链路 A 关系到正文：最终关系列表第一条为 RAG-Anything 与 bge-reranker-v2-m3。它的 source_id 指向 C05。这说明问题不需要在图中存在一个名称完全等于“Embedding model”的节点，相关关系仍可把检索带到包含精确答案的来源正文。')
code(j(D['retrieved_relations'][0]))
p('关系 VDB 的 '+rv['__id__']+' 通过端点 RAG-Anything 和 bge-reranker-v2-m3 对应 GraphML 的同一条边；边的 source_id 对应 KV 顶层的 '+D['main_chunk_id']+'。C05 在最终排序中第一，支持模型名与维度答案。')
p('链路 B 实体到图片：最终实体列表第一条为 Text Info, Vector Database, Text VDB, Multi-modal VDB，其 type=UNKNOWN，source_id 指向 C12 架构图描述块。其图节点度数为 1，唯一相邻边连接 Text Encoder，关键词为 database,processing；这条边同样来源 C12。')
local_name=D['retrieved_entities'][0]['entity_name']
code(j(D['sample_nodes'][local_name]['vector']))
p('这个合并名称的节点是实际库中较粗糙的抽取结果，不能把它当作四个已经规范化的独立实体，也不能把 UNKNOWN 理解为图片类型。图谱是由模型抽取与补齐端点产生的结构，类型或名字仍可能存在噪声。')
p('C12 的 modal_entity_name 是 RAG-Anything Framework Architecture Diagram (image)。该锚点单独存在，content 使用图像摘要；C12 的完整 content 则还含章节路径、邻近正文、图注、Image Path 与增强分析。命中来源块后，最终图文生成阶段才能读取架构图原图。')
p('链路 C 融合实体：RAG-Anything 的 source_id 同时包含正文 C05 和图片 C12 等 21 个块。图中 RAG-Anything 到架构图锚点还有 weight=10 的 belongs_to 边。这条边证明多模态知识通过锚点与共享实体进入融合图，不是“只有描述向量而没有图关联”。')
table(['对象','查找键','下一步'],[
 ('实体 VDB','ent-593c92b4435a60e3ecf1f7958626e08c','entity_name → GraphML 节点 → source_id'),
 ('关系 VDB',rv['__id__'],'src_id/tgt_id → GraphML 边 → source_id'),
 ('C05 正文块',D['main_chunk_id'],'KV content 含第 3.1 节实验设置'),
 ('C12 图片块',D['architecture_chunk_id'],'KV content 的 Image Path → parsed 图片'),
 ('文档引用','reference_id = 1','最终 15 块均引用同一份 PDF')],[91,227,186])

h('11 BGE 分数与 token 裁剪的精确核验')
p('Q01 的重排器为本地 BGE，输入是中文原问题与每个候选块 content 的文本对。回调用 sigmoid(logit) 得分，按分数降序，平分按原输入序号排列。不是对图片像素评分。35 个候选的记录均未超过重排 tokenizer 的 8192 最大输入长度，所以本题没有 BGE 输入截断。[C19]')
p('第一块分数为 0.0184968505。这个值用于相对排序，不是答案正确概率，也不是 GraphML 的 weight。项目 min_rerank_score=0，因此即使绝对分数较低也不会按该阈值淘汰；这与后来按数量和 token 预算保留块是不同机制。')
budget=D['token_budget_replay']
table(['预算项','离线核验 token 数'],[
 ('max_total_tokens','24000'),('系统提示词模板开销','559'),('已保留图谱文字上下文','5402'),('原问题','43'),('引用与安全预留','200'),('chunk 可用预算','17796'),('前 15 块累计计费','16797'),('加入第 16 块的累计计费','17944'),('原始最终提示词','21860')],[302,202])
code('available_chunk_tokens = 24000 - (559 + 5402 + 43 + 200)\n                       = 17796\n前 15 块累计 = 16797 <= 17796\n加入第 16 块 = 17944 > 17796\n因此 truncate_list_by_token_size 返回前 15 块')
p('预算用 LightRAG 默认 gpt-4o-mini 对应的 TiktokenTokenizer 计算，并非 DeepSeek 的服务端实际计费 tokenizer。块计费依据包含 content、file_path、chunk_id、rerank_score 的 JSON，而 KV 的 tokens 只计 content。因此 C05 的 KV tokens=1200，但这一步预算成本为 1315。')
p('裁剪在第一个超限块处停止，保留排序前缀，不会跳过第 16 块再尝试更短的第 17 至 20 块。max_total_tokens 也没有在此显式计算后续 image_url 对视觉模型带来的图像 token 开销。实际多模态请求的 token 与这个文字裁剪预算不能直接画等号。[C15 C16]')
rankrows=[]
for r in D['ranked_candidates']:
 rankrows.append((str(r['rank']),cid_alias(r['chunk_id']),r['type'],str(r['page'])+(f"–{r['page_end']}" if r['page_end']!=r['page'] else ''),f"{r['relevance_score']:.8f}",str(r['index']),('保留' if r['in_final'] else ('预算删除' if r['rank']<=20 else '超出 top20'))))
table(['排名','块','类型','PDF 页','BGE 分数','输入序号','结果'],rankrows,[40,38,63,50,95,56,162])
p('排名从 1 开始；rerank_request.ranked_candidates.index 从 0 开始，它索引重排前 35 块的输入池，不能当作 KV 的 chunk_order_index 或 C 编号。附录提供完整 ID 映射。')

h('12 文字提示词与八张原图')
p('最终上下文的实体行使用 entity、type、description；关系行使用 entity1、entity2、description；chunk 行使用 reference_id、content。检索结果 JSON 另外保留 source_id、file_path、chunk_id 等审计字段，所以 raw_data 与发送给生成模型的文本字段并非逐字相同。[C15 C17]')
code('Knowledge Graph Data\n  Entities:      {"entity": "...", "type": "...", "description": "..."}\n  Relationships: {"entity1": "...", "entity2": "...", "description": "..."}\nDocument Chunks\n  {"reference_id": "1", "content": "...Image Path: /...jpg..."}\nReference Document List\n  [1] /Users/drw/PycharmProjects/rag-anything-reproduction/data/RAG-Anything_arXiv_2025.pdf\n---User Query---\n  Q01 原问题')
p('Q01 的 15 个最终块为 5 正文、6 图片、2 表格、2 公式。8 条有效 Image Path 位于 6 个图片块与 2 个表格块中。公式模板只有 LaTeX 与分析，因此本次 2 个公式块不贡献原图。每个块的 file_path 都是来源 PDF 路径，不能把它误认为 15 张图片路径。')
p('RAG-Anything 遍历最终 raw_prompt 中匹配的 Image Path，检查存在性、图片格式与安全目录，编码 Base64，插入 VLM_IMAGE 标记，再在相应文字位置换成 image_url。它并不遍历所有 37 块，把全库原图都送给模型。[C18]')
p('项目 extra_safe_dirs 来自验证后的知识库图片目录，用于兼容旧存储路径中的 rag-anything-reproduction 与当前目录 RAG-Anything-reproduction 的大小写差异。库里原始路径仍保留，验证使用实际文件身份和建库文件摘要；没有把目录限制整体关闭。')
table(['发送顺序','块','类型','PDF 页','原图对应内容'],[(str(x['index']),cid_alias(x['chunk_id']),x['type'],str(x['page']),x['anchor']) for x in D['images']],[53,40,52,45,314])
p('8 个图片 payload_sha256 已与对应本地文件编码形成的 data URL 全部逐项匹配。这证明实际生成请求确实携带这些文件的字节；不能仅凭 image_count=8 就认为路径一定对应正确。下列缩略图按照发送顺序展示，PDF 可放大查看，HTML 中保留原图清晰度。')
blocks.append(('gallery',D['images'][:4]))
blocks.append(('gallery',D['images'][4:]))
p('原图扫描按每个路径出现位置处理，当前函数没有显式按图片文件去重；同一路径在多个匹配位置出现时可能重复发送。Q01 的 8 个路径各不相同。若最终没有有效图片，官方 aquery_vlm_enhanced 会重新调用普通 LightRAG.aquery 进行检索和文本回答；删除项目的诊断检索并不会删除这个官方回退分支。')
p('Q01 的 8 张图片中有架构图、结果表、模型响应对比图等；它们被发送不意味着都与本题直接相关。真正支持实验嵌入模型和维度的是 C05 的正文。这是可观测的检索冗余，不能把“图片都送到了”直接等同于“视觉召回精准”。')

h('13 论文设计与当前实现的边界')
table(['论文描述','当前代码与本次证据'],[
 ('统一表示与跨模态访问 §2.2','实体 关系 chunk 均有文字编码向量；同名实体与锚点归属关系把正文和多模态融入单一图谱'),
 ('模态感知查询 §2.3','模型可在关键词中表达 figure/table 等意图，但此路径没有单独显式计算模态偏好权重'),
 ('精确实体匹配与指定多跳扩展 §2.3','当前 mix 主路径从实体或关系向量检索进入图谱，再取相邻边或关系端点；不应宣称运行了完整的精确匹配加指定多跳算法'),
 ('结构 语义 模态融合评分 §2.3','可见实现为相似度排序、局部关系 rank/weight 排序、轮流合并和可选 chunk BGE；没有统一显式的三信号加权公式'),
 ('文字上下文与原图共同生成 §2.4','实际对应 P(q) 加 Image Path 解引用，再调用 VLM(q, P(q), 原图)')],[161,343])
p('本手册中的流程以固定提交和 LightRAG 1.4.16 的实现为准。论文架构图说明设计意图，不能替代对当前代码执行路径的核对。[F03 C11 C18]')
bullets([
 '关键词预先提供时，LightRAG 可跳过关键词生成；仅一组存在时只走有关键词的图检索分支。两组都为空且问题短于 50 个字符时可将原问题作为低层关键词；长问题则返回失败响应。',
 '向量批量预计算异常时，各 VDB 可在查询时自行计算向量，不能把正常一次批量调用推广为任何异常下都固定一次编码。',
 '同名合并、自动摘要、UNKNOWN 类型、补齐关系端点以及无向边，都会影响图的语义质量。图节点描述不是自动核验过的事实。',
 '来源上限默认每实体或每关系 300 个 ID，FIFO 策略可改变图中保留的 source_id；本库最大来源列表远低于该上限。来源维护索引保留更完整列表，正常查询仍主要读取图上的 source_id。',
 'LightRAG 对重排错误可以回退原顺序；本项目另外审计 rerank_request 并拒绝把失败回退报告为成功重排。',
 '路径失效、不在安全目录、格式错误或图像文件缺失可能令该图不进入生成；没有有效图时走官方文本回退。',
 '本次历史记录没有保存 3 个查询数值向量、每路原始相似度列表和每个来源选择的过程日志。因此可以严格还原最终对象、35 块重排顺序、15 块裁剪及 8 张图，但不能伪造每个初始候选的 cosine 值或独占分支归属。'
])

h('14 本文完成的核验与只读检查方式')
table(['检查','结果'],[
 ('图对象与 VDB 数量','实体 563 关系 998；对象键对应图节点或无序端点'),
 ('哈希 ID 公式','563 实体 998 关系 37 chunk 全部匹配'),
 ('实体 content 公式','536 普通实体为名称加描述，27 锚点为摘要；没有其他不一致'),
 ('关系 content 公式','998 条均与图属性 关键词及端点匹配'),
 ('图来源 ID','节点与边引用的所有 chunk ID 均能在 KV 找到'),
 ('Q01 诊断与实际生成块','实际提示词内 15 块内容与顺序完全匹配保存 retrieval'),
 ('Q01 两轮重排','35 个候选的排序相同'),
 ('预算重算','使用同版本模板 tokenizer 精确重现 15 个最终块'),
 ('图片 payload','8 个 data URL 摘要均匹配对应本地图片字节')],[194,310])
p('下面是只读检查某个关系到来源块的示例。运行后只读取文件，不调用模型、不改库。路径与关系名使用实际 v1 记录。')
code('from pathlib import Path\nimport json\nimport networkx as nx\nroot = Path("/Users/drw/PycharmProjects/RAG-Anything-reproduction")\nkb = root / "storage/handson-qwen4096-v1"\ngraph = nx.read_graphml(kb / "graph_chunk_entity_relation.graphml")\nchunks = json.loads((kb / "kv_store_text_chunks.json").read_text())\nedge = graph.get_edge_data("RAG-Anything", "bge-reranker-v2-m3")\nprint(edge)\nfor cid in edge["source_id"].split("<SEP>"):\n    print(cid, chunks[cid]["content"])')
p('result.json 中 retrieval.data.entities/relationships 是结构化图证据，chunks 是最终内容块，references 是文档引用；metadata.processing_info 是各阶段数量。events.text_request 保存关键词请求响应，rerank_request 保存所有候选排序与 content_sha256，actual_retrieval_prompt 保存真实生成前提示词，vision_request 保存文本与图片 payload 摘要。块 SHA 可与 KV content 的 SHA-256 连接，从而在不调用模型的情况下还原重排候选 ID。[F02]')
p('document 中的 source_id 是来源块 ID；VDB __id__ 是向量记录 ID；reference_id 是最终来源文档引用编号；chunk_order_index 是建库块顺序；rerank.index 是重排输入序号。它们分别负责不同层面的定位，不能相互替换。')

h('15 附录 chunk 完整编号映射')
table(['简称','完整 chunk ID','类型','PDF 页'],[(aliases[cid],cid,c.get('original_type','text'),str(c['page_idx']+1)+(f"–{c['page_idx_end']+1}" if c.get('page_idx_end',c['page_idx'])!=c['page_idx'] else '')) for cid,c in chunks.items()],[41,328,71,64])

h('16 附录 第一题最终实体与关系清单')
p('下列次序就是保存 retrieval 的次序，来源块使用 C 编号以便阅读。完整 description 与未简写字段另在同名 HTML 的“完整证据数据”中提供；PDF 的来源 ID 可通过前一附录还原。')
table(['次序','实体名','类型','来源块'],[(str(i),r['entity_name'],r['entity_type'],sources(r['source_id'])) for i,r in enumerate(D['retrieved_entities'],1)],[35,253,77,139])
table(['次序','关系两端','weight','来源块'],[(str(i),r['src_id']+' ↔ '+r['tgt_id'],str(r['weight']),sources(r['source_id'])) for i,r in enumerate(D['retrieved_relations'],1)],[35,277,48,144])

h('17 证据与代码索引')
p('下列代码均位于当前安装的固定版本中。C 引用表示官方代码或项目适配代码；F 引用表示实际库文件、运行记录或论文。代码中的示意省略与数字来源已在相应正文中说明。')
refs=[
 ('F01',KB/'pilot_manifest.json',None,'v1 建库身份'),('F02',ROOT/'runs/20261011_110345_7173b7cc/cv1-q01-result.json',None,'Q01 历史运行证据'),
 ('F03',ROOT/'data/RAG-Anything_arXiv_2025.pdf',None,'原论文 §2.2 §2.3 §2.4 §3.1'),
 ('F04',KB/'graph_chunk_entity_relation.graphml',None,'真实融合图'),('F05',KB/'kv_store_text_chunks.json',None,'真实内容块'),
 ('C01',SP/'lightrag/kg/networkx_impl.py','self._graph = preloaded_graph or nx.Graph()','图后端与节点 边度数'),
 ('C02',SP/'raganything/processor.py','async def _process_multimodal_content_batch_type_aware','多模态批处理阶段'),
 ('C03',SP/'raganything/processor.py','def _convert_to_lightrag_chunks_type_aware','多模态块格式与模板'),
 ('C04',SP/'raganything/processor.py','async def _store_multimodal_main_entities','锚点写入'),
 ('C05',SP/'raganything/processor.py','async def _batch_add_belongs_to_relations_type_aware','归属关系'),
 ('C06',SP/'lightrag/operate.py','async def _merge_nodes_then_upsert','普通实体合并与向量写入'),
 ('C07',SP/'lightrag/operate.py','async def _merge_edges_then_upsert','关系合并与向量写入'),
 ('C08',SP/'lightrag/kg/nano_vector_db_impl.py','async def upsert','真正调用编码器及查询'),
 ('C09',SP/'nano_vectordb/dbs.py','def _cosine_query','归一化矩阵余弦搜索'),
 ('C10',SP/'lightrag/operate.py','async def get_keywords_from_query','LightRAG 关键词机制'),
 ('C11',SP/'lightrag/operate.py','async def _perform_kg_search','批量编码与三路检索'),
 ('C12',SP/'lightrag/operate.py','async def _apply_token_truncation','实体关系先裁剪'),
 ('C13',SP/'lightrag/operate.py','async def _find_related_text_unit_from_entities','实体来源块选择'),
 ('C14',SP/'lightrag/operate.py','async def _find_related_text_unit_from_relations','关系来源块选择'),
 ('C15',SP/'lightrag/operate.py','async def _build_context_str','预算 文字模板与引用'),
 ('C16',SP/'lightrag/utils.py','async def process_chunks_unified','BGE 阈值 数量与预算裁剪'),
 ('C17',SP/'lightrag/utils.py','def generate_reference_list_from_chunks','文档引用编号'),
 ('C18',SP/'raganything/query.py','async def aquery_vlm_enhanced','原图恢复与无图回退'),
 ('C19',ROOT/'reproduce.py','async def __call__(self, query, documents, top_n=None)','本地 BGE 回调与审计'),
 ('C20',ROOT/'reproduce.py','async def run_query','同次查询结果旁路记录')]
for label,path,needle,desc in refs:
 line=next((i for i,s in enumerate(path.read_text().splitlines(),1) if needle in s),None) if needle else None
 blocks.append(('ref',label,path,line,desc))
p('本次审计的关键源文件 SHA-256 如下；用于区分此文档对应的数据快照。')
for name,digest in D['source_hashes'].items(): code(name+'\n'+digest)

# Build PDF and HTML from the same content blocks.
styles={
 'title':ParagraphStyle('Title',fontName='CNB',fontSize=23,leading=32,textColor=colors.black,spaceAfter=18,wordWrap='CJK'),
 'intro':ParagraphStyle('Intro',fontName='CN',fontSize=10.5,leading=17,spaceAfter=12,wordWrap='CJK'),
 'h':ParagraphStyle('Heading',fontName='CNB',fontSize=15,leading=22,spaceBefore=18,spaceAfter=10,wordWrap='CJK',keepWithNext=True),
 'body':ParagraphStyle('Body',fontName='CN',fontSize=9.8,leading=15.5,spaceAfter=9,wordWrap='CJK'),
 'small':ParagraphStyle('Small',fontName='CN',fontSize=8.1,leading=12,spaceAfter=6,wordWrap='CJK'),
 'code':ParagraphStyle('Code',fontName='Code',fontSize=7.9,leading=12.1,spaceAfter=1,wordWrap='CJK'),
 'cell':ParagraphStyle('Cell',fontName='CN',fontSize=8.5,leading=12.5,wordWrap='CJK'),
 'thead':ParagraphStyle('HeadCell',fontName='CNB',fontSize=8.7,leading=12.8,wordWrap='CJK'),
}
def para(s,kind='body'):return Paragraph(escape(str(s)).replace('\n','<br/>'),styles[kind])
def img_flow(path,max_h=355,width=504):
 with Image.open(path) as im: w,h=im.size
 scale=min(width/w,max_h/h)
 return RLImage(str(path),width=w*scale,height=h*scale)
def make_table(headers,rows,widths):
 arr=[[para(x,'thead') for x in headers]]+[[para(x,'cell') for x in row] for row in rows]
 t=Table(arr,colWidths=widths or [504/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#b3c0cb')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dbe2e8'))]))
 return t
story=[para(TITLE,'title'),para('基于 v1 库与 cv1 q01 的源码和运行记录  2026 年 10 月 12 日','small'),para(INTRO,'intro')]
tocrows=[(str(i),title) for i,(sid,title) in enumerate(sections,1)]
story.append(make_table(['目录','内容'],tocrows,[42,462]));story.append(PageBreak())
html_parts=[]
for block in blocks:
 typ=block[0]
 if typ=='h':
  _,title,sid=block; ph=para(title,'h');ph.section_id=sid;ph.section_title=title;story.append(ph);html_parts.append(f'<h2 id="{sid}">{html.escape(title)}</h2>')
 elif typ=='p':story.append(para(block[1]));html_parts.append('<p>'+html.escape(block[1])+'</p>')
 elif typ=='list':
  for t in block[1]:story.append(para('• '+t))
  html_parts.append('<ul>'+''.join('<li>'+html.escape(t)+'</li>' for t in block[1])+'</ul>')
 elif typ=='note':
  tb=Table([[para(block[1])]],colWidths=[504]);tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#fff3df')),('BOX',(0,0),(-1,-1),.4,colors.HexColor('#dbc59f')),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(tb);html_parts.append('<aside>'+html.escape(block[1])+'</aside>')
 elif typ=='code':
  for line in block[1].splitlines():
   story.append(Paragraph(escape(line or ' ').replace(' ','&#160;'),styles['code']))
  story.append(Spacer(1,9));html_parts.append('<pre><code>'+html.escape(block[1])+'</code></pre>')
 elif typ=='table':
  _,headers,rows,widths=block;story.append(make_table(headers,rows,widths));story.append(Spacer(1,10))
  html_parts.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>')
 elif typ=='fig':
  _,path,caption,maxh=block;story.append(KeepTogether([img_flow(path,maxh or 355),Spacer(1,5),para(caption,'small')]));story.append(Spacer(1,7))
  data=base64.b64encode(path.read_bytes()).decode();html_parts.append(f'<figure><img src="data:image/{"png" if path.suffix==".png" else "jpeg"};base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
 elif typ=='gallery':
  group=block[1];story.append(PageBreak());cells=[];cards=[]
  for x in group:
   caption=f"原图 {x['index']}  {aliases[x['chunk_id']]}  PDF 第 {x['page']} 页\n{x['anchor']}"
   cells.append([img_flow(x['actual_path'],max_h=176,width=237),Spacer(1,5),para(caption,'small')])
   data=base64.b64encode(Path(x['actual_path']).read_bytes()).decode();cards.append(f'<figure><img src="data:image/jpeg;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
  gallery=Table([[cells[0],cells[1]],[cells[2],cells[3]]],colWidths=[252,252]);gallery.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12)]));story.append(gallery);html_parts.append('<div class="gallery">'+''.join(cards)+'</div>')
 elif typ=='ref':
  _,label,path,line,desc=block;s=f'{label}  {desc}\n{path}'+(f'  第 {line} 行' if line else '')
  story.append(para(s,'small'));html_parts.append('<p class="source"><strong>'+html.escape(label+' '+desc)+'</strong><br>'+html.escape(str(path))+(f':{line}' if line else '')+'</p>')
 elif typ=='page':story.append(PageBreak())

class ReportDoc(BaseDocTemplate):
 def afterFlowable(self,flowable):
  if hasattr(flowable,'section_id'):
   self.canv.bookmarkPage(flowable.section_id);self.canv.addOutlineEntry(flowable.section_title,flowable.section_id,0,False)
def footer(c,doc):
 c.saveState();c.setFont('CN',8);c.setFillColor(colors.HexColor('#576777'));c.drawString(44,24,'RAG Anything v1  Q01 检索详解');c.drawRightString(A4[0]-44,24,str(doc.page));c.restoreState()
pdfpath=DEST/'v1-q01-retrieval-explained.pdf'
doc=ReportDoc(str(pdfpath),pagesize=A4,leftMargin=44,rightMargin=47.27,topMargin=40,bottomMargin=42,title=TITLE,author='')
doc.addPageTemplates(PageTemplate(id='main',frames=[Frame(44,42,504,A4[1]-82,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))
doc.build(story)

rawblocks=[]
for title,obj in [('实际最终实体完整字段',D['retrieved_entities']),('实际最终关系完整字段',D['retrieved_relations']),('BGE 35 块完整排序',D['ranked_candidates']),('八张图片完整路径与摘要',D['images']),('C05 原始 KV 记录',D['main_chunk']),('C12 原始 KV 记录',D['architecture_chunk']),('审计检查结果',D['checks']),('token 预算重算',D['token_budget_replay'])]:
 rawblocks.append('<details><summary>'+html.escape(title)+'</summary><pre><code>'+html.escape(j(obj))+'</code></pre></details>')
nav='<nav><strong>目录</strong>'+''.join(f'<a href="#{sid}">{html.escape(title)}</a>' for sid,title in sections)+'<a href="#raw">完整证据数据</a></nav>'
css='''body{margin:0;color:#18232e;background:#f4f6f8;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif} main{max-width:1020px;margin:auto;padding:44px 52px;background:white} h1{font-size:32px;line-height:1.45;color:#000;margin:0 0 12px} h2{font-size:25px;color:#000;margin:44px 0 18px;scroll-margin-top:16px} p,li{overflow-wrap:anywhere} aside{background:#fff5e3;padding:18px 22px;border-left:4px solid #b89343;margin:24px 0} nav{display:grid;grid-template-columns:1fr 1fr;gap:8px 22px;background:#edf2f6;padding:22px;margin:26px 0} nav strong{grid-column:1/-1} nav a{color:#254d70;text-decoration:none} pre{font:13px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;padding:18px;border-radius:5px} table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.65;margin:20px 0} th{text-align:left;background:#edf2f6} td,th{padding:9px 10px;border-bottom:1px solid #dce3e9;vertical-align:top;overflow-wrap:anywhere} .table-wrap{overflow-x:auto} figure{margin:28px 0} img{max-width:100%;height:auto} figcaption{font-size:14px;color:#526273;margin-top:9px} .gallery{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start} .source{font-size:13px;color:#425466} details{margin:16px 0;border:1px solid #d5dfe7;border-radius:6px;padding:14px} summary{cursor:pointer;font-weight:600} .meta{color:#526273;font-size:14px} @media(max-width:760px){main{padding:24px 18px}nav,.gallery{grid-template-columns:1fr}} @media print{body{background:white}main{padding:0;max-width:none}nav{display:none}h2{break-after:avoid}figure{break-inside:avoid}details{display:block}summary{font-size:14px}}'''
htmlpath=DEST/'v1-q01-retrieval-explained.html'
htmlpath.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(TITLE)+'</title><style>'+css+'</style></head><body><main><h1>'+html.escape(TITLE)+'</h1><p class="meta">基于 v1 库与 cv1-q01 的源码和运行记录 · 2026 年 10 月 12 日 · 离线文档</p><p>'+html.escape(INTRO)+'</p>'+nav+''.join(html_parts)+'<h2 id="raw">完整证据数据</h2><p>此处保留实际记录完整字段，便于搜索或复制。没有包含 API 密钥或图片 Base64 请求日志。</p>'+''.join(rawblocks)+'</main></body></html>')
print(json.dumps({'pdf':str(pdfpath),'html':str(htmlpath),'blocks':len(blocks),'sections':len(sections),'pdf_bytes':pdfpath.stat().st_size,'html_bytes':htmlpath.stat().st_size},ensure_ascii=False))
