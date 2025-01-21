#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### 【特許データ】※4つのCSVデータがあります
# 1. 20230503PA60999all
# - 紐づく項目「Publication number」、「Applicant(s) BvD ID Number(s)」、「Current direct owner(s) BvD ID Number(s)」、
# - 「Backward citations」…その特許が引用している引用特許のPublication numberが記載されています
# - 「Forward citations」…その特許が引用されている被引用特許のPublication numberが記載されています
# 2. 20230503PA60999abstract
# - 紐づく項目「Publication number」
# - 「Abstract」…要約です
# 3. 20230503PA60999claim
# - 紐づく項目「Publication number」
# - 「Claim」…請求項です
# 4. 20230503PAcampany4815
# - 紐づく項目「BvD ID number」、「Orbis ID number」
# - 「Current market capitalisation th JPY」…最近の時価総額です
# - 【M&Aデータ】※1つのCSVデータがあります
# 5. 20230504MA624620
# - 紐づく項目「Acquiror BvD ID number」、「Target BvD ID number」


# In[1]:


import cugraph
import cudf
import pandas as pd
import numpy as np
import datashader as ds
import cuxfilter
import networkx as nx
import seaborn as sns
import collections as cl
import matplotlib.pyplot as plt


# In[2]:


pa_abst_file = "./data/20230503PA60999abstract.csv"
#pa_all_file = "./data/20230503PA60999all.csv"
pa_all_file = "./data/20230524PA61332all.csv"
pa_claim_file = "./data/20230503PA60999claim.csv"
company_file = "./data/20230503PAcampany4815.csv"
ma_file = "./data/20230504MA624620.csv"


# In[3]:


pa_abst_df = pd.read_csv(pa_abst_file)
pa_all_df = pd.read_csv(pa_all_file,low_memory=False)
pa_claim_df = pd.read_csv(pa_claim_file)
company_df = pd.read_csv(company_file)
ma_df = pd.read_csv(ma_file)


# In[4]:


#pa_abst_df
#pa_all_df
#pa_claim_df
#pa_company_df
#ma_df


# In[5]:


# pa_df に統合
pa_tmp_df = pd.merge(pa_all_df, pa_abst_df, on="Publication number")
pa_df = pd.merge(pa_tmp_df, pa_claim_df, on="Publication number")


# In[6]:


#pa_df


# In[7]:


#pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', 20)
pd.set_option("display.max_colwidth", 50)


# In[8]:


pa_df


# In[9]:


pa_df["Current direct owner(s) BvD ID Number(s)"]


# In[10]:


company_ids = list(pa_df["Current direct owner(s) BvD ID Number(s)"])
# companyは";"で区切られていることがある


# In[11]:


print(company_ids[100:200])


# In[12]:


len(company_ids)


# In[13]:


len(set(company_ids))


# In[14]:


pids = list(pa_df["Publication number"])


# In[ ]:





# In[15]:


pidcompDic = cl.defaultdict(list)
for i,pid in enumerate(pids):
    comp_str = company_ids[i]
    if comp_str is np.nan:
#        print("nan")
        comps = ["NO-NAME"]
    else:
        comps = comp_str.split(';')
        
#    print(comps)
    pidcompDic[pid] = comps
    
#print(comps）

# 目的:
# 特許番号をキー、所有者を値として格納する辞書を作成します。
# 説明:
# defaultdictを使うことで、辞書に存在しないキーへのアクセス時に空のリストを初期値として自動的に追加します。

# 目的:

# 各特許番号に対応する所有者をpidcompDicに登録します。
# 動作:

# enumerate(pids):
# 特許番号リストpidsをインデックスと特許番号のペアに変換。

# comp_str = company_ids[i]:

# 特許番号リストのインデックスiを使って、対応する所有者情報を取得します。
# if comp_str is np.nan:

# np.nan（欠損値）を確認し、欠損値の場合は"NO-NAME"を登録。
# comps = comp_str.split(';'):

# 所有者情報をセミコロン';'で分割。
# 例: '12345;67890' → ['12345', '67890']
# pidcompDic[pid] = comps:

# 特許番号pidをキー、所有者リストcompsを値として辞書に登録。


# In[16]:


#pidcompDic


# In[17]:


set_pids = set(pids)

# 目的:

# 特許番号のリスト（pids）からユニークな特許番号の集合（set型）を作成します。
# 説明:

# セット型は重複を除き、集合演算（例: 共通要素の抽出）に便利です。


# In[18]:


all_c_pids = list(pa_df["Backward citations"])

# 目的:

# 特許データフレームpa_dfから「Backward citations」列をリストとして取得します。
# 説明:

# この列には各特許が参照している過去の特許番号が記載されています。


# In[19]:


pa_df["Backward citations"][29]


# In[20]:


#all_c_pids


# In[21]:


def splitCPidsStr(c_pids_str):
    c_pids_org = c_pids_str.split(';')
    c_pids = list(map(lambda x:x.strip(),c_pids_org))

    return c_pids


# 目的:

# セミコロン（;）で区切られた特許番号を分割し、余分な空白を除去したリストを返します。
# 動作:

# split(';')で文字列をリストに分割。
# strip()で各特許番号の前後の空白を削除。
# それをリストとして返す。


# In[22]:


ref_rows = []
froms = []
tos = []
for i, pid in enumerate(pids):
    c_pids_str = all_c_pids[i]
    if c_pids_str is np.nan:
        continue
    else:
        c_pids = splitCPidsStr(c_pids_str)
#        if len(c_pids) > 1:
#            print(c_pids)

# 目的:

# 各特許番号（pid）に対し、引用特許番号を処理します。
# 動作:

# enumerate(pids)で、pidsリストのインデックスと値を取得。
# 特許pidに対応するBackward citations列の値を取得。
# 値がnp.nanの場合はスキップ。
# 値を分割して特許番号のリスト（c_pids）を作成。
# 例:

# 特許番号が '12345' で、c_pids_str が '67890; 11223' の場合、c_pids = ['67890', '11223']。
    
    and_pids = list(set_pids & set(c_pids))
# 目的:
# 現在の特許番号リストset_pidsと、引用特許番号c_pidsの共通部分を取得します。
    
    for c_pid in and_pids:
        row = [pid,c_pid]
        froms.append(pid)
        tos.append(c_pid)
        ref_rows.append(row)

# 目的:

# 現在の特許pidと引用特許c_pid間の関係を追加します。
    
    # 事業パテントスペース用にエッジを追加
    for comp in pidcompDic[pid]:
        if comp == "NO-NAME":
            continue
        else:
            row = [comp,pid]
            froms.append(comp)
            tos.append(pid)
            ref_rows.append(row)


# In[23]:


#gdf = cudf.DataFrame({"src":froms,"dst":tos})
gdf = pd.DataFrame({"src":froms,"dst":tos})

# 目的:

# fromsとtosを元にエッジリスト形式のデータフレームを作成します。


# In[24]:


pd.set_option("display.max_rows", 100)


# In[25]:


gdf


# In[26]:


#G = cugraph.Graph()


# In[27]:


#G.from_cudf_edgelist(gdf, source='src', destination='dst')
G = nx.from_pandas_edgelist(gdf, source='src', target='dst')


# In[28]:


G.nodes()


# In[29]:


#hdf = cugraph.connected_components(G, connection="weak")
largest_cc = max(nx.connected_components(G), key=len)


# In[30]:


#sel_nodes_set = set(largest_cc)


# In[31]:


#sdf = hdf["labels"].value_counts()


# In[32]:


#sdf


# In[33]:


#maxlabel = sdf.index[0]


# In[34]:


#sdf.index[0]


# In[35]:


#nodes = hdf["vertex"][hdf["labels"]==maxlabel]


# In[36]:


#type(nodes)


# In[37]:


#sG = cugraph.subgraph(G,nodes)
snxG = nx.subgraph(G,largest_cc)


# In[38]:


sdf = pd.DataFrame(snxG.edges())


# In[39]:


sG = cugraph.from_pandas_edgelist(sdf, source=0, destination=1)


# In[40]:


# clustering
#parts, modularity_score = cugraph.leiden(sG, max_iter=10000, resolution=1)
parts, modularity_score = cugraph.louvain(sG, max_iter=10000, resolution=0.1)


# In[41]:


parts


# In[42]:


parts["partition"].max()


# In[43]:


parts["partition"].min()


# In[44]:


c_dicts = parts.to_dict(orient="list")


# In[45]:


#c_dicts["vertex"]


# In[46]:


# cDic[patent_id] = cluster_id
cDic = {}
for i, v in enumerate(c_dicts["vertex"]):
    cid = c_dicts["partition"][i]
#    if cid == ' ':
#        print("HERE")
        
    cDic[v] = cid


# In[47]:


#cDic["JP2014085115A"]


# In[48]:


modularity_score


# In[49]:


#cp = sns.color_palette(n_colors=parts["partition"].max()+1)
cp = sns.color_palette("muted",n_colors=parts["partition"].max()+1)


# In[50]:


cp


# In[51]:


cp[1]


# In[52]:


sG.nodes()


# In[53]:


df_page = cugraph.pagerank(sG)


# In[54]:


df_page


# In[55]:


df_page[df_page["pagerank"] == df_page["pagerank"].max()]


# In[56]:


pd.set_option("display.max_rows", 100)


# In[57]:


df_page.sort_values("pagerank", ascending=False)


# In[58]:


pd.set_option("display.max_colwidth", None)


# In[59]:


pa_df["Title"][pa_df["Publication number"] == "JP6010801003186"]


# In[60]:


pa_df["Abstract"][pa_df["Publication number"] == "JP6010801003186"]


# In[61]:


pos_gdf = cugraph.force_atlas2(sG, max_iter=10000, pos_list=None, outbound_attraction_distribution=True, lin_log_mode=False, prevent_overlapping=False, edge_weight_influence=1.0, jitter_tolerance=1.0, barnes_hut_optimize=True, barnes_hut_theta=0.5, scaling_ratio=50.0, strong_gravity_mode=False, gravity=10, verbose=False, callback=None)
#pos_gdf = cugraph.force_atlas2(sG, max_iter=1000, pos_list=None, outbound_attraction_distribution=True, lin_log_mode=True, prevent_overlapping=False, edge_weight_influence=1.0, jitter_tolerance=1.0, barnes_hut_optimize=True, barnes_hut_theta=0.5, scaling_ratio=2.0, strong_gravity_mode=False, gravity=1.0, verbose=False, callback=None)
# 目的:

# ForceAtlas2アルゴリズムを使ってグラフのノード配置（座標）を計算。
# 説明:

# 力学的モデルに基づき、ノード間の配置を最適化。
# 結果:

# 各ノードの座標を含むデータフレーム。


# In[62]:


pos_gdf


# In[63]:


edf = sG.edges().to_pandas()


# In[64]:


edf


# In[65]:


edf.to_csv("./edges.csv", index=False)


# In[66]:


nxG = nx.from_pandas_edgelist(edf, source=0, target=1)


# In[67]:


vs = list(pos_gdf["vertex"].to_pandas())
xs = list(pos_gdf["x"].to_pandas())
ys = list(pos_gdf["y"].to_pandas())


# In[68]:


#cDic[vs[1]]


# In[69]:


pos = {}
cposDic = cl.defaultdict(list)
cpidDic = cl.defaultdict(list)
pidcidDic = {}
node_sizes = []
node_colors = []
labels = []
for i, v in enumerate(vs):
    x = xs[i]
    y = ys[i]
    pos[v] = [x,y]
    cl_id = cDic[v]
    cposDic[cl_id].append([x,y])
    node_sizes.append(1)
    c = cp[cl_id]
    node_colors.append(c)
    labels.append(f"C{cl_id:02}")
    cpidDic[cl_id].append(v)
    pidcidDic[v] = cl_id

# 目的:

# 各ノードの座標、クラスタID、描画属性（色、サイズ）を準備。
# 説明:

# pos: ノードIDと座標を格納。
# cposDic: クラスタIDごとのノード座標を格納。
# node_sizes, node_colors: 描画用の属性。


# In[70]:


for cid in sorted(cposDic.keys()):
    print(f"{cid}:{len(cposDic[cid])}")
#    if cid > 10:
#        break


# In[71]:


centroidDic = {}
for cid in cposDic.keys():
    poses = np.array(cposDic[cid])
#    print(poses.shape)
#    print(poses)
    x, y = np.average(poses, axis=0)
    centroidDic[cid] = (x,y)


# In[72]:


nx.set_node_attributes(nxG,cDic,"clusterid")


# In[73]:


#nx.draw(nxG,pos,with_labels=False,node_size=node_sizes)


# In[74]:


#nx.draw_networkx_nodes(nxG,pos,nodelist=vs, node_size=node_sizes,node_color=node_colors,alpha=0.5)


# In[75]:


nx.write_gexf(nxG,"./tech.gexf")


# In[76]:


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')
ax.scatter(xs,ys,s=node_sizes,c=node_colors,alpha=0.3)
# 凡例用に空のデータをlabel付でプロットする（実際はなにもプロットしてない）
cids = sorted(centroidDic.keys())
for cid in cids:
    if cid > 10:
        break

    ax.scatter([], [], c=cp[cid], alpha=1.0, label=f"C{cid:02}")
    
ax.legend(loc="upper right")

for cid in cids:
    if cid > 10:
        break
        
    x,y = centroidDic[cid]
    ax.annotate(f"C{cid:02}",(x,y),size=8)


# In[77]:


# cluster名の抽出
from sklearn.feature_extraction.text import TfidfVectorizer

all_pids = pa_df["Publication number"].tolist()
all_titles = pa_df["Title"].tolist()
all_absts = pa_df["Abstract"].tolist()

cDocDic = cl.defaultdict(str)
vs_set = set(vs)
for i, pid in enumerate(all_pids):
    if pid not in vs_set:
        continue
        
    cid = cDic[pid]
    title = all_titles[i]
    cDocDic[cid] += f" {title}. "
#    abst = all_absts[i]
#    cDocDic[cid] += f" {title}. {abst}."


# In[78]:


#cDocDic[40]


# In[79]:


corpus = []
for cid in sorted(cDocDic.keys()):
    adoc = cDocDic[cid]
    corpus.append(adoc)


# In[80]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()


# In[81]:


vocab = vectorizer.get_feature_names_out()
#print(X.toarray())


# In[82]:


X.shape


# In[83]:


#from scipy.stats import rankdata
cWDic = {}
for i in range(X.shape[0]):
#    if i > 0:
#        break
        
    vals = X[i,:]
    res = np.argsort(vals)
    big_indices = res[-20:]
    ws = []
    for idx in big_indices:
        w = vocab[idx]
        ws.append(w)
        
    cWDic[i] = ws


# In[84]:


for cid in sorted(cWDic.keys()):
    ws = cWDic[cid]
    line = " ".join(ws)
    print(f"C{cid:02}: {line}")


# In[85]:


# 特化係数の算出
# CS = {(企業Aのクラスタxに属する特許数)/(企業Aの全特許数)} /{(クラスタxに属する特許数)/(全特許数)}

all_sum = 0
sums = []

for cid in sorted(cpidDic.keys()):
    pids = cpidDic[cid]
    sums.append(len(pids))
    all_sum += len(pids)
    
entire_ratio = np.array(sums)/all_sum


# In[86]:


entire_ratio


# In[87]:


sum(entire_ratio)


# In[88]:


all_comps = []
for comps in pidcompDic.values():
#    print(comps)
    all_comps.extend(comps)
    
comp_set = set(all_comps)

print(len(comp_set))
len_cids = len(cpidDic.keys())


# 会社、クラスタごとの特許数のカウント
compPCDic = {}
for comp in list(comp_set):
    compPCDic[comp] = np.zeros(len_cids)
    
for pid in pidcidDic.keys():
    cid = pidcidDic[pid]
    comps = pidcompDic[pid]
    for comp in comps:
            compPCDic[comp][cid] += 1

over_ten_comps = []
over_ten_comps_clusters = [[] for i in range(len_cids)]
for comp in list(comp_set):
    for cid in range(len_cids):
        num_patent = compPCDic[comp][cid]        
        if num_patent >= 10: 
            over_ten_comps_clusters[cid].append(comp)             


# In[89]:


comp_ratioDic = {}
for comp in compPCDic.keys():
    vals = compPCDic[comp]
    a_sum = sum(vals)
    ratio = vals/a_sum
    ratio = np.nan_to_num(ratio)
#    print(ratio)
#    break
    comp_ratioDic[comp] = ratio


# In[90]:


csDic = {}
for i, comp in enumerate(comp_ratioDic.keys()):        
#    print(comp)
    ratio = comp_ratioDic[comp]
#    print(ratio)
    cs_vals = ratio / entire_ratio
#    print(entire_ratio)
#    print(cs_vals)
    csDic[comp] = cs_vals


# In[91]:


#csDic


# In[92]:


len(csDic.keys())


# In[93]:


# クラスタごとに特化係数の高い企業を表示
comps = list(csDic.keys())

cidCsDic = {}
for cid in sorted(cWDic.keys()):
    cidCsDic[cid] = []
    for i,comp in enumerate(comps):
        cs_vals = csDic[comp]
        cs_val = cs_vals[cid]
        cidCsDic[cid].append(cs_val)


# In[94]:


#cidCsDic


# In[95]:


for cid in sorted(cWDic.keys()):
#    if cid > 0:
#        break 
        
    cs_vals = cidCsDic[cid]
#    print(cs_vals)
#    print(sorted(cs_vals))
    res = np.argsort(cs_vals)
#    print(res)
    compvals = []
    for idx in res[-10:]:
#        print(idx)
        cs_val = cs_vals[idx]
#        print(cs_val)
        comp = comps[idx]
        if comp in over_ten_comps_clusters[cid]:
            compval = f"{comp}:{cs_val:.04f}"
            compvals.append(compval)
        
    compvalstr = ",".join(compvals)
    print(f"C{cid:02}: {compvalstr}")


# In[96]:


#cidCsDic[0]


# In[97]:


# M&Aデータとの突合
madf = pd.read_csv("./data/20230504MA624620.csv",encoding='utf-8')


# In[98]:


madf


# In[99]:


ac_comps = list(madf["Acquiror BvD ID number"])
ta_comps = list(madf["Target BvD ID number"])


# In[100]:


def splitCompsStr(ac_comps, ta_comps):
    # 複数企業間のMAを１：１対応に分解
    all_ac_comps = []
    all_ta_comps = []
    for i,ac_comp_str in enumerate(ac_comps):
        if ac_comp_str is np.nan:
            continue
            
        ta_comp_str = ta_comps[i]
        if ta_comp_str is np.nan:
            continue
            
        ac_comp_org = ac_comp_str.split('|')
        ta_comp_org = ta_comp_str.split('|')
        for an_ac_comp in ac_comp_org:
            for an_ta_comp in ta_comp_org:
                all_ac_comps.append(an_ac_comp)
                all_ta_comps.append(an_ta_comp)

    return all_ac_comps, all_ta_comps


# In[101]:


all_ac_comps, all_ta_comps = splitCompsStr(ac_comps, ta_comps)


# In[102]:


ma_comps = set(all_ac_comps+all_ta_comps)


# In[103]:


pa_comps = set(comps)


# In[104]:


#ma_comps


# In[105]:


#ma_comps & pa_comps


# In[106]:


# maデータの会社名突合できないことがわかった
# nayose.ipynbで名寄せを実行、結果をファイルに格納
#ndf = pd.read_csv("./date/nayose_result.csv")


# In[107]:


#nayose_pmDic = {} # patent company -> ma company
#nayose_mpDic = {} # ma company -> patent company

#nayose_pa_comps = list(ndf["patent"])
#nayose_ma_comps = list(ndf["ma"])
#nayose_sims = list(ndf["similarity"])
#print(len(nayose_pa_comps))

#sel_nayose_pa_comps = []
#sel_nayose_ma_comps = []

#for i, sim in enumerate(nayose_sims):
#    if sim < 0.9:
#        continue
        
#    nayose_pa_comp = nayose_pa_comps[i]
#    nayose_ma_comp = nayose_ma_comps[i]
#    nayose_pmDic[nayose_pa_comp] = nayose_ma_comp
#    nayose_mpDic[nayose_ma_comp] = nayose_pa_comp
#    sel_nayose_pa_comps.append(nayose_pa_comp)
#    sel_nayose_ma_comps.append(nayose_ma_comp)
    
    
#nayose_pa_comps_set = set(sel_nayose_pa_comps)
#nayose_ma_comps_set = set(sel_nayose_ma_comps)


# In[108]:


#len(nayose_pa_comps_set)


# In[109]:


list_nodes = [str(n) for n in pos.keys()]


# In[110]:


len(list_nodes)


# In[111]:


set_nodes = set(list_nodes)


# In[112]:


len(set_nodes)


# In[113]:


list(set_nodes)[100:120]


# In[114]:


#ma_at_pairs = []
#for i,ac_comp in enumerate(ac_comps):
#    ta_comp = ta_comps[i]
#    if ac_comp in nayose_ma_comps_set and ta_comp in nayose_ma_comps_set:
#        if ac_comp in set_nodes and ta_comp in set_nodes:
#            ac_comp_pa = nayose_mpDic[ac_comp]
#            ta_comp_pa = nayose_mpDic[ta_comp]
#            ma_at_pairs.append((ac_comp_pa,ta_comp_pa))


# In[115]:


ma_at_pairs = []
for i,ac_comp in enumerate(all_ac_comps):
    if ac_comp is np.nan:
        continue
        
    ta_comp = all_ta_comps[i]
    if ta_comp is np.nan:
        continue
        
    if ac_comp == ta_comp:
        continue
        
    if len(set([ac_comp,ta_comp])&pa_comps) == 2:
        ma_at_pairs.append((ac_comp,ta_comp))


# In[116]:


len(ma_at_pairs)


# In[117]:


#len(ma_at_pairs)


# In[118]:


#print(ma_at_pairs[0:10])


# In[119]:


#pos.keys()


# In[120]:


# ma 座標算出
# cDic[v] = cid # cluster id
# pos[v] = [x,y]
ac_xs = []
ac_ys = []
ta_xs = []
ta_ys = []
for i,(ac_comp,ta_comp) in enumerate(ma_at_pairs):
    try:
        ac_x,ac_y = pos[ac_comp]
        ta_x,ta_y = pos[ta_comp]
    except KeyError as e:
#        print(e)
        continue
        
    ac_xs.append(ac_x)
    ac_ys.append(ac_y)
    ta_xs.append(ta_x)
    ta_ys.append(ta_y)


# In[121]:


# PAネットワーク上でMAを可視化
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')
ax.scatter(xs,ys,s=node_sizes,c=node_colors,alpha=0.3)
ax.scatter(ta_xs,ta_ys,s=[50]*len(ta_xs),marker='+',c="blue",alpha=1.0)
ax.scatter(ac_xs,ac_ys,s=[50]*len(ac_xs),marker='+',c="red",alpha=1.0)

# 凡例用に空のデータをlabel付でプロットする（実際はなにもプロットしてない）
cids = sorted(centroidDic.keys())
for cid in cids:
    if cid > 10:
        break

    ax.scatter([], [], c=cp[cid], alpha=1.0, label=f"C{cid:02}")
    
ax.legend(loc="upper right")

for cid in cids:
    if cid > 10:
        break
        
    x,y = centroidDic[cid]
    ax.annotate(f"C{cid:02}",(x,y),size=8)


# In[122]:


# 距離の評価
#    ac_xs.append(ac_x)
#    ac_ys.append(ac_y)
#    ta_xs.append(ta_x)
#    ta_ys.append(ta_y)

distances = []
for i,ac_x in enumerate(ac_xs):
    ac_y = ac_ys[i]
    ta_x = ta_xs[i]
    ta_y = ta_ys[i]
    distance = ((ta_x - ac_x)**2 + (ta_y - ta_x)**2)**0.5
    distances.append(distance)


# In[123]:


np.average(distances)


# In[124]:


all_patent_companies = []
for comp_list in pidcompDic.values():
    for comp in comp_list:
        all_patent_companies.append(comp)
        
all_patent_companies_set = set(all_patent_companies)


# In[125]:


pos_companies = list(set(list(pos.keys())) & all_patent_companies_set)


# In[126]:


import random

rand_distances = []
rand_ma_at_pairs = []
for i,ac_x in enumerate(ac_xs):
    ac_comp,ta_comp = random.sample(pos_companies,2)
    rand_ma_at_pairs.append((ac_comp,ta_comp))
    ac_x, ac_y = pos[ac_comp]
    ta_x, ta_y = pos[ta_comp]
    distance = ((ta_x - ac_x)**2 + (ta_y - ta_x)**2)**0.5
    rand_distances.append(distance)


# In[127]:


np.average(rand_distances)


# In[128]:


all_distances = distances + rand_distances

#bins = np.linspace(min(all_distances), max(all_distances), 50)
bins = np.linspace(0,0.5e6, 100)

plt.hist(rand_distances, bins, alpha = 0.5, label='Random')
plt.hist(distances, bins, alpha = 0.5, label='M&A')
plt.legend(loc='upper right')

plt.show()


# In[129]:


# クラスタ内MA数の割合
# ma_at_pairs
# rand_ma_at_pairs
# cDic[v] = cid

num_clusters = len(set(cDic.values()))
print(num_clusters)

ma_cl_counts = np.zeros((num_clusters,num_clusters),dtype=np.float32)


# In[130]:


#cDic


# In[131]:


all_c = 0
hit_c = 0
for ac_comp, ta_comp in ma_at_pairs:
    all_c += 1
    try:
        ac_cid = cDic[ac_comp]
        ta_cid = cDic[ta_comp]
        hit_c += 1
#        print(ac_cid)
    except KeyError as e:
#        print(e)
        continue
        
#    print(ac_cid)
#    print(ta_cid)
#    print("")
    ma_cl_counts[ac_cid,ta_cid] += 1
    
print(f"{hit_c}/{all_c}")


# In[132]:


import random

rand_ma_at_pairs = []
for i in range(0,len(ma_at_pairs)*10,1):
    ac_comp,ta_comp = random.sample(pos_companies,2)
    rand_ma_at_pairs.append((ac_comp,ta_comp))
    ac_x, ac_y = pos[ac_comp]
    ta_x, ta_y = pos[ta_comp]
    distance = ((ta_x - ac_x)**2 + (ta_y - ta_x)**2)**0.5
    rand_distances.append(distance)


# In[133]:


rand_ma_cl_counts = np.zeros((num_clusters,num_clusters),dtype=np.float32)
#print(rand_ma_at_pairs)
for ac_comp, ta_comp in rand_ma_at_pairs:
    try:
        ac_cid = cDic[ac_comp]
        ta_cid = cDic[ta_comp]
    except KeyError as e:
        print(e)
        continue
        
#    print(ac_cid)
#    print(ta_cid)
#    print("")
    rand_ma_cl_counts[ac_cid,ta_cid] += 1


# In[134]:


rand_ma_cl_counts += 10


# In[135]:


fig, ax = plt.subplots()
sns.heatmap(ma_cl_counts[0:20,0:20],cmap="Reds")
#ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xlabel("Target Company Cluster")
ax.set_ylabel("Acquiror Company Cluster")

# タイトルを設定する。
ax.set_title("M&A counts among Clusters")


# In[136]:


fig, ax = plt.subplots()
sns.heatmap(rand_ma_cl_counts[0:20,0:20],cmap="Reds")
#ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xlabel("Target Company Cluster")
ax.set_ylabel("Acquiror Company Cluster")

# タイトルを設定する。
ax.set_title("Random M&A counts among Clusters")


# In[137]:


ma_cl_ratio = ma_cl_counts / rand_ma_cl_counts


# In[138]:


#ma_cl_ratio
ma_cl_ratio = np.nan_to_num(ma_cl_ratio, nan=0)
#ma_cl_ratio[ma_cl_ratio < 1e-9] = 0


# In[139]:


np.amax(ma_cl_ratio)


# In[140]:


sns.heatmap(ma_cl_ratio[0:20,0:20],cmap="Reds")
#ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xlabel("Target Company Cluster")
ax.set_ylabel("Acquiror Company Cluster")

# タイトルを設定する。
ax.set_title("MA/Random counts ratio among Clusters")


# In[141]:


# ratioの高いMAの抽出
#max_idx = ind = np.unravel_index(np.argmax(ma_cl_ratio, axis=None), ma_cl_ratio.shape)
#max_idx
maratios = []
for i in range(0,num_clusters,1):
    for j in range(0,num_clusters,1):
        ratio = ma_cl_ratio[i,j]
        c = ma_cl_counts[i,j]
        if ratio == 0:
            continue
            
        row = (i,j,c,ratio)
        maratios.append(row)


# In[142]:


rdf = pd.DataFrame(maratios)
rdf.columns = ["AcqCl","TarCl","MA count","Ratio"]


# In[143]:


pd.set_option('display.max_rows', 100)

rdf.sort_values("Ratio",ascending=False)


# In[144]:


#nayose_pmDic


# In[145]:


comp_ids = list(company_df["BvD ID number"])
comp_names = list(company_df["Company name Latin alphabet"])


# In[146]:


compDic = {}
for i, comp_id in enumerate(comp_ids):
    comp_name = comp_names[i]
    compDic[comp_id] = comp_name


# In[147]:


#compDic


# In[148]:


# MA抽出
def clfilterMA(sel_acq_cid,sel_tar_cid,cDic,compDic):
    res = []
    for i,(ac_comp, ta_comp) in enumerate(ma_at_pairs):
#        print(ta_comp)
        try:
            ac_cid = cDic[ac_comp]
            ta_cid = cDic[ta_comp]
        except KeyError as e:
#            print(e)
            continue
            
        if ac_cid == sel_acq_cid and ta_cid == sel_tar_cid:
            if ac_comp in compDic.keys() and ta_comp in compDic.keys():
                ma_acq_name = compDic[ac_comp]
                ma_tar_name = compDic[ta_comp]
                res.append((ma_acq_name,ma_tar_name))
        
    return res


# In[149]:


resMA = clfilterMA(0,0,cDic,compDic)
resMA


# In[150]:


resMA = clfilterMA(1,1,cDic,compDic)
resMA


# In[151]:


resMA = clfilterMA(0,1,cDic,compDic)
resMA


# In[152]:


resMA = clfilterMA(5,0,cDic,compDic)
resMA


# In[153]:


resMA = clfilterMA(10,0,cDic,compDic)
resMA


# In[154]:


resMA = clfilterMA(0,11,cDic,compDic)
resMA


# In[ ]:





# In[155]:


#オリジナル分析
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[156]:


#1. セミコロン区切りの特許番号を分割
def split_citations(citations):
    if pd.isna(citations):
        return []
    return [cit.strip() for cit in citations.split(';')]


# In[157]:


# 2. 特許データの引用列を分割
pa_df['Backward citations split'] = pa_df['Backward citations'].apply(split_citations)


# In[158]:


# 3. グラフエッジリストの作成
edges = []
for _, row in pa_df.iterrows():
    citing_patent = row['Publication number']
    cited_patents = row['Backward citations split']
    for cited_patent in cited_patents:
        edges.append((cited_patent, citing_patent))  # 引用元 -> 引用先


# In[159]:


# 4. グラフ作成
G = nx.DiGraph()  # 有向グラフ
G.add_edges_from(edges)


# In[160]:


# 5. 次数中心性（引用されている数）と近傍中心性（引用を辿って根っこにあるもの）を計算
degree_centrality = nx.degree_centrality(G)  # 次数中心性
closeness_centrality = nx.closeness_centrality(G)  # 近傍中心性


# In[161]:


# 6. データフレームにまとめる
centrality_df = pd.DataFrame({
    'Publication number': list(degree_centrality.keys()),
    'Degree Centrality': list(degree_centrality.values()),
    'Closeness Centrality': [closeness_centrality.get(node, 0) for node in degree_centrality.keys()]
})


# In[162]:


# 7. 結果を保存
centrality_df.sort_values(by='Degree Centrality', ascending=False, inplace=True)
centrality_df.to_csv("centrality_analysis.csv", index=False)


# In[163]:


# 上位10件を表示
print(centrality_df.head(10))


# In[ ]:


# 8. ばねモデルのレイアウトを計算
pos = nx.spring_layout(G, k=0.1, iterations=50)


# In[ ]:


# 9. グラフの可視化
plt.figure(figsize=(12, 12))
nx.draw(
    G,
    pos,
    with_labels=False,  # ノードラベルは非表示
    node_size=[v * 1000 for v in degree_centrality.values()],  # 次数中心性に基づくノードサイズ
    node_color=list(closeness_centrality.values()),  # 近傍中心性に基づくノード色
    cmap=plt.cm.viridis,  # 色マップを設定
    edge_color="gray",
    alpha=0.6,
    arrows=True
)


# In[ ]:


# カラーバーを追加
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(closeness_centrality.values()), vmax=max(closeness_centrality.values())))
sm.set_array([])
plt.colorbar(sm, label="Closeness Centrality")

# タイトル設定
plt.title("Citation Network with Centralities")
plt.show()


# In[ ]:


#ページランクの計算
# 1. PageRankの計算
pagerank_scores = nx.pagerank(G, alpha=0.85)


# In[ ]:


# 2. データフレームにPageRankスコアを追加
centrality_df['PageRank'] = centrality_df['Publication number'].map(pagerank_scores)


# In[ ]:


# 3. データフレームをPageRankスコアでソート
centrality_df.sort_values(by='PageRank', ascending=False, inplace=True)


# In[ ]:


# 4. 結果を保存
centrality_df.to_csv("centrality_with_pagerank.csv", index=False)


# In[ ]:


# 上位10件を表示
print(centrality_df.head(10))


# In[ ]:


import networkx as nx
import pandas as pd

# 1. セミコロン区切りの特許番号を分割
def split_citations(citations):
    if pd.isna(citations):
        return []  # NaNの場合は空リストを返す
    return [cit.strip() for cit in citations.split(';')]


# In[ ]:


# 2. 特許データの引用列を分割
pa_df['Backward citations split'] = pa_df['Backward citations'].apply(split_citations)


# In[ ]:


# 3. グラフエッジリストの作成
edges = []
for _, row in pa_df.iterrows():
    citing_patent = row['Publication number']
    cited_patents = row['Backward citations split']
    for cited_patent in cited_patents:
        edges.append((cited_patent, citing_patent))  # 引用元 -> 引用先


# In[ ]:


# 4. グラフ作成
G = nx.DiGraph()  # 有向グラフ
G.add_edges_from(edges)


# In[ ]:


# 5. 基点の特定
# すべてのノードをチェックし、入力エッジ（引用されている）がないノードを基点として特定
roots = [node for node in G.nodes if G.in_degree(node) == 0]


# In[ ]:


# 6. 結果をデータフレームにまとめる
roots_df = pd.DataFrame({'Root Publication number': roots})
roots_df.to_csv("citation_roots.csv", index=False)


# In[ ]:


# 7. 結果を表示
print("引用の基点（ルート特許）の数:", len(roots))
print("引用の基点:")
print(roots_df.head(10))


# In[ ]:




