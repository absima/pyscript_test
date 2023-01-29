import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def D(G,M): # NB: G==D(D(G,M),M))
  Gmm = G.copy()
  Gmm.remove_edges_from(M)
  Gmm = Gmm.reverse()
  Gmm.add_edges_from(M)
  return Gmm,M

def Bipar(G, mch, NN):
  ee = G.edges()
  eeAr = np.array(ee)
  eeAr[:,1] = eeAr[:,1]+NN
  Gb = nx.DiGraph()
  Gb.add_edges_from(eeAr)
  mchAr = np.array(mch)
  mchAr[:,1] = mchAr[:,1]+NN
  mch2 = [tuple(k) for k in mchAr]
  return Gb,mch2

def drawbp(G,mch, NN, kind):
  #NN = NNbp(G)+3 #temporary
  ee = G.edges()
  for k in G.nodes():
    if G.degree(k)==0:
      G.remove_node(k)
  if kind=='original':
    mcho = mch[:]; gmo = G.copy()
    gmb,mchb = Bipar(G,mch,NN)
  #	else:
  #		gmb = G.copy(); mchb = mch[:]
  #		gmo, mcho = fromBipar(G,mch,NN)

  eeb = np.array(gmb.edges())
  gb = D(gmb,mchb)[0]
  go = gmo# and mcho
  ego = [i for i in go.edges()]

  print ('gb', gb.edges())
  ######position specifier for drawing
  eel = np.unique(eeb[:,0]).tolist()
  eer = np.unique(eeb[:,1]).tolist()
  eeu = eel+eer

  ss, tt, cc = len(eel), len(eer), len(eeu)
  #eeu = np.unique(xeg)
  pos = np.ones((cc,2))
  pos[:ss,1] = 1
  pos[ss:,1] = 2
  pos[:ss,0] = range(ss)
  pos[ss:,0] = range(tt)

  aa=dict([])
  for i,j in enumerate(eeu):
    aa[j] = pos[i]

  #	pl.clf()
  clr = ['g']*len(ego)
  for i in mcho:
    dxx = ego.index(i)
    clr[dxx] = 'r'

  fig = plt.figure(figsize=(9,3))
  plt.subplot2grid((1,3),(0,0))
  plt.title('a network digraph')
  # pl.subplot(121)
  nx.draw_circular(go, width=1.5, node_color = [[0,0.4,0.8]], node_size=400, font_color = 'w', \
  font_size=16, alpha=1, edge_color=clr, style='solid', arrows=True, with_labels=True )

  # pl.subplot(122)

  plt.subplot2grid((1,3),(0,1), colspan=2)
  plt.title('its bipartite version')
  nx.draw(gb, pos=aa, node_color = 'gray', node_size=400, font_color = 'k', \
  font_size=16, edge_color='g', with_labels=True)
  nx.draw_networkx_edges(gb, pos=aa, edgelist=mchb, \
  width=1.5, edge_color='r', style='solid', arrows=True)

  return fig


def maxflow(gg, N, capacity='capacity'):
  if True:
    # for graph input:
    V = gg.nodes()
    ee = gg.edges()
    ee = np.array([list(x) for x in ee])
    echk = np.where(ee[0]>N)[0]#.tolist()
    if not len(echk): # original
      ee[:,1]=ee[:,1]+N

    s = 2*N; t=2*N+1
    beg = np.unique(ee[:,0])
    es = np.column_stack(([s]*len(beg), beg))
    fin = np.unique(ee[:,1])
    et = np.column_stack((fin,[t]*len(fin)))

    g = nx.DiGraph()
    g.add_edges_from(ee, capacity=1)
    g.add_edges_from(es, capacity=1)
    g.add_edges_from(et, capacity=1)

    auxiliary=g.copy()
    flow_value = 0

    while True:
      try:
        path_nodes = nx.bidirectional_shortest_path(auxiliary, s, t)
      except nx.NetworkXNoPath:
        break
      path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

      flow_value += 1
      for u, v in path_edges:
        edge_attr = auxiliary[u][v]
        edge_attr[capacity] -= 1
        if edge_attr[capacity] == 0:
          auxiliary.remove_edge(u, v)

        if auxiliary.has_edge(v, u):
          auxiliary[v][u][capacity] += 1#path_capacity
        else:
          auxiliary.add_edge(v, u, capacity=1)#path_capacity)

    auxiliary.remove_nodes_from([s,t])
    g.remove_nodes_from([s,t])
    h=nx.difference(g,auxiliary)
    Ematch = h.edges()
    if not len(echk):
      Ematch = np.array(Ematch)
      Ematch[:,1]=Ematch[:,1]-N
      Ematch = Ematch.tolist()
    Ematch = [tuple(x) for x in Ematch]
    return flow_value, Ematch


######## Regular ring lattice
def regring(N,k):
  xx = np.zeros((N,N)).astype(int)
  for nn in range(N):
    for kk in range(1,k+1):
      xx[nn,nn-kk]=1
  return nx.DiGraph(xx)

######## Watts-Strogatz One-sided Directed
def WSOneSided(gg,p):
  ee = [ii for ii in gg.edges()]
  N = len(gg)
  for ii in range(len(ee)):
    rdm = np.random.rand()
    if rdm<=p: #### here too
      src,tgt = ee[ii]
      pred = gg.predecessors(tgt)
      nonpred = list(set(range(N))-set(pred)-set([tgt]))
      newsrc = np.random.choice(nonpred,1)[0]
      gg.remove_edge(src,tgt)
      gg.add_edge(newsrc,tgt)
  return gg

######## Watts-Strogatz Directed
def WSgraph(N,k,p):
  g0 = regring(N,k)
  g1 = WSOneSided(g0,p)
  g2 = g1.reverse()
  g3 = WSOneSided(g2,p)
  return g3



nV = 15
kk = 3

g1 = WSgraph(nV, kk, 0.)
g2 = WSgraph(nV, kk, 0.07)
g3 = WSgraph(nV, kk, 1.0)

fig=plt.figure(figsize=(9,3))
plt.title('Watts-Strogatz graphs')
plt.subplot(1,3,1)
plt.title('ring lattice')
nx.draw_circular(g1, node_size=400, width=1, node_color = [[0,0.4,0.8]], with_labels=True, font_color = 'w', edge_color='g'  )
plt.subplot(1,3,2)
plt.title('small-world')
nx.draw_circular(g2, node_size=400, width=1, node_color = [[0,0.4,0.8]], with_labels=True, font_color = 'w', edge_color='g')
plt.subplot(1,3,3)
plt.title('random')
nx.draw_circular(g3, node_size=400, width=1, node_color = [[0,0.4,0.8]], with_labels=True, font_color = 'w', edge_color='g')




# ##### example
xeg = np.array([1,6,2,1,3,2,3,4,4,1,4,5,2,5,5,6,6,0,6,3,6,5])
xeg = np.reshape(xeg,(int(len(xeg)/2),2))
mchAr = np.array([1, 6, 2, 1, 3, 2, 4, 5, 6, 0])
mchAr = np.reshape(mchAr,(int(len(mchAr)/2),2))



gg = nx.DiGraph()
gg.add_edges_from(xeg)


n = len(gg);
nn = 10#2*n



nMM, MM = maxflow(gg, nn) # the size of MM and list of edges in MM.
nCN = n-nMM # nCN = 0 means a perfect matching i.e any node is CN
CN_set = set(gg.nodes()) - set(np.array(MM)[:,1].tolist())




print ('A maximum matching set (MM): ', set(MM))
print ('The cardinality of MM is:    ', nMM)
print ('The number of control nodes: ', nCN)
print ('A set of driver nodes:      ', CN_set)


# pl.close('all')

fig2 = drawbp(gg,MM, nn, 'original')

display(fig, target="smallworld")
display(fig2, target="maxmatch")


      