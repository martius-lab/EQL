from graphviz import Digraph
import numpy as np


def getEdges(matrix,inputnames,outputnames,thresh=0.1):
    edges=[]
    it = np.nditer(matrix, flags=['multi_index'])
    while not it.finished:
        if np.abs(it[0])>thresh:
            edges.append((inputnames[it.multi_index[0]],outputnames[it.multi_index[1]],np.round(it[0].item(),2)))
        it.iternext()
    return edges

def functionGraph1H(classifier,thresh=0.1):
    functionGraph(classifier, thresh)

def functionGraph(classifier,thresh=0.1):
    n_in,n_out = classifier.n_in, classifier.n_out
    try:
      shortcuts  = classifier.with_shortcuts
    except AttributeError:
      shortcuts=False

    names_in = [ 'x' + str(s) for s in range(1,n_in+1)]
    names_out= [ 'y' + str(s) for s in range(1,n_out+1)]
    alledges = []
    allbiases = []
    for l in range(len(classifier.hidden_layers)+1):
        if l==0:
            inp=names_in
        else:
            inp = classifier.hidden_layers[l-1].getNodeFunctions()
        if l==len(classifier.hidden_layers):
            if shortcuts:
                inp = np.concatenate([ l.getNodeFunctions() for l in classifier.hidden_layers ])
            out = names_out
            ps = classifier.output_layer.get_params()
            W  = ps[0]
            b  = ps[1]
        else:
            out = classifier.hidden_layers[l].getWeightCorrespondence()
            ps = classifier.hidden_layers[l].get_params()
            W  = ps[0]
            b  = ps[1]

        alledges.extend(getEdges(W, inp, out ,thresh))
        allbiases.extend(list(zip(out,b)))

    nodes=list(set([e[0] for e in  alledges])) + list(set([e[1] for e in  alledges]))

    def isArgument(name):
        return ':' in name

    def arity2node(name,b1, b2):
        return '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR>
    <TD PORT="1">''' + b1 + '''</TD>
    <TD PORT="2">''' + b2 + '''</TD>
  </TR>
  <TR><TD COLSPAN="2">''' + name + '''</TD></TR>
</TABLE>>'''
    arity2set = set([n.split(':')[0] for n in nodes if isArgument(n)])
    arity2 = list(arity2set)
    arity1 = list(set([n for n in nodes if not isArgument(n)]) - arity2set)
    bias_dict = dict(allbiases)

    dot = Digraph(comment='Function Graph')

    for n in arity1:
        if n in bias_dict:
            dot.node(n,str(np.round(bias_dict[n],2)) + '\n' + n.split('-')[0])
        else:
            dot.node(n,n.split('-')[0])
    for n in arity2:
        dot.node(n,arity2node(n.split('-')[0],
                                  str(np.round(bias_dict.get(n+ ':1',0),2)),
                                  str(np.round(bias_dict.get(n+ ':2',0),2)) ),shape='plaintext')
    for e in alledges:
        dot.edge(e[0], e[1], label=str(e[2]))
    return dot
