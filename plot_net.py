import graphviz

def visualize_network(nodes, edges):
    dot = graphviz.Digraph()
    for i in nodes:
        dot.node(str(i), label=str(i))

    # Collegamento dei nodi del tensore ai nodi dei layer
    for i in edges:
        dot.edge(str(i[0]), str(i[1]))

    # Visualizzazione del grafo
    dot.render('network_graph', format='png', view=True)

# Esempio di input di un tensore e dei layer della rete
nodes = ["input_1", 'input_2', 'input_3', 'hidden', 1, 2, 3]
edges = [('input_1', 'hidden'), ('input_2', 'hidden'), ('input_3', 'hidden'), ('hidden', 1), ('hidden', 2), ('hidden', 3)]
# Visualizzazione del tensore e dei layer come grafo con Graphviz
visualize_network(nodes, edges)
