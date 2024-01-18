import pandas as pd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


def cal_degree(adj_matrix):
    """
    calculate degree for given adj_matrix
    """
    #print('cal degree...')
    adj = adj_matrix.copy()
    adj[adj==0] = np.nan
    adj = adj/adj
    in_degree = adj.sum(axis=0).values
    out_degree = adj.sum(axis=1).values
    degree = in_degree+out_degree
    return degree

def cal_avg_degree(adj_matrix):
    """
    calculate average degree for given adj_matrix
    """
    #print('cal avg_degree...')
    adj = adj_matrix.copy()
    adj[adj==0] = np.nan
    adj = adj/adj
    in_degree = adj.sum(axis=0).values
    out_degree = adj.sum(axis=1).values
    degree = in_degree+out_degree
    degree = degree.sum()/adj.shape[0]
    return degree

def cal_weight_degree(adj_matrix):
    """
    calculate weight degree for given adj_matrix
    """
    #print('cal weight_degree...')
    adj = adj_matrix.copy()
    adj[adj==0] = np.nan
    in_degree = adj.sum(axis=0).values
    out_degree = adj.sum(axis=1).values
    degree = in_degree+out_degree
    return degree

def cal_avg_weight_degree(adj_matrix):
    """
    calculate average weight degree for given adj_matrix
    """
    print('cal avg_weight_degree...')
    adj = adj_matrix.copy()
    adj[adj==0] = np.nan
    in_degree = adj.sum(axis=0).values
    out_degree = adj.sum(axis=1).values
    degree = in_degree+out_degree
    a=(adj/adj).sum().sum()
    degree = degree.sum()/a/2
    return degree

def cal_closeness(graph):
    """
    calculate closeness of given graph object
    """
    #print('cal closeness...')
    return graph.closeness(weights=graph.es['weight'])


def cal_avg_closeness(graph):
    """
    calculate average closeness of given graph object
    """
    print('cal avg_closeness...')
    return np.mean(graph.closeness(weights=graph.es['weight']))

def cal_betweenness(graph):
    """
    calculate betweeness of given graph object
    """
    #print('cal betweenness...')
    return graph.betweenness(weights=graph.es['weight'])

def cal_avg_betweenness(graph):
    """
    calculate average betweeness of given graph object
    """
    print('cal avg_betweenness...')
    return np.mean(graph.betweenness(weights=graph.es['weight']))


def cal_avg_path_length(graph):
    """
    calculate average path length
    """
    return graph.average_path_length(weights=graph.es['weight'])

def cal_diameter(graph):
    """
    calculate diameter
    """
    return graph.diameter(weights=graph.es['weight'])


def cal_transitivity(graph):
    """
    calculate transitivity
    传递性度量顶点的相邻顶点连接的概率。这有时也称为聚类系数。
    """
    print('cal transitivity...')
    return graph.transitivity_avglocal_undirected(weights=graph.es['weight'])


def cal_eigenvector_centrality(graph):
    """
    calculate eigenvector_centrality
    """
    return graph.eigenvector_centrality(weights=graph.es['weight'])


class MyGraph():
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.init_graph()
        
    def init_graph(self):
        #print('init graph...')
        self.g = ig.Graph(directed=True)
        for i in self.adj_matrix.columns:
            self.g.add_vertex(name=i)

        edges = []
        node_labels = self.adj_matrix.columns.values
        weights = []
        for i in range(self.adj_matrix.shape[0]):
            #print(i)
            for j in range(self.adj_matrix.shape[0]):
                if self.adj_matrix.iloc[i,j]!=0 and not np.isnan(self.adj_matrix.iloc[i,j]): 
                    edges.append([i,j])
                    weights.append(self.adj_matrix.iloc[i,j])
        #print('add edges...')
        self.g.add_edges(edges)
        self.g.vs['name'] = node_labels
        self.g.vs["label"] = self.g.vs['name']
        self.g.es['weight'] = weights

    def plot_adj_matrix_hist(self):
        """
        plot histogram of adj-matrix
        """
        pd.DataFrame(self.adj_matrix.values.flatten()).hist()
    
    def plot_circle(self,figsize=(10,6),
                       vertex_color="lightblue"):
        """
        plot network at form of "circle"
        """
        print('plot...')
        fig, ax = plt.subplots(figsize=figsize)
        ig.plot(
            self.g,
            target=ax,
            layout='circle',
            vertex_color=vertex_color
        )
        plt.show()


    def plot_clusters(self, figsize=(25,12),
                            vertex_size=0.5,
                            edge_width=0.7):
        """
        plot network at form of "clusters"
        """
        components = self.g.connected_components(mode='strong')
        fig, ax = plt.subplots(figsize=figsize)
        ig.plot(
            components,
            target=ax,
            palette=ig.RainbowPalette(),
            vertex_size=vertex_size,
            vertex_color=list(map(int, ig.rescale(components.membership, (0, 200), clamp=True))),
            edge_width=edge_width
        )
        plt.show()

        
    def profile(self):
        """
        describe network with key indicators
        """
        print('stats profile...')
        g=self.g
        properties_dict = {"节点数":g.vcount(),
                           "边数":g.ecount(),
                            '是否有向':g.is_directed(),
                           '是否加权':g.is_weighted(),
                            '最大度':g.maxdegree(),
                           '平均度':round(cal_avg_degree(self.adj_matrix), 5),
                           '平均加权度(度中心性)':cal_avg_weight_degree(self.adj_matrix),
                            "网络直径":round(cal_diameter(g),5),
                           "平均路径长度":round(cal_avg_path_length(g),5),
                           '平均介数中心性betweenness':round(cal_avg_betweenness(g),5),
                           '平均接近中心性closeness':round(cal_avg_closeness(g),5),
                           '聚类系数':round(cal_transitivity(g),5),
}
        return properties_dict
    
    