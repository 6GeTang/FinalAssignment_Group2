//
// Created by lenovo on 2024/8/27.
//

#ifndef FINALASSIGNMENT_ALGORITHMS_H
#define FINALASSIGNMENT_ALGORITHMS_H
#include "Graph.h"
#include <vector>
//A*
void aStarSearch(const Graph& graph, int startId, int destId); //sun  A*


//冯碧川的改动开始
// 深度优先搜索 (DFS)
void DFS(const Graph& graph, int startVertex, std::vector<bool>& visited);

// 广度优先搜索 (BFS)
void BFS(const Graph& graph, int startVertex);
void Kosaraju(const Graph& graph);

std::vector<int> topologicalSort(const Graph& graph);

std::vector<int> topologicalSortDFS(const Graph& graph);
void findArticulationPointsAndBridges(const Graph& graph, std::vector<bool>& isAP, std::vector<std::pair<int, int>>& bridges);
//冯碧川的改动结束
void prim(const Graph& graph);//liujun
std::vector<std::vector<int>> kruskal(const Graph& graph);//liujun
std::vector<std::vector<int>> biconnectComponents(Graph& graph);//liujun
std::pair<int, int> dmst(const Graph& graph); //liujun
class UnionFind {   //sun 并查集
public:
    // 构造函数，初始化 n 个节点的并查集
    UnionFind(int n);

    // 查找操作，带路径压缩
    int find(int p);

    // 合并操作，按秩合并
    void unite(int p, int q);

    // 判断两个元素是否在同一集合中
    bool connected(int p, int q);

private:
    std::vector<int> parent;  // 存储每个节点的父节点
    std::vector<int> rank;    // 存储树的秩（近似树的高度）
};



struct Edge {       //sun Borůvka算法
    int u, v, weight;
};

// Borůvka算法主函数
void boruvkaMST(int V, int E, std::vector<Edge>& edges); //sun Borůvka算法

int edmondsMST(int root, int V, std::vector<Edge>& edges, std::vector<Edge>& mstEdges); //sun Edmonds算法

//lzy部分-前
//Floyd-Warshall算法(解决所有节点对之间的最短路径问题)
void floydWarshall(const Graph& graph);

//Bellman-Ford算法:解决单源最短路径问题（支持负权边）
void bellmanFord(const Graph& graph, int source);

// Johnson算法:高效求解多源最短路径问题
std::vector<int> bellmanFord2(const Graph& graph, int source);
std::vector<int> dijkstra2(const Graph& graph, int source, const std::vector<int>& potential);
void johnson(const Graph& graph);

//使用 SPFA 算法求解单源最短路径(Bellman-Ford算法的优化版)
std::vector<int> spfa(const Graph& graph, int source);

//lzy部分-后

#endif //FINALASSIGNMENT_ALGORITHMS_H
