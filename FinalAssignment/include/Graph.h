//
// Created by lenovo on 2024/8/27.
//

#ifndef FINALASSIGNMENT_GRAPH_H
#define FINALASSIGNMENT_GRAPH_H

#include <vector>
#include <map>
#include <utility>

class Graph {
public:

    //冯碧川在此处改动
    //为有向图和拓扑排序方法添加声明
    Graph generateDirectedGraph() const;
    Graph generateTransposedGraph() const;
    std::vector<int> topologicalSort() const;

    //冯碧川的改动到此结束



    Graph(int size);
    ~Graph();

    void addVertex(int vertex);
    void addEdge(int vertex1, int vertex2);
    void weightAddEdge(int vertex1, int vertex2, int weight); // liujun
    void addEdge1(int vertex1, int vertex2, int value);//lzy
    void addEdge2(int vertex1, int vertex2, int value);//lzy
    void addEdge_directed(int vertex1, int vertex2);//tg
    void addEdge_undirected(int vertex1, int vertex2);//tg

    void weightaddEdge(int vertex1, int vertex2, double weight);  //sun
    const std::vector<std::pair<int, double>>& getNeighbors(int vertex) const;  //sun

    void printAdjList() const;
    void printAdjMatrix() const;
    void printEdgeList() const;
    void printWeightAdjList() const;  // Ding

    // 提供访问邻接表 矩阵 边列表的方法，供算法使用
    const std::map<int, std::vector<int>>& getAdjList() const;
    const std::vector<std::vector<int>>& getAdjMatrix() const;
    const std::vector<std::pair<int, int>>& getEdgeList() const;
    const std::vector<std::vector<int>>& getWeightadjEdgeList() const; // liujun

    // 新增：有向图专用函数
    void addDirectedEdge(int vertex1, int vertex2);  // New
    void addWeightedDirectedEdge(int vertex1, int vertex2, double weight);  // New
    //ading
    int getSize() const;

    // 为最小路径覆盖添加的函数声明
    Graph constructBipartiteGraph() const;

private:
    std::map<int, std::vector<int>> adjList;
    std::vector<std::vector<int>> weightadjEdgeList;
    std::vector<std::vector<std::pair<int, double>>> weightadjList;  //sun A*
    std::vector<std::vector<int>> adjMatrix;
    std::vector<std::pair<int, int>> edgeList;
    int size;
};

#endif //FINALASSIGNMENT_GRAPH_H
