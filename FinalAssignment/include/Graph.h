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
    Graph(int size);
    ~Graph();

    void addVertex(int vertex);
    void addEdge(int vertex1, int vertex2);

    void printAdjList() const;
    void printAdjMatrix() const;
    void printEdgeList() const;

    // 提供访问邻接表 矩阵 边列表的方法，供算法使用
    const std::map<int, std::vector<int>>& getAdjList() const;
    const std::vector<std::vector<int>>& getAdjMatrix() const;
    const std::vector<std::pair<int, int>>& getEdgeList() const;

    int getSize() const;

private:
    std::map<int, std::vector<int>> adjList;
    std::vector<std::vector<int>> adjMatrix;
    std::vector<std::pair<int, int>> edgeList;
    int size;
};

#endif //FINALASSIGNMENT_GRAPH_H
