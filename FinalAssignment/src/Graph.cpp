#include "Graph.h"
#include <iostream>

Graph::Graph(int size) : size(size) {
    adjMatrix.resize(size, std::vector<int>(size, 0));
}

Graph::~Graph() {}

void Graph::addVertex(int vertex) {
    if (adjList.find(vertex) == adjList.end()) {
        adjList[vertex] = std::vector<int>();
    }
}

void Graph::addEdge(int vertex1, int vertex2) {
    adjList[vertex1].push_back(vertex2);
    adjList[vertex2].push_back(vertex1); // 无向图

    adjMatrix[vertex1][vertex2] = 1;
    adjMatrix[vertex2][vertex1] = 1; // 无向图

    edgeList.emplace_back(vertex1, vertex2);
}
void Graph::weightaddEdge(int vertex1, int vertex2, double weight) {    //sun A*
    weightadjList[vertex1].push_back({vertex2, weight});
    weightadjList[vertex2].push_back({vertex1, weight}); // 无向图
}

const std::vector<std::pair<int, double>>& Graph::getNeighbors(int vertex) const {   //sun A*
    return weightadjList[vertex];
}
void Graph::printAdjList() const {
    std::cout << "Adjacency List:" << std::endl;
    for (const auto& pair : adjList) {
        std::cout << pair.first << ": ";
        for (const auto& neighbor : pair.second) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }
}

void Graph::printAdjMatrix() const {
    std::cout << "Adjacency Matrix:" << std::endl;
    for (const auto& row : adjMatrix) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void Graph::printEdgeList() const {
    std::cout << "Edge List:" << std::endl;
    for (const auto& edge : edgeList) {
        std::cout << edge.first << " - " << edge.second << std::endl;
    }
}

const std::map<int, std::vector<int>>& Graph::getAdjList() const {
    return adjList;
}

const std::vector<std::vector<int>>& Graph::getAdjMatrix() const {
    return adjMatrix;
}

const std::vector<std::pair<int, int>>& Graph::getEdgeList() const {
    return edgeList;
}

int Graph::getSize() const {
    return size;
}


//冯碧川在此下面开始改动：
// 生成有向图的方法
Graph Graph::generateDirectedGraph() const {
    Graph directedGraph(size);

    for (const auto& edge : edgeList) {
        int from = edge.first;
        int to = edge.second;

        // 在有向图中只保留一个方向的边 (from -> to)
        directedGraph.addVertex(from);
        directedGraph.addVertex(to);

        directedGraph.adjList[from].push_back(to);
        directedGraph.adjMatrix[from][to] = 1;
        directedGraph.edgeList.emplace_back(from, to);
    }

    return directedGraph;
}



// 生成转置图的方法
Graph Graph::generateTransposedGraph() const {
    Graph transposedGraph(size);

    for (const auto& edge : edgeList) {
        int from = edge.first;
        int to = edge.second;

        // 在转置图中将边的方向反转 (to -> from)
        transposedGraph.addVertex(from);
        transposedGraph.addVertex(to);

        transposedGraph.adjList[to].push_back(from);
        transposedGraph.adjMatrix[to][from] = 1;
        transposedGraph.edgeList.emplace_back(to, from);
    }

    return transposedGraph;
}

//冯碧川的改动结束

