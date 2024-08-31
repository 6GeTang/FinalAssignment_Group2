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

void Graph::addEdge1(int vertex1, int vertex2, int value) {//lzy
    adjList[vertex1].push_back(vertex2);
    adjList[vertex2].push_back(vertex1); // 无向图

    adjMatrix[vertex1][vertex2] = value;//lzy
    adjMatrix[vertex2][vertex1] = value;//lzy

    edgeList.emplace_back(vertex1, vertex2);
}

void Graph::addEdge2(int vertex1, int vertex2, int value) {//lzy
    adjList[vertex1].push_back(vertex2); // 只在vertex1的邻接列表中添加vertex2 //lzy

    adjMatrix[vertex1][vertex2] = value; // 只设置从vertex1到vertex2的值 //lzy

    edgeList.emplace_back(vertex1, vertex2); // 添加边到边的列表中，并包含边的值 //lzy
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
