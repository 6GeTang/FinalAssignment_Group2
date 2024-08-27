#include <iostream>
#include "Graph.h"
#include "Algorithms.h"

int main() {
    // 定义图的大小（用于邻接矩阵）
    int graphSize = 5;

    // 创建图对象
    Graph graph(graphSize);

    // 添加顶点（仅用于邻接表）
    for (int i = 0; i < graphSize; ++i) {
        graph.addVertex(i);
    }

    // 添加边
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    graph.addEdge(3, 4);
    graph.addEdge(4, 0);

    // 打印邻接表
    graph.printAdjList();

    // 打印邻接矩阵
    graph.printAdjMatrix();

    // 打印边列表
    graph.printEdgeList();


    return 0;
}
