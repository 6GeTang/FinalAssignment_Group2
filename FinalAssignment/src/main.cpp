#include <iostream>
#include "Graph.h"
#include "Algorithms.h"

#include <windows.h>  //内存性能测试
#include <psapi.h>//内存性能测试
#include <chrono>//时间性能测试


void printMemoryUsage() {  //内存性能测试
    PROCESS_MEMORY_COUNTERS memInfo;
    GetProcessMemoryInfo(GetCurrentProcess(), &memInfo, sizeof(memInfo));
    SIZE_T physMemUsedByMe = memInfo.WorkingSetSize;
    std::cout << "Memory usage: " << physMemUsedByMe / 1024 << " KB" << std::endl;
}

int main() {
//    // 定义图的大小（用于邻接矩阵）
//    int graphSize = 5;
//
//    // 创建图对象
//    Graph graph(graphSize);
//
//    // 添加顶点（仅用于邻接表）
//    for (int i = 0; i < graphSize; ++i) {
//        graph.addVertex(i);
//    }
//
//    // 添加边
//    graph.addEdge(0, 1);
//    graph.addEdge(1, 2);
//    graph.addEdge(2, 3);
//    graph.addEdge(3, 4);
//    graph.addEdge(4, 0);
//
//    // 打印邻接表
//    graph.printAdjList();
//
//    // 打印邻接矩阵
//    graph.printAdjMatrix();
//
//    // 打印边列表
//    graph.printEdgeList();
//

//**************************   //sun A*算法测试
//    // 在算法运行前记录内存使用
//    printMemoryUsage();
//
//    // 记录开始时间
//    auto time_start = std::chrono::high_resolution_clock::now();
//
//
//    // 创建图并添加边
//    Graph graph(5);  // 图的节点数为5
//
//    graph.weightaddEdge(0, 1, 1.0);
//    graph.weightaddEdge(0, 2, 4.0);
//    graph.weightaddEdge(1, 2, 2.0);
//    graph.weightaddEdge(1, 3, 5.0);
//    graph.weightaddEdge(2, 3, 1.0);
//    graph.weightaddEdge(3, 4, 3.0);
//
//    int startId = 0;
//    int destId = 4;
//
//    aStarSearch(graph, startId, destId);
//
//    // 记录结束时间
//    auto time_end = std::chrono::high_resolution_clock::now();
//
//    // 在算法运行后记录内存使用
//    printMemoryUsage();
//
//    // 计算时间差（以毫秒为单位）
//    std::chrono::duration<double, std::milli> duration = time_end - time_start;
//    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;


//***************************************// sun 并查集算法测试
//    UnionFind uf(10);  // 初始化并查集，包含10个元素
//
//    uf.unite(1, 2);
//    uf.unite(3, 4);
//    uf.unite(2,5);
//    uf.unite(2, 4);
//
//
//    std::cout << "1 and 4 connected? " << (uf.connected(1, 4) ? "Yes" : "No") << std::endl;



//***************************** //sun Borůvka算法
//    int V = 4;  // 图中顶点的数量
//    int E = 5;  // 图中边的数量
//
//    std::vector<Edge> edges(E);
//
//    // 初始化边 (u, v, weight)
//    edges[0] = {0, 1, 10};
//    edges[1] = {0, 2, 6};
//    edges[2] = {0, 3, 5};
//    edges[3] = {1, 3, 15};
//    edges[4] = {2, 3, 4};
//
//    // 运行Borůvka算法
//    boruvkaMST(V, E, edges);


//*********************************  //sun Edmonds算法
    int V = 4; // 顶点数量
    std::vector<Edge> edges = {
            {0, 1, 1},
            {0, 2, 5},
            {1, 2, 1},
            {1, 3, 2},
            {2, 3, 1},
            {3,2,2},
            {2,1,4},
            {1,0,3}
    };
    int root = 0;

    std::vector<Edge> mstEdges;  // 存储最小生成树中的边
    int mstWeight = edmondsMST(root, V, edges, mstEdges);

    std::cout << "Weight of the MST is " << mstWeight << std::endl;

    std::cout << "Edges in the MST:" << std::endl;
    for (const auto& edge : mstEdges) {
        std::cout << edge.u << " -> " << edge.v << " (Weight: " << edge.weight << ")" << std::endl;
    }


//    int V3 = 5;
//    std::vector<Edge> edges3 = {
//            {0, 1, 5},
//            {1, 2, 3},
//            {2, 3, 2},
//            {3, 4, 4},
//            {4, 1, 1}, // 环：1 -> 2 -> 3 -> 4 -> 1
//            {1, 3, 10},
//            {2, 0, 7},
//            {3, 0, 8}  // 另一个环：0 -> 1 -> 2 -> 0
//    };
//    int root3 = 0;
//
//
//
//
//
//    std::vector<Edge> mstEdges;  // 存储最小生成树中的边
//    int mstWeight = edmondsMST(root3, V3, edges3, mstEdges);
//
//    std::cout << "Weight of the MST is " << mstWeight << std::endl;
//
//    std::cout << "Edges in the MST:" << std::endl;
//    for (const auto& edge : mstEdges) {
//        std::cout << edge.u << " -> " << edge.v << " (Weight: " << edge.weight << ")" << std::endl;
//    }

    return 0;
}