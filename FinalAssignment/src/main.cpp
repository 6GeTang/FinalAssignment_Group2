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

void Kosaraju(const Graph& graph);

std::vector<int> topologicalSort(const Graph& graph);

std::vector<int> topologicalSortDFS(const Graph& graph);

int main() {
//    // 定义图的大小（用于邻接矩阵）
    int graphSize = 5;
//
//    // 创建图对象
    Graph graph(graphSize);

    // 添加顶点（仅用于邻接表）
    for (int i = 0; i < graphSize; ++i) {
        graph.addVertex(i);
    }
//
    // 添加边
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    graph.addEdge(3, 4);
    graph.addEdge(4, 0);

    // 打印邻接表
    std::cout << "Original Graph's Adjacency List:" << std::endl;
    graph.printAdjList();

    // 打印邻接矩阵
    std::cout << "Original Graph's Adjacency Matrix:" << std::endl;
    graph.printAdjMatrix();

    // 打印边列表
    std::cout << "Original Graph's Edge List:" << std::endl;
    graph.printEdgeList();

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

//lzy部分-前
    // 测试 Floyd
//    Graph g(5);
//    g.addEdge1(0, 1, 3);
//    g.addEdge1(0, 4, 5);
//    g.addEdge1(1, 2, 1);
//    g.addEdge1(1, 3, 2);
//    g.addEdge1(2, 3, 4);
//    g.addEdge1(3, 4, 2);
//
//    floydWarshall(g);

    //测试 Bellman-Ford算法
//    Graph g(4);
//    g.addEdge1(0,1,-1);
//    g.addEdge1(0,2,4);
//    g.addEdge1(1,2,5);
//    g.addEdge1(1,3,2);
//    g.addEdge1(2,3,1);
//
//    bellmanFord(g, 0);
    //测试johnson算法
//    Graph graph(5);
//    graph.addEdge1(0, 1, 3);
//    graph.addEdge1(0, 2, 8);
//    graph.addEdge1(1, 2, 1);
//    graph.addEdge1(1, 3, 7);
//    graph.addEdge1(2, 4, 2);
//    graph.addEdge1(3, 4, 9);
//
//    johnson(graph);
    //测试SPFA算法
//    Graph graph(5);
//    graph.addEdge1(0, 1, 3);
//    graph.addEdge1(0, 2, 8);
//    graph.addEdge1(1, 2, 1);
//    graph.addEdge1(1, 3, 7);
//    graph.addEdge1(2, 4, 2);
//    graph.addEdge1(3, 4, 9);
//
//    int source = 0;
//    std::vector<int> distances = spfa(graph, source);
//
//    std::cout << "Shortest distances from vertex " << source << " using SPFA:" << std::endl;
//    for (int i = 0; i < graph.getSize(); ++i) {
//        if (distances[i] == INT_MAX) {
//            std::cout << "No path from " << source << " to " << i << std::endl;
//        } else {
//            std::cout << "Distance from " << source << " to " << i << ": " << distances[i] << std::endl;
//        }
//    }
    //lzy部分-后


    // 测试 DFS
    std::cout << "DFS starting from vertex 0: ";
    std::vector<bool> visited(graphSize, false);
    DFS(graph, 0, visited);
    std::cout << std::endl;

    // 测试 BFS
    std::cout << "BFS starting from vertex 0: ";
    BFS(graph, 0);
    std::cout << std::endl;

    // 生成有向图
    Graph directedGraph = graph.generateDirectedGraph();

    // 打印有向图的邻接表
    std::cout << "有向图的邻接表:" << std::endl;
    directedGraph.printAdjList();

    // 打印有向图的邻接矩阵
    std::cout << "有向图的邻接矩阵:" << std::endl;
    directedGraph.printAdjMatrix();

    // 测试拓扑排序
    try {
        std::vector<int> topoOrder = topologicalSort(directedGraph);
        std::cout << "基于Kahn's 算法的拓扑排序: ";
        for (int vertex : topoOrder) {
            std::cout << vertex << " ";
        }
        std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
    }

    // 测试基于 DFS 的拓扑排序
    try {
        std::vector<int> topoOrderDFS = topologicalSortDFS(directedGraph);
        std::cout << "基于DFS的拓扑排序（DFS本身无法检测环路的存在，若是存在环路则会出现无法找到拓扑排序的情况。此处存在待优化的地方，有时间可以手动添加环路检测）: ";
        for (int vertex : topoOrderDFS) {
            std::cout << vertex << " ";
        }
        std::cout << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
    }



    //测试kosaraju算法
    Kosaraju(graph);
    //测试割点和桥算法：
    // 查找割点和桥
    std::vector<bool> isAP(graphSize, false);
    std::vector<std::pair<int, int>> bridges;
    findArticulationPointsAndBridges(graph, isAP, bridges);

    // 打印割点
    std::cout << "Articulation Points: ";
    for (int i = 0; i < graphSize; i++) {
        if (isAP[i]) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    // 打印桥
    std::cout << "Bridges: ";
    for (const auto& bridge : bridges) {
        std::cout << "(" << bridge.first << ", " << bridge.second << ") ";
    }
    std::cout << std::endl;


    return 0;
}
