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



int g[N][N]; //sun Edmonds算法
int main() {
//    // 定义图的大小（用于邻接矩阵）
//    int graphSize = 5;
////
////    // 创建图对象
//    Graph graph(graphSize);
//
//    // 添加顶点（仅用于邻接表）
//    for (int i = 0; i < graphSize; ++i) {
//        graph.addVertex(i);
//    }
////
//    // 添加边
//    graph.addEdge(0, 1);
//    graph.addEdge(1, 2);
//    graph.addEdge(2, 3);
//    graph.addEdge(3, 4);
//    graph.addEdge(4, 0);
//
//    // 打印邻接表
//    std::cout << "Original Graph's Adjacency List:" << std::endl;
//    graph.printAdjList();
//
//    // 打印邻接矩阵
//    std::cout << "Original Graph's Adjacency Matrix:" << std::endl;
//    graph.printAdjMatrix();
//
//    // 打印边列表
//    std::cout << "Original Graph's Edge List:" << std::endl;
//    graph.printEdgeList();

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
//    int V = 4; // 顶点数量
//    std::vector<Edge> edges = {
//            {0, 1, 1},
//            {0, 2, 5},
//            {1, 2, 1},
//            {1, 3, 2},
//            {2, 3, 1},
//            {3,2,2},
//            {2,1,4},
//            {1,0,3}
//    };
//    int root = 0;
//
//    std::vector<Edge> mstEdges;  // 存储最小生成树中的边
//    int mstWeight = edmondsMST(root, V, edges, mstEdges);
//
//    std::cout << "Weight of the MST is " << mstWeight << std::endl;
//
//    std::cout << "Edges in the MST:" << std::endl;
//    for (const auto& edge : mstEdges) {
//        std::cout << edge.u << " -> " << edge.v << " (Weight: " << edge.weight << ")" << std::endl;
//    }


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

    int n, m, root; //n是节点处，m是边数， root是根节点（不能从0开始）
    std::cin >> n >> m >> root;

    // 初始化图的权重为无穷大
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            g[i][j] = INF;

    Edge edges[N];
    int u, v, w;

    // 输入边的信息并处理自环和重边
    for (int i = 0; i < m; ++i) {
        std::cin >> u >> v >> w;
        if (u == v) continue;
        g[u][v] = std::min(g[u][v], w);
    }

    m = 0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (g[i][j] != INF) {
                edges[m].u = i;
                edges[m].v = j;
                edges[m++].weight = g[i][j];
            }
        }
    }

    // 调用 zhuliu 函数并输出结果
    int result = zhuliu(n, m, root, edges);
    std::cout << result << std::endl;







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


//    // 测试 DFS
//    std::cout << "DFS starting from vertex 0: ";
//    std::vector<bool> visited(graphSize, false);
//    DFS(graph, 0, visited);
//    std::cout << std::endl;
//
//    // 测试 BFS
//    std::cout << "BFS starting from vertex 0: ";
//    BFS(graph, 0);
//    std::cout << std::endl;
//
//    // 生成有向图
//    Graph directedGraph = graph.generateDirectedGraph();
//
//    // 打印有向图的邻接表
//    std::cout << "有向图的邻接表:" << std::endl;
//    directedGraph.printAdjList();
//
//    // 打印有向图的邻接矩阵
//    std::cout << "有向图的邻接矩阵:" << std::endl;
//    directedGraph.printAdjMatrix();
//
//    // 测试拓扑排序
//    try {
//        std::vector<int> topoOrder = topologicalSort(directedGraph);
//        std::cout << "基于Kahn's 算法的拓扑排序: ";
//        for (int vertex : topoOrder) {
//            std::cout << vertex << " ";
//        }
//        std::cout << std::endl;
//    } catch (const std::runtime_error& e) {
//        std::cout << e.what() << std::endl;
//    }
//
//    // 测试基于 DFS 的拓扑排序
//    try {
//        std::vector<int> topoOrderDFS = topologicalSortDFS(directedGraph);
//        std::cout << "基于DFS的拓扑排序（DFS本身无法检测环路的存在，若是存在环路则会出现无法找到拓扑排序的情况。此处存在待优化的地方，有时间可以手动添加环路检测）: ";
//        for (int vertex : topoOrderDFS) {
//            std::cout << vertex << " ";
//        }
//        std::cout << std::endl;
//    } catch (const std::runtime_error& e) {
//        std::cout << e.what() << std::endl;
//    }
//
//
//
//    //测试kosaraju算法
//    Kosaraju(graph);
//    //测试割点和桥算法：
//    // 查找割点和桥
//    std::vector<bool> isAP(graphSize, false);
//    std::vector<std::pair<int, int>> bridges;
//    findArticulationPointsAndBridges(graph, isAP, bridges);
//
//    // 打印割点
//    std::cout << "Articulation Points: ";
//    for (int i = 0; i < graphSize; i++) {
//        if (isAP[i]) {
//            std::cout << i << " ";
//        }
//    }
//    std::cout << std::endl;
//
//    // 打印桥
//    std::cout << "Bridges: ";
//    for (const auto& bridge : bridges) {
//        std::cout << "(" << bridge.first << ", " << bridge.second << ") ";
//    }
//    std::cout << std::endl;

    //测试 kruskal 和 prim
    // 创建图并添加边
    // Graph graph(5);
    //
    // graph.weightAddEdge(0, 1, 1);
    // graph.weightAddEdge(0, 2, 4);
    // graph.weightAddEdge(1, 2, 2);
    // graph.weightAddEdge(1, 3, 5);
    // graph.weightAddEdge(2, 3, 1);
    // graph.weightAddEdge(3, 4, 3);

    //测试kruskal  liujun
    // std::vector<std::vector<int>> mst =kruskal(graph);
    // for (const auto& edge : mst) {
    //     std::cout << "Edge: " << edge[0] << " -- " << edge[1] << " Weight: " << edge[2] << std::endl;
    // }
    //测试prim   liujun
    //    prim(graph);

    //测试dmst  liujun
    // std::cout << "The minimum spanning tree is: " << std::endl;
    // std::pair<int, int> result = dmst(graph);
    // std::cout << result.first << "--" << result.second << std::endl;

    //测试 biconnectComponents   liujun
    // Graph graph(5);
    // graph.addEdge(0,1);
    // graph.addEdge(0,2);
    // graph.addEdge(1,2);
    // graph.addEdge(1,3);
    // graph.addEdge(2,3);
    // graph.addEdge(3,4);
    //
    // std::vector<std::vector<int>> results = biconnectComponents(graph);
    // for (const auto& component : results) {
    //     for (int vertex : component) {
    //         std::cout << vertex << " ";
    //     }
    //     std::cout << std::endl;
    // }


//    //tg test start
//    // 创建一个更复杂的无向图
//    Graph undirectedGraph(7);
//    undirectedGraph.addVertex(0);
//    undirectedGraph.addVertex(1);
//    undirectedGraph.addVertex(2);
//    undirectedGraph.addVertex(3);
//    undirectedGraph.addVertex(4);
//    undirectedGraph.addVertex(5);
//    undirectedGraph.addVertex(6);
//
//    // 构造无向图的边（多个环和附加边）
//    undirectedGraph.addEdge_undirected(0, 1);
//    undirectedGraph.addEdge_undirected(1, 2);
//    undirectedGraph.addEdge_undirected(2, 3);
//    undirectedGraph.addEdge_undirected(3, 4);
//    undirectedGraph.addEdge_undirected(4, 5);
//    undirectedGraph.addEdge_undirected(5, 6);
//    undirectedGraph.addEdge_undirected(6, 0); // 大环
//    undirectedGraph.addEdge_undirected(0, 3); // 附加边
//    undirectedGraph.addEdge_undirected(1, 4); // 附加边
//    undirectedGraph.addEdge_undirected(2, 5); // 附加边
//
//    //测试欧拉路径和欧拉回路-无向图
//    std::cout << "Undirected Graph:" << std::endl;
//    auto result1 = findEulerianPathOrCircuit(undirectedGraph, false);
//    if (result1.first == 2) {
//        std::cout << "Eulerian Circuit found: ";
//    } else if (result1.first == 1) {
//        std::cout << "Eulerian Path found: ";
//    } else {
//        std::cout << "No Eulerian Path or Circuit found.";
//    }
//    for (int vertex : result1.second) {
//        std::cout << vertex << " ";
//    }
//    std::cout << std::endl;
//
//
//    //测试哈密顿路径-无向图
//    std::vector<int> hamiltonianPathUndirected = findHamiltonianPathOrCircuit(undirectedGraph, false, false);
//    if (!hamiltonianPathUndirected.empty()) {
//        std::cout << "Hamiltonian Path: ";
//        for (int vertex : hamiltonianPathUndirected) {
//            std::cout << vertex << " ";
//        }
//        std::cout<<std::endl;
//    } else {
//        std::cout << "No Hamiltonian Path found in the undirected graph." <<std::endl;
//    }
//
//    //测试哈密顿回路-无向图
//    std::vector<int> hamiltonianCircuitUndirected = findHamiltonianPathOrCircuit(undirectedGraph, false, true);
//    if (!hamiltonianCircuitUndirected.empty()) {
//        std::cout << "Hamiltonian Circuit: ";
//        for (int vertex : hamiltonianCircuitUndirected) {
//            std::cout << vertex << " ";
//        }
//        std::cout<<std::endl;
//    } else {
//        std::cout << "No Hamiltonian Circuit found in the undirected graph."<<std::endl;
//    }
//
//    std::cout<<std::endl;
//
//
//
//    // 创建一个更复杂的有向图
//    Graph directedGraph(7);
//    directedGraph.addVertex(0);
//    directedGraph.addVertex(1);
//    directedGraph.addVertex(2);
//    directedGraph.addVertex(3);
//    directedGraph.addVertex(4);
//    directedGraph.addVertex(5);
//    directedGraph.addVertex(6);
//
//    // 构造有向图的边（形成一个复杂的路径和环）
//    directedGraph.addEdge_directed(0, 1);
//    directedGraph.addEdge_directed(1, 2);
//    directedGraph.addEdge_directed(2, 3);
//    directedGraph.addEdge_directed(3, 4);
//    directedGraph.addEdge_directed(4, 5);
//    directedGraph.addEdge_directed(5, 6);
//    directedGraph.addEdge_directed(6, 0); // 大环
//    directedGraph.addEdge_directed(0, 3); // 附加边
//    directedGraph.addEdge_directed(1, 4); // 附加边
//    directedGraph.addEdge_directed(2, 5); // 附加边
//
//    //测试欧拉路径和欧拉回路-有向图
//    std::cout << "Directed Graph:" << std::endl;
//    auto result3 = findEulerianPathOrCircuit(directedGraph, true);
//    if (result3.first == 2) {
//        std::cout << "Eulerian Circuit found: ";
//    } else if (result3.first == 1) {
//        std::cout << "Eulerian Path found: ";
//    } else {
//        std::cout << "No Eulerian Path or Circuit found.";
//    }
//    for (int vertex : result3.second) {
//        std::cout << vertex << " ";
//    }
//    std::cout << std::endl;
//
//    //测试哈密顿回路-有向图
//    std::vector<int> hamiltonianPathDirected = findHamiltonianPathOrCircuit(directedGraph, true, false);
//    if (!hamiltonianPathDirected.empty()) {
//        std::cout << "Hamiltonian Path: ";
//        for (int vertex : hamiltonianPathDirected) {
//            std::cout << vertex << " ";
//        }
//        std::cout<<std::endl;
//    } else {
//        std::cout << "No Hamiltonian Path found in the directed graph."<<std::endl;
//    }
//
//    //测试哈密顿回路-有向图
//    std::vector<int> hamiltonianCircuitDirected = findHamiltonianPathOrCircuit(directedGraph, true, true);
//    if (!hamiltonianCircuitDirected.empty()) {
//        std::cout << "Hamiltonian Circuit: ";
//        for (int vertex : hamiltonianCircuitDirected) {
//            std::cout << vertex << " ";
//        }
//        std::cout<<std::endl;
//    } else {
//        std::cout << "No Hamiltonian Circuit found in the directed graph."<<std::endl;
//    }
//
//
//    std::cout << std::endl;
//    //测试树的遍历
//    Graph tree(7);
//    tree.addVertex(0);
//    tree.addVertex(1);
//    tree.addVertex(2);
//    tree.addVertex(3);
//    tree.addVertex(4);
//    tree.addVertex(5);
//    tree.addVertex(6);
//
//    tree.addEdge_directed(0, 1); // 左子树
//    tree.addEdge_directed(0, 2); // 右子树
//    tree.addEdge_directed(1, 3); // 左子树的左子树
//    tree.addEdge_directed(1, 4); // 左子树的右子树
//    tree.addEdge_directed(2, 5); // 右子树的左子树
//    tree.addEdge_directed(2, 6); // 右子树的右子树
//    //测试前序遍历
//    std::cout << "Preorder Traversal of the tree starting from vertex 0:" ;
//    preorderTraversal(tree, 0);
//    //测试中序遍历
//    std::cout << "Inorder Traversal of the tree starting from vertex 0:" ;
//    inorderTraversal(tree, 0);
//    //测试后序遍历
//    std::cout << "Postorder Traversal of the tree starting from vertex 0:" ;
//    postorderTraversal(tree, 0);
//    //测试层序遍历
//    std::cout << "LevelOrder Traversal of the tree starting from vertex 0:" ;
//    levelOrderTraversal(tree, 0);
//
//
//    std::cout << std::endl;
//    // 创建有向图
//    Graph graph(4);
//
//    graph.addVertex(0);
//    graph.addVertex(1);
//    graph.addVertex(2);
//    graph.addVertex(3);
//
//    graph.addEdge_directed(0, 1);
//    graph.addEdge_directed(1, 2);
//    graph.addEdge_directed(2, 3);
//
//    // 测试最大匹配和最小路径覆盖
//    int maxMatching = maxBipartiteMatching(graph.constructBipartiteGraph());
//    std::cout << "Maximum Bipartite Matching: " << maxMatching << std::endl;
//
//    int minCover = minPathCover(graph);
//    std::cout << "Minimum Path Cover: " << minCover << std::endl;
//
//
//    //tg test end
 // 无向图测试
//    std::cout << "Undirected Graph Test:" << std::endl;
//    Graph undirectedGraph(6);
//    undirectedGraph.addEdge(0, 1);
//    undirectedGraph.addEdge(0, 2);
//    undirectedGraph.addEdge(1, 3);
//    undirectedGraph.addEdge(2, 3);
//    undirectedGraph.addEdge(3, 4);
//    undirectedGraph.addEdge(4, 5);
//
//    std::cout << "\nUndirected Graph - Adjacency List:" << std::endl;
//    undirectedGraph.printAdjList();
//
//    std::cout << "\nUndirected Graph - Adjacency Matrix:" << std::endl;
//    undirectedGraph.printAdjMatrix();
//
//    std::cout << "\nUndirected Graph - Edge List:" << std::endl;
//    undirectedGraph.printEdgeList();
//
//    // 有向图测试
//    std::cout << "\nDirected Graph Test:" << std::endl;
//    // Graph directedGraph(6);
//    // directedGraph.addDirectedEdge(0, 1);
//    // directedGraph.addDirectedEdge(0, 2);
//    // directedGraph.addDirectedEdge(1, 3);
//    // directedGraph.addDirectedEdge(2, 3);
//    // directedGraph.addDirectedEdge(3, 4);
//    // directedGraph.addDirectedEdge(4, 5);
//
//    std::cout << "\nDirected Graph - Adjacency List:" << std::endl;
//    directedGraph.printAdjList();
//
//    std::cout << "\nDirected Graph - Adjacency Matrix:" << std::endl;
//    directedGraph.printAdjMatrix();
//
//    std::cout << "\nDirected Graph - Edge List:" << std::endl;
//    directedGraph.printEdgeList();
//
//    // 带权有向图测试
//    std::cout << "\nWeighted Directed Graph Test:" << std::endl;
//    Graph weightedDirectedGraph(6);
//    weightedDirectedGraph.addWeightedDirectedEdge(0, 1, 2.5);
//    weightedDirectedGraph.addWeightedDirectedEdge(0, 2, 1.5);
//    weightedDirectedGraph.addWeightedDirectedEdge(1, 3, 1.2);
//    weightedDirectedGraph.addWeightedDirectedEdge(2, 3, 0.7);
//    weightedDirectedGraph.addWeightedDirectedEdge(3, 4, 1.3);
//    weightedDirectedGraph.addWeightedDirectedEdge(4, 5, 2.8);
//
//    std::cout << "\nWeighted Directed Graph - Adjacency List:" << std::endl;
//    weightedDirectedGraph.printWeightAdjList();
//
//    std::cout << "\nDijkstra's Shortest Paths from Vertex 0 in Weighted Directed Graph:" << std::endl;
//    auto distances = Dijkstra(weightedDirectedGraph, 0);
//    for (int i = 0; i < distances.size(); ++i) {
//        std::cout << "Distance to vertex " << i << ": " << distances[i] << std::endl;
//    }
//
//    std::cout << "\nTarjan's Strongly Connected Components in Directed Graph:" << std::endl;
//    auto sccs = tarjanSCC(directedGraph);
//    for (const auto& scc : sccs) {
//        std::cout << "SCC: ";
//        for (int vertex : scc) {
//            std::cout << vertex << " ";
//        }
//        std::cout << std::endl;
//    }
//
//       // 测试随机无向图生成
//    std::cout << "\nRandom Undirected Graph:" << std::endl;
//    Graph randomUndirectedGraph = generateRandomGraph(5, 7);
//    randomUndirectedGraph.printAdjList();
//
//    // 测试随机有向图生成
//    std::cout << "\nRandom Directed Graph:" << std::endl;
//    Graph randomDirectedGraph = generateRandomDirectedGraph(5, 7);
//    randomDirectedGraph.printAdjList();
//
//    // 测试蒙特卡洛算法
//    std::cout << "\nMonte Carlo Estimation of Pi with 1000000 samples: " << std::endl;
//    double piEstimate = monteCarloPi(1000000);
//    std::cout << "Estimated Pi value: " << piEstimate << std::endl;
//
//    // 测试随机游走算法
//    std::cout << "\nRandom Walk on Directed Graph starting from vertex 0 for 10 steps:" << std::endl;
//    auto walkPath = randomWalk(directedGraph, 0, 10);
//    std::cout << "Random Walk Path: ";
//    for (int vertex : walkPath) {
//        std::cout << vertex << " ";
//    }
//    std::cout << std::endl;

    return 0;
}
