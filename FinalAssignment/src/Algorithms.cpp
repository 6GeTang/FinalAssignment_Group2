#include "Algorithms.h"
#include "Graph.h"
#include <iostream>
#include <queue>
#include <stack>
#include <vector>
#include <limits>

#include <unordered_map>  //sun
#include <cmath>  //sun
#include <limits>//sun
#include <set>//sun
#include <algorithm> //sun
#include <numeric>  // 用于 std::accumulate //sun

#include<random>
#include<ctime>

struct Node {   //sun  A*
    int id;
    double g;  // 从起点到当前节点的代价
    double h;  // 启发式估计代价
    Node* parent;

    Node(int id, double g = 0.0, double h = 0.0, Node* parent = nullptr)
            : id(id), g(g), h(h), parent(parent) {}

    double f() const { return g + h; }

    bool operator>(const Node& other) const { return f() > other.f(); }
};

// 曼哈顿距离或欧几里得距离作为启发式函数
double heuristic(int id1, int id2) {   //sun A*
    // 这里假设节点ID可以用作坐标，实际应用中可能需要根据具体问题调整启发式函数
    // 示例代码假设节点ID在图中是有意义的坐标，这里使用简单的绝对差作为启发式函数
    return abs(id1 - id2);  // 可以根据实际情况改为更合适的启发式函数
}



// A*算法
void aStarSearch(const Graph& graph, int startId, int destId) {  //sun A*
    int size = graph.getSize();

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openList;
    std::unordered_map<int, Node*> allNodes;  // 存储所有创建的节点
    std::set<int> closedSet; // 存储已经处理的节点

    Node* startNode = new Node(startId, 0.0, heuristic(startId, destId));
    openList.push(*startNode);
    allNodes[startId] = startNode;

    while (!openList.empty()) {
        Node current = openList.top();
        openList.pop();

        // 如果找到目标节点
        if (current.id == destId) {
            std::cout << "Path found: ";
            Node* node = &current;
            std::vector<int> path;  // 用于存储路径节点ID
            while (node != nullptr) {
                path.push_back(node->id);
                node = node->parent;
            }
            reverse(path.begin(), path.end());  // 将路径反转为正序
            for (int id : path) {
                std::cout << id << " ";
            }
            std::cout << std::endl;
            return;
        }


        closedSet.insert(current.id);

        // 遍历当前节点的邻居
        for (const auto& neighbor : graph.getNeighbors(current.id)) {
            int neighborId = neighbor.first;
            double weight = neighbor.second;

            if (closedSet.find(neighborId) != closedSet.end()) {
                continue; // 忽略已经处理的节点
            }

            double newG = current.g + weight;
            double newH = heuristic(neighborId, destId);

            Node* neighborNode = new Node(neighborId, newG, newH, allNodes[current.id]);

            int hash = neighborId;
            if (allNodes.find(hash) == allNodes.end() || newG < allNodes[hash]->g) {
                openList.push(*neighborNode);
                allNodes[hash] = neighborNode;
            }
        }
    }

    std::cout << "Path not found" << std::endl;
}

UnionFind::UnionFind(int n) : parent(n), rank(n, 0) {  //sun 并查集
    for (int i = 0; i < n; ++i) {
        parent[i] = i;  // 每个节点的初始父节点指向自己
    }
}

// 查找操作，带路径压缩
int UnionFind::find(int p) {      //sun 并查集
    if (parent[p] != p) {
        parent[p] = find(parent[p]);  // 递归路径压缩
    }
    return parent[p];
}

// 合并操作，按秩合并
void UnionFind::unite(int p, int q) {   //sun 并查集
    int rootP = find(p);
    int rootQ = find(q);

    if (rootP != rootQ) {
        if (rank[rootP] > rank[rootQ]) {
            parent[rootQ] = rootP;
        } else if (rank[rootP] < rank[rootQ]) {
            parent[rootP] = rootQ;
        } else {
            parent[rootQ] = rootP;
            rank[rootP]++;
        }
    }
}

// 判断两个元素是否在同一集合中
bool UnionFind::connected(int p, int q) {    //sun 并查集
    return find(p) == find(q);
}




// 查找顶点的集合编号
int find(int vertex, std::vector<int>& parent) {  //sun Borůvka算法
    if (parent[vertex] != vertex) {
        parent[vertex] = find(parent[vertex], parent);
    }
    return parent[vertex];
}

// 合并两个集合
void unionSets(int u, int v, std::vector<int>& parent, std::vector<int>& rank) {  //sun Borůvka算法
    int root_u = find(u, parent);
    int root_v = find(v, parent);

    if (root_u != root_v) {
        if (rank[root_u] > rank[root_v]) {
            parent[root_v] = root_u;
        } else if (rank[root_u] < rank[root_v]) {
            parent[root_u] = root_v;
        } else {
            parent[root_v] = root_u;
            rank[root_u]++;
        }
    }
}

// Borůvka算法主函数
void boruvkaMST(int V, int E, std::vector<Edge>& edges) {   //sun Borůvka算法
    std::vector<int> parent(V);
    std::vector<int> rank(V, 0);
    std::vector<int> cheapest(V, -1);

    // 初始化每个顶点为其自身的父节点
    for (int i = 0; i < V; i++) {
        parent[i] = i;
    }

    int numComponents = V;
    int mstWeight = 0;

    while (numComponents > 1) {
        // 初始化cheapest数组
        fill(cheapest.begin(), cheapest.end(), -1);

        // 遍历每一条边，找出每个组件的最小边
        for (int i = 0; i < E; i++) {
            int u = edges[i].u;
            int v = edges[i].v;
            int set_u = find(u, parent);
            int set_v = find(v, parent);
            //遍历所有的边，u和v是边的两个端点。set_u和set_v是顶点u和v所在的组件（通过find函数查找）。
            //
            //如果u和v属于不同的组件（set_u != set_v），则考虑将这条边作为候选边。
            // 如果当前组件还没有候选边（cheapest[set_u] == -1），或者找到的这条边比之前的候选边更小（edges[i].weight < edges[cheapest[set_u]].weight），
            // 则更新cheapest数组。
            if (set_u != set_v) {
                if (cheapest[set_u] == -1 || edges[cheapest[set_u]].weight > edges[i].weight) {
                    cheapest[set_u] = i;
                }

                if (cheapest[set_v] == -1 || edges[cheapest[set_v]].weight > edges[i].weight) {
                    cheapest[set_v] = i;
                }
            }
        }

        // 把找到的最小边加入MST
        for (int i = 0; i < V; i++) {
            if (cheapest[i] != -1) {
                int u = edges[cheapest[i]].u;
                int v = edges[cheapest[i]].v;
                int set_u = find(u, parent);
                int set_v = find(v, parent);

                if (set_u != set_v) {
                    mstWeight += edges[cheapest[i]].weight;
                    std::cout << "Edge " << u << " - " << v << " included in MST with weight " << edges[cheapest[i]].weight << std::endl;
                    unionSets(set_u, set_v, parent, rank);
                    numComponents--;
                }
            }
        }
    }

    std::cout << "Weight of MST is " << mstWeight << std::endl;
}



int edmondsMST(int root, int V, std::vector<Edge>& edges, std::vector<Edge>& mstEdges) {  //sun Edmonds算法
    std::vector<int> parent(V, -1);
    std::vector<int> minEdge(V, INT_MAX);
    std::vector<int> selectedEdge(V, -1);

    // Step 1: 选择每个顶点的最小入边
    for (auto& edge : edges) {
        if (edge.u != edge.v && edge.weight < minEdge[edge.v]) {
            parent[edge.v] = edge.u;
            minEdge[edge.v] = edge.weight;
            selectedEdge[edge.v] = &edge - &edges[0];  // 保存边的索引
        }
    }

    parent[root] = root; // 根节点没有父节点
    minEdge[root] = 0;

    // Step 2: 检查是否有环并处理
    std::vector<int> visited(V, -1);
    std::vector<int> cycle(V, -1);
    int cycleCount = 0;

    for (int v = 0; v < V; ++v) {
        if (v == root) continue;

        int u = v;
        while (u != root && visited[u] == -1) {
            visited[u] = v;
            u = parent[u];
        }

        if (u != root && visited[u] == v) { // 找到了一个环
            while (cycle[u] == -1) {
                cycle[u] = cycleCount;
                u = parent[u];
            }
            cycleCount++;
        }
    }

    if (cycleCount == 0) {
        // 没有环，直接返回最小生成树的权重
        int mstWeight = 0;
        for (int v = 0; v < V; ++v) {
            if (v != root && parent[v] != -1) {
                mstWeight += minEdge[v];
                mstEdges.push_back(edges[selectedEdge[v]]);
            }
        }
        return mstWeight;
    }

    // Step 3: 收缩环并构建新图
    std::vector<Edge> newEdges;
    std::vector<int> cycleWeight(cycleCount, 0);

    for (int i = 0; i < edges.size(); ++i) {
        int u = edges[i].u;
        int v = edges[i].v;
        int weight = edges[i].weight;

        if (cycle[v] != -1) {
            cycleWeight[cycle[v]] += minEdge[v];
            u = cycle[u];
            v = cycle[v];
            if (u != v) {
                newEdges.push_back({u, v, weight - minEdge[v]});
            }
        } else if (cycle[u] != -1) {
            u = cycle[u];
            newEdges.push_back({u, v, weight});
        } else {
            newEdges.push_back({u, v, weight});
        }
    }

    // Step 4: 递归处理收缩后的图
    int newRoot = cycle[root] == -1 ? root : cycle[root];
    int cycleMSTWeight = edmondsMST(newRoot, cycleCount + (cycle[root] == -1), newEdges, mstEdges);

    // Step 5: 还原环中的边
    return cycleMSTWeight + accumulate(cycleWeight.begin(), cycleWeight.end(), 0);
}

int inCost[N], pre[N], id[N], vis[N];  //sun Edmonds算法

int zhuliu(int n, int m, int root, Edge edges[]) { //sun Edmonds算法
    int res = 0;
    while (true) {
        for (int i = 1; i <= n; ++i) inCost[i] = INF, id[i] = vis[i] = -1;
        for (int i = 0; i < m; ++i) {
            int u = edges[i].u, v = edges[i].v, w = edges[i].weight;
            if (u != v && w < inCost[v]) {
                inCost[v] = w;
                pre[v] = u;
            }
        }
        for (int i = 1; i <= n; ++i) {
            if (i != root && inCost[i] == INF) {
                return -1;
            }
        }
        inCost[root] = 0;
        int tn = 0;
        for (int i = 1; i <= n; ++i) {
            res += inCost[i];
            int v = i;
            while (vis[v] != i && id[v] == -1 && v != root) {
                vis[v] = i;
                v = pre[v];
            }
            if (v != root && id[v] == -1) {
                id[v] = ++tn;
                for (int u = pre[v]; u != v; u = pre[u]) id[u] = tn;
            }
        }
        if (tn == 0) break;
        for (int i = 1; i <= n; ++i) {
            if (id[i] == -1) {
                id[i] = ++tn;
            }
        }
        int i = 0;
        while (i < m) {
            int vv = edges[i].v;
            edges[i].u = id[edges[i].u], edges[i].v = id[edges[i].v];
            if (edges[i].u != edges[i].v) {
                edges[i++].weight -= inCost[vv];
            } else {
                std::swap(edges[i], edges[--m]);
            }
        }
        n = tn;
        root = id[root];
    }
    return res;
}



//lzy部分-前
// 实现 Floyd-Warshall 算法
void floydWarshall(const Graph& graph) {
    int size = graph.getSize();
    const std::vector<std::vector<int>>& adjMatrix = graph.getAdjMatrix();
    std::vector<std::vector<int>> dist(size, std::vector<int>(size, INT_MAX));

    // 初始化距离矩阵
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i == j) {
                dist[i][j] = 0;
            } else if (adjMatrix[i][j]!= 0) {
                dist[i][j] = adjMatrix[i][j];
            }
        }
    }

    // Floyd-Warshall 算法核心部分
    for (int k = 0; k < size; ++k) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (dist[i][k]!= INT_MAX && dist[k][j]!= INT_MAX && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    // 输出最短路径距离矩阵
    std::cout << "Shortest path distances:\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (dist[i][j] == INT_MAX) {
                std::cout << "INF ";
            } else {
                std::cout << dist[i][j] << " ";
            }
        }
        std::cout << std::endl;
    }
}

// Bellman-Ford 算法实现
// 定义一个结构体来表示边
//struct Edge {
//    int source;
//    int destination;
//    int weight;
//};
void bellmanFord(const Graph& graph, int source) {
    int size = graph.getSize();
    std::vector<int> distance(size, INT_MAX);
    distance[source] = 0;

    // 松弛操作
    for (int i = 0; i < size - 1; i++) {
        for (const auto& edge : graph.getEdgeList()) {
            int u = edge.first;
            int v = edge.second;
            int weight = graph.getAdjMatrix()[u][v];
            if (distance[u]!= INT_MAX && distance[u] + weight < distance[v]) {
                distance[v] = distance[u] + weight;
            }
        }
    }

    // 检查负权回路
    for (const auto& edge : graph.getEdgeList()) {
        int u = edge.first;
        int v = edge.second;
        int weight = graph.getAdjMatrix()[u][v];
        if (distance[u]!= INT_MAX && distance[u] + weight < distance[v]) {
            std::cout << "there is a negative weight loop" << std::endl;
            return;
        }
    }

    // 输出从源点到各个顶点的最短距离
    std::cout << "from " << source << " : " << std::endl;
    for (int i = 0; i < size; i++) {
        if (distance[i] == INT_MAX) {
            std::cout << "to " << i << " : inaccessible" << std::endl;
        } else {
            std::cout << "to " << i << " : " << distance[i] << std::endl;
        }
    }
}

// Johnson算法:高效求解多源最短路径问题（利用Dijkstra算法和Bellman-ford算法作为子程序实现的）
// 使用 Bellman-Ford2 算法计算单源最短路径
std::vector<int> bellmanFord2(const Graph& graph, int source) {
    int size = graph.getSize();
    std::vector<int> distance(size, INT_MAX);
    distance[source] = 0;

    for (int i = 0; i < size - 1; ++i) {
        for (const auto& edge : graph.getEdgeList()) {
            int u = edge.first;
            int v = edge.second;
            int weight = graph.getAdjMatrix()[u][v];
            if (distance[u]!= INT_MAX && distance[u] + weight < distance[v]) {
                distance[v] = distance[u] + weight;
            }
        }
    }

    // 检查负权回路
    for (const auto& edge : graph.getEdgeList()) {
        int u = edge.first;
        int v = edge.second;
        int weight = graph.getAdjMatrix()[u][v];
        if (distance[u]!= INT_MAX && distance[u] + weight < distance[v]) {
            std::cerr << "Graph contains negative-weight cycle." << std::endl;
            return std::vector<int>();
        }
    }

    return distance;
}
// 使用 Dijkstra2 算法计算单源最短路径
std::vector<int> dijkstra2(const Graph& graph, int source, const std::vector<int>& potential) {
    int size = graph.getSize();
    std::vector<int> distance(size, INT_MAX);
    distance[source] = 0;

    std::vector<bool> visited(size, false);

    for (int count = 0; count < size - 1; ++count) {
        int minDistance = INT_MAX;
        int minVertex = -1;

        for (int v = 0; v < size; ++v) {
            if (!visited[v] && distance[v] < minDistance) {
                minDistance = distance[v];
                minVertex = v;
            }
        }

        if (minVertex == -1) {
            break;
        }

        visited[minVertex] = true;

        for (int v = 0; v < size; ++v) {
            int weight = graph.getAdjMatrix()[minVertex][v];
            if (weight!= INT_MAX && distance[minVertex]!= INT_MAX && distance[minVertex] + weight + potential[minVertex] - potential[v] < distance[v]) {
                distance[v] = distance[minVertex] + weight + potential[minVertex] - potential[v];
            }
        }
    }

    return distance;
}
// Johnson 算法求解多源最短路
void johnson(const Graph& graph) {
    int size = graph.getSize();
    Graph newGraph(size + 1);

    // 添加新的顶点 0 并与其他所有顶点相连，边权为 0
    for (int i = 0; i < size; ++i) {
        newGraph.addEdge1(size, i, 0);
    }

    std::vector<int> h = bellmanFord2(newGraph, size);
    if (h.empty()) {
        return;
    }

    std::vector<std::vector<int>> shortestDistances(size, std::vector<int>(size, INT_MAX));

    for (int source = 0; source < size; ++source) {
        std::vector<int> potential(size);
        for (int i = 0; i < size; ++i) {
            potential[i] = h[i];
        }
        std::vector<int> distances = dijkstra2(graph, source, potential);
        for (int destination = 0; destination < size; ++destination) {
            shortestDistances[source][destination] = distances[destination];
        }
    }

    // 输出结果
    std::cout << "All-Pairs Shortest Paths using Johnson's algorithm:" << std::endl;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (shortestDistances[i][j] == INT_MAX) {
                std::cout << "No path from " << i << " to " << j << std::endl;
            } else {
                std::cout << "Shortest distance from " << i << " to " << j << ": " << shortestDistances[i][j] << std::endl;
            }
        }
    }
}

// 使用 SPFA 算法求解单源最短路径(Bellman-Ford算法的优化版)
std::vector<int> spfa(const Graph& graph, int source) {
    int size = graph.getSize();
    std::vector<int> distance(size, INT_MAX);
    distance[source] = 0;

    std::vector<bool> inQueue(size, false);
    std::queue<int> q;
    q.push(source);
    inQueue[source] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        inQueue[u] = false;

        for (int v : graph.getAdjList().at(u)) {
            int weight = graph.getAdjMatrix()[u][v];
            if (distance[u]!= INT_MAX && distance[u] + weight < distance[v]) {
                distance[v] = distance[u] + weight;
                if (!inQueue[v]) {
                    q.push(v);
                    inQueue[v] = true;
                }
            }
        }
    }

    return distance;
}

//lzy部分-后



//冯碧川的改动开始
//拓扑排序开始 begin:
// 1.拓扑排序的方法 (基于 Kahn's 算法)
std::vector<int> topologicalSort(const Graph& graph) {
    int size = graph.getSize();
    std::vector<int> inDegree(size, 0);
    std::vector<int> topoOrder;
    std::queue<int> zeroInDegreeQueue;

    // 计算每个顶点的入度
    for (const auto& pair : graph.getAdjList()) {
        for (int neighbor : pair.second) {
            inDegree[neighbor]++;
        }
    }

    // 将所有入度为0的顶点加入队列
    for (int i = 0; i < size; ++i) {
        if (inDegree[i] == 0) {
            zeroInDegreeQueue.push(i);
        }
    }

    // 处理队列中的顶点
    while (!zeroInDegreeQueue.empty()) {
        int vertex = zeroInDegreeQueue.front();
        zeroInDegreeQueue.pop();
        topoOrder.push_back(vertex);

        // 减少邻居顶点的入度
        for (int neighbor : graph.getAdjList().at(vertex)) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                zeroInDegreeQueue.push(neighbor);
            }
        }
    }

    // 检查是否存在环
    if (topoOrder.size() != size) {
        throw std::runtime_error("此图中存在环，不可以进行拓扑排序");
    }

    return topoOrder;
}

void dfsUtil(int v, const Graph& graph, std::vector<bool>& visited, std::stack<int>& Stack) {
    visited[v] = true;

    // 遍历所有的邻居
    for (int neighbor : graph.getAdjList().at(v)) {
        if (!visited[neighbor]) {
            dfsUtil(neighbor, graph, visited, Stack);
        }
    }

    // 当前顶点访问完毕，压入栈中
    Stack.push(v);
}
// 2.DFS实现拓扑排序
std::vector<int> topologicalSortDFS(const Graph& graph) {
    int size = graph.getSize();
    std::stack<int> Stack;
    std::vector<bool> visited(size, false);

    // 对所有未访问的节点执行 DFS
    for (int i = 0; i < size; i++) {
        if (!visited[i]) {
            dfsUtil(i, graph, visited, Stack);
        }
    }

    // 将栈中的元素弹出，形成拓扑排序
    std::vector<int> topoOrder;
    while (!Stack.empty()) {
        topoOrder.push_back(Stack.top());
        Stack.pop();
    }

    // 检查图中是否有环
    if (topoOrder.size() != size) {
        throw std::runtime_error("The graph has a cycle, topological sort is not possible.");
    }

    return topoOrder;
}

//拓扑排序结束

//Kosaraju算法的实现begin:
// 第一次DFS，记录顶点的完成顺序
void fillOrder(const Graph& graph, int v, std::vector<bool>& visited, std::stack<int>& Stack) {
    visited[v] = true;

    // 访问所有邻居
    for (int neighbor : graph.getAdjList().at(v)) {
        if (!visited[neighbor]) {
            fillOrder(graph, neighbor, visited, Stack);
        }
    }

    // 当前顶点完成后压入栈中
    Stack.push(v);
}

// 第二次DFS，遍历强连通分量
void DFSUtil(const Graph& transposedGraph, int v, std::vector<bool>& visited) {
    visited[v] = true;
    std::cout << v << " ";

    for (int neighbor : transposedGraph.getAdjList().at(v)) {
        if (!visited[neighbor]) {
            DFSUtil(transposedGraph, neighbor, visited);
        }
    }
}

// Kosaraju算法实现
void Kosaraju(const Graph& graph) {
    int size = graph.getSize();
    std::stack<int> Stack;
    std::vector<bool> visited(size, false);

    // 第一次DFS，按完成时间顺序将顶点放入栈中
    for (int i = 0; i < size; i++) {
        if (!visited[i]) {
            fillOrder(graph, i, visited, Stack);
        }
    }

    // 转置图
    Graph transposedGraph = graph.generateTransposedGraph();  // 请确认这里是生成转置图，而不是简单的有向图

    // 清除访问记录
    visited.assign(size, false);

    // 处理所有顶点，按栈中顺序
    std::cout << "强连通分量如下：" << std::endl;
    while (!Stack.empty()) {
        int v = Stack.top();
        Stack.pop();
        // 在转置图上执行DFS
        if (!visited[v]) {
            DFSUtil(transposedGraph, v, visited);
            std::cout << std::endl;
        }
    }
}

//Kosaraju 算法实现结束 end

//图的割点和桥begin

void findAPandBridgesUtil(const Graph& graph, int u, std::vector<bool>& visited,
                          std::vector<int>& disc, std::vector<int>& low,
                          std::vector<int>& parent, std::vector<bool>& isAP,
                          std::vector<std::pair<int, int>>& bridges, int& time) {
    int children = 0;
    visited[u] = true;

    disc[u] = low[u] = ++time;

    for (int v : graph.getAdjList().at(u)) {
        if (!visited[v]) {
            children++;
            parent[v] = u;
            findAPandBridgesUtil(graph, v, visited, disc, low, parent, isAP, bridges, time);

            // Check if the subtree rooted at v has a connection back to one of ancestors of u
            low[u] = std::min(low[u], low[v]);

            // (1) u is root and has more than one child in DFS tree
            if (parent[u] == -1 && children > 1)
                isAP[u] = true;

            // (2) u is not root and low value of one of its children is more
            // than discovery value of u
            if (parent[u] != -1 && low[v] >= disc[u])
                isAP[u] = true;

            // (3) The condition for bridge
            if (low[v] > disc[u])
                bridges.emplace_back(u, v);

        } else if (v != parent[u]) {
            // Update low[u] for back edge
            low[u] = std::min(low[u], disc[v]);
        }
    }
}


void findArticulationPointsAndBridges(const Graph& graph, std::vector<bool>& isAP, std::vector<std::pair<int, int>>& bridges) {
    int V = graph.getSize();
    std::vector<bool> visited(V, false);
    std::vector<int> disc(V, -1), low(V, -1), parent(V, -1);
    int time = 0;

    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            findAPandBridgesUtil(graph, i, visited, disc, low, parent, isAP, bridges, time);
        }
    }
}

//图的割点和桥end

// 深度优先搜索 (DFS) 实现
void DFS(const Graph& graph, int startVertex, std::vector<bool>& visited) {
    std::stack<int> stack;
    stack.push(startVertex);

    while (!stack.empty()) {
        int vertex = stack.top();
        stack.pop();

        if (!visited[vertex]) {
            std::cout << vertex << " ";
            visited[vertex] = true;
        }

        // 为了保证顺序，逆序遍历邻居
        const auto& neighbors = graph.getAdjList().at(vertex);
        for (auto it = neighbors.rbegin(); it != neighbors.rend(); ++it) {
            if (!visited[*it]) {
                stack.push(*it);
            }
        }
    }
}

// 广度优先搜索 (BFS) 实现
void BFS(const Graph& graph, int startVertex) {
    std::queue<int> queue;
    std::vector<bool> visited(graph.getSize(), false);

    queue.push(startVertex);
    visited[startVertex] = true;

    while (!queue.empty()) {
        int vertex = queue.front();
        queue.pop();
        std::cout << vertex << " ";

        for (const auto& neighbor : graph.getAdjList().at(vertex)) {
            if (!visited[neighbor]) {
                queue.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }
}

//冯碧川的改动结束

// 定义一个比较函数，用于比较两个子向量的第三个元素  //liujun
bool compareThirdElement(const std::vector<int>& a, const std::vector<int>& b) {
    return a[2] < b[2];
}


//Kruskal算法
std::vector<std::vector<int>> kruskal(const Graph& graph) {  //liujun
    //整一个边集
    std::vector<std::vector<int>> edges = graph.getWeightadjEdgeList();
    //把边集从小到大排序
    std::sort(edges.begin(), edges.end(), compareThirdElement);
     //初始化并查集，把点整进去
     UnionFind unionFind(graph.getSize());
    //初始化最小生成树的边列表
    std::vector<std::vector<int>> mst;
//    遍历边集，看看即将加进去的边两边的点是不是一个集合,如果不是，就加进去，否则就跳过
    for (const auto& edge : edges) {
        int u = edge[0];
        int v = edge[1];
        int weight = edge[2];

        // 如果两个点不在同一个集合中，则将这两个点连接起来
        if (!unionFind.connected(u, v)) {
            //std::cout << "Add edge: " << u << " - " << v << " with weight: " << weight << std::endl;
            unionFind.unite(u, v);
            std::vector<int> edge = {u, v, weight};
            mst.push_back(edge);
        }
    }
    return mst;
    // std::cout << "Minimum Spanning Tree:" << std::endl;
    // for (const auto& edge : mst) {
    //     std::cout << edge[0] << " - " << edge[1] << " with weight: " << edge[2] << std::endl;
    // }
}


//prim算法
void prim(const Graph &graph) {  //liujun
    //初始化一个向量，用来记录哪个顶点已访问
    std::vector<int> record;
    record.reserve(graph.getSize());
    for(int i : record) {
        record[i] = 0;
    }
    //选一个初始顶点，作为以访问
    record[0] = 1;
    //整一个边集
    std::vector<std::vector<int>> edges = graph.getWeightadjEdgeList();
    //把边集从小到大排序
    std::sort(edges.begin(), edges.end(), compareThirdElement);
    //定义一个最小生成树
    std::vector<std::pair<int,int>> mst;
    for(int i = 0; i < edges.size(); i++) {
        if(record[edges[i][0]] == 1 && record[edges[i][1]] == 1) {
            continue;
        } else {
            mst.push_back(std::make_pair(edges[i][0], edges[i][1]));
            if(record[edges[i][0]] == 0) {
                record[edges[i][0]] = 1;
            } else {
                record[edges[i][1]] = 1;
            }
        }
    }
    std::cout << "Minimum Spanning Tree:" << std::endl;
    for (const auto& edge : mst) {
        std::cout << edge.first << " - " << edge.second << std::endl;
    }
}

// 辅助函数，用于DFS遍历 //liujun   beconnect Components
void DFSBC(Graph& graph, int vertex, std::vector<bool>& visited) {
    visited[vertex] = true; // 标记顶点为已访问
    std::stack<int> stack;
    stack.push(vertex); // 将顶点压入栈中

    while (!stack.empty()) {
        int currentVertex = stack.top();
        stack.pop(); // 弹出顶点

        for (int neighbor : graph.getAdjList().at(currentVertex)) {
            if (!visited[neighbor]) {
                visited[neighbor] = true; // 标记邻居顶点为已访问
                stack.push(neighbor); // 将邻居顶点压入栈中
            }
        }
    }
}



// Biconnect Components算法实现
std::vector<std::vector<int>> biconnectComponents(Graph& graph) {
    std::vector<bool> visited(graph.getSize(), false); // 访问数组
    std::vector<std::vector<int>> components; // 存储所有连通分量

    for (int i = 0; i < graph.getSize(); i++) {
        if (!visited[i]) {
            std::vector<int> component;
            DFS(graph, i, visited); // 进行DFS遍历
            components.push_back(component); // 将连通分量添加到结果中
        }
    }

    return components;
}

// DMST算法实现
std::pair<int, int> dmst(const Graph& graph) {    //liujun
    std::vector<std::vector<int>> mst = kruskal(graph);
    UnionFind unionFind(graph.getSize());
    std::vector<bool> visited(graph.getSize(), false);

    for (int i = 0; i < graph.getSize(); ++i) {
        visited[i] = true;
    }

    std::vector<std::vector<int>> cutEdges;
    int resultVectex =-1;

    for (const auto& edge : mst) {
        int x = edge[0];
        int y = edge[1];
        int xRoot = unionFind.find(x);
        int yRoot = unionFind.find(y);

        if (xRoot != yRoot) {
            visited[x] = false;
            visited[y] = false;
            cutEdges.push_back(edge);
            unionFind.unite(xRoot, yRoot);
        }
    }

    int minCutWeight = INT_MAX;
    for (const auto& edge : cutEdges) {
        int weight = edge[2];
        int vec = edge[0];
        if (weight < minCutWeight) {
            minCutWeight = weight;
            resultVectex = vec;
        }
    }

    return std::make_pair(resultVectex, minCutWeight);
}

//tg
// 欧拉路径与欧拉回路
bool isEulerian(const Graph& graph, bool isDirected, int& type) {
    if (isDirected) {
        // 对有向图的处理
        std::vector<int> inDegree(graph.getSize(), 0);
        std::vector<int> outDegree(graph.getSize(), 0);

        for (int u = 0; u < graph.getSize(); ++u) {
            for (int v : graph.getAdjList().at(u)) {
                outDegree[u]++;
                inDegree[v]++;
            }
        }

        int oddIn = 0, oddOut = 0;
        for (int i = 0; i < graph.getSize(); ++i) {
            if (outDegree[i] - inDegree[i] == 1) oddOut++;
            else if (inDegree[i] - outDegree[i] == 1) oddIn++;
            else if (inDegree[i] != outDegree[i]) {
                type = 0;  // 无欧拉路径或回路
                return false;
            }
        }

        if (oddIn == 0 && oddOut == 0) {
            type = 2;  // 存在欧拉回路
            return true;
        }
        if (oddIn == 1 && oddOut == 1) {
            type = 1;  // 存在欧拉路径但不存在欧拉回路
            return true;
        }
        type = 0;
        return false;
    } else {
        // 对无向图的处理
        int odd = 0;
        for (const auto& pair : graph.getAdjList()) {
            if (pair.second.size() % 2 != 0) {
                odd++;
            }
        }

        if (odd == 0) {
            type = 2;  // 存在欧拉回路
            return true;
        } else if (odd == 2) {
            type = 1;  // 存在欧拉路径但不存在欧拉回路
            return true;
        } else {
            type = 0;  // 无欧拉路径或回路
            return false;
        }
    }
}

std::pair<int, std::vector<int>> findEulerianPathOrCircuit(Graph& graph, bool isDirected) {
    std::vector<int> path;
    int type = 0;
    if (!isEulerian(graph, isDirected, type)) {
        return {type, path}; // 返回空路径，表示没有欧拉路径或回路
    }

    std::stack<int> stack;
    std::vector<std::vector<int>> adjMatrix = graph.getAdjMatrix();
    int curr = 0;

    // 寻找起始节点（针对欧拉路径的情况）
    if (isDirected) {
        std::vector<int> inDegree(graph.getSize(), 0);
        std::vector<int> outDegree(graph.getSize(), 0);

        for (int u = 0; u < graph.getSize(); ++u) {
            for (int v : graph.getAdjList().at(u)) {
                outDegree[u]++;
                inDegree[v]++;
            }
        }

        for (int i = 0; i < graph.getSize(); ++i) {
            if (outDegree[i] - inDegree[i] == 1) {
                curr = i; // 从出度比入度多1的节点开始
                break;
            }
        }
    } else {
        for (int i = 0; i < graph.getSize(); ++i) {
            if (graph.getAdjList().at(i).size() % 2 != 0) {
                curr = i; // 从一个度数为奇数的节点开始
                break;
            }
        }
    }

    stack.push(curr);

    while (!stack.empty()) {
        int v = curr;
        if (std::count(adjMatrix[v].begin(), adjMatrix[v].end(), 1) == 0) {
            path.push_back(v);
            curr = stack.top();
            stack.pop();
        } else {
            for (int i = 0; i < graph.getSize(); ++i) {
                if (adjMatrix[v][i] == 1) {
                    stack.push(v);
                    adjMatrix[v][i] = 0;
                    if (!isDirected) {
                        adjMatrix[i][v] = 0; // 对无向图，双向移除
                    }
                    curr = i;
                    break;
                }
            }
        }
    }

    std::reverse(path.begin(), path.end());
    return {type, path};  // 返回路径和类型
}


// 哈密顿路径与哈密顿回路
bool isHamiltonianUtil(const Graph& graph, std::vector<int>& path, std::vector<bool>& visited, int pos, bool isDirected, bool findCircuit) {
    if (pos == graph.getSize()) {
        if (findCircuit) {
            // 寻找哈密顿回路：检查是否能回到起点
            if (!isDirected) {
                if (std::find(graph.getAdjList().at(path[pos - 1]).begin(), graph.getAdjList().at(path[pos - 1]).end(), path[0]) != graph.getAdjList().at(path[pos - 1]).end()) {
                    path.push_back(path[0]);
                    return true;
                }
            } else {
                for (int neighbor : graph.getAdjList().at(path[pos - 1])) {
                    if (neighbor == path[0]) {
                        path.push_back(path[0]);
                        return true;
                    }
                }
            }
            return false;
        } else {
            // 寻找哈密顿路径：路径已构建完毕
            return true;
        }
    }

    for (int v = 0; v < graph.getSize(); ++v) {
        if (!visited[v] && std::find(graph.getAdjList().at(path[pos - 1]).begin(), graph.getAdjList().at(path[pos - 1]).end(), v) != graph.getAdjList().at(path[pos - 1]).end()) {
            path[pos] = v;
            visited[v] = true;

            if (isHamiltonianUtil(graph, path, visited, pos + 1, isDirected, findCircuit)) {
                return true;
            }

            visited[v] = false;
        }
    }

    return false;
}

std::vector<int> findHamiltonianPathOrCircuit(const Graph& graph, bool isDirected, bool findCircuit) {
    for (int startVertex = 0; startVertex < graph.getSize(); ++startVertex) {
        std::vector<int> path(graph.getSize(), -1);
        std::vector<bool> visited(graph.getSize(), false);

        path[0] = startVertex; // 从当前顶点开始
        visited[startVertex] = true;

        if (isHamiltonianUtil(graph, path, visited, 1, isDirected, findCircuit)) {
            return path;
        }
    }

    return {}; // 无哈密顿路径或回路
}


// 树的前序遍历（Preorder Traversal）
void preorderTraversalUtil(const Graph& graph, int vertex, std::vector<bool>& visited) {
    visited[vertex] = true;
    std::cout << vertex << " ";

    for (int neighbor : graph.getAdjList().at(vertex)) {
        if (!visited[neighbor]) {
            preorderTraversalUtil(graph, neighbor, visited);
        }
    }
}

void preorderTraversal(const Graph& graph, int startVertex) {
    std::vector<bool> visited(graph.getSize(), false);
    preorderTraversalUtil(graph, startVertex, visited);
    std::cout << std::endl;
}

// 树的中序遍历（Inorder Traversal）
void inorderTraversalUtil(const Graph& graph, int vertex, std::vector<bool>& visited) {
    visited[vertex] = true; // 标记当前节点为已访问
    auto neighbors = graph.getAdjList().at(vertex);

    // 处理左子树（假设第一个邻居是左子树）
    if (!neighbors.empty() && !visited[neighbors[0]]) {
        inorderTraversalUtil(graph, neighbors[0], visited);
    }

    // 访问当前节点
    std::cout << vertex << " ";

    // 处理右子树（假设第二个邻居是右子树）
    if (neighbors.size() > 1 && !visited[neighbors[1]]) {
        inorderTraversalUtil(graph, neighbors[1], visited);
    }
}

void inorderTraversal(const Graph& graph, int startVertex) {
    std::vector<bool> visited(graph.getSize(), false); // 初始化访问标志
    inorderTraversalUtil(graph, startVertex, visited); // 开始递归
    std::cout << std::endl;
}



// 树的后序遍历（Postorder Traversal）
void postorderTraversalUtil(const Graph& graph, int vertex, std::vector<bool>& visited) {
    visited[vertex] = true;

    for (int neighbor : graph.getAdjList().at(vertex)) {
        if (!visited[neighbor]) {
            postorderTraversalUtil(graph, neighbor, visited);
        }
    }

    std::cout << vertex << " ";
}

void postorderTraversal(const Graph& graph, int startVertex) {
    std::vector<bool> visited(graph.getSize(), false);
    postorderTraversalUtil(graph, startVertex, visited);
    std::cout << std::endl;
}

// 树的层序遍历（Level Order Traversal）
void levelOrderTraversal(const Graph& graph, int startVertex) {
    std::vector<bool> visited(graph.getSize(), false);
    std::queue<int> queue;

    visited[startVertex] = true;
    queue.push(startVertex);

    while (!queue.empty()) {
        int vertex = queue.front();
        std::cout << vertex << " ";
        queue.pop();

        for (int neighbor : graph.getAdjList().at(vertex)) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }

    std::cout << std::endl;
}


// 匈牙利算法实现最大匹配
bool bpm(const Graph& bipartiteGraph, int u, std::vector<bool>& visited, std::vector<int>& matchR) {
    int n = bipartiteGraph.getSize() / 2;

    if (bipartiteGraph.getAdjList().find(u) == bipartiteGraph.getAdjList().end()) {
        return false;  // 如果 u 没有邻居，直接返回 false
    }

    for (int v : bipartiteGraph.getAdjList().at(u)) {
        v -= n;  // Adjust index for the second partition
        if (!visited[v]) {
            visited[v] = true;
            if (matchR[v] < 0 || bpm(bipartiteGraph, matchR[v], visited, matchR)) {
                matchR[v] = u;
                return true;
            }
        }
    }
    return false;
}


// 计算最大匹配
int maxBipartiteMatching(const Graph& bipartiteGraph) {
    int n = bipartiteGraph.getSize() / 2;
    std::vector<int> matchR(n, -1);  // Match array for second partition

    int result = 0;
    for (int u = 0; u < n; ++u) {
        std::vector<bool> visited(n, false);
        if (bpm(bipartiteGraph, u, visited, matchR)) {
            result++;
        }
    }
    return result;
}

// 计算最小路径覆盖
int minPathCover(const Graph& graph) {
    Graph bipartiteGraph = graph.constructBipartiteGraph();
    int maxMatching = maxBipartiteMatching(bipartiteGraph);
    return graph.getSize() - maxMatching;
}
//tg end

std::vector<double> Dijkstra(const Graph& graph, int startId) {
    int size = graph.getSize();
    std::vector<double> dist(size, std::numeric_limits<double>::max()); // 使用double类型
    dist[startId] = 0.0;

    using pdi = std::pair<double, int>; // 距离和节点的对
    std::priority_queue<pdi, std::vector<pdi>, std::greater<pdi>> pq;
    pq.push({0.0, startId});

    while (!pq.empty()) {
        double currentDist = pq.top().first;
        int currentId = pq.top().second;
        pq.pop();

        if (currentDist > dist[currentId]) {
            continue;
        }

        for (const auto& neighbor : graph.getNeighbors(currentId)) {
            int neighborId = neighbor.first;
            double weight = neighbor.second; // 保持浮点数

            if (dist[currentId] + weight < dist[neighborId]) {
                dist[neighborId] = dist[currentId] + weight;
                pq.push({dist[neighborId], neighborId});
            }
        }
    }

    return dist;
}
namespace {
    void tarjanDFS(int u, int& index, const std::map<int, std::vector<int>>& adjList, std::vector<int>& indices,
                   std::vector<int>& lowLinks, std::vector<bool>& onStack, std::vector<int>& stack,
                   std::vector<std::vector<int>>& sccs) {
        indices[u] = lowLinks[u] = index++;
        stack.push_back(u);
        onStack[u] = true;

        // 确保在访问adjList.at(u)之前检查键u是否存在于adjList中
        if (adjList.find(u) != adjList.end()) {  // 新增检查
            for (int v : adjList.at(u)) {
                if (indices[v] == -1) {
                    tarjanDFS(v, index, adjList, indices, lowLinks, onStack, stack, sccs);
                    lowLinks[u] = std::min(lowLinks[u], lowLinks[v]);
                } else if (onStack[v]) {
                    lowLinks[u] = std::min(lowLinks[u], indices[v]);
                }
            }
        }

        if (lowLinks[u] == indices[u]) {
            std::vector<int> scc;
            int v;
            do {
                v = stack.back();
                stack.pop_back();
                onStack[v] = false;
                scc.push_back(v);
            } while (u != v);
            sccs.push_back(scc);
        }
    }
}

std::vector<std::vector<int>> tarjanSCC(const Graph& graph) {
    int size = graph.getSize();
    std::vector<int> indices(size, -1);
    std::vector<int> lowLinks(size, -1);
    std::vector<bool> onStack(size, false);
    std::vector<int> stack;
    std::vector<std::vector<int>> sccs;

    int index = 0;
    for (int i = 0; i < size; ++i) {
        if (indices[i] == -1) {
            tarjanDFS(i, index, graph.getAdjList(), indices, lowLinks, onStack, stack, sccs);
        }
    }

    return sccs;
}



Graph generateRandomDirectedGraph(int numVertices, int numEdges) {
    Graph directedGraph(numVertices);  // 创建一个有向图对象
    srand(time(0));  // 使用当前时间作为随机数种子

    for (int i = 0; i < numEdges; ++i) {
        int u = rand() % numVertices;
        int v = rand() % numVertices;

        // 确保我们不会创建自环（从一个顶点指向它自己的边）
        while (u == v) {
            v = rand() % numVertices;
        }

        directedGraph.addDirectedEdge(u, v);  // 添加有向边
    }

    return directedGraph;
}

// 蒙特卡洛算法实现 (计算圆周率 Pi 的近似值)
double monteCarloPi(int numSamples) {  //Ding 蒙特卡洛算法
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    int insideCircle = 0;
    for (int i = 0; i < numSamples; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        if (x * x + y * y <= 1.0) {
            insideCircle++;
        }
    }

    return (4.0 * insideCircle) / numSamples;
}

// 随机游走算法实现
std::vector<int> randomWalk(const Graph& graph, int startVertex, int steps) {  //Ding 随机游走算法
    std::vector<int> walk;
    walk.push_back(startVertex);

    std::random_device rd;
    std::mt19937 gen(rd());

    int currentVertex = startVertex;
    for (int i = 0; i < steps; ++i) {
        if (graph.getAdjList().find(currentVertex) == graph.getAdjList().end()) {
            std::cout << "Vertex " << currentVertex << " has no neighbors. Ending walk." << std::endl;
            break;
        }
        const auto& neighbors = graph.getAdjList().at(currentVertex); // 确保使用正确的邻接表

        std::cout << "Current Vertex: " << currentVertex << " has " << neighbors.size() << " neighbors." << std::endl;
        if (neighbors.empty()) {
            std::cout << "No neighbors found. Ending walk." << std::endl;
            break;  // If no neighbors, end the walk
        }

        std::uniform_int_distribution<> dist(0, neighbors.size() - 1);
        currentVertex = neighbors[dist(gen)];
        walk.push_back(currentVertex);
        std::cout << "Next Vertex: " << currentVertex << std::endl;
    }

    return walk;
}

Graph generateRandomGraph(int numVertices, int numEdges) {
    Graph graph(numVertices);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertexDist(0, numVertices - 1);
    std::uniform_int_distribution<> weightDist(1, 10);

    std::set<std::pair<int, int>> addedEdges;  // To avoid duplicate edges

    // Manually count the number of edges
    int edgeCount = 0;

    while (edgeCount < numEdges) {
        int u = vertexDist(gen);
        int v = vertexDist(gen);
        if (u == v || addedEdges.count({u, v}) || addedEdges.count({v, u})) {
            continue;  // Avoid self-loops and duplicate edges
        }

        int weight = weightDist(gen);
        graph.addDirectedEdge(u, v);  // Add the edge to the graph
        addedEdges.insert({u, v});
        edgeCount++;
    }

    return graph;
}