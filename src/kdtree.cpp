#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <queue>
#include <cmath>


using namespace Eigen;
using Point = VectorXd;
using Neighbor = std::pair<double, Point>; // {distancia, punto}
using MaxHeap = std::priority_queue<Neighbor>;

struct KDNode {
    Point point;
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;

    KDNode(const Point& pt) : point(pt), left(nullptr), right(nullptr) {}
};

using KDNodePtr = std::unique_ptr<KDNode>;

class KDTree {
public:
    KDTree(const std::vector<Point>& points, int depth = 0) {
        root = build(points, depth);
    }

    KDNodePtr build(std::vector<Point> points, int depth) {
        if (points.empty()) return nullptr;

        int k = points[0].size();
        int axis = depth % k;

        std::sort(points.begin(), points.end(), [axis](const Point& a, const Point& b) {
            return a(axis) < b(axis);
        });

        size_t median_index = points.size() / 2;
        Point median_point = points[median_index];
        //std::cout  << "Median Index: " <<  median_index  << ", Depth: " <<  depth  << ", Point: " << median_point.transpose() << "\n";
        std::vector<Point> left_points(points.begin(), points.begin() + median_index);
        std::vector<Point> right_points(points.begin() + median_index + 1, points.end());

        KDNodePtr node = std::make_unique<KDNode>(median_point);
        node->left = build(left_points, depth + 1);
        node->right = build(right_points, depth + 1);

        return node;
    }

    void printTree(KDNodePtr& node, int depth = 0) {
        if (!node) return;
        std::cout << std::string(depth * 2, ' ') << "Point: " << node->point.transpose() << "\n";
        printTree(node->left, depth + 1);
        printTree(node->right, depth + 1);
    }

    void print(){
        printTree(root);
    }

    double distance_squared(Point& point1,  Point& point2){
        double dist = (point1 - point2).squaredNorm();
        return dist;
    }

    double minSearch( KDNodePtr& node,  Point& target, int depth,std::vector<Neighbor>& knn_neighbors){
        if (!node) return std::numeric_limits<double>::max();;

        int axis = depth % target.size();
        double dist = distance_squared(node->point,target);
        Neighbor neighbor = std::make_pair(dist,node->point);
        knn_neighbors.push_back(neighbor);
        bool goLeft = target(axis) < node->point(axis);
        KDNodePtr& first = goLeft ? node->left : node->right;
        KDNodePtr& second = goLeft ? node->right : node->left;
        //std::cout << "point : " << node->point.transpose()  << ", distance : " << dist << std::endl;
        double new_dist=minSearch(first, target,depth + 1,knn_neighbors);
        double best=std::min(dist,new_dist);        
        return best;
    }

    std::vector<Point> kNearestNeighbors(Point& target,int k){
        std::vector<Neighbor> all_neighbors;
        std::vector<Point> knn_neighbors;
        double min_dist=minSearch(root,target,0,all_neighbors);
        std::sort(all_neighbors.begin(), all_neighbors.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.first < b.first;
        });
        int r=std::min(int(all_neighbors.size()),k);
        for(int i=0;i<r;i++){
            knn_neighbors.push_back(all_neighbors[i].second);
        }
        return knn_neighbors;
    }

    double kNearestNeighbors(Point& target){
        std::vector<Neighbor> all_neighbors;
        double min_dist=minSearch(root,target,0,all_neighbors);
        return min_dist;
    }
    
    std::vector<Point> kNearestNeighbors(std::vector<Point>& user_data,int k){
        std::vector<Point> knn_neighbors;
        Point target= VectorXd::Zero(user_data[0].size());
        for (unsigned int j = 0; j < user_data.size(); ++j) {
            std::vector<Point> neighbors = kNearestNeighbors(user_data[j], k);
            knn_neighbors.insert(knn_neighbors.end(), neighbors.begin(), neighbors.end());
        }
        return knn_neighbors;
    }

    int get_memory_usage() {
        int memory_usage = 0;
        std::queue<KDNodePtr*> queue;
        queue.push(&root);
        while (!queue.empty()) {
            KDNodePtr* node = queue.front();
            queue.pop();
            if (*node) {
                memory_usage += sizeof(**node);
                queue.push(&(*node)->left);
                queue.push(&(*node)->right);
            }
        }
        return memory_usage;
    }

    private:
     KDNodePtr root;
};




