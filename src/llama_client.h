#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <curl/curl.h>
#include <json/json.h>  // JSON library (e.g. libjsoncpp-dev)
#include <Eigen/Dense>

using Point = Eigen::VectorXd;
using Neighbor = std::pair<double, Point>; // {distancia, punto}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* out); 
std::string send_embedding_request(const std::string& text);
void extract_embedding(const std::string& json_str,std::vector<Point>& database); 
double cosine_similarity(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2); 
std::vector<int> find_nearest_neighbor(std::vector<Point>& user_prompt,std::vector<Point>& database);
