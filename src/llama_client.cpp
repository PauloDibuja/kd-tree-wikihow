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

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* out) {
    size_t totalSize = size * nmemb;
    out->append((char*)contents, totalSize);
    return totalSize;
}

std::string send_embedding_request(const std::string& text) {
    CURL* curl = curl_easy_init();
    std::string response;

    if (curl) {
        std::string json_payload = "{\"content\": \"" + text + "\", \"embedding_only\": true}";
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8080/embedding");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }

    return response;
}

void extract_embedding(const std::string& json_str,std::vector<Point>& database) {
    Json::Value jsonData;
    Json::Reader jsonReader;
    if (jsonReader.parse(json_str, jsonData)){
        auto tokens = jsonData[0]["embedding"];
        //std::cout << "# Tokens  = " << tokens.size() << std::endl;
        for (unsigned int i = 0; i < tokens.size(); ++i) {
            auto embedding = tokens[i];
            //std::cout << "# Embeddings  = " << embedding.size() << std::endl;
            Point embedding_vector(embedding.size());
            for (unsigned int j = 0; j < embedding.size(); ++j) {
                embedding_vector(j) = embedding[j].asFloat();
            }
            database.push_back(embedding_vector);        
        }
    }
    else{
        std::cout << "Could not parse HTTP data as JSON" << std::endl;
    }
}



double cosine_similarity(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2) {
    return vec1.dot(vec2) / (vec1.norm() * vec2.norm());
}

std::vector<int> find_nearest_neighbor(std::vector<Point>& user_prompt,std::vector<Point>& database){
    std::vector<int> nearest_neighbors;
    for (unsigned int i = 0; i < user_prompt.size(); ++i) {
        int max_index=0;
        double max_dist=std::numeric_limits<double>::min();
        for (unsigned int j = 0; j < database.size(); ++j) {
            double dist=cosine_similarity(user_prompt[i],database[j]);
            if (dist>max_dist){
                max_dist=dist;
                max_index=j;
            }
        }
        nearest_neighbors.push_back(max_index);
    }
    return nearest_neighbors;
}


