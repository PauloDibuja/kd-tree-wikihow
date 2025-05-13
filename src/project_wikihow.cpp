#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <json/json.h>  // JSON library (e.g. libjsoncpp-dev)
#include <Eigen/Dense>
#include "llama_client.h"
#include <chrono> 
#include "kdtree.cpp"

#define DIMENSION 3

using namespace std::chrono;

struct WikiHowArticle {
    std::string Question;
    std::string Answer;
};


WikiHowArticle parseJSONArray(const std::string& line) {
    std::vector<std::string> result;
    Json::CharReaderBuilder readerBuilder;
    Json::Value root;
    std::string errs;
    std::istringstream iss(line);
    if (!Json::parseFromStream(readerBuilder, iss, &root, &errs)) {
        std::cerr << "Error parsing JSON: " << errs << std::endl;
    }
    WikiHowArticle article = {
        root[0].asString(),
        root[1].asString()
    };
    return article;
}

std::vector<signed long> execute_comparison(std::vector<Point> &user_data, std::vector<Point> &vector_data, std::vector<Point> &temp_user_data, std::vector<Point> &temp_vector_data, int n_rows) {
    
    temp_user_data.clear();
    temp_vector_data.clear();

    // It assumes that temp_user_data and temp_vector_data are not considered as input of the algorithms
    temp_user_data = std::vector<Point>(user_data.begin(), user_data.begin() + n_rows);
    temp_vector_data = std::vector<Point>(vector_data.begin(), vector_data.begin() + n_rows);
    
    
    // Iterative version
    auto start1=high_resolution_clock::now();
    
    std::vector<int> nn_index = find_nearest_neighbor(temp_user_data, temp_vector_data);
    
    auto end1=high_resolution_clock::now();
    
    auto space1 = 0; // Iterative version does not use extra space
    auto search_time=duration_cast<milliseconds>(end1-start1).count();
    
    std::cout << "Time of execution | Iterative version:  " << search_time << "[ms]" << std::endl;
    std::cout << "Space complexity | Iterative version:  " << space1 << "[bytes]" << std::endl;
    
    KDTree tree(temp_vector_data);
    // KDTree version

    auto start2=high_resolution_clock::now();

    std::vector<Point> nn_index_point = tree.kNearestNeighbors(user_data, 1);

    auto end2=high_resolution_clock::now();

    auto search_time2=duration_cast<milliseconds>(end2-start2).count();
    auto space2 = tree.get_memory_usage(); // Tree version uses extra space

    std::cout << "Time of execution | KDTree version:  " << search_time2 << "[ms]" << std::endl;
    std::cout << "Space complexity | KDTree version:  " << space2 << "[bytes]" << std::endl;

    std::cout << "--------------<End Iterative / KD Tree>--------------" << std::endl << std::endl;
    return std::vector<signed long>{search_time, search_time2, space1, space2};
}

// project_wikihow <text_file.jsonl> num_rows step
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <text_file.jsonl> num_rows step <results.csv>" <<  std::endl;
        std::cerr << "It reads first n rows of the text file and tests iterative and kdtree methods with subvectors increasing by <step> " << std::endl;
        std::cerr << "text_file.jsonl [Required] - input file with jsonl format" << std::endl;
        std::cerr << "num_rows [Required] - number of rows to read from the file" << std::endl;
        std::cerr << "step [Required] - number of rows to increase the size of the subvector" << std::endl;
        std::cerr << "results.csv [Optional] - default name" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << argv[1] << std::endl;
        return 1;
    }

    std::string results_file = "results.csv";
    if (argc == 5) {
        results_file = argv[4];
    }
    std::ofstream results(results_file);
    if (!results.is_open()) {
        std::cerr << "Failed to open results file: " << results_file << std::endl;
        return 1;
    }
    results << "n_rows;Time Iterative;Time KDTree;Space Iterative; Space KD Tree" << std::endl;

    int n_rows=atoi(argv[2]);

    std::cout << "Program started!" << std::endl;
    std::string line;
    std::vector<Point> vector_data;
    std::vector<Point> user_data;
    std::vector<std::string> string_data;
    int count = 0;
    std::cout << "--------------<Database>--------------" << std::endl << std::endl;
    std::cout << std::endl;
    while (std::getline(file, line)) {
        //std::cout << "\033[A\33[2K" << std::flush;
        if (line.empty() || count>n_rows) continue;
        WikiHowArticle article = parseJSONArray(line);  
        std::cout << count << ", Input : " << article.Question << " - " << article.Answer << std::endl;
        std::string response = send_embedding_request(article.Answer);
        string_data.push_back(article.Answer);
        extract_embedding(response,vector_data);
        std::string question = send_embedding_request(article.Question);
        extract_embedding(question,user_data);
        ++count;
    }
    file.close();
    std::cout << "--------------<End Database>--------------" << std::endl << std::endl;
    std::cout << std::endl;

    
    
    std::cout << "--------------<Iterative / KD Tree>--------------" << std::endl << std::endl;
    int step = atoi(argv[3]);

    std::vector<Point> temp_user_data;
    std::vector<Point> temp_vector_data;

    for(int i = step; i <= n_rows; i+=step){
        std::cout << "--------------<Step>--------------" << std::endl << std::endl;
        std::cout << "Step: " << i << std::endl;
        auto times = execute_comparison(user_data, vector_data, temp_user_data, temp_vector_data, i);
        results << i << ";" << times[0] << ";" << times[1] << ";" << times[2] << ";" << times[3] << std::endl;
    }

    results.close();

    return 0;
}

