#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <filesystem>
#include "Timing.h"


std::string get_abs_path(const std::string& filepath) {
    try {
        return std::filesystem::absolute(filepath).string();
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error obtaining absolute path for " << filepath << ": " << e.what() << std::endl;
        return filepath;
    }
}

class GameOfLife {
public:
    explicit GameOfLife(const std::string &inputFilename) {
        std::string absInputPath = get_abs_path(inputFilename);
        std::ifstream inputFile(absInputPath);
        if (!inputFile.is_open()) {
            throw std::runtime_error("Cannot open input file: " + absInputPath);
        }

        // Read the dimensions using a comma delimiter
        char delimiter;
        inputFile >> rows >> delimiter >> columns;
        inputFile.ignore();
        grid.resize(rows, std::vector<bool>(columns, false));
        newGrid.resize(rows, std::vector<bool>(columns, false));

        for (int i = 0; i < rows; ++i) {
            std::string line;
            std::getline(inputFile, line);
            for (int j = 0; j < columns; ++j) {
                grid[i][j] = (line[j] == 'x');
            }
        }
    }

    void runGenerations(int generations) {
        for (int gen = 0; gen < generations; ++gen) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < columns; ++j) {
                    int aliveNeighbors = countAliveNeighbors(i, j);
                    if (grid[i][j]) {
                        newGrid[i][j] = (aliveNeighbors == 2 || aliveNeighbors == 3);
                    } else {
                        newGrid[i][j] = (aliveNeighbors == 3);
                    }
                }
            }
            grid.swap(newGrid);
        }
    }

    void save(const std::string& outputFilename) {
        std::string absOutputPath = get_abs_path(outputFilename);
        std::ofstream outputFile(absOutputPath);

        if (!outputFile.is_open()) {
            throw std::runtime_error("Cannot open output file: " + absOutputPath);
        }

        outputFile << rows << "," << columns << "\n";
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                outputFile << (grid[r][c] ? 'x' : '.');
            }
            outputFile << "\n";
        }

        outputFile.close();
    }

private:
    int rows{}, columns{};
    std::vector<std::vector<bool>> grid;
    std::vector<std::vector<bool>> newGrid;

    int countAliveNeighbors(int row, int col) {
        int count = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;
                int newRow = (row + i + rows) % rows;
                int newCol = (col + j + columns) % columns;
                count += grid[newRow][newCol];
            }
        }
        return count;
    }
};

int main(int argc, char* argv[]) {
    std::string inputFilename, outputFilename;
    int generations = 1;
    bool measure = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--load") == 0) {
            inputFilename = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0) {
            outputFilename = argv[++i];
        } else if (strcmp(argv[i], "--generations") == 0) {
            generations = std::stoi(argv[++i]);
        } else if
                (strcmp(argv[i], "--measure") == 0) {
            measure = true;
        } else {
            std::cerr << "Unknown command line parameter: " << argv[i] << std::endl;
            return 1;
        }
    }
    try {
        Timing *timing = Timing::getInstance();

        timing->startSetup();
        GameOfLife game(inputFilename);
        timing->stopSetup();

        timing->startComputation();
        game.runGenerations(generations);
        timing->stopComputation();

        timing->startFinalization();
        game.save(outputFilename);
        timing->stopFinalization();

        if (measure) {
            std::cout << timing->getResults() << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
