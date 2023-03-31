#include <omp.h>
#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <filesystem>
#include "Timing.h"


std::string get_abs_path(const std::string &file) {
    try {
        return std::filesystem::absolute(file).string();
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Error obtaining absolute path for " << file << ": " << e.what() << std::endl;
        return file;
    }
}

class GameOfLife {
public:
    explicit GameOfLife(const std::string &filename) {
        std::string path = get_abs_path(filename);
        std::ifstream inputFile(path);

        if (!inputFile.is_open()) {
            throw std::runtime_error("Cannot open input file: " + path);
        }

        char delimiter;
        inputFile >> rows >> delimiter >> columns;
        inputFile.ignore();

        grid = boost::dynamic_bitset<>(rows * columns);
        newGrid = boost::dynamic_bitset<>(rows * columns);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                char ch;
                inputFile.get(ch);
                setGrid(grid, i, j, (ch == 'x'));
            }
            inputFile.ignore(); // ignore the newline character
        }
    }

    void runGenerations(int generations) {
        for (int gen = 0; gen < generations; ++gen) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < columns; ++j) {
                    int alive = countAliveNeighbors(i, j);
                    if (atGrid(grid, i, j)) {
                        setGrid(newGrid, i, j, (alive == 2 || alive == 3));
                    } else {
                        setGrid(newGrid, i, j, (alive == 3));
                    }
                }
            }
            grid.swap(newGrid);
        }
    }

    void runGenerationsParallel(int generations, int threads) {
        omp_set_num_threads(threads);

        for (int gen = 0; gen < generations; ++gen) {
#pragma omp parallel for collapse(2) default(none) shared(grid, newGrid)
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < columns; ++j) {
                    int alive = countAliveNeighbors(i, j);
                    if (atGrid(grid, i, j)) {
                        setGrid(newGrid, i, j, (alive == 2 || alive == 3));
                    } else {
                        setGrid(newGrid, i, j, (alive == 3));
                    }
                }
            }
#pragma omp barrier
            grid.swap(newGrid);
        }
    }

    void save(const std::string &filename) {
        std::string path = get_abs_path(filename);
        std::ofstream outputFile(path);

        if (!outputFile.is_open()) {
            throw std::runtime_error("Cannot open output file: " + path);
        }

        outputFile << rows << "," << columns << "\n";
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                outputFile << (atGrid(grid, r, c) ? 'x' : '.');
            }
            outputFile << "\n";
        }

        outputFile.close();
    }

private:
    int rows{}, columns{};
    boost::dynamic_bitset<> grid;
    boost::dynamic_bitset<> newGrid;

    [[nodiscard]] bool atGrid(const boost::dynamic_bitset<> &g, int row, int col) const {
        return g[row * columns + col];
    }

    void setGrid(boost::dynamic_bitset<> &g, int row, int col, bool value) const {
        g[row * columns + col] = value;
    }

    [[nodiscard]] int countAliveNeighbors(int row, int col) const {
        int count = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0) continue;
                int newRow = row + i;
                int newCol = col + j;

                if (newRow < 0) newRow += rows;
                else if (newRow >= rows) newRow -= rows;

                if (newCol < 0) newCol += columns;
                else if (newCol >= columns) newCol -= columns;

                count += atGrid(grid, newRow, newCol);
            }
        }
        return count;
    }

};

int main(int argc, char *argv[]) {
    std::string input, output, mode = "seq";
    int generations = 1, threads = 1;
    bool measure = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--load") == 0) {
            input = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--generations") == 0) {
            generations = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--measure") == 0) {
            measure = true;
        } else if (strcmp(argv[i], "--mode") == 0) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0) {
            threads = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown command line parameter: " << argv[i] << std::endl;
            return 1;
        }
    }

    try {
        Timing *timing = Timing::getInstance();

        timing->startSetup();
        GameOfLife game(input);
        timing->stopSetup();

        timing->startComputation();
        if (mode == "seq") {
            game.runGenerations(generations);
        } else if (mode == "omp") {
            game.runGenerationsParallel(generations, threads);
        } else {
            std::cerr << "Invalid mode: " << mode << std::endl;
            return 1;
        }
        timing->stopComputation();

        timing->startFinalization();
        game.save(output);
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
