// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/named_parameters.hpp>

#include "dss/dss.hpp"
#include "mpi/communicator.hpp"

namespace {

constexpr std::size_t kStringsPerRank = 1000;
constexpr std::size_t kMinLen         = 3;
constexpr std::size_t kMaxLen         = 50;

std::vector<unsigned char> make_local_input(int rank) {
    std::mt19937 rng{static_cast<std::uint32_t>(0xC0FFEE ^ rank)};
    std::uniform_int_distribution<std::size_t> len_dist{kMinLen, kMaxLen};
    std::uniform_int_distribution<int> char_dist{'a', 'z'};

    std::vector<unsigned char> bytes;
    for (std::size_t i = 0; i < kStringsPerRank; ++i) {
        std::size_t const len = len_dist(rng);
        for (std::size_t j = 0; j < len; ++j) {
            bytes.push_back(static_cast<unsigned char>(char_dist(rng)));
        }
        bytes.push_back(0);
    }
    return bytes;
}

std::vector<std::string> split_nul(std::vector<unsigned char> const& bytes) {
    std::vector<std::string> out;
    auto begin = bytes.begin();
    for (auto it = bytes.begin(); it != bytes.end(); ++it) {
        if (*it == 0) {
            out.emplace_back(begin, it);
            begin = it + 1;
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    kamping::Environment env{argc, argv};
    dss_mehnert::Communicator comm{};

    int const rank      = static_cast<int>(comm.rank());
    int const num_procs = static_cast<int>(comm.size());

    auto local_input = make_local_input(rank);
    auto input_copy  = local_input;  // run_sorter consumes its argument

    auto sorted_local = dss::run_sorter(local_input, comm);

    auto gathered_sorted = comm.gatherv(kamping::send_buf(sorted_local));
    auto gathered_input  = comm.gatherv(kamping::send_buf(input_copy));

    if (rank != 0) {
        return 0;
    }

    auto sorted_strings   = split_nul(gathered_sorted);
    auto expected_strings = split_nul(gathered_input);
    std::sort(expected_strings.begin(), expected_strings.end());

    bool ok = sorted_strings == expected_strings;
    std::cout << "[example] num_procs=" << num_procs
              << " total_strings=" << expected_strings.size()
              << " result=" << (ok ? "OK" : "FAIL") << '\n';

    if (!ok) {
        std::size_t const limit = std::min<std::size_t>(8, expected_strings.size());
        for (std::size_t i = 0; i < limit; ++i) {
            std::string const& got =
                i < sorted_strings.size() ? sorted_strings[i] : std::string{"<missing>"};
            std::cout << "  [" << i << "] expected=\"" << expected_strings[i]
                      << "\" got=\"" << got << "\"\n";
        }
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
