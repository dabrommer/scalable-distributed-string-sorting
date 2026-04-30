#pragma once

#include "executables/common_cli.hpp"
#include "mpi/communicator.hpp"
#include "mpi/alltoall_combined.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/redistribution.hpp"
#include "strings/stringcontainer.hpp"

// The used types mimic a call to distributed_sorter with the parameters -i 5 -a 0 -l -p -k 2 -I
template <typename CharType, typename Communicator>
std::vector<CharType> run_sorter(std::vector<CharType>& to_sort, Communicator const& comm) {
    using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
    using PartitionPolicy = dss_mehnert::MergeSortPartitionPolicy<CharType>;

    using RedistributionPolicy = dss_mehnert::redistribution::NoRedistribution<dss_mehnert::mpi::Communicator>;
    RedistributionPolicy redistribution{};

    using Subcommunicators = RedistributionPolicy::Subcommunicators;
    Subcommunicators comms{comm};

    using dss_mehnert::mpi::AlltoallStringsConfig;

    constexpr AlltoallStringsConfig config{
        .alltoall_kind = dss_mehnert::mpi::AlltoallvCombinedKind::native,
        .compress_lcps = true,
        .compress_prefixes = true,
    };

    using Config = std::integral_constant<AlltoallStringsConfig, config>;
    constexpr auto alltoall_config = Config();

    SamplerArgs sampler_args{.sample_chars = false,
                             .sample_indexed = true,
                             .sample_random = false,
                             .sampling_factor = 2};

    SplitterSorter splitter_sorter = SplitterSorter::Sequential;

    using MergeSort = dss_mehnert::sorter::
        DistributedMergeSort<alltoall_config, RedistributionPolicy, PartitionPolicy>;

    MergeSort merge_sort{
        dss_mehnert::init_partition_policy<CharType, PartitionPolicy>(
            sampler_args,
            splitter_sorter
        ),
        std::move(redistribution)
    };

    dss_schimek::StringLcpContainer<StringSet> container(std::move(to_sort));

    merge_sort.sort(container, comms);

    // Retrieve sorted strings
    auto sorted_strings = std::move(container.release_strings());
    std::vector<CharType> to_return;

    // Print sorted strings
    for (auto i : sorted_strings) {
        auto s = i.getChars();
        auto length = i.getLength();

        // Given s and length, append to to_return
        to_return.insert(to_return.end(), s, s + length);
        to_return.push_back(0);

    }

    return to_return;
}

