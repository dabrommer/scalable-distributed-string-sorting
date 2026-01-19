#include "executables/common_cli.hpp"
#include "mpi/communicator.hpp"
#include "mpi/alltoall_combined.hpp"
#include "strings/stringcontainer.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/redistribution.hpp"


// The used types mimic a call to distributed_sorter with the parameters -i 5 -a 0 -l -p -k 2 -I
template<typename CharType, typename Communicator>
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

    // print sorted strings
    for (auto i : sorted_strings) {
        auto s = i.getChars();
        auto length = i.getLength();

        // given s and length, append to to_return
        to_return.insert(to_return.end(), s, s + length);
        to_return.push_back(0);

    }

    return to_return;

}
/*
int main() {
    kamping::Environment env;
    dss_mehnert::Communicator comm;

    std::vector<unsigned char> to_sort{'z', 'x', 0, 'a', 'b', 'c', 'd', 'e', 0, 'f', 'g', 'h', 0};
    auto result = run_sorter(to_sort, comm);

    std::string print = "PE: " + std::to_string(comm.rank()) + " ";
    for (auto c : result) {
        print += c == 0 ? ' ' : c;

    }
    std::cout << print << std::endl;
    return 0;
}*/