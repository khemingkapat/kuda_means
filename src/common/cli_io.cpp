#include "kmeans_common.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace kuda {

namespace {

std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(ch);
            break;
        }
    }
    return out;
}

std::string csv_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char ch : s) {
        if (ch == '"') {
            out += "\"\"";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('"');
    return out;
}

std::string format_json_double(double v, bool fixed_6 = true) {
    if (!std::isfinite(v)) {
        return "null";
    }
    std::ostringstream oss;
    if (fixed_6) {
        oss << std::fixed << std::setprecision(6) << v;
    } else {
        oss << std::setprecision(9) << v;
    }
    return oss.str();
}

bool parse_int_strict(const char *str, int &out) {
    if (!str || *str == '\0') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const long val = std::strtol(str, &end, 10);
    if (errno != 0 || end == str || *end != '\0') {
        return false;
    }
    if (val < std::numeric_limits<int>::min() || val > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(val);
    return true;
}

bool parse_uint32_strict(const char *str, uint32_t &out) {
    if (!str || *str == '\0' || *str == '-') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long val = std::strtoul(str, &end, 10);
    if (errno != 0 || end == str || *end != '\0') {
        return false;
    }
    if (val > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = static_cast<uint32_t>(val);
    return true;
}

bool parse_float_strict(const char *str, float &out) {
    if (!str || *str == '\0') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const float val = std::strtof(str, &end);
    if (errno != 0 || end == str || *end != '\0' || !std::isfinite(val)) {
        return false;
    }
    out = val;
    return true;
}

} // namespace

void print_usage(const char *prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  -n, --n <int>          Number of points (default: 100000)\n"
              << "  -d, --d <int>          Dimensionality (default: 16)\n"
              << "  -k, --k <int>          Number of clusters (default: 8)\n"
              << "  --iters <int>          Max Lloyd iterations (default: 50)\n"
              << "  --eps <float>          Convergence threshold epsilon >= 0 (default: 1e-4)\n"
              << "  --seed <uint>          RNG seed (default: 42)\n"
              << "  --warmup <int>         Warmup runs >= 0 (default: 1)\n"
              << "  --runs <int>           Timed benchmark runs >= 1 (default: 5)\n"
              << "  --version <str>        Implementation version (default: cpu)\n"
              << "  --no-verify            Skip V0 CPU verification on GPU runs\n"
              << "  --git-commit <str>     Override git commit provenance string\n"
              << "  --json <path>          Write structured JSON result to <path>\n"
              << "  --csv <path>           Append CSV row to <path>\n"
              << "  -h, --help             Show this help message\n";
}

CliParseStatus parse_cli_args(int argc, char **argv, KMeansConfig &cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_val = [&](const std::string &flag) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "Error: missing value for " << flag << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return CliParseStatus::Help;
        } else if (arg == "-n" || arg == "--n") {
            const char *v = need_val(arg);
            if (!v || !parse_int_strict(v, cfg.n_points)) {
                std::cerr << "Error: invalid integer for " << arg << "\n";
                return CliParseStatus::Error;
            }
        } else if (arg == "-d" || arg == "--d") {
            const char *v = need_val(arg);
            if (!v || !parse_int_strict(v, cfg.dim)) {
                std::cerr << "Error: invalid integer for " << arg << "\n";
                return CliParseStatus::Error;
            }
        } else if (arg == "-k" || arg == "--k") {
            const char *v = need_val(arg);
            if (!v || !parse_int_strict(v, cfg.k_clusters)) {
                std::cerr << "Error: invalid integer for " << arg << "\n";
                return CliParseStatus::Error;
            }
        } else if (arg == "--iters" || arg == "--max-iters") {
            const char *v = need_val(arg);
            if (!v || !parse_int_strict(v, cfg.max_iters)) {
                std::cerr << "Error: invalid integer for " << arg << "\n";
                return CliParseStatus::Error;
            }
        } else if (arg == "--eps" || arg == "--epsilon") {
            const char *v = need_val(arg);
            if (!v || !parse_float_strict(v, cfg.epsilon)) {
                std::cerr << "Error: invalid float for " << arg << "\n";
                return CliParseStatus::Error;
            }
        } else if (arg == "--seed") {
            const char *v = need_val(arg);
            if (!v || !parse_uint32_strict(v, cfg.seed)) {
                std::cerr << "Error: invalid unsigned integer for " << arg << "\n";
                return CliParseStatus::Error;
            }
        } else if (arg == "--warmup") {
            const char *v = need_val(arg);
            if (!v || !parse_int_strict(v, cfg.warmup_runs)) {
                std::cerr << "Error: invalid integer for " << arg << "\n";
                return CliParseStatus::Error;
            }
        } else if (arg == "--runs") {
            const char *v = need_val(arg);
            if (!v || !parse_int_strict(v, cfg.bench_runs)) {
                std::cerr << "Error: invalid integer for " << arg << "\n";
                return CliParseStatus::Error;
            }
        } else if (arg == "--version") {
            const char *v = need_val(arg);
            if (!v)
                return CliParseStatus::Error;
            cfg.version = v;
        } else if (arg == "--no-verify") {
            cfg.verify = false;
        } else if (arg == "--git-commit") {
            const char *v = need_val(arg);
            if (!v)
                return CliParseStatus::Error;
            cfg.git_commit = v;
        } else if (arg == "--json" || arg == "--json-out") {
            const char *v = need_val(arg);
            if (!v)
                return CliParseStatus::Error;
            cfg.json_out = v;
        } else if (arg == "--csv" || arg == "--csv-out") {
            const char *v = need_val(arg);
            if (!v)
                return CliParseStatus::Error;
            cfg.csv_out = v;
        } else {
            std::cerr << "Error: unknown argument '" << arg << "'\n";
            print_usage(argv[0]);
            return CliParseStatus::Error;
        }
    }

    if (cfg.n_points <= 0 || cfg.dim <= 0 || cfg.k_clusters <= 0 || cfg.max_iters <= 0 ||
        cfg.warmup_runs < 0 || cfg.bench_runs <= 0 || cfg.epsilon < 0.0f ||
        !std::isfinite(cfg.epsilon)) {
        std::cerr << "Error: invalid numeric configuration parameter.\n";
        return CliParseStatus::Error;
    }
    return CliParseStatus::Ok;
}

std::string detect_cpu_name() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        return "Unknown CPU";
    }
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto pos = line.find(':');
            if (pos != std::string::npos) {
                std::string name = line.substr(pos + 1);
                const auto first = name.find_first_not_of(" \t");
                if (first != std::string::npos) {
                    return name.substr(first);
                }
            }
        }
    }
    return "Unknown CPU";
}

bool write_result_json(const std::string &path, const KMeansConfig &cfg, const KMeansResult &res,
                       const VerificationReport *verify, bool default_verified) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "Error: failed to open JSON output path: " << path << "\n";
        return false;
    }

    const double ms_per_iter =
        (res.iterations > 0) ? (res.kernel_ms / static_cast<double>(res.iterations)) : 0.0;
    const bool is_verified = verify ? verify->passed : default_verified;
    const std::string &commit =
        res.hw.git_commit.empty() ? std::string(KUDA_GIT_COMMIT) : res.hw.git_commit;

    ofs << "{\n"
        << "  \"version\": \"" << json_escape(cfg.version) << "\",\n"
        << "  \"config\": {\n"
        << "    \"n_points\": " << cfg.n_points << ",\n"
        << "    \"dim\": " << cfg.dim << ",\n"
        << "    \"k_clusters\": " << cfg.k_clusters << ",\n"
        << "    \"max_iters\": " << cfg.max_iters << ",\n"
        << "    \"epsilon\": " << format_json_double(cfg.epsilon, false) << ",\n"
        << "    \"seed\": " << cfg.seed << ",\n"
        << "    \"warmup_runs\": " << cfg.warmup_runs << ",\n"
        << "    \"bench_runs\": " << cfg.bench_runs << "\n"
        << "  },\n"
        << "  \"hardware\": {\n"
        << "    \"gpu_name\": \"" << json_escape(res.hw.gpu_name) << "\",\n"
        << "    \"compute_cap\": \"" << json_escape(res.hw.compute_cap) << "\",\n"
        << "    \"sm_count\": " << res.hw.sm_count << ",\n"
        << "    \"global_mem_mb\": " << format_json_double(res.hw.global_mem_mb) << ",\n"
        << "    \"cuda_version\": \"" << json_escape(res.hw.cuda_version) << "\",\n"
        << "    \"git_commit\": \"" << json_escape(commit) << "\"\n"
        << "  },\n"
        << "  \"metrics\": {\n"
        << "    \"iterations\": " << res.iterations << ",\n"
        << "    \"sse\": " << format_json_double(res.sse) << ",\n"
        << "    \"kernel_ms\": " << format_json_double(res.kernel_ms) << ",\n"
        << "    \"e2e_ms\": " << format_json_double(res.e2e_ms) << ",\n"
        << "    \"ms_per_iter\": " << format_json_double(ms_per_iter) << "\n"
        << "  },\n"
        << "  \"verified\": " << (is_verified ? "true" : "false");

    if (verify) {
        ofs << ",\n"
            << "  \"verification\": {\n"
            << "    \"passed\": " << (verify->passed ? "true" : "false") << ",\n"
            << "    \"assignment_mismatches\": " << verify->assignment_mismatches << ",\n"
            << "    \"max_centroid_abs_diff\": "
            << format_json_double(verify->max_centroid_abs_diff, false) << ",\n"
            << "    \"sse_rel_diff\": " << format_json_double(verify->sse_rel_diff, false) << "\n"
            << "  }\n";
    } else {
        ofs << "\n";
    }
    ofs << "}\n";
    return ofs.good();
}

bool append_result_csv(const std::string &path, const KMeansConfig &cfg, const KMeansResult &res,
                       const VerificationReport *verify, bool default_verified) {
    bool write_header = false;
    {
        std::ifstream ifs(path);
        write_header = !ifs.good() || ifs.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "Error: failed to open CSV output path: " << path << "\n";
        return false;
    }

    if (write_header) {
        ofs << "version,gpu_name,compute_cap,sm_count,global_mem_mb,cuda_version,git_commit,"
               "N,D,K,iterations,sse,kernel_ms,e2e_ms,ms_per_iter,verified\n";
    }

    const double ms_per_iter =
        (res.iterations > 0) ? (res.kernel_ms / static_cast<double>(res.iterations)) : 0.0;
    const bool is_verified = verify ? verify->passed : default_verified;
    const std::string &commit =
        res.hw.git_commit.empty() ? std::string(KUDA_GIT_COMMIT) : res.hw.git_commit;

    ofs << csv_escape(cfg.version) << "," << csv_escape(res.hw.gpu_name) << ","
        << csv_escape(res.hw.compute_cap) << "," << res.hw.sm_count << "," << std::fixed
        << std::setprecision(2) << res.hw.global_mem_mb << "," << csv_escape(res.hw.cuda_version)
        << "," << csv_escape(commit) << "," << cfg.n_points << "," << cfg.dim << ","
        << cfg.k_clusters << "," << res.iterations << "," << std::setprecision(6) << res.sse << ","
        << res.kernel_ms << "," << res.e2e_ms << "," << ms_per_iter << ","
        << (is_verified ? "true" : "false") << "\n";
    return ofs.good();
}

} // namespace kuda
