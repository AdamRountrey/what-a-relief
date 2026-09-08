#include <filesystem>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string argumentAfter(const std::vector<std::string>& arguments, const std::string& flag) {
    for (size_t i = 0; i + 1 < arguments.size(); ++i) {
        if (arguments[i] == flag) {
            return arguments[i + 1];
        }
    }
    return {};
}

void writeFile(const fs::path& path, const std::string& contents) {
    std::ofstream out(path, std::ios::binary);
    out << contents;
    if (!out) {
        throw std::runtime_error("Could not write fake backend output: " + path.string());
    }
}

std::string readFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> arguments;
        for (int i = 1; i < argc; ++i) {
            arguments.emplace_back(argv[i]);
        }
        const std::string probeResult = argumentAfter(arguments, "--result");
        if (!probeResult.empty()) {
            writeFile(
                probeResult,
                "{\"status\":\"available\",\"selected_backend\":\"cuda\","
                "\"variant\":\"fake_ad_rgb\",\"optimizer\":\"fake_adam\"}\n");
            return 0;
        }

        const std::string jobPath = argumentAfter(arguments, "--job");
        if (jobPath.empty()) {
            std::cerr << "Fake backend expected --probe or --job.\n";
            return 2;
        }
        const fs::path output = fs::path(jobPath).parent_path();
        const std::string job = readFile(jobPath);
        const bool accepted = job.find("\"quality\": \"standard\"") != std::string::npos;
        if (job.find("\"encoding\": \"png16\"") == std::string::npos ||
            job.find("\"photometry\": \"linear_luminance\"") == std::string::npos ||
            job.find("\"srgb_decode\": false") == std::string::npos ||
            job.find(".tif\"") == std::string::npos) {
            throw std::runtime_error("Mitsuba job did not declare the linear PNG handoff and TIFF source audit.");
        }
        const fs::path observations = output / "input_observations";
        size_t observationCount = 0;
        for (const fs::directory_entry& entry : fs::directory_iterator(observations)) {
            if (entry.is_regular_file() && entry.path().extension() == ".png" &&
                entry.file_size() > 8) {
                ++observationCount;
            }
        }
        if (observationCount != 12 || job.find("\"observation_validity\"") == std::string::npos ||
            job.find("\"reference_height_pixels\"") == std::string::npos) {
            throw std::runtime_error("Mitsuba job did not provide six observations with validity masks and datum.");
        }
        if (job.find("\"normal_prior_pfm\"") == std::string::npos) {
            throw std::runtime_error("Mitsuba job did not declare its photometric normal prior.");
        }
        for (int axis = 0; axis < 3; ++axis) {
            const std::string name = "input_normal_" + std::to_string(axis) + ".pfm";
            std::ifstream in(output / name, std::ios::binary);
            std::string magic, dimensions, scale;
            std::getline(in, magic);
            std::getline(in, dimensions);
            std::getline(in, scale);
            if (magic != "Pf" || dimensions != "8 8" || scale != "-1.0" || job.find(name) == std::string::npos) {
                throw std::runtime_error("Invalid normal-prior PFM handoff.");
            }
            for (int y = 7; y >= 0; --y) {
                for (int x = 0; x < 8; ++x) {
                    float actual = 0;
                    in.read(reinterpret_cast<char*>(&actual), sizeof(actual));
                    const float nx = 0.02f * (x - 3);
                    const float ny = 0.03f * (y - 4);
                    const float expected = axis == 0 ? nx : axis == 1 ? ny : std::sqrt(1.0f - nx * nx - ny * ny);
                    if (!in || !std::isfinite(actual) || std::abs(actual - expected) > 1e-7f) {
                        throw std::runtime_error("Normal-prior axes, row orientation, or precision changed.");
                    }
                }
            }
        }
        const std::vector<std::string> required = {
            "inverse_height.pfm", "inverse_height.png", "inverse_normal_rgb.png",
            "inverse_normal_x.png", "inverse_normal_y.png", "inverse_normal_z.png",
            "inverse_hillshade_ul.png", "inverse_surface.ply", "height_correction.pfm",
            "height_correction.png", "render_before.png", "render_after.png",
            "material_diffuse.png", "material_specular.png", "material_roughness.png"};
        for (const std::string& name : required) {
            writeFile(output / name, "fake-contract-output\n");
        }
        const fs::path candidate = output / "unvalidated_candidate";
        if (!accepted) {
            fs::create_directories(candidate);
            for (const std::string& name : {
                 "candidate.json", "review.html", "candidate_height.pfm", "candidate_height.png",
                 "candidate_normal_rgb.png", "candidate_normal_x.png", "candidate_normal_y.png",
                 "candidate_normal_z.png", "candidate_hillshade_ul.png", "candidate_surface.ply",
                 "height_correction.pfm", "height_correction.png",
                 "review_baseline_height.png", "review_candidate_height.png"}) {
                writeFile(candidate / name, "fake-unvalidated-candidate\n");
            }
        }
        writeFile(
            output / "result.json",
            "{\n"
            "  \"schema_version\": 2,\n"
            "  \"method\": \"mitsuba_heightfield_inverse_v2\",\n"
            "  \"status\": \"complete\",\n"
            "  \"accepted\": " + std::string(accepted ? "true" : "false") + ",\n"
            "  \"candidate_saved\": " + std::string(accepted ? "false" : "true") + ",\n"
            "  \"candidate_export_status\": \"" + std::string(accepted ? "not_needed_accepted" : "saved_unvalidated") + "\",\n"
            "  \"decision\": \"" + std::string(accepted ? "fake_contract_acceptance" : "fake_contract_rejection") + "\",\n"
            "  \"selected_backend\": \"cuda\",\n"
            "  \"variant\": \"fake_ad_rgb\",\n"
            "  \"optimizer\": \"fake_adam\",\n"
            "  \"mitsuba_version\": \"test\",\n"
            "  \"drjit_version\": \"test\",\n"
            "  \"numpy_version\": \"test\",\n"
            "  \"python_version\": \"test\",\n"
            "  \"iterations_completed\": 3,\n"
            "  \"render_width\": 8,\n"
            "  \"render_height\": 8,\n"
            "  \"train_loss_before\": 0.2,\n"
            "  \"train_loss_after\": 0.1,\n"
            "  \"holdout_loss_before\": 0.3,\n"
            "  \"holdout_loss_after\": 0.31,\n"
            "  \"correction_rms_pixels\": 0.0,\n"
            "  \"correction_maximum_pixels\": 0.0\n"
            "}\n");
        writeFile(output / "progress.txt", "100\nFake backend complete\n");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
