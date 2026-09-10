#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace robust_fit {

// Independent IRLS implementation (Holland & Welsch, 1977), with a continuous
// Cauchy objective motivated by Queau et al., CVPR 2017. The initialization,
// adaptive scale and constants are application choices, not their full method.
// See docs/algorithm.md for equations, limits, and the SBL comparison.
struct Workspace {
    std::vector<double> variance;
    std::vector<double> coefficients;
    std::vector<double> reliability;
    std::vector<double> residuals;
    std::vector<double> sensorWeights;
    double signalReference = 0.0;
    double residualScale = 0.0;
    int iterations = 0;
    bool converged = false;
    cv::Matx33d covariance = cv::Matx33d::zeros();
};

inline double noiseSigma(double observation, double signalReference) {
    return 0.0025 + 0.008 * std::sqrt(std::max(0.0, observation)) +
        0.003 * std::max(0.05, signalReference);
}

inline double lowSignalWeight(double observation, double minimumIntensity) {
    if (!std::isfinite(observation) || observation <= minimumIntensity) return 0.0;
    // Keep the user's cutoff as a zero-weight floor. A heuristic noise sigma
    // above it gives a gradual entry, not a new shadow classification.
    const double width = noiseSigma(minimumIntensity, minimumIntensity);
    const double t = std::clamp((observation - minimumIntensity) / width, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

inline double residualMScale(const std::vector<double>& squaredResiduals,
    const std::vector<double>& weights, double floor) {
    // Bounded M-scale: sum h*r^2/(r^2+(c*s)^2) = sum h/2.
    // c makes E[rho(Z)] = 1/2 for standard normal Z. Reliability weights
    // are fixed input weights, not residual-dependent IRLS coefficients.
    constexpr double c = 0.6120031809624806;
    double totalWeight = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) if (weights[i] > 0.0) {
        totalWeight += weights[i];
    }
    auto aboveRoot = [&](double scale) {
        const double cs2 = (c * scale) * (c * scale);
        double sum = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) if (weights[i] > 0.0) {
            sum += weights[i] * squaredResiduals[i] / (squaredResiduals[i] + cs2);
        }
        return sum > 0.5 * totalWeight;
    };
    if (totalWeight <= 0.0 || !aboveRoot(floor)) return floor;
    // Bracket from the floor so a near-zero-weight extreme residual cannot
    // inflate the search interval and degrade root precision.
    double low = floor, high = 2.0 * floor;
    while (aboveRoot(high)) {
        low = high;
        high *= 2.0;
    }
    for (int iteration = 0; iteration < 40; ++iteration) {
        const double middle = 0.5 * (low + high);
        if (aboveRoot(middle)) low = middle;
        else high = middle;
        if (high - low <= 1.0e-6 * high) break;
    }
    return 0.5 * (low + high);
}

inline bool weightedSolve(const std::vector<cv::Vec3f>& lights,
    const std::vector<float>& observations, const std::vector<double>& weights,
    cv::Vec3d& normal, cv::Matx33d* covariance = nullptr) {
    cv::Matx33d gram = cv::Matx33d::zeros();
    cv::Vec3d rhs(0, 0, 0);
    int used = 0;
    for (size_t i = 0; i < observations.size(); ++i) {
        const double w = weights[i];
        if (w <= 0.0) continue;
        const cv::Vec3d l(lights[i]);
        gram += w * (l * l.t());
        rhs += (w * observations[i]) * l;
        ++used;
    }
    if (used < 3 || !cv::solve(gram, rhs, normal, cv::DECOMP_CHOLESKY)) return false;
    if (covariance && !cv::invert(gram, *covariance, cv::DECOMP_CHOLESKY)) return false;
    return std::isfinite(cv::norm(normal));
}

inline double objective(const std::vector<cv::Vec3f>& lights,
    const std::vector<float>& observations, const Workspace& work,
    const cv::Vec3d& normal, bool cauchy) {
    double loss = 0.0;
    for (size_t i = 0; i < observations.size(); ++i) {
        if (work.variance[i] <= 0.0) continue;
        const double r = observations[i] - cv::Vec3d(lights[i]).dot(normal);
        const double z2 = r * r / work.variance[i];
        loss += work.sensorWeights[i] * (cauchy ? std::log1p(z2) : 2.0 * (std::sqrt(1.0 + z2) - 1.0));
    }
    return loss;
}

inline void propagateNoise(const std::vector<cv::Vec3f>& lights,
    const std::vector<float>& observations, double intensity, Workspace& work) {
    // Conditional noise propagation with frozen IRLS weights, not a calibrated
    // posterior: G^-1 (L^T W Sigma_noise W L) G^-1. Model bias is not noise.
    cv::Matx33d propagated = cv::Matx33d::zeros();
    for (size_t i = 0; i < observations.size(); ++i) {
        if (work.coefficients[i] <= 0.0) continue;
        const cv::Vec3d light(lights[i]);
        const double sigma = noiseSigma(observations[i], intensity);
        const double weightedSigma = work.coefficients[i] * sigma;
        propagated += (weightedSigma * weightedSigma) * (light * light.t());
    }
    work.covariance = work.covariance * propagated * work.covariance;
}

inline bool fit(const std::vector<cv::Vec3f>& lights,
    const std::vector<float>& observations, const std::vector<unsigned char>& clipped,
    double minimumIntensity, double probableSaturation, Workspace& work,
    cv::Vec3d& normal, const std::vector<float>* headroom = nullptr) {
    const size_t count = observations.size();
    work.iterations = 0;
    work.converged = false;
    work.signalReference = 0.0;
    work.residualScale = 0.0;
    if (count < 3 || lights.size() != count || clipped.size() != count) return false;
    if (headroom && headroom->size() != count) return false;
    work.variance.assign(count, 0.0);
    work.coefficients.assign(count, 0.0);
    work.reliability.assign(count, 0.0);
    work.sensorWeights.assign(count, 0.0);
    work.residuals.assign(count, 0.0);
    double weightSum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (std::isfinite(observations[i]) && observations[i] > minimumIntensity && !clipped[i]) {
            const double headWeight = headroom ? std::clamp(static_cast<double>((*headroom)[i]), 0.0, 1.0) : 1.0;
            const double weight = headWeight * lowSignalWeight(observations[i], minimumIntensity);
            if (!std::isfinite(weight) || weight <= 0.0) continue;
            work.sensorWeights[i] = weight;
            work.coefficients[i] = weight;
            weightSum += weight;
            work.signalReference += weight * observations[i];
        }
    }
    if (weightSum <= 0.0 || !weightedSolve(lights, observations, work.coefficients, normal, &work.covariance)) return false;
    double intensity = work.signalReference /= weightSum;
    double noise = noiseSigma(intensity, intensity);
    work.residualScale = noise;
    double largestResidual = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (work.coefficients[i] > 0.0) largestResidual = std::max(largestResidual,
            std::abs(observations[i] - cv::Vec3d(lights[i]).dot(normal)));
    }
    if (largestResidual < 1.0e-6) {
        work.reliability = work.coefficients;
        work.converged = true;
        propagateNoise(lights, observations, intensity, work);
        return normal[2] > 0.0;
    }
    int unsaturated = 0;
    for (size_t i = 0; i < count; ++i) if (work.coefficients[i] > 0.0 && observations[i] < probableSaturation) ++unsaturated;
    if (!headroom && unsaturated >= 3) {
        for (size_t i = 0; i < count; ++i) if (observations[i] >= probableSaturation) {
            work.coefficients[i] = 0.0;
            work.sensorWeights[i] = 0.0;
        }
        if (!weightedSolve(lights, observations, work.coefficients, normal)) return false;
        weightSum = 0.0;
        work.signalReference = 0.0;
        for (size_t i = 0; i < count; ++i) if (work.sensorWeights[i] > 0.0) {
            weightSum += work.sensorWeights[i];
            work.signalReference += work.sensorWeights[i] * observations[i];
        }
        intensity = work.signalReference /= weightSum;
        noise = noiseSigma(intensity, intensity);
    }
    for (size_t i = 0; i < count; ++i) if (work.coefficients[i] > 0.0) work.variance[i] = noise * noise * 4.0;

    // A convex pseudo-Huber start avoids the integer-consensus hypothesis switch.
    for (int phase = 0; phase < 2; ++phase) {
        if (phase == 1) {
            for (size_t i = 0; i < count; ++i) if (work.variance[i] > 0.0) {
                const double residual = observations[i] - cv::Vec3d(lights[i]).dot(normal);
                work.residuals[i] = residual * residual;
            }
            const double scale = residualMScale(work.residuals, work.sensorWeights, noise);
            work.residualScale = scale;
            for (size_t i = 0; i < count; ++i) if (work.variance[i] > 0.0) {
                const double observationNoise = noiseSigma(observations[i], intensity);
                work.variance[i] = 1.5 * 1.5 * (scale * scale + observationNoise * observationNoise);
            }
        }
        double loss = objective(lights, observations, work, normal, phase == 1);
        work.converged = false;
        for (int iteration = 0; iteration < 80; ++iteration) {
            ++work.iterations;
            for (size_t i = 0; i < count; ++i) {
                if (work.variance[i] <= 0.0) continue;
                const double r = observations[i] - cv::Vec3d(lights[i]).dot(normal);
                const double z2 = r * r / work.variance[i];
                work.reliability[i] = work.sensorWeights[i] *
                    (phase == 1 ? 1.0 / (1.0 + z2) : 1.0 / std::sqrt(1.0 + z2));
                work.coefficients[i] = work.reliability[i] / work.variance[i];
            }
            cv::Vec3d next;
            if (!weightedSolve(lights, observations, work.coefficients, next)) return false;
            const double nextLoss = objective(lights, observations, work, next, phase == 1);
            if (nextLoss > loss + 1.0e-9 * (1.0 + loss)) break;
            const double change = cv::norm(next - normal) / std::max(1.0e-8, cv::norm(normal));
            normal = next;
            loss = nextLoss;
            if (change < 1.0e-6) {
                work.converged = true;
                break;
            }
        }
    }
    // Report weights for the returned estimate, including iteration-limited fits.
    for (size_t i = 0; i < count; ++i) {
        if (work.variance[i] <= 0.0) continue;
        const double residual = observations[i] - cv::Vec3d(lights[i]).dot(normal);
        work.reliability[i] = work.sensorWeights[i] / (1.0 + residual * residual / work.variance[i]);
        work.coefficients[i] = work.reliability[i] / work.variance[i];
    }
    cv::Vec3d unused;
    if (!weightedSolve(lights, observations, work.coefficients, unused, &work.covariance)) return false;
    propagateNoise(lights, observations, intensity, work);
    return normal[2] > 0.0;
}

} // namespace robust_fit
