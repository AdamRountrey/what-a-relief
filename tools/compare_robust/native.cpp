#include "photometric.hpp"
#include "robust_fit.hpp"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

template<class T> void read(std::istream& in, T* data, size_t count) {
    in.read(reinterpret_cast<char*>(data), count * sizeof(T));
    if (!in) throw std::runtime_error("Truncated benchmark input");
}

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("Usage: compare-native input.bin output.bin");
    cv::setNumThreads(1);
    std::ifstream in(argv[1], std::ios::binary);
    uint32_t n, count;
    read(in, &n, 1); read(in, &count, 1);
    if (!n || n > 1000000 || count < 3 || count > 256) throw std::runtime_error("Invalid benchmark dimensions");
    std::vector<float> a(static_cast<size_t>(n)*count*3), b(static_cast<size_t>(n)*count);
    std::vector<unsigned char> clipped(b.size()), headroom(b.size());
    read(in, a.data(), a.size()); read(in, b.data(), b.size());
    read(in, clipped.data(), clipped.size()); read(in, headroom.data(), headroom.size());
    // Per sample and method: unit normal XYZ, iterations, converged, seconds.
    std::vector<double> output(static_cast<size_t>(n)*4*6, std::numeric_limits<double>::quiet_NaN());
    std::vector<cv::Vec3f> lights(count);
    std::vector<float> values(count);
    std::vector<double> h(count), weights(count);
    std::vector<cv::Mat> images(count), clips(count), heads(count);
    for (uint32_t i=0; i<count; ++i) {
        images[i] = cv::Mat(1,1,CV_32F); clips[i] = cv::Mat(1,1,CV_8U); heads[i] = cv::Mat(1,1,CV_8U);
    }
    for (uint32_t p=0; p<n; ++p) {
        for (uint32_t i=0; i<count; ++i) {
            size_t k=static_cast<size_t>(p)*count+i;
            lights[i]=cv::Vec3f(a[k*3],a[k*3+1],a[k*3+2]);
            values[i]=b[k];
            h[i]=!clipped[k] && values[i]>.02f ? headroom[k]/255.0 : 0.0;
            images[i].at<float>(0)=values[i]; clips[i].at<uchar>(0)=clipped[k]; heads[i].at<uchar>(0)=headroom[k];
        }
        for (int method=0; method<4; ++method) {
            auto start=std::chrono::steady_clock::now();
            double* out=&output[(static_cast<size_t>(p)*4+method)*6];
            cv::Vec3d g;
            bool ok=false, converged=false;
            int iterations=0;
            if(method==3) {
                cv::Mat normals, albedo, residual, mask;
                PhotometricDiagnostics d;
                try {
                    solvePhotometricStereo(images,lights,cv::Mat(1,1,CV_8U,cv::Scalar(255)),.02f,
                        NormalSolverMode::Robust,.98f,LightingModel::Directional,0,0,0,{},cv::Vec3f(0,0,1),
                        normals,albedo,residual,mask,d,clips,heads);
                    ok=mask.at<uchar>(0)!=0;
                    if(ok) g=cv::Vec3d(normals.at<cv::Vec3f>(0));
                    iterations=static_cast<int>(d.robustMeanIterations);
                    converged=d.robustNonconvergedFraction==0;
                } catch(const std::exception&) { ok=false; }
            } else {
                weights=h;
                ok=robust_fit::weightedSolve(lights,values,weights,g);
                converged=method==0;
                // Independent Huber implementation; fixed delta matches 5/255.
                // Compare short and extended IRLS budgets on identical accepted data.
                if(ok && method) for(int k=0; k<(method==1?10:80); ++k) {
                    cv::Vec3d next;
                    for(uint32_t i=0;i<count;++i) {
                        double r=std::abs(values[i]-cv::Vec3d(lights[i]).dot(g));
                        weights[i]=h[i]*std::min(1.0,(5.0/255.0)/std::max(1e-12,r));
                    }
                    if(!robust_fit::weightedSolve(lights,values,weights,next)) {ok=false; break;}
                    ++iterations;
                    double change=cv::norm(next-g)/std::max(1e-8,cv::norm(g));
                    g=next;
                    if(change<1e-6) {converged=true;break;}
                }
                ok=ok && std::isfinite(cv::norm(g)) && cv::norm(g)>1e-8 && g[2]>0;
                if(ok) g/=cv::norm(g);
            }
            if(ok) for(int j=0;j<3;++j) out[j]=g[j];
            out[3]=iterations; out[4]=converged?1:0;
            out[5]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        }
    }
    std::ofstream dest(argv[2], std::ios::binary);
    dest.write(reinterpret_cast<const char*>(output.data()),output.size()*sizeof(double));
    if(!dest) throw std::runtime_error("Cannot write benchmark output");
    return 0;
} catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
