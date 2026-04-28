# Third-Party Notices

What A Relief bundles third-party runtime libraries in Windows packages. This file is a summary for release review; packaged builds also include `THIRD_PARTY_LICENSES.txt`, which contains the exact license and notice files copied from the vcpkg packages used for that build.

The current Windows runtime package is built from vcpkg's `x64-windows` triplet and includes:

| Component | Bundled runtime files | License or notice source |
| --- | --- | --- |
| OpenCV 4 | `opencv_*.dll` | Apache-2.0 plus OpenCV third-party notices |
| Abseil | `abseil_dll.dll` | Apache-2.0 |
| Protocol Buffers | `libprotobuf*.dll`, `libprotoc.dll` | BSD-style Google license |
| libjpeg-turbo | `jpeg62.dll`, `turbojpeg.dll` | IJG, modified BSD, and zlib-style notices |
| libpng | `libpng16.dll` | PNG Reference Library License |
| libwebp | `libwebp*.dll`, `libsharpyuv.dll` | BSD-style Google license and patent notice |
| XZ Utils / liblzma | `liblzma.dll` | 0BSD for liblzma |
| LibTIFF | `tiff.dll` | LibTIFF license |
| zlib | `zlib1.dll` | zlib license |

Before making a release, rebuild the package and check `build-vcpkg-direct` to confirm the bundled DLL list still matches this file. If vcpkg changes OpenCV's dependency set, update this notice and regenerate the packaged license bundle.

## Bundled Neural Models

Experimental neural-fusion builds also bundle count-specific ONNX exports of the PS-FCN photometric-stereo model for image counts from 3 through 25:

| Component | Bundled files | Source and license |
| --- | --- | --- |
| PS-FCN ONNX models | `assets/models/psfcn_*_normalize.onnx` and packaged `models/psfcn_*_normalize.onnx` | Derived from the PS-FCN pretrained checkpoint by Guanying Chen; original code and checkpoint are distributed under the MIT License |

These ONNX files are format-converted exports of the original pretrained PS-FCN model used only for optional experimental normal-map fusion. They do not change the license of What A Relief itself; they remain third-party assets and should continue to be attributed to the PS-FCN authors in source and binary distributions. The repository bundles `assets/models/NOTICE.txt` and `assets/models/LICENSE-PS-FCN.txt`, and packaged builds should include those same files under `models/`.
