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
