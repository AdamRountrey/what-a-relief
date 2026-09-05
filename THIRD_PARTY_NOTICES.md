# Third-Party Notices

what-a-relief bundles third-party runtime libraries in Windows packages. This file is a summary for release review; packaged builds also include `THIRD_PARTY_LICENSES.txt`, which contains the exact license and notice files copied from the vcpkg packages used for that build.

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

These ONNX files are format-converted exports of the original pretrained PS-FCN model used only for optional experimental normal-map fusion. They do not change the license of what-a-relief itself; they remain third-party assets and should continue to be attributed to the PS-FCN authors in source and binary distributions. The repository bundles `assets/models/NOTICE.txt` and `assets/models/LICENSE-PS-FCN.txt`, and packaged builds should include those same files under `models/`.

## Optional Mitsuba Backend Package

The separate `what-a-relief-<version>-mitsuba-backend-setup.exe` package is not
required for ordinary photometric stereo. It contains a private runtime used
only by the opt-in inverse-rendering experiment:

| Component | Bundled use | Source and license |
| --- | --- | --- |
| CPython 3.13.13 embeddable distribution | Private interpreter for the versioned worker | Python Software Foundation License; `PYTHON_LICENSE.txt` is installed with the runtime |
| Mitsuba 3.8.0 | Differentiable rendering | Wenzel Jakob and contributors; BSD-style license retained in the wheel metadata |
| Dr.Jit 1.3.1 | Automatic differentiation and CUDA/LLVM execution | Wenzel Jakob and contributors; BSD-3-Clause license retained in the wheel metadata |
| NumPy 2.3.3 | Numeric arrays and image data transfer | NumPy Developers; BSD-3-Clause and bundled-library notices retained in the wheel metadata |
| LLVM 18.1.6 `LLVM-C.dll` | CPU execution for systems without a compatible NVIDIA GPU | Apache License 2.0 with LLVM Exceptions; `LLVM_LICENSE.TXT` is installed with the runtime |

The add-on build uses exact Windows artifacts with pinned SHA-256 values and
performs a live LLVM render probe before packaging and installation. It does
not register Python globally, alter `PATH`, or use an existing Python or Conda
environment.

## Offline Validation Tools

The repository includes a Python generator and committed reference images under `tests/fixtures/mitsuba/`. Mitsuba and NumPy are development tools for intentional fixture regeneration. They are not linked into the C++ application or invoked by CTest. Their presence in the separate optional backend package is solely for the user-selected inverse-rendering feature described above.

| Component | Use | Source and license |
| --- | --- | --- |
| Mitsuba 3.8.0 / Dr.Jit | Physically based rendering of the committed robust-solver development and validation fixtures | Mitsuba 3 by Wenzel Jakob and contributors; BSD-style license reproduced in `tests/fixtures/mitsuba/LICENSE-MITSUBA.txt` |
| NumPy 2.3.3 | Deterministic sensor-noise simulation and fixture assembly | NumPy Developers; BSD-3-Clause |

The generated PNG and CSV assets are test data produced by the fixture script, not renderer binaries. Academic citations for Mitsuba 3 and Dr.Jit are in `docs/references.bib`.
