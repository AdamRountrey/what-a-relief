# Security

what-a-relief is a local desktop and command-line tool. It does not intentionally make network requests at runtime.

## Supported Use

- Use trusted image sets when possible. The program reads images through OpenCV, so malformed or adversarial image files exercise third-party image decoders.
- Treat output directories and mesh paths as user-controlled write locations. Do not run the program with elevated privileges unless you have a specific reason.
- The Windows installer is currently unsigned. Windows may warn before running it, and users should only install binaries obtained from a trusted release location.
- Build scripts use CMake/vcpkg and may download or build third-party dependencies during setup. Review dependency versions before making regulated or institutional deployments.

## Reporting Issues

Until a public issue tracker is configured, report suspected security issues privately to the project maintainer rather than posting exploit details in a public issue.

Please include:

- affected version or commit,
- operating system,
- steps to reproduce,
- whether the issue requires a malicious input image, malicious project path, or local user access.
