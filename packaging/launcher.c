// Native macOS launcher for the independently signed runtime app.
#include <mach-o/dyld.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    (void)argc;
    char raw[PATH_MAX], path[PATH_MAX], target[PATH_MAX];
    uint32_t size = sizeof(raw);
    if (_NSGetExecutablePath(raw, &size) || !realpath(raw, path)) { perror("application path"); return 1; }
    char *slash = strrchr(path, '/');
    if (!slash) return 1;
    *slash = '\0';
#if defined(__arm64__)
    const char *cpu = "arm64";
#elif defined(__x86_64__)
    const char *cpu = "x86_64";
#else
#error "Build this launcher for arm64 or x86_64"
#endif
    int length = snprintf(target, sizeof(target), "%s/../Resources/runtime/%s/Runtime.app/Contents/MacOS/runtime", path, cpu);
    if (length < 0 || (size_t)length >= sizeof(target)) return 1;
    argv[0] = target;
    execv(target, argv);
    perror("Cannot launch bundled runtime");
    return 1;
}
