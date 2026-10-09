#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int sparkle_initialize(const char *bundle_path, void (*prepare)(void),
    void (*event)(const char *, const char *));
void sparkle_check_for_updates(void);
void sparkle_check_for_updates_in_background(void);
void sparkle_resume_installation(void);
void sparkle_shutdown(void);
#ifdef __cplusplus
}
#endif
