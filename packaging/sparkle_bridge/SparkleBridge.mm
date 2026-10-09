#import <Cocoa/Cocoa.h>
#import <Sparkle/Sparkle.h>
#import "SparkleBridge.h"

// Update the outer product bundle, but wait for the actual running application.
static SPUUpdater *updater;
static SPUStandardUserDriver *driver;
static void (^resumeInstallation)(void);
static void (*prepareHost)(void);
static void (*reportEvent)(const char *, const char *);
static void report(const char *event, SUAppcastItem *item) {
    if (reportEvent) reportEvent(event, item ? item.versionString.UTF8String : "");
}

@interface HanziUpdateDelegate : NSObject <SPUUpdaterDelegate, SPUStandardUserDriverDelegate>
@end
@implementation HanziUpdateDelegate
- (BOOL)standardUserDriverShouldShowVersionHistoryForAppcastItem:(SUAppcastItem *)item {
    return NO;
}
- (void)updater:(SPUUpdater *)sender didFindValidUpdate:(SUAppcastItem *)item {
    report("update_found", item);
}
- (void)updater:(SPUUpdater *)sender willDownloadUpdate:(SUAppcastItem *)item withRequest:(NSMutableURLRequest *)request {
    report("download_started", item);
}
- (void)updater:(SPUUpdater *)sender didDownloadUpdate:(SUAppcastItem *)item {
    report("download_finished", item);
}
- (void)updater:(SPUUpdater *)sender willInstallUpdate:(SUAppcastItem *)item {
    report("installer_started", item);
}
- (void)updater:(SPUUpdater *)sender didAbortWithError:(NSError *)error {
    // Do not expose NSError descriptions containing download URLs or local paths.
    if (error.code != SUNoUpdateError) report("update_error", nil);
}
- (BOOL)updater:(SPUUpdater *)sender shouldPostponeRelaunchForUpdate:(SUAppcastItem *)item
    untilInvokingBlock:(void (^)(void))handler {
    resumeInstallation = [handler copy];
    if (prepareHost) prepareHost();
    return YES;
}
@end
static HanziUpdateDelegate *delegate;

int sparkle_initialize(const char *bundlePath, void (*prepare)(void),
    void (*event)(const char *, const char *)) {
    if (![NSThread isMainThread] || !bundlePath || !prepare) return 0;
    if (updater) return 1;
    @autoreleasepool {
        NSBundle *bundle = [NSBundle bundleWithPath:[NSString stringWithUTF8String:bundlePath]];
        if (!bundle) return 0;
        NSURL *applicationURL = [NSRunningApplication currentApplication].bundleURL;
        NSBundle *applicationBundle = applicationURL ? [NSBundle bundleWithURL:applicationURL] : nil;
        NSString *hostPath = bundle.bundleURL.URLByResolvingSymlinksInPath.path;
        NSString *applicationPath = applicationBundle.bundleURL.URLByResolvingSymlinksInPath.path;
        // An unrelated or unregistered process cannot safely authorize replacement.
        if (!applicationBundle.bundleIdentifier || !applicationPath ||
            (![applicationPath isEqualToString:hostPath] &&
             ![applicationPath hasPrefix:[hostPath stringByAppendingString:@"/Contents/"]])) return 0;
        prepareHost = prepare;
        reportEvent = event;
        delegate = [HanziUpdateDelegate new];
        driver = [[SPUStandardUserDriver alloc] initWithHostBundle:bundle delegate:delegate];
        updater = [[SPUUpdater alloc] initWithHostBundle:bundle applicationBundle:applicationBundle
            userDriver:driver delegate:delegate];
        updater.automaticallyChecksForUpdates = NO;
        NSError *error = nil;
        if (![updater startUpdater:&error]) {
            sparkle_shutdown();
            return 0;
        }
        return 1;
    }
}
void sparkle_check_for_updates(void) {
    if (![NSThread isMainThread]) return;
    if (resumeInstallation) {
        if (prepareHost) prepareHost();
    } else if (updater.canCheckForUpdates) {
        [updater checkForUpdates];
    }
}
void sparkle_check_for_updates_in_background(void) {
    if (![NSThread isMainThread] || !updater || resumeInstallation || updater.sessionInProgress) return;
    [updater checkForUpdatesInBackground];
}
void sparkle_resume_installation(void) {
    if (![NSThread isMainThread] || !resumeInstallation) return;
    void (^handler)(void) = resumeInstallation;
    resumeInstallation = nil;
    handler();
}
void sparkle_shutdown(void) {
    if (![NSThread isMainThread]) return;
    prepareHost = nullptr;
    reportEvent = nullptr;
    resumeInstallation = nil;
    updater = nil;
    driver = nil;
    delegate = nil;
}
