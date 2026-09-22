; Values supplied by native_release.py from the single PACKAGE_VERSION source.
[Setup]
AppId={#ProductId}
AppName={#ProductName}
AppVersion={#ProductVersion}
DefaultDirName={localappdata}\Programs\{#ProductId}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll,*.pyd
RestartApplications=no
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
OutputDir={#OutputDir}
OutputBaseFilename={#ProductId}-v{#ProductVersion}-windows-x64
UninstallDisplayIcon={app}\{#ProductExe}.exe

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#ProductName}"; Filename: "{app}\{#ProductExe}.exe"

[Run]
Filename: "{app}\{#ProductExe}.exe"; Flags: nowait runasoriginaluser; Check: ShouldLaunchApplication

[Code]
function ShouldLaunchApplication: Boolean;
begin
  Result := ExpandConstant('{param:HANZITEST|0}') <> '1';
end;
