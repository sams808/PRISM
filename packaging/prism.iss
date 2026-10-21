; Inno Setup script for PRISM -- wraps the PyInstaller folder dist\PRISM
; into dist\PRISM-<version>-setup.exe (Start-menu entry, optional desktop
; icon, uninstaller). Compile after PyInstaller, from the repository root:
;
;     "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" /DMyAppVersion=2.8.0 packaging\prism.iss
;
; or run build_exe.bat, which does the whole chain. The version is passed
; on the command line so this file never carries a stale number.
;
; Per-user by default (no administrator rights needed on a lab computer:
; the app lands in %LOCALAPPDATA%\Programs\PRISM); the wizard offers the
; all-users install when the account can elevate.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "PRISM"
#define MyAppPublisher "NOME group, Washington State University"
#define MyAppURL "https://github.com/sams808/PRISM"
#define MyAppExeName "PRISM.exe"
#ifndef SourceDir
  #define SourceDir "..\dist\PRISM"
#endif

[Setup]
AppId={{7C1E2B4A-5D3F-4F8A-9B6C-0E2D7A1F3C55}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename={#MyAppName}-{#MyAppVersion}-setup
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
#ifexist "..\assets\prism.ico"
SetupIconFile=..\assets\prism.ico
#endif
#ifexist "..\LICENSE"
LicenseFile=..\LICENSE
#endif
#ifexist "INSTALL.txt"
InfoBeforeFile=INSTALL.txt
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; wipe the previous version's _internal before laying down the new one: an
; overwrite install leaves files the new version no longer ships. The
; user's databases, projects and settings live elsewhere (%USERPROFILE%\
; .raman_cache, the registry) and are never touched.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{#MyAppName} install notes"; Filename: "{app}\INSTALL.txt"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
