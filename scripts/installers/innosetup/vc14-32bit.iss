[Setup]
PrivilegesRequired=none
AppName=ADMB
AppVerName=ADMB
UsePreviousAppDir=no
UsePreviousGroup=no
DefaultDirName={pf}\ADMB
DefaultGroupName=ADMB (Microsoft Visual C++ 14.0 32Bit)
UninstallDisplayIcon={app}\uninstall.exe
OutputDir=..\..\..\build
OutputBaseFilename=admb-vc14
AppPublisher=www.admb-project.org
AppPublisherURL=http://www.admb-project.org/
LicenseFile=..\..\..\LICENSE.txt

[Types]
Name: "vc14"; Description: "ADMB (Microsoft Visual C++ 14.0 32Bit)"

[Files]
Source: "..\..\..\build\dist\*"; DestDir: "{app}"; Excludes: ".svn,*\.svn"; Flags: recursesubdirs
Source: "..\..\cl\set-admb-vc14-32bit.bat"; DestDir: "{app}\bin"

[Icons]
Name: "{commondesktop}\ADMB Command Prompt (Microsoft Visual C++ 14.0 32Bit)"; Filename: "{cmd}"; Parameters: "/K ""{app}\bin\set-admb-vc14-32bit.bat"""; WorkingDir: "{app}"; Comment: "ADMB Command Prompt (Microsoft Visual C++ 14.0 32Bit)"
Name: "{group}\ADMB Command Prompt (Microsoft Visual C++ 14.0 32Bit)"; Filename: "{cmd}"; Parameters: "/K ""{app}\bin\set-admb-vc14-32bit.bat"""; WorkingDir: "{app}"; Comment: "ADMB Command Prompt (Microsoft Visual C++ 14.0 32Bit)"; OnlyBelowVersion: 6.1
Name: "{group}\Uninstall"; Filename: "{uninstallexe}"; Comment: "Uninstall ADMB"; OnlyBelowVersion: 6.1
