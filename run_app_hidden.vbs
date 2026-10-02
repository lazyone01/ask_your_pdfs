' Starts run_app.bat with no console window (used for auto-start at Windows login).
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
CreateObject("WScript.Shell").Run """" & dir & "\run_app.bat""", 0, False
