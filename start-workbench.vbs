Option Explicit

Dim files, shell, project, python
Set files = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
project = files.GetParentFolderName(WScript.ScriptFullName)
python = files.BuildPath(project, ".venv\Scripts\pythonw.exe")

If Not files.FileExists(python) Then
    MsgBox "Install GAID in the project's .venv first. See README.md.", vbExclamation, "GAID Workbench"
    WScript.Quit 1
End If

shell.CurrentDirectory = project
On Error Resume Next
shell.Run """" & python & """ -m gpt_animeimage_denoise.workbench.desktop", 0, False
If Err.Number <> 0 Then
    MsgBox Err.Description, vbExclamation, "GAID Workbench"
    WScript.Quit 1
End If
