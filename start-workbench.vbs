Option Explicit

Dim files, shell, project, python, launcher
Set files = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
project = files.GetParentFolderName(WScript.ScriptFullName)
python = files.BuildPath(project, ".venv\Scripts\pythonw.exe")
launcher = files.BuildPath(project, "src\gpt_animeimage_denoise\workbench\desktop.py")

If Not files.FileExists(python) Then
    MsgBox "Missing " & python & vbCrLf & "Follow the installation steps in README.md.", _
        vbExclamation, "GAID Workbench"
    WScript.Quit 1
End If

If Not files.FileExists(launcher) Then
    MsgBox "Missing " & launcher & vbCrLf & "Restore the project files and see README.md.", _
        vbExclamation, "GAID Workbench"
    WScript.Quit 1
End If

shell.CurrentDirectory = project
On Error Resume Next
shell.Run """" & python & """ """ & launcher & """", 0, False
If Err.Number <> 0 Then
    MsgBox Err.Description, vbExclamation, "GAID Workbench"
    WScript.Quit 1
End If
