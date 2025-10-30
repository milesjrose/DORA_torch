@echo off

REM Resize and center the console window to 2/3 of the primary monitor
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; Add-Type -Name Win32 -Namespace Native -MemberDefinition '[DllImport(\"user32.dll\")] public static extern System.IntPtr GetForegroundWindow(); [DllImport(\"user32.dll\")] public static extern bool SetWindowPos(System.IntPtr hWnd, System.IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);'; $hwnd = [Native.Win32]::GetForegroundWindow(); $wa = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea; $w = [int]($wa.Width * 2 / 3); $h = [int]($wa.Height * 2 / 3); $x = $wa.Left + [int](($wa.Width - $w)/2); $y = $wa.Top + [int](($wa.Height - $h)/2); [Native.Win32]::SetWindowPos($hwnd, [IntPtr]::Zero, $x, $y, $w, $h, 0x0044) | Out-Null;"

echo Running DORA TUI
python -m DORA_tensorised.run_tui