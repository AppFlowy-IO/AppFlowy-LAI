use std::path::PathBuf;
use std::process::Command;

#[cfg(windows)]
use winreg::{enums::*, RegKey};

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn install_path() -> Option<PathBuf> {
  None
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
pub fn install_path() -> Option<PathBuf> {
  #[cfg(target_os = "windows")]
  return None;

  #[cfg(target_os = "macos")]
  return Some(PathBuf::from("/usr/local/bin"));

  #[cfg(target_os = "linux")]
  return Some(PathBuf::from("/usr/local/bin"));
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
pub fn is_plugin_ready() -> bool {
  ollama_plugin_path().exists() || ollama_plugin_command_available()
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn is_plugin_ready() -> bool {
  false
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub(crate) fn ollama_plugin_path() -> PathBuf {
  PathBuf::new()
}

#[cfg(any(target_os = "windows", target_os = "macos", target_os = "linux"))]
pub fn ollama_plugin_path() -> std::path::PathBuf {
  #[cfg(target_os = "windows")]
  {
    // Use LOCALAPPDATA for a user-specific installation path on Windows.
    let local_appdata =
      std::env::var("LOCALAPPDATA").unwrap_or_else(|_| "C:\\Program Files".to_string());
    std::path::PathBuf::from(local_appdata).join("Programs\\appflowy_plugin\\af_ollama_plugin.exe")
  }

  #[cfg(target_os = "macos")]
  {
    let offline_app = "af_ollama_plugin";
    std::path::PathBuf::from(format!("/usr/local/bin/{}", offline_app))
  }

  #[cfg(target_os = "linux")]
  {
    let offline_app = "af_ollama_plugin";
    std::path::PathBuf::from(format!("/usr/local/bin/{}", offline_app))
  }
}

pub fn ollama_plugin_command_available() -> bool {
  if cfg!(windows) {
    #[cfg(windows)]
    {
      use std::os::windows::process::CommandExt;
      const CREATE_NO_WINDOW: u32 = 0x08000000;
      let output = Command::new("cmd")
        .args(&["/C", "where", "af_ollama_plugin"])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
      if let Ok(output) = output {
        if !output.stdout.is_empty() {
          return true;
        }
      }

      // 2. Fallback: Check registry PATH for the executable
      let path_dirs = get_windows_path_dirs();
      let plugin_exe = "af_ollama_plugin.exe";

      path_dirs.iter().any(|dir| {
        let full_path = std::path::Path::new(dir).join(plugin_exe);
        full_path.exists()
      })
    }

    #[cfg(not(windows))]
    false
  } else {
    let output = Command::new("command")
      .args(["-v", "af_ollama_plugin"])
      .output();
    match output {
      Ok(o) => !o.stdout.is_empty(),
      _ => false,
    }
  }
}

/// Retrieves directories from the Windows PATH environment variables.
///
/// This function reads the `Path` values from two registry locations on Windows:
///
/// 1. **HKEY_CURRENT_USER\Environment:**
///    Reads the user's environment variable for `Path`.
///
/// 2. **HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment:**
///    Reads the system-wide environment variable for `Path`.
///
/// The function splits the value (using `;` as the separator) for each registry key
/// into individual directory entries, trims any surrounding whitespace, and collects these
/// entries into a vector of strings.
///
/// # Returns
///
/// A `Vec<String>` containing all directories specified by the `Path` values found in the
/// registry. If a registry key or value cannot be read, that source is silently skipped.
///
/// This function is only compiled for Windows since it is marked with the `#[cfg(windows)]` attribute.
#[cfg(windows)]
fn get_windows_path_dirs() -> Vec<String> {
  let mut paths = Vec::new();

  // Check HKEY_CURRENT_USER\Environment
  let hkcu = RegKey::predef(HKEY_CURRENT_USER);
  if let Ok(env) = hkcu.open_subkey("Environment") {
    if let Ok(path) = env.get_value::<String, _>("Path") {
      paths.extend(path.split(';').map(|s| s.trim().to_string()));
    }
  }

  // Check HKEY_LOCAL_MACHINE\SYSTEM\...\Environment
  let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
  if let Ok(env) = hklm.open_subkey(r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
  {
    if let Ok(path) = env.get_value::<String, _>("Path") {
      paths.extend(path.split(';').map(|s| s.trim().to_string()));
    }
  }
  paths
}
