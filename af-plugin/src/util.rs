use anyhow::Result;
use tokio::process::Command;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatingSystem {
  Unknown,
  Windows,
  Linux,
  MacOS,
  IOS,
  Android,
}

impl OperatingSystem {
  pub fn is_not_ios(&self) -> bool {
    !matches!(self, OperatingSystem::IOS)
  }

  pub fn is_desktop(&self) -> bool {
    matches!(
      self,
      OperatingSystem::Windows | OperatingSystem::Linux | OperatingSystem::MacOS
    )
  }

  pub fn is_not_desktop(&self) -> bool {
    !self.is_desktop()
  }
}

impl From<String> for OperatingSystem {
  fn from(s: String) -> Self {
    OperatingSystem::from(s.as_str())
  }
}

impl From<&String> for OperatingSystem {
  fn from(s: &String) -> Self {
    OperatingSystem::from(s.as_str())
  }
}

impl From<&str> for OperatingSystem {
  fn from(s: &str) -> Self {
    match s {
      "windows" => OperatingSystem::Windows,
      "linux" => OperatingSystem::Linux,
      "macos" => OperatingSystem::MacOS,
      "ios" => OperatingSystem::IOS,
      "android" => OperatingSystem::Android,
      _ => OperatingSystem::Unknown,
    }
  }
}

pub fn get_operating_system() -> OperatingSystem {
  cfg_if::cfg_if! {
      if #[cfg(target_os = "android")] {
          OperatingSystem::Android
      } else if #[cfg(target_os = "ios")] {
          OperatingSystem::IOS
      } else if #[cfg(target_os = "macos")] {
          OperatingSystem::MacOS
      } else if #[cfg(target_os = "windows")] {
          OperatingSystem::Windows
      } else if #[cfg(target_os = "linux")] {
          OperatingSystem::Linux
      } else {
          OperatingSystem::Unknown
      }
  }
}

/// Checks if the current macOS is running on Apple Silicon (ARM-based) CPU.
/// Returns true if it is an Apple Silicon CPU, false otherwise.
pub async fn is_apple_silicon() -> Result<bool> {
  // Execute the `uname -m` command to get the machine architecture
  let output = Command::new("uname").arg("-m").output().await?;

  // Convert the output to a string
  let architecture = std::str::from_utf8(&output.stdout)?.trim();

  // Check if the architecture is `arm64`
  Ok(architecture == "arm64")
}
