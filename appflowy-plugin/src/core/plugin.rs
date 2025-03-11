use crate::error::PluginError;
use crate::manager::WeakPluginState;
use std::fmt::{Display, Formatter};
use std::fs;
use std::process::Command;

use crate::core::parser::ResponseParser;
use crate::core::rpc_loop::RpcLoop;
use crate::core::rpc_peer::{CloneableCallback, OneShotCallback};
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use std::io::BufReader;
use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use tokio::sync::watch;
use tokio_stream::wrappers::{ReceiverStream, WatchStream};

#[cfg(windows)]
use winreg::{enums::*, RegKey};

use tracing::{error, info, trace};

#[derive(
  Default, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct PluginId(pub(crate) i64);

impl From<i64> for PluginId {
  fn from(id: i64) -> Self {
    PluginId(id)
  }
}

/// The `Peer` trait defines the interface for the opposite side of the RPC channel,
/// designed to be used behind a pointer or as a trait object.
pub trait Peer: Send + Sync + 'static {
  /// Clones the peer into a boxed trait object.
  fn box_clone(&self) -> Arc<dyn Peer>;

  /// Sends an RPC notification to the peer with the specified method and parameters.
  fn send_rpc_notification(&self, method: &str, params: &JsonValue);

  fn stream_rpc_request(&self, method: &str, params: &JsonValue, f: CloneableCallback);

  fn async_send_rpc_request(&self, method: &str, params: &JsonValue, f: Box<dyn OneShotCallback>);
  /// Sends a synchronous RPC request to the peer and waits for the result.
  /// Returns the result of the request or an error.
  fn send_rpc_request(&self, method: &str, params: &JsonValue) -> Result<JsonValue, PluginError>;

  /// Checks if there is an incoming request pending, intended to reduce latency for bulk operations done in the background.
  fn request_is_pending(&self) -> bool;

  /// Schedules a timer to execute the handler's `idle` function after the specified `Instant`.
  /// Note: This is not a high-fidelity timer. Regular RPC messages will always take priority over idle tasks.
  fn schedule_timer(&self, after: Instant, token: usize);
}

/// The `Peer` trait object.
pub type RpcPeer = Arc<dyn Peer>;

pub struct RpcCtx {
  pub peer: RpcPeer,
}

#[derive(Debug, Clone)]
pub enum RunningState {
  ReadyToConnect,
  /// The plugin is in the process of establishing a connection
  Connecting,
  /// The plugin has successfully established a connection
  Connected {
    plugin_id: PluginId,
  },
  /// The plugin is currently running
  Running {
    plugin_id: PluginId,
  },
  /// The plugin has been stopped intentionally
  Stopped {
    plugin_id: PluginId,
  },
  /// The plugin stopped unexpectedly
  UnexpectedStop {
    plugin_id: PluginId,
  },
}

impl RunningState {
  pub fn plugin_id(&self) -> Option<PluginId> {
    match self {
      RunningState::Connecting => None,
      RunningState::Connected { plugin_id } => Some(*plugin_id),
      RunningState::Running { plugin_id } => Some(*plugin_id),
      RunningState::Stopped { plugin_id } => Some(*plugin_id),
      RunningState::UnexpectedStop { plugin_id } => Some(*plugin_id),
      RunningState::ReadyToConnect => None,
    }
  }

  pub fn is_ready(&self) -> bool {
    matches!(self, RunningState::Running { .. })
  }

  pub fn is_loading(&self) -> bool {
    matches!(
      self,
      RunningState::Connecting | RunningState::Connected { .. }
    )
  }
}

pub type RunningStateSender = Arc<watch::Sender<RunningState>>;
pub type RunningStateReceiver = watch::Receiver<RunningState>;

#[derive(Clone)]
pub struct Plugin {
  peer: RpcPeer,
  pub(crate) id: PluginId,
  pub(crate) name: String,
  #[allow(dead_code)]
  pub(crate) process: Arc<Child>,
  pub(crate) running_state: RunningStateSender,
}

impl Display for Plugin {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}, plugin id: {:?}, process id: {}",
      self.name,
      self.id,
      self.process.id()
    )
  }
}

impl Plugin {
  pub fn initialize(&self, value: JsonValue) -> Result<(), PluginError> {
    self.peer.send_rpc_request("initialize", &value)?;
    Ok(())
  }

  pub fn request(&self, method: &str, params: &JsonValue) -> Result<JsonValue, PluginError> {
    self.peer.send_rpc_request(method, params)
  }

  pub async fn async_request<P: ResponseParser>(
    &self,
    method: &str,
    params: &JsonValue,
  ) -> Result<P::ValueType, PluginError> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    self.peer.async_send_rpc_request(
      method,
      params,
      Box::new(move |result| {
        let _ = tx.send(result);
      }),
    );
    let value = rx.await.map_err(|err| {
      PluginError::Internal(anyhow!("error waiting for async response: {:?}", err))
    })??;
    let value = P::parse_json(value)?;
    Ok(value)
  }

  pub fn stream_request<P: ResponseParser>(
    &self,
    method: &str,
    params: &JsonValue,
  ) -> Result<ReceiverStream<Result<P::ValueType, PluginError>>, PluginError> {
    let (tx, stream) = tokio::sync::mpsc::channel(100);
    let stream = ReceiverStream::new(stream);
    let callback = CloneableCallback::new(move |result| match result {
      Ok(json) => {
        let result = P::parse_json(json).map_err(PluginError::from);
        let _ = tx.blocking_send(result);
      },
      Err(err) => {
        let _ = tx.blocking_send(Err(err));
      },
    });
    self.peer.stream_rpc_request(method, params, callback);
    Ok(stream)
  }

  pub fn shutdown(&self) {
    match self.peer.send_rpc_request("shutdown", &json!({})) {
      Ok(_) => {
        info!("shutting down plugin {}", self);
      },
      Err(err) => {
        error!("error sending shutdown to plugin {}: {:?}", self, err);
      },
    }
  }

  pub fn subscribe_running_state(&self) -> WatchStream<RunningState> {
    WatchStream::new(self.running_state.subscribe())
  }
}

#[derive(Debug)]
pub struct PluginInfo {
  pub name: String,
  pub exec_path: PathBuf,
  pub exec_command: String,
}

pub(crate) async fn start_plugin_process(
  plugin_info: PluginInfo,
  id: PluginId,
  state: WeakPluginState,
  running_state: RunningStateSender,
) -> Result<(), anyhow::Error> {
  trace!("start plugin process: {:?}, {:?}", id, plugin_info);
  let (tx, ret) = tokio::sync::oneshot::channel();

  let spawn_result = thread::Builder::new()
    .name(format!("<{}> core host thread", &plugin_info.name))
    .spawn(move || {
      info!("Load {} plugin", &plugin_info.name);
      let mut command = if fs::metadata(&plugin_info.exec_path).is_ok() {
        // If exec_path exists, use it to start the process
        info!(
          "[AI Plugin]: run plugin with exec_path: {:?}",
          &plugin_info.exec_path
        );
        Command::new(&plugin_info.exec_path)
      } else {
        // Otherwise, use exec_command
        info!(
          "[AI Plugin]: run plugin with command: {:?}",
          &plugin_info.exec_command
        );
        #[cfg(windows)]
        {
          match get_exec_command_path(format!("{}.exe", &plugin_info.exec_command).as_str()) {
            Some(path) => {
              info!("Found plugin executable: {:?}", path);
              Command::new(&path)
            },
            None => {
              error!(
                "Plugin executable not found: {:?}",
                &plugin_info.exec_command
              );
              Command::new(&plugin_info.exec_command)
            },
          }
        }

        #[cfg(not(windows))]
        Command::new(&plugin_info.exec_command)
      };

      let child = command.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn();

      match child {
        Ok(mut child) => {
          let child_stdin = child.stdin.take().unwrap();
          let child_stdout = child.stdout.take().unwrap();
          let mut looper = RpcLoop::new(child_stdin, running_state.clone());
          let _ = running_state.send(RunningState::Connecting);

          let peer: RpcPeer = Arc::new(looper.get_raw_peer());
          let name = plugin_info.name.clone();
          peer.send_rpc_notification("ping", &JsonValue::Array(Vec::new()));

          let plugin = Plugin {
            peer,
            process: Arc::new(child),
            name,
            id,
            running_state: running_state.clone(),
          };

          let plugin_id = plugin.id;
          state.plugin_connect(Ok(plugin));
          if let Err(err) = running_state.send(RunningState::Connected { plugin_id }) {
            error!("failed to send connected state: {:?}", err);
          }
          // Notify the main thread that the plugin has started
          let _ = tx.send(());

          let mut state = state;
          let err = looper.mainloop(
            &plugin_info.name,
            &plugin_id,
            || BufReader::new(child_stdout),
            &mut state,
          );
          let _ = running_state.send(RunningState::Stopped { plugin_id });
          state.plugin_exit(id, err);
        },
        Err(err) => {
          let _ = tx.send(());
          error!("failed to start plugin process: {:?}", err);
          state.plugin_connect(Err(err));
        },
      }
    });

  if let Err(err) = spawn_result {
    error!("[RPC] thread spawn failed for {:?}, {:?}", id, err);
    return Err(err.into());
  }
  ret.await?;
  Ok(())
}
#[allow(dead_code)]
#[cfg(unix)]
async fn ensure_executable(exec_path: &std::path::Path) -> Result<(), anyhow::Error> {
  use std::os::unix::fs::PermissionsExt;

  let metadata = tokio::fs::metadata(exec_path).await?;
  let mut permissions = metadata.permissions();
  permissions.set_mode(permissions.mode() | 0o755);
  tokio::fs::set_permissions(exec_path, permissions).await?;
  Ok(())
}

#[allow(dead_code)]
#[cfg(target_os = "macos")]
pub fn handle_macos_security_check(plugin_info: &PluginInfo) {
  trace!("macos security check: {:?}", plugin_info.exec_path);
  let mut open_manually = false;
  match xattr::list(&plugin_info.exec_path) {
    Ok(list) => {
      let mut has_lastuseddate = false;

      // https://eclecticlight.co/2023/03/16/what-is-macos-ventura-doing-tracking-provenance/
      // The com.apple.quarantine attribute is used by macOS to mark files that have been downloaded from
      // the internet or received via other potentially unsafe methods. When this attribute is set, macOS
      // employs additional security checks before allowing the file to be opened or executed
      // The presence of this attribute can cause the system to display a permission error, such as:
      // code: 1, kind: PermissionDenied, message: "Operation not permitted"
      for attr in list {
        trace!("{:?}: xattr: {:?}", plugin_info.exec_path, attr);
        if attr == "com.apple.lastuseddate#PS" {
          has_lastuseddate = true;
        }
      }

      if !has_lastuseddate {
        open_manually = true;
      }
    },
    Err(err) => {
      error!("Failed to list xattr: {:?}", err);
      open_manually = true;
    },
  }

  if open_manually {
    trace!("Open plugin file manually: {:?}", plugin_info.exec_path);
    // Using 'open' to trigger the macOS security check. After the user allows opening the binary,
    // any subsequent 'open' command will not trigger the security check and the binary will run with permission.
    if let Err(err) = std::process::Command::new("open")
      .arg(&plugin_info.exec_path)
      .output()
    {
      error!("Failed to open plugin file: {:?}", err);
    }
  }
}

#[cfg(windows)]
fn get_exec_command_path(exec_command: &str) -> Option<PathBuf> {
  let path_dirs = get_windows_path_dirs();
  path_dirs
    .iter()
    .map(|dir| std::path::Path::new(dir).join(exec_command))
    .find(|path| path.exists())
}

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
