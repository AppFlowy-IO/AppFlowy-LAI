use crate::core::parser::ResponseParser;
use crate::core::plugin::{
  start_plugin_process, Plugin, PluginId, PluginInfo, RpcCtx, RunningStateSender,
};
use crate::core::rpc_loop::Handler;
use crate::core::rpc_peer::{PluginCommand, ResponsePayload};
use crate::error::{PluginError, ReadError, RemoteError};
use anyhow::anyhow;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::HashMap;
use std::io;

use crate::util::{get_operating_system, OperatingSystem};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Weak};
use tokio::sync::RwLock;
use tracing::{error, info, instrument, trace, warn};

pub struct PluginManager {
  state: Arc<Mutex<PluginState>>,
  plugin_id_counter: Arc<AtomicI64>,
  operating_system: OperatingSystem,
  running_plugins: RwLock<HashMap<String, PluginId>>,
}

impl Default for PluginManager {
  fn default() -> Self {
    Self::new()
  }
}

impl PluginManager {
  pub fn new() -> Self {
    PluginManager {
      state: Arc::new(Mutex::new(PluginState {
        plugins: Vec::new(),
      })),
      plugin_id_counter: Arc::new(Default::default()),
      operating_system: get_operating_system(),
      running_plugins: Default::default(),
    }
  }

  pub async fn create_plugin(
    &self,
    plugin_info: PluginInfo,
    running_state: RunningStateSender,
  ) -> Result<PluginId, PluginError> {
    if self.operating_system.is_not_desktop() {
      return Err(PluginError::Internal(anyhow!(
        "plugin not supported on this platform"
      )));
    }

    if self
      .running_plugins
      .read()
      .await
      .contains_key(&plugin_info.name)
    {
      return Err(PluginError::Internal(anyhow!("plugin already running")));
    }

    let plugin_id = PluginId::from(self.plugin_id_counter.fetch_add(1, Ordering::SeqCst));
    self
      .running_plugins
      .write()
      .await
      .insert(plugin_info.name.clone(), plugin_id);

    let weak_state = WeakPluginState(Arc::downgrade(&self.state));
    start_plugin_process(plugin_info, plugin_id, weak_state, running_state).await?;
    Ok(plugin_id)
  }

  pub async fn get_plugin(&self, plugin_id: PluginId) -> Result<Weak<Plugin>, PluginError> {
    let state = self.state.lock();
    let plugin = state
      .plugins
      .iter()
      .find(|p| p.id == plugin_id)
      .ok_or(PluginError::PluginNotConnected)?;
    Ok(Arc::downgrade(plugin))
  }

  #[instrument(skip(self), err)]
  pub async fn remove_plugin(&self, id: PluginId) -> Result<(), PluginError> {
    if self.operating_system.is_not_desktop() {
      return Err(PluginError::Internal(anyhow!(
        "plugin not supported on this platform"
      )));
    }

    info!("[AI Plugin] removing plugin {:?}", id);
    self.state.lock().plugin_disconnect(id, Ok(()));

    let mut running_plugins = self.running_plugins.write().await;
    let key_to_remove = running_plugins
      .iter()
      .find(|(_name, plugin_id)| **plugin_id == id)
      .map(|(name, _)| name.clone());
    if let Some(name) = key_to_remove {
      running_plugins.remove(&name);
    }
    Ok(())
  }

  pub async fn init_plugin(
    &self,
    id: PluginId,
    init_params: Value,
  ) -> Result<Arc<Plugin>, PluginError> {
    trace!("init plugin: {:?}, {:?}", id, init_params);
    if self.operating_system.is_not_desktop() {
      return Err(PluginError::Internal(anyhow!(
        "plugin not supported on this platform"
      )));
    }

    let plugin = self
      .get_plugin(id)
      .await?
      .upgrade()
      .ok_or_else(|| PluginError::PluginNotConnected)?;
    plugin.initialize(init_params)?;
    Ok(plugin.clone())
  }

  pub async fn send_request<P: ResponseParser>(
    &self,
    id: PluginId,
    method: &str,
    request: Value,
  ) -> Result<P::ValueType, PluginError> {
    let plugin = self
      .get_plugin(id)
      .await?
      .upgrade()
      .ok_or_else(|| PluginError::PluginNotConnected)?;
    let resp = plugin.request(method, &request)?;
    let value = P::parse_json(resp)?;
    Ok(value)
  }

  pub async fn async_send_request<P: ResponseParser>(
    &self,
    id: PluginId,
    method: &str,
    request: Value,
  ) -> Result<P::ValueType, PluginError> {
    let plugin = self
      .get_plugin(id)
      .await?
      .upgrade()
      .ok_or_else(|| PluginError::PluginNotConnected)?;
    let value = plugin.async_request::<P>(method, &request).await?;
    Ok(value)
  }
}

pub struct PluginState {
  plugins: Vec<Arc<Plugin>>,
}

impl PluginState {
  pub fn plugin_connect(&mut self, plugin: Result<Plugin, io::Error>) {
    match plugin {
      Ok(plugin) => {
        info!("[RPC] {} connected", plugin);
        self.plugins.push(Arc::new(plugin));
      },
      Err(err) => {
        error!("plugin failed to connect: {:?}", err);
      },
    }
  }

  pub fn plugin_disconnect(
    &mut self,
    id: PluginId,
    error: Result<(), ReadError>,
  ) -> Option<Arc<Plugin>> {
    if let Err(err) = error {
      error!("[RPC] plugin {:?} exited with result {:?}", id, err)
    }

    let running_idx = self.plugins.iter().position(|p| p.id == id);
    match running_idx {
      Some(idx) => {
        let plugin = self.plugins.remove(idx);
        plugin.shutdown();
        Some(plugin)
      },
      None => {
        warn!("[RPC] plugin {:?} not found", id);
        None
      },
    }
  }
}

#[derive(Clone)]
pub struct WeakPluginState(Weak<Mutex<PluginState>>);

impl WeakPluginState {
  pub fn upgrade(&self) -> Option<Arc<Mutex<PluginState>>> {
    self.0.upgrade()
  }

  pub fn plugin_connect(&self, plugin: Result<Plugin, io::Error>) {
    if let Some(state) = self.upgrade() {
      state.lock().plugin_connect(plugin)
    }
  }

  pub fn plugin_exit(&self, plugin: PluginId, error: Result<(), ReadError>) {
    if let Some(core) = self.upgrade() {
      core.lock().plugin_disconnect(plugin, error);
    }
  }
}

impl Handler for WeakPluginState {
  type Request = PluginCommand<String>;

  fn handle_request(
    &mut self,
    _ctx: &RpcCtx,
    rpc: Self::Request,
  ) -> Result<ResponsePayload, RemoteError> {
    trace!("handling request: {:?}", rpc.cmd);
    Ok(ResponsePayload::empty_json())
  }
}
