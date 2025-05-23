use crate::core::plugin::{Peer, PluginId, RunningState, RunningStateSender};
use crate::core::rpc_object::RpcObject;
use crate::error::{PluginError, ReadError, RemoteError};
use parking_lot::{Condvar, Mutex};
use serde::{de, ser, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value as JsonValue};
use std::collections::{BTreeMap, BinaryHeap, VecDeque};
use std::fmt::{Debug, Display};
use std::io::Write;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use std::{cmp, io};
use tokio_stream::Stream;
use tracing::{error, info, trace, warn};

pub struct PluginCommand<T> {
  pub plugin_id: PluginId,
  pub cmd: T,
}

impl<T: Serialize> Serialize for PluginCommand<T> {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    let mut v = serde_json::to_value(&self.cmd).map_err(ser::Error::custom)?;
    v["params"]["plugin_id"] = json!(self.plugin_id);
    v.serialize(serializer)
  }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for PluginCommand<T> {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    #[derive(Deserialize)]
    struct PluginIdHelper {
      plugin_id: PluginId,
    }
    let v = JsonValue::deserialize(deserializer)?;
    let plugin_id = PluginIdHelper::deserialize(&v)
      .map_err(de::Error::custom)?
      .plugin_id;
    let cmd = T::deserialize(v).map_err(de::Error::custom)?;
    Ok(PluginCommand { plugin_id, cmd })
  }
}

pub struct RpcState<W: Write> {
  rx_queue: Mutex<VecDeque<Result<RpcObject, ReadError>>>,
  rx_cvar: Condvar,
  writer: Mutex<W>,
  request_id_counter: AtomicUsize,
  pending: Mutex<BTreeMap<usize, ResponseHandler>>,
  timers: Mutex<BinaryHeap<Timer>>,
  needs_exit: AtomicBool,
  is_blocking: AtomicBool,
  running_state: RunningStateSender,
}

impl<W: Write> RpcState<W> {
  /// Creates a new `RawPeer` instance.
  ///
  /// # Arguments
  ///
  /// * `writer` - An object implementing the `Write` trait, used for sending messages.
  ///
  /// # Returns
  ///
  /// A new `RawPeer` instance wrapped in an `Arc`.
  pub fn new(writer: W, running_state: RunningStateSender) -> Self {
    RpcState {
      rx_queue: Mutex::new(VecDeque::new()),
      rx_cvar: Condvar::new(),
      writer: Mutex::new(writer),
      request_id_counter: AtomicUsize::new(0),
      pending: Mutex::new(BTreeMap::new()),
      timers: Mutex::new(BinaryHeap::new()),
      needs_exit: AtomicBool::new(false),
      is_blocking: Default::default(),
      running_state,
    }
  }

  pub fn is_blocking(&self) -> bool {
    self.is_blocking.load(Ordering::Acquire)
  }
}

pub struct RawPeer<W: Write + 'static>(pub(crate) Arc<RpcState<W>>);

impl<W: Write + Send + 'static> Peer for RawPeer<W> {
  fn box_clone(&self) -> Arc<dyn Peer> {
    Arc::new((*self).clone())
  }
  fn send_rpc_notification(&self, method: &str, params: &JsonValue) {
    if let Err(e) = self.send(&json!({
        "method": method,
        "params": params,
    })) {
      error!(
        "send error on send_rpc_notification method {}: {}",
        method, e
      );
    }
  }

  fn stream_rpc_request(&self, method: &str, params: &JsonValue, f: CloneableCallback) {
    self.send_rpc(method, params, ResponseHandler::StreamCallback(Arc::new(f)));
  }

  fn async_send_rpc_request(&self, method: &str, params: &JsonValue, f: Box<dyn OneShotCallback>) {
    self.send_rpc(method, params, ResponseHandler::Callback(f));
  }

  fn send_rpc_request(&self, method: &str, params: &JsonValue) -> Result<JsonValue, PluginError> {
    let (tx, rx) = mpsc::channel();
    self.0.is_blocking.store(true, Ordering::Release);
    self.send_rpc(method, params, ResponseHandler::Chan(tx));
    let result = rx.recv().unwrap_or(Err(PluginError::PeerDisconnect));
    self.0.is_blocking.store(false, Ordering::Release);
    result
  }

  fn request_is_pending(&self) -> bool {
    let queue = self.0.rx_queue.lock();
    !queue.is_empty()
  }

  fn schedule_timer(&self, after: Instant, token: usize) {
    self.0.timers.lock().push(Timer {
      fire_after: after,
      token,
    });
  }
}

impl<W: Write> RawPeer<W> {
  /// Sends a JSON value to the peer.
  ///
  /// # Arguments
  ///
  /// * `json` - A reference to a `JsonValue` to be sent.
  ///
  /// # Returns
  ///
  /// A `Result` indicating success or an `io::Error` if the write operation fails.
  ///
  /// # Notes
  ///
  /// This function serializes the JSON value, appends a newline, and writes it to the underlying writer.
  fn send(&self, json: &JsonValue) -> Result<(), io::Error> {
    let mut s = serde_json::to_string(json)?;
    s.push('\n');
    self.0.writer.lock().write_all(s.as_bytes())
  }

  /// Sends a response to a previous RPC request.
  ///
  /// # Arguments
  ///
  /// * `result` - The `Response` to be sent.
  /// * `id` - The ID of the request being responded to.
  ///
  /// # Notes
  ///
  /// This function constructs a JSON response and sends it using the `send` method.
  /// It handles both successful results and errors.
  pub(crate) fn respond(&self, result: Response, id: u64) {
    let mut response = json!({ "id": id });
    match result {
      Ok(result) => match result {
        ResponsePayload::Json(value) => response["result"] = value,
        ResponsePayload::Streaming(_) | ResponsePayload::StreamEnd(_) => {
          error!("stream response not supported")
        },
      },
      Err(error) => response["error"] = json!(error),
    };
    if let Err(e) = self.send(&response) {
      error!("[RPC] error {} sending response to RPC {:?}", e, id);
    }
  }

  /// Sends an RPC request.
  ///
  /// # Arguments
  ///
  /// * `method` - The name of the RPC method to be called.
  /// * `params` - The parameters for the RPC call.
  /// * `response_handler` - A `ResponseHandler` to handle the response.
  ///
  /// # Notes
  ///
  /// This function generates a unique ID for the request, stores the response handler,
  /// and sends the RPC request. If sending fails, it immediately invokes the response handler with an error.
  fn send_rpc(&self, method: &str, params: &JsonValue, response_handler: ResponseHandler) {
    trace!("[RPC] call:{} :{:?}", method, params);
    let id = self.0.request_id_counter.fetch_add(1, Ordering::Relaxed);

    let msg = json!({
        "id": id,
        "method": method,
        "params": params,
    });

    if let Err(e) = self.send(&msg) {
      response_handler.invoke(Err(PluginError::Io(e)));
      return;
    }

    let mut pending = self.0.pending.lock();
    pending.insert(id, response_handler);
  }

  /// Processes an incoming response to an RPC request.
  ///
  /// This function is responsible for handling responses received from the peer, matching them
  /// to their corresponding requests, and invoking the appropriate callbacks. It supports both
  /// one-time responses and streaming responses.
  ///
  /// # Arguments
  ///
  /// * `&self` - A reference to the `RawPeer` instance.
  /// * `request_id: u64` - The unique identifier of the request to which this is a response.
  /// * `resp: Result<ResponsePayload, SidecarError>` - The response payload or an error.
  ///
  /// # Behavior
  ///
  /// 1. Retrieves and removes the response handler for the given `request_id` from the pending requests.
  /// 2. Determines if the response is part of a stream.
  /// 3. For streaming responses:
  ///    - If it's not the end of the stream, re-inserts the stream callback for future messages.
  ///    - If it's the end of the stream, logs this information.
  /// 4. Converts the response payload to JSON.
  /// 5. Invokes the response handler with the JSON data or error.
  ///
  /// # Concurrency
  ///
  /// This function uses mutex locks to ensure thread-safe access to shared data structures.
  /// It's designed to be called from multiple threads safely.
  ///
  /// # Error Handling
  ///
  /// - If no handler is found for the `request_id`, an error is logged.
  /// - If a non-stream response payload is `None`, a warning is logged.
  /// - Errors in the response are propagated to the response handler.
  pub(crate) fn handle_response(
    &self,
    request_id: u64,
    resp: Result<ResponsePayload, PluginError>,
  ) {
    let request_id = request_id as usize;
    let handler = {
      let mut pending = self.0.pending.lock();
      pending.remove(&request_id)
    };
    let is_stream = resp.as_ref().map(|resp| resp.is_stream()).unwrap_or(false);
    match handler {
      Some(response_handler) => {
        if is_stream {
          let is_stream_end = resp
            .as_ref()
            .map(|resp| resp.is_stream_end())
            .unwrap_or(false);
          if is_stream_end {
            trace!("[RPC] {} stream end", request_id);
          } else {
            // when steam is not end, we need to put the stream callback back to pending in order to
            // receive the next stream message.
            if let Some(callback) = response_handler.get_stream_callback() {
              let mut pending = self.0.pending.lock();
              pending.insert(request_id, ResponseHandler::StreamCallback(callback));
            }
          }
        }
        let json = resp.map(|resp| resp.into_json());
        match json {
          Ok(Some(json)) => {
            response_handler.invoke(Ok(json));
          },
          Ok(None) => {
            if !is_stream {
              warn!("[RPC] only stream response can be None");
            }
          },
          Err(err) => {
            response_handler.invoke(Err(err));
          },
        }
      },
      None => error!("[RPC] id {}'s handle not found", request_id),
    }
  }

  /// Get a message from the receive queue if available.
  pub(crate) fn try_get_rx(&self) -> Option<Result<RpcObject, ReadError>> {
    let mut queue = self.0.rx_queue.lock();
    queue.pop_front()
  }

  /// Get a message from the receive queue, waiting for at most `Duration`
  /// and returning `None` if no message is available.
  pub(crate) fn get_rx_timeout(&self, dur: Duration) -> Option<Result<RpcObject, ReadError>> {
    let mut queue = self.0.rx_queue.lock();
    let result = self.0.rx_cvar.wait_for(&mut queue, dur);
    if result.timed_out() {
      return None;
    }
    queue.pop_front()
  }

  /// Adds a message to the receive queue. The message should only
  /// be `None` if the read thread is exiting.
  pub(crate) fn put_rpc_object(&self, json: Result<RpcObject, ReadError>) {
    let mut queue = self.0.rx_queue.lock();
    queue.push_back(json);
    self.0.rx_cvar.notify_one();
  }

  /// Checks the status of the most imminent timer.
  ///
  /// # Returns
  ///
  /// - `Some(Ok(usize))`: If the most imminent timer has expired, returns its token.
  /// - `Some(Err(Duration))`: If the most imminent timer has not yet expired, returns the time until it expires.
  /// - `None`: If no timers are registered.
  pub(crate) fn check_timers(&self) -> Option<Result<usize, Duration>> {
    let mut timers = self.0.timers.lock();
    match timers.peek() {
      None => return None,
      Some(t) => {
        let now = Instant::now();
        if t.fire_after > now {
          return Some(Err(t.fire_after - now));
        }
      },
    }
    Some(Ok(timers.pop().unwrap().token))
  }

  pub(crate) fn shutdown(&self, plugin_id: &PluginId) {
    info!("[RPC] shutdown");
    self.handle_disconnect(RunningState::Stopped {
      plugin_id: *plugin_id,
    });
  }

  pub(crate) fn unexpected_disconnect<E: Debug>(&self, plugin_id: &PluginId, error: &E) {
    trace!("[RPC] disconnecting peer with error {:?}", error);
    self.handle_disconnect(RunningState::UnexpectedStop {
      plugin_id: *plugin_id,
    });
  }

  fn handle_disconnect(&self, state: RunningState) {
    let _ = self.0.running_state.send(state);
    let mut pending = self.0.pending.try_lock();
    if let Some(pending) = pending.as_mut() {
      let ids = pending.keys().cloned().collect::<Vec<_>>();
      for id in &ids {
        if let Some(callback) = pending.remove(id) {
          callback.invoke(Err(PluginError::PeerDisconnect));
        }
      }
    }
    drop(pending);

    trace!("[RPC] marking needs_exit");
    self.0.needs_exit.store(true, Ordering::SeqCst);
  }

  /// Checks if the RPC system needs to exit.
  pub(crate) fn needs_exit(&self) -> bool {
    self.0.needs_exit.load(Ordering::Relaxed)
  }

  pub(crate) fn reset_needs_exit(&self) {
    self.0.needs_exit.store(false, Ordering::SeqCst);
  }

  pub(crate) fn notify_running(&self, plugin_id: PluginId) {
    // if current running state is not equal to Running, we need to notify the plugin to start running.
    let is_running = matches!(*self.0.running_state.borrow(), RunningState::Running { .. });
    if !is_running {
      let _ = self
        .0
        .running_state
        .send(RunningState::Running { plugin_id });
    }
  }
}

impl<W: Write> Clone for RawPeer<W> {
  fn clone(&self) -> Self {
    RawPeer(self.0.clone())
  }
}

#[derive(Clone, Debug)]
pub enum ResponsePayload {
  Json(JsonValue),
  Streaming(JsonValue),
  StreamEnd(JsonValue),
}

impl ResponsePayload {
  pub fn empty_json() -> Self {
    ResponsePayload::Json(json!({}))
  }

  pub fn is_stream(&self) -> bool {
    matches!(
      self,
      ResponsePayload::Streaming(_) | ResponsePayload::StreamEnd(_)
    )
  }

  pub fn is_stream_end(&self) -> bool {
    matches!(self, ResponsePayload::StreamEnd(_))
  }

  pub fn into_json(self) -> Option<JsonValue> {
    match self {
      ResponsePayload::Json(v) => Some(v),
      ResponsePayload::Streaming(v) => Some(v),
      ResponsePayload::StreamEnd(_) => None,
    }
  }
}

impl Display for ResponsePayload {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      ResponsePayload::Json(v) => write!(f, "{}", v),
      ResponsePayload::Streaming(_) => write!(f, "stream start"),
      ResponsePayload::StreamEnd(_) => write!(f, "stream end"),
    }
  }
}

pub type Response = Result<ResponsePayload, RemoteError>;

pub trait ResponseStream: Stream<Item = Result<JsonValue, PluginError>> + Unpin + Send {}

impl<T> ResponseStream for T where T: Stream<Item = Result<JsonValue, PluginError>> + Unpin + Send {}

enum ResponseHandler {
  Chan(mpsc::Sender<Result<JsonValue, PluginError>>),
  Callback(Box<dyn OneShotCallback>),
  StreamCallback(Arc<CloneableCallback>),
}

impl ResponseHandler {
  pub fn get_stream_callback(&self) -> Option<Arc<CloneableCallback>> {
    match self {
      ResponseHandler::StreamCallback(cb) => Some(cb.clone()),
      _ => None,
    }
  }
}

pub trait OneShotCallback: Send {
  fn call(self: Box<Self>, result: Result<JsonValue, PluginError>);
}

impl<F: Send + FnOnce(Result<JsonValue, PluginError>)> OneShotCallback for F {
  fn call(self: Box<Self>, result: Result<JsonValue, PluginError>) {
    (self)(result)
  }
}

pub trait Callback: Send + Sync {
  fn call(&self, result: Result<JsonValue, PluginError>);
}

impl<F: Send + Sync + Fn(Result<JsonValue, PluginError>)> Callback for F {
  fn call(&self, result: Result<JsonValue, PluginError>) {
    (*self)(result)
  }
}

#[derive(Clone)]
pub struct CloneableCallback {
  callback: Arc<dyn Callback>,
}
impl CloneableCallback {
  pub fn new<C: Callback + 'static>(callback: C) -> Self {
    CloneableCallback {
      callback: Arc::new(callback),
    }
  }

  pub fn call(&self, result: Result<JsonValue, PluginError>) {
    self.callback.call(result)
  }
}

impl ResponseHandler {
  fn invoke(self, result: Result<JsonValue, PluginError>) {
    match self {
      ResponseHandler::Chan(tx) => {
        let _ = tx.send(result);
      },
      ResponseHandler::StreamCallback(cb) => {
        cb.call(result);
      },
      ResponseHandler::Callback(f) => f.call(result),
    }
  }
}
#[derive(Debug, PartialEq, Eq)]
struct Timer {
  fire_after: Instant,
  token: usize,
}

impl Ord for Timer {
  fn cmp(&self, other: &Timer) -> cmp::Ordering {
    other.fire_after.cmp(&self.fire_after)
  }
}

impl PartialOrd for Timer {
  fn partial_cmp(&self, other: &Timer) -> Option<cmp::Ordering> {
    Some(self.cmp(other))
  }
}
