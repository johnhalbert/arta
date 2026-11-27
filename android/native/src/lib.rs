//! JNI entry points for one local Arta conversation per Android app process.

use anyhow::{Context, Result, ensure};
use arta::{
    config::Config,
    harness::Harness,
    model::ModelId,
    provider::{
        Turn,
        llama_cpp::{LlamaCPP, LlamaCPPConfig, LlamaCPPModelConfig},
    },
    stream::Frame,
};
use futures_util::StreamExt;
use jni::{
    JNIEnv,
    objects::{JObject, JString, JValue},
    sys::{jboolean, jint, jobjectArray},
};
use llama_cpp_sys_4 as llama;
use std::{
    cell::RefCell,
    ffi::CStr,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::Mutex,
    time::Duration,
};
use tokio::{runtime::Builder, time};

const SESSION: &str = "android";
const AGENT: &str = "assistant";

static CHAT: Mutex<Chat> = Mutex::new(Chat {
    model: None,
    history: Vec::new(),
});

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub extern "system" fn JNI_OnLoad(
    _vm: *mut jni::sys::JavaVM,
    _reserved: *mut std::ffi::c_void,
) -> jint {
    unsafe { llama::ggml_set_abort_callback(Some(record_abort_message)) };
    jni::sys::JNI_VERSION_1_6
}

#[cfg(target_os = "android")]
unsafe extern "C" fn record_abort_message(message: *const std::ffi::c_char) {
    unsafe extern "C" {
        fn android_set_abort_message(message: *const std::ffi::c_char);
    }

    // ggml calls this with its error message immediately before aborting.
    unsafe { android_set_abort_message(message) };
}

struct LoadedModel {
    path: PathBuf,
    gpu_layers: u32,
    harness: Harness,
}

impl LoadedModel {
    fn open(path: PathBuf, gpu_layers: u32) -> Result<Self> {
        ensure!(
            gpu_layers == 0 || gpu_offload_available(),
            "Vulkan GPU offload is unavailable"
        );
        let mut config = Config::default();
        config.set_provider_settings(
            LlamaCPP::ID,
            LlamaCPPConfig {
                context_length: Some(2048),
                models: vec![LlamaCPPModelConfig {
                    id: config.app.model.clone(),
                    path: path.clone(),
                    gpu_layers,
                    devices: (gpu_layers == 0).then(Vec::new),
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;
        let mut harness = Harness::new(&config)?;
        let model = ModelId {
            provider: LlamaCPP::ID,
            model: &config.app.model,
        };
        harness.create_session(SESSION, &[(AGENT, model)])?;
        Ok(Self {
            path,
            gpu_layers,
            harness,
        })
    }
}

fn gpu_offload_available() -> bool {
    cfg!(feature = "vulkan") && unsafe { llama::llama_supports_gpu_offload() }
}

fn vulkan_devices() -> Vec<String> {
    if !cfg!(feature = "vulkan") {
        return Vec::new();
    }
    // The static registry owns these handles and strings. Only read device metadata here.
    unsafe {
        let backend = llama::ggml_backend_reg_by_name(c"Vulkan".as_ptr());
        if backend.is_null() {
            return Vec::new();
        }
        (0..llama::ggml_backend_reg_dev_count(backend))
            .map(|index| {
                let device = llama::ggml_backend_reg_dev_get(backend, index);
                let name = CStr::from_ptr(llama::ggml_backend_dev_name(device)).to_string_lossy();
                let description =
                    CStr::from_ptr(llama::ggml_backend_dev_description(device)).to_string_lossy();
                if description.is_empty() || description == name {
                    name.into_owned()
                } else {
                    format!("{description} ({name})")
                }
            })
            .collect()
    }
}

fn gpu_layers_from_java(layers: jint) -> Result<u32> {
    if layers == -1 {
        Ok(u32::MAX)
    } else {
        u32::try_from(layers).context("invalid GPU layer count")
    }
}

struct Chat {
    model: Option<LoadedModel>,
    history: Vec<Turn>,
}

impl Chat {
    fn replace_model(&mut self, path: PathBuf, gpu_layers: u32) -> Result<()> {
        // llama.cpp owns a process-wide backend. Release the old harness before opening another.
        let previous = self
            .model
            .take()
            .map(|model| (model.path, model.gpu_layers));
        match LoadedModel::open(path, gpu_layers) {
            Ok(model) => {
                self.model = Some(model);
                Ok(())
            }
            Err(load_error) => {
                match previous
                    .map(|(path, layers)| LoadedModel::open(path, layers))
                    .transpose()
                {
                    Ok(model) => self.model = model,
                    Err(restore_error) => {
                        self.history.clear();
                        return Err(load_error.context(format!(
                            "could not restore the previous model: {restore_error:#}"
                        )));
                    }
                }
                Err(load_error)
            }
        }
    }

    fn send(&mut self, prompt: String, env: &mut JNIEnv<'_>, sink: &JObject<'_>) -> Result<bool> {
        ensure!(!prompt.trim().is_empty(), "enter a prompt");
        let model = self.model.as_ref().context("import a GGUF model first")?;
        let agent = model.harness.agent(SESSION, AGENT)?;
        let runtime = Builder::new_current_thread().enable_time().build()?;
        // The cancellation and reply futures share JNI on this thread. Neither borrow crosses an await.
        let env = RefCell::new(env);
        let is_cancelled = || {
            env.borrow_mut()
                .call_method(sink, "isCancelled", "()Z", &[])?
                .z()
        };
        let reply = runtime.block_on(async {
            tokio::select! {
                biased;
                cancelled = wait_for_cancel(is_cancelled) => {
                    cancelled?;
                    Ok::<_, anyhow::Error>(None)
                }
                reply = async {
                    let mut chunks = agent.chat(&self.history, &prompt).await?.into_stream();
                    let mut reply = String::new();
                    while let Some(frame) = chunks.next().await {
                        let Frame::Chunk(chunk) = frame? else { continue };
                        env.borrow_mut().with_local_frame(4, |env| -> jni::errors::Result<()> {
                            let chunk = env.new_string(&chunk)?;
                            env.call_method(
                                sink, "onChunk", "(Ljava/lang/String;)V",
                                &[JValue::Object(chunk.as_ref())],
                            )?;
                            Ok(())
                        })?;
                        reply.push_str(&chunk);
                    }
                    Ok::<_, anyhow::Error>(reply)
                } => Ok(Some(reply?)),
            }
        })?;
        // Dropping the runtime waits for llama.cpp's blocking worker to release the model.
        drop(runtime);
        if let Some(reply) = reply
            && !is_cancelled()?
        {
            self.history.push(Turn { prompt, reply });
            return Ok(true);
        }
        Ok(false)
    }
}

async fn wait_for_cancel(is_cancelled: impl Fn() -> jni::errors::Result<bool>) -> Result<()> {
    let mut timer = time::interval(Duration::from_millis(40));
    loop {
        timer.tick().await;
        if is_cancelled()? {
            return Ok(());
        }
    }
}

fn with_chat<'local, T: Default>(
    mut env: JNIEnv<'local>,
    action: impl FnOnce(&mut JNIEnv<'local>, &mut Chat) -> Result<T>,
) -> T {
    let result = catch_unwind(AssertUnwindSafe(|| {
        CHAT.lock()
            .map_err(|_| anyhow::anyhow!("chat is unavailable"))
            .and_then(|mut chat| action(&mut env, &mut chat))
    }));
    let message = match result {
        Ok(Ok(value)) => return value,
        Ok(Err(error)) => format!("{error:#}"),
        Err(_) => "Arta failed inside the native chat bridge".into(),
    };
    if !env.exception_check().unwrap_or(true) {
        let _ = env.throw_new("java/lang/RuntimeException", message);
    }
    T::default()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_arta_mobile_NativeChat_loadModel<'local>(
    env: JNIEnv<'local>,
    _this: JObject<'local>,
    path: JString<'local>,
    gpu_layers: jint,
) {
    with_chat(env, |env, chat| {
        let path = PathBuf::from(String::from(env.get_string(&path)?));
        chat.replace_model(path, gpu_layers_from_java(gpu_layers)?)?;
        chat.history.clear();
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_arta_mobile_NativeChat_vulkanDevices<'local>(
    env: JNIEnv<'local>,
    _this: JObject<'local>,
) -> jobjectArray {
    with_chat(env, |env, _| {
        let devices = vulkan_devices();
        let array = env.new_object_array(
            jint::try_from(devices.len())?,
            "java/lang/String",
            JObject::null(),
        )?;
        for (index, device) in devices.into_iter().enumerate() {
            let name = env.auto_local(env.new_string(device)?);
            env.set_object_array_element(&array, index as jint, &name)?;
        }
        Ok(array.into_raw())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_arta_mobile_NativeChat_setGpuLayers<'local>(
    env: JNIEnv<'local>,
    _this: JObject<'local>,
    gpu_layers: jint,
) {
    with_chat(env, |_, chat| {
        let model = chat.model.as_ref().context("import a GGUF model first")?;
        let layers = gpu_layers_from_java(gpu_layers)?;
        if layers != model.gpu_layers {
            chat.replace_model(model.path.clone(), layers)?;
        }
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_arta_mobile_NativeChat_hasModel<'local>(
    env: JNIEnv<'local>,
    _this: JObject<'local>,
) -> jboolean {
    with_chat(env, |_, chat| Ok(jboolean::from(chat.model.is_some())))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_arta_mobile_NativeChat_reset<'local>(
    env: JNIEnv<'local>,
    _this: JObject<'local>,
) {
    with_chat(env, |_, chat| {
        chat.history.clear();
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_arta_mobile_NativeChat_restoreTurn<'local>(
    env: JNIEnv<'local>,
    _this: JObject<'local>,
    prompt: JString<'local>,
    reply: JString<'local>,
) {
    with_chat(env, |env, chat| {
        ensure!(chat.model.is_some(), "import a GGUF model first");
        let prompt = env.get_string(&prompt)?.into();
        let reply = env.get_string(&reply)?.into();
        chat.history.push(Turn { prompt, reply });
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_arta_mobile_NativeChat_send<'local>(
    env: JNIEnv<'local>,
    _this: JObject<'local>,
    prompt: JString<'local>,
    sink: JObject<'local>,
) -> jboolean {
    with_chat(env, |env, chat| {
        let prompt = env.get_string(&prompt)?.into();
        Ok(jboolean::from(chat.send(prompt, env, &sink)?))
    })
}
