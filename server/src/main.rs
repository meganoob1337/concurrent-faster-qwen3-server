mod batch;

use anyhow::Result;
use axum::{
    body::Body,
    extract::State,
    http::{StatusCode, HeaderMap},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use batch::{BatchEngine, BatchEngineConfig, BatchRequest, VoiceCloneData, build_voice_clone_prompts};
use tower_http::cors::{CorsLayer, Any};
use hound::{SampleFormat, WavSpec, WavWriter};
use qwen3_tts::{Language, ModelType, Speaker, SynthesisOptions};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Cursor, sync::Arc, sync::atomic::{AtomicU64, Ordering}};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::info;

fn rand_u64() -> u64 {
    let mut buf = [0u8; 8];
    std::fs::File::open("/dev/urandom").and_then(|mut f| { use std::io::Read; f.read_exact(&mut buf) }).unwrap_or_default();
    u64::from_ne_bytes(buf)
}

fn hash_bytes(data: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut h);
    h.finish()
}

struct Metrics {
    requests_total: AtomicU64,
    requests_streaming: AtomicU64,
    errors_total: AtomicU64,
    audio_seconds_total: AtomicU64, // stored as milliseconds
    gen_seconds_total: AtomicU64,   // stored as milliseconds
}

impl Metrics {
    fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_streaming: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            audio_seconds_total: AtomicU64::new(0),
            gen_seconds_total: AtomicU64::new(0),
        }
    }
}

struct AppState {
    tx: mpsc::Sender<BatchRequest>,
    stream_tx: mpsc::Sender<StreamingRequest>,
    semaphore: Arc<Semaphore>,
    max_inflight: usize,
    max_batch: usize,
    metrics: Arc<Metrics>,
    prompt_cache: Arc<std::sync::Mutex<HashMap<u64, Arc<qwen3_tts::VoiceClonePrompt>>>>,
    model: Arc<qwen3_tts::Qwen3TTS>,
}

#[derive(Deserialize)]
struct SpeechRequest {
    text: String,
    #[serde(default = "default_language")]
    language: String,
    #[serde(default)]
    ref_audio: Option<String>,
    #[serde(default)]
    ref_text: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    voice_id: Option<String>,
    /// Preset CustomVoice speaker name. Also accepted via `voice_id`.
    /// Not tied to `language`; any supported language works with any speaker.
    #[serde(default)]
    speaker: Option<String>,
    #[serde(default)]
    sample_rate: Option<u32>,
}

#[derive(Deserialize)]
struct PreloadRequest {
    ref_audio: String,
    #[serde(default)]
    ref_text: Option<String>,
    #[serde(default)]
    voice_id: Option<String>,
}

#[derive(Serialize)]
struct PreloadResponse {
    voice_id: String,
    cached: bool,
}

fn default_language() -> String { "spanish".into() }

fn is_custom_voice(model: &qwen3_tts::Qwen3TTS) -> bool {
    matches!(model.model_type(), Some(ModelType::CustomVoice))
}

fn model_type_name(model: &qwen3_tts::Qwen3TTS) -> &'static str {
    match model.model_type() {
        Some(ModelType::CustomVoice) => "custom_voice",
        Some(ModelType::VoiceDesign) => "voice_design",
        Some(ModelType::Base) | None => "base",
    }
}

fn speaker_list() -> String {
    Speaker::all().iter().map(|speaker| speaker.as_str()).collect::<Vec<_>>().join(", ")
}

fn fallback_speaker() -> Speaker {
    std::env::var("DEFAULT_SPEAKER")
        .ok()
        .and_then(|name| name.parse().ok())
        .unwrap_or(Speaker::Serena)
}

struct ResolvedVoice {
    voice_clone: Option<VoiceCloneData>,
    cached_prompt: Option<Arc<qwen3_tts::VoiceClonePrompt>>,
    speaker: Speaker,
}

fn resolve_voice(
    state: &AppState,
    req: &SpeechRequest,
) -> Result<ResolvedVoice, (StatusCode, String)> {
    if is_custom_voice(&state.model) {
        if req.ref_audio.as_ref().is_some_and(|audio| !audio.is_empty()) {
            return Err((
                StatusCode::BAD_REQUEST,
                "CustomVoice models do not support voice cloning".into(),
            ));
        }
        let name = req
            .speaker
            .as_deref()
            .filter(|name| !name.is_empty())
            .or(req.voice_id.as_deref().filter(|name| !name.is_empty()));
        let speaker = match name {
            Some(name) => name.parse().map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("unknown speaker '{name}'. Available: {}", speaker_list()),
                )
            })?,
            None => fallback_speaker(),
        };
        return Ok(ResolvedVoice {
            voice_clone: None,
            cached_prompt: None,
            speaker,
        });
    }

    if let Some(vid) = &req.voice_id {
        let hash = hash_bytes(vid.as_bytes());
        let prompt = state.prompt_cache.lock().ok().and_then(|cache| {
            cache
                .get(&hash)
                .or_else(|| u64::from_str_radix(vid, 16).ok().and_then(|h| cache.get(&h)))
                .cloned()
        });
        match prompt {
            Some(cached_prompt) => Ok(ResolvedVoice {
                voice_clone: None,
                cached_prompt: Some(cached_prompt),
                speaker: fallback_speaker(),
            }),
            None => Err((
                StatusCode::NOT_FOUND,
                format!("voice_id '{vid}' not found — preload first"),
            )),
        }
    } else {
        let voice_clone = decode_ref_audio(req)?;
        Ok(ResolvedVoice {
            voice_clone,
            cached_prompt: None,
            speaker: fallback_speaker(),
        })
    }
}

/// Split on `.?!` so a paragraph longer than one streaming budget can finish.
/// Commas and colons stay in the sentence. A clause is not a new generation.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

#[derive(Serialize)]
struct HealthResponse { status: &'static str, queue_depth: usize, max_batch: usize }

#[derive(Serialize)]
struct ErrorResponse { error: String }

fn parse_language(s: &str) -> Result<Language, String> {
    // The HTTP layer used to list a subset. The model supports the full set.
    s.parse().map_err(|err: anyhow::Error| err.to_string())
}

/// Max text length in characters (configurable via MAX_TEXT_CHARS env var)
fn max_text_chars() -> usize {
    std::env::var("MAX_TEXT_CHARS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000)
}

fn validate_request(req: &SpeechRequest) -> Result<(), (StatusCode, String)> {
    if req.text.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "text must not be empty".into()));
    }
    let limit = max_text_chars();
    if req.text.len() > limit {
        return Err((StatusCode::BAD_REQUEST, format!("text exceeds {limit} character limit")));
    }
    parse_language(&req.language).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    if let Some(t) = req.temperature {
        if !(0.0..=1.0).contains(&t) {
            return Err((StatusCode::BAD_REQUEST, format!("temperature must be 0.0-1.0, got {t}")));
        }
    }
    if let Some(sr) = req.sample_rate {
        if !VALID_SAMPLE_RATES.contains(&sr) {
            return Err((StatusCode::BAD_REQUEST, format!("sample_rate must be one of {:?}, got {sr}", VALID_SAMPLE_RATES)));
        }
    }
    Ok(())
}

const VALID_SAMPLE_RATES: &[u32] = &[8000, 16000, 22050, 24000, 44100, 48000];

fn resample_if_needed(samples: &[f32], from_rate: u32, to_rate: Option<u32>) -> (Vec<f32>, u32) {
    match to_rate {
        Some(rate) if rate != from_rate && VALID_SAMPLE_RATES.contains(&rate) => {
            let buf = qwen3_tts::AudioBuffer::new(samples.to_vec(), from_rate);
            match qwen3_tts::audio::resample::resample(&buf, rate) {
                Ok(resampled) => (resampled.samples, rate),
                Err(_) => (samples.to_vec(), from_rate),
            }
        }
        _ => (samples.to_vec(), from_rate),
    }
}

fn audio_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: SampleFormat::Int };
    let mut buf = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut buf, spec)?;
    for &s in samples { writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?; }
    writer.finalize()?;
    Ok(buf.into_inner())
}

fn samples_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        pcm.extend_from_slice(&v.to_le_bytes());
    }
    pcm
}

fn wav_header(sample_rate: u32, data_len: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + data_len).to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&sample_rate.to_le_bytes());
    h.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    h.extend_from_slice(&2u16.to_le_bytes()); // block align
    h.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_len.to_le_bytes());
    h
}


async fn preload_embedding(State(state): State<Arc<AppState>>, Json(req): Json<PreloadRequest>) -> Response {
    if is_custom_voice(&state.model) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "CustomVoice models do not support voice cloning".into(),
            }),
        )
            .into_response();
    }
    let bytes = match base64::engine::general_purpose::STANDARD.decode(&req.ref_audio) {
        Ok(b) => b,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: format!("invalid base64: {e}") })).into_response(),
    };
    let audio_hash = hash_bytes(&bytes);
    let voice_id = req.voice_id.unwrap_or_else(|| format!("{:016x}", audio_hash));

    // Check if already cached (by audio hash or voice_id name)
    let vid_hash = hash_bytes(voice_id.as_bytes());
    if let Ok(c) = state.prompt_cache.lock() {
        if c.contains_key(&audio_hash) || c.contains_key(&vid_hash) {
            return Json(PreloadResponse { voice_id, cached: true }).into_response();
        }
    }

    // Write temp file, load, encode
    let tmp = std::env::temp_dir().join(format!("preload_{:016x}.wav", rand_u64()));
    if let Err(e) = std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp)
        .and_then(|mut f| { use std::io::Write; f.write_all(&bytes) }) {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{e}") })).into_response();
    }
    let result = (|| -> anyhow::Result<Arc<qwen3_tts::VoiceClonePrompt>> {
        let ref_buf = qwen3_tts::AudioBuffer::load(&tmp)?;
        let prompt = state.model.create_voice_clone_prompt(&ref_buf, req.ref_text.as_deref())?;
        Ok(Arc::new(prompt))
    })();
    let _ = std::fs::remove_file(&tmp);

    match result {
        Ok(prompt) => {
            if let Ok(mut c) = state.prompt_cache.lock() {
                c.insert(audio_hash, prompt.clone());
                c.insert(vid_hash, prompt); // also cache by voice_id name
            }
            Json(PreloadResponse { voice_id, cached: false }).into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: format!("{e}") })).into_response(),
    }
}
async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let queue = state.max_inflight.saturating_sub(state.semaphore.available_permits());
    Json(HealthResponse { status: "ok", queue_depth: queue, max_batch: state.max_batch })
}

#[derive(Serialize)]
struct VoiceInfo {
    name: String,
}

#[derive(Serialize)]
struct VoicesResponse {
    model_type: String,
    voices: Vec<VoiceInfo>,
    languages: Vec<String>,
}

async fn list_voices(State(state): State<Arc<AppState>>) -> Json<VoicesResponse> {
    let model_type = model_type_name(&state.model);
    let voices = if model_type == "custom_voice" {
        Speaker::all()
            .iter()
            .map(|speaker| VoiceInfo { name: speaker.as_str().to_string() })
            .collect()
    } else {
        Vec::new()
    };
    let languages = Language::all().iter().map(|language| language.as_str().to_string()).collect();
    Json(VoicesResponse {
        model_type: model_type.to_string(),
        voices,
        languages,
    })
}

async fn metrics(State(state): State<Arc<AppState>>) -> String {
    let m = &state.metrics;
    let audio_s = m.audio_seconds_total.load(Ordering::Relaxed) as f64 / 1000.0;
    let gen_s = m.gen_seconds_total.load(Ordering::Relaxed) as f64 / 1000.0;
    let avg_rtf = if gen_s > 0.0 { audio_s / gen_s } else { 0.0 };
    format!(
        "# HELP tts_requests_total Total synthesis requests\n# TYPE tts_requests_total counter\ntts_requests_total {}\n\
         # HELP tts_requests_streaming Total streaming requests\n# TYPE tts_requests_streaming counter\ntts_requests_streaming {}\n\
         # HELP tts_errors_total Total errors\n# TYPE tts_errors_total counter\ntts_errors_total {}\n\
         # HELP tts_audio_seconds_total Total audio generated (seconds)\n# TYPE tts_audio_seconds_total counter\ntts_audio_seconds_total {:.1}\n\
         # HELP tts_gen_seconds_total Total generation time (seconds)\n# TYPE tts_gen_seconds_total counter\ntts_gen_seconds_total {:.1}\n\
         # HELP tts_avg_rtf Average real-time factor\n# TYPE tts_avg_rtf gauge\ntts_avg_rtf {:.2}\n\
         # HELP tts_queue_depth Current queue depth\n# TYPE tts_queue_depth gauge\ntts_queue_depth {}\n",
        m.requests_total.load(Ordering::Relaxed),
        m.requests_streaming.load(Ordering::Relaxed),
        m.errors_total.load(Ordering::Relaxed),
        audio_s, gen_s, avg_rtf,
        state.max_inflight.saturating_sub(state.semaphore.available_permits()),
    )
}

async fn synthesize(State(state): State<Arc<AppState>>, Json(req): Json<SpeechRequest>) -> Response {
    let t0 = std::time::Instant::now();
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    if let Err((status, msg)) = validate_request(&req) {
        state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
        return (status, Json(ErrorResponse { error: msg })).into_response();
    }

    if req.stream.unwrap_or(false) {
        state.metrics.requests_streaming.fetch_add(1, Ordering::Relaxed);
        // One generation is capped at STREAM_MAX_FRAMES. Split a paragraph on
        // sentence boundaries so each piece can finish. A normal sentence stays whole.
        if state.model.exceeds_stream_frame_budget(&req.text).unwrap_or(false) {
            let sentences = split_sentences(&req.text);
            if sentences.len() > 1 {
                return synthesize_streaming_split(state, req).await;
            }
        }
        return synthesize_streaming(state, req).await;
    }

    let permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => { state.metrics.errors_total.fetch_add(1, Ordering::Relaxed); return (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse { error: "Queue full".into() })).into_response(); },
    };

    let resolved = match resolve_voice(&state, &req) {
        Ok(resolved) => resolved,
        Err((status, msg)) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            return (status, Json(ErrorResponse { error: msg })).into_response();
        }
    };

    let (reply_tx, reply_rx) = oneshot::channel();
    let target_sample_rate = req.sample_rate;
    let batch_req = BatchRequest {
        text: req.text,
        language: parse_language(&req.language).unwrap(),
        voice_clone: resolved.voice_clone,
        cached_prompt: resolved.cached_prompt,
        speaker: resolved.speaker,
        options: SynthesisOptions { temperature: req.temperature.unwrap_or(0.7), ..SynthesisOptions::default() },
        reply: reply_tx,
    };

    if state.tx.send(batch_req).await.is_err() {
        drop(permit);
        state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Engine down".into() })).into_response();
    }

    match reply_rx.await {
        Ok(Ok(result)) => {
            drop(permit);
            let duration = result.audio.samples.len() as f32 / result.audio.sample_rate as f32;
            let rtf = duration / result.gen_time_secs;
            state.metrics.audio_seconds_total.fetch_add((duration * 1000.0) as u64, Ordering::Relaxed);
            state.metrics.gen_seconds_total.fetch_add((result.gen_time_secs * 1000.0) as u64, Ordering::Relaxed);
            info!(duration, gen_time = result.gen_time_secs, rtf, "Done");
            let (final_samples, final_rate) = resample_if_needed(&result.audio.samples, result.audio.sample_rate, target_sample_rate);
            match audio_to_wav_bytes(&final_samples, final_rate) {
                Ok(wav) => {
                    let ttfa_ms = t0.elapsed().as_millis();
                    (StatusCode::OK, [("content-type", "audio/wav"), ("x-rtf", &format!("{rtf:.2}")), ("x-ttfa-ms", &ttfa_ms.to_string())], wav).into_response()
                },
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{e:#}") })).into_response(),
            }
        }
        Ok(Err(e)) => { drop(permit); state.metrics.errors_total.fetch_add(1, Ordering::Relaxed); (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{e:#}") })).into_response() }
        Err(_) => { drop(permit); state.metrics.errors_total.fetch_add(1, Ordering::Relaxed); (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Dropped".into() })).into_response() }
    }
}


/// Streaming synthesis with automatic sentence splitting for voice clone.
/// Generates each sentence separately and streams all chunks sequentially.
async fn synthesize_streaming_split(state: Arc<AppState>, req: SpeechRequest) -> Response {
    let t0 = std::time::Instant::now();
    let sentences = split_sentences(&req.text);
    if sentences.is_empty() {
        return synthesize_streaming(state, req).await;
    }

    let (tx, mut rx) = mpsc::channel::<Result<Vec<u8>, String>>(64);

    // Spawn task to generate each sentence and forward chunks
    let state2 = state.clone();
    let voice_id = req.voice_id.clone();
    let speaker = req.speaker.clone();
    let language = req.language.clone();
    let temperature = req.temperature;
    tokio::spawn(async move {
        let mut first = true;
        for sentence in sentences {
            let (part_tx, mut part_rx) = mpsc::channel::<Result<Vec<u8>, String>>(32);
            let part_req = SpeechRequest {
                text: sentence,
                language: language.clone(),
                ref_audio: None,
                ref_text: None,
                temperature,
                stream: Some(true),
                voice_id: voice_id.clone(),
                speaker: speaker.clone(),
                sample_rate: None,
            };

            let resolved = match resolve_voice(&state2, &part_req) {
                Ok(resolved) => resolved,
                Err((_, msg)) => {
                    let _ = tx.send(Err(msg)).await;
                    return;
                }
            };

            let stream_req = StreamingRequest {
                text: part_req.text.clone(),
                language: parse_language(&part_req.language).unwrap(),
                temperature: part_req.temperature.unwrap_or(0.7),
                voice_clone: resolved.voice_clone,
                cached_prompt: resolved.cached_prompt,
                speaker: resolved.speaker,
                tx: part_tx,
            };

            if state2.stream_tx.try_send(stream_req).is_err() {
                let _ = tx.send(Err("Stream queue full".into())).await;
                return;
            }

            // Forward chunks, skip WAV header for subsequent sentences
            while let Some(chunk) = part_rx.recv().await {
                match chunk {
                    Ok(data) => {
                        if first && data.len() == 44 {
                            // First sentence WAV header — forward it
                            if tx.send(Ok(data)).await.is_err() { return; }
                            first = false;
                        } else if !first && data.len() == 44 {
                            // Subsequent sentence WAV header — skip
                            continue;
                        } else {
                            if tx.send(Ok(data)).await.is_err() { return; }
                            first = false;
                        }
                    }
                    Err(e) => { let _ = tx.send(Err(e)).await; return; }
                }
            }
        }
    });

    // Wait for first chunk (WAV header)
    let first_chunk = match rx.recv().await {
        Some(Ok(data)) => data,
        Some(Err(e)) => { state.metrics.errors_total.fetch_add(1, Ordering::Relaxed); return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })).into_response(); },
        None => { state.metrics.errors_total.fetch_add(1, Ordering::Relaxed); return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "No audio".into() })).into_response(); },
    };
    let ttfa_ms = t0.elapsed().as_millis();

    let rest = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|r: Result<Vec<u8>, String>| r.map_err(std::io::Error::other));
    let first_stream = tokio_stream::once(Ok::<Vec<u8>, std::io::Error>(first_chunk));
    let body = Body::from_stream(first_stream.chain(rest));

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "audio/wav")
        .header("transfer-encoding", "chunked")
        .header("x-audio-format", "pcm-s16le-24000-mono")
        .header("x-ttfa-ms", ttfa_ms.to_string())
        .header("x-sentence-split", "true")
        .body(body)
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "stream build failed").into_response())
}
async fn synthesize_streaming(state: Arc<AppState>, req: SpeechRequest) -> Response {
    let t0 = std::time::Instant::now();
    let language = parse_language(&req.language).unwrap();
    let text = req.text.clone();

    let resolved = match resolve_voice(&state, &req) {
        Ok(resolved) => resolved,
        Err((status, msg)) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            return (status, Json(ErrorResponse { error: msg })).into_response();
        }
    };

    let (tx, mut rx) = mpsc::channel::<Result<Vec<u8>, String>>(32);

    let stream_req = StreamingRequest {
        text,
        language,
        temperature: req.temperature.unwrap_or(0.7),
        voice_clone: resolved.voice_clone,
        cached_prompt: resolved.cached_prompt,
        speaker: resolved.speaker,
        tx,
    };
    if let Err(_) = state.stream_tx.try_send(stream_req) {
        state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
        return (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse { error: "Stream queue full".into() })).into_response();
    }

    // Wait for first chunk to measure TTFA and include in headers
    let first_chunk = match rx.recv().await {
        Some(Ok(data)) => data,
        Some(Err(e)) => { state.metrics.errors_total.fetch_add(1, Ordering::Relaxed); return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e })).into_response(); },
        None => { state.metrics.errors_total.fetch_add(1, Ordering::Relaxed); return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "No audio generated".into() })).into_response(); },
    };
    let ttfa_ms = t0.elapsed().as_millis();

    // Stream: first chunk + remaining chunks
    let rest = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|r: Result<Vec<u8>, String>| r.map_err(std::io::Error::other));
    let first_stream = tokio_stream::once(Ok::<Vec<u8>, std::io::Error>(first_chunk));
    let body = Body::from_stream(first_stream.chain(rest));

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "audio/wav")
        .header("transfer-encoding", "chunked")
        .header("x-audio-format", "pcm-s16le-24000-mono")
        .header("x-ttfa-ms", ttfa_ms.to_string())
        .body(body)
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "stream build failed").into_response())
}

struct StreamingRequest {
    text: String,
    language: Language,
    temperature: f64,
    voice_clone: Option<VoiceCloneData>,
    cached_prompt: Option<Arc<qwen3_tts::VoiceClonePrompt>>,
    speaker: Speaker,
    tx: mpsc::Sender<Result<Vec<u8>, String>>,
}

fn start_streaming_worker(model: Arc<qwen3_tts::Qwen3TTS>, cache: batch::PromptCache) -> mpsc::Sender<StreamingRequest> {
    let stream_max_batch: usize = std::env::var("STREAM_MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let stream_wait_ms: u64 = std::env::var("STREAM_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    let stream_poll_ms: u64 = std::env::var("STREAM_POLL_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let stream_chunk_frames: usize = std::env::var("STREAM_CHUNK_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(6);

    let (tx, rx) = mpsc::channel::<StreamingRequest>(16);
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));

    std::thread::spawn(move || {
        info!("Streaming worker ready (batched)");

        loop {
            // Block on first request
            let first = {
                let mut guard = rx.lock().unwrap_or_else(|e| { tracing::error!("streaming mutex poisoned, recovering"); e.into_inner() });
                match guard.blocking_recv() {
                    Some(r) => r,
                    None => break,
                }
            };

            // Collect more requests within window for batching
            let mut batch = vec![first];
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(stream_wait_ms);
            loop {
                if batch.len() >= stream_max_batch { break; }
                let mut guard = rx.lock().unwrap_or_else(|e| { tracing::error!("streaming mutex poisoned, recovering"); e.into_inner() });
                match guard.try_recv() {
                    Ok(r) => { batch.push(r); }
                    Err(_) => {
                        drop(guard);
                        if std::time::Instant::now() >= deadline { break; }
                        std::thread::sleep(std::time::Duration::from_millis(stream_poll_ms));
                    }
                }
            }

            let n = batch.len();
            info!(batch_size = n, "Streaming batch");

            // Build voice clone prompts: use cached_prompt if available, else encode
            let mut prompts: Vec<Option<std::sync::Arc<qwen3_tts::VoiceClonePrompt>>> = Vec::with_capacity(n);
            let mut to_remove: Vec<usize> = Vec::new();
            for (idx, req) in batch.iter().enumerate() {
                if let Some(p) = &req.cached_prompt {
                    prompts.push(Some(p.clone()));
                } else if req.voice_clone.is_some() {
                    prompts.push(None); // placeholder, will be filled below
                } else {
                    prompts.push(None);
                }
            }
            // Build prompts for requests that need encoding
            let needs_encode: Vec<(usize, &VoiceCloneData)> = batch.iter().enumerate()
                .filter(|(_, r)| r.cached_prompt.is_none() && r.voice_clone.is_some())
                .map(|(i, r)| (i, r.voice_clone.as_ref().unwrap()))
                .collect();
            if !needs_encode.is_empty() {
                let vc_refs: Vec<Option<&VoiceCloneData>> = batch.iter()
                    .map(|r| if r.cached_prompt.is_none() { r.voice_clone.as_ref() } else { None })
                    .collect();
                let (encoded, failed) = build_voice_clone_prompts(&model, &vc_refs, &cache);
                for &idx in failed.iter().rev() {
                    to_remove.push(idx);
                }
                for (idx, p) in encoded.into_iter().enumerate() {
                    if prompts[idx].is_none() { prompts[idx] = p; }
                }
            }

            // Send error to failed voice clone requests
            for &idx in to_remove.iter().rev() {
                prompts.remove(idx);
                let req = batch.remove(idx);
                let _ = req.tx.blocking_send(Err("Voice clone failed".into()));
            }
            if batch.is_empty() { continue; }

            // Streaming with voice_prompts uses speaker embedding (x_vector style)
            // — no ref_text replay frames to skip. skip_samples = 0 for all.
            let skip_samples: Vec<usize> = vec![0; batch.len()];

            let n = batch.len();
            let requests: Vec<(String, qwen3_tts::Language, Option<qwen3_tts::SynthesisOptions>)> = batch.iter()
                .map(|r| (r.text.clone(), r.language,
                    Some(qwen3_tts::SynthesisOptions {
                        temperature: r.temperature,
                        ..Default::default()
                    })
                )).collect();
            let prompt_refs: Vec<Option<&qwen3_tts::VoiceClonePrompt>> =
                prompts.iter().map(|p| p.as_deref()).collect();
            let speakers: Vec<Speaker> = batch.iter().map(|r| r.speaker).collect();

            let (senders, receivers): (Vec<_>, Vec<_>) = (0..n)
                .map(|_| std::sync::mpsc::channel::<qwen3_tts::AudioBuffer>()).unzip();

            // Stop flags: forward threads signal generation loop to stop early
            let stop_flags: Vec<Arc<std::sync::atomic::AtomicBool>> = (0..n)
                .map(|_| Arc::new(std::sync::atomic::AtomicBool::new(false))).collect();

            // Forward decoded audio chunks: std::sync → tokio channels as PCM
            // ICL requests skip ref_audio duration to remove ref_text replay
            // WAV header sent with first real chunk (#37)
            let forwards: Vec<std::thread::JoinHandle<()>> = receivers.into_iter()
                .zip(batch.iter().map(|r| r.tx.clone()))
                .zip(skip_samples.iter())
                .zip(stop_flags.iter().cloned())
                .map(|(((rx, tx), &skip), stop_flag)| {
                    std::thread::spawn(move || {
                        let mut remaining_skip = skip;
                        let mut header_sent = false;
                        while let Ok(audio) = rx.recv() {
                            let samples = &audio.samples;
                            // Skip ref_audio portion for ICL
                            if remaining_skip > 0 {
                                if remaining_skip >= samples.len() {
                                    remaining_skip -= samples.len();
                                    continue;
                                }
                                let trimmed = &samples[remaining_skip..];
                                remaining_skip = 0;
                                if !header_sent {
                                    if tx.blocking_send(Ok(wav_header(24000, 0xFFFFFFFF))).is_err() { stop_flag.store(true, std::sync::atomic::Ordering::Relaxed); break; }
                                    header_sent = true;
                                }
                                if tx.blocking_send(Ok(samples_to_pcm16(trimmed))).is_err() { stop_flag.store(true, std::sync::atomic::Ordering::Relaxed); break; }
                                continue;
                            }
                            if !header_sent {
                                if tx.blocking_send(Ok(wav_header(24000, 0xFFFFFFFF))).is_err() { stop_flag.store(true, std::sync::atomic::Ordering::Relaxed); break; }
                                header_sent = true;
                            }
                            if tx.blocking_send(Ok(samples_to_pcm16(samples))).is_err() { stop_flag.store(true, std::sync::atomic::Ordering::Relaxed); break; }
                        }
                    })
                }).collect();

            // Run batched streaming (decodes + sends every 10 frames ~800ms)
            let stop_refs: Vec<&std::sync::atomic::AtomicBool> = stop_flags.iter().map(|f| f.as_ref()).collect();
            if let Err(e) = model.synthesize_batch_streaming(&requests, &senders, stream_chunk_frames, &prompt_refs, &speakers, &stop_refs) {
                for req in &batch {
                    let _ = req.tx.blocking_send(Err(format!("{e}")));
                }
            }

            drop(senders);
            for f in forwards { let _ = f.join(); }
        }
    });

    tx
}

/// Max ref_audio decoded size in bytes (default 10 MB)
fn max_ref_audio_bytes() -> usize {
    std::env::var("MAX_REF_AUDIO_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(10 * 1024 * 1024)
}

fn decode_ref_audio(req: &SpeechRequest) -> Result<Option<VoiceCloneData>, (StatusCode, String)> {
    let b64 = match req.ref_audio.as_ref() {
        Some(b) if !b.is_empty() => b,
        _ => return Ok(None),
    };
    // base64 encodes 3 bytes as 4 chars; estimate decoded size without allocating
    let estimated = b64.len() / 4 * 3;
    let limit = max_ref_audio_bytes();
    if estimated > limit {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, format!("ref_audio exceeds {limit} byte limit")));
    }
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid ref_audio base64: {e}")))?;
    if bytes.len() > limit {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, format!("ref_audio exceeds {limit} byte limit")));
    }
    let tmp = std::env::temp_dir().join(format!("ref_{:016x}{:016x}.wav",
        rand_u64(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64));
    std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp)
        .and_then(|mut f| { use std::io::Write; f.write_all(&bytes) })
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("failed to write ref_audio: {e}")))?;
    // Validate WAV is loadable before accepting (#34)
    if qwen3_tts::AudioBuffer::load(&tmp).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return Err((StatusCode::BAD_REQUEST, "ref_audio is not a valid WAV file".into()));
    }
    Ok(Some(VoiceCloneData { ref_audio_path: tmp, ref_text: req.ref_text.clone(), audio_hash: hash_bytes(&bytes) }))
}

use tokio_stream::StreamExt;


/// Optional Bearer token auth middleware. Skips /health.
async fn auth_middleware(
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    // Skip auth for health check
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }
    let token = match std::env::var("API_KEY").ok() {
        Some(t) if !t.is_empty() => t,
        _ => return next.run(request).await, // no auth configured
    };
    match headers.get("authorization").and_then(|v| v.to_str().ok()) {
        Some(v) if v.strip_prefix("Bearer ").unwrap_or("") == token => next.run(request).await,
        _ => (StatusCode::UNAUTHORIZED, Json(ErrorResponse { error: "invalid or missing Bearer token".into() })).into_response(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "models/0.6b-base".into());
    let max_batch: usize = std::env::var("MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let max_wait_ms: u64 = std::env::var("MAX_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let port: u16 = std::env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8090);

    info!(model_dir = %model_dir, max_batch, max_wait_ms, port, "Starting qwen3-tts-server");

    let device = qwen3_tts::auto_device()?;
    let model = if let Ok(aux) = std::env::var("AUX_GPU") {
        let idx: usize = aux.parse().map_err(|err| {
            anyhow::anyhow!("AUX_GPU must be a CUDA device index, got {aux:?}: {err}")
        })?;
        let aux_device = qwen3_tts::parse_device(&format!("cuda:{idx}"))?;
        info!(?device, ?aux_device, "Loading model with auxiliary GPU");
        Arc::new(qwen3_tts::Qwen3TTS::from_pretrained_with_aux(
            &model_dir,
            device,
            aux_device,
        )?)
    } else {
        info!(?device, "Loading shared model");
        Arc::new(qwen3_tts::Qwen3TTS::from_pretrained(&model_dir, device)?)
    };
    info!("Shared model loaded");

    // Warmup forces CUDA kernel compilation before the first real request.
    // CustomVoice has no speaker encoder, so it warms the preset-speaker path.
    {
        let t0 = std::time::Instant::now();
        let short = qwen3_tts::SynthesisOptions { max_length: 5, ..Default::default() };
        if is_custom_voice(&model) {
            if let Err(err) = model.synthesize_with_voice(
                "warmup",
                Speaker::Serena,
                Language::English,
                Some(short),
            ) {
                tracing::warn!(%err, "Warmup synthesis failed");
            }
        } else {
            let dummy_audio = qwen3_tts::AudioBuffer::new(vec![0.0f32; 24000], 24000);
            match model.create_voice_clone_prompt(&dummy_audio, Some("warmup")) {
                Ok(prompt) => {
                    if let Err(err) = model.synthesize_voice_clone(
                        "warmup", &prompt, Language::English, Some(short),
                    ) {
                        tracing::warn!(%err, "Warmup synthesis failed");
                    }
                }
                Err(err) => tracing::warn!(%err, "Warmup voice-clone prompt failed"),
            }
        }
        info!(elapsed_ms = t0.elapsed().as_millis(), "Warmup complete");
    }

    let prompt_cache: batch::PromptCache = Arc::new(std::sync::Mutex::new(HashMap::new()));
    let tx = BatchEngine::start(model.clone(), BatchEngineConfig { max_batch_size: max_batch, max_wait_ms }, prompt_cache.clone());
    let stream_tx = start_streaming_worker(model.clone(), prompt_cache.clone());
    let max_inflight = max_batch * 2;

    let state = Arc::new(AppState {
        tx, stream_tx, semaphore: Arc::new(Semaphore::new(max_inflight)), max_inflight, max_batch,
        metrics: Arc::new(Metrics::new()),
        prompt_cache,
        model: model.clone(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/audio/speech", post(synthesize))
        .route("/v1/audio/voices", get(list_voices))
        .route("/v1/embeddings/preload", post(preload_embedding))
        .layer(middleware::from_fn(auth_middleware))
        .layer(cors)
        .with_state(state);

    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0".into());
    let listener = tokio::net::TcpListener::bind(format!("{bind_addr}:{port}")).await?;
    info!("Listening on {bind_addr}:{port}");
    let shutdown = async {
        tokio::signal::ctrl_c().await.ok();
        info!("Shutdown signal received, draining...");
    };
    axum::serve(listener, app).with_graceful_shutdown(shutdown).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_language() {
        assert!(matches!(parse_language("spanish"), Ok(Language::Spanish)));
        assert!(matches!(parse_language("es"), Ok(Language::Spanish)));
        assert!(matches!(parse_language("english"), Ok(Language::English)));
        assert!(matches!(parse_language("en"), Ok(Language::English)));
        assert!(matches!(parse_language("french"), Ok(Language::French)));
        assert!(parse_language("unknown").is_err());
    }

    #[test]
    fn test_wav_header_structure() {
        let h = wav_header(24000, 1000);
        assert_eq!(h.len(), 44);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(&h[36..40], b"data");
    }

    #[test]
    fn test_samples_to_pcm16() {
        let samples = vec![0.0f32, 1.0, -1.0, 0.5];
        let pcm = samples_to_pcm16(&samples);
        assert_eq!(pcm.len(), 8); // 4 samples * 2 bytes
        // 0.0 -> 0
        assert_eq!(i16::from_le_bytes([pcm[0], pcm[1]]), 0);
        // 1.0 -> 32767
        assert_eq!(i16::from_le_bytes([pcm[2], pcm[3]]), 32767);
        // -1.0 -> -32767
        assert_eq!(i16::from_le_bytes([pcm[4], pcm[5]]), -32767);
    }

    #[test]
    fn test_audio_to_wav_bytes() {
        let samples = vec![0.0f32; 100];
        let wav = audio_to_wav_bytes(&samples, 24000).unwrap();
        assert!(wav.len() > 44); // header + data
        assert_eq!(&wav[0..4], b"RIFF");
    }

    #[test]
    fn test_decode_ref_audio_valid_base64() {
        // Generate a real minimal WAV for validation
        let wav_bytes = audio_to_wav_bytes(&vec![0.0f32; 2400], 24000).unwrap(); // 100ms silence
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
        let req = SpeechRequest {
            text: "test".into(),
            language: "spanish".into(),
            ref_audio: Some(b64),
            ref_text: Some("ref".into()),
            temperature: None,
            stream: None,
            voice_id: None,
            speaker: None,
            sample_rate: None,
        };
        let result = decode_ref_audio(&req);
        assert!(result.is_ok());
        let vc = result.unwrap().unwrap();
        assert!(vc.ref_audio_path.exists());
        assert_eq!(vc.ref_text, Some("ref".into()));
        // Drop should clean up
        let _path = vc.ref_audio_path.clone();
        drop(vc);
        // VoiceCloneData is in batch module, Drop cleans up
    }

    #[test]
    fn test_decode_ref_audio_invalid_base64() {
        let req = SpeechRequest {
            text: "test".into(),
            language: "spanish".into(),
            ref_audio: Some("not-valid-base64!!!".into()),
            ref_text: None,
            temperature: None,
            stream: None,
            voice_id: None,
            speaker: None,
            sample_rate: None,
        };
        assert!(decode_ref_audio(&req).is_err());
    }

    #[test]
    fn test_decode_ref_audio_none() {
        let req = SpeechRequest {
            text: "test".into(),
            language: "spanish".into(),
            ref_audio: None,
            ref_text: None,
            temperature: None,
            stream: None,
            voice_id: None,
            speaker: None,
            sample_rate: None,
        };
        assert!(decode_ref_audio(&req).unwrap().is_none());
    }

    #[test]
    fn test_voice_clone_data_drop_cleanup() {
        let tmp = std::env::temp_dir().join("test_vc_drop.wav");
        std::fs::write(&tmp, b"test").unwrap();
        assert!(tmp.exists());
        let vc = batch::VoiceCloneData {
            ref_audio_path: tmp.clone(),
            ref_text: None,
            audio_hash: 0,
        };
        drop(vc);
        assert!(!tmp.exists(), "Drop should have deleted temp file");
    }

    #[test]
    fn test_voice_clone_data_drop_missing_file() {
        // Should not panic if file doesn't exist
        let vc = batch::VoiceCloneData {
            ref_audio_path: "/tmp/nonexistent_vc_test.wav".into(),
            ref_text: None,
            audio_hash: 0,
        };
        drop(vc); // should not panic
    }

    #[test]
    fn test_preset_speakers_roundtrip_and_are_not_language_bound() {
        assert_eq!(Speaker::all().len(), 9);
        assert_eq!(Language::all().len(), 10);
        for speaker in Speaker::all() {
            let parsed: Speaker = speaker.as_str().parse().unwrap();
            assert_eq!(parsed, *speaker);
            assert!(speaker_list().contains(speaker.as_str()));
        }
        for language in Language::all() {
            let parsed: Language = language.as_str().parse().unwrap();
            assert_eq!(parsed, *language);
        }
        assert!("not-a-voice".parse::<Speaker>().is_err());
    }

    #[test]
    fn test_split_sentences_keeps_clauses() {
        let parts = split_sentences("Hello, world. How are you? Fine!");
        assert_eq!(parts, vec!["Hello, world.", "How are you?", "Fine!"]);
    }

    #[test]
    fn test_split_sentences_ignores_colon_and_comma() {
        let parts = split_sentences("Wait: one, two, three things happen here");
        assert_eq!(parts, vec!["Wait: one, two, three things happen here"]);
    }

    #[test]
    fn test_metrics_struct() {
        let m = Metrics::new();
        assert_eq!(m.requests_total.load(std::sync::atomic::Ordering::Relaxed), 0);
        m.requests_total.fetch_add(5, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(m.requests_total.load(std::sync::atomic::Ordering::Relaxed), 5);
    }
}
