//! Process-local memory counters. Only enabled in a separate profiling replay.
//! The same implementation is loaded by Python through the small C ABI below.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString, c_char},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

const INTERVAL_MS: u64 = 10;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetric {
    pub status: String,
    pub method: String,
    pub scope: String,
    pub peak_kind: String,
    pub peak_bytes: Option<u64>,
    pub sample_count: usize,
    pub reason: Option<String>,
}
impl MemoryMetric {
    fn new(method: &str, scope: &str, peak_kind: &str) -> Self {
        Self {
            status: "available".into(),
            method: method.into(),
            scope: scope.into(),
            peak_kind: peak_kind.into(),
            peak_bytes: None,
            sample_count: 0,
            reason: None,
        }
    }
    fn unavailable(status: &str, reason: &str) -> Self {
        Self {
            status: status.into(),
            reason: Some(reason.into()),
            ..Self::new("", "", "")
        }
    }
    fn observe(&mut self, reading: Result<(u64, u64)>) -> Option<u64> {
        if self.status != "available" {
            return None;
        }
        match reading {
            Ok((current, peak)) => {
                self.peak_bytes = Some(self.peak_bytes.unwrap_or(0).max(peak).max(current));
                self.sample_count += 1;
                Some(current)
            }
            Err(error) => {
                // One failed read makes the metric unavailable, never a partial peak.
                *self = Self::unavailable("unavailable", &error.to_string());
                None
            }
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub phase: String,
    pub elapsed_seconds: f64,
    pub bytes: BTreeMap<String, Option<u64>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMeasurement {
    pub status: String,
    pub completed_steps: usize,
    pub sampling_interval_ms: u64,
    pub unified_memory: Option<bool>,
    pub metrics: BTreeMap<String, MemoryMetric>,
    pub snapshots: Vec<MemorySnapshot>,
}
impl MemoryMeasurement {
    pub fn validate(&self, job: &crate::Job) -> Result<()> {
        ensure!(
            self.status == "complete" && self.completed_steps == job.steps,
            "incomplete memory replay"
        );
        ensure!(
            self.sampling_interval_ms > 0,
            "missing memory sampling cadence"
        );
        ensure!(
            self.snapshots.iter().map(|s| s.phase.as_str()).eq([
                "baseline",
                "setup",
                "warmup",
                "reset",
                "rollout_end"
            ]),
            "incomplete memory phases"
        );
        for key in ["ram", "gpu"] {
            ensure!(self.metrics.contains_key(key), "missing {key} metric");
        }
        for (key, metric) in &self.metrics {
            if metric.status == "available" {
                ensure!(
                    metric.peak_bytes.is_some_and(|peak| peak > 0)
                        && metric.sample_count >= self.snapshots.len()
                        && !metric.method.is_empty()
                        && !metric.scope.is_empty(),
                    "invalid memory metric {key}"
                );
                ensure!(
                    ["sampled", "os_high_water"].contains(&metric.peak_kind.as_str()),
                    "invalid memory peak kind"
                );
                for snapshot in &self.snapshots {
                    ensure!(
                        snapshot.elapsed_seconds.is_finite() && snapshot.elapsed_seconds >= 0.,
                        "invalid memory snapshot time"
                    );
                    ensure!(
                        snapshot
                            .bytes
                            .get(key)
                            .copied()
                            .flatten()
                            .is_some_and(|bytes| Some(bytes) <= metric.peak_bytes),
                        "memory snapshot exceeds peak or is missing"
                    );
                }
                ensure!(
                    key != "gpu" || job.backend != "cpu",
                    "CPU backend cannot supply GPU memory"
                );
            } else {
                ensure!(
                    ["unavailable", "not_applicable"].contains(&metric.status.as_str())
                        && metric.peak_bytes.is_none()
                        && metric.sample_count == 0
                        && metric.reason.as_ref().is_some_and(|r| !r.is_empty()),
                    "invalid unavailable memory metric"
                );
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn ram() -> Result<(u64, u64, u64, u64)> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
    let mut rss = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: the requested flavors match both output buffers; self PID is valid.
    unsafe {
        ensure!(
            libc::proc_pid_rusage(
                std::process::id() as i32,
                libc::RUSAGE_INFO_V4,
                usage.as_mut_ptr().cast()
            ) == 0,
            "proc_pid_rusage failed"
        );
        ensure!(
            libc::getrusage(libc::RUSAGE_SELF, rss.as_mut_ptr()) == 0,
            "getrusage failed"
        );
        let (usage, rss) = (usage.assume_init(), rss.assume_init());
        Ok((
            usage.ri_resident_size,
            rss.ru_maxrss as u64,
            usage.ri_phys_footprint,
            usage.ri_lifetime_max_phys_footprint,
        ))
    }
}
#[cfg(target_os = "linux")]
fn ram() -> Result<(u64, u64, u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let kb = |key: &str| -> Result<u64> {
        let line = status
            .lines()
            .find(|line| line.starts_with(key))
            .ok_or_else(|| anyhow::anyhow!("missing {key}"))?;
        Ok(line
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("invalid {key}"))?
            .parse::<u64>()?
            * 1024)
    };
    Ok((kb("VmRSS:")?, kb("VmHWM:")?, 0, 0))
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn ram() -> Result<(u64, u64, u64, u64)> {
    anyhow::bail!("RAM counters supported on macOS and Linux")
}

#[cfg(target_os = "macos")]
mod metal {
    use std::ffi::{c_char, c_void};
    #[link(name = "Metal", kind = "framework")]
    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }
    #[link(name = "objc")]
    unsafe extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
    }
    pub struct Device(usize);
    impl Device {
        pub fn new() -> anyhow::Result<Self> {
            // SAFETY: Create returns a retained device; Device drops that reference.
            let device = unsafe { MTLCreateSystemDefaultDevice() };
            anyhow::ensure!(!device.is_null(), "no Metal device");
            Ok(Self(device as usize))
        }
        fn query(&self, name: &std::ffi::CStr) -> u64 {
            // SAFETY: selectors return NSUInteger/BOOL; the retained MTLDevice is
            // valid on sampling threads. Calls are serialized by the collector lock.
            unsafe {
                let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64 =
                    std::mem::transmute(objc_msgSend as *const ());
                send(self.0 as *mut c_void, sel_registerName(name.as_ptr()))
            }
        }
        pub fn bytes(&self) -> u64 {
            self.query(c"currentAllocatedSize")
        }
        pub fn unified(&self) -> bool {
            // BOOL has a distinct return ABI from NSUInteger.
            unsafe {
                let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool =
                    std::mem::transmute(objc_msgSend as *const ());
                send(
                    self.0 as *mut c_void,
                    sel_registerName(c"hasUnifiedMemory".as_ptr()),
                )
            }
        }
    }
    impl Drop for Device {
        fn drop(&mut self) {
            // SAFETY: typed void-return release, exactly one owned reference.
            unsafe {
                let send: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                    std::mem::transmute(objc_msgSend as *const ());
                send(self.0 as *mut c_void, sel_registerName(c"release".as_ptr()));
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod nvml {
    use anyhow::{Result, ensure};
    use std::ffi::c_void;
    type Count = unsafe extern "C" fn(*mut u32) -> u32;
    type Handle = unsafe extern "C" fn(u32, *mut *mut c_void) -> u32;
    type Processes = unsafe extern "C" fn(*mut c_void, *mut u32, *mut Process) -> u32;
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Process {
        pid: u32,
        used_gpu_memory: u64,
        gpu_instance_id: u32,
        compute_instance_id: u32,
    }
    pub struct Nvml {
        _library: libloading::Library,
        count: Count,
        handle: Handle,
        processes: Processes,
        shutdown: unsafe extern "C" fn() -> u32,
        initialized: bool,
    }
    impl Nvml {
        pub fn new() -> Result<Self> {
            // SAFETY: signatures and Process layout are the NVML v2 ABI. Retain the
            // library until all function pointers have been dropped.
            unsafe {
                let library = libloading::Library::new("libnvidia-ml.so.1")?;
                let init = *library.get::<unsafe extern "C" fn() -> u32>(b"nvmlInit_v2\0")?;
                let mut result = Self {
                    initialized: false,
                    count: *library.get(b"nvmlDeviceGetCount_v2\0")?,
                    handle: *library.get(b"nvmlDeviceGetHandleByIndex_v2\0")?,
                    processes: *library.get(b"nvmlDeviceGetComputeRunningProcesses_v2\0")?,
                    shutdown: *library.get(b"nvmlShutdown\0")?,
                    _library: library,
                };
                ensure!(init() == 0, "NVML initialization failed");
                result.initialized = true;
                Ok(result)
            }
        }
        pub fn bytes(&self) -> Result<u64> {
            let mut total = 0;
            // SAFETY: all pointers reference appropriately sized live ABI buffers.
            unsafe {
                let mut devices = 0;
                ensure!(
                    (self.count)(&mut devices) == 0 && devices > 0,
                    "NVML device enumeration failed"
                );
                for index in 0..devices {
                    let mut device = std::ptr::null_mut();
                    ensure!(
                        (self.handle)(index, &mut device) == 0,
                        "NVML device inaccessible"
                    );
                    let mut count = 0;
                    let status = (self.processes)(device, &mut count, std::ptr::null_mut());
                    ensure!(
                        status == 0 || status == 7,
                        "NVML process query failed ({status})"
                    ); // INSUFFICIENT_SIZE = 7
                    let mut entries = vec![Process::default(); count as usize + 32];
                    count = entries.len() as u32;
                    let status = (self.processes)(device, &mut count, entries.as_mut_ptr());
                    ensure!(status == 0, "NVML process enumeration failed ({status})");
                    for process in &entries[..count as usize] {
                        if process.pid == std::process::id() {
                            ensure!(
                                process.used_gpu_memory != u64::MAX,
                                "NVML per-process memory unavailable"
                            );
                            total += process.used_gpu_memory;
                        }
                    }
                }
            }
            Ok(total)
        }
    }
    impl Drop for Nvml {
        fn drop(&mut self) {
            if self.initialized {
                // SAFETY: balances this instance's successful NVML initialization.
                unsafe {
                    (self.shutdown)();
                }
            }
        }
    }
}

enum GpuProbe {
    #[cfg(target_os = "macos")]
    Metal(metal::Device),
    #[cfg(target_os = "linux")]
    Nvml(nvml::Nvml),
    Unavailable,
}
impl GpuProbe {
    fn new(backend: &str) -> (Self, MemoryMetric, Option<bool>) {
        if backend == "cpu" {
            return (
                Self::Unavailable,
                MemoryMetric::unavailable("not_applicable", "CPU backend"),
                None,
            );
        }
        #[cfg(target_os = "macos")]
        if backend == "metal" {
            return match metal::Device::new() {
                Ok(device) => {
                    let unified = device.unified();
                    (
                        Self::Metal(device),
                        MemoryMetric::new(
                            "metal_currentAllocatedSize",
                            "process_default_device_resources",
                            "sampled",
                        ),
                        Some(unified),
                    )
                }
                Err(error) => (
                    Self::Unavailable,
                    MemoryMetric::unavailable("unavailable", &error.to_string()),
                    None,
                ),
            };
        }
        #[cfg(target_os = "linux")]
        if backend == "cuda" {
            return match nvml::Nvml::new() {
                Ok(nvml) => (
                    Self::Nvml(nvml),
                    MemoryMetric::new(
                        "nvml_compute_process_usedGpuMemory",
                        "process_all_cuda_devices",
                        "sampled",
                    ),
                    None,
                ),
                Err(error) => (
                    Self::Unavailable,
                    MemoryMetric::unavailable("unavailable", &error.to_string()),
                    None,
                ),
            };
        }
        (
            Self::Unavailable,
            MemoryMetric::unavailable(
                "unavailable",
                "No verified device allocation counter for this backend/platform",
            ),
            None,
        )
    }
    fn bytes(&self) -> Result<(u64, u64)> {
        let bytes = match self {
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.bytes(),
            #[cfg(target_os = "linux")]
            Self::Nvml(nvml) => nvml.bytes()?,
            Self::Unavailable => anyhow::bail!("no GPU counter"),
        };
        Ok((bytes, bytes))
    }
}
struct Collector {
    result: MemoryMeasurement,
    gpu: GpuProbe,
    start: Instant,
}
impl Collector {
    fn sample(&mut self, phase: Option<&str>) {
        let mut bytes = BTreeMap::new();
        let usage = ram();
        let reading = usage
            .as_ref()
            .map(|r| (r.0, r.1))
            .map_err(|e| anyhow::anyhow!(e.to_string()));
        bytes.insert(
            "ram".into(),
            self.result.metrics.get_mut("ram").unwrap().observe(reading),
        );
        if let Some(metric) = self.result.metrics.get_mut("footprint") {
            bytes.insert(
                "footprint".into(),
                metric.observe(usage.map(|r| (r.2, r.3))),
            );
        }
        bytes.insert(
            "gpu".into(),
            self.result
                .metrics
                .get_mut("gpu")
                .unwrap()
                .observe(self.gpu.bytes()),
        );
        if let Some(phase) = phase {
            self.result.snapshots.push(MemorySnapshot {
                phase: phase.into(),
                elapsed_seconds: self.start.elapsed().as_secs_f64(),
                bytes,
            });
        }
    }
}
pub struct MemorySampler {
    collector: Arc<Mutex<Collector>>,
    stop: mpsc::Sender<()>,
    worker: Option<thread::JoinHandle<()>>,
}
impl MemorySampler {
    pub fn new(backend: &str) -> Self {
        let (gpu, gpu_metric, unified_memory) = GpuProbe::new(backend);
        let mut metrics = BTreeMap::from([
            (
                "ram".into(),
                MemoryMetric::new(
                    if cfg!(target_os = "macos") {
                        "proc_pid_rusage_getrusage"
                    } else {
                        "proc_status_VmRSS_VmHWM"
                    },
                    "process_resident_set",
                    "os_high_water",
                ),
            ),
            ("gpu".into(), gpu_metric),
        ]);
        if cfg!(target_os = "macos") {
            metrics.insert(
                "footprint".into(),
                MemoryMetric::new(
                    "proc_pid_rusage_v4",
                    "process_physical_footprint",
                    "os_high_water",
                ),
            );
        }
        let collector = Arc::new(Mutex::new(Collector {
            result: MemoryMeasurement {
                status: "complete".into(),
                completed_steps: 0,
                sampling_interval_ms: INTERVAL_MS,
                unified_memory,
                metrics,
                snapshots: vec![],
            },
            gpu,
            start: Instant::now(),
        }));
        collector.lock().unwrap().sample(Some("baseline"));
        let (stop, receiver) = mpsc::channel();
        let shared = collector.clone();
        let worker = thread::spawn(move || {
            while receiver.recv_timeout(Duration::from_millis(INTERVAL_MS))
                == Err(mpsc::RecvTimeoutError::Timeout)
            {
                shared.lock().unwrap().sample(None);
            }
        });
        Self {
            collector,
            stop,
            worker: Some(worker),
        }
    }
    pub fn sample(&self, phase: &str) {
        self.collector.lock().unwrap().sample(Some(phase));
    }
    pub fn finish(mut self, completed_steps: usize) -> MemoryMeasurement {
        self.stop_worker();
        let mut collector = self.collector.lock().unwrap();
        collector.sample(Some("rollout_end"));
        collector.result.completed_steps = completed_steps;
        let gpu = collector.result.metrics.get_mut("gpu").unwrap();
        if gpu.status == "available" && gpu.peak_bytes == Some(0) {
            *gpu = MemoryMetric::unavailable(
                "unavailable",
                "GPU counter reported zero throughout replay; no attributable device allocations",
            );
        }
        collector.result.clone()
    }
    fn stop_worker(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            worker.join().expect("memory sampler panicked");
        }
    }
}
impl Drop for MemorySampler {
    fn drop(&mut self) {
        self.stop_worker();
    }
}

// Private Python bridge: pointers are owned handles, not addresses exposed in results.
/// # Safety
/// `backend` must be a valid NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn physmark_memory_new(backend: *const c_char) -> *mut MemorySampler {
    Box::into_raw(Box::new(MemorySampler::new(
        unsafe { CStr::from_ptr(backend) }
            .to_str()
            .unwrap_or("unknown"),
    )))
}
/// # Safety
/// `sampler` must be a live bridge handle; `phase` a valid NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn physmark_memory_sample(sampler: *mut MemorySampler, phase: *const c_char) {
    unsafe { &*sampler }.sample(
        unsafe { CStr::from_ptr(phase) }
            .to_str()
            .unwrap_or("unknown"),
    );
}
/// # Safety
/// Consumes a live bridge handle exactly once. Free the returned JSON with memory_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn physmark_memory_finish(
    sampler: *mut MemorySampler,
    steps: usize,
) -> *mut c_char {
    let sampler = unsafe { Box::from_raw(sampler) };
    CString::new(serde_json::to_string(&sampler.finish(steps)).unwrap())
        .unwrap()
        .into_raw()
}
/// # Safety
/// `json` must be an unfreed pointer returned by memory_finish.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn physmark_memory_free(json: *mut c_char) {
    drop(unsafe { CString::from_raw(json) });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_incomplete_or_fabricated_memory_metrics() {
        let job = crate::Job {
            schema_version: 1,
            scene: crate::scene("stacks", 1, 42, 0.01).unwrap(),
            engine: "rapier".into(),
            backend: "cpu".into(),
            mode: "memory".into(),
            worlds: 1,
            steps: 1,
            warmup: 1,
            capture_stride: 1,
            threads: 1,
        };
        job.validate().unwrap();
        let sampler = MemorySampler::new("cpu");
        for phase in ["setup", "warmup", "reset"] {
            sampler.sample(phase);
        }
        let valid = sampler.finish(1);
        valid.validate(&job).unwrap();
        let mut invalid = valid.clone();
        invalid.completed_steps = 0;
        assert!(invalid.validate(&job).is_err());
        let mut invalid = valid.clone();
        invalid.snapshots.pop();
        assert!(invalid.validate(&job).is_err());
        let mut invalid = valid.clone();
        invalid.metrics.get_mut("ram").unwrap().peak_bytes = Some(0);
        assert!(invalid.validate(&job).is_err());
        let mut invalid = valid.clone();
        invalid.metrics.get_mut("gpu").unwrap().peak_bytes = Some(123);
        assert!(invalid.validate(&job).is_err());
        let mut invalid = valid.clone();
        invalid.snapshots[0]
            .bytes
            .insert("ram".into(), Some(u64::MAX));
        assert!(invalid.validate(&job).is_err());
        let mut invalid = valid;
        invalid.metrics.get_mut("ram").unwrap().sample_count = 1;
        assert!(invalid.validate(&job).is_err());
    }
    #[test]
    fn actual_ram_high_water_captures_touched_allocation() {
        let sampler = MemorySampler::new("cpu");
        let before = ram().unwrap().0;
        let mut allocation = vec![0u8; 32 * 1024 * 1024];
        allocation.fill(77);
        std::hint::black_box(&allocation);
        sampler.sample("setup");
        sampler.sample("warmup");
        sampler.sample("reset");
        let result = sampler.finish(1);
        assert!(result.metrics["ram"].peak_bytes.unwrap() >= before + 16 * 1024 * 1024);
        assert_eq!(result.metrics["gpu"].status, "not_applicable");
        assert!(result.metrics["gpu"].peak_bytes.is_none());
    }
}
