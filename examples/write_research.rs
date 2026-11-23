//! Write-path benchmark: reads per edge write, and throughput of concurrent
//! writers sharing WAL uploads.
//!
//! 1. Reads per write: sequential single-edge writes; storage get/scan
//!    requests per write from `graph_storage_metrics()`.
//! 2. Concurrency: K writers committing distinct edges into one cell; total
//!    throughput, per-write latency, and how many commits shared one WAL
//!    upload (`write_commits / wal_buffer_flushes`).
//! 3. Hub: the same, but every edge leaves one source vertex, so every write
//!    reads and rewrites the same degree counter.
//!
//! Environment:
//!   WR_OBJECT_ENV        object-store env file (S3); unset = local filesystem
//!   WR_ROOT              working directory (default: a temp dir)
//!   WR_WAL_MS            WAL flush interval in ms (default 10, graph-node's default)
//!   WR_WRITES            writes per measurement (default 400)
//!   WR_CONCURRENCY       comma list (default 1,4,16,32,64)
//!   WR_MAX_PENDING       GraphBackpressurePolicy::max_pending_durable_writes (default 16)

use std::sync::Arc;
use std::time::{Duration, Instant};

use hydradb::{
    local_object_store, object_store_from_env, EdgeMutation, GraphBackpressurePolicy, GraphDurabilityConfig, GraphOpenOptions, GraphShard,
};
use slatedb::object_store::ObjectStore;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const CELL: &str = "research-cell";
const EDGE_TYPE: &str = "FOLLOWS";

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn percentile(sorted: &[Duration], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[index].as_secs_f64() * 1e3
}

struct Counters {
    gets: u64,
    scans: u64,
    commits: u64,
    wal_flushes: u64,
}

fn counters(shard: &GraphShard) -> Counters {
    let storage = shard.graph_storage_metrics();
    let operational = shard.graph_operational_metrics();
    Counters {
        gets: storage.get_requests,
        scans: storage.scan_requests,
        commits: operational.write_commits,
        wal_flushes: storage.wal_buffer_flushes,
    }
}

/// Run `writes` single-edge writes split across `concurrency` tasks. Edge `i`
/// goes from `src_of(i)` to a fresh destination, so every write inserts.
async fn run(shard: &Arc<GraphShard>, label: &str, writes: u64, concurrency: u64, base: u64, hub: bool) -> Result<()> {
    let before = counters(shard);
    let started = Instant::now();
    let mut tasks = Vec::new();
    for worker in 0..concurrency {
        let shard = Arc::clone(shard);
        let label = label.to_string();
        tasks.push(tokio::spawn(async move {
            let mut latencies = Vec::new();
            let mut i = worker;
            while i < writes {
                let id = base + i;
                let src = if hub { base } else { id };
                let mutation = EdgeMutation {
                    cell_id: CELL.into(),
                    edge_type: EDGE_TYPE.into(),
                    src,
                    dst: 1_000_000_000 + id,
                    idempotency_key: format!("{label}-{id}"),
                };
                let t = Instant::now();
                shard.write_edge(mutation).await?;
                latencies.push(t.elapsed());
                i += concurrency;
            }
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(latencies)
        }));
    }
    let mut latencies = Vec::new();
    for task in tasks {
        latencies.extend(task.await??);
    }
    let elapsed = started.elapsed();
    let after = counters(shard);
    latencies.sort();
    let commits = after.commits - before.commits;
    let flushes = (after.wal_flushes - before.wal_flushes).max(1);
    println!(
        "{label},{concurrency},{writes},{:.1},{:.3},{:.3},{:.3},{:.2},{:.2},{commits},{flushes},{:.2}",
        writes as f64 / elapsed.as_secs_f64(),
        percentile(&latencies, 0.50),
        percentile(&latencies, 0.95),
        percentile(&latencies, 0.99),
        (after.gets - before.gets) as f64 / writes as f64,
        (after.scans - before.scans) as f64 / writes as f64,
        commits as f64 / flushes as f64,
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let wal_ms = env_u64("WR_WAL_MS", 10);
    let writes = env_u64("WR_WRITES", 400);
    let levels: Vec<u64> = std::env::var("WR_CONCURRENCY")
        .unwrap_or_else(|_| "1,4,16,32,64".into())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .collect();
    let root = std::env::var("WR_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("write-research-{}", std::process::id())));
    std::fs::create_dir_all(&root)?;
    let (store, backend): (Arc<dyn ObjectStore>, String) = match std::env::var("WR_OBJECT_ENV") {
        Ok(env) => (object_store_from_env(Some(env.clone()))?, format!("env:{env}")),
        Err(_) => {
            let objects = root.join("object-store");
            std::fs::create_dir_all(&objects)?;
            (local_object_store(&objects)?, format!("local:{}", objects.display()))
        }
    };
    let max_pending = env_u64(
        "WR_MAX_PENDING",
        GraphBackpressurePolicy::default().max_pending_durable_writes as u64,
    ) as usize;
    let options = GraphOpenOptions {
        durability: GraphDurabilityConfig::low_latency_durable(wal_ms),
        backpressure_policy: GraphBackpressurePolicy {
            max_pending_durable_writes: max_pending,
            ..GraphBackpressurePolicy::default()
        },
        ..GraphOpenOptions::default()
    };
    let path = format!("write-research-{}", std::process::id());
    let shard = Arc::new(GraphShard::open_standalone_writer_with_options(path, store, options).await?);
    eprintln!("write research: backend={backend} wal_ms={wal_ms} writes={writes} levels={levels:?} max_pending={max_pending}");
    println!("label,concurrency,writes,writes_per_s,p50_ms,p95_ms,p99_ms,gets_per_write,scans_per_write,commits,wal_flushes,commits_per_wal_flush");

    // Warm-up so the first measurement does not include writer start-up.
    run(&shard, "warmup", 50, 1, 9_000_000, false).await?;
    let mut base = 10_000_000;
    for &concurrency in &levels {
        run(&shard, "distinct", writes, concurrency, base, false).await?;
        base += 1_000_000;
    }
    for &concurrency in &levels {
        run(&shard, "hub", writes, concurrency, base, true).await?;
        base += 1_000_000;
    }
    shard.close().await?;
    Ok(())
}
