use super::*;
use slatedb::object_store::memory::InMemory;

/// Projection-aware hydration must not change any
/// result: every shape below reads a different subset of properties, or the
/// whole record, of vertices that carry many more properties than are read.
#[tokio::test]
async fn projected_hydration_matches_legacy_results() {
    let shard = GraphShard::open_standalone_writer("graph/projected-hydration", Arc::new(InMemory::new()))
        .await
        .unwrap();
    shard
        .set_vertex_metadata_batch(
            "cell-a",
            // Vertex 75 is left without a record so the dense range read has
            // a gap inside its span.
            (1..=80)
                .filter(|id| *id != 75)
                .map(|id| {
                    let mut metadata = VertexMetadata::default()
                        .with_label("Entity")
                        .with_property("id", VertexPropertyValue::Integer(id))
                        .with_property("name", VertexPropertyValue::String(format!("entity-{id:02}")))
                        .with_property("age", VertexPropertyValue::Integer(id % 7))
                        .with_property("city", VertexPropertyValue::String(format!("city-{}", id % 3)));
                    for slot in 0..50 {
                        metadata = metadata.with_property(format!("filler_{slot}"), VertexPropertyValue::Integer(id + slot));
                    }
                    (id, metadata)
                })
                .collect::<Vec<_>>(),
        )
        .await
        .unwrap();
    for source in 1..=10u64 {
        for offset in 1..=2u64 {
            let target = 10 + (source + offset * 7) % 20 + 1;
            shard
                .write_edge(EdgeMutation {
                    cell_id: "cell-a".into(),
                    edge_type: "RELATES".into(),
                    src: source,
                    dst: target,
                    idempotency_key: format!("edge-{source}-{offset}"),
                })
                .await
                .unwrap();
        }
    }

    // A hub over 40 consecutive targets: its hydration batch is dense enough
    // for the single range-scan read path.
    for target in 41..=80u64 {
        shard
            .write_edge(EdgeMutation {
                cell_id: "cell-a".into(),
                edge_type: "RELATES".into(),
                src: 1,
                dst: target,
                idempotency_key: format!("hub-{target}"),
            })
            .await
            .unwrap();
    }

    let mut compared = 0_usize;
    let mut skipped = Vec::new();
    for query in [
        "MATCH (a:Entity {id: 1})-[:RELATES]->(b) RETURN b.id AS id, b.name AS name ORDER BY id",
        "MATCH (a:Entity {id: 1})-[:RELATES*1..1]->(b) RETURN b.id AS id ORDER BY id",
        "MATCH (a:Entity {id: 1})-[:RELATES*1..2]->(b) RETURN b.id AS id ORDER BY id",
        "MATCH (a:Entity {id: 1})-[:RELATES*1..1]->(b) RETURN a.name AS src, b.filler_7 AS filler ORDER BY filler",
        "MATCH (a:Entity {id: 1})-[:RELATES]->(b) RETURN b.id ORDER BY b.id",
        "MATCH (a:Entity)-[:RELATES]->(b) RETURN a.name AS src, b.name AS dst ORDER BY src, dst",
        "MATCH (a:Entity)-[:RELATES]->(b) WHERE b.age > 3 RETURN a.id AS src, b.city AS city ORDER BY src, city",
        "MATCH (a:Entity {id: 2})-[:RELATES*1..2]->(b) RETURN DISTINCT b.name AS name ORDER BY name",
        "MATCH (a:Entity)-[:RELATES]->(b) RETURN b.city AS city, count(b) AS n ORDER BY city",
        "MATCH (a:Entity)-[:RELATES]->(b) RETURN b.missing AS missing, b.id AS id ORDER BY id",
        "MATCH (a:Entity)-[:RELATES]->(b) RETURN a.id AS id, collect(b.name) AS names ORDER BY id",
        "MATCH (a:Entity)-[:RELATES]->(b) RETURN b.id AS id, properties(b).filler_3 AS filler ORDER BY id",
        "MATCH (a:Entity)-[:RELATES]->(b) RETURN b.id AS id, size(keys(b)) AS width ORDER BY id",
        "MATCH (a:Entity)-[:RELATES]->(b) RETURN b.id AS id, b.filler_49 AS filler ORDER BY id",
        "MATCH (a:Entity)-[:RELATES]->(b) RETURN CASE WHEN b.age > 2 THEN b.name ELSE a.city END AS v ORDER BY v",
    ] {
        // A shape either engine refuses as unsupported cannot be compared;
        // any other failure is a test failure.
        let run = |mode: CypherEngineMode| {
            shard.execute_cypher_rows(QueryContext::new("cell-a", "projected-hydration").with_cypher_engine(mode), query)
        };
        let (legacy, experimental) = match (run(CypherEngineMode::Legacy).await, run(CypherEngineMode::Experimental).await) {
            (Ok(legacy), Ok(experimental)) => (legacy, experimental),
            (Err(GraphError::UnsupportedQuery { .. }), _) | (_, Err(GraphError::UnsupportedQuery { .. })) => {
                skipped.push(query);
                continue;
            }
            (legacy, experimental) => panic!("{query}: legacy {legacy:?}, experimental {experimental:?}"),
        };
        assert!(!legacy.rows.is_empty(), "fixture produced no rows for {query}");
        assert_eq!(experimental.columns, legacy.columns, "{query}");
        assert_eq!(experimental.rows, legacy.rows, "{query}");
        compared += 1;
    }
    assert!(compared >= 10, "only {compared} shapes compared; skipped {skipped:?}");
    // The Legacy engine cannot bind a variable-length relationship, so the
    // last-hop path that binds one is checked against the same walk without
    // the binding: the same targets, in the same order, each with a hop count
    // the pattern allows.
    let experimental = |query: &'static str| {
        shard.execute_cypher_rows(
            QueryContext::new("cell-a", "projected-hydration-bound").with_cypher_engine(CypherEngineMode::Experimental),
            query,
        )
    };
    let bound = experimental("MATCH (a:Entity {id: 1})-[r:RELATES*1..2]->(b) RETURN b.id AS id, size(r) AS hops ORDER BY id, hops")
        .await
        .unwrap();
    let unbound = experimental("MATCH (a:Entity {id: 1})-[:RELATES*1..2]->(b) RETURN b.id AS id ORDER BY id")
        .await
        .unwrap();
    assert!(!bound.rows.is_empty());
    assert_eq!(
        bound.rows.iter().map(|row| row.values[0].clone()).collect::<Vec<_>>(),
        unbound.rows.iter().map(|row| row.values[0].clone()).collect::<Vec<_>>(),
    );
    assert!(
        bound.rows.iter().all(|row| matches!(
            &row.values[1],
            QueryValue::Property(VertexPropertyValue::Integer(1 | 2) | VertexPropertyValue::SignedInteger(1 | 2))
                | QueryValue::Count(1 | 2)
        )),
        "{:?}",
        bound.rows
    );
    shard.close().await.unwrap();
}
