//! Which vertex properties a prepared plan can read.
//!
//! Storage hydrates a vertex by decoding its whole record, every property
//! included, even when the query returns only `v.id`. This pass walks the
//! finished physical plan and collects the name of every property any
//! expression reads (`x.p`), across every operator, subquery and union arm.
//! Storage may then decode only those properties; labels and the id are
//! always kept.
//!
//! The rule is conservative, in the spirit of `index_only`: the result is
//! `None` (hydrate everything) whenever the plan might observe a record as a
//! whole, or uses an operator this pass does not inspect:
//!
//!   * any expression that reads a binding as a value (`RETURN v`,
//!     `properties(v)`, `keys(v)`, `collect(v)`, a map projection, a path),
//!     except as the argument of `count`, which only tests for null;
//!   * procedure calls and path-binding or path-selection operators, whose
//!     outputs can carry whole vertices.
//!
//! Property names are collected for the whole query rather than per binding,
//! so a name read on one binding is kept on every vertex: a superset, never
//! a miss. Relationship property names land in the set too, harmlessly.

use std::collections::BTreeSet;

use crate::{AggregateFunction, GraphPhysicalPlan, PhysicalExpression};

/// The vertex property names `plan` can read, or `None` when it may observe
/// whole vertex records.
pub(crate) fn vertex_property_demand(plan: &GraphPhysicalPlan) -> Option<BTreeSet<String>> {
    let mut demand = BTreeSet::new();
    visit_plan(plan, &mut demand).then_some(demand)
}

/// `false` when the plan may observe whole records.
fn visit_plan(plan: &GraphPhysicalPlan, demand: &mut BTreeSet<String>) -> bool {
    use GraphPhysicalPlan as P;
    match plan {
        P::SingleRow => true,
        P::Unwind { input, expression, .. } => visit_expression(expression, demand) && visit_plan(input, demand),
        P::Union { arms, .. } => arms.iter().all(|arm| visit_plan(arm, demand)),
        P::VertexScan { scan } => {
            let mut expressions = Vec::new();
            let mut properties = Vec::new();
            scan.visit_property_reads(&mut expressions, &mut properties);
            demand.extend(properties.into_iter().map(|property| property.as_str().to_string()));
            expressions.into_iter().all(|expression| visit_expression(expression, demand))
        }
        P::OrderedRelationshipPropertyScan {
            property, items, residual, ..
        } => {
            demand.insert(property.as_str().to_string());
            items.iter().all(|item| visit_expression(&item.expression, demand))
                && residual.as_ref().is_none_or(|residual| visit_expression(residual, demand))
        }
        P::RelationshipPropertySeek { property, .. } | P::RelationshipPropertyMultiSeek { property, .. } => {
            demand.insert(property.as_str().to_string());
            true
        }
        P::RelationshipPropertyRangeProbe { property, fallback, .. } => {
            demand.insert(property.as_str().to_string());
            visit_plan(fallback, demand)
        }
        P::RelationshipTypeProbe { fallback, .. } => visit_plan(fallback, demand),
        P::Expand { input, .. } => visit_plan(input, demand),
        P::VariableExpand { input, iteration, .. } => {
            visit_plan(input, demand) && iteration.as_ref().is_none_or(|iteration| visit_plan(&iteration.plan, demand))
        }
        P::NaturalJoin { left, right, .. } => visit_plan(left, demand) && visit_plan(right, demand),
        P::Exists { input, subquery, .. } | P::ScalarSubquery { input, subquery, .. } => {
            visit_plan(input, demand) && visit_plan(subquery, demand)
        }
        P::Next { input, next } => visit_plan(input, demand) && visit_plan(next, demand),
        P::Finish { input } | P::Skip { input, .. } | P::Limit { input, .. } => visit_plan(input, demand),
        P::Filter { input, predicate, .. } => visit_expression(predicate, demand) && visit_plan(input, demand),
        P::Sort { input, items, .. } => items.iter().all(|item| visit_expression(&item.expression, demand)) && visit_plan(input, demand),
        P::Project {
            input, items, post_sort, ..
        } => {
            items.iter().all(|item| visit_expression(&item.expression, demand))
                && post_sort.iter().all(|item| visit_expression(&item.expression, demand))
                && visit_plan(input, demand)
        }
        // Outputs that can carry whole vertices: always hydrate full records.
        P::CallProcedure { .. } | P::BindPath { .. } | P::SelectPaths { .. } | P::PathUnion { .. } => false,
    }
}

/// `false` when the expression may observe a whole record.
fn visit_expression(expression: &PhysicalExpression, demand: &mut BTreeSet<String>) -> bool {
    use PhysicalExpression as E;
    match expression {
        E::Identity(_) | E::Value(_) | E::ElementTest { .. } => true,
        E::InList { expression, .. } => visit_expression(expression, demand),
        E::Property { property, .. } => {
            demand.insert(property.as_str().to_string());
            true
        }
        E::Binding(_) | E::PathConstructor(_) | E::PathNodesDistinct { .. } => false,
        E::Aggregate {
            function,
            expression,
            fraction,
            ..
        } => {
            let argument = match (function, expression.as_deref()) {
                // count(v) tests only for null, never reads the record.
                (AggregateFunction::Count, Some(E::Binding(_))) => true,
                (_, Some(argument)) => visit_expression(argument, demand),
                (_, None) => true,
            };
            argument && fraction.as_deref().is_none_or(|fraction| visit_expression(fraction, demand))
        }
        E::Function { arguments, .. } | E::List(arguments) => arguments.iter().all(|argument| visit_expression(argument, demand)),
        E::Case {
            operand,
            branches,
            default,
        } => {
            operand.as_deref().is_none_or(|operand| visit_expression(operand, demand))
                && branches
                    .iter()
                    .all(|(when, then)| visit_expression(when, demand) && visit_expression(then, demand))
                && default.as_deref().is_none_or(|default| visit_expression(default, demand))
        }
        E::Unary { expression, .. } | E::TypeTest { expression, .. } | E::Field { expression, .. } => visit_expression(expression, demand),
        E::Binary { left, right, .. } => visit_expression(left, demand) && visit_expression(right, demand),
        E::Cast { expression, request, .. } => {
            visit_expression(expression, demand) && request.as_deref().is_none_or(|request| visit_expression(request, demand))
        }
        E::Record(fields) => fields.iter().all(|(_, field)| visit_expression(field, demand)),
        E::Let { bindings, body } => bindings.iter().all(|(_, value)| visit_expression(value, demand)) && visit_expression(body, demand),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::vertex_property_demand;

    /// `None` when the engine refuses the query outright: nothing is
    /// hydrated for a query that never runs.
    fn prepared_demand(query: &str) -> Option<Option<BTreeSet<String>>> {
        crate::prepare::prepare_query_as(
            query,
            &crate::ParameterMap::new(),
            &crate::ListParameterMap::new(),
            &crate::UnknownStatistics,
            None,
        )
        .ok()
        .map(|prepared| vertex_property_demand(&prepared.physical))
    }

    fn demand(query: &str) -> Option<BTreeSet<String>> {
        prepared_demand(query).unwrap_or_else(|| panic!("the engine should accept {query}"))
    }

    fn names(names: &[&str]) -> Option<BTreeSet<String>> {
        Some(names.iter().map(|name| (*name).to_string()).collect())
    }

    #[test]
    fn id_only_traversal_demands_no_stored_property_beyond_its_seed() {
        let demand = demand("MATCH (u {id: 1})-[:R*1..3]->(v) RETURN v.id").expect("narrow demand");
        assert!(demand.len() <= 1, "{demand:?}");
    }

    #[test]
    fn every_read_property_is_demanded() {
        assert_eq!(
            demand("MATCH (a:L)-[:R]->(b:L) WHERE b.age > 3 RETURN a.name, b.city ORDER BY b.rank"),
            names(&["age", "city", "name", "rank"]),
        );
    }

    #[test]
    fn union_arms_contribute() {
        let demand = demand("MATCH (a:L) RETURN a.x AS v UNION MATCH (c:L) RETURN c.y AS v").expect("narrow demand");
        assert!(demand.contains("x") && demand.contains("y"), "{demand:?}");
    }

    /// A subquery either contributes its properties or makes the whole plan
    /// hydrate everything; it must never narrow the demand past what it reads.
    #[test]
    fn subquery_properties_are_never_dropped() {
        let query = "MATCH (a:L) WHERE EXISTS { MATCH (a)-[:R]->(b) WHERE b.flag = true } RETURN a.x";
        if let Some(demand) = demand(query) {
            assert!(demand.contains("flag") && demand.contains("x"), "{demand:?}");
        }
    }

    #[test]
    fn count_of_a_binding_does_not_need_the_record() {
        assert!(demand("MATCH (a:L)-[:R]->(b) RETURN count(b)").is_some());
    }

    #[test]
    fn whole_record_uses_demand_everything() {
        for query in [
            "MATCH (a:L)-[:R]->(b) RETURN b",
            "MATCH (a:L)-[:R]->(b) RETURN properties(b)",
            "MATCH (a:L)-[:R]->(b) RETURN keys(b)",
            "MATCH (a:L)-[:R]->(b) RETURN collect(b)",
            "MATCH p = (a:L)-[:R]->(b) RETURN p",
        ] {
            // A query the engine refuses never hydrates anything.
            if let Some(demand) = prepared_demand(query) {
                assert_eq!(demand, None, "{query}");
            }
        }
    }
}
