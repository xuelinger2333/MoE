---
type: paper
node_id: paper:cluster_topology_placement_2025
title: "Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference"
authors: ["unknown — verify before citation"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2508.09229"
  doi: null
  s2: null
tags: ["MoE", "inference", "placement", "ILP", "topology", "datacenter", "prior-art"]
added: 2026-05-18T15:30:00Z
relevance: core
note: "Author list not retrieved — verify before citation."
---

# Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference

## One-line thesis

Extend MoETuner's ILP by replacing the load-balance objective with a
**datacenter traffic minimization** objective subject to balancing
constraints, accounting for cluster topology in the cost.

## Problem / Gap

MoETuner balances load and minimizes inter-GPU comm under a uniform-cost
assumption. Real datacenter clusters have heterogeneous interconnect:
intra-node NVLink (high bandwidth) vs. inter-node InfiniBand / Ethernet
(lower bandwidth). Two cross-rank events with the same hop count may have
very different latency cost.

## Method

- Take MoETuner-style two-stage ILP as the framework
- Replace the load-equality objective in ILP 2 with a topology-weighted
  **datacenter traffic** objective
- Add explicit balancing constraints (instead of objective-level balance)
- The cost matrix between GPU pairs reflects the actual topology
  (NVLink-domain vs cross-node)

## Key Results

(Not retrieved from abstract — needs full paper read before citation)

## Assumptions

- Inherits MoETuner's assumptions (static placement, balanced, dispatch-return)
- Topology cost matrix known at deployment time
- Topology is static during deployment

## Limitations / Failure Modes

- Adds engineering burden of cluster topology probing
- Still static — no drift adaptation
- Cost model accuracy depends on collective primitive implementation

## Reusable Ingredients

- Topology cost matrix probing methodology
- Constraint-form balance (instead of objective-form)

## Open Questions

- Real-world topology variability — how much does it move the optimum?
- Is the topology-aware gain additive to MoETuner's base gain?

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
- extends → paper:go2025_moetuner
- invalidates → idea:pipelined_runtime_placement (Plan C extension)
- addresses_gap → gap:G6

## Relevance to This Project

**Critical prior art.** Pre-empts what was going to be Plan C
("topology-aware cost model") in our killed brainstorm. Together with
ExFlow + MoETuner, closes the entire static placement design space for
MoE inference. Any future system paper in this area must differentiate
against all three.
