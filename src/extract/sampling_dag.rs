use std::{collections::HashMap, mem};

use egraph_serialize::{ClassId, EGraph, NodeId};
use indexmap::IndexMap;
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::ThreadRng,
    thread_rng,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{faster_bottom_up::FasterBottomUpExtractor, Cost, ExtractionResult, Extractor};

pub struct SamplingExtractor;

impl Extractor for SamplingExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        let (initial_result, costs) = FasterBottomUpExtractor.extract_with_costs(egraph, roots);
        // Note that the initial cost is the tree cost, so it is not precise
        let (mut distributions, initial_cost) =
            self.initialize_distributions(egraph, roots, &costs);

        let n_iters = usize::min(egraph.classes().len() * 100, 1_000_000);
        let mut best_cost = initial_cost;
        let mut result = initial_result;
        'sample_iter: for _iter in 0..n_iters {
            let mut state = Default::default();
            // All roots are sampled together, since the DAG cost
            // allows sharing among roots.
            for root in roots {
                if self
                    .sample(egraph, root, &mut state, &mut distributions)
                    .is_none()
                {
                    // found a cycle; gave up this sample
                    continue 'sample_iter;
                }
                assert!(state.working_sets[&root].contains_key(root));
            }
            let working_set = merge_sets(state.working_sets.values().cloned().collect());
            let cost = working_set
                .iter()
                .map(|(_c, n)| egraph.nodes[n].cost)
                .sum::<Cost>();
            if cost < best_cost {
                println!("new best cost: {:?}", cost);
                best_cost = cost;
                result = ExtractionResult {
                    choices: working_set.into_iter().collect(),
                };
            }
        }

        result
    }
}

impl SamplingExtractor {
    fn initialize_distributions(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
        costs: &FxHashMap<ClassId, Cost>,
    ) -> (FxHashMap<ClassId, ClassDist>, Cost) {
        let alpha: Cost = (egraph.nodes.values().map(|n| n.cost).sum::<Cost>()
            / egraph.nodes.len() as f64)
            .recip()
            .try_into()
            .unwrap();

        let mut weights: FxHashMap<ClassId, Vec<(Cost, NodeId)>> = FxHashMap::default();
        for (nid, node) in egraph.nodes.iter() {
            let cid = egraph.nid_to_cid(&nid);
            let cost = node.cost
                + node
                    .children
                    .iter()
                    .map(|c| egraph.nodes[c].cost)
                    .sum::<Cost>();
            weights
                .entry(cid.clone())
                .or_default()
                .push((cost, nid.clone()));
        }

        let mut distributions = FxHashMap::default();
        for (cid, dist) in weights {
            let weights = dist.iter().map(|d| d.0);
            let nodes: Vec<NodeId> = dist.iter().map(|d| d.1.clone()).collect();
            distributions.insert(cid, ClassDist::new(weights.collect(), nodes, alpha));
        }

        (distributions, roots.iter().map(|r| costs[r]).sum())
    }

    fn sample(
        &self,
        egraph: &EGraph,
        root: &ClassId,
        state: &mut State,
        distributions: &mut FxHashMap<ClassId, ClassDist>,
    ) -> Option<()> {
        if state.vis.contains(&root) {
            return None;
        }

        state.vis.insert(root.clone());

        let (working_set, picked_nid, picked_node) = if state.picked_nodes.contains_key(&root) {
            // This node has been picked before during this sampling run,
            // we can just reuse the cost
            let reuse_root_cid = state.picked_nodes.get(&root).unwrap();
            let reuse_working_set = state.working_sets.get(reuse_root_cid).unwrap().clone();
            let mut working_set = FxHashMap::default();
            let mut todo = vec![root.clone()];
            while !todo.is_empty() {
                let cid = todo.pop().unwrap();
                let nid = reuse_working_set.get(&cid).unwrap().clone();
                working_set.insert(cid.clone(), nid.clone());
                let node = egraph.nodes.get(&nid).unwrap();
                for child in node.children.iter() {
                    let child_cid = egraph.nid_to_cid(child).clone();
                    if !working_set.contains_key(&child_cid) {
                        todo.push(child_cid);
                    }
                }
            }

            let nid = reuse_working_set.get(&root).unwrap().clone();
            let node = egraph.nodes.get(&nid).unwrap();
            (working_set, nid, node)
        } else {
            // sampling children
            let class_dist = distributions.get_mut(&root).unwrap();
            let nid = class_dist.sample().clone();
            let node = egraph.nodes.get(&nid).unwrap();
            let children = node.children.iter().map(|n| egraph.nid_to_cid(n));
            let mut child_working_sets = vec![];
            for c in children.clone() {
                self.sample(egraph, c, state, distributions)?;
            }

            // collect merged working sets of children
            for c in children {
                if let Some(child_working_set) = state.working_sets.remove(&c) {
                    child_working_sets.push(child_working_set);
                }
            }
            let mut working_set = merge_sets(child_working_sets);
            working_set.insert(root.clone(), nid.clone());

            // update picked_nodes table. This table is used
            // to avoid re-sampling the same node.
            for (cid, _nid) in working_set.iter() {
                state.picked_nodes.insert(cid.clone(), root.clone());
            }

            (working_set, nid, node)
        };

        // update costs
        let cost = picked_node.cost
            + working_set
                .iter()
                .map(|(_c, n)| egraph.nodes[n].cost)
                .sum::<Cost>();
        let class_dist = distributions.get_mut(&root).unwrap();
        if cost < class_dist.get_weight(&picked_nid) {
            class_dist.update_weight(picked_nid.clone(), cost);
        }

        state.working_sets.insert(root.clone(), working_set);

        state.vis.remove(&root);

        Some(())
    }
}

// Merging a list of sets in linear time.
fn merge_sets(
    mut child_working_sets: Vec<FxHashMap<ClassId, NodeId>>,
) -> FxHashMap<ClassId, NodeId> {
    if child_working_sets.is_empty() {
        return FxHashMap::default();
    }
    let child_working_set = child_working_sets
        .iter_mut()
        .max_by_key(|ws| ws.len())
        .unwrap();
    let mut ws = mem::take(child_working_set);
    for child_working_set in child_working_sets {
        ws.extend(child_working_set.into_iter());
    }
    ws
}

type RootClassId = ClassId;

#[derive(Default)]
struct State {
    // There could be multiple RootClassIds but we consider the first one
    picked_nodes: FxHashMap<ClassId, RootClassId>,
    // For each class node on the path to root, the set of E-classes covered by this node.
    working_sets: FxHashMap<RootClassId, FxHashMap<ClassId, NodeId>>,
    vis: FxHashSet<ClassId>,
}

struct ClassDist {
    dist: WeightedIndex<Cost>,
    weight_map: IndexMap<NodeId, Cost>,
    alpha: Cost,
    rng: ThreadRng,
}

impl ClassDist {
    fn new(mut weights: Vec<Cost>, nodes: Vec<NodeId>, alpha: Cost) -> ClassDist {
        // There are two kinds of weights here: the sampler uses
        // the negative log probability for the weight, while the
        // weight_map stores the cost.
        let mut weight_map = IndexMap::new();
        for (w, n) in weights.iter().zip(nodes.into_iter()) {
            // println!("init weight: {:?} {:?}", n, w);
            weight_map.insert(n, *w);
        }

        for weight in weights.iter_mut() {
            *weight = (-*weight * alpha).exp().try_into().unwrap();
        }
        let dist = WeightedIndex::<Cost>::new(weights.iter()).unwrap();

        ClassDist {
            dist,
            alpha,
            weight_map,
            rng: thread_rng(),
        }
    }

    fn sample(&mut self) -> &NodeId {
        let idx = self.dist.sample(&mut self.rng);
        self.weight_map.get_index(idx).unwrap().0
    }

    fn get_weight(&self, nid: &NodeId) -> Cost {
        *self.weight_map.get(nid).unwrap()
    }

    fn update_weight(&mut self, nid: NodeId, w: Cost) {
        let exp_w: Cost = (-w * self.alpha).exp().try_into().unwrap();
        // println!("update weight: {:?} {:?}", nid, w);
        let entry = self.weight_map.entry(nid).and_modify(|e| *e = w);

        let idx = entry.index();
        self.dist.update_weights(&[(idx, &exp_w)]).unwrap();
    }
}
