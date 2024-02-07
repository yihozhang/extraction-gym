use std::collections::HashMap;

use egraph_serialize::{ClassId, EGraph, NodeId};
use ordered_float::NotNan;
use rand::{distributions::{Distribution, WeightedIndex}, rngs::ThreadRng, thread_rng, Rng};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{faster_bottom_up::FasterBottomUpExtractor, Cost, ExtractionResult, Extractor};

pub struct SamplingExtractor;

impl Extractor for SamplingExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        let initial_result = FasterBottomUpExtractor.extract(egraph, roots);
        let mut distributions = self.initialize_distributions(egraph, roots, initial_result);

        let n_iters = egraph.classes().len() * 100;
        for iter in 0..n_iters {
            for root in roots {
                self.sample(egraph, *root, &mut distributions);
            }
        }

        todo!()
    }
}

impl SamplingExtractor {
    fn initialize_distributions(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
        initial_result: ExtractionResult,
    ) -> FxHashMap<ClassId, ClassDist> {
        let node_roots = roots
            .iter()
            .map(|cid| initial_result.choices[cid].clone())
            .collect::<Vec<NodeId>>();
        let mut memo = HashMap::default();
        initial_result.tree_cost_rec(egraph, &node_roots, &mut memo);

        let mut weights: FxHashMap<ClassId, Vec<(Cost, NodeId)>> = FxHashMap::default();
        for (nid, cost) in memo.into_iter() {
            let cid = egraph.nid_to_cid(&nid);
            weights.entry(*cid).or_default().push((cost, nid));
        }

        let mut distributions = FxHashMap::default();
        for (cid, dist) in weights {
            let weights = dist.iter().map(|d| d.0);
            let nodes: Vec<NodeId> = dist.iter().map(|d| d.1).collect();
            distributions[&cid] = ClassDist::new(WeightedIndex::new(weights).unwrap(), nodes);
        }

        distributions
    }

    fn sample(&self, egraph: &EGraph, root: ClassId, distributions: &mut FxHashMap<ClassId, ClassDist>) {
        let class_dist = distributions.get_mut(&root).unwrap();
        let nid = class_dist.sample();
        let node = egraph.nodes[&nid];

        let working_set: FxHashMap<ClassId, (NodeId, Cost)>  = FxHashMap::default();
        

        let children = node.children.iter().map(|n| egraph.nid_to_cid(n));
        for c in children {
            self.sample(egraph, *c, distributions);
        }
        
    }
}

struct State {
    ancestor_select: ExtractionResult,
}

struct ClassDist {
    dist: WeightedIndex<NotNan<f64>>,
    nodes: Vec<NodeId>,
    rng: ThreadRng,    
}

impl ClassDist {
    fn new(dist: WeightedIndex<NotNan<f64>>, nodes: Vec<NodeId>) -> ClassDist {
        ClassDist { dist, nodes, rng: thread_rng() }
    }

    fn sample(&mut self) -> NodeId {
        let idx = self.dist.sample(&mut self.rng);
        self.nodes[idx]
    }
}
