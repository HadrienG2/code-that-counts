use super::pool::{Aggregator, JobScheduler, Stopped};
use crossbeam_utils::CachePadded;
use hwloc2::CpuSet;

/// Behaves like Aggregator, but with a scalable reduction tree implementation
#[derive(Debug)]
pub struct AggregatorTree {
    /// Tree nodes in breadth-first order
    nodes: Vec<PaddedNode>,

    /// Root node index
    root_idx: usize,
}
//
/// Cache-padded AggregatorNode
type PaddedNode = CachePadded<AggregatorNode>;
//
/// Node of the AggregatorTree
#[derive(Debug)]
struct AggregatorNode {
    /// Aggregation node
    aggregator: Aggregator,

    /// Parent index (root node is its own parent)
    parent_idx: usize,

    /// Number of children
    num_children: u32,
}
//
impl AggregatorTree {
    /// Construct an aggregator tree for the active process' CPUset
    ///
    /// In addition to the tree, a vector of thread metadata is produced. This
    /// assigns a CpuSet to each processing thread for pinning purposes, as well
    /// as a leaf node index in the `AggregatorTree`.
    ///
    /// The `max_arity` tuning parameters controls the tradeoff between
    /// thread contention (which leads to slower atomic operations) and
    /// reduction tree depth (which leads to more atomic operations). If it is
    /// set to `usize::MAX`, the tree strictly follows the hwloc topology, so
    /// e.g. if a CPU L3 cache shard is shared by 28 children, the tree node
    /// associated with that cache shard will be shared by 28 children.
    ///
    /// As the value gets lower, tree nodes with too many children are split
    /// into subtrees, so in the example above, a `max_arity` of 2 would produce
    /// a tree with two first-level nodes with 14 recursive children, each split
    /// into two second-level nodes with 7 children, each split into one
    /// third-level node with 3 children and another with 4 children, and so on.
    ///
    pub fn new(topology: &hwloc2::Topology, max_arity: u32) -> (Self, Vec<ThreadConfig>) {
        assert!(max_arity >= Self::MIN_ARITY, "Need at least a binary tree");

        // Fill the tree
        let mut result = Self {
            nodes: Vec::new(),
            root_idx: Self::INVALID_IDX,
        };
        let mut threads = Vec::new();
        let top_children = result.add_children(
            &mut threads,
            max_arity,
            topology.object_at_root(),
            Self::INVALID_IDX,
        );

        // Set up root node
        match top_children[..] {
            // Normal case where a single root node is returned
            [AggregatedChild::Node(root_idx)] => result.root_idx = root_idx,

            // Single-threaded edge case where no root node has been created
            // because each recursive call to result.fill() deferred root node
            // creation to a higher-level hwloc topology node with more children.
            [AggregatedChild::Thread(_thread_idx)] => {
                debug_assert_eq!(threads.len(), 1);
                debug_assert_eq!(top_children, vec![AggregatedChild::Thread(0)]);
                result.nodes.push(CachePadded::new(AggregatorNode {
                    aggregator: Aggregator::new(),
                    parent_idx: Self::INVALID_IDX,
                    num_children: 1,
                }));
                result.root_idx = result.nodes.len() - 1;
                result.root().aggregator.reset(1);
            }

            // Returning multiple children cannot happen AFAIK since it means
            // hwloc would stuff all CPUs in the root node without a hierarchy
            _ => unreachable!(),
        }

        // Validate the tree in debug builds
        if cfg!(debug_assertions) {
            result.validate(topology, max_arity, &threads);
        }
        (result, threads)
    }

    /// Prepare the aggregator for a new job
    ///
    /// For scalability reasons, this only resets the root node. Lower-level
    /// AggregatorNodes are instead reset in `task_done()` by the last child of
    /// the node of interest to merge results.
    ///
    pub fn reset(&self) {
        let root = self.root();
        root.aggregator.reset(root.num_children)
    }

    /// Aggregate a thread's partial results, tell the job's result if finished
    pub fn task_done(&self, result: u64, node_idx: usize) -> Option<u64> {
        // Aggregate our result into the target node
        let node = &self.nodes[node_idx];
        let node_result = node.aggregator.task_done(result)?;

        // We're done with this node. Was it the root node?
        if node_idx != self.root_idx {
            // If not, reset this node and propagate its results to the parent
            node.aggregator.reset(node.num_children);
            self.task_done(node_result, node.parent_idx)
        } else {
            // Otherwise, the full aggregation is done
            Some(node_result)
        }
    }

    /// Wait for the job to be done or error out, collect the result
    // NOTE: If this flavor of impl Trait in trait doesn't work, parametrize the
    //       trait and impl by the JobScheduler.
    pub fn wait_for_result(&self, scheduler: &impl JobScheduler) -> Result<u64, Stopped> {
        self.root().aggregator.wait_for_result(scheduler)
    }

    /// Translate an hwloc object into nodes or leaves of the reduction tree
    fn add_children(
        &mut self,
        threads: &mut Vec<ThreadConfig>,
        max_arity: u32,
        object: &hwloc2::TopologyObject,
        parent_idx: usize,
    ) -> Vec<AggregatedChild> {
        debug_assert!(max_arity >= Self::MIN_ARITY);

        // Ignore objects with no CPUs attached early on
        let Some(object_cpuset) = object.cpuset() else { return Vec::new() };

        // Upon reaching a leaf hwloc object, emit a list of leaf threads
        let Some(first_child) = object.first_child() else {
            return Self::add_threads(threads, object_cpuset, parent_idx);
        };

        // Pass through hwloc objects with only one child, like L1 CPU caches:
        // they are not interesting from a parallel aggregation perspective, but
        // may lead to more interesting children.
        if object.arity() == 1 {
            return self.add_children(threads, max_arity, first_child, parent_idx);
        }

        // If we reached this point, then the active hwloc object has multiple
        // direct children, from the perspective of hwloc at least. Collect
        // these, binding them to an invalid parent index for now.
        let mut children = Vec::new();
        let mut child_opt = Some(first_child);
        while let Some(child) = child_opt {
            children.extend(self.add_children(threads, max_arity, child, usize::MAX));
            child_opt = child.next_sibling();
        }

        // If there turns out to be only 0 or 1 CPU-bearing children, then we
        // don't need a root node, attach any child directly to the parent.
        if children.len() < 2 {
            for child in &mut children {
                child.rebind(self, threads, parent_idx);
            }
            return children;
        }

        // If control reached this point, then we truly have multiple CPU-bearing
        // children that we want to aggregate through one or more AggregatorNodes.
        // Return a list of AggregatorNodes to which children can be attached,
        // along with a count of how many children they can host.
        let root_idx = self.nodes.len();
        let parent_capacity = self.add_nodes(max_arity, parent_idx, children.len());
        debug_assert_eq!(
            parent_capacity
                .clone()
                .into_iter()
                .map(|(_idx, num_children)| num_children)
                .sum::<u32>(),
            children.len() as u32
        );

        // Deduce an iterator of "children slots" with the right multiplicity
        let children_slots = parent_capacity
            .into_iter()
            .flat_map(|(parent_idx, capacity)| {
                std::iter::repeat(parent_idx).take(capacity as usize)
            });

        // Bind our children to these parent slots
        for (child, parent_idx) in children.iter_mut().zip(children_slots) {
            child.rebind(self, threads, parent_idx);
        }

        // Expose our root node as the child of the caller
        children.clear();
        children.push(AggregatedChild::Node(root_idx));
        children
    }

    /// Configure a set of threads based on a CPUset and a parent node index
    ///
    /// Threads will be added to the global thread list and also returned as a
    /// children list usable as an `add_children` return value, for the purpose
    /// of later being bound to AggregatorNodes.
    ///
    /// The initial parent node index can be invalid, it will be patched later on.
    ///
    fn add_threads(
        threads: &mut Vec<ThreadConfig>,
        cpuset: CpuSet,
        parent_idx: usize,
    ) -> Vec<AggregatedChild> {
        let mut children = Vec::with_capacity(usize::try_from(cpuset.weight()).unwrap());
        for cpu in cpuset {
            children.push(AggregatedChild::Thread(threads.len()));
            threads.push(ThreadConfig {
                cpuset: CpuSet::from(cpu),
                aggregator_idx: parent_idx,
            });
        }
        return children;
    }

    /// Set up a (sub-)tree of AggregatorNodes that can hold a number of children
    ///
    /// The initial parent node index can be invalid, it will be patched later on.
    ///
    /// Tell which of these nodes can hold children, and how many children each
    /// of these can hold.
    ///
    fn add_nodes(
        &mut self,
        max_arity: u32,
        parent_idx: usize,
        num_children: usize,
    ) -> impl IntoIterator<Item = (usize, u32)> + Clone {
        // Create a root node with as many children as possible
        let num_children = u32::try_from(num_children).unwrap();
        let first_batch = num_children.min(max_arity);
        let root_idx = self.nodes.len();
        self.nodes.push(CachePadded::new(AggregatorNode {
            aggregator: Aggregator::new(),
            parent_idx,
            num_children: first_batch,
        }));

        // If that does not suffice, we will add more children one by
        // one across nodes that can host them, fanning them out in a
        // round robin fashion using the following state that tracks
        // which nodes can still accept new children.
        let mut parent_start = root_idx;
        let mut parent_idx = parent_start;

        // Once all nodes are filled to capacity, we start turning
        // "scalar" children into subtrees in a breadth-first and
        // round-robin fashion:
        //
        // - First we turn the first child of the root into a subtree
        // - Then we turn the second child of the root into a subtree
        // - ...repeat until all root node children are subtrees...
        // - Then the first child of the first subtree becomes a subtree.
        // - Then the first child of the second subtree...
        // - ...repeat until first children of all root subtrees are subtrees...
        // - Then we turn to the second child of all root subtrees...
        //
        let mut subtree_parent_start = root_idx;
        let mut subtree_parent_end = self.nodes.len();
        let mut subtree_parent_idx = subtree_parent_start;
        let mut subtree_capacity = max_arity;

        // Allocate remaining children, if any
        for _ in first_batch..num_children {
            // First allocate children among nodes that still have
            // capacity to host more, in a round-robin fashion
            if self.nodes[parent_idx].num_children < max_arity {
                self.nodes[parent_idx].num_children += 1;
                parent_idx += 1;
                if parent_idx == self.nodes.len() {
                    parent_idx = parent_start;
                }
                continue;
            } else {
                debug_assert!(self
                    .nodes
                    .iter()
                    .skip(root_idx)
                    .all(|node| node.num_children == max_arity));
            }

            // Once all tree nodes have the maximal amount of children
            // allowed by arity, it's time to create a new sub-tree. We make it
            // fully filled initially so that we can handle it homogeneously
            // with respect to other sub-trees affected by this operations.
            self.nodes.push(CachePadded::new(AggregatorNode {
                aggregator: Aggregator::new(),
                parent_idx: subtree_parent_idx,
                num_children: max_arity,
            }));

            // Creating a subtree replaces an existing child and hosts the newly
            // allocated child, making room for (max_arity - 2) more children.
            let new_children = max_arity - 2;
            if new_children > 0 {
                // Distribute "holes" among the most recently created tree
                // nodes, with priority given to newer nodes further away from
                // the root (we fill the tree top to bottom and left to right)
                let num_hosts = (self.nodes.len() - root_idx + 1).min(new_children as usize);
                let new_children_share = new_children / num_hosts as u32;
                let extra_new_children = new_children as usize % num_hosts;
                //
                for (rev_idx, node) in self.nodes.iter_mut().rev().take(num_hosts).enumerate() {
                    node.num_children -= new_children_share;
                    if rev_idx < extra_new_children {
                        node.num_children -= 1;
                    }
                }

                // Update iteration state so that next time we will add
                // children to the earliest node with least children.
                let parent_end = self.nodes.len();
                parent_start = parent_end - num_hosts;
                if extra_new_children > 0 {
                    parent_idx = parent_end - extra_new_children;
                } else {
                    parent_idx = parent_start;
                }
            }

            // Decide where the next subtree will be inserted
            // Round-robin through top-level nodes whose children are
            // not all subtrees yet...
            subtree_parent_idx += 1;
            if subtree_parent_idx == subtree_parent_end {
                // ...while accounting how many children can still be
                // turned into subtrees along the way...
                subtree_capacity -= 1;
                if subtree_capacity == 0 {
                    // ...then, once all children of this layer of the
                    // tree have become subtrees, go to the next layer.
                    subtree_parent_start = subtree_parent_end;
                    subtree_parent_end = self.nodes.len();
                    subtree_capacity = max_arity;
                }
                subtree_parent_idx = subtree_parent_start;
            }
        }

        // At this point, our tree of AggregatorNodes is built.
        // Configure the associated Aggregators for the right number of children.
        for node in &mut self.nodes[root_idx..] {
            node.aggregator.reset(node.num_children);
        }

        // Now it's time to determine where children can bind in that tree.
        let mut parent_capacity = Vec::with_capacity(self.nodes.len() - subtree_parent_start);

        // Below subtree_parent_start, all nodes are full of subtree
        // children, with no room for threads & friends, so we ignore those nodes.
        for parent_idx in subtree_parent_start..subtree_parent_end {
            // Between subtree_parent_start and subtree_parent_end, we were in
            // the process of turning children into subtrees but not done yet.
            let num_children = self.nodes[parent_idx].num_children;
            let mut num_subtrees = max_arity - subtree_capacity;
            if parent_idx < subtree_parent_idx {
                num_subtrees += 1;
            }
            debug_assert!(num_children >= num_subtrees);
            let capacity = num_children - num_subtrees;
            parent_capacity.push((parent_idx, capacity));
        }
        for parent_idx in subtree_parent_end..self.nodes.len() {
            // Above subtree_parent_end, there are no subtrees yet, so every
            // child of an AggregatorNode is a direct child.
            let capacity = self.nodes[parent_idx].num_children;
            parent_capacity.push((parent_idx, capacity));
        }
        parent_capacity
    }

    /// Validate that the output tree is consistent with constructor parameters
    fn validate(&self, topology: &hwloc2::Topology, max_arity: u32, threads: &Vec<ThreadConfig>) {
        let mut children = vec![vec![]; self.nodes.len()];
        let mut full_cpuset = CpuSet::new();

        for (idx, thread) in threads.into_iter().enumerate() {
            // Check that each thread targets a single, unique CPU
            assert_eq!(thread.cpuset.weight(), 1);
            let cpu = u32::try_from(thread.cpuset.first()).unwrap();
            assert!(!full_cpuset.is_set(cpu));
            full_cpuset.set(cpu);

            // Check that each thread aggregates into a valid AggregatorNode
            assert!(thread.aggregator_idx < self.nodes.len());
            children[thread.aggregator_idx].push(AggregatedChild::Thread(idx));
        }

        // Check that threads cover all available CPUs
        assert_eq!(full_cpuset, topology.object_at_root().cpuset().unwrap());

        // Check that a root node is present
        assert_ne!(self.nodes.len(), 0);

        // Check that AggregatorNodes follow expectations
        for (idx, node) in self.nodes.iter().enumerate() {
            // Inner Aggregator has been reset properly
            assert_eq!(node.aggregator.remaining_tasks(), node.num_children);
            assert_eq!(node.aggregator.result(), 0);

            // Check that each node aggregates into a valid target
            if idx == self.root_idx {
                assert!(node.parent_idx == Self::INVALID_IDX);
            } else {
                assert!(node.parent_idx < self.nodes.len());
                children[node.parent_idx].push(AggregatedChild::Node(idx));
            }

            // Check observed arities
            assert!(node.num_children > 0);
            assert!(node.num_children <= max_arity);
        }

        // Compare node child count to actual child count
        for (node, node_children) in self.nodes.iter().zip(children.iter()) {
            assert_eq!(node.num_children, node_children.len() as u32);
        }

        // Traverse the tree from the root, make sure all nodes are reachable
        let mut reached = 0;
        let mut depth = 0;
        let mut next_nodes = vec![self.root_idx];
        while !next_nodes.is_empty() {
            depth += 1;
            for next_node in std::mem::take(&mut next_nodes) {
                reached += 1;
                for child in children[next_node].iter().copied() {
                    match child {
                        AggregatedChild::Node(idx) => next_nodes.push(idx),
                        _ => {}
                    }
                }
            }
        }
        assert_eq!(reached, self.nodes.len());

        // Print out the tree
        let mut lines = vec![String::new(); 2 * depth + 1];
        fn print_tree(
            lines: &mut Vec<String>,
            children: &Vec<Vec<AggregatedChild>>,
            root_idx: usize,
            depth: usize,
        ) {
            use std::fmt::Write;
            let parent_line = 2 * depth;
            let dash_line = parent_line + 1;
            let child_line = parent_line + 2;

            write!(&mut lines[parent_line], "N{root_idx} ").unwrap();
            write!(&mut lines[dash_line], "|").unwrap();
            for child in children[root_idx].iter() {
                match child {
                    AggregatedChild::Thread(thread_idx) => {
                        write!(&mut lines[child_line], "T{thread_idx} ").unwrap();
                    }
                    AggregatedChild::Node(child_idx) => {
                        print_tree(lines, children, *child_idx, depth + 1)
                    }
                }
            }

            let parent_len = lines[parent_line].len();
            let dash_len = lines[dash_line].len();
            let child_len = lines[child_line].len();
            lines[parent_line]
                .extend(std::iter::repeat(' ').take(child_len.saturating_sub(parent_len)));
            lines[dash_line].extend(
                std::iter::repeat('-')
                    .take(child_len.saturating_sub(dash_len + 1))
                    .chain(std::iter::once(' ')),
            );
            for subchild_line in lines.iter_mut().skip(child_line + 1) {
                let subchild_line_len = subchild_line.len();
                subchild_line.extend(
                    std::iter::repeat(' ').take(child_len.saturating_sub(subchild_line_len)),
                );
            }
        }
        print_tree(&mut lines, &children, self.root_idx, 0);
        for line in lines {
            eprintln!("{line}");
        }
    }

    /// Minimum tree arity needed for reduction to be possible
    const MIN_ARITY: u32 = 2;

    /// Invalid node index, to be patched later on, or kept for the root node
    const INVALID_IDX: usize = usize::MAX;

    /// Shared access to the root node
    fn root(&self) -> &AggregatorNode {
        &self.nodes[self.root_idx]
    }
}
//
/// Assignment of worker threads to CPUs and aggregator tree nodes
#[derive(Clone, Debug, PartialEq)]
pub struct ThreadConfig {
    /// CpuSet to be used when pinning this thread to its target CPU
    pub cpuset: hwloc2::CpuSet,

    /// Index of the parent AggregatorNode in the AggregatorTree
    pub aggregator_idx: usize,
}
//
/// Child of an AggregatorNode
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum AggregatedChild {
    /// Leaf CPU thread
    Thread(usize),

    /// Other AggregatorNode
    Node(usize),
}
//
impl AggregatedChild {
    /// Rebind this child to a different parent node
    fn rebind(
        &self,
        tree: &mut AggregatorTree,
        threads: &mut Vec<ThreadConfig>,
        parent_idx: usize,
    ) {
        match self {
            Self::Thread(idx) => {
                threads[*idx].aggregator_idx = parent_idx;
            }
            Self::Node(idx) => {
                tree.nodes[*idx].parent_idx = parent_idx;
            }
        }
    }
}

// TODO: Add a ResultAggregator trait implemented by both Aggregator and
//       AggregatorTree. Need to dummy out ThreadConfig and node_idx in the
//       regular Aggregator.

// TODO: Also do a reduction tree for FutexScheduler, then see if I can find
//       some commonalities.

#[cfg(test)]
mod tests {
    use super::*;
    use hwloc2::Topology;

    #[test]
    fn aggregator_tree() {
        let topology = Topology::new().unwrap();
        for max_arity in 2..=(std::thread::available_parallelism()
            .map(|nzu| usize::from(nzu))
            .unwrap_or(2) as u32
            + 1)
        {
            // Create an AggregatorTree, rely on debug assertions to check it
            AggregatorTree::new(&topology, max_arity);
        }
    }
}
