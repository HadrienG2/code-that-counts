use std::num::NonZeroUsize;

use super::pool::Reducer;
use crossbeam_utils::CachePadded;

// AtomicU32 can be used as a very basic Reducer that counts passing threads
impl Reducer for std::sync::atomic::AtomicU32 {
    fn reset(&self, num_threads: u32) {
        self.store(num_threads, atomic::Ordering::Relaxed)
    }

    fn has_remaining_threads(&self) -> bool {
        self.load(atomic::Ordering::Relaxed) != 0
    }

    type Contribution = ();

    fn current_result(&self) {}

    type AccumulatorId = ();

    fn thread_done(&self, (): (), ordering: atomic::Ordering, (): ()) -> Option<()> {
        match self.fetch_sub(1, ordering) {
            // All threads have reached the barrier, we're done
            1 => Some(()),

            // A number of threads greater than announced to `reset()` has
            // reached the barrier, this is a usage error.
            0 => panic!("reset() was called with an invalid number of threads"),

            // Not all threads have reached the barrier yet
            _prev_remaining_threads => None,
        }
    }
}

/// Behaves like a Reducer, but with a scalable reduction tree implementation
pub struct ReductionTree<Reducer> {
    /// Tree nodes
    nodes: Vec<CachePadded<ReductionNode<Reducer>>>,

    /// Root node index
    root_idx: NullableIdx,
}
//
/// Node of the ReductionTree
struct ReductionNode<Reducer> {
    /// Reduction node
    reducer: Reducer,

    /// Parent index
    parent_idx: NullableIdx,

    /// Number of children
    num_children: u32,
}
//
impl<R: Default + Reducer<AccumulatorId = ()>> ReductionTree<R> {
    /// Construct a reduction tree for the active process' CPUset
    ///
    /// In addition to the tree, a vector of thread metadata is produced. This
    /// assigns a CpuSet to each processing thread for pinning purposes, as well
    /// as an accumulator node inside of the `ReductionTree`.
    ///
    /// The `max_arity` tuning parameters controls the tradeoff between
    /// thread contention (which leads to slower atomic operations) and
    /// reduction tree depth (which leads to more atomic operations). If it is
    /// set to `usize::MAX`, the tree strictly follows the hwloc topology, so
    /// e.g. if a CPU L3 cache shard is shared by 28 children, the tree node
    /// associated with that cache shard will be shared by 28 children.
    ///
    /// As the value gets lower, tree nodes with too many children are split
    /// into subtrees, so in the example above, a `max_arity` of 2 could produce
    /// a tree with two first-level nodes with 14 recursive children, each split
    /// into two second-level nodes with 7 children, each split into one
    /// third-level node with 3 children and another with 4 children, and so on.
    ///
    pub fn new(topology: &hwloc2::Topology, max_arity: u32) -> (Self, Vec<ThreadConfig<usize>>) {
        assert!(max_arity >= Self::MIN_ARITY, "Need at least a binary tree");

        // Fill the tree
        let mut result = Self {
            nodes: Vec::new(),
            root_idx: NullableIdx::none(),
        };
        let mut threads = Vec::new();
        let top_children = result.add_children(&mut threads, max_arity, topology.object_at_root());

        // Set up root node
        match top_children[..] {
            // Normal case where a single root node is returned
            [ReducedChild::Node(root_idx)] => result.root_idx = NullableIdx::some(root_idx),

            // Single-threaded edge case where no root node has been created
            // because each recursive call to result.fill() deferred root node
            // creation to a higher-level hwloc topology node with more children.
            [ReducedChild::Thread(0)] => {
                assert_eq!(result.num_nodes(), 0);
                assert_eq!(threads.len(), 1);
                result
                    .nodes
                    .push(CachePadded::new(ReductionNode::new(NullableIdx::none(), 1)));
                result.root_idx = NullableIdx::some(0);
                result.root().reducer.reset(1);
            }

            // Returning multiple children cannot happen AFAIK since it means
            // hwloc would stuff all CPUs in the root node without any hierarchy.
            // And returning zero children should not happen because since we
            // are executing code, at least one CPU should be present ;)
            _ => unreachable!(),
        }

        // At this point, all threads should have a valid parent index
        let threads = threads
            .into_iter()
            .map(|t| t.map_id(|id| id.value().unwrap()))
            .collect();

        // Validate the tree in debug builds
        if cfg!(debug_assertions) {
            result.validate(topology, max_arity, &threads);
        }
        (result, threads)
    }

    /// Construct a reduction tree that follows the same structure as this one
    /// but uses a different Reducer, with freshly reset nodes
    pub fn rebind<R2: Default + Reducer>(&self) -> ReductionTree<R2> {
        ReductionTree {
            nodes: self
                .nodes
                .iter()
                .map(|node| CachePadded::new(node.rebind::<R2>()))
                .collect(),
            root_idx: self.root_idx,
        }
    }

    /// Translate an hwloc object into nodes or leaves of the reduction tree
    ///
    /// Collects thread objects mapping into individual processing threads, and
    /// returns a list of direct CPU-bearing children (threads or reduction
    /// tree nodes) which have not been assigned a parent yet.
    ///
    fn add_children(
        &mut self,
        threads: &mut Vec<ThreadConfig<NullableIdx>>,
        max_arity: u32,
        object: &hwloc2::TopologyObject,
    ) -> Vec<ReducedChild> {
        debug_assert!(max_arity >= Self::MIN_ARITY);

        // Ignore objects with no CPUs attached
        let Some(object_cpuset) = object.cpuset() else { return Vec::new() };

        // Upon reaching a leaf hwloc object, emit a list of leaf CPU threads
        let Some(first_child) = object.first_child() else {
            return Self::add_threads(threads, object_cpuset);
        };

        // Pass through hwloc objects with only one child, like L1 CPU caches:
        // they are not interesting from a parallel reduction perspective, but
        // will eventually lead to more interesting children
        if object.arity() == 1 {
            return self.add_children(threads, max_arity, first_child);
        }

        // If we reached this point, then the active hwloc object has multiple
        // direct children from the perspective of hwloc, collect these
        let mut children = Vec::new();
        let mut child_opt = Some(first_child);
        while let Some(child) = child_opt {
            children.extend(self.add_children(threads, max_arity, child));
            child_opt = child.next_sibling();
        }

        // If there turns out to be only 0 or 1 CPU-bearing children, then we
        // don't need a root reduction node, we just bind to the parent
        if children.len() < 2 {
            return children;
        }

        // If we have multiple CPU-bearing children, we want to aggregate their
        // results through a tree of ReductionNodes
        let num_children = u32::try_from(children.len()).unwrap();
        let (root_idx, children_slots) = self.add_nodes(max_arity, num_children);

        // Bind our children to nodes of the newly created reduction subtree
        for (child, parent_idx) in children.iter_mut().zip(children_slots) {
            child.set_parent(self, threads, NullableIdx::some(parent_idx));
        }

        // Expose the root node as our only direct child
        children.clear();
        children.push(ReducedChild::Node(root_idx));
        children
    }

    /// Configure a set of threads based on a CPUset and a parent node index
    ///
    /// Threads will be added to the global thread list and also returned as a
    /// children list usable as an `add_children` return value, for the purpose
    /// of later being bound to ReductionNodes.
    ///
    fn add_threads(
        threads: &mut Vec<ThreadConfig<NullableIdx>>,
        cpuset: hwloc2::CpuSet,
    ) -> Vec<ReducedChild> {
        let mut children = Vec::with_capacity(usize::try_from(cpuset.weight()).unwrap());
        for cpu in cpuset {
            children.push(ReducedChild::Thread(threads.len()));
            threads.push(ThreadConfig::new(cpu));
        }
        return children;
    }

    /// Set up a subtree of ReductionNodes to reduce results from N children
    ///
    /// Return the index of the root of the subtree (which must later be bound
    /// to a parent node if it is not the root of the whole ReductionTree) and
    /// a list of parent indices to which children can be bound.
    ///
    fn add_nodes(
        &mut self,
        max_arity: u32,
        num_children: u32,
    ) -> (usize, impl IntoIterator<Item = usize>) {
        debug_assert!(max_arity >= Self::MIN_ARITY);

        // Create a root node with as many children as needed and possible
        let first_batch = num_children.min(max_arity);
        let root_idx = self.nodes.len();
        self.nodes.push(CachePadded::new(ReductionNode::new(
            NullableIdx::none(),
            first_batch,
        )));

        // If that does not suffice, we will add more children one by
        // one across nodes that can host them, fanning them out in a
        // round robin fashion using the following state that tracks
        // which nodes can still accept new children.
        let mut parent_start = root_idx;
        let mut parent_idx = parent_start;

        // Once all nodes are filled to capacity, we start turning
        // direct children into subtrees in a breadth-first and
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
        // TODO: To make this method more straightforward, create a state object
        //       for the iteration, with one method to allocate one extra child
        //       and another method to collect the final list of children slots.
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
                    .skip(parent_start)
                    .all(|node| node.num_children == max_arity));
            }

            // Once all tree nodes have the maximal amount of children
            // allowed by arity, it's time to create a new sub-tree. We make it
            // fully filled initially so that we can handle it homogeneously
            // with respect to other existing subtrees affected by this event.
            self.nodes.push(CachePadded::new(ReductionNode::new(
                NullableIdx::some(subtree_parent_idx),
                max_arity,
            )));

            // Creating a subtree replaces an existing child, which must be
            // reallocated to it, and it also hosts the newly allocated child.
            // This leaves extra room for (max_arity - 2) more children.
            let new_children = max_arity - 2;
            if new_children > 0 {
                // Distribute those "holes" among the most recently created tree
                // nodes, with priority given to newer nodes further away from
                // the root (we fill the tree top to bottom and left to right)
                let num_hosts = (self.nodes.len() - root_idx).min(new_children as usize);
                let new_children_share = new_children / num_hosts as u32;
                let extra_new_children = new_children as usize % num_hosts;
                //
                for (rev_idx, node) in self.nodes.iter_mut().rev().take(num_hosts).enumerate() {
                    node.num_children -= new_children_share + (rev_idx < extra_new_children) as u32;
                }

                // Update iteration state so that next time we will add
                // children to the earliest added node with least children.
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

        // At this point, our tree of ReductionNodes is built.
        // Configure the associated Reducers for the right number of children.
        for node in &mut self.nodes[root_idx..] {
            node.reset();
        }

        // Now it's time to determine where children will bind in that tree.
        let mut parent_capacity = Vec::with_capacity(self.nodes.len() - subtree_parent_start);
        //
        // Below subtree_parent_start, all nodes are full of subtree
        // children, with no room for threads & friends, so we ignore those nodes.
        for parent_idx in subtree_parent_start..subtree_parent_end {
            // Between subtree_parent_start and subtree_parent_end, we were in
            // the process of turning children into subtrees but not done yet,
            // so we must discriminate direct chidren from subtrees.
            let num_children = self.nodes[parent_idx].num_children;
            let num_subtrees =
                max_arity - subtree_capacity + (parent_idx < subtree_parent_idx) as u32;
            debug_assert!(num_children >= num_subtrees);
            let capacity = num_children - num_subtrees;
            parent_capacity.push((parent_idx, capacity));
        }
        for parent_idx in subtree_parent_end..self.nodes.len() {
            // Above subtree_parent_end, there are no subtrees yet, so every
            // child of an ReductionNode is a direct child.
            let capacity = self.nodes[parent_idx].num_children;
            parent_capacity.push((parent_idx, capacity));
        }
        //
        // Translate these N-ary binding slots into unary ones for caller convenience
        let children_slots = parent_capacity
            .into_iter()
            .flat_map(|(parent_idx, capacity)| {
                std::iter::repeat(parent_idx).take(capacity as usize)
            });
        debug_assert_eq!(
            u32::try_from(children_slots.clone().count()).unwrap(),
            num_children
        );

        (root_idx, children_slots)
    }

    /// Validate that the output tree is consistent with constructor parameters
    fn validate(
        &self,
        topology: &hwloc2::Topology,
        max_arity: u32,
        threads: &Vec<ThreadConfig<usize>>,
    ) {
        let mut children = vec![vec![]; self.nodes.len()];
        let mut full_cpuset = hwloc2::CpuSet::new();

        // Check that ThreadConfigs follow expectations
        for (idx, thread) in threads.into_iter().enumerate() {
            // Check that each thread targets a single, unique CPU
            assert_eq!(thread.cpuset.weight(), 1);
            let cpu = u32::try_from(thread.cpuset.first()).unwrap();
            assert!(!full_cpuset.is_set(cpu));
            full_cpuset.set(cpu);

            // Check that each thread reduces into a valid ReductionNode
            assert!(thread.accumulator_id < self.nodes.len());
            children[thread.accumulator_id].push(ReducedChild::Thread(idx));
        }

        // Check that threads cover all available CPUs
        assert_eq!(full_cpuset, topology.object_at_root().cpuset().unwrap());

        // Check that a root node is present
        assert_ne!(self.nodes.len(), 0);
        let root_idx = self.root_idx.value().unwrap();

        // Check that ReductionNodes follow expectations
        for (idx, node) in self.nodes.iter().enumerate() {
            // Check that inner Reducer has been reset, hopefully correctly
            assert!(node.has_remaining_threads());

            // Check that each node reduces into a valid target
            if idx == root_idx {
                assert!(node.parent_idx == NullableIdx::none());
            } else {
                let parent_idx = node.parent_idx.value().unwrap();
                assert!(parent_idx < self.nodes.len());
                children[parent_idx].push(ReducedChild::Node(idx));
            }

            // Check observed arities
            assert!(node.num_children > 0);
            assert!(node.num_children <= max_arity);
        }

        // Compare node child counts to actual number of attached children
        for (node, node_children) in self.nodes.iter().zip(children.iter()) {
            assert_eq!(node.num_children, node_children.len() as u32);
        }

        // Traverse the tree from the root, make sure all nodes are reachable,
        // and measure the tree depth along the way.
        let mut reached = 0;
        let mut depth = 0;
        let mut next_nodes = vec![root_idx];
        while !next_nodes.is_empty() {
            depth += 1;
            for next_node in std::mem::take(&mut next_nodes) {
                reached += 1;
                for child in children[next_node].iter().copied() {
                    match child {
                        ReducedChild::Node(idx) => next_nodes.push(idx),
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
            children: &Vec<Vec<ReducedChild>>,
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
                    ReducedChild::Thread(thread_idx) => {
                        write!(&mut lines[child_line], "T{thread_idx} ").unwrap();
                    }
                    ReducedChild::Node(child_idx) => {
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
        print_tree(&mut lines, &children, root_idx, 0);
        for line in lines {
            println!("{line}");
        }
    }

    /// Minimum tree arity needed for reduction to be possible
    const MIN_ARITY: u32 = 2;

    /// Shared access to the root node
    fn root(&self) -> &ReductionNode<R> {
        &self.nodes[self.root_idx.value().unwrap()]
    }

    /// Current number of allocated nodes
    fn num_nodes(&self) -> u32 {
        u32::try_from(self.nodes.len()).unwrap()
    }
}
//
impl<R: Default + Reducer<AccumulatorId = ()>> Reducer for ReductionTree<R> {
    // For scalability reasons, this only resets the root node. Lower-level
    // ReductionNodes are instead reset in `thread_done()` by the last child of
    // the node of interest to merge results.
    fn reset(&self, _num_threads: u32) {
        self.root().reset()
    }

    fn has_remaining_threads(&self) -> bool {
        self.root().has_remaining_threads()
    }

    type Contribution = R::Contribution;

    fn current_result(&self) -> R::Contribution {
        self.root().current_result()
    }

    type AccumulatorId = usize;

    fn thread_done(
        &self,
        result: R::Contribution,
        ordering: atomic::Ordering,
        node_idx: usize,
    ) -> Option<R::Contribution> {
        // Add our result into the target node
        let node = &self.nodes[node_idx];
        let node_result = node.thread_done(result, ordering)?;

        // We're done with this node. Was it the root node?
        if node_idx != self.root_idx.value().unwrap() {
            // If not, reset this node and propagate its results to the parent
            node.reset();
            self.thread_done(node_result, ordering, node.parent_idx.value().unwrap())
        } else {
            // Otherwise, the full reduction is done
            Some(node_result)
        }
    }
}
//
impl<R: Default + Reducer<AccumulatorId = ()>> ReductionNode<R> {
    /// Create a new node
    fn new(parent_idx: NullableIdx, num_children: u32) -> Self {
        Self {
            reducer: R::default(),
            parent_idx,
            num_children,
        }
    }

    /// Prepare this node for the next accumulation cycle
    fn reset(&self) {
        self.reducer.reset(self.num_children)
    }

    /// Check if all children of this node are done accumulating
    fn has_remaining_threads(&self) -> bool {
        self.reducer.has_remaining_threads()
    }

    /// Check out the current sum of contributions for this node
    fn current_result(&self) -> R::Contribution {
        self.reducer.current_result()
    }

    /// Notify that one child of this node is done, aggregate result
    fn thread_done(
        &self,
        result: R::Contribution,
        ordering: atomic::Ordering,
    ) -> Option<R::Contribution> {
        self.reducer.thread_done(result, ordering, ())
    }

    /// Make a freshly reset copy of this node with the reducer type changed
    fn rebind<R2: Default + Reducer>(&self) -> ReductionNode<R2> {
        let reducer = R2::default();
        reducer.reset(self.num_children);
        ReductionNode {
            reducer,
            parent_idx: self.parent_idx,
            num_children: self.num_children,
        }
    }
}

/// Child of a ReductionNode
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ReducedChild {
    /// Leaf CPU thread
    Thread(usize),

    /// Other ReductionNode
    Node(usize),
}
//
impl ReducedChild {
    /// Rebind this child to a different parent node
    fn set_parent<R>(
        &self,
        tree: &mut ReductionTree<R>,
        threads: &mut Vec<ThreadConfig<NullableIdx>>,
        parent_idx: NullableIdx,
    ) {
        match self {
            Self::Thread(idx) => {
                threads[*idx].accumulator_id = parent_idx;
            }
            Self::Node(idx) => {
                tree.nodes[*idx].parent_idx = parent_idx;
            }
        }
    }
}

/// ReductionNode index with a null state
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct NullableIdx(Option<NonZeroUsize>);
//
impl NullableIdx {
    /// Non-null state
    fn some(x: usize) -> Self {
        Self(NonZeroUsize::new(
            x.checked_add(1)
                .expect("usize::MAX is not a supported index"),
        ))
    }

    /// Null state
    fn none() -> Self {
        Self(None)
    }

    /// Check payload
    fn value(&self) -> Option<usize> {
        self.0.map(|x| usize::from(x) - 1)
    }
}

/// Assignment of worker threads to CPUs and reducer accumulators
#[derive(Clone, Debug, PartialEq)]
pub struct ThreadConfig<AccumulatorId> {
    /// CpuSet to be used when pinning this thread to its target CPU
    cpuset: hwloc2::CpuSet,

    /// Accumulator to be used when merging results
    accumulator_id: AccumulatorId,
}
//
impl<AccumulatorId: Default> ThreadConfig<AccumulatorId> {
    /// Set up a thread config for a certain CPU index
    pub fn new(cpu: u32) -> Self {
        Self {
            cpuset: hwloc2::CpuSet::from(cpu),
            accumulator_id: AccumulatorId::default(),
        }
    }

    /// Perform some operation on the inner accumulator ID
    pub fn map_id<NewId>(self, new_id: impl FnOnce(AccumulatorId) -> NewId) -> ThreadConfig<NewId> {
        ThreadConfig {
            cpuset: self.cpuset,
            accumulator_id: new_id(self.accumulator_id),
        }
    }

    /// Bind the active thread to this CPU, get the accumulator ID
    pub fn bind_this_thread(self, topology: &mut hwloc2::Topology) -> AccumulatorId {
        topology
            .set_cpubind(self.cpuset, hwloc2::CpuBindFlags::CPUBIND_THREAD)
            .unwrap();
        self.accumulator_id
    }
}

// TODO: Also do a reduction tree for FutexScheduler, then see if I can find
//       some commonalities.
// FIXME: Roll out a new ScalableThreadPool that binds all worker threads and
//        the main thread. Make it !Send so that the main thread binding can
//        be assumed to remain valid after ThreadPool::new()
// TODO: Oh and brind the AccumulatorId to JobScheduler too
// TODO: Update multithreading-pool.md according to visible design changes.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thread::pool::BasicResultReducer;
    use hwloc2::Topology;

    #[test]
    fn aggregator_tree() {
        let topology = Topology::new().unwrap();
        for max_arity in 2..=(std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2) as u32
            + 1)
        {
            // Create a ReductionTree, rely on debug assertions to check it
            ReductionTree::<BasicResultReducer>::new(&topology, max_arity);
        }
    }
}
