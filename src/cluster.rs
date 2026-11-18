//! Cell hierarchy and nested groups (SPEC 7.1, D6).
//!
//! `bonsai_rs::tree::linkage::linkage_tree` builds the kNN-Ward dendrogram on
//! the smoothed bins but sets every branch to one, so the Ward heights are
//! gone. The k-cut is therefore done top-down here: repeatedly split the
//! current piece whose split costs most, where the cost of a node is the
//! between-children sum of squares `sum_i n_i ||c_i - c||^2`. On a binary node
//! that is exactly the Ward merge cost, so on a monotone Ward tree this is the
//! height cut. The root of a linkage tree has three children, so a split can
//! add two pieces at once and the cut may end with `k + 1` clusters.
//!
//! Costs are evaluated lazily over each candidate's leaves, `O(n p)` per
//! evaluated node, so nothing of size `n_nodes x p` is stored.

use bonsai_rs::tree::Tree;
use bonsai_rs::tree::linkage::{LinkageParams, linkage_tree};
use rayon::prelude::*;

use crate::errors::CnvErrors;
use crate::utils::traits::{CnvFloat, wide};

/// One group of cells for a pseudobulk.
#[derive(Clone, Debug)]
pub struct Group {
    /// Cell indices (rows of the caller's count matrix), ascending
    pub cells: Vec<u32>,
    /// `Some(id)` for a leaf cluster of the k-cut, `None` for an internal node
    pub leaf: Option<u32>,
}

/// Leaves (rows of the clustered matrix) below a node.
///
/// ### Params
///
/// * `tree` - The dendrogram
/// * `node` - Node to collect under
///
/// ### Returns
///
/// Leaf indices, ascending.
fn leaves_under(tree: &Tree, node: u32) -> Vec<u32> {
    let n_leaves = tree.n_leaves() as u32;
    let mut out = Vec::new();
    let mut stack = vec![node];
    while let Some(u) = stack.pop() {
        if u < n_leaves {
            out.push(u);
        } else {
            stack.extend_from_slice(tree.children(u));
        }
    }
    out.sort_unstable();
    out
}

/// Between-children sum of squares of a node.
///
/// ### Params
///
/// * `tree` - The dendrogram
/// * `node` - An internal node
/// * `data` - Clustered matrix, row-major `[leaf][feature]`
/// * `p` - Features per row
///
/// ### Returns
///
/// `sum_i n_i ||c_i - c||^2` over the node's children, in `f64`.
fn split_cost<T: CnvFloat>(tree: &Tree, node: u32, data: &[T], p: usize) -> f64 {
    let parts: Vec<(f64, Vec<f64>)> = tree
        .children(node)
        .par_iter()
        .map(|&ch| {
            let leaves = leaves_under(tree, ch);
            let mut sum = vec![0.0f64; p];
            for &l in &leaves {
                let row = &data[l as usize * p..(l as usize + 1) * p];
                for (s, &x) in sum.iter_mut().zip(row) {
                    *s += wide(x);
                }
            }
            let n = leaves.len() as f64;
            sum.iter_mut().for_each(|s| *s /= n);
            (n, sum)
        })
        .collect();
    let n_tot: f64 = parts.iter().map(|(n, _)| n).sum();
    let mut centre = vec![0.0f64; p];
    for (n, c) in &parts {
        for (m, &x) in centre.iter_mut().zip(c) {
            *m += n * x / n_tot;
        }
    }
    parts
        .iter()
        .map(|(n, c)| {
            n * c
                .iter()
                .zip(&centre)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
        })
        .sum()
}

/// Nested groups from a k-cut of the kNN-Ward tree (SPEC 7.1).
///
/// Groups follow numbat's `get_nodes_celltree`: every node split by the cut
/// (the root first, all cells) and the final clusters as leaves. With a binary
/// tree that is `2k - 1` groups. No size filter here; the pipeline drops
/// groups with `<= min_cells` cells.
///
/// ### Params
///
/// * `data` - Smoothed bins of the cells to cluster, row-major `[cell][bin]`
/// * `cells` - Caller cell index of each row
/// * `n_bins` - Bins per row
/// * `k` - Wanted number of leaf clusters, `>= 1`
/// * `params` - Linkage knobs, `None` for bonsai's defaults
///
/// ### Returns
///
/// Internal groups in split order, then leaf groups numbered from 0, or a
/// shape or linkage error.
pub fn ward_groups<T: CnvFloat>(
    data: &[T],
    cells: &[u32],
    n_bins: usize,
    k: usize,
    params: Option<LinkageParams>,
) -> Result<Vec<Group>, CnvErrors> {
    let n = cells.len();
    if data.len() != n * n_bins {
        return Err(CnvErrors::ShapeMismatch {
            what: "clustered bins",
            expected: n * n_bins,
            got: data.len(),
        });
    }
    if n == 0 {
        return Err(CnvErrors::EmptyInput { what: "cells" });
    }
    if k == 0 {
        return Err(CnvErrors::InvalidParameter {
            name: "init_k",
            value: k.to_string(),
            expected: ">= 1",
        });
    }
    let to_group = |leaves: Vec<u32>, leaf: Option<u32>| {
        let mut cells: Vec<u32> = leaves.iter().map(|&l| cells[l as usize]).collect();
        cells.sort_unstable();
        Group { cells, leaf }
    };
    if n == 1 || k == 1 {
        return Ok(vec![to_group((0..n as u32).collect(), Some(0))]);
    }

    let tree = linkage_tree(data, n, n_bins, params)?;
    let n_leaves = tree.n_leaves() as u32;
    // (cost, node) of the current pieces; leaves of the tree cost -inf
    let cost = |u: u32| {
        if u < n_leaves {
            f64::NEG_INFINITY
        } else {
            split_cost(&tree, u, data, n_bins)
        }
    };
    let mut pieces: Vec<(f64, u32)> = vec![(cost(tree.root()), tree.root())];
    let mut split = Vec::new();
    while pieces.len() < k {
        // highest cost, ties to the smaller node index
        let (i, &(c, u)) = pieces
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.0.total_cmp(&b.1.0).then(b.1.1.cmp(&a.1.1)))
            .expect("pieces is never empty");
        if c == f64::NEG_INFINITY {
            break;
        }
        pieces.swap_remove(i);
        split.push(u);
        for &ch in tree.children(u) {
            pieces.push((cost(ch), ch));
        }
    }
    // leaf numbering by smallest member, so it depends on the tree alone
    let mut leaves: Vec<Vec<u32>> = pieces
        .iter()
        .map(|&(_, u)| leaves_under(&tree, u))
        .collect();
    leaves.sort_unstable_by_key(|l| l[0]);

    let mut out: Vec<Group> = split
        .iter()
        .map(|&u| to_group(leaves_under(&tree, u), None))
        .collect();
    out.extend(
        leaves
            .into_iter()
            .enumerate()
            .map(|(i, l)| to_group(l, Some(i as u32))),
    );
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Three well separated blobs of `m` cells each in 4 features.
    fn blobs(m: usize) -> Vec<f64> {
        let mut x = Vec::new();
        for b in 0..3 {
            for i in 0..m {
                for f in 0..4 {
                    let jitter = ((i * 7 + f * 3) % 11) as f64 * 0.01;
                    x.push(if f == b { 10.0 } else { 0.0 } + jitter);
                }
            }
        }
        x
    }

    #[test]
    fn test_ward_groups_recovers_three_blobs() {
        let m = 40;
        let x = blobs(m);
        let cells: Vec<u32> = (0..3 * m as u32).collect();
        let g = ward_groups(&x, &cells, 4, 3, None).unwrap();
        let leaves: Vec<&Group> = g.iter().filter(|g| g.leaf.is_some()).collect();
        assert_eq!(leaves.len(), 3);
        for (b, leaf) in leaves.iter().enumerate() {
            let want: Vec<u32> = (b * m..(b + 1) * m).map(|c| c as u32).collect();
            assert_eq!(leaf.cells, want);
        }
        // the root is the first internal group and holds every cell
        assert_eq!(g[0].leaf, None);
        assert_eq!(g[0].cells.len(), 3 * m);
    }

    #[test]
    fn test_ward_groups_k1_single_group() {
        let x = blobs(5);
        let cells: Vec<u32> = (100..115).collect();
        let g = ward_groups(&x, &cells, 4, 1, None).unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].cells, cells);
    }

    #[test]
    fn test_split_cost_matches_ward_on_two_points() {
        // two leaves at 0 and 2 in one feature: n_a n_b / (n_a + n_b) * d^2 = 2
        let tree = Tree::from_parents(vec![2, 2, u32::MAX], vec![1.0; 3], 2).unwrap();
        let c = split_cost(&tree, 2, &[0.0f64, 2.0], 1);
        approx::assert_relative_eq!(c, 2.0, max_relative = 1e-14);
    }
}
