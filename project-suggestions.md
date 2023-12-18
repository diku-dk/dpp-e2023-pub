# Project Suggestions

These projects are intentionally open-ended.  They are intended as an
opportunity for you to showcase your mastery of the learning goals:

* Parallel algorithmic reasoning.

* Parallel cost models.

* Judging the suitability of the language/tool for the problem at
  hand.

* Applied data-parallel programming.

This means that you are free to diverge from the project descriptions
below, or come up with your own ideas, as long as they provide a
context in which you can demonstrate the course contents.

You are *not* judged on whether e.g. Futhark or ISPC or whatever
language you choose happens to be a good fit or run particularly fast
for whatever problem you end up picking, but you *are* judged on how
you evaluate its suitability.

## Porting PBBS benchmarks

The [Problem Based Benchmark
Suite](https://cmuparlay.github.io/pbbsbench/) is a collection of
benchmark programs written in parallel C++. We are interested in
porting them to a high-level parallel language (e.g. Futhark). Some of
the benchmarks are relatively trivial; others are more difficult. It
might be a good idea for a project to combine a trivial benchmark with
a more complex one. The [list of benchmarks is
here](https://cmuparlay.github.io/pbbsbench/benchmarks/index.html).
The ones listed as *Basic Building Blocks* are all pretty
straightforward. Look at the others and pick whatever looks
interesting (but talk to us first - some, e.g. rayCast, involve no
interesting parallelism, and so are not a good DPP project).
Particularly interesting to Troels are the ones related to
computational geometry:

* [delaunayRefine](https://cmuparlay.github.io/pbbsbench/benchmarks/delaunayRefine.html)
* [delaunayTriangulation](https://cmuparlay.github.io/pbbsbench/benchmarks/delaunayTriangulation.html)
* [rangeQuery2d](https://cmuparlay.github.io/pbbsbench/benchmarks/rangeQuery2d.html)

## Engineering Parallel Semisort

The paper [High-Performance and Flexible Parallel Algorithms for
Semisort and Related
Problems](https://dl.acm.org/doi/pdf/10.1145/3558481.3591071)
describes an algorithm for *semisorting*, where elements of a sequence
with the same *key* are made contiguous, but no ordering is given
between elements of different keys.  E.g. for an input

```
[1, 3, 2, 0, 3, 3, 3, 3, 0, 2]
```

the following might be the result of semisorting

```
[1, 3, 3, 3, 3, 3, 0, 0, 2, 2]
[2, 2, 1, 3, 3, 3, 3, 3, 0, 0]
...
```

Semisorting is a useful primitive in various parallel algorithms.
While the paper listed above describes the algorithm in a fork-join
manner, Troels finds the explanation both beautiful and simple, *and*
probably not too difficult to adapt to a data parallel setting. This
project is about implementing the semisort algorithm in Futhark or
ISPC (or some other data parallel languages) and evaluating the
performance.

## Big-Integer Arithmetic in Futhark

Couple of years ago, a [group of students have tackled the task of implemnting addition, multiplication and even division of big integers in Futhark](group-projects/big-num-arith/Topalovici-Nielsen-Oleson-dpp-proj.pdf).

However, there is still significant room left for improving the GPU performance of such operations; see for example [some results we have reported recently in this presentation](group-projects/big-num-arith/pres-mycroftfest-bignum.pdf). The presentation also includes references to related work.

The task of this project is to reproduce or to improve on the reported results. We suggest you start with the linked DPP project and then think on how to rewrite the implementation of the target operations in a way that improves the GPU performance. Then you should probably discuss your ideas with us, and finally go ahead and strive to derive a very efficient implementation. 

Finally, we advise you stick with addition and multiplication until you derive efficient implementations, and only if time remains you move to more complex operations such as division.

## Project Related to Automatic Differentiation

[Minpack-2 benchmark](group-projects/Mpack-2/Minpack-2.pdf) consists of a collection of problems that require computation of derivatives. The implementation language is Fortran, and each problem implementation has options for computing the primal (original program), or/and the associated Jacobian (or even Hessian). 

This task refers to porting one (or several) of the Minpack-2 benchmarks to Futhark: you need to translate only "the primal" (i.e., the original function that requires differentiation), and then you may use Futhark's support for automatic differentiation to compute the dense Jacobian/Hessians.

Many of the Minpack-2 primals result in sparse Jacobians or Hessians (i.e., the second-order derivative); hence the last step is to visualize/characterize the sparsity of the differentiated code. [Here is a paper that shows the sparsity of a several applications from Minpack-2](group-projects/Mpack-2/Efficient_Computation_of_Gradients_and_Jacobians_b.pdf)

Bonus: if time permits, you may try to optimize the computation, e.g., by packing in a safe way several unit vectors into a denser representation that contains several one entries.

## Batched Rank-k Search

For those that have not chosen this problem in the PMPH course, you are welcome to solve [the rank-k search problem](group-projects/rank-search-k/Project-RankSearch-k.pdf).
