# Data Parallel Programming (DPP), Block 2 2023

## Nvidia Acknowledgements

We are grateful to Nvidia for awarding us a teaching grant (for the
PMPH and DPP courses) that consists of two A100 GPUs. These are now
accessible on the server futharkhpa03fl.unicph.domain


## Course Structure

DPP is structured around five weeks with lectures and lab sessions
on Monday and Wednesday, followed by a final project to be
presented orally at the exam.  Throughout the course, you will hand in
four weekly assignments.  These *weeklies* count for 40\% of the
grade, while the exam counts for 60\%.

The teachers are **Cosmin Oancea** and **Troels Henriksen**.

All lectures and lab sessions will be delivered in English.  The
assignments and projects will be posted in English, and while you can  
chose to hand in solutions in either English or Danish, English is
preferred.

All course material is distributed via this GitHub page.  Assignment
handin is still on Absalon.  There is no mandated textbook for the
course - you will be assigned reading material from papers and such.

## Course schedule

[Course Catalog Web Page](https://kurser.ku.dk/course/ndak21006u/2023-2024)

### Lectures (zoom links will be posted on Absalon):

* Monday    13:00 - 15:00 (weeks 47-51, 1-3	in bi - 2-1-07/09, Ole Maaløes Vej 5, Biocenter)
* Wednesday 10:00 - 12:00 (weeks 47-51, 1-3	in aud - NBB 2.0.G.064/070, Jagtvej 155)

### Labs:

* Monday 15:00 - 17:00 (weeks 47-51, 1-3, in bi - 2-1-07/09, Ole Maaløes Vej 5, Biocenter)
* Wednesday 13:00 - 15:00 (weeks 47-51, 1-3, in bi - 2-0-17, Ole Maaløes Vej 5, Biocenter)

[Location of lectures and labs](https://skema.ku.dk/tt/tt.asp?SDB=ku2324&language=EN&folder=Reporting&style=textspreadsheet&type=student+set&idtype=id&id=195908&weeks=1-53&days=1-7&periods=1-68&width=0&height=0&template=SWSCUST+student+set+textspreadsheet)


This course schedule is tentative and will be updated as we go along.
**The schedule below will become the correct one as you enter the
week when the course starts.**

The lab sessions are aimed at providing help for the weeklies and
group project.  Do not assume you can solve them without showing
up to the lab sessions.

### Lecture plan

| Date | Time | Topic | Material |
| --- | --- | --- | --- |
| 20/11 | 13:00-15:00 | [Intro, deterministic parallelism, data parallelism, Futhark](slides/L1-determ-prog.pdf) | [Parallel Programming in Futhark](https://futhark-book.readthedocs.io/en/latest/) | |
| 20/11 | 15:00-17:00 | Lab | [Futhark exercises](bootstrap-exercises.md) | |
| 22/11 | 10:00-12:00 | [Cost models, advanced Futhark](slides/L2-advanced-futhark-cost-models.pdf) | [Guy Blelloch: Programming Parallel Algorithms](material/blelloch-programming-parallel-algorithms.pdf), [Prefix Sums and Their Applications](material/prefix-sums-and-their-applications.pdf), [A Provable Time and Space Efficient Implementation of NESL](material/a-provable-time-and-space-efficient-implementation-of-nesl.pdf) |
| 22/11 | 13:00-15:00 | Lab ([**Assignment 1 handout**](weekly-1/)) | |
| 27/11 | 10:00-12:00 | [Vector programming with ISPC](slides/L3-ispc.pdf) | [ispc: A SPMD Compiler for High-Performance CPU Programming](material/ispc_inpar_2012.pdf) | |
| 27/11 | 15:00-17:00 | Lab | |
| 29/12 | 13:00-15:00 | Lab ([**Assignment 2 handout**](weekly-2/)) | |
| 29/11 | 13:00-15:00 | [Pointer Structures](slides/L4-pointer-structures.pdf) | [The Complexity of Parallel Computations](material/wyllie.pdf) (section 4.1.2), [Aaron Hsu's PhD dissertation](material/hsu_dissertation.pdf) (sections 3.2 and 3.3, but the lecture slides should be enough). |
| 04/12 | 13:00-15:00 | [Full/irregular flattening](slides/L5-irreg-flattening.pdf) | [PMPH lecture notes, Chapter 4](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf), [Transforming High-Level Data-Parallel Programs into Vector Operations](material/flattening/NeslFlatTechPaper.pdf), [Harnessing the Multicores: Nested Data Parallelism in Haskell](material/flattening/harnessing-multicores.pdf) (not easy to read)|
| 04/12 | 15:00-17:00 | Lab | |
| 06/12 | 10:00-12:00 | [Full/irregular flattening](slides/L5-irreg-flattening.pdf) | [PMPH lecture notes, Chapter 4](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf), [Transforming High-Level Data-Parallel Programs into Vector Operations](material/flattening/NeslFlatTechPaper.pdf), [Harnessing the Multicores: Nested Data Parallelism in Haskell](material/flattening/harnessing-multicores.pdf) (not easy to read)|
| 06/12 | 13:00-15:00 | Lab ([**Assignment 3 handout**](weekly-3/)) | |
| 11/12 | 13:00-15:00 | [Polyhedral Analysis](slides/L6-polyhedral.pdf) | [PMPH Dependence Analysis](material/poly/L5-LoopParI.pdf); [Sven Verdoolaege: Presburger Formulas and Polyhedral Compilation (tutorial)](material/poly/polycomp-tutorial.pdf); [Sven Verdoolaege: Presburger Sets and Relations: from High-Level Modelling to Low-Level Implementation (slides)](material/poly/poly-in-detail.pdf), [Code Examples](material/poly/poly-code-egs/) |
| 11/12 | 15:00-17:00 | Lab | [Code Examples](material/poly/poly-code-egs/) |
| 13/12 | 10:00-12:00 | [Regular and incremental flattening](slides/L8-regular-flattening.pdf) | [Futhark: Purely Functional GPU-Programming with Nested Parallelism and In-Place Array Updates](https://futhark-lang.org/publications/pldi17.pdf),  [Incremental Flattening for Nested Data Parallelism](https://futhark-lang.org/publications/ppopp19.pdf) (particularly the latter), **Optional:** [Dataset Sensitive Autotuning of Multi-Versioned Code based on Monotonic Properties](https://futhark-lang.org/publications/tfp21.pdf) |
| 13/12 | 13:00-15:00 | Lab ([**Assignment 4 handout**](weekly-4/)) | |
| 18/12 | 13:00-15:00 | [Data-parallel automatic differentiation (slides)](slides/L9-AD.pdf) | [Automatic Differentiation in Machine Learning: a Survey, Baydin et. al.](material/AD/automatic_differentiation_in_ml_baydin.pdf), [autodiff.fut;](material/AD/autodiff.fut), [AD for an Array Language with Nested Parallelism;](material/AD/ad-sc22.pdf), [Parallelism-Preserving Automatic Differentiation for Second-Order Array Languages](material/AD/PPAD-fhpnc21.pdf), [Reverse-Mode AD of Multi-Reduce and Scan in Futhark](material/AD/ifl23-ad.pdf) |
| 18/12 | 15:00-17:00 | Lab | |
| 20/12 | 10:00-12:00 | [Data-parallel automatic differentiation](slides/L9-AD.pdf) | same material as previous lecture |
| 20/12 | 13:00-15:00 | Lab (with project proposals) | |

After New Years, *maybe* there will be no lectures (we are still thinking on it),
but labs will still be held to help with the group project.

| Date | Time | Topic | Material |
| ---  | ---  | ---   | ---      |
| 03/1 | 10:00-12:00 | No Lecture | |
| 03/1 | 13:00-15:00 | No Lab     | |
| 08/1 | 13:00-15:00 | No Lecture  |  |
| 08/1 | 15:00-17:00 | Group Project | Help with group project |
| 10/1 | 10:00-12:00 | [Halide: A DSL for Image Processing](slides/L10-Halide.pdf) | [Halide PLDI'13](material/halide-pldi13.pdf), [Stencil Fusion Optional Exercises](weekly-optional) |
| 10/1 | 13:00-15:00 | Group Project | Help with group project |
| 15/1 | 13:00-15:00 | No Lecture  |  |
| 15/1 | 15:00-17:00 | Group Project | Help with group project |
| 17/1 | 10:00-12:00 | No Lecture |  |
| 17/1 | 13:00-15:00 | Group Project | Help with group project |
| ---  | --- | ---   | ---        |
| 24/1 |             | Oral Exam  | a group of 3 members will be examined in about 1 hour |
| 25/1 |             | Oral Exam  | exam schedule, i.e., time slots for each group, will be published on Absalon |

## Weekly assignments

The weekly assignments are **mandatory**, must be solved
**individually**, and make up 40% of your final grade.  Submission is
on Absalon.

You will receive feedback a week after the handin deadline (at the
latest).  You then have another week to prepare a resubmission.  That
is, **the resubmission deadline is two weeks after the original handin
deadline**.

The assignment text and handouts will be linked in the schedule above.

## Group project and exam

The final project, along with the exam as a whole, contributes 60% of
your grade, and is done in groups of 1-3 people (although working
alone is strongly discouraged).  We have [a list of project
suggestions](project-suggestions.md), but you are free to suggest your
own (but please talk with us first).  Since the time to work on the
project is rather limited, and there is no possibility of
resubmission, you should ask for help early and often if you are
having trouble making progress.  **The project should be handed in via
Absalon on Friday the 19th of January**.  Send an email if you have
trouble meeting this deadline.

Most of the projects are about writing some parallel program, along
with a report describing the main points and challenges of the
problem.  The exam format is a group presentation followed by
individual questions about both your project **and anything else in
the curriculum**.  Each group prepares a common presentation with
slides, and each member of the group presents non-overlapping parts of
the presentation for about 10 min (or less). Then each member of the
group will answer individual questions for about 10 min.

## Practical information

You may find it useful to make use of DIKUs GPU machines in your work.
We recommend using the so-called [Hendrix
cluster](https://diku-dk.github.io/wiki/slurm-cluster#getting-access).
If you are enrolled in the course, you should already have access.
Otherwise contact Troels at <athas@sigkill.dk>. For how to access
Hendrix, follow the first link in this paragraph.

Consider using
[sshfs](https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh)
to mount the remote file system on your local machine:

```
$ mkdir remote
$ sshfs hendrix:/ remote
```

[Also see here for more
hints.](https://github.com/diku-dk/howto/blob/main/servers.md#the-hendrix-cluster)

### Using Hendrix

The DIKU systems have a [conventional HPC modules
setup](https://hpc-wiki.info/hpc/Modules), meaning you can make
additional software available with the ``module`` command. You may
need to do this inside SLURM jobs.

#### Loading CUDA

```bash
$ module load cuda
```

#### Loading Futhark

```bash
$ module load futhark
```

#### Loading ISPC

```bash
$ module load ispc
```

(Although there is no reason to use Hendrix for ISPC - it will run
fine on your machine.)

## Other resources

You are not expected to read/watch the following unless otherwise
noted, but they contain useful and interesting background information.

* [The Futhark User's Guide](https://futhark.readthedocs.io), in
  particular [Futhark Compared to Other Functional
  Languages](https://futhark.readthedocs.io/en/latest/versus-other-languages.html)

* [Troels' PhD thesis on the Futhark compiler](https://futhark-lang.org/publications/troels-henriksen-phd-thesis.pdf)

* [A library of parallel algorithms in NESL](http://www.cs.cmu.edu/~scandal/nesl/algorithms.html)

* [Functional Parallel Algorithms by Guy Blelloch](https://vimeo.com/showcase/1468571/video/16541324)

* ["Performance Matters" by Emery Berger](https://www.youtube.com/watch?v=r-TLSBdHe1A)

* [The story of `ispc`](https://pharr.org/matt/blog/2018/04/18/ispc-origins.html) (you can skip the stuff about office politics, although it might ultimately be the most valuable part of the story)

* [Scientific Benchmarking of Parallel Computing
  Systems](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf)
  (we benchmark much simpler systems and don't expect anywhere near
  this much detail, but it's useful to have thought about it)
