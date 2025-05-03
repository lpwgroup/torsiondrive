# Changelog

<!--
## vX.Y.0 / 2025-MM-DD (Unreleased)

#### Breaking Changes

#### New Features
 * [\#NN](https://github.com/lpwgroup/torsiondrive/pull/NN) 

#### Enhancements

#### Bug Fixes

#### Misc.
-->


## v1.2.0 / 2025-05-03

#### New Features
 * [\#74](https://github.com/lpwgroup/torsiondrive/pull/74) Allow TorsionDrive to use xTB's native
   optimizer (rather than exclusively geomeTRIC) when xTB is the QC engine. @xiki-tempula
 * [\#76](https://github.com/lpwgroup/torsiondrive/pull/76) Add ASE as a QM engine. @jessicaflowers

#### Enhancements
 * [\#71](https://github.com/lpwgroup/torsiondrive/pull/71) Allow either "Optimization completed" or
   "Optimization completed on the basis of negligible forces" to be judged a successful opt. @xiki-tempula
 * [\#78](https://github.com/lpwgroup/torsiondrive/pull/78) Fixes package for Python 3.12. @bennybp
 * [\#80](https://github.com/lpwgroup/torsiondrive/pull/80) Adapt package for Numpy v2. Prefer
   importing `work_queue` from `ndcctools`. Slight adjustments to testing so test suite passes
   cleanly. Add changelog and citation file. @loriab

#### Bug Fixes
 * [\#65](https://github.com/lpwgroup/torsiondrive/pull/65) Fix to allow reading Gaussian input with
   four keywords in the command line. @xiki-tempula
 * [\#69](https://github.com/lpwgroup/torsiondrive/pull/69) Fix the Gaussian 2D scan to read multiple
   constraints. @xiki-tempula


## v1.1.0 / 2021-09-11

#### New Features
 * [\#56](https://github.com/lpwgroup/torsiondrive/pull/56) Includes Gaussian as an optimization
   engine (Josh Horton @JoshHorton)

#### Bug Fixes
 * [\#61](https://github.com/lpwgroup/torsiondrive/pull/61) Restricting the scan range did not work
   properly when the range included -180 degrees (David Dotson @dotsdl)


## v1.0 / 2019-11-04

#### Enhancements
 * only for zenodo, no code changes


## v0.9.8.1 / 2018-08-07

#### Enhancements
 * This release contains updated `measure_dihedrals()` function, that provides better checking for
   linear angles, and runs faster than the original m.measure_dihedrals() function. #49 #52
   Great contribution from @hyejang


## v0.9.8 / 2019-06-19

#### Enhancements
 * [\#51](https://github.com/lpwgroup/torsiondrive/pull/51) This release mainly contains support for
   the new OpenMM engine
 * [\#50](https://github.com/lpwgroup/torsiondrive/pull/50) The cctools installation script is also
   updated to support the new Swig version 4.


## v0.9.7 / 2019-05-30

#### Enhancements
 * [\#45](https://github.com/lpwgroup/torsiondrive/pull/45) documents published on readthedocs
 * [\#47](https://github.com/lpwgroup/torsiondrive/pull/47) Refactoring the torsiondrive.tools
   package including plotting scripts. The components can now be reused in other modules for parsing
   and visualizing scan.xyz
 * [\#48](https://github.com/lpwgroup/torsiondrive/pull/48) Gradients information is now provided
   in the final qdata.txt, then geomeTRIC is used as the optimizer.
 * [\#46](https://github.com/lpwgroup/torsiondrive/pull/46) CI improvements


## v0.9.6 / 2019-04-18

#### Enhancements
 * This release contains the new "energy upper limit" feature, which allows automated scan of
   "limited dihedrals", such as C-C-C-C in benzene rings.
 * The API interface has also been updated, to support the recently developed features that are
   available in CLI, such as dihedral_ranges, energy_decrease_thresh, energy_upper_limit, extra_constraints.
 * To support the above features in API, QCFractal running torsiondrive as a service also needs an
   update, which will be worked on after this release.


## v0.9.5 / 2019-03-26

#### Enhancements
 * Interface with geomeTRIC is updated to 0.9.5
 * More robust "non-converging constrained optimization" handling behavior (skip instead of accept)
 * Improved unit test
 * New version of cctools supporting Python 3.7


## v0.9.4 / 2019-02-17

#### Enhancements
 * [\#31](https://github.com/lpwgroup/torsiondrive/pull/31) This release includes the new feature
   "Dihedral Range Limit".
 * Also, tests are improved to use pytest.fixtures, and the large example files are now pulled from
   a separate repo to avoid increasing space usage of this repo.
 * Also, the interfaces with geomeTRIC and QCEngine are updated to be compatible with their latest versions.


## v0.9.3 / 2019-01-24

#### Bug Fixes
 * This release contains a bug fix. The bug fixed causes error when using grid spacing of 24 or 40.


## v0.9.2 / 2019-01-15

#### Enhancements
 * This release includes a few latest features including "energy thresholds" and "2-D heatmap plots"


## v0.9.1 / 2018-11-01

#### Enhancements
 * Update tests with new geomeTRIC JSON API
 * The version number 0.9.1 is intentional to be consistent with geomeTRIC v0.9.1


## v0.8.3 / 2018-09-04

#### Enhancements
 * This release includes adding MANIFEST.in.


## v0.8.2 / 2018-08-28

#### Enhancements
 * This release covers the renaming from crank to torsiondrive, and adopts standard python module naming.


## v0.8.1 / 2018-08-04

#### Enhancements
 * The JSON API for calling geomeTRIC is updated to be compatible with
   geomeTRIC commit 4876a222a03091fd2323af7c02513676a52c3f31


## v0.8.1-beta / 2018-08-04

#### Enhancements
 * overwrite Examples folder when running tests


## v0.8 / 2018-07-30

#### Enhancements
 * crankAPI switch back to zero-based numbering and updated tests

