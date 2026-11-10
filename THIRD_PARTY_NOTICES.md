# Third-party source inventory

* **Sakhalin Ai**, `ikonushok/lidar-metro-obstacle-detection`, revision
  `3cfd31843b911f278c08f49e53a09b1a5efa1d92`: original 17-tree JSON and C++
  include, with an adapted Python evaluator. The original MIT license and
  file hashes are in `gabarit/vendor/sakhalin_ai/`.
* **Attention Please**, `BelousovArts/fod`, revision
  `baebf57a4169697aa61d017fcea8e2374ef135a0`: FarBevNet, residual block and
  BEV raster/geometry functions. Attribution, original file hashes and local
  changes are in `gabarit/vendor/attention_please/NOTICE.md`. No upstream
  license is declared at that revision; no MIT license is asserted for these
  files. Weights are fetched separately with a pinned identity and are not
  committed to this repository.

These attributions apply to the named third-party files, independently of the
license of the surrounding application. The banded numerical solver is a new
implementation of the standard LDL and selected-inverse recurrences.

The viewer additionally adapts `pairClosestToDistance`,
`sideAxisLabelPosition` and `axisDistanceLabels` from Sakhalin Ai's
`web/stage_4_cpu_player_axis_labels.js` at the same pinned revision.
They are in `web/static/js/winner_view_geometry.js`, under the original MIT
license (`gabarit/vendor/sakhalin_ai/LICENSE`). The original copyright and
source reference are preserved in the JavaScript header.
