# Source and rights notice

The adapted files in this directory originate from the Attention Please team's
public repository <https://github.com/BelousovArts/fod>, revision
`baebf57a4169697aa61d017fcea8e2374ef135a0`.

**No LICENSE file or license declaration was found in the source tree at that
revision. Public availability does not grant an MIT or other open-source
license. Upstream rights remain with their authors.** No permissive license is
asserted here. Verify permission with the authors before redistributing these
adaptations or their weights. The weights are downloaded separately and are
not included in this source tree.

| Local file | Original file | Original Git blob SHA-1 | Original SHA-256 |
| --- | --- | --- | --- |
| geometry.py | fod/far_bev.py (constants and NumPy geometry/raster functions only) | e590e2db4ceaa960381523e7a525229b9a2d60be | 34a8843f4e26782a9da551fe34331c5ff44e0c4aace986a5dc3477801008d4ef |
| blocks.py | fod/rail_seg_net.py (ResBlock only) | 66c66a5b5c2a4fd16c6c9c2a101fe8854defa1ef | 0a8578e1b6a432333c3dc2a2d260c9dafbee3c79ff0fa0eaa9ca1359f1504991 |
| far_bev_net.py | fod/far_bev_net.py | 1f4edd93733a12102d04824f7822f85d75ae1885 | 6b536bbbd9023184801c6b4e5dd72b5f4bd555c00c0e0c0941fe5c71d1c1c627 |

Local changes: import paths point into this isolated package; checkpoint loading
uses `torch.load(weights_only=True)` rather than unrestricted pickle loading;
CUDA autocast is disabled on hardware without BF16 support. Geometry functions
preserve the source's channel encoding, bin boundaries and coordinate sign.
The host adapter supplies its own frame conversion, confidence gates and
dependency fallback. Original source docstrings are retained.

Weight: `models/far_bev.pt`, Git blob SHA-1
`f0b5bce8c84d9b48044cdd55ee633f0fc1778081`, 20,058,363 bytes.
`scripts/fetch_path_models.py` verifies this immutable identity before use and
writes a local SHA-256 manifest alongside the downloaded checkpoint.
