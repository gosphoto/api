# bangs_scope

Узкий smoke-скоп на чёлку (2 фото). Не заменяет `bangs_gate/` (≥30).

| File | Expect | Source |
|------|--------|--------|
| `with_bangs_img4761.jpg` | reject `bangs_on_brows` | pair `20261119T141414Z_IMG_4761` / result `0193d103768c22965e24cee7aa00e4fe` (in.webp) |
| `no_bangs_bgsub.jpg` | not `bangs_on_brows` | pair `20261119T100754Z_BgSub_DSCF7930` / result `4e49bde89d8638859d0a358cf314b062` (in.jpg) |

Gate: BiSeNet hair∩oval — `with_bangs_img4761` → `bangs_on_brows`; `no_bangs_bgsub` проходит.
