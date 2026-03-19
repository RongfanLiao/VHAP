# FlameHead — Code Explanation

## What it is

`FlameHead` is a `nn.Module` that wraps the **FLAME** (Faces Learned with an Articulated Model and Expressions) parametric head model. Given FLAME parameters as input, it outputs a deformed 3D mesh and facial landmarks — differentiably, so gradients flow through it during optimization.

---

## Initialization: what gets loaded

| Buffer | Source | Description |
|---|---|---|
| `v_template` | `flame2023.pkl` | Neutral rest-pose vertices (~5023 verts) |
| `shapedirs` | pkl | PCA blend shapes for identity + expression |
| `posedirs` | pkl | Pose-corrective blend shapes |
| `J_regressor` | pkl | Regresses joint positions from vertices |
| `parents` | pkl | Kinematic tree (neck→head→jaw chain) |
| `lbs_weights` | pkl | Per-vertex skinning weights |
| `full_lmk_faces_idx/bary_coords` | `.npy` | Barycentric landmark embeddings (68+eyes) |
| `faces`, `verts_uvs`, `face_uvcoords` | `.obj` | Mesh topology + UV layout |
| `laplacian_matrix` | computed | For mesh smoothness regularization |

---

## Optional mesh surgery at init time

VHAP extends the base FLAME mesh with several optional modifications:

| Method | What it does |
|---|---|
| `add_teeth()` | Procedurally constructs 120 tooth vertices from lip positions, adds faces/UVs, sets skinning weights (upper teeth → neck, lower teeth → jaw) |
| `connect_lip_inside()` | Adds faces to close the gap inside the lips |
| `remove_lip_inside()` | Removes inner lip faces (reduces artifacts) |
| `remove_torso()` | Removes boundary/neck faces |
| `disable_deformation_on_torso()` | Zeros out expression blend shapes on neck/torso to prevent unnatural deformation |

---

## `forward()` — the LBS pipeline

$$\text{vertices} = \text{LBS}(\underbrace{v_\text{template} + B_s(\beta) + B_e(\psi)}_{\text{shaped}} + \delta_\text{static} + \delta_\text{dynamic},\ \theta)$$

In code steps:

1. **Blend shapes**: `v_shaped = v_template + shapedirs @ [shape, expr]`
2. **Personal offsets**: add `static_offset` and/or `dynamic_offset` (VHAP's key extension)
3. **LBS**: rotate/skin vertices using `posedirs`, `J_regressor`, `lbs_weights` and pose `[rotation, neck, jaw, eyes]`
4. **Translation**: global head translation
5. **Landmarks**: interpolated via barycentric coordinates on the deformed mesh

---

## Supporting modules

| Class | Role |
|---|---|
| `FlameMask` | Vertex/face region masks (skin, hair, lips, teeth, etc.) used to selectively apply losses |
| `FlameTexPCA` | PCA texture space (200 components) from FLAME texture model |
| `FlameTexPainted` | Fixed hand-painted texture map for initialization |
| `BufferContainer` | Utility `nn.Module` for named buffer collections |
