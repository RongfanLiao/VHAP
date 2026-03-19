# Background Color and Other Factors for FLAME Parameter Quality

## Goal

This note is specifically about getting better FLAME parameter sequences for:

- shape,
- expression,
- pose.

It is not mainly about prettier rendered images or cleaner NeRF exports.

## Short answer

If your goal is better FLAME shape, expression, and pose, `background_color` is usually a minor factor in this codebase.

The stronger levers are:

1. optimization budget,
2. photometric fitting,
3. landmark quality,
4. image resolution and matte quality,
5. offset settings,
6. temporal smoothness weights,
7. camera quality.

For this goal, `background_color=None` is a reasonable default.

## What `background_color` actually does

The dataset applies `background_color` in [../vhap/data/video_dataset.py](../vhap/data/video_dataset.py). It composites the RGB image with the alpha map:

$$
\mathrm{rgb}_{out} = \alpha \cdot \mathrm{rgb}_{fg} + (1-\alpha) \cdot \mathrm{rgb}_{bg}
$$

So:

- `white` replaces transparent regions with white,
- `black` replaces transparent regions with black,
- `None` leaves the RGB unchanged.

This is a data preprocessing choice. It changes the target RGB seen by the tracker.

## Why `background_color` often has little effect here

There are three reasons.

### 1. Landmark fitting does not care about background color

Landmark loss is driven by detected 2D landmark coordinates, not by the RGB background. So if a stage is landmark-dominated, background choice matters very little for shape, expression, and pose.

### 2. The default photometric renderer uses the target image as background

The tracker chooses the render background in [../vhap/model/tracker.py](../vhap/model/tracker.py). By default, `background_train` and `background_eval` are both `target` in [../vhap/config/base.py](../vhap/config/base.py).

That means for non-head pixels, the renderer copies the target image background back into the prediction. For a pure background pixel $p$:

$$
\mathrm{pred}(p) = \mathrm{gt}(p)
$$

So the RGB error becomes:

$$
|\mathrm{gt}(p) - \mathrm{pred}(p)| = 0
$$

This is the main reason changing white to black often does not change tracking much in practice.

### 3. The active photometric loss is mostly RGB-only

The tracker computes RGB loss in [../vhap/model/tracker.py](../vhap/model/tracker.py). The silhouette and background alpha losses are present there but commented out.

So alpha does not currently act as a strong direct supervision signal for the boundary.

## When `background_color` still matters

Even though it is usually a minor factor, it is not completely irrelevant.

It still matters most for:

- hair and jawline boundaries,
- semi-transparent or antialiased edges,
- imperfect alpha mattes,
- strong background clutter near the face boundary,
- downstream tasks that consume the composited RGB directly.

In other words, `background_color` mainly affects edge supervision, not the full-frame FLAME parameter fit.

## White vs black vs none

### `background_color=None`

Best when:

- you care mainly about FLAME shape, expression, and pose,
- your original RGB near the face is already usable,
- you do not want white or black halo artifacts introduced by compositing.

Tradeoff:

- you keep whatever original background content is present.

### `background_color=white`

Best when:

- you want a visually clean, consistent RGB dataset,
- you care about export quality for later pipelines,
- your mattes are good enough that white edges do not create noticeable halos.

Tradeoff:

- can create bright boundary halos around hair and fine structures if the alpha map is imperfect.

### `background_color=black`

Best when:

- you want the same kind of consistency as white,
- your downstream pipeline prefers dark backgrounds,
- your mattes look less objectionable against black.

Tradeoff:

- can create dark halos instead of bright ones.

## Other factors that matter more

### 1. Optimization budget

This is the biggest quality lever in your current wrapper.

In [../../tools/video_to_flame_param.py](../../tools/video_to_flame_param.py), the wrapper uses a debug-level budget:

- `debug_num_steps = 50`,
- `lmk_global_tracking.num_epochs = 1`,
- `rgb_global_tracking.num_epochs = 1`.

That is much smaller than the defaults in [../vhap/config/base.py](../vhap/config/base.py), where initialization stages use 500 steps and global stages use 30 epochs.

If you want better FLAME sequences, increasing the optimization budget is likely the single highest-value change.

### 2. Photometric fitting

Photometric fitting is enabled by default through `ExperimentConfig.photometric = True` in [../vhap/config/base.py](../vhap/config/base.py).

For parameter quality:

- landmark-only fitting is faster but coarser,
- photometric fitting usually improves expression detail, pose refinement, and temporal consistency.

If you care about better shape, expression, and pose, keep photometric fitting enabled unless you have a specific reason not to.

### 3. Landmark quality

Landmarks strongly affect:

- rigid initialization,
- identity shape initialization,
- early pose and expression estimation.

If landmarks are unstable, the tracker starts from a bad place and later photometric fitting has more work to do. Stable landmarks matter more than white vs black background.

### 4. Matte quality and boundary quality

Even when `background_color` itself is minor, the quality of the alpha matte still matters. Poor mattes can distort:

- hairline edges,
- cheeks and jaw boundaries,
- neck silhouette,
- photometric supervision near the contour.

Good mattes help. Bad mattes can harm. Forced white or black backgrounds can make matte errors more visible.

### 5. Static and dynamic offsets

The defaults in [../vhap/config/base.py](../vhap/config/base.py) are:

- `use_static_offset = True`,
- `use_dynamic_offset = False`.

This is a sensible starting point for FLAME parameter quality.

- Static offset helps fit subject-specific geometry.
- Dynamic offset can improve alignment, but it can also absorb motion that you may prefer to explain with expression or pose.

If your goal is clean FLAME parameters rather than maximum surface fit, static-on and dynamic-off is a good initial choice.

### 6. Temporal smoothness weights

Temporal smoothness terms in [../vhap/config/base.py](../vhap/config/base.py) directly influence sequence stability:

- `smooth_trans`,
- `smooth_rot`,
- `smooth_neck`,
- `smooth_jaw`,
- `smooth_expr`.

If the output jitters frame to frame, these weights are more relevant than background color.

Too low:

- unstable sequences,
- expression flicker,
- pose jitter.

Too high:

- oversmoothed motion,
- damped expressions,
- underfitting fast movement.

### 7. Camera quality

When camera intrinsics are known and correct, pose and shape usually become more stable. If camera parameters are estimated instead, the tracker can still work, but the solution is less constrained.

For monocular fitting, camera assumptions can influence pose and geometry more than background color does.

### 8. Input resolution and image quality

Higher-quality facial detail generally helps:

- landmark detection,
- photometric alignment,
- expression refinement.

If you downsample too aggressively, you reduce useful facial signal.

## What matters in your current wrapper specifically

For the current wrapper in [../../tools/video_to_flame_param.py](../../tools/video_to_flame_param.py):

- `data.background_color` is explicitly set to `None`.
- the render background still defaults to `target`.
- the optimization budget is intentionally small.

There is also one practical detail: the current `main()` path directly calls `video2frames()` and does not go through `_preprocess()`, so optional matting is not part of the active execution path right now.

That means in the current wrapper:

- background compositing is disabled,
- optional matting is bypassed,
- the main quality bottleneck is much more likely to be optimization budget and fitting settings than background choice.

## Practical recommendation for your goal

If your goal is better FLAME shape, expression, and pose sequences, use this order of priority:

1. Keep `background_color=None` unless you have a very specific boundary problem.
2. Increase the optimization budget well beyond the current debug settings.
3. Keep photometric fitting enabled.
4. Check landmark quality before tuning cosmetic preprocessing choices.
5. Keep static offset enabled and dynamic offset disabled as the initial baseline.
6. Tune temporal smoothness only if you actually observe jitter.

## Recommended default stance

For FLAME parameter quality alone:

- `background_color=None` is a good default.
- White or black background is mainly a dataset hygiene and export choice.
- If you have limited time, spend effort on optimization budget, landmarks, and photometric fitting first.