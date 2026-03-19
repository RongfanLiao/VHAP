from fractions import Fraction
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Optional, List
import tyro
import ffmpeg
from PIL import Image
import torch
from vhap.data.image_folder_dataset import ImageFolderDataset
from torch.utils.data import DataLoader


def _parse_ffmpeg_ratio(value: str) -> float:
    """Parse an ffmpeg ratio string such as ``'30000/1001'``."""
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _probe_video_stream(video_path: Path) -> dict:
    """Probe a video file and return its video stream metadata."""
    probe = ffmpeg.probe(str(video_path))
    video_stream = next(
        (stream for stream in probe['streams'] if stream.get('codec_type') == 'video'),
        None,
    )
    if video_stream is None:
        raise ValueError('No video stream found')
    return video_stream


def _infer_video_fps(video_stream: dict) -> float:
    """Infer the source frame rate from ffprobe metadata."""
    source_fps = _parse_ffmpeg_ratio(video_stream.get('r_frame_rate', '0/1'))
    if source_fps > 0:
        return source_fps

    source_fps = _parse_ffmpeg_ratio(video_stream.get('avg_frame_rate', '0/1'))
    if source_fps > 0:
        return source_fps

    source_fps = int(video_stream['nb_frames']) / float(video_stream['duration'])
    if source_fps > 0:
        return source_fps

    raise ValueError('Cannot get valid video fps')


def video2frames(
    video_path: Path,
    image_dir: Path,
    keep_video_name: bool = False,
    target_fps: int = 30,
    n_downsample: int = 1,
) -> None:
    """Extract frames from a video file.

    Parameters
    ----------
    video_path : Path
        Path to the source video.
    image_dir : Path
        Directory where extracted JPEG frames are written.
    keep_video_name : bool, default=False
        Whether to prefix each frame name with the input video stem.
    target_fps : int, default=30
        Frame rate used for extraction.
    n_downsample : int, default=1
        Spatial downsampling factor applied to both width and height.

    Raises
    ------
    ValueError
        If the input video does not contain a valid video stream or frame rate.
    """
    if target_fps <= 0:
        raise ValueError(f'target_fps must be positive, got {target_fps}')
    if n_downsample <= 0:
        raise ValueError(f'n_downsample must be positive, got {n_downsample}')

    print(f'Converting video {video_path} to frames with downsample scale {n_downsample}')
    image_dir.mkdir(parents=True, exist_ok=True)

    video_stream = _probe_video_stream(video_path)
    source_fps = _infer_video_fps(video_stream)

    source_num_frames = int(video_stream['nb_frames'])
    source_width, source_height = (
        int(video_stream['width']),
        int(video_stream['height']),
    )

    output_width, output_height = (
        source_width // n_downsample,
        source_height // n_downsample,
    )
    estimated_output_frames = round(source_num_frames * target_fps / source_fps)

    print(
        f'[Video]  FPS: {source_fps} | number of frames: {source_num_frames} '
        f'| resolution: {source_width}x{source_height}'
    )
    print(
        f'[Target] FPS: {target_fps} | number of frames: {estimated_output_frames} '
        f'| resolution: {output_width}x{output_height}'
    )

    output_file_prefix = f'{video_path.stem}_' if keep_video_name else ''
    (
        ffmpeg
        .input(str(video_path))
        .filter('fps', fps=f'{target_fps}')
        .filter('scale', width=output_width, height=output_height)
        .output(
            str(image_dir / f'{output_file_prefix}%06d.jpg'),
            start_number=0,
            qscale=1,  # lower values mean higher quality (1 is the best, 31 is the worst).
        )
        .overwrite_output()
        .run(quiet=True)
    )


def robust_video_matting(image_dir: Path, N_warmup: Optional[int]=10):
    print(f'Running robust video matting on images in {image_dir}')
    # model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda()
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").cuda()

    dataset = ImageFolderDataset(image_folder=image_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
    rec = [None] * 4  # Initial recurrent states.
    downsample_ratio = 0.5  #(for videos in 512x512)
    # downsample_ratio = 0.125  #(for videos in 3k)
    for item in tqdm(dataloader):
        rgb = item['rgb']
        rgb = rgb.permute(0, 3, 1, 2).float().cuda() / 255
        with torch.no_grad():
            while N_warmup:
                # use the first frame to warm up the recurrent states as if the video 
                # has N_warmup identical frames at the beginning. This trick effectively
                # removes the artifacts for the first frame.
                fgr, pha, *rec = model(rgb, *rec, downsample_ratio)  # Cycle the recurrent states.
                N_warmup -= 1

            fgr, pha, *rec = model(rgb, *rec, downsample_ratio)  # Cycle the recurrent states.
            # fgr, pha, *rec = model(rgb, *rec)  # Cycle the recurrent states.
            # fg = fgr * pha + bgr * (1 - pha)

        alpha = (pha[0, 0] * 255).cpu().numpy()
        alpha = Image.fromarray(alpha.astype('uint8'))
        alpha_path = item['image_path'][0].replace('images', 'alpha_maps')
        if not Path(alpha_path).parent.exists():
            Path(alpha_path).parent.mkdir(parents=True)
        alpha.save(alpha_path)

def style_matte(
    image_dir: Path,
    model_path: str = str(Path(__file__).resolve().parents[1] / "model_zoo" / "matting" / "stylematte_synth.pt"),
):
    """Run StyleMatte foreground matting on extracted frames.

    Saves alpha maps as grayscale JPEGs in a sibling ``alpha_maps/`` folder,
    mirroring the naming convention of :func:`robust_video_matting`.
    """
    from vhap.external.human_matting import StyleMatteEngine

    print(f"Running StyleMatte matting on images in {image_dir}")
    engine = StyleMatteEngine(device="cuda", human_matting_path=model_path)

    dataset = ImageFolderDataset(image_folder=image_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    for item in tqdm(dataloader):
        rgb = item["rgb"]  # (B, H, W, 3) uint8
        rgb = rgb.permute(0, 3, 1, 2).float().cuda() / 255  # (B, 3, H, W) [0,1]

        with torch.no_grad():
            alpha_batch = engine(rgb, return_type="alpha")  # (B, H, W) [0,1]

        for i in range(alpha_batch.shape[0]):
            alpha_np = (alpha_batch[i].cpu().numpy() * 255).astype("uint8")
            alpha_img = Image.fromarray(alpha_np)

            alpha_path = item["image_path"][i].replace("images", "alpha_maps")
            Path(alpha_path).parent.mkdir(parents=True, exist_ok=True)
            alpha_img.save(alpha_path)

    torch.cuda.empty_cache()


def downsample_frames(image_dir: Path, n_downsample: int):
    print(f'Downsample frames in {image_dir} by {n_downsample}')
    assert n_downsample in [2, 4, 8]

    image_paths = sorted(list(image_dir.glob('*.jpg')))
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        # downasample the resolution of images
        img = Image.open(image_path)
        W, H = img.size
        img = img.resize((W // n_downsample, H // n_downsample))
        img.save(image_path)
    
def main(
        input: Path, 
        target_fps: int=25, 
        downsample_scales: List[int]=[],
        matting_method: Optional[Literal['robust_video_matting']]=None,
        background_folder: Path=Path('../../BACKGROUND'),
    ):
    if not input.exists():
        matched_paths = list(input.parent.glob(f"{input.name}"))
        if len(matched_paths) == 0:
            raise FileNotFoundError(f"Cannot find the directory: {input}")
        elif len(matched_paths) == 1:
            input = matched_paths[0]
        else:
            raise FileNotFoundError(f"Found multiple matched folders: {matched_paths}")
            
    # prepare path
    if input.suffix in ['.mov', '.mp4']:
        print(f'Processing video file: {input}')
        videos = [input]
        image_dir = input.parent / input.stem / 'images'
    elif input.is_dir():
        # if input is a directory, assume all contained videos are synchronized multiview of the same scene
        print(f'Processing directory: {input}')
        videos = list(input.glob('cam_*.mp4')) + list(input.glob('images/cam_*.mp4'))
        image_dir = input / 'images'
    else:
        raise ValueError(f"Input should be a video file or a directory containing video files: {input}")
    assert len(videos) > 0, f'No video files found in {input}'

    # extract frames
    for i, video_path in enumerate(videos):
        print(f'\n[{i}/{len(videos)}] Processing video file: {video_path}')

        for n_downsample in [1] + downsample_scales:
            image_dir_ = image_dir if n_downsample == 1 else Path(str(image_dir) + f'_{n_downsample}')
            video2frames(video_path, image_dir_, keep_video_name=len(videos) > 1, target_fps=target_fps, n_downsample=n_downsample)
        
    # foreground matting
    if matting_method == 'robust_video_matting':
        robust_video_matting(image_dir)
    elif matting_method is not None:
        raise ValueError(f'Unknown matting method: {matting_method}')


if __name__ == '__main__':
    tyro.cli(main)