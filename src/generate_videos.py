"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Generates multiple videos given a model and saves them as video files using ffmpeg

Usage:
    generate_videos.py [options] <model> <output_folder>

Options:
    -n, --num_videos=<count>                number of videos to generate [default: 10]
    -o, --output_format=<ext>               save videos as [default: gif]
    -f, --number_of_frames=<count>          generate videos with that many frames [default: 16]
    --ffmpeg=<str>                          ffmpeg executable path [default: ffmpeg]
"""

import os
import docopt
import torch
import numpy as np
from trainers import videos_to_numpy
import subprocess as sp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_video(video):
    """
    Validates the video shape and data type.

    Args:
        video (numpy.ndarray): Video data to validate.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    try:
        # Check the video shape
        if len(video.shape) != 4:
            raise ValueError(f"Invalid video shape: {video.shape}. Expected 4 dimensions (frames, height, width, channels).")
        
        frames, height, width, channels = video.shape
        
        # Check specific dimensions for MoCoGAN (64x64)
        if height != 64 or width != 64:
            raise ValueError(f"Invalid video dimensions: {height}x{width}. Expected 64x64.")
        
        if channels != 3:
            raise ValueError(f"Invalid number of channels: {channels}. Expected 3 (RGB).")

        # Check the data type and values
        if video.dtype != np.uint8:
            if video.min() >= 0 and video.max() <= 255:
                video = video.astype(np.uint8)
            elif video.max() <= 1.0:
                video = (video * 255).astype(np.uint8)
            else:
                raise ValueError(f"Invalid data range: min={video.min()}, max={video.max()}. Expected 0-255 or 0-1.")

        return True, video

    except Exception as e:
        logger.error(f"Video validation failed: {str(e)}")
        return False, None

def test_ffmpeg(ffmpeg_path):
    """
    Test if ffmpeg is available and working.
    
    Args:
        ffmpeg_path (str): Path to ffmpeg executable
    
    Returns:
        bool: True if ffmpeg is working, False otherwise
    """
    try:
        result = sp.run([ffmpeg_path, "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("ffmpeg test successful")
            return True
        else:
            logger.error(f"ffmpeg test failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error testing ffmpeg: {str(e)}")
        return False

def save_video(ffmpeg, video, filename, fps=8):
    """
    Save video using ffmpeg with improved error handling and format support.
    
    Args:
        ffmpeg (str): Path to ffmpeg executable
        video (numpy.ndarray): Video data to save
        filename (str): Output filename
        fps (int): Frames per second
    """
    # Validate video before saving
    is_valid, validated_video = validate_video(video)
    if not is_valid:
        logger.error("Video validation failed")
        return False

    # Print original shape for debugging
    logger.info(f"Original video shape: {validated_video.shape}")
    
    # Ensure correct dimension ordering (F, H, W, C) where F=frames, H=height, W=width, C=channels
    if validated_video.shape[-1] != 3:  # If channels are not in the last dimension
        validated_video = np.transpose(validated_video, (0, 2, 3, 1))
    
    frames, height, width, channels = validated_video.shape
    logger.info(f"Processed video shape: {validated_video.shape}")
    
    # Verify dimensions are reasonable
    if width < height/2 or height < width/2:
        logger.warning(f"Unusual aspect ratio detected: {width}x{height}")
    
    # Prepare ffmpeg command based on output format
    if filename.endswith('.gif'):
        command = [
            ffmpeg,
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # Input from pipe
            '-filter_complex', '[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse',  # Generate optimal palette
            '-r', str(fps),
            filename
        ]
    else:
        command = [
            ffmpeg,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            filename
        ]

    try:
        # Create subprocess for ffmpeg
        pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
        
        # Write video data
        pipe.stdin.write(validated_video.tobytes())
        pipe.stdin.close()
        
        # Wait for process to complete
        returncode = pipe.wait(timeout=30)
        
        if returncode != 0:
            stderr = pipe.stderr.read().decode()
            logger.error(f"ffmpeg error (code {returncode}): {stderr}")
            return False
            
        logger.info(f"Successfully saved video to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        try:
            stderr = pipe.stderr.read().decode()
            logger.error(f"ffmpeg error details: {stderr}")
        except:
            pass
        return False

def main():
    """Main function with improved error handling and logging"""
    try:
        # Parse arguments
        args = docopt.docopt(__doc__)
        
        # Test ffmpeg availability
        if not test_ffmpeg(args["--ffmpeg"]):
            logger.error("ffmpeg test failed. Please check ffmpeg installation.")
            return

        # Load model
        logger.info("Loading model...")
        generator = torch.load(args["<model>"], map_location={'cuda:0': 'cpu'})
        generator.eval()

        # Create output directory
        output_folder = args['<output_folder>']
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate videos
        num_videos = int(args['--num_videos'])
        num_frames = int(args['--number_of_frames'])
        
        logger.info(f"Generating {num_videos} videos with {num_frames} frames each...")
        
        for i in range(num_videos):
            logger.info(f"Generating video {i+1}/{num_videos}")
            
            # Generate video
            v, _ = generator.sample_videos(1, num_frames)
            video = videos_to_numpy(v).squeeze()
            
            # Log the shape at each step for debugging
            logger.info(f"Raw video shape after generation: {v.shape}")
            logger.info(f"Video shape after numpy conversion: {video.shape}")
            
            # MoCoGAN typically outputs in format (C, T, H, W) or (T, C, H, W)
            # We need to convert to (T, H, W, C)
            if video.shape[0] == 3:  # If format is (C, T, H, W)
                video = np.transpose(video, (1, 2, 3, 0))
            elif video.shape[1] == 3:  # If format is (T, C, H, W)
                video = np.transpose(video, (0, 2, 3, 1))
                
            logger.info(f"Video shape after transpose: {video.shape}")
            
            # Scale video to 0-255 range if needed
            if video.max() <= 1.0:
                video = (video * 255).astype(np.uint8)
            
            # Save video
            output_path = os.path.join(output_folder, f"{i}.{args['--output_format']}")
            if not save_video(args["--ffmpeg"], video, output_path):
                logger.error(f"Failed to save video {i+1}")
                continue

        logger.info("Video generation completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()