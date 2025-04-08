import os
import subprocess
import glob
import time
import random

dataset_root = 'raw_videos'
vid_file_lst = ['splits/train_list.txt', 'splits/val_list.txt', 'splits/test_list.txt']
split_lst = ['training', 'validation', 'testing']
if not os.path.isdir(dataset_root):
    os.mkdir(dataset_root)

missing_vid_lst = []

# download videos for training/validation/testing splits
for vid_file, split in zip(vid_file_lst, split_lst):
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        os.mkdir(split_dir)
        
    with open(vid_file) as f:
        lines = f.readlines()
        for line in lines:
            rcp_type, vid_name = line.replace('\n', '').split('/')
            rcp_dir = os.path.join(dataset_root, split, rcp_type)
            if not os.path.isdir(rcp_dir):
                os.mkdir(rcp_dir)

            # download the video
            vid_url = 'https://www.youtube.com/watch?v=' + vid_name
            vid_prefix = os.path.join(dataset_root, split, rcp_type, vid_name)
            print(f" [INFO] Downloading {split} video {vid_name} from {vid_url}")
            
            # Use yt-dlp instead of youtube-dl and add retry logic
            max_retries = 3
            download_success = False
            
            for retry in range(max_retries):
                try:
                    # Try with specific format and extension in the output template
                    cmd = ["yt-dlp", "-f", "best", "-o", f"{vid_prefix}.%(ext)s", "--no-warnings", vid_url]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Check for any created video files regardless of extension
                    video_files = glob.glob(f"{vid_prefix}.*")
                    video_extensions = ['.mp4', '.mkv', '.webm', '.mov', '.avi']
                    found_video = False
                    
                    for file in video_files:
                        if any(file.endswith(ext) for ext in video_extensions):
                            found_video = True
                            actual_file = file
                            print(f"[INFO] Successfully downloaded {split} video {vid_name} as {os.path.basename(actual_file)}")
                            download_success = True
                            break
                    
                    if not found_video and retry < max_retries - 1:
                        print(f"[WARNING] Attempt {retry+1} succeeded but no video file found for {vid_name}. Retrying...")
                        # Exponential backoff with jitter
                        wait_time = (2 ** retry) + random.uniform(0, 1)
                        time.sleep(wait_time)
                    elif not found_video and retry == max_retries - 1:
                        # Final attempt with more verbose logging
                        print(f"[WARNING] Final attempt for {vid_name}")
                        cmd = ["yt-dlp", "--verbose", "-o", f"{os.path.dirname(vid_prefix)}/{vid_name}.%(ext)s", vid_url]
                        subprocess.run(cmd)
                        
                        # Check one more time
                        video_files = glob.glob(f"{os.path.dirname(vid_prefix)}/{vid_name}.*")
                        if any(os.path.exists(f) for f in video_files if any(f.endswith(ext) for ext in video_extensions)):
                            print(f"[INFO] Final attempt succeeded for {vid_name}")
                            download_success = True
                        else:
                            print(f"[INFO] Cannot download {split} video {vid_name} after all attempts")
                    
                    # If we found a video, break out of retry loop
                    if download_success:
                        break
                        
                except Exception as e:
                    print(f"[ERROR] Exception when downloading {vid_name} (attempt {retry+1}): {str(e)}")
                    if retry < max_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = (2 ** retry) + random.uniform(0, 1)
                        time.sleep(wait_time)
            
            # After all retries, check if we need to add to missing list
            if not download_success:
                missing_vid_lst.append('/'.join((split, rcp_type, vid_name)))
                print(f"[INFO] Cannot download {split} video {vid_name}")

# write the missing videos to file
with open('missing_videos.txt', 'w') as missing_vid:
    for line in missing_vid_lst:
        missing_vid.write(line + '\n')

print(f"[INFO] Download completed. {len(missing_vid_lst)} videos could not be downloaded.")
print("[INFO] Cleaning up temporary files...")

# sanitize and remove the intermediate files
os.system("find raw_videos -name '*.part*' -delete")
os.system("find raw_videos -name '*.f*' -delete")