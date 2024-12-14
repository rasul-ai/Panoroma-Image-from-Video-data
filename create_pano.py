import cv2
import numpy as np
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // frame_rate

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{frame_count // frame_interval}.jpg")
            cv2.imwrite(frame_name, frame)
        frame_count += 1
    cap.release()
    print(f"Frames extracted to {output_folder}")


def find_overlapping_frames(images_folder):
    """Find overlapping frames by reading the extracted images."""
    images = sorted([cv2.imread(os.path.join(images_folder, f"frame_{i}.jpg")) for i in range(9)], key=lambda x: x.shape[1])
    images = [img for img in images if img is not None]  # Exclude any None images
    
    if len(images) != 9:
        raise ValueError("Expected exactly 9 images, but got a different number!")
    
    print("Found overlapping frames.")
    return images

def stitch_frames(images):
    panorama = images[0]

    for i in range(1, len(images)):
        img = images[i]
        
        # Divide the image into 3 equal parts along the width axis
        width = img.shape[1]
        third_width = width // 3
        
        # Take the rightmost 1/3 of the image
        right_third = img[:, 2*third_width:, :]
        
        # Concatenate the right third to the panorama along the width
        panorama = np.concatenate((panorama, right_third), axis=1)
    
    print("Stitching images...")

    return panorama

def save_panorama(panorama, output_path="panorama.jpg"):
    cv2.imwrite(output_path, panorama)
    print(f"Panorama saved as {output_path}")

def main():
    video_path = "./video.mp4"
    output_folder = "key_frames"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    extract_frames(video_path, output_folder)
    images = find_overlapping_frames(output_folder)
    panorama = stitch_frames(images)
    
    save_panorama(panorama)

if __name__ == "__main__":
    main()
