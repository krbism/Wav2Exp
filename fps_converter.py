import os
import cv2
import numpy as np
import librosa
import soundfile as sf
import subprocess
import platform

import re

digits = re.compile(r'(\d+)')
template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

def generate_video(frames, audio_file, output_file_name, fps=30):

    fname = 'inference.avi'
    video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), fps, (224, 224))
 
    for i in range(len(frames)):
        img = frames[i]
        img = cv2.imread(img)
        img = cv2.resize(img, [224,224])
        # img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
        # img = np.uint8(img*255)
        video.write(img)

    print("done till here")
    
    video.release()

    no_sound_video = output_file_name + '_nosound.mp4'
    subprocess.call('ffmpeg -hide_banner -loglevel panic -i %s -c copy -an -strict -2 %s' % (fname, no_sound_video), shell=True)

    video_output_mp4 = output_file_name + '.mp4'

    # subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 %s' % (audio_file, no_sound_video, video_output_mp4), shell=True)
    command = "ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}".format(audio_file, no_sound_video, video_output_mp4)
    subprocess.call(command, shell=platform.system() != 'Windows')
    os.remove(fname)
    os.remove(no_sound_video)

    return 

def save_frames(vid_path, vid_name):
    video_stream = cv2.VideoCapture(vid_path)
    frames = []
    idx = 0

    vid_dir = os.path.join("25_FPS", vid_name)

    os.mkdir(vid_dir)

    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        cv2.imwrite(os.path.join(vid_dir, str(idx)+".jpg"), frame)
        idx = idx + 1

    wavpath = os.path.join(vid_dir, 'audio.wav')
    command = template.format(vid_path, wavpath)
    subprocess.call(command, shell=True)

def tokenize(filename):
        return tuple(int(token) if match else token
                     for token, match in
                     ((fragment, digits.search(fragment))
                      for fragment in digits.split(filename)))

def convert_fps(datapath, filelist):
    for video_name in filelist:
        path = os.path.join(datapath, video_name)
        print(path)
        # path = "M0006_front_happy_level_2_010"

        list_imgs = os.listdir(path)
        list_imgs.remove("audio.wav")

        list_imgs.sort(key=tokenize)


        frames = []
        for img in list_imgs:
            frames.append(os.path.join(path, img))



        aud = os.path.join(path, "audio.wav")
        output_path = os.path.join("temp_folder", video_name)
        video = generate_video(frames, aud, output_path)
        print(video)
        new_vid_path = os.path.join("temp_folder", video_name+"_25FPS.mp4")
        old_vid_path = output_path+".mp4"
        subprocess.call("ffmpeg -i %s -filter:v fps=fps=25 %s" % (old_vid_path, new_vid_path), shell=True)

        save_frames(new_vid_path, video_name)


if __name__ == '__main__':
    datapath = "/home/ubuntu/KRB_Projects/wav2exp/temp/30_FPS"

    filelist = os.listdir(datapath)
    convert_fps(datapath, filelist)




