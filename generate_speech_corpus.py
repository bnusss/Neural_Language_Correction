# -*- coding: utf-8 -*-
# @Author: Administrator
# @Date:   2017-07-01 13:36:58
# @Last Modified by:   Administrator
# @Last Modified time: 2017-07-01 16:05:17

import os
import sys
import md5
import time
import math
import json
import wave
import pysrt
import random
import base64
import audioop
import datetime
import requests
import argparse
import tempfile
import subprocess
import multiprocessing
from progressbar import ProgressBar, Percentage, Bar, ETA


# baidu speech params
BAIDU_ID = "9176671"
BAIDU_SPEECH_CUID       = "20-7C-8F-70-85-F7"
BAIDU_SPEECH_TOKEN      = ""
BAIDU_SPEECH_API_URL    = "http://vop.baidu.com/server_api?lan={}&cuid={}&token={}"
BAIDU_SPEECH_API_KEY    = "FSsKafkOGWMZoRGRF9QBzvKO"
BAIDU_SPEECH_SECRET_KEY = "099a6f7b8568fcef84cae7025b45a2b0"


# get speech token
def getToken():
    getTokenURL = "https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}".format(BAIDU_SPEECH_API_KEY, BAIDU_SPEECH_SECRET_KEY)
    try:
        resp = requests.post(getTokenURL)
        return resp.json()["access_token"]
    except requests.exceptions.ConnectionError:
        return "Error!"
BAIDU_SPEECH_TOKEN = getToken()


# get translate params
def preparameter(src, des, content):
    salt = random.randint(32768, 65536)
    sign = BAIDU_TRANSLATE_APPID + content + str(salt) + BAIDU_TRANSLATE_APPKEY
    m1 = md5.new()
    m1.update(sign)
    sign = m1.hexdigest()
    query_content = {
        'appid': BAIDU_TRANSLATE_APPID,
        'q': content,
        'from': src,
        'to': des,
        'salt': str(salt),
        'sign': sign
    }
    return query_content


def getDocSize(path):
    try:
        size = os.path.getsize(path)
        return int(size)
    except Exception as err:
        print(err)


def percentile(arr, percent):
    arr = sorted(arr)
    k = (len(arr) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return arr[int(k)]
    d0 = arr[int(f)] * (c - k)
    d1 = arr[int(c)] * (k - f)
    return d0 + d1


def is_same_language(lang1, lang2):
    print lang1, lang2
    return lang1 == lang2


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def extract_audio(filename, channels=1, rate=16000):
    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp.close()
    if not which("ffmpeg"):
        print "ffmpeg: Executable not found on machine."
        raise Exception("Dependency not found: ffmpeg")
    command = [which("ffmpeg"), "-y", "-i", filename.decode('utf-8'), "-ac", str(channels), "-ar", str(rate), "-loglevel", "error", temp.name]
    subprocess.call(command)
    return temp.name, rate


def parse_subtitles(filename):
    if filename == None:
        return None, None
    subripItems  = pysrt.open(filename)
    time_regions = []
    subtitles    = []
    for subripItem in subripItems:
        subtitles.append(subripItem.text)
        start_time = subripItem.start.to_time()
        end_time   = subripItem.end.to_time()
        start_sencond = datetime.timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second, microseconds=start_time.microsecond).total_seconds()
        end_sencond   = datetime.timedelta(hours=end_time.hour, minutes=end_time.minute, seconds=end_time.second, microseconds=end_time.microsecond).total_seconds()
        time_regions.append((start_sencond, end_sencond))
    return time_regions, subtitles


class FLACConverter(object):
    def __init__(self, source_path, include_before=0.1, include_after=0.1):
        self.source_path = source_path
        self.include_before = include_before
        self.include_after = include_after

    def __call__(self, region):
        try:
            start, end = region
            start = max(0, start - self.include_before)
            end += self.include_after
            temp = tempfile.NamedTemporaryFile(suffix='.wav')
            temp.close()
            if not which("ffmpeg"):
                print "ffmpeg: Executable not found on machine."
                raise Exception("Dependency not found: ffmpeg")
            command = [which("ffmpeg"),"-ss", str(start), "-t", str(end - start),
                       "-y", "-i", self.source_path,
                       "-loglevel", "error", temp.name]
            subprocess.call(command)
            # os.system('stty sane')
            return temp.name
        except KeyboardInterrupt:
            return


class SpeechRecognizer(object):
    def __init__(self, lan="zh", rate=16000, retries=3):
        self.lan = lan
        self.rate = rate
        self.retries = retries

    def __call__(self, datafname):
        try:
            audio_data = open(datafname, 'rb')
            f_len = getDocSize(datafname)
            for i in range(self.retries):
                url = BAIDU_SPEECH_API_URL.format(self.lan, BAIDU_SPEECH_CUID, BAIDU_SPEECH_TOKEN)
                headers = {
                    'Content-Type': 'audio/wav;rate={}'.format(self.rate),
                    'Content-Length': str(f_len)}
                try:
                    r = requests.post(url, data=audio_data, headers=headers)
                    if r.status_code != requests.codes.ok:
                        continue
                except requests.exceptions.ConnectionError:
                    continue

                r.encoding = 'utf-8'
                result = r.json()
                if result.has_key('result'):
                    return result['result'][0]
                else:
                    return result['err_msg']
        except KeyboardInterrupt:
            return

def save_corpus(orginal_subs, recognize_subs, corpus_filename):
    with open(corpus_filename, "w") as fw:
        for o, r in zip(orginal_subs, recognize_subs):
            try:
                fw.write((o+"::"+r+"\n").encode("utf-8"))
            except:
                pass


def cmd_Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_audio_path', help="Path to the video or audio file", nargs='?')
    parser.add_argument('subtitle_path', help="Path to the video's or audio's subtitle file", nargs='?')
    parser.add_argument('-c', '--concurrency', help="Number of concurrent API requests to make", type=int, default=8)
    parser.add_argument('-o', '--output',
                        help="Output path for subtitles (by default, subtitles are saved in \
                        the same directory and name as the source path)")

    parser.add_argument('-l', '--video_language', help="Language spoken in source video file", default="zh")
    args = parser.parse_args()

    return args

def main():
    args = cmd_Parser()
    if not args.video_audio_path:
        print "Error: You need to specify a source video or audio path."
        return 1
    if not args.subtitle_path:
        print "Warning: You need to specify a source video's or audio's subtitle path."
        return 1

    audio_filename, audio_rate = extract_audio(args.video_audio_path)
    regions, subtitles = parse_subtitles(args.subtitle_path)
    pool = multiprocessing.Pool(args.concurrency)
    converter = FLACConverter(source_path=audio_filename)
    recognizer = SpeechRecognizer(lan=args.video_language, rate=audio_rate)

    transcripts = []
    if regions:
        try:
            widgets = ["Converting speech regions to FLAC files: ", Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
            extracted_region_names = []
            for i, extracted_region_name in enumerate(pool.imap(converter, regions)):
                extracted_region_names.append(extracted_region_name)
                pbar.update(i)
            pbar.finish()

            widgets = ["Performing speech recognize to FLAC files: ", Percentage(), ' ', Bar(), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=len(regions)).start()
            for i, transcript in enumerate(pool.imap(recognizer, extracted_region_names)):
                transcripts.append(transcript)
                pbar.update(i)
            pbar.finish()
        except KeyboardInterrupt:
            pbar.finish()
            pool.terminate()
            pool.join()
            print "Cancelling transcription"

        output = args.output
        if not output:
            base, ext = os.path.splitext(args.video_audio_path)
            output = "{base}.{format}".format(base=base, format="corpus")
        save_corpus(subtitles, transcripts, output)
    else:
        pass


if __name__ == '__main__':
    main()
