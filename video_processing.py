import argparse
import logging
import os
import pathlib
import random
import sys
import time

import cv2
import torch
from ultralytics import YOLO

PATH = pathlib.Path(__file__).parent


def main():
    # define the args of the script
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_filename", type=str, default="cut.mp4")
    parser.add_argument("--output_filename", type=str, default="output.mp4")
    parser.add_argument("--logs_filename", type=str, default="processing.log")
    parser.add_argument("--output_results_directory", type=str, default="results/")
    parser.add_argument("--nb_skipped_frames_logs", type=int, default=10)
    parser.add_argument("--nb_preds_per_sec", type=int, default=10)
    parser.add_argument("--output_fps", type=int, default=30)
    parser.add_argument("--results_saved_threshold", type=int, default=10)
    parser.add_argument("--preds_threshold", type=float, default=0.5)
    parser.add_argument("--output_width", type=int, default=1280)
    parser.add_argument("--output_height", type=int, default=720)
    parser.add_argument("--model_verbose", action='store_true')
    parser.add_argument("--model_name", type=str, default="yolov8n.pt")
    parser.add_argument("--codec_name", type=str, default="mp4v")

    args = parser.parse_args()

    # input variables
    input_filename = args.input_filename
    output_filename = args.output_filename
    logs_filename = args.logs_filename
    results_directory = args.output_results_directory
    skip_frames_logs = args.nb_skipped_frames_logs
    nb_preds_per_sec = args.nb_preds_per_sec
    output_video_fps = args.output_fps
    results_saved_threshold = args.results_saved_threshold
    preds_threshold = args.preds_threshold
    resized_width = args.output_width
    resized_height = args.output_height
    model_verbose = args.model_verbose
    model_name = args.model_name
    codec_name = args.codec_name

    # set the logs
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    format_logs = "%(asctime)s - %(name)s - %(levelname)s - first frame: %(first frame)s - last frame: %(last frame)s - execution time: %(execution time)s - %(message)s"

    formatter = logging.Formatter(format_logs)
    logging.basicConfig(
        format=format_logs,
        filename=PATH.joinpath(logs_filename),
        encoding="utf-8",
        level=logging.INFO,
        force=True,
    )

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create the folder to store a sample of the model predictions
    if not os.path.exists(PATH.joinpath(results_directory)):
        os.makedirs(PATH.joinpath(results_directory))

    # initialize variables to process the video
    first_frame = 0
    frame_count = 0
    nb_detected_object = 0
    preds_frames_list = list()
    skip_frames_preds = round(output_video_fps / nb_preds_per_sec)

    # initialize the input video capture and get metadata
    if PATH.joinpath(input_filename).exists():
        input_video = cv2.VideoCapture(PATH.joinpath(input_filename))
        input_video_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_video_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_video_fps = int(input_video.get(cv2.CAP_PROP_FPS))
        input_video_nb_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        input_video_duration = int(input_video_nb_frames / input_video_fps)
        input_video_fourcc = int(input_video.get(cv2.CAP_PROP_FOURCC))
        input_video_fourcc_converted = "".join(
            [chr((input_video_fourcc >> (i * 8)) & 0xFF) for i in range(4)]
        )
        d = {"execution time": None, "first frame": None, "last frame": None}
        logger.info(
            f"video input metadata - resolution: {input_video_width}x{input_video_height} - duration: {input_video_duration} sec - nb frames: {input_video_nb_frames} - fps: {input_video_fps} - codec name: {input_video_fourcc_converted}",
            extra=d,
        )
    else:
        d = {"execution time": None, "first frame": None, "last frame": None}
        logger.error("the input file does not exist", extra=d)
        sys.exit("the input file does not exist")

    # initialize the video output file
    logger.info(
        "initializing the video output",
        extra=d,
    )
    fourcc = cv2.VideoWriter_fourcc(*codec_name)
    output_video = cv2.VideoWriter(
        PATH.joinpath(output_filename),
        fourcc,
        output_video_fps,
        (resized_width, resized_height),
    )

    # load the model
    if torch.cuda.is_available():
        model = YOLO(model_name).to("cuda")
        logger.info("loading the model on the GPU", extra=d)
    else:
        model = YOLO(model_name)
        logger.info("loading the model", extra=d)

    # start the video processing
    first_start_time = time.time()
    d = {
        "execution time": round(time.time() - first_start_time, 3),
        "first frame": first_frame,
        "last frame": frame_count,
    }
    logger.info("starting the video processing", extra=d)

    start_time = time.time()
    while True:
        ret, frame = input_video.read()

        if not ret:
            break

        frame_count += 1

        # stop the video processing if a frame is not BRG or RGB
        if len(frame.shape) < 3:
            d = {
                "execution time": round(time.time() - start_time, 3),
                "first frame": first_frame,
                "last frame": frame_count,
            }
            logger.error("the frame is not in colors RGB")
            break

        # start the objects detection
        if frame_count % skip_frames_preds == 0:
            results = model(frame, conf=preds_threshold, verbose=model_verbose)
            for result in results:
                frame = result.plot()
                if result.boxes:
                    nb_detected_object += int(result.boxes.cls.shape[0])
                    frame_dict = {"frame": frame_count, "image": frame}
                    preds_frames_list.append(frame_dict)

        # resize the frame
        resized_frame = cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
        )

        # write the frame in the output video file
        if ret == True:
            output_video.write(resized_frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        else:
            break

        # log the video processing
        if frame_count % skip_frames_logs == 0:
            d = {
                "execution time": round(time.time() - start_time, 3),
                "first frame": first_frame + 1,
                "last frame": frame_count,
            }
            logger.info(f"number of objects detected: {nb_detected_object}", extra=d)
            start_time = time.time()
            first_frame = frame_count
            nb_detected_object = 0

    # save a sample of the model predictions
    for f_dict in random.sample(preds_frames_list, results_saved_threshold):
        cv2.imwrite(
            os.path.join(
                PATH.joinpath(results_directory),
                f"output_frame_{f_dict['frame']}.png",
            ),
            f_dict["image"],
        )

    d = {
        "execution time": round(time.time() - first_start_time, 3),
        "first frame": 1,
        "last frame": frame_count,
    }
    logger.info("end of the video processing", extra=d)

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

    # get the metadata of the video output
    output_video = cv2.VideoCapture(PATH.joinpath(output_filename))
    output_video_width = int(output_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_video_height = int(output_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_fps = int(output_video.get(cv2.CAP_PROP_FPS))
    output_video_nb_frames = int(output_video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video_duration = int(output_video_nb_frames / output_video_fps)
    output_video_fourcc = int(output_video.get(cv2.CAP_PROP_FOURCC))
    output_video_fourcc_converted = "".join(
        [chr((output_video_fourcc >> (i * 8)) & 0xFF) for i in range(4)]
    )
    d = {"execution time": None, "first frame": None, "last frame": None}
    logger.info(
        f"video output metadata - resolution: {output_video_width}x{output_video_height} - duration: {output_video_duration} sec - nb frames: {output_video_nb_frames} - fps: {output_video_fps} - codec name: {output_video_fourcc_converted}",
        extra=d,
    )


if __name__ == "__main__":
    main()
