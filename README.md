# Vision Core

## Dependencies


## Setups

## Functions

Image processing;

Bulk processing (extract frames to directory then process);

Stream process (extract and process frames one-by-one)

## Run the files

    cmake --build --preset msvc-ninja-debug --clean-first

    ./build/a_demo.exe [--framesrc <image (directory to images) or video (path to video file) (default as ./data/frames/ for images and ./data/videos/video_name.mp4 for video)>] [--out <output seat states file path, default as .\runtime\seat_states.jsonl>] [--max <maximum processing images/frames cnt>] [--fps <frame-per-second to sample in video (default as 2) and frame-per-100-images to sample for images (default as 20)>] [--stream <true/false, ignore when processing images>]

Examples:

    cmake --build --preset msvc-ninja-debug --clean-first

    ./build/a_demo.exe [--framesrc <.\data\videos\v_20251123211503_cam54_002.mp4>] [--out <.\runtime\seat_states.jsonl>] [--max <10>] [--stream <true>]

## Project Part Structure

Vision_Core

|- apps

||- a_demo.cpp                 // demo file, with cli receiving inputs and allows 3 types of data processing approaches

|||- AnnotateSeatStates.cpp  

|||- AnnotateTables.cpp

|||- AnnotatePerson.cpp

|||_ ...                       // omit if ... (test files)

|-build

||_ ...

|-config

||- poly_simple_seats.json    // seat table used in 4 table version video v_20251123210004_cam54_001.mp4 and v_20251123211503_cam54_002.mp4

||_ ...

|- data

||- frames

|||- frames_vXXX               // folder storing imgs to be processed

||- models

|||- weights                   // weights folders (*.pt, postfix ".pt")

|||- person01.onnx             // newly fine-tuning model

|||- yolov8n_640.onnx          // initially downloaded pre-train model

|- include - vision

||- Config.h               

||- FrameProcessor.h           // 3 processing approaches

||- OrtYolo.h                  // calling model and conduct inference

||- Publish.h                  // use for publishing and pushing infos and states record to B and C (To be completed)

||- Snapshotter.h              // snapshot saving

||- VisionA.h                  

||_ ...

|- runtime

||- frames                   // folder cache for bulk processing (require to move imgs to newly created ./runtime/Frames_vXXX for cache if needed to save the imgs)

||- last_frame.jsonl

||- seat_states.jsonl        // output seat states jsonl file (default location of arg --out)

|- scripts                   // use for running fine-tuning on models

|- src (similar to include content)

|- tests






