# vehicle-detection-game

A demo betting game on vehicle counts from traffic camera clips.

## Start the app

From the project folder:

1. Copy the environment file. It is gitignored, so it does not come with a clone.
   On Windows: copy .env.example .env
   On Linux, macOS or Git Bash: cp .env.example .env

2. Start everything: docker compose up --build

3. Open http://localhost:8080

You should see "Vehicle Detection Game" with "Hello from the backend" below it.

The first build takes a few minutes. Later starts are fast.

To stop, press Ctrl+C and then run: docker compose down

## Weights and datasets are not included

The YOLOv9 checkpoint weights/yolov9_vehicle_detection_best.pt is not in the repository. It is gitignored and has to be copied in by hand on each machine. The same goes for the datasets in data/ and the generated clips in media/.

None of it is needed yet. The app starts and runs without the weights or any dataset.
