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

## Database migrations and seeding

A migration is a versioned change to the database structure (tables and columns). The database remembers which migrations it has already applied, so applying them again does nothing.

Seeding fills the database with starting data: the five cities (Las Vegas, Belgrade, Berlin, Moscow, Tokyo) and the demo players demo1, demo2 and demo3. Each demo player gets the starting credits and the password from DEMO_PLAYER_PASSWORD in your .env file. Seeding is safe to repeat. It only adds what is missing and never resets existing data such as player balances or edited cities. The cities are listed in backend/app/seed/cities.yaml; to add a city, append it there and start again. Placeholder clips are not seeded yet.

Both run automatically. When you run docker compose up, a short job called migrate applies any new migrations and exits, then a short job called seed adds the missing starting data and exits. The backend starts only after both succeeded. If one fails, the backend does not start, and you can read the reason with: docker compose logs migrate or docker compose logs seed

To run the migrations by hand, with the database running: docker compose run --rm migrate

To run the seed by hand: docker compose run --rm seed

To see which version the database is on: docker compose run --rm migrate alembic current

After you add a new migration file, just run docker compose up --build again. The new migration is applied and existing data stays.

To wipe the database and start from nothing, run: docker compose down -v
This deletes the database volume, so every balance and bet is lost. Use it only when you want a fresh start.

## Run the backend tests

The schema tests need a scratch PostgreSQL database. They wipe that database each time, so never point them at your real one.

1. Start a scratch database: docker run -d --rm --name vdg-test -e POSTGRES_USER=vdg -e POSTGRES_PASSWORD=vdg -e POSTGRES_DB=vdg_test -p 55432:5432 postgres:16.4

2. In the backend folder, create and activate a virtual environment, then install the packages: pip install -r requirements-dev.txt

3. Tell the tests where the database is. In PowerShell: $env:TEST_DATABASE_URL = "postgresql+psycopg://vdg:vdg@localhost:55432/vdg_test"

4. Run: pytest

5. When finished, stop the scratch database: docker stop vdg-test

## Weights and datasets are not included

The YOLOv9 checkpoint weights/yolov9_vehicle_detection_best.pt is not in the repository. It is gitignored and has to be copied in by hand on each machine. The same goes for the datasets in data/ and the generated clips in media/.

None of it is needed yet. The app starts and runs without the weights or any dataset.
