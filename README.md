# DaLat Booking Recommendation Chatbot

![GitHub](https://img.shields.io/github/license/thangbuiq/booking-bot) ![Python](https://img.shields.io/badge/python-3.11-blue) ![GitHub last commit](https://img.shields.io/github/last-commit/thangbuiq/booking-bot) ![GitHub top language](https://img.shields.io/github/languages/top/thangbuiq/booking-bot)

- This project is about booking hotels recommendation chatbot on Vietnamese hotels using GraphRag.

## Usage

### Prerequisites

#### 1. Python packages

- Install package [uv](https://github.com/astral-sh/uv):

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- Activate virtual environment:

```bash
# Create virtual environment.
uv venv --python 3.11

source .venv/bin/activate
```

- Install dependencies:

```bash
uv sync
```

#### 2. Services integration

- Install [Docker](https://docs.docker.com/get-docker/).

- Install `build-essential`:

```bash
sudo apt-get update && sudo apt-get install build-essential -y
```

- Create `.env` file:

```bash
cp .env.example .env

# Then, update the values.
```

- Start services:

```bash
docker compose up -d

# or

make up
```