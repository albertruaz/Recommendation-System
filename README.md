# Recommendation System

A modular and scalable user-product recommendation system based on PySpark ALS.

## Project Structure

```
recommendation/
├── als_config.json          # Main configuration file
├── cache/                   # Cache data directory
│   └── interactions/        # Interaction data cache
├── config/                  # Additional config files
├── core/                    # Core logic
│   ├── __init__.py
│   └── model.py            # ALS model implementation
├── database/               # Database connections
│   ├── __init__.py
│   ├── base_connector.py   # Base DB connector
│   ├── db_manager.py       # DB manager
│   ├── mysql_connector.py  # MySQL connector
│   └── postgres_connector.py # PostgreSQL connector
├── services/               # Service layer
│   ├── __init__.py
│   ├── db_service.py       # Database service
│   └── recommendation_service.py # Recommendation service
└── utils/                  # Utilities
    ├── logger.py           # Logging configuration
    └── spark_utils.py      # Spark utilities
```

## Configuration

All settings are managed in `als_config.json`:

```json
{
  "pyspark_als": {
    "max_iter": 10,
    "reg_param": 0.1,
    "rank": 10,
    "interaction_weights": {
      "view": 2.0,
      "like": 5.0,
      "cart": 7.0,
      "purchase": 10.0
    }
  },
  "default_params": {
    "days": 30,
    "top_n": 10
  },
  "database": {
    "save_to_db": true,
    "db_type": "postgres"
  }
}
```

## Setup

### Requirements

- Python 3.8.20
- conda environment: `recommend`

### Installation

1. Create and activate conda environment:

```bash
conda create -n recommend python=3.8.20
conda activate recommend
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables (.env):

```env
# MySQL
DB_HOST=your_db_host
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database

# PostgreSQL
PG_HOST=your_pg_host
PG_USER=your_pg_username
PG_PASSWORD=your_pg_password
PG_DB_NAME=your_pg_database
```

## Usage

Run with default configuration:

```bash
python run.py
```

Run with specific configuration:

```bash
python run.py als_config_custom.json
```

Results will be saved in the `output/` directory.
