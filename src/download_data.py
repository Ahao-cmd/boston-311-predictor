import requests
import pandas as pd
from pathlib import Path


API_URL = "https://data.boston.gov/api/3/action/datastore_search" #data source

RESOURCE_ID = "dff4d804-5031-443a-8409-8344efd0e5c8"# Boston 311 Service Requests - 2024


SAMPLE_SIZE = 5000


def download_sample():
    print("Downloading Boston 311 sample...")

    params = {
        "resource_id": RESOURCE_ID,
        "limit": SAMPLE_SIZE
    }

    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if not data["success"]:
        raise RuntimeError("Boston Open Data API request failed.")

    records = data["result"]["records"]

    print("Records downloaded:", len(records))
    print("Total records available:", data["result"]["total"])

    df = pd.DataFrame(records)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "boston_311_2024_sample.csv"

    df.to_csv(output_path, index=False)

    print("Saved to:", output_path)


if __name__ == "__main__":
    download_sample()