"""Add the bundled sensor log explicitly to a welding research question."""

from pathlib import Path
import pandas as pd
from examples.rag_basic import create_pipeline

if __name__ == "__main__":
    readings = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "sensor_log.csv")
    response = create_pipeline().query(
        "What defects appear in this example sensor log, and what possible FSW causes fit?",
        sensor_context=readings.tail(10).to_csv(index=False),
    )
    print(response.answer)
