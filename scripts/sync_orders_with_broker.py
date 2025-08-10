from __future__ import annotations

from src.execution.alpaca import AlpacaConnector
from src.storage.db import cancel_missing_open_orders


def main() -> None:
    connector = AlpacaConnector()
    open_orders = connector.list_open_orders()
    open_ids = [o.get("client_order_id") for o in open_orders if o.get("client_order_id")]
    updated = cancel_missing_open_orders(open_ids)
    print({"broker_open_count": len(open_ids), "db_rows_canceled": updated})


if __name__ == "__main__":
    main()


