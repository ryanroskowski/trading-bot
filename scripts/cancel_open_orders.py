from __future__ import annotations

from loguru import logger

from src.execution.alpaca import AlpacaConnector
from src.storage.db import mark_order_canceled


def main() -> None:
    connector = AlpacaConnector()
    open_orders = connector.list_open_orders()
    total = len(open_orders)
    canceled = 0
    logger.info(f"Found {total} open orders")
    for o in open_orders:
        client_id = o.get("client_order_id")
        symbol = o.get("symbol")
        try:
            ok = connector.cancel_order(client_id)
            if ok:
                mark_order_canceled(client_id)
                canceled += 1
                print(f"CANCELED {client_id} {symbol}")
            else:
                print(f"FAILED {client_id} {symbol}")
        except Exception as e:
            print(f"ERROR {client_id} {symbol}: {e}")
    logger.info(f"Canceled {canceled}/{total} open orders")


if __name__ == "__main__":
    main()


