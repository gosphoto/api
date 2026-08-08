#!/usr/bin/env python3
"""Resolve Tochka customerCode using TOCHKA_ACCESS_TOKEN. Prints only the code."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API = os.environ.get("TOCHKA_API_BASE_URL", "https://enter.tochka.com").rstrip("/")
TOKEN = os.environ.get("TOCHKA_ACCESS_TOKEN", "").strip()


def main() -> int:
    if not TOKEN:
        print("TOCHKA_ACCESS_TOKEN is empty", file=sys.stderr)
        return 1
    req = urllib.request.Request(
        f"{API}/uapi/open-banking/v1.0/customers",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        print(f"HTTP {e.code}: {err}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"request failed: {e}", file=sys.stderr)
        return 1

    root = json.loads(body)
    data = root.get("Data") if isinstance(root, dict) else None
    if not isinstance(data, dict):
        print("unexpected response shape", file=sys.stderr)
        return 1

    direct = (data.get("customerCode") or "").strip()
    if direct:
        print(direct)
        return 0

    raw = data.get("Customer")
    customers: list[dict] = []
    if isinstance(raw, dict):
        customers = [raw]
    elif isinstance(raw, list):
        customers = [c for c in raw if isinstance(c, dict)]

    business = [
        c
        for c in customers
        if str(c.get("customerType") or "").lower() == "business"
        and str(c.get("customerCode") or "").strip()
    ]
    if len(business) == 1:
        print(str(business[0]["customerCode"]).strip())
        return 0
    if len(customers) == 1 and str(customers[0].get("customerCode") or "").strip():
        print(str(customers[0]["customerCode"]).strip())
        return 0

    print(
        f"cannot pick customerCode uniquely (business={len(business)}, total={len(customers)})",
        file=sys.stderr,
    )
    for c in customers:
        print(
            f"  type={c.get('customerType')} code={c.get('customerCode')}",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
