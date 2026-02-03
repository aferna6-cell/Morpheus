path = '/home/aidan/polymarket-bot/src/kalshi_client.py'
with open(path) as f:
    content = f.read()

new_method = '''
    async def fetch_markets_by_close_date(
        self,
        *,
        max_days: int = 30,
        min_volume: int = 50,
        max_pages: int = 5,
    ) -> List[KalshiMarket]:
        """Fetch open markets closing within *max_days*, filtered by volume.

        Uses the ``min_close_ts`` / ``max_close_ts`` query params on the
        ``/markets`` endpoint so we get short-term markets regardless of
        category.  Sports multi-game parlays are excluded.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        min_ts = int(now.timestamp())
        max_ts = int((now + timedelta(days=max_days)).timestamp())

        all_markets: List[KalshiMarket] = []
        cursor: Optional[str] = None

        for _ in range(max_pages):
            params: Dict[str, Any] = {
                "limit": 200,
                "status": "open",
                "min_close_ts": min_ts,
                "max_close_ts": max_ts,
            }
            if cursor:
                params["cursor"] = cursor

            data = await self._get("/markets", params=params)
            raw = data.get("markets", [])

            for m in raw:
                ticker = m.get("ticker", "")
                # Skip sports parlays
                if "KXMVESPORTS" in ticker or "MULTIGAME" in ticker:
                    continue
                vol = int(m.get("volume", 0) or 0)
                if vol < min_volume:
                    continue

                parsed = self._parse_market(m)
                if parsed and parsed.yes_price > 0 and parsed.yes_price < 1:
                    all_markets.append(parsed)

            cursor = data.get("cursor")
            if not cursor or len(raw) < 200:
                break

        self.logger.info("kalshi_markets_by_close_date", total=len(all_markets), max_days=max_days)
        return all_markets

'''

# Insert before fetch_market method
marker = '    async def fetch_market(self, ticker: str) -> Optional[KalshiMarket]:'
if marker in content:
    content = content.replace(marker, new_method + marker)
    with open(path, 'w') as f:
        f.write(content)
    print('Added fetch_markets_by_close_date method')
else:
    print('ERROR: marker not found')
