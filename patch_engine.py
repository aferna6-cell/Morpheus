path = '/home/aidan/polymarket-bot/src/engines/kalshi_llm_engine.py'
with open(path) as f:
    content = f.read()

# Replace the _scan_once method to use close-date scanning
old_scan = '''    async def _scan_once(self) -> None:
        # Fetch markets via event-based scan (better for politics/economics)
        kalshi_markets = await self.kalshi_client.fetch_markets_by_events(
            categories=self._categories,
            min_volume=self._min_volume,
        )

        self.logger.info("kalshi_llm_scan", markets_found=len(kalshi_markets))

        for km in kalshi_markets:
            # Skip markets with no useful price data
            if km.yes_price <= 0 or km.yes_price >= 1:
                continue

            # Skip markets with no close time or that resolve too far out
            if not km.close_time:
                continue
            days_out = (km.close_time - datetime.now(timezone.utc)).total_seconds() / 86400.0
            if days_out > self._max_resolution_days:
                continue'''

new_scan = '''    async def _scan_once(self) -> None:
        # Fetch markets by close date — gets short-term markets across ALL categories
        kalshi_markets = await self.kalshi_client.fetch_markets_by_close_date(
            max_days=self._max_resolution_days,
            min_volume=self._min_volume,
        )

        self.logger.info("kalshi_llm_scan", markets_found=len(kalshi_markets))

        for km in kalshi_markets:
            # Skip markets with no useful price data (already filtered, but safety check)
            if km.yes_price <= 0 or km.yes_price >= 1:
                continue'''

if old_scan in content:
    content = content.replace(old_scan, new_scan)
    with open(path, 'w') as f:
        f.write(content)
    print('Engine updated to use close-date scanning')
else:
    print('ERROR: scan method not found as expected')
    # Debug
    import re
    m = re.search(r'async def _scan_once.*?(?=async def|\Z)', content, re.DOTALL)
    if m:
        print('Found _scan_once, first 500 chars:')
        print(m.group()[:500])
