from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from app import main


class JobConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_at_most_two_jobs_execute_at_once(self) -> None:
        previous_slots = main.job_slots
        main.job_slots = asyncio.Semaphore(main.MAX_CONCURRENT_JOBS)
        active = 0
        peak = 0

        async def fake_execute(_job_id: str) -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.02)
            finally:
                active -= 1

        try:
            with patch.object(main, "_execute_job", fake_execute):
                await asyncio.gather(*(main.run_job(f"job-{index}") for index in range(5)))
        finally:
            main.job_slots = previous_slots

        self.assertEqual(main.MAX_CONCURRENT_JOBS, 2)
        self.assertEqual(peak, 2)


if __name__ == "__main__":
    unittest.main()
