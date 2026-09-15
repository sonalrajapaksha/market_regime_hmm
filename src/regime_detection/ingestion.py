import json
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def consume_stream(
    redis_url: str,
    stream: str,
    group: str,
    consumer: str,
    handle: Callable[[dict], Awaitable[None]],
):
    """Consume JSON observations from a Redis Stream with at-least-once delivery."""
    from redis.asyncio import Redis

    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        try:
            await client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as error:
            if "BUSYGROUP" not in str(error):
                raise
        while True:
            claimed = await client.xautoclaim(stream, group, consumer, min_idle_time=60_000, start_id="0-0", count=10)
            claimed_entries = claimed[1] if len(claimed) > 1 else []
            messages = [(stream, claimed_entries)] if claimed_entries else []
            messages.extend(await client.xreadgroup(group, consumer, {stream: ">"}, count=10, block=5000))
            for _, entries in messages:
                for message_id, fields in entries:
                    try:
                        payload = json.loads(fields["payload"])
                        await handle(payload)
                        await client.xack(stream, group, message_id)
                    except Exception:
                        logger.exception("stream message processing failed", extra={"message_id": message_id})
    finally:
        await client.aclose()
