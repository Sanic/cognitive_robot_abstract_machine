from __future__ import annotations

import asyncio
import queue
import threading

from typing_extensions import Any


class ThreadSafeAsyncEvent:
    """An asyncio wrapper around the threading.Event to avoid blocking waits while remaining thread safe"""

    def __init__(self) -> None:
        self._event = threading.Event()
        """The underlying thread safe event."""

        self._loop = asyncio.get_event_loop()
        """The event loop used for asynchronous operation.."""

    @property
    def is_set(self) -> bool:
        """Whether the event is set or not"""
        return self._event.is_set()

    def set(self) -> None:
        """Set the event."""
        self._event.set()

    def clear(self) -> None:
        """Clear the event."""
        self._event.clear()

    async def wait(self) -> None:
        """Wait asynchronously for the event to be set."""
        await self._loop.run_in_executor(None, self._event.wait)


class ThreadSafeAsyncQueue:
    """An asyncio wrapper around the queue.Queue to avoid blocking waits while remaining thread safe"""

    def __init__(self) -> None:
        self._queue = queue.Queue()
        """The underlying thread safe queue."""

        self._loop = asyncio.get_event_loop()
        """The event loop used for asynchronous operation.."""

    @property
    def qsize(self) -> int:
        """The current size of the queue."""
        return self._queue.qsize()

    def put_nowait(self, item) -> None:
        """Add an item to the queue synchronously"""
        self._queue.put(item)

    def get_nowait(self) -> Any:
        """Get an item from the queue synchronously."""
        return self._queue.get()

    async def put(self, item) -> None:
        """Put an item to the queue asynchronously"""
        return await self._loop.run_in_executor(None, self._queue.put, item)

    async def get(self) -> Any:
        """Get an item from the queue asynchronously"""
        return await self._loop.run_in_executor(None, self._queue.get)
