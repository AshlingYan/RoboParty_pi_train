import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                # DEBUG: log received observation keys/types to help diagnose missing prompt
                try:
                    logger.info(f"Received obs keys: {list(obs.keys())}")
                    # also log simple summary of 'task'/'prompt' if present
                    if "task" in obs:
                        logger.info(f"obs['task'] type={type(obs['task'])} value_sample={str(obs['task'])[:200]}")
                    if "prompt" in obs:
                        logger.info(f"obs['prompt'] type={type(obs['prompt'])} value_sample={str(obs['prompt'])[:200]}")
                    if "image" in obs and isinstance(obs["image"], dict):
                        logger.info(f"image keys: {list(obs['image'].keys())}")
                except Exception:
                    logger.exception("Error logging obs summary")

                infer_time = time.monotonic()
                # Run blocking inference in a threadpool to avoid blocking the asyncio
                # event loop (model inference may be slow/heavy). This prevents the
                # websockets keepalive pings from timing out.
                try:
                    loop = asyncio.get_running_loop()
                    action = await loop.run_in_executor(None, self._policy.infer, obs)
                except Exception:
                    # If run_in_executor fails for some reason, fall back to direct call
                    action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
