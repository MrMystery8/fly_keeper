import asyncio,threading
from websockets.asyncio.server import serve
from .protocol import frame
class TelemetryServer:
    def __init__(self,host='127.0.0.1',port=8767):self.host=host;self.port=port;self.latest=None;self._thread=None
    def publish(self,payload):self.latest=frame(payload)
    def start(self):
        def run():
            async def handler(ws):
                last=None
                while True:
                    if self.latest and self.latest!=last:last=self.latest;await ws.send(last)
                    await asyncio.sleep(.05)
            async def main():
                async with serve(handler,self.host,self.port):await asyncio.Future()
            asyncio.run(main())
        self._thread=threading.Thread(target=run,daemon=True);self._thread.start()
