# FlyKeeper visualizer

Local observational UI: `npm install && npm run dev`, then open the displayed local URL. Start a run with `--visualizer`; its WebSocket is `127.0.0.1:8767`. The frontend never sends simulation commands.

The brain panel uses Cerebra's documented MaleCNS source-anchor rendering strategy and remote CC-BY atlas anchors. It supports orbit/zoom/pan. Current implementation is an overview; full skeleton selection can be added using Cerebra's MIT `atlas.ts` pattern and its source API.
