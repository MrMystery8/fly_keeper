# FlyKeeper

A MaleCNS connectome-based neural simulation is placed in a closed visual-control loop controlling a simplified goalkeeper task. The brain receives only the rendered RGB frame through DoomFly's unchanged R1-R6 mapping; `retinal_samples` uses the upstream UV layout, luminance conversion, and 3,335 mapped receptor IDs. Frames are 160×96 RGB, presented every 20 ms; luminance is normalized by upstream sampling and feeds its existing 30 mV-equivalent saturating retinal drive.

The engineering decoder uses actual DNp20 body IDs 10162 (left) and 10059 (right): positive 20-ms spike-count asymmetry selects LEFT/RIGHT, otherwise STAY. Their selection/gains are inherited from the DoomFly experimental BCI mapping; they are not claims of verified natural goalkeeper behavior. No ball coordinates, velocity, prediction, rewards, or weights reach MaleCNS.

Run headless: `python -m workspace.experiments.flykeeper.run --headless`. Add `--visualizer` to publish local observational telemetry.
