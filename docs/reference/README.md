# Reference material

Datasheets, specs and standards this project leans on. Names only — drop the
PDFs here (gitignored if large) or link to the vendor page.

- NVIDIA Jetson Orin Nano (Super) Developer Kit — carrier board spec, JetPack / CUDA version matrix
- NVIDIA DeepStream SDK documentation (Build 3 camera source)
- CuPy documentation — cuFFT bindings, pinned/unified memory
- Ettus Research USRP B210 — hardware spec, USB 3 streaming limits, GPSDO option, 10 MHz / PPS inputs
- UHD (USRP Hardware Driver) Python API — `time_spec`, `set_time_unknown_pps`, clock/time source selection
- Kismet REST / websocket API — `/devices/monitor.ws`, device field names (`kismet.device.base.*`)
- gpsd protocol (JSON, port 2947)
- PyYAML / YAML 1.1 scalar resolution (signed-exponent float gotcha)
- XDG Base Directory Specification (`$XDG_STATE_HOME`)
