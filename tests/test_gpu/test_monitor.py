def test_monitor_construction(monitor):
    assert monitor.max_gpu_memory_used == 0
    assert (monitor.stopped == False)
    stop_monitor(monitor)


def test_monitor_max_memory_used(monitor):
    assert monitor.max_memory_used == 0
    stop_monitor(monitor)


def stop_monitor(monitor):
    monitor.stop()
