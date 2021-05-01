def test_monitor_construction(monitor):
    assert 0 == monitor.max_gpu_memory_used
    assert (False == monitor.stopped)
    stop_monitor(monitor)


def test_monitor_max_memory_used(monitor):
    assert 0 == monitor.max_memory_used
    stop_monitor(monitor)


def stop_monitor(monitor):
    monitor.stop()
