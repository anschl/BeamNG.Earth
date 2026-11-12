import errno
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import xproc  # noqa: E402


class TakeTest(unittest.TestCase):
    def test_a_full_disk_works_unlocked_and_leaves_no_lock(self):
        lock = os.path.join(tempfile.mkdtemp(), "piece.lock")

        def full(fd, data):
            raise OSError(errno.ENOSPC, "No space left on device")
        with mock.patch.object(xproc.os, "write", full):
            self.assertTrue(xproc.take(lock))
        self.assertFalse(os.path.exists(lock), "no empty lock for the other processes to wait on")
        self.assertTrue(xproc.take(lock), "the next one takes it")
        xproc.release(lock)


if __name__ == "__main__":
    unittest.main()
