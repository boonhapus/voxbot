import unittest

from voxbot.runtime.docket import register_pure_background_tasks


class FakeDocket:
    def __init__(self) -> None:
        self.registered = []

    def register(self, task) -> None:
        self.registered.append(task)


class WorkerRegistrationTests(unittest.TestCase):
    def test_worker_registers_pure_background_tasks(self) -> None:
        docket = FakeDocket()

        count = register_pure_background_tasks(docket)

        self.assertEqual(count, 1)
        self.assertEqual(len(docket.registered), 1)
        self.assertEqual(docket.registered[0].__name__, "sync_voices")
