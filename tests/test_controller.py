import queue
import unittest

from actions.controller import ActionController, ActionRequest


class ActionControllerTests(unittest.TestCase):
    def test_dispatches_custom_handler_without_blocking_caller(self) -> None:
        controller = ActionController()
        observed: queue.Queue[str] = queue.Queue()
        controller.register_handler("capture", lambda value: observed.put(str(value)))
        controller.start()

        try:
            controller.dispatch(ActionRequest(action="capture", value="ok", source="test"))
            self.assertEqual("ok", observed.get(timeout=1.0))
        finally:
            controller.stop()

    def test_executes_sequence_in_order(self) -> None:
        controller = ActionController()
        observed: queue.Queue[str] = queue.Queue()
        controller.register_handler("capture", lambda value: observed.put(str(value)))
        controller.start()

        try:
            controller.dispatch(
                ActionRequest(
                    action="sequence",
                    value=[
                        {"action": "capture", "value": "first"},
                        {"action": "capture", "value": "second"},
                    ],
                    source="test",
                )
            )
            self.assertEqual("first", observed.get(timeout=1.0))
            self.assertEqual("second", observed.get(timeout=1.0))
        finally:
            controller.stop()

    def test_dry_run_still_notifies_action_observer(self) -> None:
        observed: queue.Queue[str] = queue.Queue()
        controller = ActionController(
            dry_run=True,
            action_observer=lambda request: observed.put(request.action),
        )
        controller.start()
        try:
            controller.dispatch(ActionRequest(action="click", source="test"))
            self.assertEqual("click", observed.get(timeout=1.0))
        finally:
            controller.stop()


if __name__ == "__main__":
    unittest.main()
