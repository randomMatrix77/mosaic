import json
import unittest

from mosaic.agent import AgentLoop
from mosaic.events import RequestEventEmitter, SafeAnswerStreamer


class StreamingEventTests(unittest.TestCase):
    def test_emitter_adds_request_id_and_serializes_only_known_events(self):
        payloads = []
        emitter = RequestEventEmitter("request-123", payloads.append)

        event = emitter.emit("model_loading", model="Qwen")

        self.assertEqual("request-123", event["request_id"])
        self.assertEqual(1, event["sequence"])
        self.assertIsInstance(event["timestamp_ms"], int)
        delivered = json.loads(payloads[0])
        self.assertEqual("request-123", delivered["request_id"])
        self.assertEqual("model_loading", delivered["type"])
        self.assertEqual("Qwen", delivered["model"])
        self.assertEqual(1, delivered["sequence"])
        completed = emitter.emit("answer_completed", message_id="message-1")
        self.assertEqual(2, completed["sequence"])
        self.assertGreaterEqual(completed["timestamp_ms"], event["timestamp_ms"])
        with self.assertRaises(ValueError):
            emitter.emit("unrecognized_event")

    def test_streamer_hides_fragmented_tool_syntax(self):
        visible = []
        streamer = SafeAnswerStreamer(visible.append)

        streamer.accept("<tool_")
        streamer.accept("call><function=search_memories></function></tool_call>")
        streamer.finish(is_tool_call=True)

        self.assertEqual([], visible)

    def test_streamer_hides_thinking_wrapper_then_emits_incrementally(self):
        visible = []
        streamer = SafeAnswerStreamer(visible.append)

        streamer.accept("<thi")
        streamer.accept("nk>private chain</think>Hello")
        streamer.accept(" world")
        streamer.finish(is_tool_call=False)

        self.assertEqual("Hello world", "".join(visible))

    def test_agent_streams_only_visible_final_answer(self):
        class DirectGateway:
            def generate(self, messages, tools, on_delta=None):
                self.assertEqual([], tools)
                for piece in ("<think>", "internal</think>Hel", "lo"):
                    if on_delta:
                        on_delta(piece)
                return {"content": "<think>internal</think>Hello", "tool_calls": [], "metrics": {}}

            def embed_query(self, text):
                raise AssertionError("Direct answer must not embed.")

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError(f"{left!r} != {right!r}")

        class EmptyRegistry:
            def schemas(self, allowed_tools):
                return []

        visible = []
        result = AgentLoop(DirectGateway(), EmptyRegistry(), max_tool_calls=0).run_streaming(
            [{"role": "user", "content": "Hello"}], visible.append)

        self.assertEqual("Hello", result.answer)
        self.assertEqual("Hello", "".join(visible))


if __name__ == "__main__":
    unittest.main()
