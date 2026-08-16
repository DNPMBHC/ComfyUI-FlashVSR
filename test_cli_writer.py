import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from cli_main import VideoWriter, _resolve_force_offload, main, parse_args


class TestCliVideoWriter(unittest.TestCase):
    def _parse_args(self, output_path, *extra_args):
        argv = [
            "cli_main.py", "-i", "in.mp4", "-o", output_path,
            "--device", "cpu", "--precision", "fp16", *extra_args,
        ]
        with patch("sys.argv", argv):
            return parse_args()

    def _fake_nodes(self, flashvsr_result=None, flashvsr_error=None):
        module = types.ModuleType("nodes")
        module.init_pipeline = MagicMock(return_value=object())
        module.flashvsr = MagicMock(
            return_value=flashvsr_result,
            side_effect=flashvsr_error,
        )
        module.log = MagicMock()
        module.VAE_MODEL_OPTIONS = []
        module.VAE_MODEL_MAP = {}
        return module

    def _fake_reader(self):
        reader = MagicMock()
        reader.start_frame = 0
        reader.end_frame = 1
        reader.get_info.return_value = (24.0, 1)
        reader.__iter__.return_value = iter([
            np.zeros((1, 4, 4, 3), dtype=np.float32),
        ])
        return reader

    def _fake_torch(self):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=MagicMock(),
            set_device=MagicMock(),
        )
        torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
        )
        torch.float16 = object()
        torch.bfloat16 = object()
        return torch

    def test_quality_defaults_match_node_defaults(self):
        with patch("sys.argv", ["cli_main.py", "-i", "in.mp4", "-o", "out.mp4"]):
            args = parse_args()
        self.assertEqual(args.mode, "full")
        self.assertEqual(args.local_range, 11)

    def test_force_offload_combines_init_and_runtime_flags(self):
        args = self._parse_args("out.mp4", "--no_keep_models_on_cpu")
        self.assertTrue(_resolve_force_offload(args))

        args.no_force_offload = True
        self.assertFalse(_resolve_force_offload(args))

    def test_main_passes_resolved_force_offload_to_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "out.mp4")
            args = self._parse_args(output_path, "--no_keep_models_on_cpu")
            nodes = self._fake_nodes(
                flashvsr_result=np.zeros((1, 8, 8, 3), dtype=np.float32),
            )
            writer = MagicMock()

            with patch("cli_main.parse_args", return_value=args), \
                 patch("cli_main.VideoReader", return_value=self._fake_reader()), \
                 patch("cli_main.VideoWriter", return_value=writer), \
                 patch.dict(sys.modules, {"nodes": nodes, "torch": self._fake_torch()}), \
                 patch("cli_main.gc.collect"), \
                 patch("builtins.print"):
                main()

        self.assertTrue(nodes.flashvsr.call_args.kwargs["force_offload"])
        writer.write.assert_called_once()
        writer.release.assert_called_once()

    def test_processing_failure_removes_partial_output_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "out.mp4")
            args = self._parse_args(output_path)
            nodes = self._fake_nodes(
                flashvsr_result=np.zeros((1, 8, 8, 3), dtype=np.float32),
            )
            writer = MagicMock()
            writer.write.side_effect = RuntimeError("disk full")

            def create_writer(*_args, **_kwargs):
                with open(output_path, "wb") as output_file:
                    output_file.write(b"partial")
                return writer

            with patch("cli_main.parse_args", return_value=args), \
                 patch("cli_main.VideoReader", return_value=self._fake_reader()), \
                 patch("cli_main.VideoWriter", side_effect=create_writer), \
                 patch.dict(sys.modules, {"nodes": nodes, "torch": self._fake_torch()}), \
                 patch("cli_main.gc.collect"), \
                 patch("traceback.print_exc"), \
                 patch("builtins.print") as output:
                with self.assertRaises(SystemExit) as raised:
                    main()

            self.assertEqual(raised.exception.code, 1)
            self.assertFalse(os.path.exists(output_path))
            self.assertNotIn(
                "FlashVSR processing complete!",
                "\n".join(str(call) for call in output.call_args_list),
            )

    def test_ffmpeg_command_honors_codec_and_crf(self):
        process = MagicMock()
        process.stdin.closed = False
        process.stderr.read.return_value = b""
        process.wait.return_value = 0

        with patch("cli_main.shutil.which", return_value="ffmpeg"), \
             patch("cli_main.subprocess.Popen", return_value=process) as popen:
            writer = VideoWriter("out.mp4", 24, 16, 16, codec="libx264", crf=12)
            writer.write(np.zeros((1, 16, 16, 3), dtype=np.float32))
            writer.release()

        command = popen.call_args_list[0].args[0]
        self.assertIn("libx264", command)
        self.assertEqual(command[command.index("-crf") + 1], "12")
        process.stdin.write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
