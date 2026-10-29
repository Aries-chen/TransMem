import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tiktoken
from transmem.main import ROOT, run


class EncodingTests(unittest.IsolatedAsyncioTestCase):
    async def test_dataset_loading_with_cp1252_default(self):
        original_open = Path.open

        def cp1252_open(path, mode='r', buffering=-1, encoding=None,
                        errors=None, newline=None):
            if 'b' not in mode and encoding in (None, 'locale'):
                encoding = 'cp1252'
            return original_open(path, mode, buffering, encoding, errors, newline)

        for subset in ('32k', '128k'):
            with self.subTest(subset=subset):
                args = SimpleNamespace(demo=False, data_dir=ROOT/'data/personamem',
                                       subset=subset, max_users=0)
                # Both JSONL files are loaded before the max-users validation.
                # Stop there so this test needs neither model nor database services.
                with patch.object(Path, 'open', cp1252_open), \
                        patch.object(tiktoken, 'get_encoding', return_value=None):
                    with self.assertRaisesRegex(ValueError, '--max-users must be positive'):
                        await run(args, None, {}, None)
