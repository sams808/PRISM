import unittest
from pathlib import Path

import numpy as np

import io_universal as universal


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> Path:
    return FIXTURE_DIR / name


class GenericXYFixtureTests(unittest.TestCase):
    def test_raman_snippet_loads_with_interleaved_notes(self):
        x, y, meta = universal.import_xy(_fixture("raman_snippet.txt"))
        self.assertGreaterEqual(len(x), 5)
        self.assertEqual(len(x), len(y))
        self.assertEqual(meta["selected_parser"], "generic_xy")
        self.assertTrue(np.all(np.diff(x) >= 0))

    def test_xrd_snippet_loads_with_sparse_numeric_rows(self):
        x, y, meta = universal.import_xy(_fixture("xrd_snippet.txt"))
        self.assertGreaterEqual(len(x), 4)
        self.assertEqual(len(x), len(y))
        self.assertEqual(meta["selected_parser"], "generic_xy")
        self.assertTrue(np.all(np.diff(x) >= 0))


class GenericXYAllExamplesTests(unittest.TestCase):
    def test_every_fixture_dataset_imports_via_generic_xy(self):
        for path in sorted(FIXTURE_DIR.iterdir()):
            if not path.is_file():
                continue
            with self.subTest(path=path.name):
                x, y, meta = universal.import_xy(path)
                self.assertGreaterEqual(len(x), 2)
                self.assertEqual(len(x), len(y))
                self.assertEqual(meta["selected_parser"], "generic_xy")
                self.assertTrue(np.all(np.diff(x) >= 0))


if __name__ == "__main__":
    unittest.main()
