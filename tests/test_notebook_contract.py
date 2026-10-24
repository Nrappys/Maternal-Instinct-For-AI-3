"""Static notebook contract checks. Provided for manual execution only."""
import ast
import json
from pathlib import Path
import unittest


class NotebookContract(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.notebook = json.loads((self.root/"Maternal_Instinct_Experiment.ipynb").read_text(encoding="utf-8"))

    def test_unique_cell_ids_and_parseable_code(self):
        cells = self.notebook["cells"]
        ids = [cell["id"] for cell in cells]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in cells:
            if cell["cell_type"] == "code":
                source = cell["source"]
                ast.parse("".join(source) if isinstance(source, list) else source)

    def test_run_all_cannot_directly_start_training_or_simulation(self):
        forbidden = {"learn", "fit", "train", "run", "step", "start", "_job", "_start"}
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = cell["source"]
            tree = ast.parse("".join(source) if isinstance(source, list) else source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, forbidden, cell["id"])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"Experiment", "Learner"}, cell["id"])

    def test_project_has_no_dependency_on_project_two(self):
        for path in (self.root/"maternal").glob("*.py"):
            self.assertNotIn("Maternal Instinct For AI #2", path.read_text(encoding="utf-8"))

    def test_installer_is_explicitly_opt_in_and_project_path_is_not_fixed(self):
        installer = next(cell for cell in self.notebook["cells"] if cell["id"] == "install-packages")
        tree = ast.parse("".join(installer["source"]))
        switch = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "INSTALL_PACKAGES" for target in node.targets))
        self.assertIs(switch.value.value, False)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                self.assertNotIn('PROJECT_ROOT = Path(r"C:', source)
        self.assertIn('PROJECT_ID = "maternal-instinct-3"', "".join(installer["source"]))


if __name__ == "__main__":
    unittest.main()
