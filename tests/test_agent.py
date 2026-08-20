import ast
import json
from pathlib import Path

import numpy as np
import pytest

from TreasureMaze import DOWN, RIGHT, UP, TreasureMaze


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Saucier_Bradley_ProjectTwo.ipynb"

MAZE = np.array(
    [
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    ]
)


def notebook_code_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_notebook_cell_ids_are_present_and_unique():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell_ids = [cell.get("id") for cell in notebook["cells"]]

    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))


def test_environment_shape_and_starting_cells():
    qmaze = TreasureMaze(MAZE)

    assert qmaze.observe().shape == (1, 64)
    assert qmaze.target == (7, 7)
    assert int(np.count_nonzero(MAZE)) == 51
    assert len(qmaze.free_cells) == 50


def test_valid_and_invalid_actions_receive_expected_rewards():
    qmaze = TreasureMaze(MAZE)

    assert qmaze.valid_actions((0, 0)) == [DOWN]

    _, reward, status = qmaze.act(UP)
    assert reward == pytest.approx(-0.75)
    assert status == "not_over"
    assert qmaze.state == (0, 0, "invalid")

    qmaze.reset((0, 0))
    _, reward, status = qmaze.act(DOWN)
    assert reward == pytest.approx(-0.04)
    assert status == "not_over"


def test_treasure_is_a_terminal_win():
    qmaze = TreasureMaze(MAZE, pirate=(7, 5))

    _, first_reward, first_status = qmaze.act(RIGHT)
    _, final_reward, final_status = qmaze.act(RIGHT)

    assert first_reward == pytest.approx(-0.04)
    assert first_status == "not_over"
    assert final_reward == pytest.approx(1.0)
    assert final_status == "win"


def test_notebook_is_valid_python_after_magics_are_removed():
    function_names = set()

    for source in notebook_code_cells():
        python_source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
        )
        tree = ast.parse(python_source)
        function_names.update(
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )

    assert {"play_game", "completion_check", "build_model", "qtrain"} <= function_names


def test_declared_neural_network_compiles():
    import keras

    build_source = next(source for source in notebook_code_cells() if "def build_model" in source)
    namespace = {
        "Dense": keras.layers.Dense,
        "PReLU": keras.layers.PReLU,
        "Sequential": keras.models.Sequential,
        "keras": keras,
        "num_actions": 4,
    }

    exec(build_source, namespace)
    model = namespace["build_model"](MAZE)

    assert model.input_shape == (None, 64)
    assert model.output_shape == (None, 4)
    assert len(model.layers) == 5


def test_seed_is_set_before_model_construction():
    training_source = next(
        source for source in notebook_code_cells() if "training_results = qtrain" in source
    )

    assert training_source.index("tf.random.set_seed(seed)") < training_source.index(
        "model = build_model(maze)"
    )
