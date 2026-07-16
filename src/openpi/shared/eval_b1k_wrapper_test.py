import numpy as np

from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper


class _FakePolicy:
    def __init__(self):
        self.infer_count = 0

    def infer(self, _obs):
        self.infer_count += 1
        return {"actions": np.zeros((32, 23), dtype=np.float32)}

    def reset(self):
        return None


def _observation():
    return {
        "robot_r1::proprio": np.zeros(61, dtype=np.float32),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros((16, 16, 4), dtype=np.uint8),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros((16, 16, 4), dtype=np.uint8),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.zeros((16, 16, 4), dtype=np.uint8),
    }


def test_receding_horizon_reports_current_observation_only_on_replan():
    policy = _FakePolicy()
    wrapper = B1KPolicyWrapper(
        policy=policy,
        robot="b1k/R1Pro",
        text_prompt="Turn on the radio.",
        control_mode="receding_horizon",
        action_horizon=2,
        max_len=32,
    )

    wrapper.act(_observation())
    first = dict(wrapper.last_action_provenance)
    wrapper.act(_observation())
    second = dict(wrapper.last_action_provenance)
    wrapper.act(_observation())
    third = dict(wrapper.last_action_provenance)

    assert policy.infer_count == 2
    assert (first["status"], first["request_index"], first["source_request_index"], first["plan_id"]) == (
        "current_observation_used",
        0,
        0,
        0,
    )
    assert (second["status"], second["request_index"], second["source_request_index"], second["plan_id"]) == (
        "current_observation_not_used",
        1,
        0,
        0,
    )
    assert (third["status"], third["request_index"], third["source_request_index"], third["plan_id"]) == (
        "current_observation_used",
        2,
        2,
        1,
    )
    assert [first["action_index_in_plan"], second["action_index_in_plan"], third["action_index_in_plan"]] == [
        0,
        1,
        0,
    ]
