from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
# print(sys.path)
from libero.libero import get_libero_path


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
for i in range(task_suite.n_tasks):
    task_id = i
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    env.set_init_state(init_states[init_state_id])

    dummy_action = [0.1, 0, 0, 0, 0, 0, 0,]
    for step in range(10):
        obs, reward, done, info = env.step(dummy_action)
    plt.imsave(f"task-{task_id}.png", np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        # print(obs['agentview_image'].shape)
        # print(obs["robot0_joint_pos"])
    env.close()

