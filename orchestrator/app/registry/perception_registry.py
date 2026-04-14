from pathlib import Path
import yaml


CONFIG_PATH = Path("/app/app/config/perception_config.yaml")


class PerceptionRegistry:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self):
        with self.config_path.open("r") as f:
            return yaml.safe_load(f)

    def get_task_config(self, task_name: str):
        tasks = self._config.get("tasks", {})
        if task_name not in tasks:
            raise ValueError(f"Unknown task: {task_name}")
        return tasks[task_name]

    def get_active_backend_name(self, task_name: str):
        task_cfg = self.get_task_config(task_name)
        backend = task_cfg.get("active_backend")
        if not backend:
            raise ValueError(f"No active backend configured for task: {task_name}")
        return backend

    def get_active_backend_url(self, task_name: str):
        task_cfg = self.get_task_config(task_name)
        backend_name = self.get_active_backend_name(task_name)
        backends = task_cfg.get("backends", {})
        if backend_name not in backends:
            raise ValueError(f"Backend '{backend_name}' not defined for task '{task_name}'")
        return backends[backend_name]["url"]