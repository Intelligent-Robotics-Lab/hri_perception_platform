from app.registry.perception_registry import PerceptionRegistry

TASK_NAME = "speech_recognition"
registry = PerceptionRegistry()


def get_active_asr_model():
    return registry.get_active_backend_name(TASK_NAME)


def get_active_asr_url():
    return registry.get_active_backend_url(TASK_NAME)