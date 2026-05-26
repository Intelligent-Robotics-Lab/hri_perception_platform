from app.registry.perception_registry import PerceptionRegistry

TASK_NAME = "gesture_recognition"
registry = PerceptionRegistry()


def get_active_gesture_model():
    return registry.get_active_backend_name(TASK_NAME)


def get_active_gesture_url():
    return registry.get_active_backend_url(TASK_NAME)