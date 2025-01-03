import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
)


from marker.postprocessors.editor import load_editing_model
from marker.settings import settings
from surya.model.detection.model import load_model as load_detection_model
from surya.model.detection.model import load_processor as load_detection_processor
from surya.model.ordering.model import load_model as load_order_model
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from texify.model.model import load_model as load_texify_model
from texify.model.processor import load_processor as load_texify_processor


def setup_recognition_model(langs, device=None, dtype=None):
    if device:
        rec_model = load_recognition_model(device=device, dtype=dtype)
    else:
        rec_model = load_recognition_model()
    rec_processor = load_recognition_processor()
    rec_model.processor = rec_processor
    return rec_model


def setup_detection_model(device=None, dtype=None):
    if device:
        model = load_detection_model(device=device, dtype=dtype)
    else:
        model = load_detection_model()

    processor = load_detection_processor()
    model.processor = processor
    return model


def setup_texify_model(device=None, dtype=None):
    if device:
        texify_model = load_texify_model(checkpoint=settings.TEXIFY_MODEL_NAME, device=device, dtype=dtype)
    else:
        texify_model = load_texify_model(
            checkpoint=settings.TEXIFY_MODEL_NAME, device=settings.TORCH_DEVICE_MODEL, dtype=settings.TEXIFY_DTYPE
        )
    texify_processor = load_texify_processor()
    texify_model.processor = texify_processor
    return texify_model


def setup_layout_model(device=None, dtype=None):
    if device:
        model = load_detection_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT, device=device, dtype=dtype)
    else:
        model = load_detection_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_detection_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    model.processor = processor
    return model


def setup_order_model(device=None, dtype=None):
    if device:
        model = load_order_model(device=device, dtype=dtype)
    else:
        model = load_order_model()
    processor = load_order_processor()
    model.processor = processor
    return model


def _load_all_models(langs=None, device=None, dtype=None, force_load_ocr=False):
    if device is not None:
        assert dtype is not None, "Must provide dtype if device is provided"

    # langs is optional list of languages to prune from recognition MoE model
    detection = setup_detection_model(device, dtype)
    layout = setup_layout_model(device, dtype)
    order = setup_order_model(device, dtype)
    edit = load_editing_model(device, dtype)

    # Only load recognition model if we'll need it for all pdfs
    ocr = setup_recognition_model(langs, device, dtype)
    texify = setup_texify_model(device, dtype)
    model_lst = [texify, layout, order, edit, detection, ocr]
    return model_lst


from concurrent.futures import ThreadPoolExecutor, as_completed


def load_all_models(langs=None, device=None, dtype=None, force_load_ocr=False):
    if device is not None:
        assert dtype is not None, "Must provide dtype if device is provided"

    def load_models_concurrently(load_functions_map: dict) -> dict:
        model_id_to_model = {}
        with ThreadPoolExecutor(max_workers=len(load_functions_map)) as executor:
            future_to_model_id = {executor.submit(load_fn): model_id for model_id, load_fn in load_functions_map.items()}
            for future in as_completed(future_to_model_id.keys()):
                model_id_to_model[future_to_model_id[future]] = future.result()
        return model_id_to_model

    # Define load functions for each model
    load_functions = {
        "detection": lambda: setup_detection_model(device, dtype),
        "layout": lambda: setup_layout_model(device, dtype),
        "order": lambda: setup_order_model(device, dtype),
        "edit": lambda: load_editing_model(device, dtype),
        "ocr": lambda: setup_recognition_model(langs, device, dtype),
        "texify": lambda: setup_texify_model(device, dtype),
    }

    # Load models concurrently
    models = load_models_concurrently(load_functions)

    def convert_to_half(model):
        for param in model.parameters():
            param.data = param.data.half()
            if param._grad is not None:
                param._grad.data = param._grad.data.half()
        return model

    # Return the models in the original order
    model_lst = [models["texify"], models["layout"], models["order"], models["edit"], models["detection"], models["ocr"]]
    model_lst = [convert_to_half(model) if model is not None else model for model in model_lst]
    return model_lst
