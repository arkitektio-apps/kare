from arkitekt import register
from mikro.api.schema import (
    from_xarray,
    ExperimentFragment,
    ModelFragment,
    create_model,
    ModelKind,
    RepresentationFragment,
    links,
    get_context,
    LinkableModels,
    ContextFragment,
)
import xarray as xr
import numpy as np
from typing import Optional, List
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import numpy as np
import uuid
import shutil
from csbdeep.data import RawData, create_patches
from csbdeep.data import no_background_patches, norm_percentiles, sample_percentiles
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from arkitekt.tqdm import tqdm

@register()
def gpu_is_available() -> str:
    """Check GPU

    Check if the gpu is available

    """
    from tensorflow.python.client import device_lib
    return str(device_lib.list_local_devices())

@register()
def train_care_model(
    context: ContextFragment,
    epochs: int = 10,
    patches_per_image: int = 1024,
    validation_split: float = 0.1,
) -> ModelFragment:
    """Train Care Model

    Trains a care model according on a specific context.

    Args:
        context (ContextFragment): The context

    Returns:
        ModelFragment: The Model
    """

    training_data_id = f"context_data{context.id}"

    x = links(
        LinkableModels.GRUNNLAG_REPRESENTATION,
        LinkableModels.GRUNNLAG_REPRESENTATION,
        "gt",
        context=context,
    )

    X = [t.x.data.sel(t=0, c=0).compute() for t in x]
    Y = [t.y.data.sel(t=0, c=0).compute() for t in x]

    raw_data = RawData.from_arrays(X, Y, axes="ZYX")
    print(raw_data)

    X, Y, XY_axes = create_patches(
        raw_data=raw_data,
        patch_size=(16, 64, 64),
        n_patches_per_image=patches_per_image,
        save_file=f"data/{training_data_id}.npz",
    )

    (X, Y), (X_val, Y_val), axes = load_training_data(
        f"data/{training_data_id}.npz",
        validation_split=validation_split,
        verbose=True,
    )
    config = Config(axes)

    model = CARE(config, training_data_id, basedir=".trainedmodels")

    for i in tqdm(range(epochs)):
        model.train(X, Y, validation_data=(X_val, Y_val), epochs=1)

    archive = shutil.make_archive(
        "active_model", "zip", f".trainedmodels/{training_data_id}"
    )
    model = create_model(
        "active_model.zip",
        kind=ModelKind.TENSORFLOW,
        name=f"Care Model of {context.name}",
        contexts=[context],
    )

    shutil.rmtree(f"data")
    return model


@register()
def predict(
    model: ModelFragment, representation: RepresentationFragment
) -> RepresentationFragment:
    """Predict Care

    Use a care model and some images to generate images

    Args:
        model (ImageToImageModelFragment): The model
        representations (List[RepresentationFragment]): The images

    Returns:
        List[RepresentationFragment]: The predicted images
    """

    random_dir = str(uuid.uuid4())
    generated = []

    with model.data as f:
        shutil.unpack_archive(f, f".modelcache/{random_dir}")

        care_model = CARE(config=None, name=random_dir, basedir=".modelcache")
        restored = care_model.predict(
            representation.data.sel(c=0, t=0).data.compute(), "ZXY"
        )
        generated = from_xarray(
            restored,
            name=f"Care denoised of {representation.name}",
            tags=["denoised"],
            origins=[representation],
        )

    shutil.rmtree(f".modelcache/{random_dir}")
    return generated


