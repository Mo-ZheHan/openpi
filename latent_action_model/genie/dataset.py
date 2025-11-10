"""
LeRobot-based dataset implementation for latent action model training.

This module provides a Lightning DataModule that loads and mixes multiple LeRobot datasets
with sampling weights, image augmentation, and per-dataset frame interval adjustments.

Uses StreamingLeRobotDataset for memory-efficient streaming and proper interleaved sampling.
"""

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# Datasets with lower frequency (3Hz -> 5Hz control frequency)
DATASETS_WITH_LOWER_FREQUENCY = [
    'fractal20220817_data',
    'toto',
    'berkeley_autolab_ur5',
    'nyu_franka_play_dataset_converted_externally_to_rlds',
    'ucsd_kitchen_dataset_converted_externally_to_rlds',
    'dlr_edan_shared_control_converted_externally_to_rlds',
    'dobbe',
]

# Datasets with higher frequency (15Hz -> 30Hz control frequency)
DATASETS_WITH_HIGHER_FREQUENCY = [
    'utaustin_mutex',
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
    'austin_sailor_dataset_converted_externally_to_rlds',
    'viola',
    'droid',
]


# Dataset-specific image key mappings
DATASET_IMAGE_KEYS = {
    'droid': 'observation.images.exterior_1_left',
}


def get_image_key_for_dataset(dataset_name: str) -> str:
    for pattern, image_key in DATASET_IMAGE_KEYS.items():
        if pattern in dataset_name.lower():
            return image_key
    raise ValueError(f"No image key found for dataset {dataset_name}")


class ImageAugmentation:
    """Image augmentation for training."""

    def __init__(
        self,
        random_resized_crop: Optional[Dict] = None,
        random_brightness: Optional[List[float]] = None,
        random_contrast: Optional[List[float]] = None,
        random_saturation: Optional[List[float]] = None,
        random_hue: Optional[List[float]] = None,
    ):
        self.transforms_list = []

        if random_resized_crop:
            scale = random_resized_crop.get('scale', [0.9, 0.9])
            ratio = random_resized_crop.get('ratio', [1.0, 1.0])
            self.transforms_list.append(
                transforms.RandomResizedCrop(
                    size=224,  # Will be overridden by actual resolution
                    scale=tuple(scale),
                    ratio=tuple(ratio),
                )
            )

        color_jitter_params = {}
        if random_brightness:
            color_jitter_params['brightness'] = random_brightness
        if random_contrast:
            color_jitter_params['contrast'] = random_contrast
        if random_saturation:
            color_jitter_params['saturation'] = random_saturation
        if random_hue:
            color_jitter_params['hue'] = random_hue

        if color_jitter_params:
            self.transforms_list.append(transforms.ColorJitter(**color_jitter_params))

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply augmentations to image."""
        for t in self.transforms_list:
            img = t(img)
        return img


def get_window_size(dataset_name: str, training_phase: str = 'lam') -> int:
    """Determine window size based on dataset characteristics."""
    # Special case for ego4d
    if 'ego4d' in dataset_name.lower():
        return 2

    # Low frequency datasets
    if any(ds in dataset_name for ds in DATASETS_WITH_LOWER_FREQUENCY):
        if training_phase == 'lam':
            return random.randint(3, 5)
        return 3

    # High frequency datasets
    if any(ds in dataset_name for ds in DATASETS_WITH_HIGHER_FREQUENCY):
        if training_phase == 'lam':
            return random.randint(15, 20)
        return 15

    # Default window size
    return 10


def extract_image_from_sample(sample: Dict[str, Any], image_key: str = 'observation.image') -> Image.Image:
    """
    Extract image from LeRobot sample and convert to PIL Image.

    Expects image_key with tensor in [C, H, W] format.
    Will fail explicitly if data format is unexpected.

    Args:
        sample: Sample dictionary from LeRobot dataset
        image_key: Key to use for extracting image (default: 'observation.image')
    """
    # Expect specific key - fail if not found
    if image_key not in sample:
        raise KeyError(
            f"Expected '{image_key}' key in sample. Available keys: {list(sample.keys())}"
        )

    img_tensor = sample[image_key]

    # Convert tensor/array to numpy
    if isinstance(img_tensor, torch.Tensor):
        img_array = img_tensor.cpu().numpy()
    elif isinstance(img_tensor, np.ndarray):
        img_array = img_tensor
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(img_tensor)}")

    # Expect 3D tensor [C, H, W]
    if img_array.ndim != 3:
        raise ValueError(f"Expected 3D image tensor [C, H, W], got shape {img_array.shape}")

    # Convert from [C, H, W] to [H, W, C]
    if img_array.shape[0] in [1, 3, 4]:  # Channel-first format
        img_array = np.transpose(img_array, (1, 2, 0))
    else:
        raise ValueError(
            f"Unexpected channel dimension. Expected channels in [1, 3, 4], got shape {img_array.shape}"
        )

    # Normalize to [0, 255] uint8 range
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    # Expect RGB images (3 channels)
    if img_array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {img_array.shape[-1]} channels")

    return Image.fromarray(img_array)


class InterleavedStreamingDataset(IterableDataset):
    """
    Interleaved dataset that samples from multiple StreamingLeRobotDatasets.

    This implements proper interleaved sampling similar to RLDS make_interleaved_dataset,
    where we randomly choose which dataset to sample from based on weights at each step.

    Key features:
    - True interleaved sampling (not sequential)
    - Memory-efficient streaming
    - Per-dataset shuffling
    - Weighted random dataset selection
    """

    def __init__(
        self,
        dataset_specs: List[Tuple[str, float]],  # [(repo_id, weight), ...]
        resolution: Tuple[int, int] = (224, 224),
        image_aug: bool = False,
        training_phase: str = 'lam',
        buffer_size: int = 1000,
        seed: int = 42,
        root: Optional[str] = None,
    ):
        """
        Args:
            dataset_specs: List of (repo_id, sampling_weight) tuples
            resolution: Target image resolution
            image_aug: Whether to apply image augmentation
            training_phase: Training phase ('lam' or 'post-training')
            buffer_size: Buffer size for each StreamingLeRobotDataset
            seed: Random seed for reproducibility
            root: Root directory for local datasets (optional)
        """
        super().__init__()
        self.dataset_specs = dataset_specs
        self.resolution = resolution
        self.image_aug_enabled = image_aug
        self.training_phase = training_phase
        self.buffer_size = buffer_size
        self.seed = seed
        self.root = root

        # Normalize weights
        total_weight = sum(weight for _, weight in dataset_specs)
        self.weights = [weight / total_weight for _, weight in dataset_specs]
        self.repo_ids = [repo_id for repo_id, _ in dataset_specs]

        # Get image keys for each dataset
        self.image_keys = [get_image_key_for_dataset(repo_id) for repo_id in self.repo_ids]

        # Image transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)

        # Image augmentation
        self.image_aug = None
        if image_aug:
            self.image_aug = ImageAugmentation(
                random_resized_crop={'scale': [0.9, 0.9], 'ratio': [1.0, 1.0]},
                random_brightness=[0.8, 1.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[-0.05, 0.05],
            )

        print(f"Interleaved dataset created with {len(self.repo_ids)} datasets")
        for repo_id, weight, image_key in zip(self.repo_ids, self.weights, self.image_keys):
            print(f"  - {repo_id}: weight={weight:.3f}, image_key={image_key}")

    def __iter__(self):
        """
        Create iterators for all datasets and interleave them.

        This implements the core interleaving logic: randomly select which dataset
        to sample from based on weights at each iteration step.
        """
        # Get worker info for distributed training
        worker_info = get_worker_info()
        if worker_info is not None:
            # In multi-worker setting, each worker gets a different seed
            worker_seed = self.seed + worker_info.id
        else:
            worker_seed = self.seed

        # Create RNG for this worker
        rng = np.random.default_rng(worker_seed)

        # Initialize all streaming datasets
        datasets = []
        iterators = []
        dataset_image_keys = []  # Track image keys for each dataset

        for repo_id, image_key in zip(self.repo_ids, self.image_keys):
            # Determine window size for this dataset
            window_size = get_window_size(repo_id, self.training_phase)

            # Create delta timestamps for frame pairs
            # We want [initial_frame, target_frame] where target is window_size-1 steps ahead
            delta_timestamps = {}

            # Construct full path if root is provided
            dataset_root = None
            if self.root:
                from pathlib import Path
                dataset_root = Path(self.root) / repo_id

            # For the primary image, we want current frame (0.0) and future frame
            # LeRobot uses fps from metadata to convert frame indices to time
            meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
            fps = meta.fps
            # Convert frame offset to seconds
            delta_t = (window_size - 1) / fps
            delta_timestamps = {
                # Get current and future frame using the correct image key for this dataset
                image_key: [0.0, delta_t],
            }

            # Create streaming dataset with shuffle enabled
            dataset = StreamingLeRobotDataset(
                repo_id=repo_id,
                root=dataset_root,
                delta_timestamps=delta_timestamps,
                streaming=True,
                buffer_size=self.buffer_size,
                seed=worker_seed,
                shuffle=True,  # Enable shuffling within dataset
            )

            datasets.append(dataset)
            iterators.append(iter(dataset))
            dataset_image_keys.append(image_key)

        # Interleaved sampling loop
        while True:
            # Randomly select which dataset to sample from based on weights
            dataset_idx = rng.choice(len(datasets), p=self.weights)
            dataset_name = self.repo_ids[dataset_idx]
            image_key = dataset_image_keys[dataset_idx]

            # Get next sample from selected dataset
            try:
                sample = next(iterators[dataset_idx])
            except StopIteration:
                # Dataset exhausted all shards, create a new iterator to loop
                print(f"Dataset {dataset_name} exhausted, restarting iterator")
                iterators[dataset_idx] = iter(datasets[dataset_idx])
                sample = next(iterators[dataset_idx])

            # Process the sample
            yield self._process_sample(sample, dataset_name, image_key)

    def _process_sample(self, sample: Dict[str, Any], dataset_name: str, image_key: str) -> Dict[str, Any]:
        """
        Process a sample from LeRobot dataset into the format needed for training.

        Args:
            sample: Sample from StreamingLeRobotDataset (contains stacked frames from delta_timestamps)
            dataset_name: Name of the source dataset
            image_key: The image key used for this dataset (e.g., 'observation.image')

        Returns:
            Dictionary with processed frames and actions
        """
        # Extract images - LeRobot with delta_timestamps returns stacked frames
        # Expected: sample[image_key] with shape [2, C, H, W] for initial and target frames
        if image_key not in sample:
            raise KeyError(
                f"Expected '{image_key}' in sample from {dataset_name}. "
                f"Available keys: {list(sample.keys())}"
            )

        img_data = sample[image_key]

        if not isinstance(img_data, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor for '{image_key}', got {type(img_data)}"
            )

        # Expect exactly 2 frames (initial and target)
        if img_data.dim() != 4 or img_data.shape[0] != 2:
            raise ValueError(
                f"Expected image data shape [2, C, H, W], got {img_data.shape}. "
                f"Check delta_timestamps configuration for dataset {dataset_name}."
            )

        # Extract initial and target frames
        initial_img = extract_image_from_sample({image_key: img_data[0]}, image_key)
        target_img = extract_image_from_sample({image_key: img_data[1]}, image_key)

        # Apply image augmentation
        if self.image_aug is not None:
            initial_img = self.image_aug(initial_img)
            target_img = self.image_aug(target_img)

        # Resize and convert to tensor
        initial_img = self.resize(initial_img)
        target_img = self.resize(target_img)

        initial_pixel_values = self.to_tensor(initial_img)
        target_pixel_values = self.to_tensor(target_img)

        # Extract action - must be present
        if 'action' not in sample:
            raise KeyError(
                f"Missing 'action' key in sample from {dataset_name}. "
                f"Available keys: {list(sample.keys())}"
            )

        action = sample['action']
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, np.ndarray):
            pass  # Already numpy
        else:
            raise TypeError(f"Expected action as Tensor or ndarray, got {type(action)}")

        # Extract language instruction (optional)
        task_instruction = sample.get('language_instruction', '')
        if isinstance(task_instruction, (bytes, bytearray)):
            task_instruction = task_instruction.decode('utf-8')
        elif not isinstance(task_instruction, str):
            task_instruction = str(task_instruction)

        return {
            'initial_pixel_values': initial_pixel_values,
            'target_pixel_values': target_pixel_values,
            'action': action,
            'task_instruction': task_instruction,
            'dataset_name': dataset_name,
        }


class CollatorForLatentAction:
    """Collator for batching frame pairs for latent action model training."""

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Collate samples into a batch.

        Args:
            instances: List of samples from the dataset

        Returns:
            Batched dictionary with:
                - videos: [B, 2, 3, H, W] - stacked initial and target frames
                - action: [B, T, action_dim] - action sequences
                - dataset_names: List of dataset names (optional)
        """
        # Stack images
        initial_pixel_values = torch.stack([inst['initial_pixel_values'] for inst in instances])
        target_pixel_values = torch.stack([inst['target_pixel_values'] for inst in instances])

        # Stack as [B, 2, 3, H, W]
        videos = torch.stack([initial_pixel_values, target_pixel_values], dim=1)

        # Stack actions - expect all to be numpy arrays
        actions = [torch.from_numpy(inst['action']) for inst in instances]

        # Verify all actions have the same shape
        first_shape = actions[0].shape
        for i, action in enumerate(actions[1:], 1):
            if action.shape != first_shape:
                raise ValueError(
                    f"Action shape mismatch in batch: "
                    f"instance 0 has shape {first_shape}, instance {i} has shape {action.shape}. "
                    f"All actions in a batch must have the same shape."
                )

        action = torch.stack(actions)

        # Collect dataset names
        dataset_names = [inst['dataset_name'] for inst in instances]

        return {
            'videos': videos,
            'action': action,
            'dataset_names': dataset_names,
        }


class LightningLeRobotDataset(LightningDataModule):
    """
    Lightning DataModule for training with mixed LeRobot datasets.

    This module handles loading multiple LeRobot datasets with sampling weights,
    applying image augmentation, and managing frame interval adjustments based on
    dataset characteristics.

    Uses StreamingLeRobotDataset for memory-efficient streaming and proper
    interleaved sampling similar to RLDS make_interleaved_dataset.
    """

    def __init__(
        self,
        dataset_mix: List[Tuple[str, float]],  # [(repo_id, weight), ...]
        batch_size: int = 64,
        resolution: int = 224,
        image_aug: bool = True,
        buffer_size: int = 1000,
        seed: int = 42,
        root: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            dataset_mix: List of (dataset_repo_id, sampling_weight) tuples
            batch_size: Batch size for training
            resolution: Image resolution (assumed square)
            image_aug: Whether to apply image augmentation during training
            buffer_size: Buffer size for each StreamingLeRobotDataset
            seed: Random seed for reproducibility
            root: Root directory for local datasets (optional)
        """
        super().__init__()

        self.dataset_mix = dataset_mix
        self.batch_size = batch_size
        self.resolution = (resolution, resolution)
        self.image_aug = image_aug
        self.buffer_size = buffer_size
        self.seed = seed
        self.root = root

        self.train_dataset = None

        self.collate_fn = CollatorForLatentAction()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        """Setup datasets for training only."""
        if stage == "fit":
            # Create training dataset with augmentation
            self.train_dataset = InterleavedStreamingDataset(
                dataset_specs=self.dataset_mix,
                resolution=self.resolution,
                image_aug=self.image_aug,
                training_phase='lam',
                buffer_size=self.buffer_size,
                seed=self.seed,
                root=self.root,
            )

            print("Training dataset initialized (streaming mode) - all data used for training")

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,  # IterableDataset with internal streaming
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
