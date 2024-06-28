"""This module provides mixin functionality for the Albumentations library.
It includes utility functions and classes to enhance the core capabilities.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable

from albumentations.core.serialization import load as load_transform
from albumentations.core.serialization import save as save_transform

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError, SoftTemporaryDirectory

    is_huggingface_hub_available = True
except ImportError:
    is_huggingface_hub_available = False

logger = logging.getLogger(__name__)


def require_huggingface_hub(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_huggingface_hub_available:
            raise ImportError(
                f"You need to install `huggingface_hub` to use {func.__name__}. "
                "Run `pip install huggingface_hub`, or `pip install albumentations[hub]`.",
            )
        return func(*args, **kwargs)

    return wrapper


class HubMixin:
    _CONFIG_KEYS = ("train", "eval")
    _CONFIG_FILE_NAME_TEMPLATE = "albumentations_config_{}.json"

    def _save_pretrained(self, save_directory: str | Path, filename: str) -> Path:
        """Save the transform to a specified directory.

        Args:
            save_directory (Union[str, Path]):
                Directory where the transform will be saved.
            filename (str):
                Name of the file to save the transform.

        Returns:
            Path: Path to the saved transform file.
        """
        # create save directory and path
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_path = save_directory / filename

        # save transforms
        save_transform(self, save_path, data_format="json")  # type: ignore[arg-type]

        return save_path

    @classmethod
    def _from_pretrained(cls, save_directory: str | Path, filename: str) -> object:
        """Load a transform from a specified directory.

        Args:
            save_directory (Union[str, Path]):
                Directory from where the transform will be loaded.
            filename (str):
                Name of the file to load the transform from.

        Returns:
            A.Compose: Loaded transform.
        """
        save_path = Path(save_directory) / filename
        return load_transform(save_path, data_format="json")

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        key: str = "eval",
        allow_custom_keys: bool = False,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        **push_to_hub_kwargs: Any,
    ) -> str | None:
        """Save the transform and optionally push it to the Huggingface Hub.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the transform configuration will be saved.
            key (`str`, *optional*):
                Key to identify the configuration type, one of ["train", "eval"]. Defaults to "eval".
            allow_custom_keys (`bool`, *optional*):
                Allow custom keys for the configuration. Defaults to False.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your transform to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            push_to_hub_kwargs:
                Additional key word arguments passed along to the [`push_to_hub`] method.

        Returns:
            `str` or `None`: url of the commit on the Hub if `push_to_hub=True`, `None` otherwise.
        """
        if not allow_custom_keys and key not in self._CONFIG_KEYS:
            raise ValueError(
                f"Invalid key: `{key}`. Please use key from {self._CONFIG_KEYS} keys for upload. "
                "If you want to use a custom key, set `allow_custom_keys=True`.",
            )

        # save model transforms
        filename = self._CONFIG_FILE_NAME_TEMPLATE.format(key)
        self._save_pretrained(save_directory, filename)

        # push to the Hub if required
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()  # soft-copy to avoid mutating input
            if repo_id is None:
                repo_id = Path(save_directory).name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, key=key, **kwargs)
        return None

    @classmethod
    def from_pretrained(
        cls: Any,
        directory_or_repo_id: str | Path,
        *,
        key: str = "eval",
        force_download: bool = False,
        proxies: dict[str, str] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> object:
        """Load a transform from the Huggingface Hub or a local directory.

        Args:
            directory_or_repo_id (`str`, `Path`):
                - Either the `repo_id` (string) of a repo with hosted transform on the Hub, e.g. `qubvel-hf/albu`.
                - Or a path to a `directory` containing transform config saved using
                    [`~albumentations.Compose.save_pretrained`], e.g., `../path/to/my_directory/`.
            key (`str`, *optional*):
                Key to identify the configuration type, one of ["train", "eval"]. Defaults to "eval".
            revision (`str`, *optional*):
                Revision of the repo on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the transform configuration files from the Hub, overriding
                the existing cache.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on every request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        """
        filename = cls._CONFIG_FILE_NAME_TEMPLATE.format(key)
        directory_or_repo_id = Path(directory_or_repo_id)
        transform = None

        # check if the file is already present locally
        if directory_or_repo_id.is_dir():
            if filename in os.listdir(directory_or_repo_id):
                transform = cls._from_pretrained(save_directory=directory_or_repo_id, filename=filename)
            elif is_huggingface_hub_available:
                logging.info(
                    f"{filename} not found in {Path(directory_or_repo_id).resolve()}, trying to load from the Hub.",
                )
            else:
                raise FileNotFoundError(
                    f"{filename} not found in {Path(directory_or_repo_id).resolve()}."
                    " Please install `huggingface_hub` to load from the Hub.",
                )
        if transform is not None:
            return transform

        # download the file from the Hub
        try:
            config_file = hf_hub_download(
                repo_id=directory_or_repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )
            directory, filename = Path(config_file).parent, Path(config_file).name
            return cls._from_pretrained(save_directory=directory, filename=filename)

        except HfHubHTTPError as e:
            raise HfHubHTTPError(f"{filename} not found on the HuggingFace Hub") from e

    @require_huggingface_hub
    def push_to_hub(
        self,
        repo_id: str,
        *,
        key: str = "eval",
        allow_custom_keys: bool = False,
        commit_message: str = "Push transform using huggingface_hub.",
        private: bool = False,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
    ) -> str:
        """Push the transform to the Huggingface Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
        `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
        details.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            key (`str`, *optional*):
                Key to identify the configuration type, one of ["train", "eval"]. Defaults to "eval".
            allow_custom_keys (`bool`, *optional*):
                Allow custom keys for the configuration. Defaults to False.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            branch (`str`, *optional*):
                The git branch on which to push the transform. This defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit. Defaults to `False`.

        Returns:
            The url of the commit of your transform in the given repository.
        """
        if not allow_custom_keys and key not in self._CONFIG_KEYS:
            raise ValueError(
                f"Invalid key: `{key}`. Please use key from {self._CONFIG_KEYS} keys for upload. "
                "If you still want to use a custom key, set `allow_custom_keys=True`.",
            )

        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            save_directory = Path(tmp) / repo_id
            filename = self._CONFIG_FILE_NAME_TEMPLATE.format(key)
            save_path = self._save_pretrained(save_directory, filename=filename)
            return api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=filename,
                repo_id=repo_id,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
            )
