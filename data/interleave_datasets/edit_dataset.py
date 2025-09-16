# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import math
import random
from typing import Iterable, List, Optional

from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class UnifiedEditIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    reference_keys: List[str] = [
        "reference_image_list",
        "reference_list",
        "reference_images",
        "reference_image",
        "ref_image",
        "ref_image_list",
        "conditioning_image_list",
        "conditioning_images",
        "cond_image_list",
        "cond_images",
    ]

    source_keys: List[str] = [
        "source_image",
        "input_image",
        "input_images",
        "start_image",
    ]

    target_keys: List[str] = [
        "target_image",
        "edited_image",
        "output_image",
        "end_image",
    ]

    def parse_row(self, row):
        reference_images = self._extract_reference_images(row)
        if reference_images:
            data = self._parse_reference_guided_edit(row, reference_images)
            if data:
                return data
        return self._parse_instruction_guided_edit(row)

    # ------------------------------------------------------------------
    # Helpers for reference-image-driven editing
    # ------------------------------------------------------------------
    def _parse_reference_guided_edit(self, row, reference_images):
        image_list = self._safe_get(row, "image_list")

        source_bytes = self._extract_primary_image(row, self.source_keys, image_list, take_last=False)
        target_bytes = self._extract_primary_image(row, self.target_keys, image_list, take_last=True)

        if source_bytes is None or target_bytes is None:
            return {}

        data = self._init_data()
        data = self._add_image(
            data,
            self._value_to_pil(source_bytes),
            need_loss=False,
            need_vae=True,
            need_vit=True,
            enable_cfg=False,
        )

        for reference in reference_images:
            data = self._add_image(
                data,
                self._value_to_pil(reference),
                need_loss=False,
                need_vae=True,
                need_vit=True,
                enable_cfg=False,
            )

        data = self._add_image(
            data,
            self._value_to_pil(target_bytes),
            need_loss=True,
            need_vae=False,
            need_vit=False,
        )
        return data

    def _extract_reference_images(self, row) -> List[bytes]:
        for key in self.reference_keys:
            value = self._safe_get(row, key)
            images = self._normalize_to_list(value)
            if images:
                return images
        return []

    def _extract_primary_image(self, row, keys: Iterable[str], fallback, take_last: bool) -> Optional[bytes]:
        for key in keys:
            value = self._safe_get(row, key)
            images = self._normalize_to_list(value)
            if images:
                return images[-1] if take_last else images[0]

        images = self._normalize_to_list(fallback)
        if images:
            return images[-1] if take_last else images[0]
        return None

    @staticmethod
    def _safe_get(row, key):
        if hasattr(row, "get"):
            return row.get(key, None)
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            return None

    @staticmethod
    def _normalize_to_list(value) -> List[bytes]:
        if value is None:
            return []
        if isinstance(value, str):
            return []

        def _to_bytes(item):
            if item is None or (isinstance(item, float) and math.isnan(item)):
                return None
            if isinstance(item, bytes):
                return item
            if isinstance(item, bytearray):
                return bytes(item)
            if isinstance(item, memoryview):
                return item.tobytes()
            return item

        if isinstance(value, (bytes, bytearray, memoryview)):
            normalized = _to_bytes(value)
            return [normalized] if normalized is not None else []

        if isinstance(value, Iterable) and not isinstance(value, (str, Image.Image)):
            items: List[bytes] = []
            for item in value:
                converted = _to_bytes(item)
                if converted is not None:
                    items.append(converted)
            return items

        converted = _to_bytes(value)
        return [converted] if converted is not None else []

    # ------------------------------------------------------------------
    # Existing instruction-driven editing path
    # ------------------------------------------------------------------
    def _parse_instruction_guided_edit(self, row):
        image_list = self._safe_get(row, "image_list")
        instruction_list = self._safe_get(row, "instruction_list")

        if not image_list or not instruction_list:
            return {}

        image_num = len(image_list)
        if image_num < 2:
            return {}

        start_idx = random.choice(range(image_num - 1))
        max_end = min(start_idx + 3, image_num)
        end_idx = random.choice(range(start_idx + 1, max_end))

        data = self._init_data()
        data = self._add_image(
            data,
            self._value_to_pil(image_list[start_idx]),
            need_loss=False,
            need_vae=True,
            need_vit=True,
        )

        if end_idx - start_idx > 1 and random.random() < 0.5: # concat multiple insturction
            if end_idx == image_num - 1:
                end_idx -= 1

            instruction = ""
            for idx in range(start_idx + 1, end_idx + 1):
                instruction += random.choice(instruction_list[idx-1]) + ". "
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            data = self._add_image(
                data,
                self._value_to_pil(image_list[end_idx]),
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
        else:
            for idx in range(start_idx + 1, end_idx + 1):
                instruction = random.choice(instruction_list[idx-1])
                data = self._add_text(data, instruction, need_loss=False)
                if idx != end_idx:
                    data = self._add_image(
                        data,
                        self._value_to_pil(image_list[idx]),
                        need_loss=True,
                        need_vae=True,
                        need_vit=True,
                    )
                else:
                    data = self._add_image(
                        data,
                        self._value_to_pil(image_list[idx]),
                        need_loss=True,
                        need_vae=False,
                        need_vit=False,
                    )
        return data

    @staticmethod
    def _value_to_pil(value):
        if isinstance(value, Image.Image):
            image = value
        else:
            image = Image.open(io.BytesIO(value))
        return pil_img2rgb(image)
