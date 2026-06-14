import faiss
import torch
from PIL import Image
from py_trees.common import Status
from transformers import CLIPModel, CLIPProcessor
from typing_extensions import List

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis


class ClipAnnotator(BaseAnnotator):
    """A simple CLIP annotator that uses a given vocabulary to annotate all incoming object hypothesis."""

    def __init__(self, name: str = "ClipAnnotator") -> None:
        super().__init__(name)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """Torch device to be used by the annotator."""

        self.model_name: str = "openai/clip-vit-large-patch14"

        self.clip_model = CLIPModel.from_pretrained(self.model_name)
        """CLIP model used for feature extraction."""

        self.clip_preprocess = CLIPProcessor.from_pretrained(self.model_name)
        """CLIP preprocessor used for input preprocessing."""

        self.clip_tokenizer = self.clip_preprocess.tokenizer
        """CLIP tokenizer used for text preprocessing."""

        self.vocabulary_template: str = "A photo of a {}"

        self.vocabulary: List[str] = [
            "Milk Container",
            "Box of Cornflakes",
            "Pancake Batter",
            "Instant Coffee",
            "Crepe Pan",
            "Dish Soap",
        ]
        """The vocabulary used for object classification"""

        self.vocabulary_index: faiss.IndexFlatIP = faiss.IndexFlatIP(
            self.clip_model.config.projection_dim
        )
        """A faiss index for text features used in similarity search."""

        with torch.no_grad():
            templated_texts = [
                self.vocabulary_template.format(name) for name in self.vocabulary
            ]
            inputs = self.clip_tokenizer(
                templated_texts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            text_features = self.clip_model.get_text_features(**inputs).pooler_output
            text_features /= torch.norm(text_features, p=2, dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()
        self.vocabulary_index.add(text_features)

    def update(self) -> Status:
        ohs: List[ObjectHypothesis] = self.get_cas().filter_annotations_by_type(
            ObjectHypothesis
        )
        if len(ohs) == 0:
            return Status.SUCCESS

        color_image = self.get_cas().get(CASViews.COLOR_IMAGE).copy()
        color_pil = Image.fromarray(color_image)

        # Create per-object image crop
        cropped_images = [color_pil.crop(oh.roi.roi.get_corner_points()) for oh in ohs]

        with torch.no_grad():
            inputs = self.clip_preprocess(
                images=cropped_images, return_tensors="pt"
            ).to(self.device)
            crop_features = self.clip_model.get_image_features(**inputs).pooler_output
            crop_features /= torch.norm(crop_features, p=2, dim=-1, keepdim=True)
            crop_features = crop_features.cpu().numpy()

        data, idx = self.vocabulary_index.search(crop_features, k=1)
        best_indices = idx[:, 0].astype(int)
        best_classes = [self.vocabulary[i] for i in best_indices]
        best_similarities = data[:, 0] * 100.0

        for oh, best_idx, best_class, best_similarity in zip(
            ohs, best_indices, best_classes, best_similarities
        ):
            oh.annotations.append(
                Classification(
                    source=self.name,
                    class_id=best_idx,
                    classname=best_class,
                    confidence=best_similarity,
                )
            )

            self.rk_logger.info(
                f"Classified oh {oh.id} as {best_class} ({best_similarity:.2f})"
            )

        return Status.SUCCESS
