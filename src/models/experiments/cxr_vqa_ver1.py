import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, CLIPTextModel

from src.models.losses import CELoss
from src.models.modules import CxrResNet, ProjectionHead


class CxrVQA(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.image_encoder = CxrResNet()
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            truncation_side="left",
            model_max_length=77,
        )

        self.criterion = CELoss()

        self.text_projection = ProjectionHead(
            embedding_dim=self.text_encoder.config.hidden_size,
            projection_dim=256,
        )
        self.image_projection = ProjectionHead(
            embedding_dim=512 * 3 * 3, projection_dim=256
        )

        self.temperature = temperature

    def forward(self, img, text):
        # [batch_size, 3 * 3 * 512]
        img_features = self.image_encoder(img).flatten(start_dim=1)

        inputs = self.tokenizer(
            text,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=77,
        ).to("cuda")
        outputs = self.text_encoder(**inputs)
        # [batch_size, 512]
        text_features = outputs.pooler_output

        # [batch_size, 256] for emb
        text_emb = self.text_projection(text_features)
        img_emb = self.image_projection(img_features)

        # Calculate the similarity
        logits = (text_emb @ img_emb.T) / self.temperature
        imgs_sim = img_emb @ img_emb.T
        texts_sim = text_emb @ text_emb.T

        targets = nn.Softmax(dim=-1)(
            (texts_sim + imgs_sim) / (2.0 * self.temperature)
        )
        texts_loss = self.criterion(logits, targets)
        images_loss = self.criterion(logits.T, targets.T)

        loss = (texts_loss + images_loss) / 2.0

        return loss.mean()
