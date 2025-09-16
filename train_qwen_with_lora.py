import os
import json
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from datasets import Dataset
from pathlib import Path
import logging
import warnings
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
REGIONAL_CHARACTERS = [
    "чебурашка", "колобок", "алеша попович", "крокодил_гена" 
]

@dataclass
class Config:
    model_name: str = MODEL_NAME
    output_dir: str = "./results/qwen-vl-regional"
    train_jsonl: str = "./data/annotation.json"
    image_root: Optional[str] = str(Path(__file__).resolve().parent)
    
    val_size: float = 0.2
    seed: int = 42
    
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    num_train_epochs: int = 30
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    warmup_steps: int = 3
    logging_steps: int = 1
    save_steps: int = 10
    max_length: int = 128
    eval_steps: int = 10
    
    fp16: bool = False
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False


@dataclass
class Sample:
    image_path: str
    question: str
    answer: str


class JsonlVLDataset:
    def __init__(
        self,
        jsonl_path: str,
        processor: AutoProcessor,
        image_root: Optional[str] = None,
        max_length: int = 128
    ) -> None:
        self.samples: List[Sample] = []
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length
        
        self._load_samples(jsonl_path)
        logger.info(f"Загружено {len(self.samples)} образцов для обучения")

    def _load_samples(self, jsonl_path: str) -> None:
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Файл {jsonl_path} не найден!")

        if jsonl_path.lower().endswith(".json"):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Не удалось распарсить JSON {jsonl_path}: {e}")

                if isinstance(data, dict):
                    records = (
                        data.get("data")
                        or data.get("annotations")
                        or data.get("items")
                        or []
                    )
                elif isinstance(data, list):
                    records = data
                else:
                    raise ValueError("Ожидался массив или объект-контейнер с аннотациями")

                for i, obj in enumerate(records):
                    if not isinstance(obj, dict):
                        logger.warning(f"Пропуск записи #{i+1}: ожидается объект, получено {type(obj)}")
                        continue
                    sample = self._parse_sample(obj)
                    if sample:
                        self.samples.append(sample)
        else:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        sample = self._parse_sample(obj)
                        if sample:
                            self.samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Ошибка парсинга строки {i+1}: {e}")
                        continue

    def _parse_sample(self, obj: Dict[str, Any]) -> Optional[Sample]:
        image_rel = obj.get("image") or obj.get("image_path") or obj.get("image_file")
        if image_rel is None:
            logger.warning(f"Не найдено поле с изображением: {obj}")
            return None

        question = obj.get("question") or obj.get("instruction") or obj.get("prompt")
        answer = obj.get("answer") or obj.get("output") or obj.get("response")
        
        if not question or not answer:
            logger.warning(f"Отсутствует текст: {obj}")
            return None

        if self.image_root is None or os.path.isabs(image_rel):
            image_path = image_rel
        else:
            image_path = os.path.join(self.image_root, image_rel)

        if not os.path.exists(image_path):
            logger.warning(f"Файл не найден: {image_path}")
            return None
        
        return Sample(image_path=image_path, question=question, answer=answer)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample.image_path).convert('RGB')
            
            target_size = (224, 224)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": sample.question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample.answer}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            result = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() > 0:
                        result[key] = value.squeeze(0)
                    else:
                        result[key] = value
                else:
                    result[key] = value
            
            if "input_ids" in result:
                result["labels"] = result["input_ids"].clone()
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обработки образца {idx} ({sample.image_path}): {e}")
            return {
                "input_ids": torch.tensor([151643, 151643], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                "labels": torch.tensor([151643, 151643], dtype=torch.long)
            }


class VisionLanguageDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        batch = {}
        
        keys = features[0].keys()
        
        for key in keys:
            values = [f[key] for f in features if key in f]
            if key in ["input_ids", "attention_mask", "labels"]:
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
                )
            elif key in ["pixel_values", "image_grid_thw"]:
                try:
                    batch[key] = torch.stack(values)
                except:
                    batch[key] = values[0].unsqueeze(0) if len(values) > 0 else None
            else:
                batch[key] = values
        
        return batch


class QwenVLTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        
    def setup_wandb(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"qwen-vl-regional-{timestamp}"
        
        wandb_config = {
            "model_name": self.config.model_name,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "target_modules": self.config.target_modules,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "max_length": self.config.max_length,
            "regional_characters": REGIONAL_CHARACTERS,
        }
        
        try:
            wandb.init(
                project="qwen-vl-regional-characters",
                name=run_name,
                config=wandb_config,
                tags=["qwen-vl", "vision-language", "lora", "m4"],
                notes="Fine-tuning Qwen2.5-VL на региональных персонажах с LoRA адаптерами"
            )
            logger.info(f"WandB инициализирован: {run_name}")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать WandB: {e}")
    
    def setup_model(self):
        logger.info("Настройка модели Qwen2.5-VL...")
        
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Используется MPS (Apple Silicon)")
        else:
            device = "cpu"
            logger.info("Используется CPU")
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logger.info("Модель загружена через AutoModelForCausalLM")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели через AutoModelForCausalLM: {e}")
            logger.info("Попытка загрузки через прямой импорт...")
            
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logger.info("Модель загружена через прямой импорт")
        
        if device == "mps":
            self.model = self.model.to("mps")
        
        if hasattr(self.model, 'visual'):
            for param in self.model.visual.parameters():
                param.requires_grad = False
            logger.info("Визуальный энкодер заморожен")
        elif hasattr(self.model, 'vision_model'):
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Vision model заморожен")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        try:
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except Exception as e:
            logger.error(f"Ошибка применения LoRA: {e}")
            logger.info("Продолжаем без LoRA (полное дообучение)")
        
    def train(self):
        logger.info("Начало обучения...")
        
        full_dataset = JsonlVLDataset(
            jsonl_path=self.config.train_jsonl,
            processor=self.processor,
            image_root=self.config.image_root,
            max_length=self.config.max_length
        )
        
        if len(full_dataset) == 0:
            raise ValueError("Датасет пустой!")

        import random
        import copy
        num_samples = len(full_dataset)
        indices = list(range(num_samples))
        random.Random(self.config.seed).shuffle(indices)
        val_count = max(1, int(num_samples * self.config.val_size)) if num_samples > 1 else 1
        val_indices = set(indices[:val_count])
        train_indices = [i for i in indices if i not in val_indices]

        train_dataset = copy.copy(full_dataset)
        val_dataset = copy.copy(full_dataset)
        train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
        val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

        logger.info(f"Размер train: {len(train_dataset)} | val: {len(val_dataset)} из {num_samples}")

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=False,
            bf16=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            save_strategy="steps",
            logging_strategy="steps",
            report_to=["wandb"],
            gradient_checkpointing=False,
            dataloader_pin_memory=False,
            optim="adamw_torch",
            save_total_limit=1,
            max_grad_norm=1.0,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            prediction_loss_only=True,
        )
        
        data_collator = VisionLanguageDataCollator(self.processor)
        random.shuffle(train_dataset.samples)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.processor,
            data_collator=data_collator,
        )
        
        try:
            trainer.train()
            
            output_path = os.path.join(self.config.output_dir, "final_model")
            trainer.save_model(output_path)
            logger.info(f"Модель сохранена в: {output_path}")
            
            
        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}")
            try:
                wandb.log({"training/error": str(e)})
            except:
                pass
            
            try:
                emergency_path = os.path.join(self.config.output_dir, "emergency_save")
                self.model.save_pretrained(emergency_path)
                logger.info(f"Экстренное сохранение в: {emergency_path}")
            except:
                logger.error("Не удалось выполнить экстренное сохранение")
            raise
        finally:
            try:
                wandb.finish()
            except:
                pass
        
        return trainer


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    logger.info("🚀 Запуск обучения модели Qwen2.5-VL...")
    logger.info(f"📊 Конфигурация: {config.num_train_epochs} эпох, batch_size={config.per_device_train_batch_size}")
    logger.info(f"🎯 Целевые персонажи: {', '.join(REGIONAL_CHARACTERS)}")
    
    trainer = QwenVLTrainer(config)
    
    try:
        trainer.setup_wandb()
        trainer.setup_model()
        trained_model = trainer.train()
        logger.info("🎉 Пайплайн завершен успешно!")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()