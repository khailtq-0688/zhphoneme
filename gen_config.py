import os
import yaml
from metadata import UIT_VSFC_METADATA, UIT_ViCTSD_METADATA, UIT_ViOCD_METADATA

# import sys
# sys.setrecursionlimit(100000)
from typing import Dict, List, Any, Optional, Union


class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True


class ModelConfig:
    """
    Base class for model configurations.
    This class defines common properties and methods for all model types.
    """

    def __init__(self, name: str, architecture: str):
        """
        Initialize a model configuration.

        Args:
            name: Name of the model type (e.g., "GRU", "BiGRU", "Transformer")
            architecture: Architecture type for the model (e.g., "RNNmodel", "TransformerEncoder")
        """
        self.name = name
        self.architecture = architecture

    def get_config(self) -> Dict[str, Any]:
        """
        Get the base configuration for this model type.

        Returns:
            Dict containing the model configuration.
        """
        raise NotImplementedError(
            "Subclasses must implement get_config method")

    def customize_for_tokenizer(
        self, config: Dict[str, Any], tokenizer: str, task_type: str = "text_classification"
    ) -> Dict[str, Any]:
        """
        Customize configuration based on tokenizer.

        Args:
            config: Base configuration dictionary
            tokenizer: Name of the tokenizer
            task_type: Type of task (text_classification or aspect_based)

        Returns:
            Modified configuration dictionary
        """
        base_arch = self.architecture
        if task_type == "aspect_based":
            base_arch = f"{self.architecture}_ABSA"

        if tokenizer == "vipher":
            config["model"]["architecture"] = f"{base_arch}_ViPher"
        elif tokenizer == "vipherv2":
            config["model"]["architecture"] = f"{base_arch}_ViPherV2"
            config["embedder"] = EmbedderConfig(
            ).get_embedder_config(self.name)
            if 'transformer' in self.name.lower():
                config["model"]["d_model"] = config["embedder"]["embed_dim"] * \
                    config["embedder"]["bidirectional"]
            else:
                config["model"]["input_dim"] = config["embedder"]["embed_dim"] * \
                    config["embedder"]["bidirectional"]
        else:
            config["model"]["architecture"] = base_arch
        return config


class RNNConfig(ModelConfig):
    """
    Configuration class for RNN-based models (GRU, LSTM, etc.)
    """

    def __init__(self, name: str):
        """
        Initialize an RNN configuration.

        Args:
            name: Name of the RNN model (e.g., "GRU", "BiGRU", "LSTM", "BiLSTM")
        """
        architecture = "RNNmodel"
        super().__init__(name, architecture)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the base configuration for RNN models.

        Returns:
            Dict containing the model configuration.
        """
        bidirectional = 2 if "Bi" in self.name else 1
        model_type = self.name[2:] if "Bi" in self.name else self.name

        return {
            "architecture": self.architecture,
            "model_type": model_type,
            "bidirectional": bidirectional,
            "num_layer": 3,
            "input_dim": 256,
            "d_model": 256,
            "dropout": 0.2,
            "label_smoothing": 0.1,
            "device": "cuda",
        }


class EmbedderConfig:
    """
    Configuration class for RNN models with specific embedder configurations
    """

    def get_embedder_config(self, model) -> Dict[str, Any]:
        """
        Get embedder configuration for RNN models.

        Returns:
            Dict containing the embedder configuration.
        """

        return {
            "bidirectional": 1,
            "model_type": "GRU",
            "num_layer": 1,
            "dropout": 0.3,
            "embed_dim": 512 if "transformer" in model.lower() else 256,
        }


class TransformerConfig(ModelConfig):
    """
    Configuration class for Transformer-based models
    """

    def __init__(self):
        """Initialize a Transformer configuration."""
        super().__init__("Transformer", "TransformerEncoder")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the base configuration for Transformer models.

        Returns:
            Dict containing the model configuration.
        """
        return {
            "architecture": self.architecture,
            "nhead": 8,
            "num_encoder_layers": 3,
            "d_model": 512,
            "dropout": 0.2,
            "label_smoothing": 0.1,
            "max_seq_len": 1024,
            "device": "cuda",
            "mlp_scaler": 4,
        }


class TextCNNConfig(ModelConfig):
    """
    Configuration class for TextCNN models
    """
    # TODO: remove model.input_dim in TextCNN model

    def __init__(self):
        """Initialize a TextCNN configuration."""
        super().__init__("TextCNN", "TextCNN")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the base configuration for TextCNN models.

        Returns:
            Dict containing the model configuration.
        """
        return {
            "architecture": self.architecture,
            "n_filters": 100,
            "embedding_dim": 256,
            "filter_sizes": [3, 4, 5],
            "dropout": 0.2,
            "label_smoothing": 0.1,
            "device": "cuda",
        }


class ConfigGenerator:
    """
    Main class for generating configuration files for different models and datasets.
    """

    def __init__(
        self,
        dataset_name: str,
        task_metadata: Dict[str, Dict[str, Any]],
        vocab_size: int,
        vocab_size_v2: int,
        data_paths: Dict[str, str],
        schemas: Optional[List[int]] = None,
        tokenizers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the configuration generator.

        Args:
            dataset_name: Name of the dataset (e.g., "UIT_VFSC")
            task_metadata: Dictionary containing metadata for each task
            vocab_size: Size of vocabulary
            data_paths: Dictionary of paths to train/dev/test data
            schemas: List of schema numbers to use
            tokenizers: Dictionary mapping tokenizer names to their class names
        """
        self.dataset_name = dataset_name
        self.task_metadata = task_metadata
        self.vocab_size = vocab_size
        self.vocab_size_v2 = vocab_size_v2
        self.data_paths = data_paths

        self.schemas = schemas or [1, 2]
        self.tokenizers = tokenizers or {
            "bpe": "BPETokenizer",
            "unigram": "UnigramTokenizer",
            "wordpiece": "WordPieceTokenizer",
            "vipher": "VipherTokenizer",
            "vipherv2": "VipherTokenizerV2",
        }

        # Register available model configurations
        self.model_configs = {
            "GRU": RNNConfig("GRU"),
            "BiGRU": RNNConfig("BiGRU"),
            "LSTM": RNNConfig("LSTM"),
            "BiLSTM": RNNConfig("BiLSTM"),
            "Transformer": TransformerConfig(),
            "TextCNN": TextCNNConfig(),
        }

        # Track batch size
        if self.dataset_name == "UIT_ViCTSD":
            self.batch_size = 32
        else:
            self.batch_size = 64

        # Keep track of generated files
        self.generated_files = []

    def get_base_config(self) -> Dict[str, Any]:
        """
        Get the base configuration template.

        Returns:
            Dict containing the base configuration.
        """
        config = {
            "vocab": {
                "type": "",
                "model_prefix": "",
                "model_type": "",
                "schema": -1,
                "text": "",
                "label": "",
                "vocab_size": -1,
                "path": {
                    "train": self.data_paths["train"] if "train" in self.data_paths else "",
                    "dev": self.data_paths["dev"] if "dev" in self.data_paths else "",
                    "test": self.data_paths["test"] if "test" in self.data_paths else "",
                },
                "unk_piece": "<unk>",
                "bos_piece": "<s>",
                "eos_piece": "</s>",
                "pad_piece": "<pad>",
                "space_token": "<space>",
                "pad_id": 0,
                "bos_id": 1,
                "eos_id": 2,
                "unk_id": 3,
                "space_id": 4,
                "min_freq": 5,
            },
            "dataset": {
                "train": {
                    "type": "",
                    "path": self.data_paths["train"] if "train" in self.data_paths else "",
                },
                "dev": {
                    "type": "",
                    "path": self.data_paths["dev"] if "dev" in self.data_paths else "",
                },
                "test": {
                    "type": "",
                    "path": self.data_paths["test"] if "test" in self.data_paths else "",
                },
                "batch_size": self.batch_size,
                "num_workers": 4,
            },
            "model": {},
            "training": {
                "checkpoint_path": "",
                "seed": 42,
                "learning_rate": 0.08,
                "warmup": 500,
                "patience": 10,
                "score": "f1",
            },
            "task": "TextClassification",
        }

        return config

    def generate_configs(self, model_names: List[str]) -> List[str]:
        """
        Generate configuration files for specified models.

        Args:
            model_names: List of model names to generate configs for

        Returns:
            List of paths to generated config files
        """
        # Handle special case for ABSA tasks with task-specific data paths
        if hasattr(self, 'data_paths') and isinstance(self.data_paths, dict) and any(isinstance(v, dict) for v in self.data_paths.values()):
            # This is a dataset with task-specific paths like UIT-ABSA
            for task_name, task_metadata in self.task_metadata.items():
                task_paths = self.data_paths.get(task_name, {})
                if task_paths:
                    self._generate_task_configs(
                        task_name, task_metadata, model_names, task_paths)
        else:
            # Standard dataset with single set of paths
            for task_name, task_metadata in self.task_metadata.items():
                self._generate_task_configs(
                    task_name, task_metadata, model_names)

        return self.generated_files

    def _generate_task_configs(self, task_name, task_metadata, model_names, task_paths=None):
        """Generate configs for a specific task"""
        for schema in self.schemas:
            for model_name in model_names:
                if model_name not in self.model_configs:
                    print(f"Skipping unknown model: {model_name}")
                    continue

                # Create directory structure
                base_path = f"{self.dataset_name}/{task_name}/s{schema}/{model_name}"
                config_path_prefix = f"configs/{base_path}"
                os.makedirs(config_path_prefix, exist_ok=True)

                for tokenizer, tokenizer_class in self.tokenizers.items():
                    # Skip vipher with schema 2
                    if (tokenizer == "vipher" or tokenizer == "vipherv2") and schema == 2:
                        continue

                    config_name = f"config_{tokenizer}_{model_name}_{self.dataset_name}_{task_name}.yaml"
                    config_path = os.path.join(config_path_prefix, config_name)

                    # Generate the config
                    config = self._generate_config(
                        model_name,
                        task_name,
                        task_metadata,
                        tokenizer,
                        tokenizer_class,
                        schema,
                        base_path,
                        task_paths,
                    )

                    # Write the config to file
                    with open(config_path, "w") as yaml_file:
                        yaml.dump(config, yaml_file, default_flow_style=False)
                    self.generated_files.append(config_path)

    def _generate_config(
        self,
        model_name: str,
        task_name: str,
        task_metadata: Dict[str, Any],
        tokenizer: str,
        tokenizer_class: str,
        schema: int,
        base_path: str,
        task_paths=None,
    ) -> Dict[str, Any]:
        """
        Generate a specific configuration.

        Args:
            model_name: Name of the model
            task_name: Name of the task
            task_metadata: Metadata for the task
            tokenizer: Name of the tokenizer
            tokenizer_class: Class name for the tokenizer
            schema: Schema number
            base_path: Base path for the config
            task_paths: Optional task-specific data paths

        Returns:
            Dict containing the configuration
        """
        config = self.get_base_config()
        model_config = self.model_configs[model_name]
        task_type = task_metadata.get("task_type", "text_classification")

        # Configure checkpoint path
        checkpoint_path = f"checkpoints/{base_path}/{tokenizer}"

        # Configure vocabulary settings
        config["vocab"]["type"] = tokenizer_class
        config["vocab"][
            "model_prefix"
        ] = f"{checkpoint_path}/{self.dataset_name}_{tokenizer}"
        config["vocab"]["model_type"] = tokenizer
        config["vocab"]["text"] = task_metadata["text"]
        config["vocab"]["label"] = task_metadata["label"]
        config["vocab"]["schema"] = schema

        # Handle ABSA specific fields
        if task_type == "aspect_based":
            config["vocab"]["aspect"] = task_metadata["aspect"]
            config["vocab"]["aspect_label"] = task_metadata["aspect_label"]
            config["vocab"]["task_type"] = "aspect_based"
            config["task"] = "AspectBasedClassification"

        # Set vocab size based on tokenizer and task
        if tokenizer == "vipher":
            config["vocab"]["vocab_size"] = self.vocab_size
        elif tokenizer == "vipherv2":
            config["vocab"]["vocab_size"] = self.vocab_size_v2
            if task_name == "Hotel_ABSA": # except handle for Hotel_ABSA
                config["dataset"]["train"]["max_len"] = 128
                config["dataset"]["dev"]["max_len"] = 128
                config["dataset"]["test"]["max_len"] = 128
        # Configure dataset paths if task-specific paths provided
        if task_paths:
            config["vocab"]["path"]["train"] = task_paths["train"]
            config["vocab"]["path"]["dev"] = task_paths["dev"]
            config["vocab"]["path"]["test"] = task_paths["test"]

            config["dataset"]["train"]["path"] = task_paths["train"]
            config["dataset"]["dev"]["path"] = task_paths["dev"]
            config["dataset"]["test"]["path"] = task_paths["test"]

        # Configure dataset settings
        config["dataset"]["train"]["type"] = task_metadata["name"]
        config["dataset"]["dev"]["type"] = task_metadata["name"]
        config["dataset"]["test"]["type"] = task_metadata["name"]
        if model_name == "TextCNN":
            config["dataset"]["train"]["max_len"] = 256
            config["dataset"]["dev"]["max_len"] = 256
            config["dataset"]["test"]["max_len"] = 256

        # Configure model settings
        model_dict = model_config.get_config().copy()
        model_dict = model_config.customize_for_tokenizer(
            {"model": model_dict}, tokenizer, task_type
        )
        model_dict["model"]["num_output"] = task_metadata["num_label"]

        # Add num_categories for ABSA models
        if task_type == "aspect_based" and "num_categories" in task_metadata:
            model_dict["model"]["num_categories"] = task_metadata["num_categories"]

        model_dict["model"]["name"] = self._get_model_name(
            model_name, task_name, tokenizer, {"model": model_dict["model"]}
        )
        config["model"] = model_dict["model"]

        if "embedder" in model_dict:
            config["embedder"] = model_dict["embedder"]

        # Configure training settings
        config["training"]["checkpoint_path"] = checkpoint_path
        return config

    def _get_model_name(
        self, model_name: str, task_name: str, tokenizer: str, config: Dict[str, Any]
    ) -> str:
        """
        Generate a descriptive model name.

        Args:
            model_name: Base model name
            task_name: Name of the task
            tokenizer: Name of the tokenizer
            config: Configuration dictionary

        Returns:
            Descriptive model name
        """
        # Different logic based on model type
        if "num_layer" in config["model"]:
            num_layers = config["model"]["num_layer"]
            return f"{model_name}_Model{num_layers}layer_{self.dataset_name}_{tokenizer}_{task_name}"
        elif "num_encoder_layers" in config["model"]:
            num_layers = config["model"]["num_encoder_layers"]
            return f"{model_name}_Model{num_layers}layer_{self.dataset_name}_{tokenizer}_{task_name}"
        else:
            return f"{model_name}_{self.dataset_name}_{tokenizer}_{task_name}"

    def generate_shell_scripts(self) -> None:
        """Generate shell scripts to run training for generated configs."""
        tasks = self.task_metadata.keys()
        model_names = self.model_configs.keys()

        for config_path in self.generated_files:
            for task in tasks:
                for model in model_names:
                    if f"/{task}/" in config_path and f"/{model}/" in config_path:
                        path = f"scripts/{self.dataset_name}/{task}"
                        os.makedirs(path, exist_ok=True)
                        shell_path = f"{path}/{model}scripts.sh"

                        if os.path.exists(shell_path):
                            with open(shell_path, "a") as sh_file:
                                sh_file.write(
                                    f"python main_s1.py --config-file {config_path}\n"
                                )
                        else:
                            with open(shell_path, "w") as sh_file:
                                sh_file.write("#!/bin/bash\n\n")
                                sh_file.write(
                                    f"python main_s1.py --config-file {config_path}\n"
                                )

        # Make shell scripts executable
        for task in tasks:
            for model in model_names:
                shell_path = f"scripts/{self.dataset_name}/{task}/{model}scripts.sh"
                if os.path.exists(shell_path):
                    os.chmod(shell_path, 0o755)


if __name__ == "__main__":

    rnn_models = ["GRU", "BiGRU", "LSTM", "BiLSTM"]
    transformer_models = ["Transformer"]
    textcnn_models = ["TextCNN"]

    # # Initialize the UIT VSFC config generator
    # uit_vsfc_generator = ConfigGenerator(
    #     dataset_name=UIT_VSFC_METADATA["name"],
    #     task_metadata=UIT_VSFC_METADATA["task"],
    #     vocab_size=UIT_VSFC_METADATA["vocab_size"],
    #     vocab_size_v2=UIT_VSFC_METADATA["vocab_size_v2"],
    #     data_paths=UIT_VSFC_METADATA["data_paths"],
    # )
    # uit_vsfc_generator.generate_configs(rnn_models)
    # uit_vsfc_generator.generate_configs(transformer_models)
    # uit_vsfc_generator.generate_configs(textcnn_models)
    # uit_vsfc_generator.generate_shell_scripts()
    # print(
    #     f"UIT VSFC: Generated {len(uit_vsfc_generator.generated_files)} config files")

    # # Initialize the UIT ViCTSD config generator
    # uit_victsd_generator = ConfigGenerator(
    #     dataset_name=UIT_ViCTSD_METADATA["name"],
    #     task_metadata=UIT_ViCTSD_METADATA["task"],
    #     vocab_size=UIT_ViCTSD_METADATA["vocab_size"],
    #     vocab_size_v2=UIT_ViCTSD_METADATA["vocab_size_v2"],
    #     data_paths=UIT_ViCTSD_METADATA["data_paths"],
    # )
    # uit_victsd_generator.generate_configs(rnn_models)
    # uit_victsd_generator.generate_configs(transformer_models)
    # uit_victsd_generator.generate_configs(textcnn_models)
    # uit_victsd_generator.generate_shell_scripts()
    # print(
    #     f"UIT ViCTSD: Generated {len(uit_victsd_generator.generated_files)} config files")

    # # Initialize the UIT ViOCD config generator
    # uit_viocd_generator = ConfigGenerator(
    #     dataset_name=UIT_ViOCD_METADATA["name"],
    #     task_metadata=UIT_ViOCD_METADATA["task"],
    #     vocab_size=UIT_ViOCD_METADATA["vocab_size"],
    #     vocab_size_v2=UIT_ViOCD_METADATA["vocab_size_v2"],
    #     data_paths=UIT_ViOCD_METADATA["data_paths"],
    # )

    # uit_viocd_generator.generate_configs(rnn_models)
    # uit_viocd_generator.generate_configs(transformer_models)
    # uit_viocd_generator.generate_configs(textcnn_models)
    # uit_viocd_generator.generate_shell_scripts()
    # print(
    #     f"UIT ViOCD: Generated {len(uit_viocd_generator.generated_files)} config files")

    # Import new metadata
    from metadata import UIT_VSFC_METADATA, UIT_ViCTSD_METADATA, UIT_ViOCD_METADATA, UIT_ABSA_METADATA, UIT_ViSFD_METADATA

    # Initialize the UIT-ABSA config generator for Hotel
    uit_absa_hotel_generator = ConfigGenerator(
        dataset_name=UIT_ABSA_METADATA["name"],
        task_metadata={"Hotel_ABSA": UIT_ABSA_METADATA["task"]["Hotel_ABSA"]},
        vocab_size=UIT_ABSA_METADATA["vocab_size"],
        vocab_size_v2=UIT_ABSA_METADATA["vocab_size_v2"],
        data_paths=UIT_ABSA_METADATA["data_paths"],
    )
    uit_absa_hotel_generator.generate_configs(rnn_models)
    uit_absa_hotel_generator.generate_configs(transformer_models)
    uit_absa_hotel_generator.generate_configs(textcnn_models)
    uit_absa_hotel_generator.generate_shell_scripts()
    print(
        f"UIT-ABSA Hotel: Generated {len(uit_absa_hotel_generator.generated_files)} config files")

    # Initialize the UIT-ABSA config generator for Restaurant
    uit_absa_res_generator = ConfigGenerator(
        dataset_name=UIT_ABSA_METADATA["name"],
        task_metadata={"Res_ABSA": UIT_ABSA_METADATA["task"]["Res_ABSA"]},
        vocab_size=UIT_ABSA_METADATA["vocab_size_res"],
        vocab_size_v2=UIT_ABSA_METADATA["vocab_size_res_v2"],
        data_paths=UIT_ABSA_METADATA["data_paths"],
    )
    uit_absa_res_generator.generate_configs(rnn_models)
    uit_absa_res_generator.generate_configs(transformer_models)
    uit_absa_res_generator.generate_configs(textcnn_models)
    uit_absa_res_generator.generate_shell_scripts()
    print(
        f"UIT-ABSA Restaurant: Generated {len(uit_absa_res_generator.generated_files)} config files")

    # Initialize the UIT-ViSFD config generator
    uit_visfd_generator = ConfigGenerator(
        dataset_name=UIT_ViSFD_METADATA["name"],
        task_metadata=UIT_ViSFD_METADATA["task"],
        vocab_size=UIT_ViSFD_METADATA["vocab_size"],
        vocab_size_v2=UIT_ViSFD_METADATA["vocab_size_v2"],
        data_paths=UIT_ViSFD_METADATA["data_paths"],
    )
    uit_visfd_generator.generate_configs(rnn_models)
    uit_visfd_generator.generate_configs(transformer_models)
    uit_visfd_generator.generate_configs(textcnn_models)
    uit_visfd_generator.generate_shell_scripts()
    print(
        f"UIT-ViSFD: Generated {len(uit_visfd_generator.generated_files)} config files")
