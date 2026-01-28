#!/usr/bin/env python3
"""
Cancer Drug Synergy Virtual Screening Pipeline
==============================================

A complete, production-ready implementation for predicting synergistic
drug combinations in cancer treatment using deep learning on synthetic
single-cell gene expression data.

Author: AI Assistant
Version: 1.0.0
"""

import os
import sys
import json
import logging
import random
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import argparse
from visualization_publication import create_publication_figures

# =============================================================================
# CUSTOM EXCEPTION CLASSES
# =============================================================================

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class PipelineInitializationError(PipelineError):
    """Raised when pipeline initialization fails."""
    pass


class DataGenerationError(PipelineError):
    """Raised when data generation fails."""
    pass


class ModelTrainingError(PipelineError):
    """Raised when model training fails."""
    pass


class VirtualScreeningError(PipelineError):
    """Raised when virtual screening fails."""
    pass


class ValidationError(PipelineError):
    """Raised when validation fails."""
    pass


class DataLoadError(PipelineError):
    """Raised when data loading fails."""
    pass


# =============================================================================
# ENUMERATIONS
# =============================================================================

class DrugMechanism(Enum):
    """Drug mechanism of action categories."""
    MEK_INHIBITOR = auto()
    BRAF_INHIBITOR = auto()
    PI3K_INHIBITOR = auto()
    MTOR_INHIBITOR = auto()
    PARP_INHIBITOR = auto()
    CDK4_6_INHIBITOR = auto()
    BCL2_INHIBITOR = auto()
    EGFR_INHIBITOR = auto()
    HER2_INHIBITOR = auto()
    ALK_INHIBITOR = auto()
    VEGF_INHIBITOR = auto()
    CK2_INHIBITOR = auto()
    HDAC_INHIBITOR = auto()
    BET_INHIBITOR = auto()
    PD1_ANTIBODY = auto()
    PDL1_ANTIBODY = auto()
    CTLA4_ANTIBODY = auto()
    INTERFERON = auto()
    PLATINUM_CHEMO = auto()
    TAXANE_CHEMO = auto()
    ANTIMETABOLITE = auto()
    JAK_INHIBITOR = auto()
    SRC_INHIBITOR = auto()
    WNT_INHIBITOR = auto()
    NOTCH_INHIBITOR = auto()


class CancerPathway(Enum):
    """Major cancer signaling pathways."""
    MAPK_ERK = auto()
    PI3K_AKT_MTOR = auto()
    P53_APOPTOSIS = auto()
    CELL_CYCLE = auto()
    DNA_REPAIR = auto()
    WNT_BETA_CATENIN = auto()
    NOTCH = auto()
    HEDGEHOG = auto()
    TGFB = auto()
    HIPPO = auto()
    NF_KAPPA_B = auto()
    JAK_STAT = auto()
    ANGIOGENESIS = auto()
    IMMUNE_CHECKPOINT = auto()
    EPIGENETIC = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Drug:
    """
    Represents a drug with its properties and targets.
    
    Attributes:
        name: Drug name/identifier
        mechanism: Mechanism of action
        target_genes: List of target gene names
        target_pathways: List of target pathways
        ic50_nm: IC50 value in nanomolar
        max_effect: Maximum achievable effect (0-1)
        selectivity: Target selectivity score (0-1)
        is_immunotherapy: Whether drug is immunotherapy
        is_chemotherapy: Whether drug is chemotherapy
        clinical_stage: Clinical development stage
    """
    name: str
    mechanism: DrugMechanism
    target_genes: List[str]
    target_pathways: List[CancerPathway]
    ic50_nm: float
    max_effect: float = 0.8
    selectivity: float = 0.7
    is_immunotherapy: bool = False
    is_chemotherapy: bool = False
    clinical_stage: str = "Approved"
    
    def __post_init__(self):
        """Validate drug properties after initialization."""
        if not self.name:
            raise ValidationError("Drug name cannot be empty")
        if not self.target_genes:
            raise ValidationError(f"Drug {self.name} must have at least one target gene")
        if self.ic50_nm <= 0:
            raise ValidationError(f"Drug {self.name} IC50 must be positive")
        if not 0 <= self.max_effect <= 1:
            raise ValidationError(f"Drug {self.name} max_effect must be in [0, 1]")
        if not 0 <= self.selectivity <= 1:
            raise ValidationError(f"Drug {self.name} selectivity must be in [0, 1]")


@dataclass
class CellLine:
    """
    Represents a cancer cell line with its molecular profile.
    
    Attributes:
        name: Cell line name
        cancer_type: Type of cancer
        tissue_origin: Tissue of origin
        oncogenes: List of activated oncogenes
        tumor_suppressors: List of inactivated tumor suppressors
        driver_mutations: List of driver mutations
        active_pathways: List of constitutively active pathways
        suppressed_pathways: List of suppressed pathways
        doubling_time_hours: Cell doubling time
        immune_infiltrate: Level of immune infiltration
    """
    name: str
    cancer_type: str
    tissue_origin: str
    oncogenes: List[str]
    tumor_suppressors: List[str]
    driver_mutations: List[str]
    active_pathways: List[CancerPathway]
    suppressed_pathways: List[CancerPathway]
    doubling_time_hours: float = 24.0
    immune_infiltrate: str = "low"
    
    def __post_init__(self):
        """Validate cell line properties."""
        if not self.name:
            raise ValidationError("Cell line name cannot be empty")
        if not self.cancer_type:
            raise ValidationError(f"Cell line {self.name} must have cancer type")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProjectConfig:
    """
    Complete configuration for the cancer synergy pipeline.
    
    This class holds all hyperparameters, paths, and settings needed
    to run the complete pipeline from data generation to virtual screening.
    
    Example:
        >>> config = ProjectConfig(n_genes=2000, batch_size=32)
        >>> config.validate()
        >>> print(config.to_dict())
    """
    # Data dimensions
    n_genes: int = 2000
    n_cells_per_sample: int = 100
    n_pathways: int = 15
    n_drugs: int = 50
    n_cell_lines: int = 3
    
    # Training data generation
    n_training_samples: int = 5000
    n_control_samples: int = 500
    n_single_drug_samples: int = 1500
    n_combination_samples: int = 3000
    
    # Model architecture
    gene_embed_dim: int = 256
    drug_embed_dim: int = 128
    cell_line_embed_dim: int = 64
    latent_dim: int = 512
    n_transformer_layers: int = 4
    n_attention_heads: int = 8
    dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    n_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    
    # Loss weights
    synergy_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.1
    
    # Data splits
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path("./cancer_synergy_pipeline"))
    data_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    results_dir: Path = field(default=None)
    figures_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    
    # Runtime settings
    device: str = "auto"
    random_seed: int = 42
    use_mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_every_n_epochs: int = 10
    keep_n_checkpoints: int = 3
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    # Version
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Initialize derived paths and settings."""
        # Set up directory paths
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.results_dir is None:
            self.results_dir = self.base_dir / "results"
        if self.figures_dir is None:
            self.figures_dir = self.base_dir / "figures"
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"
        
        # Auto-detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir, 
                         self.figures_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """
        Validate all configuration parameters.
        
        Raises:
            ValidationError: If any parameter is invalid.
        """
        errors = []
        
        # Dimension validation
        if self.n_genes <= 0:
            errors.append("n_genes must be positive")
        if self.n_cells_per_sample <= 0:
            errors.append("n_cells_per_sample must be positive")
        if self.n_pathways <= 0:
            errors.append("n_pathways must be positive")
        if self.n_drugs <= 0:
            errors.append("n_drugs must be positive")
        if self.n_cell_lines <= 0:
            errors.append("n_cell_lines must be positive")
        
        # Training sample validation
        if self.n_training_samples <= 0:
            errors.append("n_training_samples must be positive")
        total_samples = (self.n_control_samples + self.n_single_drug_samples + 
                        self.n_combination_samples)
        if total_samples != self.n_training_samples:
            errors.append(f"Sample counts don't sum to n_training_samples: "
                         f"{total_samples} != {self.n_training_samples}")
        
        # Architecture validation
        if self.gene_embed_dim <= 0:
            errors.append("gene_embed_dim must be positive")
        if self.drug_embed_dim <= 0:
            errors.append("drug_embed_dim must be positive")
        if self.latent_dim <= 0:
            errors.append("latent_dim must be positive")
        if self.n_transformer_layers <= 0:
            errors.append("n_transformer_layers must be positive")
        if self.n_attention_heads <= 0:
            errors.append("n_attention_heads must be positive")
        if self.latent_dim % self.n_attention_heads != 0:
            errors.append("latent_dim must be divisible by n_attention_heads")
        if not 0 <= self.dropout < 1:
            errors.append("dropout must be in [0, 1)")
        
        # Training hyperparameter validation
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.learning_rate <= 0 or self.learning_rate >= 1:
            errors.append("learning_rate must be in (0, 1)")
        if self.weight_decay < 0:
            errors.append("weight_decay must be non-negative")
        if self.n_epochs <= 0:
            errors.append("n_epochs must be positive")
        if self.early_stopping_patience <= 0:
            errors.append("early_stopping_patience must be positive")
        if self.gradient_clip_norm <= 0:
            errors.append("gradient_clip_norm must be positive")
        
        # Split validation
        total_fraction = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total_fraction - 1.0) > 1e-6:
            errors.append(f"Data split fractions must sum to 1.0, got {total_fraction}")
        
        # Device validation
        if self.device not in ["cpu", "cuda", "auto"]:
            if not self.device.startswith("cuda:"):
                errors.append(f"Invalid device: {self.device}")
        
        if errors:
            raise ValidationError(
                "Configuration validation failed:\n" + 
                "\n".join(f"  - {e}" for e in errors)
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, Enum):
                result[key] = value.name
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectConfig':
        """Create configuration from dictionary."""
        # Convert string paths back to Path objects
        path_keys = ['base_dir', 'data_dir', 'models_dir', 'results_dir', 
                    'figures_dir', 'logs_dir']
        for key in path_keys:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])
        return cls(**data)
    
    def save(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'ProjectConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logger(name: str, config: ProjectConfig) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        config: Project configuration
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.log_level))
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if config.log_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = config.logs_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# PATHWAY DATABASE
# =============================================================================

class PathwayDatabase:
    """
    Database of cancer signaling pathways and their constituent genes.
    
    This class maintains a comprehensive mapping of 15 major cancer pathways
    to their associated genes, enabling pathway-level analysis and drug
    target identification.
    
    Example:
        >>> pathway_db = PathwayDatabase(logger)
        >>> mapk_genes = pathway_db.get_pathway_genes(CancerPathway.MAPK_ERK)
        >>> print(f"MAPK pathway has {len(mapk_genes)} genes")
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the pathway database.
        
        Args:
            logger: Logger instance for logging operations
        """
        self.logger = logger
        self.pathway_genes: Dict[CancerPathway, List[str]] = {}
        self.gene_pathways: Dict[str, List[CancerPathway]] = {}
        self.all_genes: Set[str] = set()
        
        self._initialize_pathway_genes()
        self._build_gene_to_pathway_mapping()
        self._validate()
        
        self.logger.info(f"[OK] PathwayDatabase initialized: "
                        f"{len(self.pathway_genes)} pathways, "
                        f"{len(self.all_genes)} unique genes")
    
    def _initialize_pathway_genes(self) -> None:
        """Initialize all 15 cancer pathways with their genes."""
        
        # 1. MAPK/ERK Pathway (RAS-RAF-MEK-ERK cascade)
        self.pathway_genes[CancerPathway.MAPK_ERK] = [
            "KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "ARAF",
            "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "MAPK8",
            "MAPK14", "SOS1", "SOS2", "GRB2", "SHC1",
            "DUSP1", "DUSP4", "DUSP6", "SPRY1", "SPRY2",
            "ELK1", "FOS", "JUN", "MYC", "CCND1",
            "NF1", "RASA1", "PTPN11", "GAB1", "GAB2"
        ]
        
        # 2. PI3K/AKT/mTOR Pathway
        self.pathway_genes[CancerPathway.PI3K_AKT_MTOR] = [
            "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PIK3R1",
            "PIK3R2", "AKT1", "AKT2", "AKT3", "PTEN",
            "MTOR", "RPTOR", "RICTOR", "TSC1", "TSC2",
            "RHEB", "RPS6KB1", "RPS6KB2", "EIF4EBP1", "PDK1",
            "INPP4B", "PIK3IP1", "DEPTOR", "MLST8", "AKT1S1",
            "GSK3B", "FOXO1", "FOXO3", "BAD", "PRAS40"
        ]
        
        # 3. P53/Apoptosis Pathway
        self.pathway_genes[CancerPathway.P53_APOPTOSIS] = [
            "TP53", "MDM2", "MDM4", "CDKN1A", "CDKN2A",
            "BAX", "BAK1", "BCL2", "BCL2L1", "MCL1",
            "BID", "BIM", "PUMA", "NOXA", "APAF1",
            "CASP3", "CASP7", "CASP8", "CASP9", "CYCS",
            "XIAP", "BIRC2", "BIRC3", "BIRC5", "DIABLO",
            "FAS", "FASLG", "TNFRSF10A", "TNFRSF10B", "ATM",
            "ATR", "CHEK1", "CHEK2", "GADD45A", "GADD45B"
        ]
        
        # 4. Cell Cycle Pathway
        self.pathway_genes[CancerPathway.CELL_CYCLE] = [
            "CDK1", "CDK2", "CDK4", "CDK6", "CCNA1",
            "CCNA2", "CCNB1", "CCNB2", "CCND1", "CCND2",
            "CCND3", "CCNE1", "CCNE2", "RB1", "E2F1",
            "E2F2", "E2F3", "E2F4", "CDKN1A", "CDKN1B",
            "CDKN2A", "CDKN2B", "CDKN2C", "CDC25A", "CDC25B",
            "CDC25C", "WEE1", "PLK1", "AURKA", "AURKB",
            "BUB1", "MAD2L1", "CDC20", "SKP2", "FBXW7"
        ]
        
        # 5. DNA Repair Pathway
        self.pathway_genes[CancerPathway.DNA_REPAIR] = [
            "BRCA1", "BRCA2", "RAD51", "RAD51B", "RAD51C",
            "RAD51D", "RAD52", "RAD54L", "XRCC2", "XRCC3",
            "PALB2", "FANCA", "FANCD2", "MLH1", "MSH2",
            "MSH6", "PMS2", "PARP1", "PARP2", "XRCC1",
            "LIG1", "LIG3", "LIG4", "POLB", "POLD1",
            "POLE", "XPA", "XPC", "ERCC1", "ERCC2",
            "ERCC4", "ATM", "ATR", "NBN", "MRE11"
        ]
        
        # 6. WNT/β-catenin Pathway
        self.pathway_genes[CancerPathway.WNT_BETA_CATENIN] = [
            "WNT1", "WNT2", "WNT3", "WNT3A", "WNT5A",
            "WNT7A", "WNT10B", "FZD1", "FZD2", "FZD7",
            "LRP5", "LRP6", "DVL1", "DVL2", "DVL3",
            "CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3B",
            "CK1", "TCF7", "TCF7L1", "TCF7L2", "LEF1",
            "MYC", "CCND1", "AXIN2", "DKK1", "DKK3",
            "SFRP1", "SFRP2", "WIF1", "RNF43", "ZNRF3"
        ]
        
        # 7. Notch Pathway
        self.pathway_genes[CancerPathway.NOTCH] = [
            "NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "JAG1",
            "JAG2", "DLL1", "DLL3", "DLL4", "RBPJ",
            "MAML1", "MAML2", "MAML3", "HES1", "HES5",
            "HEY1", "HEY2", "HEYL", "NUMB", "NUMBL",
            "ADAM10", "ADAM17", "PSEN1", "PSEN2", "NCSTN",
            "APH1A", "APH1B", "FBXW7", "ITCH", "DTX1"
        ]
        
        # 8. Hedgehog Pathway
        self.pathway_genes[CancerPathway.HEDGEHOG] = [
            "SHH", "IHH", "DHH", "PTCH1", "PTCH2",
            "SMO", "GLI1", "GLI2", "GLI3", "SUFU",
            "STK36", "KIF7", "HHIP", "GAS1", "CDON",
            "BOC", "DISP1", "SCUBE2", "GPC3", "GPC5",
            "MYCN", "CCND1", "CCND2", "BCL2", "FOXM1"
        ]
        
        # 9. TGF-β Pathway
        self.pathway_genes[CancerPathway.TGFB] = [
            "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2",
            "TGFBR3", "SMAD2", "SMAD3", "SMAD4", "SMAD7",
            "BMP2", "BMP4", "BMP7", "BMPR1A", "BMPR1B",
            "BMPR2", "SMAD1", "SMAD5", "SMAD9", "ID1",
            "ID2", "ID3", "RUNX1", "RUNX2", "RUNX3",
            "CDKN1A", "CDKN2B", "MYC", "SERPINE1", "SNAI1"
        ]
        
        # 10. Hippo Pathway
        self.pathway_genes[CancerPathway.HIPPO] = [
            "MST1", "MST2", "SAV1", "LATS1", "LATS2",
            "MOB1A", "MOB1B", "YAP1", "WWTR1", "TEAD1",
            "TEAD2", "TEAD3", "TEAD4", "NF2", "WWC1",
            "FRMD6", "AMOT", "AMOTL1", "AMOTL2", "VGLL4",
            "CTGF", "CYR61", "ANKRD1", "BIRC5", "AXL"
        ]
        
        # 11. NF-κB Pathway
        self.pathway_genes[CancerPathway.NF_KAPPA_B] = [
            "NFKB1", "NFKB2", "RELA", "RELB", "REL",
            "NFKBIA", "NFKBIB", "NFKBIE", "IKBKA", "IKBKB",
            "IKBKG", "CHUK", "NIK", "BCL3", "TNFAIP3",
            "TNF", "TNFRSF1A", "TNFRSF1B", "IL1B", "IL1R1",
            "TRAF2", "TRAF6", "TAB1", "TAB2", "MAP3K7",
            "BIRC2", "BIRC3", "XIAP", "CFLAR", "BCL2L1"
        ]
        
        # 12. JAK/STAT Pathway
        self.pathway_genes[CancerPathway.JAK_STAT] = [
            "JAK1", "JAK2", "JAK3", "TYK2", "STAT1",
            "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B",
            "STAT6", "SOCS1", "SOCS2", "SOCS3", "CISH",
            "PIAS1", "PIAS2", "PIAS3", "PIAS4", "SHP1",
            "SHP2", "IL6", "IL6R", "IL6ST", "IFNG",
            "IFNGR1", "IFNGR2", "BCL2", "MCL1", "MYC"
        ]
        
        # 13. Angiogenesis Pathway
        self.pathway_genes[CancerPathway.ANGIOGENESIS] = [
            "VEGFA", "VEGFB", "VEGFC", "VEGFD", "PGF",
            "FLT1", "KDR", "FLT4", "NRP1", "NRP2",
            "HIF1A", "EPAS1", "ARNT", "VHL", "EGLN1",
            "EGLN2", "EGLN3", "ANGPT1", "ANGPT2", "TEK",
            "PDGFA", "PDGFB", "PDGFRA", "PDGFRB", "FGF2",
            "FGFR1", "FGFR2", "THBS1", "THBS2", "SERPINF1"
        ]
        
        # 14. Immune Checkpoint Pathway
        self.pathway_genes[CancerPathway.IMMUNE_CHECKPOINT] = [
            "PDCD1", "CD274", "PDCD1LG2", "CTLA4", "CD80",
            "CD86", "LAG3", "HAVCR2", "TIGIT", "BTLA",
            "VISTA", "IDO1", "IDO2", "TDO2", "CD47",
            "SIRPA", "CD27", "CD70", "TNFRSF9", "TNFSF9",
            "ICOS", "ICOSLG", "GITR", "GITRL", "OX40",
            "OX40L", "CD40", "CD40LG", "B7H3", "B7H4"
        ]
        
        # 15. Epigenetic Regulation Pathway
        self.pathway_genes[CancerPathway.EPIGENETIC] = [
            "DNMT1", "DNMT3A", "DNMT3B", "TET1", "TET2",
            "TET3", "IDH1", "IDH2", "EZH2", "EED",
            "SUZ12", "KMT2A", "KMT2B", "KMT2C", "KMT2D",
            "SETD2", "NSD1", "NSD2", "NSD3", "KDM1A",
            "KDM5A", "KDM5B", "KDM6A", "KDM6B", "HDAC1",
            "HDAC2", "HDAC3", "HDAC6", "SIRT1", "BRD4",
            "BRD2", "BRD3", "CREBBP", "EP300", "ARID1A"
        ]
    
    def _build_gene_to_pathway_mapping(self) -> None:
        """Build reverse mapping from genes to pathways."""
        self.gene_pathways = {}
        
        for pathway, genes in self.pathway_genes.items():
            for gene in genes:
                self.all_genes.add(gene)
                if gene not in self.gene_pathways:
                    self.gene_pathways[gene] = []
                self.gene_pathways[gene].append(pathway)
    
    def _validate(self) -> None:
        """Validate database integrity."""
        # Check all pathways are initialized
        for pathway in CancerPathway:
            if pathway not in self.pathway_genes:
                raise ValidationError(f"Pathway {pathway.name} not initialized")
            if len(self.pathway_genes[pathway]) < 15:
                self.logger.warning(
                    f"[!] Pathway {pathway.name} has only "
                    f"{len(self.pathway_genes[pathway])} genes (expected 15-40)"
                )
        
        self.logger.debug(f"[OK] PathwayDatabase validation passed")
    
    def get_pathway_genes(self, pathway: CancerPathway) -> List[str]:
        """
        Get all genes in a pathway.
        
        Args:
            pathway: Cancer pathway
            
        Returns:
            List of gene names in the pathway
        """
        if pathway not in self.pathway_genes:
            raise ValidationError(f"Unknown pathway: {pathway}")
        return self.pathway_genes[pathway].copy()
    
    def get_gene_pathways(self, gene: str) -> List[CancerPathway]:
        """
        Get all pathways containing a gene.
        
        Args:
            gene: Gene name
            
        Returns:
            List of pathways containing the gene
        """
        return self.gene_pathways.get(gene, []).copy()
    
    def get_pathway_overlap(self, pathway1: CancerPathway, 
                           pathway2: CancerPathway) -> Set[str]:
        """
        Get genes shared between two pathways.
        
        Args:
            pathway1: First pathway
            pathway2: Second pathway
            
        Returns:
            Set of shared gene names
        """
        genes1 = set(self.get_pathway_genes(pathway1))
        genes2 = set(self.get_pathway_genes(pathway2))
        return genes1 & genes2
    
    def get_all_genes(self) -> List[str]:
        """Get all unique genes across all pathways."""
        return list(self.all_genes)
    
    def get_pathway_matrix(self, gene_list: List[str]) -> np.ndarray:
        """
        Create binary pathway membership matrix.
        
        Args:
            gene_list: List of gene names
            
        Returns:
            Binary matrix of shape (n_genes, n_pathways)
        """
        n_genes = len(gene_list)
        n_pathways = len(CancerPathway)
        matrix = np.zeros((n_genes, n_pathways), dtype=np.float32)
        
        gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}
        pathway_to_idx = {p: i for i, p in enumerate(CancerPathway)}
        
        for pathway, genes in self.pathway_genes.items():
            pathway_idx = pathway_to_idx[pathway]
            for gene in genes:
                if gene in gene_to_idx:
                    gene_idx = gene_to_idx[gene]
                    matrix[gene_idx, pathway_idx] = 1.0
        
        return matrix


# =============================================================================
# DRUG DATABASE
# =============================================================================

class DrugDatabase:
    """
    Database of cancer drugs with their targets and properties.
    
    Contains 50 drugs spanning multiple mechanisms of action including
    targeted therapies, immunotherapies, and chemotherapies.
    
    Example:
        >>> drug_db = DrugDatabase(pathway_db, logger)
        >>> mek_inhibitors = drug_db.get_drugs_by_mechanism(DrugMechanism.MEK_INHIBITOR)
    """
    
    def __init__(self, pathway_db: PathwayDatabase, logger: logging.Logger):
        """
        Initialize the drug database.
        
        Args:
            pathway_db: Pathway database for validation
            logger: Logger instance
        """
        self.pathway_db = pathway_db
        self.logger = logger
        self.drugs: Dict[str, Drug] = {}
        
        self._initialize_drugs()
        self._validate()
        
        self.logger.info(f"[OK] DrugDatabase initialized: {len(self.drugs)} drugs")
    
    def _initialize_drugs(self) -> None:
        """Initialize all 50 drugs with complete properties."""
        
        # MEK Inhibitors (5 drugs)
        self.drugs["trametinib"] = Drug(
            name="trametinib",
            mechanism=DrugMechanism.MEK_INHIBITOR,
            target_genes=["MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "BRAF"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=1.0, max_effect=0.85, selectivity=0.9
        )
        
        self.drugs["cobimetinib"] = Drug(
            name="cobimetinib",
            mechanism=DrugMechanism.MEK_INHIBITOR,
            target_genes=["MAP2K1", "MAP2K2", "MAPK1", "MAPK3"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=0.9, max_effect=0.82, selectivity=0.88
        )
        
        self.drugs["binimetinib"] = Drug(
            name="binimetinib",
            mechanism=DrugMechanism.MEK_INHIBITOR,
            target_genes=["MAP2K1", "MAP2K2", "MAPK1"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=12.0, max_effect=0.78, selectivity=0.85
        )
        
        self.drugs["selumetinib"] = Drug(
            name="selumetinib",
            mechanism=DrugMechanism.MEK_INHIBITOR,
            target_genes=["MAP2K1", "MAP2K2", "MAPK3"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=14.0, max_effect=0.75, selectivity=0.82
        )
        
        self.drugs["mirdametinib"] = Drug(
            name="mirdametinib",
            mechanism=DrugMechanism.MEK_INHIBITOR,
            target_genes=["MAP2K1", "MAP2K2"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=5.0, max_effect=0.80, selectivity=0.87
        )
        
        # BRAF Inhibitors (4 drugs)
        self.drugs["vemurafenib"] = Drug(
            name="vemurafenib",
            mechanism=DrugMechanism.BRAF_INHIBITOR,
            target_genes=["BRAF", "RAF1", "ARAF", "MAPK1"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=31.0, max_effect=0.88, selectivity=0.75
        )
        
        self.drugs["dabrafenib"] = Drug(
            name="dabrafenib",
            mechanism=DrugMechanism.BRAF_INHIBITOR,
            target_genes=["BRAF", "RAF1", "MAPK1", "MAPK3"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=0.8, max_effect=0.90, selectivity=0.80
        )
        
        self.drugs["encorafenib"] = Drug(
            name="encorafenib",
            mechanism=DrugMechanism.BRAF_INHIBITOR,
            target_genes=["BRAF", "RAF1", "ARAF"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=0.4, max_effect=0.87, selectivity=0.78
        )
        
        self.drugs["sorafenib"] = Drug(
            name="sorafenib",
            mechanism=DrugMechanism.BRAF_INHIBITOR,
            target_genes=["BRAF", "RAF1", "VEGFR2", "PDGFRB", "KIT"],
            target_pathways=[CancerPathway.MAPK_ERK, CancerPathway.ANGIOGENESIS],
            ic50_nm=6.0, max_effect=0.75, selectivity=0.60
        )
        
        # PI3K Inhibitors (4 drugs)
        self.drugs["alpelisib"] = Drug(
            name="alpelisib",
            mechanism=DrugMechanism.PI3K_INHIBITOR,
            target_genes=["PIK3CA", "PIK3CB", "AKT1", "MTOR"],
            target_pathways=[CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=5.0, max_effect=0.82, selectivity=0.85
        )
        
        self.drugs["idelalisib"] = Drug(
            name="idelalisib",
            mechanism=DrugMechanism.PI3K_INHIBITOR,
            target_genes=["PIK3CD", "AKT1", "AKT2"],
            target_pathways=[CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=2.5, max_effect=0.80, selectivity=0.90
        )
        
        self.drugs["copanlisib"] = Drug(
            name="copanlisib",
            mechanism=DrugMechanism.PI3K_INHIBITOR,
            target_genes=["PIK3CA", "PIK3CD", "AKT1"],
            target_pathways=[CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=0.5, max_effect=0.85, selectivity=0.82
        )
        
        self.drugs["duvelisib"] = Drug(
            name="duvelisib",
            mechanism=DrugMechanism.PI3K_INHIBITOR,
            target_genes=["PIK3CD", "PIK3CG", "AKT1"],
            target_pathways=[CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=2.0, max_effect=0.78, selectivity=0.88
        )
        
        # mTOR Inhibitors (3 drugs)
        self.drugs["everolimus"] = Drug(
            name="everolimus",
            mechanism=DrugMechanism.MTOR_INHIBITOR,
            target_genes=["MTOR", "RPTOR", "RPS6KB1", "EIF4EBP1"],
            target_pathways=[CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=1.6, max_effect=0.80, selectivity=0.85
        )
        
        self.drugs["temsirolimus"] = Drug(
            name="temsirolimus",
            mechanism=DrugMechanism.MTOR_INHIBITOR,
            target_genes=["MTOR", "RPTOR", "RPS6KB1"],
            target_pathways=[CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=1.8, max_effect=0.78, selectivity=0.82
        )
        
        self.drugs["sirolimus"] = Drug(
            name="sirolimus",
            mechanism=DrugMechanism.MTOR_INHIBITOR,
            target_genes=["MTOR", "RPTOR", "MLST8"],
            target_pathways=[CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=0.1, max_effect=0.82, selectivity=0.90
        )
        
        # PARP Inhibitors (4 drugs)
        self.drugs["olaparib"] = Drug(
            name="olaparib",
            mechanism=DrugMechanism.PARP_INHIBITOR,
            target_genes=["PARP1", "PARP2", "BRCA1", "BRCA2", "RAD51"],
            target_pathways=[CancerPathway.DNA_REPAIR],
            ic50_nm=5.0, max_effect=0.85, selectivity=0.88
        )
        
        self.drugs["rucaparib"] = Drug(
            name="rucaparib",
            mechanism=DrugMechanism.PARP_INHIBITOR,
            target_genes=["PARP1", "PARP2", "PARP3", "BRCA1"],
            target_pathways=[CancerPathway.DNA_REPAIR],
            ic50_nm=1.4, max_effect=0.83, selectivity=0.85
        )
        
        self.drugs["niraparib"] = Drug(
            name="niraparib",
            mechanism=DrugMechanism.PARP_INHIBITOR,
            target_genes=["PARP1", "PARP2", "RAD51", "BRCA2"],
            target_pathways=[CancerPathway.DNA_REPAIR],
            ic50_nm=3.8, max_effect=0.82, selectivity=0.87
        )
        
        self.drugs["talazoparib"] = Drug(
            name="talazoparib",
            mechanism=DrugMechanism.PARP_INHIBITOR,
            target_genes=["PARP1", "PARP2", "BRCA1", "BRCA2"],
            target_pathways=[CancerPathway.DNA_REPAIR],
            ic50_nm=0.6, max_effect=0.90, selectivity=0.92
        )
        
        # CDK4/6 Inhibitors (3 drugs)
        self.drugs["palbociclib"] = Drug(
            name="palbociclib",
            mechanism=DrugMechanism.CDK4_6_INHIBITOR,
            target_genes=["CDK4", "CDK6", "RB1", "CCND1", "E2F1"],
            target_pathways=[CancerPathway.CELL_CYCLE],
            ic50_nm=11.0, max_effect=0.85, selectivity=0.90
        )
        
        self.drugs["ribociclib"] = Drug(
            name="ribociclib",
            mechanism=DrugMechanism.CDK4_6_INHIBITOR,
            target_genes=["CDK4", "CDK6", "CCND1", "RB1"],
            target_pathways=[CancerPathway.CELL_CYCLE],
            ic50_nm=10.0, max_effect=0.83, selectivity=0.88
        )
        
        self.drugs["abemaciclib"] = Drug(
            name="abemaciclib",
            mechanism=DrugMechanism.CDK4_6_INHIBITOR,
            target_genes=["CDK4", "CDK6", "CDK9", "CCND1"],
            target_pathways=[CancerPathway.CELL_CYCLE],
            ic50_nm=2.0, max_effect=0.87, selectivity=0.82
        )
        
        # BCL2 Inhibitors (2 drugs)
        self.drugs["venetoclax"] = Drug(
            name="venetoclax",
            mechanism=DrugMechanism.BCL2_INHIBITOR,
            target_genes=["BCL2", "BCL2L1", "MCL1", "BAX", "BAK1"],
            target_pathways=[CancerPathway.P53_APOPTOSIS],
            ic50_nm=0.01, max_effect=0.90, selectivity=0.95
        )
        
        self.drugs["navitoclax"] = Drug(
            name="navitoclax",
            mechanism=DrugMechanism.BCL2_INHIBITOR,
            target_genes=["BCL2", "BCL2L1", "BCL2L2", "BAX"],
            target_pathways=[CancerPathway.P53_APOPTOSIS],
            ic50_nm=0.1, max_effect=0.85, selectivity=0.80
        )
        
        # EGFR Inhibitors (3 drugs)
        self.drugs["erlotinib"] = Drug(
            name="erlotinib",
            mechanism=DrugMechanism.EGFR_INHIBITOR,
            target_genes=["EGFR", "ERBB2", "GRB2", "SOS1"],
            target_pathways=[CancerPathway.MAPK_ERK, CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=2.0, max_effect=0.80, selectivity=0.85
        )
        
        self.drugs["gefitinib"] = Drug(
            name="gefitinib",
            mechanism=DrugMechanism.EGFR_INHIBITOR,
            target_genes=["EGFR", "ERBB2", "GRB2"],
            target_pathways=[CancerPathway.MAPK_ERK, CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=33.0, max_effect=0.78, selectivity=0.82
        )
        
        self.drugs["osimertinib"] = Drug(
            name="osimertinib",
            mechanism=DrugMechanism.EGFR_INHIBITOR,
            target_genes=["EGFR", "ERBB2", "ERBB4", "GRB2"],
            target_pathways=[CancerPathway.MAPK_ERK, CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=15.0, max_effect=0.88, selectivity=0.90
        )
        
        # HER2 Inhibitors (2 drugs)
        self.drugs["lapatinib"] = Drug(
            name="lapatinib",
            mechanism=DrugMechanism.HER2_INHIBITOR,
            target_genes=["ERBB2", "EGFR", "GRB2", "SHC1"],
            target_pathways=[CancerPathway.MAPK_ERK, CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=10.8, max_effect=0.82, selectivity=0.85
        )
        
        self.drugs["neratinib"] = Drug(
            name="neratinib",
            mechanism=DrugMechanism.HER2_INHIBITOR,
            target_genes=["ERBB2", "EGFR", "ERBB4", "GRB2"],
            target_pathways=[CancerPathway.MAPK_ERK, CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=59.0, max_effect=0.80, selectivity=0.78
        )
        
        # ALK Inhibitors (3 drugs)
        self.drugs["crizotinib"] = Drug(
            name="crizotinib",
            mechanism=DrugMechanism.ALK_INHIBITOR,
            target_genes=["ALK", "MET", "ROS1", "RON"],
            target_pathways=[CancerPathway.MAPK_ERK, CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=24.0, max_effect=0.82, selectivity=0.75
        )
        
        self.drugs["alectinib"] = Drug(
            name="alectinib",
            mechanism=DrugMechanism.ALK_INHIBITOR,
            target_genes=["ALK", "RET", "LTK"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=1.9, max_effect=0.88, selectivity=0.90
        )
        
        self.drugs["lorlatinib"] = Drug(
            name="lorlatinib",
            mechanism=DrugMechanism.ALK_INHIBITOR,
            target_genes=["ALK", "ROS1", "FER"],
            target_pathways=[CancerPathway.MAPK_ERK],
            ic50_nm=0.7, max_effect=0.90, selectivity=0.88
        )
        
        # VEGF Inhibitors (3 drugs)
        self.drugs["bevacizumab"] = Drug(
            name="bevacizumab",
            mechanism=DrugMechanism.VEGF_INHIBITOR,
            target_genes=["VEGFA", "KDR", "FLT1", "NRP1"],
            target_pathways=[CancerPathway.ANGIOGENESIS],
            ic50_nm=50.0, max_effect=0.75, selectivity=0.95
        )
        
        self.drugs["sunitinib"] = Drug(
            name="sunitinib",
            mechanism=DrugMechanism.VEGF_INHIBITOR,
            target_genes=["KDR", "FLT1", "FLT4", "PDGFRA", "PDGFRB", "KIT"],
            target_pathways=[CancerPathway.ANGIOGENESIS, CancerPathway.MAPK_ERK],
            ic50_nm=1.0, max_effect=0.80, selectivity=0.65
        )
        
        self.drugs["axitinib"] = Drug(
            name="axitinib",
            mechanism=DrugMechanism.VEGF_INHIBITOR,
            target_genes=["KDR", "FLT1", "FLT4", "PDGFRB"],
            target_pathways=[CancerPathway.ANGIOGENESIS],
            ic50_nm=0.2, max_effect=0.85, selectivity=0.80
        )
        
        # CK2 Inhibitor (silmitasertib - key drug from C2S-Scale)
        self.drugs["silmitasertib"] = Drug(
            name="silmitasertib",
            mechanism=DrugMechanism.CK2_INHIBITOR,
            target_genes=["CSNK2A1", "CSNK2A2", "CSNK2B", "AKT1", "STAT3"],
            target_pathways=[CancerPathway.PI3K_AKT_MTOR, CancerPathway.JAK_STAT, 
                           CancerPathway.NF_KAPPA_B],
            ic50_nm=1.0, max_effect=0.85, selectivity=0.88
        )
        
        # HDAC Inhibitors (2 drugs)
        self.drugs["vorinostat"] = Drug(
            name="vorinostat",
            mechanism=DrugMechanism.HDAC_INHIBITOR,
            target_genes=["HDAC1", "HDAC2", "HDAC3", "HDAC6", "EP300"],
            target_pathways=[CancerPathway.EPIGENETIC, CancerPathway.CELL_CYCLE],
            ic50_nm=10.0, max_effect=0.75, selectivity=0.70
        )
        
        self.drugs["panobinostat"] = Drug(
            name="panobinostat",
            mechanism=DrugMechanism.HDAC_INHIBITOR,
            target_genes=["HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC8"],
            target_pathways=[CancerPathway.EPIGENETIC, CancerPathway.CELL_CYCLE],
            ic50_nm=5.0, max_effect=0.80, selectivity=0.65
        )
        
        # BET Inhibitors (2 drugs)
        self.drugs["jq1"] = Drug(
            name="jq1",
            mechanism=DrugMechanism.BET_INHIBITOR,
            target_genes=["BRD4", "BRD2", "BRD3", "MYC", "BCL2"],
            target_pathways=[CancerPathway.EPIGENETIC, CancerPathway.CELL_CYCLE],
            ic50_nm=77.0, max_effect=0.82, selectivity=0.85
        )
        
        self.drugs["otx015"] = Drug(
            name="otx015",
            mechanism=DrugMechanism.BET_INHIBITOR,
            target_genes=["BRD4", "BRD2", "BRD3", "MYC"],
            target_pathways=[CancerPathway.EPIGENETIC],
            ic50_nm=92.0, max_effect=0.78, selectivity=0.82
        )
        
        # PD-1 Antibodies (2 drugs)
        self.drugs["pembrolizumab"] = Drug(
            name="pembrolizumab",
            mechanism=DrugMechanism.PD1_ANTIBODY,
            target_genes=["PDCD1", "CD274", "PDCD1LG2", "CD8A", "IFNG"],
            target_pathways=[CancerPathway.IMMUNE_CHECKPOINT],
            ic50_nm=100.0, max_effect=0.70, selectivity=0.95,
            is_immunotherapy=True
        )
        
        self.drugs["nivolumab"] = Drug(
            name="nivolumab",
            mechanism=DrugMechanism.PD1_ANTIBODY,
            target_genes=["PDCD1", "CD274", "PDCD1LG2", "CD8A"],
            target_pathways=[CancerPathway.IMMUNE_CHECKPOINT],
            ic50_nm=120.0, max_effect=0.68, selectivity=0.95,
            is_immunotherapy=True
        )
        
        # PD-L1 Antibodies (2 drugs)
        self.drugs["atezolizumab"] = Drug(
            name="atezolizumab",
            mechanism=DrugMechanism.PDL1_ANTIBODY,
            target_genes=["CD274", "PDCD1LG2", "PDCD1", "CD80"],
            target_pathways=[CancerPathway.IMMUNE_CHECKPOINT],
            ic50_nm=80.0, max_effect=0.72, selectivity=0.93,
            is_immunotherapy=True
        )
        
        self.drugs["durvalumab"] = Drug(
            name="durvalumab",
            mechanism=DrugMechanism.PDL1_ANTIBODY,
            target_genes=["CD274", "PDCD1", "CD80", "CD86"],
            target_pathways=[CancerPathway.IMMUNE_CHECKPOINT],
            ic50_nm=90.0, max_effect=0.70, selectivity=0.92,
            is_immunotherapy=True
        )
        
        # CTLA-4 Antibody
        self.drugs["ipilimumab"] = Drug(
            name="ipilimumab",
            mechanism=DrugMechanism.CTLA4_ANTIBODY,
            target_genes=["CTLA4", "CD80", "CD86", "CD28"],
            target_pathways=[CancerPathway.IMMUNE_CHECKPOINT],
            ic50_nm=150.0, max_effect=0.65, selectivity=0.90,
            is_immunotherapy=True
        )
        
        # Interferon (key drug from C2S-Scale)
        self.drugs["interferon_alpha"] = Drug(
            name="interferon_alpha",
            mechanism=DrugMechanism.INTERFERON,
            target_genes=["IFNAR1", "IFNAR2", "STAT1", "STAT2", "IRF9", "ISG15"],
            target_pathways=[CancerPathway.JAK_STAT, CancerPathway.IMMUNE_CHECKPOINT],
            ic50_nm=10.0, max_effect=0.75, selectivity=0.85,
            is_immunotherapy=True
        )
        
        # Platinum Chemotherapy (2 drugs)
        self.drugs["cisplatin"] = Drug(
            name="cisplatin",
            mechanism=DrugMechanism.PLATINUM_CHEMO,
            target_genes=["TP53", "ATM", "ATR", "ERCC1", "XPA", "MLH1"],
            target_pathways=[CancerPathway.DNA_REPAIR, CancerPathway.P53_APOPTOSIS],
            ic50_nm=1000.0, max_effect=0.85, selectivity=0.50,
            is_chemotherapy=True
        )
        
        self.drugs["carboplatin"] = Drug(
            name="carboplatin",
            mechanism=DrugMechanism.PLATINUM_CHEMO,
            target_genes=["TP53", "ATM", "ATR", "ERCC1", "XPA"],
            target_pathways=[CancerPathway.DNA_REPAIR, CancerPathway.P53_APOPTOSIS],
            ic50_nm=5000.0, max_effect=0.80, selectivity=0.55,
            is_chemotherapy=True
        )
        
        # Taxane Chemotherapy (2 drugs)
        self.drugs["paclitaxel"] = Drug(
            name="paclitaxel",
            mechanism=DrugMechanism.TAXANE_CHEMO,
            target_genes=["TUBB", "MAP2", "STMN1", "BCL2", "CCNB1"],
            target_pathways=[CancerPathway.CELL_CYCLE, CancerPathway.P53_APOPTOSIS],
            ic50_nm=4.0, max_effect=0.88, selectivity=0.60,
            is_chemotherapy=True
        )
        
        self.drugs["docetaxel"] = Drug(
            name="docetaxel",
            mechanism=DrugMechanism.TAXANE_CHEMO,
            target_genes=["TUBB", "MAP2", "STMN1", "BCL2"],
            target_pathways=[CancerPathway.CELL_CYCLE, CancerPathway.P53_APOPTOSIS],
            ic50_nm=2.0, max_effect=0.90, selectivity=0.58,
            is_chemotherapy=True
        )
        
        # Antimetabolites (2 drugs)
        self.drugs["gemcitabine"] = Drug(
            name="gemcitabine",
            mechanism=DrugMechanism.ANTIMETABOLITE,
            target_genes=["RRM1", "RRM2", "TYMS", "DCK", "CDA"],
            target_pathways=[CancerPathway.DNA_REPAIR, CancerPathway.CELL_CYCLE],
            ic50_nm=20.0, max_effect=0.82, selectivity=0.55,
            is_chemotherapy=True
        )
        
        self.drugs["5_fluorouracil"] = Drug(
            name="5_fluorouracil",
            mechanism=DrugMechanism.ANTIMETABOLITE,
            target_genes=["TYMS", "DPYD", "MTHFR", "UMPS"],
            target_pathways=[CancerPathway.DNA_REPAIR, CancerPathway.CELL_CYCLE],
            ic50_nm=500.0, max_effect=0.78, selectivity=0.50,
            is_chemotherapy=True
        )
        
        # JAK Inhibitors (2 drugs)
        self.drugs["ruxolitinib"] = Drug(
            name="ruxolitinib",
            mechanism=DrugMechanism.JAK_INHIBITOR,
            target_genes=["JAK1", "JAK2", "STAT3", "STAT5A", "STAT5B"],
            target_pathways=[CancerPathway.JAK_STAT],
            ic50_nm=3.3, max_effect=0.82, selectivity=0.85
        )
        
        self.drugs["tofacitinib"] = Drug(
            name="tofacitinib",
            mechanism=DrugMechanism.JAK_INHIBITOR,
            target_genes=["JAK1", "JAK3", "STAT1", "STAT3"],
            target_pathways=[CancerPathway.JAK_STAT],
            ic50_nm=1.0, max_effect=0.80, selectivity=0.82
        )
        
        # SRC Inhibitor
        self.drugs["dasatinib"] = Drug(
            name="dasatinib",
            mechanism=DrugMechanism.SRC_INHIBITOR,
            target_genes=["SRC", "ABL1", "LCK", "YES1", "FYN"],
            target_pathways=[CancerPathway.MAPK_ERK, CancerPathway.PI3K_AKT_MTOR],
            ic50_nm=0.5, max_effect=0.85, selectivity=0.70
        )
        
        # WNT Inhibitor
        self.drugs["lgk974"] = Drug(
            name="lgk974",
            mechanism=DrugMechanism.WNT_INHIBITOR,
            target_genes=["PORCN", "WNT3A", "CTNNB1", "AXIN2"],
            target_pathways=[CancerPathway.WNT_BETA_CATENIN],
            ic50_nm=0.3, max_effect=0.78, selectivity=0.88
        )
        
        # Notch Inhibitor
        self.drugs["ly3039478"] = Drug(
            name="ly3039478",
            mechanism=DrugMechanism.NOTCH_INHIBITOR,
            target_genes=["NOTCH1", "NOTCH2", "HES1", "HEY1"],
            target_pathways=[CancerPathway.NOTCH],
            ic50_nm=0.3, max_effect=0.75, selectivity=0.85
        )
    
    def _validate(self) -> None:
        """Validate database integrity."""
        if len(self.drugs) == 0:
            raise ValidationError("Drug database is empty")
        
        # Check for drugs with invalid pathway references
        all_pathway_genes = self.pathway_db.get_all_genes()
        
        for drug_name, drug in self.drugs.items():
            # Check target genes exist in pathway database
            unknown_genes = [g for g in drug.target_genes 
                           if g not in all_pathway_genes]
            if unknown_genes:
                self.logger.debug(
                    f"  Drug {drug_name} has targets not in pathways: {unknown_genes}"
                )
        
        self.logger.debug(f"[OK] DrugDatabase validation passed")
    
    def get_drug(self, name: str) -> Optional[Drug]:
        """
        Get a drug by name.
        
        Args:
            name: Drug name (case-insensitive)
            
        Returns:
            Drug object or None if not found
        """
        return self.drugs.get(name.lower())
    
    def get_drugs_by_mechanism(self, mechanism: DrugMechanism) -> List[Drug]:
        """
        Get all drugs with a specific mechanism.
        
        Args:
            mechanism: Drug mechanism of action
            
        Returns:
            List of matching drugs
        """
        return [d for d in self.drugs.values() if d.mechanism == mechanism]
    
    def get_drugs_by_pathway(self, pathway: CancerPathway) -> List[Drug]:
        """
        Get all drugs targeting a pathway.
        
        Args:
            pathway: Cancer pathway
            
        Returns:
            List of drugs targeting the pathway
        """
        return [d for d in self.drugs.values() if pathway in d.target_pathways]
    
    def search_drugs(self, query: str) -> List[Drug]:
        """
        Search drugs by name or target.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching drugs
        """
        query = query.lower()
        results = []
        
        for drug in self.drugs.values():
            # Match name
            if query in drug.name.lower():
                results.append(drug)
                continue
            
            # Match target genes
            if any(query in gene.lower() for gene in drug.target_genes):
                results.append(drug)
                continue
            
            # Match mechanism
            if query in drug.mechanism.name.lower():
                results.append(drug)
        
        return results
    
    def get_all_drugs(self) -> List[Drug]:
        """Get all drugs in the database."""
        return list(self.drugs.values())
    
    def get_drug_names(self) -> List[str]:
        """Get all drug names."""
        return list(self.drugs.keys())


# =============================================================================
# CELL LINE DATABASE
# =============================================================================

class CellLineDatabase:
    """
    Database of cancer cell lines with molecular profiles.
    
    Contains detailed molecular characterization of cancer cell lines
    including oncogenes, tumor suppressors, mutations, and pathway activity.
    
    Example:
        >>> cell_db = CellLineDatabase(logger)
        >>> a549 = cell_db.get_cell_line("A549")
        >>> print(f"A549 oncogenes: {a549.oncogenes}")
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the cell line database.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.cell_lines: List[CellLine] = []
        self.cell_line_dict: Dict[str, CellLine] = {}
        
        self._initialize_cell_lines()
        
        self.logger.info(f"[OK] CellLineDatabase initialized: "
                        f"{len(self.cell_lines)} cell lines")
    
    def _initialize_cell_lines(self) -> None:
        """Initialize all cell lines with complete profiles."""
        
        # A549 - Non-small cell lung cancer
        a549 = CellLine(
            name="A549",
            cancer_type="Non-small cell lung cancer",
            tissue_origin="Lung",
            oncogenes=["KRAS", "MYC", "EGFR", "PIK3CA", "STK11"],
            tumor_suppressors=["TP53", "CDKN2A", "STK11", "KEAP1"],
            driver_mutations=[
                "KRAS_G12S", "STK11_Q37*", "KEAP1_G333C", "SMARCA4_Q598*"
            ],
            active_pathways=[
                CancerPathway.MAPK_ERK,
                CancerPathway.PI3K_AKT_MTOR,
                CancerPathway.NF_KAPPA_B,
                CancerPathway.CELL_CYCLE
            ],
            suppressed_pathways=[
                CancerPathway.P53_APOPTOSIS,
                CancerPathway.IMMUNE_CHECKPOINT
            ],
            doubling_time_hours=22.0,
            immune_infiltrate="low"
        )
        
        # MCF7 - Breast cancer
        mcf7 = CellLine(
            name="MCF7",
            cancer_type="Breast adenocarcinoma",
            tissue_origin="Breast",
            oncogenes=["ESR1", "PIK3CA", "CCND1", "MYC", "GATA3"],
            tumor_suppressors=["RB1", "PTEN", "CDH1", "MAP3K1"],
            driver_mutations=[
                "PIK3CA_E545K", "GATA3_D336fs", "MAP3K1_S1330*", "CDH1_loss"
            ],
            active_pathways=[
                CancerPathway.PI3K_AKT_MTOR,
                CancerPathway.CELL_CYCLE,
                CancerPathway.WNT_BETA_CATENIN,
                CancerPathway.EPIGENETIC
            ],
            suppressed_pathways=[
                CancerPathway.P53_APOPTOSIS,
                CancerPathway.DNA_REPAIR
            ],
            doubling_time_hours=29.0,
            immune_infiltrate="moderate"
        )
        
        # HCT116 - Colorectal cancer
        hct116 = CellLine(
            name="HCT116",
            cancer_type="Colorectal carcinoma",
            tissue_origin="Colon",
            oncogenes=["KRAS", "PIK3CA", "CTNNB1", "MYC", "CCND1"],
            tumor_suppressors=["CDKN2A", "APC", "MLH1", "MSH2"],
            driver_mutations=[
                "KRAS_G13D", "PIK3CA_H1047R", "CTNNB1_S45del", "MLH1_silenced"
            ],
            active_pathways=[
                CancerPathway.MAPK_ERK,
                CancerPathway.PI3K_AKT_MTOR,
                CancerPathway.WNT_BETA_CATENIN,
                CancerPathway.CELL_CYCLE
            ],
            suppressed_pathways=[
                CancerPathway.DNA_REPAIR,
                CancerPathway.TGFB
            ],
            doubling_time_hours=21.0,
            immune_infiltrate="high"  # MSI-H
        )
        
        self.cell_lines = [a549, mcf7, hct116]
        self.cell_line_dict = {cl.name: cl for cl in self.cell_lines}
    
    def get_cell_line(self, name: str) -> Optional[CellLine]:
        """
        Get a cell line by name.
        
        Args:
            name: Cell line name
            
        Returns:
            CellLine object or None if not found
        """
        return self.cell_line_dict.get(name)
    
    def get_all_cell_lines(self) -> List[CellLine]:
        """Get all cell lines."""
        return self.cell_lines.copy()
    
    def get_cell_line_names(self) -> List[str]:
        """Get all cell line names."""
        return [cl.name for cl in self.cell_lines]


# =============================================================================
# GENE EXPRESSION GENERATOR
# =============================================================================

class GeneExpressionGenerator:
    """
    Generator for synthetic single-cell gene expression data.
    
    Creates realistic gene expression profiles incorporating:
    - Baseline expression patterns
    - Cell line-specific alterations
    - Drug perturbation effects
    - Biological and technical noise
    - Dropout/sparsity patterns
    
    Example:
        >>> gen = GeneExpressionGenerator(config, pathway_db, drug_db, cell_db, logger)
        >>> baseline = gen.generate_baseline_expression(cell_line, n_cells=100)
        >>> treated = gen.apply_drug_effect(baseline, drug, dose=0.5)
    """
    
    def __init__(self, 
                 config: ProjectConfig,
                 pathway_db: PathwayDatabase,
                 drug_db: DrugDatabase,
                 cell_line_db: CellLineDatabase,
                 logger: logging.Logger):
        """
        Initialize the gene expression generator.
        
        Args:
            config: Project configuration
            pathway_db: Pathway database
            drug_db: Drug database
            cell_line_db: Cell line database
            logger: Logger instance
        """
        self.config = config
        self.pathway_db = pathway_db
        self.drug_db = drug_db
        self.cell_line_db = cell_line_db
        self.logger = logger
        
        # Create gene list
        self.gene_names = self._create_gene_list()
        self.n_genes = len(self.gene_names)
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.gene_names)}
        
        # Create pathway-gene matrix
        self.pathway_gene_matrix = self._create_pathway_gene_matrix()
        
        # Validate dimensions
        if self.n_genes != config.n_genes:
            self.logger.warning(
                f"[!] Gene count mismatch: created {self.n_genes}, "
                f"expected {config.n_genes}"
            )
        
        self.logger.info(f"[OK] GeneExpressionGenerator initialized: "
                        f"{self.n_genes} genes, {config.n_pathways} pathways")
    
    def _create_gene_list(self) -> List[str]:
        """
        Create a list of gene names for the expression matrix.
        
        Returns:
            List of exactly config.n_genes gene names
        """
        genes = set()
        
        # Add all pathway genes
        for pathway in CancerPathway:
            genes.update(self.pathway_db.get_pathway_genes(pathway))
        
        # Add additional genes to reach target count
        gene_list = list(genes)
        
        # Generate additional gene names if needed
        additional_needed = self.config.n_genes - len(gene_list)
        if additional_needed > 0:
            for i in range(additional_needed):
                gene_name = f"GENE_{i+1:04d}"
                while gene_name in genes:
                    i += 1
                    gene_name = f"GENE_{i+1:04d}"
                gene_list.append(gene_name)
        elif additional_needed < 0:
            # Truncate if we have too many
            gene_list = gene_list[:self.config.n_genes]
        
        return gene_list
    
    def _create_pathway_gene_matrix(self) -> np.ndarray:
        """
        Create binary pathway membership matrix.
        
        Returns:
            Matrix of shape (n_genes, n_pathways) with 1s indicating membership
        """
        return self.pathway_db.get_pathway_matrix(self.gene_names)
    
    def generate_baseline_expression(self, 
                                     cell_line: CellLine,
                                     n_cells: int = None) -> np.ndarray:
        """
        Generate baseline expression for a cell line.
        
        Args:
            cell_line: Cell line to generate expression for
            n_cells: Number of cells (default: config.n_cells_per_sample)
            
        Returns:
            Expression matrix of shape (n_cells, n_genes)
        """
        try:
            if n_cells is None:
                n_cells = self.config.n_cells_per_sample
            
            if n_cells <= 0:
                raise ValueError("n_cells must be positive")
            
            # Initialize with log-normal distributed expression
            # Mean expression around 2-4 in log2 space
            mean_expr = np.random.uniform(1.5, 4.0, size=self.n_genes)
            std_expr = np.random.uniform(0.3, 1.0, size=self.n_genes)
            
            # Generate expression for each cell
            expression = np.zeros((n_cells, self.n_genes), dtype=np.float32)
            for i in range(n_cells):
                expression[i, :] = np.random.normal(mean_expr, std_expr)
            
            # Apply cell line-specific genetic alterations
            expression = self.apply_genetic_alterations(expression, cell_line)
            
            # Add biological noise
            expression = self.add_biological_noise(expression)
            
            # Add technical noise
            expression = self.add_technical_noise(expression)
            
            # Apply dropout
            expression = self.apply_dropout(expression)
            
            # Normalize
            expression = self._normalize_expression(expression)
            
            # Validate output
            self._validate_expression(expression, "baseline")
            
            return expression
            
        except Exception as e:
            self.logger.error(f"[X] Error generating baseline expression: {e}")
            raise DataGenerationError(f"Baseline expression generation failed: {e}") from e
    
    def apply_genetic_alterations(self, 
                                  expression: np.ndarray,
                                  cell_line: CellLine) -> np.ndarray:
        """
        Apply cell line-specific genetic alterations to expression.
        
        Args:
            expression: Base expression matrix (n_cells, n_genes)
            cell_line: Cell line with alterations
            
        Returns:
            Modified expression matrix
        """
        expression = expression.copy()
        
        # Upregulate oncogenes
        for gene in cell_line.oncogenes:
            if gene in self.gene_to_idx:
                idx = self.gene_to_idx[gene]
                expression[:, idx] *= np.random.uniform(1.5, 3.0)
        
        # Downregulate/silence tumor suppressors
        for gene in cell_line.tumor_suppressors:
            if gene in self.gene_to_idx:
                idx = self.gene_to_idx[gene]
                expression[:, idx] *= np.random.uniform(0.1, 0.5)
        
        # Activate pathway genes for active pathways
        for pathway in cell_line.active_pathways:
            pathway_idx = list(CancerPathway).index(pathway)
            pathway_mask = self.pathway_gene_matrix[:, pathway_idx] > 0
            expression[:, pathway_mask] *= np.random.uniform(1.2, 1.8)
        
        # Suppress pathway genes for suppressed pathways
        for pathway in cell_line.suppressed_pathways:
            pathway_idx = list(CancerPathway).index(pathway)
            pathway_mask = self.pathway_gene_matrix[:, pathway_idx] > 0
            expression[:, pathway_mask] *= np.random.uniform(0.4, 0.8)
        
        return expression
    
    def add_biological_noise(self, expression: np.ndarray) -> np.ndarray:
        """
        Add cell-to-cell biological variability.
        
        Args:
            expression: Expression matrix (n_cells, n_genes)
            
        Returns:
            Expression with biological noise
        """
        # Log-normal multiplicative noise
        noise = np.random.lognormal(mean=0, sigma=0.3, size=expression.shape)
        return expression * noise
    
    def add_technical_noise(self, expression: np.ndarray) -> np.ndarray:
        """
        Add sequencing/technical noise.
        
        Args:
            expression: Expression matrix (n_cells, n_genes)
            
        Returns:
            Expression with technical noise
        """
        # Poisson-like noise proportional to expression level
        noise_level = 0.1
        noise = np.random.normal(0, noise_level * np.abs(expression) + 0.01)
        return expression + noise
    
    def apply_dropout(self, expression: np.ndarray) -> np.ndarray:
        """
        Apply dropout to create sparse expression pattern.
        
        Args:
            expression: Expression matrix (n_cells, n_genes)
            
        Returns:
            Sparse expression matrix
        """
        # Dropout probability inversely related to expression level
        # Lower expressed genes have higher dropout
        dropout_prob = np.exp(-0.5 * np.abs(expression))
        dropout_prob = np.clip(dropout_prob, 0.05, 0.8)
        
        dropout_mask = np.random.random(expression.shape) > dropout_prob
        return expression * dropout_mask
    
    def _normalize_expression(self, expression: np.ndarray) -> np.ndarray:
        """
        Normalize expression to 0-10 range.
        
        Args:
            expression: Raw expression matrix
            
        Returns:
            Normalized expression matrix
        """
        # Ensure non-negative
        expression = np.maximum(expression, 0)
        
        # Log-transform (add pseudocount)
        expression = np.log2(expression + 1)
        
        # Scale to 0-10 range
        max_val = np.percentile(expression, 99)
        if max_val > 0:
            expression = expression / max_val * 10
        
        expression = np.clip(expression, 0, 10)
        
        return expression.astype(np.float32)
    
    def apply_drug_effect(self,
                         baseline_expression: np.ndarray,
                         drug: Drug,
                         dose: float = 1.0) -> np.ndarray:
        """
        Apply drug perturbation to gene expression.
        
        Args:
            baseline_expression: (n_cells, n_genes) expression matrix
            drug: Drug object
            dose: Dose level (0-1, where 1 is maximum dose)
            
        Returns:
            Perturbed expression matrix of same shape
        """
        try:
            # Input validation
            if baseline_expression is None or baseline_expression.size == 0:
                raise ValueError("baseline_expression cannot be empty")
            
            if drug is None:
                raise ValueError("drug cannot be None")
            
            if dose < 0 or dose > 1:
                raise ValueError(f"dose must be in [0, 1], got {dose}")
            
            # Copy to avoid modifying input
            expression = baseline_expression.copy()
            
            # Calculate effective dose (with saturation)
            effective_dose = min(dose, 1.0)
            
            # Apply direct target gene effects
            for target_gene in drug.target_genes:
                if target_gene in self.gene_to_idx:
                    gene_idx = self.gene_to_idx[target_gene]
                    effect_magnitude = drug.max_effect * effective_dose
                    
                    # Different effects by mechanism
                    if drug.mechanism in [DrugMechanism.MEK_INHIBITOR,
                                         DrugMechanism.BRAF_INHIBITOR,
                                         DrugMechanism.PI3K_INHIBITOR,
                                         DrugMechanism.MTOR_INHIBITOR,
                                         DrugMechanism.PARP_INHIBITOR,
                                         DrugMechanism.CDK4_6_INHIBITOR,
                                         DrugMechanism.BCL2_INHIBITOR,
                                         DrugMechanism.EGFR_INHIBITOR,
                                         DrugMechanism.HER2_INHIBITOR,
                                         DrugMechanism.ALK_INHIBITOR,
                                         DrugMechanism.VEGF_INHIBITOR,
                                         DrugMechanism.CK2_INHIBITOR,
                                         DrugMechanism.HDAC_INHIBITOR,
                                         DrugMechanism.BET_INHIBITOR,
                                         DrugMechanism.JAK_INHIBITOR,
                                         DrugMechanism.SRC_INHIBITOR,
                                         DrugMechanism.WNT_INHIBITOR,
                                         DrugMechanism.NOTCH_INHIBITOR]:
                        # Inhibitors reduce expression
                        effect = 1.0 - effect_magnitude
                        expression[:, gene_idx] *= effect
                    
                    elif drug.mechanism in [DrugMechanism.PD1_ANTIBODY,
                                           DrugMechanism.PDL1_ANTIBODY,
                                           DrugMechanism.CTLA4_ANTIBODY,
                                           DrugMechanism.INTERFERON]:
                        # Immunotherapies increase expression
                        effect = 1.0 + effect_magnitude * 0.5
                        expression[:, gene_idx] *= effect
                    
                    elif drug.mechanism in [DrugMechanism.PLATINUM_CHEMO,
                                           DrugMechanism.TAXANE_CHEMO,
                                           DrugMechanism.ANTIMETABOLITE]:
                        # Chemotherapies broadly reduce expression
                        effect = 1.0 - effect_magnitude * 0.7
                        expression[:, gene_idx] *= effect
                    
                    else:
                        # Default: moderate reduction
                        effect = 1.0 - effect_magnitude * 0.5
                        expression[:, gene_idx] *= effect
            
            # Apply pathway-level effects
            pathway_to_idx = {p: i for i, p in enumerate(CancerPathway)}
            
            for pathway in drug.target_pathways:
                if pathway in pathway_to_idx:
                    pathway_idx = pathway_to_idx[pathway]
                    
                    # Get genes in this pathway
                    pathway_genes_mask = self.pathway_gene_matrix[:, pathway_idx] > 0
                    
                    # Apply weaker pathway-level effect
                    pathway_effect_magnitude = drug.max_effect * effective_dose * 0.3
                    pathway_effect = 1.0 - pathway_effect_magnitude
                    
                    expression[:, pathway_genes_mask] *= pathway_effect
            
            # Add stochastic variation
            noise = np.random.normal(1.0, 0.05, size=expression.shape)
            expression *= noise
            
            # Ensure non-negative
            expression = np.maximum(expression, 0)
            
            # Validate output
            if np.isnan(expression).any():
                raise ValidationError(f"Drug effect calculation produced NaN")
            
            self.logger.debug(
                f"[OK] Applied {drug.name} effect (dose={dose:.2f}): "
                f"mean change={(expression.mean() / baseline_expression.mean() - 1)*100:.1f}%"
            )
            
            return expression.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"[X] Error applying drug effect for {drug.name}: {e}")
            raise DataGenerationError(f"Drug effect application failed: {e}") from e
    
    def _validate_expression(self, expression: np.ndarray, context: str) -> None:
        """Validate expression matrix."""
        if expression.ndim != 2:
            raise ValidationError(
                f"Expression must be 2D, got {expression.ndim}D ({context})"
            )
        
        if np.isnan(expression).any():
            n_nan = np.isnan(expression).sum()
            raise ValidationError(
                f"Expression contains {n_nan} NaN values ({context})"
            )
        
        if np.isinf(expression).any():
            n_inf = np.isinf(expression).sum()
            raise ValidationError(
                f"Expression contains {n_inf} Inf values ({context})"
            )
        
        if (expression < 0).any():
            n_neg = (expression < 0).sum()
            raise ValidationError(
                f"Expression contains {n_neg} negative values ({context})"
            )
        
        self.logger.debug(
            f"[OK] Expression validated ({context}): "
            f"shape={expression.shape}, "
            f"mean={expression.mean():.3f}, "
            f"std={expression.std():.3f}, "
            f"sparsity={(expression == 0).mean()*100:.1f}%"
        )


# =============================================================================
# SYNERGY DATA GENERATOR
# =============================================================================

class SynergyDataGenerator:
    """
    Generator for drug synergy training data.
    
    Creates balanced datasets with control samples, single-drug treatments,
    and drug combinations with calculated synergy scores.
    
    Example:
        >>> gen = SynergyDataGenerator(config, expr_gen, drug_db, cell_db, logger)
        >>> expressions, metadata, labels = gen.generate_training_data()
    """
    
    def __init__(self,
                 config: ProjectConfig,
                 expression_generator: GeneExpressionGenerator,
                 drug_db: DrugDatabase,
                 cell_line_db: CellLineDatabase,
                 logger: logging.Logger):
        """
        Initialize the synergy data generator.
        
        Args:
            config: Project configuration
            expression_generator: Gene expression generator
            drug_db: Drug database
            cell_line_db: Cell line database
            logger: Logger instance
        """
        self.config = config
        self.expression_generator = expression_generator
        self.drug_db = drug_db
        self.cell_line_db = cell_line_db
        self.logger = logger
    
    def generate_training_data(self) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Generate complete training dataset.
        
        Returns:
            Tuple of:
                - expressions: (n_samples, n_cells, n_genes) array
                - metadata: List of sample metadata dictionaries
                - synergy_labels: (n_samples,) array of synergy scores
        """
        try:
            self.logger.info("Generating training data...")
            
            expressions_list = []
            metadata_list = []
            synergy_labels_list = []
            
            cell_lines = self.cell_line_db.get_all_cell_lines()
            drugs = self.drug_db.get_all_drugs()
            
            # Calculate samples per cell line
            n_controls_per_cl = self.config.n_control_samples // len(cell_lines)
            n_single_per_cl = self.config.n_single_drug_samples // len(cell_lines)
            n_combo_per_cl = self.config.n_combination_samples // len(cell_lines)
            
            for cell_line in tqdm(cell_lines, desc="Processing cell lines"):
                # Generate baseline expression for this cell line
                self.logger.debug(f"  Generating data for {cell_line.name}")
                
                # 1. Control samples (no drug)
                for _ in range(n_controls_per_cl):
                    expr = self.expression_generator.generate_baseline_expression(
                        cell_line,
                        n_cells=self.config.n_cells_per_sample
                    )
                    expressions_list.append(expr)
                    metadata_list.append({
                        'cell_line': cell_line.name,
                        'drug1': None,
                        'drug2': None,
                        'dose1': 0.0,
                        'dose2': 0.0,
                        'is_control': True,
                        'is_combination': False
                    })
                    synergy_labels_list.append(0.0)
                
                # 2. Single drug samples
                for _ in range(n_single_per_cl):
                    drug = random.choice(drugs)
                    dose = np.random.uniform(0.3, 1.0)
                    
                    baseline = self.expression_generator.generate_baseline_expression(
                        cell_line,
                        n_cells=self.config.n_cells_per_sample
                    )
                    expr = self.expression_generator.apply_drug_effect(
                        baseline, drug, dose
                    )
                    
                    expressions_list.append(expr)
                    metadata_list.append({
                        'cell_line': cell_line.name,
                        'drug1': drug.name,
                        'drug2': None,
                        'dose1': dose,
                        'dose2': 0.0,
                        'is_control': False,
                        'is_combination': False
                    })
                    synergy_labels_list.append(0.0)
                
                # 3. Drug combination samples
                drug_pairs = list(combinations(drugs, 2))
                selected_pairs = random.sample(
                    drug_pairs, 
                    min(n_combo_per_cl, len(drug_pairs))
                )
                
                for drug1, drug2 in selected_pairs:
                    dose1 = np.random.uniform(0.3, 1.0)
                    dose2 = np.random.uniform(0.3, 1.0)
                    
                    # Generate baseline
                    baseline = self.expression_generator.generate_baseline_expression(
                        cell_line,
                        n_cells=self.config.n_cells_per_sample
                    )
                    
                    # Apply single drug effects for Bliss calculation
                    expr_drug1 = self.expression_generator.apply_drug_effect(
                        baseline, drug1, dose1
                    )
                    expr_drug2 = self.expression_generator.apply_drug_effect(
                        baseline, drug2, dose2
                    )
                    
                    # Calculate individual effects
                    effect1 = 1 - (expr_drug1.mean() / baseline.mean())
                    effect2 = 1 - (expr_drug2.mean() / baseline.mean())
                    
                    # Apply combination effect
                    expr_combo = self.expression_generator.apply_drug_effect(
                        expr_drug1, drug2, dose2
                    )
                    
                    # Calculate observed effect
                    observed_effect = 1 - (expr_combo.mean() / baseline.mean())
                    
                    # Calculate synergy using Bliss independence
                    expected_effect = effect1 + effect2 - (effect1 * effect2)
                    synergy = observed_effect - expected_effect
                    
                    # Add noise to synergy score
                    synergy += np.random.normal(0, 0.05)
                    
                    # Determine if this is synergistic based on mechanisms
                    synergy = self._adjust_synergy_by_mechanism(
                        synergy, drug1, drug2, cell_line
                    )
                    
                    expressions_list.append(expr_combo)
                    metadata_list.append({
                        'cell_line': cell_line.name,
                        'drug1': drug1.name,
                        'drug2': drug2.name,
                        'dose1': dose1,
                        'dose2': dose2,
                        'is_control': False,
                        'is_combination': True
                    })
                    synergy_labels_list.append(synergy)
            
            # Convert to arrays
            expressions = np.array(expressions_list, dtype=np.float32)
            synergy_labels = np.array(synergy_labels_list, dtype=np.float32)
            
            # Validate
            self._validate_training_data(expressions, metadata_list, synergy_labels)
            
            self.logger.info(f"[OK] Generated {len(metadata_list)} samples")
            self.logger.info(f"  Controls: {sum(1 for m in metadata_list if m['is_control'])}")
            self.logger.info(f"  Single drug: {sum(1 for m in metadata_list if not m['is_control'] and not m['is_combination'])}")
            self.logger.info(f"  Combinations: {sum(1 for m in metadata_list if m['is_combination'])}")
            self.logger.info(f"  Synergy range: [{synergy_labels.min():.3f}, {synergy_labels.max():.3f}]")
            
            return expressions, metadata_list, synergy_labels
            
        except Exception as e:
            self.logger.error(f"[X] Error generating training data: {e}", exc_info=True)
            raise DataGenerationError(f"Training data generation failed: {e}") from e
    
    def _adjust_synergy_by_mechanism(self,
                                     base_synergy: float,
                                     drug1: Drug,
                                     drug2: Drug,
                                     cell_line: CellLine) -> float:
        """
        Adjust synergy score based on known drug interaction patterns.
        
        Args:
            base_synergy: Initial synergy score
            drug1: First drug
            drug2: Second drug
            cell_line: Cell line context
            
        Returns:
            Adjusted synergy score
        """
        synergy = base_synergy
        
        # Known synergistic combinations
        synergistic_pairs = [
            (DrugMechanism.MEK_INHIBITOR, DrugMechanism.BRAF_INHIBITOR),
            (DrugMechanism.PI3K_INHIBITOR, DrugMechanism.MEK_INHIBITOR),
            (DrugMechanism.CDK4_6_INHIBITOR, DrugMechanism.PI3K_INHIBITOR),
            (DrugMechanism.PARP_INHIBITOR, DrugMechanism.PLATINUM_CHEMO),
            (DrugMechanism.PD1_ANTIBODY, DrugMechanism.CTLA4_ANTIBODY),
            (DrugMechanism.CK2_INHIBITOR, DrugMechanism.INTERFERON),
        ]
        
        for mech1, mech2 in synergistic_pairs:
            if ((drug1.mechanism == mech1 and drug2.mechanism == mech2) or
                (drug1.mechanism == mech2 and drug2.mechanism == mech1)):
                synergy += np.random.uniform(0.1, 0.3)
        
        # Drugs targeting same pathway may be redundant (antagonistic)
        shared_pathways = set(drug1.target_pathways) & set(drug2.target_pathways)
        if len(shared_pathways) > 1:
            synergy -= np.random.uniform(0.05, 0.15)
        
        # Cell line context
        for pathway in drug1.target_pathways:
            if pathway in cell_line.active_pathways:
                synergy += np.random.uniform(0.02, 0.08)
        
        for pathway in drug2.target_pathways:
            if pathway in cell_line.active_pathways:
                synergy += np.random.uniform(0.02, 0.08)
        
        return np.clip(synergy, -1.0, 1.0)
    
    def _validate_training_data(self,
                                expressions: np.ndarray,
                                metadata: List[Dict],
                                synergy_labels: np.ndarray) -> None:
        """Validate generated training data."""
        if expressions.shape[0] != len(metadata):
            raise ValidationError(
                f"Expression and metadata length mismatch: "
                f"{expressions.shape[0]} vs {len(metadata)}"
            )
        
        if expressions.shape[0] != len(synergy_labels):
            raise ValidationError(
                f"Expression and labels length mismatch: "
                f"{expressions.shape[0]} vs {len(synergy_labels)}"
            )
        
        if np.isnan(expressions).any():
            raise ValidationError("Expression data contains NaN values")
        
        if np.isnan(synergy_labels).any():
            raise ValidationError("Synergy labels contain NaN values")


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class SingleCellDatasetTorch(Dataset):
    """
    PyTorch Dataset for single-cell drug synergy data.
    
    Handles encoding of expression data, drug features, and cell line
    information for model training.
    
    Example:
        >>> dataset = SingleCellDatasetTorch(expressions, metadata, labels, drug_db, logger)
        >>> sample = dataset[0]
        >>> print(sample['expression'].shape)
    """
    
    def __init__(self,
                 expressions: np.ndarray,
                 metadata: List[Dict],
                 synergy_labels: np.ndarray,
                 drug_db: DrugDatabase,
                 logger: logging.Logger):
        """
        Initialize the dataset.
        
        Args:
            expressions: (n_samples, n_cells, n_genes) expression array
            metadata: List of sample metadata dictionaries
            synergy_labels: (n_samples,) synergy scores
            drug_db: Drug database for encoding
            logger: Logger instance
        """
        self.expressions = expressions
        self.metadata = metadata
        self.synergy_labels = synergy_labels
        self.drug_db = drug_db
        self.logger = logger
        
        self.n_samples = len(metadata)
        self.n_genes = expressions.shape[2]
        self.n_pathways = len(CancerPathway)
        self.n_mechanisms = len(DrugMechanism)
        
        # Create encodings
        self._create_drug_encodings()
        self._create_cell_line_encodings()
        
        self.logger.info(f"[OK] SingleCellDatasetTorch initialized: {self.n_samples} samples")
    
    def _create_drug_encodings(self) -> None:
        """Create feature encodings for all drugs."""
        self.drug_encodings = {}
        
        for drug in self.drug_db.get_all_drugs():
            # Mechanism one-hot
            mechanism_onehot = np.zeros(self.n_mechanisms, dtype=np.float32)
            mechanism_onehot[drug.mechanism.value - 1] = 1.0
            
            # Pathway targets multi-hot
            pathway_multihot = np.zeros(self.n_pathways, dtype=np.float32)
            for pathway in drug.target_pathways:
                pathway_multihot[pathway.value - 1] = 1.0
            
            # Gene targets multi-hot
            gene_multihot = np.zeros(self.n_genes, dtype=np.float32)
            for gene in drug.target_genes:
                # Simple encoding based on hash
                gene_idx = hash(gene) % self.n_genes
                gene_multihot[gene_idx] = 1.0
            
            self.drug_encodings[drug.name] = {
                'mechanism': mechanism_onehot,
                'pathways': pathway_multihot,
                'genes': gene_multihot,
                'max_effect': np.array([drug.max_effect], dtype=np.float32),
                'selectivity': np.array([drug.selectivity], dtype=np.float32),
                'is_immunotherapy': np.array([float(drug.is_immunotherapy)], dtype=np.float32),
                'is_chemotherapy': np.array([float(drug.is_chemotherapy)], dtype=np.float32)
            }
        
        # Create null encoding for missing drugs
        self.null_drug_encoding = {
            'mechanism': np.zeros(self.n_mechanisms, dtype=np.float32),
            'pathways': np.zeros(self.n_pathways, dtype=np.float32),
            'genes': np.zeros(self.n_genes, dtype=np.float32),
            'max_effect': np.zeros(1, dtype=np.float32),
            'selectivity': np.zeros(1, dtype=np.float32),
            'is_immunotherapy': np.zeros(1, dtype=np.float32),
            'is_chemotherapy': np.zeros(1, dtype=np.float32)
        }
    
    def _create_cell_line_encodings(self) -> None:
        """Create encodings for cell lines."""
        cell_lines = ['A549', 'MCF7', 'HCT116']
        self.cell_line_to_idx = {name: i for i, name in enumerate(cell_lines)}
        self.n_cell_lines = len(cell_lines)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with all sample tensors
        """
        try:
            # Get expression (average across cells)
            expr = self.expressions[idx]  # (n_cells, n_genes)
            expr_mean = expr.mean(axis=0)  # (n_genes,)
            
            # Get metadata
            meta = self.metadata[idx]
            
            # Encode drug1
            drug1_name = meta['drug1']
            if drug1_name and drug1_name in self.drug_encodings:
                drug1_enc = self.drug_encodings[drug1_name]
            else:
                drug1_enc = self.null_drug_encoding
            
            # Encode drug2
            drug2_name = meta['drug2']
            if drug2_name and drug2_name in self.drug_encodings:
                drug2_enc = self.drug_encodings[drug2_name]
            else:
                drug2_enc = self.null_drug_encoding
            
            # Encode cell line
            cell_line_idx = self.cell_line_to_idx.get(meta['cell_line'], 0)
            
            # Build sample dictionary
            sample = {
                'expression': torch.tensor(expr_mean, dtype=torch.float32),
                'synergy_label': torch.tensor(self.synergy_labels[idx], dtype=torch.float32),
                'cell_line_idx': torch.tensor(cell_line_idx, dtype=torch.long),
                'drug1_mechanism': torch.tensor(drug1_enc['mechanism'], dtype=torch.float32),
                'drug1_pathways': torch.tensor(drug1_enc['pathways'], dtype=torch.float32),
                'drug1_genes': torch.tensor(drug1_enc['genes'], dtype=torch.float32),
                'drug1_max_effect': torch.tensor(drug1_enc['max_effect'], dtype=torch.float32),
                'drug1_selectivity': torch.tensor(drug1_enc['selectivity'], dtype=torch.float32),
                'drug1_is_immuno': torch.tensor(drug1_enc['is_immunotherapy'], dtype=torch.float32),
                'drug2_mechanism': torch.tensor(drug2_enc['mechanism'], dtype=torch.float32),
                'drug2_pathways': torch.tensor(drug2_enc['pathways'], dtype=torch.float32),
                'drug2_genes': torch.tensor(drug2_enc['genes'], dtype=torch.float32),
                'drug2_max_effect': torch.tensor(drug2_enc['max_effect'], dtype=torch.float32),
                'drug2_selectivity': torch.tensor(drug2_enc['selectivity'], dtype=torch.float32),
                'drug2_is_immuno': torch.tensor(drug2_enc['is_immunotherapy'], dtype=torch.float32),
                'dose1': torch.tensor(meta['dose1'], dtype=torch.float32),
                'dose2': torch.tensor(meta['dose2'], dtype=torch.float32),
                'is_control': torch.tensor(float(meta['is_control']), dtype=torch.float32),
                'is_combination': torch.tensor(float(meta['is_combination']), dtype=torch.float32)
            }
            
            return sample
            
        except Exception as e:
            self.logger.error(f"[X] Error getting sample {idx}: {e}")
            raise


# =============================================================================
# FOUNDATION MODEL
# =============================================================================

class CancerSynergyFoundationModel(nn.Module):
    """
    Deep learning model for cancer drug synergy prediction.
    
    Architecture:
        - Gene expression encoder
        - Drug feature encoders (x2)
        - Cell line embedding
        - Transformer-based integration
        - Synergy prediction head
        - Expression reconstruction head (auxiliary)
    
    Example:
        >>> model = CancerSynergyFoundationModel(config, logger)
        >>> outputs = model(batch)
        >>> synergy_pred = outputs['synergy_pred']
    """
    
    def __init__(self, config: ProjectConfig, logger: logging.Logger):
        """
        Initialize the model.
        
        Args:
            config: Project configuration
            logger: Logger instance
        """
        super().__init__()
        self.config = config
        self.logger = logger
        
        # Dimensions
        self.n_genes = config.n_genes
        self.n_pathways = len(CancerPathway)
        self.n_mechanisms = len(DrugMechanism)
        self.n_cell_lines = config.n_cell_lines
        
        # Gene expression encoder
        self.expr_encoder = nn.Sequential(
            nn.Linear(self.n_genes, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.gene_embed_dim),
            nn.LayerNorm(config.gene_embed_dim)
        )
        
        # Drug encoder components
        drug_input_dim = (
            self.n_mechanisms +  # mechanism one-hot
            self.n_pathways +    # pathway multi-hot
            self.n_genes +       # gene targets
            4                    # max_effect, selectivity, is_immuno, dose
        )
        
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.drug_embed_dim),
            nn.LayerNorm(config.drug_embed_dim)
        )
        
        # Cell line embedding
        self.cell_line_embedding = nn.Embedding(
            self.n_cell_lines, 
            config.cell_line_embed_dim
        )
        
        # Combined feature projection
        combined_dim = (
            config.gene_embed_dim + 
            2 * config.drug_embed_dim + 
            config.cell_line_embed_dim
        )
        
        self.feature_projection = nn.Sequential(
            nn.Linear(combined_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.n_attention_heads,
            dim_feedforward=config.latent_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_transformer_layers
        )
        
        # Synergy prediction head
        self.synergy_head = nn.Sequential(
            nn.Linear(config.latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Expression reconstruction head (auxiliary task)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, self.n_genes)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Log model summary
        n_params = sum(p.numel() for p in self.parameters())
        self.logger.info(f"[OK] CancerSynergyFoundationModel initialized: "
                        f"{n_params:,} parameters")
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode_expression(self, expression: torch.Tensor) -> torch.Tensor:
        """
        Encode gene expression features.
        
        Args:
            expression: (batch, n_genes) expression tensor
            
        Returns:
            (batch, gene_embed_dim) encoded features
        """
        return self.expr_encoder(expression)
    
    def encode_drug(self, 
                    mechanism: torch.Tensor,
                    pathways: torch.Tensor,
                    genes: torch.Tensor,
                    max_effect: torch.Tensor,
                    selectivity: torch.Tensor,
                    is_immuno: torch.Tensor,
                    dose: torch.Tensor) -> torch.Tensor:
        """
        Encode drug features.
        
        Args:
            mechanism: (batch, n_mechanisms) one-hot
            pathways: (batch, n_pathways) multi-hot
            genes: (batch, n_genes) multi-hot
            max_effect: (batch, 1) scalar
            selectivity: (batch, 1) scalar
            is_immuno: (batch, 1) boolean
            dose: (batch,) or (batch, 1) dose level
            
        Returns:
            (batch, drug_embed_dim) encoded features
        """
        # Ensure dose is 2D
        if dose.dim() == 1:
            dose = dose.unsqueeze(-1)
        
        # Concatenate all features
        drug_features = torch.cat([
            mechanism,
            pathways,
            genes,
            max_effect,
            selectivity,
            is_immuno,
            dose
        ], dim=-1)
        
        return self.drug_encoder(drug_features)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary with all input tensors
            
        Returns:
            Dictionary with predictions and embeddings
        """
        try:
            # 1. Encode expression
            expr_embed = self.encode_expression(batch['expression'])
            
            # 2. Encode drug1
            drug1_embed = self.encode_drug(
                batch['drug1_mechanism'],
                batch['drug1_pathways'],
                batch['drug1_genes'],
                batch['drug1_max_effect'],
                batch['drug1_selectivity'],
                batch['drug1_is_immuno'],
                batch['dose1']
            )
            
            # 3. Encode drug2
            drug2_embed = self.encode_drug(
                batch['drug2_mechanism'],
                batch['drug2_pathways'],
                batch['drug2_genes'],
                batch['drug2_max_effect'],
                batch['drug2_selectivity'],
                batch['drug2_is_immuno'],
                batch['dose2']
            )
            
            # 4. Encode cell line
            cell_line_embed = self.cell_line_embedding(batch['cell_line_idx'])
            
            # 5. Concatenate all features
            combined = torch.cat([
                expr_embed,
                drug1_embed,
                drug2_embed,
                cell_line_embed
            ], dim=-1)
            
            # 6. Project to latent space
            latent = self.feature_projection(combined)
            
            # 7. Add sequence dimension for transformer
            latent = latent.unsqueeze(1)  # (batch, 1, latent_dim)
            
            # 8. Pass through transformer
            transformed = self.transformer(latent)
            
            # 9. Remove sequence dimension
            transformed = transformed.squeeze(1)  # (batch, latent_dim)
            
            # 10. Predict synergy
            synergy_pred = self.synergy_head(transformed).squeeze(-1)
            
            # 11. Reconstruct expression (auxiliary task)
            expr_recon = self.reconstruction_head(transformed)
            
            return {
                'synergy_pred': synergy_pred,
                'embeddings': transformed,
                'expr_recon': expr_recon
            }
            
        except Exception as e:
            self.logger.error(f"[X] Error in forward pass: {e}")
            raise ModelTrainingError(f"Forward pass failed: {e}") from e


# =============================================================================
# MODEL TRAINER
# =============================================================================

class ModelTrainer:
    """
    Trainer for the cancer synergy foundation model.
    
    Handles training loop, validation, checkpointing, and early stopping.
    
    Example:
        >>> trainer = ModelTrainer(model, config, logger)
        >>> history = trainer.train(train_loader, val_loader)
    """
    
    def __init__(self,
                 model: CancerSynergyFoundationModel,
                 config: ProjectConfig,
                 logger: logging.Logger):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Project configuration
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Loss functions
        self.synergy_loss_fn = nn.MSELoss()
        self.reconstruction_loss_fn = nn.MSELoss()
        
        # Mixed precision
        self.use_amp = config.use_mixed_precision and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        
        self.logger.info(f"[OK] ModelTrainer initialized")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Mixed precision: {self.use_amp}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_synergy_loss = 0.0
        total_recon_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch)
                        
                        synergy_loss = self.synergy_loss_fn(
                            outputs['synergy_pred'],
                            batch['synergy_label']
                        )
                        recon_loss = self.reconstruction_loss_fn(
                            outputs['expr_recon'],
                            batch['expression']
                        )
                        loss = (self.config.synergy_loss_weight * synergy_loss + 
                               self.config.reconstruction_loss_weight * recon_loss)
                    
                    # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch)
                    
                    synergy_loss = self.synergy_loss_fn(
                        outputs['synergy_pred'],
                        batch['synergy_label']
                    )
                    recon_loss = self.reconstruction_loss_fn(
                        outputs['expr_recon'],
                        batch['expression']
                    )
                    loss = (self.config.synergy_loss_weight * synergy_loss + 
                           self.config.reconstruction_loss_weight * recon_loss)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                total_synergy_loss += synergy_loss.item()
                total_recon_loss += recon_loss.item()
                n_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning(f"[!] OOM in training batch, skipping")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        if n_batches == 0:
            raise ModelTrainingError("No batches completed in training epoch")
        
        return {
            'loss': total_loss / n_batches,
            'synergy_loss': total_synergy_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_synergy_loss = 0.0
        total_recon_loss = 0.0
        n_batches = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    # Move to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(batch)
                    
                    # Calculate losses
                    synergy_loss = self.synergy_loss_fn(
                        outputs['synergy_pred'],
                        batch['synergy_label']
                    )
                    recon_loss = self.reconstruction_loss_fn(
                        outputs['expr_recon'],
                        batch['expression']
                    )
                    loss = (self.config.synergy_loss_weight * synergy_loss + 
                           self.config.reconstruction_loss_weight * recon_loss)
                    
                    # Accumulate
                    total_loss += loss.item()
                    total_synergy_loss += synergy_loss.item()
                    total_recon_loss += recon_loss.item()
                    n_batches += 1
                    
                    # Store predictions
                    all_predictions.extend(outputs['synergy_pred'].cpu().numpy())
                    all_labels.extend(batch['synergy_label'].cpu().numpy())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.warning(f"[!] OOM during validation, skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
        
        if n_batches == 0:
            raise ModelTrainingError("No batches completed in validation")
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        mse = mean_squared_error(all_labels, all_predictions)
        mae = mean_absolute_error(all_labels, all_predictions)
        
        # Handle edge case for R²
        if np.var(all_labels) > 0:
            r2 = r2_score(all_labels, all_predictions)
        else:
            r2 = 0.0
        
        return {
            'loss': total_loss / n_batches,
            'synergy_loss': total_synergy_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history dictionary
        """
        try:
            self.logger.info("Starting training...")
            
            history = {
                'train_loss': [],
                'val_loss': [],
                'train_synergy_loss': [],
                'val_synergy_loss': [],
                'val_mse': [],
                'val_mae': [],
                'val_r2': [],
                'learning_rate': []
            }
            
            pbar = tqdm(total=self.config.n_epochs, desc="Training")
            
            for epoch in range(self.config.n_epochs):
                self.current_epoch = epoch
                
                try:
                    # Train
                    train_metrics = self.train_epoch(train_loader)
                    
                    # Validate
                    val_metrics = self.validate(val_loader)
                    
                    # Update scheduler
                    self.scheduler.step()
                    
                    # Record history
                    history['train_loss'].append(train_metrics['loss'])
                    history['val_loss'].append(val_metrics['loss'])
                    history['train_synergy_loss'].append(train_metrics['synergy_loss'])
                    history['val_synergy_loss'].append(val_metrics['synergy_loss'])
                    history['val_mse'].append(val_metrics['mse'])
                    history['val_mae'].append(val_metrics['mae'])
                    history['val_r2'].append(val_metrics['r2'])
                    history['learning_rate'].append(self.scheduler.get_last_lr()[0])
                    
                    # Check for NaN
                    if np.isnan(train_metrics['loss']):
                        raise ModelTrainingError(
                            f"Training collapsed at epoch {epoch}: loss is NaN"
                        )
                    
                    # Early stopping check
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.epochs_without_improvement = 0
                        self.save_checkpoint(
                            self.config.models_dir / "best_model.pt",
                            epoch, val_metrics
                        )
                    else:
                        self.epochs_without_improvement += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'train_loss': f"{train_metrics['loss']:.4f}",
                        'val_loss': f"{val_metrics['loss']:.4f}",
                        'val_r2': f"{val_metrics['r2']:.4f}"
                    })
                    pbar.update(1)
                    
                    # Periodic checkpoint
                    if (epoch + 1) % self.config.save_every_n_epochs == 0:
                        self.save_checkpoint(
                            self.config.models_dir / f"checkpoint_epoch_{epoch+1}.pt",
                            epoch, val_metrics
                        )
                    
                    # Early stopping
                    if self.epochs_without_improvement >= self.config.early_stopping_patience:
                        self.logger.info(
                            f"Early stopping at epoch {epoch}: "
                            f"no improvement for {self.config.early_stopping_patience} epochs"
                        )
                        break
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.error(f"[X] OOM at epoch {epoch}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
            
            pbar.close()
            
            # Log final results
            self.logger.info("")
            self.logger.info("Training Results:")
            self.logger.info("-" * 80)
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            self.logger.info(f"Final metrics:")
            self.logger.info(f"  - MSE: {history['val_mse'][-1]:.4f}")
            self.logger.info(f"  - MAE: {history['val_mae'][-1]:.4f}")
            self.logger.info(f"  - R²: {history['val_r2'][-1]:.4f}")
            
            return history
            
        except KeyboardInterrupt:
            self.logger.warning("[!] Training interrupted by user")
            self.save_checkpoint(
                self.config.models_dir / "interrupted_checkpoint.pt",
                self.current_epoch, {}
            )
            raise
            
        except Exception as e:
            self.logger.error(f"[X] Training failed: {e}", exc_info=True)
            raise ModelTrainingError(f"Training failed: {e}") from e
    
    def save_checkpoint(self,
                       filepath: Path,
                       epoch: int,
                       metrics: Dict[str, float]) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            metrics: Validation metrics
        """
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'metrics': metrics,
                'config': self.config.to_dict()
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            torch.save(checkpoint, filepath)
            self.logger.debug(f"[OK] Checkpoint saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"[X] Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        try:
            if not filepath.exists():
                raise FileNotFoundError(f"Checkpoint not found: {filepath}")
            
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            
            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.logger.info(f"[OK] Checkpoint loaded: {filepath}")
            self.logger.info(f"  Epoch: {checkpoint['epoch']}")
            self.logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"[X] Failed to load checkpoint: {e}")
            raise DataLoadError(f"Checkpoint loading failed: {e}") from e


# =============================================================================
# VIRTUAL SCREENER
# =============================================================================

class VirtualScreener:
    """
    Virtual screening system for drug combinations.
    
    Uses the trained model to predict synergy for all drug combinations
    across cell lines.
    
    Example:
        >>> screener = VirtualScreener(model, config, drug_db, cell_db, expr_gen, dataset, logger)
        >>> results = screener.screen_all_combinations("A549")
        >>> top_hits = screener.get_top_synergies(results, n_top=50)
    """
    
    def __init__(self,
                 model: CancerSynergyFoundationModel,
                 config: ProjectConfig,
                 drug_db: DrugDatabase,
                 cell_line_db: CellLineDatabase,
                 expression_generator: GeneExpressionGenerator,
                 dataset: SingleCellDatasetTorch,
                 logger: logging.Logger):
        """
        Initialize the virtual screener.
        
        Args:
            model: Trained model
            config: Project configuration
            drug_db: Drug database
            cell_line_db: Cell line database
            expression_generator: Expression generator
            dataset: Dataset (for encodings)
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.drug_db = drug_db
        self.cell_line_db = cell_line_db
        self.expression_generator = expression_generator
        self.dataset = dataset
        self.logger = logger
        self.device = torch.device(config.device)
        
        # Set model to eval mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.logger.info("[OK] VirtualScreener initialized")
    
    def screen_all_combinations(self, 
                                cell_line_name: str,
                                dose: float = 0.8) -> pd.DataFrame:
        """
        Screen all drug combinations for a cell line.
        
        Args:
            cell_line_name: Name of cell line
            dose: Dose level (0-1)
            
        Returns:
            DataFrame with predictions sorted by synergy
        """
        try:
            cell_line = self.cell_line_db.get_cell_line(cell_line_name)
            if cell_line is None:
                raise ValidationError(f"Unknown cell line: {cell_line_name}")
            
            self.logger.info(f"Screening all combinations for {cell_line_name}...")
            
            # Generate baseline expression
            baseline = self.expression_generator.generate_baseline_expression(
                cell_line,
                n_cells=self.config.n_cells_per_sample
            )
            expr_mean = baseline.mean(axis=0)  # Average across cells
            
            # Get all drug pairs
            drugs = self.drug_db.get_all_drugs()
            drug_pairs = list(combinations(drugs, 2))
            
            results = []
            
            # Process in batches
            batch_size = 64
            n_batches = (len(drug_pairs) + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for batch_idx in tqdm(range(n_batches), desc="Screening"):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(drug_pairs))
                    batch_pairs = drug_pairs[start_idx:end_idx]
                    
                    # Create batch samples
                    batch_samples = []
                    for drug1, drug2 in batch_pairs:
                        sample = self._create_screening_sample(
                            expr_mean, drug1, drug2, cell_line_name, dose
                        )
                        batch_samples.append(sample)
                    
                    # Collate batch
                    batch = self._collate_screening_batch(batch_samples)
                    
                    # Move to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    
                    # Predict
                    outputs = self.model(batch)
                    predictions = outputs['synergy_pred'].cpu().numpy()
                    
                    # Store results
                    for i, (drug1, drug2) in enumerate(batch_pairs):
                        results.append({
                            'drug1': drug1.name,
                            'drug2': drug2.name,
                            'drug1_mechanism': drug1.mechanism.name,
                            'drug2_mechanism': drug2.mechanism.name,
                            'cell_line': cell_line_name,
                            'dose': dose,
                            'predicted_synergy': predictions[i]
                        })
            
            # Create DataFrame
            df = pd.DataFrame(results)
            df = df.sort_values('predicted_synergy', ascending=False)
            df = df.reset_index(drop=True)
            
            self.logger.info(f"[OK] Screened {len(drug_pairs)} combinations")
            self.logger.info(f"  Top synergy: {df['predicted_synergy'].max():.4f}")
            self.logger.info(f"  Min synergy: {df['predicted_synergy'].min():.4f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"[X] Virtual screening failed: {e}", exc_info=True)
            raise VirtualScreeningError(f"Screening failed: {e}") from e
    
    def _create_screening_sample(self,
                                 expression: np.ndarray,
                                 drug1: Drug,
                                 drug2: Drug,
                                 cell_line_name: str,
                                 dose: float) -> Dict[str, torch.Tensor]:
        """Create a sample for screening prediction."""
        # Get drug encodings
        drug1_enc = self.dataset.drug_encodings.get(drug1.name, 
                                                     self.dataset.null_drug_encoding)
        drug2_enc = self.dataset.drug_encodings.get(drug2.name,
                                                     self.dataset.null_drug_encoding)
        
        # Get cell line index
        cell_line_idx = self.dataset.cell_line_to_idx.get(cell_line_name, 0)
        
        return {
            'expression': torch.tensor(expression, dtype=torch.float32),
            'synergy_label': torch.tensor(0.0, dtype=torch.float32),
            'cell_line_idx': torch.tensor(cell_line_idx, dtype=torch.long),
            'drug1_mechanism': torch.tensor(drug1_enc['mechanism'], dtype=torch.float32),
            'drug1_pathways': torch.tensor(drug1_enc['pathways'], dtype=torch.float32),
            'drug1_genes': torch.tensor(drug1_enc['genes'], dtype=torch.float32),
            'drug1_max_effect': torch.tensor(drug1_enc['max_effect'], dtype=torch.float32),
            'drug1_selectivity': torch.tensor(drug1_enc['selectivity'], dtype=torch.float32),
            'drug1_is_immuno': torch.tensor(drug1_enc['is_immunotherapy'], dtype=torch.float32),
            'drug2_mechanism': torch.tensor(drug2_enc['mechanism'], dtype=torch.float32),
            'drug2_pathways': torch.tensor(drug2_enc['pathways'], dtype=torch.float32),
            'drug2_genes': torch.tensor(drug2_enc['genes'], dtype=torch.float32),
            'drug2_max_effect': torch.tensor(drug2_enc['max_effect'], dtype=torch.float32),
            'drug2_selectivity': torch.tensor(drug2_enc['selectivity'], dtype=torch.float32),
            'drug2_is_immuno': torch.tensor(drug2_enc['is_immunotherapy'], dtype=torch.float32),
            'dose1': torch.tensor(dose, dtype=torch.float32),
            'dose2': torch.tensor(dose, dtype=torch.float32),
            'is_control': torch.tensor(0.0, dtype=torch.float32),
            'is_combination': torch.tensor(1.0, dtype=torch.float32)
        }
    
    def _collate_screening_batch(self, 
                                 samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate samples into a batch."""
        batch = {}
        for key in samples[0].keys():
            if isinstance(samples[0][key], torch.Tensor):
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                batch[key] = [s[key] for s in samples]
        return batch
    
    def screen_specific_drug(self,
                            anchor_drug_name: str,
                            cell_line_name: str,
                            dose: float = 0.8) -> pd.DataFrame:
        """
        Screen all partners for a specific anchor drug.
        
        Args:
            anchor_drug_name: Name of anchor drug
            cell_line_name: Cell line to screen
            dose: Dose level
            
        Returns:
            DataFrame with predictions sorted by synergy
        """
        try:
            anchor_drug = self.drug_db.get_drug(anchor_drug_name)
            if anchor_drug is None:
                raise ValidationError(f"Unknown drug: {anchor_drug_name}")
            
            cell_line = self.cell_line_db.get_cell_line(cell_line_name)
            if cell_line is None:
                raise ValidationError(f"Unknown cell line: {cell_line_name}")
            
            self.logger.info(f"Screening partners for {anchor_drug_name} in {cell_line_name}...")
            
            # Generate baseline expression
            baseline = self.expression_generator.generate_baseline_expression(
                cell_line,
                n_cells=self.config.n_cells_per_sample
            )
            expr_mean = baseline.mean(axis=0)
            
            # Get all other drugs
            all_drugs = self.drug_db.get_all_drugs()
            partner_drugs = [d for d in all_drugs if d.name != anchor_drug_name]
            
            results = []
            
            with torch.no_grad():
                batch_samples = []
                for partner_drug in partner_drugs:
                    sample = self._create_screening_sample(
                        expr_mean, anchor_drug, partner_drug, cell_line_name, dose
                    )
                    batch_samples.append(sample)
                
                # Process all at once (or batch if large)
                batch = self._collate_screening_batch(batch_samples)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                outputs = self.model(batch)
                predictions = outputs['synergy_pred'].cpu().numpy()
                
                for i, partner_drug in enumerate(partner_drugs):
                    results.append({
                        'anchor_drug': anchor_drug_name,
                        'partner_drug': partner_drug.name,
                        'partner_mechanism': partner_drug.mechanism.name,
                        'cell_line': cell_line_name,
                        'dose': dose,
                        'predicted_synergy': predictions[i]
                    })
            
            df = pd.DataFrame(results)
            df = df.sort_values('predicted_synergy', ascending=False)
            df = df.reset_index(drop=True)
            
            self.logger.info(f"[OK] Screened {len(partner_drugs)} partners for {anchor_drug_name}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"[X] Specific drug screening failed: {e}")
            raise VirtualScreeningError(f"Screening failed: {e}") from e
    
    def get_top_synergies(self,
                         results: pd.DataFrame,
                         n_top: int = 50,
                         min_synergy: float = None) -> pd.DataFrame:
        """
        Get top synergistic combinations.
        
        Args:
            results: Screening results DataFrame
            n_top: Number of top results to return
            min_synergy: Minimum synergy threshold
            
        Returns:
            DataFrame with top synergies
        """
        df = results.copy()
        
        # Filter by minimum synergy if specified
        if min_synergy is not None:
            df = df[df['predicted_synergy'] >= min_synergy]
        
        # Sort and get top N
        df = df.sort_values('predicted_synergy', ascending=False)
        df = df.head(n_top)
        df = df.reset_index(drop=True)
        
        # Log summary
        self.logger.info(f"\nTop {len(df)} synergies:")
        self.logger.info(f"  Range: [{df['predicted_synergy'].min():.4f}, "
                        f"{df['predicted_synergy'].max():.4f}]")
        
        return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class CancerSynergyPipeline:
    """
    Main pipeline orchestrator for cancer drug synergy prediction.
    
    Coordinates all components from data generation through virtual screening.
    
    Example:
        >>> pipeline = CancerSynergyPipeline()
        >>> results = pipeline.run()
    """
    
    def __init__(self, config: ProjectConfig = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Project configuration (creates default if None)
        """
        # Create or validate config
        self.config = config or ProjectConfig()
        self.config.validate()
        
        # Set up logging
        self.logger = setup_logger("CancerSynergyPipeline", self.config)
        
        # Set random seeds
        self._set_random_seeds()
        
        # Log initialization
        self._log_initialization()
        
        # Initialize databases
        self.logger.info("Initializing databases...")
        self.pathway_db = PathwayDatabase(self.logger)
        self.drug_db = DrugDatabase(self.pathway_db, self.logger)
        self.cell_line_db = CellLineDatabase(self.logger)
        
        # Initialize generators
        self.expression_generator = GeneExpressionGenerator(
            self.config, self.pathway_db, self.drug_db, 
            self.cell_line_db, self.logger
        )
        self.synergy_generator = SynergyDataGenerator(
            self.config, self.expression_generator, self.drug_db,
            self.cell_line_db, self.logger
        )
        
        self.logger.info("[OK] Pipeline initialization complete")
    
    def _set_random_seeds(self) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
        self.logger.debug(f"Random seeds set to {self.config.random_seed}")
    
    def _log_initialization(self) -> None:
        """Log initialization information."""
        self.logger.info("=" * 80)
        self.logger.info("CANCER SYNERGY VIRTUAL SCREENING PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Version: {self.config.version}")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Random seed: {self.config.random_seed}")
        self.logger.info("")
        self.logger.info("Configuration:")
        self.logger.info(f"  - Genes: {self.config.n_genes}")
        self.logger.info(f"  - Drugs: {self.config.n_drugs}")
        self.logger.info(f"  - Cell lines: {self.config.n_cell_lines}")
        self.logger.info(f"  - Pathways: {self.config.n_pathways}")
        self.logger.info(f"  - Batch size: {self.config.batch_size}")
        self.logger.info(f"  - Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  - Epochs: {self.config.n_epochs}")
        self.logger.info("")
    
    def generate_data(self) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """Generate training data."""
        return self.synergy_generator.generate_training_data()
    
    def create_dataset(self,
                      expressions: np.ndarray,
                      metadata: List[Dict],
                      synergy_labels: np.ndarray) -> SingleCellDatasetTorch:
        """Create PyTorch dataset."""
        return SingleCellDatasetTorch(
            expressions, metadata, synergy_labels,
            self.drug_db, self.logger
        )
    
    def create_data_loaders(self, 
                           dataset: SingleCellDatasetTorch) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test data loaders."""
        # Calculate split sizes
        n_total = len(dataset)
        n_train = int(n_total * self.config.train_fraction)
        n_val = int(n_total * self.config.val_fraction)
        n_test = n_total - n_train - n_val
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def build_model(self) -> CancerSynergyFoundationModel:
        """Build the foundation model."""
        return CancerSynergyFoundationModel(self.config, self.logger)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Returns:
            Dictionary with model, history, and screening results
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING CANCER SYNERGY PIPELINE")
            self.logger.info("=" * 80)
            
            # 1. Generate synthetic data
            self.logger.info("\n[1/6] Generating synthetic training data...")
            expressions, metadata, synergy_labels = self.generate_data()
            self.logger.info(f"[OK] Generated {len(metadata)} samples")
            
            # 2. Create dataset
            self.logger.info("\n[2/6] Creating PyTorch dataset...")
            dataset = self.create_dataset(expressions, metadata, synergy_labels)
            self.logger.info(f"[OK] Dataset created with {len(dataset)} samples")
            
            # 3. Split and create loaders
            self.logger.info("\n[3/6] Splitting data and creating loaders...")
            train_loader, val_loader, test_loader = self.create_data_loaders(dataset)
            self.logger.info(f"[OK] Train: {len(train_loader.dataset)}, "
                        f"Val: {len(val_loader.dataset)}, "
                        f"Test: {len(test_loader.dataset)}")
            
            # 4. Build model
            self.logger.info("\n[4/6] Building foundation model...")
            model = self.build_model()
            n_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"[OK] Model built with {n_params:,} parameters")
            
            # 5. Train model
            self.logger.info("\n[5/6] Training model...")
            trainer = ModelTrainer(model, self.config, self.logger)
            history = trainer.train(train_loader, val_loader)
            best_val_loss = min(history['val_loss'])
            self.logger.info(f"[OK] Training complete")
            self.logger.info(f"  Best validation loss: {best_val_loss:.4f}")
            
            # 6. Virtual screening
            self.logger.info("\n[6/6] Running virtual screening...")
            screener = VirtualScreener(
                model, self.config, self.drug_db, self.cell_line_db,
                self.expression_generator, dataset, self.logger
            )  # CLOSE THE PARENTHESES HERE
            
            all_results = {}
            all_top_hits = {}
            
            # Screen all combinations for each cell line
            for cell_line in self.cell_line_db.get_all_cell_lines():
                self.logger.info(f"\nScreening {cell_line.name}...")
                results = screener.screen_all_combinations(cell_line.name)
                
                # Get top synergies
                top_hits = screener.get_top_synergies(results, n_top=50)
                
                # Save results
                output_file = self.config.results_dir / f"{cell_line.name}_screening_results.csv"
                results.to_csv(output_file, index=False)
                self.logger.info(f"[OK] Results saved to {output_file}")
                
                # Display top 5
                self.logger.info(f"\nTop 5 predicted synergies for {cell_line.name}:")
                for i, row in top_hits.head(5).iterrows():
                    self.logger.info(
                        f"  {i+1}. {row['drug1']} + {row['drug2']}: "
                        f"{row['predicted_synergy']:.3f}"
                    )
                
                all_results[cell_line.name] = results
                all_top_hits[cell_line.name] = top_hits
            
            # Save configuration
            config_file = self.config.results_dir / "config.json"
            self.config.save(config_file)
            self.logger.info(f"[OK] Configuration saved to {config_file}")
            
            # 7. Generate publication figures (MOVED HERE)
            self.logger.info("\n[7/7] Generating publication figures...")
            try:
                from visualization_publication import create_publication_figures
                create_publication_figures(
                    results_dir=self.config.results_dir,
                    figures_dir=self.config.figures_dir,
                    training_history=history,
                    logger=self.logger
                )
            except Exception as e:
                self.logger.warning(f"Figure generation failed: {e}")
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return {
                'model': model,
                'history': history,
                'screening_results': all_results,
                'top_synergies': all_top_hits,
                'dataset': dataset
            }
            
        except KeyboardInterrupt:
            self.logger.warning("\n[!] Pipeline interrupted by user")
            raise
            
        except Exception as e:
            self.logger.error(f"\n[X] Pipeline failed: {e}", exc_info=True)
            raise


# =============================================================================
# UNIT TESTS
# =============================================================================

def run_unit_tests(logger: logging.Logger) -> bool:
    """
    Run basic unit tests for critical functions.
    
    Args:
        logger: Logger instance
        
    Returns:
        True if all tests pass
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING UNIT TESTS")
    logger.info("=" * 80)
    
    all_passed = True
    
    # Test 1: PathwayDatabase initialization
    try:
        logger.info("\n[Test 1] PathwayDatabase initialization...")
        pathway_db = PathwayDatabase(logger)
        assert len(pathway_db.pathway_genes) == 15, "Should have 15 pathways"
        assert len(pathway_db.all_genes) > 200, "Should have >200 unique genes"
        logger.info("[OK] Test 1 passed")
    except Exception as e:
        logger.error(f"[X] Test 1 failed: {e}")
        all_passed = False
    
    # Test 2: DrugDatabase initialization
    try:
        logger.info("\n[Test 2] DrugDatabase initialization...")
        drug_db = DrugDatabase(pathway_db, logger)
        assert len(drug_db.drugs) == 58, "Should have 58 drugs"
        assert drug_db.get_drug("silmitasertib") is not None, "Should have silmitasertib"
        assert drug_db.get_drug("interferon_alpha") is not None, "Should have interferon_alpha"
        logger.info("[OK] Test 2 passed")
    except Exception as e:
        logger.error(f"[X] Test 2 failed: {e}")
        all_passed = False
    
    # Test 3: CellLineDatabase initialization
    try:
        logger.info("\n[Test 3] CellLineDatabase initialization...")
        cell_db = CellLineDatabase(logger)
        assert len(cell_db.cell_lines) == 3, "Should have 3 cell lines"
        a549 = cell_db.get_cell_line("A549")
        assert a549 is not None, "Should have A549"
        assert len(a549.oncogenes) > 0, "A549 should have oncogenes"
        logger.info("[OK] Test 3 passed")
    except Exception as e:
        logger.error(f"[X] Test 3 failed: {e}")
        all_passed = False
    
    # Test 4: Expression generation
    try:
        logger.info("\n[Test 4] Expression generation...")
        config = ProjectConfig(n_genes=500, n_cells_per_sample=50)
        expr_gen = GeneExpressionGenerator(config, pathway_db, drug_db, cell_db, logger)
        cell_line = cell_db.get_cell_line("A549")
        expr = expr_gen.generate_baseline_expression(cell_line, n_cells=50)
        assert expr.shape == (50, 500), f"Wrong shape: {expr.shape}"
        assert not np.isnan(expr).any(), "Contains NaN"
        assert (expr >= 0).all(), "Contains negative values"
        logger.info("[OK] Test 4 passed")
    except Exception as e:
        logger.error(f"[X] Test 4 failed: {e}")
        all_passed = False
    
    # Test 5: Drug effect application
    try:
        logger.info("\n[Test 5] Drug effect application...")
        drug = drug_db.get_drug("trametinib")
        treated = expr_gen.apply_drug_effect(expr, drug, dose=0.8)
        assert treated.shape == expr.shape, "Shape changed"
        assert not np.isnan(treated).any(), "Contains NaN"
        # MEK inhibitor should reduce overall expression
        assert treated.mean() < expr.mean(), "Inhibitor didn't reduce expression"
        logger.info("[OK] Test 5 passed")
    except Exception as e:
        logger.error(f"[X] Test 5 failed: {e}")
        all_passed = False
    
    # Test 6: Model forward pass
    try:
        logger.info("\n[Test 6] Model forward pass...")
        config = ProjectConfig(n_genes=500, n_cells_per_sample=50)
        model = CancerSynergyFoundationModel(config, logger)
        
        # Create dummy batch
        batch_size = 4
        batch = {
            'expression': torch.randn(batch_size, 500),
            'cell_line_idx': torch.zeros(batch_size, dtype=torch.long),
            'drug1_mechanism': torch.zeros(batch_size, len(DrugMechanism)),
            'drug1_pathways': torch.zeros(batch_size, len(CancerPathway)),
            'drug1_genes': torch.zeros(batch_size, 500),
            'drug1_max_effect': torch.zeros(batch_size, 1),
            'drug1_selectivity': torch.zeros(batch_size, 1),
            'drug1_is_immuno': torch.zeros(batch_size, 1),
            'drug2_mechanism': torch.zeros(batch_size, len(DrugMechanism)),
            'drug2_pathways': torch.zeros(batch_size, len(CancerPathway)),
            'drug2_genes': torch.zeros(batch_size, 500),
            'drug2_max_effect': torch.zeros(batch_size, 1),
            'drug2_selectivity': torch.zeros(batch_size, 1),
            'drug2_is_immuno': torch.zeros(batch_size, 1),
            'dose1': torch.ones(batch_size),
            'dose2': torch.ones(batch_size),
        }
        
        outputs = model(batch)
        assert 'synergy_pred' in outputs, "Missing synergy_pred"
        assert outputs['synergy_pred'].shape == (batch_size,), "Wrong synergy shape"
        assert 'expr_recon' in outputs, "Missing expr_recon"
        assert outputs['expr_recon'].shape == (batch_size, 500), "Wrong recon shape"
        logger.info("[OK] Test 6 passed")
    except Exception as e:
        logger.error(f"[X] Test 6 failed: {e}")
        all_passed = False
    
    # Summary
    logger.info("\n" + "-" * 80)
    if all_passed:
        logger.info("[OK] ALL TESTS PASSED")
    else:
        logger.error("[X] SOME TESTS FAILED")
    logger.info("-" * 80 + "\n")
    
    return all_passed


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Cancer Drug Synergy Pipeline')
    parser.add_argument('--n-genes', type=int, default=2000, help='Number of genes')
    parser.add_argument('--n-cells', type=int, default=100, help='Cells per sample')
    parser.add_argument('--n-epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
                       help='Device to use for training')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    try:
        # Create configuration from arguments
        config = ProjectConfig(
            n_genes=2000,
            n_cells_per_sample=100,
            n_training_samples=5000,
            n_control_samples=500,
            n_single_drug_samples=1500,
            n_combination_samples=3000,
            batch_size=args.batch_size,
            learning_rate=1e-4,
            n_epochs=args.n_epochs,
            device=args.device,
            random_seed=args.random_seed,
            early_stopping_patience=10,
            num_workers=8,          # ADD THIS - use 8 CPU cores for data loading
            pin_memory=False        # ADD THIS - False for CPU training
        )
        
        
        # Validate configuration
        config.validate()
        
        # Set up logger for tests
        test_logger = setup_logger("UnitTests", config)
        
        # Run unit tests first
        tests_passed = run_unit_tests(test_logger)
        if not tests_passed:
            print("[!] Unit tests failed, but continuing with pipeline...")
        
        # Run the full pipeline
        pipeline = CancerSynergyPipeline(config)
        results = pipeline.run()
        
        return results
        
    except KeyboardInterrupt:
        print("\n[!] Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[X] Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
