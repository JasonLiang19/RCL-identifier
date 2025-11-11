"""Data loading and encoding package."""

from .data_loader import (
    ProteinSequenceDataset,
    read_fasta,
    read_csv_with_annotations,
    parse_fasta_with_rcl_annotations,
    load_encoding_matrix,
    create_data_loaders
)

from .encoders import (
    SequenceEncoder,
    OneHotEncoder,
    BLOSUMEncoder,
    ESM2Encoder,
    ESM2_650M_Encoder,
    get_encoder,
    EncodedDataset
)

__all__ = [
    'ProteinSequenceDataset',
    'read_fasta',
    'read_csv_with_annotations',
    'parse_fasta_with_rcl_annotations',
    'load_encoding_matrix',
    'create_data_loaders',
    'SequenceEncoder',
    'OneHotEncoder',
    'BLOSUMEncoder',
    'ESM2Encoder',
    'ESM2_650M_Encoder',
    'get_encoder',
    'EncodedDataset'
]
