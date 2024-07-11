import numpy as np
import torch

from dataset_utils.alpaca_gpt4_dataset import AlpacaGPT4Dataset  # noqa: F401
from dataset_utils.aokvqa_dataset import AOKVQADataset  # noqa: F401
from dataset_utils.cc_sbu_align_dataset import CcSbuAlignDataset  # noqa: F401
from dataset_utils.clevr_dataset import CLEVRDataset  # noqa: F401
from dataset_utils.coco_caption_dataset import COCOCaptionDataset  # noqa: F401
from dataset_utils.dial_dataset import DialDataset  # noqa: F401
from dataset_utils.dolly_dataset import DollyDataset  # noqa: F401
from dataset_utils.gqa_dataset import GQADataset  # noqa: F401
from dataset_utils.llava_dataset import LlavaDataset  # noqa: F401
from dataset_utils.nlvr_dataset import NLVRv1Dataset, NLVRv2Dataset  # noqa: F401
from dataset_utils.ocr_vqa_dataset import OCRVQADataset  # noqa: F401
from dataset_utils.snli_ve_datasets import SNLIVEDataset  # noqa: F401
from dataset_utils.text_ocr_dataset import TextOCRDataset  # noqa: F401
from dataset_utils.vqa_dataset import ConcatDataset, VQADataset  # noqa: F401
from dataset_utils.coco_sic_dataset import COCOSICDataset
from dataset_utils.coco_ric_dataset import COCORICDataset
from dataset_utils.meme_cap_dataset import MEMECAPDataset
from dataset_utils.vsr_dataset import VSRDataset
from dataset_utils.iconqa_dataset import ICONQADataset
from dataset_utils.vicuna_dataset import ShareGPTDataset
from dataset_utils.baize_dataset import QuoraDataset
from dataset_utils.instruct_dataset import MultimodalDataset
from dataset_utils.shikra_dataset import ShikraDataset
from dataset_utils.svit_dataset import SVITDataset
from dataset_utils.svit_dial_dataset import SVITDialDataset
from dataset_utils.svit_bbox_dataset import SVITBboxDataset
from dataset_utils.lrv_instruction_dataset import LRVDataset
from dataset_utils.rec_dataset import RECDataset
from dataset_utils.reg_dataset import REGDataset
from dataset_utils.gc_dataset import GCDataset
from dataset_utils.textvqa_dataset import TextVQADataset
from dataset_utils.vizwiz_vqa_dataset import VizWizVQADataset
from dataset_utils.bddx_dataset import BDDXDataset


def build_dataset(dataset_config, **kwargs):
    if isinstance(dataset_config, list):
        datasets = [build_dataset(cfg, **kwargs) for cfg in dataset_config]
        return ConcatDataset(datasets)
    dataset_type = dataset_config.pop("type")
    sample = dataset_config.pop("sample", -1)
    if dataset_type == "llava":
        dataset = LlavaDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "shikra":
        dataset = ShikraDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "rec":
        dataset = RECDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "reg":
        dataset = REGDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "gc":
        dataset = GCDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "lrv":
        dataset = LRVDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "svit":
        dataset = SVITDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "svit_dial":
        dataset = SVITDialDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "svit_bbox":
        dataset = SVITBboxDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "textvqa":
        dataset = TextVQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "vizwiz":
        dataset = VizWizVQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "vqa":
        dataset = VQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "minigpt4":
        dataset = CcSbuAlignDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "llava_dial":
        dataset = DialDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "coco_dial":
        dataset = DialDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "aokvqa":
        dataset = AOKVQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "okvqa":
        dataset = VQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "text_ocr":
        dataset = TextOCRDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "ocr_vqa":
        dataset = OCRVQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "meme_cap":
        dataset = MEMECAPDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "coco_caption":
        dataset = COCOCaptionDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "gqa":
        dataset = GQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "vsr":
        dataset = VSRDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "icon_qa":
        dataset = ICONQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "clevr":
        dataset = CLEVRDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "nlvrv1":
        dataset = NLVRv1Dataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "nlvrv2":
        dataset = NLVRv2Dataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "snlive":
        dataset = SNLIVEDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "sic":
        dataset = COCOSICDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "ric":
        dataset = COCORICDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "dolly":
        dataset = DollyDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "alpaca_gpt4":
        dataset = AlpacaGPT4Dataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "vicuna":
        dataset = ShareGPTDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "baize":
        dataset = QuoraDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "instruct":
        dataset = MultimodalDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "bddx":
        dataset = BDDXDataset(
            **dataset_config,
            **kwargs,
        )
    else:
        raise NotImplementedError

    if sample > 0:
        random_indices = np.random.choice(len(dataset), min(sample, len(dataset)), replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        subsample_dataset.collater = dataset.collater
        return subsample_dataset
    else:
        return dataset
