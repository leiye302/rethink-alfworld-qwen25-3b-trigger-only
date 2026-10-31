"""Only Bad correction supervision changes, not the marker's local weight."""
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from route_bce.core import encode_row,losses
from test_route_bce import example,tokenizer

ROOT=Path(__file__).resolve().parents[3]


def test_bad_actual_input_stops_at_marker_no_correction_or_eos(tokenizer):
    full=encode_row(example('bad'),tokenizer,supervision_mode='full')
    cut=encode_row(example('bad'),tokenizer,supervision_mode='trigger_only')
    s=cut['suffix_start']
    assert cut['input_ids'][:s]==full['input_ids'][:s]
    assert cut['input_ids'][s:]==[tokenizer.convert_tokens_to_ids('<RETHINK>')]
    assert tokenizer.eos_token_id not in cut['input_ids'][s:]
    assert cut['suffix_tokens']==1
    assert cut['ce_denominator']==full['suffix_tokens']>1
    assert cut['joint_tokenization_matches']


def test_good_original_target_and_loss_unchanged(tokenizer):
    full=encode_row(example('good'),tokenizer,supervision_mode='full')
    cut=encode_row(example('good'),tokenizer,supervision_mode='trigger_only')
    assert full['input_ids']==cut['input_ids']
    assert full['ce_denominator']==cut['ce_denominator']==full['suffix_tokens']
    logits=torch.randn(len(cut['input_ids']),len(tokenizer))
    ids=torch.tensor(cut['input_ids'])
    a=losses(logits,ids,cut['suffix_start'],0,tokenizer.convert_tokens_to_ids('<RETHINK>'))
    b=losses(logits,ids,cut['suffix_start'],0,tokenizer.convert_tokens_to_ids('<RETHINK>'),
             ce_denominator=cut['ce_denominator'])
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


def test_marker_boundary_gradient_equals_full_marker_term_not_single_token_mean():
    torch.manual_seed(6)
    ids=torch.tensor([1,2,3,8,4,9,2])
    z=torch.randn(7,16,requires_grad=True)
    full=losses(z,ids,3,1,8)
    full['loss'].backward();wanted=z.grad[2].clone()
    short=z.detach()[:4].clone().requires_grad_(True)
    cut=losses(short,ids[:4],3,1,8,ce_denominator=4)
    expected_ce=F.cross_entropy(short[2:3].float(),ids[3:4],reduction='sum')/4
    torch.testing.assert_close(cut['suffix_ce'],expected_ce)
    torch.testing.assert_close(cut['route_bce'],full['route_bce'])
    cut['loss'].backward()
    torch.testing.assert_close(short.grad[2],wanted)
    assert short.grad[:2].abs().sum()==0
    assert short.grad[3:].abs().sum()==0  # no target after marker
    wrong=losses(short.detach(),ids[:4],3,1,8)
    assert wrong['suffix_ce']>cut['suffix_ce']


def test_no_invalid_normalization_or_mode(tokenizer):
    with pytest.raises(ValueError,match='mode'):
        encode_row(example('bad'),tokenizer,supervision_mode='other')
    with pytest.raises(ValueError,match='denominator'):
        losses(torch.randn(5,12),torch.tensor([1,2,8,4,2]),2,1,8,ce_denominator=1)


def test_real_causal_model_marker_does_not_read_future_correction():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(19)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=24,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,attention_dropout=0.))
    model.eval()
    ids=torch.tensor([1,2,3,8,4,9,2])
    z=model(ids[None,:],use_cache=False).logits[0]
    complete=losses(z,ids,3,1,8)
    boundary=F.cross_entropy(z[2:3].float(),ids[3:4])/4+.1*complete['route_bce']
    boundary.backward()
    wanted=[p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    short=model(ids[None,:4],use_cache=False).logits[0]
    torch.testing.assert_close(z[2],short[2],atol=1e-6,rtol=1e-6)
    losses(short,ids[:4],3,1,8,ce_denominator=4)['loss'].backward()
    for a,b in zip(wanted,model.parameters()):
        torch.testing.assert_close(a,b.grad,atol=1e-6,rtol=1e-5)


def test_preflight_filename_and_supervision_mode_are_bound():
    script=(ROOT/'scripts/train.sh').read_text()
    trainer=(ROOT/'src/sft/route_bce/train.py').read_text()
    assert 'reports/cpu_token_preflight.json' in script
    assert 'reports/cpu_token_preflight.json' in trainer
    assert 'preflight["supervision_mode"] == c["supervision_mode"]' in trainer
    assert json.loads((ROOT/'configs/sft.json').read_text())['supervision_mode']=='trigger_only'
