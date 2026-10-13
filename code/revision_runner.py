import json
import os

import torch
import run_experiments as core
import large_data as large
from learnable_fusion import LearnableFusionLoss, finetune_epoch
from training_utils import (structural_components, evaluate_final, capture_rng_state,
                            restore_rng_state, cpu_snapshot, load_checkpoint)

BIG = ('Digit-Product', 'Hdigit')


def ensure_manifest(out, manifest):
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise RuntimeError('Configuration differs; use a new output directory')
    elif any(out.iterdir()):
        raise RuntimeError('Output directory has no configuration')
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def prepare_training(name, seed, cfg, args, run_dir, checkpoint):
    cfg = dict(cfg, seed=seed)
    device = torch.device('cuda')
    core.set_seed(seed)
    if name in BIG:
        data = large.LargeMultiviewData(name, args.dataset_dir, device, args.normalization)
    else:
        data = core.MultiviewData(name, device, path=args.dataset_dir)
    model, optimizer = large.make_network(data, cfg, device)
    loss = LearnableFusionLoss(cfg['temperature_f']).to(device)
    optimizer.add_param_group({'params': list(loss.parameters())})
    if checkpoint.is_file():
        initial = load_checkpoint(checkpoint)
        if initial['config'] != cfg:
            raise RuntimeError('Checkpoint configuration differs')
        model.load_state_dict(initial['model'])
        optimizer.load_state_dict(initial['optimizer'])
        loss.load_state_dict(initial['fusion_loss'])
        restore_rng_state(initial['rng_state'])
    else:
        core.pre_train(model, data, cfg['batch_size'], cfg['mse_epochs'], optimizer)
        initial = cpu_snapshot(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    fusion_loss=loss.state_dict(), config=cfg,
                                    rng_state=capture_rng_state()))
        pending = checkpoint.with_suffix('.pt.tmp')
        torch.save(initial, pending)
        os.replace(pending, checkpoint)
    if name in BIG:
        graphs, _, _ = large.static_component_labels(name, data, cfg, args, run_dir)
    else:
        graphs = structural_components(data, cfg['_structural_neighbors'], cfg['delta'])
    return dict(data=data, model=model, optimizer=optimizer, loss=loss,
                graphs=graphs, initial=initial, rng_state=capture_rng_state())


def train(name, seed, cfg, args, prepared):
    model, optimizer, loss = (prepared[key] for key in ('model', 'optimizer', 'loss'))
    initial = prepared['initial']
    model.load_state_dict(initial['model'])
    optimizer.load_state_dict(initial['optimizer'])
    loss.load_state_dict(initial['fusion_loss'])
    restore_rng_state(prepared['rng_state'])
    data = prepared['data']
    for epoch in range(1, cfg['con_epochs'] + 1):
        loss.begin_epoch(len(data.data_views))
        total, _ = finetune_epoch(model, data, loss, cfg, prepared['graphs'], optimizer, args, name in BIG)
        print(f'{name} seed={seed} epoch={epoch}/{cfg["con_epochs"]} loss={total:.7f}', flush=True)
    metrics, _ = evaluate_final(model, data)
    return dict(zip(('acc1', 'nmi1', 'pur1', 'ari1', 'f_score1'), map(float, metrics)))
