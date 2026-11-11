"""Portable entry point for the URDF-A simulation release."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check(profile, render=False):
    import numpy as np
    from urdf_a_model import NativeHand

    inventory = json.loads((ROOT / 'docs/model_sha256.json').read_text(encoding='utf8'))
    for name, expected in inventory.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Model checksum mismatch: ' + name)
    g = NativeHand(ROOT / 'assets/urdf_a')
    result = dict(model_files=len(inventory), native_hinges=g.model.njnt,
                  independent_coordinates=g.C.shape[1], geometry_id=g.geometry_id,
                  neutral_clearance_mm=g.clearance(), profile_loaded=False)
    if profile.is_file():
        from urdf_a_follow import Follower
        follower = Follower(g, profile)
        record = json.loads(profile.read_text(encoding='utf8'))
        follower.command(record['neutral'])
        follower.advance(.025)
        np.testing.assert_allclose(g.data.qpos, g.C @ follower.u)
        result.update(profile_loaded=True, profile_sha256=follower.profile_sha256,
                      middle_unavailable=follower.middle_unavailable)
    if render:
        from types import SimpleNamespace
        from urdf_a_live import View
        view = View(g)
        try:
            preview = ROOT / 'results/model_preview.png'
            view.draw(SimpleNamespace(chains=None)).save(preview)
            result['preview'] = str(preview)
        finally:
            view.renderer.close()
    return result


def main():
    parser = argparse.ArgumentParser(description='F1 URDF-A simulation / G7 official UDP follow')
    parser.add_argument('--profile', type=Path,
                        help='Local URDF-A wearer profile; never uploaded by default. Defaults to '
                             'calibration/local/profile.json (right) or profile_left.json (left)')
    parser.add_argument('--hand', choices=['right', 'left'], default='right',
                        help='Glove hand; a left glove is mirrored onto the right F1 model')
    parser.add_argument('--port', type=int, default=15001, help='HumanHand UDP port on localhost')
    parser.add_argument('--demo', action='store_true', help='Offline model demonstration; no glove/profile needed')
    parser.add_argument('--seconds', type=float, help='Close the offline demo after this duration')
    parser.add_argument('--check', action='store_true', help='Validate model and optional local profile, without UDP')
    parser.add_argument('--render', action='store_true', help='Also render a PNG with --check')
    args = parser.parse_args()
    if args.render and not args.check:
        parser.error('--render requires --check')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.profile is None:
        args.profile = ROOT / 'calibration/local' / ('profile_left.json' if args.hand == 'left' else 'profile.json')
    (ROOT / 'results').mkdir(exist_ok=True)
    if args.check:
        print(json.dumps(check(args.profile, args.render), ensure_ascii=False, indent=2))
    elif args.demo:
        from demo import launch
        launch(args.seconds)
    else:
        if not args.profile.is_file():
            parser.error(f'Missing local profile {args.profile}. Copy the existing URDF-A profile there, '
                         'or use --profile PATH. Use --demo to inspect the model without a glove.')
        from urdf_a_live import launch
        launch(args.profile, args.port, input_hand=args.hand)


if __name__ == '__main__':
    main()
