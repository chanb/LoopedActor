"""Shared --arch / think-budget arguments of the */eval_checkpoint.py tools, and the fprm_mem model they describe."""
from models.fprm_mem_halt import FPRMMemHaltActorValue


def add_model_args(ap):
    ap.add_argument('--arch', default='fprm_mem', choices=['fprm_mem', 'multi_block', 'single_block'])
    ap.add_argument('--act_iters', default='16:u',
                    help="fprm_mem: the checkpoint's --act_iters=C:u|v (C may be raised at test time)")
    ap.add_argument('--value_iters', type=int, default=16, help="fprm_mem: the checkpoint's --value_iters")
    ap.add_argument('--min_think_iters', type=int, default=2, help="fprm_mem: the checkpoint's --min_think_iters")
    ap.add_argument('--halt_criterion', default='sigmoid', choices=['sigmoid', 'kl', 'l2'],
                    help="fprm_mem, C:v: the checkpoint's --halt_criterion")
    ap.add_argument('--halt_threshold', type=float, default=1e-3, help="fprm_mem, C:v, kl / l2: the threshold")
    ap.add_argument('--halt_kl_reverse', action='store_true', help='fprm_mem: the checkpoint was trained with --halt_kl_reverse')
    ap.add_argument('--policy_param', default='last', choices=['last', 'sum'], help="fprm_mem: the checkpoint's --policy_param")
    ap.add_argument('--readout_token', default='thought', choices=['thought', 'sep'], help="fprm_mem: the checkpoint's --readout_token")
    ap.add_argument('--stable_softmax', action='store_true', help='fprm_mem: the checkpoint was trained with --stable_softmax')
    ap.add_argument('--thought_norm', action='store_true', help='fprm_mem: the checkpoint was trained with --cot_thought_norm')


def think_cap(args):
    """The cap C of --act_iters (None for the non-looped baselines)."""
    return int(args.act_iters.split(':')[0]) if args.arch == 'fprm_mem' else None


def cap_tag(args):
    """Output-name suffix for a test-time cap other than 16."""
    C = think_cap(args)
    return f'_cap{C}' if C not in (None, 16) else ''


def build_mem(args, **kw):
    cap, mode = args.act_iters.split(':')
    assert mode in ('u', 'v'), args.act_iters
    return FPRMMemHaltActorValue(
        halting=mode == 'v', max_think_iters=int(cap), min_think_iters=args.min_think_iters,
        halt_criterion=args.halt_criterion, halt_threshold=args.halt_threshold, halt_kl_reverse=args.halt_kl_reverse,
        stable_softmax=args.stable_softmax, policy_param=args.policy_param, value_iters=args.value_iters,
        readout_token=args.readout_token, thought_norm=args.thought_norm, **kw)
