import math
from collections import defaultdict

import torch

from src.text_encoder.ctc_text_encoder import CTCTextEncoder


class CTCTextBeamEncoder(CTCTextEncoder):
    def __init__(self, alphabet=None, n_beam=3, **kwargs):
        super().__init__(alphabet, **kwargs)

        self.n_beam = n_beam
        self.emp_id = self.char2ind[self.EMPTY_TOK]
        self.use_beam = True

    def ctc_beam_decode(self, inds):
        preds = defaultdict(list)
        preds[(self.emp_id,)] = (1.0, 0.0)  # (blank, non_blank)

        for t in inds:
            new_pref = defaultdict(list)

            top_symb_ind = torch.topk(t, k=self.n_beam).indices

            for i in top_symb_ind:
                new_pref = self.new_beam(preds, i, new_pref, t[i])

            preds = self.filter_key(new_pref, self.n_beam)

        inds = self.filter_key(preds, 1)

        return self.decode(list(inds.keys())[0])

    def filter_key(self, pref, n):
        sorted_prefixes = sorted(pref.keys(), key=lambda x: pref[x], reverse=True)
        top_k_prefixes = sorted_prefixes[:n]

        return {x: pref[x] for x in top_k_prefixes}

    def logsumexp(self, vals):
        return math.log(sum(math.exp(v) for v in vals))

    def logaddexp(self, a, b):
        return math.log(math.exp(a) + math.exp(b))

    def sum_probs(self, pref):
        total_prob = defaultdict(list)
        for k in pref.keys():
            firsts, seconds = pref[k] if isinstance(pref[k], tuple) else zip(*pref[k])

            if isinstance(pref[k], tuple):
                sum_blank = firsts
                sum_non_blank = seconds

            else:
                sum_blank = self.logsumexp(firsts)
                sum_non_blank = self.logsumexp(seconds)

            pref[k] = (sum_blank, sum_non_blank)
            total_prob[k] = self.logaddexp(sum_blank, sum_non_blank)

        return total_prob

    def merge_prefix(self, old_p, probs, new_p, new_prob, new_pref):
        last_c = old_p[-1]

        if last_c == self.emp_id:
            prob = (probs[0], probs[0] + new_prob)  # blanc * prob_t

        elif last_c == new_p:
            prob = (probs[0], probs[1] + new_prob)  # non_blanc * prob_t

        elif new_p == self.emp_id:
            prob = (probs[1] + new_prob, probs[1])

        else:
            prob = (probs[0], probs[1] + new_prob)

        if last_c != new_p:
            old_p = old_p + (new_p,)

        if old_p not in new_pref:
            new_pref[old_p] = prob

        else:
            pb0, pnb0 = new_pref[old_p]
            new_pref[old_p] = (
                self.logaddexp(pb0, prob[0]),
                self.logaddexp(pnb0, prob[1]),
            )

        return new_pref

    def new_beam(self, prefix, ind, new_pref, new_prob):
        for k in prefix.keys():  # letter and prefix
            new_pref = self.merge_prefix(
                old_p=k,
                probs=prefix[k],
                new_p=ind,
                new_prob=new_prob,
                new_pref=new_pref,
            )

        return new_pref
