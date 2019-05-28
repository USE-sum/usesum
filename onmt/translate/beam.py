from __future__ import division
import torch
from onmt.translate import penalties
import en_core_web_sm
import re

class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set(),
                 promote_inputs = True):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens
        self.allowed_to_repeat = {'the', 'a', 'an', ''}
        self.nlp = en_core_web_sm.load(disable=["parser"])
        self.promote_inputs = promote_inputs

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def word_id_to_string(self, id, vocab, src_vocab):
        the_word = ""
        if id < len(vocab.itos):
            the_word = vocab.itos[id]
        elif src_vocab is not None and (id - len(vocab.itos)) < len(src_vocab.itos):
            the_word = src_vocab.itos[id - len(vocab.itos)]  # copy attention
        return the_word

    def word_in_input_dict(self, the_word, dict):
        if the_word in dict and dict[the_word] > 0:
            return True
        return False

    def advance(self, word_probs, attn_out, fields = None, src_vocab=None, debug=False):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        promote_inputs_in_beam = self.promote_inputs
        boost_sets = []
        boost_snip4 = [0.4, 0.7, 50]
        # boost_useG7 = [0.4, 0.9, 50]
        boost_sets = boost_snip4
        debug = debug
        vocab = None
        if fields is not None:
            vocab = fields["tgt"].vocab
        num_words = word_probs.size(1)
        used_words = {}
        used_lemmas = {}
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        unigrams = []
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            # Block ngram repeats
            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    if j not in used_words:
                        used_words[j] = []
                        used_lemmas[j] = []
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    unigrams.append({})
                    for i in range(le - 1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram +
                                [hyp[i].item()])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list
                        if set(gram) & self.exclusion_tokens:
                            continue
                        tok = hyp[i]
                        the_word = ""
                        if vocab is not None:
                            the_word = self.word_id_to_string(tok, vocab, src_vocab).lower()
                            if the_word=="":
                                continue
                            used_words[j].append(the_word)
                            lema = re.sub('[\W_]+', '', self.nlp(the_word)[0].lemma_) # self.lmtzr.lemmatize(the_word).lower())
                            used_lemmas[j].append(lema)
                            if (the_word in unigrams[j] and i-unigrams[j][the_word]<=self.block_ngram_repeat*2 \
                                or lema in unigrams[j] and i-unigrams[j][lema]<=self.block_ngram_repeat*2) \
                                    and the_word not in self.allowed_to_repeat:
                                fail = True
                            unigrams[j][the_word] = i
                            unigrams[j][lema] = i
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                        if fail:
                            # print(str(j)+" prevent from repeating "+the_word+"    "+str(beam_scores[j, tok.item()]) )
                            beam_scores[j, tok.item()] = -10e20
                            # print(" after "+str(beam_scores[j, tok.item()]))
        else:
            beam_scores = word_probs[0]

        # print(used_words)
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size*25, 0,
                                                            True, True)
        prev_k = best_scores_id / num_words

        created_best_scores  = []
        created_best_scores_id = []
        created_prev_k_id = []
        i = 0
        stride = 0
        for bi in best_scores_id:
            bi = bi - prev_k[i] * num_words
            the_word = self.word_id_to_string(bi, vocab, src_vocab)
            sent = the_word
            if len(used_words)>0:
                sent = " ".join(used_words[prev_k[stride].item()])+" "+the_word
            ner_iob=""
            doc = self.nlp(sent)
            for word in doc:
                lema = re.sub('[\W_]+', '',word.lemma_)
                pos = word.tag_
                ner_iob = word.ent_iob_
            if len(the_word)==0 or len(used_words)>0 and the_word.lower() in used_words[prev_k[stride].item()] \
                    and ((len(used_words[prev_k[stride].item()]) - used_words[prev_k[stride].item()].index(the_word.lower()))<=self.block_ngram_repeat*2
                    or (len(used_lemmas[prev_k[stride].item()]) - used_lemmas[prev_k[stride].item()].index(lema.lower()))<=self.block_ngram_repeat*2 )\
                    and the_word not in self.allowed_to_repeat:
                i += 1
                continue

            if debug: # ner_iob=="B" or ner_iob=="I"
                print(the_word + "  score = " + str(best_scores[i])+"   pos "+pos+"   len(self.prev_ks)= "+str(len(self.prev_ks))+" ner_iob "+ner_iob)
            if promote_inputs_in_beam and ( (ner_iob=="B" or ner_iob=="I" or the_word[0].isupper() and len(self.prev_ks) > 0  or the_word.isdigit()) and
                    (self.word_in_input_dict(the_word, src_vocab.stoi) or self.word_in_input_dict(the_word.lower(), src_vocab.stoi) or
                     self.word_in_input_dict(lema, src_vocab.stoi) )):
                best_scores[i] *= boost_sets[0]
                # elif pos.startswith("J"):
                #     best_scores[i] *= 0.7
                if debug:
                    print("      boost03 " + the_word + "  score = " + str(best_scores[i])+"   "+pos)
            elif promote_inputs_in_beam and (self.word_in_input_dict(the_word, src_vocab.stoi) or self.word_in_input_dict(the_word.lower(), src_vocab.stoi) or
                     self.word_in_input_dict(lema, src_vocab.stoi) ) and \
                    (not pos.startswith("N") and not pos.startswith("I") and not pos.startswith("C") and not pos.startswith("D")
                    or len(the_word)>3):
                # if pos.startswith("J"):
                #     best_scores[i] *= (boost_sets[1]+ 0.15)
                # else:
                best_scores[i] *= boost_sets[1]
                if debug:
                    print("      boost07 " + the_word + "  score = " + str(best_scores[i]))
            elif promote_inputs_in_beam and (ner_iob=="B" or ner_iob=="I" or the_word[0].isupper() and len(self.prev_ks) > 0 or the_word.isdigit() ):
                if debug:
                    print("      penalty dla "+the_word+"  score = "+str(best_scores[i]))
                i += 1
                continue
            created_best_scores.append(best_scores[i])
            created_best_scores_id.append(best_scores_id[i])
            created_prev_k_id.append(prev_k[i])
            # elif pos.startswith("N") and not the_word.startswith("<") and (the_word not in src_vocab.stoi or the_word.lower() not in src_vocab.stoi
            #                               or lema not in src_vocab.stoi):
            #     best_scores[i] *= 4
            #     if debug:
            #         print("penalty2 dla "+the_word+"  score = "+str(best_scores[i]))
            i += 1
            stride += 1
            if stride >= self.size:
                break
        self.all_scores.append(self.scores)
        self.scores = torch.tensor(created_best_scores)
        best_scores_id = torch.tensor(created_best_scores_id)
        prev_k = torch.tensor(created_prev_k_id)
        if debug:
            print("--------- score ids= "+str(len(best_scores_id))+"   prevk = "+str(len(prev_k)) )
        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from

        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.size# to avoid premature prunning of beam

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, beta, cov_penalty, length_penalty):
        self.alpha = alpha
        self.beta = beta
        penalty_builder = penalties.PenaltyBuilder(cov_penalty,
                                                   length_penalty)
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty()
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """
        Function to update scores of a Beam that is not finished
        """
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attentions"
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty
