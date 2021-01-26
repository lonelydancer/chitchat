#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import math
import pdb
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from six.moves import xrange

from model.ernie_v1 import ErnieModel

from reader.tokenization import BasicTokenizer
#from eval.gen_eval import GenerationEval


class ErnieLMFinetune(object):
    def __init__(self, args, ernie_config, tokenizer):
        self.vocab = tokenizer.vocab
        self.inv_vocab = tokenizer.inv_vocab
        self.merge_subword = tokenizer.merge_subword
        self.eos_idx = self.vocab["[SEP]"]
        self.ernie_config = ernie_config
        self.weight_sharing = args.weight_sharing
        self.task_type = args.task_type
        self.max_seq_len = args.max_seq_len
        self.use_fp16 = args.use_fp16
        self.label_smooth = args.label_smooth
        self.max_dec_len = args.max_dec_len
        self.beam_size = args.beam_size
        self.tgt_type_id = args.tgt_type_id
        self.continuous_position = args.continuous_position
        self.length_penalty = args.length_penalty
        self.do_decode = args.do_decode
#        self.evaluator = GenerationEval(args, self.merge_subword)
        self.emb_keys = ["word_embedding", "role_embedding", "turn_embedding", "pos_embedding"]

    def cal_logit(self, enc_out, tgt_pos):
        enc_out = fluid.layers.reshape(x=enc_out,
                shape=[-1, self.ernie_config["hidden_size"]])
        if tgt_pos:
            tgt_pos = fluid.layers.cast(x=tgt_pos, dtype='int32')
            tgt_feat = fluid.layers.gather(input=enc_out, index=tgt_pos)
        else:
            tgt_feat = enc_out

        tgt_trans_feat = fluid.layers.fc(
            input=tgt_feat,
            size=self.ernie_config["emb_size"] or self.ernie_config["hidden_size"],
            act=self.ernie_config["hidden_act"],
            param_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.w_0",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.b_0",
                initializer=fluid.initializer.Constant(0.)))

        tgt_trans_feat = fluid.layers.layer_norm(
                tgt_trans_feat,
                begin_norm_axis=len(tgt_trans_feat.shape) - 1,
                param_attr=fluid.ParamAttr(
                    name='mask_lm_trans_layer_norm_scale',
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name='mask_lm_trans_layer_norm_bias',
                    initializer=fluid.initializer.Constant(1.)))


        seq2seq_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        if self.weight_sharing:
            fc_out = fluid.layers.matmul(
                x=tgt_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    "word_embedding"),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self.ernie_config['vocab_size']],
                dtype="float32",
                attr=seq2seq_out_bias_attr,
                is_bias=True)
        else:
            out_size = self.ernie_config["tgt_vocab_size"] or self.ernie_config['vocab_size']
            fc_out = fluid.layers.fc(input=tgt_trans_feat,
                    size=out_size,
                    param_attr=fluid.ParamAttr(
                        name="mask_lm_out_fc.w_0",
                        initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                    bias_attr=seq2seq_out_bias_attr)

        return fc_out

    def to_ternsor(self, shapes, dtypes, lod_levels):
        return [fluid.layers.data(name="placeholder_" + str(i),  shape=shapes[i], dtype=dtypes[i], \
            lod_level=lod_levels[i]) for i in range(len(shapes))]

    def create_model(self):
        '''
         token_id
         position_id
         sentence_id
         role_id
        '''


        src_ids = fluid.layers.data(name='src_ids', shape=[-1, self.max_seq_len, 1], dtype='int64')
        sent_ids = fluid.layers.data(name='sent_ids', shape=[-1, self.max_seq_len, 1], dtype='int64')
        pos_ids = fluid.layers.data(name='pos_ids', shape=[-1, self.max_seq_len, 1], dtype='int64')
        role_ids = fluid.layers.data(name='role_ids', shape=[-1, self.max_seq_len, 1], dtype='int64')
        input_mask = fluid.layers.data(name='input_mask', shape=[-1, self.max_seq_len, 1], dtype='float32')
        inputs = [src_ids, sent_ids, pos_ids, role_ids, input_mask]
#pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=80, iterable=False)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=1, iterable=False)
        '''
        x  = fluid.layers.Print(src_ids, message="Print src_ids")
        x  = fluid.layers.Print(sent_ids, message="Print sent_ids")
        fluid.layers.Print(pos_ids, message="position data")
        fluid.layers.Print(role_ids, message="role data")
        '''
#        tgt_labels, tgt_pos = inputs[-2:]

        print ('ernie begin')
        ernie = ErnieModel(
            src_ids=src_ids,
            position_ids=pos_ids,
            sentence_ids=sent_ids,
            role_ids=role_ids,
            input_mask=input_mask,
            config=self.ernie_config,
            use_fp16=self.use_fp16
        )
        fluid.layers.Print(input_mask, message="input mask123")
        print ('enc_out before')
        enc_out = ernie.get_sequence_output()
        print (enc_out)
        fluid.layers.Print(enc_out, message="enc_out")

        size=self.ernie_config['vocab_size']
        fluid.layers.Print(enc_out, message="enc_out2")
        logits = fluid.layers.fc(
        input=enc_out,
        size=self.ernie_config['vocab_size'],
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_seq_label_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_seq_label_out_b",
            initializer=fluid.initializer.Constant(0.)))

        fluid.layers.Print(logits, message="logits data")
        infers = fluid.layers.argmax(logits, axis=2)
        fluid.layers.Print(infers, message="infer data")

        labels = src_ids#[1:] 
        data = fluid.layers.Print(labels, message="Print labels:")
        data = fluid.layers.Print(infers, message="Print infers:")
        '''
        fc_out = enc_out

        if self.label_smooth:
            out_size = self.ernie_config["tgt_vocab_size"] or self.ernie_config['vocab_size']
            labels = fluid.layers.label_smooth(
                label=fluid.layers.one_hot(
                    input=tgt_labels, depth=out_size),
                epsilon=self.label_smooth)

            ce_loss = layers.softmax_with_cross_entropy(
                logits=fc_out, label=labels, soft_label=True)
            #probs = fluid.layers.log(fluid.layers.softmax(fc_out))
            #ce_loss = fluid.layers.kldiv_loss(probs, labels, reduction='batchmean')
        else:
            ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                logits=infers, label=labels, return_softmax=True)
        loss = fluid.layers.mean(x=ce_loss)
        '''
# loss = fluid.layers.fill_constant(shape=[1,-1],value=1, dtype='float32')
#pdb.set_trace()
        loss, probs = fluid.layers.softmax_with_cross_entropy(
                logits=logits, label=labels, return_softmax=True)
    
#        print ('loss')
        loss = fluid.layers.mean(x=loss)
        graph_vars = {"loss": loss}
        graph_vars['labels'] = labels
        graph_vars['infers'] = infers
        graph_vars['sent_ids'] = sent_ids
        for k, v in graph_vars.items():
            v.persistable = True
        return pyreader, graph_vars


    def post_process_seq(self, seq):
        """
        Post-process the beam-search decoded sequence. Truncate from the first
        <eos> and remove the <bos> and <eos> tokens currently.
        """
        eos_pos = len(seq)
        for i, idx in enumerate(seq):
            if idx == self.eos_idx:
                eos_pos = i
                break
        seq = seq[1:eos_pos]
        return seq

    def evaluate(self, resource, eval_phase, graph_vars, features=None,
            output_path=None, dev_count=1, gpu_id=0):
        exe, program, pyreader = resource["exe"], resource["program"], resource["pyreader"]
#   print ('abcde')
        print ('output path', output_path)
#        output_path = 'log/'
#        eval_phase = 'test'
        if eval_phase == "train":
            fetch_list = [graph_vars["loss"].name]
            if "learning_rate" in graph_vars:
                fetch_list.append(graph_vars["learning_rate"].name)
            outputs = exe.run(fetch_list=fetch_list)
            np_loss = outputs[0]
            ret = {"loss": np.mean(np_loss), "ppl": np.exp(np.mean(np_loss))}
            if "learning_rate" in graph_vars:
                ret["learning_rate"] = float(outputs[1][0])
            return ret

        if self.do_decode:
            return_numpy = False
            outfile = output_path + "/" + eval_phase
            outfile_part = outfile + ".part" + str(gpu_id)
            writer = open(outfile_part, "w")
            fetch_keys = ["finished_ids", "finished_scores", "data_ids"]
        else:
            steps = 0
            cost = 0.0
            return_numpy = True
            fetch_keys = ["loss",'word_embedding','encoder']

        fetch_list = [graph_vars[key].name for key in fetch_keys]
        time_begin = time.time()
        pyreader.start()
        while True:
            try:
                outputs = exe.run(program=program, fetch_list=fetch_list,
                        return_numpy=return_numpy)
                print (outputs)
                if not self.do_decode:
                    np_loss = outputs[0]
                    cost += np.mean(np_loss)
                    steps += 1
                else:
                    seq_ids, seq_scores, data_ids = outputs
                    seq_ids_list, seq_scores_list = [seq_ids], [
                        seq_scores] if isinstance(
                            seq_ids, paddle.fluid.core.LoDTensor) else (seq_ids, seq_scores)

                    data_ids = np.array(data_ids).reshape(-1).tolist()
                    data_idx = 0

                    for seq_ids, seq_scores in zip(seq_ids_list, seq_scores_list):
                    # How to parse the results:
                    #   Suppose the lod of seq_ids is:
                    #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
                    #   then from lod[0]:
                    #     there are 2 source sentences, beam width is 3.
                    #   from lod[1]:
                    #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
                    #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
                        #hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
                        #scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
                        print (seq_ids, seq_scores)
                        for i in range(len(seq_ids.lod()[0]) -1):  # for each source sentence
                            start = seq_ids.lod()[0][i]
                            end = seq_ids.lod()[0][i + 1]
                            max_cand = None
                            for j in range(end - start):  # for each candidate
                                sub_start = seq_ids.lod()[1][start + j]
                                sub_end = seq_ids.lod()[1][start + j + 1]
                                tokens = [self.inv_vocab.get(idx, "[UNK]")
                                    for idx in self.post_process_seq(
                                        np.array(seq_ids)[sub_start:sub_end])
                                ]
                                score = np.array(seq_scores)[sub_end - 1]
                                if (not max_cand) or score > max_cand[1]:
                                    max_cand = (tokens, score)

                            data_id = data_ids[data_idx]
                            data_idx += 1
                            print ('data_idx', data_idx)
                            pred = self.merge_subword(max_cand[0]) 
                            print (" ".join(pred).encode("utf8"))
                            writer.write("%d\t%s\n" % (data_id, " ".join(pred).encode("utf8")))

            except fluid.core.EOFException:
                pyreader.reset()
                break

        eval_result = "no result"
        if not self.do_decode:
            eval_result = "loss: %f, ppl: %f" % (cost / steps, np.exp(cost / steps))
        else:
            writer.close()
            tmp_writer = open("%s/%s_dec_finish.%d" % (output_path, eval_phase, gpu_id), "w")
            tmp_writer.close()
            if gpu_id != 0:
                return
            while True:
                ret = os.popen('find %s -maxdepth 1 -name "%s_dec_finish.*"' %
                        (output_path, eval_phase)).readlines()
                if len(ret) != dev_count:
                    time.sleep(1)
                    continue
                else:
                    break

            time_end = time.time()
            os.system("sort -t $'\t' -k 1 -n %s.part* | awk -F \"\t\" '{print $2}'> %s" %
                    (outfile, outfile))
            os.system("rm %s.part*" % (outfile))
            os.system("rm %s/%s_dec_finish.*" % (output_path, eval_phase))
            '''
            eval_result = self.evaluator.eval(outfile,
                    phase=eval_phase.split("_")[0], features=features)
            print("[%s evaluation] %s, elapsed time: %f s"
                % (eval_phase, eval_result, time_end - time_begin))
            '''
