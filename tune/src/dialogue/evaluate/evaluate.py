#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

"""Evaluate generated response."""

import argparse
import json
import re
import string

import os
import tempfile
import subprocess
import numpy as np


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["SMD", "CamRest", "MultiWOZ"], required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    args = parser.parse_args()
    return args

def evaluate(args):
    """Main evaluation function."""
    with open(args.pred_file, "r") as fin:
        data = []
        for line in fin.readlines():
            line_dic = json.loads(line)
            data.append(line_dic)
        print(f"Load prediction file from: {args.pred_file}")

    preds = []
    refs = []
    for dial in data:
        preds.append(preprocess_text(dial["pre"]))
        refs.append(preprocess_text(dial["sys"]))
    assert len(preds) == len(refs), f"{len(preds)} != {len(refs)}"

    entity_metric = EntityMetric(args)
    bleu_res = moses_multi_bleu(np.array(preds), np.array(refs), lowercase=True)
    entity_res = entity_metric.evaluate(preds, refs)
    results = {
        "BLEU": bleu_res,
        "Entity-F1": round(entity_res*100, 2)
    }

    print(json.dumps(results, indent=2))
    return

class EntityMetric(object):
    """Entity Metric for Response."""

    def __init__(self, args):
        self.dataset = args.dataset
        self.entities = self._load_entities("./data/{dataset}/entities.json".format(dataset=args.dataset))

    def evaluate(self, preds, refs):
        extracted_preds_entities = []
        extracted_refs_entities = []
        for pred, ref in zip(preds, refs):
            pred_entities = self._extract_entities(pred)
            ref_entities = self._extract_entities(ref)
            extracted_preds_entities.append(pred_entities)
            extracted_refs_entities.append(ref_entities)
        entity_f1 = self._compute_entity_f1(extracted_preds_entities, extracted_refs_entities)
        return entity_f1

    def _load_entities(self, entities_file):
        with open(entities_file, "r") as fin:
            raw_entities = json.load(fin)
        entities = set()

        if self.dataset == "SMD":
            for slot, values in raw_entities.items():
                for val in values:
                    if slot == "poi":
                        entities.add(val["address"])
                        entities.add(val["poi"])
                        entities.add(val["type"])
                    elif slot == "distance":
                        entities.add(f"{val} miles")
                    elif slot == "temperature":
                        entities.add(f"{val}f")
                    else:
                        entities.add(val)

            # add missing entities
            missed_entities = ["yoga", "tennis", "swimming", "football", " lab ", "doctor", "optometrist", "dentist",
                               "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                               "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "jill", "jack", " hr "]
            for missed_entity in missed_entities:
                entities.add(missed_entity)
            # special handle of "hr"
            entities.remove("hr")

        else:
            for slot, values in raw_entities.items():
                for val in values:
                    if self.dataset == "MultiWOZ" and slot == "choice":
                        val = f"choice-{val}"
                    # print(val)
                    entities.add(val)
                    # entities.add(val.replace(' ','_'))

        processed_entities = []
        for val in entities:
            processed_entities.append(val.lower())
        processed_entities.sort(key=lambda x: len(x), reverse=True)
        return processed_entities

    def _extract_entities(self, response):
        def _is_sub_str(str_list, sub_str):
            for str_item in str_list:
                if sub_str in str_item:
                    return True
            return False

        response = f" {response} ".lower()
        extracted_entities = []

        if self.dataset == "SMD":
            # preprocess response
            for h in range(0, 13):
                response = response.replace(f"{h} am", f"{h}am")
                response = response.replace(f"{h} pm", f"{h}pm")
            for low_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                for high_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    response = response.replace(f"{low_temp}-{high_temp}f", f"{low_temp}f-{high_temp}f")

        for entity in self.entities:
            if self.dataset == "MultiWOZ":
                success_tag = False
                if entity.startswith("choice-"):
                    entity = entity[7:]
                    if entity == "many":
                        if entity in re.sub(r"(many (other types|food types|cuisines)|how many)", " ", response):
                            success_tag = True
                    elif entity == "all":
                        if re.search(r"all (of the|expensive|moderate|cheap)", response):
                            success_tag = True
                    elif entity == "to":
                        success_tag = False
                    else:
                        if re.search(f"(there are|there is|found|have about|have)( only|) {entity}", response):
                            success_tag = True
                elif entity == "centre":
                    if entity in response.replace("cambridge towninfo centre", " "):
                        success_tag = True
                elif entity == "free":
                    if re.search(r"free (parking|internet|wifi)", response):
                        success_tag = True
                elif entity in response or entity.lower() in response.lower():
                    success_tag = True

                if success_tag:
                    extracted_entities.append(entity)
                    response = response.replace(entity, " ")

            else:
                if entity in response and not _is_sub_str(extracted_entities, entity):
                    extracted_entities.append(entity)

        return extracted_entities

    def _compute_entity_f1(self, preds, refs):
        """Compute Entity-F1."""
        def _count(pred, ref):
            tp, fp, fn = 0, 0, 0
            if len(ref) != 0:
                for g in ref:
                    if g in pred:
                        tp += 1
                    else:
                        fn += 1
                for p in set(pred):
                    if p not in ref:
                        fp += 1
            return tp, fp, fn

        tp_all, fp_all, fn_all = 0, 0, 0
        for pred, ref in zip(preds, refs):
            tp, fp, fn = _count(pred, ref)
            tp_all += tp
            fp_all += fp
            fn_all += fn

        precision = tp_all / float(tp_all + fp_all) if (tp_all + fp_all) != 0 else 0
        recall = tp_all / float(tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return f1


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    multi_bleu_path = "./data/multi-bleu.perl"
    os.chmod(multi_bleu_path, 0o755)

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score


def preprocess_text(text):
    """Preprocess utterance and table value."""
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        text = text.replace(p, f" {p} ")
    text = " ".join(text.split())
    return text


if __name__ == "__main__":
    args = setup_args()
    evaluate(args)
