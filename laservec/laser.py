# Copyright 2019 Cris Almodovar
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

import os
import logging
import tempfile
from pathlib import Path
import numpy as np
from laservec.LASER.source.lib.text_processing import Token, BPEfastApply
from laservec.LASER.source.embed import *
import langdetect


class LaserEncoder:
    def __init__(self):
        self.model_dir = Path(__file__).parent / "LASER" / "models"
        self.encoder_path = self.model_dir / "bilstm.93langs.2018-12-26.pt"
        self.bpe_codes_path = self.model_dir / "93langs.fcodes"
        logging.info(f"Loading LASER encoder from: {self.encoder_path}")
        self.encoder = SentenceEncoder(self.encoder_path,
                                  max_sentences=None,
                                  max_tokens=12000,
                                  sort_kind='mergesort',
                                  cpu=True)

    def vectorize(self, lang, text):
        if not lang:
            lang = langdetect.detect(text)
        if not lang:
            lang = "en"

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            input_fname = temp_dir / "input.txt"
            bpe_encoded_fname = temp_dir / 'input.bpe'
            raw_output_fname = temp_dir / 'output.raw'

            with input_fname.open("w") as f:
                f.write(text)

            if lang != '--':
                tok_fname = temp_dir / "tok"
                Token(str(input_fname),
                      str(tok_fname),
                      lang=lang,
                      romanize=True if lang == 'el' else False,
                      lower_case=True,
                      gzip=False,
                      verbose=True,
                      over_write=False)
                input_fname = tok_fname

            BPEfastApply(str(input_fname),
                         str(bpe_encoded_fname),
                         str(self.bpe_codes_path),
                         verbose=True, over_write=False)

            input_fname = bpe_encoded_fname
            EncodeFile(self.encoder,
                       str(input_fname),
                       str(raw_output_fname),
                       verbose=True,
                       over_write=False,
                       buffer_size=10000)

            VECTOR_LEN = 1024
            raw_embeddings = np.fromfile(str(raw_output_fname), dtype=np.float32, count=-1)
            row_count = raw_embeddings.shape[0] // VECTOR_LEN
            col_count = VECTOR_LEN
            raw_embeddings.resize(row_count, col_count)

            # Combine the individual sentence vectors by computing the mean.
            raw_embeddings = raw_embeddings.mean(axis=0)
            embedding = raw_embeddings.tolist()

            return lang, embedding
