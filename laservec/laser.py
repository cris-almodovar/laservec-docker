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

import logging
import tempfile
from pathlib import Path

import langdetect
import numpy as np
from laservec.LASER.source.embed import *
from laservec.LASER.source.lib.text_processing import Token, BPEfastApply


VECTOR_LEN = 1024


class LaserEncoder(object):
    """
    A wrapper for Facebook's LASER (Language-Agnostic SEntence Representations) toolkit.

    See: https://github.com/facebookresearch/LASER
    """
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

    def vectorize(self, text, lang=None):
        """
        Converts the given text into a 1024-dimensional vector representation
        that is agnostic to the source language. The computed vector is "universal"
        in that it lives in an embedding space that is shared among all the languages
        supported by LASER.

        :param text: The input text, consisting of one or more sentences. This parameter is mandatory.

        :param lang: The ISO 639-1 code of the source language. See: https://github.com/facebookresearch/LASER#supported-languages
                     This parameter is optional; if not specified, it will auto-detected.

        :return: A tuple (embedding, lang) containing the 1024-dimensional vector representation
                of the input text (as a numpy array), and the ISO 639-1 language code of the input text.
                Note: if the input text consists of more than one sentence, the returned
                vector is the mean of the individual sentence vectors.
        """
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

            # Tokenize the text
            if lang != '--':
                tokenized_fname = temp_dir / "input.tok"
                Token(str(input_fname),
                      str(tokenized_fname),
                      lang=lang,
                      romanize=True if lang == 'el' else False,
                      lower_case=True,
                      gzip=False,
                      verbose=True,
                      over_write=False)
                input_fname = tokenized_fname

            # Encode the text using BPE
            BPEfastApply(str(input_fname),
                         str(bpe_encoded_fname),
                         str(self.bpe_codes_path),
                         verbose=True, over_write=False)

            # Finally, compute the embedding vector.
            input_fname = bpe_encoded_fname
            EncodeFile(self.encoder,
                       str(input_fname),
                       str(raw_output_fname),
                       verbose=True,
                       over_write=False,
                       buffer_size=10000)

            raw_embeddings = np.fromfile(str(raw_output_fname), dtype=np.float32, count=-1)
            row_count = raw_embeddings.shape[0] // VECTOR_LEN
            col_count = VECTOR_LEN
            raw_embeddings.resize(row_count, col_count)

            # If the input text contains more than one sentence, a vector is computed for each sentence.
            # Combine the individual sentence vectors by computing their mean.
            embedding = raw_embeddings.mean(axis=0)
            return embedding, lang
