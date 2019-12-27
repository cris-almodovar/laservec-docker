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

from concurrent import futures
import logging
import grpc
import config

from laservec.proto import laservec_pb2
from laservec.proto import laservec_pb2_grpc
from laservec import LaserEncoder


class LaserGrpcApi(laservec_pb2_grpc.LaserGrpcApiServicer):
    """
    Implements a gRPC API on top of LaserEncoder.
    """

    def __init__(self, laser: LaserEncoder):
        self.laser = laser

    def vectorize(self, request, context):
        text = request.text
        lang = request.lang
        snippet_len = 100 if len(text) > 100 else len(text)
        snippet = text[:snippet_len] + (" ..." if len(text) > snippet_len else "")
        logging.info(f"Received request: lang={lang}, text[:{snippet_len}]={snippet}")

        embedding, lang = self.laser.vectorize(text, lang)
        return laservec_pb2.VectorizeResponse(embedding=embedding, lang=lang)

    def start(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.LASER_GRPC_API_WORKERS))
        laservec_pb2_grpc.add_LaserGrpcApiServicer_to_server(
            self,
            server)
        endpoint = f"0.0.0.0:{config.LASER_GRPC_API_PORT}"
        server.add_insecure_port(endpoint)
        server.start()
        logging.info(f"LASER gRPC API listening on: {endpoint}")
        server.wait_for_termination()

