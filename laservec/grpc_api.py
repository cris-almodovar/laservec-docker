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


class LaserGrpcApi(laservec_pb2_grpc.LaserGrpcApiServicer):

    def __init__(self, laser):
        self.laser = laser

    def vectorize(self, request, context):
        lang = request.lang
        text = request.text
        snippet_len = 100 if len(text) > 100 else len(text)
        snippet = text[:snippet_len] + (" ..." if len(text) > snippet_len else "")
        logging.info(f"Received request: lang={lang}, text={snippet}")

        lang_detected, embedding = self.laser.vectorize(lang, text)
        return laservec_pb2.VectorizeResponse(lang=lang_detected, embedding=embedding)

    def start(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.LASER_GRPC_API_WORKERS))
        laservec_pb2_grpc.add_LaserGrpcApiServicer_to_server(
            self,
            server)
        endpoint = f"0.0.0.0:{config.LASER_GRPC_API_PORT}"
        server.add_insecure_port(endpoint)
        server.start()
        logging.info(f"LASER GRPC API listening on: {endpoint}")
        server.wait_for_termination()

