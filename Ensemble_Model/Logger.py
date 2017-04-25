#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2017 The Project Authors. All Rights Reserved.
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
# ==============================================================================
""" 

"""
import os
import logging


class Logger:
    def __init__(self, path, app_name, extension="log"):
        """Create & initialise the file and terminal loggers"""
        # create logger
        self.__logger = logging.getLogger(app_name)
        self.__logger.setLevel(logging.DEBUG)
        self.path_full = os.path.abspath(os.path.join(path, app_name + "." + extension))

        # create file handler which logs even debug messages
        self.__file_handler = logging.FileHandler(self.path_full, mode='w')
        self.__file_handler.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        self.__console_handler = logging.StreamHandler()
        self.__console_handler.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.__file_handler.setFormatter(formatter)
        self.__console_handler.setFormatter(formatter)

        # add the handlers to the logger
        self.__logger.addHandler(self.__file_handler)
        self.__logger.addHandler(self.__console_handler)

        # output log
        self.__logger.info("creating " + self.path_full + " file")

    def terminate(self):
        """Terminate the logger"""
        self.__file_handler.close()
        self.__logger.removeHandler(self.__file_handler)
