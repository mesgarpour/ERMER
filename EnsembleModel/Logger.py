#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2017 University of Westminster. All Rights Reserved.
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
""" It configures the Python application logger.
"""

import os
import logging


class Logger:
    def __init__(self,
                 path: str,
                 app_name: str,
                 ext: str = "log"):
        """Initialise the objects and constants.
        :param path: the output directory path, where the log file will be saved.
        :param app_name: the application name, which will be used as the log file name.
        :param ext: the log file extension name.
        """
        # create logger
        self.__logger = logging.getLogger(app_name)
        self.__logger.setLevel(logging.DEBUG)
        self.path_full = os.path.join(path, app_name + "." + ext)

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
