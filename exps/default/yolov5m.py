#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import yolort.models as models

from yolort.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        self.model = models.__dict__['yolov5m'](upstream_version="r6.0",)
        self.model.train()
        return self.model