###########################################
# Project:      CMSIS DSP Library
# Title:        graphviz.py
# Description:  Graphviz generation for the CG Static scheduler
# 
# 
# Target Processor: Cortex-M and Cortex-A cores
# -------------------------------------------------------------------- */
# 
# Copyright (C) 2021-2023 ARM Limited or its affiliates. All rights reserved.
# 
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
############################################
from jinja2 import Environment, PackageLoader, select_autoescape
import pathlib
import os.path

def gengraph(sched,f,config):

    env = Environment(
       loader=PackageLoader("cmsisdsp.cg.scheduler"),
       autoescape=select_autoescape(),
       trim_blocks=True
    )
    
    constObjs = list(set([x[0] for x in sched.constantEdges]))
    template = env.get_template("dot_template.dot")

    nbFifos = len(sched._graph._allFIFOs)

    print(template.render(graph=sched,
      nodes=sched.nodes,
      edges=sched.edges,
      fifos=sched._graph._allFIFOs,
      nbFifos=nbFifos,
      constEdges=sched.constantEdges,
      nbConstEdges=len(sched.constantEdges),
      constObjs=constObjs,
      config=config
      ),file=f)