#!/bin/bash
 ps -ef | grep ros | awk '{print $2}'| xargs kill -9
 ps -ef | grep autoware | awk '{print $2}'| xargs kill -9
 ps -ef | grep python3 | awk '{print $2}'| xargs kill -9
