#!/bin/bash

source ./utils.sh

check_dmesg_keyword() {
    local keyword=$1
    dmesg | grep -i "$keyword" > /dev/null
    if [ $? -eq 0 ]; then
        log_info "dmesg 中存在关键词 $keyword"
    else
        log_warn "dmesg 中未找到关键词 $keyword"
    fi
}
