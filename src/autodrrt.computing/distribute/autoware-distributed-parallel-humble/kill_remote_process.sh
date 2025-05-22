ssh root@192.168.3.2 -p 19010 "ps -ef|grep ros | awk '{print \$2}'| xargs kill -9"
ssh root@192.168.3.3 -p 19010 "ps -ef|grep ros | awk '{print \$2}'| xargs kill -9"
ssh root@192.168.3.4 -p 19010 "ps -ef|grep ros | awk '{print \$2}'| xargs kill -9"
ssh root@192.168.3.5 -p 19010 "ps -ef|grep ros | awk '{print \$2}'| xargs kill -9"