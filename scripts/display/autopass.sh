#!/bin/bash

pass="$1"
shift
expect -c "
   set timeout -1
   spawn $@
   expect Password:* { send $pass\r; interact }
   sleep 1
   exit
"
