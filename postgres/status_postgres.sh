#!/bin/bash

echo "-------------------------------------------------------"
echo "WARNING -- THIS SCRIPT SHOULD NOT BE RUN FROM TERMINAL"
echo "PLEASE OPEN POSTGRES FROM THE SKILLS NETWORK TOOLBOX"
echo "-------------------------------------------------------"

if [[ -f "/tmp/postgres_details.json" ]]; then
  echo "Credentials document found; service is up"
  exit 0  
else
  echo "Credentials document not found; service is down"
  exit 1  
fi
