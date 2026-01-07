#!/bin/bash

echo "-------------------------------------------------------"
echo "WARNING -- THIS SCRIPT SHOULD NOT BE RUN FROM TERMINAL"
echo "PLEASE OPEN POSTGRES FROM THE SKILLS NETWORK TOOLBOX"
echo "-------------------------------------------------------"

if [[ ! -z ${PUKE_TOKEN+x} && ! -z ${PUKE_BASE_URL+x} && -f "/tmp/postgres_details.json" ]]; then
  echo "Attempting to stop Postgres"
  POSTGRES_ID=$(jq -r '.POSTGRES_ID' /tmp/postgres_details.json)

  result=$(curl -silent -w "####%{http_code}" --location -X DELETE "${PUKE_BASE_URL}/adapter/${POSTGRES_ID}" \
    -H "Authorization: Bearer ${PUKE_TOKEN}" \
    -H "Content-Type: application/json" \
    --data-raw '{ "component": "postgres" }') > /dev/null
  code=${result#*####}
  json=${result%####*}
  echo $result
  
  if [[ "$code" == "404" ]]; then
    echo "Failed to connect to learner DB provisioner: DB not found."
    exit 1 
  elif [[ "$code" == "403" || "$json" == "Unauthorized" ]]; then
    echo "Failed to stop Postgres due to invalid credentials."
    echo "Cloud IDE has likely been open too long, please restart or contact support."
    exit 1
  elif [[ "$code" == "500" ]]; then
    echo "Failed to stop Postgres due to other reasons."
    echo "Please try again later"
    exit 1
  elif [[ "$code" == "200" ]]; then
    echo "Successfully stopped Postgres"
    rm -Rf /tmp/postgres_details.json
    rm -Rf /tmp/postgres_cli_start
  else
    echo "Failed to stop Postgres"
    exit 1 
  fi
elif [[ ! -f "/tmp/postgres_details.json" ]]; then
  echo "Postgres credentials document not found, nothing to stop."
  exit 126
else
  echo "Puke credentials not set"
  exit 1
fi

echo "-------------------------------------------------------"
echo "WARNING -- THIS SCRIPT SHOULD NOT BE RUN FROM TERMINAL"
echo "PLEASE OPEN POSTGRES FROM THE SKILLS NETWORK TOOLBOX"
echo "-------------------------------------------------------"