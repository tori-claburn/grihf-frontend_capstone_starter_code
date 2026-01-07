#!/bin/bash

echo "-------------------------------------------------------"
echo "WARNING -- THIS SCRIPT SHOULD NOT BE RUN FROM TERMINAL"
echo "PLEASE OPEN POSTGRES FROM THE SKILLS NETWORK TOOLBOX"
echo "-------------------------------------------------------"
if [[ -f "/tmp/postgres_details.json" ]]; then
  echo "Postgres credentials document already exists, please stop the existing Postgres database before starting a new one."
  exit 126
fi 
if [[ ! -z ${PUKE_TOKEN+x} && ! -z ${PUKE_BASE_URL+x} ]]; then 
  echo "Attempting to start Postgres Database"

  result=$(curl --silent -w "####%{http_code}" --location -X POST "${PUKE_BASE_URL}/adapter" \
    -H "Authorization: Bearer ${PUKE_TOKEN}" \
    -H "Content-Type: application/json" \
    --data-raw "{ \"component\": \"postgres\", \"username\": \"${USERNAME}\" }") > /dev/null
  code=${result#*####}
  json=${result%####*}

  if [[ "$code" == "404" ]]; then
    echo "Failed to connect to learner DB provisioner."
  elif [[ "$code" == "403" || "$json" == "Unauthorized" ]]; then
    echo "Failed to launch Postgres due to invalid credentials."
    echo "Cloud IDE has likely been open too long, please restart or contact support."
  elif [[ "$code" == "500" ]]; then 
    echo "Failed to launch Postgres due to other reasons."
    echo "Please try again later"
  else
    # pull information out of the json object
    POSTGRES_ID=$(echo "$json" | jq -r .id)
    POSTGRES_USERNAME=$(echo "$json" | jq -r .username)
    POSTGRES_PASSWORD=$(echo "$json" | jq -r .password)
    POSTGRES_IP=$(echo "$json" | jq -r .ip)
    POSTGRES_PGADMIN_URL=$(echo "$json" | jq -r .pgAdminUrl)
    # write information to file 
    COMMAND="export PGPASSWORD=${POSTGRES_PASSWORD}; psql --host ${POSTGRES_IP} -p 5432 -U ${POSTGRES_USERNAME}"
    jq -n --arg c "$COMMAND" \
    --arg usr "$POSTGRES_USERNAME" \
    --arg u "$POSTGRES_PGADMIN_URL" \
    --arg p "$POSTGRES_PASSWORD" \
    --arg ip "$POSTGRES_IP" \
    --arg id "$POSTGRES_ID" \
    '{"POSTGRES_USERNAME": $usr, "POSTGRES_HOST": $ip, "POSTGRES_PORT": "5432", "POSTGRES_URL": $u, "POSTGRES_COMMAND": $c, "POSTGRES_PASSWORD": $p, "POSTGRES_TITLE": "Postgres Database", "POSTGRES_ID": $id}' > /tmp/postgres_details.json
    echo $COMMAND > /tmp/postgres_cli_start

    echo ""
    echo "Your PostgreSQL database is now ready to use and available with username: ${POSTGRES_USERNAME} password: ${POSTGRES_PASSWORD}"
    echo ""
    echo "You can access your PostgreSQL database via:"
    echo " • The browser at: ${POSTGRES_PGADMIN_URL}"
    echo " • Command Line: ${COMMAND}"
  fi
else
  echo "Puke not configured"
  exit 1
fi 
echo "-------------------------------------------------------"
echo "WARNING -- THIS SCRIPT SHOULD NOT BE RUN FROM TERMINAL"
echo "PLEASE OPEN POSTGRES FROM THE SKILLS NETWORK TOOLBOX"
echo "-------------------------------------------------------"

exit 0