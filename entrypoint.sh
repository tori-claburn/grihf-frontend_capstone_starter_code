#!/bin/bash

if [ -f "$IBMCLOUD_API_KEY_LOCATION" ]; then
  export IBMCLOUD_API_KEY=$(cat $IBMCLOUD_API_KEY_LOCATION)
fi

if [[ ! -z ${IBMCLOUD_API_KEY+x} && "${PRELAUNCH_K8S}" == "true"  ]]; then
  ibmcloud login -r us-south

  echo "Waiting for ${DOCKER_CERT_PATH}/ca.pem"
  timeout=5
  until [ -f ${DOCKER_CERT_PATH}/ca.pem ]
  do
    if [ "$timeout" == 0 ]; then
      echo "ERROR: Timeout while waiting for the file ${DOCKER_CERT_PATH}/ca.pem"
      break
    fi
    sleep 2
    echo "waiting..."
    ((timeout--))
  done

  if [ "$timeout" != 0 ]; then
    echo "${DOCKER_CERT_PATH}/ca.pem found"
    ibmcloud cr login
  fi

  # Set SN_ICR_NAMESPACE for the user
  export SN_ICR_NAMESPACE=$(ibmcloud cr namespace-list | grep sn-labs- | xargs)

  # This will upsert the "icr" secret
  kubectl create secret docker-registry icr --docker-server=us.icr.io --docker-username=iamapikey --docker-password=$IBMCLOUD_API_KEY -o yaml --save-config --dry-run=client | kubectl apply -f -

  # Set the SN context
  export SN_KUBECTL_CONTEXT=$(kubectl config current-context)

  if [ -z ${SN_ICR_NAMESPACE+x} ]; then
    echo "ERROR: SN_ICR_NAMESPACE not set"
    exit 1
  fi
fi

if [ ! -z ${DHT+x} ]; then
  DHT_DECODED=$(echo $DHT | base64 -d)
  DHT_PARTS=(${DHT_DECODED//:/ })

  docker login -u ${DHT_PARTS[0]} -p ${DHT_PARTS[1]}
  kubectl create secret docker-registry dh --docker-server=index.docker.io/v2 --docker-username=${DHT_PARTS[0]} --docker-password=${DHT_PARTS[1]}
fi

if [ -f "/etc/kube/config" ]; then
  sudo chmod go-r /etc/kube/config
fi

export MONGO_PASSWORD="$(echo "$RANDOM-$USERNAME-$RANDOM" | base64 | cut -c1-16)"
export CASSANDRA_PASSWORD="$(echo "$RANDOM-$USERNAME-$RANDOM" | base64 | cut -c1-16)"
export POSTGRES_PASSWORD="$(echo "$RANDOM-$USERNAME-$RANDOM" | base64 | cut -c1-16)"
export PGPASSWORD="$POSTGRES_PASSWORD"
export AIRFLOW_PASSWORD="$(echo "$RANDOM-$USERNAME-$RANDOM" | base64 | cut -c1-16)"
export _AIRFLOW_WWW_USER_PASSWORD="$AIRFLOW_PASSWORD"
export JAVA_HOME=$(find "/usr/lib/jvm" -type d -name "jdk-21*" -print -quit)

export PATH=$PATH:/home/theia/.local/bin


# cp /home/theia/.sn/WELCOME_README.md /home/project/README.md

cp -Rf /home/theia/.theia /home/project/.theia

sudo service cron start

# yarn theia start /home/project --hostname=0.0.0.0 --port=$THEIA_LOCAL_PORT

sudo chmod 666 "/etc/hosts"
sudo chmod 664 /home/theia/.my.cnf

node "/home/theia/applications/browser/lib/backend/main.js" /home/project --hostname=0.0.0.0 --port=$THEIA_LOCAL_PORT --log-level=warn
