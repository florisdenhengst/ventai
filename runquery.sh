#!/bin/bash

DBNAME="mimicpeine"
SCHEMA_QRY="mimiciii"
SCHEMA="public"

if [ -z $1 ]; then
	echo "Run script with query: ./runquery.sh queryfile.sql"
fi

sed "s/$SCHEMA_QRY/$SCHEMA/g" $1  | psql $DBNAME
