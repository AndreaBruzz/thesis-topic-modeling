DOCKER="docker compose up -d"
CODE="python3 main.py -ri tipster_45_kstem -fm Qrels -vc \"Terms window\" -et \"Frozen Ranking\" -em embedding_full -md NMF -tp join"

echo $DOCKER
$DOCKER

echo $CODE
eval $CODE
