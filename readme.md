# run server 
docker run -d -p 8787:8787 --name frag --mount type=bind,source=/result,target=/app/result 

docker exec -it <container-name> /bin/bash