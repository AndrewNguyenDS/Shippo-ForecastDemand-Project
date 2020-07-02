1. Build your images as defined by docker-compose.yml
   docker-compose build
2. Run the built images as local containers (-d to do it in the background)
   docker-compose up -d

3. Launch a shell from inside the app container
   docker-compose exec app bash

4. Stop the running containers
   docker-compose stop
  
  Then run the jupyter notebook:
  jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
   
About this project:
This project found the solution for the problem in the shipper capacity of Shippo. HR needs to know how many shippers we need for next week.
Thus, this project will follow some tasks:
1/ Forecast demand for Shippo by City and District
2/ Calculate the capacity of shippo if we don't change anything in a number of shippers
3/ Find the number of the shipper we need 
4/ Send the notification for HR
It brings many effects to HR and the operation of Shippo. Great new for Shippo and Operation Research team.