import os
from neo4j import GraphDatabase

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USERNAME')
pwd = os.getenv('NEO4J_PASSWORD')
print('Using URI:', uri)
try:
    drv = GraphDatabase.driver(uri, auth=(user, pwd))
    with drv.session() as s:
        print(s.run('RETURN 1 as x').single()['x'])
    print('OK')
except Exception as e:
    print('ERR', e)
    raise
