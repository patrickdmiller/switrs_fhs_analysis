'''
Patrick Miller
switrs data transformation > csv
'''

import argparse
from operator import index
import sqlite3
import csv
import re

def build_query(cols, limit=0):
  q = "SELECT  "
  for i,c in enumerate(cols):
    q+=c
    if i < len(cols)-1:
      q+=", "
  q+=" FROM collisions "
  if limit > 0:
    q+=" LIMIT "+str(limit)
  return q

def do_query(df, query):
  with sqlite3.connect(df) as con:
    pass
  
  cur = con.execute(query)
  rows = cur.fetchall()
  return rows

def write_csv(headers, data):
  with open(args.out, 'w', encoding='UTF8') as file:
    writer = csv.writer(file)
    
    writer.writerow(headers)
    writer.writerows(data)
    
    
def _BEFORE(row):
  return row

def _AFTER(row):
  return row

def _ADD_HOUR(row, col_index):
  marker = row[col_index].find(':')
  row[col_index]=int(row[col_index][0:(marker)])
  return row
  # print(row[col_index])

def _CAP_1(row, col_index):
  row[col_index]=min(row[col_index], 1)
  return row
def _CAP_INJURY(row, col_index):
  #if anyone died we're going to say there was a severe injury
  deaths =  [
    'collisions.severe_injury_count',
    'collisions.killed_victims', 
    'collisions.pedestrian_killed_count',
    'collisions.motorcyclist_killed_count',
    'collisions.bicyclist_killed_count'
  ]
  
  for d in deaths:
    
    if row[index_for_col[d]]!= None and row[index_for_col[d]] > 0:
      row[col_index] = 1
      break;
    
  row[col_index]=min(row[col_index], 1)
  return row

def _ADD_YEAR(row, col_index):
  row[col_index]=int(re.sub(r'-[0-9]{2}-[0-9]{2}','',row[col_index]))
  return row

def _ADD_MONTH(row, col_index):
  
  row[col_index]=int(re.sub(r'-[0-9]{2}', '', re.sub(r'[0-9]{4}-','', row[col_index])))
  return row
def _CHECK_NONE(row, col_index):
  if row[col_index] is None or row[col_index]=='None':
    if cols[col_index] in transform['NONE_REPLACEMENT']:
      row[col_index] = transform['NONE_REPLACEMENT'][cols[col_index]]
    else:
      print(cols[col_index], "is None", row)
    
  return row
def clean(_rows):
  rows = []
  for _r in _rows:
    r = list(_r)
    r = transform['BEFORE'](r)
    
    for i, c in enumerate(cols):
      r =_CHECK_NONE(r,i) 
      if c in transform:
        r= transform[c](r, i)    
      
        
    r = transform['AFTER'](r)
    rows.append(r)
  return rows

def clean_headers(_headers):
  headers = _headers.copy()
  for h, header in enumerate(headers):
    new_header  = header
    for modifier in transform['RENAME']:
      new_header= new_header.replace(modifier['pattern'], modifier['repl'])
    headers[h] = new_header
  return headers
cols = [
  'collisions.weather_1',
  'collisions.state_highway_indicator',
  'collisions.type_of_collision',
  'collisions.road_condition_1',
  'collisions.lighting',
  'collisions.motorcycle_collision',
  'collisions.bicycle_collision',
  'collisions.pedestrian_collision',
  'collisions.alcohol_involved',
  'collisions.severe_injury_count',
  'collisions.killed_victims',
  'collisions.pedestrian_killed_count',
  'collisions.motorcyclist_killed_count',
  'collisions.bicyclist_killed_count',
  'collisions.pcf_violation_category',
  'collisions.collision_date'
]

index_for_col = {}
for i, cname in enumerate(cols):
  index_for_col[cname]=i

transform = {
  "BEFORE":_BEFORE,
  "collisions.collision_time":_ADD_HOUR,
  "collisions.collision_date":_ADD_YEAR,
  "collisions.severe_injury_count":_CAP_INJURY,
  "AFTER":_AFTER,
  "NONE_REPLACEMENT":{
    'collisions.alcohol_involved':0,
    'collisions.location_type':'_None_',
    'collisions.weather_1':'_None_',
    'collisions.road_surface':'_None_',
    'collisions.road_condition_1':'_None_',
    'collisions.lighting':'_None_',
    'collisions.pcf_violation_category':'_None_',
    'collisions.type_of_collision':'_None_',
    'collisions.killed_victims':0,
    'collisions.state_highway_indicator':0 
  },
  "RENAME":[
    {
      'pattern':'collisions.',
      'repl':''
    },
    {
      'pattern':'_1',
      'repl':''
    },
    
  ]
}


if __name__ == '__main__':
  #input_file,  start index, end index, covar-name
  parser=argparse.ArgumentParser()
  parser.add_argument('--df', help='path to database', default="./switrs.sql")
  parser.add_argument('--out', help='path to outputfile', default="./out.csv")
  parser.add_argument('--limit', type=int, help='limit number of results', default=0)
  args=parser.parse_args()
  
  query = build_query(cols, args.limit)
  print("query:", query)
  rows = do_query(args.df, query)
  rows = clean(rows)
  # print(rows)
  headers = clean_headers(cols)
  print(headers)
  write_csv(headers, rows)
  
