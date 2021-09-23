# INFEKTA-HD

## Getting started/ Installation

First create a conda env with the dependencies.
 
 ```bash
 conda create --name infekta_env --file requirements.txt
 ```

## Usage

### Getting OSM data


Visit http://overpass-turbo.eu/ and export the data by using the query.gql (--> data/special.geojson) and query_all_buildings.gql (--> data/buildings.geojson). You can change the search area in `{{geocodeArea:Mannheim-Quadrate}}->.searchArea;` line.

### Generate Data/Routes

invoke the script for data generation `generate_routes.py` (you may have to adapt the files loaded in the header of the file).

### Simulate

Simulate with the output files (data/agents.obj and data/places.obj) by calling `python3 simulate.py`.
 This will automatically generate a folder in `data/runs16` with the current timestamp as name.