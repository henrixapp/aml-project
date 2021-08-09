import geopandas
from shapely.ops import nearest_points
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from simulate import Place, Agent, Iterany, SUCEPTIBLE
import numpy as np
def nearest(row, geom_union, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
    """Find the nearest point and return the corresponding value from specified column."""
    # Find the geometry that is closest
    nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df2[nearest]["id"].item()
    return value
def set_nearest_transport(pts3, transportation,l):
    for i,h in tqdm(l.iterrows()):
        l.loc[i,"nearest_transport"] = nearest(h,pts3,transportation)
df = geopandas.read_file("data/special.geojson")
dfagentshomes = geopandas.read_file("data/buildings.geojson")
#transportation
transportation =  df[df["public_transport"]=="platform"].copy()
#shopping and marketplaces
#(df["shop"].notnull()) |
shopping = df[ (df["amenity"]=="marketplace")].copy()
print(shopping.head())
# school and colleges are the same
colleges = df[(df["building"]=="college") | (df["amenity"]=="school")].copy()  
hospitals = df[df["amenity"]=="hospital"].copy()
offices = df[df["office"]=="company"].copy()
housing = df[(df["building"]=="house") | (df["building"]=="apartements")| (df["building"]=="resedential")].copy()
#nursing_home = df[df["social_facility"]=="nursing_home"]
print(df.head())
print("Transport",len(transportation))
print("shopping",len(shopping))
print("Colleges",len(colleges))
print("Offices", len(offices))
print("Housing",len(housing))
print("Agents places",len(dfagentshomes))
#print("nursing homes",len(nursing_home))
both = dfagentshomes.append(transportation)
both.to_file("data/combined.geojson", driver='GeoJSON')
print("=== Setting up  ===")
pts3 = transportation.geometry.unary_union
# set the nearest points
for l in [housing,shopping,colleges,offices,hospitals,dfagentshomes]: # TODO add nursing home
    set_nearest_transport(pts3,transportation,l)
print(housing["nearest_transport"])
fig, ax = plt.subplots(1, 1)
ind = 0
places = []
transpToInd = {}
for i, t in transportation.iterrows():
    places += [Place(ind,t["id"])]
    transpToInd[t["id"]] = ind
    ind += 1
transports = ind
publicT = []
for i,h in dfagentshomes.iterrows():
    places += [Place(ind,h["id"])]
    publicT += [transpToInd[h["nearest_transport"]]]
    ind += 1
agents = []
idh = 0
ind = transports
for i,h in dfagentshomes.iterrows():
    # up to 6 people live in a place
    inhabitants = int(np.floor(np.random.uniform(1,6.5)))
    for i in range(inhabitants):
        age = np.random.randint(0,5)
        workplace = np.random.randint(0,len(dfagentshomes)-1)
        workplacePublicTransport = publicT[workplace]
        it = Iterany(0,[ind,transpToInd[h["nearest_transport"]],workplacePublicTransport,transports+workplace],
        np.array([[0.9,0.1,0,0],
        [0.34,0.01,0.65,0],
        [0,0.65,0.01,0.34],
        [0,0,0.1,.9]]))
        state  =SUCEPTIBLE
        agents += [Agent(idh,0,age,state,it,0)]
        places[ind].agentsInState[state] += 1
    ind += 1

agents_filehandler = open("data/agents.obj","wb")
places_filehandler = open("data/places.obj","wb")
pickle.dump(agents,agents_filehandler)
pickle.dump(places,places_filehandler)


dfagentshomes.plot(column='nearest_transport', ax=ax)
transportation["count"] =0
transportation["count"]= transportation["id"].apply(lambda x: dfagentshomes[dfagentshomes["nearest_transport"]==x].count())
transportation.plot(column='count', ax=ax, legend=True)


plt.show()

#print(h.hist(column="nearest_transport").dtype)
#     print(near)
#     print(transportation[near["id"].first()==transportation["id"]]["count"])
#     transportation[near["id"].first()==transportation["id"]]["count"] =transportation[near["id"][0]==transportation["id"]]["count"] +1 
# # routes 