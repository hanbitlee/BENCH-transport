import random
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from mesa import Agent, Model
from mesa.time import RandomActivation
import argparse

variance = {
    "sa": 0.582,
    "pa": 0.602,
    "fv": 0.641,
    "gv": 0.552,
    "sv": 0.725,
    "cv": 0.664
}

bike_stations = gpd.read_file("station_locations.geojson")  # replace with actual file path
social_housing = gpd.read_file("social_housing.json")  # replace with actual file path
daily_co2_saving_per_adopter = 0.043  # kg of CO2 saved per day per adopter

class HouseholdAgent(Agent):
    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)
        self.h_id = unique_id
        self.position = position  # Store the position
        self.h_group = 0
        self.sta = ""
        self.income = 0
        self.gen = 0
        self.edu = 0
        self.age = 0
        self.know = 0
        self.cee_aw = 0
        self.ed_aw = 0
        self.guilt = "Null"
        self.aware = 0
        self.m_st = "Null"
        self.c_st = "Null"
        self.pbc = 0
        self.sa = 0
        self.pa = 0
        self.fv = 0
        self.gv = 0
        self.sv = 0
        self.cv = 0
        self.U = 0
        self.act = False
        self.er = 0
        self.adopted = False  # Attribute to track adoption
        self.station_access = 0  # Attribute for station access

class CommunityModel(Model):
    def __init__(self, N, case_study="NL", scenario="Ref.SSP2", data="Empirical-survey", learning="Informative", seed=None):
        super().__init__()
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.learning = learning
        self.case_study = case_study
        self.scenario = scenario
        self.data = data
        self.years = []  # Initialize list to store years
        self.adopters_per_year = []
        self.co2_saved_per_year = []
        self.cumulative_co2_saved = 0  # Initialize cumulative CO2 saved
        self.year = 2016
        self.a_com = 0
        self.debugfiles = True
        if seed is not None:
            random.seed(seed)
        self.load_data()
        self.setup_agents()

    def load_data(self):
        try:
            self.cge_nl_ssp2_h = pd.read_csv("cge-nl-ssp2-h.csv")
            print(f"Data loaded successfully for case study {self.case_study} and scenario {self.scenario}.")
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")

    def setup_agents(self):
        housing_locations = social_housing.geometry
        bike_station_locations = bike_stations.geometry

        # Check and reproject CRS to EPSG:3857 (Web Mercator) if needed
        if housing_locations.crs and housing_locations.crs.is_geographic:
            housing_locations = housing_locations.to_crs(epsg=3857)

        if bike_station_locations.crs and bike_station_locations.crs.is_geographic:
            bike_station_locations = bike_station_locations.to_crs(epsg=3857)

        # Union of bike station locations for nearest point calculation
        bike_station_union = bike_station_locations.unary_union

        housing_centroids = []
        for polygon in housing_locations:
            if isinstance(polygon, Polygon):
                centroid = polygon.centroid
            else:
                centroid = max(polygon.geoms, key=lambda p: p.area).centroid
            housing_centroids.append(centroid)

        # Shuffle the centroids to randomize their order
        random.shuffle(housing_centroids)

        # Prepare a list to store agent positions
        agent_positions = []

        # Randomly allocate agents to housing centroids
        for i in range(self.num_agents):
            # Select a random housing location
            position = random.choice(housing_centroids)
            # Create and schedule the agent
            agent = HouseholdAgent(i, self, position)
            self.schedule.add(agent)
            agent_positions.append(position)

            # Calculate distance to the nearest bike station
            nearest_station = nearest_points(position, bike_station_union)[1]
            distance = position.distance(nearest_station)

            # Check if the distance is less than 300 meters
            agent.station_access = 1 if distance < 300 else 0

            # Initialize other agent attributes
            agent.income = random.randint(800, 100000)
            agent.gen = 1 if random.uniform(0, 100) < 66.7 else 2
            sa = random.uniform(0, 100)
            agent.age = 1 if sa < 1.8 else 2 if sa < 35.1 else 3 if sa < 78.9 else 4
            ec = random.uniform(0, 100)
            agent.ecom = 1 if ec < 43.9 else 2 if ec < 64.9 else 3
            ed = random.uniform(0, 100)
            agent.edu = 1 if ed < 57.9 else 2 if ed < 82.5 else 3

            agent.know = random.uniform(1, 7)
            agent.cee_aw = random.uniform(1, 7)
            agent.ed_aw = random.uniform(1, 7)
            agent.aware = (agent.know + agent.cee_aw + agent.ed_aw) / 3
            agent.er = random.uniform(-0.03, -0.01)

            # Assign to groups based on random numbers
            rn = random.uniform(0, 100)
            if rn < 5.5:
                self.assign_group_1(agent)
            elif 5.5 <= rn < 40.19:
                self.assign_group_2(agent)
            elif 40.19 <= rn < 78.17:
                self.assign_group_3(agent)
            elif 78.17 <= rn < 91.69:
                self.assign_group_4(agent)
            elif 91.69 <= rn < 97.39:
                self.assign_group_5(agent)
            elif 97.39 <= rn < 98.36:
                self.assign_group_6(agent)
            else:
                self.assign_group_7(agent)

            # Debug print statement
            print(
                f"Agent {agent.h_id}: Income={agent.income}, Awareness={agent.aware}, Group={agent.h_group}, Access={agent.station_access}")

        # Store agent positions and bike station locations for later use
        self.agent_positions = agent_positions
        self.bike_station_locations = bike_station_locations
        self.tree = KDTree([(p.x, p.y) for p in agent_positions])  # Build KDTree for neighborhood queries

        self.plot_agents()

    def assign_group_1(self, agent):
        agent.h_group = 1
        agent.income = random.randint(800, 10000)
        agent.know = random.uniform(1, 7)
        agent.cee_aw = random.uniform(1.2, 6.8)
        agent.ed_aw = random.uniform(1, 7)
        agent.pa = random.uniform(1, 6.5)
        agent.sa = random.uniform(1, 6.5)
        agent.pbc = random.uniform(1, 7)
        agent.fv = random.gauss(4, variance["fv"] ** 0.5)
        agent.gv = random.gauss(4, variance["gv"] ** 0.5)
        agent.sv = random.gauss(4, variance["sv"] ** 0.5)
        agent.cv = random.gauss(4, variance["cv"] ** 0.5)
        print(f"Agent {agent.h_id} assigned to group 1.")

    def assign_group_2(self, agent):
        agent.h_group = 2
        agent.income = random.randint(10000, 30000)
        agent.know = random.uniform(1, 7)
        agent.cee_aw = random.uniform(1, 6.8)
        agent.ed_aw = random.uniform(1.3, 7)
        agent.pa = random.uniform(2, 7)
        agent.sa = random.uniform(1, 7)
        agent.pbc = random.uniform(1.4, 7)
        agent.fv = random.gauss(4, variance["fv"] ** 0.5)
        agent.gv = random.gauss(4, variance["gv"] ** 0.5)
        agent.sv = random.gauss(4, variance["sv"] ** 0.5)
        agent.cv = random.gauss(4, variance["cv"] ** 0.5)
        print(f"Agent {agent.h_id} assigned to group 2.")

    def assign_group_3(self, agent):
        agent.h_group = 3
        agent.income = random.randint(30001, 50000)
        agent.know = random.uniform(2.6, 7)
        agent.cee_aw = random.uniform(2.71, 7)
        agent.ed_aw = random.uniform(1.4, 7)
        agent.pa = random.uniform(1.5, 7)
        agent.sa = random.uniform(1, 7)
        agent.pbc = random.uniform(2.2, 7)
        agent.fv = random.gauss(4, variance["fv"] ** 0.5)
        agent.gv = random.gauss(4, variance["gv"] ** 0.5)
        agent.sv = random.gauss(4, variance["sv"] ** 0.5)
        agent.cv = random.gauss(4, variance["cv"] ** 0.5)
        print(f"Agent {agent.h_id} assigned to group 3.")

    def assign_group_4(self, agent):
        agent.h_group = 4
        agent.income = random.randint(50001, 70000)
        agent.know = random.uniform(2.8, 6.6)
        agent.cee_aw = random.uniform(3, 6.6)
        agent.ed_aw = random.uniform(1.8, 7)
        agent.pa = random.uniform(2.5, 7)
        agent.sa = random.uniform(1.1, 7)
        agent.pbc = random.uniform(2.2, 7)
        agent.fv = random.gauss(4, variance["fv"] ** 0.5)
        agent.gv = random.gauss(4, variance["gv"] ** 0.5)
        agent.sv = random.gauss(4, variance["sv"] ** 0.5)
        agent.cv = random.gauss(4, variance["cv"] ** 0.5)
        print(f"Agent {agent.h_id} assigned to group 4.")

    def assign_group_5(self, agent):
        agent.h_group = 5
        agent.income = random.randint(70001, 90000)
        agent.know = random.uniform(3, 5.66)
        agent.cee_aw = random.uniform(3.14, 6.42)
        agent.ed_aw = random.uniform(2, 7)
        agent.pa = random.uniform(3.5, 6.25)
        agent.sa = random.uniform(1, 5.8)
        agent.pbc = random.uniform(2.2, 7)
        agent.fv = random.gauss(4, variance["fv"] ** 0.5)
        agent.gv = random.gauss(4, variance["gv"] ** 0.5)
        agent.sv = random.gauss(4, variance["sv"] ** 0.5)
        agent.cv = random.gauss(4, variance["cv"] ** 0.5)
        print(f"Agent {agent.h_id} assigned to group 5.")

    def assign_group_6(self, agent):
        agent.h_group = 6
        agent.income = random.randint(90001, 110000)
        agent.know = random.uniform(3.1, 4.66)
        agent.cee_aw = random.uniform(2.72, 5.57)
        agent.ed_aw = random.uniform(3.66, 6.33)
        agent.pa = random.uniform(3.62, 5.75)
        agent.sa = random.uniform(2, 3.83)
        agent.pbc = random.uniform(2.4, 6.4)
        agent.fv = random.gauss(4, variance["fv"] ** 0.5)
        agent.gv = random.gauss(4, variance["gv"] ** 0.5)
        agent.sv = random.gauss(4, variance["sv"] ** 0.5)
        agent.cv = random.gauss(4, variance["cv"] ** 0.5)
        print(f"Agent {agent.h_id} assigned to group 6.")

    def assign_group_7(self, agent):
        agent.h_group = 7
        agent.income = random.randint(110001, 150000)
        agent.know = random.uniform(3.33, 5.33)
        agent.cee_aw = random.uniform(3.85, 6.21)
        agent.ed_aw = random.uniform(2.66, 6)
        agent.pa = random.uniform(2.62, 7)
        agent.sa = random.uniform(1, 5.66)
        agent.pbc = random.uniform(1.6, 6.6)
        agent.fv = random.gauss(4, variance["fv"] ** 0.5)
        agent.gv = random.gauss(4, variance["gv"] ** 0.5)
        agent.sv = random.gauss(4, variance["sv"] ** 0.5)
        agent.cv = random.gauss(4, variance["cv"] ** 0.5)
        print(f"Agent {agent.h_id} assigned to group 7.")

    def plot_agents(self):
        agent_gdf = gpd.GeoDataFrame({'geometry': self.agent_positions}, crs=self.bike_station_locations.crs)

        fig, ax = plt.subplots(figsize=(10, 10))
        social_housing.to_crs(self.bike_station_locations.crs).plot(ax=ax, color='lightgrey', edgecolor='black',
                                                                    alpha=0.5, label='Social Housing')

        self.bike_station_locations.plot(ax=ax, color='red', markersize=50, label='Bike Stations', marker='o')
        agent_gdf.plot(ax=ax, color='black', markersize=50, label='Agents', marker='o')
        plt.legend()
        plt.title('Spatial Distribution of Agents and Bike Stations')
        if self.bike_station_locations.crs.is_geographic:
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
        else:
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')

        plt.savefig(f'grid_{self.year}.png')  # Save the plot to a file

    def recall_memory(self):
        if self.year == 2016:
            for agent in self.schedule.agents:
                if self.case_study == "NL":
                    if agent.h_group == 1:
                        if random.uniform(0, 100) <= 1.8:
                            agent.act = True
                            agent.sta = "adopted"
                            agent.adopted = True  # Mark agent as adopted
                        else:
                            agent.act = False
                    elif agent.h_group == 2:
                        if random.uniform(0, 100) <= 1.4:
                            agent.act = True
                            agent.sta = "adopted"
                            agent.adopted = True  # Mark agent as adopted
                        else:
                            agent.act = False
                    elif agent.h_group == 3:
                        if random.uniform(0, 100) <= 1.5:
                            agent.act = True
                            agent.sta = "adopted"
                            agent.adopted = True  # Mark agent as adopted
                        else:
                            agent.act = False
                    elif agent.h_group == 4:
                        if random.uniform(0, 100) <= 3.6:
                            agent.act = True
                            agent.sta = "adopted"
                            agent.adopted = True  # Mark agent as adopted
                        else:
                            agent.act = False
                    elif agent.h_group in [5, 6, 7]:
                        if random.uniform(0, 100) <= 1.2:
                            agent.act = True
                            agent.sta = "adopted"
                            agent.adopted = True  # Mark agent as adopted
                        else:
                            agent.act = False
            print(f"Memory recalled for year {self.year}.")

    def debug(self):
        if self.debugfiles:
            with open("debug.csv", "a") as file:
                for agent in self.schedule.agents:
                    file.write(
                        f"{self.year},{agent.h_id},{agent.sta},{agent.act},{agent.income},{agent.gen},{agent.edu},{agent.age},{agent.know},{agent.cee_aw},{agent.ed_aw},{agent.guilt},{agent.aware},{agent.sa},{agent.pa},{agent.fv},{agent.gv},{agent.sv},{agent.cv},{agent.U}\n")
            print("Debug information written to file.")

    def knowledge(self):
        for agent in self.schedule.agents:
            agent.aware = (agent.know + agent.cee_aw + agent.ed_aw) / 3
            if self.case_study == "NL":
                agent.guilt = "L" if agent.aware < 4.6 else "H"
            elif self.case_study == "ES":
                agent.guilt = "L" if agent.aware < 5.2 else "H"
            if agent.guilt == "H":
                agent.k = agent.aware / 7
        print("Knowledge updated.")

    def motivation(self):
        for agent in self.schedule.agents:
            if agent.guilt == "H":
                if self.case_study == "NL":
                    agent.m_st = "L" if (agent.pa < 4.7 or agent.sa < 3.5) else "H"
        print("Motivation updated.")

    def consideration(self):
        for agent in self.schedule.agents:
            if self.case_study == "NL":
                if agent.m_st == "H":
                    agent.c_st = "L" if (agent.pbc < 1) else "H"
        print("Consideration updated.")

    def utility(self):
        for agent in self.schedule.agents:
            if agent.c_st == "L":
                agent.U = 0
            else:
                agent.U = (agent.age * 0.067 - agent.gen * 0.026 + agent.income * 0.019 -
                            agent.edu * 0.021 + agent.fv * 0.283 + agent.gv * 0.074 + agent.sv * 0.024 +
                           + agent.cv * 0.157 + agent.sa * 0.360 + agent.pa * 0.089 +
                           agent.fv * agent.sa * 0.108 + agent.sv * agent.sa * 0.151 +
                           agent.sv * agent.sa * 0.041 + agent.cv * agent.sa * 0.038 +
                           agent.fv * agent.pa * 0.076 + agent.gv * agent.pa * 0.166 +
                           agent.sv * agent.pa * 0.001 + agent.er)

        print("Utility calculated.")

    def go(self):
        self.year += 1
        print(f"Simulation step for year {self.year} started.")
        self.schedule.step()
        self.recall_memory()
        self.debug()
        self.update_info()

        if self.scenario == "Ref.SSP2":
            self.knowledge()
            self.motivation()
            self.consideration()
            self.utility()
            self.action()
            self.learn()
        num_adopters = sum([agent.adopted for agent in self.schedule.agents])
        self.adopters_per_year.append(num_adopters)
        self.years.append(self.year)
        self.calculate_emissions_saved(num_adopters)  # Update emissions saved
        self.plot_grid()  # Plot the grid at the end of each year
        self.print_summary()

    def action(self):
        for agent in self.schedule.agents:
            if agent.adopted:  # Skip agents that have already adopted
                agent.act = False
                continue

            if agent.U > 0 and agent.c_st == "H":
                agent.act = True
                agent.adopted = True  # Mark agent as adopted
            else:
                agent.act = False

        no_action = len([agent for agent in self.schedule.agents if not agent.act])
        a = len([agent for agent in self.schedule.agents if agent.act])
        self.a_com += a

        for agent in self.schedule.agents:
            agent.group1_a1 = len([a for a in self.schedule.agents if a.h_group == 1 and a.act])
            agent.group2_a1 = len([a for a in the schedule.agents if a.h_group == 2 and a.act])
            agent.group3_a1 = len([a for a in the schedule.agents if a.h_group == 3 and a.act])
            agent.group4_a1 = len([a for a in the schedule.agents if a.h_group == 4 and a.act])
            agent.group5_a1 = len([a for a in the schedule.agents if a.h_group in [5, 6, 7] and a.act])

        print("Actions determined for agents.")

    def learn(self):
        if self.year >= 2017:
            for agent in the schedule.agents:
                if self.learning == "No learning":
                    continue

                if self.learning in ["Slow dynamics", "Fast dynamics"]:
                    if agent.act:
                        if agent.pbc < 6.6:
                            agent.pbc += agent.pbc * 0.05

                        neighbors = self.get_neighbors(agent)
                        ngb_k_mean = sum([n.know for n in neighbors]) / len(neighbors)
                        ngb_k_median = sorted([n.know for n in neighbors])[len(neighbors) // 2]
                        ngb_k = max(ngb_k_mean, ngb_k_median)

                        ngb_ca_mean = sum([n.cee_aw for n in neighbors]) / len(neighbors)
                        ngb_ca_median = sorted([n.cee_aw for n in neighbors])[len(neighbors) // 2]
                        ngb_ca = max(ngb_ca_mean, ngb_ca_median)

                        ngb_ea_mean = sum([n.ed_aw for n in neighbors]) / len(neighbors)
                        ngb_ea_median = sorted([n.ed_aw for n in neighbors])[len(neighbors) // 2]
                        ngb_ed = max(ngb_ea_mean, ngb_ea_median)

                        ngb_pa_mean = sum([n.pa for n in neighbors]) / len(neighbors)
                        ngb_pa_median = sorted([n.pa for n in neighbors])[len(neighbors) // 2]
                        ngb_pa = max(ngb_pa_mean, ngb_pa_median)

                        ngb_sa_mean = sum([n.sa for n in neighbors]) / len(neighbors)
                        ngb_sa_median = sorted([n.sa for n in neighbors])[len(neighbors) // 2]
                        ngb_sa = max(ngb_sa_mean, ngb_sa_median)

                        ngb_pbc_mean = sum([n.pbc for n in neighbors]) / len(neighbors)
                        ngb_pbc_median = sorted([n.pbc for n in neighbors])[len(neighbors) // 2]
                        ngb_pbc = max(ngb_pbc_mean, ngb_pbc_median)

                        if len(neighbors) > 4:
                            for n in neighbors:
                                if n.know < ngb_k and n.know < 6.6:
                                    n.know += n.know * 0.05
                                if n.cee_aw < ngb_ca and n.cee_aw < 6.6:
                                    n.cee_aw += n.cee_aw * 0.05
                                if n.ed_aw < ngb_ed and n.ed_aw < 6.6:
                                    n.ed_aw += n.ed_aw * 0.05
                                if n.pa < ngb_pa and n.pa < 6.6:
                                    n.pa += n.pa * 0.05
                                if n.sa < ngb_sa and n.sa < 6.6:
                                    n.sa += n.sa * 0.05
                                if n.pbc < 6.5 and n.pbc < ngb_pbc:
                                    n.pbc += n.pbc * 0.05

                if self.learning == "Informative":
                    if agent.know <= 6.6:
                        agent.know += agent.know * 0.05
                    if agent.cee_aw <= 6.6:
                        agent.cee_aw += agent.cee_aw * 0.05
                    if agent.ed_aw <= 6.6:
                        agent.ed_aw += agent.ed_aw * 0.05
                    if agent.act:
                        if agent.pbc < 6.6:
                            agent.pbc += agent.pbc * 0.05

                        neighbors = self.get_neighbors(agent)
                        ngb_k_mean = sum([n.know for n in neighbors]) / len(neighbors)
                        ngb_k_median = sorted([n.know for n in neighbors])[len(neighbors) // 2]
                        ngb_k = max(ngb_k_mean, ngb_k_median)

                        ngb_ca_mean = sum([n.cee_aw for n in neighbors]) / len(neighbors)
                        ngb_ca_median = sorted([n.cee_aw for n in neighbors])[len(neighbors) // 2]
                        ngb_ca = max(ngb_ca_mean, ngb_ca_median)

                        ngb_ea_mean = sum([n.ed_aw for n in neighbors]) / len(neighbors)
                        ngb_ea_median = sorted([n.ed_aw for n in neighbors])[len(neighbors) // 2]
                        ngb_ed = max(ngb_ea_mean, ngb_ea_median)

                        ngb_pa_mean = sum([n.pa for n in neighbors]) / len(neighbors)
                        ngb_pa_median = sorted([n.pa for n in neighbors])[len(neighbors) // 2]
                        ngb_pa = max(ngb_pa_mean, ngb_pa_median)

                        ngb_sa_mean = sum([n.sa for n in neighbors]) / len(neighbors)
                        ngb_sa_median = sorted([n.sa for n in neighbors])[len(neighbors) // 2]
                        ngb_sa = max(ngb_sa_mean, ngb_sa_median)

                        ngb_pbc_mean = sum([n.pbc for n in neighbors]) / len(neighbors)
                        ngb_pbc_median = sorted([n.pbc for n in neighbors])[len(neighbors) // 2]
                        ngb_pbc = max(ngb_pbc_mean, ngb_pbc_median)

                        for n in neighbors:
                            if n.know < ngb_k and n.know < 6.6:
                                n.know += n.know * 0.05
                            if n.cee_aw < ngb_ca and n.cee_aw < 6.6:
                                n.cee_aw += n.cee_aw * 0.05
                            if n.ed_aw < ngb_ed and n.ed_aw < 6.6:
                                n.ed_aw += n.ed_aw * 0.05
                            if n.pa < ngb_pa and n.pa < 6.6:
                                n.pa += n.pa * 0.05
                            if n.sa < ngb_sa and n.sa < 6.6:
                                n.sa += n.sa * 0.05
                            if n.pbc < 6.5 and n.pbc < ngb_pbc:
                                n.pbc += n.pbc * 0.05

                if self.learning == "Informative-Soft":
                    if agent.know <= 6.6:
                        agent.know += agent.know * 0.05
                    if agent.cee_aw <= 6.6:
                        agent.cee_aw += agent.cee_aw * 0.05
                    if agent.ed_aw <= 6.6:
                        agent.ed_aw += agent.ed_aw * 0.05

        print("Learning process updated.")

    def update_info(self):
        for agent in the schedule.agents:
            agent.act = False
        print("Agent info updated.")

    def get_neighbors(self, agent, radius=500):
        point = agent.position
        indices = self.tree.query_radius([[point.x, point.y]], r=radius)[0]
        neighbors are [the schedule.agents[i] for i in indices if the schedule.agents[i] != agent]
        return neighbors

    def plot_grid(self):
        plt.figure(figsize=(8, 8))
        for agent in the schedule.agents:
            x, y = agent.position.x, agent.position.y
            if agent.adopted:
                plt.plot(x, y, 'bo')  # Red dot for adopted agents
            else:
                plt.plot(x, y, 'ko')  # Blue dot for non-adopted agents
        plt.title(f'Year {self.year}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.savefig(f'grid_{self.year}.png')  # Save the plot to a file
        plt.close()

    def print_summary(self):
        # Print summary statistics for the current year
        num_agents_act = sum(agent.act for agent in the schedule.agents)
        avg_income = sum(agent.income for agent in the schedule.agents) / self.num_agents
        avg_aware = sum(agent.aware for agent in the schedule.agents) / self.num_agents
        num_c_st_H = sum(agent.c_st == "H" for agent in the schedule.agents)
        num_m_st_H = sum(agent.m_st == "H" for agent in the schedule.agents)
        num_station_access = sum(agent.station_access for agent in the schedule.agents)

        print(f"Year: {self.year}")
        print(f"Number of agents taking action: {num_agents_act}")
        print(f"Average income: {avg_income}")
        print(f"Average awareness: {avg_aware}")
        print(f"Number of agents with c_st='H': {num_c_st_H}")
        print(f"Number of agents with m_st='H': {num_m_st_H}")
        print(f"Number of agents with station access: {num_station_access}")

    def end_of_simulation(self):
        # Plot the data at the end of the simulation
        self.plot_adopters()
        self.plot_co2_savings()

    def plot_adopters(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.years, self.adopters_per_year, marker='o', linestyle='-', color='b')
        plt.xlabel('Year')
        plt.ylabel('Number of Bike-share Adopters')
        plt.title('Number of Bike-share Adopters Over Time')
        plt.grid(True)
        plt.savefig('adopters_over_time.png')
        plt.close()
        print("Number of adopters plot saved as 'adopters_over_time.png'.")

    def plot_co2_savings(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.years, self.co2_saved_per_year, marker='o', linestyle='-', color='g')
        plt.xlabel('Year')
        plt.ylabel('CO2 Savings (kg)')
        plt.title('CO2 Savings Over Time')
        plt.grid(True)
        plt.savefig('co2_savings_over_time.png')
        plt.close()
        print("CO2 savings plot saved as 'co2_savings_over_time.png'.")

    def calculate_emissions_saved(self, num_adopters):
        days_per_year = 365
        co2_saved_this_year = num_adopters * daily_co2_saving_per_adopter * days_per_year
        self.cumulative_co2_saved += co2_saved_this_year
        self.co2_saved_per_year.append(self.cumulative_co2_saved)
        print(f"CO2 emissions saved this year: {co2_saved_this_year} kg")
        print(f"Cumulative CO2 emissions saved: {self.cumulative_co2_saved} kg")

def parse_args():
    parser = argparse.ArgumentParser(description="Community Simulation Model")
    parser.add_argument("--num_agents", type=int, default=759, help="Number of agents")
    parser.add_argument("--case_study", type=str, default="NL", help="Case study (ES or NL)")
    parser.add_argument("--scenario", type=str, default="Ref.SSP2", help="Scenario")
    parser.add_argument("--data", type=str, default="Empirical-survey", help="Data type")
    parser.add_argument("--learning", type=str, default="No learning", help="Learning type")
    parser.add_argument("--i1_cost", type=int, default=3000, help="Cost for investment I1")
    parser.add_argument("--i2_cost", type=int, default=4000, help="Cost for investment I2")
    parser.add_argument("--i3_cost", type=int, default=300, help="Cost for investment I3")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generator")
    return parser.parse_args()

def main():
    args = parse_args()
    model = CommunityModel(
        N=args.num_agents,
        case_study=args.case_study,
        scenario=args.scenario,
        data=args.data,
        learning=args.learning,
        seed=args.seed
    )
    for _ in range(10):  # Simulate 10 years
        model.go()
    model.end_of_simulation()

if __name__ == "__main__":
    main()