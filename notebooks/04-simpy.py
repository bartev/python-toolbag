#!/usr/bin/env python
# coding: utf-8

# https://realpython.com/simpy-simulating-with-python/

# In[1]:


get_ipython().run_line_magic('load_ext', 'blackcellmagic')
get_ipython().run_line_magic('load_ext', 'autoreload')


# In[2]:


import simpy
import random
import statistics


# In[3]:


from importlib import reload


# In[4]:


from src.sim_ex import Theater


# # Simple example

# To recap, here are the three steps to running a simulation in Python:
# 
# 1. Establish the environment.
# 2. Pass in the parameters.
# 3. Run the simulation.
# 

# In[5]:


# total amount of time each moviegoer spends moving through the theater
wait_times = []


# In[6]:


def go_to_movies(env, moviegoer, theater):
    """Moviegoer arrives at the theater

    :param env: moviegoer is controlled by the environment
    :param moviegoer: each person as they move through the system
    :param theater: get access to the params defined in Theater

    """
    arrival_time = env.now

    # generate a request to use a cashier
    # use a `with` statement to automatically release the resource
    with theater.cashier.request() as request:
        # wait for cashier to become available
        yield request
        # Use an available cashier to purchase a ticket
        yield env.process(theater.purchase_ticket(moviegoer))


    with theater.usher.request() as request:
        yield request
        yield env.process(theater.check_ticket(moviegoer))

    if random.choice([True, False]):
        with theater.server.request() as request:
            yield request
            yield env.process(theater.sell_food(moviegoer))

    wait_times.append(env.now - arrival_time)
    


# In[7]:



def run_theater(env, num_cashiers, num_servers, num_ushers):
    theater = Theater(env, num_cashiers, num_servers, num_ushers)

    for moviegoer in range(3):
        env.process(go_to_movies(env, moviegoer, theater))

    while True:
        # 0.20 = 1/5 of a min, or 12 sec
        yield env.timeout(0.20) # Wait a bit before generating a new person

        moviegoer += 1
        env.process(go_to_movies(env, moviegoer, theater))
        


# In[10]:


def get_average_wait_time(wait_times):
    average_wait = statistics.mean(wait_times)
    minutes, frac_minutes = divmod(average_wait, 1)
    seconds = frac_minutes * 60
    return round(minutes), round(seconds)


def calculate_wait_time(arrival_times, departure_times):
    # wait_times = departure_times - arrival_times
    average_wait = statistics.mean(wait_times)
    # pretty print results
    minutes, frac_minutes = divmod(average_wait, 1)
    seconds = frac_minutes * 60
    return round(minutes), round(seconds)


def get_user_input():
    num_cashiers = input("# of cashiers: ")
    num_servers = input("# of servers: ")
    num_ushers = input("# of ushers: ")
    params = [num_cashiers, num_servers, num_ushers]
    if all(str(i).isdigit() for i in params):
        params = [int(x) for x in params]
    else:
        print("""Could not parse input. The sim will use default values of 1""")
        params = [1, 1, 1]
    return params

def main():
    # setup
    random.seed(42)
    num_cashiers, num_servers, num_ushers = get_user_input()

    # run the sim
    env = simpy.Environment()
    env.process(run_theater(env, num_cashiers, num_servers, num_ushers))
    env.run(until=90)

    # view the results
    mins, secs = get_average_wait_time(wait_times)
    print(
      "Running simulation...",
      f"\nThe average wait time is {mins} minutes and {secs} seconds.",
    )


# In[28]:


main()


# In[ ]:




