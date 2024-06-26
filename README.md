



# DrPlotter: Your Go-To for Maximum ROI Chia Farming

DrPlotter is an advanced, energy-efficient GPU plotter, solver, and harvester designed for the Chia Blockchain, with break-through rewards and efficiency optimized for consumer-grade hardware.

- **Up to +442% Rewards**: Two compression modes offer a balance between optimal efficiency and optimal cost per eTB for maximum ROI.
- **Enhanced Security**: Relies solely on public farmer keys, letting you safeguard your private keys.
- **Effortless Remote GPU Use**: Enjoy a streamlined process for submitting proofs and the flexibility to use your GPU remotely with ease.
- **Supporting the Chia Ecosystem**: By seamlessly integrating with the official Chia Farmers, DrPlotter plays a part in keeping the Chia network strong and decentralized.

Offering Eco3x for enhanced energy efficiency and Pro4x for maximum plot size reduction you can achieve higher ROI than any other solution.

<p align="center" alt="DrPlotter eco3x and pro4x summary">
  <img src="images/drplotter-plots-summary-nodevfee.png" />
</p>

<sup>1: The 512 plot filter will be active until June 2024. During this period, the capacities supported by the 256 filter will be doubled.</sup>
<sup>2: At a plot filter of 256 only the 4090 has less W/eTB than standard plots on pro4x.</sup>
<sup>3: **SSD not required.** While you can use an SSD as an intermediary drive under most cases it will not improve plot time. 4090 @ 250W with PCIE 4.0 x 16 can plot in 5:30.</sup>

## The Eco3x Advantage

Eco3x compression offers an energy-efficient solution for managing chia farms, especially with the upcoming chia plot filter of 256. The table below illustrates Eco3x's efficiency compared to traditional uncompressed plots. Eco3x significantly reduces hard drive space and energy consumption, leading to considerable savings in both hardware investment and operational costs. This makes it a highly effective option for Chia users, even those facing high energy expenses, to achieve exceptional ROI.

<p align="center" alt="Table comparing eco3x advantages to regular plots">
  <img src="images/eco3x-advantage.png" />
</p>

<sub>1: Using 0.6W per installed TB</sub>
<sub>2: The "Hardware Cost per ePiB" represents the actual costs of purchasing the necessary hardware for disks and GPUs. This is based on an estimated $16 per installed TB and $800 for a used 3090 GPU</sub>

## The Pro4x Advantage

Pro4x compression significantly reduces hard drive needs for your plots to less than a quarter compared to uncompressed plots, marking a major leap in cost and space savings. By matching your hard drive capacity with your GPU's power, Pro4x achieves unmatched ROI. The table shows that with 4090 GPUs at full capacity on the 256 plot filter, you can efficiently run a 880 eTiB farm at just 35% the cost of a setup using regular plots. Notably, Pro4x enables over 1 ePiB of plots in a consumer PC case using only 14 hard drives, compared to the 56 drives a large server setup would need, enhancing space utilization, cost-effectiveness, and management simplicity.

<p align="center" alt="Table comparing pro4x advantages to regular plots">
  <img src="images/pro4x-advantage.png" />
</p>
<sub>1: Using 0.6W per installed TB</sub>
<sub>2: The "Hardware Cost per ePiB" represents the actual costs of purchasing the necessary hardware for disks and GPUs. This is based on an estimated $16 per installed TB and $1800 for a new 4090 GPU</sub>

## Strategic Efficiency: Eco3x and Pro4x vs. NoSSD's C15
In the competitive landscape of chia farming, striking the right balance between energy consumption and cost per reward is key. Chia farming involves substantial initial setup costs but low ongoing expenses. The Eco3x and Pro4x solutions, developed with these economic dynamics in mind, optimize for a slight increase in energy use to significantly reduce the cost per reward across various setups and market conditions, offering a financial advantage.

The performance of these solutions is contrasted with NoSSD's leading C15 plot format under a fixed $10,000 budget. Analysis shows that both Eco3x and Pro4x not only improve chia earnings but do so with an acceptable increase in energy consumption. Specifically, Eco3x delivers an extra 0.07 xch for 4.79 kWh more than C15, while Pro4x adds 0.10 xch for an additional 11.21 kWh.

<p align="center" alt="DrPlotter Eco3x and Pro4x vs NoSSD C15">
  <img src="images/Eco3x-Pro4x-vs-NoSSDC15-nodevfee.png" />
</p>

<sub>1: Hardware Cost includes expenses for disks and GPUs, calculated at $16 per installed TB and $1600 for a 4090 GPU. GPU costs are adjusted to utilization—e.g., at 38% utilization, only 38% of the GPU cost is counted. This approach ensures costs are proportionally represented for a fair comparison within a $10,000 investment.</sub>
<sub>2: For NoSSD a 3.5% fee from NoSSD is deducted from rewards.</sub>

By evaluating the potential xch price against energy costs, it's clear that Eco3x and Pro4x are compelling for those seeking profitability over minimal energy use. For example, if your electricity costs are $0.14 per kWh with XCH priced at $30:

```
Eco3x extra earnings - extra energy costs = 0.07 * $31 - 4.79 * $0.14  = +$1.49 vs NoSSD.
Pro4x extra earnings - extra energy costs = 0.10 * $31 - 11.21 * $0.14 = +$1.53 vs NoSSD.
```
These figures highlight Pro4x as a strong contender, offering a 50% higher daily return than NoSSD's C15 ($4.57 profit per day vs $3.04 profit per day), despite a higher energy consumption.

## Get the Most Out of Your GPU with Eco3x and Pro4x

Eco3x and Pro4x plots let you use your hard drive space and GPU power to the fullest. By allocating Eco3x and Pro4x across your HDDs based on how much space you have, you make sure your GPU is always busy, with no downtime.

**For example:**
- you have 335 TiB of storage space and a 3090 GPU, at the current plot filter of 512.

**How to Do It:**
- Use 235 TiB for Eco3x plots. This keeps 50% of your 3090 GPU busy.
- Fill the remaining 100 TiB with Pro4x plots. This uses up the other 50% of your GPU.

**The Outcome:** 
This mix lets you use all of your storage space wisely, filling it with Eco3x and Pro4x plots. You end up with a total plot size of about 1.1 ePiB (3x the size of your Eco3x space and 4x your Pro4x space). Your GPU is always active, making sure you're using your resources efficiently.


# How it works

<p align="center" alt="Diagram of DrPlotter components">
  <img src="images/drplotter-solver-harvester-server-farmer.png" />
</p>


**DrPlotter** is the plotting tool that creates Eco3x and Pro4x plots. Given your farmer and pool public keys, it produces "DrPlots" using write-once technology directly to your HDD, typically in 5-7 minutes per plot. No SSD required.

**DrChia Harvester**, in line with the Chia harvesting protocol, seamlessly integrates with your existing farmer setup. It reads existing chia supported plots and your new DrPlots. DrPlot entries requiring proof solving are sent to the DrServer. Once solved, these proofs are relayed back to the DrChia harvester and then passed onto your Chia farmer.

**DrSolver** leverages your GPU power to solve proofs for your compressed DrPlot entries.

**DrServer** is the central hub on your local network that connects all your harvesters and DrSolver's together. It efficiently manages and distributes all the tasks needed across all your harvesters and DrSolvers, to ensure a smooth running system for relaying proofs back to your farmer. 

For a more complete overview, see the video [How it works](https://www.youtube.com/watch?v=hQTV7foIRHo&t=463s).


# Using DrPlotter

- Make sure you meet the [minimum requirements](#minimum-requirements)
- Download and [install the software](#installation)
- [Start plotting](#plotting) with the DrPlotter tool.
- [Run your DrSolvers](#1-start-drserver)
- [Setup and run your DrChia harvester](#3-setup-your-drchia-harvesters) with your existing chia farmer.
- [Verify your DrPlots are submitting proofs](#verify-your-drplots-are-submitting-proofs)

## Minimum Requirements

DrPlotter Minimum Requirements:
- 24GB nvidia 3090 / A5000 / 4090 per instance.
- 128GB DDR4 RAM per instance.
- motherboard with a PCIE 4.0 x 16 slot
- 64 bit processor (onboard GPU a bonus)
- Ubuntu / Debian based system

DrSolver Minimum Requirements:
- 64MB RAM per instance.
- 24GB nvidia 3090 per instance.
- Ubuntu / Debian based system

DrChia Harvester Minimum Requirements:
- ~4GB RAM for every 1PiB of raw disk space.
- Ubuntu / Debian based system

DrServer Minimum Requirements:
- ~16MB RAM
- Ubuntu / Debian based system

## Installation

Make sure you meet the minimum requirements above. Then, download the latest .deb package from the releases page.

In the command line, run dpkg on your downloaded file, for example:

```
sudo dpkg -i drplotter_1.0.0_amd64.deb 
```

This will install drplotter, drsolver, drserver, and drchia for the harvester in the /usr/bin/ directory.

If at any point you want to remove drplotter, to uninstall run:

```
sudo dpkg -r drplotter
```

## Plotting
> [!NOTE]
> DrPlotter only supports the recommended NFT plots with pooling. This is to ensure you can verify your system is running as expected against proofs submitted to pools. You can still choose to solo pool, but it must be a plot NFT and not the OG format.

To make plots, run:

```
drplotter plot -f <your_farmer_key> -c <your_pool_contract_address> -d /your/hdd/directory/ --compression <eco3+ or pro4+>
```

This will fill the directory with plots. While plotting, you'll see progress and when plots complete, you'll see output like this:

```
Location: 
  Path: /media/nick/E14TB_14/drplots/
  Usage: 1.76 TB used of 12.73 TB total (13%)
  Est. Completion: 467 plots by 2024-01-26 09:25 (2 days, 8 hours)

Plotting File: drplot-pro4p-2024-01-24-00-35-372b8c5b9948587dcf4e6b66565cd382.drplot

  Progress          Time    ETA
  -----------------------------
  100% ##########  06:51      -

  Size       : 22.32 GiB
  Proofs     : 4.184.342.908
  Bits/proof : 45.82 (22.59% of original size)

============ 2024-01-24 00:43:42 ============
```
Bits/proof is the most important metric for compression. A standard k32 chia plot will require around 202.8 bits for each proof it stores. By comparing bits per proof, we can accurately see the compression based on the number of proofs a plot has, and not just it's physical size.

To see more plotting options, run:
```
drplotter -h
```

## Harvesting DrPlots

Note that all the components shown below could be run on the same machine locally, or split across multiple machines or VMs.

### 1. Start DrServer

DrServer is the central hub on your network that connects all the `drchia harvester` and `drsolver` instances. You only need a single instance of DrServer accessible on your network. Run the following command to start:

```
drserver
```

The default port is 8080, and you check it's running by opening a web browser to your local ip and port like so: `http://localhost:8080/`. If you want to run on a specific port, use:

```
drserver --port 8080
```


### 2. Set your DRSERVER_IP_ADDRESS environment variables

Take note of what local ip address your drserver is running on. For every different machine you have running DrPlotter services, set the environment variable `DRSERVER_IP_ADDRESS`.

- **Temporary Setting**: For a temporary setup in a bash shell, use:
     
    ```
    export DRSERVER_IP_ADDRESS='Your_drserver_ip_address:port'
    ```
    Replace 'Your_drserver_ip_address:port' with ip address and port of your machine running drserver. Append the port after a semi-colon to the ip address, e.g. `192.168.0.2:8080` for setting the port to `8080`.

- **Persistent Setup:**

  Edit your `.bashrc` file for a more persistent solution:
  ```
  nano ~/.bashrc
  ```
  Add the following line to the end of the file (with your actual drserver ip address):
  ```
  export DRSERVER_IP_ADDRESS='Your_drserver_ip_address:port'
  ```
  Save and exit, then apply changes with:
  ```
  source ~/.bashrc
  ```

### 3. Setup your DrChia harvesters

-  If you're on a new install, first run:

   ```
   drchia init
   ```
- Then, you need to copy in your ca certificates from your **farmer machine**, these are usually found at `~/.chia/mainnet/config/ssl/ca`. These let the drchia harvester securely talk with your farmer. Initialize these with your harvester:

   ```
   drchia init -c /path/to/your/farmers/ca/certificates
   ``` 

- Edit the chia config file on your harvester. In the example below we use nano:

   ```
   nano ~/.chia/mainnet/config/config.yaml
   ```
 
   Look for **farmer_peers:** under the **harvester:** section, and edit the ip to point to your farmer ip.
    e.g.
    ``` 
    harvester:
      farmer_peers:
        host: <Farmer IP Address>  <--- set to your farmer ip address, e.g. 192.168.1.23
        port: 8447
    ```
   Don't forget to save your changes.

### 4. Connect your DrChia Harvester

To verify that your machine knows the ip address of your `drserver`, you can run:
```
echo $DRSERVER_IP_ADDRESS
```
and check that the output matches the ip address of your `drserver`.


Before you run your harvester, let's change the config so that you can see log outputs to check it's working. Run:

```
drchia configure --log-level INFO
```
And now run:

```
drchia start harvester -r
```

If all is well, you can now check your logs in `~/.chia/mainnet/log/debug.log`

If you see logs similar to this:

```
2024-01-24T01:05:55.731 harvester drplotter               : INFO     Harvesting 1368 drplots with on disk size 36.82 TiB, after decompression 130.58 eTiB, extra rewards 3.55x
```

Then congrats, your drchia harvester has found your drplots and is now ready to harvest.

You'll also notice there are some warning logs, if you haven't yet setup a DrSolver to use the GPU to decmpress all those proofs. That's what we'll do next.


### 5. Connect your DrSolvers

A DrSolver can run on the same system as your `drchia harvester` or the `drserver`, as long as it has it's own dedicated GPU.

Let's first verify that your machine knows the ip address of your `drserver`, you can run:
```
echo $DRSERVER_IP_ADDRESS
```
and check that the output matches the ip address of your drserver.

If that looks good, then running a drsolver is as simple as:
```
drsolver
```

DrSolver will run and connect to the `drserver`. Once connected, it will display your connected harvesters and solvers that are all synced with the `drserver`. Below is an example output:

```
                            DrPlotter Solver v1.0.3

DrPlotter Farm Status
--------------------------------------------------------------------------------
  Status: CONNECTED                                             Uptime: 02:23:30
  DrServer: 192.168.2.44:8080

  Total Harvesters: 2                                           Total Solvers: 1

  Num DrPlots        Raw Size           Effective Size     Extra Rewards
  3911               95.80 TiB          376.20 TiB         +393%


Solver GPU: NVIDIA GeForce RTX 3090
--------------------------------------------------------------------------------
  Fan    Temp   Perf   Pwr:Usage/Cap                                Memory-Usage
  30%    61C    P2     237W / 240W                           23346MiB / 24576MiB

  Status                                            Load  1 min / 5 min / 15 min
  SOLVING                                                    14%  /  34%  /  32%


Commands: [Q]uit
```

While DrSolver is running, monitor the 5-minute and 15-minute "load" indicators to gauge the current capacity usage of your GPU as a percentage of its total capacity. 



## 6. Verify your DrPlots are Submitting Proofs

To check your DrPlots are submitting proofs, it's recommended to join a pool and adjust the difficulty setting of your pool plots to the lowest possible value, such as 1. This approach is beneficial for several reasons:

- **No Performance Impact:** Setting the difficulty to a low level for DrPlots does not affect their performance. Thus, you can monitor their operation without any compromise on efficiency.

- **Effective Monitoring:** By using a low difficulty setting, it's easier to track and ensure that you are accruing the expected number of pool points. This setting enhances the visibility of the functioning of your drplots in the pool.

- **Troubleshooting:** In case you notice discrepancies in the expected pool points, the low difficulty setting can help in detecting any problems early.

As a result, you can effectively monitor and ensure that your drplots are being farmed correctly and submitting proofs to the pool as anticipated.


# Securing your DrServer Remotely

To take advantage of the remote features of DrServer, for instance, so that you can run your DrSolver's also on remote rentable instances, you can secure your server to ensure only your DrChia harvesters and DrSolvers are able to connect using a token system.

First, on your DrServer generate a token:

```
drserver --generate-token
```

and you will see output similar to this:

```
Generated validation token: SLJNCYo0dsEfpl8nRcuV5qDPWD3sYwAoWJMU9ghzpSEafqGG
```

Next, run your DrServer with that token:

```
drserver --token SLJNCYo0dsEfpl8nRcuV5qDPWD3sYwAoWJMU9ghzpSEafqGG    <-- use your own token here
```

Alternatively, you can set the `DRPLOTTER_CLIENT_TOKEN` as an environment variable, and `drserver` will use that if you don't specify the token directly. 

If no token is passed in the command line or `DRPLOTTER_CLIENT_TOKEN` is not set in your environment for DrServer, then anyone can connect to your DrServer without authenticating.

## Authenticate your DrSolvers

If your DrServer requires a token, use that same token when starting your DrSolver, like so:

```
drsolver --drserver-ip mydrserver.com:8080 --token [YOUR TOKEN HERE WITHOUT BRACKETS]
```

Alternatively, if you don't specify a token you can set DRPLOTTER_CLIENT_TOKEN in your environment variables as well.

## Authenticate your DrChia Harvesters

To connect your harvester with your DrServer that requires a token to authenticate, make sure to set the `DRPLOTTER_CLIENT_TOKEN` and the `DRSERVER_IP_ADDRESS` in the environment variables.

