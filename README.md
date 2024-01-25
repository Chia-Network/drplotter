# Meet DrPlotter: Your Go-To for Chia Farming

*DrPlotter* is an advanced, energy-efficient GPU plotter, solver, and harvester designed for the Chia Blockchain, with break-through rewards and efficiency optimized for consumer-grade hardware. Offering Eco3x for enhanced energy efficiency and Pro4x for high-density plotting, boost rewards by over 400% compared to standard Chia plots to achieve higher ROI than any other solution.

- **Up to +413% Rewards**: Two compression modes offer a balance between optimal efficiency and optimal cost per eTB for maximum ROI.
- **Enhanced Security**: Relies solely on public farmer keys, letting you safeguard your private keys.
- **Effortless Remote GPU Use**: Enjoy a streamlined process for submitting proofs and the flexibility to use your GPU remotely with ease.
- **Supporting the Chia Ecosystem**: By seamlessly integrating with the official Chia Farmers, DrPlotter plays a part in keeping the Chia network strong and decentralized.

Below is an overview of Eco3x and Pro4x plots created by DrPlotter.
| Compression Mode                        | Eco3x                        | Pro4x                        |
|-----------------------------------------|:----------------------------:|:----------------------------:|
| Plot Size                               | 35.2 GiB                     | 24.2 GiB                     |
| Reward vs standard plots                | 297%                         | 413%                         |
| Capacity per RTX 3090 @ 260W<sup>1</sup>| ~470 TiB / 1385 eTiB         | ~200 TiB / 820 eTiB          |
| Capacity per RTX 4090 @ 330W<sup>1</sup>| ~950 TiB / 2800 eTiB         | ~380 TiB / 1560 eTiB         |
| Lower W / eTB vs standard plots         | ✔✔                           | ✔<sup>2</sup>                |
| Lower $ / eTB vs standard plots         | ✔                            | ✔✔                           |
| Plot time                               |        6-7 minutes           | 6-7 minutes                  |
| Plot direct to HDD<sup>3</sup>          | ✔                            | ✔                            |

<sup>1: capacities with current chia plot filter of 512. For a plot filter of 256 halve these values.</sup>
<br><sup>2: at a plot filter of 256 (expected June 2024), currently only the 4090 has less W/eTB than standard plots on pro4x.</sup>
<br><sup>3: **No SSD required.** While you can use an SSD as an intermediary drive under most cases it will not improve plot time.</sup>

## The Eco3x Advantage

Eco3x compression offers an energy-efficient solution for managing chia farms, especially with the upcoming chia plot filter of 256. The table below illustrates Eco3x's efficiency compared to traditional uncompressed plots for a 1 ePiB farm (around 10,000 plots). Eco3x significantly reduces hard drive space and energy consumption, leading to considerable savings in both hardware investment and operational costs. This makes it a highly effective option for Chia users, even those facing high energy expenses, to achieve exceptional ROI.

| Setup        | Physical TiB | # plots      | plot filter | gpu capacity     | daily kW usage per ePiB | Hardware Cost per ePiB<sup>1</sup> |
|--------------|-------------:|-------------:|:----:|:----------------:|:----------------:|:----------------------:|
| Uncompressed | 1,000         | 10,000        | - | -                | 14.40            | ~$16,000       |
| Eco3x w/3090 @ 260W | 334 | 10,000 | 512 | 71% | 9.24<br><sub>+36% efficiency</sub> | ~$6,144<br><sub>38% of cost</sub>|
| Eco3x w/2x3090 @ 260W | 334 | 10,000 | 256<br><sub>(coming June 2024)</sub> | 71% | 13.68<br><sub>+5% efficiency</sub>| ~$6,944<br><sub>43% of cost</sub> |
| Eco3x w/4090 @ 330W | 334 | 10,000 | 256<br><sub>(coming June 2024)</sub> | 70% | 10.38<br><sub>+28% efficiency</sub> | ~$7,144<br><sub>44% of cost</sub> |

<sub>1: The "Hardware Cost per ePiB" represents the actual costs of purchasing the necessary hardware for disks and GPUs. This is based on an estimated $16 per installed TB and $800 for a used 3090 GPU</sub>

## The Pro4x Advantage

Pro4x compression significantly reduces hard drive needs for your plots to less than a quarter compared to uncompressed plots, marking a major leap in cost and space savings. By matching your hard drive capacity with your GPU's power, Pro4x achieves unmatched ROI. The table shows that with 4090 GPUs at full capacity, even with the challenging 256 plot filter, you can efficiently run a 1.5 ePiB farm at just 39% of the normal cost. Notably, Pro4x enables over 1 ePiB of plots in a consumer PC case using only 14 hard drives, compared to the 56 drives a large server setup would need, enhancing space utilization, cost-effectiveness, and management simplicity.

| Setup        | Physical TiB | # plots      | plot filter | gpu capacity     | daily kW usage per ePiB | Hardware Cost per ePiB<sup>1</sup> |
|--------------|-------------:|-------------:|-----:|-----------------:|-----------------:|-----------------------:|
| Uncompressed | 1,000         | 10,000        | - | -                | 14.40            | ~$16000       |
| Pro4x w/4090 @ 330W | 243 | 10,000 | 512 | 64% | 8.56<br><sub>+41% efficiency</sub>| ~$5688<br><sub>36% of cost</sub> |
| Pro4x w/2x4090 @ 330W | 243 | 10,000 | 256 <br><sub>(coming June 2024)</sub> | 64% | 13.63<br><sub>+5% efficiency</sub> | ~$7488<br><sub>47% of cost</sub>|
| Pro4x w/2x4090 @ 330W | 380 | 15,600 | 256 <br><sub>(coming June 2024)</sub> | 100% | 13.63<br><sub>+5% efficiency</sub> | ~$6205<br><sub>39% of cost</sub> |

<sub>1: The "Hardware Cost per ePiB" represents the actual costs of purchasing the necessary hardware for disks and GPUs. This is based on an estimated $16 per installed TB and $1800 for a new 4090 GPU</sub>

## Get the Most Out of Your GPU with Eco3x and Pro4x

Eco3x and Pro4x plots let you use your hard drive space and GPU power to the fullest. By allocating Eco3x and Pro4x across your HDDs based on how much space you have, you make sure your GPU is always busy, with no downtime.

**For example:**
- you have 335 TiB of storage space and a 3090 GPU, at the current plot filter of 512.

**How to Do It:**
- Use 235 TiB for Eco3x plots. This keeps 50% of your 3090 GPU busy.
- Fill the remaining 100 TiB with Pro4x plots. This uses up the other 50% of your GPU.

**The Outcome:** This mix lets you use all of your storage space wisely, filling it with Eco3x and Pro4x plots. You end up with a total plot size of about 1.1 ePiB (3x the size of your Eco3x space and 4x your Pro4x space). Your GPU is always active, making sure you're using your resources efficiently.


## About the Developer and DrPlotter's Fee Structure

DrPlotter has been a labor of love and dedication that has captivated my attention for over two years. What began as a passion project has transformed into a committed effort to advance the field of Chia plot compression and support the strengths of the Chia blockchain.

To continue this journey, I've taken a completely new approach towards funding my development with DrPlotter, stepping away from the usual way of charging randomized or possibly changing fees on farming revenue. Instead, a small part of your GPU and disk space goes towards supporting and improving DrPlotter. I believe in transparency and honesty in all aspects of DrPlotter. That's why all the performance stats I present already include this fee component – ensuring that there are no hidden costs. What you see is precisely what you get. This fee structure is fixed and immutable, guaranteeing that the rate you commit to now will remain the same in the future. This consistency offers you ease in planning and peace of mind.

The specifics of this fee structure are kept confidential to safeguard the unique technology behind DrPlotter. However, my goal is always to deliver exceptional value and results. With DrPlotter, I aim to provide a reliable and cutting-edge tool that consistently meets your needs and stays at the forefront of Chia farming technology.


# How it works

[Diagram of DrPlotter, DrSolver, DrHarvester]

**DrPlotter** is the plotting tool that creates Eco3x and Pro4x plots. Given your farmer and pool public keys, it produces "DrPlots" using write-once technology directly to your HDD, typically in 6-7 minutes per plot. No SSD required.

**DrChia harvester**, in line with the Chia harvesting protocol, seamlessly integrates with your existing farmer setup. It reads existing chia supported plots and your new DrPlots. DrPlot entries requiring proof solving are sent to the Solver Server. Once solved, these proofs are relayed back to the DrChia harvester and then passed onto your Chia farmer, ensuring smooth and consistent operation.

**DrSolver** leverages your GPU power to solve proofs for your plot entries. Using a unique token system, it can be allowing flexible deployment of DrSolvers in various locations without compromising efficiency or security.

**Solver Server** is vital in enhancing computational efficiency and consistently reducing proof times. It alleviates bottlenecks and manages load during peak periods at signage points for your DrPlots by proportionally allocating compute resources according to DrPlot count, ensuring efficient and equitable proof resolution across the network.


# Using DrPlotter

- Make sure you meet the [minimum requirements](https://github.com/drnick23/drplotterv3/blob/main/README.md#minimum-requirements)
- Download and [install the software](https://github.com/drnick23/drplotterv3/blob/main/README.md#installation)
- [Start plotting](https://github.com/drnick23/drplotterv3/blob/main/README.md#plotting) with the DrPlotter tool.
- [Set your client token](https://github.com/drnick23/drplotterv3/blob/main/README.md#setting-up-your-drplotter_client_token)
- [Run your DrSolvers](https://github.com/drnick23/drplotterv3/blob/main/README.md#run-your-drsolver)
- [Setup and run your DrChia harvester](https://github.com/drnick23/drplotterv3/blob/main/README.md#setup-and-run-your-drchia-harvester) with your existing chia farmer.

## Minimum Requirements
DrPlotter Minimum Requirements:
- 24GB nvidia 3090 / A5000 / 4090
- 128GB DDR4 RAM
- motherboard with a PCIE 4.0 x 16 slot
- 64 bit processor (onboard GPU a bonus)
- Ubuntu / Debian based system

DrSolver Minimum Requirements:
- 24GB nvidia 3090.
- Ubuntu / Debian based system

DrHarvester Minimum Requirements:
- ~4GB RAM for every 1PiB of raw disk space.
- Ubuntu / Debian based system

## Installation

Make sure you meet the minimum requirements above. Then, download the latest .deb package from the releases page.

In the command line, run dpkg on your downloaded file, for example:

```
sudo dpkg -i drplotter_0.9.0_amd64.deb 
```

This will install drplotter, drsolver, and drchia for the harvester in the /usr/bin/ directory.

If at any point you want to remove drplotter, to uninstall run:

```
sudo dpkg -r drplotter
```

## Plotting
> [!NOTE]
> DrPlotter only supports the recommended NFT plots with pooling. This is to ensure you can verify your system is running as expected against proofs submitted to pools. You can still choose to solo pool, but it must be a plot NFT and not the OG format.

To make plots, run:

```
drplotter plot -f <your_farmer_key> -c <your_pool_contract_address> -d /your/hdd/directory/ --compression <eco3x or pro4x>
```

This will fill the directory with plots. While plotting, you'll see progress and when plots complete, you'll see output like this:

```
Location: 
  Path: /media/nick/E14TB_14/drplots/
  Usage: 1.76 TB used of 12.73 TB total (13%)
  Est. Completion: 467 plots by 2024-01-26 09:25 (2 days, 8 hours)

Plotting File: drplot-pro4x-2024-01-24-00-35-372b8c5b9948587dcf4e6b66565cd382.drplot

  Progress          Time    ETA
  -----------------------------
  100% ##########  06:51      -

  Size       : 24.01 GiB                   1
  Proofs     : 4.180.621.852
  Bits/proof : 49.33 (24.33% of original size)

============ 2024-01-24 00:43:42 ============
```
Bits/proof is the most imporant metric for compression. A standard k32 chia plot will require around 202.8 bits for each proof it stores. By comparing bits per proof, we can accurately see the compression based on the number of proofs a plot has, and not just it's physical size.

To see more plotting options, run:
```
drplotter -h
```

## Setting up your DRPLOTTER_CLIENT_TOKEN

DrPlotter requires a unique client token for authentication. This token links your drsolvers and harvesters. **Use the same token** across all your machines running drchia harvesters and drsolvers.

### 1. Generate your token

Run the following command to generate a new client token:
```
drsolver --generate-token
```
This command creates a new authentication token. You'll see output similar to this (note that your token will be different):

```
Generated client token: kWq9NXkHQ75zGhebkJzriknBs0IOnDux5kIqOd0aJioM6HSR
```
 
### 2. Set your DRPLOTTER_CLIENT_TOKEN environment variable
- **Temporary Setting**: For a temporary setup in a bash shell, use:
     
    ```
    export DRPLOTTER_CLIENT_TOKEN='[Your Unique Token]'
    ```
    Replace '[Your Unique Token]' with the token generated in the previous step.
- **Persistent Setup:**

  Edit your `.bashrc` file for a more persistent solution:
  ```
  nano ~/.bashrc
  ```
  Add the following line to the end of the file (with your actual token):
  ```
  export DRPLOTTER_CLIENT_TOKEN='[Your Unique Token]'
  ```
  Save and exit, then apply changes with:
  ```
  source ~/.bashrc
  ```

### 3. Verify your token is set
To verify that your token is set correctly, you can run:
```
echo $DRPLOTTER_CLIENT_TOKEN
```
and check that the output matches your token.

> [!CAUTION]
> Keep your token secure and do not share it in public forums


## Run your drsolver

Once your DRPLOTTER_CLIENT_TOKEN is set in your environment (see [previous section](https://github.com/drnick23/drplotterv3/blob/main/README.md#setting-up-your-drplotter_client_token)), run:
```
drsolver
```
DrSolver will run and connect to Solver Server. Once connected, it will display your connected harvesters and solvers that are linked using the same client token. An exmaple output is below:

[ example solver output ]

While DrSolver is running, monitor the "buffer" increase, which indicates the proof-solving process being smoothed out by your GPU. When your GPU is underutilized, it lends assistance to other network GPUs. Conversely, if your GPU is overwhelmed, the buffer draws on network resources to aid in processing your plot proofs. A failure to fill the buffer suggests your GPU has reached its maximum capacity, risking dropped proofs. Monitor the 5-minute and 15-minute "load" indicators to gauge the current capacity usage of your GPU as a percentage of its total capacity. 

## Setup and run your DrChia harvester
 
On your harvester system, set the DRPLOTTER_CLIENT_TOKEN environment variable to the one you [generated with your DrSolver](https://github.com/drnick23/drplotterv3/blob/main/README.md#setting-up-your-drplotter_client_token).

### System with existing chia harvester
 
If you already have a chia setup for your system, you can simply run:

```
drchia start harvester -r
```

Make sure to include the -r to stop any previous harvesters and replace them with the drchia harvester.

Add any new plot directories you've plotted, as you would with chia's software, e.g.:

```
drchia add plots -d /your/plots/directory
```

### New system as remote harvester

If you don't have any harvester setup on your machine, you can follow the [chia official guide to setting up a remote harvester](https://docs.chia.net/farming-on-many-machines/). You can either setup with the official chia harvester, or use drchia for setup similar to using chia:
 
-  First, run:

   ```
   drchia init
   ```

- Then, you need to copy in your ca certificates from your **farmer machine**, these are usually found at `~/.chia/mainnet/config/ssl/ca`. Initialize these with your harvester:

   ```
   drchia init -c /path/to/your/farmers/ca/certificates
   ```

- Edit the chia config file on your harvester.

   ```
   ~/.chia/mainnet/config/config.yaml
   ```
 
   Look for **farmer_peer:** under the **harvester:** section, and edit the ip to point to your farmer ip.
    e.g.
    ``` 
    harvester:
      farmer_peer:
        host: <Farmer IP Address>  <--- set to your farmer ip address, e.g. 192.168.1.23
        port: 8447
    ```
   Don't forget to save your changes.
 
Before you run your harvester, let's change the config so that you can see log outputs to check it's working. Run:

```
drchia configure --log-level INFO
```
And now run:

```
drchia start harvester -r
```

If all is well, you can now check your logs in ~/.chia/mainnet/log/debug.log

If you see logs similar to this:

```
2024-01-24T01:05:55.731 harvester drplotter               : INFO     Harvesting 1368 drplots with on disk size 36.82 TiB, after decompression 130.58 eTiB, extra rewards 3.55x
```

Then congrats, your drchia harvester has found your plots and should be harvesting your drplots.

> [!NOTE]
> If you have not yet connected a drsolver with your same DRPLOTTER_CLIENT_TOKEN, you will see a warning or error message in your harvester logs. Once your drsolver starts running, the harvester will then connect to the server and start sending plots to your solvers.


- **`eco3x` compression**: Ideal for users seeking a straightforward, energy-efficient approach. This produces `drplots` that require less energy for proof solving, maintaining effectiveness even for more demanding future plot filter levels.
- **`pro4x` compression**: Suited for advanced users focusing on optimizing ROI. Consider this mode for setups with lower energy costs or high-efficiency operations, balancing operational and hardware costs.

## FAQ:

Can I join or change pools?
Yes. You farm and choose to self pool or join any pool you want, as intended by Chia's original design and contributing to raising the Nakomoto consensus, a leading strength indicator of the Chia Blockchain.

Do I need an SSD?
While you can use an SSD as an intermediary drive, under most circumstances it will not improve any performance. As long as your HDD or method of transfer to the final destination HDD can support write speeds of 70MB/s you should notice no write delays.

What's the ideal setup?
Ideal setups range from scaling up using self-contained PC builds for both plotting and solving, to using a centralized harvester and distributing your GPUs across other machines. For users starting out, I would recommend a single PC build for plotting, solving, and harvesting.

PC Build: gaming motherboard with PCIE 4.0 x 16 and 128GB DDR 4 RAM, the lowest energy cpu you can find, nvidia 3090, 750W power supply, large PC case to house 14 HDD's, of 18TB each.

Do I need to setup my plot difficulty in my pool to a certain threshold?
No. In fact, it is recommended to set your pool difficulty to 1 or the lowest setting you can, so you can monitor your plots more effectively.

Can I use my gpu when it's not solving 100% of the time?
Currently, your gpu will be dedicated to solving plots and will consume a significant amount of RAM. You could still use your computer for normal browsing or non-GPU intensive tasks that don't require much memory. In the future, it's possible DrPlotter could offer revenue opportunities for unused GPU cycles.

