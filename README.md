# Satllite Link Dynamics Modeling and Prediction via ARA Wireless Living Lab
Author: Jie Wang (jiew@iastate.edu)

## 1. Project Overview

This project focuses on modeling satellite link dynamics due to orbital mobility, beam switching, atmospheric attenuation, to provide predictive capability of satellite network preformance metrics, i.e., round-trip time (RTT), packet loss rate, and throughput.

The project includes:
- ARA access steps
- Instructions for the end-to-end satellite tunnel setup 
- Scripts for automated data collection and data loading
- Available datasets for satellite-related research
- Algorithms for satellite dynamics modeling and prediction

## 2. ARA Access 

See [`ARA_setup.md`](ARA_setup.md) for ARA LEO container and/or ARA host computer access instructions.

## 3. Tunnel Setup

See [`VPN_tunnel_setup.md`](VPN_tunnel_setup.md) for a full guide to set up the tunnel. It includes:

- What software to install
- Google Cloud firewall configuration
- WireGuard VPN configuration
- Verification steps and common troubleshooting tips

## 4. Automated Data Collection

Check [`network_experiment/readme.md`](network_experiment/readme.md) for automated ping/iperf data collection details, manual LEO_trace data downloading, and dataset structure.

## 5. Dataset, Data Loading/Sync, and LEO Temporal Dynamics Modeling 

Check [`data_analysis/analysis_modeling.md`](data_analysis/analysis_modeling.md) for (1) available datasets, (2) data loading and time sync, and (3) temporal dynamics modeling scripts for the file stucture, model training, and model use.


## 6. Acknowledgments

We use [`the ARA Wireless Living Lab`](https://arawireless.org/) infrastructure and Google Cloud resources for experimentation.
