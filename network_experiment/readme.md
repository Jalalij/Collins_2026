# Data Collection Instructions for Latency, Throughput, Packet Loss, and LEO Traces 


1. Install required packages:
    ```bash
    pip install pandas requests
    ```

2. Update configuration:
Edit `config.py` with your server IP, Hughes API details and etc.

3. Run one experiment:
    ```bash
    python main_controller.py
    ```

4. Automate the experiment and collect data at certain times:
    - Use tmux to keep the script running even when SSH disconnects
    ```bash
    ssh -i id_rsa -Y -p 22 -t username@ip_address 'tmux new-session -A -s session_name &&  exec $SHELL'
    
    ```
    - Run the Python script:
    ```bash
    python3 controller_schedule.py
    ```

5. Download OneWeb LEO trace data:
    - Preliminaries:
        - 192.168.100.1: the OneWeb user terminal (UT) management interface through which we can obtain LEO tracking logs, e.g., SINR and Satellite IDs
        - 192.168.100.1 is only reachable from the OneWeb router 192.168.42.1
        - Both networks 192.168.42.0/24 and 192.168.100.0/24 are not reachable since they are local networks from the host computer
        - 192.168.42.0/24 can be directly reachable from the host computer 
        - The host computer is reachable from the ISU network or VPN

    - Solution
        - Step 1: Add an additional route to 192.168.100.1 via 192.168.42.1
            ```bash
            sudo ip route add 192.168.100.1 via 192.168.42.1 dev eno12429
            ```
        - Step 2: Enable port forwarding at the host computer so that whatever http traffic to the host computer will be forwarded to 192.168.100.1
            ```bash
            sudo socat tcp-listen:80,reuseaddr,fork tcp:192.168.100.1:80
            ```
    - Use: 
        - Step 1: Make sure the local computer is part of the ISU network or has ISU VPN (vpn.iastate.edu) enabled
        - Step 2: Visiting host_computer_ip at your local browser and go to host_computer_ip/antenna/aimlogs to download the OneWeb LEO trace data

    - Data Availability: ~7 days of LEO trace data will be available at the interface to download and each csv file has ~1 hour window of data

6. Access Starlink LEO trace data:
    - Git repo being used for Starlink exporter is https://github.com/joshuasing/starlink_exporter

    - Steps
        1. Check go installation
            ```bash
            go version
            ```
            Verification: If it returns `go version go1.26...`, proceed. If not, install it via
            ```bash
            apt update && apt install golang-go
            ```
            (Opitional) If a specific go version is required by the Starlink exporter Git repo, we can
            ```bash
            # unstall other version if any
            apt remove golang-go

            # install the spefic version
            wget https://go.dev/dl/go1.26.0.linux-amd64.tar.gz
            tar -C /usr/local -xzf go1.26.0.linux-amd64.tar.gz

            # check where it is stored
            which go # should retun /usr/local/go/bin/go

            # update the path
            # Add to your current session
            export PATH=$PATH:/usr/local/go/bin

            # Make it permanent for future Tmux sessions
            echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
            source ~/.bashrc
            ```

        2. Route configuration
            - Preliminary: the Starlink interface and IP address are `eth1: 10.3.X.X`. More details can be found at [Measuring OneWeb and Starlink Satellite Link Characteristics](https://arawireless.readthedocs.io/en/latest/ara_experiments/arahaul_experiments/satellite_basic.html#measuring-oneweb-and-starlink-satellite-link-characteristics).

            - Set route to the destination through the Starlink gateway
                ```bash
                route add 192.168.100.1 gw 10.3.0.1 dev eth1

                # verification
                ping -c 2 192.168.100.1 # should return 0% packet loss
                
                ```
        
        3. Clone the repository and rename `starlink_exporter_joshua` to prevent future naming confusion
            ```bash
            git clone https://github.com/joshuasing/starlink_exporter.git starlink_exporter_joshua

            cd starlink_exporter_joshua
            ```

        4. Build the Starlink exporter 
            ```bash
            cd starlink_exporter_joshua

            # Tidy and Build
            go mod tidy
            go build -o starlink_exporter ./cmd/starlink_exporter/
            ```

        5. Start and verify the exporter
            ```bash
            # star the exporter in a tmux session
            ./starlink_exporter -dish "192.168.100.1:9200" -listen ":9451"
            ```
            Example output:
            ```ini
            2026/03/06 03:38:41 INFO Starting Starlink exporter version="(version=, branch=, revision=f8020159250aaf0eb8d00056ae9ebff53bcd52ea-modified)"
            2026/03/06 03:38:41 INFO Build context build_context="(go=go1.26.0, platform=linux/amd64, user=, date=, tags=unknown)"
            2026/03/06 03:38:41 INFO Connecting to Starlink Dishy address=192.168.100.1:9200
            2026/03/06 03:38:41 INFO HTTP server listening address=:9451
            ```
            
            Open a new tmux session at the container and run
            ```bash
            curl -s localhost:9451/metrics | grep starlink_dish_ # can change starlink_dish_ to more spefic/general keywords, e.g., starlink_dish_snr
            ```
            Exported metrics example
            ```ini
            # HELP starlink_dish_snr_above_noise_floor Whether Starlink dish signal-to-noise ratio is above noise floor
            # TYPE starlink_dish_snr_above_noise_floor gauge
            starlink_dish_snr_above_noise_floor 1
            # HELP starlink_dish_snr_persistently_low Whether Starlink dish signal-to-noise ratio is persistently low
            # TYPE starlink_dish_snr_persistently_low gauge
            starlink_dish_snr_persistently_low 0
            # HELP starlink_dish_software_update_reboot_ready Whether the Starlink dish is ready to reboot to apply a software update
            # TYPE starlink_dish_software_update_reboot_ready gauge
            starlink_dish_software_update_reboot_ready 0
            # HELP starlink_dish_software_update_state Starlink dish update state
            # TYPE starlink_dish_software_update_state gauge
            starlink_dish_software_update_state 1
            # HELP starlink_dish_tilt_angle_deg Starlink dish tilt angle degrees
            # TYPE starlink_dish_tilt_angle_deg gauge
            starlink_dish_tilt_angle_deg 40.51743698120117
            # HELP starlink_dish_up Whether scraping metrics from the Starlink dish was successful
            # TYPE starlink_dish_up gauge
            starlink_dish_up 1
            # HELP starlink_dish_uplink_throughput_bps Starlink dish uplink throughput in bits/sec
            # TYPE starlink_dish_uplink_throughput_bps gauge
            starlink_dish_uplink_throughput_bps 10536.091796875
            ```

            Note: The repo above includes `azimuth_deg` and `elevation_deg` metrics but not `SINR` and `satellite ID` (for handover). Please take a look at the following git repos and beyond for modification
            - https://github.com/danopstech/starlink_exporter: it includes `starlink_dish_snr`
            - https://github.com/clarkzjw/starlink_exporter

            It will also help to add the person running experiments, if part of the ARA team, to http://starlink.com/account/settings as an admin, so he/she can see `Dubug data` on the Starlink mobile app and verify what data can be available. 

        6. Save the data buffers as raw traces

            - Modify and move the data collection script [`collect_leo_traces.sh`](/network_experiment/collect_leo_traces.sh) to the container
                ```bash
                # move the script to the starlink_exporter_joshua folder at the container
                scp path_to_file username@container_ip:path_to_the_git_repo
                ```

            - Step-by-step executation
                ```bash
                # make it executable
                chmod +x collect_leo_traces.sh

                # run it at a tmux session
                ./collect_leo_traces.sh

                # open another tmux session and verify if the buffer is being saved
                tail -f starlink_phys_traces.csv
                ```



7. Structure of Files 

```pgsql
    network_experiment/
    ├── controller_schedule.py      /* Automated time-based sampling */
    ├── main_controller.py          /* Main experiment coordinator */
    ├── ping_collector.py           /* Ping data collection */
    ├── iperf3_collector.py         /* iperf3 data collection */
    ├── config.py                   /* Shared configuration */
    └── data/                       /* Output directory */
        └── ping/
        └── iperf3/
        └── sattrace/               /* self-created folder to save LEO trace data */
        └── experiment_metadata_{experiment_time}.json # Metadata for each experiment */
```