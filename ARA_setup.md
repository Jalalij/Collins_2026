# ARA setup instruction to run LEO experiments via Starlink or OneWeb User Terminals (UTs)
Author: Jie Wang (jiew@iastate.edu)


## ARA Account Login 

- Please check out the [ARA user manual](https://arawireless.readthedocs.io/en/latest/getting_started/ara_portal_extras.html) for ARA account details and SSH public key uploading

## Remote Access to the LEO Containers

- Remote access to the respective container
    ```bash
    ssh -p 22 username@ip_of_container
    ```

    (Optional) Install tmux, the terminal multiplexer, on the ARA container, for persistent session management

    #### Step 1: Install Tmux on the ARA container by
    ```bash
    apt install tmux -y
    ```

    #### Step 2: Exit and create a tmux session to run in the background

    ```bash
    ssh -p 22 -t username@ip_of_container 'tmux new-session -A -s session_name  &&  exec $SHELL'
    ```

    Note: For OneWeb and Starlink containers created for internal use, please check the [SatCom Transition](https://iowastate.sharepoint.com/:w:/r/sites/Collins-ISUWiCI/Shared%20Documents/2025%20-%202026%20Satcom%20(Year%202)/JW%20Transition/Jie%20Wang%20SatCom%20Transition%20.docx?d=wca722899409a4c8a85c7084653d109c0&csf=1&web=1&e=H3RqCI) doc or Sarath Babu for details.

## (Optional) Remote Access to the Wilson-Hall Host Computer

- Verify with Sarath Babu for the need to access the Wilson-Hall host computer where Starlink and OneWeb UTs are connected

- If yes, generate *RSA SSH key* and share the public key with Sarath Babu

- Access the host computer via

    ```bash
    ssh -p 22 username@host_computer_ip
    ```
    For username, host_computer_ip, and related info, please check the [SatCom Transition](https://iowastate.sharepoint.com/:w:/r/sites/Collins-ISUWiCI/Shared%20Documents/2025%20-%202026%20Satcom%20(Year%202)/JW%20Transition/Jie%20Wang%20SatCom%20Transition%20.docx?d=wca722899409a4c8a85c7084653d109c0&csf=1&web=1&e=H3RqCI) for details.







