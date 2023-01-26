# UIforML
>Welcome to our Gradio Template repo. Prounounced 'gray-dee-oh' ;-].
## To run a Gradio App for development
>This method includes hotreload, courtesy of Gradio.  All you have to do is run `gradio app.py` from the app directory.
### Quick Note:
>Ensure you update the Dockerfile when you make environment changes (`pip installs`, `apt installs`, etc.) and **always** build and push the new Docker image when you have made any code changes! 

>Otherwise, production may look very different than your development instance!
### Hook into a Determined shell with VSCode
> Create a `determined_config.yaml` in your homedir on your local machine with the following:
```yaml
bind_mounts:
- container_path: /run/determined/workdir/shared_fs
    host_path: /mnt/mapr_nfs/<path_to_det_share>/determined/det_share
    propagation: rprivate
    read_only: false
- container_path: /determined_shared_fs
    host_path: /mnt/mapr_nfs/<path_to_det_checkpoints>/determined/det_checkpoints
    propagation: rprivate
    read_only: false
- container_path: /mnt/mapr_nfs
    host_path: /mnt/mapr_nfs
    propagation: rprivate
    read_only: false
debug: false
description: my-awesome-nb
entrypoint: null
environment:
add_capabilities: null
drop_capabilities: null
environment_variables: {}
force_pull_image: false
image:
    cpu: determinedai/environments:py-3.8-pytorch-1.10-tf-2.8-cpu-9119094
    cuda: <dockerhub_image_username>/<gradio_image>:<version>
pod_spec: null
ports: null
slurm: null
idle_timeout: null
notebook_idle_type: kernels_or_terminals
resources:
agent_label: ''
devices: null
resource_pool: kubernetes
slots: 1
weight: 1
work_dir: null
```
>You'll want to make sure you have the mapping to det master in your local dev environment
    ```bash
    echo $DET_MASTER
    ```

>if nothing shows up...
`export DET_MASTER=<ipaddrOfDet:port>`
better yet... add it to your `~/.bashrc` and `source ~/.bashrc`

>To test that you're really talking to `DET_MASTER` run `det shell list` from your terminal

>To start a shell with the config/docker image loaded from YOUR terminal run
    ```bash
    det shell start --show-ssh-command --config-file ~/determined_config.yaml
    ```

>Then add the ssh host to your vscode instance [Det/VSCode docs linked here](https://docs.determined.ai/latest/interfaces/ide-integration.html#visual-studio-code)

>Once you are hooked into your shell, `cd` to whichever Gradio app directory, then run `gradio app.py` in your VSCode terminal.


## To run a Gradio (prounounced gray-dee-oh) app for production

> Pull down the Docker image and run the following
```bash
docker run -d -p 8989:8989 <dockerhub_image_username>/<project>:<version> python app.py
```