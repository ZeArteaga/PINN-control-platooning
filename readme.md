# PINN Control Platooning

## Installation of HSL Linear solvers

### Linux

1.  Follow the instructions at [https://github.com/coin-or-tools/ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL) to build and install the HSL libraries.
2.  After installation, create a symbolic link for `libhsl.so`. You will likely need to create a symbolic link from `libcoinhsl.so` to `libhsl.so`. The default installation after `make install` is `/usr/local/lib/`.
    ```bash
    sudo ln -s /usr/local/lib/libcoinhsl.so /usr/local/lib/libhsl.so
    ```
3. Reopen a terminal
4. (Optional) add to `~/.bashrc` if the installation directory was not the default one:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:(custom_path/lib)
```

### Windows

(Instructions to be added later)
