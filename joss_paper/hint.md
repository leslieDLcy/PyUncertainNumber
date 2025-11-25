# Tutorial on rendering the PDF

step 1: Open Docker on the laptop

step 2: Go to terminal and type below:

```shell
docker run --rm -it \
    -v $PWD:/data \
    -u $(id -u):$(id -g) \
    openjournals/inara \
    -o pdf,crossref \
    joss_paper/paper.md
```