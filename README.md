# DJ Mix Ground Truth Extractor

**!!! The evaluation dataset is available on the following link: <https://1drv.ms/f/s!Al5-09HWyOJ3wlvkkd0ZGte119ov> !!!**

This software allows for reverse engineering actions performed by a DJ in a mix. This software is complementary to my dissertation.

## Getting Started

### Prerequisites

[Panako](http://panako.be/) is used for fingerprinting and must be installed beforehand.

### Installing

```
pip install -r requirements.txt
```

### Usage

```
$ python main.py -h
usage: main.py [-h] <mix_path> <source_track_directory_path> <output_file>

Analyse a DJ mix and reverse engineer it

positional arguments:
  <mix_path>            Path to a mix which needs to be analysed
  <source_track_directory_path>
                        Path to all source tracks
  <output_file>         Path to the output JSON file

optional arguments:
  -h, --help            show this help message and exit
```

## Authors

* **Lorin Werthen-Brabants** - *Initial work*

See also the list of [contributors](https://github.com/werthen/dj-mix-ground-truth-extractor/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
