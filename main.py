#!/usr/bin/env python
"""
Main program
"""

import json


class Main:
    def run(self):
        self.train_data     = self.read_file("train.json")
        self.classify_data  = self.read_file("classify.json")
        self.inside         = self.train_data['inside']
        self.outside        = self.train_data['outside']
        print('To train:')
        print('Inside:', self.inside)
        print('Outside:', self.outside)
        print('To classify:')
        print(self.classify_data)
        pass

    def read_file(self, name):
        with open(name) as f:
            return json.load(f)


if __name__ == '__main__':
    m = Main()
    m.run()
