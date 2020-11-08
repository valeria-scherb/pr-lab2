#!/usr/bin/env python
"""
Main program
"""

import json
import numpy as np, numpy.linalg as la


class Main:
    def run(self):
        train_data     = self.read_file("train.json")
        classify_data  = self.read_file("classify.json")
        inside         = train_data['inside']
        outside        = train_data['outside']
        print('To train:', len(inside), 'inside,', len(outside), 'outside')
        print('To classify:', len(classify_data), 'entries')
        a = np.negative(self.ker(inside[0]))
        print('Initial a_0 =', a)
        self.learn(a, inside, outside)
        pass

    def read_file(self, name):
        with open(name) as f:
            return json.load(f)

    def ker(self, x):
        x1, x2 = x[0], x[1]
        return [x1*x1, x1*x2, x2*x1, x2*x2, x1, x2, 1]

    def learn(self, a, X1, X2):
        i = 1
        while True:
            changed = False
            for x1 in X1:
                kx1 = self.ker(x1)
                if np.dot(a, kx1) >= 0:
                    a = self.adjust(a, np.negative(kx1))
                    changed = True
            if not changed:
                for x2 in X2:
                    kx2 = self.ker(x2)
                    if np.dot(a, kx2) <= 0:
                        a = self.adjust(a, kx2)
                        changed = True
            if not changed:
                break
            print('a_' + str(i) + ' =', a)
            i += 1
            # if i == 10:
            #     break
        return a
                    
    def adjust(self, al, y):
        p = np.dot(np.negative(y), np.subtract(al, y)) / \
            la.norm(np.subtract(al, y)) ** 2
        print('p:', p, ',y:', y)
        p = max(min(p, 1), 0)
        return np.add(np.multiply(p, al), np.multiply(1-p, y))


if __name__ == '__main__':
    m = Main()
    m.run()
