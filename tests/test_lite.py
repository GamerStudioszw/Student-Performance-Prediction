import json
import math
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lite_ml import features, fit, predict, solve, train
from lightweight_app import create_server


class LiteModelTests(unittest.TestCase):
    def test_linear_solver(self):
        result = solve([[3, 2], [1, 2]], [5, 5])
        self.assertAlmostEqual(result[0], 0)
        self.assertAlmostEqual(result[1], 2.5)

    def test_regression_learns_known_linear_relationship(self):
        x = [[float(i)] for i in range(30)]
        y = [3 * row[0] + 2 for row in x]
        model = fit(x, y, 0.0001)
        self.assertAlmostEqual(predict(model, [35]), 107, places=2)
        self.assertEqual(model['means'], [14.5])

    def test_invalid_inputs_and_target_exclusion(self):
        record = dict(G1=12, G2=13, studytime=2, failures=0, schoolsup='no', famsup='yes')
        self.assertEqual(features(record), features({**record, 'G3': 20}))
        for name, value in [('G1', -1), ('G2', 21), ('studytime', 1.5), ('failures', 4), ('G1', math.nan), ('schoolsup', 'maybe')]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                features({**record, name: value})

    def test_training_is_reproducible(self):
        model, report = train()
        other_model, other_report = train()
        self.assertEqual(model, other_model)
        self.assertEqual(report, other_report)
        self.assertEqual(report['rows'], report['train_rows'] + report['test_rows'])
        self.assertEqual(report['selected_model'], min(report['comparison'], key=lambda item: item['cv_mae'])['model'])
        self.assertTrue(math.isfinite(report['mae']))


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_page_and_evaluation(self):
        with urlopen(self.base, timeout=5) as response:
            self.assertIn(b'Student Performance Prediction', response.read())
        with urlopen(self.base + '/api/metrics', timeout=5) as response:
            report = json.load(response)
        self.assertEqual(report['test_rows'], 79)

    def test_prediction_and_invalid_input(self):
        record = dict(G1=10, G2=11, studytime=2, failures=0, schoolsup='no', famsup='yes')
        request = Request(self.base + '/api/predict', data=json.dumps(record).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        self.assertAlmostEqual(result['grade'], predict(self.server.model, features(record)))
        bad = Request(self.base + '/api/predict', data=b'{"G1":99}', headers={'Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as context:
            urlopen(bad, timeout=5)
        self.assertEqual(context.exception.code, 400)
        context.exception.close()

    def test_rebinding_cross_origin_and_malformed_json_are_rejected(self):
        attempts = [
            (b'{}', {'Content-Type': 'application/json', 'Origin': 'https://evil.example'}, 403),
            (b'{}', {'Content-Type': 'application/json', 'Host': 'evil.example'}, 403),
            (b'{}', {'Content-Type': 'text/plain'}, 400),
            (b'{"G1":10,"G1":20}', {'Content-Type': 'application/json'}, 400),
            (b'[' * 2000, {'Content-Type': 'application/json'}, 400),
            (b'x' * 8193, {'Content-Type': 'application/json'}, 400),
        ]
        for body, headers, status in attempts:
            with self.subTest(headers=headers, length=len(body)):
                request = Request(self.base + '/api/predict', data=body, headers=headers)
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request, timeout=5)
                self.assertEqual(caught.exception.code, status)
                self.assertEqual(caught.exception.headers.get('Cache-Control'), 'no-store')
                caught.exception.close()


if __name__ == '__main__':
    unittest.main()
