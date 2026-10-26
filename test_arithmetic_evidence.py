import unittest
from arithmetic_evidence import inspect_text, numeric_value, features_for_text


class ArithmeticTests(unittest.TestCase):
    def test_exact_and_latex(self):
        cases = [r'$3 \times 66 - 20 = 178$', r'$\frac{1}{2} = 0.5$',
                 r'$\frac{10+15+5}{2} = \frac{30}{2} = 15$',
                 r'$$\text{Total} = 45 \times 20 = \boxed{900}$$',
                 r'$2^{3}=8$', r'$-3+4=1$',
                 r'$1,000+200=1200$', '<<12/2=6>>', r'$\frac{1}{\frac{2}{3}}=1.5$']
        for text in cases:
            with self.subTest(text=text):
                checks = inspect_text(text)
                self.assertTrue(checks)
                self.assertFalse(any(c['inconsistent'] for c in checks))

    def test_wrong_arithmetic_is_not_wrong_final_label(self):
        text = r'$3\times66-20=198$' + '\n#### 178'
        self.assertEqual(sum(c['inconsistent'] for c in inspect_text(text)), 1)
        self.assertEqual(features_for_text('#### 178'), [0, 0, 0, 1])

    def test_unknown_not_partial_expression(self):
        for text in [r'$x+3=4$', r'$3x=6$', r'$2+3\text{ hours}=5$', r'$1/0=2$',
                     r'$\sqrt{4}=2$', r'$2**1000=3$', r'$3+\frac{x}{2}=5$',
                     '$2+3≈6$', 'There are 3 + 4 = 7 people.', '$1/3=0.333...$',
                     r'$a\frac{2}{3}=4$', 'no equation', '$2//3=0$', r'$2+3=2\text{ apples}+3$']:
            with self.subTest(text=text):
                self.assertEqual(inspect_text(text), [])

    def test_rounding_and_integer_exact(self):
        self.assertFalse(inspect_text('$1/3=0.33$')[0]['inconsistent'])
        self.assertTrue(inspect_text('$1/3=0.32$')[0]['inconsistent'])
        self.assertTrue(inspect_text('$10/3=3$')[0]['inconsistent'])

    def test_labels_chain_units_and_multiple_spans(self):
        checks = inspect_text(r'$x=340-240=100$ and $2+3=5\text{ apples}$')
        self.assertEqual(len(checks), 2)
        self.assertFalse(any(c['inconsistent'] for c in checks))

    def test_currency_percent_and_remainder_ambiguity(self):
        examples = [r'$\frac{8}{40}\times100=20\%$',
                    r'$\frac{47}{4}=11\text{ with a remainder of }3$',
                    '- Haircuts: $30 × 4 = $120\n- Perms: $40 × 1 = $40\n- Dye: $60 × 2 = $120',
                    'Dessert: $20 × 0.25 = $5\nTotal = $15 + $20 + $5 = **$40**',
                    '$0.40 × 600 = $240\n$15 × 24 = $360\n$400 + $240 + $360 = **$1000**']
        for text in examples:
            with self.subTest(text=text):
                self.assertFalse(any(c['inconsistent'] for c in inspect_text(text)))


if __name__ == '__main__':
    unittest.main()
