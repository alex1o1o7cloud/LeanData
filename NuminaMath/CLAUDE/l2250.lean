import Mathlib

namespace NUMINAMATH_CALUDE_james_caprisun_purchase_l2250_225006

/-- The total cost of James' Capri-sun purchase -/
def total_cost (num_boxes : ℕ) (pouches_per_box : ℕ) (cost_per_pouch : ℚ) : ℚ :=
  (num_boxes * pouches_per_box : ℕ) * cost_per_pouch

/-- Theorem stating the total cost of James' purchase -/
theorem james_caprisun_purchase :
  total_cost 10 6 (20 / 100) = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_caprisun_purchase_l2250_225006


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2250_225063

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 5)*x - k + 8 > 0) ↔ k > -1 ∧ k < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2250_225063


namespace NUMINAMATH_CALUDE_max_three_digit_gp_length_l2250_225037

/-- A geometric progression of 3-digit natural numbers -/
def ThreeDigitGP (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ (q : ℚ), q > 1 ∧
  (∀ i ≤ n, 100 ≤ a i ∧ a i < 1000) ∧
  (∀ i < n, a (i + 1) = (a i : ℚ) * q)

/-- The maximum length of a 3-digit geometric progression -/
def MaxGPLength : ℕ := 6

/-- Theorem stating that 6 is the maximum length of a 3-digit geometric progression -/
theorem max_three_digit_gp_length :
  (∃ a : ℕ → ℕ, ThreeDigitGP a MaxGPLength) ∧
  (∀ n > MaxGPLength, ∀ a : ℕ → ℕ, ¬ ThreeDigitGP a n) :=
sorry

end NUMINAMATH_CALUDE_max_three_digit_gp_length_l2250_225037


namespace NUMINAMATH_CALUDE_unripe_apples_correct_l2250_225004

/-- Calculates the number of unripe apples given the total number of apples picked,
    the number of pies that can be made, and the number of apples needed per pie. -/
def unripe_apples (total_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total_apples - (num_pies * apples_per_pie)

/-- Proves that the number of unripe apples is correct for the given scenario. -/
theorem unripe_apples_correct : unripe_apples 34 7 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_unripe_apples_correct_l2250_225004


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_265_l2250_225022

-- Define the concept of angles having the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

-- State the theorem
theorem angle_with_same_terminal_side_as_negative_265 :
  same_terminal_side (-265) 95 :=
sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_265_l2250_225022


namespace NUMINAMATH_CALUDE_thermometer_distribution_count_l2250_225093

/-- The number of senior classes -/
def num_classes : ℕ := 10

/-- The total number of thermometers to distribute -/
def total_thermometers : ℕ := 23

/-- The minimum number of thermometers each class must receive -/
def min_thermometers_per_class : ℕ := 2

/-- The number of remaining thermometers after initial distribution -/
def remaining_thermometers : ℕ := total_thermometers - num_classes * min_thermometers_per_class

/-- The number of spaces between items for divider placement -/
def spaces_for_dividers : ℕ := remaining_thermometers - 1

/-- The number of dividers needed -/
def num_dividers : ℕ := num_classes - 1

theorem thermometer_distribution_count :
  (spaces_for_dividers.choose num_dividers) = 220 := by
  sorry

end NUMINAMATH_CALUDE_thermometer_distribution_count_l2250_225093


namespace NUMINAMATH_CALUDE_fifteenth_term_is_3_to_8_l2250_225066

def sequence_term (n : ℕ) : ℤ :=
  if n % 4 == 1 then (-3) ^ (n / 4 + 1)
  else if n % 4 == 3 then 3 ^ (n / 2)
  else 1

theorem fifteenth_term_is_3_to_8 :
  sequence_term 15 = 3^8 := by sorry

end NUMINAMATH_CALUDE_fifteenth_term_is_3_to_8_l2250_225066


namespace NUMINAMATH_CALUDE_test_question_percentage_l2250_225001

theorem test_question_percentage (second_correct : ℝ) (neither_correct : ℝ) (both_correct : ℝ)
  (h1 : second_correct = 0.55)
  (h2 : neither_correct = 0.20)
  (h3 : both_correct = 0.50) :
  ∃ first_correct : ℝ,
    first_correct = 0.75 ∧
    first_correct + second_correct - both_correct + neither_correct = 1 :=
by sorry

end NUMINAMATH_CALUDE_test_question_percentage_l2250_225001


namespace NUMINAMATH_CALUDE_hotel_loss_calculation_l2250_225056

def hotel_loss (expenses : ℝ) (payment_ratio : ℝ) : ℝ :=
  expenses - (payment_ratio * expenses)

theorem hotel_loss_calculation (expenses : ℝ) (payment_ratio : ℝ) 
  (h1 : expenses = 100)
  (h2 : payment_ratio = 3/4) :
  hotel_loss expenses payment_ratio = 25 := by
sorry

end NUMINAMATH_CALUDE_hotel_loss_calculation_l2250_225056


namespace NUMINAMATH_CALUDE_prob_not_six_is_five_sevenths_l2250_225080

/-- A specially designed six-sided die -/
structure SpecialDie :=
  (sides : Nat)
  (odds_six : Rat)
  (is_valid : sides = 6 ∧ odds_six = 2/5)

/-- The probability of rolling a number other than six -/
def prob_not_six (d : SpecialDie) : Rat :=
  1 - (d.odds_six / (1 + d.odds_six))

theorem prob_not_six_is_five_sevenths (d : SpecialDie) :
  prob_not_six d = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_six_is_five_sevenths_l2250_225080


namespace NUMINAMATH_CALUDE_proportional_segments_l2250_225084

theorem proportional_segments (a b c d : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a / b = c / d) →
  a = 6 → b = 9 → c = 12 →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_proportional_segments_l2250_225084


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_l2250_225042

/-- Calculate the profit from a lemonade stand --/
theorem lemonade_stand_profit
  (lemon_cost sugar_cost cup_cost : ℕ)
  (price_per_cup cups_sold : ℕ)
  (h1 : lemon_cost = 10)
  (h2 : sugar_cost = 5)
  (h3 : cup_cost = 3)
  (h4 : price_per_cup = 4)
  (h5 : cups_sold = 21) :
  (price_per_cup * cups_sold) - (lemon_cost + sugar_cost + cup_cost) = 66 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_profit_l2250_225042


namespace NUMINAMATH_CALUDE_twelve_sided_polygon_equilateral_triangles_l2250_225067

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Function to check if a triangle is equilateral -/
def isEquilateral (t : EquilateralTriangle) : Prop := sorry

/-- Function to check if a triangle has at least two vertices from a given set -/
def hasAtLeastTwoVerticesFrom (t : EquilateralTriangle) (s : Set (ℝ × ℝ)) : Prop := sorry

/-- The main theorem -/
theorem twelve_sided_polygon_equilateral_triangles 
  (p : RegularPolygon 12) : 
  ∃ (ts : Finset EquilateralTriangle), 
    (∀ t ∈ ts, isEquilateral t ∧ 
      hasAtLeastTwoVerticesFrom t (Set.range p.vertices)) ∧ 
    ts.card ≥ 12 := by sorry

end NUMINAMATH_CALUDE_twelve_sided_polygon_equilateral_triangles_l2250_225067


namespace NUMINAMATH_CALUDE_single_point_equation_l2250_225024

/-- If the equation 3x^2 + y^2 + 6x - 12y + d = 0 represents a single point, then d = 39 -/
theorem single_point_equation (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 12 * p.2 + d = 0) → d = 39 := by
  sorry

end NUMINAMATH_CALUDE_single_point_equation_l2250_225024


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2250_225095

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 12
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2250_225095


namespace NUMINAMATH_CALUDE_green_to_blue_ratio_l2250_225065

/-- Represents the number of chairs of each color in a classroom --/
structure ClassroomChairs where
  blue : ℕ
  green : ℕ
  white : ℕ

/-- The conditions of the classroom chair problem --/
def classroom_conditions (c : ClassroomChairs) : Prop :=
  c.blue = 10 ∧
  ∃ k : ℕ, c.green = k * c.blue ∧
  c.white = c.green + c.blue - 13 ∧
  c.blue + c.green + c.white = 67

/-- The theorem stating that the ratio of green to blue chairs is 3:1 --/
theorem green_to_blue_ratio (c : ClassroomChairs) 
  (h : classroom_conditions c) : c.green = 3 * c.blue :=
sorry

end NUMINAMATH_CALUDE_green_to_blue_ratio_l2250_225065


namespace NUMINAMATH_CALUDE_quadratic_and_fractional_equations_l2250_225000

theorem quadratic_and_fractional_equations :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 5 ∧ x₂ = 1 - Real.sqrt 5 ∧
    x₁^2 - 2*x₁ - 4 = 0 ∧ x₂^2 - 2*x₂ - 4 = 0) ∧
  (∀ x : ℝ, x ≠ 4 → ((x - 5) / (x - 4) = 1 - x / (4 - x)) ↔ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_fractional_equations_l2250_225000


namespace NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l2250_225060

theorem linear_diophantine_equation_solutions
  (a b c x₀ y₀ : ℤ)
  (h_coprime : Nat.gcd a.natAbs b.natAbs = 1)
  (h_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, a * x + b * y = c ↔ ∃ t : ℤ, x = x₀ + b * t ∧ y = y₀ - a * t :=
by sorry

end NUMINAMATH_CALUDE_linear_diophantine_equation_solutions_l2250_225060


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l2250_225051

/-- The price of a child ticket -/
def child_ticket_price : ℝ := sorry

/-- The price of an adult ticket -/
def adult_ticket_price : ℝ := 2 * child_ticket_price

/-- The condition that 6 adult tickets and 5 child tickets cost $37.50 -/
axiom ticket_condition : 6 * adult_ticket_price + 5 * child_ticket_price = 37.50

/-- The theorem to prove -/
theorem concert_ticket_cost : 
  10 * adult_ticket_price + 8 * child_ticket_price = 61.78 := by sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l2250_225051


namespace NUMINAMATH_CALUDE_problem_statement_l2250_225035

theorem problem_statement (a b c d : ℝ) :
  Real.sqrt (a + b + c + d) + Real.sqrt (a^2 - 2*a + 3 - b) - Real.sqrt (b - c^2 + 4*c - 8) = 3 →
  a - b + c - d = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2250_225035


namespace NUMINAMATH_CALUDE_price_reduction_effect_l2250_225074

theorem price_reduction_effect (original_price : ℝ) (original_sales : ℝ) 
  (price_reduction_percent : ℝ) (net_effect_percent : ℝ) : 
  price_reduction_percent = 40 →
  net_effect_percent = 8 →
  ∃ (sales_increase_percent : ℝ),
    sales_increase_percent = 80 ∧
    (1 - price_reduction_percent / 100) * (1 + sales_increase_percent / 100) = 1 + net_effect_percent / 100 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_effect_l2250_225074


namespace NUMINAMATH_CALUDE_square_ending_four_identical_digits_l2250_225062

theorem square_ending_four_identical_digits (n : ℕ) (d : ℕ) 
  (h1 : d ≤ 9) 
  (h2 : ∃ k : ℕ, n^2 = 10000 * k + d * 1111) : 
  d = 0 := by
sorry

end NUMINAMATH_CALUDE_square_ending_four_identical_digits_l2250_225062


namespace NUMINAMATH_CALUDE_right_triangle_quadratic_roots_l2250_225089

theorem right_triangle_quadratic_roots (m : ℝ) : 
  let f := fun x : ℝ => x^2 - (2*m - 1)*x + 4*(m - 1)
  ∃ (a b : ℝ), 
    (f a = 0 ∧ f b = 0) ∧  -- BC and AC are roots of the quadratic equation
    (a ≠ b) ∧               -- Distinct roots
    (a > 0 ∧ b > 0) ∧       -- Positive lengths
    (a^2 + b^2 = 25) →      -- Pythagorean theorem (AB^2 = 5^2 = 25)
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_quadratic_roots_l2250_225089


namespace NUMINAMATH_CALUDE_expression_evaluation_l2250_225007

theorem expression_evaluation (a b c : ℝ) : 
  (a - (b - c)) - ((a - b) - c) = 2 * c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2250_225007


namespace NUMINAMATH_CALUDE_power_product_exponent_l2250_225076

theorem power_product_exponent (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_exponent_l2250_225076


namespace NUMINAMATH_CALUDE_hernandez_state_tax_l2250_225077

/-- Calculates the state tax for a partial-year resident --/
def calculate_state_tax (months_resident : ℕ) (taxable_income : ℝ) (tax_rate : ℝ) : ℝ :=
  let proportion_resident := months_resident / 12
  let prorated_income := taxable_income * proportion_resident
  prorated_income * tax_rate

/-- Proves that the state tax for Mr. Hernandez is $1,275 --/
theorem hernandez_state_tax :
  calculate_state_tax 9 42500 0.04 = 1275 := by
  sorry

#eval calculate_state_tax 9 42500 0.04

end NUMINAMATH_CALUDE_hernandez_state_tax_l2250_225077


namespace NUMINAMATH_CALUDE_sum_of_base3_digits_333_l2250_225032

/-- Converts a natural number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits in the base-3 representation of 333 is 3 -/
theorem sum_of_base3_digits_333 : sumDigits (toBase3 333) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_base3_digits_333_l2250_225032


namespace NUMINAMATH_CALUDE_random_events_count_l2250_225021

theorem random_events_count (total_events : ℕ) 
  (prob_certain : ℚ) (prob_impossible : ℚ) :
  total_events = 10 →
  prob_certain = 2 / 10 →
  prob_impossible = 3 / 10 →
  (total_events : ℚ) * prob_certain + 
  (total_events : ℚ) * prob_impossible + 
  (total_events - 
    (total_events * prob_certain).floor - 
    (total_events * prob_impossible).floor : ℚ) = total_events →
  total_events - 
    (total_events * prob_certain).floor - 
    (total_events * prob_impossible).floor = 5 := by
  sorry

#check random_events_count

end NUMINAMATH_CALUDE_random_events_count_l2250_225021


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2250_225030

theorem cubic_root_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) → (0 < b ∧ b < 1) → (0 < c ∧ c < 1) →
  a ≠ b → b ≠ c → a ≠ c →
  20 * a^3 - 34 * a^2 + 15 * a - 1 = 0 →
  20 * b^3 - 34 * b^2 + 15 * b - 1 = 0 →
  20 * c^3 - 34 * c^2 + 15 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1.3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2250_225030


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2250_225078

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ 
    x₂^2 + p*x₂ + q = 0 ∧ 
    |x₁ - x₂| = 2) →
  p = Real.sqrt (4*q + 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2250_225078


namespace NUMINAMATH_CALUDE_min_max_sum_sqrt_l2250_225027

theorem min_max_sum_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a^2 + b^2 + c^2 = 6) :
  2 + Real.sqrt 2 ≤ Real.sqrt (4 - a^2) + Real.sqrt (4 - b^2) + Real.sqrt (4 - c^2) ∧
  Real.sqrt (4 - a^2) + Real.sqrt (4 - b^2) + Real.sqrt (4 - c^2) ≤ 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_sqrt_l2250_225027


namespace NUMINAMATH_CALUDE_negation_equivalence_l2250_225098

theorem negation_equivalence : 
  (¬∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ (∀ x : ℝ, -1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2250_225098


namespace NUMINAMATH_CALUDE_sum_of_roots_l2250_225012

-- Define the quadratic equation
def quadratic (x p q : ℝ) : Prop := x^2 - 2*p*x + q = 0

-- Define the theorem
theorem sum_of_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ x y : ℝ, quadratic x p q ∧ quadratic y p q ∧ x ≠ y) →
  x + y = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2250_225012


namespace NUMINAMATH_CALUDE_contrapositive_of_true_implication_l2250_225086

theorem contrapositive_of_true_implication (h : ∀ x : ℝ, x < 0 → x^2 > 0) :
  ∀ x : ℝ, x^2 ≤ 0 → x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_of_true_implication_l2250_225086


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2250_225002

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (10 : ℝ)^x * (1000 : ℝ)^x = (10000 : ℝ)^4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l2250_225002


namespace NUMINAMATH_CALUDE_average_sales_is_16_l2250_225094

def january_sales : ℕ := 15
def february_sales : ℕ := 16
def march_sales : ℕ := 17

def total_months : ℕ := 3

def average_sales : ℚ := (january_sales + february_sales + march_sales : ℚ) / total_months

theorem average_sales_is_16 : average_sales = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_sales_is_16_l2250_225094


namespace NUMINAMATH_CALUDE_vertical_shift_graph_l2250_225069

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define vertical shift operation
def verticalShift (f : RealFunction) (k : ℝ) : RealFunction :=
  λ x => f x + k

-- Theorem statement
theorem vertical_shift_graph (f : RealFunction) (k : ℝ) :
  ∀ x y, y = f x ↔ (y + k) = (verticalShift f k) x :=
sorry

end NUMINAMATH_CALUDE_vertical_shift_graph_l2250_225069


namespace NUMINAMATH_CALUDE_accounting_majors_l2250_225092

theorem accounting_majors (p q r s t u : ℕ) : 
  p * q * r * s * t * u = 51030 →
  1 < p → p < q → q < r → r < s → s < t → t < u →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_accounting_majors_l2250_225092


namespace NUMINAMATH_CALUDE_weight_difference_mildred_carol_l2250_225050

/-- The weight difference between two people -/
def weight_difference (weight1 : ℕ) (weight2 : ℕ) : ℕ :=
  weight1 - weight2

/-- Mildred's weight in pounds -/
def mildred_weight : ℕ := 59

/-- Carol's weight in pounds -/
def carol_weight : ℕ := 9

theorem weight_difference_mildred_carol :
  weight_difference mildred_weight carol_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_mildred_carol_l2250_225050


namespace NUMINAMATH_CALUDE_average_of_eleven_numbers_l2250_225017

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (sixth_number : ℝ) :
  first_six_avg = 19 →
  last_six_avg = 27 →
  sixth_number = 34 →
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 22 :=
by sorry

end NUMINAMATH_CALUDE_average_of_eleven_numbers_l2250_225017


namespace NUMINAMATH_CALUDE_coin_move_termination_uniqueness_l2250_225029

-- Define the coin configuration as a function from integers to natural numbers
def CoinConfiguration := ℤ → ℕ

-- Define a legal move
def is_legal_move (c₁ c₂ : CoinConfiguration) : Prop :=
  ∃ i : ℤ, c₁ i ≥ 2 ∧
    c₂ i = c₁ i - 2 ∧
    c₂ (i - 1) = c₁ (i - 1) + 1 ∧
    c₂ (i + 1) = c₁ (i + 1) + 1 ∧
    ∀ j : ℤ, j ≠ i ∧ j ≠ (i - 1) ∧ j ≠ (i + 1) → c₂ j = c₁ j

-- Define a legal sequence of moves
def legal_sequence (c₀ : CoinConfiguration) (n : ℕ) (c : ℕ → CoinConfiguration) : Prop :=
  c 0 = c₀ ∧
  ∀ i : ℕ, i < n → is_legal_move (c i) (c (i + 1))

-- Define a terminal configuration
def is_terminal (c : CoinConfiguration) : Prop :=
  ∀ i : ℤ, c i ≤ 1

-- The main theorem
theorem coin_move_termination_uniqueness
  (c₀ : CoinConfiguration)
  (n₁ n₂ : ℕ)
  (c₁ : ℕ → CoinConfiguration)
  (c₂ : ℕ → CoinConfiguration)
  (h₁ : legal_sequence c₀ n₁ c₁)
  (h₂ : legal_sequence c₀ n₂ c₂)
  (t₁ : is_terminal (c₁ n₁))
  (t₂ : is_terminal (c₂ n₂)) :
  n₁ = n₂ ∧ c₁ n₁ = c₂ n₂ :=
sorry

end NUMINAMATH_CALUDE_coin_move_termination_uniqueness_l2250_225029


namespace NUMINAMATH_CALUDE_log_x3y2_equals_2_l2250_225036

theorem log_x3y2_equals_2 
  (x y : ℝ) 
  (h1 : Real.log (x * y^2) = 2) 
  (h2 : Real.log (x^2 * y^3) = 3) : 
  Real.log (x^3 * y^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_log_x3y2_equals_2_l2250_225036


namespace NUMINAMATH_CALUDE_simplify_expression_solve_inequality_system_l2250_225005

-- Part 1: Simplification
theorem simplify_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (2 - (x - 1) / (x + 2)) / ((x^2 + 10*x + 25) / (x^2 - 4)) = (x - 2) / (x + 5) := by
  sorry

-- Part 2: Inequality System
theorem solve_inequality_system (x : ℝ) :
  (2*x + 7 > 3 ∧ (x + 1) / 3 > (x - 1) / 2) ↔ -2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_inequality_system_l2250_225005


namespace NUMINAMATH_CALUDE_lillians_candies_l2250_225045

theorem lillians_candies (initial_candies final_candies : ℕ) 
  (h1 : initial_candies = 88)
  (h2 : final_candies = 93) :
  final_candies - initial_candies = 5 := by
  sorry

end NUMINAMATH_CALUDE_lillians_candies_l2250_225045


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2250_225061

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2250_225061


namespace NUMINAMATH_CALUDE_igor_sequence_uses_three_infinitely_l2250_225033

/-- Represents a sequence of natural numbers where each number is obtained
    from the previous one by adding n/p, where p is a prime divisor of n. -/
def IgorSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 1) ∧
  (∀ n, ∃ p, Nat.Prime p ∧ p ∣ a n ∧ a (n + 1) = a n + a n / p)

/-- The theorem stating that in an infinite IgorSequence,
    the prime 3 must be used as a divisor infinitely many times. -/
theorem igor_sequence_uses_three_infinitely (a : ℕ → ℕ) (h : IgorSequence a) :
  ∀ m, ∃ n > m, ∃ p, p = 3 ∧ Nat.Prime p ∧ p ∣ a n ∧ a (n + 1) = a n + a n / p :=
sorry

end NUMINAMATH_CALUDE_igor_sequence_uses_three_infinitely_l2250_225033


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l2250_225088

theorem root_shift_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 22*x + 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l2250_225088


namespace NUMINAMATH_CALUDE_twenty_people_handshakes_l2250_225055

/-- The number of unique handshakes in a group where each person shakes hands once with every other person -/
def number_of_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 20 people, where each person shakes hands once with every other person, there are 190 unique handshakes -/
theorem twenty_people_handshakes :
  number_of_handshakes 20 = 190 := by
  sorry

#eval number_of_handshakes 20

end NUMINAMATH_CALUDE_twenty_people_handshakes_l2250_225055


namespace NUMINAMATH_CALUDE_square_of_103_product_of_998_and_1002_l2250_225072

-- Problem 1
theorem square_of_103 : 103^2 = 10609 := by sorry

-- Problem 2
theorem product_of_998_and_1002 : 998 * 1002 = 999996 := by sorry

end NUMINAMATH_CALUDE_square_of_103_product_of_998_and_1002_l2250_225072


namespace NUMINAMATH_CALUDE_divisibility_criterion_l2250_225071

theorem divisibility_criterion (a : ℤ) : 
  35 ∣ (a^3 - 1) ↔ a % 35 = 1 ∨ a % 35 = 11 ∨ a % 35 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l2250_225071


namespace NUMINAMATH_CALUDE_total_books_correct_l2250_225052

/-- Calculates the total number of books after a purchase -/
def total_books (initial : Real) (bought : Real) : Real :=
  initial + bought

/-- Theorem: The total number of books is the sum of initial and bought books -/
theorem total_books_correct (initial : Real) (bought : Real) :
  total_books initial bought = initial + bought := by
  sorry

end NUMINAMATH_CALUDE_total_books_correct_l2250_225052


namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_value_l2250_225015

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  pos_a : a > 0
  pos_b : b > 0

/-- Theorem: For a hyperbola with given asymptotes and passing through a specific point, a + h = 16/3 -/
theorem hyperbola_a_plus_h_value (H : Hyperbola) 
  (asymptote1 : ∀ x y : ℝ, y = 3*x + 3 → (∀ t : ℝ, (y - H.k)^2/(H.a^2) - (x - H.h)^2/(H.b^2) = t))
  (asymptote2 : ∀ x y : ℝ, y = -3*x - 1 → (∀ t : ℝ, (y - H.k)^2/(H.a^2) - (x - H.h)^2/(H.b^2) = t))
  (point_on_hyperbola : (11 - H.k)^2/(H.a^2) - (2 - H.h)^2/(H.b^2) = 1) :
  H.a + H.h = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_value_l2250_225015


namespace NUMINAMATH_CALUDE_building_height_l2250_225020

/-- Given a flagpole and a building casting shadows under similar conditions,
    this theorem proves the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 65)
  : (flagpole_height / flagpole_shadow) * building_shadow = 26 := by
  sorry

end NUMINAMATH_CALUDE_building_height_l2250_225020


namespace NUMINAMATH_CALUDE_cat_food_sale_revenue_l2250_225082

/-- Calculates the total revenue from cat food sales during a promotion --/
theorem cat_food_sale_revenue : 
  let original_price : ℚ := 25
  let first_group_size : ℕ := 8
  let first_group_cases : ℕ := 3
  let first_group_discount : ℚ := 15/100
  let second_group_size : ℕ := 4
  let second_group_cases : ℕ := 2
  let second_group_discount : ℚ := 10/100
  let third_group_size : ℕ := 8
  let third_group_cases : ℕ := 1
  let third_group_discount : ℚ := 0

  let first_group_revenue := (first_group_size * first_group_cases : ℚ) * 
    (original_price * (1 - first_group_discount))
  let second_group_revenue := (second_group_size * second_group_cases : ℚ) * 
    (original_price * (1 - second_group_discount))
  let third_group_revenue := (third_group_size * third_group_cases : ℚ) * 
    (original_price * (1 - third_group_discount))

  let total_revenue := first_group_revenue + second_group_revenue + third_group_revenue

  total_revenue = 890 := by
  sorry

end NUMINAMATH_CALUDE_cat_food_sale_revenue_l2250_225082


namespace NUMINAMATH_CALUDE_interview_probability_implies_total_workers_l2250_225010

/-- The number of workers excluding Jack and Jill -/
def other_workers : ℕ := 6

/-- The probability of selecting both Jack and Jill for the interview -/
def probability : ℚ := 1 / 28

/-- The number of workers to be selected for the interview -/
def selected_workers : ℕ := 2

/-- The total number of workers -/
def total_workers : ℕ := other_workers + 2

theorem interview_probability_implies_total_workers :
  (probability = (1 : ℚ) / (total_workers.choose selected_workers)) →
  total_workers = 8 := by
  sorry

end NUMINAMATH_CALUDE_interview_probability_implies_total_workers_l2250_225010


namespace NUMINAMATH_CALUDE_total_profit_calculation_l2250_225011

/-- Represents a partner's investment information -/
structure PartnerInvestment where
  initial : ℝ
  monthly : ℝ

/-- Calculates the total annual investment for a partner -/
def annualInvestment (p : PartnerInvestment) : ℝ :=
  p.initial + 12 * p.monthly

/-- Represents the investment information for all partners -/
structure Investments where
  a : PartnerInvestment
  b : PartnerInvestment
  c : PartnerInvestment

/-- Calculates the total investment for all partners -/
def totalInvestment (inv : Investments) : ℝ :=
  annualInvestment inv.a + annualInvestment inv.b + annualInvestment inv.c

/-- The main theorem stating the total profit given the conditions -/
theorem total_profit_calculation (inv : Investments) 
    (h1 : inv.a = { initial := 45000, monthly := 1500 })
    (h2 : inv.b = { initial := 63000, monthly := 2100 })
    (h3 : inv.c = { initial := 72000, monthly := 2400 })
    (h4 : (annualInvestment inv.c / totalInvestment inv) * 60000 = 24000) :
    60000 = (totalInvestment inv * 24000) / (annualInvestment inv.c) := by
  sorry

end NUMINAMATH_CALUDE_total_profit_calculation_l2250_225011


namespace NUMINAMATH_CALUDE_election_winner_percentage_l2250_225054

theorem election_winner_percentage (total_votes winner_votes margin : ℕ) : 
  winner_votes = 720 →
  margin = 240 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l2250_225054


namespace NUMINAMATH_CALUDE_inequality_proof_l2250_225073

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z ≥ 3) :
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 ∧
  ((1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2250_225073


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2250_225068

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The given condition for the sequence -/
def SequenceCondition (a : ℕ → ℝ) : Prop :=
  2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : SequenceCondition a) : 
  a 6 = 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2250_225068


namespace NUMINAMATH_CALUDE_investment_problem_l2250_225038

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_problem :
  let principal : ℝ := 3000
  let rate : ℝ := 0.1
  let time : ℕ := 2
  compound_interest principal rate time = 3630.0000000000005 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l2250_225038


namespace NUMINAMATH_CALUDE_marble_197_is_red_l2250_225090

/-- Represents the color of a marble -/
inductive MarbleColor
| Red
| Blue
| Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  let cycleLength := 15
  let position := n % cycleLength
  if position ≤ 6 then MarbleColor.Red
  else if position ≤ 11 then MarbleColor.Blue
  else MarbleColor.Green

/-- Theorem stating that the 197th marble is red -/
theorem marble_197_is_red : marbleColor 197 = MarbleColor.Red :=
sorry

end NUMINAMATH_CALUDE_marble_197_is_red_l2250_225090


namespace NUMINAMATH_CALUDE_fish_lives_12_years_l2250_225079

/-- The lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a dog in years -/
def dog_lifespan : ℝ := 4 * hamster_lifespan

/-- The lifespan of a well-cared fish in years -/
def fish_lifespan : ℝ := dog_lifespan + 2

/-- Theorem stating that the lifespan of a well-cared fish is 12 years -/
theorem fish_lives_12_years : fish_lifespan = 12 := by
  sorry

end NUMINAMATH_CALUDE_fish_lives_12_years_l2250_225079


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l2250_225097

theorem quadratic_roots_difference (a b : ℝ) : 
  (2 : ℝ) ∈ {x : ℝ | x^2 - (a+1)*x + a = 0} ∧ 
  b ∈ {x : ℝ | x^2 - (a+1)*x + a = 0} → 
  a - b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l2250_225097


namespace NUMINAMATH_CALUDE_skittles_indeterminate_l2250_225083

/-- Given information about pencils and children, prove that the number of skittles per child cannot be determined. -/
theorem skittles_indeterminate (num_children : ℕ) (pencils_per_child : ℕ) (total_pencils : ℕ) 
  (h1 : num_children = 9)
  (h2 : pencils_per_child = 2)
  (h3 : total_pencils = 18)
  (h4 : num_children * pencils_per_child = total_pencils) :
  ∀ (skittles_per_child : ℕ), ∃ (other_skittles_per_child : ℕ), 
    other_skittles_per_child ≠ skittles_per_child ∧ 
    (∀ (total_skittles : ℕ), total_skittles = num_children * skittles_per_child → 
      total_skittles = num_children * other_skittles_per_child) :=
by
  sorry

end NUMINAMATH_CALUDE_skittles_indeterminate_l2250_225083


namespace NUMINAMATH_CALUDE_sequence_formula_l2250_225057

theorem sequence_formula (n : ℕ+) (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h1 : ∀ k, S k = a k / 2 + 1 / a k - 1)
  (h2 : ∀ k, a k > 0) :
  a n = Real.sqrt (2 * n + 1) - Real.sqrt (2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l2250_225057


namespace NUMINAMATH_CALUDE_binomial_30_3_l2250_225053

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l2250_225053


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2250_225016

def vector_a : ℝ × ℝ := (1, -2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem perpendicular_vectors (x : ℝ) : 
  (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2250_225016


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l2250_225099

theorem inscribed_quadrilateral_fourth_side 
  (r : ℝ) 
  (s : ℝ) 
  (h1 : r = 150 * Real.sqrt 3) 
  (h2 : s = 150) : 
  ∃ (x : ℝ), x = 150 * (Real.sqrt 3 - 3) ∧ 
  (s + s + s + x)^2 = 3 * (2 * r)^2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l2250_225099


namespace NUMINAMATH_CALUDE_area_ratio_is_one_twentyfifth_l2250_225075

/-- A square inscribed in a circle with a smaller square as described -/
structure InscribedSquares where
  /-- Radius of the circle -/
  r : ℝ
  /-- Side length of the larger square -/
  s : ℝ
  /-- Side length of the smaller square -/
  t : ℝ
  /-- The larger square is inscribed in the circle -/
  larger_inscribed : s = r * Real.sqrt 2
  /-- The smaller square has one side coinciding with the larger square -/
  coinciding_side : t ≤ s
  /-- Two vertices of the smaller square are on the circle -/
  smaller_on_circle : t * Real.sqrt ((s/2)^2 + (t/2)^2) = r * s

/-- The ratio of the areas of the smaller square to the larger square is 1/25 -/
theorem area_ratio_is_one_twentyfifth (sq : InscribedSquares) :
  (sq.t^2) / (sq.s^2) = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_twentyfifth_l2250_225075


namespace NUMINAMATH_CALUDE_same_point_on_bisector_l2250_225081

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the angle bisector of the first and third quadrants
def firstThirdQuadrantBisector : Set Point2D :=
  { p : Point2D | p.x = p.y }

theorem same_point_on_bisector (a b : ℝ) :
  (Point2D.mk a b = Point2D.mk b a) →
  Point2D.mk a b ∈ firstThirdQuadrantBisector := by
  sorry

end NUMINAMATH_CALUDE_same_point_on_bisector_l2250_225081


namespace NUMINAMATH_CALUDE_easter_egg_probability_l2250_225041

theorem easter_egg_probability : ∀ (total eggs : ℕ) (red_eggs : ℕ) (small_box : ℕ) (large_box : ℕ),
  total = 16 →
  red_eggs = 3 →
  small_box = 6 →
  large_box = 10 →
  small_box + large_box = total →
  (Nat.choose red_eggs 1 * Nat.choose (total - red_eggs) (small_box - 1) +
   Nat.choose red_eggs 2 * Nat.choose (total - red_eggs) (small_box - 2)) /
  Nat.choose total small_box = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_probability_l2250_225041


namespace NUMINAMATH_CALUDE_carmen_jethro_ratio_l2250_225019

-- Define the amounts of money for each person
def jethro_money : ℚ := 20
def patricia_money : ℚ := 60
def carmen_money : ℚ := 113 - jethro_money - patricia_money

-- Define the conditions
axiom patricia_triple_jethro : patricia_money = 3 * jethro_money
axiom total_money : carmen_money + jethro_money + patricia_money = 113
axiom carmen_multiple_after : ∃ (m : ℚ), carmen_money + 7 = m * jethro_money

-- Theorem to prove
theorem carmen_jethro_ratio :
  (carmen_money + 7) / jethro_money = 2 := by sorry

end NUMINAMATH_CALUDE_carmen_jethro_ratio_l2250_225019


namespace NUMINAMATH_CALUDE_work_completion_time_l2250_225044

/-- Given that two workers 'a' and 'b' can complete a job together in 4 days,
    and 'a' alone can complete the job in 12 days, prove that 'b' alone
    can complete the job in 6 days. -/
theorem work_completion_time (work_rate_a : ℚ) (work_rate_b : ℚ) :
  work_rate_a + work_rate_b = 1 / 4 →
  work_rate_a = 1 / 12 →
  work_rate_b = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2250_225044


namespace NUMINAMATH_CALUDE_place_value_comparison_l2250_225058

theorem place_value_comparison (n : Real) (h : n = 85376.4201) : 
  (10 : Real) / (1 / 10 : Real) = 100 := by
  sorry

end NUMINAMATH_CALUDE_place_value_comparison_l2250_225058


namespace NUMINAMATH_CALUDE_wizard_potion_combinations_l2250_225003

/-- Represents the number of valid potion combinations given the constraints. -/
def validPotionCombinations (plants : ℕ) (gemstones : ℕ) 
  (incompatible_2gem_1plant : ℕ) (incompatible_1gem_2plant : ℕ) : ℕ :=
  plants * gemstones - (incompatible_2gem_1plant + 2 * incompatible_1gem_2plant)

/-- Theorem stating that given the specific constraints, there are 20 valid potion combinations. -/
theorem wizard_potion_combinations : 
  validPotionCombinations 4 6 2 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_wizard_potion_combinations_l2250_225003


namespace NUMINAMATH_CALUDE_roden_fish_count_l2250_225085

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 15

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The total number of fish Roden bought -/
def total_fish : ℕ := gold_fish + blue_fish

theorem roden_fish_count : total_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_count_l2250_225085


namespace NUMINAMATH_CALUDE_divisible_by_45_sum_of_digits_l2250_225040

theorem divisible_by_45_sum_of_digits (a b : ℕ) : 
  (a < 10) →
  (b < 10) →
  (6 * 10000 + a * 1000 + 700 + 80 + b) % 45 = 0 →
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_divisible_by_45_sum_of_digits_l2250_225040


namespace NUMINAMATH_CALUDE_artist_cube_structure_surface_area_l2250_225096

/-- Represents the cube structure described in the problem -/
structure CubeStructure where
  totalCubes : ℕ
  cubeEdgeLength : ℝ
  bottomLayerSize : ℕ
  topLayerSize : ℕ

/-- Calculates the exposed surface area of the cube structure -/
def exposedSurfaceArea (cs : CubeStructure) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem artist_cube_structure_surface_area :
  ∃ (cs : CubeStructure),
    cs.totalCubes = 16 ∧
    cs.cubeEdgeLength = 1 ∧
    cs.bottomLayerSize = 3 ∧
    cs.topLayerSize = 2 ∧
    exposedSurfaceArea cs = 49 :=
  sorry

end NUMINAMATH_CALUDE_artist_cube_structure_surface_area_l2250_225096


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2250_225043

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 5*p + 3 = 0 → 
  q^2 - 5*q + 3 = 0 → 
  p^2 + q^2 + p + q = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2250_225043


namespace NUMINAMATH_CALUDE_inscribed_sphere_sum_l2250_225087

/-- A right cone with a sphere inscribed in it -/
structure InscribedSphere where
  baseRadius : ℝ
  height : ℝ
  sphereRadius : ℝ
  b : ℝ
  d : ℝ
  base_radius_positive : 0 < baseRadius
  height_positive : 0 < height
  sphere_radius_formula : sphereRadius = b * Real.sqrt d - b

/-- The theorem stating that b + d = 20 for the given conditions -/
theorem inscribed_sphere_sum (cone : InscribedSphere)
  (h1 : cone.baseRadius = 15)
  (h2 : cone.height = 30) :
  cone.b + cone.d = 20 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_sum_l2250_225087


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2250_225039

theorem mod_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1774 [ZMOD 7] ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l2250_225039


namespace NUMINAMATH_CALUDE_marble_probability_difference_l2250_225064

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 2000

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 2000

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (red_marbles * (red_marbles - 1) + black_marbles * (black_marbles - 1)) / (total_marbles * (total_marbles - 1))

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (2 * red_marbles * black_marbles) / (total_marbles * (total_marbles - 1))

/-- The theorem stating the absolute difference between P_s and P_d -/
theorem marble_probability_difference : |P_s - P_d| = 1 / 3999 := by sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l2250_225064


namespace NUMINAMATH_CALUDE_age_birth_year_problem_l2250_225009

theorem age_birth_year_problem :
  ∃ (age1 age2 : ℕ) (birth_year1 birth_year2 : ℕ),
    age1 > 11 ∧ age2 > 11 ∧
    birth_year1 ≥ 1900 ∧ birth_year1 < 2010 ∧
    birth_year2 ≥ 1900 ∧ birth_year2 < 2010 ∧
    age1 = (birth_year1 / 1000) + ((birth_year1 % 1000) / 100) + ((birth_year1 % 100) / 10) + (birth_year1 % 10) ∧
    age2 = (birth_year2 / 1000) + ((birth_year2 % 1000) / 100) + ((birth_year2 % 100) / 10) + (birth_year2 % 10) ∧
    2010 - birth_year1 = age1 ∧
    2009 - birth_year2 = age2 ∧
    age1 ≠ age2 ∧
    birth_year1 ≠ birth_year2 :=
by sorry

end NUMINAMATH_CALUDE_age_birth_year_problem_l2250_225009


namespace NUMINAMATH_CALUDE_correct_number_of_workers_l2250_225023

/-- The number of workers in the team -/
def n : ℕ := 9

/-- The number of days it takes the full team to complete the task -/
def full_team_days : ℕ := 7

/-- The number of days it takes the team minus two workers to complete the task -/
def team_minus_two_days : ℕ := 14

/-- The number of days it takes the team minus six workers to complete the task -/
def team_minus_six_days : ℕ := 42

/-- The theorem stating that n is the correct number of workers -/
theorem correct_number_of_workers :
  n * full_team_days = (n - 2) * team_minus_two_days ∧
  n * full_team_days = (n - 6) * team_minus_six_days :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_workers_l2250_225023


namespace NUMINAMATH_CALUDE_product_digit_sum_l2250_225026

theorem product_digit_sum : 
  let a := 2^20
  let b := 5^17
  let product := a * b
  (List.sum (product.digits 10)) = 8 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2250_225026


namespace NUMINAMATH_CALUDE_base8_to_base10_3206_l2250_225008

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base 8 representation of the number --/
def base8_num : List Nat := [6, 0, 2, 3]

/-- Theorem stating that the base 10 representation of 3206₈ is 1670 --/
theorem base8_to_base10_3206 : base8_to_base10 base8_num = 1670 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_3206_l2250_225008


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2250_225049

def game_condition (N : ℕ) : Prop :=
  N ≤ 999 ∧
  2*N < 1000 ∧
  2*N + 50 < 1000 ∧
  4*N + 100 < 1000 ∧
  4*N + 150 < 1000 ∧
  8*N + 300 < 1000 ∧
  8*N + 350 < 1000 ∧
  16*N + 700 < 1000 ∧
  16*N + 750 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_winning_number :
  ∃ N : ℕ, game_condition N ∧
    (∀ M : ℕ, M < N → ¬game_condition M) ∧
    N = 16 ∧
    sum_of_digits N = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2250_225049


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l2250_225047

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 19) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l2250_225047


namespace NUMINAMATH_CALUDE_initial_books_count_l2250_225025

theorem initial_books_count (initial_books sold_books given_books remaining_books : ℕ) :
  sold_books = 11 →
  given_books = 35 →
  remaining_books = 62 →
  initial_books = sold_books + given_books + remaining_books →
  initial_books = 108 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_count_l2250_225025


namespace NUMINAMATH_CALUDE_january_salary_l2250_225018

/-- The average salary calculation problem -/
theorem january_salary 
  (avg_jan_to_apr : ℝ) 
  (avg_feb_to_may : ℝ) 
  (may_salary : ℝ) 
  (h1 : avg_jan_to_apr = 8000)
  (h2 : avg_feb_to_may = 8500)
  (h3 : may_salary = 6500) :
  let jan_salary := 4 * avg_jan_to_apr - (4 * avg_feb_to_may - may_salary)
  jan_salary = 4500 := by
sorry


end NUMINAMATH_CALUDE_january_salary_l2250_225018


namespace NUMINAMATH_CALUDE_wheat_rate_proof_l2250_225014

/-- Represents the rate of the second batch of wheat in rupees per kg -/
def second_batch_rate : ℝ := 14.25

/-- Proves that the rate of the second batch of wheat is 14.25 rupees per kg -/
theorem wheat_rate_proof (first_batch_weight : ℝ) (second_batch_weight : ℝ) 
  (first_batch_rate : ℝ) (mixture_selling_rate : ℝ) (profit_percentage : ℝ) :
  first_batch_weight = 30 →
  second_batch_weight = 20 →
  first_batch_rate = 11.50 →
  mixture_selling_rate = 15.12 →
  profit_percentage = 0.20 →
  second_batch_rate = 14.25 := by
  sorry

#check wheat_rate_proof

end NUMINAMATH_CALUDE_wheat_rate_proof_l2250_225014


namespace NUMINAMATH_CALUDE_barrelCapacitiesSolution_l2250_225091

/-- Represents the capacities of three barrels --/
structure BarrelCapacities where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given capacities satisfy the problem conditions --/
def satisfiesConditions (c : BarrelCapacities) : Prop :=
  -- After first transfer, 1/4 remains in first barrel
  c.second = (3 * c.first) / 4 ∧
  -- After second transfer, 2/9 remains in second barrel
  c.third = (7 * c.first) / 12 ∧
  -- After third transfer, 50 more units needed to fill first barrel
  c.third + 50 = c.first

/-- The theorem to prove --/
theorem barrelCapacitiesSolution :
  ∃ (c : BarrelCapacities), satisfiesConditions c ∧ c.first = 120 ∧ c.second = 90 ∧ c.third = 70 := by
  sorry

end NUMINAMATH_CALUDE_barrelCapacitiesSolution_l2250_225091


namespace NUMINAMATH_CALUDE_arithmetic_progression_pairs_l2250_225028

/-- A pair of real numbers (a, b) forms an arithmetic progression with 10 and ab if
    the differences between consecutive terms are equal. -/
def is_arithmetic_progression (a b : ℝ) : Prop :=
  (a - 10 = b - a) ∧ (b - a = a * b - b)

/-- The only pairs (a, b) of real numbers such that 10, a, b, ab form an arithmetic progression
    are (4, -2) and (2.5, -5). -/
theorem arithmetic_progression_pairs :
  ∀ a b : ℝ, is_arithmetic_progression a b ↔ (a = 4 ∧ b = -2) ∨ (a = 2.5 ∧ b = -5) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_pairs_l2250_225028


namespace NUMINAMATH_CALUDE_p_recurrence_l2250_225031

/-- The probability of having a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℚ :=
  sorry

/-- The recurrence relation for p(n,k) -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end NUMINAMATH_CALUDE_p_recurrence_l2250_225031


namespace NUMINAMATH_CALUDE_bob_payment_bob_acorn_payment_l2250_225070

theorem bob_payment (alice_acorns : ℕ) (alice_price_per_acorn : ℚ) (alice_bob_price_ratio : ℕ) : ℚ :=
  let alice_total_payment := alice_acorns * alice_price_per_acorn
  alice_total_payment / alice_bob_price_ratio

theorem bob_acorn_payment : bob_payment 3600 15 9 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_bob_payment_bob_acorn_payment_l2250_225070


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l2250_225013

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag of balls -/
def bag : Multiset BallColor := 
  2 • {BallColor.Red} + 2 • {BallColor.White}

/-- Event: At least one white ball is drawn -/
def atLeastOneWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∨ outcome.second = BallColor.White

/-- Event: Both balls are red -/
def bothRed (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.Red ∧ outcome.second = BallColor.Red

/-- The probability of an event occurring -/
noncomputable def probability (event : DrawOutcome → Prop) : ℝ :=
  sorry

theorem mutually_exclusive_events :
  probability (fun outcome => atLeastOneWhite outcome ∧ bothRed outcome) = 0 ∧
  probability atLeastOneWhite + probability bothRed = 1 :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l2250_225013


namespace NUMINAMATH_CALUDE_base8_perfect_square_b_zero_l2250_225046

/-- Represents a number in base 8 of the form a1b4 -/
structure Base8Number where
  a : ℕ
  b : ℕ
  h_a_nonzero : a ≠ 0

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : ℕ :=
  512 * n.a + 64 + 8 * n.b + 4

/-- Predicate to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem base8_perfect_square_b_zero (n : Base8Number) :
  isPerfectSquare (toDecimal n) → n.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_base8_perfect_square_b_zero_l2250_225046


namespace NUMINAMATH_CALUDE_dress_cost_theorem_l2250_225034

/-- The cost of a dress given the initial and remaining number of quarters -/
def dress_cost (initial_quarters remaining_quarters : ℕ) : ℚ :=
  (initial_quarters - remaining_quarters) * (1 / 4)

/-- Theorem stating that the dress cost $35 given the initial and remaining quarters -/
theorem dress_cost_theorem (initial_quarters remaining_quarters : ℕ) 
  (h1 : initial_quarters = 160)
  (h2 : remaining_quarters = 20) :
  dress_cost initial_quarters remaining_quarters = 35 := by
  sorry

#eval dress_cost 160 20

end NUMINAMATH_CALUDE_dress_cost_theorem_l2250_225034


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2250_225048

/-- Given two right triangles with sides 6, 8, and 10, where:
    x is the side length of a square inscribed in the first triangle with one vertex at the right angle,
    y is the side length of a square inscribed in the second triangle with one side on the hypotenuse,
    then x/y = 37/35. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  (∃ (a b c : ℝ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2) →  -- right triangle condition
  (x * 8 = 24) →  -- condition for x (derived from the problem statement)
  (37 * y = 240) →  -- condition for y (derived from the problem statement)
  x / y = 37 / 35 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2250_225048


namespace NUMINAMATH_CALUDE_yellow_block_weight_l2250_225059

theorem yellow_block_weight (green_weight : ℝ) (weight_difference : ℝ) 
  (h1 : green_weight = 0.4)
  (h2 : weight_difference = 0.2) :
  green_weight + weight_difference = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_yellow_block_weight_l2250_225059
