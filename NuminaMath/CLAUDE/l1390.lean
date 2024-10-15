import Mathlib

namespace NUMINAMATH_CALUDE_angle_measure_l1390_139097

theorem angle_measure (x : ℝ) : 
  (90 - x = 2/3 * (180 - x) - 40) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l1390_139097


namespace NUMINAMATH_CALUDE_bake_sale_donation_l1390_139085

/-- The total donation to the homeless shelter given the bake sale earnings and additional personal donation -/
def total_donation_to_shelter (total_earnings : ℕ) (ingredients_cost : ℕ) (personal_donation : ℕ) : ℕ :=
  let remaining_total := total_earnings - ingredients_cost
  let shelter_donation := remaining_total / 2 + personal_donation
  shelter_donation

/-- Theorem stating that the total donation to the homeless shelter is $160 -/
theorem bake_sale_donation :
  total_donation_to_shelter 400 100 10 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_donation_l1390_139085


namespace NUMINAMATH_CALUDE_article_cost_l1390_139073

/-- The cost of an article given specific selling prices and gain percentages -/
theorem article_cost : ∃ (cost : ℝ), 
  (895 - cost) = 1.075 * (785 - cost) ∧ 
  cost > 0 ∧ 
  cost < 785 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l1390_139073


namespace NUMINAMATH_CALUDE_total_games_is_32_l1390_139004

/-- The number of games won by Jerry -/
def jerry_games : ℕ := 7

/-- The number of games won by Dave -/
def dave_games : ℕ := jerry_games + 3

/-- The number of games won by Ken -/
def ken_games : ℕ := dave_games + 5

/-- The total number of games played -/
def total_games : ℕ := ken_games + dave_games + jerry_games

theorem total_games_is_32 : total_games = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_games_is_32_l1390_139004


namespace NUMINAMATH_CALUDE_fraction_cracked_pots_is_two_fifths_l1390_139028

/-- The fraction of cracked pots given the initial number of pots,
    the revenue from selling non-cracked pots, and the price per pot. -/
def fraction_cracked_pots (initial_pots : ℕ) (revenue : ℕ) (price_per_pot : ℕ) : ℚ :=
  1 - (revenue / (initial_pots * price_per_pot) : ℚ)

/-- Theorem stating that the fraction of cracked pots is 2/5 given the problem conditions. -/
theorem fraction_cracked_pots_is_two_fifths :
  fraction_cracked_pots 80 1920 40 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cracked_pots_is_two_fifths_l1390_139028


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1390_139092

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Theorem: If S_20 = S_40 for an arithmetic sequence, then S_60 = 0 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
  (h : seq.S 20 = seq.S 40) : seq.S 60 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1390_139092


namespace NUMINAMATH_CALUDE_odd_periodic_function_sum_l1390_139083

-- Define the properties of the function f
def is_odd_periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (f 1 = 2) ∧ 
  (∀ x, f (x + 1) = f (x + 5))

-- State the theorem
theorem odd_periodic_function_sum (f : ℝ → ℝ) 
  (h : is_odd_periodic_function f) : f 12 + f 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_sum_l1390_139083


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l1390_139076

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

/-- A function that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- The main theorem -/
theorem smallest_prime_perimeter_triangle :
  ∃ (a b c : ℕ),
    a < b ∧ b < c ∧
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a > 5 ∧ b > 5 ∧ c > 5 ∧
    isValidTriangle a b c ∧
    isPrime (a + b + c) ∧
    (∀ (x y z : ℕ),
      x < y ∧ y < z ∧
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x > 5 ∧ y > 5 ∧ z > 5 ∧
      isValidTriangle x y z ∧
      isPrime (x + y + z) →
      a + b + c ≤ x + y + z) ∧
    a + b + c = 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l1390_139076


namespace NUMINAMATH_CALUDE_inequality_holds_iff_one_l1390_139082

def is_valid (x : ℕ) : Prop := x > 0 ∧ x < 100

theorem inequality_holds_iff_one (x : ℕ) (h : is_valid x) :
  (2^x : ℚ) / x.factorial > x^2 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_one_l1390_139082


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1390_139067

theorem radical_conjugate_sum_product (c d : ℝ) : 
  (c + 2 * Real.sqrt d) + (c - 2 * Real.sqrt d) = 6 ∧ 
  (c + 2 * Real.sqrt d) * (c - 2 * Real.sqrt d) = 4 → 
  c + d = 17/4 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1390_139067


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1390_139020

theorem constant_term_expansion (n : ℕ) : 
  (∃ (k : ℕ), (Nat.choose n (2*n/3 : ℕ)) = 15 ∧ 2*n/3 = k) → n = 6 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1390_139020


namespace NUMINAMATH_CALUDE_fuel_price_increase_l1390_139035

/-- Calculates the percentage increase in fuel prices given the original cost for one tank,
    and the new cost for double the capacity. -/
theorem fuel_price_increase (original_cost new_cost : ℝ) : 
  original_cost > 0 →
  new_cost > original_cost * 2 →
  (new_cost - original_cost * 2) / (original_cost * 2) * 100 = 20 :=
by
  sorry

#check fuel_price_increase

end NUMINAMATH_CALUDE_fuel_price_increase_l1390_139035


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1390_139007

/-- Perimeter of a rectangle with area equal to a right triangle --/
theorem rectangle_perimeter (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a * b = 108) : 
  let triangle_area := a * b / 2
  let rectangle_length := c / 2
  let rectangle_width := triangle_area / rectangle_length
  2 * (rectangle_length + rectangle_width) = 29.4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1390_139007


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1390_139087

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1390_139087


namespace NUMINAMATH_CALUDE_length_of_AB_l1390_139000

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = -12*x

-- Define the line
def line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧ 
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) : 
  intersection_points A B → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 30 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l1390_139000


namespace NUMINAMATH_CALUDE_age_ratio_is_one_half_l1390_139026

/-- The ratio of Pam's age to Rena's age -/
def age_ratio (p r : ℕ) : ℚ := p / r

/-- Pam's current age -/
def pam_age : ℕ := 5

theorem age_ratio_is_one_half :
  ∃ (r : ℕ), 
    r > pam_age ∧ 
    r + 10 = pam_age + 15 ∧ 
    age_ratio pam_age r = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_one_half_l1390_139026


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1390_139075

theorem pure_imaginary_condition (m : ℝ) : 
  (∀ z : ℂ, z = Complex.mk (m^2 - 1) (m + 1) → z.re = 0 ∧ z.im ≠ 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1390_139075


namespace NUMINAMATH_CALUDE_partnership_investment_l1390_139056

/-- Partnership investment problem -/
theorem partnership_investment (x : ℝ) (y : ℝ) : 
  x > 0 →  -- Raman's investment is positive
  y > 0 →  -- Lakshmi invests after a positive number of months
  y < 12 → -- Lakshmi invests before the end of the year
  (2 * x * (12 - y)) / (x * 12 + 2 * x * (12 - y) + 3 * x * 4) = 1 / 3 →
  y = 6 := by
sorry

end NUMINAMATH_CALUDE_partnership_investment_l1390_139056


namespace NUMINAMATH_CALUDE_largest_number_l1390_139096

-- Define the base conversion function
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the numbers in their respective bases
def A : List Nat := [1, 0, 1, 1, 1, 1]
def B : List Nat := [1, 2, 1, 0]
def C : List Nat := [1, 1, 2]
def D : List Nat := [6, 9]

-- Theorem statement
theorem largest_number :
  to_decimal D 12 > to_decimal A 2 ∧
  to_decimal D 12 > to_decimal B 3 ∧
  to_decimal D 12 > to_decimal C 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1390_139096


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l1390_139030

theorem least_subtrahend_for_divisibility (n m : ℕ) : 
  ∃ (x : ℕ), x = n % m ∧ 
  (∀ (y : ℕ), (n - y) % m = 0 → y ≥ x) ∧
  (n - x) % m = 0 :=
sorry

#check least_subtrahend_for_divisibility 13602 87

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l1390_139030


namespace NUMINAMATH_CALUDE_circle_symmetry_sum_l1390_139045

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a circle is symmetric with respect to a line -/
def isSymmetric (circle : Circle) (line : Line) : Prop :=
  sorry

/-- The main theorem -/
theorem circle_symmetry_sum (circle : Circle) 
    (l₁ : Line) (l₂ : Line) :
    l₁ = Line.mk 1 (-1) 4 →
    l₂ = Line.mk 1 3 0 →
    isSymmetric circle l₁ →
    isSymmetric circle l₂ →
    circle.D + circle.E = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_sum_l1390_139045


namespace NUMINAMATH_CALUDE_panda_arrangement_count_l1390_139019

/-- Represents the number of pandas -/
def num_pandas : ℕ := 9

/-- Represents the number of shortest pandas that must be at the ends -/
def num_shortest : ℕ := 3

/-- Calculates the number of ways to arrange the pandas -/
def panda_arrangements : ℕ :=
  2 * (num_pandas - num_shortest).factorial

/-- Theorem stating that the number of panda arrangements is 1440 -/
theorem panda_arrangement_count :
  panda_arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_panda_arrangement_count_l1390_139019


namespace NUMINAMATH_CALUDE_log_inequality_l1390_139013

theorem log_inequality : 
  Real.log 6 / Real.log 3 > Real.log 10 / Real.log 5 ∧ 
  Real.log 10 / Real.log 5 > Real.log 14 / Real.log 7 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1390_139013


namespace NUMINAMATH_CALUDE_plane_equation_l1390_139053

/-- The plane passing through points (0,3,-1), (4,7,1), and (2,5,0) has the equation y - 2z - 5 = 0 -/
theorem plane_equation (p q r : ℝ × ℝ × ℝ) : 
  p = (0, 3, -1) → q = (4, 7, 1) → r = (2, 5, 0) →
  ∃ (A B C D : ℤ), 
    (A > 0) ∧ 
    (Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) ∧
    (∀ (x y z : ℝ), A * x + B * y + C * z + D = 0 ↔ y - 2 * z - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_l1390_139053


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1390_139018

theorem polynomial_evaluation :
  let f (x : ℤ) := x^3 + x^2 + x + 1
  f (-2) = -5 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1390_139018


namespace NUMINAMATH_CALUDE_weight_of_three_moles_l1390_139015

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.01

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in C6H8O6 -/
def carbon_count : ℕ := 6

/-- The number of Hydrogen atoms in C6H8O6 -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in C6H8O6 -/
def oxygen_count : ℕ := 6

/-- The number of moles of C6H8O6 -/
def mole_count : ℝ := 3

/-- The molecular weight of C6H8O6 in g/mol -/
def molecular_weight : ℝ := 
  carbon_count * carbon_weight + 
  hydrogen_count * hydrogen_weight + 
  oxygen_count * oxygen_weight

/-- The total weight of 3 moles of C6H8O6 in grams -/
theorem weight_of_three_moles : 
  mole_count * molecular_weight = 528.42 := by sorry

end NUMINAMATH_CALUDE_weight_of_three_moles_l1390_139015


namespace NUMINAMATH_CALUDE_base_7_addition_problem_l1390_139050

/-- Convert a base 7 number to base 10 -/
def to_base_10 (a b c : ℕ) : ℕ := a * 7^2 + b * 7 + c

/-- Convert a base 10 number to base 7 -/
def to_base_7 (n : ℕ) : ℕ × ℕ × ℕ :=
  let hundreds := n / 49
  let remainder := n % 49
  let tens := remainder / 7
  let ones := remainder % 7
  (hundreds, tens, ones)

theorem base_7_addition_problem (X Y : ℕ) :
  (to_base_7 (to_base_10 5 X Y + to_base_10 0 5 2) = (6, 4, X)) →
  X + Y = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_7_addition_problem_l1390_139050


namespace NUMINAMATH_CALUDE_product_of_squares_minus_seven_squares_l1390_139034

theorem product_of_squares_minus_seven_squares 
  (a b c d : ℤ) : (a^2 - 7*b^2) * (c^2 - 7*d^2) = (a*c + 7*b*d)^2 - 7*(a*d + b*c)^2 :=
by sorry

end NUMINAMATH_CALUDE_product_of_squares_minus_seven_squares_l1390_139034


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_equality_l1390_139037

/-- Represents a repeating decimal with a repeating part and a period length. -/
structure RepeatingDecimal where
  repeating_part : ℕ
  period_length : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def to_rational (d : RepeatingDecimal) : ℚ :=
  d.repeating_part / (10^d.period_length - 1)

/-- The sum of the three given repeating decimals equals 10099098/29970003. -/
theorem repeating_decimal_sum_equality : 
  let d1 := RepeatingDecimal.mk 3 1
  let d2 := RepeatingDecimal.mk 4 3
  let d3 := RepeatingDecimal.mk 5 4
  to_rational d1 + to_rational d2 + to_rational d3 = 10099098 / 29970003 := by
  sorry

#eval (10099098 : ℚ) / 29970003

end NUMINAMATH_CALUDE_repeating_decimal_sum_equality_l1390_139037


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1390_139011

theorem sqrt_sum_equality : Real.sqrt (11 + 6 * Real.sqrt 2) + Real.sqrt (11 - 6 * Real.sqrt 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1390_139011


namespace NUMINAMATH_CALUDE_compute_expression_l1390_139008

theorem compute_expression : 12 + 4 * (5 - 10 / 2)^3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1390_139008


namespace NUMINAMATH_CALUDE_problem_statement_l1390_139024

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : (12 : ℝ) ^ x = (18 : ℝ) ^ y) (h2 : (12 : ℝ) ^ x = 6 ^ (x * y)) :
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1390_139024


namespace NUMINAMATH_CALUDE_kenneth_earnings_l1390_139091

def earnings_problem (E : ℝ) : Prop :=
  let joystick := 0.10 * E
  let accessories := 0.15 * E
  let phone_bill := 0.05 * E
  let snacks := 0.20 * E - 25
  let utility := 0.25 * E - 15
  let remaining := 405
  E = joystick + accessories + phone_bill + snacks + utility + remaining

theorem kenneth_earnings : 
  ∃ E : ℝ, earnings_problem E ∧ E = 1460 :=
sorry

end NUMINAMATH_CALUDE_kenneth_earnings_l1390_139091


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1390_139052

theorem min_value_quadratic (x : ℝ) : 
  4 * x^2 + 8 * x + 16 ≥ 12 ∧ ∃ y : ℝ, 4 * y^2 + 8 * y + 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1390_139052


namespace NUMINAMATH_CALUDE_tangent_lines_max_area_and_slope_l1390_139099

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y - 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (-6, -5)

-- Define point N
def point_N : ℝ × ℝ := (1, 3)

-- Theorem for tangent lines
theorem tangent_lines :
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ x = -6) ∧
    (∀ x y, l₂ x y ↔ 3*x - 4*y - 2 = 0) ∧
    (∀ l, (∀ x y, l x y → circle_C x y) →
          (l (point_M.1) (point_M.2)) →
          (∃ x₀ y₀, circle_C x₀ y₀ ∧ l x₀ y₀ ∧
            ∀ x y, circle_C x y ∧ l x y → (x, y) = (x₀, y₀)) →
          (l = l₁ ∨ l = l₂)) :=
sorry

-- Theorem for maximum area and slope
theorem max_area_and_slope :
  ∃ (max_area : ℝ) (slope₁ slope₂ : ℝ),
    max_area = 8 ∧
    slope₁ = 2 * Real.sqrt 2 ∧
    slope₂ = -2 * Real.sqrt 2 ∧
    (∀ l : ℝ → ℝ → Prop,
      (l (point_N.1) (point_N.2)) →
      (∃ A B : ℝ × ℝ,
        circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
        l A.1 A.2 ∧ l B.1 B.2 ∧ A ≠ B) →
      (∃ C : ℝ × ℝ, C = point_N) →
      (∃ area : ℝ, area ≤ max_area) ∧
      (∃ k : ℝ, (k = slope₁ ∨ k = slope₂) →
        ∀ x y, l x y ↔ y - point_N.2 = k * (x - point_N.1))) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_max_area_and_slope_l1390_139099


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l1390_139039

theorem sarahs_bowling_score :
  ∀ (sarah_score greg_score : ℕ),
    sarah_score = greg_score + 50 →
    (sarah_score + greg_score) / 2 = 110 →
    sarah_score = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l1390_139039


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1390_139016

theorem binomial_coefficient_equality (p k : ℕ) (hp : Prime p) :
  ∃ n : ℕ, (n.choose p) = ((n + k).choose p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1390_139016


namespace NUMINAMATH_CALUDE_max_value_a_l1390_139074

theorem max_value_a (a b c e : ℕ+) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 5 * e)
  (h4 : e < 100) :
  a ≤ 2961 ∧ ∃ (a' b' c' e' : ℕ+), 
    a' = 2961 ∧ 
    a' < 2 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 5 * e' ∧ 
    e' < 100 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l1390_139074


namespace NUMINAMATH_CALUDE_percent_equation_l1390_139003

theorem percent_equation (x y : ℝ) (P : ℝ) 
  (h1 : 0.25 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.25 * x) : 
  P = 15 := by
  sorry

end NUMINAMATH_CALUDE_percent_equation_l1390_139003


namespace NUMINAMATH_CALUDE_coronavirus_diameter_scientific_notation_l1390_139044

theorem coronavirus_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000125 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.25 ∧ n = -7 :=
by sorry

end NUMINAMATH_CALUDE_coronavirus_diameter_scientific_notation_l1390_139044


namespace NUMINAMATH_CALUDE_f_properties_l1390_139054

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties :
  ∀ a : ℝ,
  (a = -1/2 → 
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), ∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≥ f a y) ∧
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), ∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≤ f a y) ∧
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = 1/2 + (Real.exp 1)^2/4) ∧
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = 5/4)) ∧
  ((a ≤ -1 → ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x > f a y) ∧
   (a ≥ 0 → ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x < f a y) ∧
   (-1 < a → a < 0 → 
     ∃ z : ℝ, 0 < z ∧ 
     (∀ x y : ℝ, 0 < x → x < y → y < z → f a x > f a y) ∧
     (∀ x y : ℝ, z ≤ x → x < y → f a x < f a y))) ∧
  (-1 < a → a < 0 → 
    (∀ x : ℝ, 0 < x → f a x > 1 + a / 2 * Real.log (-a)) ↔ 1/Real.exp 1 - 1 < a) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1390_139054


namespace NUMINAMATH_CALUDE_ping_pong_sum_of_products_l1390_139027

/-- The sum of products for n ping pong balls -/
def sum_of_products (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The number of ping pong balls -/
def num_balls : ℕ := 10

theorem ping_pong_sum_of_products :
  sum_of_products num_balls = 45 := by
  sorry


end NUMINAMATH_CALUDE_ping_pong_sum_of_products_l1390_139027


namespace NUMINAMATH_CALUDE_nabla_squared_l1390_139066

theorem nabla_squared (odot nabla : ℕ) : 
  odot < 20 ∧ nabla < 20 ∧ 
  odot ≠ nabla ∧ 
  odot > 0 ∧ nabla > 0 ∧
  nabla * nabla * odot = nabla → 
  nabla * nabla = 64 := by
  sorry

end NUMINAMATH_CALUDE_nabla_squared_l1390_139066


namespace NUMINAMATH_CALUDE_three_percent_difference_l1390_139012

theorem three_percent_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) (hy : 3 = 0.05 * y) : y - x = 40 := by
  sorry

end NUMINAMATH_CALUDE_three_percent_difference_l1390_139012


namespace NUMINAMATH_CALUDE_unique_fixed_point_l1390_139079

-- Define the plane
variable (Plane : Type)

-- Define the set of all lines in the plane
variable (S : Set (Set Plane))

-- Define the function f
variable (f : Set Plane → Plane)

-- Define the notion of a point being on a line
variable (on_line : Plane → Set Plane → Prop)

-- Define the notion of a line passing through a point
variable (passes_through : Set Plane → Plane → Prop)

-- Define the notion of points being on the same circle
variable (on_same_circle : Plane → Plane → Plane → Plane → Prop)

-- Main theorem
theorem unique_fixed_point
  (h1 : ∀ l ∈ S, on_line (f l) l)
  (h2 : ∀ (X : Plane) (l₁ l₂ l₃ : Set Plane),
        l₁ ∈ S → l₂ ∈ S → l₃ ∈ S →
        passes_through l₁ X → passes_through l₂ X → passes_through l₃ X →
        on_same_circle (f l₁) (f l₂) (f l₃) X) :
  ∃! P : Plane, ∀ l ∈ S, passes_through l P → f l = P :=
sorry

end NUMINAMATH_CALUDE_unique_fixed_point_l1390_139079


namespace NUMINAMATH_CALUDE_min_value_complex_ratio_l1390_139006

theorem min_value_complex_ratio (z : ℂ) (h : z.re ≠ 0) :
  ∃ (min : ℝ), min = -8 ∧ 
  (∀ (w : ℂ), w.re ≠ 0 → (w.re^4)⁻¹ * (w^4).re ≥ min) ∧
  (∃ (w : ℂ), w.re ≠ 0 ∧ (w.re^4)⁻¹ * (w^4).re = min) :=
sorry

end NUMINAMATH_CALUDE_min_value_complex_ratio_l1390_139006


namespace NUMINAMATH_CALUDE_probability_one_black_one_white_l1390_139009

def total_balls : ℕ := 6 + 2
def black_balls : ℕ := 6
def white_balls : ℕ := 2

theorem probability_one_black_one_white :
  let total_ways := Nat.choose total_balls 2
  let favorable_ways := black_balls * white_balls
  (favorable_ways : ℚ) / total_ways = 3 / 7 := by
    sorry

end NUMINAMATH_CALUDE_probability_one_black_one_white_l1390_139009


namespace NUMINAMATH_CALUDE_matt_current_age_l1390_139010

/-- Matt's current age -/
def matt_age : ℕ := sorry

/-- Kaylee's current age -/
def kaylee_age : ℕ := 8

theorem matt_current_age : matt_age = 5 := by
  have h1 : kaylee_age + 7 = 3 * matt_age := sorry
  sorry

end NUMINAMATH_CALUDE_matt_current_age_l1390_139010


namespace NUMINAMATH_CALUDE_similarity_of_triangles_l1390_139063

-- Define the points
variable (A B C D E F G H O : Point)

-- Define the cyclic quadrilateral ABCD
def is_cyclic_quad (A B C D : Point) : Prop := sorry

-- Define the circle centered at O passing through B and D
def circle_O_passes_through (O B D : Point) : Prop := sorry

-- Define that E and F are on lines BA and BC respectively
def E_on_BA (E B A : Point) : Prop := sorry
def F_on_BC (F B C : Point) : Prop := sorry

-- Define that E and F are distinct from A, B, C
def E_F_distinct (E F A B C : Point) : Prop := sorry

-- Define H as the orthocenter of triangle DEF
def H_orthocenter_DEF (H D E F : Point) : Prop := sorry

-- Define that AC, DO, and EF are concurrent
def lines_concurrent (A C D O E F : Point) : Prop := sorry

-- Define similarity of triangles
def triangles_similar (A B C E H F : Point) : Prop := sorry

-- Theorem statement
theorem similarity_of_triangles 
  (h1 : is_cyclic_quad A B C D)
  (h2 : circle_O_passes_through O B D)
  (h3 : E_on_BA E B A)
  (h4 : F_on_BC F B C)
  (h5 : E_F_distinct E F A B C)
  (h6 : H_orthocenter_DEF H D E F)
  (h7 : lines_concurrent A C D O E F) :
  triangles_similar A B C E H F :=
sorry

end NUMINAMATH_CALUDE_similarity_of_triangles_l1390_139063


namespace NUMINAMATH_CALUDE_race_remaining_distance_l1390_139088

/-- The remaining distance in a race with specific lead changes -/
def remaining_distance (total_length initial_even alex_lead1 max_lead alex_lead2 : ℕ) : ℕ :=
  total_length - (initial_even + alex_lead1 + max_lead + alex_lead2)

/-- Theorem stating the remaining distance in the specific race scenario -/
theorem race_remaining_distance :
  remaining_distance 5000 200 300 170 440 = 3890 := by
  sorry

end NUMINAMATH_CALUDE_race_remaining_distance_l1390_139088


namespace NUMINAMATH_CALUDE_exists_triangle_with_large_inner_triangle_l1390_139084

-- Define the structure of a triangle
structure Triangle :=
  (A B C : Point)

-- Define the properties of the triangle
def is_acute (t : Triangle) : Prop := sorry

-- Define the line segments
def median (t : Triangle) : Point → Point := sorry
def angle_bisector (t : Triangle) : Point → Point := sorry
def altitude (t : Triangle) : Point → Point := sorry

-- Define the intersection points
def intersection_points (t : Triangle) : Triangle := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- The main theorem
theorem exists_triangle_with_large_inner_triangle :
  ∃ (t : Triangle),
    is_acute t ∧
    area (intersection_points t) > 0.499 * area t :=
sorry

end NUMINAMATH_CALUDE_exists_triangle_with_large_inner_triangle_l1390_139084


namespace NUMINAMATH_CALUDE_volume_of_specific_pyramid_l1390_139060

/-- Regular quadrilateral pyramid with given properties -/
structure RegularQuadPyramid where
  -- Point P is on the height VO
  p_on_height : Bool
  -- P is equidistant from base and apex
  p_midpoint : Bool
  -- Distance from P to any side face
  dist_p_to_side : ℝ
  -- Distance from P to base
  dist_p_to_base : ℝ

/-- Volume of a regular quadrilateral pyramid -/
def volume (pyramid : RegularQuadPyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific pyramid -/
theorem volume_of_specific_pyramid :
  ∀ (pyramid : RegularQuadPyramid),
    pyramid.p_on_height ∧
    pyramid.p_midpoint ∧
    pyramid.dist_p_to_side = 3 ∧
    pyramid.dist_p_to_base = 5 →
    volume pyramid = 750 :=
by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_pyramid_l1390_139060


namespace NUMINAMATH_CALUDE_constant_function_value_l1390_139005

theorem constant_function_value (g : ℝ → ℝ) (h : ∀ x : ℝ, g x = -3) :
  ∀ x : ℝ, g (3 * x - 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_value_l1390_139005


namespace NUMINAMATH_CALUDE_some_number_value_l1390_139051

theorem some_number_value (x y : ℝ) 
  (h1 : x / y = 3 / 2)
  (h2 : (7 * x + y) / (x - y) = 23) :
  y = 1 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l1390_139051


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l1390_139057

def A : Set ℝ := {-1, 0, 1}
def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a = {0}) → a = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l1390_139057


namespace NUMINAMATH_CALUDE_frog_path_count_l1390_139046

-- Define the octagon and frog movement
def Octagon := Fin 8
def adjacent (v : Octagon) : Set Octagon := {w | (v.val + 1) % 8 = w.val ∨ (v.val + 7) % 8 = w.val}

-- Define the path count function
noncomputable def a (n : ℕ) : ℝ :=
  if n % 2 = 1 then 0
  else ((2 + Real.sqrt 2) ^ ((n / 2) - 1) - (2 - Real.sqrt 2) ^ ((n / 2) - 1)) / Real.sqrt 2

-- State the theorem
theorem frog_path_count :
  ∀ n : ℕ, a n = (if n % 2 = 1 then 0
              else ((2 + Real.sqrt 2) ^ ((n / 2) - 1) - (2 - Real.sqrt 2) ^ ((n / 2) - 1)) / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_frog_path_count_l1390_139046


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l1390_139021

theorem arithmetic_equalities :
  (187 / 12 - 63 / 12 - 52 / 12 = 6) ∧
  (321321 * 123 - 123123 * 321 = 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l1390_139021


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l1390_139023

theorem angle_in_second_quadrant : ∃ θ : Real, 
  θ = -10 * Real.pi / 3 ∧ 
  π / 2 < θ % (2 * π) ∧ 
  θ % (2 * π) < π :=
sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l1390_139023


namespace NUMINAMATH_CALUDE_quadrant_I_solution_range_l1390_139095

theorem quadrant_I_solution_range (c : ℝ) :
  (∃ x y : ℝ, x - y = 5 ∧ 2 * c * x + y = 8 ∧ x > 0 ∧ y > 0) ↔ -1/2 < c ∧ c < 4/5 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_range_l1390_139095


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_after_347_moves_l1390_139033

/-- Represents the positions around a pentagon -/
inductive PentagonPosition
| Top
| RightUpper
| RightLower
| LeftLower
| LeftUpper

/-- Represents the positions for the mouse, including edges -/
inductive MousePosition
| TopLeftEdge
| LeftUpperVertex
| LeftMiddleEdge
| LeftLowerVertex
| BottomEdge
| RightLowerVertex
| RightMiddleEdge
| RightUpperVertex
| TopRightEdge
| TopVertex

/-- Function to determine the cat's position after a given number of moves -/
def catPosition (moves : ℕ) : PentagonPosition :=
  match moves % 5 with
  | 0 => PentagonPosition.LeftUpper
  | 1 => PentagonPosition.Top
  | 2 => PentagonPosition.RightUpper
  | 3 => PentagonPosition.RightLower
  | _ => PentagonPosition.LeftLower

/-- Function to determine the mouse's position after a given number of moves -/
def mousePosition (moves : ℕ) : MousePosition :=
  match moves % 10 with
  | 0 => MousePosition.TopVertex
  | 1 => MousePosition.TopLeftEdge
  | 2 => MousePosition.LeftUpperVertex
  | 3 => MousePosition.LeftMiddleEdge
  | 4 => MousePosition.LeftLowerVertex
  | 5 => MousePosition.BottomEdge
  | 6 => MousePosition.RightLowerVertex
  | 7 => MousePosition.RightMiddleEdge
  | 8 => MousePosition.RightUpperVertex
  | _ => MousePosition.TopRightEdge

theorem cat_and_mouse_positions_after_347_moves :
  (catPosition 347 = PentagonPosition.RightUpper) ∧
  (mousePosition 347 = MousePosition.RightMiddleEdge) := by
  sorry


end NUMINAMATH_CALUDE_cat_and_mouse_positions_after_347_moves_l1390_139033


namespace NUMINAMATH_CALUDE_solve_walnuts_problem_l1390_139048

def walnuts_problem (initial_walnuts boy_gathered girl_gathered girl_ate final_walnuts : ℕ) : Prop :=
  ∃ (dropped : ℕ),
    initial_walnuts + boy_gathered - dropped + girl_gathered - girl_ate = final_walnuts

theorem solve_walnuts_problem :
  walnuts_problem 12 6 5 2 20 → 
  ∃ (dropped : ℕ), dropped = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_walnuts_problem_l1390_139048


namespace NUMINAMATH_CALUDE_max_weight_for_john_and_mike_l1390_139062

/-- The maximum weight the bench can support -/
def bench_max_weight : ℝ := 1000

/-- The safety margin for one person -/
def safety_margin_one : ℝ := 0.2

/-- The safety margin for two people -/
def safety_margin_two : ℝ := 0.3

/-- John's weight -/
def john_weight : ℝ := 250

/-- Mike's weight -/
def mike_weight : ℝ := 180

/-- Theorem: The maximum weight John and Mike can put on the bar when using the bench together is 270 pounds -/
theorem max_weight_for_john_and_mike : 
  bench_max_weight * (1 - safety_margin_two) - (john_weight + mike_weight) = 270 := by
  sorry

end NUMINAMATH_CALUDE_max_weight_for_john_and_mike_l1390_139062


namespace NUMINAMATH_CALUDE_marble_problem_l1390_139071

theorem marble_problem (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 30 →
  red = 9 →
  prob_red_or_white = 5 / 6 →
  ∃ (blue white : ℕ), blue + red + white = total ∧ 
                       (red + white : ℚ) / total = prob_red_or_white ∧
                       blue = 5 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l1390_139071


namespace NUMINAMATH_CALUDE_same_color_marble_probability_same_color_marble_probability_value_l1390_139031

/-- The probability of drawing three marbles of the same color from a bag containing
    5 red marbles, 7 white marbles, and 4 green marbles, without replacement. -/
theorem same_color_marble_probability : ℚ :=
  let total_marbles : ℕ := 5 + 7 + 4
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let green_marbles : ℕ := 4
  let prob_all_red : ℚ := (red_marbles * (red_marbles - 1) * (red_marbles - 2)) /
    (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  let prob_all_white : ℚ := (white_marbles * (white_marbles - 1) * (white_marbles - 2)) /
    (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  let prob_all_green : ℚ := (green_marbles * (green_marbles - 1) * (green_marbles - 2)) /
    (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  prob_all_red + prob_all_white + prob_all_green

theorem same_color_marble_probability_value :
  same_color_marble_probability = 43 / 280 := by
  sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_same_color_marble_probability_value_l1390_139031


namespace NUMINAMATH_CALUDE_coefficient_x3y3_l1390_139038

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (2x-y)^5
def expansion_term (r : ℕ) : ℤ := 
  (2^(5-r)) * ((-1)^r : ℤ) * (binomial 5 r)

-- Define the coefficient of x^3y^3 in (x+y)(2x-y)^5
def coefficient : ℤ := 
  expansion_term 3 + 2 * expansion_term 2

-- Theorem statement
theorem coefficient_x3y3 : coefficient = 40 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y3_l1390_139038


namespace NUMINAMATH_CALUDE_R_value_for_S_12_l1390_139070

theorem R_value_for_S_12 (g : ℝ) (R S : ℝ → ℝ) :
  (∀ x, R x = g * S x - 3) →
  R 5 = 17 →
  S 12 = 12 →
  R 12 = 45 := by
sorry

end NUMINAMATH_CALUDE_R_value_for_S_12_l1390_139070


namespace NUMINAMATH_CALUDE_sum_of_integers_50_to_75_l1390_139094

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_of_integers_50_to_75 : sum_of_integers 50 75 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_50_to_75_l1390_139094


namespace NUMINAMATH_CALUDE_old_supervisor_salary_is_870_l1390_139043

/-- Calculates the old supervisor's salary given the conditions of the problem -/
def old_supervisor_salary (num_workers : ℕ) (initial_avg : ℚ) (new_avg : ℚ) (new_supervisor_salary : ℚ) : ℚ :=
  (num_workers + 1) * initial_avg - (num_workers * new_avg + new_supervisor_salary)

/-- Theorem stating that the old supervisor's salary is $870 given the problem conditions -/
theorem old_supervisor_salary_is_870 :
  old_supervisor_salary 8 430 410 690 = 870 := by
  sorry

#eval old_supervisor_salary 8 430 410 690

end NUMINAMATH_CALUDE_old_supervisor_salary_is_870_l1390_139043


namespace NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l1390_139022

noncomputable def f (x : ℝ) := x + 3^(x + 2)

theorem f_has_unique_zero_in_interval :
  ∃! x, x ∈ Set.Ioo (-2 : ℝ) (-1) ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l1390_139022


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1390_139093

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem smallest_n_satisfying_conditions : 
  ∃ (n : ℕ), is_three_digit n ∧ 
             (9 ∣ (n + 6)) ∧ 
             (6 ∣ (n - 4)) ∧ 
             (∀ m, is_three_digit m → (9 ∣ (m + 6)) → (6 ∣ (m - 4)) → n ≤ m) ∧
             n = 112 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1390_139093


namespace NUMINAMATH_CALUDE_min_altitude_inequality_l1390_139059

/-- The minimum altitude of a triangle, or zero if the points are collinear -/
noncomputable def min_altitude (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The triangle inequality for minimum altitudes -/
theorem min_altitude_inequality (A B C X : ℝ × ℝ) :
  min_altitude A B C ≤ min_altitude A B X + min_altitude A X C + min_altitude X B C := by
  sorry

end NUMINAMATH_CALUDE_min_altitude_inequality_l1390_139059


namespace NUMINAMATH_CALUDE_descending_order_abc_l1390_139098

theorem descending_order_abc : 3^34 > 2^51 ∧ 2^51 > 4^25 := by
  sorry

end NUMINAMATH_CALUDE_descending_order_abc_l1390_139098


namespace NUMINAMATH_CALUDE_triangle_side_expression_l1390_139049

/-- Given a triangle with sides a, b, and c, prove that |a-b+c|-|c-a-b| = 2c-2b -/
theorem triangle_side_expression (a b c : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  |a - b + c| - |c - a - b| = 2 * c - 2 * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_expression_l1390_139049


namespace NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l1390_139065

theorem cos_36_minus_cos_72_eq_half : Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l1390_139065


namespace NUMINAMATH_CALUDE_solve_equation_l1390_139081

theorem solve_equation : ∃ x : ℝ, 2 * x = (26 - x) + 19 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1390_139081


namespace NUMINAMATH_CALUDE_simplify_quadratic_expression_l1390_139036

theorem simplify_quadratic_expression (a : ℝ) : -2 * a^2 + 4 * a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_quadratic_expression_l1390_139036


namespace NUMINAMATH_CALUDE_smallest_two_three_digit_multiples_sum_l1390_139041

/-- The smallest positive two-digit number -/
def smallest_two_digit : ℕ := 10

/-- The smallest positive three-digit number -/
def smallest_three_digit : ℕ := 100

/-- The smallest positive two-digit multiple of 5 -/
def c : ℕ := smallest_two_digit

/-- The smallest positive three-digit multiple of 7 -/
def d : ℕ := 
  (smallest_three_digit + 7 - 1) / 7 * 7

theorem smallest_two_three_digit_multiples_sum :
  c + d = 115 := by sorry

end NUMINAMATH_CALUDE_smallest_two_three_digit_multiples_sum_l1390_139041


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_correct_l1390_139047

/-- The smallest positive five-digit number divisible by 2, 3, 5, 7, and 11 -/
def smallest_five_digit_multiple : ℕ := 11550

/-- The five smallest prime numbers -/
def smallest_primes : List ℕ := [2, 3, 5, 7, 11]

theorem smallest_five_digit_multiple_correct :
  (∀ p ∈ smallest_primes, smallest_five_digit_multiple % p = 0) ∧
  smallest_five_digit_multiple ≥ 10000 ∧
  smallest_five_digit_multiple < 100000 ∧
  (∀ n : ℕ, n < smallest_five_digit_multiple →
    n < 10000 ∨ (∃ p ∈ smallest_primes, n % p ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_correct_l1390_139047


namespace NUMINAMATH_CALUDE_sqrt_24_simplification_l1390_139058

theorem sqrt_24_simplification : Real.sqrt 24 = 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_24_simplification_l1390_139058


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l1390_139032

/-- Represents the percentage of men who attended the company picnic -/
def percentage_men_attended : ℝ := 0.2

theorem company_picnic_attendance 
  (percent_women_attended : ℝ) 
  (percent_men_total : ℝ) 
  (percent_total_attended : ℝ) 
  (h1 : percent_women_attended = 0.4)
  (h2 : percent_men_total = 0.45)
  (h3 : percent_total_attended = 0.31000000000000007) :
  percentage_men_attended = 
    (percent_total_attended - (1 - percent_men_total) * percent_women_attended) / percent_men_total :=
by sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l1390_139032


namespace NUMINAMATH_CALUDE_parabola_directrix_l1390_139042

/-- Given a parabola with equation x^2 = (1/2)y, its directrix is y = -1/8 -/
theorem parabola_directrix (x y : ℝ) : 
  (x^2 = (1/2) * y) → (∃ p : ℝ, p = 1/4 ∧ y = -p/2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1390_139042


namespace NUMINAMATH_CALUDE_price_reduction_profit_l1390_139086

/-- Represents the daily sales and profit model for a product in a shopping mall -/
structure SalesModel where
  baseItems : ℕ  -- Base number of items sold per day
  baseProfit : ℕ  -- Base profit per item in yuan
  salesIncrease : ℕ  -- Additional items sold per yuan of price reduction
  priceReduction : ℕ  -- Amount of price reduction in yuan

/-- Calculates the daily profit given a SalesModel -/
def dailyProfit (model : SalesModel) : ℕ :=
  let newItems := model.baseItems + model.salesIncrease * model.priceReduction
  let newProfit := model.baseProfit - model.priceReduction
  newItems * newProfit

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 2100 yuan -/
theorem price_reduction_profit (model : SalesModel) 
  (h1 : model.baseItems = 30)
  (h2 : model.baseProfit = 50)
  (h3 : model.salesIncrease = 2)
  (h4 : model.priceReduction = 20) :
  dailyProfit model = 2100 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_profit_l1390_139086


namespace NUMINAMATH_CALUDE_triangle_iff_positive_f_l1390_139014

/-- The polynomial f(x, y, z) = (x + y + z)(-x + y + z)(x - y + z)(x + y - z) -/
def f (x y z : ℝ) : ℝ := (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

/-- A predicate to check if three numbers form a triangle -/
def is_triangle (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y > z ∧ y + z > x ∧ z + x > y

theorem triangle_iff_positive_f :
  ∀ x y z : ℝ, is_triangle (|x|) (|y|) (|z|) ↔ f x y z > 0 := by sorry

end NUMINAMATH_CALUDE_triangle_iff_positive_f_l1390_139014


namespace NUMINAMATH_CALUDE_circle_unique_dual_symmetry_l1390_139069

-- Define the shapes
inductive Shape
  | Parallelogram
  | Circle
  | EquilateralTriangle
  | RegularPentagon

-- Define symmetry properties
def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Circle => true
  | Shape.EquilateralTriangle => true
  | Shape.RegularPentagon => true
  | _ => false

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => true
  | Shape.Circle => true
  | _ => false

-- Theorem statement
theorem circle_unique_dual_symmetry :
  ∀ s : Shape, (isAxiallySymmetric s ∧ isCentrallySymmetric s) ↔ s = Shape.Circle :=
by sorry

end NUMINAMATH_CALUDE_circle_unique_dual_symmetry_l1390_139069


namespace NUMINAMATH_CALUDE_prime_sum_product_l1390_139068

theorem prime_sum_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 101 → p * q = 194 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_product_l1390_139068


namespace NUMINAMATH_CALUDE_max_expenditure_l1390_139072

def linear_regression (x : ℝ) (b a e : ℝ) : ℝ := b * x + a + e

theorem max_expenditure (x : ℝ) (e : ℝ) 
  (h1 : x = 10) 
  (h2 : |e| ≤ 0.5) : 
  linear_regression x 0.8 2 e ≤ 10.5 := by
  sorry

end NUMINAMATH_CALUDE_max_expenditure_l1390_139072


namespace NUMINAMATH_CALUDE_cuboid_diagonal_l1390_139040

theorem cuboid_diagonal (a b : ℤ) :
  (∃ c : ℕ+, ∃ d : ℕ+, d * d = a * a + b * b + c * c) ↔ 2 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_l1390_139040


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_property_l1390_139061

/-- A fourth-degree polynomial with real coefficients -/
def FourthDegreePolynomial : Type := ℝ → ℝ

/-- The property that |f(-2)| = |f(0)| = |f(1)| = |f(3)| = |f(4)| = 16 -/
def HasSpecifiedValues (f : FourthDegreePolynomial) : Prop :=
  |f (-2)| = 16 ∧ |f 0| = 16 ∧ |f 1| = 16 ∧ |f 3| = 16 ∧ |f 4| = 16

/-- The main theorem -/
theorem fourth_degree_polynomial_property (f : FourthDegreePolynomial) 
  (h : HasSpecifiedValues f) : |f 5| = 208 := by
  sorry


end NUMINAMATH_CALUDE_fourth_degree_polynomial_property_l1390_139061


namespace NUMINAMATH_CALUDE_target_shooting_problem_l1390_139025

theorem target_shooting_problem (p : ℝ) (k₀ : ℕ) (h_p : p = 0.7) (h_k₀ : k₀ = 16) :
  ∃ n : ℕ, (n = 22 ∨ n = 23) ∧ 
    (k₀ : ℝ) ≤ n * p + p ∧ 
    (k₀ : ℝ) ≥ n * p - (1 - p) :=
sorry

end NUMINAMATH_CALUDE_target_shooting_problem_l1390_139025


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l1390_139017

theorem smallest_positive_integer_congruence :
  ∃ (y : ℕ), y > 0 ∧ (56 * y + 8) % 26 = 6 % 26 ∧
  ∀ (z : ℕ), z > 0 ∧ (56 * z + 8) % 26 = 6 % 26 → y ≤ z :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l1390_139017


namespace NUMINAMATH_CALUDE_c_is_winner_l1390_139002

-- Define the candidates
inductive Candidate
| A
| B
| C

-- Define the election result
structure ElectionResult where
  total_voters : Nat
  votes : Candidate → Nat
  vote_count_valid : votes Candidate.A + votes Candidate.B + votes Candidate.C = total_voters

-- Define the winner selection rule
def is_winner (result : ElectionResult) (c : Candidate) : Prop :=
  ∀ other : Candidate, result.votes c ≥ result.votes other

-- Theorem statement
theorem c_is_winner (result : ElectionResult) 
  (h_total : result.total_voters = 30)
  (h_a : result.votes Candidate.A = 12)
  (h_b : result.votes Candidate.B = 3)
  (h_c : result.votes Candidate.C = 15) :
  is_winner result Candidate.C :=
by sorry

end NUMINAMATH_CALUDE_c_is_winner_l1390_139002


namespace NUMINAMATH_CALUDE_f_max_value_l1390_139078

open Real

-- Define the function f
def f (x : ℝ) := (3 + 2*x)^3 * (4 - x)^4

-- Define the interval
def I : Set ℝ := {x | -3/2 < x ∧ x < 4}

-- State the theorem
theorem f_max_value :
  ∃ (x_max : ℝ), x_max ∈ I ∧
  f x_max = 432 * (11/7)^7 ∧
  x_max = 6/7 ∧
  ∀ (x : ℝ), x ∈ I → f x ≤ f x_max :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l1390_139078


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1390_139055

theorem boys_to_girls_ratio (total_students girls : ℕ) 
  (h1 : total_students = 1040)
  (h2 : girls = 400) :
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1390_139055


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l1390_139029

/-- Given a triangle ABC with side lengths a and c, and angle B, 
    prove the length of side b and the area of the triangle. -/
theorem triangle_side_and_area 
  (a c : ℝ) 
  (B : ℝ) 
  (ha : a = 3 * Real.sqrt 3) 
  (hc : c = 2) 
  (hB : B = 150 * π / 180) : 
  ∃ (b S : ℝ), 
    b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
    b = 7 ∧
    S = (1/2) * a * c * Real.sin B ∧ 
    S = (3/2) * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_and_area_l1390_139029


namespace NUMINAMATH_CALUDE_functional_equation_properties_l1390_139089

/-- A function satisfying the given properties -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x, f x > 0) ∧ 
  (∀ a b, f a * f b = f (a + b))

/-- Main theorem stating the properties of f -/
theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (f 0 = 1) ∧
  (∀ a, f (-a) = 1 / f a) ∧
  (∀ a, f a = (f (3 * a)) ^ (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l1390_139089


namespace NUMINAMATH_CALUDE_olivia_checking_time_l1390_139001

def time_spent_checking (num_problems : ℕ) (time_per_problem : ℕ) (total_time : ℕ) : ℕ :=
  total_time - (num_problems * time_per_problem)

theorem olivia_checking_time :
  time_spent_checking 7 4 31 = 3 := by sorry

end NUMINAMATH_CALUDE_olivia_checking_time_l1390_139001


namespace NUMINAMATH_CALUDE_max_sum_perpendicular_distances_l1390_139090

-- Define a triangle with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a ≥ b
  h_bc : b ≥ c
  h_pos : 0 < c

-- Define the inradius of a triangle
def inradius (t : Triangle) : ℝ := sorry

-- Define the sum of perpendicular distances from a point to the sides of the triangle
def sum_perpendicular_distances (t : Triangle) (P : ℝ × ℝ) : ℝ := sorry

theorem max_sum_perpendicular_distances (t : Triangle) :
  ∀ P, sum_perpendicular_distances t P ≤ 2 * (inradius t) * (t.a + t.b + t.c) :=
sorry

end NUMINAMATH_CALUDE_max_sum_perpendicular_distances_l1390_139090


namespace NUMINAMATH_CALUDE_range_of_x_l1390_139077

theorem range_of_x (x : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 9/a + 1/b = 2 ∧ 
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' ≥ x^2 + 2*x)) → 
  -4 ≤ x ∧ x ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l1390_139077


namespace NUMINAMATH_CALUDE_max_intersection_area_theorem_l1390_139064

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- Represents the maximum possible intersection area of two rectangles --/
def max_intersection_area (r1 r2 : Rectangle) : ℕ :=
  min r1.width r2.width * min r1.height r2.height

theorem max_intersection_area_theorem (r1 r2 : Rectangle) :
  r1.width < r1.height →
  r2.width > r2.height →
  2011 < area r1 →
  area r1 < 2020 →
  2011 < area r2 →
  area r2 < 2020 →
  max_intersection_area r1 r2 ≤ 1764 ∧
  ∃ (r1' r2' : Rectangle),
    r1'.width < r1'.height ∧
    r2'.width > r2'.height ∧
    2011 < area r1' ∧
    area r1' < 2020 ∧
    2011 < area r2' ∧
    area r2' < 2020 ∧
    max_intersection_area r1' r2' = 1764 := by
  sorry

#check max_intersection_area_theorem

end NUMINAMATH_CALUDE_max_intersection_area_theorem_l1390_139064


namespace NUMINAMATH_CALUDE_triangle_area_is_11_over_2_l1390_139080

-- Define the vertices of the triangle
def D : ℝ × ℝ := (2, -3)
def E : ℝ × ℝ := (0, 4)
def F : ℝ × ℝ := (3, -1)

-- Define the function to calculate the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_is_11_over_2 : triangleArea D E F = 11 / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_11_over_2_l1390_139080
