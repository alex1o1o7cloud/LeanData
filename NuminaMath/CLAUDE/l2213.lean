import Mathlib

namespace NUMINAMATH_CALUDE_total_spending_calculation_l2213_221387

def shirt_price : ℝ := 13.04
def shirt_tax_rate : ℝ := 0.07
def jacket_price : ℝ := 12.27
def jacket_tax_rate : ℝ := 0.085
def scarf_price : ℝ := 7.90
def hat_price : ℝ := 9.13
def scarf_hat_tax_rate : ℝ := 0.065

def total_cost (price : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate)

theorem total_spending_calculation :
  total_cost shirt_price shirt_tax_rate +
  total_cost jacket_price jacket_tax_rate +
  total_cost scarf_price scarf_hat_tax_rate +
  total_cost hat_price scarf_hat_tax_rate =
  45.4027 := by sorry

end NUMINAMATH_CALUDE_total_spending_calculation_l2213_221387


namespace NUMINAMATH_CALUDE_test_examination_l2213_221357

/-- The number of boys examined in a test --/
def num_boys : ℕ := 50

/-- The number of girls examined in the test --/
def num_girls : ℕ := 100

/-- The percentage of boys who pass the test --/
def boys_pass_rate : ℚ := 1/2

/-- The percentage of girls who pass the test --/
def girls_pass_rate : ℚ := 2/5

/-- The percentage of total students who fail the test --/
def total_fail_rate : ℚ := 5667/10000

theorem test_examination :
  num_boys = 50 ∧
  (num_boys * (1 - boys_pass_rate) + num_girls * (1 - girls_pass_rate)) / (num_boys + num_girls) = total_fail_rate :=
by sorry

end NUMINAMATH_CALUDE_test_examination_l2213_221357


namespace NUMINAMATH_CALUDE_smallest_k_with_remainders_l2213_221320

theorem smallest_k_with_remainders (k : ℕ) : 
  k > 1 ∧ 
  k % 13 = 1 ∧ 
  k % 8 = 1 ∧ 
  k % 4 = 1 ∧
  (∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 8 = 1 ∧ m % 4 = 1 → k ≤ m) →
  k = 105 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainders_l2213_221320


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_final_probability_sum_l2213_221352

/-- Probability of a palindrome in a four-letter sequence -/
def letter_palindrome_prob : ℚ := 1 / 676

/-- Probability of a palindrome in a four-digit sequence -/
def digit_palindrome_prob : ℚ := 1 / 100

/-- Total number of possible license plates -/
def total_plates : ℕ := 26^4 * 10^4

/-- Number of favorable outcomes (license plates with at least one palindrome) -/
def favorable_outcomes : ℕ := 155

/-- Denominator of the final probability fraction -/
def prob_denominator : ℕ := 13520

theorem license_plate_palindrome_probability :
  (favorable_outcomes : ℚ) / prob_denominator =
  letter_palindrome_prob + digit_palindrome_prob - letter_palindrome_prob * digit_palindrome_prob :=
by sorry

theorem final_probability_sum :
  favorable_outcomes + prob_denominator = 13675 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_final_probability_sum_l2213_221352


namespace NUMINAMATH_CALUDE_lakeside_volleyball_club_players_l2213_221383

/-- The number of players in the Lakeside Volleyball Club -/
def num_players : ℕ := 80

/-- The cost of a pair of shoes in dollars -/
def shoe_cost : ℕ := 10

/-- The additional cost of a uniform compared to a pair of shoes in dollars -/
def uniform_additional_cost : ℕ := 15

/-- The total expenditure for all gear in dollars -/
def total_expenditure : ℕ := 5600

/-- Theorem stating that the number of players in the Lakeside Volleyball Club is 80 -/
theorem lakeside_volleyball_club_players :
  num_players = (total_expenditure / (2 * (shoe_cost + (shoe_cost + uniform_additional_cost)))) :=
by sorry

end NUMINAMATH_CALUDE_lakeside_volleyball_club_players_l2213_221383


namespace NUMINAMATH_CALUDE_smallest_number_l2213_221304

theorem smallest_number (a b c d : ℚ) (ha : a = 0) (hb : b = 5) (hc : c = -0.3) (hd : d = -1/3) :
  min a (min b (min c d)) = d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l2213_221304


namespace NUMINAMATH_CALUDE_vegan_nut_free_menu_fraction_l2213_221313

theorem vegan_nut_free_menu_fraction :
  let total_vegan_dishes : ℕ := 8
  let vegan_menu_fraction : ℚ := 1/4
  let nut_containing_vegan_dishes : ℕ := 5
  let nut_free_vegan_dishes : ℕ := total_vegan_dishes - nut_containing_vegan_dishes
  let nut_free_vegan_fraction : ℚ := nut_free_vegan_dishes / total_vegan_dishes
  nut_free_vegan_fraction * vegan_menu_fraction = 3/32 :=
by sorry

end NUMINAMATH_CALUDE_vegan_nut_free_menu_fraction_l2213_221313


namespace NUMINAMATH_CALUDE_philips_farm_l2213_221301

theorem philips_farm (cows ducks pigs : ℕ) : 
  ducks = (3 * cows) / 2 →
  pigs = (cows + ducks) / 5 →
  cows + ducks + pigs = 60 →
  cows = 20 := by
sorry

end NUMINAMATH_CALUDE_philips_farm_l2213_221301


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2213_221376

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * 3*x = 108 → x = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l2213_221376


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2213_221329

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (where a > 0, b > 0),
    if one of its asymptotes is tangent to the curve y = √(x - 1),
    then its eccentricity is √5/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y = b / a * x ∧ y = Real.sqrt (x - 1) ∧
   (∀ (x' y' : ℝ), y' = b / a * x' → y' ≠ Real.sqrt (x' - 1) ∨ (x' = x ∧ y' = y))) →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2213_221329


namespace NUMINAMATH_CALUDE_cupcakes_for_classes_l2213_221341

/-- The number of fourth-grade classes for which Jessa needs to make cupcakes -/
def num_fourth_grade_classes : ℕ := 3

/-- The number of students in each fourth-grade class -/
def students_per_fourth_grade : ℕ := 30

/-- The number of students in the P.E. class -/
def students_in_pe : ℕ := 50

/-- The total number of cupcakes Jessa needs to make -/
def total_cupcakes : ℕ := 140

theorem cupcakes_for_classes :
  num_fourth_grade_classes * students_per_fourth_grade + students_in_pe = total_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_for_classes_l2213_221341


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2213_221307

theorem coin_flip_probability (p : ℝ) (h : p = 1 / 2) :
  p * (1 - p) * (1 - p) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2213_221307


namespace NUMINAMATH_CALUDE_green_green_pairs_l2213_221385

/-- Represents the distribution of students and pairs in a math competition. -/
structure Competition :=
  (total_students : ℕ)
  (blue_shirts : ℕ)
  (green_shirts : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ)

/-- The main theorem about the number of green-green pairs in the competition. -/
theorem green_green_pairs (c : Competition) 
  (h1 : c.total_students = 150)
  (h2 : c.blue_shirts = 68)
  (h3 : c.green_shirts = 82)
  (h4 : c.total_pairs = 75)
  (h5 : c.blue_blue_pairs = 30)
  (h6 : c.total_students = c.blue_shirts + c.green_shirts)
  (h7 : c.total_students = 2 * c.total_pairs) :
  ∃ (green_green_pairs : ℕ), green_green_pairs = 37 ∧ 
    c.total_pairs = c.blue_blue_pairs + green_green_pairs + (c.blue_shirts - 2 * c.blue_blue_pairs) :=
sorry

end NUMINAMATH_CALUDE_green_green_pairs_l2213_221385


namespace NUMINAMATH_CALUDE_green_or_purple_probability_l2213_221308

/-- The probability of drawing a green or purple marble from a bag -/
theorem green_or_purple_probability 
  (green : ℕ) (purple : ℕ) (white : ℕ) 
  (h_green : green = 4) 
  (h_purple : purple = 3) 
  (h_white : white = 6) : 
  (green + purple : ℚ) / (green + purple + white) = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_green_or_purple_probability_l2213_221308


namespace NUMINAMATH_CALUDE_surviving_positions_32_l2213_221310

/-- Represents the selection process for an international exchange event. -/
def SelectionProcess (n : ℕ) : Prop :=
  n > 0 ∧ ∃ k, 2^k = n

/-- Represents a valid initial position in the selection process. -/
def ValidPosition (n : ℕ) (p : ℕ) : Prop :=
  1 ≤ p ∧ p ≤ n

/-- Represents a position that survives all elimination rounds. -/
def SurvivingPosition (n : ℕ) (p : ℕ) : Prop :=
  ValidPosition n p ∧ ∃ k, 2^k = p

/-- The main theorem stating that positions 16 and 32 are the only surviving positions in a 32-student selection process. -/
theorem surviving_positions_32 :
  SelectionProcess 32 →
  ∀ p, SurvivingPosition 32 p ↔ (p = 16 ∨ p = 32) :=
by sorry

end NUMINAMATH_CALUDE_surviving_positions_32_l2213_221310


namespace NUMINAMATH_CALUDE_third_term_5_4_l2213_221322

/-- Decomposition function that returns the n-th term in the decomposition of m^k -/
def decomposition (m : ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  2 * m * k - 1 + 2 * (n - 1)

/-- Theorem stating that the third term in the decomposition of 5^4 is 125 -/
theorem third_term_5_4 : decomposition 5 4 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_third_term_5_4_l2213_221322


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l2213_221374

theorem functional_equation_solutions (f : ℕ → ℕ) 
  (h : ∀ n m : ℕ, f (3 * n + 2 * m) = f n * f m) : 
  (∀ n, f n = 0) ∨ 
  (∀ n, f n = 1) ∨ 
  ((∀ n, n ≠ 0 → f n = 0) ∧ f 0 = 1) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l2213_221374


namespace NUMINAMATH_CALUDE_prob_heart_then_king_is_one_fiftytwo_l2213_221381

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of hearts in a standard deck
def hearts_in_deck : ℕ := 13

-- Define the number of kings in a standard deck
def kings_in_deck : ℕ := 4

-- Define the probability of drawing a heart first and a king second
def prob_heart_then_king : ℚ := hearts_in_deck / standard_deck * kings_in_deck / (standard_deck - 1)

-- Theorem statement
theorem prob_heart_then_king_is_one_fiftytwo :
  prob_heart_then_king = 1 / standard_deck :=
by sorry

end NUMINAMATH_CALUDE_prob_heart_then_king_is_one_fiftytwo_l2213_221381


namespace NUMINAMATH_CALUDE_smallest_multiple_of_2019_l2213_221342

/-- A number of the form abcabcabc... where a, b, and c are digits -/
def RepeatingDigitNumber (a b c : ℕ) : ℕ := 
  a * 100000000 + b * 10000000 + c * 1000000 +
  a * 100000 + b * 10000 + c * 1000 +
  a * 100 + b * 10 + c

/-- The smallest multiple of 2019 of the form abcabcabc... -/
def SmallestMultiple : ℕ := 673673673

theorem smallest_multiple_of_2019 :
  (∀ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 →
    RepeatingDigitNumber a b c % 2019 = 0 →
    RepeatingDigitNumber a b c ≥ SmallestMultiple) ∧
  SmallestMultiple % 2019 = 0 ∧
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧
    RepeatingDigitNumber a b c = SmallestMultiple :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_2019_l2213_221342


namespace NUMINAMATH_CALUDE_train_crossing_time_l2213_221348

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 1050 →
  train_speed_kmh = 126 →
  (2 * train_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2213_221348


namespace NUMINAMATH_CALUDE_highest_power_of_three_in_M_l2213_221311

def M : ℕ := sorry  -- Definition of M as concatenation of 2-digit integers from 10 to 81

theorem highest_power_of_three_in_M : 
  ∃ (k : ℕ), (3^2 ∣ M) ∧ ¬(3^(2+1) ∣ M) :=
sorry

end NUMINAMATH_CALUDE_highest_power_of_three_in_M_l2213_221311


namespace NUMINAMATH_CALUDE_circle_radius_comparison_l2213_221325

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of three circles being pairwise disjoint and collinear
def pairwiseDisjointCollinear (c1 c2 c3 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let (x3, y3) := c3.center
  (x2 - x1) * (y3 - y1) = (y2 - y1) * (x3 - x1) ∧ 
  (c1.radius + c2.radius < Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)) ∧
  (c2.radius + c3.radius < Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)) ∧
  (c3.radius + c1.radius < Real.sqrt ((x3 - x1)^2 + (y3 - y1)^2))

-- Define the property of a circle touching three other circles externally
def touchesExternally (c : Circle) (c1 c2 c3 : Circle) : Prop :=
  let (x, y) := c.center
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let (x3, y3) := c3.center
  (Real.sqrt ((x - x1)^2 + (y - y1)^2) = c.radius + c1.radius) ∧
  (Real.sqrt ((x - x2)^2 + (y - y2)^2) = c.radius + c2.radius) ∧
  (Real.sqrt ((x - x3)^2 + (y - y3)^2) = c.radius + c3.radius)

-- The main theorem
theorem circle_radius_comparison 
  (c1 c2 c3 c : Circle) 
  (h1 : pairwiseDisjointCollinear c1 c2 c3)
  (h2 : touchesExternally c c1 c2 c3) :
  c.radius > c2.radius :=
sorry

end NUMINAMATH_CALUDE_circle_radius_comparison_l2213_221325


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l2213_221356

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ 
                 a^2 + b^2 = r^2 ∧
                 (a - 7)^2 + (b - 7)^2 = r^2 ∧
                 b = 4/3 * a

-- Define the tangent line l
def tangent_line_l (x y : ℝ) : Prop :=
  (y = -3/4 * x) ∨ (x + y + 5 * Real.sqrt 2 - 7 = 0) ∨ (x + y - 5 * Real.sqrt 2 - 7 = 0)

theorem circle_and_tangent_line :
  (∀ x y, circle_C x y ↔ (x - 3)^2 + (y - 4)^2 = 25) ∧
  (∀ x y, tangent_line_l x y ↔ 
    ((x + y = 0 ∨ x = y) ∧ 
     ∃ (t : ℝ), (x - 3 + t)^2 + (y - 4 + 3/4 * t)^2 = 25 ∧
                ((x + t)^2 + (y + 3/4 * t)^2 > 25 ∨ (x - t)^2 + (y - 3/4 * t)^2 > 25))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l2213_221356


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l2213_221399

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l2213_221399


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l2213_221397

theorem binomial_coefficient_problem (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x, (2 + x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l2213_221397


namespace NUMINAMATH_CALUDE_notebook_cost_l2213_221396

theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (h1 : notebook_cost + pencil_cost = 2.20)
  (h2 : notebook_cost = pencil_cost + 2) : 
  notebook_cost = 2.10 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2213_221396


namespace NUMINAMATH_CALUDE_maple_trees_after_planting_l2213_221372

/-- The number of maple trees in the park after planting -/
def total_maple_trees (initial_maple_trees planted_maple_trees : ℕ) : ℕ :=
  initial_maple_trees + planted_maple_trees

/-- Theorem: The total number of maple trees after planting is equal to
    the sum of the initial number of maple trees and the number of maple trees being planted -/
theorem maple_trees_after_planting 
  (initial_maple_trees planted_maple_trees : ℕ) : 
  total_maple_trees initial_maple_trees planted_maple_trees = 
  initial_maple_trees + planted_maple_trees := by
  sorry

#eval total_maple_trees 2 9

end NUMINAMATH_CALUDE_maple_trees_after_planting_l2213_221372


namespace NUMINAMATH_CALUDE_max_earnings_l2213_221362

def max_work_hours : ℕ := 80
def regular_hours : ℕ := 20
def regular_wage : ℚ := 8
def regular_tips : ℚ := 2
def overtime_wage_multiplier : ℚ := 1.25
def overtime_tips : ℚ := 3
def bonus_per_5_hours : ℚ := 20

def overtime_hours : ℕ := max_work_hours - regular_hours

def regular_earnings : ℚ := regular_hours * (regular_wage + regular_tips)
def overtime_wage : ℚ := regular_wage * overtime_wage_multiplier
def overtime_earnings : ℚ := overtime_hours * (overtime_wage + overtime_tips)
def bonus : ℚ := (overtime_hours / 5) * bonus_per_5_hours

def total_earnings : ℚ := regular_earnings + overtime_earnings + bonus

theorem max_earnings :
  total_earnings = 1220 := by sorry

end NUMINAMATH_CALUDE_max_earnings_l2213_221362


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2213_221346

theorem decimal_to_fraction :
  (2.75 : ℚ) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2213_221346


namespace NUMINAMATH_CALUDE_prime_sum_divisible_by_six_l2213_221314

theorem prime_sum_divisible_by_six (p q r : Nat) : 
  Prime p → Prime q → Prime r → p > 3 → q > 3 → r > 3 → Prime (p + q + r) → 
  (6 ∣ p + q) ∨ (6 ∣ p + r) ∨ (6 ∣ q + r) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_divisible_by_six_l2213_221314


namespace NUMINAMATH_CALUDE_derivative_of_f_l2213_221394

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x => 2 * x - 1 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2213_221394


namespace NUMINAMATH_CALUDE_minimize_f_minimum_l2213_221395

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

/-- The theorem stating that 82/43 minimizes the minimum value of f(x) -/
theorem minimize_f_minimum (a : ℝ) :
  (∀ x, f (82/43) x ≤ f a x) → a = 82/43 := by
  sorry

end NUMINAMATH_CALUDE_minimize_f_minimum_l2213_221395


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2213_221370

theorem cubic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^3 - 6*x^2 + a*x - 6 = 0 ∧ x = 3) →
  (∃ x y : ℝ, x^3 - 6*x^2 + a*x - 6 = 0 ∧ y^3 - 6*y^2 + a*y - 6 = 0 ∧ x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2213_221370


namespace NUMINAMATH_CALUDE_max_constant_term_l2213_221367

def p₁ (a : ℝ) (x : ℝ) : ℝ := x - a

def p₂ (r s t : ℕ) (x : ℝ) : ℝ := (x - 1) ^ r * (x - 2) ^ s * (x + 3) ^ t

def constant_term (a : ℝ) (r s t : ℕ) : ℝ :=
  (-1) ^ (r + s) * 2 ^ s * 3 ^ t - a

theorem max_constant_term :
  ∀ a : ℝ, ∀ r s t : ℕ,
    r ≥ 1 → s ≥ 1 → t ≥ 1 → r + s + t = 4 →
    constant_term a r s t ≤ 21 ∧
    (constant_term (-3) 1 1 2 = 21) :=
sorry

end NUMINAMATH_CALUDE_max_constant_term_l2213_221367


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2213_221330

/-- A right triangle with medians m₁ and m₂ drawn from the vertices of the acute angles has hypotenuse of length 3√(336/13) when m₁ = 6 and m₂ = √48 -/
theorem right_triangle_hypotenuse (m₁ m₂ : ℝ) (h₁ : m₁ = 6) (h₂ : m₂ = Real.sqrt 48) :
  ∃ (a b c : ℝ), 
    a^2 + b^2 = c^2 ∧  -- right triangle condition
    (b^2 + (3*a/2)^2 = m₁^2) ∧  -- first median condition
    (a^2 + (3*b/2)^2 = m₂^2) ∧  -- second median condition
    c = 3 * Real.sqrt (336/13) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2213_221330


namespace NUMINAMATH_CALUDE_complex_coordinates_of_i_times_one_minus_i_l2213_221309

theorem complex_coordinates_of_i_times_one_minus_i :
  let i : ℂ := Complex.I
  (i * (1 - i)).re = 1 ∧ (i * (1 - i)).im = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_coordinates_of_i_times_one_minus_i_l2213_221309


namespace NUMINAMATH_CALUDE_range_of_a_a_upper_bound_range_of_a_characterization_l2213_221354

def A : Set ℝ := {x | x - 1 > 0}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) : (A ∩ B a).Nonempty → a > 1 := by sorry

theorem a_upper_bound : ∀ a : ℝ, a > 1 → (A ∩ B a).Nonempty := by sorry

theorem range_of_a_characterization :
  ∀ a : ℝ, (A ∩ B a).Nonempty ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_a_upper_bound_range_of_a_characterization_l2213_221354


namespace NUMINAMATH_CALUDE_dans_marbles_l2213_221323

/-- 
Given that Dan gave some marbles to Mary and has some marbles left,
prove that the initial number of marbles is equal to the sum of
the marbles given and the marbles left.
-/
theorem dans_marbles (marbles_given : ℕ) (marbles_left : ℕ) :
  marbles_given + marbles_left = 64 → marbles_given = 14 → marbles_left = 50 := by
  sorry

#check dans_marbles

end NUMINAMATH_CALUDE_dans_marbles_l2213_221323


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2213_221355

theorem quadratic_root_difference (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + 12 = 0 ∧ 
                x₂^2 + p*x₂ + 12 = 0 ∧ 
                x₁ - x₂ = 1) → 
  p = 7 ∨ p = -7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2213_221355


namespace NUMINAMATH_CALUDE_talent_show_gender_difference_l2213_221306

theorem talent_show_gender_difference (total : ℕ) (girls : ℕ) :
  total = 34 →
  girls = 28 →
  girls > total - girls →
  girls - (total - girls) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_talent_show_gender_difference_l2213_221306


namespace NUMINAMATH_CALUDE_value_of_M_l2213_221386

theorem value_of_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1500) ∧ (M = 4500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l2213_221386


namespace NUMINAMATH_CALUDE_jackets_sold_after_noon_l2213_221347

theorem jackets_sold_after_noon :
  let total_jackets : ℕ := 214
  let price_before_noon : ℚ := 31.95
  let price_after_noon : ℚ := 18.95
  let total_receipts : ℚ := 5108.30
  let jackets_after_noon : ℕ := 133
  let jackets_before_noon : ℕ := total_jackets - jackets_after_noon
  (jackets_before_noon : ℚ) * price_before_noon + (jackets_after_noon : ℚ) * price_after_noon = total_receipts :=
by
  sorry

#check jackets_sold_after_noon

end NUMINAMATH_CALUDE_jackets_sold_after_noon_l2213_221347


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_840_l2213_221390

theorem least_n_factorial_divisible_by_840 :
  ∀ n : ℕ, n > 0 → (n.factorial % 840 = 0) → n ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_840_l2213_221390


namespace NUMINAMATH_CALUDE_triangle_properties_l2213_221338

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) (h1 : (t.a + t.c) * Real.sin t.A = Real.sin t.A + Real.sin t.C)
    (h2 : t.c^2 + t.c = t.b^2 - 1) (h3 : t.a = 1) (h4 : t.c = 2) :
    t.B = 2 * Real.pi / 3 ∧ (1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2213_221338


namespace NUMINAMATH_CALUDE_solution_to_equation_l2213_221369

theorem solution_to_equation (x y : ℝ) : 
  (x - 7)^2 + (y - 8)^2 + (x - y)^2 = 1/3 ↔ x = 7 + 1/3 ∧ y = 7 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2213_221369


namespace NUMINAMATH_CALUDE_binomial_difference_squares_l2213_221327

theorem binomial_difference_squares (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅ + a₇)^2 = -2187 := by
sorry

end NUMINAMATH_CALUDE_binomial_difference_squares_l2213_221327


namespace NUMINAMATH_CALUDE_cost_of_one_each_l2213_221303

/-- Represents the cost of goods A, B, and C -/
structure GoodsCost where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The total cost of a combination of goods -/
def total_cost (g : GoodsCost) (a b c : ℝ) : ℝ :=
  a * g.A + b * g.B + c * g.C

theorem cost_of_one_each (g : GoodsCost) 
  (h1 : total_cost g 3 7 1 = 315)
  (h2 : total_cost g 4 10 1 = 420) :
  total_cost g 1 1 1 = 105 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_one_each_l2213_221303


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2213_221380

theorem complex_equation_solution (a : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = -2 * Complex.I → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2213_221380


namespace NUMINAMATH_CALUDE_interest_difference_theorem_l2213_221366

/-- The difference between compound and simple interest over 2 years at 10% per annum -/
def interestDifference (P : ℝ) : ℝ :=
  P * ((1 + 0.1)^2 - 1) - P * 0.1 * 2

/-- The problem statement -/
theorem interest_difference_theorem (P : ℝ) :
  interestDifference P = 18 → P = 1800 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_theorem_l2213_221366


namespace NUMINAMATH_CALUDE_brother_siblings_sibling_product_l2213_221351

/-- Represents a family with sisters and brothers -/
structure Family where
  sisters : Nat
  brothers : Nat

/-- Theorem: In a family where one sister has 4 sisters and 6 brothers,
    her brother has 5 sisters and 6 brothers -/
theorem brother_siblings (f : Family) (h : f.sisters = 5 ∧ f.brothers = 7) :
  ∃ (s b : Nat), s = 5 ∧ b = 6 := by
  sorry

/-- Corollary: The product of the number of sisters and brothers
    that the brother has is 30 -/
theorem sibling_product (f : Family) (h : f.sisters = 5 ∧ f.brothers = 7) :
  ∃ (s b : Nat), s * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_brother_siblings_sibling_product_l2213_221351


namespace NUMINAMATH_CALUDE_fuel_capacity_ratio_l2213_221315

theorem fuel_capacity_ratio (original_cost : ℝ) (price_increase : ℝ) (new_cost : ℝ) :
  original_cost = 200 →
  price_increase = 0.2 →
  new_cost = 480 →
  (new_cost / (original_cost * (1 + price_increase))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_fuel_capacity_ratio_l2213_221315


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l2213_221328

theorem complex_modulus_equality (n : ℝ) (hn : 0 < n) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 26 → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l2213_221328


namespace NUMINAMATH_CALUDE_mail_delivery_l2213_221317

theorem mail_delivery (total : ℕ) (johann : ℕ) (friends : ℕ) : 
  total = 180 → 
  johann = 98 → 
  friends = 2 → 
  (total - johann) % friends = 0 → 
  (total - johann) / friends = 41 :=
by sorry

end NUMINAMATH_CALUDE_mail_delivery_l2213_221317


namespace NUMINAMATH_CALUDE_dice_probability_l2213_221332

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The total number of possible outcomes when rolling seven dice -/
def total_outcomes : ℕ := num_sides ^ num_dice

/-- The number of ways to get exactly one pair with the other five dice all different -/
def one_pair_outcomes : ℕ := num_sides * (num_dice.choose 2) * (num_sides - 1) * (num_sides - 2) * (num_sides - 3) * (num_sides - 4) * (num_sides - 5)

/-- The number of ways to get exactly two pairs with the other three dice all different -/
def two_pairs_outcomes : ℕ := (num_sides.choose 2) * (num_dice.choose 2) * ((num_dice - 2).choose 2) * (num_sides - 2) * (num_sides - 3) * (num_sides - 4)

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ := one_pair_outcomes + two_pairs_outcomes

/-- The probability of getting at least one pair but no three of a kind when rolling seven standard six-sided dice -/
theorem dice_probability : (favorable_outcomes : ℚ) / total_outcomes = 315 / 972 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l2213_221332


namespace NUMINAMATH_CALUDE_susie_score_l2213_221334

/-- Calculates the total score in a math contest given the number of correct, incorrect, and unanswered questions. -/
def calculateScore (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℤ :=
  2 * (correct : ℤ) - (incorrect : ℤ)

/-- Theorem stating that Susie's score in the math contest is 20 points. -/
theorem susie_score : calculateScore 15 10 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_susie_score_l2213_221334


namespace NUMINAMATH_CALUDE_tea_trader_profit_percentage_l2213_221375

/-- Calculates the profit percentage for a tea trader --/
theorem tea_trader_profit_percentage
  (tea1_weight : ℝ)
  (tea1_cost : ℝ)
  (tea2_weight : ℝ)
  (tea2_cost : ℝ)
  (sale_price : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : sale_price = 20) :
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_weight := tea1_weight + tea2_weight
  let cost_per_kg := total_cost / total_weight
  let profit_per_kg := sale_price - cost_per_kg
  let profit_percentage := (profit_per_kg / cost_per_kg) * 100
  profit_percentage = 25 := by
sorry


end NUMINAMATH_CALUDE_tea_trader_profit_percentage_l2213_221375


namespace NUMINAMATH_CALUDE_max_sum_fourth_powers_l2213_221391

theorem max_sum_fourth_powers (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 16) :
  ∃ (M : ℝ), M = 64 ∧ a^4 + b^4 + c^4 + d^4 ≤ M ∧ 
  ∃ (a' b' c' d' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 = 16 ∧ a'^4 + b'^4 + c'^4 + d'^4 = M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_fourth_powers_l2213_221391


namespace NUMINAMATH_CALUDE_museum_paintings_l2213_221364

theorem museum_paintings (removed : ℕ) (remaining : ℕ) (initial : ℕ) : 
  removed = 3 → remaining = 95 → initial = remaining + removed → initial = 98 := by
  sorry

end NUMINAMATH_CALUDE_museum_paintings_l2213_221364


namespace NUMINAMATH_CALUDE_kishore_saved_ten_percent_l2213_221337

/-- Represents Mr. Kishore's financial situation --/
structure KishoreFinances where
  rent : ℕ
  milk : ℕ
  groceries : ℕ
  education : ℕ
  petrol : ℕ
  miscellaneous : ℕ
  savings : ℕ

/-- Calculates the total expenses --/
def totalExpenses (k : KishoreFinances) : ℕ :=
  k.rent + k.milk + k.groceries + k.education + k.petrol + k.miscellaneous

/-- Calculates the total monthly salary --/
def totalSalary (k : KishoreFinances) : ℕ :=
  totalExpenses k + k.savings

/-- Calculates the percentage saved --/
def percentageSaved (k : KishoreFinances) : ℚ :=
  (k.savings : ℚ) / (totalSalary k : ℚ) * 100

/-- Theorem: Mr. Kishore saved 10% of his monthly salary --/
theorem kishore_saved_ten_percent (k : KishoreFinances)
    (h1 : k.rent = 5000)
    (h2 : k.milk = 1500)
    (h3 : k.groceries = 4500)
    (h4 : k.education = 2500)
    (h5 : k.petrol = 2000)
    (h6 : k.miscellaneous = 3940)
    (h7 : k.savings = 2160) :
    percentageSaved k = 10 := by
  sorry

end NUMINAMATH_CALUDE_kishore_saved_ten_percent_l2213_221337


namespace NUMINAMATH_CALUDE_no_rational_solution_l2213_221316

theorem no_rational_solution : ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  1 / (x - y)^2 + 1 / (y - z)^2 + 1 / (z - x)^2 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l2213_221316


namespace NUMINAMATH_CALUDE_exists_permutation_equals_sixteen_l2213_221302

-- Define the set of operations
inductive Operation
  | Div : Operation
  | Add : Operation
  | Mul : Operation

-- Define a function to apply an operation
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Div => a / b
  | Operation.Add => a + b
  | Operation.Mul => a * b

-- Define a function to evaluate the expression given a permutation of operations
def evaluate (ops : List Operation) : ℚ :=
  match ops with
  | [op1, op2, op3] => applyOp op3 (applyOp op2 (applyOp op1 8 2) 3) 4
  | _ => 0  -- Invalid permutation

-- Theorem statement
theorem exists_permutation_equals_sixteen :
  ∃ (ops : List Operation),
    (ops.length = 3) ∧
    (Operation.Div ∈ ops) ∧
    (Operation.Add ∈ ops) ∧
    (Operation.Mul ∈ ops) ∧
    (evaluate ops = 16) := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_equals_sixteen_l2213_221302


namespace NUMINAMATH_CALUDE_expression_equality_l2213_221345

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 2 / y) :
  (x - 2 / x) * (y + 2 / y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2213_221345


namespace NUMINAMATH_CALUDE_rectangle_ratio_sum_l2213_221339

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Definition of the specific rectangle in the problem -/
def problemRectangle : Rectangle :=
  { A := ⟨0, 0⟩
  , B := ⟨6, 0⟩
  , C := ⟨6, 3⟩
  , D := ⟨0, 3⟩ }

/-- Point E on BC -/
def E : Point :=
  ⟨6, 1⟩

/-- Point F on CE -/
def F : Point :=
  ⟨6, 2⟩

/-- Theorem statement -/
theorem rectangle_ratio_sum (r s t : ℕ) :
  (r > 0 ∧ s > 0 ∧ t > 0) →
  (Nat.gcd r (Nat.gcd s t) = 1) →
  (∃ (P Q : Point),
    P.x = Q.x ∧ 
    P.y < Q.y ∧
    Q.y < problemRectangle.D.y ∧
    P.x > problemRectangle.A.x ∧
    P.x < problemRectangle.B.x ∧
    (P.x - problemRectangle.A.x) / (Q.x - P.x) = r / s ∧
    (Q.x - P.x) / (problemRectangle.B.x - Q.x) = s / t) →
  r + s + t = 20 := by
    sorry

end NUMINAMATH_CALUDE_rectangle_ratio_sum_l2213_221339


namespace NUMINAMATH_CALUDE_min_value_theorem_l2213_221392

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) :
  (1 / x + 1 / (3 * y)) ≥ 4 ∧ 
  ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ 
    Real.log 2 * x₀ + Real.log 8 * y₀ = Real.log 2 ∧
    1 / x₀ + 1 / (3 * y₀) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2213_221392


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l2213_221361

/-- Calculates the cost of tax-free items given total purchase, sales tax, and tax rate -/
def cost_of_tax_free_items (total_purchase : ℚ) (sales_tax : ℚ) (tax_rate : ℚ) : ℚ :=
  total_purchase - sales_tax / tax_rate

/-- Theorem stating that given the problem conditions, the cost of tax-free items is 20 -/
theorem tax_free_items_cost : 
  let total_purchase : ℚ := 25
  let sales_tax : ℚ := 30 / 100  -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 6 / 100    -- 6%
  cost_of_tax_free_items total_purchase sales_tax tax_rate = 20 := by
  sorry


end NUMINAMATH_CALUDE_tax_free_items_cost_l2213_221361


namespace NUMINAMATH_CALUDE_triangle_properties_l2213_221344

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The radius of the incircle of a triangle -/
def incircle_radius (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def triangle_area (t : Triangle) : ℝ := sorry

/-- Theorem: In triangle ABC, if a = 8 and the incircle radius is √3, then A = π/3 and the area is 11√3 -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 8) 
  (h2 : incircle_radius t = Real.sqrt 3) : 
  t.A = π/3 ∧ triangle_area t = 11 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2213_221344


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l2213_221360

theorem solution_set_of_equation (x y : ℝ) : 
  (|x*y| + |x - y + 1| = 0) ↔ ((x = 0 ∧ y = 1) ∨ (x = -1 ∧ y = 0)) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l2213_221360


namespace NUMINAMATH_CALUDE_three_positions_from_six_people_l2213_221319

def number_of_people : ℕ := 6
def number_of_positions : ℕ := 3

theorem three_positions_from_six_people :
  (number_of_people.factorial) / ((number_of_people - number_of_positions).factorial) = 120 :=
by sorry

end NUMINAMATH_CALUDE_three_positions_from_six_people_l2213_221319


namespace NUMINAMATH_CALUDE_chicken_egg_production_roberto_chicken_problem_l2213_221398

/-- Represents the problem of determining the number of eggs each chicken needs to produce per week -/
theorem chicken_egg_production (num_chickens : ℕ) (chicken_cost : ℚ) (weekly_feed_cost : ℚ) 
  (eggs_per_dozen : ℕ) (dozen_cost : ℚ) (weeks : ℕ) : ℚ :=
  let total_chicken_cost := num_chickens * chicken_cost
  let total_feed_cost := weeks * weekly_feed_cost
  let total_chicken_expenses := total_chicken_cost + total_feed_cost
  let total_egg_expenses := weeks * dozen_cost
  let eggs_per_week := eggs_per_dozen
  (eggs_per_week / num_chickens : ℚ)

/-- Proves that each chicken needs to produce 3 eggs per week to be cheaper than buying eggs after 81 weeks -/
theorem roberto_chicken_problem : 
  chicken_egg_production 4 20 1 12 2 81 = 3 := by
  sorry

end NUMINAMATH_CALUDE_chicken_egg_production_roberto_chicken_problem_l2213_221398


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l2213_221379

theorem absolute_value_equals_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_square_l2213_221379


namespace NUMINAMATH_CALUDE_decimal_point_shift_l2213_221358

theorem decimal_point_shift (x : ℝ) :
  (x * 10 = 760.8) → (x = 76.08) :=
by sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l2213_221358


namespace NUMINAMATH_CALUDE_range_of_m_l2213_221350

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x - floor x}

-- Define set B
def B (m : ℝ) : Set ℝ := {y | 0 ≤ y ∧ y ≤ m}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (A ⊂ B m) ↔ m ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2213_221350


namespace NUMINAMATH_CALUDE_complex_modulus_range_l2213_221321

theorem complex_modulus_range (a : ℝ) : 
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) ↔ 
  a ∈ Set.Icc (-(Real.sqrt 5) / 5) ((Real.sqrt 5) / 5) := by sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l2213_221321


namespace NUMINAMATH_CALUDE_investment_return_calculation_l2213_221335

theorem investment_return_calculation (total_investment : ℝ) (combined_return_rate : ℝ) 
  (investment_1 : ℝ) (return_rate_1 : ℝ) (investment_2 : ℝ) :
  total_investment = 2000 →
  combined_return_rate = 0.085 →
  investment_1 = 500 →
  return_rate_1 = 0.07 →
  investment_2 = 1500 →
  investment_1 + investment_2 = total_investment →
  investment_1 * return_rate_1 + investment_2 * ((total_investment * combined_return_rate - investment_1 * return_rate_1) / investment_2) = total_investment * combined_return_rate →
  (total_investment * combined_return_rate - investment_1 * return_rate_1) / investment_2 = 0.09 :=
by sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l2213_221335


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2213_221318

theorem root_sum_reciprocal (p : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁^2 - 6*p*x₁ + p^2 = 0)
  (h2 : x₂^2 - 6*p*x₂ + p^2 = 0)
  (h3 : x₁ ≠ x₂)
  (h4 : p ≠ 0) :
  1 / (x₁ + p) + 1 / (x₂ + p) = 1 / p :=
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2213_221318


namespace NUMINAMATH_CALUDE_market_equilibrium_change_l2213_221389

-- Define the demand and supply functions
def demand (p : ℝ) : ℝ := 150 - p
def supply (p : ℝ) : ℝ := 3 * p - 10

-- Define the new demand function after increase
def new_demand (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

-- Define the equilibrium condition
def is_equilibrium (p : ℝ) : Prop := demand p = supply p

-- Define the new equilibrium condition
def is_new_equilibrium (α : ℝ) (p : ℝ) : Prop := new_demand α p = supply p

-- State the theorem
theorem market_equilibrium_change (α : ℝ) :
  (∃ p₀ : ℝ, is_equilibrium p₀) →
  (∃ p₁ : ℝ, is_new_equilibrium α p₁ ∧ p₁ = 1.25 * p₀) →
  α = 1.4 := by sorry

end NUMINAMATH_CALUDE_market_equilibrium_change_l2213_221389


namespace NUMINAMATH_CALUDE_a_range_l2213_221368

/-- Given points A and B, if line AB is symmetric about x-axis and intersects a circle, then a is in [1/4, 2] -/
theorem a_range (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (a, 0)
  let line_AB_symmetric : Prop := ∃ (k : ℝ), ∀ (x y : ℝ), y = k * (x - a) ↔ -y = k * (x - (-2)) + 3
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 1}
  let line_AB_intersects_circle : Prop := ∃ (p : ℝ × ℝ), p ∈ circle ∧ ∃ (k : ℝ), p.2 - 0 = k * (p.1 - a)
  line_AB_symmetric → line_AB_intersects_circle → a ∈ Set.Icc (1/4 : ℝ) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_a_range_l2213_221368


namespace NUMINAMATH_CALUDE_exponent_simplification_l2213_221393

theorem exponent_simplification :
  (10 ^ 0.7) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ (-0.1)) * (10 ^ 0.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2213_221393


namespace NUMINAMATH_CALUDE_max_distance_to_line_family_l2213_221300

/-- The maximum distance from a point to a family of lines --/
theorem max_distance_to_line_family (a : ℝ) : 
  let P : ℝ × ℝ := (1, -1)
  let line := {(x, y) : ℝ × ℝ | a * x + 3 * y + 2 * a - 6 = 0}
  ∃ (Q : ℝ × ℝ), Q ∈ line ∧ 
    ∀ (R : ℝ × ℝ), R ∈ line → 
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  ∧ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 3 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_max_distance_to_line_family_l2213_221300


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_progression_l2213_221365

def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_progression
  (a : ℕ → ℝ)
  (h_ap : arithmetic_progression a)
  (h_sum : a 1 + a 2 + a 3 = 168)
  (h_diff : a 2 - a 5 = 42) :
  a 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_progression_l2213_221365


namespace NUMINAMATH_CALUDE_fraction_inequality_l2213_221377

theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / (a - c)^2 > e / (b - d)^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2213_221377


namespace NUMINAMATH_CALUDE_elevation_equals_depression_l2213_221388

/-- The elevation angle from point a to point b -/
def elevation_angle (a b : Point) : ℝ := sorry

/-- The depression angle from point b to point a -/
def depression_angle (b a : Point) : ℝ := sorry

/-- Theorem stating that the elevation angle from a to b equals the depression angle from b to a -/
theorem elevation_equals_depression (a b : Point) :
  elevation_angle a b = depression_angle b a := by sorry

end NUMINAMATH_CALUDE_elevation_equals_depression_l2213_221388


namespace NUMINAMATH_CALUDE_problem_solution_l2213_221324

def f (x : ℝ) := |x + 1| + |x - 2|
def g (a x : ℝ) := |x + 1| - |x - a| + a

theorem problem_solution :
  (∀ x : ℝ, f x ≤ 5 ↔ x ∈ Set.Icc (-2) 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2213_221324


namespace NUMINAMATH_CALUDE_triangle_side_bounds_l2213_221333

/-- Given a triangle ABC with side lengths a, b, c forming an arithmetic sequence
    and satisfying a² + b² + c² = 21, prove that √6 < b ≤ √7 -/
theorem triangle_side_bounds (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : ∃ d : ℝ, a = b - d ∧ c = b + d)  -- arithmetic sequence
  (h5 : a^2 + b^2 + c^2 = 21) :
  Real.sqrt 6 < b ∧ b ≤ Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_bounds_l2213_221333


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2213_221331

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0 ∧ (∀ y : ℝ, y^2 + y + m = 0 → y = x)) → m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2213_221331


namespace NUMINAMATH_CALUDE_distance_on_line_l2213_221343

/-- The distance between two points on a line --/
theorem distance_on_line (m k a b c d : ℝ) :
  b = m * a + k →
  d = m * c + k →
  Real.sqrt ((a - c)^2 + (b - d)^2) = |a - c| * Real.sqrt (1 + m^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l2213_221343


namespace NUMINAMATH_CALUDE_prime_square_divisibility_l2213_221312

theorem prime_square_divisibility (p : Nat) (h_prime : Prime p) (h_gt_3 : p > 3) :
  96 ∣ (4 * p^2 - 100) ↔ p^2 % 24 = 25 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_divisibility_l2213_221312


namespace NUMINAMATH_CALUDE_ball_cost_price_l2213_221384

theorem ball_cost_price (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) 
  (h1 : selling_price = 720)
  (h2 : num_balls_sold = 17)
  (h3 : num_balls_loss = 5) :
  ∃ (cost_price : ℕ), 
    cost_price * num_balls_sold - cost_price * num_balls_loss = selling_price ∧
    cost_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_ball_cost_price_l2213_221384


namespace NUMINAMATH_CALUDE_x_y_z_relation_l2213_221373

theorem x_y_z_relation (x y z : ℝ) : 
  x = 100.48 → 
  y = 100.48 → 
  x * z = y^2 → 
  z = 1 :=
by sorry

end NUMINAMATH_CALUDE_x_y_z_relation_l2213_221373


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_when_two_not_in_solution_set_l2213_221353

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - a|

-- Theorem for part I
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 2 ∧ ∃ y : ℝ, f 1 y = 2 :=
sorry

-- Theorem for part II
theorem range_of_a_when_two_not_in_solution_set :
  ∀ a : ℝ, (f a 2 > 5) ↔ (a < -5/2 ∨ a > 5/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_when_two_not_in_solution_set_l2213_221353


namespace NUMINAMATH_CALUDE_difference_of_squares_l2213_221378

theorem difference_of_squares : 550^2 - 450^2 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2213_221378


namespace NUMINAMATH_CALUDE_pentagon_sum_l2213_221326

/-- Given integers u and v with 0 < v < u, and points A, B, C, D, E defined as follows:
    A = (u, v)
    B is the reflection of A across y = -x
    C is the reflection of B across the y-axis
    D is the reflection of C across the x-axis
    E is the reflection of D across the y-axis
    If the area of pentagon ABCDE is 500, then u + v = 21 -/
theorem pentagon_sum (u v : ℤ) (hu : u > 0) (hv : v > 0) (huv : u > v)
  (harea : 6 * u * v - 2 * v^2 = 500) : u + v = 21 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_sum_l2213_221326


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l2213_221349

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 + Real.log x / Real.log 2

theorem zero_point_in_interval :
  ∃ c ∈ Set.Ioo 0 1, f c = 0 := by sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l2213_221349


namespace NUMINAMATH_CALUDE_intersection_equality_l2213_221382

theorem intersection_equality (m : ℝ) : 
  ({3, 4, m^2 - 3*m - 1} ∩ {2*m, -3} : Set ℝ) = {-3} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l2213_221382


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2213_221359

/-- Proves that the cost price of one meter of cloth is 140 Rs. given the conditions -/
theorem cloth_cost_price
  (total_length : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 30)
  (h2 : selling_price = 4500)
  (h3 : profit_per_meter = 10) :
  (selling_price - total_length * profit_per_meter) / total_length = 140 :=
by
  sorry

#check cloth_cost_price

end NUMINAMATH_CALUDE_cloth_cost_price_l2213_221359


namespace NUMINAMATH_CALUDE_joan_music_store_spending_l2213_221363

/-- The amount Joan spent at the music store -/
def total_spent (trumpet_cost music_tool_cost song_book_cost : ℚ) : ℚ :=
  trumpet_cost + music_tool_cost + song_book_cost

/-- Proof that Joan spent $163.28 at the music store -/
theorem joan_music_store_spending :
  total_spent 149.16 9.98 4.14 = 163.28 := by
  sorry

end NUMINAMATH_CALUDE_joan_music_store_spending_l2213_221363


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2213_221340

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) - 1
  f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2213_221340


namespace NUMINAMATH_CALUDE_trapezoid_diagonals_l2213_221336

/-- A trapezoid with bases a and c, legs b and d, and diagonals e and f. -/
structure Trapezoid (a c b d e f : ℝ) : Prop where
  positive_a : 0 < a
  positive_c : 0 < c
  positive_b : 0 < b
  positive_d : 0 < d
  positive_e : 0 < e
  positive_f : 0 < f
  a_greater_c : a > c

/-- The diagonals of a trapezoid can be expressed in terms of its sides. -/
theorem trapezoid_diagonals (a c b d e f : ℝ) (trap : Trapezoid a c b d e f) :
  e^2 = a * c + (a * d^2 - c * b^2) / (a - c) ∧
  f^2 = a * c + (a * b^2 - c * d^2) / (a - c) := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_diagonals_l2213_221336


namespace NUMINAMATH_CALUDE_point_on_hyperbola_l2213_221305

/-- A point (x, y) lies on the hyperbola y = -4/x if and only if xy = -4 -/
def lies_on_hyperbola (x y : ℝ) : Prop := x * y = -4

/-- The point (-2, 2) lies on the hyperbola y = -4/x -/
theorem point_on_hyperbola : lies_on_hyperbola (-2) 2 := by sorry

end NUMINAMATH_CALUDE_point_on_hyperbola_l2213_221305


namespace NUMINAMATH_CALUDE_initial_water_percentage_l2213_221371

theorem initial_water_percentage
  (V₁ : ℝ) (V₂ : ℝ) (P_f : ℝ)
  (h₁ : V₁ = 20)
  (h₂ : V₂ = 20)
  (h₃ : P_f = 5)
  : ∃ P_i : ℝ, P_i = 10 ∧ (P_i / 100) * V₁ = (P_f / 100) * (V₁ + V₂) := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l2213_221371
