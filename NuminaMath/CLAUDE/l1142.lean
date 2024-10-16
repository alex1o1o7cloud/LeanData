import Mathlib

namespace NUMINAMATH_CALUDE_card_collection_problem_l1142_114290

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of squares of the first n natural numbers -/
def sum_squares_first_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The average value of cards in a collection where each number k from 1 to n appears k times -/
def average_card_value (n : ℕ) : ℚ :=
  (sum_squares_first_n n : ℚ) / (sum_first_n n : ℚ)

theorem card_collection_problem :
  ∃ m : ℕ, average_card_value m = 56 ∧ m = 84 := by
  sorry

end NUMINAMATH_CALUDE_card_collection_problem_l1142_114290


namespace NUMINAMATH_CALUDE_min_value_theorem_l1142_114233

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_line : 6 * a + 3 * b = 1) :
  1 / (5 * a + 2 * b) + 2 / (a + b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1142_114233


namespace NUMINAMATH_CALUDE_johns_remaining_money_l1142_114228

/-- Calculates the remaining money for John after transactions --/
def remaining_money (initial_amount : ℚ) (sister_fraction : ℚ) (groceries_cost : ℚ) (gift_cost : ℚ) : ℚ :=
  initial_amount - (sister_fraction * initial_amount) - groceries_cost - gift_cost

/-- Theorem stating that John's remaining money is $11.67 --/
theorem johns_remaining_money :
  remaining_money 100 (1/3) 40 15 = 35/3 :=
by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l1142_114228


namespace NUMINAMATH_CALUDE_bottle_height_l1142_114208

/-- Represents a bottle composed of two cylinders -/
structure Bottle where
  r1 : ℝ  -- radius of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h_right : ℝ  -- water height when right side up
  h_upside : ℝ  -- water height when upside down

/-- The total height of the bottle -/
def total_height (b : Bottle) : ℝ :=
  29

/-- Theorem stating that the total height of the bottle is 29 cm -/
theorem bottle_height (b : Bottle) 
  (h_r1 : b.r1 = 1) 
  (h_r2 : b.r2 = 3) 
  (h_right : b.h_right = 20) 
  (h_upside : b.h_upside = 28) : 
  total_height b = 29 := by
  sorry

end NUMINAMATH_CALUDE_bottle_height_l1142_114208


namespace NUMINAMATH_CALUDE_inequality_solutions_l1142_114283

-- Define the inequalities
def ineq1a (x : ℝ) := 2*x + 8 > 5*x + 2
def ineq1b (x : ℝ) := 2*x + 8 + 4/(x-1) > 5*x + 2 + 4/(x-1)

def ineq2a (x : ℝ) := 2*x + 8 < 5*x + 2
def ineq2b (x : ℝ) := 2*x + 8 + 4/(x-1) < 5*x + 2 + 4/(x-1)

def ineq3a (x : ℝ) := 3/(x-1) > (x+2)/(x-2)
def ineq3b (x : ℝ) := 3/(x-1) + (3*x-4)/(x-1) > (x+2)/(x-2) + (3*x-4)/(x-1)

-- Define the theorem
theorem inequality_solutions :
  (∃ x : ℝ, ineq1a x ≠ ineq1b x) ∧
  (∀ x : ℝ, ineq2a x ↔ ineq2b x) ∧
  (∀ x : ℝ, x ≠ 1 → x ≠ 2 → (ineq3a x ↔ ineq3b x)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solutions_l1142_114283


namespace NUMINAMATH_CALUDE_total_stones_l1142_114230

/-- The number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- The conditions for the stone distribution -/
def validDistribution (p : StonePiles) : Prop :=
  p.pile5 = 6 * p.pile3 ∧
  p.pile2 = 2 * (p.pile3 + p.pile5) ∧
  p.pile1 = p.pile5 / 3 ∧
  p.pile1 = p.pile4 - 10 ∧
  p.pile4 = p.pile2 / 2

/-- The theorem stating that the total number of stones is 60 -/
theorem total_stones (p : StonePiles) (h : validDistribution p) : 
  p.pile1 + p.pile2 + p.pile3 + p.pile4 + p.pile5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stones_l1142_114230


namespace NUMINAMATH_CALUDE_kg_to_ton_conversion_min_to_hour_conversion_kg_to_g_conversion_l1142_114246

-- Define conversion rates
def kg_to_ton : ℝ := 1000
def min_to_hour : ℝ := 60
def kg_to_g : ℝ := 1000

-- Theorem statements
theorem kg_to_ton_conversion : 56 / kg_to_ton = 0.056 := by sorry

theorem min_to_hour_conversion : 45 / min_to_hour = 0.75 := by sorry

theorem kg_to_g_conversion : 0.3 * kg_to_g = 300 := by sorry

end NUMINAMATH_CALUDE_kg_to_ton_conversion_min_to_hour_conversion_kg_to_g_conversion_l1142_114246


namespace NUMINAMATH_CALUDE_system_solution_l1142_114258

theorem system_solution (x y k : ℚ) 
  (eq1 : 3 * x + 2 * y = k + 1)
  (eq2 : 2 * x + 3 * y = k)
  (sum_condition : x + y = 2) :
  k = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1142_114258


namespace NUMINAMATH_CALUDE_polygon_area_bounds_l1142_114264

-- Define the type for polygons
structure Polygon :=
  (vertices : List (Int × Int))
  (convex : Bool)
  (area : ℝ)

-- Define the theorem
theorem polygon_area_bounds :
  ∃ (a b c : ℝ) (α : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧
    (∀ n : ℕ, ∃ P : Polygon,
      P.convex = true ∧
      P.vertices.length = n ∧
      P.area < a * (n : ℝ)^3) ∧
    (∀ n : ℕ, ∀ P : Polygon,
      P.vertices.length = n →
      P.area ≥ b * (n : ℝ)^2) ∧
    (∀ n : ℕ, ∀ P : Polygon,
      P.vertices.length = n →
      P.area ≥ c * (n : ℝ)^(2 + α)) :=
sorry

end NUMINAMATH_CALUDE_polygon_area_bounds_l1142_114264


namespace NUMINAMATH_CALUDE_percentage_passed_both_subjects_l1142_114294

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 30)
  (h2 : failed_english = 42)
  (h3 : failed_both = 28) :
  100 - (failed_hindi + failed_english - failed_both) = 56 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_subjects_l1142_114294


namespace NUMINAMATH_CALUDE_triangle_properties_l1142_114239

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def given_triangle : Triangle where
  A := sorry
  B := sorry
  C := sorry
  a := 5
  b := 6
  c := sorry

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : t.a = 5)
  (h2 : t.b = 6)
  (h3 : Real.cos t.B = -4/5) :
  t.A = π/6 ∧ 
  t.c = -4 + 3 * Real.sqrt 3 ∧ 
  Real.sin (2 * t.B + t.A) = (7 - 24 * Real.sqrt 3) / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1142_114239


namespace NUMINAMATH_CALUDE_polygon_25_sides_diagonals_l1142_114280

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem polygon_25_sides_diagonals : num_diagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_polygon_25_sides_diagonals_l1142_114280


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1142_114256

theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 30)
  (area2 : w * h = 18)
  (area3 : l * h = 15) :
  l * w * h = 90 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1142_114256


namespace NUMINAMATH_CALUDE_prob_three_odd_dice_l1142_114257

def num_dice : ℕ := 5
def num_odd : ℕ := 3

theorem prob_three_odd_dice :
  (num_dice.choose num_odd : ℚ) * (1 / 2) ^ num_dice = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_dice_l1142_114257


namespace NUMINAMATH_CALUDE_factorization_equality_l1142_114214

theorem factorization_equality (x y : ℝ) : 3*x^2 + 6*x*y + 3*y^2 = 3*(x+y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1142_114214


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l1142_114243

theorem textbook_weight_difference :
  let chemistry_weight : ℝ := 7.125
  let geometry_weight : ℝ := 0.625
  chemistry_weight - geometry_weight = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l1142_114243


namespace NUMINAMATH_CALUDE_cos_double_angle_from_series_sum_l1142_114248

theorem cos_double_angle_from_series_sum (θ : ℝ) 
  (h : ∑' n, (Real.cos θ) ^ (2 * n) = 9) : 
  Real.cos (2 * θ) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_from_series_sum_l1142_114248


namespace NUMINAMATH_CALUDE_estimate_households_with_three_plus_houses_l1142_114204

/-- Estimate the number of households owning 3 or more houses -/
theorem estimate_households_with_three_plus_houses
  (total_households : ℕ)
  (ordinary_households : ℕ)
  (high_income_households : ℕ)
  (sample_ordinary : ℕ)
  (sample_high_income : ℕ)
  (sample_ordinary_with_three_plus : ℕ)
  (sample_high_income_with_three_plus : ℕ)
  (h1 : total_households = 100000)
  (h2 : ordinary_households = 99000)
  (h3 : high_income_households = 1000)
  (h4 : sample_ordinary = 990)
  (h5 : sample_high_income = 100)
  (h6 : sample_ordinary_with_three_plus = 50)
  (h7 : sample_high_income_with_three_plus = 70)
  (h8 : total_households = ordinary_households + high_income_households) :
  ⌊(sample_ordinary_with_three_plus : ℚ) / sample_ordinary * ordinary_households +
   (sample_high_income_with_three_plus : ℚ) / sample_high_income * high_income_households⌋ = 5700 :=
by sorry


end NUMINAMATH_CALUDE_estimate_households_with_three_plus_houses_l1142_114204


namespace NUMINAMATH_CALUDE_function_satisfying_condition_l1142_114247

theorem function_satisfying_condition (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, |f x - f y| = 2 * |x - y|) :
  ∃ c : ℝ, (∀ x : ℝ, f x = 2 * x + c) ∨ (∀ x : ℝ, f x = -2 * x + c) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_condition_l1142_114247


namespace NUMINAMATH_CALUDE_not_P_sufficient_for_not_q_l1142_114207

-- Define the propositions P and q
def P (x : ℝ) : Prop := |5*x - 2| > 3
def q (x : ℝ) : Prop := 1 / (x^2 + 4*x - 5) > 0

-- State the theorem
theorem not_P_sufficient_for_not_q :
  (∀ x : ℝ, ¬(P x) → ¬(q x)) ∧
  ¬(∀ x : ℝ, ¬(q x) → ¬(P x)) :=
sorry

end NUMINAMATH_CALUDE_not_P_sufficient_for_not_q_l1142_114207


namespace NUMINAMATH_CALUDE_problem_statement_l1142_114219

theorem problem_statement (a b c m n : ℝ) 
  (h1 : a - b = m) 
  (h2 : b - c = n) : 
  a^2 + b^2 + c^2 - a*b - b*c - c*a = m^2 + n^2 + m*n := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1142_114219


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l1142_114295

/-- Given that 1/3 of homes are termite-ridden and 4/7 of termite-ridden homes are collapsing,
    prove that 3/21 of homes are termite-ridden but not collapsing. -/
theorem termite_ridden_not_collapsing 
  (total_homes : ℕ) 
  (termite_ridden : ℕ) 
  (collapsing : ℕ) 
  (h1 : termite_ridden = total_homes / 3) 
  (h2 : collapsing = termite_ridden * 4 / 7) : 
  (termite_ridden - collapsing) = total_homes * 3 / 21 := by
  sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l1142_114295


namespace NUMINAMATH_CALUDE_x_range_theorem_l1142_114260

-- Define the condition from the original problem
def satisfies_equation (x y : ℝ) : Prop :=
  x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y)

-- Define the range of x
def x_range (x : ℝ) : Prop :=
  x ∈ Set.Icc 4 20 ∪ {0}

-- Theorem statement
theorem x_range_theorem :
  ∀ x y : ℝ, satisfies_equation x y → x_range x :=
by
  sorry

end NUMINAMATH_CALUDE_x_range_theorem_l1142_114260


namespace NUMINAMATH_CALUDE_modular_inverse_100_mod_101_l1142_114215

theorem modular_inverse_100_mod_101 : ∃ x : ℕ, x ≤ 100 ∧ (100 * x) % 101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_100_mod_101_l1142_114215


namespace NUMINAMATH_CALUDE_factorization_problems_l1142_114299

theorem factorization_problems :
  (∀ x y : ℝ, 6*x*y - 9*x^2*y = 3*x*y*(2-3*x)) ∧
  (∀ a : ℝ, (a^2+1)^2 - 4*a^2 = (a+1)^2*(a-1)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l1142_114299


namespace NUMINAMATH_CALUDE_chess_class_percentage_l1142_114221

/-- Proves that 20% of students attend chess class given the conditions of the problem -/
theorem chess_class_percentage (total_students : ℕ) (swimming_students : ℕ) 
  (h1 : total_students = 1000)
  (h2 : swimming_students = 20)
  (h3 : ∀ (chess_percentage : ℚ), 
    chess_percentage * total_students * (1/10) = swimming_students) :
  ∃ (chess_percentage : ℚ), chess_percentage = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_chess_class_percentage_l1142_114221


namespace NUMINAMATH_CALUDE_fraction_to_decimal_plus_two_l1142_114278

theorem fraction_to_decimal_plus_two : (7 : ℚ) / 16 + 2 = (2.4375 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_plus_two_l1142_114278


namespace NUMINAMATH_CALUDE_money_sum_l1142_114272

/-- Given three people A, B, and C with a total amount of money, prove the sum of A and C's money. -/
theorem money_sum (total money_B_C money_C : ℕ) 
  (h1 : total = 900)
  (h2 : money_B_C = 750)
  (h3 : money_C = 250) :
  ∃ (money_A : ℕ), money_A + money_C = 400 :=
by sorry

end NUMINAMATH_CALUDE_money_sum_l1142_114272


namespace NUMINAMATH_CALUDE_same_terminal_side_l1142_114206

-- Define a function to normalize angles to the range [0, 360)
def normalizeAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

-- Theorem stating that -390° and 330° have the same terminal side
theorem same_terminal_side : normalizeAngle (-390) = normalizeAngle 330 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l1142_114206


namespace NUMINAMATH_CALUDE_correct_system_l1142_114203

/-- Represents the money owned by person A -/
def money_A : ℝ := sorry

/-- Represents the money owned by person B -/
def money_B : ℝ := sorry

/-- Condition 1: If B gives half of his money to A, then A will have 50 units of money -/
axiom condition1 : money_A + (1/2 : ℝ) * money_B = 50

/-- Condition 2: If A gives two-thirds of his money to B, then B will have 50 units of money -/
axiom condition2 : (2/3 : ℝ) * money_A + money_B = 50

/-- The system of equations correctly represents the given conditions -/
theorem correct_system : 
  (money_A + (1/2 : ℝ) * money_B = 50) ∧ 
  ((2/3 : ℝ) * money_A + money_B = 50) := by sorry

end NUMINAMATH_CALUDE_correct_system_l1142_114203


namespace NUMINAMATH_CALUDE_ratio_is_two_l1142_114275

/-- Three integers a, b, and c where a < b < c and a = 0 -/
def IntegerTriple := {abc : ℤ × ℤ × ℤ // abc.1 < abc.2.1 ∧ abc.2.1 < abc.2.2 ∧ abc.1 = 0}

/-- Three integers p, q, r where p < q < r and r ≠ 0 -/
def GeometricTriple := {pqr : ℤ × ℤ × ℤ // pqr.1 < pqr.2.1 ∧ pqr.2.1 < pqr.2.2 ∧ pqr.2.2 ≠ 0}

/-- The mean of three integers is half the median -/
def MeanHalfMedian (abc : IntegerTriple) : Prop :=
  (abc.val.1 + abc.val.2.1 + abc.val.2.2) / 3 = abc.val.2.1 / 2

/-- The product of three integers is 0 -/
def ProductZero (abc : IntegerTriple) : Prop :=
  abc.val.1 * abc.val.2.1 * abc.val.2.2 = 0

/-- Three integers are in geometric progression -/
def GeometricProgression (pqr : GeometricTriple) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ r ≠ 1 ∧ pqr.val.2.1 = pqr.val.1 * r ∧ pqr.val.2.2 = pqr.val.2.1 * r

/-- Sum of squares equals square of sum -/
def SumSquaresEqualSquareSum (abc : IntegerTriple) (pqr : GeometricTriple) : Prop :=
  abc.val.1^2 + abc.val.2.1^2 + abc.val.2.2^2 = (pqr.val.1 + pqr.val.2.1 + pqr.val.2.2)^2

theorem ratio_is_two (abc : IntegerTriple) (pqr : GeometricTriple)
  (h1 : MeanHalfMedian abc)
  (h2 : ProductZero abc)
  (h3 : GeometricProgression pqr)
  (h4 : SumSquaresEqualSquareSum abc pqr) :
  abc.val.2.2 / abc.val.2.1 = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_is_two_l1142_114275


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l1142_114282

theorem prime_arithmetic_sequence_ones_digit (p q r s : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧
  p > 10 ∧
  q = p + 10 ∧ r = q + 10 ∧ s = r + 10 →
  p % 10 = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l1142_114282


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l1142_114252

theorem largest_n_with_unique_k : ∃ (k : ℤ), 
  (5 : ℚ)/12 < (7 : ℚ)/(7 + k) ∧ (7 : ℚ)/(7 + k) < 4/9 ∧ 
  (∀ (m : ℕ) (j : ℤ), m > 7 → 
    ((5 : ℚ)/12 < (m : ℚ)/(m + j) ∧ (m : ℚ)/(m + j) < 4/9) → 
    (∃ (l : ℤ), l ≠ j ∧ (5 : ℚ)/12 < (m : ℚ)/(m + l) ∧ (m : ℚ)/(m + l) < 4/9)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l1142_114252


namespace NUMINAMATH_CALUDE_cosine_period_proof_l1142_114266

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, and d are positive constants,
    and the graph covers three periods from 0 to 3π, prove that b = 2. -/
theorem cosine_period_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_period : (3 : ℝ) * (2 * π / b) = 3 * π) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_period_proof_l1142_114266


namespace NUMINAMATH_CALUDE_sequence_prime_value_l1142_114222

theorem sequence_prime_value (p : ℕ) (a : ℕ → ℤ) : 
  Prime p →
  a 0 = 0 →
  a 1 = 1 →
  (∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - p * a n) →
  (∃ m : ℕ, a m = -1) →
  p = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_prime_value_l1142_114222


namespace NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l1142_114236

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sequence_sum (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Theorem: The sum of the specific arithmetic sequence is 18599100 -/
theorem specific_arithmetic_sequence_sum :
  arithmetic_sequence_sum 2008 (-1776) 11 = 18599100 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l1142_114236


namespace NUMINAMATH_CALUDE_problem_statement_l1142_114287

def p : Prop := ∀ x : ℝ, x^2 - 1 ≥ -1

def q : Prop := 4 + 2 = 7

theorem problem_statement : 
  p ∧ ¬q ∧ ¬(p ∧ q) ∧ (p ∨ q) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1142_114287


namespace NUMINAMATH_CALUDE_ads_ratio_l1142_114211

def problem (ads_page1 ads_page2 ads_page3 ads_page4 ads_clicked : ℕ) : Prop :=
  ads_page1 = 12 ∧
  ads_page2 = 2 * ads_page1 ∧
  ads_page3 = ads_page2 + 24 ∧
  ads_page4 = (3 * ads_page2) / 4 ∧
  ads_clicked = 68

theorem ads_ratio (ads_page1 ads_page2 ads_page3 ads_page4 ads_clicked : ℕ) :
  problem ads_page1 ads_page2 ads_page3 ads_page4 ads_clicked →
  (ads_clicked : ℚ) / (ads_page1 + ads_page2 + ads_page3 + ads_page4 : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ads_ratio_l1142_114211


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_zero_l1142_114281

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_point_one_zero :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_zero_l1142_114281


namespace NUMINAMATH_CALUDE_integral_sin_over_square_l1142_114226

open Real MeasureTheory

/-- The definite integral of sin(x) / (1 + cos(x) + sin(x))^2 from 0 to π/2 equals ln(2) - 1/2 -/
theorem integral_sin_over_square : ∫ x in (0)..(π/2), sin x / (1 + cos x + sin x)^2 = log 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_over_square_l1142_114226


namespace NUMINAMATH_CALUDE_mary_nickels_problem_l1142_114284

theorem mary_nickels_problem (initial : ℕ) (given : ℕ) (total : ℕ) : 
  given = 5 → total = 12 → initial + given = total → initial = 7 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_problem_l1142_114284


namespace NUMINAMATH_CALUDE_max_a_value_l1142_114231

theorem max_a_value (a : ℤ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ x : ℤ, (2*x + 3 > 3*x - 1 ∧ 6*x - a ≥ 2*x + 2) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    (∀ x : ℤ, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → ¬(2*x + 3 > 3*x - 1 ∧ 6*x - a ≥ 2*x + 2))) →
  (∃ y : ℝ, y ≥ 0 ∧ (y + a)/(y - 1) + 2*a/(1 - y) = 2) →
  (∀ a' : ℤ, 
    (∃ (x₁' x₂' x₃' : ℤ), 
      (∀ x : ℤ, (2*x + 3 > 3*x - 1 ∧ 6*x - a' ≥ 2*x + 2) ↔ (x = x₁' ∨ x = x₂' ∨ x = x₃')) ∧
      (∀ x : ℤ, x ≠ x₁' ∧ x ≠ x₂' ∧ x ≠ x₃' → ¬(2*x + 3 > 3*x - 1 ∧ 6*x - a' ≥ 2*x + 2))) →
    (∃ y : ℝ, y ≥ 0 ∧ (y + a')/(y - 1) + 2*a'/(1 - y) = 2) →
    a' ≤ a) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_max_a_value_l1142_114231


namespace NUMINAMATH_CALUDE_path_area_is_775_l1142_114250

/-- Represents the dimensions and cost of a rectangular field with a surrounding path. -/
structure FieldWithPath where
  fieldLength : ℝ
  fieldWidth : ℝ
  pathWidth : ℝ
  pathCostPerSqm : ℝ
  totalPathCost : ℝ

/-- Calculates the area of the path surrounding a rectangular field. -/
def pathArea (f : FieldWithPath) : ℝ :=
  let totalLength := f.fieldLength + 2 * f.pathWidth
  let totalWidth := f.fieldWidth + 2 * f.pathWidth
  totalLength * totalWidth - f.fieldLength * f.fieldWidth

/-- Theorem stating that the area of the path is 775 sq m for the given field dimensions. -/
theorem path_area_is_775 (f : FieldWithPath)
  (h1 : f.fieldLength = 95)
  (h2 : f.fieldWidth = 55)
  (h3 : f.pathWidth = 2.5)
  (h4 : f.pathCostPerSqm = 2)
  (h5 : f.totalPathCost = 1550) :
  pathArea f = 775 := by
  sorry

end NUMINAMATH_CALUDE_path_area_is_775_l1142_114250


namespace NUMINAMATH_CALUDE_area_of_three_arc_region_sum_of_coefficients_l1142_114297

/-- The area of a region bounded by three circular arcs -/
theorem area_of_three_arc_region :
  let r : ℝ := 5  -- radius of each circle
  let θ : ℝ := π / 2  -- central angle of each arc (90 degrees in radians)
  let sector_area : ℝ := (θ / (2 * π)) * π * r^2  -- area of one sector
  let triangle_side : ℝ := r * Real.sqrt 2  -- side length of the equilateral triangle
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2  -- area of the equilateral triangle
  let region_area : ℝ := 3 * sector_area - triangle_area  -- area of the bounded region
  region_area = -125 * Real.sqrt 3 / 4 + 75 * π / 4 := by
    sorry

/-- The sum of coefficients in the area expression -/
theorem sum_of_coefficients :
  let a : ℝ := -125 / 4
  let b : ℝ := 3
  let c : ℝ := 75 / 4
  ⌊a + b + c⌋ = -9 := by
    sorry

end NUMINAMATH_CALUDE_area_of_three_arc_region_sum_of_coefficients_l1142_114297


namespace NUMINAMATH_CALUDE_unique_real_solution_l1142_114218

theorem unique_real_solution :
  ∃! x : ℝ, (x^12 + 1) * (x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 12 * x^11 :=
by sorry

end NUMINAMATH_CALUDE_unique_real_solution_l1142_114218


namespace NUMINAMATH_CALUDE_perpendicular_planes_condition_l1142_114253

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between lines
variable (perp_planes : Plane → Plane → Prop)
variable (perp_lines : Line → Line → Prop)

-- Define the relation for a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define the relation for two lines being different
variable (different : Line → Line → Prop)

-- Define the relation for two lines intersecting
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem perpendicular_planes_condition 
  (α β : Plane) (m n l₁ l₂ : Line)
  (h1 : in_plane m α)
  (h2 : in_plane n α)
  (h3 : different m n)
  (h4 : in_plane l₁ β)
  (h5 : in_plane l₂ β)
  (h6 : intersect l₁ l₂)
  (h7 : perp_lines m l₁)
  (h8 : perp_lines m l₂) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_condition_l1142_114253


namespace NUMINAMATH_CALUDE_original_number_is_seventeen_l1142_114288

theorem original_number_is_seventeen : 
  ∀ x : ℕ, 
  (∀ y : ℕ, y < 6 → ¬(23 ∣ (x + y))) → 
  (23 ∣ (x + 6)) → 
  x = 17 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_seventeen_l1142_114288


namespace NUMINAMATH_CALUDE_team_selection_ways_l1142_114245

def num_boys : ℕ := 10
def num_girls : ℕ := 12
def team_size : ℕ := 8
def boys_in_team : ℕ := 4
def girls_in_team : ℕ := 4

theorem team_selection_ways :
  (Nat.choose num_boys boys_in_team) * (Nat.choose num_girls girls_in_team) = 103950 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_ways_l1142_114245


namespace NUMINAMATH_CALUDE_log_product_sum_l1142_114209

theorem log_product_sum (a b : ℕ+) 
  (h1 : Real.log b / Real.log a = 3)
  (h2 : b - a = 858) : 
  a + b = 738 := by sorry

end NUMINAMATH_CALUDE_log_product_sum_l1142_114209


namespace NUMINAMATH_CALUDE_triangle_inequality_l1142_114201

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * (a^2 * b^2 + b^2 * c^2 + a^2 * c^2) > a^4 + b^4 + c^4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1142_114201


namespace NUMINAMATH_CALUDE_no_set_M_exists_l1142_114277

theorem no_set_M_exists : ¬ ∃ (M : Set ℕ),
  (∀ m : ℕ, m > 1 → ∃ a b : ℕ, a ∈ M ∧ b ∈ M ∧ a + b = m) ∧
  (∀ a b c d : ℕ, a ∈ M → b ∈ M → c ∈ M → d ∈ M → 
    a > 10 → b > 10 → c > 10 → d > 10 → 
    a + b = c + d → (a = c ∨ a = d)) :=
by sorry

end NUMINAMATH_CALUDE_no_set_M_exists_l1142_114277


namespace NUMINAMATH_CALUDE_visitors_scientific_notation_l1142_114271

-- Define 1.12 million
def visitors : ℝ := 1.12 * 1000000

-- Define scientific notation
def scientific_notation (x : ℝ) (base : ℝ) (exponent : ℤ) : Prop :=
  x = base * (10 : ℝ) ^ exponent ∧ 1 ≤ base ∧ base < 10

-- Theorem statement
theorem visitors_scientific_notation :
  scientific_notation visitors 1.12 6 := by
  sorry

end NUMINAMATH_CALUDE_visitors_scientific_notation_l1142_114271


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1142_114263

theorem a_plus_b_value (a b : ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 3) 
  (hab : |a-b| = -(a-b)) : 
  a + b = 5 ∨ a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1142_114263


namespace NUMINAMATH_CALUDE_rectangle_perpendicular_point_theorem_l1142_114229

/-- Given a rectangle ABCD with point E on diagonal BD such that AE is perpendicular to BD -/
structure RectangleWithPerpendicularPoint where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Distance from E to DC -/
  n : ℝ
  /-- Distance from E to BC -/
  EC : ℝ
  /-- Distance from E to AB -/
  x : ℝ
  /-- Length of diagonal BD -/
  d : ℝ
  /-- EC is 1 -/
  h_EC : EC = 1
  /-- ABCD is a rectangle -/
  h_rectangle : AB > 0 ∧ BC > 0
  /-- E is on diagonal BD -/
  h_E_on_BD : d > 0
  /-- AE is perpendicular to BD -/
  h_AE_perp_BD : True

/-- The main theorem about the rectangle with perpendicular point -/
theorem rectangle_perpendicular_point_theorem (r : RectangleWithPerpendicularPoint) :
  /- Part a -/
  (r.d - r.x * Real.sqrt (1 + r.x^2))^2 = r.x^4 * (1 + r.x^2) ∧
  /- Part b -/
  r.n = r.x^3 ∧
  /- Part c -/
  r.d^(2/3) - r.x^(2/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perpendicular_point_theorem_l1142_114229


namespace NUMINAMATH_CALUDE_cube_diagonal_length_l1142_114200

theorem cube_diagonal_length (s : ℝ) (h : s = 15) :
  let diagonal := Real.sqrt (3 * s^2)
  diagonal = 15 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_diagonal_length_l1142_114200


namespace NUMINAMATH_CALUDE_bicycle_trip_speed_l1142_114268

/-- The speed of the second part of a bicycle trip satisfies an equation based on given conditions. -/
theorem bicycle_trip_speed (v : ℝ) : v > 0 → 0.7 + 10 / v = 17 / 7.99 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_trip_speed_l1142_114268


namespace NUMINAMATH_CALUDE_davids_age_l1142_114255

theorem davids_age (david : ℕ) (yuan : ℕ) : 
  yuan = david + 7 → yuan = 2 * david → david = 7 := by
  sorry

end NUMINAMATH_CALUDE_davids_age_l1142_114255


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_144_l1142_114232

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_144_l1142_114232


namespace NUMINAMATH_CALUDE_paper_distribution_l1142_114262

theorem paper_distribution (total_students : ℕ) (total_sheets : ℕ) (leftover_sheets : ℕ)
  (h1 : total_students = 24)
  (h2 : total_sheets = 50)
  (h3 : leftover_sheets = 2)
  (h4 : ∃ (girls : ℕ), girls * 3 = total_students) :
  ∃ (girls : ℕ), girls * 3 = total_students ∧ 
    (total_sheets - leftover_sheets) / girls = 6 := by
  sorry

end NUMINAMATH_CALUDE_paper_distribution_l1142_114262


namespace NUMINAMATH_CALUDE_derivative_exp_derivative_polynomial_derivative_trig_l1142_114210

-- Function 1
theorem derivative_exp (x : ℝ) : 
  deriv (fun x => 2 * Real.exp x) x = 2 * Real.exp x := by sorry

-- Function 2
theorem derivative_polynomial (x : ℝ) : 
  deriv (fun x => 2 * x^5 - 3 * x^2 + 5 * x - 4) x = 10 * x^4 - 6 * x + 5 := by sorry

-- Function 3
theorem derivative_trig (x : ℝ) : 
  deriv (fun x => 3 * Real.cos x - 4 * Real.sin x) x = -3 * Real.sin x - 4 * Real.cos x := by sorry

end NUMINAMATH_CALUDE_derivative_exp_derivative_polynomial_derivative_trig_l1142_114210


namespace NUMINAMATH_CALUDE_kareem_largest_l1142_114235

def jose_calc (x : Int) : Int :=
  ((x - 2) * 3) + 5

def thuy_calc (x : Int) : Int :=
  (x * 3 - 2) + 5

def kareem_calc (x : Int) : Int :=
  ((x - 2) + 5) * 3

theorem kareem_largest (start : Int) :
  start = 15 →
  kareem_calc start > jose_calc start ∧
  kareem_calc start > thuy_calc start :=
by
  sorry

#eval jose_calc 15
#eval thuy_calc 15
#eval kareem_calc 15

end NUMINAMATH_CALUDE_kareem_largest_l1142_114235


namespace NUMINAMATH_CALUDE_triangle_to_square_area_ratio_l1142_114224

/-- The ratio of the area of a specific triangle to the area of a square -/
theorem triangle_to_square_area_ratio :
  let square_side : ℝ := 10
  let triangle_vertices : List (ℝ × ℝ) := [(2, 4), (4, 4), (4, 6)]
  let triangle_area := abs ((4 - 2) * (6 - 4) / 2)
  let square_area := square_side ^ 2
  triangle_area / square_area = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_to_square_area_ratio_l1142_114224


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l1142_114259

theorem similar_triangle_shortest_side 
  (a b c : ℝ) -- sides of the first triangle
  (k : ℝ) -- scaling factor
  (h1 : a^2 + b^2 = c^2) -- Pythagorean theorem for the first triangle
  (h2 : a ≤ b) -- a is the shortest side of the first triangle
  (h3 : c = 39) -- hypotenuse of the first triangle
  (h4 : a = 15) -- shortest side of the first triangle
  (h5 : k * c = 117) -- hypotenuse of the second triangle
  : k * a = 45 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l1142_114259


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1142_114296

/-- Given an ellipse with minimum distance 5 and maximum distance 15 from a point on the ellipse to a focus, 
    the length of its minor axis is 10√3. -/
theorem ellipse_minor_axis_length (min_dist max_dist : ℝ) (h1 : min_dist = 5) (h2 : max_dist = 15) :
  let a := (max_dist + min_dist) / 2
  let c := (max_dist - min_dist) / 2
  let b := Real.sqrt (a^2 - c^2)
  2 * b = 10 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1142_114296


namespace NUMINAMATH_CALUDE_sequence_minus_two_is_geometric_l1142_114238

/-- Given a sequence a and its partial sums s, prove {a n - 2} is geometric -/
theorem sequence_minus_two_is_geometric
  (a : ℕ+ → ℝ)  -- The sequence a_n
  (s : ℕ+ → ℝ)  -- The sequence of partial sums s_n
  (h : ∀ n : ℕ+, s n + a n = 2 * n)  -- The given condition
  : ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) - 2 = r * (a n - 2) :=
sorry

end NUMINAMATH_CALUDE_sequence_minus_two_is_geometric_l1142_114238


namespace NUMINAMATH_CALUDE_approximate_12000_accuracy_l1142_114242

/-- Represents an approximate number with its value and significant digits -/
structure ApproximateNumber where
  value : ℕ
  significantDigits : ℕ

/-- Determines the number of significant digits in an approximate number -/
def countSignificantDigits (n : ℕ) : ℕ :=
  sorry

theorem approximate_12000_accuracy :
  let n : ApproximateNumber := ⟨12000, countSignificantDigits 12000⟩
  n.significantDigits = 2 := by sorry

end NUMINAMATH_CALUDE_approximate_12000_accuracy_l1142_114242


namespace NUMINAMATH_CALUDE_solution_set_implies_b_power_a_l1142_114244

theorem solution_set_implies_b_power_a (a b : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 3) ↔ x^2 < a*x + b) → 
  b^a = 81 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_b_power_a_l1142_114244


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l1142_114225

-- Define the pentagon and its angles
structure Pentagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  G : ℝ

-- Define the theorem
theorem pentagon_angle_sum (p : Pentagon) 
  (h1 : p.A = 40)
  (h2 : p.F = p.G) : 
  p.B + p.D = 70 := by
  sorry

#check pentagon_angle_sum

end NUMINAMATH_CALUDE_pentagon_angle_sum_l1142_114225


namespace NUMINAMATH_CALUDE_cookie_tin_weight_is_9_l1142_114205

/-- The weight of a tin of cookies in ounces -/
def cookie_tin_weight (chip_bag_weight : ℕ) (num_chip_bags : ℕ) (cookie_tin_multiplier : ℕ) (total_weight_pounds : ℕ) : ℕ :=
  let total_weight_ounces : ℕ := total_weight_pounds * 16
  let total_chip_weight : ℕ := chip_bag_weight * num_chip_bags
  let num_cookie_tins : ℕ := num_chip_bags * cookie_tin_multiplier
  let total_cookie_weight : ℕ := total_weight_ounces - total_chip_weight
  total_cookie_weight / num_cookie_tins

/-- Theorem stating that a tin of cookies weighs 9 ounces under the given conditions -/
theorem cookie_tin_weight_is_9 :
  cookie_tin_weight 20 6 4 21 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookie_tin_weight_is_9_l1142_114205


namespace NUMINAMATH_CALUDE_largest_n_l1142_114269

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating the largest possible value of n -/
theorem largest_n : ∃ (x y : ℕ),
  x < 10 ∧ 
  y < 10 ∧ 
  x ≠ y ∧
  isPrime x ∧ 
  isPrime y ∧ 
  isPrime (10 * y + x) ∧
  1000 ≤ x * y * (10 * y + x) ∧ 
  x * y * (10 * y + x) < 10000 ∧
  ∀ (a b : ℕ), 
    a < 10 → 
    b < 10 → 
    a ≠ b →
    isPrime a → 
    isPrime b → 
    isPrime (10 * b + a) →
    1000 ≤ a * b * (10 * b + a) →
    a * b * (10 * b + a) < 10000 →
    a * b * (10 * b + a) ≤ x * y * (10 * y + x) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_l1142_114269


namespace NUMINAMATH_CALUDE_female_officers_count_female_officers_count_proof_l1142_114291

/-- The number of female officers on a police force, given:
  * 10% of female officers were on duty
  * 200 officers were on duty in total
  * Half of the officers on duty were female
-/
theorem female_officers_count : ℕ :=
  let total_on_duty : ℕ := 200
  let female_ratio_on_duty : ℚ := 1/2
  let female_on_duty_ratio : ℚ := 1/10
  1000

/-- Proof that the number of female officers is correct -/
theorem female_officers_count_proof :
  let total_on_duty : ℕ := 200
  let female_ratio_on_duty : ℚ := 1/2
  let female_on_duty_ratio : ℚ := 1/10
  female_officers_count = 1000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_female_officers_count_proof_l1142_114291


namespace NUMINAMATH_CALUDE_trapezoid_area_l1142_114265

theorem trapezoid_area (large_triangle_area small_triangle_area : ℝ)
  (num_trapezoids : ℕ) (h1 : large_triangle_area = 36)
  (h2 : small_triangle_area = 4) (h3 : num_trapezoids = 4) :
  (large_triangle_area - small_triangle_area) / num_trapezoids = 8 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1142_114265


namespace NUMINAMATH_CALUDE_perimeter_sum_equals_original_l1142_114213

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithIncircle where
  perimeter : ℝ
  incircle : Set ℝ × ℝ

/-- Represents a triangle cut off from the original triangle -/
structure CutOffTriangle where
  perimeter : ℝ
  touchesIncircle : Bool

/-- The theorem stating that the perimeter of the original triangle
    is equal to the sum of the perimeters of the cut-off triangles -/
theorem perimeter_sum_equals_original
  (original : TriangleWithIncircle)
  (cutoff1 cutoff2 cutoff3 : CutOffTriangle)
  (h1 : cutoff1.touchesIncircle = true)
  (h2 : cutoff2.touchesIncircle = true)
  (h3 : cutoff3.touchesIncircle = true) :
  original.perimeter = cutoff1.perimeter + cutoff2.perimeter + cutoff3.perimeter :=
sorry

end NUMINAMATH_CALUDE_perimeter_sum_equals_original_l1142_114213


namespace NUMINAMATH_CALUDE_income_percentage_increase_l1142_114292

/-- Calculates the percentage increase in monthly income given initial and new weekly incomes -/
theorem income_percentage_increase 
  (initial_job_income initial_freelance_income : ℚ)
  (new_job_income new_freelance_income : ℚ)
  (weeks_per_month : ℕ)
  (h1 : initial_job_income = 60)
  (h2 : initial_freelance_income = 40)
  (h3 : new_job_income = 120)
  (h4 : new_freelance_income = 60)
  (h5 : weeks_per_month = 4) :
  let initial_monthly_income := (initial_job_income + initial_freelance_income) * weeks_per_month
  let new_monthly_income := (new_job_income + new_freelance_income) * weeks_per_month
  (new_monthly_income - initial_monthly_income) / initial_monthly_income * 100 = 80 :=
by sorry


end NUMINAMATH_CALUDE_income_percentage_increase_l1142_114292


namespace NUMINAMATH_CALUDE_fill_box_with_cubes_l1142_114227

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.depth

/-- Finds the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Calculates the side length of the largest cube that can fit evenly into the box -/
def largestCubeSideLength (d : BoxDimensions) : ℕ :=
  gcd3 d.length d.width d.depth

/-- Calculates the number of cubes needed to fill the box completely -/
def numberOfCubes (d : BoxDimensions) : ℕ :=
  boxVolume d / (largestCubeSideLength d)^3

/-- The main theorem stating that 80 cubes are needed to fill the given box -/
theorem fill_box_with_cubes (d : BoxDimensions) 
  (h1 : d.length = 30) (h2 : d.width = 48) (h3 : d.depth = 12) : 
  numberOfCubes d = 80 := by
  sorry

end NUMINAMATH_CALUDE_fill_box_with_cubes_l1142_114227


namespace NUMINAMATH_CALUDE_find_m_value_l1142_114217

theorem find_m_value (α : Real) (m : Real) :
  let P : Real × Real := (-8 * m, -6 * Real.sin (30 * π / 180))
  (∃ (r : Real), r > 0 ∧ P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α) →
  Real.cos α = -4/5 →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l1142_114217


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l1142_114249

theorem correct_mean_calculation (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 20 ∧ original_mean = 150 ∧ incorrect_value = 135 ∧ correct_value = 160 →
  (n : ℚ) * original_mean - incorrect_value + correct_value = n * (151.25 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l1142_114249


namespace NUMINAMATH_CALUDE_scale_division_l1142_114251

/-- Given a scale of length 198 inches divided into 8 equal parts, 
    prove that the length of each part is 24.75 inches. -/
theorem scale_division (total_length : ℝ) (num_parts : ℕ) 
  (h1 : total_length = 198) 
  (h2 : num_parts = 8) :
  total_length / num_parts = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l1142_114251


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1142_114273

theorem fraction_unchanged (x y : ℝ) (h : x + y ≠ 0) :
  (3 * (2 * y)) / (2 * x + 2 * y) = (3 * y) / (x + y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1142_114273


namespace NUMINAMATH_CALUDE_max_value_sum_of_squares_l1142_114261

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem max_value_sum_of_squares (u v w : V) 
  (hu : ‖u‖ = 3) (hv : ‖v‖ = 1) (hw : ‖w‖ = 2) : 
  ‖u - 3 • v‖^2 + ‖v - 3 • w‖^2 + ‖w - 3 • u‖^2 ≤ 224 := by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_squares_l1142_114261


namespace NUMINAMATH_CALUDE_hyperbola_line_slope_l1142_114298

/-- Given two points on a hyperbola with a specific midpoint, prove that the slope of the line connecting them is 9/4 -/
theorem hyperbola_line_slope (A B : ℝ × ℝ) : 
  (A.1^2 - A.2^2/9 = 1) →  -- A is on the hyperbola
  (B.1^2 - B.2^2/9 = 1) →  -- B is on the hyperbola
  ((A.1 + B.1)/2 = -1) →   -- x-coordinate of midpoint
  ((A.2 + B.2)/2 = -4) →   -- y-coordinate of midpoint
  (B.2 - A.2)/(B.1 - A.1) = 9/4 :=  -- slope of line AB
by sorry

end NUMINAMATH_CALUDE_hyperbola_line_slope_l1142_114298


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1142_114289

/-- The equation (x+5)(x+2) = m + 3x has exactly one real solution if and only if m = 6 -/
theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1142_114289


namespace NUMINAMATH_CALUDE_tyrones_dimes_l1142_114220

/-- Given Tyrone's coin collection and total money, prove the number of dimes he has. -/
theorem tyrones_dimes (value_without_dimes : ℚ) (total_value : ℚ) (dime_value : ℚ) :
  value_without_dimes = 11 →
  total_value = 13 →
  dime_value = 1 / 10 →
  (total_value - value_without_dimes) / dime_value = 20 :=
by sorry

end NUMINAMATH_CALUDE_tyrones_dimes_l1142_114220


namespace NUMINAMATH_CALUDE_max_value_trigonometric_function_l1142_114285

open Real

theorem max_value_trigonometric_function :
  ∃ (max : ℝ), max = 6 - 4 * Real.sqrt 2 ∧
  ∀ θ : ℝ, θ ∈ Set.Ioo 0 (π / 2) →
    (2 * sin θ * cos θ) / ((sin θ + 1) * (cos θ + 1)) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_trigonometric_function_l1142_114285


namespace NUMINAMATH_CALUDE_ratio_of_segments_l1142_114270

/-- Given four points A, B, C, and D on a line in that order, with AB = 4, BC = 3, and AD = 20,
    prove that the ratio of AC to BD is 7/16. -/
theorem ratio_of_segments (A B C D : ℝ) : 
  A < B ∧ B < C ∧ C < D → -- Points lie on a line in order
  B - A = 4 →             -- AB = 4
  C - B = 3 →             -- BC = 3
  D - A = 20 →            -- AD = 20
  (C - A) / (D - B) = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l1142_114270


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_fourth_l1142_114216

theorem cos_alpha_minus_pi_fourth (α : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.tan (α + π/4) = -3) : 
  Real.cos (α - π/4) = 3 * Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_fourth_l1142_114216


namespace NUMINAMATH_CALUDE_v_3_equals_262_l1142_114202

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

/-- The value of x -/
def x : ℝ := 3

/-- The value of v_3 using Horner's method for the first three terms -/
def v_3 : ℝ := ((7*x + 6)*x + 5)*x + 4

/-- Theorem stating that v_3 equals 262 -/
theorem v_3_equals_262 : v_3 = 262 := by
  sorry

end NUMINAMATH_CALUDE_v_3_equals_262_l1142_114202


namespace NUMINAMATH_CALUDE_current_speed_l1142_114286

/-- The speed of the current given a motorboat's constant speed and trip times -/
theorem current_speed (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) 
  (h1 : boat_speed = 30)
  (h2 : upstream_time = 40 / 60)
  (h3 : downstream_time = 25 / 60) :
  ∃ c : ℝ, c = 90 / 13 ∧ 
  (boat_speed - c) * upstream_time = (boat_speed + c) * downstream_time :=
sorry

end NUMINAMATH_CALUDE_current_speed_l1142_114286


namespace NUMINAMATH_CALUDE_completing_square_sum_l1142_114254

theorem completing_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 9 = 0 ↔ (x + b)^2 = c) → b + c = -3 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l1142_114254


namespace NUMINAMATH_CALUDE_machine_working_time_yesterday_l1142_114276

/-- The total working time of an industrial machine, including downtime -/
def total_working_time (shirts_produced : ℕ) (production_rate : ℕ) (downtime : ℕ) : ℕ :=
  shirts_produced * production_rate + downtime

/-- Proof that the machine worked for 38 minutes yesterday -/
theorem machine_working_time_yesterday :
  total_working_time 9 2 20 = 38 := by
  sorry

end NUMINAMATH_CALUDE_machine_working_time_yesterday_l1142_114276


namespace NUMINAMATH_CALUDE_marble_ratio_theorem_l1142_114234

/-- Represents the number of marbles Elsa has at different points in the day -/
structure MarbleCount where
  initial : ℕ
  after_breakfast : ℕ
  after_lunch : ℕ
  after_mom_purchase : ℕ
  final : ℕ

/-- Represents the marble transactions throughout the day -/
structure MarbleTransactions where
  lost_at_breakfast : ℕ
  given_to_susie : ℕ
  bought_by_mom : ℕ

/-- Theorem stating the ratio of marbles Susie gave back to Elsa to the marbles Elsa gave to Susie -/
theorem marble_ratio_theorem (m : MarbleCount) (t : MarbleTransactions) : 
  m.initial = 40 →
  t.lost_at_breakfast = 3 →
  t.given_to_susie = 5 →
  t.bought_by_mom = 12 →
  m.final = 54 →
  m.after_breakfast = m.initial - t.lost_at_breakfast →
  m.after_lunch = m.after_breakfast - t.given_to_susie →
  m.after_mom_purchase = m.after_lunch + t.bought_by_mom →
  (m.final - m.after_mom_purchase) / t.given_to_susie = 2 :=
by sorry

end NUMINAMATH_CALUDE_marble_ratio_theorem_l1142_114234


namespace NUMINAMATH_CALUDE_tank_circumference_l1142_114241

theorem tank_circumference (h_A h_B c_A : ℝ) (h_A_pos : h_A > 0) (h_B_pos : h_B > 0) (c_A_pos : c_A > 0) :
  h_A = 10 →
  h_B = 6 →
  c_A = 6 →
  (π * (c_A / (2 * π))^2 * h_A) = 0.6 * (π * (c_B / (2 * π))^2 * h_B) →
  c_B = 10 :=
by
  sorry

#check tank_circumference

end NUMINAMATH_CALUDE_tank_circumference_l1142_114241


namespace NUMINAMATH_CALUDE_square_side_length_l1142_114223

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 8)
  (h2 : rectangle_length = 2)
  (h3 : rectangle_width > 0)
  (h4 : rectangle_length > 0) :
  ∃ (square_side : ℝ), 
    square_side > 0 ∧ 
    square_side * square_side = rectangle_width * rectangle_length ∧
    square_side = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1142_114223


namespace NUMINAMATH_CALUDE_average_weight_of_all_boys_l1142_114279

theorem average_weight_of_all_boys (group1_count : ℕ) (group1_avg : ℝ) 
  (group2_count : ℕ) (group2_avg : ℝ) : 
  group1_count = 16 → 
  group1_avg = 50.25 → 
  group2_count = 8 → 
  group2_avg = 45.15 → 
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  let total_count := group1_count + group2_count
  (total_weight / total_count) = 48.55 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_all_boys_l1142_114279


namespace NUMINAMATH_CALUDE_gcf_4320_2550_l1142_114267

theorem gcf_4320_2550 : Nat.gcd 4320 2550 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_4320_2550_l1142_114267


namespace NUMINAMATH_CALUDE_circle_C_is_symmetric_l1142_114240

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetry operation
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

-- Define circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- Theorem statement
theorem circle_C_is_symmetric :
  ∀ (x y : ℝ), 
    (∃ (x' y' : ℝ), symmetric_point x' y' = (x, y) ∧ original_circle x' y') ↔ 
    circle_C x y :=
sorry

end NUMINAMATH_CALUDE_circle_C_is_symmetric_l1142_114240


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1142_114274

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∃ a b : ℝ, a = -1 ∧ b = 2 ∧ a * b = -2) ∧
  (∃ a b : ℝ, a * b = -2 ∧ (a ≠ -1 ∨ b ≠ 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1142_114274


namespace NUMINAMATH_CALUDE_doubled_to_original_ratio_l1142_114293

theorem doubled_to_original_ratio (x : ℝ) : 3 * (2 * x + 9) = 57 → (2 * x) / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_doubled_to_original_ratio_l1142_114293


namespace NUMINAMATH_CALUDE_average_weight_of_children_l1142_114237

theorem average_weight_of_children (boys_count : ℕ) (girls_count : ℕ) 
  (boys_avg_weight : ℝ) (girls_avg_weight : ℝ) :
  boys_count = 8 →
  girls_count = 6 →
  boys_avg_weight = 160 →
  girls_avg_weight = 130 →
  let total_children := boys_count + girls_count
  let total_weight := boys_count * boys_avg_weight + girls_count * girls_avg_weight
  (total_weight / total_children : ℝ) = 147 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l1142_114237


namespace NUMINAMATH_CALUDE_square_side_length_l1142_114212

/-- Given a square and a regular hexagon where:
    1) The perimeter of the square equals the perimeter of the hexagon
    2) Each side of the hexagon measures 6 cm
    Prove that the length of one side of the square is 9 cm -/
theorem square_side_length (square_perimeter hexagon_perimeter : ℝ) 
  (hexagon_side : ℝ) (h1 : square_perimeter = hexagon_perimeter) 
  (h2 : hexagon_side = 6) (h3 : hexagon_perimeter = 6 * hexagon_side) :
  square_perimeter / 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1142_114212
