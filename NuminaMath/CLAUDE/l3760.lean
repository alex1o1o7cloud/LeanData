import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3760_376083

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_a2 : a 2 = 18) 
  (h_a4 : a 4 = 8) :
  ∃ q : ℝ, (q = 2/3 ∨ q = -2/3) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3760_376083


namespace NUMINAMATH_CALUDE_a_range_for_increasing_f_l3760_376017

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- State the theorem
theorem a_range_for_increasing_f :
  ∀ a : ℝ, 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  (a ≥ 3/2 ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_a_range_for_increasing_f_l3760_376017


namespace NUMINAMATH_CALUDE_construction_material_total_l3760_376052

theorem construction_material_total (gravel sand : ℝ) 
  (h1 : gravel = 5.91) (h2 : sand = 8.11) : 
  gravel + sand = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_construction_material_total_l3760_376052


namespace NUMINAMATH_CALUDE_total_wheels_is_132_l3760_376097

/-- The number of bicycles in the storage area -/
def num_bicycles : ℕ := 24

/-- The number of tricycles in the storage area -/
def num_tricycles : ℕ := 14

/-- The number of unicycles in the storage area -/
def num_unicycles : ℕ := 10

/-- The number of quadbikes in the storage area -/
def num_quadbikes : ℕ := 8

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

/-- The number of wheels on a unicycle -/
def wheels_per_unicycle : ℕ := 1

/-- The number of wheels on a quadbike -/
def wheels_per_quadbike : ℕ := 4

/-- The total number of wheels in the storage area -/
def total_wheels : ℕ := 
  num_bicycles * wheels_per_bicycle +
  num_tricycles * wheels_per_tricycle +
  num_unicycles * wheels_per_unicycle +
  num_quadbikes * wheels_per_quadbike

theorem total_wheels_is_132 : total_wheels = 132 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_132_l3760_376097


namespace NUMINAMATH_CALUDE_starfish_arms_l3760_376037

theorem starfish_arms (num_starfish : ℕ) (seastar_arms : ℕ) (total_arms : ℕ) :
  num_starfish = 7 →
  seastar_arms = 14 →
  total_arms = 49 →
  ∃ (starfish_arms : ℕ), num_starfish * starfish_arms + seastar_arms = total_arms ∧ starfish_arms = 5 :=
by sorry

end NUMINAMATH_CALUDE_starfish_arms_l3760_376037


namespace NUMINAMATH_CALUDE_dice_game_probability_l3760_376015

theorem dice_game_probability (n : ℕ) (max_score : ℕ) (num_dice : ℕ) (num_faces : ℕ) :
  let p_max_score := (1 / num_faces : ℚ) ^ num_dice
  let p_not_max_score := 1 - p_max_score
  n = 23 ∧ max_score = 18 ∧ num_dice = 3 ∧ num_faces = 6 →
  p_max_score * p_not_max_score ^ (n - 1) = (1 / 216 : ℚ) * (1 - 1 / 216 : ℚ) ^ 22 :=
by sorry

end NUMINAMATH_CALUDE_dice_game_probability_l3760_376015


namespace NUMINAMATH_CALUDE_nth_equation_proof_l3760_376012

theorem nth_equation_proof (n : ℕ+) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

#check nth_equation_proof

end NUMINAMATH_CALUDE_nth_equation_proof_l3760_376012


namespace NUMINAMATH_CALUDE_odd_numbers_theorem_l3760_376023

theorem odd_numbers_theorem (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_numbers_theorem_l3760_376023


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3760_376071

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {y | y^2 + y = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3760_376071


namespace NUMINAMATH_CALUDE_min_k_value_l3760_376050

/-- A special number is a three-digit number with all digits different and non-zero -/
def is_special_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10 ∧
  ∀ i, (n / 10^i) % 10 ≠ 0

/-- F(n) is the sum of three new numbers obtained by swapping digits of n, divided by 111 -/
def F (n : ℕ) : ℚ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ((d2 * 100 + d1 * 10 + d3) + (d3 * 100 + d2 * 10 + d1) + (d1 * 100 + d3 * 10 + d2)) / 111

theorem min_k_value (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 1 ≤ y ∧ y ≤ 9)
  (hs : is_special_number (100 * x + 32)) (ht : is_special_number (150 + y))
  (h_sum : F (100 * x + 32) + F (150 + y) = 19) :
  let s := 100 * x + 32
  let t := 150 + y
  let k := F s - F t
  ∃ k₀, k ≥ k₀ ∧ k₀ = -7 :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l3760_376050


namespace NUMINAMATH_CALUDE_cabbages_on_plot_l3760_376058

/-- Calculates the total number of cabbages that can be planted on a rectangular plot. -/
def total_cabbages (length width density : ℕ) : ℕ :=
  length * width * density

/-- Theorem stating the total number of cabbages on the given plot. -/
theorem cabbages_on_plot :
  total_cabbages 16 12 9 = 1728 := by
  sorry

#eval total_cabbages 16 12 9

end NUMINAMATH_CALUDE_cabbages_on_plot_l3760_376058


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3760_376025

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 3}
def B : Set ℝ := {x | x - 2 > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3760_376025


namespace NUMINAMATH_CALUDE_complex_product_minus_p_l3760_376024

theorem complex_product_minus_p :
  let P : ℂ := 7 + 3 * Complex.I
  let Q : ℂ := 2 * Complex.I
  let R : ℂ := 7 - 3 * Complex.I
  (P * Q * R) - P = 113 * Complex.I - 7 := by sorry

end NUMINAMATH_CALUDE_complex_product_minus_p_l3760_376024


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l3760_376029

/-- Represents scientific notation as a pair of a coefficient and an exponent -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  let billion : ℝ := 1000000000
  let amount : ℝ := 10.58 * billion
  toScientificNotation amount = ScientificNotation.mk 1.058 10 (by norm_num) :=
by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l3760_376029


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l3760_376018

theorem quadratic_equation_roots_condition (k : ℝ) : 
  (∃ p q : ℝ, 3 * p^2 + 6 * p + k = 0 ∧ 
              3 * q^2 + 6 * q + k = 0 ∧ 
              |p - q| = (1/2) * (p^2 + q^2)) ↔ 
  (k = -16 + 12 * Real.sqrt 2 ∨ k = -16 - 12 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l3760_376018


namespace NUMINAMATH_CALUDE_nonagon_angle_measure_l3760_376054

theorem nonagon_angle_measure :
  ∀ (small_angle large_angle : ℝ),
  (9 : ℝ) * small_angle + (9 : ℝ) * large_angle = (7 : ℝ) * 180 →
  6 * small_angle + 3 * large_angle = (7 : ℝ) * 180 →
  large_angle = 3 * small_angle →
  large_angle = 252 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_angle_measure_l3760_376054


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3760_376004

theorem hyperbola_asymptote_slope (x y m : ℝ) : 
  (((x^2 : ℝ) / 49) - ((y^2 : ℝ) / 36) = 1) →
  (∃ (k : ℝ), y = k * m * x ∧ y = -k * m * x) →
  (m > 0) →
  (m = 6/7) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3760_376004


namespace NUMINAMATH_CALUDE_solve_equation_l3760_376060

theorem solve_equation : ∃ x : ℝ, 25 - 5 = 3 + x - 4 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3760_376060


namespace NUMINAMATH_CALUDE_bakery_profit_is_175_l3760_376076

/-- Calculates the total profit for Uki's bakery over five days -/
def bakery_profit : ℝ :=
  let cupcake_price : ℝ := 1.50
  let cookie_price : ℝ := 2.00
  let biscuit_price : ℝ := 1.00
  let cupcake_cost : ℝ := 0.75
  let cookie_cost : ℝ := 1.00
  let biscuit_cost : ℝ := 0.50
  let daily_cupcakes : ℝ := 20
  let daily_cookies : ℝ := 10
  let daily_biscuits : ℝ := 20
  let days : ℝ := 5
  let daily_profit : ℝ := 
    (cupcake_price - cupcake_cost) * daily_cupcakes +
    (cookie_price - cookie_cost) * daily_cookies +
    (biscuit_price - biscuit_cost) * daily_biscuits
  daily_profit * days

theorem bakery_profit_is_175 : bakery_profit = 175 := by
  sorry

end NUMINAMATH_CALUDE_bakery_profit_is_175_l3760_376076


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3760_376049

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℚ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ = 129 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3760_376049


namespace NUMINAMATH_CALUDE_statements_proof_l3760_376041

theorem statements_proof :
  (∀ a b c : ℝ, a > b → c < 0 → a^3 * c < b^3 * c) ∧
  (∀ a b c : ℝ, c > a → a > b → b > 0 → a / (c - a) > b / (c - b)) ∧
  (∀ a b : ℝ, a > b → (1 : ℝ) / a > (1 : ℝ) / b → a > 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_statements_proof_l3760_376041


namespace NUMINAMATH_CALUDE_eighth_term_value_l3760_376070

/-- An arithmetic sequence with 30 terms, first term 5, and last term 86 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  let d := (86 - 5) / 29
  5 + (n - 1) * d

theorem eighth_term_value :
  arithmetic_sequence 8 = 592 / 29 :=
by sorry

end NUMINAMATH_CALUDE_eighth_term_value_l3760_376070


namespace NUMINAMATH_CALUDE_unique_root_condition_l3760_376099

theorem unique_root_condition (k : ℝ) : 
  (∃! x : ℝ, (1/2) * Real.log (k * x) = Real.log (x + 1)) ↔ (k = 4 ∨ k < 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_condition_l3760_376099


namespace NUMINAMATH_CALUDE_composite_sum_l3760_376088

theorem composite_sum (a b c d : ℕ) (h1 : c > b) (h2 : a + b + c + d = a * b - c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + c = x * y := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_l3760_376088


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l3760_376047

theorem sin_cos_difference_equals_half : 
  Real.sin (-(10 * π / 180)) * Real.cos (160 * π / 180) - 
  Real.sin (80 * π / 180) * Real.sin (200 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l3760_376047


namespace NUMINAMATH_CALUDE_eddys_climbing_rate_l3760_376077

/-- Proves that Eddy's climbing rate is 500 ft/hr given the conditions of the problem --/
theorem eddys_climbing_rate (hillary_climb_rate : ℝ) (hillary_descent_rate : ℝ) 
  (start_time : ℝ) (pass_time : ℝ) (base_camp_distance : ℝ) (hillary_stop_distance : ℝ) :
  hillary_climb_rate = 800 →
  hillary_descent_rate = 1000 →
  start_time = 6 →
  pass_time = 12 →
  base_camp_distance = 5000 →
  hillary_stop_distance = 1000 →
  ∃ (eddy_climb_rate : ℝ), eddy_climb_rate = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_eddys_climbing_rate_l3760_376077


namespace NUMINAMATH_CALUDE_exam_score_problem_l3760_376033

theorem exam_score_problem (correct_score : ℕ) (wrong_score : ℕ) 
  (total_score : ℕ) (num_correct : ℕ) :
  correct_score = 3 →
  wrong_score = 1 →
  total_score = 180 →
  num_correct = 75 →
  ∃ (num_wrong : ℕ), 
    total_score = correct_score * num_correct - wrong_score * num_wrong ∧
    num_correct + num_wrong = 120 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3760_376033


namespace NUMINAMATH_CALUDE_stating_greatest_N_with_property_l3760_376019

/-- 
Given a positive integer N, this function type represents the existence of 
N integers x_1, ..., x_N such that x_i^2 - x_i x_j is not divisible by 1111 for any i ≠ j.
-/
def HasProperty (N : ℕ+) : Prop :=
  ∃ (x : Fin N → ℤ), ∀ (i j : Fin N), i ≠ j → ¬(1111 ∣ (x i)^2 - (x i) * (x j))

/-- 
Theorem stating that 1000 is the greatest positive integer satisfying the property.
-/
theorem greatest_N_with_property : 
  HasProperty 1000 ∧ ∀ (N : ℕ+), N > 1000 → ¬HasProperty N :=
sorry

end NUMINAMATH_CALUDE_stating_greatest_N_with_property_l3760_376019


namespace NUMINAMATH_CALUDE_cow_value_increase_l3760_376048

def starting_weight : ℝ := 732
def weight_increase_factor : ℝ := 1.35
def price_per_pound : ℝ := 2.75

theorem cow_value_increase : 
  let new_weight := starting_weight * weight_increase_factor
  let value_at_new_weight := new_weight * price_per_pound
  let value_at_starting_weight := starting_weight * price_per_pound
  value_at_new_weight - value_at_starting_weight = 704.55 := by sorry

end NUMINAMATH_CALUDE_cow_value_increase_l3760_376048


namespace NUMINAMATH_CALUDE_evaluate_expression_l3760_376081

theorem evaluate_expression : (32 / (7 + 3 - 5)) * 8 = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3760_376081


namespace NUMINAMATH_CALUDE_part_one_part_two_l3760_376090

-- Part 1
theorem part_one (x : ℝ) (h1 : x^2 - 4*x + 3 < 0) 
  (h2 : |x - 1| ≤ 2) (h3 : (x + 3) / (x - 2) ≥ 0) : 
  2 < x ∧ x ≤ 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : a > 0) 
  (h_suff : ∀ x : ℝ, ¬(x^2 - 4*a*x + 3*a^2 < 0) → 
    ¬(|x - 1| ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0))
  (h_not_nec : ∃ x : ℝ, ¬(x^2 - 4*a*x + 3*a^2 < 0) ∧ 
    (|x - 1| ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0)) :
  a > 3/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3760_376090


namespace NUMINAMATH_CALUDE_sum_of_angles_octagon_pentagon_l3760_376011

theorem sum_of_angles_octagon_pentagon : 
  ∀ (octagon_angle pentagon_angle : ℝ),
  (octagon_angle = 180 * (8 - 2) / 8) →
  (pentagon_angle = 180 * (5 - 2) / 5) →
  octagon_angle + pentagon_angle = 243 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_angles_octagon_pentagon_l3760_376011


namespace NUMINAMATH_CALUDE_haircut_cost_l3760_376038

-- Define the constants
def hair_growth_rate : ℝ := 1.5
def max_hair_length : ℝ := 9
def min_hair_length : ℝ := 6
def tip_percentage : ℝ := 0.2
def annual_haircut_cost : ℝ := 324

-- Define the theorem
theorem haircut_cost (haircut_cost : ℝ) : 
  hair_growth_rate * 12 / (max_hair_length - min_hair_length) * 
  (haircut_cost * (1 + tip_percentage)) = annual_haircut_cost → 
  haircut_cost = 45 := by
sorry

end NUMINAMATH_CALUDE_haircut_cost_l3760_376038


namespace NUMINAMATH_CALUDE_intersection_points_form_line_l3760_376042

theorem intersection_points_form_line (s : ℝ) :
  let x := s + 15
  let y := 2*s - 8
  (2*x + 3*y = 8*s + 6) ∧ (x + 2*y = 5*s - 1) →
  y = 2*x - 38 := by
sorry

end NUMINAMATH_CALUDE_intersection_points_form_line_l3760_376042


namespace NUMINAMATH_CALUDE_unique_solution_system_l3760_376053

theorem unique_solution_system (x y : ℝ) : 
  x ≥ 0 ∧ y ≥ 0 ∧
  y * Real.sqrt (2 * x) - x * Real.sqrt (2 * y) = 6 ∧
  x * y^2 - x^2 * y = 30 →
  x = 1/2 ∧ y = 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3760_376053


namespace NUMINAMATH_CALUDE_soap_brand_survey_l3760_376005

theorem soap_brand_survey (total : ℕ) (neither : ℕ) (only_a : ℕ) (both_to_only_b_ratio : ℕ) 
  (h1 : total = 180)
  (h2 : neither = 80)
  (h3 : only_a = 60)
  (h4 : both_to_only_b_ratio = 3) :
  ∃ (both : ℕ), 
    neither + only_a + both + both_to_only_b_ratio * both = total ∧ 
    both = 10 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_survey_l3760_376005


namespace NUMINAMATH_CALUDE_triangle_properties_l3760_376085

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.c - t.a) * Real.cos t.B = t.b * Real.cos t.A)
  (h2 : t.b = 6)
  (h3 : t.c = 2 * t.a) :
  t.B = π / 3 ∧ 
  (1 / 2 : ℝ) * t.a * t.c * Real.sin t.B = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3760_376085


namespace NUMINAMATH_CALUDE_factory_production_equation_l3760_376098

/-- Represents the production equation for a factory with monthly growth rate --/
theorem factory_production_equation (april_production : ℝ) (quarter_production : ℝ) (x : ℝ) :
  april_production = 500000 →
  quarter_production = 1820000 →
  50 + 50 * (1 + x) + 50 * (1 + x)^2 = 182 :=
by sorry

end NUMINAMATH_CALUDE_factory_production_equation_l3760_376098


namespace NUMINAMATH_CALUDE_length_of_AE_l3760_376026

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (AF : ℝ)
  (CE : ℝ)
  (ED : ℝ)
  (area : ℝ)

/-- Theorem stating the length of AE in the given quadrilateral -/
theorem length_of_AE (q : Quadrilateral) 
  (h1 : q.AF = 30)
  (h2 : q.CE = 40)
  (h3 : q.ED = 50)
  (h4 : q.area = 7200) : 
  ∃ AE : ℝ, AE = 322.5 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AE_l3760_376026


namespace NUMINAMATH_CALUDE_pharmacist_weights_impossibility_l3760_376021

theorem pharmacist_weights_impossibility :
  ¬∃ (w₁ w₂ w₃ : ℝ),
    w₁ < 90 ∧ w₂ < 90 ∧ w₃ < 90 ∧
    w₁ + w₂ + w₃ = 100 ∧
    w₁ + 2*w₂ + w₃ = 101 ∧
    w₁ + w₂ + 2*w₃ = 102 :=
sorry

end NUMINAMATH_CALUDE_pharmacist_weights_impossibility_l3760_376021


namespace NUMINAMATH_CALUDE_intersection_values_l3760_376031

-- Define the function f(x) = ax² + (3-a)x + 1
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (3 - a) * x + 1

-- Define the condition for intersection with x-axis at only one point
def intersects_once (a : ℝ) : Prop :=
  ∃! x, f a x = 0

-- State the theorem
theorem intersection_values : 
  ∀ a : ℝ, intersects_once a ↔ a = 0 ∨ a = 1 ∨ a = 9 := by sorry

end NUMINAMATH_CALUDE_intersection_values_l3760_376031


namespace NUMINAMATH_CALUDE_triangle_area_l3760_376074

theorem triangle_area (a c : Real) (B : Real) 
  (h1 : a = Real.sqrt 2)
  (h2 : c = 2 * Real.sqrt 2)
  (h3 : B = 30 * π / 180) :
  (1/2) * a * c * Real.sin B = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3760_376074


namespace NUMINAMATH_CALUDE_range_of_sine_function_l3760_376006

theorem range_of_sine_function (ω : ℝ) (φ : ℝ) :
  ω > 0 →
  (∀ x, 3 * Real.sin (ω * x - π / 6) = 3 * Real.cos (2 * x + φ)) →
  (∀ x ∈ Set.Icc 0 (π / 2), 
    -3/2 ≤ 3 * Real.sin (ω * x - π / 6) ∧ 
    3 * Real.sin (ω * x - π / 6) ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_sine_function_l3760_376006


namespace NUMINAMATH_CALUDE_divisibility_condition_l3760_376057

theorem divisibility_condition (M : ℕ) : 
  M > 0 ∧ M < 10 → (5 ∣ 1989^M + M^1989 ↔ M = 1 ∨ M = 4) := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3760_376057


namespace NUMINAMATH_CALUDE_playground_total_l3760_376001

/-- The number of children on the playground at recess -/
def total_children (soccer_boys soccer_girls swings_boys swings_girls snacks_boys snacks_girls : ℕ) : ℕ :=
  soccer_boys + soccer_girls + swings_boys + swings_girls + snacks_boys + snacks_girls

/-- Theorem stating the total number of children on the playground -/
theorem playground_total :
  total_children 27 35 15 20 10 5 = 112 := by
  sorry

end NUMINAMATH_CALUDE_playground_total_l3760_376001


namespace NUMINAMATH_CALUDE_power_sum_implications_l3760_376067

theorem power_sum_implications (a b c : ℝ) : 
  (¬ ((a^2013 + b^2013 + c^2013 = 0) → (a^2014 + b^2014 + c^2014 = 0))) ∧ 
  ((a^2014 + b^2014 + c^2014 = 0) → (a^2015 + b^2015 + c^2015 = 0)) ∧ 
  (¬ ((a^2013 + b^2013 + c^2013 = 0 ∧ a^2015 + b^2015 + c^2015 = 0) → (a^2014 + b^2014 + c^2014 = 0))) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_implications_l3760_376067


namespace NUMINAMATH_CALUDE_triangle_with_small_angle_l3760_376056

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a set of n points
def PointSet (n : ℕ) := Fin n → Point

-- Define a function to calculate the angle between three points
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem statement
theorem triangle_with_small_angle (n : ℕ) (points : PointSet n) :
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    ∃ (θ : ℝ), θ ≤ 180 / n ∧
      (θ = angle (points i) (points j) (points k) ∨
       θ = angle (points j) (points k) (points i) ∨
       θ = angle (points k) (points i) (points j)) :=
sorry

end NUMINAMATH_CALUDE_triangle_with_small_angle_l3760_376056


namespace NUMINAMATH_CALUDE_medium_revenue_is_24_l3760_376068

/-- Represents the revenue from Tonya's lemonade stand -/
structure LemonadeRevenue where
  total : ℕ
  small : ℕ
  large_cups : ℕ
  small_price : ℕ
  medium_price : ℕ
  large_price : ℕ

/-- Calculates the revenue from medium lemonades -/
def medium_revenue (r : LemonadeRevenue) : ℕ :=
  r.total - r.small - (r.large_cups * r.large_price)

/-- Theorem: The revenue from medium lemonades is 24 -/
theorem medium_revenue_is_24 (r : LemonadeRevenue) 
  (h1 : r.total = 50)
  (h2 : r.small = 11)
  (h3 : r.large_cups = 5)
  (h4 : r.small_price = 1)
  (h5 : r.medium_price = 2)
  (h6 : r.large_price = 3) :
  medium_revenue r = 24 := by
  sorry

end NUMINAMATH_CALUDE_medium_revenue_is_24_l3760_376068


namespace NUMINAMATH_CALUDE_mikes_shopping_expense_l3760_376013

/-- Calculates the total amount Mike spent given the costs and discounts of items. -/
def total_spent (food_cost wallet_cost shirt_cost shoes_cost belt_cost : ℚ)
  (shirt_discount shoes_discount belt_discount : ℚ) : ℚ :=
  food_cost + wallet_cost +
  shirt_cost * (1 - shirt_discount) +
  shoes_cost * (1 - shoes_discount) +
  belt_cost * (1 - belt_discount)

/-- Theorem stating the total amount Mike spent given the conditions. -/
theorem mikes_shopping_expense :
  let food_cost : ℚ := 30
  let wallet_cost : ℚ := food_cost + 60
  let shirt_cost : ℚ := wallet_cost / 3
  let shoes_cost : ℚ := 2 * wallet_cost
  let belt_cost : ℚ := shoes_cost - 45
  let shirt_discount : ℚ := 20 / 100
  let shoes_discount : ℚ := 15 / 100
  let belt_discount : ℚ := 10 / 100
  total_spent food_cost wallet_cost shirt_cost shoes_cost belt_cost
    shirt_discount shoes_discount belt_discount = 418.5 := by sorry

end NUMINAMATH_CALUDE_mikes_shopping_expense_l3760_376013


namespace NUMINAMATH_CALUDE_b_equals_one_b_non_negative_l3760_376040

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem 1
theorem b_equals_one (b c : ℝ) :
  c = -3 →
  quadratic 2 b c (-1) = -2 →
  b = 1 := by sorry

-- Theorem 2
theorem b_non_negative (b c p : ℝ) :
  b + c = -2 →
  b > c →
  quadratic 2 b c p = -2 →
  b ≥ 0 := by sorry

end NUMINAMATH_CALUDE_b_equals_one_b_non_negative_l3760_376040


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3760_376075

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, a (n + 1) - a n = 1) →  -- Common difference is 1
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Sum formula for arithmetic sequence
  (S 6 = 4 * S 3) →  -- Given condition
  a 10 = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3760_376075


namespace NUMINAMATH_CALUDE_max_value_theorem_l3760_376051

theorem max_value_theorem (a b : ℝ) (h : a^2 - b^2 = -1) :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x y : ℝ), x^2 - y^2 = -1 → (|x| + 1) / y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3760_376051


namespace NUMINAMATH_CALUDE_strawberry_pies_l3760_376089

/-- The number of pies that can be made from strawberries picked by Christine and Rachel -/
def number_of_pies (christine_pounds : ℕ) (rachel_multiplier : ℕ) (pounds_per_pie : ℕ) : ℕ :=
  (christine_pounds + christine_pounds * rachel_multiplier) / pounds_per_pie

/-- Theorem: Christine and Rachel can make 10 pies given the conditions -/
theorem strawberry_pies : number_of_pies 10 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_pies_l3760_376089


namespace NUMINAMATH_CALUDE_rattle_ownership_l3760_376092

structure Brother :=
  (id : ℕ)
  (claims_ownership : Bool)

def Alice := Unit

def determine_owner (b1 b2 : Brother) (a : Alice) : Brother :=
  sorry

theorem rattle_ownership (b1 b2 : Brother) (a : Alice) :
  b1.id = 1 →
  b2.id = 2 →
  b1.claims_ownership = true →
  (determine_owner b1 b2 a).id = 1 :=
sorry

end NUMINAMATH_CALUDE_rattle_ownership_l3760_376092


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l3760_376079

theorem shopping_tax_calculation (total_amount : ℝ) (total_amount_pos : total_amount > 0) :
  let clothing_percent : ℝ := 0.40
  let food_percent : ℝ := 0.20
  let electronics_percent : ℝ := 0.15
  let other_percent : ℝ := 0.25
  let clothing_tax : ℝ := 0.12
  let food_tax : ℝ := 0
  let electronics_tax : ℝ := 0.05
  let other_tax : ℝ := 0.20
  let total_tax := 
    clothing_percent * total_amount * clothing_tax +
    food_percent * total_amount * food_tax +
    electronics_percent * total_amount * electronics_tax +
    other_percent * total_amount * other_tax
  (total_tax / total_amount) * 100 = 10.55 := by
sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l3760_376079


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3760_376084

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x : ℝ | x * (4 - x) < 0}

theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3760_376084


namespace NUMINAMATH_CALUDE_expression_value_l3760_376010

theorem expression_value : (5^2 - 5) * (6^2 - 6) - (7^2 - 7) = 558 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3760_376010


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3760_376046

theorem quadratic_solution_difference_squared :
  ∀ p q : ℝ,
  (2 * p^2 + 11 * p - 21 = 0) →
  (2 * q^2 + 11 * q - 21 = 0) →
  p ≠ q →
  (p - q)^2 = 289/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3760_376046


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3760_376059

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + r^2 + 5 * r - 3) - (r^3 + 3 * r^2 + 9 * r - 2) = r^3 - 2 * r^2 - 4 * r - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3760_376059


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3760_376016

theorem complex_fraction_simplification : 
  (((3875/1000) * (1/5) + (155/4) * (9/100) - (155/400)) / 
   ((13/6) + (((108/25) - (42/25) - (33/25)) * (5/11) - (2/7)) / (44/35) + (35/24))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3760_376016


namespace NUMINAMATH_CALUDE_no_real_solution_for_exponential_equation_l3760_376087

theorem no_real_solution_for_exponential_equation :
  ¬∃ (x : ℝ), (2 : ℝ)^x + (3 : ℝ)^x = (4 : ℝ)^x + 4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_exponential_equation_l3760_376087


namespace NUMINAMATH_CALUDE_factorial15_base16_zeros_l3760_376086

/-- The number of trailing zeros in n when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 15 factorial -/
def factorial15 : ℕ :=
  sorry

theorem factorial15_base16_zeros :
  trailingZeros factorial15 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial15_base16_zeros_l3760_376086


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l3760_376096

theorem largest_divisor_of_n_squared_div_72 (n : ℕ) (h : n > 0) (h_div : 72 ∣ n^2) :
  ∃ (t : ℕ), t > 0 ∧ t ∣ n ∧ ∀ (k : ℕ), k > 0 → k ∣ n → k ≤ t :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l3760_376096


namespace NUMINAMATH_CALUDE_pet_shop_kittens_l3760_376039

/-- Calculates the number of kittens in a pet shop given the following conditions:
  * The pet shop has 2 puppies
  * A puppy costs $20
  * A kitten costs $15
  * The total stock is worth $100
-/
theorem pet_shop_kittens (num_puppies : ℕ) (puppy_cost kitten_cost total_stock : ℚ) : 
  num_puppies = 2 → 
  puppy_cost = 20 → 
  kitten_cost = 15 → 
  total_stock = 100 → 
  (total_stock - num_puppies * puppy_cost) / kitten_cost = 4 := by
  sorry

#check pet_shop_kittens

end NUMINAMATH_CALUDE_pet_shop_kittens_l3760_376039


namespace NUMINAMATH_CALUDE_remainder_is_six_l3760_376008

/-- The divisor polynomial -/
def divisor (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

/-- The dividend polynomial -/
def dividend (x : ℂ) : ℂ := x^60 + x^48 + x^36 + x^24 + x^12 + 1

/-- Theorem stating that the remainder is 6 -/
theorem remainder_is_six : ∃ q : ℂ → ℂ, ∀ x : ℂ, 
  dividend x = (divisor x) * (q x) + 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_six_l3760_376008


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_l3760_376061

theorem imaginary_part_of_i : Complex.im Complex.I = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_l3760_376061


namespace NUMINAMATH_CALUDE_lottery_probability_l3760_376027

theorem lottery_probability : 
  let powerball_count : ℕ := 30
  let luckyball_count : ℕ := 49
  let luckyball_draw : ℕ := 6
  1 / (powerball_count * (Nat.choose luckyball_count luckyball_draw)) = 1 / 419514480 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l3760_376027


namespace NUMINAMATH_CALUDE_total_shoes_l3760_376014

/-- The number of shoes owned by each person -/
structure ShoeCount where
  daniel : ℕ
  christopher : ℕ
  brian : ℕ
  edward : ℕ
  jacob : ℕ

/-- The conditions of the shoe ownership problem -/
def shoe_conditions (s : ShoeCount) : Prop :=
  s.daniel = 15 ∧
  s.christopher = 37 ∧
  s.brian = s.christopher + 5 ∧
  s.edward = (7 * s.brian) / 2 ∧
  s.jacob = (2 * s.edward) / 3

/-- The theorem stating the total number of shoes -/
theorem total_shoes (s : ShoeCount) (h : shoe_conditions s) :
  s.daniel + s.christopher + s.brian + s.edward + s.jacob = 339 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l3760_376014


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3760_376073

theorem inequality_solution_set (x : ℝ) :
  x ≠ 3/2 →
  ((x - 4) / (3 - 2*x) < 0) ↔ (x < 3/2 ∨ x > 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3760_376073


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3760_376082

theorem complex_equation_sum (a b : ℝ) (h : (3 * b : ℂ) + (2 * a - 2) * Complex.I = 1 - Complex.I) : 
  a + b = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3760_376082


namespace NUMINAMATH_CALUDE_square_inscribed_circle_tangent_l3760_376007

/-- Given a square with side length a, where two adjacent sides are divided into 6 and 10 equal parts respectively,
    prove that the line segment connecting the first division point (a/6) on one side to the fourth division point (4a/10)
    on the adjacent side is tangent to the inscribed circle of the square. -/
theorem square_inscribed_circle_tangent (a : ℝ) (h : a > 0) : 
  let square := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a}
  let P := (a / 6, 0)
  let Q := (0, 2 * a / 5)
  let circle_center := (a / 2, a / 2)
  let circle_radius := a / 2
  let circle := {(x, y) : ℝ × ℝ | (x - a / 2)^2 + (y - a / 2)^2 = (a / 2)^2}
  let line_PQ := {(x, y) : ℝ × ℝ | y = -12/5 * x + 2*a/5}
  (∀ point ∈ line_PQ, point ∉ circle) ∧ 
  (∃ point ∈ line_PQ, (point.1 - circle_center.1)^2 + (point.2 - circle_center.2)^2 = circle_radius^2)
  := by sorry

end NUMINAMATH_CALUDE_square_inscribed_circle_tangent_l3760_376007


namespace NUMINAMATH_CALUDE_X_equals_three_l3760_376028

/-- The length of the unknown segment X in the diagram --/
def X : ℝ := sorry

/-- The total length of the top side of the figure --/
def top_length : ℝ := 3 + 2 + X + 4

/-- The length of the bottom side of the figure --/
def bottom_length : ℝ := 12

/-- Theorem stating that X equals 3 --/
theorem X_equals_three : X = 3 := by
  sorry

end NUMINAMATH_CALUDE_X_equals_three_l3760_376028


namespace NUMINAMATH_CALUDE_seventh_term_is_four_l3760_376065

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  a₁_eq_one : a 1 = 1
  a₃_a₅_eq_4a₄_minus_4 : a 3 * a 5 = 4 * (a 4 - 1)

/-- The 7th term of the geometric sequence is 4 -/
theorem seventh_term_is_four (seq : GeometricSequence) : seq.a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_four_l3760_376065


namespace NUMINAMATH_CALUDE_log_problem_l3760_376091

theorem log_problem (y : ℝ) (p : ℝ) 
  (h1 : Real.log 5 / Real.log 9 = y)
  (h2 : Real.log 125 / Real.log 3 = p * y) : 
  p = 6 := by sorry

end NUMINAMATH_CALUDE_log_problem_l3760_376091


namespace NUMINAMATH_CALUDE_calculation_proof_l3760_376072

theorem calculation_proof : 99 * (5/8) - 0.625 * 68 + 6.25 * 0.1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3760_376072


namespace NUMINAMATH_CALUDE_february_roses_l3760_376036

def rose_sequence (october november december january : ℕ) : Prop :=
  november - october = december - november ∧
  december - november = january - december ∧
  november > october ∧ december > november ∧ january > december

theorem february_roses 
  (october november december january : ℕ) 
  (h : rose_sequence october november december january) 
  (oct_val : october = 108) 
  (nov_val : november = 120) 
  (dec_val : december = 132) 
  (jan_val : january = 144) : 
  january + (january - december) = 156 := by
sorry

end NUMINAMATH_CALUDE_february_roses_l3760_376036


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3760_376094

theorem sin_cos_identity : 
  Real.sin (160 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (200 * π / 180) * Real.cos (80 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3760_376094


namespace NUMINAMATH_CALUDE_odd_m_triple_g_16_l3760_376032

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 ∧ n % 2 ≠ 0 then n / 3
  else n / 2

theorem odd_m_triple_g_16 (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 16) :
  m = 59 ∨ m = 91 := by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_16_l3760_376032


namespace NUMINAMATH_CALUDE_male_democrat_ratio_l3760_376095

/-- Proves the ratio of male democrats to total male participants in a meeting --/
theorem male_democrat_ratio (total_participants : ℕ) (female_democrats : ℕ) :
  total_participants = 750 →
  female_democrats = 125 →
  female_democrats * 2 ≤ total_participants →
  3 * female_democrats * 2 = total_participants →
  (total_participants / 3 - female_democrats) * 4 = total_participants - female_democrats * 2 :=
by
  sorry

#check male_democrat_ratio

end NUMINAMATH_CALUDE_male_democrat_ratio_l3760_376095


namespace NUMINAMATH_CALUDE_point_D_coordinates_l3760_376093

/-- Given points A and B, and the relation between vectors AD and AB,
    prove that the coordinates of point D are (-7, 9) -/
theorem point_D_coordinates (A B D : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (-1, 5) → 
  D - A = 3 • (B - A) → 
  D = (-7, 9) := by
sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l3760_376093


namespace NUMINAMATH_CALUDE_exactly_seventeen_solutions_l3760_376003

/-- The number of ordered pairs of complex numbers satisfying the given equations -/
def num_solutions : ℕ := 17

/-- The property that a pair of complex numbers satisfies the given equations -/
def satisfies_equations (a b : ℂ) : Prop :=
  a^5 * b^3 = 1 ∧ a^9 * b^2 = 1

/-- The theorem stating that there are exactly 17 solutions -/
theorem exactly_seventeen_solutions :
  ∃! (s : Set (ℂ × ℂ)), 
    (∀ (p : ℂ × ℂ), p ∈ s ↔ satisfies_equations p.1 p.2) ∧
    Finite s ∧
    Nat.card s = num_solutions :=
sorry

end NUMINAMATH_CALUDE_exactly_seventeen_solutions_l3760_376003


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l3760_376044

/-- The number of marshmallows Haley can hold -/
def haley_marshmallows : ℕ := sorry

/-- The number of marshmallows Michael can hold -/
def michael_marshmallows : ℕ := 3 * haley_marshmallows

/-- The number of marshmallows Brandon can hold -/
def brandon_marshmallows : ℕ := michael_marshmallows / 2

/-- The total number of marshmallows all three kids can hold -/
def total_marshmallows : ℕ := 44

theorem marshmallow_challenge : 
  haley_marshmallows + michael_marshmallows + brandon_marshmallows = total_marshmallows ∧ 
  haley_marshmallows = 8 := by sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l3760_376044


namespace NUMINAMATH_CALUDE_sam_money_left_l3760_376043

/-- The amount of money Sam has left after her purchases -/
def money_left : ℕ :=
  let initial_dimes : ℕ := 19
  let initial_quarters : ℕ := 6
  let candy_bars : ℕ := 4
  let dimes_per_candy : ℕ := 3
  let lollipops : ℕ := 1
  let initial_money : ℕ := initial_dimes * 10 + initial_quarters * 25
  let candy_cost : ℕ := candy_bars * dimes_per_candy * 10
  let lollipop_cost : ℕ := lollipops * 25
  initial_money - (candy_cost + lollipop_cost)

theorem sam_money_left : money_left = 195 := by
  sorry

end NUMINAMATH_CALUDE_sam_money_left_l3760_376043


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l3760_376080

theorem cos_2alpha_value (α : ℝ) (a : ℝ × ℝ) :
  a = (Real.cos α, Real.sqrt 2 / 2) →
  ‖a‖ = Real.sqrt 3 / 2 →
  Real.cos (2 * α) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l3760_376080


namespace NUMINAMATH_CALUDE_a_bounds_l3760_376034

def a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => a n + (1 / (n+1)^2) * (a n)^2

theorem a_bounds (n : ℕ) : (n+1)/(n+2) < a n ∧ a n < n+1 := by
  sorry

end NUMINAMATH_CALUDE_a_bounds_l3760_376034


namespace NUMINAMATH_CALUDE_pencil_count_l3760_376009

theorem pencil_count (pens pencils : ℕ) : 
  (pens : ℚ) / pencils = 5 / 6 →
  pencils = pens + 7 →
  pencils = 42 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_l3760_376009


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3760_376069

/-- A line passing through a point and parallel to another line -/
def parallel_line (p : ℝ × ℝ) (m : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.2 = m * q.1 + (p.2 - m * p.1)}

theorem parallel_line_equation :
  let p : ℝ × ℝ := (0, 7)
  let m : ℝ := -4
  parallel_line p m = {q : ℝ × ℝ | q.2 = -4 * q.1 + 7} := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3760_376069


namespace NUMINAMATH_CALUDE_no_positive_sheep_solution_l3760_376035

theorem no_positive_sheep_solution : ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ y = 3 * x + 15 ∧ x = y - y / 3 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_sheep_solution_l3760_376035


namespace NUMINAMATH_CALUDE_two_vectors_basis_iff_linearly_independent_not_any_two_vectors_form_basis_l3760_376062

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

/-- Two vectors form a basis for a 2-dimensional vector space if and only if they are linearly independent. -/
theorem two_vectors_basis_iff_linearly_independent (v w : V) :
  Submodule.span ℝ {v, w} = ⊤ ↔ LinearIndependent ℝ ![v, w] :=
sorry

/-- It is not true that any two vectors in a 2-dimensional vector space form a basis. -/
theorem not_any_two_vectors_form_basis :
  ¬ ∀ (v w : V), Submodule.span ℝ {v, w} = ⊤ :=
sorry

end NUMINAMATH_CALUDE_two_vectors_basis_iff_linearly_independent_not_any_two_vectors_form_basis_l3760_376062


namespace NUMINAMATH_CALUDE_area_of_triangle_BEF_l3760_376030

-- Define the rectangle ABCD
structure Rectangle :=
  (a : ℝ) (b : ℝ)
  (area_eq : a * b = 30)

-- Define points E and F
structure Points (rect : Rectangle) :=
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)
  (E_on_AB : E.2 = 0 ∧ 0 ≤ E.1 ∧ E.1 ≤ rect.a)
  (F_on_BC : F.1 = rect.a ∧ 0 ≤ F.2 ∧ F.2 ≤ rect.b)

-- Define the theorem
theorem area_of_triangle_BEF
  (rect : Rectangle)
  (pts : Points rect)
  (area_CGF : ℝ)
  (area_EGF : ℝ)
  (h1 : area_CGF = 2)
  (h2 : area_EGF = 3) :
  (1/2) * pts.E.1 * pts.F.2 = 35/8 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_BEF_l3760_376030


namespace NUMINAMATH_CALUDE_range_of_m_l3760_376002

-- Define p as a proposition depending on m
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0

-- Define q as a proposition depending on m
def q (m : ℝ) : Prop := m > 2

-- Define the set of m satisfying the conditions
def S : Set ℝ := {m : ℝ | p m ∧ ¬(q m) ∧ ¬(¬(p m)) ∧ ¬(p m ∧ q m)}

-- Theorem statement
theorem range_of_m : S = {m : ℝ | 1 < m ∧ m ≤ 2} := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3760_376002


namespace NUMINAMATH_CALUDE_tournament_matches_l3760_376022

/-- A tournament with the given rules --/
structure Tournament :=
  (num_players : ℕ)
  (num_players_per_match : ℕ)
  (points_per_match : Fin 3 → ℕ)
  (eliminated_per_match : ℕ)

/-- The number of matches played in a tournament --/
def num_matches (t : Tournament) : ℕ :=
  t.num_players - 1

/-- The theorem stating the number of matches in the specific tournament --/
theorem tournament_matches :
  ∀ t : Tournament,
    t.num_players = 999 ∧
    t.num_players_per_match = 3 ∧
    t.points_per_match 0 = 2 ∧
    t.points_per_match 1 = 1 ∧
    t.points_per_match 2 = 0 ∧
    t.eliminated_per_match = 1 →
    num_matches t = 998 :=
by sorry

end NUMINAMATH_CALUDE_tournament_matches_l3760_376022


namespace NUMINAMATH_CALUDE_rug_on_floor_l3760_376020

theorem rug_on_floor (rug_length : ℝ) (rug_width : ℝ) (floor_area : ℝ) : 
  rug_length = 2 →
  rug_width = 7 →
  floor_area = 64 →
  rug_length * rug_width ≤ floor_area →
  (floor_area - rug_length * rug_width) / floor_area = 25 / 32 := by
  sorry

end NUMINAMATH_CALUDE_rug_on_floor_l3760_376020


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3760_376078

/-- Given a rectangle with vertices at (-2, y), (6, y), (-2, 2), and (6, 2),
    where y is positive, and an area of 64 square units, y must equal 10. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  (6 - (-2)) * (y - 2) = 64 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3760_376078


namespace NUMINAMATH_CALUDE_problem1_l3760_376055

theorem problem1 (x : ℝ) : (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l3760_376055


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_three_fifths_l3760_376063

theorem fraction_of_powers_equals_three_fifths :
  (3^2011 + 3^2011) / (3^2010 + 3^2012) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_three_fifths_l3760_376063


namespace NUMINAMATH_CALUDE_power_two_ge_two_times_l3760_376064

theorem power_two_ge_two_times (n : ℕ) : 2^n ≥ 2*n := by
  sorry

end NUMINAMATH_CALUDE_power_two_ge_two_times_l3760_376064


namespace NUMINAMATH_CALUDE_seashells_given_to_joan_l3760_376000

/-- Given that Sam initially found 35 seashells and now has 17 seashells,
    prove that the number of seashells he gave to Joan is 18. -/
theorem seashells_given_to_joan 
  (initial_seashells : ℕ) 
  (current_seashells : ℕ) 
  (h1 : initial_seashells = 35) 
  (h2 : current_seashells = 17) : 
  initial_seashells - current_seashells = 18 := by
sorry

end NUMINAMATH_CALUDE_seashells_given_to_joan_l3760_376000


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3760_376045

theorem perfect_square_condition (p : ℕ) : 
  Nat.Prime p → (∃ (x : ℕ), 7^p - p - 16 = x^2) ↔ p = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3760_376045


namespace NUMINAMATH_CALUDE_domain_of_f_2x_minus_1_l3760_376066

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem domain_of_f_2x_minus_1 :
  (∀ x ∈ domain_f_x_plus_1, f (x + 1) = f (x + 1)) →
  {x : ℝ | f (2 * x - 1) = f (2 * x - 1)} = Set.Icc 0 (5/2) := by sorry

end NUMINAMATH_CALUDE_domain_of_f_2x_minus_1_l3760_376066
