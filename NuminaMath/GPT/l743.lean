import Mathlib

namespace NUMINAMATH_GPT_inequality_proof_l743_74365

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a ≥ 1) :
    (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ (Real.sqrt 3) / (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l743_74365


namespace NUMINAMATH_GPT_incorrect_conclusion_l743_74383

theorem incorrect_conclusion (p q : ℝ) (h1 : p < 0) (h2 : q < 0) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ (x1 * |x1| + p * x1 + q = 0) ∧ (x2 * |x2| + p * x2 + q = 0) ∧ (x3 * |x3| + p * x3 + q = 0) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_l743_74383


namespace NUMINAMATH_GPT_question_inequality_l743_74321

theorem question_inequality (m : ℝ) :
  (∀ x : ℝ, ¬ (m * x ^ 2 - m * x - 1 ≥ 0)) ↔ (-4 < m ∧ m ≤ 0) :=
sorry

end NUMINAMATH_GPT_question_inequality_l743_74321


namespace NUMINAMATH_GPT_production_days_l743_74313

theorem production_days (n P : ℕ) (h1 : P = 60 * n) (h2 : (P + 90) / (n + 1) = 65) : n = 5 := sorry

end NUMINAMATH_GPT_production_days_l743_74313


namespace NUMINAMATH_GPT_triangle_perimeter_l743_74388

-- Given conditions
def inradius : ℝ := 2.5
def area : ℝ := 40

-- The formula relating inradius, area, and perimeter
def perimeter_formula (r a p : ℝ) : Prop := a = r * p / 2

-- Prove the perimeter p of the triangle
theorem triangle_perimeter : ∃ (p : ℝ), perimeter_formula inradius area p ∧ p = 32 := by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l743_74388


namespace NUMINAMATH_GPT_decrease_percent_in_revenue_l743_74302

theorem decrease_percent_in_revenue 
  (T C : ℝ) 
  (original_revenue : ℝ := T * C)
  (new_tax : ℝ := 0.80 * T)
  (new_consumption : ℝ := 1.15 * C)
  (new_revenue : ℝ := new_tax * new_consumption) :
  ((original_revenue - new_revenue) / original_revenue) * 100 = 8 := 
sorry

end NUMINAMATH_GPT_decrease_percent_in_revenue_l743_74302


namespace NUMINAMATH_GPT_fred_games_this_year_l743_74317

variable (last_year_games : ℕ)
variable (difference : ℕ)

theorem fred_games_this_year (h1 : last_year_games = 36) (h2 : difference = 11) : 
  last_year_games - difference = 25 := 
by
  sorry

end NUMINAMATH_GPT_fred_games_this_year_l743_74317


namespace NUMINAMATH_GPT_sum_of_numbers_l743_74339

theorem sum_of_numbers (avg : ℝ) (num : ℕ) (h1 : avg = 5.2) (h2 : num = 8) : 
  (avg * num = 41.6) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l743_74339


namespace NUMINAMATH_GPT_problem1_problem2_l743_74330

theorem problem1 : |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.sqrt 2 = Real.sqrt 3 + Real.sqrt 2 := 
by {
  sorry
}

theorem problem2 : Real.sqrt 5 * (Real.sqrt 5 - 1 / Real.sqrt 5) = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem1_problem2_l743_74330


namespace NUMINAMATH_GPT_sum_3n_terms_l743_74351

variable {a_n : ℕ → ℝ} -- Definition of the sequence
variable {S : ℕ → ℝ} -- Definition of the sum function

-- Conditions
axiom sum_n_terms (n : ℕ) : S n = 3
axiom sum_2n_terms (n : ℕ) : S (2 * n) = 15

-- Question and correct answer
theorem sum_3n_terms (n : ℕ) : S (3 * n) = 63 := 
sorry -- Proof to be provided

end NUMINAMATH_GPT_sum_3n_terms_l743_74351


namespace NUMINAMATH_GPT_number_of_boys_exceeds_girls_by_l743_74300

theorem number_of_boys_exceeds_girls_by (girls boys: ℕ) (h1: girls = 34) (h2: boys = 841) : boys - girls = 807 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_exceeds_girls_by_l743_74300


namespace NUMINAMATH_GPT_num_two_digit_prime_with_units_digit_3_eq_6_l743_74345

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_two_digit_prime_with_units_digit_3_eq_6_l743_74345


namespace NUMINAMATH_GPT_total_wet_surface_area_is_correct_l743_74392

noncomputable def wet_surface_area (cistern_length cistern_width water_depth platform_length platform_width platform_height : ℝ) : ℝ :=
  let two_longer_walls := 2 * (cistern_length * water_depth)
  let two_shorter_walls := 2 * (cistern_width * water_depth)
  let area_walls := two_longer_walls + two_shorter_walls
  let area_bottom := cistern_length * cistern_width
  let submerged_height := water_depth - platform_height
  let two_longer_sides_platform := 2 * (platform_length * submerged_height)
  let two_shorter_sides_platform := 2 * (platform_width * submerged_height)
  let area_platform_sides := two_longer_sides_platform + two_shorter_sides_platform
  area_walls + area_bottom + area_platform_sides

theorem total_wet_surface_area_is_correct :
  wet_surface_area 8 4 1.25 1 0.5 0.75 = 63.5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_total_wet_surface_area_is_correct_l743_74392


namespace NUMINAMATH_GPT_percent_kindergarten_combined_l743_74309

-- Define the constants provided in the problem
def studentsPinegrove : ℕ := 150
def studentsMaplewood : ℕ := 250

def percentKindergartenPinegrove : ℝ := 18.0
def percentKindergartenMaplewood : ℝ := 14.0

-- The proof statement
theorem percent_kindergarten_combined :
  (27.0 + 35.0) / (150.0 + 250.0) * 100.0 = 15.5 :=
by 
  sorry

end NUMINAMATH_GPT_percent_kindergarten_combined_l743_74309


namespace NUMINAMATH_GPT_find_numbers_with_conditions_l743_74356

theorem find_numbers_with_conditions (n : ℕ) (hn1 : n % 100 = 0) (hn2 : (n.divisors).card = 12) : 
  n = 200 ∨ n = 500 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_with_conditions_l743_74356


namespace NUMINAMATH_GPT_root_expression_value_l743_74355

theorem root_expression_value 
  (p q r s : ℝ)
  (h1 : p + q + r + s = 15)
  (h2 : p*q + p*r + p*s + q*r + q*s + r*s = 35)
  (h3 : p*q*r + p*q*s + q*r*s + p*r*s = 27)
  (h4 : p*q*r*s = 9)
  (h5 : ∀ x : ℝ, x^4 - 15*x^3 + 35*x^2 - 27*x + 9 = 0 → (x = p ∨ x = q ∨ x = r ∨ x = s)) :
  (p / (1 / p + q*r) + q / (1 / q + r*s) + r / (1 / r + s*p) + s / (1 / s + p*q) = 155 / 123) := 
sorry

end NUMINAMATH_GPT_root_expression_value_l743_74355


namespace NUMINAMATH_GPT_point_above_line_l743_74318

-- Define the point P with coordinates (-2, t)
variable (t : ℝ)

-- Define the line equation
def line_eq (x y : ℝ) : ℝ := 2 * x - 3 * y + 6

-- Proving that t must be greater than 2/3 for the point P to be above the line
theorem point_above_line : (line_eq (-2) t < 0) -> t > 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_point_above_line_l743_74318


namespace NUMINAMATH_GPT_fraction_equals_one_l743_74382

/-- Given the fraction (12-11+10-9+8-7+6-5+4-3+2-1) / (1-2+3-4+5-6+7-8+9-10+11),
    prove that its value is equal to 1. -/
theorem fraction_equals_one :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end NUMINAMATH_GPT_fraction_equals_one_l743_74382


namespace NUMINAMATH_GPT_count_integer_values_not_satisfying_inequality_l743_74322

theorem count_integer_values_not_satisfying_inequality : 
  ∃ n : ℕ, 
  (n = 3) ∧ (∀ x : ℤ, (4 * x^2 + 22 * x + 21 ≤ 25) → (-2 ≤ x ∧ x ≤ 0)) :=
by
  sorry

end NUMINAMATH_GPT_count_integer_values_not_satisfying_inequality_l743_74322


namespace NUMINAMATH_GPT_arithmetic_geo_sum_l743_74306

theorem arithmetic_geo_sum (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a (n + 1) = a n + d) →
  (d = 2) →
  (a 3) ^ 2 = (a 1) * (a 4) →
  (a 2 + a 3 = -10) := 
by
  intros h_arith h_d h_geo
  sorry

end NUMINAMATH_GPT_arithmetic_geo_sum_l743_74306


namespace NUMINAMATH_GPT_domain_of_v_l743_74376

noncomputable def v (x : ℝ) : ℝ := 1 / (x ^ (1/3) + x^2 - 1)

theorem domain_of_v : ∀ x, x ≠ 1 → x ^ (1/3) + x^2 - 1 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_v_l743_74376


namespace NUMINAMATH_GPT_find_g_inv_neg_fifteen_sixtyfour_l743_74374

noncomputable def g (x : ℝ) : ℝ := (x^6 - 1) / 4

theorem find_g_inv_neg_fifteen_sixtyfour : g⁻¹ (-15/64) = 1/2 :=
by
  sorry  -- Proof is not required

end NUMINAMATH_GPT_find_g_inv_neg_fifteen_sixtyfour_l743_74374


namespace NUMINAMATH_GPT_final_result_is_110_l743_74329

def chosen_number : ℕ := 63
def multiplier : ℕ := 4
def subtracted_value : ℕ := 142

def final_result : ℕ := (chosen_number * multiplier) - subtracted_value

theorem final_result_is_110 : final_result = 110 := by
  sorry

end NUMINAMATH_GPT_final_result_is_110_l743_74329


namespace NUMINAMATH_GPT_total_hats_l743_74378

theorem total_hats (B G : ℕ) (cost_blue cost_green total_cost green_quantity : ℕ)
  (h1 : cost_blue = 6)
  (h2 : cost_green = 7)
  (h3 : total_cost = 530)
  (h4 : green_quantity = 20)
  (h5 : G = green_quantity)
  (h6 : total_cost = B * cost_blue + G * cost_green) :
  B + G = 85 :=
by
  sorry

end NUMINAMATH_GPT_total_hats_l743_74378


namespace NUMINAMATH_GPT_field_length_l743_74394

theorem field_length (w l: ℕ) (hw1: l = 2 * w) (hw2: 8 * 8 = 64) (hw3: 64 = l * w / 2) : l = 16 := 
by
  sorry

end NUMINAMATH_GPT_field_length_l743_74394


namespace NUMINAMATH_GPT_gcd_266_209_l743_74354

-- Definitions based on conditions
def a : ℕ := 266
def b : ℕ := 209

-- Theorem stating the GCD of a and b
theorem gcd_266_209 : Nat.gcd a b = 19 :=
by {
  -- Declare the specific integers as conditions
  let a := 266
  let b := 209
  -- Use the Euclidean algorithm (steps within the proof are not required)
  -- State that the conclusion is the GCD of a and b 
  sorry
}

end NUMINAMATH_GPT_gcd_266_209_l743_74354


namespace NUMINAMATH_GPT_calculate_monthly_rent_l743_74389

theorem calculate_monthly_rent (P : ℝ) (R : ℝ) (T : ℝ) (M : ℝ) (rent : ℝ) :
  P = 12000 →
  R = 0.06 →
  T = 400 →
  M = 0.1 →
  rent = 103.70 :=
by
  intros hP hR hT hM
  sorry

end NUMINAMATH_GPT_calculate_monthly_rent_l743_74389


namespace NUMINAMATH_GPT_fractional_equation_solution_l743_74337

theorem fractional_equation_solution (x : ℝ) (h₁ : x ≠ 0) : (1 / x = 2 / (x + 3)) → x = 3 := by
  sorry

end NUMINAMATH_GPT_fractional_equation_solution_l743_74337


namespace NUMINAMATH_GPT_pascal_triangle_43rd_element_in_51_row_l743_74369

theorem pascal_triangle_43rd_element_in_51_row :
  (Nat.choose 50 42) = 10272278170 :=
  by
  -- proof construction here
  sorry

end NUMINAMATH_GPT_pascal_triangle_43rd_element_in_51_row_l743_74369


namespace NUMINAMATH_GPT_pentagonal_tiles_count_l743_74381

theorem pentagonal_tiles_count (a b : ℕ) (h1 : a + b = 30) (h2 : 3 * a + 5 * b = 120) : b = 15 :=
by
  sorry

end NUMINAMATH_GPT_pentagonal_tiles_count_l743_74381


namespace NUMINAMATH_GPT_train_speed_computed_l743_74360

noncomputable def train_speed_in_kmh (train_length : ℝ) (platform_length : ℝ) (time_in_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time_in_seconds
  speed_mps * 3.6

theorem train_speed_computed :
  train_speed_in_kmh 250 50.024 15 = 72.006 := by
  sorry

end NUMINAMATH_GPT_train_speed_computed_l743_74360


namespace NUMINAMATH_GPT_min_number_of_squares_l743_74399

theorem min_number_of_squares (length width : ℕ) (h_length : length = 10) (h_width : width = 9) : 
  ∃ n, n = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_number_of_squares_l743_74399


namespace NUMINAMATH_GPT_clothing_discounted_to_fraction_of_original_price_l743_74385

-- Given conditions
variable (P : ℝ) (f : ℝ)

-- Price during first sale is fP, price during second sale is 0.5P
-- Price decreased by 40% from first sale to second sale
def price_decrease_condition : Prop :=
  f * P - (1/2) * P = 0.4 * (f * P)

-- The main theorem to prove
theorem clothing_discounted_to_fraction_of_original_price (h : price_decrease_condition P f) :
  f = 5/6 :=
sorry

end NUMINAMATH_GPT_clothing_discounted_to_fraction_of_original_price_l743_74385


namespace NUMINAMATH_GPT_train_passes_jogger_in_37_seconds_l743_74372

-- Define the parameters
def jogger_speed_kmph : ℝ := 9
def train_speed_kmph : ℝ := 45
def headstart : ℝ := 250
def train_length : ℝ := 120

-- Convert speeds from km/h to m/s
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

-- Calculate relative speed in m/s
noncomputable def relative_speed : ℝ :=
  train_speed_mps - jogger_speed_mps

-- Calculate total distance to be covered in meters
def total_distance : ℝ :=
  headstart + train_length

-- Calculate time taken to pass the jogger in seconds
noncomputable def time_to_pass : ℝ :=
  total_distance / relative_speed

theorem train_passes_jogger_in_37_seconds :
  time_to_pass = 37 :=
by
  -- Proof would be here
  sorry

end NUMINAMATH_GPT_train_passes_jogger_in_37_seconds_l743_74372


namespace NUMINAMATH_GPT_exists_root_between_l743_74384

-- Given definitions and conditions
variables (a b c : ℝ)
variables (ha : a ≠ 0)
variables (x1 x2 : ℝ)
variable (h1 : a * x1^2 + b * x1 + c = 0)    -- root of the first equation
variable (h2 : -a * x2^2 + b * x2 + c = 0)   -- root of the second equation

-- Proof statement
theorem exists_root_between (a b c : ℝ) (ha : a ≠ 0) (x1 x2 : ℝ)
    (h1 : a * x1^2 + b * x1 + c = 0) (h2 : -a * x2^2 + b * x2 + c = 0) :
    ∃ x3 : ℝ, 
      (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) ∧ 
      (1 / 2 * a * x3^2 + b * x3 + c = 0) :=
sorry

end NUMINAMATH_GPT_exists_root_between_l743_74384


namespace NUMINAMATH_GPT_calculate_expr_l743_74327

theorem calculate_expr : 1 - Real.sqrt 9 = -2 := by
  sorry

end NUMINAMATH_GPT_calculate_expr_l743_74327


namespace NUMINAMATH_GPT_find_smaller_number_l743_74350

theorem find_smaller_number (x : ℕ) (hx : x + 4 * x = 45) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l743_74350


namespace NUMINAMATH_GPT_tangent_sum_l743_74349

theorem tangent_sum :
  (Finset.sum (Finset.range 2019) (λ k => Real.tan ((k + 1) * Real.pi / 47) * Real.tan ((k + 2) * Real.pi / 47))) = -2021 :=
by
  -- proof will be completed here
  sorry

end NUMINAMATH_GPT_tangent_sum_l743_74349


namespace NUMINAMATH_GPT_min_abs_val_sum_l743_74338

theorem min_abs_val_sum : ∃ x : ℝ, (∀ y : ℝ, |y - 1| + |y - 2| + |y - 3| ≥ |x - 1| + |x - 2| + |x - 3|) ∧ |x - 1| + |x - 2| + |x - 3| = 1 :=
sorry

end NUMINAMATH_GPT_min_abs_val_sum_l743_74338


namespace NUMINAMATH_GPT_quadratic_inequality_l743_74375

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) 
  (h₁ : quadratic_function a b c 1 = quadratic_function a b c 3) 
  (h₂ : quadratic_function a b c 1 > quadratic_function a b c 4) : 
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l743_74375


namespace NUMINAMATH_GPT_max_blue_cubes_visible_l743_74326

def max_visible_blue_cubes (board : ℕ × ℕ × ℕ → ℕ) : ℕ :=
  board (0, 0, 0)

theorem max_blue_cubes_visible (board : ℕ × ℕ × ℕ → ℕ) :
  max_visible_blue_cubes board = 12 :=
sorry

end NUMINAMATH_GPT_max_blue_cubes_visible_l743_74326


namespace NUMINAMATH_GPT_quadratic_ineq_real_solutions_l743_74361

theorem quadratic_ineq_real_solutions (d : ℝ) (h₀ : 0 < d) :
  (∀ x : ℝ, x^2 - 8 * x + d < 0 → 0 < d ∧ d < 16) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_ineq_real_solutions_l743_74361


namespace NUMINAMATH_GPT_value_of_a3_l743_74304

def a_n (n : ℕ) : ℤ := (-1)^n * (n^2 + 1)

theorem value_of_a3 : a_n 3 = -10 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_value_of_a3_l743_74304


namespace NUMINAMATH_GPT_sqrt_four_eq_pm_two_l743_74303

theorem sqrt_four_eq_pm_two : ∀ (x : ℝ), x * x = 4 ↔ x = 2 ∨ x = -2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_four_eq_pm_two_l743_74303


namespace NUMINAMATH_GPT_at_least_two_inequalities_hold_l743_74340

theorem at_least_two_inequalities_hold 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a + b + c ≥ a * b * c) : 
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨ (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨ (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) := 
sorry

end NUMINAMATH_GPT_at_least_two_inequalities_hold_l743_74340


namespace NUMINAMATH_GPT_rick_iron_clothing_l743_74367

theorem rick_iron_clothing :
  let shirts_per_hour := 4
  let pants_per_hour := 3
  let jackets_per_hour := 2
  let hours_shirts := 3
  let hours_pants := 5
  let hours_jackets := 2
  let total_clothing := (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants) + (jackets_per_hour * hours_jackets)
  total_clothing = 31 := by
  sorry

end NUMINAMATH_GPT_rick_iron_clothing_l743_74367


namespace NUMINAMATH_GPT_triangle_area_l743_74301

open Real

-- Define the conditions
variables (a : ℝ) (B : ℝ) (cosA : ℝ)
variable (S : ℝ)

-- Given conditions of the problem
def triangle_conditions : Prop :=
  a = 5 ∧ B = π / 3 ∧ cosA = 11 / 14

-- State the theorem to be proved
theorem triangle_area (h : triangle_conditions a B cosA) : S = 10 * sqrt 3 :=
sorry

end NUMINAMATH_GPT_triangle_area_l743_74301


namespace NUMINAMATH_GPT_fraction_oil_is_correct_l743_74341

noncomputable def fraction_oil_third_bottle (C : ℚ) (oil1 : ℚ) (oil2 : ℚ) (water1 : ℚ) (water2 : ℚ) := 
  (oil1 + oil2) / (oil1 + oil2 + water1 + water2)

theorem fraction_oil_is_correct (C : ℚ) (hC : C > 0) :
  let oil1 := C / 2
  let oil2 := C / 2
  let water1 := C / 2
  let water2 := 3 * C / 4
  fraction_oil_third_bottle C oil1 oil2 water1 water2 = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_fraction_oil_is_correct_l743_74341


namespace NUMINAMATH_GPT_proposition_equivalence_l743_74393

open Classical

theorem proposition_equivalence
  (p q : Prop) :
  ¬(p ∨ q) ↔ (¬p ∧ ¬q) :=
by sorry

end NUMINAMATH_GPT_proposition_equivalence_l743_74393


namespace NUMINAMATH_GPT_ticket_cost_l743_74370

noncomputable def calculate_cost (x : ℝ) : ℝ :=
  6 * (1.1 * x) + 5 * (x / 2)

theorem ticket_cost (x : ℝ) (h : 4 * (1.1 * x) + 3 * (x / 2) = 28.80) : 
  calculate_cost x = 44.41 := by
  sorry

end NUMINAMATH_GPT_ticket_cost_l743_74370


namespace NUMINAMATH_GPT_total_eggs_l743_74395

noncomputable def total_eggs_in_all_containers (n : ℕ) (f l : ℕ) : ℕ :=
  n * (f * l)

theorem total_eggs (f l : ℕ) :
  (f = 14 + 20 - 1) →
  (l = 3 + 2 - 1) →
  total_eggs_in_all_containers 28 f l = 3696 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_total_eggs_l743_74395


namespace NUMINAMATH_GPT_domain_range_equal_l743_74342

noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

theorem domain_range_equal {a b : ℝ} (hb : b > 0) :
  (∀ y, ∃ x, f a b x = y) ↔ (a = -4 ∨ a = 0) :=
sorry

end NUMINAMATH_GPT_domain_range_equal_l743_74342


namespace NUMINAMATH_GPT_total_meat_supply_l743_74328

-- Definitions of the given conditions
def lion_consumption_per_day : ℕ := 25
def tiger_consumption_per_day : ℕ := 20
def duration_days : ℕ := 2

-- Statement of the proof problem
theorem total_meat_supply :
  (lion_consumption_per_day + tiger_consumption_per_day) * duration_days = 90 :=
by
  sorry

end NUMINAMATH_GPT_total_meat_supply_l743_74328


namespace NUMINAMATH_GPT_english_vocab_related_to_reading_level_l743_74324

theorem english_vocab_related_to_reading_level (N : ℕ) (K_squared : ℝ) (critical_value : ℝ) (p_value : ℝ)
  (hN : N = 100)
  (hK_squared : K_squared = 7)
  (h_critical_value : critical_value = 6.635)
  (h_p_value : p_value = 0.010) :
  p_value <= 0.01 → K_squared > critical_value → true :=
by
  intro h_p_value_le h_K_squared_gt
  sorry

end NUMINAMATH_GPT_english_vocab_related_to_reading_level_l743_74324


namespace NUMINAMATH_GPT_find_smallest_x_l743_74311

noncomputable def smallest_pos_real_x : ℝ :=
  55 / 7

theorem find_smallest_x (x : ℝ) (h : x > 0) (hx : ⌊x^2⌋ - x * ⌊x⌋ = 6) : x = smallest_pos_real_x :=
  sorry

end NUMINAMATH_GPT_find_smallest_x_l743_74311


namespace NUMINAMATH_GPT_s_plus_t_l743_74359

def g (x : ℝ) : ℝ := 3 * x ^ 4 + 9 * x ^ 3 - 7 * x ^ 2 + 2 * x + 4
def h (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

noncomputable def s (x : ℝ) : ℝ := 3 * x ^ 2 + 3
noncomputable def t (x : ℝ) : ℝ := 3 * x + 6

theorem s_plus_t : s 1 + t (-1) = 9 := by
  sorry

end NUMINAMATH_GPT_s_plus_t_l743_74359


namespace NUMINAMATH_GPT_quadratic_roots_abs_eq_l743_74347

theorem quadratic_roots_abs_eq (x1 x2 m : ℝ) (h1 : x1 > 0) (h2 : x2 < 0) 
  (h_eq_roots : ∀ x, x^2 - (x1 + x2)*x + x1*x2 = 0) : 
  ∃ q : ℝ, q = x^2 - (1 - 4*m)/x + 2 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_abs_eq_l743_74347


namespace NUMINAMATH_GPT_matrix_problem_l743_74305

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)
variable (I : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = !![2, 1; 4, 3]) :
  B * A = !![2, 1; 4, 3] :=
sorry

end NUMINAMATH_GPT_matrix_problem_l743_74305


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_sin_2alpha_expr_l743_74336

open Real

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
by
  sorry

theorem sin_2alpha_expr (α : ℝ) (h : tan α = 2) :
  (sin (2 * α)) / (sin (α) ^ 2 + sin (α) * cos (α)) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_sin_2alpha_expr_l743_74336


namespace NUMINAMATH_GPT_dhoni_remaining_earnings_l743_74315

theorem dhoni_remaining_earnings (rent_percent dishwasher_percent : ℝ) 
  (h1 : rent_percent = 20) (h2 : dishwasher_percent = 15) : 
  100 - (rent_percent + dishwasher_percent) = 65 := 
by 
  sorry

end NUMINAMATH_GPT_dhoni_remaining_earnings_l743_74315


namespace NUMINAMATH_GPT_packs_of_red_balls_l743_74380

/-
Julia bought some packs of red balls, R packs.
Julia bought 10 packs of yellow balls.
Julia bought 8 packs of green balls.
There were 19 balls in each package.
Julia bought 399 balls in total.
The goal is to prove that the number of packs of red balls Julia bought, R, is equal to 3.
-/

theorem packs_of_red_balls (R : ℕ) (balls_per_pack : ℕ) (packs_yellow : ℕ) (packs_green : ℕ) (total_balls : ℕ) 
  (h1 : balls_per_pack = 19) (h2 : packs_yellow = 10) (h3 : packs_green = 8) (h4 : total_balls = 399) 
  (h5 : total_balls = R * balls_per_pack + (packs_yellow + packs_green) * balls_per_pack) : 
  R = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_packs_of_red_balls_l743_74380


namespace NUMINAMATH_GPT_range_of_a_l743_74344

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def B (x : ℝ) (a : ℝ) : Prop := x > a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, A x → B x a) → a < -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l743_74344


namespace NUMINAMATH_GPT_beverage_distribution_l743_74379

theorem beverage_distribution (total_cans : ℕ) (number_of_children : ℕ) (hcans : total_cans = 5) (hchildren : number_of_children = 8) :
  (total_cans / number_of_children : ℚ) = 5 / 8 :=
by
  -- Given the conditions
  have htotal_cans : total_cans = 5 := hcans
  have hnumber_of_children : number_of_children = 8 := hchildren
  
  -- we need to show the beverage distribution
  rw [htotal_cans, hnumber_of_children]
  exact by norm_num

end NUMINAMATH_GPT_beverage_distribution_l743_74379


namespace NUMINAMATH_GPT_square_binomial_unique_a_l743_74364

theorem square_binomial_unique_a (a : ℝ) : 
  (∃ r s : ℝ, (ax^2 - 8*x + 16) = (r*x + s)^2) ↔ a = 1 :=
by
  sorry

end NUMINAMATH_GPT_square_binomial_unique_a_l743_74364


namespace NUMINAMATH_GPT_find_multiple_l743_74396

theorem find_multiple (x y m : ℕ) (h1 : y + x = 50) (h2 : y = m * x - 43) (h3 : y = 31) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l743_74396


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_expression_l743_74391

def quadratic_expr (x y : ℝ) : ℝ := x^2 - x * y + y^2

def constraint (x y : ℝ) : Prop := x + y = 5

theorem minimum_value_of_quadratic_expression :
  ∃ m, ∀ x y, constraint x y → quadratic_expr x y ≥ m ∧ (∃ x y, constraint x y ∧ quadratic_expr x y = m) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_expression_l743_74391


namespace NUMINAMATH_GPT_max_lcm_15_2_3_5_6_9_10_l743_74346

theorem max_lcm_15_2_3_5_6_9_10 : 
  max (max (max (max (max (Nat.lcm 15 2) (Nat.lcm 15 3)) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10) = 45 :=
by
  sorry

end NUMINAMATH_GPT_max_lcm_15_2_3_5_6_9_10_l743_74346


namespace NUMINAMATH_GPT_train_speed_l743_74320

theorem train_speed (l t: ℝ) (h1: l = 441) (h2: t = 21) : l / t = 21 := by
  sorry

end NUMINAMATH_GPT_train_speed_l743_74320


namespace NUMINAMATH_GPT_unique_infinite_sequence_l743_74353

-- Defining conditions for the infinite sequence of negative integers
variable (a : ℕ → ℤ)
  
-- Condition 1: Elements in sequence are negative integers
def sequence_negative : Prop :=
  ∀ n, a n < 0 

-- Condition 2: For every positive integer n, the first n elements taken modulo n have n distinct remainders
def distinct_mod_remainders (n : ℕ) : Prop :=
  ∀ i j, i < n → j < n → i ≠ j → (a i % n ≠ a j % n) 

-- The main theorem statement
theorem unique_infinite_sequence (a : ℕ → ℤ) 
  (h1 : sequence_negative a) 
  (h2 : ∀ n, distinct_mod_remainders a n) :
  ∀ k : ℤ, ∃! n, a n = k :=
sorry

end NUMINAMATH_GPT_unique_infinite_sequence_l743_74353


namespace NUMINAMATH_GPT_min_value_exp_l743_74386

theorem min_value_exp (a b : ℝ) (h_condition : a - 3 * b + 6 = 0) : 
  ∃ (m : ℝ), m = 2^a + 1 / 8^b ∧ m ≥ (1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_min_value_exp_l743_74386


namespace NUMINAMATH_GPT_f_odd_f_increasing_on_2_infty_solve_inequality_f_l743_74310

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem f_odd (x : ℝ) (hx : x ≠ 0) : f (-x) = -f x := by
  sorry

theorem f_increasing_on_2_infty (x₁ x₂ : ℝ) (hx₁ : 2 < x₁) (hx₂ : 2 < x₂) (h : x₁ < x₂) : f x₁ < f x₂ := by
  sorry

theorem solve_inequality_f (x : ℝ) (hx : -5 < x ∧ x < -1) : f (2*x^2 + 5*x + 8) + f (x - 3 - x^2) < 0 := by
  sorry

end NUMINAMATH_GPT_f_odd_f_increasing_on_2_infty_solve_inequality_f_l743_74310


namespace NUMINAMATH_GPT_g_g_2_eq_394_l743_74366

def g (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem g_g_2_eq_394 : g (g 2) = 394 :=
by
  sorry

end NUMINAMATH_GPT_g_g_2_eq_394_l743_74366


namespace NUMINAMATH_GPT_intersection_complement_P_CUQ_l743_74335

universe U

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}
def CUQ : Set ℕ := U \ Q

theorem intersection_complement_P_CUQ : 
  (P ∩ CUQ) = {1, 2} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_complement_P_CUQ_l743_74335


namespace NUMINAMATH_GPT_brenda_bought_stones_l743_74373

-- Given Conditions
def n_bracelets : ℕ := 3
def n_stones_per_bracelet : ℕ := 12

-- Problem Statement: Prove Betty bought the correct number of stone-shaped stars
theorem brenda_bought_stones :
  let n_total_stones := n_bracelets * n_stones_per_bracelet
  n_total_stones = 36 := 
by 
  -- proof goes here, but we omit it with sorry
  sorry

end NUMINAMATH_GPT_brenda_bought_stones_l743_74373


namespace NUMINAMATH_GPT_gov_addresses_l743_74331

theorem gov_addresses (S H K : ℕ) 
  (H1 : S = 2 * H) 
  (H2 : K = S + 10) 
  (H3 : S + H + K = 40) : 
  S = 12 := 
sorry 

end NUMINAMATH_GPT_gov_addresses_l743_74331


namespace NUMINAMATH_GPT_hot_drinks_prediction_at_2_deg_l743_74390

-- Definition of the regression equation as a function
def regression_equation (x : ℝ) : ℝ :=
  -2.35 * x + 147.77

-- The statement to be proved
theorem hot_drinks_prediction_at_2_deg :
  abs (regression_equation 2 - 143) < 1 :=
sorry

end NUMINAMATH_GPT_hot_drinks_prediction_at_2_deg_l743_74390


namespace NUMINAMATH_GPT_tan_quadruple_angle_l743_74377

theorem tan_quadruple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (4 * θ) = -24 / 7 :=
sorry

end NUMINAMATH_GPT_tan_quadruple_angle_l743_74377


namespace NUMINAMATH_GPT_number_of_cows_l743_74333

def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def chicken_cost : ℕ := 100 * 5
def installation_cost : ℕ := 6 * 100
def equipment_cost : ℕ := 6000
def total_cost : ℕ := 147700

theorem number_of_cows : 
  (total_cost - (land_cost + house_cost + chicken_cost + installation_cost + equipment_cost)) / 1000 = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_cows_l743_74333


namespace NUMINAMATH_GPT_amount_of_bill_l743_74358

noncomputable def TD : ℝ := 360
noncomputable def BD : ℝ := 418.9090909090909
noncomputable def FV (TD BD : ℝ) : ℝ := TD * BD / (BD - TD)

theorem amount_of_bill :
  FV TD BD = 2568 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_bill_l743_74358


namespace NUMINAMATH_GPT_girls_ran_9_miles_l743_74368

def boys_laps : ℕ := 34
def additional_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6

def girls_laps : ℕ := boys_laps + additional_laps
def girls_miles : ℚ := girls_laps * lap_distance

theorem girls_ran_9_miles : girls_miles = 9 := by
  sorry

end NUMINAMATH_GPT_girls_ran_9_miles_l743_74368


namespace NUMINAMATH_GPT_frank_is_15_years_younger_than_john_l743_74319

variables (F J : ℕ)

theorem frank_is_15_years_younger_than_john
  (h1 : J + 3 = 2 * (F + 3))
  (h2 : F + 4 = 16) : J - F = 15 := by
  sorry

end NUMINAMATH_GPT_frank_is_15_years_younger_than_john_l743_74319


namespace NUMINAMATH_GPT_sara_letters_ratio_l743_74363

variable (L_J : ℕ) (L_F : ℕ) (L_T : ℕ)

theorem sara_letters_ratio (hLJ : L_J = 6) (hLF : L_F = 9) (hLT : L_T = 33) : 
  (L_T - (L_J + L_F)) / L_J = 3 := by
  sorry

end NUMINAMATH_GPT_sara_letters_ratio_l743_74363


namespace NUMINAMATH_GPT_find_wall_width_l743_74397

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.1325
def brick_height : ℝ := 0.08

-- Define the dimensions of the wall in meters
def wall_length : ℝ := 7
def wall_height : ℝ := 15.5
def number_of_bricks : ℝ := 4094.3396226415093

-- Volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Total volume of bricks used
def total_brick_volume : ℝ := number_of_bricks * brick_volume

-- Wall volume in terms of width W
def wall_volume (W : ℝ) : ℝ := wall_length * W * wall_height

-- The theorem we want to prove
theorem find_wall_width (W : ℝ) (h : wall_volume W = total_brick_volume) : W = 0.08 := by
  sorry

end NUMINAMATH_GPT_find_wall_width_l743_74397


namespace NUMINAMATH_GPT_abigail_money_loss_l743_74348

theorem abigail_money_loss
  (initial_amount : ℕ)
  (spent_amount : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  initial_amount - spent_amount - remaining_amount = 6 :=
by sorry

end NUMINAMATH_GPT_abigail_money_loss_l743_74348


namespace NUMINAMATH_GPT_general_formula_constant_c_value_l743_74343

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) - a n = d

-- Given sequence {a_n}
variables {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ)
-- Conditions
variables (h1 : a 3 * a 4 = 117) (h2 : a 2 + a 5 = 22) (hd_pos : d > 0)
-- Proof that the general formula for the sequence {a_n} is a_n = 4n - 3
theorem general_formula :
  (∀ n, a n = 4 * n - 3) :=
sorry

-- Given new sequence {b_n}
variables (b : ℕ → ℕ → ℝ) {c : ℝ} (hc : c ≠ 0)
-- New condition that bn is an arithmetic sequence
variables (h_b1 : b 1 = S 1 / (1 + c)) (h_b2 : b 2 = S 2 / (2 + c)) (h_b3 : b 3 = S 3 / (3 + c))
-- Proof that c = -1/2 is the correct constant
theorem constant_c_value :
  (c = -1 / 2) :=
sorry

end NUMINAMATH_GPT_general_formula_constant_c_value_l743_74343


namespace NUMINAMATH_GPT_invalid_root_l743_74316

theorem invalid_root (a_1 a_0 : ℤ) : ¬(19 * (1/7 : ℚ)^3 + 98 * (1/7 : ℚ)^2 + a_1 * (1/7 : ℚ) + a_0 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_invalid_root_l743_74316


namespace NUMINAMATH_GPT_harvest_season_duration_l743_74332

theorem harvest_season_duration (weekly_rent : ℕ) (total_rent_paid : ℕ) : 
    (weekly_rent = 388) →
    (total_rent_paid = 527292) →
    (total_rent_paid / weekly_rent = 1360) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_harvest_season_duration_l743_74332


namespace NUMINAMATH_GPT_equilateral_triangle_l743_74362

theorem equilateral_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) 
  (h3 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) 
  (h4 : b = c) : 
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_equilateral_triangle_l743_74362


namespace NUMINAMATH_GPT_age_difference_l743_74387

variable (A B C : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 13) : (A + B) - (B + C) = 13 := by
  sorry

end NUMINAMATH_GPT_age_difference_l743_74387


namespace NUMINAMATH_GPT_find_d_share_l743_74398

def money_distribution (a b c d : ℕ) (x : ℕ) := 
  a = 5 * x ∧ 
  b = 2 * x ∧ 
  c = 4 * x ∧ 
  d = 3 * x ∧ 
  (c = d + 500)

theorem find_d_share (a b c d x : ℕ) (h : money_distribution a b c d x) : d = 1500 :=
by
  --proof would go here
  sorry

end NUMINAMATH_GPT_find_d_share_l743_74398


namespace NUMINAMATH_GPT_percentage_increase_each_year_is_50_l743_74312

-- Definitions based on conditions
def students_passed_three_years_ago : ℕ := 200
def students_passed_this_year : ℕ := 675

-- The prove statement
theorem percentage_increase_each_year_is_50
    (N3 N0 : ℕ)
    (P : ℚ)
    (h1 : N3 = students_passed_three_years_ago)
    (h2 : N0 = students_passed_this_year)
    (h3 : N0 = N3 * (1 + P)^3) :
  P = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_each_year_is_50_l743_74312


namespace NUMINAMATH_GPT_similarity_transformation_l743_74325

theorem similarity_transformation (C C' : ℝ × ℝ) (r : ℝ) (h1 : r = 3) (h2 : C = (4, 1))
  (h3 : C' = (r * 4, r * 1)) : (C' = (12, 3) ∨ C' = (-12, -3)) := by
  sorry

end NUMINAMATH_GPT_similarity_transformation_l743_74325


namespace NUMINAMATH_GPT_sum_of_numbers_l743_74308

theorem sum_of_numbers : 
  12345 + 23451 + 34512 + 45123 + 51234 = 166665 := 
sorry

end NUMINAMATH_GPT_sum_of_numbers_l743_74308


namespace NUMINAMATH_GPT_doug_initial_marbles_l743_74371

theorem doug_initial_marbles 
  (ed_marbles : ℕ)
  (doug_marbles : ℕ)
  (lost_marbles : ℕ)
  (ed_condition : ed_marbles = doug_marbles + 5)
  (lost_condition : lost_marbles = 3)
  (ed_value : ed_marbles = 27) :
  doug_marbles + lost_marbles = 25 :=
by
  sorry

end NUMINAMATH_GPT_doug_initial_marbles_l743_74371


namespace NUMINAMATH_GPT_inequality_solution_l743_74352

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (x + 2) ≤ 3 ↔ x ∈ Set.Iic (-6) ∪ Set.Ioi (-2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l743_74352


namespace NUMINAMATH_GPT_total_amount_spent_l743_74334

-- Define the variables B and D representing the amounts Ben and David spent.
variables (B D : ℝ)

-- Define the conditions based on the given problem.
def conditions : Prop :=
  (D = 0.60 * B) ∧ (B = D + 14)

-- The main theorem stating the total amount spent by Ben and David is 56.
theorem total_amount_spent (h : conditions B D) : B + D = 56 :=
sorry  -- Proof omitted.

end NUMINAMATH_GPT_total_amount_spent_l743_74334


namespace NUMINAMATH_GPT_intersecting_diagonals_of_parallelogram_l743_74323

theorem intersecting_diagonals_of_parallelogram (A C : ℝ × ℝ) (hA : A = (2, -3)) (hC : C = (14, 9)) :
    ∃ M : ℝ × ℝ, M = (8, 3) ∧ M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_intersecting_diagonals_of_parallelogram_l743_74323


namespace NUMINAMATH_GPT_cos_pi_plus_alpha_l743_74357

-- Define the angle α and conditions given
variable (α : Real) (h1 : 0 < α) (h2 : α < π/2)

-- Given condition sine of α
variable (h3 : Real.sin α = 4/5)

-- Define the cosine identity to prove the assertion
theorem cos_pi_plus_alpha (h1 : 0 < α) (h2 : α < π/2) (h3 : Real.sin α = 4/5) :
  Real.cos (π + α) = -3/5 :=
sorry

end NUMINAMATH_GPT_cos_pi_plus_alpha_l743_74357


namespace NUMINAMATH_GPT_smallest_positive_integer_b_l743_74314
-- Import the necessary library

-- Define the conditions and problem statement
def smallest_b_factors (r s : ℤ) := r + s

theorem smallest_positive_integer_b :
  ∃ r s : ℤ, r * s = 1800 ∧ ∀ r' s' : ℤ, r' * s' = 1800 → smallest_b_factors r s ≤ smallest_b_factors r' s' :=
by
  -- Declare that the smallest positive integer b satisfying the conditions is 85
  use 45, 40
  -- Check the core condition
  have rs_eq_1800 := (45 * 40 = 1800)
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_b_l743_74314


namespace NUMINAMATH_GPT_range_of_a_plus_b_l743_74307

variable {a b : ℝ}

-- Assumptions
def are_positive_and_unequal (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b
def equation_holds (a b : ℝ) : Prop := a^2 - a + b^2 - b + a * b = 0

-- Problem Statement
theorem range_of_a_plus_b (h₁ : are_positive_and_unequal a b) (h₂ : equation_holds a b) : 1 < a + b ∧ a + b < 4 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_plus_b_l743_74307
