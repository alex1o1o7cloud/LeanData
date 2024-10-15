import Mathlib

namespace NUMINAMATH_GPT_solve_for_a_l2332_233218

theorem solve_for_a {a x : ℝ} (H : (x - 2) * (a * x^2 - x + 1) = a * x^3 + (-1 - 2 * a) * x^2 + 3 * x - 2 ∧ (-1 - 2 * a) = 0) : a = -1/2 := sorry

end NUMINAMATH_GPT_solve_for_a_l2332_233218


namespace NUMINAMATH_GPT_min_students_with_blue_eyes_and_backpack_l2332_233283

theorem min_students_with_blue_eyes_and_backpack :
  ∀ (students : Finset ℕ), 
  (∀ s, s ∈ students → s = 1) →
  ∃ A B : Finset ℕ, 
    A.card = 18 ∧ B.card = 24 ∧ students.card = 35 ∧ 
    (A ∩ B).card ≥ 7 :=
by
  sorry

end NUMINAMATH_GPT_min_students_with_blue_eyes_and_backpack_l2332_233283


namespace NUMINAMATH_GPT_infinite_series_equals_3_l2332_233254

noncomputable def infinite_series_sum := ∑' (k : ℕ), (12^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))

theorem infinite_series_equals_3 : infinite_series_sum = 3 := by
  sorry

end NUMINAMATH_GPT_infinite_series_equals_3_l2332_233254


namespace NUMINAMATH_GPT_original_cost_price_l2332_233219

theorem original_cost_price (C : ℝ) (h : C + 0.15 * C + 0.05 * C + 0.10 * C = 6400) : C = 4923 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_price_l2332_233219


namespace NUMINAMATH_GPT_tangent_line_at_1_f_positive_iff_a_leq_2_l2332_233200

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_at_1 (a : ℝ) (h : a = 4) : 
  ∃ k b : ℝ, (k = -2) ∧ (b = 2) ∧ (∀ x : ℝ, f x a = k * (x - 1) + b) :=
sorry

theorem f_positive_iff_a_leq_2 : 
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_at_1_f_positive_iff_a_leq_2_l2332_233200


namespace NUMINAMATH_GPT_greatest_divisor_of_420_and_90_l2332_233275

-- Define divisibility
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- Main problem statement
theorem greatest_divisor_of_420_and_90 {d : ℕ} :
  (divides d 420) ∧ (d < 60) ∧ (divides d 90) → d ≤ 30 := 
sorry

end NUMINAMATH_GPT_greatest_divisor_of_420_and_90_l2332_233275


namespace NUMINAMATH_GPT_total_amount_received_l2332_233222

def initial_price_tv : ℕ := 500
def tv_increase_rate : ℚ := 2 / 5
def initial_price_phone : ℕ := 400
def phone_increase_rate : ℚ := 0.40

theorem total_amount_received :
  initial_price_tv + initial_price_tv * tv_increase_rate + initial_price_phone + initial_price_phone * phone_increase_rate = 1260 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_received_l2332_233222


namespace NUMINAMATH_GPT_Dan_must_exceed_speed_l2332_233280

theorem Dan_must_exceed_speed (distance : ℝ) (Cara_speed : ℝ) (delay : ℝ) (time_Cara : ℝ) (Dan_time : ℝ) : 
  distance = 120 ∧ Cara_speed = 30 ∧ delay = 1 ∧ time_Cara = distance / Cara_speed ∧ time_Cara = 4 ∧ Dan_time = time_Cara - delay ∧ Dan_time < 4 → 
  (distance / Dan_time) > 40 :=
by
  sorry

end NUMINAMATH_GPT_Dan_must_exceed_speed_l2332_233280


namespace NUMINAMATH_GPT_subtraction_is_addition_of_negatives_l2332_233260

theorem subtraction_is_addition_of_negatives : (-1) - 3 = -4 := by
  sorry

end NUMINAMATH_GPT_subtraction_is_addition_of_negatives_l2332_233260


namespace NUMINAMATH_GPT_debate_team_has_11_boys_l2332_233286

def debate_team_boys_count (num_groups : Nat) (members_per_group : Nat) (num_girls : Nat) : Nat :=
  let total_members := num_groups * members_per_group
  total_members - num_girls

theorem debate_team_has_11_boys :
  debate_team_boys_count 8 7 45 = 11 :=
by
  sorry

end NUMINAMATH_GPT_debate_team_has_11_boys_l2332_233286


namespace NUMINAMATH_GPT_artist_paints_33_square_meters_l2332_233215

/-
Conditions:
1. The artist has 14 cubes.
2. Each cube has an edge of 1 meter.
3. The cubes are arranged in a pyramid-like structure with three layers.
4. The top layer has 1 cube, the middle layer has 4 cubes, and the bottom layer has 9 cubes.
-/

def exposed_surface_area (num_cubes : Nat) (layer1 : Nat) (layer2 : Nat) (layer3 : Nat) : Nat :=
  let layer1_area := 5 -- Each top layer cube has 5 faces exposed
  let layer2_edge_cubes := 4 -- Count of cubes on the edge in middle layer
  let layer2_area := layer2_edge_cubes * 3 -- Each middle layer edge cube has 3 faces exposed
  let layer3_area := 9 -- Each bottom layer cube has 1 face exposed
  let top_faces := layer1 + layer2 + layer3 -- All top faces exposed
  layer1_area + layer2_area + layer3_area + top_faces

theorem artist_paints_33_square_meters :
  exposed_surface_area 14 1 4 9 = 33 := 
sorry

end NUMINAMATH_GPT_artist_paints_33_square_meters_l2332_233215


namespace NUMINAMATH_GPT_theta_terminal_side_l2332_233216

theorem theta_terminal_side (alpha : ℝ) (theta : ℝ) (h1 : alpha = 1560) (h2 : -360 < theta ∧ theta < 360) :
    (theta = 120 ∨ theta = -240) := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_theta_terminal_side_l2332_233216


namespace NUMINAMATH_GPT_find_mass_of_aluminum_l2332_233279

noncomputable def mass_of_aluminum 
  (rho_A : ℝ) (rho_M : ℝ) (delta_m : ℝ) : ℝ :=
  rho_A * delta_m / (rho_M - rho_A)

theorem find_mass_of_aluminum :
  mass_of_aluminum 2700 8900 0.06 = 26 := by
  sorry

end NUMINAMATH_GPT_find_mass_of_aluminum_l2332_233279


namespace NUMINAMATH_GPT_smallest_possible_value_l2332_233237

theorem smallest_possible_value (x : ℕ) (m : ℕ) :
  (x > 0) →
  (Nat.gcd 36 m = x + 3) →
  (Nat.lcm 36 m = x * (x + 3)) →
  m = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_l2332_233237


namespace NUMINAMATH_GPT_min_value_of_expr_l2332_233281

noncomputable def min_value (x y : ℝ) : ℝ :=
  (4 * x^2) / (y + 1) + (y^2) / (2*x + 2)

theorem min_value_of_expr : 
  ∀ (x y : ℝ), (0 < x) → (0 < y) → (2 * x + y = 2) →
  min_value x y = 4 / 5 :=
by
  intros x y hx hy hxy
  sorry

end NUMINAMATH_GPT_min_value_of_expr_l2332_233281


namespace NUMINAMATH_GPT_solve_system_unique_solution_l2332_233229

theorem solve_system_unique_solution:
  ∃! (x y : ℚ), 3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ x = 57 / 31 ∧ y = 97 / 31 := by
  sorry

end NUMINAMATH_GPT_solve_system_unique_solution_l2332_233229


namespace NUMINAMATH_GPT_number_of_lists_correct_l2332_233274

noncomputable def number_of_lists : Nat :=
  15 ^ 4

theorem number_of_lists_correct :
  number_of_lists = 50625 := by
  sorry

end NUMINAMATH_GPT_number_of_lists_correct_l2332_233274


namespace NUMINAMATH_GPT_annual_income_of_A_l2332_233259

def monthly_income_ratios (A_income B_income : ℝ) : Prop := A_income / B_income = 5 / 2
def B_income_increase (B_income C_income : ℝ) : Prop := B_income = C_income + 0.12 * C_income

theorem annual_income_of_A (A_income B_income C_income : ℝ)
  (h1 : monthly_income_ratios A_income B_income)
  (h2 : B_income_increase B_income C_income)
  (h3 : C_income = 13000) :
  12 * A_income = 436800 :=
by 
  sorry

end NUMINAMATH_GPT_annual_income_of_A_l2332_233259


namespace NUMINAMATH_GPT_find_b_l2332_233228

theorem find_b (a b : ℝ) (h₁ : 2 * a + 3 = 5) (h₂ : b - a = 2) : b = 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_b_l2332_233228


namespace NUMINAMATH_GPT_range_of_a_if_in_first_quadrant_l2332_233241

noncomputable def is_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem range_of_a_if_in_first_quadrant (a : ℝ) :
  is_first_quadrant ((1 + a * Complex.I) / (2 - Complex.I)) ↔ (-1/2 : ℝ) < a ∧ a < 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_if_in_first_quadrant_l2332_233241


namespace NUMINAMATH_GPT_taxi_fare_80_miles_l2332_233206

theorem taxi_fare_80_miles (fare_60 : ℝ) (flat_rate : ℝ) (proportional_rate : ℝ) (d : ℝ) (charge_60 : ℝ) 
  (h1 : fare_60 = 150) (h2 : flat_rate = 20) (h3 : proportional_rate * 60 = charge_60) (h4 : charge_60 = (fare_60 - flat_rate)) 
  (h5 : proportional_rate * 80 = d - flat_rate) : d = 193 := 
by
  sorry

end NUMINAMATH_GPT_taxi_fare_80_miles_l2332_233206


namespace NUMINAMATH_GPT_minimum_positive_period_l2332_233268

open Real

noncomputable def function := fun x : ℝ => 3 * sin (2 * x + π / 3)

theorem minimum_positive_period : ∃ T > 0, ∀ x, function (x + T) = function x ∧ (∀ T', T' > 0 → (∀ x, function (x + T') = function x) → T ≤ T') :=
  sorry

end NUMINAMATH_GPT_minimum_positive_period_l2332_233268


namespace NUMINAMATH_GPT_find_larger_number_l2332_233294

theorem find_larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
  sorry

end NUMINAMATH_GPT_find_larger_number_l2332_233294


namespace NUMINAMATH_GPT_find_k_value_l2332_233225

theorem find_k_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (2 * x1^2 + k * x1 - 2 * k + 1 = 0) ∧ 
                (2 * x2^2 + k * x2 - 2 * k + 1 = 0) ∧ 
                (x1 ≠ x2)) ∧
  ((x1^2 + x2^2 = 29/4)) ↔ (k = 3) := 
sorry

end NUMINAMATH_GPT_find_k_value_l2332_233225


namespace NUMINAMATH_GPT_percentage_shoes_polished_l2332_233205

theorem percentage_shoes_polished (total_pairs : ℕ) (shoes_to_polish : ℕ)
  (total_individual_shoes : ℕ := total_pairs * 2)
  (shoes_polished : ℕ := total_individual_shoes - shoes_to_polish)
  (percentage_polished : ℚ := (shoes_polished : ℚ) / total_individual_shoes * 100) :
  total_pairs = 10 → shoes_to_polish = 11 → percentage_polished = 45 :=
by
  intros hpairs hleft
  sorry

end NUMINAMATH_GPT_percentage_shoes_polished_l2332_233205


namespace NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l2332_233242

theorem distance_between_foci_of_hyperbola :
  (∀ x y : ℝ, (y = 2 * x + 3) ∨ (y = -2 * x + 1)) →
  ∀ p : ℝ × ℝ, (p = (2, 1)) →
  ∃ d : ℝ, d = 2 * Real.sqrt 30 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l2332_233242


namespace NUMINAMATH_GPT_integer_solutions_count_l2332_233236

theorem integer_solutions_count (B : ℤ) (C : ℤ) (h : B = 3) : C = 4 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_count_l2332_233236


namespace NUMINAMATH_GPT_garden_area_increase_l2332_233248

theorem garden_area_increase :
  let length := 80
  let width := 20
  let additional_fence := 60
  let original_area := length * width
  let original_perimeter := 2 * (length + width)
  let total_fence := original_perimeter + additional_fence
  let side_of_square := total_fence / 4
  let square_area := side_of_square * side_of_square
  square_area - original_area = 2625 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_increase_l2332_233248


namespace NUMINAMATH_GPT_inequality_least_one_l2332_233285

theorem inequality_least_one {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4) :=
by
  sorry

end NUMINAMATH_GPT_inequality_least_one_l2332_233285


namespace NUMINAMATH_GPT_tetrahedron_inequality_l2332_233226

variables (S A B C : Point)
variables (SA SB SC : Real)
variables (ABC : Plane)
variables (z : Real)
variable (h1 : angle B S C = π / 2)
variable (h2 : Project (point S) ABC = Orthocenter triangle ABC)
variable (h3 : RadiusInscribedCircle triangle ABC = z)

theorem tetrahedron_inequality :
  SA^2 + SB^2 + SC^2 >= 18 * z^2 :=
sorry

end NUMINAMATH_GPT_tetrahedron_inequality_l2332_233226


namespace NUMINAMATH_GPT_max_cities_l2332_233276

theorem max_cities (n : ℕ) (h1 : ∀ (c : Fin n), ∃ (neighbors : Finset (Fin n)), neighbors.card ≤ 3 ∧ c ∈ neighbors) (h2 : ∀ (c1 c2 : Fin n), c1 ≠ c2 → ∃ c : Fin n, c1 ≠ c ∧ c2 ≠ c) : n ≤ 10 := 
sorry

end NUMINAMATH_GPT_max_cities_l2332_233276


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l2332_233220
open Classical

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 1 → x^2 ≥ 3)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 < 3) := 
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l2332_233220


namespace NUMINAMATH_GPT_determine_digits_in_base_l2332_233266

theorem determine_digits_in_base (x y z b : ℕ) (h1 : 1993 = x * b^2 + y * b + z) (h2 : x + y + z = 22) :
  x = 2 ∧ y = 15 ∧ z = 5 ∧ b = 28 :=
sorry

end NUMINAMATH_GPT_determine_digits_in_base_l2332_233266


namespace NUMINAMATH_GPT_average_of_middle_three_l2332_233269

-- Define the conditions based on the problem statement
def isPositiveWhole (n: ℕ) := n > 0
def areDifferent (a b c d e: ℕ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def isMaximumDifference (a b c d e: ℕ) := max a (max b (max c (max d e))) - min a (min b (min c (min d e)))
def isSecondSmallest (a b c d e: ℕ) := b = 3 ∧ (a < b ∧ (c < b ∨ d < b ∨ e < b) ∧ areDifferent a b c d e)
def totalSumIs30 (a b c d e: ℕ) := a + b + c + d + e = 30

-- Average of the middle three numbers calculated
theorem average_of_middle_three {a b c d e: ℕ} (cond1: isPositiveWhole a)
  (cond2: isPositiveWhole b) (cond3: isPositiveWhole c) (cond4: isPositiveWhole d)
  (cond5: isPositiveWhole e) (cond6: areDifferent a b c d e) (cond7: b = 3)
  (cond8: max a (max c (max d e)) - min a (min c (min d e)) = 16)
  (cond9: totalSumIs30 a b c d e) : (a + c + d) / 3 = 4 :=
by sorry

end NUMINAMATH_GPT_average_of_middle_three_l2332_233269


namespace NUMINAMATH_GPT_sqrt_product_eq_l2332_233255

theorem sqrt_product_eq : Real.sqrt (5^2 * 7^6) = 1715 := by
  sorry

end NUMINAMATH_GPT_sqrt_product_eq_l2332_233255


namespace NUMINAMATH_GPT_square_area_from_conditions_l2332_233244

theorem square_area_from_conditions :
  ∀ (r s l b : ℝ), 
  l = r / 4 →
  r = s →
  l * b = 35 →
  b = 5 →
  s^2 = 784 := 
by 
  intros r s l b h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_square_area_from_conditions_l2332_233244


namespace NUMINAMATH_GPT_part1_part2_l2332_233282

noncomputable def f (a : ℝ) (x : ℝ) := (a * x - 1) * (x - 1)

theorem part1 (h : ∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 2) : a = 1/2 :=
  sorry

theorem part2 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f a x < 0 ↔ 1 < x ∧ x < 1/a) ∨
  (a = 1 → ∀ x : ℝ, ¬(f a x < 0)) ∨
  (∀ x : ℝ, f a x < 0 ↔ 1/a < x ∧ x < 1) :=
  sorry

end NUMINAMATH_GPT_part1_part2_l2332_233282


namespace NUMINAMATH_GPT_sin_add_alpha_l2332_233295

theorem sin_add_alpha (α : ℝ) (h : Real.cos (α - π / 3) = -1 / 2) : 
    Real.sin (π / 6 + α) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_sin_add_alpha_l2332_233295


namespace NUMINAMATH_GPT_find_p_from_parabola_and_distance_l2332_233265

theorem find_p_from_parabola_and_distance 
  (p : ℝ) (hp : p > 0) 
  (M : ℝ × ℝ) (hM : M = (8 / p, 4))
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (hMF : dist M F = 4) : 
  p = 4 :=
sorry

end NUMINAMATH_GPT_find_p_from_parabola_and_distance_l2332_233265


namespace NUMINAMATH_GPT_min_inequality_l2332_233277

theorem min_inequality (r s u v : ℝ) : 
  min (min (r - s^2) (min (s - u^2) (min (u - v^2) (v - r^2)))) ≤ 1 / 4 :=
by sorry

end NUMINAMATH_GPT_min_inequality_l2332_233277


namespace NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l2332_233271

variables (p q : Prop)

theorem condition_sufficient_but_not_necessary (hpq : ∀ q, (¬p → ¬q)) (hpns : ¬ (¬p → ¬q ↔ p → q)) : (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l2332_233271


namespace NUMINAMATH_GPT_find_line_equation_l2332_233235

-- Define the conditions for the x-intercept and inclination angle
def x_intercept (x : ℝ) (line : ℝ → ℝ) : Prop :=
  line x = 0

def inclination_angle (θ : ℝ) (k : ℝ) : Prop :=
  k = Real.tan θ

-- Define the properties of the line we're working with
def line (x : ℝ) : ℝ := -x + 5

theorem find_line_equation :
  x_intercept 5 line ∧ inclination_angle (3 * Real.pi / 4) (-1) → (∀ x, line x = -x + 5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_line_equation_l2332_233235


namespace NUMINAMATH_GPT_rainfall_second_week_l2332_233214

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) (first_week_rainfall : ℝ) (second_week_rainfall : ℝ) :
  total_rainfall = 35 →
  ratio = 1.5 →
  total_rainfall = first_week_rainfall + second_week_rainfall →
  second_week_rainfall = ratio * first_week_rainfall →
  second_week_rainfall = 21 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rainfall_second_week_l2332_233214


namespace NUMINAMATH_GPT_simplified_factorial_fraction_l2332_233252

theorem simplified_factorial_fraction :
  (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_simplified_factorial_fraction_l2332_233252


namespace NUMINAMATH_GPT_probability_blue_given_popped_is_18_over_53_l2332_233256

section PopcornProblem

/-- Representation of probabilities -/
def prob_white : ℚ := 1 / 2
def prob_yellow : ℚ := 1 / 4
def prob_blue : ℚ := 1 / 4

def pop_white_given_white : ℚ := 1 / 2
def pop_yellow_given_yellow : ℚ := 3 / 4
def pop_blue_given_blue : ℚ := 9 / 10

/-- Joint probabilities of kernel popping -/
def prob_white_popped : ℚ := prob_white * pop_white_given_white
def prob_yellow_popped : ℚ := prob_yellow * pop_yellow_given_yellow
def prob_blue_popped : ℚ := prob_blue * pop_blue_given_blue

/-- Total probability of popping -/
def prob_popped : ℚ := prob_white_popped + prob_yellow_popped + prob_blue_popped

/-- Conditional probability of being a blue kernel given that it popped -/
def prob_blue_given_popped : ℚ := prob_blue_popped / prob_popped

/-- The main theorem to prove the final probability -/
theorem probability_blue_given_popped_is_18_over_53 :
  prob_blue_given_popped = 18 / 53 :=
by sorry

end PopcornProblem

end NUMINAMATH_GPT_probability_blue_given_popped_is_18_over_53_l2332_233256


namespace NUMINAMATH_GPT_digits_C_not_make_1C34_divisible_by_4_l2332_233213

theorem digits_C_not_make_1C34_divisible_by_4 :
  ∀ (C : ℕ), (C ≥ 0) ∧ (C ≤ 9) → ¬ (1034 + 100 * C) % 4 = 0 :=
by sorry

end NUMINAMATH_GPT_digits_C_not_make_1C34_divisible_by_4_l2332_233213


namespace NUMINAMATH_GPT_night_crew_fraction_of_day_l2332_233253

variable (D : ℕ) -- Number of workers in the day crew
variable (N : ℕ) -- Number of workers in the night crew
variable (total_boxes : ℕ) -- Total number of boxes loaded by both crews

-- Given conditions
axiom day_fraction : D > 0 ∧ N > 0 ∧ total_boxes > 0
axiom night_workers_fraction : N = (4 * D) / 5
axiom day_crew_boxes_fraction : (5 * total_boxes) / 7 = (5 * D)
axiom night_crew_boxes_fraction : (2 * total_boxes) / 7 = (2 * N)

-- To prove
theorem night_crew_fraction_of_day : 
  let F_d := (5 : ℚ) / (7 * D)
  let F_n := (2 : ℚ) / (7 * N)
  F_n = (5 / 14) * F_d :=
by
  sorry

end NUMINAMATH_GPT_night_crew_fraction_of_day_l2332_233253


namespace NUMINAMATH_GPT_integral_sin3_cos_l2332_233221

open Real

theorem integral_sin3_cos :
  ∫ z in (π / 4)..(π / 2), sin z ^ 3 * cos z = 3 / 16 := by
  sorry

end NUMINAMATH_GPT_integral_sin3_cos_l2332_233221


namespace NUMINAMATH_GPT_additional_kgs_l2332_233288

variables (P R A : ℝ)
variables (h1 : R = 0.80 * P) (h2 : R = 34.2) (h3 : 684 = A * R)

theorem additional_kgs :
  A = 20 :=
by
  sorry

end NUMINAMATH_GPT_additional_kgs_l2332_233288


namespace NUMINAMATH_GPT_parallel_lines_a_l2332_233273

-- Definitions of the lines
def l1 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y - 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := 6 * x + a * y + 2 = 0

-- The main theorem to prove
theorem parallel_lines_a (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = 3) := 
sorry

end NUMINAMATH_GPT_parallel_lines_a_l2332_233273


namespace NUMINAMATH_GPT_inequality_solution_l2332_233247

theorem inequality_solution (x: ℝ) (h1: x ≠ -1) (h2: x ≠ 0) :
  (x-2)/(x+1) + (x-3)/(3*x) ≥ 2 ↔ x ∈ Set.Iic (-3) ∪ Set.Icc (-1) (-1/2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2332_233247


namespace NUMINAMATH_GPT_train_speed_equals_36_0036_l2332_233251

noncomputable def train_speed (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_equals_36_0036 :
  train_speed 70 6.999440044796416 = 36.0036 :=
by
  unfold train_speed
  sorry

end NUMINAMATH_GPT_train_speed_equals_36_0036_l2332_233251


namespace NUMINAMATH_GPT_remaining_seeds_l2332_233211

def initial_seeds : Nat := 54000
def seeds_per_zone : Nat := 3123
def number_of_zones : Nat := 7

theorem remaining_seeds (initial_seeds seeds_per_zone number_of_zones : Nat) : 
  initial_seeds - (seeds_per_zone * number_of_zones) = 32139 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_seeds_l2332_233211


namespace NUMINAMATH_GPT_area_of_BDOE_l2332_233293

namespace Geometry

noncomputable def areaQuadrilateralBDOE (AE CD AB BC AC : ℝ) : ℝ :=
  if AE = 2 ∧ CD = 11 ∧ AB = 8 ∧ BC = 8 ∧ AC = 6 then
    189 * Real.sqrt 55 / 88
  else
    0

theorem area_of_BDOE :
  areaQuadrilateralBDOE 2 11 8 8 6 = 189 * Real.sqrt 55 / 88 :=
by 
  sorry

end Geometry

end NUMINAMATH_GPT_area_of_BDOE_l2332_233293


namespace NUMINAMATH_GPT_equal_after_operations_l2332_233240

theorem equal_after_operations :
  let initial_first_number := 365
  let initial_second_number := 24
  let first_number_after_n_operations := initial_first_number - 19 * 11
  let second_number_after_n_operations := initial_second_number + 12 * 11
  first_number_after_n_operations = second_number_after_n_operations := sorry

end NUMINAMATH_GPT_equal_after_operations_l2332_233240


namespace NUMINAMATH_GPT_students_interested_in_both_l2332_233291

theorem students_interested_in_both (total_students interested_in_sports interested_in_entertainment not_interested interested_in_both : ℕ)
  (h_total_students : total_students = 1400)
  (h_interested_in_sports : interested_in_sports = 1250)
  (h_interested_in_entertainment : interested_in_entertainment = 952)
  (h_not_interested : not_interested = 60)
  (h_equation : not_interested + interested_in_both + (interested_in_sports - interested_in_both) + (interested_in_entertainment - interested_in_both) = total_students) :
  interested_in_both = 862 :=
by
  sorry

end NUMINAMATH_GPT_students_interested_in_both_l2332_233291


namespace NUMINAMATH_GPT_cube_sum_is_integer_l2332_233250

theorem cube_sum_is_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^3 + 1/a^3 = m :=
sorry

end NUMINAMATH_GPT_cube_sum_is_integer_l2332_233250


namespace NUMINAMATH_GPT_hexagon_side_relation_l2332_233262

noncomputable def hexagon (a b c d e f : ℝ) :=
  ∃ (i j k l m n : ℝ), 
    i = 120 ∧ j = 120 ∧ k = 120 ∧ l = 120 ∧ m = 120 ∧ n = 120 ∧  
    a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f ∧ f = a

theorem hexagon_side_relation
  (a b c d e f : ℝ)
  (ha : hexagon a b c d e f) :
  d - a = b - e ∧ b - e = f - c :=
by
  sorry

end NUMINAMATH_GPT_hexagon_side_relation_l2332_233262


namespace NUMINAMATH_GPT_molecular_weight_of_7_moles_AlPO4_is_correct_l2332_233238

def atomic_weight_Al : Float := 26.98
def atomic_weight_P : Float := 30.97
def atomic_weight_O : Float := 16.00

def molecular_weight_AlPO4 : Float :=
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

noncomputable def weight_of_7_moles_AlPO4 : Float :=
  7 * molecular_weight_AlPO4

theorem molecular_weight_of_7_moles_AlPO4_is_correct :
  weight_of_7_moles_AlPO4 = 853.65 := by
  -- computation goes here
  sorry

end NUMINAMATH_GPT_molecular_weight_of_7_moles_AlPO4_is_correct_l2332_233238


namespace NUMINAMATH_GPT_max_distance_from_point_on_circle_to_line_l2332_233207

noncomputable def center_of_circle : ℝ × ℝ := (5, 3)
noncomputable def radius_of_circle : ℝ := 3
noncomputable def line_eqn (x y : ℝ) : ℝ := 3 * x + 4 * y - 2
noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ := (|a * px + b * py + c|) / (Real.sqrt (a * a + b * b))

theorem max_distance_from_point_on_circle_to_line :
  let Cx := (center_of_circle.1)
  let Cy := (center_of_circle.2)
  let d := distance_point_to_line Cx Cy 3 4 (-2)
  d + radius_of_circle = 8 := by
  sorry

end NUMINAMATH_GPT_max_distance_from_point_on_circle_to_line_l2332_233207


namespace NUMINAMATH_GPT_cat_catches_total_birds_l2332_233272

theorem cat_catches_total_birds :
  let morning_birds := 15
  let morning_success_rate := 0.60
  let afternoon_birds := 25
  let afternoon_success_rate := 0.80
  let night_birds := 20
  let night_success_rate := 0.90
  
  let morning_caught := morning_birds * morning_success_rate
  let afternoon_initial_caught := 2 * morning_caught
  let afternoon_caught := min (afternoon_birds * afternoon_success_rate) afternoon_initial_caught
  let night_caught := night_birds * night_success_rate

  let total_caught := morning_caught + afternoon_caught + night_caught
  total_caught = 47 := 
by
  sorry

end NUMINAMATH_GPT_cat_catches_total_birds_l2332_233272


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2332_233290

-- Define the propositions P and Q
def P (a b : ℝ) : Prop := a^2 + b^2 > 2 * a * b
def Q (a b : ℝ) : Prop := abs (a + b) < abs a + abs b

-- Define the conditions for P and Q
def condition_for_P (a b : ℝ) : Prop := a ≠ b
def condition_for_Q (a b : ℝ) : Prop := a * b < 0

-- Define the statement
theorem necessary_but_not_sufficient (a b : ℝ) :
  (P a b → Q a b) ∧ ¬ (Q a b → P a b) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2332_233290


namespace NUMINAMATH_GPT_fraction_inequality_l2332_233231

open Real

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a < b + c) :
  a / (1 + a) < b / (1 + b) + c / (1 + c) :=
  sorry

end NUMINAMATH_GPT_fraction_inequality_l2332_233231


namespace NUMINAMATH_GPT_shortest_distance_l2332_233289

-- Define the line and the circle
def is_on_line (P : ℝ × ℝ) : Prop := P.snd = P.fst - 1

def is_on_circle (Q : ℝ × ℝ) : Prop := Q.fst^2 + Q.snd^2 + 4 * Q.fst - 2 * Q.snd + 4 = 0

-- Define the square of the Euclidean distance between two points
def dist_squared (P Q : ℝ × ℝ) : ℝ := (P.fst - Q.fst)^2 + (P.snd - Q.snd)^2

-- State the theorem regarding the shortest distance between the points on the line and the circle
theorem shortest_distance : ∃ P Q : ℝ × ℝ, is_on_line P ∧ is_on_circle Q ∧ dist_squared P Q = 1 := sorry

end NUMINAMATH_GPT_shortest_distance_l2332_233289


namespace NUMINAMATH_GPT_captain_age_l2332_233298

theorem captain_age (C : ℕ) (h1 : ∀ W : ℕ, W = C + 3) 
                    (h2 : 21 * 11 = 231) 
                    (h3 : 21 - 1 = 20) 
                    (h4 : 20 * 9 = 180)
                    (h5 : 231 - 180 = 51) :
  C = 24 :=
by
  sorry

end NUMINAMATH_GPT_captain_age_l2332_233298


namespace NUMINAMATH_GPT_line_passes_through_parabola_vertex_l2332_233209

theorem line_passes_through_parabola_vertex :
  ∃ (a : ℝ), (∃ (b : ℝ), b = a ∧ (a = 0 ∨ a = 1)) :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_parabola_vertex_l2332_233209


namespace NUMINAMATH_GPT_hyperbola_eccentricity_range_l2332_233223

theorem hyperbola_eccentricity_range 
(a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
(hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
(parabola_eq : ∀ y x, y^2 = 8 * a * x)
(right_vertex : A = (a, 0))
(focus : F = (2 * a, 0))
(P : ℝ × ℝ)
(asymptote_eq : P = (x0, b / a * x0))
(perpendicular_condition : (x0 ^ 2 - (3 * a - b^2 / a^2) * x0 + 2 * a^2 = 0))
(hyperbola_properties: c^2 = a^2 + b^2) :
1 < c / a ∧ c / a <= 3 * Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_range_l2332_233223


namespace NUMINAMATH_GPT_C_pow_50_l2332_233264

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℝ :=
![![3, 1], ![-4, -1]]

theorem C_pow_50 :
  (C ^ 50) = ![![101, 50], ![-200, -99]] :=
by
  sorry

end NUMINAMATH_GPT_C_pow_50_l2332_233264


namespace NUMINAMATH_GPT_country_x_income_l2332_233297

theorem country_x_income (I : ℝ) (h1 : I > 40000) (_ : 0.15 * 40000 + 0.20 * (I - 40000) = 8000) : I = 50000 :=
sorry

end NUMINAMATH_GPT_country_x_income_l2332_233297


namespace NUMINAMATH_GPT_tristan_study_hours_l2332_233243

theorem tristan_study_hours :
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  saturday_hours = 2 := by
{
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  sorry
}

end NUMINAMATH_GPT_tristan_study_hours_l2332_233243


namespace NUMINAMATH_GPT_probability_at_least_one_black_ball_l2332_233217

theorem probability_at_least_one_black_ball :
  let total_balls := 6
  let red_balls := 2
  let white_ball := 1
  let black_balls := 3
  let total_combinations := Nat.choose total_balls 2
  let non_black_combinations := Nat.choose (total_balls - black_balls) 2
  let probability := 1 - (non_black_combinations / total_combinations : ℚ)
  probability = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_black_ball_l2332_233217


namespace NUMINAMATH_GPT_percentage_increase_in_spending_l2332_233296

variables (P Q : ℝ)
-- Conditions
def price_increase (P : ℝ) := 1.25 * P
def quantity_decrease (Q : ℝ) := 0.88 * Q

-- Mathemtically equivalent proof problem in Lean:
theorem percentage_increase_in_spending (P Q : ℝ) : 
  (price_increase P) * (quantity_decrease Q) / (P * Q) = 1.10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_spending_l2332_233296


namespace NUMINAMATH_GPT_product_of_two_numbers_l2332_233234

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2332_233234


namespace NUMINAMATH_GPT_polynomial_third_and_fourth_equal_l2332_233227

theorem polynomial_third_and_fourth_equal (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p + q = 1)
  (h_eq : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = (8 : ℝ) / 11 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_third_and_fourth_equal_l2332_233227


namespace NUMINAMATH_GPT_students_in_class_l2332_233292

theorem students_in_class (n : ℕ) (h1 : (n : ℝ) * 100 = (n * 100 + 60 - 10)) 
  (h2 : (n : ℝ) * 98 = ((n : ℝ) * 100 - 50)) : n = 25 :=
sorry

end NUMINAMATH_GPT_students_in_class_l2332_233292


namespace NUMINAMATH_GPT_impossible_distance_l2332_233287

noncomputable def radius_O1 : ℝ := 2
noncomputable def radius_O2 : ℝ := 5

theorem impossible_distance :
  ∀ (d : ℝ), ¬ (radius_O1 ≠ radius_O2 → ¬ (d < abs (radius_O2 - radius_O1) ∨ d > radius_O2 + radius_O1) → d = 5) :=
by
  sorry

end NUMINAMATH_GPT_impossible_distance_l2332_233287


namespace NUMINAMATH_GPT_minimum_ratio_cone_cylinder_l2332_233246

theorem minimum_ratio_cone_cylinder (r : ℝ) (h : ℝ) (a : ℝ) :
  (h = 4 * r) →
  (a^2 = r^2 * h^2 / (h - 2 * r)) →
  (∀ h > 0, ∃ V_cone V_cylinder, 
    V_cone = (1/3) * π * a^2 * h ∧ 
    V_cylinder = π * r^2 * (2 * r) ∧ 
    V_cone / V_cylinder = (4 / 3)) := 
sorry

end NUMINAMATH_GPT_minimum_ratio_cone_cylinder_l2332_233246


namespace NUMINAMATH_GPT_inequality_sqrt_l2332_233278

theorem inequality_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_sqrt_l2332_233278


namespace NUMINAMATH_GPT_find_d_l2332_233267

-- Define the six-digit number as a function of d
def six_digit_num (d : ℕ) : ℕ := 3 * 100000 + 2 * 10000 + 5 * 1000 + 4 * 100 + 7 * 10 + d

-- Define the sum of digits of the six-digit number
def sum_of_digits (d : ℕ) : ℕ := 3 + 2 + 5 + 4 + 7 + d

-- The statement we want to prove
theorem find_d (d : ℕ) : sum_of_digits d % 3 = 0 ↔ d = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l2332_233267


namespace NUMINAMATH_GPT_complex_root_sixth_power_sum_equals_38908_l2332_233208

noncomputable def omega : ℂ :=
  -- By definition, omega should satisfy the below properties.
  -- The exact value of omega is not being defined, we will use algebraic properties in the proof.
  sorry

theorem complex_root_sixth_power_sum_equals_38908 : 
  ∀ (ω : ℂ), ω^3 = 1 ∧ ¬(ω.re = 1) → (2 - ω + 2 * ω^2)^6 + (2 + ω - 2 * ω^2)^6 = 38908 :=
by
  -- Proof will utilize given conditions:
  -- 1. ω^3 = 1
  -- 2. ω is not real (or ω.re is not 1)
  sorry

end NUMINAMATH_GPT_complex_root_sixth_power_sum_equals_38908_l2332_233208


namespace NUMINAMATH_GPT_sum_n_k_eq_eight_l2332_233224

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the theorem to prove that n + k = 8 given the conditions
theorem sum_n_k_eq_eight {n k : ℕ} 
  (h1 : binom n k * 3 = binom n (k + 1))
  (h2 : binom n (k + 1) * 5 = binom n (k + 2) * 3) : n + k = 8 := by
  sorry

end NUMINAMATH_GPT_sum_n_k_eq_eight_l2332_233224


namespace NUMINAMATH_GPT_probability_one_boy_one_girl_l2332_233239

-- Define the total number of students (5), the number of boys (3), and the number of girls (2).
def total_students : Nat := 5
def boys : Nat := 3
def girls : Nat := 2

-- Define the probability calculation in Lean.
noncomputable def select_2_students_prob : ℚ :=
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose boys 1 * Nat.choose girls 1
  favorable_combinations / total_combinations

-- The statement we need to prove is that this probability is 3/5
theorem probability_one_boy_one_girl : select_2_students_prob = 3 / 5 := sorry

end NUMINAMATH_GPT_probability_one_boy_one_girl_l2332_233239


namespace NUMINAMATH_GPT_symmetric_points_origin_l2332_233201

theorem symmetric_points_origin (a b : ℝ) (h : (1, 2) = (-a, -b)) : a = -1 ∧ b = -2 :=
sorry

end NUMINAMATH_GPT_symmetric_points_origin_l2332_233201


namespace NUMINAMATH_GPT_ab_max_min_sum_l2332_233284

-- Define the conditions
variables {a b : ℝ}
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : a + 4 * b = 4

-- Problem (1)
theorem ab_max : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → a * b ≤ 1 :=
by sorry

-- Problem (2)
theorem min_sum : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → (1 / a) + (4 / b) ≥ 25 / 4 :=
by sorry

end NUMINAMATH_GPT_ab_max_min_sum_l2332_233284


namespace NUMINAMATH_GPT_S_12_l2332_233257

variable {S : ℕ → ℕ}

-- Given conditions
axiom S_4 : S 4 = 4
axiom S_8 : S 8 = 12

-- Goal: Prove S_12
theorem S_12 : S 12 = 24 :=
by
  sorry

end NUMINAMATH_GPT_S_12_l2332_233257


namespace NUMINAMATH_GPT_exists_b_c_with_integral_roots_l2332_233202

theorem exists_b_c_with_integral_roots :
  ∃ (b c : ℝ), (∃ (p q : ℤ), (x^2 + b * x + c = 0) ∧ (x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
               ((x - p) * (x - q) = x^2 - (p + q) * x + p*q)) ∧
              (∃ (r s : ℤ), (x^2 + (b+1) * x + (c+1) = 0) ∧ 
              ((x - r) * (x - s) = x^2 - (r + s) * x + r*s)) :=
by
  sorry

end NUMINAMATH_GPT_exists_b_c_with_integral_roots_l2332_233202


namespace NUMINAMATH_GPT_intersection_points_of_parabolas_l2332_233249

open Real

theorem intersection_points_of_parabolas (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ y1 y2 : ℝ, y1 = c ∧ y2 = (-2 * b^2 / (9 * a)) + c ∧ 
    ((y1 = a * (0)^2 + b * (0) + c) ∧ (y2 = a * (-b / (3 * a))^2 + b * (-b / (3 * a)) + c))) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_of_parabolas_l2332_233249


namespace NUMINAMATH_GPT_least_two_multiples_of_15_gt_450_l2332_233230

-- Define a constant for the base multiple
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

-- Define a constant for being greater than 450
def is_greater_than_450 (n : ℕ) : Prop :=
  n > 450

-- Two least positive multiples of 15 greater than 450
theorem least_two_multiples_of_15_gt_450 :
  (is_multiple_of_15 465 ∧ is_greater_than_450 465 ∧
   is_multiple_of_15 480 ∧ is_greater_than_450 480) :=
by
  sorry

end NUMINAMATH_GPT_least_two_multiples_of_15_gt_450_l2332_233230


namespace NUMINAMATH_GPT_necessarily_negative_l2332_233270

theorem necessarily_negative (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : -2 < b ∧ b < 0) (h3 : 0 < c ∧ c < 1) : b + c < 0 :=
sorry

end NUMINAMATH_GPT_necessarily_negative_l2332_233270


namespace NUMINAMATH_GPT_perfect_square_expression_5_l2332_233261

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def expression_1 : ℕ := 3^3 * 4^4 * 7^7
def expression_2 : ℕ := 3^4 * 4^3 * 7^6
def expression_3 : ℕ := 3^5 * 4^6 * 7^5
def expression_4 : ℕ := 3^6 * 4^5 * 7^4
def expression_5 : ℕ := 3^4 * 4^6 * 7^4

theorem perfect_square_expression_5 : is_perfect_square expression_5 :=
sorry

end NUMINAMATH_GPT_perfect_square_expression_5_l2332_233261


namespace NUMINAMATH_GPT_bacon_sold_l2332_233203

variable (B : ℕ) -- Declare the variable for the number of slices of bacon sold

-- Define the given conditions as Lean definitions
def pancake_price := 4
def bacon_price := 2
def stacks_sold := 60
def total_raised := 420

-- The revenue from pancake sales alone
def pancake_revenue := stacks_sold * pancake_price
-- The revenue from bacon sales
def bacon_revenue := total_raised - pancake_revenue

-- Statement of the theorem
theorem bacon_sold :
  B = bacon_revenue / bacon_price :=
sorry

end NUMINAMATH_GPT_bacon_sold_l2332_233203


namespace NUMINAMATH_GPT_instrument_accuracy_confidence_l2332_233263

noncomputable def instrument_accuracy (n : ℕ) (s : ℝ) (gamma : ℝ) (q : ℝ) : ℝ × ℝ :=
  let lower := s * (1 - q)
  let upper := s * (1 + q)
  (lower, upper)

theorem instrument_accuracy_confidence :
  ∀ (n : ℕ) (s : ℝ) (gamma : ℝ) (q : ℝ),
    n = 12 →
    s = 0.6 →
    gamma = 0.99 →
    q = 0.9 →
    0.06 < (instrument_accuracy n s gamma q).fst ∧
    (instrument_accuracy n s gamma q).snd < 1.14 :=
by
  intros n s gamma q h_n h_s h_gamma h_q
  -- proof would go here
  sorry

end NUMINAMATH_GPT_instrument_accuracy_confidence_l2332_233263


namespace NUMINAMATH_GPT_pipes_fill_cistern_together_in_15_minutes_l2332_233233

-- Define the problem's conditions in Lean
def PipeA_rate := (1 / 2) / 15
def PipeB_rate := (1 / 3) / 10

-- Define the combined rate
def combined_rate := PipeA_rate + PipeB_rate

-- Define the time to fill the cistern by both pipes working together
def time_to_fill_cistern := 1 / combined_rate

-- State the theorem to prove
theorem pipes_fill_cistern_together_in_15_minutes :
  time_to_fill_cistern = 15 := by
  sorry

end NUMINAMATH_GPT_pipes_fill_cistern_together_in_15_minutes_l2332_233233


namespace NUMINAMATH_GPT_prove_total_weekly_allowance_l2332_233204

noncomputable def total_weekly_allowance : ℕ :=
  let students := 200
  let group1 := students * 45 / 100
  let group2 := students * 30 / 100
  let group3 := students * 15 / 100
  let group4 := students - group1 - group2 - group3  -- Remaining students
  let daily_allowance := group1 * 6 + group2 * 4 + group3 * 7 + group4 * 10
  daily_allowance * 7

theorem prove_total_weekly_allowance :
  total_weekly_allowance = 8330 := by
  sorry

end NUMINAMATH_GPT_prove_total_weekly_allowance_l2332_233204


namespace NUMINAMATH_GPT_brownies_on_counter_l2332_233258

-- Define the initial number of dozen brownies
def initial_dozens : ℕ := 2

-- Define the conversion from dozens to brownies
def dozen_to_brownies (d : ℕ) : ℕ := d * 12

-- Define the initial number of brownies
def initial_brownies : ℕ := dozen_to_brownies initial_dozens

-- Define the number of brownies father ate
def father_ate : ℕ := 8

-- Define the number of brownies Mooney ate
def mooney_ate : ℕ := 4

-- Define the number of dozen brownies made the next morning
def next_morning_dozens : ℕ := 2

-- Define the number of brownies made the next morning
def next_morning_brownies : ℕ := dozen_to_brownies next_morning_dozens

-- Calculate the remaining brownies after father and Mooney ate some
def remaining_brownies : ℕ := initial_brownies - father_ate - mooney_ate

-- Calculate the total number of brownies after adding the new ones the next morning
def total_brownies : ℕ := remaining_brownies + next_morning_brownies

theorem brownies_on_counter : total_brownies = 36 := by
  sorry

end NUMINAMATH_GPT_brownies_on_counter_l2332_233258


namespace NUMINAMATH_GPT_van_capacity_l2332_233245

theorem van_capacity (s a v : ℕ) (h1 : s = 2) (h2 : a = 6) (h3 : v = 2) : (s + a) / v = 4 := by
  sorry

end NUMINAMATH_GPT_van_capacity_l2332_233245


namespace NUMINAMATH_GPT_point_in_quadrant_l2332_233212

theorem point_in_quadrant (m n : ℝ) (h₁ : 2 * (m - 1)^2 - 7 = -5) (h₂ : n > 3) :
  (m = 0 → 2*m - 3 < 0 ∧ (3*n - m)/2 > 0) ∧ 
  (m = 2 → 2*m - 3 > 0 ∧ (3*n - m)/2 > 0) :=
by 
  sorry

end NUMINAMATH_GPT_point_in_quadrant_l2332_233212


namespace NUMINAMATH_GPT_ratio_of_triangle_and_hexagon_l2332_233210

variable {n m : ℝ}

-- Conditions:
def is_regular_hexagon (ABCDEF : Type) : Prop := sorry
def area_of_hexagon (ABCDEF : Type) (n : ℝ) : Prop := sorry
def area_of_triangle_ACE (ABCDEF : Type) (m : ℝ) : Prop := sorry
  
theorem ratio_of_triangle_and_hexagon
  (ABCDEF : Type)
  (H1 : is_regular_hexagon ABCDEF)
  (H2 : area_of_hexagon ABCDEF n)
  (H3 : area_of_triangle_ACE ABCDEF m) :
  m / n = 2 / 3 := 
  sorry

end NUMINAMATH_GPT_ratio_of_triangle_and_hexagon_l2332_233210


namespace NUMINAMATH_GPT_average_number_of_fish_is_75_l2332_233232

-- Define the conditions
def BoastPool_fish := 75
def OnumLake_fish := BoastPool_fish + 25
def RiddlePond_fish := OnumLake_fish / 2

-- Prove the average number of fish
theorem average_number_of_fish_is_75 :
  (BoastPool_fish + OnumLake_fish + RiddlePond_fish) / 3 = 75 :=
by
  sorry

end NUMINAMATH_GPT_average_number_of_fish_is_75_l2332_233232


namespace NUMINAMATH_GPT_find_a_l2332_233299

theorem find_a (a : ℝ) : (∃ b : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) -> a = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2332_233299
