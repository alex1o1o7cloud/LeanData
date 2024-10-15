import Mathlib

namespace NUMINAMATH_GPT_loss_per_metre_is_5_l1852_185268

-- Definitions
def selling_price (total_meters : ℕ) : ℕ := 18000
def cost_price_per_metre : ℕ := 65
def total_meters : ℕ := 300

-- Loss per meter calculation
def loss_per_metre (selling_price : ℕ) (cost_price_per_metre : ℕ) (total_meters : ℕ) : ℕ :=
  ((cost_price_per_metre * total_meters) - selling_price) / total_meters

-- Theorem statement
theorem loss_per_metre_is_5 : loss_per_metre (selling_price total_meters) cost_price_per_metre total_meters = 5 :=
by
  sorry

end NUMINAMATH_GPT_loss_per_metre_is_5_l1852_185268


namespace NUMINAMATH_GPT_hash_hash_hash_100_l1852_185288

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_100 : hash (hash (hash 100)) = 11.08 :=
by sorry

end NUMINAMATH_GPT_hash_hash_hash_100_l1852_185288


namespace NUMINAMATH_GPT_shaded_area_l1852_185275

theorem shaded_area (side_len : ℕ) (triangle_base : ℕ) (triangle_height : ℕ)
  (h1 : side_len = 40) (h2 : triangle_base = side_len / 2)
  (h3 : triangle_height = side_len / 2) : 
  side_len^2 - 2 * (1/2 * triangle_base * triangle_height) = 1200 := 
  sorry

end NUMINAMATH_GPT_shaded_area_l1852_185275


namespace NUMINAMATH_GPT_scientific_notation_113700_l1852_185291

theorem scientific_notation_113700 : (113700 : ℝ) = 1.137 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_113700_l1852_185291


namespace NUMINAMATH_GPT_find_divisor_l1852_185265

theorem find_divisor (x : ℝ) (h : x / n = 0.01 * (x * n)) : n = 10 :=
sorry

end NUMINAMATH_GPT_find_divisor_l1852_185265


namespace NUMINAMATH_GPT_expand_and_solve_solve_quadratic_l1852_185249

theorem expand_and_solve (x : ℝ) :
  6 * (x - 3) * (x + 5) = 6 * x^2 + 12 * x - 90 :=
by sorry

theorem solve_quadratic (x : ℝ) :
  6 * x^2 + 12 * x - 90 = 0 ↔ x = -5 ∨ x = 3 :=
by sorry

end NUMINAMATH_GPT_expand_and_solve_solve_quadratic_l1852_185249


namespace NUMINAMATH_GPT_value_multiplied_by_l1852_185272

theorem value_multiplied_by (x : ℝ) (h : (7.5 / 6) * x = 15) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_multiplied_by_l1852_185272


namespace NUMINAMATH_GPT_terminal_side_second_quadrant_l1852_185211

theorem terminal_side_second_quadrant (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end NUMINAMATH_GPT_terminal_side_second_quadrant_l1852_185211


namespace NUMINAMATH_GPT_fruit_prices_l1852_185218

theorem fruit_prices :
  (∃ x y : ℝ, 60 * x + 40 * y = 1520 ∧ 30 * x + 50 * y = 1360 ∧ x = 12 ∧ y = 20) :=
sorry

end NUMINAMATH_GPT_fruit_prices_l1852_185218


namespace NUMINAMATH_GPT_t_shaped_region_slope_divides_area_in_half_l1852_185277

theorem t_shaped_region_slope_divides_area_in_half :
  ∃ (m : ℚ), (m = 4 / 11) ∧ (
    let area1 := 2 * (m * 2 * 4)
    let area2 := ((4 - m * 2) * 4) + 6
    area1 = area2
  ) :=
by
  sorry

end NUMINAMATH_GPT_t_shaped_region_slope_divides_area_in_half_l1852_185277


namespace NUMINAMATH_GPT_function_identity_l1852_185287

theorem function_identity (f : ℕ → ℕ) (h₁ : ∀ n, 0 < f n)
  (h₂ : ∀ n, f (n + 1) > f (f n)) :
∀ n, f n = n :=
sorry

end NUMINAMATH_GPT_function_identity_l1852_185287


namespace NUMINAMATH_GPT_hyperbola_smaller_focus_l1852_185228

noncomputable def smaller_focus_coordinates : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 3
  let b := 7
  let c := Real.sqrt (a^2 + b^2)
  (h - c, k)

theorem hyperbola_smaller_focus :
  (smaller_focus_coordinates = (Real.sqrt 58 - 2.62, 20)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_smaller_focus_l1852_185228


namespace NUMINAMATH_GPT_tetrahedron_volume_lower_bound_l1852_185266

noncomputable def volume_tetrahedron (d1 d2 d3 : ℝ) : ℝ := sorry

theorem tetrahedron_volume_lower_bound {d1 d2 d3 : ℝ} (h1 : d1 > 0) (h2 : d2 > 0) (h3 : d3 > 0) :
  volume_tetrahedron d1 d2 d3 ≥ (1 / 3) * d1 * d2 * d3 :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_lower_bound_l1852_185266


namespace NUMINAMATH_GPT_anns_age_l1852_185270

theorem anns_age (a b : ℕ)
  (h1 : a + b = 72)
  (h2 : ∃ y, y = a - b)
  (h3 : b = a / 3 + 2 * (a - b)) : a = 36 :=
by
  sorry

end NUMINAMATH_GPT_anns_age_l1852_185270


namespace NUMINAMATH_GPT_range_of_m_l1852_185263

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (hf : ∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ y, f y = x) :
  (∀ x, ∃ y, y = f (x + m) - f (x - m)) →
  -1 ≤ m ∧ m ≤ 1 :=
by
  intro hF
  sorry

end NUMINAMATH_GPT_range_of_m_l1852_185263


namespace NUMINAMATH_GPT_sum_of_solutions_l1852_185276

theorem sum_of_solutions (x1 x2 : ℝ) (h : ∀ (x : ℝ), x^2 - 10 * x + 14 = 0 → x = x1 ∨ x = x2) :
  x1 + x2 = 10 :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_l1852_185276


namespace NUMINAMATH_GPT_students_speaking_both_languages_l1852_185207

theorem students_speaking_both_languages:
  ∀ (total E T N B : ℕ),
    total = 150 →
    E = 55 →
    T = 85 →
    N = 30 →
    (total - N) = 120 →
    (E + T - B) = 120 → B = 20 :=
by
  intros total E T N B h_total h_E h_T h_N h_langs h_equiv
  sorry

end NUMINAMATH_GPT_students_speaking_both_languages_l1852_185207


namespace NUMINAMATH_GPT_tan_double_beta_alpha_value_l1852_185210

open Real

-- Conditions
def alpha_in_interval (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def beta_in_interval (β : ℝ) : Prop := π / 2 < β ∧ β < π
def cos_beta (β : ℝ) : Prop := cos β = -1 / 3
def sin_alpha_plus_beta (α β : ℝ) : Prop := sin (α + β) = (4 - sqrt 2) / 6

-- Proof problem 1: Prove that tan 2β = 4√2 / 7 given the conditions
theorem tan_double_beta (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  tan (2 * β) = (4 * sqrt 2) / 7 :=
by sorry

-- Proof problem 2: Prove that α = π / 4 given the conditions
theorem alpha_value (α β : ℝ) (h1 : alpha_in_interval α) (h2 : beta_in_interval β)
  (h3 : cos_beta β) (h4 : sin_alpha_plus_beta α β) :
  α = π / 4 :=
by sorry

end NUMINAMATH_GPT_tan_double_beta_alpha_value_l1852_185210


namespace NUMINAMATH_GPT_bees_count_on_fifth_day_l1852_185278

theorem bees_count_on_fifth_day
  (initial_count : ℕ) (h_initial : initial_count = 1)
  (growth_factor : ℕ) (h_growth : growth_factor = 3) :
  let bees_at_day (n : ℕ) : ℕ := initial_count * (growth_factor + 1) ^ n
  bees_at_day 5 = 1024 := 
by {
  sorry
}

end NUMINAMATH_GPT_bees_count_on_fifth_day_l1852_185278


namespace NUMINAMATH_GPT_complete_square_l1852_185257

-- Definitions based on conditions
def row_sum_piece2 := 2 + 1 + 3 + 1
def total_sum_square := 4 * row_sum_piece2
def sum_piece1 := 7
def sum_piece2 := 8
def sum_piece3 := 8
def total_given_pieces := sum_piece1 + sum_piece2 + sum_piece3
def sum_missing_piece := total_sum_square - total_given_pieces

-- Statement to prove that the missing piece has the correct sum
theorem complete_square : (sum_missing_piece = 5) :=
by 
  -- It is a placeholder for the proof steps, the actual proof steps are not needed
  sorry

end NUMINAMATH_GPT_complete_square_l1852_185257


namespace NUMINAMATH_GPT_observer_height_proof_l1852_185239

noncomputable def height_observer (d m α β : ℝ) : ℝ :=
  let cot_alpha := 1 / Real.tan α
  let cot_beta := 1 / Real.tan β
  let u := (d * (m * cot_beta - d)) / (2 * d - m * (cot_beta - cot_alpha))
  20 + Real.sqrt (400 + u * m * cot_alpha - u^2)

theorem observer_height_proof :
  height_observer 290 40 (11.4 * Real.pi / 180) (4.7 * Real.pi / 180) = 52 := sorry

end NUMINAMATH_GPT_observer_height_proof_l1852_185239


namespace NUMINAMATH_GPT_area_of_trapezium_l1852_185205

-- Definitions
def length_parallel_side_1 : ℝ := 4
def length_parallel_side_2 : ℝ := 5
def perpendicular_distance : ℝ := 6

-- Statement
theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side_1 + length_parallel_side_2) * perpendicular_distance = 27 :=
by
  sorry

end NUMINAMATH_GPT_area_of_trapezium_l1852_185205


namespace NUMINAMATH_GPT_arithmetic_mean_l1852_185284

variable {x b c : ℝ}

theorem arithmetic_mean (hx : x ≠ 0) (hb : b ≠ c) : 
  (1 / 2) * ((x + b) / x + (x - c) / x) = 1 + (b - c) / (2 * x) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_l1852_185284


namespace NUMINAMATH_GPT_odd_indexed_terms_geometric_sequence_l1852_185232

open Nat

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (2 * n + 3) = r * a (2 * n + 1)

theorem odd_indexed_terms_geometric_sequence (b : ℕ → ℝ) (h : ∀ n, b n * b (n + 1) = 3 ^ n) :
  is_geometric_sequence b 3 :=
by
  sorry

end NUMINAMATH_GPT_odd_indexed_terms_geometric_sequence_l1852_185232


namespace NUMINAMATH_GPT_regular_price_of_pony_jeans_l1852_185203

-- Define the regular price of fox jeans
def fox_jeans_price := 15

-- Define the given conditions
def pony_discount_rate := 0.18
def total_savings := 9
def total_discount_rate := 0.22

-- State the problem: Prove the regular price of pony jeans
theorem regular_price_of_pony_jeans : 
  ∃ P, P * pony_discount_rate = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_regular_price_of_pony_jeans_l1852_185203


namespace NUMINAMATH_GPT_annual_interest_rate_is_correct_l1852_185271

-- Definitions of the conditions
def true_discount : ℚ := 210
def bill_amount : ℚ := 1960
def time_period_years : ℚ := 3 / 4

-- The present value of the bill
def present_value : ℚ := bill_amount - true_discount

-- The formula for simple interest given principal, rate, and time
def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T / 100

-- Proof statement
theorem annual_interest_rate_is_correct : 
  ∃ (R : ℚ), simple_interest present_value R time_period_years = true_discount ∧ R = 16 :=
by
  use 16
  sorry

end NUMINAMATH_GPT_annual_interest_rate_is_correct_l1852_185271


namespace NUMINAMATH_GPT_diff_of_squares_odd_divisible_by_8_l1852_185242

theorem diff_of_squares_odd_divisible_by_8 (m n : ℤ) :
  ((2 * m + 1) ^ 2 - (2 * n + 1) ^ 2) % 8 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_diff_of_squares_odd_divisible_by_8_l1852_185242


namespace NUMINAMATH_GPT_quotient_of_m_and_n_l1852_185295

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem quotient_of_m_and_n (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : f m = f n) (h4 : ∀ x ∈ Set.Icc (m^2) n, f x ≤ 2) :
  n / m = Real.exp 2 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_m_and_n_l1852_185295


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1852_185244

noncomputable def simplified_expr (x y : ℝ) : ℝ :=
  ((-2 * x + y)^2 - (2 * x - y) * (y + 2 * x) - 6 * y) / (2 * y)

theorem simplify_and_evaluate :
  let x := -1
  let y := 2
  simplified_expr x y = 1 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1852_185244


namespace NUMINAMATH_GPT_total_population_l1852_185292

-- Define the predicates for g, b, and s based on t
variables (g b t s : ℕ)

-- The conditions given in the problem
def condition1 : Prop := g = 4 * t
def condition2 : Prop := b = 6 * g
def condition3 : Prop := s = t / 2

-- The theorem stating the total population is equal to (59 * t) / 2
theorem total_population (g b t s : ℕ) (h1 : condition1 g t) (h2 : condition2 b g) (h3 : condition3 s t) :
  b + g + t + s = 59 * t / 2 :=
by sorry

end NUMINAMATH_GPT_total_population_l1852_185292


namespace NUMINAMATH_GPT_find_integer_l1852_185253

theorem find_integer
  (x y : ℤ)
  (h1 : 4 * x + y = 34)
  (h2 : 2 * x - y = 20)
  (h3 : y^2 = 4) :
  y = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_l1852_185253


namespace NUMINAMATH_GPT_infinite_series_problem_l1852_185262

noncomputable def infinite_series_sum : ℝ := ∑' n : ℕ, (2 * (n + 1)^2 - 3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))

theorem infinite_series_problem :
  infinite_series_sum = -4 :=
by sorry

end NUMINAMATH_GPT_infinite_series_problem_l1852_185262


namespace NUMINAMATH_GPT_expansion_correct_l1852_185212

noncomputable def P (x y : ℝ) : ℝ := 2 * x^25 - 5 * x^8 + 2 * x * y^3 - 9

noncomputable def M (x : ℝ) : ℝ := 3 * x^7

theorem expansion_correct (x y : ℝ) :
  (P x y) * (M x) = 6 * x^32 - 15 * x^15 + 6 * x^8 * y^3 - 27 * x^7 :=
by
  sorry

end NUMINAMATH_GPT_expansion_correct_l1852_185212


namespace NUMINAMATH_GPT_find_speeds_l1852_185289

theorem find_speeds 
  (x v u : ℝ)
  (hx : x = u / 4)
  (hv : 0 < v)
  (hu : 0 < u)
  (t_car : 30 / v + 1.25 = 30 / x)
  (meeting_cars : 0.05 * v + 0.05 * u = 5) :
  x = 15 ∧ v = 40 ∧ u = 60 :=
by 
  sorry

end NUMINAMATH_GPT_find_speeds_l1852_185289


namespace NUMINAMATH_GPT_original_bales_l1852_185282

/-
There were some bales of hay in the barn. Jason stacked 23 bales in the barn today.
There are now 96 bales of hay in the barn. Prove that the original number of bales of hay 
in the barn was 73.
-/

theorem original_bales (stacked : ℕ) (total : ℕ) (original : ℕ) 
  (h1 : stacked = 23) (h2 : total = 96) : original = 73 :=
by
  sorry

end NUMINAMATH_GPT_original_bales_l1852_185282


namespace NUMINAMATH_GPT_pipe_A_fill_time_l1852_185231

theorem pipe_A_fill_time (x : ℝ) (h1 : ∀ t : ℝ, t = 45) (h2 : ∀ t : ℝ, t = 18) :
  (1/x + 1/45 = 1/18) → x = 30 :=
by {
  -- Proof is omitted
  sorry
}

end NUMINAMATH_GPT_pipe_A_fill_time_l1852_185231


namespace NUMINAMATH_GPT_find_a_l1852_185226

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

theorem find_a (a : ℝ) : f_prime a 1 = 2 → a = -3 := by
  intros h
  -- skipping the proof, as it is not required
  sorry

end NUMINAMATH_GPT_find_a_l1852_185226


namespace NUMINAMATH_GPT_average_of_last_three_numbers_l1852_185214

theorem average_of_last_three_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 60) 
  (h2 : (a + b + c) / 3 = 55) : 
  (d + e + f) / 3 = 65 :=
sorry

end NUMINAMATH_GPT_average_of_last_three_numbers_l1852_185214


namespace NUMINAMATH_GPT_arithmetic_seq_proof_l1852_185281

theorem arithmetic_seq_proof
  (x : ℕ → ℝ)
  (h : ∀ n ≥ 3, x (n-1) = (x n + x (n-1) + x (n-2)) / 3):
  (x 300 - x 33) / (x 333 - x 3) = 89 / 110 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_proof_l1852_185281


namespace NUMINAMATH_GPT_count_original_scissors_l1852_185273

def originalScissors (addedScissors totalScissors : ℕ) : ℕ := totalScissors - addedScissors

theorem count_original_scissors :
  ∃ (originalScissorsCount : ℕ), originalScissorsCount = originalScissors 13 52 := 
  sorry

end NUMINAMATH_GPT_count_original_scissors_l1852_185273


namespace NUMINAMATH_GPT_ordered_pairs_l1852_185297

theorem ordered_pairs (a b : ℝ) (hapos : 0 < a) (hbpos : 0 < b) (x : ℕ → ℝ)
  (h : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a * x (n + 1) - b * x n| < ε) :
  (a = 0 ∧ 0 < b) ∨ (0 < a ∧ |b / a| < 1) :=
sorry

end NUMINAMATH_GPT_ordered_pairs_l1852_185297


namespace NUMINAMATH_GPT_complement_intersection_l1852_185250

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 3}

theorem complement_intersection : (U \ N) ∩ M = {4, 5} :=
by 
  sorry

end NUMINAMATH_GPT_complement_intersection_l1852_185250


namespace NUMINAMATH_GPT_marco_score_percentage_less_l1852_185267

theorem marco_score_percentage_less
  (average_score : ℕ)
  (margaret_score : ℕ)
  (margaret_more_than_marco : ℕ)
  (h1 : average_score = 90)
  (h2 : margaret_score = 86)
  (h3 : margaret_more_than_marco = 5) :
  (average_score - (margaret_score - margaret_more_than_marco)) * 100 / average_score = 10 :=
by
  sorry

end NUMINAMATH_GPT_marco_score_percentage_less_l1852_185267


namespace NUMINAMATH_GPT_pure_milk_in_final_solution_l1852_185229

noncomputable def final_quantity_of_milk (initial_milk : ℕ) (milk_removed_each_step : ℕ) (steps : ℕ) : ℝ :=
  let remaining_milk_step1 := initial_milk - milk_removed_each_step
  let proportion := (milk_removed_each_step : ℝ) / (initial_milk : ℝ)
  let milk_removed_step2 := proportion * remaining_milk_step1
  remaining_milk_step1 - milk_removed_step2

theorem pure_milk_in_final_solution :
  final_quantity_of_milk 30 9 2 = 14.7 :=
by
  sorry

end NUMINAMATH_GPT_pure_milk_in_final_solution_l1852_185229


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l1852_185204

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, x^2 / 9 + y^2 / 5 = 1 → (x = 2 ∧ y = 0) ∨ (x = -2 ∧ y = 0)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l1852_185204


namespace NUMINAMATH_GPT_hike_up_days_l1852_185279

theorem hike_up_days (R_up R_down D_down D_up : ℝ) 
  (H1 : R_up = 8) 
  (H2 : R_down = 1.5 * R_up)
  (H3 : D_down = 24)
  (H4 : D_up / R_up = D_down / R_down) : 
  D_up / R_up = 2 :=
by
  sorry

end NUMINAMATH_GPT_hike_up_days_l1852_185279


namespace NUMINAMATH_GPT_p_and_q_necessary_not_sufficient_l1852_185283

variable (a m x : ℝ) (P Q : Prop)

def p (a m : ℝ) : Prop := a < 0 ∧ m^2 - 4 * a * m + 3 * a^2 < 0

def q (m : ℝ) : Prop := ∀ x > 0, x + 4 / x ≥ 1 - m

theorem p_and_q_necessary_not_sufficient :
  (∀ (a m : ℝ), p a m → q m) ∧ (∀ a : ℝ, -1 ≤ a ∧ a < 0) :=
sorry

end NUMINAMATH_GPT_p_and_q_necessary_not_sufficient_l1852_185283


namespace NUMINAMATH_GPT_complement_union_l1852_185261

open Set

def U : Set ℕ := {x | x < 6}

def A : Set ℕ := {1, 3}

def B : Set ℕ := {3, 5}

theorem complement_union :
  (U \ (A ∪ B)) = {0, 2, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l1852_185261


namespace NUMINAMATH_GPT_ratio_of_geometric_sequence_sum_l1852_185234

theorem ratio_of_geometric_sequence_sum (a : ℕ → ℕ) 
    (q : ℕ) (h_q_pos : 0 < q) (h_q_ne_one : q ≠ 1)
    (h_geo_seq : ∀ n : ℕ, a (n + 1) = a n * q)
    (h_arith_seq : 2 * a (3 + 2) = a 3 - a (3 + 1)) :
  (a 4 * (1 - q ^ 4) / (1 - q)) / (a 4 * (1 - q ^ 2) / (1 - q)) = 5 / 4 := 
  sorry

end NUMINAMATH_GPT_ratio_of_geometric_sequence_sum_l1852_185234


namespace NUMINAMATH_GPT_granger_bought_12_cans_of_spam_l1852_185259

theorem granger_bought_12_cans_of_spam : 
  ∀ (S : ℕ), 
    (3 * 5 + 4 * 2 + 3 * S = 59) → 
    (S = 12) := 
by
  intro S h
  sorry

end NUMINAMATH_GPT_granger_bought_12_cans_of_spam_l1852_185259


namespace NUMINAMATH_GPT_suff_and_nec_eq_triangle_l1852_185213

noncomputable def triangle (A B C: ℝ) (a b c : ℝ) : Prop :=
(B + C = 2 * A) ∧ (b + c = 2 * a)

theorem suff_and_nec_eq_triangle (A B C a b c : ℝ) (h : triangle A B C a b c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_suff_and_nec_eq_triangle_l1852_185213


namespace NUMINAMATH_GPT_gcf_of_lcm_9_15_and_10_21_is_5_l1852_185235

theorem gcf_of_lcm_9_15_and_10_21_is_5
  (h9 : 9 = 3 ^ 2)
  (h15 : 15 = 3 * 5)
  (h10 : 10 = 2 * 5)
  (h21 : 21 = 3 * 7) :
  Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end NUMINAMATH_GPT_gcf_of_lcm_9_15_and_10_21_is_5_l1852_185235


namespace NUMINAMATH_GPT_sqrt_product_simplification_l1852_185252

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (21 * q) :=
by sorry

end NUMINAMATH_GPT_sqrt_product_simplification_l1852_185252


namespace NUMINAMATH_GPT_required_force_18_inch_wrench_l1852_185241

def inverse_force (l : ℕ) (k : ℕ) : ℕ := k / l

def extra_force : ℕ := 50

def initial_force : ℕ := 300

noncomputable
def handle_length_1 : ℕ := 12

noncomputable
def handle_length_2 : ℕ := 18

noncomputable
def adjusted_force : ℕ := inverse_force handle_length_2 (initial_force * handle_length_1)

theorem required_force_18_inch_wrench : 
  adjusted_force + extra_force = 250 := 
by
  sorry

end NUMINAMATH_GPT_required_force_18_inch_wrench_l1852_185241


namespace NUMINAMATH_GPT_fish_catch_l1852_185299

theorem fish_catch (B : ℕ) (K : ℕ) (hB : B = 5) (hK : K = 2 * B) : B + K = 15 :=
by
  sorry

end NUMINAMATH_GPT_fish_catch_l1852_185299


namespace NUMINAMATH_GPT_compute_expression_l1852_185280

theorem compute_expression (x : ℤ) (h : x = 3) : (x^8 + 24 * x^4 + 144) / (x^4 + 12) = 93 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_compute_expression_l1852_185280


namespace NUMINAMATH_GPT_cowboy_shortest_distance_l1852_185251

noncomputable def distance : ℝ :=
  let C := (0, 5)
  let B := (-10, 11)
  let C' := (0, -5)
  5 + Real.sqrt ((C'.1 - B.1)^2 + (C'.2 - B.2)^2)

theorem cowboy_shortest_distance :
  distance = 5 + Real.sqrt 356 :=
by
  sorry

end NUMINAMATH_GPT_cowboy_shortest_distance_l1852_185251


namespace NUMINAMATH_GPT_Tim_total_payment_l1852_185237

-- Define the context for the problem
def manicure_cost : ℝ := 30
def tip_percentage : ℝ := 0.3

-- Define the total amount paid as the sum of the manicure cost and the tip
def total_amount_paid (cost : ℝ) (tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- The theorem to be proven
theorem Tim_total_payment : total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end NUMINAMATH_GPT_Tim_total_payment_l1852_185237


namespace NUMINAMATH_GPT_chameleons_changed_color_l1852_185221

-- Define a structure to encapsulate the conditions
structure ChameleonProblem where
  total_chameleons : ℕ
  initial_blue : ℕ -> ℕ
  remaining_blue : ℕ -> ℕ
  red_after_change : ℕ -> ℕ

-- Provide the specific problem instance
def chameleonProblemInstance : ChameleonProblem := {
  total_chameleons := 140,
  initial_blue := λ x => 5 * x,
  remaining_blue := id, -- remaining_blue(x) = x
  red_after_change := λ x => 3 * (140 - 5 * x)
}

-- Define the main theorem
theorem chameleons_changed_color (x : ℕ) :
  (chameleonProblemInstance.initial_blue x - chameleonProblemInstance.remaining_blue x) = 80 :=
by
  sorry

end NUMINAMATH_GPT_chameleons_changed_color_l1852_185221


namespace NUMINAMATH_GPT_simplified_value_l1852_185220

-- Define the given expression
def expr := (10^0.6) * (10^0.4) * (10^0.4) * (10^0.1) * (10^0.5) / (10^0.3)

-- State the theorem
theorem simplified_value : expr = 10^1.7 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_simplified_value_l1852_185220


namespace NUMINAMATH_GPT_largest_lcm_value_is_60_l1852_185254

-- Define the conditions
def lcm_values : List ℕ := [Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 9, Nat.lcm 15 12, Nat.lcm 15 10, Nat.lcm 15 15]

-- State the proof problem
theorem largest_lcm_value_is_60 : lcm_values.maximum = some 60 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_largest_lcm_value_is_60_l1852_185254


namespace NUMINAMATH_GPT_correct_addition_l1852_185223

-- Define the initial conditions and goal
theorem correct_addition (x : ℕ) : (x + 26 = 61) → (x + 62 = 97) :=
by
  intro h
  -- Proof steps would be provided here
  sorry

end NUMINAMATH_GPT_correct_addition_l1852_185223


namespace NUMINAMATH_GPT_dinner_potatoes_l1852_185243

def lunch_potatoes : ℕ := 5
def total_potatoes : ℕ := 7

theorem dinner_potatoes : total_potatoes - lunch_potatoes = 2 :=
by
  sorry

end NUMINAMATH_GPT_dinner_potatoes_l1852_185243


namespace NUMINAMATH_GPT_print_shop_x_charges_l1852_185246

theorem print_shop_x_charges (x : ℝ) (h1 : ∀ y : ℝ, y = 1.70) (h2 : 40 * x + 20 = 40 * 1.70) : x = 1.20 :=
by
  sorry

end NUMINAMATH_GPT_print_shop_x_charges_l1852_185246


namespace NUMINAMATH_GPT_fraction_equality_solution_l1852_185294

theorem fraction_equality_solution (x : ℝ) : (5 + x) / (7 + x) = (2 + x) / (3 + x) → x = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fraction_equality_solution_l1852_185294


namespace NUMINAMATH_GPT_number_of_elements_in_set_P_l1852_185247

theorem number_of_elements_in_set_P
  (p q : ℕ) -- we are dealing with non-negative integers here
  (h1 : p = 3 * q)
  (h2 : p + q = 4500)
  : p = 3375 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_number_of_elements_in_set_P_l1852_185247


namespace NUMINAMATH_GPT_compute_105_squared_l1852_185258

theorem compute_105_squared : 105^2 = 11025 :=
by
  sorry

end NUMINAMATH_GPT_compute_105_squared_l1852_185258


namespace NUMINAMATH_GPT_traveler_distance_l1852_185296

theorem traveler_distance (a b c d : ℕ) (h1 : a = 24) (h2 : b = 15) (h3 : c = 10) (h4 : d = 9) :
  let net_ns := a - c
  let net_ew := b - d
  let distance := Real.sqrt ((net_ns ^ 2) + (net_ew ^ 2))
  distance = 2 * Real.sqrt 58 := 
by
  sorry

end NUMINAMATH_GPT_traveler_distance_l1852_185296


namespace NUMINAMATH_GPT_find_m_l1852_185201

open Set

variable (A B : Set ℝ) (m : ℝ)

theorem find_m (h : A = {-1, 2, 2 * m - 1}) (h2 : B = {2, m^2}) (h3 : B ⊆ A) : m = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l1852_185201


namespace NUMINAMATH_GPT_tetrahedron_pythagorean_theorem_l1852_185290

noncomputable section

variables {a b c : ℝ} {S_ABC S_VAB S_VBC S_VAC : ℝ}

-- Conditions
def is_right_triangle (a b c : ℝ) := c^2 = a^2 + b^2
def is_right_tetrahedron (S_ABC S_VAB S_VBC S_VAC : ℝ) := 
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2

-- Theorem Statement
theorem tetrahedron_pythagorean_theorem (a b c S_ABC S_VAB S_VBC S_VAC : ℝ) 
  (h1 : is_right_triangle a b c)
  (h2 : S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2) :
  S_ABC^2 = S_VAB^2 + S_VBC^2 + S_VAC^2 := 
by sorry

end NUMINAMATH_GPT_tetrahedron_pythagorean_theorem_l1852_185290


namespace NUMINAMATH_GPT_sum_y_coordinates_of_other_vertices_l1852_185227

theorem sum_y_coordinates_of_other_vertices (x1 y1 x2 y2 : ℤ) 
  (h1 : (x1, y1) = (2, 10)) (h2 : (x2, y2) = (-6, -6)) :
  (∃ y3 y4 : ℤ, (4 : ℤ) = y3 + y4) :=
by
  sorry

end NUMINAMATH_GPT_sum_y_coordinates_of_other_vertices_l1852_185227


namespace NUMINAMATH_GPT_prism_diagonal_and_surface_area_l1852_185209

/-- 
  A rectangular prism has dimensions of 12 inches, 16 inches, and 21 inches.
  Prove that the length of the diagonal is 29 inches, 
  and the total surface area of the prism is 1560 square inches.
-/
theorem prism_diagonal_and_surface_area :
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  d = 29 ∧ S = 1560 := by
  let a := 12
  let b := 16
  let c := 21
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let S := 2 * (a * b + b * c + c * a)
  sorry

end NUMINAMATH_GPT_prism_diagonal_and_surface_area_l1852_185209


namespace NUMINAMATH_GPT_min_value_of_expression_l1852_185216

noncomputable def problem_statement : Prop :=
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ ((1/x) + (1/y) + (1/z) = 9) ∧ (x^2 * y^3 * z^2 = 1/2268)

theorem min_value_of_expression :
  problem_statement := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1852_185216


namespace NUMINAMATH_GPT_find_a3_a4_a5_l1852_185230

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 2 * a n

noncomputable def sum_first_three (a : ℕ → ℝ) : Prop :=
a 0 + a 1 + a 2 = 21

theorem find_a3_a4_a5 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : sum_first_three a) :
  a 2 + a 3 + a 4 = 84 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_a4_a5_l1852_185230


namespace NUMINAMATH_GPT_total_amount_is_4200_l1852_185285

variables (p q r : ℕ)
variable (total_amount : ℕ)
variable (r_has_two_thirds : total_amount / 3 * 2 = 2800)
variable (r_value : r = 2800)

theorem total_amount_is_4200 (h1 : total_amount / 3 * 2 = 2800) (h2 : r = 2800) : total_amount = 4200 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_is_4200_l1852_185285


namespace NUMINAMATH_GPT_solve_equation_l1852_185274

theorem solve_equation 
  (x : ℚ)
  (h : (x^2 + 3*x + 4)/(x + 5) = x + 6) :
  x = -13/4 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1852_185274


namespace NUMINAMATH_GPT_vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l1852_185206

/-- The test consists of 30 questions, each with two possible answers (one correct and one incorrect). 
    Vitya can proceed in such a way that he can guarantee to know all the correct answers no later than:
    (a) after the 29th attempt (and answer all questions correctly on the 30th attempt)
    (b) after the 24th attempt (and answer all questions correctly on the 25th attempt)
    - Vitya initially does not know any of the answers.
    - The test is always the same.
-/
def vitya_test (k : Nat) : Prop :=
  k = 30 ∧ (∀ (attempts : Fin 30 → Bool), attempts 30 = attempts 29 ∧ attempts 30)

theorem vitya_knows_answers_29_attempts :
  vitya_test 30 :=
by 
  sorry

theorem vitya_knows_answers_24_attempts :
  vitya_test 25 :=
by 
  sorry

end NUMINAMATH_GPT_vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l1852_185206


namespace NUMINAMATH_GPT_inequality_of_sums_l1852_185219

theorem inequality_of_sums
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1)
  (h2 : 0 < a2)
  (h3 : a1 > a2)
  (h4 : b1 ≥ a1)
  (h5 : b1 * b2 ≥ a1 * a2) :
  b1 + b2 ≥ a1 + a2 :=
by
  -- Here we don't provide the proof
  sorry

end NUMINAMATH_GPT_inequality_of_sums_l1852_185219


namespace NUMINAMATH_GPT_numerator_of_fraction_l1852_185217

theorem numerator_of_fraction (y x : ℝ) (hy : y > 0) (h : (9 * y) / 20 + x / y = 0.75 * y) : x = 3 :=
sorry

end NUMINAMATH_GPT_numerator_of_fraction_l1852_185217


namespace NUMINAMATH_GPT_dave_fifth_store_car_count_l1852_185215

theorem dave_fifth_store_car_count :
  let cars_first_store := 30
  let cars_second_store := 14
  let cars_third_store := 14
  let cars_fourth_store := 21
  let mean := 20.8
  let total_cars := mean * 5
  let total_cars_first_four := cars_first_store + cars_second_store + cars_third_store + cars_fourth_store
  total_cars - total_cars_first_four = 25 := by
sorry

end NUMINAMATH_GPT_dave_fifth_store_car_count_l1852_185215


namespace NUMINAMATH_GPT_rectangle_cut_into_square_l1852_185240

theorem rectangle_cut_into_square (a b : ℝ) (h : a ≤ 4 * b) : 4 * b ≥ a := 
by 
  exact h

end NUMINAMATH_GPT_rectangle_cut_into_square_l1852_185240


namespace NUMINAMATH_GPT_double_series_evaluation_l1852_185238

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m * n * (m + n + 2))) = (3 / 2 : ℝ) :=
sorry

end NUMINAMATH_GPT_double_series_evaluation_l1852_185238


namespace NUMINAMATH_GPT_exists_increasing_sequence_l1852_185255

theorem exists_increasing_sequence (n : ℕ) : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∃ x : ℕ → ℕ, (∀ i : ℕ, 1 ≤ i → i ≤ n → x i < x (i + 1)) :=
by
  sorry

end NUMINAMATH_GPT_exists_increasing_sequence_l1852_185255


namespace NUMINAMATH_GPT_intersection_A_B_l1852_185256

noncomputable def A : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def B : Set ℝ := {y | ∃ x, y = Real.log (x^2 + 1) ∧ y ≥ 0}

theorem intersection_A_B : A ∩ {x | ∃ y, y = Real.log (x^2 + 1) ∧ y ≥ 0} = {x | 0 < x ∧ x < 2} :=
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1852_185256


namespace NUMINAMATH_GPT_circle_diameter_l1852_185233

-- The problem statement in Lean 4

theorem circle_diameter
  (d α β : ℝ) :
  ∃ r: ℝ,
  r * 2 = d * (Real.sin α) * (Real.sin β) / (Real.cos ((α + β) / 2) * (Real.sin ((α - β) / 2))) :=
sorry

end NUMINAMATH_GPT_circle_diameter_l1852_185233


namespace NUMINAMATH_GPT_determine_rectangle_R_area_l1852_185222

def side_length_large_square (s : ℕ) : Prop :=
  s = 4

def area_rectangle_R (s : ℕ) (area_R : ℕ) : Prop :=
  s * s - (1 * 4 + 1 * 1) = area_R

theorem determine_rectangle_R_area :
  ∃ (s : ℕ) (area_R : ℕ), side_length_large_square s ∧ area_rectangle_R s area_R :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_rectangle_R_area_l1852_185222


namespace NUMINAMATH_GPT_geom_sum_eq_six_l1852_185236

variable (a : ℕ → ℝ)
variable (r : ℝ) -- common ratio for geometric sequence

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a (n + 1) > 0
axiom given_eq : a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36

-- Proof statement
theorem geom_sum_eq_six : a 2 + a 5 = 6 :=
sorry

end NUMINAMATH_GPT_geom_sum_eq_six_l1852_185236


namespace NUMINAMATH_GPT_prob_same_color_is_correct_l1852_185260

noncomputable def prob_same_color : ℚ :=
  let green_prob := (8 : ℚ) / 10
  let red_prob := (2 : ℚ) / 10
  (green_prob)^2 + (red_prob)^2

theorem prob_same_color_is_correct :
  prob_same_color = 17 / 25 := by
  sorry

end NUMINAMATH_GPT_prob_same_color_is_correct_l1852_185260


namespace NUMINAMATH_GPT_inequality_proof_l1852_185225

variables {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_min : min (min (a * b) (b * c)) (c * a) ≥ 1) :
  (↑((a^2 + 1) * (b^2 + 1) * (c^2 + 1)) ^ (1 / 3 : ℝ)) ≤ ((a + b + c) / 3) ^ 2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1852_185225


namespace NUMINAMATH_GPT_six_degree_below_zero_is_minus_six_degrees_l1852_185298

def temp_above_zero (temp: Int) : String := "+" ++ toString temp ++ "°C"

def temp_below_zero (temp: Int) : String := "-" ++ toString temp ++ "°C"

-- Statement of the theorem
theorem six_degree_below_zero_is_minus_six_degrees:
  temp_below_zero 6 = "-6°C" :=
by
  sorry

end NUMINAMATH_GPT_six_degree_below_zero_is_minus_six_degrees_l1852_185298


namespace NUMINAMATH_GPT_max_intersection_points_circles_lines_l1852_185224

-- Definitions based on the conditions
def num_circles : ℕ := 3
def num_lines : ℕ := 2

-- Function to calculate the number of points of intersection
def max_points_of_intersection (num_circles num_lines : ℕ) : ℕ :=
  (num_circles * (num_circles - 1) / 2) * 2 + 
  num_circles * num_lines * 2 + 
  (num_lines * (num_lines - 1) / 2)

-- The proof statement
theorem max_intersection_points_circles_lines :
  max_points_of_intersection num_circles num_lines = 19 :=
by
  sorry

end NUMINAMATH_GPT_max_intersection_points_circles_lines_l1852_185224


namespace NUMINAMATH_GPT_value_of_squared_difference_l1852_185245

theorem value_of_squared_difference (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) :
  (x - y)^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_squared_difference_l1852_185245


namespace NUMINAMATH_GPT_original_fraction_is_two_thirds_l1852_185293

theorem original_fraction_is_two_thirds
  (x y : ℕ)
  (h1 : x / (y + 1) = 1 / 2)
  (h2 : (x + 1) / y = 1) :
  x / y = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_original_fraction_is_two_thirds_l1852_185293


namespace NUMINAMATH_GPT_expression_for_f_minimum_positive_period_of_f_range_of_f_l1852_185264

noncomputable def f (x : ℝ) : ℝ :=
  let A := (2, 0) 
  let B := (0, 2)
  let C := (Real.cos (2 * x), Real.sin (2 * x))
  let AB := (B.1 - A.1, B.2 - A.2) 
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.fst * AC.fst + AB.snd * AC.snd 

theorem expression_for_f (x : ℝ) :
  f x = 2 * Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4) + 4 :=
by sorry

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by sorry

theorem range_of_f (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) :
  2 < f x ∧ f x ≤ 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_expression_for_f_minimum_positive_period_of_f_range_of_f_l1852_185264


namespace NUMINAMATH_GPT_skittles_total_l1852_185208

-- Define the conditions
def skittles_per_friend : ℝ := 40.0
def number_of_friends : ℝ := 5.0

-- Define the target statement using the conditions
theorem skittles_total : (skittles_per_friend * number_of_friends = 200.0) :=
by 
  -- Using sorry to placeholder the proof
  sorry

end NUMINAMATH_GPT_skittles_total_l1852_185208


namespace NUMINAMATH_GPT_find_y_l1852_185200

theorem find_y (y: ℕ)
  (h1: ∃ (k : ℕ), y = 9 * k)
  (h2: y^2 > 225)
  (h3: y < 30)
: y = 18 ∨ y = 27 := 
sorry

end NUMINAMATH_GPT_find_y_l1852_185200


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_eq_1_l1852_185269

theorem arithmetic_sequence_a6_eq_1
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : S 11 = 11)
  (h2 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h3 : ∃ d, ∀ n, a n = a 1 + (n - 1) * d) :
  a 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_eq_1_l1852_185269


namespace NUMINAMATH_GPT_total_money_correct_l1852_185202

def shelly_has_total_money : Prop :=
  ∃ (ten_dollar_bills five_dollar_bills : ℕ), 
    ten_dollar_bills = 10 ∧
    five_dollar_bills = ten_dollar_bills - 4 ∧
    (10 * ten_dollar_bills + 5 * five_dollar_bills = 130)

theorem total_money_correct : shelly_has_total_money :=
by
  sorry

end NUMINAMATH_GPT_total_money_correct_l1852_185202


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1852_185286

theorem quadratic_no_real_roots (k : ℝ) : (∀ x : ℝ, x^2 + 2*x + k ≠ 0) ↔ k > 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1852_185286


namespace NUMINAMATH_GPT_division_proof_l1852_185248

theorem division_proof :
  ((2 * 4 * 6) / (1 + 3 + 5 + 7) - (1 * 3 * 5) / (2 + 4 + 6)) / (1 / 2) = 3.5 :=
by
  -- definitions based on conditions
  let numerator1 := 2 * 4 * 6
  let denominator1 := 1 + 3 + 5 + 7
  let numerator2 := 1 * 3 * 5
  let denominator2 := 2 + 4 + 6
  -- the statement of the theorem
  sorry

end NUMINAMATH_GPT_division_proof_l1852_185248
