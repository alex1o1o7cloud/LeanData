import Mathlib

namespace NUMINAMATH_GPT_vector_n_value_l1333_133361

theorem vector_n_value {n : ℤ} (hAB : (2, 4) = (2, 4)) (hBC : (-2, n) = (-2, n)) (hAC : (0, 2) = (2 + -2, 4 + n)) : n = -2 :=
by
  sorry

end NUMINAMATH_GPT_vector_n_value_l1333_133361


namespace NUMINAMATH_GPT_total_hours_worked_l1333_133342

def hours_day1 : ℝ := 2.5
def increment_day2 : ℝ := 0.5
def hours_day2 : ℝ := hours_day1 + increment_day2
def hours_day3 : ℝ := 3.75

theorem total_hours_worked :
  hours_day1 + hours_day2 + hours_day3 = 9.25 :=
sorry

end NUMINAMATH_GPT_total_hours_worked_l1333_133342


namespace NUMINAMATH_GPT_find_number_l1333_133379

theorem find_number : ∃ n : ℕ, (∃ x : ℕ, x / 15 = 4 ∧ x^2 = n) ∧ n = 3600 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1333_133379


namespace NUMINAMATH_GPT_binom_eight_four_l1333_133352

theorem binom_eight_four : (Nat.choose 8 4) = 70 :=
by
  sorry

end NUMINAMATH_GPT_binom_eight_four_l1333_133352


namespace NUMINAMATH_GPT_profit_share_difference_l1333_133354

theorem profit_share_difference
    (P_A P_B P_C P_D : ℕ) (R_A R_B R_C R_D parts_A parts_B parts_C parts_D : ℕ) (profit_B : ℕ)
    (h1 : P_A = 8000) (h2 : P_B = 10000) (h3 : P_C = 12000) (h4 : P_D = 15000)
    (h5 : R_A = 3) (h6 : R_B = 5) (h7 : R_C = 6) (h8 : R_D = 7)
    (h9: profit_B = 2000) :
    profit_B / R_B = 400 ∧ P_C * R_C / R_B - P_A * R_A / R_B = 1200 :=
by
  sorry

end NUMINAMATH_GPT_profit_share_difference_l1333_133354


namespace NUMINAMATH_GPT_abs_neg_six_l1333_133324

theorem abs_neg_six : abs (-6) = 6 :=
sorry

end NUMINAMATH_GPT_abs_neg_six_l1333_133324


namespace NUMINAMATH_GPT_sally_initial_peaches_l1333_133338

section
variables 
  (peaches_after : ℕ)
  (peaches_picked : ℕ)
  (initial_peaches : ℕ)

theorem sally_initial_peaches 
    (h1 : peaches_picked = 42)
    (h2 : peaches_after = 55)
    (h3 : peaches_after = initial_peaches + peaches_picked) : 
    initial_peaches = 13 := 
by 
  sorry
end

end NUMINAMATH_GPT_sally_initial_peaches_l1333_133338


namespace NUMINAMATH_GPT_find_u_v_l1333_133355

theorem find_u_v (u v : ℤ) (huv_pos : 0 < v ∧ v < u) (area_eq : u^2 + 3 * u * v = 615) : 
  u + v = 45 :=
sorry

end NUMINAMATH_GPT_find_u_v_l1333_133355


namespace NUMINAMATH_GPT_pictures_remaining_l1333_133323

-- Define the initial number of pictures taken at the zoo and museum
def zoo_pictures : Nat := 50
def museum_pictures : Nat := 8
-- Define the number of pictures deleted
def deleted_pictures : Nat := 38

-- Define the total number of pictures taken initially and remaining after deletion
def total_pictures : Nat := zoo_pictures + museum_pictures
def remaining_pictures : Nat := total_pictures - deleted_pictures

theorem pictures_remaining : remaining_pictures = 20 := 
by 
  -- This theorem states that, given the conditions, the remaining pictures count must be 20
  sorry

end NUMINAMATH_GPT_pictures_remaining_l1333_133323


namespace NUMINAMATH_GPT_number_of_planes_l1333_133397

theorem number_of_planes (total_wings: ℕ) (wings_per_plane: ℕ) 
  (h1: total_wings = 50) (h2: wings_per_plane = 2) : 
  total_wings / wings_per_plane = 25 := by 
  sorry

end NUMINAMATH_GPT_number_of_planes_l1333_133397


namespace NUMINAMATH_GPT_minimum_value_abs_sum_l1333_133329

theorem minimum_value_abs_sum (α β γ : ℝ) (h1 : α + β + γ = 2) (h2 : α * β * γ = 4) : 
  |α| + |β| + |γ| ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_abs_sum_l1333_133329


namespace NUMINAMATH_GPT_binomial_product_result_l1333_133384

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end NUMINAMATH_GPT_binomial_product_result_l1333_133384


namespace NUMINAMATH_GPT_smallest_sum_of_big_in_circle_l1333_133347

theorem smallest_sum_of_big_in_circle (arranged_circle : Fin 8 → ℕ) (h_circle : ∀ n, arranged_circle n ∈ Finset.range (9) ∧ arranged_circle n > 0) :
  (∀ n, (arranged_circle n > arranged_circle (n + 1) % 8 ∧ arranged_circle n > arranged_circle (n + 7) % 8) ∨ (arranged_circle n < arranged_circle (n + 1) % 8 ∧ arranged_circle n < arranged_circle (n + 7) % 8)) →
  ∃ big_indices : Finset (Fin 8), big_indices.card = 4 ∧ big_indices.sum arranged_circle = 23 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_big_in_circle_l1333_133347


namespace NUMINAMATH_GPT_solve_for_x_l1333_133396

-- Define the necessary condition
def problem_statement (x : ℚ) : Prop :=
  x / 4 - x - 3 / 6 = 1

-- Prove that if the condition holds, then x = -14/9
theorem solve_for_x (x : ℚ) (h : problem_statement x) : x = -14 / 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1333_133396


namespace NUMINAMATH_GPT_find_a_l1333_133335

theorem find_a (a : ℝ) (x_values y_values : List ℝ)
  (h_y : ∀ x, List.getD y_values x 0 = 2.1 * List.getD x_values x 1 - 0.3) :
  a = 10 :=
by
  have h_mean_x : (1 + 2 + 3 + 4 + 5) / 5 = 3 := by norm_num
  have h_sum_y : (2 + 3 + 7 + 8 + a) / 5 = (2.1 * 3 - 0.3) := by sorry
  sorry

end NUMINAMATH_GPT_find_a_l1333_133335


namespace NUMINAMATH_GPT_fraction_order_l1333_133328

theorem fraction_order :
  (21:ℚ) / 17 < (23:ℚ) / 18 ∧ (23:ℚ) / 18 < (25:ℚ) / 19 :=
by
  sorry

end NUMINAMATH_GPT_fraction_order_l1333_133328


namespace NUMINAMATH_GPT_pat_kate_mark_ratio_l1333_133376

variables (P K M r : ℚ) 

theorem pat_kate_mark_ratio (h1 : P + K + M = 189) 
                            (h2 : P = r * K) 
                            (h3 : P = (1 / 3) * M) 
                            (h4 : M = K + 105) :
  r = 4 / 3 :=
sorry

end NUMINAMATH_GPT_pat_kate_mark_ratio_l1333_133376


namespace NUMINAMATH_GPT_lowest_price_per_component_l1333_133360

def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 6
def fixed_monthly_costs : ℝ := 16500
def components_per_month : ℕ := 150

theorem lowest_price_per_component (price_per_component : ℝ) :
  let total_cost_per_component := production_cost_per_component + shipping_cost_per_component
  let total_production_and_shipping_cost := total_cost_per_component * components_per_month
  let total_cost := total_production_and_shipping_cost + fixed_monthly_costs
  price_per_component = total_cost / components_per_month → price_per_component = 196 :=
by
  sorry

end NUMINAMATH_GPT_lowest_price_per_component_l1333_133360


namespace NUMINAMATH_GPT_car_travel_distance_l1333_133357

theorem car_travel_distance :
  let a := 36
  let d := -12
  let n := 4
  let S := (n / 2) * (2 * a + (n - 1) * d)
  S = 72 := by
    sorry

end NUMINAMATH_GPT_car_travel_distance_l1333_133357


namespace NUMINAMATH_GPT_quadratic_range_l1333_133313

theorem quadratic_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_range_l1333_133313


namespace NUMINAMATH_GPT_multiplication_expansion_l1333_133309

theorem multiplication_expansion (y : ℤ) :
  (y^4 + 9 * y^2 + 81) * (y^2 - 9) = y^6 - 729 :=
by
  sorry

end NUMINAMATH_GPT_multiplication_expansion_l1333_133309


namespace NUMINAMATH_GPT_abs_neg_2023_l1333_133321

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_l1333_133321


namespace NUMINAMATH_GPT_items_in_descending_order_l1333_133320

-- Assume we have four real numbers representing the weights of the items.
variables (C S B K : ℝ)

-- The conditions given in the problem.
axiom h1 : S > B
axiom h2 : C + B > S + K
axiom h3 : K + C = S + B

-- Define a predicate to check if the weights are in descending order.
def DescendingOrder (C S B K : ℝ) : Prop :=
  C > S ∧ S > B ∧ B > K

-- The theorem to prove the descending order of weights.
theorem items_in_descending_order : DescendingOrder C S B K :=
sorry

end NUMINAMATH_GPT_items_in_descending_order_l1333_133320


namespace NUMINAMATH_GPT_fraction_product_equals_64_l1333_133305

theorem fraction_product_equals_64 : 
  (1 / 4) * (8 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) * (1 / 8192) * (16384 / 1) = 64 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_equals_64_l1333_133305


namespace NUMINAMATH_GPT_transformer_minimum_load_l1333_133333

-- Define the conditions as hypotheses
def running_current_1 := 40
def running_current_2 := 60
def running_current_3 := 25

def start_multiplier_1 := 2
def start_multiplier_2 := 3
def start_multiplier_3 := 4

def units_1 := 3
def units_2 := 2
def units_3 := 1

def starting_current_1 := running_current_1 * start_multiplier_1
def starting_current_2 := running_current_2 * start_multiplier_2
def starting_current_3 := running_current_3 * start_multiplier_3

def total_starting_current_1 := starting_current_1 * units_1
def total_starting_current_2 := starting_current_2 * units_2
def total_starting_current_3 := starting_current_3 * units_3

def total_combined_minimum_current_load := 
  total_starting_current_1 + total_starting_current_2 + total_starting_current_3

-- The theorem to prove that the total combined minimum current load is 700A
theorem transformer_minimum_load : total_combined_minimum_current_load = 700 := by
  sorry

end NUMINAMATH_GPT_transformer_minimum_load_l1333_133333


namespace NUMINAMATH_GPT_additional_miles_needed_l1333_133389

theorem additional_miles_needed :
  ∀ (h : ℝ), (25 + 75 * h) / (5 / 8 + h) = 60 → 75 * h = 62.5 := 
by
  intros h H
  -- the rest of the proof goes here
  sorry

end NUMINAMATH_GPT_additional_miles_needed_l1333_133389


namespace NUMINAMATH_GPT_card_draw_sequential_same_suit_l1333_133388

theorem card_draw_sequential_same_suit : 
  let hearts := 13
  let diamonds := 13
  let total_suits := hearts + diamonds
  ∃ ways : ℕ, ways = total_suits * (hearts - 1) :=
by
  sorry

end NUMINAMATH_GPT_card_draw_sequential_same_suit_l1333_133388


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1333_133399

theorem isosceles_triangle_base_length
  (a b c: ℕ) 
  (h_iso: a = b ∨ a = c ∨ b = c)
  (h_perimeter: a + b + c = 21)
  (h_side: a = 5 ∨ b = 5 ∨ c = 5) :
  c = 5 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1333_133399


namespace NUMINAMATH_GPT_sin_ratio_equal_one_or_neg_one_l1333_133327

theorem sin_ratio_equal_one_or_neg_one
  (a b : Real)
  (h1 : Real.cos (a + b) = 1/4)
  (h2 : Real.cos (a - b) = 3/4) :
  (Real.sin a) / (Real.sin b) = 1 ∨ (Real.sin a) / (Real.sin b) = -1 :=
sorry

end NUMINAMATH_GPT_sin_ratio_equal_one_or_neg_one_l1333_133327


namespace NUMINAMATH_GPT_initial_workers_number_l1333_133371

-- Define the initial problem
variables {W : ℕ} -- Number of initial workers
variables (Work1 : ℕ := W * 8) -- Work done for the first hole
variables (Work2 : ℕ := (W + 65) * 6) -- Work done for the second hole
variables (Depth1 : ℕ := 30) -- Depth of the first hole
variables (Depth2 : ℕ := 55) -- Depth of the second hole

-- Expressing the conditions and question
theorem initial_workers_number : 8 * W * 55 = 30 * (W + 65) * 6 → W = 45 :=
by
  sorry

end NUMINAMATH_GPT_initial_workers_number_l1333_133371


namespace NUMINAMATH_GPT_number_of_puppies_l1333_133363

theorem number_of_puppies (P K : ℕ) (h1 : K = 2 * P + 14) (h2 : K = 78) : P = 32 :=
by sorry

end NUMINAMATH_GPT_number_of_puppies_l1333_133363


namespace NUMINAMATH_GPT_weekly_allowance_l1333_133315

theorem weekly_allowance (A : ℝ) (h1 : A / 2 + 6 = 11) : A = 10 := 
by 
  sorry

end NUMINAMATH_GPT_weekly_allowance_l1333_133315


namespace NUMINAMATH_GPT_minimum_sum_of_dimensions_l1333_133307

theorem minimum_sum_of_dimensions {a b c : ℕ} (h1 : a * b * c = 2310) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  a + b + c ≥ 42 := 
sorry

end NUMINAMATH_GPT_minimum_sum_of_dimensions_l1333_133307


namespace NUMINAMATH_GPT_min_positive_integer_expression_l1333_133387

theorem min_positive_integer_expression : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (m: ℝ) / 3 + 27 / (m: ℝ) ≥ (n: ℝ) / 3 + 27 / (n: ℝ)) ∧ (n / 3 + 27 / n = 6) :=
sorry

end NUMINAMATH_GPT_min_positive_integer_expression_l1333_133387


namespace NUMINAMATH_GPT_inequality_solution_l1333_133308

theorem inequality_solution 
  (x : ℝ) 
  (h1 : (x + 3) / 2 ≤ x + 2) 
  (h2 : 2 * (x + 4) > 4 * x + 2) : 
  -1 ≤ x ∧ x < 3 := sorry

end NUMINAMATH_GPT_inequality_solution_l1333_133308


namespace NUMINAMATH_GPT_percentage_volume_taken_by_cubes_l1333_133350

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

def volume_of_cube (side : ℝ) : ℝ := side ^ 3

noncomputable def total_cubes_fit (l w h side : ℝ) : ℝ := 
  (l / side) * (w / side) * (h / side)

theorem percentage_volume_taken_by_cubes (l w h side : ℝ) (hl : l = 12) (hw : w = 6) (hh : h = 9) (hside : side = 3) :
  volume_of_box l w h ≠ 0 → 
  (total_cubes_fit l w h side * volume_of_cube side / volume_of_box l w h) * 100 = 100 :=
by
  intros
  rw [hl, hw, hh, hside]
  simp only [volume_of_box, volume_of_cube, total_cubes_fit]
  sorry

end NUMINAMATH_GPT_percentage_volume_taken_by_cubes_l1333_133350


namespace NUMINAMATH_GPT_find_a7_l1333_133356

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, ∃ r, a (n + m) = (a n) * (r ^ m)

def sequence_properties (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ a 3 = 3 ∧ a 11 = 27

theorem find_a7 (a : ℕ → ℝ) (h : sequence_properties a) : a 7 = 9 := 
sorry

end NUMINAMATH_GPT_find_a7_l1333_133356


namespace NUMINAMATH_GPT_simplify_expression_l1333_133394

theorem simplify_expression (x : ℝ) :
  x * (4 * x^3 - 3 * x + 2) - 6 * (2 * x^3 + x^2 - 3 * x + 4) = 4 * x^4 - 12 * x^3 - 9 * x^2 + 20 * x - 24 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1333_133394


namespace NUMINAMATH_GPT_sum_of_roots_l1333_133378

theorem sum_of_roots (x : ℝ) :
  (3 * x - 2) * (x - 3) + (3 * x - 2) * (2 * x - 8) = 0 ->
  x = 2 / 3 ∨ x = 11 / 3 ->
  (2 / 3) + (11 / 3) = 13 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1333_133378


namespace NUMINAMATH_GPT_inequality_holds_l1333_133311

variable {a b c r : ℝ}
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

/-- 
To prove that the inequality r (ab + bc + ca) + (3 - r) (1/a + 1/b + 1/c) ≥ 9 
is true for all r satisfying 0 < r < 3 and for arbitrary positive reals a, b, c. 
-/
theorem inequality_holds (h : 0 < r ∧ r < 3) : 
  r * (a * b + b * c + c * a) + (3 - r) * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1333_133311


namespace NUMINAMATH_GPT_kids_wearing_shoes_l1333_133369

-- Definitions based on the problem's conditions
def total_kids := 22
def kids_with_socks := 12
def kids_with_both := 6
def barefoot_kids := 8

-- Theorem statement
theorem kids_wearing_shoes :
  (∃ (kids_with_shoes : ℕ), 
     (kids_with_shoes = (total_kids - barefoot_kids) - (kids_with_socks - kids_with_both) + kids_with_both) ∧ 
     kids_with_shoes = 8) :=
by
  sorry

end NUMINAMATH_GPT_kids_wearing_shoes_l1333_133369


namespace NUMINAMATH_GPT_Ludwig_daily_salary_l1333_133302

theorem Ludwig_daily_salary 
(D : ℝ)
(h_weekly_earnings : 4 * D + (3 / 2) * D = 55) :
D = 10 := 
by
  sorry

end NUMINAMATH_GPT_Ludwig_daily_salary_l1333_133302


namespace NUMINAMATH_GPT_brittany_age_when_returning_l1333_133325

def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_duration : ℕ := 4

theorem brittany_age_when_returning : (rebecca_age + age_difference + vacation_duration) = 32 := by
  sorry

end NUMINAMATH_GPT_brittany_age_when_returning_l1333_133325


namespace NUMINAMATH_GPT_boat_distance_downstream_is_68_l1333_133377

variable (boat_speed : ℕ) (stream_speed : ℕ) (time_hours : ℕ)

-- Given conditions
def effective_speed_downstream (boat_speed stream_speed : ℕ) : ℕ := boat_speed + stream_speed
def distance_downstream (speed time : ℕ) : ℕ := speed * time

theorem boat_distance_downstream_is_68 
  (h1 : boat_speed = 13) 
  (h2 : stream_speed = 4) 
  (h3 : time_hours = 4) : 
  distance_downstream (effective_speed_downstream boat_speed stream_speed) time_hours = 68 := 
by 
  sorry

end NUMINAMATH_GPT_boat_distance_downstream_is_68_l1333_133377


namespace NUMINAMATH_GPT_boat_speed_l1333_133373

theorem boat_speed (b s : ℝ) (h1 : b + s = 7) (h2 : b - s = 5) : b = 6 := 
by
  sorry

end NUMINAMATH_GPT_boat_speed_l1333_133373


namespace NUMINAMATH_GPT_product_of_three_numbers_is_correct_l1333_133303

noncomputable def sum_three_numbers_product (x y z n : ℚ) : Prop :=
  x + y + z = 200 ∧
  8 * x = y - 12 ∧
  8 * x = z + 12 ∧
  (x * y * z = 502147200 / 4913)

theorem product_of_three_numbers_is_correct :
  ∃ (x y z n : ℚ), sum_three_numbers_product x y z n :=
by
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_is_correct_l1333_133303


namespace NUMINAMATH_GPT_expansion_coeff_sum_l1333_133380

theorem expansion_coeff_sum
  (a : ℕ → ℤ)
  (h : ∀ x y : ℤ, (x - 2 * y) ^ 5 * (x + 3 * y) ^ 4 = 
    a 9 * x ^ 9 + 
    a 8 * x ^ 8 * y + 
    a 7 * x ^ 7 * y ^ 2 + 
    a 6 * x ^ 6 * y ^ 3 + 
    a 5 * x ^ 5 * y ^ 4 + 
    a 4 * x ^ 4 * y ^ 5 + 
    a 3 * x ^ 3 * y ^ 6 + 
    a 2 * x ^ 2 * y ^ 7 + 
    a 1 * x * y ^ 8 + 
    a 0 * y ^ 9) :
  a 0 + a 8 = -2602 := by
  sorry

end NUMINAMATH_GPT_expansion_coeff_sum_l1333_133380


namespace NUMINAMATH_GPT_how_many_halves_to_sum_one_and_one_half_l1333_133346

theorem how_many_halves_to_sum_one_and_one_half : 
  (3 / 2) / (1 / 2) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_how_many_halves_to_sum_one_and_one_half_l1333_133346


namespace NUMINAMATH_GPT_expectation_is_four_thirds_l1333_133300

-- Define the probability function
def P_ξ (k : ℕ) : ℚ :=
  if k = 0 then (1/2)^2 * (2/3)
  else if k = 1 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3)
  else if k = 2 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3) + (1/2) * (1/2) * (1/3)
  else if k = 3 then (1/2) * (1/2) * (1/3)
  else 0

-- Define the expected value function
def E_ξ : ℚ :=
  0 * P_ξ 0 + 1 * P_ξ 1 + 2 * P_ξ 2 + 3 * P_ξ 3

-- Formal statement of the problem
theorem expectation_is_four_thirds : E_ξ = 4 / 3 :=
  sorry

end NUMINAMATH_GPT_expectation_is_four_thirds_l1333_133300


namespace NUMINAMATH_GPT_find_a_l1333_133301

noncomputable def f (x a : ℝ) : ℝ := 4 * x ^ 2 - 4 * a * x + a ^ 2 - 2 * a + 2

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 2 → f y a ≤ f x a) ∧ f 0 a = 3 ∧ f 2 a = 3 → 
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_find_a_l1333_133301


namespace NUMINAMATH_GPT_find_general_formula_sum_b_n_less_than_two_l1333_133385

noncomputable def a_n (n : ℕ) : ℕ := n

noncomputable def S_n (n : ℕ) : ℚ := (n^2 + n) / 2

noncomputable def b_n (n : ℕ) : ℚ := 1 / S_n n

theorem find_general_formula (n : ℕ) : b_n n = 2 / (n^2 + n) := by 
  sorry

theorem sum_b_n_less_than_two (n : ℕ) :
  Finset.sum (Finset.range n) (λ k => b_n (k + 1)) < 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_general_formula_sum_b_n_less_than_two_l1333_133385


namespace NUMINAMATH_GPT_jim_loan_inequality_l1333_133375

noncomputable def A (t : ℕ) : ℝ := 1500 * (1.06 ^ t)

theorem jim_loan_inequality : ∃ t : ℕ, A t > 3000 ∧ ∀ t' : ℕ, t' < t → A t' ≤ 3000 :=
by
  sorry

end NUMINAMATH_GPT_jim_loan_inequality_l1333_133375


namespace NUMINAMATH_GPT_total_number_of_seats_l1333_133337

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end NUMINAMATH_GPT_total_number_of_seats_l1333_133337


namespace NUMINAMATH_GPT_range_of_m_l1333_133310

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1333_133310


namespace NUMINAMATH_GPT_eval_expression_l1333_133331

theorem eval_expression :
  let x := 2
  let y := -3
  let z := 1
  x^2 + y^2 - z^2 + 2 * x * y + 3 * z = 0 := by
sorry

end NUMINAMATH_GPT_eval_expression_l1333_133331


namespace NUMINAMATH_GPT_solve_system1_l1333_133348

structure SystemOfEquations :=
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)

def system1 : SystemOfEquations :=
  { a₁ := 1, b₁ := -3, c₁ := 4,
    a₂ := 2, b₂ := -1, c₂ := 3 }

theorem solve_system1 :
  ∃ x y : ℝ, x - 3 * y = 4 ∧ 2 * x - y = 3 ∧ x = 1 ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system1_l1333_133348


namespace NUMINAMATH_GPT_length_of_first_train_l1333_133395

noncomputable def first_train_length 
  (speed_first_train_km_h : ℕ) 
  (speed_second_train_km_h : ℕ) 
  (length_second_train_m : ℕ) 
  (time_seconds : ℝ) 
  (relative_speed_m_s : ℝ) : ℝ :=
  let relative_speed_mps := (speed_first_train_km_h + speed_second_train_km_h) * (5 / 18)
  let distance_covered := relative_speed_mps * time_seconds
  let length_first_train := distance_covered - length_second_train_m
  length_first_train

theorem length_of_first_train : 
  first_train_length 40 50 165 11.039116870650348 25 = 110.9779217662587 :=
by 
  rw [first_train_length]
  sorry

end NUMINAMATH_GPT_length_of_first_train_l1333_133395


namespace NUMINAMATH_GPT_dealer_gross_profit_l1333_133391

noncomputable def computeGrossProfit (purchasePrice initialMarkupRate discountRate salesTaxRate: ℝ) : ℝ :=
  let initialSellingPrice := purchasePrice / (1 - initialMarkupRate)
  let discount := discountRate * initialSellingPrice
  let discountedPrice := initialSellingPrice - discount
  let salesTax := salesTaxRate * discountedPrice
  let finalSellingPrice := discountedPrice + salesTax
  finalSellingPrice - purchasePrice - discount

theorem dealer_gross_profit 
  (purchasePrice : ℝ)
  (initialMarkupRate : ℝ)
  (discountRate : ℝ)
  (salesTaxRate : ℝ) 
  (grossProfit : ℝ) :
  purchasePrice = 150 →
  initialMarkupRate = 0.25 →
  discountRate = 0.10 →
  salesTaxRate = 0.05 →
  grossProfit = 19 →
  computeGrossProfit purchasePrice initialMarkupRate discountRate salesTaxRate = grossProfit :=
  by
    intros hp hm hd hs hg
    rw [hp, hm, hd, hs, hg]
    rw [computeGrossProfit]
    sorry

end NUMINAMATH_GPT_dealer_gross_profit_l1333_133391


namespace NUMINAMATH_GPT_exists_odd_a_b_and_positive_k_l1333_133358

theorem exists_odd_a_b_and_positive_k (m : ℤ) :
  ∃ (a b : ℤ) (k : ℕ), a % 2 = 1 ∧ b % 2 = 1 ∧ k > 0 ∧ 2 * m = a^5 + b^5 + k * 2^100 := 
sorry

end NUMINAMATH_GPT_exists_odd_a_b_and_positive_k_l1333_133358


namespace NUMINAMATH_GPT_construction_work_rate_l1333_133314

theorem construction_work_rate (C : ℝ) 
  (h1 : ∀ t1 : ℝ, t1 = 10 → t1 * 8 = 80)
  (h2 : ∀ t2 : ℝ, t2 = 15 → t2 * C + 80 ≥ 300)
  (h3 : ∀ t : ℝ, t = 25 → ∀ t1 t2 : ℝ, t = t1 + t2 → t1 = 10 → t2 = 15)
  : C = 14.67 :=
by
  sorry

end NUMINAMATH_GPT_construction_work_rate_l1333_133314


namespace NUMINAMATH_GPT_line_intersects_parabola_at_one_point_l1333_133382
   
   theorem line_intersects_parabola_at_one_point (k : ℝ) :
     (∃ y : ℝ, (x = 3 * y^2 - 7 * y + 2 ∧ x = k) → x = k) ↔ k = (-25 / 12) :=
   by
     -- your proof goes here
     sorry
   
end NUMINAMATH_GPT_line_intersects_parabola_at_one_point_l1333_133382


namespace NUMINAMATH_GPT_log_ordering_l1333_133374

theorem log_ordering 
  (a b c : ℝ) 
  (ha: a = Real.log 3 / Real.log 2) 
  (hb: b = Real.log 2 / Real.log 3) 
  (hc: c = Real.log 0.5 / Real.log 10) : 
  a > b ∧ b > c := 
by 
  sorry

end NUMINAMATH_GPT_log_ordering_l1333_133374


namespace NUMINAMATH_GPT_sugar_percentage_first_solution_l1333_133317

theorem sugar_percentage_first_solution 
  (x : ℝ) (h1 : 0 < x ∧ x < 100) 
  (h2 : 17 = 3 / 4 * x + 1 / 4 * 38) : 
  x = 10 :=
sorry

end NUMINAMATH_GPT_sugar_percentage_first_solution_l1333_133317


namespace NUMINAMATH_GPT_radius_of_circle_l1333_133392

variables (O P A B : Type) [MetricSpace O] [MetricSpace P] [MetricSpace A] [MetricSpace B]
variables (circle_radius : ℝ) (PA PB OP : ℝ)

theorem radius_of_circle
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  (circle_radius : ℝ)
  : circle_radius = 7 :=
by sorry

end NUMINAMATH_GPT_radius_of_circle_l1333_133392


namespace NUMINAMATH_GPT_non_real_roots_interval_l1333_133341

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end NUMINAMATH_GPT_non_real_roots_interval_l1333_133341


namespace NUMINAMATH_GPT_problem1_problem2_l1333_133343

-- Problem (1) proof statement
theorem problem1 (a : ℝ) (h : a ≠ 0) : 
  3 * a^2 * a^3 + a^7 / a^2 = 4 * a^5 :=
by
  sorry

-- Problem (2) proof statement
theorem problem2 (x : ℝ) : 
  (x - 1)^2 - x * (x + 1) + (-2023)^0 = -3 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1333_133343


namespace NUMINAMATH_GPT_total_books_l1333_133393

def sam_books : ℕ := 110
def joan_books : ℕ := 102

theorem total_books : sam_books + joan_books = 212 := by
  sorry

end NUMINAMATH_GPT_total_books_l1333_133393


namespace NUMINAMATH_GPT_avg_two_ab_l1333_133322

-- Defining the weights and conditions
variables (A B C : ℕ)

-- The conditions provided in the problem
def avg_three (A B C : ℕ) := (A + B + C) / 3 = 45
def avg_two_bc (B C : ℕ) := (B + C) / 2 = 43
def weight_b (B : ℕ) := B = 35

-- The target proof statement
theorem avg_two_ab (A B C : ℕ) (h1 : avg_three A B C) (h2 : avg_two_bc B C) (h3 : weight_b B) : (A + B) / 2 = 42 := 
sorry

end NUMINAMATH_GPT_avg_two_ab_l1333_133322


namespace NUMINAMATH_GPT_katherine_has_5_bananas_l1333_133304

/-- Katherine has 4 apples -/
def apples : ℕ := 4

/-- Katherine has 3 times as many pears as apples -/
def pears : ℕ := 3 * apples

/-- Katherine has a total of 21 pieces of fruit (apples + pears + bananas) -/
def total_fruit : ℕ := 21

/-- Define the number of bananas Katherine has -/
def bananas : ℕ := total_fruit - (apples + pears)

/-- Prove that Katherine has 5 bananas -/
theorem katherine_has_5_bananas : bananas = 5 := by
  sorry

end NUMINAMATH_GPT_katherine_has_5_bananas_l1333_133304


namespace NUMINAMATH_GPT_change_from_fifteen_dollars_l1333_133344

theorem change_from_fifteen_dollars : 
  ∀ (cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid : ℕ),
  cost_eggs = 3 →
  cost_pancakes = 2 →
  cost_mug_cocoa = 2 →
  num_mugs = 2 →
  tax = 1 →
  additional_pancakes = 2 →
  additional_mug = 2 →
  paid = 15 →
  paid - (cost_eggs + cost_pancakes + (num_mugs * cost_mug_cocoa) + tax + additional_pancakes + additional_mug) = 1 :=
by
  intros cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid
  sorry

end NUMINAMATH_GPT_change_from_fifteen_dollars_l1333_133344


namespace NUMINAMATH_GPT_four_digit_numbers_proof_l1333_133390

noncomputable def four_digit_numbers_total : ℕ := 9000
noncomputable def two_digit_numbers_total : ℕ := 90
noncomputable def max_distinct_products : ℕ := 4095
noncomputable def cannot_be_expressed_as_product : ℕ := four_digit_numbers_total - max_distinct_products

theorem four_digit_numbers_proof :
  cannot_be_expressed_as_product = 4905 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_numbers_proof_l1333_133390


namespace NUMINAMATH_GPT_area_of_woods_l1333_133383

def width := 8 -- the width in miles
def length := 3 -- the length in miles
def area (w : Nat) (l : Nat) : Nat := w * l -- the area function for a rectangle

theorem area_of_woods : area width length = 24 := by
  sorry

end NUMINAMATH_GPT_area_of_woods_l1333_133383


namespace NUMINAMATH_GPT_factorize_expression_l1333_133336

variable {a b : ℕ}

theorem factorize_expression (a b : ℕ) : 9 * a - 6 * b = 3 * (3 * a - 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1333_133336


namespace NUMINAMATH_GPT_jerry_original_butterflies_l1333_133316

/-- Define the number of butterflies Jerry originally had -/
def original_butterflies (let_go : ℕ) (now_has : ℕ) : ℕ := let_go + now_has

/-- Given conditions -/
def let_go : ℕ := 11
def now_has : ℕ := 82

/-- Theorem to prove the number of butterflies Jerry originally had -/
theorem jerry_original_butterflies : original_butterflies let_go now_has = 93 :=
by
  sorry

end NUMINAMATH_GPT_jerry_original_butterflies_l1333_133316


namespace NUMINAMATH_GPT_geometric_prog_y_90_common_ratio_l1333_133367

theorem geometric_prog_y_90_common_ratio :
  ∀ (y : ℝ), y = 90 → ∃ r : ℝ, r = (90 + y) / (30 + y) ∧ r = (180 + y) / (90 + y) ∧ r = 3 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_geometric_prog_y_90_common_ratio_l1333_133367


namespace NUMINAMATH_GPT_cost_price_of_radio_l1333_133340

theorem cost_price_of_radio (C : ℝ) 
  (overhead_expenses : ℝ := 20) 
  (selling_price : ℝ := 300) 
  (profit_percent : ℝ := 22.448979591836732) :
  C = 228.57 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_radio_l1333_133340


namespace NUMINAMATH_GPT_find_k_l1333_133398

theorem find_k (Z K : ℤ) (h1 : 2000 < Z) (h2 : Z < 3000) (h3 : K > 1) (h4 : Z = K * K^2) (h5 : ∃ n : ℤ, n^3 = Z) : K = 13 :=
by
-- Solution omitted
sorry

end NUMINAMATH_GPT_find_k_l1333_133398


namespace NUMINAMATH_GPT_area_PQR_l1333_133319

-- Define the coordinates of the points
def P : ℝ × ℝ := (-3, 4)
def Q : ℝ × ℝ := (4, 9)
def R : ℝ × ℝ := (5, -3)

-- Function to calculate the area of a triangle given three points
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

-- Statement to prove the area of triangle PQR is 44.5
theorem area_PQR : area_of_triangle P Q R = 44.5 := sorry

end NUMINAMATH_GPT_area_PQR_l1333_133319


namespace NUMINAMATH_GPT_handshake_count_l1333_133332

-- Define the conditions
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_people : ℕ := num_companies * reps_per_company
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

-- Define the theorem to prove
theorem handshake_count : 
  total_people * (total_people - 1 - (reps_per_company - 1)) / 2 = 160 := 
by
  -- Here should be the proof, but it is omitted
  sorry

end NUMINAMATH_GPT_handshake_count_l1333_133332


namespace NUMINAMATH_GPT_equivalent_conditions_l1333_133349

open Real

theorem equivalent_conditions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / x + 1 / y + 1 / z ≤ 1) ↔
  (∀ a b c d : ℝ, a + b + c > d → a^2 * x + b^2 * y + c^2 * z > d^2) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_conditions_l1333_133349


namespace NUMINAMATH_GPT_part_I_solution_part_II_solution_l1333_133359

-- Part (I) proof problem: Prove the solution set for a specific inequality
theorem part_I_solution (x : ℝ) : -6 < x ∧ x < 10 / 3 → |2 * x - 2| + x + 1 < 9 :=
by
  sorry

-- Part (II) proof problem: Prove the range of 'a' for a given inequality to hold
theorem part_II_solution (a : ℝ) : (-3 ≤ a ∧ a ≤ 17 / 3) →
  (∀ x : ℝ, x ≥ 2 → |a * x + a - 4| + x + 1 ≤ (x + 2)^2) :=
by
  sorry

end NUMINAMATH_GPT_part_I_solution_part_II_solution_l1333_133359


namespace NUMINAMATH_GPT_centroid_plane_distance_l1333_133353

theorem centroid_plane_distance :
  ∀ (α β γ : ℝ) (p q r : ℝ),
    (1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4) →
    (p = α / 3) →
    (q = β / 3) →
    (r = γ / 3) →
    (1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 4) :=
by sorry

end NUMINAMATH_GPT_centroid_plane_distance_l1333_133353


namespace NUMINAMATH_GPT_inequalities_for_m_gt_n_l1333_133334

open Real

theorem inequalities_for_m_gt_n (m n : ℕ) (hmn : m > n) : 
  (1 + 1 / (m : ℝ)) ^ m > (1 + 1 / (n : ℝ)) ^ n ∧ 
  (1 + 1 / (m : ℝ)) ^ (m + 1) < (1 + 1 / (n : ℝ)) ^ (n + 1) := 
by
  sorry

end NUMINAMATH_GPT_inequalities_for_m_gt_n_l1333_133334


namespace NUMINAMATH_GPT_omega_in_abc_l1333_133330

variables {R : Type*}
variables [LinearOrderedField R]
variables {a b c ω x y z : R} 

theorem omega_in_abc 
  (distinct_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ω ≠ a ∧ ω ≠ b ∧ ω ≠ c)
  (h1 : x + y + z = 1)
  (h2 : a^2 * x + b^2 * y + c^2 * z = ω^2)
  (h3 : a^3 * x + b^3 * y + c^3 * z = ω^3)
  (h4 : a^4 * x + b^4 * y + c^4 * z = ω^4):
  ω = a ∨ ω = b ∨ ω = c :=
sorry

end NUMINAMATH_GPT_omega_in_abc_l1333_133330


namespace NUMINAMATH_GPT_polynomial_inequality_solution_l1333_133318

theorem polynomial_inequality_solution (x : ℝ) :
  x^4 + x^3 - 10 * x^2 + 25 * x > 0 ↔ x > 0 :=
sorry

end NUMINAMATH_GPT_polynomial_inequality_solution_l1333_133318


namespace NUMINAMATH_GPT_part1_l1333_133368

def A (a : ℝ) : Set ℝ := {x | 2 * a - 3 < x ∧ x < a + 1}
def B : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem part1 (a : ℝ) (h : a = 0) : A a ∩ B = {x | -1 < x ∧ x < 1} :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_part1_l1333_133368


namespace NUMINAMATH_GPT_flag_arrangement_remainder_l1333_133351

theorem flag_arrangement_remainder :
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  M % div = 441 := 
by
  -- Definitions
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  -- Proof
  sorry

end NUMINAMATH_GPT_flag_arrangement_remainder_l1333_133351


namespace NUMINAMATH_GPT_find_expression_for_a_n_l1333_133365

noncomputable def a_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2^n

theorem find_expression_for_a_n (a : ℕ → ℕ) (h : a_sequence a) (initial : a 1 = 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_for_a_n_l1333_133365


namespace NUMINAMATH_GPT_overall_percentage_good_fruits_l1333_133362

theorem overall_percentage_good_fruits
  (oranges_bought : ℕ)
  (bananas_bought : ℕ)
  (apples_bought : ℕ)
  (pears_bought : ℕ)
  (oranges_rotten_percent : ℝ)
  (bananas_rotten_percent : ℝ)
  (apples_rotten_percent : ℝ)
  (pears_rotten_percent : ℝ)
  (h_oranges : oranges_bought = 600)
  (h_bananas : bananas_bought = 400)
  (h_apples : apples_bought = 800)
  (h_pears : pears_bought = 200)
  (h_oranges_rotten : oranges_rotten_percent = 0.15)
  (h_bananas_rotten : bananas_rotten_percent = 0.03)
  (h_apples_rotten : apples_rotten_percent = 0.12)
  (h_pears_rotten : pears_rotten_percent = 0.25) :
  let total_fruits := oranges_bought + bananas_bought + apples_bought + pears_bought
  let rotten_oranges := oranges_rotten_percent * oranges_bought
  let rotten_bananas := bananas_rotten_percent * bananas_bought
  let rotten_apples := apples_rotten_percent * apples_bought
  let rotten_pears := pears_rotten_percent * pears_bought
  let good_oranges := oranges_bought - rotten_oranges
  let good_bananas := bananas_bought - rotten_bananas
  let good_apples := apples_bought - rotten_apples
  let good_pears := pears_bought - rotten_pears
  let total_good_fruits := good_oranges + good_bananas + good_apples + good_pears
  (total_good_fruits / total_fruits) * 100 = 87.6 :=
by
  sorry

end NUMINAMATH_GPT_overall_percentage_good_fruits_l1333_133362


namespace NUMINAMATH_GPT_vector_parallel_addition_l1333_133345

theorem vector_parallel_addition 
  (x : ℝ)
  (a : ℝ × ℝ := (2, 1))
  (b : ℝ × ℝ := (x, -2)) 
  (h_parallel : 2 / x = 1 / -2) :
  a + b = (-2, -1) := 
by
  -- While the proof is omitted, the statement is complete and correct.
  sorry

end NUMINAMATH_GPT_vector_parallel_addition_l1333_133345


namespace NUMINAMATH_GPT_sequence_ratio_l1333_133381

variable (a : ℕ → ℝ) -- Define the sequence a_n
variable (q : ℝ) (h_q : q > 0) -- q is the common ratio and it is positive

-- Define the conditions
axiom geom_seq_pos : ∀ n : ℕ, 0 < a n
axiom geom_seq_def : ∀ n : ℕ, a (n + 1) = q * a n
axiom arith_seq_def : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2

theorem sequence_ratio : (a 11 + a 13) / (a 8 + a 10) = 27 := 
by
  sorry

end NUMINAMATH_GPT_sequence_ratio_l1333_133381


namespace NUMINAMATH_GPT_bank_balance_after_2_years_l1333_133366

noncomputable def compound_interest (P₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P₀ * (1 + r)^n

theorem bank_balance_after_2_years :
  compound_interest 100 0.10 2 = 121 := 
  by
  sorry

end NUMINAMATH_GPT_bank_balance_after_2_years_l1333_133366


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1333_133372

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_arith : -a 5 + a 6 = 2 * a 4) :
  q = -1 ∨ q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1333_133372


namespace NUMINAMATH_GPT_real_estate_commission_l1333_133326

theorem real_estate_commission (commission_rate commission selling_price : ℝ) 
  (h1 : commission_rate = 0.06) 
  (h2 : commission = 8880) : 
  selling_price = 148000 :=
by
  sorry

end NUMINAMATH_GPT_real_estate_commission_l1333_133326


namespace NUMINAMATH_GPT_remainder_six_n_mod_four_l1333_133306

theorem remainder_six_n_mod_four (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by sorry

end NUMINAMATH_GPT_remainder_six_n_mod_four_l1333_133306


namespace NUMINAMATH_GPT_vasya_filling_time_l1333_133339

-- Definition of conditions
def hose_filling_time (x : ℝ) : Prop :=
  ∀ (first_hose_mult second_hose_mult : ℝ), 
    first_hose_mult = x ∧
    second_hose_mult = 5 * x ∧
    (5 * second_hose_mult - 5 * first_hose_mult) = 1

-- Conclusion
theorem vasya_filling_time (x : ℝ) (first_hose_mult second_hose_mult : ℝ) :
  hose_filling_time x → 25 * x = 1 * (60 + 15) := sorry

end NUMINAMATH_GPT_vasya_filling_time_l1333_133339


namespace NUMINAMATH_GPT_rectangle_y_value_l1333_133364

theorem rectangle_y_value
  (E : (ℝ × ℝ)) (F : (ℝ × ℝ)) (G : (ℝ × ℝ)) (H : (ℝ × ℝ))
  (hE : E = (0, 0)) (hF : F = (0, 5)) (hG : ∃ y : ℝ, G = (y, 5))
  (hH : ∃ y : ℝ, H = (y, 0)) (area : ℝ) (h_area : area = 35)
  (hy_pos : ∃ y : ℝ, y > 0)
  : ∃ y : ℝ, y = 7 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_y_value_l1333_133364


namespace NUMINAMATH_GPT_find_r_l1333_133386

theorem find_r (r : ℝ) (cone1_radius cone2_radius cone3_radius : ℝ) (sphere_radius : ℝ)
  (cone_height_eq : cone1_radius = 2 * r ∧ cone2_radius = 3 * r ∧ cone3_radius = 10 * r)
  (sphere_touch : sphere_radius = 2)
  (center_eq_dist : ∀ {P Q : ℝ}, dist P Q = 2 → dist Q r = 2) :
  r = 1 := 
sorry

end NUMINAMATH_GPT_find_r_l1333_133386


namespace NUMINAMATH_GPT_longest_side_of_garden_l1333_133312

theorem longest_side_of_garden (l w : ℝ) (h1 : 2 * l + 2 * w = 225) (h2 : l * w = 8 * 225) :
  l = 93.175 ∨ w = 93.175 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_of_garden_l1333_133312


namespace NUMINAMATH_GPT_inequality_neg_mul_l1333_133370

theorem inequality_neg_mul (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
    sorry

end NUMINAMATH_GPT_inequality_neg_mul_l1333_133370
