import Mathlib

namespace NUMINAMATH_GPT_value_of_a8_l143_14315

theorem value_of_a8 (a : ℕ → ℝ) :
  (1 + x) ^ 10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x) ^ 2 + a 3 * (1 - x) ^ 3 +
  a 4 * (1 - x) ^ 4 + a 5 * (1 - x) ^ 5 + a 6 * (1 - x) ^ 6 + a 7 * (1 - x) ^ 7 + 
  a 8 * (1 - x) ^ 8 + a 9 * (1 - x) ^ 9 + a 10 * (1 - x) ^ 10 → 
  a 8 = 180 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a8_l143_14315


namespace NUMINAMATH_GPT_parabola_translation_vertex_l143_14300

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Define the translation of the parabola
def translated_parabola (x : ℝ) : ℝ := (x + 3)^2 - 4*(x + 3) + 2 - 2 -- Adjust x + 3 for shift left and subtract 2 for shift down

-- The vertex coordinates function
def vertex_coords (f : ℝ → ℝ) (x_vertex : ℝ) : ℝ × ℝ := (x_vertex, f x_vertex)

-- Define the original vertex
def original_vertex : ℝ × ℝ := vertex_coords original_parabola 2

-- Define the translated vertex we expect
def expected_translated_vertex : ℝ × ℝ := vertex_coords translated_parabola (-1)

-- Statement of the problem
theorem parabola_translation_vertex :
  expected_translated_vertex = (-1, -4) :=
  sorry

end NUMINAMATH_GPT_parabola_translation_vertex_l143_14300


namespace NUMINAMATH_GPT_no_solution_a4_plus_6_eq_b3_mod_13_l143_14324

theorem no_solution_a4_plus_6_eq_b3_mod_13 :
  ¬ ∃ (a b : ℤ), (a^4 + 6) % 13 = b^3 % 13 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_a4_plus_6_eq_b3_mod_13_l143_14324


namespace NUMINAMATH_GPT_combined_weight_of_jake_and_sister_l143_14393

theorem combined_weight_of_jake_and_sister (j s : ℕ) (h1 : j = 188) (h2 : j - 8 = 2 * s) : j + s = 278 :=
sorry

end NUMINAMATH_GPT_combined_weight_of_jake_and_sister_l143_14393


namespace NUMINAMATH_GPT_cloth_gain_percentage_l143_14331

theorem cloth_gain_percentage 
  (x : ℝ) -- x represents the cost price of 1 meter of cloth
  (CP : ℝ := 30 * x) -- CP of 30 meters of cloth
  (profit : ℝ := 10 * x) -- profit from selling 30 meters of cloth
  (SP : ℝ := CP + profit) -- selling price of 30 meters of cloth
  (gain_percentage : ℝ := (profit / CP) * 100) : 
  gain_percentage = 33.33 := 
sorry

end NUMINAMATH_GPT_cloth_gain_percentage_l143_14331


namespace NUMINAMATH_GPT_percentage_A_of_B_l143_14367

variable {A B C D : ℝ}

theorem percentage_A_of_B (
  h1: A = 0.125 * C)
  (h2: B = 0.375 * D)
  (h3: D = 1.225 * C)
  (h4: C = 0.805 * B) :
  A = 0.100625 * B := by
  -- Sufficient proof steps would go here
  sorry

end NUMINAMATH_GPT_percentage_A_of_B_l143_14367


namespace NUMINAMATH_GPT_find_roots_of_parabola_l143_14399

-- Define the conditions given in the problem
variables (a b c : ℝ)
variable (a_nonzero : a ≠ 0)
variable (passes_through_1_0 : a * 1^2 + b * 1 + c = 0)
variable (axis_of_symmetry : -b / (2 * a) = -2)

-- Lean theorem statement
theorem find_roots_of_parabola (a b c : ℝ) (a_nonzero : a ≠ 0)
(passes_through_1_0 : a * 1^2 + b * 1 + c = 0) (axis_of_symmetry : -b / (2 * a) = -2) :
  (a * (-5)^2 + b * (-5) + c = 0) ∧ (a * 1^2 + b * 1 + c = 0) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_roots_of_parabola_l143_14399


namespace NUMINAMATH_GPT_factor_t_squared_minus_144_l143_14340

theorem factor_t_squared_minus_144 (t : ℝ) : 
  t ^ 2 - 144 = (t - 12) * (t + 12) := 
by 
  -- Here you would include the proof steps which are not needed for this task.
  sorry

end NUMINAMATH_GPT_factor_t_squared_minus_144_l143_14340


namespace NUMINAMATH_GPT_distance_to_school_l143_14375

def jerry_one_way_time : ℝ := 15  -- Jerry's one-way time in minutes
def carson_speed_mph : ℝ := 8  -- Carson's speed in miles per hour
def minutes_per_hour : ℝ := 60  -- Number of minutes in one hour

noncomputable def carson_speed_mpm : ℝ := carson_speed_mph / minutes_per_hour -- Carson's speed in miles per minute
def carson_one_way_time : ℝ := jerry_one_way_time -- Carson's one-way time is the same as Jerry's round trip time / 2

-- Prove that the distance to the school is 2 miles.
theorem distance_to_school : carson_speed_mpm * carson_one_way_time = 2 := by
  sorry

end NUMINAMATH_GPT_distance_to_school_l143_14375


namespace NUMINAMATH_GPT_supremum_neg_frac_l143_14343

noncomputable def supremum_expression (a b : ℝ) : ℝ :=
  - (1 / (2 * a) + 2 / b)

theorem supremum_neg_frac {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  ∃ M : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → supremum_expression x y ≤ M)
  ∧ (∀ N : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → supremum_expression x y ≤ N) → M ≤ N)
  ∧ M = -9 / 2 :=
sorry

end NUMINAMATH_GPT_supremum_neg_frac_l143_14343


namespace NUMINAMATH_GPT_time_to_reach_ticket_window_l143_14316

-- Define the conditions as per the problem
def rate_kit : ℕ := 2 -- feet per minute (rate)
def remaining_distance : ℕ := 210 -- feet

-- Goal: To prove the time required to reach the ticket window is 105 minutes
theorem time_to_reach_ticket_window : remaining_distance / rate_kit = 105 :=
by sorry

end NUMINAMATH_GPT_time_to_reach_ticket_window_l143_14316


namespace NUMINAMATH_GPT_fill_blanks_l143_14378

/-
Given the following conditions:
1. 20 * (x1 - 8) = 20
2. x2 / 2 + 17 = 20
3. 3 * x3 - 4 = 20
4. (x4 + 8) / 12 = y4
5. 4 * x5 = 20
6. 20 * (x6 - y6) = 100

Prove that:
1. x1 = 9
2. x2 = 6
3. x3 = 8
4. x4 = 4 and y4 = 1
5. x5 = 5
6. x6 = 7 and y6 = 2
-/
theorem fill_blanks (x1 x2 x3 x4 y4 x5 x6 y6 : ℕ) :
  20 * (x1 - 8) = 20 →
  x2 / 2 + 17 = 20 →
  3 * x3 - 4 = 20 →
  (x4 + 8) / 12 = y4 →
  4 * x5 = 20 →
  20 * (x6 - y6) = 100 →
  x1 = 9 ∧
  x2 = 6 ∧
  x3 = 8 ∧
  x4 = 4 ∧
  y4 = 1 ∧
  x5 = 5 ∧
  x6 = 7 ∧
  y6 = 2 :=
by
  sorry

end NUMINAMATH_GPT_fill_blanks_l143_14378


namespace NUMINAMATH_GPT_range_of_a_l143_14366

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l143_14366


namespace NUMINAMATH_GPT_father_current_age_l143_14348

theorem father_current_age (F S : ℕ) 
  (h₁ : F - 6 = 5 * (S - 6)) 
  (h₂ : (F + 6) + (S + 6) = 78) : 
  F = 51 := 
sorry

end NUMINAMATH_GPT_father_current_age_l143_14348


namespace NUMINAMATH_GPT_problem_l143_14384

theorem problem (x : ℝ) (h : x + 1 / x = 5) : x ^ 2 + (1 / x) ^ 2 = 23 := 
sorry

end NUMINAMATH_GPT_problem_l143_14384


namespace NUMINAMATH_GPT_max_diff_distance_l143_14346

def hyperbola_right_branch (x y : ℝ) : Prop := 
  (x^2 / 9) - (y^2 / 16) = 1 ∧ x > 0

def circle_1 (x y : ℝ) : Prop := 
  (x + 5)^2 + y^2 = 4

def circle_2 (x y : ℝ) : Prop := 
  (x - 5)^2 + y^2 = 1

theorem max_diff_distance 
  (P M N : ℝ × ℝ) 
  (hp : hyperbola_right_branch P.fst P.snd) 
  (hm : circle_1 M.fst M.snd) 
  (hn : circle_2 N.fst N.snd) :
  |dist P M - dist P N| ≤ 9 := 
sorry

end NUMINAMATH_GPT_max_diff_distance_l143_14346


namespace NUMINAMATH_GPT_tangent_segment_length_l143_14319

-- Setting up the necessary definitions and theorem.
def radius := 10
def seg1 := 4
def seg2 := 2

theorem tangent_segment_length :
  ∃ X : ℝ, X = 8 ∧
  (radius^2 = X^2 + ((X + seg1 + seg2) / 2)^2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_segment_length_l143_14319


namespace NUMINAMATH_GPT_largest_element_in_A_inter_B_l143_14363

def A : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2023 }
def B : Set ℕ := { n | ∃ k : ℤ, n = 3 * k + 2 ∧ n > 0 }

theorem largest_element_in_A_inter_B : ∃ x ∈ (A ∩ B), ∀ y ∈ (A ∩ B), y ≤ x ∧ x = 2021 := by
  sorry

end NUMINAMATH_GPT_largest_element_in_A_inter_B_l143_14363


namespace NUMINAMATH_GPT_maple_trees_remaining_l143_14395

-- Define the initial number of maple trees in the park
def initial_maple_trees : ℝ := 9.0

-- Define the number of maple trees that will be cut down
def cut_down_maple_trees : ℝ := 2.0

-- Define the expected number of maple trees left after cutting down
def remaining_maple_trees : ℝ := 7.0

-- Theorem to prove the remaining number of maple trees is correct
theorem maple_trees_remaining :
  initial_maple_trees - cut_down_maple_trees = remaining_maple_trees := by
  admit -- sorry can be used alternatively

end NUMINAMATH_GPT_maple_trees_remaining_l143_14395


namespace NUMINAMATH_GPT_vitamin_C_relationship_l143_14303

variables (A O G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + O + G = 275
def condition2 : Prop := 2 * A + 3 * O + 4 * G = 683

-- Rewrite the math proof problem statement
theorem vitamin_C_relationship (h1 : condition1 A O G) (h2 : condition2 A O G) : O + 2 * G = 133 :=
by {
  sorry
}

end NUMINAMATH_GPT_vitamin_C_relationship_l143_14303


namespace NUMINAMATH_GPT_sophia_daily_saving_l143_14383

theorem sophia_daily_saving (total_days : ℕ) (total_saving : ℝ) (h1 : total_days = 20) (h2 : total_saving = 0.20) : 
  (total_saving / total_days) = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_sophia_daily_saving_l143_14383


namespace NUMINAMATH_GPT_part1_part2_l143_14352
-- Importing the entire Mathlib library for required definitions

-- Define the sequence a_n with the conditions given in the problem
def a : ℕ → ℚ
| 0       => 1
| (n + 1) => a n / (2 * a n + 1)

-- Prove the given claims
theorem part1 (n : ℕ) : a n = (1 : ℚ) / (2 * n + 1) :=
sorry

def b (n : ℕ) : ℚ := a n * a (n + 1)

-- The sum of the first n terms of the sequence b_n is denoted as T_n
def T : ℕ → ℚ
| 0       => 0
| (n + 1) => T n + b n

-- Prove the given sum
theorem part2 (n : ℕ) : T n = (n : ℚ) / (2 * n + 1) :=
sorry

end NUMINAMATH_GPT_part1_part2_l143_14352


namespace NUMINAMATH_GPT_average_age_decrease_l143_14398

-- Define the conditions as given in the problem
def original_strength : ℕ := 12
def new_students : ℕ := 12

def original_avg_age : ℕ := 40
def new_students_avg_age : ℕ := 32

def decrease_in_avg_age (O N : ℕ) (OA NA : ℕ) : ℕ :=
  let total_original_age := O * OA
  let total_new_students_age := N * NA
  let total_students := O + N
  let new_avg_age := (total_original_age + total_new_students_age) / total_students
  OA - new_avg_age

theorem average_age_decrease :
  decrease_in_avg_age original_strength new_students original_avg_age new_students_avg_age = 4 :=
sorry

end NUMINAMATH_GPT_average_age_decrease_l143_14398


namespace NUMINAMATH_GPT_driver_net_rate_of_pay_l143_14347

theorem driver_net_rate_of_pay
  (hours : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℚ)
  (gas_cost_per_gallon : ℚ)
  (net_rate_of_pay : ℚ)
  (h1 : hours = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_cost_per_gallon = 2.50)
  (h6 : net_rate_of_pay = 25) :
  net_rate_of_pay = (hours * speed * pay_per_mile - (hours * speed / fuel_efficiency) * gas_cost_per_gallon) / hours := 
by sorry

end NUMINAMATH_GPT_driver_net_rate_of_pay_l143_14347


namespace NUMINAMATH_GPT_percentage_of_tip_is_25_l143_14354

-- Definitions of the costs
def cost_samosas : ℕ := 3 * 2
def cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2

-- Definition of total food cost
def total_food_cost : ℕ := cost_samosas + cost_pakoras + cost_mango_lassi

-- Definition of the total meal cost including tax
def total_meal_cost_with_tax : ℕ := 25

-- Definition of the tip
def tip : ℕ := total_meal_cost_with_tax - total_food_cost

-- Definition of the percentage of the tip
def percentage_tip : ℕ := (tip * 100) / total_food_cost

-- The theorem to be proved
theorem percentage_of_tip_is_25 :
  percentage_tip = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_tip_is_25_l143_14354


namespace NUMINAMATH_GPT_ratio_circle_to_triangle_area_l143_14349

theorem ratio_circle_to_triangle_area 
  (h d : ℝ) 
  (h_pos : 0 < h) 
  (d_pos : 0 < d) 
  (R : ℝ) 
  (R_def : R = h / 2) :
  (π * R^2) / (1/2 * h * d) = (π * h) / (2 * d) :=
by sorry

end NUMINAMATH_GPT_ratio_circle_to_triangle_area_l143_14349


namespace NUMINAMATH_GPT_prove_inequality_l143_14329

open Real

noncomputable def inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : Prop :=
  3 + (a + b + c) + (1/a + 1/b + 1/c) + (a/b + b/c + c/a) ≥ 
  3 * (a + 1) * (b + 1) * (c + 1) / (a * b * c + 1)

theorem prove_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  inequality a b c h1 h2 h3 := 
  sorry

end NUMINAMATH_GPT_prove_inequality_l143_14329


namespace NUMINAMATH_GPT_range_of_expression_l143_14359

theorem range_of_expression (x : ℝ) : (x + 2 ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_expression_l143_14359


namespace NUMINAMATH_GPT_sawyer_saw_octopuses_l143_14350

def number_of_legs := 40
def legs_per_octopus := 8

theorem sawyer_saw_octopuses : number_of_legs / legs_per_octopus = 5 := 
by
  sorry

end NUMINAMATH_GPT_sawyer_saw_octopuses_l143_14350


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l143_14338

theorem distance_between_parallel_lines :
  let A := 3
  let B := 2
  let C1 := -1
  let C2 := 1 / 2
  let d := |C2 - C1| / Real.sqrt (A^2 + B^2)
  d = 3 / Real.sqrt 13 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l143_14338


namespace NUMINAMATH_GPT_difference_of_squares_l143_14309

-- Definition of the constants a and b as given in the problem
def a := 502
def b := 498

theorem difference_of_squares : a^2 - b^2 = 4000 := by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l143_14309


namespace NUMINAMATH_GPT_lengths_of_trains_l143_14334

noncomputable def km_per_hour_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def length_of_train (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem lengths_of_trains (Va Vb : ℝ) : Va = 60 ∧ Vb < Va ∧ length_of_train (km_per_hour_to_m_per_s Va) 42 = (700 : ℝ) 
    → length_of_train (km_per_hour_to_m_per_s Vb * (42 / 56)) 56 = (700 : ℝ) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_lengths_of_trains_l143_14334


namespace NUMINAMATH_GPT_integer_solutions_system_inequalities_l143_14357

theorem integer_solutions_system_inequalities:
  {x : ℤ} → (2 * x - 1 < x + 1) → (1 - 2 * (x - 1) ≤ 3) → x = 0 ∨ x = 1 := 
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_integer_solutions_system_inequalities_l143_14357


namespace NUMINAMATH_GPT_find_B_l143_14304

theorem find_B (N : ℕ) (A B : ℕ) (H1 : N = 757000000 + A * 10000 + B * 1000 + 384) (H2 : N % 357 = 0) : B = 5 :=
sorry

end NUMINAMATH_GPT_find_B_l143_14304


namespace NUMINAMATH_GPT_prob_same_color_seven_red_and_five_green_l143_14345

noncomputable def probability_same_color (red_plat : ℕ) (green_plat : ℕ) : ℚ :=
  let total_plates := red_plat + green_plat
  let total_pairs := (total_plates.choose 2) -- total ways to select 2 plates
  let red_pairs := (red_plat.choose 2) -- ways to select 2 red plates
  let green_pairs := (green_plat.choose 2) -- ways to select 2 green plates
  (red_pairs + green_pairs) / total_pairs

theorem prob_same_color_seven_red_and_five_green :
  probability_same_color 7 5 = 31 / 66 :=
by
  sorry

end NUMINAMATH_GPT_prob_same_color_seven_red_and_five_green_l143_14345


namespace NUMINAMATH_GPT_barycentric_vector_identity_l143_14396

variables {A B C X : Type} [AddCommGroup X] [Module ℝ X]
variables (α β γ : ℝ) (A B C X : X)

-- Defining the barycentric coordinates condition
axiom barycentric_coords : α • A + β • B + γ • C = X

-- Additional condition that sum of coordinates is 1
axiom sum_coords : α + β + γ = 1

-- The theorem to prove
theorem barycentric_vector_identity :
  (X - A) = β • (B - A) + γ • (C - A) :=
sorry

end NUMINAMATH_GPT_barycentric_vector_identity_l143_14396


namespace NUMINAMATH_GPT_patio_total_tiles_l143_14388

theorem patio_total_tiles (s : ℕ) (red_tiles : ℕ) (h1 : s % 2 = 1) (h2 : red_tiles = 2 * s - 1) (h3 : red_tiles = 61) :
  s * s = 961 :=
by
  sorry

end NUMINAMATH_GPT_patio_total_tiles_l143_14388


namespace NUMINAMATH_GPT_arnel_kept_fifty_pencils_l143_14335

theorem arnel_kept_fifty_pencils
    (num_boxes : ℕ) (pencils_each_box : ℕ) (friends : ℕ) (pencils_each_friend : ℕ) (total_pencils : ℕ)
    (boxes_pencils : ℕ) (friends_pencils : ℕ) :
    num_boxes = 10 →
    pencils_each_box = 5 →
    friends = 5 →
    pencils_each_friend = 8 →
    friends_pencils = friends * pencils_each_friend →
    boxes_pencils = num_boxes * pencils_each_box →
    total_pencils = boxes_pencils + friends_pencils →
    (total_pencils - friends_pencils) = 50 :=
by
    sorry

end NUMINAMATH_GPT_arnel_kept_fifty_pencils_l143_14335


namespace NUMINAMATH_GPT_number_of_integer_terms_l143_14302

noncomputable def count_integer_terms_in_sequence (n : ℕ) (k : ℕ) (a : ℕ) : ℕ :=
  if h : a = k * 3 ^ n then n + 1 else 0

theorem number_of_integer_terms :
  count_integer_terms_in_sequence 5 (2^3 * 5) 9720 = 6 :=
by sorry

end NUMINAMATH_GPT_number_of_integer_terms_l143_14302


namespace NUMINAMATH_GPT_remainder_of_7_pow_4_div_100_l143_14308

theorem remainder_of_7_pow_4_div_100 :
  (7^4) % 100 = 1 := 
sorry

end NUMINAMATH_GPT_remainder_of_7_pow_4_div_100_l143_14308


namespace NUMINAMATH_GPT_max_expression_sum_l143_14386

open Real

theorem max_expression_sum :
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ 
  (2 * x^2 - 3 * x * y + 4 * y^2 = 15 ∧ 
  (3 * x^2 + 2 * x * y + y^2 = 50 * sqrt 3 + 65)) :=
sorry

#eval 65 + 50 + 3 + 1 -- this should output 119

end NUMINAMATH_GPT_max_expression_sum_l143_14386


namespace NUMINAMATH_GPT_difference_of_squares_l143_14342

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l143_14342


namespace NUMINAMATH_GPT_calculate_polygon_sides_l143_14314

-- Let n be the number of sides of the regular polygon with each exterior angle of 18 degrees
def regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : Prop :=
  exterior_angle = 18 ∧ n * exterior_angle = 360

theorem calculate_polygon_sides (n : ℕ) (exterior_angle : ℝ) :
  regular_polygon_sides n exterior_angle → n = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_calculate_polygon_sides_l143_14314


namespace NUMINAMATH_GPT_f_minimum_positive_period_and_max_value_l143_14392

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.cos x) + (1 + (Real.tan x)^2) * (Real.cos x)^2

theorem f_minimum_positive_period_and_max_value :
  (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ π) ∧ (∃ M, ∀ x : ℝ, f x ≤ M ∧ M = 3 / 2) := by
  sorry

end NUMINAMATH_GPT_f_minimum_positive_period_and_max_value_l143_14392


namespace NUMINAMATH_GPT_prime_divisors_of_1890_l143_14306

theorem prime_divisors_of_1890 : ∃ (S : Finset ℕ), (S.card = 4) ∧ (∀ p ∈ S, Nat.Prime p) ∧ 1890 = S.prod id :=
by
  sorry

end NUMINAMATH_GPT_prime_divisors_of_1890_l143_14306


namespace NUMINAMATH_GPT_area_of_inscribed_rectangle_l143_14397

theorem area_of_inscribed_rectangle (r : ℝ) (h : r = 6) (ratio : ℝ) (hr : ratio = 3 / 1) :
  ∃ (length width : ℝ), (width = 2 * r) ∧ (length = ratio * width) ∧ (length * width = 432) :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_rectangle_l143_14397


namespace NUMINAMATH_GPT_first_month_sale_eq_6435_l143_14318

theorem first_month_sale_eq_6435 (s2 s3 s4 s5 s6 : ℝ)
  (h2 : s2 = 6927) (h3 : s3 = 6855) (h4 : s4 = 7230) (h5 : s5 = 6562) (h6 : s6 = 7391)
  (avg : ℝ) (h_avg : avg = 6900) :
  let total_sales := 6 * avg
  let other_months_sales := s2 + s3 + s4 + s5 + s6
  let first_month_sale := total_sales - other_months_sales
  first_month_sale = 6435 :=
by
  sorry

end NUMINAMATH_GPT_first_month_sale_eq_6435_l143_14318


namespace NUMINAMATH_GPT_solve_problem_l143_14337

noncomputable def problem_statement : Prop :=
  (2015 : ℝ) / (2015^2 - 2016 * 2014) = 2015

theorem solve_problem : problem_statement := by
  -- Proof steps will be filled in here.
  sorry

end NUMINAMATH_GPT_solve_problem_l143_14337


namespace NUMINAMATH_GPT_sector_angle_l143_14312

theorem sector_angle (r α : ℝ) (h₁ : 2 * r + α * r = 4) (h₂ : (1 / 2) * α * r^2 = 1) : α = 2 :=
sorry

end NUMINAMATH_GPT_sector_angle_l143_14312


namespace NUMINAMATH_GPT_eel_species_count_l143_14333

theorem eel_species_count (sharks eels whales total : ℕ)
    (h_sharks : sharks = 35)
    (h_whales : whales = 5)
    (h_total : total = 55)
    (h_species_sum : sharks + eels + whales = total) : eels = 15 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_eel_species_count_l143_14333


namespace NUMINAMATH_GPT_lorelei_roses_l143_14379

theorem lorelei_roses :
  let red_flowers := 12
  let pink_flowers := 18
  let yellow_flowers := 20
  let orange_flowers := 8
  let lorelei_red := (50 / 100) * red_flowers
  let lorelei_pink := (50 / 100) * pink_flowers
  let lorelei_yellow := (25 / 100) * yellow_flowers
  let lorelei_orange := (25 / 100) * orange_flowers
  lorelei_red + lorelei_pink + lorelei_yellow + lorelei_orange = 22 :=
by
  sorry

end NUMINAMATH_GPT_lorelei_roses_l143_14379


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l143_14311

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) := sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l143_14311


namespace NUMINAMATH_GPT_angle_B_is_40_degrees_l143_14317

theorem angle_B_is_40_degrees (angle_A angle_B angle_C : ℝ)
  (h1 : angle_A = 3 * angle_B)
  (h2 : angle_B = 2 * angle_C)
  (triangle_sum : angle_A + angle_B + angle_C = 180) :
  angle_B = 40 :=
by
  sorry

end NUMINAMATH_GPT_angle_B_is_40_degrees_l143_14317


namespace NUMINAMATH_GPT_sugar_recipes_l143_14341

theorem sugar_recipes (container_sugar recipe_sugar : ℚ) 
  (h1 : container_sugar = 56 / 3) 
  (h2 : recipe_sugar = 3 / 2) :
  container_sugar / recipe_sugar = 112 / 9 := sorry

end NUMINAMATH_GPT_sugar_recipes_l143_14341


namespace NUMINAMATH_GPT_cricket_game_initial_overs_l143_14355

theorem cricket_game_initial_overs
    (run_rate_initial : ℝ)
    (run_rate_remaining : ℝ)
    (remaining_overs : ℕ)
    (target_score : ℝ)
    (initial_overs : ℕ) :
    run_rate_initial = 3.2 →
    run_rate_remaining = 5.25 →
    remaining_overs = 40 →
    target_score = 242 →
    initial_overs = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cricket_game_initial_overs_l143_14355


namespace NUMINAMATH_GPT_x_minus_p_eq_2_minus_2p_l143_14387

theorem x_minus_p_eq_2_minus_2p (x p : ℝ) (h1 : |x - 3| = p + 1) (h2 : x < 3) : x - p = 2 - 2 * p := 
sorry

end NUMINAMATH_GPT_x_minus_p_eq_2_minus_2p_l143_14387


namespace NUMINAMATH_GPT_integer_values_of_n_satisfy_inequality_l143_14307

theorem integer_values_of_n_satisfy_inequality :
  ∃ S : Finset ℤ, (∀ n ∈ S, -100 < n^3 ∧ n^3 < 100) ∧ S.card = 9 :=
by
  -- Sorry provides the placeholder for where the proof would go
  sorry

end NUMINAMATH_GPT_integer_values_of_n_satisfy_inequality_l143_14307


namespace NUMINAMATH_GPT_ashok_avg_first_five_l143_14353

-- Define the given conditions 
def avg (n : ℕ) (s : ℕ) : ℕ := s / n

def total_marks (average : ℕ) (num_subjects : ℕ) : ℕ := average * num_subjects

variables (avg_six_subjects : ℕ := 76)
variables (sixth_subject_marks : ℕ := 86)
variables (total_six_subjects : ℕ := total_marks avg_six_subjects 6)
variables (total_first_five_subjects : ℕ := total_six_subjects - sixth_subject_marks)
variables (avg_first_five_subjects : ℕ := avg 5 total_first_five_subjects)

-- State the theorem
theorem ashok_avg_first_five 
  (h1 : avg_six_subjects = 76)
  (h2 : sixth_subject_marks = 86)
  (h3 : avg_first_five_subjects = 74)
  : avg 5 (total_marks 76 6 - 86) = 74 := 
sorry

end NUMINAMATH_GPT_ashok_avg_first_five_l143_14353


namespace NUMINAMATH_GPT_square_is_six_l143_14390

def represents_digit (square triangle circle : ℕ) : Prop :=
  square < 10 ∧ triangle < 10 ∧ circle < 10 ∧
  square ≠ triangle ∧ square ≠ circle ∧ triangle ≠ circle

theorem square_is_six :
  ∃ (square triangle circle : ℕ), represents_digit square triangle circle ∧ triangle = 1 ∧ circle = 9 ∧ (square + triangle + 100 * 1 + 10 * 9) = 117 ∧ square = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_square_is_six_l143_14390


namespace NUMINAMATH_GPT_greatest_q_minus_r_l143_14385

theorem greatest_q_minus_r :
  ∃ q r : ℤ, q > 0 ∧ r > 0 ∧ 975 = 23 * q + r ∧ q - r = 33 := sorry

end NUMINAMATH_GPT_greatest_q_minus_r_l143_14385


namespace NUMINAMATH_GPT_submarine_rise_l143_14325

theorem submarine_rise (initial_depth final_depth : ℤ) (h_initial : initial_depth = -27) (h_final : final_depth = -18) :
  final_depth - initial_depth = 9 :=
by
  rw [h_initial, h_final]
  norm_num 

end NUMINAMATH_GPT_submarine_rise_l143_14325


namespace NUMINAMATH_GPT_ganesh_average_speed_l143_14310

variable (D : ℝ) -- the distance between towns X and Y

theorem ganesh_average_speed :
  let time_x_to_y := D / 43
  let time_y_to_x := D / 34
  let total_distance := 2 * D
  let total_time := time_x_to_y + time_y_to_x
  let avg_speed := total_distance / total_time
  avg_speed = 37.97 := by
    sorry

end NUMINAMATH_GPT_ganesh_average_speed_l143_14310


namespace NUMINAMATH_GPT_exists_positive_integers_for_equation_l143_14365

theorem exists_positive_integers_for_equation :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a^4 = b^3 + c^2 :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_integers_for_equation_l143_14365


namespace NUMINAMATH_GPT_find_general_term_l143_14305

theorem find_general_term (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2 * a n + n^2) :
  ∀ n, a n = 7 * 2^(n - 1) - n^2 - 2 * n - 3 :=
by
  sorry

end NUMINAMATH_GPT_find_general_term_l143_14305


namespace NUMINAMATH_GPT_percent_profit_l143_14327

theorem percent_profit (C S : ℝ) (h : 55 * C = 50 * S) : 
  100 * ((S - C) / C) = 10 :=
by
  sorry

end NUMINAMATH_GPT_percent_profit_l143_14327


namespace NUMINAMATH_GPT_bonnie_roark_wire_length_ratio_l143_14313

noncomputable def ratio_of_wire_lengths : ℚ :=
let bonnie_wire_per_piece := 8
let bonnie_pieces := 12
let bonnie_total_wire := bonnie_pieces * bonnie_wire_per_piece

let bonnie_side := bonnie_wire_per_piece
let bonnie_volume := bonnie_side^3

let roark_side := 2
let roark_volume := roark_side^3
let roark_cubes := bonnie_volume / roark_volume

let roark_wire_per_piece := 2
let roark_pieces_per_cube := 12
let roark_wire_per_cube := roark_pieces_per_cube * roark_wire_per_piece
let roark_total_wire := roark_cubes * roark_wire_per_cube

let ratio := bonnie_total_wire / roark_total_wire
ratio 

theorem bonnie_roark_wire_length_ratio :
  ratio_of_wire_lengths = (1 : ℚ) / 16 := 
sorry

end NUMINAMATH_GPT_bonnie_roark_wire_length_ratio_l143_14313


namespace NUMINAMATH_GPT_eval_expression_correct_l143_14339

def eval_expression : ℤ :=
  -(-1) + abs (-1)

theorem eval_expression_correct : eval_expression = 2 :=
  by
    sorry

end NUMINAMATH_GPT_eval_expression_correct_l143_14339


namespace NUMINAMATH_GPT_evaluate_expression_l143_14381

theorem evaluate_expression : (2^(2 + 1) - 4 * (2 - 1)^2)^2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l143_14381


namespace NUMINAMATH_GPT_negation_of_P_l143_14330

-- Define the proposition P
def P : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

-- Define the negation of P
def not_P : Prop := ∀ x : ℝ, Real.exp x - x - 1 > 0

-- The theorem statement
theorem negation_of_P : ¬ P = not_P :=
by
  sorry

end NUMINAMATH_GPT_negation_of_P_l143_14330


namespace NUMINAMATH_GPT_possible_value_of_b_l143_14391

theorem possible_value_of_b (a b : ℕ) (H1 : b ∣ (5 * a - 1)) (H2 : b ∣ (a - 10)) (H3 : ¬ b ∣ (3 * a + 5)) : 
  b = 49 :=
sorry

end NUMINAMATH_GPT_possible_value_of_b_l143_14391


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l143_14394

theorem sufficient_not_necessary_condition (x : ℝ) : (x < -1 → x^2 - 1 > 0) ∧ (x^2 - 1 > 0 → x < -1 ∨ x > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l143_14394


namespace NUMINAMATH_GPT_complex_multiplication_l143_14371

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 - 2 * i) = 2 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l143_14371


namespace NUMINAMATH_GPT_max_gcd_13n_plus_4_8n_plus_3_l143_14361

theorem max_gcd_13n_plus_4_8n_plus_3 : ∃ n : ℕ, n > 0 ∧ Int.gcd (13 * n + 4) (8 * n + 3) = 11 := 
sorry

end NUMINAMATH_GPT_max_gcd_13n_plus_4_8n_plus_3_l143_14361


namespace NUMINAMATH_GPT_percent_of_a_l143_14369

theorem percent_of_a (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10 / 3) * a :=
sorry

end NUMINAMATH_GPT_percent_of_a_l143_14369


namespace NUMINAMATH_GPT_inconsistent_proportion_l143_14351

theorem inconsistent_proportion (a b : ℝ) (h1 : 3 * a = 5 * b) (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (a / b = 3 / 5) :=
sorry

end NUMINAMATH_GPT_inconsistent_proportion_l143_14351


namespace NUMINAMATH_GPT_find_k_l143_14320

theorem find_k (x₁ x₂ k : ℝ) (hx : x₁ + x₂ = 3) (h_prod : x₁ * x₂ = k) (h_cond : x₁ * x₂ + 2 * x₁ + 2 * x₂ = 1) : k = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l143_14320


namespace NUMINAMATH_GPT_single_train_car_passenger_count_l143_14360

theorem single_train_car_passenger_count (P : ℕ) 
  (h1 : ∀ (plane_capacity train_capacity : ℕ), plane_capacity = 366 →
    train_capacity = 16 * P →
      (train_capacity = (2 * plane_capacity) + 228)) : 
  P = 60 :=
by
  sorry

end NUMINAMATH_GPT_single_train_car_passenger_count_l143_14360


namespace NUMINAMATH_GPT_description_of_T_l143_14380

-- Define the conditions
def T := { p : ℝ × ℝ | (∃ (c : ℝ), ((c = 5 ∨ c = p.1 + 3 ∨ c = p.2 - 6) ∧ (5 ≥ p.1 + 3) ∧ (5 ≥ p.2 - 6))) }

-- The main theorem
theorem description_of_T : 
  ∃ p : ℝ × ℝ, 
    (p = (2, 11)) ∧ 
    ∀ q ∈ T, 
      (q.fst = 2 ∧ q.snd ≤ 11) ∨ 
      (q.snd = 11 ∧ q.fst ≤ 2) ∨ 
      (q.snd = q.fst + 9 ∧ q.fst ≤ 2) :=
sorry

end NUMINAMATH_GPT_description_of_T_l143_14380


namespace NUMINAMATH_GPT_rectangle_width_l143_14323

theorem rectangle_width (w : ℝ) (h_length : w * 2 = l) (h_area : w * l = 50) : w = 5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l143_14323


namespace NUMINAMATH_GPT_bridge_length_l143_14389

theorem bridge_length (train_length : ℕ) (train_cross_bridge_time : ℕ) (train_cross_lamp_time : ℕ) (bridge_length : ℕ) :
  train_length = 600 →
  train_cross_bridge_time = 70 →
  train_cross_lamp_time = 20 →
  bridge_length = 1500 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_bridge_length_l143_14389


namespace NUMINAMATH_GPT_describe_S_l143_14301

def S : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) }

theorem describe_S :
  S = { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) } := 
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_describe_S_l143_14301


namespace NUMINAMATH_GPT_ball_radius_l143_14382

theorem ball_radius 
  (r_cylinder : ℝ) (h_rise : ℝ) (v_approx : ℝ)
  (r_cylinder_value : r_cylinder = 12)
  (h_rise_value : h_rise = 6.75)
  (v_approx_value : v_approx = 3053.628) :
  ∃ (r_ball : ℝ), (4 / 3) * Real.pi * r_ball^3 = v_approx ∧ r_ball = 9 := 
by 
  use 9
  sorry

end NUMINAMATH_GPT_ball_radius_l143_14382


namespace NUMINAMATH_GPT_sequence_properties_l143_14356

theorem sequence_properties (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, (a n)^2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0) :
  a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l143_14356


namespace NUMINAMATH_GPT_batsman_average_increase_l143_14344

theorem batsman_average_increase
  (A : ℤ)
  (h1 : (16 * A + 85) / 17 = 37) :
  37 - A = 3 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_increase_l143_14344


namespace NUMINAMATH_GPT_expected_coins_basilio_per_day_l143_14372

/-- The expected number of gold coins received by Basilio per day is 5.25 -/
def expected_coins_received_by_basilio (n : ℕ) (p : ℚ) : ℚ :=
  if n = 20 ∧ p = (1 / 2 : ℚ) then 5.25 else 0

theorem expected_coins_basilio_per_day :
  expected_coins_received_by_basilio 20 (1 / 2) = 5.25 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_expected_coins_basilio_per_day_l143_14372


namespace NUMINAMATH_GPT_simplify_expression_correct_l143_14326

noncomputable def simplify_expression (m n : ℝ) : ℝ :=
  ( (2 - n) / (n - 1) + 4 * ((m - 1) / (m - 2)) ) /
  ( n^2 * ((m - 1) / (n - 1)) + m^2 * ((2 - n) / (m - 2)) )

theorem simplify_expression_correct :
  simplify_expression (Real.rpow 400 (1/4)) (Real.sqrt 5) = (Real.sqrt 5) / 5 := 
sorry

end NUMINAMATH_GPT_simplify_expression_correct_l143_14326


namespace NUMINAMATH_GPT_average_age_of_club_l143_14358

theorem average_age_of_club (S_f S_m S_c : ℕ) (females males children : ℕ) (avg_females avg_males avg_children : ℕ) :
  females = 12 →
  males = 20 →
  children = 8 →
  avg_females = 28 →
  avg_males = 40 →
  avg_children = 10 →
  S_f = avg_females * females →
  S_m = avg_males * males →
  S_c = avg_children * children →
  (S_f + S_m + S_c) / (females + males + children) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_average_age_of_club_l143_14358


namespace NUMINAMATH_GPT_odd_expression_l143_14362

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

theorem odd_expression (p q : ℕ) (hp : is_odd p) (hq : is_odd q) : is_odd (2 * p * p - q) :=
by
  sorry

end NUMINAMATH_GPT_odd_expression_l143_14362


namespace NUMINAMATH_GPT_inequality_A_only_inequality_B_not_always_l143_14368

theorem inequality_A_only (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  a < c / 3 := 
sorry

theorem inequality_B_not_always (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  ¬ (b < c / 3) := 
sorry

end NUMINAMATH_GPT_inequality_A_only_inequality_B_not_always_l143_14368


namespace NUMINAMATH_GPT_sum_of_nonnegative_reals_l143_14370

theorem sum_of_nonnegative_reals (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 52) (h2 : a * b + b * c + c * a = 24) (h3 : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :
  a + b + c = 10 :=
sorry

end NUMINAMATH_GPT_sum_of_nonnegative_reals_l143_14370


namespace NUMINAMATH_GPT_find_x_y_l143_14364

theorem find_x_y (x y : ℝ) 
  (h1 : 3 * x = 0.75 * y)
  (h2 : x + y = 30) : x = 6 ∧ y = 24 := 
by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_find_x_y_l143_14364


namespace NUMINAMATH_GPT_solution_pairs_l143_14321

def equation (r p : ℤ) : Prop := r^2 - r * (p + 6) + p^2 + 5 * p + 6 = 0

theorem solution_pairs :
  ∀ (r p : ℤ),
    equation r p ↔ (r = 3 ∧ p = 1) ∨ (r = 4 ∧ p = 1) ∨ 
                    (r = 0 ∧ p = -2) ∨ (r = 4 ∧ p = -2) ∨ 
                    (r = 0 ∧ p = -3) ∨ (r = 3 ∧ p = -3) :=
by
  sorry

end NUMINAMATH_GPT_solution_pairs_l143_14321


namespace NUMINAMATH_GPT_area_of_inscribed_square_in_ellipse_l143_14374

open Real

noncomputable def inscribed_square_area : ℝ := 32

theorem area_of_inscribed_square_in_ellipse :
  ∀ (x y : ℝ),
  (x^2 / 4 + y^2 / 8 = 1) →
  (x = t - t) ∧ (y = (t + t) / sqrt 2) ∧ 
  (t = sqrt 4) → inscribed_square_area = 32 :=
  sorry

end NUMINAMATH_GPT_area_of_inscribed_square_in_ellipse_l143_14374


namespace NUMINAMATH_GPT_evaluate_expression_l143_14336

theorem evaluate_expression : 
  (1 / 10 : ℝ) + (2 / 20 : ℝ) - (3 / 60 : ℝ) = 0.15 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l143_14336


namespace NUMINAMATH_GPT_no_tiling_10x10_1x4_l143_14322

-- Define the problem using the given conditions
def checkerboard_tiling (n k : ℕ) : Prop :=
  ∃ t : ℕ, t * k = n * n ∧ n % k = 0

-- Prove that it is impossible to tile a 10x10 board with 1x4 tiles
theorem no_tiling_10x10_1x4 : ¬ checkerboard_tiling 10 4 :=
sorry

end NUMINAMATH_GPT_no_tiling_10x10_1x4_l143_14322


namespace NUMINAMATH_GPT_speed_of_man_upstream_l143_14332

def speed_of_man_in_still_water : ℝ := 32
def speed_of_man_downstream : ℝ := 39

theorem speed_of_man_upstream (V_m V_s : ℝ) :
  V_m = speed_of_man_in_still_water →
  V_m + V_s = speed_of_man_downstream →
  V_m - V_s = 25 :=
sorry

end NUMINAMATH_GPT_speed_of_man_upstream_l143_14332


namespace NUMINAMATH_GPT_min_value_expression_l143_14328

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + 1 / b) * (b + 4 / a) ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l143_14328


namespace NUMINAMATH_GPT_ways_to_place_7_balls_into_3_boxes_l143_14376

theorem ways_to_place_7_balls_into_3_boxes :
  (Nat.choose (7 + 3 - 1) (3 - 1)) = 36 :=
by
  sorry

end NUMINAMATH_GPT_ways_to_place_7_balls_into_3_boxes_l143_14376


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l143_14377

variable (x y : ℝ)

theorem ratio_of_x_to_y (h : 0.10 * x = 0.20 * y) : x / y = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l143_14377


namespace NUMINAMATH_GPT_actual_distance_traveled_l143_14373

-- Given conditions
variables (D : ℝ)
variables (H : D / 5 = (D + 20) / 15)

-- The proof problem statement
theorem actual_distance_traveled : D = 10 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l143_14373
