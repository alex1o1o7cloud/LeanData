import Mathlib

namespace residue_7_1234_mod_13_l264_264325

theorem residue_7_1234_mod_13 : (7^1234 : ℕ) % 13 = 4 :=
by
  have h1: 7 % 13 = 7 := rfl
  have h2: (7^2) % 13 = 10 := by norm_num
  have h3: (7^3) % 13 = 5 := by norm_num
  have h4: (7^4) % 13 = 9 := by norm_num
  have h5: (7^5) % 13 = 11 := by norm_num
  have h6: (7^6) % 13 = 12 := by norm_num
  have h7: (7^7) % 13 = 6 := by norm_num
  have h8: (7^8) % 13 = 3 := by norm_num
  have h9: (7^9) % 13 = 8 := by norm_num
  have h10: (7^10) % 13 = 4 := by norm_num
  have h11: (7^11) % 13 = 2 := by norm_num
  have h12: (7^12) % 13 = 1 := by norm_num
  sorry

end residue_7_1234_mod_13_l264_264325


namespace value_of_y_at_x_3_l264_264835

theorem value_of_y_at_x_3 (a b c : ℝ) (h : a * (-3 : ℝ)^5 + b * (-3)^3 + c * (-3) - 5 = 7) :
  a * (3 : ℝ)^5 + b * 3^3 + c * 3 - 5 = -17 :=
by
  sorry

end value_of_y_at_x_3_l264_264835


namespace seed_mixture_ryegrass_percent_l264_264441

theorem seed_mixture_ryegrass_percent (R : ℝ) :
  let X := 0.40
  let percentage_X_in_mixture := 1 / 3
  let percentage_Y_in_mixture := 2 / 3
  let final_ryegrass := 0.30
  (final_ryegrass = percentage_X_in_mixture * X + percentage_Y_in_mixture * R) → 
  R = 0.25 :=
by
  intros X percentage_X_in_mixture percentage_Y_in_mixture final_ryegrass H
  sorry

end seed_mixture_ryegrass_percent_l264_264441


namespace largest_y_coordinate_l264_264367

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49 + (y - 3)^2 / 25 = 0) → y = 3 :=
by
  intro h
  -- This is where the proofs steps would go if required.
  sorry

end largest_y_coordinate_l264_264367


namespace _l264_264829
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l264_264829


namespace arithmetic_sequence_a3_l264_264414

theorem arithmetic_sequence_a3 (a1 d : ℤ) (h : a1 + (a1 + d) + (a1 + 2 * d) + (a1 + 3 * d) + (a1 + 4 * d) = 20) : 
  a1 + 2 * d = 4 := by
  sorry

end arithmetic_sequence_a3_l264_264414


namespace length_GH_of_tetrahedron_l264_264458

noncomputable def tetrahedron_edge_length : ℕ := 24

theorem length_GH_of_tetrahedron
  (a b c d e f : ℕ)
  (h1 : a = 8) 
  (h2 : b = 16) 
  (h3 : c = 24) 
  (h4 : d = 35) 
  (h5 : e = 45) 
  (h6 : f = 55)
  (hEF : f = 55)
  (hEGF : e + b > f)
  (hEHG: e + c > a ∧ e + c > d) 
  (hFHG : b + c > a ∧ b + f > c ∧ c + a > b):
   tetrahedron_edge_length = c := 
sorry

end length_GH_of_tetrahedron_l264_264458


namespace target_avg_weekly_income_l264_264030

-- Define the weekly incomes for the past 5 weeks
def past_incomes : List ℤ := [406, 413, 420, 436, 395]

-- Define the average income over the next 2 weeks
def avg_income_next_two_weeks : ℤ := 365

-- Define the target average weekly income over the 7-week period
theorem target_avg_weekly_income : 
  ((past_incomes.sum + 2 * avg_income_next_two_weeks) / 7 = 400) :=
sorry

end target_avg_weekly_income_l264_264030


namespace intersection_of_function_and_inverse_l264_264174

theorem intersection_of_function_and_inverse (b a : Int) 
  (h₁ : a = 2 * (-4) + b) 
  (h₂ : a = (-4 - b) / 2) 
  : a = -4 :=
by
  sorry

end intersection_of_function_and_inverse_l264_264174


namespace total_juice_sold_3_days_l264_264260

def juice_sales_problem (V_L V_M V_S : ℕ) (d1 d2 d3 : ℕ) :=
  (d1 = V_L + 4 * V_M) ∧ 
  (d2 = 2 * V_L + 6 * V_S) ∧ 
  (d3 = V_L + 3 * V_M + 3 * V_S) ∧
  (d1 = d2) ∧
  (d2 = d3)

theorem total_juice_sold_3_days (V_L V_M V_S d1 d2 d3 : ℕ) 
  (h : juice_sales_problem V_L V_M V_S d1 d2 d3) 
  (h_VM : V_M = 3) 
  (h_VL : V_L = 6) : 
  3 * d1 = 54 := 
by 
  -- Proof will be filled in
  sorry

end total_juice_sold_3_days_l264_264260


namespace product_zero_when_a_is_three_l264_264057

theorem product_zero_when_a_is_three (a : ℤ) (h : a = 3) :
  (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  cases h
  sorry

end product_zero_when_a_is_three_l264_264057


namespace odds_against_C_winning_l264_264249

theorem odds_against_C_winning (prob_A: ℚ) (prob_B: ℚ) (prob_C: ℚ)
    (odds_A: prob_A = 1 / 5) (odds_B: prob_B = 2 / 5) 
    (total_prob: prob_A + prob_B + prob_C = 1):
    ((1 - prob_C) / prob_C) = 3 / 2 :=
by
  sorry

end odds_against_C_winning_l264_264249


namespace largest_square_side_l264_264739

theorem largest_square_side {m n : ℕ} (h1 : m = 72) (h2 : n = 90) : Nat.gcd m n = 18 :=
by
  sorry

end largest_square_side_l264_264739


namespace abs_diff_51st_terms_correct_l264_264462

-- Definition of initial conditions for sequences A and C
def seqA_first_term : ℤ := 40
def seqA_common_difference : ℤ := 8

def seqC_first_term : ℤ := 40
def seqC_common_difference : ℤ := -5

-- Definition of the nth term function for an arithmetic sequence
def nth_term (a₁ d n : ℤ) : ℤ := a₁ + d * (n - 1)

-- 51st term of sequence A
def a_51 : ℤ := nth_term seqA_first_term seqA_common_difference 51

-- 51st term of sequence C
def c_51 : ℤ := nth_term seqC_first_term seqC_common_difference 51

-- Absolute value of the difference
def abs_diff_51st_terms : ℤ := Int.natAbs (a_51 - c_51)

-- The theorem to be proved
theorem abs_diff_51st_terms_correct : abs_diff_51st_terms = 650 := by
  sorry

end abs_diff_51st_terms_correct_l264_264462


namespace valid_plantings_count_l264_264205

-- Define the grid structure
structure Grid3x3 :=
  (sections : Fin 9 → String)

noncomputable def crops := ["corn", "wheat", "soybeans", "potatoes", "oats"]

-- Define the adjacency relationships and restrictions as predicates
def adjacent (i j : Fin 9) : Prop :=
  (i = j + 1 ∧ j % 3 ≠ 2) ∨ (i = j - 1 ∧ i % 3 ≠ 2) ∨ (i = j + 3) ∨ (i = j - 3)

def valid_crop_planting (g : Grid3x3) : Prop :=
  ∀ i j, adjacent i j →
    (¬(g.sections i = "corn" ∧ g.sections j = "wheat") ∧ 
    ¬(g.sections i = "wheat" ∧ g.sections j = "corn") ∧
    ¬(g.sections i = "soybeans" ∧ g.sections j = "potatoes") ∧
    ¬(g.sections i = "potatoes" ∧ g.sections j = "soybeans") ∧
    ¬(g.sections i = "oats" ∧ g.sections j = "potatoes") ∧ 
    ¬(g.sections i = "potatoes" ∧ g.sections j = "oats"))

noncomputable def count_valid_plantings : Nat :=
  -- Placeholder for the actual count computing function
  sorry

theorem valid_plantings_count : count_valid_plantings = 5 :=
  sorry

end valid_plantings_count_l264_264205


namespace binom_20_6_l264_264937

theorem binom_20_6 : nat.choose 20 6 = 19380 := 
by 
  sorry

end binom_20_6_l264_264937


namespace square_fold_distance_l264_264927

noncomputable def distance_from_A (area : ℝ) (visible_equal : Bool) : ℝ :=
  if area = 18 ∧ visible_equal then 2 * Real.sqrt 6 else 0

theorem square_fold_distance (area : ℝ) (visible_equal : Bool) :
  area = 18 → visible_equal → distance_from_A area visible_equal = 2 * Real.sqrt 6 :=
by
  sorry

end square_fold_distance_l264_264927


namespace lemuel_total_points_l264_264262

theorem lemuel_total_points (two_point_shots : ℕ) (three_point_shots : ℕ) (points_from_two : ℕ) (points_from_three : ℕ) :
  two_point_shots = 7 →
  three_point_shots = 3 →
  points_from_two = 2 →
  points_from_three = 3 →
  two_point_shots * points_from_two + three_point_shots * points_from_three = 23 :=
by
  sorry

end lemuel_total_points_l264_264262


namespace johns_weekly_allowance_l264_264474

theorem johns_weekly_allowance
    (A : ℝ)
    (h1 : ∃ A, (4/15) * A = 0.64) :
    A = 2.40 :=
by
  sorry

end johns_weekly_allowance_l264_264474


namespace breadth_remains_the_same_l264_264302

variable (L B : ℝ)

theorem breadth_remains_the_same 
  (A : ℝ) (hA : A = L * B) 
  (L_half : ℝ) (hL_half : L_half = L / 2) 
  (B' : ℝ)
  (A' : ℝ) (hA' : A' = L_half * B') 
  (hA_change : A' = 0.5 * A) : 
  B' = B :=
  sorry

end breadth_remains_the_same_l264_264302


namespace rightmost_three_digits_of_5_pow_1994_l264_264186

theorem rightmost_three_digits_of_5_pow_1994 : (5 ^ 1994) % 1000 = 625 :=
by
  sorry

end rightmost_three_digits_of_5_pow_1994_l264_264186


namespace cookies_left_over_l264_264356

def abigail_cookies : Nat := 53
def beatrice_cookies : Nat := 65
def carson_cookies : Nat := 26
def pack_size : Nat := 10

theorem cookies_left_over : (abigail_cookies + beatrice_cookies + carson_cookies) % pack_size = 4 := 
by
  sorry

end cookies_left_over_l264_264356


namespace original_cube_volume_l264_264884

theorem original_cube_volume (a : ℕ) (V_cube V_new : ℕ)
  (h1 : V_cube = a^3)
  (h2 : V_new = (a + 2) * a * (a - 2))
  (h3 : V_cube = V_new + 24) :
  V_cube = 216 :=
by
  sorry

end original_cube_volume_l264_264884


namespace base_r_representation_26_eq_32_l264_264668

theorem base_r_representation_26_eq_32 (r : ℕ) : 
  26 = 3 * r + 6 → r = 8 :=
by
  sorry

end base_r_representation_26_eq_32_l264_264668


namespace slower_train_crosses_faster_in_36_seconds_l264_264334

-- Define the conditions of the problem
def speed_fast_train_kmph : ℚ := 110
def speed_slow_train_kmph : ℚ := 90
def length_fast_train_km : ℚ := 1.10
def length_slow_train_km : ℚ := 0.90

-- Convert speeds to m/s
def speed_fast_train_mps : ℚ := speed_fast_train_kmph * (1000 / 3600)
def speed_slow_train_mps : ℚ := speed_slow_train_kmph * (1000 / 3600)

-- Relative speed when moving in opposite directions
def relative_speed_mps : ℚ := speed_fast_train_mps + speed_slow_train_mps

-- Convert lengths to meters
def length_fast_train_m : ℚ := length_fast_train_km * 1000
def length_slow_train_m : ℚ := length_slow_train_km * 1000

-- Combined length of both trains in meters
def combined_length_m : ℚ := length_fast_train_m + length_slow_train_m

-- Time taken for the slower train to cross the faster train
def crossing_time : ℚ := combined_length_m / relative_speed_mps

theorem slower_train_crosses_faster_in_36_seconds :
  crossing_time = 36 := by
  sorry

end slower_train_crosses_faster_in_36_seconds_l264_264334


namespace population_ratios_l264_264364

variable (P_X P_Y P_Z : Nat)

theorem population_ratios
  (h1 : P_Y = 2 * P_Z)
  (h2 : P_X = 10 * P_Z) : P_X / P_Y = 5 := by
  sorry

end population_ratios_l264_264364


namespace solve_fractional_equation_l264_264146

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l264_264146


namespace range_of_a_l264_264975

noncomputable def parabola_locus (x : ℝ) : ℝ := x^2 / 4

def angle_sum_property (a k : ℝ) : Prop :=
  2 * a * k^2 + 2 * k + a = 0

def discriminant_nonnegative (a : ℝ) : Prop :=
  4 - 8 * a^2 ≥ 0

theorem range_of_a (a : ℝ) :
  (- (Real.sqrt 2) / 2) ≤ a ∧ a ≤ (Real.sqrt 2) / 2 :=
  sorry

end range_of_a_l264_264975


namespace tangent_line_at_0_g_nonpositive_f_relationship_l264_264076

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x - 1
noncomputable def g (x : ℝ) : ℝ := (fun x => Real.exp x * (Real.cos x - Real.sin x) - 1) x

theorem tangent_line_at_0 : (∀ x, f x - 0 * x = 0) := sorry
theorem g_nonpositive (x : ℝ) (hx : 0 ≤ x ∧ x < Real.pi) : g x ≤ 0 := sorry
theorem f_relationship (m n : ℝ) (hm : 0 < m ∧ m < Real.pi / 2) (hn : 0 < n ∧ n < Real.pi / 2) : 
  f (m + n) - f m < f n := sorry

end tangent_line_at_0_g_nonpositive_f_relationship_l264_264076


namespace number_of_dozens_l264_264887

theorem number_of_dozens (x : Nat) (h : x = 16 * (3 * 4)) : x / 12 = 16 :=
by
  sorry

end number_of_dozens_l264_264887


namespace amaya_movie_watching_time_l264_264506

theorem amaya_movie_watching_time :
  let uninterrupted_time_1 := 35
  let uninterrupted_time_2 := 45
  let uninterrupted_time_3 := 20
  let rewind_time_1 := 5
  let rewind_time_2 := 15
  let total_uninterrupted := uninterrupted_time_1 + uninterrupted_time_2 + uninterrupted_time_3
  let total_rewind := rewind_time_1 + rewind_time_2
  let total_time := total_uninterrupted + total_rewind
  total_time = 120 := by
  sorry

end amaya_movie_watching_time_l264_264506


namespace correct_order_of_operations_l264_264909

def order_of_operations (e : String) : String :=
  if e = "38 * 50 - 25 / 5" then
    "multiplication, division, subtraction"
  else
    "unknown"

theorem correct_order_of_operations :
  order_of_operations "38 * 50 - 25 / 5" = "multiplication, division, subtraction" :=
by
  sorry

end correct_order_of_operations_l264_264909


namespace fan_airflow_in_one_week_l264_264642

-- Define the conditions
def fan_airflow_per_second : ℕ := 10
def fan_working_minutes_per_day : ℕ := 10
def seconds_per_minute : ℕ := 60
def days_per_week : ℕ := 7

-- Define the proof problem
theorem fan_airflow_in_one_week : (fan_airflow_per_second * fan_working_minutes_per_day * seconds_per_minute * days_per_week = 42000) := 
by sorry

end fan_airflow_in_one_week_l264_264642


namespace jars_water_fraction_l264_264230

theorem jars_water_fraction (S L W : ℝ) (h1 : W = 1/6 * S) (h2 : W = 1/5 * L) : 
  (2 * W / L) = 2 / 5 :=
by
  -- We are only stating the theorem here, not proving it.
  sorry

end jars_water_fraction_l264_264230


namespace ice_cream_orders_l264_264437

variables (V C S M O T : ℕ)

theorem ice_cream_orders :
  (V = 56) ∧ (C = 28) ∧ (S = 70) ∧ (M = 42) ∧ (O = 84) ↔
  (V = 2 * C) ∧
  (S = 25 * T / 100) ∧
  (M = 15 * T / 100) ∧
  (T = 280) ∧
  (V = 20 * T / 100) ∧
  (V + C + S + M + O = T) :=
by
  sorry

end ice_cream_orders_l264_264437


namespace circle_polar_equation_l264_264074

-- Definitions and conditions
def circle_equation_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * y = 0

def polar_coordinates (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem to be proven
theorem circle_polar_equation (ρ θ : ℝ) :
  (∀ x y : ℝ, circle_equation_cartesian x y → 
  polar_coordinates ρ θ x y) → ρ = 2 * Real.sin θ :=
by
  -- This is a placeholder for the proof
  sorry

end circle_polar_equation_l264_264074


namespace statement2_statement3_l264_264383

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Conditions for the statements
axiom cond1 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = q ∧ f a b c q = p
axiom cond2 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = f a b c q
axiom cond3 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c (p + q) = c

-- Statement 2 correctness
theorem statement2 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c p = f a b c q) : 
  f a b c (p + q) = c :=
sorry

-- Statement 3 correctness
theorem statement3 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c (p + q) = c) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end statement2_statement3_l264_264383


namespace grover_total_profit_l264_264981

-- Definitions based on conditions
def original_price : ℝ := 10
def discount_first_box : ℝ := 0.20
def discount_second_box : ℝ := 0.30
def discount_third_box : ℝ := 0.40
def packs_first_box : ℕ := 20
def packs_second_box : ℕ := 30
def packs_third_box : ℕ := 40
def masks_per_pack : ℕ := 5
def price_per_mask_first_box : ℝ := 0.75
def price_per_mask_second_box : ℝ := 0.85
def price_per_mask_third_box : ℝ := 0.95

-- Computations
def cost_first_box := original_price - (discount_first_box * original_price)
def cost_second_box := original_price - (discount_second_box * original_price)
def cost_third_box := original_price - (discount_third_box * original_price)

def total_cost := cost_first_box + cost_second_box + cost_third_box

def revenue_first_box := packs_first_box * masks_per_pack * price_per_mask_first_box
def revenue_second_box := packs_second_box * masks_per_pack * price_per_mask_second_box
def revenue_third_box := packs_third_box * masks_per_pack * price_per_mask_third_box

def total_revenue := revenue_first_box + revenue_second_box + revenue_third_box

def total_profit := total_revenue - total_cost

-- Proof statement
theorem grover_total_profit : total_profit = 371.5 := by
  sorry

end grover_total_profit_l264_264981


namespace tangent_lines_to_circle_l264_264067

noncomputable theory
open Real

def circle_eqn (x y k : ℝ) : ℝ := x^2 + y^2 + k * x + 2 * y + k^2 - 15

theorem tangent_lines_to_circle :
  {k : ℝ | ∃ (x y : ℝ), circle_eqn x y k = 0} ∩
  {k : ℝ | ∀ (x y : ℝ), circle_eqn x 2 k > 0 → x ≠ 1 ∨ y ≠ 2} = 
  {k : ℝ | - (8 * sqrt 3 / 3) < k ∧ k < -3} ∪ {k : ℝ | 2 < k ∧ k < 8 * sqrt 3 / 3} :=
sorry

end tangent_lines_to_circle_l264_264067


namespace value_of_expression_l264_264628

variables {a b c d e f : ℝ}

theorem value_of_expression
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 / 4 :=
sorry

end value_of_expression_l264_264628


namespace cost_of_bag_l264_264602

variable (cost_per_bag : ℝ)
variable (chips_per_bag : ℕ := 24)
variable (calories_per_chip : ℕ := 10)
variable (total_calories : ℕ := 480)
variable (total_cost : ℝ := 4)

theorem cost_of_bag :
  (chips_per_bag * (total_calories / calories_per_chip / chips_per_bag) = (total_calories / calories_per_chip)) →
  (total_cost / (total_calories / (calories_per_chip * chips_per_bag))) = 2 :=
by
  sorry

end cost_of_bag_l264_264602


namespace original_number_l264_264844

theorem original_number (x : ℝ) (h1 : 74 * x = 19732) : x = 267 := by
  sorry

end original_number_l264_264844


namespace log_negative_l264_264250

open Real

theorem log_negative (a : ℝ) (h : a > 0) : log (-a) = log a := sorry

end log_negative_l264_264250


namespace solve_system_of_equations_l264_264442

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end solve_system_of_equations_l264_264442


namespace factorization_divisibility_l264_264815

theorem factorization_divisibility (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end factorization_divisibility_l264_264815


namespace number_of_tiles_per_row_l264_264290

-- Define the conditions
def area (a : ℝ) : ℝ := a * a
def side_length (area : ℝ) : ℝ := real.sqrt area
def feet_to_inches (feet : ℝ) : ℝ := feet * 12
def tiles_per_row (room_length_inches : ℝ) (tile_size_inches : ℝ) : ℕ := 
  int.to_nat ⟨room_length_inches / tile_size_inches, by sorry⟩

-- Given constants in the problem
def area_of_room : ℝ := 256
def tile_size : ℝ := 8

-- Derived lengths
def length_of_side := side_length area_of_room
def length_of_side_in_inches := feet_to_inches length_of_side

-- The theorem to prove
theorem number_of_tiles_per_row : tiles_per_row length_of_side_in_inches tile_size = 24 :=
sorry

end number_of_tiles_per_row_l264_264290


namespace amaya_movie_watching_time_l264_264507

theorem amaya_movie_watching_time :
  let uninterrupted_time_1 := 35
  let uninterrupted_time_2 := 45
  let uninterrupted_time_3 := 20
  let rewind_time_1 := 5
  let rewind_time_2 := 15
  let total_uninterrupted := uninterrupted_time_1 + uninterrupted_time_2 + uninterrupted_time_3
  let total_rewind := rewind_time_1 + rewind_time_2
  let total_time := total_uninterrupted + total_rewind
  total_time = 120 := by
  sorry

end amaya_movie_watching_time_l264_264507


namespace ben_david_bagel_cost_l264_264240

theorem ben_david_bagel_cost (B D : ℝ)
  (h1 : D = 0.5 * B)
  (h2 : B = D + 16) :
  B + D = 48 := 
sorry

end ben_david_bagel_cost_l264_264240


namespace range_of_m_l264_264980

noncomputable def p (m : ℝ) : Prop :=
  (m > 2)

noncomputable def q (m : ℝ) : Prop :=
  (m > 1)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l264_264980


namespace value_of_b7b9_l264_264100

-- Define arithmetic sequence and geometric sequence with given conditions
variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- The given conditions in the problem
def a_seq_arithmetic (a : ℕ → ℝ) := ∀ n, a n = a 1 + (n - 1) • (a 2 - a 1)
def b_seq_geometric (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n, b (n + 1) = r * b n
def given_condition (a : ℕ → ℝ) := 2 * a 5 - (a 8)^2 + 2 * a 11 = 0
def b8_eq_a8 (a b : ℕ → ℝ) := b 8 = a 8

-- The statement to prove
theorem value_of_b7b9 : a_seq_arithmetic a → b_seq_geometric b → given_condition a → b8_eq_a8 a b → b 7 * b 9 = 4 := by
  intros a_arith b_geom cond b8a8
  sorry

end value_of_b7b9_l264_264100


namespace inscribed_sphere_radius_l264_264830

theorem inscribed_sphere_radius (h1 h2 h3 h4 : ℝ) (S1 S2 S3 S4 V : ℝ)
  (h1_ge : h1 ≥ 1) (h2_ge : h2 ≥ 1) (h3_ge : h3 ≥ 1) (h4_ge : h4 ≥ 1)
  (volume : V = (1/3) * S1 * h1)
  : (∃ r : ℝ, 3 * V = (S1 + S2 + S3 + S4) * r ∧ r = 1 / 4) :=
by
  sorry

end inscribed_sphere_radius_l264_264830


namespace total_hours_is_900_l264_264479

-- Definitions for the video length, speeds, and number of videos watched
def video_length : ℕ := 100
def lila_speed : ℕ := 2
def roger_speed : ℕ := 1
def num_videos : ℕ := 6

-- Definition of total hours watched
def total_hours_watched : ℕ :=
  let lila_time_per_video := video_length / lila_speed
  let roger_time_per_video := video_length / roger_speed
  (lila_time_per_video * num_videos) + (roger_time_per_video * num_videos)

-- Prove that the total hours watched is 900
theorem total_hours_is_900 : total_hours_watched = 900 :=
by
  -- Proving the equation step-by-step
  sorry

end total_hours_is_900_l264_264479


namespace maximum_value_of_expression_l264_264061

theorem maximum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz * (x + y + z)) / ((x + y)^2 * (y + z)^2) ≤ (1 / 4) :=
sorry

end maximum_value_of_expression_l264_264061


namespace eliana_additional_steps_first_day_l264_264376

variables (x : ℝ)

def eliana_first_day_steps := 200 + x
def eliana_second_day_steps := 2 * eliana_first_day_steps
def eliana_third_day_steps := eliana_second_day_steps + 100
def eliana_total_steps := eliana_first_day_steps + eliana_second_day_steps + eliana_third_day_steps

theorem eliana_additional_steps_first_day : eliana_total_steps = 1600 → x = 100 :=
by {
  sorry
}

end eliana_additional_steps_first_day_l264_264376


namespace toy_robot_shipment_l264_264354

-- Define the conditions provided in the problem
def thirty_percent_displayed (total: ℕ) : ℕ := (3 * total) / 10
def seventy_percent_stored (total: ℕ) : ℕ := (7 * total) / 10

-- The main statement to prove: if 70% of the toy robots equal 140, then the total number of toy robots is 200
theorem toy_robot_shipment (total : ℕ) (h : seventy_percent_stored total = 140) : total = 200 :=
by
  -- We will fill in the proof here
  sorry

end toy_robot_shipment_l264_264354


namespace pump_fill_time_without_leak_l264_264652

def time_with_leak := 10
def leak_empty_time := 10

def combined_rate_with_leak := 1 / time_with_leak
def leak_rate := 1 / leak_empty_time

def T : ℝ := 5

theorem pump_fill_time_without_leak
  (time_with_leak : ℝ)
  (leak_empty_time : ℝ)
  (combined_rate_with_leak : ℝ)
  (leak_rate : ℝ)
  (T : ℝ)
  (h1 : combined_rate_with_leak = 1 / time_with_leak)
  (h2 : leak_rate = 1 / leak_empty_time)
  (h_combined : 1 / T - leak_rate = combined_rate_with_leak) :
  T = 5 :=
by {
  sorry
}

end pump_fill_time_without_leak_l264_264652


namespace equilateral_triangle_circumcircle_point_distance_sum_l264_264229

-- Define the equilateral triangle, points, and distances
variables {A B C P : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space P]
variables {a : ℝ} -- side length of the equilateral triangle
variables (h_equilateral : equilateral A B C a)
variables (h_circumcircle : circumcircle A B C)
variables (h_point_on_circle : point_on_circumcircle P A B C)

-- Objective: Prove PB + PC = PA
theorem equilateral_triangle_circumcircle_point_distance_sum
  (h_P_distances : PB + PC = PA) :
  PB + PC = PA :=
sorry

end equilateral_triangle_circumcircle_point_distance_sum_l264_264229


namespace polynomial_roots_l264_264544

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end polynomial_roots_l264_264544


namespace probJackAndJillChosen_l264_264727

-- Define the probabilities of each worker being chosen
def probJack : ℝ := 0.20
def probJill : ℝ := 0.15

-- Define the probability that Jack and Jill are both chosen
def probJackAndJill : ℝ := probJack * probJill

-- Theorem stating the probability that Jack and Jill are both chosen
theorem probJackAndJillChosen : probJackAndJill = 0.03 := 
by
  -- Replace this sorry with the complete proof
  sorry

end probJackAndJillChosen_l264_264727


namespace range_for_a_l264_264075

def f (a : ℝ) (x : ℝ) := 2 * x^3 - a * x^2 + 1

def two_zeros_in_interval (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (1/2 ≤ x1 ∧ x1 ≤ 2) ∧ (1/2 ≤ x2 ∧ x2 ≤ 2) ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0

theorem range_for_a {a : ℝ} : (3/2 : ℝ) < a ∧ a ≤ (17/4 : ℝ) ↔ two_zeros_in_interval a :=
by sorry

end range_for_a_l264_264075


namespace quadratic_roots_m_eq_two_l264_264958

theorem quadratic_roots_m_eq_two (m : ℝ) (x₁ x₂ : ℝ) 
  (h_eq : m * x₁^2 - (m + 2) * x₁ + m / 4 = 0)
  (h1 : m * x₂^2 - (m + 2) * x₂ + m / 4 = 0) 
  (h2 : (x₁ ≠ x₂)) 
  (h3 : 1 / x₁ + 1 / x₂ = 4 * m) : 
  m = 2 :=
by sorry

end quadratic_roots_m_eq_two_l264_264958


namespace hexagon_angles_sum_l264_264583

theorem hexagon_angles_sum (α β γ δ ε ζ : ℝ)
  (h1 : α + γ + ε = 180)
  (h2 : β + δ + ζ = 180) : 
  α + β + γ + δ + ε + ζ = 360 :=
by 
  sorry

end hexagon_angles_sum_l264_264583


namespace alissa_presents_l264_264811

theorem alissa_presents :
  let Ethan_presents := 31
  let Alissa_presents := Ethan_presents + 22
  Alissa_presents = 53 :=
by
  sorry

end alissa_presents_l264_264811


namespace roots_of_polynomial_l264_264535

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l264_264535


namespace theta_in_second_quadrant_l264_264687

-- Our problem conditions and theorem
theorem theta_in_second_quadrant (θ : ℝ) 
  (h1 : sin(π / 2 + θ) < 0) 
  (h2 : tan(π - θ) > 0) : 
  (∃ k : ℤ, θ = π / 2 + k * π ∨ θ = -π / 2 + k * π) :=
by sorry

end theta_in_second_quadrant_l264_264687


namespace cubic_difference_l264_264090

theorem cubic_difference (x : ℝ) (h : (x + 16) ^ (1/3) - (x - 16) ^ (1/3) = 4) : 
  235 < x^2 ∧ x^2 < 240 := 
sorry

end cubic_difference_l264_264090


namespace entree_cost_14_l264_264703

theorem entree_cost_14 (D E : ℝ) (h1 : D + E = 23) (h2 : E = D + 5) : E = 14 :=
sorry

end entree_cost_14_l264_264703


namespace find_largest_t_l264_264549

theorem find_largest_t (t : ℝ) : 
  (15 * t^2 - 38 * t + 14) / (4 * t - 3) + 6 * t = 7 * t - 2 → t ≤ 1 := 
by 
  intro h
  sorry

end find_largest_t_l264_264549


namespace floor_tiling_l264_264768

-- Define that n can be expressed as 7k for some integer k.
theorem floor_tiling (n : ℕ) (h : ∃ x : ℕ, n^2 = 7 * x) : ∃ k : ℕ, n = 7 * k := by
  sorry

end floor_tiling_l264_264768


namespace num_dislikers_tv_books_games_is_correct_l264_264438

-- Definitions of the conditions as given in step A
def total_people : ℕ := 1500
def pct_dislike_tv : ℝ := 0.4
def pct_dislike_tv_books : ℝ := 0.15
def pct_dislike_tv_books_games : ℝ := 0.5

-- Calculate intermediate values
def num_tv_dislikers := pct_dislike_tv * total_people
def num_tv_books_dislikers := pct_dislike_tv_books * num_tv_dislikers
def num_tv_books_games_dislikers := pct_dislike_tv_books_games * num_tv_books_dislikers

-- Final proof statement ensuring the correctness of the solution
theorem num_dislikers_tv_books_games_is_correct :
  num_tv_books_games_dislikers = 45 := by
  -- Sorry placeholder for the proof. In actual Lean usage, this would require fulfilling the proof obligations.
  sorry

end num_dislikers_tv_books_games_is_correct_l264_264438


namespace balls_in_boxes_l264_264564

theorem balls_in_boxes :
  (∃ x1 x2 x3 : ℕ, x1 + x2 + x3 = 7) →
  (multichoose 7 3) = 36 :=
by
  sorry

end balls_in_boxes_l264_264564


namespace smallest_possible_sum_l264_264559

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_ne : x ≠ y) (h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 50 :=
sorry

end smallest_possible_sum_l264_264559


namespace sqrt_expr_value_l264_264665

theorem sqrt_expr_value : sqrt ((41 * 40 * 39 * 38) + 1) = 1559 := 
by sorry

end sqrt_expr_value_l264_264665


namespace smaller_pack_size_l264_264375

theorem smaller_pack_size {x : ℕ} (total_eggs large_pack_size large_packs : ℕ) (eggs_in_smaller_packs : ℕ) :
  total_eggs = 79 → large_pack_size = 11 → large_packs = 5 → eggs_in_smaller_packs = total_eggs - large_pack_size * large_packs →
  x * 1 = eggs_in_smaller_packs → x = 24 :=
by sorry

end smaller_pack_size_l264_264375


namespace celine_smartphones_l264_264214

-- Definitions based on the conditions
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops_bought : ℕ := 2
def initial_amount : ℕ := 3000
def change_received : ℕ := 200

-- The proof goal is to show that the number of smartphones bought is 4
theorem celine_smartphones (laptop_cost smartphone_cost num_laptops_bought initial_amount change_received : ℕ)
  (h1 : laptop_cost = 600)
  (h2 : smartphone_cost = 400)
  (h3 : num_laptops_bought = 2)
  (h4 : initial_amount = 3000)
  (h5 : change_received = 200) :
  (initial_amount - change_received - num_laptops_bought * laptop_cost) / smartphone_cost = 4 := 
by
  sorry

end celine_smartphones_l264_264214


namespace percentage_cut_second_week_l264_264790

noncomputable def calculate_final_weight (initial_weight : ℝ) (percentage1 : ℝ) (percentage2 : ℝ) (percentage3 : ℝ) : ℝ :=
  let weight_after_first_week := (1 - percentage1 / 100) * initial_weight
  let weight_after_second_week := (1 - percentage2 / 100) * weight_after_first_week
  let final_weight := (1 - percentage3 / 100) * weight_after_second_week
  final_weight

theorem percentage_cut_second_week : 
  ∀ (initial_weight : ℝ) (final_weight : ℝ), (initial_weight = 250) → (final_weight = 105) →
    (calculate_final_weight initial_weight 30 x 25 = final_weight) → 
    x = 20 := 
by 
  intros initial_weight final_weight h1 h2 h3
  sorry

end percentage_cut_second_week_l264_264790


namespace determine_disco_ball_price_l264_264662

variable (x y z : ℝ)

-- Given conditions
def budget_constraint : Prop := 4 * x + 10 * y + 20 * z = 600
def food_cost : Prop := y = 0.85 * x
def decoration_cost : Prop := z = x / 2 - 10

-- Goal
theorem determine_disco_ball_price (h1 : budget_constraint x y z) (h2 : food_cost x y) (h3 : decoration_cost x z) :
  x = 35.56 :=
sorry 

end determine_disco_ball_price_l264_264662


namespace scrabble_letter_values_l264_264871

-- Definitions based on conditions
def middle_letter_value : ℕ := 8
def final_score : ℕ := 30

-- The theorem we need to prove
theorem scrabble_letter_values (F T : ℕ)
  (h1 : 3 * (F + middle_letter_value + T) = final_score) :
  F = 1 ∧ T = 1 :=
sorry

end scrabble_letter_values_l264_264871


namespace denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l264_264804

variable (DenyMotion : Prop) (AcknowledgeStillness : Prop) (LeadsToRelativism : Prop)
variable (LeadsToSophistry : Prop)

theorem denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry
  (h1 : DenyMotion)
  (h2 : AcknowledgeStillness)
  (h3 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToRelativism)
  (h4 : DenyMotion ∧ AcknowledgeStillness → ¬LeadsToSophistry):
  ¬ (DenyMotion ∧ AcknowledgeStillness → LeadsToRelativism ∧ LeadsToSophistry) :=
by sorry

end denying_motion_and_acknowledging_stillness_does_not_lead_to_relativism_and_sophistry_l264_264804


namespace find_a_values_l264_264091

def setA : Set ℝ := {-1, 1/2, 1}
def setB (a : ℝ) : Set ℝ := {x | a * x^2 = 1 ∧ a ≥ 0}

def full_food (A B : Set ℝ) : Prop := A ⊆ B ∨ B ⊆ A
def partial_food (A B : Set ℝ) : Prop := (∃ x, x ∈ A ∧ x ∈ B) ∧ ¬(A ⊆ B ∨ B ⊆ A)

theorem find_a_values :
  ∀ a : ℝ, full_food setA (setB a) ∨ partial_food setA (setB a) ↔ a = 0 ∨ a = 1 ∨ a = 4 := 
by
  sorry

end find_a_values_l264_264091


namespace solve_fractional_equation_l264_264153

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l264_264153


namespace expression_simplifies_to_one_l264_264626

theorem expression_simplifies_to_one :
  ( (105^2 - 8^2) / (80^2 - 13^2) ) * ( (80 - 13) * (80 + 13) / ( (105 - 8) * (105 + 8) ) ) = 1 :=
by
  sorry

end expression_simplifies_to_one_l264_264626


namespace total_cost_is_13_l264_264489

-- Definition of pencil cost
def pencil_cost : ℕ := 2

-- Definition of pen cost based on pencil cost
def pen_cost : ℕ := pencil_cost + 9

-- The total cost of both items
def total_cost := pencil_cost + pen_cost

theorem total_cost_is_13 : total_cost = 13 := by
  sorry

end total_cost_is_13_l264_264489


namespace pork_price_increase_l264_264411

variable (x : ℝ)
variable (P_aug P_oct : ℝ)
variable (P_aug := 32)
variable (P_oct := 64)

theorem pork_price_increase :
  P_aug * (1 + x) ^ 2 = P_oct :=
sorry

end pork_price_increase_l264_264411


namespace sum_of_first_n_terms_l264_264701

open BigOperators

def a (n : ℕ) : ℝ := n / 3^n

noncomputable def S (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), a k

theorem sum_of_first_n_terms (n : ℕ) :
  S n = (3 / 4) - (2 * n + 3) / (4 * 3^n) :=
by
  sorry

end sum_of_first_n_terms_l264_264701


namespace total_chocolate_bars_in_colossal_box_l264_264637

theorem total_chocolate_bars_in_colossal_box :
  let colossal_boxes := 350
  let sizable_boxes := 49
  let small_boxes := 75
  colossal_boxes * sizable_boxes * small_boxes = 1287750 :=
by
  sorry

end total_chocolate_bars_in_colossal_box_l264_264637


namespace inequality_for_positive_a_b_n_l264_264968

theorem inequality_for_positive_a_b_n (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry

end inequality_for_positive_a_b_n_l264_264968


namespace distance_focus_directrix_l264_264297

theorem distance_focus_directrix (p : ℝ) :
  (∀ (x y : ℝ), y^2 = 2 * p * x ∧ x = 6 ∧ dist (x, y) (p/2, 0) = 10) →
  abs (p) = 8 :=
by
  sorry

end distance_focus_directrix_l264_264297


namespace quadratic_function_passes_through_origin_l264_264612

theorem quadratic_function_passes_through_origin (a : ℝ) :
  ((a - 1) * 0^2 - 0 + a^2 - 1 = 0) → a = -1 :=
by
  intros h
  sorry

end quadratic_function_passes_through_origin_l264_264612


namespace value_of_c_l264_264468

theorem value_of_c (c : ℝ) : (∀ x : ℝ, x * (4 * x + 1) < c ↔ x > -5 / 2 ∧ x < 3) → c = 27 :=
by
  intros h
  sorry

end value_of_c_l264_264468


namespace stock_investment_decrease_l264_264216

theorem stock_investment_decrease (x : ℝ) (d1 d2 : ℝ) (hx : x > 0)
  (increase : x * 1.30 = 1.30 * x) :
  d1 = 20 ∧ d2 = 3.85 → 1.30 * (1 - d1 / 100) * (1 - d2 / 100) = 1 := by
  sorry

end stock_investment_decrease_l264_264216


namespace incircle_angle_b_l264_264453

open Real

theorem incircle_angle_b
    (α β γ : ℝ)
    (h1 : α + β + γ = 180)
    (angle_AOC_eq_4_MKN : ∀ (MKN : ℝ), 4 * MKN = 180 - (180 - γ) / 2 - (180 - α) / 2) :
    β = 108 :=
by
  -- Proof will be handled here.
  sorry

end incircle_angle_b_l264_264453


namespace John_finishes_at_610PM_l264_264588

def TaskTime : Nat := 55
def StartTime : Nat := 14 * 60 + 30 -- 2:30 PM in minutes
def EndSecondTask : Nat := 16 * 60 + 20 -- 4:20 PM in minutes

theorem John_finishes_at_610PM (h1 : TaskTime * 2 = EndSecondTask - StartTime) : 
  (EndSecondTask + TaskTime * 2) = (18 * 60 + 10) :=
by
  sorry

end John_finishes_at_610PM_l264_264588


namespace not_valid_base_five_l264_264447

theorem not_valid_base_five (k : ℕ) (h₁ : k = 5) : ¬(∀ d ∈ [3, 2, 5, 0, 1], d < k) :=
by
  sorry

end not_valid_base_five_l264_264447


namespace adoption_days_l264_264925

def initial_puppies : ℕ := 15
def additional_puppies : ℕ := 62
def adoption_rate : ℕ := 7

def total_puppies : ℕ := initial_puppies + additional_puppies

theorem adoption_days :
  total_puppies / adoption_rate = 11 :=
by
  sorry

end adoption_days_l264_264925


namespace willie_cream_from_farm_l264_264681

variable (total_needed amount_to_buy amount_from_farm : ℕ)

theorem willie_cream_from_farm :
  total_needed = 300 → amount_to_buy = 151 → amount_from_farm = total_needed - amount_to_buy → amount_from_farm = 149 := by
  intros
  sorry

end willie_cream_from_farm_l264_264681


namespace calculate_fraction_l264_264799

variable (a b : ℝ)

theorem calculate_fraction (h : a ≠ b) : (2 * a / (a - b)) + (2 * b / (b - a)) = 2 := by
  sorry

end calculate_fraction_l264_264799


namespace no_infinite_subset_of_natural_numbers_l264_264054

theorem no_infinite_subset_of_natural_numbers {
  S : Set ℕ 
} (hS_infinite : S.Infinite) :
  ¬ (∀ a b : ℕ, a ∈ S → b ∈ S → a^2 - a * b + b^2 ∣ (a * b)^2) :=
sorry

end no_infinite_subset_of_natural_numbers_l264_264054


namespace rain_probability_in_two_locations_l264_264809

noncomputable def probability_no_rain_A : ℝ := 0.3
noncomputable def probability_no_rain_B : ℝ := 0.4

-- The probability of raining at a location is 1 - the probability of no rain at that location
noncomputable def probability_rain_A : ℝ := 1 - probability_no_rain_A
noncomputable def probability_rain_B : ℝ := 1 - probability_no_rain_B

-- The rain status in location A and location B are independent
theorem rain_probability_in_two_locations :
  probability_rain_A * probability_rain_B = 0.42 := by
  sorry

end rain_probability_in_two_locations_l264_264809


namespace solve_equation_l264_264158

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l264_264158


namespace find_height_of_triangular_prism_l264_264649

-- Define the conditions
def volume (V : ℝ) : Prop := V = 120
def base_side1 (a : ℝ) : Prop := a = 3
def base_side2 (b : ℝ) : Prop := b = 4

-- The final proof problem
theorem find_height_of_triangular_prism (V : ℝ) (a : ℝ) (b : ℝ) (h : ℝ) 
  (h1 : volume V) (h2 : base_side1 a) (h3 : base_side2 b) : h = 20 :=
by
  -- The actual proof goes here
  sorry

end find_height_of_triangular_prism_l264_264649


namespace sufficient_and_necessary_cond_l264_264217

theorem sufficient_and_necessary_cond (x : ℝ) : |x| > 2 ↔ (x > 2) :=
sorry

end sufficient_and_necessary_cond_l264_264217


namespace complementary_angle_l264_264570

theorem complementary_angle (angle_deg : ℕ) (angle_min : ℕ) 
  (h1 : angle_deg = 37) (h2 : angle_min = 38) : 
  exists (comp_deg : ℕ) (comp_min : ℕ), comp_deg = 52 ∧ comp_min = 22 :=
by
  sorry

end complementary_angle_l264_264570


namespace triangle_area_l264_264355

noncomputable def a := 5
noncomputable def b := 4
noncomputable def s := (13 : ℝ) / 2 -- semi-perimeter
noncomputable def area := Real.sqrt (s * (s - a) * (s - b) * (s - b))

theorem triangle_area :
  a + 2 * b = 13 →
  (a > 0) → (b > 0) →
  (a < 2 * b) →
  (a + b > b) → 
  (a + b > b) →
  area = Real.sqrt 61.09375 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- We assume validity of these conditions and skip the proof for brevity.
  sorry

end triangle_area_l264_264355


namespace circle_equation_l264_264238

theorem circle_equation
  (a b r : ℝ) 
  (h1 : a^2 + b^2 = r^2) 
  (h2 : (a - 2)^2 + b^2 = r^2) 
  (h3 : b / (a - 2) = 1) : 
  (x - 1)^2 + (y + 1)^2 = 2 := 
by
  sorry

end circle_equation_l264_264238


namespace mass_percentage_ca_in_compound_l264_264464

noncomputable def mass_percentage_ca_in_cac03 : ℝ :=
  let mm_ca := 40.08
  let mm_c := 12.01
  let mm_o := 16.00
  let mm_caco3 := mm_ca + mm_c + 3 * mm_o
  (mm_ca / mm_caco3) * 100

theorem mass_percentage_ca_in_compound (mp : ℝ) (h : mp = mass_percentage_ca_in_cac03) : mp = 40.04 := by
  sorry

end mass_percentage_ca_in_compound_l264_264464


namespace dot_product_AB_BC_l264_264575

theorem dot_product_AB_BC (AB BC : ℝ) (B : ℝ) 
  (h1 : AB = 3) (h2 : BC = 4) (h3 : B = π/6) :
  (AB * BC * Real.cos (π - B) = -6 * Real.sqrt 3) :=
by
  rw [h1, h2, h3]
  sorry

end dot_product_AB_BC_l264_264575


namespace initial_amount_l264_264601

-- Define the conditions
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def num_small_glasses : ℕ := 8
def num_large_glasses : ℕ := 5
def change_left : ℕ := 1

-- Define the pieces based on conditions
def total_cost_small_glasses : ℕ := num_small_glasses * cost_small_glass
def total_cost_large_glasses : ℕ := num_large_glasses * cost_large_glass
def total_cost_glasses : ℕ := total_cost_small_glasses + total_cost_large_glasses

-- The theorem we need to prove
theorem initial_amount (h1 : total_cost_small_glasses = 24)
                       (h2 : total_cost_large_glasses = 25)
                       (h3 : total_cost_glasses = 49) : total_cost_glasses + change_left = 50 :=
by sorry

end initial_amount_l264_264601


namespace find_B_l264_264270

theorem find_B (A B : ℕ) (h1 : Prime A) (h2 : Prime B) (h3 : A > 0) (h4 : B > 0) 
  (h5 : 1 / A - 1 / B = 192 / (2005^2 - 2004^2)) : B = 211 :=
sorry

end find_B_l264_264270


namespace miles_run_on_tuesday_l264_264749

-- Defining the distances run on specific days
def distance_monday : ℝ := 4.2
def distance_wednesday : ℝ := 3.6
def distance_thursday : ℝ := 4.4

-- Average distance run on each of the days Terese runs
def average_distance : ℝ := 4
-- Number of days Terese runs
def running_days : ℕ := 4

-- Defining the total distance calculated using the average distance and number of days
def total_distance : ℝ := average_distance * running_days

-- Defining the total distance run on Monday, Wednesday, and Thursday
def total_other_days : ℝ := distance_monday + distance_wednesday + distance_thursday

-- The distance run on Tuesday can be defined as the difference between the total distance and the total distance on other days
theorem miles_run_on_tuesday : 
  total_distance - total_other_days = 3.8 :=
by
  sorry

end miles_run_on_tuesday_l264_264749


namespace Helen_raisins_l264_264080

/-- Given that Helen baked 19 chocolate chip cookies yesterday, baked some raisin cookies and 237 chocolate chip cookies this morning,
    and baked 25 more chocolate chip cookies than raisin cookies in total,
    prove that the number of raisin cookies (R) she baked is 231. -/
theorem Helen_raisins (R : ℕ) (h1 : 25 + R = 256) : R = 231 :=
by
  sorry

end Helen_raisins_l264_264080


namespace video_minutes_per_week_l264_264110

theorem video_minutes_per_week
  (daily_videos : ℕ := 3)
  (short_video_length : ℕ := 2)
  (long_video_multiplier : ℕ := 6)
  (days_in_week : ℕ := 7) :
  (2 * short_video_length + long_video_multiplier * short_video_length) * days_in_week = 112 := 
by 
  -- conditions
  let short_videos_per_day := 2
  let long_video_length := long_video_multiplier * short_video_length
  let daily_total := short_videos_per_day * short_video_length + long_video_length
  let weekly_total := daily_total * days_in_week
  -- proof
  sorry

end video_minutes_per_week_l264_264110


namespace greatest_consecutive_integers_sum_36_l264_264187

theorem greatest_consecutive_integers_sum_36 : ∀ (x : ℤ), (x + (x + 1) + (x + 2) = 36) → (x + 2 = 13) :=
by
  sorry

end greatest_consecutive_integers_sum_36_l264_264187


namespace find_k_l264_264330
-- Import the necessary library

-- Given conditions as definitions
def circle_eq (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8 * x + y^2 + 2 * y + k = 0

def radius_sq : ℝ := 25  -- since radius = 5, radius squared is 25

-- The statement to prove
theorem find_k (x y k : ℝ) : circle_eq x y k → radius_sq = 25 → k = -8 :=
by
  sorry

end find_k_l264_264330


namespace find_m_l264_264988

-- Define the lines l1 and l2
def line1 (x y : ℝ) (m : ℝ) : Prop := x + m^2 * y + 6 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

-- The statement that two lines are parallel
def lines_parallel (m : ℝ) : Prop :=
  ∀ (x y : ℝ), line1 x y m → line2 x y m

-- The mathematically equivalent proof problem
theorem find_m (m : ℝ) (H_parallel : lines_parallel m) : m = 0 ∨ m = -1 :=
sorry

end find_m_l264_264988


namespace solve_fractional_equation_l264_264147

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l264_264147


namespace range_of_a_l264_264116

theorem range_of_a (a : ℝ) (h1 : 0 < a) :
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≤ 0 → x^2 - x - 6 ≤ 0) ∧
  (¬ (∀ x : ℝ, x^2 - x - 6 ≤ 0 → x^2 - 4*a*x + 3*a^2 ≤ 0)) →
  0 < a ∧ a ≤ 1 :=
sorry

end range_of_a_l264_264116


namespace percentage_mutant_frogs_is_33_l264_264682

def num_extra_legs_frogs := 5
def num_two_heads_frogs := 2
def num_bright_red_frogs := 2
def num_normal_frogs := 18

def total_mutant_frogs := num_extra_legs_frogs + num_two_heads_frogs + num_bright_red_frogs
def total_frogs := total_mutant_frogs + num_normal_frogs

theorem percentage_mutant_frogs_is_33 :
  Float.round (100 * total_mutant_frogs.toFloat / total_frogs.toFloat) = 33 :=
by 
  -- placeholder for the proof
  sorry

end percentage_mutant_frogs_is_33_l264_264682


namespace select_numbers_with_second_largest_seven_l264_264221

open Finset

theorem select_numbers_with_second_largest_seven : 
  (univ.filter (λ s : Finset ℕ, s.card = 4 ∧ 7 ∈ s ∧ secondLargest s = some 7)).card = 45 :=
sorry

end select_numbers_with_second_largest_seven_l264_264221


namespace race_distance_l264_264097

theorem race_distance
  (x y z d : ℝ) 
  (h1 : d / x = (d - 25) / y)
  (h2 : d / y = (d - 15) / z)
  (h3 : d / x = (d - 35) / z) : 
  d = 75 := 
sorry

end race_distance_l264_264097


namespace loaves_of_bread_l264_264231

variable (B : ℕ) -- Number of loaves of bread Erik bought
variable (total_money : ℕ := 86) -- Money given to Erik
variable (money_left : ℕ := 59) -- Money left after purchase
variable (cost_bread : ℕ := 3) -- Cost of each loaf of bread
variable (cost_oj : ℕ := 6) -- Cost of each carton of orange juice
variable (num_oj : ℕ := 3) -- Number of cartons of orange juice bought

theorem loaves_of_bread (h1 : total_money - money_left = num_oj * cost_oj + B * cost_bread) : B = 3 := 
by sorry

end loaves_of_bread_l264_264231


namespace evaluate_product_l264_264813

-- Define the given numerical values
def a : ℝ := 2.5
def b : ℝ := 50.5
def c : ℝ := 0.15

-- State the theorem we want to prove
theorem evaluate_product : a * (b + c) = 126.625 := by
  sorry

end evaluate_product_l264_264813


namespace smallest_cookie_packages_l264_264771

/-- The smallest number of cookie packages Zoey can buy in order to buy an equal number of cookie
and milk packages. -/
theorem smallest_cookie_packages (n : ℕ) (h1 : ∃ k : ℕ, 5 * k = 7 * n) : n = 7 :=
sorry

end smallest_cookie_packages_l264_264771


namespace suzanna_bike_distance_l264_264173

variable (constant_rate : ℝ) (time_minutes : ℝ) (interval : ℝ) (distance_per_interval : ℝ)

theorem suzanna_bike_distance :
  (constant_rate = 1 / interval) ∧ (interval = 5) ∧ (distance_per_interval = constant_rate * interval) ∧ (time_minutes = 30) →
  ((time_minutes / interval) * distance_per_interval = 6) :=
by
  intros
  sorry

end suzanna_bike_distance_l264_264173


namespace three_x_plus_three_y_plus_three_z_l264_264716

theorem three_x_plus_three_y_plus_three_z (x y z : ℝ) 
  (h1 : y + z = 20 - 5 * x)
  (h2 : x + z = -18 - 5 * y)
  (h3 : x + y = 10 - 5 * z) :
  3 * x + 3 * y + 3 * z = 36 / 7 := by
  sorry

end three_x_plus_three_y_plus_three_z_l264_264716


namespace sally_total_score_l264_264413

theorem sally_total_score :
  ∀ (correct incorrect unanswered : ℕ) (score_correct score_incorrect : ℝ),
    correct = 17 →
    incorrect = 8 →
    unanswered = 5 →
    score_correct = 1 →
    score_incorrect = -0.25 →
    (correct * score_correct +
     incorrect * score_incorrect +
     unanswered * 0) = 15 :=
by
  intros correct incorrect unanswered score_correct score_incorrect
  intros h_corr h_incorr h_unan h_sc h_si
  sorry

end sally_total_score_l264_264413


namespace number_of_intersections_l264_264051

theorem number_of_intersections : 
  (∃ p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ 9 * p.1^2 + p.2^2 = 1) 
  ∧ (∃! p₁ p₂ : ℝ × ℝ, p₁ ≠ p₂ ∧ p₁.1^2 + 9 * p₁.2^2 = 9 ∧ 9 * p₁.1^2 + p₁.2^2 = 1 ∧
    p₂.1^2 + 9 * p₂.2^2 = 9 ∧ 9 * p₂.1^2 + p₂.2^2 = 1) :=
by
  -- The proof will be here
  sorry

end number_of_intersections_l264_264051


namespace prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l264_264263

variable {p a b : ℤ}

theorem prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes
  (hp : Prime p) (hp_ne_3 : p ≠ 3)
  (h1 : p ∣ (a + b)) (h2 : p^2 ∣ (a^3 + b^3)) :
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) :=
sorry

end prime_number_p_squared_divides_sum_or_cube_divides_sum_of_cubes_l264_264263


namespace number_of_lines_through_point_intersect_hyperbola_once_l264_264921

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 = 1

noncomputable def point_P : ℝ × ℝ :=
  (-4, 1)

noncomputable def line_through (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  l P

noncomputable def one_point_intersection (l : ℝ × ℝ → Prop) (H : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, l p ∧ H p.1 p.2

theorem number_of_lines_through_point_intersect_hyperbola_once :
  (∃ (l₁ l₂ : ℝ × ℝ → Prop),
    line_through point_P l₁ ∧
    line_through point_P l₂ ∧
    one_point_intersection l₁ hyperbola ∧
    one_point_intersection l₂ hyperbola ∧
    l₁ ≠ l₂) ∧ ¬ (∃ (l₃ : ℝ × ℝ → Prop),
    line_through point_P l₃ ∧
    one_point_intersection l₃ hyperbola ∧
    ∃! (other_line : ℝ × ℝ → Prop),
    line_through point_P other_line ∧
    one_point_intersection other_line hyperbola ∧
    l₃ ≠ other_line) :=
sorry

end number_of_lines_through_point_intersect_hyperbola_once_l264_264921


namespace exists_valid_arrangement_n_1_exists_valid_arrangement_n_gt_1_l264_264899

-- Define the conditions
def num_mathematicians (n : ℕ) : ℕ := 6 * n + 4
def num_meetings (n : ℕ) : ℕ := 2 * n + 1
def num_4_person_tables (n : ℕ) : ℕ := 1
def num_6_person_tables (n : ℕ) : ℕ := n

-- Define the constraint on arrangements
def valid_arrangement (n : ℕ) : Prop :=
  -- A placeholder for the actual arrangement checking logic.
  -- This should ensure no two people sit next to or opposite each other more than once.
  sorry

-- Proof of existence of a valid arrangement when n = 1
theorem exists_valid_arrangement_n_1 : valid_arrangement 1 :=
sorry

-- Proof of existence of a valid arrangement when n > 1
theorem exists_valid_arrangement_n_gt_1 (n : ℕ) (h : n > 1) : valid_arrangement n :=
sorry

end exists_valid_arrangement_n_1_exists_valid_arrangement_n_gt_1_l264_264899


namespace total_pears_picked_is_correct_l264_264440

-- Define the number of pears picked by Sara and Sally
def pears_picked_by_Sara : ℕ := 45
def pears_picked_by_Sally : ℕ := 11

-- The total number of pears picked
def total_pears_picked := pears_picked_by_Sara + pears_picked_by_Sally

-- The theorem statement: prove that the total number of pears picked is 56
theorem total_pears_picked_is_correct : total_pears_picked = 56 := by
  sorry

end total_pears_picked_is_correct_l264_264440


namespace min_abs_x1_x2_l264_264247

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin x - 2 * Real.sqrt 3 * Real.cos x

theorem min_abs_x1_x2 
  (a x1 x2 : ℝ)
  (h_symmetry : ∃ c : ℝ, c = -Real.pi / 6 ∧ (∀ x, f a (x - c) = f a x))
  (h_product : f a x1 * f a x2 = -16) :
  ∃ m : ℝ, m = abs (x1 + x2) ∧ m = 2 * Real.pi / 3 :=
by sorry

end min_abs_x1_x2_l264_264247


namespace percent_defective_units_shipped_for_sale_l264_264724

variable (total_units : ℕ)
variable (defective_units_percentage : ℝ := 0.08)
variable (shipped_defective_units_percentage : ℝ := 0.05)

theorem percent_defective_units_shipped_for_sale :
  defective_units_percentage * shipped_defective_units_percentage * 100 = 0.4 :=
by
  sorry

end percent_defective_units_shipped_for_sale_l264_264724


namespace residue_7_pow_1234_l264_264324

theorem residue_7_pow_1234 : (7^1234) % 13 = 4 := by
  sorry

end residue_7_pow_1234_l264_264324


namespace ellipse_k_values_l264_264399

theorem ellipse_k_values (k : ℝ) :
  (∃ a b : ℝ, a = (k + 8) ∧ b = 9 ∧ 
  (b > a → (a * (1 - (1 / 2) ^ 2) = b - a) ∧ k = 4) ∧ 
  (a > b → (b * (1 - (1 / 2) ^ 2) = a - b) ∧ k = -5/4)) :=
sorry

end ellipse_k_values_l264_264399


namespace sequence_sum_a5_a6_l264_264392

-- Given sequence partial sum definition
def partial_sum (n : ℕ) : ℕ := n^3

-- Definition of sequence term a_n
def a (n : ℕ) : ℕ := partial_sum n - partial_sum (n - 1)

-- Main theorem to prove a_5 + a_6 = 152
theorem sequence_sum_a5_a6 : a 5 + a 6 = 152 :=
by
  sorry

end sequence_sum_a5_a6_l264_264392


namespace oliver_used_fraction_l264_264882

variable (x : ℚ)

/--
Oliver had 135 stickers. He used a fraction x of his stickers, gave 2/5 of the remaining to his friend, and kept the remaining 54 stickers. Prove that he used 1/3 of his stickers.
-/
theorem oliver_used_fraction (h : 135 - (135 * x) - (2 / 5) * (135 - 135 * x) = 54) : 
  x = 1 / 3 := 
sorry

end oliver_used_fraction_l264_264882


namespace min_value_of_Box_l264_264407

theorem min_value_of_Box (c d : ℤ) (hcd : c * d = 42) (distinct_values : c ≠ d ∧ c ≠ 85 ∧ d ≠ 85) :
  ∃ (Box : ℤ), (c^2 + d^2 = Box) ∧ (Box = 85) :=
by
  sorry

end min_value_of_Box_l264_264407


namespace greatest_area_difference_l264_264318

theorem greatest_area_difference 
  (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) :
  abs ((l * w) - (l' * w')) ≤ 1521 := 
sorry

end greatest_area_difference_l264_264318


namespace total_marks_prove_total_marks_l264_264475

def average_marks : ℝ := 40
def number_of_candidates : ℕ := 50

theorem total_marks (average_marks : ℝ) (number_of_candidates : ℕ) : Real :=
  average_marks * number_of_candidates

theorem prove_total_marks : total_marks average_marks number_of_candidates = 2000 := 
by
  sorry

end total_marks_prove_total_marks_l264_264475


namespace factor_expression_l264_264528

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l264_264528


namespace area_of_rectangle_l264_264620

theorem area_of_rectangle (length width : ℝ) (h1 : length = 15) (h2 : width = length * 0.9) : length * width = 202.5 := by
  sorry

end area_of_rectangle_l264_264620


namespace seating_arrangement_l264_264259

def num_ways_seated (total_passengers : ℕ) (window_seats : ℕ) : ℕ :=
  window_seats * (total_passengers - 1) * (total_passengers - 2) * (total_passengers - 3)

theorem seating_arrangement (passengers_seats taxi_window_seats : ℕ)
  (h1 : passengers_seats = 4) (h2 : taxi_window_seats = 2) :
  num_ways_seated passengers_seats taxi_window_seats = 12 :=
by
  -- proof will go here
  sorry

end seating_arrangement_l264_264259


namespace find_expression_l264_264446

theorem find_expression 
  (E a : ℤ) 
  (h1 : (E + (3 * a - 8)) / 2 = 74) 
  (h2 : a = 28) : 
  E = 72 := 
by
  sorry

end find_expression_l264_264446


namespace xy_parallel_ab_l264_264264

open EuclideanGeometry

variable {A B C M K X Y : Point}
variable {triangle_ABC : Triangle A B C}
variable {circumcircle_ABC : Circle circumcenter radius}
variable {midpoint_M : Midpoint M A C}
variable {K_on_minor_arc_AC : InMinorArc circumcircle_ABC K A C}
variable {AKM_ninety_deg : ∠(A, K, M) = 90}
variable {intersection_X : X = LineIntersection (line_through B K) (line_through A M)}
variable {A_altitude_meet_BM_at_Y : Y = LineIntersection (line_through_A_altitude A B C) (line_through B M)}

theorem xy_parallel_ab :
  Parallel (line_through X Y) (line_through A B) := 
by
  sorry

end xy_parallel_ab_l264_264264


namespace special_even_diff_regular_l264_264360

def first_n_even_sum (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

def special_even_sum (n : ℕ) : ℕ :=
  let sum_cubes := (n * (n + 1) / 2) ^ 2
  let sum_squares := n * (n + 1) * (2 * n + 1) / 6
  2 * (sum_cubes + sum_squares)

theorem special_even_diff_regular : 
  let n := 100
  special_even_sum n - first_n_even_sum n = 51403900 :=
by
  sorry

end special_even_diff_regular_l264_264360


namespace point_B_represents_2_or_neg6_l264_264694

def A : ℤ := -2

def B (move : ℤ) : ℤ := A + move

theorem point_B_represents_2_or_neg6 (move : ℤ) (h : move = 4 ∨ move = -4) : 
  B move = 2 ∨ B move = -6 :=
by
  cases h with
  | inl h1 => 
    rw [h1]
    unfold B
    unfold A
    simp
  | inr h1 => 
    rw [h1]
    unfold B
    unfold A
    simp

end point_B_represents_2_or_neg6_l264_264694


namespace roots_of_polynomial_l264_264542

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end roots_of_polynomial_l264_264542


namespace quadrilateral_area_is_33_l264_264352

-- Definitions for the points and their coordinates
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 4, y := 0}
def B : Point := {x := 0, y := 12}
def C : Point := {x := 10, y := 0}
def E : Point := {x := 3, y := 3}

-- Define the quadrilateral area computation
noncomputable def areaQuadrilateral (O B E C : Point) : ℝ :=
  let triangle_area (p1 p2 p3 : Point) :=
    abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2
  triangle_area O B E + triangle_area O E C

-- Statement to prove
theorem quadrilateral_area_is_33 : areaQuadrilateral {x := 0, y := 0} B E C = 33 := by
  sorry

end quadrilateral_area_is_33_l264_264352


namespace reflection_over_vector_l264_264819

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l264_264819


namespace friends_recycled_pounds_l264_264935

theorem friends_recycled_pounds (total_points chloe_points each_points pounds_per_point : ℕ)
  (h1 : each_points = pounds_per_point / 6)
  (h2 : total_points = 5)
  (h3 : chloe_points = pounds_per_point / 6)
  (h4 : pounds_per_point = 28) 
  (h5 : total_points - chloe_points = 1) :
  pounds_per_point = 6 :=
by
  sorry

end friends_recycled_pounds_l264_264935


namespace five_digit_palindromes_count_l264_264660

theorem five_digit_palindromes_count : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) → 
  900 = 9 * 10 * 10 := 
by
  intro h
  sorry

end five_digit_palindromes_count_l264_264660


namespace hyperbola_h_k_a_b_sum_l264_264722

noncomputable def h : ℝ := 1
noncomputable def k : ℝ := -3
noncomputable def a : ℝ := 3
noncomputable def c : ℝ := 3 * Real.sqrt 5
noncomputable def b : ℝ := 6

theorem hyperbola_h_k_a_b_sum :
  h + k + a + b = 7 :=
by
  sorry

end hyperbola_h_k_a_b_sum_l264_264722


namespace initial_pennies_l264_264343

theorem initial_pennies (P : ℕ)
  (h1 : P - (P / 2 + 1) = P / 2 - 1)
  (h2 : (P / 2 - 1) - (P / 4 + 1 / 2) = P / 4 - 3 / 2)
  (h3 : (P / 4 - 3 / 2) - (P / 8 + 3 / 4) = P / 8 - 9 / 4)
  (h4 : P / 8 - 9 / 4 = 1)
  : P = 26 := 
by
  sorry

end initial_pennies_l264_264343


namespace mabel_total_tomatoes_l264_264426

theorem mabel_total_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 :=
by
  sorry

end mabel_total_tomatoes_l264_264426


namespace factor_expression_l264_264522

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l264_264522


namespace units_digit_31_2020_units_digit_37_2020_l264_264478

theorem units_digit_31_2020 : ((31 ^ 2020) % 10) = 1 := by
  sorry

theorem units_digit_37_2020 : ((37 ^ 2020) % 10) = 1 := by
  sorry

end units_digit_31_2020_units_digit_37_2020_l264_264478


namespace greatest_integer_m_divisor_l264_264766

theorem greatest_integer_m_divisor :
  ∃ m : ℕ, (∀ k : ℕ, (k > m → ¬ (20^k ∣ 50!))) ∧ m = 12 :=
by
  sorry

end greatest_integer_m_divisor_l264_264766


namespace find_a_minus_b_l264_264065

theorem find_a_minus_b (a b : ℝ) (h1 : a + b = 12) (h2 : a^2 - b^2 = 48) : a - b = 4 :=
by
  sorry

end find_a_minus_b_l264_264065


namespace arrangements_7_people_no_A_at_head_no_B_in_middle_l264_264176

theorem arrangements_7_people_no_A_at_head_no_B_in_middle :
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  total_arrangements - 2 * A_at_head + overlap = 3720 :=
by
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  show total_arrangements - 2 * A_at_head + overlap = 3720
  sorry

end arrangements_7_people_no_A_at_head_no_B_in_middle_l264_264176


namespace annulus_area_l264_264185

theorem annulus_area (r_inner r_outer : ℝ) (h_inner : r_inner = 8) (h_outer : r_outer = 2 * r_inner) :
  π * r_outer ^ 2 - π * r_inner ^ 2 = 192 * π :=
by
  sorry

end annulus_area_l264_264185


namespace Ivan_walk_time_l264_264377

variables (u v : ℝ) (T t : ℝ)

-- Define the conditions
def condition1 : Prop := T = 10 * v / u
def condition2 : Prop := T + 70 = t
def condition3 : Prop := v * t = u * T + v * (t - T + 70)

-- Problem statement: Given the conditions, prove T = 80
theorem Ivan_walk_time (h1 : condition1 u v T) (h2 : condition2 T t) (h3 : condition3 u v T t) : 
  T = 80 := by
  sorry

end Ivan_walk_time_l264_264377


namespace find_polynomial_l264_264592

noncomputable def polynomial_p (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ t x y a b c : ℝ,
    (P (t * x) (t * y) = t ^ n * P x y) ∧
    (P (a + b) c + P (b + c) a + P (c + a) b = 0) ∧
    (P 1 0 = 1)

theorem find_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) (h : polynomial_p n P) :
  ∀ x y : ℝ, P x y = x^n - y^n :=
sorry

end find_polynomial_l264_264592


namespace speed_of_stream_l264_264026

theorem speed_of_stream
  (v : ℝ)
  (h1 : ∀ t : ℝ, t = 7)
  (h2 : ∀ d : ℝ, d = 72)
  (h3 : ∀ s : ℝ, s = 21)
  : (72 / (21 - v) + 72 / (21 + v) = 7) → v = 3 :=
by
  intro h
  sorry

end speed_of_stream_l264_264026


namespace range_of_a_l264_264078

theorem range_of_a (a : ℝ) : 
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)) → a > 1 :=
by
  sorry

end range_of_a_l264_264078


namespace three_person_subcommittees_from_eight_l264_264405

theorem three_person_subcommittees_from_eight :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end three_person_subcommittees_from_eight_l264_264405


namespace equation_of_line_l264_264614

theorem equation_of_line (A B : ℝ × ℝ) (M : ℝ × ℝ) (hM : M = (-1, 2)) (hA : A.2 = 0) (hB : B.1 = 0) (hMid : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = 4) ∧ ∀ (x y : ℝ), y = a * x + b * y + c → 2 * x - y + 4 = 0 := 
  sorry

end equation_of_line_l264_264614


namespace determine_parity_of_f_l264_264349

def parity_of_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = 0

theorem determine_parity_of_f (f : ℝ → ℝ) :
  (∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) →
  parity_of_f f :=
sorry

end determine_parity_of_f_l264_264349


namespace abs_monotonic_increasing_even_l264_264358

theorem abs_monotonic_increasing_even :
  (∀ x : ℝ, |x| = |(-x)|) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → |x1| ≤ |x2|) :=
by
  sorry

end abs_monotonic_increasing_even_l264_264358


namespace find_angle_l264_264009

theorem find_angle (r1 r2 : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 2) 
(h_shaded : ∀ α : ℝ, 0 < α ∧ α < 2 * π → 
  (360 / 360 * pi * r1^2 + (α / (2 * π)) * pi * r2^2 - (α / (2 * π)) * pi * r1^2 = (1/3) * (pi * r2^2))) : 
  (∀ α : ℝ, 0 < α ∧ α < 2 * π ↔ 
  α = π / 3 ) :=
by
  sorry

end find_angle_l264_264009


namespace factorize1_factorize2_factorize3_factorize4_l264_264058

-- Statement for the first equation
theorem factorize1 (a x : ℝ) : 
  a * x^2 - 7 * a * x + 6 * a = a * (x - 6) * (x - 1) :=
sorry

-- Statement for the second equation
theorem factorize2 (x y : ℝ) : 
  x * y^2 - 9 * x = x * (y + 3) * (y - 3) :=
sorry

-- Statement for the third equation
theorem factorize3 (x y : ℝ) : 
  1 - x^2 + 2 * x * y - y^2 = (1 + x - y) * (1 - x + y) :=
sorry

-- Statement for the fourth equation
theorem factorize4 (x y : ℝ) : 
  8 * (x^2 - 2 * y^2) - x * (7 * x + y) + x * y = (x + 4 * y) * (x - 4 * y) :=
sorry

end factorize1_factorize2_factorize3_factorize4_l264_264058


namespace proof_statement_l264_264197

variables {K_c A_c K_d B_d A_d B_c : ℕ}

def conditions (K_c A_c K_d B_d A_d B_c : ℕ) :=
  K_c > A_c ∧ K_d > B_d ∧ A_d > K_d ∧ B_c > A_c

noncomputable def statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : Prop :=
  A_d > max K_d B_d

theorem proof_statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : statement K_c A_c K_d B_d A_d B_c h :=
sorry

end proof_statement_l264_264197


namespace reflection_matrix_is_correct_l264_264820

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l264_264820


namespace solve_equation_l264_264132

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l264_264132


namespace remainder_b_div_6_l264_264306

theorem remainder_b_div_6 (a b : ℕ) (r_a r_b : ℕ) 
  (h1 : a ≡ r_a [MOD 6]) 
  (h2 : b ≡ r_b [MOD 6]) 
  (h3 : a > b) 
  (h4 : (a - b) % 6 = 5) 
  : b % 6 = 0 := 
sorry

end remainder_b_div_6_l264_264306


namespace rectangle_area_l264_264951

theorem rectangle_area 
  (P : ℝ) (r : ℝ) (hP : P = 40) (hr : r = 3 / 2) : 
  ∃ (length width : ℝ), 2 * (length + width) = P ∧ length = 3 * (width / 2) ∧ (length * width) = 96 :=
by
  sorry

end rectangle_area_l264_264951


namespace breadth_halved_of_percentage_change_area_l264_264300

theorem breadth_halved_of_percentage_change_area {L B B' : ℝ} (h : 0 < L ∧ 0 < B) 
  (h1 : L / 2 * B' = 0.5 * (L * B)) : B' = 0.5 * B :=
sorry

end breadth_halved_of_percentage_change_area_l264_264300


namespace explicit_form_of_f_l264_264737

noncomputable def f (x : ℝ) : ℝ := sorry

theorem explicit_form_of_f :
  (∀ x : ℝ, f x + f (x + 3) = 0) →
  (∀ x : ℝ, -1 < x ∧ x ≤ 1 → f x = 2 * x - 3) →
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → f x = -2 * x + 9) :=
by
  intros h1 h2
  sorry

end explicit_form_of_f_l264_264737


namespace mul_exponent_result_l264_264466

theorem mul_exponent_result : 112 * (5^4) = 70000 := 
by 
  sorry

end mul_exponent_result_l264_264466


namespace radius_of_circle_l264_264028

theorem radius_of_circle (P : ℝ) (PQ QR : ℝ) (distance_center_P : ℝ) (r : ℝ) :
  P = 17 ∧ PQ = 12 ∧ QR = 8 ∧ (PQ * (PQ + QR) = (distance_center_P - r) * (distance_center_P + r)) → r = 7 :=
by
  sorry

end radius_of_circle_l264_264028


namespace samantha_coins_worth_l264_264282

-- Define the conditions and the final question with an expected answer.
theorem samantha_coins_worth (n d : ℕ) (h1 : n + d = 30)
  (h2 : 10 * n + 5 * d = 5 * n + 10 * d + 120) :
  (5 * n + 10 * d) = 165 := 
sorry

end samantha_coins_worth_l264_264282


namespace expected_value_eight_l264_264770

-- Define the 10-sided die roll outcomes
def outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the value function for a roll outcome
def value (x : ℕ) : ℕ :=
  if x % 2 = 0 then x  -- even value
  else 2 * x  -- odd value

-- Calculate the expected value
def expected_value : ℚ :=
  (1 / 10 : ℚ) * (2 + 2 + 6 + 4 + 10 + 6 + 14 + 8 + 18 + 10)

-- The theorem stating the expected value equals 8
theorem expected_value_eight :
  expected_value = 8 := by
  sorry

end expected_value_eight_l264_264770


namespace factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l264_264910

-- Definitions from conditions
theorem factorization_option_a (a b : ℝ) : a^4 * b - 6 * a^3 * b + 9 * a^2 * b = a^2 * b * (a^2 - 6 * a + 9) ↔ a^2 * b * (a - 3)^2 ≠ a^2 * b * (a^2 - 6 * a - 9) := sorry

theorem factorization_option_b (x : ℝ) : (x^2 - x + 1/4) = (x - 1/2)^2 := sorry

theorem factorization_option_c (x : ℝ) : x^2 - 2 * x + 4 = (x - 2)^2 ↔ x^2 - 2 * x + 4 ≠ x^2 - 4 * x + 4 := sorry

theorem factorization_option_d (x y : ℝ) : 4 * x^2 - y^2 = (2 * x + y) * (2 * x - y) ↔ (4 * x + y) * (4 * x - y) ≠ (2 * x + y) * (2 * x - y) := sorry

-- Main theorem that states option B's factorization is correct
theorem correct_factorization_b (x : ℝ) (h1 : x^2 - x + 1/4 = (x - 1/2)^2)
                                (h2 : ∀ (a b : ℝ), a^4 * b - 6 * a^3 * b + 9 * a^2 * b ≠ a^2 * b * (a^2 - 6 * a - 9))
                                (h3 : ∀ (x : ℝ), x^2 - 2 * x + 4 ≠ (x - 2)^2)
                                (h4 : ∀ (x y : ℝ), 4 * x^2 - y^2 ≠ (4 * x + y) * (4 * x - y)) : 
                                (x^2 - x + 1/4 = (x - 1/2)^2) := 
                                by 
                                sorry

end factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l264_264910


namespace algebraic_expression_evaluation_l264_264627

theorem algebraic_expression_evaluation (a b : ℝ) (h : -2 * a + 3 * b + 8 = 18) : 9 * b - 6 * a + 2 = 32 := by
  sorry

end algebraic_expression_evaluation_l264_264627


namespace volleyball_team_arrangements_l264_264591

theorem volleyball_team_arrangements (n : ℕ) (n_pos : 0 < n) :
  ∃ arrangements : ℕ, arrangements = 2^n * (Nat.factorial n)^2 :=
sorry

end volleyball_team_arrangements_l264_264591


namespace arithmetic_sequence_a3_l264_264836

theorem arithmetic_sequence_a3 (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : a2 = a1 + (a1 + a5 - a1) / 4)
  (h2 : a3 = a1 + 2 * (a1 + a5 - a1) / 4) 
  (h3 : a4 = a1 + 3 * (a1 + a5 - a1) / 4) 
  (h4 : a5 = a1 + 4 * (a1 + a5 - a1) / 4)
  (h_sum : 5 * a3 = 15) : 
  a3 = 3 :=
sorry

end arithmetic_sequence_a3_l264_264836


namespace greatest_area_difference_l264_264319

theorem greatest_area_difference 
  (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) :
  abs ((l * w) - (l' * w')) ≤ 1521 := 
sorry

end greatest_area_difference_l264_264319


namespace sqrt_12_estimate_l264_264673

theorem sqrt_12_estimate : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end sqrt_12_estimate_l264_264673


namespace sin_inequality_solution_set_l264_264584

theorem sin_inequality_solution_set : 
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x < - Real.sqrt 3 / 2} =
  {x : ℝ | (4 * Real.pi / 3) < x ∧ x < (5 * Real.pi / 3)} := by
  sorry

end sin_inequality_solution_set_l264_264584


namespace binom_20_6_l264_264938

theorem binom_20_6 : nat.choose 20 6 = 19380 := 
by 
  sorry

end binom_20_6_l264_264938


namespace price_of_20_percent_stock_l264_264547

theorem price_of_20_percent_stock (annual_income : ℝ) (investment : ℝ) (dividend_rate : ℝ) (price_of_stock : ℝ) :
  annual_income = 1000 →
  investment = 6800 →
  dividend_rate = 20 →
  price_of_stock = 136 :=
by
  intros h_income h_investment h_dividend_rate
  sorry

end price_of_20_percent_stock_l264_264547


namespace find_z_l264_264018

/-- x and y are positive integers. When x is divided by 9, the remainder is 2, 
and when x is divided by 7, the remainder is 4. When y is divided by 13, 
the remainder is 12. The least possible value of y - x is 14. 
Prove that the number that y is divided by to get a remainder of 3 is 22. -/
theorem find_z (x y z : ℕ) (hx9 : x % 9 = 2) (hx7 : x % 7 = 4) (hy13 : y % 13 = 12) (hyx : y = x + 14) 
: y % z = 3 → z = 22 := 
by 
  sorry

end find_z_l264_264018


namespace solve_problem_l264_264268

theorem solve_problem : 
  ∃ p q : ℝ, 
    (p ≠ q) ∧ 
    ((∀ x : ℝ, (x = p ∨ x = q) ↔ (x-4)*(x+4) = 24*x - 96)) ∧ 
    (p > q) ∧ 
    (p - q = 16) :=
by
  sorry

end solve_problem_l264_264268


namespace intersection_of_sets_l264_264401

theorem intersection_of_sets :
  let A := {1, 2}
  let B := {x : ℝ | x^2 - 3 * x + 2 = 0}
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_sets_l264_264401


namespace compare_abc_l264_264842

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem compare_abc : a < c ∧ c < b :=
by
  -- The proof will be provided here.
  sorry

end compare_abc_l264_264842


namespace probability_odd_number_die_l264_264310

theorem probability_odd_number_die :
  let total_outcomes := 6
  let favorable_outcomes := 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_odd_number_die_l264_264310


namespace max_value_of_y_l264_264198

theorem max_value_of_y (x : ℝ) (h : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 :=
sorry

end max_value_of_y_l264_264198


namespace rectangle_perimeter_l264_264646

theorem rectangle_perimeter (s : ℝ) (h1 : 4 * s = 180) :
    let length := s
    let width := s / 3
    2 * (length + width) = 120 := 
by
  sorry

end rectangle_perimeter_l264_264646


namespace area_of_tangent_segments_annulus_l264_264390

noncomputable def circle := {center : ℝ × ℝ // center = (0, 0)}
def radius := 3
def segment_length := 3

def internalSegmentArea {R r : ℝ} (annulus_inner_radius : ℝ) (annulus_outer_radius : ℝ) : ℝ :=
  π * (annulus_outer_radius ^ 2 - annulus_inner_radius ^ 2)

theorem area_of_tangent_segments_annulus 
  (c : circle) 
  (r : ℝ) 
  (l : ℝ) 
  (h_radius : r = radius)
  (h_length : l = segment_length) : 
  internalSegmentArea 3 (3 * real.sqrt 5 / 2) = 9 * π / 4 :=
by
  sorry

end area_of_tangent_segments_annulus_l264_264390


namespace coral_third_week_pages_l264_264942

theorem coral_third_week_pages :
  let total_pages := 600
  let week1_read := total_pages / 2
  let remaining_after_week1 := total_pages - week1_read
  let week2_read := remaining_after_week1 * 0.30
  let remaining_after_week2 := remaining_after_week1 - week2_read
  remaining_after_week2 = 210 :=
by
  sorry

end coral_third_week_pages_l264_264942


namespace cost_of_camel_is_6000_l264_264336

noncomputable def cost_of_camel : ℕ := 6000

variables (C H O E : ℕ)
variables (cost_of_camel_rs cost_of_horses cost_of_oxen cost_of_elephants : ℕ)

-- Conditions
axiom cond1 : 10 * C = 24 * H
axiom cond2 : 16 * H = 4 * O
axiom cond3 : 6 * O = 4 * E
axiom cond4 : 10 * E = 150000

theorem cost_of_camel_is_6000
    (cond1 : 10 * C = 24 * H)
    (cond2 : 16 * H = 4 * O)
    (cond3 : 6 * O = 4 * E)
    (cond4 : 10 * E = 150000) :
  cost_of_camel = 6000 := 
sorry

end cost_of_camel_is_6000_l264_264336


namespace roots_of_polynomial_l264_264537

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l264_264537


namespace value_of_f_prime_at_1_l264_264971

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem value_of_f_prime_at_1 : deriv f 1 = 1 :=
by
  sorry

end value_of_f_prime_at_1_l264_264971


namespace A_inter_B_empty_iff_A_union_B_eq_B_iff_l264_264562

open Set

variable (a x : ℝ)

def A (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 1 + a}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem A_inter_B_empty_iff {a : ℝ} :
  (A a ∩ B = ∅) ↔ 0 ≤ a ∧ a ≤ 4 :=
by 
  sorry

theorem A_union_B_eq_B_iff {a : ℝ} :
  (A a ∪ B = B) ↔ a < -4 :=
by
  sorry

end A_inter_B_empty_iff_A_union_B_eq_B_iff_l264_264562


namespace hectares_per_day_initial_l264_264206

variable (x : ℝ) -- x is the number of hectares one tractor ploughs initially per day

-- Condition 1: A field can be ploughed by 6 tractors in 4 days.
def total_area_initial := 6 * x * 4

-- Condition 2: 6 tractors plough together a certain number of hectares per day, denoted as x hectares/day.
-- This is incorporated in the variable declaration of x.

-- Condition 3: If 2 tractors are moved to another field, the remaining 4 tractors can plough the same field in 5 days.
-- Condition 4: One of the 4 tractors ploughs 144 hectares a day when 4 tractors plough the field in 5 days.
def total_area_with_4_tractors := 4 * 144 * 5

-- The statement that equates the two total area expressions.
theorem hectares_per_day_initial : total_area_initial x = total_area_with_4_tractors := by
  sorry

end hectares_per_day_initial_l264_264206


namespace platform_length_is_correct_l264_264201

-- Given Definitions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 42
def time_to_cross_pole : ℝ := 18

-- Definition to prove
theorem platform_length_is_correct :
  ∃ L : ℝ, L = 400 ∧ (length_of_train + L) / time_to_cross_platform = length_of_train / time_to_cross_pole :=
by
  sorry

end platform_length_is_correct_l264_264201


namespace scalene_triangle_angle_difference_l264_264033

def scalene_triangle (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem scalene_triangle_angle_difference (x y : ℝ) :
  (x + y = 100) → scalene_triangle x y 80 → (x - y = 80) :=
by
  intros h1 h2
  sorry

end scalene_triangle_angle_difference_l264_264033


namespace car_speed_is_104_mph_l264_264481

noncomputable def speed_of_car_in_mph
  (fuel_efficiency_km_per_liter : ℝ) -- car travels 64 km per liter
  (fuel_consumption_gallons : ℝ) -- fuel tank decreases by 3.9 gallons
  (time_hours : ℝ) -- period of 5.7 hours
  (gallon_to_liter : ℝ) -- 1 gallon is 3.8 liters
  (km_to_mile : ℝ) -- 1 mile is 1.6 km
  : ℝ :=
  let fuel_consumption_liters := fuel_consumption_gallons * gallon_to_liter
  let distance_km := fuel_efficiency_km_per_liter * fuel_consumption_liters
  let distance_miles := distance_km / km_to_mile
  let speed_mph := distance_miles / time_hours
  speed_mph

theorem car_speed_is_104_mph 
  (fuel_efficiency_km_per_liter : ℝ := 64)
  (fuel_consumption_gallons : ℝ := 3.9)
  (time_hours : ℝ := 5.7)
  (gallon_to_liter : ℝ := 3.8)
  (km_to_mile : ℝ := 1.6)
  : speed_of_car_in_mph fuel_efficiency_km_per_liter fuel_consumption_gallons time_hours gallon_to_liter km_to_mile = 104 :=
  by
    sorry

end car_speed_is_104_mph_l264_264481


namespace percentage_less_than_a_plus_d_l264_264341

def symmetric_distribution (a d : ℝ) (p : ℝ) : Prop :=
  p = (68 / 100 : ℝ) ∧ 
  (p / 2) = (34 / 100 : ℝ)

theorem percentage_less_than_a_plus_d (a d : ℝ) 
  (symmetry : symmetric_distribution a d (68 / 100)) : 
  (0.5 + (34 / 100) : ℝ) = (84 / 100 : ℝ) :=
by
  sorry

end percentage_less_than_a_plus_d_l264_264341


namespace joe_height_is_82_l264_264128

-- Given the conditions:
def Sara_height (x : ℝ) : Prop := true

def Joe_height (j : ℝ) (x : ℝ) : Prop := j = 6 + 2 * x

def combined_height (j : ℝ) (x : ℝ) : Prop := j + x = 120

-- We need to prove:
theorem joe_height_is_82 (x j : ℝ) 
  (h1 : combined_height j x)
  (h2 : Joe_height j x) :
  j = 82 := 
by 
  sorry

end joe_height_is_82_l264_264128


namespace remainder_of_349_by_17_is_9_l264_264192

theorem remainder_of_349_by_17_is_9 :
  349 % 17 = 9 :=
sorry

end remainder_of_349_by_17_is_9_l264_264192


namespace count_integers_satisfying_inequality_l264_264712

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), S.card = 8 ∧ ∀ n ∈ S, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 :=
by
  sorry

end count_integers_satisfying_inequality_l264_264712


namespace nurses_count_l264_264017

theorem nurses_count (total personnel_ratio d_ratio n_ratio : ℕ)
  (ratio_eq: personnel_ratio = 280)
  (ratio_condition: d_ratio = 5)
  (person_count: n_ratio = 9) :
  n_ratio * (personnel_ratio / (d_ratio + n_ratio)) = 180 := by
  -- Total personnel = 280
  -- Ratio of doctors to nurses = 5/9
  -- Prove that the number of nurses is 180
  -- sorry is used to skip proof
  sorry

end nurses_count_l264_264017


namespace second_tap_empty_time_l264_264024

theorem second_tap_empty_time :
  ∃ T : ℝ, (1 / 4 - 1 / T = 3 / 28) → T = 7 :=
by
  sorry

end second_tap_empty_time_l264_264024


namespace length_LM_in_triangle_l264_264725

theorem length_LM_in_triangle 
  (A B C K L M : Type*) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K] [MetricSpace L] [MetricSpace M]
  (angle_A: Real) (angle_B: Real) (angle_C: Real)
  (AK: Real) (BL: Real) (MC: Real) (KL: Real) (KM: Real)
  (H1: angle_A = 90) (H2: angle_B = 30) (H3: angle_C = 60) 
  (H4: AK = 4) (H5: BL = 31) (H6: MC = 3) 
  (H7: KL = KM) : 
  (LM = 20) :=
sorry

end length_LM_in_triangle_l264_264725


namespace children_without_candies_l264_264631

/-- There are 73 children standing in a circle. An evil Santa Claus walks around 
    the circle in a clockwise direction and distributes candies. First, he gives one candy 
    to the first child, then skips 1 child, gives one candy to the next child, 
    skips 2 children, gives one candy to the next child, skips 3 children, and so on.
    
    After distributing 2020 candies, he leaves. 
    
    This theorem states that the number of children who did not receive any candies 
    is 36. -/
theorem children_without_candies : 
  let n := 73
  let a : ℕ → ℕ := λk, (k * (k + 1) / 2) % n
  ∃ m : ℕ, (distributed_positions m 2020 73 = 37) → (73 - 37) = 36
  sorry

end children_without_candies_l264_264631


namespace tiles_per_row_l264_264292

theorem tiles_per_row (area : ℝ) (tile_length : ℝ) (h1 : area = 256) (h2 : tile_length = 2/3) : 
  (16 * 12) / (8) = 24 :=
by {
  sorry
}

end tiles_per_row_l264_264292


namespace gambler_final_amount_l264_264487

-- Define initial amount of money
def initial_amount := 100

-- Define the multipliers
def win_multiplier := 4 / 3
def loss_multiplier := 2 / 3
def double_win_multiplier := 5 / 3

-- Define the gambler scenario (WWLWLWLW)
def scenario := [double_win_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier, loss_multiplier, win_multiplier]

-- Function to compute final amount given initial amount, number of wins and losses, and the scenario
def final_amount (initial: ℚ) (multipliers: List ℚ) : ℚ :=
  multipliers.foldl (· * ·) initial

-- Prove that the final amount after all multipliers are applied is approximately equal to 312.12
theorem gambler_final_amount : abs (final_amount initial_amount scenario - 312.12) < 0.01 :=
by
  sorry

end gambler_final_amount_l264_264487


namespace distance_between_trees_l264_264412

theorem distance_between_trees (l : ℕ) (n : ℕ) (d : ℕ) (h_length : l = 225) (h_trees : n = 26) (h_segments : n - 1 = 25) : d = 9 :=
sorry

end distance_between_trees_l264_264412


namespace factor_expression_l264_264526

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l264_264526


namespace residue_7_1234_mod_13_l264_264326

theorem residue_7_1234_mod_13 : (7^1234 : ℕ) % 13 = 4 :=
by
  have h1: 7 % 13 = 7 := rfl
  have h2: (7^2) % 13 = 10 := by norm_num
  have h3: (7^3) % 13 = 5 := by norm_num
  have h4: (7^4) % 13 = 9 := by norm_num
  have h5: (7^5) % 13 = 11 := by norm_num
  have h6: (7^6) % 13 = 12 := by norm_num
  have h7: (7^7) % 13 = 6 := by norm_num
  have h8: (7^8) % 13 = 3 := by norm_num
  have h9: (7^9) % 13 = 8 := by norm_num
  have h10: (7^10) % 13 = 4 := by norm_num
  have h11: (7^11) % 13 = 2 := by norm_num
  have h12: (7^12) % 13 = 1 := by norm_num
  sorry

end residue_7_1234_mod_13_l264_264326


namespace arithmetic_sequence_solution_l264_264086

theorem arithmetic_sequence_solution
  (a b c : ℤ)
  (h1 : a + 1 = b - a)
  (h2 : b - a = c - b)
  (h3 : c - b = -9 - c) :
  b = -5 ∧ a * c = 21 :=
by sorry

end arithmetic_sequence_solution_l264_264086


namespace amelia_wins_l264_264508

noncomputable def ameliaWinsProbability (a b : ℚ) : ℚ :=
  let heads_heads := a * b
  let amelia_wins_first := a * (1 - b)
  let blaine_wins_first := (1 - a) * b
  let both_tails := (1 - a) * (1 - b)
  let geometric_sum := (both_tails : ℚ) / (1 - both_tails)
  let amelia_wins_subsequent := (1 - a) * geometric_sum * a
  amelia_wins_first + amelia_wins_subsequent

theorem amelia_wins (Ha : 3 / 7) (Hb : 1 / 4) :
  ameliaWinsProbability (3/7) (1/4) = 9 / 14 :=
by
  sorry

end amelia_wins_l264_264508


namespace count_factors_multiple_of_150_l264_264984

theorem count_factors_multiple_of_150 (n : ℕ) (h : n = 2^10 * 3^14 * 5^8) : 
  ∃ k, k = 980 ∧ ∀ d : ℕ, d ∣ n → 150 ∣ d → (d.factors.card = k) := sorry

end count_factors_multiple_of_150_l264_264984


namespace polygon_interior_angles_eq_360_l264_264551

theorem polygon_interior_angles_eq_360 (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
sorry

end polygon_interior_angles_eq_360_l264_264551


namespace fan_airflow_in_one_week_l264_264641

-- Define the conditions
def fan_airflow_per_second : ℕ := 10
def fan_working_minutes_per_day : ℕ := 10
def seconds_per_minute : ℕ := 60
def days_per_week : ℕ := 7

-- Define the proof problem
theorem fan_airflow_in_one_week : (fan_airflow_per_second * fan_working_minutes_per_day * seconds_per_minute * days_per_week = 42000) := 
by sorry

end fan_airflow_in_one_week_l264_264641


namespace sum_of_consecutive_integers_with_product_272_l264_264182

theorem sum_of_consecutive_integers_with_product_272 :
    ∃ (x y : ℕ), x * y = 272 ∧ y = x + 1 ∧ x + y = 33 :=
by
  sorry

end sum_of_consecutive_integers_with_product_272_l264_264182


namespace extremum_point_iff_nonnegative_condition_l264_264972

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + x) - (a * x) / (x + 1)

theorem extremum_point_iff (a : ℝ) (h : 0 < a) :
  (∃ (x : ℝ), x = 1 ∧ ∀ (f' : ℝ), f' = (1 + x - a) / (x + 1)^2 ∧ f' = 0) ↔ a = 2 :=
by
  sorry

theorem nonnegative_condition (a : ℝ) (h0 : 0 < a) :
  (∀ (x : ℝ), x ∈ Set.Ici 0 → f a x ≥ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end extremum_point_iff_nonnegative_condition_l264_264972


namespace possible_values_expression_l264_264963

-- Defining the main expression 
def main_expression (a b c d : ℝ) : ℝ :=
  (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)

-- The theorem that we need to prove
theorem possible_values_expression (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  main_expression a b c d ∈ {5, 1, -3} :=
sorry

end possible_values_expression_l264_264963


namespace combination_20_6_l264_264939

theorem combination_20_6 : Nat.choose 20 6 = 38760 :=
by
  sorry

end combination_20_6_l264_264939


namespace mabel_total_tomatoes_l264_264428

theorem mabel_total_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 :=
by
  sorry

end mabel_total_tomatoes_l264_264428


namespace failed_english_is_45_l264_264580

-- Definitions of the given conditions
def total_students : ℝ := 1 -- representing 100%
def failed_hindi : ℝ := 0.35
def failed_both : ℝ := 0.2
def passed_both : ℝ := 0.4

-- The goal is to prove that the percentage of students who failed in English is 45%

theorem failed_english_is_45 :
  let failed_at_least_one := total_students - passed_both
  let failed_english := failed_at_least_one - failed_hindi + failed_both
  failed_english = 0.45 :=
by
  -- The steps and manipulation will go here, but for now we skip with sorry
  sorry

end failed_english_is_45_l264_264580


namespace mutant_frog_percentage_proof_l264_264683

/-- Number of frogs with extra legs -/
def frogs_with_extra_legs := 5

/-- Number of frogs with 2 heads -/
def frogs_with_two_heads := 2

/-- Number of frogs that are bright red -/
def frogs_bright_red := 2

/-- Number of normal frogs -/
def normal_frogs := 18

/-- Total number of mutant frogs -/
def total_mutant_frogs := frogs_with_extra_legs + frogs_with_two_heads + frogs_bright_red

/-- Total number of frogs -/
def total_frogs := total_mutant_frogs + normal_frogs

/-- Calculate the percentage of mutant frogs rounded to the nearest integer -/
def mutant_frog_percentage : ℕ := (total_mutant_frogs * 100 / total_frogs).toNat

theorem mutant_frog_percentage_proof:
  mutant_frog_percentage = 33 := 
  by 
    -- Proof skipped
    sorry

end mutant_frog_percentage_proof_l264_264683


namespace intercept_sum_l264_264020

theorem intercept_sum (x y : ℤ) (h1 : 0 ≤ x) (h2 : x < 42) (h3 : 0 ≤ y) (h4 : y < 42)
  (h : 5 * x ≡ 3 * y - 2 [ZMOD 42]) : (x + y) = 36 :=
by
  sorry

end intercept_sum_l264_264020


namespace solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l264_264698

-- Define the function f(x) and g(x)
def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- Define the inequality problem when a = 2
theorem solution_set_for_f_when_a_2 : 
  { x : ℝ | f x 2 ≤ 6 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

-- Prove the range of values for a when f(x) + g(x) ≥ 3
theorem range_of_a_for_f_plus_g_ge_3 : 
  ∀ a : ℝ, (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a :=
by
  sorry

end solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l264_264698


namespace find_number_l264_264862

theorem find_number (N x : ℝ) (h : x = 9) (h1 : N - (5 / x) = 4 + (4 / x)) : N = 5 :=
by
  sorry

end find_number_l264_264862


namespace mabel_total_tomatoes_l264_264427

theorem mabel_total_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 :=
by
  sorry

end mabel_total_tomatoes_l264_264427


namespace total_animal_legs_l264_264758

theorem total_animal_legs (total_animals : ℕ) (sheep : ℕ) (chickens : ℕ) : 
  total_animals = 20 ∧ sheep = 10 ∧ chickens = 10 ∧ 
  2 * chickens + 4 * sheep = 60 :=
by 
  sorry

end total_animal_legs_l264_264758


namespace primes_p_p2_p4_l264_264623

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem primes_p_p2_p4 (p : ℕ) (hp : is_prime p) (hp2 : is_prime (p + 2)) (hp4 : is_prime (p + 4)) :
  p = 3 :=
sorry

end primes_p_p2_p4_l264_264623


namespace John_completes_work_alone_10_days_l264_264107

theorem John_completes_work_alone_10_days
  (R : ℕ)
  (T : ℕ)
  (W : ℕ)
  (H1 : R = 40)
  (H2 : T = 8)
  (H3 : 1/10 = (1/R) + (1/W))
  : W = 10 := sorry

end John_completes_work_alone_10_days_l264_264107


namespace polynomial_roots_l264_264546

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end polynomial_roots_l264_264546


namespace rate_of_interest_per_annum_l264_264911

def simple_interest (P T R : ℕ) : ℕ :=
  (P * T * R) / 100

theorem rate_of_interest_per_annum :
  let P_B := 5000
  let T_B := 2
  let P_C := 3000
  let T_C := 4
  let total_interest := 1980
  ∃ R : ℕ, 
      simple_interest P_B T_B R + simple_interest P_C T_C R = total_interest ∧
      R = 9 :=
by
  sorry

end rate_of_interest_per_annum_l264_264911


namespace no_always_1x3_rectangle_l264_264553

/-- From a sheet of graph paper measuring 8 x 8 cells, 12 rectangles of size 1 x 2 were cut out along the grid lines. 
Prove that it is not necessarily possible to always find a 1 x 3 checkered rectangle in the remaining part. -/
theorem no_always_1x3_rectangle (grid_size : ℕ) (rectangles_removed : ℕ) (rect_size : ℕ) :
  grid_size = 64 → rectangles_removed * rect_size = 24 → ¬ (∀ remaining_cells, remaining_cells ≥ 0 → remaining_cells ≤ 64 → ∃ (x y : ℕ), remaining_cells = x * y ∧ x = 1 ∧ y = 3) :=
  by
  intro h1 h2 h3
  /- Exact proof omitted for brevity -/
  sorry

end no_always_1x3_rectangle_l264_264553


namespace total_contribution_is_1040_l264_264038

-- Definitions of contributions based on conditions.
def Niraj_contribution : ℕ := 80
def Brittany_contribution : ℕ := 3 * Niraj_contribution
def Angela_contribution : ℕ := 3 * Brittany_contribution

-- Statement to prove that total contribution is $1040.
theorem total_contribution_is_1040 : Niraj_contribution + Brittany_contribution + Angela_contribution = 1040 := by
  sorry

end total_contribution_is_1040_l264_264038


namespace compute_expression_l264_264936

theorem compute_expression : 1004^2 - 996^2 - 1000^2 + 1000^2 = 16000 := 
by sorry

end compute_expression_l264_264936


namespace optionD_is_deductive_l264_264767

-- Conditions related to the reasoning options
inductive ReasoningProcess where
  | optionA : ReasoningProcess
  | optionB : ReasoningProcess
  | optionC : ReasoningProcess
  | optionD : ReasoningProcess

-- Definitions matching the equivalent Lean problem
def isDeductiveReasoning (rp : ReasoningProcess) : Prop :=
  match rp with
  | ReasoningProcess.optionA => False
  | ReasoningProcess.optionB => False
  | ReasoningProcess.optionC => False
  | ReasoningProcess.optionD => True

-- The proposition we need to prove
theorem optionD_is_deductive :
  isDeductiveReasoning ReasoningProcess.optionD = True := by
  sorry

end optionD_is_deductive_l264_264767


namespace complex_number_second_quadrant_l264_264228

theorem complex_number_second_quadrant 
  : (2 + 3 * Complex.I) / (1 - Complex.I) ∈ { z : Complex | z.re < 0 ∧ z.im > 0 } := 
by
  sorry

end complex_number_second_quadrant_l264_264228


namespace find_certain_number_l264_264863

theorem find_certain_number
  (t b c : ℝ)
  (average1 : (t + b + c + 14 + 15) / 5 = 12)
  (average2 : (t + b + c + x) / 4 = 15)
  (x : ℝ) :
  x = 29 :=
by
  sorry

end find_certain_number_l264_264863


namespace abs_algebraic_expression_l264_264854

theorem abs_algebraic_expression (x : ℝ) (h : |2 * x - 3| - 3 + 2 * x = 0) : |2 * x - 5| = 5 - 2 * x := 
by sorry

end abs_algebraic_expression_l264_264854


namespace simplify_trig_expression_l264_264746

theorem simplify_trig_expression :
  (Real.cos (72 * Real.pi / 180) * Real.sin (78 * Real.pi / 180) +
   Real.sin (72 * Real.pi / 180) * Real.sin (12 * Real.pi / 180) = 1 / 2) :=
by sorry

end simplify_trig_expression_l264_264746


namespace correct_81st_in_set_s_l264_264425

def is_in_set_s (x : ℕ) : Prop :=
  ∃ n : ℕ, x = 8 * n + 5

noncomputable def find_81st_in_set_s : ℕ :=
  8 * 80 + 5

theorem correct_81st_in_set_s : find_81st_in_set_s = 645 := by
  sorry

end correct_81st_in_set_s_l264_264425


namespace minimum_value_of_f_range_of_a_l264_264558

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x a : ℝ) := -x^2 + a * x - 3

theorem minimum_value_of_f :
  ∃ x_min : ℝ, ∀ x : ℝ, 0 < x → f x ≥ -1/Real.exp 1 := sorry -- This statement asserts that the minimum value of f(x) is -1/e.

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * f x ≥ g x a) → a ≤ 4 := sorry -- This statement asserts that if 2f(x) ≥ g(x) for all x > 0, then a is at most 4.

end minimum_value_of_f_range_of_a_l264_264558


namespace gcd_153_68_eq_17_l264_264516

theorem gcd_153_68_eq_17 : Int.gcd 153 68 = 17 :=
by
  sorry

end gcd_153_68_eq_17_l264_264516


namespace investor_should_choose_first_plan_l264_264509

noncomputable def plan1Dist : probability_distribution ℝ := 
  gaussian 8 3

noncomputable def plan2Dist : probability_distribution ℝ := 
  gaussian 6 2

theorem investor_should_choose_first_plan :
  (plan1Dist.prob (set.Ioi 5)) > (plan2Dist.prob (set.Ioi 5)) :=
by sorry

end investor_should_choose_first_plan_l264_264509


namespace rectangular_floor_length_l264_264645

theorem rectangular_floor_length
    (cost_per_square : ℝ)
    (total_cost : ℝ)
    (carpet_length : ℝ)
    (carpet_width : ℝ)
    (floor_width : ℝ)
    (floor_area : ℝ) 
    (H1 : cost_per_square = 15)
    (H2 : total_cost = 225)
    (H3 : carpet_length = 2)
    (H4 : carpet_width = 2)
    (H5 : floor_width = 6)
    (H6 : floor_area = floor_width * carpet_length * carpet_width * 15): 
    floor_area / floor_width = 10 :=
by
  sorry

end rectangular_floor_length_l264_264645


namespace correct_equation_for_t_l264_264371

-- Define the rates for Doug and Dave
def dougRate : ℝ := 1 / 5
def daveRate : ℝ := 1 / 7

-- Combined rate
def combinedRate : ℝ := dougRate + daveRate

-- Theorem to prove the correct equation for time t
theorem correct_equation_for_t (t : ℝ) : combinedRate * (t - 1) = 1 :=
by
  -- Proof omitted, as only the statement is required
  sorry

end correct_equation_for_t_l264_264371


namespace number_of_dimes_l264_264579

theorem number_of_dimes (k : ℕ) (dimes quarters : ℕ) (value : ℕ)
  (h1 : 3 * k = dimes)
  (h2 : 2 * k = quarters)
  (h3 : value = (10 * dimes) + (25 * quarters))
  (h4 : value = 400) :
  dimes = 15 :=
by {
  sorry
}

end number_of_dimes_l264_264579


namespace trig_expression_value_l264_264841

theorem trig_expression_value (θ : ℝ)
  (h1 : Real.sin (Real.pi + θ) = 1/4) :
  (Real.cos (Real.pi + θ) / (Real.cos θ * (Real.cos (Real.pi + θ) - 1)) + 
  Real.sin (Real.pi / 2 - θ) / (Real.cos (θ + 2 * Real.pi) * Real.cos (Real.pi + θ) + Real.cos (-θ))) = 32 :=
by
  sorry

end trig_expression_value_l264_264841


namespace suitable_survey_method_l264_264761

-- Definitions based on conditions
def large_population (n : ℕ) : Prop := n > 10000  -- Example threshold for large population
def impractical_comprehensive_survey : Prop := true  -- Given in condition

-- The statement of the problem
theorem suitable_survey_method (n : ℕ) (h1 : large_population n) (h2 : impractical_comprehensive_survey) : 
  ∃ method : String, method = "sampling survey" :=
sorry

end suitable_survey_method_l264_264761


namespace factorize_a3_minus_ab2_l264_264949

theorem factorize_a3_minus_ab2 (a b: ℝ) : 
  a^3 - a * b^2 = a * (a + b) * (a - b) :=
by
  sorry

end factorize_a3_minus_ab2_l264_264949


namespace correct_statement_four_l264_264035

variable {α : Type*} (A B S : Set α) (U : Set α)

theorem correct_statement_four (h1 : U = Set.univ) (h2 : A ∩ B = U) : A = U ∧ B = U := by
  sorry

end correct_statement_four_l264_264035


namespace percent_of_y_l264_264190

theorem percent_of_y (y : ℝ) : 0.30 * (0.80 * y) = 0.24 * y :=
by sorry

end percent_of_y_l264_264190


namespace quadrilateral_perimeter_l264_264624

noncomputable def EG (FH : ℝ) : ℝ := Real.sqrt ((FH + 5) ^ 2 + FH ^ 2)

theorem quadrilateral_perimeter 
  (EF FH GH : ℝ) 
  (h1 : EF = 12)
  (h2 : FH = 7)
  (h3 : GH = FH) :
  EF + FH + GH + EG FH = 26 + Real.sqrt 193 :=
by
  rw [h1, h2, h3]
  sorry

end quadrilateral_perimeter_l264_264624


namespace total_amount_paid_l264_264798

-- Definitions
def original_aquarium_price : ℝ := 120
def aquarium_discount : ℝ := 0.5
def aquarium_coupon : ℝ := 0.1
def aquarium_sales_tax : ℝ := 0.05

def plants_decorations_price_before_discount : ℝ := 75
def plants_decorations_discount : ℝ := 0.15
def plants_decorations_sales_tax : ℝ := 0.08

def fish_food_price : ℝ := 25
def fish_food_sales_tax : ℝ := 0.06

-- Final result to be proved
theorem total_amount_paid : 
  let discounted_aquarium_price := original_aquarium_price * (1 - aquarium_discount)
  let coupon_aquarium_price := discounted_aquarium_price * (1 - aquarium_coupon)
  let total_aquarium_price := coupon_aquarium_price * (1 + aquarium_sales_tax)
  let discounted_plants_decorations_price := plants_decorations_price_before_discount * (1 - plants_decorations_discount)
  let total_plants_decorations_price := discounted_plants_decorations_price * (1 + plants_decorations_sales_tax)
  let total_fish_food_price := fish_food_price * (1 + fish_food_sales_tax)
  total_aquarium_price + total_plants_decorations_price + total_fish_food_price = 152.05 :=
by 
  sorry

end total_amount_paid_l264_264798


namespace fraction_cubed_equality_l264_264044

-- Constants for the problem
def A : ℝ := 81000
def B : ℝ := 9000

-- Problem statement
theorem fraction_cubed_equality : (A^3) / (B^3) = 729 :=
by
  sorry

end fraction_cubed_equality_l264_264044


namespace average_speed_round_trip_l264_264782

theorem average_speed_round_trip (D T : ℝ) (h1 : D = 51 * T) : (2 * D) / (3 * T) = 34 := 
by
  sorry

end average_speed_round_trip_l264_264782


namespace f_is_odd_function_f_is_increasing_f_max_min_in_interval_l264_264244

variable {f : ℝ → ℝ}

-- The conditions:
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom positive_for_positive : ∀ x : ℝ, x > 0 → f x > 0
axiom f_one_is_two : f 1 = 2

-- The proof tasks:
theorem f_is_odd_function : ∀ x : ℝ, f (-x) = -f x := 
sorry

theorem f_is_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := 
sorry

theorem f_max_min_in_interval : 
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≤ 6) ∧ (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -6) :=
sorry

end f_is_odd_function_f_is_increasing_f_max_min_in_interval_l264_264244


namespace problem_statement_l264_264569

theorem problem_statement (a b c : ℝ)
  (h : a * b * c = ( Real.sqrt ( (a + 2) * (b + 3) ) ) / (c + 1)) :
  6 * 15 * 7 = 1.5 :=
sorry

end problem_statement_l264_264569


namespace find_y_l264_264753

-- Define the conditions (inversely proportional and sum condition)
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k
def sum_condition (x y : ℝ) : Prop := x + y = 50 ∧ x = 3 * y

-- Given these conditions, prove the value of y when x = -12
theorem find_y (k x y : ℝ)
  (h1 : inversely_proportional x y k)
  (h2 : sum_condition 37.5 12.5)
  (hx : x = -12) :
  y = -39.0625 :=
sorry

end find_y_l264_264753


namespace infinity_gcd_binom_l264_264733

theorem infinity_gcd_binom {k l : ℕ} : ∃ᶠ m in at_top, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 :=
sorry

end infinity_gcd_binom_l264_264733


namespace solve_fractional_equation_l264_264144

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l264_264144


namespace consumer_installment_credit_l264_264654

theorem consumer_installment_credit (C : ℝ) (A : ℝ) (h1 : A = 0.36 * C) 
    (h2 : 75 = A / 2) : C = 416.67 :=
by
  sorry

end consumer_installment_credit_l264_264654


namespace maximum_rectangle_area_l264_264241

variable (x y : ℝ)

def area (x y : ℝ) : ℝ :=
  x * y

def similarity_condition (x y : ℝ) : Prop :=
  (11 - x) / (y - 6) = 2

theorem maximum_rectangle_area :
  ∃ (x y : ℝ), similarity_condition x y ∧ area x y = 66 :=  by
  sorry

end maximum_rectangle_area_l264_264241


namespace reflection_matrix_over_vector_l264_264826

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l264_264826


namespace cost_of_fencing_l264_264184

theorem cost_of_fencing
  (length width : ℕ)
  (ratio : 3 * width = 2 * length ∧ length * width = 5766)
  (cost_per_meter_in_paise : ℕ := 50)
  : (cost_per_meter_in_paise / 100 : ℝ) * 2 * (length + width) = 155 := 
by
  -- definitions
  sorry

end cost_of_fencing_l264_264184


namespace matilda_jellybeans_l264_264596

/-- Suppose Matilda has half as many jellybeans as Matt.
    Suppose Matt has ten times as many jellybeans as Steve.
    Suppose Steve has 84 jellybeans.
    Then Matilda has 420 jellybeans. -/
theorem matilda_jellybeans
    (matilda_jellybeans : ℕ)
    (matt_jellybeans : ℕ)
    (steve_jellybeans : ℕ)
    (h1 : matilda_jellybeans = matt_jellybeans / 2)
    (h2 : matt_jellybeans = 10 * steve_jellybeans)
    (h3 : steve_jellybeans = 84) : matilda_jellybeans = 420 := 
sorry

end matilda_jellybeans_l264_264596


namespace fraction_power_simplification_l264_264043

theorem fraction_power_simplification:
  (81000/9000)^3 = 729 → (81000^3) / (9000^3) = 729 :=
by 
  intro h
  rw [<- h]
  sorry

end fraction_power_simplification_l264_264043


namespace jason_flames_per_minute_l264_264191

theorem jason_flames_per_minute :
  (∀ (t : ℕ), t % 15 = 0 -> (5 * (t / 15) = 20)) :=
sorry

end jason_flames_per_minute_l264_264191


namespace expectation_of_X_l264_264685

-- Conditions:
-- Defect rate of the batch of products is 0.05
def defect_rate : ℚ := 0.05

-- 5 items are randomly selected for quality inspection
def n : ℕ := 5

-- The probability of obtaining a qualified product in each trial
def P : ℚ := 1 - defect_rate

-- Question:
-- The random variable X, representing the number of qualified products, follows a binomial distribution.
-- Expectation of X
def expectation_X : ℚ := n * P

-- Prove that the mathematical expectation E(X) is equal to 4.75
theorem expectation_of_X :
  expectation_X = 4.75 := 
sorry

end expectation_of_X_l264_264685


namespace largest_intersection_value_l264_264047

theorem largest_intersection_value (b c d : ℝ) :
  ∀ x : ℝ, (x^7 - 12*x^6 + 44*x^5 - 24*x^4 + b*x^3 = c*x - d) → x ≤ 6 := sorry

end largest_intersection_value_l264_264047


namespace simplified_expression_correct_l264_264747

noncomputable def simplified_expression : ℝ := 0.3 * 0.8 + 0.1 * 0.5

theorem simplified_expression_correct : simplified_expression = 0.29 := by 
  sorry

end simplified_expression_correct_l264_264747


namespace solve_equation_l264_264136

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l264_264136


namespace ConvexPolygon_l264_264745

structure ConvexPolygon (n : ℕ) :=
  (vertices : Fin n → Point)
  (convex : ∀ (i j k : Fin n), ccw (vertices i) (vertices j) (vertices k))

def isRightAngle (x y z : Point) : Prop :=
  ∠ x y z = π / 2 -- assuming angle measure in radians

theorem ConvexPolygon.rectangle_of_four_right_angles {n : ℕ} (P : ConvexPolygon n)
  (h1 : n = 4) 
  (h2 : ∀(i : Fin 4), isRightAngle (P.vertices i) (P.vertices (i + 1) % 4) (P.vertices (i + 2) % 4)) :
  ∃ (a b c d : Point), P.vertices = ![a, b, c, d] ∧ isRectangle a b c d :=
sorry

end ConvexPolygon_l264_264745


namespace inequality_comparison_l264_264010

theorem inequality_comparison (x y : ℝ) (h : x ≠ y) : x^4 + y^4 > x^3 * y + x * y^3 :=
  sorry

end inequality_comparison_l264_264010


namespace ratio_c_div_d_l264_264337

theorem ratio_c_div_d (a b d : ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : d = 0.05 * a) (c : ℝ) (h4 : c = b / a) : c / d = 1 / 320 := 
sorry

end ratio_c_div_d_l264_264337


namespace discount_percentage_l264_264604

theorem discount_percentage (marked_price sale_price cost_price : ℝ) (gain1 gain2 : ℝ)
  (h1 : gain1 = 0.35)
  (h2 : gain2 = 0.215)
  (h3 : sale_price = 30)
  (h4 : cost_price = marked_price / (1 + gain1))
  (h5 : marked_price = cost_price * (1 + gain2)) :
  ((sale_price - marked_price) / sale_price) * 100 = 10.009 :=
sorry

end discount_percentage_l264_264604


namespace markup_is_correct_l264_264357

noncomputable def profit (S : ℝ) : ℝ := 0.12 * S
noncomputable def expenses (S : ℝ) : ℝ := 0.10 * S
noncomputable def cost (S : ℝ) : ℝ := S - (profit S + expenses S)
noncomputable def markup (S : ℝ) : ℝ :=
  ((S - cost S) / (cost S)) * 100

theorem markup_is_correct:
  markup 10 = 28.21 :=
by
  sorry

end markup_is_correct_l264_264357


namespace polynomial_root_sum_l264_264451

theorem polynomial_root_sum : 
  ∀ (r1 r2 r3 r4 : ℝ), 
  (r1^4 - r1 - 504 = 0) ∧ 
  (r2^4 - r2 - 504 = 0) ∧ 
  (r3^4 - r3 - 504 = 0) ∧ 
  (r4^4 - r4 - 504 = 0) → 
  r1^4 + r2^4 + r3^4 + r4^4 = 2016 := by
sorry

end polynomial_root_sum_l264_264451


namespace sum_series_eq_half_l264_264224

theorem sum_series_eq_half :
  ∑' n : ℕ, (3^(n+1) / (9^(n+1) - 1)) = 1/2 := 
sorry

end sum_series_eq_half_l264_264224


namespace minimum_value_f_l264_264454

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / 2 + 2 / (Real.sin x)

theorem minimum_value_f (x : ℝ) (h : 0 < x ∧ x ≤ Real.pi / 2) :
  ∃ y, (∀ z, 0 < z ∧ z ≤ Real.pi / 2 → f z ≥ y) ∧ y = 5 / 2 :=
sorry

end minimum_value_f_l264_264454


namespace sum_of_consecutive_integers_with_product_272_l264_264181

theorem sum_of_consecutive_integers_with_product_272 :
    ∃ (x y : ℕ), x * y = 272 ∧ y = x + 1 ∧ x + y = 33 :=
by
  sorry

end sum_of_consecutive_integers_with_product_272_l264_264181


namespace find_acute_angle_correct_l264_264845

noncomputable def find_acute_angle (θ : ℝ) : Prop :=
  ∀ (M N : ℝ × ℝ), 
  M = (-real.sqrt 3, real.sqrt 2) ∧ 
  N = (real.sqrt 2, -real.sqrt 3) ∧ 
  θ = real.arctan 1 → 
  θ = real.pi / 4

-- Statement only, no proof included.
theorem find_acute_angle_correct (θ : ℝ) : find_acute_angle θ := 
  sorry

end find_acute_angle_correct_l264_264845


namespace count_households_in_apartment_l264_264276

noncomputable def total_households 
  (houses_left : ℕ)
  (houses_right : ℕ)
  (floors_above : ℕ)
  (floors_below : ℕ) 
  (households_per_house : ℕ) : ℕ :=
(houses_left + houses_right) * (floors_above + floors_below) * households_per_house

theorem count_households_in_apartment : 
  ∀ (houses_left houses_right floors_above floors_below households_per_house : ℕ),
  houses_left = 1 →
  houses_right = 6 →
  floors_above = 1 →
  floors_below = 3 →
  households_per_house = 3 →
  total_households houses_left houses_right floors_above floors_below households_per_house = 105 :=
by
  intros houses_left houses_right floors_above floors_below households_per_house hl hr fa fb hh
  rw [hl, hr, fa, fb, hh]
  unfold total_households
  norm_num
  sorry

end count_households_in_apartment_l264_264276


namespace neg_p_necessary_but_not_sufficient_for_neg_q_l264_264269

variable (p q : Prop)

theorem neg_p_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) : 
  (¬p → ¬q) ∧ (¬q → ¬p) := 
sorry

end neg_p_necessary_but_not_sufficient_for_neg_q_l264_264269


namespace number_of_integer_values_for_a_l264_264966

theorem number_of_integer_values_for_a :
  (∃ (a : Int), ∃ (p q : Int), p * q = -12 ∧ p + q = a ∧ p ≠ q) →
  (∃ (n : Nat), n = 6) := by
  sorry

end number_of_integer_values_for_a_l264_264966


namespace four_circles_rectangle_l264_264629

open Set

theorem four_circles_rectangle 
(O S: EuclideanSpace ℝ n) 
(S1 S2 S3 S4 : Set (EuclideanSpace ℝ n))
(A1 A2 A3 A4 B1 B2 B3 B4 : EuclideanSpace ℝ n)
(hcircle: CircleProp O S)
(hcenters: ∀ (O1 O2 O3 O4 : EuclideanSpace ℝ n), O1 ∈ S1 → O2 ∈ S2 → O3 ∈ S3 → O4 ∈ S4 
→ SetOfCentersOnCircle O)
(hint1: A1 ∈ S1 ∧ A1 ∈ S2)
(hint2: A2 ∈ S2 ∧ A2 ∈ S3)
(hint3: A3 ∈ S3 ∧ A3 ∈ S4)
(hint4: A4 ∈ S4 ∧ A4 ∈ S1)
(hA_on_S: ∀ (A : EuclideanSpace ℝ n), A ∈ {A1, A2, A3, A4} → A ∈ S)
(hB_distinct: B1 ≠ B2 ∧ B2 ≠ B3 ∧ B3 ≠ B4 ∧ B4 ≠ B1 ∧ B1 ≠ B3 ∧ B2 ≠ B4)
(hB_inside: ∀ (B : EuclideanSpace ℝ n), B ∈ {B1, B2, B3, B4} → B ∈ interior (closure S)) :
Rectangle B1 B2 B3 B4 :=
sorry

end four_circles_rectangle_l264_264629


namespace minimum_draws_divisible_by_3_or_5_l264_264721

theorem minimum_draws_divisible_by_3_or_5 (n : ℕ) (h : n = 90) :
  ∃ k, k = 49 ∧ ∀ (draws : ℕ), draws < k → ¬ (∃ x, 1 ≤ x ∧ x ≤ n ∧ (x % 3 = 0 ∨ x % 5 = 0)) :=
by {
  sorry
}

end minimum_draws_divisible_by_3_or_5_l264_264721


namespace remainder_n_plus_1008_l264_264469

variable (n : ℕ)

theorem remainder_n_plus_1008 (h1 : n % 4 = 1) (h2 : n % 5 = 3) : (n + 1008) % 4 = 1 := by
  sorry

end remainder_n_plus_1008_l264_264469


namespace evaluate_polynomial_at_3_l264_264763

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

theorem evaluate_polynomial_at_3 : f 3 = 1 := by
  sorry

end evaluate_polynomial_at_3_l264_264763


namespace find_k_l264_264778

theorem find_k (k : ℕ) : 5 ^ k = 5 * 25^2 * 125^3 → k = 14 := by
  sorry

end find_k_l264_264778


namespace cyclist_final_speed_l264_264346

def u : ℝ := 16
def a : ℝ := 0.5
def t : ℕ := 7200

theorem cyclist_final_speed : 
  (u + a * t) * 3.6 = 13017.6 := by
  sorry

end cyclist_final_speed_l264_264346


namespace quadratic_equation_m_condition_l264_264853

theorem quadratic_equation_m_condition (m : ℝ) :
  (m + 1 ≠ 0) ↔ (m ≠ -1) :=
by sorry

end quadratic_equation_m_condition_l264_264853


namespace lana_trip_longer_by_25_percent_l264_264622

-- Define the dimensions of the rectangular field
def length_field : ℕ := 3
def width_field : ℕ := 1

-- Define Tom's path distance
def tom_path_distance : ℕ := length_field + width_field

-- Define Lana's path distance
def lana_path_distance : ℕ := 2 + 1 + 1 + 1

-- Define the percentage increase calculation
def percentage_increase (initial final : ℕ) : ℕ :=
  (final - initial) * 100 / initial

-- Define the theorem to be proven
theorem lana_trip_longer_by_25_percent :
  percentage_increase tom_path_distance lana_path_distance = 25 :=
by
  sorry

end lana_trip_longer_by_25_percent_l264_264622


namespace cost_of_song_book_l264_264106

-- Definitions of the constants:
def cost_of_flute : ℝ := 142.46
def cost_of_music_stand : ℝ := 8.89
def total_spent : ℝ := 158.35

-- Definition of the combined cost of the flute and music stand:
def combined_cost := cost_of_flute + cost_of_music_stand

-- The final theorem to prove that the cost of the song book is $7.00:
theorem cost_of_song_book : total_spent - combined_cost = 7.00 := by
  sorry

end cost_of_song_book_l264_264106


namespace fraction_sum_equals_decimal_l264_264226

theorem fraction_sum_equals_decimal : 
  (3 / 30 + 9 / 300 + 27 / 3000 = 0.139) :=
by sorry

end fraction_sum_equals_decimal_l264_264226


namespace physics_majors_consecutive_probability_l264_264889

open Nat

-- Define the total number of seats and the specific majors
def totalSeats : ℕ := 10
def mathMajors : ℕ := 4
def physicsMajors : ℕ := 3
def chemistryMajors : ℕ := 2
def biologyMajors : ℕ := 1

-- Assuming a round table configuration
def probabilityPhysicsMajorsConsecutive : ℚ :=
  (3 * (Nat.factorial (totalSeats - physicsMajors))) / (Nat.factorial (totalSeats - 1))

-- Declare the theorem
theorem physics_majors_consecutive_probability : 
  probabilityPhysicsMajorsConsecutive = 1 / 24 :=
by
  sorry

end physics_majors_consecutive_probability_l264_264889


namespace train_crosses_signal_pole_in_18_seconds_l264_264338

-- Define the given conditions
def train_length := 300  -- meters
def platform_length := 450  -- meters
def time_to_cross_platform := 45  -- seconds

-- Define the question and the correct answer
def time_to_cross_signal_pole := 18  -- seconds (this is what we need to prove)

-- Define the total distance the train covers when crossing the platform
def total_distance_crossing_platform := train_length + platform_length  -- meters

-- Define the speed of the train
def train_speed := total_distance_crossing_platform / time_to_cross_platform  -- meters per second

theorem train_crosses_signal_pole_in_18_seconds :
  300 / train_speed = time_to_cross_signal_pole :=
by
  -- train_speed is defined directly in terms of the given conditions
  unfold train_speed total_distance_crossing_platform train_length platform_length time_to_cross_platform
  sorry

end train_crosses_signal_pole_in_18_seconds_l264_264338


namespace find_a_l264_264273

open Set Real

-- Defining sets A and B, and the condition A ∩ B = {3}
def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- Mathematically equivalent proof statement
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
  sorry

end find_a_l264_264273


namespace reflection_matrix_l264_264823

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l264_264823


namespace B_can_win_with_initial_config_B_l264_264634

def initial_configuration_B := (6, 2, 1)

def A_starts_and_B_wins (config : (Nat × Nat × Nat)) : Prop := sorry

theorem B_can_win_with_initial_config_B : A_starts_and_B_wins initial_configuration_B :=
sorry

end B_can_win_with_initial_config_B_l264_264634


namespace trigonometric_identity_proof_l264_264967

theorem trigonometric_identity_proof 
  (α : ℝ) 
  (h1 : Real.tan (2 * α) = 3 / 4) 
  (h2 : α ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2))
  (h3 : ∃ x : ℝ, (Real.sin (x + 2) + Real.sin (α - x) - 2 * Real.sin α) = 0) : 
  Real.cos (2 * α) = -4 / 5 ∧ Real.tan (α / 2) = (1 - Real.sqrt 10) / 3 := 
sorry

end trigonometric_identity_proof_l264_264967


namespace find_m_n_sum_l264_264256

theorem find_m_n_sum (m n : ℕ) (hm : m > 1) (hn : n > 1) 
  (h : 2005^2 + m^2 = 2004^2 + n^2) : 
  m + n = 211 :=
sorry

end find_m_n_sum_l264_264256


namespace johns_money_l264_264500

theorem johns_money (total_money ali_less nada_more: ℤ) (h1: total_money = 67) 
  (h2: ali_less = -5) (h3: nada_more = 4): 
  ∃ (j: ℤ), (n: ℤ), ali_less = n - 5 ∧ nada_more = 4 * n ∧ total_money = n + (n - 5) + (4 * n) → j = 48 :=
by
  sorry

end johns_money_l264_264500


namespace adult_ticket_cost_is_16_l264_264351

-- Define the problem
def group_size := 6 + 10 -- Total number of people
def child_tickets := 6 -- Number of children
def adult_tickets := 10 -- Number of adults
def child_ticket_cost := 10 -- Cost per child ticket
def total_ticket_cost := 220 -- Total cost for all tickets

-- Prove the cost of an adult ticket
theorem adult_ticket_cost_is_16 : 
  (total_ticket_cost - (child_tickets * child_ticket_cost)) / adult_tickets = 16 := by
  sorry

end adult_ticket_cost_is_16_l264_264351


namespace arithmetic_sequence_problem_l264_264557

variable (a : ℕ → ℤ) -- defining the sequence {a_n}
variable (S : ℕ → ℤ) -- defining the sum of the first n terms S_n

theorem arithmetic_sequence_problem (m : ℕ) (h1 : m > 1) 
  (h2 : a (m - 1) + a (m + 1) - a m ^ 2 = 0) 
  (h3 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end arithmetic_sequence_problem_l264_264557


namespace inequality_solution_l264_264897

theorem inequality_solution (x : ℝ) (h : (x + 1) / 2 ≥ x / 3) : x ≥ -3 :=
by
  sorry

end inequality_solution_l264_264897


namespace reflection_matrix_is_correct_l264_264824

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l264_264824


namespace f_satisfies_equation_l264_264969

noncomputable def f (x : ℝ) : ℝ := (20 / 3) * x * (Real.sqrt (1 - x^2))

theorem f_satisfies_equation (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 2 * f (Real.sin x * -1) + 3 * f (Real.sin x) = 4 * Real.sin x * Real.cos x) →
  (∀ x ∈ Set.Icc (-Real.sqrt 2 / 2) (Real.sqrt 2 / 2), f x = (20 / 3) * x * (Real.sqrt (1 - x^2))) :=
by
  intro h
  sorry

end f_satisfies_equation_l264_264969


namespace erick_total_revenue_l264_264470

def lemon_price_increase := 4
def grape_price_increase := lemon_price_increase / 2
def original_lemon_price := 8
def original_grape_price := 7
def lemons_sold := 80
def grapes_sold := 140

def new_lemon_price := original_lemon_price + lemon_price_increase -- $12 per lemon
def new_grape_price := original_grape_price + grape_price_increase -- $9 per grape

def revenue_from_lemons := lemons_sold * new_lemon_price -- $960
def revenue_from_grapes := grapes_sold * new_grape_price -- $1260

def total_revenue := revenue_from_lemons + revenue_from_grapes

theorem erick_total_revenue : total_revenue = 2220 := by
  -- Skipping proof with sorry
  sorry

end erick_total_revenue_l264_264470


namespace ratio_of_c_and_d_l264_264991

theorem ratio_of_c_and_d
  (x y c d : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c)
  (h2 : 9 * y - 12 * x = d) :
  c / d = -2 / 3 := 
  sorry

end ratio_of_c_and_d_l264_264991


namespace total_weekly_water_consumption_l264_264879

-- Definitions coming from the conditions of the problem
def num_cows : Nat := 40
def water_per_cow_per_day : Nat := 80
def num_sheep : Nat := 10 * num_cows
def water_per_sheep_per_day : Nat := water_per_cow_per_day / 4
def days_in_week : Nat := 7

-- To prove statement: 
theorem total_weekly_water_consumption :
  let weekly_water_cow := water_per_cow_per_day * days_in_week
  let total_weekly_water_cows := weekly_water_cow * num_cows
  let daily_water_sheep := water_per_sheep_per_day
  let weekly_water_sheep := daily_water_sheep * days_in_week
  let total_weekly_water_sheep := weekly_water_sheep * num_sheep
  total_weekly_water_cows + total_weekly_water_sheep = 78400 := 
by
  sorry

end total_weekly_water_consumption_l264_264879


namespace gcd_360_504_l264_264952

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l264_264952


namespace r_minus_p_value_l264_264572

theorem r_minus_p_value (p q r : ℝ)
  (h₁ : (p + q) / 2 = 10)
  (h₂ : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end r_minus_p_value_l264_264572


namespace must_be_divisor_of_p_l264_264422

theorem must_be_divisor_of_p (p q r s : ℕ) (hpq : Nat.gcd p q = 40) (hqr : Nat.gcd q r = 50) 
  (hrs : Nat.gcd r s = 75) (hsp : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) : 17 ∣ p :=
sorry

end must_be_divisor_of_p_l264_264422


namespace total_cost_l264_264491

theorem total_cost (cost_pencil cost_pen : ℕ) 
(h1 : cost_pen = cost_pencil + 9) 
(h2 : cost_pencil = 2) : 
cost_pencil + cost_pen = 13 := 
by 
  -- Proof would go here 
  sorry

end total_cost_l264_264491


namespace no_neighboring_beads_same_color_probability_l264_264062

theorem no_neighboring_beads_same_color_probability : 
  let total_beads := 9
  let count_red := 4
  let count_white := 3
  let count_blue := 2
  let total_permutations := Nat.factorial total_beads / (Nat.factorial count_red * Nat.factorial count_white * Nat.factorial count_blue)
  ∃ valid_permutations : ℕ,
  valid_permutations = 100 ∧
  valid_permutations / total_permutations = 5 / 63 := by
  sorry

end no_neighboring_beads_same_color_probability_l264_264062


namespace correct_average_l264_264333

theorem correct_average (incorrect_avg : ℝ) (n : ℕ) (wrong_num correct_num : ℝ)
  (h_avg : incorrect_avg = 23)
  (h_n : n = 10)
  (h_wrong : wrong_num = 26)
  (h_correct : correct_num = 36) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 24 :=
by
  -- Proof goes here
  sorry

end correct_average_l264_264333


namespace solve_fractional_equation_l264_264154

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l264_264154


namespace product_of_integers_l264_264831

theorem product_of_integers :
  ∃ (a b c d e : ℤ),
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} =
      {-1, 5, 8, 9, 11, 12, 14, 18, 20, 24}) ∧
    a * b * c * d * e = -2002 :=
by {
  -- The statement formulation does not require a proof, hence here we end with sorry.
  sorry
}

end product_of_integers_l264_264831


namespace doubled_dimensions_new_volume_l264_264638

-- Define the original volume condition
def original_volume_condition (π r h : ℝ) : Prop := π * r^2 * h = 5

-- Define the new volume function after dimensions are doubled
def new_volume (π r h : ℝ) : ℝ := π * (2 * r)^2 * (2 * h)

-- The Lean statement for the proof problem 
theorem doubled_dimensions_new_volume (π r h : ℝ) (h_orig : original_volume_condition π r h) : 
  new_volume π r h = 40 :=
by 
  sorry

end doubled_dimensions_new_volume_l264_264638


namespace largest_four_digit_number_divisible_by_4_with_digit_sum_20_l264_264011

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def digit_sum_is_20 (n : ℕ) : Prop :=
  (n / 1000) + ((n % 1000) / 100) + ((n % 100) / 10) + (n % 10) = 20

theorem largest_four_digit_number_divisible_by_4_with_digit_sum_20 :
  ∃ n : ℕ, is_four_digit n ∧ is_divisible_by_4 n ∧ digit_sum_is_20 n ∧ ∀ m : ℕ, is_four_digit m ∧ is_divisible_by_4 m ∧ digit_sum_is_20 m → m ≤ n :=
  sorry

end largest_four_digit_number_divisible_by_4_with_digit_sum_20_l264_264011


namespace largest_square_area_l264_264040

theorem largest_square_area (XY YZ XZ : ℝ)
  (h1 : XZ^2 = XY^2 + YZ^2)
  (h2 : XY^2 + YZ^2 + XZ^2 = 450) :
  XZ^2 = 225 :=
by
  sorry

end largest_square_area_l264_264040


namespace candidate_lost_by_1650_votes_l264_264480

theorem candidate_lost_by_1650_votes (total_votes : ℕ) (pct_candidate : ℝ) (pct_rival : ℝ) : 
  total_votes = 5500 → 
  pct_candidate = 0.35 → 
  pct_rival = 0.65 → 
  ((pct_rival * total_votes) - (pct_candidate * total_votes)) = 1650 := 
by
  intros h1 h2 h3
  sorry

end candidate_lost_by_1650_votes_l264_264480


namespace eight_b_value_l264_264396

theorem eight_b_value (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 8 * b = 16 :=
by
  sorry

end eight_b_value_l264_264396


namespace investment_amount_l264_264774

theorem investment_amount (R T V : ℝ) (hT : T = 0.9 * R) (hV : V = 0.99 * R) (total_sum : R + T + V = 6936) : R = 2400 :=
by sorry

end investment_amount_l264_264774


namespace parabola_x_intercepts_count_l264_264251

theorem parabola_x_intercepts_count :
  let a := -3
  let b := 4
  let c := -1
  let discriminant := b ^ 2 - 4 * a * c
  discriminant ≥ 0 →
  let num_roots := if discriminant > 0 then 2 else if discriminant = 0 then 1 else 0
  num_roots = 2 := 
by {
  sorry
}

end parabola_x_intercepts_count_l264_264251


namespace calculate_gross_income_l264_264892
noncomputable def gross_income (net_income : ℝ) (tax_rate : ℝ) : ℝ := net_income / (1 - tax_rate)

theorem calculate_gross_income : gross_income 20000 0.13 = 22989 :=
by
  sorry

end calculate_gross_income_l264_264892


namespace heartsuit_ratio_l264_264089

def heartsuit (n m : ℕ) : ℕ := n^4 * m^3

theorem heartsuit_ratio :
  (heartsuit 2 4) / (heartsuit 4 2) = 1 / 2 := by
  sorry

end heartsuit_ratio_l264_264089


namespace possible_values_of_expression_l264_264962

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (v : ℤ), v ∈ ({5, 1, -3, -5} : Set ℤ) ∧ v = (Int.sign a + Int.sign b + Int.sign c + Int.sign d + Int.sign (a * b * c * d)) :=
by
  sorry

end possible_values_of_expression_l264_264962


namespace hexagon_side_lengths_l264_264802

theorem hexagon_side_lengths (n m : ℕ) (AB BC : ℕ) (P : ℕ) :
  n + m = 6 ∧ n * 4 + m * 7 = 38 ∧ AB = 4 ∧ BC = 7 → m = 4 :=
by
  sorry

end hexagon_side_lengths_l264_264802


namespace factor_expression_l264_264521

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l264_264521


namespace mike_total_spent_on_toys_l264_264598

theorem mike_total_spent_on_toys :
  let marbles := 9.05
  let football := 4.95
  let baseball := 6.52
  marbles + football + baseball = 20.52 :=
by
  sorry

end mike_total_spent_on_toys_l264_264598


namespace range_of_sum_l264_264672

theorem range_of_sum (a b c d : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
sorry

end range_of_sum_l264_264672


namespace ellipse_is_correct_l264_264794

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = -1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 16) = 1

-- Define the conditions
def ellipse_focus_vertex_of_hyperbola_vertex_and_focus (x y : ℝ) : Prop :=
  hyperbola_eq x y ∧ ellipse_eq x y

-- Theorem stating that the ellipse equation holds given the conditions
theorem ellipse_is_correct :
  ∀ (x y : ℝ), ellipse_focus_vertex_of_hyperbola_vertex_and_focus x y →
  ellipse_eq x y := by
  intros x y h
  sorry

end ellipse_is_correct_l264_264794


namespace quotient_when_divided_by_44_l264_264923

theorem quotient_when_divided_by_44 :
  ∃ N Q : ℕ, (N % 44 = 0) ∧ (N % 39 = 15) ∧ (N / 44 = Q) ∧ (Q = 3) :=
by {
  sorry
}

end quotient_when_divided_by_44_l264_264923


namespace ratio_of_areas_is_two_thirds_l264_264869

noncomputable def PQ := 10
noncomputable def PR := 6
noncomputable def QR := 4
noncomputable def r_PQ := PQ / 2
noncomputable def r_PR := PR / 2
noncomputable def r_QR := QR / 2
noncomputable def area_semi_PQ := (1 / 2) * Real.pi * r_PQ^2
noncomputable def area_semi_PR := (1 / 2) * Real.pi * r_PR^2
noncomputable def area_semi_QR := (1 / 2) * Real.pi * r_QR^2
noncomputable def shaded_area := (area_semi_PQ - area_semi_PR) + area_semi_QR
noncomputable def total_area_circle := Real.pi * r_PQ^2
noncomputable def unshaded_area := total_area_circle - shaded_area
noncomputable def ratio := shaded_area / unshaded_area

theorem ratio_of_areas_is_two_thirds : ratio = 2 / 3 := by
  sorry

end ratio_of_areas_is_two_thirds_l264_264869


namespace total_number_of_drivers_l264_264460

theorem total_number_of_drivers (N : ℕ) (A_drivers : ℕ) (B_sample : ℕ) (C_sample : ℕ) (D_sample : ℕ)
  (A_sample : ℕ)
  (hA : A_drivers = 96)
  (hA_sample : A_sample = 12)
  (hB_sample : B_sample = 21)
  (hC_sample : C_sample = 25)
  (hD_sample : D_sample = 43) :
  N = 808 :=
by
  -- skipping the proof here
  sorry

end total_number_of_drivers_l264_264460


namespace carpet_dimensions_l264_264930

theorem carpet_dimensions
  (x y q : ℕ)
  (h_dim : y = 2 * x)
  (h_room1 : ((q^2 + 50^2) = (q * 2 - 50)^2 + (50 * 2 - q)^2))
  (h_room2 : ((q^2 + 38^2) = (q * 2 - 38)^2 + (38 * 2 - q)^2)) :
  x = 25 ∧ y = 50 :=
sorry

end carpet_dimensions_l264_264930


namespace average_income_N_O_l264_264608

variable (M N O : ℝ)

-- Condition declaration
def condition1 : Prop := M + N = 10100
def condition2 : Prop := M + O = 10400
def condition3 : Prop := M = 4000

-- Theorem statement
theorem average_income_N_O (h1 : condition1 M N) (h2 : condition2 M O) (h3 : condition3 M) :
  (N + O) / 2 = 6250 :=
sorry

end average_income_N_O_l264_264608


namespace eighth_triangular_number_l264_264445

def triangular_number (n: ℕ) : ℕ := n * (n + 1) / 2

theorem eighth_triangular_number : triangular_number 8 = 36 :=
by
  -- Proof here
  sorry

end eighth_triangular_number_l264_264445


namespace mass_percentage_Na_in_NaClO_l264_264550

theorem mass_percentage_Na_in_NaClO :
  let mass_Na : ℝ := 22.99
  let mass_Cl : ℝ := 35.45
  let mass_O : ℝ := 16.00
  let mass_NaClO : ℝ := mass_Na + mass_Cl + mass_O
  (mass_Na / mass_NaClO) * 100 = 30.89 := by
sorry

end mass_percentage_Na_in_NaClO_l264_264550


namespace jerry_can_throw_things_l264_264741

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25
def office_points_threshold : ℕ := 100
def interruptions : ℕ := 2
def insults : ℕ := 4

theorem jerry_can_throw_things : 
  (office_points_threshold - (points_for_interrupting * interruptions + points_for_insulting * insults)) / points_for_throwing = 2 :=
by 
  sorry

end jerry_can_throw_things_l264_264741


namespace burpees_percentage_contribution_l264_264519

theorem burpees_percentage_contribution :
  let total_time : ℝ := 20
  let jumping_jacks : ℝ := 30
  let pushups : ℝ := 22
  let situps : ℝ := 45
  let burpees : ℝ := 15
  let lunges : ℝ := 25

  let jumping_jacks_rate := jumping_jacks / total_time
  let pushups_rate := pushups / total_time
  let situps_rate := situps / total_time
  let burpees_rate := burpees / total_time
  let lunges_rate := lunges / total_time

  let total_rate := jumping_jacks_rate + pushups_rate + situps_rate + burpees_rate + lunges_rate

  (burpees_rate / total_rate) * 100 = 10.95 :=
by
  sorry

end burpees_percentage_contribution_l264_264519


namespace function_monotonicity_l264_264876

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  (a^x) / (b^x + c^x) + (b^x) / (a^x + c^x) + (c^x) / (a^x + b^x)

theorem function_monotonicity (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f a b c x ≤ f a b c y) ∧
  (∀ x y : ℝ, y ≤ x → x < 0 → f a b c x ≤ f a b c y) :=
by
  sorry

end function_monotonicity_l264_264876


namespace committee_member_count_l264_264311

theorem committee_member_count (n : ℕ) (M : ℕ) (Q : ℚ) 
  (h₁ : M = 6) 
  (h₂ : 2 * n = M) 
  (h₃ : Q = 0.4) 
  (h₄ : Q = (n - 1) / (M - 1)) : 
  n = 3 :=
by
  sorry

end committee_member_count_l264_264311


namespace solution_set_of_f_neg_2x_l264_264697

def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem solution_set_of_f_neg_2x (a b : ℝ) (hf_sol : ∀ x : ℝ, (a * x - 1) * (x + b) > 0 ↔ -1 < x ∧ x < 3) :
  ∀ x : ℝ, f a b (-2 * x) < 0 ↔ (x < -3/2 ∨ x > 1/2) :=
by
  sorry

end solution_set_of_f_neg_2x_l264_264697


namespace equality_holds_iff_l264_264102

theorem equality_holds_iff (k t x y z : ℤ) (h_arith_prog : x + z = 2 * y) :
  (k * y^3 = x^3 + z^3) ↔ (k = 2 * (3 * t^2 + 1)) := by
  sorry

end equality_holds_iff_l264_264102


namespace probability_of_same_type_l264_264948

-- Definitions for the given conditions
def total_books : ℕ := 12 + 9
def novels : ℕ := 12
def biographies : ℕ := 9

-- Define the number of ways to pick any two books
def total_ways_to_pick_two_books : ℕ := Nat.choose total_books 2

-- Define the number of ways to pick two novels
def ways_to_pick_two_novels : ℕ := Nat.choose novels 2

-- Define the number of ways to pick two biographies
def ways_to_pick_two_biographies : ℕ := Nat.choose biographies 2

-- Define the number of ways to pick two books of the same type
def ways_to_pick_two_books_of_same_type : ℕ := ways_to_pick_two_novels + ways_to_pick_two_biographies

-- Calculate the probability
noncomputable def probability_same_type (total_ways ways_same_type : ℕ) : ℚ :=
  ways_same_type / total_ways

theorem probability_of_same_type :
  probability_same_type total_ways_to_pick_two_books ways_to_pick_two_books_of_same_type = 17 / 35 := by
  sorry

end probability_of_same_type_l264_264948


namespace solve_for_x_l264_264567

theorem solve_for_x : ∀ (x : ℝ), (2 * x + 3) / 5 = 11 → x = 26 :=
by {
  sorry
}

end solve_for_x_l264_264567


namespace k_range_l264_264393

noncomputable def range_of_k (k : ℝ): Prop :=
  ∀ x : ℤ, (x - 2) * (x + 1) > 0 ∧ (2 * x + 5) * (x + k) < 0 → x = -2

theorem k_range:
  (∃ k : ℝ, range_of_k k) ↔ -3 ≤ k ∧ k < 2 :=
by
  sorry

end k_range_l264_264393


namespace find_number_l264_264989

theorem find_number (N p q : ℝ) (h₁ : N / p = 8) (h₂ : N / q = 18) (h₃ : p - q = 0.2777777777777778) : N = 4 :=
sorry

end find_number_l264_264989


namespace regina_earnings_l264_264127

-- Definitions based on conditions
def num_cows := 20
def num_pigs := 4 * num_cows
def price_per_pig := 400
def price_per_cow := 800

-- Total earnings calculation based on definitions
def earnings_from_cows := num_cows * price_per_cow
def earnings_from_pigs := num_pigs * price_per_pig
def total_earnings := earnings_from_cows + earnings_from_pigs

-- Proof statement
theorem regina_earnings : total_earnings = 48000 := by
  sorry

end regina_earnings_l264_264127


namespace proof_of_problem_l264_264114

noncomputable def problem_statement (a b c x y z : ℝ) : Prop :=
  23 * x + b * y + c * z = 0 ∧
  a * x + 33 * y + c * z = 0 ∧
  a * x + b * y + 52 * z = 0 ∧
  a ≠ 23 ∧
  x ≠ 0 →
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1

theorem proof_of_problem (a b c x y z : ℝ) (h : problem_statement a b c x y z) : 
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1 :=
sorry

end proof_of_problem_l264_264114


namespace parabola_fixed_point_thm_l264_264700

-- Define the parabola condition
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

-- Define the focus condition
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the slope product condition
def slope_product (A B : ℝ × ℝ) : Prop :=
  (A.1 ≠ 0 ∧ B.1 ≠ 0) → ((A.2 / A.1) * (B.2 / B.1) = -1 / 3)

-- Define the fixed point condition
def fixed_point (A B : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, A ≠ B ∧ (x = 12) ∧ ((A.2 - B.2) / (A.1 - B.1)) * 12 = A.2

-- Problem statement in Lean
theorem parabola_fixed_point_thm (A B : ℝ × ℝ) (p : ℝ) :
  (∃ O : ℝ × ℝ, O = (0, 0)) →
  (∃ C : ℝ → ℝ → ℝ → Prop, C = parabola) →
  (∃ F : ℝ × ℝ, focus F) →
  parabola A.2 A.1 p →
  parabola B.2 B.1 p →
  slope_product A B →
  fixed_point A B :=
by 
-- Sorry is used to skip the proof
sorry

end parabola_fixed_point_thm_l264_264700


namespace sum_MN_MK_eq_14_sqrt4_3_l264_264482

theorem sum_MN_MK_eq_14_sqrt4_3
  (MN MK : ℝ)
  (area: ℝ)
  (angle_LMN : ℝ)
  (h_area : area = 49)
  (h_angle_LMN : angle_LMN = 30) :
  MN + MK = 14 * (Real.sqrt (Real.sqrt 3)) :=
by
  sorry

end sum_MN_MK_eq_14_sqrt4_3_l264_264482


namespace johns_age_fraction_l264_264109

theorem johns_age_fraction (F M J : ℕ) 
  (hF : F = 40) 
  (hFM : F = M + 4) 
  (hJM : J = M - 16) : 
  J / F = 1 / 2 := 
by
  -- We don't need to fill in the proof, adding sorry to skip it
  sorry

end johns_age_fraction_l264_264109


namespace width_of_each_glass_pane_l264_264496

noncomputable def width_of_pane (num_panes : ℕ) (total_area : ℝ) (length_of_pane : ℝ) : ℝ :=
  total_area / num_panes / length_of_pane

theorem width_of_each_glass_pane :
  width_of_pane 8 768 12 = 8 := by
  sorry

end width_of_each_glass_pane_l264_264496


namespace probability_weight_range_l264_264068

open ProbabilityTheory

variable {X : Type}
variable [MeasureSpace X]

-- Given normal distribution with μ = 50 and σ = 0.1
def normal_distribution : Measure X := sorry

noncomputable def P_49_9_50_1 : ℝ := 0.6826
noncomputable def P_49_8_50_2 : ℝ := 0.9544

theorem probability_weight_range :
  (μ = 50) → (σ = 0.1) → 
  (P(49.9 < X ∧ X < 50.1) = 0.6826) → 
  (P(49.8 < X ∧ X < 50.2) = 0.9544) →
  P(49.8 < X ∧ X < 50.1) = 0.8185 :=
by
  intros
  sorry

end probability_weight_range_l264_264068


namespace polar_coordinates_of_point_l264_264669

theorem polar_coordinates_of_point :
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  r = 4 ∧ theta = Real.pi / 3 :=
by
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  have h_r : r = 4 := by {
    -- Calculation for r
    sorry
  }
  have h_theta : theta = Real.pi / 3 := by {
    -- Calculation for theta
    sorry
  }
  exact ⟨h_r, h_theta⟩

end polar_coordinates_of_point_l264_264669


namespace domain_of_f_l264_264170

def domain_f (x : ℝ) : Prop :=
  2 * x - 3 ≥ 0 ∧ x ≠ 3

def domain_set : Set ℝ :=
  { x | (3 / 2) ≤ x ∧ x < 3 ∨ 3 < x }

theorem domain_of_f :
  { x : ℝ | domain_f x } = domain_set := by
  sorry

end domain_of_f_l264_264170


namespace can_form_sets_l264_264016

def clearly_defined (s : Set α) : Prop := ∀ x ∈ s, True
def not_clearly_defined (s : Set α) : Prop := ¬clearly_defined s

def cubes := {x : Type | True} -- Placeholder for the actual definition
def major_supermarkets := {x : Type | True} -- Placeholder for the actual definition
def difficult_math_problems := {x : Type | True} -- Placeholder for the actual definition
def famous_dancers := {x : Type | True} -- Placeholder for the actual definition
def products_2012 := {x : Type | True} -- Placeholder for the actual definition
def points_on_axes := {x : ℝ × ℝ | x.1 = 0 ∨ x.2 = 0}

theorem can_form_sets :
  (clearly_defined cubes) ∧
  (not_clearly_defined major_supermarkets) ∧
  (not_clearly_defined difficult_math_problems) ∧
  (not_clearly_defined famous_dancers) ∧
  (clearly_defined products_2012) ∧
  (clearly_defined points_on_axes) →
  True := 
by {
  -- Your proof goes here
  sorry
}

end can_form_sets_l264_264016


namespace max_value_of_m_l264_264554

theorem max_value_of_m (x m : ℝ) (h1 : x^2 - 4*x - 5 > 0) (h2 : x^2 - 2*x + 1 - m^2 > 0) (hm : m > 0) 
(hsuff : ∀ (x : ℝ), (x < -1 ∨ x > 5) → (x > m + 1 ∨ x < 1 - m)) : m ≤ 2 :=
sorry

end max_value_of_m_l264_264554


namespace square_side_length_l264_264246

theorem square_side_length (x : ℝ) 
  (h : x^2 = 6^2 + 8^2) : x = 10 := 
by sorry

end square_side_length_l264_264246


namespace milk_cartons_total_l264_264258

theorem milk_cartons_total (regular_milk soy_milk : ℝ) (h1 : regular_milk = 0.5) (h2 : soy_milk = 0.1) :
  regular_milk + soy_milk = 0.6 :=
by
  rw [h1, h2]
  norm_num

end milk_cartons_total_l264_264258


namespace back_seat_tickets_sold_l264_264025

def total_tickets : ℕ := 20000
def main_seat_price : ℕ := 55
def back_seat_price : ℕ := 45
def total_revenue : ℕ := 955000

theorem back_seat_tickets_sold :
  ∃ (M B : ℕ), 
    M + B = total_tickets ∧ 
    main_seat_price * M + back_seat_price * B = total_revenue ∧ 
    B = 14500 :=
by
  sorry

end back_seat_tickets_sold_l264_264025


namespace minimum_handshakes_l264_264651

def binom (n k : ℕ) : ℕ := n.choose k

theorem minimum_handshakes (n_A n_B k_A k_B : ℕ) (h1 : binom (n_A + n_B) 2 + n_A + n_B = 465)
  (h2 : n_A < n_B) (h3 : k_A = n_A) (h4 : k_B = n_B) : k_A = 15 :=
by sorry

end minimum_handshakes_l264_264651


namespace dap_equiv_48_dips_l264_264859

variables (dap dop dip : Type) [CommRing dap] [CommRing dop] [CommRing dip]

-- Define equivalences between daps, dops, and dips
def equivalence_dap_dop : dap ≃ₐ[dop] (dop →ₐ[dip] dap) := sorry
def equivalence_dop_dip : dop ≃ₐ[dip] (dip →ₐ[dap] dop) := sorry

-- Proportions given in the conditions
def prop1 (d : dap) (o : dop) : 5 * d = 4 * o := sorry
def prop2 (o : dop) (i : dip) : 3 * o = 8 * i := sorry

-- The proof statement
theorem dap_equiv_48_dips : ∀ (d : dap) (i : dip), (15 * d = 32 * i) → (d = 22.5 * i) := 
by
  intros
  sorry

end dap_equiv_48_dips_l264_264859


namespace initial_number_of_angelfish_l264_264368

/-- The initial number of fish in the tank. -/
def initial_total_fish (A : ℕ) := 94 + A + 89 + 58

/-- The remaining number of fish for each species after sale. -/
def remaining_fish (A : ℕ) := 64 + (A - 48) + 72 + 34

/-- Given: 
1. The total number of remaining fish in the tank is 198.
2. The initial number of fish for each species: 94 guppies, A angelfish, 89 tiger sharks, 58 Oscar fish.
3. The number of fish sold: 30 guppies, 48 angelfish, 17 tiger sharks, 24 Oscar fish.
Prove: The initial number of angelfish is 76. -/
theorem initial_number_of_angelfish (A : ℕ) (h : remaining_fish A = 198) : A = 76 :=
sorry

end initial_number_of_angelfish_l264_264368


namespace total_oranges_l264_264417

theorem total_oranges (joan_oranges : ℕ) (sara_oranges : ℕ) 
                      (h1 : joan_oranges = 37) 
                      (h2 : sara_oranges = 10) :
  joan_oranges + sara_oranges = 47 := by
  sorry

end total_oranges_l264_264417


namespace regina_earnings_l264_264124

def num_cows : ℕ := 20

def num_pigs (num_cows : ℕ) : ℕ := 4 * num_cows

def price_per_pig : ℕ := 400
def price_per_cow : ℕ := 800

def earnings (num_cows num_pigs price_per_cow price_per_pig : ℕ) : ℕ :=
  num_cows * price_per_cow + num_pigs * price_per_pig

theorem regina_earnings :
  earnings num_cows (num_pigs num_cows) price_per_cow price_per_pig = 48000 :=
by
  -- proof omitted
  sorry

end regina_earnings_l264_264124


namespace tan_cos_sin_fraction_l264_264961

theorem tan_cos_sin_fraction (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α + 2 * Real.sin α) / (Real.cos α - 3 * Real.sin α) = -1 / 2 := 
by
  sorry

end tan_cos_sin_fraction_l264_264961


namespace unique_solution_l264_264837
-- Import necessary mathematical library

-- Define mathematical statement
theorem unique_solution (N : ℕ) (hN: N > 0) :
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (m + (1 / 2 : ℝ) * (m + n - 1) * (m + n - 2) = N) :=
by {
  sorry
}

end unique_solution_l264_264837


namespace extra_men_needed_l264_264795

theorem extra_men_needed
  (total_length : ℕ) (total_days : ℕ) (initial_men : ℕ)
  (completed_days : ℕ) (completed_work : ℕ) (remaining_work : ℕ)
  (remaining_days : ℕ) (total_man_days_needed : ℕ)
  (number_of_men_needed : ℕ) (extra_men_needed : ℕ)
  (h1 : total_length = 10)
  (h2 : total_days = 60)
  (h3 : initial_men = 30)
  (h4 : completed_days = 20)
  (h5 : completed_work = 2)
  (h6 : remaining_work = total_length - completed_work)
  (h7 : remaining_days = total_days - completed_days)
  (h8 : total_man_days_needed = remaining_work * (completed_days * initial_men) / completed_work)
  (h9 : number_of_men_needed = total_man_days_needed / remaining_days)
  (h10 : extra_men_needed = number_of_men_needed - initial_men)
  : extra_men_needed = 30 :=
by sorry

end extra_men_needed_l264_264795


namespace solve_equation_l264_264134

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l264_264134


namespace find_13th_result_l264_264293

theorem find_13th_result
  (avg_25 : ℕ → ℕ)
  (avg_1_to_12 : ℕ → ℕ)
  (avg_14_to_25 : ℕ → ℕ)
  (h1 : avg_25 25 = 50)
  (h2 : avg_1_to_12 12 = 14)
  (h3 : avg_14_to_25 12 = 17) :
  ∃ (X : ℕ), X = 878 := sorry

end find_13th_result_l264_264293


namespace ptolemys_theorem_l264_264309

-- Definition of the variables describing the lengths of the sides and diagonals
variables {a b c d m n : ℝ}

-- We declare that they belong to a cyclic quadrilateral
def cyclic_quadrilateral (a b c d m n : ℝ) : Prop :=
∃ (A B C D : ℝ), 
  A + C = 180 ∧ 
  B + D = 180 ∧ 
  m = (A * C) ∧ 
  n = (B * D) ∧ 
  a = (A * B) ∧ 
  b = (B * C) ∧ 
  c = (C * D) ∧ 
  d = (D * A)

-- The theorem statement in Lean form
theorem ptolemys_theorem (h : cyclic_quadrilateral a b c d m n) : m * n = a * c + b * d :=
sorry

end ptolemys_theorem_l264_264309


namespace sin_cos_sixth_power_l264_264735

theorem sin_cos_sixth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1/2) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 :=
by
  sorry

end sin_cos_sixth_power_l264_264735


namespace arithmetic_sequence_probability_correct_l264_264957

noncomputable def arithmetic_sequence_probability : ℚ := 
  let total_ways := Nat.choose 5 3
  let arithmetic_sequences := 4
  (arithmetic_sequences : ℚ) / (total_ways : ℚ)

theorem arithmetic_sequence_probability_correct :
  arithmetic_sequence_probability = 0.4 := by
  unfold arithmetic_sequence_probability
  sorry

end arithmetic_sequence_probability_correct_l264_264957


namespace policeman_hats_difference_l264_264657

theorem policeman_hats_difference
  (hats_simpson : ℕ)
  (hats_obrien_now : ℕ)
  (hats_obrien_before : ℕ)
  (H : hats_simpson = 15)
  (H_hats_obrien_now : hats_obrien_now = 34)
  (H_hats_obrien_twice : hats_obrien_before = hats_obrien_now + 1) :
  hats_obrien_before - 2 * hats_simpson = 5 :=
by
  sorry

end policeman_hats_difference_l264_264657


namespace find_x2_plus_y2_l264_264252

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + 2 * x + 2 * y = 88) :
    x^2 + y^2 = 304 / 9 := sorry

end find_x2_plus_y2_l264_264252


namespace exists_n_good_not_n_add_1_good_l264_264873

-- Define the sum of digits function S
def S (k : ℕ) : ℕ := (k.digits 10).sum

-- Define what it means for a number to be n-good
def n_good (a n : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), (a_seq 0 = a) ∧ (∀ i : Fin n, a_seq i.succ = a_seq i - S (a_seq i))

-- Define the main theorem
theorem exists_n_good_not_n_add_1_good : ∀ n : ℕ, ∃ a : ℕ, n_good a n ∧ ¬n_good a (n + 1) :=
by
  sorry

end exists_n_good_not_n_add_1_good_l264_264873


namespace oil_bill_january_l264_264773

-- Define the problem in Lean
theorem oil_bill_january (F J : ℝ) (h1 : F / J = 3 / 2) (h2 : (F + 30) / J = 5 / 3) : J = 180 := 
sorry

end oil_bill_january_l264_264773


namespace possible_values_expression_l264_264964

theorem possible_values_expression (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ (x : ℝ), x ∈ {5, 1, -3} ∧ x = (a / |a| + b / |b| + c / |c| + d / |d| + (abcd / |abcd|)) :=
by
  sorry

end possible_values_expression_l264_264964


namespace solve_exponent_equation_l264_264379

theorem solve_exponent_equation (x y z : ℕ) :
  7^x + 1 = 3^y + 5^z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  sorry

end solve_exponent_equation_l264_264379


namespace find_larger_number_l264_264912

theorem find_larger_number (L S : ℤ) (h₁ : L - S = 1000) (h₂ : L = 10 * S + 10) : L = 1110 :=
sorry

end find_larger_number_l264_264912


namespace franks_age_l264_264387

variable (F G : ℕ)

def gabriel_younger_than_frank : Prop := G = F - 3
def total_age_is_seventeen : Prop := F + G = 17

theorem franks_age (h1 : gabriel_younger_than_frank F G) (h2 : total_age_is_seventeen F G) : F = 10 :=
by
  sorry

end franks_age_l264_264387


namespace train_speed_l264_264792

/-- Proof problem: Speed calculation of a train -/
theorem train_speed :
  ∀ (length : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ),
    length = 40 →
    time_seconds = 0.9999200063994881 →
    speed_kmph = (length / 1000) / (time_seconds / 3600) →
    speed_kmph = 144 :=
by
  intros length time_seconds speed_kmph h_length h_time_seconds h_speed_kmph
  rw [h_length, h_time_seconds] at h_speed_kmph
  -- sorry is used to skip the proof steps
  sorry 

end train_speed_l264_264792


namespace triangle_side_length_l264_264394

theorem triangle_side_length 
  (a b c : ℝ) 
  (cosA : ℝ) 
  (h1: a = Real.sqrt 5) 
  (h2: c = 2) 
  (h3: cosA = 2 / 3) 
  (h4: a^2 = b^2 + c^2 - 2 * b * c * cosA) : 
  b = 3 := 
by 
  sorry

end triangle_side_length_l264_264394


namespace apogee_reach_second_stage_model_engine_off_time_l264_264339

-- Given conditions
def altitudes := [(0, 0), (1, 24), (2, 96), (4, 386), (5, 514), (6, 616), (9, 850), (13, 994), (14, 1000), (16, 976), (19, 850), (24, 400)]
def second_stage_curve (x : ℝ) : ℝ := -6 * x^2 + 168 * x - 176

-- Proof problems
theorem apogee_reach : (14, 1000) ∈ altitudes :=
sorry  -- Need to prove the inclusion of the apogee point in the table

theorem second_stage_model : 
    second_stage_curve 14 = 1000 ∧ 
    second_stage_curve 16 = 976 ∧ 
    second_stage_curve 19 = 850 ∧ 
    ∃ n, n = 4 :=
sorry  -- Need to prove the analytical expression is correct and n = 4

theorem engine_off_time : 
    ∃ t : ℝ, t = 14 + 5 * Real.sqrt 6 ∧ second_stage_curve t = 100 :=
sorry  -- Need to prove the engine off time calculation

end apogee_reach_second_stage_model_engine_off_time_l264_264339


namespace present_cost_after_two_years_l264_264751

-- Defining variables and constants
def initial_cost : ℝ := 75
def inflation_rate : ℝ := 0.05
def first_year_increase1 : ℝ := 0.20
def first_year_decrease1 : ℝ := 0.20
def second_year_increase2 : ℝ := 0.30
def second_year_decrease2 : ℝ := 0.25

theorem present_cost_after_two_years : presents_cost = 77.40 :=
by
  let adjusted_initial_cost := initial_cost + (initial_cost * inflation_rate)
  let increased_cost_year1 := adjusted_initial_cost + (adjusted_initial_cost * first_year_increase1)
  let decreased_cost_year1 := increased_cost_year1 - (increased_cost_year1 * first_year_decrease1)
  let adjusted_cost_year1 := decreased_cost_year1 + (decreased_cost_year1 * inflation_rate)
  let increased_cost_year2 := adjusted_cost_year1 + (adjusted_cost_year1 * second_year_increase2)
  let decreased_cost_year2 := increased_cost_year2 - (increased_cost_year2 * second_year_decrease2)
  let presents_cost := decreased_cost_year2
  have h := (presents_cost : ℝ)
  have h := presents_cost
  sorry

end present_cost_after_two_years_l264_264751


namespace student_count_estimate_l264_264498

theorem student_count_estimate 
  (n : Nat) 
  (h1 : 80 ≤ n) 
  (h2 : 100 ≤ n) 
  (h3 : 20 * n = 8000) : 
  n = 400 := 
by 
  sorry

end student_count_estimate_l264_264498


namespace mike_notebooks_total_l264_264878

theorem mike_notebooks_total
  (red_notebooks : ℕ)
  (green_notebooks : ℕ)
  (blue_notebooks_cost : ℕ)
  (total_cost : ℕ)
  (red_cost : ℕ)
  (green_cost : ℕ)
  (blue_cost : ℕ)
  (h1 : red_notebooks = 3)
  (h2 : red_cost = 4)
  (h3 : green_notebooks = 2)
  (h4 : green_cost = 2)
  (h5 : total_cost = 37)
  (h6 : blue_cost = 3)
  (h7 : total_cost = red_notebooks * red_cost + green_notebooks * green_cost + blue_notebooks_cost) :
  (red_notebooks + green_notebooks + blue_notebooks_cost / blue_cost = 12) :=
by {
  sorry
}

end mike_notebooks_total_l264_264878


namespace bc_approx_A_l264_264455

theorem bc_approx_A (A B C D E : ℝ) 
    (hA : 0 < A ∧ A < 1) (hB : 0 < B ∧ B < 1) (hC : 0 < C ∧ C < 1)
    (hD : 0 < D ∧ D < 1) (hE : 1 < E ∧ E < 2)
    (hA_val : A = 0.2) (hB_val : B = 0.4) (hC_val : C = 0.6) (hD_val : D = 0.8) :
    abs (B * C - A) < abs (B * C - B) ∧ abs (B * C - A) < abs (B * C - C) ∧ abs (B * C - A) < abs (B * C - D) := 
by 
  sorry

end bc_approx_A_l264_264455


namespace triangle_angles_and_type_l264_264495

theorem triangle_angles_and_type
  (largest_angle : ℝ)
  (smallest_angle : ℝ)
  (middle_angle : ℝ)
  (h1 : largest_angle = 90)
  (h2 : largest_angle = 3 * smallest_angle)
  (h3 : largest_angle + smallest_angle + middle_angle = 180) :
  (largest_angle = 90 ∧ middle_angle = 60 ∧ smallest_angle = 30 ∧ largest_angle = 90) := by
  sorry

end triangle_angles_and_type_l264_264495


namespace find_x_l264_264999

-- Definition of the problem conditions
def angle_ABC : ℝ := 85
def angle_BAC : ℝ := 55
def sum_angles_triangle (a b c : ℝ) : Prop := a + b + c = 180
def corresponding_angle (a b : ℝ) : Prop := a = b
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90

-- The theorem to prove
theorem find_x :
  ∀ (x BCA : ℝ), sum_angles_triangle angle_ABC angle_BAC BCA ∧ corresponding_angle BCA 40 ∧ right_triangle_sum BCA x → x = 50 :=
by
  intros x BCA h
  sorry

end find_x_l264_264999


namespace odd_function_f_l264_264916

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_f (f_odd : ∀ x : ℝ, f (-x) = - f x)
                       (f_lt_0 : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = - x * (x + 1) :=
by
  sorry

end odd_function_f_l264_264916


namespace range_of_b_l264_264865

theorem range_of_b (b : ℝ) :
  (∃ (x y : ℝ), 
    0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3 ∧ 
    y = x + b ∧ (x - 2)^2 + (y - 3)^2 = 4) ↔ 
    1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3 :=
by
  sorry

end range_of_b_l264_264865


namespace child_wants_to_buy_3_toys_l264_264023

/- 
  Problem Statement:
  There are 10 toys, and the number of ways to select a certain number 
  of those toys in any order is 120. We need to find out how many toys 
  were selected.
-/

def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem child_wants_to_buy_3_toys :
  ∃ r : ℕ, r ≤ 10 ∧ comb 10 r = 120 :=
by
  use 3
  -- Here you would write the proof
  sorry

end child_wants_to_buy_3_toys_l264_264023


namespace jameson_badminton_medals_l264_264587

theorem jameson_badminton_medals (total_medals track_medals : ℕ) (swimming_medals : ℕ) :
  total_medals = 20 →
  track_medals = 5 →
  swimming_medals = 2 * track_medals →
  total_medals - (track_medals + swimming_medals) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end jameson_badminton_medals_l264_264587


namespace graph_passes_through_point_l264_264088

theorem graph_passes_through_point (a : ℝ) (x y : ℝ) (h : a < 0) : (1 - a)^0 - 1 = -1 :=
by
  sorry

end graph_passes_through_point_l264_264088


namespace balloon_count_correct_l264_264004

def gold_balloons : ℕ := 141
def black_balloons : ℕ := 150
def silver_balloons : ℕ := 2 * gold_balloons
def total_balloons : ℕ := gold_balloons + silver_balloons + black_balloons

theorem balloon_count_correct : total_balloons = 573 := by
  sorry

end balloon_count_correct_l264_264004


namespace alice_bob_task_l264_264220

theorem alice_bob_task (t : ℝ) (h₁ : 1/4 + 1/6 = 5/12) (h₂ : t - 1/2 ≠ 0) :
    (5/12) * (t - 1/2) = 1 :=
sorry

end alice_bob_task_l264_264220


namespace olivia_spent_89_l264_264461

-- Define initial and subsequent amounts
def initial_amount : ℕ := 100
def atm_amount : ℕ := 148
def after_supermarket : ℕ := 159

-- Total amount before supermarket
def total_before_supermarket : ℕ := initial_amount + atm_amount

-- Amount spent
def amount_spent : ℕ := total_before_supermarket - after_supermarket

-- Proof that Olivia spent 89 dollars
theorem olivia_spent_89 : amount_spent = 89 := sorry

end olivia_spent_89_l264_264461


namespace area_of_transformed_region_l264_264266

noncomputable def matrix_transformation_area (A : Matrix (Fin 2) (Fin 2) ℝ) (area_T : ℝ) : ℝ :=
  matrix.det A * area_T

theorem area_of_transformed_region : 
  let T_area := 15
  let A := Matrix.of_list 2 2 [[3, 4], [6, -2]]
  let T'_area := matrix_transformation_area A T_area
  T'_area = 450 :=
by
  let T_area := 15
  let A := Matrix.of_list 2 2 [[3, 4], [6, -2]]
  let T'_area := matrix_transformation_area A T_area
  show T'_area = 450
  sorry

end area_of_transformed_region_l264_264266


namespace calc_value_of_ab_bc_ca_l264_264590

theorem calc_value_of_ab_bc_ca (a b c : ℝ) (h1 : a + b + c = 35) (h2 : ab + bc + ca = 320) (h3 : abc = 600) : 
  (a + b) * (b + c) * (c + a) = 10600 := 
by sorry

end calc_value_of_ab_bc_ca_l264_264590


namespace find_minimum_value_l264_264236

open Real

noncomputable def g (x : ℝ) : ℝ := 
  x + x / (x^2 + 2) + x * (x + 3) / (x^2 + 3) + 3 * (x + 1) / (x * (x^2 + 3))

theorem find_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 4 := 
sorry

end find_minimum_value_l264_264236


namespace matilda_jellybeans_l264_264597

/-- Suppose Matilda has half as many jellybeans as Matt.
    Suppose Matt has ten times as many jellybeans as Steve.
    Suppose Steve has 84 jellybeans.
    Then Matilda has 420 jellybeans. -/
theorem matilda_jellybeans
    (matilda_jellybeans : ℕ)
    (matt_jellybeans : ℕ)
    (steve_jellybeans : ℕ)
    (h1 : matilda_jellybeans = matt_jellybeans / 2)
    (h2 : matt_jellybeans = 10 * steve_jellybeans)
    (h3 : steve_jellybeans = 84) : matilda_jellybeans = 420 := 
sorry

end matilda_jellybeans_l264_264597


namespace painting_time_equation_l264_264372

theorem painting_time_equation (t : ℝ) :
  let Doug_rate := (1 : ℝ) / 5
  let Dave_rate := (1 : ℝ) / 7
  let combined_rate := Doug_rate + Dave_rate
  (combined_rate * (t - 1) = 1) :=
sorry

end painting_time_equation_l264_264372


namespace regina_earnings_l264_264126

-- Definitions based on conditions
def num_cows := 20
def num_pigs := 4 * num_cows
def price_per_pig := 400
def price_per_cow := 800

-- Total earnings calculation based on definitions
def earnings_from_cows := num_cows * price_per_cow
def earnings_from_pigs := num_pigs * price_per_pig
def total_earnings := earnings_from_cows + earnings_from_pigs

-- Proof statement
theorem regina_earnings : total_earnings = 48000 := by
  sorry

end regina_earnings_l264_264126


namespace solve_fractional_equation_l264_264151

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l264_264151


namespace extra_coverage_calculation_l264_264373

/-- Define the conditions -/
def bag_coverage : ℕ := 500
def lawn_length : ℕ := 35
def lawn_width : ℕ := 48
def number_of_bags : ℕ := 6

/-- Define the main theorem to prove -/
theorem extra_coverage_calculation :
  number_of_bags * bag_coverage - (lawn_length * lawn_width) = 1320 := 
by
  sorry

end extra_coverage_calculation_l264_264373


namespace medicine_dosage_per_kg_l264_264485

theorem medicine_dosage_per_kg :
  ∀ (child_weight parts dose_per_part total_dose dose_per_kg : ℕ),
    (child_weight = 30) →
    (parts = 3) →
    (dose_per_part = 50) →
    (total_dose = parts * dose_per_part) →
    (dose_per_kg = total_dose / child_weight) →
    dose_per_kg = 5 :=
by
  intros child_weight parts dose_per_part total_dose dose_per_kg
  intros h1 h2 h3 h4 h5
  sorry

end medicine_dosage_per_kg_l264_264485


namespace sum_of_squares_nonzero_l264_264320

theorem sum_of_squares_nonzero {a b : ℝ} (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
sorry

end sum_of_squares_nonzero_l264_264320


namespace mabel_tomatoes_l264_264434

theorem mabel_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 := by
  sorry

end mabel_tomatoes_l264_264434


namespace symmetric_point_origin_l264_264609

-- Define the original point
def original_point : ℝ × ℝ := (4, -1)

-- Define a function to find the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem symmetric_point_origin : symmetric_point original_point = (-4, 1) :=
sorry

end symmetric_point_origin_l264_264609


namespace smallest_n_for_imaginary_part_l264_264274

noncomputable def z : Complex := Complex.ofReal (Real.cos (1 / 1000)) + Complex.I * Complex.ofReal (Real.sin (1 / 1000))

theorem smallest_n_for_imaginary_part :
  ∃ (n : ℕ), (Complex.im (z^(n : ℕ)) > 1 / 2) ∧ ∀ (m : ℕ), m < n → Complex.im (z^(m : ℕ)) ≤ 1 / 2 :=
begin
  sorry  -- Proof omitted as per instructions
end

end smallest_n_for_imaginary_part_l264_264274


namespace entree_cost_14_l264_264705

theorem entree_cost_14 (D E : ℝ) (h1 : D + E = 23) (h2 : E = D + 5) : E = 14 :=
sorry

end entree_cost_14_l264_264705


namespace relationship_among_a_ae_ea_minus_one_l264_264265

theorem relationship_among_a_ae_ea_minus_one (a : ℝ) (h : 0 < a ∧ a < 1) :
  (Real.exp a - 1 > a ∧ a > Real.exp a - 1 ∧ a > a^(Real.exp 1)) :=
by
  sorry

end relationship_among_a_ae_ea_minus_one_l264_264265


namespace nth_equation_l264_264277

theorem nth_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = (n - 1) * 10 + 1 :=
sorry

end nth_equation_l264_264277


namespace weight_of_bowling_ball_l264_264436

-- Define weights of bowling ball and canoe
variable (b c : ℚ)

-- Problem conditions
def cond1 : Prop := (9 * b = 5 * c)
def cond2 : Prop := (4 * c = 120)

-- The statement to prove
theorem weight_of_bowling_ball (h1 : cond1 b c) (h2 : cond2 c) : b = 50 / 3 := sorry

end weight_of_bowling_ball_l264_264436


namespace weight_of_fish_in_barrel_l264_264194

/-- 
Given a barrel with an initial weight of 54 kg when full of fish,
and a weight of 29 kg after removing half of the fish,
prove that the initial weight of the fish in the barrel was 50 kg.
-/
theorem weight_of_fish_in_barrel (B F : ℝ)
  (h1: B + F = 54)
  (h2: B + F / 2 = 29) : F = 50 := 
sorry

end weight_of_fish_in_barrel_l264_264194


namespace find_a_l264_264561

open Set
open Real

def A : Set ℝ := {-1, 1}
def B (a : ℝ) : Set ℝ := {x | a * x ^ 2 = 1}

theorem find_a (a : ℝ) (h : (A ∩ (B a)) = (B a)) : a = 1 :=
sorry

end find_a_l264_264561


namespace distance_points_l264_264512

theorem distance_points : 
  let P1 := (2, -1)
  let P2 := (7, 6)
  dist P1 P2 = Real.sqrt 74 :=
by
  sorry

end distance_points_l264_264512


namespace residue_7_1234_mod_13_l264_264321

theorem residue_7_1234_mod_13 :
  (7 : ℤ) ^ 1234 % 13 = 12 :=
by
  -- given conditions as definitions
  have h1 : (7 : ℤ) % 13 = 7 := by norm_num
  
  -- auxiliary calculations
  have h2 : (49 : ℤ) % 13 = 10 := by norm_num
  have h3 : (100 : ℤ) % 13 = 9 := by norm_num
  have h4 : (81 : ℤ) % 13 = 3 := by norm_num
  have h5 : (729 : ℤ) % 13 = 1 := by norm_num

  -- the actual problem we want to prove
  sorry

end residue_7_1234_mod_13_l264_264321


namespace calculation_result_l264_264915

theorem calculation_result : (1000 * 7 / 10 * 17 * 5^2 = 297500) :=
by sorry

end calculation_result_l264_264915


namespace factorize_expression_l264_264677

theorem factorize_expression (a : ℝ) : a^3 - 4 * a^2 + 4 * a = a * (a - 2)^2 := 
by
  sorry

end factorize_expression_l264_264677


namespace area_of_regionM_l264_264582

/-
Define the conditions as separate predicates in Lean.
-/

def cond1 (x y : ℝ) : Prop := y - x ≥ abs (x + y)

def cond2 (x y : ℝ) : Prop := (x^2 + 8*x + y^2 + 6*y) / (2*y - x - 8) ≤ 0

/-
Define region \( M \) by combining the conditions.
-/

def regionM (x y : ℝ) : Prop := cond1 x y ∧ cond2 x y

/-
Define the main theorem to compute the area of the region \( M \).
-/

theorem area_of_regionM : 
  ∀ x y : ℝ, (regionM x y) → (calculateAreaOfM) := sorry

/-
A placeholder definition to calculate the area of M. 
-/

noncomputable def calculateAreaOfM : ℝ := 8

end area_of_regionM_l264_264582


namespace units_digit_37_pow_37_l264_264329

theorem units_digit_37_pow_37 : (37 ^ 37) % 10 = 7 := by
  -- The proof is omitted as per instructions.
  sorry

end units_digit_37_pow_37_l264_264329


namespace max_min_z_in_region_l264_264059

open Real

noncomputable def bounding_region (x y : ℝ) : Prop := 
  (0 ≤ x) ∧ (y ≤ 2) ∧ (y ≥ (x^2 / 2))

noncomputable def z (x y : ℝ) : ℝ := 
  2 * x^3 - 6 * x * y + 3 * y^2

theorem max_min_z_in_region :
  ∃ c d : ℝ, 
    (∀ x y, bounding_region x y → z x y ≤ 12) ∧ 
    (∀ x y, bounding_region x y → z x y ≥ -1) ∧ 
    (∃ x y, bounding_region x y ∧ z x y = 12) ∧
    (∃ x y, bounding_region x y ∧ z x y = -1) :=
sorry

end max_min_z_in_region_l264_264059


namespace find_number_l264_264922

noncomputable def number_divided_by_seven_is_five_fourteen (x : ℝ) : Prop :=
  x / 7 = 5 / 14

theorem find_number (x : ℝ) (h : number_divided_by_seven_is_five_fourteen x) : x = 2.5 :=
by
  sorry

end find_number_l264_264922


namespace reflection_over_vector_l264_264818

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l264_264818


namespace hyperbola_slope_product_l264_264699

open Real

theorem hyperbola_slope_product
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h : ∀ {x y : ℝ}, x ≠ 0 → (x^2 / a^2 - y^2 / b^2 = 1) → 
    ∀ {k1 k2 : ℝ}, (x = 0 ∨ y = 0) → (k1 * k2 = ((b^2) / (a^2)))) :
  (b^2 / a^2 = 3) :=
by 
  sorry

end hyperbola_slope_product_l264_264699


namespace calculate_expression_l264_264361

theorem calculate_expression :
  -1 ^ 2023 + (Real.pi - 3.14) ^ 0 + |(-2 : ℝ)| = 2 :=
by
  sorry

end calculate_expression_l264_264361


namespace polynomial_coefficients_sum_l264_264555

theorem polynomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ), 
  (∀ x : ℚ, (3 * x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 +
                            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
                            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  (a_0 = -512) →
  ((a_0 + a_1 * (1/3) + a_2 * (1/3)^2 + a_3 * (1/3)^3 + 
    a_4 * (1/3)^4 + a_5 * (1/3)^5 + a_6 * (1/3)^6 + 
    a_7 * (1/3)^7 + a_8 * (1/3)^8 + a_9 * (1/3)^9) = -1) →
  (a_1 / 3 + a_2 / 3^2 + a_3 / 3^3 + a_4 / 3^4 + a_5 / 3^5 + 
   a_6 / 3^6 + a_7 / 3^7 + a_8 / 3^8 + a_9 / 3^9 = 511) :=
by 
  -- The proof would go here
  sorry

end polynomial_coefficients_sum_l264_264555


namespace relationship_of_a_b_c_l264_264243

noncomputable def a : ℝ := Real.log 3 / Real.log 2  -- a = log2(1/3)
noncomputable def b : ℝ := Real.exp (1 / 3)  -- b = e^(1/3)
noncomputable def c : ℝ := 1 / 3  -- c = e^ln(1/3) = 1/3

theorem relationship_of_a_b_c : b > c ∧ c > a :=
by
  -- Proof would go here
  sorry

end relationship_of_a_b_c_l264_264243


namespace fraction_identity_l264_264408

theorem fraction_identity (a b : ℝ) (h : 2 * a = 5 * b) : a / b = 5 / 2 := by
  sorry

end fraction_identity_l264_264408


namespace roses_to_sister_l264_264083

theorem roses_to_sister (total_roses roses_to_mother roses_to_grandmother roses_kept : ℕ) 
  (h1 : total_roses = 20)
  (h2 : roses_to_mother = 6)
  (h3 : roses_to_grandmother = 9)
  (h4 : roses_kept = 1) : 
  total_roses - (roses_to_mother + roses_to_grandmother + roses_kept) = 4 :=
by
  sorry

end roses_to_sister_l264_264083


namespace crease_points_ellipse_l264_264120

theorem crease_points_ellipse (R a : ℝ) (x y : ℝ) (h1 : 0 < R) (h2 : 0 < a) (h3 : a < R) : 
  (x - a / 2) ^ 2 / (R / 2) ^ 2 + y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2) ≥ 1 :=
by
  -- Omitted detailed proof steps
  sorry

end crease_points_ellipse_l264_264120


namespace total_cost_is_13_l264_264490

-- Definition of pencil cost
def pencil_cost : ℕ := 2

-- Definition of pen cost based on pencil cost
def pen_cost : ℕ := pencil_cost + 9

-- The total cost of both items
def total_cost := pencil_cost + pen_cost

theorem total_cost_is_13 : total_cost = 13 := by
  sorry

end total_cost_is_13_l264_264490


namespace ratio_diagonals_of_squares_l264_264189

variable (d₁ d₂ : ℝ)

theorem ratio_diagonals_of_squares (h : ∃ k : ℝ, d₂ = k * d₁) (h₁ : 1 < k ∧ k < 9) : 
  (∃ k : ℝ, 4 * (d₂ / Real.sqrt 2) = k * 4 * (d₁ / Real.sqrt 2)) → k = 5 := by
  sorry

end ratio_diagonals_of_squares_l264_264189


namespace lemuel_total_points_l264_264261

theorem lemuel_total_points (two_point_shots : ℕ) (three_point_shots : ℕ) (points_from_two : ℕ) (points_from_three : ℕ) :
  two_point_shots = 7 →
  three_point_shots = 3 →
  points_from_two = 2 →
  points_from_three = 3 →
  two_point_shots * points_from_two + three_point_shots * points_from_three = 23 :=
by
  sorry

end lemuel_total_points_l264_264261


namespace chocolate_syrup_amount_l264_264488

theorem chocolate_syrup_amount (x : ℝ) (H1 : 2 * x + 6 = 14) : x = 4 :=
by
  sorry

end chocolate_syrup_amount_l264_264488


namespace three_Y_five_l264_264719

-- Define the operation Y
def Y (a b : ℕ) : ℕ := 3 * b + 8 * a - a^2

-- State the theorem to prove the value of 3 Y 5
theorem three_Y_five : Y 3 5 = 30 :=
by
  sorry

end three_Y_five_l264_264719


namespace topsoil_cost_l264_264762

theorem topsoil_cost (cost_per_cubic_foot : ℝ) (cubic_yards : ℝ) (conversion_factor : ℝ) : 
  cubic_yards = 8 →
  cost_per_cubic_foot = 7 →
  conversion_factor = 27 →
  ∃ total_cost : ℝ, total_cost = 1512 :=
by
  intros h1 h2 h3
  sorry

end topsoil_cost_l264_264762


namespace divisible_by_24_l264_264200

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, n^4 + 2 * n^3 + 11 * n^2 + 10 * n = 24 * k := sorry

end divisible_by_24_l264_264200


namespace possible_values_of_expression_l264_264965

noncomputable def sign (x : ℝ) : ℝ :=
if x > 0 then 1 else -1

theorem possible_values_of_expression
  (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  in expr ∈ {5, 1, -1, -5} :=
by
  let expr := (sign a) + (sign b) + (sign c) + (sign d) + (sign (a * b * c * d))
  sorry

end possible_values_of_expression_l264_264965


namespace solve_fractional_equation_l264_264143

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l264_264143


namespace general_term_arithmetic_sum_terms_geometric_l264_264691

section ArithmeticSequence

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Conditions for Part 1
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  S 5 - S 2 = 195 ∧ d = -2 ∧
  ∀ n, S n = n * (a 1 + (n - 1) * (d / 2))

-- Prove the general term formula for the sequence {a_n}
theorem general_term_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) 
    (h : sum_arithmetic_sequence S a d) : 
    ∀ n, a n = -2 * n + 73 :=
sorry

end ArithmeticSequence


section GeometricSequence

variables {b : ℕ → ℝ} {n : ℕ} {T : ℕ → ℝ} {a : ℕ → ℝ}

-- Conditions for Part 2
def sum_geometric_sequence (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 1 = 13 ∧ b 2 = 65 ∧ a 4 = 65

-- Prove the sum of the first n terms for the sequence {b_n}
theorem sum_terms_geometric (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ)
    (h : sum_geometric_sequence b T a) : 
    ∀ n, T n = 13 * (5^n - 1) / 4 :=
sorry

end GeometricSequence

end general_term_arithmetic_sum_terms_geometric_l264_264691


namespace Amanda_income_if_report_not_finished_l264_264111

def hourly_pay : ℝ := 50.0
def hours_worked_per_day : ℝ := 10.0
def percent_withheld : ℝ := 0.2

theorem Amanda_income_if_report_not_finished : 
  let total_daily_income := hourly_pay * hours_worked_per_day in
  let amount_withheld := percent_withheld * total_daily_income in
  let amount_received := total_daily_income - amount_withheld in
  amount_received = 400 :=
by
  sorry

end Amanda_income_if_report_not_finished_l264_264111


namespace functional_equation_solution_l264_264950

theorem functional_equation_solution (f : ℕ+ → ℕ+) :
  (∀ n : ℕ+, f (f (f n)) + f (f n) + f n = 3 * n) →
  ∀ n : ℕ+, f n = n :=
by
  intro h
  sorry

end functional_equation_solution_l264_264950


namespace planes_divide_space_l264_264618

-- Definition of a triangular prism
def triangular_prism (V : Type) (P : Set (Set V)) : Prop :=
  ∃ (A B C D E F : V),
    P = {{A, B, C}, {D, E, F}, {A, B, D, E}, {B, C, E, F}, {C, A, F, D}}

-- The condition: planes containing the faces of a triangular prism
def planes_containing_faces (V : Type) (P : Set (Set V)) : Prop :=
  triangular_prism V P

-- Proof statement: The planes containing the faces of a triangular prism divide the space into 21 parts
theorem planes_divide_space (V : Type) (P : Set (Set V))
  (h : planes_containing_faces V P) :
  ∃ parts : ℕ, parts = 21 := by
  sorry

end planes_divide_space_l264_264618


namespace original_price_of_sarees_l264_264896
open Real

theorem original_price_of_sarees (P : ℝ) (h : 0.70 * 0.80 * P = 224) : P = 400 :=
sorry

end original_price_of_sarees_l264_264896


namespace Jerry_throw_count_l264_264740

theorem Jerry_throw_count : 
  let interrupt_points := 5
  let insult_points := 10
  let throw_points := 25
  let threshold := 100
  let interrupt_count := 2
  let insult_count := 4
  let current_points := (interrupt_count * interrupt_points) + (insult_count * insult_points)
  let additional_points := threshold - current_points
  let throw_count := additional_points / throw_points
  in throw_count = 2 :=
by {
  have h1 : current_points = (2 * 5) + (4 * 10) := rfl,
  have h2 : current_points = 10 + 40 := by { rw [Nat.mul_def, Nat.add_def], },
  have h3 : current_points = 50 := by { rw Nat.add_def },
  have h4 : additional_points = 100 - 50 := rfl,
  have h5 : additional_points = 50 := by { rw Nat.sub_def },
  have h6 : throw_count = 50 / 25 := rfl,
  show throw_count = 2,
  rw Nat.div_def,
  exact h6
} sorry

end Jerry_throw_count_l264_264740


namespace russian_players_pairing_probability_l264_264780

theorem russian_players_pairing_probability :
  let total_players := 10
  let russian_players := 4
  (russian_players * (russian_players - 1)) / (total_players * (total_players - 1)) * 
  ((russian_players - 2) * (russian_players - 3)) / ((total_players - 2) * (total_players - 3)) = 1 / 21 :=
by
  sorry

end russian_players_pairing_probability_l264_264780


namespace rita_hours_per_month_l264_264723

theorem rita_hours_per_month :
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  let h_remaining := t - h_completed
  let h := h_remaining / m
  h = 220
:= by 
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  have h_remaining := t - h_completed
  have h := h_remaining / m
  sorry

end rita_hours_per_month_l264_264723


namespace total_balloons_l264_264001

theorem total_balloons (Gold Silver Black Total : Nat) (h1 : Gold = 141)
  (h2 : Silver = 2 * Gold) (h3 : Black = 150) (h4 : Total = Gold + Silver + Black) :
  Total = 573 := 
by
  sorry

end total_balloons_l264_264001


namespace sarah_shirts_l264_264888

theorem sarah_shirts (loads : ℕ) (pieces_per_load : ℕ) (sweaters : ℕ) 
  (total_pieces : ℕ) (shirts : ℕ) : 
  loads = 9 → pieces_per_load = 5 → sweaters = 2 →
  total_pieces = loads * pieces_per_load → shirts = total_pieces - sweaters → 
  shirts = 43 :=
by
  intros h_loads h_pieces_per_load h_sweaters h_total_pieces h_shirts
  sorry

end sarah_shirts_l264_264888


namespace melissa_points_per_game_l264_264117

theorem melissa_points_per_game (total_points : ℕ) (games_played : ℕ) (h1 : total_points = 1200) (h2 : games_played = 10) : (total_points / games_played) = 120 := 
by
  -- Here we would insert the proof steps, but we use sorry to represent the omission
  sorry

end melissa_points_per_game_l264_264117


namespace polar_to_cartesian_correct_l264_264077

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_correct : polar_to_cartesian 2 (5 * Real.pi / 6) = (-Real.sqrt 3, 1) :=
by
  sorry -- We are not required to provide the proof here

end polar_to_cartesian_correct_l264_264077


namespace maximize_savings_l264_264919

-- Definitions for the conditions
def initial_amount : ℝ := 15000

def discount_option1 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.75
  let after_second : ℝ := after_first * 0.90
  after_second * 0.95

def discount_option2 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.70
  let after_second : ℝ := after_first * 0.90
  after_second * 0.90

-- Theorem to compare the final amounts
theorem maximize_savings : discount_option2 initial_amount < discount_option1 initial_amount := 
  sorry

end maximize_savings_l264_264919


namespace dogs_in_pet_shop_l264_264196

variable (D C B x : ℕ)

theorem dogs_in_pet_shop 
  (h1 : D = 7 * x) 
  (h2 : B = 8 * x)
  (h3 : D + B = 330) : 
  D = 154 :=
by
  sorry

end dogs_in_pet_shop_l264_264196


namespace price_reduction_l264_264494

theorem price_reduction (x y : ℕ) (h1 : (13 - x) * y = 781) (h2 : y ≤ 100) : x = 2 :=
sorry

end price_reduction_l264_264494


namespace factor_expression_l264_264532

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l264_264532


namespace required_fencing_l264_264785

-- Given definitions and conditions
def area (L W : ℕ) : ℕ := L * W

def fencing (W L : ℕ) : ℕ := 2 * W + L

theorem required_fencing
  (L W : ℕ)
  (hL : L = 10)
  (hA : area L W = 600) :
  fencing W L = 130 := by
  sorry

end required_fencing_l264_264785


namespace music_commercials_ratio_l264_264374

theorem music_commercials_ratio (T C: ℕ) (hT: T = 112) (hC: C = 40) : (T - C) / C = 9 / 5 := by
  sorry

end music_commercials_ratio_l264_264374


namespace quad_roots_sum_l264_264095

theorem quad_roots_sum {x₁ x₂ : ℝ} (h1 : x₁ + x₂ = 5) (h2 : x₁ * x₂ = -6) :
  1 / x₁ + 1 / x₂ = -5 / 6 :=
by
  sorry

end quad_roots_sum_l264_264095


namespace factor_expression_l264_264527

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l264_264527


namespace max_ratio_for_hoop_contact_l264_264903

theorem max_ratio_for_hoop_contact
  (m m_h R g : ℝ)
  (h1 : 0 < m) 
  (h2 : 0 < m_h)
  (h3 : 0 < R)
  (h4 : 0 < g) :
  (m / m_h ≤ 3 / 2) :=
sorry

end max_ratio_for_hoop_contact_l264_264903


namespace total_roses_in_a_week_l264_264007

theorem total_roses_in_a_week : 
  let day1 := 24 
  let day2 := day1 + 6
  let day3 := day2 + 6
  let day4 := day3 + 6
  let day5 := day4 + 6
  let day6 := day5 + 6
  let day7 := day6 + 6
  (day1 + day2 + day3 + day4 + day5 + day6 + day7) = 294 :=
by
  sorry

end total_roses_in_a_week_l264_264007


namespace daps_dips_equivalence_l264_264856

theorem daps_dips_equivalence :
  (∃ dap dop dip : Type,
    (5 : ℝ) * ∀ x : dap, x = (4 : ℝ) * ∀ y : dop, y ∧
    (3 : ℝ) * ∀ z : dop, z = (8 : ℝ) * ∀ w : dip, w) →
  (22.5 : ℝ) * ∀ x : dap, x = (48 : ℝ) * ∀ y : dip, y :=
begin
  sorry
end

end daps_dips_equivalence_l264_264856


namespace net_rate_of_pay_l264_264486

theorem net_rate_of_pay
  (hours_travelled : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℝ)
  (price_per_gallon : ℝ)
  (net_rate_of_pay : ℝ) :
  hours_travelled = 3 →
  speed = 50 →
  fuel_efficiency = 25 →
  pay_per_mile = 0.60 →
  price_per_gallon = 2.50 →
  net_rate_of_pay = 25 := by
  sorry

end net_rate_of_pay_l264_264486


namespace value_of_x_l264_264851

theorem value_of_x (x : ℕ) :
  (1 / 8) * 2 ^ 36 = 8 ^ x → x = 11 :=
by
  intro h
  have h1 : (1 / 8) = 2⁻³ := by sorry
  have h2 : (2⁻³) * 2 ^ 36 = 2 ^ 33 := by sorry
  have h3 : 8 ^ x = (2 ^ 3) ^ x := by sorry
  have h4 : (2 ^ 3) ^ x = 2 ^ (3 * x) := by sorry
  have h5 : 2 ^ 33 = 2 ^ (3 * x) := by sorry
  have h6 : 33 = 3 * x := by sorry
  exact nat.div_eq_of_lt 33 3 sorry

end value_of_x_l264_264851


namespace lines_intersect_at_l264_264235

theorem lines_intersect_at :
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 7 * y = -3 * x - 4 ∧ x = 54 / 5 ∧ y = -26 / 5 := 
by
  sorry

end lines_intersect_at_l264_264235


namespace fraction_bounds_l264_264806

theorem fraction_bounds (x y : ℝ) (h : x^2 * y^2 + x * y + 1 = 3 * y^2) : 
0 ≤ (y - x) / (x + 4 * y) ∧ (y - x) / (x + 4 * y) ≤ 4 := 
sorry

end fraction_bounds_l264_264806


namespace sum_of_consecutive_integers_is_33_l264_264179

theorem sum_of_consecutive_integers_is_33 :
  ∃ (x : ℕ), x * (x + 1) = 272 ∧ x + (x + 1) = 33 :=
by
  sorry

end sum_of_consecutive_integers_is_33_l264_264179


namespace arc_length_of_curve_is_sqrt_2_l264_264359

noncomputable def f (x : ℝ) : ℝ := 1 + Real.arcsin x - Real.sqrt (1 - x^2)

noncomputable def f' (x : ℝ) : ℝ := (1 + x) / Real.sqrt (1 - x^2)

noncomputable def arc_length (a b : ℝ) (f' : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt(1 + (f' x)^2)

theorem arc_length_of_curve_is_sqrt_2 : arc_length 0 (3 / 4) f' = Real.sqrt 2 := 
by 
  sorry

end arc_length_of_curve_is_sqrt_2_l264_264359


namespace trapezoid_fraction_l264_264046

theorem trapezoid_fraction 
  (shorter_base longer_base side_length : ℝ)
  (angle_adjacent : ℝ)
  (h1 : shorter_base = 120)
  (h2 : longer_base = 180)
  (h3 : side_length = 130)
  (h4 : angle_adjacent = 60) :
  ∃ fraction : ℝ, fraction = 1 / 2 :=
by
  sorry

end trapezoid_fraction_l264_264046


namespace n_congruence_mod_9_l264_264933

def n : ℕ := 2 + 333 + 5555 + 77777 + 999999 + 2222222 + 44444444 + 666666666

theorem n_congruence_mod_9 : n % 9 = 4 :=
by
  sorry

end n_congruence_mod_9_l264_264933


namespace janet_dresses_l264_264416

theorem janet_dresses : 
  ∃ D : ℕ, 
    (D / 2) * (2 / 3) + (D / 2) * (6 / 3) = 32 → D = 24 := 
by {
  sorry
}

end janet_dresses_l264_264416


namespace race_time_l264_264913

theorem race_time (t_A t_B : ℝ) (v_A v_B : ℝ)
  (h1 : t_B = t_A + 7)
  (h2 : v_A * t_A = 80)
  (h3 : v_B * t_B = 80)
  (h4 : v_A * (t_A + 7) = 136) :
  t_A = 10 :=
by
  sorry

end race_time_l264_264913


namespace right_triangle_hypotenuse_length_l264_264786

theorem right_triangle_hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = 10) (h₂ : b = 24) (h₃ : c^2 = a^2 + b^2) : c = 26 :=
by
  -- sorry is used to skip the actual proof
  sorry

end right_triangle_hypotenuse_length_l264_264786


namespace correct_table_count_l264_264653

def stools_per_table : ℕ := 8
def chairs_per_table : ℕ := 2
def legs_per_stool : ℕ := 3
def legs_per_chair : ℕ := 4
def legs_per_table : ℕ := 4
def total_legs : ℕ := 656

theorem correct_table_count (t : ℕ) :
  stools_per_table * legs_per_stool * t +
  chairs_per_table * legs_per_chair * t +
  legs_per_table * t = total_legs → t = 18 :=
by
  intros h
  sorry

end correct_table_count_l264_264653


namespace sqrt_of_S_l264_264764

def initial_time := 16 * 3600 + 11 * 60 + 22
def initial_date := 16
def total_seconds_in_a_day := 86400
def total_seconds_in_an_hour := 3600

theorem sqrt_of_S (S : ℕ) (hS : S = total_seconds_in_a_day + total_seconds_in_an_hour) : 
  Real.sqrt S = 300 := 
sorry

end sqrt_of_S_l264_264764


namespace fruit_basket_count_l264_264081

-- Define the number of apples and oranges
def apples := 7
def oranges := 12

-- Condition: A fruit basket must contain at least two pieces of fruit
def min_pieces_of_fruit := 2

-- Problem: Prove that there are 101 different fruit baskets containing at least two pieces of fruit
theorem fruit_basket_count (n_apples n_oranges n_min_pieces : Nat) (h_apples : n_apples = apples) (h_oranges : n_oranges = oranges) (h_min_pieces : n_min_pieces = min_pieces_of_fruit) :
  (n_apples = 7) ∧ (n_oranges = 12) ∧ (n_min_pieces = 2) → (104 - 3 = 101) :=
by
  sorry

end fruit_basket_count_l264_264081


namespace number_of_kids_at_circus_l264_264605

theorem number_of_kids_at_circus (K A : ℕ) 
(h1 : ∀ x, 5 * x = 1 / 2 * 10 * x)
(h2 : 5 * K + 10 * A = 50) : K = 2 :=
sorry

end number_of_kids_at_circus_l264_264605


namespace factor_expression_l264_264529

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l264_264529


namespace circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l264_264870

theorem circumscribe_quadrilateral_a : 
  ∃ (x : ℝ), 2 * x + 4 * x + 5 * x + 3 * x = 360 
          ∧ (2 * x + 5 * x = 180) 
          ∧ (4 * x + 3 * x = 180) := sorry

theorem circumscribe_quadrilateral_b : 
  ∃ (x : ℝ), 5 * x + 7 * x + 8 * x + 9 * x = 360 
          ∧ (5 * x + 8 * x ≠ 180) 
          ∧ (7 * x + 9 * x ≠ 180) := sorry

end circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l264_264870


namespace gcd_1729_1337_l264_264463

theorem gcd_1729_1337 : Nat.gcd 1729 1337 = 7 := 
by
  sorry

end gcd_1729_1337_l264_264463


namespace percent_decrease_l264_264720

theorem percent_decrease(call_cost_1980 call_cost_2010 : ℝ) (h₁ : call_cost_1980 = 50) (h₂ : call_cost_2010 = 5) :
  ((call_cost_1980 - call_cost_2010) / call_cost_1980 * 100) = 90 :=
by
  sorry

end percent_decrease_l264_264720


namespace sufficient_condition_not_necessary_condition_l264_264983

variable {a b : ℝ} 

theorem sufficient_condition (h : a < b ∧ b < 0) : a ^ 2 > b ^ 2 :=
sorry

theorem not_necessary_condition : ¬ (∀ {a b : ℝ}, a ^ 2 > b ^ 2 → a < b ∧ b < 0) :=
sorry

end sufficient_condition_not_necessary_condition_l264_264983


namespace roots_of_cubic_l264_264843

theorem roots_of_cubic (a b c d r s t : ℝ) 
  (h1 : r + s + t = -b / a)
  (h2 : r * s + r * t + s * t = c / a)
  (h3 : r * s * t = -d / a) :
  1 / (r ^ 2) + 1 / (s ^ 2) + 1 / (t ^ 2) = (c ^ 2 - 2 * b * d) / (d ^ 2) := 
sorry

end roots_of_cubic_l264_264843


namespace factor_expression_l264_264525

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end factor_expression_l264_264525


namespace convex_pentagon_probability_l264_264055

-- Defining the number of chords and the probability calculation as per the problem's conditions
def number_of_chords (n : ℕ) : ℕ := (n * (n - 1)) / 2
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem conditions
def eight_points_on_circle : ℕ := 8
def chords_chosen : ℕ := 5

-- Total number of chords from eight points
def total_chords : ℕ := number_of_chords eight_points_on_circle

-- The probability calculation
def probability_convex_pentagon :=
  binom 8 5 / binom total_chords chords_chosen

-- Statement to be proven
theorem convex_pentagon_probability :
  probability_convex_pentagon = 1 / 1755 := sorry

end convex_pentagon_probability_l264_264055


namespace factor_polynomials_l264_264814

theorem factor_polynomials :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 3) * (x^2 + 6*x + 12) :=
by
  sorry

end factor_polynomials_l264_264814


namespace coins_in_bag_l264_264917

theorem coins_in_bag (x : ℕ) (h : x + x / 2 + x / 4 = 105) : x = 60 :=
by
  sorry

end coins_in_bag_l264_264917


namespace points_per_draw_l264_264444

-- Definitions based on conditions
def total_games : ℕ := 20
def wins : ℕ := 14
def losses : ℕ := 2
def total_points : ℕ := 46
def points_per_win : ℕ := 3
def points_per_loss : ℕ := 0

-- Calculation of the number of draws and points per draw
def draws : ℕ := total_games - wins - losses
def points_wins : ℕ := wins * points_per_win
def points_draws : ℕ := total_points - points_wins

-- Theorem statement
theorem points_per_draw : points_draws / draws = 1 := by
  sorry

end points_per_draw_l264_264444


namespace at_least_two_inequalities_hold_l264_264113

theorem at_least_two_inequalities_hold 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a + b + c ≥ a * b * c) : 
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨ (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨ (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) := 
sorry

end at_least_two_inequalities_hold_l264_264113


namespace total_time_to_watch_movie_l264_264505

-- Define the conditions and the question
def uninterrupted_viewing_time : ℕ := 35 + 45 + 20
def rewinding_time : ℕ := 5 + 15
def total_time : ℕ := uninterrupted_viewing_time + rewinding_time

-- Lean statement of the proof problem
theorem total_time_to_watch_movie : total_time = 120 := by
  -- This is where the proof would go
  sorry

end total_time_to_watch_movie_l264_264505


namespace sqrt_41_40_39_38_plus_1_l264_264666

theorem sqrt_41_40_39_38_plus_1 : Real.sqrt ((41 * 40 * 39 * 38) + 1) = 1559 := by
  sorry

end sqrt_41_40_39_38_plus_1_l264_264666


namespace num_three_person_subcommittees_from_eight_l264_264404

def num_committees (n k : ℕ) : ℕ := (Nat.fact n) / ((Nat.fact k) * (Nat.fact (n - k)))

theorem num_three_person_subcommittees_from_eight (n : ℕ) (h : n = 8) : num_committees n 3 = 56 :=
by
  rw [h]
  sorry

end num_three_person_subcommittees_from_eight_l264_264404


namespace neg_product_B_l264_264034

def expr_A := (-1 / 3) * (1 / 4) * (-6)
def expr_B := (-9) * (1 / 8) * (-4 / 7) * 7 * (-1 / 3)
def expr_C := (-3) * (-1 / 2) * 7 * 0
def expr_D := (-1 / 5) * 6 * (-2 / 3) * (-5) * (-1 / 2)

theorem neg_product_B :
  expr_B < 0 :=
by
  sorry

end neg_product_B_l264_264034


namespace same_root_implies_a_vals_l264_264254

-- Define the first function f(x) = x - a
def f (x a : ℝ) : ℝ := x - a

-- Define the second function g(x) = x^2 + ax - 2
def g (x a : ℝ) : ℝ := x^2 + a * x - 2

-- Theorem statement
theorem same_root_implies_a_vals (a : ℝ) (x : ℝ) (hf : f x a = 0) (hg : g x a = 0) : a = 1 ∨ a = -1 := 
sorry

end same_root_implies_a_vals_l264_264254


namespace max_area_difference_l264_264312

theorem max_area_difference (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) : 
  abs ((l * w) - (l' * w')) ≤ 1600 := 
by 
  sorry

end max_area_difference_l264_264312


namespace coral_must_read_pages_to_finish_book_l264_264944

theorem coral_must_read_pages_to_finish_book
  (total_pages first_week_read second_week_percentage pages_remaining first_week_left second_week_read : ℕ)
  (initial_pages_read : ℕ := total_pages / 2)
  (remaining_after_first_week : ℕ := total_pages - initial_pages_read)
  (read_second_week : ℕ := remaining_after_first_week * second_week_percentage / 100)
  (remaining_after_second_week : ℕ := remaining_after_first_week - read_second_week)
  (final_pages_to_read : ℕ := remaining_after_second_week):
  total_pages = 600 → first_week_read = 300 → second_week_percentage = 30 →
  pages_remaining = 300 → first_week_left = 300 → second_week_read = 90 →
  remaining_after_first_week = 300 - 300 →
  remaining_after_second_week = remaining_after_first_week - second_week_read →
  third_week_read = remaining_after_second_week →
  third_week_read = 210 := by
  sorry

end coral_must_read_pages_to_finish_book_l264_264944


namespace swimming_speed_l264_264924

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (swim_time : ℝ) (distance : ℝ) :
  water_speed = 8 →
  swim_time = 8 →
  distance = 16 →
  distance = (v - water_speed) * swim_time →
  v = 10 := 
by
  intros h1 h2 h3 h4
  sorry

end swimming_speed_l264_264924


namespace find_x_min_construction_cost_l264_264760

-- Define the conditions for Team A and Team B
def Team_A_Daily_Construction (x : ℕ) : ℕ := x + 300
def Team_A_Daily_Cost : ℕ := 3600
def Team_B_Daily_Construction (x : ℕ) : ℕ := x
def Team_B_Daily_Cost : ℕ := 2200

-- Condition: The number of days Team A needs to construct 1800m^2 is equal to the number of days Team B needs to construct 1200m^2
def construction_days (x : ℕ) : Prop := 
  1800 / (x + 300) = 1200 / x

-- Define the total days worked and the minimum construction area condition
def total_days : ℕ := 22
def min_construction_area : ℕ := 15000

-- Define the construction cost function given the number of days each team works
def construction_cost (m : ℕ) : ℕ := 
  3600 * m + 2200 * (total_days - m)

-- Main theorem: Prove that x = 600 satisfies the conditions
theorem find_x (x : ℕ) (h : x = 600) : construction_days x := by sorry

-- Second theorem: Prove that the minimum construction cost is 56800 yuan
theorem min_construction_cost (m : ℕ) (h : m ≥ 6) : construction_cost m = 56800 := by sorry

end find_x_min_construction_cost_l264_264760


namespace ab_cd_eq_neg_37_over_9_l264_264409

theorem ab_cd_eq_neg_37_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a + b + d = 2)
  (h3 : a + c + d = 3)
  (h4 : b + c + d = -3) :
  a * b + c * d = -37 / 9 := by
  sorry

end ab_cd_eq_neg_37_over_9_l264_264409


namespace balls_in_boxes_l264_264563

theorem balls_in_boxes :
  (Nat.choose (7 + 3 - 1) (3 - 1)) = 36 := by
  sorry

end balls_in_boxes_l264_264563


namespace mutated_frog_percentage_l264_264684

theorem mutated_frog_percentage 
  (extra_legs : ℕ) 
  (two_heads : ℕ) 
  (bright_red : ℕ) 
  (normal_frogs : ℕ) 
  (h_extra_legs : extra_legs = 5) 
  (h_two_heads : two_heads = 2) 
  (h_bright_red : bright_red = 2) 
  (h_normal_frogs : normal_frogs = 18) 
  : ((extra_legs + two_heads + bright_red) * 100 / (extra_legs + two_heads + bright_red + normal_frogs)).round = 33 := 
by
  sorry

end mutated_frog_percentage_l264_264684


namespace num_three_person_subcommittees_from_eight_l264_264403

def num_committees (n k : ℕ) : ℕ := (Nat.fact n) / ((Nat.fact k) * (Nat.fact (n - k)))

theorem num_three_person_subcommittees_from_eight (n : ℕ) (h : n = 8) : num_committees n 3 = 56 :=
by
  rw [h]
  sorry

end num_three_person_subcommittees_from_eight_l264_264403


namespace reflection_matrix_over_vector_is_correct_l264_264817

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l264_264817


namespace edward_dunk_a_clown_tickets_l264_264947

-- Definitions for conditions
def total_tickets : ℕ := 79
def rides : ℕ := 8
def tickets_per_ride : ℕ := 7

-- Theorem statement
theorem edward_dunk_a_clown_tickets :
  let tickets_spent_on_rides := rides * tickets_per_ride
  let tickets_remaining := total_tickets - tickets_spent_on_rides
  tickets_remaining = 23 :=
by
  sorry

end edward_dunk_a_clown_tickets_l264_264947


namespace circle_center_sum_l264_264296

theorem circle_center_sum (x y : ℝ) (hx : (x, y) = (3, -4)) :
  (x + y) = -1 :=
by {
  -- We are given that the center of the circle is (3, -4)
  sorry -- Proof is omitted
}

end circle_center_sum_l264_264296


namespace roots_of_quadratic_implies_values_l264_264014

theorem roots_of_quadratic_implies_values (a b : ℝ) :
  (∃ x : ℝ, x^2 + 2 * (1 + a) * x + (3 * a^2 + 4 * a * b + 4 * b^2 + 2) = 0) →
  a = 1 ∧ b = -1/2 :=
by
  sorry

end roots_of_quadratic_implies_values_l264_264014


namespace mabel_tomatoes_l264_264431

theorem mabel_tomatoes :
  ∃ (t1 t2 t3 t4 : ℕ), 
    t1 = 8 ∧ 
    t2 = t1 + 4 ∧ 
    t3 = 3 * (t1 + t2) ∧ 
    t4 = 3 * (t1 + t2) ∧ 
    (t1 + t2 + t3 + t4) = 140 :=
by
  sorry

end mabel_tomatoes_l264_264431


namespace smallest_pos_int_mod_congruence_l264_264905

theorem smallest_pos_int_mod_congruence : ∃ n : ℕ, 0 < n ∧ n ≡ 2 [MOD 31] ∧ 5 * n ≡ 409 [MOD 31] :=
by
  sorry

end smallest_pos_int_mod_congruence_l264_264905


namespace polynomial_roots_l264_264539

theorem polynomial_roots :
  (∀ x, x^3 - 3*x^2 - x + 3 = 0 ↔ (x = 1 ∨ x = -1 ∨ x = 3)) :=
by
  intro x
  split
  {
    intro h
    have h1 : x = 1 ∨ x = -1 ∨ x = 3
    {
      sorry
    }
    exact h1
  }
  {
    intro h
    cases h
    {
      rw h
      simp
    }
    {
      cases h
      {
        rw h
        simp
      }
      {
        rw h
        simp
      }
    }
  }

end polynomial_roots_l264_264539


namespace breadth_remains_the_same_l264_264303

variable (L B : ℝ)

theorem breadth_remains_the_same 
  (A : ℝ) (hA : A = L * B) 
  (L_half : ℝ) (hL_half : L_half = L / 2) 
  (B' : ℝ)
  (A' : ℝ) (hA' : A' = L_half * B') 
  (hA_change : A' = 0.5 * A) : 
  B' = B :=
  sorry

end breadth_remains_the_same_l264_264303


namespace round_wins_probability_l264_264578

theorem round_wins_probability : 
  let p_A := (1:ℚ)/2
  let p_C := (1:ℚ)/6
  let p_M := 2 * p_C
  let total_rounds := 7
  let prob_A := p_A^4
  let prob_M := p_M^2
  let prob_C := p_C
  let specific_seq_prob := prob_A * prob_M * prob_C
  let arrangements := (nat.factorial total_rounds) / ((nat.factorial 4) * (nat.factorial 2) * (nat.factorial 1))
  (specific_seq_prob * arrangements = 35 / 288) :=
sorry

end round_wins_probability_l264_264578


namespace Sohan_work_time_l264_264242

theorem Sohan_work_time (G R S : ℚ) (h1 : G + R + S = 1/16) (h2 : G + R = 1/24) : S = 1/48 :=
by
  sorry

end Sohan_work_time_l264_264242


namespace multiplication_correct_l264_264585

theorem multiplication_correct :
  23 * 195 = 4485 :=
by
  sorry

end multiplication_correct_l264_264585


namespace max_value_on_interval_l264_264616

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f x ≤ 5 :=
by
  sorry

end max_value_on_interval_l264_264616


namespace find_dihedral_angle_l264_264169

noncomputable def dihedral_angle_problem :
  Type := sorry  -- We may need custom types for geometric objects

def is_isosceles_right_triangle (ABC : Triangle ℝ) (AB : ℝ) : Prop :=
  (ABC.angleA = ABC.angleB = π/4) ∧ (ABC.hypotenuse = AB)

def is_midpoint (D H : Point ℝ) (A C : Line ℝ) : Prop :=
  H = (A + C) / 2 ∧ D.rectilinear_above H

def height (ABCD : Pyramid ℝ) : ℝ :=
  2

theorem find_dihedral_angle (ABCD : Pyramid ℝ)
  (hypotenuse : ℝ) (height : ℝ) (mid_ac : Point ℝ) :
  ∀ {a b c d: Point ℝ}, 
  is_isosceles_right_triangle a b c 
  → is_midpoint d a c 
  → dihedral_angle a b d = arcsin (sqrt (3 / 5)) :=
sorry

end find_dihedral_angle_l264_264169


namespace tangent_product_power_l264_264850

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180))
  * (1 + Real.tan (2 * Real.pi / 180))
  * (1 + Real.tan (3 * Real.pi / 180))
  * (1 + Real.tan (4 * Real.pi / 180))
  * (1 + Real.tan (5 * Real.pi / 180))
  * (1 + Real.tan (6 * Real.pi / 180))
  * (1 + Real.tan (7 * Real.pi / 180))
  * (1 + Real.tan (8 * Real.pi / 180))
  * (1 + Real.tan (9 * Real.pi / 180))
  * (1 + Real.tan (10 * Real.pi / 180))
  * (1 + Real.tan (11 * Real.pi / 180))
  * (1 + Real.tan (12 * Real.pi / 180))
  * (1 + Real.tan (13 * Real.pi / 180))
  * (1 + Real.tan (14 * Real.pi / 180))
  * (1 + Real.tan (15 * Real.pi / 180))
  * (1 + Real.tan (16 * Real.pi / 180))
  * (1 + Real.tan (17 * Real.pi / 180))
  * (1 + Real.tan (18 * Real.pi / 180))
  * (1 + Real.tan (19 * Real.pi / 180))
  * (1 + Real.tan (20 * Real.pi / 180))
  * (1 + Real.tan (21 * Real.pi / 180))
  * (1 + Real.tan (22 * Real.pi / 180))
  * (1 + Real.tan (23 * Real.pi / 180))
  * (1 + Real.tan (24 * Real.pi / 180))
  * (1 + Real.tan (25 * Real.pi / 180))
  * (1 + Real.tan (26 * Real.pi / 180))
  * (1 + Real.tan (27 * Real.pi / 180))
  * (1 + Real.tan (28 * Real.pi / 180))
  * (1 + Real.tan (29 * Real.pi / 180))
  * (1 + Real.tan (30 * Real.pi / 180))
  * (1 + Real.tan (31 * Real.pi / 180))
  * (1 + Real.tan (32 * Real.pi / 180))
  * (1 + Real.tan (33 * Real.pi / 180))
  * (1 + Real.tan (34 * Real.pi / 180))
  * (1 + Real.tan (35 * Real.pi / 180))
  * (1 + Real.tan (36 * Real.pi / 180))
  * (1 + Real.tan (37 * Real.pi / 180))
  * (1 + Real.tan (38 * Real.pi / 180))
  * (1 + Real.tan (39 * Real.pi / 180))
  * (1 + Real.tan (40 * Real.pi / 180))
  * (1 + Real.tan (41 * Real.pi / 180))
  * (1 + Real.tan (42 * Real.pi / 180))
  * (1 + Real.tan (43 * Real.pi / 180))
  * (1 + Real.tan (44 * Real.pi / 180))
  * (1 + Real.tan (45 * Real.pi / 180))
  * (1 + Real.tan (46 * Real.pi / 180))
  * (1 + Real.tan (47 * Real.pi / 180))
  * (1 + Real.tan (48 * Real.pi / 180))
  * (1 + Real.tan (49 * Real.pi / 180))
  * (1 + Real.tan (50 * Real.pi / 180))
  * (1 + Real.tan (51 * Real.pi / 180))
  * (1 + Real.tan (52 * Real.pi / 180))
  * (1 + Real.tan (53 * Real.pi / 180))
  * (1 + Real.tan (54 * Real.pi / 180))
  * (1 + Real.tan (55 * Real.pi / 180))
  * (1 + Real.tan (56 * Real.pi / 180))
  * (1 + Real.tan (57 * Real.pi / 180))
  * (1 + Real.tan (58 * Real.pi / 180))
  * (1 + Real.tan (59 * Real.pi / 180))
  * (1 + Real.tan (60 * Real.pi / 180))

theorem tangent_product_power : tangent_product = 2^30 := by
  sorry

end tangent_product_power_l264_264850


namespace image_center_coordinates_l264_264225

-- Define the point reflecting across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the point translation by adding some units to the y-coordinate
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Define the initial point and translation
def initial_point : ℝ × ℝ := (3, -4)
def translation_units : ℝ := 5

-- Prove the final coordinates of the image of the center of circle Q
theorem image_center_coordinates : translate_y (reflect_x initial_point) translation_units = (3, 9) :=
  sorry

end image_center_coordinates_l264_264225


namespace smallest_x_satisfying_equation_l264_264906

theorem smallest_x_satisfying_equation :
  ∀ x : ℝ, (2 * x ^ 2 + 24 * x - 60 = x * (x + 13)) → x = -15 ∨ x = 4 ∧ ∃ y : ℝ, y = -15 ∨ y = 4 ∧ y ≤ x :=
by
  sorry

end smallest_x_satisfying_equation_l264_264906


namespace solve_equation_l264_264139

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l264_264139


namespace cost_of_student_ticket_l264_264759

theorem cost_of_student_ticket
  (cost_adult : ℤ)
  (total_tickets : ℤ)
  (total_revenue : ℤ)
  (adult_tickets : ℤ)
  (student_tickets : ℤ)
  (H1 : cost_adult = 6)
  (H2 : total_tickets = 846)
  (H3 : total_revenue = 3846)
  (H4 : adult_tickets = 410)
  (H5 : student_tickets = 436)
  : (total_revenue = adult_tickets * cost_adult + student_tickets * (318 / 100)) :=
by
  -- mathematical proof steps would go here
  sorry

end cost_of_student_ticket_l264_264759


namespace original_number_is_10_l264_264995

theorem original_number_is_10 (x : ℝ) (h : 2 * x + 5 = x / 2 + 20) : x = 10 := 
by {
  sorry
}

end original_number_is_10_l264_264995


namespace franks_age_l264_264386

variable (F G : ℕ)

def gabriel_younger_than_frank : Prop := G = F - 3
def total_age_is_seventeen : Prop := F + G = 17

theorem franks_age (h1 : gabriel_younger_than_frank F G) (h2 : total_age_is_seventeen F G) : F = 10 :=
by
  sorry

end franks_age_l264_264386


namespace lauri_ate_days_l264_264286

theorem lauri_ate_days
    (simone_rate : ℚ)
    (simone_days : ℕ)
    (lauri_rate : ℚ)
    (total_apples : ℚ)
    (simone_apples : ℚ)
    (lauri_apples : ℚ)
    (lauri_days : ℚ) :
  simone_rate = 1/2 → 
  simone_days = 16 →
  lauri_rate = 1/3 →
  total_apples = 13 →
  simone_apples = simone_rate * simone_days →
  lauri_apples = total_apples - simone_apples →
  lauri_days = lauri_apples / lauri_rate →
  lauri_days = 15 :=
by
  intros
  sorry

end lauri_ate_days_l264_264286


namespace kendra_total_earnings_l264_264730

-- Definitions of the conditions based on the problem statement
def kendra_earnings_2014 : ℕ := 30000 - 8000
def laurel_earnings_2014 : ℕ := 30000
def kendra_earnings_2015 : ℕ := laurel_earnings_2014 + (laurel_earnings_2014 / 5)

-- The statement to be proved
theorem kendra_total_earnings : kendra_earnings_2014 + kendra_earnings_2015 = 58000 :=
by
  -- Using Lean tactics for the proof
  sorry

end kendra_total_earnings_l264_264730


namespace vertex_angle_isosceles_l264_264860

theorem vertex_angle_isosceles (a b c : ℝ)
  (isosceles: (a = b ∨ b = c ∨ c = a))
  (angle_sum : a + b + c = 180)
  (one_angle_is_70 : a = 70 ∨ b = 70 ∨ c = 70) :
  a = 40 ∨ a = 70 ∨ b = 40 ∨ b = 70 ∨ c = 40 ∨ c = 70 :=
by sorry

end vertex_angle_isosceles_l264_264860


namespace cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l264_264726

noncomputable def p1 (x : ℝ) : ℝ :=
  if x < -1 ∨ x > 1 then 0 else 0.5

noncomputable def p2 (y : ℝ) : ℝ :=
  if y < 0 ∨ y > 2 then 0 else 0.5

noncomputable def F1 (x : ℝ) : ℝ :=
  if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1

noncomputable def F2 (y : ℝ) : ℝ :=
  if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1

noncomputable def p (x : ℝ) (y : ℝ) : ℝ :=
  if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25

noncomputable def F (x : ℝ) (y : ℝ) : ℝ :=
  if x ≤ -1 ∨ y ≤ 0 then 0
  else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y 
  else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
  else if x > 1 ∧ y ≤ 2 then 0.5 * y
  else 1

theorem cumulative_distribution_F1 (x : ℝ) : 
  F1 x = if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1 := by sorry

theorem cumulative_distribution_F2 (y : ℝ) : 
  F2 y = if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1 := by sorry

theorem joint_density (x : ℝ) (y : ℝ) : 
  p x y = if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25 := by sorry

theorem joint_cumulative_distribution (x : ℝ) (y : ℝ) : 
  F x y = if x ≤ -1 ∨ y ≤ 0 then 0
          else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y
          else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
          else if x > 1 ∧ y ≤ 2 then 0.5 * y
          else 1 := by sorry

end cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l264_264726


namespace math_problem_statements_are_correct_l264_264388

theorem math_problem_statements_are_correct (a b : ℝ) (h : a > b ∧ b > 0) :
  (¬ (b / a > (b + 3) / (a + 3))) ∧ ((3 * a + 2 * b) / (2 * a + 3 * b) < a / b) ∧
  (¬ (2 * Real.sqrt a < Real.sqrt (a - b) + Real.sqrt b)) ∧ 
  (Real.log ((a + b) / 2) > (Real.log a + Real.log b) / 2) :=
by
  sorry

end math_problem_statements_are_correct_l264_264388


namespace volunteer_selection_probability_l264_264787

theorem volunteer_selection_probability :
  ∀ (students total_students remaining_students selected_volunteers : ℕ),
    total_students = 2018 →
    remaining_students = total_students - 18 →
    selected_volunteers = 50 →
    (selected_volunteers : ℚ) / total_students = (25 : ℚ) / 1009 :=
by
  intros students total_students remaining_students selected_volunteers
  intros h1 h2 h3
  sorry

end volunteer_selection_probability_l264_264787


namespace max_xy_l264_264071

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy ≤ 81 :=
by sorry

end max_xy_l264_264071


namespace graph_inequality_solution_l264_264702

noncomputable def solution_set : Set (Real × Real) := {
  p | let x := p.1
       let y := p.2
       (y^2 - (Real.arcsin (Real.sin x))^2) *
       (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
       (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0
}

theorem graph_inequality_solution
  (x y : ℝ) :
  (y^2 - (Real.arcsin (Real.sin x))^2) *
  (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
  (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0 ↔
  (x, y) ∈ solution_set :=
by
  sorry

end graph_inequality_solution_l264_264702


namespace average_of_remaining_two_numbers_l264_264476

theorem average_of_remaining_two_numbers (S a₁ a₂ a₃ a₄ : ℝ)
    (h₁ : S / 6 = 3.95)
    (h₂ : (a₁ + a₂) / 2 = 3.8)
    (h₃ : (a₃ + a₄) / 2 = 3.85) :
    (S - (a₁ + a₂ + a₃ + a₄)) / 2 = 4.2 := 
sorry

end average_of_remaining_two_numbers_l264_264476


namespace mabel_tomatoes_l264_264430

theorem mabel_tomatoes :
  ∃ (t1 t2 t3 t4 : ℕ), 
    t1 = 8 ∧ 
    t2 = t1 + 4 ∧ 
    t3 = 3 * (t1 + t2) ∧ 
    t4 = 3 * (t1 + t2) ∧ 
    (t1 + t2 + t3 + t4) = 140 :=
by
  sorry

end mabel_tomatoes_l264_264430


namespace break_even_price_correct_l264_264119

-- Conditions
def variable_cost_per_handle : ℝ := 0.60
def fixed_cost_per_week : ℝ := 7640
def handles_per_week : ℝ := 1910

-- Define the correct answer for the price per handle to break even
def break_even_price_per_handle : ℝ := 4.60

-- The statement to prove
theorem break_even_price_correct :
  fixed_cost_per_week + (variable_cost_per_handle * handles_per_week) / handles_per_week = break_even_price_per_handle :=
by
  -- The proof is omitted
  sorry

end break_even_price_correct_l264_264119


namespace x_zero_sufficient_not_necessary_for_sin_zero_l264_264335

theorem x_zero_sufficient_not_necessary_for_sin_zero :
  (∀ x : ℝ, x = 0 → Real.sin x = 0) ∧ (∃ y : ℝ, Real.sin y = 0 ∧ y ≠ 0) :=
by
  sorry

end x_zero_sufficient_not_necessary_for_sin_zero_l264_264335


namespace simplify_expression_l264_264907

theorem simplify_expression :
  (2021^3 - 3 * 2021^2 * 2022 + 4 * 2021 * 2022^2 - 2022^3 + 2) / (2021 * 2022) = 
  1 + (1 / 2021) :=
by
  sorry

end simplify_expression_l264_264907


namespace multiply_increase_by_196_l264_264210

theorem multiply_increase_by_196 (x : ℕ) (h : 14 * x = 14 + 196) : x = 15 :=
sorry

end multiply_increase_by_196_l264_264210


namespace factor_expression_l264_264530

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l264_264530


namespace correct_relation_l264_264978

def A : Set ℝ := { x | x > 1 }

theorem correct_relation : 2 ∈ A := by
  -- Proof would go here
  sorry

end correct_relation_l264_264978


namespace simultaneous_equations_solution_l264_264162

theorem simultaneous_equations_solution (x y : ℚ) (h1 : 3 * x - 4 * y = 11) (h2 : 9 * x + 6 * y = 33) : 
  x = 11 / 3 ∧ y = 0 :=
by {
  sorry
}

end simultaneous_equations_solution_l264_264162


namespace workout_goal_l264_264284

def monday_situps : ℕ := 12
def tuesday_situps : ℕ := 19
def wednesday_situps_needed : ℕ := 59

theorem workout_goal : monday_situps + tuesday_situps + wednesday_situps_needed = 90 := by
  sorry

end workout_goal_l264_264284


namespace find_p4_q4_l264_264172

-- Definitions
def p (x : ℝ) : ℝ := 3 * (x - 6) * (x - 2)
def q (x : ℝ) : ℝ := (x - 6) * (x + 3)

-- Statement to prove
theorem find_p4_q4 : (p 4) / (q 4) = 6 / 7 :=
by
  sorry

end find_p4_q4_l264_264172


namespace cos_neg_13pi_over_4_l264_264308

theorem cos_neg_13pi_over_4 : Real.cos (-13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_13pi_over_4_l264_264308


namespace amanda_pay_if_not_finished_l264_264112

-- Define Amanda's hourly rate and daily work hours.
def amanda_hourly_rate : ℝ := 50
def amanda_daily_hours : ℝ := 10

-- Define the percentage of pay Jose will withhold.
def withholding_percentage : ℝ := 0.20

-- Define Amanda's total pay if she finishes the sales report.
def amanda_total_pay : ℝ := amanda_hourly_rate * amanda_daily_hours

-- Define the amount withheld if she does not finish the sales report.
def withheld_amount : ℝ := amanda_total_pay * withholding_percentage

-- Define the amount Amanda will receive if she does not finish the sales report.
def amanda_final_pay_not_finished : ℝ := amanda_total_pay - withheld_amount

-- The theorem to prove:
theorem amanda_pay_if_not_finished : amanda_final_pay_not_finished = 400 := by
  sorry

end amanda_pay_if_not_finished_l264_264112


namespace min_fraction_ineq_l264_264953

theorem min_fraction_ineq (x y : ℝ) (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) :
  ∃ z, (z = x * y / (x^2 + 2 * y^2)) ∧ z = 1 / 3 := sorry

end min_fraction_ineq_l264_264953


namespace sqrt_12_estimate_l264_264674

theorem sqrt_12_estimate : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end sqrt_12_estimate_l264_264674


namespace spider_total_distance_l264_264215

theorem spider_total_distance
    (radius : ℝ)
    (diameter : ℝ)
    (half_diameter : ℝ)
    (final_leg : ℝ)
    (total_distance : ℝ) :
    radius = 75 →
    diameter = 2 * radius →
    half_diameter = diameter / 2 →
    final_leg = 90 →
    (half_diameter ^ 2 + final_leg ^ 2 = diameter ^ 2) →
    total_distance = diameter + half_diameter + final_leg →
    total_distance = 315 :=
by
  intros
  sorry

end spider_total_distance_l264_264215


namespace sum_of_sequence_l264_264223

theorem sum_of_sequence :
  3 + 15 + 27 + 53 + 65 + 17 + 29 + 41 + 71 + 83 = 404 :=
by
  sorry

end sum_of_sequence_l264_264223


namespace Maxwell_age_l264_264577

theorem Maxwell_age :
  ∀ (sister_age maxwell_age : ℕ),
    (sister_age = 2) → 
    (maxwell_age + 2 = 2 * (sister_age + 2)) →
    (maxwell_age = 6) :=
by
  intros sister_age maxwell_age h1 h2
  -- Definitions and hypotheses come directly from conditions
  sorry

end Maxwell_age_l264_264577


namespace no_integer_solutions_system_l264_264439

theorem no_integer_solutions_system :
  ¬(∃ x y z : ℤ, 
    x^6 + x^3 + x^3 * y + y = 147^157 ∧ 
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147) :=
  sorry

end no_integer_solutions_system_l264_264439


namespace wall_length_proof_l264_264621

noncomputable def volume_of_brick (length width height : ℝ) : ℝ := length * width * height

noncomputable def total_volume (brick_volume num_of_bricks : ℝ) : ℝ := brick_volume * num_of_bricks

theorem wall_length_proof
  (height_of_wall : ℝ) (width_of_walls : ℝ) (num_of_bricks : ℝ)
  (length_of_brick width_of_brick height_of_brick : ℝ)
  (total_volume_of_bricks : ℝ) :
  total_volume (volume_of_brick length_of_brick width_of_brick height_of_brick) num_of_bricks = total_volume_of_bricks →
  volume_of_brick length_of_wall height_of_wall width_of_walls = total_volume_of_bricks →
  height_of_wall = 600 →
  width_of_walls = 2 →
  num_of_bricks = 2909.090909090909 →
  length_of_brick = 5 →
  width_of_brick = 11 →
  height_of_brick = 6 →
  total_volume_of_bricks = 960000 →
  length_of_wall = 800 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end wall_length_proof_l264_264621


namespace voting_total_participation_l264_264866

theorem voting_total_participation:
  ∀ (x : ℝ),
  0.35 * x + 0.65 * x = x ∧
  0.65 * x = 0.45 * (x + 80) →
  (x + 80 = 260) :=
by
  intros x h
  sorry

end voting_total_participation_l264_264866


namespace total_balloons_l264_264002

theorem total_balloons (Gold Silver Black Total : Nat) (h1 : Gold = 141)
  (h2 : Silver = 2 * Gold) (h3 : Black = 150) (h4 : Total = Gold + Silver + Black) :
  Total = 573 := 
by
  sorry

end total_balloons_l264_264002


namespace penguin_seafood_protein_l264_264742

theorem penguin_seafood_protein
  (digest : ℝ) -- representing 30% 
  (digested : ℝ) -- representing 9 grams 
  (h : digest = 0.30) 
  (h1 : digested = 9) :
  ∃ x : ℝ, digested = digest * x ∧ x = 30 :=
by
  sorry

end penguin_seafood_protein_l264_264742


namespace solve_equation_l264_264161

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l264_264161


namespace geometric_sequence_fourth_term_l264_264611

theorem geometric_sequence_fourth_term (a₁ a₂ a₃ : ℝ) (r : ℝ)
    (h₁ : a₁ = 5^(3/4))
    (h₂ : a₂ = 5^(1/2))
    (h₃ : a₃ = 5^(1/4))
    (geometric_seq : a₂ = a₁ * r ∧ a₃ = a₂ * r) :
    a₃ * r = 1 := 
by
  sorry

end geometric_sequence_fourth_term_l264_264611


namespace selection_combinations_l264_264099

noncomputable def num_possible_selections (boys : ℕ) (girls : ℕ) (include_kimi : Bool) (stone_goes : Bool) : ℕ :=
  if boys = 3 ∧ girls = 2 then
    if include_kimi ∧ ¬ stone_goes then
      3 -- Combinations when Kimi goes and Stone doesn't
    else if ¬ include_kimi ∧ stone_goes then
      3 -- Combinations when Kimi doesn't go and Stone goes
    else
      6 -- Remaining combinations without specific constraints on Kimi and Stone
  else
    0

theorem selection_combinations : num_possible_selections 3 2 true false + num_possible_selections 3 2 false true + num_possible_selections 3 2 false false = 12 :=
by
  sorry

end selection_combinations_l264_264099


namespace sin_over_cos_inequality_l264_264886

-- Define the main theorem and condition
theorem sin_over_cos_inequality (t : ℝ) (h₁ : 0 < t) (h₂ : t ≤ Real.pi / 2) : 
  (Real.sin t / t)^3 > Real.cos t := 
sorry

end sin_over_cos_inequality_l264_264886


namespace MichelangeloCeilingPainting_l264_264118

theorem MichelangeloCeilingPainting (total_ceiling week1_ceiling next_week_fraction : ℕ) 
  (a1 : total_ceiling = 28) 
  (a2 : week1_ceiling = 12) 
  (a3 : total_ceiling - (week1_ceiling + next_week_fraction * week1_ceiling) = 13) : 
  next_week_fraction = 1 / 4 := 
by 
  sorry

end MichelangeloCeilingPainting_l264_264118


namespace kwik_e_tax_revenue_l264_264750

def price_federal : ℕ := 50
def price_state : ℕ := 30
def price_quarterly : ℕ := 80

def num_federal : ℕ := 60
def num_state : ℕ := 20
def num_quarterly : ℕ := 10

def revenue_federal := num_federal * price_federal
def revenue_state := num_state * price_state
def revenue_quarterly := num_quarterly * price_quarterly

def total_revenue := revenue_federal + revenue_state + revenue_quarterly

theorem kwik_e_tax_revenue : total_revenue = 4400 := by
  sorry

end kwik_e_tax_revenue_l264_264750


namespace sin_arithmetic_sequence_l264_264380

noncomputable def sin_value (a : ℝ) := Real.sin (a * (Real.pi / 180))

theorem sin_arithmetic_sequence (a : ℝ) : 
  (0 < a) ∧ (a < 360) ∧ (sin_value a + sin_value (3 * a) = 2 * sin_value (2 * a)) ↔ a = 90 ∨ a = 270 :=
by 
  sorry

end sin_arithmetic_sequence_l264_264380


namespace pieces_of_fudge_l264_264022

def pan_length : ℝ := 27.5
def pan_width : ℝ := 17.5
def pan_height : ℝ := 2.5
def cube_side : ℝ := 2.3

def volume (l w h : ℝ) : ℝ := l * w * h

def V_pan : ℝ := volume pan_length pan_width pan_height
def V_cube : ℝ := volume cube_side cube_side cube_side

theorem pieces_of_fudge : ⌊V_pan / V_cube⌋ = 98 := by
  -- calculation can be filled in here in the actual proof
  sorry

end pieces_of_fudge_l264_264022


namespace auction_site_TVs_correct_l264_264656

-- Define the number of TVs Beatrice looked at in person
def in_person_TVs : Nat := 8

-- Define the number of TVs Beatrice looked at online
def online_TVs : Nat := 3 * in_person_TVs

-- Define the total number of TVs Beatrice looked at
def total_TVs : Nat := 42

-- Define the number of TVs Beatrice looked at on the auction site
def auction_site_TVs : Nat := total_TVs - (in_person_TVs + online_TVs)

-- Prove that the number of TVs Beatrice looked at on the auction site is 10
theorem auction_site_TVs_correct : auction_site_TVs = 10 :=
by
  sorry

end auction_site_TVs_correct_l264_264656


namespace remainder_of_b97_is_52_l264_264421

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem remainder_of_b97_is_52 : (b 97) % 81 = 52 := 
sorry

end remainder_of_b97_is_52_l264_264421


namespace factor_expression_l264_264524

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l264_264524


namespace javier_average_hits_l264_264728

-- Define the total number of games Javier plays and the first set number of games
def total_games := 30
def first_set_games := 20

-- Define the hit averages for the first set of games and the desired season average
def average_hits_first_set := 2
def desired_season_average := 3

-- Define the total hits Javier needs to achieve the desired average by the end of the season
def total_hits_needed : ℕ := total_games * desired_season_average

-- Define the hits Javier made in the first set of games
def hits_made_first_set : ℕ := first_set_games * average_hits_first_set

-- Define the remaining games and the hits Javier needs to achieve in these games to meet his target
def remaining_games := total_games - first_set_games
def hits_needed_remaining_games : ℕ := total_hits_needed - hits_made_first_set

-- Define the average hits Javier needs in the remaining games to meet his target
def average_needed_remaining_games (remaining_games hits_needed_remaining_games : ℕ) : ℕ :=
  hits_needed_remaining_games / remaining_games

theorem javier_average_hits : 
  average_needed_remaining_games remaining_games hits_needed_remaining_games = 5 := 
by
  -- The proof is omitted.
  sorry

end javier_average_hits_l264_264728


namespace find_other_number_l264_264175

theorem find_other_number (LCM HCF number1 number2 : ℕ) 
  (hLCM : LCM = 7700) 
  (hHCF : HCF = 11) 
  (hNumber1 : number1 = 308)
  (hProductEquality : number1 * number2 = LCM * HCF) :
  number2 = 275 :=
by
  -- proof omitted
  sorry

end find_other_number_l264_264175


namespace distance_between_points_l264_264941

theorem distance_between_points 
  (a b c d : ℝ) 
  (h1 : a = 5) 
  (h2 : c = 10) 
  (h3 : b = 2 * a + 3) 
  (h4 : d = 2 * c + 3) 
  : (Real.sqrt ((c - a)^2 + (d - b)^2)) = 5 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_l264_264941


namespace reflection_matrix_is_correct_l264_264821

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l264_264821


namespace third_consecutive_even_integer_l264_264619

theorem third_consecutive_even_integer (n : ℤ) (h : (n + 2) + (n + 6) = 156) : (n + 4) = 78 :=
sorry

end third_consecutive_even_integer_l264_264619


namespace harmonic_mean_pairs_l264_264679

theorem harmonic_mean_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) 
    (hmean : (2 * x * y) / (x + y) = 2^30) :
    (∃! n, n = 29) :=
by
  sorry

end harmonic_mean_pairs_l264_264679


namespace entree_cost_14_l264_264708

-- Define the conditions as given in part a)
def total_cost (e d : ℕ) : Prop := e + d = 23
def entree_more (e d : ℕ) : Prop := e = d + 5

-- The theorem to be proved
theorem entree_cost_14 (e d : ℕ) (h1 : total_cost e d) (h2 : entree_more e d) : e = 14 := 
by 
  sorry

end entree_cost_14_l264_264708


namespace book_width_l264_264756

noncomputable def phi_conjugate : ℝ := (Real.sqrt 5 - 1) / 2

theorem book_width {w l : ℝ} (h_ratio : w / l = phi_conjugate) (h_length : l = 14) :
  w = 7 * Real.sqrt 5 - 7 :=
by
  sorry

end book_width_l264_264756


namespace max_area_difference_160_perimeter_rectangles_l264_264317

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end max_area_difference_160_perimeter_rectangles_l264_264317


namespace total_votes_l264_264473

theorem total_votes (V : ℝ) (h1 : 0.35 * V + (0.35 * V + 1650) = V) : V = 5500 := 
by 
  sorry

end total_votes_l264_264473


namespace solve_equation_l264_264160

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l264_264160


namespace arithmetic_sequence_l264_264867

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence (a1 d : ℝ) (h_d : d ≠ 0) 
  (h1 : a1 + (a1 + 2 * d) = 8) 
  (h2 : (a1 + d) * (a1 + 8 * d) = (a1 + 3 * d) * (a1 + 3 * d)) :
  a_n a1 d 5 = 13 := 
by 
  sorry

end arithmetic_sequence_l264_264867


namespace problem_l264_264568

theorem problem (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 + 2007 = 2008 :=
by
  sorry

end problem_l264_264568


namespace min_diff_f_l264_264389

def f (x : ℝ) := 2017 * x ^ 2 - 2018 * x + 2019 * 2020

theorem min_diff_f (t : ℝ) : 
  let f_max := max (f t) (f (t + 2))
  let f_min := min (f t) (f (t + 2))
  (f_max - f_min) ≥ 2017 :=
sorry

end min_diff_f_l264_264389


namespace axis_of_symmetry_parabola_l264_264295

theorem axis_of_symmetry_parabola (a b : ℝ) (h₁ : a = -3) (h₂ : b = 6) :
  -b / (2 * a) = 1 :=
by
  sorry

end axis_of_symmetry_parabola_l264_264295


namespace sufficient_but_not_necessary_l264_264833

def sequence_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def abs_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > abs (a n)

theorem sufficient_but_not_necessary (a : ℕ → ℝ) :
  (abs_condition a → sequence_increasing a) ∧ ¬ (sequence_increasing a → abs_condition a) :=
by
  sorry

end sufficient_but_not_necessary_l264_264833


namespace sector_central_angle_l264_264695

theorem sector_central_angle (r l α : ℝ) (h1 : 2 * r + l = 6) (h2 : 1/2 * l * r = 2) :
  α = l / r → (α = 1 ∨ α = 4) :=
by
  sorry

end sector_central_angle_l264_264695


namespace rabbit_turtle_travel_distance_l264_264895

-- Define the initial conditions and their values
def rabbit_velocity : ℕ := 40 -- meters per minute when jumping
def rabbit_jump_time : ℕ := 3 -- minutes of jumping
def rabbit_rest_time : ℕ := 2 -- minutes of resting
def rabbit_start_time : ℕ := 9 * 60 -- 9:00 AM in minutes from midnight

def turtle_velocity : ℕ := 10 -- meters per minute
def turtle_start_time : ℕ := 6 * 60 + 40 -- 6:40 AM in minutes from midnight
def lead_time : ℕ := 15 -- turtle leads the rabbit by 15 seconds at the end

-- Define the final distance the turtle traveled by the time rabbit arrives
def distance_traveled_by_turtle (total_time : ℕ) : ℕ :=
  total_time * turtle_velocity

-- Define time intervals for periodic calculations (in minutes)
def time_interval : ℕ := 5

-- Define the total distance rabbit covers in one periodic interval
def rabbit_distance_in_interval : ℕ :=
  rabbit_velocity * rabbit_jump_time

-- Calculate total time taken by the rabbit to close the gap before starting actual run
def initial_time_to_close_gap (gap : ℕ) : ℕ := 
  gap * time_interval / rabbit_distance_in_interval

-- Define the total time the rabbit travels
def total_travel_time : ℕ :=
  initial_time_to_close_gap ((rabbit_start_time - turtle_start_time) * turtle_velocity) + 97

-- Define the total distance condition to be proved as 2370 meters
theorem rabbit_turtle_travel_distance :
  distance_traveled_by_turtle (total_travel_time + lead_time) = 2370 :=
  by sorry

end rabbit_turtle_travel_distance_l264_264895


namespace no_integer_cube_eq_3n_squared_plus_3n_plus_7_l264_264053

theorem no_integer_cube_eq_3n_squared_plus_3n_plus_7 :
  ¬ ∃ x n : ℤ, x^3 = 3 * n^2 + 3 * n + 7 := 
sorry

end no_integer_cube_eq_3n_squared_plus_3n_plus_7_l264_264053


namespace sum_of_coefficients_l264_264271

theorem sum_of_coefficients (s : ℕ → ℝ) (a b c : ℝ) : 
  s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 17 ∧ 
  (∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) → 
  a + b + c = 12 := 
by
  sorry

end sum_of_coefficients_l264_264271


namespace point_in_third_quadrant_cos_sin_l264_264305

theorem point_in_third_quadrant_cos_sin (P : ℝ × ℝ) (hP : P = (Real.cos (2009 * Real.pi / 180), Real.sin (2009 * Real.pi / 180))) :
  P.1 < 0 ∧ P.2 < 0 :=
by
  sorry

end point_in_third_quadrant_cos_sin_l264_264305


namespace vector_orthogonality_solution_l264_264248

theorem vector_orthogonality_solution :
  let a := (3, -2)
  let b := (x, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  x = 2 / 3 :=
by
  intro h
  sorry

end vector_orthogonality_solution_l264_264248


namespace farmer_harvest_correct_l264_264347

def estimated_harvest : ℕ := 48097
def additional_harvest : ℕ := 684
def total_harvest : ℕ := 48781

theorem farmer_harvest_correct : estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvest_correct_l264_264347


namespace water_added_to_solution_l264_264202

theorem water_added_to_solution :
  let initial_volume := 340
  let initial_sugar := 0.20 * initial_volume
  let added_sugar := 3.2
  let added_kola := 6.8
  let final_sugar := initial_sugar + added_sugar
  let final_percentage_sugar := 19.66850828729282 / 100
  let final_volume := final_sugar / final_percentage_sugar
  let added_water := final_volume - initial_volume - added_sugar - added_kola
  added_water = 12 :=
by
  sorry

end water_added_to_solution_l264_264202


namespace erick_total_earnings_l264_264471

theorem erick_total_earnings
    (original_lemon_price : ℕ)
    (lemon_price_increase : ℕ)
    (original_grape_price : ℕ)
    (lemons_count : ℕ)
    (grapes_count : ℕ) :
    let new_lemon_price := original_lemon_price + lemon_price_increase,
        total_lemons_earning := lemons_count * new_lemon_price,
        grape_price_increase := lemon_price_increase / 2,
        new_grape_price := original_grape_price + grape_price_increase,
        total_grapes_earning := grapes_count * new_grape_price,
        total_earning := total_lemons_earning + total_grapes_earning
    in total_earning = 2220 := by
  sorry

end erick_total_earnings_l264_264471


namespace time_to_traverse_nth_mile_l264_264211

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 2) :
  (∃ k : ℝ, (∀ d : ℝ, d = n - 1 → (s_n = k / d)) ∧ (s_2 = 1 / 2)) → 
  t_n = 2 * (n - 1) :=
by 
  sorry

end time_to_traverse_nth_mile_l264_264211


namespace r_plus_s_value_l264_264615

theorem r_plus_s_value :
  (∃ (r s : ℝ) (line_intercepts : ∀ x y, y = -1/2 * x + 8 ∧ ((x = 16 ∧ y = 0) ∨ (x = 0 ∧ y = 8))), 
    s = -1/2 * r + 8 ∧ (16 * 8 / 2) = 2 * (16 * s / 2) ∧ r + s = 12) :=
sorry

end r_plus_s_value_l264_264615


namespace problem_1_problem_2_l264_264402

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), Real.cos (2 * x))
noncomputable def vec_b : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).fst * vec_b.fst + (vec_a x).snd * vec_b.snd + 2

theorem problem_1 (x : ℝ) : x ∈ Set.Icc (k * Real.pi - (5 / 12) * Real.pi) (k * Real.pi + (1 / 12) * Real.pi) → ∃ k : ℤ, ∀ x : ℝ, f (x) = Real.sin (2 * x + (1 / 3) * Real.pi) + 2 :=
sorry

theorem problem_2 (x : ℝ) : x ∈ Set.Icc (π / 6) (2 * π / 3) → f (π / 6) = (Real.sqrt 3 / 2) + 2 ∧ f (7 * π / 12) = 1 :=
sorry

end problem_1_problem_2_l264_264402


namespace smaller_angle_linear_pair_l264_264902

theorem smaller_angle_linear_pair (a b : ℝ) (h1 : a + b = 180) (h2 : a = 5 * b) : b = 30 := by
  sorry

end smaller_angle_linear_pair_l264_264902


namespace april_revenue_l264_264222

def revenue_after_tax (initial_roses : ℕ) (initial_tulips : ℕ) (initial_daisies : ℕ)
                      (final_roses : ℕ) (final_tulips : ℕ) (final_daisies : ℕ)
                      (price_rose : ℝ) (price_tulip : ℝ) (price_daisy : ℝ) (tax_rate : ℝ) : ℝ :=
(price_rose * (initial_roses - final_roses) + price_tulip * (initial_tulips - final_tulips) + price_daisy * (initial_daisies - final_daisies)) * (1 + tax_rate)

theorem april_revenue :
  revenue_after_tax 13 10 8 4 3 1 4 3 2 0.10 = 78.10 := by
  sorry

end april_revenue_l264_264222


namespace replace_digits_divisible_by_13_l264_264019

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

theorem replace_digits_divisible_by_13 :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ 
  (3 * 10^6 + x * 10^4 + y * 10^2 + 3) % 13 = 0 ∧
  (x = 2 ∧ y = 3 ∨ 
   x = 5 ∧ y = 2 ∨ 
   x = 8 ∧ y = 1 ∨ 
   x = 9 ∧ y = 5 ∨ 
   x = 6 ∧ y = 6 ∨ 
   x = 3 ∧ y = 7 ∨ 
   x = 0 ∧ y = 8) :=
by
  sorry

end replace_digits_divisible_by_13_l264_264019


namespace naomi_wash_time_l264_264881

theorem naomi_wash_time (C T S : ℕ) (h₁ : T = 2 * C) (h₂ : S = 2 * C - 15) (h₃ : C + T + S = 135) : C = 30 :=
by
  sorry

end naomi_wash_time_l264_264881


namespace entree_cost_l264_264710

/-- 
Prove that if the total cost is 23 and the entree costs 5 more than the dessert, 
then the cost of the entree is 14.
-/
theorem entree_cost (D : ℝ) (H1 : D + (D + 5) = 23) : D + 5 = 14 :=
by
  -- note: no proof required as per instructions
  sorry

end entree_cost_l264_264710


namespace probability_third_term_three_a_plus_b_l264_264115

open Finset

def T : Finset (Equiv.Perm (Fin 6)) :=
  univ.filter (λ σ, σ 0 ≠ 1)

def favorable_permutations : Nat :=
  (T.filter (λ σ, σ 2 = 3)).card

def total_permutations : Nat :=
  T.card

def probability : Rat :=
  favorable_permutations / total_permutations

theorem probability_third_term_three :
  probability = 4 / 25 :=
by
  sorry

theorem a_plus_b :
  let ⟨a, b, h⟩ := probability.num_denom in
  a + b = 29 :=
by
  sorry

end probability_third_term_three_a_plus_b_l264_264115


namespace daps_equivalent_to_48_dips_l264_264857

noncomputable def conversion_daps_to_dops : ℚ := 5 / 4
noncomputable def conversion_dops_to_dips : ℚ := 3 / 8
noncomputable def conversion_daps_to_dips : ℚ := conversion_daps_to_dops * conversion_dops_to_dips

theorem daps_equivalent_to_48_dips :
  ∀ (daps dops dips : Type) (eq1 : 5*daps = 4*dops) (eq2 : 3*dops = 8*dips), 
  (48:ℚ) * conversion_daps_to_dips = (22.5:ℚ) :=
by
  sorry

end daps_equivalent_to_48_dips_l264_264857


namespace tomatoes_harvest_ratio_l264_264164

noncomputable def tomatoes_ratio (w t f : ℕ) (g r : ℕ) : ℕ × ℕ :=
  if (w = 400) ∧ ((w + t + f) = 2000) ∧ ((g = 700) ∧ (r = 700) ∧ ((g + r) = f)) ∧ (t = 200) then 
    (2, 1)
  else 
    sorry

theorem tomatoes_harvest_ratio : 
  ∀ (w t f : ℕ) (g r : ℕ), 
  (w = 400) → 
  (w + t + f = 2000) → 
  (g = 700) → 
  (r = 700) → 
  (g + r = f) → 
  (t = 200) →
  tomatoes_ratio w t f g r = (2, 1) :=
by {
  -- insert proof here
  sorry
}

end tomatoes_harvest_ratio_l264_264164


namespace total_contribution_l264_264039

theorem total_contribution : 
  ∀ (Niraj_contribution : ℕ) (Brittany_contribution Angela_contribution : ℕ),
    (Brittany_contribution = 3 * Niraj_contribution) →
    (Angela_contribution = 3 * Brittany_contribution) →
    (Niraj_contribution = 80) →
    (Niraj_contribution + Brittany_contribution + Angela_contribution = 1040) :=
  by assumption sorry

end total_contribution_l264_264039


namespace combination_20_6_l264_264940

theorem combination_20_6 : Nat.choose 20 6 = 38760 :=
by
  sorry

end combination_20_6_l264_264940


namespace solve_fractional_equation_l264_264156

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l264_264156


namespace only_prime_in_sequence_is_67_l264_264800

/-- Define the sequence of numbers composed by repetition of '67'. -/
def sequence (n : ℕ) : ℕ := 67 * (10^(2*n) - 1) / 99

/-- Prove that the only prime number in the sequence sequence(n) for n = 1 to 10 is 67. -/
theorem only_prime_in_sequence_is_67 : 
  (∀ n, 1 ≤ n → n ≤ 10 → ¬ prime (sequence n)) ∧ prime (sequence 1) := 
by
  sorry

end only_prime_in_sequence_is_67_l264_264800


namespace solve_for_x_l264_264287

theorem solve_for_x (x : ℝ) (h : (7 * x + 3) / (x + 5) - 5 / (x + 5) = 2 / (x + 5)) : x = 4 / 7 :=
by
  sorry

end solve_for_x_l264_264287


namespace value_of_expression_l264_264625

def expression (x y z : ℤ) : ℤ :=
  x^2 + y^2 - z^2 + 2 * x * y + x * y * z

theorem value_of_expression (x y z : ℤ) (h1 : x = 2) (h2 : y = -3) (h3 : z = 1) : 
  expression x y z = -7 := by
  sorry

end value_of_expression_l264_264625


namespace smallest_perimeter_of_scalene_triangle_with_conditions_l264_264031

def is_odd_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 2 = 1

-- Define a scalene triangle
structure ScaleneTriangle :=
  (a b c : ℕ)
  (a_ne_b : a ≠ b)
  (a_ne_c : a ≠ c)
  (b_ne_c : b ≠ c)
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a)

-- Define the problem conditions
def problem_conditions (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  a < b ∧ b < c ∧
  Nat.Prime (a + b + c) ∧
  (∃ (t : ScaleneTriangle), t.a = a ∧ t.b = b ∧ t.c = c)

-- Define the proposition
theorem smallest_perimeter_of_scalene_triangle_with_conditions :
  ∃ (a b c : ℕ), problem_conditions a b c ∧ a + b + c = 23 :=
sorry

end smallest_perimeter_of_scalene_triangle_with_conditions_l264_264031


namespace g_a_eq_g_inv_a_l264_264420

noncomputable def f (a x : ℝ) : ℝ :=
  a * real.sqrt (1 - x^2) + real.sqrt (1 + x) + real.sqrt (1 - x)

-- Define t in terms of x 
def t (x : ℝ) : ℝ := real.sqrt (1 + x) + real.sqrt (1 - x)

-- m(t) function derived from f(x)
noncomputable def m (a t : ℝ) : ℝ := (1/2 : ℝ) * a * t^2 + t - a

-- Define the maximum value g(a) based on derived m(t)
noncomputable def g (a : ℝ) : ℝ :=
  if a > -1/2 then a + 2
  else if -real.sqrt 2 / 2 < a ∧ a <= -1/2 then -a - 1/(2*a)
  else real.sqrt 2

-- Proof statement that g(a) = g(1/a) for specific conditions on a
theorem g_a_eq_g_inv_a (a : ℝ) : 
  g a = g (1/a) ↔ (a = 1 ∨ (-real.sqrt 2 ≤ a ∧ a ≤ -(real.sqrt 2)/2)) :=
sorry

end g_a_eq_g_inv_a_l264_264420


namespace molecular_weight_of_compound_l264_264465

theorem molecular_weight_of_compound :
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  molecular_weight = 156.22615 :=
by
  -- conditions
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  -- prove statement
  have h1 : average_atomic_weight_c = 12.05015 := by sorry
  have h2 : molecular_weight = 156.22615 := by sorry
  exact h2

end molecular_weight_of_compound_l264_264465


namespace residue_7_1234_mod_13_l264_264322

theorem residue_7_1234_mod_13 :
  (7 : ℤ) ^ 1234 % 13 = 12 :=
by
  -- given conditions as definitions
  have h1 : (7 : ℤ) % 13 = 7 := by norm_num
  
  -- auxiliary calculations
  have h2 : (49 : ℤ) % 13 = 10 := by norm_num
  have h3 : (100 : ℤ) % 13 = 9 := by norm_num
  have h4 : (81 : ℤ) % 13 = 3 := by norm_num
  have h5 : (729 : ℤ) % 13 = 1 := by norm_num

  -- the actual problem we want to prove
  sorry

end residue_7_1234_mod_13_l264_264322


namespace mabel_tomatoes_l264_264429

theorem mabel_tomatoes :
  ∃ (t1 t2 t3 t4 : ℕ), 
    t1 = 8 ∧ 
    t2 = t1 + 4 ∧ 
    t3 = 3 * (t1 + t2) ∧ 
    t4 = 3 * (t1 + t2) ∧ 
    (t1 + t2 + t3 + t4) = 140 :=
by
  sorry

end mabel_tomatoes_l264_264429


namespace measure_angle_A_value_b_sin_B_div_c_l264_264096

variables {a b c : ℝ} (A B C : ℝ)
-- Conditions
noncomputable def is_geom_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def given_equation (a b c : ℝ) : Prop :=
  a^2 - c^2 = a * c - b * c

-- Prove that given the conditions, ∠A = 60°
theorem measure_angle_A (h1 : is_geom_progression a b c) (h2 : given_equation a b c) : A = 60 :=
by
  sorry

-- Prove that given the conditions, the value of (b * sin B) / c is √3/2
theorem value_b_sin_B_div_c (h1 : is_geom_progression a b c) (h2 : given_equation a b c) (h3 : A = 60) : (b * sin B) / c = sqrt 3 / 2 :=
by
  sorry

end measure_angle_A_value_b_sin_B_div_c_l264_264096


namespace line_passes_through_vertex_count_l264_264680

theorem line_passes_through_vertex_count :
  (∃ a : ℝ, ∀ (x : ℝ), x = 0 → (x + a = a^2)) ↔ (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end line_passes_through_vertex_count_l264_264680


namespace number_from_division_l264_264920

theorem number_from_division (number : ℝ) (h : number / 2000 = 0.012625) : number = 25.25 :=
by
  sorry

end number_from_division_l264_264920


namespace min_internal_fence_length_l264_264348

-- Setup the given conditions in Lean 4
def total_land_area (length width : ℕ) : ℕ := length * width

def sotkas_to_m2 (sotkas : ℕ) : ℕ := sotkas * 100

-- Assume a father had three sons and left them an inheritance of land
def land_inheritance := 9 -- in sotkas

-- The dimensions of the land
def length := 25 
def width := 36

-- Prove that:
theorem min_internal_fence_length :
  ∃ (ways : ℕ) (min_length : ℕ),
    total_land_area length width = sotkas_to_m2 land_inheritance ∧
    (∀ (l1 l2 l3 w1 w2 w3 : ℕ),
      l1 * w1 = sotkas_to_m2 3 ∧ l2 * w2 = sotkas_to_m2 3 ∧ l3 * w3 = sotkas_to_m2 3 →
      ways = 4 ∧ min_length = 49) :=
by
  sorry

end min_internal_fence_length_l264_264348


namespace factorize_n_squared_minus_nine_l264_264378

theorem factorize_n_squared_minus_nine (n : ℝ) : n^2 - 9 = (n + 3) * (n - 3) := 
sorry

end factorize_n_squared_minus_nine_l264_264378


namespace counterexample_exists_l264_264801

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := ¬ is_prime n ∧ n > 1

theorem counterexample_exists :
  ∃ n, is_composite n ∧ is_composite (n - 3) ∧ n = 18 := by
  sorry

end counterexample_exists_l264_264801


namespace dihedral_angle_proof_l264_264689

noncomputable def angle_between_planes 
  (α β : Real) : Real :=
  Real.arcsin (Real.sin α * Real.sin β)

theorem dihedral_angle_proof 
  (α β : Real) 
  (α_non_neg : 0 ≤ α) 
  (α_non_gtr : α ≤ Real.pi / 2) 
  (β_non_neg : 0 ≤ β) 
  (β_non_gtr : β ≤ Real.pi / 2) :
  angle_between_planes α β = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end dihedral_angle_proof_l264_264689


namespace regina_earnings_l264_264125

def num_cows : ℕ := 20

def num_pigs (num_cows : ℕ) : ℕ := 4 * num_cows

def price_per_pig : ℕ := 400
def price_per_cow : ℕ := 800

def earnings (num_cows num_pigs price_per_cow price_per_pig : ℕ) : ℕ :=
  num_cows * price_per_cow + num_pigs * price_per_pig

theorem regina_earnings :
  earnings num_cows (num_pigs num_cows) price_per_cow price_per_pig = 48000 :=
by
  -- proof omitted
  sorry

end regina_earnings_l264_264125


namespace rectangle_side_l264_264617

theorem rectangle_side (x : ℝ) (w : ℝ) (P : ℝ) (hP : P = 30) (h : 2 * (x + w) = P) : w = 15 - x :=
by
  -- Proof goes here
  sorry

end rectangle_side_l264_264617


namespace visiting_plans_correct_l264_264883

-- Define the number of students
def num_students : ℕ := 4

-- Define the number of places to visit
def num_places : ℕ := 3

-- Define the total number of visiting plans without any restrictions
def total_visiting_plans : ℕ := num_places ^ num_students

-- Define the number of visiting plans where no one visits Haxi Station
def no_haxi_visiting_plans : ℕ := (num_places - 1) ^ num_students

-- Define the number of visiting plans where Haxi Station has at least one visitor
def visiting_plans_with_haxi : ℕ := total_visiting_plans - no_haxi_visiting_plans

-- Prove that the number of different visiting plans with at least one student visiting Haxi Station is 65
theorem visiting_plans_correct : visiting_plans_with_haxi = 65 := by
  -- Omitted proof
  sorry

end visiting_plans_correct_l264_264883


namespace find_remainder_l264_264193

variable (x y remainder : ℕ)
variable (h1 : x = 7 * y + 3)
variable (h2 : 2 * x = 18 * y + remainder)
variable (h3 : 11 * y - x = 1)

theorem find_remainder : remainder = 2 := 
by
  sorry

end find_remainder_l264_264193


namespace amaya_total_time_l264_264503

-- Define the times as per the conditions
def first_segment : Nat := 35 + 5
def second_segment : Nat := 45 + 15
def third_segment : Nat := 20

-- Define the total time by summing up all segments
def total_time : Nat := first_segment + second_segment + third_segment

-- The theorem to prove
theorem amaya_total_time : total_time = 120 := by
  -- Let's explicitly state the expected result here
  have h1 : first_segment = 40 := rfl
  have h2 : second_segment = 60 := rfl
  have h3 : third_segment = 20 := rfl
  have h_sum : total_time = 40 + 60 + 20 := by
    rw [h1, h2, h3]
  simp [total_time, h_sum]
  -- Finally, the result is 120
  exact rfl

end amaya_total_time_l264_264503


namespace student_number_in_eighth_group_l264_264204

-- Definitions corresponding to each condition
def students : ℕ := 50
def group_size : ℕ := 5
def third_group_student_number : ℕ := 12
def kth_group_number (k : ℕ) (n : ℕ) : ℕ := n + (k - 3) * group_size

-- Main statement to prove
theorem student_number_in_eighth_group :
  kth_group_number 8 third_group_student_number = 37 :=
  by
  sorry

end student_number_in_eighth_group_l264_264204


namespace solve_equation_l264_264141

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l264_264141


namespace percentage_value_l264_264087

theorem percentage_value (M : ℝ) (h : (25 / 100) * M = (55 / 100) * 1500) : M = 3300 :=
by
  sorry

end percentage_value_l264_264087


namespace solve_for_x_l264_264257

theorem solve_for_x (x y z w : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) 
(h3 : x * z + y * w = 50) (h4 : z - w = 5) : x = 20 := 
by 
  sorry

end solve_for_x_l264_264257


namespace breadth_halved_of_percentage_change_area_l264_264301

theorem breadth_halved_of_percentage_change_area {L B B' : ℝ} (h : 0 < L ∧ 0 < B) 
  (h1 : L / 2 * B' = 0.5 * (L * B)) : B' = 0.5 * B :=
sorry

end breadth_halved_of_percentage_change_area_l264_264301


namespace gcd_gt_one_l264_264066

-- Defining the given conditions and the statement to prove
theorem gcd_gt_one (a b x y : ℕ) (h : (a^2 + b^2) ∣ (a * x + b * y)) : 
  Nat.gcd (x^2 + y^2) (a^2 + b^2) > 1 := 
sorry

end gcd_gt_one_l264_264066


namespace digit_B_condition_l264_264874

theorem digit_B_condition {B : ℕ} (h10 : ∃ d : ℕ, 58709310 = 10 * d)
  (h5 : ∃ e : ℕ, 58709310 = 5 * e)
  (h6 : ∃ f : ℕ, 58709310 = 6 * f)
  (h4 : ∃ g : ℕ, 58709310 = 4 * g)
  (h3 : ∃ h : ℕ, 58709310 = 3 * h)
  (h2 : ∃ i : ℕ, 58709310 = 2 * i) :
  B = 0 := by
  sorry

end digit_B_condition_l264_264874


namespace sum_of_arithmetic_sequence_l264_264327

theorem sum_of_arithmetic_sequence (a d : ℤ) (n : ℕ) (h1 : a = -3) (h2 : d = 7) (h3 : n = 10) :
  let aₙ := a + (n - 1) * d in
  (a + aₙ) * n / 2 = 285 :=
by
  sorry

end sum_of_arithmetic_sequence_l264_264327


namespace area_of_triangle_formed_by_tangency_points_l264_264663

theorem area_of_triangle_formed_by_tangency_points :
  let r1 := 1
  let r2 := 3
  let r3 := 5
  let O1O2 := r1 + r2
  let O2O3 := r2 + r3
  let O1O3 := r1 + r3
  let s := (O1O2 + O2O3 + O1O3) / 2
  let A := Real.sqrt (s * (s - O1O2) * (s - O2O3) * (s - O1O3))
  let r := A / s
  r^2 = 5 / 3 := 
by
  sorry

end area_of_triangle_formed_by_tangency_points_l264_264663


namespace second_less_than_first_l264_264904

-- Define the given conditions
def third_number : ℝ := sorry
def first_number : ℝ := 0.65 * third_number
def second_number : ℝ := 0.58 * third_number

-- Problem statement: Prove that the second number is approximately 10.77% less than the first number
theorem second_less_than_first : 
  (first_number - second_number) / first_number * 100 = 10.77 := 
sorry

end second_less_than_first_l264_264904


namespace sweets_distribution_l264_264278

theorem sweets_distribution (S X : ℕ) (h1 : S = 112 * X) (h2 : S = 80 * (X + 6)) :
  X = 15 := 
by
  sorry

end sweets_distribution_l264_264278


namespace Kyle_rose_cost_l264_264832

/-- Given the number of roses Kyle picked last year, the number of roses he picked this year, 
and the cost of one rose, prove that the total cost he has to spend to buy the remaining roses 
is correct. -/
theorem Kyle_rose_cost (last_year_roses this_year_roses total_roses_needed cost_per_rose : ℕ)
    (h_last_year_roses : last_year_roses = 12) 
    (h_this_year_roses : this_year_roses = last_year_roses / 2) 
    (h_total_roses_needed : total_roses_needed = 2 * last_year_roses) 
    (h_cost_per_rose : cost_per_rose = 3) : 
    (total_roses_needed - this_year_roses) * cost_per_rose = 54 := 
by
sorry

end Kyle_rose_cost_l264_264832


namespace power_modulus_difference_l264_264664

theorem power_modulus_difference (m : ℤ) :
  (51 % 6 = 3) → (9 % 6 = 3) → ((51 : ℤ)^1723 - (9 : ℤ)^1723) % 6 = 0 :=
by 
  intros h1 h2
  sorry

end power_modulus_difference_l264_264664


namespace notebooks_to_sell_to_earn_profit_l264_264918

-- Define the given conditions
def notebooks_purchased : ℕ := 2000
def cost_per_notebook : ℚ := 0.15
def selling_price_per_notebook : ℚ := 0.30
def desired_profit : ℚ := 120

-- Define the total cost
def total_cost := notebooks_purchased * cost_per_notebook

-- Define the total revenue needed
def total_revenue_needed := total_cost + desired_profit

-- Define the number of notebooks to be sold to achieve the total revenue
def notebooks_to_sell := total_revenue_needed / selling_price_per_notebook

-- Prove that the number of notebooks to be sold is 1400 to make a profit of $120
theorem notebooks_to_sell_to_earn_profit : notebooks_to_sell = 1400 := 
by {
  sorry
}

end notebooks_to_sell_to_earn_profit_l264_264918


namespace parabola_focus_distance_l264_264976

noncomputable def distance_to_focus (p : ℝ) (M : ℝ × ℝ) : ℝ :=
  let focus := (p, 0)
  let (x1, y1) := M
  let (x2, y2) := focus
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) + p

theorem parabola_focus_distance
  (M : ℝ × ℝ) (p : ℝ)
  (hM : M = (2, 2))
  (hp : p = 1) :
  distance_to_focus p M = Real.sqrt 5 + 1 :=
by
  sorry

end parabola_focus_distance_l264_264976


namespace sin_alpha_value_l264_264070

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan (π - α) + 3 = 0) : 
  Real.sin α = 3 * Real.sqrt 10 / 10 := 
by
  sorry

end sin_alpha_value_l264_264070


namespace solve_equation_l264_264159

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l264_264159


namespace kiril_konstantinovich_age_is_full_years_l264_264418

theorem kiril_konstantinovich_age_is_full_years
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  (years = 48) →
  (months = 48) →
  (weeks = 48) →
  (days = 48) →
  (hours = 48) →
  Int.floor (
    years + 
    (months / 12 : ℝ) + 
    (weeks * 7 / 365 : ℝ) + 
    (days / 365 : ℝ) + 
    (hours / (24 * 365) : ℝ)
  ) = 53 :=
by
  intro hyears hmonths hweeks hdays hhours
  rw [hyears, hmonths, hweeks, hdays, hhours]
  sorry

end kiril_konstantinovich_age_is_full_years_l264_264418


namespace find_lines_passing_through_A_parallel_to_beta_and_angle_with_alpha_l264_264643

-- Define the given elements as variables
variable (A : Point) (alpha beta : Plane) (theta : ℝ)

-- Statement of the problem
theorem find_lines_passing_through_A_parallel_to_beta_and_angle_with_alpha :
  ∃ (B C : Point),
    line_through A B ∧ line_through A C ∧
    parallel_to (line_through A B) beta ∧ parallel_to (line_through A C) beta ∧
    angle_with_plane (line_through A B) alpha = θ ∧ angle_with_plane (line_through A C) alpha = θ :=
by
  sorry

end find_lines_passing_through_A_parallel_to_beta_and_angle_with_alpha_l264_264643


namespace thursday_occurs_five_times_l264_264288

-- Definitions and assumptions based on conditions
def july_has_five_tuesdays (N : ℕ) : Prop :=
∃ (day : ℕ), (1 ≤ day ∧ day ≤ 3) ∧
  (day + 7 ≤ 31) ∧ (day + 14 ≤ 31) ∧ (day + 21 ≤ 31) ∧ (day + 28 ≤ 31)

def august_has_31_days : Prop := true

-- The theorem stating the question
theorem thursday_occurs_five_times (N : ℕ) (h1 : july_has_five_tuesdays N) (h2 : august_has_31_days) :
  ∃ (day : ℕ), (1 ≤ day ∧ day ≤ 7) ∧
    ((day + 4) mod 7 = 4) ∧
    list_count (list.map (λ d, (d + (1 - day.days))) (list.range 31)) 4 = 5 :=
sorry

end thursday_occurs_five_times_l264_264288


namespace series_evaluation_l264_264520

noncomputable def series_sum : ℝ :=
  ∑' m : ℕ, (∑' n : ℕ, (m^2 * n) / (3^m * (n * 3^m + m * 3^n)))

theorem series_evaluation : series_sum = 9 / 32 :=
by
  sorry

end series_evaluation_l264_264520


namespace water_consumption_total_l264_264880

def number_of_cows : ℕ := 40
def water_per_cow_per_day : ℕ := 80
def sheep_multiplicative_factor : ℕ := 10
def water_factor_cow_to_sheep : ℕ := 1 / 4
def days_in_week : ℕ := 7

theorem water_consumption_total :
  let cows_water_per_week := number_of_cows * water_per_cow_per_day * days_in_week in
  let number_of_sheep := number_of_cows * sheep_multiplicative_factor in
  let water_per_sheep_per_day := water_per_cow_per_day * water_factor_cow_to_sheep in
  let sheep_water_per_week := number_of_sheep * water_per_sheep_per_day * days_in_week in
  cows_water_per_week + sheep_water_per_week = 78400 := sorry

end water_consumption_total_l264_264880


namespace square_area_is_8_point_0_l264_264779

theorem square_area_is_8_point_0 (A B C D E F : ℝ) 
    (h_square : E + F = 4)
    (h_diag : 1 + 2 + 1 = 4) : 
    ∃ (s : ℝ), s^2 = 8 :=
by
  sorry

end square_area_is_8_point_0_l264_264779


namespace length_of_RS_l264_264868

-- Define the lengths of the edges of the tetrahedron
def edge_lengths : List ℕ := [9, 16, 22, 31, 39, 48]

-- Given the edge PQ has length 48
def PQ_length : ℕ := 48

-- We need to prove that the length of edge RS is 9
theorem length_of_RS :
  ∃ (RS : ℕ), RS = 9 ∧
  ∃ (PR QR PS SQ : ℕ),
  [PR, QR, PS, SQ] ⊆ edge_lengths ∧
  PR + QR > PQ_length ∧
  PR + PQ_length > QR ∧
  QR + PQ_length > PR ∧
  PS + SQ > PQ_length ∧
  PS + PQ_length > SQ ∧
  SQ + PQ_length > PS :=
by
  sorry

end length_of_RS_l264_264868


namespace matches_start_with_l264_264599

-- Let M be the number of matches Nate started with
variables (M : ℕ)

-- Given conditions
def dropped_creek (dropped : ℕ) := dropped = 10
def eaten_by_dog (eaten : ℕ) := eaten = 2 * 10
def matches_left (final_matches : ℕ) := final_matches = 40

-- Prove that the number of matches Nate started with is 70
theorem matches_start_with 
  (h1 : dropped_creek 10)
  (h2 : eaten_by_dog 20)
  (h3 : matches_left 40) 
  : M = 70 :=
sorry

end matches_start_with_l264_264599


namespace non_receivers_after_2020_candies_l264_264630

noncomputable def count_non_receivers (k n : ℕ) : ℕ := 
sorry

theorem non_receivers_after_2020_candies :
  count_non_receivers 73 2020 = 36 :=
sorry

end non_receivers_after_2020_candies_l264_264630


namespace purely_imaginary_a_eq_2_l264_264864

theorem purely_imaginary_a_eq_2 (a : ℝ) (h : (2 - a) / 2 = 0) : a = 2 :=
sorry

end purely_imaginary_a_eq_2_l264_264864


namespace probability_mask_with_ear_loops_l264_264783

-- Definitions from the conditions
def production_ratio_regular : ℝ := 0.8
def production_ratio_surgical : ℝ := 0.2
def proportion_ear_loops_regular : ℝ := 0.1
def proportion_ear_loops_surgical : ℝ := 0.2

-- Theorem statement based on the translated proof problem
theorem probability_mask_with_ear_loops :
  production_ratio_regular * proportion_ear_loops_regular +
  production_ratio_surgical * proportion_ear_loops_surgical = 0.12 :=
by
  -- Proof omitted
  sorry

end probability_mask_with_ear_loops_l264_264783


namespace series_converges_to_one_l264_264366

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n : ℝ)^2 - 2 * (n : ℝ) + 1) / ((n : ℝ)^4 - (n : ℝ)^3 + (n : ℝ)^2 - (n : ℝ) + 1) else 0

theorem series_converges_to_one : series_sum = 1 := 
  sorry

end series_converges_to_one_l264_264366


namespace equation_of_line_AB_l264_264122

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨2, -1⟩

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the center C
def C : Point := ⟨1, 0⟩

-- The equation of line AB we want to verify
def line_AB (P : Point) := P.x - P.y - 3 = 0

-- The theorem to prove
theorem equation_of_line_AB :
  (circle_eq P.x P.y ∧ P = ⟨2, -1⟩ ∧ C = ⟨1, 0⟩) → line_AB P :=
by
  sorry

end equation_of_line_AB_l264_264122


namespace part1_part2_l264_264633

-- Definitions and conditions
def total_length : ℝ := 64
def ratio_larger_square_area : ℝ := 2.25
def total_area : ℝ := 160

-- Given problem parts
theorem part1 (x : ℝ) (h : (64 - 4 * x) / 4 * (64 - 4 * x) / 4 = 2.25 * x * x) : x = 6.4 :=
by
  -- Proof needs to be provided
  sorry

theorem part2 (y : ℝ) (h : (16 - y) * (16 - y) + y * y = 160) : y = 4 ∧ (64 - 4 * y) = 48 :=
by
  -- Proof needs to be provided
  sorry

end part1_part2_l264_264633


namespace kendra_total_earnings_l264_264731

theorem kendra_total_earnings (laurel2014 kendra2014 kendra2015 : ℕ) 
  (h1 : laurel2014 = 30000)
  (h2 : kendra2014 = laurel2014 - 8000)
  (h3 : kendra2015 = 1.20 * laurel2014) :
  kendra2014 + kendra2015 = 58000 :=
by
  sorry

end kendra_total_earnings_l264_264731


namespace prob_not_all_same_l264_264013

-- Definition: Rolling five fair 6-sided dice
def all_outcomes := Finset.pi (Finset.range 5) (λ _, Finset.range 6)
def same_outcome_count := 6
def total_outcomes := 6 ^ 5

-- Theorem: The probability that not all five dice show the same number is 1295/1296
theorem prob_not_all_same :
  (total_outcomes - same_outcome_count) / total_outcomes = 1295 / 1296 :=
by
  sorry

end prob_not_all_same_l264_264013


namespace sum_of_coefficients_1_to_7_l264_264556

noncomputable def polynomial_expression (x : ℝ) : ℝ :=
  (1 + x) * (1 - 2 * x) ^ 7

theorem sum_of_coefficients_1_to_7 :
  let a₀ := (polynomial_expression 0)
  let a₈ := ((Nat.choose 7 7 : ℝ) * (-2) ^ 7)
  let sum := a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇
  (polynomial_expression 1) = -2 →
  sum = 125 := by
  sorry

end sum_of_coefficients_1_to_7_l264_264556


namespace union_complement_U_A_B_l264_264979

def U : Set Int := {-1, 0, 1, 2, 3}

def A : Set Int := {-1, 0, 1}

def B : Set Int := {0, 1, 2}

def complement_U_A : Set Int := {u | u ∈ U ∧ u ∉ A}

theorem union_complement_U_A_B : (complement_U_A ∪ B) = {0, 1, 2, 3} :=
by
  sorry

end union_complement_U_A_B_l264_264979


namespace minimum_a_l264_264255

theorem minimum_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - (x - a) * |x - a| - 2 ≥ 0) → a ≥ Real.sqrt 3 := 
by 
  sorry

end minimum_a_l264_264255


namespace max_students_gcd_l264_264517

def numPens : Nat := 1802
def numPencils : Nat := 1203
def numErasers : Nat := 1508
def numNotebooks : Nat := 2400

theorem max_students_gcd : Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numErasers) numNotebooks = 1 := by
  sorry

end max_students_gcd_l264_264517


namespace solve_equation_l264_264133

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l264_264133


namespace shorter_leg_of_right_triangle_with_hypotenuse_65_l264_264996

theorem shorter_leg_of_right_triangle_with_hypotenuse_65 (a b : ℕ) (h : a^2 + b^2 = 65^2) : a = 16 ∨ b = 16 :=
by sorry

end shorter_leg_of_right_triangle_with_hypotenuse_65_l264_264996


namespace tom_bought_8_kg_of_apples_l264_264901

/-- 
   Given:
   - The cost of apples is 70 per kg.
   - 9 kg of mangoes at a rate of 55 per kg.
   - Tom paid a total of 1055.

   Prove that Tom purchased 8 kg of apples.
 -/
theorem tom_bought_8_kg_of_apples 
  (A : ℕ) 
  (h1 : 70 * A + 55 * 9 = 1055) : 
  A = 8 :=
sorry

end tom_bought_8_kg_of_apples_l264_264901


namespace sally_quarters_l264_264744

theorem sally_quarters (initial_quarters spent_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 760) 
  (h2 : spent_quarters = 418) 
  (calc_final : final_quarters = initial_quarters - spent_quarters) : 
  final_quarters = 342 := 
by 
  rw [h1, h2] at calc_final 
  exact calc_final

end sally_quarters_l264_264744


namespace solve_fractional_equation_l264_264142

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l264_264142


namespace matilda_jellybeans_l264_264594

theorem matilda_jellybeans (steve_jellybeans : ℕ) (h_steve : steve_jellybeans = 84)
  (h_matt : ℕ) (h_matt_calc : h_matt = 10 * steve_jellybeans)
  (h_matilda : ℕ) (h_matilda_calc : h_matilda = h_matt / 2) :
  h_matilda = 420 := by
  sorry

end matilda_jellybeans_l264_264594


namespace pure_imaginary_number_solution_l264_264093

-- Definition of the problem
theorem pure_imaginary_number_solution (a : ℝ) (h1 : a^2 - 4 = 0) (h2 : a^2 - 3 * a + 2 ≠ 0) : a = -2 :=
sorry

end pure_imaginary_number_solution_l264_264093


namespace lucy_flour_used_l264_264593

theorem lucy_flour_used
  (initial_flour : ℕ := 500)
  (final_flour : ℕ := 130)
  (flour_needed_to_buy : ℤ := 370)
  (used_flour : ℕ) :
  initial_flour - used_flour = 2 * final_flour → used_flour = 240 :=
by
  sorry

end lucy_flour_used_l264_264593


namespace temperature_on_friday_l264_264294

-- Define the temperatures on different days
variables (T W Th F : ℝ)

-- Define the conditions
def condition1 : Prop := (T + W + Th) / 3 = 32
def condition2 : Prop := (W + Th + F) / 3 = 34
def condition3 : Prop := T = 38

-- State the theorem to prove the temperature on Friday
theorem temperature_on_friday (h1 : condition1 T W Th) (h2 : condition2 W Th F) (h3 : condition3 T) : F = 44 :=
  sorry

end temperature_on_friday_l264_264294


namespace incorrect_pair_l264_264456

def roots_of_polynomial (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

theorem incorrect_pair : ¬ ∃ x : ℝ, (y = x - 1 ∧ y = x + 1 ∧ roots_of_polynomial x) :=
by
  sorry

end incorrect_pair_l264_264456


namespace average_salary_of_managers_l264_264484

theorem average_salary_of_managers 
    (num_managers num_associates : ℕ) 
    (avg_salary_associates avg_salary_company : ℝ) 
    (H_managers : num_managers = 15) 
    (H_associates : num_associates = 75) 
    (H_avg_associates : avg_salary_associates = 30000) 
    (H_avg_company : avg_salary_company = 40000) : 
    ∃ M : ℝ, 15 * M + 75 * 30000 = 90 * 40000 ∧ M = 90000 := 
by
    use 90000
    rw [H_managers, H_associates, H_avg_associates, H_avg_company]
    split
    · linarith
    · rfl

end average_salary_of_managers_l264_264484


namespace f_sum_2018_2019_l264_264450

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom even_shifted_function (x : ℝ) : f (x + 1) = f (-x + 1)
axiom f_neg1 : f (-1) = -1

theorem f_sum_2018_2019 : f 2018 + f 2019 = -1 :=
by sorry

end f_sum_2018_2019_l264_264450


namespace andrew_beth_heads_probability_l264_264037

theorem andrew_beth_heads_probability :
  let X := binomial 5 (1/2 : ℝ)
  let Y := binomial 6 (1/2 : ℝ)
  P(X >= Y) = 0.5 :=
sorry

end andrew_beth_heads_probability_l264_264037


namespace base8_subtraction_l264_264658

theorem base8_subtraction : (53 - 26 : ℕ) = 25 :=
by sorry

end base8_subtraction_l264_264658


namespace fairfield_middle_school_geography_players_l264_264932

/-- At Fairfield Middle School, there are 24 players on the football team.
All players are enrolled in at least one of the subjects: history or geography.
There are 10 players taking history and 6 players taking both subjects.
We need to prove that the number of players taking geography is 20. -/
theorem fairfield_middle_school_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_subjects_players : ℕ)
  (h1 : total_players = 24)
  (h2 : history_players = 10)
  (h3 : both_subjects_players = 6) :
  total_players - (history_players - both_subjects_players) = 20 :=
by {
  sorry
}

end fairfield_middle_school_geography_players_l264_264932


namespace determinant_matrix_A_l264_264365

open Matrix

def matrix_A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, 0, -2], ![1, 3, 4], ![0, -1, 1]]

theorem determinant_matrix_A :
  det matrix_A = 33 :=
by
  sorry

end determinant_matrix_A_l264_264365


namespace polynomial_roots_l264_264382

theorem polynomial_roots (k r : ℝ) (hk_pos : k > 0) 
(h_sum : r + 1 = 2 * k) (h_prod : r * 1 = k) : 
  r = 1 ∧ (∀ x, (x - 1) * (x - 1) = x^2 - 2 * x + 1) := 
by 
  sorry

end polynomial_roots_l264_264382


namespace midpoint_x_coordinate_l264_264960

theorem midpoint_x_coordinate (M N : ℝ × ℝ)
  (hM : M.1 ^ 2 = 4 * M.2)
  (hN : N.1 ^ 2 = 4 * N.2)
  (h_dist : (Real.sqrt ((M.1 - 1)^2 + M.2^2)) + (Real.sqrt ((N.1 - 1)^2 + N.2^2)) = 6) :
  (M.1 + N.1) / 2 = 2 := 
sorry

end midpoint_x_coordinate_l264_264960


namespace roots_poly_sum_l264_264875

noncomputable def Q (z : ℂ) (a b c : ℝ) : ℂ := z^3 + (a:ℂ)*z^2 + (b:ℂ)*z + (c:ℂ)

theorem roots_poly_sum (a b c : ℝ) (u : ℂ)
  (h1 : u.im = 0) -- Assuming u is a real number
  (h2 : Q (u + 5 * Complex.I) a b c = 0)
  (h3 : Q (u + 15 * Complex.I) a b c = 0)
  (h4 : Q (2 * u - 6) a b c = 0) :
  a + b + c = -196 := by
  sorry

end roots_poly_sum_l264_264875


namespace sum_is_square_l264_264084

theorem sum_is_square (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : Nat.gcd a b = 1) (h5 : Nat.gcd b c = 1) (h6 : Nat.gcd c a = 1) 
  (h7 : (1:ℚ)/a + (1:ℚ)/b = (1:ℚ)/c) : ∃ k : ℕ, a + b = k ^ 2 := 
by 
  sorry

end sum_is_square_l264_264084


namespace joe_height_l264_264130

theorem joe_height (S J : ℕ) (h1 : S + J = 120) (h2 : J = 2 * S + 6) : J = 82 :=
by
  sorry

end joe_height_l264_264130


namespace relationship_a_b_l264_264992

-- Definitions of the two quadratic equations having a single common root
def has_common_root (a b : ℝ) : Prop :=
  ∃ t : ℝ, (t^2 + a * t + b = 0) ∧ (t^2 + b * t + a = 0)

-- Theorem stating the relationship between a and b
theorem relationship_a_b (a b : ℝ) (h : has_common_root a b) : a ≠ b → a + b + 1 = 0 :=
by sorry

end relationship_a_b_l264_264992


namespace inequality_solution_l264_264834

theorem inequality_solution (x : ℝ) : (x^3 - 12*x^2 + 36*x > 0) ↔ (0 < x ∧ x < 6) ∨ (x > 6) := by
  sorry

end inequality_solution_l264_264834


namespace joe_height_l264_264131

theorem joe_height (S J : ℕ) (h1 : S + J = 120) (h2 : J = 2 * S + 6) : J = 82 :=
by
  sorry

end joe_height_l264_264131


namespace entree_cost_14_l264_264704

theorem entree_cost_14 (D E : ℝ) (h1 : D + E = 23) (h2 : E = D + 5) : E = 14 :=
sorry

end entree_cost_14_l264_264704


namespace seq_eventually_reaches_one_l264_264788

theorem seq_eventually_reaches_one (a : ℕ → ℤ) (h₁ : a 1 > 0) :
  (∀ n, n % 4 = 0 → a (n + 1) = a n / 2) →
  (∀ n, n % 4 = 1 → a (n + 1) = 3 * a n + 1) →
  (∀ n, n % 4 = 2 → a (n + 1) = 2 * a n - 1) →
  (∀ n, n % 4 = 3 → a (n + 1) = (a n + 1) / 4) →
  ∃ m, a m = 1 :=
by
  sorry

end seq_eventually_reaches_one_l264_264788


namespace value_of_expression_l264_264714

theorem value_of_expression (a b : ℤ) (h : a - 2 * b - 3 = 0) : 9 - 2 * a + 4 * b = 3 := 
by 
  sorry

end value_of_expression_l264_264714


namespace total_amount_l264_264218

theorem total_amount (x y z total : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : y = 27) : total = 117 :=
by
  -- Proof here
  sorry

end total_amount_l264_264218


namespace sixth_root_of_7528758090625_l264_264050

theorem sixth_root_of_7528758090625 :
  (1 * 50^6 + 6 * 50^5 + 15 * 50^4 + 20 * 50^3 + 15 * 50^2 + 6 * 50 + 1)^(1 / 6) = 51 :=
by
  sorry

end sixth_root_of_7528758090625_l264_264050


namespace reflection_matrix_l264_264822

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l264_264822


namespace problem1_problem2_l264_264934

theorem problem1 : 27^((2:ℝ)/(3:ℝ)) - 2^(Real.logb 2 3) * Real.logb 2 (1/8) = 18 := 
by
  sorry -- proof omitted

theorem problem2 : 1/(Real.sqrt 5 - 2) - (Real.sqrt 5 + 2)^0 - Real.sqrt ((2 - Real.sqrt 5)^2) = 2*(Real.sqrt 5 - 1) := 
by
  sorry -- proof omitted

end problem1_problem2_l264_264934


namespace values_of_d_divisible_by_13_l264_264955

def base8to10 (d : ℕ) : ℕ := 3 * 8^3 + d * 8^2 + d * 8 + 7

theorem values_of_d_divisible_by_13 (d : ℕ) (h : d ≥ 0 ∧ d < 8) :
  (1543 + 72 * d) % 13 = 0 ↔ d = 1 ∨ d = 2 :=
by sorry

end values_of_d_divisible_by_13_l264_264955


namespace necessary_but_not_sufficient_l264_264021

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 3 * x - 4 = 0) -> (x = 4 ∨ x = -1) ∧ ¬(x = 4 ∨ x = -1 -> x = 4) :=
by sorry

end necessary_but_not_sufficient_l264_264021


namespace trigonometric_identity_l264_264063

theorem trigonometric_identity (α : ℝ) (h : Real.tan (π + α) = 2) :
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π + α) - Real.cos (π + α)) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l264_264063


namespace area_of_inscribed_rectangle_l264_264796

variable (b h x : ℝ)

def is_isosceles_triangle (b h : ℝ) : Prop :=
  b > 0 ∧ h > 0

def is_inscribed_rectangle (b h x : ℝ) : Prop :=
  x > 0 ∧ x < h 

theorem area_of_inscribed_rectangle (h_pos : is_isosceles_triangle b h) 
                                    (rect_pos : is_inscribed_rectangle b h x) : 
                                    ∃ A : ℝ, A = (b / (2 * h)) * x ^ 2 :=
by
  sorry

end area_of_inscribed_rectangle_l264_264796


namespace total_cost_l264_264492

theorem total_cost (cost_pencil cost_pen : ℕ) 
(h1 : cost_pen = cost_pencil + 9) 
(h2 : cost_pencil = 2) : 
cost_pencil + cost_pen = 13 := 
by 
  -- Proof would go here 
  sorry

end total_cost_l264_264492


namespace range_of_m_l264_264715

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 1) * x + 1

theorem range_of_m (m : ℝ) : (∀ x, x ≤ 1 → f m x ≥ f m 1) ↔ 0 ≤ m ∧ m ≤ 1 / 3 := by
  sorry

end range_of_m_l264_264715


namespace polynomial_roots_l264_264545

theorem polynomial_roots :
  (∀ x, x^3 - 3 * x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := 
by
  sorry

end polynomial_roots_l264_264545


namespace product_of_x_values_l264_264566

noncomputable def find_product_of_x : ℚ :=
  let x1 := -20
  let x2 := -20 / 7
  (x1 * x2)

theorem product_of_x_values :
  (∃ x : ℚ, abs (20 / x + 4) = 3) ->
  find_product_of_x = 400 / 7 :=
by
  sorry

end product_of_x_values_l264_264566


namespace coral_third_week_pages_l264_264943

theorem coral_third_week_pages :
  let total_pages := 600
  let week1_read := total_pages / 2
  let remaining_after_week1 := total_pages - week1_read
  let week2_read := remaining_after_week1 * 0.30
  let remaining_after_week2 := remaining_after_week1 - week2_read
  remaining_after_week2 = 210 :=
by
  sorry

end coral_third_week_pages_l264_264943


namespace remaining_hair_length_is_1_l264_264281

-- Variables to represent the inches of hair
variable (initial_length cut_length : ℕ)

-- Given initial length and cut length
def initial_length_is_14 (initial_length : ℕ) := initial_length = 14
def cut_length_is_13 (cut_length : ℕ) := cut_length = 13

-- Definition of the remaining hair length
def remaining_length (initial_length cut_length : ℕ) := initial_length - cut_length

-- Main theorem: Proving the remaining hair length is 1 inch
theorem remaining_hair_length_is_1 : initial_length_is_14 initial_length → cut_length_is_13 cut_length → remaining_length initial_length cut_length = 1 := by
  intros h1 h2
  rw [initial_length_is_14, cut_length_is_13] at *
  simp [remaining_length]
  sorry

end remaining_hair_length_is_1_l264_264281


namespace max_a9_l264_264885

theorem max_a9 (a : Fin 18 → ℕ) (h_pos: ∀ i, 1 ≤ a i) (h_incr: ∀ i j, i < j → a i < a j) (h_sum: (Finset.univ : Finset (Fin 18)).sum a = 2001) : a 8 ≤ 192 :=
by
  -- Proof goes here
  sorry

end max_a9_l264_264885


namespace three_person_subcommittees_from_eight_l264_264406

theorem three_person_subcommittees_from_eight :
  (Nat.choose 8 3) = 56 :=
by
  sorry

end three_person_subcommittees_from_eight_l264_264406


namespace apples_in_first_group_l264_264748

variable (A O : ℝ) (X : ℕ)

-- Given conditions
axiom h1 : A = 0.21
axiom h2 : X * A + 3 * O = 1.77
axiom h3 : 2 * A + 5 * O = 1.27 

-- Goal: Prove that the number of apples in the first group is 6
theorem apples_in_first_group : X = 6 := 
by 
  sorry

end apples_in_first_group_l264_264748


namespace e_is_dq_sequence_l264_264384

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a₀, ∀ n, a n = a₀ + n * d

def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ q b₀, q > 0 ∧ ∀ n, b n = b₀ * q^n

def is_dq_sequence (c : ℕ → ℕ) : Prop :=
  ∃ a b, is_arithmetic_sequence a ∧ is_geometric_sequence b ∧ ∀ n, c n = a n + b n

def e (n : ℕ) : ℕ :=
  n + 2^n

theorem e_is_dq_sequence : is_dq_sequence e :=
  sorry

end e_is_dq_sequence_l264_264384


namespace coordinates_of_point_P_l264_264245

theorem coordinates_of_point_P 
  (x y : ℝ)
  (h1 : y = x^3 - x)
  (h2 : (3 * x^2 - 1) = 2)
  (h3 : ∀ x y, x + 2 * y = 0 → ∃ m, -1/(m) = 2) :
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end coordinates_of_point_P_l264_264245


namespace total_amount_before_brokerage_l264_264890

variable (A : ℝ)

theorem total_amount_before_brokerage 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 106.25) 
  (h2 : brokerage_rate = 1 / 400) :
  A = 42500 / 399 :=
by
  sorry

end total_amount_before_brokerage_l264_264890


namespace solve_fractional_equation_l264_264152

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l264_264152


namespace jameson_badminton_medals_l264_264586

theorem jameson_badminton_medals:
  ∀ (total track: ℕ) (swimming_multiple: ℕ),
  total = 20 → track = 5 → swimming_multiple = 2 →
  ∃ (badminton: ℕ), badminton = 20 - (track + swimming_multiple * track) ∧ badminton = 5 :=
by
  intros total track swimming_multiple ht ht5 hsm
  use 5
  simp [ht5, hsm, ht]
  sorry

end jameson_badminton_medals_l264_264586


namespace eggs_collected_week_l264_264810

def num_chickens : ℕ := 6
def num_ducks : ℕ := 4
def num_geese : ℕ := 2
def eggs_per_chicken : ℕ := 3
def eggs_per_duck : ℕ := 2
def eggs_per_goose : ℕ := 1

def eggs_per_day (num_birds eggs_per_bird : ℕ) : ℕ := num_birds * eggs_per_bird

def eggs_collected_monday_to_saturday : ℕ :=
  6 * (eggs_per_day num_chickens eggs_per_chicken +
       eggs_per_day num_ducks eggs_per_duck +
       eggs_per_day num_geese eggs_per_goose)

def eggs_collected_sunday : ℕ :=
  eggs_per_day num_chickens (eggs_per_chicken - 1) +
  eggs_per_day num_ducks (eggs_per_duck - 1) +
  eggs_per_day num_geese (eggs_per_goose - 1)

def total_eggs_collected : ℕ :=
  eggs_collected_monday_to_saturday + eggs_collected_sunday

theorem eggs_collected_week : total_eggs_collected = 184 :=
by sorry

end eggs_collected_week_l264_264810


namespace original_side_length_l264_264219

theorem original_side_length (x : ℝ) 
  (h1 : (x - 4) * (x - 3) = 120) : x = 12 :=
sorry

end original_side_length_l264_264219


namespace picture_edge_distance_l264_264493

theorem picture_edge_distance 
    (wall_width : ℕ) 
    (picture_width : ℕ) 
    (centered : Bool) 
    (h_w : wall_width = 22) 
    (h_p : picture_width = 4) 
    (h_c : centered = true) : 
    ∃ (distance : ℕ), distance = 9 := 
by
  sorry

end picture_edge_distance_l264_264493


namespace entree_cost_14_l264_264706

-- Define the conditions as given in part a)
def total_cost (e d : ℕ) : Prop := e + d = 23
def entree_more (e d : ℕ) : Prop := e = d + 5

-- The theorem to be proved
theorem entree_cost_14 (e d : ℕ) (h1 : total_cost e d) (h2 : entree_more e d) : e = 14 := 
by 
  sorry

end entree_cost_14_l264_264706


namespace shopkeeper_loss_l264_264789

theorem shopkeeper_loss
    (total_stock : ℝ)
    (stock_sold_profit_percent : ℝ)
    (stock_profit_percent : ℝ)
    (stock_sold_loss_percent : ℝ)
    (stock_loss_percent : ℝ) :
    total_stock = 12500 →
    stock_sold_profit_percent = 0.20 →
    stock_profit_percent = 0.10 →
    stock_sold_loss_percent = 0.80 →
    stock_loss_percent = 0.05 →
    ∃ loss_amount, loss_amount = 250 :=
by
  sorry

end shopkeeper_loss_l264_264789


namespace evaluate_at_3_l264_264808

def f (x : ℝ) : ℝ := 9 * x^4 + 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem evaluate_at_3 : f 3 = 876 := by
  sorry

end evaluate_at_3_l264_264808


namespace coral_must_read_pages_to_finish_book_l264_264945

theorem coral_must_read_pages_to_finish_book
  (total_pages first_week_read second_week_percentage pages_remaining first_week_left second_week_read : ℕ)
  (initial_pages_read : ℕ := total_pages / 2)
  (remaining_after_first_week : ℕ := total_pages - initial_pages_read)
  (read_second_week : ℕ := remaining_after_first_week * second_week_percentage / 100)
  (remaining_after_second_week : ℕ := remaining_after_first_week - read_second_week)
  (final_pages_to_read : ℕ := remaining_after_second_week):
  total_pages = 600 → first_week_read = 300 → second_week_percentage = 30 →
  pages_remaining = 300 → first_week_left = 300 → second_week_read = 90 →
  remaining_after_first_week = 300 - 300 →
  remaining_after_second_week = remaining_after_first_week - second_week_read →
  third_week_read = remaining_after_second_week →
  third_week_read = 210 := by
  sorry

end coral_must_read_pages_to_finish_book_l264_264945


namespace cos_2x_quadratic_l264_264060

theorem cos_2x_quadratic (x : ℝ) (a b c : ℝ)
  (h : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0)
  (h_a : a = 4) (h_b : b = 2) (h_c : c = -1) :
  4 * (Real.cos (2 * x)) ^ 2 + 2 * Real.cos (2 * x) - 1 = 0 := sorry

end cos_2x_quadratic_l264_264060


namespace dice_probability_l264_264331

open ProbabilityTheory

-- Definitions based on conditions
def dice_faces := {n : ℕ | 1 ≤ n ∧ n ≤ 6}
def event (a b c : ℕ) : Prop := (a - 1) * (b - 1) * (c - 1) ≠ 0

-- The proof task itself
theorem dice_probability :
  ∀ (a b c : ℕ),
  (a ∈ dice_faces) ∧ (b ∈ dice_faces) ∧ (c ∈ dice_faces) →
  (probability (event a b c) = 125 / 216) :=
by
  sorry

end dice_probability_l264_264331


namespace expected_sequence_examples_geometric_sequence_expected_arithmetic_sequence_expected_l264_264272

-- Definition of an nth order expected sequence
def is_expected_sequence (n : ℕ) (a : ℕ → ℚ) : Prop :=
  (finset.range n).sum a = 0 ∧ (finset.range n).sum (λ i, |a i|) = 1

-- Part I: Examples of 3rd and 4th order expected sequences
theorem expected_sequence_examples :
  is_expected_sequence 3 (λ i, if i = 0 then -1/2 else if i = 1 then 0 else 1/2) ∧
  is_expected_sequence 4 (λ i, if i = 0 then -3/8 else if i = 1 then -1/8 else if i = 2 then 1/8 else 3/8) :=
by
  sorry

-- Part II: Geometric sequence with common ratio q that forms a 2014th order expected sequence
theorem geometric_sequence_expected (a : ℕ → ℚ) (q : ℚ) (h : is_expected_sequence 2014 a) :
  (∀ n, a (n + 1) = q * a n) → q = -1 :=
by
  sorry

-- Part III: Arithmetic sequence general formula for 2k-th order expected sequence
theorem arithmetic_sequence_expected (a : ℕ → ℚ) (k : ℕ) (h₀ : is_expected_sequence (2 * k) a)
  (h₁ : ∀ n, a (n + 1) - a n > 0) :
  ∃ d : ℚ, d = 1 / k^2 ∧ ∀ n, a n = (n / k^2) - (2 * k + 1) / (2 * k^2) :=
by
  sorry

end expected_sequence_examples_geometric_sequence_expected_arithmetic_sequence_expected_l264_264272


namespace trig_identity_l264_264514

theorem trig_identity :
  (2 * (Real.cos (Real.pi / 6)) - Real.tan (Real.pi / 3) + (Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4)) = 1/2) :=
by
  -- Here we use known trigonometric values at specific angles
  have h1 : Real.cos (Real.pi / 6) = sqrt 3 / 2 := Real.cos_pi_div_six,
  have h2 : Real.tan (Real.pi / 3) = sqrt 3 := Real.tan_pi_div_three,
  have h3 : Real.sin (Real.pi / 4) = sqrt 2 / 2 := Real.sin_pi_div_four,
  have h4 : Real.cos (Real.pi / 4) = sqrt 2 / 2 := Real.cos_pi_div_four,

  -- Use these known values to simplify the expression
  calc
    (2 * (Real.cos (Real.pi / 6)) - Real.tan (Real.pi / 3) + (Real.sin (Real.pi / 4) * Real.cos (Real.pi / 4)))
        = 2 * (sqrt 3 / 2) - sqrt 3 + (sqrt 2 / 2 * sqrt 2 / 2) : by rw [h1, h2, h3, h4]
    ... = sqrt 3 - sqrt 3 + 1/2 : by norm_num
    ... = 1/2 : by norm_num

end trig_identity_l264_264514


namespace cost_per_person_is_correct_l264_264171

-- Define the given conditions
def fee_per_30_minutes : ℕ := 4000
def bikes : ℕ := 4
def hours : ℕ := 3
def people : ℕ := 6

-- Calculate the correct answer based on the given conditions
noncomputable def cost_per_person : ℕ :=
  let fee_per_hour := 2 * fee_per_30_minutes
  let fee_per_3_hours := hours * fee_per_hour
  let total_cost := bikes * fee_per_3_hours
  total_cost / people

-- The theorem to be proved
theorem cost_per_person_is_correct : cost_per_person = 16000 := sorry

end cost_per_person_is_correct_l264_264171


namespace candidate_percentage_l264_264340

theorem candidate_percentage (P : ℚ) (votes_cast : ℚ) (loss : ℚ)
  (h1 : votes_cast = 2000) 
  (h2 : loss = 640) 
  (h3 : (P / 100) * votes_cast + (P / 100) * votes_cast + loss = votes_cast) :
  P = 34 :=
by 
  sorry

end candidate_percentage_l264_264340


namespace sum_of_tangents_slopes_at_vertices_l264_264008

noncomputable def curve (x : ℝ) := (x + 3) * (x ^ 2 + 3)

theorem sum_of_tangents_slopes_at_vertices {x_A x_B x_C : ℝ}
  (h1 : curve x_A = x_A * (x_A ^ 2 + 6 * x_A + 9) + 3)
  (h2 : curve x_B = x_B * (x_B ^ 2 + 6 * x_B + 9) + 3)
  (h3 : curve x_C = x_C * (x_C ^ 2 + 6 * x_C + 9) + 3)
  : (3 * x_A ^ 2 + 6 * x_A + 3) + (3 * x_B ^ 2 + 6 * x_B + 3) + (3 * x_C ^ 2 + 6 * x_C + 3) = 237 :=
sorry

end sum_of_tangents_slopes_at_vertices_l264_264008


namespace sufficient_condition_implies_range_of_p_l264_264777

open Set Real

theorem sufficient_condition_implies_range_of_p (p : ℝ) :
  (∀ x : ℝ, 4 * x + p < 0 → x^2 - x - 2 > 0) →
  (∃ x : ℝ, x^2 - x - 2 > 0 ∧ ¬ (4 * x + p < 0)) →
  p ∈ Set.Ici 4 :=
by
  sorry

end sufficient_condition_implies_range_of_p_l264_264777


namespace personal_income_tax_l264_264893

theorem personal_income_tax {X: ℝ} (gross_income: ℝ) (net_income: ℝ) (tax_rate: ℝ) (h1: tax_rate = 0.13) (h2: net_income = gross_income * (1 - tax_rate)) (h3: net_income = 20000) : gross_income ≈ 22989 :=
by sorry

end personal_income_tax_l264_264893


namespace greatest_area_difference_l264_264315

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) (h₁ : 2 * l₁ + 2 * w₁ = 160) (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  abs (l₁ * w₁ - l₂ * w₂) = 1521 :=
sorry

end greatest_area_difference_l264_264315


namespace polynomial_solution_l264_264954

noncomputable def p (x : ℝ) := 2 * Real.sqrt 3 * x^4 - 6

theorem polynomial_solution (x : ℝ) : 
  (p (x^4) - p (x^4 - 3) = (p x)^3 - 18) :=
by
  sorry

end polynomial_solution_l264_264954


namespace train_passes_man_in_approximately_18_seconds_l264_264648

noncomputable def length_of_train : ℝ := 330 -- meters
noncomputable def speed_of_train : ℝ := 60 -- kmph
noncomputable def speed_of_man : ℝ := 6 -- kmph

noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * (5/18)

noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_of_train + speed_of_man)

noncomputable def time_to_pass (length : ℝ) (speed : ℝ) : ℝ := length / speed

theorem train_passes_man_in_approximately_18_seconds :
  abs (time_to_pass length_of_train relative_speed_mps - 18) < 1 :=
by
  sorry

end train_passes_man_in_approximately_18_seconds_l264_264648


namespace entree_cost_l264_264709

/-- 
Prove that if the total cost is 23 and the entree costs 5 more than the dessert, 
then the cost of the entree is 14.
-/
theorem entree_cost (D : ℝ) (H1 : D + (D + 5) = 23) : D + 5 = 14 :=
by
  -- note: no proof required as per instructions
  sorry

end entree_cost_l264_264709


namespace part1_l264_264064

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 :=
by
  sorry

end part1_l264_264064


namespace arithmetic_sequence_sum_l264_264328

theorem arithmetic_sequence_sum :
  let a := -3
  let d := 7
  let n := 10
  let s := n * (2 * a + (n - 1) * d) / 2
  s = 285 :=
by
  -- Details of the proof are omitted as per instructions
  sorry

end arithmetic_sequence_sum_l264_264328


namespace prime_of_form_a2_minus_1_l264_264534

theorem prime_of_form_a2_minus_1 (a : ℕ) (p : ℕ) (ha : a ≥ 2) (hp : p = a^2 - 1) (prime_p : Nat.Prime p) : p = 3 := 
by 
  sorry

end prime_of_form_a2_minus_1_l264_264534


namespace max_area_difference_l264_264313

theorem max_area_difference (l w l' w' : ℕ) 
  (h1 : 2 * l + 2 * w = 160) 
  (h2 : 2 * l' + 2 * w' = 160) : 
  abs ((l * w) - (l' * w')) ≤ 1600 := 
by 
  sorry

end max_area_difference_l264_264313


namespace percentage_of_original_price_l264_264754
-- Define the original price and current price in terms of real numbers
def original_price : ℝ := 25
def current_price : ℝ := 20

-- Lean statement to verify the correctness of the percentage calculation
theorem percentage_of_original_price :
  (current_price / original_price) * 100 = 80 := 
by
  sorry

end percentage_of_original_price_l264_264754


namespace calculate_paving_cost_l264_264772

theorem calculate_paving_cost
  (length : ℝ) (width : ℝ) (rate_per_sq_meter : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_rate : rate_per_sq_meter = 1200) :
  (length * width * rate_per_sq_meter = 24750) :=
by
  sorry

end calculate_paving_cost_l264_264772


namespace cost_of_song_book_l264_264104

theorem cost_of_song_book 
  (flute_cost : ℝ) 
  (stand_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : flute_cost = 142.46) 
  (h2 : stand_cost = 8.89) 
  (h3 : total_cost = 158.35) : 
  total_cost - (flute_cost + stand_cost) = 7.00 := 
by 
  sorry

end cost_of_song_book_l264_264104


namespace book_price_range_l264_264332

variable (x : ℝ) -- Assuming x is a real number

theorem book_price_range 
    (hA : ¬(x ≥ 20)) 
    (hB : ¬(x ≤ 15)) : 
    15 < x ∧ x < 20 := 
by
  sorry

end book_price_range_l264_264332


namespace same_number_of_friends_l264_264743

theorem same_number_of_friends (n : ℕ) (friends : Fin n → Fin n) :
  (∃ i j : Fin n, i ≠ j ∧ friends i = friends j) :=
by
  -- The proof is omitted.
  sorry

end same_number_of_friends_l264_264743


namespace solve_equation_l264_264138

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l264_264138


namespace license_plates_count_l264_264849

theorem license_plates_count :
  let num_vowels := 5
  let num_letters := 26
  let num_odd_digits := 5
  let num_even_digits := 5
  num_vowels * num_letters * num_letters * num_odd_digits * num_even_digits = 84500 :=
by
  sorry

end license_plates_count_l264_264849


namespace average_salary_of_managers_l264_264483

theorem average_salary_of_managers (m_avg : ℝ) (assoc_avg : ℝ) (company_avg : ℝ)
  (managers : ℕ) (associates : ℕ) (total_employees : ℕ)
  (h_assoc_avg : assoc_avg = 30000) (h_company_avg : company_avg = 40000)
  (h_managers : managers = 15) (h_associates : associates = 75) (h_total_employees : total_employees = 90)
  (h_total_employees_def : total_employees = managers + associates)
  (h_total_salary_managers : ∀ m_avg, total_employees * company_avg = managers * m_avg + associates * assoc_avg) :
  m_avg = 90000 :=
by
  sorry

end average_salary_of_managers_l264_264483


namespace proposition_false_l264_264894

theorem proposition_false (x y : ℤ) (h : x + y = 5) : ¬ (x = 1 ∧ y = 4) := by 
  sorry

end proposition_false_l264_264894


namespace m_range_positive_real_number_l264_264898

theorem m_range_positive_real_number (m : ℝ) (x : ℝ) 
  (h : m * x - 1 = 2 * x) (h_pos : x > 0) : m > 2 :=
sorry

end m_range_positive_real_number_l264_264898


namespace find_A_l264_264908

theorem find_A (A : ℤ) (h : 10 + A = 15) : A = 5 := by
  sorry

end find_A_l264_264908


namespace surface_area_of_sphere_l264_264395

noncomputable def sphere_surface_area : ℝ :=
  let AB := 2
  let SA := 2
  let SB := 2
  let SC := 2
  let ABC_is_isosceles_right := true -- denotes the property
  let SABC_on_sphere := true -- denotes the property
  let R := (2 * Real.sqrt 3) / 3
  let surface_area := 4 * Real.pi * R^2
  surface_area

theorem surface_area_of_sphere : sphere_surface_area = (16 * Real.pi) / 3 := 
sorry

end surface_area_of_sphere_l264_264395


namespace find_f_neg_3_l264_264400

theorem find_f_neg_3
    (a : ℝ)
    (f : ℝ → ℝ)
    (h : ∀ x, f x = a^2 * x^3 + a * Real.sin x + abs x + 1)
    (h_f3 : f 3 = 5) :
    f (-3) = 3 :=
by
    sorry

end find_f_neg_3_l264_264400


namespace upstream_speed_l264_264644

variable (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)

def speed_of_man_in_still_water := V_m = 35
def speed_of_man_downstream := V_downstream = 45
def speed_of_man_upstream := V_upstream = 25

theorem upstream_speed
  (h1: speed_of_man_in_still_water V_m)
  (h2: speed_of_man_downstream V_downstream)
  : speed_of_man_upstream V_upstream :=
by
  -- Placeholder for the proof
  sorry

end upstream_speed_l264_264644


namespace conditions_neither_necessary_nor_sufficient_l264_264776

theorem conditions_neither_necessary_nor_sufficient :
  (¬(0 < x ∧ x < 2) ↔ (¬(-1 / 2 < x ∨ x < 1)) ∨ (¬(-1 / 2 < x ∧ x < 1))) :=
by sorry

end conditions_neither_necessary_nor_sufficient_l264_264776


namespace expression_equality_l264_264199

theorem expression_equality : 
  (∀ (x : ℝ) (a k n : ℝ), (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n → a = 6 ∧ k = -5 ∧ n = -6) → 
  ∀ (a k n : ℝ), a = 6 → k = -5 → n = -6 → a - n + k = 7 :=
by
  intro h
  intros a k n ha hk hn
  rw [ha, hk, hn]
  norm_num

end expression_equality_l264_264199


namespace distance_between_points_is_sqrt_5_l264_264101

noncomputable def distance_between_polar_points : ℝ :=
  let xA := 1 * Real.cos (3/4 * Real.pi)
  let yA := 1 * Real.sin (3/4 * Real.pi)
  let xB := 2 * Real.cos (Real.pi / 4)
  let yB := 2 * Real.sin (Real.pi / 4)
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2)

theorem distance_between_points_is_sqrt_5 :
  distance_between_polar_points = Real.sqrt 5 :=
by
  sorry

end distance_between_points_is_sqrt_5_l264_264101


namespace solve_equation_l264_264135

theorem solve_equation (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end solve_equation_l264_264135


namespace f_zero_eq_zero_f_periodic_l264_264846

def odd_function {α : Type*} [AddGroup α] (f : α → α) : Prop :=
∀ x, f (-x) = -f (x)

def symmetric_about (c : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x, f (c + x) = f (c - x)

variable (f : ℝ → ℝ)
variables (h_odd : odd_function f) (h_sym : symmetric_about 1 f)

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_periodic : ∀ x, f (x + 4) = f x :=
sorry

end f_zero_eq_zero_f_periodic_l264_264846


namespace price_of_AC_l264_264755

theorem price_of_AC (x : ℝ) (price_car price_ac : ℝ)
  (h1 : price_car = 3 * x) 
  (h2 : price_ac = 2 * x) 
  (h3 : price_car = price_ac + 500) : 
  price_ac = 1000 := sorry

end price_of_AC_l264_264755


namespace entree_cost_14_l264_264707

-- Define the conditions as given in part a)
def total_cost (e d : ℕ) : Prop := e + d = 23
def entree_more (e d : ℕ) : Prop := e = d + 5

-- The theorem to be proved
theorem entree_cost_14 (e d : ℕ) (h1 : total_cost e d) (h2 : entree_more e d) : e = 14 := 
by 
  sorry

end entree_cost_14_l264_264707


namespace number_of_valid_schedules_l264_264342

-- Definitions from conditions
def members : Finset (Fin 10) := Finset.univ
def days : Finset (Fin 5) := Finset.univ

def isValidSchedule (sched : Array (Finset (Fin 10)) 5) : Prop :=
  (sched 0).card = 2 ∧ (sched 1).card = 2 ∧ (sched 2).card = 2 ∧ (sched 3).card = 2 ∧ (sched 4).card = 2 ∧
  (∃ i, {0, 1, 2, 3, 4}.mem i ∧ A ∈ sched i ∧ B ∈ sched i) ∧
  (¬ (∃ i, {0, 1, 2, 3, 4}.mem i ∧ C ∈ sched i ∧ D ∈ sched i))

-- The proof statement
theorem number_of_valid_schedules : 
  ∑ sched in {sched : Array (Finset (Fin 10)) 5 | isValidSchedule sched}.toFinset, 
  1 = 5400 :=
sorry

end number_of_valid_schedules_l264_264342


namespace price_restoration_l264_264510

theorem price_restoration {P : ℝ} (hP : P > 0) :
  (P - 0.85 * P) / (0.85 * P) * 100 = 17.65 :=
by
  sorry

end price_restoration_l264_264510


namespace min_value_a_l264_264085

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 6 > 0 → x > a) ∧
  ¬ (∀ x : ℝ, x > a → x^2 - x - 6 > 0) ↔ a = 3 :=
sorry

end min_value_a_l264_264085


namespace determine_m_for_divisibility_by_11_l264_264052

def is_divisible_by_11 (n : ℤ) : Prop :=
  n % 11 = 0

def sum_digits_odd_pos : ℤ :=
  8 + 6 + 2 + 8

def sum_digits_even_pos (m : ℤ) : ℤ :=
  5 + m + 4

theorem determine_m_for_divisibility_by_11 :
  ∃ m : ℤ, is_divisible_by_11 (sum_digits_odd_pos - sum_digits_even_pos m) ∧ m = 4 := 
by
  sorry

end determine_m_for_divisibility_by_11_l264_264052


namespace central_angle_of_cone_development_diagram_l264_264183

-- Given conditions: radius of the base of the cone and slant height
def radius_base := 1
def slant_height := 3

-- Target theorem: prove the central angle of the lateral surface development diagram is 120 degrees
theorem central_angle_of_cone_development_diagram : 
  ∃ n : ℝ, (2 * π) = (n * π * slant_height) / 180 ∧ n = 120 :=
by
  use 120
  sorry

end central_angle_of_cone_development_diagram_l264_264183


namespace remainder_of_concatenated_numbers_l264_264734

def concatenatedNumbers : ℕ :=
  let digits := List.range (50) -- [0, 1, 2, ..., 49]
  digits.foldl (fun acc d => acc * 10 ^ (Nat.digits 10 d).length + d) 0

theorem remainder_of_concatenated_numbers :
  concatenatedNumbers % 50 = 49 :=
by
  sorry

end remainder_of_concatenated_numbers_l264_264734


namespace circle_radius_l264_264345

theorem circle_radius (k r : ℝ) (h : k > 8) 
  (h1 : r = |k - 8|)
  (h2 : r = k / Real.sqrt 5) : 
  r = 8 * Real.sqrt 5 + 8 := 
sorry

end circle_radius_l264_264345


namespace student_marks_l264_264791

theorem student_marks :
  let max_marks := 300
  let passing_percentage := 0.60
  let failed_by := 20
  let passing_marks := max_marks * passing_percentage
  let marks_obtained := passing_marks - failed_by
  marks_obtained = 160 := by
sorry

end student_marks_l264_264791


namespace min_value_of_xy_l264_264693

theorem min_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 4 * x * y - x - 2 * y = 4) : 
  xy >= 2 :=
sorry

end min_value_of_xy_l264_264693


namespace k_interval_l264_264848

noncomputable def f (x k : ℝ) : ℝ := x^2 + (1 - k) * x - k

theorem k_interval (k : ℝ) :
  (∃! x : ℝ, 2 < x ∧ x < 3 ∧ f x k = 0) ↔ (2 < k ∧ k < 3) :=
by
  sorry

end k_interval_l264_264848


namespace polynomial_roots_l264_264538

theorem polynomial_roots :
  (∀ x, x^3 - 3*x^2 - x + 3 = 0 ↔ (x = 1 ∨ x = -1 ∨ x = 3)) :=
by
  intro x
  split
  {
    intro h
    have h1 : x = 1 ∨ x = -1 ∨ x = 3
    {
      sorry
    }
    exact h1
  }
  {
    intro h
    cases h
    {
      rw h
      simp
    }
    {
      cases h
      {
        rw h
        simp
      }
      {
        rw h
        simp
      }
    }
  }

end polynomial_roots_l264_264538


namespace prob_train_or_airplane_prob_not_ship_l264_264032

variables (P : Set (Set ℝ)) -- Type for probability spaces

-- Define the probabilities for each means of transportation
def P_T : ℝ := 0.3
def P_S : ℝ := 0.1
def P_C : ℝ := 0.2
def P_A : ℝ := 0.4

-- Proving the two statements:
-- 1. Probability of going by train or airplane
theorem prob_train_or_airplane : P_T + P_A = 0.7 := 
sorry

-- 2. Probability of not going by ship
theorem prob_not_ship : 1 - P_S = 0.9 := 
sorry

end prob_train_or_airplane_prob_not_ship_l264_264032


namespace relationship_between_a_b_c_l264_264840

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_between_a_b_c_l264_264840


namespace original_average_age_older_l264_264167

theorem original_average_age_older : 
  ∀ (n : ℕ) (T : ℕ), (T = n * 40) →
  (T + 408) / (n + 12) = 36 →
  40 - 36 = 4 :=
by
  intros n T hT hNewAvg
  sorry

end original_average_age_older_l264_264167


namespace solve_equation_l264_264157

theorem solve_equation : ∀ (x : ℝ), x ≠ 1 → (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 :=
by
  intros x hx heq
  -- sorry to skip the proof
  sorry

end solve_equation_l264_264157


namespace alpha_lt_beta_of_acute_l264_264565

open Real

theorem alpha_lt_beta_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : 2 * sin α = sin α * cos β + cos α * sin β) : α < β :=
by
  sorry

end alpha_lt_beta_of_acute_l264_264565


namespace slower_speed_percentage_l264_264208

theorem slower_speed_percentage (S S' T T' D : ℝ) (h1 : T = 8) (h2 : T' = T + 24) (h3 : D = S * T) (h4 : D = S' * T') : 
  (S' / S) * 100 = 25 := by
  sorry

end slower_speed_percentage_l264_264208


namespace logan_buys_15_pounds_of_corn_l264_264803

theorem logan_buys_15_pounds_of_corn (c b : ℝ) 
    (h1 : 1.20 * c + 0.60 * b = 27) 
    (h2 : b + c = 30) : 
    c = 15.0 :=
by
  sorry

end logan_buys_15_pounds_of_corn_l264_264803


namespace john_has_48_l264_264499

variable (Ali Nada John : ℕ)

theorem john_has_48 
  (h1 : Ali + Nada + John = 67)
  (h2 : Ali = Nada - 5)
  (h3 : John = 4 * Nada) : 
  John = 48 := 
by 
  sorry

end john_has_48_l264_264499


namespace abi_suji_age_ratio_l264_264178

theorem abi_suji_age_ratio (A S : ℕ) (h1 : S = 24) 
  (h2 : (A + 3) / (S + 3) = 11 / 9) : A / S = 5 / 4 := 
by 
  sorry

end abi_suji_age_ratio_l264_264178


namespace blocks_total_l264_264005

theorem blocks_total (blocks_initial : ℕ) (blocks_added : ℕ) (total_blocks : ℕ) 
  (h1 : blocks_initial = 86) (h2 : blocks_added = 9) : total_blocks = 95 :=
by
  sorry

end blocks_total_l264_264005


namespace value_of_m_l264_264970

theorem value_of_m :
  ∀ m : ℝ, (x : ℝ) → (x^2 - 5 * x + m = (x - 3) * (x - 2)) → m = 6 :=
by
  sorry

end value_of_m_l264_264970


namespace dice_probability_l264_264781

/-- 
Given five 15-sided dice, we want to prove the probability of exactly two dice showing a two-digit number (10-15) and three dice showing a one-digit number (1-9) is equal to 108/625.
-/
theorem dice_probability:
  let p_one_digit := 9 / 15 in
  let p_two_digit := 6 / 15 in
  (∃ f : (Fin 5) → Bool,
    (∃ s : Finset (Fin 5), s.card = 2 ∧ (∀ i ∈ s, f i = true) ∧ (∀ i ∉ s, f i = false))
      ∧ (bern_prob (prob {i | f i = false}) = p_one_digit)
      ∧ (bern_prob (prob {i | f i = true}) = p_two_digit)) →
  ((5.choose 2) * (p_two_digit ^ 2) * (p_one_digit ^ 3)) = 108 / 625 :=
by
  sorry

end dice_probability_l264_264781


namespace sum_nine_terms_of_arithmetic_sequence_l264_264072

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

theorem sum_nine_terms_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_of_first_n_terms a S)
  (h3 : a 5 = 7) :
  S 9 = 63 := by
  sorry

end sum_nine_terms_of_arithmetic_sequence_l264_264072


namespace percentage_of_amount_l264_264381

theorem percentage_of_amount :
  (0.25 * 300) = 75 :=
by
  sorry

end percentage_of_amount_l264_264381


namespace weight_of_B_l264_264168

variable (W_A W_B W_C W_D : ℝ)

theorem weight_of_B (h1 : (W_A + W_B + W_C + W_D) / 4 = 60)
                    (h2 : (W_A + W_B) / 2 = 55)
                    (h3 : (W_B + W_C) / 2 = 50)
                    (h4 : (W_C + W_D) / 2 = 65) :
                    W_B = 50 :=
by sorry

end weight_of_B_l264_264168


namespace cubic_inches_in_one_cubic_foot_l264_264227

-- Definition for the given conversion between foot and inches
def foot_to_inches : ℕ := 12

-- The theorem to prove the cubic conversion
theorem cubic_inches_in_one_cubic_foot : (foot_to_inches ^ 3) = 1728 := by
  -- Skipping the actual proof
  sorry

end cubic_inches_in_one_cubic_foot_l264_264227


namespace four_digit_palindrome_divisible_by_11_probability_zero_l264_264784

theorem four_digit_palindrome_divisible_by_11_probability_zero :
  (∃ a b : ℕ, 2 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (1001 * a + 110 * b) % 11 = 0) = false :=
by sorry

end four_digit_palindrome_divisible_by_11_probability_zero_l264_264784


namespace original_number_is_10_l264_264994

theorem original_number_is_10 (x : ℝ) (h : 2 * x + 5 = x / 2 + 20) : x = 10 := 
by {
  sorry
}

end original_number_is_10_l264_264994


namespace split_fraction_l264_264163

theorem split_fraction (n d a b x y : ℤ) (h_d : d = a * b) (h_ad : a.gcd b = 1) (h_frac : (n:ℚ) / (d:ℚ) = 58 / 77) (h_eq : 11 * x + 7 * y = 58) : 
  (58:ℚ) / 77 = (4:ℚ) / 7 + (2:ℚ) / 11 :=
by
  sorry

end split_fraction_l264_264163


namespace trigonometric_proof_l264_264513

noncomputable def cos30 : ℝ := Real.sqrt 3 / 2
noncomputable def tan60 : ℝ := Real.sqrt 3
noncomputable def sin45 : ℝ := Real.sqrt 2 / 2
noncomputable def cos45 : ℝ := Real.sqrt 2 / 2

theorem trigonometric_proof :
  2 * cos30 - tan60 + sin45 * cos45 = 1 / 2 :=
by
  sorry

end trigonometric_proof_l264_264513


namespace gcd_9_factorial_7_factorial_square_l264_264234

theorem gcd_9_factorial_7_factorial_square : Nat.gcd (Nat.factorial 9) ((Nat.factorial 7) ^ 2) = 362880 :=
by
  sorry

end gcd_9_factorial_7_factorial_square_l264_264234


namespace dilation_result_l264_264449

noncomputable def dilation (c a : ℂ) (k : ℝ) : ℂ := k * (c - a) + a

theorem dilation_result :
  dilation (3 - 1* I) (1 + 2* I) 4 = 9 + 6* I :=
by
  sorry

end dilation_result_l264_264449


namespace solve_equation_l264_264137

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l264_264137


namespace parts_of_a_number_l264_264350

theorem parts_of_a_number 
  (a p q : ℝ) 
  (x y z : ℝ)
  (h1 : y + z = p * x)
  (h2 : x + y = q * z)
  (h3 : x + y + z = a) :
  x = a / (1 + p) ∧ y = a * (p * q - 1) / ((p + 1) * (q + 1)) ∧ z = a / (1 + q) := 
by 
  sorry

end parts_of_a_number_l264_264350


namespace joe_height_is_82_l264_264129

-- Given the conditions:
def Sara_height (x : ℝ) : Prop := true

def Joe_height (j : ℝ) (x : ℝ) : Prop := j = 6 + 2 * x

def combined_height (j : ℝ) (x : ℝ) : Prop := j + x = 120

-- We need to prove:
theorem joe_height_is_82 (x j : ℝ) 
  (h1 : combined_height j x)
  (h2 : Joe_height j x) :
  j = 82 := 
by 
  sorry

end joe_height_is_82_l264_264129


namespace find_positive_integer_solutions_l264_264237

-- Define the problem conditions
variable {x y z : ℕ}

-- Main theorem statement
theorem find_positive_integer_solutions 
  (h1 : Prime y)
  (h2 : ¬ 3 ∣ z)
  (h3 : ¬ y ∣ z)
  (h4 : x^3 - y^3 = z^2) : 
  x = 8 ∧ y = 7 ∧ z = 13 := 
sorry

end find_positive_integer_solutions_l264_264237


namespace opposite_z_is_E_l264_264298

noncomputable def cube_faces := ["A", "B", "C", "D", "E", "z"]

def opposite_face (net : List String) (face : String) : String :=
  if face = "z" then "E" else sorry  -- generalize this function as needed

theorem opposite_z_is_E :
  opposite_face cube_faces "z" = "E" :=
by
  sorry

end opposite_z_is_E_l264_264298


namespace reflection_matrix_over_vector_is_correct_l264_264816

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l264_264816


namespace minimum_value_of_function_l264_264985

theorem minimum_value_of_function (x : ℝ) (hx : x > 4) : 
    (∃ y : ℝ, y = x + 9 / (x - 4) ∧ (∀ z : ℝ, (∃ w : ℝ, w > 4 ∧ z = w + 9 / (w - 4)) → z ≥ 10) ∧ y = 10) :=
sorry

end minimum_value_of_function_l264_264985


namespace prime_ge_5_div_24_l264_264424

theorem prime_ge_5_div_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : p ≥ 5) : 24 ∣ p^2 - 1 := 
sorry

end prime_ge_5_div_24_l264_264424


namespace balloon_count_correct_l264_264003

def gold_balloons : ℕ := 141
def black_balloons : ℕ := 150
def silver_balloons : ℕ := 2 * gold_balloons
def total_balloons : ℕ := gold_balloons + silver_balloons + black_balloons

theorem balloon_count_correct : total_balloons = 573 := by
  sorry

end balloon_count_correct_l264_264003


namespace marge_final_plant_count_l264_264275

/-- Define the initial conditions of the garden -/
def initial_seeds : ℕ := 23
def marigold_seeds : ℕ := 10
def sunflower_seeds : ℕ := 8
def lavender_seeds : ℕ := 5
def seeds_without_growth : ℕ := 5

/-- Growth rates for each type of plant -/
def marigold_growth_rate : ℕ := 4
def sunflower_growth_rate : ℕ := 4
def lavender_growth_rate : ℕ := 3

/-- Impact of animals -/
def marigold_eaten_by_squirrels : ℕ := 2
def sunflower_eaten_by_rabbits : ℕ := 1

/-- Impact of pest control -/
def marigold_pest_control_reduction : ℕ := 0
def sunflower_pest_control_reduction : ℕ := 0
def lavender_pest_control_protected : ℕ := 2

/-- Impact of weeds -/
def weeds_strangled_plants : ℕ := 2

/-- Weeds left as plants -/
def weeds_kept_as_plants : ℕ := 1

/-- Marge's final number of plants -/
def survived_plants :=
  (marigold_growth_rate - marigold_eaten_by_squirrels - marigold_pest_control_reduction) +
  (sunflower_growth_rate - sunflower_eaten_by_rabbits - sunflower_pest_control_reduction) +
  (lavender_growth_rate - (lavender_growth_rate - lavender_pest_control_protected)) - weeds_strangled_plants

theorem marge_final_plant_count :
  survived_plants + weeds_kept_as_plants = 6 :=
by
  sorry

end marge_final_plant_count_l264_264275


namespace reflection_matrix_is_correct_l264_264825

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l264_264825


namespace arithmetic_expression_eval_l264_264659

theorem arithmetic_expression_eval : 8 / 4 - 3 - 9 + 3 * 9 = 17 :=
by
  sorry

end arithmetic_expression_eval_l264_264659


namespace range_of_a_l264_264073

theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a = 0) → a < 5 := 
by sorry

end range_of_a_l264_264073


namespace sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l264_264926

noncomputable def volume_of_spheres (V : ℝ) : ℝ :=
  V * (27 / 26)

noncomputable def volume_of_tetrahedrons (V : ℝ) : ℝ :=
  (3 * V * Real.sqrt 3) / (13 * Real.pi)

theorem sum_volumes_of_spheres (V : ℝ) : 
  (∑' n : ℕ, (V * (1/27)^n)) = volume_of_spheres V :=
sorry

theorem sum_volumes_of_tetrahedrons (V : ℝ) (r : ℝ) : 
  (∑' n : ℕ, (8/9 / Real.sqrt 3 * (r^3) * (1/27)^n * (1/26))) = volume_of_tetrahedrons V :=
sorry

end sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l264_264926


namespace find_f_x_squared_l264_264688

-- Define the function f with the given condition
noncomputable def f (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem find_f_x_squared : f (x^2) = (x^2 + 1)^2 :=
by
  sorry

end find_f_x_squared_l264_264688


namespace average_weight_of_class_l264_264203

theorem average_weight_of_class (n_boys n_girls : ℕ) (avg_weight_boys avg_weight_girls : ℝ)
    (h_boys : n_boys = 5) (h_girls : n_girls = 3)
    (h_avg_weight_boys : avg_weight_boys = 60) (h_avg_weight_girls : avg_weight_girls = 50) :
    (n_boys * avg_weight_boys + n_girls * avg_weight_girls) / (n_boys + n_girls) = 56.25 := 
by
  sorry

end average_weight_of_class_l264_264203


namespace min_value_expression_l264_264671

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) (hxy : x + y = 6) : 
  ( (x - 1)^2 / (y - 2) + ( (y - 1)^2 / (x - 2) ) ) >= 8 :=
by 
  sorry

end min_value_expression_l264_264671


namespace arithmetic_mean_geom_mean_ratio_l264_264571

theorem arithmetic_mean_geom_mean_ratio {a b : ℝ} (h1 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h2 : a > b) (h3 : b > 0) : 
  (∃ k : ℤ, k = 34 ∧ abs ((a / b) - 34) ≤ 0.5) :=
sorry

end arithmetic_mean_geom_mean_ratio_l264_264571


namespace sum_divisible_by_10_l264_264956

-- Define the problem statement
theorem sum_divisible_by_10 {n : ℕ} : (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 10 = 0 ↔ ∃ t : ℕ, n = 5 * t + 1 :=
by sorry

end sum_divisible_by_10_l264_264956


namespace final_price_on_monday_l264_264041

-- Definitions based on the conditions
def saturday_price : ℝ := 50
def sunday_increase : ℝ := 1.2
def monday_discount : ℝ := 0.2

-- The statement to prove
theorem final_price_on_monday : 
  let sunday_price := saturday_price * sunday_increase
  let monday_price := sunday_price * (1 - monday_discount)
  monday_price = 48 :=
by
  sorry

end final_price_on_monday_l264_264041


namespace a5_value_l264_264959

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Assume the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom sum_S6 : S 6 = 12
axiom term_a2 : a 2 = 5
axiom sum_formula (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Prove a5 is -1
theorem a5_value (h_arith : arithmetic_sequence a)
  (h_S6 : S 6 = 12) (h_a2 : a 2 = 5) (h_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 5 = -1 :=
sorry

end a5_value_l264_264959


namespace total_airflow_in_one_week_l264_264640

-- Define the conditions
def airflow_rate : ℕ := 10 -- liters per second
def working_time_per_day : ℕ := 10 -- minutes per day
def days_per_week : ℕ := 7

-- Define the conversion factors
def minutes_to_seconds : ℕ := 60

-- Define the total working time in seconds
def total_working_time_per_week : ℕ := working_time_per_day * days_per_week * minutes_to_seconds

-- Define the expected total airflow in one week
def expected_total_airflow : ℕ := airflow_rate * total_working_time_per_week

-- Prove that the expected total airflow is 42000 liters
theorem total_airflow_in_one_week : expected_total_airflow = 42000 := 
by
  -- assertion is correct given the conditions above 
  -- skip the proof
  sorry

end total_airflow_in_one_week_l264_264640


namespace find_max_value_l264_264398

theorem find_max_value (f : ℝ → ℝ) (h₀ : f 0 = -5) (h₁ : ∀ x, deriv f x = 4 * x^3 - 4 * x) :
  ∃ x, f x = -5 ∧ (∀ y, f y ≤ f x) ∧ x = 0 :=
sorry

end find_max_value_l264_264398


namespace problem_l264_264847

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (M m : ℕ)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 1 ≥ 1
axiom h3 : a 2 ≤ 5
axiom h4 : a 5 ≥ 8

-- Sum function for arithmetic sequence
axiom h5 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

-- Definition of M and m based on S_15
axiom hM : M = max (S 15)
axiom hm : m = min (S 15)

theorem problem (h : S 15 = M + m) : M + m = 600 :=
  sorry

end problem_l264_264847


namespace number_of_tiles_in_each_row_l264_264291

-- Define the given conditions
def area_of_room : ℝ := 256
def tile_size_in_inches : ℝ := 8
def inches_per_foot : ℝ := 12

-- Length of the room in feet derived from the given area
def side_length_in_feet := Real.sqrt area_of_room

-- Convert side length from feet to inches
def side_length_in_inches := side_length_in_feet * inches_per_foot

-- The question: Prove that the number of tiles in each row is 24
theorem number_of_tiles_in_each_row :
  side_length_in_inches / tile_size_in_inches = 24 :=
sorry

end number_of_tiles_in_each_row_l264_264291


namespace factor_expression_l264_264523

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l264_264523


namespace _l264_264828
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l264_264828


namespace area_of_transformed_region_l264_264267

theorem area_of_transformed_region : 
  let T : ℝ := 15
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![6, -2]]
  (abs (Matrix.det A) * T = 450) := 
  sorry

end area_of_transformed_region_l264_264267


namespace trigonometric_identity_l264_264518

theorem trigonometric_identity :
  (3 / (Real.sin (20 * Real.pi / 180))^2) - 
  (1 / (Real.cos (20 * Real.pi / 180))^2) + 
  64 * (Real.sin (20 * Real.pi / 180))^2 = 32 :=
by sorry

end trigonometric_identity_l264_264518


namespace parabola_min_y1_y2_squared_l264_264977

theorem parabola_min_y1_y2_squared (x1 x2 y1 y2 : ℝ) :
  (y1^2 = 4 * x1) ∧
  (y2^2 = 4 * x2) ∧
  (x1 * x2 = 16) →
  (y1^2 + y2^2 ≥ 32) :=
by
  intro h
  sorry

end parabola_min_y1_y2_squared_l264_264977


namespace find_natural_pairs_l264_264805

theorem find_natural_pairs (a b : ℕ) :
  (∃ A, A * A = a ^ 2 + 3 * b) ∧ (∃ B, B * B = b ^ 2 + 3 * a) ↔ 
  (a = 1 ∧ b = 1) ∨ (a = 11 ∧ b = 11) ∨ (a = 16 ∧ b = 11) :=
by
  sorry

end find_natural_pairs_l264_264805


namespace time_to_fill_by_B_l264_264900

/-- 
Assume a pool with two taps, A and B, fills in 30 minutes when both are open.
When both are open for 10 minutes, and then only B is open for another 40 minutes, the pool fills up.
Prove that if only tap B is opened, it would take 60 minutes to fill the pool.
-/
theorem time_to_fill_by_B
  (r_A r_B : ℝ)
  (H1 : (r_A + r_B) * 30 = 1)
  (H2 : ((r_A + r_B) * 10 + r_B * 40) = 1) :
  1 / r_B = 60 :=
by
  sorry

end time_to_fill_by_B_l264_264900


namespace average_marks_l264_264928

/--
Given:
1. The average marks in physics (P) and mathematics (M) is 90.
2. The average marks in physics (P) and chemistry (C) is 70.
3. The student scored 110 marks in physics (P).

Prove that the average marks the student scored in the 3 subjects (P, C, M) is 70.
-/
theorem average_marks (P C M : ℝ) 
  (h1 : (P + M) / 2 = 90)
  (h2 : (P + C) / 2 = 70)
  (h3 : P = 110) : 
  (P + C + M) / 3 = 70 :=
sorry

end average_marks_l264_264928


namespace prime_exponent_condition_l264_264775

theorem prime_exponent_condition (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n)
  (h : 2^p + 3^p = a^n) : n = 1 :=
sorry

end prime_exponent_condition_l264_264775


namespace mabel_tomatoes_l264_264433

theorem mabel_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 := by
  sorry

end mabel_tomatoes_l264_264433


namespace math_problem_l264_264993

theorem math_problem
  (x : ℕ) (y : ℕ)
  (h1 : x = (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id)
  (h2 : y = ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card)
  (h3 : x + y = 611) :
  (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id = 605 ∧
  ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card = 6 := 
by
  sorry

end math_problem_l264_264993


namespace range_of_m_l264_264718

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, mx^2 - 4*x + 1 = 0 ∧ ∀ y : ℝ, mx^2 - 4*x + 1 = 0 → y = x) → m ≤ 4 :=
sorry

end range_of_m_l264_264718


namespace positive_diff_solutions_l264_264012

theorem positive_diff_solutions (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 14) (h2 : 2 * x2 - 3 = -14) : 
  x1 - x2 = 14 := 
by
  sorry

end positive_diff_solutions_l264_264012


namespace amount_charged_for_kids_l264_264729

theorem amount_charged_for_kids (K A: ℝ) (H1: A = 2 * K) (H2: 8 * K + 10 * A = 84) : K = 3 :=
by
  sorry

end amount_charged_for_kids_l264_264729


namespace solve_fractional_equation_l264_264145

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 1 ∧ x ≠ 1) : 3 / (x - 1) = 5 + 3 * x / (1 - x) ↔ x = 4 :=
by 
  split
  -- Forward direction (Given the equation, show x = 4)
  intro h
  -- Multiply out and simplify as shown in the solution steps
  -- Sorry statement skips the detailed proof
  sorry
  -- Converse direction (Substitute x = 4 and verify it satisfies the equation)
  intro hx
  -- Substitution and verification steps
  -- Sorry statement skips the detailed proof
  sorry

end solve_fractional_equation_l264_264145


namespace complete_contingency_table_chi_sq_test_result_expected_value_X_l264_264289

noncomputable def probability_set := {x : ℚ // x ≥ 0 ∧ x ≤ 1}

variable (P : probability_set → probability_set)

-- Conditions from the problem
def P_A_given_not_B : probability_set := ⟨2 / 5, by norm_num⟩
def P_B_given_not_A : probability_set := ⟨5 / 8, by norm_num⟩
def P_B : probability_set := ⟨3 / 4, by norm_num⟩

-- Definitions related to counts and probabilities
def total_students : ℕ := 200
def male_students := P_A_given_not_B.val * total_students
def female_students := total_students - male_students
def score_exceeds_85 := P_B.val * total_students
def score_not_exceeds_85 := total_students - score_exceeds_85

-- Expected counts based on given probabilities
def male_score_not_exceeds_85 := P_A_given_not_B.val * score_not_exceeds_85
def female_score_not_exceeds_85 := score_not_exceeds_85 - male_score_not_exceeds_85
def male_score_exceeds_85 := male_students - male_score_not_exceeds_85
def female_score_exceeds_85 := female_students - female_score_not_exceeds_85

-- Chi-squared test independence 
def chi_squared := (total_students * (male_score_not_exceeds_85 * female_score_exceeds_85 - female_score_not_exceeds_85 * male_score_exceeds_85) ^ 2) / 
                    (male_students * female_students * score_not_exceeds_85 * score_exceeds_85)
def is_related : Prop := chi_squared > 10.828

-- Expected distributions and expectation of X
def P_X_0 := (1 / 4) ^ 2 * (1 / 3) ^ 2
def P_X_1 := 2 * (3 / 4) * (1 / 4) * (1 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (1 / 4) ^ 2
def P_X_2 := (3 / 4) ^ 2 * (1 / 3) ^ 2 + (1 / 4) ^ 2 * (2 / 3) ^ 2 + 2 * (2 / 3) * (1 / 3) * (3 / 4) * (1 / 4)
def P_X_3 := (3 / 4) ^ 2 * 2 * (2 / 3) * (1 / 3) + 2 * (3 / 4) * (1 / 4) * (2 / 3) ^ 2
def P_X_4 := (3 / 4) ^ 2 * (2 / 3) ^ 2
def expectation_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3 + 4 * P_X_4

-- Lean theorem statements for answers using the above definitions
theorem complete_contingency_table :
  male_score_not_exceeds_85 + female_score_not_exceeds_85 = score_not_exceeds_85 ∧
  male_score_exceeds_85 + female_score_exceeds_85 = score_exceeds_85 ∧
  male_students + female_students = total_students := sorry

theorem chi_sq_test_result :
  is_related = true := sorry

theorem expected_value_X :
  expectation_X = 17 / 6 := sorry

end complete_contingency_table_chi_sq_test_result_expected_value_X_l264_264289


namespace reflection_matrix_over_vector_l264_264827

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l264_264827


namespace max_area_difference_160_perimeter_rectangles_l264_264316

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end max_area_difference_160_perimeter_rectangles_l264_264316


namespace inheritance_amount_l264_264435

theorem inheritance_amount (x : ℝ) (total_taxes_paid : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
  (federal_tax_paid : ℝ) (state_tax_base : ℝ) (state_tax_paid : ℝ) 
  (federal_tax_eq : federal_tax_paid = federal_tax_rate * x)
  (state_tax_base_eq : state_tax_base = x - federal_tax_paid)
  (state_tax_eq : state_tax_paid = state_tax_rate * state_tax_base)
  (total_taxes_eq : total_taxes_paid = federal_tax_paid + state_tax_paid) 
  (total_taxes_val : total_taxes_paid = 18000)
  (federal_tax_rate_val : federal_tax_rate = 0.25)
  (state_tax_rate_val : state_tax_rate = 0.15)
  : x = 50000 :=
sorry

end inheritance_amount_l264_264435


namespace combined_selling_price_correct_l264_264353

def cost_A : ℕ := 500
def cost_B : ℕ := 800
def cost_C : ℕ := 1200
def profit_A : ℕ := 25
def profit_B : ℕ := 30
def profit_C : ℕ := 20

def selling_price (cost profit_percentage : ℕ) : ℕ :=
  cost + (profit_percentage * cost / 100)

def combined_selling_price : ℕ :=
  selling_price cost_A profit_A + selling_price cost_B profit_B + selling_price cost_C profit_C

theorem combined_selling_price_correct : combined_selling_price = 3105 := by
  sorry

end combined_selling_price_correct_l264_264353


namespace maximum_value_of_a_l264_264736

theorem maximum_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) : a ≤ 2924 := 
sorry

end maximum_value_of_a_l264_264736


namespace real_root_exists_l264_264752

noncomputable def polynomial := 2 * X^5 + X^4 - 20 * X^3 - 10 * X^2 + 2 * X + 1

theorem real_root_exists : Polynomial.eval (Real.sqrt 3 + Real.sqrt 2) polynomial = 0 := 
by
  sorry

end real_root_exists_l264_264752


namespace number_of_special_permutations_l264_264419

theorem number_of_special_permutations : 
  (Finset.card {p : Finset (Fin 12) // 
     ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} a_{12} : Fin 12),
     Set.univ = {a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_{10}, a_{11}, a_{12}} ∧
     a_1 > a_2 ∧ a_2 > a_3 ∧ a_3 > a_4 ∧ a_4 > a_5 ∧ a_5 > a_6 ∧
     a_6 < a_7 ∧ a_7 < a_8 ∧ a_8 < a_9 ∧ a_9 < a_{10} ∧ a_{10} < a_{11} ∧ a_{11} < a_{12}} = 462 :=
begin
  sorry
end

end number_of_special_permutations_l264_264419


namespace probability_of_prime_sum_l264_264472

/-- 
  We will define a function that returns the probability that the sum of the results 
  of two six-sided dice is a prime number.
-/

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def dice_sums_prime_probability : ℚ :=
  let outcomes : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6)
  let prime_sums : Finset ℕ := outcomes.image (λ p, p.1 + p.2 + 2).filter is_prime
  (prime_sums.card : ℚ) / (outcomes.card : ℚ)

theorem probability_of_prime_sum :
  dice_sums_prime_probability = 5 / 12 :=
by 
  sorry

end probability_of_prime_sum_l264_264472


namespace sqrt_sum_eq_eight_l264_264467

theorem sqrt_sum_eq_eight :
  Real.sqrt (24 - 8 * Real.sqrt 3) + Real.sqrt (24 + 8 * Real.sqrt 3) = 8 := by
  sorry

end sqrt_sum_eq_eight_l264_264467


namespace expected_total_rainfall_10_days_l264_264678

theorem expected_total_rainfall_10_days :
  let P_sun := 0.5
  let P_rain3 := 0.3
  let P_rain6 := 0.2
  let daily_rain := (P_sun * 0) + (P_rain3 * 3) + (P_rain6 * 6)
  daily_rain * 10 = 21 :=
by
  sorry

end expected_total_rainfall_10_days_l264_264678


namespace ratio_of_amounts_l264_264717

theorem ratio_of_amounts (B J P : ℝ) (hB : B = 60) (hP : P = (1 / 3) * B) (hJ : J = B - 20) : J / P = 2 :=
by
  have hP_val : P = 20 := by sorry
  have hJ_val : J = 40 := by sorry
  have ratio : J / P = 40 / 20 := by sorry
  show J / P = 2
  sorry

end ratio_of_amounts_l264_264717


namespace entree_cost_l264_264711

/-- 
Prove that if the total cost is 23 and the entree costs 5 more than the dessert, 
then the cost of the entree is 14.
-/
theorem entree_cost (D : ℝ) (H1 : D + (D + 5) = 23) : D + 5 = 14 :=
by
  -- note: no proof required as per instructions
  sorry

end entree_cost_l264_264711


namespace monotonically_increasing_intervals_inequality_solution_set_l264_264391

-- Given conditions for f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a*x^3 + b*x^2 + c*x + d

-- Ⅰ) Prove the intervals of monotonic increase
theorem monotonically_increasing_intervals (a c : ℝ) (x : ℝ) (h_f : ∀ x, f a 0 c 0 x = a*x^3 + c*x)
  (h_a : a = 1) (h_c : c = -3) :
  (∀ x < -1, f a 0 c 0 x < 0) ∧ (∀ x > 1, f a 0 c 0 x > 0) := 
sorry

-- Ⅱ) Prove the solution sets for the inequality given m
theorem inequality_solution_set (m x : ℝ) :
  (m = 0 → x > 0) ∧
  (m > 0 → (x > 4*m ∨ 0 < x ∧ x < m)) ∧
  (m < 0 → (x > 0 ∨ 4*m < x ∧ x < m)) :=
sorry

end monotonically_increasing_intervals_inequality_solution_set_l264_264391


namespace probability_of_choosing_perfect_square_is_0_08_l264_264209

-- Definitions for the conditions
def n : ℕ := 100
def p : ℚ := 1 / 200
def probability (m : ℕ) : ℚ := if m ≤ 50 then p else 3 * p
def perfect_squares_before_50 : Finset ℕ := {1, 4, 9, 16, 25, 36, 49}
def perfect_squares_between_51_and_100 : Finset ℕ := {64, 81, 100}
def total_perfect_squares : Finset ℕ := perfect_squares_before_50 ∪ perfect_squares_between_51_and_100

-- Statement to prove that the probability of selecting a perfect square is 0.08
theorem probability_of_choosing_perfect_square_is_0_08 :
  (perfect_squares_before_50.card * p + perfect_squares_between_51_and_100.card * 3 * p) = 0.08 := 
by
  -- Adding sorry to skip the proof
  sorry

end probability_of_choosing_perfect_square_is_0_08_l264_264209


namespace carpenter_wood_split_l264_264636

theorem carpenter_wood_split :
  let original_length : ℚ := 35 / 8
  let first_cut : ℚ := 5 / 3
  let second_cut : ℚ := 9 / 4
  let remaining_length := original_length - first_cut - second_cut
  let part_length := remaining_length / 3
  part_length = 11 / 72 :=
sorry

end carpenter_wood_split_l264_264636


namespace correct_option_is_B_l264_264195

def satisfy_triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem correct_option_is_B :
  satisfy_triangle_inequality 3 4 5 ∧
  ¬ satisfy_triangle_inequality 1 1 2 ∧
  ¬ satisfy_triangle_inequality 1 4 6 ∧
  ¬ satisfy_triangle_inequality 2 3 7 :=
by
  sorry

end correct_option_is_B_l264_264195


namespace sum_of_consecutive_integers_is_33_l264_264180

theorem sum_of_consecutive_integers_is_33 :
  ∃ (x : ℕ), x * (x + 1) = 272 ∧ x + (x + 1) = 33 :=
by
  sorry

end sum_of_consecutive_integers_is_33_l264_264180


namespace tencent_technological_innovation_basis_tencent_innovative_development_analysis_l264_264036

-- Define the dialectical materialist basis conditions
variable (dialectical_negation essence_innovation development_perspective unity_of_opposites : Prop)

-- Define Tencent's emphasis on technological innovation
variable (tencent_innovation : Prop)

-- Define the relationship between Tencent's development and materialist view of development
variable (unity_of_things_developmental progressiveness_tortuosity quantitative_qualitative_changes : Prop)
variable (tencent_development : Prop)

-- Prove that Tencent's emphasis on technological innovation aligns with dialectical materialism
theorem tencent_technological_innovation_basis :
  dialectical_negation ∧ essence_innovation ∧ development_perspective ∧ unity_of_opposites → tencent_innovation :=
by sorry

-- Prove that Tencent's innovative development aligns with dialectical materialist view of development
theorem tencent_innovative_development_analysis :
  unity_of_things_developmental ∧ progressiveness_tortuosity ∧ quantitative_qualitative_changes → tencent_development :=
by sorry

end tencent_technological_innovation_basis_tencent_innovative_development_analysis_l264_264036


namespace largest_integer_l264_264385

theorem largest_integer (a b c d : ℤ) 
  (h1 : a + b + c = 210) 
  (h2 : a + b + d = 230) 
  (h3 : a + c + d = 245) 
  (h4 : b + c + d = 260) : 
  d = 105 :=
by 
  sorry

end largest_integer_l264_264385


namespace profit_percentage_calculation_l264_264094

noncomputable def profit_percentage (SP CP : ℝ) : ℝ := ((SP - CP) / CP) * 100

theorem profit_percentage_calculation (SP : ℝ) (h : CP = 0.92 * SP) : |profit_percentage SP (0.92 * SP) - 8.70| < 0.01 :=
by
  sorry

end profit_percentage_calculation_l264_264094


namespace nathan_tomato_plants_l264_264600

theorem nathan_tomato_plants (T: ℕ) : 
  5 * 14 + T * 16 = 186 * 7 / 6 + 9 * 10 :=
  sorry

end nathan_tomato_plants_l264_264600


namespace f_5_eq_2_l264_264049

def f : ℕ → ℤ :=
sorry

axiom f_initial_condition : f 1 = 2

axiom f_functional_eq (a b : ℕ) : f (a + b) = 2 * f a + 2 * f b - 3 * f (a * b)

theorem f_5_eq_2 : f 5 = 2 :=
sorry

end f_5_eq_2_l264_264049


namespace daps_to_dips_l264_264855

theorem daps_to_dips : 
  (∀ a b c d : ℝ, (5 * a = 4 * b) → (3 * b = 8 * c) → (c = 48 * d) → (a = 22.5 * d)) := 
by
  intros a b c d h1 h2 h3
  sorry

end daps_to_dips_l264_264855


namespace length_major_axis_eq_six_l264_264613

-- Define the given equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 9) = 1

-- The theorem stating the length of the major axis
theorem length_major_axis_eq_six (x y : ℝ) (h : ellipse_equation x y) : 
  2 * (Real.sqrt 9) = 6 :=
by
  sorry

end length_major_axis_eq_six_l264_264613


namespace sum_of_three_squares_l264_264501

theorem sum_of_three_squares (s t : ℤ) (h1 : 3 * s + 2 * t = 27)
                             (h2 : 2 * s + 3 * t = 23) (h3 : s + 2 * t = 13) :
  3 * s = 21 :=
sorry

end sum_of_three_squares_l264_264501


namespace max_pies_without_ingredients_l264_264048

theorem max_pies_without_ingredients (total_pies half_chocolate two_thirds_marshmallows three_fifths_cayenne one_eighth_peanuts : ℕ) 
  (h1 : total_pies = 48) 
  (h2 : half_chocolate = total_pies / 2)
  (h3 : two_thirds_marshmallows = 2 * total_pies / 3) 
  (h4 : three_fifths_cayenne = 3 * total_pies / 5)
  (h5 : one_eighth_peanuts = total_pies / 8) : 
  ∃ pies_without_any_ingredients, pies_without_any_ingredients = 16 :=
  by 
    sorry

end max_pies_without_ingredients_l264_264048


namespace frequency_of_hits_l264_264511

theorem frequency_of_hits (n m : ℕ) (h_n : n = 20) (h_m : m = 15) : (m / n : ℚ) = 0.75 := by
  sorry

end frequency_of_hits_l264_264511


namespace tan_sum_identity_l264_264477

theorem tan_sum_identity : (1 + Real.tan (Real.pi / 180)) * (1 + Real.tan (44 * Real.pi / 180)) = 2 := 
by sorry

end tan_sum_identity_l264_264477


namespace no_a_for_empty_intersection_a_in_range_for_subset_union_l264_264079

open Set

def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}
def B : Set ℝ := {x | x^2 - 4 * x - 5 > 0}

-- Problem 1: There is no a such that A ∩ B = ∅
theorem no_a_for_empty_intersection : ∀ a : ℝ, A a ∩ B = ∅ → False := by
  sorry

-- Problem 2: If A ∪ B = B, then a ∈ (-∞, -4) ∪ (5, ∞)
theorem a_in_range_for_subset_union (a : ℝ) : A a ∪ B = B → a ∈ Iio (-4) ∪ Ioi 5 := by
  sorry

end no_a_for_empty_intersection_a_in_range_for_subset_union_l264_264079


namespace sunil_interest_l264_264606

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sunil_interest :
  let A := 19828.80
  let r := 0.08
  let n := 1
  let t := 2
  let P := 19828.80 / (1 + 0.08) ^ 2
  P * (1 + r / n) ^ (n * t) = 19828.80 →
  A - P = 2828.80 :=
by
  sorry

end sunil_interest_l264_264606


namespace packs_sold_to_uncle_is_correct_l264_264239

-- Define the conditions and constants
def total_packs_needed := 50
def packs_sold_to_grandmother := 12
def packs_sold_to_neighbor := 5
def packs_left_to_sell := 26

-- Calculate total packs sold so far
def total_packs_sold := total_packs_needed - packs_left_to_sell

-- Calculate total packs sold to grandmother and neighbor
def packs_sold_to_grandmother_and_neighbor := packs_sold_to_grandmother + packs_sold_to_neighbor

-- The pack sold to uncle
def packs_sold_to_uncle := total_packs_sold - packs_sold_to_grandmother_and_neighbor

-- Prove the packs sold to uncle
theorem packs_sold_to_uncle_is_correct : packs_sold_to_uncle = 7 := by
  -- The proof steps are omitted
  sorry

end packs_sold_to_uncle_is_correct_l264_264239


namespace Xiaoxi_has_largest_final_answer_l264_264589

def Laura_final : ℕ := 8 - 2 * 3 + 3
def Navin_final : ℕ := (8 * 3) - 2 + 3
def Xiaoxi_final : ℕ := (8 - 2 + 3) * 3

theorem Xiaoxi_has_largest_final_answer : 
  Xiaoxi_final > Laura_final ∧ Xiaoxi_final > Navin_final :=
by
  unfold Laura_final Navin_final Xiaoxi_final
  -- Proof steps would go here, but we skip them as per instructions
  sorry

end Xiaoxi_has_largest_final_answer_l264_264589


namespace mabel_tomatoes_l264_264432

theorem mabel_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 := by
  sorry

end mabel_tomatoes_l264_264432


namespace sqrt_12_bounds_l264_264676

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 :=
by
  sorry

end sqrt_12_bounds_l264_264676


namespace angle_in_third_quadrant_l264_264632

-- Define the concept of an angle being in a specific quadrant
def is_in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

-- Prove that -1200° is in the third quadrant
theorem angle_in_third_quadrant :
  is_in_third_quadrant (240) → is_in_third_quadrant (-1200 % 360 + 360 * (if -1200 % 360 ≤ 0 then 1 else 0)) :=
by
  sorry

end angle_in_third_quadrant_l264_264632


namespace faith_overtime_hours_per_day_l264_264533

noncomputable def normal_pay_per_hour : ℝ := 13.50
noncomputable def normal_daily_hours : ℕ := 8
noncomputable def normal_weekly_days : ℕ := 5
noncomputable def total_weekly_earnings : ℝ := 675
noncomputable def overtime_rate_multiplier : ℝ := 1.5

noncomputable def normal_weekly_hours := normal_daily_hours * normal_weekly_days
noncomputable def normal_weekly_earnings := normal_pay_per_hour * normal_weekly_hours
noncomputable def overtime_earnings := total_weekly_earnings - normal_weekly_earnings
noncomputable def overtime_pay_per_hour := normal_pay_per_hour * overtime_rate_multiplier
noncomputable def total_overtime_hours := overtime_earnings / overtime_pay_per_hour
noncomputable def overtime_hours_per_day := total_overtime_hours / normal_weekly_days

theorem faith_overtime_hours_per_day :
  overtime_hours_per_day = 1.33 := 
by 
  sorry

end faith_overtime_hours_per_day_l264_264533


namespace matilda_jellybeans_l264_264595

theorem matilda_jellybeans (steve_jellybeans : ℕ) (h_steve : steve_jellybeans = 84)
  (h_matt : ℕ) (h_matt_calc : h_matt = 10 * steve_jellybeans)
  (h_matilda : ℕ) (h_matilda_calc : h_matilda = h_matt / 2) :
  h_matilda = 420 := by
  sorry

end matilda_jellybeans_l264_264595


namespace arithmetic_mean_solution_l264_264607

/-- Given the arithmetic mean of six expressions is 30, prove the values of x and y are as follows. -/
theorem arithmetic_mean_solution (x y : ℝ) (h : ((2 * x - y) + 20 + (3 * x + y) + 16 + (x + 5) + (y + 8)) / 6 = 30) (hy : y = 10) : 
  x = 18.5 :=
by {
  sorry
}

end arithmetic_mean_solution_l264_264607


namespace second_recipe_cup_count_l264_264443

theorem second_recipe_cup_count (bottle_ounces : ℕ) (ounces_per_cup : ℕ)
  (first_recipe_cups : ℕ) (third_recipe_cups : ℕ) (bottles_needed : ℕ)
  (total_ounces : bottle_ounces = 16)
  (ounce_to_cup : ounces_per_cup = 8)
  (first_recipe : first_recipe_cups = 2)
  (third_recipe : third_recipe_cups = 3)
  (bottles : bottles_needed = 3) :
  (bottles_needed * bottle_ounces) / ounces_per_cup - first_recipe_cups - third_recipe_cups = 1 :=
by
  sorry

end second_recipe_cup_count_l264_264443


namespace asymptote_equation_l264_264974

theorem asymptote_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + Real.sqrt (a^2 + b^2) = 2 * b) →
  (4 * x = 3 * y) ∨ (4 * x = -3 * y) :=
by
  sorry

end asymptote_equation_l264_264974


namespace max_subset_card_l264_264732

theorem max_subset_card (n : ℕ) : 
  ∃ (B : Finset ℕ), B ⊆ Finset.range (n + 1) ∧ 
  (∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → ¬(x + y) % (x - y) = 0) ∧ 
  B.card = Int.ceil (n / 3.0) := sorry

end max_subset_card_l264_264732


namespace cost_of_song_book_l264_264103

theorem cost_of_song_book 
  (flute_cost : ℝ) 
  (stand_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : flute_cost = 142.46) 
  (h2 : stand_cost = 8.89) 
  (h3 : total_cost = 158.35) : 
  total_cost - (flute_cost + stand_cost) = 7.00 := 
by 
  sorry

end cost_of_song_book_l264_264103


namespace evaluate_expression_l264_264812

theorem evaluate_expression :
  200 * (200 - 3) + (200 ^ 2 - 8 ^ 2) = 79336 :=
by
  sorry

end evaluate_expression_l264_264812


namespace least_multiple_of_13_gt_450_l264_264188

theorem least_multiple_of_13_gt_450 : ∃ (n : ℕ), (455 = 13 * n) ∧ 455 > 450 ∧ ∀ m : ℕ, (13 * m > 450) → 455 ≤ 13 * m :=
by
  sorry

end least_multiple_of_13_gt_450_l264_264188


namespace max_xy_l264_264986

theorem max_xy : 
  ∃ x y : ℕ, 5 * x + 3 * y = 100 ∧ x > 0 ∧ y > 0 ∧ x * y = 165 :=
by
  sorry

end max_xy_l264_264986


namespace grain_spilled_correct_l264_264213

variable (original_grain : ℕ) (remaining_grain : ℕ) (grain_spilled : ℕ)

theorem grain_spilled_correct : 
  original_grain = 50870 → remaining_grain = 918 → grain_spilled = original_grain - remaining_grain → grain_spilled = 49952 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end grain_spilled_correct_l264_264213


namespace abs_diff_eq_seven_l264_264987

theorem abs_diff_eq_seven (m n : ℤ) (h1 : |m| = 5) (h2 : |n| = 2) (h3 : m * n < 0) : |m - n| = 7 := 
sorry

end abs_diff_eq_seven_l264_264987


namespace problem_DE_length_l264_264603

theorem problem_DE_length
  (AB AD : ℝ)
  (AB_eq : AB = 7)
  (AD_eq : AD = 10)
  (area_eq : 7 * CE = 140)
  (DC CE DE : ℝ)
  (DC_eq : DC = 7)
  (CE_eq : CE = 20)
  : DE = Real.sqrt 449 :=
by
  sorry

end problem_DE_length_l264_264603


namespace solve_fractional_equation_l264_264155

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l264_264155


namespace solve_fractional_equation_l264_264149

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l264_264149


namespace find_n_l264_264696

theorem find_n (n : ℕ) (hn : (n - 2) * (n - 3) / 12 = 14 / 3) : n = 10 := by
  sorry

end find_n_l264_264696


namespace sum_of_coordinates_of_point_D_l264_264279

theorem sum_of_coordinates_of_point_D
  (N : ℝ × ℝ := (6,2))
  (C : ℝ × ℝ := (10, -2))
  (h : ∃ D : ℝ × ℝ, (N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))) :
  ∃ (D : ℝ × ℝ), D.1 + D.2 = 8 := 
by
  obtain ⟨D, hD⟩ := h
  sorry

end sum_of_coordinates_of_point_D_l264_264279


namespace quentavious_gum_count_l264_264123

def initial_nickels : Nat := 5
def remaining_nickels : Nat := 2
def gum_per_nickel : Nat := 2
def traded_nickels (initial remaining : Nat) : Nat := initial - remaining
def total_gum (trade_n gum_per_n : Nat) : Nat := trade_n * gum_per_n

theorem quentavious_gum_count : total_gum (traded_nickels initial_nickels remaining_nickels) gum_per_nickel = 6 := by
  sorry

end quentavious_gum_count_l264_264123


namespace xy_value_is_one_l264_264690

open Complex

theorem xy_value_is_one (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : x * y = 1 :=
by
  sorry

end xy_value_is_one_l264_264690


namespace no_equilateral_triangle_on_integer_lattice_l264_264370

theorem no_equilateral_triangle_on_integer_lattice :
  ∀ (A B C : ℤ × ℤ), 
  A ≠ B → B ≠ C → C ≠ A →
  (dist A B = dist B C ∧ dist B C = dist C A) → 
  false :=
by sorry

end no_equilateral_triangle_on_integer_lattice_l264_264370


namespace hour_minute_hand_coincide_at_l264_264015

noncomputable def coinciding_time : ℚ :=
  90 / (6 - 0.5)

theorem hour_minute_hand_coincide_at : coinciding_time = 16 + 4 / 11 := 
  sorry

end hour_minute_hand_coincide_at_l264_264015


namespace find_monthly_income_l264_264285

-- Given condition
def deposit : ℝ := 3400
def percentage : ℝ := 0.15

-- Goal: Prove Sheela's monthly income
theorem find_monthly_income : (deposit / percentage) = 22666.67 := by
  -- Skip the proof for now
  sorry

end find_monthly_income_l264_264285


namespace bottles_produced_by_twenty_machines_l264_264914

-- Definitions corresponding to conditions
def bottles_per_machine_per_minute (total_machines : ℕ) (total_bottles : ℕ) : ℕ :=
  total_bottles / total_machines

def bottles_produced (machines : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  machines * rate * time

-- Given conditions
axiom six_machines_rate : ∀ (machines total_bottles : ℕ), machines = 6 → total_bottles = 270 →
  bottles_per_machine_per_minute machines total_bottles = 45

-- Prove the question == answer given conditions
theorem bottles_produced_by_twenty_machines :
  bottles_produced 20 45 4 = 3600 :=
by sorry

end bottles_produced_by_twenty_machines_l264_264914


namespace bus_speed_with_stoppages_l264_264232

theorem bus_speed_with_stoppages :
  ∀ (speed_excluding_stoppages : ℕ) (stop_minutes : ℕ) (total_minutes : ℕ)
  (speed_including_stoppages : ℕ),
  speed_excluding_stoppages = 80 →
  stop_minutes = 15 →
  total_minutes = 60 →
  speed_including_stoppages = (speed_excluding_stoppages * (total_minutes - stop_minutes) / total_minutes) →
  speed_including_stoppages = 60 := by
  sorry

end bus_speed_with_stoppages_l264_264232


namespace minimize_G_l264_264982

noncomputable def F (p q : ℝ) : ℝ :=
  2 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def G (p : ℝ) : ℝ :=
  max (F p 0) (F p 1)

theorem minimize_G :
  ∀ (p : ℝ), 0 ≤ p ∧ p ≤ 0.75 → G p = G 0 → p = 0 :=
by
  intro p hp hG
  -- The proof goes here
  sorry

end minimize_G_l264_264982


namespace least_value_x_l264_264573

theorem least_value_x (x : ℕ) (p q : ℕ) (h_prime_p : Nat.Prime p) (h_prime_q : Nat.Prime q)
  (h_distinct : p ≠ q) (h_diff : q - p = 3) (h_even_prim : x / (11 * p * q) = 2) : x = 770 := by
  sorry

end least_value_x_l264_264573


namespace smallest_n_for_pencil_purchase_l264_264082

theorem smallest_n_for_pencil_purchase (a b c d n : ℕ)
  (h1 : 6 * a + 10 * b = n)
  (h2 : 6 * c + 10 * d = n + 2)
  (h3 : 7 * a + 12 * b > 7 * c + 12 * d)
  (h4 : 3 * (c - a) + 5 * (d - b) = 1)
  (h5 : d - b > 0) :
  n = 100 :=
by
  sorry

end smallest_n_for_pencil_purchase_l264_264082


namespace additional_length_of_track_l264_264029

theorem additional_length_of_track (rise : ℝ) (grade1 grade2 : ℝ) (h_rise : rise = 800) (h_grade1 : grade1 = 0.04) (h_grade2 : grade2 = 0.02) :
  (rise / grade2) - (rise / grade1) = 20000 :=
by
  sorry

end additional_length_of_track_l264_264029


namespace valid_x_values_l264_264793

noncomputable def valid_triangle_sides (x : ℕ) : Prop :=
  8 + 11 > x + 3 ∧ 8 + (x + 3) > 11 ∧ 11 + (x + 3) > 8

theorem valid_x_values :
  {x : ℕ | valid_triangle_sides x ∧ x > 0} = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} :=
by
  sorry

end valid_x_values_l264_264793


namespace avg_xy_36_l264_264304

-- Given condition: The average of the numbers 2, 6, 10, x, and y is 18
def avg_condition (x y : ℝ) : Prop :=
  (2 + 6 + 10 + x + y) / 5 = 18

-- Goal: To prove that the average of x and y is 36
theorem avg_xy_36 (x y : ℝ) (h : avg_condition x y) : (x + y) / 2 = 36 :=
by
  sorry

end avg_xy_36_l264_264304


namespace cats_left_l264_264212

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) 
  (h1 : siamese_cats = 13) (h2 : house_cats = 5) (h3 : cats_sold = 10) : 
  siamese_cats + house_cats - cats_sold = 8 :=
by
  sorry

end cats_left_l264_264212


namespace ball_hit_ground_in_time_l264_264655

theorem ball_hit_ground_in_time :
  ∃ t : ℝ, t ≥ 0 ∧ -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 :=
by sorry

end ball_hit_ground_in_time_l264_264655


namespace minimal_inverse_presses_l264_264738

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem minimal_inverse_presses (x : ℚ) (h : x = 50) : 
  ∃ n, n = 2 ∧ (reciprocal^[n] x = x) :=
by
  sorry

end minimal_inverse_presses_l264_264738


namespace cost_of_song_book_l264_264105

-- Definitions of the constants:
def cost_of_flute : ℝ := 142.46
def cost_of_music_stand : ℝ := 8.89
def total_spent : ℝ := 158.35

-- Definition of the combined cost of the flute and music stand:
def combined_cost := cost_of_flute + cost_of_music_stand

-- The final theorem to prove that the cost of the song book is $7.00:
theorem cost_of_song_book : total_spent - combined_cost = 7.00 := by
  sorry

end cost_of_song_book_l264_264105


namespace evaluate_fraction_eq_10_pow_10_l264_264056

noncomputable def evaluate_fraction (a b c : ℕ) : ℕ :=
  (a ^ 20) / ((a * b) ^ 10)

theorem evaluate_fraction_eq_10_pow_10 :
  evaluate_fraction 30 3 10 = 10 ^ 10 :=
by
  -- We define what is given and manipulate it directly to form a proof outline.
  sorry

end evaluate_fraction_eq_10_pow_10_l264_264056


namespace center_of_circle_is_correct_l264_264344

-- Define the conditions as Lean functions and statements
def is_tangent (x y : ℝ) : Prop :=
  (3 * x + 4 * y = 48) ∨ (3 * x + 4 * y = -12)

def is_on_line (x y : ℝ) : Prop := x = y

-- Define the proof statement
theorem center_of_circle_is_correct (x y : ℝ) (h1 : is_tangent x y) (h2 : is_on_line x y) :
  (x, y) = (18 / 7, 18 / 7) :=
sorry

end center_of_circle_is_correct_l264_264344


namespace equation_of_curve_C_distance_AB_when_R_max_l264_264838

-- Given Circle M
def circle_M (x y : ℝ) := (x + 1)^2 + y^2 = 1

-- Given Circle N
def circle_N (x y : ℝ) := (x - 1)^2 + y^2 = 9

-- Circle P being tangent
def externally_tangent (x y cx cy r : ℝ) := (x - cx)^2 + y^2 = r^2

def internally_tangent (x y cx cy r : ℝ) := (x - cx)^2 + y^2 = r^2

-- Centers and radii
def center_M := (-1 : ℝ, 0 : ℝ)
def center_N := (1 : ℝ, 0 : ℝ)
def radius_M := 1
def radius_N := 3

-- Proving the equations and conditions
theorem equation_of_curve_C : (∀ (x y : ℝ), externally_tangent x y (-1) 0 1 
                            → internally_tangent x y 1 0 3 
                            → (x^2 / 4 + y^2 / 3 = 1)) :=
sorry

theorem distance_AB_when_R_max : (∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1)
                            → (externally_tangent x y 2 0 2 
                            → ∃ A B : ℝ × ℝ,
                            (circle_M A.1 A.2 ∧ circle_M B.1 B.2)
                            ∧ (A ≠ B) 
                            ∧ |A.1 - B.1| = (18 / 7))) :=
sorry

end equation_of_curve_C_distance_AB_when_R_max_l264_264838


namespace fx_properties_l264_264452

-- Definition of the function
def f (x : ℝ) : ℝ := x * |x|

-- Lean statement for the proof problem
theorem fx_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) :=
by
  -- Definition used directly from the conditions
  sorry

end fx_properties_l264_264452


namespace sum_h_k_a_b_l264_264891

def h : ℤ := 3
def k : ℤ := -5
def a : ℤ := 7
def b : ℤ := 4

theorem sum_h_k_a_b : h + k + a + b = 9 := by
  sorry

end sum_h_k_a_b_l264_264891


namespace log_expression_defined_l264_264369

theorem log_expression_defined (x : ℝ) : ∃ c : ℝ, (∀ x > c, (x > 7^8)) :=
by
  existsi 7^8
  intro x hx
  sorry

end log_expression_defined_l264_264369


namespace num_black_cars_l264_264757

theorem num_black_cars (total_cars : ℕ) (one_third_blue : ℚ) (one_half_red : ℚ) 
  (h1 : total_cars = 516) (h2 : one_third_blue = 1/3) (h3 : one_half_red = 1/2) :
  total_cars - (total_cars * one_third_blue + total_cars * one_half_red) = 86 :=
by
  sorry

end num_black_cars_l264_264757


namespace incorrect_weight_estimation_l264_264797

variables (x y : ℝ)

/-- Conditions -/
def regression_equation (x : ℝ) : ℝ := 0.85 * x - 85.71

/-- Incorrect conclusion -/
theorem incorrect_weight_estimation : regression_equation 160 ≠ 50.29 :=
by 
  sorry

end incorrect_weight_estimation_l264_264797


namespace min_value_reciprocal_sum_l264_264560

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (a = 1 ∧ b = 1) → (1 / a + 1 / b = 2) := by
  intros h
  sorry

end min_value_reciprocal_sum_l264_264560


namespace fraction_cubed_equality_l264_264045

-- Constants for the problem
def A : ℝ := 81000
def B : ℝ := 9000

-- Problem statement
theorem fraction_cubed_equality : (A^3) / (B^3) = 729 :=
by
  sorry

end fraction_cubed_equality_l264_264045


namespace arithmetic_sequence_sum_l264_264998

variable {a : ℕ → ℝ} -- The arithmetic sequence {a_n} represented by a function a : ℕ → ℝ

/-- Given that the sum of some terms of an arithmetic sequence is 25, prove the sum of other terms -/
theorem arithmetic_sequence_sum (h : a 3 + a 4 + a 5 + a 6 + a 7 = 25) : a 2 + a 8 = 10 := by
    sorry

end arithmetic_sequence_sum_l264_264998


namespace Maxwell_age_l264_264576

theorem Maxwell_age :
  ∀ (sister_age maxwell_age : ℕ),
    (sister_age = 2) → 
    (maxwell_age + 2 = 2 * (sister_age + 2)) →
    (maxwell_age = 6) :=
by
  intros sister_age maxwell_age h1 h2
  -- Definitions and hypotheses come directly from conditions
  sorry

end Maxwell_age_l264_264576


namespace isosceles_right_triangle_area_l264_264931

theorem isosceles_right_triangle_area
  (a b c : ℝ) 
  (h1 : a = b) 
  (h2 : c = a * Real.sqrt 2) 
  (area : ℝ) 
  (h_area : area = 50)
  (h3 : (1/2) * a * b = area) :
  (a + b + c) / area = 0.4 + 0.2 * Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_area_l264_264931


namespace sextuple_angle_terminal_side_on_xaxis_l264_264233

-- Define angle and conditions
variable (α : ℝ)
variable (isPositiveAngle : 0 < α ∧ α < 360)
variable (sextupleAngleOnXAxis : ∃ k : ℕ, 6 * α = k * 360)

-- Prove the possible values of the angle
theorem sextuple_angle_terminal_side_on_xaxis :
  α = 60 ∨ α = 120 ∨ α = 180 ∨ α = 240 ∨ α = 300 :=
  sorry

end sextuple_angle_terminal_side_on_xaxis_l264_264233


namespace polar_of_C2_rectangular_of_l_max_distance_to_line_l264_264415

open Complex Real Topology

-- Conditions
def C1_param_x (α : ℝ) : ℝ := 1 + 2 * cos α
def C1_param_y (α : ℝ) : ℝ := sqrt 3 * sin α

def C2_param_x (α : ℝ) : ℝ := (1 + 2 * cos α) / 2
def C2_param_y (α : ℝ) : ℝ := (sqrt 3 * sin α) / 3

-- Problem 1: Proving polar and rectangular equations
theorem polar_of_C2 (α : ℝ) : ∃ ρ θ, (ρ^2 - ρ*cos θ - (3/4) = 0) ∧ 
  ∀ x y, x = (1/2) + cos α ∧ y = sin α → false := 
by
  sorry

theorem rectangular_of_l : ∀ ρ θ, 4*ρ*sin(θ + π/3) + 1 = 0 → 
  ∃ x y, 2*sqrt 3* x + 2* y + 1 = 0 := 
by
  sorry

-- Conditions and structure for Problem 2
def C3 (x y : ℝ) : Prop := (y^2 / 3) + x^2 = 1

theorem max_distance_to_line (P : ℝ × ℝ) (hP : C3 P.1 P.2) :
  ∃ d : ℝ, d = (1 + 2 * sqrt 6) / 4 := 
by
  sorry

end polar_of_C2_rectangular_of_l_max_distance_to_line_l264_264415


namespace solve_fractional_equation_l264_264150

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l264_264150


namespace roots_of_polynomial_l264_264543

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end roots_of_polynomial_l264_264543


namespace woman_traveled_by_bus_l264_264497

noncomputable def travel_by_bus : ℕ :=
  let total_distance := 1800
  let distance_by_plane := total_distance / 4
  let distance_by_train := total_distance / 6
  let distance_by_taxi := total_distance / 8
  let remaining_distance := total_distance - (distance_by_plane + distance_by_train + distance_by_taxi)
  let distance_by_rental := remaining_distance * 2 / 3
  distance_by_rental / 2

theorem woman_traveled_by_bus :
  travel_by_bus = 275 :=
by 
  sorry

end woman_traveled_by_bus_l264_264497


namespace molecular_weight_3_moles_l264_264946

theorem molecular_weight_3_moles
  (C_weight : ℝ)
  (H_weight : ℝ)
  (N_weight : ℝ)
  (O_weight : ℝ)
  (Molecular_formula : ℕ → ℕ → ℕ → ℕ → Prop)
  (molecular_weight : ℝ)
  (moles : ℝ) :
  C_weight = 12.01 →
  H_weight = 1.008 →
  N_weight = 14.01 →
  O_weight = 16.00 →
  Molecular_formula 13 9 5 7 →
  molecular_weight = 156.13 + 9.072 + 70.05 + 112.00 →
  moles = 3 →
  3 * molecular_weight = 1041.756 :=
by
  sorry

end molecular_weight_3_moles_l264_264946


namespace exponent_division_l264_264515

theorem exponent_division : (23 ^ 11) / (23 ^ 8) = 12167 := 
by {
  sorry
}

end exponent_division_l264_264515


namespace point_on_x_axis_l264_264253

theorem point_on_x_axis (x : ℝ) (A : ℝ × ℝ) (h : A = (2 - x, x + 3)) (hy : A.snd = 0) : A = (5, 0) :=
by
  sorry

end point_on_x_axis_l264_264253


namespace fermats_little_theorem_analogue_l264_264280

theorem fermats_little_theorem_analogue 
  (a : ℤ) (h1 : Int.gcd a 561 = 1) : a ^ 560 ≡ 1 [ZMOD 561] := 
sorry

end fermats_little_theorem_analogue_l264_264280


namespace prime_p_equals_2_l264_264006

theorem prime_p_equals_2 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs: Nat.Prime s)
  (h_sum : p + q + r = 2 * s) (h_order : 1 < p ∧ p < q ∧ q < r) : p = 2 :=
sorry

end prime_p_equals_2_l264_264006


namespace pavan_total_distance_l264_264121

theorem pavan_total_distance:
  ∀ (D : ℝ),
  (∃ Time1 Time2,
    Time1 = (D / 2) / 30 ∧
    Time2 = (D / 2) / 25 ∧
    Time1 + Time2 = 11)
  → D = 150 :=
by
  intros D h
  sorry

end pavan_total_distance_l264_264121


namespace directrix_of_parabola_l264_264548

theorem directrix_of_parabola (a : ℝ) (h : a = -4) : ∃ k : ℝ, k = 1/16 ∧ ∀ x : ℕ, y = ax ^ 2 → y = k := 
by 
  sorry

end directrix_of_parabola_l264_264548


namespace cost_of_playing_cards_l264_264877

theorem cost_of_playing_cards 
  (allowance_each : ℕ)
  (combined_allowance : ℕ)
  (sticker_box_cost : ℕ)
  (number_of_sticker_packs : ℕ)
  (number_of_packs_Dora_got : ℕ)
  (cost_of_playing_cards : ℕ)
  (h1 : allowance_each = 9)
  (h2 : combined_allowance = allowance_each * 2)
  (h3 : sticker_box_cost = 2)
  (h4 : number_of_packs_Dora_got = 2)
  (h5 : number_of_sticker_packs = number_of_packs_Dora_got * 2)
  (h6 : combined_allowance - number_of_sticker_packs * sticker_box_cost = cost_of_playing_cards) :
  cost_of_playing_cards = 10 :=
sorry

end cost_of_playing_cards_l264_264877


namespace work_rate_D_time_A_B_D_time_D_l264_264769

def workRate (person : String) : ℚ :=
  if person = "A" then 1/12 else
  if person = "B" then 1/6 else
  if person = "A_D" then 1/4 else
  0

theorem work_rate_D : workRate "A_D" - workRate "A" = 1/6 := by
  sorry

theorem time_A_B_D : (1 / (workRate "A" + workRate "B" + (workRate "A_D" - workRate "A"))) = 2.4 := by
  sorry
  
theorem time_D : (1 / (workRate "A_D" - workRate "A")) = 6 := by
  sorry

end work_rate_D_time_A_B_D_time_D_l264_264769


namespace total_airflow_in_one_week_l264_264639

-- Define the conditions
def airflow_rate : ℕ := 10 -- liters per second
def working_time_per_day : ℕ := 10 -- minutes per day
def days_per_week : ℕ := 7

-- Define the conversion factors
def minutes_to_seconds : ℕ := 60

-- Define the total working time in seconds
def total_working_time_per_week : ℕ := working_time_per_day * days_per_week * minutes_to_seconds

-- Define the expected total airflow in one week
def expected_total_airflow : ℕ := airflow_rate * total_working_time_per_week

-- Prove that the expected total airflow is 42000 liters
theorem total_airflow_in_one_week : expected_total_airflow = 42000 := 
by
  -- assertion is correct given the conditions above 
  -- skip the proof
  sorry

end total_airflow_in_one_week_l264_264639


namespace f_is_odd_l264_264299

open Real

noncomputable def f (x : ℝ) (n : ℕ) : ℝ :=
  (1 + sin x)^(2 * n) - (1 - sin x)^(2 * n)

theorem f_is_odd (n : ℕ) (h : n > 0) : ∀ x : ℝ, f (-x) n = -f x n :=
by
  intros x
  -- Proof goes here
  sorry

end f_is_odd_l264_264299


namespace determinant_of_A_l264_264667

-- Define the 2x2 matrix A
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![7, -2], ![-3, 6]]

-- The statement to be proved
theorem determinant_of_A : Matrix.det A = 36 := 
  by sorry

end determinant_of_A_l264_264667


namespace coeff_of_linear_term_l264_264448

def quadratic_eqn (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem coeff_of_linear_term :
  ∀ (x : ℝ), (quadratic_eqn x = 0) → (∃ c_b : ℝ, quadratic_eqn x = x^2 + c_b * x + 3 ∧ c_b = -2) :=
by
  sorry

end coeff_of_linear_term_l264_264448


namespace train_length_is_250_l264_264929

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) (station_length : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600 * time_sec) - station_length

theorem train_length_is_250 :
  train_length 36 45 200 = 250 :=
by
  sorry

end train_length_is_250_l264_264929


namespace necessary_and_sufficient_condition_l264_264069

-- Define the arithmetic sequence
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a_1 + (n - 1) * d

-- Define the sum of the first k terms of an arithmetic sequence
def sum_arithmetic_seq (a_1 : ℤ) (d : ℤ) (k : ℤ) : ℤ :=
  (k * (2 * a_1 + (k - 1) * d)) / 2

-- Prove that d > 0 is a necessary and sufficient condition for S_3n - S_2n > S_2n - S_n
/-- Necessary and sufficient condition for the inequality S_{3n} - S_{2n} > S_{2n} - S_n -/
theorem necessary_and_sufficient_condition {a_1 d n : ℤ} :
  d > 0 ↔ sum_arithmetic_seq a_1 d (3 * n) - sum_arithmetic_seq a_1 d (2 * n) > 
             sum_arithmetic_seq a_1 d (2 * n) - sum_arithmetic_seq a_1 d n :=
by sorry

end necessary_and_sufficient_condition_l264_264069


namespace correct_equation_l264_264166

theorem correct_equation (x : ℝ) (hx : x > 80) : 
  353 / (x - 80) - 353 / x = 5 / 3 :=
sorry

end correct_equation_l264_264166


namespace sqrt_12_bounds_l264_264675

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 :=
by
  sorry

end sqrt_12_bounds_l264_264675


namespace part1_find_a_b_part2_inequality_l264_264973

theorem part1_find_a_b (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = |2 * x + 1| + |x + a|) 
  (h_sol : ∀ x, f x ≤ 3 ↔ b ≤ x ∧ x ≤ 1) : 
  a = -1 ∧ b = -1 :=
sorry

theorem part2_inequality (m n : ℝ) (a : ℝ) (h_m : 0 < m) (h_n : 0 < n) 
  (h_eq : (1 / (2 * m)) + (2 / n) + 2 * a = 0) (h_a : a = -1) : 
  4 * m^2 + n^2 ≥ 4 :=
sorry

end part1_find_a_b_part2_inequality_l264_264973


namespace percentage_increase_in_consumption_l264_264459

theorem percentage_increase_in_consumption 
  (T C : ℝ) 
  (h1 : 0.8 * T * C * (1 + P / 100) = 0.88 * T * C)
  : P = 10 := 
by 
  sorry

end percentage_increase_in_consumption_l264_264459


namespace problem1_problem2_l264_264839

-- Definitions and assumptions
def p (m : ℝ) : Prop := ∀x y : ℝ, (x^2)/(4 - m) + (y^2)/m = 1 → ∃ c : ℝ, c^2 < (4 - m) ∧ c^2 < m
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0
def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ m ≥ 1 := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hp : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end problem1_problem2_l264_264839


namespace difference_of_percentages_l264_264574

variable (x y : ℝ)

theorem difference_of_percentages :
  (0.60 * (50 + x)) - (0.45 * (30 + y)) = 16.5 + 0.60 * x - 0.45 * y := 
sorry

end difference_of_percentages_l264_264574


namespace company_employee_count_l264_264000

theorem company_employee_count (E : ℝ) (H1 : E > 0) (H2 : 0.60 * E = 0.55 * (E + 30)) : E + 30 = 360 :=
by
  -- The proof steps would go here, but that is not required.
  sorry

end company_employee_count_l264_264000


namespace fraction_power_simplification_l264_264042

theorem fraction_power_simplification:
  (81000/9000)^3 = 729 → (81000^3) / (9000^3) = 729 :=
by 
  intro h
  rw [<- h]
  sorry

end fraction_power_simplification_l264_264042


namespace coordinates_of_B_l264_264861

theorem coordinates_of_B (a : ℝ) (h : a - 2 = 0) : (a + 2, a - 1) = (4, 1) :=
by
  sorry

end coordinates_of_B_l264_264861


namespace solve_equation_l264_264140

theorem solve_equation : (x : ℝ) (hx : x ≠ 1) (h : 3 / (x - 1) = 5 + 3 * x / (1 - x)) : x = 4 :=
sorry

end solve_equation_l264_264140


namespace tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l264_264686

open Real

theorem tan_alpha_minus_pi_over_4_eq_neg_3_over_4 (α β : ℝ) 
  (h1 : tan (α + β) = 1 / 2) 
  (h2 : tan β = 1 / 3) : 
  tan (α - π / 4) = -3 / 4 :=
sorry

end tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l264_264686


namespace unique_intersection_of_line_and_parabola_l264_264807

theorem unique_intersection_of_line_and_parabola :
  ∃! k : ℚ, ∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k → k = 25 / 3 :=
by
  sorry

end unique_intersection_of_line_and_parabola_l264_264807


namespace exists_m_such_that_m_poly_is_zero_mod_p_l264_264552

theorem exists_m_such_that_m_poly_is_zero_mod_p (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 7 = 1) :
  ∃ m : ℕ, m > 0 ∧ (m^3 + m^2 - 2*m - 1) % p = 0 := 
sorry

end exists_m_such_that_m_poly_is_zero_mod_p_l264_264552


namespace total_legs_of_camden_dogs_l264_264362

-- Defining the number of dogs Justin has
def justin_dogs : ℕ := 14

-- Defining the number of dogs Rico has
def rico_dogs : ℕ := justin_dogs + 10

-- Defining the number of dogs Camden has
def camden_dogs : ℕ := 3 * rico_dogs / 4

-- Defining the total number of legs Camden's dogs have
def camden_dogs_legs : ℕ := camden_dogs * 4

-- The proof statement
theorem total_legs_of_camden_dogs : camden_dogs_legs = 72 :=
by
  -- skip proof
  sorry

end total_legs_of_camden_dogs_l264_264362


namespace tank_capacity_ratio_l264_264610

-- Definitions from the problem conditions
def tank1_filled : ℝ := 300
def tank2_filled : ℝ := 450
def tank2_percentage_filled : ℝ := 0.45
def additional_needed : ℝ := 1250

-- Theorem statement
theorem tank_capacity_ratio (C1 C2 : ℝ) 
  (h1 : tank1_filled + tank2_filled + additional_needed = C1 + C2)
  (h2 : tank2_filled = tank2_percentage_filled * C2) : 
  C1 / C2 = 2 :=
by
  sorry

end tank_capacity_ratio_l264_264610


namespace dips_to_daps_l264_264858

theorem dips_to_daps : 
  ∀ (daps dops dips : Type) (eq1 : 5 * daps = 4 * dops) (eq2 : 3 * dops = 8 * dips),
  (48 * dips = 22.5 * daps) :=
begin
  intros,
  sorry
end

end dips_to_daps_l264_264858


namespace must_be_divisor_of_p_l264_264423

theorem must_be_divisor_of_p (p q r s : ℕ) (hpq : Nat.gcd p q = 40) (hqr : Nat.gcd q r = 50) 
  (hrs : Nat.gcd r s = 75) (hsp : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) : 17 ∣ p :=
sorry

end must_be_divisor_of_p_l264_264423


namespace mango_selling_price_l264_264207

theorem mango_selling_price
  (CP SP_loss SP_profit : ℝ)
  (h1 : SP_loss = 0.8 * CP)
  (h2 : SP_profit = 1.05 * CP)
  (h3 : SP_profit = 6.5625) :
  SP_loss = 5.00 :=
by
  sorry

end mango_selling_price_l264_264207


namespace math_problem_l264_264661

def calc_expr : Int := 
  54322 * 32123 - 54321 * 32123 + 54322 * 99000 - 54321 * 99001

theorem math_problem :
  calc_expr = 76802 := 
by
  sorry

end math_problem_l264_264661


namespace total_time_to_watch_movie_l264_264504

-- Define the conditions and the question
def uninterrupted_viewing_time : ℕ := 35 + 45 + 20
def rewinding_time : ℕ := 5 + 15
def total_time : ℕ := uninterrupted_viewing_time + rewinding_time

-- Lean statement of the proof problem
theorem total_time_to_watch_movie : total_time = 120 := by
  -- This is where the proof would go
  sorry

end total_time_to_watch_movie_l264_264504


namespace residue_7_pow_1234_l264_264323

theorem residue_7_pow_1234 : (7^1234) % 13 = 4 := by
  sorry

end residue_7_pow_1234_l264_264323


namespace x_equals_eleven_l264_264852

theorem x_equals_eleven (x : ℕ) 
  (h : (1 / 8) * 2^36 = 8^x) : x = 11 :=
sorry

end x_equals_eleven_l264_264852


namespace negation_of_proposition_exists_negation_of_proposition_l264_264990

theorem negation_of_proposition : 
  (∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) ↔ ¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) :=
by
  sorry

theorem exists_negation_of_proposition : 
  (¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0)) ↔ ∃ x : ℝ, 2^x - 2*x - 2 < 0 :=
by
  sorry

end negation_of_proposition_exists_negation_of_proposition_l264_264990


namespace invalid_votes_percentage_is_correct_l264_264098

-- Definitions based on conditions
def total_votes : ℕ := 5500
def other_candidate_votes : ℕ := 1980
def valid_votes_percentage_other : ℚ := 0.45

-- Derived values
def valid_votes : ℚ := other_candidate_votes / valid_votes_percentage_other
def invalid_votes : ℚ := total_votes - valid_votes
def invalid_votes_percentage : ℚ := (invalid_votes / total_votes) * 100

-- Proof statement
theorem invalid_votes_percentage_is_correct :
  invalid_votes_percentage = 20 := sorry

end invalid_votes_percentage_is_correct_l264_264098


namespace harry_less_than_half_selena_l264_264283

-- Definitions of the conditions
def selena_book_pages := 400
def harry_book_pages := 180
def half (n : ℕ) := n / 2

-- The theorem to prove that Harry's book is 20 pages less than half of Selena's book.
theorem harry_less_than_half_selena :
  harry_book_pages = half selena_book_pages - 20 := 
by
  sorry

end harry_less_than_half_selena_l264_264283


namespace greatest_area_difference_l264_264314

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) (h₁ : 2 * l₁ + 2 * w₁ = 160) (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  abs (l₁ * w₁ - l₂ * w₂) = 1521 :=
sorry

end greatest_area_difference_l264_264314


namespace parabola_y_axis_symmetry_l264_264177

theorem parabola_y_axis_symmetry (a b c d : ℝ) (r : ℝ) :
  (2019^2 + 2019 * a + b = 0) ∧ (2019^2 + 2019 * c + d = 0) ∧
  (a = -(2019 + r)) ∧ (c = -(2019 - r)) →
  b = -d :=
by
  sorry

end parabola_y_axis_symmetry_l264_264177


namespace point_in_third_quadrant_l264_264092

theorem point_in_third_quadrant (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : -m^2 < 0 ∧ -n < 0 :=
by
  sorry

end point_in_third_quadrant_l264_264092


namespace complete_square_result_l264_264765

theorem complete_square_result (x : ℝ) :
  (x^2 - 4 * x - 3 = 0) → ((x - 2) ^ 2 = 7) :=
by sorry

end complete_square_result_l264_264765


namespace ratio_of_x_to_y_l264_264027

theorem ratio_of_x_to_y (x y : ℝ) (R : ℝ) (h1 : x = R * y) (h2 : x - y = 0.909090909090909 * x) : R = 11 := by
  sorry

end ratio_of_x_to_y_l264_264027


namespace integer_count_satisfying_inequality_l264_264713

theorem integer_count_satisfying_inequality :
  { n : ℤ | -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 }.to_finset.card = 8 := 
sorry

end integer_count_satisfying_inequality_l264_264713


namespace polynomial_roots_l264_264540

theorem polynomial_roots :
  (∀ x, x^3 - 3*x^2 - x + 3 = 0 ↔ (x = 1 ∨ x = -1 ∨ x = 3)) :=
by
  intro x
  split
  {
    intro h
    have h1 : x = 1 ∨ x = -1 ∨ x = 3
    {
      sorry
    }
    exact h1
  }
  {
    intro h
    cases h
    {
      rw h
      simp
    }
    {
      cases h
      {
        rw h
        simp
      }
      {
        rw h
        simp
      }
    }
  }

end polynomial_roots_l264_264540


namespace roots_of_polynomial_l264_264536

-- Define the polynomial P(x) = x^3 - 3x^2 - x + 3
def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

-- Define the statement to prove the roots of the polynomial
theorem roots_of_polynomial :
  ∀ x : ℝ, (P x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l264_264536


namespace no_real_roots_l264_264307

-- Define the coefficients of the quadratic equation
def a : ℝ := 1
def b : ℝ := 2
def c : ℝ := 4

-- Define the quadratic equation
def quadratic_eqn (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant
def discriminant : ℝ := b^2 - 4 * a * c

-- State the theorem: The quadratic equation has no real roots because the discriminant is negative
theorem no_real_roots : discriminant < 0 := by
  unfold discriminant
  unfold a b c
  sorry

end no_real_roots_l264_264307


namespace prob_two_correct_A_B_C_prob_dist_and_expectation_l264_264165

noncomputable def P_A := 2/3
noncomputable def P_B := 1/3
noncomputable def P_C := 1/3

-- Question 1: Prove the probability of exactly two correct answers is 1/3
theorem prob_two_correct_A_B_C : 
  let p_a := P_A 
  let p_b := P_B 
  let p_c := P_C 
  p_a * p_b * (1 - p_c) + p_a * (1 - p_b) * p_c + (1 - p_a) * p_b * p_c = 1/3 := by
  sorry

-- Question 2: Prove the probability distribution and expectation
theorem prob_dist_and_expectation :
  let p_a := P_A 
  let p_b := P_B 
  let p_c := P_C 
  let P_X := fun x =>  
    match x with
    | 0 => (1 - p_a) * (1 - p_b) * (1 - p_c)
    | 1 => p_a * (1 - p_b) * (1 - p_c)
    | 2 => (1 - p_a) * p_b * (1 - p_c) + (1 - p_a) * (1 - p_b) * p_c
    | 3 => p_a * p_b * (1 - p_c) + p_a * (1 - p_b) * p_c
    | 4 => (1 - p_a) * p_b * p_c
    | 5 => p_a * p_b * p_c
    | _ => 0
  let E_X := 0 * (1 - p_a) * (1 - p_b) * (1 - p_c) + 
            1 * p_a * (1 - p_b) * (1 - p_c) + 
            2 * ((1 - p_a) * p_b * (1 - p_c) + (1 - p_a) * (1 - p_b) * p_c) + 
            3 * (p_a * p_b * (1 - p_c) + p_a * (1 - p_b) * p_c) + 
            4 * (1 - p_a) * p_b * p_c + 
            5 * p_a * p_b * p_c
  (P_X 0 = 4/27) ∧ (P_X 1 = 8/27) ∧ (P_X 2 = 4/27) ∧ (P_X 3 = 8/27) ∧ (P_X 4 = 1/27) ∧ (P_X 5 = 2/27) ∧ (E_X = 2) := by
  sorry

end prob_two_correct_A_B_C_prob_dist_and_expectation_l264_264165


namespace factor_expression_l264_264531

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l264_264531


namespace amaya_total_time_l264_264502

-- Define the times as per the conditions
def first_segment : Nat := 35 + 5
def second_segment : Nat := 45 + 15
def third_segment : Nat := 20

-- Define the total time by summing up all segments
def total_time : Nat := first_segment + second_segment + third_segment

-- The theorem to prove
theorem amaya_total_time : total_time = 120 := by
  -- Let's explicitly state the expected result here
  have h1 : first_segment = 40 := rfl
  have h2 : second_segment = 60 := rfl
  have h3 : third_segment = 20 := rfl
  have h_sum : total_time = 40 + 60 + 20 := by
    rw [h1, h2, h3]
  simp [total_time, h_sum]
  -- Finally, the result is 120
  exact rfl

end amaya_total_time_l264_264502


namespace circle_occupies_62_8_percent_l264_264670

noncomputable def largestCirclePercentage (length : ℝ) (width : ℝ) : ℝ :=
  let radius := width / 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := length * width
  (circle_area / rectangle_area) * 100

theorem circle_occupies_62_8_percent : largestCirclePercentage 5 4 = 62.8 := 
by 
  /- Sorry, skipping the proof -/
  sorry

end circle_occupies_62_8_percent_l264_264670


namespace max_ab_bc_ca_l264_264692

theorem max_ab_bc_ca (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 3) :
  ab + bc + ca ≤ 3 :=
sorry

end max_ab_bc_ca_l264_264692


namespace roots_of_polynomial_l264_264541

def P (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial :
  {x : ℝ | P x = 0} = {1, -1, 3} := 
sorry

end roots_of_polynomial_l264_264541


namespace milk_production_days_l264_264410

theorem milk_production_days (x : ℕ) (h : x > 0) :
  let daily_production_per_cow := (x + 1) / (x * (x + 2))
  let total_daily_production := (x + 4) * daily_production_per_cow
  ((x + 7) / total_daily_production) = (x * (x + 2) * (x + 7)) / ((x + 1) * (x + 4)) := 
by
  sorry

end milk_production_days_l264_264410


namespace solve_fractional_equation_l264_264148

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
    (3 / (x - 1) = 5 + 3 * x / (1 - x)) → x = 4 := 
by
  sorry

end solve_fractional_equation_l264_264148


namespace volumes_relation_l264_264581

-- Definitions and conditions based on the problem
variables {a b c : ℝ} (h_triangle : a > b) (h_triangle2 : b > c) (h_acute : 0 < θ ∧ θ < π)

-- The heights from vertices
variables (AD BE CF : ℝ)

-- Volumes of the tetrahedrons formed after folding
variables (V1 V2 V3 : ℝ)

-- The heights are given:
noncomputable def height_AD (BC : ℝ) (theta : ℝ) := AD
noncomputable def height_BE (CA : ℝ) (theta : ℝ) := BE
noncomputable def height_CF (AB : ℝ) (theta : ℝ) := CF

-- Using these heights and the acute nature of the triangle
noncomputable def volume_V1 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V1
noncomputable def volume_V2 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V2
noncomputable def volume_V3 (BC : ℝ) (CA : ℝ) (AB : ℝ) := V3

-- The theorem stating the relationship between volumes
theorem volumes_relation
  (h_triangle: a > b)
  (h_triangle2: b > c)
  (h_acute: 0 < θ ∧ θ < π)
  (h_volumes: V1 > V2 ∧ V2 > V3):
  V1 > V2 ∧ V2 > V3 :=
sorry

end volumes_relation_l264_264581


namespace cos_alpha_second_quadrant_l264_264397

theorem cos_alpha_second_quadrant (α : ℝ) (h₁ : (π / 2) < α ∧ α < π) (h₂ : Real.sin α = 5 / 13) :
  Real.cos α = -12 / 13 :=
by
  sorry

end cos_alpha_second_quadrant_l264_264397


namespace lights_on_fourth_tier_l264_264997

def number_lights_topmost_tier (total_lights : ℕ) : ℕ :=
  total_lights / 127

def number_lights_tier (tier : ℕ) (lights_topmost : ℕ) : ℕ :=
  2^(tier - 1) * lights_topmost

theorem lights_on_fourth_tier (total_lights : ℕ) (H : total_lights = 381) : number_lights_tier 4 (number_lights_topmost_tier total_lights) = 24 :=
by
  rw [H]
  sorry

end lights_on_fourth_tier_l264_264997


namespace cooking_oil_remaining_l264_264635

theorem cooking_oil_remaining (initial_weight : ℝ) (fraction_used : ℝ) (remaining_weight : ℝ) :
  initial_weight = 5 → fraction_used = 4 / 5 → remaining_weight = 21 / 5 → initial_weight * (1 - fraction_used) ≠ remaining_weight → initial_weight * (1 - fraction_used) = 1 :=
by 
  intros h_initial_weight h_fraction_used h_remaining_weight h_contradiction
  sorry

end cooking_oil_remaining_l264_264635


namespace ratio_black_bears_to_white_bears_l264_264363

theorem ratio_black_bears_to_white_bears
  (B W Br : ℕ)
  (hB : B = 60)
  (hBr : Br = B + 40)
  (h_total : B + W + Br = 190) :
  B / W = 2 :=
by
  sorry

end ratio_black_bears_to_white_bears_l264_264363


namespace chosen_number_l264_264647

theorem chosen_number (x : ℝ) (h : 2 * x - 138 = 102) : x = 120 := by
  sorry

end chosen_number_l264_264647


namespace sum_non_solutions_is_neg21_l264_264872

noncomputable def sum_of_non_solutions (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2) : ℝ :=
  -21

theorem sum_non_solutions_is_neg21 (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) = 2) : 
  ∃! (x1 x2 : ℝ), ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2 → x = x1 ∨ x = x2 ∧ x1 + x2 = -21 :=
sorry

end sum_non_solutions_is_neg21_l264_264872


namespace cost_to_replace_is_800_l264_264108

-- Definitions based on conditions
def trade_in_value (num_movies : ℕ) (trade_in_price : ℕ) : ℕ :=
  num_movies * trade_in_price

def dvd_cost (num_movies : ℕ) (dvd_price : ℕ) : ℕ :=
  num_movies * dvd_price

def replacement_cost (num_movies : ℕ) (trade_in_price : ℕ) (dvd_price : ℕ) : ℕ :=
  dvd_cost num_movies dvd_price - trade_in_value num_movies trade_in_price

-- Problem statement: it costs John $800 to replace his movies
theorem cost_to_replace_is_800 (num_movies trade_in_price dvd_price : ℕ)
  (h1 : num_movies = 100) (h2 : trade_in_price = 2) (h3 : dvd_price = 10) :
  replacement_cost num_movies trade_in_price dvd_price = 800 :=
by
  -- Proof would go here
  sorry

end cost_to_replace_is_800_l264_264108


namespace triangle_area_is_180_l264_264457

theorem triangle_area_is_180 {a b c : ℕ} (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) 
  (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 : ℚ) * a * b = 180 :=
by
  sorry

end triangle_area_is_180_l264_264457


namespace tax_calculation_correct_l264_264650

def calculate_tax (s e b1 b2 r1 r2 r3 : ℝ) : ℝ :=
  let taxable_income := s - e
  if taxable_income ≤ b1 then
    taxable_income * r1
  else if taxable_income ≤ b2 then
    (b1 * r1) + (taxable_income - b1) * r2
  else
    (b1 * r1) + (b2 - b1) * r2 + (taxable_income - b2) * r3

theorem tax_calculation_correct :
  calculate_tax 20000 5000 3000 12000 0.03 0.10 0.20 = 1590 :=
by repeat { sorry }

end tax_calculation_correct_l264_264650
