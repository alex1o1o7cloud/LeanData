import Mathlib

namespace NUMINAMATH_GPT_simplify_expression_l243_24344

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  ((2 * x^2)^3 - 6 * x^3 * (x^3 - 2 * x^2)) / (2 * x^4) = x^2 + 6 * x :=
by 
  -- We provide 'sorry' hack to skip the proof
  -- Replace this with the actual proof to ensure correctness.
  sorry

end NUMINAMATH_GPT_simplify_expression_l243_24344


namespace NUMINAMATH_GPT_factorial_div_result_l243_24314

theorem factorial_div_result : Nat.factorial 13 / Nat.factorial 11 = 156 :=
sorry

end NUMINAMATH_GPT_factorial_div_result_l243_24314


namespace NUMINAMATH_GPT_no_roots_in_interval_l243_24350

theorem no_roots_in_interval (a : ℝ) (x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h_eq: a ^ x + a ^ (-x) = 2 * a) : x < -1 ∨ x > 1 :=
sorry

end NUMINAMATH_GPT_no_roots_in_interval_l243_24350


namespace NUMINAMATH_GPT_total_stops_is_seven_l243_24304

-- Definitions of conditions
def initial_stops : ℕ := 3
def additional_stops : ℕ := 4

-- Statement to be proved
theorem total_stops_is_seven : initial_stops + additional_stops = 7 :=
by {
  -- this is a placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_total_stops_is_seven_l243_24304


namespace NUMINAMATH_GPT_parallel_lines_implies_slope_l243_24386

theorem parallel_lines_implies_slope (a : ℝ) :
  (∀ (x y: ℝ), ax + 2 * y = 0) ∧ (∀ (x y: ℝ), x + y = 1) → (a = 2) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_implies_slope_l243_24386


namespace NUMINAMATH_GPT_articles_produced_l243_24323

theorem articles_produced (x y : ℕ) :
  (x * x * x * (1 / (x^2 : ℝ))) = x → (y * y * y * (1 / (x^2 : ℝ))) = (y^3 / x^2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_articles_produced_l243_24323


namespace NUMINAMATH_GPT_circle_radius_twice_value_l243_24315

theorem circle_radius_twice_value (r_x r_y v : ℝ) (h1 : π * r_x^2 = π * r_y^2)
  (h2 : 2 * π * r_x = 12 * π) (h3 : r_y = 2 * v) : v = 3 := by
  sorry

end NUMINAMATH_GPT_circle_radius_twice_value_l243_24315


namespace NUMINAMATH_GPT_income_is_10000_l243_24391

theorem income_is_10000 (x : ℝ) (h : 10 * x = 8 * x + 2000) : 10 * x = 10000 := by
  have h1 : 2 * x = 2000 := by
    linarith
  have h2 : x = 1000 := by
    linarith
  linarith

end NUMINAMATH_GPT_income_is_10000_l243_24391


namespace NUMINAMATH_GPT_plane_equation_through_points_perpendicular_l243_24341

theorem plane_equation_through_points_perpendicular {M N : ℝ × ℝ × ℝ} (hM : M = (2, -1, 4)) (hN : N = (3, 2, -1)) :
  ∃ A B C d : ℝ, (∀ x y z : ℝ, A * x + B * y + C * z + d = 0 ↔ (x, y, z) = M ∨ (x, y, z) = N ∧ A + B + C = 0) ∧
  (4, -3, -1, -7) = (A, B, C, d) := 
sorry

end NUMINAMATH_GPT_plane_equation_through_points_perpendicular_l243_24341


namespace NUMINAMATH_GPT_inequality_ab_sum_eq_five_l243_24316

noncomputable def inequality_solution (a b : ℝ) : Prop :=
  (∀ x : ℝ, (x < 1) → (x < a) → (x > b) ∨ (x > 4) → (x < a) → (x > b))

theorem inequality_ab_sum_eq_five (a b : ℝ) 
  (h : inequality_solution a b) : a + b = 5 :=
sorry

end NUMINAMATH_GPT_inequality_ab_sum_eq_five_l243_24316


namespace NUMINAMATH_GPT_vents_per_zone_l243_24398

theorem vents_per_zone (total_cost : ℝ) (number_of_zones : ℝ) (cost_per_vent : ℝ) (h_total_cost : total_cost = 20000) (h_zones : number_of_zones = 2) (h_cost_per_vent : cost_per_vent = 2000) : 
  (total_cost / cost_per_vent) / number_of_zones = 5 :=
by 
  sorry

end NUMINAMATH_GPT_vents_per_zone_l243_24398


namespace NUMINAMATH_GPT_determine_good_numbers_l243_24385

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), (∀ k : Fin n, ∃ m : ℕ, k.1 + (a k).1 + 1 = m * m)

theorem determine_good_numbers :
  is_good_number 13 ∧ is_good_number 15 ∧ is_good_number 17 ∧ is_good_number 19 ∧ ¬is_good_number 11 :=
by
  sorry

end NUMINAMATH_GPT_determine_good_numbers_l243_24385


namespace NUMINAMATH_GPT_find_breadth_l243_24307

theorem find_breadth (p l : ℕ) (h_p : p = 600) (h_l : l = 100) (h_perimeter : p = 2 * (l + b)) : b = 200 :=
by
  sorry

end NUMINAMATH_GPT_find_breadth_l243_24307


namespace NUMINAMATH_GPT_candle_duration_1_hour_per_night_l243_24340

-- Definitions based on the conditions
def burn_rate_2_hours (candles: ℕ) (nights: ℕ) : ℕ := nights / candles -- How long each candle lasts when burned for 2 hours per night

-- Given conditions provided
def nights_24 : ℕ := 24
def candles_6 : ℕ := 6

-- The duration a candle lasts when burned for 2 hours every night
def candle_duration_2_hours_per_night : ℕ := burn_rate_2_hours candles_6 nights_24 -- => 4 (not evaluated here)

-- Theorem to prove the duration a candle lasts when burned for 1 hour every night
theorem candle_duration_1_hour_per_night : candle_duration_2_hours_per_night * 2 = 8 :=
by
  sorry -- The proof is omitted, only the statement is required

-- Note: candle_duration_2_hours_per_night = 4 by the given conditions 
-- This leads to 4 * 2 = 8, which matches the required number of nights the candle lasts when burned for 1 hour per night.

end NUMINAMATH_GPT_candle_duration_1_hour_per_night_l243_24340


namespace NUMINAMATH_GPT_find_natural_numbers_l243_24359

theorem find_natural_numbers (n : ℕ) :
  (∀ k : ℕ, k^2 + ⌊ (n : ℝ) / (k^2 : ℝ) ⌋ ≥ 1991) ∧
  (∃ k_0 : ℕ, k_0^2 + ⌊ (n : ℝ) / (k_0^2 : ℝ) ⌋ < 1992) ↔
  990208 ≤ n ∧ n ≤ 991231 :=
by sorry

end NUMINAMATH_GPT_find_natural_numbers_l243_24359


namespace NUMINAMATH_GPT_transform_expression_l243_24378

variable {a : ℝ}

theorem transform_expression (h : a - 1 < 0) : 
  (a - 1) * Real.sqrt (-1 / (a - 1)) = -Real.sqrt (1 - a) :=
by
  sorry

end NUMINAMATH_GPT_transform_expression_l243_24378


namespace NUMINAMATH_GPT_solve_quadratic_l243_24313

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 → x = 1 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_quadratic_l243_24313


namespace NUMINAMATH_GPT_eight_packets_weight_l243_24395

variable (weight_per_can : ℝ)
variable (weight_per_packet : ℝ)

-- Conditions
axiom h1 : weight_per_can = 1
axiom h2 : 3 * weight_per_can = 8 * weight_per_packet
axiom h3 : weight_per_packet = 6 * weight_per_can

-- Question to be proved: 8 packets weigh 12 kg
theorem eight_packets_weight : 8 * weight_per_packet = 12 :=
by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_eight_packets_weight_l243_24395


namespace NUMINAMATH_GPT_degree_of_monomial_3ab_l243_24365

variable (a b : ℕ)

def monomialDegree (x y : ℕ) : ℕ :=
  x + y

theorem degree_of_monomial_3ab : monomialDegree 1 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_degree_of_monomial_3ab_l243_24365


namespace NUMINAMATH_GPT_eighth_graders_taller_rows_remain_ordered_l243_24329

-- Part (a)

theorem eighth_graders_taller {n : ℕ} (h8 : Fin n → ℚ) (h7 : Fin n → ℚ)
  (ordered8 : ∀ i j : Fin n, i ≤ j → h8 i ≤ h8 j)
  (ordered7 : ∀ i j : Fin n, i ≤ j → h7 i ≤ h7 j)
  (initial_condition : ∀ i : Fin n, h8 i > h7 i) :
  ∀ i : Fin n, h8 i > h7 i :=
sorry

-- Part (b)

theorem rows_remain_ordered {m n : ℕ} (h : Fin m → Fin n → ℚ)
  (row_ordered : ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k)
  (column_ordered_after : ∀ j : Fin n, ∀ i k : Fin m, i ≤ k → h i j ≤ h k j) :
  ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k :=
sorry

end NUMINAMATH_GPT_eighth_graders_taller_rows_remain_ordered_l243_24329


namespace NUMINAMATH_GPT_percentage_increase_l243_24335

theorem percentage_increase :
  let original_employees := 852
  let new_employees := 1065
  let increase := new_employees - original_employees
  let percentage := (increase.toFloat / original_employees.toFloat) * 100
  percentage = 25 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l243_24335


namespace NUMINAMATH_GPT_evaluate_expression_l243_24318

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/2) (hz : z = 8) : 
  x^3 * y^4 * z = 1/128 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l243_24318


namespace NUMINAMATH_GPT_optimal_station_placement_l243_24377

def distance_between_buildings : ℕ := 50
def workers_in_building (n : ℕ) : ℕ := n

def total_walking_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

theorem optimal_station_placement : ∃ x : ℝ, x = 150 ∧ (∀ y : ℝ, total_walking_distance x ≤ total_walking_distance y) :=
  sorry

end NUMINAMATH_GPT_optimal_station_placement_l243_24377


namespace NUMINAMATH_GPT_calculate_expression_l243_24393

variable (x : ℝ)

theorem calculate_expression : (1/2 * x^3)^2 = 1/4 * x^6 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l243_24393


namespace NUMINAMATH_GPT_sum_of_distinct_selections_is_34_l243_24319

-- Define a 4x4 grid filled sequentially from 1 to 16
def grid : List (List ℕ) := [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16]
]

-- Define a type for selections from the grid ensuring distinct rows and columns.
structure Selection where
  row : ℕ
  col : ℕ
  h_row : row < 4
  h_col : col < 4

-- Define the sum of any selection of 4 numbers from distinct rows and columns in the grid.
def sum_of_selection (selections : List Selection) : ℕ :=
  if h : List.length selections = 4 then
    List.sum (List.map (λ sel => (grid.get! sel.row).get! sel.col) selections)
  else 0

-- The main theorem
theorem sum_of_distinct_selections_is_34 (selections : List Selection) 
  (h_distinct_rows : List.Nodup (List.map (λ sel => sel.row) selections))
  (h_distinct_cols : List.Nodup (List.map (λ sel => sel.col) selections)) :
  sum_of_selection selections = 34 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sum_of_distinct_selections_is_34_l243_24319


namespace NUMINAMATH_GPT_quadratic_function_properties_l243_24326

noncomputable def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_properties_l243_24326


namespace NUMINAMATH_GPT_new_average_amount_l243_24334

theorem new_average_amount (A : ℝ) (H : A = 14) (new_amount : ℝ) (H1 : new_amount = 56) : 
  ((7 * A + new_amount) / 8) = 19.25 :=
by
  rw [H, H1]
  norm_num

end NUMINAMATH_GPT_new_average_amount_l243_24334


namespace NUMINAMATH_GPT_average_after_19_innings_is_23_l243_24381

-- Definitions for the conditions given in the problem
variables {A : ℝ} -- Let A be the average score before the 19th inning

-- Conditions: The cricketer scored 95 runs in the 19th inning and his average increased by 4 runs.
def total_runs_after_18_innings (A : ℝ) : ℝ := 18 * A
def total_runs_after_19th_inning (A : ℝ) : ℝ := total_runs_after_18_innings A + 95
def new_average_after_19_innings (A : ℝ) : ℝ := A + 4

-- The statement of the problem as a Lean theorem
theorem average_after_19_innings_is_23 :
  (18 * A + 95) / 19 = A + 4 → A = 19 → (A + 4) = 23 :=
by
  intros hA h_avg_increased
  sorry

end NUMINAMATH_GPT_average_after_19_innings_is_23_l243_24381


namespace NUMINAMATH_GPT_no_arithmetic_sqrt_of_neg_real_l243_24394

theorem no_arithmetic_sqrt_of_neg_real (x : ℝ) (h : x < 0) : ¬ ∃ y : ℝ, y * y = x :=
by
  sorry

end NUMINAMATH_GPT_no_arithmetic_sqrt_of_neg_real_l243_24394


namespace NUMINAMATH_GPT_work_completion_time_l243_24373

theorem work_completion_time (A_work_rate B_work_rate C_work_rate : ℝ) 
  (hA : A_work_rate = 1 / 8) 
  (hB : B_work_rate = 1 / 16) 
  (hC : C_work_rate = 1 / 16) : 
  1 / (A_work_rate + B_work_rate + C_work_rate) = 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_work_completion_time_l243_24373


namespace NUMINAMATH_GPT_parabola_vertex_coordinates_l243_24371

theorem parabola_vertex_coordinates :
  ∃ (h k : ℝ), (∀ (x : ℝ), (y = (x - h)^2 + k) = (y = (x-1)^2 + 2)) ∧ h = 1 ∧ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_coordinates_l243_24371


namespace NUMINAMATH_GPT_find_a_l243_24370

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end NUMINAMATH_GPT_find_a_l243_24370


namespace NUMINAMATH_GPT_income_fraction_from_tips_l243_24389

variable (S T : ℝ)

theorem income_fraction_from_tips :
  (T = (9 / 4) * S) → (T / (S + T) = 9 / 13) :=
by
  sorry

end NUMINAMATH_GPT_income_fraction_from_tips_l243_24389


namespace NUMINAMATH_GPT_find_velocity_l243_24396

variable (k V : ℝ)
variable (P A : ℕ)

theorem find_velocity (k_eq : k = 1 / 200) 
  (initial_cond : P = 4 ∧ A = 2 ∧ V = 20) 
  (new_cond : P = 16 ∧ A = 4) : 
  V = 20 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_velocity_l243_24396


namespace NUMINAMATH_GPT_manufacturer_price_l243_24388

theorem manufacturer_price :
  ∃ M : ℝ, 
    (∃ R : ℝ, 
      R = 1.15 * M ∧
      ∃ D : ℝ, 
        D = 0.85 * R ∧
        R - D = 57.5) ∧
    M = 333.33 := 
by
  sorry

end NUMINAMATH_GPT_manufacturer_price_l243_24388


namespace NUMINAMATH_GPT_quadratic_eq_has_two_distinct_real_roots_l243_24333

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Problem statement: Prove that the quadratic equation x^2 + 3x - 2 = 0 has two distinct real roots
theorem quadratic_eq_has_two_distinct_real_roots :
  discriminant 1 3 (-2) > 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_quadratic_eq_has_two_distinct_real_roots_l243_24333


namespace NUMINAMATH_GPT_alexis_total_sewing_time_l243_24399

-- Define the time to sew a skirt and a coat
def t_skirt : ℕ := 2
def t_coat : ℕ := 7

-- Define the numbers of skirts and coats
def n_skirts : ℕ := 6
def n_coats : ℕ := 4

-- Define the total time
def total_time : ℕ := t_skirt * n_skirts + t_coat * n_coats

-- State the theorem
theorem alexis_total_sewing_time : total_time = 40 :=
by
  -- the proof would go here; we're skipping the proof as per instructions
  sorry

end NUMINAMATH_GPT_alexis_total_sewing_time_l243_24399


namespace NUMINAMATH_GPT_no_function_satisfies_inequality_l243_24352

theorem no_function_satisfies_inequality (f : ℝ → ℝ) :
  ¬ ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| :=
sorry

end NUMINAMATH_GPT_no_function_satisfies_inequality_l243_24352


namespace NUMINAMATH_GPT_net_change_is_minus_0_19_l243_24360

-- Define the yearly change factors as provided in the conditions
def yearly_changes : List ℚ := [6/5, 11/10, 7/10, 4/5, 11/10]

-- Compute the net change over the five years
def net_change (changes : List ℚ) : ℚ :=
  changes.foldl (λ acc x => acc * x) 1 - 1

-- Define the target value for the net change
def target_net_change : ℚ := -19 / 100

-- The theorem to prove the net change calculated matches the target net change
theorem net_change_is_minus_0_19 : net_change yearly_changes = target_net_change :=
  by
    sorry

end NUMINAMATH_GPT_net_change_is_minus_0_19_l243_24360


namespace NUMINAMATH_GPT_user_level_1000_l243_24317

noncomputable def user_level (points : ℕ) : ℕ :=
if points >= 1210 then 18
else if points >= 1000 then 17
else if points >= 810 then 16
else if points >= 640 then 15
else if points >= 490 then 14
else if points >= 360 then 13
else if points >= 250 then 12
else if points >= 160 then 11
else if points >= 90 then 10
else 0

theorem user_level_1000 : user_level 1000 = 17 :=
by {
  -- proof will be written here
  sorry
}

end NUMINAMATH_GPT_user_level_1000_l243_24317


namespace NUMINAMATH_GPT_chloe_pawn_loss_l243_24345

theorem chloe_pawn_loss (sophia_lost : ℕ) (total_left : ℕ) (total_initial : ℕ) (each_start : ℕ) (sophia_initial : ℕ) :
  sophia_lost = 5 → total_left = 10 → each_start = 8 → total_initial = 16 → sophia_initial = 8 →
  ∃ (chloe_lost : ℕ), chloe_lost = 1 :=
by
  sorry

end NUMINAMATH_GPT_chloe_pawn_loss_l243_24345


namespace NUMINAMATH_GPT_deck_width_l243_24353

theorem deck_width (w : ℝ) : 
  (10 + 2 * w) * (12 + 2 * w) = 360 → w = 4 := 
by 
  sorry

end NUMINAMATH_GPT_deck_width_l243_24353


namespace NUMINAMATH_GPT_winnie_retains_lollipops_l243_24309

theorem winnie_retains_lollipops :
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  lollipops_total % friends = 10 :=
by
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  show lollipops_total % friends = 10
  sorry

end NUMINAMATH_GPT_winnie_retains_lollipops_l243_24309


namespace NUMINAMATH_GPT_range_of_omega_l243_24380

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ (a b : ℝ), a ≠ b ∧ 0 ≤ a ∧ a ≤ π/2 ∧ 0 ≤ b ∧ b ≤ π/2 ∧ f ω a + f ω b = 4) ↔ 5 ≤ ω ∧ ω < 9 :=
sorry

end NUMINAMATH_GPT_range_of_omega_l243_24380


namespace NUMINAMATH_GPT_tank_base_length_width_difference_l243_24328

variable (w l h : ℝ)

theorem tank_base_length_width_difference :
  (l = 5 * w) →
  (h = (1/2) * w) →
  (l * w * h = 3600) →
  (|l - w - 45.24| < 0.01) := 
by
  sorry

end NUMINAMATH_GPT_tank_base_length_width_difference_l243_24328


namespace NUMINAMATH_GPT_total_spent_l243_24306

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_friends : ℕ := 4

theorem total_spent : (original_price * (1 - discount_rate) * number_of_friends) = 40 := by
  sorry

end NUMINAMATH_GPT_total_spent_l243_24306


namespace NUMINAMATH_GPT_average_of_xyz_l243_24320

variable (x y z : ℝ)

theorem average_of_xyz (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 := by
  sorry

end NUMINAMATH_GPT_average_of_xyz_l243_24320


namespace NUMINAMATH_GPT_train_pass_time_l243_24339

-- Definitions based on the conditions
def train_length : ℕ := 280  -- train length in meters
def train_speed_kmh : ℕ := 72  -- train speed in km/hr
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 5 / 18)  -- train speed in m/s

-- Theorem statement
theorem train_pass_time : (train_length / train_speed_ms) = 14 := by
  sorry

end NUMINAMATH_GPT_train_pass_time_l243_24339


namespace NUMINAMATH_GPT_quadratic_real_roots_l243_24366

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k^2 * x^2 - (2 * k + 1) * x + 1 = 0 ∧ ∃ x2 : ℝ, k^2 * x2^2 - (2 * k + 1) * x2 + 1 = 0)
  ↔ (k ≥ -1/4 ∧ k ≠ 0) := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l243_24366


namespace NUMINAMATH_GPT_maximum_possible_savings_is_63_l243_24390

-- Definitions of the conditions
def doughnut_price := 8
def doughnut_discount_2 := 14
def doughnut_discount_4 := 26

def croissant_price := 10
def croissant_discount_3 := 28
def croissant_discount_5 := 45

def muffin_price := 6
def muffin_discount_2 := 11
def muffin_discount_6 := 30

-- Quantities to purchase
def doughnut_qty := 20
def croissant_qty := 15
def muffin_qty := 18

-- Prices calculated from quantities
def total_price_without_discount :=
  doughnut_qty * doughnut_price + croissant_qty * croissant_price + muffin_qty * muffin_price

def total_price_with_discount :=
  5 * doughnut_discount_4 + 3 * croissant_discount_5 + 3 * muffin_discount_6

def maximum_savings := total_price_without_discount - total_price_with_discount

theorem maximum_possible_savings_is_63 : maximum_savings = 63 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_maximum_possible_savings_is_63_l243_24390


namespace NUMINAMATH_GPT_proposition_neg_p_and_q_false_l243_24397

theorem proposition_neg_p_and_q_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end NUMINAMATH_GPT_proposition_neg_p_and_q_false_l243_24397


namespace NUMINAMATH_GPT_simplification_l243_24367

theorem simplification (b : ℝ) : 3 * b * (3 * b^3 + 2 * b) - 2 * b^2 = 9 * b^4 + 4 * b^2 :=
by
  sorry

end NUMINAMATH_GPT_simplification_l243_24367


namespace NUMINAMATH_GPT_system_of_equations_solution_l243_24305

theorem system_of_equations_solution (b : ℝ) :
  (∀ (a : ℝ), ∃ (x y : ℝ), (x - 1)^2 + y^2 = 1 ∧ a * x + y = a * b) ↔ 0 ≤ b ∧ b ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l243_24305


namespace NUMINAMATH_GPT_cindy_correct_answer_l243_24308

/-- 
Cindy accidentally first subtracted 9 from a number, then multiplied the result 
by 2 before dividing by 6, resulting in an answer of 36. 
Following these steps, she was actually supposed to subtract 12 from the 
number and then divide by 8. What would her answer have been had she worked the 
problem correctly?
-/
theorem cindy_correct_answer :
  ∀ (x : ℝ), (2 * (x - 9) / 6 = 36) → ((x - 12) / 8 = 13.125) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_cindy_correct_answer_l243_24308


namespace NUMINAMATH_GPT_multiplication_correct_l243_24362

theorem multiplication_correct : 3795421 * 8634.25 = 32774670542.25 := by
  sorry

end NUMINAMATH_GPT_multiplication_correct_l243_24362


namespace NUMINAMATH_GPT_books_added_after_lunch_l243_24343

-- Definitions for the given conditions
def initial_books : Int := 100
def books_borrowed_lunch : Int := 50
def books_remaining_lunch : Int := initial_books - books_borrowed_lunch
def books_borrowed_evening : Int := 30
def books_remaining_evening : Int := 60

-- Let X be the number of books added after lunchtime
variable (X : Int)

-- The proof goal in Lean statement
theorem books_added_after_lunch (h : books_remaining_lunch + X - books_borrowed_evening = books_remaining_evening) :
  X = 40 := by
  sorry

end NUMINAMATH_GPT_books_added_after_lunch_l243_24343


namespace NUMINAMATH_GPT_option_d_is_correct_l243_24351

theorem option_d_is_correct (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b :=
by
  sorry

end NUMINAMATH_GPT_option_d_is_correct_l243_24351


namespace NUMINAMATH_GPT_triangle_inequality_proof_l243_24310

noncomputable def triangle_inequality (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) : Prop :=
  Real.pi / 3 ≤ (a * A + b * B + c * C) / (a + b + c) ∧ (a * A + b * B + c * C) / (a + b + c) < Real.pi / 2

theorem triangle_inequality_proof (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h₁: A + B + C = Real.pi) (h₂: ∀ {x y : ℝ}, A ≥ B  → a ≥ b → A * b + B * a ≤ A * a + B * b) 
  (h₃: ∀ {x y : ℝ}, x + y > 0 → A * x + B * y + C * (a + b - x - y) > 0) : 
  triangle_inequality A B C a b c hABC :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_proof_l243_24310


namespace NUMINAMATH_GPT_max_value_of_f_l243_24379

noncomputable def f (x : ℝ) : ℝ := 3^x - 9^x

theorem max_value_of_f : ∃ x : ℝ, f x = 1 / 4 := sorry

end NUMINAMATH_GPT_max_value_of_f_l243_24379


namespace NUMINAMATH_GPT_exist_xyz_modular_l243_24342

theorem exist_xyz_modular {n a b c : ℕ} (hn : 0 < n) (ha : a ≤ 3 * n ^ 2 + 4 * n) (hb : b ≤ 3 * n ^ 2 + 4 * n) (hc : c ≤ 3 * n ^ 2 + 4 * n) :
  ∃ (x y z : ℤ), abs x ≤ 2 * n ∧ abs y ≤ 2 * n ∧ abs z ≤ 2 * n ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 :=
sorry

end NUMINAMATH_GPT_exist_xyz_modular_l243_24342


namespace NUMINAMATH_GPT_determine_digit_I_l243_24382

theorem determine_digit_I (F I V E T H R N : ℕ) (hF : F = 8) (hE_odd : E = 1 ∨ E = 3 ∨ E = 5 ∨ E = 7 ∨ E = 9)
  (h_diff : F ≠ I ∧ F ≠ V ∧ F ≠ E ∧ F ≠ T ∧ F ≠ H ∧ F ≠ R ∧ F ≠ N 
             ∧ I ≠ V ∧ I ≠ E ∧ I ≠ T ∧ I ≠ H ∧ I ≠ R ∧ I ≠ N 
             ∧ V ≠ E ∧ V ≠ T ∧ V ≠ H ∧ V ≠ R ∧ V ≠ N 
             ∧ E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ N 
             ∧ T ≠ H ∧ T ≠ R ∧ T ≠ N 
             ∧ H ≠ R ∧ H ≠ N 
             ∧ R ≠ N)
  (h_verify_sum : (10^3 * 8 + 10^2 * I + 10 * V + E) + (10^4 * T + 10^3 * H + 10^2 * R + 11 * E) = 10^3 * N + 10^2 * I + 10 * N + E) :
  I = 4 := 
sorry

end NUMINAMATH_GPT_determine_digit_I_l243_24382


namespace NUMINAMATH_GPT_percentage_loss_is_correct_l243_24372

noncomputable def initial_cost : ℝ := 300
noncomputable def selling_price : ℝ := 255
noncomputable def loss : ℝ := initial_cost - selling_price
noncomputable def percentage_loss : ℝ := (loss / initial_cost) * 100

theorem percentage_loss_is_correct :
  percentage_loss = 15 :=
sorry

end NUMINAMATH_GPT_percentage_loss_is_correct_l243_24372


namespace NUMINAMATH_GPT_rectangular_field_area_l243_24336

noncomputable def a : ℝ := 14
noncomputable def c : ℝ := 17
noncomputable def b := Real.sqrt (c^2 - a^2)
noncomputable def area := a * b

theorem rectangular_field_area : area = 14 * Real.sqrt 93 := by
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l243_24336


namespace NUMINAMATH_GPT_solve_for_x_l243_24300

theorem solve_for_x (x : ℝ) (h : 2 * x + 10 = (1 / 2) * (5 * x + 30)) : x = -10 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l243_24300


namespace NUMINAMATH_GPT_intersection_of_A_and_complement_of_B_l243_24392

noncomputable def U : Set ℝ := Set.univ

noncomputable def A : Set ℝ := { x : ℝ | 2^x * (x - 2) < 1 }
noncomputable def B : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (1 - x) }
noncomputable def B_complement : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_A_and_complement_of_B :
  A ∩ B_complement = { x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_complement_of_B_l243_24392


namespace NUMINAMATH_GPT_smaller_molds_radius_l243_24321

theorem smaller_molds_radius (r : ℝ) : 
  (∀ V_large V_small : ℝ, 
     V_large = (2/3) * π * (2:ℝ)^3 ∧
     V_small = (2/3) * π * r^3 ∧
     8 * V_small = V_large) → r = 1 := by
  sorry

end NUMINAMATH_GPT_smaller_molds_radius_l243_24321


namespace NUMINAMATH_GPT_marley_fruits_l243_24338

theorem marley_fruits 
    (louis_oranges : ℕ := 5) (louis_apples : ℕ := 3)
    (samantha_oranges : ℕ := 8) (samantha_apples : ℕ := 7)
    (marley_oranges : ℕ := 2 * louis_oranges)
    (marley_apples : ℕ := 3 * samantha_apples) :
    marley_oranges + marley_apples = 31 := by
  sorry

end NUMINAMATH_GPT_marley_fruits_l243_24338


namespace NUMINAMATH_GPT_find_n_l243_24364

theorem find_n : (∃ n : ℕ, 2^3 * 8^3 = 2^(2 * n)) ↔ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l243_24364


namespace NUMINAMATH_GPT_smallest_a_l243_24361

theorem smallest_a 
  (a : ℤ) (P : ℤ → ℤ) 
  (h_pos : 0 < a) 
  (hP1 : P 1 = a) (hP5 : P 5 = a) (hP7 : P 7 = a) (hP9 : P 9 = a) 
  (hP2 : P 2 = -a) (hP4 : P 4 = -a) (hP6 : P 6 = -a) (hP8 : P 8 = -a) : 
  a ≥ 336 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_l243_24361


namespace NUMINAMATH_GPT_ratio_problem_l243_24354

theorem ratio_problem (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 3 * d) : d = 15 / 7 := by
  sorry

end NUMINAMATH_GPT_ratio_problem_l243_24354


namespace NUMINAMATH_GPT_first_class_product_rate_l243_24348

theorem first_class_product_rate
  (total_products : ℕ)
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (pass_rate_correct : pass_rate = 0.95)
  (first_class_rate_correct : first_class_rate_among_qualified = 0.2) :
  (first_class_rate_among_qualified * pass_rate : ℝ) = 0.19 :=
by
  rw [pass_rate_correct, first_class_rate_correct]
  norm_num


end NUMINAMATH_GPT_first_class_product_rate_l243_24348


namespace NUMINAMATH_GPT_find_positive_value_of_X_l243_24384

-- define the relation X # Y
def rel (X Y : ℝ) : ℝ := X^2 + Y^2

theorem find_positive_value_of_X (X : ℝ) (h : rel X 7 = 250) : X = Real.sqrt 201 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_value_of_X_l243_24384


namespace NUMINAMATH_GPT_shortest_routes_l243_24311

def side_length : ℕ := 10
def refuel_distance : ℕ := 30
def num_squares_per_refuel := refuel_distance / side_length

theorem shortest_routes (A B : Type) (distance_AB : ℕ) (shortest_paths : Π (A B : Type), ℕ) : 
  shortest_paths A B = 54 := by
  sorry

end NUMINAMATH_GPT_shortest_routes_l243_24311


namespace NUMINAMATH_GPT_complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l243_24368

def U : Set ℝ := {x | x ≥ -2}
def A : Set ℝ := {x | 2 < x ∧ x < 10}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}

theorem complement_A :
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x ≥ 10} :=
by sorry

theorem complement_A_intersection_B :
  (U \ A) ∩ B = {2} :=
by sorry

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 8} :=
by sorry

theorem complement_intersection_A_B :
  U \ (A ∩ B) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x > 8} :=
by sorry

end NUMINAMATH_GPT_complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l243_24368


namespace NUMINAMATH_GPT_ratio_of_x_intercepts_l243_24358

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v : ℝ)
  (hu : u = -b / 5) (hv : v = -b / 3) : u / v = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_of_x_intercepts_l243_24358


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_is_2_l243_24347

variable {a : ℕ → ℝ} (h : ∀ n : ℕ, a n * a (n + 1) = 4 ^ n)

theorem geometric_sequence_common_ratio_is_2 : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_is_2_l243_24347


namespace NUMINAMATH_GPT_max_range_f_plus_2g_l243_24374

noncomputable def max_val_of_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) : ℝ :=
  9

theorem max_range_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) :
  ∃ (a b : ℝ), (-3 ≤ a ∧ a ≤ 5) ∧ (-8 ≤ b ∧ b ≤ 4) ∧ b = 9 := 
sorry

end NUMINAMATH_GPT_max_range_f_plus_2g_l243_24374


namespace NUMINAMATH_GPT_painting_cost_l243_24322

theorem painting_cost (total_cost : ℕ) (num_paintings : ℕ) (price : ℕ)
  (h1 : total_cost = 104)
  (h2 : 10 < num_paintings)
  (h3 : num_paintings < 60)
  (h4 : total_cost = num_paintings * price)
  (h5 : price ∈ {d ∈ {d : ℕ | d > 0} | total_cost % d = 0}) :
  price = 2 ∨ price = 4 ∨ price = 8 :=
by
  sorry

end NUMINAMATH_GPT_painting_cost_l243_24322


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l243_24375

variable (a b : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_cond1 : a ≥ (1 / a) + (2 / b))
variable (h_cond2 : b ≥ (3 / a) + (2 / b))

/-- Statement 1: Prove that a + b ≥ 4 under the given conditions. -/
theorem problem1 : (a + b) ≥ 4 := 
by 
  sorry

/-- Statement 2: Prove that a^2 + b^2 ≥ 3 + 2√6 under the given conditions. -/
theorem problem2 : (a^2 + b^2) ≥ (3 + 2 * Real.sqrt 6) := 
by 
  sorry

/-- Statement 3: Prove that (1/a) + (1/b) < 1 + (√2/2) under the given conditions. -/
theorem problem3 : (1 / a) + (1 / b) < 1 + (Real.sqrt 2 / 2) := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l243_24375


namespace NUMINAMATH_GPT_mower_value_drop_l243_24312

theorem mower_value_drop :
  ∀ (initial_value value_six_months value_after_year : ℝ) (percentage_drop_six_months percentage_drop_next_year : ℝ),
  initial_value = 100 →
  percentage_drop_six_months = 0.25 →
  value_six_months = initial_value * (1 - percentage_drop_six_months) →
  value_after_year = 60 →
  percentage_drop_next_year = 1 - (value_after_year / value_six_months) →
  percentage_drop_next_year * 100 = 20 :=
by
  intros initial_value value_six_months value_after_year percentage_drop_six_months percentage_drop_next_year
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_mower_value_drop_l243_24312


namespace NUMINAMATH_GPT_ratio_of_volumes_l243_24330

variables (A B : ℚ)

theorem ratio_of_volumes 
  (h1 : (3/8) * A = (5/8) * B) :
  A / B = 5 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_volumes_l243_24330


namespace NUMINAMATH_GPT_number_of_white_balls_l243_24387

theorem number_of_white_balls (a : ℕ) (h1 : 3 + a ≠ 0) (h2 : (3 : ℚ) / (3 + a) = 3 / 7) : a = 4 :=
sorry

end NUMINAMATH_GPT_number_of_white_balls_l243_24387


namespace NUMINAMATH_GPT_custom_operation_example_l243_24383

def custom_operation (a b : ℚ) : ℚ :=
  a^3 - 2 * a * b + 4

theorem custom_operation_example : custom_operation 4 (-9) = 140 :=
by
  sorry

end NUMINAMATH_GPT_custom_operation_example_l243_24383


namespace NUMINAMATH_GPT_prime_solution_exists_l243_24327

theorem prime_solution_exists (p : ℕ) (hp : Nat.Prime p) : ∃ x y z : ℤ, x^2 + y^2 + (p:ℤ) * z = 2003 := 
by 
  sorry

end NUMINAMATH_GPT_prime_solution_exists_l243_24327


namespace NUMINAMATH_GPT_range_of_a_l243_24332

-- Define the function f(x) = x^2 - 3x
def f (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the interval as a closed interval from -1 to 1
def interval : Set ℝ := Set.Icc (-1) (1)

-- State the main proposition
theorem range_of_a (a : ℝ) :
  (∃ x ∈ interval, -x^2 + 3 * x + a > 0) ↔ a > -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l243_24332


namespace NUMINAMATH_GPT_sum_of_values_l243_24325

def f (x : Int) : Int := Int.natAbs x - 3
def g (x : Int) : Int := -x

def fogof (x : Int) : Int := f (g (f x))

theorem sum_of_values :
  (fogof (-5)) + (fogof (-4)) + (fogof (-3)) + (fogof (-2)) + (fogof (-1)) + (fogof 0) + (fogof 1) + (fogof 2) + (fogof 3) + (fogof 4) + (fogof 5) = -17 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_values_l243_24325


namespace NUMINAMATH_GPT_points_difference_l243_24301

theorem points_difference :
  let points_td := 7
  let points_epc := 1
  let points_fg := 3
  
  let touchdowns_BG := 6
  let epc_BG := 4
  let fg_BG := 2
  
  let touchdowns_CF := 8
  let epc_CF := 6
  let fg_CF := 3
  
  let total_BG := touchdowns_BG * points_td + epc_BG * points_epc + fg_BG * points_fg
  let total_CF := touchdowns_CF * points_td + epc_CF * points_epc + fg_CF * points_fg
  
  total_CF - total_BG = 19 := by
  sorry

end NUMINAMATH_GPT_points_difference_l243_24301


namespace NUMINAMATH_GPT_find_multiple_l243_24355

theorem find_multiple (a b m : ℤ) (h1 : a * b = m * (a + b) + 12) 
(h2 : b = 10) (h3 : b - a = 6) : m = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_multiple_l243_24355


namespace NUMINAMATH_GPT_travelers_on_liner_l243_24324

theorem travelers_on_liner (a : ℕ) : 
  250 ≤ a ∧ a ≤ 400 ∧ a % 15 = 7 ∧ a % 25 = 17 → a = 292 ∨ a = 367 :=
by
  sorry

end NUMINAMATH_GPT_travelers_on_liner_l243_24324


namespace NUMINAMATH_GPT_find_center_radius_l243_24303

noncomputable def circle_center_radius (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y - 6 = 0 → 
  ∃ (h k r : ℝ), (x + 1) * (x + 1) + (y - 2) * (y - 2) = r ∧ h = -1 ∧ k = 2 ∧ r = 11

theorem find_center_radius :
  circle_center_radius x y :=
sorry

end NUMINAMATH_GPT_find_center_radius_l243_24303


namespace NUMINAMATH_GPT_min_value_of_quadratic_l243_24337

theorem min_value_of_quadratic (x y z : ℝ) 
  (h1 : x + 2 * y - 5 * z = 3)
  (h2 : x - 2 * y - z = -5) : 
  ∃ z' : ℝ,  x = 3 * z' - 1 ∧ y = z' + 2 ∧ (11 * z' * z' - 2 * z' + 5 = (54 : ℝ) / 11) :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l243_24337


namespace NUMINAMATH_GPT_total_salaries_l243_24356

theorem total_salaries (A_salary B_salary : ℝ)
  (hA : A_salary = 1500)
  (hsavings : 0.05 * A_salary = 0.15 * B_salary) :
  A_salary + B_salary = 2000 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_salaries_l243_24356


namespace NUMINAMATH_GPT_total_amount_paid_l243_24369

theorem total_amount_paid (sales_tax : ℝ) (tax_rate : ℝ) (cost_tax_free_items : ℝ) : 
  sales_tax = 1.28 → tax_rate = 0.08 → cost_tax_free_items = 12.72 → 
  (sales_tax / tax_rate + sales_tax + cost_tax_free_items) = 30.00 :=
by
  intros h1 h2 h3
  -- Proceed with the proof using h1, h2, and h3
  sorry

end NUMINAMATH_GPT_total_amount_paid_l243_24369


namespace NUMINAMATH_GPT_total_pieces_of_chicken_needed_l243_24363

def friedChickenDinnerPieces := 8
def chickenPastaPieces := 2
def barbecueChickenPieces := 4
def grilledChickenSaladPieces := 1

def friedChickenDinners := 4
def chickenPastaOrders := 8
def barbecueChickenOrders := 5
def grilledChickenSaladOrders := 6

def totalChickenPiecesNeeded :=
  (friedChickenDinnerPieces * friedChickenDinners) +
  (chickenPastaPieces * chickenPastaOrders) +
  (barbecueChickenPieces * barbecueChickenOrders) +
  (grilledChickenSaladPieces * grilledChickenSaladOrders)

theorem total_pieces_of_chicken_needed : totalChickenPiecesNeeded = 74 := by
  sorry

end NUMINAMATH_GPT_total_pieces_of_chicken_needed_l243_24363


namespace NUMINAMATH_GPT_infinite_geometric_series_second_term_l243_24302

theorem infinite_geometric_series_second_term (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 16) (h3 : S = a / (1 - r)) : a * r = 3 := 
sorry

end NUMINAMATH_GPT_infinite_geometric_series_second_term_l243_24302


namespace NUMINAMATH_GPT_collinear_points_m_equals_4_l243_24346

theorem collinear_points_m_equals_4 (m : ℝ)
  (h1 : (3 - 12) / (1 - -2) = (-6 - 12) / (m - -2)) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_collinear_points_m_equals_4_l243_24346


namespace NUMINAMATH_GPT_evaluate_fraction_l243_24376

theorem evaluate_fraction : (3 / (1 - 3 / 4) = 12) := by
  have h : (1 - 3 / 4) = 1 / 4 := by
    sorry
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l243_24376


namespace NUMINAMATH_GPT_calculate_gfg3_l243_24331

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 5 * x + 2

theorem calculate_gfg3 : g (f (g 3)) = 192 := by
  sorry

end NUMINAMATH_GPT_calculate_gfg3_l243_24331


namespace NUMINAMATH_GPT_large_cartridge_pages_correct_l243_24357

-- Define the conditions
def small_cartridge_pages : ℕ := 600
def medium_cartridge_pages : ℕ := 2 * 3 * small_cartridge_pages / 6
def large_cartridge_pages : ℕ := 2 * 3 * medium_cartridge_pages / 6

-- The theorem to prove
theorem large_cartridge_pages_correct :
  large_cartridge_pages = 1350 :=
by
  sorry

end NUMINAMATH_GPT_large_cartridge_pages_correct_l243_24357


namespace NUMINAMATH_GPT_ratio_of_numbers_l243_24349

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 124) (h2 : y = 3 * x) : x / Nat.gcd x y = 1 ∧ y / Nat.gcd x y = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l243_24349
