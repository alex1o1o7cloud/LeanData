import Mathlib

namespace NUMINAMATH_GPT_units_digit_of_x_l989_98915

theorem units_digit_of_x 
  (a x : ℕ) 
  (h1 : a * x = 14^8) 
  (h2 : a % 10 = 9) : 
  x % 10 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_of_x_l989_98915


namespace NUMINAMATH_GPT_hypotenuse_length_l989_98902

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end NUMINAMATH_GPT_hypotenuse_length_l989_98902


namespace NUMINAMATH_GPT_clea_ride_time_l989_98957

-- Definitions: Let c be Clea's walking speed without the bag and s be the speed of the escalator

variables (c s : ℝ)

-- Conditions translated into equations
def distance_without_bag := 80 * c
def distance_with_bag_and_escalator := 38 * (0.7 * c + s)

-- The problem: Prove that the time t for Clea to ride down the escalator while just standing on it with the bag is 57 seconds.
theorem clea_ride_time :
  (38 * (0.7 * c + s) = 80 * c) ->
  (t = 80 * (38 / 53.4)) ->
  t = 57 :=
sorry

end NUMINAMATH_GPT_clea_ride_time_l989_98957


namespace NUMINAMATH_GPT_solve_system_of_equations_l989_98903

theorem solve_system_of_equations (x y : ℝ) (h1 : 2 * x + 3 * y = 7) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
    -- The proof is not required, so we put a sorry here.
    sorry

end NUMINAMATH_GPT_solve_system_of_equations_l989_98903


namespace NUMINAMATH_GPT_tammy_earnings_after_3_weeks_l989_98940

noncomputable def oranges_picked_per_day (num_trees : ℕ) (oranges_per_tree : ℕ) : ℕ :=
  num_trees * oranges_per_tree

noncomputable def packs_sold_per_day (oranges_per_day : ℕ) (oranges_per_pack : ℕ) : ℕ :=
  oranges_per_day / oranges_per_pack

noncomputable def total_packs_sold_in_weeks (packs_per_day : ℕ) (days_in_week : ℕ) (num_weeks : ℕ) : ℕ :=
  packs_per_day * days_in_week * num_weeks

noncomputable def money_earned (total_packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  total_packs * price_per_pack

theorem tammy_earnings_after_3_weeks :
  let num_trees := 10
  let oranges_per_tree := 12
  let oranges_per_pack := 6
  let price_per_pack := 2
  let days_in_week := 7
  let num_weeks := 3
  oranges_picked_per_day num_trees oranges_per_tree /
  oranges_per_pack *
  days_in_week *
  num_weeks *
  price_per_pack = 840 :=
by {
  sorry
}

end NUMINAMATH_GPT_tammy_earnings_after_3_weeks_l989_98940


namespace NUMINAMATH_GPT_possible_values_of_f_zero_l989_98914

theorem possible_values_of_f_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x * f y) :
  f 0 = 0 ∨ f 0 = 1 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_f_zero_l989_98914


namespace NUMINAMATH_GPT_henry_collection_cost_l989_98931

def initial_figures : ℕ := 3
def total_needed : ℕ := 8
def cost_per_figure : ℕ := 6

theorem henry_collection_cost : 
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  total_cost = 30 := 
by
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  sorry

end NUMINAMATH_GPT_henry_collection_cost_l989_98931


namespace NUMINAMATH_GPT_total_driving_routes_l989_98934

def num_starting_points : ℕ := 4
def num_destinations : ℕ := 3

theorem total_driving_routes (h1 : ¬(num_starting_points = 0)) (h2 : ¬(num_destinations = 0)) : 
  num_starting_points * num_destinations = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_driving_routes_l989_98934


namespace NUMINAMATH_GPT_sum_of_numbers_le_1_1_l989_98946

theorem sum_of_numbers_le_1_1 :
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  filtered.sum = 1.4 :=
by
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  have : filtered = [0.9, 0.5] := sorry
  have : filtered.sum = 1.4 := sorry
  exact this

end NUMINAMATH_GPT_sum_of_numbers_le_1_1_l989_98946


namespace NUMINAMATH_GPT_sum_real_imag_l989_98950

theorem sum_real_imag (z : ℂ) (hz : z = 3 - 4 * I) : z.re + z.im = -1 :=
by {
  -- Because the task asks for no proof, we're leaving it with 'sorry'.
  sorry
}

end NUMINAMATH_GPT_sum_real_imag_l989_98950


namespace NUMINAMATH_GPT_total_games_played_l989_98908

-- Define the conditions as Lean 4 definitions
def games_won : Nat := 12
def games_lost : Nat := 4

-- Prove the total number of games played is 16
theorem total_games_played : games_won + games_lost = 16 := 
by
  -- Place a proof placeholder
  sorry

end NUMINAMATH_GPT_total_games_played_l989_98908


namespace NUMINAMATH_GPT_perpendicular_chords_cosine_bound_l989_98918

theorem perpendicular_chords_cosine_bound 
  (a b : ℝ) 
  (h_ab : a > b) 
  (h_b0 : b > 0) 
  (θ1 θ2 : ℝ) 
  (x y : ℝ → ℝ) 
  (h_ellipse : ∀ t, x t = a * Real.cos t ∧ y t = b * Real.sin t) 
  (h_theta1 : ∃ t1, (x t1 = a * Real.cos θ1 ∧ y t1 = b * Real.sin θ1)) 
  (h_theta2 : ∃ t2, (x t2 = a * Real.cos θ2 ∧ y t2 = b * Real.sin θ2)) 
  (h_perpendicular: θ1 = θ2 + π / 2 ∨ θ1 = θ2 - π / 2) :
  0 ≤ |Real.cos (θ1 - θ2)| ∧ |Real.cos (θ1 - θ2)| ≤ (a ^ 2 - b ^ 2) / (a ^ 2 + b ^ 2) :=
sorry

end NUMINAMATH_GPT_perpendicular_chords_cosine_bound_l989_98918


namespace NUMINAMATH_GPT_platform_length_l989_98917

theorem platform_length (train_length : ℕ) (time_cross_platform : ℕ) (time_cross_pole : ℕ) (train_speed : ℕ) (L : ℕ)
  (h1 : train_length = 500) 
  (h2 : time_cross_platform = 65) 
  (h3 : time_cross_pole = 25) 
  (h4 : train_speed = train_length / time_cross_pole)
  (h5 : train_speed = (train_length + L) / time_cross_platform) :
  L = 800 := 
sorry

end NUMINAMATH_GPT_platform_length_l989_98917


namespace NUMINAMATH_GPT_graph_passes_through_quadrants_l989_98947

theorem graph_passes_through_quadrants (k : ℝ) (h : k < 0) :
  ∀ (x y : ℝ), (y = k * x - k) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_quadrants_l989_98947


namespace NUMINAMATH_GPT_max_non_managers_l989_98949

-- Definitions of the problem conditions
variable (m n : ℕ)
variable (h : m = 8)
variable (hratio : (7:ℚ) / 24 < m / n)

-- The theorem we need to prove
theorem max_non_managers (m n : ℕ) (h : m = 8) (hratio : ((7:ℚ) / 24 < m / n)) :
  n ≤ 27 := 
sorry

end NUMINAMATH_GPT_max_non_managers_l989_98949


namespace NUMINAMATH_GPT_minimum_a_l989_98955

noncomputable def f (x : ℝ) := x - Real.exp (x - Real.exp 1)

theorem minimum_a (a : ℝ) (x1 x2 : ℝ) (hx : x2 - x1 ≥ Real.exp 1)
  (hy : Real.exp x1 = 1 + Real.log (x2 - a)) : a ≥ Real.exp 1 - 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_a_l989_98955


namespace NUMINAMATH_GPT_speed_of_stream_l989_98943

theorem speed_of_stream (vs : ℝ) (h : ∀ (d : ℝ), d / (57 - vs) = 2 * (d / (57 + vs))) : vs = 19 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l989_98943


namespace NUMINAMATH_GPT_cartesian_to_polar_curve_C_l989_98974

theorem cartesian_to_polar_curve_C (x y : ℝ) (θ ρ : ℝ) 
  (h1 : x = ρ * Real.cos θ)
  (h2 : y = ρ * Real.sin θ)
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * Real.cos θ :=
sorry

end NUMINAMATH_GPT_cartesian_to_polar_curve_C_l989_98974


namespace NUMINAMATH_GPT_total_cost_is_correct_l989_98910

def num_children : ℕ := 5
def daring_children : ℕ := 3
def ferris_wheel_cost_per_child : ℕ := 5
def merry_go_round_cost_per_child : ℕ := 3
def ice_cream_cones_per_child : ℕ := 2
def ice_cream_cost_per_cone : ℕ := 8

def total_spent_on_ferris_wheel : ℕ := daring_children * ferris_wheel_cost_per_child
def total_spent_on_merry_go_round : ℕ := num_children * merry_go_round_cost_per_child
def total_spent_on_ice_cream : ℕ := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone

def total_spent : ℕ := total_spent_on_ferris_wheel + total_spent_on_merry_go_round + total_spent_on_ice_cream

theorem total_cost_is_correct : total_spent = 110 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l989_98910


namespace NUMINAMATH_GPT_isosceles_triangle_k_value_l989_98977

theorem isosceles_triangle_k_value 
(side1 : ℝ)
(side2 side3 : ℝ)
(k : ℝ)
(h1 : side1 = 3 ∨ side2 = 3 ∨ side3 = 3)
(h2 : side1 = side2 ∨ side1 = side3 ∨ side2 = side3)
(h3 : Polynomial.eval side1 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side2 (Polynomial.C k + Polynomial.X ^ 2) = 0 
    ∨ Polynomial.eval side3 (Polynomial.C k + Polynomial.X ^ 2) = 0) :
k = 3 ∨ k = 4 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_k_value_l989_98977


namespace NUMINAMATH_GPT_jackson_souvenirs_total_l989_98959

def jacksons_collections := 
  let hermit_crabs := 120
  let spiral_shells_per_hermit_crab := 8
  let starfish_per_spiral_shell := 5
  let sand_dollars_per_starfish := 3
  let coral_structures_per_sand_dollars := 4
  let spiral_shells := hermit_crabs * spiral_shells_per_hermit_crab
  let starfish := spiral_shells * starfish_per_spiral_shell
  let sand_dollars := starfish * sand_dollars_per_starfish
  let coral_structures := sand_dollars / coral_structures_per_sand_dollars
  hermit_crabs + spiral_shells + starfish + sand_dollars + coral_structures

theorem jackson_souvenirs_total : jacksons_collections = 22880 := by sorry

end NUMINAMATH_GPT_jackson_souvenirs_total_l989_98959


namespace NUMINAMATH_GPT_range_of_a_l989_98972

open Real

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ , x^2 + a * x + 1 < 0) ↔ (-2 : ℝ) ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l989_98972


namespace NUMINAMATH_GPT_right_triangle_height_l989_98992

theorem right_triangle_height
  (h : ℕ)
  (base : ℕ)
  (rectangle_area : ℕ)
  (same_area : (1 / 2 : ℚ) * base * h = rectangle_area)
  (base_eq_width : base = 5)
  (rectangle_area_eq : rectangle_area = 45) :
  h = 18 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_height_l989_98992


namespace NUMINAMATH_GPT_x_minus_y_value_l989_98945

theorem x_minus_y_value (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 3) (h3 : x + y < 0) : x - y = 1 ∨ x - y = 5 := by
  sorry

end NUMINAMATH_GPT_x_minus_y_value_l989_98945


namespace NUMINAMATH_GPT_jerry_total_logs_l989_98927

-- Given conditions
def pine_logs_per_tree := 80
def maple_logs_per_tree := 60
def walnut_logs_per_tree := 100

def pine_trees_cut := 8
def maple_trees_cut := 3
def walnut_trees_cut := 4

-- Formulate the problem
theorem jerry_total_logs : 
  pine_logs_per_tree * pine_trees_cut + 
  maple_logs_per_tree * maple_trees_cut + 
  walnut_logs_per_tree * walnut_trees_cut = 1220 := 
by 
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_jerry_total_logs_l989_98927


namespace NUMINAMATH_GPT_function_passes_through_fixed_point_l989_98916

noncomputable def passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : Prop :=
  ∃ y : ℝ, y = a^(1-1) + 1 ∧ y = 2

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : passes_through_fixed_point a h :=
by
  sorry

end NUMINAMATH_GPT_function_passes_through_fixed_point_l989_98916


namespace NUMINAMATH_GPT_find_a_purely_imaginary_z1_z2_l989_98980

noncomputable def z1 (a : ℝ) : ℂ := ⟨a^2 - 3, a + 5⟩
noncomputable def z2 (a : ℝ) : ℂ := ⟨a - 1, a^2 + 2 * a - 1⟩

theorem find_a_purely_imaginary_z1_z2 (a : ℝ)
    (h_imaginary : ∃ b : ℝ, z2 a - z1 a = ⟨0, b⟩) : 
    a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_purely_imaginary_z1_z2_l989_98980


namespace NUMINAMATH_GPT_jebb_take_home_pay_l989_98939

-- We define the given conditions
def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

-- We define the function for the tax amount
def tax_amount (pay : ℝ) (rate : ℝ) : ℝ := pay * rate

-- We define the function for take-home pay
def take_home_pay (pay : ℝ) (rate : ℝ) : ℝ := pay - tax_amount pay rate

-- We state the theorem that needs to be proved
theorem jebb_take_home_pay : take_home_pay total_pay tax_rate = 585 := 
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_jebb_take_home_pay_l989_98939


namespace NUMINAMATH_GPT_transportable_load_l989_98935

theorem transportable_load 
  (mass_of_load : ℝ) 
  (num_boxes : ℕ) 
  (box_capacity : ℝ) 
  (num_trucks : ℕ) 
  (truck_capacity : ℝ) 
  (h1 : mass_of_load = 13.5) 
  (h2 : box_capacity = 0.35) 
  (h3 : truck_capacity = 1.5) 
  (h4 : num_trucks = 11)
  (boxes_condition : ∀ (n : ℕ), n * box_capacity ≥ mass_of_load) :
  mass_of_load ≤ num_trucks * truck_capacity :=
by
  sorry

end NUMINAMATH_GPT_transportable_load_l989_98935


namespace NUMINAMATH_GPT_greatest_multiple_of_5_and_6_less_than_1000_l989_98996

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n, (n % 5 = 0) ∧ (n % 6 = 0) ∧ (n < 1000) ∧ (∀ m, (m % 5 = 0) ∧ (m % 6 = 0) ∧ (m < 1000) → m ≤ n) ∧ n = 990 :=
by sorry

end NUMINAMATH_GPT_greatest_multiple_of_5_and_6_less_than_1000_l989_98996


namespace NUMINAMATH_GPT_different_total_scores_l989_98971

noncomputable def basket_scores (x y z : ℕ) : ℕ := x + 2 * y + 3 * z

def total_baskets := 7
def score_range := {n | 7 ≤ n ∧ n ≤ 21}

theorem different_total_scores : 
  ∃ (count : ℕ), count = 15 ∧ 
  ∀ n ∈ score_range, ∃ (x y z : ℕ), x + y + z = total_baskets ∧ basket_scores x y z = n :=
sorry

end NUMINAMATH_GPT_different_total_scores_l989_98971


namespace NUMINAMATH_GPT_positive_integers_solution_l989_98926

open Nat

theorem positive_integers_solution (a b m n : ℕ) (r : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
  (h_gcd : Nat.gcd m n = 1) :
  (a^2 + b^2)^m = (a * b)^n ↔ a = 2^r ∧ b = 2^r ∧ m = 2 * r ∧ n = 2 * r + 1 :=
sorry

end NUMINAMATH_GPT_positive_integers_solution_l989_98926


namespace NUMINAMATH_GPT_shaded_area_eq_63_l989_98942

noncomputable def rect1_width : ℕ := 4
noncomputable def rect1_height : ℕ := 12
noncomputable def rect2_width : ℕ := 5
noncomputable def rect2_height : ℕ := 7
noncomputable def overlap_width : ℕ := 4
noncomputable def overlap_height : ℕ := 5

theorem shaded_area_eq_63 :
  (rect1_width * rect1_height) + (rect2_width * rect2_height) - (overlap_width * overlap_height) = 63 := by
  sorry

end NUMINAMATH_GPT_shaded_area_eq_63_l989_98942


namespace NUMINAMATH_GPT_pencil_sharpening_time_l989_98938

theorem pencil_sharpening_time (t : ℕ) :
  let hand_crank_rate := 45
  let electric_rate := 20
  let sharpened_by_hand := (60 * t) / hand_crank_rate
  let sharpened_by_electric := (60 * t) / electric_rate
  (sharpened_by_electric = sharpened_by_hand + 10) → 
  t = 6 :=
by
  intros hand_crank_rate electric_rate sharpened_by_hand sharpened_by_electric h
  sorry

end NUMINAMATH_GPT_pencil_sharpening_time_l989_98938


namespace NUMINAMATH_GPT_daniel_candy_removal_l989_98925

theorem daniel_candy_removal (n k : ℕ) (h1 : n = 24) (h2 : k = 4) : ∃ m : ℕ, n % k = 0 → m = 0 :=
by
  sorry

end NUMINAMATH_GPT_daniel_candy_removal_l989_98925


namespace NUMINAMATH_GPT_tod_north_distance_l989_98932

-- Given conditions as variables
def speed : ℕ := 25  -- speed in miles per hour
def time : ℕ := 6    -- time in hours
def west_distance : ℕ := 95  -- distance to the west in miles

-- Prove the distance to the north given conditions
theorem tod_north_distance : time * speed - west_distance = 55 := by
  sorry

end NUMINAMATH_GPT_tod_north_distance_l989_98932


namespace NUMINAMATH_GPT_bacteria_original_count_l989_98956

theorem bacteria_original_count (current: ℕ) (increase: ℕ) (hc: current = 8917) (hi: increase = 8317) : current - increase = 600 :=
by
  sorry

end NUMINAMATH_GPT_bacteria_original_count_l989_98956


namespace NUMINAMATH_GPT_trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l989_98999

-- Part a
theorem trihedral_sum_of_angles_le_sum_of_plane_angles
  (α β γ : ℝ) (ASB BSC CSA : ℝ)
  (h1 : α ≤ ASB)
  (h2 : β ≤ BSC)
  (h3 : γ ≤ CSA) :
  α + β + γ ≤ ASB + BSC + CSA :=
sorry

-- Part b
theorem trihedral_sum_of_angles_ge_half_sum_of_plane_angles
  (α_S β_S γ_S : ℝ) (ASB BSC CSA : ℝ) 
  (h_acute : ASB < (π / 2) ∧ BSC < (π / 2) ∧ CSA < (π / 2))
  (h1 : α_S ≥ (1/2) * ASB)
  (h2 : β_S ≥ (1/2) * BSC)
  (h3 : γ_S ≥ (1/2) * CSA) :
  α_S + β_S + γ_S ≥ (1/2) * (ASB + BSC + CSA) :=
sorry

end NUMINAMATH_GPT_trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l989_98999


namespace NUMINAMATH_GPT_club_positions_l989_98958

def num_ways_to_fill_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)

theorem club_positions : num_ways_to_fill_positions 12 = 665280 := by 
  sorry

end NUMINAMATH_GPT_club_positions_l989_98958


namespace NUMINAMATH_GPT_find_divided_number_l989_98933

-- Declare the constants and assumptions
variables (d q r : ℕ)
variables (n : ℕ)
variables (h_d : d = 20)
variables (h_q : q = 6)
variables (h_r : r = 2)
variables (h_def : n = d * q + r)

-- State the theorem we want to prove
theorem find_divided_number : n = 122 :=
by
  sorry

end NUMINAMATH_GPT_find_divided_number_l989_98933


namespace NUMINAMATH_GPT_remainder_when_four_times_n_minus_9_l989_98912

theorem remainder_when_four_times_n_minus_9
  (n : ℤ) (h : n % 5 = 3) : (4 * n - 9) % 5 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_when_four_times_n_minus_9_l989_98912


namespace NUMINAMATH_GPT_shopkeeper_profit_percentage_l989_98954

theorem shopkeeper_profit_percentage (P : ℝ) : (70 / 100) * (1 + P / 100) = 1 → P = 700 / 3 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_profit_percentage_l989_98954


namespace NUMINAMATH_GPT_find_a1_l989_98924

theorem find_a1 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (1 - a n)) (h2 : a 8 = 2)
: a 1 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_a1_l989_98924


namespace NUMINAMATH_GPT_part1_part2_l989_98979

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) * Real.exp x

theorem part1 (a x : ℝ) (h : a > 0) : f a x + a / Real.exp 1 > 0 := by
  sorry

theorem part2 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f (-1/2) x1 = f (-1/2) x2) : x1 + x2 < 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l989_98979


namespace NUMINAMATH_GPT_trigonometric_identity_l989_98904

open Real

noncomputable def sin_alpha (x y : ℝ) : ℝ :=
  y / sqrt (x^2 + y^2)

noncomputable def tan_alpha (x y : ℝ) : ℝ :=
  y / x

theorem trigonometric_identity (x y : ℝ) (h_x : x = 3/5) (h_y : y = -4/5) :
  sin_alpha x y * tan_alpha x y = 16/15 :=
by {
  -- math proof to be provided here
  sorry
}

end NUMINAMATH_GPT_trigonometric_identity_l989_98904


namespace NUMINAMATH_GPT_simplify_expression_l989_98998

theorem simplify_expression (x : ℝ) : 
  (4 * x + 6 * x^3 + 8 - (3 - 6 * x^3 - 4 * x)) = 12 * x^3 + 8 * x + 5 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l989_98998


namespace NUMINAMATH_GPT_restaurant_bill_split_l989_98941

def original_bill : ℝ := 514.16
def tip_rate : ℝ := 0.18
def number_of_people : ℕ := 9
def final_amount_per_person : ℝ := 67.41

theorem restaurant_bill_split :
  final_amount_per_person = (1 + tip_rate) * original_bill / number_of_people :=
by
  sorry

end NUMINAMATH_GPT_restaurant_bill_split_l989_98941


namespace NUMINAMATH_GPT_ellipse_formula_max_area_triangle_l989_98976

-- Definitions for Ellipse part
def ellipse_eq (x y a : ℝ) := (x^2 / a^2) + (y^2 / 3) = 1
def eccentricity (a : ℝ) := (Real.sqrt (a^2 - 3)) / a = 1 / 2

-- Definition for Circle intersection part
def circle_intersection_cond (t : ℝ) := (0 < t) ∧ (t < (2 * Real.sqrt 21) / 7)

-- Main theorem for ellipse equation
theorem ellipse_formula (a : ℝ) (h1 : a > Real.sqrt 3) (h2 : eccentricity a) :
  ellipse_eq x y 2 :=
sorry

-- Main theorem for maximum area of triangle ABC
theorem max_area_triangle (t : ℝ) (h : circle_intersection_cond t) :
  ∃ S, S = (3 * Real.sqrt 7) / 7 :=
sorry

end NUMINAMATH_GPT_ellipse_formula_max_area_triangle_l989_98976


namespace NUMINAMATH_GPT_union_is_correct_l989_98921

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_is_correct : A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_GPT_union_is_correct_l989_98921


namespace NUMINAMATH_GPT_b6_b8_value_l989_98953

def arithmetic_seq (a : ℕ → ℕ) := ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d
def nonzero_sequence (a : ℕ → ℕ) := ∀ n : ℕ, a n ≠ 0
def geometric_seq (b : ℕ → ℕ) := ∃ r : ℕ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b6_b8_value (a b : ℕ → ℕ) (d : ℕ) 
  (h_arith : arithmetic_seq a) 
  (h_nonzero : nonzero_sequence a) 
  (h_cond1 : 2 * a 3 = a 1^2) 
  (h_cond2 : a 1 = d)
  (h_geo : geometric_seq b)
  (h_b13 : b 13 = a 2)
  (h_b1 : b 1 = a 1) :
  b 6 * b 8 = 72 := 
sorry

end NUMINAMATH_GPT_b6_b8_value_l989_98953


namespace NUMINAMATH_GPT_product_of_integers_between_sqrt_115_l989_98995

theorem product_of_integers_between_sqrt_115 :
  ∃ a b : ℕ, 100 < 115 ∧ 115 < 121 ∧ a = 10 ∧ b = 11 ∧ a * b = 110 := by
  sorry

end NUMINAMATH_GPT_product_of_integers_between_sqrt_115_l989_98995


namespace NUMINAMATH_GPT_multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l989_98993

def y : ℕ := 32 + 48 + 64 + 96 + 200 + 224 + 1600

theorem multiple_of_4 : y % 4 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_8 : y % 8 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_16 : y % 16 = 0 := by
  -- proof needed
  sorry

theorem not_multiple_of_32 : y % 32 ≠ 0 := by
  -- proof needed
  sorry

end NUMINAMATH_GPT_multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l989_98993


namespace NUMINAMATH_GPT_total_rainbow_nerds_l989_98905

-- Definitions based on the conditions
def num_purple_candies : ℕ := 10
def num_yellow_candies : ℕ := num_purple_candies + 4
def num_green_candies : ℕ := num_yellow_candies - 2

-- The statement to be proved
theorem total_rainbow_nerds : num_purple_candies + num_yellow_candies + num_green_candies = 36 := by
  -- Using the provided definitions to automatically infer
  sorry

end NUMINAMATH_GPT_total_rainbow_nerds_l989_98905


namespace NUMINAMATH_GPT_fraction_allocated_for_school_l989_98968

-- Conditions
def days_per_week : ℕ := 5
def hours_per_day : ℕ := 4
def earnings_per_hour : ℕ := 5
def allocation_for_school : ℕ := 75

-- Proof statement
theorem fraction_allocated_for_school :
  let weekly_hours := days_per_week * hours_per_day
  let weekly_earnings := weekly_hours * earnings_per_hour
  allocation_for_school / weekly_earnings = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_fraction_allocated_for_school_l989_98968


namespace NUMINAMATH_GPT_material_for_7_quilts_l989_98983

theorem material_for_7_quilts (x : ℕ) (h1 : ∀ y : ℕ, y = 7 * x) (h2 : 36 = 12 * x) : 7 * x = 21 := 
by 
  sorry

end NUMINAMATH_GPT_material_for_7_quilts_l989_98983


namespace NUMINAMATH_GPT_sum_of_solutions_l989_98900

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l989_98900


namespace NUMINAMATH_GPT_rectangular_box_in_sphere_radius_l989_98922

theorem rectangular_box_in_sphere_radius (a b c s : ℝ) 
  (h1 : a + b + c = 40) 
  (h2 : 2 * a * b + 2 * b * c + 2 * a * c = 608) 
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) : 
  s = 16 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_box_in_sphere_radius_l989_98922


namespace NUMINAMATH_GPT_minimum_additional_marbles_l989_98964

-- Definitions corresponding to the conditions
def friends := 12
def initial_marbles := 40

-- Sum of the first n natural numbers definition
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the necessary number of additional marbles
theorem minimum_additional_marbles (h1 : friends = 12) (h2 : initial_marbles = 40) : 
  ∃ additional_marbles, additional_marbles = sum_first_n friends - initial_marbles := by
  sorry

end NUMINAMATH_GPT_minimum_additional_marbles_l989_98964


namespace NUMINAMATH_GPT_correct_formula_for_xy_l989_98987

theorem correct_formula_for_xy :
  (∀ x y, (x = 1 ∧ y = 3) ∨ (x = 2 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 4 ∧ y = 21) ∨ (x = 5 ∧ y = 31) →
    y = x^2 + x + 1) :=
sorry

end NUMINAMATH_GPT_correct_formula_for_xy_l989_98987


namespace NUMINAMATH_GPT_find_m_l989_98951

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the function to calculate m * a + b
def m_a_plus_b (m : ℝ) : ℝ × ℝ := (2 * m - 1, 3 * m + 2)

-- Define the vector a - 2 * b
def a_minus_2b : ℝ × ℝ := (4, -1)

-- Define the condition for parallelism
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The theorem that states the equivalence
theorem find_m (m : ℝ) (H : parallel (m_a_plus_b m) a_minus_2b) : m = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l989_98951


namespace NUMINAMATH_GPT_correctly_subtracted_value_l989_98913

theorem correctly_subtracted_value (x : ℤ) (h1 : 122 = x - 64) : 
  x - 46 = 140 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_correctly_subtracted_value_l989_98913


namespace NUMINAMATH_GPT_total_surface_area_calc_l989_98975

/-- Given a cube with a total volume of 1 cubic foot, cut into four pieces by three parallel cuts:
1) The first cut is 0.4 feet from the top.
2) The second cut is 0.3 feet below the first.
3) The third cut is 0.1 feet below the second.
Prove that the total surface area of the new solid is 6 square feet. -/
theorem total_surface_area_calc :
  ∀ (A B C D : ℝ), 
    A = 0.4 → 
    B = 0.3 → 
    C = 0.1 → 
    D = 1 - (A + B + C) → 
    (6 : ℝ) = 6 := 
by 
  intros A B C D hA hB hC hD 
  sorry

end NUMINAMATH_GPT_total_surface_area_calc_l989_98975


namespace NUMINAMATH_GPT_sin_theta_value_l989_98929

theorem sin_theta_value (f : ℝ → ℝ)
  (hx : ∀ x, f x = 3 * Real.sin x - 8 * Real.cos (x / 2) ^ 2)
  (h_cond : ∀ x, f x ≤ f θ) : Real.sin θ = 3 / 5 := 
sorry

end NUMINAMATH_GPT_sin_theta_value_l989_98929


namespace NUMINAMATH_GPT_average_value_is_2020_l989_98962

namespace CardsAverage

theorem average_value_is_2020 (n : ℕ) (h : (2020 * 3 * ((n * (n + 1)) + 2) = n * (n + 1) * (2 * n + 1) + 6 * (n + 1))) : n = 3015 := 
by
  sorry

end CardsAverage

end NUMINAMATH_GPT_average_value_is_2020_l989_98962


namespace NUMINAMATH_GPT_determine_avery_height_l989_98937

-- Define Meghan's height
def meghan_height : ℕ := 188

-- Define range of players' heights
def height_range : ℕ := 33

-- Define the predicate to determine Avery's height
def avery_height : ℕ := meghan_height - height_range

-- The theorem we need to prove
theorem determine_avery_height : avery_height = 155 := by
  sorry

end NUMINAMATH_GPT_determine_avery_height_l989_98937


namespace NUMINAMATH_GPT_min_value_of_y_l989_98990

noncomputable def y (x : ℝ) : ℝ := x^2 + 26 * x + 7

theorem min_value_of_y : ∃ x : ℝ, y x = -162 :=
by
  use -13
  sorry

end NUMINAMATH_GPT_min_value_of_y_l989_98990


namespace NUMINAMATH_GPT_solve_for_x_l989_98930

theorem solve_for_x (x : ℝ) : 45 - 5 = 3 * x + 10 → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l989_98930


namespace NUMINAMATH_GPT_find_initial_volume_l989_98961

noncomputable def initial_volume_of_solution (V : ℝ) : Prop :=
  let initial_jasmine := 0.05 * V
  let added_jasmine := 8
  let added_water := 2
  let new_total_volume := V + added_jasmine + added_water
  let new_jasmine := 0.125 * new_total_volume
  initial_jasmine + added_jasmine = new_jasmine

theorem find_initial_volume : ∃ V : ℝ, initial_volume_of_solution V ∧ V = 90 :=
by
  use 90
  unfold initial_volume_of_solution
  sorry

end NUMINAMATH_GPT_find_initial_volume_l989_98961


namespace NUMINAMATH_GPT_solve_for_first_expedition_weeks_l989_98986

-- Define the variables according to the given conditions.
variables (x : ℕ)
variables (days_in_week : ℕ := 7)
variables (total_days_on_island : ℕ := 126)

-- Define the total number of weeks spent on the expeditions.
def total_weeks_on_expeditions (x : ℕ) : ℕ := 
  x + (x + 2) + 2 * (x + 2)

-- Convert total days to weeks.
def total_weeks := total_days_on_island / days_in_week

-- Prove the equation
theorem solve_for_first_expedition_weeks
  (h : total_weeks_on_expeditions x = total_weeks):
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_first_expedition_weeks_l989_98986


namespace NUMINAMATH_GPT_find_m_range_l989_98969

-- Definitions
def p (x : ℝ) : Prop := abs (2 * x + 1) ≤ 3
def q (x m : ℝ) (h : m > 0) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0

-- Problem Statement
theorem find_m_range : 
  (∀ (x : ℝ) (h : m > 0), (¬ (p x)) → (¬ (q x m h))) ∧ 
  (∃ (x : ℝ), ¬ (p x) ∧ ¬ (q x m h)) → 
  ∃ (m : ℝ), m ≥ 3 := 
sorry

end NUMINAMATH_GPT_find_m_range_l989_98969


namespace NUMINAMATH_GPT_area_of_path_correct_l989_98952

noncomputable def area_of_path (length_field : ℝ) (width_field : ℝ) (path_width : ℝ) : ℝ :=
  let length_total := length_field + 2 * path_width
  let width_total := width_field + 2 * path_width
  let area_total := length_total * width_total
  let area_field := length_field * width_field
  area_total - area_field

theorem area_of_path_correct :
  area_of_path 75 55 3.5 = 959 := 
by
  sorry

end NUMINAMATH_GPT_area_of_path_correct_l989_98952


namespace NUMINAMATH_GPT_Kolya_is_correct_Valya_is_incorrect_l989_98928

noncomputable def Kolya_probability (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

noncomputable def Valya_probability_not_losing (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

theorem Kolya_is_correct (x : ℝ) (hx : x > 0) : Kolya_probability x = 1 / 2 :=
by
  sorry

theorem Valya_is_incorrect (x : ℝ) (hx : x > 0) : Valya_probability_not_losing x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_Kolya_is_correct_Valya_is_incorrect_l989_98928


namespace NUMINAMATH_GPT_circles_positional_relationship_l989_98978

theorem circles_positional_relationship
  (r1 r2 d : ℝ)
  (h1 : r1 = 1)
  (h2 : r2 = 5)
  (h3 : d = 3) :
  d < r2 - r1 := 
by
  sorry

end NUMINAMATH_GPT_circles_positional_relationship_l989_98978


namespace NUMINAMATH_GPT_range_of_a_l989_98944

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → a < x ∧ x < 5) → a ≤ 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l989_98944


namespace NUMINAMATH_GPT_number_of_players_l989_98970

theorem number_of_players (n : ℕ) (G : ℕ) (h : G = 2 * n * (n - 1)) : n = 19 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_players_l989_98970


namespace NUMINAMATH_GPT_danil_claim_false_l989_98982

theorem danil_claim_false (E O : ℕ) (hE : E % 2 = 0) (hO : O % 2 = 0) (h : O = E + 15) : false :=
by sorry

end NUMINAMATH_GPT_danil_claim_false_l989_98982


namespace NUMINAMATH_GPT_intersection_subset_proper_l989_98920

-- Definitions of P and Q
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- The problem statement to prove
theorem intersection_subset_proper : P ∩ Q ⊂ P := by
  sorry

end NUMINAMATH_GPT_intersection_subset_proper_l989_98920


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l989_98948

theorem arithmetic_sequence_sum
  (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (H1 : a1 = -2017)
  (H2 : (S 2013 : ℤ) / 2013 - (S 2011 : ℤ) / 2011 = 2)
  (H3 : ∀ n : ℕ, S n = n * a1 + (n * (n - 1) / 2) * d) :
  S 2017 = -2017 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l989_98948


namespace NUMINAMATH_GPT_pencils_total_l989_98991

theorem pencils_total (original_pencils : ℕ) (added_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : original_pencils = 41) 
  (h2 : added_pencils = 30) 
  (h3 : total_pencils = original_pencils + added_pencils) : 
  total_pencils = 71 := 
by
  sorry

end NUMINAMATH_GPT_pencils_total_l989_98991


namespace NUMINAMATH_GPT_income_after_selling_more_l989_98967

theorem income_after_selling_more (x y : ℝ)
  (h1 : 26 * x + 14 * y = 264) 
  : 39 * x + 21 * y = 396 := 
by 
  sorry

end NUMINAMATH_GPT_income_after_selling_more_l989_98967


namespace NUMINAMATH_GPT_sequence_general_formula_l989_98911

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  ∀ n, a n = n^2 - n + 1 :=
by sorry

end NUMINAMATH_GPT_sequence_general_formula_l989_98911


namespace NUMINAMATH_GPT_class_gpa_l989_98923

theorem class_gpa (n : ℕ) (h_n : n = 60)
  (n1 : ℕ) (h_n1 : n1 = 20) (gpa1 : ℕ) (h_gpa1 : gpa1 = 15)
  (n2 : ℕ) (h_n2 : n2 = 15) (gpa2 : ℕ) (h_gpa2 : gpa2 = 17)
  (n3 : ℕ) (h_n3 : n3 = 25) (gpa3 : ℕ) (h_gpa3 : gpa3 = 19) :
  (20 * 15 + 15 * 17 + 25 * 19 : ℕ) / 60 = 1717 / 100 := 
sorry

end NUMINAMATH_GPT_class_gpa_l989_98923


namespace NUMINAMATH_GPT_motorcycle_time_l989_98984

theorem motorcycle_time (v_m v_b d t_m : ℝ) 
  (h1 : 12 * v_m + 9 * v_b = d)
  (h2 : 21 * v_b + 8 * v_m = d)
  (h3 : v_m = 3 * v_b) :
  t_m = 15 :=
by
  sorry

end NUMINAMATH_GPT_motorcycle_time_l989_98984


namespace NUMINAMATH_GPT_solution_xyz_uniqueness_l989_98906

theorem solution_xyz_uniqueness (x y z : ℝ) :
  x + y + z = 3 ∧ x^2 + y^2 + z^2 = 3 ∧ x^3 + y^3 + z^3 = 3 → x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_xyz_uniqueness_l989_98906


namespace NUMINAMATH_GPT_f_2014_odd_f_2014_not_even_l989_98960

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 1 / x
| (n + 1), x => 1 / (x + f n x)

theorem f_2014_odd :
  ∀ x : ℝ, f 2014 x = - f 2014 (-x) :=
sorry

theorem f_2014_not_even :
  ∃ x : ℝ, f 2014 x ≠ f 2014 (-x) :=
sorry

end NUMINAMATH_GPT_f_2014_odd_f_2014_not_even_l989_98960


namespace NUMINAMATH_GPT_greatest_integer_part_expected_winnings_l989_98985

noncomputable def expected_winnings_one_envelope : ℝ := 500

noncomputable def expected_winnings_two_envelopes : ℝ := 625

noncomputable def expected_winnings_three_envelopes : ℝ := 695.3125

theorem greatest_integer_part_expected_winnings :
  ⌊expected_winnings_three_envelopes⌋ = 695 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_integer_part_expected_winnings_l989_98985


namespace NUMINAMATH_GPT_pythagorean_triple_l989_98973

theorem pythagorean_triple {a b c : ℕ} (h : a * a + b * b = c * c) (gcd_abc : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ m n : ℕ, a = 2 * m * n ∧ b = m * m - n * n ∧ c = m * m + n * n :=
sorry

end NUMINAMATH_GPT_pythagorean_triple_l989_98973


namespace NUMINAMATH_GPT_solve_equation_l989_98909

theorem solve_equation (x : ℝ) : (x - 1) * (x + 3) = 5 ↔ x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l989_98909


namespace NUMINAMATH_GPT_div_binomial_expansion_l989_98981

theorem div_binomial_expansion
  (a n b : Nat)
  (hb : a^n ∣ b) :
  a^(n+1) ∣ (a+1)^b - 1 := by
  sorry

end NUMINAMATH_GPT_div_binomial_expansion_l989_98981


namespace NUMINAMATH_GPT_rectangle_sides_equal_perimeter_and_area_l989_98963

theorem rectangle_sides_equal_perimeter_and_area (x y : ℕ) (h : 2 * x + 2 * y = x * y) : 
    (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 4) :=
by sorry

end NUMINAMATH_GPT_rectangle_sides_equal_perimeter_and_area_l989_98963


namespace NUMINAMATH_GPT_unique_solution_l989_98907

def is_solution (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ) (hx : 0 < x), 
    ∃! (y : ℝ) (hy : 0 < y), 
      x * f y + y * f x ≤ 2

theorem unique_solution : ∀ (f : ℝ → ℝ), 
  is_solution f ↔ (∀ x, 0 < x → f x = 1 / x) :=
by
  intros
  sorry

end NUMINAMATH_GPT_unique_solution_l989_98907


namespace NUMINAMATH_GPT_probability_colored_ball_l989_98997

theorem probability_colored_ball (total_balls blue_balls green_balls white_balls : ℕ)
  (h_total : total_balls = 40)
  (h_blue : blue_balls = 15)
  (h_green : green_balls = 5)
  (h_white : white_balls = 20)
  (h_disjoint : total_balls = blue_balls + green_balls + white_balls) :
  (blue_balls + green_balls) / total_balls = 1 / 2 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_probability_colored_ball_l989_98997


namespace NUMINAMATH_GPT_jim_can_bake_loaves_l989_98966

-- Define the amounts of flour in different locations
def flour_cupboard : ℕ := 200  -- in grams
def flour_counter : ℕ := 100   -- in grams
def flour_pantry : ℕ := 100    -- in grams

-- Define the amount of flour required for one loaf of bread
def flour_per_loaf : ℕ := 200  -- in grams

-- Total loaves Jim can bake
def loaves_baked (f_c f_k f_p f_r : ℕ) : ℕ :=
  (f_c + f_k + f_p) / f_r

-- Theorem to prove the solution
theorem jim_can_bake_loaves :
  loaves_baked flour_cupboard flour_counter flour_pantry flour_per_loaf = 2 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_jim_can_bake_loaves_l989_98966


namespace NUMINAMATH_GPT_boards_cannot_be_covered_by_dominos_l989_98994

-- Definitions of the boards
def board_6x4 := (6 : ℕ) * (4 : ℕ)
def board_5x5 := (5 : ℕ) * (5 : ℕ)
def board_L_shaped := (5 : ℕ) * (5 : ℕ) - (2 : ℕ) * (2 : ℕ)
def board_3x7 := (3 : ℕ) * (7 : ℕ)
def board_plus_shaped := (3 : ℕ) * (3 : ℕ) + (1 : ℕ) * (3 : ℕ)

-- Definition to check if a board can't be covered by dominoes
def cannot_be_covered_by_dominos (n : ℕ) : Prop := n % 2 = 1

-- Theorem stating which specific boards cannot be covered by dominoes
theorem boards_cannot_be_covered_by_dominos :
  cannot_be_covered_by_dominos board_5x5 ∧
  cannot_be_covered_by_dominos board_L_shaped ∧
  cannot_be_covered_by_dominos board_3x7 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_boards_cannot_be_covered_by_dominos_l989_98994


namespace NUMINAMATH_GPT_gcd_79625_51575_l989_98919

theorem gcd_79625_51575 : Nat.gcd 79625 51575 = 25 :=
by
  sorry

end NUMINAMATH_GPT_gcd_79625_51575_l989_98919


namespace NUMINAMATH_GPT_joe_fruit_probability_l989_98989

theorem joe_fruit_probability :
  let prob_same := (1 / 4) ^ 3
  let total_prob_same := 4 * prob_same
  let prob_diff := 1 - total_prob_same
  prob_diff = 15 / 16 :=
by
  sorry

end NUMINAMATH_GPT_joe_fruit_probability_l989_98989


namespace NUMINAMATH_GPT_overall_percentage_gain_is_0_98_l989_98901

noncomputable def original_price : ℝ := 100
noncomputable def increased_price := original_price * 1.32
noncomputable def after_first_discount := increased_price * 0.90
noncomputable def final_price := after_first_discount * 0.85
noncomputable def overall_gain := final_price - original_price
noncomputable def overall_percentage_gain := (overall_gain / original_price) * 100

theorem overall_percentage_gain_is_0_98 :
  overall_percentage_gain = 0.98 := by
  sorry

end NUMINAMATH_GPT_overall_percentage_gain_is_0_98_l989_98901


namespace NUMINAMATH_GPT_solve_inequality_l989_98965

theorem solve_inequality : 
  {x : ℝ | -4 * x^2 + 7 * x + 2 < 0} = {x : ℝ | x < -1/4} ∪ {x : ℝ | 2 < x} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l989_98965


namespace NUMINAMATH_GPT_trigonometric_identity_l989_98936

theorem trigonometric_identity : 
  (Real.sin (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (138 * Real.pi / 180) * Real.cos (72 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l989_98936


namespace NUMINAMATH_GPT_programmer_debugging_hours_l989_98988

theorem programmer_debugging_hours 
  (total_hours : ℕ)
  (flow_chart_fraction coding_fraction : ℚ)
  (flow_chart_fraction_eq : flow_chart_fraction = 1/4)
  (coding_fraction_eq : coding_fraction = 3/8)
  (hours_worked : total_hours = 48) :
  ∃ (debugging_hours : ℚ), debugging_hours = 18 := 
by
  sorry

end NUMINAMATH_GPT_programmer_debugging_hours_l989_98988
