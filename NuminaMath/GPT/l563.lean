import Mathlib

namespace NUMINAMATH_GPT_combined_length_of_trains_l563_56316

def length_of_train (speed_kmhr : ℕ) (time_sec : ℕ) : ℚ :=
  (speed_kmhr : ℚ) / 3600 * time_sec

theorem combined_length_of_trains :
  let L1 := length_of_train 300 33
  let L2 := length_of_train 250 44
  let L3 := length_of_train 350 28
  L1 + L2 + L3 = 8.52741 := by
  sorry

end NUMINAMATH_GPT_combined_length_of_trains_l563_56316


namespace NUMINAMATH_GPT_parabola_distance_x_coord_l563_56364

theorem parabola_distance_x_coord
  (M : ℝ × ℝ) 
  (hM : M.2^2 = 4 * M.1)
  (hMF : (M.1 - 1)^2 + M.2^2 = 4^2)
  : M.1 = 3 :=
sorry

end NUMINAMATH_GPT_parabola_distance_x_coord_l563_56364


namespace NUMINAMATH_GPT_solution_set_of_inequality_l563_56375

theorem solution_set_of_inequality (x : ℝ) :
  (abs x * (x - 1) ≥ 0) ↔ (x ≥ 1 ∨ x = 0) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l563_56375


namespace NUMINAMATH_GPT_necessary_not_sufficient_l563_56348

theorem necessary_not_sufficient (a b c d : ℝ) : 
  (a + c > b + d) → (a > b ∧ c > d) :=
sorry

end NUMINAMATH_GPT_necessary_not_sufficient_l563_56348


namespace NUMINAMATH_GPT_original_weight_of_beef_l563_56398

variable (W : ℝ)

def first_stage_weight := 0.80 * W
def second_stage_weight := 0.70 * (first_stage_weight W)
def third_stage_weight := 0.75 * (second_stage_weight W)

theorem original_weight_of_beef :
  third_stage_weight W = 392 → W = 933.33 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_original_weight_of_beef_l563_56398


namespace NUMINAMATH_GPT_green_balls_count_l563_56345

theorem green_balls_count (b g : ℕ) (h1 : b = 15) (h2 : 5 * g = 3 * b) : g = 9 :=
by
  sorry

end NUMINAMATH_GPT_green_balls_count_l563_56345


namespace NUMINAMATH_GPT_square_properties_l563_56337

theorem square_properties (perimeter : ℝ) (h1 : perimeter = 40) :
  ∃ (side length area diagonal : ℝ), side = 10 ∧ length = 10 ∧ area = 100 ∧ diagonal = 10 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_square_properties_l563_56337


namespace NUMINAMATH_GPT_set_of_x_values_l563_56323

theorem set_of_x_values (x : ℝ) : (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := by
  sorry

end NUMINAMATH_GPT_set_of_x_values_l563_56323


namespace NUMINAMATH_GPT_james_carrot_sticks_left_l563_56308

variable (original_carrot_sticks : ℕ)
variable (eaten_before_dinner : ℕ)
variable (eaten_after_dinner : ℕ)
variable (given_away_during_dinner : ℕ)

theorem james_carrot_sticks_left 
  (h1 : original_carrot_sticks = 50)
  (h2 : eaten_before_dinner = 22)
  (h3 : eaten_after_dinner = 15)
  (h4 : given_away_during_dinner = 8) :
  original_carrot_sticks - eaten_before_dinner - eaten_after_dinner - given_away_during_dinner = 5 := 
sorry

end NUMINAMATH_GPT_james_carrot_sticks_left_l563_56308


namespace NUMINAMATH_GPT_solve_for_x_l563_56314

theorem solve_for_x : (3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5) = 800.0000000000001 → x = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l563_56314


namespace NUMINAMATH_GPT_greatest_possible_radius_of_circle_l563_56355

theorem greatest_possible_radius_of_circle
  (π : Real)
  (r : Real)
  (h : π * r^2 < 100 * π) :
  ∃ (n : ℕ), n = 9 ∧ (r : ℝ) ≤ 10 ∧ (r : ℝ) ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_radius_of_circle_l563_56355


namespace NUMINAMATH_GPT_distance_P_to_y_axis_l563_56362

-- Definition: Given point P in Cartesian coordinates
def P : ℝ × ℝ := (-3, -4)

-- Definition: Function to calculate distance to y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ := abs p.1

-- Theorem: The distance from point P to the y-axis is 3
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 :=
by
  sorry

end NUMINAMATH_GPT_distance_P_to_y_axis_l563_56362


namespace NUMINAMATH_GPT_humans_can_live_l563_56346

variable (earth_surface : ℝ)
variable (water_fraction : ℝ := 3 / 5)
variable (inhabitable_land_fraction : ℝ := 2 / 3)

def inhabitable_fraction : ℝ := (1 - water_fraction) * inhabitable_land_fraction

theorem humans_can_live :
  inhabitable_fraction = 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_humans_can_live_l563_56346


namespace NUMINAMATH_GPT_a1_b1_sum_l563_56320

-- Definitions from the conditions:
def strict_inc_seq (s : ℕ → ℕ) : Prop := ∀ n, s n < s (n + 1)

def positive_int_seq (s : ℕ → ℕ) : Prop := ∀ n, s n > 0

def a : ℕ → ℕ := sorry -- Define the sequence 'a' (details skipped).

def b : ℕ → ℕ := sorry -- Define the sequence 'b' (details skipped).

-- Conditions given:
axiom cond_a_inc : strict_inc_seq a

axiom cond_b_inc : strict_inc_seq b

axiom cond_a_pos : positive_int_seq a

axiom cond_b_pos : positive_int_seq b

axiom cond_a10_b10_lt_2017 : a 10 = b 10 ∧ a 10 < 2017

axiom cond_a_rec : ∀ n, a (n + 2) = a (n + 1) + a n

axiom cond_b_rec : ∀ n, b (n + 1) = 2 * b n

-- The theorem to prove:
theorem a1_b1_sum : a 1 + b 1 = 5 :=
sorry

end NUMINAMATH_GPT_a1_b1_sum_l563_56320


namespace NUMINAMATH_GPT_pastries_selection_l563_56347

/--
Clara wants to purchase six pastries from an ample supply of five types: muffins, eclairs, croissants, scones, and turnovers. 
Prove that there are 210 possible selections using the stars and bars theorem.
-/
theorem pastries_selection : ∃ (selections : ℕ), selections = (Nat.choose (6 + 5 - 1) (5 - 1)) ∧ selections = 210 := by
  sorry

end NUMINAMATH_GPT_pastries_selection_l563_56347


namespace NUMINAMATH_GPT_inequality_minus_x_plus_3_l563_56315

variable (x y : ℝ)

theorem inequality_minus_x_plus_3 (h : x < y) : -x + 3 > -y + 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_minus_x_plus_3_l563_56315


namespace NUMINAMATH_GPT_domain_f₁_range_f₂_l563_56329

noncomputable def f₁ (x : ℝ) : ℝ := (x - 2)^0 / Real.sqrt (x + 1)
noncomputable def f₂ (x : ℝ) : ℝ := 2 * x - Real.sqrt (x - 1)

theorem domain_f₁ : ∀ x : ℝ, x > -1 ∧ x ≠ 2 → ∃ y : ℝ, y = f₁ x :=
by
  sorry

theorem range_f₂ : ∀ y : ℝ, y ≥ 15 / 8 → ∃ x : ℝ, y = f₂ x :=
by
  sorry

end NUMINAMATH_GPT_domain_f₁_range_f₂_l563_56329


namespace NUMINAMATH_GPT_simplify_expression_l563_56374

variable (x : ℝ)

theorem simplify_expression : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3 - 3 * x^3) 
  = (-x^3 - x^2 + 23 * x - 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l563_56374


namespace NUMINAMATH_GPT_arithmetic_evaluation_l563_56341

theorem arithmetic_evaluation : (64 / 0.08) - 2.5 = 797.5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_evaluation_l563_56341


namespace NUMINAMATH_GPT_smallest_n_divisibility_l563_56318

theorem smallest_n_divisibility :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (m > 0) ∧ (72 ∣ m^2) ∧ (1728 ∣ m^3) → (n ≤ m)) ∧
  (72 ∣ 12^2) ∧ (1728 ∣ 12^3) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisibility_l563_56318


namespace NUMINAMATH_GPT_printer_time_equation_l563_56359

theorem printer_time_equation (x : ℝ) (rate1 rate2 : ℝ) (flyers1 flyers2 : ℝ)
  (h1 : rate1 = 100) (h2 : flyers1 = 1000) (h3 : flyers2 = 1000) 
  (h4 : flyers1 / rate1 = 10) (h5 : flyers1 / (rate1 + rate2) = 4) : 
  1 / 10 + 1 / x = 1 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_printer_time_equation_l563_56359


namespace NUMINAMATH_GPT_donna_pizza_slices_l563_56376

theorem donna_pizza_slices :
  ∀ (total_slices : ℕ) (half_eaten_for_lunch : ℕ) (one_third_eaten_for_dinner : ℕ),
  total_slices = 12 →
  half_eaten_for_lunch = total_slices / 2 →
  one_third_eaten_for_dinner = half_eaten_for_lunch / 3 →
  (half_eaten_for_lunch - one_third_eaten_for_dinner) = 4 :=
by
  intros total_slices half_eaten_for_lunch one_third_eaten_for_dinner
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_donna_pizza_slices_l563_56376


namespace NUMINAMATH_GPT_evaluate_expression_l563_56353

noncomputable def expr : ℚ := (3 ^ 512 + 7 ^ 513) ^ 2 - (3 ^ 512 - 7 ^ 513) ^ 2
noncomputable def k : ℚ := 28 * 2.1 ^ 512

theorem evaluate_expression : expr = k * 10 ^ 513 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l563_56353


namespace NUMINAMATH_GPT_smallest_solution_x4_minus_50x2_plus_625_eq_0_l563_56317

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end NUMINAMATH_GPT_smallest_solution_x4_minus_50x2_plus_625_eq_0_l563_56317


namespace NUMINAMATH_GPT_sum_of_ages_l563_56367

def Tyler_age : ℕ := 5

def Clay_age (T C : ℕ) : Prop :=
  T = 3 * C + 1

theorem sum_of_ages (C : ℕ) (h : Clay_age Tyler_age C) :
  Tyler_age + C = 6 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l563_56367


namespace NUMINAMATH_GPT_nathan_blankets_l563_56385

theorem nathan_blankets (b : ℕ) (hb : 21 = (b / 2) * 3) : b = 14 :=
by sorry

end NUMINAMATH_GPT_nathan_blankets_l563_56385


namespace NUMINAMATH_GPT_find_f_neg2_l563_56372

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a * f b
axiom f_pos (x : ℝ) : f x > 0
axiom f_one : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_f_neg2_l563_56372


namespace NUMINAMATH_GPT_percentage_decrease_l563_56389

-- Define the initial conditions
def total_cans : ℕ := 600
def initial_people : ℕ := 40
def new_total_cans : ℕ := 420

-- Use the conditions to define the resulting quantities
def cans_per_person : ℕ := total_cans / initial_people
def new_people : ℕ := new_total_cans / cans_per_person

-- Prove the percentage decrease in the number of people
theorem percentage_decrease :
  let original_people := initial_people
  let new_people := new_people
  let decrease := original_people - new_people
  let percentage_decrease := (decrease * 100) / original_people
  percentage_decrease = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l563_56389


namespace NUMINAMATH_GPT_square_side_factor_l563_56366

theorem square_side_factor (k : ℝ) (h : k^2 = 1) : k = 1 :=
sorry

end NUMINAMATH_GPT_square_side_factor_l563_56366


namespace NUMINAMATH_GPT_parallel_lines_slope_l563_56383

-- Define the equations of the lines in Lean
def line1 (x : ℝ) : ℝ := 7 * x + 3
def line2 (c : ℝ) (x : ℝ) : ℝ := (3 * c) * x + 5

-- State the theorem: if the lines are parallel, then c = 7/3
theorem parallel_lines_slope (c : ℝ) :
  (∀ x : ℝ, (7 * x + 3 = (3 * c) * x + 5)) → c = (7/3) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l563_56383


namespace NUMINAMATH_GPT_intersection_P_Q_l563_56333

open Set

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | abs x < 2}

theorem intersection_P_Q : P ∩ Q = {1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l563_56333


namespace NUMINAMATH_GPT_maggi_initial_packages_l563_56335

theorem maggi_initial_packages (P : ℕ) (h1 : 4 * P - 5 = 12) : P = 4 :=
sorry

end NUMINAMATH_GPT_maggi_initial_packages_l563_56335


namespace NUMINAMATH_GPT_find_natural_pairs_l563_56309

theorem find_natural_pairs (a b : ℕ) :
  (∃ A, A * A = a ^ 2 + 3 * b) ∧ (∃ B, B * B = b ^ 2 + 3 * a) ↔ 
  (a = 1 ∧ b = 1) ∨ (a = 11 ∧ b = 11) ∨ (a = 16 ∧ b = 11) :=
by
  sorry

end NUMINAMATH_GPT_find_natural_pairs_l563_56309


namespace NUMINAMATH_GPT_problem_l563_56371

open Set

-- Definitions for set A and set B
def setA : Set ℝ := { x | x^2 + 2 * x - 3 < 0 }
def setB : Set ℤ := { k : ℤ | true }
def evenIntegers : Set ℝ := { x : ℝ | ∃ k : ℤ, x = 2 * k }

-- The intersection of set A and even integers over ℝ
def A_cap_B : Set ℝ := setA ∩ evenIntegers

-- The Proposition that A_cap_B equals {-2, 0}
theorem problem : A_cap_B = ({-2, 0} : Set ℝ) :=
by 
  sorry

end NUMINAMATH_GPT_problem_l563_56371


namespace NUMINAMATH_GPT_common_remainder_zero_l563_56365

theorem common_remainder_zero (n r : ℕ) (h1: n > 1) 
(h2 : n % 25 = r) (h3 : n % 7 = r) (h4 : n = 175) : r = 0 :=
by
  sorry

end NUMINAMATH_GPT_common_remainder_zero_l563_56365


namespace NUMINAMATH_GPT_total_games_in_season_l563_56340

theorem total_games_in_season :
  let num_teams := 100
  let num_sub_leagues := 5
  let teams_per_league := 20
  let games_per_pair := 6
  let teams_advancing := 4
  let playoff_teams := num_sub_leagues * teams_advancing
  let sub_league_games := (teams_per_league * (teams_per_league - 1) / 2) * games_per_pair
  let total_sub_league_games := sub_league_games * num_sub_leagues
  let playoff_games := (playoff_teams * (playoff_teams - 1)) / 2 
  let total_games := total_sub_league_games + playoff_games
  total_games = 5890 :=
by
  sorry

end NUMINAMATH_GPT_total_games_in_season_l563_56340


namespace NUMINAMATH_GPT_min_ratio_number_l563_56380

theorem min_ratio_number (H T U : ℕ) (h1 : H - T = 8 ∨ T - H = 8) (hH : 1 ≤ H ∧ H ≤ 9) (hT : 0 ≤ T ∧ T ≤ 9) (hU : 0 ≤ U ∧ U ≤ 9) :
  100 * H + 10 * T + U = 190 :=
by sorry

end NUMINAMATH_GPT_min_ratio_number_l563_56380


namespace NUMINAMATH_GPT_sum_positive_132_l563_56392

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem sum_positive_132 {a: ℕ → ℝ}
  (h1: a 66 < 0)
  (h2: a 67 > 0)
  (h3: a 67 > |a 66|):
  ∃ n, ∀ k < n, S k > 0 :=
by
  have h4 : (a 67 - a 66) > 0 := sorry
  have h5 : a 67 + a 66 > 0 := sorry
  have h6 : 66 * (a 67 + a 66) > 0 := sorry
  have h7 : S 132 = 66 * (a 67 + a 66) := sorry
  existsi 132
  intro k hk
  sorry

end NUMINAMATH_GPT_sum_positive_132_l563_56392


namespace NUMINAMATH_GPT_diophantine_eq_solutions_l563_56391

theorem diophantine_eq_solutions (p q r k : ℕ) (hp : p > 1) (hq : q > 1) (hr : r > 1) 
  (hp_prime : Prime p) (hq_prime : Prime q) (hr_prime : Prime r) (hk : k > 0) :
  p^2 + q^2 + 49*r^2 = 9*k^2 - 101 ↔ 
  (p = 3 ∧ q = 5 ∧ r = 3 ∧ k = 8) ∨ (p = 5 ∧ q = 3 ∧ r = 3 ∧ k = 8) :=
by sorry

end NUMINAMATH_GPT_diophantine_eq_solutions_l563_56391


namespace NUMINAMATH_GPT_minimum_value_ineq_l563_56302

open Real

theorem minimum_value_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
    (x + 1 / (y * y)) * (x + 1 / (y * y) - 500) + (y + 1 / (x * x)) * (y + 1 / (x * x) - 500) ≥ -125000 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_ineq_l563_56302


namespace NUMINAMATH_GPT_brown_shoes_count_l563_56306

-- Definitions based on given conditions
def total_shoes := 66
def black_shoe_ratio := 2

theorem brown_shoes_count (B : ℕ) (H1 : black_shoe_ratio * B + B = total_shoes) : B = 22 :=
by
  -- Proof here is replaced with sorry for the purpose of this exercise
  sorry

end NUMINAMATH_GPT_brown_shoes_count_l563_56306


namespace NUMINAMATH_GPT_find_value_am2_bm_minus_7_l563_56336

variable {a b m : ℝ}

theorem find_value_am2_bm_minus_7
  (h : a * m^2 + b * m + 5 = 0) : a * m^2 + b * m - 7 = -12 :=
by
  sorry

end NUMINAMATH_GPT_find_value_am2_bm_minus_7_l563_56336


namespace NUMINAMATH_GPT_pascal_triangle_45th_number_l563_56358

theorem pascal_triangle_45th_number (n k : ℕ) (h1 : n = 47) (h2 : k = 44) : 
  Nat.choose (n - 1) k = 1035 :=
by
  sorry

end NUMINAMATH_GPT_pascal_triangle_45th_number_l563_56358


namespace NUMINAMATH_GPT_dragons_total_games_played_l563_56313

theorem dragons_total_games_played (y x : ℕ)
  (h1 : x = 55 * y / 100)
  (h2 : x + 8 = 60 * (y + 12) / 100) :
  y + 12 = 28 :=
by
  sorry

end NUMINAMATH_GPT_dragons_total_games_played_l563_56313


namespace NUMINAMATH_GPT_solve_for_x_l563_56310

theorem solve_for_x : ∀ x : ℝ, (x - 5) ^ 3 = (1 / 27)⁻¹ → x = 8 := by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l563_56310


namespace NUMINAMATH_GPT_marbles_count_l563_56388

theorem marbles_count (red green blue total : ℕ) (h_red : red = 38)
  (h_green : green = red / 2) (h_total : total = 63) 
  (h_sum : total = red + green + blue) : blue = 6 :=
by
  sorry

end NUMINAMATH_GPT_marbles_count_l563_56388


namespace NUMINAMATH_GPT_ratio_ravi_kiran_l563_56352

-- Definitions for the conditions
def ratio_money_ravi_giri := 6 / 7
def money_ravi := 36
def money_kiran := 105

-- The proof problem
theorem ratio_ravi_kiran : (money_ravi : ℕ) / money_kiran = 12 / 35 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_ravi_kiran_l563_56352


namespace NUMINAMATH_GPT_geometric_series_has_value_a_l563_56370

theorem geometric_series_has_value_a (a : ℝ) (S : ℕ → ℝ)
  (h : ∀ n, S (n + 1) = a * (1 / 4) ^ n + 6) :
  a = -3 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_series_has_value_a_l563_56370


namespace NUMINAMATH_GPT_simplify_and_rationalize_denominator_l563_56325

theorem simplify_and_rationalize_denominator :
  ( (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 14) = 3 * Real.sqrt 420 / 42 ) := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_and_rationalize_denominator_l563_56325


namespace NUMINAMATH_GPT_find_a_l563_56381

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + x

-- Define the derivative of the function f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x + 1

-- The main theorem: if the tangent at x = 1 is parallel to the line y = 2x, then a = 1
theorem find_a (a : ℝ) : f' 1 a = 2 → a = 1 :=
by
  intro h
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_find_a_l563_56381


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l563_56312

section DecreasingNumber

def is_decreasing_number (a b c d : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  10 * a + b - (10 * b + c) = 10 * c + d

theorem problem_part1 (a : ℕ) :
  is_decreasing_number a 3 1 2 → a = 4 :=
by
  intro h
  -- Proof steps
  sorry

theorem problem_part2 (a b c d : ℕ) :
  is_decreasing_number a b c d →
  (100 * a + 10 * b + c + 100 * b + 10 * c + d) % 9 = 0 →
  8165 = max_value :=
by
  intro h1 h2
  -- Proof steps
  sorry

end DecreasingNumber

end NUMINAMATH_GPT_problem_part1_problem_part2_l563_56312


namespace NUMINAMATH_GPT_minimum_sum_of_dimensions_l563_56350

-- Define the problem as a Lean 4 statement
theorem minimum_sum_of_dimensions (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 2184) : 
  x + y + z = 36 := 
sorry

end NUMINAMATH_GPT_minimum_sum_of_dimensions_l563_56350


namespace NUMINAMATH_GPT_total_cost_is_correct_l563_56357

-- Define the conditions
def piano_cost : ℝ := 500
def lesson_cost_per_lesson : ℝ := 40
def number_of_lessons : ℝ := 20
def discount_rate : ℝ := 0.25

-- Define the total cost of lessons before discount
def total_lesson_cost_before_discount : ℝ := lesson_cost_per_lesson * number_of_lessons

-- Define the discount amount
def discount_amount : ℝ := discount_rate * total_lesson_cost_before_discount

-- Define the total cost of lessons after discount
def total_lesson_cost_after_discount : ℝ := total_lesson_cost_before_discount - discount_amount

-- Define the total cost of everything
def total_cost : ℝ := piano_cost + total_lesson_cost_after_discount

-- The statement to be proven
theorem total_cost_is_correct : total_cost = 1100 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l563_56357


namespace NUMINAMATH_GPT_vector_dot_product_l563_56349

variables (a b : ℝ × ℝ)
variables (ha : a = (1, -1)) (hb : b = (-1, 2))

theorem vector_dot_product : 
  ((2 • a + b) • a) = -1 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_vector_dot_product_l563_56349


namespace NUMINAMATH_GPT_isosceles_triangle_problem_l563_56399

theorem isosceles_triangle_problem 
  (a h b : ℝ) 
  (area_relation : (1/2) * a * h = (1/3) * a ^ 2) 
  (leg_relation : b = a - 1)
  (height_relation : h = (2/3) * a) 
  (pythagorean_theorem : h ^ 2 + (a / 2) ^ 2 = b ^ 2) : 
  a = 6 ∧ b = 5 ∧ h = 4 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_problem_l563_56399


namespace NUMINAMATH_GPT_hotel_charge_comparison_l563_56393

def charge_R (R G : ℝ) (P : ℝ) : Prop :=
  P = 0.8 * R ∧ P = 0.9 * G

def discounted_charge_R (R2 : ℝ) (R : ℝ) : Prop :=
  R2 = 0.85 * R

theorem hotel_charge_comparison (R G P R2 : ℝ)
  (h1 : charge_R R G P)
  (h2 : discounted_charge_R R2 R)
  (h3 : R = 1.125 * G) :
  R2 = 0.95625 * G := by
  sorry

end NUMINAMATH_GPT_hotel_charge_comparison_l563_56393


namespace NUMINAMATH_GPT_odd_square_minus_one_multiple_of_eight_l563_56378

theorem odd_square_minus_one_multiple_of_eight (a : ℤ) 
  (h₁ : a > 0) 
  (h₂ : a % 2 = 1) : 
  ∃ k : ℤ, a^2 - 1 = 8 * k :=
by
  sorry

end NUMINAMATH_GPT_odd_square_minus_one_multiple_of_eight_l563_56378


namespace NUMINAMATH_GPT_unique_quadruple_exists_l563_56342

theorem unique_quadruple_exists :
  ∃! (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
  a + b + c + d = 2 ∧
  a^2 + b^2 + c^2 + d^2 = 3 ∧
  (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 18 := by
  sorry

end NUMINAMATH_GPT_unique_quadruple_exists_l563_56342


namespace NUMINAMATH_GPT_dingding_minimum_correct_answers_l563_56305

theorem dingding_minimum_correct_answers (x : ℕ) :
  (5 * x - (30 - x) > 100) → x ≥ 22 :=
by
  sorry

end NUMINAMATH_GPT_dingding_minimum_correct_answers_l563_56305


namespace NUMINAMATH_GPT_oshea_bought_basil_seeds_l563_56339

-- Define the number of large and small planters and their capacities.
def large_planters := 4
def seeds_per_large_planter := 20
def small_planters := 30
def seeds_per_small_planter := 4

-- The theorem statement: Oshea bought 200 basil seeds
theorem oshea_bought_basil_seeds :
  large_planters * seeds_per_large_planter + small_planters * seeds_per_small_planter = 200 :=
by sorry

end NUMINAMATH_GPT_oshea_bought_basil_seeds_l563_56339


namespace NUMINAMATH_GPT_letters_with_line_not_dot_l563_56382

-- Defining the conditions
def num_letters_with_dot_and_line : ℕ := 9
def num_letters_with_dot_only : ℕ := 7
def total_letters : ℕ := 40

-- Proving the number of letters with a straight line but not a dot
theorem letters_with_line_not_dot :
  (num_letters_with_dot_and_line + num_letters_with_dot_only + x = total_letters) → x = 24 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_letters_with_line_not_dot_l563_56382


namespace NUMINAMATH_GPT_number_added_l563_56300

def initial_number : ℕ := 9
def final_resultant : ℕ := 93

theorem number_added : ∃ x : ℕ, 3 * (2 * initial_number + x) = final_resultant ∧ x = 13 := by
  sorry

end NUMINAMATH_GPT_number_added_l563_56300


namespace NUMINAMATH_GPT_trigonometric_identity_tan_two_l563_56379

theorem trigonometric_identity_tan_two (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 11 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_tan_two_l563_56379


namespace NUMINAMATH_GPT_diameter_of_circumscribed_circle_l563_56386

noncomputable def right_triangle_circumcircle_diameter (a b : ℕ) : ℕ :=
  let hypotenuse := (a * a + b * b).sqrt
  if hypotenuse = max a b then hypotenuse else 2 * max a b

theorem diameter_of_circumscribed_circle
  (a b : ℕ)
  (h : a = 16 ∨ b = 16)
  (h1 : a = 12 ∨ b = 12) :
  right_triangle_circumcircle_diameter a b = 16 ∨ right_triangle_circumcircle_diameter a b = 20 :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_diameter_of_circumscribed_circle_l563_56386


namespace NUMINAMATH_GPT_smallest_value_of_x_l563_56322

theorem smallest_value_of_x :
  ∃ x : ℝ, (x / 4 + 2 / (3 * x) = 5 / 6) ∧ (∀ y : ℝ,
    (y / 4 + 2 / (3 * y) = 5 / 6) → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_value_of_x_l563_56322


namespace NUMINAMATH_GPT_number_of_people_in_group_l563_56327

theorem number_of_people_in_group 
    (N : ℕ)
    (old_person_weight : ℕ) (new_person_weight : ℕ)
    (average_weight_increase : ℕ) :
    old_person_weight = 70 →
    new_person_weight = 94 →
    average_weight_increase = 3 →
    N * average_weight_increase = new_person_weight - old_person_weight →
    N = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_in_group_l563_56327


namespace NUMINAMATH_GPT_correct_quotient_divide_8_l563_56351

theorem correct_quotient_divide_8 (N : ℕ) (Q : ℕ) 
  (h1 : N = 7 * 12 + 5) 
  (h2 : N / 8 = Q) : 
  Q = 11 := 
by
  sorry

end NUMINAMATH_GPT_correct_quotient_divide_8_l563_56351


namespace NUMINAMATH_GPT_pencils_in_drawer_after_operations_l563_56331

def initial_pencils : ℝ := 2
def pencils_added : ℝ := 3.5
def pencils_removed : ℝ := 1.2

theorem pencils_in_drawer_after_operations : ⌊initial_pencils + pencils_added - pencils_removed⌋ = 4 := by
  sorry

end NUMINAMATH_GPT_pencils_in_drawer_after_operations_l563_56331


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l563_56343

-- Define the conditions
def side_of_square : ℕ := 35
def area_of_square : ℕ := 1225
def radius_of_sphere : ℕ := side_of_square
def length_of_prism : ℕ := (2 * radius_of_sphere) / 5
def width_of_prism : ℕ := 10
variable (h : ℕ) -- height of the prism

-- The theorem to prove
theorem volume_of_rectangular_prism :
  area_of_square = side_of_square * side_of_square →
  length_of_prism = (2 * radius_of_sphere) / 5 →
  radius_of_sphere = side_of_square →
  volume_of_prism = (length_of_prism * width_of_prism * h)
  → volume_of_prism = 140 * h :=
by sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l563_56343


namespace NUMINAMATH_GPT_option_d_not_necessarily_true_l563_56361

theorem option_d_not_necessarily_true (a b c : ℝ) (h: a > b) : ¬(a * c^2 > b * c^2) ↔ c = 0 :=
by sorry

end NUMINAMATH_GPT_option_d_not_necessarily_true_l563_56361


namespace NUMINAMATH_GPT_min_dot_product_value_l563_56330

noncomputable def dot_product_minimum (x : ℝ) : ℝ :=
  8 * x^2 + 4 * x

theorem min_dot_product_value :
  (∀ x, dot_product_minimum x ≥ -1 / 2) ∧ (∃ x, dot_product_minimum x = -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_min_dot_product_value_l563_56330


namespace NUMINAMATH_GPT_number_of_three_digit_numbers_divisible_by_17_l563_56301

theorem number_of_three_digit_numbers_divisible_by_17 : 
  let k_min := Nat.ceil (100 / 17)
  let k_max := Nat.floor (999 / 17)
  ∃ n, 
    (n = k_max - k_min + 1) ∧ 
    (n = 53) := 
by
    sorry

end NUMINAMATH_GPT_number_of_three_digit_numbers_divisible_by_17_l563_56301


namespace NUMINAMATH_GPT_scientific_notation_of_604800_l563_56338

theorem scientific_notation_of_604800 : 604800 = 6.048 * 10^5 := 
sorry

end NUMINAMATH_GPT_scientific_notation_of_604800_l563_56338


namespace NUMINAMATH_GPT_clara_meeting_time_l563_56304

theorem clara_meeting_time (d T : ℝ) :
  (d / 20 = T - 0.5) →
  (d / 12 = T + 0.5) →
  (d / T = 15) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_clara_meeting_time_l563_56304


namespace NUMINAMATH_GPT_number_of_female_students_l563_56311

theorem number_of_female_students (T S f_sample : ℕ) (H_total : T = 1600) (H_sample_size : S = 200) (H_females_in_sample : f_sample = 95) : 
  ∃ F, 95 / 200 = F / 1600 ∧ F = 760 := by 
sorry

end NUMINAMATH_GPT_number_of_female_students_l563_56311


namespace NUMINAMATH_GPT_passengers_on_bus_l563_56344

theorem passengers_on_bus (initial_passengers : ℕ) (got_on : ℕ) (got_off : ℕ) (final_passengers : ℕ) :
  initial_passengers = 28 → got_on = 7 → got_off = 9 → final_passengers = initial_passengers + got_on - got_off → final_passengers = 26 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_passengers_on_bus_l563_56344


namespace NUMINAMATH_GPT_percentage_reduction_is_20_l563_56354

def original_employees : ℝ := 243.75
def reduced_employees : ℝ := 195

theorem percentage_reduction_is_20 :
  (original_employees - reduced_employees) / original_employees * 100 = 20 := 
  sorry

end NUMINAMATH_GPT_percentage_reduction_is_20_l563_56354


namespace NUMINAMATH_GPT_find_m_l563_56394

theorem find_m (m : ℝ) (h1 : |m - 3| = 4) (h2 : m - 7 ≠ 0) : m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_l563_56394


namespace NUMINAMATH_GPT_cars_more_than_trucks_l563_56356

theorem cars_more_than_trucks (total_vehicles : ℕ) (trucks : ℕ) (h : total_vehicles = 69) (h' : trucks = 21) :
  (total_vehicles - trucks) - trucks = 27 :=
by
  sorry

end NUMINAMATH_GPT_cars_more_than_trucks_l563_56356


namespace NUMINAMATH_GPT_total_cost_is_eight_times_l563_56360

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_eight_times_l563_56360


namespace NUMINAMATH_GPT_total_wages_l563_56390

-- Definitions and conditions
def A_one_day_work : ℚ := 1 / 10
def B_one_day_work : ℚ := 1 / 15
def A_share_wages : ℚ := 2040

-- Stating the problem in Lean
theorem total_wages (X : ℚ) : (3 / 5) * X = A_share_wages → X = 3400 := 
  by 
  sorry

end NUMINAMATH_GPT_total_wages_l563_56390


namespace NUMINAMATH_GPT_tg_sum_equal_l563_56303

variable {a b c : ℝ}
variable {φA φB φC : ℝ}

-- The sides of the triangle are labeled such that a >= b >= c.
axiom sides_ineq : a ≥ b ∧ b ≥ c

-- The angles between the median and the altitude from vertices A, B, and C.
axiom angles_def : true -- This axiom is a placeholder. In actual use, we would define φA, φB, φC properly using the given geometric setup.

theorem tg_sum_equal : Real.tan φA + Real.tan φC = Real.tan φB := 
by 
  sorry

end NUMINAMATH_GPT_tg_sum_equal_l563_56303


namespace NUMINAMATH_GPT_first_sculpture_weight_is_five_l563_56324

variable (w x y z : ℝ)

def hourly_wage_exterminator := 70
def daily_hours := 20
def price_per_pound := 20
def second_sculpture_weight := 7
def total_income := 1640

def income_exterminator := daily_hours * hourly_wage_exterminator
def income_sculptures := total_income - income_exterminator
def income_second_sculpture := second_sculpture_weight * price_per_pound
def income_first_sculpture := income_sculptures - income_second_sculpture

def weight_first_sculpture := income_first_sculpture / price_per_pound

theorem first_sculpture_weight_is_five :
  weight_first_sculpture = 5 := sorry

end NUMINAMATH_GPT_first_sculpture_weight_is_five_l563_56324


namespace NUMINAMATH_GPT_largest_is_A_minus_B_l563_56321

noncomputable def A := 3 * 1005^1006
noncomputable def B := 1005^1006
noncomputable def C := 1004 * 1005^1005
noncomputable def D := 3 * 1005^1005
noncomputable def E := 1005^1005
noncomputable def F := 1005^1004

theorem largest_is_A_minus_B :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_is_A_minus_B_l563_56321


namespace NUMINAMATH_GPT_find_value_of_expression_l563_56395

theorem find_value_of_expression (m : ℝ) (h_m : m^2 - 3 * m + 1 = 0) : 2 * m^2 - 6 * m - 2024 = -2026 := by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l563_56395


namespace NUMINAMATH_GPT_ratio_albert_betty_l563_56368

theorem ratio_albert_betty (A M B : ℕ) (h1 : A = 2 * M) (h2 : M = A - 10) (h3 : B = 5) :
  A / B = 4 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_ratio_albert_betty_l563_56368


namespace NUMINAMATH_GPT_telephone_number_problem_l563_56307

theorem telephone_number_problem :
  ∃ A B C D E F G H I J : ℕ,
    (A > B) ∧ (B > C) ∧ (D > E) ∧ (E > F) ∧ (G > H) ∧ (H > I) ∧ (I > J) ∧
    (D = E + 1) ∧ (E = F + 1) ∧ (D % 2 = 0) ∧ 
    (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2) ∧ (G % 2 = 1) ∧ (H % 2 = 1) ∧ (I % 2 = 1) ∧ (J % 2 = 1) ∧
    (A + B + C = 7) ∧ (B + C + F = 10) ∧ (A = 7) :=
sorry

end NUMINAMATH_GPT_telephone_number_problem_l563_56307


namespace NUMINAMATH_GPT_no_solution_inequality_l563_56363

theorem no_solution_inequality (a : ℝ) : (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ (a ≤ 2 ∨ a ≥ 4) := 
sorry

end NUMINAMATH_GPT_no_solution_inequality_l563_56363


namespace NUMINAMATH_GPT_root_exists_l563_56387

variable {R : Type} [LinearOrderedField R]
variables (a b c : R)

def f (x : R) : R := a * x^2 + b * x + c

theorem root_exists (h : f a b c ((a - b - c) / (2 * a)) = 0) : f a b c (-1) = 0 ∨ f a b c 1 = 0 := by
  sorry

end NUMINAMATH_GPT_root_exists_l563_56387


namespace NUMINAMATH_GPT_office_expense_reduction_l563_56332

theorem office_expense_reduction (x : ℝ) (h : 0 ≤ x) (h' : x ≤ 1) : 
  2500 * (1 - x) ^ 2 = 1600 :=
sorry

end NUMINAMATH_GPT_office_expense_reduction_l563_56332


namespace NUMINAMATH_GPT_sum_not_divisible_by_10_iff_l563_56334

theorem sum_not_divisible_by_10_iff (n : ℕ) :
  ¬ (1981^n + 1982^n + 1983^n + 1984^n) % 10 = 0 ↔ n % 4 = 0 :=
sorry

end NUMINAMATH_GPT_sum_not_divisible_by_10_iff_l563_56334


namespace NUMINAMATH_GPT_largest_angle_in_ratio_triangle_l563_56373

theorem largest_angle_in_ratio_triangle (a b c : ℕ) (h_ratios : 2 * c = 3 * b ∧ 3 * b = 4 * a)
  (h_sum : a + b + c = 180) : max a (max b c) = 80 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_ratio_triangle_l563_56373


namespace NUMINAMATH_GPT_express_2_175_billion_in_scientific_notation_l563_56328

-- Definition of scientific notation
def scientific_notation (a : ℝ) (n : ℤ) (value : ℝ) : Prop :=
  value = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Theorem stating the problem
theorem express_2_175_billion_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), scientific_notation a n 2.175e9 ∧ a = 2.175 ∧ n = 9 :=
by
  sorry

end NUMINAMATH_GPT_express_2_175_billion_in_scientific_notation_l563_56328


namespace NUMINAMATH_GPT_train_length_correct_l563_56396

open Real

-- Define the conditions
def bridge_length : ℝ := 150
def time_to_cross_bridge : ℝ := 7.5
def time_to_cross_lamp_post : ℝ := 2.5

-- Define the length of the train
def train_length : ℝ := 75

theorem train_length_correct :
  ∃ L : ℝ, (L / time_to_cross_lamp_post = (L + bridge_length) / time_to_cross_bridge) ∧ L = train_length :=
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l563_56396


namespace NUMINAMATH_GPT_stepa_multiplied_numbers_l563_56384

theorem stepa_multiplied_numbers (x : ℤ) (hx : (81 * x) % 16 = 0) :
  ∃ (a b : ℕ), a * b = 54 ∧ a < 10 ∧ b < 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_stepa_multiplied_numbers_l563_56384


namespace NUMINAMATH_GPT_number_of_true_propositions_is_one_l563_56397

-- Define propositions
def prop1 (a b c : ℝ) : Prop := a > b ∧ c ≠ 0 → a * c > b * c
def prop2 (a b c : ℝ) : Prop := a > b → a * c^2 > b * c^2
def prop3 (a b c : ℝ) : Prop := a * c^2 > b * c^2 → a > b
def prop4 (a b : ℝ) : Prop := a > b → (1 / a) < (1 / b)
def prop5 (a b c d : ℝ) : Prop := a > b ∧ b > 0 ∧ c > d → a * c > b * d

-- The main theorem stating the number of true propositions
theorem number_of_true_propositions_is_one (a b c d : ℝ) :
  (prop3 a b c) ∧ (¬ prop1 a b c) ∧ (¬ prop2 a b c) ∧ (¬ prop4 a b) ∧ (¬ prop5 a b c d) :=
by
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_is_one_l563_56397


namespace NUMINAMATH_GPT_journey_total_distance_l563_56369

theorem journey_total_distance (D : ℝ) 
  (train_fraction : ℝ := 3/5) 
  (bus_fraction : ℝ := 7/20) 
  (walk_distance : ℝ := 6.5) 
  (total_fraction : ℝ := 1) : 
  (1 - (train_fraction + bus_fraction)) * D = walk_distance → D = 130 := 
by
  sorry

end NUMINAMATH_GPT_journey_total_distance_l563_56369


namespace NUMINAMATH_GPT_total_oranges_l563_56326

theorem total_oranges (a b c : ℕ) (h1 : a = 80) (h2 : b = 60) (h3 : c = 120) : a + b + c = 260 :=
by
  sorry

end NUMINAMATH_GPT_total_oranges_l563_56326


namespace NUMINAMATH_GPT_train_speed_l563_56377

theorem train_speed 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h_train_length : train_length = 400) 
  (h_bridge_length : bridge_length = 300) 
  (h_crossing_time : crossing_time = 45) : 
  (train_length + bridge_length) / crossing_time = 700 / 45 := 
  by
    rw [h_train_length, h_bridge_length, h_crossing_time]
    sorry

end NUMINAMATH_GPT_train_speed_l563_56377


namespace NUMINAMATH_GPT_james_bike_ride_l563_56319

variable {D P : ℝ}

theorem james_bike_ride :
  (∃ D P, 3 * D + (18 + 18 * 0.25) = 55.5 ∧ (18 = D * (1 + P / 100))) → P = 20 := by
  sorry

end NUMINAMATH_GPT_james_bike_ride_l563_56319
