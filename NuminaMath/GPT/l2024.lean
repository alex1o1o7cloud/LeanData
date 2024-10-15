import Mathlib

namespace NUMINAMATH_GPT_least_integer_value_of_x_l2024_202493

theorem least_integer_value_of_x (x : ℤ) (h : 3 * |x| + 4 < 19) : x = -4 :=
by sorry

end NUMINAMATH_GPT_least_integer_value_of_x_l2024_202493


namespace NUMINAMATH_GPT_coefficient_of_1_div_x_l2024_202459

open Nat

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℝ :=
  (1 / Real.sqrt x - 3)^n

theorem coefficient_of_1_div_x (x : ℝ) (n : ℕ) (h1 : n ∈ {m | m > 0}) (h2 : binomial_expansion x n = 16) :
  ∃ c : ℝ, c = 54 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_1_div_x_l2024_202459


namespace NUMINAMATH_GPT_probability_king_then_queen_l2024_202417

-- Definitions based on the conditions:
def total_cards : ℕ := 52
def ranks_per_suit : ℕ := 13
def suits : ℕ := 4
def kings : ℕ := 4
def queens : ℕ := 4

-- The problem statement rephrased as a theorem:
theorem probability_king_then_queen :
  (kings / total_cards : ℚ) * (queens / (total_cards - 1)) = 4 / 663 := 
by {
  sorry
}

end NUMINAMATH_GPT_probability_king_then_queen_l2024_202417


namespace NUMINAMATH_GPT_trader_cloth_sale_l2024_202473

theorem trader_cloth_sale (total_SP : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) (SP_per_meter : ℕ)
  (h1 : total_SP = 8400) (h2 : profit_per_meter = 12) (h3 : cost_per_meter = 128) (h4 : SP_per_meter = cost_per_meter + profit_per_meter) :
  ∃ (x : ℕ), SP_per_meter * x = total_SP ∧ x = 60 :=
by
  -- We will skip the proof using sorry
  sorry

end NUMINAMATH_GPT_trader_cloth_sale_l2024_202473


namespace NUMINAMATH_GPT_cookies_leftover_l2024_202445

def amelia_cookies := 52
def benjamin_cookies := 63
def chloe_cookies := 25
def total_cookies := amelia_cookies + benjamin_cookies + chloe_cookies
def package_size := 15

theorem cookies_leftover :
  total_cookies % package_size = 5 := by
  sorry

end NUMINAMATH_GPT_cookies_leftover_l2024_202445


namespace NUMINAMATH_GPT_num_of_integers_abs_leq_six_l2024_202472

theorem num_of_integers_abs_leq_six (x : ℤ) : 
  (|x - 3| ≤ 6) → ∃ (n : ℕ), n = 13 := 
by 
  sorry

end NUMINAMATH_GPT_num_of_integers_abs_leq_six_l2024_202472


namespace NUMINAMATH_GPT_find_p_l2024_202402

theorem find_p (p : ℝ) (h : 0 < p ∧ p < 1) : 
  p + (1 - p) * p + (1 - p)^2 * p = 0.784 → p = 0.4 :=
by
  intros h_eq
  sorry

end NUMINAMATH_GPT_find_p_l2024_202402


namespace NUMINAMATH_GPT_D_144_l2024_202430

def D (n : ℕ) : ℕ :=
  if n = 1 then 1 else sorry

theorem D_144 : D 144 = 51 := by
  sorry

end NUMINAMATH_GPT_D_144_l2024_202430


namespace NUMINAMATH_GPT_tan_x_eq_2_solution_l2024_202432

noncomputable def solution_set_tan_2 : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2}

theorem tan_x_eq_2_solution :
  {x : ℝ | Real.tan x = 2} = solution_set_tan_2 :=
by
  sorry

end NUMINAMATH_GPT_tan_x_eq_2_solution_l2024_202432


namespace NUMINAMATH_GPT_square_root_then_square_l2024_202478

theorem square_root_then_square (x : ℕ) (hx : x = 49) : (Nat.sqrt x) ^ 2 = 49 := by
  sorry

end NUMINAMATH_GPT_square_root_then_square_l2024_202478


namespace NUMINAMATH_GPT_probability_of_red_buttons_l2024_202436

noncomputable def initialJarA : ℕ := 16 -- total buttons in Jar A (6 red, 10 blue)
noncomputable def initialRedA : ℕ := 6 -- initial red buttons in Jar A
noncomputable def initialBlueA : ℕ := 10 -- initial blue buttons in Jar A

noncomputable def initialJarB : ℕ := 5 -- total buttons in Jar B (2 red, 3 blue)
noncomputable def initialRedB : ℕ := 2 -- initial red buttons in Jar B
noncomputable def initialBlueB : ℕ := 3 -- initial blue buttons in Jar B

noncomputable def transferRed : ℕ := 3
noncomputable def transferBlue : ℕ := 3

noncomputable def finalRedA : ℕ := initialRedA - transferRed
noncomputable def finalBlueA : ℕ := initialBlueA - transferBlue

noncomputable def finalRedB : ℕ := initialRedB + transferRed
noncomputable def finalBlueB : ℕ := initialBlueB + transferBlue

noncomputable def remainingJarA : ℕ := finalRedA + finalBlueA
noncomputable def finalJarB : ℕ := finalRedB + finalBlueB

noncomputable def probRedA : ℚ := finalRedA / remainingJarA
noncomputable def probRedB : ℚ := finalRedB / finalJarB

noncomputable def combinedProb : ℚ := probRedA * probRedB

theorem probability_of_red_buttons :
  combinedProb = 3 / 22 := sorry

end NUMINAMATH_GPT_probability_of_red_buttons_l2024_202436


namespace NUMINAMATH_GPT_train_crossing_time_l2024_202437

theorem train_crossing_time
  (length_train : ℕ)
  (speed_train_kmph : ℕ)
  (total_length : ℕ)
  (htotal_length : total_length = 225)
  (hlength_train : length_train = 150)
  (hspeed_train_kmph : speed_train_kmph = 45) : 
  (total_length / (speed_train_kmph * 1000 / 3600)) = 18 := by 
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2024_202437


namespace NUMINAMATH_GPT_problem_integer_and_decimal_parts_eq_2_l2024_202443

theorem problem_integer_and_decimal_parts_eq_2 :
  let x := 3
  let y := 2 - Real.sqrt 3
  2 * x^3 - (y^3 + 1 / y^3) = 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_integer_and_decimal_parts_eq_2_l2024_202443


namespace NUMINAMATH_GPT_quadrilateral_type_l2024_202488

theorem quadrilateral_type (m n p q : ℝ) (h : m^2 + n^2 + p^2 + q^2 = 2 * m * n + 2 * p * q) : 
  (m = n ∧ p = q) ∨ (m ≠ n ∧ p ≠ q ∧ ∃ k : ℝ, k^2 * (m^2 + n^2) = p^2 + q^2) := 
sorry

end NUMINAMATH_GPT_quadrilateral_type_l2024_202488


namespace NUMINAMATH_GPT_problem_statement_l2024_202474

noncomputable def h (y : ℂ) : ℂ := y^5 - y^3 + 1
noncomputable def p (y : ℂ) : ℂ := y^2 - 3

theorem problem_statement (y_1 y_2 y_3 y_4 y_5 : ℂ) (hroots : ∀ y, h y = 0 ↔ y = y_1 ∨ y = y_2 ∨ y = y_3 ∨ y = y_4 ∨ y = y_5) :
  (p y_1) * (p y_2) * (p y_3) * (p y_4) * (p y_5) = 22 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2024_202474


namespace NUMINAMATH_GPT_bryan_travel_ratio_l2024_202429

theorem bryan_travel_ratio
  (walk_time : ℕ)
  (bus_time : ℕ)
  (evening_walk_time : ℕ)
  (total_travel_hours : ℕ)
  (days_per_year : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_total : ℕ)
  (daily_travel_time : ℕ) :
  walk_time = 5 →
  bus_time = 20 →
  evening_walk_time = 5 →
  total_travel_hours = 365 →
  days_per_year = 365 →
  minutes_per_hour = 60 →
  minutes_total = total_travel_hours * minutes_per_hour →
  daily_travel_time = (walk_time + bus_time + evening_walk_time) * 2 →
  (minutes_total / daily_travel_time = days_per_year) →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bryan_travel_ratio_l2024_202429


namespace NUMINAMATH_GPT_Cherry_weekly_earnings_l2024_202416

theorem Cherry_weekly_earnings :
  let cost_3_5 := 2.50
  let cost_6_8 := 4.00
  let cost_9_12 := 6.00
  let cost_13_15 := 8.00
  let num_5kg := 4
  let num_8kg := 2
  let num_10kg := 3
  let num_14kg := 1
  let daily_earnings :=
    (num_5kg * cost_3_5) + (num_8kg * cost_6_8) + (num_10kg * cost_9_12) + (num_14kg * cost_13_15)
  let weekly_earnings := daily_earnings * 7
  weekly_earnings = 308 := by
  sorry

end NUMINAMATH_GPT_Cherry_weekly_earnings_l2024_202416


namespace NUMINAMATH_GPT_common_ratio_of_geometric_seq_l2024_202494

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the geometric sequence property
def geometric_seq_property (a2 a3 a6 : ℤ) : Prop :=
  a3 * a3 = a2 * a6

-- State the main theorem
theorem common_ratio_of_geometric_seq (a d : ℤ) (h : ¬d = 0) :
  geometric_seq_property (arithmetic_seq a d 2) (arithmetic_seq a d 3) (arithmetic_seq a d 6) →
  ∃ q : ℤ, q = 3 ∨ q = 1 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_seq_l2024_202494


namespace NUMINAMATH_GPT_solve_eq_solve_ineq_l2024_202462

-- Proof Problem 1 statement
theorem solve_eq (x : ℝ) : (2 / (x + 3) - (x - 3) / (2 * x + 6) = 1) → (x = 1 / 3) :=
by sorry

-- Proof Problem 2 statement
theorem solve_ineq (x : ℝ) : (2 * x - 1 > 3 * (x - 1)) ∧ ((5 - x) / 2 < x + 4) → (-1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_GPT_solve_eq_solve_ineq_l2024_202462


namespace NUMINAMATH_GPT_two_digit_number_l2024_202470

theorem two_digit_number (x y : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h1 : x^2 + y^2 = 10*x + y + 11) (h2 : 2*x*y = 10*x + y - 5) :
  10*x + y = 95 ∨ 10*x + y = 15 := 
sorry

end NUMINAMATH_GPT_two_digit_number_l2024_202470


namespace NUMINAMATH_GPT_three_Z_five_l2024_202427

def Z (a b : ℤ) : ℤ := b + 10 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = 8 := sorry

end NUMINAMATH_GPT_three_Z_five_l2024_202427


namespace NUMINAMATH_GPT_cylinder_surface_area_proof_l2024_202457

noncomputable def sphere_volume := (500 * Real.pi) / 3
noncomputable def cylinder_base_diameter := 8
noncomputable def cylinder_surface_area := 80 * Real.pi

theorem cylinder_surface_area_proof :
  ∀ (R : ℝ) (r h : ℝ), 
    (4 * Real.pi / 3) * R^3 = (500 * Real.pi) / 3 → -- sphere volume condition
    2 * r = cylinder_base_diameter →               -- base diameter condition
    r * r + (h / 2)^2 = R^2 →                      -- Pythagorean theorem (half height)
    2 * Real.pi * r * h + 2 * Real.pi * r^2 = cylinder_surface_area := -- surface area formula
by
  intros R r h sphere_vol_cond base_diameter_cond pythagorean_cond
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_proof_l2024_202457


namespace NUMINAMATH_GPT_complement_union_M_N_l2024_202434

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end NUMINAMATH_GPT_complement_union_M_N_l2024_202434


namespace NUMINAMATH_GPT_max_load_truck_l2024_202428

theorem max_load_truck (bag_weight : ℕ) (num_bags : ℕ) (remaining_load : ℕ) 
  (h1 : bag_weight = 8) (h2 : num_bags = 100) (h3 : remaining_load = 100) : 
  bag_weight * num_bags + remaining_load = 900 :=
by
  -- We leave the proof step intentionally, as per instructions.
  sorry

end NUMINAMATH_GPT_max_load_truck_l2024_202428


namespace NUMINAMATH_GPT_odd_integers_count_between_fractions_l2024_202486

theorem odd_integers_count_between_fractions :
  ∃ (count : ℕ), count = 14 ∧
  ∀ (n : ℤ), (25:ℚ)/3 < (n : ℚ) ∧ (n : ℚ) < (73 : ℚ)/2 ∧ (n % 2 = 1) :=
sorry

end NUMINAMATH_GPT_odd_integers_count_between_fractions_l2024_202486


namespace NUMINAMATH_GPT_pond_depth_range_l2024_202414

theorem pond_depth_range (d : ℝ) (adam_false : d < 10) (ben_false : d > 8) (carla_false : d ≠ 7) : 
    8 < d ∧ d < 10 :=
by
  sorry

end NUMINAMATH_GPT_pond_depth_range_l2024_202414


namespace NUMINAMATH_GPT_largest_area_of_rotating_triangle_l2024_202406

def Point := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (13, 0)
def C : Point := (21, 0)

def line (P : Point) (slope : ℝ) (x : ℝ) : ℝ := P.2 + slope * (x - P.1)

def l_A (x : ℝ) : ℝ := line A 1 x
def l_B (x : ℝ) : ℝ := x
def l_C (x : ℝ) : ℝ := line C (-1) x

def rotating_triangle_max_area (l_A l_B l_C : ℝ → ℝ) : ℝ := 116.5

theorem largest_area_of_rotating_triangle :
  rotating_triangle_max_area l_A l_B l_C = 116.5 :=
sorry

end NUMINAMATH_GPT_largest_area_of_rotating_triangle_l2024_202406


namespace NUMINAMATH_GPT_days_to_finish_job_l2024_202467

def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 4 / 15
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

theorem days_to_finish_job (A B C : ℚ) (h1 : A + B = work_rate_a_b) (h2 : C = work_rate_c) :
  1 / (A + B + C) = 3 :=
by
  sorry

end NUMINAMATH_GPT_days_to_finish_job_l2024_202467


namespace NUMINAMATH_GPT_initial_tanks_hold_fifteen_fish_l2024_202465

theorem initial_tanks_hold_fifteen_fish (t : Nat) (additional_tanks : Nat) (fish_per_additional_tank : Nat) (total_fish : Nat) :
  t = 3 ∧ additional_tanks = 3 ∧ fish_per_additional_tank = 10 ∧ total_fish = 75 → 
  ∀ (F : Nat), (F * t) = 45 → F = 15 :=
by
  sorry

end NUMINAMATH_GPT_initial_tanks_hold_fifteen_fish_l2024_202465


namespace NUMINAMATH_GPT_distinct_prime_factors_2310_l2024_202491

theorem distinct_prime_factors_2310 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ (S.card = 5) ∧ (S.prod id = 2310) := by
  sorry

end NUMINAMATH_GPT_distinct_prime_factors_2310_l2024_202491


namespace NUMINAMATH_GPT_new_pyramid_volume_l2024_202440

/-- Given an original pyramid with volume 40 cubic inches, where the length is doubled, 
    the width is tripled, and the height is increased by 50%, 
    prove that the volume of the new pyramid is 360 cubic inches. -/
theorem new_pyramid_volume (V : ℝ) (l w h : ℝ) 
  (h_volume : V = 1 / 3 * l * w * h) 
  (h_original : V = 40) : 
  (2 * l) * (3 * w) * (1.5 * h) / 3 = 360 :=
by
  sorry

end NUMINAMATH_GPT_new_pyramid_volume_l2024_202440


namespace NUMINAMATH_GPT_curved_surface_area_of_cone_l2024_202401

noncomputable def slant_height : ℝ := 22
noncomputable def radius : ℝ := 7
noncomputable def pi : ℝ := Real.pi

theorem curved_surface_area_of_cone :
  abs (pi * radius * slant_height - 483.22) < 0.01 := 
by
  sorry

end NUMINAMATH_GPT_curved_surface_area_of_cone_l2024_202401


namespace NUMINAMATH_GPT_angle_comparison_l2024_202469

theorem angle_comparison :
  let A := 60.4
  let B := 60.24
  let C := 60.24
  A > B ∧ B = C :=
by
  sorry

end NUMINAMATH_GPT_angle_comparison_l2024_202469


namespace NUMINAMATH_GPT_cos_inequality_range_l2024_202489

theorem cos_inequality_range (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 2 * Real.pi) (h₃ : Real.cos x ≤ 1 / 2) :
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3) := 
sorry

end NUMINAMATH_GPT_cos_inequality_range_l2024_202489


namespace NUMINAMATH_GPT_marc_watch_days_l2024_202454

theorem marc_watch_days (bought_episodes : ℕ) (watch_fraction : ℚ) (episodes_per_day : ℚ) (total_days : ℕ) : 
  bought_episodes = 50 → 
  watch_fraction = 1 / 10 → 
  episodes_per_day = (50 : ℚ) * watch_fraction → 
  total_days = (bought_episodes : ℚ) / episodes_per_day →
  total_days = 10 := 
sorry

end NUMINAMATH_GPT_marc_watch_days_l2024_202454


namespace NUMINAMATH_GPT_julia_tuesday_kids_l2024_202412

theorem julia_tuesday_kids :
  ∃ x : ℕ, (∃ y : ℕ, y = 6 ∧ y = x + 1) → x = 5 := 
by
  sorry

end NUMINAMATH_GPT_julia_tuesday_kids_l2024_202412


namespace NUMINAMATH_GPT_EricBenJackMoneySum_l2024_202446

noncomputable def EricBenJackTotal (E B J : ℕ) :=
  (E + B + J : ℕ)

theorem EricBenJackMoneySum :
  ∀ (E B J : ℕ), (E = B - 10) → (B = J - 9) → (J = 26) → (EricBenJackTotal E B J) = 50 :=
by
  intros E B J
  intro hE hB hJ
  rw [hJ] at hB
  rw [hB] at hE
  sorry

end NUMINAMATH_GPT_EricBenJackMoneySum_l2024_202446


namespace NUMINAMATH_GPT_original_denominator_l2024_202460

theorem original_denominator (d : ℕ) (h : 3 * (d : ℚ) = 2) : d = 3 := 
by
  sorry

end NUMINAMATH_GPT_original_denominator_l2024_202460


namespace NUMINAMATH_GPT_polygon_is_quadrilateral_l2024_202449

-- Problem statement in Lean 4
theorem polygon_is_quadrilateral 
  (n : ℕ) 
  (h₁ : (n - 2) * 180 = 360) :
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_polygon_is_quadrilateral_l2024_202449


namespace NUMINAMATH_GPT_verify_total_amount_l2024_202475

noncomputable def total_withdrawable_amount (a r : ℝ) : ℝ :=
  a / r * ((1 + r) ^ 5 - (1 + r))

theorem verify_total_amount (a r : ℝ) (h_r_nonzero : r ≠ 0) :
  total_withdrawable_amount a r = a / r * ((1 + r)^5 - (1 + r)) :=
by
  sorry

end NUMINAMATH_GPT_verify_total_amount_l2024_202475


namespace NUMINAMATH_GPT_jane_total_score_l2024_202468

theorem jane_total_score :
  let correct_answers := 17
  let incorrect_answers := 12
  let unanswered_questions := 6
  let total_questions := 35
  let points_per_correct := 1
  let points_per_incorrect := -0.25
  let correct_points := correct_answers * points_per_correct
  let incorrect_points := incorrect_answers * points_per_incorrect
  let total_score := correct_points + incorrect_points
  total_score = 14 :=
by
  sorry

end NUMINAMATH_GPT_jane_total_score_l2024_202468


namespace NUMINAMATH_GPT_three_letter_words_with_A_at_least_once_l2024_202435

theorem three_letter_words_with_A_at_least_once :
  let total_words := 4^3
  let words_without_A := 3^3
  total_words - words_without_A = 37 :=
by
  let total_words := 4^3
  let words_without_A := 3^3
  sorry

end NUMINAMATH_GPT_three_letter_words_with_A_at_least_once_l2024_202435


namespace NUMINAMATH_GPT_trapezoid_area_l2024_202464

-- Define the given conditions in the problem
variables (EF GH h EG FH : ℝ)
variables (EF_parallel_GH : true) -- EF and GH are parallel (not used in the calculation)
variables (EF_eq_70 : EF = 70)
variables (GH_eq_40 : GH = 40)
variables (h_eq_15 : h = 15)
variables (EG_eq_20 : EG = 20)
variables (FH_eq_25 : FH = 25)

-- Define the main theorem to prove
theorem trapezoid_area (EF GH h EG FH : ℝ) 
  (EF_eq_70 : EF = 70) 
  (GH_eq_40 : GH = 40) 
  (h_eq_15 : h = 15) 
  (EG_eq_20 : EG = 20) 
  (FH_eq_25 : FH = 25) : 
  0.5 * (EF + GH) * h = 825 := 
by 
  sorry

end NUMINAMATH_GPT_trapezoid_area_l2024_202464


namespace NUMINAMATH_GPT_sum_of_two_numbers_l2024_202413

theorem sum_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : x * y = 200) : (x + y = 30) :=
by sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l2024_202413


namespace NUMINAMATH_GPT_farmer_land_l2024_202492

variable (A C G P T : ℝ)
variable (h1 : C = 0.90 * A)
variable (h2 : G = 0.10 * C)
variable (h3 : P = 0.80 * C)
variable (h4 : T = 450)
variable (h5 : C = G + P + T)

theorem farmer_land (A : ℝ) (h1 : C = 0.90 * A) (h2 : G = 0.10 * C) (h3 : P = 0.80 * C) (h4 : T = 450) (h5 : C = G + P + T) : A = 5000 := by
  sorry

end NUMINAMATH_GPT_farmer_land_l2024_202492


namespace NUMINAMATH_GPT_sum_angles_triangle_complement_l2024_202438

theorem sum_angles_triangle_complement (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 180 - C = 130) : A + B = 130 :=
by
  have hC : C = 50 := by linarith
  linarith

end NUMINAMATH_GPT_sum_angles_triangle_complement_l2024_202438


namespace NUMINAMATH_GPT_sequences_properties_l2024_202498

-- Definitions based on the problem conditions
def geom_sequence (a : ℕ → ℕ) := ∃ q : ℕ, a 1 = 2 ∧ a 3 = 18 ∧ ∀ n, a (n + 1) = a n * q
def arith_sequence (b : ℕ → ℕ) := b 1 = 2 ∧ ∃ d : ℕ, ∀ n, b (n + 1) = b n + d
def condition (a : ℕ → ℕ) (b : ℕ → ℕ) := a 1 + a 2 + a 3 > 20 ∧ a 1 + a 2 + a 3 = b 1 + b 2 + b 3 + b 4

-- Proof statement: proving the general term of the geometric sequence and the sum of the arithmetic sequence
theorem sequences_properties (a : ℕ → ℕ) (b : ℕ → ℕ) :
  geom_sequence a → arith_sequence b → condition a b →
  (∀ n, a n = 2 * 3^(n - 1)) ∧ (∀ n, S_n = 3 / 2 * n^2 + 1 / 2 * n) :=
by
  sorry

end NUMINAMATH_GPT_sequences_properties_l2024_202498


namespace NUMINAMATH_GPT_min_value_fraction_l2024_202403

theorem min_value_fraction (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : m + 2 * n = 1) : 
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_min_value_fraction_l2024_202403


namespace NUMINAMATH_GPT_passengers_at_18_max_revenue_l2024_202482

noncomputable def P (t : ℝ) : ℝ :=
if 10 ≤ t ∧ t < 20 then 500 - 4 * (20 - t)^2 else
if 20 ≤ t ∧ t ≤ 30 then 500 else 0

noncomputable def Q (t : ℝ) : ℝ :=
if 10 ≤ t ∧ t < 20 then -8 * t - (1800 / t) + 320 else
if 20 ≤ t ∧ t ≤ 30 then 1400 / t else 0

-- 1. Prove P(18) = 484
theorem passengers_at_18 : P 18 = 484 := sorry

-- 2. Prove that Q(t) is maximized at t = 15 with a maximum value of 80
theorem max_revenue : ∃ t, Q t = 80 ∧ t = 15 := sorry

end NUMINAMATH_GPT_passengers_at_18_max_revenue_l2024_202482


namespace NUMINAMATH_GPT_simplify_fraction_l2024_202425

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2024_202425


namespace NUMINAMATH_GPT_not_sufficient_nor_necessary_l2024_202476

theorem not_sufficient_nor_necessary (a b : ℝ) (hb : b ≠ 0) :
  ¬ ((a > b) ↔ (1 / a < 1 / b)) :=
by
  sorry

end NUMINAMATH_GPT_not_sufficient_nor_necessary_l2024_202476


namespace NUMINAMATH_GPT_find_bettys_balance_l2024_202471

-- Define the conditions as hypotheses
def balance_in_bettys_account (B : ℕ) : Prop :=
  -- Gina has two accounts with a combined balance equal to $1,728
  (2 * (B / 4)) = 1728

-- State the theorem to be proven
theorem find_bettys_balance (B : ℕ) (h : balance_in_bettys_account B) : B = 3456 :=
by
  -- The proof is provided here as a "sorry"
  sorry

end NUMINAMATH_GPT_find_bettys_balance_l2024_202471


namespace NUMINAMATH_GPT_lucy_bought_cakes_l2024_202410

theorem lucy_bought_cakes (cookies chocolate total c : ℕ) (h1 : cookies = 4) (h2 : chocolate = 16) (h3 : total = 42) (h4 : c = total - (cookies + chocolate)) : c = 22 := by
  sorry

end NUMINAMATH_GPT_lucy_bought_cakes_l2024_202410


namespace NUMINAMATH_GPT_symmetric_line_eq_l2024_202452

theorem symmetric_line_eq (x y : ℝ) :
    3 * x - 4 * y + 5 = 0 ↔ 3 * x + 4 * (-y) + 5 = 0 :=
sorry

end NUMINAMATH_GPT_symmetric_line_eq_l2024_202452


namespace NUMINAMATH_GPT_value_of_x_l2024_202480

theorem value_of_x (x : ℝ) (h : 0.5 * x - (1 / 3) * x = 110) : x = 660 :=
sorry

end NUMINAMATH_GPT_value_of_x_l2024_202480


namespace NUMINAMATH_GPT_spending_difference_l2024_202447

-- Define the conditions
def spent_on_chocolate : ℤ := 7
def spent_on_candy_bar : ℤ := 2

-- The theorem to be proven
theorem spending_difference : (spent_on_chocolate - spent_on_candy_bar = 5) :=
by sorry

end NUMINAMATH_GPT_spending_difference_l2024_202447


namespace NUMINAMATH_GPT_maximum_ab_value_l2024_202420

noncomputable def ab_max (a b : ℝ) : ℝ :=
  if a > 0 then 2 * a * a - a * a * Real.log a else 0

theorem maximum_ab_value : ∀ (a b : ℝ), (∀ (x : ℝ), (Real.exp x - a * x + a) ≥ b) →
   ab_max a b ≤ if a = Real.exp (3 / 2) then (Real.exp 3) / 2 else sorry :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_maximum_ab_value_l2024_202420


namespace NUMINAMATH_GPT_center_of_the_hyperbola_l2024_202442

def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

structure Point where
  x : ℝ
  y : ℝ

def center_of_hyperbola_is (p : Point) : Prop :=
  hyperbola_eq (p.x + 3) (p.y + 4)

theorem center_of_the_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → center_of_hyperbola_is {x := 3, y := 4} :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_center_of_the_hyperbola_l2024_202442


namespace NUMINAMATH_GPT_find_p_power_l2024_202419

theorem find_p_power (p : ℕ) (h1 : p % 2 = 0) (h2 : (p + 1) % 10 = 7) : 
  (p % 10)^3 % 10 = (p % 10)^1 % 10 :=
by
  sorry

end NUMINAMATH_GPT_find_p_power_l2024_202419


namespace NUMINAMATH_GPT_book_store_sold_total_copies_by_saturday_l2024_202477

def copies_sold_on_monday : ℕ := 15
def copies_sold_on_tuesday : ℕ := copies_sold_on_monday * 2
def copies_sold_on_wednesday : ℕ := copies_sold_on_tuesday + (copies_sold_on_tuesday / 2)
def copies_sold_on_thursday : ℕ := copies_sold_on_wednesday + (copies_sold_on_wednesday / 2)
def copies_sold_on_friday_pre_promotion : ℕ := copies_sold_on_thursday + (copies_sold_on_thursday / 2)
def copies_sold_on_friday_post_promotion : ℕ := copies_sold_on_friday_pre_promotion + (copies_sold_on_friday_pre_promotion / 4)
def copies_sold_on_saturday : ℕ := copies_sold_on_friday_pre_promotion * 7 / 10

def total_copies_sold_by_saturday : ℕ :=
  copies_sold_on_monday + copies_sold_on_tuesday + copies_sold_on_wednesday +
  copies_sold_on_thursday + copies_sold_on_friday_post_promotion + copies_sold_on_saturday

theorem book_store_sold_total_copies_by_saturday : total_copies_sold_by_saturday = 357 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_book_store_sold_total_copies_by_saturday_l2024_202477


namespace NUMINAMATH_GPT_solution_to_inequalities_l2024_202456

theorem solution_to_inequalities (x : ℝ) : 
  (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 1) ↔ (1 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_GPT_solution_to_inequalities_l2024_202456


namespace NUMINAMATH_GPT_apples_total_l2024_202426

theorem apples_total (initial_apples : ℕ) (additional_apples : ℕ) (total_apples : ℕ) : 
  initial_apples = 56 → 
  additional_apples = 49 → 
  total_apples = initial_apples + additional_apples → 
  total_apples = 105 :=
by 
  intros h_initial h_additional h_total 
  rw [h_initial, h_additional] at h_total 
  exact h_total

end NUMINAMATH_GPT_apples_total_l2024_202426


namespace NUMINAMATH_GPT_symmetric_points_origin_l2024_202415

theorem symmetric_points_origin (a b : ℤ) (h1 : a = -5) (h2 : b = -1) : a - b = -4 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_origin_l2024_202415


namespace NUMINAMATH_GPT_infinite_series_sum_eq_one_fourth_l2024_202484

theorem infinite_series_sum_eq_one_fourth :
  (∑' n : ℕ, 3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+2))) = 1 / 4 :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_eq_one_fourth_l2024_202484


namespace NUMINAMATH_GPT_baker_new_cakes_bought_l2024_202448

variable (total_cakes initial_sold sold_more_than_bought : ℕ)

def new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) : ℕ :=
  total_cakes - (initial_sold + sold_more_than_bought)

theorem baker_new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) 
  (h1 : total_cakes = 170)
  (h2 : initial_sold = 78)
  (h3 : sold_more_than_bought = 47) :
  new_cakes_bought total_cakes initial_sold sold_more_than_bought = 78 :=
  sorry

end NUMINAMATH_GPT_baker_new_cakes_bought_l2024_202448


namespace NUMINAMATH_GPT_find_f1_and_f_prime1_l2024_202424

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_def : ∀ x : ℝ, f x = 2 * x^2 - f' 1 * x - 3

-- Proof using conditions
theorem find_f1_and_f_prime1 : f 1 + (f' 1) = -1 :=
sorry

end NUMINAMATH_GPT_find_f1_and_f_prime1_l2024_202424


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2024_202455

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = 2^x - 4

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h1 : is_even_function f)
  (h2 : satisfies_condition f) :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2024_202455


namespace NUMINAMATH_GPT_even_integers_count_l2024_202433

theorem even_integers_count (n : ℤ) (m : ℤ) (total_even : ℤ) 
  (h1 : m = 45) (h2 : total_even = 10) (h3 : m % 2 = 1) :
  (∃ k : ℤ, ∀ x : ℤ, 0 ≤ x ∧ x < total_even → k = n + 2 * x) ∧ (n = 26) :=
by
  sorry

end NUMINAMATH_GPT_even_integers_count_l2024_202433


namespace NUMINAMATH_GPT_incorrect_conclusion_D_l2024_202453

def parabola (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem incorrect_conclusion_D :
  ∀ x : ℝ, x < 2 → ∃ y1 y2 : ℝ, y1 = parabola x ∧ y2 = parabola (x + 1) ∧ y1 > y2 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_D_l2024_202453


namespace NUMINAMATH_GPT_sum_sequence_S_n_l2024_202405

variable {S : ℕ+ → ℚ}
noncomputable def S₁ : ℚ := 1 / 2
noncomputable def S₂ : ℚ := 5 / 6
noncomputable def S₃ : ℚ := 49 / 72
noncomputable def S₄ : ℚ := 205 / 288

theorem sum_sequence_S_n (n : ℕ+) :
  (S 1 = S₁) ∧ (S 2 = S₂) ∧ (S 3 = S₃) ∧ (S 4 = S₄) ∧ (∀ n : ℕ+, S n = n / (n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sum_sequence_S_n_l2024_202405


namespace NUMINAMATH_GPT_man_rate_in_still_water_l2024_202409

theorem man_rate_in_still_water (speed_with_stream speed_against_stream : ℝ)
  (h1 : speed_with_stream = 22) (h2 : speed_against_stream = 10) :
  (speed_with_stream + speed_against_stream) / 2 = 16 := by
  sorry

end NUMINAMATH_GPT_man_rate_in_still_water_l2024_202409


namespace NUMINAMATH_GPT_average_postcards_collected_per_day_l2024_202439

theorem average_postcards_collected_per_day 
    (a : ℕ) (d : ℕ) (n : ℕ) 
    (h_a : a = 10)
    (h_d : d = 12)
    (h_n : n = 7) :
    (a + (a + (n - 1) * d)) / 2 = 46 := by
  sorry

end NUMINAMATH_GPT_average_postcards_collected_per_day_l2024_202439


namespace NUMINAMATH_GPT_other_student_questions_l2024_202418

theorem other_student_questions (m k o : ℕ) (h1 : m = k - 3) (h2 : k = o + 8) (h3 : m = 40) : o = 35 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_other_student_questions_l2024_202418


namespace NUMINAMATH_GPT_oliver_learning_vowels_l2024_202497

theorem oliver_learning_vowels : 
  let learn := 5
  let rest_days (n : Nat) := n
  let total_days :=
    (learn + rest_days 1) + -- For 'A'
    (learn + rest_days 2) + -- For 'E'
    (learn + rest_days 3) + -- For 'I'
    (learn + rest_days 4) + -- For 'O'
    (rest_days 5 + learn)  -- For 'U' and 'Y'
  total_days = 40 :=
by
  sorry

end NUMINAMATH_GPT_oliver_learning_vowels_l2024_202497


namespace NUMINAMATH_GPT_find_b_l2024_202479

variable (b : ℝ)

theorem find_b 
    (h₁ : 0 < b)
    (h₂ : b < 4)
    (area_ratio : ∃ k : ℝ, k = 4/16 ∧ (4 + b) / -b = 2 * k) :
  b = -4/3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2024_202479


namespace NUMINAMATH_GPT_function_properties_l2024_202481

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 3)

theorem function_properties :
  (∃ x : ℝ, f x = -1) = false ∧ 
  (∃ x_0 : ℝ, -1 < x_0 ∧ x_0 < 0 ∧ deriv f x_0 = 0) ∧ 
  (∀ x : ℝ, -3 < x → f x > -1 / 2) ∧ 
  (∃ x_0 : ℝ, -3 < x_0 ∧ ∀ x : ℝ, -3 < x → f x_0 ≤ f x) :=
by
  sorry

end NUMINAMATH_GPT_function_properties_l2024_202481


namespace NUMINAMATH_GPT_fraction_calculation_l2024_202411

theorem fraction_calculation : 
  ( (1 / 5 + 1 / 7) / (3 / 8 + 2 / 9) ) = (864 / 1505) := 
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l2024_202411


namespace NUMINAMATH_GPT_total_boys_and_girls_sum_to_41_l2024_202421

theorem total_boys_and_girls_sum_to_41 (Rs : ℕ) (amount_per_boy : ℕ) (amount_per_girl : ℕ) (total_amount : ℕ) (num_boys : ℕ) :
  Rs = 1 ∧ amount_per_boy = 12 * Rs ∧ amount_per_girl = 8 * Rs ∧ total_amount = 460 * Rs ∧ num_boys = 33 →
  ∃ num_girls : ℕ, num_boys + num_girls = 41 :=
by
  sorry

end NUMINAMATH_GPT_total_boys_and_girls_sum_to_41_l2024_202421


namespace NUMINAMATH_GPT_prism_width_calculation_l2024_202495

theorem prism_width_calculation 
  (l h d : ℝ) 
  (h_l : l = 4) 
  (h_h : h = 10) 
  (h_d : d = 14) :
  ∃ w : ℝ, w = 4 * Real.sqrt 5 ∧ (l^2 + w^2 + h^2 = d^2) := 
by
  use 4 * Real.sqrt 5
  sorry

end NUMINAMATH_GPT_prism_width_calculation_l2024_202495


namespace NUMINAMATH_GPT_meteorological_forecasts_inaccuracy_l2024_202407

theorem meteorological_forecasts_inaccuracy :
  let pA_accurate := 0.8
  let pB_accurate := 0.7
  let pA_inaccurate := 1 - pA_accurate
  let pB_inaccurate := 1 - pB_accurate
  pA_inaccurate * pB_inaccurate = 0.06 :=
by
  sorry

end NUMINAMATH_GPT_meteorological_forecasts_inaccuracy_l2024_202407


namespace NUMINAMATH_GPT_Shawna_situps_l2024_202458

theorem Shawna_situps :
  ∀ (goal_per_day : ℕ) (total_days : ℕ) (tuesday_situps : ℕ) (wednesday_situps : ℕ),
  goal_per_day = 30 →
  total_days = 3 →
  tuesday_situps = 19 →
  wednesday_situps = 59 →
  (goal_per_day * total_days) - (tuesday_situps + wednesday_situps) = 12 :=
by
  intros goal_per_day total_days tuesday_situps wednesday_situps
  sorry

end NUMINAMATH_GPT_Shawna_situps_l2024_202458


namespace NUMINAMATH_GPT_tom_age_ratio_l2024_202485

theorem tom_age_ratio (T N : ℕ) (h1 : sum_ages = T) (h2 : T - N = 3 * (sum_ages_N_years_ago))
  (h3 : sum_ages = T) (h4 : sum_ages_N_years_ago = T - 4 * N) :
  T / N = 11 / 2 := 
by
  sorry

end NUMINAMATH_GPT_tom_age_ratio_l2024_202485


namespace NUMINAMATH_GPT_profit_is_35_percent_l2024_202444

def cost_price (C : ℝ) := C
def initial_selling_price (C : ℝ) := 1.20 * C
def second_selling_price (C : ℝ) := 1.50 * C
def final_selling_price (C : ℝ) := 1.35 * C

theorem profit_is_35_percent (C : ℝ) : 
    final_selling_price C - cost_price C = 0.35 * cost_price C :=
by
    sorry

end NUMINAMATH_GPT_profit_is_35_percent_l2024_202444


namespace NUMINAMATH_GPT_vector_simplification_l2024_202451

variables (V : Type) [AddCommGroup V]

variables (CE AC DE AD : V)

theorem vector_simplification :
  CE + AC - DE - AD = 0 :=
by
  sorry

end NUMINAMATH_GPT_vector_simplification_l2024_202451


namespace NUMINAMATH_GPT_gcd_lcm_product_l2024_202404

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 75) :
  (Nat.gcd a b) * (Nat.lcm a b) = 2250 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l2024_202404


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2024_202431

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def roots_of_quadratic_eq := {x : ℕ | x^2 - 5 * x + 6 = 0}

theorem isosceles_triangle_perimeter
  (a b c : ℕ)
  (h_isosceles : is_isosceles_triangle a b c)
  (h_roots : (a ∈ roots_of_quadratic_eq) ∧ (b ∈ roots_of_quadratic_eq) ∧ (c ∈ roots_of_quadratic_eq)) :
  (a + b + c = 7 ∨ a + b + c = 8) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2024_202431


namespace NUMINAMATH_GPT_pow_simplification_l2024_202496

theorem pow_simplification :
  9^6 * 3^3 / 27^4 = 27 :=
by
  sorry

end NUMINAMATH_GPT_pow_simplification_l2024_202496


namespace NUMINAMATH_GPT_tablespoons_in_half_cup_l2024_202461

theorem tablespoons_in_half_cup
    (grains_per_cup : ℕ)
    (half_cup : ℕ)
    (tbsp_to_tsp : ℕ)
    (grains_per_tsp : ℕ)
    (h1 : grains_per_cup = 480)
    (h2 : half_cup = grains_per_cup / 2)
    (h3 : tbsp_to_tsp = 3)
    (h4 : grains_per_tsp = 10) :
    (half_cup / (tbsp_to_tsp * grains_per_tsp) = 8) :=
by
  sorry

end NUMINAMATH_GPT_tablespoons_in_half_cup_l2024_202461


namespace NUMINAMATH_GPT_equilateral_triangle_octagon_area_ratio_l2024_202408

theorem equilateral_triangle_octagon_area_ratio
  (s_t s_o : ℝ)
  (h_triangle_area : (s_t^2 * Real.sqrt 3) / 4 = 2 * s_o^2 * (1 + Real.sqrt 2)) :
  s_t / s_o = Real.sqrt (8 * Real.sqrt 3 * (1 + Real.sqrt 2) / 3) :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_octagon_area_ratio_l2024_202408


namespace NUMINAMATH_GPT_mixed_number_eval_l2024_202499

theorem mixed_number_eval :
  -|-(18/5 : ℚ)| - (- (12 /5 : ℚ)) + (4/5 : ℚ) = - (2 / 5 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_mixed_number_eval_l2024_202499


namespace NUMINAMATH_GPT_gnomes_remaining_in_ravenswood_l2024_202490

theorem gnomes_remaining_in_ravenswood 
  (westerville_gnomes : ℕ)
  (ravenswood_initial_gnomes : ℕ)
  (taken_gnomes : ℕ)
  (remaining_gnomes : ℕ)
  (h1 : westerville_gnomes = 20)
  (h2 : ravenswood_initial_gnomes = 4 * westerville_gnomes)
  (h3 : taken_gnomes = (40 * ravenswood_initial_gnomes) / 100)
  (h4 : remaining_gnomes = ravenswood_initial_gnomes - taken_gnomes) :
  remaining_gnomes = 48 :=
by
  sorry

end NUMINAMATH_GPT_gnomes_remaining_in_ravenswood_l2024_202490


namespace NUMINAMATH_GPT_sum_of_common_ratios_eq_three_l2024_202423

theorem sum_of_common_ratios_eq_three
  (k a2 a3 b2 b3 : ℕ)
  (p r : ℕ)
  (h_nonconst1 : k ≠ 0)
  (h_nonconst2 : p ≠ r)
  (h_seq1 : a3 = k * p ^ 2)
  (h_seq2 : b3 = k * r ^ 2)
  (h_seq3 : a2 = k * p)
  (h_seq4 : b2 = k * r)
  (h_eq : a3 - b3 = 3 * (a2 - b2)) :
  p + r = 3 := 
sorry

end NUMINAMATH_GPT_sum_of_common_ratios_eq_three_l2024_202423


namespace NUMINAMATH_GPT_linear_eq_m_val_l2024_202450

theorem linear_eq_m_val (m : ℤ) (x : ℝ) : (5 * x ^ (m - 2) + 1 = 0) → (m = 3) :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_m_val_l2024_202450


namespace NUMINAMATH_GPT_least_positive_integer_exists_l2024_202422

theorem least_positive_integer_exists 
  (exists_k : ∃ k, (1 ≤ k ∧ k ≤ 2 * 5) ∧ (5^2 - 5 + k) % k = 0)
  (not_all_k : ¬(∀ k, (1 ≤ k ∧ k ≤ 2 * 5) → (5^2 - 5 + k) % k = 0)) :
  5 = 5 := 
by
  trivial

end NUMINAMATH_GPT_least_positive_integer_exists_l2024_202422


namespace NUMINAMATH_GPT_value_of_x_if_additive_inverses_l2024_202441

theorem value_of_x_if_additive_inverses (x : ℝ) 
  (h : 4 * x - 1 + (3 * x - 6) = 0) : x = 1 := by
sorry

end NUMINAMATH_GPT_value_of_x_if_additive_inverses_l2024_202441


namespace NUMINAMATH_GPT_largest_number_is_a_l2024_202466

-- Define the numbers in their respective bases
def a := 8 * 9 + 5
def b := 3 * 5^2 + 0 * 5 + 1 * 5^0
def c := 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0

theorem largest_number_is_a : a > b ∧ a > c :=
by
  -- These are the expected results, rest is the proof steps which we skip using sorry
  have ha : a = 77 := rfl
  have hb : b = 76 := rfl
  have hc : c = 9 := rfl
  sorry

end NUMINAMATH_GPT_largest_number_is_a_l2024_202466


namespace NUMINAMATH_GPT_area_of_path_cost_of_constructing_path_l2024_202400

-- Definitions for the problem
def original_length : ℕ := 75
def original_width : ℕ := 40
def path_width : ℕ := 25 / 10  -- 2.5 converted to a Lean-readable form

-- Conditions
def new_length := original_length + 2 * path_width
def new_width := original_width + 2 * path_width

def area_with_path := new_length * new_width
def area_without_path := original_length * original_width

-- Statements to prove
theorem area_of_path : area_with_path - area_without_path = 600 := sorry

def cost_per_sq_m : ℕ := 2
def total_cost := (area_with_path - area_without_path) * cost_per_sq_m

theorem cost_of_constructing_path : total_cost = 1200 := sorry

end NUMINAMATH_GPT_area_of_path_cost_of_constructing_path_l2024_202400


namespace NUMINAMATH_GPT_largest_int_value_of_m_l2024_202487

variable {x y m : ℤ}

theorem largest_int_value_of_m (h1 : x + 2 * y = 2 * m + 1)
                              (h2 : 2 * x + y = m + 2)
                              (h3 : x - y > 2) : m = -2 := 
sorry

end NUMINAMATH_GPT_largest_int_value_of_m_l2024_202487


namespace NUMINAMATH_GPT_two_digit_number_is_27_l2024_202463

theorem two_digit_number_is_27 :
  ∃ n : ℕ, (n / 10 < 10) ∧ (n % 10 < 10) ∧ 
  (100*(n) = 37*(10*(n) + 1)) ∧ 
  n = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_two_digit_number_is_27_l2024_202463


namespace NUMINAMATH_GPT_area_of_triangle_l2024_202483

theorem area_of_triangle (A : ℝ) (b : ℝ) (a : ℝ) (hA : A = 60) (hb : b = 4) (ha : a = 2 * Real.sqrt 3) : 
  1 / 2 * a * b * Real.sin (60 * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l2024_202483
