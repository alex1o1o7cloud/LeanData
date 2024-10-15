import Mathlib

namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1393_139330

theorem sufficient_but_not_necessary_condition (b : ℝ) :
  (∀ x : ℝ, b * x^2 - b * x + 1 > 0) ↔ (b = 0 ∨ (0 < b ∧ b < 4)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1393_139330


namespace NUMINAMATH_GPT_milk_for_18_cookies_l1393_139304

def milk_needed_to_bake_cookies (cookies : ℕ) (milk_per_24_cookies : ℚ) (quarts_to_pints : ℚ) : ℚ :=
  (milk_per_24_cookies * quarts_to_pints) * (cookies / 24)

theorem milk_for_18_cookies :
  milk_needed_to_bake_cookies 18 4.5 2 = 6.75 :=
by
  sorry

end NUMINAMATH_GPT_milk_for_18_cookies_l1393_139304


namespace NUMINAMATH_GPT_sqrt_seven_lt_three_l1393_139380

theorem sqrt_seven_lt_three : Real.sqrt 7 < 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_seven_lt_three_l1393_139380


namespace NUMINAMATH_GPT_total_votes_cast_l1393_139344

theorem total_votes_cast (total_votes : ℕ) (brenda_votes : ℕ) (percentage_brenda : ℚ) 
  (h1 : brenda_votes = 40) (h2 : percentage_brenda = 0.25) 
  (h3 : brenda_votes = percentage_brenda * total_votes) : total_votes = 160 := 
by sorry

end NUMINAMATH_GPT_total_votes_cast_l1393_139344


namespace NUMINAMATH_GPT_spherical_coordinate_cone_l1393_139388

-- Define spherical coordinates
structure SphericalCoordinate :=
  (ρ : ℝ)
  (θ : ℝ)
  (φ : ℝ)

-- Definition to describe the cone condition
def isCone (d : ℝ) (p : SphericalCoordinate) : Prop :=
  p.φ = d

-- The main theorem to state the problem
theorem spherical_coordinate_cone (d : ℝ) :
  ∀ (p : SphericalCoordinate), isCone d p → ∃ (ρ : ℝ), ∃ (θ : ℝ), (p = ⟨ρ, θ, d⟩) := sorry

end NUMINAMATH_GPT_spherical_coordinate_cone_l1393_139388


namespace NUMINAMATH_GPT_parallelogram_height_l1393_139338

theorem parallelogram_height
  (area : ℝ)
  (base : ℝ)
  (h_area : area = 375)
  (h_base : base = 25) :
  (area / base) = 15 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_height_l1393_139338


namespace NUMINAMATH_GPT_julia_played_with_kids_on_tuesday_l1393_139322

theorem julia_played_with_kids_on_tuesday (total: ℕ) (monday: ℕ) (tuesday: ℕ) 
  (h1: total = 18) (h2: monday = 4) : 
  tuesday = (total - monday) :=
by
  sorry

end NUMINAMATH_GPT_julia_played_with_kids_on_tuesday_l1393_139322


namespace NUMINAMATH_GPT_abs_eq_zero_solve_l1393_139375

theorem abs_eq_zero_solve (a b : ℚ) (h : |a - (1/2 : ℚ)| + |b + 5| = 0) : a + b = -9 / 2 := 
by
  sorry

end NUMINAMATH_GPT_abs_eq_zero_solve_l1393_139375


namespace NUMINAMATH_GPT_initial_rows_l1393_139311

theorem initial_rows (r T : ℕ) (h1 : T = 42 * r) (h2 : T = 28 * (r + 12)) : r = 24 :=
by
  sorry

end NUMINAMATH_GPT_initial_rows_l1393_139311


namespace NUMINAMATH_GPT_find_sequence_l1393_139396

def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → 
    a (n + 1) = (a n * a (n - 1)) / 
               Real.sqrt (a n^2 + a (n - 1)^2 + 1)

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 5

def sequence_property (F : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = Real.sqrt (1 / (Real.exp (F n * Real.log 10) - 1))

theorem find_sequence (a : ℕ → ℝ) (F : ℕ → ℝ) :
  initial_conditions a →
  recurrence_relation a →
  (∀ n : ℕ, n ≥ 2 →
    F (n + 1) = F n + F (n - 1)) →
  sequence_property F a :=
by
  intros h_initial h_recur h_F
  sorry

end NUMINAMATH_GPT_find_sequence_l1393_139396


namespace NUMINAMATH_GPT_count_routes_from_P_to_Q_l1393_139367

variable (P Q R S T : Type)
variable (roadPQ roadPS roadPT roadQR roadQS roadRS roadST : Prop)

theorem count_routes_from_P_to_Q :
  ∃ (routes : ℕ), routes = 16 :=
by
  sorry

end NUMINAMATH_GPT_count_routes_from_P_to_Q_l1393_139367


namespace NUMINAMATH_GPT_wallpaper_job_completion_l1393_139320

theorem wallpaper_job_completion (x : ℝ) (y : ℝ) 
  (h1 : ∀ a b : ℝ, (a = 1.5) → (7/x + (7-a)/(x-3) = 1)) 
  (h2 : y = x - 3) 
  (h3 : x - y = 3) : 
  (x = 14) ∧ (y = 11) :=
sorry

end NUMINAMATH_GPT_wallpaper_job_completion_l1393_139320


namespace NUMINAMATH_GPT_two_lines_in_3d_space_l1393_139381

theorem two_lines_in_3d_space : 
  ∀ x y z : ℝ, x^2 + 2 * x * (y + z) + y^2 = z^2 + 2 * z * (y + x) + x^2 → 
  (∃ a : ℝ, y = -z ∧ x = 0) ∨ (∃ b : ℝ, z = - (2 / 3) * x) :=
  sorry

end NUMINAMATH_GPT_two_lines_in_3d_space_l1393_139381


namespace NUMINAMATH_GPT_tom_chocolates_l1393_139354

variable (n : ℕ)

-- Lisa's box holds 64 chocolates and has unit dimensions (1^3 = 1 cubic unit)
def lisa_chocolates := 64
def lisa_volume := 1

-- Tom's box has dimensions thrice Lisa's and hence its volume (3^3 = 27 cubic units)
def tom_volume := 27

-- Number of chocolates Tom's box holds
theorem tom_chocolates : lisa_chocolates * tom_volume = 1728 := by
  -- calculations with known values
  sorry

end NUMINAMATH_GPT_tom_chocolates_l1393_139354


namespace NUMINAMATH_GPT_cubic_sum_div_pqr_eq_three_l1393_139327

theorem cubic_sum_div_pqr_eq_three (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p + q + r = 6) :
  (p^3 + q^3 + r^3) / (p * q * r) = 3 := 
by
  sorry

end NUMINAMATH_GPT_cubic_sum_div_pqr_eq_three_l1393_139327


namespace NUMINAMATH_GPT_choir_members_l1393_139372

theorem choir_members (n : ℕ) :
  (150 < n) ∧ (n < 250) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 8 = 5) → n = 159 :=
by
  sorry

end NUMINAMATH_GPT_choir_members_l1393_139372


namespace NUMINAMATH_GPT_octal_to_decimal_l1393_139319

theorem octal_to_decimal (n_octal : ℕ) (h : n_octal = 123) : 
  let d0 := 3 * 8^0
  let d1 := 2 * 8^1
  let d2 := 1 * 8^2
  n_octal = 64 + 16 + 3 :=
by
  sorry

end NUMINAMATH_GPT_octal_to_decimal_l1393_139319


namespace NUMINAMATH_GPT_smallest_7_digit_number_divisible_by_all_l1393_139333

def smallest_7_digit_number : ℕ := 7207200

theorem smallest_7_digit_number_divisible_by_all :
  smallest_7_digit_number >= 1000000 ∧ smallest_7_digit_number < 10000000 ∧
  smallest_7_digit_number % 35 = 0 ∧ 
  smallest_7_digit_number % 112 = 0 ∧ 
  smallest_7_digit_number % 175 = 0 ∧ 
  smallest_7_digit_number % 288 = 0 ∧ 
  smallest_7_digit_number % 429 = 0 ∧ 
  smallest_7_digit_number % 528 = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_7_digit_number_divisible_by_all_l1393_139333


namespace NUMINAMATH_GPT_evaluate_64_pow_7_over_6_l1393_139339

theorem evaluate_64_pow_7_over_6 : (64 : ℝ)^(7 / 6) = 128 := by
  have h : (64 : ℝ) = 2^6 := by norm_num
  rw [h]
  norm_num
  sorry

end NUMINAMATH_GPT_evaluate_64_pow_7_over_6_l1393_139339


namespace NUMINAMATH_GPT_trivia_team_students_l1393_139397

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (total_students : ℕ) :
  not_picked = 17 →
  groups = 8 →
  students_per_group = 6 →
  total_students = not_picked + groups * students_per_group →
  total_students = 65 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_trivia_team_students_l1393_139397


namespace NUMINAMATH_GPT_diamond_45_15_eq_3_l1393_139389

noncomputable def diamond (x y : ℝ) : ℝ := x / y

theorem diamond_45_15_eq_3 :
  ∀ (x y : ℝ), 
    (∀ x y : ℝ, (x * y) / y = x * (x / y)) ∧
    (∀ x : ℝ, (x / 1) / x = x / 1) ∧
    (∀ x y : ℝ, x / y = x / y) ∧
    1 / 1 = 1
    → diamond 45 15 = 3 :=
by
  intros x y H
  sorry

end NUMINAMATH_GPT_diamond_45_15_eq_3_l1393_139389


namespace NUMINAMATH_GPT_remainder_of_9876543210_div_101_l1393_139324

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end NUMINAMATH_GPT_remainder_of_9876543210_div_101_l1393_139324


namespace NUMINAMATH_GPT_problem1_problem2_l1393_139392

-- Problem (1) Lean Statement
theorem problem1 (c a b : ℝ) (hc : c > a) (ha : a > b) (hb : b > 0) : 
  a / (c - a) > b / (c - b) :=
sorry

-- Problem (2) Lean Statement
theorem problem2 (x : ℝ) (hx : x > 2) : 
  ∃ (xmin : ℝ), xmin = 6 ∧ (x = 6 → (x + 16 / (x - 2)) = 10) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1393_139392


namespace NUMINAMATH_GPT_solve_for_k_and_j_l1393_139350

theorem solve_for_k_and_j (k j : ℕ) (h1 : 64 / k = 8) (h2 : k * j = 128) : k = 8 ∧ j = 16 := by
  sorry

end NUMINAMATH_GPT_solve_for_k_and_j_l1393_139350


namespace NUMINAMATH_GPT_period_of_trig_sum_l1393_139357

theorem period_of_trig_sum : ∀ x : ℝ, 2 * Real.sin x + 3 * Real.cos x = 2 * Real.sin (x + 2 * Real.pi) + 3 * Real.cos (x + 2 * Real.pi) := 
sorry

end NUMINAMATH_GPT_period_of_trig_sum_l1393_139357


namespace NUMINAMATH_GPT_nesbitt_inequality_l1393_139312

theorem nesbitt_inequality {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) ∧ (a = b ∧ b = c → a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2) :=
sorry

end NUMINAMATH_GPT_nesbitt_inequality_l1393_139312


namespace NUMINAMATH_GPT_original_rectangle_perimeter_l1393_139370

theorem original_rectangle_perimeter (l w : ℝ) (h1 : w = l / 2)
  (h2 : 2 * (w + l / 3) = 40) : 2 * l + 2 * w = 72 :=
by
  sorry

end NUMINAMATH_GPT_original_rectangle_perimeter_l1393_139370


namespace NUMINAMATH_GPT_trapezoid_shaded_fraction_l1393_139323

theorem trapezoid_shaded_fraction (total_strips : ℕ) (shaded_strips : ℕ)
  (h_total : total_strips = 7) (h_shaded : shaded_strips = 4) :
  (shaded_strips : ℚ) / (total_strips : ℚ) = 4 / 7 := 
by
  sorry

end NUMINAMATH_GPT_trapezoid_shaded_fraction_l1393_139323


namespace NUMINAMATH_GPT_blue_paint_amount_l1393_139309

theorem blue_paint_amount
  (blue_white_ratio : ℚ := 4 / 5)
  (white_paint : ℚ := 15)
  (blue_paint : ℚ) :
  blue_paint = 12 :=
by
  sorry

end NUMINAMATH_GPT_blue_paint_amount_l1393_139309


namespace NUMINAMATH_GPT_sum_excluded_values_domain_l1393_139347

theorem sum_excluded_values_domain (x : ℝ) :
  (3 * x^2 - 9 * x + 6 = 0) → (x = 1 ∨ x = 2) ∧ (1 + 2 = 3) :=
by {
  -- given that 3x² - 9x + 6 = 0, we need to show that x = 1 or x = 2, and that their sum is 3
  sorry
}

end NUMINAMATH_GPT_sum_excluded_values_domain_l1393_139347


namespace NUMINAMATH_GPT_problem_statement_l1393_139336

variable (x : ℝ)

theorem problem_statement (h : x^2 - x - 1 = 0) : 1995 + 2 * x - x^3 = 1994 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1393_139336


namespace NUMINAMATH_GPT_race_distance_difference_l1393_139376

theorem race_distance_difference
  (d : ℕ) (tA tB : ℕ)
  (h_d: d = 80) 
  (h_tA: tA = 20) 
  (h_tB: tB = 25) :
  (d / tA) * tA = d ∧ (d - (d / tB) * tA) = 16 := 
by
  sorry

end NUMINAMATH_GPT_race_distance_difference_l1393_139376


namespace NUMINAMATH_GPT_conditions_necessary_sufficient_l1393_139337

variables (p q r s : Prop)

theorem conditions_necessary_sufficient :
  ((p → r) ∧ (¬ (r → p)) ∧ (q → r) ∧ (s → r) ∧ (q → s)) →
  ((s ↔ q) ∧ ((p → q) ∧ ¬ (q → p)) ∧ ((¬ p → ¬ s) ∧ ¬ (¬ s → ¬ p))) := by
  sorry

end NUMINAMATH_GPT_conditions_necessary_sufficient_l1393_139337


namespace NUMINAMATH_GPT_car_daily_rental_cost_l1393_139359

theorem car_daily_rental_cost 
  (x : ℝ)
  (cost_per_mile : ℝ)
  (budget : ℝ)
  (miles : ℕ)
  (h1 : cost_per_mile = 0.18)
  (h2 : budget = 75)
  (h3 : miles = 250)
  (h4 : x + (miles * cost_per_mile) = budget) : 
  x = 30 := 
sorry

end NUMINAMATH_GPT_car_daily_rental_cost_l1393_139359


namespace NUMINAMATH_GPT_carl_additional_gift_bags_l1393_139301

theorem carl_additional_gift_bags (definite_visitors additional_visitors extravagant_bags average_bags total_bags_needed : ℕ) :
  definite_visitors = 50 →
  additional_visitors = 40 →
  extravagant_bags = 10 →
  average_bags = 20 →
  total_bags_needed = 90 →
  (total_bags_needed - (extravagant_bags + average_bags)) = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_carl_additional_gift_bags_l1393_139301


namespace NUMINAMATH_GPT_larry_wins_probability_l1393_139313

noncomputable def probability_larry_wins (p_L : ℚ) (p_J : ℚ) : ℚ :=
  let q_L := 1 - p_L
  let q_J := 1 - p_J
  let r := q_L * q_J
  p_L / (1 - r)

theorem larry_wins_probability
  (p_L : ℚ) (p_J : ℚ) (h1 : p_L = 3 / 5) (h2 : p_J = 1 / 3) :
  probability_larry_wins p_L p_J = 9 / 11 :=
by 
  sorry

end NUMINAMATH_GPT_larry_wins_probability_l1393_139313


namespace NUMINAMATH_GPT_consecutive_sum_150_l1393_139387

theorem consecutive_sum_150 : ∃ (n : ℕ), n ≥ 2 ∧ (∃ a : ℕ, (n * (2 * a + n - 1)) / 2 = 150) :=
sorry

end NUMINAMATH_GPT_consecutive_sum_150_l1393_139387


namespace NUMINAMATH_GPT_age_ratio_in_years_l1393_139340

theorem age_ratio_in_years (p c x : ℕ) 
  (H1 : p - 2 = 3 * (c - 2)) 
  (H2 : p - 4 = 4 * (c - 4)) 
  (H3 : (p + x) / (c + x) = 2) : 
  x = 4 :=
sorry

end NUMINAMATH_GPT_age_ratio_in_years_l1393_139340


namespace NUMINAMATH_GPT_eq_perp_bisector_BC_area_triangle_ABC_l1393_139326

section Triangle_ABC

open Real

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Define the equation of the perpendicular bisector
theorem eq_perp_bisector_BC : ∀ x y : ℝ, 2 * x + y - 4 = 0 :=
sorry

-- Define the area of the triangle ABC
noncomputable def triangle_area : ℝ :=
1 / 2 * (abs ((-1 * 3 + 3 * (-2) + 3 * 4) - (3 * 4 + 1 * (-2) + 3*(-1))))

theorem area_triangle_ABC : triangle_area = 7 :=
sorry

end Triangle_ABC

end NUMINAMATH_GPT_eq_perp_bisector_BC_area_triangle_ABC_l1393_139326


namespace NUMINAMATH_GPT_lacy_percentage_correct_l1393_139314

variable (x : ℕ)

-- Definitions from the conditions
def total_problems := 8 * x
def missed_problems := 2 * x
def answered_problems := total_problems - missed_problems
def bonus_problems := x
def bonus_points := 2 * bonus_problems
def regular_points := answered_problems - bonus_problems
def total_points_scored := bonus_points + regular_points
def total_available_points := 8 * x + 2 * x

theorem lacy_percentage_correct :
  total_points_scored / total_available_points * 100 = 90 := by
  -- Proof steps would go here, but are not required per instructions.
  sorry

end NUMINAMATH_GPT_lacy_percentage_correct_l1393_139314


namespace NUMINAMATH_GPT_students_taking_neither_l1393_139302

variable (total_students math_students physics_students both_students : ℕ)
variable (h1 : total_students = 80)
variable (h2 : math_students = 50)
variable (h3 : physics_students = 40)
variable (h4 : both_students = 25)

theorem students_taking_neither (h1 : total_students = 80)
    (h2 : math_students = 50)
    (h3 : physics_students = 40)
    (h4 : both_students = 25) :
    total_students - (math_students - both_students + physics_students - both_students + both_students) = 15 :=
by
    sorry

end NUMINAMATH_GPT_students_taking_neither_l1393_139302


namespace NUMINAMATH_GPT_complement_of_union_l1393_139352

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union (hU : U = {1, 2, 3, 4}) (hM : M = {1, 2}) (hN : N = {2, 3}) :
  (U \ (M ∪ N)) = {4} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_union_l1393_139352


namespace NUMINAMATH_GPT_coconut_transport_l1393_139360

theorem coconut_transport (coconuts total_coconuts barbie_capacity bruno_capacity combined_capacity trips : ℕ)
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : bruno_capacity = 8)
  (h4 : combined_capacity = barbie_capacity + bruno_capacity)
  (h5 : combined_capacity = 12)
  (h6 : trips = total_coconuts / combined_capacity) :
  trips = 12 :=
by
  sorry

end NUMINAMATH_GPT_coconut_transport_l1393_139360


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1393_139317

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith : ∀ k, S (k + 1) - S k = S 1 - S 0)
  (h_S5 : S 5 = 10) (h_S10 : S 10 = 18) : S 15 = 26 :=
by
  -- Rest of the proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1393_139317


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1393_139390

def M : Set ℕ := {0, 2, 4}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N :
  {x | x ∈ M ∧ x ∈ N} = {0, 4} := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1393_139390


namespace NUMINAMATH_GPT_total_travel_time_l1393_139361

theorem total_travel_time (x y : ℝ) : 
  (x / 50 + y / 70 + 0.5) = (7 * x + 5 * y) / 350 + 0.5 :=
by
  sorry

end NUMINAMATH_GPT_total_travel_time_l1393_139361


namespace NUMINAMATH_GPT_farmer_spending_l1393_139368

theorem farmer_spending (X : ℝ) (hc : 0.80 * X + 0.60 * X = 49) : X = 35 := 
by
  sorry

end NUMINAMATH_GPT_farmer_spending_l1393_139368


namespace NUMINAMATH_GPT_percentage_increase_second_year_is_20_l1393_139334

noncomputable def find_percentage_increase_second_year : ℕ :=
  let P₀ := 1000
  let P₁ := P₀ + (10 * P₀) / 100
  let Pf := 1320
  let P := (Pf - P₁) * 100 / P₁
  P

theorem percentage_increase_second_year_is_20 :
  find_percentage_increase_second_year = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_second_year_is_20_l1393_139334


namespace NUMINAMATH_GPT_find_principal_l1393_139379

theorem find_principal :
  ∃ P r : ℝ, (8820 = P * (1 + r) ^ 2) ∧ (9261 = P * (1 + r) ^ 3) → (P = 8000) :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l1393_139379


namespace NUMINAMATH_GPT_possible_values_of_k_l1393_139391

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_k_l1393_139391


namespace NUMINAMATH_GPT_smallest_cars_number_l1393_139386

theorem smallest_cars_number :
  ∃ N : ℕ, N > 2 ∧ (N % 5 = 2) ∧ (N % 6 = 2) ∧ (N % 7 = 2) ∧ N = 212 := by
  sorry

end NUMINAMATH_GPT_smallest_cars_number_l1393_139386


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l1393_139332

theorem sum_of_squares_of_consecutive_integers (b : ℕ) (h : (b-1) * b * (b+1) = 12 * ((b-1) + b + (b+1))) : 
  (b - 1) * (b - 1) + b * b + (b + 1) * (b + 1) = 110 := 
by sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l1393_139332


namespace NUMINAMATH_GPT_equivalence_of_statements_l1393_139329

variable (P Q : Prop)

theorem equivalence_of_statements (h : P → Q) :
  (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end NUMINAMATH_GPT_equivalence_of_statements_l1393_139329


namespace NUMINAMATH_GPT_zoo_feeding_ways_l1393_139351

-- Noncomputable is used for definitions that are not algorithmically computable
noncomputable def numFeedingWays : Nat :=
  4 * 3 * 3 * 2 * 2

theorem zoo_feeding_ways :
  ∀ (pairs : Fin 4 → (String × String)), -- Representing pairs of animals
  numFeedingWays = 144 :=
by
  sorry

end NUMINAMATH_GPT_zoo_feeding_ways_l1393_139351


namespace NUMINAMATH_GPT_bags_of_cookies_l1393_139374

theorem bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 33) (h2 : cookies_per_bag = 11) : total_cookies / cookies_per_bag = 3 :=
by
  sorry

end NUMINAMATH_GPT_bags_of_cookies_l1393_139374


namespace NUMINAMATH_GPT_green_eyes_students_l1393_139356

def total_students := 45
def brown_hair_condition (green_eyes : ℕ) := 3 * green_eyes
def both_attributes := 9
def neither_attributes := 5

theorem green_eyes_students (green_eyes : ℕ) :
  (total_students = (green_eyes - both_attributes) + both_attributes
    + (brown_hair_condition green_eyes - both_attributes) + neither_attributes) →
  green_eyes = 10 :=
by
  sorry

end NUMINAMATH_GPT_green_eyes_students_l1393_139356


namespace NUMINAMATH_GPT_max_voters_after_T_l1393_139364

theorem max_voters_after_T (x : ℕ) (n : ℕ) (y : ℕ) (T : ℕ)  
  (h1 : x <= 10)
  (h2 : x > 0)
  (h3 : (nx + y) ≤ (n + 1) * (x - 1))
  (h4 : ∀ k, (x - k ≥ 0) ↔ (n ≤ T + 5)) :
  ∃ (m : ℕ), m = 5 := 
sorry

end NUMINAMATH_GPT_max_voters_after_T_l1393_139364


namespace NUMINAMATH_GPT_student_percentage_in_math_l1393_139394

theorem student_percentage_in_math (M H T : ℝ) (H_his : H = 84) (H_third : T = 69) (H_avg : (M + H + T) / 3 = 75) : M = 72 :=
by
  sorry

end NUMINAMATH_GPT_student_percentage_in_math_l1393_139394


namespace NUMINAMATH_GPT_keiko_speed_l1393_139318

theorem keiko_speed (a b : ℝ) (s : ℝ) (h1 : (2 * a + 2 * π * (b + 8)) / s = (2 * a + 2 * π * b) / s + 48) : s = π / 3 :=
by {
  sorry -- proof is not required
}

end NUMINAMATH_GPT_keiko_speed_l1393_139318


namespace NUMINAMATH_GPT_tangent_line_sum_l1393_139378

theorem tangent_line_sum {f : ℝ → ℝ} (h_tangent : ∀ x, f x = (1/2 * x) + 2) :
  (f 1) + (deriv f 1) = 3 :=
by
  -- derive the value at x=1 and the derivative manually based on h_tangent
  sorry

end NUMINAMATH_GPT_tangent_line_sum_l1393_139378


namespace NUMINAMATH_GPT_jackson_running_l1393_139384

variable (x : ℕ)

theorem jackson_running (h : x + 4 = 7) : x = 3 := by
  sorry

end NUMINAMATH_GPT_jackson_running_l1393_139384


namespace NUMINAMATH_GPT_volume_rectangular_solid_l1393_139321

theorem volume_rectangular_solid
  (a b c : ℝ) 
  (h1 : a * b = 12)
  (h2 : b * c = 8)
  (h3 : a * c = 6) :
  a * b * c = 24 :=
sorry

end NUMINAMATH_GPT_volume_rectangular_solid_l1393_139321


namespace NUMINAMATH_GPT_johns_cocktail_not_stronger_l1393_139366

-- Define the percentage of alcohol in each beverage
def beer_percent_alcohol : ℝ := 0.05
def liqueur_percent_alcohol : ℝ := 0.10
def vodka_percent_alcohol : ℝ := 0.40
def whiskey_percent_alcohol : ℝ := 0.50

-- Define the weights of the liquids used in the cocktails
def john_liqueur_weight : ℝ := 400
def john_whiskey_weight : ℝ := 100
def ivan_vodka_weight : ℝ := 400
def ivan_beer_weight : ℝ := 100

-- Calculate the alcohol contents in each cocktail
def johns_cocktail_alcohol : ℝ := john_liqueur_weight * liqueur_percent_alcohol + john_whiskey_weight * whiskey_percent_alcohol
def ivans_cocktail_alcohol : ℝ := ivan_vodka_weight * vodka_percent_alcohol + ivan_beer_weight * beer_percent_alcohol

-- The proof statement to prove that John's cocktail is not stronger than Ivan's cocktail
theorem johns_cocktail_not_stronger :
  johns_cocktail_alcohol ≤ ivans_cocktail_alcohol :=
by
  -- Add proof steps here
  sorry

end NUMINAMATH_GPT_johns_cocktail_not_stronger_l1393_139366


namespace NUMINAMATH_GPT_black_ink_cost_l1393_139349

theorem black_ink_cost (B : ℕ) 
  (h1 : 2 * B + 3 * 15 + 2 * 13 = 50 + 43) : B = 11 :=
by
  sorry

end NUMINAMATH_GPT_black_ink_cost_l1393_139349


namespace NUMINAMATH_GPT_find_b_l1393_139353

def nabla (a b : ℤ) (h : a ≠ b) : ℤ := (a + b) / (a - b)

theorem find_b (b : ℤ) (h : 3 ≠ b) (h_eq : nabla 3 b h = -4) : b = 5 :=
sorry

end NUMINAMATH_GPT_find_b_l1393_139353


namespace NUMINAMATH_GPT_consecutive_page_numbers_sum_l1393_139382

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) = 19881) : n + (n + 1) = 283 :=
sorry

end NUMINAMATH_GPT_consecutive_page_numbers_sum_l1393_139382


namespace NUMINAMATH_GPT_common_integer_root_l1393_139373

theorem common_integer_root (a x : ℤ) : (a * x + a = 7) ∧ (3 * x - a = 17) → a = 1 :=
by
    sorry

end NUMINAMATH_GPT_common_integer_root_l1393_139373


namespace NUMINAMATH_GPT_quadratic_rewrite_l1393_139341

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 25) (h2 : 2 * d * e = -40) (h3 : e^2 + f = -75) : d * e = -20 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_l1393_139341


namespace NUMINAMATH_GPT_net_income_correct_l1393_139377

-- Definition of income before tax
def total_income_before_tax : ℝ := 45000

-- Definition of tax rate
def tax_rate : ℝ := 0.13

-- Definition of tax amount
def tax_amount : ℝ := tax_rate * total_income_before_tax

-- Definition of net income after tax
def net_income_after_tax : ℝ := total_income_before_tax - tax_amount

-- Theorem statement
theorem net_income_correct : net_income_after_tax = 39150 := by
  sorry

end NUMINAMATH_GPT_net_income_correct_l1393_139377


namespace NUMINAMATH_GPT_find_number_l1393_139343

theorem find_number (x : ℕ) (h : (x / 5) - 154 = 6) : x = 800 := by
  sorry

end NUMINAMATH_GPT_find_number_l1393_139343


namespace NUMINAMATH_GPT_carson_clouds_l1393_139369

theorem carson_clouds (C D : ℕ) (h1 : D = 3 * C) (h2 : C + D = 24) : C = 6 :=
by
  sorry

end NUMINAMATH_GPT_carson_clouds_l1393_139369


namespace NUMINAMATH_GPT_compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l1393_139385

theorem compare_pi_314 : Real.pi > 3.14 :=
by sorry

theorem compare_neg_sqrt3_neg_sqrt2 : -Real.sqrt 3 < -Real.sqrt 2 :=
by sorry

theorem compare_2_sqrt5 : 2 < Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l1393_139385


namespace NUMINAMATH_GPT_angle_bisector_segment_conditional_equality_l1393_139363

theorem angle_bisector_segment_conditional_equality
  (a1 b1 a2 b2 : ℝ)
  (h1 : ∃ (P : ℝ), ∃ (e1 e2 : ℝ → ℝ), (e1 P = a1 ∧ e2 P = b1) ∧ (e1 P = a2 ∧ e2 P = b2)) :
  (1 / a1 + 1 / b1 = 1 / a2 + 1 / b2) :=
by 
  sorry

end NUMINAMATH_GPT_angle_bisector_segment_conditional_equality_l1393_139363


namespace NUMINAMATH_GPT_yellow_balls_in_bag_l1393_139335

open Classical

theorem yellow_balls_in_bag (Y : ℕ) (hY1 : (Y/(Y+2): ℝ) * ((Y-1)/(Y+1): ℝ) = 0.5) : Y = 5 := by
  sorry

end NUMINAMATH_GPT_yellow_balls_in_bag_l1393_139335


namespace NUMINAMATH_GPT_part1_part2_l1393_139383

variables (x y z : ℝ)

-- Conditions
def conditions := (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 1)

-- Part 1: Prove 2(x^2 + y^2 + z^2) + 9xyz >= 1
theorem part1 (h : conditions x y z) : 2 * (x^2 + y^2 + z^2) + 9 * x * y * z ≥ 1 :=
sorry

-- Part 2: Prove xy + yz + zx - 3xyz ≤ 1/4
theorem part2 (h : conditions x y z) : x * y + y * z + z * x - 3 * x * y * z ≤ 1 / 4 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1393_139383


namespace NUMINAMATH_GPT_simplify_fraction_l1393_139362

theorem simplify_fraction :
  ∀ (x : ℝ),
    (18 * x^4 - 9 * x^3 - 86 * x^2 + 16 * x + 96) /
    (18 * x^4 - 63 * x^3 + 22 * x^2 + 112 * x - 96) =
    (2 * x + 3) / (2 * x - 3) :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1393_139362


namespace NUMINAMATH_GPT_total_legs_arms_tentacles_correct_l1393_139346

-- Define the counts of different animals
def num_horses : Nat := 2
def num_dogs : Nat := 5
def num_cats : Nat := 7
def num_turtles : Nat := 3
def num_goat : Nat := 1
def num_snakes : Nat := 4
def num_spiders : Nat := 2
def num_birds : Nat := 3
def num_starfish : Nat := 1
def num_octopus : Nat := 1
def num_three_legged_dogs : Nat := 1

-- Define the legs, arms, and tentacles for each type of animal
def legs_per_horse : Nat := 4
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def legs_per_turtle : Nat := 4
def legs_per_goat : Nat := 4
def legs_per_snake : Nat := 0
def legs_per_spider : Nat := 8
def legs_per_bird : Nat := 2
def arms_per_starfish : Nat := 5
def tentacles_per_octopus : Nat := 6
def legs_per_three_legged_dog : Nat := 3

-- Define the total number of legs, arms, and tentacles
def total_legs_arms_tentacles : Nat := 
  (num_horses * legs_per_horse) + 
  (num_dogs * legs_per_dog) + 
  (num_cats * legs_per_cat) + 
  (num_turtles * legs_per_turtle) + 
  (num_goat * legs_per_goat) + 
  (num_snakes * legs_per_snake) + 
  (num_spiders * legs_per_spider) + 
  (num_birds * legs_per_bird) + 
  (num_starfish * arms_per_starfish) + 
  (num_octopus * tentacles_per_octopus) + 
  (num_three_legged_dogs * legs_per_three_legged_dog)

-- The theorem to prove
theorem total_legs_arms_tentacles_correct :
  total_legs_arms_tentacles = 108 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_legs_arms_tentacles_correct_l1393_139346


namespace NUMINAMATH_GPT_trigonometric_identities_l1393_139342

theorem trigonometric_identities
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (sinα : Real.sin α = 4 / 5)
  (cosβ : Real.cos β = 12 / 13) :
  Real.sin (α + β) = 63 / 65 ∧ Real.tan (α - β) = 33 / 56 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identities_l1393_139342


namespace NUMINAMATH_GPT_range_of_f_l1393_139345

noncomputable def f (x : ℝ) : ℝ := 2^(x + 2) - 4^x

def domain_M (x : ℝ) : Prop := 1 < x ∧ x < 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, domain_M x ∧ f x = y) ↔ -32 < y ∧ y < 4 :=
sorry

end NUMINAMATH_GPT_range_of_f_l1393_139345


namespace NUMINAMATH_GPT_age_difference_l1393_139371

theorem age_difference (A B C : ℕ) (h1 : B = 20) (h2 : C = B / 2) (h3 : A + B + C = 52) : A - B = 2 := by
  sorry

end NUMINAMATH_GPT_age_difference_l1393_139371


namespace NUMINAMATH_GPT_liars_positions_l1393_139307

structure Islander :=
  (position : Nat)
  (statement : String)

-- Define our islanders
def A : Islander := { position := 1, statement := "My closest tribesman in this line is 3 meters away from me." }
def D : Islander := { position := 4, statement := "My closest tribesman in this line is 1 meter away from me." }
def E : Islander := { position := 5, statement := "My closest tribesman in this line is 2 meters away from me." }

-- Define the other islanders with dummy statements
def B : Islander := { position := 2, statement := "" }
def C : Islander := { position := 3, statement := "" }
def F : Islander := { position := 6, statement := "" }

-- Define the main theorem
theorem liars_positions (knights_count : Nat) (liars_count : Nat) (is_knight : Islander → Bool)
  (is_lair : Islander → Bool) : 
  ( ∀ x, is_knight x ↔ ¬is_lair x ) → -- Knight and liar are mutually exclusive
  knights_count = 3 → 
  liars_count = 3 →
  is_knight A = false → 
  is_knight D = false → 
  is_knight E = false → 
  is_lair A = true ∧
  is_lair D = true ∧
  is_lair E = true := by
  sorry

end NUMINAMATH_GPT_liars_positions_l1393_139307


namespace NUMINAMATH_GPT_untouched_shapes_after_moves_l1393_139348

-- Definitions
def num_shapes : ℕ := 12
def num_triangles : ℕ := 3
def num_squares : ℕ := 4
def num_pentagons : ℕ := 5
def total_moves : ℕ := 10
def petya_moves_first : Prop := True
def vasya_strategy : Prop := True  -- Vasya's strategy to minimize untouched shapes
def petya_strategy : Prop := True  -- Petya's strategy to maximize untouched shapes

-- Theorem
theorem untouched_shapes_after_moves : num_shapes = 12 ∧ num_triangles = 3 ∧ num_squares = 4 ∧ num_pentagons = 5 ∧
                                        total_moves = 10 ∧ petya_moves_first ∧ vasya_strategy ∧ petya_strategy → 
                                        num_shapes - 5 = 6 :=
by
  sorry

end NUMINAMATH_GPT_untouched_shapes_after_moves_l1393_139348


namespace NUMINAMATH_GPT_dawson_marks_l1393_139398

theorem dawson_marks :
  ∀ (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) (M : ℕ),
  max_marks = 220 →
  passing_percentage = 30 →
  failed_by = 36 →
  M = (passing_percentage * max_marks / 100) - failed_by →
  M = 30 := by
  intros max_marks passing_percentage failed_by M h_max h_percent h_failed h_M
  rw [h_max, h_percent, h_failed] at h_M
  norm_num at h_M
  exact h_M

end NUMINAMATH_GPT_dawson_marks_l1393_139398


namespace NUMINAMATH_GPT_shaded_area_l1393_139395

-- Definitions based on given conditions
def Rectangle (A B C D : ℝ) := True -- Placeholder for the geometric definition of a rectangle

-- Total area of the non-shaded part
def non_shaded_area : ℝ := 10

-- Problem statement in Lean
theorem shaded_area (A B C D : ℝ) :
  Rectangle A B C D →
  (exists shaded_area : ℝ, shaded_area = 14 ∧ non_shaded_area + shaded_area = A * B) :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l1393_139395


namespace NUMINAMATH_GPT_perpendicular_line_sum_l1393_139300

theorem perpendicular_line_sum (a b c : ℝ) (h1 : a + 4 * c - 2 = 0) (h2 : 2 - 5 * c + b = 0) 
  (perpendicular : (a / -4) * (2 / 5) = -1) : a + b + c = -4 := 
sorry

end NUMINAMATH_GPT_perpendicular_line_sum_l1393_139300


namespace NUMINAMATH_GPT_correct_transformation_l1393_139315

theorem correct_transformation (a b c : ℝ) (h : (b / (a^2 + 1)) > (c / (a^2 + 1))) : b > c :=
by {
  -- Placeholder proof
  sorry
}

end NUMINAMATH_GPT_correct_transformation_l1393_139315


namespace NUMINAMATH_GPT_postcards_remainder_l1393_139328

theorem postcards_remainder :
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  total % 15 = 3 :=
by
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  show total % 15 = 3
  sorry

end NUMINAMATH_GPT_postcards_remainder_l1393_139328


namespace NUMINAMATH_GPT_gcd_360_128_is_8_l1393_139306

def gcd_360_128 : ℕ :=
  gcd 360 128

theorem gcd_360_128_is_8 : gcd_360_128 = 8 :=
  by
    -- Proof goes here (use sorry for now)
    sorry

end NUMINAMATH_GPT_gcd_360_128_is_8_l1393_139306


namespace NUMINAMATH_GPT_dave_paid_4_more_than_doug_l1393_139355

theorem dave_paid_4_more_than_doug :
  let slices := 8
  let plain_cost := 8
  let anchovy_additional_cost := 2
  let total_cost := plain_cost + anchovy_additional_cost
  let cost_per_slice := total_cost / slices
  let dave_slices := 5
  let doug_slices := slices - dave_slices
  -- Calculate payments
  let dave_payment := dave_slices * cost_per_slice
  let doug_payment := doug_slices * cost_per_slice
  dave_payment - doug_payment = 4 :=
by
  sorry

end NUMINAMATH_GPT_dave_paid_4_more_than_doug_l1393_139355


namespace NUMINAMATH_GPT_original_fraction_l1393_139358

theorem original_fraction (n d : ℝ) (h1 : n + d = 5.25) (h2 : (n + 3) / (2 * d) = 1 / 3) : n / d = 2 / 33 :=
by
  sorry

end NUMINAMATH_GPT_original_fraction_l1393_139358


namespace NUMINAMATH_GPT_n_digit_numbers_modulo_3_l1393_139303

def a (i : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then if i = 0 then 1 else 0 else 2 * a i (n - 1) + a ((i + 1) % 3) (n - 1) + a ((i + 2) % 3) (n - 1)

theorem n_digit_numbers_modulo_3 (n : ℕ) (h : 0 < n) : 
  (a 0 n) = (4^n + 2) / 3 :=
sorry

end NUMINAMATH_GPT_n_digit_numbers_modulo_3_l1393_139303


namespace NUMINAMATH_GPT_train_passing_platform_time_l1393_139393

theorem train_passing_platform_time
  (L_train : ℝ) (L_plat : ℝ) (time_to_cross_tree : ℝ) (time_to_pass_platform : ℝ)
  (H1 : L_train = 2400) 
  (H2 : L_plat = 800)
  (H3 : time_to_cross_tree = 60) :
  time_to_pass_platform = 80 :=
by
  -- add proof here
  sorry

end NUMINAMATH_GPT_train_passing_platform_time_l1393_139393


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1393_139316

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  ((a > 2) ∧ (b > 2) → (a + b > 4)) ∧ ¬((a + b > 4) → (a > 2) ∧ (b > 2)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1393_139316


namespace NUMINAMATH_GPT_gaussian_guardians_total_points_l1393_139308

theorem gaussian_guardians_total_points :
  let daniel := 7
  let curtis := 8
  let sid := 2
  let emily := 11
  let kalyn := 6
  let hyojeong := 12
  let ty := 1
  let winston := 7
  daniel + curtis + sid + emily + kalyn + hyojeong + ty + winston = 54 := by
  sorry

end NUMINAMATH_GPT_gaussian_guardians_total_points_l1393_139308


namespace NUMINAMATH_GPT_prime_product_2002_l1393_139325

theorem prime_product_2002 {a b c d : ℕ} (ha_prime : Prime a) (hb_prime : Prime b) (hc_prime : Prime c) (hd_prime : Prime d)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : a + c = d)
  (h2 : a * (a + b + c + d) = c * (d - b))
  (h3 : 1 + b * c + d = b * d) :
  a * b * c * d = 2002 := 
by 
  sorry

end NUMINAMATH_GPT_prime_product_2002_l1393_139325


namespace NUMINAMATH_GPT_find_distance_PQ_of_polar_coords_l1393_139331

theorem find_distance_PQ_of_polar_coords (α β : ℝ) (h : β - α = 2 * Real.pi / 3) :
  let P := (5, α)
  let Q := (12, β)
  dist P Q = Real.sqrt 229 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_PQ_of_polar_coords_l1393_139331


namespace NUMINAMATH_GPT_conditional_probability_l1393_139310

variable (pA pB pAB : ℝ)
variable (h1 : pA = 0.2)
variable (h2 : pB = 0.18)
variable (h3 : pAB = 0.12)

theorem conditional_probability : (pAB / pB = 2 / 3) :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_conditional_probability_l1393_139310


namespace NUMINAMATH_GPT_multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l1393_139399

def y : ℕ := 48 + 72 + 144 + 192 + 336 + 384 + 3072

theorem multiple_of_4 : ∃ k : ℕ, y = 4 * k := by
  sorry

theorem multiple_of_8 : ∃ k : ℕ, y = 8 * k := by
  sorry

theorem not_multiple_of_16 : ¬ ∃ k : ℕ, y = 16 * k := by
  sorry

theorem multiple_of_24 : ∃ k : ℕ, y = 24 * k := by
  sorry

end NUMINAMATH_GPT_multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l1393_139399


namespace NUMINAMATH_GPT_interval_between_doses_l1393_139305

noncomputable def dose_mg : ℕ := 2 * 375

noncomputable def total_mg_per_day : ℕ := 3000

noncomputable def hours_in_day : ℕ := 24

noncomputable def doses_per_day := total_mg_per_day / dose_mg

noncomputable def hours_between_doses := hours_in_day / doses_per_day

theorem interval_between_doses : hours_between_doses = 6 :=
by
  sorry

end NUMINAMATH_GPT_interval_between_doses_l1393_139305


namespace NUMINAMATH_GPT_value_of_x2_plus_inv_x2_l1393_139365

theorem value_of_x2_plus_inv_x2 (x : ℝ) (hx : x ≠ 0) (h : x^4 + 1 / x^4 = 47) : x^2 + 1 / x^2 = 7 :=
sorry

end NUMINAMATH_GPT_value_of_x2_plus_inv_x2_l1393_139365
