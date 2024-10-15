import Mathlib

namespace NUMINAMATH_GPT_little_john_initial_money_l2194_219452

def sweets_cost : ℝ := 2.25
def friends_donation : ℝ := 2 * 2.20
def money_left : ℝ := 3.85

theorem little_john_initial_money :
  sweets_cost + friends_donation + money_left = 10.50 :=
by
  sorry

end NUMINAMATH_GPT_little_john_initial_money_l2194_219452


namespace NUMINAMATH_GPT_inequality_inequality_l2194_219497

theorem inequality_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b) ^ 2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b) ^ 2 / (8 * b) :=
sorry

end NUMINAMATH_GPT_inequality_inequality_l2194_219497


namespace NUMINAMATH_GPT_joan_mortgage_payoff_l2194_219420

/-- Joan's mortgage problem statement. -/
theorem joan_mortgage_payoff (a r : ℕ) (total : ℕ) (n : ℕ) : a = 100 → r = 3 → total = 12100 → 
    total = a * (1 - r^n) / (1 - r) → n = 5 :=
by intros ha hr htotal hgeom; sorry

end NUMINAMATH_GPT_joan_mortgage_payoff_l2194_219420


namespace NUMINAMATH_GPT_abes_age_l2194_219456

theorem abes_age (A : ℕ) (h : A + (A - 7) = 29) : A = 18 :=
by
  sorry

end NUMINAMATH_GPT_abes_age_l2194_219456


namespace NUMINAMATH_GPT_part1_part2_part3_l2194_219493

def is_beautiful_point (x y : ℝ) (a b : ℝ) : Prop :=
  a = -x ∧ b = x - y

def beautiful_points (x y : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let a := -x
  let b := x - y
  ((a, b), (b, a))

theorem part1 (x y : ℝ) (h : (x, y) = (4, 1)) :
  beautiful_points x y = ((-4, 3), (3, -4)) := by
  sorry

theorem part2 (x y : ℝ) (h : x = 2) (h' : (-x = 2 - y)) :
  y = 4 := by
  sorry

theorem part3 (x y : ℝ) (h : ((-x, x-y) = (-2, 7)) ∨ ((x-y, -x) = (-2, 7))) :
  (x = 2 ∧ y = -5) ∨ (x = -7 ∧ y = -5) := by
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l2194_219493


namespace NUMINAMATH_GPT_find_k_b_l2194_219410

-- Define the sets A and B
def A : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }
def B : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }

-- Define the mapping f
def f (p : ℝ × ℝ) (k b : ℝ) : ℝ × ℝ := (k * p.1, p.2 + b)

-- Define the conditions
def condition (f : (ℝ × ℝ) → ℝ × ℝ) :=
  f (3,1) = (6,2)

-- Statement: Prove that the values of k and b are 2 and 1 respectively
theorem find_k_b : ∃ (k b : ℝ), f (3, 1) k b = (6, 2) ∧ k = 2 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_b_l2194_219410


namespace NUMINAMATH_GPT_reciprocal_of_neg_2023_l2194_219442

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_2023_l2194_219442


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2194_219447

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2194_219447


namespace NUMINAMATH_GPT_find_common_ratio_l2194_219444

noncomputable def common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) : ℝ :=
3

theorem find_common_ratio 
( a : ℕ → ℝ) 
( d : ℝ) 
(h1 : d ≠ 0)
(h2 : ∀ n, a (n + 1) = a n + d)
(h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) :
common_ratio_of_geometric_sequence a d h1 h2 h3 = 3 :=
sorry

end NUMINAMATH_GPT_find_common_ratio_l2194_219444


namespace NUMINAMATH_GPT_kibble_consumption_rate_l2194_219449

-- Kira fills her cat's bowl with 3 pounds of kibble before going to work.
def initial_kibble : ℚ := 3

-- There is still 1 pound left when she returns.
def remaining_kibble : ℚ := 1

-- Kira was away from home for 8 hours.
def time_away : ℚ := 8

-- Calculate the amount of kibble eaten
def kibble_eaten : ℚ := initial_kibble - remaining_kibble

-- Calculate the rate of consumption (hours per pound)
def rate_of_consumption (time: ℚ) (kibble: ℚ) : ℚ := time / kibble

-- Theorem statement: It takes 4 hours for Kira's cat to eat a pound of kibble.
theorem kibble_consumption_rate : rate_of_consumption time_away kibble_eaten = 4 := by
  sorry

end NUMINAMATH_GPT_kibble_consumption_rate_l2194_219449


namespace NUMINAMATH_GPT_sum_of_s_and_t_eq_neg11_l2194_219435

theorem sum_of_s_and_t_eq_neg11 (s t : ℝ) 
  (h1 : ∀ x, x = 3 → x^2 + s * x + t = 0)
  (h2 : ∀ x, x = -4 → x^2 + s * x + t = 0) :
  s + t = -11 :=
sorry

end NUMINAMATH_GPT_sum_of_s_and_t_eq_neg11_l2194_219435


namespace NUMINAMATH_GPT_min_difference_xue_jie_ti_neng_li_l2194_219407

theorem min_difference_xue_jie_ti_neng_li : 
  ∀ (shu hsue jie ti neng li zhan shi : ℕ), 
  shu = 8 ∧ hsue = 1 ∧ jie = 4 ∧ ti = 3 ∧ neng = 9 ∧ li = 5 ∧ zhan = 7 ∧ shi = 2 →
  (shu * 1000 + hsue * 100 + jie * 10 + ti) = 1842 →
  (neng * 10 + li) = 95 →
  1842 - 95 = 1747 := 
by
  intros shu hsue jie ti neng li zhan shi h_digits h_xue_jie_ti h_neng_li
  sorry

end NUMINAMATH_GPT_min_difference_xue_jie_ti_neng_li_l2194_219407


namespace NUMINAMATH_GPT_janet_additional_money_needed_l2194_219406

def janet_savings : ℕ := 2225
def monthly_rent : ℕ := 1250
def advance_months : ℕ := 2
def deposit : ℕ := 500

theorem janet_additional_money_needed :
  (advance_months * monthly_rent + deposit - janet_savings) = 775 :=
by
  sorry

end NUMINAMATH_GPT_janet_additional_money_needed_l2194_219406


namespace NUMINAMATH_GPT_maria_needs_flour_l2194_219400

-- Definitions from conditions
def cups_of_flour_per_cookie (c : ℕ) (f : ℚ) : ℚ := f / c

def total_cups_of_flour (cps_per_cookie : ℚ) (num_cookies : ℕ) : ℚ := cps_per_cookie * num_cookies

-- Given values
def cookies_20 := 20
def flour_3 := 3
def cookies_100 := 100

theorem maria_needs_flour :
  total_cups_of_flour (cups_of_flour_per_cookie cookies_20 flour_3) cookies_100 = 15 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_maria_needs_flour_l2194_219400


namespace NUMINAMATH_GPT_region_in_plane_l2194_219492

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

theorem region_in_plane (x y : ℝ) :
  (f x + f y ≤ 0) ∧ (f x - f y ≥ 0) ↔
  ((x - 3)^2 + (y - 3)^2 ≤ 8) ∧ ((x ≥ y ∧ x + y ≥ 6) ∨ (x ≤ y ∧ x + y ≤ 6)) :=
by
  sorry

end NUMINAMATH_GPT_region_in_plane_l2194_219492


namespace NUMINAMATH_GPT_range_of_a_l2194_219437

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * x + (a - 1) * Real.log x

theorem range_of_a (a : ℝ) (h1 : 1 < a) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 > x2 → f a x1 - f a x2 > x2 - x1) ↔ (1 < a ∧ a ≤ 5) :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_range_of_a_l2194_219437


namespace NUMINAMATH_GPT_arithmetic_sequence_d_range_l2194_219411

theorem arithmetic_sequence_d_range (d : ℝ) :
  (10 + 4 * d > 0) ∧ (10 + 5 * d < 0) ↔ (-5/2 < d) ∧ (d < -2) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_d_range_l2194_219411


namespace NUMINAMATH_GPT_number_of_students_without_A_l2194_219417

theorem number_of_students_without_A (total_students : ℕ) (A_chemistry : ℕ) (A_physics : ℕ) (A_both : ℕ) (h1 : total_students = 40)
    (h2 : A_chemistry = 10) (h3 : A_physics = 18) (h4 : A_both = 5) :
    total_students - (A_chemistry + A_physics - A_both) = 17 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_students_without_A_l2194_219417


namespace NUMINAMATH_GPT_must_divide_l2194_219441

-- Proving 5 is a divisor of q

variables {p q r s : ℕ}

theorem must_divide (h1 : Nat.gcd p q = 30) (h2 : Nat.gcd q r = 42)
                   (h3 : Nat.gcd r s = 66) (h4 : 80 < Nat.gcd s p)
                   (h5 : Nat.gcd s p < 120) :
                   5 ∣ q :=
sorry

end NUMINAMATH_GPT_must_divide_l2194_219441


namespace NUMINAMATH_GPT_quadratic_no_real_roots_iff_l2194_219464

theorem quadratic_no_real_roots_iff (m : ℝ) : (∀ x : ℝ, x^2 + 3 * x + m ≠ 0) ↔ m > 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_iff_l2194_219464


namespace NUMINAMATH_GPT_smallest_percent_increase_from_2_to_3_l2194_219471

def percent_increase (initial final : ℕ) : ℚ := 
  ((final - initial : ℕ) : ℚ) / (initial : ℕ) * 100

def value_at_question : ℕ → ℕ
| 1 => 100
| 2 => 200
| 3 => 300
| 4 => 500
| 5 => 1000
| 6 => 2000
| 7 => 4000
| 8 => 8000
| 9 => 16000
| 10 => 32000
| 11 => 64000
| 12 => 125000
| 13 => 250000
| 14 => 500000
| 15 => 1000000
| _ => 0  -- Default case for questions out of range

theorem smallest_percent_increase_from_2_to_3 :
  let p1 := percent_increase (value_at_question 1) (value_at_question 2)
  let p2 := percent_increase (value_at_question 2) (value_at_question 3)
  let p3 := percent_increase (value_at_question 3) (value_at_question 4)
  let p11 := percent_increase (value_at_question 11) (value_at_question 12)
  let p14 := percent_increase (value_at_question 14) (value_at_question 15)
  p2 < p1 ∧ p2 < p3 ∧ p2 < p11 ∧ p2 < p14 :=
by
  sorry

end NUMINAMATH_GPT_smallest_percent_increase_from_2_to_3_l2194_219471


namespace NUMINAMATH_GPT_digits_interchanged_l2194_219474

theorem digits_interchanged (a b k : ℤ) (h : 10 * a + b = k * (a + b) + 2) :
  10 * b + a = (k + 9) * (a + b) + 2 :=
by
  sorry

end NUMINAMATH_GPT_digits_interchanged_l2194_219474


namespace NUMINAMATH_GPT_snow_at_least_once_l2194_219476

noncomputable def prob_snow_at_least_once (p1 p2 p3: ℚ) : ℚ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

theorem snow_at_least_once : 
  prob_snow_at_least_once (1/2) (2/3) (3/4) = 23 / 24 := 
by
  sorry

end NUMINAMATH_GPT_snow_at_least_once_l2194_219476


namespace NUMINAMATH_GPT_not_square_n5_plus_7_l2194_219416

theorem not_square_n5_plus_7 (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, k^2 = n^5 + 7 := 
by
  sorry

end NUMINAMATH_GPT_not_square_n5_plus_7_l2194_219416


namespace NUMINAMATH_GPT_incorrect_option_l2194_219423

noncomputable def f : ℝ → ℝ := sorry
def is_odd (g : ℝ → ℝ) := ∀ x, g (-(2 * x + 1)) = -g (2 * x + 1)
def is_even (g : ℝ → ℝ) := ∀ x, g (x + 2) = g (-x + 2)

theorem incorrect_option (h₁ : is_odd f) (h₂ : is_even f) (h₃ : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = 3 - x) :
  ¬ (∀ x, f x = f (-x - 2)) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_option_l2194_219423


namespace NUMINAMATH_GPT_possible_values_of_ratio_l2194_219448

theorem possible_values_of_ratio (a d : ℝ) (h : a ≠ 0) (h_eq : a^2 - 6 * a * d + 8 * d^2 = 0) : 
  ∃ x : ℝ, (x = 1/2 ∨ x = 1/4) ∧ x = d/a :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_ratio_l2194_219448


namespace NUMINAMATH_GPT_problem1_problem2_l2194_219431

theorem problem1 :
  (-1)^2 + (Real.pi - 2022)^0 + 2 * Real.sin (60 * Real.pi / 180) - abs (1 - Real.sqrt 3) = 3 :=
by 
  sorry

theorem problem2 (x : ℝ) :
  (2 / (x + 1) + 1 = x / (x - 1)) → x = 3 :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2194_219431


namespace NUMINAMATH_GPT_triangle_area_gt_half_l2194_219424

-- We are given two altitudes h_a and h_b such that both are greater than 1
variables {a h_a h_b : ℝ}

-- Conditions: h_a > 1 and h_b > 1
axiom ha_gt_one : h_a > 1
axiom hb_gt_one : h_b > 1

-- Prove that the area of the triangle is greater than 1/2
theorem triangle_area_gt_half :
  ∃ a : ℝ, a > 1 ∧ ∃ h_a : ℝ, h_a > 1 ∧ (1 / 2) * a * h_a > (1 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_area_gt_half_l2194_219424


namespace NUMINAMATH_GPT_white_tiles_in_square_l2194_219496

theorem white_tiles_in_square :
  ∀ (n : ℕ), (n * n = 81) → (n ^ 2 - (2 * n - 1)) = 6480 :=
by
  intro n
  intro hn
  sorry

end NUMINAMATH_GPT_white_tiles_in_square_l2194_219496


namespace NUMINAMATH_GPT_original_triangle_area_l2194_219498

theorem original_triangle_area (A_new : ℝ) (r : ℝ) (A_original : ℝ) 
  (h1 : r = 3) 
  (h2 : A_new = 54) 
  (h3 : A_new = r^2 * A_original) : 
  A_original = 6 := 
by 
  sorry

end NUMINAMATH_GPT_original_triangle_area_l2194_219498


namespace NUMINAMATH_GPT_tony_running_speed_l2194_219421

theorem tony_running_speed :
  (∀ R : ℝ, (4 / 2 * 60) + 2 * ((4 / R) * 60) = 168 → R = 10) :=
sorry

end NUMINAMATH_GPT_tony_running_speed_l2194_219421


namespace NUMINAMATH_GPT_q_at_4_l2194_219482

def q (x : ℝ) : ℝ := |x - 3|^(1/3) + 3 * |x - 3|^(1/5) + 2 

theorem q_at_4 : q 4 = 6 := by
  sorry

end NUMINAMATH_GPT_q_at_4_l2194_219482


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l2194_219414

theorem arithmetic_seq_sum (a d : ℕ) (S : ℕ → ℕ) (n : ℕ) :
  S 6 = 36 →
  S 12 = 144 →
  S (6 * n) = 576 →
  (∀ m, S m = m * (2 * a + (m - 1) * d) / 2) →
  n = 4 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l2194_219414


namespace NUMINAMATH_GPT_geometric_series_sum_l2194_219455

-- Definition of the geometric sum function in Lean
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r^n) / (1 - r))

-- Specific terms for the problem
def a : ℚ := 2
def r : ℚ := 2 / 5
def n : ℕ := 5

-- The target sum we aim to prove
def target_sum : ℚ := 10310 / 3125

-- The theorem stating that the calculated sum equals the target sum
theorem geometric_series_sum : geometric_sum a r n = target_sum :=
by sorry

end NUMINAMATH_GPT_geometric_series_sum_l2194_219455


namespace NUMINAMATH_GPT_intersection_of_circle_and_line_in_polar_coordinates_l2194_219443

noncomputable section

def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ
def line_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

theorem intersection_of_circle_and_line_in_polar_coordinates :
  ∀ θ ρ, (0 < θ ∧ θ < Real.pi) →
  circle_polar_eq ρ θ →
  line_polar_eq ρ θ →
  ρ = 1 ∧ θ = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_circle_and_line_in_polar_coordinates_l2194_219443


namespace NUMINAMATH_GPT_permutations_sum_divisible_by_37_l2194_219477

theorem permutations_sum_divisible_by_37 (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
    ∃ k, (100 * a + 10 * b + c) + (100 * a + 10 * c + b) + (100 * b + 10 * a + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a) = 37 * k := 
by
  sorry

end NUMINAMATH_GPT_permutations_sum_divisible_by_37_l2194_219477


namespace NUMINAMATH_GPT_Q_value_ratio_l2194_219491

noncomputable def g (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

noncomputable def roots : Fin 2009 → ℂ := sorry -- Define distinct roots s1, s2, ..., s2009

noncomputable def Q (z : ℂ) : ℂ := sorry -- Define the polynomial Q of degree 2009

theorem Q_value_ratio :
  (∀ j : Fin 2009, Q (roots j + 2 / roots j) = 0) →
  (Q (2) / Q (-2) = 361 / 400) :=
sorry

end NUMINAMATH_GPT_Q_value_ratio_l2194_219491


namespace NUMINAMATH_GPT_determine_m_l2194_219419

theorem determine_m 
  (f : ℝ → ℝ) 
  (m : ℕ) 
  (h_nat: 0 < m) 
  (h_f: ∀ x, f x = x ^ (m^2 - 2 * m - 3)) 
  (h_no_intersection: ∀ x, f x ≠ 0) 
  (h_symmetric_origin : ∀ x, f (-x) = -f x) : 
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_l2194_219419


namespace NUMINAMATH_GPT_find_all_k_l2194_219433

theorem find_all_k :
  ∃ (k : ℝ), ∃ (v : ℝ × ℝ), v ≠ 0 ∧ (∃ (v₀ v₁ : ℝ), v = (v₀, v₁) 
  ∧ (3 * v₀ + 6 * v₁) = k * v₀ ∧ (4 * v₀ + 3 * v₁) = k * v₁) 
  ↔ k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6 :=
by
  -- here goes the proof
  sorry

end NUMINAMATH_GPT_find_all_k_l2194_219433


namespace NUMINAMATH_GPT_closed_pipe_length_l2194_219432

def speed_of_sound : ℝ := 333
def fundamental_frequency : ℝ := 440

theorem closed_pipe_length :
  ∃ l : ℝ, l = 0.189 ∧ fundamental_frequency = speed_of_sound / (4 * l) :=
by
  sorry

end NUMINAMATH_GPT_closed_pipe_length_l2194_219432


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2194_219446

theorem solve_equation_1 :
  ∀ x : ℝ, (2 * x - 1) ^ 2 = 9 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, x ^ 2 - 4 * x - 12 = 0 ↔ (x = 6 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2194_219446


namespace NUMINAMATH_GPT_baseball_wins_l2194_219486

-- Define the constants and conditions
def total_games : ℕ := 130
def won_more_than_lost (L W : ℕ) : Prop := W = 3 * L + 14
def total_games_played (L W : ℕ) : Prop := W + L = total_games

-- Define the theorem statement
theorem baseball_wins (L W : ℕ) 
  (h1 : won_more_than_lost L W)
  (h2 : total_games_played L W) : 
  W = 101 :=
  sorry

end NUMINAMATH_GPT_baseball_wins_l2194_219486


namespace NUMINAMATH_GPT_veg_eaters_l2194_219489

variable (n_veg_only n_both : ℕ)

theorem veg_eaters
  (h1 : n_veg_only = 15)
  (h2 : n_both = 11) :
  n_veg_only + n_both = 26 :=
by sorry

end NUMINAMATH_GPT_veg_eaters_l2194_219489


namespace NUMINAMATH_GPT_kevin_total_distance_l2194_219415

def v1 : ℝ := 10
def t1 : ℝ := 0.5
def v2 : ℝ := 20
def t2 : ℝ := 0.5
def v3 : ℝ := 8
def t3 : ℝ := 0.25

theorem kevin_total_distance : v1 * t1 + v2 * t2 + v3 * t3 = 17 := by
  sorry

end NUMINAMATH_GPT_kevin_total_distance_l2194_219415


namespace NUMINAMATH_GPT_smallest_integer_k_l2194_219445

theorem smallest_integer_k (k : ℕ) : 
  (k > 1 ∧ 
   k % 13 = 1 ∧ 
   k % 7 = 1 ∧ 
   k % 5 = 1 ∧ 
   k % 3 = 1) ↔ k = 1366 := 
sorry

end NUMINAMATH_GPT_smallest_integer_k_l2194_219445


namespace NUMINAMATH_GPT_intersection_range_l2194_219430

noncomputable def f (a : ℝ) (x : ℝ) := a * x
noncomputable def g (x : ℝ) := Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := f a x - g x

theorem intersection_range (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = g x1 ∧ f a x2 = g x2) ↔
  0 < a ∧ a < 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_intersection_range_l2194_219430


namespace NUMINAMATH_GPT_Danny_more_wrappers_than_caps_l2194_219439

-- Define the conditions
def bottle_caps_park := 11
def wrappers_park := 28

-- State the theorem representing the problem
theorem Danny_more_wrappers_than_caps:
  wrappers_park - bottle_caps_park = 17 :=
by
  sorry

end NUMINAMATH_GPT_Danny_more_wrappers_than_caps_l2194_219439


namespace NUMINAMATH_GPT_A_inter_B_is_empty_l2194_219478

def A : Set (ℤ × ℤ) := {p | ∃ x : ℤ, p = (x, x + 1)}
def B : Set ℤ := {y | ∃ x : ℤ, y = 2 * x}

theorem A_inter_B_is_empty : A ∩ (fun p => p.2 ∈ B) = ∅ :=
by {
  sorry
}

end NUMINAMATH_GPT_A_inter_B_is_empty_l2194_219478


namespace NUMINAMATH_GPT_polygon_sides_eq_n_l2194_219438

theorem polygon_sides_eq_n
  (sum_except_two_angles : ℝ)
  (angle_equal : ℝ)
  (h1 : sum_except_two_angles = 2970)
  (h2 : angle_equal * 2 < 180)
  : ∃ n : ℕ, 180 * (n - 2) = 2970 + 2 * angle_equal ∧ n = 19 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_n_l2194_219438


namespace NUMINAMATH_GPT_algebraic_expression_value_l2194_219422

theorem algebraic_expression_value (a b : ℝ) (h₁ : a^2 - 3 * a + 1 = 0) (h₂ : b^2 - 3 * b + 1 = 0) :
  (1 / (a^2 + 1) + 1 / (b^2 + 1)) = 1 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2194_219422


namespace NUMINAMATH_GPT_montoya_family_budget_on_food_l2194_219451

def spending_on_groceries : ℝ := 0.6
def spending_on_eating_out : ℝ := 0.2

theorem montoya_family_budget_on_food :
  spending_on_groceries + spending_on_eating_out = 0.8 :=
  by
  sorry

end NUMINAMATH_GPT_montoya_family_budget_on_food_l2194_219451


namespace NUMINAMATH_GPT_at_least_one_woman_probability_l2194_219499

noncomputable def probability_at_least_one_woman_selected 
  (total_men : ℕ) (total_women : ℕ) (selected_people : ℕ) : ℚ :=
  1 - (8 / 12 * 7 / 11 * 6 / 10 * 5 / 9)

theorem at_least_one_woman_probability :
  probability_at_least_one_woman_selected 8 4 4 = 85 / 99 := 
sorry

end NUMINAMATH_GPT_at_least_one_woman_probability_l2194_219499


namespace NUMINAMATH_GPT_min_value_a_plus_b_plus_c_l2194_219404

-- Define the main conditions
variables {a b c : ℝ}
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
variables (h_eq : a^2 + 2*a*b + 4*b*c + 2*c*a = 16)

-- Define the theorem
theorem min_value_a_plus_b_plus_c : 
  (∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (a^2 + 2*a*b + 4*b*c + 2*c*a = 16) → a + b + c ≥ 4) :=
sorry

end NUMINAMATH_GPT_min_value_a_plus_b_plus_c_l2194_219404


namespace NUMINAMATH_GPT_find_side_length_l2194_219494

noncomputable def side_length_of_equilateral_triangle (t : ℝ) (Q : ℝ × ℝ) : Prop :=
  let D := (0, 0)
  let E := (t, 0)
  let F := (t/2, t * (Real.sqrt 3) / 2)
  let DQ := Real.sqrt ((Q.1 - D.1) ^ 2 + (Q.2 - D.2) ^ 2)
  let EQ := Real.sqrt ((Q.1 - E.1) ^ 2 + (Q.2 - E.2) ^ 2)
  let FQ := Real.sqrt ((Q.1 - F.1) ^ 2 + (Q.2 - F.2) ^ 2)
  DQ = 2 ∧ EQ = 2 * Real.sqrt 2 ∧ FQ = 3

theorem find_side_length :
  ∃ t Q, side_length_of_equilateral_triangle t Q → t = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_find_side_length_l2194_219494


namespace NUMINAMATH_GPT_store_earnings_correct_l2194_219428

theorem store_earnings_correct :
  let graphics_cards_sold : ℕ := 10
  let hard_drives_sold : ℕ := 14
  let cpus_sold : ℕ := 8
  let ram_pairs_sold : ℕ := 4
  let graphics_card_price : ℝ := 600
  let hard_drive_price : ℝ := 80
  let cpu_price : ℝ := 200
  let ram_pair_price : ℝ := 60
  graphics_cards_sold * graphics_card_price +
  hard_drives_sold * hard_drive_price +
  cpus_sold * cpu_price +
  ram_pairs_sold * ram_pair_price = 8960 := 
by
  sorry

end NUMINAMATH_GPT_store_earnings_correct_l2194_219428


namespace NUMINAMATH_GPT_compute_expression_l2194_219454

theorem compute_expression (p q : ℝ) (h1 : p + q = 6) (h2 : p * q = 10) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + p * q^3 + p^5 * q^3 = 38676 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_compute_expression_l2194_219454


namespace NUMINAMATH_GPT_intersection_subset_l2194_219495

def set_A : Set ℝ := {x | -4 < x ∧ x < 2}
def set_B : Set ℝ := {x | x > 1 ∨ x < -5}
def set_C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m}

theorem intersection_subset (m : ℝ) :
  (set_A ∩ set_B) ⊆ set_C m ↔ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_subset_l2194_219495


namespace NUMINAMATH_GPT_smallest_four_digit_mod_8_l2194_219484

theorem smallest_four_digit_mod_8 :
  ∃ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 8 = 5 ∧ (∀ y : ℕ, y >= 1000 ∧ y < 10000 ∧ y % 8 = 5 → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_mod_8_l2194_219484


namespace NUMINAMATH_GPT_corn_syrup_content_sport_formulation_l2194_219427

def standard_ratio_flavoring : ℕ := 1
def standard_ratio_corn_syrup : ℕ := 12
def standard_ratio_water : ℕ := 30

def sport_ratio_flavoring_to_corn_syrup : ℕ := 3 * standard_ratio_flavoring
def sport_ratio_flavoring_to_water : ℕ := standard_ratio_flavoring / 2

def sport_ratio_flavoring : ℕ := 1
def sport_ratio_corn_syrup : ℕ := sport_ratio_flavoring * sport_ratio_flavoring_to_corn_syrup
def sport_ratio_water : ℕ := (sport_ratio_flavoring * standard_ratio_water) / 2

def water_content_sport_formulation : ℕ := 30

theorem corn_syrup_content_sport_formulation : 
  (sport_ratio_corn_syrup / sport_ratio_water) * water_content_sport_formulation = 2 :=
by
  sorry

end NUMINAMATH_GPT_corn_syrup_content_sport_formulation_l2194_219427


namespace NUMINAMATH_GPT_population_of_males_l2194_219457

theorem population_of_males (total_population : ℕ) (num_parts : ℕ) (part_population : ℕ) 
  (male_population : ℕ) (female_population : ℕ) (children_population : ℕ) :
  total_population = 600 →
  num_parts = 4 →
  part_population = total_population / num_parts →
  children_population = 2 * male_population →
  male_population = part_population →
  male_population = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_population_of_males_l2194_219457


namespace NUMINAMATH_GPT_sin_cos_identity_l2194_219465

theorem sin_cos_identity (θ : Real) (h1 : 0 < θ ∧ θ < π) (h2 : Real.sin θ * Real.cos θ = - (1/8)) :
  Real.sin (2 * Real.pi + θ) - Real.sin ((Real.pi / 2) - θ) = (Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l2194_219465


namespace NUMINAMATH_GPT_angle_C_measure_ratio_inequality_l2194_219488

open Real

variables (A B C a b c : ℝ)

-- Assumptions
variable (ABC_is_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b)
variable (sin_condition : sin (2 * C - π / 2) = 1/2)
variable (inequality_condition : a^2 + b^2 < c^2)

theorem angle_C_measure :
  0 < C ∧ C < π ∧ C = 2 * π / 3 := sorry

theorem ratio_inequality :
  1 < (a + b) / c ∧ (a + b) / c ≤ 2 * sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_angle_C_measure_ratio_inequality_l2194_219488


namespace NUMINAMATH_GPT_range_of_k_l2194_219440

theorem range_of_k (x y k : ℝ) (h1 : x - y = k - 1) (h2 : 3 * x + 2 * y = 4 * k + 5) (hk : 2 * x + 3 * y > 7) : k > 1 / 3 := 
sorry

end NUMINAMATH_GPT_range_of_k_l2194_219440


namespace NUMINAMATH_GPT_required_fraction_l2194_219463

theorem required_fraction
  (total_members : ℝ)
  (top_10_lists : ℝ) :
  total_members = 775 →
  top_10_lists = 193.75 →
  top_10_lists / total_members = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_required_fraction_l2194_219463


namespace NUMINAMATH_GPT_no_such_n_l2194_219468

theorem no_such_n (n : ℕ) (h_positive : n > 0) : 
  ¬ ∃ k : ℕ, (n^2 + 1) = k * (Nat.floor (Real.sqrt n))^2 + 2 := by
  sorry

end NUMINAMATH_GPT_no_such_n_l2194_219468


namespace NUMINAMATH_GPT_total_amount_shared_l2194_219487

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.20 * z) (hz : z = 100) :
  x + y + z = 370 := by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l2194_219487


namespace NUMINAMATH_GPT_find_hansol_weight_l2194_219429

variable (H : ℕ)

theorem find_hansol_weight (h : H + (H + 4) = 88) : H = 42 :=
by
  sorry

end NUMINAMATH_GPT_find_hansol_weight_l2194_219429


namespace NUMINAMATH_GPT_general_formula_sum_first_n_terms_l2194_219480

-- Definitions for arithmetic sequence, geometric aspects and sum conditions 
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}
variable {b_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Given conditions
axiom sum_condition (S3 S5 : ℕ) : S3 + S5 = 50
axiom common_difference : d ≠ 0
axiom first_term (a1 : ℕ) : a_n 1 = a1
axiom geometric_conditions (a1 a4 a13 : ℕ)
  (h1 : a_n 1 = a1) (h4 : a_n 4 = a4) (h13 : a_n 13 = a13) :
  a4 = a1 + 3 * d ∧ a13 = a1 + 12 * d ∧ (a4 ^ 2 = a1 * a13)

-- Proving the general formula for a_n
theorem general_formula (a_n : ℕ → ℕ)
  (h : ∀ (n : ℕ), a_n n = 2 * n + 1) : 
  a_n n = 2 * n + 1 := 
sorry

-- Proving the sum of the first n terms of sequence {b_n}
theorem sum_first_n_terms (a_n b_n : ℕ → ℕ) (T_n : ℕ → ℕ)
  (h_bn : ∀ (n : ℕ), b_n n = (2 * n + 1) * 2 ^ (n - 1))
  (h_Tn: ∀ (n : ℕ), T_n n = 1 + (2 * n - 1) * 2^n) :
  T_n n = 1 + (2 * n - 1) * 2^n :=
sorry

end NUMINAMATH_GPT_general_formula_sum_first_n_terms_l2194_219480


namespace NUMINAMATH_GPT_sheela_total_income_l2194_219403

-- Define the monthly income as I
def monthly_income (I : Real) : Prop :=
  4500 = 0.28 * I

-- Define the annual income computed from monthly income
def annual_income (I : Real) : Real :=
  I * 12

-- Define the interest earned from savings account 
def interest_savings (principal : Real) (monthly_rate : Real) : Real :=
  principal * (monthly_rate * 12)

-- Define the interest earned from fixed deposit
def interest_fixed (principal : Real) (annual_rate : Real) : Real :=
  principal * annual_rate

-- Overall total income after one year calculation
def overall_total_income (annual_income : Real) (interest_savings : Real) (interest_fixed : Real) : Real :=
  annual_income + interest_savings + interest_fixed

-- Given conditions
variable (I : Real)
variable (principal_savings : Real := 4500)
variable (principal_fixed : Real := 3000)
variable (monthly_rate_savings : Real := 0.02)
variable (annual_rate_fixed : Real := 0.06)

-- Theorem statement to be proved
theorem sheela_total_income :
  monthly_income I →
  overall_total_income (annual_income I) 
                      (interest_savings principal_savings monthly_rate_savings)
                      (interest_fixed principal_fixed annual_rate_fixed)
  = 194117.16 :=
by
  sorry

end NUMINAMATH_GPT_sheela_total_income_l2194_219403


namespace NUMINAMATH_GPT_heaviest_object_can_be_weighed_is_40_number_of_different_weights_is_40_l2194_219466

def weights : List ℕ := [1, 3, 9, 27]

theorem heaviest_object_can_be_weighed_is_40 : 
  List.sum weights = 40 :=
by
  sorry

theorem number_of_different_weights_is_40 :
  List.range (List.sum weights) = List.range 40 :=
by
  sorry

end NUMINAMATH_GPT_heaviest_object_can_be_weighed_is_40_number_of_different_weights_is_40_l2194_219466


namespace NUMINAMATH_GPT_smallest_y_value_smallest_y_value_is_neg6_l2194_219462

theorem smallest_y_value :
  ∀ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) → (y = -3 ∨ y = -6) :=
by
  sorry

theorem smallest_y_value_is_neg6 :
  ∃ y : ℝ, (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) ∧ (y = -6) :=
by
  have H := smallest_y_value
  sorry

end NUMINAMATH_GPT_smallest_y_value_smallest_y_value_is_neg6_l2194_219462


namespace NUMINAMATH_GPT_cost_of_one_dozen_pens_l2194_219490

theorem cost_of_one_dozen_pens (x n : ℕ) (h₁ : 5 * n * x + 5 * x = 200) (h₂ : ∀ p : ℕ, p > 0 → p ≠ x * 5 → x * 5 ≠ x) :
  12 * 5 * x = 120 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_dozen_pens_l2194_219490


namespace NUMINAMATH_GPT_evaluate_expression_l2194_219412

theorem evaluate_expression (x y : ℤ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^4 + 3 * x^2 - 2 * y + 2 * y^2) / 6 = 22 :=
by
  -- Conditions from the problem
  rw [h₁, h₂]
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2194_219412


namespace NUMINAMATH_GPT_percentage_increase_of_gross_l2194_219470

theorem percentage_increase_of_gross
  (P R : ℝ)
  (price_drop : ℝ := 0.20)
  (quantity_increase : ℝ := 0.60)
  (original_gross : ℝ := P * R)
  (new_price : ℝ := (1 - price_drop) * P)
  (new_quantity_sold : ℝ := (1 + quantity_increase) * R)
  (new_gross : ℝ := new_price * new_quantity_sold)
  (percentage_increase : ℝ := ((new_gross - original_gross) / original_gross) * 100) :
  percentage_increase = 28 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_of_gross_l2194_219470


namespace NUMINAMATH_GPT_total_trees_after_planting_l2194_219460

-- Define the initial counts of the trees
def initial_maple_trees : ℕ := 2
def initial_poplar_trees : ℕ := 5
def initial_oak_trees : ℕ := 4

-- Define the planting rules
def maple_trees_planted (initial_maple : ℕ) : ℕ := 3 * initial_maple
def poplar_trees_planted (initial_poplar : ℕ) : ℕ := 3 * initial_poplar

-- Calculate the total number of each type of tree after planting
def total_maple_trees (initial_maple : ℕ) : ℕ :=
  initial_maple + maple_trees_planted initial_maple

def total_poplar_trees (initial_poplar : ℕ) : ℕ :=
  initial_poplar + poplar_trees_planted initial_poplar

def total_oak_trees (initial_oak : ℕ) : ℕ := initial_oak

-- Calculate the total number of trees in the park
def total_trees (initial_maple initial_poplar initial_oak : ℕ) : ℕ :=
  total_maple_trees initial_maple + total_poplar_trees initial_poplar + total_oak_trees initial_oak

-- The proof statement
theorem total_trees_after_planting :
  total_trees initial_maple_trees initial_poplar_trees initial_oak_trees = 32 := 
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_total_trees_after_planting_l2194_219460


namespace NUMINAMATH_GPT_fifth_equation_pattern_l2194_219401

theorem fifth_equation_pattern :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by sorry

end NUMINAMATH_GPT_fifth_equation_pattern_l2194_219401


namespace NUMINAMATH_GPT_box_length_l2194_219426

theorem box_length :
  ∃ (length : ℝ), 
  let box_height := 8
  let box_width := 10
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let num_blocks := 40
  let box_volume := box_height * box_width * length
  let block_volume := block_height * block_width * block_length
  num_blocks * block_volume = box_volume ∧ length = 12 := by
  sorry

end NUMINAMATH_GPT_box_length_l2194_219426


namespace NUMINAMATH_GPT_man_twice_son_age_l2194_219413

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 18) (h2 : M = S + 20) 
  (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  -- Proof steps can be added here later
  sorry

end NUMINAMATH_GPT_man_twice_son_age_l2194_219413


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2194_219434

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}
variable (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * d)
variable (h_S_n : ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2)

theorem arithmetic_sequence_problem
  (h1 : S_n 5 = 2 * a_n 5)
  (h2 : a_n 3 = -4) :
  a_n 9 = -22 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2194_219434


namespace NUMINAMATH_GPT_inverse_propositions_l2194_219405

-- Given conditions
lemma right_angles_equal : ∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90) :=
sorry

lemma equal_angles_right : ∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2) :=
sorry

-- Theorem to be proven
theorem inverse_propositions :
  (∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90)) ↔
  (∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2)) :=
sorry

end NUMINAMATH_GPT_inverse_propositions_l2194_219405


namespace NUMINAMATH_GPT_systematic_sampling_removal_count_l2194_219458

theorem systematic_sampling_removal_count :
  ∀ (N n : ℕ), N = 3204 ∧ n = 80 → N % n = 4 := 
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_removal_count_l2194_219458


namespace NUMINAMATH_GPT_determine_pairs_l2194_219472

theorem determine_pairs (a b : ℕ) (h : 2017^a = b^6 - 32 * b + 1) : 
  (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end NUMINAMATH_GPT_determine_pairs_l2194_219472


namespace NUMINAMATH_GPT_dance_team_recruitment_l2194_219408

theorem dance_team_recruitment 
  (total_students choir_students track_field_students dance_students : ℕ)
  (h1 : total_students = 100)
  (h2 : choir_students = 2 * track_field_students)
  (h3 : dance_students = choir_students + 10)
  (h4 : total_students = track_field_students + choir_students + dance_students) : 
  dance_students = 46 :=
by {
  -- The proof goes here, but it is not required as per instructions
  sorry
}

end NUMINAMATH_GPT_dance_team_recruitment_l2194_219408


namespace NUMINAMATH_GPT_quadratic_range_l2194_219473

noncomputable def f : ℝ → ℝ := sorry -- Quadratic function with a positive coefficient for its quadratic term

axiom symmetry_condition : ∀ x : ℝ, f x = f (4 - x)

theorem quadratic_range (x : ℝ) (h1 : f (1 - 2 * x ^ 2) < f (1 + 2 * x - x ^ 2)) : -2 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_range_l2194_219473


namespace NUMINAMATH_GPT_average_production_l2194_219459

theorem average_production (n : ℕ) :
  let total_past_production := 50 * n
  let total_production_including_today := 100 + total_past_production
  let average_production := total_production_including_today / (n + 1)
  average_production = 55
  -> n = 9 :=
by
  sorry

end NUMINAMATH_GPT_average_production_l2194_219459


namespace NUMINAMATH_GPT_arithmeticSeqModulus_l2194_219436

-- Define the arithmetic sequence
def arithmeticSeqSum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

-- The main theorem to prove
theorem arithmeticSeqModulus : arithmeticSeqSum 2 5 102 % 20 = 12 := by
  sorry

end NUMINAMATH_GPT_arithmeticSeqModulus_l2194_219436


namespace NUMINAMATH_GPT_number_of_boys_in_class_l2194_219475

theorem number_of_boys_in_class 
  (n : ℕ)
  (average_height : ℝ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_average_height : ℝ)
  (initial_average_height : average_height = 185)
  (incorrect_record : incorrect_height = 166)
  (correct_record : correct_height = 106)
  (actual_avg : actual_average_height = 183) 
  (total_height_incorrect : ℝ) 
  (total_height_correct : ℝ) 
  (total_height_eq : total_height_incorrect = 185 * n)
  (correct_total_height_eq : total_height_correct = 185 * n - (incorrect_height - correct_height))
  (actual_total_height_eq : total_height_correct = actual_average_height * n) :
  n = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_in_class_l2194_219475


namespace NUMINAMATH_GPT_parakeets_per_cage_is_2_l2194_219402

variables (cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

def number_of_parakeets_each_cage : ℕ :=
  (total_birds - cages * parrots_per_cage) / cages

theorem parakeets_per_cage_is_2
  (hcages : cages = 4)
  (hparrots_per_cage : parrots_per_cage = 8)
  (htotal_birds : total_birds = 40) :
  number_of_parakeets_each_cage cages parrots_per_cage total_birds = 2 := 
by
  sorry

end NUMINAMATH_GPT_parakeets_per_cage_is_2_l2194_219402


namespace NUMINAMATH_GPT_Kimberley_collected_10_pounds_l2194_219418

variable (K H E total : ℝ)

theorem Kimberley_collected_10_pounds (h_total : total = 35) (h_Houston : H = 12) (h_Ela : E = 13) :
    K + H + E = total → K = 10 :=
by
  intros h_sum
  rw [h_Houston, h_Ela] at h_sum
  linarith

end NUMINAMATH_GPT_Kimberley_collected_10_pounds_l2194_219418


namespace NUMINAMATH_GPT_smallest_multiplier_to_perfect_square_l2194_219467

theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, k > 0 ∧ ∀ m : ℕ, (2010 * m = k * k) → m = 2010 :=
by
  sorry

end NUMINAMATH_GPT_smallest_multiplier_to_perfect_square_l2194_219467


namespace NUMINAMATH_GPT_sum_of_common_ratios_l2194_219469

variable {k p r : ℝ}

-- Condition 1: geometric sequences with distinct common ratios
-- Condition 2: a_3 - b_3 = 3(a_2 - b_2)
def geometric_sequences (k p r : ℝ) : Prop :=
  (k ≠ 0) ∧ (p ≠ r) ∧ (k * p^2 - k * r^2 = 3 * (k * p - k * r))

theorem sum_of_common_ratios (k p r : ℝ) (h : geometric_sequences k p r) : p + r = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_l2194_219469


namespace NUMINAMATH_GPT_car_trip_proof_l2194_219409

def initial_oil_quantity (y : ℕ → ℕ) : Prop :=
  y 0 = 50

def consumption_rate (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = y (t - 1) - 5

def relationship_between_y_and_t (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = 50 - 5 * t

def oil_left_after_8_hours (y : ℕ → ℕ) : Prop :=
  y 8 = 10

theorem car_trip_proof (y : ℕ → ℕ) :
  initial_oil_quantity y ∧ consumption_rate y ∧ relationship_between_y_and_t y ∧ oil_left_after_8_hours y :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_car_trip_proof_l2194_219409


namespace NUMINAMATH_GPT_product_of_series_l2194_219425

theorem product_of_series :
  (1 - 1/2^2) * (1 - 1/3^2) * (1 - 1/4^2) * (1 - 1/5^2) * (1 - 1/6^2) *
  (1 - 1/7^2) * (1 - 1/8^2) * (1 - 1/9^2) * (1 - 1/10^2) = 11 / 20 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_series_l2194_219425


namespace NUMINAMATH_GPT_right_triangle_perimeter_l2194_219481

theorem right_triangle_perimeter 
  (a b c : ℕ) (h : a = 11) (h1 : a * a + b * b = c * c) (h2 : a < c) : a + b + c = 132 :=
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l2194_219481


namespace NUMINAMATH_GPT_actual_distance_is_correct_l2194_219483

noncomputable def actual_distance_in_meters (scale : ℕ) (map_distance_cm : ℝ) : ℝ :=
  (map_distance_cm * scale) / 100

theorem actual_distance_is_correct
  (scale : ℕ)
  (map_distance_cm : ℝ)
  (h_scale : scale = 3000000)
  (h_map_distance : map_distance_cm = 4) :
  actual_distance_in_meters scale map_distance_cm = 1.2 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_is_correct_l2194_219483


namespace NUMINAMATH_GPT_students_playing_both_football_and_cricket_l2194_219450

theorem students_playing_both_football_and_cricket
  (total_students : ℕ)
  (students_playing_football : ℕ)
  (students_playing_cricket : ℕ)
  (students_neither_football_nor_cricket : ℕ) :
  total_students = 250 →
  students_playing_football = 160 →
  students_playing_cricket = 90 →
  students_neither_football_nor_cricket = 50 →
  (students_playing_football + students_playing_cricket - (total_students - students_neither_football_nor_cricket)) = 50 :=
by
  intros h_total h_football h_cricket h_neither
  sorry

end NUMINAMATH_GPT_students_playing_both_football_and_cricket_l2194_219450


namespace NUMINAMATH_GPT_inequality_proof_l2194_219453

theorem inequality_proof (x : ℝ) : 3 * x - 6 > 5 * (x - 2) → x < 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2194_219453


namespace NUMINAMATH_GPT_Karen_wall_paint_area_l2194_219461

theorem Karen_wall_paint_area :
  let height_wall := 10
  let width_wall := 15
  let height_window := 3
  let width_window := 5
  let height_door := 2
  let width_door := 6
  let area_wall := height_wall * width_wall
  let area_window := height_window * width_window
  let area_door := height_door * width_door
  let area_to_paint := area_wall - area_window - area_door
  area_to_paint = 123 := by
{
  sorry
}

end NUMINAMATH_GPT_Karen_wall_paint_area_l2194_219461


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l2194_219485

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_about_x_axis (P : Point3D) : Point3D :=
  { x := P.x, y := -P.y, z := -P.z }

theorem symmetric_point_coordinates :
  symmetric_about_x_axis {x := 1, y := 3, z := 6} = {x := 1, y := -3, z := -6} :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l2194_219485


namespace NUMINAMATH_GPT_new_credit_card_balance_l2194_219479

theorem new_credit_card_balance (i g x r n : ℝ)
    (h_i : i = 126)
    (h_g : g = 60)
    (h_x : x = g / 2)
    (h_r : r = 45)
    (h_n : n = (i + g + x) - r) :
    n = 171 :=
sorry

end NUMINAMATH_GPT_new_credit_card_balance_l2194_219479
