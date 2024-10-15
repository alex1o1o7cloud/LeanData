import Mathlib

namespace NUMINAMATH_GPT_find_interest_rate_l1003_100317

def interest_rate_borrowed (p_borrowed: ℝ) (p_lent: ℝ) (time: ℝ) (rate_lent: ℝ) (gain: ℝ) (r: ℝ) : Prop :=
  let interest_from_ramu := p_lent * rate_lent * time / 100
  let interest_to_anwar := p_borrowed * r * time / 100
  gain = interest_from_ramu - interest_to_anwar

theorem find_interest_rate :
  interest_rate_borrowed 3900 5655 3 9 824.85 5.95 := sorry

end NUMINAMATH_GPT_find_interest_rate_l1003_100317


namespace NUMINAMATH_GPT_storybook_pages_l1003_100354

def reading_start_date := 10
def reading_end_date := 20
def pages_per_day := 11
def number_of_days := reading_end_date - reading_start_date + 1
def total_pages := pages_per_day * number_of_days

theorem storybook_pages : total_pages = 121 := by
  sorry

end NUMINAMATH_GPT_storybook_pages_l1003_100354


namespace NUMINAMATH_GPT_points_lie_on_hyperbola_l1003_100379

theorem points_lie_on_hyperbola (s : ℝ) :
  let x := 2 * (Real.exp s + Real.exp (-s))
  let y := 4 * (Real.exp s - Real.exp (-s))
  (x^2) / 16 - (y^2) / 64 = 1 :=
by
  sorry

end NUMINAMATH_GPT_points_lie_on_hyperbola_l1003_100379


namespace NUMINAMATH_GPT_simplify_expression_l1003_100389

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 3 * (2 - i) + i * (3 + 2 * i) = 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1003_100389


namespace NUMINAMATH_GPT_problem1_problem2_l1003_100302

-- Definition of a double root equation with the given condition
def is_double_root_equation (a b c : ℝ) := 
  ∃ x1 x2 : ℝ, a * x1 = 2 * a * x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0

-- Proving that x² - 6x + 8 = 0 is a double root equation
theorem problem1 : is_double_root_equation 1 (-6) 8 :=
  sorry

-- Proving that if (x-8)(x-n) = 0 is a double root equation, n is either 4 or 16
theorem problem2 (n : ℝ) (h : is_double_root_equation 1 (-8 - n) (8 * n)) :
  n = 4 ∨ n = 16 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1003_100302


namespace NUMINAMATH_GPT_largest_of_five_consecutive_composite_integers_under_40_l1003_100348

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def five_consecutive_composite_integers_under_40 : List ℕ :=
[32, 33, 34, 35, 36]

theorem largest_of_five_consecutive_composite_integers_under_40 :
  ∀ n ∈ five_consecutive_composite_integers_under_40,
  n < 40 ∧ ∀ k, (k ∈ five_consecutive_composite_integers_under_40 →
  ¬ is_prime k) →
  List.maximum five_consecutive_composite_integers_under_40 = some 36 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_five_consecutive_composite_integers_under_40_l1003_100348


namespace NUMINAMATH_GPT_window_side_length_is_five_l1003_100330

def pane_width (x : ℝ) : ℝ := x
def pane_height (x : ℝ) : ℝ := 3 * x
def border_width : ℝ := 1
def pane_rows : ℕ := 2
def pane_columns : ℕ := 3

theorem window_side_length_is_five (x : ℝ) (h : pane_height x = 3 * pane_width x) : 
  (3 * x + 4 = 6 * x + 3) -> (3 * x + 4 = 5) :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_window_side_length_is_five_l1003_100330


namespace NUMINAMATH_GPT_greatest_k_for_quadratic_roots_diff_l1003_100346

theorem greatest_k_for_quadratic_roots_diff (k : ℝ)
  (H : ∀ x: ℝ, (x^2 + k * x + 8 = 0) → (∃ a b : ℝ, a ≠ b ∧ (a - b)^2 = 84)) :
  k = 2 * Real.sqrt 29 :=
by
  sorry

end NUMINAMATH_GPT_greatest_k_for_quadratic_roots_diff_l1003_100346


namespace NUMINAMATH_GPT_in_range_p_1_to_100_l1003_100385

def p (m n : ℤ) : ℤ :=
  2 * m^2 - 6 * m * n + 5 * n^2

-- Predicate that asserts k is in the range of p
def in_range_p (k : ℤ) : Prop :=
  ∃ m n : ℤ, p m n = k

-- Lean statement for the theorem
theorem in_range_p_1_to_100 :
  {k : ℕ | 1 ≤ k ∧ k ≤ 100 ∧ in_range_p k} = 
  {1, 2, 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, 26, 29, 32, 34, 36, 37, 40, 41, 45, 49, 50, 52, 53, 58, 61, 64, 65, 68, 72, 73, 74, 80, 81, 82, 85, 89, 90, 97, 98, 100} :=
  by
    sorry

end NUMINAMATH_GPT_in_range_p_1_to_100_l1003_100385


namespace NUMINAMATH_GPT_probability_king_even_coords_2008_l1003_100324

noncomputable def king_probability_even_coords (turns : ℕ) : ℝ :=
  let p_stay := 0.4
  let p_edge := 0.1
  let p_diag := 0.05
  if turns = 2008 then
    (5 ^ 2008 + 1) / (2 * 5 ^ 2008)
  else
    0 -- default value for other cases

theorem probability_king_even_coords_2008 :
  king_probability_even_coords 2008 = (5 ^ 2008 + 1) / (2 * 5 ^ 2008) :=
by
  sorry

end NUMINAMATH_GPT_probability_king_even_coords_2008_l1003_100324


namespace NUMINAMATH_GPT_find_m_l1003_100333

theorem find_m {A B : Set ℝ} (m : ℝ) :
  (A = {x : ℝ | x^2 + x - 12 = 0}) →
  (B = {x : ℝ | mx + 1 = 0}) →
  (A ∩ B = {3}) →
  m = -1 / 3 := 
by
  intros hA hB h_inter
  sorry

end NUMINAMATH_GPT_find_m_l1003_100333


namespace NUMINAMATH_GPT_Xiao_Ming_min_steps_l1003_100361

-- Problem statement: Prove that the minimum number of steps Xiao Ming needs to move from point A to point B is 5,
-- given his movement pattern and the fact that he can reach eight different positions from point C.

def min_steps_from_A_to_B : ℕ :=
  5

theorem Xiao_Ming_min_steps (A B C : Type) (f : A → B → C) : 
  (min_steps_from_A_to_B = 5) :=
by
  sorry

end NUMINAMATH_GPT_Xiao_Ming_min_steps_l1003_100361


namespace NUMINAMATH_GPT_original_price_is_135_l1003_100396

-- Problem Statement:
variable (P : ℝ)  -- Let P be the original price of the potion

-- Conditions
axiom potion_cost : (1 / 15) * P = 9

-- Proof Goal
theorem original_price_is_135 : P = 135 :=
by
  sorry

end NUMINAMATH_GPT_original_price_is_135_l1003_100396


namespace NUMINAMATH_GPT_tank_capacity_l1003_100375

theorem tank_capacity (C : ℝ) (h : (3 / 4) * C + 9 = (7 / 8) * C) : C = 72 :=
sorry

end NUMINAMATH_GPT_tank_capacity_l1003_100375


namespace NUMINAMATH_GPT_max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l1003_100357

noncomputable def max_perimeter_of_right_angled_quadrilateral (r : ℝ) : ℝ :=
  4 * r * Real.sqrt 2

theorem max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2
  (r : ℝ) :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 4 * r^2 → 2 * (x + y) ≤ max_perimeter_of_right_angled_quadrilateral r)
  ∧ (k = max_perimeter_of_right_angled_quadrilateral r) :=
sorry

end NUMINAMATH_GPT_max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l1003_100357


namespace NUMINAMATH_GPT_g_of_f_neg_5_l1003_100352

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 8

-- Assume g(42) = 17
axiom g_f_5_eq_17 : ∀ (g : ℝ → ℝ), g (f 5) = 17

-- State the theorem to be proven
theorem g_of_f_neg_5 (g : ℝ → ℝ) : g (f (-5)) = 17 :=
by
  sorry

end NUMINAMATH_GPT_g_of_f_neg_5_l1003_100352


namespace NUMINAMATH_GPT_calc_value_l1003_100395

theorem calc_value : 2 + 3 * 4 - 5 + 6 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_calc_value_l1003_100395


namespace NUMINAMATH_GPT_additional_people_needed_l1003_100387

-- Define the conditions
def num_people_initial := 9
def work_done_initial := 3 / 5
def days_initial := 14
def days_remaining := 4

-- Calculated values based on conditions
def work_rate_per_person : ℚ :=
  work_done_initial / (num_people_initial * days_initial)

def work_remaining : ℚ := 1 - work_done_initial

def total_people_needed : ℚ :=
  work_remaining / (work_rate_per_person * days_remaining)

-- Formulate the statement to prove
theorem additional_people_needed :
  total_people_needed - num_people_initial = 12 :=
by
  sorry

end NUMINAMATH_GPT_additional_people_needed_l1003_100387


namespace NUMINAMATH_GPT_prism_pyramid_sum_l1003_100380

theorem prism_pyramid_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  let new_faces := faces - 1 + 4
  let new_edges := edges + 4
  let new_vertices := vertices + 1
  new_faces + new_edges + new_vertices = 34 :=
by
  sorry

end NUMINAMATH_GPT_prism_pyramid_sum_l1003_100380


namespace NUMINAMATH_GPT_min_abc_value_l1003_100341

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem min_abc_value
  (a b c : ℕ)
  (h1: is_prime a)
  (h2 : is_prime b)
  (h3 : is_prime c)
  (h4 : a^5 ∣ (b^2 - c))
  (h5 : ∃ k : ℕ, (b + c) = k^2) :
  a * b * c = 1958 := sorry

end NUMINAMATH_GPT_min_abc_value_l1003_100341


namespace NUMINAMATH_GPT_sum_of_squares_l1003_100315

variable {x y : ℝ}

theorem sum_of_squares (h1 : x + y = 20) (h2 : x * y = 100) : x^2 + y^2 = 200 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l1003_100315


namespace NUMINAMATH_GPT_age_difference_l1003_100365

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 18) : A = C + 18 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1003_100365


namespace NUMINAMATH_GPT_consecutive_days_without_meeting_l1003_100367

/-- In March 1987, there are 31 days, starting on a Sunday.
There are 11 club meetings to be held, and no meetings are on Saturdays or Sundays.
This theorem proves that there will be at least three consecutive days without a meeting. -/
theorem consecutive_days_without_meeting (meetings : Finset ℕ) :
  (∀ x ∈ meetings, 1 ≤ x ∧ x ≤ 31 ∧ ¬ ∃ k, x = 7 * k + 1 ∨ x = 7 * k + 2) →
  meetings.card = 11 →
  ∃ i, 1 ≤ i ∧ i + 2 ≤ 31 ∧ ¬ (i ∈ meetings ∨ (i + 1) ∈ meetings ∨ (i + 2) ∈ meetings) :=
by
  sorry

end NUMINAMATH_GPT_consecutive_days_without_meeting_l1003_100367


namespace NUMINAMATH_GPT_square_area_relation_l1003_100328

variable {lA lB : ℝ}

theorem square_area_relation (h : lB = 4 * lA) : lB^2 = 16 * lA^2 :=
by sorry

end NUMINAMATH_GPT_square_area_relation_l1003_100328


namespace NUMINAMATH_GPT_binom_squared_l1003_100364

theorem binom_squared :
  (Nat.choose 12 11) ^ 2 = 144 := 
by
  -- Mathematical steps would go here.
  sorry

end NUMINAMATH_GPT_binom_squared_l1003_100364


namespace NUMINAMATH_GPT_original_number_of_professors_l1003_100390

theorem original_number_of_professors (p : ℕ) 
  (h1 : 6480 % p = 0) 
  (h2 : 11200 % (p + 3) = 0) 
  (h3 : 6480 / p < 11200 / (p + 3))
  (h4 : 5 ≤ p) : 
  p = 5 :=
by {
  -- The body of the proof goes here.
  sorry
}

end NUMINAMATH_GPT_original_number_of_professors_l1003_100390


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1003_100358

variable (x y : ℚ)

theorem simplify_and_evaluate_expression :
    x = 2 / 15 → y = 3 / 2 → 
    (2 * x + y)^2 - (3 * x - y)^2 + 5 * x * (x - y) = 1 :=
by 
  intros h1 h2
  subst h1
  subst h2
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1003_100358


namespace NUMINAMATH_GPT_sequence_converges_to_one_l1003_100309

noncomputable def u (n : ℕ) : ℝ :=
1 + (Real.sin n) / n

theorem sequence_converges_to_one :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n - 1| ≤ ε :=
sorry

end NUMINAMATH_GPT_sequence_converges_to_one_l1003_100309


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1003_100399

theorem distance_between_A_and_B (x : ℝ) (boat_speed : ℝ) (flow_speed : ℝ) (dist_AC : ℝ) (total_time : ℝ) :
  (boat_speed = 8) →
  (flow_speed = 2) →
  (dist_AC = 2) →
  (total_time = 3) →
  (x = 10 ∨ x = 12.5) :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_between_A_and_B_l1003_100399


namespace NUMINAMATH_GPT_archer_total_fish_caught_l1003_100356

noncomputable def total_fish_caught (initial : ℕ) (second_extra : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := initial + second_extra
  let total_after_two_rounds := initial + second_round
  let third_round := second_round + (third_round_percentage * second_round) / 100
  total_after_two_rounds + third_round

theorem archer_total_fish_caught :
  total_fish_caught 8 12 60 = 60 :=
by
  -- Theorem statement to prove the total fish caught equals 60 given the conditions.
  sorry

end NUMINAMATH_GPT_archer_total_fish_caught_l1003_100356


namespace NUMINAMATH_GPT_exactly_two_statements_true_l1003_100344

noncomputable def f : ℝ → ℝ := sorry -- Definition of f satisfying the conditions

-- Conditions
axiom functional_eq (x : ℝ) : f (x + 3/2) + f x = 0
axiom odd_function (x : ℝ) : f (- x - 3/4) = - f (x - 3/4)

-- Proof statement
theorem exactly_two_statements_true : 
  (¬(∀ (T : ℝ), T > 0 → (∀ (x : ℝ), f (x + T) = f x) → T = 3/2) ∧
   (∀ (x : ℝ), f (-x - 3/4) = - f (x - 3/4)) ∧
   (¬(∀ (x : ℝ), f x = f (-x)))) :=
sorry

end NUMINAMATH_GPT_exactly_two_statements_true_l1003_100344


namespace NUMINAMATH_GPT_verify_incorrect_operation_l1003_100376

theorem verify_incorrect_operation (a : ℝ) :
  ¬ ((-a^2)^3 = -a^5) :=
by
  sorry

end NUMINAMATH_GPT_verify_incorrect_operation_l1003_100376


namespace NUMINAMATH_GPT_driving_time_in_fog_is_correct_l1003_100334

-- Define constants for speeds (in miles per minute)
def speed_sunny : ℚ := 35 / 60
def speed_rain : ℚ := 25 / 60
def speed_fog : ℚ := 15 / 60

-- Total distance and time
def total_distance : ℚ := 19.5
def total_time : ℚ := 45

-- Time variables for rain and fog
variables (t_r t_f : ℚ)

-- Define the driving distance equation
def distance_eq : Prop :=
  speed_sunny * (total_time - t_r - t_f) + speed_rain * t_r + speed_fog * t_f = total_distance

-- Prove the time driven in fog equals 10.25 minutes
theorem driving_time_in_fog_is_correct (h : distance_eq t_r t_f) : t_f = 10.25 :=
sorry

end NUMINAMATH_GPT_driving_time_in_fog_is_correct_l1003_100334


namespace NUMINAMATH_GPT_votes_distribution_l1003_100321

theorem votes_distribution (W : ℕ) 
  (h1 : W + (W - 53) + (W - 79) + (W - 105) = 963) 
  : W = 300 ∧ 247 = W - 53 ∧ 221 = W - 79 ∧ 195 = W - 105 :=
by
  sorry

end NUMINAMATH_GPT_votes_distribution_l1003_100321


namespace NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1987_l1003_100397

theorem rightmost_three_digits_of_7_pow_1987 :
  7^1987 % 1000 = 543 :=
by
  sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1987_l1003_100397


namespace NUMINAMATH_GPT_university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l1003_100347

-- Part 1
def probability_A_exactly_one_subject : ℚ :=
  3 * (1/2) * (1/2)^2

def probability_B_exactly_one_subject (m : ℚ) : ℚ :=
  (1/6) * (2/5)^2 + (5/6) * (3/5) * (2/5) * 2

theorem university_A_pass_one_subject : probability_A_exactly_one_subject = 3/8 :=
sorry

theorem university_B_pass_one_subject_when_m_3_5 : probability_B_exactly_one_subject (3/5) = 32/75 :=
sorry

-- Part 2
def expected_A : ℚ :=
  3 * (1/2)

def expected_B (m : ℚ) : ℚ :=
  ((17 - 7 * m) / 30) + (2 * (3 + 14 * m) / 30) + (3 * m / 10)

theorem preferred_range_of_m : 0 < m ∧ m < 11/15 → expected_A > expected_B m :=
sorry

end NUMINAMATH_GPT_university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l1003_100347


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_ratio_l1003_100307

section
variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables {d : ℝ}

-- Definition of the arithmetic sequence and its sum
def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
  ∀ n, a n = a₁ + (n - 1) * d

def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given conditions
-- 1. S is the sum of the first n terms of the arithmetic sequence a
axiom sn_arith_seq : sum_arithmetic_sequence S a

-- 2. a_1, a_3, and a_4 form a geometric sequence
axiom geom_seq : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)

-- Goal is to prove the given ratio equation
theorem arithmetic_geometric_sequence_ratio (h : ∀ n, a n = -4 * d + (n - 1) * d) :
  (S 3 - S 2) / (S 5 - S 3) = 2 :=
sorry
end

end NUMINAMATH_GPT_arithmetic_geometric_sequence_ratio_l1003_100307


namespace NUMINAMATH_GPT_arithmetic_progression_x_value_l1003_100329

theorem arithmetic_progression_x_value (x: ℝ) (h1: 3*x - 1 - (2*x - 3) = 4*x + 1 - (3*x - 1)) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_x_value_l1003_100329


namespace NUMINAMATH_GPT_gcd_a_b_l1003_100304

def a : ℕ := 333333333
def b : ℕ := 555555555

theorem gcd_a_b : Nat.gcd a b = 111111111 := 
by
  sorry

end NUMINAMATH_GPT_gcd_a_b_l1003_100304


namespace NUMINAMATH_GPT_banana_popsicles_count_l1003_100343

theorem banana_popsicles_count 
  (grape_popsicles cherry_popsicles total_popsicles : ℕ)
  (h1 : grape_popsicles = 2)
  (h2 : cherry_popsicles = 13)
  (h3 : total_popsicles = 17) :
  total_popsicles - (grape_popsicles + cherry_popsicles) = 2 := by
  sorry

end NUMINAMATH_GPT_banana_popsicles_count_l1003_100343


namespace NUMINAMATH_GPT_find_b_l1003_100398

def f (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ (b : ℝ), f b = 3 :=
by
  use 2
  show f 2 = 3
  sorry

end NUMINAMATH_GPT_find_b_l1003_100398


namespace NUMINAMATH_GPT_solve_inequality_l1003_100337

-- Define the odd and monotonically decreasing function
noncomputable def f : ℝ → ℝ := sorry

-- Assume the given conditions
axiom odd_f : ∀ x, f (-x) = -f x
axiom decreasing_f : ∀ x y, x < y → y < 0 → f x > f y
axiom f_at_2 : f 2 = 0

-- The proof statement
theorem solve_inequality (x : ℝ) : (x - 1) * f (x + 1) > 0 ↔ -3 < x ∧ x < -1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_solve_inequality_l1003_100337


namespace NUMINAMATH_GPT_tan_of_angle_in_second_quadrant_l1003_100372

theorem tan_of_angle_in_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.cos (π / 2 - α) = 4 / 5) : Real.tan α = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_of_angle_in_second_quadrant_l1003_100372


namespace NUMINAMATH_GPT_is_linear_equation_D_l1003_100373

theorem is_linear_equation_D :
  (∀ (x y : ℝ), 2 * x + 3 * y = 7 → false) ∧
  (∀ (x : ℝ), 3 * x ^ 2 = 3 → false) ∧
  (∀ (x : ℝ), 6 = 2 / x - 1 → false) ∧
  (∀ (x : ℝ), 2 * x - 1 = 20 → true) 
:= by {
  sorry
}

end NUMINAMATH_GPT_is_linear_equation_D_l1003_100373


namespace NUMINAMATH_GPT_average_speed_l1003_100327

def total_distance : ℝ := 200
def total_time : ℝ := 40

theorem average_speed (d t : ℝ) (h₁: d = total_distance) (h₂: t = total_time) : d / t = 5 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end NUMINAMATH_GPT_average_speed_l1003_100327


namespace NUMINAMATH_GPT_expression_evaluation_l1003_100394

theorem expression_evaluation:
  ( (1/3)^2000 * 27^669 + Real.sin (60 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) + (2009 + Real.sin (25 * Real.pi / 180))^0 ) = 
  (2 + 29/54) := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1003_100394


namespace NUMINAMATH_GPT_f_eq_f_at_neg_one_f_at_neg_500_l1003_100363

noncomputable def f : ℝ → ℝ := sorry

theorem f_eq : ∀ x y : ℝ, f (x * y) + x = x * f y + f x := sorry
theorem f_at_neg_one : f (-1) = 1 := sorry

theorem f_at_neg_500 : f (-500) = 999 := sorry

end NUMINAMATH_GPT_f_eq_f_at_neg_one_f_at_neg_500_l1003_100363


namespace NUMINAMATH_GPT_sequence_formula_l1003_100378

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n: ℕ, a (n + 1) = 2 * a n + n * (1 + 2^n)) :
  ∀ n : ℕ, a n = 2^(n - 2) * (n^2 - n + 6) - n - 1 :=
by intro n; sorry

end NUMINAMATH_GPT_sequence_formula_l1003_100378


namespace NUMINAMATH_GPT_domain_of_c_x_l1003_100369

theorem domain_of_c_x (k : ℝ) :
  (∀ x : ℝ, -5 * x ^ 2 + 3 * x + k ≠ 0) ↔ k < -9 / 20 := 
sorry

end NUMINAMATH_GPT_domain_of_c_x_l1003_100369


namespace NUMINAMATH_GPT_total_animals_in_shelter_l1003_100386

def initial_cats : ℕ := 15
def adopted_cats := initial_cats / 3
def replacement_cats := 2 * adopted_cats
def current_cats := initial_cats - adopted_cats + replacement_cats
def additional_dogs := 2 * current_cats
def total_animals := current_cats + additional_dogs

theorem total_animals_in_shelter : total_animals = 60 := by
  sorry

end NUMINAMATH_GPT_total_animals_in_shelter_l1003_100386


namespace NUMINAMATH_GPT_x_must_be_negative_l1003_100332

theorem x_must_be_negative (x y : ℝ) (h1 : y ≠ 0) (h2 : y > 0) (h3 : x / y < -3) : x < 0 :=
by 
  sorry

end NUMINAMATH_GPT_x_must_be_negative_l1003_100332


namespace NUMINAMATH_GPT_right_triangle_other_side_l1003_100336

theorem right_triangle_other_side (c a : ℝ) (h_c : c = 10) (h_a : a = 6) : ∃ b : ℝ, b^2 = c^2 - a^2 ∧ b = 8 :=
by
  use 8
  rw [h_c, h_a]
  simp
  sorry

end NUMINAMATH_GPT_right_triangle_other_side_l1003_100336


namespace NUMINAMATH_GPT_probability_of_graduate_degree_l1003_100314

variables (G C N : ℕ)
axiom h1 : G / N = 1 / 8
axiom h2 : C / N = 2 / 3

noncomputable def total_college_graduates (G C : ℕ) : ℕ := G + C

noncomputable def probability_graduate_degree (G C : ℕ) : ℚ := G / (total_college_graduates G C)

theorem probability_of_graduate_degree :
  probability_graduate_degree 3 16 = 3 / 19 :=
by 
  -- Here, we need to prove that the probability of picking a college graduate with a graduate degree
  -- is 3 / 19 given the conditions.
  sorry

end NUMINAMATH_GPT_probability_of_graduate_degree_l1003_100314


namespace NUMINAMATH_GPT_probability_of_correct_match_l1003_100322

theorem probability_of_correct_match :
  let n := 3
  let total_arrangements := Nat.factorial n
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = ((1: ℤ) / 6) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_correct_match_l1003_100322


namespace NUMINAMATH_GPT_stationery_problem_l1003_100300

variables (S E : ℕ)

theorem stationery_problem
  (h1 : S - E = 30)
  (h2 : 4 * E = S) :
  S = 40 :=
by
  sorry

end NUMINAMATH_GPT_stationery_problem_l1003_100300


namespace NUMINAMATH_GPT_sheila_weekly_earnings_l1003_100360

-- Definitions for conditions
def hours_per_day_on_MWF : ℕ := 8
def days_worked_on_MWF : ℕ := 3
def hours_per_day_on_TT : ℕ := 6
def days_worked_on_TT : ℕ := 2
def hourly_rate : ℕ := 10

-- Total weekly hours worked
def total_weekly_hours : ℕ :=
  (hours_per_day_on_MWF * days_worked_on_MWF) + (hours_per_day_on_TT * days_worked_on_TT)

-- Total weekly earnings
def weekly_earnings : ℕ :=
  total_weekly_hours * hourly_rate

-- Lean statement for the proof
theorem sheila_weekly_earnings : weekly_earnings = 360 :=
  sorry

end NUMINAMATH_GPT_sheila_weekly_earnings_l1003_100360


namespace NUMINAMATH_GPT_infinite_series_eval_l1003_100310

open Filter
open Real
open Topology
open BigOperators

-- Define the relevant expression for the infinite sum
noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, (n / (n^4 - 4 * n^2 + 8))

-- The theorem statement
theorem infinite_series_eval : infinite_series_sum = 5 / 24 :=
by sorry

end NUMINAMATH_GPT_infinite_series_eval_l1003_100310


namespace NUMINAMATH_GPT_simplify_expression_l1003_100368

theorem simplify_expression (x : ℝ) : (2 * x)^5 + (3 * x) * x^4 + 2 * x^3 = 35 * x^5 + 2 * x^3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1003_100368


namespace NUMINAMATH_GPT_winning_percentage_is_65_l1003_100326

theorem winning_percentage_is_65 
  (total_games won_games : ℕ) 
  (h1 : total_games = 280) 
  (h2 : won_games = 182) :
  ((won_games : ℚ) / (total_games : ℚ)) * 100 = 65 :=
by
  sorry

end NUMINAMATH_GPT_winning_percentage_is_65_l1003_100326


namespace NUMINAMATH_GPT_two_digit_multiple_condition_l1003_100312

theorem two_digit_multiple_condition :
  ∃ x : ℕ, 10 ≤ x ∧ x < 100 ∧ ∃ k : ℤ, x = 30 * k + 2 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_multiple_condition_l1003_100312


namespace NUMINAMATH_GPT_length_of_ladder_l1003_100370

theorem length_of_ladder (a b : ℝ) (ha : a = 20) (hb : b = 15) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 25 := by
  sorry

end NUMINAMATH_GPT_length_of_ladder_l1003_100370


namespace NUMINAMATH_GPT_increasing_C_l1003_100359

theorem increasing_C (e R r : ℝ) (n : ℕ) (h₁ : 0 < e) (h₂ : 0 < R) (h₃ : 0 < r) (h₄ : 0 < n) :
    ∀ n1 n2 : ℕ, n1 < n2 → (e^2 * n1) / (R + n1 * r) < (e^2 * n2) / (R + n2 * r) :=
by
  sorry

end NUMINAMATH_GPT_increasing_C_l1003_100359


namespace NUMINAMATH_GPT_num_valid_m_l1003_100325

theorem num_valid_m (m : ℕ) : (∃ n : ℕ, n * (m^2 - 3) = 1722) → ∃ p : ℕ, p = 3 := 
  by
  sorry

end NUMINAMATH_GPT_num_valid_m_l1003_100325


namespace NUMINAMATH_GPT_geometric_sequence_sixth_term_l1003_100393

theorem geometric_sequence_sixth_term (a r : ℕ) (h₁ : a = 8) (h₂ : a * r ^ 3 = 64) : a * r ^ 5 = 256 :=
by
  -- to be filled (proof skipped)
  sorry

end NUMINAMATH_GPT_geometric_sequence_sixth_term_l1003_100393


namespace NUMINAMATH_GPT_consumer_credit_amount_l1003_100339

theorem consumer_credit_amount
  (C A : ℝ)
  (h1 : A = 0.20 * C)
  (h2 : 57 = 1/3 * A) :
  C = 855 := by
  sorry

end NUMINAMATH_GPT_consumer_credit_amount_l1003_100339


namespace NUMINAMATH_GPT_inclination_angle_range_l1003_100308

theorem inclination_angle_range :
  let Γ := fun x y : ℝ => x * abs x + y * abs y = 1
  let line (m : ℝ) := fun x y : ℝ => y = m * (x - 1)
  ∀ m : ℝ,
  (∃ p1 p2 p3 : ℝ × ℝ, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    line m p1.1 p1.2 ∧ Γ p1.1 p1.2 ∧ 
    line m p2.1 p2.2 ∧ Γ p2.1 p2.2 ∧ 
    line m p3.1 p3.2 ∧ Γ p3.1 p3.2) →
  (∃ θ : ℝ, θ ∈ (Set.Ioo (Real.pi / 2) (3 * Real.pi / 4) ∪ 
                  Set.Ioo (3 * Real.pi / 4) (Real.pi - Real.arctan (Real.sqrt 2 / 2)))) :=
sorry

end NUMINAMATH_GPT_inclination_angle_range_l1003_100308


namespace NUMINAMATH_GPT_diophantine_solution_range_l1003_100374

theorem diophantine_solution_range {a b c n : ℤ} (coprime_ab : Int.gcd a b = 1) :
  (∃ (x y : ℕ), a * x + b * y = c ∧ ∀ k : ℤ, k ≥ 1 → ∃ (x y : ℕ), a * (x + k * b) + b * (y - k * a) = c) → 
  ((n - 1) * a * b + a + b ≤ c ∧ c ≤ (n + 1) * a * b) :=
sorry

end NUMINAMATH_GPT_diophantine_solution_range_l1003_100374


namespace NUMINAMATH_GPT_jon_awake_hours_per_day_l1003_100377

def regular_bottle_size : ℕ := 16
def larger_bottle_size : ℕ := 20
def weekly_fluid_intake : ℕ := 728
def larger_bottle_daily_intake : ℕ := 40
def larger_bottle_weekly_intake : ℕ := 280
def regular_bottle_weekly_intake : ℕ := 448
def regular_bottles_per_week : ℕ := 28
def regular_bottles_per_day : ℕ := 4
def hours_per_bottle : ℕ := 4

theorem jon_awake_hours_per_day
  (h1 : jon_drinks_regular_bottle_every_4_hours)
  (h2 : jon_drinks_two_larger_bottles_daily)
  (h3 : jon_drinks_728_ounces_per_week) :
  jon_is_awake_hours_per_day = 16 :=
by
  sorry

def jon_drinks_regular_bottle_every_4_hours : Prop :=
  ∀ hours : ℕ, hours * regular_bottle_size / hours_per_bottle = 1

def jon_drinks_two_larger_bottles_daily : Prop :=
  larger_bottle_size = (regular_bottle_size * 5) / 4 ∧ 
  larger_bottle_daily_intake = 2 * larger_bottle_size

def jon_drinks_728_ounces_per_week : Prop :=
  weekly_fluid_intake = 728

def jon_is_awake_hours_per_day : ℕ :=
  regular_bottles_per_day * hours_per_bottle

end NUMINAMATH_GPT_jon_awake_hours_per_day_l1003_100377


namespace NUMINAMATH_GPT_storage_methods_l1003_100349

-- Definitions for the vertices and edges of the pyramid
structure Pyramid :=
  (P A B C D : Type)
  
-- Edges of the pyramid represented by pairs of vertices
def edges (P A B C D : Type) := [(P, A), (P, B), (P, C), (P, D), (A, B), (A, C), (A, D), (B, C), (B, D), (C, D)]

-- Safe storage condition: No edges sharing a common vertex in the same warehouse
def safe (edge1 edge2 : (Type × Type)) : Prop :=
  edge1.1 ≠ edge2.1 ∧ edge1.1 ≠ edge2.2 ∧ edge1.2 ≠ edge2.1 ∧ edge1.2 ≠ edge2.2

-- The number of different methods to store the chemical products safely
def number_of_safe_storage_methods : Nat :=
  -- We should replace this part by actual calculation or combinatorial methods relevant to the problem
  48

theorem storage_methods (P A B C D : Type) : number_of_safe_storage_methods = 48 :=
  sorry

end NUMINAMATH_GPT_storage_methods_l1003_100349


namespace NUMINAMATH_GPT_cubic_sum_l1003_100306

theorem cubic_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_sum_l1003_100306


namespace NUMINAMATH_GPT_speed_increase_l1003_100301

theorem speed_increase (v_initial: ℝ) (t_initial: ℝ) (t_new: ℝ) :
  v_initial = 60 → t_initial = 1 → t_new = 0.5 →
  v_new = (1 / (t_new / 60)) →
  v_increase = v_new - v_initial →
  v_increase = 60 :=
by
  sorry

end NUMINAMATH_GPT_speed_increase_l1003_100301


namespace NUMINAMATH_GPT_remainder_of_127_div_25_is_2_l1003_100340

theorem remainder_of_127_div_25_is_2 : ∃ r, 127 = 25 * 5 + r ∧ r = 2 := by
  have h1 : 127 = 25 * 5 + (127 - 25 * 5) := by rw [mul_comm 25 5, mul_comm 5 25]
  have h2 : 127 - 25 * 5 = 2 := by norm_num
  exact ⟨127 - 25 * 5, h1, h2⟩

end NUMINAMATH_GPT_remainder_of_127_div_25_is_2_l1003_100340


namespace NUMINAMATH_GPT_ratio_length_to_width_l1003_100391

theorem ratio_length_to_width
  (w l : ℕ)
  (pond_length : ℕ)
  (field_length : ℕ)
  (pond_area : ℕ)
  (field_area : ℕ)
  (pond_to_field_area_ratio : ℚ)
  (field_length_given : field_length = 28)
  (pond_length_given : pond_length = 7)
  (pond_area_def : pond_area = pond_length * pond_length)
  (pond_to_field_area_ratio_def : pond_to_field_area_ratio = 1 / 8)
  (field_area_def : field_area = pond_area * 8)
  (field_area_calc : field_area = field_length * w) :
  (field_length / w) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_length_to_width_l1003_100391


namespace NUMINAMATH_GPT_cost_of_each_adult_meal_is_8_l1003_100382

/- Define the basic parameters and conditions -/
def total_people : ℕ := 11
def kids : ℕ := 2
def total_cost : ℕ := 72
def kids_eat_free (k : ℕ) := k = 0

/- The number of adults is derived from the total people minus kids -/
def num_adults : ℕ := total_people - kids

/- The cost per adult meal can be defined and we need to prove it equals to $8 -/
def cost_per_adult (total_cost : ℕ) (num_adults : ℕ) : ℕ := total_cost / num_adults

/- The statement to prove that the cost per adult meal is $8 -/
theorem cost_of_each_adult_meal_is_8 : cost_per_adult total_cost num_adults = 8 := by
  sorry

end NUMINAMATH_GPT_cost_of_each_adult_meal_is_8_l1003_100382


namespace NUMINAMATH_GPT_decompose_max_product_l1003_100331

theorem decompose_max_product (a : ℝ) (h_pos : a > 0) :
  ∃ x y : ℝ, x + y = a ∧ x * y ≤ (a / 2) * (a / 2) :=
by
  sorry

end NUMINAMATH_GPT_decompose_max_product_l1003_100331


namespace NUMINAMATH_GPT_frac_e_a_l1003_100381

variable (a b c d e : ℚ)

theorem frac_e_a (h1 : a / b = 5) (h2 : b / c = 1 / 4) (h3 : c / d = 7) (h4 : d / e = 1 / 2) :
  e / a = 8 / 35 :=
sorry

end NUMINAMATH_GPT_frac_e_a_l1003_100381


namespace NUMINAMATH_GPT_candy_count_in_third_set_l1003_100351

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end NUMINAMATH_GPT_candy_count_in_third_set_l1003_100351


namespace NUMINAMATH_GPT_speed_against_current_l1003_100305

theorem speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) (man_speed_against_current : ℝ) 
  (h : speed_with_current = 12) (h1 : current_speed = 2) : man_speed_against_current = 8 :=
by
  sorry

end NUMINAMATH_GPT_speed_against_current_l1003_100305


namespace NUMINAMATH_GPT_eighth_triangular_number_l1003_100338

def triangular_number (n: ℕ) : ℕ := n * (n + 1) / 2

theorem eighth_triangular_number : triangular_number 8 = 36 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_eighth_triangular_number_l1003_100338


namespace NUMINAMATH_GPT_transformations_map_onto_itself_l1003_100350

noncomputable def recurring_pattern_map_count (s : ℝ) : ℕ := sorry

theorem transformations_map_onto_itself (s : ℝ) :
  recurring_pattern_map_count s = 2 := sorry

end NUMINAMATH_GPT_transformations_map_onto_itself_l1003_100350


namespace NUMINAMATH_GPT_square_division_rectangles_l1003_100383

theorem square_division_rectangles (k l : ℕ) (h_square : exists s : ℝ, 0 < s) 
(segment_division : ∀ (p q : ℝ), exists r : ℕ, r = s * k ∧ r = s * l) :
  ∃ n : ℕ, n = k * l :=
sorry

end NUMINAMATH_GPT_square_division_rectangles_l1003_100383


namespace NUMINAMATH_GPT_rectangle_area_error_l1003_100366

theorem rectangle_area_error (A B : ℝ) :
  let A' := 1.08 * A
  let B' := 1.08 * B
  let actual_area := A * B
  let measured_area := A' * B'
  let percentage_error := ((measured_area - actual_area) / actual_area) * 100
  percentage_error = 16.64 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_error_l1003_100366


namespace NUMINAMATH_GPT_total_cars_made_in_two_days_l1003_100311

-- Define the number of cars made yesterday.
def cars_yesterday : ℕ := 60

-- Define the number of cars made today, which is twice the number of cars made yesterday.
def cars_today : ℕ := 2 * cars_yesterday

-- Define the total number of cars made over the two days.
def total_cars : ℕ := cars_yesterday + cars_today

-- Proof statement: Prove that the total number of cars made in these two days is 180.
theorem total_cars_made_in_two_days : total_cars = 180 := by
  -- Here, we would provide the proof, but we'll use sorry to skip it.
  sorry

end NUMINAMATH_GPT_total_cars_made_in_two_days_l1003_100311


namespace NUMINAMATH_GPT_josh_marbles_l1003_100313

theorem josh_marbles (original_marble : ℝ) (given_marble : ℝ)
  (h1 : original_marble = 22.5) (h2 : given_marble = 20.75) :
  original_marble + given_marble = 43.25 := by
  sorry

end NUMINAMATH_GPT_josh_marbles_l1003_100313


namespace NUMINAMATH_GPT_joe_first_lift_weight_l1003_100318

theorem joe_first_lift_weight (x y : ℕ) (h1 : x + y = 1500) (h2 : 2 * x = y + 300) : x = 600 :=
by
  sorry

end NUMINAMATH_GPT_joe_first_lift_weight_l1003_100318


namespace NUMINAMATH_GPT_closest_point_on_parabola_l1003_100392

/-- The coordinates of the point on the parabola y^2 = x that is closest to the line x - 2y + 4 = 0 are (1,1). -/
theorem closest_point_on_parabola (y : ℝ) (x : ℝ) (h_parabola : y^2 = x) (h_line : x - 2*y + 4 = 0) :
  (x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_closest_point_on_parabola_l1003_100392


namespace NUMINAMATH_GPT_probability_of_winning_second_lawsuit_l1003_100371

theorem probability_of_winning_second_lawsuit
  (P_W1 P_L1 P_W2 P_L2 : ℝ)
  (h1 : P_W1 = 0.30)
  (h2 : P_L1 = 0.70)
  (h3 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20)
  (h4 : P_L2 = 1 - P_W2) :
  P_W2 = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_winning_second_lawsuit_l1003_100371


namespace NUMINAMATH_GPT_purple_chip_count_l1003_100362

theorem purple_chip_count :
  ∃ (x : ℕ), (x > 5) ∧ (x < 11) ∧
  (∃ (blue green purple red : ℕ),
    (2^6) * (5^2) * 11 * 7 = (blue * 1) * (green * 5) * (purple * x) * (red * 11) ∧ purple = 1) :=
sorry

end NUMINAMATH_GPT_purple_chip_count_l1003_100362


namespace NUMINAMATH_GPT_pencil_cost_l1003_100320

theorem pencil_cost (p e : ℝ) (h1 : p + e = 3.40) (h2 : p = 3 + e) : p = 3.20 :=
by
  sorry

end NUMINAMATH_GPT_pencil_cost_l1003_100320


namespace NUMINAMATH_GPT_train_crosses_platform_in_34_seconds_l1003_100316

theorem train_crosses_platform_in_34_seconds 
    (train_speed_kmph : ℕ) 
    (time_cross_man_sec : ℕ) 
    (platform_length_m : ℕ) 
    (h_speed : train_speed_kmph = 72) 
    (h_time : time_cross_man_sec = 18) 
    (h_platform_length : platform_length_m = 320) 
    : (platform_length_m + (train_speed_kmph * 1000 / 3600) * time_cross_man_sec) / (train_speed_kmph * 1000 / 3600) = 34 :=
by
    sorry

end NUMINAMATH_GPT_train_crosses_platform_in_34_seconds_l1003_100316


namespace NUMINAMATH_GPT_striped_jerseys_count_l1003_100323

-- Define the cost of long-sleeved jerseys
def cost_long_sleeved := 15
-- Define the cost of striped jerseys
def cost_striped := 10
-- Define the number of long-sleeved jerseys bought
def num_long_sleeved := 4
-- Define the total amount spent
def total_spent := 80

-- Define a theorem to prove the number of striped jerseys bought
theorem striped_jerseys_count : ∃ x : ℕ, x * cost_striped = total_spent - num_long_sleeved * cost_long_sleeved ∧ x = 2 := 
by 
-- TODO: The proof steps would go here, but for this exercise, we use 'sorry' to skip the proof.
sorry

end NUMINAMATH_GPT_striped_jerseys_count_l1003_100323


namespace NUMINAMATH_GPT_average_remaining_numbers_l1003_100303

theorem average_remaining_numbers (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50 : ℝ) = 38) 
  (h_discard : 45 ∈ numbers ∧ 55 ∈ numbers) :
  let new_sum := numbers.sum - 45 - 55
  let new_len := 50 - 2
  (new_sum / new_len : ℝ) = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_average_remaining_numbers_l1003_100303


namespace NUMINAMATH_GPT_probability_of_meeting_at_cafe_l1003_100353

open Set

/-- Define the unit square where each side represents 1 hour (from 2:00 to 3:00 PM). -/
def unit_square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

/-- Define the overlap condition for Cara and David meeting at the café. -/
def overlap_region : Set (ℝ × ℝ) :=
  { p | max (p.1 - 0.5) 0 ≤ p.2 ∧ p.2 ≤ min (p.1 + 0.5) 1 }

/-- The area of the overlap region within the unit square. -/
noncomputable def overlap_area : ℝ :=
  ∫ x in Icc 0 1, (min (x + 0.5) 1 - max (x - 0.5) 0)

theorem probability_of_meeting_at_cafe : overlap_area / 1 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_meeting_at_cafe_l1003_100353


namespace NUMINAMATH_GPT_round_trip_time_l1003_100342

noncomputable def time_to_complete_trip (speed_without_load speed_with_load distance rest_stops_in_minutes : ℝ) : ℝ :=
  let rest_stops_in_hours := rest_stops_in_minutes / 60
  let half_rest_time := 2 * rest_stops_in_hours
  let total_rest_time := 2 * half_rest_time
  let travel_time_with_load := distance / speed_with_load
  let travel_time_without_load := distance / speed_without_load
  travel_time_with_load + travel_time_without_load + total_rest_time

theorem round_trip_time :
  time_to_complete_trip 13 11 143 30 = 26 :=
sorry

end NUMINAMATH_GPT_round_trip_time_l1003_100342


namespace NUMINAMATH_GPT_factorial_division_l1003_100335

-- Definitions of factorial used in Lean according to math problem statement.
open Nat

-- Statement of the proof problem in Lean 4.
theorem factorial_division : (12! - 11!) / 10! = 121 := by
  sorry

end NUMINAMATH_GPT_factorial_division_l1003_100335


namespace NUMINAMATH_GPT_solve_for_x_l1003_100355

theorem solve_for_x (x : ℝ) : 3 * x^2 + 15 = 3 * (2 * x + 20) → (x = 5 ∨ x = -3) :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1003_100355


namespace NUMINAMATH_GPT_john_days_ran_l1003_100384

theorem john_days_ran 
  (total_distance : ℕ) (daily_distance : ℕ) 
  (h1 : total_distance = 10200) (h2 : daily_distance = 1700) :
  total_distance / daily_distance = 6 :=
by
  sorry

end NUMINAMATH_GPT_john_days_ran_l1003_100384


namespace NUMINAMATH_GPT_f_diff_l1003_100345

-- Define the function f(n)
def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n + 1 + 1)).sum (λ i => 1 / (n + i + 1))

-- The theorem stating the main problem
theorem f_diff (k : ℕ) : 
  f (k + 1) - f k = (1 / (3 * k + 2)) + (1 / (3 * k + 3)) + (1 / (3 * k + 4)) - (1 / (k + 1)) :=
by
  sorry

end NUMINAMATH_GPT_f_diff_l1003_100345


namespace NUMINAMATH_GPT_triangle_cosine_identity_l1003_100388

theorem triangle_cosine_identity
  (a b c : ℝ)
  (α β γ : ℝ)
  (hα : α = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))
  (hβ : β = Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)))
  (hγ : γ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))
  (habc_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b / c + c / b) * Real.cos α + 
  (c / a + a / c) * Real.cos β + 
  (a / b + b / a) * Real.cos γ = 3 := 
sorry

end NUMINAMATH_GPT_triangle_cosine_identity_l1003_100388


namespace NUMINAMATH_GPT_mixed_oil_rate_per_litre_l1003_100319

variables (volume1 : ℝ) (price1 : ℝ) (volume2 : ℝ) (price2 : ℝ)

def total_cost (v p : ℝ) : ℝ := v * p
def total_volume (v1 v2 : ℝ) : ℝ := v1 + v2

theorem mixed_oil_rate_per_litre (h1 : volume1 = 10) (h2 : price1 = 55) (h3 : volume2 = 5) (h4 : price2 = 66) :
  (total_cost volume1 price1 + total_cost volume2 price2) / total_volume volume1 volume2 = 58.67 := 
by
  sorry

end NUMINAMATH_GPT_mixed_oil_rate_per_litre_l1003_100319
