import Mathlib

namespace NUMINAMATH_GPT_perfect_square_difference_l984_98401

theorem perfect_square_difference (m n : ℕ) (h : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, k^2 = m - n :=
sorry

end NUMINAMATH_GPT_perfect_square_difference_l984_98401


namespace NUMINAMATH_GPT_percent_of_x_is_y_l984_98458

theorem percent_of_x_is_y 
    (x y : ℝ) 
    (h : 0.30 * (x - y) = 0.20 * (x + y)) : 
    y / x = 0.2 :=
  sorry

end NUMINAMATH_GPT_percent_of_x_is_y_l984_98458


namespace NUMINAMATH_GPT_fastest_pipe_is_4_l984_98418

/-- There are five pipes with flow rates Q_1, Q_2, Q_3, Q_4, and Q_5.
    The ordering of their flow rates is given by:
    (1) Q_1 > Q_3
    (2) Q_2 < Q_4
    (3) Q_3 < Q_5
    (4) Q_4 > Q_1
    (5) Q_5 < Q_2
    We need to prove that single pipe Q_4 will fill the pool the fastest.
 -/
theorem fastest_pipe_is_4 
  (Q1 Q2 Q3 Q4 Q5 : ℝ)
  (h1 : Q1 > Q3)
  (h2 : Q2 < Q4)
  (h3 : Q3 < Q5)
  (h4 : Q4 > Q1)
  (h5 : Q5 < Q2) :
  Q4 > Q1 ∧ Q4 > Q2 ∧ Q4 > Q3 ∧ Q4 > Q5 :=
by
  sorry

end NUMINAMATH_GPT_fastest_pipe_is_4_l984_98418


namespace NUMINAMATH_GPT_tan_ratio_l984_98491

theorem tan_ratio (α β : ℝ) (h : Real.sin (2 * α) = 3 * Real.sin (2 * β)) :
  (Real.tan (α - β) / Real.tan (α + β)) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_tan_ratio_l984_98491


namespace NUMINAMATH_GPT_problem_statement_l984_98470

theorem problem_statement (x Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
    10 * (6 * x + 14 * Real.pi) = 4 * Q := 
sorry

end NUMINAMATH_GPT_problem_statement_l984_98470


namespace NUMINAMATH_GPT_power_mod_eight_l984_98422

theorem power_mod_eight (n : ℕ) : (3^101 + 5) % 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_power_mod_eight_l984_98422


namespace NUMINAMATH_GPT_log_expression_evaluation_l984_98408

open Real

theorem log_expression_evaluation : log 5 * log 20 + (log 2) ^ 2 = 1 := 
sorry

end NUMINAMATH_GPT_log_expression_evaluation_l984_98408


namespace NUMINAMATH_GPT_inner_tetrahedron_volume_ratio_l984_98495

noncomputable def volume_ratio_of_tetrahedrons (s : ℝ) : ℝ :=
  let V_original := (s^3 * Real.sqrt 2) / 12
  let a := (Real.sqrt 6 / 9) * s
  let V_inner := (a^3 * Real.sqrt 2) / 12
  V_inner / V_original

theorem inner_tetrahedron_volume_ratio {s : ℝ} (hs : s > 0) : volume_ratio_of_tetrahedrons s = 1 / 243 :=
by
  sorry

end NUMINAMATH_GPT_inner_tetrahedron_volume_ratio_l984_98495


namespace NUMINAMATH_GPT_no_real_roots_of_geom_seq_l984_98441

theorem no_real_roots_of_geom_seq (a b c : ℝ) (h_geom_seq : b^2 = a * c) : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  -- You can assume the steps of proving here
  sorry

end NUMINAMATH_GPT_no_real_roots_of_geom_seq_l984_98441


namespace NUMINAMATH_GPT_red_paint_intensity_l984_98414

theorem red_paint_intensity (x : ℝ) (h1 : 0.5 * 10 + 0.5 * x = 15) : x = 20 :=
sorry

end NUMINAMATH_GPT_red_paint_intensity_l984_98414


namespace NUMINAMATH_GPT_actual_cost_before_decrease_l984_98429

theorem actual_cost_before_decrease (x : ℝ) (h : 0.76 * x = 1064) : x = 1400 :=
by
  sorry

end NUMINAMATH_GPT_actual_cost_before_decrease_l984_98429


namespace NUMINAMATH_GPT_unknown_sum_of_digits_l984_98443

theorem unknown_sum_of_digits 
  (A B C D : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h2 : D = 1)
  (h3 : (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D) : 
  A + B = 0 := 
sorry

end NUMINAMATH_GPT_unknown_sum_of_digits_l984_98443


namespace NUMINAMATH_GPT_expand_expression_l984_98464

theorem expand_expression (x y : ℝ) : 12 * (3 * x - 4 * y + 2) = 36 * x - 48 * y + 24 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l984_98464


namespace NUMINAMATH_GPT_find_y_l984_98404

noncomputable def x : ℝ := 3.3333333333333335

theorem find_y (y x: ℝ) (h1: x = 3.3333333333333335) (h2: x * 10 / y = x^2) :
  y = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l984_98404


namespace NUMINAMATH_GPT_range_of_m_l984_98412

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → (7 / 4) ≤ (x^2 - 3 * x + 4) ∧ (x^2 - 3 * x + 4) ≤ 4) ↔ (3 / 2 ≤ m ∧ m ≤ 3) := 
sorry

end NUMINAMATH_GPT_range_of_m_l984_98412


namespace NUMINAMATH_GPT_side_length_of_square_l984_98453

theorem side_length_of_square (s : ℚ) (h : s^2 = 9/16) : s = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l984_98453


namespace NUMINAMATH_GPT_accurate_place_24000_scientific_notation_46400000_l984_98468

namespace MathProof

def accurate_place (n : ℕ) : String :=
  if n = 24000 then "hundred's place" else "unknown"

def scientific_notation (n : ℕ) : String :=
  if n = 46400000 then "4.64 × 10^7" else "unknown"

theorem accurate_place_24000 : accurate_place 24000 = "hundred's place" :=
by
  sorry

theorem scientific_notation_46400000 : scientific_notation 46400000 = "4.64 × 10^7" :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_accurate_place_24000_scientific_notation_46400000_l984_98468


namespace NUMINAMATH_GPT_find_r_l984_98489

def cubic_function (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_r (p q r : ℝ) (h1 : cubic_function p q r (-1) = 0) :
  r = p - 2 :=
sorry

end NUMINAMATH_GPT_find_r_l984_98489


namespace NUMINAMATH_GPT_least_of_consecutive_odds_l984_98438

noncomputable def average_of_consecutive_odds (n : ℕ) (start : ℤ) : ℤ :=
start + (2 * (n - 1))

theorem least_of_consecutive_odds
    (n : ℕ)
    (mean : ℤ)
    (h : n = 30 ∧ mean = 526) : 
    average_of_consecutive_odds 1 (mean * 2 - (n - 1)) = 497 :=
by
  sorry

end NUMINAMATH_GPT_least_of_consecutive_odds_l984_98438


namespace NUMINAMATH_GPT_carbonic_acid_formation_l984_98483

-- Definition of amounts of substances involved
def moles_CO2 : ℕ := 3
def moles_H2O : ℕ := 3

-- Stoichiometric condition derived from the equation CO2 + H2O → H2CO3
def stoichiometric_ratio (a b c : ℕ) : Prop := (a = b) ∧ (a = c)

-- The main statement to prove
theorem carbonic_acid_formation : 
  stoichiometric_ratio moles_CO2 moles_H2O 3 :=
by
  sorry

end NUMINAMATH_GPT_carbonic_acid_formation_l984_98483


namespace NUMINAMATH_GPT_division_of_negatives_l984_98440

theorem division_of_negatives (x y : Int) (h1 : y ≠ 0) (h2 : -x = 150) (h3 : -y = 25) : (-150) / (-25) = 6 :=
by
  sorry

end NUMINAMATH_GPT_division_of_negatives_l984_98440


namespace NUMINAMATH_GPT_cookies_left_l984_98462

theorem cookies_left (initial_cookies : ℕ) (cookies_eaten : ℕ) (cookies_left : ℕ) :
  initial_cookies = 28 → cookies_eaten = 21 → cookies_left = initial_cookies - cookies_eaten → cookies_left = 7 :=
by
  intros h_initial h_eaten h_left
  rw [h_initial, h_eaten] at h_left
  exact h_left

end NUMINAMATH_GPT_cookies_left_l984_98462


namespace NUMINAMATH_GPT_fraction_solution_l984_98437

theorem fraction_solution (x : ℝ) (h1 : (x - 4) / (x^2) = 0) (h2 : x ≠ 0) : x = 4 :=
sorry

end NUMINAMATH_GPT_fraction_solution_l984_98437


namespace NUMINAMATH_GPT_ratio_d_s_l984_98419

theorem ratio_d_s (s d : ℝ) 
  (h : (25 * 25 * s^2) / (25 * s + 50 * d)^2 = 0.81) :
  d / s = 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_ratio_d_s_l984_98419


namespace NUMINAMATH_GPT_probability_not_siblings_l984_98426

noncomputable def num_individuals : ℕ := 6
noncomputable def num_pairs : ℕ := num_individuals / 2
noncomputable def total_pairs : ℕ := num_individuals * (num_individuals - 1) / 2
noncomputable def sibling_pairs : ℕ := num_pairs
noncomputable def non_sibling_pairs : ℕ := total_pairs - sibling_pairs

theorem probability_not_siblings :
  (non_sibling_pairs : ℚ) / total_pairs = 4 / 5 := 
by sorry

end NUMINAMATH_GPT_probability_not_siblings_l984_98426


namespace NUMINAMATH_GPT_division_problem_l984_98436

-- Define the involved constants and operations
def expr1 : ℚ := 5 / 2 * 3
def expr2 : ℚ := 100 / expr1

-- Formulate the final equality
theorem division_problem : expr2 = 40 / 3 :=
  by sorry

end NUMINAMATH_GPT_division_problem_l984_98436


namespace NUMINAMATH_GPT_angle_CBD_is_4_l984_98442

theorem angle_CBD_is_4 (angle_ABC : ℝ) (angle_ABD : ℝ) (h₁ : angle_ABC = 24) (h₂ : angle_ABD = 20) : angle_ABC - angle_ABD = 4 :=
by 
  sorry

end NUMINAMATH_GPT_angle_CBD_is_4_l984_98442


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l984_98477

-- Definitions for the conditions
def rate_of_work (men : ℕ) (length : ℕ) (days : ℕ) : ℕ :=
  length / days / men

def work_rate_first_group (M : ℕ) : ℕ :=
  rate_of_work M 48 2

def work_rate_second_group : ℕ :=
  rate_of_work 2 36 3

theorem number_of_men_in_first_group (M : ℕ) 
  (h₁ : work_rate_first_group M = 24)
  (h₂ : work_rate_second_group = 12) :
  M = 4 :=
  sorry

end NUMINAMATH_GPT_number_of_men_in_first_group_l984_98477


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_3_5_7_11_l984_98445

theorem smallest_four_digit_divisible_by_3_5_7_11 : 
  ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ 
          n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 1155 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_3_5_7_11_l984_98445


namespace NUMINAMATH_GPT_intersection_nonempty_condition_l984_98407

theorem intersection_nonempty_condition (m n : ℝ) :
  (∃ x : ℝ, (m - 1 < x ∧ x < m + 1) ∧ (3 - n < x ∧ x < 4 - n)) ↔ (2 < m + n ∧ m + n < 5) := 
by
  sorry

end NUMINAMATH_GPT_intersection_nonempty_condition_l984_98407


namespace NUMINAMATH_GPT_height_on_hypotenuse_of_right_triangle_l984_98452

theorem height_on_hypotenuse_of_right_triangle (a b : ℝ) (h_a : a = 2) (h_b : b = 3) :
  ∃ h : ℝ, h = (6 * Real.sqrt 13) / 13 :=
by
  sorry

end NUMINAMATH_GPT_height_on_hypotenuse_of_right_triangle_l984_98452


namespace NUMINAMATH_GPT_find_x_l984_98473

variables (a b c : ℝ)

theorem find_x (h : a ≥ 0) (h' : b ≥ 0) (h'' : c ≥ 0) : 
  ∃ x ≥ 0, x = Real.sqrt ((b - c)^2 - a^2) :=
by
  use Real.sqrt ((b - c)^2 - a^2)
  sorry

end NUMINAMATH_GPT_find_x_l984_98473


namespace NUMINAMATH_GPT_ratio_of_areas_l984_98406

theorem ratio_of_areas (h a b R : ℝ) (h_triangle : a^2 + b^2 = h^2) (h_circumradius : R = h / 2) :
  (π * R^2) / (1/2 * a * b) = π * h / (4 * R) :=
by sorry

end NUMINAMATH_GPT_ratio_of_areas_l984_98406


namespace NUMINAMATH_GPT_rectangle_problem_l984_98420

def rectangle_perimeter (L B : ℕ) : ℕ :=
  2 * (L + B)

theorem rectangle_problem (L B : ℕ) (h1 : L - B = 23) (h2 : L * B = 2520) : rectangle_perimeter L B = 206 := by
  sorry

end NUMINAMATH_GPT_rectangle_problem_l984_98420


namespace NUMINAMATH_GPT_smallest_a1_l984_98461

noncomputable def a_seq (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 13 * a (n - 1) - 2 * n

noncomputable def positive_sequence (a : ℕ → ℝ) : Prop :=
∀ i, a i > 0

theorem smallest_a1 : ∃ a : ℕ → ℝ, a_seq a ∧ positive_sequence a ∧ a 1 = 13 / 36 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a1_l984_98461


namespace NUMINAMATH_GPT_sum_of_digits_d_l984_98496

theorem sum_of_digits_d (d : ℕ) (exchange_rate : 10 * d / 7 - 60 = d) : 
  (d = 140) -> (Nat.digits 10 140).sum = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_d_l984_98496


namespace NUMINAMATH_GPT_johns_contribution_correct_l984_98460

noncomputable def average_contribution_before : Real := sorry
noncomputable def total_contributions_by_15 : Real := 15 * average_contribution_before
noncomputable def new_average_contribution : Real := 150
noncomputable def johns_contribution : Real := average_contribution_before * 15 + 1377.3

-- The theorem we want to prove
theorem johns_contribution_correct :
  (new_average_contribution = (total_contributions_by_15 + johns_contribution) / 16) ∧
  (new_average_contribution = 2.2 * average_contribution_before) :=
sorry

end NUMINAMATH_GPT_johns_contribution_correct_l984_98460


namespace NUMINAMATH_GPT_sam_pennies_l984_98415

def pennies_from_washing_clothes (total_money_cents : ℤ) (quarters : ℤ) : ℤ :=
  total_money_cents - (quarters * 25)

theorem sam_pennies :
  pennies_from_washing_clothes 184 7 = 9 :=
by
  sorry

end NUMINAMATH_GPT_sam_pennies_l984_98415


namespace NUMINAMATH_GPT_base_k_for_repeating_series_equals_fraction_l984_98421

-- Define the fraction 5/29
def fraction := 5 / 29

-- Define the repeating series in base k
def repeating_series (k : ℕ) : ℚ :=
  (1 / k) / (1 - 1 / k^2) + (3 / k^2) / (1 - 1 / k^2)

-- State the problem
theorem base_k_for_repeating_series_equals_fraction (k : ℕ) (hk1 : 0 < k) (hk2 : k ≠ 1):
  repeating_series k = fraction ↔ k = 8 := sorry

end NUMINAMATH_GPT_base_k_for_repeating_series_equals_fraction_l984_98421


namespace NUMINAMATH_GPT_total_books_l984_98490

theorem total_books (joan_books : ℕ) (tom_books : ℕ) (h1 : joan_books = 10) (h2 : tom_books = 38) : joan_books + tom_books = 48 :=
by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_total_books_l984_98490


namespace NUMINAMATH_GPT_central_angle_unit_circle_l984_98499

theorem central_angle_unit_circle :
  ∀ (θ : ℝ), (∃ (A : ℝ), A = 1 ∧ (A = 1 / 2 * θ)) → θ = 2 :=
by
  intro θ
  rintro ⟨A, hA1, hA2⟩
  sorry

end NUMINAMATH_GPT_central_angle_unit_circle_l984_98499


namespace NUMINAMATH_GPT_avg_values_l984_98448

theorem avg_values (z : ℝ) : (0 + 3 * z + 6 * z + 12 * z + 24 * z) / 5 = 9 * z :=
by
  sorry

end NUMINAMATH_GPT_avg_values_l984_98448


namespace NUMINAMATH_GPT_problem1_problem2_l984_98454

theorem problem1 (x : ℝ) : (5 - 2 * x) ^ 2 - 16 = 0 ↔ x = 1 / 2 ∨ x = 9 / 2 := 
by 
  sorry

theorem problem2 (x : ℝ) : 2 * (x - 3) = x^2 - 9 ↔ x = 3 ∨ x = -1 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l984_98454


namespace NUMINAMATH_GPT_ticket_cost_is_correct_l984_98430

-- Conditions
def total_amount_raised : ℕ := 620
def number_of_tickets_sold : ℕ := 155

-- Definition of cost per ticket
def cost_per_ticket : ℕ := total_amount_raised / number_of_tickets_sold

-- The theorem to be proven
theorem ticket_cost_is_correct : cost_per_ticket = 4 :=
by
  sorry

end NUMINAMATH_GPT_ticket_cost_is_correct_l984_98430


namespace NUMINAMATH_GPT_isosceles_trapezoid_fewest_axes_l984_98459

def equilateral_triangle_axes : Nat := 3
def isosceles_trapezoid_axes : Nat := 1
def rectangle_axes : Nat := 2
def regular_pentagon_axes : Nat := 5

theorem isosceles_trapezoid_fewest_axes :
  isosceles_trapezoid_axes < equilateral_triangle_axes ∧
  isosceles_trapezoid_axes < rectangle_axes ∧
  isosceles_trapezoid_axes < regular_pentagon_axes :=
by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_fewest_axes_l984_98459


namespace NUMINAMATH_GPT_shares_correct_l984_98413

open Real

-- Problem setup
def original_problem (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 1020 ∧
  a = (3 / 4) * b ∧
  b = (2 / 3) * c ∧
  c = (1 / 4) * d ∧
  d = (5 / 6) * e

-- Goal
theorem shares_correct : ∃ (a b c d e : ℝ),
  original_problem a b c d e ∧
  abs (a - 58.17) < 0.01 ∧
  abs (b - 77.56) < 0.01 ∧
  abs (c - 116.34) < 0.01 ∧
  abs (d - 349.02) < 0.01 ∧
  abs (e - 419.42) < 0.01 := by
  sorry

end NUMINAMATH_GPT_shares_correct_l984_98413


namespace NUMINAMATH_GPT_B_takes_6_days_to_complete_work_alone_l984_98472

theorem B_takes_6_days_to_complete_work_alone 
    (work_duration_A : ℕ) 
    (work_payment : ℚ)
    (work_days_with_C : ℕ) 
    (payment_C : ℚ) 
    (combined_work_rate_A_B_C : ℚ)
    (amount_to_be_shared_A_B : ℚ) 
    (combined_daily_earning_A_B : ℚ) :
  work_duration_A = 6 ∧
  work_payment = 3360 ∧ 
  work_days_with_C = 3 ∧ 
  payment_C = 420.00000000000017 ∧ 
  combined_work_rate_A_B_C = 1 / 3 ∧ 
  amount_to_be_shared_A_B = 2940 ∧ 
  combined_daily_earning_A_B = 980 → 
  work_duration_A = 6 ∧
  (∃ (work_duration_B : ℕ), work_duration_B = 6) :=
by 
  sorry

end NUMINAMATH_GPT_B_takes_6_days_to_complete_work_alone_l984_98472


namespace NUMINAMATH_GPT_weight_of_pecans_l984_98474

theorem weight_of_pecans (total_weight_of_nuts almonds_weight pecans_weight : ℝ)
  (h1 : total_weight_of_nuts = 0.52)
  (h2 : almonds_weight = 0.14)
  (h3 : pecans_weight = total_weight_of_nuts - almonds_weight) :
  pecans_weight = 0.38 :=
  by
    sorry

end NUMINAMATH_GPT_weight_of_pecans_l984_98474


namespace NUMINAMATH_GPT_total_males_below_50_is_2638_l984_98486

def branchA_total_employees := 4500
def branchA_percentage_males := 60 / 100
def branchA_percentage_males_at_least_50 := 40 / 100

def branchB_total_employees := 3500
def branchB_percentage_males := 50 / 100
def branchB_percentage_males_at_least_50 := 55 / 100

def branchC_total_employees := 2200
def branchC_percentage_males := 35 / 100
def branchC_percentage_males_at_least_50 := 70 / 100

def males_below_50_branchA := (1 - branchA_percentage_males_at_least_50) * (branchA_percentage_males * branchA_total_employees)
def males_below_50_branchB := (1 - branchB_percentage_males_at_least_50) * (branchB_percentage_males * branchB_total_employees)
def males_below_50_branchC := (1 - branchC_percentage_males_at_least_50) * (branchC_percentage_males * branchC_total_employees)

def total_males_below_50 := males_below_50_branchA + males_below_50_branchB + males_below_50_branchC

theorem total_males_below_50_is_2638 : total_males_below_50 = 2638 := 
by
  -- Numerical evaluation and equality verification here
  sorry

end NUMINAMATH_GPT_total_males_below_50_is_2638_l984_98486


namespace NUMINAMATH_GPT_find_d_minus_r_l984_98423

theorem find_d_minus_r :
  ∃ (d r : ℕ), d > 1 ∧ 1083 % d = r ∧ 1455 % d = r ∧ 2345 % d = r ∧ d - r = 1 := by
  sorry

end NUMINAMATH_GPT_find_d_minus_r_l984_98423


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_geometric_sequence_solution_l984_98402

-- Problem 1: Arithmetic sequence
noncomputable def arithmetic_general_term (n : ℕ) : ℕ := 30 - 3 * n
noncomputable def arithmetic_sum_terms (n : ℕ) : ℝ := -1.5 * n^2 + 28.5 * n

theorem arithmetic_sequence_solution (n : ℕ) (a8 a10 : ℕ) (sequence : ℕ → ℝ) :
  a8 = 6 → a10 = 0 → (sequence n = arithmetic_general_term n) ∧ (sequence n = arithmetic_sum_terms n) ∧ (n = 9 ∨ n = 10) := 
sorry

-- Problem 2: Geometric sequence
noncomputable def geometric_general_term (n : ℕ) : ℝ := 2^(n-2)
noncomputable def geometric_sum_terms (n : ℕ) : ℝ := 2^(n-1) - 0.5

theorem geometric_sequence_solution (n : ℕ) (a1 a4 : ℝ) (sequence : ℕ → ℝ):
  a1 = 0.5 → a4 = 4 → (sequence n = geometric_general_term n) ∧ (sequence n = geometric_sum_terms n) := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_geometric_sequence_solution_l984_98402


namespace NUMINAMATH_GPT_infinitely_many_solutions_l984_98431

theorem infinitely_many_solutions (b : ℝ) : (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := sorry

end NUMINAMATH_GPT_infinitely_many_solutions_l984_98431


namespace NUMINAMATH_GPT_add_base_3_l984_98411

def base3_addition : Prop :=
  2 + (1 * 3^2 + 2 * 3^1 + 0 * 3^0) + 
  (2 * 3^2 + 0 * 3^1 + 1 * 3^0) + 
  (1 * 3^3 + 2 * 3^1 + 0 * 3^0) = 
  (1 * 3^3) + (1 * 3^2) + (0 * 3^1) + (2 * 3^0)

theorem add_base_3 : base3_addition :=
by 
  -- We will skip the proof as per instructions
  sorry

end NUMINAMATH_GPT_add_base_3_l984_98411


namespace NUMINAMATH_GPT_area_of_right_triangle_l984_98471

theorem area_of_right_triangle
  (BC AC : ℝ)
  (h1 : BC * AC = 16) : 
  0.5 * BC * AC = 8 := by 
  sorry

end NUMINAMATH_GPT_area_of_right_triangle_l984_98471


namespace NUMINAMATH_GPT_quadratic_rewrite_sum_l984_98463

theorem quadratic_rewrite_sum :
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  a + b + c = 143.25 :=
by 
  let a : ℝ := -3
  let b : ℝ := 9 / 2
  let c : ℝ := 567 / 4
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_sum_l984_98463


namespace NUMINAMATH_GPT_range_of_a_l984_98475

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x^2 + 2 * |x - a| ≥ a^2) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l984_98475


namespace NUMINAMATH_GPT_david_wins_2011th_even_l984_98493

theorem david_wins_2011th_even :
  ∃ n : ℕ, (∃ k : ℕ, k = 2011 ∧ n = 2 * k) ∧ (∀ a b : ℕ, a < b → a + b < b * a) ∧ (n % 2 = 0) := 
sorry

end NUMINAMATH_GPT_david_wins_2011th_even_l984_98493


namespace NUMINAMATH_GPT_computation_correct_l984_98450

theorem computation_correct : 12 * ((216 / 3) + (36 / 6) + (16 / 8) + 2) = 984 := 
by 
  sorry

end NUMINAMATH_GPT_computation_correct_l984_98450


namespace NUMINAMATH_GPT_buddy_cards_on_thursday_is_32_l984_98480

def buddy_cards_on_monday := 30
def buddy_cards_on_tuesday := buddy_cards_on_monday / 2
def buddy_cards_on_wednesday := buddy_cards_on_tuesday + 12
def buddy_cards_bought_on_thursday := buddy_cards_on_tuesday / 3
def buddy_cards_on_thursday := buddy_cards_on_wednesday + buddy_cards_bought_on_thursday

theorem buddy_cards_on_thursday_is_32 : buddy_cards_on_thursday = 32 :=
by sorry

end NUMINAMATH_GPT_buddy_cards_on_thursday_is_32_l984_98480


namespace NUMINAMATH_GPT_problem_solution_l984_98484

theorem problem_solution (x y z : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) (h3 : 0.6 * y = z) : 
  z = 60 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l984_98484


namespace NUMINAMATH_GPT_red_sequence_57_eq_103_l984_98409

-- Definitions based on conditions described in the problem
def red_sequence : Nat → Nat
| 0 => 1  -- First number is 1
| 1 => 2  -- Next even number
| 2 => 4  -- Next even number
-- Continue defining based on patterns from problem
| (n+3) => -- Each element recursively following the pattern
 sorry  -- Detailed pattern definition is skipped

-- Main theorem: the 57th number in the red subsequence is 103
theorem red_sequence_57_eq_103 : red_sequence 56 = 103 :=
 sorry

end NUMINAMATH_GPT_red_sequence_57_eq_103_l984_98409


namespace NUMINAMATH_GPT_no_food_dogs_l984_98427

theorem no_food_dogs (total_dogs watermelon_liking salmon_liking chicken_liking ws_liking sc_liking wc_liking wsp_liking : ℕ) 
    (h_total : total_dogs = 100)
    (h_watermelon : watermelon_liking = 20) 
    (h_salmon : salmon_liking = 70) 
    (h_chicken : chicken_liking = 10) 
    (h_ws : ws_liking = 10) 
    (h_sc : sc_liking = 5) 
    (h_wc : wc_liking = 3) 
    (h_wsp : wsp_liking = 2) :
    (total_dogs - ((watermelon_liking - ws_liking - wc_liking + wsp_liking) + 
    (salmon_liking - ws_liking - sc_liking + wsp_liking) + 
    (chicken_liking - sc_liking - wc_liking + wsp_liking) + 
    (ws_liking - wsp_liking) + 
    (sc_liking - wsp_liking) + 
    (wc_liking - wsp_liking) + wsp_liking)) = 28 :=
  by sorry

end NUMINAMATH_GPT_no_food_dogs_l984_98427


namespace NUMINAMATH_GPT_distinct_stone_arrangements_l984_98481

-- Define the set of 12 unique stones
def stones := Finset.range 12

-- Define the number of unique placements without considering symmetries
def placements : ℕ := stones.card.factorial

-- Define the number of symmetries (6 rotations and 6 reflections)
def symmetries : ℕ := 12

-- The total number of distinct configurations accounting for symmetries
def distinct_arrangements : ℕ := placements / symmetries

-- The main theorem stating the number of distinct arrangements
theorem distinct_stone_arrangements : distinct_arrangements = 39916800 := by 
  sorry

end NUMINAMATH_GPT_distinct_stone_arrangements_l984_98481


namespace NUMINAMATH_GPT_roots_of_cubic_l984_98487

/-- Let p, q, and r be the roots of the polynomial x^3 - 15x^2 + 10x + 24 = 0. 
   The value of (1 + p)(1 + q)(1 + r) is equal to 2. -/
theorem roots_of_cubic (p q r : ℝ)
  (h1 : p + q + r = 15)
  (h2 : p * q + q * r + r * p = 10)
  (h3 : p * q * r = -24) :
  (1 + p) * (1 + q) * (1 + r) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_roots_of_cubic_l984_98487


namespace NUMINAMATH_GPT_cake_slices_l984_98497

theorem cake_slices (S : ℕ) (h : 347 * S = 6 * 375 + 526) : S = 8 :=
sorry

end NUMINAMATH_GPT_cake_slices_l984_98497


namespace NUMINAMATH_GPT_min_value_n_l984_98465

noncomputable def minN : ℕ :=
  5

theorem min_value_n :
  ∀ (S : Finset ℕ), (∀ n ∈ S, 1 ≤ n ∧ n ≤ 9) ∧ S.card = minN → 
    (∃ T ⊆ S, T ≠ ∅ ∧ 10 ∣ (T.sum id)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_n_l984_98465


namespace NUMINAMATH_GPT_find_radius_l984_98476

noncomputable def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Prop := sorry

theorem find_radius (C1 : ℝ × ℝ × ℝ) (r1 : ℝ) (C2 : ℝ × ℝ × ℝ) (r : ℝ) :
  C1 = (3, 5, 0) →
  r1 = 2 →
  C2 = (0, 5, -8) →
  (sphere ((3, 5, -8) : ℝ × ℝ × ℝ) (2 * Real.sqrt 17)) →
  r = Real.sqrt 59 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_radius_l984_98476


namespace NUMINAMATH_GPT_popsicle_sticks_sum_l984_98446

-- Define the number of popsicle sticks each person has
def Gino_popsicle_sticks : Nat := 63
def my_popsicle_sticks : Nat := 50

-- Formulate the theorem stating the sum of popsicle sticks
theorem popsicle_sticks_sum : Gino_popsicle_sticks + my_popsicle_sticks = 113 := by
  sorry

end NUMINAMATH_GPT_popsicle_sticks_sum_l984_98446


namespace NUMINAMATH_GPT_seventh_term_value_l984_98498

theorem seventh_term_value (a d : ℤ) (h1 : a = 12) (h2 : a + 3 * d = 18) : a + 6 * d = 24 := 
by
  sorry

end NUMINAMATH_GPT_seventh_term_value_l984_98498


namespace NUMINAMATH_GPT_mr_green_yield_l984_98410

noncomputable def steps_to_feet (steps : ℕ) : ℝ :=
  steps * 2.5

noncomputable def total_yield (steps_x : ℕ) (steps_y : ℕ) (yield_potato_per_sqft : ℝ) (yield_carrot_per_sqft : ℝ) : ℝ :=
  let width := steps_to_feet steps_x
  let height := steps_to_feet steps_y
  let area := width * height
  (area * yield_potato_per_sqft) + (area * yield_carrot_per_sqft)

theorem mr_green_yield :
  total_yield 20 25 0.5 0.25 = 2343.75 :=
by
  sorry

end NUMINAMATH_GPT_mr_green_yield_l984_98410


namespace NUMINAMATH_GPT_other_diagonal_length_l984_98416

theorem other_diagonal_length (d2 : ℝ) (A : ℝ) (d1 : ℝ) 
  (h1 : d2 = 120) 
  (h2 : A = 4800) 
  (h3 : A = (d1 * d2) / 2) : d1 = 80 :=
by
  sorry

end NUMINAMATH_GPT_other_diagonal_length_l984_98416


namespace NUMINAMATH_GPT_nathan_has_83_bananas_l984_98485

def nathan_bananas (bunches_eight bananas_eight bunches_seven bananas_seven: Nat) : Nat :=
  bunches_eight * bananas_eight + bunches_seven * bananas_seven

theorem nathan_has_83_bananas (h1 : bunches_eight = 6) (h2 : bananas_eight = 8) (h3 : bunches_seven = 5) (h4 : bananas_seven = 7) : 
  nathan_bananas bunches_eight bananas_eight bunches_seven bananas_seven = 83 := by
  sorry

end NUMINAMATH_GPT_nathan_has_83_bananas_l984_98485


namespace NUMINAMATH_GPT_total_water_in_heaters_l984_98457

theorem total_water_in_heaters (wallace_capacity : ℕ) (catherine_capacity : ℕ) 
(wallace_water : ℕ) (catherine_water : ℕ) :
  wallace_capacity = 40 →
  (wallace_water = (3 * wallace_capacity) / 4) →
  wallace_capacity = 2 * catherine_capacity →
  (catherine_water = (3 * catherine_capacity) / 4) →
  wallace_water + catherine_water = 45 :=
by
  sorry

end NUMINAMATH_GPT_total_water_in_heaters_l984_98457


namespace NUMINAMATH_GPT_union_A_B_l984_98451

def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

theorem union_A_B : A ∪ B = {x | x > 0} :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_l984_98451


namespace NUMINAMATH_GPT_lattice_point_count_l984_98403

theorem lattice_point_count :
  (∃ (S : Finset (ℤ × ℤ)), S.card = 16 ∧ ∀ (p : ℤ × ℤ), p ∈ S → (|p.1| - 1) ^ 2 + (|p.2| - 1) ^ 2 < 2) :=
sorry

end NUMINAMATH_GPT_lattice_point_count_l984_98403


namespace NUMINAMATH_GPT_quadratic_intersects_x_axis_l984_98478

theorem quadratic_intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x : ℝ, a * x^2 - (3 * a + 1) * x + 3 = 0 := 
by {
  -- The proof will go here
  sorry
}

end NUMINAMATH_GPT_quadratic_intersects_x_axis_l984_98478


namespace NUMINAMATH_GPT_inverse_of_problem_matrix_is_zero_matrix_l984_98482

def det (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

def zero_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 0], ![0, 0]]

noncomputable def problem_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, -6], ![-2, 3]]

theorem inverse_of_problem_matrix_is_zero_matrix :
  det problem_matrix = 0 → problem_matrix⁻¹ = zero_matrix :=
by
  intro h
  -- Proof steps will be written here
  sorry

end NUMINAMATH_GPT_inverse_of_problem_matrix_is_zero_matrix_l984_98482


namespace NUMINAMATH_GPT_sum_of_remainders_mod_11_l984_98434

theorem sum_of_remainders_mod_11
    (a b c d : ℤ)
    (h₁ : a % 11 = 2)
    (h₂ : b % 11 = 4)
    (h₃ : c % 11 = 6)
    (h₄ : d % 11 = 8) :
    (a + b + c + d) % 11 = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_11_l984_98434


namespace NUMINAMATH_GPT_difference_in_cans_l984_98417

-- Definitions of the conditions
def total_cans_collected : ℕ := 9
def cans_in_bag : ℕ := 7

-- Statement of the proof problem
theorem difference_in_cans :
  total_cans_collected - cans_in_bag = 2 := by
  sorry

end NUMINAMATH_GPT_difference_in_cans_l984_98417


namespace NUMINAMATH_GPT_expr_eval_l984_98488

theorem expr_eval : 3^3 - 3^2 + 3^1 - 3^0 = 20 := by
  sorry

end NUMINAMATH_GPT_expr_eval_l984_98488


namespace NUMINAMATH_GPT_interest_rate_l984_98444

theorem interest_rate (total_investment : ℝ) (investment1 : ℝ) (investment2 : ℝ) (rate2 : ℝ) (interest1 : ℝ → ℝ) (interest2 : ℝ → ℝ) :
  (total_investment = 5400) →
  (investment1 = 3000) →
  (investment2 = total_investment - investment1) →
  (rate2 = 0.10) →
  (interest1 investment1 = investment1 * (interest1 1)) →
  (interest2 investment2 = investment2 * rate2) →
  interest1 investment1 = interest2 investment2 →
  interest1 1 = 0.08 :=
by
  intros
  sorry

end NUMINAMATH_GPT_interest_rate_l984_98444


namespace NUMINAMATH_GPT_ratio_roses_to_lilacs_l984_98424

theorem ratio_roses_to_lilacs
  (L: ℕ) -- number of lilacs sold
  (G: ℕ) -- number of gardenias sold
  (R: ℕ) -- number of roses sold
  (hL: L = 10) -- defining lilacs sold as 10
  (hG: G = L / 2) -- defining gardenias sold as half the lilacs
  (hTotal: R + L + G = 45) -- defining total flowers sold as 45
  : R / L = 3 :=
by {
  -- The actual proof would go here, but we skip it as per instructions
  sorry
}

end NUMINAMATH_GPT_ratio_roses_to_lilacs_l984_98424


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l984_98456

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l984_98456


namespace NUMINAMATH_GPT_election_total_polled_votes_l984_98455

theorem election_total_polled_votes (V : ℝ) (invalid_votes : ℝ) (candidate_votes : ℝ) (margin : ℝ)
  (h1 : candidate_votes = 0.3 * V)
  (h2 : margin = 5000)
  (h3 : V = 0.3 * V + (0.3 * V + margin))
  (h4 : invalid_votes = 100) :
  V + invalid_votes = 12600 :=
by
  sorry

end NUMINAMATH_GPT_election_total_polled_votes_l984_98455


namespace NUMINAMATH_GPT_xiaoxia_exceeds_xiaoming_l984_98447

theorem xiaoxia_exceeds_xiaoming (n : ℕ) : 
  52 + 15 * n > 70 + 12 * n := 
sorry

end NUMINAMATH_GPT_xiaoxia_exceeds_xiaoming_l984_98447


namespace NUMINAMATH_GPT_quadratic_common_root_inverse_other_roots_l984_98469

variables (p q r s : ℝ)
variables (hq : q ≠ -1) (hs : s ≠ -1)

theorem quadratic_common_root_inverse_other_roots :
  (∃ a b : ℝ, (a ≠ b) ∧ (a^2 + p * a + q = 0) ∧ (a * b = 1) ∧ (b^2 + r * b + s = 0)) ↔ 
  (p * r = (q + 1) * (s + 1) ∧ p * (q + 1) * s = r * (s + 1) * q) :=
sorry

end NUMINAMATH_GPT_quadratic_common_root_inverse_other_roots_l984_98469


namespace NUMINAMATH_GPT_incorrect_statement_l984_98494

-- Conditions
variable (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
variables (triangleABC : Triangle A B C) (triangleDEF : Triangle D E F)

-- Congruence of triangles
axiom congruent_triangles : triangleABC ≌ triangleDEF

-- Proving incorrect statement
theorem incorrect_statement : ¬ (AB = EF) := by
  sorry

end NUMINAMATH_GPT_incorrect_statement_l984_98494


namespace NUMINAMATH_GPT_triangle_perimeter_l984_98439

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 19)
  (ineq1 : a + b > c) (ineq2 : a + c > b) (ineq3 : b + c > a) : a + b + c = 44 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l984_98439


namespace NUMINAMATH_GPT_speed_ratio_l984_98479

theorem speed_ratio (v_A v_B : ℝ) (t : ℝ) (h1 : v_A = 200 / t) (h2 : v_B = 120 / t) : 
  v_A / v_B = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_speed_ratio_l984_98479


namespace NUMINAMATH_GPT_num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l984_98433

def total_students : ℕ := 800

def percentage_blue_shirts : ℕ := 45
def percentage_red_shirts : ℕ := 23
def percentage_green_shirts : ℕ := 15

def percentage_black_pants : ℕ := 30
def percentage_khaki_pants : ℕ := 25
def percentage_jeans_pants : ℕ := 10

def percentage_white_shoes : ℕ := 40
def percentage_black_shoes : ℕ := 20
def percentage_brown_shoes : ℕ := 15

def students_other_color_shirts : ℕ :=
  total_students * (100 - (percentage_blue_shirts + percentage_red_shirts + percentage_green_shirts)) / 100

def students_other_types_pants : ℕ :=
  total_students * (100 - (percentage_black_pants + percentage_khaki_pants + percentage_jeans_pants)) / 100

def students_other_color_shoes : ℕ :=
  total_students * (100 - (percentage_white_shoes + percentage_black_shoes + percentage_brown_shoes)) / 100

theorem num_students_other_color_shirts : students_other_color_shirts = 136 := by
  sorry

theorem num_students_other_types_pants : students_other_types_pants = 280 := by
  sorry

theorem num_students_other_color_shoes : students_other_color_shoes = 200 := by
  sorry

end NUMINAMATH_GPT_num_students_other_color_shirts_num_students_other_types_pants_num_students_other_color_shoes_l984_98433


namespace NUMINAMATH_GPT_part1_part2_l984_98425

theorem part1 (m : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - 2 * x + 4 * y - 4 = 0 ∧ y = x + m) → -3 - 3 * Real.sqrt 2 < m ∧ m < -3 + 3 * Real.sqrt 2) :=
sorry

theorem part2 (m x1 x2 y1 y2 : ℝ) (h1 : x1 + x2 = -(m + 1)) (h2 : x1 * x2 = (m^2 + 4 * m - 4) / 2) 
(h3 : (x - x1) * (x - x2) + (x1 + m) * (x2 + m) = 0) : 
  m = -4 ∨ m = 1 →
  (∀ x y : ℝ, y = x + m ↔ x - y - 4 = 0 ∨ x - y + 1 = 0) :=
sorry

end NUMINAMATH_GPT_part1_part2_l984_98425


namespace NUMINAMATH_GPT_system1_solution_system2_solution_system3_solution_l984_98405

theorem system1_solution (x y : ℝ) : 
  (x = 3/2) → (y = 1/2) → (x + 3 * y = 3) ∧ (x - y = 1) :=
by intros; sorry

theorem system2_solution (x y : ℝ) : 
  (x = 0) → (y = 2/5) → ((x + 3 * y) / 2 = 3 / 5) ∧ (5 * (x - 2 * y) = -4) :=
by intros; sorry

theorem system3_solution (x y z : ℝ) : 
  (x = 1) → (y = 2) → (z = 3) → 
  (3 * x + 4 * y + z = 14) ∧ (x + 5 * y + 2 * z = 17) ∧ (2 * x + 2 * y - z = 3) :=
by intros; sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_system3_solution_l984_98405


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l984_98467

-- Define the necessary conditions
variables {a b c d : ℝ}

-- State the main theorem
theorem necessary_but_not_sufficient (h₁ : a > b) (h₂ : c > d) : (a + c > b + d) :=
by
  -- Placeholder for the proof (insufficient as per the context problem statement)
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l984_98467


namespace NUMINAMATH_GPT_units_digit_of_j_squared_plus_3_power_j_l984_98400

def j : ℕ := 2023^3 + 3^2023 + 2023

theorem units_digit_of_j_squared_plus_3_power_j (j : ℕ) (h : j = 2023^3 + 3^2023 + 2023) : 
  ((j^2 + 3^j) % 10) = 6 := 
  sorry

end NUMINAMATH_GPT_units_digit_of_j_squared_plus_3_power_j_l984_98400


namespace NUMINAMATH_GPT_xuzhou_test_2014_l984_98492

variables (A B C D : ℝ) -- Assume A, B, C, D are real numbers.

theorem xuzhou_test_2014 :
  (C < D) → (A > B) :=
sorry

end NUMINAMATH_GPT_xuzhou_test_2014_l984_98492


namespace NUMINAMATH_GPT_surface_is_plane_l984_98428

-- Define cylindrical coordinates
structure CylindricalCoordinate where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define the property for a constant θ
def isConstantTheta (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  coord.θ = c

-- Define the plane in cylindrical coordinates
def isPlane (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  isConstantTheta c coord

-- Theorem: The surface described by θ = c in cylindrical coordinates is a plane.
theorem surface_is_plane (c : ℝ) (coord : CylindricalCoordinate) :
    isPlane c coord ↔ isConstantTheta c coord := sorry

end NUMINAMATH_GPT_surface_is_plane_l984_98428


namespace NUMINAMATH_GPT_charlyn_viewable_area_l984_98432

noncomputable def charlyn_sees_area (side_length viewing_distance : ℝ) : ℝ :=
  let inner_viewable_area := (side_length^2 - (side_length - 2 * viewing_distance)^2)
  let rectangular_area := 4 * (side_length * viewing_distance)
  let circular_corner_area := 4 * ((viewing_distance^2 * Real.pi) / 4)
  inner_viewable_area + rectangular_area + circular_corner_area

theorem charlyn_viewable_area :
  let side_length := 7
  let viewing_distance := 1.5
  charlyn_sees_area side_length viewing_distance = 82 := 
by
  sorry

end NUMINAMATH_GPT_charlyn_viewable_area_l984_98432


namespace NUMINAMATH_GPT_Paula_initial_cans_l984_98435

theorem Paula_initial_cans :
  ∀ (cans rooms_lost : ℕ), rooms_lost = 10 → 
  (40 / (rooms_lost / 5) = cans + 5 → cans = 20) :=
by
  intros cans rooms_lost h_rooms_lost h_calculation
  sorry

end NUMINAMATH_GPT_Paula_initial_cans_l984_98435


namespace NUMINAMATH_GPT_Mary_younger_than_Albert_l984_98449

-- Define the basic entities and conditions
def Betty_age : ℕ := 11
def Albert_age : ℕ := 4 * Betty_age
def Mary_age : ℕ := Albert_age / 2

-- Define the property to prove
theorem Mary_younger_than_Albert : Albert_age - Mary_age = 22 :=
by 
  sorry

end NUMINAMATH_GPT_Mary_younger_than_Albert_l984_98449


namespace NUMINAMATH_GPT_sum_of_sequence_l984_98466

variable (S a b : ℝ)

theorem sum_of_sequence :
  (S - a) / 100 = 2022 →
  (S - b) / 100 = 2023 →
  (a + b) / 2 = 51 →
  S = 202301 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_sequence_l984_98466
