import Mathlib

namespace NUMINAMATH_GPT_eval_fraction_expression_l2138_213820
noncomputable def inner_expr := 2 + 2
noncomputable def middle_expr := 2 + (1 / inner_expr)
noncomputable def outer_expr := 2 + (1 / middle_expr)

theorem eval_fraction_expression : outer_expr = 22 / 9 := by
  sorry

end NUMINAMATH_GPT_eval_fraction_expression_l2138_213820


namespace NUMINAMATH_GPT_jacqueline_guavas_l2138_213813

theorem jacqueline_guavas 
  (G : ℕ) 
  (plums : ℕ := 16) 
  (apples : ℕ := 21) 
  (given : ℕ := 40) 
  (remaining : ℕ := 15) 
  (initial_fruits : ℕ := plums + G + apples)
  (total_fruits_after_given : ℕ := remaining + given) : 
  initial_fruits = total_fruits_after_given → G = 18 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_jacqueline_guavas_l2138_213813


namespace NUMINAMATH_GPT_number_of_divisors_30_l2138_213876

theorem number_of_divisors_30 : 
  ∃ (d : ℕ), d = 2 * 2 * 2 ∧ d = 8 :=
  by sorry

end NUMINAMATH_GPT_number_of_divisors_30_l2138_213876


namespace NUMINAMATH_GPT_find_t_l2138_213894

-- Define the utility function based on hours of reading and playing basketball
def utility (reading_hours : ℝ) (basketball_hours : ℝ) : ℝ :=
  reading_hours * basketball_hours

-- Define the conditions for Wednesday and Thursday utilities
def wednesday_utility (t : ℝ) : ℝ :=
  t * (10 - t)

def thursday_utility (t : ℝ) : ℝ :=
  (3 - t) * (t + 4)

-- The main theorem stating the equivalence of utilities implies t = 3
theorem find_t (t : ℝ) (h : wednesday_utility t = thursday_utility t) : t = 3 :=
by
  -- Skip proof with sorry
  sorry

end NUMINAMATH_GPT_find_t_l2138_213894


namespace NUMINAMATH_GPT_length_of_de_l2138_213842

theorem length_of_de
  {a b c d e : ℝ} 
  (h1 : b - a = 5) 
  (h2 : c - a = 11) 
  (h3 : e - a = 22) 
  (h4 : c - b = 2 * (d - c)) :
  e - d = 8 :=
by 
  sorry

end NUMINAMATH_GPT_length_of_de_l2138_213842


namespace NUMINAMATH_GPT_chords_from_nine_points_l2138_213801

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end NUMINAMATH_GPT_chords_from_nine_points_l2138_213801


namespace NUMINAMATH_GPT_Chris_age_l2138_213824

theorem Chris_age (a b c : ℚ) 
  (h1 : a + b + c = 30)
  (h2 : c - 5 = 2 * a)
  (h3 : b = (3/4) * a - 1) :
  c = 263/11 := by
  sorry

end NUMINAMATH_GPT_Chris_age_l2138_213824


namespace NUMINAMATH_GPT_triangle_angle_side_inequality_l2138_213857

variable {A B C : Type} -- Variables for points in the triangle
variable {a b : ℝ} -- Variables for the lengths of sides opposite to angles A and B
variable {A_angle B_angle : ℝ} -- Variables for the angles at A and B in triangle ABC

-- Define that we are in a triangle setting
def is_triangle (A B C : Type) := True

-- Define the assumption for the proof by contradiction
def assumption (a b : ℝ) := a ≤ b

theorem triangle_angle_side_inequality (h_triangle : is_triangle A B C)
(h_angle : A_angle > B_angle) 
(h_assumption : assumption a b) : a > b := 
sorry

end NUMINAMATH_GPT_triangle_angle_side_inequality_l2138_213857


namespace NUMINAMATH_GPT_problem_acd_div_b_l2138_213856

theorem problem_acd_div_b (a b c d : ℤ) (x : ℝ)
    (h1 : x = (a + b * Real.sqrt c) / d)
    (h2 : (7 * x) / 4 + 2 = 6 / x) :
    (a * c * d) / b = -322 := sorry

end NUMINAMATH_GPT_problem_acd_div_b_l2138_213856


namespace NUMINAMATH_GPT_no_such_function_exists_l2138_213821

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := 
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l2138_213821


namespace NUMINAMATH_GPT_line_intersects_parabola_exactly_once_at_m_l2138_213838

theorem line_intersects_parabola_exactly_once_at_m :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 7 = m) → (∃! m : ℝ, m = 25 / 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_line_intersects_parabola_exactly_once_at_m_l2138_213838


namespace NUMINAMATH_GPT_average_boxes_per_day_by_third_day_l2138_213849

theorem average_boxes_per_day_by_third_day (day1 day2 day3_part1 day3_part2 : ℕ) :
  day1 = 318 →
  day2 = 312 →
  day3_part1 = 180 →
  day3_part2 = 162 →
  ((day1 + day2 + (day3_part1 + day3_part2)) / 3) = 324 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_boxes_per_day_by_third_day_l2138_213849


namespace NUMINAMATH_GPT_find_abc_l2138_213882

theorem find_abc (a b c : ℝ)
  (h1 : ∀ x : ℝ, (x < -6 ∨ (|x - 31| ≤ 1)) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h2 : a < b) :
  a + 2 * b + 3 * c = 76 :=
sorry

end NUMINAMATH_GPT_find_abc_l2138_213882


namespace NUMINAMATH_GPT_imo_34_l2138_213883

-- Define the input conditions
variables (R r ρ : ℝ)

-- The main theorem we need to prove
theorem imo_34 { R r ρ : ℝ } (hR : R = 1) : 
  ρ ≤ 1 - (1/3) * (1 + r)^2 :=
sorry

end NUMINAMATH_GPT_imo_34_l2138_213883


namespace NUMINAMATH_GPT_official_exchange_rate_l2138_213879

theorem official_exchange_rate (E : ℝ)
  (h1 : 70 = 10 * (7 / 5) * E) :
  E = 5 :=
by
  sorry

end NUMINAMATH_GPT_official_exchange_rate_l2138_213879


namespace NUMINAMATH_GPT_sequence_periodic_l2138_213826

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = (1 + a n) / (1 - a n)

theorem sequence_periodic :
  ∃ a : ℕ → ℝ, sequence a ∧ a 2016 = 3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_periodic_l2138_213826


namespace NUMINAMATH_GPT_ronald_laundry_frequency_l2138_213885

variable (Tim_laundry_frequency Ronald_laundry_frequency : ℕ)

theorem ronald_laundry_frequency :
  (Tim_laundry_frequency = 9) →
  (18 % Ronald_laundry_frequency = 0) →
  (18 % Tim_laundry_frequency = 0) →
  (Ronald_laundry_frequency ≠ 1) →
  (Ronald_laundry_frequency ≠ 18) →
  (Ronald_laundry_frequency ≠ 9) →
  (Ronald_laundry_frequency = 3) :=
by
  intros hTim hRonaldMultiple hTimMultiple hNot1 hNot18 hNot9
  sorry

end NUMINAMATH_GPT_ronald_laundry_frequency_l2138_213885


namespace NUMINAMATH_GPT_domain_of_log_function_l2138_213836

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem domain_of_log_function : {
  x : ℝ // ∃ y : ℝ, f y = x
} = { x : ℝ | x > 1 / 2 } := by
sorry

end NUMINAMATH_GPT_domain_of_log_function_l2138_213836


namespace NUMINAMATH_GPT_parallel_lines_solution_l2138_213803

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (x + a * y + 6 = 0) → (a - 2) * x + 3 * y + 2 * a = 0) → (a = -1) :=
by
  intro h
  -- Add more formal argument insights if needed
  sorry

end NUMINAMATH_GPT_parallel_lines_solution_l2138_213803


namespace NUMINAMATH_GPT_expenditure_increase_l2138_213822

theorem expenditure_increase (x : ℝ) (h₁ : 3 * x / (3 * x + 2 * x) = 3 / 5)
  (h₂ : 2 * x / (3 * x + 2 * x) = 2 / 5)
  (h₃ : ((5 * x) + 0.15 * (5 * x)) = 5.75 * x) 
  (h₄ : (2 * x + 0.06 * 2 * x) = 2.12 * x) 
  : ((3.63 * x - 3 * x) / (3 * x) * 100) = 21 := 
  by
  sorry

end NUMINAMATH_GPT_expenditure_increase_l2138_213822


namespace NUMINAMATH_GPT_alice_travel_time_l2138_213812

theorem alice_travel_time (distance_AB : ℝ) (bob_speed : ℝ) (alice_speed : ℝ) (max_time_diff_hr : ℝ) (time_conversion : ℝ) :
  distance_AB = 60 →
  bob_speed = 40 →
  alice_speed = 60 →
  max_time_diff_hr = 0.5 →
  time_conversion = 60 →
  max_time_diff_hr * time_conversion = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_alice_travel_time_l2138_213812


namespace NUMINAMATH_GPT_sequence_result_l2138_213814

theorem sequence_result :
  (1 + 2)^2 + 1 = 10 ∧
  (2 + 3)^2 + 1 = 26 ∧
  (4 + 5)^2 + 1 = 82 →
  (3 + 4)^2 + 1 = 50 :=
by sorry

end NUMINAMATH_GPT_sequence_result_l2138_213814


namespace NUMINAMATH_GPT_spring_bud_cup_eq_289_l2138_213807

theorem spring_bud_cup_eq_289 (x : ℕ) (h : x + x = 578) : x = 289 :=
sorry

end NUMINAMATH_GPT_spring_bud_cup_eq_289_l2138_213807


namespace NUMINAMATH_GPT_percentage_increase_l2138_213899

theorem percentage_increase (N : ℝ) (P : ℝ) (h1 : N + (P / 100) * N - (N - 25 / 100 * N) = 30) (h2 : N = 80) : P = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l2138_213899


namespace NUMINAMATH_GPT_find_number_l2138_213873

theorem find_number (n : ℕ) : gcd 30 n = 10 ∧ 70 ≤ n ∧ n ≤ 80 ∧ 200 ≤ lcm 30 n ∧ lcm 30 n ≤ 300 → (n = 70 ∨ n = 80) :=
sorry

end NUMINAMATH_GPT_find_number_l2138_213873


namespace NUMINAMATH_GPT_compare_fractions_l2138_213848

variables {a b : ℝ}

theorem compare_fractions (h : a + b > 0) : 
  (a / (b^2)) + (b / (a^2)) ≥ (1 / a) + (1 / b) :=
sorry

end NUMINAMATH_GPT_compare_fractions_l2138_213848


namespace NUMINAMATH_GPT_find_value_of_a4_plus_a5_l2138_213851

variables {S_n : ℕ → ℕ} {a_n : ℕ → ℕ} {d : ℤ} 

-- Conditions
def arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (d : ℤ) : Prop :=
∀ n : ℕ, S_n n = n * a_n 1 + (n * (n - 1) / 2) * d

def a_3_S_3_condition (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) : Prop := 
a_n 3 = 3 ∧ S_n 3 = 3

-- Question
theorem find_value_of_a4_plus_a5 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℤ):
  arithmetic_sequence_sum S_n a_n d →
  a_3_S_3_condition a_n S_n →
  a_n 4 + a_n 5 = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a4_plus_a5_l2138_213851


namespace NUMINAMATH_GPT_farmer_goats_sheep_unique_solution_l2138_213816

theorem farmer_goats_sheep_unique_solution:
  ∃ g h : ℕ, 0 < g ∧ 0 < h ∧ 28 * g + 30 * h = 1200 ∧ h > g :=
by
  sorry

end NUMINAMATH_GPT_farmer_goats_sheep_unique_solution_l2138_213816


namespace NUMINAMATH_GPT_sum_gcd_lcm_l2138_213827

theorem sum_gcd_lcm (a b : ℕ) (h_a : a = 75) (h_b : b = 4500) :
  Nat.gcd a b + Nat.lcm a b = 4575 := by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_l2138_213827


namespace NUMINAMATH_GPT_TV_height_l2138_213802

theorem TV_height (Area Width Height : ℝ) (h_area : Area = 21) (h_width : Width = 3) (h_area_def : Area = Width * Height) : Height = 7 := 
by
  sorry

end NUMINAMATH_GPT_TV_height_l2138_213802


namespace NUMINAMATH_GPT_smallest_integer_l2138_213854

theorem smallest_integer {x y z : ℕ} (h1 : 2*y = x) (h2 : 3*y = z) (h3 : x + y + z = 60) : y = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_l2138_213854


namespace NUMINAMATH_GPT_triangle_right_triangle_l2138_213861

variable {A B C : Real}  -- Define the angles A, B, and C

theorem triangle_right_triangle (sin_A sin_B sin_C : Real)
  (h : sin_A^2 + sin_B^2 = sin_C^2) 
  (triangle_cond : A + B + C = 180) : 
  (A = 90) ∨ (B = 90) ∨ (C = 90) := 
  sorry

end NUMINAMATH_GPT_triangle_right_triangle_l2138_213861


namespace NUMINAMATH_GPT_sum_of_x_l2138_213818

-- define the function f as an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- define the function f as strictly monotonic on the interval (0, +∞)
def is_strictly_monotonic_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- define the main problem statement
theorem sum_of_x (f : ℝ → ℝ) (x : ℝ) (h1 : is_even_function f) (h2 : is_strictly_monotonic_on_positive f) (h3 : x ≠ 0)
  (hx : f (x^2 - 2*x - 1) = f (x + 1)) : 
  ∃ (x1 x2 x3 x4 : ℝ), (x1 + x2 + x3 + x4 = 4) ∧
                        (x1^2 - 3*x1 - 2 = 0) ∧
                        (x2^2 - 3*x2 - 2 = 0) ∧
                        (x3^2 - x3 = 0) ∧
                        (x4^2 - x4 = 0) :=
sorry

end NUMINAMATH_GPT_sum_of_x_l2138_213818


namespace NUMINAMATH_GPT_bucket_full_weight_l2138_213800

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
    (h1 : x + (3/4) * y = p) 
    (h2 : x + (1/3) * y = q) : 
    x + y = (8 * p - 3 * q) / 5 :=
sorry

end NUMINAMATH_GPT_bucket_full_weight_l2138_213800


namespace NUMINAMATH_GPT_find_m_n_l2138_213825

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem find_m_n (m n : ℕ) (h1 : binom (n+1) (m+1) / binom (n+1) m = 5 / 3) 
  (h2 : binom (n+1) m / binom (n+1) (m-1) = 5 / 3) : m = 3 ∧ n = 6 :=
  sorry

end NUMINAMATH_GPT_find_m_n_l2138_213825


namespace NUMINAMATH_GPT_inequality_1_inequality_2_l2138_213898

noncomputable def f (x : ℝ) : ℝ := |x - 2| - 3
noncomputable def g (x : ℝ) : ℝ := |x + 3|

theorem inequality_1 (x : ℝ) : f x < g x ↔ x > -2 := 
by sorry

theorem inequality_2 (a : ℝ) : (∀ x : ℝ, f x < g x + a) ↔ a > 2 := 
by sorry

end NUMINAMATH_GPT_inequality_1_inequality_2_l2138_213898


namespace NUMINAMATH_GPT_pie_piece_cost_l2138_213808

theorem pie_piece_cost (pieces_per_pie : ℕ) (pies_per_hour : ℕ) (total_earnings : ℝ) :
  pieces_per_pie = 3 → pies_per_hour = 12 → total_earnings = 138 →
  (total_earnings / (pieces_per_pie * pies_per_hour)) = 3.83 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_pie_piece_cost_l2138_213808


namespace NUMINAMATH_GPT_area_of_gray_region_l2138_213839

theorem area_of_gray_region :
  (radius_smaller = (2 : ℝ) / 2) →
  (radius_larger = 4 * radius_smaller) →
  (gray_area = π * radius_larger ^ 2 - π * radius_smaller ^ 2) →
  gray_area = 15 * π :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_area_of_gray_region_l2138_213839


namespace NUMINAMATH_GPT_find_years_lent_to_B_l2138_213837

def principal_B := 5000
def principal_C := 3000
def rate := 8
def time_C := 4
def total_interest := 1760

-- Interest calculation for B
def interest_B (n : ℕ) := (principal_B * rate * n) / 100

-- Interest calculation for C (constant time of 4 years)
def interest_C := (principal_C * rate * time_C) / 100

-- Total interest received
def total_interest_received (n : ℕ) := interest_B n + interest_C

theorem find_years_lent_to_B (n : ℕ) (h : total_interest_received n = total_interest) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_years_lent_to_B_l2138_213837


namespace NUMINAMATH_GPT_apples_per_pie_l2138_213841

theorem apples_per_pie (total_apples : ℕ) (apples_given : ℕ) (pies : ℕ) : 
  total_apples = 47 ∧ apples_given = 27 ∧ pies = 5 →
  (total_apples - apples_given) / pies = 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_apples_per_pie_l2138_213841


namespace NUMINAMATH_GPT_ball_bounce_height_l2138_213830

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (3 / 4 : ℝ)^k < 2) ∧ ∀ n < k, ¬ (20 * (3 / 4 : ℝ)^n < 2) :=
sorry

end NUMINAMATH_GPT_ball_bounce_height_l2138_213830


namespace NUMINAMATH_GPT_james_tip_percentage_l2138_213875

theorem james_tip_percentage :
  let ticket_cost : ℝ := 100
  let dinner_cost : ℝ := 120
  let limo_cost_per_hour : ℝ := 80
  let limo_hours : ℕ := 6
  let total_cost_with_tip : ℝ := 836
  let total_cost_without_tip : ℝ := 2 * ticket_cost + limo_hours * limo_cost_per_hour + dinner_cost
  let tip : ℝ := total_cost_with_tip - total_cost_without_tip
  let percentage_tip : ℝ := (tip / dinner_cost) * 100
  percentage_tip = 30 :=
by
  sorry

end NUMINAMATH_GPT_james_tip_percentage_l2138_213875


namespace NUMINAMATH_GPT_benny_lunch_cost_l2138_213877

theorem benny_lunch_cost :
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  total_cost = 24 :=
by
  let person := 3;
  let cost_per_lunch := 8;
  let total_cost := person * cost_per_lunch;
  have h : total_cost = 24 := by
    sorry
  exact h

end NUMINAMATH_GPT_benny_lunch_cost_l2138_213877


namespace NUMINAMATH_GPT_total_credit_hours_l2138_213853

def max_courses := 40
def max_courses_per_semester := 5
def max_courses_per_semester_credit := 3
def max_additional_courses_last_semester := 2
def max_additional_course_credit := 4
def sid_courses_multiplier := 4
def sid_additional_courses_multiplier := 2

theorem total_credit_hours (total_max_courses : Nat) 
                           (avg_max_courses_per_semester : Nat) 
                           (max_course_credit : Nat) 
                           (extra_max_courses_last_sem : Nat) 
                           (extra_max_course_credit : Nat) 
                           (sid_courses_mult : Nat) 
                           (sid_extra_courses_mult : Nat) 
                           (max_total_courses : total_max_courses = max_courses)
                           (max_avg_courses_per_semester : avg_max_courses_per_semester = max_courses_per_semester)
                           (max_course_credit_def : max_course_credit = max_courses_per_semester_credit)
                           (extra_max_courses_last_sem_def : extra_max_courses_last_sem = max_additional_courses_last_semester)
                           (extra_max_courses_credit_def : extra_max_course_credit = max_additional_course_credit)
                           (sid_courses_mult_def : sid_courses_mult = sid_courses_multiplier)
                           (sid_extra_courses_mult_def : sid_extra_courses_mult = sid_additional_courses_multiplier) : 
  total_max_courses * max_course_credit + extra_max_courses_last_sem * extra_max_course_credit + 
  (sid_courses_mult * total_max_courses - sid_extra_courses_mult * extra_max_courses_last_sem) * max_course_credit + sid_extra_courses_mult * extra_max_courses_last_sem * extra_max_course_credit = 606 := 
  by 
    sorry

end NUMINAMATH_GPT_total_credit_hours_l2138_213853


namespace NUMINAMATH_GPT_advertisement_broadcasting_methods_l2138_213865

/-- A TV station is broadcasting 5 different advertisements.
There are 3 different commercial advertisements.
There are 2 different Olympic promotional advertisements.
The last advertisement must be an Olympic promotional advertisement.
The two Olympic promotional advertisements cannot be broadcast consecutively.
Prove that the total number of different broadcasting methods is 18. -/
theorem advertisement_broadcasting_methods : 
  ∃ (arrangements : ℕ), arrangements = 18 := sorry

end NUMINAMATH_GPT_advertisement_broadcasting_methods_l2138_213865


namespace NUMINAMATH_GPT_triangle_area_decrease_l2138_213871

theorem triangle_area_decrease (B H : ℝ) : 
  let A_original := (B * H) / 2
  let H_new := 0.60 * H
  let B_new := 1.40 * B
  let A_new := (B_new * H_new) / 2
  A_new = 0.42 * A_original :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_decrease_l2138_213871


namespace NUMINAMATH_GPT_arrival_time_difference_l2138_213860

theorem arrival_time_difference
  (d : ℝ) (r_H : ℝ) (r_A : ℝ) (h₁ : d = 2) (h₂ : r_H = 12) (h₃ : r_A = 6) :
  (d / r_A * 60) - (d / r_H * 60) = 10 :=
by
  sorry

end NUMINAMATH_GPT_arrival_time_difference_l2138_213860


namespace NUMINAMATH_GPT_equation_zero_solution_l2138_213896

-- Define the conditions and the answer
def equation_zero (x : ℝ) : Prop := (x^2 + x - 2) / (x - 1) = 0
def non_zero_denominator (x : ℝ) : Prop := x - 1 ≠ 0
def solution_x (x : ℝ) : Prop := x = -2

-- The main theorem
theorem equation_zero_solution (x : ℝ) (h1 : equation_zero x) (h2 : non_zero_denominator x) : solution_x x := 
sorry

end NUMINAMATH_GPT_equation_zero_solution_l2138_213896


namespace NUMINAMATH_GPT_factor_correct_l2138_213806

noncomputable def factor_fraction (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_correct (a b c : ℝ) : 
  factor_fraction a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end NUMINAMATH_GPT_factor_correct_l2138_213806


namespace NUMINAMATH_GPT_large_planks_need_15_nails_l2138_213835

-- Definitions based on given conditions
def total_nails : ℕ := 20
def small_planks_nails : ℕ := 5

-- Question: How many nails do the large planks need together?
-- Prove that the large planks need 15 nails together given the conditions.
theorem large_planks_need_15_nails : total_nails - small_planks_nails = 15 :=
by
  sorry

end NUMINAMATH_GPT_large_planks_need_15_nails_l2138_213835


namespace NUMINAMATH_GPT_trajectory_is_eight_rays_l2138_213862

open Real

def trajectory_of_point (x y : ℝ) : Prop :=
  abs (abs x - abs y) = 2

theorem trajectory_is_eight_rays :
  ∃ (x y : ℝ), trajectory_of_point x y :=
sorry

end NUMINAMATH_GPT_trajectory_is_eight_rays_l2138_213862


namespace NUMINAMATH_GPT_max_full_marks_probability_l2138_213823

-- Define the total number of mock exams
def total_mock_exams : ℕ := 20
-- Define the number of full marks scored in mock exams
def full_marks_in_mocks : ℕ := 8

-- Define the probability of event A (scoring full marks in the first test)
def P_A : ℚ := full_marks_in_mocks / total_mock_exams

-- Define the probability of not scoring full marks in the first test
def P_neg_A : ℚ := 1 - P_A

-- Define the probability of event B (scoring full marks in the second test)
def P_B : ℚ := 1 / 2

-- Define the maximum probability of scoring full marks in either the first or the second test
def max_probability : ℚ := P_A + P_neg_A * P_B

-- The main theorem conjecture
theorem max_full_marks_probability :
  max_probability = 7 / 10 :=
by
  -- Inserting placeholder to skip the proof for now
  sorry

end NUMINAMATH_GPT_max_full_marks_probability_l2138_213823


namespace NUMINAMATH_GPT_new_songs_added_l2138_213878

-- Define the initial, deleted, and final total number of songs as constants
def initial_songs : ℕ := 8
def deleted_songs : ℕ := 5
def total_songs_now : ℕ := 33

-- Define and prove the number of new songs added
theorem new_songs_added : total_songs_now - (initial_songs - deleted_songs) = 30 :=
by
  sorry

end NUMINAMATH_GPT_new_songs_added_l2138_213878


namespace NUMINAMATH_GPT_moles_of_H2O_formed_l2138_213809

theorem moles_of_H2O_formed
  (moles_H2SO4 : ℕ)
  (moles_H2O : ℕ)
  (H : moles_H2SO4 = 3)
  (H' : moles_H2O = 3) :
  moles_H2O = 3 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_H2O_formed_l2138_213809


namespace NUMINAMATH_GPT_Harold_speed_is_one_more_l2138_213869

variable (Adrienne_speed Harold_speed : ℝ)
variable (distance_when_Harold_catches_Adr : ℝ)
variable (time_difference : ℝ)

axiom Adrienne_speed_def : Adrienne_speed = 3
axiom Harold_catches_distance : distance_when_Harold_catches_Adr = 12
axiom time_difference_def : time_difference = 1

theorem Harold_speed_is_one_more :
  Harold_speed - Adrienne_speed = 1 :=
by 
  have Adrienne_time := (distance_when_Harold_catches_Adr - Adrienne_speed * time_difference) / Adrienne_speed 
  have Harold_time := distance_when_Harold_catches_Adr / Harold_speed
  have := Adrienne_time = Harold_time - time_difference
  sorry

end NUMINAMATH_GPT_Harold_speed_is_one_more_l2138_213869


namespace NUMINAMATH_GPT_total_students_sampled_l2138_213852

theorem total_students_sampled (freq_ratio : ℕ → ℕ → ℕ) (second_group_freq : ℕ) 
  (ratio_condition : freq_ratio 2 1 = 2 ∧ freq_ratio 2 3 = 3) : 
  (6 + second_group_freq + 18) = 48 := 
by 
  sorry

end NUMINAMATH_GPT_total_students_sampled_l2138_213852


namespace NUMINAMATH_GPT_binary_addition_l2138_213866

theorem binary_addition (a b : ℕ) :
  (a = (2^0 + 2^2 + 2^4 + 2^6)) → (b = (2^0 + 2^3 + 2^6)) →
  (a + b = 158) :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_binary_addition_l2138_213866


namespace NUMINAMATH_GPT_speed_in_still_water_l2138_213840

-- Define the given conditions
def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

-- State the theorem to be proven
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 40 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l2138_213840


namespace NUMINAMATH_GPT_number_base_addition_l2138_213892

theorem number_base_addition (A B : ℕ) (h1: A = 2 * B) (h2: 2 * B^2 + 2 * B + 4 + 10 * B + 5 = (3 * B)^2 + 3 * (3 * B) + 4) : 
  A + B = 9 := 
by 
  sorry

end NUMINAMATH_GPT_number_base_addition_l2138_213892


namespace NUMINAMATH_GPT_sum_of_transformed_numbers_l2138_213843

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
    3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_transformed_numbers_l2138_213843


namespace NUMINAMATH_GPT_p_arithmetic_square_root_l2138_213864

theorem p_arithmetic_square_root {p : ℕ} (hp : p ≠ 2) (a : ℤ) (ha : a ≠ 0) :
  (∃ x1 x2 : ℤ, x1^2 = a ∧ x2^2 = a ∧ x1 ≠ x2) ∨ ¬ (∃ x : ℤ, x^2 = a) :=
  sorry

end NUMINAMATH_GPT_p_arithmetic_square_root_l2138_213864


namespace NUMINAMATH_GPT_white_area_of_sign_l2138_213874

theorem white_area_of_sign :
  let total_area : ℕ := 6 * 18
  let black_area_C : ℕ := 11
  let black_area_A : ℕ := 10
  let black_area_F : ℕ := 12
  let black_area_E : ℕ := 9
  let total_black_area : ℕ := black_area_C + black_area_A + black_area_F + black_area_E
  let white_area : ℕ := total_area - total_black_area
  white_area = 66 := by
  sorry

end NUMINAMATH_GPT_white_area_of_sign_l2138_213874


namespace NUMINAMATH_GPT_dot_product_a_b_l2138_213832

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (4, -3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Statement of the theorem to prove
theorem dot_product_a_b : dot_product vector_a vector_b = -1 := 
by sorry

end NUMINAMATH_GPT_dot_product_a_b_l2138_213832


namespace NUMINAMATH_GPT_binary_multiplication_correct_l2138_213850

theorem binary_multiplication_correct :
  (0b1101 : ℕ) * (0b1011 : ℕ) = (0b10011011 : ℕ) :=
by
  sorry

end NUMINAMATH_GPT_binary_multiplication_correct_l2138_213850


namespace NUMINAMATH_GPT_buzz_waiter_ratio_l2138_213855

def total_slices : Nat := 78
def waiter_condition (W : Nat) : Prop := W - 20 = 28

theorem buzz_waiter_ratio (W : Nat) (h : waiter_condition W) : 
  let buzz_slices := total_slices - W
  let ratio_buzz_waiter := buzz_slices / W
  ratio_buzz_waiter = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_buzz_waiter_ratio_l2138_213855


namespace NUMINAMATH_GPT_profit_percent_calc_l2138_213833

theorem profit_percent_calc (SP CP : ℝ) (h : CP = 0.25 * SP) : (SP - CP) / CP * 100 = 300 :=
by
  sorry

end NUMINAMATH_GPT_profit_percent_calc_l2138_213833


namespace NUMINAMATH_GPT_scientific_notation_of_12000_l2138_213829

theorem scientific_notation_of_12000 : 12000 = 1.2 * 10^4 := 
by sorry

end NUMINAMATH_GPT_scientific_notation_of_12000_l2138_213829


namespace NUMINAMATH_GPT_factorize_expression_l2138_213828

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l2138_213828


namespace NUMINAMATH_GPT_find_x_l2138_213811

theorem find_x (x : ℝ) (h : x ^ 2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
sorry

end NUMINAMATH_GPT_find_x_l2138_213811


namespace NUMINAMATH_GPT_four_digit_numbers_divisible_by_17_l2138_213859

theorem four_digit_numbers_divisible_by_17 :
  ∃ n, (∀ x, 1000 ≤ x ∧ x ≤ 9999 ∧ x % 17 = 0 ↔ ∃ k, x = 17 * k ∧ 59 ≤ k ∧ k ≤ 588) ∧ n = 530 := 
sorry

end NUMINAMATH_GPT_four_digit_numbers_divisible_by_17_l2138_213859


namespace NUMINAMATH_GPT_marathon_speed_ratio_l2138_213845

theorem marathon_speed_ratio (M D : ℝ) (J : ℝ) (H1 : D = 9) (H2 : J = 4/3 * M) (H3 : M + J + D = 23) :
  D / M = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_marathon_speed_ratio_l2138_213845


namespace NUMINAMATH_GPT_simplify_expression_l2138_213847

variable (p q r : ℝ)
variable (hp : p ≠ 2)
variable (hq : q ≠ 3)
variable (hr : r ≠ 4)

theorem simplify_expression : 
  (p^2 - 4) / (4 - r^2) * (q^2 - 9) / (2 - p^2) * (r^2 - 16) / (3 - q^2) = -1 :=
by
  -- Skipping the proof using sorry
  sorry

end NUMINAMATH_GPT_simplify_expression_l2138_213847


namespace NUMINAMATH_GPT_resulting_figure_perimeter_l2138_213804

def original_square_side : ℕ := 100

def original_square_area : ℕ := original_square_side * original_square_side

def rect1_side1 : ℕ := original_square_side
def rect1_side2 : ℕ := original_square_side / 2

def rect2_side1 : ℕ := original_square_side
def rect2_side2 : ℕ := original_square_side / 2

def new_figure_perimeter : ℕ :=
  3 * original_square_side + 4 * (original_square_side / 2)

theorem resulting_figure_perimeter :
  new_figure_perimeter = 500 :=
by {
    sorry
}

end NUMINAMATH_GPT_resulting_figure_perimeter_l2138_213804


namespace NUMINAMATH_GPT_minimum_value_of_fraction_l2138_213819

theorem minimum_value_of_fraction (x : ℝ) (hx : x > 10) : ∃ m, m = 30 ∧ ∀ y > 10, (y * y) / (y - 10) ≥ m :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_of_fraction_l2138_213819


namespace NUMINAMATH_GPT_evaluate_sum_of_squares_l2138_213810

theorem evaluate_sum_of_squares 
  (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 20) 
  (h2 : 4 * x + y = 25) : (x + y)^2 = 49 :=
  sorry

end NUMINAMATH_GPT_evaluate_sum_of_squares_l2138_213810


namespace NUMINAMATH_GPT_y_value_is_32_l2138_213895

-- Define the conditions
variables (y : ℝ) (hy_pos : y > 0) (hy_eq : y^2 = 1024)

-- State the theorem
theorem y_value_is_32 : y = 32 :=
by
  -- The proof will be written here
  sorry

end NUMINAMATH_GPT_y_value_is_32_l2138_213895


namespace NUMINAMATH_GPT_snacks_displayed_at_dawn_l2138_213887

variable (S : ℝ)
variable (SoldMorning : ℝ)
variable (SoldAfternoon : ℝ)

axiom cond1 : SoldMorning = (3 / 5) * S
axiom cond2 : SoldAfternoon = 180
axiom cond3 : SoldMorning = SoldAfternoon

theorem snacks_displayed_at_dawn : S = 300 :=
by
  sorry

end NUMINAMATH_GPT_snacks_displayed_at_dawn_l2138_213887


namespace NUMINAMATH_GPT_fixed_point_coordinates_l2138_213884

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  2 * a^(x + 1) - 3

theorem fixed_point_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_coordinates_l2138_213884


namespace NUMINAMATH_GPT_number_of_pieces_of_paper_used_l2138_213890

theorem number_of_pieces_of_paper_used
  (P : ℕ)
  (h1 : 1 / 5 > 0)
  (h2 : 2 / 5 > 0)
  (h3 : 1 < (P : ℝ) * (1 / 5) + 2 / 5 ∧ (P : ℝ) * (1 / 5) + 2 / 5 ≤ 2) : 
  P = 8 :=
sorry

end NUMINAMATH_GPT_number_of_pieces_of_paper_used_l2138_213890


namespace NUMINAMATH_GPT_distance_from_integer_l2138_213868

theorem distance_from_integer (a : ℝ) (h : a > 0) (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ abs (m * a - k) ≤ (1 / n) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_integer_l2138_213868


namespace NUMINAMATH_GPT_knights_statements_l2138_213846

theorem knights_statements (r ℓ : Nat) (hr : r ≥ 2) (hℓ : ℓ ≥ 2)
  (h : 2 * r * ℓ = 230) :
  (r + ℓ) * (r + ℓ - 1) - 230 = 526 :=
by
  sorry

end NUMINAMATH_GPT_knights_statements_l2138_213846


namespace NUMINAMATH_GPT_solve_for_x_l2138_213886

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 8) * x = 14 ↔ x = 392 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l2138_213886


namespace NUMINAMATH_GPT_Suzanna_rides_8_miles_in_40_minutes_l2138_213867

theorem Suzanna_rides_8_miles_in_40_minutes :
  (∀ n : ℕ, Suzanna_distance_in_n_minutes = (n / 10) * 2) → Suzanna_distance_in_40_minutes = 8 :=
by
  sorry

-- Definitions for Suzanna's distance conditions
def Suzanna_distance_in_n_minutes (n : ℕ) : ℕ := (n / 10) * 2

noncomputable def Suzanna_distance_in_40_minutes := Suzanna_distance_in_n_minutes 40

#check Suzanna_rides_8_miles_in_40_minutes

end NUMINAMATH_GPT_Suzanna_rides_8_miles_in_40_minutes_l2138_213867


namespace NUMINAMATH_GPT_jellybeans_in_jar_l2138_213805

theorem jellybeans_in_jar (num_kids_normal : ℕ) (num_absent : ℕ) (num_jellybeans_each : ℕ) (num_leftover : ℕ) 
  (h1 : num_kids_normal = 24) (h2 : num_absent = 2) (h3 : num_jellybeans_each = 3) (h4 : num_leftover = 34) : 
  (num_kids_normal - num_absent) * num_jellybeans_each + num_leftover = 100 :=
by sorry

end NUMINAMATH_GPT_jellybeans_in_jar_l2138_213805


namespace NUMINAMATH_GPT_problem_statement_l2138_213844

theorem problem_statement (a b : ℝ) (h : (1 / a + 1 / b) / (1 / a - 1 / b) = 2023) : (a + b) / (a - b) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2138_213844


namespace NUMINAMATH_GPT_book_cost_proof_l2138_213831

variable (C1 C2 : ℝ)

theorem book_cost_proof (h1 : C1 + C2 = 460)
                        (h2 : C1 * 0.85 = C2 * 1.19) :
    C1 = 268.53 := by
  sorry

end NUMINAMATH_GPT_book_cost_proof_l2138_213831


namespace NUMINAMATH_GPT_sin_identity_l2138_213893

theorem sin_identity (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) :
  Real.sin ((3 * π / 4) - α) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_identity_l2138_213893


namespace NUMINAMATH_GPT_fraction_evaluation_l2138_213888

theorem fraction_evaluation :
  ( (1 / 2 * 1 / 3 * 1 / 4 * 1 / 5 + 3 / 2 * 3 / 4 * 3 / 5) / 
    (1 / 2 * 2 / 3 * 2 / 5) ) = 41 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l2138_213888


namespace NUMINAMATH_GPT_find_function_l2138_213815

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1) :
  ∀ x : ℝ, f x = if x ≠ 0.5 then 1 / (0.5 - x) else 0.5 :=
by
  sorry

end NUMINAMATH_GPT_find_function_l2138_213815


namespace NUMINAMATH_GPT_find_height_of_cuboid_l2138_213897

variable (A : ℝ) (V : ℝ) (h : ℝ)

theorem find_height_of_cuboid (h_eq : h = V / A) (A_eq : A = 36) (V_eq : V = 252) : h = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_height_of_cuboid_l2138_213897


namespace NUMINAMATH_GPT_f_2015_l2138_213872

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 3) = f x
axiom f_interval : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = 2^x

theorem f_2015 : f 2015 = -2 := sorry

end NUMINAMATH_GPT_f_2015_l2138_213872


namespace NUMINAMATH_GPT_ratio_d_c_l2138_213863

theorem ratio_d_c (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hc : c ≠ 0) 
  (h1 : 8 * x - 5 * y = c) (h2 : 10 * y - 16 * x = d) : d / c = -2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_d_c_l2138_213863


namespace NUMINAMATH_GPT_pulley_distance_l2138_213858

theorem pulley_distance (r₁ r₂ d l : ℝ):
    r₁ = 10 →
    r₂ = 6 →
    l = 30 →
    (d = 2 * Real.sqrt 229) :=
by
    intros h₁ h₂ h₃
    sorry

end NUMINAMATH_GPT_pulley_distance_l2138_213858


namespace NUMINAMATH_GPT_range_of_m_l2138_213817

variables (m : ℝ)

def p : Prop := ∀ x : ℝ, 0 < x → (1/2 : ℝ)^x + m - 1 < 0
def q : Prop := ∃ x : ℝ, 0 < x ∧ m * x^2 + 4 * x - 1 = 0

theorem range_of_m (h : p m ∧ q m) : -4 ≤ m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2138_213817


namespace NUMINAMATH_GPT_necessary_for_A_l2138_213870

-- Define the sets A, B, C as non-empty sets
variables {α : Type*} (A B C : Set α)
-- Non-empty sets
axiom non_empty_A : ∃ x, x ∈ A
axiom non_empty_B : ∃ x, x ∈ B
axiom non_empty_C : ∃ x, x ∈ C

-- Conditions
axiom union_condition : A ∪ B = C
axiom subset_condition : ¬ (B ⊆ A)

-- Statement to prove
theorem necessary_for_A (x : α) : (x ∈ C → x ∈ A) ∧ ¬(x ∈ C ↔ x ∈ A) :=
sorry

end NUMINAMATH_GPT_necessary_for_A_l2138_213870


namespace NUMINAMATH_GPT_mul_97_97_eq_9409_l2138_213834

theorem mul_97_97_eq_9409 : 97 * 97 = 9409 := 
  sorry

end NUMINAMATH_GPT_mul_97_97_eq_9409_l2138_213834


namespace NUMINAMATH_GPT_josh_bought_6_CDs_l2138_213880

theorem josh_bought_6_CDs 
  (numFilms : ℕ)   (numBooks : ℕ) (numCDs : ℕ)
  (costFilm : ℕ)   (costBook : ℕ) (costCD : ℕ)
  (totalSpent : ℕ) :
  numFilms = 9 → 
  numBooks = 4 → 
  costFilm = 5 → 
  costBook = 4 → 
  costCD = 3 → 
  totalSpent = 79 → 
  numCDs = (totalSpent - numFilms * costFilm - numBooks * costBook) / costCD → 
  numCDs = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  exact h7

end NUMINAMATH_GPT_josh_bought_6_CDs_l2138_213880


namespace NUMINAMATH_GPT_incorrect_expression_l2138_213889

theorem incorrect_expression : ¬ (5 = (Real.sqrt (-5))^2) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_expression_l2138_213889


namespace NUMINAMATH_GPT_professor_has_to_grade_405_more_problems_l2138_213881

theorem professor_has_to_grade_405_more_problems
  (problems_per_paper : ℕ)
  (total_papers : ℕ)
  (graded_papers : ℕ)
  (remaining_papers := total_papers - graded_papers)
  (p : ℕ := remaining_papers * problems_per_paper) :
  problems_per_paper = 15 ∧ total_papers = 45 ∧ graded_papers = 18 → p = 405 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_professor_has_to_grade_405_more_problems_l2138_213881


namespace NUMINAMATH_GPT_abs_inequality_solution_rational_inequality_solution_l2138_213891

theorem abs_inequality_solution (x : ℝ) : (|x - 2| + |2 * x - 3| < 4) ↔ (1 / 3 < x ∧ x < 3) :=
sorry

theorem rational_inequality_solution (x : ℝ) : 
  (x^2 - 3 * x) / (x^2 - x - 2) ≤ x ↔ (x ∈ Set.Icc (-1) 0 ∪ {1} ∪ Set.Ioi 2) := 
sorry

#check abs_inequality_solution
#check rational_inequality_solution

end NUMINAMATH_GPT_abs_inequality_solution_rational_inequality_solution_l2138_213891
