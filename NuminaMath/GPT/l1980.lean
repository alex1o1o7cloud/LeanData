import Mathlib

namespace intervals_union_l1980_198055

open Set

noncomputable def I (a b : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < b}

theorem intervals_union {I1 I2 I3 : Set ℝ} (h1 : ∃ (a1 b1 : ℝ), I1 = I a1 b1)
  (h2 : ∃ (a2 b2 : ℝ), I2 = I a2 b2) (h3 : ∃ (a3 b3 : ℝ), I3 = I a3 b3)
  (h_non_empty : (I1 ∩ I2 ∩ I3).Nonempty) (h_not_contained : ¬ (I1 ⊆ I2) ∧ ¬ (I1 ⊆ I3) ∧ ¬ (I2 ⊆ I1) ∧ ¬ (I2 ⊆ I3) ∧ ¬ (I3 ⊆ I1) ∧ ¬ (I3 ⊆ I2)) :
  I1 ⊆ (I2 ∪ I3) ∨ I2 ⊆ (I1 ∪ I3) ∨ I3 ⊆ (I1 ∪ I2) :=
sorry

end intervals_union_l1980_198055


namespace smallest_N_l1980_198018

theorem smallest_N (N : ℕ) (h : 7 * N = 999999) : N = 142857 :=
sorry

end smallest_N_l1980_198018


namespace ratio_future_age_l1980_198083

variable (S M : ℕ)

theorem ratio_future_age (h1 : (S : ℝ) / M = 7 / 2) (h2 : S - 6 = 78) : 
  ((S + 16) : ℝ) / (M + 16) = 5 / 2 := 
by
  sorry

end ratio_future_age_l1980_198083


namespace greatest_value_of_squares_exists_max_value_of_squares_l1980_198030

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 702 :=
sorry

theorem exists_max_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 702 :=
sorry

end greatest_value_of_squares_exists_max_value_of_squares_l1980_198030


namespace amount_decreased_is_5_l1980_198053

noncomputable def x : ℕ := 50
noncomputable def equation (x y : ℕ) : Prop := (1 / 5) * x - y = 5

theorem amount_decreased_is_5 : ∃ y : ℕ, equation x y ∧ y = 5 :=
by
  sorry

end amount_decreased_is_5_l1980_198053


namespace original_planned_production_l1980_198005

theorem original_planned_production (x : ℝ) (hx1 : x ≠ 0) (hx2 : 210 / x - 210 / (1.5 * x) = 5) : x = 14 :=
by sorry

end original_planned_production_l1980_198005


namespace arith_seq_common_diff_l1980_198084

/-
Given:
- an arithmetic sequence {a_n} with common difference d,
- the sum of the first n terms S_n = n * a_1 + n * (n - 1) / 2 * d,
- b_n = S_n / n,

Prove that the common difference of the sequence {a_n - b_n} is d/2.
-/

theorem arith_seq_common_diff (a b : ℕ → ℚ) (a1 d : ℚ) 
  (h1 : ∀ n, a n = a1 + n * d) 
  (h2 : ∀ n, b n = (a1 + n - 1 * d + n * (n - 1) / 2 * d) / n) : 
  ∀ n, (a n - b n) - (a (n + 1) - b (n + 1)) = d / 2 := 
    sorry

end arith_seq_common_diff_l1980_198084


namespace trigonometric_identity_l1980_198004

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (6 * Real.sin (2 * x) + 2 * Real.cos (2 * x)) / (Real.cos (2 * x) - 3 * Real.sin (2 * x)) = -2 / 5 := by
  sorry

end trigonometric_identity_l1980_198004


namespace pictures_at_museum_l1980_198059

variable (M : ℕ)

-- Definitions from conditions
def pictures_at_zoo : ℕ := 50
def pictures_deleted : ℕ := 38
def pictures_left : ℕ := 20

-- Theorem to prove the total number of pictures taken including the museum pictures
theorem pictures_at_museum :
  pictures_at_zoo + M - pictures_deleted = pictures_left → M = 8 :=
by
  sorry

end pictures_at_museum_l1980_198059


namespace car_rental_total_cost_l1980_198016

theorem car_rental_total_cost 
  (rental_cost : ℕ)
  (gallons : ℕ)
  (cost_per_gallon : ℕ)
  (cost_per_mile : ℚ)
  (miles_driven : ℕ)
  (H1 : rental_cost = 150)
  (H2 : gallons = 8)
  (H3 : cost_per_gallon = 350 / 100)
  (H4 : cost_per_mile = 50 / 100)
  (H5 : miles_driven = 320) :
  rental_cost + gallons * cost_per_gallon + miles_driven * cost_per_mile = 338 :=
  sorry

end car_rental_total_cost_l1980_198016


namespace express_function_as_chain_of_equalities_l1980_198085

theorem express_function_as_chain_of_equalities (x : ℝ) : 
  ∃ (u : ℝ), (u = 2 * x - 5) ∧ ((2 * x - 5) ^ 10 = u ^ 10) :=
by 
  sorry

end express_function_as_chain_of_equalities_l1980_198085


namespace circle_represents_real_l1980_198087

theorem circle_represents_real
  (a : ℝ)
  (h : ∀ x y : ℝ, x^2 + y^2 + 2*y + 2*a - 1 = 0 → ∃ r : ℝ, r > 0) : 
  a < 1 := 
sorry

end circle_represents_real_l1980_198087


namespace percentage_both_questions_correct_l1980_198036

-- Definitions for the conditions in the problem
def percentage_first_question_correct := 85
def percentage_second_question_correct := 65
def percentage_neither_question_correct := 5
def percentage_one_or_more_questions_correct := 100 - percentage_neither_question_correct

-- Theorem stating that 55 percent answered both questions correctly
theorem percentage_both_questions_correct :
  percentage_first_question_correct + percentage_second_question_correct - percentage_one_or_more_questions_correct = 55 :=
by
  sorry

end percentage_both_questions_correct_l1980_198036


namespace find_ordered_pair_l1980_198068

theorem find_ordered_pair {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = -2 * a ∨ x = b)
  (h2 : b = -2 * -2 * a) : (a, b) = (-1/2, -1/2) :=
by
  sorry

end find_ordered_pair_l1980_198068


namespace terminating_fraction_count_l1980_198037

theorem terminating_fraction_count :
  (∃ n_values : Finset ℕ, (∀ n ∈ n_values, 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = k * 49)) ∧ n_values.card = 10) :=
by
  -- Placeholder for the proof, does not contribute to the conditions-direct definitions.
  sorry

end terminating_fraction_count_l1980_198037


namespace no_nonzero_integers_satisfy_conditions_l1980_198096

theorem no_nonzero_integers_satisfy_conditions :
  ¬ ∃ a b x y : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0) ∧ (a * x - b * y = 16) ∧ (a * y + b * x = 1) :=
by
  sorry

end no_nonzero_integers_satisfy_conditions_l1980_198096


namespace intersection_is_correct_l1980_198034

noncomputable def M : Set ℝ := { x | 1 + x ≥ 0 }
noncomputable def N : Set ℝ := { x | 4 / (1 - x) > 0 }
noncomputable def intersection : Set ℝ := { x | -1 ≤ x ∧ x < 1 }

theorem intersection_is_correct : M ∩ N = intersection := by
  sorry

end intersection_is_correct_l1980_198034


namespace exists_a_b_l1980_198050

theorem exists_a_b (S : Finset ℕ) (hS : S.card = 43) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (a^2 - b^2) % 100 = 0 := 
by
  sorry

end exists_a_b_l1980_198050


namespace BC_work_time_l1980_198082

-- Definitions
def rateA : ℚ := 1 / 4 -- A's rate of work
def rateB : ℚ := 1 / 4 -- B's rate of work
def rateAC : ℚ := 1 / 3 -- A and C's combined rate of work

-- To prove
theorem BC_work_time : 1 / (rateB + (rateAC - rateA)) = 3 := by
  sorry

end BC_work_time_l1980_198082


namespace correct_mean_l1980_198033

theorem correct_mean (mean n incorrect_value correct_value : ℝ) 
  (hmean : mean = 150) (hn : n = 20) (hincorrect : incorrect_value = 135) (hcorrect : correct_value = 160):
  (mean * n - incorrect_value + correct_value) / n = 151.25 :=
by
  sorry

end correct_mean_l1980_198033


namespace polynomial_remainder_correct_l1980_198076

noncomputable def remainder_polynomial (x : ℝ) : ℝ := x ^ 100

def divisor_polynomial (x : ℝ) : ℝ := x ^ 2 - 3 * x + 2

def polynomial_remainder (x : ℝ) : ℝ := 2 ^ 100 * (x - 1) - (x - 2)

theorem polynomial_remainder_correct : ∀ x : ℝ, (remainder_polynomial x) % (divisor_polynomial x) = polynomial_remainder x := by
  sorry

end polynomial_remainder_correct_l1980_198076


namespace cos_A_and_sin_2B_minus_A_l1980_198074

variable (A B C a b c : ℝ)
variable (h1 : a * Real.sin A = 4 * b * Real.sin B)
variable (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2))

theorem cos_A_and_sin_2B_minus_A :
  Real.cos A = -Real.sqrt 5 / 5 ∧ Real.sin (2 * B - A) = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_A_and_sin_2B_minus_A_l1980_198074


namespace marias_workday_ends_at_3_30_pm_l1980_198009
open Nat

theorem marias_workday_ends_at_3_30_pm :
  let start_time := (7 : Nat)
  let lunch_start_time := (11 + (30 / 60))
  let work_duration := (8 : Nat)
  let lunch_break := (30 / 60 : Nat)
  let end_time := (15 + (30 / 60) : Nat)
  (start_time + work_duration + lunch_break) - (lunch_start_time - start_time) = end_time := by
  sorry

end marias_workday_ends_at_3_30_pm_l1980_198009


namespace sufficient_material_for_box_l1980_198015

theorem sufficient_material_for_box :
  ∃ (l w h : ℕ), l * w * h ≥ 1995 ∧ 2 * (l * w + w * h + h * l) ≤ 958 :=
  sorry

end sufficient_material_for_box_l1980_198015


namespace fraction_of_area_above_line_l1980_198002

open Real

-- Define the points and the line between them
noncomputable def pointA : (ℝ × ℝ) := (2, 3)
noncomputable def pointB : (ℝ × ℝ) := (5, 1)

-- Define the vertices of the square
noncomputable def square_vertices : List (ℝ × ℝ) := [(2, 1), (5, 1), (5, 4), (2, 4)]

-- Define the equation of the line
noncomputable def line_eq (x : ℝ) : ℝ :=
  (-2/3) * x + 13/3

-- Define the vertical and horizontal boundaries
noncomputable def x_min : ℝ := 2
noncomputable def x_max : ℝ := 5
noncomputable def y_min : ℝ := 1
noncomputable def y_max : ℝ := 4

-- Calculate the area of the triangle formed below the line
noncomputable def triangle_area : ℝ := 0.5 * 2 * 3

-- Calculate the area of the square
noncomputable def square_area : ℝ := 3 * 3

-- The fraction of the area above the line
noncomputable def area_fraction_above : ℝ := (square_area - triangle_area) / square_area

-- Prove the fraction of the area of the square above the line is 2/3
theorem fraction_of_area_above_line : area_fraction_above = 2 / 3 :=
  sorry

end fraction_of_area_above_line_l1980_198002


namespace factor_adjustment_l1980_198060

theorem factor_adjustment (a b : ℝ) (h : a * b = 65.08) : a / 100 * (100 * b) = 65.08 :=
by
  sorry

end factor_adjustment_l1980_198060


namespace eval_expression_l1980_198094

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end eval_expression_l1980_198094


namespace quadratic_function_properties_l1980_198007

noncomputable def f (x : ℝ) : ℝ := -5 / 2 * x^2 + 15 * x - 25 / 2

theorem quadratic_function_properties :
  (∃ a : ℝ, ∀ x : ℝ, (f x = a * (x - 1) * (x - 5)) ∧ (f 3 = 10)) → 
  (f x = -5 / 2 * x^2 + 15 * x - 25 / 2) :=
by 
  sorry

end quadratic_function_properties_l1980_198007


namespace num_dislikers_tv_books_games_is_correct_l1980_198042

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

end num_dislikers_tv_books_games_is_correct_l1980_198042


namespace carrie_payment_l1980_198001

def num_shirts := 8
def cost_per_shirt := 12
def total_shirt_cost := num_shirts * cost_per_shirt

def num_pants := 4
def cost_per_pant := 25
def total_pant_cost := num_pants * cost_per_pant

def num_jackets := 4
def cost_per_jacket := 75
def total_jacket_cost := num_jackets * cost_per_jacket

def num_skirts := 3
def cost_per_skirt := 30
def total_skirt_cost := num_skirts * cost_per_skirt

def num_shoes := 2
def cost_per_shoe := 50
def total_shoe_cost := num_shoes * cost_per_shoe

def total_cost := total_shirt_cost + total_pant_cost + total_jacket_cost + total_skirt_cost + total_shoe_cost

def mom_share := (2 / 3 : ℚ) * total_cost
def carrie_share := total_cost - mom_share

theorem carrie_payment : carrie_share = 228.67 :=
by
  sorry

end carrie_payment_l1980_198001


namespace convex_polyhedron_formula_l1980_198029

theorem convex_polyhedron_formula
  (V E F t h T H : ℕ)
  (hF : F = 40)
  (hFaces : F = t + h)
  (hVertex : 2 * T + H = 7)
  (hEdges : E = (3 * t + 6 * h) / 2)
  (hEuler : V - E + F = 2)
  : 100 * H + 10 * T + V = 367 := 
sorry

end convex_polyhedron_formula_l1980_198029


namespace max_marks_l1980_198090

theorem max_marks (marks_secured : ℝ) (percentage : ℝ) (max_marks : ℝ) 
  (h1 : marks_secured = 332) 
  (h2 : percentage = 83) 
  (h3 : percentage = (marks_secured / max_marks) * 100) 
  : max_marks = 400 :=
by
  sorry

end max_marks_l1980_198090


namespace percentage_in_biology_is_correct_l1980_198058

/-- 
There are 840 students at a college.
546 students are not enrolled in a biology class.
We need to show what percentage of students are enrolled in biology classes.
--/

def num_students := 840
def not_in_biology := 546

def percentage_in_biology : ℕ := 
  ((num_students - not_in_biology) * 100) / num_students

theorem percentage_in_biology_is_correct : percentage_in_biology = 35 := 
  by
    -- proof is skipped
    sorry

end percentage_in_biology_is_correct_l1980_198058


namespace top_width_of_channel_l1980_198064

theorem top_width_of_channel (b : ℝ) (A : ℝ) (h : ℝ) (w : ℝ) : 
  b = 8 ∧ A = 700 ∧ h = 70 ∧ (A = (1/2) * (w + b) * h) → w = 12 := 
by 
  intro h1
  sorry

end top_width_of_channel_l1980_198064


namespace minimum_stamps_l1980_198049

theorem minimum_stamps (c f : ℕ) (h : 3 * c + 4 * f = 50) : c + f = 13 :=
sorry

end minimum_stamps_l1980_198049


namespace question1_question2_l1980_198044

variable (α : ℝ)

theorem question1 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos α ^ 2 + Real.cos (2 * α)) = -15 / 23 := by
  sorry

theorem question2 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    Real.tan (α - 5 * π / 4) = -7 := by
  sorry

end question1_question2_l1980_198044


namespace remainder_problem_l1980_198054

theorem remainder_problem : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end remainder_problem_l1980_198054


namespace find_a5_find_a31_div_a29_l1980_198000

noncomputable def geo_diff_seq (a : ℕ → ℕ) (d : ℕ) :=
∀ n : ℕ, n > 0 → (a (n + 2) / a (n + 1)) - (a (n + 1) / a n) = d

theorem find_a5 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 5 = 105 :=
sorry

theorem find_a31_div_a29 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 31 / a 29 = 3363 :=
sorry

end find_a5_find_a31_div_a29_l1980_198000


namespace items_sold_each_house_l1980_198013

-- Define the conditions
def visits_day_one : ℕ := 20
def visits_day_two : ℕ := 2 * visits_day_one
def sale_percentage_day_two : ℝ := 0.8
def total_sales : ℕ := 104

-- Define the number of items sold at each house
variable (x : ℕ)

-- Define the main Lean 4 statement for the proof
theorem items_sold_each_house (h1 : 20 * x + 32 * x = 104) : x = 2 :=
by
  -- Proof would go here
  sorry

end items_sold_each_house_l1980_198013


namespace count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l1980_198024

open Nat

def num180Unchanged : Nat := 
  let valid_pairs := [(0, 0), (1, 1), (8, 8), (6, 9), (9, 6)];
  let middle_digits := [0, 1, 8];
  (valid_pairs.length) * ((valid_pairs.length + 1) * (valid_pairs.length + 1) * middle_digits.length)

def num180UnchangedDivBy4 : Nat :=
  let valid_div4_pairs := [(0, 0), (1, 6), (6, 0), (6, 8), (8, 0), (8, 8), (9, 6)];
  let middle_digits := [0, 1, 8];
  valid_div4_pairs.length * (valid_div4_pairs.length / 5) * middle_digits.length

def sum180UnchangedNumbers : Nat :=
   1959460200 -- The sum by the given problem

theorem count_7_digit_nums_180_reversible : num180Unchanged = 300 :=
sorry

theorem count_7_digit_nums_180_reversible_divis_by_4 : num180UnchangedDivBy4 = 75 :=
sorry

theorem sum_of_7_digit_nums_180_reversible : sum180UnchangedNumbers = 1959460200 :=
sorry

end count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l1980_198024


namespace market_trips_l1980_198021

theorem market_trips (d_school_round: ℝ) (d_market_round: ℝ) (num_school_trips_per_day: ℕ) (num_school_days_per_week: ℕ) (total_week_mileage: ℝ) :
  d_school_round = 5 →
  d_market_round = 4 →
  num_school_trips_per_day = 2 →
  num_school_days_per_week = 4 →
  total_week_mileage = 44 →
  (total_week_mileage - (d_school_round * num_school_trips_per_day * num_school_days_per_week)) / d_market_round = 1 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end market_trips_l1980_198021


namespace smallest_positive_debt_resolved_l1980_198097

theorem smallest_positive_debt_resolved : ∃ (D : ℕ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 240 * g) ∧ D = 80 := by
  sorry

end smallest_positive_debt_resolved_l1980_198097


namespace cos_inequality_for_triangle_l1980_198099

theorem cos_inequality_for_triangle (A B C : ℝ) (h : A + B + C = π) :
  (1 / 3) * (Real.cos A + Real.cos B + Real.cos C) ≤ (1 / 2) ∧
  (1 / 2) ≤ Real.sqrt ((1 / 3) * (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2)) :=
by
  sorry

end cos_inequality_for_triangle_l1980_198099


namespace correct_letter_is_P_l1980_198077

variable (x : ℤ)

-- Conditions
def date_behind_C := x
def date_behind_A := x + 2
def date_behind_B := x + 11
def date_behind_P := x + 13 -- Based on problem setup
def date_behind_Q := x + 14 -- Continuous sequence assumption
def date_behind_R := x + 15 -- Continuous sequence assumption
def date_behind_S := x + 16 -- Continuous sequence assumption

-- Proof statement
theorem correct_letter_is_P :
  ∃ y, (y = date_behind_P ∧ x + y = date_behind_A + date_behind_B) := by
  sorry

end correct_letter_is_P_l1980_198077


namespace sufficient_but_not_necessary_condition_l1980_198069

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h : x > 1) : x > 0 :=
by
  sorry

end sufficient_but_not_necessary_condition_l1980_198069


namespace tom_read_books_l1980_198023

theorem tom_read_books :
  let books_may := 2
  let books_june := 6
  let books_july := 10
  books_may + books_june + books_july = 18 := by
  sorry

end tom_read_books_l1980_198023


namespace water_remaining_l1980_198035

theorem water_remaining (initial_water : ℕ) (evap_rate : ℕ) (days : ℕ) : 
  initial_water = 500 → evap_rate = 1 → days = 50 → 
  initial_water - evap_rate * days = 450 :=
by
  intros h₁ h₂ h₃
  sorry

end water_remaining_l1980_198035


namespace sovereign_states_upper_bound_l1980_198040

theorem sovereign_states_upper_bound (n : ℕ) (k : ℕ) : 
  (∃ (lines : ℕ) (border_stop_moving : Prop) (countries_disappear : Prop)
     (create_un : Prop) (total_countries : ℕ),
        (lines = n)
        ∧ (border_stop_moving = true)
        ∧ (countries_disappear = true)
        ∧ (create_un = true)
        ∧ (total_countries = k)) 
  → k ≤ (n^3 + 5*n) / 6 + 1 := 
sorry

end sovereign_states_upper_bound_l1980_198040


namespace circle_equation_of_tangent_circle_l1980_198039

theorem circle_equation_of_tangent_circle
  (h : ∀ x y: ℝ, x^2/4 - y^2 = 1 → (x = 2 ∨ x = -2) → y = 0)
  (asymptote : ∀ x y : ℝ, (y = (1/2)*x ∨ y = -(1/2)*x) → (x - 2)^2 + y^2 = (4/5))
  : ∃ k : ℝ, (∀ x y : ℝ, (x - 2)^2 + y^2 = k) → k = 4/5 := by
  sorry

end circle_equation_of_tangent_circle_l1980_198039


namespace sqrt_23_parts_xy_diff_l1980_198088

-- Problem 1: Integer and decimal parts of sqrt(23)
theorem sqrt_23_parts : ∃ (integer_part : ℕ) (decimal_part : ℝ), 
  integer_part = 4 ∧ decimal_part = Real.sqrt 23 - 4 ∧
  (integer_part : ℝ) + decimal_part = Real.sqrt 23 :=
by
  sorry

-- Problem 2: x - y for 9 + sqrt(3) = x + y with given conditions
theorem xy_diff : 
  ∀ (x y : ℝ), x = 10 → y = Real.sqrt 3 - 1 → x - y = 11 - Real.sqrt 3 :=
by
  sorry

end sqrt_23_parts_xy_diff_l1980_198088


namespace union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l1980_198022

open Set

noncomputable def U : Set ℤ := {x | -2 < x ∧ x < 2}
def A : Set ℤ := {x | x^2 - 5 * x - 6 = 0}
def B : Set ℤ := {x | x^2 = 1}

theorem union_of_A_and_B : A ∪ B = {-1, 1, 6} :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
by
  sorry

theorem complement_of_intersection_in_U : U \ (A ∩ B) = {0, 1} :=
by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l1980_198022


namespace find_c_exactly_two_common_points_l1980_198073

theorem find_c_exactly_two_common_points (c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^3 - 3*x1 + c = 0) ∧ (x2^3 - 3*x2 + c = 0)) ↔ (c = -2 ∨ c = 2) := 
sorry

end find_c_exactly_two_common_points_l1980_198073


namespace quarter_more_than_whole_l1980_198028

theorem quarter_more_than_whole (x : ℝ) (h : x / 4 = 9 + x) : x = -12 :=
by
  sorry

end quarter_more_than_whole_l1980_198028


namespace y_intercept_range_l1980_198075

-- Define the points A and B
def pointA : ℝ × ℝ := (-1, -2)
def pointB : ℝ × ℝ := (2, 3)

-- We define the predicate for the line intersection condition
def line_intersects_segment (c : ℝ) : Prop :=
  let x_val_a := -1
  let y_val_a := -2
  let x_val_b := 2
  let y_val_b := 3
  -- Line equation at point A
  let eqn_a := x_val_a + y_val_a - c
  -- Line equation at point B
  let eqn_b := x_val_b + y_val_b - c
  -- We assert that the line must intersect the segment AB
  eqn_a ≤ 0 ∧ eqn_b ≥ 0 ∨ eqn_a ≥ 0 ∧ eqn_b ≤ 0

-- The main theorem to prove the range of c
theorem y_intercept_range : 
  ∃ c_min c_max : ℝ, c_min = -3 ∧ c_max = 5 ∧
  ∀ c, line_intersects_segment c ↔ c_min ≤ c ∧ c ≤ c_max :=
by
  existsi -3
  existsi 5
  sorry

end y_intercept_range_l1980_198075


namespace janice_initial_sentences_l1980_198071

theorem janice_initial_sentences:
  ∀ (r t1 t2 t3 t4: ℕ), 
  r = 6 → 
  t1 = 20 → 
  t2 = 15 → 
  t3 = 40 → 
  t4 = 18 → 
  (t1 * r + t2 * r + t4 * r - t3 = 536 - 258) → 
  536 - (t1 * r + t2 * r + t4 * r - t3) = 258 := by
  intros
  sorry

end janice_initial_sentences_l1980_198071


namespace point_in_fourth_quadrant_l1980_198052

open Complex

theorem point_in_fourth_quadrant (z : ℂ) (h : (3 + 4 * I) * z = 25) : 
  Complex.arg z > -π / 2 ∧ Complex.arg z < 0 := 
by
  sorry

end point_in_fourth_quadrant_l1980_198052


namespace seven_points_unit_distance_l1980_198078

theorem seven_points_unit_distance :
  ∃ (A B C D E F G : ℝ × ℝ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
     E ≠ F ∧ E ≠ G ∧
     F ≠ G) ∧
    (∀ (P Q R : ℝ × ℝ),
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E ∨ P = F ∨ P = G) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E ∨ Q = F ∨ Q = G) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E ∨ R = F ∨ R = G) →
      P ≠ Q → P ≠ R → Q ≠ R →
      (dist P Q = 1 ∨ dist P R = 1 ∨ dist Q R = 1)) :=
sorry

end seven_points_unit_distance_l1980_198078


namespace unique_geometric_sequence_l1980_198063

theorem unique_geometric_sequence (a : ℝ) (q : ℝ) (a_n b_n : ℕ → ℝ) 
    (h1 : a > 0) 
    (h2 : a_n 1 = a) 
    (h3 : b_n 1 - a_n 1 = 1) 
    (h4 : b_n 2 - a_n 2 = 2) 
    (h5 : b_n 3 - a_n 3 = 3) 
    (h6 : ∀ n, a_n (n + 1) = a_n n * q) 
    (h7 : ∀ n, b_n (n + 1) = b_n n * q) : 
    a = 1 / 3 := sorry

end unique_geometric_sequence_l1980_198063


namespace parallel_vectors_x_value_l1980_198027

theorem parallel_vectors_x_value (x : ℝ) :
  (∀ k : ℝ, k ≠ 0 → (4, 2) = (k * x, k * (-3))) → x = -6 :=
by
  sorry

end parallel_vectors_x_value_l1980_198027


namespace price_of_each_brownie_l1980_198070

variable (B : ℝ)

theorem price_of_each_brownie (h : 4 * B + 10 + 28 = 50) : B = 3 := by
  -- proof steps would go here
  sorry

end price_of_each_brownie_l1980_198070


namespace prove_fraction_l1980_198031

variables {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

def forms_arithmetic_sequence (x y z : ℝ) : Prop :=
2 * y = x + z

theorem prove_fraction
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h_ar : forms_arithmetic_sequence (a 1 + 2 * b 1) (a 3 + 4 * b 3) (a 5 + 8 * b 5)) :
  (b 3 * b 7) / (b 4 ^ 2) = 1 / 4 :=
sorry

end prove_fraction_l1980_198031


namespace age_of_B_l1980_198020

theorem age_of_B (A B C : ℕ) (h1 : A + B + C = 90)
                  (h2 : (A - 10) = (B - 10) / 2)
                  (h3 : (B - 10) / 2 = (C - 10) / 3) : 
                  B = 30 :=
by sorry

end age_of_B_l1980_198020


namespace gcd_pow_sub_one_l1980_198045

theorem gcd_pow_sub_one (m n : ℕ) (h1 : m = 2^2024 - 1) (h2 : n = 2^2000 - 1) : Nat.gcd m n = 2^24 - 1 := 
by
  sorry

end gcd_pow_sub_one_l1980_198045


namespace product_of_two_integers_l1980_198010

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) (h3 : x > y) : x * y = 168 := by
  sorry

end product_of_two_integers_l1980_198010


namespace joan_total_money_l1980_198066

-- Define the number of each type of coin found
def dimes_jacket : ℕ := 15
def dimes_shorts : ℕ := 4
def nickels_shorts : ℕ := 7
def quarters_jeans : ℕ := 12
def pennies_jeans : ℕ := 2
def nickels_backpack : ℕ := 8
def pennies_backpack : ℕ := 23

-- Calculate the total number of each type of coin
def total_dimes : ℕ := dimes_jacket + dimes_shorts
def total_nickels : ℕ := nickels_shorts + nickels_backpack
def total_quarters : ℕ := quarters_jeans
def total_pennies : ℕ := pennies_jeans + pennies_backpack

-- Calculate the total value of each type of coin
def value_dimes : ℝ := total_dimes * 0.10
def value_nickels : ℝ := total_nickels * 0.05
def value_quarters : ℝ := total_quarters * 0.25
def value_pennies : ℝ := total_pennies * 0.01

-- Calculate the total amount of money found
def total_money : ℝ := value_dimes + value_nickels + value_quarters + value_pennies

-- Proof statement
theorem joan_total_money : total_money = 5.90 := by
  sorry

end joan_total_money_l1980_198066


namespace mr_brown_financial_outcome_l1980_198026

theorem mr_brown_financial_outcome :
  ∃ (C₁ C₂ : ℝ), (2.40 = 1.25 * C₁) ∧ (2.40 = 0.75 * C₂) ∧ ((2.40 + 2.40) - (C₁ + C₂) = -0.32) :=
by
  sorry

end mr_brown_financial_outcome_l1980_198026


namespace geometric_sequence_formula_l1980_198047

variable {q : ℝ} -- Common ratio
variable {m n : ℕ} -- Positive natural numbers
variable {b : ℕ → ℝ} -- Geometric sequence

-- This is only necessary if importing Mathlib didn't bring it in
noncomputable def geom_sequence (m n : ℕ) (b : ℕ → ℝ) (q : ℝ) : Prop :=
  b n = b m * q^(n - m)

theorem geometric_sequence_formula (q : ℝ) (m n : ℕ) (b : ℕ → ℝ) 
  (hmn : 0 < m ∧ 0 < n) :
  geom_sequence m n b q :=
by sorry

end geometric_sequence_formula_l1980_198047


namespace solve_abs_inequality_l1980_198032

theorem solve_abs_inequality :
  { x : ℝ | 3 ≤ |x - 2| ∧ |x - 2| ≤ 6 } = { x : ℝ | -4 ≤ x ∧ x ≤ -1 } ∪ { x : ℝ | 5 ≤ x ∧ x ≤ 8 } :=
sorry

end solve_abs_inequality_l1980_198032


namespace time_to_cross_platform_l1980_198056

-- Definition of the given conditions
def length_of_train : ℕ := 1500 -- in meters
def time_to_cross_tree : ℕ := 120 -- in seconds
def length_of_platform : ℕ := 500 -- in meters
def speed : ℚ := length_of_train / time_to_cross_tree -- speed in meters per second

-- Definition of the total distance to cross the platform
def total_distance : ℕ := length_of_train + length_of_platform

-- Theorem to prove the time taken to cross the platform
theorem time_to_cross_platform : (total_distance / speed) = 160 :=
by
  -- Placeholder for the proof
  sorry

end time_to_cross_platform_l1980_198056


namespace ascending_order_l1980_198092

theorem ascending_order : (3 / 8 : ℝ) < 0.75 ∧ 
                          0.75 < (1 + 2 / 5 : ℝ) ∧ 
                          (1 + 2 / 5 : ℝ) < 1.43 ∧
                          1.43 < (13 / 8 : ℝ) :=
by
  sorry

end ascending_order_l1980_198092


namespace find_distance_between_sides_of_trapezium_l1980_198057

variable (side1 side2 h area : ℝ)
variable (h1 : side1 = 20)
variable (h2 : side2 = 18)
variable (h3 : area = 228)
variable (trapezium_area : area = (1 / 2) * (side1 + side2) * h)

theorem find_distance_between_sides_of_trapezium : h = 12 := by
  sorry

end find_distance_between_sides_of_trapezium_l1980_198057


namespace factor_expression_l1980_198086

theorem factor_expression (y : ℤ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) :=
by 
  sorry

end factor_expression_l1980_198086


namespace system_solution_l1980_198093

theorem system_solution (x y : ℝ) 
  (h1 : 0 < x + y) 
  (h2 : x + y ≠ 1) 
  (h3 : 2 * x - y ≠ 0)
  (eq1 : (x + y) * 2^(y - 2 * x) = 6.25) 
  (eq2 : (x + y) * (1 / (2 * x - y)) = 5) :
x = 9 ∧ y = 16 := 
sorry

end system_solution_l1980_198093


namespace find_value_l1980_198098

theorem find_value : (100 + (20 / 90)) * 90 = 120 := by
  sorry

end find_value_l1980_198098


namespace largest_num_consecutive_integers_sum_45_l1980_198046

theorem largest_num_consecutive_integers_sum_45 : 
  ∃ n : ℕ, (0 < n) ∧ (n * (n + 1) / 2 = 45) ∧ (∀ m : ℕ, (0 < m) → m * (m + 1) / 2 = 45 → m ≤ n) :=
by {
  sorry
}

end largest_num_consecutive_integers_sum_45_l1980_198046


namespace variance_of_binomial_distribution_l1980_198081

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_binomial_distribution :
  binomial_variance 10 (2/5) = 12 / 5 :=
by
  sorry

end variance_of_binomial_distribution_l1980_198081


namespace book_cost_price_l1980_198061

theorem book_cost_price
  (C : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : SP = 1.25 * C)
  (h2 : 0.95 * P = SP)
  (h3 : SP = 62.5) : 
  C = 50 := 
by
  sorry

end book_cost_price_l1980_198061


namespace existence_of_subset_A_l1980_198091

def M : Set ℚ := {x : ℚ | 0 < x ∧ x < 1}

theorem existence_of_subset_A :
  ∃ A ⊆ M, ∀ m ∈ M, ∃! (S : Finset ℚ), (∀ a ∈ S, a ∈ A) ∧ (S.sum id = m) :=
sorry

end existence_of_subset_A_l1980_198091


namespace students_at_end_of_year_l1980_198065

-- Define the initial number of students
def initial_students : Nat := 10

-- Define the number of students who left during the year
def students_left : Nat := 4

-- Define the number of new students who arrived during the year
def new_students : Nat := 42

-- Proof problem: the number of students at the end of the year
theorem students_at_end_of_year : initial_students - students_left + new_students = 48 := by
  sorry

end students_at_end_of_year_l1980_198065


namespace shaded_area_l1980_198080

theorem shaded_area (r₁ r₂ r₃ : ℝ) (h₁ : 0 < r₁) (h₂ : 0 < r₂) (h₃ : 0 < r₃) (h₁₂ : r₁ < r₂) (h₂₃ : r₂ < r₃)
    (area_shaded_div_area_unshaded : (r₁^2 * π) + (r₂^2 * π) + (r₃^2 * π) = 77 * π)
    (shaded_by_unshaded_ratio : ∀ S U : ℝ, S = (3 / 7) * U) :
    ∃ S : ℝ, S = (1617 * π) / 70 :=
by
  sorry

end shaded_area_l1980_198080


namespace normal_cost_of_car_wash_l1980_198062

-- Conditions
variables (C : ℝ) (H1 : 20 * C > 0) (H2 : 0.60 * (20 * C) = 180)

-- Theorem to be proved
theorem normal_cost_of_car_wash (C : ℝ) (H1 : 20 * C > 0) (H2 : 0.60 * (20 * C) = 180) : C = 15 :=
by
  -- proof omitted
  sorry

end normal_cost_of_car_wash_l1980_198062


namespace sin_x_lt_a_l1980_198008

theorem sin_x_lt_a (a θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (hθ : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2 * n - 1) * Real.pi - θ < x ∧ x < 2 * n * Real.pi + θ} = {x : ℝ | Real.sin x < a} :=
sorry

end sin_x_lt_a_l1980_198008


namespace decreasing_function_a_leq_zero_l1980_198079

theorem decreasing_function_a_leq_zero (a : ℝ) :
  (∀ x y : ℝ, x < y → ax^3 - x ≥ ay^3 - y) → a ≤ 0 :=
by
  sorry

end decreasing_function_a_leq_zero_l1980_198079


namespace students_played_both_l1980_198051

theorem students_played_both (C B X total : ℕ) (hC : C = 500) (hB : B = 600) (hTotal : total = 880) (hInclusionExclusion : C + B - X = total) : X = 220 :=
by
  rw [hC, hB, hTotal] at hInclusionExclusion
  sorry

end students_played_both_l1980_198051


namespace problem_solution_l1980_198012

-- Define the conditions
variables {a c b d x y z q : Real}
axiom h1 : a^x = c^q ∧ c^q = b
axiom h2 : c^y = a^z ∧ a^z = d

-- State the theorem
theorem problem_solution : xy = zq :=
by
  sorry

end problem_solution_l1980_198012


namespace sufficient_but_not_necessary_l1980_198019

variables {a b : ℝ}

theorem sufficient_but_not_necessary (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end sufficient_but_not_necessary_l1980_198019


namespace tile_arrangement_probability_l1980_198017

theorem tile_arrangement_probability :
  let X := 5
  let O := 4
  let total_tiles := 9
  (1 : ℚ) / (Nat.choose total_tiles X) = 1 / 126 :=
by
  sorry

end tile_arrangement_probability_l1980_198017


namespace current_walnut_trees_l1980_198043

theorem current_walnut_trees (x : ℕ) (h : x + 55 = 77) : x = 22 :=
by
  sorry

end current_walnut_trees_l1980_198043


namespace sum_of_ages_is_55_l1980_198038

def sum_of_ages (Y : ℕ) (interval : ℕ) (number_of_children : ℕ) : ℕ :=
  let ages := List.range number_of_children |>.map (λ i => Y + i * interval)
  ages.sum

theorem sum_of_ages_is_55 :
  sum_of_ages 7 2 5 = 55 :=
by
  sorry

end sum_of_ages_is_55_l1980_198038


namespace number_of_8_digit_increasing_integers_mod_1000_l1980_198095

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_8_digit_increasing_integers_mod_1000 :
  let M := choose 9 8
  M % 1000 = 9 :=
by
  let M := choose 9 8
  show M % 1000 = 9
  sorry

end number_of_8_digit_increasing_integers_mod_1000_l1980_198095


namespace julia_miles_l1980_198067

theorem julia_miles (total_miles darius_miles julia_miles : ℕ) 
  (h1 : darius_miles = 679)
  (h2 : total_miles = 1677)
  (h3 : total_miles = darius_miles + julia_miles) :
  julia_miles = 998 :=
by
  sorry

end julia_miles_l1980_198067


namespace number_of_integer_values_for_a_l1980_198011

theorem number_of_integer_values_for_a :
  (∃ (a : Int), ∃ (p q : Int), p * q = -12 ∧ p + q = a ∧ p ≠ q) →
  (∃ (n : Nat), n = 6) := by
  sorry

end number_of_integer_values_for_a_l1980_198011


namespace lawrence_walked_total_distance_l1980_198089

noncomputable def distance_per_day : ℝ := 4.0
noncomputable def number_of_days : ℝ := 3.0
noncomputable def total_distance_walked (distance_per_day : ℝ) (number_of_days : ℝ) : ℝ :=
  distance_per_day * number_of_days

theorem lawrence_walked_total_distance :
  total_distance_walked distance_per_day number_of_days = 12.0 :=
by
  -- The detailed proof is omitted as per the instructions.
  sorry

end lawrence_walked_total_distance_l1980_198089


namespace greatest_possible_x_exists_greatest_x_l1980_198048

theorem greatest_possible_x (x : ℤ) (h1 : 6.1 * (10 : ℝ) ^ x < 620) : x ≤ 2 :=
sorry

theorem exists_greatest_x : ∃ x : ℤ, 6.1 * (10 : ℝ) ^ x < 620 ∧ x = 2 :=
sorry

end greatest_possible_x_exists_greatest_x_l1980_198048


namespace harry_morning_routine_time_l1980_198014

-- Define the conditions in Lean.
def buy_coffee_and_bagel_time : ℕ := 15 -- minutes
def read_and_eat_time : ℕ := 2 * buy_coffee_and_bagel_time -- twice the time for buying coffee and bagel is 30 minutes

-- Define the total morning routine time in Lean.
def total_morning_routine_time : ℕ := buy_coffee_and_bagel_time + read_and_eat_time

-- The final proof problem statement.
theorem harry_morning_routine_time :
  total_morning_routine_time = 45 :=
by
  unfold total_morning_routine_time
  unfold read_and_eat_time
  unfold buy_coffee_and_bagel_time
  sorry

end harry_morning_routine_time_l1980_198014


namespace evaluate_g_at_3_l1980_198041

def g (x : ℝ) : ℝ := 7 * x^3 - 8 * x^2 - 5 * x + 7

theorem evaluate_g_at_3 : g 3 = 109 := by
  sorry

end evaluate_g_at_3_l1980_198041


namespace vector_addition_l1980_198025

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

-- State the problem as a theorem
theorem vector_addition : a + b = (-1, 5) := by
  -- the proof should go here
  sorry

end vector_addition_l1980_198025


namespace folded_triangle_sqrt_equals_l1980_198072

noncomputable def folded_triangle_length_squared (s : ℕ) (d : ℕ) : ℚ :=
  let x := (2 * s * s - 2 * d * s)/(2 * d)
  let y := (2 * s * s - 2 * (s - d) * s)/(2 * (s - d))
  x * x - x * y + y * y

theorem folded_triangle_sqrt_equals :
  folded_triangle_length_squared 15 11 = (60118.9025 / 1681 : ℚ) := sorry

end folded_triangle_sqrt_equals_l1980_198072


namespace minimum_possible_value_of_BC_l1980_198003

def triangle_ABC_side_lengths_are_integers (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def angle_A_is_twice_angle_B (A B C : ℝ) : Prop :=
  A = 2 * B

def CA_is_nine (CA : ℕ) : Prop :=
  CA = 9

theorem minimum_possible_value_of_BC
  (a b c : ℕ) (A B C : ℝ) (CA : ℕ)
  (h1 : triangle_ABC_side_lengths_are_integers a b c)
  (h2 : angle_A_is_twice_angle_B A B C)
  (h3 : CA_is_nine CA) :
  ∃ (BC : ℕ), BC = 12 := 
sorry

end minimum_possible_value_of_BC_l1980_198003


namespace last_digit_sum_l1980_198006

theorem last_digit_sum (a b : ℕ) (exp : ℕ)
  (h₁ : a = 1993) (h₂ : b = 1995) (h₃ : exp = 2002) :
  ((a ^ exp + b ^ exp) % 10) = 4 := 
by
  sorry

end last_digit_sum_l1980_198006
