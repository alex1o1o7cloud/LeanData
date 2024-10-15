import Mathlib

namespace NUMINAMATH_GPT_whatsapp_messages_total_l770_77035

-- Define conditions
def messages_monday : ℕ := 300
def messages_tuesday : ℕ := 200
def messages_wednesday : ℕ := messages_tuesday + 300
def messages_thursday : ℕ := 2 * messages_wednesday
def messages_friday : ℕ := messages_thursday + (20 * messages_thursday) / 100
def messages_saturday : ℕ := messages_friday - (10 * messages_friday) / 100

-- Theorem statement to be proved
theorem whatsapp_messages_total :
  messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday + messages_saturday = 4280 :=
by 
  sorry

end NUMINAMATH_GPT_whatsapp_messages_total_l770_77035


namespace NUMINAMATH_GPT_imag_part_z_is_3_l770_77034

namespace ComplexMultiplication

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := (1 + 2 * i) * (2 - i)

-- Define the imaginary part of a complex number
def imag_part (z : ℂ) : ℂ := Complex.im z

-- Statement to prove: The imaginary part of z = 3
theorem imag_part_z_is_3 : imag_part z = 3 := by
  sorry

end ComplexMultiplication

end NUMINAMATH_GPT_imag_part_z_is_3_l770_77034


namespace NUMINAMATH_GPT_zero_point_interval_l770_77073

noncomputable def f (x : ℝ) : ℝ := -x^3 - 3 * x + 5

theorem zero_point_interval: 
  ∃ x₀ : ℝ, f x₀ = 0 → 1 < x₀ ∧ x₀ < 2 :=
sorry

end NUMINAMATH_GPT_zero_point_interval_l770_77073


namespace NUMINAMATH_GPT_alice_distance_from_start_l770_77086

theorem alice_distance_from_start :
  let hexagon_side := 3
  let distance_walked := 10
  let final_distance := 3 * Real.sqrt 3 / 2
  final_distance =
    let a := (0, 0)
    let b := (3, 0)
    let c := (4.5, 3 * Real.sqrt 3 / 2)
    let d := (1.5, 3 * Real.sqrt 3 / 2)
    let e := (0, 3 * Real.sqrt 3 / 2)
    dist a e := sorry

end NUMINAMATH_GPT_alice_distance_from_start_l770_77086


namespace NUMINAMATH_GPT_find_coordinates_of_D_l770_77019

theorem find_coordinates_of_D
  (A B C D : ℝ × ℝ)
  (hA : A = (-1, 2))
  (hB : B = (0, 0))
  (hC : C = (1, 7))
  (hParallelogram : ∃ u v, u * (B - A) + v * (C - D) = (0, 0) ∧ u * (C - D) + v * (B - A) = (0, 0)) :
  D = (0, 9) :=
sorry

end NUMINAMATH_GPT_find_coordinates_of_D_l770_77019


namespace NUMINAMATH_GPT_FGH_supermarkets_US_l770_77055

/-- There are 60 supermarkets in the FGH chain,
all of them are either in the US or Canada,
there are 14 more FGH supermarkets in the US than in Canada.
Prove that there are 37 FGH supermarkets in the US. -/
theorem FGH_supermarkets_US (C U : ℕ) (h1 : C + U = 60) (h2 : U = C + 14) : U = 37 := by
  sorry

end NUMINAMATH_GPT_FGH_supermarkets_US_l770_77055


namespace NUMINAMATH_GPT_avg_rate_of_change_eq_l770_77083

variable (Δx : ℝ)

def function_y (x : ℝ) : ℝ := x^2 + 1

theorem avg_rate_of_change_eq : (function_y (1 + Δx) - function_y 1) / Δx = 2 + Δx :=
by
  sorry

end NUMINAMATH_GPT_avg_rate_of_change_eq_l770_77083


namespace NUMINAMATH_GPT_second_student_catches_up_l770_77082

open Nat

-- Definitions for the problems
def distance_first_student (n : ℕ) : ℕ := 7 * n
def distance_second_student (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem statement indicating the second student catches up with the first at n = 13
theorem second_student_catches_up : ∃ n, (distance_first_student n = distance_second_student n) ∧ n = 13 := 
by 
  sorry

end NUMINAMATH_GPT_second_student_catches_up_l770_77082


namespace NUMINAMATH_GPT_trains_cross_each_other_in_5_76_seconds_l770_77068

noncomputable def trains_crossing_time (l1 l2 v1_kmh v2_kmh : ℕ) : ℚ :=
  let v1 := (v1_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let v2 := (v2_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let total_distance := (l1 : ℚ) + (l2 : ℚ)
  let relative_velocity := v1 + v2
  total_distance / relative_velocity

theorem trains_cross_each_other_in_5_76_seconds :
  trains_crossing_time 100 60 60 40 = 160 / 27.78 := by
  sorry

end NUMINAMATH_GPT_trains_cross_each_other_in_5_76_seconds_l770_77068


namespace NUMINAMATH_GPT_entrance_exam_proof_l770_77081

-- Define the conditions
variables (x y : ℕ)
variables (h1 : x + y = 70)
variables (h2 : 3 * x - y = 38)

-- The proof goal
theorem entrance_exam_proof : x = 27 :=
by
  -- The actual proof steps are omitted here
  sorry

end NUMINAMATH_GPT_entrance_exam_proof_l770_77081


namespace NUMINAMATH_GPT_factorial_sum_power_of_two_l770_77024

theorem factorial_sum_power_of_two (a b c n : ℕ) (h : a ≤ b ∧ b ≤ c) :
  a! + b! + c! = 2^n →
  (a = 1 ∧ b = 1 ∧ c = 2) ∨
  (a = 1 ∧ b = 1 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 4) ∨
  (a = 2 ∧ b = 3 ∧ c = 5) :=
by
  sorry

end NUMINAMATH_GPT_factorial_sum_power_of_two_l770_77024


namespace NUMINAMATH_GPT_bubble_gum_cost_l770_77014

-- Define the conditions
def total_cost : ℕ := 2448
def number_of_pieces : ℕ := 136

-- Main theorem to state that each piece of bubble gum costs 18 cents
theorem bubble_gum_cost : total_cost / number_of_pieces = 18 :=
by
  sorry

end NUMINAMATH_GPT_bubble_gum_cost_l770_77014


namespace NUMINAMATH_GPT_teaching_arrangements_l770_77065

-- Define the conditions
structure Conditions :=
  (teach_A : ℕ)
  (teach_B : ℕ)
  (teach_C : ℕ)
  (teach_D : ℕ)
  (max_teach_AB : ∀ t, t = teach_A ∨ t = teach_B → t ≤ 2)
  (max_teach_CD : ∀ t, t = teach_C ∨ t = teach_D → t ≤ 1)
  (total_periods : ℕ)
  (teachers_per_period : ℕ)

-- Constants and assumptions
def problem_conditions : Conditions := {
  teach_A := 2,
  teach_B := 2,
  teach_C := 1,
  teach_D := 1,
  max_teach_AB := by sorry,
  max_teach_CD := by sorry,
  total_periods := 2,
  teachers_per_period := 2
}

-- Define the proof goal
theorem teaching_arrangements (c : Conditions) :
  c = problem_conditions → ∃ arrangements, arrangements = 19 :=
by
  sorry

end NUMINAMATH_GPT_teaching_arrangements_l770_77065


namespace NUMINAMATH_GPT_star_operation_result_l770_77045

def set_minus (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∉ B}

def set_star (A B : Set ℝ) : Set ℝ :=
  set_minus A B ∪ set_minus B A

def A : Set ℝ := { y : ℝ | y ≥ 0 }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 3 }

theorem star_operation_result :
  set_star A B = {x : ℝ | (-3 ≤ x ∧ x < 0) ∨ (x > 3)} :=
  sorry

end NUMINAMATH_GPT_star_operation_result_l770_77045


namespace NUMINAMATH_GPT_intersection_A_complement_B_eq_minus_three_to_zero_l770_77058

-- Define the set A
def A : Set ℝ := { x : ℝ | x^2 + x - 6 ≤ 0 }

-- Define the set B
def B : Set ℝ := { y : ℝ | ∃ x : ℝ, y = Real.sqrt x ∧ 0 ≤ x ∧ x ≤ 4 }

-- Define the complement of B
def C_RB : Set ℝ := { y : ℝ | ¬ (y ∈ B) }

-- The proof problem
theorem intersection_A_complement_B_eq_minus_three_to_zero :
  (A ∩ C_RB) = { x : ℝ | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_complement_B_eq_minus_three_to_zero_l770_77058


namespace NUMINAMATH_GPT_cos_four_alpha_sub_9pi_over_2_l770_77063

open Real

theorem cos_four_alpha_sub_9pi_over_2 (α : ℝ) 
  (cond : 4.53 * (1 + cos (2 * α - 2 * π) + cos (4 * α + 2 * π) - cos (6 * α - π)) /
                  (cos (2 * π - 2 * α) + 2 * cos (2 * α + π) ^ 2 - 1) = 2 * cos (2 * α)) :
  cos (4 * α - 9 * π / 2) = cos (4 * α - π / 2) :=
by sorry

end NUMINAMATH_GPT_cos_four_alpha_sub_9pi_over_2_l770_77063


namespace NUMINAMATH_GPT_cookies_difference_l770_77097

theorem cookies_difference :
  let bags := 9
  let boxes := 8
  let cookies_per_bag := 7
  let cookies_per_box := 12
  8 * 12 - 9 * 7 = 33 := 
by
  sorry

end NUMINAMATH_GPT_cookies_difference_l770_77097


namespace NUMINAMATH_GPT_find_5y_45_sevenths_l770_77030

theorem find_5y_45_sevenths (x y : ℝ) 
(h1 : 3 * x + 4 * y = 0) 
(h2 : x = y + 3) : 
5 * y = -45 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_5y_45_sevenths_l770_77030


namespace NUMINAMATH_GPT_total_amount_spent_l770_77031

noncomputable def food_price : ℝ := 160
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def tip_rate : ℝ := 0.20

theorem total_amount_spent :
  let sales_tax := sales_tax_rate * food_price
  let total_before_tip := food_price + sales_tax
  let tip := tip_rate * total_before_tip
  let total_amount := total_before_tip + tip
  total_amount = 211.20 :=
by
  -- include the proof logic here if necessary
  sorry

end NUMINAMATH_GPT_total_amount_spent_l770_77031


namespace NUMINAMATH_GPT_ticTacToeConfigCorrect_l770_77069

def ticTacToeConfigCount (board : Fin 3 → Fin 3 → Option Char) : Nat := 
  sorry -- this function will count the configurations according to the game rules

theorem ticTacToeConfigCorrect (board : Fin 3 → Fin 3 → Option Char) :
  ticTacToeConfigCount board = 438 := 
  sorry

end NUMINAMATH_GPT_ticTacToeConfigCorrect_l770_77069


namespace NUMINAMATH_GPT_paolo_coconuts_l770_77061

theorem paolo_coconuts
  (P : ℕ)
  (dante_coconuts : ℕ := 3 * P)
  (dante_sold : ℕ := 10)
  (dante_left : ℕ := 32)
  (h : dante_left + dante_sold = dante_coconuts) : P = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_paolo_coconuts_l770_77061


namespace NUMINAMATH_GPT_monotonically_increasing_interval_l770_77036

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos ((2 / 3) * x - (5 * Real.pi / 12))

theorem monotonically_increasing_interval 
  (φ : ℝ) (h1 : -Real.pi / 2 < φ) (h2 : φ < 0) 
  (h3 : 2 * (Real.pi / 8) + φ = Real.pi / 4) : 
  ∀ x : ℝ, (-(Real.pi / 2) ≤ x) ∧ (x ≤ Real.pi / 2) ↔ ∃ k : ℤ, x ∈ [(-7 * Real.pi / 8 + 3 * k * Real.pi), (5 * Real.pi / 8 + 3 * k * Real.pi)] :=
sorry

end NUMINAMATH_GPT_monotonically_increasing_interval_l770_77036


namespace NUMINAMATH_GPT_evaluate_expression_l770_77013

theorem evaluate_expression (x : ℝ) : (1 - x^2) * (1 + x^4) = 1 - x^2 + x^4 - x^6 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l770_77013


namespace NUMINAMATH_GPT_angle_problem_l770_77079

-- Definitions for degrees and minutes
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Adding two angles
def add_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := a1.minutes + a2.minutes
  let extra_degrees := total_minutes / 60
  { degrees := a1.degrees + a2.degrees + extra_degrees,
    minutes := total_minutes % 60 }

-- Subtracting two angles
def sub_angles (a1 a2 : Angle) : Angle :=
  let total_minutes := if a1.minutes < a2.minutes then a1.minutes + 60 else a1.minutes
  let extra_deg := if a1.minutes < a2.minutes then 1 else 0
  { degrees := a1.degrees - a2.degrees - extra_deg,
    minutes := total_minutes - a2.minutes }

-- Multiplying an angle by a constant
def mul_angle (a : Angle) (k : ℕ) : Angle :=
  let total_minutes := a.minutes * k
  let extra_degrees := total_minutes / 60
  { degrees := a.degrees * k + extra_degrees,
    minutes := total_minutes % 60 }

-- Given angles
def angle1 : Angle := { degrees := 24, minutes := 31}
def angle2 : Angle := { degrees := 62, minutes := 10}

-- Prove the problem statement
theorem angle_problem : sub_angles (mul_angle angle1 4) angle2 = { degrees := 35, minutes := 54} :=
  sorry

end NUMINAMATH_GPT_angle_problem_l770_77079


namespace NUMINAMATH_GPT_base_conversion_addition_l770_77009

theorem base_conversion_addition :
  (214 % 8 / 32 % 5 + 343 % 9 / 133 % 4) = 9134 / 527 :=
by sorry

end NUMINAMATH_GPT_base_conversion_addition_l770_77009


namespace NUMINAMATH_GPT_pentagon_square_ratio_l770_77015

theorem pentagon_square_ratio (s p : ℕ) (h1 : 4 * s = 20) (h2 : 5 * p = 20) :
  p / s = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_square_ratio_l770_77015


namespace NUMINAMATH_GPT_parallel_trans_l770_77016

variables {Line : Type} (a b c : Line)

-- Define parallel relation
def parallel (x y : Line) : Prop := sorry -- Replace 'sorry' with the actual definition

-- The main theorem
theorem parallel_trans (h1 : parallel a c) (h2 : parallel b c) : parallel a b :=
sorry

end NUMINAMATH_GPT_parallel_trans_l770_77016


namespace NUMINAMATH_GPT_point_on_x_axis_m_eq_2_l770_77005

theorem point_on_x_axis_m_eq_2 (m : ℝ) (h : (m + 5, m - 2).2 = 0) : m = 2 :=
sorry

end NUMINAMATH_GPT_point_on_x_axis_m_eq_2_l770_77005


namespace NUMINAMATH_GPT_smallest_positive_period_monotonic_decreasing_interval_l770_77053

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.sin x * Real.cos x

theorem smallest_positive_period (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧ T > 0 → T = Real.pi :=
by
  sorry

theorem monotonic_decreasing_interval :
  (∀ x, x ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8) → ∃ k : ℤ, 
     f (x + k * π) = f x ∧ f (x + k * π) ≤ f (x + (k + 1) * π)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_monotonic_decreasing_interval_l770_77053


namespace NUMINAMATH_GPT_ellipse_foci_distance_l770_77046

theorem ellipse_foci_distance :
  (∀ x y : ℝ, x^2 / 56 + y^2 / 14 = 8) →
  ∃ d : ℝ, d = 8 * Real.sqrt 21 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l770_77046


namespace NUMINAMATH_GPT_ana_salary_after_changes_l770_77043

-- Definitions based on conditions in part (a)
def initial_salary : ℝ := 2000
def raise_factor : ℝ := 1.20
def cut_factor : ℝ := 0.80

-- Statement of the proof problem
theorem ana_salary_after_changes : 
  (initial_salary * raise_factor * cut_factor) = 1920 :=
by
  sorry

end NUMINAMATH_GPT_ana_salary_after_changes_l770_77043


namespace NUMINAMATH_GPT_total_wholesale_cost_is_correct_l770_77094

-- Given values
def retail_price_pants : ℝ := 36
def markup_pants : ℝ := 0.8

def retail_price_shirt : ℝ := 45
def markup_shirt : ℝ := 0.6

def retail_price_jacket : ℝ := 120
def markup_jacket : ℝ := 0.5

noncomputable def wholesale_cost_pants : ℝ := retail_price_pants / (1 + markup_pants)
noncomputable def wholesale_cost_shirt : ℝ := retail_price_shirt / (1 + markup_shirt)
noncomputable def wholesale_cost_jacket : ℝ := retail_price_jacket / (1 + markup_jacket)

noncomputable def total_wholesale_cost : ℝ :=
  wholesale_cost_pants + wholesale_cost_shirt + wholesale_cost_jacket

theorem total_wholesale_cost_is_correct :
  total_wholesale_cost = 128.125 := by
  sorry

end NUMINAMATH_GPT_total_wholesale_cost_is_correct_l770_77094


namespace NUMINAMATH_GPT_range_of_a_l770_77066

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (x + 2) - abs (x - 1) ≥ a^3 - 4 * a^2 - 3) → a ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l770_77066


namespace NUMINAMATH_GPT_totalFriendsAreFour_l770_77029

-- Define the friends
def friends := ["Mary", "Sam", "Keith", "Alyssa"]

-- Define the number of friends
def numberOfFriends (f : List String) : ℕ := f.length

-- Claim that the number of friends is 4
theorem totalFriendsAreFour : numberOfFriends friends = 4 :=
by
  -- Skip proof
  sorry

end NUMINAMATH_GPT_totalFriendsAreFour_l770_77029


namespace NUMINAMATH_GPT_simplify_expression_l770_77080

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l770_77080


namespace NUMINAMATH_GPT_largest_whole_number_satisfying_inequality_l770_77003

theorem largest_whole_number_satisfying_inequality : ∃ n : ℤ, (1 / 3 + n / 7 < 1) ∧ (∀ m : ℤ, (1 / 3 + m / 7 < 1) → m ≤ n) ∧ n = 4 :=
sorry

end NUMINAMATH_GPT_largest_whole_number_satisfying_inequality_l770_77003


namespace NUMINAMATH_GPT_minimum_gloves_needed_l770_77022

-- Definitions based on conditions:
def participants : Nat := 43
def gloves_per_participant : Nat := 2

-- Problem statement proving the minimum number of gloves needed
theorem minimum_gloves_needed : participants * gloves_per_participant = 86 := by
  -- sorry allows us to omit the proof, focusing only on the formal statement
  sorry

end NUMINAMATH_GPT_minimum_gloves_needed_l770_77022


namespace NUMINAMATH_GPT_cylinder_to_sphere_volume_ratio_l770_77018

theorem cylinder_to_sphere_volume_ratio:
  ∀ (a r : ℝ), (a^2 = π * r^2) → (a^3)/( (4/3) * π * r^3) = 3/2 :=
by
  intros a r h
  sorry

end NUMINAMATH_GPT_cylinder_to_sphere_volume_ratio_l770_77018


namespace NUMINAMATH_GPT_evaluate_expression_l770_77002

theorem evaluate_expression : 5000 * 5000^3000 = 5000^3001 := 
by sorry

end NUMINAMATH_GPT_evaluate_expression_l770_77002


namespace NUMINAMATH_GPT_find_c_l770_77064

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end NUMINAMATH_GPT_find_c_l770_77064


namespace NUMINAMATH_GPT_max_pies_without_ingredients_l770_77098

def total_pies : ℕ := 30
def blueberry_pies : ℕ := total_pies / 3
def raspberry_pies : ℕ := (3 * total_pies) / 5
def blackberry_pies : ℕ := (5 * total_pies) / 6
def walnut_pies : ℕ := total_pies / 10

theorem max_pies_without_ingredients : 
  (total_pies - blackberry_pies) = 5 :=
by 
  -- We only require the proof part.
  sorry

end NUMINAMATH_GPT_max_pies_without_ingredients_l770_77098


namespace NUMINAMATH_GPT_money_left_after_deductions_l770_77056

-- Define the weekly income
def weekly_income : ℕ := 500

-- Define the tax deduction as 10% of the weekly income
def tax : ℕ := (10 * weekly_income) / 100

-- Define the weekly water bill
def water_bill : ℕ := 55

-- Define the tithe as 10% of the weekly income
def tithe : ℕ := (10 * weekly_income) / 100

-- Define the total deductions
def total_deductions : ℕ := tax + water_bill + tithe

-- Define the money left
def money_left : ℕ := weekly_income - total_deductions

-- The statement to prove
theorem money_left_after_deductions : money_left = 345 := by
  sorry

end NUMINAMATH_GPT_money_left_after_deductions_l770_77056


namespace NUMINAMATH_GPT_least_positive_integer_lemma_l770_77021

theorem least_positive_integer_lemma :
  ∃ x : ℕ, x > 0 ∧ x + 7237 ≡ 5017 [MOD 12] ∧ (∀ y : ℕ, y > 0 ∧ y + 7237 ≡ 5017 [MOD 12] → x ≤ y) :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_lemma_l770_77021


namespace NUMINAMATH_GPT_divisible_by_56_l770_77085

theorem divisible_by_56 (n : ℕ) (h1 : ∃ k, 3 * n + 1 = k * k) (h2 : ∃ m, 4 * n + 1 = m * m) : 56 ∣ n := 
sorry

end NUMINAMATH_GPT_divisible_by_56_l770_77085


namespace NUMINAMATH_GPT_work_completion_days_l770_77025

theorem work_completion_days (d : ℝ) : (1 / 15 + 1 / d = 1 / 11.25) → d = 45 := sorry

end NUMINAMATH_GPT_work_completion_days_l770_77025


namespace NUMINAMATH_GPT_jake_peaches_l770_77052

noncomputable def steven_peaches : ℕ := 15
noncomputable def jake_fewer : ℕ := 7

theorem jake_peaches : steven_peaches - jake_fewer = 8 :=
by
  sorry

end NUMINAMATH_GPT_jake_peaches_l770_77052


namespace NUMINAMATH_GPT_height_of_flagpole_l770_77051

-- Define the given conditions
variables (h : ℝ) -- height of the flagpole
variables (s_f : ℝ) (s_b : ℝ) (h_b : ℝ) -- s_f: shadow length of flagpole, s_b: shadow length of building, h_b: height of building

-- Problem conditions
def flagpole_shadow := (s_f = 45)
def building_shadow := (s_b = 50)
def building_height := (h_b = 20)

-- Mathematically equivalent statement
theorem height_of_flagpole
  (h_f : ℝ) (hsf : flagpole_shadow s_f) (hsb : building_shadow s_b) (hhb : building_height h_b)
  (similar_conditions : h / s_f = h_b / s_b) :
  h_f = 18 :=
by
  sorry

end NUMINAMATH_GPT_height_of_flagpole_l770_77051


namespace NUMINAMATH_GPT_find_r_l770_77075

theorem find_r (r : ℝ) (h₁ : 0 < r) (h₂ : ∀ x y : ℝ, (x - y = r → x^2 + y^2 = r → False)) : r = 2 :=
sorry

end NUMINAMATH_GPT_find_r_l770_77075


namespace NUMINAMATH_GPT_find_x_2y_3z_l770_77001

theorem find_x_2y_3z (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (h1 : x ≤ y) (h2 : y ≤ z) (h3 : x + y + z = 12) (h4 : x * y + y * z + z * x = 41) :
  x + 2 * y + 3 * z = 29 :=
by
  sorry

end NUMINAMATH_GPT_find_x_2y_3z_l770_77001


namespace NUMINAMATH_GPT_variation_relationship_l770_77067

theorem variation_relationship (k j : ℝ) (y z x : ℝ) (h1 : x = k * y^3) (h2 : y = j * z^(1/5)) :
  ∃ m : ℝ, x = m * z^(3/5) :=
by
  sorry

end NUMINAMATH_GPT_variation_relationship_l770_77067


namespace NUMINAMATH_GPT_basketball_card_price_l770_77012

variable (x : ℝ)

def total_cost_basketball_cards (x : ℝ) : ℝ := 2 * x
def total_cost_baseball_cards : ℝ := 5 * 4
def total_spent : ℝ := 50 - 24

theorem basketball_card_price :
  total_cost_basketball_cards x + total_cost_baseball_cards = total_spent ↔ x = 3 := by
  sorry

end NUMINAMATH_GPT_basketball_card_price_l770_77012


namespace NUMINAMATH_GPT_compound_interest_principal_l770_77048

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (Real.exp (T * Real.log (1 + R / 100)) - 1)

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem compound_interest_principal :
  let P_SI := 2800.0000000000027
  let R_SI := 5
  let T_SI := 3
  let P_CI := 4000
  let R_CI := 10
  let T_CI := 2
  let SI := simple_interest P_SI R_SI T_SI
  let CI := 2 * SI
  CI = compound_interest P_CI R_CI T_CI → P_CI = 4000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_compound_interest_principal_l770_77048


namespace NUMINAMATH_GPT_second_occurrence_at_55_l770_77026

/-- On the highway, starting from 3 kilometers, there is a speed limit sign every 4 kilometers,
and starting from 10 kilometers, there is a speed monitoring device every 9 kilometers.
The first time both types of facilities are encountered simultaneously is at 19 kilometers.
The second time both types of facilities are encountered simultaneously is at 55 kilometers. -/
theorem second_occurrence_at_55 :
  ∀ (k : ℕ), (∃ n m : ℕ, 3 + 4 * n = k ∧ 10 + 9 * m = k ∧ 19 + 36 = k) := sorry

end NUMINAMATH_GPT_second_occurrence_at_55_l770_77026


namespace NUMINAMATH_GPT_deepak_present_age_l770_77072

theorem deepak_present_age (R D : ℕ) (h1 : R =  4 * D / 3) (h2 : R + 10 = 26) : D = 12 :=
by
  sorry

end NUMINAMATH_GPT_deepak_present_age_l770_77072


namespace NUMINAMATH_GPT_remainder_M_mod_32_l770_77090

def M := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_M_mod_32 : M % 32 = 17 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_M_mod_32_l770_77090


namespace NUMINAMATH_GPT_range_of_x_l770_77093

noncomputable def problem_statement (x : ℝ) : Prop :=
  ∀ m : ℝ, abs m ≤ 2 → m * x^2 - 2 * x - m + 1 < 0 

theorem range_of_x (x : ℝ) :
  problem_statement x → ( ( -1 + Real.sqrt 7) / 2 < x ∧ x < ( 1 + Real.sqrt 3) / 2) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_x_l770_77093


namespace NUMINAMATH_GPT_fifteenth_number_in_base_5_l770_77042

theorem fifteenth_number_in_base_5 :
  ∃ n : ℕ, n = 15 ∧ (n : ℕ) = 3 * 5^1 + 0 * 5^0 :=
by
  sorry

end NUMINAMATH_GPT_fifteenth_number_in_base_5_l770_77042


namespace NUMINAMATH_GPT_cubs_more_home_runs_l770_77007

-- Define the conditions for the Chicago Cubs
def cubs_home_runs_third_inning : Nat := 2
def cubs_home_runs_fifth_inning : Nat := 1
def cubs_home_runs_eighth_inning : Nat := 2

-- Define the conditions for the Cardinals
def cardinals_home_runs_second_inning : Nat := 1
def cardinals_home_runs_fifth_inning : Nat := 1

-- Total home runs scored by each team
def total_cubs_home_runs : Nat :=
  cubs_home_runs_third_inning + cubs_home_runs_fifth_inning + cubs_home_runs_eighth_inning

def total_cardinals_home_runs : Nat :=
  cardinals_home_runs_second_inning + cardinals_home_runs_fifth_inning

-- The statement to prove
theorem cubs_more_home_runs : total_cubs_home_runs - total_cardinals_home_runs = 3 := by
  sorry

end NUMINAMATH_GPT_cubs_more_home_runs_l770_77007


namespace NUMINAMATH_GPT_find_units_digit_of_n_l770_77011

-- Define the problem conditions
def units_digit (a : ℕ) : ℕ := a % 10

theorem find_units_digit_of_n (m n : ℕ) (h1 : units_digit m = 3) (h2 : units_digit (m * n) = 6) (h3 : units_digit (14^8) = 6) :
  units_digit n = 2 :=
  sorry

end NUMINAMATH_GPT_find_units_digit_of_n_l770_77011


namespace NUMINAMATH_GPT_cube_volume_given_face_area_l770_77092

theorem cube_volume_given_face_area (s : ℝ) (h : s^2 = 36) : s^3 = 216 := by
  sorry

end NUMINAMATH_GPT_cube_volume_given_face_area_l770_77092


namespace NUMINAMATH_GPT_least_value_of_sum_l770_77040

theorem least_value_of_sum (x y z : ℤ) 
  (h_cond : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z ≥ 56 :=
sorry

end NUMINAMATH_GPT_least_value_of_sum_l770_77040


namespace NUMINAMATH_GPT_katie_speed_l770_77044

theorem katie_speed (eugene_speed : ℝ)
  (brianna_ratio : ℝ)
  (katie_ratio : ℝ)
  (h1 : eugene_speed = 4)
  (h2 : brianna_ratio = 2 / 3)
  (h3 : katie_ratio = 7 / 5) :
  katie_ratio * (brianna_ratio * eugene_speed) = 56 / 15 := 
by
  sorry

end NUMINAMATH_GPT_katie_speed_l770_77044


namespace NUMINAMATH_GPT_solve_for_x_l770_77059

theorem solve_for_x :
  ∀ x : ℤ, (35 - (23 - (15 - x)) = (12 * 2) / 1 / 2) → x = -21 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_for_x_l770_77059


namespace NUMINAMATH_GPT_least_positive_divisible_by_primes_l770_77099

theorem least_positive_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7
  ∃ n : ℕ, n > 0 ∧ (n % p1 = 0) ∧ (n % p2 = 0) ∧ (n % p3 = 0) ∧ (n % p4 = 0) ∧ 
  (∀ m : ℕ, m > 0 → (m % p1 = 0) ∧ (m % p2 = 0) ∧ (m % p3 = 0) ∧ (m % p4 = 0) → m ≥ n) ∧ n = 210 := 
by {
  sorry
}

end NUMINAMATH_GPT_least_positive_divisible_by_primes_l770_77099


namespace NUMINAMATH_GPT_pair_of_operations_equal_l770_77047

theorem pair_of_operations_equal :
  (-3) ^ 3 = -(3 ^ 3) ∧
  (¬((-2) ^ 4 = -(2 ^ 4))) ∧
  (¬((3 / 2) ^ 2 = (2 / 3) ^ 2)) ∧
  (¬(2 ^ 3 = 3 ^ 2)) :=
by 
  sorry

end NUMINAMATH_GPT_pair_of_operations_equal_l770_77047


namespace NUMINAMATH_GPT_union_eq_set_l770_77037

noncomputable def M : Set ℤ := {x | |x| < 2}
noncomputable def N : Set ℤ := {-2, -1, 0}

theorem union_eq_set : M ∪ N = {-2, -1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_union_eq_set_l770_77037


namespace NUMINAMATH_GPT_binomial_distribution_parameters_l770_77057

noncomputable def E (n : ℕ) (p : ℝ) : ℝ := n * p
noncomputable def D (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_distribution_parameters (n : ℕ) (p : ℝ) 
  (h1 : E n p = 2.4) (h2 : D n p = 1.44) : 
  n = 6 ∧ p = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_binomial_distribution_parameters_l770_77057


namespace NUMINAMATH_GPT_initial_profit_percentage_l770_77060

theorem initial_profit_percentage
  (CP : ℝ)
  (h1 : CP = 2400)
  (h2 : ∀ SP : ℝ, 15 / 100 * CP = 120 + SP) :
  ∃ P : ℝ, (P / 100) * CP = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_profit_percentage_l770_77060


namespace NUMINAMATH_GPT_ticket_price_difference_l770_77017

noncomputable def price_difference (adult_price total_cost : ℕ) (num_adults num_children : ℕ) (child_price : ℕ) : ℕ :=
  adult_price - child_price

theorem ticket_price_difference :
  ∀ (adult_price total_cost num_adults num_children child_price : ℕ),
  adult_price = 19 →
  total_cost = 77 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_price + num_children * child_price = total_cost →
  price_difference adult_price total_cost num_adults num_children child_price = 6 :=
by
  intros
  simp [price_difference]
  sorry

end NUMINAMATH_GPT_ticket_price_difference_l770_77017


namespace NUMINAMATH_GPT_sum_first_n_terms_of_geometric_seq_l770_77070

variable {α : Type*} [LinearOrderedField α] (a r : α) (n : ℕ)

def geometric_sequence (a r : α) (n : ℕ) : α :=
  a * r ^ (n - 1)

def sum_geometric_sequence (a r : α) (n : ℕ) : α :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_first_n_terms_of_geometric_seq (h₁ : a * r + a * r^3 = 20) 
    (h₂ : a * r^2 + a * r^4 = 40) :
  sum_geometric_sequence a r n = 2^(n + 1) - 2 := 
sorry

end NUMINAMATH_GPT_sum_first_n_terms_of_geometric_seq_l770_77070


namespace NUMINAMATH_GPT_length_de_l770_77010

theorem length_de (a b c d e : ℝ) (ab bc cd de ac ae : ℝ)
  (H1 : ab = 5)
  (H2 : bc = 2 * cd)
  (H3 : ac = ab + bc)
  (H4 : ac = 11)
  (H5 : ae = ab + bc + cd + de)
  (H6 : ae = 18) :
  de = 4 :=
by {
  sorry
}

-- Explanation:
-- a, b, c, d, e are points on a straight line
-- ab, bc, cd, de, ac, ae are lengths of segments between these points
-- H1: ab = 5
-- H2: bc = 2 * cd
-- H3: ac = ab + bc
-- H4: ac = 11
-- H5: ae = ab + bc + cd + de
-- H6: ae = 18
-- Prove that de = 4

end NUMINAMATH_GPT_length_de_l770_77010


namespace NUMINAMATH_GPT_exists_infinite_bisecting_circles_l770_77084

-- Define circle and bisecting condition
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def bisects (B C : Circle) : Prop :=
  let chord_len := (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2
  (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2 = B.radius^2

-- Define the theorem statement
theorem exists_infinite_bisecting_circles (C1 C2 : Circle) (h : C1.center ≠ C2.center) :
  ∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧
  ∀ (b_center : ℝ × ℝ), (∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧ B.center = b_center) ↔
  2 * (C2.center.1 - C1.center.1) * b_center.1 + 2 * (C2.center.2 - C1.center.2) * b_center.2 =
  (C2.center.1^2 - C1.center.1^2) + (C2.center.2^2 - C1.center.2^2) + (C2.radius^2 - C1.radius^2) := 
sorry

end NUMINAMATH_GPT_exists_infinite_bisecting_circles_l770_77084


namespace NUMINAMATH_GPT_find_number_eq_l770_77076

theorem find_number_eq : ∃ x : ℚ, (35 / 100) * x = (25 / 100) * 40 ∧ x = 200 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_number_eq_l770_77076


namespace NUMINAMATH_GPT_jim_reads_less_hours_l770_77050

-- Conditions
def initial_speed : ℕ := 40 -- pages per hour
def initial_pages_per_week : ℕ := 600 -- pages
def speed_increase_factor : ℚ := 1.5
def new_pages_per_week : ℕ := 660 -- pages

-- Calculations based on conditions
def initial_hours_per_week : ℚ := initial_pages_per_week / initial_speed
def new_speed : ℚ := initial_speed * speed_increase_factor
def new_hours_per_week : ℚ := new_pages_per_week / new_speed

-- Theorem Statement
theorem jim_reads_less_hours :
  initial_hours_per_week - new_hours_per_week = 4 :=
  sorry

end NUMINAMATH_GPT_jim_reads_less_hours_l770_77050


namespace NUMINAMATH_GPT_each_child_receive_amount_l770_77087

def husband_weekly_contribution : ℕ := 335
def wife_weekly_contribution : ℕ := 225
def weeks_in_month : ℕ := 4
def months : ℕ := 6
def children : ℕ := 4

noncomputable def total_weekly_contribution : ℕ := husband_weekly_contribution + wife_weekly_contribution
noncomputable def total_savings : ℕ := total_weekly_contribution * (weeks_in_month * months)
noncomputable def half_savings : ℕ := total_savings / 2
noncomputable def amount_per_child : ℕ := half_savings / children

theorem each_child_receive_amount :
  amount_per_child = 1680 :=
by
  sorry

end NUMINAMATH_GPT_each_child_receive_amount_l770_77087


namespace NUMINAMATH_GPT_one_minus_repeating_decimal_l770_77020

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ := x

theorem one_minus_repeating_decimal:
  ∀ (x : ℚ), x = 1/3 → 1 - x = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_one_minus_repeating_decimal_l770_77020


namespace NUMINAMATH_GPT_students_left_is_31_l770_77074

-- Define the conditions based on the problem statement
def total_students : ℕ := 124
def checked_out_early : ℕ := 93

-- Define the theorem that states the problem we want to prove
theorem students_left_is_31 :
  total_students - checked_out_early = 31 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_students_left_is_31_l770_77074


namespace NUMINAMATH_GPT_savings_fraction_l770_77004

variable (P : ℝ) 
variable (S : ℝ)
variable (E : ℝ)
variable (T : ℝ)

theorem savings_fraction :
  (12 * P * S) = 2 * P * (1 - S) → S = 1 / 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_savings_fraction_l770_77004


namespace NUMINAMATH_GPT_distance_between_planes_is_zero_l770_77023

def plane1 (x y z : ℝ) : Prop := x - 2 * y + 2 * z = 9
def plane2 (x y z : ℝ) : Prop := 2 * x - 4 * y + 4 * z = 18

theorem distance_between_planes_is_zero :
  (∀ x y z : ℝ, plane1 x y z ↔ plane2 x y z) → 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_planes_is_zero_l770_77023


namespace NUMINAMATH_GPT_ratio_of_segments_l770_77027

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_segments_l770_77027


namespace NUMINAMATH_GPT_anna_stamp_count_correct_l770_77038

-- Defining the initial counts of stamps
def anna_initial := 37
def alison_initial := 28
def jeff_initial := 31

-- Defining the operations
def alison_gives_half_to_anna := alison_initial / 2
def anna_after_receiving_from_alison := anna_initial + alison_gives_half_to_anna
def anna_final := anna_after_receiving_from_alison - 2 + 1

-- Formalizing the proof problem
theorem anna_stamp_count_correct : anna_final = 50 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_anna_stamp_count_correct_l770_77038


namespace NUMINAMATH_GPT_color_divisors_with_conditions_l770_77039

/-- Define the primes, product of the first 100 primes, and set S -/
def first_100_primes : List Nat := sorry -- Assume we have the list of first 100 primes
def product_of_first_100_primes : Nat := first_100_primes.foldr (· * ·) 1
def S := {d : Nat | d > 1 ∧ ∃ m, product_of_first_100_primes = m * d}

/-- Statement of the problem in Lean 4 -/
theorem color_divisors_with_conditions :
  (∃ (k : Nat), (∀ (coloring : S → Fin k), 
    (∀ s1 s2 s3 : S, (s1 * s2 * s3 = product_of_first_100_primes) → (coloring s1 = coloring s2 ∨ coloring s1 = coloring s3 ∨ coloring s2 = coloring s3)) ∧
    (∀ c : Fin k, ∃ s : S, coloring s = c))) ↔ k = 100 := 
by
  sorry

end NUMINAMATH_GPT_color_divisors_with_conditions_l770_77039


namespace NUMINAMATH_GPT_red_or_black_prob_red_black_or_white_prob_l770_77033

-- Defining the probabilities
def prob_red : ℚ := 5 / 12
def prob_black : ℚ := 4 / 12
def prob_white : ℚ := 2 / 12
def prob_green : ℚ := 1 / 12

-- Question 1: Probability of drawing a red or black ball
theorem red_or_black_prob : prob_red + prob_black = 3 / 4 :=
by sorry

-- Question 2: Probability of drawing a red, black, or white ball
theorem red_black_or_white_prob : prob_red + prob_black + prob_white = 11 / 12 :=
by sorry

end NUMINAMATH_GPT_red_or_black_prob_red_black_or_white_prob_l770_77033


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l770_77006

theorem sum_arithmetic_sequence (m : ℕ) (S : ℕ → ℕ) 
  (h1 : S m = 30) 
  (h2 : S (3 * m) = 90) : 
  S (2 * m) = 60 := 
sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l770_77006


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l770_77077

theorem repeating_decimal_to_fraction : 
  (∃ (x : ℚ), x = 7 + 3 / 9) → 7 + 3 / 9 = 22 / 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l770_77077


namespace NUMINAMATH_GPT_second_player_wins_optimal_play_l770_77096

def players_take_turns : Prop := sorry
def win_condition (box_count : ℕ) : Prop := box_count = 21

theorem second_player_wins_optimal_play (boxes : Fin 11 → ℕ)
    (h_turns : players_take_turns)
    (h_win : ∀ i : Fin 11, win_condition (boxes i)) : 
    ∃ P : ℕ, P = 2 :=
sorry

end NUMINAMATH_GPT_second_player_wins_optimal_play_l770_77096


namespace NUMINAMATH_GPT_count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l770_77049

def is_progressive_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 : ℕ), 1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 ≤ 9 ∧
                          n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5

theorem count_five_digit_progressive_numbers : ∃ n, n = 126 :=
by
  sorry

theorem find_110th_five_digit_progressive_number : ∃ n, n = 34579 :=
by
  sorry

end NUMINAMATH_GPT_count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l770_77049


namespace NUMINAMATH_GPT_inequality_AM_GM_l770_77028

theorem inequality_AM_GM (a b t : ℝ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 0 < t) : 
  (a^2 / (b^t - 1) + b^(2 * t) / (a^t - 1)) ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_inequality_AM_GM_l770_77028


namespace NUMINAMATH_GPT_abe_age_sum_l770_77088

theorem abe_age_sum (x : ℕ) : 25 + (25 - x) = 29 ↔ x = 21 :=
by sorry

end NUMINAMATH_GPT_abe_age_sum_l770_77088


namespace NUMINAMATH_GPT_total_votes_is_240_l770_77041

-- Defining the problem conditions
variables (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ)
def score : ℤ := likes - dislikes
def percentage_likes : ℚ := 3 / 4
def percentage_dislikes : ℚ := 1 / 4

-- Stating the given conditions
axiom h1 : total_votes = likes + dislikes
axiom h2 : (likes : ℤ) = (percentage_likes * total_votes)
axiom h3 : (dislikes : ℤ) = (percentage_dislikes * total_votes)
axiom h4 : score = 120

-- The statement to prove
theorem total_votes_is_240 : total_votes = 240 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_is_240_l770_77041


namespace NUMINAMATH_GPT_Austin_work_hours_on_Wednesdays_l770_77091

variable {W : ℕ}

theorem Austin_work_hours_on_Wednesdays
  (h1 : 5 * 2 + 5 * W + 5 * 3 = 25 + 5 * W)
  (h2 : 6 * (25 + 5 * W) = 180)
  : W = 1 := by
  sorry

end NUMINAMATH_GPT_Austin_work_hours_on_Wednesdays_l770_77091


namespace NUMINAMATH_GPT_problem_projection_eq_l770_77054

variable (m n : ℝ × ℝ)
variable (m_val : m = (1, 2))
variable (n_val : n = (2, 3))

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def projection (u v : ℝ × ℝ) : ℝ :=
  (dot_product u v) / (magnitude v)

theorem problem_projection_eq : projection m n = (8 * Real.sqrt 13) / 13 :=
by
  rw [m_val, n_val]
  sorry

end NUMINAMATH_GPT_problem_projection_eq_l770_77054


namespace NUMINAMATH_GPT_triangle_ABC_right_angled_l770_77078

variable {α : Type*} [LinearOrderedField α]

variables (a b c : α)
variables (A B C : ℝ)

theorem triangle_ABC_right_angled
  (h1 : b^2 = c^2 + a^2 - c * a)
  (h2 : Real.sin A = 2 * Real.sin C)
  (h3 : Real.cos B = 1 / 2) :
  B = (Real.pi / 2) := by
  sorry

end NUMINAMATH_GPT_triangle_ABC_right_angled_l770_77078


namespace NUMINAMATH_GPT_rug_area_calculation_l770_77071

theorem rug_area_calculation (length_floor width_floor strip_width : ℕ)
  (h_length : length_floor = 10)
  (h_width : width_floor = 8)
  (h_strip : strip_width = 2) :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := by
  sorry

end NUMINAMATH_GPT_rug_area_calculation_l770_77071


namespace NUMINAMATH_GPT_giants_need_to_win_more_games_l770_77008

/-- The Giants baseball team is trying to make their league playoff.
They have played 20 games and won 12 of them. To make the playoffs, they need to win 2/3 of 
their games over the season. If there are 10 games left, how many do they have to win to
make the playoffs? 
-/
theorem giants_need_to_win_more_games (played won needed_won total remaining required_wins additional_wins : ℕ)
    (h1 : played = 20)
    (h2 : won = 12)
    (h3 : remaining = 10)
    (h4 : total = played + remaining)
    (h5 : total = 30)
    (h6 : required_wins = 2 * total / 3)
    (h7 : additional_wins = required_wins - won) :
    additional_wins = 8 := 
    by
      -- sorry should be used if the proof steps were required.
sorry

end NUMINAMATH_GPT_giants_need_to_win_more_games_l770_77008


namespace NUMINAMATH_GPT_original_number_is_3_l770_77095

theorem original_number_is_3 
  (A B C D E : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 8) 
  (h2 : (8 + B + C + D + E) / 5 = 9): 
  A = 3 :=
sorry

end NUMINAMATH_GPT_original_number_is_3_l770_77095


namespace NUMINAMATH_GPT_inequality_solution_set_l770_77000

theorem inequality_solution_set (a b c : ℝ)
  (h1 : a < 0)
  (h2 : b = -a)
  (h3 : c = -2 * a) :
  ∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -1 ∨ x > 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l770_77000


namespace NUMINAMATH_GPT_number_of_a_l770_77089

theorem number_of_a (h : ∃ a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2) : 
  ∃! a : ℝ, ∃! x : ℝ, |x^2 + 2 * a * x + 3 * a| ≤ 2 :=
sorry

end NUMINAMATH_GPT_number_of_a_l770_77089


namespace NUMINAMATH_GPT_probability_of_a_plus_b_gt_5_l770_77062

noncomputable def all_events : Finset (ℕ × ℕ) := 
  { (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4) }

noncomputable def successful_events : Finset (ℕ × ℕ) :=
  { (2, 4), (3, 3), (3, 4) }

theorem probability_of_a_plus_b_gt_5 : 
  (successful_events.card : ℚ) / (all_events.card : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_of_a_plus_b_gt_5_l770_77062


namespace NUMINAMATH_GPT_smallest_x_for_quadratic_l770_77032

theorem smallest_x_for_quadratic :
  ∃ x, 8 * x^2 - 38 * x + 35 = 0 ∧ (∀ y, 8 * y^2 - 38 * y + 35 = 0 → x ≤ y) ∧ x = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_for_quadratic_l770_77032
