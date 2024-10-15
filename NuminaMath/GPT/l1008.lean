import Mathlib

namespace NUMINAMATH_GPT_work_completion_by_C_l1008_100896

theorem work_completion_by_C
  (A_work_rate : ℝ)
  (B_work_rate : ℝ)
  (C_work_rate : ℝ)
  (A_days_worked : ℝ)
  (B_days_worked : ℝ)
  (C_days_worked : ℝ)
  (A_total_days : ℝ)
  (B_total_days : ℝ)
  (C_completion_partial_work : ℝ)
  (H1 : A_work_rate = 1 / 40)
  (H2 : B_work_rate = 1 / 40)
  (H3 : A_days_worked = 10)
  (H4 : B_days_worked = 10)
  (H5 : C_days_worked = 10)
  (H6 : C_completion_partial_work = 1/2) :
  C_work_rate = 1 / 20 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_by_C_l1008_100896


namespace NUMINAMATH_GPT_jordan_time_for_7_miles_l1008_100801

noncomputable def time_for_7_miles (jordan_miles : ℕ) (jordan_time : ℤ) : ℤ :=
  jordan_miles * jordan_time 

theorem jordan_time_for_7_miles :
  ∃ jordan_time : ℤ, (time_for_7_miles 7 (16 / 3)) = 112 / 3 :=
by
  sorry

end NUMINAMATH_GPT_jordan_time_for_7_miles_l1008_100801


namespace NUMINAMATH_GPT_misread_number_l1008_100857

theorem misread_number (X : ℕ) :
  (average_10_initial : ℕ) = 18 →
  (incorrect_read : ℕ) = 26 →
  (average_10_correct : ℕ) = 22 →
  (10 * 22 - 10 * 18 = X + 26 - 26) →
  X = 66 :=
by sorry

end NUMINAMATH_GPT_misread_number_l1008_100857


namespace NUMINAMATH_GPT_min_value_x_add_y_l1008_100867

variable {x y : ℝ}
variable (hx : 0 < x) (hy : 0 < y)
variable (h : 2 * x + 8 * y - x * y = 0)

theorem min_value_x_add_y : x + y ≥ 18 :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_min_value_x_add_y_l1008_100867


namespace NUMINAMATH_GPT_shirt_discount_l1008_100851

theorem shirt_discount (original_price discounted_price : ℕ) 
  (h1 : original_price = 22) 
  (h2 : discounted_price = 16) : 
  original_price - discounted_price = 6 := 
by
  sorry

end NUMINAMATH_GPT_shirt_discount_l1008_100851


namespace NUMINAMATH_GPT_silvia_percentage_shorter_l1008_100831

theorem silvia_percentage_shorter :
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  (abs (( (j - s) / j) * 100 - 25) < 1) :=
by
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  show (abs (( (j - s) / j) * 100 - 25) < 1)
  sorry

end NUMINAMATH_GPT_silvia_percentage_shorter_l1008_100831


namespace NUMINAMATH_GPT_alice_total_cost_usd_is_correct_l1008_100808

def tea_cost_yen : ℕ := 250
def sandwich_cost_yen : ℕ := 350
def conversion_rate : ℕ := 100
def total_cost_usd (tea_cost_yen sandwich_cost_yen conversion_rate : ℕ) : ℕ :=
  (tea_cost_yen + sandwich_cost_yen) / conversion_rate

theorem alice_total_cost_usd_is_correct :
  total_cost_usd tea_cost_yen sandwich_cost_yen conversion_rate = 6 := 
by
  sorry

end NUMINAMATH_GPT_alice_total_cost_usd_is_correct_l1008_100808


namespace NUMINAMATH_GPT_johnny_marbles_l1008_100849

def num_ways_to_choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem johnny_marbles :
  num_ways_to_choose_marbles 7 3 = 35 :=
by
  sorry

end NUMINAMATH_GPT_johnny_marbles_l1008_100849


namespace NUMINAMATH_GPT_sum_of_valid_single_digit_z_l1008_100864

theorem sum_of_valid_single_digit_z :
  let valid_z (z : ℕ) := z < 10 ∧ (16 + z) % 3 = 0
  let sum_z := (Finset.filter valid_z (Finset.range 10)).sum id
  sum_z = 15 :=
by
  -- Proof steps are omitted
  sorry

end NUMINAMATH_GPT_sum_of_valid_single_digit_z_l1008_100864


namespace NUMINAMATH_GPT_widgets_production_l1008_100823

variables (A B C : ℝ)
variables (P : ℝ)

-- Conditions provided
def condition1 : Prop := 7 * A + 11 * B = 305
def condition2 : Prop := 8 * A + 22 * C = P

-- The question we need to answer
def question : Prop :=
  ∃ Q : ℝ, Q = 8 * (A + B + C)

theorem widgets_production (h1 : condition1 A B) (h2 : condition2 A C P) :
  question A B C :=
sorry

end NUMINAMATH_GPT_widgets_production_l1008_100823


namespace NUMINAMATH_GPT_unique_solution_for_a_eq_1_l1008_100876

def equation (a x : ℝ) : Prop :=
  5^(x^2 - 6 * a * x + 9 * a^2) = a * x^2 - 6 * a^2 * x + 9 * a^3 + a^2 - 6 * a + 6

theorem unique_solution_for_a_eq_1 :
  (∃! x : ℝ, equation 1 x) ∧ 
  (∀ a : ℝ, (∃! x : ℝ, equation a x) → a = 1) :=
sorry

end NUMINAMATH_GPT_unique_solution_for_a_eq_1_l1008_100876


namespace NUMINAMATH_GPT_cookies_in_jar_l1008_100853

theorem cookies_in_jar (C : ℕ) (h : C - 1 = (C + 5) / 2) : C = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cookies_in_jar_l1008_100853


namespace NUMINAMATH_GPT_combined_height_difference_is_correct_l1008_100804

-- Define the initial conditions
def uncle_height : ℕ := 72
def james_initial_height : ℕ := (2 * uncle_height) / 3
def sarah_initial_height : ℕ := (3 * james_initial_height) / 4

-- Define the growth spurts
def james_growth_spurt : ℕ := 10
def sarah_growth_spurt : ℕ := 12

-- Define their heights after growth spurts
def james_final_height : ℕ := james_initial_height + james_growth_spurt
def sarah_final_height : ℕ := sarah_initial_height + sarah_growth_spurt

-- Define the combined height of James and Sarah after growth spurts
def combined_height : ℕ := james_final_height + sarah_final_height

-- Define the combined height difference between uncle and both James and Sarah now
def combined_height_difference : ℕ := combined_height - uncle_height

-- Lean statement to prove the combined height difference
theorem combined_height_difference_is_correct : combined_height_difference = 34 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_combined_height_difference_is_correct_l1008_100804


namespace NUMINAMATH_GPT_total_pizzas_served_l1008_100852

def lunch_pizzas : ℚ := 12.5
def dinner_pizzas : ℚ := 8.25

theorem total_pizzas_served : lunch_pizzas + dinner_pizzas = 20.75 := by
  sorry

end NUMINAMATH_GPT_total_pizzas_served_l1008_100852


namespace NUMINAMATH_GPT_opposite_neg_one_half_l1008_100835

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_opposite_neg_one_half_l1008_100835


namespace NUMINAMATH_GPT_find_f_8_l1008_100834

def f (n : ℕ) : ℕ := n^2 - 3 * n + 20

theorem find_f_8 : f 8 = 60 := 
by 
sorry

end NUMINAMATH_GPT_find_f_8_l1008_100834


namespace NUMINAMATH_GPT_max_g_eq_25_l1008_100888

-- Define the function g on positive integers.
def g : ℕ → ℤ
| n => if n < 12 then n + 14 else g (n - 7)

-- Prove that the maximum value of g is 25.
theorem max_g_eq_25 : ∀ n : ℕ, 1 ≤ n → g n ≤ 25 ∧ (∃ n : ℕ, 1 ≤ n ∧ g n = 25) := by
  sorry

end NUMINAMATH_GPT_max_g_eq_25_l1008_100888


namespace NUMINAMATH_GPT_lines_skew_iff_a_ne_20_l1008_100821

variable {t u a : ℝ}
-- Definitions for the lines
def line1 (t : ℝ) (a : ℝ) := (2 + 3 * t, 3 + 4 * t, a + 5 * t)
def line2 (u : ℝ) := (3 + 6 * u, 2 + 5 * u, 1 + 2 * u)

-- Condition for lines to intersect
def lines_intersect (t u a : ℝ) :=
  2 + 3 * t = 3 + 6 * u ∧
  3 + 4 * t = 2 + 5 * u ∧
  a + 5 * t = 1 + 2 * u

-- The main theorem stating when lines are skew
theorem lines_skew_iff_a_ne_20 (a : ℝ) :
  (¬ ∃ t u : ℝ, lines_intersect t u a) ↔ a ≠ 20 := 
by 
  sorry

end NUMINAMATH_GPT_lines_skew_iff_a_ne_20_l1008_100821


namespace NUMINAMATH_GPT_distribution_of_books_l1008_100809

theorem distribution_of_books :
  let A := 2 -- number of identical art albums (type A)
  let B := 3 -- number of identical stamp albums (type B)
  let friends := 4 -- number of friends
  let total_ways := 5 -- total number of ways to distribute books 
  (A + B) = friends + 1 →
  total_ways = 5 := 
by
  intros A B friends total_ways h
  sorry

end NUMINAMATH_GPT_distribution_of_books_l1008_100809


namespace NUMINAMATH_GPT_triangle_is_obtuse_l1008_100850

-- Define the sides of the triangle with the given ratio
def a (x : ℝ) := 3 * x
def b (x : ℝ) := 4 * x
def c (x : ℝ) := 6 * x

-- The theorem statement
theorem triangle_is_obtuse (x : ℝ) (hx : 0 < x) : 
  (a x)^2 + (b x)^2 < (c x)^2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_obtuse_l1008_100850


namespace NUMINAMATH_GPT_fish_swim_eastward_l1008_100803

-- Define the conditions
variables (E : ℕ)
variable (total_fish_left : ℕ := 2870)
variable (fish_westward : ℕ := 1800)
variable (fish_north : ℕ := 500)
variable (fishwestward_not_caught : ℕ := fish_westward / 4)
variable (fishnorth_not_caught : ℕ := fish_north)
variable (fish_tobe_left_after_caught : ℕ := total_fish_left - fishwestward_not_caught - fishnorth_not_caught)

-- Define the theorem to prove
theorem fish_swim_eastward (h : 3 / 5 * E = fish_tobe_left_after_caught) : E = 3200 := 
by
  sorry

end NUMINAMATH_GPT_fish_swim_eastward_l1008_100803


namespace NUMINAMATH_GPT_range_of_a_l1008_100830

def inequality_system_has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (x + a ≥ 0) ∧ (1 - 2 * x > x - 2)

theorem range_of_a (a : ℝ) : inequality_system_has_solution a ↔ a > -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1008_100830


namespace NUMINAMATH_GPT_evaluate_expression_l1008_100890

theorem evaluate_expression (a x : ℤ) (h : x = a + 7) : x - a + 3 = 10 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1008_100890


namespace NUMINAMATH_GPT_minimum_value_expression_l1008_100856

theorem minimum_value_expression {a b c : ℤ} (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  4 * (a^2 + b^2 + c^2) - (a + b + c)^2 = 8 := 
sorry

end NUMINAMATH_GPT_minimum_value_expression_l1008_100856


namespace NUMINAMATH_GPT_tangent_ellipse_hyperbola_l1008_100891

theorem tangent_ellipse_hyperbola (m : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 → x^2 - m * (y + 3)^2 = 4) → m = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_tangent_ellipse_hyperbola_l1008_100891


namespace NUMINAMATH_GPT_value_of_double_operation_l1008_100838

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_double_operation :
  op2 (op1 10) = -10 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_double_operation_l1008_100838


namespace NUMINAMATH_GPT_walking_speed_is_4_l1008_100858

def distance : ℝ := 20
def total_time : ℝ := 3.75
def running_distance : ℝ := 10
def running_speed : ℝ := 8
def walking_distance : ℝ := 10

theorem walking_speed_is_4 (W : ℝ) 
  (H1 : running_distance + walking_distance = distance)
  (H2 : running_speed > 0)
  (H3 : walking_distance > 0)
  (H4 : W > 0)
  (H5 : walking_distance / W + running_distance / running_speed = total_time) :
  W = 4 :=
by sorry

end NUMINAMATH_GPT_walking_speed_is_4_l1008_100858


namespace NUMINAMATH_GPT_final_volume_solution_l1008_100859

variables (V2 V12 V_final : ℝ)

-- Given conditions
def V2_percent_solution (V2 : ℝ) := true
def V12_percent_solution (V12 : ℝ) := V12 = 18
def mixture_equation (V2 V12 V_final : ℝ) := 0.02 * V2 + 0.12 * V12 = 0.05 * V_final
def total_volume (V2 V12 V_final : ℝ) := V_final = V2 + V12

theorem final_volume_solution (V2 V_final : ℝ) (hV2: V2_percent_solution V2)
    (hV12 : V12_percent_solution V12) (h_mix : mixture_equation V2 V12 V_final)
    (h_total : total_volume V2 V12 V_final) : V_final = 60 :=
sorry

end NUMINAMATH_GPT_final_volume_solution_l1008_100859


namespace NUMINAMATH_GPT_gold_silver_weight_problem_l1008_100822

theorem gold_silver_weight_problem (x y : ℕ) (h1 : 9 * x = 11 * y) (h2 : (10 * y + x) - (8 * x + y) = 13) :
  9 * x = 11 * y ∧ (10 * y + x) - (8 * x + y) = 13 :=
by
  refine ⟨h1, h2⟩

end NUMINAMATH_GPT_gold_silver_weight_problem_l1008_100822


namespace NUMINAMATH_GPT_unknown_number_value_l1008_100898

theorem unknown_number_value (a x : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * x * 35 * 63) : x = 25 := by
  sorry

end NUMINAMATH_GPT_unknown_number_value_l1008_100898


namespace NUMINAMATH_GPT_solve_system_l1008_100878

-- Define the system of equations
def eq1 (x y : ℚ) : Prop := 4 * x - 3 * y = -10
def eq2 (x y : ℚ) : Prop := 6 * x + 5 * y = -13

-- Define the solution
def solution (x y : ℚ) : Prop := x = -89 / 38 ∧ y = 0.21053

-- Prove that the given solution satisfies both equations
theorem solve_system : ∃ x y : ℚ, eq1 x y ∧ eq2 x y ∧ solution x y :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1008_100878


namespace NUMINAMATH_GPT_Bethany_total_riding_hours_l1008_100865

-- Define daily riding hours
def Monday_hours : Nat := 1
def Wednesday_hours : Nat := 1
def Friday_hours : Nat := 1
def Tuesday_hours : Nat := 1 / 2
def Thursday_hours : Nat := 1 / 2
def Saturday_hours : Nat := 2

-- Define total weekly hours
def weekly_hours : Nat :=
  Monday_hours + Wednesday_hours + Friday_hours + (Tuesday_hours + Thursday_hours) + Saturday_hours

-- Definition to account for the 2-week period
def total_hours (weeks : Nat) : Nat := weeks * weekly_hours

-- Prove that Bethany rode 12 hours over 2 weeks
theorem Bethany_total_riding_hours : total_hours 2 = 12 := by
  sorry

end NUMINAMATH_GPT_Bethany_total_riding_hours_l1008_100865


namespace NUMINAMATH_GPT_percentage_increase_is_50_l1008_100863

-- Definition of the given values
def original_time : ℕ := 30
def new_time : ℕ := 45

-- Assertion stating that the percentage increase is 50%
theorem percentage_increase_is_50 :
  (new_time - original_time) * 100 / original_time = 50 := 
sorry

end NUMINAMATH_GPT_percentage_increase_is_50_l1008_100863


namespace NUMINAMATH_GPT_reflections_composition_rotation_l1008_100887

variable {α : ℝ} -- defining the angle α
variable {O : ℝ × ℝ} -- defining the point O, assuming the plane is represented as ℝ × ℝ

-- Define the lines that form the sides of the angle
variable (L1 L2 : ℝ × ℝ → Prop)

-- Assume α is the angle between L1 and L2 with O as the vertex
variable (hL1 : (L1 O))
variable (hL2 : (L2 O))

-- Assume reflections across L1 and L2
def reflect (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

theorem reflections_composition_rotation :
  ∀ A : ℝ × ℝ, (reflect (reflect A L1) L2) = sorry := 
sorry

end NUMINAMATH_GPT_reflections_composition_rotation_l1008_100887


namespace NUMINAMATH_GPT_isolating_and_counting_bacteria_process_l1008_100886

theorem isolating_and_counting_bacteria_process
  (soil_sampling : Prop)
  (spreading_dilution_on_culture_medium : Prop)
  (decompose_urea : Prop) :
  (soil_sampling ∧ spreading_dilution_on_culture_medium ∧ decompose_urea) →
  (Sample_dilution ∧ Selecting_colonies_that_can_grow ∧ Identification) :=
sorry

end NUMINAMATH_GPT_isolating_and_counting_bacteria_process_l1008_100886


namespace NUMINAMATH_GPT_part_a_part_b_l1008_100848

theorem part_a (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^a - 1)) :=
sorry

theorem part_b (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^(a + 1) - 1)) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1008_100848


namespace NUMINAMATH_GPT_greatest_odd_factors_l1008_100802

theorem greatest_odd_factors (n : ℕ) (h1 : n < 1000) (h2 : ∀ k : ℕ, k * k = n → (k < 32)) :
  n = 31 * 31 :=
by
  sorry

end NUMINAMATH_GPT_greatest_odd_factors_l1008_100802


namespace NUMINAMATH_GPT_units_digit_7_pow_5_l1008_100882

theorem units_digit_7_pow_5 : (7^5) % 10 = 7 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_5_l1008_100882


namespace NUMINAMATH_GPT_slope_interval_non_intersect_l1008_100846

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5

def Q : ℝ × ℝ := (10, 10)

theorem slope_interval_non_intersect (r s : ℝ) (h : ∀ m : ℝ,
  ¬∃ x : ℝ, parabola x = m * (x - 10) + 10 ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end NUMINAMATH_GPT_slope_interval_non_intersect_l1008_100846


namespace NUMINAMATH_GPT_grandfather_age_correct_l1008_100805

-- Let's define the conditions
def xiaowen_age : ℕ := 13
def grandfather_age : ℕ := 5 * xiaowen_age + 8

-- The statement to prove
theorem grandfather_age_correct : grandfather_age = 73 := by
  sorry

end NUMINAMATH_GPT_grandfather_age_correct_l1008_100805


namespace NUMINAMATH_GPT_joe_lowest_dropped_score_l1008_100844

theorem joe_lowest_dropped_score (A B C D : ℕ) 
  (hmean_before : (A + B + C + D) / 4 = 35)
  (hmean_after : (A + B + C) / 3 = 40)
  (hdrop : D = min A (min B (min C D))) :
  D = 20 :=
by sorry

end NUMINAMATH_GPT_joe_lowest_dropped_score_l1008_100844


namespace NUMINAMATH_GPT_sector_area_l1008_100807

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) : 
  arc_length = π / 3 ∧ central_angle = π / 6 → arc_length = central_angle * r → area = 1 / 2 * central_angle * r^2 → area = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l1008_100807


namespace NUMINAMATH_GPT_pipe_pumping_rate_l1008_100899

theorem pipe_pumping_rate (R : ℕ) (h : 5 * R + 5 * 192 = 1200) : R = 48 := by
  sorry

end NUMINAMATH_GPT_pipe_pumping_rate_l1008_100899


namespace NUMINAMATH_GPT_loss_percentage_is_30_l1008_100862

theorem loss_percentage_is_30
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 1900)
  (h2 : selling_price = 1330) :
  (cost_price - selling_price) / cost_price * 100 = 30 :=
by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_loss_percentage_is_30_l1008_100862


namespace NUMINAMATH_GPT_find_fg_minus_gf_l1008_100811

def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 5
def g (x : ℝ) : ℝ := 2 * x + 1

theorem find_fg_minus_gf (x : ℝ) : f (g x) - g (f x) = 6 * x^2 + 12 * x + 11 := 
by 
  sorry

end NUMINAMATH_GPT_find_fg_minus_gf_l1008_100811


namespace NUMINAMATH_GPT_arithmetic_seq_term_298_eq_100_l1008_100813

-- Define the arithmetic sequence
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the specific sequence given in the problem
def a_n (n : ℕ) : ℕ := arithmetic_seq 1 3 n

-- State the theorem
theorem arithmetic_seq_term_298_eq_100 : a_n 100 = 298 :=
by
  -- Proof will be filled in
  sorry

end NUMINAMATH_GPT_arithmetic_seq_term_298_eq_100_l1008_100813


namespace NUMINAMATH_GPT_cost_of_paving_l1008_100816

-- Definitions based on the given conditions
def length : ℝ := 6.5
def width : ℝ := 2.75
def rate : ℝ := 600

-- Theorem statement to prove the cost of paving
theorem cost_of_paving : length * width * rate = 10725 := by
  -- Calculation steps would go here, but we omit them with sorry
  sorry

end NUMINAMATH_GPT_cost_of_paving_l1008_100816


namespace NUMINAMATH_GPT_incorrect_proposition_l1008_100885

-- Variables and conditions
variable (p q : Prop)
variable (m x a b c : ℝ)
variable (hreal : 1 + 4 * m ≥ 0)

-- Theorem statement
theorem incorrect_proposition :
  ¬ (∀ m > 0, (∃ x : ℝ, x^2 + x - m = 0) → m > 0) :=
sorry

end NUMINAMATH_GPT_incorrect_proposition_l1008_100885


namespace NUMINAMATH_GPT_madeline_flower_count_l1008_100817

theorem madeline_flower_count 
    (r w : ℕ) 
    (b_percent : ℝ) 
    (total : ℕ) 
    (h_r : r = 4)
    (h_w : w = 2)
    (h_b_percent : b_percent = 0.40)
    (h_total : r + w + (b_percent * total) = total) : 
    total = 10 :=
by 
    sorry

end NUMINAMATH_GPT_madeline_flower_count_l1008_100817


namespace NUMINAMATH_GPT_central_angle_of_section_l1008_100871

theorem central_angle_of_section (A : ℝ) (hA : 0 < A) (prob : ℝ) (hprob : prob = 1 / 4) :
  ∃ θ : ℝ, (θ / 360) = prob :=
by
  use 90
  sorry

end NUMINAMATH_GPT_central_angle_of_section_l1008_100871


namespace NUMINAMATH_GPT_sales_tax_percentage_l1008_100874

theorem sales_tax_percentage (total_amount : ℝ) (tip_percentage : ℝ) (food_price : ℝ) (tax_percentage : ℝ) : 
  total_amount = 158.40 ∧ tip_percentage = 0.20 ∧ food_price = 120 → tax_percentage = 0.10 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sales_tax_percentage_l1008_100874


namespace NUMINAMATH_GPT_value_of_f_2011_l1008_100812

noncomputable def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 7

theorem value_of_f_2011 (a b c : ℝ) (h : f a b c (-2011) = -17) : f a b c 2011 = 31 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_f_2011_l1008_100812


namespace NUMINAMATH_GPT_inequality_for_pos_reals_l1008_100875

open Real Nat

theorem inequality_for_pos_reals
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 1/a + 1/b = 1)
  (n : ℕ) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_for_pos_reals_l1008_100875


namespace NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_same_plane_l1008_100889

-- Define a plane as a placeholder for other properties
axiom Plane : Type
-- Define Line as a placeholder for other properties
axiom Line : Type

-- Definition of what it means for a line to be perpendicular to a plane
axiom perpendicular_to_plane (l : Line) (π : Plane) : Prop

-- Definition of parallel lines
axiom parallel_lines (l1 l2 : Line) : Prop

-- Define the proof problem in Lean 4
theorem lines_parallel_if_perpendicular_to_same_plane
    (π : Plane) (l1 l2 : Line)
    (h1 : perpendicular_to_plane l1 π)
    (h2 : perpendicular_to_plane l2 π) :
    parallel_lines l1 l2 :=
sorry

end NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_same_plane_l1008_100889


namespace NUMINAMATH_GPT_percent_employed_l1008_100870

theorem percent_employed (E : ℝ) : 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30 -- 1 - percent_females
  (percent_males * E = employed_males) → E = 70 := 
by 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30
  intro h
  sorry

end NUMINAMATH_GPT_percent_employed_l1008_100870


namespace NUMINAMATH_GPT_negation_of_even_sum_l1008_100824

variables (a b : Int)

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem negation_of_even_sum (h : ¬(is_even a ∧ is_even b)) : ¬is_even (a + b) :=
sorry

end NUMINAMATH_GPT_negation_of_even_sum_l1008_100824


namespace NUMINAMATH_GPT_betty_needs_more_money_l1008_100860

-- Define the variables and conditions
def wallet_cost : ℕ := 100
def parents_gift : ℕ := 15
def grandparents_gift : ℕ := parents_gift * 2
def initial_betty_savings : ℕ := wallet_cost / 2
def total_savings : ℕ := initial_betty_savings + parents_gift + grandparents_gift

-- Prove that Betty needs 5 more dollars to buy the wallet
theorem betty_needs_more_money : total_savings + 5 = wallet_cost :=
by
  sorry

end NUMINAMATH_GPT_betty_needs_more_money_l1008_100860


namespace NUMINAMATH_GPT_sum_squares_mod_divisor_l1008_100837

-- Define the sum of the squares from 1 to 10
def sum_squares := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2)

-- Define the divisor
def divisor := 11

-- Prove that the remainder of sum_squares when divided by divisor is 0
theorem sum_squares_mod_divisor : sum_squares % divisor = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_squares_mod_divisor_l1008_100837


namespace NUMINAMATH_GPT_sum_of_three_consecutive_cubes_divisible_by_9_l1008_100843

theorem sum_of_three_consecutive_cubes_divisible_by_9 (n : ℕ) : 
  (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_cubes_divisible_by_9_l1008_100843


namespace NUMINAMATH_GPT_radius_of_circumcircle_l1008_100894

-- Definitions of sides of a triangle and its area
variables {a b c t : ℝ}

-- Condition that t is the area of a triangle with sides a, b, and c
def is_triangle_area (a b c t : ℝ) : Prop := -- Placeholder condition stating these values form a triangle
sorry

-- Statement to prove the given radius formula for the circumscribed circle
theorem radius_of_circumcircle (h : is_triangle_area a b c t) : 
  ∃ r : ℝ, r = abc / (4 * t) :=
sorry

end NUMINAMATH_GPT_radius_of_circumcircle_l1008_100894


namespace NUMINAMATH_GPT_not_quadratic_eq3_l1008_100855

-- Define the equations as functions or premises
def eq1 (x : ℝ) := 9 * x^2 = 7 * x
def eq2 (y : ℝ) := abs (y^2) = 8
def eq3 (y : ℝ) := 3 * y * (y - 1) = y * (3 * y + 1)
def eq4 (x : ℝ) := abs 2 * (x^2 + 1) = abs 10

-- Define what it means to be a quadratic equation
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x = (a * x^2 + b * x + c = 0)

-- Prove that eq3 is not a quadratic equation
theorem not_quadratic_eq3 : ¬ is_quadratic eq3 :=
sorry

end NUMINAMATH_GPT_not_quadratic_eq3_l1008_100855


namespace NUMINAMATH_GPT_patch_area_difference_l1008_100825

theorem patch_area_difference :
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  area_difference = 100 := 
by
  -- Definitions
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  -- Proof (intentionally left as sorry)
  -- Lean should be able to use the initial definitions to verify the theorem statement.
  sorry

end NUMINAMATH_GPT_patch_area_difference_l1008_100825


namespace NUMINAMATH_GPT_coeff_x3_in_expansion_l1008_100840

theorem coeff_x3_in_expansion : (Polynomial.coeff ((Polynomial.C 1 - Polynomial.C 2 * Polynomial.X)^6) 3) = -160 := 
by 
  sorry

end NUMINAMATH_GPT_coeff_x3_in_expansion_l1008_100840


namespace NUMINAMATH_GPT_units_digit_p_l1008_100895

theorem units_digit_p (p : ℕ) (h1 : p % 2 = 0) (h2 : ((p ^ 3 % 10) - (p ^ 2 % 10)) % 10 = 0) 
(h3 : (p + 4) % 10 = 0) : p % 10 = 6 :=
sorry

end NUMINAMATH_GPT_units_digit_p_l1008_100895


namespace NUMINAMATH_GPT_horizontal_asymptote_condition_l1008_100861

open Polynomial

def polynomial_deg_with_horiz_asymp (p : Polynomial ℝ) : Prop :=
  degree p ≤ 4

theorem horizontal_asymptote_condition (p : Polynomial ℝ) :
  polynomial_deg_with_horiz_asymp p :=
sorry

end NUMINAMATH_GPT_horizontal_asymptote_condition_l1008_100861


namespace NUMINAMATH_GPT_num_A_is_9_l1008_100832

-- Define the total number of animals
def total_animals : ℕ := 17

-- Define the number of animal B
def num_B : ℕ := 8

-- Define the number of animal A
def num_A : ℕ := total_animals - num_B

-- Statement to prove
theorem num_A_is_9 : num_A = 9 :=
by
  sorry

end NUMINAMATH_GPT_num_A_is_9_l1008_100832


namespace NUMINAMATH_GPT_trapezoid_area_is_correct_l1008_100879

def square_side_lengths : List ℕ := [1, 3, 5, 7]
def total_base_length : ℕ := square_side_lengths.sum
def tallest_square_height : ℕ := 7

noncomputable def trapezoid_area_between_segment_and_base : ℚ :=
  let height_at_x (x : ℚ) : ℚ := x * (7/16)
  let base_1 := 4
  let base_2 := 9
  let height_1 := height_at_x base_1
  let height_2 := height_at_x base_2
  ((height_1 + height_2) * (base_2 - base_1) / 2)

theorem trapezoid_area_is_correct :
  trapezoid_area_between_segment_and_base = 14.21875 :=
sorry

end NUMINAMATH_GPT_trapezoid_area_is_correct_l1008_100879


namespace NUMINAMATH_GPT_roots_polynomial_sum_l1008_100847

theorem roots_polynomial_sum (p q : ℂ) (hp : p^2 - 6 * p + 10 = 0) (hq : q^2 - 6 * q + 10 = 0) :
  p^4 + p^5 * q^3 + p^3 * q^5 + q^4 = 16056 := by
  sorry

end NUMINAMATH_GPT_roots_polynomial_sum_l1008_100847


namespace NUMINAMATH_GPT_geom_seq_sum_2016_2017_l1008_100845

noncomputable def geom_seq (n : ℕ) (a1 q : ℝ) : ℝ := a1 * q ^ (n - 1)

noncomputable def sum_geometric_seq (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then
  a1 * n
else
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_sum_2016_2017 :
  (a1 = 2) →
  (geom_seq 2 a1 q + geom_seq 5 a1 q = 0) →
  sum_geometric_seq a1 q 2016 + sum_geometric_seq a1 q 2017 = 2 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_2016_2017_l1008_100845


namespace NUMINAMATH_GPT_inequality_holds_for_all_xyz_in_unit_interval_l1008_100820

theorem inequality_holds_for_all_xyz_in_unit_interval :
  ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → 
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z)) :=
by
  intros x y z hx hy hz
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_xyz_in_unit_interval_l1008_100820


namespace NUMINAMATH_GPT_paper_folding_holes_l1008_100880

def folded_paper_holes (folds: Nat) (holes: Nat) : Nat :=
  match folds with
  | 0 => holes
  | n+1 => 2 * folded_paper_holes n holes

theorem paper_folding_holes : folded_paper_holes 3 1 = 8 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_paper_folding_holes_l1008_100880


namespace NUMINAMATH_GPT_number_multiplied_by_any_integer_results_in_itself_l1008_100836

theorem number_multiplied_by_any_integer_results_in_itself (N : ℤ) (h : ∀ (x : ℤ), N * x = N) : N = 0 :=
  sorry

end NUMINAMATH_GPT_number_multiplied_by_any_integer_results_in_itself_l1008_100836


namespace NUMINAMATH_GPT_correct_statement_D_l1008_100828

def is_correct_option (n : ℕ) := n = 4

theorem correct_statement_D : is_correct_option 4 :=
  sorry

end NUMINAMATH_GPT_correct_statement_D_l1008_100828


namespace NUMINAMATH_GPT_find_fraction_l1008_100868

theorem find_fraction (x y : ℕ) (h₁ : x / (y + 1) = 1 / 2) (h₂ : (x + 1) / y = 1) : x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_GPT_find_fraction_l1008_100868


namespace NUMINAMATH_GPT_molecular_weight_C4H10_l1008_100818

theorem molecular_weight_C4H10 (molecular_weight_six_moles : ℝ) (h : molecular_weight_six_moles = 390) :
  molecular_weight_six_moles / 6 = 65 :=
by
  -- proof to be filled in here
  sorry

end NUMINAMATH_GPT_molecular_weight_C4H10_l1008_100818


namespace NUMINAMATH_GPT_triangle_one_interior_angle_61_degrees_l1008_100883

theorem triangle_one_interior_angle_61_degrees
  (x : ℝ) : 
  (x + 75 + 2 * x + 25 + 3 * x - 22 = 360) → 
  (1 / 2 * (2 * x + 25) = 61 ∨ 
   1 / 2 * (3 * x - 22) = 61 ∨ 
   1 / 2 * (x + 75) = 61) :=
by
  intros h_sum
  sorry

end NUMINAMATH_GPT_triangle_one_interior_angle_61_degrees_l1008_100883


namespace NUMINAMATH_GPT_geometric_series_sum_l1008_100873

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (last_term : ℝ) 
  (h_a : a = 1) (h_r : r = -3) 
  (h_last_term : last_term = 6561) 
  (h_last_term_eq : a * r^n = last_term) : 
  a * (r^n - 1) / (r - 1) = 4921.25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1008_100873


namespace NUMINAMATH_GPT_jars_needed_l1008_100854

-- Definitions based on the given conditions
def total_cherry_tomatoes : ℕ := 56
def cherry_tomatoes_per_jar : ℕ := 8

-- Lean theorem to prove the question
theorem jars_needed (total_cherry_tomatoes cherry_tomatoes_per_jar : ℕ) (h1 : total_cherry_tomatoes = 56) (h2 : cherry_tomatoes_per_jar = 8) : (total_cherry_tomatoes / cherry_tomatoes_per_jar) = 7 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_jars_needed_l1008_100854


namespace NUMINAMATH_GPT_maximize_S_n_at_24_l1008_100815

noncomputable def a_n (n : ℕ) : ℝ := 142 + (n - 1) * (-2)
noncomputable def b_n (n : ℕ) : ℝ := 142 + (n - 1) * (-6)
noncomputable def S_n (n : ℕ) : ℝ := (n / 2.0) * (2 * 142 + (n - 1) * (-6))

theorem maximize_S_n_at_24 : ∀ (n : ℕ), S_n n ≤ S_n 24 :=
by sorry

end NUMINAMATH_GPT_maximize_S_n_at_24_l1008_100815


namespace NUMINAMATH_GPT_area_of_triangle_l1008_100839

theorem area_of_triangle :
  let A := (10, 1)
  let B := (15, 8)
  let C := (10, 8)
  ∃ (area : ℝ), 
  area = 17.5 ∧ 
  area = 1 / 2 * (abs (B.1 - C.1)) * (abs (C.2 - A.2)) :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1008_100839


namespace NUMINAMATH_GPT_surface_area_of_solid_l1008_100829

-- Define a unit cube and the number of cubes
def unitCube : Type := { faces : ℕ // faces = 6 }
def numCubes : ℕ := 10

-- Define the surface area contribution from different orientations
def surfaceAreaFacingUs (cubes : ℕ) : ℕ := 2 * cubes -- faces towards and away
def verticalSidesArea (heightCubes : ℕ) : ℕ := 2 * heightCubes -- left and right vertical sides
def horizontalSidesArea (widthCubes : ℕ) : ℕ := 2 * widthCubes -- top and bottom horizontal sides

-- Define the surface area for the given configuration of 10 cubes
def totalSurfaceArea (cubes : ℕ) (height : ℕ) (width : ℕ) : ℕ :=
  (surfaceAreaFacingUs cubes) + (verticalSidesArea height) + (horizontalSidesArea width)

-- Assumptions based on problem description
def heightCubes : ℕ := 3
def widthCubes : ℕ := 4

-- The theorem we want to prove
theorem surface_area_of_solid : totalSurfaceArea numCubes heightCubes widthCubes = 34 := by
  sorry

end NUMINAMATH_GPT_surface_area_of_solid_l1008_100829


namespace NUMINAMATH_GPT_right_triangle_area_perimeter_l1008_100827

theorem right_triangle_area_perimeter (a b : ℕ) (h₁ : a = 36) (h₂ : b = 48) : 
  (1/2) * (a * b) = 864 ∧ a + b + Nat.sqrt (a * a + b * b) = 144 := by
  sorry

end NUMINAMATH_GPT_right_triangle_area_perimeter_l1008_100827


namespace NUMINAMATH_GPT_solution_set_ineq_l1008_100800

theorem solution_set_ineq (x : ℝ) : (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_ineq_l1008_100800


namespace NUMINAMATH_GPT_abs_fraction_lt_one_l1008_100826

theorem abs_fraction_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) : 
  |(x - y) / (1 - x * y)| < 1 := 
sorry

end NUMINAMATH_GPT_abs_fraction_lt_one_l1008_100826


namespace NUMINAMATH_GPT_product_of_roots_of_cubic_polynomial_l1008_100881

theorem product_of_roots_of_cubic_polynomial :
  let a := 1
  let b := -15
  let c := 75
  let d := -50
  ∀ x : ℝ, (x^3 - 15*x^2 + 75*x - 50 = 0) →
  (x + 1) * (x + 1) * (x + 50) - 1 * (x + 50) = 0 → -d / a = 50 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_of_cubic_polynomial_l1008_100881


namespace NUMINAMATH_GPT_sqrt_nine_over_four_l1008_100842

theorem sqrt_nine_over_four (x : ℝ) : x = 3 / 2 ∨ x = - (3 / 2) ↔ x * x = 9 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_sqrt_nine_over_four_l1008_100842


namespace NUMINAMATH_GPT_technicians_count_l1008_100810

theorem technicians_count {T R : ℕ} (h1 : T + R = 12) (h2 : 2 * T + R = 18) : T = 6 :=
sorry

end NUMINAMATH_GPT_technicians_count_l1008_100810


namespace NUMINAMATH_GPT_cos_double_angle_l1008_100833

variable {α : ℝ}

theorem cos_double_angle (h1 : (Real.tan α - (1 / Real.tan α) = 3 / 2)) (h2 : (α > π / 4) ∧ (α < π / 2)) :
  Real.cos (2 * α) = -3 / 5 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_l1008_100833


namespace NUMINAMATH_GPT_number_of_operations_to_equal_l1008_100884

theorem number_of_operations_to_equal (a b : ℤ) (da db : ℤ) (initial_diff change_per_operation : ℤ) (n : ℤ) 
(h1 : a = 365) 
(h2 : b = 24) 
(h3 : da = 19) 
(h4 : db = 12) 
(h5 : initial_diff = a - b) 
(h6 : change_per_operation = da + db) 
(h7 : initial_diff = 341) 
(h8 : change_per_operation = 31) 
(h9 : initial_diff = change_per_operation * n) :
n = 11 := 
by
  sorry

end NUMINAMATH_GPT_number_of_operations_to_equal_l1008_100884


namespace NUMINAMATH_GPT_problem_l1008_100841

variable (a b c : ℝ)

def a_def : a = Real.log (1 / 2) := sorry
def b_def : b = Real.exp (1 / Real.exp 1) := sorry
def c_def : c = Real.exp (-2) := sorry

theorem problem (ha : a = Real.log (1 / 2)) 
               (hb : b = Real.exp (1 / Real.exp 1)) 
               (hc : c = Real.exp (-2)) : 
               a < c ∧ c < b := 
by
  rw [ha, hb, hc]
  sorry

end NUMINAMATH_GPT_problem_l1008_100841


namespace NUMINAMATH_GPT_percentage_cut_l1008_100819

def original_budget : ℝ := 840
def cut_amount : ℝ := 588

theorem percentage_cut : (cut_amount / original_budget) * 100 = 70 :=
by
  sorry

end NUMINAMATH_GPT_percentage_cut_l1008_100819


namespace NUMINAMATH_GPT_eventually_periodic_sequence_l1008_100893

theorem eventually_periodic_sequence
  (a : ℕ → ℕ) (h_pos : ∀ n, 0 < a n)
  (h_div : ∀ n m, (a (n + 2 * m)) ∣ (a n + a (n + m))) :
  ∃ N d, 0 < N ∧ 0 < d ∧ ∀ n, N < n → a n = a (n + d) :=
by
  sorry

end NUMINAMATH_GPT_eventually_periodic_sequence_l1008_100893


namespace NUMINAMATH_GPT_determine_a_l1008_100814

theorem determine_a :
  ∃ (a b c d : ℕ), 
  (18 ^ a) * (9 ^ (4 * a - 1)) * (27 ^ c) = (2 ^ 6) * (3 ^ b) * (7 ^ d) ∧ 
  a * c = 4 / (2 * b + d) ∧ 
  b^2 - 4 * a * c = d ∧ 
  a = 6 := 
by
  sorry

end NUMINAMATH_GPT_determine_a_l1008_100814


namespace NUMINAMATH_GPT_grace_putting_down_mulch_hours_l1008_100872

/-- Grace's earnings conditions and hours calculation in September. -/
theorem grace_putting_down_mulch_hours :
  ∃ h : ℕ, 
    6 * 63 + 11 * 9 + 9 * h = 567 ∧
    h = 10 :=
by
  sorry

end NUMINAMATH_GPT_grace_putting_down_mulch_hours_l1008_100872


namespace NUMINAMATH_GPT_john_unanswered_questions_l1008_100866

theorem john_unanswered_questions
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : z = 9 :=
sorry

end NUMINAMATH_GPT_john_unanswered_questions_l1008_100866


namespace NUMINAMATH_GPT_change_in_surface_area_zero_l1008_100897

-- Original rectangular solid dimensions
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

-- Smaller prism dimensions
structure SmallerPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Conditions
def originalSolid : RectangularSolid := { length := 4, width := 3, height := 2 }
def removedPrism : SmallerPrism := { length := 1, width := 1, height := 2 }

-- Surface area calculation function
def surface_area (solid : RectangularSolid) : ℝ := 
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

-- Calculate the change in surface area
theorem change_in_surface_area_zero :
  let original_surface_area := surface_area originalSolid
  let removed_surface_area := (removedPrism.length * removedPrism.height)
  let new_exposed_area := (removedPrism.length * removedPrism.height)
  (original_surface_area - removed_surface_area + new_exposed_area) = original_surface_area :=
by
  sorry

end NUMINAMATH_GPT_change_in_surface_area_zero_l1008_100897


namespace NUMINAMATH_GPT_hyperbola_y_relation_l1008_100869

theorem hyperbola_y_relation {k y₁ y₂ : ℝ} 
  (A_on_hyperbola : y₁ = k / 2) 
  (B_on_hyperbola : y₂ = k / 3) 
  (k_positive : 0 < k) : 
  y₁ > y₂ := 
sorry

end NUMINAMATH_GPT_hyperbola_y_relation_l1008_100869


namespace NUMINAMATH_GPT_distinct_remainders_l1008_100806

theorem distinct_remainders (n : ℕ) (hn : 0 < n) : 
  ∀ (i j : ℕ), (i < n) → (j < n) → (2 * i + 1 ≠ 2 * j + 1) → 
  ((2 * i + 1) ^ (2 * i + 1) % 2^n ≠ (2 * j + 1) ^ (2 * j + 1) % 2^n) :=
by
  sorry

end NUMINAMATH_GPT_distinct_remainders_l1008_100806


namespace NUMINAMATH_GPT_find_x_plus_y_l1008_100892

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2023) 
                           (h2 : x + 2023 * Real.sin y = 2022) 
                           (h3 : (Real.pi / 2) ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2023 + Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l1008_100892


namespace NUMINAMATH_GPT_incorrect_average_calculated_initially_l1008_100877

theorem incorrect_average_calculated_initially 
    (S : ℕ) 
    (h1 : (S + 75) / 10 = 51) 
    (h2 : (S + 25) = a) 
    : a / 10 = 46 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_average_calculated_initially_l1008_100877
