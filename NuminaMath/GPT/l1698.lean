import Mathlib

namespace find_number_l1698_169822

theorem find_number (x N : ℕ) (h₁ : x = 32) (h₂ : N - (23 - (15 - x)) = (12 * 2 / 1 / 2)) : N = 88 :=
sorry

end find_number_l1698_169822


namespace intercepts_sum_eq_seven_l1698_169833

theorem intercepts_sum_eq_seven :
    (∃ a b c, (∀ y, (3 * y^2 - 9 * y + 4 = a) → y = 0) ∧ 
              (∀ y, (3 * y^2 - 9 * y + 4 = 0) → (y = b ∨ y = c)) ∧ 
              (a + b + c = 7)) := 
sorry

end intercepts_sum_eq_seven_l1698_169833


namespace chewbacca_gum_pieces_l1698_169861

theorem chewbacca_gum_pieces (y : ℚ)
  (h1 : ∀ x : ℚ, x ≠ 0 → (15 - y) = 15 * (25 + 2 * y) / 25) :
  y = 5 / 2 :=
by
  sorry

end chewbacca_gum_pieces_l1698_169861


namespace prime_gt_3_divides_exp_l1698_169847

theorem prime_gt_3_divides_exp (p : ℕ) (hprime : Nat.Prime p) (hgt3 : p > 3) :
  42 * p ∣ 3^p - 2^p - 1 :=
sorry

end prime_gt_3_divides_exp_l1698_169847


namespace segments_divide_ratio_3_to_1_l1698_169839

-- Define points and segments
structure Point :=
  (x : ℝ) (y : ℝ)

structure Segment :=
  (A B : Point)

-- Define T-shaped figure consisting of 22 unit squares
noncomputable def T_shaped_figure : ℕ := 22

-- Define line p passing through point V
structure Line :=
  (p : Point → Point)
  (passes_through : Point)

-- Define equal areas condition
def equal_areas (white_area gray_area : ℝ) : Prop := 
  white_area = gray_area

-- Define the problem
theorem segments_divide_ratio_3_to_1
  (AB : Segment)
  (V : Point)
  (white_area gray_area : ℝ)
  (p : Line)
  (h1 : equal_areas white_area gray_area)
  (h2 : T_shaped_figure = 22)
  (h3 : p.passes_through = V) :
  ∃ (C : Point), (p.p AB.A = C) ∧ ((abs (AB.A.x - C.x)) / (abs (C.x - AB.B.x))) = 3 :=
sorry

end segments_divide_ratio_3_to_1_l1698_169839


namespace find_principal_amount_l1698_169806

theorem find_principal_amount
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (h1 : SI = 4034.25)
  (h2 : R = 9)
  (h3 : T = 5) :
  P = 8965 :=
by
  sorry

end find_principal_amount_l1698_169806


namespace inequality_relationship_cannot_be_established_l1698_169860

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_relationship_cannot_be_established :
  ¬ (1 / (a - b) > 1 / a) :=
by sorry

end inequality_relationship_cannot_be_established_l1698_169860


namespace bobs_income_after_changes_l1698_169895

variable (initial_salary : ℝ) (february_increase_rate : ℝ) (march_reduction_rate : ℝ)

def february_salary (initial_salary : ℝ) (increase_rate : ℝ) : ℝ :=
  initial_salary * (1 + increase_rate)

def march_salary (february_salary : ℝ) (reduction_rate : ℝ) : ℝ :=
  february_salary * (1 - reduction_rate)

theorem bobs_income_after_changes (h1 : initial_salary = 2750)
  (h2 : february_increase_rate = 0.15)
  (h3 : march_reduction_rate = 0.10) :
  march_salary (february_salary initial_salary february_increase_rate) march_reduction_rate = 2846.25 := 
sorry

end bobs_income_after_changes_l1698_169895


namespace correct_average_of_15_numbers_l1698_169892

theorem correct_average_of_15_numbers
  (initial_average : ℝ)
  (num_numbers : ℕ)
  (incorrect1 incorrect2 correct1 correct2 : ℝ)
  (initial_average_eq : initial_average = 37)
  (num_numbers_eq : num_numbers = 15)
  (incorrect1_eq : incorrect1 = 52)
  (incorrect2_eq : incorrect2 = 39)
  (correct1_eq : correct1 = 64)
  (correct2_eq : correct2 = 27) :
  (initial_average * num_numbers - incorrect1 - incorrect2 + correct1 + correct2) / num_numbers = 37 :=
by
  rw [initial_average_eq, num_numbers_eq, incorrect1_eq, incorrect2_eq, correct1_eq, correct2_eq]
  sorry

end correct_average_of_15_numbers_l1698_169892


namespace find_salary_J_l1698_169826

variables (J F M A May : ℝ)

def avg_salary_J_F_M_A (J F M A : ℝ) : Prop :=
  (J + F + M + A) / 4 = 8000

def avg_salary_F_M_A_May (F M A May : ℝ) : Prop :=
  (F + M + A + May) / 4 = 8700

def salary_May (May : ℝ) : Prop :=
  May = 6500

theorem find_salary_J (h1 : avg_salary_J_F_M_A J F M A) (h2 : avg_salary_F_M_A_May F M A May) (h3 : salary_May May) :
  J = 3700 :=
sorry

end find_salary_J_l1698_169826


namespace problem_solution_l1698_169859

theorem problem_solution (x : ℝ) (h : x - 29 = 63) : (x - 47 = 45) :=
by
  sorry

end problem_solution_l1698_169859


namespace inequality_proof_l1698_169838

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a ^ 2 + 1)) + (b / Real.sqrt (b ^ 2 + 1)) + (c / Real.sqrt (c ^ 2 + 1)) ≤ (3 / 2) :=
by
  sorry

end inequality_proof_l1698_169838


namespace masha_lives_on_seventh_floor_l1698_169835

/-- Masha lives in apartment No. 290, which is in the 4th entrance of a 17-story building.
The number of apartments is the same in all entrances of the building on all 17 floors; apartment numbers start from 1.
We need to prove that Masha lives on the 7th floor. -/
theorem masha_lives_on_seventh_floor 
  (n_apartments_per_floor : ℕ) 
  (total_floors : ℕ := 17) 
  (entrances : ℕ := 4) 
  (masha_apartment : ℕ := 290) 
  (start_apartment : ℕ := 1) 
  (h1 : (masha_apartment - start_apartment + 1) > 0) 
  (h2 : masha_apartment ≤ entrances * total_floors * n_apartments_per_floor)
  (h4 : masha_apartment > (entrances - 1) * total_floors * n_apartments_per_floor)  
   : ((masha_apartment - ((entrances - 1) * total_floors * n_apartments_per_floor) - 1) / n_apartments_per_floor) + 1 = 7 := 
by
  sorry

end masha_lives_on_seventh_floor_l1698_169835


namespace linda_needs_additional_batches_l1698_169857

theorem linda_needs_additional_batches:
  let classmates := 24
  let cookies_per_classmate := 10
  let dozen := 12
  let cookies_per_batch := 4 * dozen
  let cookies_needed := classmates * cookies_per_classmate
  let chocolate_chip_batches := 2
  let oatmeal_raisin_batches := 1
  let cookies_made := (chocolate_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let remaining_cookies := cookies_needed - cookies_made
  let additional_batches := remaining_cookies / cookies_per_batch
  additional_batches = 2 :=
by
  sorry

end linda_needs_additional_batches_l1698_169857


namespace hyperbola_sum_l1698_169821

noncomputable def h : ℝ := -3
noncomputable def k : ℝ := 1
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 50
noncomputable def b : ℝ := Real.sqrt (c ^ 2 - a ^ 2)

theorem hyperbola_sum :
  h + k + a + b = 2 + Real.sqrt 34 := by
  sorry

end hyperbola_sum_l1698_169821


namespace Tammy_earnings_3_weeks_l1698_169863

theorem Tammy_earnings_3_weeks
  (trees : ℕ)
  (oranges_per_tree_per_day : ℕ)
  (oranges_per_pack : ℕ)
  (price_per_pack : ℕ)
  (weeks : ℕ) :
  trees = 10 →
  oranges_per_tree_per_day = 12 →
  oranges_per_pack = 6 →
  price_per_pack = 2 →
  weeks = 3 →
  (trees * oranges_per_tree_per_day * weeks * 7) / oranges_per_pack * price_per_pack = 840 :=
by
  intro ht ht12 h6 h2 h3
  -- proof to be filled in here
  sorry

end Tammy_earnings_3_weeks_l1698_169863


namespace arithmetic_progression_five_numbers_arithmetic_progression_four_numbers_l1698_169898

-- Statement for Problem 1: Number of ways to draw five numbers forming an arithmetic progression
theorem arithmetic_progression_five_numbers :
  ∃ (N : ℕ), N = 968 :=
  sorry

-- Statement for Problem 2: Number of ways to draw four numbers forming an arithmetic progression with a fifth number being arbitrary
theorem arithmetic_progression_four_numbers :
  ∃ (N : ℕ), N = 111262 :=
  sorry

end arithmetic_progression_five_numbers_arithmetic_progression_four_numbers_l1698_169898


namespace tim_weekly_earnings_l1698_169883

-- Definitions based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- The theorem that we need to prove
theorem tim_weekly_earnings :
  (tasks_per_day * pay_per_task * days_per_week : ℝ) = 720 :=
by
  sorry -- Skipping the proof

end tim_weekly_earnings_l1698_169883


namespace total_wait_time_difference_l1698_169858

theorem total_wait_time_difference :
  let kids_swings := 6
  let kids_slide := 4 * kids_swings
  let wait_time_swings := [210, 420, 840] -- in seconds
  let total_wait_time_swings := wait_time_swings.sum
  let wait_time_slide := [45, 90, 180] -- in seconds
  let total_wait_time_slide := wait_time_slide.sum
  let total_wait_time_all_kids_swings := kids_swings * total_wait_time_swings
  let total_wait_time_all_kids_slide := kids_slide * total_wait_time_slide
  let difference := total_wait_time_all_kids_swings - total_wait_time_all_kids_slide
  difference = 1260 := sorry

end total_wait_time_difference_l1698_169858


namespace charlotte_age_l1698_169805

theorem charlotte_age : 
  ∀ (B C E : ℝ), 
    (B = 4 * C) → 
    (E = C + 5) → 
    (B = E) → 
    C = 5 / 3 :=
by
  intros B C E h1 h2 h3
  /- start of the proof -/
  sorry

end charlotte_age_l1698_169805


namespace weight_of_new_person_l1698_169845

theorem weight_of_new_person {avg_increase : ℝ} (n : ℕ) (p : ℝ) (w : ℝ) (h : n = 8) (h1 : avg_increase = 2.5) (h2 : w = 67):
  p = 87 :=
by
  sorry

end weight_of_new_person_l1698_169845


namespace abs_diff_eq_0_5_l1698_169807

noncomputable def x : ℝ := 3.7
noncomputable def y : ℝ := 4.2

theorem abs_diff_eq_0_5 (hx : ⌊x⌋ + (y - ⌊y⌋) = 3.2) (hy : (x - ⌊x⌋) + ⌊y⌋ = 4.7) :
  |x - y| = 0.5 :=
by
  sorry

end abs_diff_eq_0_5_l1698_169807


namespace clock_hands_angle_seventy_degrees_l1698_169866

theorem clock_hands_angle_seventy_degrees (t : ℝ) (h : t ≥ 0 ∧ t ≤ 60):
    let hour_angle := 210 + 30 * (t / 60)
    let minute_angle := 360 * (t / 60)
    let angle := abs (hour_angle - minute_angle)
    (angle = 70 ∨ angle = 290) ↔ (t = 25 ∨ t = 52) :=
by apply sorry

end clock_hands_angle_seventy_degrees_l1698_169866


namespace problem_l1698_169828

-- Definitions according to the conditions
def red_balls : ℕ := 1
def black_balls (n : ℕ) : ℕ := n
def total_balls (n : ℕ) : ℕ := red_balls + black_balls n
noncomputable def probability_red (n : ℕ) : ℚ := (red_balls : ℚ) / (total_balls n : ℚ)
noncomputable def variance (n : ℕ) : ℚ := (black_balls n : ℚ) / (total_balls n : ℚ)^2

-- The theorem we want to prove
theorem problem (n : ℕ) (h : 0 < n) : 
  (∀ m : ℕ, n < m → probability_red m < probability_red n) ∧ 
  (∀ m : ℕ, n < m → variance m < variance n) :=
sorry

end problem_l1698_169828


namespace expected_carrot_yield_l1698_169893

-- Condition definitions
def num_steps_width : ℕ := 16
def num_steps_length : ℕ := 22
def step_length : ℝ := 1.75
def avg_yield_per_sqft : ℝ := 0.75

-- Theorem statement
theorem expected_carrot_yield : 
  (num_steps_width * step_length) * (num_steps_length * step_length) * avg_yield_per_sqft = 808.5 :=
by
  sorry

end expected_carrot_yield_l1698_169893


namespace gaussian_solutions_count_l1698_169803

noncomputable def solve_gaussian (x : ℝ) : ℕ :=
  if h : x^2 = 2 * (⌊x⌋ : ℝ) + 1 then 
    1 
  else
    0

theorem gaussian_solutions_count :
  ∀ x : ℝ, solve_gaussian x = 2 :=
sorry

end gaussian_solutions_count_l1698_169803


namespace non_neg_int_solutions_l1698_169855

theorem non_neg_int_solutions (n : ℕ) (a b : ℤ) :
  n^2 = a + b ∧ n^3 = a^2 + b^2 → n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end non_neg_int_solutions_l1698_169855


namespace pumps_fill_time_l1698_169800

def fill_time {X Y Z : ℝ} (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : Prop :=
  1 / (X + Y + Z) = 36 / 13

theorem pumps_fill_time (X Y Z : ℝ) (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : 
  1 / (X + Y + Z) = 36 / 13 :=
by
  sorry

end pumps_fill_time_l1698_169800


namespace puppies_left_l1698_169877

namespace AlyssaPuppies

def initPuppies : ℕ := 12
def givenAway : ℕ := 7
def remainingPuppies : ℕ := 5

theorem puppies_left (initPuppies givenAway remainingPuppies : ℕ) : 
  initPuppies - givenAway = remainingPuppies :=
by
  sorry

end AlyssaPuppies

end puppies_left_l1698_169877


namespace remainder_sum_mod9_l1698_169848

theorem remainder_sum_mod9 :
  ((2469 + 2470 + 2471 + 2472 + 2473 + 2474) % 9) = 6 := 
by 
  sorry

end remainder_sum_mod9_l1698_169848


namespace eval_expr1_eval_expr2_l1698_169894

theorem eval_expr1 : (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
by
  -- proof goes here
  sorry

theorem eval_expr2 : (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) / (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180))) = Real.sqrt 2 :=
by
  -- proof goes here
  sorry

end eval_expr1_eval_expr2_l1698_169894


namespace product_xyz_l1698_169817

theorem product_xyz {x y z a b c : ℝ} 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = b^2) 
  (h3 : x^3 + y^3 + z^3 = c^3) : 
  x * y * z = (a^3 - 3 * a * b^2 + 2 * c^3) / 6 :=
by
  sorry

end product_xyz_l1698_169817


namespace garden_area_difference_l1698_169856
-- Import the entire Mathlib

-- Lean Statement
theorem garden_area_difference :
  let length_Alice := 15
  let width_Alice := 30
  let length_Bob := 18
  let width_Bob := 28
  let area_Alice := length_Alice * width_Alice
  let area_Bob := length_Bob * width_Bob
  let difference := area_Bob - area_Alice
  difference = 54 :=
by
  sorry

end garden_area_difference_l1698_169856


namespace bill_sun_vs_sat_l1698_169830

theorem bill_sun_vs_sat (B_Sat B_Sun J_Sun : ℕ) 
  (h1 : B_Sun = 6)
  (h2 : J_Sun = 2 * B_Sun)
  (h3 : B_Sat + B_Sun + J_Sun = 20) : 
  B_Sun - B_Sat = 4 :=
by
  sorry

end bill_sun_vs_sat_l1698_169830


namespace problem1_problem2_l1698_169829

section Problems

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 - a * x + 1

-- Problem 1: Tangent line problem for a = 1
def tangent_line_eqn (x : ℝ) : Prop :=
  let a := 1
  let f := f a
  (∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b)

-- Problem 2: Minimum value problem
def min_value_condition (a : ℝ) : Prop :=
  f a (1 / 4) = (11 / 12)

theorem problem1 : tangent_line_eqn 0 :=
  sorry

theorem problem2 : min_value_condition (1 / 4) :=
  sorry

end Problems

end problem1_problem2_l1698_169829


namespace trigonometric_simplification_l1698_169862

open Real

theorem trigonometric_simplification (α : ℝ) :
  (3.4113 * sin α * cos (3 * α) + 9 * sin α * cos α - sin (3 * α) * cos (3 * α) - 3 * sin (3 * α) * cos α) = 
  2 * sin (2 * α)^3 :=
by
  -- Placeholder for the proof
  sorry

end trigonometric_simplification_l1698_169862


namespace find_function_f_l1698_169873

-- The function f maps positive integers to positive integers
def f : ℕ+ → ℕ+ := sorry

-- The statement to be proved
theorem find_function_f (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, (f m)^2 + f n ∣ (m^2 + n)^2) : ∀ n : ℕ+, f n = n :=
sorry

end find_function_f_l1698_169873


namespace tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l1698_169850

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem tangent_line_at_x_equals_1 (a : ℝ) (x : ℝ) (h₀ : a = 2) (h₁ : x = 1) : 
  3 * x - (f a 1) - 1 = 0 := 
sorry

theorem monotonic_intervals (a x : ℝ) (h₀ : x > 0) :
  ((a >= 0 ∧ ∀ (x : ℝ), x > 0 → (f a x) > (f a (x - 1))) ∨ 
  (a < 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < -1/a → (f a x) > (f a (x - 1)) ∧ ∀ (x : ℝ), x > -1/a → (f a x) < (f a (x - 1)))) :=
sorry

theorem range_of_a (a x : ℝ) (h₀ : 0 < x) (h₁ : f a x < 2) : a < -1 / Real.exp (3) :=
sorry

end tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l1698_169850


namespace inequality_positive_real_numbers_l1698_169882

theorem inequality_positive_real_numbers
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a * b + b * c + c * a = 1) :
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ (3 / 2) :=
  sorry

end inequality_positive_real_numbers_l1698_169882


namespace slices_per_pack_l1698_169811

theorem slices_per_pack (sandwiches : ℕ) (slices_per_sandwich : ℕ) (packs_of_bread : ℕ) (total_slices : ℕ) 
  (h1 : sandwiches = 8) (h2 : slices_per_sandwich = 2) (h3 : packs_of_bread = 4) : 
  total_slices = 4 :=
by
  sorry

end slices_per_pack_l1698_169811


namespace initial_cards_l1698_169852

theorem initial_cards (taken left initial : ℕ) (h1 : taken = 59) (h2 : left = 17) (h3 : initial = left + taken) : initial = 76 :=
by
  sorry

end initial_cards_l1698_169852


namespace fraction_equation_solution_l1698_169824

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) : 
  (1 / (x - 2) = 3 / x) → x = 3 := 
by 
  sorry

end fraction_equation_solution_l1698_169824


namespace simplest_form_fraction_l1698_169820

theorem simplest_form_fraction 
  (m n a : ℤ) (h_f1 : (2 * m) / (10 * m * n) = 1 / (5 * n))
  (h_f2 : (m^2 - n^2) / (m + n) = (m - n))
  (h_f3 : (2 * a) / (a^2) = 2 / a) : 
  ∀ (f : ℤ), f = (m^2 + n^2) / (m + n) → 
    (∀ (k : ℤ), k ≠ 1 → (m^2 + n^2) / (m + n) ≠ k * f) :=
by
  intros f h_eq k h_kneq1
  sorry

end simplest_form_fraction_l1698_169820


namespace point_in_second_quadrant_l1698_169878

theorem point_in_second_quadrant (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so its x-coordinate is negative
  (h2 : 0 < P.2) -- Point P is in the second quadrant, so its y-coordinate is positive
  (h3 : |P.2| = 3) -- The distance from P to the x-axis is 3
  (h4 : |P.1| = 4) -- The distance from P to the y-axis is 4
  : P = (-4, 3) := 
  sorry

end point_in_second_quadrant_l1698_169878


namespace proof_ab_lt_1_l1698_169802

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem proof_ab_lt_1 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a > f b) : a * b < 1 :=
by
  -- Sorry to skip the proof
  sorry

end proof_ab_lt_1_l1698_169802


namespace no_prime_pair_summing_to_53_l1698_169879

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end no_prime_pair_summing_to_53_l1698_169879


namespace route_B_is_faster_by_7_5_minutes_l1698_169814

def distance_A := 10  -- miles
def normal_speed_A := 30  -- mph
def construction_distance_A := 2  -- miles
def construction_speed_A := 15  -- mph
def distance_B := 8  -- miles
def normal_speed_B := 40  -- mph
def school_zone_distance_B := 1  -- miles
def school_zone_speed_B := 10  -- mph

noncomputable def time_for_normal_speed_A : ℝ := (distance_A - construction_distance_A) / normal_speed_A * 60  -- minutes
noncomputable def time_for_construction_A : ℝ := construction_distance_A / construction_speed_A * 60  -- minutes
noncomputable def total_time_A : ℝ := time_for_normal_speed_A + time_for_construction_A

noncomputable def time_for_normal_speed_B : ℝ := (distance_B - school_zone_distance_B) / normal_speed_B * 60  -- minutes
noncomputable def time_for_school_zone_B : ℝ := school_zone_distance_B / school_zone_speed_B * 60  -- minutes
noncomputable def total_time_B : ℝ := time_for_normal_speed_B + time_for_school_zone_B

theorem route_B_is_faster_by_7_5_minutes : total_time_B + 7.5 = total_time_A := by
  sorry

end route_B_is_faster_by_7_5_minutes_l1698_169814


namespace range_of_m_l1698_169812

noncomputable def problem_statement
  (x y m : ℝ) : Prop :=
  (x - 2 * y + 5 ≥ 0) ∧
  (3 - x ≥ 0) ∧
  (x + y ≥ 0) ∧
  (m > 0)

theorem range_of_m (x y m : ℝ) :
  problem_statement x y m →
  ((∀ x y, problem_statement x y m → x^2 + y^2 ≤ m^2) ↔ m ≥ 3 * Real.sqrt 2) :=
by 
  intro h
  sorry

end range_of_m_l1698_169812


namespace train_length_l1698_169841

theorem train_length (x : ℕ) (h1 : (310 + x) / 18 = x / 8) : x = 248 :=
  sorry

end train_length_l1698_169841


namespace smallest_product_of_non_factors_l1698_169813

theorem smallest_product_of_non_factors (a b : ℕ) (h_a : a ∣ 48) (h_b : b ∣ 48) (h_distinct : a ≠ b) (h_prod_non_factor : ¬ (a * b ∣ 48)) : a * b = 18 :=
sorry

end smallest_product_of_non_factors_l1698_169813


namespace ak_not_perfect_square_l1698_169872

theorem ak_not_perfect_square (a b : ℕ → ℤ)
  (h1 : ∀ k, b k = a k + 9)
  (h2 : ∀ k, a (k + 1) = 8 * b k + 8)
  (h3 : ∃ k1 k2, a k1 = 1988 ∧ b k2 = 1988) :
  ∀ k, ¬ ∃ n, a k = n * n :=
by
  sorry

end ak_not_perfect_square_l1698_169872


namespace max_d_value_l1698_169827

theorem max_d_value : ∀ (d e : ℕ), (d < 10) → (e < 10) → (5 * 10^5 + d * 10^4 + 5 * 10^3 + 2 * 10^2 + 2 * 10 + e ≡ 0 [MOD 22]) → (e % 2 = 0) → (d + e = 10) → d ≤ 8 :=
by
  intros d e h1 h2 h3 h4 h5
  sorry

end max_d_value_l1698_169827


namespace product_of_consecutive_even_numbers_divisible_by_8_l1698_169831

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 
  8 ∣ (2 * n) * (2 * n + 2) :=
by sorry

end product_of_consecutive_even_numbers_divisible_by_8_l1698_169831


namespace value_of_expression_l1698_169880

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 5 = 23 :=
by
  -- proof goes here
  sorry

end value_of_expression_l1698_169880


namespace expansion_of_binomials_l1698_169834

theorem expansion_of_binomials (a : ℝ) : (a + 2) * (a - 3) = a^2 - a - 6 :=
  sorry

end expansion_of_binomials_l1698_169834


namespace parabola_intersection_probability_correct_l1698_169875

noncomputable def parabola_intersection_probability : ℚ := sorry

theorem parabola_intersection_probability_correct :
  parabola_intersection_probability = 209 / 216 := sorry

end parabola_intersection_probability_correct_l1698_169875


namespace cube_roll_sums_l1698_169897

def opposite_faces_sum_to_seven (a b : ℕ) : Prop := a + b = 7

def valid_cube_faces : Prop := 
  opposite_faces_sum_to_seven 1 6 ∧
  opposite_faces_sum_to_seven 2 5 ∧
  opposite_faces_sum_to_seven 3 4

def max_min_sums : ℕ × ℕ := (342, 351)

theorem cube_roll_sums (faces_sum_seven : valid_cube_faces) : 
  ∃ cube_sums : ℕ × ℕ, cube_sums = max_min_sums := sorry

end cube_roll_sums_l1698_169897


namespace width_of_hall_l1698_169815

variable (L W H : ℕ) -- Length, Width, Height of the hall
variable (expenditure cost : ℕ) -- Expenditure and cost per square meter

-- Given conditions
def hall_length : L = 20 := by sorry
def hall_height : H = 5 := by sorry
def total_expenditure : expenditure = 28500 := by sorry
def cost_per_sq_meter : cost = 30 := by sorry

-- Derived value
def total_area_to_cover (W : ℕ) : ℕ :=
  (2 * (L * W) + 2 * (L * H) + 2 * (W * H))

theorem width_of_hall (W : ℕ) (h: total_area_to_cover L W H * cost = expenditure) : W = 15 := by
  sorry

end width_of_hall_l1698_169815


namespace product_third_fourth_term_l1698_169801

theorem product_third_fourth_term (a d : ℝ) : 
  (a + 7 * d = 20) → (d = 2) → 
  ( (a + 2 * d) * (a + 3 * d) = 120 ) := 
by 
  intros h1 h2
  sorry

end product_third_fourth_term_l1698_169801


namespace complementary_angle_difference_l1698_169864

theorem complementary_angle_difference (x : ℝ) (h : 3 * x + 5 * x = 90) : 
    abs ((5 * x) - (3 * x)) = 22.5 :=
by
  -- placeholder proof
  sorry

end complementary_angle_difference_l1698_169864


namespace smallest_n_fact_expr_l1698_169818

theorem smallest_n_fact_expr : ∃ n : ℕ, (∀ m : ℕ, m = 6 → n! = (n - 4) * (n - 3) * (n - 2) * (n - 1) * n * (n + 1)) ∧ n = 23 := by
  sorry

end smallest_n_fact_expr_l1698_169818


namespace square_root_problem_l1698_169865

theorem square_root_problem
  (x : ℤ) (y : ℤ)
  (hx : x = Nat.sqrt 16)
  (hy : y^2 = 9) :
  x^2 + y^2 + x - 2 = 27 := by
  sorry

end square_root_problem_l1698_169865


namespace neg_prop1_true_neg_prop2_false_l1698_169837

-- Proposition 1: The logarithm of a positive number is always positive
def prop1 : Prop := ∀ x : ℝ, x > 0 → Real.log x > 0

-- Negation of Proposition 1: There exists a positive number whose logarithm is not positive
def neg_prop1 : Prop := ∃ x : ℝ, x > 0 ∧ Real.log x ≤ 0

-- Proposition 2: For all x in the set of integers Z, the last digit of x^2 is not 3
def prop2 : Prop := ∀ x : ℤ, (x * x % 10 ≠ 3)

-- Negation of Proposition 2: There exists an x in the set of integers Z such that the last digit of x^2 is 3
def neg_prop2 : Prop := ∃ x : ℤ, (x * x % 10 = 3)

-- Proof that the negation of Proposition 1 is true
theorem neg_prop1_true : neg_prop1 := 
  by sorry

-- Proof that the negation of Proposition 2 is false
theorem neg_prop2_false : ¬ neg_prop2 := 
  by sorry

end neg_prop1_true_neg_prop2_false_l1698_169837


namespace angle_is_20_l1698_169810

theorem angle_is_20 (x : ℝ) (h : 180 - x = 2 * (90 - x) + 20) : x = 20 :=
by
  sorry

end angle_is_20_l1698_169810


namespace near_square_qoutient_l1698_169889

def is_near_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem near_square_qoutient (n : ℕ) (hn : is_near_square n) : 
  ∃ a b : ℕ, is_near_square a ∧ is_near_square b ∧ n = a / b := 
sorry

end near_square_qoutient_l1698_169889


namespace train_length_is_400_l1698_169840

-- Conditions from a)
def train_speed_kmph : ℕ := 180
def crossing_time_sec : ℕ := 8

-- The corresponding length in meters
def length_of_train : ℕ := 400

-- The problem statement to prove
theorem train_length_is_400 :
  (train_speed_kmph * 1000 / 3600) * crossing_time_sec = length_of_train := by
  -- Proof is skipped as per the requirement
  sorry

end train_length_is_400_l1698_169840


namespace smallest_egg_count_l1698_169870

theorem smallest_egg_count : ∃ n : ℕ, n > 100 ∧ n % 12 = 10 ∧ n = 106 :=
by {
  sorry
}

end smallest_egg_count_l1698_169870


namespace oil_spent_amount_l1698_169876

theorem oil_spent_amount :
  ∀ (P R M : ℝ), R = 25 → P = (R / 0.75) → ((M / R) - (M / P) = 5) → M = 500 :=
by
  intros P R M hR hP hOil
  sorry

end oil_spent_amount_l1698_169876


namespace factor_expression_l1698_169854

variable (b : ℤ)

theorem factor_expression : 280 * b^2 + 56 * b = 56 * b * (5 * b + 1) :=
by
  sorry

end factor_expression_l1698_169854


namespace scarves_per_yarn_correct_l1698_169881

def scarves_per_yarn (total_yarns total_scarves : ℕ) : ℕ :=
  total_scarves / total_yarns

theorem scarves_per_yarn_correct :
  scarves_per_yarn (2 + 6 + 4) 36 = 3 :=
by
  sorry

end scarves_per_yarn_correct_l1698_169881


namespace range_of_a_l1698_169867

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a^2 ≤ 0 → false) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end range_of_a_l1698_169867


namespace weight_of_replaced_student_l1698_169842

-- Define the conditions as hypotheses
variable (W : ℝ)
variable (h : W - 46 = 40)

-- Prove that W = 86
theorem weight_of_replaced_student : W = 86 :=
by
  -- We should conclude the proof; for now, we leave a placeholder
  sorry

end weight_of_replaced_student_l1698_169842


namespace cash_calculation_l1698_169887

theorem cash_calculation 
  (value_gold_coin : ℕ) (value_silver_coin : ℕ) 
  (num_gold_coins : ℕ) (num_silver_coins : ℕ) 
  (total_money : ℕ) : 
  value_gold_coin = 50 → 
  value_silver_coin = 25 → 
  num_gold_coins = 3 → 
  num_silver_coins = 5 → 
  total_money = 305 → 
  (total_money - (num_gold_coins * value_gold_coin + num_silver_coins * value_silver_coin) = 30) := 
by
  intros h1 h2 h3 h4 h5
  sorry

end cash_calculation_l1698_169887


namespace quotient_product_larger_integer_l1698_169869

theorem quotient_product_larger_integer
  (x y : ℕ)
  (h1 : y / x = 7 / 3)
  (h2 : x * y = 189)
  : y = 21 := 
sorry

end quotient_product_larger_integer_l1698_169869


namespace value_of_f_l1698_169836

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom f_has_derivative : ∀ x, deriv f x = f' x
axiom f_equation : ∀ x, f x = 3 * x^2 + 2 * x * (f' 1)

-- Proof goal
theorem value_of_f'_at_3 : f' 3 = 6 := by
  sorry

end value_of_f_l1698_169836


namespace find_number_of_valid_polynomials_l1698_169843

noncomputable def number_of_polynomials_meeting_constraints : Nat :=
  sorry

theorem find_number_of_valid_polynomials : number_of_polynomials_meeting_constraints = 11 :=
  sorry

end find_number_of_valid_polynomials_l1698_169843


namespace tire_swap_distance_l1698_169899

theorem tire_swap_distance : ∃ x : ℕ, 
  (1 - x / 11000) * 9000 = (1 - x / 9000) * 11000 ∧ x = 4950 := 
by
  sorry

end tire_swap_distance_l1698_169899


namespace rhombus_longer_diagonal_l1698_169846

theorem rhombus_longer_diagonal 
  (a b : ℝ) 
  (h₁ : a = 61) 
  (h₂ : b = 44) :
  ∃ d₂ : ℝ, d₂ = 2 * Real.sqrt (a * a - (b / 2) * (b / 2)) :=
sorry

end rhombus_longer_diagonal_l1698_169846


namespace initial_number_of_balls_l1698_169816

theorem initial_number_of_balls (T B : ℕ) (P : ℚ) (after3_blue : ℕ) (prob : ℚ) :
  B = 7 → after3_blue = B - 3 → prob = after3_blue / T → prob = 1/3 → T = 15 :=
by
  sorry

end initial_number_of_balls_l1698_169816


namespace sum_first_five_terms_l1698_169884

-- Define the geometric sequence
noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^n

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n
  else a1 * (1 - q^(n + 1)) / (1 - q)

-- Given conditions
def a1 : ℝ := 1
def q : ℝ := 2
def n : ℕ := 5

-- The theorem to be proven
theorem sum_first_five_terms : sum_geometric_sequence a1 q (n-1) = 31 := by
  sorry

end sum_first_five_terms_l1698_169884


namespace walt_age_l1698_169888

theorem walt_age (T W : ℕ) 
  (h1 : T = 3 * W)
  (h2 : T + 12 = 2 * (W + 12)) : 
  W = 12 :=
by
  sorry

end walt_age_l1698_169888


namespace Problem_l1698_169808

theorem Problem (x y : ℝ) (h1 : 2*x + 2*y = 10) (h2 : x*y = -15) : 4*(x^2) + 4*(y^2) = 220 := 
by
  sorry

end Problem_l1698_169808


namespace john_total_distance_l1698_169849

theorem john_total_distance :
  let speed := 55 -- John's speed in mph
  let time1 := 2 -- Time before lunch in hours
  let time2 := 3 -- Time after lunch in hours
  let distance1 := speed * time1 -- Distance before lunch
  let distance2 := speed * time2 -- Distance after lunch
  let total_distance := distance1 + distance2 -- Total distance

  total_distance = 275 :=
by
  sorry

end john_total_distance_l1698_169849


namespace machine_present_value_l1698_169853

theorem machine_present_value
  (rate_of_decay : ℝ) (n_periods : ℕ) (final_value : ℝ) (initial_value : ℝ)
  (h_decay : rate_of_decay = 0.25)
  (h_periods : n_periods = 2)
  (h_final_value : final_value = 225) :
  initial_value = 400 :=
by
  -- The proof would go here. 
  sorry

end machine_present_value_l1698_169853


namespace pills_per_week_l1698_169874

theorem pills_per_week (hours_per_pill : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) 
(h1: hours_per_pill = 6) (h2: hours_per_day = 24) (h3: days_per_week = 7) :
(hours_per_day / hours_per_pill) * days_per_week = 28 :=
by
  sorry

end pills_per_week_l1698_169874


namespace find_x10_l1698_169871

theorem find_x10 (x : ℕ → ℝ) :
  x 1 = 1 ∧ x 2 = 1 ∧ (∀ n ≥ 2, x (n + 1) = (x n * x (n - 1)) / (x n + x (n - 1))) →
  x 10 = 1 / 55 :=
by sorry

end find_x10_l1698_169871


namespace integer_sequence_perfect_square_l1698_169844

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ a 2 = 4 ∧ ∀ n ≥ 2, a n = (a (n - 1) * a (n + 1) + 1) ^ (1 / 2)

theorem integer_sequence {a : ℕ → ℝ} : 
  seq a → ∀ n, ∃ k : ℤ, a n = k := 
by sorry

theorem perfect_square {a : ℕ → ℝ} :
  seq a → ∀ n, ∃ k : ℤ, 2 * a n * a (n + 1) + 1 = k ^ 2 :=
by sorry

end integer_sequence_perfect_square_l1698_169844


namespace distance_between_parallel_lines_l1698_169896

theorem distance_between_parallel_lines 
  (r : ℝ) (d : ℝ) 
  (h1 : 3 * (2 * r^2) = 722 + (19 / 4) * d^2) 
  (h2 : 3 * (2 * r^2) = 578 + (153 / 4) * d^2) : 
  d = 6 :=
by
  sorry

end distance_between_parallel_lines_l1698_169896


namespace find_a_and_b_l1698_169825

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x + 1))

theorem find_a_and_b
  (a b : ℝ)
  (h1 : a < b)
  (h2 : f a = f ((- (b + 1)) / (b + 2)))
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) :
  a = - 2 / 5 ∧ b = - 1 / 3 :=
sorry

end find_a_and_b_l1698_169825


namespace distance_from_neg2_l1698_169868

theorem distance_from_neg2 (x : ℝ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 := 
by sorry

end distance_from_neg2_l1698_169868


namespace find_ab_l1698_169851

-- Define the statement to be proven
theorem find_ab (a b : ℕ) (h1 : (a + b) % 3 = 2)
                           (h2 : b % 5 = 3)
                           (h3 : (b - a) % 11 = 1) :
  10 * a + b = 23 := 
sorry

end find_ab_l1698_169851


namespace driving_distance_l1698_169804

def miles_per_gallon : ℕ := 20
def gallons_of_gas : ℕ := 5

theorem driving_distance :
  miles_per_gallon * gallons_of_gas = 100 :=
  sorry

end driving_distance_l1698_169804


namespace estimate_y_value_l1698_169890

theorem estimate_y_value : 
  ∀ (x : ℝ), x = 25 → 0.50 * x - 0.81 = 11.69 :=
by 
  intro x h
  rw [h]
  norm_num


end estimate_y_value_l1698_169890


namespace symmetric_point_reflection_l1698_169819

theorem symmetric_point_reflection (x y : ℝ) : (2, -(-5)) = (2, 5) := by
  sorry

end symmetric_point_reflection_l1698_169819


namespace sine_of_negative_90_degrees_l1698_169886

theorem sine_of_negative_90_degrees : Real.sin (-(Real.pi / 2)) = -1 := 
sorry

end sine_of_negative_90_degrees_l1698_169886


namespace inequality_flip_l1698_169823

theorem inequality_flip (a b : ℤ) (c : ℤ) (h1 : a < b) (h2 : c < 0) : 
  c * a > c * b :=
sorry

end inequality_flip_l1698_169823


namespace women_stockbrokers_2005_l1698_169809

-- Define the context and conditions
def women_stockbrokers_2000 : ℕ := 10000
def percent_increase_2005 : ℕ := 100

-- Statement to prove the number of women stockbrokers in 2005
theorem women_stockbrokers_2005 : women_stockbrokers_2000 + women_stockbrokers_2000 * percent_increase_2005 / 100 = 20000 := by
  sorry

end women_stockbrokers_2005_l1698_169809


namespace parallel_vectors_l1698_169832

variable {k m : ℝ}

theorem parallel_vectors (h₁ : (2 : ℝ) = k * m) (h₂ : m = 2 * k) : m = 2 ∨ m = -2 :=
by
  sorry

end parallel_vectors_l1698_169832


namespace product_of_ys_l1698_169885

theorem product_of_ys (x y : ℤ) (h1 : x^3 + y^2 - 3 * y + 1 < 0)
                                     (h2 : 3 * x^3 - y^2 + 3 * y > 0) : 
  (y = 1 ∨ y = 2) → (1 * 2 = 2) :=
by {
  sorry
}

end product_of_ys_l1698_169885


namespace lcm_12_20_correct_l1698_169891

def lcm_12_20_is_60 : Nat := Nat.lcm 12 20

theorem lcm_12_20_correct : Nat.lcm 12 20 = 60 := by
  -- assumed factorization conditions as prerequisites
  have h₁ : Nat.primeFactors 12 = {2, 3} := sorry
  have h₂ : Nat.primeFactors 20 = {2, 5} := sorry
  -- the main proof goal
  exact sorry

end lcm_12_20_correct_l1698_169891
