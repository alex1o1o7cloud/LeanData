import Mathlib

namespace circle_symmetry_l2389_238976

def initial_circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 1 = 0

def standard_form_eq (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

def symmetric_circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

theorem circle_symmetry :
  (∀ x y : ℝ, initial_circle_eq x y ↔ standard_form_eq x y) →
  (∀ x y : ℝ, standard_form_eq x y → symmetric_circle_eq (-x) (-y)) →
  ∀ x y : ℝ, initial_circle_eq x y → symmetric_circle_eq x y :=
by
  intros h1 h2 x y hxy
  sorry

end circle_symmetry_l2389_238976


namespace change_is_41_l2389_238948

-- Define the cost of shirts and sandals as given in the problem conditions
def cost_of_shirts : ℕ := 10 * 5
def cost_of_sandals : ℕ := 3 * 3
def total_cost : ℕ := cost_of_shirts + cost_of_sandals

-- Define the amount given
def amount_given : ℕ := 100

-- Calculate the change
def change := amount_given - total_cost

-- State the theorem
theorem change_is_41 : change = 41 := 
by 
  -- Filling this with justification steps would be the actual proof
  -- but it's not required, so we use 'sorry' to indicate the theorem
  sorry

end change_is_41_l2389_238948


namespace largest_square_side_length_largest_rectangle_dimensions_l2389_238927

variable (a b : ℝ)

-- Part a
theorem largest_square_side_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ s : ℝ, s = (a * b) / (a + b) :=
sorry

-- Part b
theorem largest_rectangle_dimensions (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ x y : ℝ, (x = a / 2 ∧ y = b / 2) :=
sorry

end largest_square_side_length_largest_rectangle_dimensions_l2389_238927


namespace zeoland_speeding_fine_l2389_238995

-- Define the conditions
def fine_per_mph (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ) : ℕ :=
  total_fine / (actual_speed - speed_limit)

-- Variables for the given problem
variables (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ)
variable (fine_per_mph_over_limit : ℕ)

-- Theorem statement
theorem zeoland_speeding_fine :
  total_fine = 256 ∧ speed_limit = 50 ∧ actual_speed = 66 →
  fine_per_mph total_fine actual_speed speed_limit = 16 :=
by
  sorry

end zeoland_speeding_fine_l2389_238995


namespace overall_average_score_l2389_238931

theorem overall_average_score
  (mean_morning mean_evening : ℕ)
  (ratio_morning_evening : ℚ) 
  (h1 : mean_morning = 90)
  (h2 : mean_evening = 80)
  (h3 : ratio_morning_evening = 4 / 5) : 
  ∃ overall_mean : ℚ, overall_mean = 84 :=
by
  sorry

end overall_average_score_l2389_238931


namespace B_correct_A_inter_B_correct_l2389_238922

def A := {x : ℝ | 1 < x ∧ x < 8}
def B := {x : ℝ | x^2 - 5 * x - 14 ≥ 0}

theorem B_correct : B = {x : ℝ | x ≤ -2 ∨ x ≥ 7} := 
sorry

theorem A_inter_B_correct : A ∩ B = {x : ℝ | 7 ≤ x ∧ x < 8} :=
sorry

end B_correct_A_inter_B_correct_l2389_238922


namespace expenditure_representation_correct_l2389_238914

-- Define the representation of income
def income_representation (income : ℝ) : ℝ :=
  income

-- Define the representation of expenditure
def expenditure_representation (expenditure : ℝ) : ℝ :=
  -expenditure

-- Condition: an income of 10.5 yuan is represented as +10.5 yuan.
-- We need to prove: an expenditure of 6 yuan is represented as -6 yuan.
theorem expenditure_representation_correct (h : income_representation 10.5 = 10.5) : 
  expenditure_representation 6 = -6 :=
by
  sorry

end expenditure_representation_correct_l2389_238914


namespace factor_expression_l2389_238979

theorem factor_expression (b : ℚ) : 
  294 * b^3 + 63 * b^2 - 21 * b = 21 * b * (14 * b^2 + 3 * b - 1) :=
by 
  sorry

end factor_expression_l2389_238979


namespace average_operating_time_l2389_238997

-- Definition of problem conditions
def cond1 : Nat := 5 -- originally had 5 air conditioners
def cond2 : Nat := 6 -- after installing 1 more
def total_hours : Nat := 24 * 5 -- total operating hours allowable in 24 hours

-- Formalize the average operating time calculation
theorem average_operating_time : (total_hours / cond2) = 20 := by
  sorry

end average_operating_time_l2389_238997


namespace certain_number_divisible_l2389_238954

theorem certain_number_divisible (x : ℤ) (n : ℤ) (h1 : 0 < n ∧ n < 11) (h2 : x - n = 11 * k) (h3 : n = 1) : x = 12 :=
by sorry

end certain_number_divisible_l2389_238954


namespace lcm_of_18_and_30_l2389_238985

theorem lcm_of_18_and_30 : Nat.lcm 18 30 = 90 := 
by
  sorry

end lcm_of_18_and_30_l2389_238985


namespace min_value_3x_4y_l2389_238947

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 25 :=
sorry

end min_value_3x_4y_l2389_238947


namespace find_a_l2389_238988

def f(x : ℚ) : ℚ := x / 3 + 2
def g(x : ℚ) : ℚ := 5 - 2 * x

theorem find_a (a : ℚ) (h : f (g a) = 4) : a = -1 / 2 :=
by
  sorry

end find_a_l2389_238988


namespace sin_double_angle_l2389_238920

theorem sin_double_angle (θ : ℝ)
  (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) :
  Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
sorry

end sin_double_angle_l2389_238920


namespace part1_a1_union_part2_A_subset_complement_B_l2389_238943

open Set Real

-- Definitions for Part (1)
def A : Set ℝ := {x | (x - 1) / (x - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a^2 - 1 < 0}

-- Statement for Part (1)
theorem part1_a1_union (a : ℝ) (h : a = 1) : A ∪ B 1 = {x | 0 < x ∧ x < 5} :=
sorry

-- Definitions for Part (2)
def complement_B (a : ℝ) : Set ℝ := {x | x ≤ a - 1 ∨ x ≥ a + 1}

-- Statement for Part (2)
theorem part2_A_subset_complement_B : (∀ x, (1 < x ∧ x < 5) → (x ≤ a - 1 ∨ x ≥ a + 1)) → (a ≤ 0 ∨ a ≥ 6) :=
sorry

end part1_a1_union_part2_A_subset_complement_B_l2389_238943


namespace test_total_questions_l2389_238969

theorem test_total_questions (total_points : ℕ) (num_5_point_questions : ℕ) (points_per_5_point_question : ℕ) (points_per_10_point_question : ℕ) : 
  total_points = 200 → 
  num_5_point_questions = 20 → 
  points_per_5_point_question = 5 → 
  points_per_10_point_question = 10 → 
  (total_points = (num_5_point_questions * points_per_5_point_question) + 
    ((total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) * points_per_10_point_question) →
  (num_5_point_questions + (total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end test_total_questions_l2389_238969


namespace min_three_beverages_overlap_l2389_238978

variable (a b c d : ℝ)
variable (ha : a = 0.9)
variable (hb : b = 0.8)
variable (hc : c = 0.7)

theorem min_three_beverages_overlap : d = 0.7 :=
by
  sorry

end min_three_beverages_overlap_l2389_238978


namespace g_at_8_equals_minus_30_l2389_238939

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_8_equals_minus_30 :
  (∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2) →
  g 8 = -30 :=
by
  intro h
  sorry

end g_at_8_equals_minus_30_l2389_238939


namespace g_at_neg1_l2389_238998

-- Defining even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

-- Given functions f and g
variables (f g : ℝ → ℝ)

-- Given conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom fg_relation : ∀ x : ℝ, f x - g x = 2^(1 - x)

-- Proof statement
theorem g_at_neg1 : g (-1) = -3 / 2 :=
by
  sorry

end g_at_neg1_l2389_238998


namespace mary_baseball_cards_l2389_238986

theorem mary_baseball_cards :
  let initial_cards := 18
  let torn_cards := 8
  let fred_gifted_cards := 26
  let bought_cards := 40
  let exchanged_cards := 10
  let lost_cards := 5
  
  let remaining_cards := initial_cards - torn_cards
  let after_gift := remaining_cards + fred_gifted_cards
  let after_buy := after_gift + bought_cards
  let after_exchange := after_buy - exchanged_cards + exchanged_cards
  let final_count := after_exchange - lost_cards
  
  final_count = 71 :=
by
  sorry

end mary_baseball_cards_l2389_238986


namespace betty_eggs_used_l2389_238933

-- Conditions as definitions
def ratio_sugar_cream_cheese (sugar cream_cheese : ℚ) : Prop :=
  sugar / cream_cheese = 1 / 4

def ratio_vanilla_cream_cheese (vanilla cream_cheese : ℚ) : Prop :=
  vanilla / cream_cheese = 1 / 2

def ratio_eggs_vanilla (eggs vanilla : ℚ) : Prop :=
  eggs / vanilla = 2

-- Given conditions
def sugar_used : ℚ := 2 -- cups of sugar

-- The statement to prove
theorem betty_eggs_used (cream_cheese vanilla eggs : ℚ) 
  (h1 : ratio_sugar_cream_cheese sugar_used cream_cheese)
  (h2 : ratio_vanilla_cream_cheese vanilla cream_cheese)
  (h3 : ratio_eggs_vanilla eggs vanilla) :
  eggs = 8 :=
sorry

end betty_eggs_used_l2389_238933


namespace nested_radical_solution_l2389_238911

theorem nested_radical_solution :
  (∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x ≥ 0) ∧ ∀ x : ℝ, x = Real.sqrt (18 + x) → x ≥ 0 → x = 6 :=
by
  sorry

end nested_radical_solution_l2389_238911


namespace lowest_score_l2389_238987

theorem lowest_score (score1 score2 : ℕ) (max_score : ℕ) (desired_mean : ℕ) (lowest_possible_score : ℕ) 
  (h_score1 : score1 = 82) (h_score2 : score2 = 75) (h_max_score : max_score = 100) (h_desired_mean : desired_mean = 85)
  (h_lowest_possible_score : lowest_possible_score = 83) : 
  ∃ x1 x2 : ℕ, x1 = max_score ∧ x2 = lowest_possible_score ∧ (score1 + score2 + x1 + x2) / 4 = desired_mean := by
  sorry

end lowest_score_l2389_238987


namespace geometric_series_sum_l2389_238918

theorem geometric_series_sum :
  let a := 2 / 3
  let r := 1 / 3
  a / (1 - r) = 1 :=
by
  sorry

end geometric_series_sum_l2389_238918


namespace adjusted_volume_bowling_ball_l2389_238963

noncomputable def bowling_ball_diameter : ℝ := 40
noncomputable def hole1_diameter : ℝ := 5
noncomputable def hole1_depth : ℝ := 10
noncomputable def hole2_diameter : ℝ := 4
noncomputable def hole2_depth : ℝ := 12
noncomputable def expected_adjusted_volume : ℝ := 10556.17 * Real.pi

theorem adjusted_volume_bowling_ball :
  let radius := bowling_ball_diameter / 2
  let volume_ball := (4 / 3) * Real.pi * radius^3
  let hole1_radius := hole1_diameter / 2
  let hole1_volume := Real.pi * hole1_radius^2 * hole1_depth
  let hole2_radius := hole2_diameter / 2
  let hole2_volume := Real.pi * hole2_radius^2 * hole2_depth
  let adjusted_volume := volume_ball - hole1_volume - hole2_volume
  adjusted_volume = expected_adjusted_volume :=
by
  sorry

end adjusted_volume_bowling_ball_l2389_238963


namespace initial_number_of_men_l2389_238951

theorem initial_number_of_men (P : ℝ) (M : ℝ) (h1 : P = 15 * M * (P / (15 * M))) (h2 : P = 12.5 * (M + 200) * (P / (12.5 * (M + 200)))) : M = 1000 :=
by
  sorry

end initial_number_of_men_l2389_238951


namespace men_in_first_group_l2389_238994

theorem men_in_first_group (M : ℕ) (h1 : (M * 7 * 18) = (12 * 7 * 12)) : M = 8 :=
by sorry

end men_in_first_group_l2389_238994


namespace ay_bz_cx_lt_S_squared_l2389_238981

theorem ay_bz_cx_lt_S_squared 
  (S : ℝ) (a b c x y z : ℝ) 
  (hS : 0 < S) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h1 : a + x = S) 
  (h2 : b + y = S) 
  (h3 : c + z = S) : 
  a * y + b * z + c * x < S^2 := 
sorry

end ay_bz_cx_lt_S_squared_l2389_238981


namespace find_months_contributed_l2389_238957

theorem find_months_contributed (x : ℕ) (profit_A profit_total : ℝ)
  (contrib_A : ℝ) (contrib_B : ℝ) (months_B : ℕ) :
  profit_A / profit_total = (contrib_A * x) / (contrib_A * x + contrib_B * months_B) →
  profit_A = 4800 →
  profit_total = 8400 →
  contrib_A = 5000 →
  contrib_B = 6000 →
  months_B = 5 →
  x = 8 :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end find_months_contributed_l2389_238957


namespace cards_traded_between_Padma_and_Robert_l2389_238930

def total_cards_traded (padma_first_trade padma_second_trade robert_first_trade robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + padma_second_trade + robert_first_trade + robert_second_trade

theorem cards_traded_between_Padma_and_Robert (h1 : padma_first_trade = 2) 
                                            (h2 : robert_first_trade = 10)
                                            (h3 : padma_second_trade = 15)
                                            (h4 : robert_second_trade = 8) :
                                            total_cards_traded 2 15 10 8 = 35 := 
by 
  sorry

end cards_traded_between_Padma_and_Robert_l2389_238930


namespace arithmetic_sequence_general_term_geometric_sequence_sum_l2389_238971

section ArithmeticSequence

variable {a_n : ℕ → ℤ} {d : ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a_n (n + 1) - a_n n = d

theorem arithmetic_sequence_general_term (h : is_arithmetic_sequence a_n 2) :
  ∃ a1 : ℤ, ∀ n, a_n n = 2 * n + a1 :=
sorry

end ArithmeticSequence

section GeometricSequence

variable {b_n : ℕ → ℤ} {a_n : ℕ → ℤ}

def is_geometric_sequence_with_reference (b_n : ℕ → ℤ) (a_n : ℕ → ℤ) :=
  b_n 1 = a_n 1 ∧ b_n 2 = a_n 4 ∧ b_n 3 = a_n 13

theorem geometric_sequence_sum (h : is_geometric_sequence_with_reference b_n a_n)
  (h_arith : is_arithmetic_sequence a_n 2) :
  ∃ b1 : ℤ, ∀ n, b_n n = b1 * 3^(n - 1) ∧
                (∃ Sn : ℕ → ℤ, Sn n = (3 * (3^n - 1)) / 2) :=
sorry

end GeometricSequence

end arithmetic_sequence_general_term_geometric_sequence_sum_l2389_238971


namespace polynomial_value_l2389_238929

theorem polynomial_value (x y : ℝ) (h : x + 2 * y = 6) : 2 * x + 4 * y - 5 = 7 :=
by
  sorry

end polynomial_value_l2389_238929


namespace a_capital_used_l2389_238936

theorem a_capital_used (C P x : ℕ) (h_b_contributes : 3 * C / 4 - C ≥ 0) 
(h_b_receives : 2 * P / 3 - P ≥ 0) 
(h_b_money_used : 10 > 0) 
(h_ratio : 1 / 2 = x / 30) 
: x = 15 :=
sorry

end a_capital_used_l2389_238936


namespace compound_interest_rate_l2389_238984

theorem compound_interest_rate (
  P : ℝ) (r : ℝ)  (A : ℕ → ℝ) :
  A 2 = 2420 ∧ A 3 = 3025 ∧ 
  (∀ n : ℕ, A n = P * (1 + r / 100)^n) → r = 25 :=
by
  sorry

end compound_interest_rate_l2389_238984


namespace smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l2389_238977

theorem smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits : 
  ∃ n : ℕ, n < 10000 ∧ 1000 ≤ n ∧ (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ 
    (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1 ∧ d % 2 = 0) ∧ 
    (n % 11 = 0)) ∧ n = 1056 :=
by
  sorry

end smallest_four_digit_number_divisible_by_11_with_two_even_two_odd_digits_l2389_238977


namespace total_pay_XY_l2389_238975

-- Assuming X's pay is 120% of Y's pay and Y's pay is 268.1818181818182,
-- Prove that the total pay to X and Y is 590.00.
theorem total_pay_XY (Y_pay : ℝ) (X_pay : ℝ) (total_pay : ℝ) :
  Y_pay = 268.1818181818182 →
  X_pay = 1.2 * Y_pay →
  total_pay = X_pay + Y_pay →
  total_pay = 590.00 :=
by
  intros hY hX hT
  sorry

end total_pay_XY_l2389_238975


namespace largest_digit_divisible_by_4_l2389_238966

theorem largest_digit_divisible_by_4 :
  ∃ (A : ℕ), A ≤ 9 ∧ (∃ n : ℕ, 100000 * 4 + 10000 * A + 67994 = n * 4) ∧ 
  (∀ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (∃ m : ℕ, 100000 * 4 + 10000 * B + 67994 = m * 4) → B ≤ A) :=
sorry

end largest_digit_divisible_by_4_l2389_238966


namespace range_of_k_for_obtuse_triangle_l2389_238949

theorem range_of_k_for_obtuse_triangle (k : ℝ) (a b c : ℝ) (h₁ : a = k) (h₂ : b = k + 2) (h₃ : c = k + 4) : 
  2 < k ∧ k < 6 :=
by
  sorry

end range_of_k_for_obtuse_triangle_l2389_238949


namespace fixed_point_of_function_l2389_238905

theorem fixed_point_of_function (a : ℝ) : 
  (a - 1) * 2^1 - 2 * a = -2 := by
  sorry

end fixed_point_of_function_l2389_238905


namespace completing_the_square_equation_l2389_238915

theorem completing_the_square_equation : 
  ∀ (x : ℝ), (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intro x
  sorry

end completing_the_square_equation_l2389_238915


namespace jen_total_birds_l2389_238912

-- Define the number of chickens and ducks
variables (C D : ℕ)

-- Define the conditions
def ducks_condition (C D : ℕ) : Prop := D = 4 * C + 10
def num_ducks (D : ℕ) : Prop := D = 150

-- Define the total number of birds
def total_birds (C D : ℕ) : ℕ := C + D

-- Prove that the total number of birds is 185 given the conditions
theorem jen_total_birds (C D : ℕ) (h1 : ducks_condition C D) (h2 : num_ducks D) : total_birds C D = 185 :=
by
  sorry

end jen_total_birds_l2389_238912


namespace ab_equals_six_l2389_238955

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l2389_238955


namespace balloon_minimum_volume_l2389_238945

theorem balloon_minimum_volume 
  (p V : ℝ)
  (h1 : p * V = 24000)
  (h2 : p ≤ 40000) : 
  V ≥ 0.6 :=
  sorry

end balloon_minimum_volume_l2389_238945


namespace total_distance_traveled_l2389_238904

-- Define the parameters and conditions
def hoursPerDay : ℕ := 2
def daysPerWeek : ℕ := 5
def daysPeriod1 : ℕ := 3
def daysPeriod2 : ℕ := 2
def speedPeriod1 : ℕ := 12 -- speed in km/h from Monday to Wednesday
def speedPeriod2 : ℕ := 9 -- speed in km/h from Thursday to Friday

-- This is the theorem we want to prove
theorem total_distance_traveled : (daysPeriod1 * hoursPerDay * speedPeriod1) + (daysPeriod2 * hoursPerDay * speedPeriod2) = 108 :=
by
  sorry

end total_distance_traveled_l2389_238904


namespace expression_D_divisible_by_9_l2389_238924

theorem expression_D_divisible_by_9 (k : ℕ) (hk : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
by
  sorry

end expression_D_divisible_by_9_l2389_238924


namespace general_term_arithmetic_sum_first_n_terms_geometric_l2389_238935

-- Definitions and assumptions based on given conditions
def a (n : ℕ) : ℤ := 2 * n + 1

-- Given conditions
def initial_a1 : ℤ := 3
def common_difference : ℤ := 2

-- Validate the general formula for the arithmetic sequence
theorem general_term_arithmetic : ∀ n : ℕ, a n = 2 * n + 1 := 
by sorry

-- Definitions and assumptions for geometric sequence
def b (n : ℕ) : ℤ := 3^n

-- Sum of the first n terms of the geometric sequence
def Sn (n : ℕ) : ℤ := 3 / 2 * (3^n - 1)

-- Validate the sum formula for the geometric sequence
theorem sum_first_n_terms_geometric (n : ℕ) : Sn n = 3 / 2 * (3^n - 1) := 
by sorry

end general_term_arithmetic_sum_first_n_terms_geometric_l2389_238935


namespace acronym_XYZ_length_l2389_238990

theorem acronym_XYZ_length :
  let X_length := 2 * Real.sqrt 2
  let Y_length := 1 + 2 * Real.sqrt 2
  let Z_length := 4 + Real.sqrt 5
  X_length + Y_length + Z_length = 5 + 4 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end acronym_XYZ_length_l2389_238990


namespace solve_system_l2389_238917

variable (y : ℝ) (x1 x2 x3 x4 x5 : ℝ)

def system_of_equations :=
  x5 + x2 = y * x1 ∧
  x1 + x3 = y * x2 ∧
  x2 + x4 = y * x3 ∧
  x3 + x5 = y * x4 ∧
  x4 + x1 = y * x3

theorem solve_system :
  (y = 2 → x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5) ∧
  ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) →
   x1 + x2 + x3 + x4 + x5 = 0 ∧ ∀ (x1 x5 : ℝ), system_of_equations y x1 x2 x3 x4 x5) :=
sorry

end solve_system_l2389_238917


namespace regular_tetrahedron_ratio_l2389_238928

/-- In plane geometry, the ratio of the radius of the circumscribed circle to the 
inscribed circle of an equilateral triangle is 2:1, --/
def ratio_radii_equilateral_triangle : ℚ := 2 / 1

/-- In space geometry, we study the relationship between the radii of the circumscribed
sphere and the inscribed sphere of a regular tetrahedron. --/
def ratio_radii_regular_tetrahedron : ℚ := 3 / 1

/-- Prove the ratio of the radius of the circumscribed sphere to the inscribed sphere
of a regular tetrahedron is 3 : 1, given the ratio is 2 : 1 for the equilateral triangle. --/
theorem regular_tetrahedron_ratio : 
  ratio_radii_equilateral_triangle = 2 / 1 → 
  ratio_radii_regular_tetrahedron = 3 / 1 :=
by
  sorry

end regular_tetrahedron_ratio_l2389_238928


namespace complement_union_eq_l2389_238952

-- Define the sets U, M, and N
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

-- Define the complement of a set within another set
def complement (S T : Set ℕ) : Set ℕ := { x | x ∈ S ∧ x ∉ T }

-- Define the union of M and N
def union_M_N : Set ℕ := {x | x ∈ M ∨ x ∈ N}

-- State the theorem
theorem complement_union_eq :
  complement U union_M_N = {4} :=
sorry

end complement_union_eq_l2389_238952


namespace height_on_hypotenuse_correct_l2389_238991

noncomputable def height_on_hypotenuse (a b : ℝ) (ha : a = 3) (hb : b = 4) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let area := (a * b) / 2
  (2 * area) / c

theorem height_on_hypotenuse_correct (h : ℝ) : 
  height_on_hypotenuse 3 4 rfl rfl = 12 / 5 :=
by
  sorry

end height_on_hypotenuse_correct_l2389_238991


namespace MrKishore_petrol_expense_l2389_238909

theorem MrKishore_petrol_expense 
  (rent milk groceries education misc savings salary expenses petrol : ℝ)
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_education : education = 2500)
  (h_misc : misc = 700)
  (h_savings : savings = 1800)
  (h_salary : salary = 18000)
  (h_expenses_equation : expenses = rent + milk + groceries + education + petrol + misc)
  (h_savings_equation : savings = salary * 0.10)
  (h_total_equation : salary = expenses + savings) :
  petrol = 2000 :=
by
  sorry

end MrKishore_petrol_expense_l2389_238909


namespace largest_number_is_B_l2389_238953
open Real

noncomputable def A := 0.989
noncomputable def B := 0.998
noncomputable def C := 0.899
noncomputable def D := 0.9899
noncomputable def E := 0.8999

theorem largest_number_is_B :
  B = max (max (max (max A B) C) D) E :=
by
  sorry

end largest_number_is_B_l2389_238953


namespace distinct_roots_and_ratios_l2389_238958

open Real

theorem distinct_roots_and_ratios (a b : ℝ) (h1 : a^2 - 3*a - 1 = 0) (h2 : b^2 - 3*b - 1 = 0) (h3 : a ≠ b) :
  b/a + a/b = -11 :=
sorry

end distinct_roots_and_ratios_l2389_238958


namespace serpent_ridge_trail_length_l2389_238999

/-- Phoenix hiked the Serpent Ridge Trail last week. It took her five days to complete the trip.
The first two days she hiked a total of 28 miles. The second and fourth days she averaged 15 miles per day.
The last three days she hiked a total of 42 miles. The total hike for the first and third days was 30 miles.
How many miles long was the trail? -/
theorem serpent_ridge_trail_length
  (a b c d e : ℕ)
  (h1 : a + b = 28)
  (h2 : b + d = 30)
  (h3 : c + d + e = 42)
  (h4 : a + c = 30) :
  a + b + c + d + e = 70 :=
sorry

end serpent_ridge_trail_length_l2389_238999


namespace min_value_part1_l2389_238906

open Real

theorem min_value_part1 (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by {
  sorry
}

end min_value_part1_l2389_238906


namespace inequality_proof_l2389_238938

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c)) ≥ 1 / 3 := 
sorry

end inequality_proof_l2389_238938


namespace natural_number_pairs_l2389_238907

theorem natural_number_pairs (a b : ℕ) (p q : ℕ) :
  a ≠ b →
  (∃ p, a + b = 2^p) →
  (∃ q, ab + 1 = 2^q) →
  (a = 1 ∧ b = 2^p - 1 ∨ a = 2^q - 1 ∧ b = 2^q + 1) :=
by intro hne hp hq; sorry

end natural_number_pairs_l2389_238907


namespace evaluate_expression_l2389_238960

theorem evaluate_expression :
  (3 / 2) * ((8 / 3) * ((15 / 8) - (5 / 6))) / (((7 / 8) + (11 / 6)) / (13 / 4)) = 5 :=
by
  sorry

end evaluate_expression_l2389_238960


namespace snack_bar_training_count_l2389_238970

noncomputable def num_trained_in_snack_bar 
  (total_employees : ℕ) 
  (trained_in_buffet : ℕ) 
  (trained_in_dining_room : ℕ) 
  (trained_in_two_restaurants : ℕ) 
  (trained_in_three_restaurants : ℕ) : ℕ :=
  total_employees - trained_in_buffet - trained_in_dining_room + 
  trained_in_two_restaurants + trained_in_three_restaurants

theorem snack_bar_training_count : 
  num_trained_in_snack_bar 39 17 18 4 2 = 8 :=
sorry

end snack_bar_training_count_l2389_238970


namespace fraction_of_6_l2389_238967

theorem fraction_of_6 (x y : ℕ) (h : (x / y : ℚ) * 6 + 6 = 10) : (x / y : ℚ) = 2 / 3 :=
by
  sorry

end fraction_of_6_l2389_238967


namespace chocolate_bar_min_breaks_l2389_238982

theorem chocolate_bar_min_breaks (n : ℕ) (h : n = 40) : ∃ k : ℕ, k = n - 1 := 
by 
  sorry

end chocolate_bar_min_breaks_l2389_238982


namespace solution_set_of_inequality_l2389_238921

theorem solution_set_of_inequality (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_mono : ∀ {x1 x2}, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0) (h_f1 : f 1 = 0) :
  {x | (x - 1) * f x > 0} = {x | -1 < x ∧ x < 1} ∪ {x | 1 < x} :=
by
  sorry

end solution_set_of_inequality_l2389_238921


namespace sum_of_edges_of_geometric_progression_solid_l2389_238903

theorem sum_of_edges_of_geometric_progression_solid
  (a : ℝ)
  (r : ℝ)
  (volume_eq : a^3 = 512)
  (surface_eq : 2 * (64 / r + 64 * r + 64) = 352)
  (r_value : r = 1.25 ∨ r = 0.8) :
  4 * (8 / r + 8 + 8 * r) = 97.6 := by
  sorry

end sum_of_edges_of_geometric_progression_solid_l2389_238903


namespace intersection_S_T_l2389_238925

def S : Set ℝ := { y | y ≥ 0 }
def T : Set ℝ := { x | x > 1 }

theorem intersection_S_T :
  S ∩ T = { z | z > 1 } :=
sorry

end intersection_S_T_l2389_238925


namespace find_center_of_circle_l2389_238973

theorem find_center_of_circle (x y : ℝ) :
  4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 16 = 0 →
  (x + 1)^2 + (y - 3)^2 = 6 :=
by
  intro h
  sorry

end find_center_of_circle_l2389_238973


namespace find_k_l2389_238926

theorem find_k (n m : ℕ) (hn : n > 0) (hm : m > 0) (h : (1 : ℚ) / n^2 + 1 / m^2 = k / (n^2 + m^2)) : k = 4 :=
sorry

end find_k_l2389_238926


namespace sad_children_count_l2389_238983

-- Definitions of conditions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 18
def girls : ℕ := 42
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4

-- Calculate the number of children who are either happy or sad
def happy_or_sad_children : ℕ := total_children - neither_happy_nor_sad_children

-- Prove that the number of sad children is 10
theorem sad_children_count : happy_or_sad_children - happy_children = 10 := by
  sorry

end sad_children_count_l2389_238983


namespace find_t_l2389_238934

open Real

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem find_t (t : ℝ) 
  (area_eq_50 : area_of_triangle 3 15 15 0 0 t = 50) :
  t = 325 / 12 ∨ t = 125 / 12 := 
sorry

end find_t_l2389_238934


namespace zero_in_M_l2389_238959

def M : Set ℤ := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M :=
by
  sorry

end zero_in_M_l2389_238959


namespace max_side_of_triangle_with_perimeter_30_l2389_238980

theorem max_side_of_triangle_with_perimeter_30 
  (a b c : ℕ) 
  (h1 : a + b + c = 30) 
  (h2 : a ≥ b) 
  (h3 : b ≥ c) 
  (h4 : a < b + c) 
  (h5 : b < a + c) 
  (h6 : c < a + b) 
  : a ≤ 14 :=
sorry

end max_side_of_triangle_with_perimeter_30_l2389_238980


namespace Winnie_the_Pooh_guarantee_kilogram_l2389_238941

noncomputable def guarantee_minimum_honey : Prop :=
  ∃ (a1 a2 a3 a4 a5 : ℝ), 
    a1 + a2 + a3 + a4 + a5 = 3 ∧
    min (min (a1 + a2) (a2 + a3)) (min (a3 + a4) (a4 + a5)) ≥ 1

theorem Winnie_the_Pooh_guarantee_kilogram :
  guarantee_minimum_honey :=
sorry

end Winnie_the_Pooh_guarantee_kilogram_l2389_238941


namespace order_numbers_l2389_238932

theorem order_numbers : (5 / 2 : ℝ) < (3 : ℝ) ∧ (3 : ℝ) < Real.sqrt (10) := 
by
  sorry

end order_numbers_l2389_238932


namespace max_length_is_3sqrt2_l2389_238902

noncomputable def max_vector_length (θ : ℝ) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) : ℝ :=
  let OP₁ := (Real.cos θ, Real.sin θ)
  let OP₂ := (2 + Real.sin θ, 2 - Real.cos θ)
  let P₁P₂ := (OP₂.1 - OP₁.1, OP₂.2 - OP₁.2)
  Real.sqrt ((P₁P₂.1)^2 + (P₁P₂.2)^2)

theorem max_length_is_3sqrt2 : ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → max_vector_length θ sorry = 3 * Real.sqrt 2 := 
sorry

end max_length_is_3sqrt2_l2389_238902


namespace nat_solution_unique_l2389_238956

theorem nat_solution_unique (n : ℕ) (h : 2 * n - 1 / n^5 = 3 - 2 / n) : 
  n = 1 :=
sorry

end nat_solution_unique_l2389_238956


namespace number_of_letters_l2389_238901

-- Definitions and Conditions, based on the given problem
variables (n : ℕ) -- n is the number of different letters in the local language

-- Given: The people have lost 129 words due to the prohibition of the seventh letter
def words_lost_due_to_prohibition (n : ℕ) : ℕ := 2 * n

-- The main theorem to prove
theorem number_of_letters (h : 129 = words_lost_due_to_prohibition n) : n = 65 :=
by sorry

end number_of_letters_l2389_238901


namespace factor_determines_d_l2389_238944

theorem factor_determines_d (d : ℚ) :
  (∀ x : ℚ, x - 4 ∣ d * x^3 - 8 * x^2 + 5 * d * x - 12) → d = 5 / 3 := by
  sorry

end factor_determines_d_l2389_238944


namespace minimum_trucks_needed_l2389_238937

theorem minimum_trucks_needed {n : ℕ} (total_weight : ℕ) (box_weight : ℕ → ℕ) (truck_capacity : ℕ) :
  (total_weight = 10 ∧ truck_capacity = 3 ∧ (∀ b, box_weight b ≤ 1) ∧ (∃ n, 3 * n ≥ total_weight)) → n ≥ 5 :=
by
  -- We need to prove the statement based on the given conditions.
  sorry

end minimum_trucks_needed_l2389_238937


namespace range_of_a_l2389_238993

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (2 * x - 3) - 2 * a ≥ abs (x + a)) ↔ ( -3/2 ≤ a ∧ a < -1/2) := 
by sorry

end range_of_a_l2389_238993


namespace joseph_total_payment_l2389_238968
-- Importing necessary libraries

-- Defining the variables and conditions
variables (W : ℝ) -- The cost for the water heater

-- Conditions
def condition1 := 3 * W -- The cost for the refrigerator
def condition2 := 2 * W = 500 -- The electric oven
def condition3 := 300 -- The cost for the air conditioner
def condition4 := 100 -- The cost for the washing machine

-- Calculate total cost
def total_cost := (3 * W) + W + 500 + 300 + 100

-- The theorem stating the total amount Joseph pays
theorem joseph_total_payment : total_cost = 1900 :=
by 
  have hW := condition2;
  sorry

end joseph_total_payment_l2389_238968


namespace exists_real_x_for_sequence_floor_l2389_238996

open Real

theorem exists_real_x_for_sequence_floor (a : Fin 1998 → ℕ)
  (h1 : ∀ n : Fin 1998, 0 ≤ a n)
  (h2 : ∀ (i j : Fin 1998), (i.val + j.val ≤ 1997) → (a i + a j ≤ a ⟨i.val + j.val, sorry⟩ ∧ a ⟨i.val + j.val, sorry⟩ ≤ a i + a j + 1)) :
  ∃ x : ℝ, ∀ n : Fin 1998, a n = ⌊(n.val + 1) * x⌋ :=
sorry

end exists_real_x_for_sequence_floor_l2389_238996


namespace radius_of_fourth_circle_is_12_l2389_238946

theorem radius_of_fourth_circle_is_12 (r : ℝ) (radii : Fin 7 → ℝ) 
  (h_geometric : ∀ i, radii (Fin.succ i) = r * radii i) 
  (h_smallest : radii 0 = 6)
  (h_largest : radii 6 = 24) :
  radii 3 = 12 :=
by
  sorry

end radius_of_fourth_circle_is_12_l2389_238946


namespace target_runs_correct_l2389_238974

noncomputable def target_runs (run_rate1 : ℝ) (ovs1 : ℕ) (run_rate2 : ℝ) (ovs2 : ℕ) : ℝ :=
  (run_rate1 * ovs1) + (run_rate2 * ovs2)

theorem target_runs_correct : target_runs 4.5 12 8.052631578947368 38 = 360 :=
by
  sorry

end target_runs_correct_l2389_238974


namespace union_sets_l2389_238916

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_sets : A ∪ B = {2, 3, 5, 6} := by
  sorry

end union_sets_l2389_238916


namespace field_ratio_l2389_238992

theorem field_ratio (w l: ℕ) (h: l = 36)
  (h_area_ratio: 81 = (1/8) * l * w)
  (h_multiple: ∃ k : ℕ, l = k * w) :
  l / w = 2 :=
by 
  sorry

end field_ratio_l2389_238992


namespace sufficient_condition_parallel_planes_l2389_238919

-- Definitions for lines and planes
variable {Line Plane : Type}
variable {m n : Line}
variable {α β : Plane}

-- Relations between lines and planes
variable (parallel_line : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Condition for sufficient condition for α parallel β
theorem sufficient_condition_parallel_planes
  (h1 : parallel_line m n)
  (h2 : perpendicular_line_plane m α)
  (h3 : perpendicular_line_plane n β) :
  parallel_plane α β :=
sorry

end sufficient_condition_parallel_planes_l2389_238919


namespace common_elements_count_l2389_238964

theorem common_elements_count (S T : Set ℕ) (hS : S = {n | ∃ k : ℕ, k < 3000 ∧ n = 5 * (k + 1)})
    (hT : T = {n | ∃ k : ℕ, k < 3000 ∧ n = 8 * (k + 1)}) :
    S ∩ T = {n | ∃ m : ℕ, m < 375 ∧ n = 40 * (m + 1)} :=
by {
  sorry
}

end common_elements_count_l2389_238964


namespace train_length_is_549_95_l2389_238989

noncomputable def length_of_train 
(speed_of_train : ℝ) -- 63 km/hr
(speed_of_man : ℝ) -- 3 km/hr
(time_to_cross : ℝ) -- 32.997 seconds
: ℝ := 
(speed_of_train - speed_of_man) * (5 / 18) * time_to_cross

theorem train_length_is_549_95 (speed_of_train : ℝ) (speed_of_man : ℝ) (time_to_cross : ℝ) :
    speed_of_train = 63 → speed_of_man = 3 → time_to_cross = 32.997 →
    length_of_train speed_of_train speed_of_man time_to_cross = 549.95 :=
by
  intros h_train h_man h_time
  rw [h_train, h_man, h_time]
  norm_num
  sorry

end train_length_is_549_95_l2389_238989


namespace probability_selecting_cooking_l2389_238961

theorem probability_selecting_cooking :
  (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end probability_selecting_cooking_l2389_238961


namespace fraction_of_green_marbles_half_l2389_238913

-- Definitions based on given conditions
def initial_fraction (x : ℕ) : ℚ := 1 / 3

-- Number of blue, red, and green marbles initially
def blue_marbles (x : ℕ) : ℚ := initial_fraction x * x
def red_marbles (x : ℕ) : ℚ := initial_fraction x * x
def green_marbles (x : ℕ) : ℚ := initial_fraction x * x

-- Number of green marbles after doubling
def doubled_green_marbles (x : ℕ) : ℚ := 2 * green_marbles x

-- New total number of marbles
def new_total_marbles (x : ℕ) : ℚ := blue_marbles x + red_marbles x + doubled_green_marbles x

-- New fraction of green marbles after doubling
def new_fraction_of_green_marbles (x : ℕ) : ℚ := doubled_green_marbles x / new_total_marbles x

theorem fraction_of_green_marbles_half (x : ℕ) (hx : x > 0) :
  new_fraction_of_green_marbles x = 1 / 2 :=
by
  sorry

end fraction_of_green_marbles_half_l2389_238913


namespace range_of_a_l2389_238940

theorem range_of_a (a : ℝ)
  (A : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≥ 0})
  (B : Set ℝ := {x : ℝ | x ≥ a - 1})
  (H : A ∪ B = Set.univ) :
  a ≤ 2 :=
by
  sorry

end range_of_a_l2389_238940


namespace product_of_smallest_primes_l2389_238910

theorem product_of_smallest_primes :
  2 * 3 * 11 = 66 :=
by
  sorry

end product_of_smallest_primes_l2389_238910


namespace nina_money_l2389_238908

theorem nina_money :
  ∃ (m C : ℝ), 
    m = 6 * C ∧ 
    m = 8 * (C - 1) ∧ 
    m = 24 :=
by
  sorry

end nina_money_l2389_238908


namespace total_pens_bought_l2389_238962

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l2389_238962


namespace original_denominator_is_15_l2389_238972

theorem original_denominator_is_15
  (d : ℕ)
  (h1 : (6 : ℚ) / (d + 3) = 1 / 3)
  (h2 : 3 = 3) : d = 15 := -- h2 is trivial but included according to the problem condition
by
  sorry

end original_denominator_is_15_l2389_238972


namespace gray_region_area_l2389_238965

theorem gray_region_area (d_small r_large r_small π : ℝ) (h1 : d_small = 6)
    (h2 : r_large = 3 * r_small) (h3 : r_small = d_small / 2) :
    (π * r_large ^ 2 - π * r_small ^ 2) = 72 * π := 
by
  -- The proof will be filled here
  sorry

end gray_region_area_l2389_238965


namespace hyperbola_sufficient_condition_l2389_238900

-- Define the condition for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  (3 - k) * (k - 1) < 0

-- Lean 4 statement to prove that k > 3 is a sufficient condition for the given equation
theorem hyperbola_sufficient_condition (k : ℝ) (h : k > 3) :
  represents_hyperbola k :=
sorry

end hyperbola_sufficient_condition_l2389_238900


namespace avg_price_of_towels_l2389_238942

def towlesScenario (t1 t2 t3 : ℕ) (price1 price2 price3 : ℕ) : ℕ :=
  (t1 * price1 + t2 * price2 + t3 * price3) / (t1 + t2 + t3)

theorem avg_price_of_towels :
  towlesScenario 3 5 2 100 150 500 = 205 := by
  sorry

end avg_price_of_towels_l2389_238942


namespace domain_of_rational_func_l2389_238923

noncomputable def rational_func (x : ℝ) : ℝ := (2 * x ^ 3 - 3 * x ^ 2 + 5 * x - 1) / (x ^ 2 - 5 * x + 6)

theorem domain_of_rational_func : 
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ↔ (∃ y : ℝ, rational_func y = x) :=
by
  sorry

end domain_of_rational_func_l2389_238923


namespace f_5_5_l2389_238950

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_even (x : ℝ) : f x = f (-x) := sorry

lemma f_recurrence (x : ℝ) : f (x + 2) = - (1 / f x) := sorry

lemma f_interval (x : ℝ) (h : 2 ≤ x ∧ x ≤ 3) : f x = x := sorry

theorem f_5_5 : f 5.5 = 2.5 :=
by
  sorry

end f_5_5_l2389_238950
