import Mathlib

namespace arithmetic_sequence_a7_l162_162286

noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence_a7 
  (a : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 4 - a 1) / 3)
  (h_a1 : a 1 = 3)
  (h_a4 : a 4 = 5) : 
  a 7 = 7 :=
by
  sorry

end arithmetic_sequence_a7_l162_162286


namespace product_of_repeating_decimal_and_five_l162_162679

noncomputable def repeating_decimal : ℚ :=
  456 / 999

theorem product_of_repeating_decimal_and_five : 
  (repeating_decimal * 5) = 760 / 333 :=
by
  -- The proof is omitted.
  sorry

end product_of_repeating_decimal_and_five_l162_162679


namespace valid_parameterizations_l162_162614

theorem valid_parameterizations :
  (∀ t : ℝ, ∃ x y : ℝ, (x = 0 + 4 * t) ∧ (y = -4 + 8 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = 3 + 1 * t) ∧ (y = 2 + 2 * t) ∧ (y = 2 * x - 4)) ∧
  (∀ t : ℝ, ∃ x y : ℝ, (x = -1 + 2 * t) ∧ (y = -6 + 4 * t) ∧ (y = 2 * x - 4)) :=
by
  -- Proof goes here
  sorry

end valid_parameterizations_l162_162614


namespace sum_of_roots_eq_6_l162_162270

theorem sum_of_roots_eq_6 : ∀ (x1 x2 : ℝ), (x1 * x1 = x1 ∧ x1 * x2 = x2) → (x1 + x2 = 6) :=
by
   intro x1 x2 hx
   have H : x1 + x2 = 6 := sorry
   exact H

end sum_of_roots_eq_6_l162_162270


namespace sum_of_squares_l162_162452

theorem sum_of_squares (x y z : ℕ) (h1 : x + y + z = 30)
  (h2 : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 12) :
  x^2 + y^2 + z^2 = 504 :=
by
  sorry

end sum_of_squares_l162_162452


namespace range_of_k_l162_162395

open Set

variable {k : ℝ}

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

theorem range_of_k (h : (compl A) ∩ B k ≠ ∅) : 0 < k ∧ k < 3 := sorry

end range_of_k_l162_162395


namespace div_by_13_l162_162298

theorem div_by_13 (a b c : ℤ) (h : (a + b + c) % 13 = 0) : 
  (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) % 13 = 0 :=
by
  sorry

end div_by_13_l162_162298


namespace find_radius_closest_tenth_l162_162188

noncomputable def lattice_point_probability (d : ℝ) : ℝ := π * d^2

theorem find_radius_closest_tenth
  (square_area : ℝ)
  (probability : ℝ)
  (lattice_point_probability_eq : lattice_point_probability d = 1 / 4)
  (square_area_eq : square_area = 1000000)
  (probability_eq : probability = 1 / 4) :
  (d ≈ 0.3) :=
by
  have : π * d^2 = 1 / 4, from lattice_point_probability_eq
  sorry

end find_radius_closest_tenth_l162_162188


namespace workers_cut_down_correct_l162_162473

def initial_oak_trees : ℕ := 9
def remaining_oak_trees : ℕ := 7
def cut_down_oak_trees : ℕ := initial_oak_trees - remaining_oak_trees

theorem workers_cut_down_correct : cut_down_oak_trees = 2 := by
  sorry

end workers_cut_down_correct_l162_162473


namespace g_extreme_value_f_ge_g_l162_162246

noncomputable def f (x : ℝ) : ℝ := Real.exp (x + 1) - 2 / x + 1
noncomputable def g (x : ℝ) : ℝ := Real.log x / x + 2

theorem g_extreme_value :
  ∃ (x : ℝ), x = Real.exp 1 ∧ g x = 1 / Real.exp 1 + 2 :=
by sorry

theorem f_ge_g (x : ℝ) (hx : 0 < x) : f x >= g x :=
by sorry

end g_extreme_value_f_ge_g_l162_162246


namespace correct_calculation_l162_162949

theorem correct_calculation :
  -4^2 / (-2)^3 * (-1 / 8) = -1 / 4 := by
  sorry

end correct_calculation_l162_162949


namespace sum_of_three_consecutive_odd_integers_l162_162772

-- Define the variables and conditions
variables (a : ℤ) (h1 : (a + (a + 4) = 100))

-- Define the statement that needs to be proved
theorem sum_of_three_consecutive_odd_integers (ha : a = 48) : a + (a + 2) + (a + 4) = 150 := by
  sorry

end sum_of_three_consecutive_odd_integers_l162_162772


namespace sequence_general_term_l162_162718

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2) 
    (h_a₁ : S 1 = 1) (h_an : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) : 
  ∀ n, a n = 2 * n - 1 := 
by
  sorry

end sequence_general_term_l162_162718


namespace cousins_assignment_l162_162984

open Finset

def choose (n k : ℕ) : ℕ := nat.choose n k

theorem cousins_assignment (five_cousins : ℕ := 5) (rooms : ℕ := 4) (small_room_limit : ℕ := 2) :
  (∑ (r : ℕ) in {((choose 5 3) + (choose 5 3) + (choose 5 2 * choose 3 2 * 2)), 1}, r) = 80 :=
by
  sorry

end cousins_assignment_l162_162984


namespace problem1_problem2_l162_162895

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

end problem1_problem2_l162_162895


namespace min_a_for_decreasing_f_l162_162751

theorem min_a_for_decreasing_f {a : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 1 - a / (2 * Real.sqrt x) ≤ 0) →
  a ≥ 4 :=
sorry

end min_a_for_decreasing_f_l162_162751


namespace probability_other_side_green_l162_162659

-- Definitions based on the conditions
def Card : Type := ℕ
def num_cards : ℕ := 8
def blue_blue : ℕ := 4
def blue_green : ℕ := 2
def green_green : ℕ := 2

def total_green_sides : ℕ := (green_green * 2) + blue_green
def green_opposite_green_side : ℕ := green_green * 2

theorem probability_other_side_green (h_total_green_sides : total_green_sides = 6)
(h_green_opposite_green_side : green_opposite_green_side = 4) :
  (green_opposite_green_side / total_green_sides : ℚ) = 2 / 3 := 
by
  sorry

end probability_other_side_green_l162_162659


namespace value_of_expression_l162_162712

-- Define the hypothesis and the goal
theorem value_of_expression (x y : ℝ) (h : 3 * y - x^2 = -5) : 6 * y - 2 * x^2 - 6 = -16 := by
  sorry

end value_of_expression_l162_162712


namespace solve_for_a_l162_162297

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by 
  sorry

end solve_for_a_l162_162297


namespace greatest_possible_triangle_perimeter_l162_162110

noncomputable def triangle_perimeter (x : ℕ) : ℕ :=
  x + 2 * x + 18

theorem greatest_possible_triangle_perimeter :
  (∃ (x : ℕ), 7 ≤ x ∧ x < 18 ∧ ∀ y : ℕ, (7 ≤ y ∧ y < 18) → triangle_perimeter y ≤ triangle_perimeter x) ∧
  triangle_perimeter 17 = 69 :=
by
  sorry

end greatest_possible_triangle_perimeter_l162_162110


namespace map_point_to_result_l162_162617

def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem map_point_to_result :
  f 2 0 = (2, 2) :=
by
  unfold f
  simp

end map_point_to_result_l162_162617


namespace digit_matching_equalities_l162_162198

theorem digit_matching_equalities :
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ 99 → 0 ≤ b ∧ b ≤ 99 →
    ((a = 98 ∧ b = 1 ∧ (98 + 1)^2 = 100*98 + 1) ∨
     (a = 20 ∧ b = 25 ∧ (20 + 25)^2 = 100*20 + 25)) :=
by
  intros a b ha hb
  sorry

end digit_matching_equalities_l162_162198


namespace quadratic_no_real_roots_l162_162635

theorem quadratic_no_real_roots (a b : ℝ) (h : ∃ x : ℝ, x^2 + b * x + a = 0) : false :=
sorry

end quadratic_no_real_roots_l162_162635


namespace area_ratio_l162_162104

variables {rA rB : ℝ} (C_A C_B : ℝ)

#check C_A = 2 * Real.pi * rA
#check C_B = 2 * Real.pi * rB

theorem area_ratio (h : (60 / 360) * C_A = (40 / 360) * C_B) : (Real.pi * rA^2) / (Real.pi * rB^2) = 4 / 9 := by
  sorry

end area_ratio_l162_162104


namespace darwin_final_money_l162_162376

def initial_amount : ℕ := 600
def spent_on_gas (initial : ℕ) : ℕ := initial * 1 / 3
def remaining_after_gas (initial spent_gas : ℕ) : ℕ := initial - spent_gas
def spent_on_food (remaining : ℕ) : ℕ := remaining * 1 / 4
def final_amount (remaining spent_food : ℕ) : ℕ := remaining - spent_food

theorem darwin_final_money :
  final_amount (remaining_after_gas initial_amount (spent_on_gas initial_amount)) (spent_on_food (remaining_after_gas initial_amount (spent_on_gas initial_amount))) = 300 :=
by
  sorry

end darwin_final_money_l162_162376


namespace john_spent_l162_162293

-- Given definitions from the conditions.
def total_time_in_hours := 4
def additional_minutes := 35
def break_time_per_break := 10
def number_of_breaks := 5
def cost_per_5_minutes := 0.75
def playing_cost (total_time_in_hours additional_minutes break_time_per_break number_of_breaks : ℕ) 
  (cost_per_5_minutes : ℝ) : ℝ :=
  let total_minutes := total_time_in_hours * 60 + additional_minutes
  let break_time := number_of_breaks * break_time_per_break
  let actual_playing_time := total_minutes - break_time
  let number_of_intervals := actual_playing_time / 5
  number_of_intervals * cost_per_5_minutes

-- Statement to be proved.
theorem john_spent (total_time_in_hours := 4) (additional_minutes := 35) (break_time_per_break := 10) 
  (number_of_breaks := 5) (cost_per_5_minutes := 0.75) :
  playing_cost total_time_in_hours additional_minutes break_time_per_break number_of_breaks cost_per_5_minutes = 33.75 := 
by
  sorry

end john_spent_l162_162293


namespace james_weight_with_lifting_straps_l162_162113

theorem james_weight_with_lifting_straps
  (initial_weight : ℝ)
  (additional_weight_20m : ℝ)
  (additional_percent_10m : ℝ)
  (additional_percent_straps : ℝ)
  (distance_20m_weight : ℝ)
  (expected_weight_10m_straps : ℝ) :
  initial_weight = 300 →
  additional_weight_20m = 50 →
  additional_percent_10m = 0.3 →
  additional_percent_straps = 0.2 →
  distance_20m_weight = 350 →
  expected_weight_10m_straps = 546 →
  let increased_weight_20m := initial_weight + additional_weight_20m in
  let increased_weight_10m := increased_weight_20m + increased_weight_20m * additional_percent_10m in
  let final_weight := increased_weight_10m + increased_weight_10m * additional_percent_straps in
  final_weight = expected_weight_10m_straps :=
by
  intros h_initial_weight h_additional_weight_20m h_additional_percent_10m h_additional_percent_straps h_distance_20m_weight h_expected_weight_10m_straps
  let increased_weight_20m := initial_weight + additional_weight_20m
  let increased_weight_10m := increased_weight_20m + increased_weight_20m * additional_percent_10m
  let final_weight := increased_weight_10m + increased_weight_10m * additional_percent_straps
  have : final_weight = expected_weight_10m_straps, by
    rw [h_initial_weight, h_additional_weight_20m, h_additional_percent_10m, h_additional_percent_straps, h_distance_20m_weight, h_expected_weight_10m_straps]
    sorry
  exact this

end james_weight_with_lifting_straps_l162_162113


namespace least_possible_b_l162_162329

theorem least_possible_b (a b : ℕ) (h1 : a + b = 120) (h2 : (Prime a ∨ ∃ p : ℕ, Prime p ∧ a = 2 * p)) (h3 : Prime b) (h4 : a > b) : b = 7 :=
sorry

end least_possible_b_l162_162329


namespace pencils_removed_l162_162199

theorem pencils_removed (initial_pencils removed_pencils remaining_pencils : ℕ) 
  (h1 : initial_pencils = 87) 
  (h2 : remaining_pencils = 83) 
  (h3 : removed_pencils = initial_pencils - remaining_pencils) : 
  removed_pencils = 4 :=
sorry

end pencils_removed_l162_162199


namespace team_score_is_correct_l162_162930

-- Definitions based on given conditions
def connor_score : ℕ := 2
def amy_score : ℕ := connor_score + 4
def jason_score : ℕ := 2 * amy_score
def combined_score : ℕ := connor_score + amy_score + jason_score
def emily_score : ℕ := 3 * combined_score
def team_score : ℕ := connor_score + amy_score + jason_score + emily_score

-- Theorem stating team_score should be 80
theorem team_score_is_correct : team_score = 80 := by
  sorry

end team_score_is_correct_l162_162930


namespace negative_option_is_B_l162_162519

-- Define the options as constants
def optionA : ℤ := -( -2 )
def optionB : ℤ := (-1) ^ 2023
def optionC : ℤ := |(-1) ^ 2|
def optionD : ℤ := (-5) ^ 2

-- Prove that the negative number among the options is optionB
theorem negative_option_is_B : optionB = -1 := 
by
  rw [optionB]
  sorry

end negative_option_is_B_l162_162519


namespace not_exists_set_of_9_numbers_min_elements_l162_162782

theorem not_exists_set_of_9_numbers (s : Finset ℕ) 
  (h_len : s.card = 9) 
  (h_median : ∑ x in (s.filter (λ x, x ≤ 2)), 1 ≤ 5) 
  (h_other : ∑ x in (s.filter (λ x, x ≤ 13)), 1 ≤ 4) 
  (h_avg : ∑ x in s = 63) :
  False := sorry

theorem min_elements (n : ℕ) (h_nat: n ≥ 5) :
  ∃ s : Finset ℕ, s.card = 2 * n + 1 ∧
                  ∑ x in (s.filter (λ x, x ≤ 2)), 1 = n + 1 ∧ 
                  ∑ x in (s.filter (λ x, x ≤ 13)), 1 = n ∧
                  ∑ x in s = 14 * n + 7 := sorry

end not_exists_set_of_9_numbers_min_elements_l162_162782


namespace quad_condition_l162_162028

noncomputable def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x - 4 * a

theorem quad_condition (a : ℝ) : (-16 ≤ a ∧ a ≤ 0) → (∀ x : ℝ, quadratic a x > 0) ↔ (¬ ∃ x : ℝ, quadratic a x ≤ 0) := by
  sorry

end quad_condition_l162_162028


namespace number_of_people_in_family_l162_162511

-- Define the conditions
def planned_spending : ℝ := 15
def savings_percentage : ℝ := 0.40
def cost_per_orange : ℝ := 1.5

-- Define the proof target: the number of people in the family
theorem number_of_people_in_family : 
  planned_spending * savings_percentage / cost_per_orange = 4 := 
by
  -- sorry to skip the proof; this is for statement only
  sorry

end number_of_people_in_family_l162_162511


namespace reciprocal_of_neg_2023_l162_162621

-- Define the number and its proposed reciprocal
def x : ℤ := -2023
def r : ℚ := -1 / 2023

-- State the theorem that the reciprocal of x is r
theorem reciprocal_of_neg_2023: x * r = 1 := by
  sorry

end reciprocal_of_neg_2023_l162_162621


namespace find_a_to_make_f_odd_l162_162847

noncomputable def f (a : ℝ) (x : ℝ): ℝ := x^3 * (Real.log (Real.exp x + 1) + a * x)

theorem find_a_to_make_f_odd :
  (∃ a : ℝ, ∀ x : ℝ, f a (-x) = -f a x) ↔ a = -1/2 :=
by 
  sorry

end find_a_to_make_f_odd_l162_162847


namespace pirate_treasure_probability_l162_162035

noncomputable def probability (treasure: ℕ) (noTreasureNoTraps: ℕ) : Rational :=
  (Nat.binom 8 4) * ((1 / 3) ^ treasure) * ((1 / 2) ^ noTreasureNoTraps)

theorem pirate_treasure_probability :
  let treasure := 4
  let noTreasureNoTraps := 4 in
  probability treasure noTreasureNoTraps = 35 / 648 :=
by
  have natBinom : Nat.binom 8 4 = 70 := by sorry
  have fractionExpr1 : (1 / 3 : Rational) ^ treasure = 1 / 81 := by sorry
  have fractionExpr2 : (1 / 2 : Rational) ^ noTreasureNoTraps = 1 / 16 := by sorry
  have multiplyExpr : 70 * (1 / 81) * (1 / 16) = 35 / 648 := by sorry
  simp [probability, natBinom, fractionExpr1, fractionExpr2, multiplyExpr]
  exact rfl

end pirate_treasure_probability_l162_162035


namespace union_of_A_and_B_l162_162175

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem union_of_A_and_B :
  A ∪ B = {-1, 0, 1, 2, 4} :=
by
  sorry

end union_of_A_and_B_l162_162175


namespace tens_digit_8_pow_2023_l162_162529

theorem tens_digit_8_pow_2023 : (8 ^ 2023 % 100) / 10 % 10 = 1 := 
sorry

end tens_digit_8_pow_2023_l162_162529


namespace shaded_area_semicircle_rotation_l162_162386

theorem shaded_area_semicircle_rotation (R : ℝ) (α : ℝ)
(hR : R > 0)
(hα : α = π / 4) :
  let S_0 := (π * R^2) / 2 in
  ∃ x y a b : ℝ, 
    (x + b = S_0) ∧ 
    (b + a = S_0) ∧ 
    (x = a) ∧ 
    (a + y = π * R^2 / 2) :=
begin
  sorry
end

end shaded_area_semicircle_rotation_l162_162386


namespace exists_perfect_square_intersection_l162_162524

theorem exists_perfect_square_intersection : ∃ n : ℕ, n > 1 ∧ ∃ k : ℕ, (2^n - n) = k^2 :=
by sorry

end exists_perfect_square_intersection_l162_162524


namespace part1_max_value_part2_three_distinct_real_roots_l162_162705

def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem part1_max_value (m : ℝ) (h_max : ∀ x, f x m ≤ f 2 m) : m = 6 := by
  sorry

theorem part2_three_distinct_real_roots (a : ℝ) (h_m : (m = 6))
  (h_a : ∀ x₁ x₂ x₃ : ℝ, f x₁ m = a ∧ f x₂ m = a ∧ f x₃ m = a →
     x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) : 0 < a ∧ a < 32 := by
  sorry

end part1_max_value_part2_three_distinct_real_roots_l162_162705


namespace max_value_when_a_zero_exactly_one_zero_range_l162_162085

noncomputable def f (a x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * log x

theorem max_value_when_a_zero : 
  (∀ x > 0, f 0 x ≤ f 0 1) ∧ f 0 1 = -1 :=
by sorry

theorem exactly_one_zero_range : 
  (∀ a > 0, ∃! x > 0, f a x = 0) ∧ 
  (∀ a ≤ 0, ¬ ∃ x > 0, f a x = 0) :=
by sorry

end max_value_when_a_zero_exactly_one_zero_range_l162_162085


namespace bobby_initial_candy_count_l162_162367

theorem bobby_initial_candy_count (C : ℕ) (h : C + 4 + 14 = 51) : C = 33 :=
by
  sorry

end bobby_initial_candy_count_l162_162367


namespace smallest_m_l162_162300

noncomputable def f (x : ℝ) : ℝ := sorry

theorem smallest_m (f : ℝ → ℝ) (x y : ℝ) (hx : 0 ≤ x) (hy : y ≤ 1) (h_eq : f 0 = f 1) 
(h_lt : forall x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → |f x - f y| < |x - y|): 
|f x - f y| < 1 / 2 := 
sorry

end smallest_m_l162_162300


namespace parallel_line_with_intercept_sum_l162_162387

theorem parallel_line_with_intercept_sum (c : ℝ) :
  (∀ x y : ℝ, 2 * x + 3 * y + 5 = 0 → 2 * x + 3 * y + c = 0) ∧ 
  (-c / 3 - c / 2 = 6) → 
  (10 * x + 15 * y - 36 = 0) :=
by
  sorry

end parallel_line_with_intercept_sum_l162_162387


namespace quadratic_complete_square_l162_162478

theorem quadratic_complete_square (x p q : ℤ) 
  (h_eq : x^2 - 6 * x + 3 = 0) 
  (h_pq_form : x^2 - 6 * x + (p - x)^2 = q) 
  (h_int : ∀ t, t = p + q) : p + q = 3 := sorry

end quadratic_complete_square_l162_162478


namespace dennis_initial_money_l162_162358

def initial_money (shirt_cost: ℕ) (ten_dollar_bills: ℕ) (loose_coins: ℕ) : ℕ :=
  shirt_cost + (10 * ten_dollar_bills) + loose_coins

theorem dennis_initial_money : initial_money 27 2 3 = 50 :=
by 
  -- Here would go the proof steps based on the solution steps identified before
  sorry

end dennis_initial_money_l162_162358


namespace signs_of_x_and_y_l162_162105

theorem signs_of_x_and_y (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ -2) : x > 0 ∧ y > 0 :=
sorry

end signs_of_x_and_y_l162_162105


namespace fraction_identity_l162_162207

noncomputable def simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : ℝ :=
  (1 / (2 * a * b)) + (b / (4 * a))

theorem fraction_identity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  simplify_fraction a b h₁ h₂ = (2 + b^2) / (4 * a * b) :=
by sorry

end fraction_identity_l162_162207


namespace inequality_one_inequality_two_l162_162121

variable (a b c : ℝ)

-- First Inequality Proof Statement
theorem inequality_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 := 
sorry

-- Second Inequality Proof Statement
theorem inequality_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) : 
  (a^3 + b^3 + c^3 + 1/a + 1/b + 1/c) ≥ 2 * (a + b + c) := 
sorry

end inequality_one_inequality_two_l162_162121


namespace simultaneous_inequalities_l162_162899

theorem simultaneous_inequalities (x : ℝ) 
    (h1 : x^3 - 11 * x^2 + 10 * x < 0) 
    (h2 : x^3 - 12 * x^2 + 32 * x > 0) : 
    (1 < x ∧ x < 4) ∨ (8 < x ∧ x < 10) :=
sorry

end simultaneous_inequalities_l162_162899


namespace T_5_value_l162_162864

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + (1 / y)^m

theorem T_5_value (y : ℝ) (h : y + 1 / y = 5) : T y 5 = 2525 := 
by {
  sorry
}

end T_5_value_l162_162864


namespace volume_of_region_l162_162234

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region : 
  (∫ x in -∞..∞, ∫ y in -∞..∞, ∫ z in -∞..∞, if f x y z ≤ 6 then 1 else 0) = 24 := 
sorry

end volume_of_region_l162_162234


namespace min_value_expression_l162_162952

theorem min_value_expression (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 14) ∧ (∀ z : ℝ, (z = (x + 10) / Real.sqrt (x - 4)) → y ≤ z) := sorry

end min_value_expression_l162_162952


namespace find_value_of_expression_l162_162986

-- Define non-negative variables
variables (x y z : ℝ) 

-- Conditions
def cond1 := x ^ 2 + x * y + y ^ 2 / 3 = 25
def cond2 := y ^ 2 / 3 + z ^ 2 = 9
def cond3 := z ^ 2 + z * x + x ^ 2 = 16

-- Target statement to be proven
theorem find_value_of_expression (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 z x) : 
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
sorry

end find_value_of_expression_l162_162986


namespace acute_triangle_orthocenter_l162_162289

variables (A B C H : Point) (a b c h_a h_b h_c : Real)

def acute_triangle (α β γ : Point) : Prop := 
-- Definition that ensures triangle αβγ is acute
sorry

def orthocenter (α β γ ω : Point) : Prop := 
-- Definition that ω is the orthocenter of triangle αβγ 
sorry

def sides_of_triangle (α β γ : Point) : (Real × Real × Real) := 
-- Function that returns the side lengths of triangle αβγ as (a, b, c)
sorry

def altitudes_of_triangle (α β γ θ : Point) : (Real × Real × Real) := 
-- Function that returns the altitudes of triangle αβγ with orthocenter θ as (h_a, h_b, h_c)
sorry

theorem acute_triangle_orthocenter 
  (A B C H : Point)
  (a b c h_a h_b h_c : Real)
  (ht : acute_triangle A B C)
  (orth : orthocenter A B C H)
  (sides : sides_of_triangle A B C = (a, b, c))
  (alts : altitudes_of_triangle A B C H = (h_a, h_b, h_c)) :
  AH * h_a + BH * h_b + CH * h_c = (a^2 + b^2 + c^2) / 2 :=
by sorry


end acute_triangle_orthocenter_l162_162289


namespace negation_of_prop_original_l162_162882

-- Definitions and conditions as per the problem
def prop_original : Prop :=
  ∃ x : ℝ, x^2 + x + 1 ≤ 0

def prop_negation : Prop :=
  ∀ x : ℝ, x^2 + x + 1 > 0

-- The theorem states the mathematical equivalence
theorem negation_of_prop_original : ¬ prop_original ↔ prop_negation := 
sorry

end negation_of_prop_original_l162_162882


namespace probability_of_two_sunny_days_l162_162750

def prob_two_sunny_days (prob_sunny prob_rain : ℚ) (days : ℕ) : ℚ :=
  (days.choose 2) * (prob_sunny^2 * prob_rain^(days-2))

theorem probability_of_two_sunny_days :
  prob_two_sunny_days (2/5) (3/5) 3 = 36/125 :=
by 
  sorry

end probability_of_two_sunny_days_l162_162750


namespace valid_pairs_for_area_18_l162_162992

theorem valid_pairs_for_area_18 (w l : ℕ) (hw : 0 < w) (hl : 0 < l) (h_area : w * l = 18) (h_lt : w < l) :
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) :=
sorry

end valid_pairs_for_area_18_l162_162992


namespace rectangle_area_l162_162025

theorem rectangle_area (length_of_rectangle radius_of_circle side_of_square : ℝ)
  (h1 : length_of_rectangle = (2 / 5) * radius_of_circle)
  (h2 : radius_of_circle = side_of_square)
  (h3 : side_of_square * side_of_square = 1225)
  (breadth_of_rectangle : ℝ)
  (h4 : breadth_of_rectangle = 10) : 
  length_of_rectangle * breadth_of_rectangle = 140 := 
by 
  sorry

end rectangle_area_l162_162025


namespace quadratic_function_value_at_2_l162_162144

theorem quadratic_function_value_at_2 
  (a b c : ℝ) (h_a : a ≠ 0) 
  (h1 : 7 = a * (-3)^2 + b * (-3) + c)
  (h2 : 7 = a * (5)^2 + b * 5 + c)
  (h3 : -8 = c) :
  a * 2^2 + b * 2 + c = -8 := by 
  sorry

end quadratic_function_value_at_2_l162_162144


namespace remainder_of_3_pow_2023_mod_5_l162_162482

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l162_162482


namespace value_of_a_for_positive_root_l162_162061

theorem value_of_a_for_positive_root :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ (3*x - 1)/(x - 3) = a/(3 - x) - 1) → a = -8 :=
by
  sorry

end value_of_a_for_positive_root_l162_162061


namespace sum_of_first_2m_terms_l162_162064

variable (m : ℕ)
variable (S : ℕ → ℤ)

-- Conditions
axiom Sm : S m = 100
axiom S3m : S (3 * m) = -150

-- Theorem statement
theorem sum_of_first_2m_terms : S (2 * m) = 50 :=
by
  sorry

end sum_of_first_2m_terms_l162_162064


namespace verify_extrema_l162_162725

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * x^4 - 2 * x^3 + (11 / 2) * x^2 - 6 * x + (9 / 4)

theorem verify_extrema :
  f 1 = 0 ∧ f 2 = 1 ∧ f 3 = 0 := by
  sorry

end verify_extrema_l162_162725


namespace interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l162_162252

noncomputable def f (x k : ℝ) : ℝ := Real.log x - k * x + 1

theorem interval_increase_for_k_eq_2 :
  ∃ k : ℝ, k = 2 → 
  ∃ a b : ℝ, 0 < b ∧ b = 1 / 2 ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → (Real.log x - 2 * x + 1 < Real.log x - 2 * x + 1)) := 
sorry

theorem range_of_k_if_f_leq_0 :
  ∀ (k : ℝ), (∀ x : ℝ, 0 < x → Real.log x - k * x + 1 ≤ 0) →
  ∃ k_min : ℝ, k_min = 1 ∧ k ≥ k_min :=
sorry

end interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l162_162252


namespace optimal_direction_l162_162809

-- Define the conditions as hypotheses
variables (a : ℝ) (V_first V_second : ℝ) (d : ℝ)
variable (speed_rel : V_first = 2 * V_second)
variable (dist : d = a)

-- Create a theorem statement for the problem
theorem optimal_direction (H : d = a) (vel_rel : V_first = 2 * V_second) : true := 
  sorry

end optimal_direction_l162_162809


namespace reciprocal_of_neg_2023_l162_162623

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by sorry

end reciprocal_of_neg_2023_l162_162623


namespace min_value_of_f_l162_162835

noncomputable def f (x : ℝ) := x + 2 * Real.cos x

theorem min_value_of_f :
  ∀ (x : ℝ), -Real.pi / 2 ≤ x ∧ x ≤ 0 → f x ≥ f (-Real.pi / 2) :=
by
  intro x hx
  -- conditions are given, statement declared, but proof is not provided
  sorry

end min_value_of_f_l162_162835


namespace intersection_P_Q_l162_162608

def P : Set ℝ := { x | x^2 - 9 < 0 }
def Q : Set ℤ := { x | -1 ≤ x ∧ x ≤ 3 }

theorem intersection_P_Q : (P ∩ (coe '' Q)) = { x : ℝ | x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 } :=
by sorry

end intersection_P_Q_l162_162608


namespace problem1_problem2_l162_162173

-- Problem 1
theorem problem1 (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 0) :
  (x^2 + x) / (x^2 - 2 * x + 1) / (2 / (x - 1) - 1 / x) = x^2 / (x - 1) := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (hx1 : x > 0) :
  (2 * x + 1) / 3 - (5 * x - 1) / 2 < 1 ∧ 
  (5 * x - 1 < 3 * (x + 2)) →
  x = 1 ∨ x = 2 ∨ x = 3 := by
  sorry

end problem1_problem2_l162_162173


namespace fraction_work_AC_l162_162926

theorem fraction_work_AC (total_payment Rs B_payment : ℝ)
  (payment_AC : ℝ)
  (h1 : total_payment = 529)
  (h2 : B_payment = 12)
  (h3 : payment_AC = total_payment - B_payment) : 
  payment_AC / total_payment = 517 / 529 :=
by
  rw [h1, h2] at h3
  rw [h3]
  norm_num
  sorry

end fraction_work_AC_l162_162926


namespace total_workers_is_28_l162_162856

noncomputable def avg_salary_total : ℝ := 750
noncomputable def num_type_a : ℕ := 5
noncomputable def avg_salary_type_a : ℝ := 900
noncomputable def num_type_b : ℕ := 4
noncomputable def avg_salary_type_b : ℝ := 800
noncomputable def avg_salary_type_c : ℝ := 700

theorem total_workers_is_28 :
  ∃ (W : ℕ) (C : ℕ),
  W * avg_salary_total = num_type_a * avg_salary_type_a + num_type_b * avg_salary_type_b + C * avg_salary_type_c ∧
  W = num_type_a + num_type_b + C ∧
  W = 28 :=
by
  sorry

end total_workers_is_28_l162_162856


namespace find_value_of_c_l162_162224

theorem find_value_of_c (c : ℝ) (h1 : c > 0) (h2 : c + ⌊c⌋ = 23.2) : c = 11.7 :=
sorry

end find_value_of_c_l162_162224


namespace gas_pressure_in_final_container_l162_162365

variable (k : ℝ) (p_initial p_second p_final : ℝ) (v_initial v_second v_final v_half : ℝ)

theorem gas_pressure_in_final_container 
  (h1 : v_initial = 3.6)
  (h2 : p_initial = 6)
  (h3 : v_second = 7.2)
  (h4 : v_final = 3.6)
  (h5 : v_half = v_second / 2)
  (h6 : p_initial * v_initial = k)
  (h7 : p_second * v_second = k)
  (h8 : p_final * v_final = k) :
  p_final = 6 := 
sorry

end gas_pressure_in_final_container_l162_162365


namespace min_elements_l162_162779

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end min_elements_l162_162779


namespace point_on_line_l162_162215

theorem point_on_line (k : ℝ) (x y : ℝ) (h : x = -1/3 ∧ y = 4) (line_eq : 1 + 3 * k * x = -4 * y) : k = 17 :=
by
  rcases h with ⟨hx, hy⟩
  sorry

end point_on_line_l162_162215


namespace health_risk_probability_l162_162928

theorem health_risk_probability :
  let p := 26
  let q := 57
  p + q = 83 :=
by {
  sorry
}

end health_risk_probability_l162_162928


namespace no_set_of_9_numbers_l162_162784

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end no_set_of_9_numbers_l162_162784


namespace part1_part2_part3_l162_162066

-- Part 1: Proving a₁ for given a₃, p, and q
theorem part1 (a : ℕ → ℝ) (p q : ℝ) (h1 : p = (1/2)) (h2 : q = 2) 
  (h3 : a 3 = 41 / 20) (h4 : ∀ n, a (n + 1) = p * a n + q / a n) :
  a 1 = 1 ∨ a 1 = 4 := 
sorry

-- Part 2: Finding the sum Sₙ of the first n terms given a₁ and p * q = 0
theorem part2 (a : ℕ → ℝ) (p q : ℝ) (h1 : a 1 = 5) (h2 : p * q = 0) 
  (h3 : ∀ n, a (n + 1) = p * a n + q / a n) 
  (S : ℕ → ℝ) (n : ℕ) :
    S n = (25 * n + q * n + q - 25) / 10 ∨ 
    S n = (25 * n + q * n) / 10 ∨ 
    S n = (5 * (p^n - 1)) / (p - 1) ∨ 
    S n = 5 * n :=
sorry

-- Part 3: Proving the range of p given a₁, q and that the sequence is monotonically decreasing
theorem part3 (a : ℕ → ℝ) (p q : ℝ) (h1 : a 1 = 2) (h2 : q = 1) 
  (h3 : ∀ n, a (n + 1) = p * a n + q / a n) 
  (h4 : ∀ n, a (n + 1) < a n) :
  1/2 < p ∧ p < 3/4 :=
sorry

end part1_part2_part3_l162_162066


namespace remainder_product_191_193_197_mod_23_l162_162336

theorem remainder_product_191_193_197_mod_23 :
  (191 * 193 * 197) % 23 = 14 := by
  sorry

end remainder_product_191_193_197_mod_23_l162_162336


namespace butanoic_acid_molecular_weight_l162_162203

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_weight_butanoic_acid : ℝ :=
  4 * atomic_weight_C + 8 * atomic_weight_H + 2 * atomic_weight_O

theorem butanoic_acid_molecular_weight :
  molecular_weight_butanoic_acid = 88.104 :=
by
  -- proof not required
  sorry

end butanoic_acid_molecular_weight_l162_162203


namespace person_income_l162_162998

/-- If the income and expenditure of a person are in the ratio 15:8 and the savings are Rs. 7000, then the income of the person is Rs. 15000. -/
theorem person_income (x : ℝ) (income expenditure : ℝ) (savings : ℝ) 
  (h1 : income = 15 * x) 
  (h2 : expenditure = 8 * x) 
  (h3 : savings = income - expenditure) 
  (h4 : savings = 7000) : 
  income = 15000 := 
by 
  sorry

end person_income_l162_162998


namespace students_solved_only_B_l162_162816

variable (A B C : Prop)
variable (n x y b c d : ℕ)

-- Conditions given in the problem
axiom h1 : n = 25
axiom h2 : x + y + b + c + d = n
axiom h3 : b + d = 2 * (c + d)
axiom h4 : x = y + 1
axiom h5 : x + b + c = 2 * (b + c)

-- Theorem to be proved
theorem students_solved_only_B : b = 6 :=
by
  sorry

end students_solved_only_B_l162_162816


namespace least_not_lucky_multiple_of_6_l162_162913

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_not_lucky_multiple_of_6 : ∃ k : ℕ, k > 0 ∧ k % 6 = 0 ∧ ¬ is_lucky k ∧ ∀ m : ℕ, m > 0 ∧ m % 6 = 0 ∧ ¬ is_lucky m → k ≤ m :=
  sorry

end least_not_lucky_multiple_of_6_l162_162913


namespace eccentricity_of_ellipse_l162_162693

theorem eccentricity_of_ellipse (m n : ℝ) (h1 : 1 / m + 2 / n = 1) (h2 : 0 < m) (h3 : 0 < n) (h4 : m * n = 8) :
  let a := n
  let b := m
  let c := Real.sqrt (a^2 - b^2)
  let e := c / a
  e = Real.sqrt 3 / 2 := 
sorry

end eccentricity_of_ellipse_l162_162693


namespace frequency_of_middle_group_l162_162111

theorem frequency_of_middle_group :
  ∃ m : ℝ, m + (1/3) * m = 200 ∧ (1/3) * m = 50 :=
by
  sorry

end frequency_of_middle_group_l162_162111


namespace find_number_l162_162905

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
by
  sorry

end find_number_l162_162905


namespace max_d_for_range_of_fx_l162_162833

theorem max_d_for_range_of_fx : 
  ∀ (d : ℝ), (∃ x : ℝ, x^2 + 4*x + d = -3) → d ≤ 1 := 
by
  sorry

end max_d_for_range_of_fx_l162_162833


namespace power_of_10_digits_l162_162303

theorem power_of_10_digits (n : ℕ) (hn : n > 1) :
  (∃ k : ℕ, (2^(n-1) < 10^k ∧ 10^k < 2^n) ∨ (5^(n-1) < 10^k ∧ 10^k < 5^n)) ∧ ¬((∃ k : ℕ, 2^(n-1) < 10^k ∧ 10^k < 2^n) ∧ (∃ k : ℕ, 5^(n-1) < 10^k ∧ 10^k < 5^n)) :=
sorry

end power_of_10_digits_l162_162303


namespace december_revenue_is_20_over_7_times_average_l162_162500

variable (D : ℚ) (N : ℚ) (J : ℚ)

-- Conditions: Revenue in November and January
def november_revenue_condition : Prop := N = (3/5) * D
def january_revenue_condition : Prop := J = (1/6) * N

-- Theorem
theorem december_revenue_is_20_over_7_times_average (h1 : november_revenue_condition D N) (h2 : january_revenue_condition J N) :
  D = (20 / 7) * ((N + J) / 2) :=
by 
  sorry

end december_revenue_is_20_over_7_times_average_l162_162500


namespace range_of_m_intersection_l162_162956

theorem range_of_m_intersection (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, y - k * x - 1 = 0 ∧ (x^2 / 4) + (y^2 / m) = 1) ↔ (m ∈ Set.Ico 1 4 ∪ Set.Ioi 4) :=
by
  sorry

end range_of_m_intersection_l162_162956


namespace balanced_apple_trees_l162_162545

theorem balanced_apple_trees: 
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (x1 * y2 - x1 * y4 - x3 * y2 + x3 * y4 = 0) ∧
    (x2 * y1 - x2 * y3 - x4 * y1 + x4 * y3 = 0) ∧
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4) :=
  sorry

end balanced_apple_trees_l162_162545


namespace daily_coffee_machine_cost_l162_162290

def coffee_machine_cost := 200 -- $200
def discount := 20 -- $20
def daily_coffee_cost := 2 * 4 -- $8/day
def days_to_pay_off := 36 -- 36 days

theorem daily_coffee_machine_cost :
  (days_to_pay_off * daily_coffee_cost - (coffee_machine_cost - discount)) / days_to_pay_off = 3 := 
by
  -- Using the given conditions: 
  -- coffee_machine_cost = 200
  -- discount = 20
  -- daily_coffee_cost = 8
  -- days_to_pay_off = 36
  sorry

end daily_coffee_machine_cost_l162_162290


namespace balls_in_boxes_l162_162562

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), balls = 5 ∧ boxes = 3 → 
  ∃ (ways : ℕ), ways = 21 :=
by
  sorry

end balls_in_boxes_l162_162562


namespace Caitlin_age_l162_162929

theorem Caitlin_age (Aunt_Anna_age : ℕ) (Brianna_age : ℕ) (Caitlin_age : ℕ)
    (h1 : Aunt_Anna_age = 48)
    (h2 : Brianna_age = Aunt_Anna_age / 3)
    (h3 : Caitlin_age = Brianna_age - 6) : 
    Caitlin_age = 10 := by 
  -- proof here
  sorry

end Caitlin_age_l162_162929


namespace original_average_l162_162875

theorem original_average (n : ℕ) (k : ℕ) (new_avg : ℝ) 
  (h1 : n = 35) 
  (h2 : k = 5) 
  (h3 : new_avg = 125) : 
  (new_avg / k) = 25 :=
by
  rw [h2, h3]
  simp
  sorry

end original_average_l162_162875


namespace calculate_value_is_neg_seventeen_l162_162676

theorem calculate_value_is_neg_seventeen : -3^2 + (-2)^3 = -17 :=
by
  sorry

end calculate_value_is_neg_seventeen_l162_162676


namespace custom_op_subtraction_l162_162269

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_subtraction :
  (custom_op 4 2) - (custom_op 2 4) = -8 := by
  sorry

end custom_op_subtraction_l162_162269


namespace reciprocal_of_neg_five_l162_162325

theorem reciprocal_of_neg_five : ∃ b : ℚ, (-5) * b = 1 ∧ b = -1/5 :=
by
  sorry

end reciprocal_of_neg_five_l162_162325


namespace floor_diff_l162_162940

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l162_162940


namespace expression_not_equal_one_l162_162374

-- Definitions of the variables and the conditions
def a : ℝ := sorry  -- Non-zero real number a
def y : ℝ := sorry  -- Real number y

axiom h1 : a ≠ 0
axiom h2 : y ≠ a
axiom h3 : y ≠ -a

-- The main theorem statement
theorem expression_not_equal_one (h1 : a ≠ 0) (h2 : y ≠ a) (h3 : y ≠ -a) : 
  ( (a / (a - y) + y / (a + y)) / (y / (a - y) - a / (a + y)) ) ≠ 1 :=
sorry

end expression_not_equal_one_l162_162374


namespace solve_negative_integer_sum_l162_162001

theorem solve_negative_integer_sum (N : ℤ) (h1 : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end solve_negative_integer_sum_l162_162001


namespace choose_president_vp_and_committee_l162_162282

theorem choose_president_vp_and_committee :
  ∃ (n : ℕ) (k : ℕ), n = 10 ∧ k = 3 ∧ 
  let ways_to_choose_president := 10 in
  let ways_to_choose_vp := 9 in
  let ways_to_choose_committee := (Nat.choose 8 3) in
  ways_to_choose_president * ways_to_choose_vp * ways_to_choose_committee = 5040 :=
begin
  use [10, 3],
  simp [Nat.choose],
  sorry
end

end choose_president_vp_and_committee_l162_162282


namespace quadratic_equation_in_one_variable_l162_162498

def is_quadratic_in_one_variable (eq : String) : Prop :=
  match eq with
  | "2x^2 + 5y + 1 = 0" => False
  | "ax^2 + bx - c = 0" => ∃ (a b c : ℝ), a ≠ 0
  | "1/x^2 + x = 2" => False
  | "x^2 = 0" => True
  | _ => False

theorem quadratic_equation_in_one_variable :
  is_quadratic_in_one_variable "x^2 = 0" := by
  sorry

end quadratic_equation_in_one_variable_l162_162498


namespace polynomial_sum_l162_162590

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l162_162590


namespace geometric_sequence_correct_l162_162573

theorem geometric_sequence_correct (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 8)
  (h2 : a 2 * a 3 = -8)
  (h_geom : ∀ (n : ℕ), a (n + 1) = a n * r) :
  a 4 = -1 :=
by {
  sorry
}

end geometric_sequence_correct_l162_162573


namespace least_not_lucky_multiple_of_6_l162_162914

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_not_lucky_multiple_of_6 : ∃ k : ℕ, k > 0 ∧ k % 6 = 0 ∧ ¬ is_lucky k ∧ ∀ m : ℕ, m > 0 ∧ m % 6 = 0 ∧ ¬ is_lucky m → k ≤ m :=
  sorry

end least_not_lucky_multiple_of_6_l162_162914


namespace greatest_multiple_of_4_l162_162873

theorem greatest_multiple_of_4 (x : ℕ) (h₀ : 0 < x) (h₁ : x % 4 = 0) (h₂ : x^3 < 5000) : x ≤ 16 :=
by {
  -- Skipping the proof as per instructions.
  sorry,
}

example : ∃ x : ℕ, 0 < x ∧ x % 4 = 0 ∧ x^3 < 5000 ∧ x = 16 :=
by {
  use 16,
  split,
  -- Proof steps for 0 < 16
  exact nat.zero_lt_bit0 nat.zero_lt_bit0,
  split,
  -- Proof steps for 16 % 4 = 0
  norm_num,
  split,
  -- Proof steps for 16^3 < 5000
  norm_num,
  -- Proof that x = 16
  refl,
}

end greatest_multiple_of_4_l162_162873


namespace number_of_unique_products_l162_162020

-- Define the sets a and b
def setA : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}
def setB : Set ℕ := {2, 4, 6, 19, 21, 24, 27, 31, 35}

-- Define the number of unique products
def numUniqueProducts : ℕ := 405

-- Statement that needs to be proved
theorem number_of_unique_products :
  (∀ A1 ∈ setA, ∀ B ∈ setB, ∀ A2 ∈ setA, ∃ p, p = A1 * B * A2) ∧ 
  (∃ count, count = 45 * 9) ∧ 
  (∃ result, result = numUniqueProducts) :=
  by {
    sorry
  }

end number_of_unique_products_l162_162020


namespace cost_of_parakeet_l162_162476

theorem cost_of_parakeet
  (P Py K : ℕ) -- defining the costs of parakeet, puppy, and kitten
  (h1 : Py = 3 * P) -- puppy is three times the cost of parakeet
  (h2 : P = K / 2) -- parakeet is half the cost of kitten
  (h3 : 2 * Py + 2 * K + 3 * P = 130) -- total cost equation
  : P = 10 := 
sorry

end cost_of_parakeet_l162_162476


namespace floor_diff_l162_162939

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l162_162939


namespace simplify_expression_l162_162879

theorem simplify_expression (x : ℝ) 
  (h1 : x^2 - 4*x + 3 = (x-3)*(x-1))
  (h2 : x^2 - 6*x + 9 = (x-3)^2)
  (h3 : x^2 - 6*x + 8 = (x-2)*(x-4))
  (h4 : x^2 - 8*x + 15 = (x-3)*(x-5)) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = (x-1)*(x-5) / ((x-2)*(x-4)) :=
by
  sorry

end simplify_expression_l162_162879


namespace candy_sampling_percentage_l162_162277

theorem candy_sampling_percentage (total_percentage caught_percentage not_caught_percentage : ℝ) 
  (h1 : caught_percentage = 22 / 100) 
  (h2 : total_percentage = 24.444444444444443 / 100) 
  (h3 : not_caught_percentage = 2.444444444444443 / 100) :
  total_percentage = caught_percentage + not_caught_percentage :=
by
  sorry

end candy_sampling_percentage_l162_162277


namespace relation_between_gender_and_big_sciences_probability_at_least_one_female_l162_162616

variable (students : Nat)
variable (male_students female_students : Nat)
variable (big_sciences non_big_sciences : Nat)
variable (male_big_sciences male_non_big_sciences : Nat)
variable (female_big_sciences female_non_big_sciences : Nat)
variable (n : ℝ)

noncomputable def K_square (a b c d : ℕ) : ℝ :=
  let n := (a + b + c + d) in
  (n * ((a * d - b * c)^2)) / (↑a + ↑b) / (↑c + ↑d) / (↑a + ↑c) / (↑b + ↑d)

theorem relation_between_gender_and_big_sciences (
  h1 : students = 100
  h2 : male_students = 55
  h3 : female_students = 45
  h4 : big_sciences = 60
  h5 : non_big_sciences = 40
  h6 : male_big_sciences = 40
  h7 : male_non_big_sciences = 15
  h8 : female_big_sciences = 20
  h9 : female_non_big_sciences = 25
  h10: K_square 40 15 20 25 = 8.249
) : true :=
sorry

noncomputable def C (n k : ℕ) : ℕ :=
Nat.choose n k

theorem probability_at_least_one_female (
  h1: (C 6 2) = 15
  h2: (C 2 1 * C 4 1 + C 2 2) = 9
) : (9 / 15) = 3 / 5 :=
sorry

end relation_between_gender_and_big_sciences_probability_at_least_one_female_l162_162616


namespace weight_difference_l162_162117

open Real

def yellow_weight : ℝ := 0.6
def green_weight : ℝ := 0.4
def red_weight : ℝ := 0.8
def blue_weight : ℝ := 0.5

def weights : List ℝ := [yellow_weight, green_weight, red_weight, blue_weight]

theorem weight_difference : (List.maximum weights).getD 0 - (List.minimum weights).getD 0 = 0.4 :=
by
  sorry

end weight_difference_l162_162117


namespace no_primes_divisible_by_45_l162_162710

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_primes_divisible_by_45 : 
  ∀ p, is_prime p → ¬ (45 ∣ p) := 
by
  sorry

end no_primes_divisible_by_45_l162_162710


namespace james_can_lift_546_pounds_l162_162112

def initial_lift_20m : ℝ := 300
def increase_10m : ℝ := 0.30
def strap_increase : ℝ := 0.20
def additional_weight_20m : ℝ := 50
def final_lift_10m_with_straps : ℝ := 546

theorem james_can_lift_546_pounds :
  let initial_lift_10m := initial_lift_20m * (1 + increase_10m)
  let updated_lift_20m := initial_lift_20m + additional_weight_20m
  let ratio := initial_lift_10m / initial_lift_20m
  let updated_lift_10m := updated_lift_20m * ratio
  let lift_with_straps := updated_lift_10m * (1 + strap_increase)
  lift_with_straps = final_lift_10m_with_straps :=
by
  sorry

end james_can_lift_546_pounds_l162_162112


namespace problem_sqrt_conjecture_l162_162131

theorem problem_sqrt_conjecture (n : ℕ) (hn : 1 ≤ n) :
  sqrt (n + 1 / (n + 2)) = (n + 1) * sqrt (1 / (n + 2)) :=
by
  sorry

end problem_sqrt_conjecture_l162_162131


namespace remainder_3_pow_2023_mod_5_l162_162489

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l162_162489


namespace total_boxes_stacked_l162_162031

/-- Definitions used in conditions --/
def box_width : ℕ := 1
def box_length : ℕ := 1
def land_width : ℕ := 44
def land_length : ℕ := 35
def first_day_layers : ℕ := 7
def second_day_layers : ℕ := 3

/-- Theorem stating the number of boxes stacked in two days --/
theorem total_boxes_stacked : first_day_layers * (land_width * land_length) + second_day_layers * (land_width * land_length) = 15400 := by
  sorry

end total_boxes_stacked_l162_162031


namespace probability_personA_not_personB_l162_162335

theorem probability_personA_not_personB :
  let n := Nat.choose 5 3
  let m := Nat.choose 1 1 * Nat.choose 3 2
  (m / n : ℚ) = 3 / 10 :=
by
  -- Proof omitted
  sorry

end probability_personA_not_personB_l162_162335


namespace largest_interior_angle_of_triangle_l162_162314

theorem largest_interior_angle_of_triangle (a b c ext : ℝ)
    (h1 : a + b + c = 180)
    (h2 : a / 4 = b / 5)
    (h3 : a / 4 = c / 6)
    (h4 : c + 120 = a + 180) : c = 72 :=
by
  sorry

end largest_interior_angle_of_triangle_l162_162314


namespace range_of_m_for_real_roots_value_of_m_for_specific_roots_l162_162254

open Real

variable {m x : ℝ}

def quadratic (m : ℝ) (x : ℝ) := x^2 + 2*(m-1)*x + m^2 + 2 = 0
  
theorem range_of_m_for_real_roots (h : ∃ x : ℝ, quadratic m x) : m ≤ -1/2 :=
sorry

theorem value_of_m_for_specific_roots
  (h : quadratic m x)
  (Hroots : ∃ x1 x2 : ℝ, quadratic m x1 ∧ quadratic m x2 ∧ (x1 - x2)^2 = 18 - x1 * x2) :
  m = -2 :=
sorry

end range_of_m_for_real_roots_value_of_m_for_specific_roots_l162_162254


namespace other_root_l162_162443

-- Define the condition that one root of the quadratic equation is -3
def is_root (a b c : ℤ) (x : ℚ) : Prop := a * x^2 + b * x + c = 0

-- Define the quadratic equation 7x^2 + mx - 6 = 0
def quadratic_eq (m : ℤ) (x : ℚ) : Prop := is_root 7 m (-6) x

-- Prove that the other root is 2/7 given that one root is -3
theorem other_root (m : ℤ) (h : quadratic_eq m (-3)) : quadratic_eq m (2 / 7) :=
by
  sorry

end other_root_l162_162443


namespace shared_earnings_eq_27_l162_162990

theorem shared_earnings_eq_27
    (shoes_pairs : ℤ) (shoes_cost : ℤ) (shirts : ℤ) (shirts_cost : ℤ)
    (h1 : shoes_pairs = 6) (h2 : shoes_cost = 3)
    (h3 : shirts = 18) (h4 : shirts_cost = 2) :
    (shoes_pairs * shoes_cost + shirts * shirts_cost) / 2 = 27 := by
  sorry

end shared_earnings_eq_27_l162_162990


namespace sufficient_y_wages_l162_162922

noncomputable def days_sufficient_for_y_wages (Wx Wy : ℝ) (total_money : ℝ) : ℝ :=
  total_money / Wy

theorem sufficient_y_wages
  (Wx Wy : ℝ)
  (H1 : ∀(D : ℝ), total_money = D * Wx → D = 36 )
  (H2 : total_money = 20 * (Wx + Wy)) :
  days_sufficient_for_y_wages Wx Wy total_money = 45 := by
  sorry

end sufficient_y_wages_l162_162922


namespace fraction_spent_on_furniture_l162_162595

variable (original_savings : ℕ)
variable (tv_cost : ℕ)
variable (f : ℚ)

-- Defining the conditions
def conditions := original_savings = 500 ∧ tv_cost = 100 ∧
  f = (original_savings - tv_cost) / original_savings

-- The theorem we want to prove
theorem fraction_spent_on_furniture : conditions original_savings tv_cost f → f = 4 / 5 := by
  sorry

end fraction_spent_on_furniture_l162_162595


namespace income_growth_relation_l162_162609

-- Define all the conditions
def initial_income : ℝ := 1.3
def third_week_income : ℝ := 2
def growth_rate (x : ℝ) : ℝ := (1 + x)^2  -- Compound interest style growth over 2 weeks.

-- Theorem: proving the relationship given the conditions
theorem income_growth_relation (x : ℝ) : initial_income * growth_rate x = third_week_income :=
by
  unfold initial_income third_week_income growth_rate
  sorry  -- Proof not required.

end income_growth_relation_l162_162609


namespace find_f2_l162_162400

noncomputable def f (x : ℝ) : ℝ := (4*x + 2/x + 3) / 3

theorem find_f2 (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x - f (1 / x) = 2 * x + 1) : f 2 = 4 :=
  by
  sorry

end find_f2_l162_162400


namespace simplify_correct_l162_162742

open Polynomial

noncomputable def simplify_expression (y : ℚ) : Polynomial ℚ :=
  (3 * (Polynomial.C y) + 2) * (2 * (Polynomial.C y)^12 + 3 * (Polynomial.C y)^11 - (Polynomial.C y)^9 - (Polynomial.C y)^8)

theorem simplify_correct (y : ℚ) : 
  simplify_expression y = 6 * (Polynomial.C y)^13 + 13 * (Polynomial.C y)^12 + 6 * (Polynomial.C y)^11 - 3 * (Polynomial.C y)^10 - 5 * (Polynomial.C y)^9 - 2 * (Polynomial.C y)^8 := 
by 
  simp [simplify_expression]
  sorry

end simplify_correct_l162_162742


namespace number_of_divisors_of_2310_l162_162230

theorem number_of_divisors_of_2310 : 
  let n := 2310 in
  let prime_factors := (2, 1) :: (3, 1) :: (5, 1) :: (7, 1) :: (11, 1) :: [] in
  n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 →
  (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 32 :=
begin
  intro h,
  sorry
end

end number_of_divisors_of_2310_l162_162230


namespace bamboo_consumption_correct_l162_162672

-- Define the daily bamboo consumption for adult and baby pandas
def adult_daily_bamboo : ℕ := 138
def baby_daily_bamboo : ℕ := 50

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total bamboo consumed by an adult panda in a week
def adult_weekly_bamboo := adult_daily_bamboo * days_in_week

-- Define the total bamboo consumed by a baby panda in a week
def baby_weekly_bamboo := baby_daily_bamboo * days_in_week

-- Define the total bamboo consumed by both pandas in a week
def total_bamboo_consumed := adult_weekly_bamboo + baby_weekly_bamboo

-- The theorem states that the total bamboo consumption in a week is 1316 pounds
theorem bamboo_consumption_correct : total_bamboo_consumed = 1316 := by
  sorry

end bamboo_consumption_correct_l162_162672


namespace must_be_negative_when_x_is_negative_l162_162268

open Real

theorem must_be_negative_when_x_is_negative (x : ℝ) (h : x < 0) : x^3 < 0 ∧ -x^4 < 0 := 
by
  sorry

end must_be_negative_when_x_is_negative_l162_162268


namespace unique_nonzero_solution_l162_162337

theorem unique_nonzero_solution (x : ℝ) (h : x ≠ 0) : (3 * x)^3 = (9 * x)^2 → x = 3 :=
by
  sorry

end unique_nonzero_solution_l162_162337


namespace find_s_for_g_neg1_zero_l162_162585

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 :=
by
  sorry

end find_s_for_g_neg1_zero_l162_162585


namespace range_of_a_l162_162253

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else a * x^2 - x + 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ -1) ↔ (a ≥ 1/12) :=
by
  sorry

end range_of_a_l162_162253


namespace expression_bounds_l162_162296

noncomputable def expression (p q r s : ℝ) : ℝ :=
  Real.sqrt (p^2 + (2 - q)^2) + Real.sqrt (q^2 + (2 - r)^2) +
  Real.sqrt (r^2 + (2 - s)^2) + Real.sqrt (s^2 + (2 - p)^2)

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 2) (hq : 0 ≤ q ∧ q ≤ 2)
  (hr : 0 ≤ r ∧ r ≤ 2) (hs : 0 ≤ s ∧ s ≤ 2) : 
  4 * Real.sqrt 2 ≤ expression p q r s ∧ expression p q r s ≤ 8 :=
by
  sorry

end expression_bounds_l162_162296


namespace exists_m_divisible_l162_162695

-- Define the function f
def f (x : ℕ) : ℕ := 3 * x + 2

-- Define the 100th iterate of f
def f_iter (n : ℕ) : ℕ := 3^n

-- Define the condition that needs to be proven
theorem exists_m_divisible : ∃ m : ℕ, (3^100 * m + (3^100 - 1)) % 1988 = 0 :=
sorry

end exists_m_divisible_l162_162695


namespace inequality_ge_zero_l162_162738

theorem inequality_ge_zero (x y z : ℝ) : 
  4 * x * (x + y) * (x + z) * (x + y + z) + y^2 * z^2 ≥ 0 := 
sorry

end inequality_ge_zero_l162_162738


namespace geom_sequence_a_n_l162_162976

variable {a : ℕ → ℝ}

-- Given conditions
def is_geom_seq (a : ℕ → ℝ) : Prop :=
  |a 1| = 1 ∧ a 5 = -8 * a 2 ∧ a 5 > a 2

-- Statement to prove
theorem geom_sequence_a_n (h : is_geom_seq a) : ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end geom_sequence_a_n_l162_162976


namespace least_gumballs_to_ensure_five_gumballs_of_same_color_l162_162910

-- Define the number of gumballs for each color
def red_gumballs := 12
def white_gumballs := 10
def blue_gumballs := 11

-- Define the minimum number of gumballs required to ensure five of the same color
def min_gumballs_to_ensure_five_of_same_color := 13

-- Prove the question == answer given conditions
theorem least_gumballs_to_ensure_five_gumballs_of_same_color :
  (red_gumballs + white_gumballs + blue_gumballs) = 33 → min_gumballs_to_ensure_five_of_same_color = 13 :=
by {
  sorry
}

end least_gumballs_to_ensure_five_gumballs_of_same_color_l162_162910


namespace sum_of_distinct_prime_factors_of_462_l162_162157

theorem sum_of_distinct_prime_factors_of_462 : 
  ∑ p in {2, 3, 7, 11}, p = 23 := by
  have pf462 : unique_factorization_monoid.factors 462 = {2, 3, 7, 11} :=
    by sorry -- Demonstrate or assume correct factorization
  sorry -- Conclude the sum

end sum_of_distinct_prime_factors_of_462_l162_162157


namespace count_multiples_of_30_between_two_multiples_l162_162379

theorem count_multiples_of_30_between_two_multiples : 
  let lower := 900
  let upper := 27000
  let multiple := 30
  let count := (upper / multiple) - (lower / multiple) + 1
  count = 871 :=
by
  let lower := 900
  let upper := 27000
  let multiple := 30
  let count := (upper / multiple) - (lower / multiple) + 1
  sorry

end count_multiples_of_30_between_two_multiples_l162_162379


namespace find_angle_degree_l162_162832

theorem find_angle_degree (x : ℝ) (h : 90 - x = 0.4 * (180 - x)) : x = 30 := by
  sorry

end find_angle_degree_l162_162832


namespace num_two_wheelers_wheels_eq_two_l162_162108

variable (num_two_wheelers num_four_wheelers total_wheels : ℕ)

def total_wheels_eq : Prop :=
  2 * num_two_wheelers + 4 * num_four_wheelers = total_wheels

theorem num_two_wheelers_wheels_eq_two (h1 : num_four_wheelers = 13)
                                        (h2 : total_wheels = 54)
                                        (h_total_eq : total_wheels_eq num_two_wheelers num_four_wheelers total_wheels) :
  2 * num_two_wheelers = 2 :=
by
  unfold total_wheels_eq at h_total_eq
  sorry

end num_two_wheelers_wheels_eq_two_l162_162108


namespace solve_fabric_price_l162_162722

-- Defining the variables
variables (x y : ℕ)

-- Conditions as hypotheses
def condition1 := 7 * x = 9 * y
def condition2 := x - y = 36

-- Theorem statement to prove the system of equations
theorem solve_fabric_price (h1 : condition1 x y) (h2 : condition2 x y) :
  (7 * x = 9 * y) ∧ (x - y = 36) :=
by
  -- No proof is provided
  sorry

end solve_fabric_price_l162_162722


namespace fg_of_one_eq_onehundredandfive_l162_162267

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2)^3

theorem fg_of_one_eq_onehundredandfive : f (g 1) = 105 :=
by
  -- proof would go here
  sorry

end fg_of_one_eq_onehundredandfive_l162_162267


namespace remainder_3_pow_2023_mod_5_l162_162487

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l162_162487


namespace probability_first_greater_than_second_l162_162838

open ProbabilityTheory

/-- From cards numbered 1, 2, 3, 4, and 5, one card is drawn, replaced, and another card is drawn.
    The probability that the number on the first card drawn is greater than the number on the second
    card drawn is 2/5. -/
theorem probability_first_greater_than_second :
  let E := {1, 2, 3, 4, 5}
  in
  ∃ p : ℚ,
    (p = 2 / 5) ∧
    (∃ (draw : E × E),
      probability (
        draw.fst > draw.snd
      ) = p) :=
by
  sorry

end probability_first_greater_than_second_l162_162838


namespace floor_difference_l162_162943

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l162_162943


namespace polynomial_sum_l162_162587

def p (x : ℝ) := -4 * x^2 + 2 * x - 5
def q (x : ℝ) := -6 * x^2 + 4 * x - 9
def r (x : ℝ) := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l162_162587


namespace commute_times_abs_difference_l162_162514

theorem commute_times_abs_difference (x y : ℝ)
  (h_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (h_var : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) 
  : |x - y| = 4 :=
sorry

end commute_times_abs_difference_l162_162514


namespace Robie_boxes_with_him_l162_162602

-- Definition of the given conditions
def total_cards : Nat := 75
def cards_per_box : Nat := 10
def cards_not_placed : Nat := 5
def boxes_given_away : Nat := 2

-- Definition of the proof that Robie has 5 boxes with him
theorem Robie_boxes_with_him : ((total_cards - cards_not_placed) / cards_per_box) - boxes_given_away = 5 := by
  sorry

end Robie_boxes_with_him_l162_162602


namespace smaller_number_l162_162006

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 :=
by
  sorry

end smaller_number_l162_162006


namespace problem1_l162_162371

noncomputable def log6_7 : ℝ := Real.logb 6 7
noncomputable def log7_6 : ℝ := Real.logb 7 6

theorem problem1 : log6_7 > log7_6 := 
by
  sorry

end problem1_l162_162371


namespace train_boxcar_capacity_l162_162135

theorem train_boxcar_capacity :
  let red_boxcars := 3
  let blue_boxcars := 4
  let black_boxcars := 7
  let black_boxcar_capacity := 4000
  let blue_boxcar_capacity := 2 * black_boxcar_capacity
  let red_boxcar_capacity := 3 * blue_boxcar_capacity
  (red_boxcars * red_boxcar_capacity + blue_boxcars * blue_boxcar_capacity + black_boxcars * black_boxcar_capacity) = 132000 :=
by
  sorry

end train_boxcar_capacity_l162_162135


namespace symmetric_point_correct_l162_162656

-- Define the coordinates of point A
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D :=
  { x := -3
    y := -4
    z := 5 }

-- Define the symmetry function with respect to the plane xOz
def symmetric_xOz (p : Point3D) : Point3D :=
  { p with y := -p.y }

-- The expected coordinates of the point symmetric to A with respect to the plane xOz
def D_expected : Point3D :=
  { x := -3
    y := 4
    z := 5 }

-- Theorem stating that the symmetric point of A with respect to the plane xOz is D_expected
theorem symmetric_point_correct :
  symmetric_xOz A = D_expected := 
by 
  sorry

end symmetric_point_correct_l162_162656


namespace floor_sqrt_23_squared_l162_162219

theorem floor_sqrt_23_squared : (⌊Real.sqrt 23⌋) ^ 2 = 16 := by
  have h1 : (4:ℝ) < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < (5:ℝ) := sorry
  have h3 : (⌊Real.sqrt 23⌋ : ℝ) = 4 :=
    by sorry
  show 4^2 = 16 from by sorry

end floor_sqrt_23_squared_l162_162219


namespace people_came_later_l162_162238

theorem people_came_later (lollipop_ratio initial_people lollipops : ℕ) 
  (h1 : lollipop_ratio = 5) 
  (h2 : initial_people = 45) 
  (h3 : lollipops = 12) : 
  (lollipops * lollipop_ratio - initial_people) = 15 := by 
  sorry

end people_came_later_l162_162238


namespace convert_108_kmph_to_mps_l162_162535

-- Definitions and assumptions
def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * (1000 / 3600)

-- Theorem statement
theorem convert_108_kmph_to_mps : kmph_to_mps 108 = 30 := 
by
  sorry

end convert_108_kmph_to_mps_l162_162535


namespace line_intersection_l162_162820

-- Definitions for the parametric lines
def line1 (t : ℝ) : ℝ × ℝ := (3 + t, 2 * t)
def line2 (u : ℝ) : ℝ × ℝ := (-1 + 3 * u, 4 - u)

-- Statement that expresses the intersection point condition
theorem line_intersection :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (30 / 7, 18 / 7) :=
by
  sorry

end line_intersection_l162_162820


namespace floor_difference_l162_162945

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l162_162945


namespace base_2_representation_of_123_is_1111011_l162_162641

theorem base_2_representation_of_123_is_1111011 : 
  ∃ (n : ℕ), n = 123 ∧ n.toNatBinary = "1111011" :=
sorry

end base_2_representation_of_123_is_1111011_l162_162641


namespace acute_triangle_area_relation_l162_162447

open Real

variables (A B C R : ℝ)
variables (acute_triangle : Prop)
variables (S p_star : ℝ)

-- Conditions
axiom acute_triangle_condition : acute_triangle
axiom area_formula : S = (R^2 / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))
axiom semiperimeter_formula : p_star = (R / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))

-- Theorem to prove
theorem acute_triangle_area_relation (h : acute_triangle) : S = p_star * R := 
by {
  sorry 
}

end acute_triangle_area_relation_l162_162447


namespace quadratic_coeff_b_is_4_sqrt_15_l162_162596

theorem quadratic_coeff_b_is_4_sqrt_15 :
  ∃ m b : ℝ, (x^2 + bx + 72 = (x + m)^2 + 12) → (m = 2 * Real.sqrt 15) → (b = 4 * Real.sqrt 15) ∧ b > 0 :=
by
  -- Note: Proof not included as per the instruction.
  sorry

end quadratic_coeff_b_is_4_sqrt_15_l162_162596


namespace sum_of_distinct_prime_factors_of_462_l162_162166

theorem sum_of_distinct_prime_factors_of_462 :
  let factors := [2, 3, 7, 11] in -- The list of distinct prime factors of 462.
  factors.sum = 23 :=             -- We want to prove that their sum is 23.
by
  sorry

end sum_of_distinct_prime_factors_of_462_l162_162166


namespace three_units_away_from_neg_one_l162_162132

def is_three_units_away (x : ℝ) (y : ℝ) : Prop := abs (x - y) = 3

theorem three_units_away_from_neg_one :
  { x : ℝ | is_three_units_away x (-1) } = {2, -4} := 
by
  sorry

end three_units_away_from_neg_one_l162_162132


namespace apartments_per_floor_l162_162347

theorem apartments_per_floor (floors apartments_per: ℕ) (total_people : ℕ) (each_apartment_houses : ℕ)
    (h1 : floors = 25)
    (h2 : each_apartment_houses = 2)
    (h3 : total_people = 200)
    (h4 : floors * apartments_per * each_apartment_houses = total_people) :
    apartments_per = 4 := 
sorry

end apartments_per_floor_l162_162347


namespace jason_and_lisa_cards_l162_162114

-- Define the number of cards Jason originally had
def jason_original_cards (remaining: ℕ) (given_away: ℕ) : ℕ :=
  remaining + given_away

-- Define the number of cards Lisa originally had
def lisa_original_cards (remaining: ℕ) (given_away: ℕ) : ℕ :=
  remaining + given_away

-- State the main theorem to be proved
theorem jason_and_lisa_cards :
  jason_original_cards 4 9 + lisa_original_cards 7 15 = 35 :=
by
  sorry

end jason_and_lisa_cards_l162_162114


namespace travel_time_l162_162744

theorem travel_time (distance speed : ℝ) (h1 : distance = 300) (h2 : speed = 60) : 
  distance / speed = 5 := 
by
  sorry

end travel_time_l162_162744


namespace police_emergency_number_has_prime_divisor_gt_seven_l162_162805

theorem police_emergency_number_has_prime_divisor_gt_seven (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
by
  sorry

end police_emergency_number_has_prime_divisor_gt_seven_l162_162805


namespace smallest_five_digit_multiple_of_53_l162_162647

theorem smallest_five_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 53 = 0 ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_multiple_of_53_l162_162647


namespace problem_statement_l162_162434

noncomputable def a : ℝ := 6 * Real.sqrt 2
noncomputable def b : ℝ := 18 * Real.sqrt 2
noncomputable def c : ℝ := 6 * Real.sqrt 21
noncomputable def d : ℝ := 24 * Real.sqrt 2
noncomputable def e : ℝ := 48 * Real.sqrt 2
noncomputable def N : ℝ := 756 * Real.sqrt 10

axiom condition_a : a^2 + b^2 + c^2 + d^2 + e^2 = 504
axiom positive_numbers : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0

theorem problem_statement : N + a + b + c + d + e = 96 * Real.sqrt 2 + 6 * Real.sqrt 21 + 756 * Real.sqrt 10 :=
by
  -- We'll insert the proof here later
  sorry

end problem_statement_l162_162434


namespace squirrel_travel_distance_l162_162037

def squirrel_distance (height : ℕ) (circumference : ℕ) (rise_per_circuit : ℕ) : ℕ :=
  let circuits := height / rise_per_circuit
  let horizontal_distance := circuits * circumference
  Nat.sqrt (height * height + horizontal_distance * horizontal_distance)

theorem squirrel_travel_distance :
  (squirrel_distance 16 3 4) = 20 := by
  sorry

end squirrel_travel_distance_l162_162037


namespace inequality_abc_l162_162399

theorem inequality_abc (a b c : ℝ) (h1 : a ∈ Set.Icc (-1 : ℝ) 2) (h2 : b ∈ Set.Icc (-1 : ℝ) 2) (h3 : c ∈ Set.Icc (-1 : ℝ) 2) : 
  a * b * c + 4 ≥ a * b + b * c + c * a := 
sorry

end inequality_abc_l162_162399


namespace base_2_representation_of_123_l162_162644

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by sorry

end base_2_representation_of_123_l162_162644


namespace log_expression_simplifies_to_one_l162_162886

theorem log_expression_simplifies_to_one :
  (Real.log 5)^2 + Real.log 50 * Real.log 2 = 1 :=
by 
  sorry

end log_expression_simplifies_to_one_l162_162886


namespace inscribed_triangle_ratio_l162_162348

theorem inscribed_triangle_ratio 
  (a b c : ℝ) (h1: a = 10) (h2: b = 15) (h3: c = 19)
  (r' s' : ℝ) (h4 : r' < s') (h5 : r' + s' = a) :
  r' = 3 ∧ s' = 7 :=
by
  sorry

end inscribed_triangle_ratio_l162_162348


namespace inequality_to_prove_l162_162356

variable {r r1 r2 r3 m : ℝ}
variable {A B C : ℝ}

-- Conditions
-- r is the radius of an inscribed circle in a triangle
-- r1, r2, r3 are radii of circles each touching two sides of the triangle and the inscribed circle
-- m is a real number such that m >= 1/2

axiom r_radii_condition : r > 0
axiom r1_radii_condition : r1 > 0
axiom r2_radii_condition : r2 > 0
axiom r3_radii_condition : r3 > 0
axiom m_condition : m ≥ 1/2

-- Inequality to prove
theorem inequality_to_prove : 
  (r1 * r2) ^ m + (r2 * r3) ^ m + (r3 * r1) ^ m ≥ 3 * (r / 3) ^ (2 * m) := 
sorry

end inequality_to_prove_l162_162356


namespace wheres_waldo_books_published_l162_162634

theorem wheres_waldo_books_published (total_minutes : ℕ) (minutes_per_puzzle : ℕ) (puzzles_per_book : ℕ)
  (h1 : total_minutes = 1350) (h2 : minutes_per_puzzle = 3) (h3 : puzzles_per_book = 30) :
  total_minutes / minutes_per_puzzle / puzzles_per_book = 15 :=
by
  sorry

end wheres_waldo_books_published_l162_162634


namespace stutterer_square_number_unique_l162_162798

-- Definitions based on problem conditions
def is_stutterer (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧ (n / 100 = (n % 1000) / 100) ∧ ((n % 1000) % 100 = n % 10 * 10 + n % 10)

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The theorem statement
theorem stutterer_square_number_unique : ∃ n, is_stutterer n ∧ is_square n ∧ n = 7744 :=
by
  sorry

end stutterer_square_number_unique_l162_162798


namespace value_of_g_800_l162_162431

noncomputable def g : ℝ → ℝ :=
sorry

theorem value_of_g_800 (g_eq : ∀ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), g (x * y) = g x / (y^2))
  (g_at_1000 : g 1000 = 4) : g 800 = 625 / 2 :=
sorry

end value_of_g_800_l162_162431


namespace sum_of_rationals_l162_162499

theorem sum_of_rationals (r1 r2 : ℚ) : ∃ r : ℚ, r = r1 + r2 :=
sorry

end sum_of_rationals_l162_162499


namespace correct_total_cost_l162_162043

noncomputable def total_cost_after_discount : ℝ :=
  let sandwich_cost := 4
  let soda_cost := 3
  let sandwich_count := 7
  let soda_count := 5
  let total_items := sandwich_count + soda_count
  let total_cost := sandwich_count * sandwich_cost + soda_count * soda_cost
  let discount := if total_items ≥ 10 then 0.1 * total_cost else 0
  total_cost - discount

theorem correct_total_cost :
  total_cost_after_discount = 38.7 :=
by
  -- The proof would go here
  sorry

end correct_total_cost_l162_162043


namespace coordinate_minimizes_z_l162_162416

-- Definitions for conditions
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def equation_holds (x y : ℝ) : Prop := (1 / x) + (1 / (2 * y)) + (3 / (2 * x * y)) = 1

def z_def (x y : ℝ) : ℝ := x * y

-- Statement
theorem coordinate_minimizes_z (x y : ℝ) (h1 : in_first_quadrant x y) (h2 : equation_holds x y) :
    z_def x y = 9 / 2 ∧ (x = 3 ∧ y = 3 / 2) :=
    sorry

end coordinate_minimizes_z_l162_162416


namespace f_2a_eq_3_l162_162954

noncomputable def f (x : ℝ) : ℝ := 2^x + 1 / 2^x

theorem f_2a_eq_3 (a : ℝ) (h : f a = Real.sqrt 5) : f (2 * a) = 3 := by
  sorry

end f_2a_eq_3_l162_162954


namespace travis_apples_l162_162014

theorem travis_apples
  (price_per_box : ℕ)
  (num_apples_per_box : ℕ)
  (total_money : ℕ)
  (total_boxes : ℕ)
  (total_apples : ℕ)
  (h1 : price_per_box = 35)
  (h2 : num_apples_per_box = 50)
  (h3 : total_money = 7000)
  (h4 : total_boxes = total_money / price_per_box)
  (h5 : total_apples = total_boxes * num_apples_per_box) :
  total_apples = 10000 :=
sorry

end travis_apples_l162_162014


namespace negation_of_proposition_l162_162754

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, sin x ≥ -1) ↔ ∃ x : ℝ, sin x < -1 :=
by sorry

end negation_of_proposition_l162_162754


namespace evaluate_expression_l162_162685

def x : ℚ := 1 / 4
def y : ℚ := 1 / 3
def z : ℚ := 12

theorem evaluate_expression : x^3 * y^4 * z = 1 / 432 := 
by
  sorry

end evaluate_expression_l162_162685


namespace inscribed_circle_radius_l162_162981

theorem inscribed_circle_radius (a r : ℝ) (unit_square : a = 1)
  (touches_arc_AC : ∀ (x : ℝ × ℝ), x.1^2 + x.2^2 = (a - r)^2)
  (touches_arc_BD : ∀ (y : ℝ × ℝ), y.1^2 + y.2^2 = (a - r)^2)
  (touches_side_AB : ∀ (z : ℝ × ℝ), z.1 = r ∨ z.2 = r) :
  r = 3 / 8 := by sorry

end inscribed_circle_radius_l162_162981


namespace expectation_and_variance_binomial_l162_162063

noncomputable def X : ProbabilityDistributions.Binomial ℕ := {
  trials := 10,
  p := 0.6
}

theorem expectation_and_variance_binomial :
  (ProbabilityDistributions.Binomial.expectation X = 10 * 0.6) ∧
  (ProbabilityDistributions.Binomial.variance X = 10 * 0.6 * (1 - 0.6)) :=
by sorry

end expectation_and_variance_binomial_l162_162063


namespace floor_sq_minus_sq_floor_l162_162936

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l162_162936


namespace find_unknown_rate_l162_162915

def blankets_cost (num : ℕ) (rate : ℕ) (discount_tax : ℕ) (is_discount : Bool) : ℕ :=
  if is_discount then rate * (100 - discount_tax) / 100 * num
  else (rate * (100 + discount_tax) / 100) * num

def total_cost := blankets_cost 3 100 10 true +
                  blankets_cost 4 150 0 false +
                  blankets_cost 3 200 20 false

def avg_cost (total : ℕ) (num : ℕ) : ℕ :=
  total / num

theorem find_unknown_rate
  (unknown_rate : ℕ)
  (h1 : total_cost + 2 * unknown_rate = 1800)
  (h2 : avg_cost (total_cost + 2 * unknown_rate) 12 = 150) :
  unknown_rate = 105 :=
by
  sorry

end find_unknown_rate_l162_162915


namespace TJs_average_time_per_km_l162_162310

theorem TJs_average_time_per_km :
  let total_distance := 10
  let first_half_time := 20
  let second_half_time := 30
  let total_time := first_half_time + second_half_time
  let average_time_per_km := total_time / total_distance
  average_time_per_km = 5 :=
by
  let total_distance := 10
  let first_half_time := 20
  let second_half_time := 30
  let total_time := first_half_time + second_half_time
  let average_time_per_km := total_time / total_distance
  show average_time_per_km = 5 from
    sorry  -- proof goes here

end TJs_average_time_per_km_l162_162310


namespace price_per_cake_l162_162446

def number_of_cakes_per_day := 4
def number_of_working_days_per_week := 5
def total_amount_collected := 640
def number_of_weeks := 4

theorem price_per_cake :
  let total_cakes_per_week := number_of_cakes_per_day * number_of_working_days_per_week
  let total_cakes_in_four_weeks := total_cakes_per_week * number_of_weeks
  let price_per_cake := total_amount_collected / total_cakes_in_four_weeks
  price_per_cake = 8 := by
sorry

end price_per_cake_l162_162446


namespace max_rect_area_with_given_perimeter_l162_162768

-- Define the variables used in the problem
def length_of_wire := 12
def max_area (x : ℝ) := -(x - 3)^2 + 9

-- Lean Statement for the problem
theorem max_rect_area_with_given_perimeter : ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 6 → (x * (6 - x) ≤ A)) ∧ A = 9 :=
by
  sorry

end max_rect_area_with_given_perimeter_l162_162768


namespace inequality_l162_162124

noncomputable def x : ℝ := Real.sqrt 3
noncomputable def y : ℝ := Real.log 2 / Real.log 3
noncomputable def z : ℝ := Real.cos 2

theorem inequality : z < y ∧ y < x := by
  sorry

end inequality_l162_162124


namespace largest_six_consecutive_nonprime_under_50_l162_162258

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → (m = 1 ∨ m = n)

def consecutiveNonPrimes (m : ℕ) : Prop :=
  ∀ i : ℕ, i < 6 → ¬ isPrime (m + i)

theorem largest_six_consecutive_nonprime_under_50 (n : ℕ) :
  (n < 50 ∧ consecutiveNonPrimes n) →
  n + 5 = 35 :=
by
  intro h
  sorry

end largest_six_consecutive_nonprime_under_50_l162_162258


namespace paint_house_l162_162851

theorem paint_house (n s h : ℕ) (h_pos : 0 < h)
    (rate_eq : ∀ (x : ℕ), 0 < x → ∃ t : ℕ, x * t = n * h) :
    (n + s) * (nh / (n + s)) = n * h := 
sorry

end paint_house_l162_162851


namespace expectation_X_is_one_probability_A_at_least_4_out_of_5_l162_162891

open Probability

noncomputable def classAProb := 3/5
noncomputable def classBProb := 1/2

def X_distribution : PMF ℚ :=
  PMF.of_fintype (
    [(-10, 1/5), (0, 1/2), (10, 3/10)].to_finset
  ) sorry

def expectation_X : ℚ :=
  expectation X_distribution

theorem expectation_X_is_one :
  expectation_X = 1 :=
sorry

def rounds (i : ℕ) (outcomes : Fin i → ℚ) :=
(fin_fun (λ n, if outcomes n = 10 then 1 else 0))

def probability_A_wins (n : ℕ) :=
let p_A := (3/5) * (1/2) + (2/5) * (1/2) in
(probability (binomial n (p_A) ≥ 4) with sorry)

theorem probability_A_at_least_4_out_of_5 : probability_A_wins 5 = 2304 / 3125 :=
sorry

end expectation_X_is_one_probability_A_at_least_4_out_of_5_l162_162891


namespace yuan_representation_l162_162724

-- Define the essential conditions and numeric values
def receiving (amount : Int) : Int := amount
def spending (amount : Int) : Int := -amount

-- The main theorem statement
theorem yuan_representation :
  receiving 80 = 80 ∧ spending 50 = -50 → receiving (-50) = spending 50 :=
by
  intros h
  sorry

end yuan_representation_l162_162724


namespace orchestra_members_l162_162883

theorem orchestra_members : ∃ (x : ℕ), (130 < x) ∧ (x < 260) ∧ (x % 6 = 1) ∧ (x % 5 = 2) ∧ (x % 7 = 3) ∧ (x = 241) :=
by
  sorry

end orchestra_members_l162_162883


namespace sum_of_possible_values_of_k_l162_162730

open Complex

theorem sum_of_possible_values_of_k (x y z k : ℂ) (hxyz : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h : x / (1 - y + z) = k ∧ y / (1 - z + x) = k ∧ z / (1 - x + y) = k) : k = 1 :=
by
  sorry

end sum_of_possible_values_of_k_l162_162730


namespace ratio_of_sums_eq_19_over_17_l162_162533

theorem ratio_of_sums_eq_19_over_17 :
  let a₁ := 5
  let d₁ := 3
  let l₁ := 59
  let a₂ := 4
  let d₂ := 4
  let l₂ := 64
  let n₁ := 19  -- from solving l₁ = a₁ + (n₁ - 1) * d₁
  let n₂ := 16  -- from solving l₂ = a₂ + (n₂ - 1) * d₂
  let S₁ := n₁ * (a₁ + l₁) / 2
  let S₂ := n₂ * (a₂ + l₂) / 2
  S₁ / S₂ = 19 / 17 := by sorry

end ratio_of_sums_eq_19_over_17_l162_162533


namespace mark_purchased_cans_l162_162428

theorem mark_purchased_cans : ∀ (J M : ℕ), 
    (J = 40) → 
    (100 - J = 6 * M / 5) → 
    M = 27 := by
  sorry

end mark_purchased_cans_l162_162428


namespace find_other_man_age_l162_162140

variable (avg_age_men inc_age_man other_man_age avg_age_women total_age_increase : ℕ)

theorem find_other_man_age 
    (h1 : inc_age_man = 2) 
    (h2 : ∀ m, m = 8 * (avg_age_men + inc_age_man))
    (h3 : ∃ y, y = 22) 
    (h4 : ∀ w, w = 29) 
    (h5 : total_age_increase = 2 * avg_age_women - (22 + other_man_age)) :
  total_age_increase = 16 → other_man_age = 20 :=
by
  intros
  sorry

end find_other_man_age_l162_162140


namespace weight_of_mixture_is_correct_l162_162760

noncomputable def weight_mixture_kg (weight_per_liter_a weight_per_liter_b ratio_a ratio_b total_volume_liters : ℕ) : ℝ :=
  let volume_a := (ratio_a * total_volume_liters) / (ratio_a + ratio_b)
  let volume_b := (ratio_b * total_volume_liters) / (ratio_a + ratio_b)
  let weight_a := (volume_a * weight_per_liter_a) 
  let weight_b := (volume_b * weight_per_liter_b) 
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_mixture_kg 900 700 3 2 4 = 3.280 := 
sorry

end weight_of_mixture_is_correct_l162_162760


namespace minimize_S_n_l162_162547

noncomputable def S_n (n : ℕ) : ℝ := 2 * (n : ℝ) ^ 2 - 30 * (n : ℝ)

theorem minimize_S_n :
  ∃ n : ℕ, S_n n = 2 * (7 : ℝ) ^ 2 - 30 * (7 : ℝ) ∨ S_n n = 2 * (8 : ℝ) ^ 2 - 30 * (8 : ℝ) := by
  sorry

end minimize_S_n_l162_162547


namespace smaller_number_is_three_l162_162008

theorem smaller_number_is_three (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 36) : min x y = 3 :=
sorry

end smaller_number_is_three_l162_162008


namespace dennis_initial_money_l162_162359

def initial_money (shirt_cost: ℕ) (ten_dollar_bills: ℕ) (loose_coins: ℕ) : ℕ :=
  shirt_cost + (10 * ten_dollar_bills) + loose_coins

theorem dennis_initial_money : initial_money 27 2 3 = 50 :=
by 
  -- Here would go the proof steps based on the solution steps identified before
  sorry

end dennis_initial_money_l162_162359


namespace probability_two_win_one_lose_l162_162517

noncomputable def p_A : ℚ := 1 / 5
noncomputable def p_B : ℚ := 3 / 8
noncomputable def p_C : ℚ := 2 / 7

noncomputable def P_two_win_one_lose : ℚ :=
  p_A * p_B * (1 - p_C) +
  p_A * p_C * (1 - p_B) +
  p_B * p_C * (1 - p_A)

theorem probability_two_win_one_lose :
  P_two_win_one_lose = 49 / 280 :=
by
  sorry

end probability_two_win_one_lose_l162_162517


namespace sandy_books_cost_l162_162603

theorem sandy_books_cost :
  ∀ (x : ℕ),
  (1280 + 880) / (x + 55) = 18 → 
  x = 65 :=
by
  intros x h
  sorry

end sandy_books_cost_l162_162603


namespace largest_result_among_expressions_l162_162197

def E1 : ℕ := 992 * 999 + 999
def E2 : ℕ := 993 * 998 + 998
def E3 : ℕ := 994 * 997 + 997
def E4 : ℕ := 995 * 996 + 996

theorem largest_result_among_expressions : E4 > E1 ∧ E4 > E2 ∧ E4 > E3 :=
by sorry

end largest_result_among_expressions_l162_162197


namespace pow_mod_remainder_l162_162492

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l162_162492


namespace complex_proof_problem_l162_162233

theorem complex_proof_problem (i : ℂ) (h1 : i^2 = -1) :
  (i^2 + i^3 + i^4) / (1 - i) = (1 / 2) - (1 / 2) * i :=
by
  -- Proof will be provided here
  sorry

end complex_proof_problem_l162_162233


namespace postit_notes_area_l162_162149

theorem postit_notes_area (length width adhesive_len : ℝ) (num_notes : ℕ)
  (h_length : length = 9.4) (h_width : width = 3.7) (h_adh_len : adhesive_len = 0.6) (h_num_notes : num_notes = 15) :
  (length + (length - adhesive_len) * (num_notes - 1)) * width = 490.62 :=
by
  rw [h_length, h_width, h_adh_len, h_num_notes]
  sorry

end postit_notes_area_l162_162149


namespace total_area_is_71_l162_162210

noncomputable def area_of_combined_regions 
  (PQ QR RS TU : ℕ) 
  (PQRSTU_is_rectangle : true) 
  (right_angles : true): ℕ :=
  let Area_PQRSTU := PQ * QR
  let VU := TU - PQ
  let WT := TU - RS
  let Area_triangle_PVU := (1 / 2) * VU * PQ
  let Area_triangle_RWT := (1 / 2) * WT * RS
  Area_PQRSTU + Area_triangle_PVU + Area_triangle_RWT

theorem total_area_is_71
  (PQ QR RS TU : ℕ) 
  (h1 : PQ = 8)
  (h2 : QR = 6)
  (h3 : RS = 5)
  (h4 : TU = 10)
  (PQRSTU_is_rectangle : true)
  (right_angles : true) :
  area_of_combined_regions PQ QR RS TU PQRSTU_is_rectangle right_angles = 71 :=
by
  -- The proof is omitted as per the instructions
  sorry

end total_area_is_71_l162_162210


namespace B_days_to_complete_work_l162_162791

theorem B_days_to_complete_work 
  (W : ℝ) -- Define the amount of work
  (A_rate : ℝ := W / 15) -- A can complete the work in 15 days
  (B_days : ℝ) -- B can complete the work in B_days days
  (B_rate : ℝ := W / B_days) -- B's rate of work
  (total_days : ℝ := 12) -- Total days to complete the work
  (A_days_after_B_leaves : ℝ := 10) -- Days A works alone after B leaves
  (work_done_together : ℝ := 2 * (A_rate + B_rate)) -- Work done together in 2 days
  (work_done_by_A : ℝ := 10 * A_rate) -- Work done by A alone in 10 days
  (total_work_done : ℝ := work_done_together + work_done_by_A) -- Total work done
  (h_total_work_done : total_work_done = W) -- Total work equals W
  : B_days = 10 :=
sorry

end B_days_to_complete_work_l162_162791


namespace joan_gave_melanie_apples_l162_162292

theorem joan_gave_melanie_apples (original_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : original_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  sorry

end joan_gave_melanie_apples_l162_162292


namespace robert_balls_l162_162308

theorem robert_balls (R T : ℕ) (hR : R = 25) (hT : T = 40 / 2) : R + T = 45 :=
by
  sorry

end robert_balls_l162_162308


namespace any_power_ends_in_12890625_l162_162987

theorem any_power_ends_in_12890625 (a : ℕ) (m k : ℕ) (h : a = 10^m * k + 12890625) : ∀ (n : ℕ), 0 < n → ((a ^ n) % 10^8 = 12890625 % 10^8) :=
by
  intros
  sorry

end any_power_ends_in_12890625_l162_162987


namespace fraction_identity_l162_162208

noncomputable def simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : ℝ :=
  (1 / (2 * a * b)) + (b / (4 * a))

theorem fraction_identity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  simplify_fraction a b h₁ h₂ = (2 + b^2) / (4 * a * b) :=
by sorry

end fraction_identity_l162_162208


namespace line_intersects_circle_l162_162527

theorem line_intersects_circle (a : ℝ) (h : a ≠ 0) :
  ∃ p : ℝ × ℝ, (p.1 ^ 2 + p.2 ^ 2 = 9) ∧ (a * p.1 - p.2 + 2 * a = 0) :=
by
  sorry

end line_intersects_circle_l162_162527


namespace fish_per_bowl_l162_162762

theorem fish_per_bowl (num_bowls num_fish : ℕ) (h1 : num_bowls = 261) (h2 : num_fish = 6003) :
  num_fish / num_bowls = 23 :=
by {
  sorry
}

end fish_per_bowl_l162_162762


namespace solve_for_a_l162_162757

def quadratic_has_roots (a x1 x2 : ℝ) : Prop :=
  x1 + x2 = a ∧ x1 * x2 = -6 * a^2

theorem solve_for_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : quadratic_has_roots a x1 x2) (h3 : x2 - x1 = 10) : a = 2 :=
by
  sorry

end solve_for_a_l162_162757


namespace Amanda_second_day_tickets_l162_162041

/-- Amanda's ticket sales problem set up -/
def Amanda_total_tickets := 80
def Amanda_first_day_tickets := 5 * 4
def Amanda_third_day_tickets := 28

theorem Amanda_second_day_tickets :
  ∃ (tickets_sold_second_day : ℕ), tickets_sold_second_day = 32 :=
by
  let first_day := Amanda_first_day_tickets
  let third_day := Amanda_third_day_tickets
  let needed_before_third := Amanda_total_tickets - third_day
  let second_day := needed_before_third - first_day
  use second_day
  sorry

end Amanda_second_day_tickets_l162_162041


namespace hens_on_farm_l162_162178

theorem hens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H + R = 75) : H = 67 :=
by
  sorry

end hens_on_farm_l162_162178


namespace math_problem_l162_162870

theorem math_problem (n : ℕ) (h : n > 0) : 
  1957 ∣ (1721^(2*n) - 73^(2*n) - 521^(2*n) + 212^(2*n)) :=
sorry

end math_problem_l162_162870


namespace problem_expression_value_l162_162651

theorem problem_expression_value :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 : ℤ) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 : ℤ) = 6608 :=
by
  sorry

end problem_expression_value_l162_162651


namespace seventy_five_inverse_mod_seventy_six_l162_162536

-- Lean 4 statement for the problem.
theorem seventy_five_inverse_mod_seventy_six : (75 : ℤ) * 75 % 76 = 1 :=
by
  sorry

end seventy_five_inverse_mod_seventy_six_l162_162536


namespace charles_whistles_l162_162605

theorem charles_whistles (S C : ℕ) (h1 : S = 45) (h2 : S = C + 32) : C = 13 := 
by
  sorry

end charles_whistles_l162_162605


namespace relationship_depends_on_b_l162_162761

theorem relationship_depends_on_b (a b : ℝ) : 
  (a + b > a - b ∨ a + b < a - b ∨ a + b = a - b) ↔ (b > 0 ∨ b < 0 ∨ b = 0) :=
by
  sorry

end relationship_depends_on_b_l162_162761


namespace jack_keeps_deers_weight_is_correct_l162_162581

-- Define conditions
def monthly_hunt_count : Float := 7.5
def fraction_of_year_hunting_season : Float := 1 / 3
def deers_per_hunt : Float := 2.5
def weight_per_deer : Float := 600
def weight_kept_per_deer : Float := 0.65

-- Prove the total weight of the deer Jack keeps
theorem jack_keeps_deers_weight_is_correct :
  (12 * fraction_of_year_hunting_season) * monthly_hunt_count * deers_per_hunt * weight_per_deer * weight_kept_per_deer = 29250 :=
by
  sorry

end jack_keeps_deers_weight_is_correct_l162_162581


namespace value_of_expression_l162_162497

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x - 4)^2 = 100 :=
by
  -- Given the hypothesis h: x = -2
  -- Need to show: (3 * x - 4)^2 = 100
  sorry

end value_of_expression_l162_162497


namespace bird_families_flew_to_Asia_l162_162339

-- Variables/Parameters
variable (A : ℕ) (X : ℕ)
axiom hA : A = 47
axiom hX : X = A + 47

-- Theorem Statement
theorem bird_families_flew_to_Asia : X = 94 :=
by
  sorry

end bird_families_flew_to_Asia_l162_162339


namespace janet_time_to_home_l162_162977

-- Janet's initial and final positions
def initial_position : ℕ × ℕ := (0, 0) -- (x, y)
def north_blocks : ℕ := 3
def west_multiplier : ℕ := 7
def south_blocks : ℕ := 8
def east_multiplier : ℕ := 2
def speed_blocks_per_minute : ℕ := 2

def west_blocks : ℕ := west_multiplier * north_blocks
def east_blocks : ℕ := east_multiplier * south_blocks

-- Net movement calculations
def net_south_blocks : ℕ := south_blocks - north_blocks
def net_west_blocks : ℕ := west_blocks - east_blocks

-- Time calculation
def total_blocks_to_home : ℕ := net_south_blocks + net_west_blocks
def time_to_home : ℕ := total_blocks_to_home / speed_blocks_per_minute

theorem janet_time_to_home : time_to_home = 5 := by
  -- Proof goes here
  sorry

end janet_time_to_home_l162_162977


namespace definite_integral_ln_l162_162383

open Real

theorem definite_integral_ln (a b : ℝ) (h₁ : a = 1) (h₂ : b = exp 1) :
  ∫ x in a..b, (1 + log x) = exp 1 := by
  sorry

end definite_integral_ln_l162_162383


namespace grill_ran_for_16_hours_l162_162177

def coals_burn_time_A (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 15 * 20)) 0

def coals_burn_time_B (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 10 * 30)) 0

def total_grill_time (bags_A bags_B : List ℕ) : ℕ :=
  coals_burn_time_A bags_A + coals_burn_time_B bags_B

def bags_A : List ℕ := [60, 75, 45]
def bags_B : List ℕ := [50, 70, 40, 80]

theorem grill_ran_for_16_hours :
  total_grill_time bags_A bags_B = 960 / 60 :=
by
  unfold total_grill_time coals_burn_time_A coals_burn_time_B
  unfold bags_A bags_B
  norm_num
  sorry

end grill_ran_for_16_hours_l162_162177


namespace balls_in_boxes_l162_162561

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), balls = 5 ∧ boxes = 3 → 
  ∃ (ways : ℕ), ways = 21 :=
by
  sorry

end balls_in_boxes_l162_162561


namespace probability_interval_l162_162242

open ProbabilityTheory

-- Define the normal random variable X with mean 5 and variance 4
def X : ProbabilityMassFunction ℝ := ⟨λ x, pdf (Normal 5 (sqrt 4)) x⟩

-- State the property to prove: P(1 < X ≤ 7) = 0.9759
theorem probability_interval :
  (ProbabilityMassFunction.prob X (λ x, 1 < x ∧ x ≤ 7)) = 0.9759 :=
sorry

end probability_interval_l162_162242


namespace floor_sqrt_23_squared_l162_162218

theorem floor_sqrt_23_squared : (⌊Real.sqrt 23⌋) ^ 2 = 16 := by
  have h1 : (4:ℝ) < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < (5:ℝ) := sorry
  have h3 : (⌊Real.sqrt 23⌋ : ℝ) = 4 :=
    by sorry
  show 4^2 = 16 from by sorry

end floor_sqrt_23_squared_l162_162218


namespace negative_integer_solution_l162_162003

theorem negative_integer_solution (N : ℤ) (h : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end negative_integer_solution_l162_162003


namespace largest_uncovered_squares_l162_162673

theorem largest_uncovered_squares (board_size : ℕ) (total_squares : ℕ) (domino_size : ℕ) 
  (odd_property : ∀ (n : ℕ), n % 2 = 1 → (n - domino_size) % 2 = 1)
  (can_place_more : ∀ (placed_squares odd_squares : ℕ), placed_squares + domino_size ≤ total_squares → odd_squares - domino_size % 2 = 1 → odd_squares ≥ 0)
  : ∃ max_uncovered : ℕ, max_uncovered = 7 := by
  sorry

end largest_uncovered_squares_l162_162673


namespace time_between_four_and_five_straight_line_l162_162200

theorem time_between_four_and_five_straight_line :
  ∃ t : ℚ, t = 21 + 9/11 ∨ t = 54 + 6/11 :=
by
  sorry

end time_between_four_and_five_straight_line_l162_162200


namespace S₉_eq_81_l162_162955

variable (aₙ : ℕ → ℕ) (S : ℕ → ℕ)
variable (n : ℕ)
variable (a₁ d : ℕ)

-- Conditions
axiom S₃_eq_9 : S 3 = 9
axiom S₆_eq_36 : S 6 = 36
axiom S_n_def : ∀ n, S n = n * a₁ + n * (n - 1) / 2 * d

-- Proof obligation
theorem S₉_eq_81 : S 9 = 81 :=
by
  sorry

end S₉_eq_81_l162_162955


namespace task_force_at_least_two_executives_l162_162460

open Nat

theorem task_force_at_least_two_executives :
  let total_members := 12
  let executives := 5
  let total_task_forces := Nat.choose total_members 5
  let zero_executive_forces := Nat.choose (total_members - executives) 5
  let one_executive_forces := Nat.choose executives 1 * Nat.choose (total_members - executives) 4
  total_task_forces - (zero_executive_forces + one_executive_forces) = 596 :=
by
  intros total_members executives total_task_forces zero_executive_forces one_executive_forces
  let total_members := 12
  let executives := 5
  let total_task_forces := Nat.choose total_members 5
  let zero_executive_forces := Nat.choose (total_members - executives) 5
  let one_executive_forces := Nat.choose executives 1 * Nat.choose (total_members - executives) 4
  sorry

end task_force_at_least_two_executives_l162_162460


namespace number_of_turns_to_wind_tape_l162_162096

theorem number_of_turns_to_wind_tape (D δ L : ℝ) 
(hD : D = 22) 
(hδ : δ = 0.018) 
(hL : L = 90000) : 
∃ n : ℕ, n = 791 := 
sorry

end number_of_turns_to_wind_tape_l162_162096


namespace woman_born_second_half_20th_century_l162_162925

theorem woman_born_second_half_20th_century (x : ℕ) (hx : 45 < x ∧ x < 50) (h_year : x * x = 2025) :
  x * x - x = 1980 :=
by {
  -- Add the crux of the problem here.
  sorry
}

end woman_born_second_half_20th_century_l162_162925


namespace lunks_to_apples_l162_162264

theorem lunks_to_apples :
  (∀ (a b c d e f : ℕ), (7 * b = 4 * a) → (3 * d = 5 * c) → c = 24 → f * e = d → e = 27) :=
by sorry

end lunks_to_apples_l162_162264


namespace total_oranges_picked_l162_162598

theorem total_oranges_picked : 
  let M := 100 in
  let T := 3 * M in
  let W := 70 in
  M + T + W = 470 :=
by
  let M := 100
  let T := 3 * M
  let W := 70
  show M + T + W = 470
  sorry

end total_oranges_picked_l162_162598


namespace harry_friday_speed_l162_162708

theorem harry_friday_speed :
  let monday_speed := 10
  let tuesday_thursday_speed := monday_speed + monday_speed * (50 / 100)
  let friday_speed := tuesday_thursday_speed + tuesday_thursday_speed * (60 / 100)
  friday_speed = 24 :=
by
  sorry

end harry_friday_speed_l162_162708


namespace avg_diff_condition_l162_162414

variable (a b c : ℝ)

theorem avg_diff_condition (h1 : (a + b) / 2 = 110) (h2 : (b + c) / 2 = 150) : a - c = -80 :=
by
  sorry

end avg_diff_condition_l162_162414


namespace florist_sold_16_roses_l162_162179

-- Definitions for initial and final states
def initial_roses : ℕ := 37
def picked_roses : ℕ := 19
def final_roses : ℕ := 40

-- Defining the variable for number of roses sold
variable (x : ℕ)

-- The statement to prove
theorem florist_sold_16_roses
  (h : initial_roses - x + picked_roses = final_roses) : x = 16 := 
by
  -- Placeholder for proof
  sorry

end florist_sold_16_roses_l162_162179


namespace polynomial_sum_l162_162588

def p (x : ℝ) := -4 * x^2 + 2 * x - 5
def q (x : ℝ) := -6 * x^2 + 4 * x - 9
def r (x : ℝ) := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l162_162588


namespace b6_value_l162_162119

noncomputable def a_seq : ℕ → ℝ
| 0       := 1 + (1 / real.nthRoot 4 2)
| (n + 1) := (2 - a_seq n * b_seq n) / (1 - b_seq n)

noncomputable def b_seq : ℕ → ℝ
| n       := a_seq n / (a_seq n - 1)

def condition1 (i : ℕ) : Prop := a_seq i * b_seq i - a_seq i - b_seq i = 0

def condition2 (i : ℕ) : Prop := a_seq (i + 1) = (2 - a_seq i * b_seq i) / (1 - b_seq i)

def initial_condition : Prop := a_seq 0 = 1 + (1 / real.nthRoot 4 2)

theorem b6_value : b_seq 5 = 257 :=
by
  unfold a_seq b_seq
  -- to simplify the recurrence evaluation steps
  sorry

end b6_value_l162_162119


namespace initial_percentage_decrease_l162_162885

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₀ : P > 0)
  (initial_decrease : ∀ (x : ℝ), P * (1 - x / 100) * 1.3 = P * 1.04) :
  x = 20 :=
by 
  sorry

end initial_percentage_decrease_l162_162885


namespace real_part_one_div_one_sub_eq_half_l162_162592

noncomputable def realPart {z : ℂ} (hz_nonreal : z ≠ conj z) (hz_norm : ∥z∥ = 1) : ℝ :=
  re (1 / (1 - z))

theorem real_part_one_div_one_sub_eq_half
  (z : ℂ) (hz_nonreal : z ≠ conj z) (hz_norm : ∥z∥ = 1) :
  realPart hz_nonreal hz_norm = 1 / 2 :=
sorry

end real_part_one_div_one_sub_eq_half_l162_162592


namespace production_today_l162_162388

-- Definitions based on given conditions
def n := 9
def avg_past_days := 50
def avg_new_days := 55
def total_past_production := n * avg_past_days
def total_new_production := (n + 1) * avg_new_days

-- Theorem: Prove the number of units produced today
theorem production_today : total_new_production - total_past_production = 100 := by
  sorry

end production_today_l162_162388


namespace ballsInBoxes_theorem_l162_162560

noncomputable def ballsInBoxes : ℕ :=
  let distributions := [(5,0,0), (4,1,0), (3,2,0), (3,1,1), (2,2,1)]
  distributions.foldl (λ acc (x : ℕ × ℕ × ℕ) =>
    acc + match x with
      | (5,0,0) => 3
      | (4,1,0) => 6
      | (3,2,0) => 6
      | (3,1,1) => 6
      | (2,2,1) => 3
      | _ => 0
  ) 0

theorem ballsInBoxes_theorem : ballsInBoxes = 24 :=
by
  unfold ballsInBoxes
  rfl

end ballsInBoxes_theorem_l162_162560


namespace initial_guinea_fowls_l162_162920

theorem initial_guinea_fowls (initial_chickens initial_turkeys : ℕ) 
  (initial_guinea_fowls : ℕ) (lost_chickens lost_turkeys lost_guinea_fowls : ℕ) 
  (total_birds_end : ℕ) (days : ℕ)
  (hc : initial_chickens = 300) (ht : initial_turkeys = 200) 
  (lc : lost_chickens = 20) (lt : lost_turkeys = 8) (lg : lost_guinea_fowls = 5) 
  (d : days = 7) (tb : total_birds_end = 349) :
  initial_guinea_fowls = 80 := 
by 
  sorry

end initial_guinea_fowls_l162_162920


namespace f_max_a_zero_f_zero_range_l162_162078

-- Part 1: Proving the maximum value when a = 0
theorem f_max_a_zero : ∀ (f : ℝ → ℝ) (x : ℝ),
  (f x = (-1 / x) - Real.log x) ∧ x > 0 → 
  ∃ x_max : ℝ, f x_max = -1 ∧ 
  (∀ x > 0, f x ≤ -1) := 
sorry

-- Part 2: Proving the range of a for exactly one zero of f(x)
theorem f_zero_range (a : ℝ) : (0 < a) → 
  ∀ (f : ℝ → ℝ) (x : ℝ), 
  (f x = a * x - 1 / x - (a + 1) * Real.log x) ∧ x > 0 →
  ∃! (x_zero : ℝ), f x_zero = 0 :=
sorry

end f_max_a_zero_f_zero_range_l162_162078


namespace max_value_when_a_zero_range_of_a_for_one_zero_l162_162086

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * Real.log x

theorem max_value_when_a_zero :
  ∃ x : ℝ, x > 0 ∧ f 0 x = -1 :=
by sorry

theorem range_of_a_for_one_zero :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ f a x = 0) ↔ a > 0 :=
by sorry

end max_value_when_a_zero_range_of_a_for_one_zero_l162_162086


namespace parallel_implies_not_contained_l162_162441

variables {Line Plane : Type} (l : Line) (α : Plane)

-- Define the predicate for a line being parallel to a plane
def parallel (l : Line) (α : Plane) : Prop := sorry

-- Define the predicate for a line not being contained in a plane
def not_contained (l : Line) (α : Plane) : Prop := sorry

theorem parallel_implies_not_contained (l : Line) (α : Plane) (h : parallel l α) : not_contained l α :=
sorry

end parallel_implies_not_contained_l162_162441


namespace cube_root_eq_self_l162_162853

theorem cube_root_eq_self (a : ℝ) (h : a^(3:ℕ) = a) : a = 1 ∨ a = -1 ∨ a = 0 := 
sorry

end cube_root_eq_self_l162_162853


namespace ratio_men_to_women_l162_162501

theorem ratio_men_to_women
  (W M : ℕ)      -- W is the number of women, M is the number of men
  (avg_height_all : ℕ) (avg_height_female : ℕ) (avg_height_male : ℕ)
  (h1 : avg_height_all = 180)
  (h2 : avg_height_female = 170)
  (h3 : avg_height_male = 182) 
  (h_avg : (170 * W + 182 * M) / (W + M) = 180) :
  M = 5 * W :=
by
  sorry

end ratio_men_to_women_l162_162501


namespace part_I_part_II_l162_162554

open Real  -- Specify that we are working with real numbers

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

-- The first theorem: Prove the result for a = 1
theorem part_I (x : ℝ) : f x 1 + x > 0 ↔ (x > -3 ∧ x < 1 ∨ x > 3) :=
by
  sorry

-- The second theorem: Prove the range of a such that f(x) ≤ 3 for all x
theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 3) ↔ (-5 ≤ a ∧ a ≤ 1) :=
by
  sorry

end part_I_part_II_l162_162554


namespace probability_females_not_less_than_males_l162_162513

noncomputable def prob_female_not_less_than_male : ℚ :=
  let total_students := 5
  let females := 2
  let males := 3
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose females 2 + females * males
  favorable_combinations / total_combinations

theorem probability_females_not_less_than_males (total_students females males : ℕ) :
  total_students = 5 → females = 2 → males = 3 →
  prob_female_not_less_than_male = 7 / 10 :=
by intros; sorry

end probability_females_not_less_than_males_l162_162513


namespace intersection_unique_element_l162_162299

noncomputable def A := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def B (r : ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem intersection_unique_element (r : ℝ) (hr : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ A ∧ p ∈ B r) → (r = 3 ∨ r = 7) :=
sorry

end intersection_unique_element_l162_162299


namespace max_value_when_a_zero_range_of_a_for_one_zero_l162_162077

-- Definition of the function f(x)
def f (a x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * ln x

-- Proof Problem 1: maximum value of f(x) when a = 0
theorem max_value_when_a_zero :
  ∀ x > 0, f 0 x ≤ -1 ∧ f 0 1 = -1 := 
sorry

-- Proof Problem 2: range of values for a such that f(x) has exactly one zero
theorem range_of_a_for_one_zero :
  (∀ a : ℝ, (∃! x : ℝ, f a x = 0) ↔ (a > 0)) :=
sorry

end max_value_when_a_zero_range_of_a_for_one_zero_l162_162077


namespace least_lucky_multiple_of_six_not_lucky_l162_162912

def digit_sum (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % digit_sum n = 0

def is_multiple_of_six (n : ℕ) : Prop :=
  n % 6 = 0

theorem least_lucky_multiple_of_six_not_lucky : ∃ n, is_multiple_of_six n ∧ ¬ is_lucky_integer n ∧ (∀ m, is_multiple_of_six m ∧ ¬ is_lucky_integer m → n ≤ m) :=
by {
  use 12,
  split,
  { sorry },  -- Proof that 12 is a multiple of 6
  split,
  { sorry },  -- Proof that 12 is not a lucky integer
  { sorry },  -- Proof that there is no smaller multiple of 6 that is not a lucky integer
}

end least_lucky_multiple_of_six_not_lucky_l162_162912


namespace divisors_remainders_l162_162363

theorem divisors_remainders (n : ℕ) (h : ∀ k : ℕ, 1001 ≤ k ∧ k ≤ 2012 → ∃ d : ℕ, d ∣ n ∧ d % 2013 = k) :
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 2012 → ∃ d : ℕ, d ∣ n^2 ∧ d % 2013 = m :=
by sorry

end divisors_remainders_l162_162363


namespace diagonal_length_not_possible_l162_162555

-- Define the side lengths of the parallelogram
def sides_of_parallelogram : ℕ × ℕ := (6, 8)

-- Define the length of a diagonal that cannot exist
def invalid_diagonal_length : ℕ := 15

-- Statement: Prove that a diagonal of length 15 cannot exist for such a parallelogram.
theorem diagonal_length_not_possible (a b d : ℕ) 
  (h₁ : sides_of_parallelogram = (a, b)) 
  (h₂ : d = invalid_diagonal_length) 
  : d ≥ a + b := 
sorry

end diagonal_length_not_possible_l162_162555


namespace polynomial_abs_sum_roots_l162_162682

theorem polynomial_abs_sum_roots (p q r m : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2500) (h3 : p * q * r = -m) :
  |p| + |q| + |r| = 100 :=
sorry

end polynomial_abs_sum_roots_l162_162682


namespace steve_distance_l162_162141

theorem steve_distance (D : ℝ) (S : ℝ) 
  (h1 : 2 * S = 10)
  (h2 : (D / S) + (D / (2 * S)) = 6) :
  D = 20 :=
by
  sorry

end steve_distance_l162_162141


namespace find_initial_solution_liters_l162_162657

-- Define the conditions
def percentage_initial_solution_alcohol := 0.26
def added_water := 5
def percentage_new_mixture_alcohol := 0.195

-- Define the initial amount of the solution
def initial_solution_liters (x : ℝ) : Prop :=
  0.26 * x = 0.195 * (x + 5)

-- State the proof problem
theorem find_initial_solution_liters : initial_solution_liters 15 :=
by
  sorry

end find_initial_solution_liters_l162_162657


namespace fraction_multiplication_l162_162637

theorem fraction_multiplication : (1 / 3) * (1 / 4) * (1 / 5) * 60 = 1 := by
  sorry

end fraction_multiplication_l162_162637


namespace solve_for_x_l162_162101

def log_eq (x : ℝ) := log 3 ((x + 5)^2) + log (1/3) (x - 1) = 4

theorem solve_for_x (x : ℝ) (h : log_eq x) : x = (71 + Real.sqrt 4617) / 2 :=
sorry

end solve_for_x_l162_162101


namespace age_of_new_person_l162_162747

theorem age_of_new_person 
    (n : ℕ) 
    (T : ℕ := n * 14) 
    (n_eq : n = 9) 
    (new_average : (T + A) / (n + 1) = 16) 
    (A : ℕ) : A = 34 :=
by
  sorry

end age_of_new_person_l162_162747


namespace smallest_m_div_18_l162_162881

noncomputable def smallest_multiple_18 : ℕ :=
  900

theorem smallest_m_div_18 : (∃ m: ℕ, (m % 18 = 0) ∧ (∀ d ∈ m.digits 10, d = 9 ∨ d = 0) ∧ ∀ k: ℕ, k % 18 = 0 → (∀ d ∈ k.digits 10, d = 9 ∨ d = 0) → m ≤ k) → 900 / 18 = 50 :=
by
  intro h
  sorry

end smallest_m_div_18_l162_162881


namespace shaded_rectangle_ratio_l162_162921

variable (a : ℝ) (h : 0 < a)  -- side length of the square is 'a' and it is positive

theorem shaded_rectangle_ratio :
  (∃ l w : ℝ, (l = a / 2 ∧ w = a / 3 ∧ (l * w = a^2 / 6) ∧ (a^2 / 6 = a * a / 6))) → (l / w = 1.5) :=
by {
  -- Proof is to be provided
  sorry
}

end shaded_rectangle_ratio_l162_162921


namespace police_emergency_number_has_prime_divisor_gt_7_l162_162801

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := sorry

end police_emergency_number_has_prime_divisor_gt_7_l162_162801


namespace sqrt_double_sqrt_four_l162_162328

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_double_sqrt_four :
  sqrt (sqrt 4) = sqrt 2 ∨ sqrt (sqrt 4) = -sqrt 2 :=
by
  sorry

end sqrt_double_sqrt_four_l162_162328


namespace yogurt_price_l162_162126

theorem yogurt_price (x y : ℝ) (h1 : 4 * x + 4 * y = 14) (h2 : 2 * x + 8 * y = 13) : x = 2.5 :=
by
  sorry

end yogurt_price_l162_162126


namespace fourth_intersection_point_l162_162425

def intersect_curve_circle : Prop :=
  let curve_eq (x y : ℝ) : Prop := x * y = 1
  let circle_intersects_points (h k s : ℝ) : Prop :=
    ∃ (x1 y1 x2 y2 x3 y3 : ℝ), 
    (x1, y1) = (3, (1 : ℝ) / 3) ∧ 
    (x2, y2) = (-4, -(1 : ℝ) / 4) ∧ 
    (x3, y3) = ((1 : ℝ) / 6, 6) ∧ 
    (x1 - h)^2 + (y1 - k)^2 = s^2 ∧
    (x2 - h)^2 + (y2 - k)^2 = s^2 ∧
    (x3 - h)^2 + (y3 - k)^2 = s^2 
  let fourth_point_of_intersection (x y : ℝ) : Prop := 
    x = -(1 : ℝ) / 2 ∧ 
    y = -2
  curve_eq 3 ((1 : ℝ) / 3) ∧
  curve_eq (-4) (-(1 : ℝ) / 4) ∧
  curve_eq ((1 : ℝ) / 6) 6 ∧
  ∃ h k s, circle_intersects_points h k s →
  ∃ (x4 y4 : ℝ), curve_eq x4 y4 ∧
  fourth_point_of_intersection x4 y4

theorem fourth_intersection_point :
  intersect_curve_circle := by
  sorry

end fourth_intersection_point_l162_162425


namespace find_m_range_l162_162549

/--
Given:
1. Proposition \( p \) (p): The equation \(\frac{x^2}{2} + \frac{y^2}{m} = 1\) represents an ellipse with foci on the \( y \)-axis.
2. Proposition \( q \) (q): \( f(x) = \frac{4}{3}x^3 - 2mx^2 + (4m-3)x - m \) is monotonically increasing on \((-\infty, +\infty)\).

Prove:
If \( \neg p \land q \) is true, then the range of values for \( m \) is \( [1, 2] \).
-/

def p (m : ℝ) : Prop :=
  m > 2

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, (4 * x^2 - 4 * m * x + 4 * m - 3) >= 0

theorem find_m_range (m : ℝ) (hpq : ¬ p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end find_m_range_l162_162549


namespace smallest_number_of_weights_l162_162016

/-- The smallest number of weights in a set that can be divided into 4, 5, and 6 equal piles is 11. -/
theorem smallest_number_of_weights (n : ℕ) (M : ℕ) : (∀ k : ℕ, (k = 4 ∨ k = 5 ∨ k = 6) → M % k = 0) → n = 11 :=
sorry

end smallest_number_of_weights_l162_162016


namespace calculate_expression_l162_162201

def thirteen_power_thirteen_div_thirteen_power_twelve := 13 ^ 13 / 13 ^ 12
def expression := (thirteen_power_thirteen_div_thirteen_power_twelve ^ 3) * (3 ^ 3)
/- We define the main statement to be proven -/
theorem calculate_expression : (expression / 2 ^ 6) = 926 := sorry

end calculate_expression_l162_162201


namespace score_ordering_l162_162134

-- Definition of the problem conditions in Lean 4:
def condition1 (Q K : ℝ) : Prop := Q ≠ K
def condition2 (M Q S K : ℝ) : Prop := M < Q ∧ M < S ∧ M < K
def condition3 (S Q M K : ℝ) : Prop := S > Q ∧ S > M ∧ S > K

-- Theorem statement in Lean 4:
theorem score_ordering (M Q S K : ℝ) (h1 : condition1 Q K) (h2 : condition2 M Q S K) (h3 : condition3 S Q M K) : 
  M < Q ∧ Q < S :=
by
  sorry

end score_ordering_l162_162134


namespace trapezoid_perimeter_l162_162427

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h1 : AB = 5)
  (h2 : CD = 5)
  (h3 : AD = 16)
  (h4 : BC = 8) :
  AB + BC + CD + AD = 34 :=
by
  sorry

end trapezoid_perimeter_l162_162427


namespace cos_pi_over_3_plus_2alpha_correct_l162_162391

noncomputable def cos_pi_over_3_plus_2alpha (α : Real) (h : Real.sin (Real.pi / 3 - α) = 1 / 4) : Real :=
  Real.cos (Real.pi / 3 + 2 * α)

theorem cos_pi_over_3_plus_2alpha_correct (α : Real) (h : Real.sin (Real.pi / 3 - α) = 1 / 4) :
  cos_pi_over_3_plus_2alpha α h = -7 / 8 :=
by
  sorry

end cos_pi_over_3_plus_2alpha_correct_l162_162391


namespace reginald_apples_sold_l162_162740

theorem reginald_apples_sold 
  (apple_price : ℝ) 
  (bike_cost : ℝ)
  (repair_percentage : ℝ)
  (remaining_fraction : ℝ)
  (discount_apples : ℕ)
  (free_apples : ℕ)
  (total_apples_sold : ℕ) : 
  apple_price = 1.25 → 
  bike_cost = 80 → 
  repair_percentage = 0.25 → 
  remaining_fraction = 0.2 → 
  discount_apples = 5 → 
  free_apples = 1 → 
  (∃ (E : ℝ), (125 = E ∧ total_apples_sold = 120)) → 
  total_apples_sold = 120 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end reginald_apples_sold_l162_162740


namespace sin_2012_eq_neg_sin_32_l162_162903

theorem sin_2012_eq_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = - Real.sin (32 * Real.pi / 180) :=
by
  sorry

end sin_2012_eq_neg_sin_32_l162_162903


namespace girl_walking_speed_l162_162662

-- Definitions of the conditions
def distance := 30 -- in kilometers
def time := 6 -- in hours

-- Definition of the walking speed function
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- The theorem we want to prove
theorem girl_walking_speed : speed distance time = 5 := by
  sorry

end girl_walking_speed_l162_162662


namespace base8_to_base10_l162_162211

theorem base8_to_base10 :
  ∀ (n : ℕ), (n = 2 * 8^2 + 4 * 8^1 + 3 * 8^0) → (n = 163) :=
by
  intros n hn
  sorry

end base8_to_base10_l162_162211


namespace profit_percentage_l162_162749

theorem profit_percentage (C S : ℝ) (h : 30 * C = 24 * S) :
  (S - C) / C * 100 = 25 :=
by sorry

end profit_percentage_l162_162749


namespace parallelepiped_edges_parallel_to_axes_l162_162720

theorem parallelepiped_edges_parallel_to_axes 
  (V : ℝ) (a b c : ℝ) 
  (integer_coords : ∀ (x y z : ℝ), x = a ∨ x = 0 ∧ y = b ∨ y = 0 ∧ z = c ∨ z = 0) 
  (volume_cond : V = 2011) 
  (volume_def : V = a * b * c) 
  (a_int : a ∈ ℤ) 
  (b_int : b ∈ ℤ) 
  (c_int : c ∈ ℤ) : 
  a = 1 ∧ b = 1 ∧ c = 2011 ∨ 
  a = 1 ∧ b = 2011 ∧ c = 1 ∨ 
  a = 2011 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end parallelepiped_edges_parallel_to_axes_l162_162720


namespace part_1_part_2_part_3_l162_162960

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

theorem part_1 (h : f 1 m = 5) : m = 4 :=
sorry

theorem part_2 (m : ℝ) (h : m = 4) : ∀ x : ℝ, f (-x) m = -f x m :=
sorry

theorem part_3 (m : ℝ) (h : m = 4) : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 m < f x2 m :=
sorry

end part_1_part_2_part_3_l162_162960


namespace find_m_from_power_function_l162_162251

theorem find_m_from_power_function :
  (∃ a : ℝ, (2 : ℝ) ^ a = (Real.sqrt 2) / 2) →
  (∃ m : ℝ, (m : ℝ) ^ (-1 / 2 : ℝ) = 2) →
  ∃ m : ℝ, m = 1 / 4 :=
by
  intro h1 h2
  sorry

end find_m_from_power_function_l162_162251


namespace total_oranges_l162_162597

def monday_oranges : ℕ := 100
def tuesday_oranges : ℕ := 3 * monday_oranges
def wednesday_oranges : ℕ := 70

theorem total_oranges : monday_oranges + tuesday_oranges + wednesday_oranges = 470 := by
  sorry

end total_oranges_l162_162597


namespace sequence_property_exists_l162_162526

theorem sequence_property_exists :
  ∃ a₁ a₂ a₃ a₄ : ℝ, 
  a₂ - a₁ = a₃ - a₂ ∧ a₃ - a₂ = a₄ - a₃ ∧
  (a₃ / a₁ = a₄ / a₃) ∧ ∃ r : ℝ, r ≠ 0 ∧ a₁ = -4 * r ∧ a₂ = -3 * r ∧ a₃ = -2 * r ∧ a₄ = -r :=
by
  sorry

end sequence_property_exists_l162_162526


namespace floor_difference_l162_162944

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l162_162944


namespace min_value_expr_l162_162675

-- Define the given expression
def given_expr (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x) + 200

-- Define the minimum value we need to prove
def min_value : ℝ :=
  -6290.25

-- The statement of the theorem
theorem min_value_expr :
  ∃ x : ℝ, ∀ y : ℝ, given_expr y ≥ min_value := by
  sorry

end min_value_expr_l162_162675


namespace police_emergency_number_prime_divisor_l162_162807

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 1000 * k + 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_prime_divisor_l162_162807


namespace unique_solution_zmod_11_l162_162934

theorem unique_solution_zmod_11 : 
  ∀ (n : ℕ), 
  (2 ≤ n → 
  (∀ x : ZMod n, (x^2 - 3 * x + 5 = 0) → (∃! x : ZMod n, x^2 - (3 : ZMod n) * x + (5 : ZMod n) = 0)) → 
  n = 11) := 
by
  sorry

end unique_solution_zmod_11_l162_162934


namespace tree_growth_l162_162474

theorem tree_growth (x : ℝ) : 4*x + 4*2*x + 4*2 + 4*3 = 32 → x = 1 :=
by
  intro h
  sorry

end tree_growth_l162_162474


namespace domain_of_function_l162_162054

theorem domain_of_function :
  {x : ℝ | (x^2 - 9*x + 20 ≥ 0) ∧ (|x - 5| + |x + 2| ≠ 0)} = {x : ℝ | x ≤ 4 ∨ x ≥ 5} :=
by
  sorry

end domain_of_function_l162_162054


namespace problem_l162_162467

open Real

theorem problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
sorry

end problem_l162_162467


namespace sin_double_angle_l162_162564

-- Define the conditions and the goal
theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
sorry

end sin_double_angle_l162_162564


namespace power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l162_162495

theorem power_of_3_mod_5 (n : ℕ) : (3^n % 5) = 
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 4
  else if n % 4 = 3 then 2
  else 1 := by
  sorry

theorem remainder_of_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  have h : 2023 % 4 = 3 := by norm_num
  rw power_of_3_mod_5 2023
  simp [h]
  sorry

end power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l162_162495


namespace part1_max_value_a_0_part2_unique_zero_l162_162089

noncomputable def f (a x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * Real.log x

theorem part1_max_value_a_0 : (∀ x > 0, f 0 x ≤ -1) ∧ (∃ x > 0, f 0 x = -1) :=
sorry

theorem part2_unique_zero : ∀ a > 0, ∃! x > 0, f a x = 0 :=
sorry

end part1_max_value_a_0_part2_unique_zero_l162_162089


namespace find_matrix_M_find_line_l_l162_162700

open Matrix

-- Given conditions
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ := !![6, 2; 4, 4]
def eigenvector : Fin 2 → ℝ := λ i, if i = 0 then 1 else 1
def point_A : Fin 2 → ℝ := λ i, if i = 0 then -1 else 2
def point_A_prime : Fin 2 → ℝ := λ i, if i = 0 then -2 else 4
def line_m (x y : ℝ) : Prop := x - y = 6

-- Matrix satisfies eigenvalue-eigenvector equation
def matrix_eigenvalue : Prop := 
  ∃ λ = 8, (matrix_M.mulVec eigenvector = λ • eigenvector)

-- Matrix transforms point A to A'
def transformation_A_to_A_prime : Prop := 
  matrix_M.mulVec point_A = point_A_prime

-- Find matrix M and verify its characteristics
theorem find_matrix_M : matrix_M = !![6, 2; 4, 4]
∧ matrix_eigenvalue
∧ transformation_A_to_A_prime := 
begin
  sorry -- Proof is not required
end

-- Find the original line l given M^-1 transforms l to m
def line_l (x y : ℝ) : Prop := x - y = 12

-- Statement considering the inverse transformation
theorem find_line_l (x y : ℝ) : Prop :=
  (M⁻¹).mulVec !![x, y] = !![x_m', y_m'] →
  (line_m x_m' y_m') → (line_l x y) := 
begin
  sorry -- Proof is not required
end

end find_matrix_M_find_line_l_l162_162700


namespace sum_of_distinct_prime_factors_of_462_l162_162165

theorem sum_of_distinct_prime_factors_of_462 :
  let factors := [2, 3, 7, 11] in -- The list of distinct prime factors of 462.
  factors.sum = 23 :=             -- We want to prove that their sum is 23.
by
  sorry

end sum_of_distinct_prime_factors_of_462_l162_162165


namespace line_equation_l162_162842

open Real

-- Define the points A, B, and C
def A : ℝ × ℝ := ⟨1, 4⟩
def B : ℝ × ℝ := ⟨3, 2⟩
def C : ℝ × ℝ := ⟨2, -1⟩

-- Definition for a line passing through point C
-- and having equal distance to points A and B
def is_line_equation (l : ℝ → ℝ → Prop) :=
  ∀ x y, (l x y ↔ (x + y - 1 = 0 ∨ x - 2 = 0))

-- Our main statement
theorem line_equation :
  ∃ l : ℝ → ℝ → Prop, is_line_equation l ∧ (l 2 (-1)) :=
by
  sorry  -- Proof goes here.

end line_equation_l162_162842


namespace zoo_charge_for_child_l162_162311

theorem zoo_charge_for_child (charge_adult : ℕ) (total_people total_bill children : ℕ) (charge_child : ℕ) : 
  charge_adult = 8 → total_people = 201 → total_bill = 964 → children = 161 → 
  total_bill - (total_people - children) * charge_adult = children * charge_child → 
  charge_child = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end zoo_charge_for_child_l162_162311


namespace least_sales_needed_not_lose_money_l162_162902

noncomputable def old_salary : ℝ := 75000
noncomputable def new_salary_base : ℝ := 45000
noncomputable def commission_rate : ℝ := 0.15
noncomputable def sale_amount : ℝ := 750

theorem least_sales_needed_not_lose_money : 
  ∃ (n : ℕ), n * (commission_rate * sale_amount) ≥ (old_salary - new_salary_base) ∧ n = 267 := 
by
  -- The proof will show that n = 267 is the least number of sales needed to not lose money.
  existsi 267
  sorry

end least_sales_needed_not_lose_money_l162_162902


namespace transportation_cost_l162_162459

-- Definitions for the conditions
def number_of_original_bags : ℕ := 80
def weight_of_original_bag : ℕ := 50
def total_cost_original : ℕ := 6000

def scale_factor_bags : ℕ := 3
def scale_factor_weight : ℚ := 3 / 5

-- Derived quantities
def number_of_new_bags : ℕ := scale_factor_bags * number_of_original_bags
def weight_of_new_bag : ℚ := scale_factor_weight * weight_of_original_bag
def cost_per_original_bag : ℚ := total_cost_original / number_of_original_bags
def cost_per_new_bag : ℚ := cost_per_original_bag * (weight_of_new_bag / weight_of_original_bag)

-- Final cost calculation
def total_cost_new : ℚ := number_of_new_bags * cost_per_new_bag

-- The statement that needs to be proved
theorem transportation_cost : total_cost_new = 10800 := sorry

end transportation_cost_l162_162459


namespace sum_distinct_prime_factors_462_l162_162160

theorem sum_distinct_prime_factors_462 : 
  ∏ x in {2, 3, 7, 11}, x = 462 → (∑ x in {2, 3, 7, 11}, x) = 23 :=
by
  intro h
  -- Proof goes here
  sorry

end sum_distinct_prime_factors_462_l162_162160


namespace calc_neg_half_times_neg_two_pow_l162_162369

theorem calc_neg_half_times_neg_two_pow :
  - (0.5 ^ 20) * ((-2) ^ 26) = -64 := by
  sorry

end calc_neg_half_times_neg_two_pow_l162_162369


namespace number_of_outfits_l162_162445

-- Definitions based on conditions
def trousers : ℕ := 4
def shirts : ℕ := 8
def jackets : ℕ := 3
def belts : ℕ := 2

-- The statement to prove
theorem number_of_outfits : trousers * shirts * jackets * belts = 192 := by
  sorry

end number_of_outfits_l162_162445


namespace volume_of_region_l162_162236

theorem volume_of_region :
  ∃ (V : ℝ), V = 9 ∧
  ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 6 :=
sorry

end volume_of_region_l162_162236


namespace initial_peanuts_l162_162332

theorem initial_peanuts (x : ℕ) (h : x + 4 = 8) : x = 4 :=
sorry

end initial_peanuts_l162_162332


namespace tetrahedral_dice_sum_6_probability_l162_162892

theorem tetrahedral_dice_sum_6_probability :
  let outcomes : Finset (ℕ × ℕ) := { (a, b) | a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4} }
  let favorable : Finset (ℕ × ℕ) := { (a, b) ∈ outcomes | a + b = 6 }
  let total_combinations : ℕ := Finset.card outcomes
  let favorable_combinations : ℕ := Finset.card favorable
  (favorable_combinations : ℚ) / total_combinations = 3 / 16 :=
by
  sorry

end tetrahedral_dice_sum_6_probability_l162_162892


namespace quadratic_real_roots_range_k_l162_162699

-- Define the quadratic function
def quadratic_eq (k x : ℝ) : ℝ := k * x^2 - 6 * x + 9

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the conditions for the quadratic equation to have distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ discriminant a b c > 0

theorem quadratic_real_roots_range_k (k : ℝ) :
  has_two_distinct_real_roots k (-6) 9 ↔ k < 1 ∧ k ≠ 0 := 
by
  sorry

end quadratic_real_roots_range_k_l162_162699


namespace ab_non_positive_l162_162962

-- Define the conditions as a structure if necessary.
variables {a b : ℝ}

-- State the theorem.
theorem ab_non_positive (h : 3 * a + 8 * b = 0) : a * b ≤ 0 :=
sorry

end ab_non_positive_l162_162962


namespace polynomial_expansion_sum_l162_162099

theorem polynomial_expansion_sum :
  let A := 4
  let B := 10
  let C := 1
  let D := 21
  (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 36 := 
by
  -- Proof needs to be filled
  sorry

end polynomial_expansion_sum_l162_162099


namespace power_mod_l162_162486

theorem power_mod : (3^2023) % 5 = 2 := by
  have h1 : 3^1 % 5 = 3 := by sorry
  have h2 : 3^2 % 5 = 4 := by sorry
  have h3 : 3^3 % 5 = 2 := by sorry
  have h4 : 3^4 % 5 = 1 := by sorry
  have periodicity : ∀ k: ℕ, (3^(4*k)) % 5 = 1 := by sorry
  have remainder : 2023 % 4 = 3 := by sorry
  show (3^2023) % 5 = 2 from by
    rw [Nat.mod_eq_of_lt 5 4] at remainder -- remainder shows 2023 mod 4 = 3
    exact h3

end power_mod_l162_162486


namespace remainder_of_3_pow_2023_mod_5_l162_162483

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l162_162483


namespace sequence_equality_l162_162636

noncomputable def a (x : ℝ) (n : ℕ) : ℝ := 1 + x^(n+1) + x^(n+2)

theorem sequence_equality (x : ℝ) (hx : x = 0 ∨ x = 1 ∨ x = -1) (n : ℕ) (hn : n ≥ 3) :
  (a x n)^2 = (a x (n-1)) * (a x (n+1)) :=
by sorry

end sequence_equality_l162_162636


namespace population_in_2060_l162_162532

noncomputable def population (year : ℕ) : ℕ :=
  if h : (year - 2000) % 20 = 0 then
    250 * 2 ^ ((year - 2000) / 20)
  else
    0 -- This handles non-multiples of 20 cases, which are irrelevant here

theorem population_in_2060 : population 2060 = 2000 := by
  sorry

end population_in_2060_l162_162532


namespace points_lie_on_line_l162_162240

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  x + y = 4 :=
by
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  sorry

end points_lie_on_line_l162_162240


namespace common_points_circle_ellipse_l162_162823

theorem common_points_circle_ellipse :
    (∃ (p1 p2: ℝ × ℝ),
        p1 ≠ p2 ∧
        (p1, p2).fst.1 ^ 2 + (p1, p2).fst.2 ^ 2 = 4 ∧
        9 * (p1, p2).fst.1 ^ 2 + 4 * (p1, p2).fst.2 ^ 2 = 36 ∧
        (p1, p2).snd.1 ^ 2 + (p1, p2).snd.2 ^ 2 = 4 ∧
        9 * (p1, p2).snd.1 ^ 2 + 4 * (p1, p2).snd.2 ^ 2 = 36) :=
sorry

end common_points_circle_ellipse_l162_162823


namespace compare_y1_y2_l162_162302

def parabola (x : ℝ) (c : ℝ) : ℝ := -x^2 + 4 * x + c

theorem compare_y1_y2 (c y1 y2 : ℝ) :
  parabola (-1) c = y1 →
  parabola 1 c = y2 →
  y1 < y2 :=
by
  intro h1 h2
  sorry

end compare_y1_y2_l162_162302


namespace three_students_with_B_l162_162382

-- Define the students and their statements as propositions
variables (Eva B_Frank B_Gina B_Harry : Prop)

-- Condition 1: Eva said, "If I get a B, then Frank will get a B."
axiom Eva_statement : Eva → B_Frank

-- Condition 2: Frank said, "If I get a B, then Gina will get a B."
axiom Frank_statement : B_Frank → B_Gina

-- Condition 3: Gina said, "If I get a B, then Harry will get a B."
axiom Gina_statement : B_Gina → B_Harry

-- Condition 4: Only three students received a B.
axiom only_three_Bs : (Eva ∧ B_Frank ∧ B_Gina ∧ B_Harry) → False

-- The theorem we need to prove: The three students who received B's are Frank, Gina, and Harry.
theorem three_students_with_B (h_B_Frank : B_Frank) (h_B_Gina : B_Gina) (h_B_Harry : B_Harry) : ¬Eva :=
by
  sorry

end three_students_with_B_l162_162382


namespace candies_indeterminable_l162_162170

theorem candies_indeterminable
  (num_bags : ℕ) (cookies_per_bag : ℕ) (total_cookies : ℕ) (known_candies : ℕ) :
  num_bags = 26 →
  cookies_per_bag = 2 →
  total_cookies = 52 →
  num_bags * cookies_per_bag = total_cookies →
  ∀ (candies : ℕ), candies = known_candies → false :=
by
  intros
  sorry

end candies_indeterminable_l162_162170


namespace six_digit_number_divisible_9_22_l162_162353

theorem six_digit_number_divisible_9_22 (d : ℕ) (h0 : 0 ≤ d) (h1 : d ≤ 9)
  (h2 : 9 ∣ (220140 + d)) (h3 : 22 ∣ (220140 + d)) : 220140 + d = 520146 :=
sorry

end six_digit_number_divisible_9_22_l162_162353


namespace min_expression_value_l162_162863

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  a^2 + 4 * a * b + 8 * b^2 + 10 * b * c + 3 * c^2

theorem min_expression_value (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 3) :
  minimum_value a b c ≥ 27 :=
sorry

end min_expression_value_l162_162863


namespace two_disjoint_triangles_l162_162116

/-- Jody has 6 distinguishable balls and 6 distinguishable sticks,
    all of the same length. Prove that the number of ways to use
    the sticks to connect the balls so that two disjoint non-interlocking 
    triangles are formed, considering rotations and reflections of the 
    same arrangement to be indistinguishable, is 200. -/
theorem two_disjoint_triangles (balls : Finset ℕ) (sticks : Finset ℕ)
  (h_balls : balls.card = 6) (h_sticks : sticks.card = 6) :
  ∃ (n : ℕ), n = 200 :=
sorry

end two_disjoint_triangles_l162_162116


namespace largest_x_l162_162057

-- Definitions from the conditions
def eleven_times_less_than_150 (x : ℕ) : Prop := 11 * x < 150

-- Statement of the proof problem
theorem largest_x : ∃ x : ℕ, eleven_times_less_than_150 x ∧ ∀ y : ℕ, eleven_times_less_than_150 y → y ≤ x := 
sorry

end largest_x_l162_162057


namespace square_area_rational_l162_162274

-- Define the condition: the side length of the square is a rational number.
def is_rational (x : ℚ) : Prop := true

-- Define the theorem to be proved: If the side length of a square is rational, then its area is rational.
theorem square_area_rational (s : ℚ) (h : is_rational s) : is_rational (s * s) := 
sorry

end square_area_rational_l162_162274


namespace Charles_has_13_whistles_l162_162607

-- Conditions
def Sean_whistles : ℕ := 45
def more_whistles_than_Charles : ℕ := 32

-- Let C be the number of whistles Charles has
def C : ℕ := Sean_whistles - more_whistles_than_Charles

-- Theorem to be proven
theorem Charles_has_13_whistles : C = 13 := by
  -- skipping proof
  sorry

end Charles_has_13_whistles_l162_162607


namespace number_of_wins_and_losses_l162_162890

theorem number_of_wins_and_losses (x y : ℕ) (h1 : x + y = 15) (h2 : 3 * x + y = 41) :
  x = 13 ∧ y = 2 :=
sorry

end number_of_wins_and_losses_l162_162890


namespace sample_size_l162_162661

theorem sample_size (k n : ℕ) (h_ratio : 3 * n / (3 + 4 + 7) = 9) : n = 42 :=
by
  sorry

end sample_size_l162_162661


namespace iron_wire_square_rectangle_l162_162918

theorem iron_wire_square_rectangle 
  (total_length : ℕ) 
  (rect_length : ℕ) 
  (h1 : total_length = 28) 
  (h2 : rect_length = 12) :
  (total_length / 4 = 7) ∧
  ((total_length / 2) - rect_length = 2) :=
by 
  sorry

end iron_wire_square_rectangle_l162_162918


namespace total_pencils_correct_l162_162632

def pencils_in_drawer : ℕ := 43
def pencils_on_desk_originally : ℕ := 19
def pencils_added_by_dan : ℕ := 16
def total_pencils : ℕ := pencils_in_drawer + pencils_on_desk_originally + pencils_added_by_dan

theorem total_pencils_correct : total_pencils = 78 := by
  sorry

end total_pencils_correct_l162_162632


namespace value_of_a_for_positive_root_l162_162060

theorem value_of_a_for_positive_root :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ (3*x - 1)/(x - 3) = a/(3 - x) - 1) → a = -8 :=
by
  sorry

end value_of_a_for_positive_root_l162_162060


namespace part1_part2_l162_162406

open Real

theorem part1 (m : ℝ) (h : ∀ x : ℝ, abs (x - 2) + abs (x - 3) ≥ m) : m ≤ 1 := 
sorry

theorem part2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 = 1 / a + 1 / (2 * b) + 1 / (3 * c)) : a + 2 * b + 3 * c ≥ 9 := 
sorry

end part1_part2_l162_162406


namespace score_analysis_l162_162660

open Real

noncomputable def deviations : List ℝ := [8, -3, 12, -7, -10, -4, -8, 1, 0, 10]
def benchmark : ℝ := 85

theorem score_analysis :
  let highest_score := benchmark + List.maximum deviations
  let lowest_score := benchmark + List.minimum deviations
  let sum_deviations := List.sum deviations
  let average_deviation := sum_deviations / List.length deviations
  let average_score := benchmark + average_deviation
  highest_score = 97 ∧ lowest_score = 75 ∧ average_score = 84.9 :=
by
  sorry -- This is the placeholder for the proof

end score_analysis_l162_162660


namespace travel_time_l162_162746

def speed : ℝ := 60  -- Speed of the car in miles per hour
def distance : ℝ := 300  -- Distance to the campground in miles

theorem travel_time : distance / speed = 5 := by
  sorry

end travel_time_l162_162746


namespace reflection_point_sum_l162_162880

theorem reflection_point_sum (m b : ℝ) (H : ∀ x y : ℝ, (1, 2) = (x, y) ∨ (7, 6) = (x, y) → 
    y = m * x + b) : m + b = 8.5 := by
  sorry

end reflection_point_sum_l162_162880


namespace max_value_Sn_l162_162698

theorem max_value_Sn (a₁ : ℚ) (r : ℚ) (S : ℕ → ℚ)
  (h₀ : a₁ = 3 / 2)
  (h₁ : r = -1 / 2)
  (h₂ : ∀ n, S n = a₁ * (1 - r ^ n) / (1 - r))
  : ∀ n, S n ≤ 3 / 2 ∧ (∃ m, S m = 3 / 2) :=
by sorry

end max_value_Sn_l162_162698


namespace transformed_solution_equiv_l162_162402

noncomputable def quadratic_solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f x > 0}

noncomputable def transformed_solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (10^x) > 0}

theorem transformed_solution_equiv (f : ℝ → ℝ) :
  quadratic_solution_set f = {x | x < -1 ∨ x > 1 / 2} →
  transformed_solution_set f = {x | x > -Real.log 2} :=
by sorry

end transformed_solution_equiv_l162_162402


namespace triangle_angle_sum_l162_162327

theorem triangle_angle_sum (A : ℕ) (h1 : A = 55) (h2 : ∀ (B : ℕ), B = 2 * A) : (A + 2 * A = 165) :=
by
  sorry

end triangle_angle_sum_l162_162327


namespace trigonometric_identity_l162_162839

theorem trigonometric_identity
  (α : ℝ) 
  (h : Real.tan α = -1 / 2) :
  (Real.cos α - Real.sin α)^2 / Real.cos (2 * α) = 3 := 
by 
  sorry

end trigonometric_identity_l162_162839


namespace math_problem_l162_162047

theorem math_problem :
  101 * 102^2 - 101 * 98^2 = 80800 :=
by
  sorry

end math_problem_l162_162047


namespace systematic_sampling_twentieth_group_number_l162_162894

theorem systematic_sampling_twentieth_group_number 
  (total_students : ℕ) 
  (total_groups : ℕ) 
  (first_group_number : ℕ) 
  (interval : ℕ) 
  (n : ℕ) 
  (drawn_number : ℕ) :
  total_students = 400 →
  total_groups = 20 →
  first_group_number = 11 →
  interval = 20 →
  n = 20 →
  drawn_number = 11 + 20 * (n - 1) →
  drawn_number = 391 :=
by
  sorry

end systematic_sampling_twentieth_group_number_l162_162894


namespace quadractic_inequality_solution_l162_162854

theorem quadractic_inequality_solution (a b : ℝ) (h₁ : ∀ x : ℝ, -4 ≤ x ∧ x ≤ 3 → x^2 - (a+1) * x + b ≤ 0) : a + b = -14 :=
by 
  -- Proof construction is omitted
  sorry

end quadractic_inequality_solution_l162_162854


namespace fill_grid_power_of_two_l162_162373

theorem fill_grid_power_of_two (n : ℕ) (h : ∃ m : ℕ, n = 2^m) :
  ∃ f : ℕ → ℕ → ℕ, 
    (∀ i j : ℕ, i < n → j < n → 1 ≤ f i j ∧ f i j ≤ 2 * n - 1) ∧
    (∀ k, 1 ≤ k ∧ k ≤ n → (∀ i, i < n → ∀ j, j < n → i ≠ j → f i k ≠ f j k))
:= by
  sorry

end fill_grid_power_of_two_l162_162373


namespace part1_max_value_a_0_part2_unique_zero_l162_162088

noncomputable def f (a x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * Real.log x

theorem part1_max_value_a_0 : (∀ x > 0, f 0 x ≤ -1) ∧ (∃ x > 0, f 0 x = -1) :=
sorry

theorem part2_unique_zero : ∀ a > 0, ∃! x > 0, f a x = 0 :=
sorry

end part1_max_value_a_0_part2_unique_zero_l162_162088


namespace flour_needed_correct_l162_162436

-- Define the total flour required and the flour already added
def total_flour : ℕ := 8
def flour_already_added : ℕ := 2

-- Define the equation to determine the remaining flour needed
def flour_needed : ℕ := total_flour - flour_already_added

-- Prove that the flour needed to be added is 6 cups
theorem flour_needed_correct : flour_needed = 6 := by
  sorry

end flour_needed_correct_l162_162436


namespace polynomial_expansion_sum_l162_162100

theorem polynomial_expansion_sum :
  let A := 4
  let B := 10
  let C := 1
  let D := 21
  (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 36 := 
by
  -- Proof needs to be filled
  sorry

end polynomial_expansion_sum_l162_162100


namespace crackers_per_box_l162_162052

-- Given conditions
variables (x : ℕ)
variable (darren_boxes : ℕ := 4)
variable (calvin_boxes : ℕ := 2 * darren_boxes - 1)
variable (total_crackers : ℕ := 264)

-- Using the given conditions, create the proof statement to show x = 24
theorem crackers_per_box:
  11 * x = total_crackers → x = 24 :=
by
  sorry

end crackers_per_box_l162_162052


namespace millennium_run_time_l162_162887

theorem millennium_run_time (M A B : ℕ) (h1 : B = 100) (h2 : B = A + 10) (h3 : A = M - 30) : M = 120 := by
  sorry

end millennium_run_time_l162_162887


namespace find_ABC_l162_162537

theorem find_ABC :
    ∃ (A B C : ℚ), 
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5 → 
        (x^2 - 9) / ((x - 2) * (x - 4) * (x - 5)) = A / (x - 2) + B / (x - 4) + C / (x - 5)) 
    ∧ A = 5 / 3 ∧ B = -7 / 2 ∧ C = 8 / 3 := 
sorry

end find_ABC_l162_162537


namespace second_quadrant_inequality_l162_162858

theorem second_quadrant_inequality (a b : ℝ) (h₁ : a < 0) (h₂ : b > 0) : a / b < 0 :=
by sorry

end second_quadrant_inequality_l162_162858


namespace probability_all_same_color_l162_162789

theorem probability_all_same_color :
  let total_marbles := 5 + 7 + 4
  let total_draws := nat.choose total_marbles 3
  let prob_all_red := (5 / total_marbles) * ((5 - 1) / (total_marbles - 1)) * ((5 - 2) / (total_marbles - 2))
  let prob_all_white := (7 / total_marbles) * ((7 - 1) / (total_marbles - 1)) * ((7 - 2) / (total_marbles - 2))
  let prob_all_green := (4 / total_marbles) * ((4 - 1) / (total_marbles - 1)) * ((4 - 2) / (total_marbles - 2))
  prob_all_red + prob_all_white + prob_all_green = 43 / 280 := sorry

end probability_all_same_color_l162_162789


namespace base_of_isosceles_triangle_l162_162755

theorem base_of_isosceles_triangle (a b side equil_perim iso_perim : ℕ) 
  (h1 : equil_perim = 60)
  (h2 : 3 * side = equil_perim)
  (h3 : iso_perim = 50)
  (h4 : 2 * side + b = iso_perim)
  : b = 10 :=
by
  sorry

end base_of_isosceles_triangle_l162_162755


namespace circle_condition_l162_162312

def represents_circle (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x + 1/2)^2 + (y + m)^2 = 5/4 - m

theorem circle_condition (m : ℝ) : represents_circle m ↔ m < 5/4 :=
by sorry

end circle_condition_l162_162312


namespace triangles_point_distance_inequality_l162_162171

open Set

variables {A₁ A₂ A₃ B₁ B₂ B₃ X Y : ℝ × ℝ}

-- Assume A₁, A₂, A₃ are vertices of the first triangle and B₁, B₂, B₃ are vertices of the second triangle
-- Assume X is a point inside the triangle A₁A₂A₃
-- Assume Y is a point inside the triangle B₁B₂B₃

noncomputable def point_in_triangle (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  ∃ μ ν : ℝ, 0 ≤ μ ∧ 0 ≤ ν ∧ μ + ν ≤ 1 ∧ P = (1 - μ - ν) • A + μ • B + ν • C

theorem triangles_point_distance_inequality 
  (hX : point_in_triangle X A₁ A₂ A₃)
  (hY : point_in_triangle Y B₁ B₂ B₃) :
  ∃ (i j : Fin 3), dist X Y < dist (list.nth_le [A₁, A₂, A₃] i sorry) (list.nth_le [B₁, B₂, B₃] j sorry) :=
sorry

end triangles_point_distance_inequality_l162_162171


namespace rotten_oranges_found_l162_162734

def initial_oranges : ℕ := 7 * 12
def reserved_oranges : ℕ := initial_oranges / 4
def remaining_oranges_after_reserving : ℕ := initial_oranges - reserved_oranges
def sold_yesterday : ℕ := (3 * remaining_oranges_after_reserving) / 7
def remaining_oranges_after_selling : ℕ := remaining_oranges_after_reserving - sold_yesterday
def oranges_left_today : ℕ := 32

theorem rotten_oranges_found :
  (remaining_oranges_after_selling - oranges_left_today) = 4 :=
by
  sorry

end rotten_oranges_found_l162_162734


namespace second_number_in_set_l162_162455

theorem second_number_in_set (avg1 avg2 n1 n2 n3 : ℕ) (h1 : avg1 = (10 + 70 + 19) / 3) (h2 : avg2 = avg1 + 7) (h3 : n1 = 20) (h4 : n3 = 60) :
  n2 = n3 := 
  sorry

end second_number_in_set_l162_162455


namespace sale_price_after_discounts_l162_162924

/-- The sale price of the television as a percentage of its original price after successive discounts of 25% followed by 10%. -/
theorem sale_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 350 → discount1 = 0.25 → discount2 = 0.10 →
  (original_price * (1 - discount1) * (1 - discount2) / original_price) * 100 = 67.5 :=
by
  intro h_price h_discount1 h_discount2
  sorry

end sale_price_after_discounts_l162_162924


namespace beavers_working_l162_162904

theorem beavers_working (a b : ℝ) (h₁ : a = 2.0) (h₂ : b = 1.0) : a + b = 3.0 := 
by 
  rw [h₁, h₂]
  norm_num

end beavers_working_l162_162904


namespace number_of_divisible_permutations_l162_162968

def digit_list := [1, 3, 1, 1, 5, 2, 1, 5, 2]
def count_permutations (d : List ℕ) (n : ℕ) : ℕ :=
  let fact := Nat.factorial
  let number := fact 8 / (fact 3 * fact 2 * fact 1)
  number

theorem number_of_divisible_permutations : count_permutations digit_list 2 = 3360 := by
  sorry

end number_of_divisible_permutations_l162_162968


namespace original_number_is_509_l162_162908

theorem original_number_is_509 (n : ℕ) (h : n - 5 = 504) : n = 509 :=
by {
    sorry
}

end original_number_is_509_l162_162908


namespace Norbs_age_l162_162451

def guesses : List ℕ := [24, 28, 30, 32, 36, 38, 41, 44, 47, 49]

def is_prime (n : ℕ) : Prop := Nat.Prime n

def two_off_by_one (n : ℕ) (guesses : List ℕ) : Prop := 
  (n - 1 ∈ guesses) ∧ (n + 1 ∈ guesses)

def at_least_half_too_low (n : ℕ) (guesses : List ℕ) : Prop := 
  (guesses.filter (· < n)).length ≥ guesses.length / 2

theorem Norbs_age : 
  ∃ x, is_prime x ∧ two_off_by_one x guesses ∧ at_least_half_too_low x guesses ∧ x = 37 := 
by 
  sorry

end Norbs_age_l162_162451


namespace simplify_polynomial_l162_162741

variable (y : ℤ)

theorem simplify_polynomial :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 6 * y^10 + 2 * y^9 + 4) = 
  15 * y^13 - y^12 + 12 * y^11 - 6 * y^10 - 4 * y^9 + 12 * y - 8 :=
by
  sorry

end simplify_polynomial_l162_162741


namespace schlaf_flachs_divisible_by_271_l162_162142

theorem schlaf_flachs_divisible_by_271 
(S C F H L A : ℕ) 
(hS : S ≠ 0) 
(hF : F ≠ 0) 
(hS_digit : S < 10)
(hC_digit : C < 10)
(hF_digit : F < 10)
(hH_digit : H < 10)
(hL_digit : L < 10)
(hA_digit : A < 10) :
  (100000 * S + 10000 * C + 1000 * H + 100 * L + 10 * A + F - 
   (100000 * F + 10000 * L + 1000 * A + 100 * C + 10 * H + S)) % 271 = 0 ↔ 
  C = L ∧ H = A := 
sorry

end schlaf_flachs_divisible_by_271_l162_162142


namespace problem_l162_162466

open Real

theorem problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
sorry

end problem_l162_162466


namespace reciprocal_of_neg_2023_l162_162624

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by sorry

end reciprocal_of_neg_2023_l162_162624


namespace smallest_sum_is_minus_half_l162_162539

def smallest_sum (x: ℝ) : ℝ := x^2 + x

theorem smallest_sum_is_minus_half : ∃ x : ℝ, ∀ y : ℝ, smallest_sum y ≥ smallest_sum (-1/2) :=
by
  use -1/2
  intros y
  sorry

end smallest_sum_is_minus_half_l162_162539


namespace maximize_fraction_l162_162594

theorem maximize_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 :=
sorry

end maximize_fraction_l162_162594


namespace range_of_a_l162_162959

-- Define the input conditions and requirements, and then state the theorem.
def is_acute_angle_cos_inequality (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2

theorem range_of_a (a : ℝ) :
  is_acute_angle_cos_inequality a 1 3 ∧ is_acute_angle_cos_inequality 1 3 a ∧
  is_acute_angle_cos_inequality 3 a 1 ↔ 2 * Real.sqrt 2 < a ∧ a < Real.sqrt 10 :=
by
  sorry

end range_of_a_l162_162959


namespace part1_max_value_part2_range_of_a_l162_162071

-- Definition of f(x) for general a
def f (a x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Part 1: Prove the maximum value of f(x) when a = 0
theorem part1_max_value (x : ℝ) (hx : x > 0) : f 0 x ≤ -1 := by
  sorry

-- Part 2: Prove the range of a such that f(x) has exactly one zero
theorem part2_range_of_a (a : ℝ) : (∃! x : ℝ, x > 0 ∧ f a x = 0) ↔ a ∈ Set.Ioi 0 := by
  sorry

end part1_max_value_part2_range_of_a_l162_162071


namespace intersection_A_B_l162_162964

def A : Set ℝ := { x | 1 < x - 1 ∧ x - 1 ≤ 3 }
def B : Set ℝ := { 2, 3, 4 }

theorem intersection_A_B : A ∩ B = {3, 4} := 
by 
  sorry

end intersection_A_B_l162_162964


namespace trajectory_eq_of_midpoint_l162_162350

theorem trajectory_eq_of_midpoint (x y m n : ℝ) (hM_on_circle : m^2 + n^2 = 1)
  (hP_midpoint : (2*x = 3 + m) ∧ (2*y = n)) :
  (2*x - 3)^2 + 4*y^2 = 1 := 
sorry

end trajectory_eq_of_midpoint_l162_162350


namespace max_value_when_a_zero_range_of_a_for_one_zero_l162_162087

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * Real.log x

theorem max_value_when_a_zero :
  ∃ x : ℝ, x > 0 ∧ f 0 x = -1 :=
by sorry

theorem range_of_a_for_one_zero :
  ∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ f a x = 0) ↔ a > 0 :=
by sorry

end max_value_when_a_zero_range_of_a_for_one_zero_l162_162087


namespace external_angle_theorem_proof_l162_162288

theorem external_angle_theorem_proof
    (x : ℝ)
    (FAB : ℝ)
    (BCA : ℝ)
    (ABC : ℝ)
    (h1 : FAB = 70)
    (h2 : BCA = 20 + x)
    (h3 : ABC = x + 20)
    (h4 : FAB = ABC + BCA) : 
    x = 15 :=
  by
  sorry

end external_angle_theorem_proof_l162_162288


namespace remainder_of_3_pow_2023_mod_5_l162_162481

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end remainder_of_3_pow_2023_mod_5_l162_162481


namespace third_row_number_l162_162828

-- Define the conditions to fill the grid
def grid (n : Nat) := Fin 4 → Fin 4 → Fin n

-- Ensure each number 1-4 in each cell such that numbers do not repeat
def unique_in_row (g : grid 4) : Prop :=
  ∀ i j1 j2, j1 ≠ j2 → g i j1 ≠ g i j2

def unique_in_col (g : grid 4) : Prop :=
  ∀ j i1 i2, i1 ≠ i2 → g i1 j ≠ g i1 j

-- Define the external hints condition, encapsulating the provided hints.
def hints_condition (g : grid 4) : Prop :=
  -- Example placeholders for hint conditions that would be expanded accordingly.
  g 0 0 = 3 ∨ g 0 1 = 2 -- First row hints interpreted constraints
  -- Additional hint conditions to be added accordingly

-- Prove the correct number formed by the numbers in the third row is 4213
theorem third_row_number (g : grid 4) :
  unique_in_row g ∧ unique_in_col g ∧ hints_condition g →
  (g 2 0 = 4 ∧ g 2 1 = 2 ∧ g 2 2 = 1 ∧ g 2 3 = 3) :=
by
  sorry

end third_row_number_l162_162828


namespace minimum_workers_needed_to_make_profit_l162_162033

-- Given conditions
def fixed_maintenance_fee : ℝ := 550
def setup_cost : ℝ := 200
def wage_per_hour : ℝ := 18
def widgets_per_worker_per_hour : ℝ := 6
def sell_price_per_widget : ℝ := 3.5
def work_hours_per_day : ℝ := 8

-- Definitions derived from conditions
def daily_wage_per_worker := wage_per_hour * work_hours_per_day
def daily_revenue_per_worker := widgets_per_worker_per_hour * work_hours_per_day * sell_price_per_widget
def total_daily_cost (n : ℝ) := fixed_maintenance_fee + setup_cost + n * daily_wage_per_worker

-- Prove that the number of workers needed to make a profit is at least 32
theorem minimum_workers_needed_to_make_profit (n : ℕ) (h : (total_daily_cost (n : ℝ)) < n * daily_revenue_per_worker) :
  n ≥ 32 := by
  -- We fill the sorry for proof to pass Lean check
  sorry

end minimum_workers_needed_to_make_profit_l162_162033


namespace final_score_correct_l162_162792

def innovation_score : ℕ := 88
def comprehensive_score : ℕ := 80
def language_score : ℕ := 75

def weight_innovation : ℕ := 5
def weight_comprehensive : ℕ := 3
def weight_language : ℕ := 2

def final_score : ℕ :=
  (innovation_score * weight_innovation + comprehensive_score * weight_comprehensive +
   language_score * weight_language) /
  (weight_innovation + weight_comprehensive + weight_language)

theorem final_score_correct :
  final_score = 83 :=
by
  -- proof goes here
  sorry

end final_score_correct_l162_162792


namespace silver_dollars_l162_162439

variable (C : ℕ)
variable (H : ℕ)
variable (P : ℕ)

theorem silver_dollars (h1 : H = P + 5) (h2 : P = C + 16) (h3 : C + P + H = 205) : C = 56 :=
by
  sorry

end silver_dollars_l162_162439


namespace gear_ratios_l162_162681

variable (x y z w : ℝ)
variable (ω_A ω_B ω_C ω_D : ℝ)
variable (k : ℝ)

theorem gear_ratios (h : x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D) : 
    ω_A/ω_B = yzw/xzw ∧ ω_B/ω_C = xzw/xyw ∧ ω_C/ω_D = xyw/xyz ∧ ω_A/ω_C = yzw/xyw := 
sorry

end gear_ratios_l162_162681


namespace garden_length_l162_162317

theorem garden_length (w l : ℝ) (h1 : l = 2 + 3 * w) (h2 : 2 * l + 2 * w = 100) : l = 38 :=
sorry

end garden_length_l162_162317


namespace reciprocal_neg_5_l162_162324

theorem reciprocal_neg_5 : ∃ x : ℚ, -5 * x = 1 ∧ x = -1/5 :=
by
  sorry

end reciprocal_neg_5_l162_162324


namespace lunks_for_apples_l162_162265

noncomputable def lunks_per_kunks := 7 / 4
noncomputable def kunks_per_apples := 3 / 5
def apples_needed := 24
noncomputable def kunks_needed_for_apples := (kunks_per_apples * apples_needed).ceil
noncomputable def lunks_needed := (lunks_per_kunks * kunks_needed_for_apples).ceil

theorem lunks_for_apples :
  lunks_needed = 27 :=
sorry

end lunks_for_apples_l162_162265


namespace common_ratio_geometric_sequence_l162_162396

-- Definition of a geometric sequence and given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_geo : is_geometric_sequence a)
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 1 / 4) : q = 1 / 2 :=
by 
  sorry

end common_ratio_geometric_sequence_l162_162396


namespace no_set_of_9_numbers_l162_162785

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end no_set_of_9_numbers_l162_162785


namespace magnitude_quotient_l162_162817

open Complex

theorem magnitude_quotient : 
  abs ((1 + 2 * I) / (2 - I)) = 1 := 
by 
  sorry

end magnitude_quotient_l162_162817


namespace coeff_x5_in_expansion_l162_162831

-- Define the problem statement as a theorem in Lean 4
theorem coeff_x5_in_expansion :
  (Nat.choose 8 4) - (Nat.choose 8 5) = 14 :=
sorry

end coeff_x5_in_expansion_l162_162831


namespace find_f_of_power_function_l162_162464

theorem find_f_of_power_function (a : ℝ) (alpha : ℝ) (f : ℝ → ℝ) 
  (h1 : 0 < a ∧ a ≠ 1) 
  (h2 : ∀ x, f x = x^alpha) 
  (h3 : ∀ x, a^(x-2) + 3 = f (2)): 
  f 2 = 4 := 
  sorry

end find_f_of_power_function_l162_162464


namespace polynomial_coefficient_sum_l162_162097

theorem polynomial_coefficient_sum :
  ∃ A B C D : ℝ, (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) ∧ A + B + C + D = 36 :=
begin
  -- Definitions from the conditions.
  let f1 := λ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7),
  let f2 := λ A B C D x : ℝ, A * x^3 + B * x^2 + C * x + D,

  -- We claim the existence of such constants A, B, C, D and the condition A + B + C + D = 36.
  use [4, 10, 1, 21],
  split,
  {
    intro x,
    calc (x + 3) * (4 * x^2 - 2 * x + 7) = 4 * x^3 + 10 * x^2 + x + 21 : by ring,
  },
  {
    -- Verify the sum of these constants.
    norm_num,
  }
end

end polynomial_coefficient_sum_l162_162097


namespace f_max_a_zero_f_zero_range_l162_162079

-- Part 1: Proving the maximum value when a = 0
theorem f_max_a_zero : ∀ (f : ℝ → ℝ) (x : ℝ),
  (f x = (-1 / x) - Real.log x) ∧ x > 0 → 
  ∃ x_max : ℝ, f x_max = -1 ∧ 
  (∀ x > 0, f x ≤ -1) := 
sorry

-- Part 2: Proving the range of a for exactly one zero of f(x)
theorem f_zero_range (a : ℝ) : (0 < a) → 
  ∀ (f : ℝ → ℝ) (x : ℝ), 
  (f x = a * x - 1 / x - (a + 1) * Real.log x) ∧ x > 0 →
  ∃! (x_zero : ℝ), f x_zero = 0 :=
sorry

end f_max_a_zero_f_zero_range_l162_162079


namespace total_money_l162_162271

variable (Sally Jolly Molly : ℕ)

-- Conditions
def condition1 (Sally : ℕ) : Prop := Sally - 20 = 80
def condition2 (Jolly : ℕ) : Prop := Jolly + 20 = 70
def condition3 (Molly : ℕ) : Prop := Molly + 30 = 100

-- The theorem to prove
theorem total_money (h1: condition1 Sally)
                    (h2: condition2 Jolly)
                    (h3: condition3 Molly) :
  Sally + Jolly + Molly = 220 :=
by
  sorry

end total_money_l162_162271


namespace isosceles_triangle_perimeter_l162_162841

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6 ∨ a = 7) (h₂ : b = 6 ∨ b = 7) (h₃ : a ≠ b) :
  (2 * a + b = 19) ∨ (2 * b + a = 20) :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_perimeter_l162_162841


namespace domain_of_v_l162_162769

noncomputable def v (x : ℝ) : ℝ := 1 / (x^(1/3))

theorem domain_of_v : {x : ℝ | ∃ y, y = v x} = {x : ℝ | x ≠ 0} := by
  sorry

end domain_of_v_l162_162769


namespace min_value_fraction_l162_162834

theorem min_value_fraction (x : ℝ) (h : x > 9) : (x^2 + 81) / (x - 9) ≥ 27 := 
  sorry

end min_value_fraction_l162_162834


namespace other_root_l162_162702

theorem other_root (x : ℚ) (h : 48 * x^2 + 29 = 35 * x + 12) : x = 3 / 4 ∨ x = 1 / 3 := 
by {
  -- Proof can be filled in here
  sorry
}

end other_root_l162_162702


namespace probability_one_hits_correct_l162_162767

-- Define the probabilities for A hitting and B hitting
noncomputable def P_A : ℝ := 0.4
noncomputable def P_B : ℝ := 0.5

-- Calculate the required probability
noncomputable def probability_one_hits : ℝ :=
  P_A * (1 - P_B) + (1 - P_A) * P_B

-- Statement of the theorem
theorem probability_one_hits_correct :
  probability_one_hits = 0.5 := by 
  sorry

end probability_one_hits_correct_l162_162767


namespace max_men_with_all_items_l162_162719

theorem max_men_with_all_items (total_men married men_with_TV men_with_radio men_with_AC men_with_car men_with_smartphone : ℕ) 
  (H_married : married = 2300) 
  (H_TV : men_with_TV = 2100) 
  (H_radio : men_with_radio = 2600) 
  (H_AC : men_with_AC = 1800) 
  (H_car : men_with_car = 2500) 
  (H_smartphone : men_with_smartphone = 2200) : 
  ∃ m, m ≤ married ∧ m ≤ men_with_TV ∧ m ≤ men_with_radio ∧ m ≤ men_with_AC ∧ m ≤ men_with_car ∧ m ≤ men_with_smartphone ∧ m = 1800 := 
  sorry

end max_men_with_all_items_l162_162719


namespace sum_of_areas_of_tangent_circles_l162_162630

theorem sum_of_areas_of_tangent_circles :
  ∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧
    (r + s = 3) ∧
    (r + t = 4) ∧
    (s + t = 5) ∧
    π * (r^2 + s^2 + t^2) = 14 * π :=
by
  sorry

end sum_of_areas_of_tangent_circles_l162_162630


namespace yarn_for_second_ball_l162_162294

variable (first_ball second_ball third_ball : ℝ) (yarn_used : ℝ)

-- Conditions
variable (h1 : first_ball = second_ball / 2)
variable (h2 : third_ball = 3 * first_ball)
variable (h3 : third_ball = 27)

-- Question: Prove that the second ball used 18 feet of yarn.
theorem yarn_for_second_ball (h1 : first_ball = second_ball / 2) (h2 : third_ball = 3 * first_ball) (h3 : third_ball = 27) :
  second_ball = 18 := by
  sorry

end yarn_for_second_ball_l162_162294


namespace projection_is_correct_l162_162018

theorem projection_is_correct :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (4, -1)
  let p : ℝ × ℝ := (15/58, 35/58)
  let d : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)
  ∃ v : ℝ × ℝ, 
    (a.1 * v.1 + a.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧
    (b.1 * v.1 + b.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧ 
    (p.1 * d.1 + p.2 * d.2 = 0) :=
sorry

end projection_is_correct_l162_162018


namespace shortest_is_Bob_l162_162423

variable {Person : Type}
variable [LinearOrder Person]

variable (Amy Bob Carla Dan Eric : Person)

-- Conditions
variable (h1 : Amy > Carla)
variable (h2 : Dan < Eric)
variable (h3 : Dan > Bob)
variable (h4 : Eric < Carla)

theorem shortest_is_Bob : ∀ p : Person, p = Bob :=
by
  intro p
  sorry

end shortest_is_Bob_l162_162423


namespace star_evaluation_l162_162821

def star (X Y : ℚ) := (X + Y) / 4

theorem star_evaluation : star (star 3 8) 6 = 35 / 16 := by
  sorry

end star_evaluation_l162_162821


namespace evaluate_expression_l162_162222

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by sorry

end evaluate_expression_l162_162222


namespace selling_price_l162_162799

-- Definitions for conditions
variables (CP SP_loss SP_profit : ℝ)
variable (h1 : SP_loss = 0.8 * CP)
variable (h2 : SP_profit = 1.05 * CP)
variable (h3 : SP_profit = 11.8125)

-- Theorem statement to prove
theorem selling_price (h1 : SP_loss = 0.8 * CP) (h2 : SP_profit = 1.05 * CP) (h3 : SP_profit = 11.8125) :
  SP_loss = 9 := 
sorry

end selling_price_l162_162799


namespace problem_statement_l162_162584

-- Define the function g and specify its properties
def g : ℕ → ℕ := sorry

axiom g_property (a b : ℕ) : g (a^2 + b^2) + g (a + b) = (g a)^2 + (g b)^2

-- Define the values of m and t that arise from the constraints on g(49)
def m : ℕ := 2
def t : ℕ := 106

-- Prove that the product m * t is 212
theorem problem_statement : m * t = 212 :=
by {
  -- Since g_property is an axiom, we use it to derive that
  -- g(49) can only take possible values 0 and 106,
  -- thus m = 2 and t = 106.
  sorry
}

end problem_statement_l162_162584


namespace student_correct_answers_l162_162172

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 64) : C = 88 :=
by
  sorry

end student_correct_answers_l162_162172


namespace fourth_intersection_point_l162_162287

noncomputable def fourth_point_of_intersection : Prop :=
  let hyperbola (x y : ℝ) := x * y = 1
  let circle (x y : ℝ) := (x - 1)^2 + (y + 1)^2 = 10
  let known_points : List (ℝ × ℝ) := [(3, 1/3), (-4, -1/4), (1/2, 2)]
  let fourth_point := (-1/6, -6)
  (hyperbola 3 (1/3)) ∧ (hyperbola (-4) (-1/4)) ∧ (hyperbola (1/2) 2) ∧
  (circle 3 (1/3)) ∧ (circle (-4) (-1/4)) ∧ (circle (1/2) 2) ∧ 
  (hyperbola (-1/6) (-6)) ∧ (circle (-1/6) (-6)) ∧ 
  ∀ (x y : ℝ), (hyperbola x y) → (circle x y) → ((x, y) = fourth_point ∨ (x, y) ∈ known_points)
  
theorem fourth_intersection_point :
  fourth_point_of_intersection :=
sorry

end fourth_intersection_point_l162_162287


namespace find_f_2009_l162_162245

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom cond2 : f 1 = 2

theorem find_f_2009 : f 2009 = 2 := by
  sorry

end find_f_2009_l162_162245


namespace equidistant_point_on_x_axis_l162_162334

theorem equidistant_point_on_x_axis (x : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 0)) (hB : B = (3, 5)) :
  (Real.sqrt ((x - (-3))^2)) = (Real.sqrt ((x - 3)^2 + 25)) →
  x = 25 / 12 := 
by 
  sorry

end equidistant_point_on_x_axis_l162_162334


namespace four_x_sq_plus_nine_y_sq_l162_162261

theorem four_x_sq_plus_nine_y_sq (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 9)
  (h2 : x * y = -12) : 
  4 * x^2 + 9 * y^2 = 225 := 
by
  sorry

end four_x_sq_plus_nine_y_sq_l162_162261


namespace darwin_leftover_money_l162_162377

theorem darwin_leftover_money :
  ∀ (initial_amount gas_expense_ratio food_expense_ratio : ℕ),
    initial_amount = 600 →
    gas_expense_ratio = 3 →
    food_expense_ratio = 4 →
    let gas_expense := initial_amount / gas_expense_ratio in
    let remaining_after_gas := initial_amount - gas_expense in
    let food_expense := remaining_after_gas / food_expense_ratio in
    let remaining_after_food := remaining_after_gas - food_expense in
    remaining_after_food = 300 :=
by
  intro initial_amount gas_expense_ratio food_expense_ratio
  intro h_initial h_gas_ratio h_food_ratio
  let gas_expense := initial_amount / gas_expense_ratio
  let remaining_after_gas := initial_amount - gas_expense
  let food_expense := remaining_after_gas / food_expense_ratio
  let remaining_after_food := remaining_after_gas - food_expense
  sorry

end darwin_leftover_money_l162_162377


namespace determine_n_l162_162413

theorem determine_n (k : ℕ) (n : ℕ) (h1 : 21^k ∣ n) (h2 : 7^k - k^7 = 1) : n = 1 :=
sorry

end determine_n_l162_162413


namespace max_GREECE_val_l162_162214

variables (V E R I A G C : ℕ)
noncomputable def verify : Prop :=
  (V * 100 + E * 10 + R - (I * 10 + A)) = G^(R^E) * (G * 100 + R * 10 + E + E * 100 + C * 10 + E) ∧
  G ≠ 0 ∧ E ≠ 0 ∧ V ≠ 0 ∧ I ≠ 0 ∧
  V ≠ E ∧ V ≠ R ∧ V ≠ I ∧ V ≠ A ∧ V ≠ G ∧ V ≠ C ∧
  E ≠ R ∧ E ≠ I ∧ E ≠ A ∧ E ≠ G ∧ E ≠ C ∧
  R ≠ I ∧ R ≠ A ∧ R ≠ G ∧ R ≠ C ∧
  I ≠ A ∧ I ≠ G ∧ I ≠ C ∧
  A ≠ G ∧ A ≠ C ∧
  G ≠ C

theorem max_GREECE_val : ∃ V E R I A G C : ℕ, verify V E R I A G C ∧ (G * 100000 + R * 10000 + E * 1000 + E * 100 + C * 10 + E = 196646) :=
sorry

end max_GREECE_val_l162_162214


namespace taxi_ride_cost_l162_162192

-- Lean statement
theorem taxi_ride_cost (base_fare : ℝ) (rate1 : ℝ) (rate1_miles : ℝ) (rate2 : ℝ) (total_miles : ℝ) 
  (h_base_fare : base_fare = 2.00)
  (h_rate1 : rate1 = 0.30)
  (h_rate1_miles : rate1_miles = 3)
  (h_rate2 : rate2 = 0.40)
  (h_total_miles : total_miles = 8) :
  let rate1_cost := rate1 * rate1_miles
  let rate2_cost := rate2 * (total_miles - rate1_miles)
  base_fare + rate1_cost + rate2_cost = 4.90 := by
  sorry

end taxi_ride_cost_l162_162192


namespace determine_roles_l162_162444

/-
We have three inhabitants K, M, R.
One of them is a truth-teller (tt), one is a liar (l), 
and one is a trickster (tr).
K states: "I am a trickster."
M states: "That is true."
R states: "I am not a trickster."
A truth-teller always tells the truth.
A liar always lies.
A trickster sometimes lies and sometimes tells the truth.
-/

inductive Role
| truth_teller | liar | trickster

open Role

def inhabitant_role (K M R : Role) : Prop :=
  ((K = liar) ∧ (M = trickster) ∧ (R = truth_teller)) ∧
  (K = trickster → K ≠ K) ∧
  (M = truth_teller → M = truth_teller) ∧
  (R = trickster → R ≠ R)

theorem determine_roles (K M R : Role) : inhabitant_role K M R :=
sorry

end determine_roles_l162_162444


namespace cyclic_sum_inequality_l162_162392

noncomputable def cyclic_sum (f : ℝ → ℝ → ℝ) (x y z : ℝ) : ℝ :=
  f x y + f y z + f z x

theorem cyclic_sum_inequality
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hx : x = a + (1 / b) - 1) 
  (hy : y = b + (1 / c) - 1) 
  (hz : z = c + (1 / a) - 1)
  (hpx : x > 0) (hpy : y > 0) (hpz : z > 0) :
  cyclic_sum (fun x y => (x * y) / (Real.sqrt (x * y) + 2)) x y z ≥ 1 :=
sorry

end cyclic_sum_inequality_l162_162392


namespace power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l162_162494

theorem power_of_3_mod_5 (n : ℕ) : (3^n % 5) = 
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 4
  else if n % 4 = 3 then 2
  else 1 := by
  sorry

theorem remainder_of_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  have h : 2023 % 4 = 3 := by norm_num
  rw power_of_3_mod_5 2023
  simp [h]
  sorry

end power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l162_162494


namespace fraction_of_bread_slices_eaten_l162_162440

theorem fraction_of_bread_slices_eaten
    (total_slices : ℕ)
    (slices_used_for_sandwich : ℕ)
    (remaining_slices : ℕ)
    (slices_eaten_for_breakfast : ℕ)
    (h1 : total_slices = 12)
    (h2 : slices_used_for_sandwich = 2)
    (h3 : remaining_slices = 6)
    (h4 : total_slices - slices_used_for_sandwich - remaining_slices = slices_eaten_for_breakfast) :
    slices_eaten_for_breakfast / total_slices = 1 / 3 :=
sorry

end fraction_of_bread_slices_eaten_l162_162440


namespace mow_lawn_time_l162_162991

noncomputable def time_to_mow (length width swath_width overlap speed : ℝ) : ℝ :=
  let effective_swath := (swath_width - overlap) / 12 -- Convert inches to feet
  let strips_needed := width / effective_swath
  let total_distance := strips_needed * length
  total_distance / speed

theorem mow_lawn_time : time_to_mow 100 140 30 6 4500 = 1.6 :=
by
  sorry

end mow_lawn_time_l162_162991


namespace factory_X_bulbs_percentage_l162_162688

theorem factory_X_bulbs_percentage (p : ℝ) (hx : 0.59 * p + 0.65 * (1 - p) = 0.62) : p = 0.5 :=
sorry

end factory_X_bulbs_percentage_l162_162688


namespace radius_of_circle_is_zero_l162_162540

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 2 * x^2 - 8 * x + 2 * y^2 - 4 * y + 10 = 0

-- Define the goal: To prove that given this equation, the radius of the circle is 0
theorem radius_of_circle_is_zero :
  ∀ x y : ℝ, circle_eq x y → (x - 2)^2 + (y - 1)^2 = 0 :=
sorry

end radius_of_circle_is_zero_l162_162540


namespace pen_ratio_l162_162381

theorem pen_ratio (R J D : ℕ) (pen_cost : ℚ) (total_spent : ℚ) (total_pens : ℕ) 
  (hR : R = 4)
  (hJ : J = 3 * R)
  (h_total_spent : total_spent = 33)
  (h_pen_cost : pen_cost = 1.5)
  (h_total_pens : total_pens = total_spent / pen_cost)
  (h_pens_expr : D + J + R = total_pens) :
  D / J = 1 / 2 :=
by
  sorry

end pen_ratio_l162_162381


namespace ball_hits_ground_time_l162_162877

def ball_height (t : ℝ) : ℝ := -20 * t^2 + 30 * t + 60

theorem ball_hits_ground_time :
  ∃ t : ℝ, ball_height t = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
sorry

end ball_hits_ground_time_l162_162877


namespace number_of_topless_cubical_box_figures_l162_162611

def valid_placements : Finset (Finset Nat) :=
Finset.fromList [{1, 2}, {5, 6}]

def number_of_valid_placements : Finset (Finset Nat) :=
valid_placements

theorem number_of_topless_cubical_box_figures :
  number_of_valid_placements.card = 2 := by
  sorry

end number_of_topless_cubical_box_figures_l162_162611


namespace base_2_representation_of_123_l162_162642

theorem base_2_representation_of_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
sorry

end base_2_representation_of_123_l162_162642


namespace find_max_n_l162_162965

variables {α : Type*} [LinearOrderedField α]

-- Define the sum S_n of the first n terms of an arithmetic sequence
noncomputable def S_n (a d : α) (n : ℕ) : α := 
  (n : α) / 2 * (2 * a + (n - 1) * d)

-- Given conditions
variable {a d : α}
axiom S11_pos : S_n a d 11 > 0
axiom S12_neg : S_n a d 12 < 0

theorem find_max_n : ∃ (n : ℕ), ∀ k < n, S_n a d k ≤ S_n a d n ∧ (k ≠ n → S_n a d k < S_n a d n) :=
sorry

end find_max_n_l162_162965


namespace police_emergency_number_has_prime_gt_7_l162_162802

theorem police_emergency_number_has_prime_gt_7 (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
sorry

end police_emergency_number_has_prime_gt_7_l162_162802


namespace objective_function_range_l162_162845

theorem objective_function_range (x y : ℝ) 
  (h1 : x + 2 * y > 2) 
  (h2 : 2 * x + y ≤ 4) 
  (h3 : 4 * x - y ≥ 1) : 
  ∃ z_min z_max : ℝ, (∀ z : ℝ, z = 3 * x + y → z_min ≤ z ∧ z ≤ z_max) ∧ z_min = 1 ∧ z_max = 6 := 
sorry

end objective_function_range_l162_162845


namespace last_digit_fifth_power_l162_162448

theorem last_digit_fifth_power (R : ℤ) : (R^5 - R) % 10 = 0 := 
sorry

end last_digit_fifth_power_l162_162448


namespace negate_universal_to_existential_l162_162394

variable {f : ℝ → ℝ}

theorem negate_universal_to_existential :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔
  (∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0) :=
  sorry

end negate_universal_to_existential_l162_162394


namespace percentage_of_sikhs_is_10_l162_162109

-- Definitions based on the conditions
def total_boys : ℕ := 850
def percent_muslims : ℕ := 34
def percent_hindus : ℕ := 28
def other_community_boys : ℕ := 238

-- The problem statement to prove
theorem percentage_of_sikhs_is_10 :
  ((total_boys - ((percent_muslims * total_boys / 100) + (percent_hindus * total_boys / 100) + other_community_boys))
  * 100 / total_boys) = 10 := 
by
  sorry

end percentage_of_sikhs_is_10_l162_162109


namespace min_value_a_plus_2b_l162_162843

theorem min_value_a_plus_2b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_condition : (a + b) / (a * b) = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_a_plus_2b_l162_162843


namespace ratio_of_part_to_whole_l162_162301

theorem ratio_of_part_to_whole : 
  (1 / 4) * (2 / 5) * P = 15 → 
  (40 / 100) * N = 180 → 
  P / N = 1 / 6 := 
by
  intros h1 h2
  sorry

end ratio_of_part_to_whole_l162_162301


namespace product_of_solutions_l162_162683

theorem product_of_solutions (α β : ℝ) (h : 2 * α^2 + 8 * α - 45 = 0 ∧ 2 * β^2 + 8 * β - 45 = 0 ∧ α ≠ β) :
  α * β = -22.5 :=
sorry

end product_of_solutions_l162_162683


namespace play_role_assignments_l162_162187

def specific_role_assignments (men women remaining either_gender_roles : ℕ) : ℕ :=
  men * women * Nat.choose remaining either_gender_roles

theorem play_role_assignments :
  specific_role_assignments 6 7 11 4 = 13860 := by
  -- The given problem statement implies evaluating the specific role assignments
  sorry

end play_role_assignments_l162_162187


namespace intersection_M_N_l162_162256

def M : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def N : Set ℝ := {x | -4 < x ∧ x ≤ 2}
def intersection : Set ℝ := {-1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l162_162256


namespace proportion_solution_l162_162412

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 7 / 8) : x = 6 / 7 :=
by {
  sorry
}

end proportion_solution_l162_162412


namespace max_value_when_a_zero_exactly_one_zero_range_l162_162084

noncomputable def f (a x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * log x

theorem max_value_when_a_zero : 
  (∀ x > 0, f 0 x ≤ f 0 1) ∧ f 0 1 = -1 :=
by sorry

theorem exactly_one_zero_range : 
  (∀ a > 0, ∃! x > 0, f a x = 0) ∧ 
  (∀ a ≤ 0, ¬ ∃ x > 0, f a x = 0) :=
by sorry

end max_value_when_a_zero_exactly_one_zero_range_l162_162084


namespace volume_of_region_l162_162235

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region : 
  ∫∫∫ {xyz : ℝ × ℝ × ℝ | f xyz.1 xyz.2 xyz.3 ≤ 6} 1 = 27/2 := 
sorry

end volume_of_region_l162_162235


namespace not_rented_two_bedroom_units_l162_162974

theorem not_rented_two_bedroom_units (total_units : ℕ)
  (units_rented_ratio : ℚ)
  (total_rented_units : ℕ)
  (one_bed_room_rented_ratio two_bed_room_rented_ratio three_bed_room_rented_ratio : ℚ)
  (one_bed_room_rented_count two_bed_room_rented_count three_bed_room_rented_count : ℕ)
  (x : ℕ) 
  (total_two_bed_room_units rented_two_bed_room_units : ℕ)
  (units_ratio_condition : 2*x + 3*x + 4*x = total_rented_units)
  (total_units_condition : total_units = 1200)
  (ratio_condition : units_rented_ratio = 7/12)
  (rented_units_condition : total_rented_units = (7/12) * total_units)
  (one_bed_condition : one_bed_room_rented_ratio = 2/5)
  (two_bed_condition : two_bed_room_rented_ratio = 1/2)
  (three_bed_condition : three_bed_room_rented_ratio = 3/8)
  (one_bed_count : one_bed_room_rented_count = 2 * x)
  (two_bed_count : two_bed_room_rented_count = 3 * x)
  (three_bed_count : three_bed_room_rented_count = 4 * x)
  (x_value : x = total_rented_units / 9)
  (total_two_bed_units_calc : total_two_bed_room_units = 2 * two_bed_room_rented_count)
  : total_two_bed_room_units - two_bed_room_rented_count = 231 :=
  by
  sorry

end not_rented_two_bedroom_units_l162_162974


namespace max_area_rectangle_l162_162319

theorem max_area_rectangle (P : ℕ) (hP : P = 40) : ∃ A : ℕ, A = 100 ∧ ∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A := by
  sorry

end max_area_rectangle_l162_162319


namespace solve_trigonometric_problem_l162_162209

noncomputable def trigonometric_problem : Prop :=
  let θ := real.pi / 10  -- 18 degrees
  let η := 3 * real.pi / 10  -- 54 degrees
  let γ := 2 * real.pi / 5  -- 72 degrees
  let δ := real.pi / 5  -- 36 degrees
  sin θ * sin η * sin γ * sin δ = (real.sqrt 5 + 1) / 16

theorem solve_trigonometric_problem : trigonometric_problem :=
by
  sorry

end solve_trigonometric_problem_l162_162209


namespace least_x_divisible_by_3_l162_162896

theorem least_x_divisible_by_3 : ∃ x : ℕ, (∀ y : ℕ, (2 + 3 + 5 + 7 + y) % 3 = 0 → y = 1) :=
by
  sorry

end least_x_divisible_by_3_l162_162896


namespace monotonicity_and_extrema_l162_162732

open Real

noncomputable def f (x : ℝ) : ℝ := log (2 * x + 3) + x ^ 2

theorem monotonicity_and_extrema :
  (∀ x, -3 / 2 < x ∧ x < -1 → 0 < deriv f x) ∧
  (∀ x, -1 < x ∧ x < -1 / 2 → deriv f x < 0) ∧
  (∀ x, -1 / 2 < x → 0 < deriv f x) ∧
  ∃ x₁ x₂ x₃ x₄,
    x₁ = -3 / 4 ∧ x₂ = -1 / 2 ∧ x₃ = 1 / 4 ∧
    f x₁ = log (3 / 2) + 9 / 16 ∧
    f x₂ = log 2 + 1 / 4 ∧
    f x₃ = log (7 / 2) + 1 / 16 ∧
    (f x₃ = max (f x₁) (max (f x₂) (f x))) ∧
    (f x₂ = min (f x₁) (min (f x₂) (f x)))
        :=
begin
  sorry
end

end monotonicity_and_extrema_l162_162732


namespace total_cakes_served_l162_162808

-- Define the conditions
def cakes_lunch_today := 5
def cakes_dinner_today := 6
def cakes_yesterday := 3

-- Define the theorem we want to prove
theorem total_cakes_served : (cakes_lunch_today + cakes_dinner_today + cakes_yesterday) = 14 :=
by
  -- The proof is not required, so we use sorry to skip it
  sorry

end total_cakes_served_l162_162808


namespace complement_of_A_in_R_l162_162551

open Set

variable (R : Set ℝ) (A : Set ℝ)

def real_numbers : Set ℝ := {x | true}

def set_A : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem complement_of_A_in_R : (real_numbers \ set_A) = {y | y < 0} := by
  sorry

end complement_of_A_in_R_l162_162551


namespace instantaneous_velocity_at_2_l162_162316

def displacement (t : ℝ) : ℝ := 14 * t - t^2 

def velocity (t : ℝ) : ℝ :=
  sorry -- The velocity function which is the derivative of displacement

theorem instantaneous_velocity_at_2 :
  velocity 2 = 10 := 
  sorry

end instantaneous_velocity_at_2_l162_162316


namespace part1_solution_set_part2_range_a_l162_162092

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 1)

theorem part1_solution_set :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3 / 2} ∪ {x : ℝ | x ≥ 3 / 2} := 
sorry

theorem part2_range_a (a : ℝ) : 
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ (-1 ≤ a ∧ a ≤ 2) := 
sorry

end part1_solution_set_part2_range_a_l162_162092


namespace car_speed_travel_l162_162509

theorem car_speed_travel (v : ℝ) :
  600 = 3600 / 6 ∧
  (6 : ℝ) = (3600 / v) + 2 →
  v = 900 :=
by
  sorry

end car_speed_travel_l162_162509


namespace average_tree_height_l162_162421

theorem average_tree_height : 
  ∀ (T₁ T₂ T₃ T₄ T₅ T₆ : ℕ),
  T₂ = 27 ->
  ((T₁ = 3 * T₂) ∨ (T₁ = T₂ / 3)) ->
  ((T₃ = 3 * T₂) ∨ (T₃ = T₂ / 3)) ->
  ((T₄ = 3 * T₃) ∨ (T₄ = T₃ / 3)) ->
  ((T₅ = 3 * T₄) ∨ (T₅ = T₄ / 3)) ->
  ((T₆ = 3 * T₅) ∨ (T₆ = T₅ / 3)) ->
  (T₁ + T₂ + T₃ + T₄ + T₅ + T₆) / 6 = 22 := 
by 
  intros T₁ T₂ T₃ T₄ T₅ T₆ hT2 hT1 hT3 hT4 hT5 hT6
  sorry

end average_tree_height_l162_162421


namespace total_fishes_l162_162777

theorem total_fishes (Will_catfish : ℕ) (Will_eels : ℕ) (Henry_multiplier : ℕ) (Henry_return_fraction : ℚ) :
  Will_catfish = 16 → Will_eels = 10 → Henry_multiplier = 3 → Henry_return_fraction = 1 / 2 →
  (Will_catfish + Will_eels) + (Henry_multiplier * Will_catfish - (Henry_multiplier * Will_catfish / 2)) = 50 := 
by
  intros h1 h2 h3 h4
  sorry

end total_fishes_l162_162777


namespace determine_q_l162_162378

theorem determine_q (q : ℝ) (x1 x2 x3 x4 : ℝ) 
  (h_first_eq : x1^2 - 5 * x1 + q = 0 ∧ x2^2 - 5 * x2 + q = 0)
  (h_second_eq : x3^2 - 7 * x3 + 2 * q = 0 ∧ x4^2 - 7 * x4 + 2 * q = 0)
  (h_relation : x3 = 2 * x1) : 
  q = 6 :=
by
  sorry

end determine_q_l162_162378


namespace chickens_and_rabbits_l162_162794

theorem chickens_and_rabbits (c r : ℕ) 
    (h1 : c = 2 * r - 5)
    (h2 : 2 * c + r = 92) : ∃ c r : ℕ, (c = 2 * r - 5) ∧ (2 * c + r = 92) := 
by 
    -- proof steps
    sorry

end chickens_and_rabbits_l162_162794


namespace find_sum_of_abc_l162_162599

variable (a b c : ℝ)

-- Given conditions
axiom h1 : a^2 + a * b + b^2 = 1
axiom h2 : b^2 + b * c + c^2 = 3
axiom h3 : c^2 + c * a + a^2 = 4

-- Positivity constraints
axiom ha : a > 0
axiom hb : b > 0
axiom hc : c > 0

theorem find_sum_of_abc : a + b + c = Real.sqrt 7 := 
by
  sorry

end find_sum_of_abc_l162_162599


namespace solve_for_x_l162_162120

theorem solve_for_x (x : ℝ) (h1 : 8 * x^2 + 8 * x - 2 = 0) (h2 : 32 * x^2 + 68 * x - 8 = 0) : 
    x = 1 / 8 := 
    sorry

end solve_for_x_l162_162120


namespace geometric_series_sum_l162_162167

theorem geometric_series_sum :
  let a := 2
  let r := 2
  let n := 11
  let S := a * (r^n - 1) / (r - 1)
  S = 4094 := by
  sorry

end geometric_series_sum_l162_162167


namespace greatest_possible_red_points_l162_162735

theorem greatest_possible_red_points (R B : ℕ) (h1 : R + B = 25)
    (h2 : ∀ r1 r2, r1 < R → r2 < R → r1 ≠ r2 → ∃ (n : ℕ), (∃ b1 : ℕ, b1 < B) ∧ ¬∃ b2 : ℕ, b2 < B) :
  R ≤ 13 :=
by {
  sorry
}

end greatest_possible_red_points_l162_162735


namespace smaller_number_is_three_l162_162009

theorem smaller_number_is_three (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 36) : min x y = 3 :=
sorry

end smaller_number_is_three_l162_162009


namespace greatest_value_divisible_by_3_l162_162571

theorem greatest_value_divisible_by_3 :
  ∃ (a : ℕ), (168026 + 1000 * a) % 3 = 0 ∧ a ≤ 9 ∧ ∀ b : ℕ, (168026 + 1000 * b) % 3 = 0 → b ≤ 9 → a ≥ b :=
sorry

end greatest_value_divisible_by_3_l162_162571


namespace Charles_has_13_whistles_l162_162606

-- Conditions
def Sean_whistles : ℕ := 45
def more_whistles_than_Charles : ℕ := 32

-- Let C be the number of whistles Charles has
def C : ℕ := Sean_whistles - more_whistles_than_Charles

-- Theorem to be proven
theorem Charles_has_13_whistles : C = 13 := by
  -- skipping proof
  sorry

end Charles_has_13_whistles_l162_162606


namespace count_negative_terms_in_sequence_l162_162963

theorem count_negative_terms_in_sequence : 
  ∃ (s : List ℕ), (∀ n ∈ s, n^2 - 8*n + 12 < 0) ∧ s.length = 3 ∧ (∀ n ∈ s, 2 < n ∧ n < 6) :=
by
  sorry

end count_negative_terms_in_sequence_l162_162963


namespace original_decimal_l162_162438

variable (x : ℝ)

theorem original_decimal (h : x - x / 100 = 1.485) : x = 1.5 :=
sorry

end original_decimal_l162_162438


namespace initial_men_colouring_l162_162716

theorem initial_men_colouring (M : ℕ) : 
  (∀ m : ℕ, ∀ d : ℕ, ∀ l : ℕ, m * d = 48 * 2 → 8 * 0.75 = 6 → M = 4) :=
by
  sorry

end initial_men_colouring_l162_162716


namespace probability_of_multiple_l162_162195

open Nat

-- Define the conditions
def whole_numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 12}
def valid_square_numbers := {n : ℕ | n = 1 ∨ n = 4 ∨ n = 9}
def total_assignments := 12 * 11 * 10

-- Function to check if Al's number is a multiple of both Bill's and Cal's numbers
def is_multiple (al b c : ℕ) : Prop := al % b = 0 ∧ al % c = 0

noncomputable def valid_assignments : ℕ :=
  (if is_multiple 4 1 2 then 2 else 0) +
  (if is_multiple 4 2 1 then 2 else 0) +
  (if is_multiple 9 1 3 then 2 else 0) +
  (if is_multiple 9 3 1 then 2 else 0)

-- Probability calculation
def probability := valid_assignments.toRat / total_assignments.toRat

-- The actual proof statement
theorem probability_of_multiple : 
  probability = (1 : ℚ) / 330 :=
by
  sorry

end probability_of_multiple_l162_162195


namespace simplify_expression_l162_162993

variable (q : ℚ)

theorem simplify_expression :
  (2 * q^3 - 7 * q^2 + 3 * q - 4) + (5 * q^2 - 4 * q + 8) = 2 * q^3 - 2 * q^2 - q + 4 :=
by
  sorry

end simplify_expression_l162_162993


namespace polynomial_coefficient_sum_l162_162098

theorem polynomial_coefficient_sum :
  ∃ A B C D : ℝ, (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) ∧ A + B + C + D = 36 :=
begin
  -- Definitions from the conditions.
  let f1 := λ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7),
  let f2 := λ A B C D x : ℝ, A * x^3 + B * x^2 + C * x + D,

  -- We claim the existence of such constants A, B, C, D and the condition A + B + C + D = 36.
  use [4, 10, 1, 21],
  split,
  {
    intro x,
    calc (x + 3) * (4 * x^2 - 2 * x + 7) = 4 * x^3 + 10 * x^2 + x + 21 : by ring,
  },
  {
    -- Verify the sum of these constants.
    norm_num,
  }
end

end polynomial_coefficient_sum_l162_162098


namespace remainder_3_pow_2023_mod_5_l162_162488

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l162_162488


namespace quadratic_inequality_solution_l162_162417

variables {x p q : ℝ}

theorem quadratic_inequality_solution
  (h1 : ∀ x, x^2 + p * x + q < 0 ↔ -1/2 < x ∧ x < 1/3) : 
  ∀ x, q * x^2 + p * x + 1 > 0 ↔ -2 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l162_162417


namespace market_value_of_house_l162_162291

theorem market_value_of_house 
  (M : ℝ) -- Market value of the house
  (S : ℝ) -- Selling price of the house
  (P : ℝ) -- Pre-tax amount each person gets
  (after_tax : ℝ := 135000) -- Each person's amount after taxes
  (tax_rate : ℝ := 0.10) -- Tax rate
  (num_people : ℕ := 4) -- Number of people splitting the revenue
  (over_market_value_rate : ℝ := 0.20): 
  S = M + over_market_value_rate * M → 
  (num_people * P) = S → 
  after_tax = (1 - tax_rate) * P → 
  M = 500000 := 
by
  sorry

end market_value_of_house_l162_162291


namespace overall_percentage_change_in_membership_l162_162811

theorem overall_percentage_change_in_membership :
  let M := 1
  let fall_inc := 1.08
  let winter_inc := 1.15
  let spring_dec := 0.81
  (M * fall_inc * winter_inc * spring_dec - M) / M * 100 = 24.2 := by
  sorry

end overall_percentage_change_in_membership_l162_162811


namespace max_value_of_f_l162_162143

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem max_value_of_f : 
  ∀ x ∈ (Set.Icc (Real.pi / 2) (3 * Real.pi / 2)), 
  f x ≤ (3 * Real.pi / 2 + 1) :=
by
  sorry

end max_value_of_f_l162_162143


namespace John_writing_years_l162_162860

def books_written (total_earnings per_book_earning : ℕ) : ℕ :=
  total_earnings / per_book_earning

def books_per_year (months_in_year months_per_book : ℕ) : ℕ :=
  months_in_year / months_per_book

def years_writing (total_books books_per_year : ℕ) : ℕ :=
  total_books / books_per_year

theorem John_writing_years :
  let total_earnings := 3600000
  let per_book_earning := 30000
  let months_in_year := 12
  let months_per_book := 2
  let total_books := books_written total_earnings per_book_earning
  let books_per_year := books_per_year months_in_year months_per_book
  years_writing total_books books_per_year = 20 := by
sorry

end John_writing_years_l162_162860


namespace hyperbola_condition_l162_162748

theorem hyperbola_condition (m : ℝ) : (∀ x y : ℝ, x^2 + m * y^2 = 1 → m < 0 ↔ x ≠ 0 ∧ y ≠ 0) :=
by
  sorry

end hyperbola_condition_l162_162748


namespace equivalent_statements_l162_162898

variable (P Q : Prop)

theorem equivalent_statements :
  (P → Q) ↔ (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end equivalent_statements_l162_162898


namespace prime_consecutive_fraction_equivalence_l162_162570

theorem prime_consecutive_fraction_equivalence (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hq_p_consec : p + 1 ≤ q ∧ Nat.Prime (p + 1) -> p + 1 = q) (hpq : p < q) (frac_eq : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := sorry

end prime_consecutive_fraction_equivalence_l162_162570


namespace areas_of_triangle_and_parallelogram_are_equal_l162_162810

theorem areas_of_triangle_and_parallelogram_are_equal (b : ℝ) :
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1/2) * b * triangle_height
  area_parallelogram = area_triangle :=
by
  -- conditions
  let parallelogram_height := 100
  let triangle_height := 200
  let area_parallelogram := b * parallelogram_height
  let area_triangle := (1 / 2) * b * triangle_height
  -- relationship
  show area_parallelogram = area_triangle
  sorry

end areas_of_triangle_and_parallelogram_are_equal_l162_162810


namespace largest_c_for_range_l162_162538

noncomputable def g (x c : ℝ) : ℝ := x^2 - 6*x + c

theorem largest_c_for_range (c : ℝ) : (∃ x : ℝ, g x c = 2) ↔ c ≤ 11 := 
sorry

end largest_c_for_range_l162_162538


namespace mary_brought_stickers_l162_162867

theorem mary_brought_stickers (friends_stickers : Nat) (other_stickers : Nat) (left_stickers : Nat) 
                              (total_students : Nat) (num_friends : Nat) (stickers_per_friend : Nat) 
                              (stickers_per_other_student : Nat) :
  friends_stickers = num_friends * stickers_per_friend →
  left_stickers = 8 →
  total_students = 17 →
  num_friends = 5 →
  stickers_per_friend = 4 →
  stickers_per_other_student = 2 →
  other_stickers = (total_students - 1 - num_friends) * stickers_per_other_student →
  (friends_stickers + other_stickers + left_stickers) = 50 :=
by
  intros
  sorry

end mary_brought_stickers_l162_162867


namespace exists_almost_square_divides_2010_l162_162151

noncomputable def almost_square (a b : ℕ) : Prop :=
  (a = b + 1 ∨ b = a + 1) ∧ a * b = 2010

theorem exists_almost_square_divides_2010 :
  ∃ (a b : ℕ), almost_square a b :=
sorry

end exists_almost_square_divides_2010_l162_162151


namespace water_lilies_half_pond_l162_162668

theorem water_lilies_half_pond (growth_rate : ℕ → ℕ) (start_day : ℕ) (full_covered_day : ℕ) 
  (h_growth : ∀ n, growth_rate (n + 1) = 2 * growth_rate n) 
  (h_start : growth_rate start_day = 1) 
  (h_full_covered : growth_rate full_covered_day = 2^(full_covered_day - start_day)) : 
  growth_rate (full_covered_day - 1) = 2^(full_covered_day - start_day - 1) :=
by
  sorry

end water_lilies_half_pond_l162_162668


namespace combined_soldiers_correct_l162_162580

-- Define the parameters for the problem
def interval : ℕ := 5
def wall_length : ℕ := 7300
def soldiers_per_tower : ℕ := 2

-- Calculate the number of towers and the total number of soldiers
def num_towers : ℕ := wall_length / interval
def combined_soldiers : ℕ := num_towers * soldiers_per_tower

-- Prove that the combined number of soldiers is as expected
theorem combined_soldiers_correct : combined_soldiers = 2920 := 
by
  sorry

end combined_soldiers_correct_l162_162580


namespace arithmetic_problem_l162_162773

theorem arithmetic_problem : 1357 + 3571 + 5713 - 7135 = 3506 :=
by
  sorry

end arithmetic_problem_l162_162773


namespace minimum_set_size_l162_162780

theorem minimum_set_size (n : ℕ) :
  (2 * n + 1) ≥ 11 :=
begin
  have h1 : 7 * (2 * n + 1) ≤ 15 * n + 2,
  sorry,
  have h2 : 14 * n + 7 ≤ 15 * n + 2,
  sorry,
  have h3 : n ≥ 5,
  sorry,
  show 2 * n + 1 ≥ 11,
  from calc
    2 * n + 1 = 2 * 5 + 1 : by linarith
          ... ≥ 11 : by linarith,
end

end minimum_set_size_l162_162780


namespace valid_third_side_l162_162957

-- Define a structure for the triangle with given sides
structure Triangle where
  a : ℝ
  b : ℝ
  x : ℝ

-- Define the conditions using the triangle inequality theorem
def valid_triangle (T : Triangle) : Prop :=
  T.a + T.x > T.b ∧ T.b + T.x > T.a ∧ T.a + T.b > T.x

-- Given values of a and b, and the condition on x
def specific_triangle : Triangle :=
  { a := 4, b := 9, x := 6 }

-- Statement to prove valid_triangle holds for specific_triangle
theorem valid_third_side : valid_triangle specific_triangle :=
by
  -- Import or assumptions about inequalities can be skipped or replaced by sorry
  sorry

end valid_third_side_l162_162957


namespace find_projection_l162_162017

noncomputable def projection_vector (v : ℝ × ℝ) (a b : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
a + (λ t, t • (b - a)) = p ∧ 
(p.1 * v.1 + p.2 * v.2 = 0)

theorem find_projection :
  ∃ p : ℝ × ℝ, 
  projection_vector (7, -3) (-3, 2) (4, -1) p ∧ p = (15/58, 35/58) :=
begin
  use (15/58, 35/58),
  sorry
end

end find_projection_l162_162017


namespace professor_D_error_l162_162196

noncomputable def polynomial_calculation_error (n : ℕ) : Prop :=
  ∃ (f : ℝ → ℝ), (∀ i : ℕ, i ≤ n+1 → f i = 2^i) ∧ f (n+2) ≠ 2^(n+2) - n - 3

theorem professor_D_error (n : ℕ) : polynomial_calculation_error n :=
  sorry

end professor_D_error_l162_162196


namespace car_speed_l162_162506

theorem car_speed (distance: ℚ) (hours minutes: ℚ) (h_distance: distance = 360) (h_hours: hours = 4) (h_minutes: minutes = 30) : 
  (distance / (hours + (minutes / 60))) = 80 := by
  sorry

end car_speed_l162_162506


namespace book_cost_l162_162241

theorem book_cost (p : ℝ) (h1 : 14 * p < 25) (h2 : 16 * p > 28) : 1.75 < p ∧ p < 1.7857 :=
by
  -- This is where the proof would go
  sorry

end book_cost_l162_162241


namespace power_mod_l162_162485

theorem power_mod : (3^2023) % 5 = 2 := by
  have h1 : 3^1 % 5 = 3 := by sorry
  have h2 : 3^2 % 5 = 4 := by sorry
  have h3 : 3^3 % 5 = 2 := by sorry
  have h4 : 3^4 % 5 = 1 := by sorry
  have periodicity : ∀ k: ℕ, (3^(4*k)) % 5 = 1 := by sorry
  have remainder : 2023 % 4 = 3 := by sorry
  show (3^2023) % 5 = 2 from by
    rw [Nat.mod_eq_of_lt 5 4] at remainder -- remainder shows 2023 mod 4 = 3
    exact h3

end power_mod_l162_162485


namespace train_length_calculation_l162_162040

noncomputable def length_of_train 
  (time : ℝ) (speed_train : ℝ) (speed_man : ℝ) : ℝ :=
  let speed_relative := speed_train - speed_man
  let speed_relative_mps := speed_relative * (5 / 18)
  speed_relative_mps * time

theorem train_length_calculation :
  length_of_train 29.997600191984642 63 3 = 1666.67 := 
by
  sorry

end train_length_calculation_l162_162040


namespace sum_of_digits_inequality_l162_162432

def sum_of_digits (n : ℕ) : ℕ := -- Definition of the sum of digits function
  -- This should be defined, for demonstration we use a placeholder
  sorry

theorem sum_of_digits_inequality (n : ℕ) (h : n > 0) :
  sum_of_digits n ≤ 8 * sum_of_digits (8 * n) :=
sorry

end sum_of_digits_inequality_l162_162432


namespace prove_n_eq_one_l162_162557

-- Definitions of the vectors a and b
def vector_a (n : ℝ) : ℝ × ℝ := (1, n)
def vector_b (n : ℝ) : ℝ × ℝ := (-1, n - 2)

-- Definition of collinearity between two vectors
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem to prove that if a and b are collinear, then n = 1
theorem prove_n_eq_one (n : ℝ) (h_collinear : collinear (vector_a n) (vector_b n)) : n = 1 :=
sorry

end prove_n_eq_one_l162_162557


namespace parabola_focus_coordinates_l162_162458

theorem parabola_focus_coordinates :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 18) ∧ 
    ∃ (p : ℝ), y = 9 * x^2 → x^2 = 4 * p * y ∧ p = 1 / 18 :=
by
  sorry

end parabola_focus_coordinates_l162_162458


namespace arithmetic_mean_12_24_36_48_l162_162152

theorem arithmetic_mean_12_24_36_48 : (12 + 24 + 36 + 48) / 4 = 30 :=
by
  sorry

end arithmetic_mean_12_24_36_48_l162_162152


namespace inequality_and_equality_conditions_l162_162696

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ abc ≤ 1 ∧ ((a + b + c = 3) → (a = 1 ∧ b = 1 ∧ c = 1)) := 
by 
  sorry

end inequality_and_equality_conditions_l162_162696


namespace optionC_is_quadratic_l162_162774

-- Define what it means to be a quadratic equation in one variable.
def isQuadraticInOneVariable (eq : Expr) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ eq = a * x^2 + b * x + c = 0

-- Define the given options
def optionA : Expr := x^2 + 1 / x^2 = 0
def optionB (a b c : ℝ) : Expr := a * x^2 + b * x + c = 0
def optionC : Expr := (x - 1) * (x + 2) = 1 
def optionD (y : ℝ) : Expr := 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- Define the proof problem
theorem optionC_is_quadratic :
  isQuadraticInOneVariable optionC :=
sorry

end optionC_is_quadratic_l162_162774


namespace length_of_train_l162_162194

theorem length_of_train (speed_kmh : ℝ) (time_min : ℝ) (tunnel_length_m : ℝ) (train_length_m : ℝ) :
  speed_kmh = 78 → time_min = 1 → tunnel_length_m = 500 → train_length_m = 800.2 :=
by
  sorry

end length_of_train_l162_162194


namespace part1_solution_set_m1_part2_find_m_l162_162728

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (m+1) * x^2 - m * x + m - 1

theorem part1_solution_set_m1 :
  { x : ℝ | f x 1 > 0 } = { x : ℝ | x < 0 } ∪ { x : ℝ | x > 0.5 } :=
by
  sorry

theorem part2_find_m :
  (∀ x : ℝ, f x m + 1 > 0 ↔ x > 1.5 ∧ x < 3) → m = -9/7 :=
by
  sorry

end part1_solution_set_m1_part2_find_m_l162_162728


namespace sin_6phi_l162_162566

theorem sin_6phi (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * (Real.sqrt 8)) / 5) : 
  Real.sin (6 * φ) = -198 * Real.sqrt 2 / 15625 :=
by
  sorry

end sin_6phi_l162_162566


namespace count_valid_pairs_l162_162544

open Nat

theorem count_valid_pairs : 
  (∑ y in Icc 1 147, (floor ((150 - y : ℤ) / (↑y * (y + 1) * (y + 2)))) : ℕ) = 33 :=
by
  sorry

end count_valid_pairs_l162_162544


namespace max_value_a_zero_range_a_one_zero_l162_162091

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - 1/x - (a + 1) * Real.log x

-- Problem (1): Prove that for a = 0, the maximum value of f(x) is -1.
theorem max_value_a_zero : ∃ x > (0:ℝ), ∀ y > 0, f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
by
  -- Proof not needed, inserting sorry
  sorry

-- Problem (2): Prove the range of values for a for which f(x) has exactly one zero is (0, +∞).
theorem range_a_one_zero : ∀ (a : ℝ), (∃ x > (0:ℝ), f a x = 0 ∧ ∀ y ≠ x, y > 0 → f a y ≠ 0) ↔ a > 0 :=
by
  -- Proof not needed, inserting sorry
  sorry

end max_value_a_zero_range_a_one_zero_l162_162091


namespace complex_division_example_l162_162457

theorem complex_division_example : (2 - (1 : ℂ) * Complex.I) / (1 - (1 : ℂ) * Complex.I) = (3 / 2) + (1 / 2) * Complex.I :=
by
  sorry

end complex_division_example_l162_162457


namespace boat_travel_distance_l162_162658

theorem boat_travel_distance
  (D : ℝ) -- Distance traveled in both directions
  (t : ℝ) -- Time in hours it takes to travel upstream
  (speed_boat : ℝ) -- Speed of the boat in still water
  (speed_stream : ℝ) -- Speed of the stream
  (time_diff : ℝ) -- Difference in time between downstream and upstream travel
  (h1 : speed_boat = 10)
  (h2 : speed_stream = 2)
  (h3 : time_diff = 1.5)
  (h4 : D = 8 * t)
  (h5 : D = 12 * (t - time_diff)) :
  D = 36 := by
  sorry

end boat_travel_distance_l162_162658


namespace remainder_of_division_l162_162496

def p (x : ℝ) : ℝ := 8*x^4 - 10*x^3 + 16*x^2 - 18*x + 5
def d (x : ℝ) : ℝ := 4*x - 8

theorem remainder_of_division :
  (p 2) = 81 :=
by
  sorry

end remainder_of_division_l162_162496


namespace smallest_five_digit_divisible_by_53_l162_162649

theorem smallest_five_digit_divisible_by_53 : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ 53 ∣ n ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_divisible_by_53_l162_162649


namespace find_k_for_quadratic_has_one_real_root_l162_162056

theorem find_k_for_quadratic_has_one_real_root (k : ℝ) : 
  (∃ x : ℝ, (3 * x - 4) * (x + 6) = -53 + k * x) ↔ (k = 14 + 2 * Real.sqrt 87 ∨ k = 14 - 2 * Real.sqrt 87) :=
sorry

end find_k_for_quadratic_has_one_real_root_l162_162056


namespace robert_total_balls_l162_162305

-- Define the conditions
def robert_initial_balls : ℕ := 25
def tim_balls : ℕ := 40

-- Mathematically equivalent proof problem
theorem robert_total_balls : 
  robert_initial_balls + (tim_balls / 2) = 45 := by
  sorry

end robert_total_balls_l162_162305


namespace rectangular_solid_edges_sum_l162_162631

theorem rectangular_solid_edges_sum
  (b s : ℝ)
  (h_vol : (b / s) * b * (b * s) = 432)
  (h_sa : 2 * ((b ^ 2 / s) + b ^ 2 * s + b ^ 2) = 432)
  (h_gp : 0 < s ∧ s ≠ 1) :
  4 * (b / s + b + b * s) = 144 := 
by
  sorry

end rectangular_solid_edges_sum_l162_162631


namespace negation_statement_l162_162852

theorem negation_statement (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) : x^2 - x ≠ 0 :=
by sorry

end negation_statement_l162_162852


namespace sum_of_distinct_prime_factors_of_462_l162_162163

-- Given a number n, define its prime factors.
def prime_factors (n : ℕ) : List ℕ :=
  if h : n = 462 then [2, 3, 7, 11] else []

-- Defines the sum of a list of natural numbers.
def sum_list (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

-- The main theorem statement.
theorem sum_of_distinct_prime_factors_of_462 : sum_list (prime_factors 462) = 23 :=
by
  sorry

end sum_of_distinct_prime_factors_of_462_l162_162163


namespace find_a20_l162_162029

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom a_arithmetic : ∀ n, a (n + 1) = a 1 + n * d
axiom a1_a3_a5_eq_105 : a 1 + a 3 + a 5 = 105
axiom a2_a4_a6_eq_99 : a 2 + a 4 + a 6 = 99

theorem find_a20 : a 20 = 1 :=
by sorry

end find_a20_l162_162029


namespace line_equation_l162_162704

noncomputable def line_intersects_at_point (a1 a2 b1 b2 c1 c2 : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 * a1 + p.2 * b1 = c1 ∧ p.1 * a2 + p.2 * b2 = c2

noncomputable def point_on_line (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

theorem line_equation
  (p : ℝ × ℝ)
  (h1 : line_intersects_at_point 3 2 2 3 5 5 p)
  (h2 : point_on_line 0 1 (-5) p)
  : ∃ a b c : ℝ,  a * p.1 + b * p.2 + (-5) = 0 :=
sorry

end line_equation_l162_162704


namespace max_value_of_f_when_a_is_zero_range_of_a_for_unique_zero_l162_162083

-- Part (1)
theorem max_value_of_f_when_a_is_zero (x : ℝ) (h : x > 0) : 
  let f := λ x, - (1 / x) - log x in
  (∀ x > 0, f x ≤ -1) ∧ (∃ x > 0, f x = -1) :=
sorry

-- Part (2)
theorem range_of_a_for_unique_zero (a : ℝ) :
  (∀ x : ℝ, f x = 0 ↔ x = 1) → a ∈ set.Ioi 0 :=
sorry

noncomputable def f (x a : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * log x

end max_value_of_f_when_a_is_zero_range_of_a_for_unique_zero_l162_162083


namespace number_of_divisors_2310_l162_162229

-- Define the number whose divisors are being counted
def n : ℕ := 2310

-- Define the prime factorization of the number
def factorization : n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num

-- Define the formula for the number of divisors
def num_divisors (n : ℕ) : ℕ :=
  let e_1 := 1
  let e_2 := 1
  let e_3 := 1
  let e_4 := 1
  let e_5 := 1
  in (e_1 + 1) * (e_2 + 1) * (e_3 + 1) * (e_4 + 1) * (e_5 + 1)

-- State the problem in a theorem
theorem number_of_divisors_2310 : num_divisors n = 32 :=
by
  rw [num_divisors, factorization]
  sorry

end number_of_divisors_2310_l162_162229


namespace intersection_is_correct_l162_162707

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_is_correct : A ∩ B = {2, 4, 8} :=
by
  sorry

end intersection_is_correct_l162_162707


namespace rear_revolutions_l162_162042

variable (r_r : ℝ)  -- radius of the rear wheel
variable (r_f : ℝ)  -- radius of the front wheel
variable (n_f : ℕ)  -- number of revolutions of the front wheel
variable (n_r : ℕ)  -- number of revolutions of the rear wheel

-- Condition: radius of the front wheel is 2 times the radius of the rear wheel.
axiom front_radius : r_f = 2 * r_r

-- Condition: the front wheel makes 10 revolutions.
axiom front_revolutions : n_f = 10

-- Theorem statement to prove
theorem rear_revolutions : n_r = 20 :=
sorry

end rear_revolutions_l162_162042


namespace min_elements_l162_162778

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end min_elements_l162_162778


namespace find_f_2010_l162_162463

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f x + f (3 - x)

theorem find_f_2010 : f 2010 = 0 := sorry

end find_f_2010_l162_162463


namespace two_trains_distance_before_meeting_l162_162813

noncomputable def distance_one_hour_before_meeting (speed_A speed_B : ℕ) : ℕ :=
  speed_A + speed_B

theorem two_trains_distance_before_meeting (speed_A speed_B total_distance : ℕ) (h_speed_A : speed_A = 60) (h_speed_B : speed_B = 40) (h_total_distance : total_distance ≤ 250) :
  distance_one_hour_before_meeting speed_A speed_B = 100 :=
by
  sorry

end two_trains_distance_before_meeting_l162_162813


namespace find_smaller_circle_radius_l162_162244

noncomputable def smaller_circle_radius (R : ℝ) : ℝ :=
  R / (Real.sqrt 2 - 1)

theorem find_smaller_circle_radius (R : ℝ) (x : ℝ) :
  (∀ (c1 c2 c3 c4 : ℝ),  c1 = c2 ∧ c2 = c3 ∧ c3 = c4 ∧ c4 = x
  ∧ c1 + c2 = 2 * c3 * Real.sqrt 2)
  → x = smaller_circle_radius R :=
by 
  intros h
  sorry

end find_smaller_circle_radius_l162_162244


namespace find_value_of_x_plus_5_l162_162995

-- Define a variable x
variable (x : ℕ)

-- Define the condition given in the problem
def condition := x - 10 = 15

-- The statement we need to prove
theorem find_value_of_x_plus_5 (h : x - 10 = 15) : x + 5 = 30 := 
by sorry

end find_value_of_x_plus_5_l162_162995


namespace set_intersection_l162_162556

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem set_intersection :
  A ∩ B = { x | -1 < x ∧ x < 1 } := 
sorry

end set_intersection_l162_162556


namespace lunks_for_apples_l162_162263

theorem lunks_for_apples : ∀ (lun_to_kun : ℕ) (num_lun : ℕ) (num_kun : ℕ) (kun_to_app : ℕ) (num_kun2 : ℕ) (num_app : ℕ),
  lun_to_kun = 7 ∧ num_kun = 4 ∧ kun_to_app = 3 ∧ num_kun2 = 5 ∧ num_app = 24 → 
  ((num_app * kun_to_app * num_lun / (num_kun2 * lun_to_kun)) ≤ 27) :=
by
  intros lun_to_kun num_lun num_kun kun_to_app num_kun2 num_app
  assume h_conditions
  sorry

end lunks_for_apples_l162_162263


namespace total_dog_weight_l162_162827

theorem total_dog_weight (weight_evans_dog weight_ivans_dog : ℕ)
  (h₁ : weight_evans_dog = 63)
  (h₂ : weight_evans_dog = 7 * weight_ivans_dog) :
  weight_evans_dog + weight_ivans_dog = 72 :=
sorry

end total_dog_weight_l162_162827


namespace theta_in_second_quadrant_l162_162697

theorem theta_in_second_quadrant
  (θ : ℝ)
  (h1 : Real.sin θ > 0)
  (h2 : Real.tan θ < 0) :
  (π / 2 < θ) ∧ (θ < π) :=
by
  sorry

end theta_in_second_quadrant_l162_162697


namespace number_of_special_permutations_l162_162295

noncomputable def count_special_permutations : ℕ :=
  (Nat.choose 12 6)

theorem number_of_special_permutations : count_special_permutations = 924 :=
  by
    sorry

end number_of_special_permutations_l162_162295


namespace find_value_l162_162364

theorem find_value
  (y1 y2 y3 y4 y5 : ℝ)
  (h1 : y1 + 4 * y2 + 9 * y3 + 16 * y4 + 25 * y5 = 3)
  (h2 : 4 * y1 + 9 * y2 + 16 * y3 + 25 * y4 + 36 * y5 = 20)
  (h3 : 9 * y1 + 16 * y2 + 25 * y3 + 36 * y4 + 49 * y5 = 150) :
  16 * y1 + 25 * y2 + 36 * y3 + 49 * y4 + 64 * y5 = 336 :=
by
  sorry

end find_value_l162_162364


namespace count_positive_integers_x_satisfying_inequality_l162_162692

theorem count_positive_integers_x_satisfying_inequality :
  ∃ n : ℕ, n = 6 ∧ (∀ x : ℕ, (144 ≤ x^2 ∧ x^2 ≤ 289) → (x = 12 ∨ x = 13 ∨ x = 14 ∨ x = 15 ∨ x = 16 ∨ x = 17)) :=
sorry

end count_positive_integers_x_satisfying_inequality_l162_162692


namespace ellipse_slope_product_l162_162840

theorem ellipse_slope_product (x₀ y₀ : ℝ) (hp : x₀^2 / 4 + y₀^2 / 3 = 1) :
  (y₀ / (x₀ + 2)) * (y₀ / (x₀ - 2)) = -3 / 4 :=
by
  -- The proof is omitted.
  sorry

end ellipse_slope_product_l162_162840


namespace robert_balls_l162_162307

theorem robert_balls (R T : ℕ) (hR : R = 25) (hT : T = 40 / 2) : R + T = 45 :=
by
  sorry

end robert_balls_l162_162307


namespace count_perfect_cubes_l162_162969

theorem count_perfect_cubes (a b : ℕ) (h₁ : 200 < a) (h₂ : a < 1500) (h₃ : b = 6^3) :
  (∃! n : ℕ, 200 < n^3 ∧ n^3 < 1500) :=
sorry

end count_perfect_cubes_l162_162969


namespace parallelogram_area_correct_l162_162627

noncomputable def parallelogram_area (a b : ℝ) (α : ℝ) (h : a < b) : ℝ :=
  (4 * a^2 - b^2) / 4 * (Real.tan α)

theorem parallelogram_area_correct (a b α : ℝ) (h : a < b) :
  parallelogram_area a b α h = (4 * a^2 - b^2) / 4 * (Real.tan α) :=
by
  sorry

end parallelogram_area_correct_l162_162627


namespace maximize_volume_l162_162346

-- Define the given dimensions
def length := 90
def width := 48

-- Define the volume function based on the height h
def volume (h : ℝ) : ℝ := h * (length - 2 * h) * (width - 2 * h)

-- Define the height that maximizes the volume
def optimal_height := 10

-- Define the maximum volume obtained at the optimal height
def max_volume := 19600

-- State the proof problem
theorem maximize_volume : 
  (∃ h : ℝ, volume h ≤ volume optimal_height) ∧
  volume optimal_height = max_volume := 
by
  sorry

end maximize_volume_l162_162346


namespace probability_passing_through_C_l162_162191

theorem probability_passing_through_C :
  (∀ p : rat, p = 1 →
  (∀ x y : rat, x = y / 2 →
  (∀ a b : rat, a = b / 2 →
  (p(C) = 21 / 32)))) :=
begin
  sorry
end

end probability_passing_through_C_l162_162191


namespace sum_of_roots_l162_162123

noncomputable def equation (x : ℝ) := 2 * (x^2 + 1 / x^2) - 3 * (x + 1 / x) = 1

theorem sum_of_roots (r s : ℝ) (hr : equation r) (hs : equation s) (hne : r ≠ s) :
  r + s = -5 / 2 :=
sorry

end sum_of_roots_l162_162123


namespace sum_of_hundreds_and_tens_digits_of_product_l162_162836

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def seq_num (a : ℕ) (x : ℕ) := List.foldr (λ _ acc => acc * 1000 + a) 0 (List.range x)

noncomputable def num_a := seq_num 707 101
noncomputable def num_b := seq_num 909 101

noncomputable def product := num_a * num_b

theorem sum_of_hundreds_and_tens_digits_of_product :
  hundreds_digit product + tens_digit product = 8 := by
  sorry

end sum_of_hundreds_and_tens_digits_of_product_l162_162836


namespace sum_of_distinct_prime_factors_of_462_l162_162164

-- Given a number n, define its prime factors.
def prime_factors (n : ℕ) : List ℕ :=
  if h : n = 462 then [2, 3, 7, 11] else []

-- Defines the sum of a list of natural numbers.
def sum_list (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

-- The main theorem statement.
theorem sum_of_distinct_prime_factors_of_462 : sum_list (prime_factors 462) = 23 :=
by
  sorry

end sum_of_distinct_prime_factors_of_462_l162_162164


namespace sale_price_after_discounts_l162_162663

def original_price : ℝ := 400.00
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.10
def discount3 : ℝ := 0.10

theorem sale_price_after_discounts (orig : ℝ) (d1 d2 d3 : ℝ) :
  orig = original_price →
  d1 = discount1 →
  d2 = discount2 →
  d3 = discount3 →
  orig * (1 - d1) * (1 - d2) * (1 - d3) = 243.00 := by
  sorry

end sale_price_after_discounts_l162_162663


namespace age_problem_l162_162351

theorem age_problem (A N : ℕ) (h₁: A = 18) (h₂: N * (A + 3) - N * (A - 3) = A) : N = 3 := by
  sorry

end age_problem_l162_162351


namespace parabola_c_value_l162_162917

theorem parabola_c_value (b c : ℝ) 
  (h1 : 6 = 2^2 + 2 * b + c) 
  (h2 : 20 = 4^2 + 4 * b + c) : 
  c = 0 :=
by {
  -- We state that we're skipping the proof
  sorry
}

end parabola_c_value_l162_162917


namespace eccentricity_of_hyperbola_l162_162405

theorem eccentricity_of_hyperbola {a b c e : ℝ} (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : b = 2 * a)
  (h₄ : c^2 = a^2 + b^2) :
  e = Real.sqrt 5 :=
by
  sorry

end eccentricity_of_hyperbola_l162_162405


namespace interest_rate_l162_162355

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

end interest_rate_l162_162355


namespace average_visitors_remaining_days_l162_162115

-- Definitions
def visitors_monday := 50
def visitors_tuesday := 2 * visitors_monday
def total_week_visitors := 250
def days_remaining := 5
def remaining_visitors := total_week_visitors - (visitors_monday + visitors_tuesday)
def average_remaining_visitors_per_day := remaining_visitors / days_remaining

-- Theorem statement
theorem average_visitors_remaining_days : average_remaining_visitors_per_day = 20 :=
by
  -- Proof is skipped
  sorry

end average_visitors_remaining_days_l162_162115


namespace age_difference_l162_162655

variable (P M Mo N : ℚ)

-- Given conditions as per problem statement
axiom ratio_P_M : (P / M) = 3 / 5
axiom ratio_M_Mo : (M / Mo) = 3 / 4
axiom ratio_Mo_N : (Mo / N) = 5 / 7
axiom sum_ages : P + M + Mo + N = 228

-- Statement to prove
theorem age_difference (ratio_P_M : (P / M) = 3 / 5)
                        (ratio_M_Mo : (M / Mo) = 3 / 4)
                        (ratio_Mo_N : (Mo / N) = 5 / 7)
                        (sum_ages : P + M + Mo + N = 228) :
  N - P = 69.5 := 
sorry

end age_difference_l162_162655


namespace find_a_l162_162058

def has_root_greater_than_zero (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ ((3 * x - 1) / (x - 3) = a / (3 - x) - 1)

theorem find_a (a : ℝ) : has_root_greater_than_zero a → a = -8 :=
by
  sorry

end find_a_l162_162058


namespace clock_strikes_twelve_l162_162558

def clock_strike_interval (strikes : Nat) (time : Nat) : Nat :=
  if strikes > 1 then time / (strikes - 1) else 0

def total_time_for_strikes (strikes : Nat) (interval : Nat) : Nat :=
  if strikes > 1 then (strikes - 1) * interval else 0

theorem clock_strikes_twelve (interval_six : Nat) (time_six : Nat) (time_twelve : Nat) :
  interval_six = clock_strike_interval 6 time_six →
  time_twelve = total_time_for_strikes 12 interval_six →
  time_six = 30 →
  time_twelve = 66 :=
by
  -- The proof will go here
  sorry

end clock_strikes_twelve_l162_162558


namespace train_pass_time_l162_162183

def speed_jogger := 9   -- in km/hr
def distance_ahead := 240   -- in meters
def length_train := 150   -- in meters
def speed_train := 45   -- in km/hr

noncomputable def time_to_pass_jogger : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := distance_ahead + length_train
  total_distance / relative_speed

theorem train_pass_time : time_to_pass_jogger = 39 :=
  by
    sorry

end train_pass_time_l162_162183


namespace prime_pairs_solution_l162_162023

theorem prime_pairs_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  7 * p * q^2 + p = q^3 + 43 * p^3 + 1 ↔ (p = 2 ∧ q = 7) :=
by
  sorry

end prime_pairs_solution_l162_162023


namespace find_a_minus_c_l162_162415

theorem find_a_minus_c (a b c : ℝ) (h1 : (a + b) / 2 = 110) (h2 : (b + c) / 2 = 170) : a - c = -120 :=
by
  sorry

end find_a_minus_c_l162_162415


namespace smallest_five_digit_divisible_by_53_l162_162648

theorem smallest_five_digit_divisible_by_53 : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ 53 ∣ n ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_divisible_by_53_l162_162648


namespace quadratic_inequality_solution_l162_162753

theorem quadratic_inequality_solution (x : ℝ) : (2 * x^2 - 5 * x - 3 < 0) ↔ (-1/2 < x ∧ x < 3) :=
by
  sorry

end quadratic_inequality_solution_l162_162753


namespace sum_of_prime_factors_462_eq_23_l162_162162

theorem sum_of_prime_factors_462_eq_23 : ∑ p in {2, 3, 7, 11}, p = 23 := by
  sorry

end sum_of_prime_factors_462_eq_23_l162_162162


namespace area_of_right_triangle_l162_162424

variable (a b : ℝ)

theorem area_of_right_triangle (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ (S : ℝ), S = a * b :=
sorry

end area_of_right_triangle_l162_162424


namespace digits_exceed_10_power_15_l162_162971

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem digits_exceed_10_power_15 (x : ℝ) 
  (h : log3 (log2 (log2 x)) = 3) : log10 x > 10^15 := 
sorry

end digits_exceed_10_power_15_l162_162971


namespace find_value_of_a_l162_162953

noncomputable def log_base_four (a : ℝ) : ℝ := Real.log a / Real.log 4

theorem find_value_of_a (a : ℝ) (h : log_base_four a = (1 : ℝ) / (2 : ℝ)) : a = 2 := by
  sorry

end find_value_of_a_l162_162953


namespace scale_length_l162_162189

theorem scale_length (num_parts : ℕ) (part_length : ℕ) (total_length : ℕ) 
  (h1 : num_parts = 5) (h2 : part_length = 16) : total_length = 80 :=
by
  sorry

end scale_length_l162_162189


namespace alice_preferred_numbers_l162_162362

def is_multiple_of_7 (n : ℕ) : Prop :=
  n % 7 = 0

def is_not_multiple_of_3 (n : ℕ) : Prop :=
  ¬ (n % 3 = 0)

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def alice_pref_num (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧ is_multiple_of_7 n ∧ is_not_multiple_of_3 n ∧ is_prime (digit_sum n)

theorem alice_preferred_numbers :
  ∀ n, alice_pref_num n ↔ n = 119 ∨ n = 133 ∨ n = 140 := 
sorry

end alice_preferred_numbers_l162_162362


namespace license_plate_problem_l162_162849

noncomputable def license_plate_ways : ℕ :=
  let letters := 26
  let digits := 10
  let both_same := letters * digits * 1 * 1
  let digits_adj_same := letters * digits * 1 * letters
  let letters_adj_same := letters * digits * digits * 1
  digits_adj_same + letters_adj_same - both_same

theorem license_plate_problem :
  9100 = license_plate_ways :=
by
  -- Skipping the detailed proof for now
  sorry

end license_plate_problem_l162_162849


namespace smaller_number_is_24_l162_162026

theorem smaller_number_is_24 (x y : ℕ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) : x = 24 :=
by
  sorry

end smaller_number_is_24_l162_162026


namespace reciprocal_of_neg_2023_l162_162626

theorem reciprocal_of_neg_2023 : ∃ x : ℝ, (-2023) * x = 1 ∧ x = -1 / 2023 := 
by {
  existsi (-1 / 2023),
  split,
  { -- Prove that the product of -2023 and -1/2023 is 1
    unfold has_mul.mul,
    norm_num,
  },
  { -- Prove that x is indeed -1/2023
    refl,
  }
}

end reciprocal_of_neg_2023_l162_162626


namespace sum_of_reciprocals_l162_162759

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 16) (h₂ : x * y = 55) :
  (1 / x + 1 / y) = 16 / 55 :=
by
  sorry

end sum_of_reciprocals_l162_162759


namespace smallest_five_digit_multiple_of_53_l162_162645

theorem smallest_five_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 53 = 0 ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_multiple_of_53_l162_162645


namespace least_lucky_multiple_of_six_not_lucky_l162_162911

def digit_sum (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % digit_sum n = 0

def is_multiple_of_six (n : ℕ) : Prop :=
  n % 6 = 0

theorem least_lucky_multiple_of_six_not_lucky : ∃ n, is_multiple_of_six n ∧ ¬ is_lucky_integer n ∧ (∀ m, is_multiple_of_six m ∧ ¬ is_lucky_integer m → n ≤ m) :=
by {
  use 12,
  split,
  { sorry },  -- Proof that 12 is a multiple of 6
  split,
  { sorry },  -- Proof that 12 is not a lucky integer
  { sorry },  -- Proof that there is no smaller multiple of 6 that is not a lucky integer
}

end least_lucky_multiple_of_six_not_lucky_l162_162911


namespace find_10th_integer_l162_162572

-- Defining the conditions
def avg_20_consecutive_integers (avg : ℝ) : Prop :=
  avg = 23.65

def consecutive_integer_sequence (n : ℤ) (a : ℤ) : Prop :=
  a = n + 9

-- The main theorem statement
theorem find_10th_integer (n : ℤ) (avg : ℝ) (h_avg : avg_20_consecutive_integers avg) (h_seq : consecutive_integer_sequence n 23) :
  n = 14 :=
sorry

end find_10th_integer_l162_162572


namespace find_divisor_l162_162973

-- Definitions from the conditions
def remainder : ℤ := 8
def quotient : ℤ := 43
def dividend : ℤ := 997
def is_prime (n : ℤ) : Prop := n ≠ 1 ∧ (∀ d : ℤ, d ∣ n → d = 1 ∨ d = n)

-- The proof problem statement
theorem find_divisor (d : ℤ) 
  (hd : is_prime d) 
  (hdiv : dividend = (d * quotient) + remainder) : 
  d = 23 := 
sorry

end find_divisor_l162_162973


namespace eu_countries_2012_forms_set_l162_162019

def higher_level_skills_students := false -- Condition A can't form a set.
def tall_trees := false -- Condition B can't form a set.
def developed_cities := false -- Condition D can't form a set.
def eu_countries_2012 := true -- Condition C forms a set.

theorem eu_countries_2012_forms_set : 
  higher_level_skills_students = false ∧ tall_trees = false ∧ developed_cities = false ∧ eu_countries_2012 = true :=
by {
  sorry
}

end eu_countries_2012_forms_set_l162_162019


namespace discount_policy_l162_162475

-- Define the prices of the fruits
def lemon_price := 2
def papaya_price := 1
def mango_price := 4

-- Define the quantities Tom buys
def lemons_bought := 6
def papayas_bought := 4
def mangos_bought := 2

-- Define the total amount paid by Tom
def amount_paid := 21

-- Define the total number of fruits bought
def total_fruits_bought := lemons_bought + papayas_bought + mangos_bought

-- Define the total cost without discount
def total_cost_without_discount := 
  (lemons_bought * lemon_price) + 
  (papayas_bought * papaya_price) + 
  (mangos_bought * mango_price)

-- Calculate the discount
def discount := total_cost_without_discount - amount_paid

-- The discount policy
theorem discount_policy : discount = 3 ∧ total_fruits_bought = 12 :=
by 
  sorry

end discount_policy_l162_162475


namespace sin_x_eq_x_div_100_has_63_roots_l162_162970

noncomputable def count_sin_eq_x_div_100_roots : ℕ := 63

theorem sin_x_eq_x_div_100_has_63_roots :
  let f x := sin x - x / 100 in
  (∀ x : ℝ, f x = 0 ↔ x ∈ Icc (-100) 100) ∧
  ∃ S : Finset ℝ, (S.card = count_sin_eq_x_div_100_roots) ∧ (∀ x ∈ S, f x = 0) :=
by sorry

end sin_x_eq_x_div_100_has_63_roots_l162_162970


namespace distance_big_rock_correct_l162_162341

noncomputable def rower_in_still_water := 7 -- km/h
noncomputable def river_flow := 2 -- km/h
noncomputable def total_trip_time := 1 -- hour

def distance_to_big_rock (D : ℝ) :=
  (D / (rower_in_still_water - river_flow)) + (D / (rower_in_still_water + river_flow)) = total_trip_time

theorem distance_big_rock_correct {D : ℝ} (h : distance_to_big_rock D) : D = 45 / 14 :=
sorry

end distance_big_rock_correct_l162_162341


namespace reciprocal_of_neg_five_l162_162326

theorem reciprocal_of_neg_five : ∃ b : ℚ, (-5) * b = 1 ∧ b = -1/5 :=
by
  sorry

end reciprocal_of_neg_five_l162_162326


namespace petri_dishes_count_l162_162723

def germs_total : ℕ := 5400000
def germs_per_dish : ℕ := 500
def petri_dishes : ℕ := germs_total / germs_per_dish

theorem petri_dishes_count : petri_dishes = 10800 := by
  sorry

end petri_dishes_count_l162_162723


namespace probability_r_div_k_is_square_l162_162465

open Set

def r_set : Set ℤ := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def k_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem probability_r_div_k_is_square :
  let total_pairs := (r_set.card : ℚ) * (k_set.card : ℚ)
  let square_r_set := {r ∈ r_set | ∃ k ∈ k_set, (r : ℚ) = k^2}
in (square_r_set.card : ℚ) / total_pairs = 8/63 :=
by
  sorry

end probability_r_div_k_is_square_l162_162465


namespace molecular_weight_of_9_moles_CCl4_l162_162480

-- Define the atomic weight of Carbon (C) and Chlorine (Cl)
def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45

-- Define the molecular formula for carbon tetrachloride (CCl4)
def molecular_formula_CCl4 : ℝ := atomic_weight_C + (4 * atomic_weight_Cl)

-- Define the molecular weight of one mole of carbon tetrachloride (CCl4)
def molecular_weight_one_mole_CCl4 : ℝ := molecular_formula_CCl4

-- Define the number of moles
def moles_CCl4 : ℝ := 9

-- Define the result to check
def molecular_weight_nine_moles_CCl4 : ℝ := molecular_weight_one_mole_CCl4 * moles_CCl4

-- State the theorem to prove the molecular weight of 9 moles of carbon tetrachloride is 1384.29 grams
theorem molecular_weight_of_9_moles_CCl4 :
  molecular_weight_nine_moles_CCl4 = 1384.29 := by
  sorry

end molecular_weight_of_9_moles_CCl4_l162_162480


namespace reciprocal_of_neg_2023_l162_162622

-- Define the number and its proposed reciprocal
def x : ℤ := -2023
def r : ℚ := -1 / 2023

-- State the theorem that the reciprocal of x is r
theorem reciprocal_of_neg_2023: x * r = 1 := by
  sorry

end reciprocal_of_neg_2023_l162_162622


namespace prasanna_speed_l162_162430

variable (L_speed P_speed time apart : ℝ)
variable (h1 : L_speed = 40)
variable (h2 : time = 1)
variable (h3 : apart = 78)

theorem prasanna_speed :
  P_speed = apart - (L_speed * time) / time := 
by
  rw [h1, h2, h3]
  simp
  sorry

end prasanna_speed_l162_162430


namespace original_price_l162_162127

theorem original_price (P : ℝ) (h₁ : 0.30 * P = 46) : P = 153.33 :=
  sorry

end original_price_l162_162127


namespace find_fourth_student_in_sample_l162_162280

theorem find_fourth_student_in_sample :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 48 ∧ 
           (∀ (k : ℕ), k = 29 → 1 ≤ k ∧ k ≤ 48 ∧ ((k = 5 + 2 * 12) ∨ (k = 41 - 12)) ∧ n = 17) :=
sorry

end find_fourth_student_in_sample_l162_162280


namespace intersection_C_U_M_N_l162_162408

open Set

-- Define U, M and N
def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

-- Define complement C_U M in U
def C_U_M : Set ℕ := U \ M

-- The theorem to prove
theorem intersection_C_U_M_N : (C_U_M ∩ N) = {3} := by
  sorry

end intersection_C_U_M_N_l162_162408


namespace skateboarder_speed_l162_162449

theorem skateboarder_speed :
  let distance := 293.33
  let time := 20
  let feet_per_mile := 5280
  let seconds_per_hour := 3600
  let speed_ft_per_sec := distance / time
  let speed_mph := speed_ft_per_sec * (feet_per_mile / seconds_per_hour)
  speed_mph = 21.5 :=
by
  sorry

end skateboarder_speed_l162_162449


namespace max_area_rectangle_l162_162320

theorem max_area_rectangle (x : ℝ) (h : 2 * x + 2 * (20 - x) = 40) : 
  ∃ (A : ℝ), A = 100 ∧ (∀ y, (A_area y h) ≤ 100) :=
by
  sorry

end max_area_rectangle_l162_162320


namespace perpendicular_lines_condition_l162_162147

theorem perpendicular_lines_condition (a : ℝ) :
  (6 * a + 3 * 4 = 0) ↔ (a = -2) :=
sorry

end perpendicular_lines_condition_l162_162147


namespace quadratic_inequality_solution_l162_162758

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end quadratic_inequality_solution_l162_162758


namespace number_consisting_of_11_hundreds_11_tens_and_11_units_l162_162128

theorem number_consisting_of_11_hundreds_11_tens_and_11_units :
  11 * 100 + 11 * 10 + 11 = 1221 :=
by
  sorry

end number_consisting_of_11_hundreds_11_tens_and_11_units_l162_162128


namespace president_vice_committee_count_l162_162283

open Nat

noncomputable def choose_president_vice_committee (total_people : ℕ) : ℕ :=
  let choose : ℕ := 56 -- binomial 8 3 is 56
  total_people * (total_people - 1) * choose

theorem president_vice_committee_count :
  choose_president_vice_committee 10 = 5040 :=
by
  sorry

end president_vice_committee_count_l162_162283


namespace car_speed_l162_162507

-- Definitions from conditions
def distance : ℝ := 360
def time : ℝ := 4.5

-- Statement to prove
theorem car_speed : (distance / time) = 80 := by
  sorry

end car_speed_l162_162507


namespace max_value_proof_l162_162005

noncomputable def max_expression_value (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_sum : a + b + c + d ≤ 4) : ℝ :=
  let expr := Real.root 4 (a^2 * (a + b)) +
              Real.root 4 (b^2 * (b + c)) +
              Real.root 4 (c^2 * (c + d)) +
              Real.root 4 (d^2 * (d + a))
  in expr

theorem max_value_proof (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_sum : a + b + c + d ≤ 4) :
  max_expression_value a b c d h_pos h_sum ≤ 4 * Real.root 4 2 :=
sorry

end max_value_proof_l162_162005


namespace find_numbers_l162_162933

theorem find_numbers (n : ℕ) (h1 : n ≥ 2) (a : ℕ) (ha : a ≠ 1) (ha_min : ∀ d, d ∣ n → d ≠ 1 → a ≤ d) (b : ℕ) (hb : b ∣ n) :
  n = a^2 + b^2 ↔ n = 8 ∨ n = 20 :=
by sorry

end find_numbers_l162_162933


namespace rational_function_domain_l162_162822

noncomputable def h (x : ℝ) : ℝ := (x^3 - 3*x^2 - 4*x + 5) / (x^2 - 5*x + 4)

theorem rational_function_domain :
  {x : ℝ | ∃ y, h y = h x } = {x : ℝ | x ≠ 1 ∧ x ≠ 4} := 
sorry

end rational_function_domain_l162_162822


namespace reciprocal_of_neg_2023_l162_162625

theorem reciprocal_of_neg_2023 : ∃ x : ℝ, (-2023) * x = 1 ∧ x = -1 / 2023 := 
by {
  existsi (-1 / 2023),
  split,
  { -- Prove that the product of -2023 and -1/2023 is 1
    unfold has_mul.mul,
    norm_num,
  },
  { -- Prove that x is indeed -1/2023
    refl,
  }
}

end reciprocal_of_neg_2023_l162_162625


namespace closest_correct_option_l162_162393

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f x = f (-x + 16)) -- y = f(x + 8) is an even function
variable (h2 : ∀ a b, 8 < a → 8 < b → a < b → f b < f a) -- f is decreasing on (8, +∞)

theorem closest_correct_option :
  f 7 > f 10 := by
  -- Insert proof here
  sorry

end closest_correct_option_l162_162393


namespace harry_terry_difference_l162_162410

theorem harry_terry_difference :
  let H := 12 - (3 + 6)
  let T := 12 - 3 + 6 * 2
  H - T = -18 :=
by
  sorry

end harry_terry_difference_l162_162410


namespace ratio_Cheryl_C_to_Cyrus_Y_l162_162525

noncomputable def Cheryl_C : ℕ := 126
noncomputable def Madeline_M : ℕ := 63
noncomputable def Total_pencils : ℕ := 231
noncomputable def Cyrus_Y : ℕ := Total_pencils - Cheryl_C - Madeline_M

theorem ratio_Cheryl_C_to_Cyrus_Y : 
  Cheryl_C = 2 * Madeline_M → 
  Madeline_M + Cheryl_C + Cyrus_Y = Total_pencils → 
  Cheryl_C / Cyrus_Y = 3 :=
by
  intros h1 h2
  sorry

end ratio_Cheryl_C_to_Cyrus_Y_l162_162525


namespace boxcar_capacity_l162_162136

theorem boxcar_capacity : 
  let red_count := 3 in
  let blue_count := 4 in
  let black_count := 7 in
  let black_capacity := 4000 in
  let blue_capacity := 2 * black_capacity in
  let red_capacity := 3 * blue_capacity in
  (red_count * red_capacity + blue_count * blue_capacity + black_count * black_capacity = 132000) :=
by
  let red_count := 3
  let blue_count := 4
  let black_count := 7
  let black_capacity := 4000
  let blue_capacity := 2 * black_capacity
  let red_capacity := 3 * blue_capacity
  have h1 : red_count * red_capacity + blue_count * blue_capacity + black_count * black_capacity = 132000 := by
    sorry
  exact h1

end boxcar_capacity_l162_162136


namespace slope_of_chord_in_ellipse_l162_162426

noncomputable def slope_of_chord (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem slope_of_chord_in_ellipse :
  ∀ (x1 y1 x2 y2 : ℝ),
    (x1^2 / 16 + y1^2 / 9 = 1) →
    (x2^2 / 16 + y2^2 / 9 = 1) →
    ((x1 + x2) = -2) →
    ((y1 + y2) = 4) →
    slope_of_chord x1 y1 x2 y2 = 9 / 32 :=
by
  intro x1 y1 x2 y2 h1 h2 h3 h4
  sorry

end slope_of_chord_in_ellipse_l162_162426


namespace smallest_five_digit_divisible_by_53_l162_162650

theorem smallest_five_digit_divisible_by_53 : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ 53 ∣ n ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_divisible_by_53_l162_162650


namespace exterior_angle_decreases_l162_162931

theorem exterior_angle_decreases (n : ℕ) (hn : n ≥ 3) (n' : ℕ) (hn' : n' ≥ n) :
  (360 : ℝ) / n' < (360 : ℝ) / n := by sorry

end exterior_angle_decreases_l162_162931


namespace tan_half_alpha_l162_162390

theorem tan_half_alpha (α : ℝ) (h1 : 180 * (Real.pi / 180) < α) 
  (h2 : α < 270 * (Real.pi / 180)) 
  (h3 : Real.sin ((270 * (Real.pi / 180)) + α) = 4 / 5) : 
  Real.tan (α / 2) = -1 / 3 :=
by 
  -- Informal note: proof would be included here.
  sorry

end tan_half_alpha_l162_162390


namespace pq_sufficient_but_not_necessary_condition_l162_162248

theorem pq_sufficient_but_not_necessary_condition (p q : Prop) (hpq : p ∧ q) :
  ¬¬p = p :=
by
  sorry

end pq_sufficient_but_not_necessary_condition_l162_162248


namespace intersection_of_A_and_B_l162_162407

def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1, 3, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {4} :=
by
  sorry

end intersection_of_A_and_B_l162_162407


namespace ratio_of_areas_of_concentric_circles_l162_162766

theorem ratio_of_areas_of_concentric_circles (C1 C2 : ℝ) (h1 : (60 / 360) * C1 = (45 / 360) * C2) :
  (C1 / C2) ^ 2 = (9 / 16) := by
  sorry

end ratio_of_areas_of_concentric_circles_l162_162766


namespace alex_charge_per_trip_l162_162516

theorem alex_charge_per_trip (x : ℝ)
  (savings_needed : ℝ) (n_trips : ℝ) (worth_groceries : ℝ) (charge_per_grocery_percent : ℝ) :
  savings_needed = 100 → 
  n_trips = 40 →
  worth_groceries = 800 →
  charge_per_grocery_percent = 0.05 →
  n_trips * x + charge_per_grocery_percent * worth_groceries = savings_needed →
  x = 1.5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end alex_charge_per_trip_l162_162516


namespace cistern_length_l162_162032

def cistern_conditions (L : ℝ) : Prop := 
  let width := 4
  let depth := 1.25
  let wet_surface_area := 42.5
  (L * width) + (2 * (L * depth)) + (2 * (width * depth)) = wet_surface_area

theorem cistern_length : 
  ∃ L : ℝ, cistern_conditions L ∧ L = 5 := sorry

end cistern_length_l162_162032


namespace correct_statement_l162_162927

-- Defining the conditions
def freq_eq_prob : Prop :=
  ∀ (f p : ℝ), f = p

def freq_objective : Prop :=
  ∀ (f : ℝ) (n : ℕ), f = f

def freq_stabilizes : Prop :=
  ∀ (p : ℝ), ∃ (f : ℝ) (n : ℕ), f = p

def prob_random : Prop :=
  ∀ (p : ℝ), p = p

-- The statement we need to prove
theorem correct_statement :
  ¬freq_eq_prob ∧ ¬freq_objective ∧ freq_stabilizes ∧ ¬prob_random :=
by
  sorry

end correct_statement_l162_162927


namespace mary_has_34_lambs_l162_162733

def mary_lambs (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) (traded_lambs : ℕ) (found_lambs : ℕ): ℕ :=
  initial_lambs + (lambs_with_babies * babies_per_lamb) - traded_lambs + found_lambs

theorem mary_has_34_lambs :
  mary_lambs 12 4 3 5 15 = 34 :=
by
  -- This line is in place of the actual proof.
  sorry

end mary_has_34_lambs_l162_162733


namespace xinjiang_arable_land_increase_reason_l162_162389

theorem xinjiang_arable_land_increase_reason
  (global_climate_warm: Prop)
  (annual_rainfall_increase: Prop)
  (reserve_arable_land_development: Prop)
  (national_land_policies_adjustment: Prop)
  (arable_land_increased: Prop) :
  (arable_land_increased → reserve_arable_land_development) :=
sorry

end xinjiang_arable_land_increase_reason_l162_162389


namespace Lewis_more_items_than_Samantha_l162_162982

def Tanya_items : ℕ := 4
def Samantha_items : ℕ := 4 * Tanya_items
def Lewis_items : ℕ := 20

theorem Lewis_more_items_than_Samantha : (Lewis_items - Samantha_items) = 4 := by
  sorry

end Lewis_more_items_than_Samantha_l162_162982


namespace max_sum_when_product_is_399_l162_162278

theorem max_sum_when_product_is_399 :
  ∃ (X Y Z : ℕ), X * Y * Z = 399 ∧ X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X ∧ X + Y + Z = 29 :=
by
  sorry

end max_sum_when_product_is_399_l162_162278


namespace angle_B_eq_18_l162_162259

theorem angle_B_eq_18 
  (A B : ℝ) 
  (h1 : A = 4 * B) 
  (h2 : 90 - B = 4 * (90 - A)) : 
  B = 18 :=
by
  sorry

end angle_B_eq_18_l162_162259


namespace scientific_notation_of_74850000_l162_162909

theorem scientific_notation_of_74850000 : 74850000 = 7.485 * 10^7 :=
  by
  sorry

end scientific_notation_of_74850000_l162_162909


namespace total_games_won_l162_162576

theorem total_games_won 
  (bulls_games : ℕ) (heat_games : ℕ) (knicks_games : ℕ)
  (bulls_condition : bulls_games = 70)
  (heat_condition : heat_games = bulls_games + 5)
  (knicks_condition : knicks_games = 2 * heat_games) :
  bulls_games + heat_games + knicks_games = 295 :=
by
  sorry

end total_games_won_l162_162576


namespace sum_of_tens_and_units_of_product_is_zero_l162_162619

-- Define the repeating patterns used to create the 999-digit numbers
def pattern1 : ℕ := 400
def pattern2 : ℕ := 606

-- Function to construct a 999-digit number by repeating a 3-digit pattern 333 times
def repeat_pattern (pat : ℕ) (times : ℕ) : ℕ := pat * (10 ^ (3 * times - 3))

-- Define the two 999-digit numbers
def num1 : ℕ := repeat_pattern pattern1 333
def num2 : ℕ := repeat_pattern pattern2 333

-- Function to compute the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Function to compute the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define the product of the two numbers
def product : ℕ := num1 * num2

-- Function to compute the sum of the tens and units digits of a number
def sum_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

-- The statement to be proven
theorem sum_of_tens_and_units_of_product_is_zero :
  sum_digits product = 0 := 
sorry -- Proof steps are omitted

end sum_of_tens_and_units_of_product_is_zero_l162_162619


namespace minimize_t_l162_162247

variable (Q : ℝ) (Q_1 Q_2 Q_3 Q_4 Q_5 Q_6 Q_7 Q_8 Q_9 : ℝ)

-- Definition of the sum of undirected lengths
def t (Q : ℝ) := 
  abs (Q - Q_1) + abs (Q - Q_2) + abs (Q - Q_3) + 
  abs (Q - Q_4) + abs (Q - Q_5) + abs (Q - Q_6) + 
  abs (Q - Q_7) + abs (Q - Q_8) + abs (Q - Q_9)

-- Statement that t is minimized when Q = Q_5
theorem minimize_t : ∀ Q : ℝ, t Q ≥ t Q_5 := 
sorry

end minimize_t_l162_162247


namespace length_of_each_train_l162_162150

noncomputable def length_of_train : ℝ := 
  let speed_fast := 46 -- in km/hr
  let speed_slow := 36 -- in km/hr
  let relative_speed := speed_fast - speed_slow -- 10 km/hr
  let relative_speed_km_per_sec := relative_speed / 3600.0 -- converting to km/sec
  let time_sec := 18.0 -- time in seconds
  let distance_km := relative_speed_km_per_sec * time_sec -- calculates distance in km
  distance_km * 1000.0 -- converts to meters

theorem length_of_each_train : length_of_train = 50 :=
  by
    sorry

end length_of_each_train_l162_162150


namespace focus_of_parabola_eq_l162_162225

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := -5 * x^2 + 10 * x - 2

-- Statement of the theorem to find the focus of the given parabola
theorem focus_of_parabola_eq (x : ℝ) : 
  let vertex_x := 1
  let vertex_y := 3
  let a := -5
  ∃ focus_x focus_y, 
    focus_x = vertex_x ∧ 
    focus_y = vertex_y - (1 / (4 * a)) ∧
    focus_x = 1 ∧
    focus_y = 59 / 20 := 
  sorry

end focus_of_parabola_eq_l162_162225


namespace net_change_of_Toronto_Stock_Exchange_l162_162531

theorem net_change_of_Toronto_Stock_Exchange :
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  (monday + tuesday + wednesday + thursday + friday) = -119 :=
by
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  have h : (monday + tuesday + wednesday + thursday + friday) = -119 := sorry
  exact h

end net_change_of_Toronto_Stock_Exchange_l162_162531


namespace part_one_part_two_l162_162080

noncomputable def f (a x : ℝ) : ℝ :=
  a * x - (1 : ℝ) / x - (a + 1) * Real.log x

theorem part_one (a : ℝ) (h : a = 0) : 
  ∃ x : ℝ, x > 0 ∧ f a x ≤ -1 := 
begin
  sorry
end

theorem part_two : 
  ∀ a : ℝ, (∃ x : ℝ, f a x = 0 ∧ ∀ y : ℝ, y > 0 → f a y ≠ 0 \iff 0 < a) :=
begin
  sorry
end

end part_one_part_two_l162_162080


namespace product_of_second_and_fourth_term_l162_162330

theorem product_of_second_and_fourth_term (a : ℕ → ℤ) (d : ℤ) (h₁ : a 10 = 25) (h₂ : d = 3)
  (h₃ : ∀ n, a n = a 1 + (n - 1) * d) : a 2 * a 4 = 7 :=
by
  -- Assuming necessary conditions are defined
  sorry

end product_of_second_and_fourth_term_l162_162330


namespace intersection_of_A_and_B_l162_162272

def set_A : Set ℝ := {x | -x^2 - x + 6 > 0}
def set_B : Set ℝ := {x | 5 / (x - 3) ≤ -1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | -2 ≤ x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l162_162272


namespace derek_joe_ratio_l162_162726

theorem derek_joe_ratio (D J T : ℝ) (h0 : J = 23) (h1 : T = 30) (h2 : T = (1/3 : ℝ) * D + 16) :
  D / J = 42 / 23 :=
by
  sorry

end derek_joe_ratio_l162_162726


namespace boat_distance_downstream_l162_162906

theorem boat_distance_downstream
  (speed_boat : ℕ)
  (speed_stream : ℕ)
  (time_downstream : ℕ)
  (h1 : speed_boat = 22)
  (h2 : speed_stream = 5)
  (h3 : time_downstream = 8) :
  speed_boat + speed_stream * time_downstream = 216 :=
by
  sorry

end boat_distance_downstream_l162_162906


namespace real_part_of_one_over_one_minus_z_l162_162591

open Complex

noncomputable def real_part_fraction {z : ℂ} (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) : ℝ :=
  re (1 / (1 - z))

theorem real_part_of_one_over_one_minus_z (z : ℂ) (hz1 : norm z = 1) (hz2 : ¬(z.im = 0)) :
  real_part_fraction hz1 hz2 = 1 / 2 :=
by
  sorry

end real_part_of_one_over_one_minus_z_l162_162591


namespace gcd_in_base3_l162_162553

def gcd_2134_1455_is_97 : ℕ :=
  gcd 2134 1455

def base3 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (n : ℕ) : List ℕ :=
      if n = 0 then [] else aux (n / 3) ++ [n % 3]
    aux n

theorem gcd_in_base3 :
  gcd_2134_1455_is_97 = 97 ∧ base3 97 = [1, 0, 1, 2, 1] :=
by
  sorry

end gcd_in_base3_l162_162553


namespace calc_value_l162_162522

theorem calc_value (a b x : ℤ) (h₁ : a = 153) (h₂ : b = 147) (h₃ : x = 900) : x^2 / (a^2 - b^2) = 450 :=
by
  rw [h₁, h₂, h₃]
  -- Proof follows from the calculation in the provided steps
  sorry

end calc_value_l162_162522


namespace probability_X_greater_than_4_l162_162401

noncomputable def normalDist (μ σ : ℝ) : ProbabilityDistribution := sorry

-- Assume X follows a normal distribution with mean 3 and standard deviation 1
axiom X_follows_normal : ∀ X : ℝ, has_pdf (normalDist 3 1) X

-- Given condition: P(2 ≤ X ≤ 4) = 0.6826
axiom probability_within_one_std_dev :
  ∀ (X : ℝ), P 2 ≤ X ∧ X ≤ 4 = 0.6826

-- Theorem statement: P(X > 4) = 0.1587
theorem probability_X_greater_than_4 :
  ∀ (X : ℝ), P X > 4 = 0.1587 := sorry

end probability_X_greater_than_4_l162_162401


namespace police_emergency_number_prime_divisor_l162_162806

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 1000 * k + 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_prime_divisor_l162_162806


namespace C_share_correct_l162_162340

noncomputable def C_share (B_invest: ℝ) (total_profit: ℝ) : ℝ :=
  let A_invest := 3 * B_invest
  let C_invest := (3 * B_invest) * (3/2)
  let total_invest := (3 * B_invest + B_invest + C_invest)
  (C_invest / total_invest) * total_profit

theorem C_share_correct (B_invest total_profit: ℝ) 
  (hA : ∀ x: ℝ, A_invest = 3 * x)
  (hC : ∀ x: ℝ, C_invest = (3 * x) * (3/2)) :
  C_share B_invest 12375 = 6551.47 :=
by
  sorry

end C_share_correct_l162_162340


namespace choose_president_vp_and_committee_l162_162281

theorem choose_president_vp_and_committee :
  ∃ (n : ℕ) (k : ℕ), n = 10 ∧ k = 3 ∧ 
  let ways_to_choose_president := 10 in
  let ways_to_choose_vp := 9 in
  let ways_to_choose_committee := (Nat.choose 8 3) in
  ways_to_choose_president * ways_to_choose_vp * ways_to_choose_committee = 5040 :=
begin
  use [10, 3],
  simp [Nat.choose],
  sorry
end

end choose_president_vp_and_committee_l162_162281


namespace fraction_oil_is_correct_l162_162737

noncomputable def fraction_oil_third_bottle (C : ℚ) (oil1 : ℚ) (oil2 : ℚ) (water1 : ℚ) (water2 : ℚ) := 
  (oil1 + oil2) / (oil1 + oil2 + water1 + water2)

theorem fraction_oil_is_correct (C : ℚ) (hC : C > 0) :
  let oil1 := C / 2
  let oil2 := C / 2
  let water1 := C / 2
  let water2 := 3 * C / 4
  fraction_oil_third_bottle C oil1 oil2 water1 water2 = 4 / 9 := by
  sorry

end fraction_oil_is_correct_l162_162737


namespace inequality_proof_l162_162850

variable (a b : ℝ)

theorem inequality_proof (h : a < b) : 1 - a > 1 - b :=
sorry

end inequality_proof_l162_162850


namespace min_distinct_sums_max_distinct_sums_l162_162546

theorem min_distinct_sums (n : ℕ) (h : 0 < n) : ∃ a b, (a + (n - 1) * b) = (n * (n + 1)) / 2 := sorry

theorem max_distinct_sums (n : ℕ) (h : 0 < n) : 
  ∃ m, m = 2^n - 1 := sorry

end min_distinct_sums_max_distinct_sums_l162_162546


namespace max_integer_value_l162_162569

theorem max_integer_value (x : ℝ) : 
  ∃ (n : ℤ), n = 15 ∧ ∀ x : ℝ, 
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ n :=
by
  sorry

end max_integer_value_l162_162569


namespace greatest_mult_of_4_lessthan_5000_l162_162874

theorem greatest_mult_of_4_lessthan_5000 :
  ∃ x : ℕ, (0 < x) ∧ (x % 4 = 0) ∧ (x^3 < 5000) ∧ (∀ y : ℕ, (0 < y) ∧ (y % 4 = 0) ∧ (y^3 < 5000) → y ≤ x) := 
sorry

end greatest_mult_of_4_lessthan_5000_l162_162874


namespace max_value_f_when_a_zero_range_a_for_single_zero_l162_162075

noncomputable def f (a x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * Real.log x

theorem max_value_f_when_a_zero :
  (∀ x > (0 : ℝ), f 0 x ≤ -1) ∧ (∃ x > (0 : ℝ), f 0 x = -1) :=
sorry

theorem range_a_for_single_zero :
  (∀ a : ℝ, (∃ x > (0 : ℝ), f a x = 0) ↔ (0 < a ∧ a < ∞)) :=
sorry

end max_value_f_when_a_zero_range_a_for_single_zero_l162_162075


namespace sum_digits_10_pow_100_minus_100_l162_162046

open Nat

/-- Define the condition: 10^100 - 100 as an expression. -/
def subtract_100_from_power_10 (n : ℕ) : ℕ :=
  10^n - 100

/-- Sum the digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- The goal is to prove the sum of the digits of 10^100 - 100 equals 882. -/
theorem sum_digits_10_pow_100_minus_100 :
  sum_of_digits (subtract_100_from_power_10 100) = 882 :=
by
  sorry

end sum_digits_10_pow_100_minus_100_l162_162046


namespace intersection_correct_l162_162583

noncomputable def set_M : Set ℝ := { x | x^2 + x - 6 ≤ 0 }
noncomputable def set_N : Set ℝ := { x | abs (2 * x + 1) > 3 }
noncomputable def set_intersection : Set ℝ := { x | (x ∈ set_M) ∧ (x ∈ set_N) }

theorem intersection_correct : 
  set_intersection = { x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 2) } := 
by 
  sorry

end intersection_correct_l162_162583


namespace max_modulus_l162_162694

open Complex

theorem max_modulus (z : ℂ) (h : abs z = 1) : ∃ M, M = 6 ∧ ∀ w, abs (z - w) ≤ M :=
by
  use 6
  sorry

end max_modulus_l162_162694


namespace simplify_fraction_l162_162205

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b)) :=
by
  sorry

end simplify_fraction_l162_162205


namespace ratio_of_women_working_in_retail_l162_162419

-- Define the population of Los Angeles
def population_LA : ℕ := 6000000

-- Define the proportion of women in Los Angeles
def half_population : ℕ := population_LA / 2

-- Define the number of women working in retail
def women_retail : ℕ := 1000000

-- Define the total number of women in Los Angeles
def total_women : ℕ := half_population

-- The statement to be proven:
theorem ratio_of_women_working_in_retail :
  (women_retail / total_women : ℚ) = 1 / 3 :=
by {
  -- The proof goes here
  sorry
}

end ratio_of_women_working_in_retail_l162_162419


namespace problem_1_solution_set_problem_2_range_of_a_l162_162404

-- Define the function f(x)
def f (x a : ℝ) := |2 * x - a| + |x - 1|

-- Problem 1: Solution set of the inequality f(x) ≥ 2 when a = 3
theorem problem_1_solution_set :
  { x : ℝ | f x 3 ≥ 2 } = { x : ℝ | x ≤ 2/3 ∨ x ≥ 2 } :=
sorry

-- Problem 2: Range of a such that f(x) ≥ 5 - x for all x ∈ ℝ
theorem problem_2_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 5 - x) ↔ a ≥ 6 :=
sorry

end problem_1_solution_set_problem_2_range_of_a_l162_162404


namespace find_number_l162_162830

theorem find_number:
  ∃ x : ℝ, x + 1.35 + 0.123 = 1.794 ∧ x = 0.321 :=
by
  sorry

end find_number_l162_162830


namespace q1_correct_q2_correct_l162_162677

-- Defining the necessary operations
def q1_lhs := 8 / (-2) - (-4) * (-3)
def q2_lhs := (-2) ^ 3 / 4 * (5 - (-3) ^ 2)

-- Theorem statements to prove that they are equal to 8
theorem q1_correct : q1_lhs = 8 := sorry
theorem q2_correct : q2_lhs = 8 := sorry

end q1_correct_q2_correct_l162_162677


namespace thirteenth_term_geometric_sequence_l162_162756

theorem thirteenth_term_geometric_sequence 
  (a : ℕ → ℕ) 
  (r : ℝ)
  (h₁ : a 7 = 7) 
  (h₂ : a 10 = 21)
  (h₃ : ∀ (n : ℕ), a (n + 1) = a n * r) : 
  a 13 = 63 := 
by
  -- proof needed
  sorry

end thirteenth_term_geometric_sequence_l162_162756


namespace floor_sq_minus_sq_floor_l162_162938

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l162_162938


namespace jackson_points_l162_162420

theorem jackson_points (team_total_points : ℕ) (other_players_count : ℕ) (other_players_avg_score : ℕ) 
  (total_points_by_team : team_total_points = 72) 
  (total_points_by_others : other_players_count = 7) 
  (avg_points_by_others : other_players_avg_score = 6) :
  ∃ points_by_jackson : ℕ, points_by_jackson = 30 :=
by
  sorry

end jackson_points_l162_162420


namespace complementary_union_correct_l162_162095

open Set

variable (U A B CU_B : Set ℕ)
variable (U_def : U = {0, 1, 2, 3, 4})
variable (A_def : A = {1, 2, 3})
variable (B_def : B = {2, 4})
variable (CU_B_def : CU_B = (U \ A) ∪ B)

theorem complementary_union_correct : CU_B = {0, 2, 4} := by
  rw [CU_B_def, U_def, A_def, B_def]
  dsimp
  sorry

end complementary_union_correct_l162_162095


namespace minimum_people_in_troupe_l162_162796

-- Let n be the number of people in the troupe.
variable (n : ℕ)

-- Conditions: n must be divisible by 8, 10, and 12.
def is_divisible_by (m k : ℕ) := m % k = 0
def divides_all (n : ℕ) := is_divisible_by n 8 ∧ is_divisible_by n 10 ∧ is_divisible_by n 12

-- The minimum number of people in the troupe that can form groups of 8, 10, or 12 with none left over.
theorem minimum_people_in_troupe (n : ℕ) : divides_all n → n = 120 :=
by
  sorry

end minimum_people_in_troupe_l162_162796


namespace sequence_expression_l162_162980

theorem sequence_expression (s a : ℕ → ℝ) (h : ∀ n : ℕ, 1 ≤ n → s n = (3 / 2 * (a n - 1))) :
  ∀ n : ℕ, 1 ≤ n → a n = 3^n :=
by
  sorry

end sequence_expression_l162_162980


namespace find_k_l162_162503

theorem find_k (α β k : ℝ) (h₁ : α^2 - α + k - 1 = 0) (h₂ : β^2 - β + k - 1 = 0) (h₃ : α^2 - 2*α - β = 4) :
  k = -4 :=
sorry

end find_k_l162_162503


namespace quadratic_eq_integer_roots_iff_l162_162586

theorem quadratic_eq_integer_roots_iff (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x * y = n ∧ x + y = 4) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end quadratic_eq_integer_roots_iff_l162_162586


namespace rational_add_positive_square_l162_162266

theorem rational_add_positive_square (a : ℚ) : a^2 + 1 > 0 := by
  sorry

end rational_add_positive_square_l162_162266


namespace polynomial_sum_l162_162589

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end polynomial_sum_l162_162589


namespace minimum_value_l162_162344

def f (x : ℝ) : ℝ := |x - 4| + |x + 7| + |x - 5|

theorem minimum_value : ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x = 4 :=
by
  -- Sorry is used here to skip the proof
  sorry

end minimum_value_l162_162344


namespace find_speeds_and_circumference_l162_162148

variable (Va Vb : ℝ)
variable (l : ℝ)

axiom smaller_arc_condition : 10 * (Va + Vb) = 150
axiom larger_arc_condition : 14 * (Va + Vb) = l - 150
axiom travel_condition : l / Va = 90 / Vb 

theorem find_speeds_and_circumference :
  Va = 12 ∧ Vb = 3 ∧ l = 360 := by
  sorry

end find_speeds_and_circumference_l162_162148


namespace a_2016_value_l162_162398

theorem a_2016_value (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6) 
  (rec : ∀ n, a (n + 2) = a (n + 1) - a n) : a 2016 = -3 :=
sorry

end a_2016_value_l162_162398


namespace circle_standard_equation_l162_162541

theorem circle_standard_equation (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y - 1)^2) ∧
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y + 2)^2) →
  (∃ x y : ℝ, (x - 2) ^ 2 + (y - 1) ^ 2 = 2) :=
sorry

end circle_standard_equation_l162_162541


namespace max_value_expression_l162_162004

theorem max_value_expression (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 * (a + b))) +
   Real.sqrt (Real.sqrt (b^2 * (b + c))) +
   Real.sqrt (Real.sqrt (c^2 * (c + d))) +
   Real.sqrt (Real.sqrt (d^2 * (d + a)))) ≤ 4 * Real.sqrt (Real.sqrt 2) := by
  sorry

end max_value_expression_l162_162004


namespace max_value_f_at_a0_l162_162073

def f (x : ℝ) : ℝ := - (1 / x) - Real.log x

theorem max_value_f_at_a0 : ∀ x > 0 , ∃ y : ℝ, y = f x → y ≤ f 1 := 
by
  sorry

end max_value_f_at_a0_l162_162073


namespace minimum_set_size_l162_162781

theorem minimum_set_size (n : ℕ) :
  (2 * n + 1) ≥ 11 :=
begin
  have h1 : 7 * (2 * n + 1) ≤ 15 * n + 2,
  sorry,
  have h2 : 14 * n + 7 ≤ 15 * n + 2,
  sorry,
  have h3 : n ≥ 5,
  sorry,
  show 2 * n + 1 ≥ 11,
  from calc
    2 * n + 1 = 2 * 5 + 1 : by linarith
          ... ≥ 11 : by linarith,
end

end minimum_set_size_l162_162781


namespace truck_distance_l162_162357

theorem truck_distance :
  let a1 := 8
  let d := 9
  let n := 40
  let an := a1 + (n - 1) * d
  let S_n := n / 2 * (a1 + an)
  S_n = 7340 :=
by
  sorry

end truck_distance_l162_162357


namespace cube_sum_decomposition_l162_162461

theorem cube_sum_decomposition : 
  (∃ (a b c d e : ℤ), (1000 * x^3 + 27) = (a * x + b) * (c * x^2 + d * x + e) ∧ (a + b + c + d + e = 92)) :=
by
  sorry

end cube_sum_decomposition_l162_162461


namespace lucky_sum_mod_1000_l162_162916

def is_lucky (n : ℕ) : Prop := ∀ d ∈ n.digits 10, d = 7

def first_twenty_lucky_numbers : List ℕ :=
  [7, 77] ++ List.replicate 18 777

theorem lucky_sum_mod_1000 :
  (first_twenty_lucky_numbers.sum % 1000) = 70 := 
sorry

end lucky_sum_mod_1000_l162_162916


namespace range_of_x_inequality_l162_162703

theorem range_of_x_inequality (a : ℝ) (x : ℝ)
  (h : -1 ≤ a ∧ a ≤ 1) : 
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end range_of_x_inequality_l162_162703


namespace david_older_than_scott_l162_162793

-- Define the ages of Richard, David, and Scott
variables (R D S : ℕ)

-- Given conditions
def richard_age_eq : Prop := R = D + 6
def richard_twice_scott : Prop := R + 8 = 2 * (S + 8)
def david_current_age : Prop := D = 14

-- Prove the statement
theorem david_older_than_scott (h1 : richard_age_eq R D) (h2 : richard_twice_scott R S) (h3 : david_current_age D) :
  D - S = 8 :=
  sorry

end david_older_than_scott_l162_162793


namespace am_gm_problem_l162_162469

theorem am_gm_problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
by 
  sorry

end am_gm_problem_l162_162469


namespace max_distinct_numbers_example_l162_162985

def max_distinct_numbers (a b c d e : ℕ) : ℕ := sorry

theorem max_distinct_numbers_example
  (A B : ℕ) :
  max_distinct_numbers 100 200 400 A B = 64 := sorry

end max_distinct_numbers_example_l162_162985


namespace inclination_of_line_through_vertex_equilateral_triangle_l162_162664

noncomputable def inclination_angles (x : ℝ) : ℝ × ℝ :=
  let α := Real.arctan (Real.sqrt 3 / 5)
  in (α, 60 - α)

theorem inclination_of_line_through_vertex_equilateral_triangle (x : ℝ) (α β : ℝ)
  (h_divides_base : 2 * x = AD ∧ x = CD ∧ AC = AD + 2 * CD)
  (h_angles : α = Real.arctan (Real.sqrt 3 / 5) ∧ β = 60 - α) :
  inclination_angles x = (α, β) :=
by
  sorry

end inclination_of_line_through_vertex_equilateral_triangle_l162_162664


namespace smallest_five_digit_multiple_of_53_l162_162646

theorem smallest_five_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 53 = 0 ∧ n = 10017 :=
by
  sorry

end smallest_five_digit_multiple_of_53_l162_162646


namespace factorization_identity_l162_162384

theorem factorization_identity (a b : ℝ) : (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2 :=
by
  sorry

end factorization_identity_l162_162384


namespace company_employees_count_l162_162578

theorem company_employees_count :
  (females : ℕ) ->
  (advanced_degrees : ℕ) ->
  (college_degree_only_males : ℕ) ->
  (advanced_degrees_females : ℕ) ->
  (110 = females) ->
  (90 = advanced_degrees) ->
  (35 = college_degree_only_males) ->
  (55 = advanced_degrees_females) ->
  (females - advanced_degrees_females + college_degree_only_males + advanced_degrees = 180) :=
by
  intros females advanced_degrees college_degree_only_males advanced_degrees_females
  intro h_females h_advanced_degrees h_college_degree_only_males h_advanced_degrees_females
  sorry

end company_employees_count_l162_162578


namespace manager_monthly_salary_l162_162024

theorem manager_monthly_salary (average_salary_20 : ℝ) (new_average_salary_21 : ℝ) (m : ℝ) 
  (h1 : average_salary_20 = 1300) 
  (h2 : new_average_salary_21 = 1400) 
  (h3 : 20 * average_salary_20 + m = 21 * new_average_salary_21) : 
  m = 3400 := 
by 
  -- Proof is omitted
  sorry

end manager_monthly_salary_l162_162024


namespace tangent_line_at_point_l162_162313

theorem tangent_line_at_point :
  ∀ (x y : ℝ) (h : y = x^3 - 2 * x + 1),
    ∃ (m b : ℝ), (1, 0) = (x, y) → (m = 1) ∧ (b = -1) ∧ (∀ (z : ℝ), z = m * x + b) := sorry

end tangent_line_at_point_l162_162313


namespace pony_wait_time_l162_162145

-- Definitions of the conditions
def cycle_time_monster_A : ℕ := 2 + 1 -- hours (2 awake, 1 rest)
def cycle_time_monster_B : ℕ := 3 + 2 -- hours (3 awake, 2 rest)

-- The theorem to prove the correct answer
theorem pony_wait_time :
  Nat.lcm cycle_time_monster_A cycle_time_monster_B = 15 :=
by
  -- Skip the proof
  sorry

end pony_wait_time_l162_162145


namespace regression_shows_positive_correlation_l162_162169

-- Define the regression equations as constants
def reg_eq_A (x : ℝ) : ℝ := -2.1 * x + 1.8
def reg_eq_B (x : ℝ) : ℝ := 1.2 * x + 1.5
def reg_eq_C (x : ℝ) : ℝ := -0.5 * x + 2.1
def reg_eq_D (x : ℝ) : ℝ := -0.6 * x + 3

-- Define the condition for positive correlation
def positive_correlation (b : ℝ) : Prop := b > 0

-- The theorem statement to prove
theorem regression_shows_positive_correlation : 
  positive_correlation 1.2 := 
by
  sorry

end regression_shows_positive_correlation_l162_162169


namespace travel_time_l162_162745

def speed : ℝ := 60  -- Speed of the car in miles per hour
def distance : ℝ := 300  -- Distance to the campground in miles

theorem travel_time : distance / speed = 5 := by
  sorry

end travel_time_l162_162745


namespace lunks_needed_for_24_apples_l162_162262

-- Define the conditions as Lean definitions
def lunks_per_kunks := 7 / 4
def kunks_per_apples := 3 / 5
def apples_needed := 24

-- State the theorem
theorem lunks_needed_for_24_apples : 
  let k := (3 * apples_needed) / 5 in 
  let rounded_k := k.ceil in 
  let l := (7 * rounded_k) / 4 in 
  l.ceil = 27 :=
by 
  let k := (3 * apples_needed) / 5
  let rounded_k := k.ceil
  let l := (7 * rounded_k) / 4
  have h1 : k = (3 * apples_needed) / 5 := rfl
  have h2 : rounded_k = k.ceil := rfl
  have h3 : l = (7 * rounded_k) / 4 := rfl
  show l.ceil = 27, from sorry

end lunks_needed_for_24_apples_l162_162262


namespace train_speed_and_length_l162_162231

theorem train_speed_and_length (V l : ℝ) 
  (h1 : 7 * V = l) 
  (h2 : 25 * V = 378 + l) : 
  V = 21 ∧ l = 147 :=
by
  sorry

end train_speed_and_length_l162_162231


namespace floor_sqrt_23_squared_eq_16_l162_162220

theorem floor_sqrt_23_squared_eq_16 :
  (Int.floor (Real.sqrt 23))^2 = 16 :=
by
  have h1 : 4 < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < 5 := sorry
  have floor_sqrt_23 : Int.floor (Real.sqrt 23) = 4 := sorry
  rw [floor_sqrt_23]
  norm_num

end floor_sqrt_23_squared_eq_16_l162_162220


namespace sum_of_two_primes_l162_162628

theorem sum_of_two_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 93) : p * q = 178 := 
sorry

end sum_of_two_primes_l162_162628


namespace fruit_basket_cost_l162_162180

theorem fruit_basket_cost :
  let bananas_cost   := 4 * 1
  let apples_cost    := 3 * 2
  let strawberries_cost := (24 / 12) * 4
  let avocados_cost  := 2 * 3
  let grapes_cost    := 2 * 2
  bananas_cost + apples_cost + strawberries_cost + avocados_cost + grapes_cost = 28 := 
by
  let groceries_cost := bananas_cost + apples_cost + strawberries_cost + avocados_cost + grapes_cost
  exact sorry

end fruit_basket_cost_l162_162180


namespace traveling_cost_l162_162036

def area_road_length_parallel (length width : ℕ) := width * length

def area_road_breadth_parallel (length width : ℕ) := width * length

def area_intersection (width : ℕ) := width * width

def total_area_of_roads  (length breadth width : ℕ) : ℕ :=
  (area_road_length_parallel length width) + (area_road_breadth_parallel breadth width) - area_intersection width

def cost_of_traveling_roads (total_area_of_roads cost_per_sq_m : ℕ) := total_area_of_roads * cost_per_sq_m

theorem traveling_cost
  (length breadth width cost_per_sq_m : ℕ)
  (h_length : length = 80)
  (h_breadth : breadth = 50)
  (h_width : width = 10)
  (h_cost_per_sq_m : cost_per_sq_m = 3)
  : cost_of_traveling_roads (total_area_of_roads length breadth width) cost_per_sq_m = 3600 :=
by
  sorry

end traveling_cost_l162_162036


namespace negative_integer_solution_l162_162002

theorem negative_integer_solution (N : ℤ) (h : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end negative_integer_solution_l162_162002


namespace floor_diff_l162_162941

theorem floor_diff (x : ℝ) (h : x = 13.2) : 
  (Int.floor (x^2) - (Int.floor x) * (Int.floor x) = 5) := by
  sorry

end floor_diff_l162_162941


namespace total_number_of_rulers_l162_162331

-- Given conditions
def initial_rulers : ℕ := 11
def rulers_added_by_tim : ℕ := 14

-- Given question and desired outcome
def total_rulers (initial_rulers rulers_added_by_tim : ℕ) : ℕ :=
  initial_rulers + rulers_added_by_tim

-- The proof problem statement
theorem total_number_of_rulers : total_rulers 11 14 = 25 := by
  sorry

end total_number_of_rulers_l162_162331


namespace find_balcony_seat_cost_l162_162671

-- Definitions based on conditions
variable (O B : ℕ) -- Number of orchestra tickets and cost of balcony ticket
def orchestra_ticket_cost : ℕ := 12
def total_tickets : ℕ := 370
def total_cost : ℕ := 3320
def tickets_difference : ℕ := 190

-- Lean statement to prove the cost of a balcony seat
theorem find_balcony_seat_cost :
  (2 * O + tickets_difference = total_tickets) ∧
  (orchestra_ticket_cost * O + B * (O + tickets_difference) = total_cost) →
  B = 8 :=
by
  sorry

end find_balcony_seat_cost_l162_162671


namespace cost_of_45_roses_l162_162520

theorem cost_of_45_roses (cost_15_roses : ℕ → ℝ) 
  (h1 : cost_15_roses 15 = 25) 
  (h2 : ∀ (n m : ℕ), cost_15_roses n / n = cost_15_roses m / m )
  (h3 : ∀ (n : ℕ), n > 30 → cost_15_roses n = (1 - 0.10) * cost_15_roses n) :
  cost_15_roses 45 = 67.5 :=
by
  sorry

end cost_of_45_roses_l162_162520


namespace buratino_cafe_workdays_l162_162996

-- Define the conditions as given in the problem statement
def days_in_april (d : Nat) : Prop := d >= 1 ∧ d <= 30
def is_monday (d : Nat) : Prop := d = 1 ∨ d = 8 ∨ d = 15 ∨ d = 22 ∨ d = 29

-- Define the period April 1 to April 13
def period_1_13 (d : Nat) : Prop := d >= 1 ∧ d <= 13

-- Define the statements made by Kolya
def kolya_statement_1 : Prop := ∀ d : Nat, days_in_april d → (d >= 1 ∧ d <= 20) → ¬is_monday d → ∃ n : Nat, n = 18
def kolya_statement_2 : Prop := ∀ d : Nat, days_in_april d → (d >= 10 ∧ d <= 30) → ¬is_monday d → ∃ n : Nat, n = 18

-- Define the condition stating Kolya made a mistake once
def kolya_made_mistake_once : Prop := kolya_statement_1 ∨ kolya_statement_2

-- The proof problem: Prove the number of working days from April 1 to April 13 is 11
theorem buratino_cafe_workdays : period_1_13 (d) → (¬is_monday d → (∃ n : Nat, n = 11)) := sorry

end buratino_cafe_workdays_l162_162996


namespace area_ratio_l162_162321

-- Define the conditions: perimeters relation
def condition (a b : ℝ) := 4 * a = 16 * b

-- Define the theorem to be proved
theorem area_ratio (a b : ℝ) (h : condition a b) : (a * a) = 16 * (b * b) :=
sorry

end area_ratio_l162_162321


namespace sphere_radius_equal_l162_162888

theorem sphere_radius_equal (r : ℝ) 
  (hvol : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_equal_l162_162888


namespace train_length_l162_162193

theorem train_length (bridge_length time_seconds speed_kmh : ℝ) (S : speed_kmh = 64) (T : time_seconds = 45) (B : bridge_length = 300) : 
  ∃ (train_length : ℝ), train_length = 500 := 
by
  -- Add your proof here 
  sorry

end train_length_l162_162193


namespace sin_480_eq_sqrt3_div_2_l162_162471

theorem sin_480_eq_sqrt3_div_2 : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_480_eq_sqrt3_div_2_l162_162471


namespace seating_arrangement_six_people_l162_162812

theorem seating_arrangement_six_people : 
  ∃ (n : ℕ), n = 216 ∧ 
  (∀ (a b c d e f : ℕ),
    -- Alice, Bob, and Carla indexing
    1 ≤ a ∧ a ≤ 6 ∧ 
    1 ≤ b ∧ b ≤ 6 ∧ 
    1 ≤ c ∧ c ≤ 6 ∧ 
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    (a ≠ b + 1 ∧ a ≠ b - 1) ∧
    (a ≠ c + 1 ∧ a ≠ c - 1) ∧
    
    -- Derek, Eric, and Fiona indexing
    1 ≤ d ∧ d ≤ 6 ∧ 
    1 ≤ e ∧ e ≤ 6 ∧ 
    1 ≤ f ∧ f ≤ 6 ∧ 
    d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    (d ≠ e + 1 ∧ d ≠ e - 1) ∧
    (d ≠ f + 1 ∧ d ≠ f - 1)) -> 
  n = 216 := 
sorry

end seating_arrangement_six_people_l162_162812


namespace probability_ab_not_selected_together_l162_162515

open Set

noncomputable def total_combinations : ℕ := (Finset.univ.subset_card_eq 4).choose 2
noncomputable def ab_combinations : ℕ := (Finset.univ.subset_card_eq 2).choose 2

theorem probability_ab_not_selected_together :
  (total_combinations : ℚ) = 6 → 
  (ab_combinations : ℚ) = 1 →
  (p : ℚ) = 1 - (ab_combinations / total_combinations) →
  p = 5 / 6 := by sorry

end probability_ab_not_selected_together_l162_162515


namespace part1_part2_l162_162067

-- Prove Part (1)
theorem part1 (M : ℕ) (N : ℕ) (h : M = 9) (h2 : N - 4 + 6 = M) : N = 7 :=
sorry

-- Prove Part (2)
theorem part2 (M : ℕ) (h : M = 9) : M - 4 = 5 ∨ M + 4 = 13 :=
sorry

end part1_part2_l162_162067


namespace smaller_number_l162_162007

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 :=
by
  sorry

end smaller_number_l162_162007


namespace total_material_ordered_l162_162510

theorem total_material_ordered (c b s : ℝ) (hc : c = 0.17) (hb : b = 0.17) (hs : s = 0.5) :
  c + b + s = 0.84 :=
by sorry

end total_material_ordered_l162_162510


namespace tetrahedron_volume_l162_162523

variables {P Q R S : ℝ × ℝ × ℝ}

-- Given conditions
def PQ : ℝ := 3
def PR : ℝ := 4
def PS : ℝ := 5
def QR : ℝ := Real.sqrt 17
def QS : ℝ := 2 * Real.sqrt 10
def RS : ℝ := Real.sqrt 29

-- Volume calculation
theorem tetrahedron_volume :
  ∀ P Q R S : ℝ × ℝ × ℝ, 
  dist P Q = PQ →
  dist P R = PR →
  dist P S = PS →
  dist Q R = QR →
  dist Q S = QS →
  dist R S = RS →
  volume_tetrahedron P Q R S = 6 :=
by
  intros,
  -- Proof needs to be provided.
  sorry

noncomputable def volume_tetrahedron (P Q R S : ℝ × ℝ × ℝ) : ℝ := 
  (1 / 6) * abs ((vector_triple_product (to_vector P R) (to_vector P Q) (to_vector P S)).det)

def to_vector (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

def vector_triple_product (u v w : ℝ × ℝ × ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![u.1 * (v.2 * w.3 - v.3 * w.2) - u.2 * (v.1 * w.3 - v.3 * w.1) + u.3 * (v.1 * w.2 - v.2 * w.1)]]

end tetrahedron_volume_l162_162523


namespace eval_fraction_expr_l162_162826

theorem eval_fraction_expr :
  (2 ^ 2010 * 3 ^ 2012) / (6 ^ 2011) = 3 / 2 := 
sorry

end eval_fraction_expr_l162_162826


namespace find_a1_l162_162250

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with a common ratio q > 0
axiom geom_seq : (∀ n, a (n + 1) = a n * q)

-- Given conditions of the problem
def condition1 : q > 0 := sorry
def condition2 : a 5 * a 7 = 4 * (a 4) ^ 2 := sorry
def condition3 : a 2 = 1 := sorry

-- Prove that a_1 = sqrt 2 / 2
theorem find_a1 : a 1 = (Real.sqrt 2) / 2 := sorry

end find_a1_l162_162250


namespace area_of_largest_medallion_is_314_l162_162763

noncomputable def largest_medallion_area_in_square (side: ℝ) (π: ℝ) : ℝ :=
  let diameter := side
  let radius := diameter / 2
  let area := π * radius^2
  area

theorem area_of_largest_medallion_is_314 :
  largest_medallion_area_in_square 20 3.14 = 314 := 
  sorry

end area_of_largest_medallion_is_314_l162_162763


namespace remainder_8_digit_non_decreasing_integers_mod_1000_l162_162979

noncomputable def M : ℕ :=
  Nat.choose 17 8

theorem remainder_8_digit_non_decreasing_integers_mod_1000 :
  M % 1000 = 310 :=
by
  sorry

end remainder_8_digit_non_decreasing_integers_mod_1000_l162_162979


namespace possible_values_of_expression_l162_162053

theorem possible_values_of_expression (x : ℝ) (h : 3 ≤ x ∧ x ≤ 4) : 
  40 ≤ x^2 + 7 * x + 10 ∧ x^2 + 7 * x + 10 ≤ 54 := 
sorry

end possible_values_of_expression_l162_162053


namespace inequality_solution_l162_162872

theorem inequality_solution (a : ℝ) (h : a > 0) :
  (if a = 2 then {x : ℝ | false}
   else if 0 < a ∧ a < 2 then {x : ℝ | 1 < x ∧ x ≤ 2 / a}
   else if a > 2 then {x : ℝ | 2 / a ≤ x ∧ x < 1}
   else ∅) =
    {x : ℝ | (a + 2) * x - 4 ≤ 2 * (x - 1)} :=
by
  sorry

end inequality_solution_l162_162872


namespace slower_train_speed_l162_162477

theorem slower_train_speed (v : ℝ) (faster_train_speed : ℝ) (time_pass : ℝ) (train_length : ℝ) :
  (faster_train_speed = 46) →
  (time_pass = 36) →
  (train_length = 50) →
  (v = 36) :=
by
  intro h1 h2 h3
  -- Formal proof goes here
  sorry

end slower_train_speed_l162_162477


namespace trig_relationship_l162_162243

theorem trig_relationship : 
  let a := Real.sin (145 * Real.pi / 180)
  let b := Real.cos (52 * Real.pi / 180)
  let c := Real.tan (47 * Real.pi / 180)
  a < b ∧ b < c :=
by 
  sorry

end trig_relationship_l162_162243


namespace police_emergency_number_has_prime_divisor_gt_7_l162_162800

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := sorry

end police_emergency_number_has_prime_divisor_gt_7_l162_162800


namespace ellipse_problem_l162_162862

-- Definitions of conditions from the problem
def F1 := (0, 0)
def F2 := (6, 0)
def ellipse_equation (x y h k a b : ℝ) := ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

-- The main statement to be proved
theorem ellipse_problem :
  let h := 3
  let k := 0
  let a := 5
  let c := 3
  let b := Real.sqrt (a^2 - c^2)
  h + k + a + b = 12 :=
by
  -- Proof would go here
  sorry

end ellipse_problem_l162_162862


namespace pesticide_residue_comparison_l162_162345

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem pesticide_residue_comparison (a : ℝ) (ha : a > 0) :
  (f a = (1 / (1 + a^2))) ∧ 
  (if a = 2 * Real.sqrt 2 then f a = 16 / (4 + a^2)^2 else 
   if a > 2 * Real.sqrt 2 then f a > 16 / (4 + a^2)^2 else 
   f a < 16 / (4 + a^2)^2) ∧
  (f 0 = 1) ∧ 
  (f 1 = 1 / 2) := sorry

end pesticide_residue_comparison_l162_162345


namespace prove_smallest_number_l162_162897

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

lemma smallest_number_to_add (n : ℕ) (k : ℕ) (h: sum_of_digits n % k = r) : n % k = r →
  n % k = r → (k - r) = 7 :=
by
  sorry

theorem prove_smallest_number (n : ℕ) (k : ℕ) (r : ℕ) :
  (27452 % 9 = r) ∧ (9 - r = 7) :=
by
  sorry

end prove_smallest_number_l162_162897


namespace intersection_A_B_l162_162094

def set_A : Set ℝ := {x | x > 0}
def set_B : Set ℝ := {x | x < 4}

theorem intersection_A_B :
  set_A ∩ set_B = {x | 0 < x ∧ x < 4} := sorry

end intersection_A_B_l162_162094


namespace cost_of_meatballs_is_five_l162_162137

-- Define the conditions
def cost_of_pasta : ℕ := 1
def cost_of_sauce : ℕ := 2
def total_cost_of_meal (servings : ℕ) (cost_per_serving : ℕ) : ℕ := servings * cost_per_serving

-- Define the cost of meatballs calculation
def cost_of_meatballs (total_cost pasta_cost sauce_cost : ℕ) : ℕ :=
  total_cost - pasta_cost - sauce_cost

-- State the theorem we want to prove
theorem cost_of_meatballs_is_five :
  cost_of_meatballs (total_cost_of_meal 8 1) cost_of_pasta cost_of_sauce = 5 :=
by
  -- This part will include the proof steps
  sorry

end cost_of_meatballs_is_five_l162_162137


namespace fraction_subtraction_identity_l162_162568

theorem fraction_subtraction_identity (x y : ℕ) (hx : x = 3) (hy : y = 4) : (1 / (x : ℚ) - 1 / (y : ℚ) = 1 / 12) :=
by
  sorry

end fraction_subtraction_identity_l162_162568


namespace total_balloons_correct_l162_162932

-- Define the number of balloons each person has
def dan_balloons : ℕ := 29
def tim_balloons : ℕ := 7 * dan_balloons
def molly_balloons : ℕ := 5 * dan_balloons

-- Define the total number of balloons
def total_balloons : ℕ := dan_balloons + tim_balloons + molly_balloons

-- The theorem to prove
theorem total_balloons_correct : total_balloons = 377 :=
by
  -- This part is where the proof will go
  sorry

end total_balloons_correct_l162_162932


namespace squares_on_sides_of_triangle_l162_162450

theorem squares_on_sides_of_triangle (A B C : ℕ) (hA : A = 3^2) (hB : B = 4^2) (hC : C = 5^2) : 
  A + B = C :=
by 
  rw [hA, hB, hC] 
  exact Nat.add_comm 9 16 ▸ rfl

end squares_on_sides_of_triangle_l162_162450


namespace pattern_equation_l162_162130

theorem pattern_equation (n : ℕ) (h : n ≥ 1) : 
  (Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2))) :=
by
  sorry

end pattern_equation_l162_162130


namespace similar_inscribed_triangle_exists_l162_162600

variable {α : Type*} [LinearOrderedField α]

-- Representing points and triangles
structure Point (α : Type*) := (x : α) (y : α)
structure Triangle (α : Type*) := (A B C : Point α)

-- Definitions for inscribed triangles and similarity conditions
def isInscribed (inner outer : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

def areSimilar (Δ1 Δ2 : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

-- Main theorem
theorem similar_inscribed_triangle_exists (Δ₁ Δ₂ : Triangle α) (h_ins : isInscribed Δ₂ Δ₁) :
  ∃ Δ₃ : Triangle α, isInscribed Δ₃ Δ₂ ∧ areSimilar Δ₁ Δ₃ :=
sorry

end similar_inscribed_triangle_exists_l162_162600


namespace inequality_l162_162731

theorem inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 :=
sorry

end inequality_l162_162731


namespace min_distance_value_l162_162713

theorem min_distance_value (x1 x2 y1 y2 : ℝ) 
  (h1 : (e ^ x1 + 2 * x1) / (3 * y1) = 1 / 3)
  (h2 : (x2 - 1) / y2 = 1 / 3) :
  ((x1 - x2)^2 + (y1 - y2)^2) = 8 / 5 :=
by
  sorry

end min_distance_value_l162_162713


namespace baseball_game_earnings_l162_162790

theorem baseball_game_earnings (W S : ℝ) 
  (h1 : W + S = 4994.50) 
  (h2 : W = S - 1330.50) : 
  S = 3162.50 := 
by 
  sorry

end baseball_game_earnings_l162_162790


namespace max_roads_no_intersections_l162_162422

theorem max_roads_no_intersections (V : ℕ) (hV : V = 100) : 
  ∃ E : ℕ, E ≤ 3 * V - 6 ∧ E = 294 := 
by 
  sorry

end max_roads_no_intersections_l162_162422


namespace function_symmetry_l162_162403

noncomputable def f (ω : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 6))

theorem function_symmetry (ω : ℝ) (hω : ω > 0) (hT : (2 * Real.pi / ω) = 4 * Real.pi) :
  ∃ (k : ℤ), f ω (2 * k * Real.pi - Real.pi / 3) = f ω 0 := by
  sorry

end function_symmetry_l162_162403


namespace min_segments_for_octagon_perimeter_l162_162470

/-- Given an octagon formed by cutting a smaller rectangle from a larger rectangle,
the minimum number of distinct line segment lengths needed to calculate the perimeter 
of this octagon is 3. --/
theorem min_segments_for_octagon_perimeter (a b c d e f g h : ℝ)
  (cond : a = c ∧ b = d ∧ e = g ∧ f = h) :
  ∃ (u v w : ℝ), u ≠ v ∧ v ≠ w ∧ u ≠ w :=
by
  sorry

end min_segments_for_octagon_perimeter_l162_162470


namespace f_value_at_3_l162_162462

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

def periodic_shift (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (x + 2) = f x + 2

theorem f_value_at_3 (h_odd : odd_function f) (h_value : f (-1) = 1/2) (h_periodic : periodic_shift f) : 
  f 3 = 3 / 2 := 
sorry

end f_value_at_3_l162_162462


namespace diamonds_in_F10_l162_162190

def diamonds_in_figure (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 3 * (Nat.add (Nat.mul (n - 1) n) 0) / 2

theorem diamonds_in_F10 : diamonds_in_figure 10 = 136 :=
by
  sorry

end diamonds_in_F10_l162_162190


namespace power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l162_162493

theorem power_of_3_mod_5 (n : ℕ) : (3^n % 5) = 
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 4
  else if n % 4 = 3 then 2
  else 1 := by
  sorry

theorem remainder_of_3_pow_2023_mod_5 : 3^2023 % 5 = 2 := by
  have h : 2023 % 4 = 3 := by norm_num
  rw power_of_3_mod_5 2023
  simp [h]
  sorry

end power_of_3_mod_5_remainder_of_3_pow_2023_mod_5_l162_162493


namespace number_of_streams_l162_162285

theorem number_of_streams (S A B C D : Type) (f : S → A) (f1 : A → B) :
  (∀ (x : ℕ), x = 1000 → 
  (x * 375 / 1000 = 375 ∧ x * 625 / 1000 = 625) ∧ 
  (S ≠ C ∧ S ≠ D ∧ C ≠ D)) →
  -- Introduce some conditions to represent the described transition process
  -- Specifically the conditions mentioning the lakes and transitions 
  ∀ (transition_count : ℕ), 
    (transition_count = 4) →
    ∃ (number_of_streams : ℕ), number_of_streams = 3 := 
sorry

end number_of_streams_l162_162285


namespace cube_volume_l162_162315

theorem cube_volume (d_AF : Real) (h : d_AF = 6 * Real.sqrt 2) : ∃ (V : Real), V = 216 :=
by {
  sorry
}

end cube_volume_l162_162315


namespace calculate_expression_l162_162204

theorem calculate_expression :
  16 * (1/2) * 4 * (1/16) / 2 = 1 := 
by
  sorry

end calculate_expression_l162_162204


namespace house_height_l162_162366

theorem house_height
  (tree_height : ℕ) (tree_shadow : ℕ)
  (house_shadow : ℕ) (h : ℕ) :
  tree_height = 15 →
  tree_shadow = 18 →
  house_shadow = 72 →
  (h / tree_height) = (house_shadow / tree_shadow) →
  h = 60 :=
by
  intros h1 h2 h3 h4
  have h5 : h / 15 = 72 / 18 := by
    rw [h1, h2, h3] at h4
    exact h4
  sorry

end house_height_l162_162366


namespace parabola_ellipse_tangency_l162_162884

theorem parabola_ellipse_tangency :
  ∃ (a b : ℝ), (∀ x y, y = x^2 - 5 → (x^2 / a) + (y^2 / b) = 1) →
               (∃ x, y = x^2 - 5 ∧ (x^2 / a) + ((x^2 - 5)^2 / b) = 1) ∧
               a = 1/10 ∧ b = 1 :=
by
  sorry

end parabola_ellipse_tangency_l162_162884


namespace count_perfect_cube_or_fourth_power_lt_1000_l162_162411

theorem count_perfect_cube_or_fourth_power_lt_1000 :
  ∃ n, n = 14 ∧ (∀ x, (0 < x ∧ x < 1000 ∧ (∃ k, x = k^3 ∨ x = k^4)) ↔ ∃ i, i < n) :=
by sorry

end count_perfect_cube_or_fourth_power_lt_1000_l162_162411


namespace rectangular_plot_perimeter_l162_162613

theorem rectangular_plot_perimeter (w : ℝ) (P : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  (cost_per_meter = 6.5) →
  (total_cost = 1430) →
  (P = 2 * (w + (w + 10))) →
  (cost_per_meter * P = total_cost) →
  P = 220 :=
by
  sorry

end rectangular_plot_perimeter_l162_162613


namespace stratified_sampling_third_year_students_l162_162176

/-- 
A university's mathematics department has a total of 5000 undergraduate students, 
with the first, second, third, and fourth years having a ratio of their numbers as 4:3:2:1. 
If stratified sampling is employed to select a sample of 200 students from all undergraduates,
prove that the number of third-year students to be sampled is 40.
-/
theorem stratified_sampling_third_year_students :
  let total_students := 5000
  let ratio_first_second_third_fourth := (4, 3, 2, 1)
  let sample_size := 200
  let third_year_ratio := 2
  let total_ratio_units := 4 + 3 + 2 + 1
  let proportion_third_year := third_year_ratio / total_ratio_units
  let expected_third_year_students := sample_size * proportion_third_year
  expected_third_year_students = 40 :=
by
  sorry

end stratified_sampling_third_year_students_l162_162176


namespace log_base_equal_l162_162385

noncomputable def logx (b x : ℝ) := Real.log x / Real.log b

theorem log_base_equal {x : ℝ} (h : 0 < x ∧ x ≠ 1) :
  logx 81 x = logx 16 2 → x = 3 :=
by
  intro h1
  sorry

end log_base_equal_l162_162385


namespace number_of_positive_divisors_2310_l162_162227

theorem number_of_positive_divisors_2310 : 
  let n := 2310 in
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)] in
  let t := (factorization.map (λ p : ℕ × ℕ, p.snd + 1)).prod in
  t = 32 :=
by
  let n := 2310
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
  let t := (factorization.map (λ p, p.snd + 1)).prod
  sorry

end number_of_positive_divisors_2310_l162_162227


namespace Amelia_wins_probability_correct_l162_162518

-- Define the probabilities
def probability_Amelia_heads := 1 / 3
def probability_Blaine_heads := 2 / 5

-- The infinite geometric series sum calculation for Amelia to win
def probability_Amelia_wins :=
  probability_Amelia_heads * (1 / (1 - (1 - probability_Amelia_heads) * (1 - probability_Blaine_heads)))

-- Given values p and q from the conditions
def p := 5
def q := 9

-- The correct answer $\frac{5}{9}$
def Amelia_wins_correct := 5 / 9

-- Prove that the probability calculation matches the given $\frac{5}{9}$, and find q - p
theorem Amelia_wins_probability_correct :
  probability_Amelia_wins = Amelia_wins_correct ∧ q - p = 4 := by
  sorry

end Amelia_wins_probability_correct_l162_162518


namespace largest_among_four_l162_162260

theorem largest_among_four (a b : ℝ) (h : 0 < a ∧ a < b ∧ a + b = 1) :
  a^2 + b^2 = max (max (max a (1/2)) (2*a*b)) (a^2 + b^2) :=
by
  sorry

end largest_among_four_l162_162260


namespace max_xy_l162_162729

variable {x y : ℝ}

theorem max_xy (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 * x + 8 * y = 48) : x * y ≤ 24 :=
sorry

end max_xy_l162_162729


namespace days_per_week_equals_two_l162_162859

-- Definitions based on conditions
def hourly_rate : ℕ := 10
def hours_per_delivery : ℕ := 3
def total_weeks : ℕ := 6
def total_earnings : ℕ := 360

-- Proof statement: determine the number of days per week Jamie delivers flyers is 2
theorem days_per_week_equals_two (d : ℕ) :
  10 * (total_weeks * d * hours_per_delivery) = total_earnings → d = 2 := by
  sorry

end days_per_week_equals_two_l162_162859


namespace decimal_to_binary_123_l162_162639

/-- The base 2 representation of 123 in decimal is 1111011 in binary. -/
theorem decimal_to_binary_123 : nat.to_digits 2 123 = [1, 1, 1, 1, 0, 1, 1] :=
by 
  sorry

end decimal_to_binary_123_l162_162639


namespace sufficient_condition_l162_162146

theorem sufficient_condition (a : ℝ) (h : a ≥ 10) : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → x^2 - a ≤ 0 :=
by
  sorry

end sufficient_condition_l162_162146


namespace pow_mod_remainder_l162_162491

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l162_162491


namespace area_of_rectangle_perimeter_of_rectangle_l162_162107

-- Define the input conditions
variables (AB AC BC : ℕ)
def is_right_triangle (a b c : ℕ) : Prop := a * a + b * b = c * c
def area_rect (l w : ℕ) : ℕ := l * w
def perimeter_rect (l w : ℕ) : ℕ := 2 * (l + w)

-- Given the conditions for the problem
axiom AB_eq_15 : AB = 15
axiom AC_eq_17 : AC = 17
axiom right_triangle : is_right_triangle AB BC AC

-- Prove the area and perimeter of the rectangle
theorem area_of_rectangle : area_rect AB BC = 120 := by sorry

theorem perimeter_of_rectangle : perimeter_rect AB BC = 46 := by sorry

end area_of_rectangle_perimeter_of_rectangle_l162_162107


namespace rhinos_horn_segment_area_l162_162276

theorem rhinos_horn_segment_area :
  let full_circle_area (r : ℝ) := π * r^2
  let quarter_circle_area (r : ℝ) := (1 / 4) * full_circle_area r
  let half_circle_area (r : ℝ) := (1 / 2) * full_circle_area r
  let larger_quarter_circle_area := quarter_circle_area 4
  let smaller_half_circle_area := half_circle_area 2
  let rhinos_horn_segment_area := larger_quarter_circle_area - smaller_half_circle_area
  rhinos_horn_segment_area = 2 * π := 
by sorry 

end rhinos_horn_segment_area_l162_162276


namespace simplify_fraction_l162_162206

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b)) :=
by
  sorry

end simplify_fraction_l162_162206


namespace find_a_l162_162059

def has_root_greater_than_zero (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ ((3 * x - 1) / (x - 3) = a / (3 - x) - 1)

theorem find_a (a : ℝ) : has_root_greater_than_zero a → a = -8 :=
by
  sorry

end find_a_l162_162059


namespace puppies_per_dog_l162_162893

def dogs := 15
def puppies := 75

theorem puppies_per_dog : puppies / dogs = 5 :=
by {
  sorry
}

end puppies_per_dog_l162_162893


namespace power_series_rational_function_l162_162825

-- Define the power series
def power_series (f : ℚ → ℚ) : ℕ → bool := 
λ n, if f (2^(-n)) ≠ 0 then true else false

-- Define the main problem
theorem power_series_rational_function (f : ℚ → ℚ) (hf : ∀ n, f (2^(-n)) = 0 ∨ f (2^(-n)) = 1) 
(h : ∃ (q : ℚ), f 0.5 = q) :
∃ (g : ℚ → ℚ), ∃ (p q : ℚ), ∀ x, g x = (p / q) :=
sorry

end power_series_rational_function_l162_162825


namespace least_positive_24x_16y_l162_162900

theorem least_positive_24x_16y (x y : ℤ) : ∃ a : ℕ, a > 0 ∧ a = 24 * x + 16 * y ∧ ∀ b : ℕ, b = 24 * x + 16 * y → b > 0 → b ≥ a :=
sorry

end least_positive_24x_16y_l162_162900


namespace range_of_a_l162_162846

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a + 3) * Real.exp (a * x)

theorem range_of_a (a : ℝ) : 
  (∀ x y, x ≤ y → f a x ≤ f a y) ∨ (∀ x y, x ≤ y → f a x ≥ f a y) → 
  a ∈ Set.Ico (-2 : ℝ) 0 :=
sorry

end range_of_a_l162_162846


namespace initial_mixture_equals_50_l162_162972

theorem initial_mixture_equals_50 (x : ℝ) (h1 : 0.10 * x + 10 = 0.25 * (x + 10)) : x = 50 :=
by
  sorry

end initial_mixture_equals_50_l162_162972


namespace height_is_geometric_mean_of_bases_l162_162988

-- Given conditions
variables (a c m : ℝ)
-- we declare the condition that the given trapezoid is symmetric and tangential
variables (isSymmetricTangentialTrapezoid : Prop)

-- The theorem to be proven
theorem height_is_geometric_mean_of_bases 
(isSymmetricTangentialTrapezoid: isSymmetricTangentialTrapezoid) 
: m = Real.sqrt (a * c) :=
sorry

end height_is_geometric_mean_of_bases_l162_162988


namespace reciprocal_neg_5_l162_162323

theorem reciprocal_neg_5 : ∃ x : ℚ, -5 * x = 1 ∧ x = -1/5 :=
by
  sorry

end reciprocal_neg_5_l162_162323


namespace solve_system_of_equations_l162_162610

theorem solve_system_of_equations :
  ∃ (x y z : ℤ), (x + y + z = 6) ∧ (x + y * z = 7) ∧ 
  ((x = 7 ∧ y = 0 ∧ z = -1) ∨ 
   (x = 7 ∧ y = -1 ∧ z = 0) ∨ 
   (x = 1 ∧ y = 3 ∧ z = 2) ∨ 
   (x = 1 ∧ y = 2 ∧ z = 3)) :=
sorry

end solve_system_of_equations_l162_162610


namespace unique_solution_quadratic_eq_l162_162829

theorem unique_solution_quadratic_eq (p : ℝ) (h_nonzero : p ≠ 0) : (∀ x : ℝ, p * x^2 - 20 * x + 4 = 0) → p = 25 :=
by
  sorry

end unique_solution_quadratic_eq_l162_162829


namespace cent_piece_value_l162_162013

theorem cent_piece_value (Q P : ℕ) 
  (h1 : Q + P = 29)
  (h2 : 25 * Q + P = 545)
  (h3 : Q = 17) : 
  P = 120 := by
  sorry

end cent_piece_value_l162_162013


namespace election_majority_l162_162975

theorem election_majority
  (total_votes : ℕ)
  (winning_percent : ℝ)
  (other_percent : ℝ)
  (votes_cast : total_votes = 700)
  (winning_share : winning_percent = 0.84)
  (other_share : other_percent = 0.16) :
  ∃ majority : ℕ, majority = 476 := by
  sorry

end election_majority_l162_162975


namespace remainder_of_repeated_23_l162_162154

theorem remainder_of_repeated_23 {n : ℤ} (n : ℤ) (hn : n = 23 * 10^(2*23)) : 
  (n % 32) = 19 :=
sorry

end remainder_of_repeated_23_l162_162154


namespace total_amount_collected_l162_162889

-- Define ticket prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def total_tickets_sold : ℕ := 130
def adult_tickets_sold : ℕ := 40

-- Calculate the number of child tickets sold
def child_tickets_sold : ℕ := total_tickets_sold - adult_tickets_sold

-- Calculate the total amount collected from adult tickets
def total_adult_amount_collected : ℕ := adult_tickets_sold * adult_ticket_price

-- Calculate the total amount collected from child tickets
def total_child_amount_collected : ℕ := child_tickets_sold * child_ticket_price

-- Prove the total amount collected from ticket sales
theorem total_amount_collected : total_adult_amount_collected + total_child_amount_collected = 840 := by
  sorry

end total_amount_collected_l162_162889


namespace total_kids_in_lawrence_county_l162_162530

def kids_stayed_home : ℕ := 644997
def kids_went_to_camp : ℕ := 893835
def kids_from_outside : ℕ := 78

theorem total_kids_in_lawrence_county : kids_stayed_home + kids_went_to_camp = 1538832 := by
  sorry

end total_kids_in_lawrence_county_l162_162530


namespace max_value_when_a_zero_range_of_a_for_one_zero_l162_162076

-- Definition of the function f(x)
def f (a x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * ln x

-- Proof Problem 1: maximum value of f(x) when a = 0
theorem max_value_when_a_zero :
  ∀ x > 0, f 0 x ≤ -1 ∧ f 0 1 = -1 := 
sorry

-- Proof Problem 2: range of values for a such that f(x) has exactly one zero
theorem range_of_a_for_one_zero :
  (∀ a : ℝ, (∃! x : ℝ, f a x = 0) ↔ (a > 0)) :=
sorry

end max_value_when_a_zero_range_of_a_for_one_zero_l162_162076


namespace police_emergency_number_has_prime_gt_7_l162_162803

theorem police_emergency_number_has_prime_gt_7 (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
sorry

end police_emergency_number_has_prime_gt_7_l162_162803


namespace problem_quadratic_roots_l162_162068

theorem problem_quadratic_roots (m : ℝ) :
  (∀ x : ℝ, (m + 3) * x^2 - 4 * m * x + 2 * m - 1 = 0 →
    (∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ |x₁| > x₂)) ↔ -3 < m ∧ m < 0 :=
sorry

end problem_quadratic_roots_l162_162068


namespace car_speed_l162_162505

theorem car_speed (distance: ℚ) (hours minutes: ℚ) (h_distance: distance = 360) (h_hours: hours = 4) (h_minutes: minutes = 30) : 
  (distance / (hours + (minutes / 60))) = 80 := by
  sorry

end car_speed_l162_162505


namespace vector_b_exist_l162_162409

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem vector_b_exist 
  (a : ℝ × ℝ)
  (hab : vector_magnitude (6, -2 * sqrt 3) = 2 * vector_magnitude a)
  (hbc : dot_product a (6, -2 * sqrt 3) = vector_magnitude a * vector_magnitude (6, -2 * sqrt 3) * real.cos (π / 3))
  (hbb: ∀ b : ℝ × ℝ, b = (6, -2 * sqrt 3) ∨ b = (0, 4 * sqrt 3)) :
  (∀ b : ℝ × ℝ,
    vector_magnitude b = 2 * vector_magnitude a ∧
    dot_product a b = vector_magnitude a * vector_magnitude b * real.cos (π / 3) → 
    (b = (0, 4*sqrt 3) ∨ b = (6, -2 * sqrt 3))) := 
sorry

end vector_b_exist_l162_162409


namespace product_of_radii_l162_162062

theorem product_of_radii (x y r₁ r₂ : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hr₁ : (x - r₁)^2 + (y - r₁)^2 = r₁^2)
  (hr₂ : (x - r₂)^2 + (y - r₂)^2 = r₂^2)
  (hroots : r₁ + r₂ = 2 * (x + y)) : r₁ * r₂ = x^2 + y^2 := by
  sorry

end product_of_radii_l162_162062


namespace ratio_is_five_ninths_l162_162907

-- Define the conditions
def total_profit : ℕ := 48000
def total_income : ℕ := 108000

-- Define the total spending based on conditions
def total_spending : ℕ := total_income - total_profit

-- Define the ratio of spending to income
def ratio_spending_to_income : ℚ := total_spending / total_income

-- The theorem we need to prove
theorem ratio_is_five_ninths : ratio_spending_to_income = 5 / 9 := 
  sorry

end ratio_is_five_ninths_l162_162907


namespace last_number_of_ratio_l162_162343

theorem last_number_of_ratio (A B C : ℕ) (h1 : 5 * B = A) (h2 : 4 * B = C) (h3 : A + B + C = 1000) : C = 400 :=
by
  sorry

end last_number_of_ratio_l162_162343


namespace shortest_chord_value_of_m_l162_162706

theorem shortest_chord_value_of_m :
  (∃ m : ℝ,
      (∀ x y : ℝ, mx + y - 2 * m - 1 = 0) ∧
      (∀ x y : ℝ, x ^ 2 + y ^ 2 - 2 * x - 4 * y = 0) ∧
      (mx + y - 2 * m - 1 = 0 → ∃ x y : ℝ, (x, y) = (2, 1))
  ) → m = -1 :=
by
  sorry

end shortest_chord_value_of_m_l162_162706


namespace number_of_divisors_2310_l162_162226

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end number_of_divisors_2310_l162_162226


namespace range_of_f_l162_162380

noncomputable def f : ℝ → ℝ := λ x, if x = -5 then 0 else 3 * (x - 4)

theorem range_of_f :
  (Set.range f) = Set.Ioo (-∞) (-27) ∪ Set.Ioo (-27) ∞ := by
  sorry

end range_of_f_l162_162380


namespace rational_expression_l162_162133

theorem rational_expression {x : ℚ} : (∃ a : ℚ, x / (x^2 + x + 1) = a) → (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) := by
  sorry

end rational_expression_l162_162133


namespace contractor_original_days_l162_162795

noncomputable def original_days (total_laborers absent_laborers working_laborers days_worked : ℝ) : ℝ :=
  (working_laborers * days_worked) / (total_laborers - absent_laborers)

-- Our conditions:
def total_laborers : ℝ := 21.67
def absent_laborers : ℝ := 5
def working_laborers : ℝ := 16.67
def days_worked : ℝ := 13

-- Our main theorem:
theorem contractor_original_days :
  original_days total_laborers absent_laborers working_laborers days_worked = 10 := 
by
  sorry

end contractor_original_days_l162_162795


namespace president_vice_committee_count_l162_162284

open Nat

noncomputable def choose_president_vice_committee (total_people : ℕ) : ℕ :=
  let choose : ℕ := 56 -- binomial 8 3 is 56
  total_people * (total_people - 1) * choose

theorem president_vice_committee_count :
  choose_president_vice_committee 10 = 5040 :=
by
  sorry

end president_vice_committee_count_l162_162284


namespace negation_of_P_l162_162093

-- Define the proposition P
def P : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

-- Define the negation of P
def not_P : Prop := ∀ x : ℝ, Real.exp x - x - 1 > 0

-- The theorem statement
theorem negation_of_P : ¬ P = not_P :=
by
  sorry

end negation_of_P_l162_162093


namespace max_value_f_at_a0_l162_162072

def f (x : ℝ) : ℝ := - (1 / x) - Real.log x

theorem max_value_f_at_a0 : ∀ x > 0 , ∃ y : ℝ, y = f x → y ≤ f 1 := 
by
  sorry

end max_value_f_at_a0_l162_162072


namespace charity_event_equation_l162_162997

variable (x : ℕ)

theorem charity_event_equation : x + 5 * (12 - x) = 48 :=
sorry

end charity_event_equation_l162_162997


namespace quadratic_inequality_solution_set_l162_162848

variable (a b c : ℝ)

theorem quadratic_inequality_solution_set (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 → (-1 / 3 < x ∧ x < 2)) :
  ∀ x : ℝ, cx^2 + bx + a < 0 → (-3 < x ∧ x < 1 / 2) :=
by
  sorry

end quadratic_inequality_solution_set_l162_162848


namespace radius_of_larger_circle_l162_162012

theorem radius_of_larger_circle (r : ℝ) (R : ℝ) 
  (h1 : ∀ a b c : ℝ, a = 2 ∧ b = 2 ∧ c = 2) 
  (h2 : ∀ x y z : ℝ, (x = 4) ∧ (y = 4) ∧ (z = 4) ) 
  (h3 : ∀ A B : ℝ, A * 2 = 2) : 
  R = 2 + 2 * Real.sqrt 3 :=
by
  sorry

end radius_of_larger_circle_l162_162012


namespace smallest_positive_integer_l162_162771

-- Given integers m and n, prove the smallest positive integer of the form 2017m + 48576n
theorem smallest_positive_integer (m n : ℤ) : 
  ∃ m n : ℤ, 2017 * m + 48576 * n = 1 := by
sorry

end smallest_positive_integer_l162_162771


namespace expected_salary_correct_l162_162504

open_locale classical
noncomputable theory

def probability_distribution (n k : ℕ) : ℕ → ℚ
| 0 := 1 / (nat.choose n k : ℚ)
| 1 := (nat.choose k 1) * (nat.choose (n - k) (k - 1)) / (nat.choose n k : ℚ)
| 2 := (nat.choose k 2) * (nat.choose (n - k) (k - 2)) / (nat.choose n k : ℚ)
| 3 := (nat.choose k 3) * (nat.choose (n - k) (k - 3)) / (nat.choose n k : ℚ)
| 4 := 1 / (nat.choose n k : ℚ)
| _ := 0

def expected_salary (n k : ℕ) : ℚ :=
  let p := probability_distribution n k in
  3500 * p 4 + 2800 * p 3 + 2100 * (p 0 + p 1 + p 2)

theorem expected_salary_correct :
  expected_salary 8 4 = 2280 := 
by
  unfold expected_salary probability_distribution
  sorry

end expected_salary_correct_l162_162504


namespace range_of_a_l162_162548

theorem range_of_a 
  (x y a : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy : x + y + 3 = x * y) 
  (h_a : ∀ x y : ℝ, (x + y)^2 - a * (x + y) + 1 ≥ 0) :
  a ≤ 37 / 6 := 
sorry

end range_of_a_l162_162548


namespace calc_fraction_l162_162044

theorem calc_fraction :
  ((1 / 3 + 1 / 6) * (4 / 7) * (5 / 9) = 10 / 63) :=
by
  sorry

end calc_fraction_l162_162044


namespace single_room_cost_l162_162815

theorem single_room_cost (total_rooms : ℕ) (single_rooms : ℕ) (double_room_cost : ℕ) 
  (total_revenue : ℤ) (x : ℤ) : 
  total_rooms = 260 → 
  single_rooms = 64 → 
  double_room_cost = 60 → 
  total_revenue = 14000 → 
  64 * x + (total_rooms - single_rooms) * double_room_cost = total_revenue → 
  x = 35 := 
by 
  intros h_total_rooms h_single_rooms h_double_room_cost h_total_revenue h_eqn 
  -- Add steps for proving if necessary
  sorry

end single_room_cost_l162_162815


namespace jeans_discount_rates_l162_162437

theorem jeans_discount_rates
    (M F P : ℝ) 
    (regular_price_moose jeans_regular_price_fox jeans_regular_price_pony : ℝ)
    (moose_count fox_count pony_count : ℕ)
    (total_discount : ℝ) :
    regular_price_moose = 20 →
    regular_price_fox = 15 →
    regular_price_pony = 18 →
    moose_count = 2 →
    fox_count = 3 →
    pony_count = 2 →
    total_discount = 12.48 →
    (M + F + P = 0.32) →
    (F + P = 0.20) →
    (moose_count * M * regular_price_moose + fox_count * F * regular_price_fox + pony_count * P * regular_price_pony = total_discount) →
    M = 0.12 ∧ F = 0.0533 ∧ P = 0.1467 :=
by
  intros
  sorry -- The proof is not required

end jeans_discount_rates_l162_162437


namespace line_through_two_points_l162_162878

theorem line_through_two_points (P Q : ℝ × ℝ) (hP : P = (2, 5)) (hQ : Q = (2, -5)) :
  (∀ (x y : ℝ), (x, y) = P ∨ (x, y) = Q → x = 2) :=
by
  sorry

end line_through_two_points_l162_162878


namespace minimum_questions_needed_a_l162_162333

theorem minimum_questions_needed_a (n : ℕ) (m : ℕ) (h1 : m = n) (h2 : m < 2 ^ n) :
  ∃Q : ℕ, Q = n := sorry

end minimum_questions_needed_a_l162_162333


namespace edges_parallel_to_axes_l162_162721

theorem edges_parallel_to_axes (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ)
  (hx : x1 = 0 ∨ y1 = 0 ∨ z1 = 0)
  (hy : x2 = x1 + 1 ∨ y2 = y1 + 1 ∨ z2 = z1 + 1)
  (hz : x3 = x1 + 1 ∨ y3 = y1 + 1 ∨ z3 = z1 + 1)
  (hv : x4*y4*z4 = 2011) :
  (x2-x1 ∣ 2011) ∧ (y2-y1 ∣ 2011) ∧ (z2-z1 ∣ 2011) := 
sorry

end edges_parallel_to_axes_l162_162721


namespace dennis_initial_money_l162_162360

theorem dennis_initial_money :
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  cost_of_shirts + total_change = 50 :=
by
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  show cost_of_shirts + total_change = 50
  sorry

end dennis_initial_money_l162_162360


namespace count_ordered_pairs_l162_162935

open Finset

-- Define ℕ set 1 to 10
def U : Finset ℕ := (range 11).filter (λ x, x > 0)

-- Define the conditions
def conditions (A B : Finset ℕ) : Prop :=
  A ∪ B = U ∧
  A ∩ B = ∅ ∧
  ¬ A.card ∈ A ∧
  A.card % 2 = 0 ∧
  ¬ B.card ∈ B

-- Define the function to count valid sets
noncomputable def count_valid_sets : ℕ :=
  ∑ n in U.filter (λ n, n % 2 = 0 ∧ (n ≠ 0 ∧ n ≠ 10)), 
    (choose (10 - 1 - n) n) + (choose (10 - 1 - 10 + n) (10 - n))

-- The final theorem
theorem count_ordered_pairs : count_valid_sets = 43 := by sorry

end count_ordered_pairs_l162_162935


namespace poly_eq_l162_162453

-- Definition of the polynomials f(x) and g(x)
def f (x : ℝ) := x^4 + 4*x^3 + 8*x
def g (x : ℝ) := 10*x^4 + 30*x^3 + 29*x^2 + 2*x + 5

-- Define p(x) as a function that satisfies the given condition
def p (x : ℝ) := 9*x^4 + 26*x^3 + 29*x^2 - 6*x + 5

-- Prove that the function p(x) satisfies the equation
theorem poly_eq : ∀ x : ℝ, p x + f x = g x :=
by
  intro x
  -- Add a marker to indicate that this is where the proof would go
  sorry

end poly_eq_l162_162453


namespace pow_mod_remainder_l162_162490

theorem pow_mod_remainder :
  (3 ^ 2023) % 5 = 2 :=
by sorry

end pow_mod_remainder_l162_162490


namespace sum_of_distinct_prime_factors_of_number_is_10_l162_162684

-- Define the constant number 9720
def number : ℕ := 9720

-- Define the distinct prime factors of 9720
def distinct_prime_factors_of_number : List ℕ := [2, 3, 5]

-- Sum function for the list of distinct prime factors
def sum_of_distinct_prime_factors (lst : List ℕ) : ℕ :=
  lst.foldr (.+.) 0

-- The main theorem to prove
theorem sum_of_distinct_prime_factors_of_number_is_10 :
  sum_of_distinct_prime_factors distinct_prime_factors_of_number = 10 := by
  sorry

end sum_of_distinct_prime_factors_of_number_is_10_l162_162684


namespace polynomial_simplification_l162_162638

noncomputable def given_polynomial (x : ℝ) : ℝ :=
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 + 15 - 17 * x + 19 * x^2 + 2 * x^3

theorem polynomial_simplification (x : ℝ) :
  given_polynomial x = 2 * x^3 - x^2 - 11 * x + 27 :=
by
  -- The proof is skipped
  sorry

end polynomial_simplification_l162_162638


namespace floor_sqrt_23_squared_l162_162216

theorem floor_sqrt_23_squared : (Nat.floor (Real.sqrt 23)) ^ 2 = 16 :=
by
  -- Proof is omitted
  sorry

end floor_sqrt_23_squared_l162_162216


namespace oranges_to_put_back_l162_162814

variables (A O x : ℕ)

theorem oranges_to_put_back
    (h1 : 40 * A + 60 * O = 560)
    (h2 : A + O = 10)
    (h3 : (40 * A + 60 * (O - x)) / (10 - x) = 50) : x = 6 := 
sorry

end oranges_to_put_back_l162_162814


namespace garden_roller_length_l162_162349

noncomputable def length_of_garden_roller (d : ℝ) (A : ℝ) (revolutions : ℕ) (π : ℝ) : ℝ :=
  let r := d / 2
  let area_in_one_revolution := A / revolutions
  let L := area_in_one_revolution / (2 * π * r)
  L

theorem garden_roller_length :
  length_of_garden_roller 1.2 37.714285714285715 5 (22 / 7) = 2 := by
  sorry

end garden_roller_length_l162_162349


namespace probability_both_selected_is_correct_l162_162027

def prob_selection_x : ℚ := 1 / 7
def prob_selection_y : ℚ := 2 / 9
def prob_both_selected : ℚ := prob_selection_x * prob_selection_y

theorem probability_both_selected_is_correct : prob_both_selected = 2 / 63 := 
by 
  sorry

end probability_both_selected_is_correct_l162_162027


namespace max_value_of_f_when_a_is_zero_range_of_a_for_unique_zero_l162_162082

-- Part (1)
theorem max_value_of_f_when_a_is_zero (x : ℝ) (h : x > 0) : 
  let f := λ x, - (1 / x) - log x in
  (∀ x > 0, f x ≤ -1) ∧ (∃ x > 0, f x = -1) :=
sorry

-- Part (2)
theorem range_of_a_for_unique_zero (a : ℝ) :
  (∀ x : ℝ, f x = 0 ↔ x = 1) → a ∈ set.Ioi 0 :=
sorry

noncomputable def f (x a : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * log x

end max_value_of_f_when_a_is_zero_range_of_a_for_unique_zero_l162_162082


namespace solution_to_problem_l162_162223

theorem solution_to_problem (a x y n m : ℕ) (h1 : a * (x^n - x^m) = (a * x^m - 4) * y^2)
  (h2 : m % 2 = n % 2) (h3 : (a * x) % 2 = 1) : 
  x = 1 :=
sorry

end solution_to_problem_l162_162223


namespace non_convergence_uniform_integrable_seq_l162_162601

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [ProbMeasure P]

/-- Definitions from conditions --/
noncomputable def U : Ω → ℝ := sorry -- Uniform[0, 1]
noncomputable def V : Ω → ℝ := sorry -- Independent, Uniform[0, 1]
noncomputable def 𝔾 : MeasurableSpace Ω := sorry -- σ(V)

noncomputable def X_kn (k n : ℕ) : Ω → ℝ :=
  λ ω, n * (if n * U ω ≤ 1 then 1 else 0) * (if k-1 < n * V ω ∧ n * V ω ≤ k then 1 else 0)

noncomputable def xi_n (n : ℕ) : Ω → ℝ :=
  sorry -- List of X_kn like (X_11, X_12, X_22, X_13, X_23, X_33, ...)

-- Statement
theorem non_convergence_uniform_integrable_seq :
  (∀ (n : ℕ), UniformIntegrable (xi_n n) P) ∧ 
  (∃ (ω : Ω), (tsum (λ n, xi_n n ω) = 0 ∧ 
                 tsum (λ n, condexp 𝔾 (xi_n n) ω) ≠ 0)) :=
sorry

end non_convergence_uniform_integrable_seq_l162_162601


namespace am_gm_problem_l162_162468

theorem am_gm_problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
by 
  sorry

end am_gm_problem_l162_162468


namespace number_of_real_solutions_is_one_l162_162372

noncomputable def num_real_solutions (a b c d : ℝ) : ℕ :=
  let x := Real.sin (a + b + c)
  let y := Real.sin (b + c + d)
  let z := Real.sin (c + d + a)
  let w := Real.sin (d + a + b)
  if (a + b + c + d) % 360 = 0 then 1 else 0

theorem number_of_real_solutions_is_one (a b c d : ℝ) (h : (a + b + c + d) % 360 = 0) :
  num_real_solutions a b c d = 1 :=
by
  sorry

end number_of_real_solutions_is_one_l162_162372


namespace not_p_is_necessary_but_not_sufficient_l162_162552

-- Definitions based on the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = d

def not_p (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ n : ℕ, a (n + 2) - a (n + 1) ≠ d

def not_q (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ¬ is_arithmetic_sequence a d

-- Proof problem statement
theorem not_p_is_necessary_but_not_sufficient (d : ℝ) (a : ℕ → ℝ) :
  (not_p a d → not_q a d) ∧ (not_q a d → not_p a d) = False := 
sorry

end not_p_is_necessary_but_not_sufficient_l162_162552


namespace lcm_20_45_36_l162_162153

-- Definitions from the problem
def num1 : ℕ := 20
def num2 : ℕ := 45
def num3 : ℕ := 36

-- Statement of the proof problem
theorem lcm_20_45_36 : Nat.lcm (Nat.lcm num1 num2) num3 = 180 := by
  sorry

end lcm_20_45_36_l162_162153


namespace factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l162_162819

-- Proof for (1)
theorem factorize_polynomial_1 (a : ℝ) : 2*a^2 - 8*a + 8 = 2*(a - 2)^2 :=
by
  sorry

-- Proof for (2)
theorem factorize_polynomial_2 (x y : ℝ) : x^2 - y^2 + 3*x - 3*y = (x - y)*(x + y + 3) :=
by
  sorry

-- Proof for (3)
theorem triangle_shape (a b c : ℝ) (h : a^2 - ab - ac + bc = 0) : 
  (a = b ∨ a = c) :=
by
  sorry

end factorize_polynomial_1_factorize_polynomial_2_triangle_shape_l162_162819


namespace lattice_point_condition_l162_162034

theorem lattice_point_condition (b : ℚ) :
  (∀ (m : ℚ), (1 / 3 < m ∧ m < b) →
    ∀ x : ℤ, (0 < x ∧ x ≤ 200) →
      ¬ ∃ y : ℤ, y = m * x + 3) →
  b = 68 / 203 := 
sorry

end lattice_point_condition_l162_162034


namespace find_l_find_C3_l162_162701

-- Circle definitions
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 7 = 0

-- Given line passes through common points of C1 and C2
theorem find_l (x y : ℝ) (h1 : C1 x y) (h2 : C2 x y) : x = 1 := by
  sorry

-- Circle C3 passes through intersection points of C1 and C2, and its center lies on y = x
def C3 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def on_line_y_eq_x (x y : ℝ) : Prop := y = x

theorem find_C3 (x y : ℝ) (hx : C3 x y) (hy : on_line_y_eq_x x y) : (x - 1)^2 + (y - 1)^2 = 1 := by
  sorry

end find_l_find_C3_l162_162701


namespace sum_of_roots_proof_l162_162139

noncomputable def sum_of_roots (x1 x2 x3 : ℝ) : ℝ :=
  let eq1 := (11 - x1)^3 + (13 - x1)^3 = (24 - 2 * x1)^3
  let eq2 := (11 - x2)^3 + (13 - x2)^3 = (24 - 2 * x2)^3
  let eq3 := (11 - x3)^3 + (13 - x3)^3 = (24 - 2 * x3)^3
  x1 + x2 + x3

theorem sum_of_roots_proof : sum_of_roots 11 12 13 = 36 :=
  sorry

end sum_of_roots_proof_l162_162139


namespace max_value_a_zero_range_a_one_zero_l162_162090

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - 1/x - (a + 1) * Real.log x

-- Problem (1): Prove that for a = 0, the maximum value of f(x) is -1.
theorem max_value_a_zero : ∃ x > (0:ℝ), ∀ y > 0, f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
by
  -- Proof not needed, inserting sorry
  sorry

-- Problem (2): Prove the range of values for a for which f(x) has exactly one zero is (0, +∞).
theorem range_a_one_zero : ∀ (a : ℝ), (∃ x > (0:ℝ), f a x = 0 ∧ ∀ y ≠ x, y > 0 → f a y ≠ 0) ↔ a > 0 :=
by
  -- Proof not needed, inserting sorry
  sorry

end max_value_a_zero_range_a_one_zero_l162_162090


namespace polynomial_behavior_l162_162051

noncomputable def Q (x : ℝ) : ℝ := x^6 - 6 * x^5 + 10 * x^4 - x^3 - x + 12

theorem polynomial_behavior : 
  (∀ x : ℝ, x < 0 → Q x > 0) ∧ (∃ x : ℝ, x > 0 ∧ Q x = 0) := 
by 
  sorry

end polynomial_behavior_l162_162051


namespace problem_part1_problem_part2_l162_162397

variable (a b : ℝ)

theorem problem_part1 (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 4) :
  9 / a + 1 / b ≥ 4 :=
sorry

theorem problem_part2 (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 4) :
  ∃ a b, (a + 3 / b) * (b + 3 / a) = 12 :=
sorry

end problem_part1_problem_part2_l162_162397


namespace limit_expr_at_pi_l162_162049

theorem limit_expr_at_pi :
  (Real.exp π - Real.exp x) / (Real.sin (5*x) - Real.sin (3*x)) = 1 / 2 * Real.exp π :=
by
  sorry

end limit_expr_at_pi_l162_162049


namespace x_plus_inv_x_eq_8_then_power_4_l162_162714

theorem x_plus_inv_x_eq_8_then_power_4 (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end x_plus_inv_x_eq_8_then_power_4_l162_162714


namespace fg_of_neg2_l162_162961

def f (x : ℤ) : ℤ := x^2 + 4
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_of_neg2 : f (g (-2)) = 20 := by
  sorry

end fg_of_neg2_l162_162961


namespace mean_temperature_correct_l162_162612

-- Define the list of temperatures
def temperatures : List ℤ := [-8, -5, -5, -6, 0, 4]

-- Define the mean temperature calculation
def mean_temperature (temps: List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

-- The theorem we want to prove
theorem mean_temperature_correct :
  mean_temperature temperatures = -10 / 3 :=
by
  sorry

end mean_temperature_correct_l162_162612


namespace glass_bottles_in_second_scenario_l162_162010

theorem glass_bottles_in_second_scenario
  (G P x : ℕ)
  (h1 : 3 * G = 600)
  (h2 : G = P + 150)
  (h3 : x * G + 5 * P = 1050) :
  x = 4 :=
by 
  -- Proof is omitted
  sorry

end glass_bottles_in_second_scenario_l162_162010


namespace is_opposite_if_differ_in_sign_l162_162574

-- Define opposite numbers based on the given condition in the problem:
def opposite_numbers (a b : ℝ) : Prop := a = -b

-- State the theorem based on the translation in c)
theorem is_opposite_if_differ_in_sign (a b : ℝ) (h : a = -b) : opposite_numbers a b := by
  sorry

end is_opposite_if_differ_in_sign_l162_162574


namespace new_angle_after_rotation_l162_162999

def initial_angle : ℝ := 25
def rotation_clockwise : ℝ := 350
def equivalent_rotation := rotation_clockwise - 360  -- equivalent to -10 degrees

theorem new_angle_after_rotation :
  initial_angle + equivalent_rotation = 15 := by
  sorry

end new_angle_after_rotation_l162_162999


namespace simplify_expression_l162_162138

theorem simplify_expression (x : ℝ) : 4 * x - 3 * x^2 + 6 + (8 - 5 * x + 2 * x^2) = - x^2 - x + 14 := by
  sorry

end simplify_expression_l162_162138


namespace sum_of_distinct_prime_factors_of_462_l162_162155

theorem sum_of_distinct_prime_factors_of_462 : 
  ∑ p in (Nat.factors 462).toFinset, p = 23 := by
  sorry

end sum_of_distinct_prime_factors_of_462_l162_162155


namespace calculate_molar_mass_l162_162103

-- Definitions from the conditions
def number_of_moles : ℝ := 8
def weight_in_grams : ℝ := 1600

-- Goal: Prove that the molar mass is 200 grams/mole
theorem calculate_molar_mass : (weight_in_grams / number_of_moles) = 200 :=
by
  sorry

end calculate_molar_mass_l162_162103


namespace no_common_root_of_quadratics_l162_162456

theorem no_common_root_of_quadratics (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬ ∃ x₀ : ℝ, (x₀^2 + b * x₀ + c = 0 ∧ x₀^2 + a * x₀ + d = 0) := 
by
  sorry

end no_common_root_of_quadratics_l162_162456


namespace probability_lucy_picks_from_mathematics_l162_162715

open Rat

theorem probability_lucy_picks_from_mathematics :
  let alphabet_size := 26
  let unique_letters_in_mathematics := 8
  let probability := mk_p nat.gcd 8 alphabet_size 
  let simplified_probability := 4 / 13
  probability = simplified_probability :=
by
  sorry

end probability_lucy_picks_from_mathematics_l162_162715


namespace humans_can_live_l162_162418

variable (earth_surface : ℝ)
variable (water_fraction : ℝ := 3 / 5)
variable (inhabitable_land_fraction : ℝ := 2 / 3)

def inhabitable_fraction : ℝ := (1 - water_fraction) * inhabitable_land_fraction

theorem humans_can_live :
  inhabitable_fraction = 4 / 15 :=
by
  sorry

end humans_can_live_l162_162418


namespace remainder_when_divided_by_x_minus_2_l162_162770

-- We define the polynomial f(x)
def f (x : ℝ) := x^4 - 6 * x^3 + 11 * x^2 + 20 * x - 8

-- We need to show that the remainder when f(x) is divided by (x - 2) is 44
theorem remainder_when_divided_by_x_minus_2 : f 2 = 44 :=
by {
  -- this is where the proof would go
  sorry
}

end remainder_when_divided_by_x_minus_2_l162_162770


namespace newspaper_price_l162_162966

-- Define the conditions as variables
variables 
  (P : ℝ)                    -- Price per edition for Wednesday, Thursday, and Friday
  (total_cost : ℝ := 28)     -- Total cost over 8 weeks
  (sunday_cost : ℝ := 2)     -- Cost of Sunday edition
  (weeks : ℕ := 8)           -- Number of weeks
  (wednesday_thursday_friday_editions : ℕ := 3 * weeks) -- Total number of editions for Wednesday, Thursday, and Friday over 8 weeks

-- Math proof problem statement
theorem newspaper_price : 
  (total_cost - weeks * sunday_cost) / wednesday_thursday_friday_editions = 0.5 :=
  sorry

end newspaper_price_l162_162966


namespace average_rate_of_change_l162_162050

def f (x : ℝ) : ℝ := x^2 - 1

theorem average_rate_of_change : (f 1.1) - (f 1) / (1.1 - 1) = 2.1 :=
by
  sorry

end average_rate_of_change_l162_162050


namespace floor_difference_l162_162946

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l162_162946


namespace lorry_weight_l162_162665

theorem lorry_weight : 
  let empty_lorry_weight := 500
  let apples_weight := 10 * 55
  let oranges_weight := 5 * 45
  let watermelons_weight := 3 * 125
  let firewood_weight := 2 * 75
  let loaded_items_weight := apples_weight + oranges_weight + watermelons_weight + firewood_weight
  let total_weight := empty_lorry_weight + loaded_items_weight
  total_weight = 1800 :=
by 
  sorry

end lorry_weight_l162_162665


namespace largest_n_for_positive_sum_l162_162065

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

def arithmetic_sum (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem largest_n_for_positive_sum (n : ℕ) :
  ∀ (a : ℕ) (S : ℕ → ℤ), (a_1 = 9 ∧ a_5 = 1 ∧ S n > 0) → n = 9 :=
sorry

end largest_n_for_positive_sum_l162_162065


namespace sum_of_non_solutions_l162_162593

theorem sum_of_non_solutions (A B C x: ℝ) 
  (h1 : A = 2) 
  (h2 : B = C / 2) 
  (h3 : C = 28) 
  (eq_inf_solutions : ∀ x, (x ≠ -C ∧ x ≠ -14) → 
  (x + B) * (A * x + 56) = 2 * ((x + C) * (x + 14))) : 
  (-14 + -28) = -42 :=
by
  sorry

end sum_of_non_solutions_l162_162593


namespace range_of_a_l162_162866

variable (a x : ℝ)

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def M (a : ℝ) : Set ℝ := if a = 2 then {2} else {x | 2 ≤ x ∧ x ≤ a}

theorem range_of_a (a : ℝ) (p : x ∈ M a) (h : a ≥ 2) (hpq : Set.Subset (M a) A) : 2 ≤ a ∧ a ≤ 4 :=
  sorry

end range_of_a_l162_162866


namespace max_value_Tn_l162_162680

noncomputable def geom_seq (a : ℕ → ℝ) : Prop := 
∀ n : ℕ, a (n+1) = 2 * a n

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
a 0 * (1 - (2 : ℝ)^n) / (1 - (2 : ℝ))

noncomputable def T_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(9 * sum_first_n_terms a n - sum_first_n_terms a (2 * n)) / (a n * (2 : ℝ)^n)

theorem max_value_Tn (a : ℕ → ℝ) (h : geom_seq a) : 
  ∃ n, T_n a n ≤ 3 :=
sorry

end max_value_Tn_l162_162680


namespace greatest_possible_red_points_l162_162736

-- Definition of the problem in Lean 4
theorem greatest_possible_red_points (points : Finset ℕ) (red blue : Finset ℕ)
  (h_total_points : points.card = 25)
  (h_disjoint : red ∩ blue = ∅)
  (h_union : red ∪ blue = points)
  (h_segment : ∀ (r ∈ red), ∃! b ∈ blue, true) :
  red.card ≤ 13 :=
begin
  -- We assert that the greatest number of red points is at most 13.
  sorry
end

end greatest_possible_red_points_l162_162736


namespace ratio_of_waist_to_hem_l162_162678

theorem ratio_of_waist_to_hem
  (cuffs_length : ℕ)
  (hem_length : ℕ)
  (ruffles_length : ℕ)
  (num_ruffles : ℕ)
  (lace_cost_per_meter : ℕ)
  (total_spent : ℕ)
  (waist_length : ℕ) :
  cuffs_length = 50 →
  hem_length = 300 →
  ruffles_length = 20 →
  num_ruffles = 5 →
  lace_cost_per_meter = 6 →
  total_spent = 36 →
  waist_length = (total_spent / lace_cost_per_meter * 100) -
                (2 * cuffs_length + hem_length + num_ruffles * ruffles_length) →
  waist_length / hem_length = 1 / 3 :=
by
  sorry

end ratio_of_waist_to_hem_l162_162678


namespace quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus_l162_162352

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

def has_exactly_two_axes_of_symmetry (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on symmetry conditions
  sorry

def is_rectangle (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on properties of rectangle
  sorry

def is_rhombus (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on properties of rhombus
  sorry

theorem quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus
  (q : Quadrilateral)
  (h : has_exactly_two_axes_of_symmetry q) :
  is_rectangle q ∨ is_rhombus q := by
  sorry

end quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus_l162_162352


namespace min_goals_in_previous_three_matches_l162_162857

theorem min_goals_in_previous_three_matches 
  (score1 score2 score3 score4 : ℕ)
  (total_after_seven_matches : ℕ)
  (previous_three_goal_sum : ℕ) :
  score1 = 18 →
  score2 = 12 →
  score3 = 15 →
  score4 = 14 →
  total_after_seven_matches ≥ 100 →
  previous_three_goal_sum = total_after_seven_matches - (score1 + score2 + score3 + score4) →
  (previous_three_goal_sum / 3 : ℝ) < ((score1 + score2 + score3 + score4) / 4 : ℝ) →
  previous_three_goal_sum ≥ 41 :=
by
  sorry

end min_goals_in_previous_three_matches_l162_162857


namespace triangle_classification_triangle_classification_right_triangle_classification_obtuse_l162_162304

def triangle_nature (a b c R : ℝ) : String :=
  if a^2 + b^2 + c^2 - 8 * R^2 > 0 then "acute"
  else if a^2 + b^2 + c^2 - 8 * R^2 = 0 then "right"
  else "obtuse"

theorem triangle_classification (a b c R : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : c = max a (max b c)):
  a^2 + b^2 + c^2 - 8 * R^2 > 0 →
  triangle_nature a b c R = "acute" :=
sorry

theorem triangle_classification_right (a b c R : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : c = max a (max b c)):
  a^2 + b^2 + c^2 - 8 * R^2 = 0 →
  triangle_nature a b c R = "right" :=
sorry

theorem triangle_classification_obtuse (a b c R : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : c = max a (max b c)):
  a^2 + b^2 + c^2 - 8 * R^2 < 0 →
  triangle_nature a b c R = "obtuse" :=
sorry

end triangle_classification_triangle_classification_right_triangle_classification_obtuse_l162_162304


namespace base8_to_base10_l162_162212

theorem base8_to_base10 :
  ∀ (n : ℕ), (n = 2 * 8^2 + 4 * 8^1 + 3 * 8^0) → (n = 163) :=
by
  intros n hn
  sorry

end base8_to_base10_l162_162212


namespace commission_percentage_l162_162543

theorem commission_percentage 
  (total_amount : ℝ) 
  (h1 : total_amount = 800) 
  (commission_first_500 : ℝ) 
  (h2 : commission_first_500 = 0.20 * 500) 
  (excess_amount : ℝ) 
  (h3 : excess_amount = (total_amount - 500)) 
  (commission_excess : ℝ) 
  (h4 : commission_excess = 0.25 * excess_amount) 
  (total_commission : ℝ) 
  (h5 : total_commission = commission_first_500 + commission_excess) 
  : (total_commission / total_amount) * 100 = 21.875 := 
by
  sorry

end commission_percentage_l162_162543


namespace probability_two_red_cards_l162_162670

theorem probability_two_red_cards 
  (num_suits : ℕ) (cards_per_suit : ℕ) (num_red_suits : ℕ)
  (deck_size := num_suits * cards_per_suit) 
  (num_red_cards := num_red_suits * cards_per_suit) 
  (num_black_suits := num_suits - num_red_suits) :
  num_suits = 5 →
  cards_per_suit = 13 →
  num_red_suits = 3 →
  num_black_suits = 2 →
  deck_size = 65 →
  num_red_cards = 39 →
  (Nat.choose deck_size 2) = 2080 →
  (Nat.choose num_red_cards 2) = 741 →
  ((Nat.choose num_red_cards 2) : ℚ / (Nat.choose deck_size 2) : ℚ) = 741 / 2080 :=
by 
  intros _ _ _ _ _ _ _ _ 
  sorry

end probability_two_red_cards_l162_162670


namespace quadratic_inequality_solution_range_l162_162528

theorem quadratic_inequality_solution_range (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0) ↔ (-3 / 2 < k ∧ k < 0) := sorry

end quadratic_inequality_solution_range_l162_162528


namespace third_pasture_cows_l162_162633

theorem third_pasture_cows (x y : ℝ) (H1 : x + 27 * y = 18) (H2 : 2 * x + 84 * y = 51) : 
  10 * x + 10 * 3 * y = 60 -> 60 / 3 = 20 :=
by
  sorry

end third_pasture_cows_l162_162633


namespace initial_speed_is_sixty_l162_162667

variable (D T : ℝ)

-- Condition: Two-thirds of the distance is covered in one-third of the total time.
def two_thirds_distance_in_one_third_time (V : ℝ) : Prop :=
  (2 * D / 3) / V = T / 3

-- Condition: The remaining distance is covered at 15 kmph.
def remaining_distance_at_fifteen_kmph : Prop :=
  (D / 3) / 15 = T - T / 3

-- Given that 30T = D from simplification in the solution.
def distance_time_relationship : Prop :=
  D = 30 * T

-- Prove that the initial speed V is 60 kmph.
theorem initial_speed_is_sixty (V : ℝ) (h1 : two_thirds_distance_in_one_third_time D T V) (h2 : remaining_distance_at_fifteen_kmph D T) (h3 : distance_time_relationship D T) : V = 60 := 
  sorry

end initial_speed_is_sixty_l162_162667


namespace milk_original_price_l162_162983

theorem milk_original_price 
  (budget : ℝ) (cost_celery : ℝ) (cost_cereal_discounted : ℝ) 
  (cost_bread : ℝ) (cost_potato : ℝ) (num_potatoes : ℝ) 
  (remainder : ℝ) (discount_milk : ℝ) (total_spent : ℝ) (cost_milk_after_discount : ℝ)
  (total_cost_discounted_items : ℝ) :
  budget = 60 →
  cost_celery = 5 →
  cost_cereal_discounted = 12 * 0.5 →
  cost_bread = 8 →
  cost_potato = 1 →
  num_potatoes = 6 →
  remainder = 26 →
  discount_milk = 0.1 →
  total_cost_discounted_items = cost_celery + cost_cereal_discounted + cost_bread + num_potatoes * cost_potato →
  total_spent = budget - remainder →
  cost_milk_after_discount = total_spent - total_cost_discounted_items →
  total_spent = 34 →
  cost_milk_after_discount = 9 →
  10 = cost_milk_after_discount / (1 - discount_milk) := 
by 
  intros,
  sorry

end milk_original_price_l162_162983


namespace spring_summer_work_hours_l162_162861

def john_works_spring_summer : Prop :=
  ∀ (work_hours_winter_week : ℕ) (weeks_winter : ℕ) (earnings_winter : ℕ)
    (weeks_spring_summer : ℕ) (earnings_spring_summer : ℕ) (hourly_rate : ℕ),
    work_hours_winter_week = 40 →
    weeks_winter = 8 →
    earnings_winter = 3200 →
    weeks_spring_summer = 24 →
    earnings_spring_summer = 4800 →
    hourly_rate = earnings_winter / (work_hours_winter_week * weeks_winter) →
    (earnings_spring_summer / hourly_rate) / weeks_spring_summer = 20

theorem spring_summer_work_hours : john_works_spring_summer :=
  sorry

end spring_summer_work_hours_l162_162861


namespace sqrt_range_l162_162689

theorem sqrt_range (a : ℝ) : 2 * a - 1 ≥ 0 ↔ a ≥ 1 / 2 :=
by sorry

end sqrt_range_l162_162689


namespace sum_distinct_prime_factors_462_l162_162159

theorem sum_distinct_prime_factors_462 : 
  ∏ x in {2, 3, 7, 11}, x = 462 → (∑ x in {2, 3, 7, 11}, x) = 23 :=
by
  intro h
  -- Proof goes here
  sorry

end sum_distinct_prime_factors_462_l162_162159


namespace sin_6_phi_l162_162567

theorem sin_6_phi (φ : ℝ) (h : complex.exp (complex.I * φ) = (3 + complex.I * real.sqrt 8) / 5) :
  real.sin (6 * φ) = -396 * real.sqrt 2 / 15625 :=
by sorry

end sin_6_phi_l162_162567


namespace problem_statement_l162_162837

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem problem_statement :
  (∀ x : ℝ, f (x) = 0 → x = - Real.pi / 6) ∧ (∀ x : ℝ, f (x) = 4 * Real.cos (2 * x - Real.pi / 6)) := sorry

end problem_statement_l162_162837


namespace decimal_to_binary_equivalent_123_l162_162640

theorem decimal_to_binary_equivalent_123 :
  let n := 123
  let binary := 1111011
  nat.to_digits 2 n = to_digits 1 binary := by
  sorry

end decimal_to_binary_equivalent_123_l162_162640


namespace TJs_average_time_l162_162309

theorem TJs_average_time 
  (total_distance : ℝ) 
  (distance_half : ℝ)
  (time_first_half : ℝ) 
  (time_second_half : ℝ) 
  (H1 : total_distance = 10) 
  (H2 : distance_half = total_distance / 2) 
  (H3 : time_first_half = 20) 
  (H4 : time_second_half = 30) :
  (time_first_half + time_second_half) / total_distance = 5 :=
by
  sorry

end TJs_average_time_l162_162309


namespace price_of_70_cans_l162_162787

noncomputable def regular_price_per_can : ℝ := 0.55
noncomputable def discount_rate_case : ℝ := 0.25
noncomputable def bulk_discount_rate : ℝ := 0.10
noncomputable def cans_per_case : ℕ := 24
noncomputable def total_cans_purchased : ℕ := 70

theorem price_of_70_cans :
  let discounted_price_per_can := regular_price_per_can * (1 - discount_rate_case)
  let discounted_price_for_cases := 48 * discounted_price_per_can
  let bulk_discount := if 70 >= 3 * cans_per_case then discounted_price_for_cases * bulk_discount_rate else 0
  let final_price_for_cases := discounted_price_for_cases - bulk_discount
  let additional_cans := total_cans_purchased % cans_per_case
  let price_for_additional_cans := additional_cans * discounted_price_per_can
  final_price_for_cases + price_for_additional_cans = 26.895 :=
by sorry

end price_of_70_cans_l162_162787


namespace max_product_of_roots_l162_162232

noncomputable def max_prod_roots_m : ℝ :=
  let m := 4.5
  m

theorem max_product_of_roots (m : ℕ) (h : 36 - 8 * m ≥ 0) : m = max_prod_roots_m :=
  sorry

end max_product_of_roots_l162_162232


namespace solve_fraction_zero_l162_162275

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 16) / (4 - x) = 0) (h2 : 4 - x ≠ 0) : x = -4 :=
sorry

end solve_fraction_zero_l162_162275


namespace necessarily_negative_sum_l162_162989

theorem necessarily_negative_sum 
  (u v w : ℝ)
  (hu : -1 < u ∧ u < 0)
  (hv : 0 < v ∧ v < 1)
  (hw : -2 < w ∧ w < -1) :
  v + w < 0 :=
sorry

end necessarily_negative_sum_l162_162989


namespace parabola_vertex_l162_162950

theorem parabola_vertex (a b c : ℝ) :
  (∀ x, y = ax^2 + bx + c ↔ 
   y = a*((x+3)^2) + 4) ∧
   (∀ x y, (x, y) = ((1:ℝ), (2:ℝ))) →
   a + b + c = 3 := by
  sorry

end parabola_vertex_l162_162950


namespace proposition_holds_n_2019_l162_162237

theorem proposition_holds_n_2019 (P: ℕ → Prop) 
  (H1: ∀ k : ℕ, k > 0 → ¬ P (k + 1) → ¬ P k) 
  (H2: P 2018) : 
  P 2019 :=
by 
  sorry

end proposition_holds_n_2019_l162_162237


namespace sum_of_prime_factors_462_eq_23_l162_162161

theorem sum_of_prime_factors_462_eq_23 : ∑ p in {2, 3, 7, 11}, p = 23 := by
  sorry

end sum_of_prime_factors_462_eq_23_l162_162161


namespace even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l162_162690

theorem even_ngon_parallel_edges (n : ℕ) (h : n % 2 = 0) :
  ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

theorem odd_ngon_no_two_parallel_edges (n : ℕ) (h : n % 2 = 1) :
  ¬ ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

end even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l162_162690


namespace charles_whistles_l162_162604

theorem charles_whistles (S C : ℕ) (h1 : S = 45) (h2 : S = C + 32) : C = 13 := 
by
  sorry

end charles_whistles_l162_162604


namespace total_fish_l162_162776

-- Definitions based on the problem conditions
def will_catch_catfish : ℕ := 16
def will_catch_eels : ℕ := 10
def henry_trout_per_catfish : ℕ := 3
def fraction_to_return : ℚ := 1/2

-- Calculation of required quantities
def will_total_fish : ℕ := will_catch_catfish + will_catch_eels
def henry_target_trout : ℕ := henry_trout_per_catfish * will_catch_catfish
def henry_return_trout : ℚ := fraction_to_return * henry_target_trout
def henry_kept_trout : ℤ := henry_target_trout -  henry_return_trout.to_nat

-- Goal statement to prove
theorem total_fish (will_catch_catfish = 16) (will_catch_eels = 10) 
  (henry_trout_per_catfish = 3) (fraction_to_return = 1/2) :
  will_total_fish + henry_kept_trout = 50 :=
by
  sorry

end total_fish_l162_162776


namespace number_of_divisors_of_2310_l162_162228

theorem number_of_divisors_of_2310 : 
  let n := 2310 in 
  let prime_factors := [2, 3, 5, 7, 11] in
  ∃ k : ℕ, k = prime_factors.length ∧
  (∀ i, i < k → prime_factors.nth i = some 2 ∨ prime_factors.nth i = some 3 ∨ prime_factors.nth i = some 5 ∨ prime_factors.nth i = some 7 ∨ prime_factors.nth i = some 11) →
  (n.factorization.to_nat * 1).0 = 32 :=
begin
  sorry
end

end number_of_divisors_of_2310_l162_162228


namespace not_exists_set_of_9_numbers_min_elements_l162_162783

theorem not_exists_set_of_9_numbers (s : Finset ℕ) 
  (h_len : s.card = 9) 
  (h_median : ∑ x in (s.filter (λ x, x ≤ 2)), 1 ≤ 5) 
  (h_other : ∑ x in (s.filter (λ x, x ≤ 13)), 1 ≤ 4) 
  (h_avg : ∑ x in s = 63) :
  False := sorry

theorem min_elements (n : ℕ) (h_nat: n ≥ 5) :
  ∃ s : Finset ℕ, s.card = 2 * n + 1 ∧
                  ∑ x in (s.filter (λ x, x ≤ 2)), 1 = n + 1 ∧ 
                  ∑ x in (s.filter (λ x, x ≤ 13)), 1 = n ∧
                  ∑ x in s = 14 * n + 7 := sorry

end not_exists_set_of_9_numbers_min_elements_l162_162783


namespace volume_parallelepiped_l162_162824

noncomputable def volume_of_parallelepiped (m n p d : ℝ) : ℝ :=
  if h : m > 0 ∧ n > 0 ∧ p > 0 ∧ d > 0 then
    m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2)
  else 0

theorem volume_parallelepiped (m n p d : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hd : d > 0) :
  volume_of_parallelepiped m n p d = m * n * p * (d^3) / (m^2 + n^2 + p^2)^(3/2) := by
  sorry

end volume_parallelepiped_l162_162824


namespace initial_pigs_l162_162011

theorem initial_pigs (x : ℕ) (h1 : x + 22 = 86) : x = 64 :=
by
  sorry

end initial_pigs_l162_162011


namespace nutty_professor_mixture_weight_l162_162454

/-- The Nutty Professor's problem translated to Lean 4 -/
theorem nutty_professor_mixture_weight :
  let cashews_weight := 20
  let cashews_cost_per_pound := 6.75
  let brazil_nuts_cost_per_pound := 5.00
  let mixture_cost_per_pound := 5.70
  ∃ (brazil_nuts_weight : ℝ), cashews_weight * cashews_cost_per_pound + brazil_nuts_weight * brazil_nuts_cost_per_pound =
                             (cashews_weight + brazil_nuts_weight) * mixture_cost_per_pound ∧
                             (cashews_weight + brazil_nuts_weight = 50) := 
sorry

end nutty_professor_mixture_weight_l162_162454


namespace mult_469158_9999_l162_162786

theorem mult_469158_9999 : 469158 * 9999 = 4691176842 := 
by sorry

end mult_469158_9999_l162_162786


namespace fixed_point_l162_162752

noncomputable def fixed_point_function (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : (ℝ × ℝ) :=
  (1, a^(1 - (1 : ℝ)) + 5)

theorem fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : fixed_point_function a h₀ h₁ = (1, 6) :=
by 
  sorry

end fixed_point_l162_162752


namespace total_weight_is_correct_l162_162919

-- Define the variables
def envelope_weight : ℝ := 8.5
def additional_weight_per_envelope : ℝ := 2
def num_envelopes : ℝ := 880

-- Define the total weight calculation
def total_weight : ℝ := num_envelopes * (envelope_weight + additional_weight_per_envelope)

-- State the theorem to prove that the total weight is as expected
theorem total_weight_is_correct : total_weight = 9240 :=
by
  sorry

end total_weight_is_correct_l162_162919


namespace solve_negative_integer_sum_l162_162000

theorem solve_negative_integer_sum (N : ℤ) (h1 : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end solve_negative_integer_sum_l162_162000


namespace mala_usha_speed_ratio_l162_162435

noncomputable def drinking_speed_ratio (M U : ℝ) (tM tU : ℝ) (fracU : ℝ) (total_bottle : ℝ) : ℝ :=
  let U_speed := fracU * total_bottle / tU
  let M_speed := (total_bottle - fracU * total_bottle) / tM
  M_speed / U_speed

theorem mala_usha_speed_ratio :
  drinking_speed_ratio (3/50) (1/50) 10 20 (4/10) 1 = 3 :=
by
  sorry

end mala_usha_speed_ratio_l162_162435


namespace billy_total_problems_solved_l162_162674

theorem billy_total_problems_solved :
  ∃ (Q : ℕ), (3 * Q = 132) ∧ ((Q) + (2 * Q) + (3 * Q) = 264) :=
by
  sorry

end billy_total_problems_solved_l162_162674


namespace floor_sqrt_23_squared_eq_16_l162_162221

theorem floor_sqrt_23_squared_eq_16 :
  (Int.floor (Real.sqrt 23))^2 = 16 :=
by
  have h1 : 4 < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < 5 := sorry
  have floor_sqrt_23 : Int.floor (Real.sqrt 23) = 4 := sorry
  rw [floor_sqrt_23]
  norm_num

end floor_sqrt_23_squared_eq_16_l162_162221


namespace sum_of_areas_is_858_l162_162871

def length1 : ℕ := 1
def length2 : ℕ := 9
def length3 : ℕ := 25
def length4 : ℕ := 49
def length5 : ℕ := 81
def length6 : ℕ := 121

def base_width : ℕ := 3

def area (width : ℕ) (length : ℕ) : ℕ :=
  width * length

def total_area_of_rectangles : ℕ :=
  area base_width length1 +
  area base_width length2 +
  area base_width length3 +
  area base_width length4 +
  area base_width length5 +
  area base_width length6

theorem sum_of_areas_is_858 : total_area_of_rectangles = 858 := by
  sorry

end sum_of_areas_is_858_l162_162871


namespace max_a_plus_ab_plus_abc_l162_162727

noncomputable def f (a b c: ℝ) := a + a * b + a * b * c

theorem max_a_plus_ab_plus_abc (a b c : ℝ) (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h2 : a + b + c = 1) :
  ∃ x, (f a b c ≤ x) ∧ (∀ y, f a b c ≤ y → y = 1) :=
sorry

end max_a_plus_ab_plus_abc_l162_162727


namespace sum_of_distinct_prime_factors_of_462_l162_162156

theorem sum_of_distinct_prime_factors_of_462 : 
  ∑ p in (Nat.factors 462).toFinset, p = 23 := by
  sorry

end sum_of_distinct_prime_factors_of_462_l162_162156


namespace solve_system_l162_162125

theorem solve_system 
  (x y z : ℝ)
  (h1 : x + 2 * y = 10)
  (h2 : y = 3)
  (h3 : x - 3 * y + z = 7) :
  x = 4 ∧ y = 3 ∧ z = 12 :=
by
  sorry

end solve_system_l162_162125


namespace ninth_term_correct_l162_162629

noncomputable def ninth_term_of_arithmetic_sequence
  (third_term fifteenth_term : ℚ) : ℚ :=
  (third_term + fifteenth_term) / 2

theorem ninth_term_correct :
  let third_term := (3 : ℚ) / 8
      fifteenth_term := (7 : ℚ) / 9
  in ninth_term_of_arithmetic_sequence third_term fifteenth_term = (83 : ℚ) / 144 :=
by
let third_term := (3 : ℚ) / 8
let fifteenth_term := (7 : ℚ) / 9
simp [ninth_term_of_arithmetic_sequence, third_term, fifteenth_term]
sorry

end ninth_term_correct_l162_162629


namespace hexagon_internal_angle_A_l162_162579

theorem hexagon_internal_angle_A
  (B C D E F : ℝ) 
  (hB : B = 134) 
  (hC : C = 98) 
  (hD : D = 120) 
  (hE : E = 139) 
  (hF : F = 109) 
  (H : B + C + D + E + F + A = 720) : A = 120 := 
sorry

end hexagon_internal_angle_A_l162_162579


namespace possible_ratio_in_interval_l162_162122

theorem possible_ratio_in_interval (n : ℕ) (h : n ≥ 3) :
  ∃ (s t : ℕ), s > 0 ∧ t > 0 ∧ (1 : ℚ) ≤ (t : ℚ) / s ∧ (t : ℚ) / s < n - 1 :=
sorry

end possible_ratio_in_interval_l162_162122


namespace eval_expr_l162_162055

-- Define the expression
def expr : ℚ := 2 + 3 / (2 + 1 / (2 + 1 / 2))

-- The theorem to prove the evaluation of the expression
theorem eval_expr : expr = 13 / 4 :=
by
  sorry

end eval_expr_l162_162055


namespace inequality_proof_l162_162691

theorem inequality_proof (x y z : ℝ) (h : x * y * z + x + y + z = 4) : 
    (y * z + 6)^2 + (z * x + 6)^2 + (x * y + 6)^2 ≥ 8 * (x * y * z + 5) :=
by
  sorry

end inequality_proof_l162_162691


namespace combination_30_2_l162_162354

theorem combination_30_2 : Nat.choose 30 2 = 435 := by
  sorry

end combination_30_2_l162_162354


namespace fraction_of_girls_l162_162855

variable (T G B : ℕ) -- The total number of students, number of girls, and number of boys
variable (x : ℚ) -- The fraction of the number of girls

-- Definitions based on the given conditions
def fraction_condition : Prop := x * G = (1/6) * T
def ratio_condition : Prop := (B : ℚ) / (G : ℚ) = 2
def total_students : Prop := T = B + G

-- The statement we need to prove
theorem fraction_of_girls (h1 : fraction_condition T G x)
                          (h2 : ratio_condition B G)
                          (h3 : total_students T G B):
  x = 1/2 :=
by
  sorry

end fraction_of_girls_l162_162855


namespace algebraic_expression_value_l162_162273

variables {m n : ℝ}

theorem algebraic_expression_value (h : n = 3 - 5 * m) : 10 * m + 2 * n - 3 = 3 :=
by sorry

end algebraic_expression_value_l162_162273


namespace total_pieces_of_junk_mail_l162_162666

-- Definition of the problem based on given conditions
def pieces_per_house : ℕ := 4
def number_of_blocks : ℕ := 16
def houses_per_block : ℕ := 17

-- Statement of the theorem to prove the total number of pieces of junk mail
theorem total_pieces_of_junk_mail :
  (houses_per_block * pieces_per_house * number_of_blocks) = 1088 :=
by
  sorry

end total_pieces_of_junk_mail_l162_162666


namespace balls_per_bag_l162_162512

theorem balls_per_bag (total_balls bags_used: Nat) (h1: total_balls = 36) (h2: bags_used = 9) : total_balls / bags_used = 4 := by
  sorry

end balls_per_bag_l162_162512


namespace num_possible_radii_l162_162370

theorem num_possible_radii :
  ∃ S : Finset ℕ, S.card = 11 ∧ ∀ r ∈ S, (∃ k : ℕ, 150 = k * r) ∧ r ≠ 150 :=
by
  sorry

end num_possible_radii_l162_162370


namespace slope_of_line_l162_162239

-- Defining the conditions
def intersects_on_line (s x y : ℝ) : Prop :=
  (2 * x + 3 * y = 8 * s + 6) ∧ (x + 2 * y = 5 * s - 1)

-- Theorem stating that the slope of the line on which all intersections lie is 2
theorem slope_of_line {s x y : ℝ} :
  (∃ s x y, intersects_on_line s x y) → (∃ (m : ℝ), m = 2) :=
by sorry

end slope_of_line_l162_162239


namespace probability_of_three_5s_in_eight_rolls_l162_162797

-- Conditions
def total_outcomes : ℕ := 6 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3

-- The probability that the number 5 appears exactly three times in eight rolls of a fair die
theorem probability_of_three_5s_in_eight_rolls :
  (favorable_outcomes / total_outcomes : ℚ) = (56 / 1679616 : ℚ) :=
by
  sorry

end probability_of_three_5s_in_eight_rolls_l162_162797


namespace probability_not_below_x_axis_half_l162_162869

-- Define the vertices of the parallelogram
def P : (ℝ × ℝ) := (4, 4)
def Q : (ℝ × ℝ) := (-2, -2)
def R : (ℝ × ℝ) := (-8, -2)
def S : (ℝ × ℝ) := (-2, 4)

-- Define a predicate for points within the parallelogram
def in_parallelogram (A B C D : ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

-- Define the area function
def area_of_parallelogram (A B C D : ℝ × ℝ) : ℝ := sorry

noncomputable def probability_not_below_x_axis (A B C D : ℝ × ℝ) : ℝ :=
  let total_area := area_of_parallelogram A B C D
  let area_above_x_axis := area_of_parallelogram (0, 0) D A (0, 0) / 2
  area_above_x_axis / total_area

theorem probability_not_below_x_axis_half :
  probability_not_below_x_axis P Q R S = 1 / 2 :=
sorry

end probability_not_below_x_axis_half_l162_162869


namespace find_y_of_arithmetic_mean_l162_162249

theorem find_y_of_arithmetic_mean (y : ℝ) (h: (7 + 12 + 19 + 8 + 10 + y) / 6 = 15) : y = 34 :=
by {
  -- Skipping the proof
  sorry
}

end find_y_of_arithmetic_mean_l162_162249


namespace one_intersection_point_two_intersection_points_l162_162069

variables (k : ℝ)

-- Condition definitions
def parabola_eq (y x : ℝ) : Prop := y^2 = -4 * x
def line_eq (x y k : ℝ) : Prop := y + 1 = k * (x - 2)
def discriminant_non_negative (a b c : ℝ) : Prop := b^2 - 4 * a * c ≥ 0

-- Mathematically equivalent proof problem 1
theorem one_intersection_point (k : ℝ) : 
  (k = 1/2 ∨ k = -1 ∨ k = 0) → 
  ∃ x y : ℝ, parabola_eq y x ∧ line_eq x y k := sorry

-- Mathematically equivalent proof problem 2
theorem two_intersection_points (k : ℝ) : 
  (-1 < k ∧ k < 1/2 ∧ k ≠ 0) → 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
  (x₁ ≠ x₂ ∧ y₁ ≠ y₂) ∧ parabola_eq y₁ x₁ ∧ parabola_eq y₂ x₂ ∧ 
  line_eq x₁ y₁ k ∧ line_eq x₂ y₂ k := sorry

end one_intersection_point_two_intersection_points_l162_162069


namespace part_I_part_II_l162_162255

-- Definition of the sequence a_n with given conditions
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1 else (n^2 + n) / 2

-- Define the sum of the first n terms S_n
def S_n (n : ℕ) : ℕ :=
  (n + 2) / 3 * a_n n

-- Define the sequence b_n in terms of a_n
def b_n (n : ℕ) : ℚ := 1 / a_n n

-- Define the sum of the first n terms of b_n
def T_n (n : ℕ) : ℚ :=
  2 * (1 - 1 / (n + 1))

-- Theorem statement for part (I)
theorem part_I (n : ℕ) : 
  a_n 2 = 3 ∧ a_n 3 = 6 ∧ (∀ (n : ℕ), n ≥ 2 → a_n n = (n^2 + n) / 2) := sorry

-- Theorem statement for part (II)
theorem part_II (n : ℕ) : 
  T_n n = 2 * (1 - 1 / (n + 1)) := sorry

end part_I_part_II_l162_162255


namespace train_length_is_500_l162_162039

def train_speed_km_per_hr : ℝ := 63
def man_speed_km_per_hr : ℝ := 3
def crossing_time_s : ℝ := 29.997600191984642
def relative_speed_km_per_hr : ℝ := train_speed_km_per_hr - man_speed_km_per_hr
def relative_speed_m_per_s : ℝ := (relative_speed_km_per_hr * 1000) / 3600
def train_length : ℝ := relative_speed_m_per_s * crossing_time_s

theorem train_length_is_500 :
  train_length = 500 := sorry

end train_length_is_500_l162_162039


namespace dennis_initial_money_l162_162361

theorem dennis_initial_money :
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  cost_of_shirts + total_change = 50 :=
by
  let cost_of_shirts := 27
  let change_bills := 2 * 10
  let change_coins := 3
  let total_change := change_bills + change_coins
  show cost_of_shirts + total_change = 50
  sorry

end dennis_initial_money_l162_162361


namespace candy_from_sister_is_5_l162_162542

noncomputable def candy_received_from_sister (candy_from_neighbors : ℝ) (pieces_per_day : ℝ) (days : ℕ) : ℝ :=
  pieces_per_day * days - candy_from_neighbors

theorem candy_from_sister_is_5 :
  candy_received_from_sister 11.0 8.0 2 = 5.0 :=
by
  sorry

end candy_from_sister_is_5_l162_162542


namespace largest_integer_with_square_three_digits_base_7_l162_162978

theorem largest_integer_with_square_three_digits_base_7 : 
  ∃ M : ℕ, (7^2 ≤ M^2 ∧ M^2 < 7^3) ∧ ∀ n : ℕ, (7^2 ≤ n^2 ∧ n^2 < 7^3) → n ≤ M := 
sorry

end largest_integer_with_square_three_digits_base_7_l162_162978


namespace div_scaled_result_l162_162102

theorem div_scaled_result :
  (2994 : ℝ) / 14.5 = 171 :=
by
  have cond1 : (29.94 : ℝ) / 1.45 = 17.1 := sorry
  have cond2 : (2994 : ℝ) = 100 * 29.94 := sorry
  have cond3 : (14.5 : ℝ) = 10 * 1.45 := sorry
  sorry

end div_scaled_result_l162_162102


namespace base_2_representation_of_123_l162_162643

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l162_162643


namespace greatest_QPN_value_l162_162653

theorem greatest_QPN_value (N : ℕ) (Q P : ℕ) (QPN : ℕ) :
  (NN : ℕ) =
  10 * N + N ∧
  QPN = 100 * Q + 10 * P + N ∧
  N < 10 ∧ N ≥ 1 ∧
  NN * N = QPN ∧
  NN >= 10 ∧ NN < 100  -- Ensuring NN is a two-digit number
  → QPN <= 396 := sorry

end greatest_QPN_value_l162_162653


namespace side_length_of_smaller_square_l162_162669

theorem side_length_of_smaller_square (s : ℝ) (A1 A2 : ℝ) (h1 : 5 * 5 = A1 + A2) (h2 : 2 * A2 = A1 + 25)  : s = 5 * Real.sqrt 3 / 3 :=
by
  sorry

end side_length_of_smaller_square_l162_162669


namespace exists_three_naturals_sum_perfect_square_no_four_naturals_sum_perfect_square_l162_162030

-- Definition for the condition that ab + 10 is a perfect square
def is_perfect_square_sum (a b : ℕ) : Prop := ∃ k : ℕ, a * b + 10 = k * k

-- Problem: Existence of three different natural numbers for which the sum of the product of any two with 10 is a perfect square
theorem exists_three_naturals_sum_perfect_square :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_perfect_square_sum a b ∧ is_perfect_square_sum b c ∧ is_perfect_square_sum c a := sorry

-- Problem: Non-existence of four different natural numbers for which the sum of the product of any two with 10 is a perfect square
theorem no_four_naturals_sum_perfect_square :
  ¬ ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d ∧
    is_perfect_square_sum a b ∧ is_perfect_square_sum a c ∧ is_perfect_square_sum a d ∧
    is_perfect_square_sum b c ∧ is_perfect_square_sum b d ∧ is_perfect_square_sum c d := sorry

end exists_three_naturals_sum_perfect_square_no_four_naturals_sum_perfect_square_l162_162030


namespace problem1_problem2_l162_162865

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 2 * a) * Real.log (x + 1) - 2 * x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x^2 - x * Real.log (x + 1)

theorem problem1 (x : ℝ) : (x + 2) * Real.log (x + 1) - 2 * x
  is_monotonically_increasing_on Ioo (-1:ℝ) ∞ ∧
  (∀ x, (x + 2) * Real.log (x + 1) - 2 * x = 0 ↔ x = 0) := 
sorry

theorem problem2 (x1 x2 x3 y1 y2 y3 : ℝ) (hx : x1 + x2 = 2 * x3) :
  ∃ (a : ℝ), ∀ (x : ℝ), x ∈ (Ioo 0 ∞) →
    let k := (y2 - y1) / (x2 - x1),
        gx := ∂ (g a) / ∂ x ↔ g' (x2) = k in
    (a = 0) := sorry

end problem1_problem2_l162_162865


namespace harry_speed_on_friday_l162_162709

theorem harry_speed_on_friday :
  ∀ (speed_monday speed_tuesday_to_thursday speed_friday : ℝ)
  (ran_50_percent_faster ran_60_percent_faster: ℝ),
  speed_monday = 10 →
  ran_50_percent_faster = 0.50 →
  ran_60_percent_faster = 0.60 →
  speed_tuesday_to_thursday = speed_monday + (ran_50_percent_faster * speed_monday) →
  speed_friday = speed_tuesday_to_thursday + (ran_60_percent_faster * speed_tuesday_to_thursday) →
  speed_friday = 24 := by {
  intros speed_monday speed_tuesday_to_thursday speed_friday ran_50_percent_faster ran_60_percent_faster,
  intros h0 h1 h2 h3 h4,
  rw [h0, h1, h2, h3, h4],
  norm_num,
}

end harry_speed_on_friday_l162_162709


namespace part1_max_value_part2_range_of_a_l162_162070

-- Definition of f(x) for general a
def f (a x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Part 1: Prove the maximum value of f(x) when a = 0
theorem part1_max_value (x : ℝ) (hx : x > 0) : f 0 x ≤ -1 := by
  sorry

-- Part 2: Prove the range of a such that f(x) has exactly one zero
theorem part2_range_of_a (a : ℝ) : (∃! x : ℝ, x > 0 ∧ f a x = 0) ↔ a ∈ Set.Ioi 0 := by
  sorry

end part1_max_value_part2_range_of_a_l162_162070


namespace largest_hexagon_angle_l162_162615

-- We define the conditions first
def angle_ratios (x : ℝ) := [3*x, 3*x, 3*x, 4*x, 5*x, 6*x]
def sum_of_angles (angles : List ℝ) := angles.sum = 720

-- Now we state our proof goal
theorem largest_hexagon_angle :
  ∀ (x : ℝ), sum_of_angles (angle_ratios x) → 6 * x = 180 :=
by
  intro x
  intro h
  sorry

end largest_hexagon_angle_l162_162615


namespace fraction_of_total_amount_l162_162654

-- Conditions
variable (p q r : ℕ)
variable (total_amount amount_r : ℕ)
variable (total_amount_eq : total_amount = 6000)
variable (amount_r_eq : amount_r = 2400)

-- Mathematical statement
theorem fraction_of_total_amount :
  amount_r / total_amount = 2 / 5 :=
by
  -- Sorry to skip the proof, as instructed
  sorry

end fraction_of_total_amount_l162_162654


namespace robert_total_balls_l162_162306

-- Define the conditions
def robert_initial_balls : ℕ := 25
def tim_balls : ℕ := 40

-- Mathematically equivalent proof problem
theorem robert_total_balls : 
  robert_initial_balls + (tim_balls / 2) = 45 := by
  sorry

end robert_total_balls_l162_162306


namespace mary_change_l162_162868

/-- 
Calculate the change Mary will receive after buying tickets for herself and her 3 children 
at the circus, given the ticket prices and special group rate discount.
-/
theorem mary_change :
  let adult_ticket := 2
  let child_ticket := 1
  let discounted_child_ticket := 0.5 * child_ticket
  let total_cost_with_discount := adult_ticket + 2 * child_ticket + discounted_child_ticket
  let payment := 20
  payment - total_cost_with_discount = 15.50 :=
by
  sorry

end mary_change_l162_162868


namespace police_emergency_number_has_prime_divisor_gt_seven_l162_162804

theorem police_emergency_number_has_prime_divisor_gt_seven (k : ℤ) :
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ (1000 * k + 133) :=
by
  sorry

end police_emergency_number_has_prime_divisor_gt_seven_l162_162804


namespace percentage_of_360_l162_162652

theorem percentage_of_360 (percentage : ℝ) : 
  (percentage / 100) * 360 = 93.6 → percentage = 26 := 
by
  intro h
  -- proof missing
  sorry

end percentage_of_360_l162_162652


namespace cylinder_surface_area_l162_162717

theorem cylinder_surface_area
  (l : ℝ) (r : ℝ) (unfolded_square_side : ℝ) (base_circumference : ℝ)
  (hl : unfolded_square_side = 2 * π)
  (hl_gen : l = 2 * π)
  (hc : base_circumference = 2 * π)
  (hr : r = 1) :
  2 * π * r * (r + l) = 2 * π + 4 * π^2 :=
by
  sorry

end cylinder_surface_area_l162_162717


namespace sin_pi_minus_2alpha_l162_162565

theorem sin_pi_minus_2alpha (α : ℝ) (h1 : Real.sin (π / 2 + α) = -3 / 5) (h2 : π / 2 < α ∧ α < π) : 
  Real.sin (π - 2 * α) = -24 / 25 := by
  sorry

end sin_pi_minus_2alpha_l162_162565


namespace largest_subset_no_multiples_l162_162015

theorem largest_subset_no_multiples : ∀ (S : Finset ℕ), (S = Finset.range 101) → 
  ∃ (A : Finset ℕ), A ⊆ S ∧ (∀ x ∈ A, ∀ y ∈ A, x ≠ y → ¬(x ∣ y) ∧ ¬(y ∣ x)) ∧ A.card = 50 :=
by
  sorry

end largest_subset_no_multiples_l162_162015


namespace star_operation_l162_162185

def star (a b : ℚ) : ℚ := 2 * a - b + 1

theorem star_operation :
  star 1 (star 2 (-3)) = -5 :=
by
  -- Calcualtion follows the steps given in the solution, 
  -- but this line is here just to satisfy the 'rewrite the problem' instruction.
  sorry

end star_operation_l162_162185


namespace speed_of_stream_l162_162022

theorem speed_of_stream
  (v_a v_s : ℝ)
  (h1 : v_a - v_s = 4)
  (h2 : v_a + v_s = 6) :
  v_s = 1 :=
by {
  sorry
}

end speed_of_stream_l162_162022


namespace number_of_assignment_methods_l162_162765

theorem number_of_assignment_methods {teachers questions : ℕ} (h_teachers : teachers = 5) (h_questions : questions = 3) (h_at_least_one_teacher : ∀ q, q ∈ finset.range questions → ∃ t, t ∈ finset.range teachers) : 
  ∃ n, n = 150 :=
by 
  sorry

end number_of_assignment_methods_l162_162765


namespace floor_sq_minus_sq_floor_l162_162937

noncomputable def floor_13_2 : ℤ := int.floor 13.2

theorem floor_sq_minus_sq_floor :
  int.floor ((13.2 : ℝ) ^ 2) - (floor_13_2 * floor_13_2) = 5 :=
by
  let floor_13_2_sq := floor_13_2 * floor_13_2
  have h1 : int.floor (13.2 : ℝ) = 13 := by norm_num
  have h2 : int.floor ((13.2 : ℝ) ^ 2) = 174 := by norm_num
  rw [h1] at floor_13_2
  rw [h2]
  rw [floor_13_2, floor_13_2, floor_13_2]
  exact sorry

end floor_sq_minus_sq_floor_l162_162937


namespace perfect_square_trinomial_l162_162534

theorem perfect_square_trinomial (x : ℝ) : (x + 9)^2 = x^2 + 18 * x + 81 := by
  sorry

end perfect_square_trinomial_l162_162534


namespace noncongruent_triangles_count_l162_162257

/-- Prove that the number of noncongruent integer-sided triangles 
with positive area and perimeter less than 20, 
which are neither equilateral, isosceles, nor right triangles, is 15. -/
theorem noncongruent_triangles_count : 
  ∃ n : ℕ, 
  (∀ (a b c : ℕ) (h : a ≤ b ∧ b ≤ c),
    a + b + c < 20 ∧ a + b > c ∧ a^2 + b^2 ≠ c^2 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → n ≥ 15) :=
sorry

end noncongruent_triangles_count_l162_162257


namespace bob_wins_even_n_l162_162479

def game_of_islands (n : ℕ) (even_n : n % 2 = 0) : Prop :=
  ∃ strategy : (ℕ → ℕ), -- strategy is a function representing each player's move
    ∀ A B : ℕ → ℕ, -- A and B represent the moves of Alice and Bob respectively
    (A 0 + B 1) = n → (A (A 0 + 1) ≠ B (A 0 + 1)) -- Bob can always mirror Alice’s move.

theorem bob_wins_even_n (n : ℕ) (h : n % 2 = 0) : game_of_islands n h :=
sorry

end bob_wins_even_n_l162_162479


namespace sum_of_coordinates_of_other_endpoint_l162_162618

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (midpoint_cond : (x + 1) / 2 = 3)
  (midpoint_cond2 : (y - 3) / 2 = 5) :
  x + y = 18 :=
sorry

end sum_of_coordinates_of_other_endpoint_l162_162618


namespace max_value_f_when_a_zero_range_a_for_single_zero_l162_162074

noncomputable def f (a x : ℝ) : ℝ := a * x - (1 / x) - (a + 1) * Real.log x

theorem max_value_f_when_a_zero :
  (∀ x > (0 : ℝ), f 0 x ≤ -1) ∧ (∃ x > (0 : ℝ), f 0 x = -1) :=
sorry

theorem range_a_for_single_zero :
  (∀ a : ℝ, (∃ x > (0 : ℝ), f a x = 0) ↔ (0 < a ∧ a < ∞)) :=
sorry

end max_value_f_when_a_zero_range_a_for_single_zero_l162_162074


namespace transaction_result_l162_162182

theorem transaction_result
  (house_selling_price store_selling_price : ℝ)
  (house_loss_perc : ℝ)
  (store_gain_perc : ℝ)
  (house_selling_price_eq : house_selling_price = 15000)
  (store_selling_price_eq : store_selling_price = 15000)
  (house_loss_perc_eq : house_loss_perc = 0.1)
  (store_gain_perc_eq : store_gain_perc = 0.3) :
  (store_selling_price + house_selling_price - ((house_selling_price / (1 - house_loss_perc)) + (store_selling_price / (1 + store_gain_perc)))) = 1795 :=
by
  sorry

end transaction_result_l162_162182


namespace ironman_age_greater_than_16_l162_162764

variable (Ironman_age : ℕ)
variable (Thor_age : ℕ := 1456)
variable (CaptainAmerica_age : ℕ := Thor_age / 13)
variable (PeterParker_age : ℕ := CaptainAmerica_age / 7)

theorem ironman_age_greater_than_16
  (Thor_13_times_CaptainAmerica : Thor_age = 13 * CaptainAmerica_age)
  (CaptainAmerica_7_times_PeterParker : CaptainAmerica_age = 7 * PeterParker_age)
  (Thor_age_given : Thor_age = 1456) :
  Ironman_age > 16 :=
by
  sorry

end ironman_age_greater_than_16_l162_162764


namespace floor_difference_l162_162947

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ - ⌊x⌋ * ⌊x⌋) = 5 := by
  have h1 : ⌊x⌋ = 13 := by sorry
  have h2 : ⌊x^2⌋ = 174 := by sorry
  calc
  ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 174 - 13 * 13 : by rw [h1, h2]
                 ... = 5 : by norm_num

end floor_difference_l162_162947


namespace percentage_parents_agree_l162_162923

def total_parents : ℕ := 800
def disagree_parents : ℕ := 640

theorem percentage_parents_agree : 
  ((total_parents - disagree_parents) / total_parents : ℚ) * 100 = 20 := 
by 
  sorry

end percentage_parents_agree_l162_162923


namespace revenue_fall_percentage_l162_162876

theorem revenue_fall_percentage:
  let oldRevenue := 72.0
  let newRevenue := 48.0
  (oldRevenue - newRevenue) / oldRevenue * 100 = 33.33 :=
by
  let oldRevenue := 72.0
  let newRevenue := 48.0
  sorry

end revenue_fall_percentage_l162_162876


namespace car_speed_l162_162508

-- Definitions from conditions
def distance : ℝ := 360
def time : ℝ := 4.5

-- Statement to prove
theorem car_speed : (distance / time) = 80 := by
  sorry

end car_speed_l162_162508


namespace number_of_groups_eq_five_l162_162472

-- Define conditions
def total_eggs : ℕ := 35
def eggs_per_group : ℕ := 7

-- Statement to prove the number of groups
theorem number_of_groups_eq_five : total_eggs / eggs_per_group = 5 := by
  sorry

end number_of_groups_eq_five_l162_162472


namespace time_after_midnight_1453_minutes_l162_162338

def minutes_to_time (minutes : Nat) : Nat × Nat :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_of_day (hours : Nat) : Nat × Nat :=
  let days := hours / 24
  let remaining_hours := hours % 24
  (days, remaining_hours)

theorem time_after_midnight_1453_minutes : 
  let midnight := (0, 0) -- Midnight as a tuple of hours and minutes
  let total_minutes := 1453
  let (total_hours, minutes) := minutes_to_time total_minutes
  let (days, hours) := time_of_day total_hours
  days = 1 ∧ hours = 0 ∧ minutes = 13
  := by
    let midnight := (0, 0)
    let total_minutes := 1453
    let (total_hours, minutes) := minutes_to_time total_minutes
    let (days, hours) := time_of_day total_hours
    sorry

end time_after_midnight_1453_minutes_l162_162338


namespace missing_digit_in_decimal_representation_of_power_of_two_l162_162318

theorem missing_digit_in_decimal_representation_of_power_of_two :
  (∃ m : ℕ, m < 10 ∧
   ∀ (n : ℕ), (0 ≤ n ∧ n < 10 → n ≠ m) →
     (45 - m) % 9 = (2^29) % 9) :=
sorry

end missing_digit_in_decimal_representation_of_power_of_two_l162_162318


namespace Lisa_quiz_goal_l162_162521

theorem Lisa_quiz_goal (total_quizzes : ℕ) (required_percentage : ℝ) (a_scored : ℕ) (completed_quizzes : ℕ) : 
  total_quizzes = 60 → 
  required_percentage = 0.75 → 
  a_scored = 30 → 
  completed_quizzes = 40 → 
  ∃ lower_than_a_quizzes : ℕ, lower_than_a_quizzes = 5 :=
by
  intros total_quizzes_eq req_percent_eq a_scored_eq completed_quizzes_eq
  sorry

end Lisa_quiz_goal_l162_162521


namespace probability_correct_l162_162106
noncomputable def probability_no_2_in_id : ℚ :=
  let total_ids := 5000
  let valid_ids := 2916
  valid_ids / total_ids

theorem probability_correct : probability_no_2_in_id = 729 / 1250 := by
  sorry

end probability_correct_l162_162106


namespace jerry_daughters_games_l162_162582

theorem jerry_daughters_games (x y : ℕ) (h : 4 * x + 2 * x + 4 * y + 2 * y = 96) (hx : x = y) :
  x = 8 ∧ y = 8 :=
by
  have h1 : 6 * x + 6 * y = 96 := by linarith
  have h2 : x = y := hx
  sorry

end jerry_daughters_games_l162_162582


namespace red_car_count_l162_162901

-- Define the ratio and the given number of black cars
def ratio_red_to_black (R B : ℕ) : Prop := R * 8 = B * 3

-- Define the given number of black cars
def black_cars : ℕ := 75

-- State the theorem we want to prove
theorem red_car_count : ∃ R : ℕ, ratio_red_to_black R black_cars ∧ R = 28 :=
by
  sorry

end red_car_count_l162_162901


namespace balls_in_boxes_l162_162711

theorem balls_in_boxes :
  let balls := 5
  let boxes := 4
  boxes ^ balls = 1024 :=
by
  sorry

end balls_in_boxes_l162_162711


namespace Cody_games_l162_162818

/-- Cody had nine old video games he wanted to get rid of.
He decided to give four of the games to his friend Jake,
three games to his friend Sarah, and one game to his friend Luke.
On Saturday he bought five new games.
How many games does Cody have now? -/
theorem Cody_games (nine_games initially: ℕ) (jake_games: ℕ) (sarah_games: ℕ) (luke_games: ℕ) (saturday_games: ℕ)
  (h_initial: initially = 9)
  (h_jake: jake_games = 4)
  (h_sarah: sarah_games = 3)
  (h_luke: luke_games = 1)
  (h_saturday: saturday_games = 5) :
  ((initially - (jake_games + sarah_games + luke_games)) + saturday_games) = 6 :=
by
  sorry

end Cody_games_l162_162818


namespace least_subtraction_for_divisibility_l162_162168

/-- 
  Theorem: The least number that must be subtracted from 9857621 so that 
  the result is divisible by 17 is 8.
-/
theorem least_subtraction_for_divisibility :
  ∃ k : ℕ, 9857621 % 17 = k ∧ k = 8 :=
by
  sorry

end least_subtraction_for_divisibility_l162_162168


namespace temperature_difference_l162_162442

theorem temperature_difference (t_low t_high : ℝ) (h_low : t_low = -2) (h_high : t_high = 5) :
  t_high - t_low = 7 :=
by
  rw [h_low, h_high]
  norm_num

end temperature_difference_l162_162442


namespace option_C_is_quadratic_l162_162775

-- Definitions based on conditions
def option_A (x : ℝ) : Prop := x^2 + (1/x^2) = 0
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (x - 1) * (x + 2) = 1
def option_D (x y : ℝ) : Prop := 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- Statement to prove option C is a quadratic equation in one variable.
theorem option_C_is_quadratic : ∀ x : ℝ, (option_C x) → (∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0) :=
by
  intros x hx
  -- To be proven
  sorry

end option_C_is_quadratic_l162_162775


namespace niko_percentage_profit_l162_162129

theorem niko_percentage_profit
    (pairs_sold : ℕ)
    (cost_per_pair : ℕ)
    (profit_5_pairs : ℕ)
    (total_profit : ℕ)
    (num_pairs_remaining : ℕ)
    (cost_remaining_pairs : ℕ)
    (profit_remaining_pairs : ℕ)
    (percentage_profit : ℕ)
    (cost_5_pairs : ℕ):
    pairs_sold = 9 →
    cost_per_pair = 2 →
    profit_5_pairs = 1 →
    total_profit = 3 →
    num_pairs_remaining = 4 →
    cost_remaining_pairs = 8 →
    profit_remaining_pairs = 2 →
    percentage_profit = 25 →
    cost_5_pairs = 10 →
    (profit_remaining_pairs * 100 / cost_remaining_pairs) = percentage_profit :=
by
    intros
    sorry

end niko_percentage_profit_l162_162129


namespace infinite_series_sum_l162_162045

theorem infinite_series_sum :
  ∑' n : ℕ, n / (8 : ℝ) ^ n = (8 / 49 : ℝ) :=
sorry

end infinite_series_sum_l162_162045


namespace problem1_problem2_l162_162174

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := 
by
  sorry

theorem problem2 (a b : ℝ) (h1 : abs a < 1) (h2 : abs b < 1) : abs (1 - a * b) > abs (a - b) := 
by
  sorry

end problem1_problem2_l162_162174


namespace k_is_perfect_square_l162_162433

theorem k_is_perfect_square (m n : ℤ) (hm : m > 0) (hn : n > 0) (k := ((m + n) ^ 2) / (4 * m * (m - n) ^ 2 + 4)) :
  ∃ (a : ℤ), k = a ^ 2 := by
  sorry

end k_is_perfect_square_l162_162433


namespace coloring_ways_l162_162687

-- Definitions for colors
inductive Color
| red
| green

open Color

-- Definition of the coloring function
def color (n : ℕ) : Color := sorry

-- Conditions:
-- 1. Each positive integer is colored either red or green
def condition1 (n : ℕ) : n > 0 → (color n = red ∨ color n = green) := sorry

-- 2. The sum of any two different red numbers is a red number
def condition2 (r1 r2 : ℕ) : r1 ≠ r2 → color r1 = red → color r2 = red → color (r1 + r2) = red := sorry

-- 3. The sum of any two different green numbers is a green number
def condition3 (g1 g2 : ℕ) : g1 ≠ g2 → color g1 = green → color g2 = green → color (g1 + g2) = green := sorry

-- The required theorem
theorem coloring_ways : ∃! (f : ℕ → Color), 
  (∀ n, n > 0 → (f n = red ∨ f n = green)) ∧ 
  (∀ r1 r2, r1 ≠ r2 → f r1 = red → f r2 = red → f (r1 + r2) = red) ∧
  (∀ g1 g2, g1 ≠ g2 → f g1 = green → f g2 = green → f (g1 + g2) = green) :=
sorry

end coloring_ways_l162_162687


namespace floor_sqrt_23_squared_l162_162217

theorem floor_sqrt_23_squared : (Nat.floor (Real.sqrt 23)) ^ 2 = 16 :=
by
  -- Proof is omitted
  sorry

end floor_sqrt_23_squared_l162_162217


namespace fraction_multiplication_division_l162_162202

theorem fraction_multiplication_division :
  ((3 / 4) * (5 / 6)) / (7 / 8) = 5 / 7 :=
by
  sorry

end fraction_multiplication_division_l162_162202


namespace power_mod_l162_162484

theorem power_mod : (3^2023) % 5 = 2 := by
  have h1 : 3^1 % 5 = 3 := by sorry
  have h2 : 3^2 % 5 = 4 := by sorry
  have h3 : 3^3 % 5 = 2 := by sorry
  have h4 : 3^4 % 5 = 1 := by sorry
  have periodicity : ∀ k: ℕ, (3^(4*k)) % 5 = 1 := by sorry
  have remainder : 2023 % 4 = 3 := by sorry
  show (3^2023) % 5 = 2 from by
    rw [Nat.mod_eq_of_lt 5 4] at remainder -- remainder shows 2023 mod 4 = 3
    exact h3

end power_mod_l162_162484


namespace fruit_basket_cost_is_28_l162_162181

def basket_total_cost : ℕ := 4 * 1 + 3 * 2 + (24 / 12) * 4 + 2 * 3 + 2 * 2

theorem fruit_basket_cost_is_28 : basket_total_cost = 28 := by
  sorry

end fruit_basket_cost_is_28_l162_162181


namespace predict_grandson_height_l162_162184

noncomputable def heights : List ℝ := [173, 170, 176, 182]

theorem predict_grandson_height (heights : List ℝ) :
  let b := (heights.nth_le 0  sorry * heights.nth_le 1 sorry + heights.nth_le 1 sorry * heights.nth_le 2 sorry + heights.nth_le 2 sorry * heights.nth_le 3 sorry - 3 * heights.nth_le 0 sorry * heights.nth_le 2 sorry) / (heights.nth_le 0 sorry ^ 2 + heights.nth_le 1 sorry ^ 2 + heights.nth_le 2 sorry ^ 2 - 3 * heights.nth_le 0 sorry ^ 2)
  let a := 3
  let y := b * heights.nth_le 3 sorry + a
  y = 185 :=
by
  sorry

end predict_grandson_height_l162_162184


namespace lcm_24_90_l162_162951

theorem lcm_24_90 : lcm 24 90 = 360 :=
by 
-- lcm is the least common multiple of 24 and 90.
-- lcm 24 90 is defined as 360.
sorry

end lcm_24_90_l162_162951


namespace probability_of_two_As_l162_162038

open Probability.Proba

variable {Ω : Type} [ProbabilitySpace Ω]

def pa : ℚ := 4/5
def ph : ℚ := 3/5
def pg : ℚ := 2/5

theorem probability_of_two_As :
  let P_A := pa
  let P_H := ph
  let P_G := pg
  P(X = 2) = 58/125
:= 
by
  sorry

end probability_of_two_As_l162_162038


namespace find_x_l162_162186

theorem find_x (x : ℝ) (h1 : x ≠ 0) (h2 : x = (1 / x) * (-x) + 3) : x = 2 :=
by
  sorry

end find_x_l162_162186


namespace product_of_consecutive_integers_is_perfect_square_l162_162739

theorem product_of_consecutive_integers_is_perfect_square (n : ℤ) :
    n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1) ^ 2 :=
sorry

end product_of_consecutive_integers_is_perfect_square_l162_162739


namespace find_b_l162_162550

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 4) (h2 : b - a = 1) : b = 2 :=
sorry

end find_b_l162_162550


namespace range_of_m_l162_162575

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y - x * y = 0) : 
  ∀ m : ℝ, (xy ≥ m^2 - 6 * m ↔ -2 ≤ m ∧ m ≤ 8) :=
sorry

end range_of_m_l162_162575


namespace correct_average_weight_l162_162577

theorem correct_average_weight 
  (n : ℕ) 
  (w_avg : ℝ) 
  (W_init : ℝ)
  (d1 : ℝ)
  (d2 : ℝ)
  (d3 : ℝ)
  (W_adj : ℝ)
  (w_corr : ℝ)
  (h1 : n = 30)
  (h2 : w_avg = 58.4)
  (h3 : W_init = n * w_avg)
  (h4 : d1 = 62 - 56)
  (h5 : d2 = 59 - 65)
  (h6 : d3 = 54 - 50)
  (h7 : W_adj = W_init + d1 + d2 + d3)
  (h8 : w_corr = W_adj / n) :
  w_corr = 58.5 := 
sorry

end correct_average_weight_l162_162577


namespace right_triangle_legs_l162_162375

theorem right_triangle_legs (c a b : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : ab = c^2 / 4) :
  a = c * (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ b = c * (Real.sqrt 6 - Real.sqrt 2) / 4 := 
sorry

end right_triangle_legs_l162_162375


namespace dan_money_left_l162_162213

def money_left (initial : ℝ) (candy_bar : ℝ) (chocolate : ℝ) (soda : ℝ) (gum : ℝ) : ℝ :=
  initial - candy_bar - chocolate - soda - gum

theorem dan_money_left :
  money_left 10 2 3 1.5 1.25 = 2.25 :=
by
  sorry

end dan_money_left_l162_162213


namespace ballsInBoxes_theorem_l162_162559

noncomputable def ballsInBoxes : ℕ :=
  let distributions := [(5,0,0), (4,1,0), (3,2,0), (3,1,1), (2,2,1)]
  distributions.foldl (λ acc (x : ℕ × ℕ × ℕ) =>
    acc + match x with
      | (5,0,0) => 3
      | (4,1,0) => 6
      | (3,2,0) => 6
      | (3,1,1) => 6
      | (2,2,1) => 3
      | _ => 0
  ) 0

theorem ballsInBoxes_theorem : ballsInBoxes = 24 :=
by
  unfold ballsInBoxes
  rfl

end ballsInBoxes_theorem_l162_162559


namespace largest_angle_in_triangle_l162_162279

noncomputable def angle_sum : ℝ := 120 -- $\frac{4}{3}$ of 90 degrees
noncomputable def angle_difference : ℝ := 20

theorem largest_angle_in_triangle :
  ∃ (a b c : ℝ), a + b + c = 180 ∧ a + b = angle_sum ∧ b = a + angle_difference ∧
  max a (max b c) = 70 :=
by
  sorry

end largest_angle_in_triangle_l162_162279


namespace hours_learning_english_each_day_l162_162686

theorem hours_learning_english_each_day (E : ℕ) 
  (h_chinese_each_day : ∀ (d : ℕ), d = 7) 
  (days : ℕ) 
  (h_total_days : days = 5) 
  (h_total_hours : ∀ (t : ℕ), t = 65) 
  (total_learning_time : 5 * (E + 7) = 65) :
  E = 6 :=
by
  sorry

end hours_learning_english_each_day_l162_162686


namespace floor_difference_l162_162942

theorem floor_difference (x : ℝ) (h : x = 13.2) : 
  (⌊x^2⌋ : ℤ) - ((⌊x⌋ : ℤ) * (⌊x⌋ : ℤ)) = 5 := 
by
  -- proof skipped
  sorry

end floor_difference_l162_162942


namespace probability_interval_l162_162958

-- Definitions based on the given conditions
noncomputable def ξ : MeasureTheory.ProbabilitySpace ℝ :=
  MeasureTheory.gaussian 0 (σ^2)

axiom normal_distribution_symmetry {σ : ℝ} (hσ : 0 < σ) :
  P(ξ > 2) = 0.023

-- The proof problem in Lean 4 statement
theorem probability_interval {σ : ℝ} (hσ : 0 < σ) :
  P(-2 ≤ ξ ∧ ξ ≤ 2) = 0.954 := by
  -- the proof is to be filled in
  sorry

end probability_interval_l162_162958


namespace sum_of_distinct_prime_factors_of_462_l162_162158

theorem sum_of_distinct_prime_factors_of_462 : 
  ∑ p in {2, 3, 7, 11}, p = 23 := by
  have pf462 : unique_factorization_monoid.factors 462 = {2, 3, 7, 11} :=
    by sorry -- Demonstrate or assume correct factorization
  sorry -- Conclude the sum

end sum_of_distinct_prime_factors_of_462_l162_162158


namespace travel_time_home_to_community_center_l162_162563

-- Definitions and assumptions based on the conditions
def time_to_library := 30 -- in minutes
def distance_to_library := 5 -- in miles
def time_spent_at_library := 15 -- in minutes
def distance_to_community_center := 3 -- in miles
noncomputable def cycling_speed := time_to_library / distance_to_library -- in minutes per mile

-- Time calculation to reach the community center from the library
noncomputable def time_from_library_to_community_center := distance_to_community_center * cycling_speed -- in minutes

-- Total time spent to travel from home to the community center
noncomputable def total_time_home_to_community_center :=
  time_to_library + time_spent_at_library + time_from_library_to_community_center

-- The proof statement verifying the total time
theorem travel_time_home_to_community_center : total_time_home_to_community_center = 63 := by
  sorry

end travel_time_home_to_community_center_l162_162563


namespace probability_of_spade_or_king_l162_162342

open Classical

-- Pack of cards containing 52 cards
def total_cards := 52

-- Number of spades in the deck
def num_spades := 13

-- Number of kings in the deck
def num_kings := 4

-- Number of overlap (king of spades)
def num_king_of_spades := 1

-- Total favorable outcomes
def total_favorable_outcomes := num_spades + num_kings - num_king_of_spades

-- Probability of drawing a spade or a king
def probability_spade_or_king := (total_favorable_outcomes : ℚ) / total_cards

theorem probability_of_spade_or_king : probability_spade_or_king = 4 / 13 := by
  sorry

end probability_of_spade_or_king_l162_162342


namespace Jiyeol_average_score_l162_162429

theorem Jiyeol_average_score (K M E : ℝ)
  (h1 : (K + M) / 2 = 26.5)
  (h2 : (M + E) / 2 = 34.5)
  (h3 : (K + E) / 2 = 29) :
  (K + M + E) / 3 = 30 := 
sorry

end Jiyeol_average_score_l162_162429


namespace solve_inequality_l162_162994

theorem solve_inequality {a x : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ (x : ℝ), (a > 1 ∧ (a^(2/3) ≤ x ∧ x < a^(3/4) ∨ x > a)) ∨ (0 < a ∧ a < 1 ∧ (a^(3/4) < x ∧ x ≤ a^(2/3) ∨ 0 < x ∧ x < a))) :=
sorry

end solve_inequality_l162_162994


namespace cos_135_eq_neg_sqrt_2_div_2_point_Q_coordinates_l162_162048

noncomputable def cos_135_deg : Real := - (Real.sqrt 2) / 2

theorem cos_135_eq_neg_sqrt_2_div_2 : Real.cos (135 * Real.pi / 180) = cos_135_deg := sorry

noncomputable def point_Q : Real × Real :=
  (- (Real.sqrt 2) / 2, (Real.sqrt 2) / 2)

theorem point_Q_coordinates :
  ∃ (Q : Real × Real), Q = point_Q ∧ Q = (Real.cos (135 * Real.pi / 180), Real.sin (135 * Real.pi / 180)) := sorry

end cos_135_eq_neg_sqrt_2_div_2_point_Q_coordinates_l162_162048


namespace part1_part2_l162_162502

-- Part (1)
theorem part1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : (x - x^2) < sin x ∧ sin x < x := sorry

-- Part (2)
theorem part2 (a : ℝ) (h : ∃ f : ℝ → ℝ, f = λ x, cos (a * x) - log (1 - x^2) ∧ ∀ x, f 0 = 0 ∧ f' x | x = 0 = 0 ∧ f'' x | x = 0 < 0) : a < -sqrt 2 ∨ a > sqrt 2 := sorry

end part1_part2_l162_162502


namespace no_real_roots_x2_plus_4_l162_162620

theorem no_real_roots_x2_plus_4 : ¬ ∃ x : ℝ, x^2 + 4 = 0 := by
  sorry

end no_real_roots_x2_plus_4_l162_162620


namespace expenditure_ratio_l162_162021

theorem expenditure_ratio (I_A I_B E_A E_B : ℝ) (h1 : I_A / I_B = 5 / 6)
  (h2 : I_B = 7200) (h3 : 1800 = I_A - E_A) (h4 : 1600 = I_B - E_B) :
  E_A / E_B = 3 / 4 :=
sorry

end expenditure_ratio_l162_162021


namespace nine_digit_numbers_divisible_by_2_l162_162967

noncomputable def permutations_count_divisible_by_2 (digits: List ℕ) : ℕ :=
  let n := digits.length
  let freqs := digits.foldr (λ d acc, acc.update d (acc.get d + 1)) (Std.HashMap.empty ℕ ℕ)
  let factorial (n : ℕ) : ℕ :=
    if n = 0 then 1 else n * factorial (n - 1)
  factorial n / (freqs.toList.foldr (λ (_, f) acc, acc * factorial f) 1)

theorem nine_digit_numbers_divisible_by_2 :
  permutations_count_divisible_by_2 [1, 1, 1, 3, 5, 5, 2, 2, 2] = 3360 :=
  by
  -- proof steps can be filled here
  sorry

end nine_digit_numbers_divisible_by_2_l162_162967


namespace area_of_larger_square_is_16_times_l162_162322

-- Define the problem conditions
def perimeter_condition (a b : ℝ) : Prop :=
  4 * a = 4 * 4 * b

-- Define the relationship between the areas of the squares given the side lengths
def area_ratio (a b : ℝ) : ℝ :=
  (a * a) / (b * b)

theorem area_of_larger_square_is_16_times (a b : ℝ) (h : perimeter_condition a b) : area_ratio a b = 16 :=
by 
  unfold perimeter_condition at h
  rw [mul_assoc, mul_comm 4 b] at h
  have ha : a = 4 * b := (mul_right_inj' (ne_of_gt (show 0 < (4:ℝ), by norm_num))).mp h
  unfold area_ratio
  rw [ha, mul_pow, pow_two, mul_pow]
  exact (by norm_num : (4:ℝ)^2 = 16)

end area_of_larger_square_is_16_times_l162_162322


namespace part_one_part_two_l162_162081

noncomputable def f (a x : ℝ) : ℝ :=
  a * x - (1 : ℝ) / x - (a + 1) * Real.log x

theorem part_one (a : ℝ) (h : a = 0) : 
  ∃ x : ℝ, x > 0 ∧ f a x ≤ -1 := 
begin
  sorry
end

theorem part_two : 
  ∀ a : ℝ, (∃ x : ℝ, f a x = 0 ∧ ∀ y : ℝ, y > 0 → f a y ≠ 0 \iff 0 < a) :=
begin
  sorry
end

end part_one_part_two_l162_162081


namespace problem1_problem2_l162_162788

theorem problem1 : (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) * Real.sin (40 * Real.pi / 180) = -1 := 
by
  sorry

theorem problem2 (x : ℝ) : 
  (2 * Real.cos x ^ 4 - 2 * Real.cos x ^ 2 + 1 / 2) /
  (2 * Real.tan (Real.pi / 4 - x) * Real.sin (Real.pi / 4 + x) ^ 2) = 
  Real.sin (2 * x) / 4 := 
by
  sorry

end problem1_problem2_l162_162788


namespace travel_time_l162_162743

theorem travel_time (distance speed : ℝ) (h1 : distance = 300) (h2 : speed = 60) : 
  distance / speed = 5 := 
by
  sorry

end travel_time_l162_162743


namespace determine_ab_l162_162844

theorem determine_ab (a b : ℕ) (h1: a + b = 30) (h2: 2 * a * b + 14 * a = 5 * b + 290) : a * b = 104 := by
  -- the proof would be written here
  sorry

end determine_ab_l162_162844


namespace evaluates_to_m_times_10_pow_1012_l162_162948

theorem evaluates_to_m_times_10_pow_1012 :
  let a := (3:ℤ) ^ 1010
  let b := (4:ℤ) ^ 1012
  (a + b) ^ 2 - (a - b) ^ 2 = 10 ^ 3642 := by
  sorry

end evaluates_to_m_times_10_pow_1012_l162_162948


namespace matrix_power_2018_l162_162118

open Matrix
open Complex

def A : Matrix (Fin 3) (Fin 3) ℂ :=
  ![
    ![sqrt 3 / 2, 0, -1 / 2],
    ![0, -1, 0],
    ![1 / 2, 0, sqrt 3 / 2]
  ]

theorem matrix_power_2018 :
  A ^ 2018 = 
    ![
      ![1 / 2, 0, -sqrt 3 / 2],
      ![0, 1, 0],
      ![sqrt 3 / 2, 0, 1 / 2]
    ] := 
  by
  sorry

end matrix_power_2018_l162_162118


namespace simplify_fraction_l162_162368

variable {R : Type*} [Field R]
variables (x y z : R)

theorem simplify_fraction : (6 * x * y / (5 * z ^ 2)) * (10 * z ^ 3 / (9 * x * y)) = (4 * z) / 3 := by
  sorry

end simplify_fraction_l162_162368
