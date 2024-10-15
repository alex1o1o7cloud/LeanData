import Mathlib

namespace NUMINAMATH_GPT_fraction_to_decimal_equiv_l2047_204789

theorem fraction_to_decimal_equiv : (5 : ℚ) / (16 : ℚ) = 0.3125 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_equiv_l2047_204789


namespace NUMINAMATH_GPT_altered_solution_ratio_l2047_204794

variable (b d w : ℕ)
variable (b' d' w' : ℕ)
variable (ratio_orig_bd_ratio_orig_dw_ratio_orig_bw : Rat)
variable (ratio_new_bd_ratio_new_dw_ratio_new_bw : Rat)

noncomputable def orig_ratios (ratio_orig_bd ratio_orig_bw : Rat) (d w : ℕ) : Prop := 
    ratio_orig_bd = 2 / 40 ∧ ratio_orig_bw = 40 / 100

noncomputable def new_ratios (ratio_new_bd : Rat) (d' : ℕ) : Prop :=
    ratio_new_bd = 6 / 40 ∧ d' = 60

noncomputable def new_solution (w' : ℕ) : Prop :=
    w' = 300

theorem altered_solution_ratio : 
    ∀ (orig_ratios: Prop) (new_ratios: Prop) (new_solution: Prop),
    orig_ratios ∧ new_ratios ∧ new_solution →
    (d' / w = 2 / 5) :=
by
    sorry

end NUMINAMATH_GPT_altered_solution_ratio_l2047_204794


namespace NUMINAMATH_GPT_fraction_subtraction_l2047_204713

theorem fraction_subtraction : (9 / 23) - (5 / 69) = 22 / 69 :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l2047_204713


namespace NUMINAMATH_GPT_number_of_four_digit_numbers_l2047_204760

theorem number_of_four_digit_numbers (digits: Finset ℕ) (h: digits = {1, 1, 2, 0}) :
  ∃ count : ℕ, (count = 9) ∧ 
  (∀ n ∈ digits, n ≠ 0 → n * 1000 + n ≠ 0) := 
sorry

end NUMINAMATH_GPT_number_of_four_digit_numbers_l2047_204760


namespace NUMINAMATH_GPT_regular_price_coffee_l2047_204762

theorem regular_price_coffee (y : ℝ) (h1 : 0.4 * y / 4 = 4) : y = 40 :=
by
  sorry

end NUMINAMATH_GPT_regular_price_coffee_l2047_204762


namespace NUMINAMATH_GPT_sum_of_b_for_one_solution_l2047_204795

theorem sum_of_b_for_one_solution (b : ℝ) (has_single_solution : ∃ x, 3 * x^2 + (b + 12) * x + 11 = 0) :
  ∃ b₁ b₂ : ℝ, (3 * x^2 + (b + 12) * x + 11) = 0 ∧ b₁ + b₂ = -24 := by
  sorry

end NUMINAMATH_GPT_sum_of_b_for_one_solution_l2047_204795


namespace NUMINAMATH_GPT_smallest_four_digit_in_pascal_l2047_204740

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end NUMINAMATH_GPT_smallest_four_digit_in_pascal_l2047_204740


namespace NUMINAMATH_GPT_rational_cubes_rational_values_l2047_204766

theorem rational_cubes_rational_values {a b : ℝ} (ha : 0 < a) (hb : 0 < b) 
  (hab : a + b = 1) (ha3 : ∃ r : ℚ, a^3 = r) (hb3 : ∃ s : ℚ, b^3 = s) : 
  ∃ r s : ℚ, a = r ∧ b = s :=
sorry

end NUMINAMATH_GPT_rational_cubes_rational_values_l2047_204766


namespace NUMINAMATH_GPT_inscribed_square_ab_l2047_204776

theorem inscribed_square_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^2 + b^2 = 32) : 2 * a * b = -7 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_square_ab_l2047_204776


namespace NUMINAMATH_GPT_farmer_rewards_l2047_204797

theorem farmer_rewards (x y : ℕ) (h1 : x + y = 60) (h2 : 1000 * x + 3000 * y = 100000) : x = 40 ∧ y = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_farmer_rewards_l2047_204797


namespace NUMINAMATH_GPT_range_of_function_x_geq_0_l2047_204708

theorem range_of_function_x_geq_0 :
  ∀ (x : ℝ), x ≥ 0 → ∃ (y : ℝ), y ≥ 3 ∧ (y = x^2 + 2 * x + 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_x_geq_0_l2047_204708


namespace NUMINAMATH_GPT_column_sum_correct_l2047_204792

theorem column_sum_correct : 
  -- Define x to be the sum of the first column (which is also the minuend of the second column)
  ∃ x : ℕ, 
  -- x should match the expected valid sum provided:
  (x = 1001) := 
sorry

end NUMINAMATH_GPT_column_sum_correct_l2047_204792


namespace NUMINAMATH_GPT_minimal_blue_chips_value_l2047_204775

noncomputable def minimal_blue_chips (r g b : ℕ) : Prop :=
b ≥ r / 3 ∧
b ≤ g / 4 ∧
r + g ≥ 75

theorem minimal_blue_chips_value : ∃ (b : ℕ), minimal_blue_chips 33 44 b ∧ b = 11 :=
by
  have b := 11
  use b
  sorry

end NUMINAMATH_GPT_minimal_blue_chips_value_l2047_204775


namespace NUMINAMATH_GPT_lawrence_walked_total_distance_l2047_204709

noncomputable def distance_per_day : ℝ := 4.0
noncomputable def number_of_days : ℝ := 3.0
noncomputable def total_distance_walked (distance_per_day : ℝ) (number_of_days : ℝ) : ℝ :=
  distance_per_day * number_of_days

theorem lawrence_walked_total_distance :
  total_distance_walked distance_per_day number_of_days = 12.0 :=
by
  -- The detailed proof is omitted as per the instructions.
  sorry

end NUMINAMATH_GPT_lawrence_walked_total_distance_l2047_204709


namespace NUMINAMATH_GPT_find_c_exactly_two_common_points_l2047_204729

theorem find_c_exactly_two_common_points (c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^3 - 3*x1 + c = 0) ∧ (x2^3 - 3*x2 + c = 0)) ↔ (c = -2 ∨ c = 2) := 
sorry

end NUMINAMATH_GPT_find_c_exactly_two_common_points_l2047_204729


namespace NUMINAMATH_GPT_correct_equation_l2047_204763

theorem correct_equation : -(-5) = |-5| :=
by
  -- sorry is used here to skip the actual proof steps which are not required.
  sorry

end NUMINAMATH_GPT_correct_equation_l2047_204763


namespace NUMINAMATH_GPT_existence_of_subset_A_l2047_204722

def M : Set ℚ := {x : ℚ | 0 < x ∧ x < 1}

theorem existence_of_subset_A :
  ∃ A ⊆ M, ∀ m ∈ M, ∃! (S : Finset ℚ), (∀ a ∈ S, a ∈ A) ∧ (S.sum id = m) :=
sorry

end NUMINAMATH_GPT_existence_of_subset_A_l2047_204722


namespace NUMINAMATH_GPT_ratio_future_age_l2047_204737

variable (S M : ℕ)

theorem ratio_future_age (h1 : (S : ℝ) / M = 7 / 2) (h2 : S - 6 = 78) : 
  ((S + 16) : ℝ) / (M + 16) = 5 / 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_future_age_l2047_204737


namespace NUMINAMATH_GPT_express_function_as_chain_of_equalities_l2047_204731

theorem express_function_as_chain_of_equalities (x : ℝ) : 
  ∃ (u : ℝ), (u = 2 * x - 5) ∧ ((2 * x - 5) ^ 10 = u ^ 10) :=
by 
  sorry

end NUMINAMATH_GPT_express_function_as_chain_of_equalities_l2047_204731


namespace NUMINAMATH_GPT_fruit_seller_sp_l2047_204790

theorem fruit_seller_sp (CP SP : ℝ)
    (h1 : SP = 0.75 * CP)
    (h2 : 19.93 = 1.15 * CP) :
    SP = 13.00 :=
by
  sorry

end NUMINAMATH_GPT_fruit_seller_sp_l2047_204790


namespace NUMINAMATH_GPT_polynomial_abs_sum_eq_81_l2047_204723

theorem polynomial_abs_sum_eq_81 
  (a a_1 a_2 a_3 a_4 : ℝ) 
  (h : (1 - 2 * x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4)
  (ha : a > 0) 
  (ha_2 : a_2 > 0) 
  (ha_4 : a_4 > 0) 
  (ha_1 : a_1 < 0) 
  (ha_3 : a_3 < 0): 
  |a| + |a_1| + |a_2| + |a_3| + |a_4| = 81 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_abs_sum_eq_81_l2047_204723


namespace NUMINAMATH_GPT_janice_initial_sentences_l2047_204727

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

end NUMINAMATH_GPT_janice_initial_sentences_l2047_204727


namespace NUMINAMATH_GPT_charity_event_fund_raising_l2047_204751

theorem charity_event_fund_raising :
  let n := 9
  let I := 2000
  let p := 0.10
  let increased_total := I * (1 + p)
  let amount_per_person := increased_total / n
  amount_per_person = 244.44 := by
  sorry

end NUMINAMATH_GPT_charity_event_fund_raising_l2047_204751


namespace NUMINAMATH_GPT_remainder_of_expression_l2047_204758

theorem remainder_of_expression (a b c d : ℕ) (h1 : a = 8) (h2 : b = 20) (h3 : c = 34) (h4 : d = 3) :
  (a * b ^ c + d ^ c) % 7 = 5 := 
by 
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_remainder_of_expression_l2047_204758


namespace NUMINAMATH_GPT_sum_of_integers_l2047_204749

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 103) 
                        (h2 : Nat.gcd a b = 1) 
                        (h3 : a < 20) 
                        (h4 : b < 20) : 
                        a + b = 19 :=
  by sorry

end NUMINAMATH_GPT_sum_of_integers_l2047_204749


namespace NUMINAMATH_GPT_find_angle_B_l2047_204786

variables {A B C a b c : ℝ} (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3)

theorem find_angle_B (h1 : 3 * a * Real.cos C = 2 * c * Real.cos A) (h2 : Real.tan A = 1 / 3) : B = 3 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_B_l2047_204786


namespace NUMINAMATH_GPT_value_of_fg3_l2047_204759

namespace ProofProblem

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2

theorem value_of_fg3 : f (g 3) = 83 := 
by 
  sorry -- Proof not needed

end ProofProblem

end NUMINAMATH_GPT_value_of_fg3_l2047_204759


namespace NUMINAMATH_GPT_students_played_both_l2047_204720

theorem students_played_both (C B X total : ℕ) (hC : C = 500) (hB : B = 600) (hTotal : total = 880) (hInclusionExclusion : C + B - X = total) : X = 220 :=
by
  rw [hC, hB, hTotal] at hInclusionExclusion
  sorry

end NUMINAMATH_GPT_students_played_both_l2047_204720


namespace NUMINAMATH_GPT_joan_total_money_l2047_204733

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

end NUMINAMATH_GPT_joan_total_money_l2047_204733


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_condition_l2047_204799

noncomputable def p (x : ℝ) : Prop := (x - 2) * (x - 1) > 0

noncomputable def q (x : ℝ) : Prop := x - 2 > 0 ∨ x - 1 > 0

theorem neither_sufficient_nor_necessary_condition (x : ℝ) : ¬(p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_condition_l2047_204799


namespace NUMINAMATH_GPT_pictures_at_museum_l2047_204703

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

end NUMINAMATH_GPT_pictures_at_museum_l2047_204703


namespace NUMINAMATH_GPT_percentage_in_biology_is_correct_l2047_204735

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

end NUMINAMATH_GPT_percentage_in_biology_is_correct_l2047_204735


namespace NUMINAMATH_GPT_train_length_l2047_204754

theorem train_length (L V : ℝ) (h1 : V = L / 15) (h2 : V = (L + 100) / 40) : L = 60 := by
  sorry

end NUMINAMATH_GPT_train_length_l2047_204754


namespace NUMINAMATH_GPT_derivative_at_one_l2047_204788

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * Real.log x

theorem derivative_at_one : (deriv f 1) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_derivative_at_one_l2047_204788


namespace NUMINAMATH_GPT_possible_values_of_a_l2047_204755

def A (a : ℤ) : Set ℤ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℤ) : Set ℤ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem possible_values_of_a (a : ℤ) :
  A a ∩ B a = {2, 5} ↔ a = -1 ∨ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l2047_204755


namespace NUMINAMATH_GPT_exists_a_b_l2047_204719

theorem exists_a_b (S : Finset ℕ) (hS : S.card = 43) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (a^2 - b^2) % 100 = 0 := 
by
  sorry

end NUMINAMATH_GPT_exists_a_b_l2047_204719


namespace NUMINAMATH_GPT_odd_expressions_l2047_204743

theorem odd_expressions (m n p : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) (hp : p % 2 = 0) : 
  ((2 * m * n + 5) ^ 2 % 2 = 1) ∧ (5 * m * n + p % 2 = 1) := 
by
  sorry

end NUMINAMATH_GPT_odd_expressions_l2047_204743


namespace NUMINAMATH_GPT_trevor_spends_more_l2047_204707

theorem trevor_spends_more (T R Q : ℕ) 
  (hT : T = 80) 
  (hR : R = 2 * Q) 
  (hTotal : 4 * (T + R + Q) = 680) : 
  T = R + 20 :=
by
  sorry

end NUMINAMATH_GPT_trevor_spends_more_l2047_204707


namespace NUMINAMATH_GPT_cricket_team_members_count_l2047_204756

theorem cricket_team_members_count 
(captain_age : ℕ) (wk_keeper_age : ℕ) (whole_team_avg_age : ℕ)
(remaining_players_avg_age : ℕ) (n : ℕ) 
(h1 : captain_age = 28)
(h2 : wk_keeper_age = captain_age + 3)
(h3 : whole_team_avg_age = 25)
(h4 : remaining_players_avg_age = 24)
(h5 : (n * whole_team_avg_age - (captain_age + wk_keeper_age)) / (n - 2) = remaining_players_avg_age) :
n = 11 := 
sorry

end NUMINAMATH_GPT_cricket_team_members_count_l2047_204756


namespace NUMINAMATH_GPT_factor_adjustment_l2047_204724

theorem factor_adjustment (a b : ℝ) (h : a * b = 65.08) : a / 100 * (100 * b) = 65.08 :=
by
  sorry

end NUMINAMATH_GPT_factor_adjustment_l2047_204724


namespace NUMINAMATH_GPT_crayons_left_l2047_204787

-- Define the initial number of crayons and the number taken
def initial_crayons : ℕ := 7
def crayons_taken : ℕ := 3

-- Prove the number of crayons left in the drawer
theorem crayons_left : initial_crayons - crayons_taken = 4 :=
by
  sorry

end NUMINAMATH_GPT_crayons_left_l2047_204787


namespace NUMINAMATH_GPT_Tyler_age_l2047_204779

variable (T B S : ℕ)

theorem Tyler_age :
  (T = B - 3) ∧
  (S = B + 2) ∧
  (S = 2 * T) ∧
  (T + B + S = 30) →
  T = 5 := by
  sorry

end NUMINAMATH_GPT_Tyler_age_l2047_204779


namespace NUMINAMATH_GPT_range_of_values_l2047_204764

theorem range_of_values (x : ℝ) (h1 : x - 1 ≥ 0) (h2 : x ≠ 0) : x ≥ 1 := 
sorry

end NUMINAMATH_GPT_range_of_values_l2047_204764


namespace NUMINAMATH_GPT_sum_of_decimals_as_fraction_l2047_204780

axiom decimal_to_fraction :
  0.2 = 2 / 10 ∧
  0.04 = 4 / 100 ∧
  0.006 = 6 / 1000 ∧
  0.0008 = 8 / 10000 ∧
  0.00010 = 10 / 100000 ∧
  0.000012 = 12 / 1000000

theorem sum_of_decimals_as_fraction:
  0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 + 0.000012 = (3858:ℚ) / 15625 :=
by
  have h := decimal_to_fraction
  sorry

end NUMINAMATH_GPT_sum_of_decimals_as_fraction_l2047_204780


namespace NUMINAMATH_GPT_expression_equals_12_l2047_204750

-- Define the values of a, b, c, and k
def a : ℤ := 10
def b : ℤ := 15
def c : ℤ := 3
def k : ℤ := 2

-- Define the expression to be evaluated
def expr : ℤ := (a - (b - k * c)) - ((a - b) - k * c)

-- Prove that the expression equals 12
theorem expression_equals_12 : expr = 12 :=
by
  -- The proof will go here, leaving a placeholder for now
  sorry

end NUMINAMATH_GPT_expression_equals_12_l2047_204750


namespace NUMINAMATH_GPT_find_ordered_pair_l2047_204730

theorem find_ordered_pair {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = -2 * a ∨ x = b)
  (h2 : b = -2 * -2 * a) : (a, b) = (-1/2, -1/2) :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l2047_204730


namespace NUMINAMATH_GPT_arith_seq_common_diff_l2047_204738

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

end NUMINAMATH_GPT_arith_seq_common_diff_l2047_204738


namespace NUMINAMATH_GPT_Zack_kept_5_marbles_l2047_204782

-- Define the initial number of marbles Zack had
def Zack_initial_marbles : ℕ := 65

-- Define the number of marbles each friend receives
def marbles_per_friend : ℕ := 20

-- Define the total number of friends
def friends : ℕ := 3

noncomputable def marbles_given_away : ℕ := friends * marbles_per_friend

-- Define the amount of marbles kept by Zack
noncomputable def marbles_kept_by_Zack : ℕ := Zack_initial_marbles - marbles_given_away

-- The theorem to prove
theorem Zack_kept_5_marbles : marbles_kept_by_Zack = 5 := by
  -- Proof skipped with sorry
  sorry

end NUMINAMATH_GPT_Zack_kept_5_marbles_l2047_204782


namespace NUMINAMATH_GPT_correct_letter_is_P_l2047_204732

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

end NUMINAMATH_GPT_correct_letter_is_P_l2047_204732


namespace NUMINAMATH_GPT_daily_rate_first_week_l2047_204778

-- Definitions from given conditions
variable (x : ℝ) (h1 : ∀ y : ℝ, 0 ≤ y)
def cost_first_week := 7 * x
def additional_days_cost := 16 * 14
def total_cost := cost_first_week + additional_days_cost

-- Theorem to solve the problem
theorem daily_rate_first_week (h : total_cost = 350) : x = 18 :=
sorry

end NUMINAMATH_GPT_daily_rate_first_week_l2047_204778


namespace NUMINAMATH_GPT_max_value_of_expression_l2047_204772

-- Define the real numbers p, q, r and the conditions
variables {p q r : ℝ}

-- Define the main goal
theorem max_value_of_expression 
(h : 9 * p^2 + 4 * q^2 + 25 * r^2 = 4) : 
  (5 * p + 3 * q + 10 * r) ≤ (10 * Real.sqrt 13 / 3) :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2047_204772


namespace NUMINAMATH_GPT_HCF_of_two_numbers_l2047_204793

theorem HCF_of_two_numbers (a b : ℕ) (h1 : a * b = 2562) (h2 : Nat.lcm a b = 183) : Nat.gcd a b = 14 := 
by
  sorry

end NUMINAMATH_GPT_HCF_of_two_numbers_l2047_204793


namespace NUMINAMATH_GPT_C_increases_as_n_increases_l2047_204753

theorem C_increases_as_n_increases (e n R r : ℝ) (he : 0 < e) (hn : 0 < n) (hR : 0 < R) (hr : 0 < r) :
  0 < (2 * e * n * R + e * n^2 * r) / (R + n * r)^2 :=
by
  sorry

end NUMINAMATH_GPT_C_increases_as_n_increases_l2047_204753


namespace NUMINAMATH_GPT_students_at_end_of_year_l2047_204717

-- Define the initial number of students
def initial_students : Nat := 10

-- Define the number of students who left during the year
def students_left : Nat := 4

-- Define the number of new students who arrived during the year
def new_students : Nat := 42

-- Proof problem: the number of students at the end of the year
theorem students_at_end_of_year : initial_students - students_left + new_students = 48 := by
  sorry

end NUMINAMATH_GPT_students_at_end_of_year_l2047_204717


namespace NUMINAMATH_GPT_new_daily_average_wage_l2047_204781

theorem new_daily_average_wage (x : ℝ) : 
  (∀ y : ℝ, 25 - x = y) → 
  (∀ z : ℝ, 20 * (25 - x) = 30 * (10)) → 
  x = 10 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_new_daily_average_wage_l2047_204781


namespace NUMINAMATH_GPT_janet_extra_cost_l2047_204791

theorem janet_extra_cost :
  let clarinet_hourly_rate := 40
  let clarinet_hours_per_week := 3
  let clarinet_weeks_per_year := 50
  let clarinet_yearly_cost := clarinet_hourly_rate * clarinet_hours_per_week * clarinet_weeks_per_year

  let piano_hourly_rate := 28
  let piano_hours_per_week := 5
  let piano_weeks_per_year := 50
  let piano_yearly_cost := piano_hourly_rate * piano_hours_per_week * piano_weeks_per_year
  let piano_discount_rate := 0.10
  let piano_discounted_yearly_cost := piano_yearly_cost * (1 - piano_discount_rate)

  let violin_hourly_rate := 35
  let violin_hours_per_week := 2
  let violin_weeks_per_year := 50
  let violin_yearly_cost := violin_hourly_rate * violin_hours_per_week * violin_weeks_per_year
  let violin_discount_rate := 0.15
  let violin_discounted_yearly_cost := violin_yearly_cost * (1 - violin_discount_rate)

  let singing_hourly_rate := 45
  let singing_hours_per_week := 1
  let singing_weeks_per_year := 50
  let singing_yearly_cost := singing_hourly_rate * singing_hours_per_week * singing_weeks_per_year

  let combined_cost := piano_discounted_yearly_cost + violin_discounted_yearly_cost + singing_yearly_cost
  combined_cost - clarinet_yearly_cost = 5525 := 
  sorry

end NUMINAMATH_GPT_janet_extra_cost_l2047_204791


namespace NUMINAMATH_GPT_lance_read_yesterday_l2047_204784

-- Definitions based on conditions
def total_pages : ℕ := 100
def pages_tomorrow : ℕ := 35
def pages_yesterday (Y : ℕ) : ℕ := Y
def pages_today (Y : ℕ) : ℕ := Y - 5

-- The statement that we need to prove
theorem lance_read_yesterday (Y : ℕ) (h : pages_yesterday Y + pages_today Y + pages_tomorrow = total_pages) : Y = 35 :=
by sorry

end NUMINAMATH_GPT_lance_read_yesterday_l2047_204784


namespace NUMINAMATH_GPT_circle_represents_real_l2047_204711

theorem circle_represents_real
  (a : ℝ)
  (h : ∀ x y : ℝ, x^2 + y^2 + 2*y + 2*a - 1 = 0 → ∃ r : ℝ, r > 0) : 
  a < 1 := 
sorry

end NUMINAMATH_GPT_circle_represents_real_l2047_204711


namespace NUMINAMATH_GPT_day_of_week_after_n_days_l2047_204769

theorem day_of_week_after_n_days (birthday : ℕ) (n : ℕ) (day_of_week : ℕ) :
  birthday = 4 → (n % 7) = 2 → day_of_week = 6 :=
by sorry

end NUMINAMATH_GPT_day_of_week_after_n_days_l2047_204769


namespace NUMINAMATH_GPT_amount_c_is_1600_l2047_204798

-- Given conditions
def total_money : ℕ := 2000
def ratio_b_c : (ℕ × ℕ) := (4, 16)

-- Define the total_parts based on the ratio
def total_parts := ratio_b_c.fst + ratio_b_c.snd

-- Define the value of each part
def value_per_part := total_money / total_parts

-- Calculate the amount for c
def amount_c_gets := ratio_b_c.snd * value_per_part

-- Main theorem stating the problem
theorem amount_c_is_1600 : amount_c_gets = 1600 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_amount_c_is_1600_l2047_204798


namespace NUMINAMATH_GPT_min_value_of_X_l2047_204765

theorem min_value_of_X (n : ℕ) (h : n ≥ 2) 
  (X : Finset ℕ) 
  (B : Fin n → Finset ℕ) 
  (hB : ∀ i, (B i).card = 2) :
  ∃ (Y : Finset ℕ), Y.card = n ∧ ∀ i, (Y ∩ (B i)).card ≤ 1 →
  X.card = 2 * n - 1 :=
sorry

end NUMINAMATH_GPT_min_value_of_X_l2047_204765


namespace NUMINAMATH_GPT_cost_effective_for_3000_cost_equal_at_2500_l2047_204745

def cost_company_A (x : Nat) : Nat :=
  2 * x / 10 + 500

def cost_company_B (x : Nat) : Nat :=
  4 * x / 10

theorem cost_effective_for_3000 : cost_company_A 3000 < cost_company_B 3000 := 
by {
  sorry
}

theorem cost_equal_at_2500 : cost_company_A 2500 = cost_company_B 2500 := 
by {
  sorry
}

end NUMINAMATH_GPT_cost_effective_for_3000_cost_equal_at_2500_l2047_204745


namespace NUMINAMATH_GPT_polynomial_remainder_correct_l2047_204736

noncomputable def remainder_polynomial (x : ℝ) : ℝ := x ^ 100

def divisor_polynomial (x : ℝ) : ℝ := x ^ 2 - 3 * x + 2

def polynomial_remainder (x : ℝ) : ℝ := 2 ^ 100 * (x - 1) - (x - 2)

theorem polynomial_remainder_correct : ∀ x : ℝ, (remainder_polynomial x) % (divisor_polynomial x) = polynomial_remainder x := by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_correct_l2047_204736


namespace NUMINAMATH_GPT_prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l2047_204768

-- Problem 1
theorem prob1_part1 : |-4 + 6| = 2 := sorry
theorem prob1_part2 : |-2 - 4| = 6 := sorry

-- Problem 2
theorem find_integers_x :
  {x : ℤ | |x + 2| + |x - 1| = 3} = {-2, -1, 0, 1} :=
sorry

-- Problem 3
theorem prob3 (a : ℤ) (h : -4 ≤ a ∧ a ≤ 6) : |a + 4| + |a - 6| = 10 :=
sorry

-- Problem 4
theorem min_value_prob4 :
  ∃ (a : ℤ), |a - 1| + |a + 5| + |a - 4| = 9 ∧ ∀ (b : ℤ), |b - 1| + |b + 5| + |b - 4| ≥ 9 :=
sorry

end NUMINAMATH_GPT_prob1_part1_prob1_part2_find_integers_x_prob3_min_value_prob4_l2047_204768


namespace NUMINAMATH_GPT_value_of_m_l2047_204777

theorem value_of_m (a b m : ℝ)
    (h1: 2 ^ a = m)
    (h2: 5 ^ b = m)
    (h3: 1 / a + 1 / b = 1 / 2) :
    m = 100 :=
sorry

end NUMINAMATH_GPT_value_of_m_l2047_204777


namespace NUMINAMATH_GPT_minimum_value_of_reciprocals_l2047_204744

theorem minimum_value_of_reciprocals {m n : ℝ} 
  (hmn : m > 0 ∧ n > 0 ∧ (m * n > 0)) 
  (hline : 2 * m + 2 * n = 1) : 
  (1 / m + 1 / n) = 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_reciprocals_l2047_204744


namespace NUMINAMATH_GPT_sqrt_23_parts_xy_diff_l2047_204712

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

end NUMINAMATH_GPT_sqrt_23_parts_xy_diff_l2047_204712


namespace NUMINAMATH_GPT_shaded_area_l2047_204718

theorem shaded_area (r₁ r₂ r₃ : ℝ) (h₁ : 0 < r₁) (h₂ : 0 < r₂) (h₃ : 0 < r₃) (h₁₂ : r₁ < r₂) (h₂₃ : r₂ < r₃)
    (area_shaded_div_area_unshaded : (r₁^2 * π) + (r₂^2 * π) + (r₃^2 * π) = 77 * π)
    (shaded_by_unshaded_ratio : ∀ S U : ℝ, S = (3 / 7) * U) :
    ∃ S : ℝ, S = (1617 * π) / 70 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l2047_204718


namespace NUMINAMATH_GPT_decreasing_function_a_leq_zero_l2047_204741

theorem decreasing_function_a_leq_zero (a : ℝ) :
  (∀ x y : ℝ, x < y → ax^3 - x ≥ ay^3 - y) → a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_a_leq_zero_l2047_204741


namespace NUMINAMATH_GPT_top_width_of_channel_l2047_204716

theorem top_width_of_channel (b : ℝ) (A : ℝ) (h : ℝ) (w : ℝ) : 
  b = 8 ∧ A = 700 ∧ h = 70 ∧ (A = (1/2) * (w + b) * h) → w = 12 := 
by 
  intro h1
  sorry

end NUMINAMATH_GPT_top_width_of_channel_l2047_204716


namespace NUMINAMATH_GPT_spherical_distance_between_points_l2047_204761

noncomputable def spherical_distance (R : ℝ) (α : ℝ) : ℝ :=
  α * R

theorem spherical_distance_between_points 
  (R : ℝ) 
  (α : ℝ) 
  (hR : R > 0) 
  (hα : α = π / 6) : 
  spherical_distance R α = (π / 6) * R :=
by
  rw [hα]
  unfold spherical_distance
  ring

end NUMINAMATH_GPT_spherical_distance_between_points_l2047_204761


namespace NUMINAMATH_GPT_greatest_possible_x_exists_greatest_x_l2047_204721

theorem greatest_possible_x (x : ℤ) (h1 : 6.1 * (10 : ℝ) ^ x < 620) : x ≤ 2 :=
sorry

theorem exists_greatest_x : ∃ x : ℤ, 6.1 * (10 : ℝ) ^ x < 620 ∧ x = 2 :=
sorry

end NUMINAMATH_GPT_greatest_possible_x_exists_greatest_x_l2047_204721


namespace NUMINAMATH_GPT_total_arrangements_l2047_204757

-- Defining the selection and arrangement problem conditions
def select_and_arrange (n m : ℕ) : ℕ :=
  Nat.choose n m * Nat.factorial m

-- Specifying the specific problem's constraints and results
theorem total_arrangements : select_and_arrange 8 2 * select_and_arrange 6 2 = 60 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_arrangements_l2047_204757


namespace NUMINAMATH_GPT_time_to_cross_platform_l2047_204726

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

end NUMINAMATH_GPT_time_to_cross_platform_l2047_204726


namespace NUMINAMATH_GPT_min_value_of_A_sq_sub_B_sq_l2047_204746

noncomputable def A (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (2 * x + 2) + Real.sqrt (2 * y + 2) + Real.sqrt (2 * z + 2)

theorem min_value_of_A_sq_sub_B_sq (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  A x y z ^ 2 - B x y z ^ 2 ≥ 36 :=
  sorry

end NUMINAMATH_GPT_min_value_of_A_sq_sub_B_sq_l2047_204746


namespace NUMINAMATH_GPT_number_of_unique_products_l2047_204742

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

end NUMINAMATH_GPT_number_of_unique_products_l2047_204742


namespace NUMINAMATH_GPT_y_intercept_range_l2047_204706

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

end NUMINAMATH_GPT_y_intercept_range_l2047_204706


namespace NUMINAMATH_GPT_book_cost_price_l2047_204725

theorem book_cost_price
  (C : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : SP = 1.25 * C)
  (h2 : 0.95 * P = SP)
  (h3 : SP = 62.5) : 
  C = 50 := 
by
  sorry

end NUMINAMATH_GPT_book_cost_price_l2047_204725


namespace NUMINAMATH_GPT_parabola_vertex_in_fourth_quadrant_l2047_204704

theorem parabola_vertex_in_fourth_quadrant (a c : ℝ) (h : -a > 0 ∧ c < 0) :
  a < 0 ∧ c < 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_in_fourth_quadrant_l2047_204704


namespace NUMINAMATH_GPT_max_marks_l2047_204710

theorem max_marks (marks_secured : ℝ) (percentage : ℝ) (max_marks : ℝ) 
  (h1 : marks_secured = 332) 
  (h2 : percentage = 83) 
  (h3 : percentage = (marks_secured / max_marks) * 100) 
  : max_marks = 400 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l2047_204710


namespace NUMINAMATH_GPT_problem_statement_l2047_204785

noncomputable def a := Real.log 2 / Real.log 14
noncomputable def b := Real.log 2 / Real.log 7
noncomputable def c := Real.log 2 / Real.log 4

theorem problem_statement : (1 / a - 1 / b + 1 / c) = 3 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2047_204785


namespace NUMINAMATH_GPT_folded_triangle_sqrt_equals_l2047_204728

noncomputable def folded_triangle_length_squared (s : ℕ) (d : ℕ) : ℚ :=
  let x := (2 * s * s - 2 * d * s)/(2 * d)
  let y := (2 * s * s - 2 * (s - d) * s)/(2 * (s - d))
  x * x - x * y + y * y

theorem folded_triangle_sqrt_equals :
  folded_triangle_length_squared 15 11 = (60118.9025 / 1681 : ℚ) := sorry

end NUMINAMATH_GPT_folded_triangle_sqrt_equals_l2047_204728


namespace NUMINAMATH_GPT_find_distance_between_sides_of_trapezium_l2047_204705

variable (side1 side2 h area : ℝ)
variable (h1 : side1 = 20)
variable (h2 : side2 = 18)
variable (h3 : area = 228)
variable (trapezium_area : area = (1 / 2) * (side1 + side2) * h)

theorem find_distance_between_sides_of_trapezium : h = 12 := by
  sorry

end NUMINAMATH_GPT_find_distance_between_sides_of_trapezium_l2047_204705


namespace NUMINAMATH_GPT_primes_digit_sum_difference_l2047_204783

def is_prime (a : ℕ) : Prop := Nat.Prime a

def sum_digits (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

theorem primes_digit_sum_difference (p q r : ℕ) (n : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (hneq : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (hpqr : p * q * r = 1899 * 10^n + 962) :
  (sum_digits p + sum_digits q + sum_digits r - sum_digits (p * q * r) = 8) := 
sorry

end NUMINAMATH_GPT_primes_digit_sum_difference_l2047_204783


namespace NUMINAMATH_GPT_sabrina_total_leaves_l2047_204739

-- Definitions based on conditions
def basil_leaves := 12
def twice_the_sage_leaves (sages : ℕ) := 2 * sages = basil_leaves
def five_fewer_than_verbena (sages verbenas : ℕ) := sages + 5 = verbenas

-- Statement to prove
theorem sabrina_total_leaves (sages verbenas : ℕ) 
    (h1 : twice_the_sage_leaves sages) 
    (h2 : five_fewer_than_verbena sages verbenas) :
    basil_leaves + sages + verbenas = 29 :=
sorry

end NUMINAMATH_GPT_sabrina_total_leaves_l2047_204739


namespace NUMINAMATH_GPT_julia_miles_l2047_204734

theorem julia_miles (total_miles darius_miles julia_miles : ℕ) 
  (h1 : darius_miles = 679)
  (h2 : total_miles = 1677)
  (h3 : total_miles = darius_miles + julia_miles) :
  julia_miles = 998 :=
by
  sorry

end NUMINAMATH_GPT_julia_miles_l2047_204734


namespace NUMINAMATH_GPT_amount_decreased_is_5_l2047_204715

noncomputable def x : ℕ := 50
noncomputable def equation (x y : ℕ) : Prop := (1 / 5) * x - y = 5

theorem amount_decreased_is_5 : ∃ y : ℕ, equation x y ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_amount_decreased_is_5_l2047_204715


namespace NUMINAMATH_GPT_smallest_integer_n_l2047_204748

theorem smallest_integer_n (m n : ℕ) (r : ℝ) :
  (m = (n + r)^3) ∧ (0 < r) ∧ (r < 1 / 2000) ∧ (m = n^3 + 3 * n^2 * r + 3 * n * r^2 + r^3) →
  n = 26 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_integer_n_l2047_204748


namespace NUMINAMATH_GPT_original_amount_of_milk_is_720_l2047_204770

variable (M : ℝ) -- The original amount of milk in milliliters

theorem original_amount_of_milk_is_720 :
  ((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)) - ((2 / 3) * (((5 / 6) * M) - ((2 / 5) * ((5 / 6) * M)))) = 120 → 
  M = 720 := by
  sorry

end NUMINAMATH_GPT_original_amount_of_milk_is_720_l2047_204770


namespace NUMINAMATH_GPT_minimum_people_correct_answer_l2047_204752

theorem minimum_people_correct_answer (people questions : ℕ) (common_correct : ℕ) (h_people : people = 21) (h_questions : questions = 15) (h_common_correct : ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ people → 1 ≤ b ∧ b ≤ people → a ≠ b → common_correct ≥ 1) :
  ∃ (min_correct : ℕ), min_correct = 7 := 
sorry

end NUMINAMATH_GPT_minimum_people_correct_answer_l2047_204752


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2047_204714

open Complex

theorem point_in_fourth_quadrant (z : ℂ) (h : (3 + 4 * I) * z = 25) : 
  Complex.arg z > -π / 2 ∧ Complex.arg z < 0 := 
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2047_204714


namespace NUMINAMATH_GPT_range_of_a_l2047_204796

theorem range_of_a (a : ℝ) (h_a : a > 0) :
  (∃ t : ℝ, (5 * t + 1)^2 + (12 * t - 1)^2 = 2 * a * (5 * t + 1)) ↔ (0 < a ∧ a ≤ 17 / 25) := 
sorry

end NUMINAMATH_GPT_range_of_a_l2047_204796


namespace NUMINAMATH_GPT_operation_on_b_l2047_204767

variables (t b b' : ℝ)
variable (C : ℝ := t * b ^ 4)
variable (e : ℝ := 16 * C)

theorem operation_on_b :
  tb'^4 = 16 * tb^4 → b' = 2 * b := by
  sorry

end NUMINAMATH_GPT_operation_on_b_l2047_204767


namespace NUMINAMATH_GPT_medical_team_formation_l2047_204771

theorem medical_team_formation (m f : ℕ) (h_m : m = 5) (h_f : f = 4) :
  (m + f).choose 3 - m.choose 3 - f.choose 3 = 70 :=
by
  sorry

end NUMINAMATH_GPT_medical_team_formation_l2047_204771


namespace NUMINAMATH_GPT_smallest_positive_debt_resolved_l2047_204702

theorem smallest_positive_debt_resolved : ∃ (D : ℕ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 240 * g) ∧ D = 80 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_debt_resolved_l2047_204702


namespace NUMINAMATH_GPT_number_of_turns_l2047_204774

/-
  Given the cyclist's speed v = 5 m/s, time duration t = 5 s,
  and the circumference of the wheel c = 1.25 m, 
  prove that the number of complete turns n the wheel makes is equal to 20.
-/
theorem number_of_turns (v t c : ℝ) (h_v : v = 5) (h_t : t = 5) (h_c : c = 1.25) : 
  (v * t) / c = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_turns_l2047_204774


namespace NUMINAMATH_GPT_total_tickets_sold_l2047_204700

theorem total_tickets_sold :
  ∃(S : ℕ), 4 * S + 6 * 388 = 2876 ∧ S + 388 = 525 :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l2047_204700


namespace NUMINAMATH_GPT_odd_function_strictly_decreasing_inequality_solutions_l2047_204747

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom positive_for_neg_x (x : ℝ) : x < 0 → f x > 0

theorem odd_function : ∀ (x : ℝ), f (-x) = -f x := sorry

theorem strictly_decreasing : ∀ (x₁ x₂ : ℝ), x₁ > x₂ → f x₁ < f x₂ := sorry

theorem inequality_solutions (a x : ℝ) :
  (a = 0 ∧ false) ∨ 
  (a > 3 ∧ 3 < x ∧ x < a) ∨ 
  (a < 3 ∧ a < x ∧ x < 3) := sorry

end NUMINAMATH_GPT_odd_function_strictly_decreasing_inequality_solutions_l2047_204747


namespace NUMINAMATH_GPT_average_blinks_in_normal_conditions_l2047_204773

theorem average_blinks_in_normal_conditions (blink_gaming : ℕ) (k : ℚ) (blink_normal : ℚ) 
  (h_blink_gaming : blink_gaming = 10)
  (h_k : k = (3 / 5))
  (h_condition : blink_gaming = blink_normal - k * blink_normal) : 
  blink_normal = 25 := 
by 
  sorry

end NUMINAMATH_GPT_average_blinks_in_normal_conditions_l2047_204773


namespace NUMINAMATH_GPT_no_nonzero_integers_satisfy_conditions_l2047_204701

theorem no_nonzero_integers_satisfy_conditions :
  ¬ ∃ a b x y : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0) ∧ (a * x - b * y = 16) ∧ (a * y + b * x = 1) :=
by
  sorry

end NUMINAMATH_GPT_no_nonzero_integers_satisfy_conditions_l2047_204701
