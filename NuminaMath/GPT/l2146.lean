import Mathlib

namespace jim_travels_20_percent_of_jill_l2146_214650

def john_distance : ℕ := 15
def jill_travels_less : ℕ := 5
def jim_distance : ℕ := 2
def jill_distance : ℕ := john_distance - jill_travels_less

theorem jim_travels_20_percent_of_jill :
  (jim_distance * 100) / jill_distance = 20 := by
  sorry

end jim_travels_20_percent_of_jill_l2146_214650


namespace total_number_of_bills_received_l2146_214626

open Nat

-- Definitions based on the conditions:
def total_withdrawal_amount : ℕ := 600
def bill_denomination : ℕ := 20

-- Mathematically equivalent proof problem
theorem total_number_of_bills_received : (total_withdrawal_amount / bill_denomination) = 30 := 
by
  sorry

end total_number_of_bills_received_l2146_214626


namespace meal_cost_before_tax_and_tip_l2146_214689

theorem meal_cost_before_tax_and_tip (total_expenditure : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (base_meal_cost : ℝ):
  total_expenditure = 35.20 →
  tax_rate = 0.08 →
  tip_rate = 0.18 →
  base_meal_cost * (1 + tax_rate + tip_rate) = total_expenditure →
  base_meal_cost = 28 :=
by
  intros h_total h_tax h_tip h_eq
  sorry

end meal_cost_before_tax_and_tip_l2146_214689


namespace find_n_solution_l2146_214646

def product_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.prod

theorem find_n_solution : ∃ n : ℕ, n > 0 ∧ n^2 - 17 * n + 56 = product_of_digits n ∧ n = 4 := 
by
  sorry

end find_n_solution_l2146_214646


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l2146_214631

noncomputable def arithmetic_sequence (a n d : ℝ) : ℝ := 
  a + (n - 1) * d

noncomputable def geometric_sequence_sum (b1 r n : ℝ) : ℝ := 
  b1 * (1 - r^n) / (1 - r)

theorem arithmetic_sequence_general_formula (a1 d : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) : 
  ∀ n, arithmetic_sequence a1 n d = (n + 1) / 2 :=
by 
  sorry

theorem geometric_sequence_sum_first_n_terms (a1 d b1 b4 : ℝ) (h1 : a1 + 2 * d = 2) (h2 : 3 * a1 + 3 * d = 9 / 2) 
  (h3 : b1 = a1) (h4 : b4 = arithmetic_sequence a1 15 d) (h5 : b4 = 8) :
  ∀ n, geometric_sequence_sum b1 2 n = 2^n - 1 :=
by 
  sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_first_n_terms_l2146_214631


namespace irrational_sqrt3_l2146_214643

def is_irrational (x : ℝ) : Prop := ∀ (a b : ℤ), b ≠ 0 → x ≠ a / b

theorem irrational_sqrt3 :
  let A := 22 / 7
  let B := 0
  let C := Real.sqrt 3
  let D := 3.14
  is_irrational C :=
by
  sorry

end irrational_sqrt3_l2146_214643


namespace find_remainder_mod_105_l2146_214678

-- Define the conditions as a set of hypotheses
variables {n a b c : ℕ}
variables (hn : n > 0)
variables (ha : a < 3) (hb : b < 5) (hc : c < 7)
variables (h3 : n % 3 = a) (h5 : n % 5 = b) (h7 : n % 7 = c)
variables (heq : 4 * a + 3 * b + 2 * c = 30)

-- State the theorem
theorem find_remainder_mod_105 : n % 105 = 29 :=
by
  -- Hypotheses block for documentation
  have ha_le : 0 ≤ a := sorry
  have hb_le : 0 ≤ b := sorry
  have hc_le : 0 ≤ c := sorry
  sorry

end find_remainder_mod_105_l2146_214678


namespace set_complement_union_eq_l2146_214645

open Set

variable (U : Set ℕ) (P : Set ℕ) (Q : Set ℕ)

theorem set_complement_union_eq :
  U = {1, 2, 3, 4, 5, 6} →
  P = {1, 3, 5} →
  Q = {1, 2, 4} →
  (U \ P) ∪ Q = {1, 2, 4, 6} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end set_complement_union_eq_l2146_214645


namespace empire_state_building_height_l2146_214670

theorem empire_state_building_height (h_top_floor : ℕ) (h_antenna_spire : ℕ) (total_height : ℕ) :
  h_top_floor = 1250 ∧ h_antenna_spire = 204 ∧ total_height = h_top_floor + h_antenna_spire → total_height = 1454 :=
by
  sorry

end empire_state_building_height_l2146_214670


namespace ants_rice_transport_l2146_214617

/-- 
Given:
  1) 12 ants can move 24 grains of rice in 6 trips.

Prove:
  How many grains of rice can 9 ants move in 9 trips?
-/
theorem ants_rice_transport :
  (9 * 9 * (24 / (12 * 6))) = 27 := 
sorry

end ants_rice_transport_l2146_214617


namespace weight_left_after_two_deliveries_l2146_214692

-- Definitions and conditions
def initial_load : ℝ := 50000
def first_store_percentage : ℝ := 0.1
def second_store_percentage : ℝ := 0.2

-- The statement to be proven
theorem weight_left_after_two_deliveries :
  let weight_after_first_store := initial_load * (1 - first_store_percentage)
  let weight_after_second_store := weight_after_first_store * (1 - second_store_percentage)
  weight_after_second_store = 36000 :=
by sorry  -- Proof omitted

end weight_left_after_two_deliveries_l2146_214692


namespace emily_quiz_score_l2146_214630

theorem emily_quiz_score :
  ∃ x : ℕ, 94 + 88 + 92 + 85 + 97 + x = 6 * 90 :=
by
  sorry

end emily_quiz_score_l2146_214630


namespace find_x_l2146_214623

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 6) : x = 14 :=
by
  sorry

end find_x_l2146_214623


namespace not_lucky_1994_l2146_214657

def is_valid_month (m : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12

def is_valid_day (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 31

def is_lucky_year (y : ℕ) : Prop :=
  ∃ (m d : ℕ), is_valid_month m ∧ is_valid_day d ∧ m * d = y

theorem not_lucky_1994 : ¬ is_lucky_year 94 := 
by
  sorry

end not_lucky_1994_l2146_214657


namespace cookies_left_after_week_l2146_214636

theorem cookies_left_after_week (cookies_in_jar : ℕ) (total_taken_out_in_4_days : ℕ) (same_amount_each_day : Prop)
  (h1 : cookies_in_jar = 70) (h2 : total_taken_out_in_4_days = 24) :
  ∃ (cookies_left : ℕ), cookies_left = 28 :=
by
  sorry

end cookies_left_after_week_l2146_214636


namespace incorrect_locus_proof_l2146_214615

-- Conditions given in the problem
def condition_A (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus ↔ conditions p)

def condition_B (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (conditions p ↔ p ∈ locus)

def condition_C (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∈ locus → conditions p) ∧ (∃ q, conditions q ∧ q ∈ locus)

def condition_D (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (p ∉ locus ↔ ¬ conditions p) ∧ (p ∈ locus ↔ conditions p)

def condition_E (locus : Set Point) (conditions : Point → Prop) :=
  ∀ p, (conditions p ↔ p ∈ locus) ∧ (¬ conditions p ↔ p ∉ locus)

-- Statement to be proved
theorem incorrect_locus_proof (locus : Set Point) (conditions : Point → Prop) :
  ¬ condition_C locus conditions :=
sorry

end incorrect_locus_proof_l2146_214615


namespace number_of_candidates_is_9_l2146_214698

-- Defining the problem
def num_ways_to_select_president_and_vp (n : ℕ) : ℕ :=
  n * (n - 1)

-- Main theorem statement
theorem number_of_candidates_is_9 (n : ℕ) (h : num_ways_to_select_president_and_vp n = 72) : n = 9 :=
by
  sorry

end number_of_candidates_is_9_l2146_214698


namespace equal_intercepts_l2146_214611

theorem equal_intercepts (a : ℝ) (h : ∃ (x y : ℝ), (x = (2 + a) / a ∧ y = 2 + a ∧ x = y)) :
  a = -2 ∨ a = 1 :=
by sorry

end equal_intercepts_l2146_214611


namespace find_other_endpoint_l2146_214654

set_option pp.funBinderTypes true

def circle_center : (ℝ × ℝ) := (5, -2)
def diameter_endpoint1 : (ℝ × ℝ) := (1, 2)
def diameter_endpoint2 : (ℝ × ℝ) := (9, -6)

theorem find_other_endpoint (c : ℝ × ℝ) (e1 : ℝ × ℝ) (e2 : ℝ × ℝ) : 
  c = circle_center ∧ e1 = diameter_endpoint1 → e2 = diameter_endpoint2 := by
  sorry

end find_other_endpoint_l2146_214654


namespace ratio_of_roots_ratio_l2146_214669

noncomputable def sum_roots_first_eq (a b c : ℝ) := b / a
noncomputable def product_roots_first_eq (a b c : ℝ) := c / a
noncomputable def sum_roots_second_eq (a b c : ℝ) := a / c
noncomputable def product_roots_second_eq (a b c : ℝ) := b / c

theorem ratio_of_roots_ratio (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c ≠ 0)
  (h3 : (b ^ 2 - 4 * a * c) > 0)
  (h4 : (a ^ 2 - 4 * c * b) > 0)
  (h5 : sum_roots_first_eq a b c ≥ 0)
  (h6 : product_roots_first_eq a b c = 9 * sum_roots_second_eq a b c) :
  sum_roots_first_eq a b c / product_roots_second_eq a b c = -3 :=
sorry

end ratio_of_roots_ratio_l2146_214669


namespace Tim_younger_than_Jenny_l2146_214604

def Tim_age : Nat := 5
def Rommel_age (T : Nat) : Nat := 3 * T
def Jenny_age (R : Nat) : Nat := R + 2

theorem Tim_younger_than_Jenny :
  let T := Tim_age
  let R := Rommel_age T
  let J := Jenny_age R
  J - T = 12 :=
by
  sorry

end Tim_younger_than_Jenny_l2146_214604


namespace total_arrangements_l2146_214664

def total_members : ℕ := 6
def days : ℕ := 3
def people_per_day : ℕ := 2

def A_cannot_on_14 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 14 = 1

def B_cannot_on_16 (arrangement : ℕ → ℕ) : Prop :=
  ¬ arrangement 16 = 2

theorem total_arrangements (arrangement : ℕ → ℕ) :
  (∀ arrangement, A_cannot_on_14 arrangement ∧ B_cannot_on_16 arrangement) →
  (total_members.choose 2 * (total_members - 2).choose 2 - 
  2 * (total_members - 1).choose 1 * (total_members - 2).choose 2 +
  (total_members - 2).choose 1 * (total_members - 3).choose 1)
  = 42 := 
by
  sorry

end total_arrangements_l2146_214664


namespace initial_gummy_worms_l2146_214653

variable (G : ℕ)

theorem initial_gummy_worms (h : (G : ℚ) / 16 = 4) : G = 64 :=
by
  sorry

end initial_gummy_worms_l2146_214653


namespace burger_share_per_person_l2146_214635

-- Definitions based on conditions
def foot_to_inches : ℕ := 12
def burger_length_foot : ℕ := 1
def burger_length_inches : ℕ := burger_length_foot * foot_to_inches

theorem burger_share_per_person : (burger_length_inches / 2) = 6 := by
  sorry

end burger_share_per_person_l2146_214635


namespace cosine_double_angle_identity_l2146_214624

theorem cosine_double_angle_identity (α : ℝ) (h : Real.sin (α + 7 * Real.pi / 6) = 1) :
  Real.cos (2 * α - 2 * Real.pi / 3) = 1 := by
  sorry

end cosine_double_angle_identity_l2146_214624


namespace intersection_M_N_l2146_214622

noncomputable def M : Set ℝ := { x | x^2 ≤ x }
noncomputable def N : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_M_N :
  M ∩ N = { x | 0 < x ∧ x ≤ 1 } :=
  sorry

end intersection_M_N_l2146_214622


namespace part1_part2_l2146_214619

/-
Part 1: Given the conditions of parabola and line intersection, prove the range of slope k of the line.
-/
theorem part1 (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  k > -2 + 2 * Real.sqrt 2 ∨ k < -2 - 2 * Real.sqrt 2 :=
  sorry

/-
Part 2: Given the conditions of locus of point Q on the line segment P1P2, prove the equation of the locus.
-/
theorem part2 (x y : ℝ) (k : ℝ) (h1 : ∀ x, y = x^2) (h2 : ∀ x, y = k * (x + 1) - 1) :
  2 * x - y + 1 = 0 ∧ (-Real.sqrt 2 - 1 < x ∧ x < Real.sqrt 2 - 1 ∧ x ≠ -1) :=
  sorry

end part1_part2_l2146_214619


namespace geometric_sequence_a6_a8_sum_l2146_214627

theorem geometric_sequence_a6_a8_sum 
  (a : ℕ → ℕ) (q : ℕ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 1 + a 3 = 5)
  (h2 : a 2 + a 4 = 10) : 
  a 6 + a 8 = 160 := 
sorry

end geometric_sequence_a6_a8_sum_l2146_214627


namespace seven_by_seven_grid_partition_l2146_214672

theorem seven_by_seven_grid_partition : 
  ∀ (x y : ℕ), 4 * x + 3 * y = 49 ∧ x + y ≥ 16 → x = 1 :=
by sorry

end seven_by_seven_grid_partition_l2146_214672


namespace binary_sum_correct_l2146_214607

-- Definitions of the binary numbers
def bin1 : ℕ := 0b1011
def bin2 : ℕ := 0b101
def bin3 : ℕ := 0b11001
def bin4 : ℕ := 0b1110
def bin5 : ℕ := 0b100101

-- The statement to prove
theorem binary_sum_correct : bin1 + bin2 + bin3 + bin4 + bin5 = 0b1111010 := by
  sorry

end binary_sum_correct_l2146_214607


namespace find_a_b_l2146_214687

theorem find_a_b (a b : ℤ) : (∀ (s : ℂ), s^2 + s - 1 = 0 → a * s^18 + b * s^17 + 1 = 0) → (a = 987 ∧ b = -1597) :=
by
  sorry

end find_a_b_l2146_214687


namespace magician_earned_4_dollars_l2146_214679

-- Define conditions
def price_per_deck := 2
def initial_decks := 5
def decks_left := 3

-- Define the number of decks sold
def decks_sold := initial_decks - decks_left

-- Define the total money earned
def money_earned := decks_sold * price_per_deck

-- Theorem to prove the money earned is 4 dollars
theorem magician_earned_4_dollars : money_earned = 4 := by
  sorry

end magician_earned_4_dollars_l2146_214679


namespace polynomial_sum_l2146_214674

noncomputable def p (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
noncomputable def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
noncomputable def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = 12 * x - 12 := by
  sorry

end polynomial_sum_l2146_214674


namespace quiz_total_points_l2146_214660

theorem quiz_total_points (points : ℕ → ℕ) 
  (h1 : ∀ n, points (n+1) = points n + 4)
  (h2 : points 2 = 39) : 
  (points 0 + points 1 + points 2 + points 3 + points 4 + points 5 + points 6 + points 7) = 360 :=
sorry

end quiz_total_points_l2146_214660


namespace range_of_a_l2146_214602

noncomputable def p (a : ℝ) : Prop :=
∀ (x : ℝ), x > -1 → (x^2) / (x + 1) ≥ a

noncomputable def q (a : ℝ) : Prop :=
∃ (x : ℝ), (a*x^2 - a*x + 1 = 0)

theorem range_of_a (a : ℝ) :
  ¬ p a ∧ ¬ q a ∧ (p a ∨ q a) ↔ (a = 0 ∨ a ≥ 4) :=
by sorry

end range_of_a_l2146_214602


namespace each_friend_gets_four_pieces_l2146_214633

noncomputable def pieces_per_friend : ℕ :=
  let oranges := 80
  let pieces_per_orange := 10
  let friends := 200
  (oranges * pieces_per_orange) / friends

theorem each_friend_gets_four_pieces :
  pieces_per_friend = 4 :=
by
  sorry

end each_friend_gets_four_pieces_l2146_214633


namespace exists_integers_m_n_for_inequalities_l2146_214639

theorem exists_integers_m_n_for_inequalities (a b : ℝ) (h : a ≠ b) : ∃ (m n : ℤ), 
  (a * (m : ℝ) + b * (n : ℝ) < 0) ∧ (b * (m : ℝ) + a * (n : ℝ) > 0) :=
sorry

end exists_integers_m_n_for_inequalities_l2146_214639


namespace area_of_circle_l2146_214642

theorem area_of_circle :
  (∃ (x y : ℝ), x^2 + y^2 - 6 * x + 8 * y = -9) →
  ∃ A : ℝ, A = 16 * Real.pi :=
by
  -- We need to prove the area is 16π
  sorry

end area_of_circle_l2146_214642


namespace soccer_team_games_count_l2146_214610

variable (total_games won_games : ℕ)
variable (h1 : won_games = 70)
variable (h2 : won_games = total_games / 2)

theorem soccer_team_games_count : total_games = 140 :=
by
  -- Proof goes here
  sorry

end soccer_team_games_count_l2146_214610


namespace find_number_l2146_214658

theorem find_number (x : ℝ) (hx : (50 + 20 / x) * x = 4520) : x = 90 :=
sorry

end find_number_l2146_214658


namespace statement_C_l2146_214649

variables (a b c d : ℝ)

theorem statement_C (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by sorry

end statement_C_l2146_214649


namespace express_in_scientific_notation_l2146_214632

theorem express_in_scientific_notation (x : ℝ) (h : x = 720000) : x = 7.2 * 10^5 :=
by sorry

end express_in_scientific_notation_l2146_214632


namespace calc_expression1_calc_expression2_l2146_214651

theorem calc_expression1 : (1 / 3)^0 + Real.sqrt 27 - abs (-3) + Real.tan (Real.pi / 4) = 1 + 3 * Real.sqrt 3 - 2 :=
by
  sorry

theorem calc_expression2 (x : ℝ) : (x + 2)^2 - 2 * (x - 1) = x^2 + 2 * x + 6 :=
by
  sorry

end calc_expression1_calc_expression2_l2146_214651


namespace at_least_fifty_same_leading_coefficient_l2146_214620

-- Define what it means for two quadratic polynomials to intersect exactly once
def intersect_once (P Q : Polynomial ℝ) : Prop :=
∃ x, P.eval x = Q.eval x ∧ ∀ y ≠ x, P.eval y ≠ Q.eval y

-- Define the main theorem and its conditions
theorem at_least_fifty_same_leading_coefficient 
  (polynomials : Fin 100 → Polynomial ℝ)
  (h1 : ∀ i j, i ≠ j → intersect_once (polynomials i) (polynomials j))
  (h2 : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
        ¬∃ x, (polynomials i).eval x = (polynomials j).eval x ∧ (polynomials j).eval x = (polynomials k).eval x) : 
  ∃ (S : Finset (Fin 100)), S.card ≥ 50 ∧ ∃ a, ∀ i ∈ S, (polynomials i).leadingCoeff = a :=
sorry

end at_least_fifty_same_leading_coefficient_l2146_214620


namespace chebyshev_substitution_even_chebyshev_substitution_odd_l2146_214675

def T (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the first kind
def U (n : ℕ) (x : ℝ) : ℝ := sorry -- Chebyshev polynomial of the second kind

theorem chebyshev_substitution_even (k : ℕ) (α : ℝ) :
  T (2 * k) (Real.sin α) = (-1)^k * Real.cos ((2 * k) * α) ∧
  U ((2 * k) - 1) (Real.sin α) = (-1)^(k + 1) * (Real.sin ((2 * k) * α) / Real.cos α) :=
by
  sorry

theorem chebyshev_substitution_odd (k : ℕ) (α : ℝ) :
  T (2 * k + 1) (Real.sin α) = (-1)^k * Real.sin ((2 * k + 1) * α) ∧
  U (2 * k) (Real.sin α) = (-1)^k * (Real.cos ((2 * k + 1) * α) / Real.cos α) :=
by
  sorry

end chebyshev_substitution_even_chebyshev_substitution_odd_l2146_214675


namespace a_values_unique_solution_l2146_214644

theorem a_values_unique_solution :
  (∀ a : ℝ, ∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) →
  (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end a_values_unique_solution_l2146_214644


namespace production_days_l2146_214668

theorem production_days (n : ℕ) 
    (h1 : 70 * n + 90 = 75 * (n + 1)) : n = 3 := 
sorry

end production_days_l2146_214668


namespace max_S_is_9_l2146_214662

-- Definitions based on the conditions
def a (n : ℕ) : ℤ := 28 - 3 * n
def S (n : ℕ) : ℤ := n * (25 + a n) / 2

-- The theorem to be proved
theorem max_S_is_9 : ∃ n : ℕ, n = 9 ∧ S n = 117 :=
by
  sorry

end max_S_is_9_l2146_214662


namespace new_average_income_l2146_214671

/-!
# Average Monthly Income Problem

## Problem Statement
Given:
1. The average monthly income of a family of 4 earning members was Rs. 735.
2. One of the earning members died, and the average income changed.
3. The income of the deceased member was Rs. 1170.

Prove that the new average monthly income of the family is Rs. 590.
-/

theorem new_average_income (avg_income : ℝ) (num_members : ℕ) (income_deceased : ℝ) (new_num_members : ℕ) 
  (h1 : avg_income = 735) 
  (h2 : num_members = 4) 
  (h3 : income_deceased = 1170) 
  (h4 : new_num_members = 3) : 
  (num_members * avg_income - income_deceased) / new_num_members = 590 := 
by 
  sorry

end new_average_income_l2146_214671


namespace angle_A_is_pi_div_3_length_b_l2146_214681

open Real

theorem angle_A_is_pi_div_3
  (A B C : ℝ) (a b c : ℝ)
  (hABC : A + B + C = π)
  (m : ℝ × ℝ) (n : ℝ × ℝ)
  (hm : m = (sqrt 3, cos (π - A) - 1))
  (hn : n = (cos (π / 2 - A), 1))
  (horthogonal : m.1 * n.1 + m.2 * n.2 = 0) :
  A = π / 3 := 
sorry

theorem length_b 
  (A B : ℝ) (a b : ℝ)
  (hA : A = π / 3)
  (ha : a = 2)
  (hcosB : cos B = sqrt 3 / 3) :
  b = 4 * sqrt 2 / 3 :=
sorry

end angle_A_is_pi_div_3_length_b_l2146_214681


namespace diagonal_less_than_half_perimeter_l2146_214656

theorem diagonal_less_than_half_perimeter (a b c d x : ℝ) 
  (h1 : x < a + b) (h2 : x < c + d) : x < (a + b + c + d) / 2 := 
by
  sorry

end diagonal_less_than_half_perimeter_l2146_214656


namespace simplify_expression_l2146_214696

variable (x y : ℝ)

theorem simplify_expression : (-(3 * x * y - 2 * x ^ 2) - 2 * (3 * x ^ 2 - x * y)) = (-4 * x ^ 2 - x * y) :=
by
  sorry

end simplify_expression_l2146_214696


namespace normal_price_of_article_l2146_214621

theorem normal_price_of_article 
  (P : ℝ) 
  (h : (P * 0.88 * 0.78 * 0.85) * 1.06 = 144) : 
  P = 144 / (0.88 * 0.78 * 0.85 * 1.06) :=
sorry

end normal_price_of_article_l2146_214621


namespace eval_g_six_times_at_2_l2146_214647

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem eval_g_six_times_at_2 : g (g (g (g (g (g 2))))) = 4 := sorry

end eval_g_six_times_at_2_l2146_214647


namespace purchasing_plans_count_l2146_214612

theorem purchasing_plans_count :
  ∃ n : ℕ, n = 2 ∧ (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = 35) :=
sorry

end purchasing_plans_count_l2146_214612


namespace arithmetic_mean_of_two_digit_multiples_of_5_l2146_214629

theorem arithmetic_mean_of_two_digit_multiples_of_5:
  let smallest := 10
  let largest := 95
  let num_terms := 18
  let sum := 945
  let mean := (sum : ℝ) / (num_terms : ℝ)
  mean = 52.5 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_5_l2146_214629


namespace g_value_l2146_214695

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def g (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) - 1

theorem g_value (ω φ : ℝ) (h : ∀ x : ℝ, f ω φ (π / 4 - x) = f ω φ (π / 4 + x)) :
  g ω φ (π / 4) = -1 :=
sorry

end g_value_l2146_214695


namespace complement_of_angle_A_l2146_214667

theorem complement_of_angle_A (A : ℝ) (h : A = 76) : 90 - A = 14 := by
  sorry

end complement_of_angle_A_l2146_214667


namespace intersection_x_value_l2146_214641

theorem intersection_x_value :
  (∃ x y : ℝ, y = 5 * x - 20 ∧ y = 110 - 3 * x ∧ x = 16.25) := sorry

end intersection_x_value_l2146_214641


namespace expr_C_always_positive_l2146_214608

-- Define the expressions as Lean definitions
def expr_A (x : ℝ) : ℝ := x^2
def expr_B (x : ℝ) : ℝ := abs (-x + 1)
def expr_C (x : ℝ) : ℝ := (-x)^2 + 2
def expr_D (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem expr_C_always_positive : ∀ (x : ℝ), expr_C x > 0 :=
by
  sorry

end expr_C_always_positive_l2146_214608


namespace r_squared_sum_l2146_214680

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l2146_214680


namespace glucose_solution_l2146_214655

theorem glucose_solution (x : ℝ) (h : (15 / 100 : ℝ) = (6.75 / x)) : x = 45 :=
sorry

end glucose_solution_l2146_214655


namespace Petya_cannot_achieve_goal_l2146_214628

theorem Petya_cannot_achieve_goal (n : ℕ) (h : n ≥ 2) :
  ¬ (∃ (G : ℕ → Prop), (∀ i : ℕ, (G i ↔ (G ((i + 2) % (2 * n))))) ∨ (G (i + 1) ≠ G (i + 2))) :=
sorry

end Petya_cannot_achieve_goal_l2146_214628


namespace annie_passes_bonnie_first_l2146_214691

def bonnie_speed (v : ℝ) := v
def annie_speed (v : ℝ) := 1.3 * v
def track_length := 500

theorem annie_passes_bonnie_first (v t : ℝ) (ht : 0.3 * v * t = track_length) : 
  (annie_speed v * t) / track_length = 4 + 1 / 3 :=
by 
  sorry

end annie_passes_bonnie_first_l2146_214691


namespace solve_system_l2146_214652

-- Definitions for the system of equations.
def system_valid (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

-- Main theorem to prove.
theorem solve_system (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : 
  system_valid y x₁ x₂ x₃ x₄ x₅ →
  ((y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨ 
  (y = 2 → ∃ (t : ℝ), x₁ = t ∧ x₂ = t ∧ x₃ = t ∧ x₄ = t ∧ x₅ = t) ∨ 
  (y^2 + y - 1 = 0 → ∃ (u v : ℝ), 
    x₁ = u ∧ 
    x₅ = v ∧ 
    x₂ = y * u - v ∧ 
    x₃ = -y * (u + v) ∧ 
    x₄ = y * v - u ∧ 
    (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2))) :=
by
  intro h
  sorry

end solve_system_l2146_214652


namespace volume_ratio_of_smaller_snowball_l2146_214600

theorem volume_ratio_of_smaller_snowball (r : ℝ) (k : ℝ) :
  let V₀ := (4/3) * π * r^3
  let S := 4 * π * r^2
  let V_large := (4/3) * π * (2 * r)^3
  let V_large_half := V_large / 2
  let new_r := (V_large_half / ((4/3) * π))^(1/3)
  let reduction := 2*r - new_r
  let remaining_r := r - reduction
  let remaining_V := (4/3) * π * remaining_r^3
  let volume_ratio := remaining_V / V₀ 
  volume_ratio = 1/5 :=
by
  -- Proof goes here
  sorry

end volume_ratio_of_smaller_snowball_l2146_214600


namespace allan_has_4_more_balloons_than_jake_l2146_214693

namespace BalloonProblem

def initial_balloons_allan : Nat := 6
def initial_balloons_jake : Nat := 2
def additional_balloons_jake : Nat := 3
def additional_balloons_allan : Nat := 4
def given_balloons_jake : Nat := 2
def given_balloons_allan : Nat := 3

def final_balloons_allan : Nat := (initial_balloons_allan + additional_balloons_allan) - given_balloons_allan
def final_balloons_jake : Nat := (initial_balloons_jake + additional_balloons_jake) - given_balloons_jake

theorem allan_has_4_more_balloons_than_jake :
  final_balloons_allan = final_balloons_jake + 4 :=
by
  -- proof is skipped with sorry
  sorry

end BalloonProblem

end allan_has_4_more_balloons_than_jake_l2146_214693


namespace measure_of_angle_C_l2146_214659

theorem measure_of_angle_C (A B C : ℕ) (h1 : A + B = 150) (h2 : A + B + C = 180) : C = 30 := 
by
  sorry

end measure_of_angle_C_l2146_214659


namespace triangle_angle_tangent_condition_l2146_214637

theorem triangle_angle_tangent_condition
  (A B C : ℝ)
  (h1 : A + C = 2 * B)
  (h2 : Real.tan A * Real.tan C = 2 + Real.sqrt 3) :
  (A = Real.pi / 4 ∧ B = Real.pi / 3 ∧ C = 5 * Real.pi / 12) ∨
  (A = 5 * Real.pi / 12 ∧ B = Real.pi / 3 ∧ C = Real.pi / 4) :=
  sorry

end triangle_angle_tangent_condition_l2146_214637


namespace concurrent_segments_unique_solution_l2146_214603

theorem concurrent_segments_unique_solution (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  4^c - 1 = (2^a - 1) * (2^b - 1) ↔ (a = 1 ∧ b = 2 * c) ∨ (a = 2 * c ∧ b = 1) :=
by
  sorry

end concurrent_segments_unique_solution_l2146_214603


namespace no_month_5_mondays_and_5_thursdays_l2146_214665

theorem no_month_5_mondays_and_5_thursdays (n : ℕ) (h : n = 28 ∨ n = 29 ∨ n = 30 ∨ n = 31) :
  ¬ (∃ (m : ℕ) (t : ℕ), m = 5 ∧ t = 5 ∧ 5 * (m + t) ≤ n) := by sorry

end no_month_5_mondays_and_5_thursdays_l2146_214665


namespace radius_of_sphere_find_x_for_equation_l2146_214606

-- Problem I2.1
theorem radius_of_sphere (r : ℝ) (V : ℝ) (h : V = 36 * π) : r = 3 :=
sorry

-- Problem I2.2
theorem find_x_for_equation (x : ℝ) (r : ℝ) (h_r : r = 3) (h : r^x + r^(1-x) = 4) (h_x_pos : x > 0) : x = 1 :=
sorry

end radius_of_sphere_find_x_for_equation_l2146_214606


namespace necessary_but_not_sufficient_l2146_214683

-- Definitions extracted from the problem conditions
def isEllipse (k : ℝ) : Prop := (9 - k > 0) ∧ (k - 7 > 0) ∧ (9 - k ≠ k - 7)

-- The necessary but not sufficient condition for the ellipse equation
theorem necessary_but_not_sufficient : 
  (7 < k ∧ k < 9) → isEllipse k → (isEllipse k ↔ (7 < k ∧ k < 9)) := 
by 
  sorry

end necessary_but_not_sufficient_l2146_214683


namespace min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l2146_214697

-- Define the first problem
theorem min_cuts_for_eleven_day_stay : 
  (∀ (chain_len num_days : ℕ), chain_len = 11 ∧ num_days = 11 
  → (∃ (cuts : ℕ), cuts = 2)) := 
sorry

-- Define the second problem
theorem max_days_with_n_cuts : 
  (∀ (n chain_len days : ℕ), chain_len = (n + 1) * 2 ^ n - 1 
  → days = (n + 1) * 2 ^ n - 1) := 
sorry

end min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l2146_214697


namespace man_l2146_214661

theorem man's_speed_upstream (v : ℝ) (downstream_speed : ℝ) (stream_speed : ℝ) :
  downstream_speed = v + stream_speed → stream_speed = 1 → downstream_speed = 10 → v - stream_speed = 8 :=
by
  intros h1 h2 h3
  sorry

end man_l2146_214661


namespace total_age_l2146_214648

variable (A B : ℝ)

-- Conditions
def condition1 : Prop := A / B = 3 / 4
def condition2 : Prop := A - 10 = (1 / 2) * (B - 10)

-- Statement
theorem total_age : condition1 A B → condition2 A B → A + B = 35 := by
  sorry

end total_age_l2146_214648


namespace abc_plus_2p_zero_l2146_214684

variable (a b c p : ℝ)

-- Define the conditions
def cond1 : Prop := a + 2 / b = p
def cond2 : Prop := b + 2 / c = p
def cond3 : Prop := c + 2 / a = p
def nonzero_and_distinct : Prop := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- The main statement we want to prove
theorem abc_plus_2p_zero (h1 : cond1 a b p) (h2 : cond2 b c p) (h3 : cond3 c a p) (h4 : nonzero_and_distinct a b c) : 
  a * b * c + 2 * p = 0 := 
by 
  sorry

end abc_plus_2p_zero_l2146_214684


namespace larger_number_value_l2146_214682

theorem larger_number_value (L S : ℕ) (h1 : L - S = 20775) (h2 : L = 23 * S + 143) : L = 21713 :=
sorry

end larger_number_value_l2146_214682


namespace count_triangles_in_3x3_grid_l2146_214601

/--
In a 3x3 grid of dots, the number of triangles formed by connecting the dots is 20.
-/
def triangles_in_3x3_grid : Prop :=
  let num_rows := 3
  let num_cols := 3
  let total_triangles := 20
  ∃ (n : ℕ), n = total_triangles ∧ n = 20

theorem count_triangles_in_3x3_grid : triangles_in_3x3_grid :=
by {
  -- Insert the proof here
  sorry
}

end count_triangles_in_3x3_grid_l2146_214601


namespace therapy_hours_l2146_214694

variable (F A n : ℕ)
variable (h1 : F = A + 20)
variable (h2 : F + 2 * A = 188)
variable (h3 : F + A * (n - 1) = 300)

theorem therapy_hours : n = 5 := by
  sorry

end therapy_hours_l2146_214694


namespace evaluate_expression_l2146_214686

theorem evaluate_expression (x : ℝ) : 
  (36 + 12 * x) ^ 2 - (12^2 * x^2 + 36^2) = 864 * x :=
by
  sorry

end evaluate_expression_l2146_214686


namespace definite_integral_ln_squared_l2146_214638

noncomputable def integralFun : ℝ → ℝ := λ x => x * (Real.log x) ^ 2

theorem definite_integral_ln_squared (f : ℝ → ℝ) (a b : ℝ):
  (f = integralFun) → 
  (a = 1) → 
  (b = 2) → 
  ∫ x in a..b, f x = 2 * (Real.log 2) ^ 2 - 2 * Real.log 2 + 3 / 4 :=
by
  intros hfa hao hbo
  rw [hfa, hao, hbo]
  sorry

end definite_integral_ln_squared_l2146_214638


namespace volume_of_water_in_prism_l2146_214673

-- Define the given dimensions and conditions
def length_x := 20 -- cm
def length_y := 30 -- cm
def length_z := 40 -- cm
def angle := 30 -- degrees
def total_volume := 24 -- liters

-- The wet fraction of the upper surface
def wet_fraction := 1 / 4

-- Correct answer to be proven
def volume_water := 18.8 -- liters

theorem volume_of_water_in_prism :
  -- Given the conditions
  (length_x = 20) ∧ (length_y = 30) ∧ (length_z = 40) ∧ (angle = 30) ∧ (wet_fraction = 1 / 4) ∧ (total_volume = 24) →
  -- Prove that the volume of water is as calculated
  volume_water = 18.8 :=
sorry

end volume_of_water_in_prism_l2146_214673


namespace functional_eqn_even_function_l2146_214677

variable {R : Type*} [AddGroup R] (f : R → ℝ)

theorem functional_eqn_even_function
  (h_not_zero : ∃ x, f x ≠ 0)
  (h_func_eq : ∀ a b, f (a + b) + f (a - b) = 2 * f a + 2 * f b) :
  ∀ x, f (-x) = f x :=
by
  sorry

end functional_eqn_even_function_l2146_214677


namespace rectangle_area_l2146_214688

variable (w l : ℕ)
variable (A : ℕ)
variable (H1 : l = 5 * w)
variable (H2 : 2 * l + 2 * w = 180)

theorem rectangle_area : A = 1125 :=
by
  sorry

end rectangle_area_l2146_214688


namespace two_person_subcommittees_l2146_214663

def committee_size := 8
def sub_committee_size := 2
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem two_person_subcommittees : combination committee_size sub_committee_size = 28 := by
  sorry

end two_person_subcommittees_l2146_214663


namespace complex_powers_i_l2146_214634

theorem complex_powers_i (i : ℂ) (h : i^2 = -1) :
  (i^123 - i^321 + i^432 = -2 * i + 1) :=
by
  -- sorry to skip the proof
  sorry

end complex_powers_i_l2146_214634


namespace sin_double_angle_l2146_214618

theorem sin_double_angle (alpha : ℝ) (h1 : Real.cos (alpha + π / 4) = 3 / 5)
  (h2 : π / 2 ≤ alpha ∧ alpha ≤ 3 * π / 2) : Real.sin (2 * alpha) = 7 / 25 := 
sorry

end sin_double_angle_l2146_214618


namespace sam_read_pages_l2146_214625

-- Define conditions
def assigned_pages : ℕ := 25
def harrison_pages : ℕ := assigned_pages + 10
def pam_pages : ℕ := harrison_pages + 15
def sam_pages : ℕ := 2 * pam_pages

-- Prove the target theorem
theorem sam_read_pages : sam_pages = 100 := by
  sorry

end sam_read_pages_l2146_214625


namespace smallest_angle_equilateral_triangle_l2146_214613

-- Definitions corresponding to the conditions
structure EquilateralTriangle :=
(vertices : Fin 3 → ℝ × ℝ)
(equilateral : ∀ i j, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))

def point_on_line_segment (p1 p2 : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
((1 - t) * p1.1 + t * p2.1, (1 - t) * p1.2 + t * p2.2)

-- Given an equilateral triangle ABC with vertices A, B, C,
-- and points D on AB, E on AC, D1 on BC, and E1 on BC,
-- such that AB = DB + BD_1 and AC = CE + CE_1,
-- prove the smallest angle between DE_1 and ED_1 is 60 degrees.

theorem smallest_angle_equilateral_triangle
  (ABC : EquilateralTriangle)
  (A B C D E D₁ E₁ : ℝ × ℝ)
  (on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = point_on_line_segment A B t)
  (on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = point_on_line_segment A C t)
  (on_BC : ∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ D₁ = point_on_line_segment B C t₁ ∧
                         0 ≤ t₂ ∧ t₂ ≤ 1 ∧ E₁ = point_on_line_segment B C t₂)
  (AB_property : dist A B = dist D B + dist B D₁)
  (AC_property : dist A C = dist E C + dist C E₁) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 60 ∧ θ = 60 :=
sorry

end smallest_angle_equilateral_triangle_l2146_214613


namespace smallest_t_satisfies_equation_l2146_214609

def satisfies_equation (t x y : ℤ) : Prop :=
  (x^2 + y^2)^2 + 2 * t * x * (x^2 + y^2) = t^2 * y^2

theorem smallest_t_satisfies_equation : ∃ t x y : ℤ, t > 0 ∧ x > 0 ∧ y > 0 ∧ satisfies_equation t x y ∧
  ∀ t' x' y' : ℤ, t' > 0 ∧ x' > 0 ∧ y' > 0 ∧ satisfies_equation t' x' y' → t' ≥ t :=
sorry

end smallest_t_satisfies_equation_l2146_214609


namespace kelsey_more_than_ekon_l2146_214605

theorem kelsey_more_than_ekon :
  ∃ (K E U : ℕ), (K = 160) ∧ (E = U - 17) ∧ (K + E + U = 411) ∧ (K - E = 43) :=
by
  sorry

end kelsey_more_than_ekon_l2146_214605


namespace parallel_lines_m_l2146_214666

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, x + 2 * m * y - 1 = 0 → (3 * m - 1) * x - m * y - 1 = 0)
  → m = 0 ∨ m = 1 / 6 := 
sorry

end parallel_lines_m_l2146_214666


namespace james_total_earnings_l2146_214614

-- Define the earnings for January
def januaryEarnings : ℕ := 4000

-- Define the earnings for February based on January
def februaryEarnings : ℕ := 2 * januaryEarnings

-- Define the earnings for March based on February
def marchEarnings : ℕ := februaryEarnings - 2000

-- Define the total earnings including January, February, and March
def totalEarnings : ℕ := januaryEarnings + februaryEarnings + marchEarnings

-- State the theorem: total earnings should be 18000
theorem james_total_earnings : totalEarnings = 18000 := by
  sorry

end james_total_earnings_l2146_214614


namespace proof_combination_l2146_214699

open Classical

theorem proof_combination :
  (∃ x : ℝ, x^3 < 1) ∧ (¬ ∃ x : ℚ, x^2 = 2) ∧ (¬ ∀ x : ℕ, x^3 > x^2) ∧ (∀ x : ℝ, x^2 + 1 > 0) :=
by
  have h1 : ∃ x : ℝ, x^3 < 1 := sorry
  have h2 : ¬ ∃ x : ℚ, x^2 = 2 := sorry
  have h3 : ¬ ∀ x : ℕ, x^3 > x^2 := sorry
  have h4 : ∀ x : ℝ, x^2 + 1 > 0 := sorry
  exact ⟨h1, h2, h3, h4⟩

end proof_combination_l2146_214699


namespace cannonball_maximum_height_l2146_214640

def height_function (t : ℝ) := -20 * t^2 + 100 * t + 36

theorem cannonball_maximum_height :
  ∃ t₀ : ℝ, ∀ t : ℝ, height_function t ≤ height_function t₀ ∧ height_function t₀ = 161 :=
by
  sorry

end cannonball_maximum_height_l2146_214640


namespace middle_part_of_sum_is_120_l2146_214690

theorem middle_part_of_sum_is_120 (x : ℚ) (h : 2 * x + x + (1 / 2) * x = 120) : 
  x = 240 / 7 := sorry

end middle_part_of_sum_is_120_l2146_214690


namespace sector_area_l2146_214676

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) :
  (1 / 2) * α * r ^ 2 = Real.pi := by
  sorry

end sector_area_l2146_214676


namespace opposite_of_83_is_84_l2146_214616

noncomputable def opposite_number (k : ℕ) (n : ℕ) : ℕ :=
if k < n / 2 then k + n / 2 else k - n / 2

theorem opposite_of_83_is_84 : opposite_number 83 100 = 84 := 
by
  -- Assume the conditions of the problem here for the proof.
  sorry

end opposite_of_83_is_84_l2146_214616


namespace prime_divisor_congruent_one_mod_p_l2146_214685

theorem prime_divisor_congruent_one_mod_p (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ q ∣ p^p - 1 ∧ q % p = 1 :=
sorry

end prime_divisor_congruent_one_mod_p_l2146_214685
