import Mathlib

namespace inequality_proof_l1297_129700

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (b + c)) + (1 / (a + c)) + (1 / (a + b)) ≥ 9 / (2 * (a + b + c)) :=
by
  sorry

end inequality_proof_l1297_129700


namespace number_of_combinations_l1297_129736

noncomputable def countOddNumbers (n : ℕ) : ℕ := (n + 1) / 2

noncomputable def countPrimesLessThan30 : ℕ := 9 -- {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def countMultiplesOfFour (n : ℕ) : ℕ := n / 4

theorem number_of_combinations : countOddNumbers 40 * countPrimesLessThan30 * countMultiplesOfFour 40 = 1800 := by
  sorry

end number_of_combinations_l1297_129736


namespace number_of_students_in_Diligence_before_transfer_l1297_129715

-- Define the total number of students and the transfer information
def total_students : ℕ := 50
def transferred_students : ℕ := 2

-- Define the number of students in Diligence before the transfer
def students_in_Diligence_before : ℕ := 23

-- Let's prove that the number of students in Diligence before the transfer is 23
theorem number_of_students_in_Diligence_before_transfer :
  (total_students / 2) - transferred_students = students_in_Diligence_before :=
by {
  -- The proof is omitted as instructed
  sorry
}

end number_of_students_in_Diligence_before_transfer_l1297_129715


namespace parametric_to_line_segment_l1297_129779

theorem parametric_to_line_segment :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 5 →
  ∃ x y : ℝ, x = 3 * t^2 + 2 ∧ y = t^2 - 1 ∧ (x - 3 * y = 5) ∧ (-1 ≤ y ∧ y ≤ 24) :=
by
  sorry

end parametric_to_line_segment_l1297_129779


namespace ones_digit_of_power_l1297_129776

theorem ones_digit_of_power (n : ℕ) : 
  (13 ^ (13 * (12 ^ 12)) % 10) = 9 :=
by
  sorry

end ones_digit_of_power_l1297_129776


namespace choir_robe_costs_l1297_129799

theorem choir_robe_costs:
  ∀ (total_robes needed_robes total_cost robe_cost : ℕ),
  total_robes = 30 →
  needed_robes = 30 - 12 →
  total_cost = 36 →
  total_cost = needed_robes * robe_cost →
  robe_cost = 2 :=
by
  intros total_robes needed_robes total_cost robe_cost
  intro h_total_robes h_needed_robes h_total_cost h_cost_eq
  sorry

end choir_robe_costs_l1297_129799


namespace average_price_over_3_months_l1297_129790

theorem average_price_over_3_months (dMay : ℕ) 
  (pApril pMay pJune : ℝ) 
  (h1 : pApril = 1.20) 
  (h2 : pMay = 1.20) 
  (h3 : pJune = 3.00) 
  (h4 : dApril = 2 / 3 * dMay) 
  (h5 : dJune = 2 * dApril) :
  ((dApril * pApril + dMay * pMay + dJune * pJune) / (dApril + dMay + dJune) = 2) := 
by sorry

end average_price_over_3_months_l1297_129790


namespace find_q_l1297_129711

-- Given conditions
noncomputable def digits_non_zero (p q r : Nat) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0

noncomputable def three_digit_number (p q r : Nat) : Nat :=
  100 * p + 10 * q + r

noncomputable def two_digit_number (q r : Nat) : Nat :=
  10 * q + r

noncomputable def one_digit_number (r : Nat) : Nat := r

noncomputable def numbers_sum_to (p q r sum : Nat) : Prop :=
  three_digit_number p q r + two_digit_number q r + one_digit_number r = sum

-- The theorem to prove
theorem find_q (p q r : Nat) (hpq : digits_non_zero p q r)
  (hsum : numbers_sum_to p q r 912) : q = 5 := sorry

end find_q_l1297_129711


namespace range_of_a_l1297_129730

noncomputable def parabola_locus (x : ℝ) : ℝ := x^2 / 4

def angle_sum_property (a k : ℝ) : Prop :=
  2 * a * k^2 + 2 * k + a = 0

def discriminant_nonnegative (a : ℝ) : Prop :=
  4 - 8 * a^2 ≥ 0

theorem range_of_a (a : ℝ) :
  (- (Real.sqrt 2) / 2) ≤ a ∧ a ≤ (Real.sqrt 2) / 2 :=
  sorry

end range_of_a_l1297_129730


namespace bacteria_after_time_l1297_129739

def initial_bacteria : ℕ := 1
def division_time : ℕ := 20  -- time in minutes for one division
def total_time : ℕ := 180  -- total time in minutes

def divisions := total_time / division_time

theorem bacteria_after_time : (initial_bacteria * 2 ^ divisions) = 512 := by
  exact sorry

end bacteria_after_time_l1297_129739


namespace geometric_sequence_a5_l1297_129766

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 * a 5 = 16) (h2 : a 4 = 8) (h3 : ∀ n, a n > 0) : a 5 = 16 := 
by
  sorry

end geometric_sequence_a5_l1297_129766


namespace find_number_l1297_129731

theorem find_number (x : ℝ) : (8^3 * x^3) / 679 = 549.7025036818851 -> x = 9 :=
by
  sorry

end find_number_l1297_129731


namespace lemonade_cups_count_l1297_129724

theorem lemonade_cups_count :
  ∃ x y : ℕ, x + y = 400 ∧ x + 2 * y = 546 ∧ x = 254 :=
by
  sorry

end lemonade_cups_count_l1297_129724


namespace find_sum_of_min_area_ks_l1297_129769

def point := ℝ × ℝ

def A : point := (2, 9)
def B : point := (14, 18)

def is_int (k : ℝ) : Prop := ∃ (n : ℤ), k = n

def min_triangle_area (P Q R : point) : ℝ := sorry
-- Placeholder for the area formula of a triangle given three points

def valid_ks (k : ℝ) : Prop :=
  is_int k ∧ min_triangle_area A B (6, k) ≠ 0

theorem find_sum_of_min_area_ks :
  (∃ k1 k2 : ℤ, valid_ks k1 ∧ valid_ks k2 ∧ (k1 + k2) = 31) :=
sorry

end find_sum_of_min_area_ks_l1297_129769


namespace problem_I_problem_II_l1297_129756

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}
def B : Set ℝ := {x : ℝ | x ≥ 1}

-- Define the complement of A in the universal set U which is ℝ
def complement_U_A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

-- Define the union of complement_U_A and B
def union_complement_U_A_B : Set ℝ := complement_U_A ∪ B

-- Proof Problem I: Prove that the set A is as specified
theorem problem_I : A = {x : ℝ | x ≤ -1 ∨ x ≥ 3} := sorry

-- Proof Problem II: Prove that the union of the complement of A and B is as specified
theorem problem_II : union_complement_U_A_B = {x : ℝ | x > -1} := sorry

end problem_I_problem_II_l1297_129756


namespace problem_I_problem_II_problem_III_l1297_129793

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 3

theorem problem_I (a b : ℝ) (h_a : a = 0) :
  (b ≥ 0 → ∀ x : ℝ, 3 * x^2 + b ≥ 0) ∧
  (b < 0 → 
    ∀ x : ℝ, (x < -Real.sqrt (-b / 3) ∨ x > Real.sqrt (-b / 3)) → 
      3 * x^2 + b > 0) := sorry

theorem problem_II (b : ℝ) :
  ∃ x0 : ℝ, f x0 0 b = x0 ∧ (3 * x0^2 + b = 0) ↔ b = -3 := sorry

theorem problem_III :
  ∀ a b : ℝ, ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧
    (3 * x1^2 + 2 * a * x1 + b = 0) ∧
    (3 * x2^2 + 2 * a * x2 + b = 0) ∧
    (f x1 a b = x1) ∧
    (f x2 a b = x2)) := sorry

end problem_I_problem_II_problem_III_l1297_129793


namespace line_through_origin_and_intersection_of_lines_l1297_129772

theorem line_through_origin_and_intersection_of_lines 
  (x y : ℝ)
  (h1 : x - 3 * y + 4 = 0)
  (h2 : 2 * x + y + 5 = 0) :
  3 * x + 19 * y = 0 :=
sorry

end line_through_origin_and_intersection_of_lines_l1297_129772


namespace find_u_value_l1297_129796

theorem find_u_value (h : ∃ n : ℕ, n = 2012) : ∃ u : ℕ, u = 2015 := 
by
  sorry

end find_u_value_l1297_129796


namespace find_second_number_l1297_129704

theorem find_second_number 
  (x : ℕ)
  (h1 : (55 + x + 507 + 2 + 684 + 42) / 6 = 223)
  : x = 48 := 
by 
  sorry

end find_second_number_l1297_129704


namespace complex_number_identity_l1297_129706

theorem complex_number_identity (i : ℂ) (hi : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end complex_number_identity_l1297_129706


namespace total_value_is_76_percent_of_dollar_l1297_129732

def coin_values : List Nat := [1, 5, 20, 50]

def total_value (coins : List Nat) : Nat :=
  List.sum coins

def percentage_of_dollar (value : Nat) : Nat :=
  value * 100 / 100

theorem total_value_is_76_percent_of_dollar :
  percentage_of_dollar (total_value coin_values) = 76 := by
  sorry

end total_value_is_76_percent_of_dollar_l1297_129732


namespace best_marksman_score_l1297_129746

def team_size : ℕ := 6
def total_points : ℕ := 497
def hypothetical_best_score : ℕ := 92
def hypothetical_average : ℕ := 84

theorem best_marksman_score :
  let total_with_hypothetical_best := team_size * hypothetical_average
  let difference := total_with_hypothetical_best - total_points
  let actual_best_score := hypothetical_best_score - difference
  actual_best_score = 85 := 
by
  -- Definitions in Lean are correctly set up
  intro total_with_hypothetical_best difference actual_best_score
  sorry

end best_marksman_score_l1297_129746


namespace time_to_cross_tree_l1297_129727

theorem time_to_cross_tree (length_train : ℝ) (length_platform : ℝ) (time_to_pass_platform : ℝ) (h1 : length_train = 1200) (h2 : length_platform = 1200) (h3 : time_to_pass_platform = 240) : 
  (length_train / ((length_train + length_platform) / time_to_pass_platform)) = 120 := 
by
    sorry

end time_to_cross_tree_l1297_129727


namespace combined_parent_age_difference_l1297_129721

def father_age_at_sobha_birth : ℕ := 38
def mother_age_at_brother_birth : ℕ := 36
def brother_younger_than_sobha : ℕ := 4
def sister_younger_than_brother : ℕ := 3
def father_age_at_sister_birth : ℕ := 45
def mother_age_at_youngest_birth : ℕ := 34
def youngest_younger_than_sister : ℕ := 6

def mother_age_at_sobha_birth := mother_age_at_brother_birth - brother_younger_than_sobha
def father_age_at_youngest_birth := father_age_at_sister_birth + youngest_younger_than_sister

def combined_age_difference_at_sobha_birth := father_age_at_sobha_birth - mother_age_at_sobha_birth
def compounded_difference_at_sobha_brother_birth := 
  (father_age_at_sobha_birth + brother_younger_than_sobha) - mother_age_at_brother_birth
def mother_age_at_sister_birth := mother_age_at_brother_birth + sister_younger_than_brother
def compounded_difference_at_sobha_sister_birth := father_age_at_sister_birth - mother_age_at_sister_birth
def compounded_difference_at_youngest_birth := father_age_at_youngest_birth - mother_age_at_youngest_birth

def combined_age_difference := 
  combined_age_difference_at_sobha_birth + 
  compounded_difference_at_sobha_brother_birth + 
  compounded_difference_at_sobha_sister_birth + 
  compounded_difference_at_youngest_birth 

theorem combined_parent_age_difference : combined_age_difference = 35 := by
  sorry

end combined_parent_age_difference_l1297_129721


namespace value_of_x_l1297_129792

theorem value_of_x (u w z y x : ℤ) (h1 : u = 95) (h2 : w = u + 10) (h3 : z = w + 25) (h4 : y = z + 15) (h5 : x = y + 12) : x = 157 := by
  sorry

end value_of_x_l1297_129792


namespace sum_of_digits_5_pow_eq_2_pow_l1297_129733

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_5_pow_eq_2_pow (n : ℕ) (h : sum_of_digits (5^n) = 2^n) : n = 3 :=
by
  sorry

end sum_of_digits_5_pow_eq_2_pow_l1297_129733


namespace factor_tree_value_l1297_129725

-- Define the values and their relationships
def A := 900
def B := 3 * (3 * 2)
def D := 3 * 2
def C := 5 * (5 * 2)
def E := 5 * 2

-- Define the theorem and provide the conditions
theorem factor_tree_value :
  (B = 3 * D) →
  (D = 3 * 2) →
  (C = 5 * E) →
  (E = 5 * 2) →
  (A = B * C) →
  A = 900 := by
  intros hB hD hC hE hA
  sorry

end factor_tree_value_l1297_129725


namespace find_y_l1297_129754

theorem find_y (x y : ℕ) (h1 : x = 2407) (h2 : x^y + y^x = 2408) : y = 1 :=
sorry

end find_y_l1297_129754


namespace value_of_b_minus_d_squared_l1297_129777

theorem value_of_b_minus_d_squared (a b c d : ℤ) 
  (h1 : a - b - c + d = 18) 
  (h2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := 
by 
  sorry

end value_of_b_minus_d_squared_l1297_129777


namespace apples_in_blue_basket_l1297_129787

-- Define the number of bananas in the blue basket
def bananas := 12

-- Define the total number of fruits in the blue basket
def totalFruits := 20

-- Define the number of apples as total fruits minus bananas
def apples := totalFruits - bananas

-- Prove that the number of apples in the blue basket is 8
theorem apples_in_blue_basket : apples = 8 := by
  sorry

end apples_in_blue_basket_l1297_129787


namespace initial_red_martians_l1297_129794

/-- Red Martians always tell the truth, while Blue Martians lie and then turn red.
    In a group of 2018 Martians, they answered in the sequence 1, 2, 3, ..., 2018 to the question
    of how many of them were red at that moment. Prove that the initial number of red Martians was 0 or 1. -/
theorem initial_red_martians (N : ℕ) (answers : Fin (N+1) → ℕ) :
  (∀ i : Fin (N+1), answers i = i.succ) → N = 2018 → (initial_red_martians_count = 0 ∨ initial_red_martians_count = 1)
:= sorry

end initial_red_martians_l1297_129794


namespace skylar_starting_age_l1297_129710

-- Conditions of the problem
def annual_donation : ℕ := 8000
def current_age : ℕ := 71
def total_amount_donated : ℕ := 440000

-- Question and proof statement
theorem skylar_starting_age :
  (current_age - total_amount_donated / annual_donation) = 16 := 
by
  sorry

end skylar_starting_age_l1297_129710


namespace sachin_is_younger_by_8_years_l1297_129749

variable (S R : ℕ)

-- Conditions
axiom age_of_sachin : S = 28
axiom ratio_of_ages : S * 9 = R * 7

-- Goal
theorem sachin_is_younger_by_8_years (S R : ℕ) (h1 : S = 28) (h2 : S * 9 = R * 7) : R - S = 8 :=
by
  sorry

end sachin_is_younger_by_8_years_l1297_129749


namespace number_above_210_is_165_l1297_129708

def triangular_number (k : ℕ) : ℕ := k * (k + 1) / 2
def tetrahedral_number (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6
def row_start (k : ℕ) : ℕ := tetrahedral_number (k - 1) + 1

theorem number_above_210_is_165 :
  ∀ k, triangular_number k = 210 →
  ∃ n, n = 165 → 
  ∀ m, row_start (k - 1) ≤ m ∧ m < row_start k →
  m = 210 →
  n = m - triangular_number (k - 1) :=
  sorry

end number_above_210_is_165_l1297_129708


namespace ratio_of_areas_l1297_129762

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l1297_129762


namespace obtuse_angles_at_intersection_l1297_129781

theorem obtuse_angles_at_intersection (lines_intersect_x_at_diff_points : Prop) (lines_not_perpendicular : Prop) 
(lines_form_obtuse_angle_at_intersection : Prop) : 
(lines_intersect_x_at_diff_points ∧ lines_not_perpendicular ∧ lines_form_obtuse_angle_at_intersection) → 
  ∃ (n : ℕ), n = 2 :=
by 
  sorry

end obtuse_angles_at_intersection_l1297_129781


namespace one_fourth_one_third_two_fifths_l1297_129740

theorem one_fourth_one_third_two_fifths (N : ℝ)
  (h₁ : 0.40 * N = 300) :
  (1/4) * (1/3) * (2/5) * N = 25 := 
sorry

end one_fourth_one_third_two_fifths_l1297_129740


namespace range_of_m_l1297_129726

theorem range_of_m : 
  ∀ m : ℝ, m = 3 * Real.sqrt 2 - 1 → 3 < m ∧ m < 4 := 
by
  -- the proof will go here
  sorry

end range_of_m_l1297_129726


namespace part_a_part_b_l1297_129713

-- This definition states that a number p^m is a divisor of a-1
def divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  (p ^ m) ∣ (a - 1)

-- This definition states that (p^(m+1)) is not a divisor of a-1
def not_divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  ¬ (p ^ (m + 1) ∣ (a - 1))

-- Part (a): Prove divisibility
theorem part_a (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  p ^ (m + n) ∣ a ^ (p ^ n) - 1 := 
sorry

-- Part (b): Prove non-divisibility
theorem part_b (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  ¬ p ^ (m + n + 1) ∣ a ^ (p ^ n) - 1 := 
sorry

end part_a_part_b_l1297_129713


namespace replace_asterisk_l1297_129722

theorem replace_asterisk (star : ℝ) : ((36 / 18) * (star / 72) = 1) → star = 36 :=
by
  intro h
  sorry

end replace_asterisk_l1297_129722


namespace tommy_needs_to_save_l1297_129783

theorem tommy_needs_to_save (books : ℕ) (cost_per_book : ℕ) (money_he_has : ℕ) 
  (total_cost : ℕ) (money_needed : ℕ) 
  (h1 : books = 8)
  (h2 : cost_per_book = 5)
  (h3 : money_he_has = 13)
  (h4 : total_cost = books * cost_per_book) :
  money_needed = total_cost - money_he_has ∧ money_needed = 27 :=
by 
  sorry

end tommy_needs_to_save_l1297_129783


namespace problem_I_problem_II_l1297_129709

open Set Real

-- Problem (I)
theorem problem_I (x : ℝ) :
  (|x - 2| ≥ 4 - |x - 1|) ↔ x ∈ Iic (-1/2) ∪ Ici (7/2) :=
by
  sorry

-- Problem (II)
theorem problem_II (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 1/2/n = 1) :
  m + 2 * n ≥ 4 :=
by
  sorry

end problem_I_problem_II_l1297_129709


namespace original_number_is_3199_l1297_129788

theorem original_number_is_3199 (n : ℕ) (k : ℕ) (h1 : k = 3200) (h2 : (n + k) % 8 = 0) : n = 3199 :=
sorry

end original_number_is_3199_l1297_129788


namespace area_of_quadrilateral_centroids_l1297_129797

noncomputable def square_side_length : ℝ := 40
noncomputable def point_Q_XQ : ℝ := 15
noncomputable def point_Q_YQ : ℝ := 35

theorem area_of_quadrilateral_centroids (h1 : square_side_length = 40)
    (h2 : point_Q_XQ = 15)
    (h3 : point_Q_YQ = 35) :
    ∃ (area : ℝ), area = 800 / 9 :=
by
  sorry

end area_of_quadrilateral_centroids_l1297_129797


namespace real_solutions_eq_pos_neg_2_l1297_129714

theorem real_solutions_eq_pos_neg_2 (x : ℝ) :
  ( (x - 1) ^ 2 * (x - 5) * (x - 5) / (x - 5) = 4) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

end real_solutions_eq_pos_neg_2_l1297_129714


namespace john_has_388_pennies_l1297_129753

theorem john_has_388_pennies (k : ℕ) (j : ℕ) (hk : k = 223) (hj : j = k + 165) : j = 388 := by
  sorry

end john_has_388_pennies_l1297_129753


namespace root_product_is_27_l1297_129735

open Real

noncomputable def cube_root (x : ℝ) := x ^ (1 / 3 : ℝ)
noncomputable def fourth_root (x : ℝ) := x ^ (1 / 4 : ℝ)
noncomputable def square_root (x : ℝ) := x ^ (1 / 2 : ℝ)

theorem root_product_is_27 : 
  (cube_root 27) * (fourth_root 81) * (square_root 9) = 27 := 
by
  sorry

end root_product_is_27_l1297_129735


namespace minimum_value_of_sum_l1297_129728

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1/a + 2/b + 3/c = 2) : a + 2*b + 3*c = 18 ↔ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end minimum_value_of_sum_l1297_129728


namespace solve_for_pure_imaginary_l1297_129789

theorem solve_for_pure_imaginary (x : ℝ) 
  (h1 : x^2 - 1 = 0) 
  (h2 : x - 1 ≠ 0) 
  : x = -1 :=
sorry

end solve_for_pure_imaginary_l1297_129789


namespace circumcircle_radius_l1297_129798

open Real

theorem circumcircle_radius (a b c A B C S R : ℝ) 
  (h1 : S = (1/2) * sin A * sin B * sin C)
  (h2 : S = (1/2) * a * b * sin C)
  (h3 : ∀ x y, x = y → x * cos 0 = y * cos 0):
  R = (1/2) :=
by
  sorry

end circumcircle_radius_l1297_129798


namespace car_overtakes_buses_l1297_129771

/-- 
  Buses leave the airport every 3 minutes. 
  A bus takes 60 minutes to travel from the airport to the city center. 
  A car takes 35 minutes to travel from the airport to the city center. 
  Prove that the car overtakes 8 buses on its way to the city center excluding the bus it left with.
--/
theorem car_overtakes_buses (arr_bus : ℕ) (arr_car : ℕ) (interval : ℕ) (diff : ℕ) : 
  interval = 3 → arr_bus = 60 → arr_car = 35 → diff = arr_bus - arr_car →
  ∃ n : ℕ, n = diff / interval ∧ n = 8 := by
  sorry

end car_overtakes_buses_l1297_129771


namespace grandma_red_bacon_bits_l1297_129795

theorem grandma_red_bacon_bits:
  ∀ (mushrooms cherryTomatoes pickles baconBits redBaconBits : ℕ),
    mushrooms = 3 →
    cherryTomatoes = 2 * mushrooms →
    pickles = 4 * cherryTomatoes →
    baconBits = 4 * pickles →
    redBaconBits = 1 / 3 * baconBits →
    redBaconBits = 32 := 
by
  intros mushrooms cherryTomatoes pickles baconBits redBaconBits
  intros h1 h2 h3 h4 h5
  sorry

end grandma_red_bacon_bits_l1297_129795


namespace find_factors_of_224_l1297_129765

theorem find_factors_of_224 : ∃ (a b c : ℕ), a * b * c = 224 ∧ c = 2 * a ∧ a ≠ b ∧ b ≠ c :=
by
  -- Prove that the factors meeting the criteria exist
  sorry

end find_factors_of_224_l1297_129765


namespace total_income_per_minute_l1297_129744

theorem total_income_per_minute :
  let black_shirt_price := 30
  let black_shirt_quantity := 250
  let white_shirt_price := 25
  let white_shirt_quantity := 200
  let red_shirt_price := 28
  let red_shirt_quantity := 100
  let blue_shirt_price := 25
  let blue_shirt_quantity := 50

  let black_discount := 0.05
  let white_discount := 0.08
  let red_discount := 0.10

  let total_black_income_before_discount := black_shirt_quantity * black_shirt_price
  let total_white_income_before_discount := white_shirt_quantity * white_shirt_price
  let total_red_income_before_discount := red_shirt_quantity * red_shirt_price
  let total_blue_income_before_discount := blue_shirt_quantity * blue_shirt_price

  let total_income_before_discount :=
    total_black_income_before_discount + total_white_income_before_discount + total_red_income_before_discount + total_blue_income_before_discount

  let total_black_discount := black_discount * total_black_income_before_discount
  let total_white_discount := white_discount * total_white_income_before_discount
  let total_red_discount := red_discount * total_red_income_before_discount

  let total_discount :=
    total_black_discount + total_white_discount + total_red_discount

  let total_income_after_discount :=
    total_income_before_discount - total_discount

  let total_minutes := 40
  let total_income_per_minute := total_income_after_discount / total_minutes

  total_income_per_minute = 387.38 := by
  sorry

end total_income_per_minute_l1297_129744


namespace determine_k_l1297_129780

variable (x y z k : ℝ)

theorem determine_k
  (h1 : 9 / (x - y) = 16 / (z + y))
  (h2 : k / (x + z) = 16 / (z + y)) :
  k = 25 := by
  sorry

end determine_k_l1297_129780


namespace solution_set_of_inequality_l1297_129707

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 + 4 * x - 5 > 0 ↔ (x < -5 ∨ x > 1) :=
sorry

end solution_set_of_inequality_l1297_129707


namespace number_of_customers_l1297_129742

theorem number_of_customers (offices_sandwiches : Nat)
                            (group_per_person_sandwiches : Nat)
                            (total_sandwiches : Nat)
                            (half_group : Nat) :
  (offices_sandwiches = 3 * 10) →
  (total_sandwiches = 54) →
  (half_group * group_per_person_sandwiches = total_sandwiches - offices_sandwiches) →
  (2 * half_group = 12) := 
by
  sorry

end number_of_customers_l1297_129742


namespace sequence_general_term_l1297_129775

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 4 else 4 * (-1 / 3)^(n - 1) 

theorem sequence_general_term (n : ℕ) (hn : n ≥ 1) 
  (hrec : ∀ n, 3 * a_n (n + 1) + a_n n = 0)
  (hinit : a_n 2 = -4 / 3) :
  a_n n = 4 * (-1 / 3)^(n - 1) := by
  sorry

end sequence_general_term_l1297_129775


namespace equation_negative_roots_iff_l1297_129737

theorem equation_negative_roots_iff (a : ℝ) :
  (∃ x < 0, 4^x - 2^(x-1) + a = 0) ↔ (-1/2 < a ∧ a ≤ 1/16) := 
sorry

end equation_negative_roots_iff_l1297_129737


namespace minimize_S_l1297_129741

theorem minimize_S (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 23) : n = 7 ↔ ∃ (m : ℕ), (∀ k ≤ m, a k <= 0) ∧ m = 7 :=
by
  sorry

end minimize_S_l1297_129741


namespace seven_digit_divisible_by_eleven_l1297_129716

theorem seven_digit_divisible_by_eleven (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 9) 
  (h3 : 10 - n ≡ 0 [MOD 11]) : n = 10 :=
by
  sorry

end seven_digit_divisible_by_eleven_l1297_129716


namespace trip_is_400_miles_l1297_129761

def fuel_per_mile_empty_plane := 20
def fuel_increase_per_person := 3
def fuel_increase_per_bag := 2
def number_of_passengers := 30
def number_of_crew := 5
def bags_per_person := 2
def total_fuel_needed := 106000

def fuel_consumption_per_mile :=
  fuel_per_mile_empty_plane +
  (number_of_passengers + number_of_crew) * fuel_increase_per_person +
  (number_of_passengers + number_of_crew) * bags_per_person * fuel_increase_per_bag

def trip_length := total_fuel_needed / fuel_consumption_per_mile

theorem trip_is_400_miles : trip_length = 400 := 
by sorry

end trip_is_400_miles_l1297_129761


namespace min_value_is_3_plus_2_sqrt_2_l1297_129778

noncomputable def minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = a*b) : ℝ :=
a + b

theorem min_value_is_3_plus_2_sqrt_2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = a*b) :
  minimum_value a b h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_is_3_plus_2_sqrt_2_l1297_129778


namespace min_value_of_a_l1297_129785

theorem min_value_of_a (a : ℝ) (h : a > 0) (h₁ : ∀ x : ℝ, |x - a| + |1 - x| ≥ 1) : a ≥ 2 := 
sorry

end min_value_of_a_l1297_129785


namespace find_b_l1297_129791

theorem find_b (a b c : ℕ) (h1 : a * b + b * c - c * a = 0) (h2 : a - c = 101) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) : b = 2550 :=
sorry

end find_b_l1297_129791


namespace minimum_area_sum_l1297_129755

-- Define the coordinates and the conditions
variable {x1 y1 x2 y2 : ℝ}
variable (on_parabola_A : y1^2 = x1)
variable (on_parabola_B : y2^2 = x2)
variable (y1_pos : y1 > 0)
variable (y2_neg : y2 < 0)
variable (dot_product : x1 * x2 + y1 * y2 = 2)

-- Define the function to calculate areas
noncomputable def area_sum (y1 y2 x1 x2 : ℝ) : ℝ :=
  1/2 * 2 * (y1 - y2) + 1/2 * 1/4 * y1

theorem minimum_area_sum :
  ∃ y1 y2 x1 x2, y1^2 = x1 ∧ y2^2 = x2 ∧ y1 > 0 ∧ y2 < 0 ∧ x1 * x2 + y1 * y2 = 2 ∧
  (area_sum y1 y2 x1 x2 = 3) := sorry

end minimum_area_sum_l1297_129755


namespace find_base_l1297_129719

theorem find_base (a : ℕ) (ha : a > 11) (hB : 11 = 11) :
  (3 * a^2 + 9 * a + 6) + (5 * a^2 + 7 * a + 5) = (9 * a^2 + 7 * a + 11) → 
  a = 12 :=
sorry

end find_base_l1297_129719


namespace smaller_of_two_integers_l1297_129734

theorem smaller_of_two_integers (m n : ℕ) (h1 : 100 ≤ m ∧ m < 1000) (h2 : 100 ≤ n ∧ n < 1000)
  (h3 : (m + n) / 2 = m + n / 1000) : min m n = 999 :=
by {
  sorry
}

end smaller_of_two_integers_l1297_129734


namespace trigonometric_identity_l1297_129786

open Real

theorem trigonometric_identity (α φ : ℝ) :
  cos α ^ 2 + cos φ ^ 2 + cos (α + φ) ^ 2 - 2 * cos α * cos φ * cos (α + φ) = 1 :=
sorry

end trigonometric_identity_l1297_129786


namespace valid_triangle_side_l1297_129757

theorem valid_triangle_side (x : ℝ) (h1 : 2 + x > 6) (h2 : 2 + 6 > x) (h3 : x + 6 > 2) : x = 6 :=
by
  sorry

end valid_triangle_side_l1297_129757


namespace find_two_digits_l1297_129729

theorem find_two_digits (a b : ℕ) (h₁: a ≤ 9) (h₂: b ≤ 9)
  (h₃: (4 + a + b) % 9 = 0) (h₄: (10 * a + b) % 4 = 0) :
  (a = 3 ∧ b = 2) ∨ (a = 6 ∧ b = 8) :=
by {
  sorry
}

end find_two_digits_l1297_129729


namespace fraction_simplification_l1297_129717

open Real -- Open the Real namespace for real number operations

theorem fraction_simplification (a x : ℝ) : 
  (sqrt (a^2 + x^2) - (x^2 + a^2) / sqrt (a^2 + x^2)) / (a^2 + x^2) = 0 := 
sorry

end fraction_simplification_l1297_129717


namespace mean_equality_l1297_129743

theorem mean_equality (x : ℚ) : 
  (3 + 7 + 15) / 3 = (x + 10) / 2 → x = 20 / 3 := 
by 
  sorry

end mean_equality_l1297_129743


namespace max_value_inequality_l1297_129703

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c) / ((a + b)^2 * (b + c)^2) ≤ 1 / 4) :=
sorry

end max_value_inequality_l1297_129703


namespace polynomial_roots_l1297_129751

theorem polynomial_roots :
  (∃ x : ℝ, x^4 - 16*x^3 + 91*x^2 - 216*x + 180 = 0) ↔ (x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 6) := 
sorry

end polynomial_roots_l1297_129751


namespace RiverJoe_popcorn_shrimp_price_l1297_129760

theorem RiverJoe_popcorn_shrimp_price
  (price_catfish : ℝ)
  (total_orders : ℕ)
  (total_revenue : ℝ)
  (orders_popcorn_shrimp : ℕ)
  (catfish_revenue : ℝ)
  (popcorn_shrimp_price : ℝ) :
  price_catfish = 6.00 →
  total_orders = 26 →
  total_revenue = 133.50 →
  orders_popcorn_shrimp = 9 →
  catfish_revenue = (total_orders - orders_popcorn_shrimp) * price_catfish →
  catfish_revenue + orders_popcorn_shrimp * popcorn_shrimp_price = total_revenue →
  popcorn_shrimp_price = 3.50 :=
by
  intros price_catfish_eq total_orders_eq total_revenue_eq orders_popcorn_shrimp_eq catfish_revenue_eq revenue_eq
  sorry

end RiverJoe_popcorn_shrimp_price_l1297_129760


namespace sum_of_x_coords_Q3_l1297_129763

-- Definitions
def Q1_vertices_sum_x (S : ℝ) := S = 1050

def Q2_vertices_sum_x (S' : ℝ) (S : ℝ) := S' = S

def Q3_vertices_sum_x (S'' : ℝ) (S' : ℝ) := S'' = S'

-- Lean 4 statement
theorem sum_of_x_coords_Q3 (S : ℝ) (S' : ℝ) (S'' : ℝ) :
  Q1_vertices_sum_x S →
  Q2_vertices_sum_x S' S →
  Q3_vertices_sum_x S'' S' →
  S'' = 1050 :=
by
  sorry

end sum_of_x_coords_Q3_l1297_129763


namespace distance_between_foci_l1297_129767

-- Defining the given ellipse equation 
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 150 * x + 4 * y^2 + 8 * y + 9 = 0

-- Statement to prove the distance between the foci
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 46.2 := 
sorry

end distance_between_foci_l1297_129767


namespace greatest_multiple_of_four_l1297_129774

theorem greatest_multiple_of_four (x : ℕ) (hx : x > 0) (h4 : x % 4 = 0) (hcube : x^3 < 800) : x ≤ 8 :=
by {
  sorry
}

end greatest_multiple_of_four_l1297_129774


namespace ticket_cost_calculation_l1297_129770

theorem ticket_cost_calculation :
  let adult_price := 12
  let child_price := 10
  let num_adults := 3
  let num_children := 3
  let total_cost := (num_adults * adult_price) + (num_children * child_price)
  total_cost = 66 := 
by
  rfl -- or add sorry to skip proof

end ticket_cost_calculation_l1297_129770


namespace solve_quadratic_l1297_129723

theorem solve_quadratic :
  ∀ x, (x^2 - x - 12 = 0) → (x = -3 ∨ x = 4) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_l1297_129723


namespace product_of_translated_roots_l1297_129748

noncomputable def roots (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_translated_roots
  {d e : ℝ}
  (h_d : roots 3 4 (-7) d)
  (h_e : roots 3 4 (-7) e)
  (sum_roots : d + e = -4 / 3)
  (product_roots : d * e = -7 / 3) :
  (d - 1) * (e - 1) = 1 :=
by
  sorry

end product_of_translated_roots_l1297_129748


namespace exponentiation_problem_l1297_129750

theorem exponentiation_problem 
(a b : ℝ) 
(h : a ^ b = 1 / 8) : a ^ (-3 * b) = 512 := 
sorry

end exponentiation_problem_l1297_129750


namespace circle_range_of_a_l1297_129784

theorem circle_range_of_a (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + a = 0) → a < 5 := by
  sorry

end circle_range_of_a_l1297_129784


namespace divide_fractions_l1297_129759

theorem divide_fractions : (3 / 8) / (1 / 4) = 3 / 2 :=
by sorry

end divide_fractions_l1297_129759


namespace calculation_expression_solve_system_of_equations_l1297_129752

-- Part 1: Prove the calculation
theorem calculation_expression :
  (6 - 2 * Real.sqrt 3) * Real.sqrt 3 - Real.sqrt ((2 - Real.sqrt 2) ^ 2) + 1 / Real.sqrt 2 = 
  6 * Real.sqrt 3 - 8 + 3 * Real.sqrt 2 / 2 :=
by
  -- proof will be here
  sorry

-- Part 2: Prove the solution of the system of equations
theorem solve_system_of_equations (x y : ℝ) :
  (5 * x - y = -9) ∧ (3 * x + y = 1) → (x = -1 ∧ y = 4) :=
by
  -- proof will be here
  sorry

end calculation_expression_solve_system_of_equations_l1297_129752


namespace remainder_of_12_pow_2012_mod_5_l1297_129718

theorem remainder_of_12_pow_2012_mod_5 : (12 ^ 2012) % 5 = 1 :=
by
  sorry

end remainder_of_12_pow_2012_mod_5_l1297_129718


namespace second_valve_emits_more_l1297_129745

noncomputable def V1 : ℝ := 12000 / 120 -- Rate of first valve (100 cubic meters/minute)
noncomputable def V2 : ℝ := 12000 / 48 - V1 -- Rate of second valve

theorem second_valve_emits_more : V2 - V1 = 50 :=
by
  sorry

end second_valve_emits_more_l1297_129745


namespace find_minimal_x_l1297_129702

-- Conditions
variables (x y : ℕ)
variable (pos_x : x > 0)
variable (pos_y : y > 0)
variable (h : 3 * x^7 = 17 * y^11)

-- Proof Goal
theorem find_minimal_x : ∃ a b c d : ℕ, x = a^c * b^d ∧ a + b + c + d = 30 :=
by {
  sorry
}

end find_minimal_x_l1297_129702


namespace area_of_rectangle_l1297_129747

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l1297_129747


namespace distance_between_foci_of_ellipse_l1297_129720

theorem distance_between_foci_of_ellipse :
  ∀ x y : ℝ,
  9 * x^2 - 36 * x + 4 * y^2 + 16 * y + 16 = 0 →
  2 * Real.sqrt (9 - 4) = 2 * Real.sqrt 5 :=
by 
  sorry

end distance_between_foci_of_ellipse_l1297_129720


namespace solution_fractional_equation_l1297_129782

noncomputable def solve_fractional_equation : Prop :=
  ∀ x : ℝ, (4/(x-2) = 2/x) ↔ x = -2

theorem solution_fractional_equation :
  solve_fractional_equation :=
by
  sorry

end solution_fractional_equation_l1297_129782


namespace Mr_Pendearly_optimal_speed_l1297_129764

noncomputable def optimal_speed (d t : ℝ) : ℝ := d / t

theorem Mr_Pendearly_optimal_speed :
  ∀ (d t : ℝ),
  (d = 45 * (t + 1/15)) →
  (d = 75 * (t - 1/15)) →
  optimal_speed d t = 56.25 :=
by
  intros d t h1 h2
  have h_d_eq_45 := h1
  have h_d_eq_75 := h2
  sorry

end Mr_Pendearly_optimal_speed_l1297_129764


namespace black_king_eventually_in_check_l1297_129758

theorem black_king_eventually_in_check 
  (n : ℕ) (h1 : n = 1000) (r : ℕ) (h2 : r = 499)
  (rooks : Fin r → (ℕ × ℕ)) (king : ℕ × ℕ)
  (take_not_allowed : ∀ rk : Fin r, rooks rk ≠ king) :
  ∃ m : ℕ, m ≤ 1000 ∧ (∃ t : Fin r, rooks t = king) :=
by
  sorry

end black_king_eventually_in_check_l1297_129758


namespace washing_machine_heavy_washes_l1297_129768

theorem washing_machine_heavy_washes
  (H : ℕ)                                  -- The number of heavy washes
  (heavy_wash_gallons : ℕ := 20)            -- Gallons of water for a heavy wash
  (regular_wash_gallons : ℕ := 10)          -- Gallons of water for a regular wash
  (light_wash_gallons : ℕ := 2)             -- Gallons of water for a light wash
  (num_regular_washes : ℕ := 3)             -- Number of regular washes
  (num_light_washes : ℕ := 1)               -- Number of light washes
  (num_bleach_rinses : ℕ := 2)              -- Number of bleach rinses (extra light washes)
  (total_water_needed : ℕ := 76)            -- Total gallons of water needed
  (h_regular_wash_water : num_regular_washes * regular_wash_gallons = 30)
  (h_light_wash_water : num_light_washes * light_wash_gallons = 2)
  (h_bleach_rinse_water : num_bleach_rinses * light_wash_gallons = 4) :
  20 * H + 30 + 2 + 4 = 76 → H = 2 :=
by
  intros
  sorry

end washing_machine_heavy_washes_l1297_129768


namespace zeros_of_f_l1297_129705

noncomputable def f (x : ℝ) : ℝ := x^3 - 16 * x

theorem zeros_of_f :
  ∃ a b c : ℝ, (a = -4) ∧ (b = 0) ∧ (c = 4) ∧ (f a = 0) ∧ (f b = 0) ∧ (f c = 0) :=
by
  sorry

end zeros_of_f_l1297_129705


namespace find_k_l1297_129701

variable (m n p k : ℝ)

-- Conditions
def cond1 : Prop := m = 2 * n + 5
def cond2 : Prop := p = 3 * m - 4
def cond3 : Prop := m + 4 = 2 * (n + k) + 5
def cond4 : Prop := p + 3 = 3 * (m + 4) - 4

theorem find_k (h1 : cond1 m n)
               (h2 : cond2 m p)
               (h3 : cond3 m n k)
               (h4 : cond4 m p) :
               k = 2 :=
  sorry

end find_k_l1297_129701


namespace total_points_earned_l1297_129773

def defeated_enemies := 15
def points_per_enemy := 12
def level_completion_points := 20
def special_challenges_completed := 5
def points_per_special_challenge := 10

theorem total_points_earned :
  defeated_enemies * points_per_enemy
  + level_completion_points
  + special_challenges_completed * points_per_special_challenge = 250 :=
by
  -- The proof would be developed here.
  sorry

end total_points_earned_l1297_129773


namespace sum_of_all_N_l1297_129712

-- Define the machine's processing rules
def process (N : ℕ) : ℕ :=
  if N % 2 = 1 then 4 * N + 2 else N / 2

-- Define the 6-step process starting from N
def six_steps (N : ℕ) : ℕ :=
  process (process (process (process (process (process N)))))

-- Definition for the main theorem
theorem sum_of_all_N (N : ℕ) : six_steps N = 10 → N = 640 :=
by 
  sorry

end sum_of_all_N_l1297_129712


namespace unique_solution_l1297_129738

noncomputable def f (x : ℝ) : ℝ := 2^x + 3^x + 6^x

theorem unique_solution : ∀ x : ℝ, f x = 7^x ↔ x = 2 :=
by
  sorry

end unique_solution_l1297_129738
