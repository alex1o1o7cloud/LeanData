import Mathlib

namespace find_interval_l1160_116052

theorem find_interval (n : ℕ) 
  (h1 : n < 500) 
  (h2 : n ∣ 9999) 
  (h3 : n + 4 ∣ 99) : (1 ≤ n) ∧ (n ≤ 125) := 
sorry

end find_interval_l1160_116052


namespace rhombus_area_l1160_116080

theorem rhombus_area
  (d1 d2 : ℝ)
  (hd1 : d1 = 14)
  (hd2 : d2 = 20) :
  (d1 * d2) / 2 = 140 := by
  -- Problem: Given diagonals of length 14 cm and 20 cm,
  -- prove that the area of the rhombus is 140 square centimeters.
  sorry

end rhombus_area_l1160_116080


namespace solution_set_l1160_116085

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set (c1 : ∀ x : ℝ, f x + f' x > 1)
                     (c2 : f 0 = 2) :
  {x : ℝ | e^x * f x > e^x + 1} = {x : ℝ | 0 < x} :=
sorry

end solution_set_l1160_116085


namespace pine_tree_taller_than_birch_l1160_116060

def height_birch : ℚ := 49 / 4
def height_pine : ℚ := 74 / 4

def height_difference : ℚ :=
  height_pine - height_birch

theorem pine_tree_taller_than_birch :
  height_difference = 25 / 4 :=
by
  sorry

end pine_tree_taller_than_birch_l1160_116060


namespace connie_marbles_l1160_116076

-- Define the initial number of marbles that Connie had
def initial_marbles : ℝ := 73.5

-- Define the number of marbles that Connie gave away
def marbles_given : ℝ := 70.3

-- Define the expected number of marbles remaining
def marbles_remaining : ℝ := 3.2

-- State the theorem: prove that initial_marbles - marbles_given = marbles_remaining
theorem connie_marbles :
  initial_marbles - marbles_given = marbles_remaining :=
sorry

end connie_marbles_l1160_116076


namespace complex_division_l1160_116067

theorem complex_division :
  (1 - 2 * Complex.I) / (2 + Complex.I) = -Complex.I :=
by sorry

end complex_division_l1160_116067


namespace int_fraction_not_integer_l1160_116004

theorem int_fraction_not_integer (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ (k : ℤ), a^2 + b^2 = k * (a^2 - b^2) := 
sorry

end int_fraction_not_integer_l1160_116004


namespace quadratic_has_real_roots_l1160_116032

theorem quadratic_has_real_roots (k : ℝ) :
  (∃ (x : ℝ), (k-2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ (3 / 2) ∧ k ≠ 2) :=
by
  sorry

end quadratic_has_real_roots_l1160_116032


namespace area_correct_l1160_116014

open BigOperators

def Rectangle (PQ RS : ℕ) := PQ * RS

def PointOnSegment (a b : ℕ) (ratio : ℚ) : ℚ :=
ratio * (b - a)

def area_of_PTUS : ℚ :=
Rectangle 10 6 - (0.5 * 6 * (10 / 3) + 0.5 * 10 * 6)

theorem area_correct :
  area_of_PTUS = 20 := by
  sorry

end area_correct_l1160_116014


namespace range_of_f_l1160_116020

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f : Set.Icc (-(3 / 2)) 3 = Set.image f (Set.Icc 0 (Real.pi / 2)) :=
  sorry

end range_of_f_l1160_116020


namespace cannot_use_square_difference_formula_l1160_116050

theorem cannot_use_square_difference_formula (x y : ℝ) :
  ¬ ∃ a b : ℝ, (2 * x + 3 * y) * (-3 * y - 2 * x) = (a + b) * (a - b) :=
sorry

end cannot_use_square_difference_formula_l1160_116050


namespace intersect_setA_setB_l1160_116038

def setA : Set ℝ := {x | x < 2}
def setB : Set ℝ := {x | 3 - 2 * x > 0}

theorem intersect_setA_setB :
  setA ∩ setB = {x | x < 3 / 2} :=
by
  -- proof goes here
  sorry

end intersect_setA_setB_l1160_116038


namespace number_of_friends_l1160_116075

/- Define the conditions -/
def sandwiches_per_friend : Nat := 3
def total_sandwiches : Nat := 12

/- Define the mathematical statement to be proven -/
theorem number_of_friends : (total_sandwiches / sandwiches_per_friend) = 4 :=
by
  sorry

end number_of_friends_l1160_116075


namespace necessary_but_not_sufficient_l1160_116089

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x, x ≥ a → x^2 - x - 2 ≥ 0) ∧ (∃ x, x ≥ a ∧ ¬(x^2 - x - 2 ≥ 0)) ↔ a ≥ 2 := 
sorry

end necessary_but_not_sufficient_l1160_116089


namespace max_value_ineq_l1160_116045

theorem max_value_ineq (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end max_value_ineq_l1160_116045


namespace value_of_x_for_real_y_l1160_116064

theorem value_of_x_for_real_y (x y : ℝ) (h : 4 * y^2 - 2 * x * y + 2 * x + 9 = 0) : x ≤ -3 ∨ x ≥ 12 :=
sorry

end value_of_x_for_real_y_l1160_116064


namespace felix_chopped_at_least_91_trees_l1160_116084

def cost_to_sharpen := 5
def total_spent := 35
def trees_per_sharpen := 13

theorem felix_chopped_at_least_91_trees :
  (total_spent / cost_to_sharpen) * trees_per_sharpen = 91 := by
  sorry

end felix_chopped_at_least_91_trees_l1160_116084


namespace library_books_l1160_116063

theorem library_books (a : ℕ) (R L : ℕ) :
  (∃ R, a = 12 * R + 7) ∧ (∃ L, a = 25 * L - 5) ∧ 500 < a ∧ a < 650 → a = 595 :=
by
  sorry

end library_books_l1160_116063


namespace czakler_inequality_czakler_equality_pairs_l1160_116000

theorem czakler_inequality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (xy - 10)^2 ≥ 64 :=
sorry

theorem czakler_equality_pairs (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
(xy - 10)^2 = 64 ↔ (x, y) = (1,2) ∨ (x, y) = (-3, -6) :=
sorry

end czakler_inequality_czakler_equality_pairs_l1160_116000


namespace johns_meeting_distance_l1160_116081

theorem johns_meeting_distance (d t: ℝ) 
    (h1 : d = 40 * (t + 1.5))
    (h2 : d - 40 = 60 * (t - 2)) :
    d = 420 :=
by sorry

end johns_meeting_distance_l1160_116081


namespace probability_of_selection_l1160_116099

theorem probability_of_selection (total_students : ℕ) (eliminated_students : ℕ) (groups : ℕ) (selected_students : ℕ)
(h1 : total_students = 1003) 
(h2 : eliminated_students = 3)
(h3 : groups = 20)
(h4 : selected_students = 50) : 
(selected_students : ℝ) / (total_students : ℝ) = 50 / 1003 :=
by
  sorry

end probability_of_selection_l1160_116099


namespace turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l1160_116009

-- Definitions based on conditions
def turnover_first_four_days : ℝ := 450
def turnover_fifth_day : ℝ := 0.12 * turnover_first_four_days
def total_turnover_five_days : ℝ := turnover_first_four_days + turnover_fifth_day

-- Proof statement for part 1
theorem turnover_five_days_eq_504 :
  total_turnover_five_days = 504 := 
sorry

-- Definitions and conditions for part 2
def turnover_february : ℝ := 350
def turnover_april : ℝ := total_turnover_five_days
def growth_rate (x : ℝ) : Prop := (1 + x)^2 * turnover_february = turnover_april

-- Proof statement for part 2
theorem monthly_growth_rate_eq_20_percent :
  ∃ x : ℝ, growth_rate x ∧ x = 0.2 := 
sorry

end turnover_five_days_eq_504_monthly_growth_rate_eq_20_percent_l1160_116009


namespace range_of_a_l1160_116022

noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x < 1 then a^x else (a - 3) * x + 4 * a

theorem range_of_a (a : ℝ) (h : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 0 < a ∧ a ≤ 3 / 4 :=
sorry

end range_of_a_l1160_116022


namespace distance_between_P1_and_P2_l1160_116091

-- Define the two points
def P1 : ℝ × ℝ := (2, 3)
def P2 : ℝ × ℝ := (5, 10)

-- Define the distance function
noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Define the theorem we want to prove
theorem distance_between_P1_and_P2 :
  distance P1 P2 = Real.sqrt 58 :=
by sorry

end distance_between_P1_and_P2_l1160_116091


namespace distance_between_points_l1160_116029

def distance_on_line (a b : ℝ) : ℝ := |b - a|

theorem distance_between_points (a b : ℝ) : distance_on_line a b = |b - a| :=
by sorry

end distance_between_points_l1160_116029


namespace fencing_required_for_field_l1160_116001

noncomputable def fence_length (L W : ℕ) : ℕ := 2 * W + L

theorem fencing_required_for_field :
  ∀ (L W : ℕ), (L = 20) → (440 = L * W) → fence_length L W = 64 :=
by
  intros L W hL hA
  sorry

end fencing_required_for_field_l1160_116001


namespace problem_subtraction_of_negatives_l1160_116090

theorem problem_subtraction_of_negatives :
  12.345 - (-3.256) = 15.601 :=
sorry

end problem_subtraction_of_negatives_l1160_116090


namespace arithmetic_square_root_of_3_neg_2_l1160_116011

theorem arithmetic_square_root_of_3_neg_2 : Real.sqrt (3 ^ (-2: Int)) = 1 / 3 := 
by 
  sorry

end arithmetic_square_root_of_3_neg_2_l1160_116011


namespace sum_fractions_l1160_116033

theorem sum_fractions : 
  (1/2 + 1/6 + 1/12 + 1/20 + 1/30 + 1/42 = 6/7) :=
by
  sorry

end sum_fractions_l1160_116033


namespace geometricSeqMinimumValue_l1160_116082

noncomputable def isMinimumValue (a : ℕ → ℝ) (n m : ℕ) (value : ℝ) : Prop :=
  ∀ b : ℝ, (1 / a n + b / a m) ≥ value

theorem geometricSeqMinimumValue {a : ℕ → ℝ}
  (h1 : ∀ n, a n > 0)
  (h2 : a 7 = (Real.sqrt 2) / 2)
  (h3 : ∀ n, ∀ m, a n * a m = a (n + m)) :
  isMinimumValue a 3 11 4 :=
sorry

end geometricSeqMinimumValue_l1160_116082


namespace total_wheels_of_four_wheelers_l1160_116040

-- Define the number of four-wheelers and wheels per four-wheeler
def number_of_four_wheelers : ℕ := 13
def wheels_per_four_wheeler : ℕ := 4

-- Prove the total number of wheels for the 13 four-wheelers
theorem total_wheels_of_four_wheelers : (number_of_four_wheelers * wheels_per_four_wheeler) = 52 :=
by sorry

end total_wheels_of_four_wheelers_l1160_116040


namespace monthly_income_of_B_l1160_116028

variable (x y : ℝ)

-- Monthly incomes in the ratio 5:6
axiom income_ratio (A_income B_income : ℝ) : A_income = 5 * x ∧ B_income = 6 * x

-- Monthly expenditures in the ratio 3:4
axiom expenditure_ratio (A_expenditure B_expenditure : ℝ) : A_expenditure = 3 * y ∧ B_expenditure = 4 * y

-- Savings of A and B
axiom savings_A (A_income A_expenditure : ℝ) : 1800 = A_income - A_expenditure
axiom savings_B (B_income B_expenditure : ℝ) : 1600 = B_income - B_expenditure

-- The theorem to prove
theorem monthly_income_of_B (B_income : ℝ) (x y : ℝ) 
  (h1 : A_income = 5 * x)
  (h2 : B_income = 6 * x)
  (h3: A_expenditure = 3 * y)
  (h4: B_expenditure = 4 * y)
  (h5 : 1800 = 5 * x - 3 * y)
  (h6 : 1600 = 6 * x - 4 * y)
  : B_income = 7200 := by
  sorry

end monthly_income_of_B_l1160_116028


namespace find_coals_per_bag_l1160_116021

open Nat

variable (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ)

def coal_per_bag (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ) : ℕ :=
  (totalTime / timePerSet) * burnRate / totalBags

theorem find_coals_per_bag :
  coal_per_bag 15 20 240 3 = 60 :=
by
  sorry

end find_coals_per_bag_l1160_116021


namespace inequality_solution_l1160_116092

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end inequality_solution_l1160_116092


namespace larger_of_two_numbers_l1160_116059

  theorem larger_of_two_numbers (x y : ℕ) 
    (h₁ : x + y = 37) 
    (h₂ : x - y = 5) 
    : x = 21 :=
  sorry
  
end larger_of_two_numbers_l1160_116059


namespace incorrect_statement_l1160_116071

noncomputable def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem incorrect_statement
  (a b c : ℤ) (h₀ : a ≠ 0)
  (h₁ : 2 * a + b = 0)
  (h₂ : f a b c 1 = 3)
  (h₃ : f a b c 2 = 8) :
  ¬ (f a b c (-1) = 0) :=
sorry

end incorrect_statement_l1160_116071


namespace square_distance_between_intersections_l1160_116069

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 7)^2 + y^2 = 4

-- Problem: Prove the square of the distance between intersection points P and Q
theorem square_distance_between_intersections :
  (∃ (x y1 y2 : ℝ), circle1 x y1 ∧ circle2 x y1 ∧ circle1 x y2 ∧ circle2 x y2 ∧ y1 ≠ y2) →
  ∃ d : ℝ, d^2 = 15.3664 :=
by
  sorry

end square_distance_between_intersections_l1160_116069


namespace least_days_to_repay_twice_l1160_116061

-- Define the initial conditions
def borrowed_amount : ℝ := 15
def daily_interest_rate : ℝ := 0.10
def interest_per_day : ℝ := borrowed_amount * daily_interest_rate
def total_amount_to_repay : ℝ := 2 * borrowed_amount

-- Define the condition we want to prove
theorem least_days_to_repay_twice : ∃ (x : ℕ), (borrowed_amount + interest_per_day * x) ≥ total_amount_to_repay ∧ x = 10 :=
by
  sorry

end least_days_to_repay_twice_l1160_116061


namespace minimum_shirts_to_save_money_by_using_Acme_l1160_116044

-- Define the cost functions for Acme and Gamma
def Acme_cost (x : ℕ) : ℕ := 60 + 8 * x
def Gamma_cost (x : ℕ) : ℕ := 12 * x

-- State the theorem to prove that for x = 16, Acme is cheaper than Gamma
theorem minimum_shirts_to_save_money_by_using_Acme : ∀ x ≥ 16, Acme_cost x < Gamma_cost x :=
by
  intros x hx
  sorry

end minimum_shirts_to_save_money_by_using_Acme_l1160_116044


namespace determine_d_l1160_116037

variables (a b c d : ℝ)

-- Conditions given in the problem
def condition1 (a b d : ℝ) : Prop := d / a = (d - 25) / b
def condition2 (b c d : ℝ) : Prop := d / b = (d - 15) / c
def condition3 (a c d : ℝ) : Prop := d / a = (d - 35) / c

-- Final statement to prove
theorem determine_d (a b c : ℝ) (d : ℝ) :
    condition1 a b d ∧ condition2 b c d ∧ condition3 a c d → d = 75 :=
by sorry

end determine_d_l1160_116037


namespace find_k_x_l1160_116095

-- Define the nonzero polynomial condition
def nonzero_poly (p : Polynomial ℝ) : Prop :=
  ¬ (p = 0)

-- Define the conditions from the problem statement
def conditions (h k : Polynomial ℝ) : Prop :=
  nonzero_poly h ∧ nonzero_poly k ∧ (h.comp k = h * k) ∧ (k.eval 3 = 58)

-- State the main theorem to be proven
theorem find_k_x (h k : Polynomial ℝ) (cond : conditions h k) : 
  k = Polynomial.C 1 + Polynomial.C 49 * Polynomial.X + Polynomial.C (-49) * Polynomial.X^2 :=
sorry

end find_k_x_l1160_116095


namespace equal_division_of_cookie_l1160_116093

theorem equal_division_of_cookie (total_area : ℝ) (friends : ℕ) (area_per_person : ℝ) 
  (h1 : total_area = 81.12) 
  (h2 : friends = 6) 
  (h3 : area_per_person = total_area / friends) : 
  area_per_person = 13.52 :=
by 
  sorry

end equal_division_of_cookie_l1160_116093


namespace sin_half_angle_product_lt_quarter_l1160_116019

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h_sum : A + B + C = Real.pi) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end sin_half_angle_product_lt_quarter_l1160_116019


namespace max_value_f_l1160_116068

theorem max_value_f (x y z : ℝ) (hxyz : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (1 - y * z + z) * (1 - x * z + x) * (1 - x * y + y) ≤ 1 :=
sorry

end max_value_f_l1160_116068


namespace probability_green_ball_l1160_116010

/-- 
Given three containers with specific numbers of red and green balls, 
and the probability of selecting each container being equal, 
the probability of picking a green ball when choosing a container randomly is 7/12.
-/
theorem probability_green_ball :
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  (green_I + green_II + green_III) = 7 / 12 :=
by 
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  have : (green_I + green_II + green_III) = (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) := by rfl
  have : (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) = (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) := by rfl
  have : (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) = (1 / 9 + 2 / 9 + 1 / 4) := by rfl
  have : (1 / 9 + 2 / 9 + 1 / 4) = (4 / 36 + 8 / 36 + 9 / 36) := by rfl
  have : (4 / 36 + 8 / 36 + 9 / 36) = 21 / 36 := by rfl
  have : 21 / 36 = 7 / 12 := by rfl
  rfl

end probability_green_ball_l1160_116010


namespace smallest_n_divisibility_l1160_116047

theorem smallest_n_divisibility (n : ℕ) (h : 1 ≤ n) :
  (∀ k, 1 ≤ k ∧ k ≤ n → n^3 - n ∣ k) ∨ (∃ k, 1 ≤ k ∧ k ≤ n ∧ ¬ (n^3 - n ∣ k)) :=
sorry

end smallest_n_divisibility_l1160_116047


namespace sum_of_conjugates_eq_30_l1160_116074

theorem sum_of_conjugates_eq_30 :
  (15 - Real.sqrt 2023) + (15 + Real.sqrt 2023) = 30 :=
sorry

end sum_of_conjugates_eq_30_l1160_116074


namespace solve_equations_l1160_116053

-- Prove that the solutions to the given equations are correct.
theorem solve_equations :
  (∀ x : ℝ, (x * (x - 4) = 2 * x - 8) ↔ (x = 4 ∨ x = 2)) ∧
  (∀ x : ℝ, ((2 * x) / (2 * x - 3) - (4 / (2 * x + 3)) = 1) ↔ (x = 10.5)) :=
by
  sorry

end solve_equations_l1160_116053


namespace cos_2alpha_plus_5pi_by_12_l1160_116002

open Real

noncomputable def alpha : ℝ := sorry

axiom alpha_obtuse : π / 2 < alpha ∧ alpha < π

axiom sin_alpha_plus_pi_by_3 : sin (alpha + π / 3) = -4 / 5

theorem cos_2alpha_plus_5pi_by_12 : 
  cos (2 * alpha + 5 * π / 12) = 17 * sqrt 2 / 50 :=
by sorry

end cos_2alpha_plus_5pi_by_12_l1160_116002


namespace ab_plus_cd_111_333_l1160_116025

theorem ab_plus_cd_111_333 (a b c d : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a + b + d = 5) 
  (h3 : a + c + d = 20) 
  (h4 : b + c + d = 15) : 
  a * b + c * d = 111.333 := 
by
  sorry

end ab_plus_cd_111_333_l1160_116025


namespace tan_alpha_l1160_116013

theorem tan_alpha (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan β = 1 / 3) :
  Real.tan α = 2 / 9 := by
  sorry

end tan_alpha_l1160_116013


namespace greatest_consecutive_integers_sum_55_l1160_116046

theorem greatest_consecutive_integers_sum_55 :
  ∃ N a : ℤ, (N * (2 * a + N - 1)) = 110 ∧ (∀ M a' : ℤ, (M * (2 * a' + M - 1)) = 110 → N ≥ M) :=
sorry

end greatest_consecutive_integers_sum_55_l1160_116046


namespace area_of_polygon_ABCDEF_l1160_116077

-- Definitions based on conditions
def AB : ℕ := 8
def BC : ℕ := 10
def DC : ℕ := 5
def FA : ℕ := 7
def GF : ℕ := 3
def ED : ℕ := 7
def height_GF_ED : ℕ := 2

-- Area calculations based on given conditions
def area_ABCG : ℕ := AB * BC
def area_trapezoid_GFED : ℕ := (GF + ED) * height_GF_ED / 2

-- Proof statement
theorem area_of_polygon_ABCDEF :
  area_ABCG - area_trapezoid_GFED = 70 :=
by
  simp [area_ABCG, area_trapezoid_GFED]
  sorry

end area_of_polygon_ABCDEF_l1160_116077


namespace length_of_cloth_l1160_116083

theorem length_of_cloth (L : ℝ) (h : 35 = (L + 4) * (35 / L - 1)) : L = 10 :=
sorry

end length_of_cloth_l1160_116083


namespace distance_PQ_is_12_miles_l1160_116051

-- Define the conditions
def average_speed_PQ := 40 -- mph
def average_speed_QP := 45 -- mph
def time_difference := 2 -- minutes

-- Main proof statement to show that the distance is 12 miles
theorem distance_PQ_is_12_miles 
    (x : ℝ) 
    (h1 : average_speed_PQ > 0) 
    (h2 : average_speed_QP > 0) 
    (h3 : abs ((x / average_speed_PQ * 60) - (x / average_speed_QP * 60)) = time_difference) 
    : x = 12 := 
by
  sorry

end distance_PQ_is_12_miles_l1160_116051


namespace focus_of_parabola_l1160_116055

theorem focus_of_parabola (a k : ℝ) (h_eq : ∀ x : ℝ, k = 6 ∧ a = 9) :
  (0, (1 / (4 * a)) + k) = (0, 217 / 36) := sorry

end focus_of_parabola_l1160_116055


namespace equation1_solution_equation2_solution_l1160_116056

-- Equation 1 Statement
theorem equation1_solution (x : ℝ) : 
  (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ↔ x = -20 :=
by sorry

-- Equation 2 Statement
theorem equation2_solution (x : ℝ) : 
  (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ↔ x = 67 / 23 :=
by sorry

end equation1_solution_equation2_solution_l1160_116056


namespace sufficient_and_necessary_condition_l1160_116031

variable (a_n : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
variable (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
variable (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d))

theorem sufficient_and_necessary_condition (d : ℚ) (h_arith_seq : ∀ n : ℕ, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n : ℕ, S n = n * (a_n 1 + (n - 1) / 2 * d)) :
  (d > 0) ↔ (S 4 + S 6 > 2 * S 5) := by
  sorry

end sufficient_and_necessary_condition_l1160_116031


namespace problem_statement_l1160_116078

theorem problem_statement (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) = x * f y + y * f x) →
  (∀ x : ℝ, x > 1 → f x < 0) →

  -- Conclusion 1: f(1) = 0, f(-1) = 0
  f 1 = 0 ∧ f (-1) = 0 ∧

  -- Conclusion 2: f(x) is an odd function: f(-x) = -f(x)
  (∀ x : ℝ, f (-x) = -f x) ∧

  -- Conclusion 3: f(x) is decreasing on (1, +∞)
  (∀ x1 x2 : ℝ, x1 > 1 → x2 > 1 → x1 < x2 → f x1 < f x2) := sorry

end problem_statement_l1160_116078


namespace packs_of_green_bouncy_balls_l1160_116023

/-- Maggie bought 10 bouncy balls in each pack of red, yellow, and green bouncy balls.
    She bought 4 packs of red bouncy balls, 8 packs of yellow bouncy balls, and some 
    packs of green bouncy balls. In total, she bought 160 bouncy balls. This theorem 
    aims to prove how many packs of green bouncy balls Maggie bought. 
 -/
theorem packs_of_green_bouncy_balls (red_packs : ℕ) (yellow_packs : ℕ) (total_balls : ℕ) (balls_per_pack : ℕ) 
(pack : ℕ) :
  red_packs = 4 →
  yellow_packs = 8 →
  balls_per_pack = 10 →
  total_balls = 160 →
  red_packs * balls_per_pack + yellow_packs * balls_per_pack + pack * balls_per_pack = total_balls →
  pack = 4 :=
by
  intros h_red h_yellow h_balls_per_pack h_total_balls h_eq
  sorry

end packs_of_green_bouncy_balls_l1160_116023


namespace intersection_A_B_l1160_116035

noncomputable def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by 
  sorry

end intersection_A_B_l1160_116035


namespace candies_distribution_l1160_116066

theorem candies_distribution (C : ℕ) (hC : C / 150 = C / 300 + 24) : C / 150 = 48 :=
by sorry

end candies_distribution_l1160_116066


namespace labor_hired_l1160_116003

noncomputable def Q_d (P : ℝ) : ℝ := 60 - 14 * P
noncomputable def Q_s (P : ℝ) : ℝ := 20 + 6 * P
noncomputable def MPL (L : ℝ) : ℝ := 160 / (L^2)
def wage : ℝ := 5

theorem labor_hired (L P : ℝ) (h_eq_price: 60 - 14 * P = 20 + 6 * P) (h_eq_wage: 160 / (L^2) * 2 = wage) :
  L = 8 :=
by
  have h1 : 60 - 14 * P = 20 + 6 * P := h_eq_price
  have h2 : 160 / (L^2) * 2 = wage := h_eq_wage
  sorry

end labor_hired_l1160_116003


namespace breadth_of_rectangular_plot_l1160_116012

theorem breadth_of_rectangular_plot (b l A : ℝ) (h1 : l = 3 * b) (h2 : A = 588) (h3 : A = l * b) : b = 14 :=
by
  -- We start our proof here
  sorry

end breadth_of_rectangular_plot_l1160_116012


namespace power_function_through_point_l1160_116072

noncomputable def f : ℝ → ℝ := sorry

theorem power_function_through_point (h : ∀ x, ∃ a : ℝ, f x = x^a) (h1 : f 3 = 27) :
  f x = x^3 :=
sorry

end power_function_through_point_l1160_116072


namespace simpsons_hats_l1160_116088

variable (S : ℕ)
variable (O : ℕ)

-- Define the conditions: O'Brien's hats before losing one
def obriens_hats_before : Prop := O = 2 * S + 5

-- Define the current number of O'Brien's hats
def obriens_current_hats : Prop := O = 34 + 1

-- Main theorem statement
theorem simpsons_hats : obriens_hats_before S O ∧ obriens_current_hats O → S = 15 := 
by
  sorry

end simpsons_hats_l1160_116088


namespace no_complete_divisibility_l1160_116058

-- Definition of non-divisibility
def not_divides (m n : ℕ) := ¬ (m ∣ n)

theorem no_complete_divisibility (a b c d : ℕ) (h : a * d - b * c > 1) : 
  not_divides (a * d - b * c) a ∨ not_divides (a * d - b * c) b ∨ not_divides (a * d - b * c) c ∨ not_divides (a * d - b * c) d :=
by 
  sorry

end no_complete_divisibility_l1160_116058


namespace ferris_wheel_seats_l1160_116057

theorem ferris_wheel_seats (S : ℕ) (h1 : ∀ (p : ℕ), p = 9) (h2 : ∀ (r : ℕ), r = 18) (h3 : 9 * S = 18) : S = 2 :=
by
  sorry

end ferris_wheel_seats_l1160_116057


namespace jose_speed_l1160_116018

theorem jose_speed
  (distance : ℕ) (time : ℕ)
  (h_distance : distance = 4)
  (h_time : time = 2) :
  distance / time = 2 := by
  sorry

end jose_speed_l1160_116018


namespace eq_circle_value_of_k_l1160_116039

noncomputable def circle_center : Prod ℝ ℝ := (2, 3)
noncomputable def circle_radius := 2
noncomputable def line_equation (k : ℝ) : ℝ → ℝ := fun x => k * x - 1
noncomputable def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 4

theorem eq_circle (x y : ℝ) : 
  circle_equation x y ↔ (x - 2)^2 + (y - 3)^2 = 4 := 
by sorry

theorem value_of_k (k : ℝ) : 
  (∀ M N : Prod ℝ ℝ, 
  circle_equation M.1 M.2 ∧ circle_equation N.1 N.2 ∧ 
  line_equation k M.1 = M.2 ∧ line_equation k N.1 = N.2 ∧ 
  M ≠ N ∧ 
  (circle_center.1 - M.1) * (circle_center.1 - N.1) + 
  (circle_center.2 - M.2) * (circle_center.2 - N.2) = 0) → 
  (k = 1 ∨ k = 7) := 
by sorry

end eq_circle_value_of_k_l1160_116039


namespace solve_abs_equation_l1160_116062

theorem solve_abs_equation (x : ℝ) (h : |2001 * x - 2001| = 2001) : x = 0 ∨ x = 2 := by
  sorry

end solve_abs_equation_l1160_116062


namespace total_canoes_proof_l1160_116043

def n_canoes_january : ℕ := 5
def n_canoes_february : ℕ := 3 * n_canoes_january
def n_canoes_march : ℕ := 3 * n_canoes_february
def n_canoes_april : ℕ := 3 * n_canoes_march

def total_canoes_built : ℕ :=
  n_canoes_january + n_canoes_february + n_canoes_march + n_canoes_april

theorem total_canoes_proof : total_canoes_built = 200 := 
  by
  sorry

end total_canoes_proof_l1160_116043


namespace min_m_value_l1160_116065

noncomputable def f (x a : ℝ) : ℝ := 2 ^ (abs (x - a))

theorem min_m_value :
  ∀ a, (∀ x, f (1 + x) a = f (1 - x) a) →
  ∃ m : ℝ, (∀ x : ℝ, x ≥ m → ∀ y : ℝ, y ≥ x → f y a ≥ f x a) ∧ m = 1 :=
by
  intros a h
  sorry

end min_m_value_l1160_116065


namespace minimum_milk_candies_l1160_116049

/-- A supermarket needs to purchase candies with the following conditions:
 1. The number of watermelon candies is at most 3 times the number of chocolate candies.
 2. The number of milk candies is at least 4 times the number of chocolate candies.
 3. The sum of chocolate candies and watermelon candies is at least 2020.

 Prove that the minimum number of milk candies that need to be purchased is 2020. -/
theorem minimum_milk_candies (x y z : ℕ)
  (h1 : y ≤ 3 * x)
  (h2 : z ≥ 4 * x)
  (h3 : x + y ≥ 2020) :
  z ≥ 2020 :=
sorry

end minimum_milk_candies_l1160_116049


namespace identify_jars_l1160_116054

namespace JarIdentification

/-- Definitions of Jar labels -/
inductive JarLabel
| Nickels
| Dimes
| Nickels_and_Dimes

open JarLabel

/-- Mislabeling conditions for each jar -/
def mislabeled (jarA : JarLabel) (jarB : JarLabel) (jarC : JarLabel) : Prop :=
  ((jarA ≠ Nickels) ∧ (jarB ≠ Dimes) ∧ (jarC ≠ Nickels_and_Dimes)) ∧
  ((jarC = Nickels ∨ jarC = Dimes))

/-- Given the result of a coin draw from the jar labeled "Nickels and Dimes" -/
def jarIdentity (jarA jarB jarC : JarLabel) (drawFromC : String) : Prop :=
  if drawFromC = "Nickel" then
    jarC = Nickels ∧ jarA = Nickels_and_Dimes ∧ jarB = Dimes
  else if drawFromC = "Dime" then
    jarC = Dimes ∧ jarB = Nickels_and_Dimes ∧ jarA = Nickels
  else 
    false

/-- Main theorem to prove the identification of jars -/
theorem identify_jars (jarA jarB jarC : JarLabel) (draw : String)
  (h1 : mislabeled jarA jarB jarC) :
  jarIdentity jarA jarB jarC draw :=
by
  sorry

end JarIdentification

end identify_jars_l1160_116054


namespace not_exists_implies_bounds_l1160_116027

variable (a : ℝ)

/-- If there does not exist an x such that x^2 + (a - 1) * x + 1 < 0, then -1 ≤ a ∧ a ≤ 3. -/
theorem not_exists_implies_bounds : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 3) :=
by sorry

end not_exists_implies_bounds_l1160_116027


namespace inequalities_hold_l1160_116097

theorem inequalities_hold 
  (x y z a b c : ℕ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)   -- Given that x, y, z are positive integers
  (ha : a > 0) (hb : b > 0) (hc : c > 0)   -- Given that a, b, c are positive integers
  (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 ≤ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ∧ 
  x^3 + y^3 + z^3 ≤ a^3 + b^3 + c^3 ∧ 
  x^2 * y * z + y^2 * z * x + z^2 * x * y ≤ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by
  sorry

end inequalities_hold_l1160_116097


namespace ratio_brother_to_joanna_l1160_116042

/-- Definitions for the conditions -/
def joanna_money : ℝ := 8
def sister_money : ℝ := 4 -- since it's half of Joanna's money
def total_money : ℝ := 36

/-- Stating the theorem -/
theorem ratio_brother_to_joanna (x : ℝ) (h : joanna_money + 8*x + sister_money = total_money) :
  x = 3 :=
by 
  -- The ratio of brother's money to Joanna's money is 3:1
  sorry

end ratio_brother_to_joanna_l1160_116042


namespace sector_area_l1160_116016

theorem sector_area (r : ℝ) (α : ℝ) (h1 : 2 * r + α * r = 16) (h2 : α = 2) :
  1 / 2 * α * r^2 = 16 :=
by
  sorry

end sector_area_l1160_116016


namespace min_value_lemma_min_value_achieved_l1160_116024

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 + x)^2)

theorem min_value_lemma : ∀ (x : ℝ), f x ≥ Real.sqrt 5 := 
by
  intro x
  sorry

theorem min_value_achieved : ∃ (x : ℝ), f x = Real.sqrt 5 :=
by
  use 1 / 3
  sorry

end min_value_lemma_min_value_achieved_l1160_116024


namespace tree_leaves_not_shed_l1160_116007

-- Definitions of conditions based on the problem.
variable (initial_leaves : ℕ) (shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves : ℕ)

-- Setting the conditions
def conditions :=
  initial_leaves = 5000 ∧
  shed_week1 = initial_leaves / 5 ∧
  shed_week2 = 30 * (initial_leaves - shed_week1) / 100 ∧
  shed_week3 = 60 * shed_week2 / 100 ∧
  shed_week4 = 50 * (initial_leaves - shed_week1 - shed_week2 - shed_week3) / 100 ∧
  shed_week5 = 2 * shed_week3 / 3 ∧
  remaining_leaves = initial_leaves - shed_week1 - shed_week2 - shed_week3 - shed_week4 - shed_week5

-- The proof statement
theorem tree_leaves_not_shed (h : conditions initial_leaves shed_week1 shed_week2 shed_week3 shed_week4 shed_week5 remaining_leaves) :
  remaining_leaves = 560 :=
sorry

end tree_leaves_not_shed_l1160_116007


namespace total_test_points_l1160_116041

theorem total_test_points (total_questions two_point_questions four_point_questions points_per_two_question points_per_four_question : ℕ) 
  (h1 : total_questions = 40)
  (h2 : four_point_questions = 10)
  (h3 : points_per_two_question = 2)
  (h4 : points_per_four_question = 4)
  (h5 : two_point_questions = total_questions - four_point_questions)
  : (two_point_questions * points_per_two_question) + (four_point_questions * points_per_four_question) = 100 :=
by
  sorry

end total_test_points_l1160_116041


namespace manager_salary_l1160_116030

def avg_salary_employees := 1500
def num_employees := 20
def avg_salary_increase := 600
def num_total_people := num_employees + 1

def total_salary_employees := num_employees * avg_salary_employees
def new_avg_salary := avg_salary_employees + avg_salary_increase
def total_salary_with_manager := num_total_people * new_avg_salary

theorem manager_salary : total_salary_with_manager - total_salary_employees = 14100 :=
by
  sorry

end manager_salary_l1160_116030


namespace anna_more_candy_than_billy_l1160_116026

theorem anna_more_candy_than_billy :
  let anna_candy_per_house := 14
  let billy_candy_per_house := 11
  let anna_houses := 60
  let billy_houses := 75
  let anna_total_candy := anna_candy_per_house * anna_houses
  let billy_total_candy := billy_candy_per_house * billy_houses
  anna_total_candy - billy_total_candy = 15 :=
by
  sorry

end anna_more_candy_than_billy_l1160_116026


namespace problem1_problem2_l1160_116008

noncomputable def f (x a : ℝ) := |x - 2 * a|
noncomputable def g (x a : ℝ) := |x + a|

theorem problem1 (x m : ℝ): (∃ x, f x 1 - g x 1 ≥ m) → m ≤ 3 :=
by
  sorry

theorem problem2 (a : ℝ): (∀ x, f x a + g x a ≥ 3) → (a ≥ 1 ∨ a ≤ -1) :=
by
  sorry

end problem1_problem2_l1160_116008


namespace monotonicity_f_l1160_116098

open Set

noncomputable def f (a x : ℝ) : ℝ := a * x / (x - 1)

theorem monotonicity_f (a : ℝ) (h : a ≠ 0) :
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → (if a > 0 then f a x1 > f a x2 else if a < 0 then f a x1 < f a x2 else False)) :=
by
  sorry

end monotonicity_f_l1160_116098


namespace pirates_gold_coins_l1160_116048

theorem pirates_gold_coins (S a b c d e : ℕ) (h1 : a = S / 3) (h2 : b = S / 4) (h3 : c = S / 5) (h4 : d = S / 6) (h5 : e = 90) :
  S = 1800 :=
by
  -- Definitions and assumptions would go here
  sorry

end pirates_gold_coins_l1160_116048


namespace greatest_two_digit_multiple_of_17_l1160_116017

theorem greatest_two_digit_multiple_of_17 : 
  ∃ (n : ℕ), n < 100 ∧ n ≥ 10 ∧ 17 ∣ n ∧ ∀ m, m < 100 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ 85 :=
by
  use 85
  -- Prove conditions follow sorry
  sorry

end greatest_two_digit_multiple_of_17_l1160_116017


namespace polynomial_solution_characterization_l1160_116087

theorem polynomial_solution_characterization (P : ℝ → ℝ → ℝ) (h : ∀ x y z : ℝ, P x (2 * y * z) + P y (2 * z * x) + P z (2 * x * y) = P (x + y + z) (x * y + y * z + z * x)) :
  ∃ (a b : ℝ), ∀ x y : ℝ, P x y = a * x + b * (x^2 + 2 * y) :=
sorry

end polynomial_solution_characterization_l1160_116087


namespace range_of_a_for_common_points_l1160_116096

theorem range_of_a_for_common_points (a : ℝ) : (∃ x : ℝ, x > 0 ∧ ax^2 = Real.exp x) ↔ a ≥ Real.exp 2 / 4 :=
sorry

end range_of_a_for_common_points_l1160_116096


namespace min_total_fund_Required_l1160_116015

noncomputable def sell_price_A (x : ℕ) : ℕ := x + 10
noncomputable def cost_A (x : ℕ) : ℕ := 600
noncomputable def cost_B (x : ℕ) : ℕ := 400

def num_barrels_A_B_purchased (x : ℕ) := cost_A x / (sell_price_A x) = cost_B x / x

noncomputable def total_cost (m : ℕ) : ℕ := 10 * m + 10000

theorem min_total_fund_Required (price_A price_B m total : ℕ) :
  price_B = 20 →
  price_A = 30 →
  price_A = price_B + 10 →
  (num_barrels_A_B_purchased price_B) →
  total = total_cost m →
  m = 250 →
  total = 12500 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end min_total_fund_Required_l1160_116015


namespace red_balls_estimate_l1160_116079

/-- There are several red balls and 4 black balls in a bag.
Each ball is identical except for color.
A ball is drawn and put back into the bag. This process is repeated 100 times.
Among those 100 draws, 40 times a black ball is drawn.
Prove that the number of red balls (x) is 6. -/
theorem red_balls_estimate (x : ℕ) (h_condition : (4 / (4 + x) = 40 / 100)) : x = 6 :=
by
    sorry

end red_balls_estimate_l1160_116079


namespace quadratic_roots_satisfy_condition_l1160_116094
variable (x1 x2 m : ℝ)

theorem quadratic_roots_satisfy_condition :
  ( ∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1 + x2 = -m) ∧ 
    (x1 * x2 = 5) ∧ (x1 = 2 * |x2| - 3) ) →
  m = -9 / 2 :=
by
  sorry

end quadratic_roots_satisfy_condition_l1160_116094


namespace necessary_but_not_sufficient_condition_l1160_116005

-- Define the sets M and P
def M (x : ℝ) : Prop := x > 2
def P (x : ℝ) : Prop := x < 3

-- Statement of the problem
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (M x ∨ P x) → (x ∈ { y : ℝ | 2 < y ∧ y < 3 }) :=
sorry

end necessary_but_not_sufficient_condition_l1160_116005


namespace compare_log_values_l1160_116070

noncomputable def a : ℝ := (Real.log 2) / 2
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := (Real.log 5) / 5

theorem compare_log_values : c < a ∧ a < b := by
  -- Proof is omitted
  sorry

end compare_log_values_l1160_116070


namespace cakes_served_during_lunch_today_l1160_116006

-- Define the conditions as parameters
variables
  (L : ℕ)   -- Number of cakes served during lunch today
  (D : ℕ := 6)  -- Number of cakes served during dinner today
  (Y : ℕ := 3)  -- Number of cakes served yesterday
  (T : ℕ := 14)  -- Total number of cakes served

-- Define the theorem to prove L = 5
theorem cakes_served_during_lunch_today : L + D + Y = T → L = 5 :=
by
  sorry

end cakes_served_during_lunch_today_l1160_116006


namespace quadratic_inequality_l1160_116034

theorem quadratic_inequality (a b c d x1 x2 x3 x4 : ℝ)
  (h1 : x1 + x2 = -a) 
  (h2 : x1 * x2 = b)
  (h3 : x3 + x4 = -c)
  (h4 : x3 * x4 = d)
  (h5 : b > d)
  (h6 : b > 0)
  (h7 : d > 0) :
  a^2 - c^2 > b - d :=
by
  sorry

end quadratic_inequality_l1160_116034


namespace fernandez_family_children_l1160_116036

-- Conditions definition
variables (m : ℕ) -- age of the mother
variables (x : ℕ) -- number of children
variables (y : ℕ) -- average age of the children

-- Given conditions
def average_age_family (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + 50 + 70 + x * y) / (3 + x) = 25

def average_age_mother_children (m : ℕ) (x : ℕ) (y : ℕ) : Prop :=
  (m + x * y) / (1 + x) = 18

-- Goal statement
theorem fernandez_family_children
  (m : ℕ) (x : ℕ) (y : ℕ)
  (h1 : average_age_family m x y)
  (h2 : average_age_mother_children m x y) :
  x = 9 :=
sorry

end fernandez_family_children_l1160_116036


namespace alex_casey_meet_probability_l1160_116073

noncomputable def probability_meet : ℚ :=
  let L := (1:ℚ) / 3;
  let area_of_square := 1;
  let area_of_triangles := (1 / 2) * L ^ 2;
  let area_of_meeting_region := area_of_square - 2 * area_of_triangles;
  area_of_meeting_region / area_of_square

theorem alex_casey_meet_probability :
  probability_meet = 8 / 9 :=
by
  sorry

end alex_casey_meet_probability_l1160_116073


namespace neg_square_positive_l1160_116086

theorem neg_square_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := sorry

end neg_square_positive_l1160_116086
