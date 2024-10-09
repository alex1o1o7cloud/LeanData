import Mathlib

namespace chord_bisected_by_point_of_ellipse_l2328_232889

theorem chord_bisected_by_point_of_ellipse 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1)
  (bisecting_point : ∃ x y : ℝ, x = 4 ∧ y = 2) :
  ∃ a b c : ℝ, a = 1 ∧ b = 2 ∧ c = -8 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 :=
by
   sorry

end chord_bisected_by_point_of_ellipse_l2328_232889


namespace Jazmin_strip_width_l2328_232815

theorem Jazmin_strip_width (a b c : ℕ) (ha : a = 44) (hb : b = 33) (hc : c = 55) : Nat.gcd (Nat.gcd a b) c = 11 := by
  sorry

end Jazmin_strip_width_l2328_232815


namespace fraction_simplification_l2328_232800

-- We define the given fractions
def a := 3 / 7
def b := 2 / 9
def c := 5 / 12
def d := 1 / 4

-- We state the main theorem
theorem fraction_simplification : (a - b) / (c + d) = 13 / 42 := by
  -- Skipping proof for the equivalence problem
  sorry

end fraction_simplification_l2328_232800


namespace determinant_range_l2328_232814

theorem determinant_range (x : ℝ) : 
  (2 * x - (3 - x) > 0) ↔ (x > 1) :=
by
  sorry

end determinant_range_l2328_232814


namespace opposite_numbers_power_l2328_232842

theorem opposite_numbers_power (a b : ℝ) (h : a + b = 0) : (a + b) ^ 2023 = 0 :=
by 
  sorry

end opposite_numbers_power_l2328_232842


namespace measure_angle_4_l2328_232890

theorem measure_angle_4 (m1 m2 m3 m5 m6 m4 : ℝ) 
  (h1 : m1 = 82) 
  (h2 : m2 = 34) 
  (h3 : m3 = 19) 
  (h4 : m5 = m6 + 10) 
  (h5 : m1 + m2 + m3 + m5 + m6 = 180)
  (h6 : m4 + m5 + m6 = 180) : 
  m4 = 135 :=
by
  -- Placeholder for the full proof, omitted due to instructions
  sorry

end measure_angle_4_l2328_232890


namespace problem_statement_l2328_232873

namespace ProofProblem

variable (t : ℚ) (y : ℚ)

/-- Given equations and condition, we want to prove y = 21 / 2 -/
theorem problem_statement (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : y = 21 / 2 :=
by sorry

end ProofProblem

end problem_statement_l2328_232873


namespace value_of_expression_l2328_232856

theorem value_of_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2) : (1 / 3) * x ^ 8 * y ^ 9 = 2 / 3 :=
by
  -- Proof can be filled in here
  sorry

end value_of_expression_l2328_232856


namespace four_digit_number_exists_l2328_232813

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), 
  B = 3 * A ∧ 
  C = A + B ∧ 
  D = 3 * B ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ 
  1000 * A + 100 * B + 10 * C + D = 1349 :=
by {
  sorry 
}

end four_digit_number_exists_l2328_232813


namespace probability_of_die_showing_1_after_5_steps_l2328_232858

def prob_showing_1 (steps : ℕ) : ℚ :=
  if steps = 5 then 37 / 192 else 0

theorem probability_of_die_showing_1_after_5_steps :
  prob_showing_1 5 = 37 / 192 :=
sorry

end probability_of_die_showing_1_after_5_steps_l2328_232858


namespace evaluate_expression_at_2_l2328_232854

theorem evaluate_expression_at_2 : ∀ (x : ℕ), x = 2 → (x^x)^(x^(x^x)) = 4294967296 := by
  intros x h
  rw [h]
  sorry

end evaluate_expression_at_2_l2328_232854


namespace find_number_l2328_232857

-- Define the conditions: 0.80 * x - 20 = 60
variables (x : ℝ)
axiom condition : 0.80 * x - 20 = 60

-- State the theorem that x = 100 given the condition
theorem find_number : x = 100 :=
by
  sorry

end find_number_l2328_232857


namespace find_fifth_integer_l2328_232852

theorem find_fifth_integer (x y : ℤ) (h_pos : x > 0)
  (h_mean_median : (x + 2 + x + 7 + x + y) / 5 = x + 7) :
  y = 22 :=
sorry

end find_fifth_integer_l2328_232852


namespace balloons_remaining_l2328_232874

def bags_round : ℕ := 5
def balloons_per_bag_round : ℕ := 20
def bags_long : ℕ := 4
def balloons_per_bag_long : ℕ := 30
def balloons_burst : ℕ := 5

def total_round_balloons : ℕ := bags_round * balloons_per_bag_round
def total_long_balloons : ℕ := bags_long * balloons_per_bag_long
def total_balloons : ℕ := total_round_balloons + total_long_balloons
def balloons_left : ℕ := total_balloons - balloons_burst

theorem balloons_remaining : balloons_left = 215 := by 
  -- We leave this as sorry since the proof is not required
  sorry

end balloons_remaining_l2328_232874


namespace john_needs_20_nails_l2328_232834

-- Define the given conditions
def large_planks (n : ℕ) := n = 12
def small_planks (n : ℕ) := n = 10
def nails_for_large_planks (n : ℕ) := n = 15
def nails_for_small_planks (n : ℕ) := n = 5

-- Define the total number of nails needed
def total_nails_needed (n : ℕ) :=
  ∃ (lp sp np_large np_small : ℕ),
  large_planks lp ∧ small_planks sp ∧ nails_for_large_planks np_large ∧ nails_for_small_planks np_small ∧ n = np_large + np_small

-- The theorem statement
theorem john_needs_20_nails : total_nails_needed 20 :=
by { sorry }

end john_needs_20_nails_l2328_232834


namespace johns_score_is_101_l2328_232840

variable (c w s : ℕ)
variable (h1 : s = 40 + 5 * c - w)
variable (h2 : s > 100)
variable (h3 : c ≤ 40)
variable (h4 : ∀ s' > 100, s' < s → ∃ c' w', s' = 40 + 5 * c' - w')

theorem johns_score_is_101 : s = 101 := by
  sorry

end johns_score_is_101_l2328_232840


namespace difference_in_x_coordinates_is_constant_l2328_232822

variable {a x₀ y₀ k : ℝ}

-- Define the conditions
def point_on_x_axis (a : ℝ) : Prop := true

def passes_through_fixed_point_and_tangent (a : ℝ) : Prop :=
  a = 1

def equation_of_curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

def tangent_condition (a x₀ y₀ : ℝ) (k : ℝ) : Prop :=
  a > 2 ∧ y₀ > 0 ∧ y₀^2 = 4 * x₀ ∧ 
  (4 * x₀ - 2 * y₀ * y₀ + y₀^2 = 0)

-- The statement
theorem difference_in_x_coordinates_is_constant (a x₀ y₀ k : ℝ) :
  point_on_x_axis a →
  passes_through_fixed_point_and_tangent a →
  equation_of_curve_C x₀ y₀ →
  tangent_condition a x₀ y₀ k → 
  a - x₀ = 2 :=
by
  intro h1 h2 h3 h4 
  sorry

end difference_in_x_coordinates_is_constant_l2328_232822


namespace necessary_but_not_sufficient_l2328_232825

def is_hyperbola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, a * x^2 + b * y^2 = c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b < 0

theorem necessary_but_not_sufficient (a b c : ℝ) (p : a * b < 0) (q : is_hyperbola a b c) :
  (∀ (a b c : ℝ), is_hyperbola a b c → a * b < 0) ∧ (¬ ∀ (a b c : ℝ), a * b < 0 → is_hyperbola a b c) :=
by
  sorry

end necessary_but_not_sufficient_l2328_232825


namespace solve_system_of_inequalities_l2328_232810

theorem solve_system_of_inequalities (x : ℝ) 
  (h1 : -3 * x^2 + 7 * x + 6 > 0) 
  (h2 : 4 * x - 4 * x^2 > -3) : 
  -1/2 < x ∧ x < 3/2 :=
sorry

end solve_system_of_inequalities_l2328_232810


namespace Pradeep_marks_l2328_232838

variable (T : ℕ) (P : ℕ) (F : ℕ)

def passing_marks := P * T / 100

theorem Pradeep_marks (hT : T = 925) (hP : P = 20) (hF : F = 25) :
  (passing_marks P T) - F = 160 :=
by
  sorry

end Pradeep_marks_l2328_232838


namespace remainder_of_division_l2328_232879

variable (a : ℝ) (b : ℝ)

theorem remainder_of_division : a = 28 → b = 10.02 → ∃ r : ℝ, 0 ≤ r ∧ r < b ∧ ∃ q : ℤ, a = q * b + r ∧ r = 7.96 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end remainder_of_division_l2328_232879


namespace molly_age_l2328_232823

theorem molly_age (S M : ℕ) (h1 : S / M = 4 / 3) (h2 : S + 6 = 34) : M = 21 :=
by
  sorry

end molly_age_l2328_232823


namespace option_B_is_not_polynomial_l2328_232869

-- Define what constitutes a polynomial
def is_polynomial (expr : String) : Prop :=
  match expr with
  | "-26m" => True
  | "3m+5n" => True
  | "0" => True
  | _ => False

-- Given expressions
def expr_A := "-26m"
def expr_B := "m-n=1"
def expr_C := "3m+5n"
def expr_D := "0"

-- The Lean statement confirming option B is not a polynomial
theorem option_B_is_not_polynomial : ¬is_polynomial expr_B :=
by
  -- Since this statement requires a proof, we use 'sorry' as a placeholder.
  sorry

end option_B_is_not_polynomial_l2328_232869


namespace wheel_revolutions_l2328_232833

theorem wheel_revolutions (x y : ℕ) (h1 : y = x + 300)
  (h2 : 10 / (x : ℝ) = 10 / (y : ℝ) + 1 / 60) : 
  x = 300 ∧ y = 600 := 
by sorry

end wheel_revolutions_l2328_232833


namespace domain_of_f_l2328_232871

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : 
  {x : ℝ | ¬ ((x - 3) + (x - 9) = 0)} = 
  {x : ℝ | x ≠ 6} := 
by
  sorry

end domain_of_f_l2328_232871


namespace conference_center_distance_l2328_232816

variables (d t: ℝ)

theorem conference_center_distance
  (h1: ∃ t: ℝ, d = 45 * (t + 1.5))
  (h2: ∃ t: ℝ, d - 45 = 55 * (t - 1.25)):
  d = 478.125 :=
by
  sorry

end conference_center_distance_l2328_232816


namespace sufficient_but_not_necessary_l2328_232861

-- Definitions of conditions
def p (x : ℝ) : Prop := 1 / (x + 1) > 0
def q (x : ℝ) : Prop := (1/x > 0)

-- Main theorem statement
theorem sufficient_but_not_necessary :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
sorry

end sufficient_but_not_necessary_l2328_232861


namespace original_square_perimeter_l2328_232862

theorem original_square_perimeter (p : ℕ) (x : ℕ) 
  (h1: p = 56) 
  (h2: 28 * x = p) : 4 * (2 * (x + 4 * x)) = 40 :=
by
  sorry

end original_square_perimeter_l2328_232862


namespace johns_total_profit_l2328_232829

theorem johns_total_profit
  (cost_price : ℝ) (selling_price : ℝ) (bags_sold : ℕ)
  (h_cost : cost_price = 4) (h_sell : selling_price = 8) (h_bags : bags_sold = 30) :
  (selling_price - cost_price) * bags_sold = 120 := by
    sorry

end johns_total_profit_l2328_232829


namespace find_tangent_point_l2328_232827

noncomputable def exp_neg (x : ℝ) : ℝ := Real.exp (-x)

theorem find_tangent_point :
  ∃ P : ℝ × ℝ, P = (-Real.log 2, 2) ∧ P.snd = exp_neg P.fst ∧ deriv exp_neg P.fst = -2 :=
by
  sorry

end find_tangent_point_l2328_232827


namespace solve_quadratic_l2328_232867

theorem solve_quadratic (x : ℝ) : (x - 2) * (x + 3) = 0 → (x = 2 ∨ x = -3) :=
by
  sorry

end solve_quadratic_l2328_232867


namespace quinton_total_fruit_trees_l2328_232819

-- Define the given conditions
def num_apple_trees := 2
def width_apple_tree_ft := 10
def space_between_apples_ft := 12
def width_peach_tree_ft := 12
def space_between_peaches_ft := 15
def total_space_ft := 71

-- Definition that calculates the total number of fruit trees Quinton wants to plant
def total_fruit_trees : ℕ := 
  let space_apple_trees := num_apple_trees * width_apple_tree_ft + space_between_apples_ft
  let space_remaining_for_peaches := total_space_ft - space_apple_trees
  1 + space_remaining_for_peaches / (width_peach_tree_ft + space_between_peaches_ft) + num_apple_trees

-- The statement to prove
theorem quinton_total_fruit_trees : total_fruit_trees = 4 := by
  sorry

end quinton_total_fruit_trees_l2328_232819


namespace volume_parallelepiped_eq_20_l2328_232895

theorem volume_parallelepiped_eq_20 (k : ℝ) (h : k > 0) (hvol : abs (3 * k^2 - 7 * k - 6) = 20) :
  k = 13 / 3 :=
sorry

end volume_parallelepiped_eq_20_l2328_232895


namespace Margarita_vs_Ricciana_l2328_232872

-- Definitions based on the conditions.
def Ricciana_run : ℕ := 20
def Ricciana_jump : ℕ := 4
def Ricciana_total : ℕ := Ricciana_run + Ricciana_jump

def Margarita_run : ℕ := 18
def Margarita_jump : ℕ := 2 * Ricciana_jump - 1
def Margarita_total : ℕ := Margarita_run + Margarita_jump

-- The statement to be proved.
theorem Margarita_vs_Ricciana : (Margarita_total - Ricciana_total = 1) :=
by
  sorry

end Margarita_vs_Ricciana_l2328_232872


namespace volume_pyramid_problem_l2328_232878

noncomputable def volume_of_pyramid : ℝ :=
  1 / 3 * 10 * 1.5

theorem volume_pyramid_problem :
  ∀ (AB BC CG : ℝ)
  (M : ℝ × ℝ × ℝ),
  AB = 4 →
  BC = 2 →
  CG = 3 →
  M = (2, 5, 1.5) →
  volume_of_pyramid = 5 := 
by
  intros AB BC CG M hAB hBC hCG hM
  sorry

end volume_pyramid_problem_l2328_232878


namespace onions_left_on_shelf_l2328_232820

def initial_onions : ℕ := 98
def sold_onions : ℕ := 65
def remaining_onions : ℕ := initial_onions - sold_onions

theorem onions_left_on_shelf : remaining_onions = 33 :=
by 
  -- Proof would go here
  sorry

end onions_left_on_shelf_l2328_232820


namespace problem_l2328_232896

open Real

theorem problem (x y : ℝ) (h1 : 3 * x + 2 * y = 8) (h2 : 2 * x + 3 * y = 11) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 2041 / 25 :=
sorry

end problem_l2328_232896


namespace min_rice_weight_l2328_232836

theorem min_rice_weight (o r : ℝ) (h1 : o ≥ 4 + 2 * r) (h2 : o ≤ 3 * r) : r ≥ 4 :=
sorry

end min_rice_weight_l2328_232836


namespace negation_statement_l2328_232841

variable {α : Type} (teacher generous : α → Prop)

theorem negation_statement :
  ¬ ∀ x, teacher x → generous x ↔ ∃ x, teacher x ∧ ¬ generous x := by
sorry

end negation_statement_l2328_232841


namespace area_of_triangle_l2328_232882

theorem area_of_triangle :
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  triangle_area = 34 :=
by {
  -- Definitions
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  -- Proof (normally written here, but omitted with 'sorry')
  sorry
}

end area_of_triangle_l2328_232882


namespace inverse_proportion_value_of_m_l2328_232884

theorem inverse_proportion_value_of_m (m : ℤ) (x : ℝ) (y : ℝ) : 
  y = (m - 2) * x ^ (m^2 - 5) → (m = -2) := 
by
  sorry

end inverse_proportion_value_of_m_l2328_232884


namespace find_initial_quarters_l2328_232843

-- Define the initial number of dimes, nickels, and quarters (unknown)
def initial_dimes : ℕ := 2
def initial_nickels : ℕ := 5
def initial_quarters (Q : ℕ) := Q

-- Define the additional coins given by Linda’s mother
def additional_dimes : ℕ := 2
def additional_quarters : ℕ := 10
def additional_nickels : ℕ := 2 * initial_nickels

-- Define the total number of each type of coin after Linda receives the additional coins
def total_dimes : ℕ := initial_dimes + additional_dimes
def total_quarters (Q : ℕ) : ℕ := additional_quarters + initial_quarters Q
def total_nickels : ℕ := initial_nickels + additional_nickels

-- Define the total number of coins
def total_coins (Q : ℕ) : ℕ := total_dimes + total_quarters Q + total_nickels

theorem find_initial_quarters : ∃ Q : ℕ, total_coins Q = 35 ∧ Q = 6 := by
  -- Provide the corresponding proof here
  sorry

end find_initial_quarters_l2328_232843


namespace range_of_m_l2328_232809

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, f (3 + 2 * Real.sin θ) < m) → m > 4 :=
sorry

end range_of_m_l2328_232809


namespace number_of_Slurpees_l2328_232806

theorem number_of_Slurpees
  (total_money : ℕ)
  (cost_per_Slurpee : ℕ)
  (change : ℕ)
  (spent_money := total_money - change)
  (number_of_Slurpees := spent_money / cost_per_Slurpee)
  (h1 : total_money = 20)
  (h2 : cost_per_Slurpee = 2)
  (h3 : change = 8) :
  number_of_Slurpees = 6 := by
  sorry

end number_of_Slurpees_l2328_232806


namespace valid_integers_count_l2328_232850

def count_valid_integers : ℕ :=
  let digits : List ℕ := [0, 1, 2, 3, 4, 6, 7, 8, 9]
  let first_digit_count := 7  -- from 2 to 9 excluding 5
  let second_digit_count := 8
  let third_digit_count := 7
  let fourth_digit_count := 6
  first_digit_count * second_digit_count * third_digit_count * fourth_digit_count

theorem valid_integers_count : count_valid_integers = 2352 := by
  -- intermediate step might include nice counting macros
  sorry

end valid_integers_count_l2328_232850


namespace find_a2_plus_a8_l2328_232855

variable {a_n : ℕ → ℤ}  -- Assume the sequence is indexed by natural numbers and maps to integers

-- Define the condition in the problem
def seq_property (a_n : ℕ → ℤ) := a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 25

-- Statement to prove
theorem find_a2_plus_a8 (h : seq_property a_n) : a_n 2 + a_n 8 = 10 :=
sorry

end find_a2_plus_a8_l2328_232855


namespace inequality_solution_l2328_232860

theorem inequality_solution (x : ℝ) (h₁ : 1 - x < 0) (h₂ : x - 3 ≤ 0) : 1 < x ∧ x ≤ 3 :=
by
  sorry

end inequality_solution_l2328_232860


namespace number_at_100th_row_1000th_column_l2328_232868

axiom cell_numbering_rule (i j : ℕ) : ℕ

/-- 
  The cell located at the intersection of the 100th row and the 1000th column
  on an infinitely large chessboard, sequentially numbered with specific rules,
  will receive the number 900.
-/
theorem number_at_100th_row_1000th_column : cell_numbering_rule 100 1000 = 900 :=
sorry

end number_at_100th_row_1000th_column_l2328_232868


namespace determine_m_l2328_232846

theorem determine_m (m : ℝ) : (∀ x : ℝ, (0 < x ∧ x < 2) ↔ -1/2 * x^2 + 2 * x + m * x > 0) → m = -1 :=
by
  intro h
  sorry

end determine_m_l2328_232846


namespace ratio_of_powers_l2328_232891

theorem ratio_of_powers (a x : ℝ) (h : a^(2 * x) = Real.sqrt 2 - 1) : (a^(3 * x) + a^(-3 * x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end ratio_of_powers_l2328_232891


namespace price_of_paint_models_max_boxes_of_paint_A_l2328_232888

-- Define the conditions as hypotheses
def price_paint_model_A_B (x y : ℕ) : Prop :=
  x + 2 * y = 56 ∧ 2 * x + y = 64

def total_cost_constraint (x y m : ℕ) : Prop :=
  24 * m + 16 * (200 - m) ≤ 3920

-- Prove the prices of the paint models given the conditions
theorem price_of_paint_models :
  ∃ (x y : ℕ), price_paint_model_A_B x y ∧ x = 24 ∧ y = 16 :=
by
  sorry

-- Prove the maximum number of boxes of paint model A the school can purchase given the total cost constraint
theorem max_boxes_of_paint_A :
  ∃ (m : ℕ), total_cost_constraint 24 16 m ∧ m = 90 :=
by
  sorry

end price_of_paint_models_max_boxes_of_paint_A_l2328_232888


namespace triangle_max_perimeter_l2328_232804

noncomputable def max_perimeter_triangle_ABC (a b c : ℝ) (A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) : ℝ := 
  a + b + c

theorem triangle_max_perimeter (a b c A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) :
  max_perimeter_triangle_ABC a b c A B C h1 h2 ≤ 6 * Real.sqrt 3 :=
sorry

end triangle_max_perimeter_l2328_232804


namespace probability_of_boys_and_girls_l2328_232837

def total_outcomes := Nat.choose 7 4
def only_boys_outcomes := Nat.choose 4 4
def both_boys_and_girls_outcomes := total_outcomes - only_boys_outcomes
def probability := both_boys_and_girls_outcomes / total_outcomes

theorem probability_of_boys_and_girls :
  probability = 34 / 35 :=
by
  sorry

end probability_of_boys_and_girls_l2328_232837


namespace ratio_january_february_l2328_232880

variable (F : ℕ)

def total_savings := 19 + F + 8 

theorem ratio_january_february (h : total_savings F = 46) : 19 / F = 1 := by
  sorry

end ratio_january_february_l2328_232880


namespace appropriate_grouping_43_neg78_27_neg52_l2328_232808

theorem appropriate_grouping_43_neg78_27_neg52 :
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  (a + c) + (b + d) = -60 :=
by
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  sorry

end appropriate_grouping_43_neg78_27_neg52_l2328_232808


namespace even_odd_product_l2328_232897

theorem even_odd_product (n : ℕ) (i : Fin n → Fin n) (h_perm : ∀ j : Fin n, ∃ k : Fin n, i k = j) :
  (∃ l, l % 2 = 0) → 
  ∀ (k : Fin n), ¬(i k = k) → 
  (n % 2 = 0 → (∃ m : ℤ, m + 1 % 2 = 1) ∨ (∃ m : ℤ, m + 1 % 2 = 0)) ∧ 
  (n % 2 = 1 → (∃ m : ℤ, m + 1 % 2 = 0)) :=
by
  sorry

end even_odd_product_l2328_232897


namespace complete_the_square_l2328_232859

-- Definition of the initial condition
def eq1 : Prop := ∀ x : ℝ, x^2 + 4 * x + 1 = 0

-- The goal is to prove if the initial condition holds, then the desired result holds.
theorem complete_the_square (x : ℝ) (h : x^2 + 4 * x + 1 = 0) : (x + 2)^2 = 3 := by
  sorry

end complete_the_square_l2328_232859


namespace find_a_l2328_232830

theorem find_a (a : ℝ) : 
  (∃ l : ℝ, l = 2 * Real.sqrt 3 ∧ 
  ∃ y, y ≤ 6 ∧ 
  (∀ x, x^2 + y^2 = a^2 ∧ 
  x^2 + y^2 + a * y - 6 = 0)) → 
  a = 2 ∨ a = -2 :=
by sorry

end find_a_l2328_232830


namespace max_value_of_expression_l2328_232864

open Classical
open Real

theorem max_value_of_expression (a b : ℝ) (c : ℝ) (h1 : a^2 + b^2 = c^2 + ab) (h2 : c = 1) :
  ∃ x : ℝ, x = (1 / 2) * b + a ∧ x = (sqrt 21) / 3 := 
sorry

end max_value_of_expression_l2328_232864


namespace fraction_equals_decimal_l2328_232877

theorem fraction_equals_decimal : (1 / 4 : ℝ) = 0.25 := 
sorry

end fraction_equals_decimal_l2328_232877


namespace value_of_b_l2328_232848

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, (-x^2 + b * x - 7 < 0) ↔ (x < 2 ∨ x > 6)) → b = 8 :=
by
  sorry

end value_of_b_l2328_232848


namespace find_98_real_coins_l2328_232831

-- We will define the conditions as variables and state the goal as a theorem.

-- Variables:
variable (Coin : Type) -- Type representing coins
variable [Fintype Coin] -- 100 coins in total, therefore a Finite type
variable (number_of_coins : ℕ) (h100 : number_of_coins = 100)
variable (real : Coin → Prop) -- Predicate indicating if the coin is real
variable (lighter_fake : Coin → Prop) -- Predicate indicating if the coin is the lighter fake
variable (balance_scale : Coin → Coin → Prop) -- Balance scale result

-- Conditions:
axiom real_coins_count : ∃ R : Finset Coin, R.card = 99 ∧ (∀ c ∈ R, real c)
axiom fake_coin_exists : ∃ F : Coin, lighter_fake F ∧ ¬ real F

theorem find_98_real_coins : ∃ S : Finset Coin, S.card = 98 ∧ (∀ c ∈ S, real c) := by
  sorry

end find_98_real_coins_l2328_232831


namespace haley_number_of_shirts_l2328_232807

-- Define the given information
def washing_machine_capacity : ℕ := 7
def total_loads : ℕ := 5
def number_of_sweaters : ℕ := 33
def number_of_shirts := total_loads * washing_machine_capacity - number_of_sweaters

-- The statement that needs to be proven
theorem haley_number_of_shirts : number_of_shirts = 2 := by
  sorry

end haley_number_of_shirts_l2328_232807


namespace probability_playing_one_instrument_l2328_232811

noncomputable def total_people : ℕ := 800
noncomputable def fraction_playing_instruments : ℚ := 1 / 5
noncomputable def number_playing_two_or_more : ℕ := 32

theorem probability_playing_one_instrument :
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  (number_playing_exactly_one / total_people) = 1 / 6.25 :=
by 
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  have key : (number_playing_exactly_one / total_people) = 1 / 6.25 := sorry
  exact key

end probability_playing_one_instrument_l2328_232811


namespace highest_probability_of_red_ball_l2328_232805

theorem highest_probability_of_red_ball (red yellow white blue : ℕ) (H1 : red = 5) (H2 : yellow = 4) (H3 : white = 1) (H4 : blue = 3) :
  (red : ℚ) / (red + yellow + white + blue) > (yellow : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (white : ℚ) / (red + yellow + white + blue) ∧
  (red : ℚ) / (red + yellow + white + blue) > (blue : ℚ) / (red + yellow + white + blue) := 
by {
  sorry
}

end highest_probability_of_red_ball_l2328_232805


namespace find_k_l2328_232893

theorem find_k (x y k : ℤ) (h₁ : x = -3) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) : k = 6 :=
by
  rw [h₁, h₂] at h₃
  -- Substitute x and y in the equation
  -- 2 * (-3) + k * 2 = 6
  sorry

end find_k_l2328_232893


namespace find_degree_of_alpha_l2328_232853

theorem find_degree_of_alpha
  (x : ℝ)
  (alpha : ℝ := x + 40)
  (beta : ℝ := 3 * x - 40)
  (h_parallel : alpha + beta = 180) :
  alpha = 85 :=
by
  sorry

end find_degree_of_alpha_l2328_232853


namespace pairs_with_green_shirts_l2328_232801

theorem pairs_with_green_shirts (r g t p rr_pairs gg_pairs : ℕ)
  (h1 : r = 60)
  (h2 : g = 90)
  (h3 : t = 150)
  (h4 : p = 75)
  (h5 : rr_pairs = 28)
  : gg_pairs = 43 := 
sorry

end pairs_with_green_shirts_l2328_232801


namespace problem_statement_l2328_232826

noncomputable def g : ℝ → ℝ
| x => if x < 0 then -x
            else if x < 5 then x + 3
            else 2 * x ^ 2

theorem problem_statement : g (-6) + g 3 + g 8 = 140 :=
by
  -- Proof goes here
  sorry

end problem_statement_l2328_232826


namespace sum_of_xs_l2328_232839

theorem sum_of_xs (x y z : ℂ) : (x + y * z = 8) ∧ (y + x * z = 12) ∧ (z + x * y = 11) → 
    ∃ S, ∀ (xi yi zi : ℂ), (xi + yi * zi = 8) ∧ (yi + xi * zi = 12) ∧ (zi + xi * yi = 11) →
        xi + yi + zi = S :=
by
  sorry

end sum_of_xs_l2328_232839


namespace storage_space_remaining_l2328_232866

def total_space_remaining (first_floor second_floor: ℕ) (boxes: ℕ) : ℕ :=
  first_floor + second_floor - boxes

theorem storage_space_remaining :
  ∀ (first_floor second_floor boxes: ℕ),
  (first_floor = 2 * second_floor) →
  (boxes = 5000) →
  (boxes = second_floor / 4) →
  total_space_remaining first_floor second_floor boxes = 55000 :=
by
  intros first_floor second_floor boxes h1 h2 h3
  sorry

end storage_space_remaining_l2328_232866


namespace transformation_correct_l2328_232898

theorem transformation_correct (a b c : ℝ) : a = b → ac = bc :=
by sorry

end transformation_correct_l2328_232898


namespace red_black_probability_l2328_232828

-- Define the number of cards and ranks
def num_cards : ℕ := 64
def num_ranks : ℕ := 16

-- Define the suits and their properties
def suits := 6
def red_suits := 3
def black_suits := 3
def cards_per_suit := num_ranks

-- Define the number of red and black cards
def red_cards := red_suits * cards_per_suit
def black_cards := black_suits * cards_per_suit

-- Prove the probability that the top card is red and the second card is black
theorem red_black_probability : 
  (red_cards * black_cards) / (num_cards * (num_cards - 1)) = 3 / 4 := by 
  sorry

end red_black_probability_l2328_232828


namespace temperature_notation_l2328_232899

-- Define what it means to denote temperatures in degrees Celsius
def denote_temperature (t : ℤ) : String :=
  if t < 0 then "-" ++ toString t ++ "°C"
  else if t > 0 then "+" ++ toString t ++ "°C"
  else toString t ++ "°C"

-- Theorem statement
theorem temperature_notation (t : ℤ) (ht : t = 2) : denote_temperature t = "+2°C" :=
by
  -- Proof goes here
  sorry

end temperature_notation_l2328_232899


namespace truck_left_1_hour_later_l2328_232835

theorem truck_left_1_hour_later (v_car v_truck : ℝ) (time_to_pass : ℝ) : 
  v_car = 55 ∧ v_truck = 65 ∧ time_to_pass = 6.5 → 
  1 = time_to_pass - (time_to_pass * (v_car / v_truck)) := 
by
  intros h
  sorry

end truck_left_1_hour_later_l2328_232835


namespace average_of_remaining_two_l2328_232832

-- Given conditions
def average_of_six (S : ℝ) := S / 6 = 3.95
def average_of_first_two (S1 : ℝ) := S1 / 2 = 4.2
def average_of_next_two (S2 : ℝ) := S2 / 2 = 3.85

-- Prove that the average of the remaining 2 numbers equals 3.8
theorem average_of_remaining_two (S S1 S2 Sr : ℝ) (h1 : average_of_six S) (h2 : average_of_first_two S1) (h3: average_of_next_two S2) (h4 : Sr = S - S1 - S2) :
  Sr / 2 = 3.8 :=
by
  -- We can use the assumptions h1, h2, h3, and h4 to reach the conclusion
  sorry

end average_of_remaining_two_l2328_232832


namespace binomials_product_l2328_232876

noncomputable def poly1 (x y : ℝ) : ℝ := 2 * x^2 + 3 * y - 4
noncomputable def poly2 (y : ℝ) : ℝ := y + 6

theorem binomials_product (x y : ℝ) :
  (poly1 x y) * (poly2 y) = 2 * x^2 * y + 12 * x^2 + 3 * y^2 + 14 * y - 24 :=
by sorry

end binomials_product_l2328_232876


namespace eval_expression_l2328_232803

theorem eval_expression :
    (727 * 727) - (726 * 728) = 1 := by
  sorry

end eval_expression_l2328_232803


namespace largest_integer_l2328_232851

theorem largest_integer (n : ℕ) : n ^ 200 < 5 ^ 300 → n <= 11 :=
by
  sorry

end largest_integer_l2328_232851


namespace decimal_expansion_2023rd_digit_l2328_232883

theorem decimal_expansion_2023rd_digit 
  (x : ℚ) 
  (hx : x = 7 / 26) 
  (decimal_expansion : ℕ → ℕ)
  (hdecimal : ∀ n : ℕ, decimal_expansion n = if n % 12 = 0 
                        then 2 
                        else if n % 12 = 1 
                          then 7 
                          else if n % 12 = 2 
                            then 9 
                            else if n % 12 = 3 
                              then 2 
                              else if n % 12 = 4 
                                then 3 
                                else if n % 12 = 5 
                                  then 0 
                                  else if n % 12 = 6 
                                    then 7 
                                    else if n % 12 = 7 
                                      then 6 
                                      else if n % 12 = 8 
                                        then 9 
                                        else if n % 12 = 9 
                                          then 2 
                                          else if n % 12 = 10 
                                            then 3 
                                            else 0) :
  decimal_expansion 2023 = 0 :=
sorry

end decimal_expansion_2023rd_digit_l2328_232883


namespace find_a_l2328_232863

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a (a : ℝ) : g a = 0 → a = 7 / 5 := by
  intro h
  sorry

end find_a_l2328_232863


namespace incorrect_divisor_l2328_232881

theorem incorrect_divisor (D x : ℕ) (h1 : D = 24 * x) (h2 : D = 48 * 36) : x = 72 := by
  sorry

end incorrect_divisor_l2328_232881


namespace units_digit_of_product_l2328_232844

theorem units_digit_of_product : 
  (4 * 6 * 9) % 10 = 6 := 
by
  sorry

end units_digit_of_product_l2328_232844


namespace number_of_shirts_is_20_l2328_232885

/-- Given the conditions:
1. The total price for some shirts is 360,
2. The total price for 45 sweaters is 900,
3. The average price of a sweater exceeds that of a shirt by 2,
prove that the number of shirts is 20. -/

theorem number_of_shirts_is_20
  (S : ℕ) (P_shirt P_sweater : ℝ)
  (h1 : S * P_shirt = 360)
  (h2 : 45 * P_sweater = 900)
  (h3 : P_sweater = P_shirt + 2) :
  S = 20 :=
by
  sorry

end number_of_shirts_is_20_l2328_232885


namespace simplify_and_evaluate_l2328_232847

variable (x y : ℤ)

theorem simplify_and_evaluate (h1 : x = 1) (h2 : y = 1) :
    2 * (x - 2 * y) ^ 2 - (2 * y + x) * (-2 * y + x) = 5 := by
    sorry

end simplify_and_evaluate_l2328_232847


namespace three_dice_prime_probability_l2328_232818

noncomputable def rolling_three_dice_prime_probability : ℚ :=
  sorry

theorem three_dice_prime_probability : rolling_three_dice_prime_probability = 1 / 24 :=
  sorry

end three_dice_prime_probability_l2328_232818


namespace hares_cuts_l2328_232865

-- Definitions representing the given conditions
def intermediates_fallen := 10
def end_pieces_fixed := 2
def total_logs := intermediates_fallen + end_pieces_fixed

-- Theorem statement
theorem hares_cuts : total_logs - 1 = 11 := by 
  sorry

end hares_cuts_l2328_232865


namespace three_more_than_seven_in_pages_l2328_232824

theorem three_more_than_seven_in_pages : 
  ∀ (pages : List Nat), (∀ n, n ∈ pages → 1 ≤ n ∧ n ≤ 530) ∧ (List.length pages = 530) →
  ((List.count 3 (pages.bind (λ n => Nat.digits 10 n))) - (List.count 7 (pages.bind (λ n => Nat.digits 10 n)))) = 100 :=
by
  intros pages h
  sorry

end three_more_than_seven_in_pages_l2328_232824


namespace minimum_odd_correct_answers_l2328_232845

theorem minimum_odd_correct_answers (students : Fin 50 → Fin 5) :
  (∀ S : Finset (Fin 50), S.card = 40 → 
    (∃ x ∈ S, students x = 3) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, students x₁ = 2 ∧ x₁ ≠ x₂ ∧ students x₂ = 2) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, students x₁ = 1 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ students x₂ = 1 ∧ students x₃ = 1) ∧ 
    (∃ x₁ ∈ S, ∃ x₂ ∈ S, ∃ x₃ ∈ S, ∃ x₄ ∈ S, students x₁ = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₄ ∧ students x₂ = 0 ∧ students x₃ = 0 ∧ students x₄ = 0)) →
  (∃ S : Finset (Fin 50), (∀ x ∈ S, (students x = 1 ∨ students x = 3)) ∧ S.card = 23) :=
by
  sorry

end minimum_odd_correct_answers_l2328_232845


namespace A_more_than_B_l2328_232812

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := A = (1/3) * (B + C)
def condition2 : Prop := B = (2/7) * (A + C)
def condition3 : Prop := A + B + C = 1080

-- Conclusion
theorem A_more_than_B (A B C : ℝ) (h1 : condition1 A B C) (h2 : condition2 A B C) (h3 : condition3 A B C) :
  A - B = 30 :=
sorry

end A_more_than_B_l2328_232812


namespace probability_of_three_different_colors_draw_l2328_232892

open ProbabilityTheory

def number_of_blue_chips : ℕ := 4
def number_of_green_chips : ℕ := 5
def number_of_red_chips : ℕ := 6
def number_of_yellow_chips : ℕ := 3
def total_number_of_chips : ℕ := 18

def P_B : ℚ := number_of_blue_chips / total_number_of_chips
def P_G : ℚ := number_of_green_chips / total_number_of_chips
def P_R : ℚ := number_of_red_chips / total_number_of_chips
def P_Y : ℚ := number_of_yellow_chips / total_number_of_chips

def P_different_colors : ℚ := 2 * ((P_B * P_G + P_B * P_Y + P_G * P_R + P_G * P_Y + P_R * P_B + P_R * P_G + P_Y * P_B + P_Y * P_G + P_R * P_Y) +
                                    (P_B * P_R + P_B * P_Y + P_G * P_R + P_G * P_Y + P_R * P_B + P_R * P_G + P_Y * P_B + P_Y * P_G))

theorem probability_of_three_different_colors_draw :
  P_different_colors = 141 / 162 :=
by
  -- Placeholder for the actual proof.
  sorry

end probability_of_three_different_colors_draw_l2328_232892


namespace james_training_hours_in_a_year_l2328_232870

-- Definitions based on conditions
def trains_twice_a_day : ℕ := 2
def hours_per_training : ℕ := 4
def days_trains_per_week : ℕ := 7 - 2
def weeks_per_year : ℕ := 52

-- Resultant computation
def daily_training_hours : ℕ := trains_twice_a_day * hours_per_training
def weekly_training_hours : ℕ := daily_training_hours * days_trains_per_week
def yearly_training_hours : ℕ := weekly_training_hours * weeks_per_year

-- Statement to prove
theorem james_training_hours_in_a_year : yearly_training_hours = 2080 := by
  -- proof goes here
  sorry

end james_training_hours_in_a_year_l2328_232870


namespace max_bag_weight_is_50_l2328_232894

noncomputable def max_weight_allowed (people bags_per_person more_bags_allowed total_weight : ℕ) : ℝ := 
  total_weight / ((people * bags_per_person) + more_bags_allowed)

theorem max_bag_weight_is_50 : ∀ (people bags_per_person more_bags_allowed total_weight : ℕ), 
  people = 6 → 
  bags_per_person = 5 → 
  more_bags_allowed = 90 → 
  total_weight = 6000 →
  max_weight_allowed people bags_per_person more_bags_allowed total_weight = 50 := 
by 
  sorry

end max_bag_weight_is_50_l2328_232894


namespace product_of_fractions_l2328_232875

theorem product_of_fractions : (2 : ℚ) / 9 * (4 : ℚ) / 5 = 8 / 45 :=
by 
  sorry

end product_of_fractions_l2328_232875


namespace smallest_y2_l2328_232849

theorem smallest_y2 :
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  y2 < y1 ∧ y2 < y3 ∧ y2 < y4 :=
by
  let y1 := 1 / (-2)
  let y2 := 1 / (-1)
  let y3 := 1 / (1)
  let y4 := 1 / (2)
  show y2 < y1 ∧ y2 < y3 ∧ y2 < y4
  sorry

end smallest_y2_l2328_232849


namespace power_of_seven_l2328_232821

theorem power_of_seven : 
  (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = (7 ^ (3 / 28)) :=
by
  sorry

end power_of_seven_l2328_232821


namespace find_hourly_wage_l2328_232887

noncomputable def hourly_wage_inexperienced (x : ℝ) : Prop :=
  let sailors_total := 17
  let inexperienced_sailors := 5
  let experienced_sailors := sailors_total - inexperienced_sailors
  let wage_experienced := (6 / 5) * x
  let total_hours_month := 240
  let total_monthly_earnings_experienced := 34560
  (experienced_sailors * wage_experienced * total_hours_month) = total_monthly_earnings_experienced

theorem find_hourly_wage (x : ℝ) : hourly_wage_inexperienced x → x = 10 :=
by
  sorry

end find_hourly_wage_l2328_232887


namespace mike_baseball_cards_l2328_232817

theorem mike_baseball_cards (initial_cards birthday_cards traded_cards : ℕ)
  (h1 : initial_cards = 64) 
  (h2 : birthday_cards = 18) 
  (h3 : traded_cards = 20) :
  initial_cards + birthday_cards - traded_cards = 62 :=
by 
  -- assumption:
  sorry

end mike_baseball_cards_l2328_232817


namespace find_2a_plus_b_l2328_232886

open Real

variables {a b : ℝ}

-- Conditions
def angles_in_first_quadrant (a b : ℝ) : Prop := 
  0 < a ∧ a < π / 2 ∧ 0 < b ∧ b < π / 2

def cos_condition (a b : ℝ) : Prop :=
  5 * cos a ^ 2 + 3 * cos b ^ 2 = 2

def sin_condition (a b : ℝ) : Prop :=
  5 * sin (2 * a) + 3 * sin (2 * b) = 0

-- Problem statement
theorem find_2a_plus_b (a b : ℝ) 
  (h1 : angles_in_first_quadrant a b)
  (h2 : cos_condition a b)
  (h3 : sin_condition a b) :
  2 * a + b = π / 2 := 
sorry

end find_2a_plus_b_l2328_232886


namespace binary_add_mul_l2328_232802

def x : ℕ := 0b101010
def y : ℕ := 0b11010
def z : ℕ := 0b1110
def result : ℕ := 0b11000000000

theorem binary_add_mul : ((x + y) * z) = result := by
  sorry

end binary_add_mul_l2328_232802
