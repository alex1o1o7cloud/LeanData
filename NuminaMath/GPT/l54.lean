import Data.Rat.Basic
import Mathlib
import Mathlib.Algebra.ArithMoyers
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Factorial.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Floor
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Basic
import Mathlib.Topology.EuclideanSpace.Basic

namespace annual_return_percentage_l54_54901

theorem annual_return_percentage (initial_value final_value gain : ℕ)
    (h1 : initial_value = 8000)
    (h2 : final_value = initial_value + 400)
    (h3 : gain = final_value - initial_value) :
    (gain * 100 / initial_value) = 5 := by
  sorry

end annual_return_percentage_l54_54901


namespace probability_same_color_l54_54461

theorem probability_same_color :
  let total_balls := 18
  let green_balls := 10
  let white_balls := 8
  let prob_two_green := (green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1))
  let prob_two_white := (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))
  let prob_same_color := prob_two_green + prob_two_white
  in prob_same_color = 73 / 153 :=
by
  let total_balls := 18
  let green_balls := 10
  let white_balls := 8
  let prob_two_green := (green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1))
  let prob_two_white := (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))
  let prob_same_color := prob_two_green + prob_two_white
  have : prob_two_green = 90 / 306 := by sorry
  have : prob_two_white = 56 / 306 := by sorry
  have : prob_same_color = 146 / 306 := by sorry
  have : 146 / 306 = 73 / 153 := by sorry
  show prob_same_color = 73 / 153 from sorry

end probability_same_color_l54_54461


namespace quadrilateral_area_ratio_l54_54714

theorem quadrilateral_area_ratio (A B C D P : Point)
  (h : vector_trans A P + 3 * vector_trans B P + 2 * vector_trans C P + 4 * vector_trans D P = 0) :
  (area_quadrilateral A B C D) / (area_triangle A P D) = 10 :=
sorry

end quadrilateral_area_ratio_l54_54714


namespace fourth_root_squared_cubed_l54_54073

theorem fourth_root_squared_cubed (x : ℝ) (h : (x^(1/4))^2^3 = 1296) : x = 256 :=
sorry

end fourth_root_squared_cubed_l54_54073


namespace trivia_team_score_l54_54879

-- Definitions based on the conditions
def total_members : ℕ := 15
def absent_members : ℕ := 6
def points_per_member : ℕ := 3

-- Calculating number of present members
def present_members : ℕ := total_members - absent_members

-- Proving the total points scored
theorem trivia_team_score : present_members * points_per_member = 27 :=
by {
  -- Here we explicitly compute the multiplication and confirm it equals 27
  change (15 - 6) * 3 = 27,
  rw Nat.sub_def,
  rw Nat.sub_def,
  -- You can also manually verify instead of relying on rw for a more elaborate proof
  exact rfl
}

end trivia_team_score_l54_54879


namespace sally_cookies_baked_l54_54550

def art_cookie_area (b1 b2 h : ℝ) : ℝ :=
  1 / 2 * (b1 + b2) * h

def sally_cookie_area (r : ℝ) : ℝ :=
  π * r^2

theorem sally_cookies_baked 
  (b1 b2 h : ℝ) (art_cookies_no : ℕ) (diameter : ℝ)
  (art_cookie_area = 10) (total_art_area = 200)
  (sally_cookie_area = 4 * π)
  : art_cookie_area b1 b2 h * art_cookies_no = total_art_area →
    sally_cookie_area (diameter / 2) * 16 = total_art_area :=
sorry

end sally_cookies_baked_l54_54550


namespace vasya_grades_l54_54670

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧
    (grades.filter (λ x, x = 5)).length > 2 ∧
    List.sorted (≤) grades ∧
    grades.nth 2 = some 4 ∧
    (grades.sum : ℚ) / 5 = 3.8 ∧
    grades = [2, 3, 4, 5, 5] := by
  sorry

end vasya_grades_l54_54670


namespace odd_function_product_nonpositive_l54_54199

noncomputable def is_odd_function (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_product_nonpositive (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) : 
  ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by 
  sorry

end odd_function_product_nonpositive_l54_54199


namespace find_f_of_3_l54_54254

theorem find_f_of_3 :
  (∀ u, f u = u^2 + 2) → f 3 = 11 :=
by
  intro h
  exact h 3

end find_f_of_3_l54_54254


namespace rational_roots_of_polynomial_l54_54344

theorem rational_roots_of_polynomial (a b : ℚ) : 
    ∀ (P : ℚ[X]), P = (X - a)^3 * (X - b)^2 → a ≠ b → (∃ a b : ℚ, true) :=
begin
  sorry
end

end rational_roots_of_polynomial_l54_54344


namespace age_difference_two_children_l54_54092

/-!
# Age difference between two children in a family

## Given:
- 10 years ago, the average age of a family of 4 members was 24 years.
- Two children have been born since then.
- The present average age of the family (now 6 members) is the same, 24 years.
- The present age of the youngest child (Y1) is 3 years.

## Prove:
The age difference between the two children is 2 years.
-/

theorem age_difference_two_children :
  let Y1 := 3
  let Y2 := 5
  let total_age_10_years_ago := 4 * 24
  let total_age_now := 6 * 24
  let increase_age_10_years := total_age_now - total_age_10_years_ago
  let increase_due_to_original_members := 4 * 10
  let increase_due_to_children := increase_age_10_years - increase_due_to_original_members
  Y1 + Y2 = increase_due_to_children
  → Y2 - Y1 = 2 :=
by
  intros
  sorry

end age_difference_two_children_l54_54092


namespace parabola_inequality_l54_54604

theorem parabola_inequality (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 2 * k * x + (k^2 + 2 * k + 2) > x^2 + 2 * k * x - 2 * k^2 - 1) ↔ (-1 < k ∧ k < 3) := 
sorry

end parabola_inequality_l54_54604


namespace min_groups_needed_l54_54853

theorem min_groups_needed (total_members : ℕ) (max_group_size : ℕ) : total_members = 30 → max_group_size = 12 → (∀ group_size, group_size ∣ total_members → group_size ≤ max_group_size → (total_members / group_size) ≥ 3) → (total_members / max_group_size) = 3 :=
by
  intros h_total h_max h_condition
  have group_size := 10
  have group_size_property : group_size ∣ total_members := sorry
  have group_size_le_max : group_size ≤ max_group_size := sorry
  specialize h_condition group_size group_size_property group_size_le_max
  exact h_condition

end min_groups_needed_l54_54853


namespace sufficient_fabric_for_dresses_l54_54862

-- Definitions based on the conditions
def fabric_length : ℕ := 140
def fabric_width : ℕ := 75
def dress_length_required : ℕ := 45
def dress_width_required : ℕ := 26
def num_dresses : ℕ := 8

-- Statement of the proof problem
theorem sufficient_fabric_for_dresses :
  (fabric_length // dress_length_required * (fabric_width // dress_width_required)) ≥ num_dresses :=
by sorry

end sufficient_fabric_for_dresses_l54_54862


namespace minimalBananasTotal_is_408_l54_54802

noncomputable def minimalBananasTotal : ℕ :=
  let b₁ := 11 * 8
  let b₂ := 13 * 8
  let b₃ := 27 * 8
  b₁ + b₂ + b₃

theorem minimalBananasTotal_is_408 : minimalBananasTotal = 408 := by
  sorry

end minimalBananasTotal_is_408_l54_54802


namespace probability_of_one_after_extensions_l54_54426

theorem probability_of_one_after_extensions :
  let initial_sequence := (List.range 9).map (λ x, x + 1)
  let final_sequence (n : Nat) := (List.range n).flatMap (λ k, List.range (k + 1)).map (λ x, x + 1)
  let extended_sequence := final_sequence 2017
  let count_ones : ℕ := extended_sequence.count (λ x, x = 1)
  let total_elements : ℕ := extended_sequence.length
  in (count_ones / total_elements : ℚ) = (2018 / 2026 : ℚ) := sorry

end probability_of_one_after_extensions_l54_54426


namespace range_of_a_l54_54555

noncomputable def inequality_condition (x : ℝ) (a : ℝ) : Prop :=
  a - 2 * x - |Real.log x| ≤ 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → inequality_condition x a) ↔ a ≤ 1 + Real.log 2 :=
sorry

end range_of_a_l54_54555


namespace solve_inequality_l54_54767

theorem solve_inequality : 
  {x : ℝ | -4 * x^2 + 7 * x + 2 < 0} = {x : ℝ | x < -1/4} ∪ {x : ℝ | 2 < x} :=
by
  sorry

end solve_inequality_l54_54767


namespace sum_of_squares_of_medians_l54_54821

theorem sum_of_squares_of_medians (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) :
  let m_a := real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)
  let m_b := real.sqrt ((2 * c^2 + 2 * a^2 - b^2) / 4)
  let m_c := real.sqrt ((2 * a^2 + 2 * b^2 - c^2) / 4)
  (m_a^2 + m_b^2 + m_c^2) = 442.5 :=
by
  sorry

end sum_of_squares_of_medians_l54_54821


namespace uncovered_side_length_l54_54866

theorem uncovered_side_length
  (A : ℝ) (F : ℝ)
  (h1 : A = 600)
  (h2 : F = 130) :
  ∃ L : ℝ, L = 120 :=
by {
  sorry
}

end uncovered_side_length_l54_54866


namespace evaluate_81_power_5_div_4_l54_54184

-- Define the conditions
def base_factorized : ℕ := 3 ^ 4
def power_rule (b : ℕ) (m n : ℝ) : ℝ := (b : ℝ) ^ m ^ n

-- Define the primary calculation
noncomputable def power_calculation : ℝ := 81 ^ (5 / 4)

-- Prove that the calculation equals 243
theorem evaluate_81_power_5_div_4 : power_calculation = 243 := 
by
  have h1 : base_factorized = 81 := by sorry
  have h2 : power_rule 3 4 (5 / 4) = 3 ^ 5 := by sorry
  have h3 : 3 ^ 5 = 243 := by sorry
  have h4 : power_calculation = power_rule 3 4 (5 / 4) := by sorry
  rw [h1, h2, h3, h4]
  exact h3

end evaluate_81_power_5_div_4_l54_54184


namespace proof_problem_l54_54389

variable (f : ℝ → ℝ)
variable (c : ℝ)

-- Given conditions
axiom H1 : ∀ (a b : ℝ), c * b^2 * f(a) = a^2 * f(b)  
axiom H2 : c = 2
axiom H3 : f 6 ≠ 0

-- The statement to prove
theorem proof_problem : (f(7) - f(3)) / f(6) = 5 / 9 :=
by
  sorry

end proof_problem_l54_54389


namespace find_x_value_l54_54950

theorem find_x_value :
  let a := (2021 : ℝ)
  let b := (2022 : ℝ)
  ∀ x : ℝ, (a / b - b / a + x = 0) → (x = b / a - a / b) :=
  by
    intros a b x h
    sorry

end find_x_value_l54_54950


namespace sum_twodigit_odd_multiples_of_9_l54_54833

-- Define the conditions.
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- The sequence of two-digit odd multiples of 9.
def twodigit_odd_multiples_of_9 : list ℕ := [27, 45, 63, 81, 99]

-- The main theorem.
theorem sum_twodigit_odd_multiples_of_9 : 
  ∑ n in twodigit_odd_multiples_of_9, n = 315 := by
  sorry

end sum_twodigit_odd_multiples_of_9_l54_54833


namespace cauchy_schwarz_inequality_l54_54015

theorem cauchy_schwarz_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) :
  (Finset.univ.sum x) * (Finset.univ.sum (λ i, 1 / (x i))) ≥ n^2 :=
by
  sorry

end cauchy_schwarz_inequality_l54_54015


namespace quadratic_passes_through_constant_point_l54_54642

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - m * x + 2 * m + 1

theorem quadratic_passes_through_constant_point :
  ∀ m : ℝ, f m 2 = 13 :=
by
  intro m
  unfold f
  simp
  rfl

end quadratic_passes_through_constant_point_l54_54642


namespace range_of_y_is_correct_l54_54610

noncomputable def range_of_y (n : ℝ) : ℝ :=
  if n > 2 then 1 / n else 2 * n^2 + 1

theorem range_of_y_is_correct :
  (∀ n, 0 < range_of_y n ∧ range_of_y n < 1 / 2 ∧ n > 2) ∨ (∀ n, 1 ≤ range_of_y n ∧ n ≤ 2) :=
sorry

end range_of_y_is_correct_l54_54610


namespace hyperbola_arithmetic_sequence_and_chord_length_l54_54453

theorem hyperbola_arithmetic_sequence_and_chord_length
  (h : hyperbola ℝ) 
  (h_center : h.center = (0, 0)) 
  (h_foci : h.foci = [((sqrt 5 : ℝ), 0), (-(sqrt 5 : ℝ), 0)])
  (h_eccentricity : h.eccentricity = (sqrt 5 / 2 : ℝ))
  (A B O : ℝ × ℝ)
  (F : ℝ × ℝ := ((sqrt 5 : ℝ), 0))
  (OA OB AB : ℝ)
  (h_OA : |OA| = 3 * m)
  (h_AB : |AB| = 4 * m)
  (h_OB : |OB| = 7 * m)
  (CD_length : ℝ) :
  (OA + AB = OB) ∧ (CD_length = (4 * sqrt 5) / 3) := sorry

end hyperbola_arithmetic_sequence_and_chord_length_l54_54453


namespace minimum_b_pupusa_pairs_l54_54420

theorem minimum_b_pupusa_pairs :
  ∃ b : ℕ, (∀ a : ℕ, a < 391 → b > 17 ∧ (Nat.lcm a b) > (Nat.lcm a 391)) ∧ b = 18 :=
by
  have h1 : 391 = 17 * 23 := by norm_num
  use 18
  intro a ha
  have h2 : a < 391 := ha
  have h3 : Nat.gcd a 391 ∣ 391 := Nat.gcd_dvd_right a 391
  sorry

end minimum_b_pupusa_pairs_l54_54420


namespace trig_expression_evaluation_l54_54956

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) :
  Real.sin θ ^ 2 + (Real.sin θ * Real.cos θ) - 2 * (Real.cos θ ^ 2) = 4 / 5 := 
by
  sorry

end trig_expression_evaluation_l54_54956


namespace triangle_area_l54_54279

noncomputable def area_of_triangle (b c : ℝ) (C : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin C

theorem triangle_area (a b c : ℝ) (C: ℝ) (h1 : b = 1) (h2 : c = sqrt 3) (h3 : C = (2 * Real.pi) / 3) :
  area_of_triangle b c C = sqrt 3 / 4 :=
by
  simp [area_of_triangle, h1, h2, h3]
  calc
    (1 / 2) * 1 * sqrt 3 * Real.sin ((2 * Real.pi) / 3)
        = (1 / 2) * sqrt 3 * Real.sin ((2 * Real.pi) / 3) : by rw [mul_one]
    ... = (sqrt 3) / 4 : by rw [Real.sin, pi_div_two, ...] -- continue calculation here
  sorry

end triangle_area_l54_54279


namespace number_of_correct_statements_is_3_l54_54035

-- Conditions as propositions
def condition1 : Prop := ∀ (flowchart : Flowchart), flowchart.has_start ∧ flowchart.has_end
def condition2 : Prop := ∀ (flowchart : Flowchart), 
  (∀ block, block.is_input → flowchart.precedes_start block) ∧ 
  (∀ block, block.is_output → flowchart.succeeds_end block)
def condition3 : Prop := ∀ block, block.is_decision ↔ block.has_multiple_exits
def condition4 : Prop := ∀ (flowchart : Flowchart), 
  ∃ decision_block : Block, ¬unique_conditions_in_decision_block flowchart decision_block

-- Proof problem stating that the number of correct conditions is 3
theorem number_of_correct_statements_is_3 : 
  (condition1 ∧ condition2 ∧ condition3) ∧ ¬condition4 ∧ ¬(∃ (Q1 Q2 Q3 Q4 : Prop), Q1 ∧ Q2 ∧ Q3 ∧ Q4 ∧ ¬(Q1 ∧ ¬Q2) ∧ ¬(Q2 ∧ ¬Q3) ∧ ¬(Q3 ∧ ¬Q4)) :=
by
  sorry

end number_of_correct_statements_is_3_l54_54035


namespace sum_of_integers_satisfying_conditions_l54_54944

-- Define a function to check if the sum of digits of n is divisible by 3
def sum_of_digits_div_by_3 (n : ℕ) : Prop :=
  (n.digits 10).sum % 3 = 0

-- Define the inequality condition
def inequality_condition (n : ℕ) : Prop :=
  1.5 * n - 6.3 < 7.5 

-- Main theorem statement encapsulating the problem
theorem sum_of_integers_satisfying_conditions :
  (Finset.range 10).filter (λ n, inequality_condition n ∧ sum_of_digits_div_by_3 n).sum id = 18 :=
by
  sorry

end sum_of_integers_satisfying_conditions_l54_54944


namespace exponentiation_81_5_4_eq_243_l54_54176

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end exponentiation_81_5_4_eq_243_l54_54176


namespace count_valid_three_digit_numbers_l54_54228

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ 
           (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ 
                          a ≥ 1 ∧ a ≤ 9 ∧ 
                          b ≥ 0 ∧ b ≤ 9 ∧ 
                          c ≥ 0 ∧ c ≤ 9 ∧ 
                          (a = b ∨ b = c ∨ a = c ∨ 
                           a + b > c ∧ a + c > b ∧ b + c > a)) ∧
           n = 57 := 
sorry

end count_valid_three_digit_numbers_l54_54228


namespace arithmetic_mean_geometric_mean_l54_54157

noncomputable theory

variables {r_1 r_3 r_4 r_5 t_1 t_3 t_4 t_5 : ℝ}

-- Proportional constant k (implicitly defined in terms of areas)
def area (k r : ℝ) := k * r^2

-- Given Areas of the polygons
def t1 := area k r_1
def t3 := area k r_3
def t4 := area k r_4
def t5 := area k r_5

-- Theorem statements
theorem arithmetic_mean (h_t1 : t1 = k * r_1^2) (h_t3 : t3 = k * r_3^2) (h_t5 : t5 = k * r_5^2) :
  (t1 + t3) / 2 = t5 :=
sorry

theorem geometric_mean (h_t1 : t1 = k * r_1^2) (h_t3 : t3 = k * r_3^2) (h_t4 : t4 = k * r_4^2) :
  real.sqrt (t1 * t3) = t4 :=
sorry

end arithmetic_mean_geometric_mean_l54_54157


namespace degree_g_l54_54373

-- Define the polynomial degrees
def degree (p : Polynomial ℝ) : ℕ := p.natDegree

-- Define functions for f, g, and h
variable (f g h : Polynomial ℝ)

-- Given Conditions
axiom degree_h : degree h = 9
axiom degree_f : degree f = 3
axiom h_def : h = f.comp(g) + g

-- Theorem to prove
theorem degree_g : degree g = 3 :=
sorry

end degree_g_l54_54373


namespace smallest_number_among_list_l54_54889

noncomputable def is_smallest (a b c d : ℝ) : Prop :=
  a < b ∧ a < c ∧ a < d ∧ 
  (b = -2) ∧ (c = sqrt 3) ∧ (d = 0) 

theorem smallest_number_among_list : is_smallest (-π) (-2) (sqrt 3) 0 := 
by {
  sorry
}

end smallest_number_among_list_l54_54889


namespace no_valid_arrangement_l54_54317

theorem no_valid_arrangement :
  ∀ (table : Fin 300 → Fin 300 → Int), 
  (∀ i j, table i j = 1 ∨ table i j = -1) →
  abs (∑ i j, table i j) < 30000 →
  (∀ i j, abs (∑ x in Finset.finRange 3, ∑ y in Finset.finRange 5, table (i + x) (j + y)) > 3) →
  (∀ i j, abs (∑ x in Finset.finRange 5, ∑ y in Finset.finRange 3, table (i + x) (j + y)) > 3) →
  False :=
by
  intros table h_entries h_total_sum h_rect_sum_3x5 h_rect_sum_5x3
  sorry

end no_valid_arrangement_l54_54317


namespace problem_l54_54611

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else real.sqrt x

theorem problem (x : ℝ) : f (f (-2)) = real.sqrt 5 := by
  sorry

end problem_l54_54611


namespace triangle_area_l54_54644

noncomputable def area_of_triangle (A B C : ℝ) [fact (A = 2)] [fact (B = 1)] [fact (C = π / 6)] : ℝ :=
  1 / 2 * B * (sqrt (A^2 - B^2)) * (sin C)

theorem triangle_area (A B C : ℝ) [fact (A = 2)] [fact (B = 1)] [fact (C = π / 6)] :
  area_of_triangle A B C = sqrt 3 / 2 :=
by sorry

end triangle_area_l54_54644


namespace bob_candies_count_l54_54146

-- Define the conditions
def isNeighbor (sam bob : Prop) := True  -- Bob is Sam's next door neighbor
def accompanyHome (bob : Prop) := True  -- Bob decided to accompany Sam home
def bobShare := {chewingGums := 15, chocolateBars := 20, assortedCandies := 15}

-- State the theorem
theorem bob_candies_count (sam bob : Prop) (h1 : isNeighbor sam bob) (h2 : accompanyHome bob) : 
  bobShare.assortedCandies = 15 :=
sorry

end bob_candies_count_l54_54146


namespace charles_pictures_after_work_l54_54915

variable (initial_papers : ℕ)
variable (draw_today : ℕ)
variable (draw_yesterday_morning : ℕ)
variable (papers_left : ℕ)

theorem charles_pictures_after_work :
    initial_papers = 20 →
    draw_today = 6 →
    draw_yesterday_morning = 6 →
    papers_left = 2 →
    initial_papers - (draw_today + draw_yesterday_morning + 6) = papers_left →
    6 = (initial_papers - draw_today - draw_yesterday_morning - papers_left) := 
by
  intros h1 h2 h3 h4 h5
  exact sorry

end charles_pictures_after_work_l54_54915


namespace claire_photos_l54_54080

theorem claire_photos : ∃ C : ℕ, (3 * C = C + 16) ∧ C = 8 := 
by
  use 8
  split
  · simp [mul_eq_mul_right_iff]
  · refl
  sorry

end claire_photos_l54_54080


namespace color_cube_ways_l54_54930

noncomputable def number_of_ways_to_color_cube (colors : Set ℕ) (vertices : Set ℕ) : ℕ :=
  -- let's assume a function that calculates the number of ways to color the cube
  sorry

theorem color_cube_ways (colors : Set ℕ) (vertices : Set ℕ) (h_colors: colors.card = 4) (h_vertices: vertices.card = 8) (h_disjoint: ∀ e ∈ vertices.prod vertices, e.1 ≠ e.2):
  number_of_ways_to_color_cube colors vertices = 2652 :=
sorry

end color_cube_ways_l54_54930


namespace fraction_of_field_planted_l54_54291

theorem fraction_of_field_planted : 
  let field_area := 5 * 6
  let triangle_area := (5 * 6) / 2
  let a := (41 * 3) / 33  -- derived from the given conditions
  let square_area := a^2
  let planted_area := triangle_area - square_area
  (planted_area / field_area) = (404 / 841) := 
by
  sorry

end fraction_of_field_planted_l54_54291


namespace relationship_among_a_b_c_l54_54238

variable (x y : ℝ)
variable (hx_pos : x > 0) (hy_pos : y > 0) (hxy_ne : x ≠ y)

noncomputable def a := (x + y) / 2
noncomputable def b := Real.sqrt (x * y)
noncomputable def c := 2 / ((1 / x) + (1 / y))

theorem relationship_among_a_b_c :
    a > b ∧ b > c := by
    sorry

end relationship_among_a_b_c_l54_54238


namespace right_triangle_DEF_properties_l54_54293

theorem right_triangle_DEF_properties :
  ∀ (D E F : Type) (DE DF EF : ℝ),
  DE = 15 → DF = 20 → EF = 25 →
  (DE ^ 2 + DF ^ 2 = EF ^ 2) → -- condition check for right triangle
  let median_distance := EF / 2 in
  let area := (1 / 2) * DE * DF in
  median_distance = 12.5 ∧ area = 150 := 
by
  intros
  sorry

end right_triangle_DEF_properties_l54_54293


namespace ratio_of_x_intercepts_l54_54065

theorem ratio_of_x_intercepts (c : ℝ) (u v : ℝ) (h1 : c ≠ 0) 
  (h2 : u = -c / 8) (h3 : v = -c / 4) : u / v = 1 / 2 :=
by {
  sorry
}

end ratio_of_x_intercepts_l54_54065


namespace fixed_point_quadratic_l54_54640

theorem fixed_point_quadratic : 
  (∀ m : ℝ, 3 * a ^ 2 - m * a + 2 * m + 1 = b) → (a = 2 ∧ b = 13) := 
by sorry

end fixed_point_quadratic_l54_54640


namespace charles_draws_yesterday_after_work_l54_54910

theorem charles_draws_yesterday_after_work :
  ∀ (initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work : ℕ),
    initial_papers = 20 →
    today_drawn = 6 →
    yesterday_drawn_before_work = 6 →
    current_papers_left = 2 →
    (initial_papers - (today_drawn + yesterday_drawn_before_work) - yesterday_drawn_after_work = current_papers_left) →
    yesterday_drawn_after_work = 6 :=
by
  intros initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work
  intro h1 h2 h3 h4 h5
  sorry

end charles_draws_yesterday_after_work_l54_54910


namespace charles_pictures_after_work_l54_54913

variable (initial_papers : ℕ)
variable (draw_today : ℕ)
variable (draw_yesterday_morning : ℕ)
variable (papers_left : ℕ)

theorem charles_pictures_after_work :
    initial_papers = 20 →
    draw_today = 6 →
    draw_yesterday_morning = 6 →
    papers_left = 2 →
    initial_papers - (draw_today + draw_yesterday_morning + 6) = papers_left →
    6 = (initial_papers - draw_today - draw_yesterday_morning - papers_left) := 
by
  intros h1 h2 h3 h4 h5
  exact sorry

end charles_pictures_after_work_l54_54913


namespace vector_parallel_unique_solution_l54_54983

def is_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

theorem vector_parallel_unique_solution (m : ℝ) :
  let a := (m^2 - 1, m + 1)
  let b := (1, -2)
  a ≠ (0, 0) → is_parallel a b → m = 1/2 := by
  sorry

end vector_parallel_unique_solution_l54_54983


namespace lambda_range_ordinate_midpoint_constant_Sn_value_Tn_value_l54_54625

open Nat

/-- Function f(x) = 1/2 + log_2(x / (1 - x)) -/
def f (x : ℝ) : ℝ :=
  1 / 2 + Real.log2 (x / (1 - x))

/-- Sum S_n = sum of f(i/n) for i from 1 to n-1 -/
noncomputable def S (n : ℕ) [Fact (1 < n)] : ℝ :=
  ∑ i in Finset.range (n - 1), f (i / n)

/-- Sequence a_n -/
def a (n : ℕ) : ℝ :=
  if n = 1 then 2 / 3 else 1 / ((S n + 1) * (S (n + 1) + 1))

/-- Sum of the first n terms of the sequence a_n -/
noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

/-- Conditions requirement for the range λ with T_n < λ (S_{n+1} + 1) -/
theorem lambda_range (Δ : ℕ → ℝ → Prop) : 
  ∃ λ : ℝ, ∀ n : ℕ, (n ≥ 1) → T n < λ * (S (n + 1) + 1) → (1 / 2) < λ :=
sorry

/-- Prove that the ordinate of the midpoint M is 1/2 -/
theorem ordinate_midpoint_constant :
  ∀ (x1 x2 : ℝ), x1 + x2 = 1 → 1 / 2 * ((1 / 2 + Real.log2 (x1 / (1 - x1))) + 
  (1 / 2 + Real.log2 ((1 - x1) / x1))) = 1 / 2 :=
sorry

/-- Prove that S_n = (n - 1) / 2 -/
theorem Sn_value (n ≥ 2) : 
  S n = (n - 1) / 2 :=
sorry

/-- Prove that T_n = 2n / (n + 2) -/
theorem Tn_value (n : ℕ) :
  T n = 2 * n / (n + 2) :=
sorry

end lambda_range_ordinate_midpoint_constant_Sn_value_Tn_value_l54_54625


namespace real_solutions_of_equation_l54_54923

theorem real_solutions_of_equation :
  ∃ (x1 x2 : ℝ), (2^(4 * x1) - 5 * 2^(2 * x1 + 1) + 4 = 0) ∧
                 (2^(4 * x2) - 5 * 2^(2 * x2 + 1) + 4 = 0) ∧ 
  ∀ x, (2^(4 * x) - 5 * 2^(2 * x + 1) + 4 = 0) → x = x1 ∨ x = x2 :=
by
  sorry

end real_solutions_of_equation_l54_54923


namespace remainder_of_division_l54_54060

theorem remainder_of_division (n1 n2 n3 n4 : ℕ) (h : {n1, n2, n3, n4} = {10, 11, 12, 13}) :
  (if h2s : 11 ∈ {n1, n2, n3, n4} ∧ 12 ∈ {n1, n2, n3, n4} then 
    let smin := 11 in
    let slar := 12 in
    slar % smin = 1
  else 
    false) := 
sorry

end remainder_of_division_l54_54060


namespace arithmetic_geometric_comparison_l54_54961

theorem arithmetic_geometric_comparison
  (a b : ℕ → ℝ)
  (n : ℕ)
  (b₁ b₊d : ℝ)
  (d : ℝ)
  (h_pos: d > 0)
  (h_a_arithmetic: ∀ n >= 1, a n = b₁ + (n - 1) * d)
  (h_b_geometric: ∀ n >= 1, b n = b₁ * (b₊d / b₁) ^ (n - 1))
  (h_a1_b1: a 1 = b₁)
  (h_a2_b2: a 2 = b₁ + d)
  (h_b1_pos: b₁ > 0)
: ∀ n ≥ 3, a n < b n := 
sorry

end arithmetic_geometric_comparison_l54_54961


namespace seth_can_erase_all_numbers_l54_54781

theorem seth_can_erase_all_numbers : 
  ∀ (board : Finset ℕ), board = Finset.filter (λ n, 1 ≤ n ∧ n ≤ 100) (Finset.range 101) →
  ∀ (b c : ℕ), b ≠ c → b ∈ board → c ∈ board →
  (∃ x y : ℤ, x * y = (c : ℤ) ∧ x + y = - (b : ℤ)) →
  (Finset.erase (Finset.erase board b) c) = ∅ :=
by
  intros board h_board b c h_diff h_b_in h_c_in h_solution
  sorry

end seth_can_erase_all_numbers_l54_54781


namespace cyclic_quadrilateral_BXMY_l54_54750

/-- Statement of the math problem in Lean 4 --/

theorem cyclic_quadrilateral_BXMY 
  (A B C M P Q X Y : Type) 
  [HasMidpoint M A C] 
  [HasSegment P A M] 
  [HasSegment Q C M]
  (h_mid : M = midpoint A C)
  (h_pq : PQ = segment_length A C / 2)
  (h_circle_ABQ : Circle (circumscribed (Triangle A B Q)) ∩ line BC = X ∧ X ≠ B)
  (h_circle_BCP : Circle (circumscribed (Triangle B C P)) ∩ line AB = Y ∧ Y ≠ B) : 
  cyclic (Quadrilateral B X M Y) := sorry

end cyclic_quadrilateral_BXMY_l54_54750


namespace integer_multiple_special_digits_l54_54763

theorem integer_multiple_special_digits (k : ℕ) (h : k > 1) : 
  ∃ (m : ℕ), (0 < m ∧ m < k^4 ∧ (m % k = 0) ∧ (∀ d ∈ list.digits 10 m, d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 9)) :=
by
  sorry

end integer_multiple_special_digits_l54_54763


namespace unique_midpoints_in_acute_triangle_l54_54349

noncomputable theory

variables {A B C A1 B1 C1 : Type*} [euclidean_geometry A] [acute_triangle ABC] 

def orthogonal_projection (P Q : Type*) [euclidean_geometry P] [euclidean_geometry Q] : Type* := sorry

def is_midpoint (M P Q : Type*) [euclidean_geometry M] [euclidean_geometry P] [euclidean_geometry Q] : Prop := 
  dist M P = dist M Q ∧ dist M P + dist M Q = dist P Q

def medians (ABC : Type*) [acute_triangle ABC] : Type* := 
  sorry -- definition for the triangle whose side lengths are the medians of triangle ABC

theorem unique_midpoints_in_acute_triangle (ABC : Type*) [acute_triangle ABC] :
  ∃! (A1 : orthogonal_projection B C) (B1 : orthogonal_projection C A) (C1 : orthogonal_projection A B), 
  is_midpoint A1 B1 C1 ∧ is_midpoint B1 C1 A1 ∧ is_midpoint C1 A1 B1 
  ∧ similar_triangles (triangle A1 B1 C1) (medians ABC) := 
sorry

end unique_midpoints_in_acute_triangle_l54_54349


namespace printed_value_l54_54638

theorem printed_value (X S : ℕ) (h1 : X = 5) (h2 : S = 0) : 
  (∃ n, S = (n * (3 * n + 7)) / 2 ∧ S ≥ 15000) → 
  X = 5 + 3 * 122 - 3 :=
by 
  sorry

end printed_value_l54_54638


namespace inequality_sqrt_diff_inequality_frac_l54_54844

-- Problem 1: Prove the inequality for all a > 0
theorem inequality_sqrt_diff (a : ℝ) (h : a > 0) : 
  sqrt(a + 5) - sqrt(a + 3) > sqrt(a + 6) - sqrt(a + 4) := 
sorry

-- Problem 2: Prove that at least one of the given inequalities holds for positive x, y such that x + y > 2
theorem inequality_frac (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
sorry

end inequality_sqrt_diff_inequality_frac_l54_54844


namespace total_clouds_count_l54_54500

def carson_clouds := 12
def little_brother_clouds := 5 * carson_clouds
def older_sister_clouds := carson_clouds / 2
def cousin_clouds := 2 * (older_sister_clouds + carson_clouds)

theorem total_clouds_count : carson_clouds + little_brother_clouds + older_sister_clouds + cousin_clouds = 114 := by
  sorry

end total_clouds_count_l54_54500


namespace ratio_of_spending_l54_54774

-- Define the initial amount of money
def initial_amount : ℝ := 400

-- Define the amount spent on school supplies as a fourth of the initial amount
def spent_on_supplies : ℝ := initial_amount / 4

-- Define the remaining amount after buying school supplies
def remaining_after_supplies : ℝ := initial_amount - spent_on_supplies

-- Define the amount left after spending on food for the faculty
def remaining_after_food : ℝ := 150

-- Calculate the amount spent on food
def spent_on_food : ℝ := remaining_after_supplies - remaining_after_food

-- Assert the ratio of the money spent on food to the amount left after buying school supplies
theorem ratio_of_spending : spent_on_food / remaining_after_supplies = 1 / 2 := by
  sorry

end ratio_of_spending_l54_54774


namespace compute_expression_l54_54723

noncomputable def real_log_base (b x : ℝ) := (Real.log x) / (Real.log b)

theorem compute_expression 
  (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h_eq : (real_log_base 4 x)^6 + (real_log_base 5 y)^6 + 18 = 12 * (real_log_base 4 x) * (real_log_base 5 y) ) :
  x^(Real.cbrt 4) + y^(Real.cbrt 4) = 4^(4 / 3) + 5^(4 / 3) :=
sorry

end compute_expression_l54_54723


namespace num_ways_distribute_pens_l54_54207

-- Conditions:
def num_friends : ℕ := 4
def num_pens : ℕ := 10
def at_least_one_pen_each (dist : fin num_friends → ℕ) : Prop := 
  ∀ i : fin num_friends, dist i ≥ 1
def at_most_five_pens_each (dist : fin num_friends → ℕ) : Prop :=
  ∀ i : fin num_friends, dist i ≤ 5
def total_pens_distributed (dist : fin num_friends → ℕ) : Prop :=
  finset.univ.sum dist = num_pens

-- Proof Problem:
theorem num_ways_distribute_pens :
  {dist : fin num_friends → ℕ // 
    at_least_one_pen_each dist ∧ at_most_five_pens_each dist ∧ total_pens_distributed dist}.card = 50 := 
sorry

end num_ways_distribute_pens_l54_54207


namespace rectangle_area_l54_54300

theorem rectangle_area (s : ℝ) (h_perimeter : 10 * s = 160) : 4 * s ^ 2 = 1024 := 
by
  -- We are given the perimeter condition: 10 * s = 160
  have h_s : s = 16, from 
    sorry -- since 10 * s = 160 leads to s = 16
  -- Applying the area formula: area = 4 * s ^ 2
  calc
    4 * s ^ 2 = 4 * (16 ^ 2) : by rw h_s
    ...      = 4 * 256 : by norm_num
    ...      = 1024 : by norm_num

end rectangle_area_l54_54300


namespace diameter_of_circle_with_inscribed_right_triangle_l54_54079

theorem diameter_of_circle_with_inscribed_right_triangle (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (right_triangle : a^2 + b^2 = c^2) : c = 10 :=
by
  subst h1
  subst h2
  simp at right_triangle
  sorry

end diameter_of_circle_with_inscribed_right_triangle_l54_54079


namespace cannot_repaint_infinitely_l54_54807

def round_fence (k : ℕ) := { idx : ℕ // idx < k }

structure Fence :=
(num_sections : ℕ)
(sections : round_fence num_sections → Fin 4)
(h_num_sections_even : num_sections % 2 = 0)
(h_num_sections_min : num_sections ≥ 6)

theorem cannot_repaint_infinitely (n : ℕ) (h : n ≥ 3) :
  let num_sections := 2 * n
  ∃ f : Fence, f.num_sections = num_sections →
  (∀ op : {triplet : round_fence f.num_sections → Fin 3 // ∀ i : Fin 3, triplet.val i ≠ triplet.val ((i + 1) % 3)},
    let f' : Fence := {
      num_sections := f.num_sections,
      sections := λ idx, if idx = triplet.1 0 ∨ idx = triplet.1 1 ∨ idx = triplet.1 2 then 4 else f.sections idx,
      h_num_sections_even := f.h_num_sections_even,
      h_num_sections_min := f.h_num_sections_min
    }
    ∃ k : ℕ, k = 0) :=
sorry

end cannot_repaint_infinitely_l54_54807


namespace average_density_of_stone_l54_54132

-- Definitions for the given conditions
def base_area := 25  -- cm^2
def density_water := 1  -- g/cm^3
def water_rise_on_block := 1.5  -- cm
def water_rise_with_stone := 0.5  -- cm

-- The proof statement
theorem average_density_of_stone :
  let S := base_area
  let ρ := density_water
  let h₁ := water_rise_on_block
  let h₂ := water_rise_with_stone
  ρ * h₁ * S / (h₂ * S) = 3 := by
  sorry

end average_density_of_stone_l54_54132


namespace probability_of_quit_l54_54401

noncomputable def probability_both_quit_from_same_tribe 
    (total_contestants quitting_contestants : ℕ) 
    (tribe_sizes : List ℕ) 
    (total_quit : ℕ := quitting_contestants := 2)
    (total_contestants := tribe_sizes.sum) 
    (tribe1 tribe2 tribe3 : ℕ := 6) : ℚ :=
begin
    -- Define the total number of ways to choose k quitters from n contestants
    let choose (n k : ℕ) : ℕ := n.choose k,

    -- Total ways to choose 2 quitters from 18 contestants
    let total_ways : ℕ := choose total_contestants quitting_contestants,

    -- Ways for both quitters from the same tribe (3 tribes, each calculation has 6 choose 2)
    let same_tribe_ways : ℕ := 3 * choose tribe1 quitting_contestants,

    -- Desired probability
    let prob := (same_tribe_ways : ℚ) / (total_ways : ℚ),

    exact prob
end

/-- The probability that both contestants who quit are from the same tribe is 5/17. -/
theorem probability_of_quit : probability_both_quit_from_same_tribe 18 2 [6, 6, 6] = 5 / 17 := by sorry

end probability_of_quit_l54_54401


namespace determine_m_l54_54612

theorem determine_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x : ℝ, x > 0 → f x = (m^2 - m - 1) * x^(m^2 - 2m - 1)) →
  (∀ x1 x2 : ℝ, 0 < x1 < x2 → f x1 < f x2) →
  m = -1 :=
by
  sorry

end determine_m_l54_54612


namespace fraction_zero_iff_x_is_four_l54_54826

theorem fraction_zero_iff_x_is_four (x : ℝ) (h_ne_zero: x + 4 ≠ 0) :
  (16 - x^2) / (x + 4) = 0 ↔ x = 4 :=
sorry

end fraction_zero_iff_x_is_four_l54_54826


namespace least_number_to_subtract_l54_54448

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (h : 1387 = n + k * 15) : n = 7 :=
by
  sorry

end least_number_to_subtract_l54_54448


namespace exists_large_triangle_containing_all_points_l54_54504

theorem exists_large_triangle_containing_all_points 
  {N : ℕ} 
  (points : Fin N → EuclideanSpace ℝ (Fin 2)) 
  (h : ∀ (i j k : Fin N), i ≠ j ∧ i ≠ k ∧ j ≠ k → triangle_area (points i) (points j) (points k) ≤ 1) 
  : ∃ (A B C : EuclideanSpace ℝ (Fin 2)), (∀ x : Fin N, point_in_triangle (points x) A B C) ∧ triangle_area A B C ≤ 4 := 
sorry

end exists_large_triangle_containing_all_points_l54_54504


namespace find_C_in_terms_of_D_l54_54340

noncomputable def h (C D x : ℝ) : ℝ := C * x - 3 * D ^ 2
noncomputable def k (D x : ℝ) : ℝ := D * x + 1

theorem find_C_in_terms_of_D (C D : ℝ) (h_eq : h C D (k D 2) = 0) (h_def : ∀ x, h C D x = C * x - 3 * D ^ 2) (k_def : ∀ x, k D x = D * x + 1) (D_ne_neg1 : D ≠ -1) : 
C = (3 * D ^ 2) / (2 * D + 1) := 
by 
  sorry

end find_C_in_terms_of_D_l54_54340


namespace min_unique_numbers_written_by_vasya_l54_54800

def integer_part (x : ℝ) : ℤ := ⌊x⌋
def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋
def vasya_numbers (x : ℝ) : ℝ × ℝ := (integer_part x, 1 / fractional_part x)

theorem min_unique_numbers_written_by_vasya (x : ℕ → ℝ) (h : ∀ i j, i ≠ j → x i ≠ x j) : 
  20 ≤ ((λ n, (integer_part (x n), 1 / fractional_part (x n))) '' set.univ).to_finset.card :=
sorry

end min_unique_numbers_written_by_vasya_l54_54800


namespace length_of_AP_l54_54298

-- Given conditions
def side_ABCD : ℝ := 8
def ZY : ℝ := 12
def XY : ℝ := 8
def shaded_area : ℝ := (XY * ZY) / 3

-- Definitions involving geometrical relationships
def area_WXYZ : ℝ := XY * ZY
def AD_parallel_WX : Prop := true -- placeholder, since we don't need the property itself in the proof
def P_on_WX : Prop := true -- placeholder

-- Proof goal
theorem length_of_AP :
  ∃ AP : ℝ, 
    let PD := shaded_area / side_ABCD in
    AP = side_ABCD - PD ∧ PD * side_ABCD = shaded_area := 
  by {
    let PD := shaded_area / side_ABCD,
    exact ⟨side_ABCD - PD, by sorry⟩
  }

end length_of_AP_l54_54298


namespace find_coefficients_l54_54711

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (A B C D : V)
variable (m n : ℝ)

-- Conditions from the problem
def conditions (h : V) : Prop := 
  ∃ (t : ℝ), t • (B - C) = D - C ∧ t = 3

noncomputable def values_of_m_and_n : Prop := 
    m = -1/3 ∧ n = 4/3

-- The theorem statement
theorem find_coefficients (h : conditions D) : values_of_m_and_n :=
sorry

end find_coefficients_l54_54711


namespace mac_runs_faster_than_apple_l54_54894

theorem mac_runs_faster_than_apple :
  let Apple_speed := 3 -- miles per hour
  let Mac_speed := 4 -- miles per hour
  let Distance := 24 -- miles
  let Apple_time := Distance / Apple_speed -- hours
  let Mac_time := Distance / Mac_speed -- hours
  let Time_difference := (Apple_time - Mac_time) * 60 -- converting hours to minutes
  Time_difference = 120 := by
  sorry

end mac_runs_faster_than_apple_l54_54894


namespace Hannah_number_is_1189_l54_54929

noncomputable theory

-- Definitions of the sequences each student skips
def Alice_skips (n : ℕ) : Prop := n % 4 = 0
def Barbara_skips (n : ℕ) : Prop := ¬ Alice_skips n ∧ (n / 4 + 1) % 5 = 0
def Candice_skips (n : ℕ) : Prop := ¬ Alice_skips n ∧ ¬ Barbara_skips n ∧ (n / 5 + 1) % 6 = 0
def Debbie_Eliza_Fatima_skips (n : ℕ) : Prop :=
  ¬ Alice_skips n ∧ ¬ Barbara_skips n ∧ ¬ Candice_skips n ∧ (n / 6 + 1) % 7 = 0
def George_says (n : ℕ) : Prop :=
  ¬ Alice_skips n ∧ ¬ Barbara_skips n ∧ ¬ Candice_skips n ∧ ¬ Debbie_Eliza_Fatima_skips n

-- Defining Hannah's number
def Hannah_says (n : ℕ) : Prop :=
  ¬ George_says n ∧ ∀ m, m ≠ n → George_says m

-- The final statement to prove
theorem Hannah_number_is_1189 : Hannah_says 1189 :=
by
  sorry

end Hannah_number_is_1189_l54_54929


namespace find_m_l54_54628

theorem find_m (m : ℝ) :
  let a : ℝ × ℝ := (2, m)
  let b : ℝ × ℝ := (1, -1)
  (b.1 * (a.1 + 2 * b.1) + b.2 * (a.2 + 2 * b.2) = 0) → 
  m = 6 := by 
  sorry

end find_m_l54_54628


namespace integer_multiple_special_digits_l54_54762

theorem integer_multiple_special_digits (k : ℕ) (h : k > 1) : 
  ∃ (m : ℕ), (0 < m ∧ m < k^4 ∧ (m % k = 0) ∧ (∀ d ∈ list.digits 10 m, d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 9)) :=
by
  sorry

end integer_multiple_special_digits_l54_54762


namespace relationship_of_a_b_c_l54_54224

variable {f : ℝ → ℝ}

theorem relationship_of_a_b_c
  (hf_strict_increasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0)
  (a_def : a = f (0.64))
  (b_def : b = f (Real.log 5 / Real.log 2))
  (c_def : c = f (2 ^ 0.8)) : 
  b < a ∧ a < c := 
sorry

end relationship_of_a_b_c_l54_54224


namespace sequence_a_5_l54_54204

theorem sequence_a_5 (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → S n = 2 * a n - 3) (h2 : ∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) :
  a 5 = 48 := by
  -- The proof and implementations are omitted
  sorry

end sequence_a_5_l54_54204


namespace quadratic_roots_product_l54_54432

/-
Given a quadratic equation of the form -49 = -2x^2 + 6x, we want to prove
that the product of its roots is -24.5.
-/
noncomputable def prod_of_roots : Prop :=
  let a := 2
  let c := -49 in
  (c / a) = -24.5

theorem quadratic_roots_product : prod_of_roots :=
  by
    sorry

end quadratic_roots_product_l54_54432


namespace second_number_is_34_l54_54053

theorem second_number_is_34 (x y z : ℝ) (h1 : x + y + z = 120) 
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 34 :=
by 
  sorry

end second_number_is_34_l54_54053


namespace increasing_sequence_lambda_range_l54_54236

open Nat

theorem increasing_sequence_lambda_range (λ : ℝ) (a : ℕ+ → ℝ) (h : ∀ n : ℕ+, a n = n^2 + λ * n) :
  (∀ n : ℕ+, a (n + 1) - a n > 0) ↔ λ > -3 :=
by
  sorry

end increasing_sequence_lambda_range_l54_54236


namespace number_of_standard_deviations_l54_54380

theorem number_of_standard_deviations (μ σ v : ℝ) (hμ : μ = 17.5) (hσ : σ = 2.5) (hv : v = 12.5) : 
  (v - μ) / σ = -2 := 
by 
  rw [hμ, hσ, hv]
  norm_num
  sorry

end number_of_standard_deviations_l54_54380


namespace minimum_distance_to_line_l54_54637

-- Define the function f(x) = x^2 - ln(x)
def f (x : ℝ) := x^2 - Real.log x

-- Define the line equation x - y - 2 = 0
def line (x y : ℝ) := x - y - 2 = 0

-- Point P (x, f(x)) is on the curve
def point_on_curve (x : ℝ) := (x, f x)

-- Define the distance from point (x1, y1) to the line Ax + By + C = 0
def distance_from_point_to_line (x1 y1 A B C : ℝ) :=
  abs (A * x1 + B * y1 + C) / Real.sqrt (A^2 + B^2)

-- The proof problem: prove the minimum distance from P to the line is √2
theorem minimum_distance_to_line : 
  ∃ (x : ℝ), point_on_curve x → distance_from_point_to_line x (f x) 1 (-1) (-2) = Real.sqrt 2 :=
sorry

end minimum_distance_to_line_l54_54637


namespace domino_placing_comparison_l54_54353

theorem domino_placing_comparison (N S : ℕ) :
  let board_size := 8
  let domino_size := (2, 1)
  dominos_not_overlap : (∀ (placements : list (ℕ × ℕ)), 
                              (∀ (p : ℕ × ℕ), p ∈ placements → 
                                  p.1 < board_size ∧ p.2 < board_size)
                            ∧ (∀ (p q : ℕ × ℕ), p ≠ q → 
                                  ¬(p.1 = q.1 ∧ abs (p.2 - q.2) < domino_size.2)
                               ∧ ¬(p.2 = q.2 ∧ abs (p.1 - q.1) < domino_size.1)))
  (ways_to_place_dominoes :
    let placements32 := (list.rep (board_size * board_size // 2) domino_size)
    let placements16 := (list.rep (board_size * board_size // 4) domino_size)
    ∃ valid_placements_32: list (list (ℕ × ℕ)), valid_placements_32.length = N
    ∃ valid_placements_16: list (list (ℕ × ℕ)), valid_placements_16.length = S
  )
  → S > N := 
sorry

end domino_placing_comparison_l54_54353


namespace poly_divisibility_l54_54014

open Polynomial

noncomputable def poly_divisible (n : ℕ) : Prop :=
  ∀ (x : ℂ), ((x^2 + x + 1) ∣ (x+1)^(2*n+1) + x^(n+2))

theorem poly_divisibility (n : ℕ) : poly_divisible n :=
begin
  sorry
end

end poly_divisibility_l54_54014


namespace determine_pounds_purchased_l54_54858

noncomputable def calculate_pounds_purchased 
  (cost_rate : ℝ) 
  (sell_rate : ℝ) 
  (total_profit : ℝ)
  (cost_per_pound : ℝ := cost_rate⁻¹)
  (sell_per_pound : ℝ := sell_rate⁻¹)
  (profit_per_pound : ℝ := sell_per_pound - cost_per_pound) 
: ℝ :=
  total_profit / profit_per_pound

theorem determine_pounds_purchased 
  (cost_rate : ℝ) (h_cost_rate : cost_rate = 3 / 0.50) 
  (sell_rate : ℝ) (h_sell_rate : sell_rate = 4 / 1.00) 
  (total_profit : ℝ) (h_total_profit : total_profit = 8) 
: calculate_pounds_purchased cost_rate sell_rate total_profit = 96 :=
  by 
    rw [←h_cost_rate, ←h_sell_rate, ←h_total_profit]
    sorry

end determine_pounds_purchased_l54_54858


namespace inequality_proof_l54_54455

theorem inequality_proof 
  {x : ℝ} (n : ℕ) (hx : 0 < x) (hn : 0 < n) :
  (n : ℝ) / ((x + 1) * (x + 2 * n)) ≤ (finset.sum (finset.range (2 * n)) (λ i, (-1 : ℝ)^i * (1 / (x + (i + 1))))) ∧
  (finset.sum (finset.range (2 * n)) (λ i, (-1 : ℝ)^i * (1 / (x + (i + 1))))) < (n : ℝ) / ((x + 0.5) * (x + 2 * n + 0.5)) :=
sorry

end inequality_proof_l54_54455


namespace find_f_2013_l54_54536

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, f(x - y) = f(x) + f(y) - 2 * x * y

theorem find_f_2013 : f(2013) = 2013^2 := 
by 
  -- Proof steps, skipped with sorry
  sorry

end find_f_2013_l54_54536


namespace sampling_interval_and_elimination_l54_54803

theorem sampling_interval_and_elimination (N n : ℕ) (hN : N = 92) (hn : n = 30) :
  let k := N / n in
  k = 3 ∧ 2 = k - 1 := by
  sorry

end sampling_interval_and_elimination_l54_54803


namespace count_integer_values_not_satisfying_inequality_l54_54202

theorem count_integer_values_not_satisfying_inequality : 
  ∃ n : ℕ, 
  (n = 3) ∧ (∀ x : ℤ, (4 * x^2 + 22 * x + 21 ≤ 25) → (-2 ≤ x ∧ x ≤ 0)) :=
by
  sorry

end count_integer_values_not_satisfying_inequality_l54_54202


namespace drink_price_is_correct_l54_54739

variable (sandwich_price : ℝ)
variable (coupon_fraction : ℝ)
variable (avocado_upgrade_cost : ℝ)
variable (salad_cost : ℝ)
variable (total_lunch_cost : ℝ)

def discount := sandwich_price * coupon_fraction
def discounted_sandwich_price := sandwich_price - discount
def upgraded_sandwich_price := discounted_sandwich_price + avocado_upgrade_cost
def combined_meal_cost := upgraded_sandwich_price + salad_cost
def drink_cost := total_lunch_cost - combined_meal_cost

theorem drink_price_is_correct
  (h1 : sandwich_price = 8)
  (h2 : coupon_fraction = 1/4)
  (h3 : avocado_upgrade_cost = 1)
  (h4 : salad_cost = 3)
  (h5 : total_lunch_cost = 12) : drink_cost = 2 := by
  sorry

end drink_price_is_correct_l54_54739


namespace find_a_of_extreme_value_at_one_l54_54779

-- Define the function f(x) = x^3 - a * x
def f (x a : ℝ) : ℝ := x^3 - a * x
  
-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 3 * x^2 - a

-- The theorem statement: for f(x) having an extreme value at x = 1, the corresponding a must be 3
theorem find_a_of_extreme_value_at_one (a : ℝ) : 
  (f' 1 a = 0) ↔ (a = 3) :=
by
  sorry

end find_a_of_extreme_value_at_one_l54_54779


namespace cube_volume_and_diagonal_from_surface_area_l54_54105

theorem cube_volume_and_diagonal_from_surface_area
    (A : ℝ) (h : A = 150) :
    ∃ (V : ℝ) (d : ℝ), V = 125 ∧ d = 5 * Real.sqrt 3 :=
by
  sorry

end cube_volume_and_diagonal_from_surface_area_l54_54105


namespace find_angle_between_vectors_l54_54240

noncomputable def vec_length (v : ℝ × ℝ × ℝ) : ℝ := 
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def theta {a b : ℝ × ℝ × ℝ}
  (ha : vec_length a = 4)
  (hb : vec_length b = 3)
  (hab : vec_length (a - b) = real.sqrt 37) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let cosine := (dot_product / (4 * 3)) in
  real.acos (cosine)

theorem find_angle_between_vectors
  {a b : ℝ × ℝ × ℝ}
  (ha : vec_length a = 4)
  (hb : vec_length b = 3)
  (hab : vec_length (a - b) = real.sqrt 37) :
  theta ha hb hab = 2 * real.pi / 3 :=
sorry

end find_angle_between_vectors_l54_54240


namespace vasya_grades_l54_54677

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧ 
    (grades.nthLe 2 (by linarith) = 4) ∧ 
    (grades.sum = 19) ∧
    (grades.count 5 > List.foldl (fun acc n => if n ≠ 5 then acc + grades.count n else acc) 0 grades) ∧ 
    (grades.sorted = [2, 3, 4, 5, 5]) :=
sorry

end vasya_grades_l54_54677


namespace binomial_expansion_l54_54933

theorem binomial_expansion :
  (∑ k in Finset.range 51, (Nat.choose 50 k) * (1 / 2) ^ (50 - k) * (-1) ^ k) = (1 / 2) ^ 50 :=
sorry

end binomial_expansion_l54_54933


namespace log_a_b_is_constant_l54_54851

open Real

-- Definitions of the conditions
def r (a : ℝ) : ℝ := log 10 (a ^ 2)
def C (b : ℝ) : ℝ := log 10 (b ^ 6)

-- Circle circumference formula
axiom circumference {a b : ℝ} : C b = 2 * π * r a

theorem log_a_b_is_constant (a b : ℝ) (ha : a > 0) (hb : b > 0) (hc : circumference) :
  log a b = (2 * π) / 3 :=
by sorry

end log_a_b_is_constant_l54_54851


namespace vasya_grades_l54_54674

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧
    (grades.filter (λ x, x = 5)).length > 2 ∧
    List.sorted (≤) grades ∧
    grades.nth 2 = some 4 ∧
    (grades.sum : ℚ) / 5 = 3.8 ∧
    grades = [2, 3, 4, 5, 5] := by
  sorry

end vasya_grades_l54_54674


namespace polar_equation_of_curve_C_polar_coordinates_of_point_T_l54_54256

noncomputable def polar_equation_curve_C (α : ℝ) (hα : α ∈ set.Icc (- real.pi / 2) (real.pi / 2)) : ℝ × ℝ :=
  let x := real.cos α
  let y := 1 + real.sin α
  ⟨x, y⟩

theorem polar_equation_of_curve_C :
  ∀ (r θ : ℝ), θ ∈ set.Icc 0 (real.pi / 2) → 
  (r = 2 * real.sin θ ↔ ∃ α, α ∈ set.Icc (- real.pi / 2) (real.pi / 2) ∧ 
  (real.cos α, 1 + real.sin α) = (real.cos θ, r)) :=
by sorry

theorem polar_coordinates_of_point_T (OT : ℝ) (hOT : OT = real.sqrt 3) :
  ∃ θ, θ = real.pi / 3 ∧ ∃ r, r = OT :=
by sorry

end polar_equation_of_curve_C_polar_coordinates_of_point_T_l54_54256


namespace area_of_region_is_4pi_l54_54427

-- Define the equation as a condition
def equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 6 * y + 9 = 0

-- Define the area enclosed by the region defined by the equations.
theorem area_of_region_is_4pi :
  (∃ (x y : ℝ), equation x y) → (4 * π) :=
sorry

end area_of_region_is_4pi_l54_54427


namespace common_ratio_l54_54335

-- Definitions for the geometric sequence
variables {a_n : ℕ → ℝ} {S_n q : ℝ}

-- Conditions provided in the problem
def condition1 (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  S_n 3 = a_n 1 + a_n 2 + a_n 3

def condition2 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2 + a_n 3) = a_n 4 - 2

def condition3 (a_n : ℕ → ℝ) (S_n : ℝ) : Prop :=
  3 * (a_n 1 + a_n 2) = a_n 3 - 2

-- The theorem we want to prove
theorem common_ratio (a_n : ℕ → ℝ) (q : ℝ) :
  condition2 a_n S_n ∧ condition3 a_n S_n → q = 4 :=
by
  sorry

end common_ratio_l54_54335


namespace incircle_tangent_reciprocal_equality_l54_54753

open Real EuclideanGeometry

theorem incircle_tangent_reciprocal_equality
  (A B C D E M N : Point)
  (hABD : IncircleTangency X; IncircleTangency Y)
  (h₁ : SegmentCollinear [A, D, B])
  (h₂ : SegmentCollinear [A, E, C])
  (h₃ : \(\angle BAD = \angle CAE\))
  (M_tangent : Tangent X BC)
  (N_tangent : Tangent Y BC) :
  (1 / dist M B + 1 / dist M D = 1 / dist N C + 1 / dist N E) :=
by
  sorry

end incircle_tangent_reciprocal_equality_l54_54753


namespace chef_served_173_guests_l54_54098

noncomputable def total_guests_served : ℕ :=
  let adults := 58
  let children := adults - 35
  let seniors := 2 * children
  let teenagers := seniors - 15
  let toddlers := teenagers / 2
  adults + children + seniors + teenagers + toddlers

theorem chef_served_173_guests : total_guests_served = 173 :=
  by
    -- Proof will be provided here.
    sorry

end chef_served_173_guests_l54_54098


namespace universal_proposition_is_optionA_l54_54439

    constant Circle : Type
    constant has_circumscribed_quadrilateral : Circle → Prop
    constant sqrt3_gt_sqrt2 : Prop := (real.sqrt 3 > real.sqrt 2)
    constant sqrt3_lt_sqrt2 : Prop := (real.sqrt 3 < real.sqrt 2)
    constant is_right_triangle : (ℕ → ℕ → ℕ → Prop) := 
      λ a b c, a^2 + b^2 = c^2

    -- Define the statements
    def optionA : Prop := ∀ c : Circle, has_circumscribed_quadrilateral c
    def optionB : Prop := sqrt3_gt_sqrt2
    def optionC : Prop := sqrt3_lt_sqrt2
    def optionD : Prop := is_right_triangle 3 4 5

    -- The theorem we want to prove
    theorem universal_proposition_is_optionA :
      (∀ (p : Prop), p = optionA → (optionB ∨ optionC ∨ optionD) → False) :=
    by
      sorry
    
end universal_proposition_is_optionA_l54_54439


namespace evaluate_81_power_5_div_4_l54_54186

-- Define the conditions
def base_factorized : ℕ := 3 ^ 4
def power_rule (b : ℕ) (m n : ℝ) : ℝ := (b : ℝ) ^ m ^ n

-- Define the primary calculation
noncomputable def power_calculation : ℝ := 81 ^ (5 / 4)

-- Prove that the calculation equals 243
theorem evaluate_81_power_5_div_4 : power_calculation = 243 := 
by
  have h1 : base_factorized = 81 := by sorry
  have h2 : power_rule 3 4 (5 / 4) = 3 ^ 5 := by sorry
  have h3 : 3 ^ 5 = 243 := by sorry
  have h4 : power_calculation = power_rule 3 4 (5 / 4) := by sorry
  rw [h1, h2, h3, h4]
  exact h3

end evaluate_81_power_5_div_4_l54_54186


namespace vasya_grades_l54_54673

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧
    (grades.filter (λ x, x = 5)).length > 2 ∧
    List.sorted (≤) grades ∧
    grades.nth 2 = some 4 ∧
    (grades.sum : ℚ) / 5 = 3.8 ∧
    grades = [2, 3, 4, 5, 5] := by
  sorry

end vasya_grades_l54_54673


namespace eval_exp_l54_54175

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end eval_exp_l54_54175


namespace vasya_grades_l54_54660

variables
  (grades : List ℕ)
  (length_grades : grades.length = 5)
  (median_grade : grades.nthLe 2 sorry = 4)  -- Assuming 0-based indexing
  (mean_grade : (grades.sum : ℚ) / 5 = 3.8)
  (most_frequent_A : ∀ n ∈ grades, n ≤ 5)

theorem vasya_grades (h : ∀ x ∈ grades, x ≤ 5 ∧ ∃ k, grades.nthLe 3 sorry = 5 ∧ grades.count 5 > grades.count x):
  ∃ g1 g2 g3 g4 g5 : ℕ, grades = [g1, g2, g3, g4, g5] ∧ g1 ≤ g2 ∧ g2 ≤ g3 ∧ g3 ≤ g4 ∧ g4 ≤ g5 ∧ [g1, g2, g3, g4, g5] = [2, 3, 4, 5, 5] :=
sorry

end vasya_grades_l54_54660


namespace largest_angle_l54_54395

noncomputable section

open Real

theorem largest_angle {u : ℝ} (h : 1 ≤ u) : 
  let a := sqrt (2 * u - 1)
  let b := sqrt (2 * u + 1)
  let c := 2 * sqrt u 
  (a^2 + b^2 = c^2) → 
  largestAngle a b c = 90 := 
by
  sorry

end largest_angle_l54_54395


namespace polygon_gcd_condition_l54_54938

theorem polygon_gcd_condition (n : ℕ) (hn : n ≥ 4) :
  (∀ P : polygon, 
    P.is_convex ∧ P.vertices_integers ∧ P.sides = n →
    (∀ (S : ℝ) (S1 S2 S3 ... Sn : ℝ), 
      S = P.area ∧ 
      (∀ i, Si = P.triangle_area i) → 
      (gcd (2 * S1) (2 * S2) ... (2 * Sn) ∣ 2 * S)) 
  ) ↔ (n = 4 ∨ n = 5) := 
sorry

end polygon_gcd_condition_l54_54938


namespace brian_tape_needed_l54_54148

-- Definitions of conditions
def tape_needed_for_box (short_side: ℕ) (long_side: ℕ) : ℕ := 
  2 * short_side + long_side

def total_tape_needed (num_short_long_boxes: ℕ) (short_side: ℕ) (long_side: ℕ) (num_square_boxes: ℕ) (side: ℕ) : ℕ := 
  (num_short_long_boxes * tape_needed_for_box short_side long_side) + (num_square_boxes * 3 * side)

-- Theorem statement
theorem brian_tape_needed : total_tape_needed 5 15 30 2 40 = 540 := 
by 
  sorry

end brian_tape_needed_l54_54148


namespace vasya_grades_l54_54675

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧ 
    (grades.nthLe 2 (by linarith) = 4) ∧ 
    (grades.sum = 19) ∧
    (grades.count 5 > List.foldl (fun acc n => if n ≠ 5 then acc + grades.count n else acc) 0 grades) ∧ 
    (grades.sorted = [2, 3, 4, 5, 5]) :=
sorry

end vasya_grades_l54_54675


namespace number_of_paths_enclosing_one_bounded_region_l54_54137

-- Definitions
def up_right_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m -- number of up-right paths from (0, 0) to (m, n)

def line_y_eq_x_sub (a : ℝ) (x : ℝ) : ℝ :=
  x - a -- the equation y = x - a

-- The main theorem to be proven
theorem number_of_paths_enclosing_one_bounded_region :
  let a := 2.021 in
  let p := (0, 0) in
  let q := (7, 7) in
  let line := line_y_eq_x_sub a in
  (up_right_paths 10 4 - up_right_paths 11 3) = 637 :=
by
  sorry

end number_of_paths_enclosing_one_bounded_region_l54_54137


namespace rational_terms_not_adjacent_probability_l54_54659

theorem rational_terms_not_adjacent_probability :
  let n := 8
  let binom_expansion := λ (r : ℕ), (nat.choose n r) * (2^r) * (x ^ ((16 - 3 * r) / 4))
  let rational_terms := {r | r ∈ {0, 4, 8}}
  let total_permutations := nat.factorial 9
  let irr_terms_permutations := nat.factorial 6
  let slots := 7 -- number of slots where 3 rational terms can go in a non-adjacent way
  let rational_terms_arrangements := irr_terms_permutations * (nat.factorial 7) / (nat.factorial 4)
  let probability := rational_terms_arrangements / total_permutations
  in probability = 5 / 12 :=
by
  sorry

end rational_terms_not_adjacent_probability_l54_54659


namespace adoption_combinations_l54_54482

theorem adoption_combinations (parrots snakes rabbits : ℕ) 
  (emily_choices : parrots + rabbits) (susan_choices : snakes)
  (john_choices : parrots + rabbits) 
  (h_parrots : parrots = 20) (h_snakes : snakes = 10) (h_rabbits : rabbits = 12) 
  (h_emily_choices : emily_choices = 32) (h_john_choices : john_choices = 32) 
  (h_susan_choices : susan_choices = 10) : 
  emily_choices * susan_choices * john_choices = 4800 :=
by 
  rw [h_parrots, h_snakes, h_rabbits] at *
  have h_scenario1 : 20 * 10 * 12 = (20 : ℕ) * (10 : ℕ) * (12 : ℕ) := by norm_num
  have h_scenario2 : 12 * 10 * 20 = (12 : ℕ) * (10 : ℕ) * (20 : ℕ) := by norm_num
  have h_combined : 20 * 10 * 12 + 12 * 10 * 20 = 4800 := by norm_num
  assumption sorry

end adoption_combinations_l54_54482


namespace problem1_problem2_problem3_l54_54347

section
  variables (ABC : Type) [Triangle ABC]
  variables {h1 h2 h3 u v w : ℝ}
  variables {M : Point ABC}

  -- Problem 1
  theorem problem1 (H1 : ∀ (h1 u : ℝ), 0 < u ∧ 0 < h1 → h1 / u ≥ 9) 
                   (H2 : ∀ (h2 v : ℝ), 0 < v ∧ 0 < h2 → h2 / v ≥ 9) 
                   (H3 : ∀ (h3 w : ℝ), 0 < w ∧ 0 < h3 → h3 / w ≥ 9) 
                   (cond1 : 0 < u) (cond2 : 0 < v) (cond3 : 0 < w) 
                   (cond4 : 0 < h1) (cond5 : 0 < h2) (cond6 : 0 < h3) :
                   h1 / u + h2 / v + h3 / w ≥ 9 := 
  sorry

  -- Problem 2
  theorem problem2 (H1 : ∀ (h1 h2 h3 u v w : ℝ), 0 < u ∧ 0 < v ∧ 0 < w ∧ 0 < h1 ∧ 0 < h2 ∧ 0 < h3 → 
                          h1 * h2 * h3 ≥ 27 * u * v * w) 
                   (cond1 : 0 < u) (cond2 : 0 < v) (cond3 : 0 < w) 
                   (cond4 : 0 < h1) (cond5 : 0 < h2) (cond6 : 0 < h3) : 
                   h1 * h2 * h3 ≥ 27 * u * v * w := 
  sorry

  -- Problem 3
  theorem problem3 (H1 : ∀ (h1 h2 h3 u v w : ℝ), 0 < u ∧ 0 < v ∧ 0 < w ∧ 0 < h1 ∧ 0 < h2 ∧ 0 < h3 → 
                          (h1 - u) * (h2 - v) * (h3 - w) ≥ 8 * u * v * w) 
                   (cond1 : 0 < u) (cond2 : 0 < v) (cond3 : 0 < w) 
                   (cond4 : 0 < h1) (cond5 : 0 < h2) (cond6 : 0 < h3) : 
                   (h1 - u) * (h2 - v) * (h3 - w) ≥ 8 * u * v * w := 
  sorry
end

end problem1_problem2_problem3_l54_54347


namespace angle_between_vectors_l54_54627

namespace VectorAngle

open Real

def vec_a : ℝ × ℝ := (-1, 0)
def vec_b : ℝ × ℝ := (sqrt 3 / 2, 1 / 2)

theorem angle_between_vectors :
  let dot_product := vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2
  let magnitude_a := sqrt (vec_a.1^2 + vec_a.2^2)
  let magnitude_b := sqrt (vec_b.1^2 + vec_b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = - (sqrt 3) / 2 → 
  arccos cos_theta = 5 * π / 6 := by
  sorry

end VectorAngle

end angle_between_vectors_l54_54627


namespace find_size_of_C_and_area_of_triangle_l54_54591

variables (A B C : ℝ) (m : ℝ) (AB AC S : ℝ)

-- Condition: A, B, and C are the internal angles of triangle ABC.
-- Condition: The equation x^2 + sqrt(3) mx - m + 1 = 0 has roots tan A and tan B.
axiom triangle_angles : A + B + C = π
axiom tan_roots : ∀ x : ℝ, x^2 + real.sqrt 3 * m * x - m + 1 = 0 ↔ x = real.tan A ∨ x = real.tan B

-- Given lengths of sides AB and AC.
axiom AB_length : AB = real.sqrt 6
axiom AC_length : AC = 2

-- Proof goals:
-- 1. The size of C is π/3.
-- 2. The area S of triangle ABC is (sqrt(6) + 3√2) / 4.

theorem find_size_of_C_and_area_of_triangle :
  C = π / 3 ∧ S = (real.sqrt 6 + 3 * real.sqrt 2) / 4 :=
sorry

end find_size_of_C_and_area_of_triangle_l54_54591


namespace find_BAC_l54_54379

-- Define the conditions
variables {α β γ : ℝ}  -- Angles in the triangle
variables {x y z : ℝ}  -- Half-angles
variables {AH CL O : Prop}  -- Geometric assumptions for the proof

-- State the given conditions
def conditions (AH CL O : Prop) : Prop :=
  -- Conditions are the altitude and angle bisector intersection
  AH ∧ CL ∧ O ∧ 
  -- Difference between angle COH and half of angle ABC is 46 degrees
  (46 = 90 - z - y)

-- State the proof problem
theorem find_BAC (h : conditions AH CL O) : α = 92 := 
  sorry

end find_BAC_l54_54379


namespace exists_infinite_sequence_l54_54165

theorem exists_infinite_sequence : 
  ∃ (a : ℕ → ℕ), (∀ i, 0 < a i) 
  ∧ (∀ m n, 1 ≤ m ∧ m < n → 
        ¬ ((∑ i in finset.range (n-m), a (m + i + 1)) 
          ∣ ∑ i in finset.range m, a (i + 1))) := 
sorry

end exists_infinite_sequence_l54_54165


namespace median_divides_triangle_l54_54964

theorem median_divides_triangle (A B C N K M : Type) 
  [is_triangle A B C]
  (hN : on_extension_of_side A C N ∧ euclidean_distance C N = euclidean_distance A C)
  (hK : midpoint K A B)
  (hM : intersect_line_segment K N (line B C) M)
  : ratio_segment B M M C = 2 :=
sorry

end median_divides_triangle_l54_54964


namespace valentine_floral_requirement_l54_54322

theorem valentine_floral_requirement:
  let nursing_home_roses := 90
  let nursing_home_tulips := 80
  let nursing_home_lilies := 100
  let shelter_roses := 120
  let shelter_tulips := 75
  let shelter_lilies := 95
  let maternity_ward_roses := 100
  let maternity_ward_tulips := 110
  let maternity_ward_lilies := 85
  let total_roses := nursing_home_roses + shelter_roses + maternity_ward_roses
  let total_tulips := nursing_home_tulips + shelter_tulips + maternity_ward_tulips
  let total_lilies := nursing_home_lilies + shelter_lilies + maternity_ward_lilies
  let total_flowers := total_roses + total_tulips + total_lilies
  total_roses = 310 ∧
  total_tulips = 265 ∧
  total_lilies = 280 ∧
  total_flowers = 855 :=
by
  sorry

end valentine_floral_requirement_l54_54322


namespace total_paintable_wall_area_l54_54927

def length : ℝ := 14
def width : ℝ := 11
def height : ℝ := 9
def unpaintable_area : ℝ := 50
def num_rooms : ℝ := 3

theorem total_paintable_wall_area : 3 * (2 * (length * height) + 2 * (width * height) - unpaintable_area) = 1200 := 
by
  sorry

end total_paintable_wall_area_l54_54927


namespace length_of_minor_axis_of_C1_l54_54247

variables (a b : ℝ) (C1 C2 : set (ℝ × ℝ))

-- Definitions of the ellipse and hyperbola
def ellipse (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def hyperbola (x y : ℝ) := x^2 - y^2 / 4 = 1

-- Condition for common focus and distance relation
def common_focus : Prop := (∃ f : ℝ, ∀ x y, f = real.sqrt 5) ∧ a^2 - b^2 = 5

-- Intersections of asymptote and circle
def intersects_asymptote_and_circle (x y : ℝ) : Prop :=
  y = 2 * x ∧ (x^2 / b^2 + 5 + y^2 / b^2 = 1)

-- Given problem in Lean statement
theorem length_of_minor_axis_of_C1 :
  common_focus ∧ (∀ M N, intersects_asymptote_and_circle M.1 M.2 ∧ intersects_asymptote_and_circle N.1 N.2 ∧ C1 M.1 M.2 ∧ C1 N.1 N.2 → 
    distance M N = 2 * √2) →
  2 * b = √2 :=
by {sorry}

end length_of_minor_axis_of_C1_l54_54247


namespace incenter_of_A1B1C_l54_54354

-- Definitions of points as Lean terms
variables (A B C A_1 B_1 L_3: Type) [Point A] [Point B] [Point C] [Point A_1] [Point B_1] [Point L_3]
variables (AC BC AL3 BL3 AA1 BB1 : ℝ)

-- Conditions
axioms
  (h1 : AL3 = AA1)                   -- AA1 = AL3
  (h2 : BL3 = BB1)                   -- BB1 = BL3
  (h3 : AC > 0)
  (h4 : BC > 0)
  (h5 : AL3 / BL3 = AC / BC)         -- AL3 / BL3 = AC / BC

-- Conjecture to be proven
theorem incenter_of_A1B1C : is_incenter L_3 A_1 B_1 C :=
sorry

end incenter_of_A1B1C_l54_54354


namespace intersection_of_P_and_Q_l54_54235

def P (x : ℝ) : Prop := 1 < x ∧ x < 4
def Q (x : ℝ) : Prop := 2 < x ∧ x < 3

theorem intersection_of_P_and_Q (x : ℝ) : P x ∧ Q x ↔ 2 < x ∧ x < 3 := by
  sorry

end intersection_of_P_and_Q_l54_54235


namespace fraction_married_men_l54_54445

-- Define the problem conditions
def num_faculty : ℕ := 100
def women_perc : ℕ := 60
def married_perc : ℕ := 60
def single_men_perc : ℚ := 3/4

-- We need to calculate the fraction of men who are married.
theorem fraction_married_men :
  (60 : ℚ) / 100 = women_perc / num_faculty →
  (60 : ℚ) / 100 = married_perc / num_faculty →
  (3/4 : ℚ) = single_men_perc →
  ∃ (fraction : ℚ), fraction = 1/4 :=
by
  intro h1 h2 h3
  sorry

end fraction_married_men_l54_54445


namespace math_proof_l54_54338

variables {A B C AM BM b c : ℝ}

-- Given initial conditions
def conditions 
  (triangle : Triangle) 
  (a b c : ℝ) 
  (angles : Angles A B C) 
  (cos_rule : 2 * b * Real.cos C = 2 * a + c) : Prop :=
  triangle.has_sides a b c ∧ triangle.has_angles A B C

-- Part 1: Proving angle B
def angle_B_proof 
  (triangle : Triangle) 
  (cos_rule : 2 * b * Real.cos C = 2 * a + c) 
  (B_value : B = 2 * π / 3) 
  (a b c A B C : ℝ) 
  (angles : Angles A B C) : Prop :=
  A + B + C = π ∧ cos_rule → B_value

-- Part 2: Proving the area of triangle BMC
def area_BMC_proof 
  (triangle: Triangle) 
  (b_value : b = 9) 
  (point_M : Point M on_AC triangle) 
  (AM_value : 2 * AM = triangle.side MC) 
  (equal_angles : Angles.equal (triangle.angle MAB) (triangle.angle MBA)) 
  (area_value : triangle.area BMC = 9 * Real.sqrt 3 / 2) : Prop :=
  b_value ∧ AM_value ∧ equal_angles → area_value

-- Aggregate the conditions and proofs
theorem math_proof 
  (triangle : Triangle) 
  (a b c A B C AM BM : ℝ) 
  (angles : Angles A B C)
  (cos_rule : 2 * b * Real.cos C = 2 * a + c) 
  (B_value : B = 2 * π / 3) 
  (b_value : b = 9) 
  (point_M : Point M on_AC triangle) 
  (AM_value : 2 * AM = triangle.side MC) 
  (equal_angles : Angles.equal (triangle.angle MAB) (triangle.angle MBA)) 
  (area_value : triangle.area BMC = 9 * Real.sqrt 3 / 2) : Prop :=
  sorry

end math_proof_l54_54338


namespace range_m_necessary_not_sufficient_l54_54722

def f (x : ℝ) : ℝ := Real.sqrt (2 + x) + Real.log (4 - x)

def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

def p (x : ℝ) : Prop := x ∈ A

def q (x : ℝ) : Prop := ∃ (m : ℝ), x ∈ B m

theorem range_m_necessary_not_sufficient (m : ℝ) : 
  ∀ x, (p x → q x) ∧ ¬ (q x → p x) ↔ m ∈ Set.Iio (5 / 2) :=
sorry

end range_m_necessary_not_sufficient_l54_54722


namespace vasya_grades_l54_54682

-- Given conditions
constants (a1 a2 a3 a4 a5 : ℕ)
axiom grade_median : a3 = 4
axiom grade_sum : a1 + a2 + a3 + a4 + a5 = 19
axiom most_A_grades : ∀ (n : ℕ), n ≠ 5 → (∃ m, m > 0 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 5 → (if a1 = n ∨ a2 = n ∨ a3 = n ∨ a4 = n ∨ a5 = n then m > 1 else m = 0))

-- Prove that the grades are (2, 3, 4, 5, 5)
theorem vasya_grades : (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨ 
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) := 
by sorry

end vasya_grades_l54_54682


namespace largest_distance_l54_54724

noncomputable def z (z : ℂ) : Prop := complex.abs z = 3

theorem largest_distance (z : ℂ) (h : z z) :
  complex.abs ((1 + 2 * complex.I) * z^3 - z^4) ≤ 216 :=
sorry

end largest_distance_l54_54724


namespace speaking_orders_l54_54103

-- Given conditions
def students : Set String := {"A", "B", "C", "D", "E", "F"}
def required_speaking_orders : Nat := 4
def named_students_included : Set String := {"A", "B"}

-- Statement to prove
theorem speaking_orders (x : students):
    students = 6 ∧ required_speaking_orders = 4 ∧ (named_students_included ⊆ x) → 
    ∑ (comb : Finset (Finset String)), comb.card = required_speaking_orders ∧ (named_students_included ⊆ comb) ∧ 
    x = comb.card → 
    comb.card = 336 := 
sorry

end speaking_orders_l54_54103


namespace initial_apples_value_l54_54699

-- Definitions for the conditions
def picked_apples : ℤ := 105
def total_apples : ℤ := 161

-- Statement to prove
theorem initial_apples_value : ∀ (initial_apples : ℤ), 
  initial_apples + picked_apples = total_apples → 
  initial_apples = total_apples - picked_apples := 
by 
  sorry

end initial_apples_value_l54_54699


namespace next_price_reduction_l54_54418

-- Definitions based on given conditions
def initial_price : ℝ := 1024
def price_after_1_year : ℝ := 640
def price_after_2_years : ℝ := 400
def price_last_week : ℝ := 250
def reduction_factor : ℝ := 5 / 8

-- Theorem to prove the next price reduction calculation
theorem next_price_reduction :
  price_last_week * reduction_factor = 156.25 := 
by
  sorry

end next_price_reduction_l54_54418


namespace cone_volume_to_surface_area_ratio_l54_54243

/-- Given that the lateral surface of a cone unfolds into a semicircle with a radius of 2,
prove that the ratio of the volume of the cone to the total surface area of the cone is √3/9. -/
theorem cone_volume_to_surface_area_ratio : 
  ∀ (r : ℝ) (h : ℝ),
  lateral_surface_semicircle_of_cone_unfolds radius 2 →
  ratio_of_volume_to_surface_area_of_cone = (ℝ.sqrt 3 / 9) :=
by
  sorry

end cone_volume_to_surface_area_ratio_l54_54243


namespace zero_of_polynomial_l54_54120

theorem zero_of_polynomial (P : Polynomial ℝ) (h_deg : natDegree P = 4) 
  (h_leading : leadingCoeff P = 1) (h_int_coeffs : ∀ x, coefficient P x ∈ ℤ) 
  (h_real_zeros : P.eval 3 = 0 ∧ P.eval (-1) = 0) : 
  P.eval (complex.mk (3 / 2) (sqrt 15 / 2)) = 0 :=
sorry

end zero_of_polynomial_l54_54120


namespace infinitely_many_singular_pairs_l54_54200

def largestPrimeFactor (n : ℕ) : ℕ := sorry -- definition of largest prime factor

def isSingularPair (p q : ℕ) : Prop :=
  p ≠ q ∧ ∀ (n : ℕ), n ≥ 2 → largestPrimeFactor n * largestPrimeFactor (n + 1) ≠ p * q

theorem infinitely_many_singular_pairs : ∃ (S : ℕ → (ℕ × ℕ)), ∀ i, isSingularPair (S i).1 (S i).2 :=
sorry

end infinitely_many_singular_pairs_l54_54200


namespace impossibleArrangement_l54_54306

-- Define the parameters of the table
def n : ℕ := 300

-- Define the properties of an arrangement.
def isValidArrangement (arr : ℕ × ℕ → ℤ) : Prop :=
  (∀ i j, arr (i, j) = 1 ∨ arr (i, j) = -1) ∧
  (|∑ i in finset.range n, ∑ j in finset.range n, arr (i, j)| < 30000) ∧
  (∀ i j, 
    (i ≤ n - 3 ∧ j ≤ n - 5 → |∑ a in finset.range 3, ∑ b in finset.range 5, arr (i + a, j + b)| > 3) ∧
    (i ≤ n - 5 ∧ j ≤ n - 3 → |∑ a in finset.range 5, ∑ b in finset.range 3, arr (i + a, j + b)| > 3))
  
-- Formalizing the problem in Lean.
theorem impossibleArrangement : ¬ ∃ arr : (ℕ × ℕ → ℤ), isValidArrangement arr :=
by
  sorry

end impossibleArrangement_l54_54306


namespace number_of_pencil_cartons_l54_54870

-- Define the conditions according to the question
def cost_of_pencil_carton : ℕ := 20
def number_of_marker_cartons : ℕ := 10
def cost_of_marker_carton : ℕ := 4
def total_spent : ℤ := 600

-- The Lean definition to prove
theorem number_of_pencil_cartons (P : ℕ) :
  P * cost_of_pencil_carton + number_of_marker_cartons * cost_of_marker_carton * 10 = total_spent
  → P = 10 :=
by
  intro h,
  sorry

end number_of_pencil_cartons_l54_54870


namespace kolya_time_segment_DE_l54_54302

-- Definitions representing the conditions
def time_petya_route : ℝ := 12  -- Petya takes 12 minutes
def time_kolya_route : ℝ := 12  -- Kolya also takes 12 minutes
def kolya_speed_factor : ℝ := 1.2

-- Proof problem: Prove that Kolya spends 1 minute traveling the segment D-E
theorem kolya_time_segment_DE 
    (v : ℝ)  -- Assume v is Petya's speed
    (time_petya_A_B_C : ℝ := time_petya_route)  
    (time_kolya_A_D_E_F_C : ℝ := time_kolya_route)
    (kolya_fast_factor : ℝ := kolya_speed_factor)
    : (time_petya_A_B_C / kolya_fast_factor - time_petya_A_B_C) / (2 / kolya_fast_factor) = 1 := 
by 
    sorry

end kolya_time_segment_DE_l54_54302


namespace second_solution_sugar_percentage_l54_54083

theorem second_solution_sugar_percentage
  (initial_solution_pct : ℝ)
  (second_solution_pct : ℝ)
  (initial_solution_amount : ℝ)
  (final_solution_pct : ℝ)
  (replaced_fraction : ℝ)
  (final_amount : ℝ) :
  initial_solution_pct = 0.1 →
  final_solution_pct = 0.17 →
  replaced_fraction = 1/4 →
  initial_solution_amount = 100 →
  final_amount = 100 →
  second_solution_pct = 0.38 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end second_solution_sugar_percentage_l54_54083


namespace tan_arithmetic_geometric_sequences_l54_54973

-- Definitions for sequences
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, b (n + 1) = b n * r

-- Main theorem statement 
theorem tan_arithmetic_geometric_sequences :
  ∀ (a b : ℕ → ℝ) (S : ℕ → ℝ),
    arithmetic_sequence a →
    sum_of_first_n_terms a S →
    S 11 = 22 * real.pi / 3 →
    geometric_sequence b →
    b 5 * b 7 = real.pi^2 / 4 →
    real.tan (a 6 + b 6) = real.sqrt 3 / 3 := 
by sorry

end tan_arithmetic_geometric_sequences_l54_54973


namespace expand_and_simplify_l54_54189

variable (x : ℝ)

theorem expand_and_simplify : (7 * x - 3) * 3 * x^2 = 21 * x^3 - 9 * x^2 := by
  sorry

end expand_and_simplify_l54_54189


namespace geometric_sequence_general_term_sum_of_b_n_l54_54583

noncomputable theory

variable {α : Type*} [linear_ordered_field α]

/-- General term formula for the geometric sequence. -/
theorem geometric_sequence_general_term (S_3 : α) (hS3 : S_3 = 42) (h : ∀ a2 a3 a6 a7 : α, 
16 * a2 * a6 = a3 * a7) :
  ∃ a b : α, ∀ n : ℕ, a_n = a * b^(n-1) := 
sorry

/-- Inequality for the sum T_n of sequence b_n. -/
theorem sum_of_b_n (h : ∀ n : ℕ, b_n = 1 / ((real.log 2 a_n) * (real.log 2 a_{n+1}))) :
  ∀ n : ℕ, (1 / 3 : α) ≤ T_n ∧ T_n < (1 / 2 : α) :=
sorry

end geometric_sequence_general_term_sum_of_b_n_l54_54583


namespace smallest_value_of_AC_is_six_l54_54158

noncomputable def find_smallest_AC : ℕ :=
  let AC := 6 in
  let BD := 6 in
  let CD := 6 in
  if h1 : (AC^2 - BD^2 = (AC - CD)^2) ∧ (36 = BD^2) then
    AC
  else
    0  -- Placeholder to satisfy Lean's need for an else

theorem smallest_value_of_AC_is_six : find_smallest_AC = 6 := 
by 
  sorry

end smallest_value_of_AC_is_six_l54_54158


namespace jean_vs_pauline_cost_l54_54357

-- Definitions based on the conditions given
def patty_cost (ida_cost : ℕ) : ℕ := ida_cost + 10
def ida_cost (jean_cost : ℕ) : ℕ := jean_cost + 30
def pauline_cost : ℕ := 30

noncomputable def total_cost (jean_cost : ℕ) : ℕ :=
jean_cost + ida_cost jean_cost + patty_cost (ida_cost jean_cost) + pauline_cost

-- Lean 4 statement to prove the required condition
theorem jean_vs_pauline_cost :
  ∃ (jean_cost : ℕ), total_cost jean_cost = 160 ∧ pauline_cost - jean_cost = 10 :=
by
  sorry

end jean_vs_pauline_cost_l54_54357


namespace max_intersection_points_l54_54070

-- Definitions for polynomials
noncomputable def p (x : ℝ) : ℝ := sorry  -- Assume exists
noncomputable def q (x : ℝ) : ℝ := sorry  -- Assume exists

-- Conditions on the degrees and leading coefficients
axiom p_deg : degree p = 5
axiom q_deg : degree q = 4
axiom p_leading : leading_coeff p = 1
axiom q_leading : leading_coeff q = 1

-- Proof statement
theorem max_intersection_points : ∃ x : ℝ, p(x) = q(x) → nat.card {x : ℝ | p x = q x} ≤ 5 :=
by
  sorry

end max_intersection_points_l54_54070


namespace find_f_one_l54_54981

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f_defined_for_neg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 2 * x^2 - 1

-- Statement that needs to be proven
theorem find_f_one (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : f_defined_for_neg f) :
  f 1 = -1 :=
  sorry

end find_f_one_l54_54981


namespace factorize_expression_l54_54190

variable {x y : ℝ}

theorem factorize_expression :
  3 * x^2 - 27 * y^2 = 3 * (x + 3 * y) * (x - 3 * y) :=
by
  sorry

end factorize_expression_l54_54190


namespace no_nat_triplet_exists_l54_54164

theorem no_nat_triplet_exists (x y z : ℕ) : ¬ (x ^ 2 + y ^ 2 = 7 * z ^ 2) := 
sorry

end no_nat_triplet_exists_l54_54164


namespace red_carpet_area_required_l54_54747

theorem red_carpet_area_required
  (length_in_feet : ℕ)
  (width_in_feet : ℕ)
  (feet_per_yard : ℕ)
  (length_in_feet = 12)
  (width_in_feet = 9)
  (feet_per_yard = 3) :
  (length_in_feet / feet_per_yard) * (width_in_feet / feet_per_yard) = 12 := 
sorry

end red_carpet_area_required_l54_54747


namespace root_in_interval_l54_54273

noncomputable def f (m x : ℝ) := m * 3^x - x + 3

theorem root_in_interval (m : ℝ) (h1 : m < 0) (h2 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f m x = 0) : -3 < m ∧ m < -2/3 :=
by
  sorry

end root_in_interval_l54_54273


namespace cyclist_distance_l54_54467

def time_in_minutes : ℝ := 2.5
def time_in_hours : ℝ := time_in_minutes / 60
def speed_km_per_hr : ℝ := 18
def distance_km : ℝ := speed_km_per_hr * time_in_hours
def distance_m : ℝ := distance_km * 1000

theorem cyclist_distance :
  distance_m = 750 := by
  sorry

end cyclist_distance_l54_54467


namespace InequalityProof_l54_54634

theorem InequalityProof (m n : ℝ) (h : m > n) : m / 4 > n / 4 :=
by sorry

end InequalityProof_l54_54634


namespace area_of_transformed_graph_l54_54023

noncomputable def transformation_area {α : Type*} [LinearOrder α]
  (x1 x2 x3 : α) (g : α → ℝ) 
  (area_original : ℝ) : ℝ :=
  4 * 3 * area_original  -- This directly computes the new area after transformation.

theorem area_of_transformed_graph 
  (x1 x2 x3 : ℝ) 
  (g : ℝ → ℝ)
  (h_domain : set α := {x1, x2, x3})
  (h_area : ∀ (a b c : ℝ), set.mem a h_domain → set.mem b h_domain → set.mem c h_domain → 
              triangle_area (a, g(a)) (b, g(b)) (c, g(c)) = 48) : 
  triangle_area (4 * x1, 3 * g(x1)) (4 * x2, 3 * g(x2)) (4 * x3, 3 * g(x3)) = 576 :=
  by sorry

end area_of_transformed_graph_l54_54023


namespace correct_statements_are_three_l54_54925

def contrapositive_correct: Prop := 
  (∀ x: ℝ, x^2 - 3 * x + 2 = 0 → x = 1) ↔ (∀ x: ℝ, x ≠ 1 → x^2 - 3 * x + 2 ≠ 0)

def necessary_not_sufficient: Prop :=
  ∀ a: ℝ, (a ≠ 0 → a^2 + a ≠ 0) ∧ ¬(a^2 + a ≠ 0 → a ≠ 0)

def false_implication: Prop :=
  ∀ {p q: Prop}, ¬(p ∧ q) → ¬p ∧ ¬q

def negation_correct: Prop :=
  (∃ x0: ℝ, x0^2 + x0 + 1 < 0) ↔ (∀ x: ℝ, x^2 + x + 1 ≥ 0)

def number_of_correct_statements: Nat :=
  (if contrapositive_correct then 1 else 0) +
  (if necessary_not_sufficient then 1 else 0) +
  (if false_implication then 0 else 1) + 
  (if negation_correct then 1 else 0)

theorem correct_statements_are_three: number_of_correct_statements = 3 :=
by { sorry }

end correct_statements_are_three_l54_54925


namespace exists_permutation_with_unique_real_root_l54_54327

open Polynomial

theorem exists_permutation_with_unique_real_root
  (a : ℕ → ℝ) (hpos : ∀ n, 0 < a n):
  ∃ (b : ℕ → ℝ), (∃ σ : Fin (2*n + 2) → Fin (2*n + 2), Perm σ) ∧
  (∃! x : ℝ, eval (∑ i in Finset.range (2 * n + 2), (b i) * X ^ i) x = 0) :=
sorry

end exists_permutation_with_unique_real_root_l54_54327


namespace jo_climbs_8_stairs_l54_54696

open Nat

def g : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| n => if (n % 2 = 0) then g (n - 1) + g (n - 2) else g (n - 1) + g (n - 2) + g (n - 3)

theorem jo_climbs_8_stairs : g 8 = 54 := by
  sorry

end jo_climbs_8_stairs_l54_54696


namespace max_number_of_phone_calls_l54_54010

theorem max_number_of_phone_calls (m : ℕ) (a : Fin m → ℕ) : 
  (odd m) ∧ (∀ i : Fin m, a (i + 1) % m = a i) ∧ (∃ n : ℕ, n ≤ 21 * 20 / 2) ∧ (∀ i j k : Fin 21, i ≠ j ∧ j ≠ k ∧ k ≠ i → ¬ (∀ x, x ∈ {i, j, k} → (x < i + 1))) → 
  ∃ n, n = 101 :=
sorry

end max_number_of_phone_calls_l54_54010


namespace max_rectangles_proof_l54_54960

noncomputable def max_rectangles (B : Type) [board: B is_rectangular] (R_partition : list B) : Prop :=
  ∃ R, R ∈ R_partition ∧ R.total = 35 ∧ (∀ P, P ⊆ B → P.no_overlap → P.no_gaps → P.cardinality ≤ 35)

theorem max_rectangles_proof (B : Type) [board: B is_rectangular] (R_partition : list B) :
  max_rectangles B R_partition := 
sorry

end max_rectangles_proof_l54_54960


namespace f_2008_eq_zero_l54_54230

noncomputable def f : ℝ → ℝ := sorry

-- f is odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- f satisfies f(x + 2) = -f(x)
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x

theorem f_2008_eq_zero : f 2008 = 0 :=
by
  sorry

end f_2008_eq_zero_l54_54230


namespace common_tangent_l54_54542

-- Definition of the ellipse and hyperbola
def ellipse (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144
def hyperbola (x y : ℝ) : Prop := 7 * x^2 - 32 * y^2 = 224

-- The statement to prove
theorem common_tangent :
  (∀ x y : ℝ, ellipse x y → hyperbola x y → ((x + y + 5 = 0) ∨ (x + y - 5 = 0) ∨ (x - y + 5 = 0) ∨ (x - y - 5 = 0))) := 
sorry

end common_tangent_l54_54542


namespace quadratic_inequality_solution_l54_54946

theorem quadratic_inequality_solution (x : ℝ) : 
  ((x : ℝ) ∈ set.Icc 16 24) ↔ ((x - 16) * (x - 24) ≤ 0) := 
by
  -- outline of proof steps, but the proof itself is not required
  sorry

end quadratic_inequality_solution_l54_54946


namespace series_diverges_l54_54305

theorem series_diverges :
  ¬Summable (λ n : ℕ, (3^n * Nat.factorial n) / (n^n)) :=
by
  sorry

end series_diverges_l54_54305


namespace vending_machine_users_l54_54405

theorem vending_machine_users (p_fail p_double p_single : ℚ) (total_snacks : ℕ) (P : ℕ) :
  p_fail = 1 / 6 ∧ p_double = 1 / 10 ∧ p_single = 1 - 1 / 6 - 1 / 10 ∧
  total_snacks = 28 →
  P = 30 :=
by
  intros h
  sorry

end vending_machine_users_l54_54405


namespace total_litter_pieces_l54_54090

-- Define the number of glass bottles and aluminum cans as constants.
def glass_bottles : ℕ := 10
def aluminum_cans : ℕ := 8

-- State the theorem that the sum of glass bottles and aluminum cans is 18.
theorem total_litter_pieces : glass_bottles + aluminum_cans = 18 := by
  sorry

end total_litter_pieces_l54_54090


namespace value_of_J_l54_54822

theorem value_of_J (J : ℕ) : 32^4 * 4^4 = 2^J → J = 28 :=
by
  intro h
  sorry

end value_of_J_l54_54822


namespace smallest_positive_period_l54_54613

def isCosFunction (f : ℝ → ℝ) (ω : ℝ) : Prop :=
  f = λ x, Real.cos (ω * x + (Real.pi / 3))

def isMonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≥ f y

def isMonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y

theorem smallest_positive_period (f : ℝ → ℝ) (ω : ℝ) 
  (h1 : ω > 0) 
  (h2 : isCosFunction f ω) 
  (h3 : isMonotonicallyDecreasing f 0 (4 * Real.pi / 3))
  (h4 : isMonotonicallyIncreasing f (4 * Real.pi / 3) (2 * Real.pi)) : 
  (∃ T : ℝ, T = 4 * Real.pi) :=
sorry

end smallest_positive_period_l54_54613


namespace range_g_l54_54518

def g (x : ℝ) : ℝ := x / (2 * x^2 - 3 * x + 4)

theorem range_g : set.range g = set.Icc (-1 : ℝ) (1 / 23) :=
by
  sorry

end range_g_l54_54518


namespace system_of_equations_solution_cases_l54_54456

theorem system_of_equations_solution_cases
  (x y a b : ℝ) :
  (a = b → x + y = 2 * a) ∧
  (a = -b → ¬ (∃ (x y : ℝ), (x / (x - a)) + (y / (y - b)) = 2 ∧ a * x + b * y = 2 * a * b)) :=
by
  sorry

end system_of_equations_solution_cases_l54_54456


namespace not_relatively_prime_in_27_of_99_l54_54359

theorem not_relatively_prime_in_27_of_99 :
  ∀ (S : Finset ℕ), S.card = 27 → (∀ x ∈ S, x < 100) → ∃ a b ∈ S, a ≠ b ∧ Nat.gcd a b > 1 :=
by 
  intro S hScard hSlt100
  sorry

end not_relatively_prime_in_27_of_99_l54_54359


namespace graph_shift_l54_54038

theorem graph_shift :
  ∀ (x : ℝ),
    (2 * sin(2 * (x - π / 12))) = (√3 * sin(2 * x) - cos(2 * x)) :=
by
  sorry

end graph_shift_l54_54038


namespace shift_equivalence_l54_54805

-- Define the base function
def base_function (x : ℝ) : ℝ := 3 * sin (2 * x)

-- Define the transformed function
def transformed_function (x : ℝ) : ℝ := 3 * sin (2 * x + π / 4)

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := base_function (x + π / 8)

-- Proposition stating the transformation is equivalent to the shift
theorem shift_equivalence : 
  ∀ x : ℝ, transformed_function x = shifted_function x :=
by
  sorry

end shift_equivalence_l54_54805


namespace seq_2008th_term_l54_54560

theorem seq_2008th_term (a : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → (∑ i in finset.range n, a (i + 1)) / n = n) : 
  a 2008 = 4015 :=
  by
  sorry

end seq_2008th_term_l54_54560


namespace find_a_l54_54548

-- Let p be a parameter related to the standard form of a parabola
def p : ℝ := -1

-- Define the parameter a in terms of p
def a : ℝ := 4 * p

-- Define the parabola with directrix x = 1
def parabola_directrix (x y : ℝ) : Prop :=
  y^2 = a * x ∧ x = 1

-- The main theorem to prove
theorem find_a :
  a = -4 :=
by
  unfold a
  unfold p
  calc
    4 * (-1) = -4 : by linarith

end find_a_l54_54548


namespace max_lg_sum_correct_l54_54571

noncomputable def max_lg_sum (x y : ℝ) : ℝ :=
  if h : x + y = 5 ∧ x > 0 ∧ y > 0 then
    let z := 2 * Real.log10 (5 / 2) in z
  else 0

theorem max_lg_sum_correct (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 5) :
  max_lg_sum x y = 2 * Real.log10 (5 / 2) :=
by
  sorry

end max_lg_sum_correct_l54_54571


namespace impossible_arrangement_l54_54320

theorem impossible_arrangement :
  ∀ (f : Fin 300 → Fin 300 → ℤ),
    (∀ i j, -1 ≤ f i j ∧ f i j ≤ 1) →
    (∀ i j, abs (∑ u : Fin 3, ∑ v : Fin 5, f (i + u) (j + v)) > 3) →
    abs (∑ i : Fin 300, ∑ j : Fin 300, f i j) < 30000 →
    false :=
by
  intros f h_bound h_rect h_sum
  sorry

end impossible_arrangement_l54_54320


namespace number_divisible_by_7_last_digits_l54_54743

theorem number_divisible_by_7_last_digits :
  ∀ d : ℕ, d ≤ 9 → ∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d :=
by
  sorry

end number_divisible_by_7_last_digits_l54_54743


namespace points_C_D_E_F_are_concyclic_l54_54329

noncomputable def circles_cyclic_condition (Γ₁ Γ₂ : Circle) (O₁ O₂ A B C D E F : Point) 
  (h1 : Γ₁.center = O₁)
  (h2 : Γ₂.center = O₂)
  (h3 : Γ₁ = circle_eq O₁.radius)
  (h4 : Γ₂ = circle_eq O₂.radius)
  (h5 : circle_intersect Γ₁ Γ₂ = {A, B})
  (h6 : is_obtuse_angle (O₁AO₂))
  (h7 : circ_circumcircle (triangle O₁ A O₂) = R)
  (h8 : R ∩ Γ₁ = {C, A} ∧ C ≠ A)
  (h9 : R ∩ Γ₂ = {D, A} ∧ D ≠ A)
  (h10 : line_intersection (line(C, B)) Γ₂ = E)
  (h11 : line_intersection (line(D, B)) Γ₁ = F)
  : Prop :=
  point_concyclic C D E F

theorem points_C_D_E_F_are_concyclic
  (Γ₁ Γ₂ : Circle) (O₁ O₂ A B C D E F : Point)
  (h1 : Γ₁.center = O₁)
  (h2 : Γ₂.center = O₂)
  (h3 : Γ₁ = circle_eq O₁.radius)
  (h4 : Γ₂ = circle_eq O₂.radius)
  (h5 : circle_intersect Γ₁ Γ₂ = {A, B})
  (h6 : is_obtuse_angle (O₁AO₂))
  (h7 : circ_circumcircle (triangle O₁ A O₂) = R)
  (h8 : R ∩ Γ₁ = {C, A} ∧ C ≠ A)
  (h9 : R ∩ Γ₂ = {D, A} ∧ D ≠ A)
  (h10 : line_intersection (line(C, B)) Γ₂ = E)
  (h11 : line_intersection (line(D, B)) Γ₁ = F) :
  circles_cyclic_condition Γ₁ Γ₂ O₁ O₂ A B C D E F h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 := 
sorry

end points_C_D_E_F_are_concyclic_l54_54329


namespace sum_of_products_equal_l54_54007

theorem sum_of_products_equal 
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)
  (h1 : a1 + a2 + a3 = b1 + b2 + b3)
  (h2 : b1 + b2 + b3 = c1 + c2 + c3)
  (h3 : c1 + c2 + c3 = a1 + b1 + c1)
  (h4 : a1 + b1 + c1 = a2 + b2 + c2)
  (h5 : a2 + b2 + c2 = a3 + b3 + c3) :
  a1 * b1 * c1 + a2 * b2 * c2 + a3 * b3 * c3 = a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 :=
by 
  sorry

end sum_of_products_equal_l54_54007


namespace concentration_in_thermos_l54_54362

def initial_coffee_volume : ℕ := 300
def pour_volume : ℕ := 200
def iterations : ℕ := 6

theorem concentration_in_thermos :
  let final_concentration : ℚ := (1 : ℚ) - (1 : ℚ) / 3 ^ iterations in
  final_concentration = 0.2 :=
by sorry

end concentration_in_thermos_l54_54362


namespace find_vector_at_t_neg3_l54_54116

def vector (t : ℝ) : ℝ × ℝ :=
  (if t = 1 then (2, 5) else
   if t = 4 then (8, -7) else
   sorry)

theorem find_vector_at_t_neg3 :
  let a := (0, 9)
  let d := (2, -4)
  vector (-3) = (a.1 + (-3) * d.1, a.2 + (-3) * d.2) :=
sorry

end find_vector_at_t_neg3_l54_54116


namespace intersection_A_B_l54_54589

def A : Set ℕ := {x | -1 < x ∧ x < 3}

def B : Set ℤ := {x | -2 ≤ x ∧ x < 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  -- Proof omitted
  sorry

end intersection_A_B_l54_54589


namespace vasya_grades_l54_54669

def proves_grades_are_five (grades : List ℕ) : Prop :=
  (grades.length = 5) ∧ (grades.sum = 19) ∧ (grades.sorted.nth 2 = some 4) ∧
  ((grades.count 5) > (grades.count n) for n in (grades.erase_all [5]))

theorem vasya_grades : exists (grades : List ℕ), proves_grades_are_five grades ∧ grades = [2, 3, 4, 5, 5] := 
by
  sorry

end vasya_grades_l54_54669


namespace triangle_perimeter_correct_triangle_area_correct_l54_54131

def side_lengths := (15 : ℝ, 10 : ℝ, 12 : ℝ)

def perimeter (a b c : ℝ) := a + b + c

def semi_perimeter (a b c : ℝ) := perimeter a b c / 2

def herons_formula (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def triangle_perimeter : ℝ := perimeter 15 10 12
noncomputable def triangle_area : ℝ := herons_formula 15 10 12

theorem triangle_perimeter_correct : triangle_perimeter = 37 := by
  sorry

theorem triangle_area_correct : triangle_area ≈ 58.6 := by
  sorry

end triangle_perimeter_correct_triangle_area_correct_l54_54131


namespace rainfall_in_july_l54_54050

-- Defining the rainfall amounts for each month
def march_rain : ℝ := 3.79
def april_rain : ℝ := 4.5
def may_rain : ℝ := 3.95
def june_rain : ℝ := 3.09
-- Define the average rainfall and the number of months
def average_rain : ℝ := 4
def total_months : ℕ := 5

-- Using the given conditions to prove the rainfall in July
theorem rainfall_in_july :
  let total_rain_march_to_june := march_rain + april_rain + may_rain + june_rain in
  let total_rain_march_to_july := average_rain * (total_months : ℝ) in
  (total_rain_march_to_july - total_rain_march_to_june) = 4.67 :=
by
  let total_rain_march_to_june := march_rain + april_rain + may_rain + june_rain in
  let total_rain_march_to_july := average_rain * (total_months : ℝ) in
  show (total_rain_march_to_july - total_rain_march_to_june) = 4.67 from sorry

end rainfall_in_july_l54_54050


namespace exponentiation_81_5_4_eq_243_l54_54178

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end exponentiation_81_5_4_eq_243_l54_54178


namespace company_storage_cost_l54_54848

theorem company_storage_cost
  (l w h : ℕ) (total_volume : ℕ) (cost_per_box : ℚ)
  (h1 : l = 15)
  (h2 : w = 12)
  (h3 : h = 10)
  (h4 : total_volume = 1080000)
  (h5 : cost_per_box = 0.2) :
  (total_volume / (l * w * h) * cost_per_box) = 120 := 
by {
  -- Define the volume of one box
  have box_volume : ℕ := l * w * h,
  -- Calculate the number of boxes
  have number_of_boxes : ℕ := total_volume / box_volume,
  -- Calculate the total cost
  have total_cost : ℚ := number_of_boxes * cost_per_box,
  -- Prove the statement
  have h_box_vol : box_volume = 1800 := by rw [h1, h2, h3]; norm_num,
  have h_num_boxes : number_of_boxes = 600 := by rw [← nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num)]; exact h4.symm ▸ h_box_vol,
  have h_total_cost : total_cost = 120 := by rw [← nat.cast_mul, h_num_boxes, ← h5]; norm_num,
  exact h_total_cost
  sorry
}

end company_storage_cost_l54_54848


namespace C_is_a_liar_l54_54655

def is_knight_or_liar (P : Prop) : Prop :=
P = true ∨ P = false

variable (A B C : Prop)

-- A, B and C can only be true (knight) or false (liar)
axiom a1 : is_knight_or_liar A
axiom a2 : is_knight_or_liar B
axiom a3 : is_knight_or_liar C

-- A says "B is a liar", meaning if A is a knight, B is a liar, and if A is a liar, B is a knight
axiom a4 : A = true → B = false
axiom a5 : A = false → B = true

-- B says "A and C are of the same type", meaning if B is a knight, A and C are of the same type, otherwise they are not
axiom a6 : B = true → (A = C)
axiom a7 : B = false → (A ≠ C)

-- Prove that C is a liar
theorem C_is_a_liar : C = false :=
by
  sorry

end C_is_a_liar_l54_54655


namespace find_a_l54_54265

theorem find_a 
  (a b c : ℚ) 
  (h1 : a + b = c) 
  (h2 : b + c + 2 * b = 11) 
  (h3 : c = 7) :
  a = 17 / 3 :=
by
  sorry

end find_a_l54_54265


namespace calculate_K_3_27_12_l54_54203

def K (x y z : ℝ) : ℝ := x / y + y / z + z / x

theorem calculate_K_3_27_12 : K 3 27 12 = 229 / 36 := by
  sorry

end calculate_K_3_27_12_l54_54203


namespace age_ratio_problem_l54_54417

def age_condition (s a : ℕ) : Prop :=
  s - 2 = 2 * (a - 2) ∧ s - 4 = 3 * (a - 4)

def future_ratio (s a x : ℕ) : Prop :=
  (s + x) * 2 = (a + x) * 3

theorem age_ratio_problem :
  ∃ s a x : ℕ, age_condition s a ∧ future_ratio s a x ∧ x = 2 :=
by
  sorry

end age_ratio_problem_l54_54417


namespace domain_of_shifted_function_l54_54242

theorem domain_of_shifted_function (f : ℝ → ℝ) (D : set ℝ) (hD : ∀ x, f x ≠ 0 → x ∈ D) :
  (D = set.Icc 0 4) → ∀ x ∈ set.Icc 1 3, f (x + 1) + f (x - 1) ≠ 0 :=
by
  intro h1
  intro x hx
  cases hx with hx1 hx2
  have h2 : x + 1 ∈ D := sorry
  have h3 : x - 1 ∈ D := sorry
  rw h1 at h2 h3
  exact sorry

end domain_of_shifted_function_l54_54242


namespace abs_eq_three_system1_system2_l54_54768

theorem abs_eq_three : ∀ x : ℝ, |x| = 3 ↔ x = 3 ∨ x = -3 := 
by sorry

theorem system1 : ∀ x y : ℝ, (y * (x - 1) = 0) ∧ (2 * x + 5 * y = 7) → 
(x = 7 / 2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) := 
by sorry

theorem system2 : ∀ x y : ℝ, (x * y - 2 * x - y + 2 = 0) ∧ (x + 6 * y = 3) ∧ (3 * x + y = 8) → 
(x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 2) := 
by sorry

end abs_eq_three_system1_system2_l54_54768


namespace simplify_polynomial_l54_54016

theorem simplify_polynomial :
  (3 * x ^ 4 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 10) + (7 * x ^ 5 - 3 * x ^ 4 + x ^ 3 - 7 * x ^ 2 + 2 * x - 2)
  = 7 * x ^ 5 - x ^ 3 - 2 * x ^ 2 - 6 * x + 8 :=
by sorry

end simplify_polynomial_l54_54016


namespace bakery_combinations_l54_54847

theorem bakery_combinations 
  (total_breads : ℕ) (bread_types : Finset ℕ) (purchases : Finset ℕ)
  (h_total : total_breads = 8)
  (h_bread_types : bread_types.card = 5)
  (h_purchases : purchases.card = 2) : 
  ∃ (combinations : ℕ), combinations = 70 := 
sorry

end bakery_combinations_l54_54847


namespace moderate_intensity_pushups_l54_54849

theorem moderate_intensity_pushups :
  let normal_heart_rate := 80
  let k := 7
  let y (x : ℕ) := 80 * (Real.log (Real.sqrt (x / 12)) + 1)
  let t (x : ℕ) := y x / normal_heart_rate
  let f (t : ℝ) := k * Real.exp t
  28 ≤ f (Real.log (Real.sqrt 3)) + 1 ∧ f (Real.log (Real.sqrt 3)) + 1 ≤ 34 :=
sorry

end moderate_intensity_pushups_l54_54849


namespace term_69_l54_54578

noncomputable def original_sequence : ℕ → ℕ := sorry -- given sequence a_n
def new_sequence (a : ℕ → ℕ) : ℕ → ℕ
| n := if (n % 4 = 0) then a (n / 4 + 1) else sorry 

theorem term_69 (a : ℕ → ℕ) (n : ℕ) (H : three_numbers_inserted a) : 
  (new_sequence a 69 = a 18) :=
sorry

end term_69_l54_54578


namespace pushups_percentage_l54_54525

def total_exercises : ℕ := 12 + 8 + 20

def percentage_pushups (total_ex: ℕ) : ℕ := (8 * 100) / total_ex

theorem pushups_percentage (h : total_exercises = 40) : percentage_pushups total_exercises = 20 :=
by
  sorry

end pushups_percentage_l54_54525


namespace eval_exp_l54_54172

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end eval_exp_l54_54172


namespace rounding_scenario_b_smaller_l54_54633

noncomputable def rounding_effects (a b c a' b' c' : ℕ) : Prop :=
(a' = a + 1) ∧ (b' = b + 1) ∧ (c' = c - 1) ∧
(a > 0) ∧ (b > 0) ∧ (c > 0) → 
(↑a' ^ 2 / ↑b' + (↑c') ^ 3 < ↑a ^ 2 / ↑b + (↑c) ^ 3)

theorem rounding_scenario_b_smaller (a b c a' b' c' : ℕ) (h : rounding_effects a b c a' b' c') : 
  ↑a' ^ 2 / ↑b' + (↑c') ^ 3 < ↑a ^ 2 / ↑b + (↑c) ^ 3 :=
begin
  sorry
end

end rounding_scenario_b_smaller_l54_54633


namespace partition_solution_l54_54857

noncomputable def partitions (a m n x : ℝ) : Prop :=
  a = x + n * (a - m * x)

theorem partition_solution (a m n : ℝ) (h : n * m < 1) :
  partitions a m n (a * (1 - n) / (1 - n * m)) :=
by
  sorry

end partition_solution_l54_54857


namespace tiles_crossed_in_rectangle_l54_54474

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

theorem tiles_crossed_in_rectangle (width length : ℕ) (h_width : width = 12) (h_length : length = 20) :
  (width + length - gcd width length) = 28 :=
by
  rw [h_width, h_length]
  have h_gcd : gcd 12 20 = 4 := by sorry
  rw [h_gcd]
  norm_num

end tiles_crossed_in_rectangle_l54_54474


namespace tangents_intersection_on_NH_l54_54289

-- Definitions based on the conditions of the problem
variables {A B C H M P X N J K : Point}
variables {Γ Γ1 Γ2 : Circle}

-- Given conditions as assumptions
axiom triangle_acute (h_ABC : triangle A B C) (h_acute : acute_triangle A B C)
axiom orthocenter_H (h_orthocenter : orthocenter A B C H)
axiom circumcircle_Γ (h_circumcircle : circumcircle A B C Γ)
axiom midpoint_M (h_midpointM : midpoint_segment B C M)
axiom midpoint_P (h_midpointP : midpoint_segment A H P)
axiom AM_meets_Γ_at_X (h_AM_X : line_segments_intersect (line_through A M) Γ X 2)
axiom N_on_BC_tangent_to_Γ (h_N_tangent : N_on_BC_and_tangent_to_Γ N X Γ)
axiom J_K_on_circle_MP (h_JK_MP : on_circle_diameter_MP M P J K)
axiom angle_relationships (h_angles : angle A J P = angle H N M)
axiom Γ1_pass_through (h_Γ1_points : passes_through Γ1 K H J)
axiom Γ2_pass_through (h_Γ2_points : passes_through Γ2 K M N)

-- Statement to prove
theorem tangents_intersection_on_NH : intersection_of_common_external_tangent Γ1 Γ2 ∈ line_through N H :=
sorry

end tangents_intersection_on_NH_l54_54289


namespace distance_between_C_and_D_l54_54859

-- Definitions from the problem conditions directly
def f (x : ℝ) : ℝ := 5 * x ^ 2 + 2 * x - 6
def line (y : ℝ) : ℝ := -2

-- The proof problem: Prove p - q = 16 given the derived results
theorem distance_between_C_and_D : 
  ∃ (p q : ℕ), (f (-2 + 2 * Real.sqrt 21 / 10) = -2) ∧ 
               (f (-2 - 2 * Real.sqrt 21 / 10) = -2) ∧ 
               (Nat.gcd p q = 1) ∧ 
               (2 * Real.sqrt 21 / 5 = Real.sqrt p / q) ∧ 
               (p - q = 16) :=
begin
  sorry
end

end distance_between_C_and_D_l54_54859


namespace slope_line_l2_l54_54606

theorem slope_line_l2 (l1 l2 : Type) [line l1] [line l2] : 
  (slope l1 = -real.sqrt 3) → (inclination l2 = (inclination l1) / 2) → 
  slope l2 = real.sqrt 3 := 
by 
  sorry

end slope_line_l2_l54_54606


namespace vasya_grades_l54_54671

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧
    (grades.filter (λ x, x = 5)).length > 2 ∧
    List.sorted (≤) grades ∧
    grades.nth 2 = some 4 ∧
    (grades.sum : ℚ) / 5 = 3.8 ∧
    grades = [2, 3, 4, 5, 5] := by
  sorry

end vasya_grades_l54_54671


namespace solve_inequality_system_l54_54366

theorem solve_inequality_system (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1)) →
  ((1 / 2) * x - 1 ≤ 7 - (3 / 2) * x) →
  (2 < x ∧ x ≤ 4) :=
by
  intro h1 h2
  sorry

end solve_inequality_system_l54_54366


namespace linear_regression_equation_l54_54099

theorem linear_regression_equation :
  ∑ i in Finset.range 8, xi i = 52 →
  ∑ i in Finset.range 8, yi i = 228 →
  ∑ i in Finset.range 8, (xi i)^2 = 478 →
  ∑ i in Finset.range 8, (xi i) * (yi i) = 1849 →
  let n := 8 in
  let x̄ := (∑ i in Finset.range 8, xi i) / n in
  let ȳ := (∑ i in Finset.range 8, yi i) / n in
  let b := (∑ i in Finset.range 8, (xi i) * (yi i) - n * x̄ * ȳ) / (∑ i in Finset.range 8, (xi i)^2 - n * x̄^2) in
  let a := ȳ - b * x̄ in
  a ≈ 11.47 ∧ b ≈ 2.62 ∧ (∀ x, yhat x = a + b * x) :=
begin
  intros h1 h2 h3 h4 n x̄ ȳ b a,
  sorry -- Proof to be filled in
end

end linear_regression_equation_l54_54099


namespace last_three_positions_l54_54770

theorem last_three_positions (n : ℕ) (h : n = 2009) :
  (initial_positions n) = [1, 2, 1600] :=
by
  sorry

end last_three_positions_l54_54770


namespace smallest_integer_divisible_20_perfect_cube_square_l54_54168

theorem smallest_integer_divisible_20_perfect_cube_square :
  ∃ (n : ℕ), n > 0 ∧ n % 20 = 0 ∧ (∃ (m : ℕ), n^2 = m^3) ∧ (∃ (k : ℕ), n^3 = k^2) ∧ n = 1000000 :=
by {
  sorry -- Replace this placeholder with an appropriate proof.
}

end smallest_integer_divisible_20_perfect_cube_square_l54_54168


namespace Maria_trip_time_l54_54352

/-- 
Given:
- Maria drove 80 miles on a freeway.
- Maria drove 20 miles on a rural road.
- Her speed on the rural road was half of her speed on the freeway.
- Maria spent 40 minutes driving on the rural road.

Prove that Maria's entire trip took 120 minutes.
-/ 
theorem Maria_trip_time
  (distance_freeway : ℕ)
  (distance_rural : ℕ)
  (rural_speed_ratio : ℕ → ℕ)
  (time_rural_minutes : ℕ) 
  (time_freeway : ℕ)
  (total_time : ℕ) 
  (speed_rural : ℕ)
  (speed_freeway : ℕ) 
  :
  distance_freeway = 80 ∧
  distance_rural = 20 ∧ 
  rural_speed_ratio (speed_freeway) = speed_rural ∧ 
  time_rural_minutes = 40 ∧
  time_rural_minutes = 20 / speed_rural ∧
  speed_freeway = 2 * speed_rural ∧
  time_freeway = distance_freeway / speed_freeway ∧
  total_time = time_rural_minutes + time_freeway → 
  total_time = 120 :=
by
  intros
  sorry

end Maria_trip_time_l54_54352


namespace simple_interest_fraction_l54_54789

theorem simple_interest_fraction (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (F : ℝ)
  (h1 : R = 5)
  (h2 : T = 4)
  (h3 : SI = (P * R * T) / 100)
  (h4 : SI = F * P) :
  F = 1/5 :=
by
  sorry

end simple_interest_fraction_l54_54789


namespace range_of_a_l54_54558

noncomputable def range_a : set ℝ :=
  {a | ∃ x : ℝ, x > 0 ∧ a - 2 * x - abs (Real.log x) ≤ 0}

theorem range_of_a :
  range_a = {a : ℝ | a ≤ 1 + Real.log 2} :=
by
  sorry

end range_of_a_l54_54558


namespace correct_operation_l54_54827

theorem correct_operation :
    (∀ (b x y a : ℝ) (n : ℕ), 
      (b^3 * b^3 = b^9 → false) ∧
      ((-x^3 * y) * (x * y^2) = x^4 * y^3 → false) ∧
      ((-2 * x^3)^2 = -4 * x^6 → false) ∧
      ((-a^(3 * n))^2 = a^(6 * n))) :=
by {
  intros b x y a n, 
  split,
  { intro h, have : b^3 * b^3 = b^(3+3), by ring_exp,
    rw this at h, contradiction },
  split,
  { intro h, have : (-x^3 * y) * (x * y^2) = -x^(3+1) * y^(1+2), by ring_exp,
    rw this at h, contradiction },
  split,
  { intro h, have : (-2 * x^3)^2 = 4 * x^(3*2), by ring_exp,
    rw this at h, contradiction },
  { intro h, have : (-a^(3 * n))^2 = a^(2*3 * n), by ring_exp,
    rw this, exact h }
}

end correct_operation_l54_54827


namespace annual_return_percentage_correct_l54_54899

variables (initial_value final_value gain : ℝ)
variables (initial_value_eq : initial_value = 8000)
variables (final_value_eq : final_value = initial_value + 400)
variables (gain_eq : gain = final_value - initial_value)
variables (annual_return_percentage : ℝ)

theorem annual_return_percentage_correct : 
  annual_return_percentage = (gain / initial_value * 100) :=
by
  rw [initial_value_eq, final_value_eq, gain_eq]
  have h : final_value = 8400 := by
    rw [initial_value_eq, final_value_eq]
    rw [initial_value_eq]
    sorry
  have h_gain : gain = 400 := by
    rw [gain_eq, h]
    sorry
  have h_percentage : annual_return_percentage = (400 / 8000 * 100) := by
    rw [←h_gain, initial_value_eq]
    sorry
  exact h_percentage

end annual_return_percentage_correct_l54_54899


namespace fraction_of_smart_int_div_by_25_l54_54922

def is_smart_int (n : ℤ) : Prop :=
  (n % 2 = 0) ∧ (50 < n) ∧ (n < 200) ∧ (n.digits.sum = 10)

def count_smart_int (f : ℤ → Prop) : ℕ :=
  finset.card {n | f n}

theorem fraction_of_smart_int_div_by_25 : 
  (count_smart_int (λ n, is_smart_int n ∧ n % 25 = 0)) / (count_smart_int is_smart_int) = 0 :=
sorry

end fraction_of_smart_int_div_by_25_l54_54922


namespace count_valid_numbers_l54_54685

-- Definitions based on given conditions
def digit_set : Set ℕ := {0, 2, 4, 6, 7, 8}
def fixed_digits_sum := 2 + 0 + 1 + 6 + 0 + 2

-- Statement of the problem
theorem count_valid_numbers : 
  let total_ways := 5 * (6 * 6 * 6 * 2) in
  total_ways = 2160 := 
sorry

end count_valid_numbers_l54_54685


namespace max_min_difference_l54_54041

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * sin x * cos x / (1 + sin x + cos x)

-- The interval of x
def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ π / 2

-- The statement of the problem
theorem max_min_difference : 
  ∀ (M N : ℝ), 
  (∀ x, domain x → f x ≤ M) ∧ 
  (∀ x, domain x → f x ≥ N) → 
  (∃ x1 x2, domain x1 ∧ f x1 = M ∧ domain x2 ∧ f x2 = N) → 
  M - N = sqrt 2 - 1 :=
by
  sorry

end max_min_difference_l54_54041


namespace teddy_bears_partition_l54_54011

/-
Problem Statement:
Our teddy bears are each in conflict with at most three others.

Prove that we can divide them into two groups such that each bear is in the same group with at most one bear it is in conflict with.
-/

noncomputable def can_divide_bears (bears : Finset ι) (conflict : ι → ι → Prop) : Prop :=
  ∀ b ∈ bears, (∃ (grouping : ι → bool),
    (∀ b ∈ bears, ∃ c ∈ bears, conflict b c → grouping b ≠ grouping c) ∧
    (∀ b ∈ bears, ∃ c ∈ bears, grouping b = grouping c → ¬ conflict b c))

theorem teddy_bears_partition (bears : Finset ι) (conflict : ι → ι → Prop)
  (h_conflict : ∀ b ∈ bears, Finset.card (bears.filter (conflict b)) ≤ 3) :
  can_divide_bears bears conflict :=
  sorry

end teddy_bears_partition_l54_54011


namespace problem_proof_l54_54624

def I : set char := {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
def M : set char := {'c', 'd', 'e'}
def N : set char := {'a', 'c', 'f'}
def CI (S : set char) : set char := I \ S
def B : set char := {'b', 'g'}

theorem problem_proof : B = (CI M ∩ CI N) := by
  sorry

end problem_proof_l54_54624


namespace range_of_a_l54_54214

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x, ∃ y, y = (3 : ℝ) * x^2 + 2 * a * x + (a + 6) ∧ (y = 0)) :
  (a < -3 ∨ a > 6) :=
by { sorry }

end range_of_a_l54_54214


namespace cubed_expression_value_l54_54339

open Real

theorem cubed_expression_value (a b c : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b + 2 * c = 0) :
  (a^3 + b^3 + 2 * c^3) / (a * b * c) = -3 * (a^2 - a * b + b^2) / (2 * a * b) :=
  sorry

end cubed_expression_value_l54_54339


namespace relationship_among_abc_l54_54568

variable (a b c : ℝ)

theorem relationship_among_abc (h₁ : a = Real.log 0.5 / Real.log 2) (h₂ : b = Real.sqrt 2) (h₃ : c = 0.5^2) : a < c ∧ c < b :=
by
  sorry

end relationship_among_abc_l54_54568


namespace grasshopper_jump_l54_54470

theorem grasshopper_jump :
  ∃ (x y : ℤ), 80 * x - 50 * y = 170 ∧ x + y ≤ 7 := by
  sorry

end grasshopper_jump_l54_54470


namespace impossibleArrangement_l54_54309

-- Define the parameters of the table
def n : ℕ := 300

-- Define the properties of an arrangement.
def isValidArrangement (arr : ℕ × ℕ → ℤ) : Prop :=
  (∀ i j, arr (i, j) = 1 ∨ arr (i, j) = -1) ∧
  (|∑ i in finset.range n, ∑ j in finset.range n, arr (i, j)| < 30000) ∧
  (∀ i j, 
    (i ≤ n - 3 ∧ j ≤ n - 5 → |∑ a in finset.range 3, ∑ b in finset.range 5, arr (i + a, j + b)| > 3) ∧
    (i ≤ n - 5 ∧ j ≤ n - 3 → |∑ a in finset.range 5, ∑ b in finset.range 3, arr (i + a, j + b)| > 3))
  
-- Formalizing the problem in Lean.
theorem impossibleArrangement : ¬ ∃ arr : (ℕ × ℕ → ℤ), isValidArrangement arr :=
by
  sorry

end impossibleArrangement_l54_54309


namespace intersection_of_A_and_B_l54_54587

open Set

def A := { x : ℕ | -1 < x ∧ x < 3 }
def B := { x : ℤ | -2 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end intersection_of_A_and_B_l54_54587


namespace vasya_grades_l54_54667

def proves_grades_are_five (grades : List ℕ) : Prop :=
  (grades.length = 5) ∧ (grades.sum = 19) ∧ (grades.sorted.nth 2 = some 4) ∧
  ((grades.count 5) > (grades.count n) for n in (grades.erase_all [5]))

theorem vasya_grades : exists (grades : List ℕ), proves_grades_are_five grades ∧ grades = [2, 3, 4, 5, 5] := 
by
  sorry

end vasya_grades_l54_54667


namespace circles_intersect_l54_54517

-- Definition of the first circle
def circleC := { p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 4 }

-- Definition of the second circle
def circleM := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 9 }

-- Prove that the circles intersect
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, p ∈ circleC ∧ p ∈ circleM := 
sorry

end circles_intersect_l54_54517


namespace find_m_values_l54_54998

def is_solution (m : ℝ) : Prop :=
  let A : Set ℝ := {1, -2}
  let B : Set ℝ := {x | m * x + 1 = 0}
  B ⊆ A

theorem find_m_values :
  {m : ℝ | is_solution m} = {0, -1, 1 / 2} :=
by
  sorry

end find_m_values_l54_54998


namespace proof_value_l54_54506

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 2*x + 1) / (x^3 - 2*x^2 - x + 2)

def holes (f : ℝ → ℝ) : ℕ := 1 -- hole at x = -1

def vertical_asymptotes (f : ℝ → ℝ) : ℕ := 2 -- VA at x = 1, 2

def horizontal_asymptotes (f : ℝ → ℝ) : ℕ := 1 -- HA at y = 0

def oblique_asymptotes (f : ℝ → ℝ) : ℕ := 0 -- no OA

def expression_value (a b c d : ℕ) : ℕ :=
  a + 2 * b + 3 * c + 4 * d

theorem proof_value :
  let a := holes f,
      b := vertical_asymptotes f,
      c := horizontal_asymptotes f,
      d := oblique_asymptotes f
  in expression_value a b c d = 8 := by
  let a := holes f
  let b := vertical_asymptotes f
  let c := horizontal_asymptotes f
  let d := oblique_asymptotes f
  have ha : a = 1 := rfl
  have hb : b = 2 := rfl
  have hc : c = 1 := rfl
  have hd : d = 0 := rfl
  rw [ha, hb, hc, hd]
  dsimp [expression_value]
  rfl

end proof_value_l54_54506


namespace total_weight_l54_54882

variable (a b c d : ℝ)

-- Conditions
axiom h1 : a + b = 250
axiom h2 : b + c = 235
axiom h3 : c + d = 260
axiom h4 : a + d = 275

-- Proving the total weight
theorem total_weight : a + b + c + d = 510 := by
  sorry

end total_weight_l54_54882


namespace functional_relationship_l54_54036

-- Define the conditions and question for Scenario ①
def scenario1 (x y k : ℝ) (h1 : k ≠ 0) : Prop :=
  y = k / x

-- Define the conditions and question for Scenario ②
def scenario2 (n S k : ℝ) (h2 : k ≠ 0) : Prop :=
  S = k / n

-- Define the conditions and question for Scenario ③
def scenario3 (t s k : ℝ) (h3 : k ≠ 0) : Prop :=
  s = k * t

-- The main theorem
theorem functional_relationship (x y n S t s k : ℝ) (h1 : k ≠ 0) :
  (scenario1 x y k h1) ∧ (scenario2 n S k h1) ∧ ¬(scenario3 t s k h1) := 
sorry

end functional_relationship_l54_54036


namespace ratio_is_correct_l54_54000

-- Define the side lengths of the triangles
def large_triangle_side : ℝ := 12
def small_triangle_side : ℝ := 3

-- Function to calculate the area of an equilateral triangle given its side length
def equilateral_triangle_area (s : ℝ) : ℝ := (√3 / 4) * s^2

-- Areas of the large and small triangles
def large_triangle_area : ℝ := equilateral_triangle_area large_triangle_side
def small_triangle_area : ℝ := equilateral_triangle_area small_triangle_side

-- Area of the remaining polygon
def remaining_polygon_area : ℝ := large_triangle_area - small_triangle_area

-- The desired ratio
def area_ratio : ℝ := small_triangle_area / remaining_polygon_area

-- The theorem to prove the ratio is 1/15
theorem ratio_is_correct : area_ratio = 1 / 15 := by
  sorry

end ratio_is_correct_l54_54000


namespace Z_divisible_by_10001_l54_54336

def is_eight_digit_integer (Z : Nat) : Prop :=
  (10^7 ≤ Z) ∧ (Z < 10^8)

def first_four_equal_last_four (Z : Nat) : Prop :=
  ∃ (a b c d : Nat), a ≠ 0 ∧ (Z = 1001 * (1000 * a + 100 * b + 10 * c + d))

theorem Z_divisible_by_10001 (Z : Nat) (h1 : is_eight_digit_integer Z) (h2 : first_four_equal_last_four Z) : 
  10001 ∣ Z :=
sorry

end Z_divisible_by_10001_l54_54336


namespace find_value_f5_fdd5_l54_54616

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (f'' : ℝ → ℝ)

-- The tangent condition at point (5, f(5))
def tangent_condition : Prop :=
    ∃ t : ℝ, (∀ x, f' x = - 1) ∧ (∀ x, t = - x + 8)

-- The main theorem to be proved
theorem find_value_f5_fdd5 (h1 : tangent_condition f f' f'') : f 5 + f'' 5 = 2 :=
by
  sorry

end find_value_f5_fdd5_l54_54616


namespace proof_problem_l54_54113

-- Definitions
def line (P : Type) := P → P → Prop   -- Assuming type P represents points in space
def plane (P : Type) := P → Prop      -- Assuming a plane is defined by a predicate over points

variables {P : Type} (l : line P) (m : line P) (α : plane P)

-- Conditions
def contained_in_plane (l : line P) (α : plane P) := ∀ (p1 p2 : P), l p1 p2 → α p1 ∧ α p2
def not_contained_in_plane (m : line P) (α : plane P) := ∃ (p1 p2 : P), m p1 p2 ∧ (¬ α p1 ∨ ¬ α p2)
def perpendicular_to_plane (m : line P) (α : plane P) := ∀ (p : P), α p → ∀ (q : P), m p q → q ≠ p
def perpendicular_to_line (m : line P) (l : line P) := ∀ (p1 p2 p3 : P), l p1 p2 → m p1 p3 → ¬ (p1 = p3)

-- Statement p
def p_statement (m : line P) (α : plane P) (l : line P) :=
  perpendicular_to_plane m α → perpendicular_to_line m l

-- Mathematically equivalent proof problem
theorem proof_problem
  (h1 : contained_in_plane l α)
  (h2 : not_contained_in_plane m α)
  (h3 : p_statement m α l) :
  let contrapositive := (∀ (p1 p2 p3 : P), (m p1 p3 → (∀ p4, α p4 → ¬ (p4 = p1))) → ¬ l p1 p2)
  let inverse := (∃ (p : P), ¬perpendicular_to_plane m α ∧ perpendicular_to_line m l)
  let converse := (∃ (p : P), perpendicular_to_line m l ∧ perpendicular_to_plane m α)
  in contrapositive ∧ ¬inverse ∧ ¬converse :=
sorry

end proof_problem_l54_54113


namespace range_of_linear_function_l54_54021

theorem range_of_linear_function (c d : ℝ) (h : 0 < c) :
  (set.range (λ x, c * x + d)) = set.Icc (-c + d) (2 * c + d) :=
by
  sorry

end range_of_linear_function_l54_54021


namespace annual_return_percentage_correct_l54_54898

variables (initial_value final_value gain : ℝ)
variables (initial_value_eq : initial_value = 8000)
variables (final_value_eq : final_value = initial_value + 400)
variables (gain_eq : gain = final_value - initial_value)
variables (annual_return_percentage : ℝ)

theorem annual_return_percentage_correct : 
  annual_return_percentage = (gain / initial_value * 100) :=
by
  rw [initial_value_eq, final_value_eq, gain_eq]
  have h : final_value = 8400 := by
    rw [initial_value_eq, final_value_eq]
    rw [initial_value_eq]
    sorry
  have h_gain : gain = 400 := by
    rw [gain_eq, h]
    sorry
  have h_percentage : annual_return_percentage = (400 / 8000 * 100) := by
    rw [←h_gain, initial_value_eq]
    sorry
  exact h_percentage

end annual_return_percentage_correct_l54_54898


namespace positive_diff_after_add_five_l54_54029

theorem positive_diff_after_add_five (y : ℝ) 
  (h : (45 + y)/2 = 32) : |45 - (y + 5)| = 21 :=
by 
  sorry

end positive_diff_after_add_five_l54_54029


namespace quadrilateral_centroid_area_ratio_l54_54334

-- Given conditions
variables {P Q R S H_P H_Q H_R H_S : Type}
variables [AddCommGroup P] [Module ℝ P] [AffineSpace P ℝ]
variables [AddCommGroup Q] [Module ℝ Q] [AffineSpace Q ℝ]
variables [AddCommGroup R] [Module ℝ R] [AffineSpace R ℝ]
variables [AddCommGroup S] [Module ℝ S] [AffineSpace S ℝ]

-- P, Q, R, S are points
variables (P Q R S : P)

-- Let H_P, H_Q, H_R, H_S be the centroids of triangles QRS, PRS, PQS, PQR respectively
def centroid (A B C : P) : P := (1 / 3 : ℝ) • (A + B + C)

def H_P := centroid Q R S
def H_Q := centroid P R S
def H_R := centroid P Q S
def H_S := centroid P Q R

-- Define areas of quadrilaterals
def area (A B C D : P) : ℝ := sorry  -- Definition of area here

-- The proof statement
theorem quadrilateral_centroid_area_ratio :
  area H_P H_Q H_R H_S / area P Q R S = 1 / 9 :=
sorry

end quadrilateral_centroid_area_ratio_l54_54334


namespace ellipse_eccentricity_min_value_l54_54248

theorem ellipse_eccentricity_min_value (m: ℝ) (a : ℝ) (c : ℝ) (e : ℝ) :
  (∃ E F1 F2 : (ℝ × ℝ),
    ∃ (x y : ℝ),
    (0 < m) ∧
    (x, y) ∈ { p : (ℝ × ℝ) | ((p.1 ^ 2) / (m + 1) + p.2 ^ 2 = 1) } ∧
    y = x + 2 ∧
    (F1, F2) ∈ { f : ((ℝ × ℝ) × (ℝ × ℝ)) | ∀ q : (ℝ × ℝ),
      |q.fst - f.fst.fst| + |q.snd - f.fst.snd| = √(m + 1)} ∧
    (by
      let Δ := 16 * (m + 1) ^ 2 - 12 * (m + 2) * (m + 1) in
      4 * (m + 1) * (m - 2) ≥ 0) ∧
    m = 2 ∧
    a = √3 ∧ 
    c = √(a * a - 1) ∧
    e = c / a) →
    e = √6 / 3 := 
sorry

end ellipse_eccentricity_min_value_l54_54248


namespace find_gamma_delta_l54_54059

theorem find_gamma_delta (γ δ : ℝ) (h : ∀ x : ℝ, (x - γ) / (x + δ) = (x^2 - 90 * x + 1980) / (x^2 + 60 * x - 3240)) : 
  γ + δ = 140 :=
sorry

end find_gamma_delta_l54_54059


namespace vasya_grades_l54_54663

variables
  (grades : List ℕ)
  (length_grades : grades.length = 5)
  (median_grade : grades.nthLe 2 sorry = 4)  -- Assuming 0-based indexing
  (mean_grade : (grades.sum : ℚ) / 5 = 3.8)
  (most_frequent_A : ∀ n ∈ grades, n ≤ 5)

theorem vasya_grades (h : ∀ x ∈ grades, x ≤ 5 ∧ ∃ k, grades.nthLe 3 sorry = 5 ∧ grades.count 5 > grades.count x):
  ∃ g1 g2 g3 g4 g5 : ℕ, grades = [g1, g2, g3, g4, g5] ∧ g1 ≤ g2 ∧ g2 ≤ g3 ∧ g3 ≤ g4 ∧ g4 ≤ g5 ∧ [g1, g2, g3, g4, g5] = [2, 3, 4, 5, 5] :=
sorry

end vasya_grades_l54_54663


namespace arithmetic_geometric_sequence_solution_l54_54241

theorem arithmetic_geometric_sequence_solution 
  (a1 a2 b1 b2 b3 : ℝ) 
  (h1 : -2 * 2 + a2 = a1)
  (h2 : a1 * 2 - 8 = a2)
  (h3 : b2 ^ 2 = -2 * -8)
  (h4 : b2 = -4) :
  (a2 - a1) / b2 = 1 / 2 :=
by 
  sorry

end arithmetic_geometric_sequence_solution_l54_54241


namespace impossible_arrangement_l54_54319

theorem impossible_arrangement :
  ∀ (f : Fin 300 → Fin 300 → ℤ),
    (∀ i j, -1 ≤ f i j ∧ f i j ≤ 1) →
    (∀ i j, abs (∑ u : Fin 3, ∑ v : Fin 5, f (i + u) (j + v)) > 3) →
    abs (∑ i : Fin 300, ∑ j : Fin 300, f i j) < 30000 →
    false :=
by
  intros f h_bound h_rect h_sum
  sorry

end impossible_arrangement_l54_54319


namespace transformation_parameters_l54_54037

def f (x : ℝ) : ℝ :=
if -3 ≤ x ∧ x < -1 then 2 * x
else if -1 ≤ x ∧ x ≤ 1 then - x ^ 2 + 2
else if 1 < x ∧ x ≤ 3 then x - 1
else 0 -- This handles any x outside the given intervals

def h (x p q r : ℝ) : ℝ := p * f (q * x) + r


theorem transformation_parameters : 
  ∃ p q r : ℝ, 
  h(x, p, q, r) = 2 * f (1 / 3 * x) - 5 :=
  by
    exists (2 : ℝ), (1 / 3 : ℝ), (-5 : ℝ)
    sorry

end transformation_parameters_l54_54037


namespace findMultipleOfSamsMoney_l54_54497

-- Define the conditions specified in the problem
def SamMoney : ℕ := 75
def TotalMoney : ℕ := 200
def BillyHasLess (x : ℕ) : ℕ := x * SamMoney - 25

-- State the theorem to prove
theorem findMultipleOfSamsMoney (x : ℕ) 
  (h1 : SamMoney + BillyHasLess x = TotalMoney) : x = 2 :=
by
  -- Placeholder for the proof
  sorry

end findMultipleOfSamsMoney_l54_54497


namespace at_least_triplets_l54_54345

theorem at_least_triplets (m n : ℕ) (S : set (ℕ × ℕ)) 
  (hS1 : ∀ (a b : ℕ), (a, b) ∈ S → 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ a ≠ b)
  (hS2 : S.card = m) : 
  ∃ N : ℕ, N ≥ (4 * m / (3 * n)) * (m - (n^2 / 4)) ∧ 
  ∀ (a b c : ℕ), (a, b) ∈ S → (b, c) ∈ S → (c, a) ∈ S → (a, b, c) ∈ (S.triples) :=
sorry

end at_least_triplets_l54_54345


namespace XM_eq_XA_l54_54328

noncomputable def acute_triangle (A B C : Type) := 
  ∃ (ABC : Triangle), 
  (Triangle.isAcute ABC) ∧ (A ∈ ABC.vertices) ∧ (B ∈ ABC.vertices) ∧ (C ∈ ABC.vertices)

noncomputable def point_inside {A B C X : Type} (ABC : Triangle A B C) := 
  ABC.contains X ∧ ¬ABC.onBoundary X

noncomputable def is_midpoint_of_arc {A B C M : Type} (ABC : Circle A) := 
  midpoint M (arcContaining A (arcNotContaining B C))

noncomputable def angles_condition_met {A B C X : Type} := 
  (angle B A X = 2 * angle X B A) ∧ (angle X A C = 2 * angle A C X)

theorem XM_eq_XA
  {A B C M X : Type}
  (ABC : acute_triangle A B C)
  (X_inside : point_inside ABC X)
  (angle_cond : angles_condition_met A B C X)
  (M_midpoint : is_midpoint_of_arc A B C M) :
  distance X M = distance X A :=
by
  sorry

end XM_eq_XA_l54_54328


namespace k_range_l54_54252

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -x^3 + 2*x^2 - x
  else if 1 ≤ x then Real.log x
  else 0 -- Technically, we don't care outside (0, +∞), so this else case doesn't matter.

theorem k_range (k : ℝ) :
  (∀ t : ℝ, 0 < t → f t < k * t) ↔ k ∈ (Set.Ioi (1 / Real.exp 1)) :=
by
  sorry

end k_range_l54_54252


namespace impossible_arrangement_of_1_and_neg1_in_300_by_300_table_l54_54310

theorem impossible_arrangement_of_1_and_neg1_in_300_by_300_table :
  ¬∃ (table : ℕ → ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ 300 ∧ 1 ≤ j ∧ j ≤ 300 → table i j = 1 ∨ table i j = -1) ∧
    abs (∑ i in finset.range 300, ∑ j in finset.range 300, table i j) < 30000 ∧
    ∀ i j, (1 ≤ i ∧ i ≤ 298) ∧ (1 ≤ j ∧ j ≤ 295) →
      abs (∑ di in finset.range 3, ∑ dj in finset.range 5, table (i + di) (j + dj)) > 3 ∧
    ∀ i j, (1 ≤ i ∧ i ≤ 296) ∧ (1 ≤ j ∧ j ≤ 298) →
      abs (∑ di in finset.range 5, ∑ dj in finset.range 3, table (i + di) (j + dj)) > 3 := sorry

end impossible_arrangement_of_1_and_neg1_in_300_by_300_table_l54_54310


namespace circle_equation_correct_l54_54434

theorem circle_equation_correct : 
  ∀ (x y : ℝ), (-3, 4) = (-3, 4) ∧ real.sqrt 3 > 0 → 
  (bind x y, (x- -3)^2 + (y-4)^2 = 3) do
  sorry

end circle_equation_correct_l54_54434


namespace speed_of_second_part_l54_54107

-- Definitions based on conditions
def total_distance : ℕ := 50
def distance_first_part : ℕ := 25
def speed_first_part : ℕ := 60
def average_speed : ℕ := 40

-- The problem statement we need to prove
theorem speed_of_second_part :
  (total_distance = 50) →
  (distance_first_part = 25) →
  (speed_first_part = 60) →
  (average_speed = 40) →
  (let total_time := total_distance / average_speed in
   let time_first_part := distance_first_part / speed_first_part in
   let time_second_part := total_time - time_first_part in
   let distance_second_part := total_distance - distance_first_part in
   distance_second_part / time_second_part = 30) :=
by intros; sorry

end speed_of_second_part_l54_54107


namespace solve_inequality_l54_54020

theorem solve_inequality (x : ℝ) : 
  3 * (2 * x - 1) - 2 * (x + 1) ≤ 1 → x ≤ 3 / 2 :=
by
  sorry

end solve_inequality_l54_54020


namespace six_digit_repeat_divisible_by_10101_l54_54874

theorem six_digit_repeat_divisible_by_10101:
  ∀ (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9),
  let n := 10000 * (10 * a + b) + 100 * (10 * a + b) + (10 * a + b) in
  10101 ∣ n :=
by { sorry }

end six_digit_repeat_divisible_by_10101_l54_54874


namespace students_taking_german_l54_54649

theorem students_taking_german (total_students french_students both_students neither_students : ℕ) 
                                (h_total : total_students = 78) 
                                (h_french : french_students = 41)
                                (h_both : both_students = 9) 
                                (h_neither : neither_students = 24) : 
                                ∃ german_students : ℕ, german_students = 22 :=
by
  let total_taking_french_or_german := total_students - neither_students
  let only_french_students := french_students - both_students
  let only_german_students := total_taking_french_or_german - only_french_students - both_students
  exists only_german_students + both_students
  have h1 : only_german_students = 13 := by
    have h_taking_fg : total_taking_french_or_german = 54 := by
      rw [h_total, h_neither]
      exact Nat.sub_eq_of_eq_add'
    exact calc
      only_german_students = total_taking_french_or_german - only_french_students - both_students := rfl
      ... = 54 - (french_students - both_students) - both_students := by rw [h_french, h_both, h_taking_fg]
      ... = 54 - (41 - 9) - 9 := rfl
      ... = 54 - 32 - 9 := rfl
      ... = 13 := Nat.sub_sub 54 32 9
  have h2 : only_german_students + both_students = 22 := by
    rw [h1, h_both]
    exact Nat.add_comm 13 9
  exact h2

end students_taking_german_l54_54649


namespace commutativity_associativity_l54_54647

variables {α : Type*} (op : α → α → α)

-- Define conditions as hypotheses
axiom cond1 : ∀ a b c : α, op a (op b c) = op b (op c a)
axiom cond2 : ∀ a b c : α, op a b = op a c → b = c
axiom cond3 : ∀ a b c : α, op a c = op b c → a = b

-- Commutativity statement
theorem commutativity (a b : α) : op a b = op b a := sorry

-- Associativity statement
theorem associativity (a b c : α) : op (op a b) c = op a (op b c) := sorry

end commutativity_associativity_l54_54647


namespace factorize_expression_l54_54936

theorem factorize_expression (a b : ℝ) : b^2 - ab + a - b = (b - 1) * (b - a) :=
by
  sorry

end factorize_expression_l54_54936


namespace min_value_of_x2_y2_z2_l54_54730

noncomputable def min_square_sum (x y z k : ℝ) : ℝ :=
  x^2 + y^2 + z^2

theorem min_value_of_x2_y2_z2 (x y z k : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = k) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ (x y z k : ℝ), (x^3 + y^3 + z^3 - 3 * x * y * z = k ∧ k ≥ -1) -> min_square_sum x y z k ≥ min_val :=
by
  sorry

end min_value_of_x2_y2_z2_l54_54730


namespace max_well_fed_pikes_l54_54095

theorem max_well_fed_pikes (n : ℕ) (h₀ : n = 40) (h₁ : ∀ x : ℕ, x is_well_fed ↔ ∃ y, y = 3 * x) : 
  ∃ m : ℕ, m = 13 :=
by {
  sorry
}

end max_well_fed_pikes_l54_54095


namespace theta_range_l54_54603

theorem theta_range (θ : ℝ) : ( ∀ x ∈ Icc 0 1, x^2 * cos θ - x * (1 - x) + (1 - x)^2 * sin θ > 0 ) ↔
  ( π / 12 < θ ∧ θ < 5 * π / 12 ) := sorry

end theta_range_l54_54603


namespace exists_unique_x1_l54_54201

noncomputable def seq (x : ℝ) : ℕ → ℝ
| 0     := x
| (n+1) := seq n * (seq n + (1/(n : ℝ + 1)))

theorem exists_unique_x1 : 
  ∃! x1 : ℝ, ∀ n : ℕ, 
    let x := seq x1
    in 0 < x n ∧ x n < x (n + 1) ∧ x (n + 1) < 1 := sorry

end exists_unique_x1_l54_54201


namespace average_speed_entire_journey_l54_54451

-- Define the average speed for the journey from x to y
def speed_xy := 60

-- Define the average speed for the journey from y to x
def speed_yx := 30

-- Definition for the distance (D) (it's an abstract value, so we don't need to specify)
variable (D : ℝ) (hD : D > 0)

-- Theorem stating that the average speed for the entire journey is 40 km/hr
theorem average_speed_entire_journey : 
  2 * D / ((D / speed_xy) + (D / speed_yx)) = 40 := 
by 
  sorry

end average_speed_entire_journey_l54_54451


namespace digit_frequency_difference_l54_54287

theorem digit_frequency_difference (n : ℕ) (h : (n = 2019)) :
  let count_digit (d : ℕ) (k : ℕ) := ((list.range k).map (λ x, (x.digits 10).count d)).sum in
  (count_digit 1 2019) - (count_digit 2 2019) = 999 :=
by
  -- count digits function
  let count_digit := fun (d : ℕ) (k : ℕ) => ((list.range k).map (fun x => (x.digits 10).count d)).sum in
  have h_n : n = 2019, from h,
  show (count_digit 1 n) - (count_digit 2 n) = 999, from sorry

end digit_frequency_difference_l54_54287


namespace percentage_decrease_of_y_l54_54643

variable (x y z : ℝ)

def condition1 : Prop := x = 1.20 * y
def condition2 : Prop := x = 0.48 * z
def percentage_decrease (a b : ℝ) : ℝ := (b - a) / b * 100

theorem percentage_decrease_of_y (h1 : condition1 x y) (h2 : condition2 x z) : 
  percentage_decrease y z = 60 :=
sorry

end percentage_decrease_of_y_l54_54643


namespace quadratic_two_distinct_real_roots_l54_54276

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 - 6 * x + k = 0) ↔ k < 9 :=
by
  sorry

end quadratic_two_distinct_real_roots_l54_54276


namespace min_cuts_for_30_sided_polygons_l54_54124

theorem min_cuts_for_30_sided_polygons (n : ℕ) (h : n = 73) : 
  ∃ k : ℕ, (∀ m : ℕ, m < k → (m + 1) ≤ 2 * m - 1972) ∧ (k = 1970) :=
sorry

end min_cuts_for_30_sided_polygons_l54_54124


namespace greatest_roses_purchased_l54_54085

def cost_individual_rose : ℝ := 7.30
def cost_dozen_roses : ℝ := 36
def cost_two_dozen_roses : ℝ := 50
def total_money : ℝ := 680
def roses_per_dozen : ℕ := 12
def roses_per_two_dozen : ℕ := 24

theorem greatest_roses_purchased (n : ℕ) (h₁ : n = ⌊total_money / cost_two_dozen_roses⌋)
                                  (h₂ : total_money - n * cost_two_dozen_roses < cost_two_dozen_roses) :
  let remaining_money := total_money - n * cost_two_dozen_roses in
  let remaining_roses := ⌊remaining_money / cost_individual_rose⌋ in
  n * roses_per_two_dozen + remaining_roses = 316 :=
by
  sorry

end greatest_roses_purchased_l54_54085


namespace triangle_angles_l54_54787

theorem triangle_angles (r_a r_b r_c R : ℝ)
    (h1 : r_a + r_b = 3 * R)
    (h2 : r_b + r_c = 2 * R) :
    ∃ (A B C : ℝ), A = 30 ∧ B = 60 ∧ C = 90 :=
sorry

end triangle_angles_l54_54787


namespace toys_divisible_by_41_l54_54040

theorem toys_divisible_by_41 (T : ℕ) :
  (∃ (n : ℤ), T = 41 * n) :=
begin
  sorry,
end

end toys_divisible_by_41_l54_54040


namespace driver_final_position_total_fuel_consumed_total_fare_collected_l54_54127

/-- Taxi distances traveled for each batch -/
def batch_distances : List Int := [5, 2, -4, -3, 6]

/-- Fuel consumed per kilometer -/
def fuel_rate : Float := 0.3

/-- Basic fare for distances not exceeding 3km -/
def base_fare : Float := 8

/-- Additional fare per kilometer for distance exceeding 3km -/
def extra_fare_per_km : Float := 1.6

/-- Prove that the driver ends up 6 km south of the starting point -/
theorem driver_final_position :
  (batch_distances.sum = 6) :=
by
  sorry

/-- Prove that the total fuel consumption is 6 liters -/
theorem total_fuel_consumed :
  (batch_distances.map (Int.natAbs ∘ coe : Nat).sum * fuel_rate = 6) :=
by
  sorry

/-- Prove that the total fare is 49.6 yuan -/
theorem total_fare_collected :
  let fare := fun (dist : Int) =>
    if dist <= 3 then base_fare else base_fare + ((float_of_int dist).natAbs - 3) * extra_fare_per_km
in
(batch_distances.map fare).sum = 49.6 :=
by
  sorry

end driver_final_position_total_fuel_consumed_total_fare_collected_l54_54127


namespace sum_of_angles_outside_pentagon_l54_54864

theorem sum_of_angles_outside_pentagon (A B C D E : Point) (circumcircle : Circle) (h : inscribed_pentagon circumcircle [A, B, C, D, E]) :
  sum_of_inscribed_angles_outside_pentagon circumcircle [A, B, C, D, E] = 720 :=
by sorry

end sum_of_angles_outside_pentagon_l54_54864


namespace remainder_2048_mod_13_l54_54742

theorem remainder_2048_mod_13 : 2048 % 13 = 7 := by
  sorry

end remainder_2048_mod_13_l54_54742


namespace sqrt_expression_simplify_l54_54905

theorem sqrt_expression_simplify : 
  abs (sqrt 2 - 1) - ((Real.pi + 1) ^ 0) + sqrt ((-3) ^ 2) = sqrt 2 + 1 :=
by
  sorry

end sqrt_expression_simplify_l54_54905


namespace sqrt_three_expression_l54_54149

theorem sqrt_three_expression : 2 * real.sqrt 3 - real.sqrt 3 = real.sqrt 3 :=
by
  sorry

end sqrt_three_expression_l54_54149


namespace general_term_l54_54791

def S (n : ℕ) : ℕ := n^2 + 3 * n

def a (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = 2 * n + 2 :=
by {
  sorry
}

end general_term_l54_54791


namespace brian_tape_needed_l54_54147

-- Definitions of conditions
def tape_needed_for_box (short_side: ℕ) (long_side: ℕ) : ℕ := 
  2 * short_side + long_side

def total_tape_needed (num_short_long_boxes: ℕ) (short_side: ℕ) (long_side: ℕ) (num_square_boxes: ℕ) (side: ℕ) : ℕ := 
  (num_short_long_boxes * tape_needed_for_box short_side long_side) + (num_square_boxes * 3 * side)

-- Theorem statement
theorem brian_tape_needed : total_tape_needed 5 15 30 2 40 = 540 := 
by 
  sorry

end brian_tape_needed_l54_54147


namespace train_length_correct_l54_54480

def train_length (speed_kph : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_mps := speed_kph * 1000 / 3600
  speed_mps * time_sec

theorem train_length_correct :
  train_length 90 10 = 250 := by
  sorry

end train_length_correct_l54_54480


namespace vasya_grades_l54_54678

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧ 
    (grades.nthLe 2 (by linarith) = 4) ∧ 
    (grades.sum = 19) ∧
    (grades.count 5 > List.foldl (fun acc n => if n ≠ 5 then acc + grades.count n else acc) 0 grades) ∧ 
    (grades.sorted = [2, 3, 4, 5, 5]) :=
sorry

end vasya_grades_l54_54678


namespace coefficient_of_x_in_expansion_l54_54776

theorem coefficient_of_x_in_expansion :
  (coeff (1 - 2 * x) ^ 4) = -8 :=
sorry

end coefficient_of_x_in_expansion_l54_54776


namespace water_level_same_l54_54210

-- Define the densities of water and ice
constant ρ_water : ℝ
constant ρ_ice : ℝ
-- Assume ice is less dense than water
axiom ice_less_dense_than_water : ρ_ice < ρ_water

-- Define the initial volume of the water in the molds
constant V : ℝ

-- Volume of ice formed when V volume of water freezes
def W := (V * ρ_water) / ρ_ice

-- Volume submerged when ice is placed in water
def U := V

theorem water_level_same (ρ_water ρ_ice : ℝ) (h1 : ρ_ice < ρ_water) :
  U = V :=
by
  -- The proof is omitted
  sorry

end water_level_same_l54_54210


namespace largest_apartment_size_l54_54138

theorem largest_apartment_size (r B : ℝ) (hr : r = 1.20) (hB : B = 840) : 
  (s : ℝ) -> 1.20 * s = 840 -> s = 700 :=
by
  intro s
  intro h
  have hs : s = B / r := by
    rw [hr, hB]
  sorry

end largest_apartment_size_l54_54138


namespace min_max_values_l54_54586

noncomputable def expression (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  ( (x₁ ^ 2 / x₂) + (x₂ ^ 2 / x₃) + (x₃ ^ 2 / x₄) + (x₄ ^ 2 / x₁) ) /
  ( x₁ + x₂ + x₃ + x₄ )

theorem min_max_values
  (a b : ℝ) (x₁ x₂ x₃ x₄ : ℝ)
  (h₀ : 0 < a) (h₁ : a < b)
  (h₂ : a ≤ x₁) (h₃ : x₁ ≤ b)
  (h₄ : a ≤ x₂) (h₅ : x₂ ≤ b)
  (h₆ : a ≤ x₃) (h₇ : x₃ ≤ b)
  (h₈ : a ≤ x₄) (h₉ : x₄ ≤ b) :
  expression x₁ x₂ x₃ x₄ ≥ 1 / b ∧ expression x₁ x₂ x₃ x₄ ≤ 1 / a :=
  sorry

end min_max_values_l54_54586


namespace find_f_l54_54980

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 3) * a * x^3 + (3 - a) * x^2 - 7 * x + 5

def f' (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * (3 - a) * x - 7

theorem find_f (a : ℝ) (x : ℝ) (h_pos : 0 < a) (h_bound : ∀ x ∈ set.Icc (-2:ℝ) (2:ℝ), abs (f' a x) ≤ 7) :
  f a x = x^3 - 7 * x + 5 :=
sorry

end find_f_l54_54980


namespace fixed_intersection_point_l54_54754

variables {A B C : Point} [h_circle_ABC : Circle (tri A B C)]
variables {X : Point} (hX : on_circle X (arc A B h_circle_ABC))
variables {O1 O2 : Point} (h_O1 : incenter O1 (tri C A X)) (h_O2 : incenter O2 (tri C B X))
variables h_fixed : fixed_point T (circumcircle (tri X O1 O2))

theorem fixed_intersection_point :
  ∃ T : Point, (intersects (circumcircle (tri X O1 O2)) (circle (tri A B C)) T) ∧
              (∀ X' on (arc A B h_circle_ABC), 
                 intersects (circumcircle (tri X' (incenter (tri C A X')) (incenter (tri C B X')))) 
                            (circle (tri A B C)) T)
:= sorry

end fixed_intersection_point_l54_54754


namespace greatest_five_digit_multiple_of_6_l54_54813

def is_mult_of_6 (n : ℕ) : Prop :=
  n % 6 = 0

theorem greatest_five_digit_multiple_of_6 : ∃! n : ℕ,
  (∃ p : list ℕ, p.perm [1, 5, 7, 8, 6] ∧ 
                n = nat.of_digits 10 p ∧ 
                is_mult_of_6 n) ∧
  ∀ m, (∃ p : list ℕ, p.perm [1, 5, 7, 8, 6] ∧ 
                      m = nat.of_digits 10 p ∧ 
                      is_mult_of_6 m) → m ≤ n :=
begin
  sorry
end

end greatest_five_digit_multiple_of_6_l54_54813


namespace subset_with_distance_condition_l54_54348

open Set Real

-- Define the conditions
variable {n : ℕ}
variable (S : Set (ℝ × ℝ))
variable (Hn : n > 14) 
variable (HS_card : S.card = n)
variable (H_dist : ∀ (p1 p2 : ℝ × ℝ), p1 ∈ S → p2 ∈ S → p1 ≠ p2 → dist p1 p2 ≥ 1)

-- The theorem statement
theorem subset_with_distance_condition : 
  ∃ (T : Set (ℝ × ℝ)), T ⊆ S ∧ T.card ≥ n / 7 ∧ ∀ (p1 p2 : ℝ × ℝ), p1 ∈ T → p2 ∈ T → p1 ≠ p2 → dist p1 p2 ≥ sqrt 3 :=
sorry

end subset_with_distance_condition_l54_54348


namespace max_S_optimal_values_min_S_optimal_values_l54_54093

-- Define the variables and conditions for the maximization problem
def maximize_S_conditions (x1 x2 x3 x4 x5 : ℕ) :=
  x1 + x2 + x3 + x4 + x5 = 2006 ∧
  (∀ i, i ∈ [x1, x2, x3, x4, x5] → i > 0)

def maximize_S (x1 x2 x3 x4 x5 : ℕ) :=
  ∑ i in { (i, j) : Finset.product (Finset.range 5) (Finset.range 5) | i < j }, x1*x2 + x1*x3 + x1*x4 + x1*x5 + x2*x3 + x2*x4 + x2*x5 + x3*x4 + x3*x5 + x4*x5

theorem max_S_optimal_values : ∃ (x1 x2 x3 x4 x5 : ℕ), maximize_S_conditions x1 x2 x3 x4 x5 ∧ maximize_S x1 x2 x3 x4 x5 := 402*401 + 401*401 + 401*401 + 401*401 + 401*401 :=
  ⟨402, 401, 401, 401, 401, ⟨rfl, sorry⟩⟩ -- Fill in the proof

-- Define the variables and conditions for the minimization problem
def minimize_S_conditions (x1 x2 x3 x4 x5 : ℕ) :=
  x1 + x2 + x3 + x4 + x5 = 2006 ∧
  (∀ i j, i ≠ j → |i - j| ≤ 2) ∧
  (∀ i, i ∈ [x1, x2, x3, x4, x5] → i > 0)

def minimize_S (x1 x2 x3 x4 x5 : ℕ) :=
  ∑ i in { (i, j) : Finset.product (Finset.range 5) (Finset.range 5) | i < j }, x1*x2 + x1*x3 + x1*x4 + x1*x5 + x2*x3 + x2*x4 + x2*x5 + x3*x4 + x3*x5 + x4*x5

theorem min_S_optimal_values : ∃ (x1 x2 x3 x4 x5 : ℕ), minimize_S_conditions x1 x2 x3 x4 x5 ∧ minimize_S x1 x2 x3 x4 x5 := 402*402 + 402*402 + 402*400 + 402*400 + 401*400 + 401*400 :=
  ⟨402, 402, 402, 400, 400, ⟨rfl, sorry, sorry⟩⟩ -- Fill in the proof

end max_S_optimal_values_min_S_optimal_values_l54_54093


namespace smallest_n_logarithm_l54_54169

theorem smallest_n_logarithm :
  ∃ n : ℕ, 0 < n ∧ 
  (Real.log (Real.log n / Real.log 3) / Real.log 3^2 =
  Real.log (Real.log n / Real.log 2) / Real.log 2^3) ∧ 
  n = 9 :=
by
  sorry

end smallest_n_logarithm_l54_54169


namespace lattice_points_count_l54_54471

def lattice_point (p : ℤ × ℤ) := 
  ∃ (x y : ℤ), p = (x, y)

def inside_region (p : ℝ × ℝ) := 
  let (x, y) := p in 
  y ≤ |x| ∧ y ≤ -x^2 + 4

theorem lattice_points_count : 
  let points := {p : ℤ × ℤ | inside_region (p.1, p.2)} 
  in points.to_finset.card = 13 := 
sorry

end lattice_points_count_l54_54471


namespace correct_choice_is_D_l54_54519

open Real

theorem correct_choice_is_D :
  (∀ x ∈ ℝ, x^2 + 1 = 0) = False ∧
  (∃ x ∈ ℝ, x^2 + 1 = 0) = False ∧
  (∀ x ∈ ℝ, sin x < tan x) = False ∧
  (∃ x ∈ ℝ, sin x < tan x) = True :=
by
  sorry

end correct_choice_is_D_l54_54519


namespace min_eccentricity_of_ellipse_l54_54581

theorem min_eccentricity_of_ellipse (a b c : ℝ) (h0 : a > b) (h1 : b > 0)
  (he : c = real.sqrt (a^2 - b^2))
  (hpq : forall (P Q : ℝ × ℝ), (P.1 - a, P.2) • (Q.1 - a, Q.2) = (1/2)*(a + c)^2) 
  : ∃ e : ℝ, e = 1 - real.sqrt 2/2 :=
sorry

end min_eccentricity_of_ellipse_l54_54581


namespace angle_between_reflected_rays_l54_54855

theorem angle_between_reflected_rays (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ φ : ℝ, φ = arccos (1 - 2 * sin(α) ^ 2 * sin(β) ^ 2) :=
begin
  sorry
end

end angle_between_reflected_rays_l54_54855


namespace negation_proposition_l54_54391

-- Definitions based on the conditions
def original_proposition : Prop := ∃ x : ℝ, x^2 + 3*x + 2 < 0

-- Theorem requiring proof
theorem negation_proposition : (¬ original_proposition) = ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_proposition_l54_54391


namespace distance_from_C_to_line_AB_l54_54259

noncomputable def vector3d := ℝ × ℝ × ℝ

def pointA : vector3d := (-1, 0, 0)
def pointB : vector3d := (0, 1, -1)
def pointC : vector3d := (-1, -1, 2)

def sub (u v : vector3d) : vector3d := (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def magnitude (v : vector3d) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

def dot_product (u v : vector3d) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
    
theorem distance_from_C_to_line_AB:
  let AC := sub pointC pointA in
  let AB := sub pointB pointA in
  real.sqrt (magnitude AC ^ 2 - (dot_product AC AB / magnitude AB) ^ 2) = real.sqrt 2 :=
by
  sorry

end distance_from_C_to_line_AB_l54_54259


namespace solution_exists_l54_54608

-- Define the given condition as a hypothesis
theorem solution_exists (a x : ℝ) 
  (h₁ : ∃ x, 2 * (x + 1) = 3 * (x - 1)) : 
  2 * [2 * (x + 3) - 3 * (x - a)] = 3 * a → x = 10 := 
by
  sorry

end solution_exists_l54_54608


namespace no_valid_arrangement_l54_54315

theorem no_valid_arrangement :
  ∀ (table : Fin 300 → Fin 300 → Int), 
  (∀ i j, table i j = 1 ∨ table i j = -1) →
  abs (∑ i j, table i j) < 30000 →
  (∀ i j, abs (∑ x in Finset.finRange 3, ∑ y in Finset.finRange 5, table (i + x) (j + y)) > 3) →
  (∀ i j, abs (∑ x in Finset.finRange 5, ∑ y in Finset.finRange 3, table (i + x) (j + y)) > 3) →
  False :=
by
  intros table h_entries h_total_sum h_rect_sum_3x5 h_rect_sum_5x3
  sorry

end no_valid_arrangement_l54_54315


namespace butterfly_black_dots_l54_54406

theorem butterfly_black_dots (b f : ℕ) (total_butterflies : b = 397) (total_black_dots : f = 4764) : f / b = 12 :=
by
  sorry

end butterfly_black_dots_l54_54406


namespace ellipse_equation_line_l_equation_l54_54582

-- Definitions for conditions
def ellipse_center_origin := true
def ellipse_axes_aligned := true
def ellipse_vertex_A : Prop := A = (0, 2)
def distance_focus_B (F : Point) (B : Point) := dist F B = 2

-- Theorem I: Equation of the ellipse
theorem ellipse_equation (A F : Point) (B : Point) 
  (hA : ellipse_vertex_A) 
  (hdist : distance_focus_B F B) 
  : (ellipse_center_origin ∧ ellipse_axes_aligned) →
    A = (0, 2) →
    (F.x = √(a^2 - b^2) ∧ b = 2) →
    a^2 = 12 →
    (\(fracx^2 / 12 + frac y^2 / 4 = 1\). sorry

-- Definitions for conditions in II
def line_l_passing_point := (0, -3)
def point_MN_on_ellipse (M N : Point) := (* Check that M, N lie on the ellipse *)

-- Theorem II: Equation of line l
theorem line_l_equation (A M N : Point)
  (hM : point_MN_on_ellipse M N)
  (hA_eq_AN : dist A M = dist A N) 
  (hl : line_l_passing_point) 
  : (line_passing (0,-3) intersect_ellipse_at_two) →
    A = (0, 2) →
    (\x = ± \frac{\sqrt{6}}{3}x - 3. sorry


end ellipse_equation_line_l_equation_l54_54582


namespace smallest_positive_int_l54_54820

open Nat

theorem smallest_positive_int (x : ℕ) :
  (x % 6 = 3) ∧ (x % 8 = 5) ∧ (x % 9 = 2) → x = 237 := by
  sorry

end smallest_positive_int_l54_54820


namespace D_is_divisor_of_n_squared_D_divides_n_squared_iff_prime_l54_54728

def positive_divisors (n : ℕ) : List ℕ := (List.range n).filter (λ d, d > 0 ∧ n % d = 0)

noncomputable def D (n : ℕ) (divs : List ℕ) : ℕ :=
List.foldr (λ (pair : ℕ × ℕ) acc, acc + pair.fst * pair.snd)
  0 (List.zip divs.init divs.tail)

theorem D_is_divisor_of_n_squared (n : ℕ) (h : 1 < n) (divisors : List ℕ)
  (hd : divisors = positive_divisors n) :
  D n divisors < n^2 → D n divisors ∣ n^2 :=
  sorry

theorem D_divides_n_squared_iff_prime (n : ℕ) (h : 1 < n) :
  D n (positive_divisors n) < n^2 →
  (D n (positive_divisors n) ∣ n^2 ↔ Nat.prime n) :=
  sorry

end D_is_divisor_of_n_squared_D_divides_n_squared_iff_prime_l54_54728


namespace functional_equation_solution_l54_54544

noncomputable def f (x : ℝ) : ℝ := x

theorem functional_equation_solution (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    f(2 * x * f(3 * y)) + f(27 * y^3 * f(2 * x)) = 6 * x * y + 54 * x * y^3 :=
by
  sorry

end functional_equation_solution_l54_54544


namespace percentage_chain_l54_54268

theorem percentage_chain (n : ℝ) (h : n = 6000) : 0.1 * (0.3 * (0.5 * n)) = 90 := by
  sorry

end percentage_chain_l54_54268


namespace black_ball_with_equal_whites_and_blacks_l54_54056

theorem black_ball_with_equal_whites_and_blacks (n_black n_white : ℕ) (h_black_even : n_black = 5) (h_white_even : n_white = 4) :
  ∃ k, k < 9 ∧ (nth_ball k = black ∧ (number_of_white_balls_to_right k = number_of_black_balls_to_right k)) :=
by
  sorry

end black_ball_with_equal_whites_and_blacks_l54_54056


namespace volume_of_figure_eq_half_l54_54150

-- Define a cube data structure and its properties
structure Cube where
  edge_length : ℝ
  h_el : edge_length = 1

-- Define a function to calculate volume of the figure
noncomputable def volume_of_figure (c : Cube) : ℝ := sorry

-- Example cube
def example_cube : Cube := { edge_length := 1, h_el := rfl }

-- Theorem statement
theorem volume_of_figure_eq_half (c : Cube) : volume_of_figure c = 1 / 2 := by
  sorry

end volume_of_figure_eq_half_l54_54150


namespace length_CF_area_triangle_ACF_l54_54831

noncomputable def circles_intersect_at (R : ℝ) (A B : ℝ) := sorry

noncomputable def point_on_circle (R : ℝ) (P : ℝ) := sorry

noncomputable def perpendicular_line (CD B F : ℝ) := sorry

variables {R : ℝ} (A B C D F : ℝ)

def problem_conditions := 
  circles_intersect_at 5 A B ∧ 
  point_on_circle 5 C ∧ 
  point_on_circle 5 D ∧ 
  (B lies_on_segment CD) ∧ 
  (angle CAD = (π / 2)) ∧ 
  perpendicular_line CD B F ∧ 
  (BF = BD) ∧ 
  (BC = 6)

theorem length_CF (h : problem_conditions A B C D F) : CF = 10 := 
sorry 

theorem area_triangle_ACF (h : problem_conditions A B C D F) : Area_of_ΔACF = 7 := 
sorry

end length_CF_area_triangle_ACF_l54_54831


namespace triangle_dist_ratio_triangle_dist_ratio_right_l54_54840

theorem triangle_dist_ratio (A B C D : Point) (hA : ∠BAC < 90°) :
  D ∈ triangle A B C →
  BC / min AD (min BD CD) ≥ 2 * sin (∠BAC) :=
sorry

theorem triangle_dist_ratio_right (A B C D : Point) (hA : ∠BAC ≥ 90°) :
  D ∈ triangle A B C →
  BC / min AD (min BD CD) ≥ 2 :=
sorry

end triangle_dist_ratio_triangle_dist_ratio_right_l54_54840


namespace find_a1_l54_54048

-- Define the sequence (a_n)
def seq (a : ℕ → ℕ) := ∀ n ≥ 2, (∑ i in finset.range (n + 1), a i) = n ^ 2 * a n

-- The given conditions in the problem
variables (a : ℕ → ℕ)
variables (h_seq : seq a)
variables (h_a64 : a 64 = 2)

-- The goal is to prove that a_1 = 4160
theorem find_a1 : a 1 = 4160 :=
by { sorry }

end find_a1_l54_54048


namespace length_of_AC_l54_54689

theorem length_of_AC (A B C : ℝ) (h : 0 < A) (h0 : ∠BAC = A) (h1 : ∠BCA = 3 * A) (h2 : AB = 10) (h3 : BC = 8) : ∃ AC, AC = 3 :=
  by 
  sorry

end length_of_AC_l54_54689


namespace measure_angle_RPQ_l54_54299

theorem measure_angle_RPQ (P Q R S : Type) (m : ∀ T : Type, Prop)
  (on_line_RS : P ∈ {S, R}) 
  (bisects_angle : m (QP bisects ∠SQR)) 
  (equals_PQ_PR : PQ = PR) 
  (angle_RSQ : m (∠RSQ = 3 * y)) 
  (angle_RPQ : m (∠RPQ = 2 * y)) 
  (triangle_angle_sum : ∀ A B C : Type, m (∠A + ∠B + ∠C = 180)) :
  m (∠RPQ = 72) := 
sorry

end measure_angle_RPQ_l54_54299


namespace luca_drink_cost_l54_54737

theorem luca_drink_cost (sandwich_price : ℕ) (coupon_fraction : ℚ) (avocado_extra : ℕ) (salad_cost : ℕ) (total_bill : ℕ) (drink_cost: ℕ) 
  (h_sandwich_price : sandwich_price = 8)
  (h_coupon_fraction : coupon_fraction = 1 / 4)
  (h_avocado_extra : avocado_extra = 1)
  (h_salad_cost : salad_cost = 3)
  (h_total_bill : total_bill = 12)
  (h_drink_cost : drink_cost = 2)
  (h : total_bill = sandwich_price * (1 - coupon_fraction).toNat + avocado_extra + salad_cost + drink_cost) :
  drink_cost = 2 := sorry

end luca_drink_cost_l54_54737


namespace problem_1_problem_2_problem_3_l54_54962

-- Definitions based on the conditions
def a_seq (n : ℕ) : ℕ → ℝ
| 0       := 1
| (n + 1) := 1 + 2 / a_seq n

def b_seq (n : ℕ) : ℝ := (a_seq n - 2) / (a_seq n + 1)

def c_seq (n : ℕ) : ℝ := n * b_seq n

def s_seq (n : ℕ) : ℝ := ∑ i in Finset.range n, c_seq i

-- Statement of the problems
theorem problem_1 (n : ℕ) : b_seq n = (-1 / 2) ^ n := sorry

theorem problem_2 (n : ℕ) : a_seq n = (2^(n+1) + (-1)^n) / (2^n + (-1)^(n-1)) := sorry

theorem problem_3 (n : ℕ) (m : ℕ) : 
  (m : ℝ) / 32 + 3 / 2 * s_seq n + n * ((-1 / 2)^(n+1)) - ((-1 / 2)^n) / 3 > 0 ↔ 
  m ≥ 11 := sorry

end problem_1_problem_2_problem_3_l54_54962


namespace polynomial_degree_number_of_terms_l54_54785

-- Define the polynomial
def P (x y : ℝ) : ℝ := 2 * x * y - x^2 * y + 3 * x^3 * y - 5

-- Define functions to calculate the degree of a term and the number of terms in a polynomial
def termDegree (term : ℝ) (xDegree yDegree : ℕ) : ℕ := xDegree + yDegree

def polynomialDegree (P : ℝ → ℝ → ℝ) : ℕ :=
  max (max (termDegree (2 * x * y) 1 1) (termDegree (- x^2 * y) 2 1))
      (max (termDegree (3 * x^3 * y) 3 1) (termDegree (-5) 0 0))

def numberOfTerms (P : ℝ → ℝ → ℝ) : ℕ := 4

-- Statement to prove the degree and number of terms of the polynomial
theorem polynomial_degree_number_of_terms :
  polynomialDegree P = 4 ∧ numberOfTerms P = 4 :=
by
  -- We leave the proof as a sorry, as per the instruction
  sorry

end polynomial_degree_number_of_terms_l54_54785


namespace area_of_isosceles_trapezoid_l54_54086

variable (a b c d : ℝ) -- Variables for sides and bases of the trapezoid

-- Define isosceles trapezoid with given sides and bases
def is_isosceles_trapezoid (a b c d : ℝ) (h : ℝ) :=
  a = b ∧ c = 10 ∧ d = 16 ∧ (∃ (h : ℝ), a^2 = h^2 + ((d - c) / 2)^2 ∧ a = 5)

-- Lean theorem for the area of the isosceles trapezoid
theorem area_of_isosceles_trapezoid :
  ∀ (a b c d : ℝ) (h : ℝ), is_isosceles_trapezoid a b c d h
  → (1 / 2) * (c + d) * h = 52 :=
by
  sorry

end area_of_isosceles_trapezoid_l54_54086


namespace linear_coefficient_of_quadratic_term_is_negative_five_l54_54484

theorem linear_coefficient_of_quadratic_term_is_negative_five (a b c : ℝ) (x : ℝ) :
  (2 * x^2 = 5 * x - 3) →
  (a = 2) →
  (b = -5) →
  (c = 3) →
  (a * x^2 + b * x + c = 0) :=
by
  sorry

end linear_coefficient_of_quadratic_term_is_negative_five_l54_54484


namespace range_m_necessary_not_sufficient_l54_54721

def f (x : ℝ) : ℝ := Real.sqrt (2 + x) + Real.log (4 - x)

def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

def p (x : ℝ) : Prop := x ∈ A

def q (x : ℝ) : Prop := ∃ (m : ℝ), x ∈ B m

theorem range_m_necessary_not_sufficient (m : ℝ) : 
  ∀ x, (p x → q x) ∧ ¬ (q x → p x) ↔ m ∈ Set.Iio (5 / 2) :=
sorry

end range_m_necessary_not_sufficient_l54_54721


namespace log_base_9_of_3_l54_54528

theorem log_base_9_of_3 : log 9 3 = 1 / 2 :=
by sorry

end log_base_9_of_3_l54_54528


namespace chess_tournament_l54_54459

theorem chess_tournament : 
  ∀ (n : ℕ), (17 * 16 * n) / 2 = 272 → n = 2 := 
by 
  intros n h,
  sorry

end chess_tournament_l54_54459


namespace weight_lifting_difference_l54_54288

variable (F S : ℕ)

theorem weight_lifting_difference :
  F = 600 → F + S = 1500 → 2 * F - S = 300 :=
by intros hF hFS
   rw [hF] at hFS
   have hS : S = 900 := by linarith
   simp [hF, hS]
   sorry

end weight_lifting_difference_l54_54288


namespace number_of_permutations_satisfying_condition_l54_54546

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0      => 1
  | (n+1)  => (n+1) * factorial n

def is_permutation (l : List ℕ) (l' : List ℕ) : Prop :=
  l'.perm l

def condition (a1 a2 a3 a4 a5 a6 : ℕ) : Prop :=
  (a1 + 1) / 2 * (a2 + 2) / 2 * (a3 + 3) / 2 * (a4 + 4) / 2 * (a5 + 5) / 2 * (a6 + 6) / 2 > factorial 6

theorem number_of_permutations_satisfying_condition :
  (List.permutations [1, 2, 3, 4, 5, 6]).filter (fun l => match l with
                                                            | [a1, a2, a3, a4, a5, a6] => condition a1 a2 a3 a4 a5 a6
                                                            | _ => false
                                                          ).length = 719 :=
  sorry

end number_of_permutations_satisfying_condition_l54_54546


namespace impossibleArrangement_l54_54308

-- Define the parameters of the table
def n : ℕ := 300

-- Define the properties of an arrangement.
def isValidArrangement (arr : ℕ × ℕ → ℤ) : Prop :=
  (∀ i j, arr (i, j) = 1 ∨ arr (i, j) = -1) ∧
  (|∑ i in finset.range n, ∑ j in finset.range n, arr (i, j)| < 30000) ∧
  (∀ i j, 
    (i ≤ n - 3 ∧ j ≤ n - 5 → |∑ a in finset.range 3, ∑ b in finset.range 5, arr (i + a, j + b)| > 3) ∧
    (i ≤ n - 5 ∧ j ≤ n - 3 → |∑ a in finset.range 5, ∑ b in finset.range 3, arr (i + a, j + b)| > 3))
  
-- Formalizing the problem in Lean.
theorem impossibleArrangement : ¬ ∃ arr : (ℕ × ℕ → ℤ), isValidArrangement arr :=
by
  sorry

end impossibleArrangement_l54_54308


namespace bathroom_module_area_150_l54_54163

def total_cost_eq (B : ℕ) : Prop :=
  20000 + 2 * 12000 + (2000 - 400 - 2 * B) * 100 = 174000

theorem bathroom_module_area_150 :
  ∃ B : ℕ, total_cost_eq B ∧ B = 150 :=
by
  exists 150
  unfold total_cost_eq
  rfl

end bathroom_module_area_150_l54_54163


namespace circle_area_l54_54752

-- Definitions
def point := ℝ × ℝ

def on_circle (A B : point) (center : point) (radius : ℝ) : Prop :=
  (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
  (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2

def tangents_intersect_on_x_axis (A B center : point) : Prop :=
  ∃ x : ℝ, let C := (x,0) in 
            (C.1 - center.1)*(A.1 - center.1) + (C.2 - center.2)*(A.2 - center.2) = 0 ∧
            (C.1 - center.1)*(B.1 - center.1) + (C.2 - center.2)*(B.2 - center.2) = 0

-- Theorem to prove
theorem circle_area (A B : point) (r : ℝ) 
  (hA : A = (2, 5)) 
  (hB : B = (8, 3)) 
  (center : point)
  (h_on_circle : on_circle A B center r)
  (h_tangents : tangents_intersect_on_x_axis A B center) 
  : ∃ area : ℝ, area = 226 * π / 9 :=
begin
  sorry
end

end circle_area_l54_54752


namespace tournament_total_games_l54_54006

def total_number_of_games (num_teams : ℕ) (group_size : ℕ) (num_groups : ℕ) (teams_for_knockout : ℕ) : ℕ :=
  let games_per_group := (group_size * (group_size - 1)) / 2
  let group_stage_games := num_groups * games_per_group
  let knockout_teams := num_groups * teams_for_knockout
  let knockout_games := knockout_teams - 1
  group_stage_games + knockout_games

theorem tournament_total_games : total_number_of_games 32 4 8 2 = 63 := by
  sorry

end tournament_total_games_l54_54006


namespace value_of_x_l54_54381

theorem value_of_x (x : ℚ) (h : (x + 10 + 17 + 3 * x + 15 + 3 * x + 6) / 5 = 26) : x = 82 / 7 :=
by
  sorry

end value_of_x_l54_54381


namespace delta_gj_l54_54002

def vj := 120
def total := 770
def gj := total - vj

theorem delta_gj : gj - 5 * vj = 50 := by
  sorry

end delta_gj_l54_54002


namespace C_power_50_l54_54705

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℤ :=
  !![ [3, 1], [-8, -3] ]

theorem C_power_50 :
  C ^ 50 = 1 :=
  sorry

end C_power_50_l54_54705


namespace initial_ducks_count_l54_54769

theorem initial_ducks_count :
  ∃ D : ℕ, D + 20 = 33 ∧ D = 13 :=
begin
  use 13,
  split,
  { exact rfl, },
  { exact rfl, }
end

end initial_ducks_count_l54_54769


namespace impossibleArrangement_l54_54307

-- Define the parameters of the table
def n : ℕ := 300

-- Define the properties of an arrangement.
def isValidArrangement (arr : ℕ × ℕ → ℤ) : Prop :=
  (∀ i j, arr (i, j) = 1 ∨ arr (i, j) = -1) ∧
  (|∑ i in finset.range n, ∑ j in finset.range n, arr (i, j)| < 30000) ∧
  (∀ i j, 
    (i ≤ n - 3 ∧ j ≤ n - 5 → |∑ a in finset.range 3, ∑ b in finset.range 5, arr (i + a, j + b)| > 3) ∧
    (i ≤ n - 5 ∧ j ≤ n - 3 → |∑ a in finset.range 5, ∑ b in finset.range 3, arr (i + a, j + b)| > 3))
  
-- Formalizing the problem in Lean.
theorem impossibleArrangement : ¬ ∃ arr : (ℕ × ℕ → ℤ), isValidArrangement arr :=
by
  sorry

end impossibleArrangement_l54_54307


namespace eval_power_l54_54181

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end eval_power_l54_54181


namespace calc_mod_residue_l54_54499

theorem calc_mod_residue :
  (245 * 15 - 18 * 8 + 5) % 17 = 0 := by
  sorry

end calc_mod_residue_l54_54499


namespace traverse_all_segments_l54_54425

theorem traverse_all_segments (n : ℕ) (polygon : Type) [semigroup polygon] :
  ∃ path : list polygon, (all_segments_traversed path) :=
begin
  sorry
end

end traverse_all_segments_l54_54425


namespace power_mod_eq_remainder_l54_54071

theorem power_mod_eq_remainder (b m e : ℕ) (hb : b = 17) (hm : m = 23) (he : e = 2090) : 
  b^e % m = 12 := 
  by sorry

end power_mod_eq_remainder_l54_54071


namespace initial_number_of_girls_l54_54209

-- Definitions according to conditions
variables (b g : ℚ)

-- Condition 1: 20 girls leave, remaining boys to girls ratio is 3:1.
def cond1 := 3 * (g - 20) = b

-- Condition 2: After 60 boys leave, remaining girls to boys ratio is 6:1.
def cond2 := 6 * (b - 60) = (g - 20)

-- Main theorem to prove
theorem initial_number_of_girls (h1 : cond1) (h2 : cond2) : g = 700 / 17 := 
sorry

end initial_number_of_girls_l54_54209


namespace shooting_probability_l54_54873

theorem shooting_probability (p : ℝ) (h : p = 0.9) : 
  (∀ (n : ℕ), n = 100 → (hits : ℕ), hits = 90 → p = hits / n) → 
  (p = 0.9 → ∀ (x : ℕ), x = 1 → (successful : bool), successful = tt → p = 0.9) :=
by
  sorry

end shooting_probability_l54_54873


namespace find_train_speed_l54_54479

def length_of_platform : ℝ := 210.0168
def time_to_pass_platform : ℝ := 34
def time_to_pass_man : ℝ := 20 
def speed_of_train (L : ℝ) (V : ℝ) : Prop :=
  V = (L + length_of_platform) / time_to_pass_platform ∧ V = L / time_to_pass_man

theorem find_train_speed (L V : ℝ) (h : speed_of_train L V) : V = 54.00432 := sorry

end find_train_speed_l54_54479


namespace csc_alpha_at_point_l54_54795

-- Conditions
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def point_on_unit_circle : Prop := ∃ x : ℝ, unit_circle x (3/5)

-- Question
theorem csc_alpha_at_point : point_on_unit_circle → ∃ α : ℝ, real.csc α = 5 / 3 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  have sin_alpha : real.sin α = 3 / 5 := by
    sorry
  use α
  rw [real.csc_eq],
  rw sin_alpha,
  norm_num
  sorry

end csc_alpha_at_point_l54_54795


namespace cover_all_rows_by_T13_l54_54865

-- Triangular number definition
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Required condition: Prove that T_13 is the smallest triangular number to cover all rows
def rows_covered (n : ℕ) : Prop :=
  ∀ r, 1 ≤ r ∧ r ≤ 10 → ∃ m, triangular m ∈ row_range r

def row_range (r : ℕ) : set ℕ :=
  if r % 2 = 1 then
    { n : ℕ | 10 * (r - 1) + n }
  else
    { n : ℕ | 10 * r - (n - 1) }

theorem cover_all_rows_by_T13 :
  ∀ n, rows_covered n ↔ 13 ≤ n ∧ triangular 13 = 91 :=
sorry

end cover_all_rows_by_T13_l54_54865


namespace no_triples_of_consecutive_numbers_l54_54094

theorem no_triples_of_consecutive_numbers (n : ℤ) (a : ℕ) (h : 1 ≤ a ∧ a ≤ 9) :
  ¬(3 * n^2 + 2 = 1111 * a) :=
by sorry

end no_triples_of_consecutive_numbers_l54_54094


namespace solution_set_inequality_l54_54049

theorem solution_set_inequality (x : ℝ) : 3 * x - 2 > x → x > 1 := by
  sorry

end solution_set_inequality_l54_54049


namespace minimize_J_l54_54372

noncomputable def H (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ :=
  if p < 0 then 0 else if p > 1 then 1 else if (9 * p - 5 > 4 - 7 * p) then 9 * p - 5 else 4 - 7 * p

theorem minimize_J :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ J p = J (9 / 16) := by
  sorry

end minimize_J_l54_54372


namespace triangle_XYZ_is_right_triangle_at_X_l54_54710

-- Define the points
variables {A B C D X Y Z : Type}

-- Conditions: ABX and CDX are right triangles with right angles at X
variables (hABX : ∀ (A B X : ℝ), triangle.ABX (right_angle X A B))
variables (hCDX : ∀ (C D X : ℝ), triangle.CDX (right_angle X C D))

-- Intersections: Y = (AC) ∩ (BD) and Z = (AD) ∩ (BC)
variables (hY : ∃ (Y : Type), intersection_point AC BD Y)
variables (hZ : ∃ (Z : Type), intersection_point AD BC Z)

-- Goal: Prove that XYZ is a right triangle at X
theorem triangle_XYZ_is_right_triangle_at_X 
  (hXY : line Y X)
  (hXZ : line Z X) : 
  ∠ Y X Z = 90° := 
sorry

end triangle_XYZ_is_right_triangle_at_X_l54_54710


namespace raisin_fraction_of_mixture_l54_54447

noncomputable def raisin_nut_cost_fraction (R : ℝ) : ℝ :=
  let raisin_cost := 3 * R
  let nut_cost := 4 * (4 * R)
  let total_cost := raisin_cost + nut_cost
  raisin_cost / total_cost

theorem raisin_fraction_of_mixture (R : ℝ) : raisin_nut_cost_fraction R = 3 / 19 :=
by
  sorry

end raisin_fraction_of_mixture_l54_54447


namespace angle_bisector_inequality_l54_54999

noncomputable def incenter :=
  sorry -- definition of incenter will be omitted due to its complexity

variables {A B C I A' B' C' : Π (A B C : Point), Point}

-- Given a triangle ABC and its incenter I
axiom triangle_ABC : triangle A B C
axiom incenter_I : incenter A B C = I

-- Given that angle bisectors intersect the opposite sides at A', B', and C' respectively
axiom angle_bisectors_meet :
  (IsAngleBisector (A, I) B' C) ∧
  (IsAngleBisector (B, I) A' C) ∧
  (IsAngleBisector (C, I) A B') 

-- Define angle bisector functions (these should follow from angle_bisectors_meet axiom)
noncomputable def AI := line_segment A I
noncomputable def BI := line_segment B I
noncomputable def CI := line_segment C I
noncomputable def AA' := line_segment A A'
noncomputable def BB' := line_segment B B'
noncomputable def CC' := line_segment C C'

theorem angle_bisector_inequality :
  (1/4 : ℝ) < (AI * BI * CI) / (AA' * BB' * CC') ∧ (AI * BI * CI) / (AA' * BB' * CC') ≤ 8/27 :=
by 
  sorry -- Proof is omitted 

end angle_bisector_inequality_l54_54999


namespace average_of_D_E_F_l54_54771

theorem average_of_D_E_F (D E F : ℝ) 
  (h1 : 2003 * F - 4006 * D = 8012) 
  (h2 : 2003 * E + 6009 * D = 10010) : 
  (D + E + F) / 3 = 3 := 
by 
  sorry

end average_of_D_E_F_l54_54771


namespace usual_time_eq_three_l54_54875

variable (S T : ℝ)
variable (usual_speed : S > 0)
variable (usual_time : T > 0)
variable (reduced_speed : S' = 6/7 * S)
variable (reduced_time : T' = T + 0.5)

theorem usual_time_eq_three (h : 7/6 = T' / T) : T = 3 :=
by
  -- proof to be filled in
  sorry

end usual_time_eq_three_l54_54875


namespace vector_magnitude_l54_54245

variables {a b : EuclideanSpace ℝ ℝ'}

-- Define the conditions
def is_unit_vector (v : EuclideanSpace ℝ ℝ') : Prop :=
  ∥v∥ = 1

def angle_between (u v : EuclideanSpace ℝ ℝ') (θ : ℝ) : Prop :=
  real_inner u v = ∥u∥ * ∥v∥ * real.cos θ

-- The proof statement
theorem vector_magnitude (a b : EuclideanSpace ℝ ℝ') (h1 : is_unit_vector a) (h2 : is_unit_vector b) (h3 : angle_between a b (2 * real.pi / 3)) :
  ∥a - 3 • b∥ = real.sqrt 13 :=
sorry

end vector_magnitude_l54_54245


namespace number_of_positive_integer_pairs_l54_54941

theorem number_of_positive_integer_pairs (x y : ℕ) (h : 20 * x + 6 * y = 2006) : 
  ∃ n, n = 34 ∧ ∀ (x y : ℕ), 20 * x + 6 * y = 2006 → 0 < x → 0 < y → 
  (∃ k, x = 3 * k + 1 ∧ y = 331 - 10 * k ∧ 0 ≤ k ∧ k ≤ 33) :=
sorry

end number_of_positive_integer_pairs_l54_54941


namespace solve_inequality_system_l54_54369

-- Define the conditions and the correct answer
def system_of_inequalities (x : ℝ) : Prop :=
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)

def solution_set (x : ℝ) : Prop :=
  2 < x ∧ x ≤ 4

-- State that solving the system of inequalities is equivalent to the solution set
theorem solve_inequality_system (x : ℝ) : system_of_inequalities x ↔ solution_set x :=
  sorry

end solve_inequality_system_l54_54369


namespace excellent_pyramid_minimum_l54_54126

-- Define the subtraction pyramid and properties of the excellent pyramid.
def subtraction_pyramid (pyramid : List (List ℕ)) : Prop :=
  ∀ (n : Nat) (i : Nat), n > 0 → i < (pyramid.length - n) →
  (pyramid[n][i] = pyramid[n-1][i] - pyramid[n-1][i+1] ∨
   pyramid[n][i] = pyramid[n-1][i+1] - pyramid[n-1][i])

def is_excellent (pyramid : List (List ℕ)) : Prop :=
  ∃ top_level unique_level, pyramid.top_level = [0] ∧
  list.set top_level ∧
  (∀ (xs : List ℕ), list.mem xs pyramid)

def minimum_levels (pyramid : List (List ℕ)) : Prop :=
  pyramid.length ≥ 3

def minimum_significant (pyramid : List (List ℕ)) : ℕ :=
  list.maximum (list.flatten pyramid)

-- The statement to prove in Lean:
theorem excellent_pyramid_minimum :
  ∃ (pyramid : List (List ℕ)), is_excellent pyramid ∧ minimum_levels pyramid ∧ minimum_significant pyramid = 2 :=
begin
  -- Construction of the excellent pyramid is omitted
  sorry
end

end excellent_pyramid_minimum_l54_54126


namespace janina_spend_on_supplies_each_day_l54_54693

theorem janina_spend_on_supplies_each_day 
  (rent : ℝ)
  (p : ℝ)
  (n : ℕ)
  (H1 : rent = 30)
  (H2 : p = 2)
  (H3 : n = 21) :
  (n : ℝ) * p - rent = 12 := 
by
  sorry

end janina_spend_on_supplies_each_day_l54_54693


namespace min_value_of_f_l54_54846

def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

theorem min_value_of_f : ∃ x ∈ Set.Ico 0 (2 * Real.pi), f x = -3 * Real.sqrt 3 / 2 := 
by
  sorry

end min_value_of_f_l54_54846


namespace function_equiv_proof_l54_54193

noncomputable def function_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) = f x * y

theorem function_equiv_proof : ∀ f : ℝ → ℝ,
  function_solution f ↔ (∀ x : ℝ, f x = 0 ∨ f x = x ∨ f x = -x) := 
sorry

end function_equiv_proof_l54_54193


namespace smallest_sector_angle_l54_54801

theorem smallest_sector_angle :
  ∃ a_1 d : ℕ, 
    (∀ n : ℕ, 0 < n ∧ n ≤ 16 → ∃ k : ℕ, k = a_1 + (n - 1) * d ∧ k ∈ ℕ) ∧ 
    (2 * a_1 + 15 * d = 45) ∧ 
    a_1 = 3 :=
by sorry

end smallest_sector_angle_l54_54801


namespace tourist_check_total_l54_54128

-- Definitions
def total_value : ℕ → ℕ → ℕ := λ F H, 50 * F + 100 * H
def remaining_value : ℕ → ℕ := λ checks_spent, 1800 - (checks_spent * 50)
def average_value : ℕ → ℕ → ℕ := λ remaining_checks remaining_amount, remaining_amount / remaining_checks

-- Main statement
theorem tourist_check_total (F H : ℕ) (h1 : total_value F H = 1800)
                            (checks_spent : ℕ)
                            (remaining_checks : ℕ)
                            (remaining_amount : ℕ)
                            (h2 : checks_spent = 24)
                            (h3 : remaining_value checks_spent = 600)
                            (h4 : average_value remaining_checks remaining_amount = 100)
                            (h5 : remaining_checks = 6) :
    F + H = 30 := sorry

end tourist_check_total_l54_54128


namespace proof_problem_l54_54215

variables {m n : ℝ}

theorem proof_problem (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 2 * m * n) :
  (mn : ℝ) ≥ 1 ∧ (m^2 + n^2 ≥ 2) :=
  sorry

end proof_problem_l54_54215


namespace log_base_9_of_3_l54_54530

theorem log_base_9_of_3 : log 9 3 = 1 / 2 := by
  -- Contextual conditions specified in Lean
  have h1 : 9 = 3 ^ 2 := by norm_num
  have h2 : log 3 9 = 2 := by
    rw [h1, log_pow, log_self]
    norm_num
  rw [log_div_log, h2]
  norm_num

-- helper lemmas
lemma log_pow (a b n : ℝ) (h : 0 < a) (h' : a ≠ 1) : log b (a ^ n) = n * log b a := by
  sorry

lemma log_self (a : ℝ) (h : 0 < a) (h' : a ≠ 1) : log a a = 1 := by
  sorry

lemma log_div_log {a b : ℝ} (h : 0 < a) (h' : a ≠ 1) (h₁ : 0 < b) (h₂ : b ≠ 1) :
   log a b = 1 / log b a := by
  sorry

end log_base_9_of_3_l54_54530


namespace carpet_square_cost_l54_54475

theorem carpet_square_cost:
  ∀ (width_floor height_floor side_carpet n_carpet total_cost cost_per_carpet : ℕ),
    width_floor = 6 →
    height_floor = 10 →
    side_carpet = 2 →
    total_cost = 225 →
    n_carpet = (width_floor * height_floor) / (side_carpet * side_carpet) →
    cost_per_carpet = total_cost / n_carpet →
    cost_per_carpet = 15 :=
begin
  intros,
  sorry
end

end carpet_square_cost_l54_54475


namespace problem_statement_l54_54111

-- Definition of the conditions
def group := { boys := 3, girls := 2 }
def event_1 := "Exactly 1 boy"
def event_2 := "Exactly 2 girls"
def selection := 2 -- Random selection of 2 students

-- The mutual exclusivity and non-complementarity conditions.
-- 'mutually_exclusive' means the events cannot both happen.
def mutually_exclusive : Prop := 
  (event_1 "and" event_2) = false

-- 'not_complementary' means that the union of the events is not the entire sample space.
def not_complementary : Prop :=
  event_1 "or" event_2 ≠ sample_space

-- The main proof statement
theorem problem_statement : mutually_exclusive ∧ not_complementary := by
  sorry

end problem_statement_l54_54111


namespace skittles_total_correct_l54_54409

def number_of_students : ℕ := 9
def skittles_per_student : ℕ := 3
def total_skittles : ℕ := 27

theorem skittles_total_correct : number_of_students * skittles_per_student = total_skittles := by
  sorry

end skittles_total_correct_l54_54409


namespace charles_draws_yesterday_after_work_l54_54911

theorem charles_draws_yesterday_after_work :
  ∀ (initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work : ℕ),
    initial_papers = 20 →
    today_drawn = 6 →
    yesterday_drawn_before_work = 6 →
    current_papers_left = 2 →
    (initial_papers - (today_drawn + yesterday_drawn_before_work) - yesterday_drawn_after_work = current_papers_left) →
    yesterday_drawn_after_work = 6 :=
by
  intros initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work
  intro h1 h2 h3 h4 h5
  sorry

end charles_draws_yesterday_after_work_l54_54911


namespace triangle_problems_l54_54229

open Real

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}

def triangle_sides_and_angles (a b c : ℝ) (A B C : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π

def perpendicular (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0

noncomputable def area_of_triangle (a b c : ℝ) (A : ℝ) : ℝ :=
  1 / 2 * b * c * sin A

theorem triangle_problems
  (h1 : triangle_sides_and_angles a b c A B C)
  (h2 : m = (1, 1))
  (h3 : n = (sqrt 3 / 2 - sin B * sin C, cos B * cos C))
  (h4 : perpendicular m n)
  (h5 : a = 1)
  (h6 : b = sqrt 3 * c) :
  A = π / 6 ∧ area_of_triangle a b c A = sqrt 3 / 4 :=
by
  sorry

end triangle_problems_l54_54229


namespace vasya_grades_l54_54680

-- Given conditions
constants (a1 a2 a3 a4 a5 : ℕ)
axiom grade_median : a3 = 4
axiom grade_sum : a1 + a2 + a3 + a4 + a5 = 19
axiom most_A_grades : ∀ (n : ℕ), n ≠ 5 → (∃ m, m > 0 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 5 → (if a1 = n ∨ a2 = n ∨ a3 = n ∨ a4 = n ∨ a5 = n then m > 1 else m = 0))

-- Prove that the grades are (2, 3, 4, 5, 5)
theorem vasya_grades : (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨ 
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) := 
by sorry

end vasya_grades_l54_54680


namespace check_triangle_l54_54828

theorem check_triangle (a b c : ℕ) : (a, b, c) ∈ {(2, 3, 4)} ↔ (a + b > c ∧ a + c > b ∧ b + c > a) := by
  sorry

end check_triangle_l54_54828


namespace number_of_possible_values_l54_54974

theorem number_of_possible_values (a : ℤ) 
  (h₀ : ∃ x ∈ Ioo (-1 : ℝ) 1, exp x + x - a = 0) : 
  (∃ S : Finset ℤ, S.card = 4 ∧ ∀ n ∈ S, n = a) :=
sorry

end number_of_possible_values_l54_54974


namespace solve_inequality_system_l54_54368

-- Define the conditions and the correct answer
def system_of_inequalities (x : ℝ) : Prop :=
  (5 * x - 1 > 3 * (x + 1)) ∧ ((1/2) * x - 1 ≤ 7 - (3/2) * x)

def solution_set (x : ℝ) : Prop :=
  2 < x ∧ x ≤ 4

-- State that solving the system of inequalities is equivalent to the solution set
theorem solve_inequality_system (x : ℝ) : system_of_inequalities x ↔ solution_set x :=
  sorry

end solve_inequality_system_l54_54368


namespace problem_l54_54630

open Function

variables {ι : Type*} [LinearOrder ι] {a : ι → ℝ} (n : ℕ)
  (h_sort : ∀ i j, i < j → a i < a j) (h_pos : ∀ i, 0 < a i)
  (h_len : ∃ S : Finset ι, S.card = n)

noncomputable def f (x : ℝ) : ℝ :=
∑ i in S, a i / (a i - x)

theorem problem (h : ∃ S : Finset ι, S.card = n) :
  ∃ (roots : Fin n → ℝ), ∀ i, f (roots i) = 2015 ∧ (∀ j, j ≠ i → roots i ≠ roots j) :=
sorry

end problem_l54_54630


namespace triangle_median_and_midpoint_l54_54645

noncomputable def midpoint (A B : Point) : Point :=
  (A + B) / 2

theorem triangle_median_and_midpoint (X Y Z W : Point)
  (h1 : distance X Y = 6)
  (h2 : distance Y Z = 8)
  (h3 : angle X Y Z = π / 2)
  (h4 : W = midpoint Y Z) :
  distance W Z = 4 ∧ distance X W = 5 :=
by
  sorry

end triangle_median_and_midpoint_l54_54645


namespace hexagon_coloring_l54_54928

theorem hexagon_coloring (colors : Finset ℕ) (vertices : Finset ℕ)
  (h_colors : colors.card = 7) (h_vertices : vertices.card = 6) :
  ∃ n : ℕ, n = 27216 ∧ 
  ∀ (coloring : vertices → colors), 
    (∀ (u v : ℕ), 
      u ≠ v → diagonal u v → coloring u ≠ coloring v)
  :=
by sorry

end hexagon_coloring_l54_54928


namespace find_smallest_N_l54_54057

-- Definitions for the conditions
def total_balls : ℕ := 111
def colors := {red, green, blue, white}

-- Given assumptions
def condition (N : ℕ) : Prop := 
  ∀ (R G B W : ℕ), R + G + B + W = total_balls →
  (∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    ¬ (R ≥ x ∧ G ≥ y ∧ B ≥ z ∧ W ≥ 100 - x - y - z)) →
  N ≤ total_balls

-- Define the main problem statement
theorem find_smallest_N :
  ∃ N, N = 88 ∧ condition N := 
sorry

end find_smallest_N_l54_54057


namespace math_problem_l54_54217

noncomputable def x_y_solution (x y : ℝ) : Prop :=
  (abs (x - 2 * y - 3) + (y - 2 * x) ^ 2 = 0) → (x + y = -3)

theorem math_problem (x y : ℝ) : x_y_solution x y :=
begin
  sorry
end

end math_problem_l54_54217


namespace evaluate_81_power_5_div_4_l54_54185

-- Define the conditions
def base_factorized : ℕ := 3 ^ 4
def power_rule (b : ℕ) (m n : ℝ) : ℝ := (b : ℝ) ^ m ^ n

-- Define the primary calculation
noncomputable def power_calculation : ℝ := 81 ^ (5 / 4)

-- Prove that the calculation equals 243
theorem evaluate_81_power_5_div_4 : power_calculation = 243 := 
by
  have h1 : base_factorized = 81 := by sorry
  have h2 : power_rule 3 4 (5 / 4) = 3 ^ 5 := by sorry
  have h3 : 3 ^ 5 = 243 := by sorry
  have h4 : power_calculation = power_rule 3 4 (5 / 4) := by sorry
  rw [h1, h2, h3, h4]
  exact h3

end evaluate_81_power_5_div_4_l54_54185


namespace average_age_of_team_l54_54775

def age_distribution : List (ℕ × ℕ) := [(13, 2), (14, 6), (15, 8), (16, 3), (17, 2), (18, 1)]

theorem average_age_of_team : 
  let total_sum_of_ages := (13 * 2) + (14 * 6) + (15 * 8) + (16 * 3) + (17 * 2) + (18 * 1),
      total_number_of_players := 2 + 6 + 8 + 3 + 2 + 1
  in total_sum_of_ages / total_number_of_players = 15 := 
by
  sorry

end average_age_of_team_l54_54775


namespace number_of_factors_l54_54343

theorem number_of_factors (b : ℕ) (h1 : b = 42) :
  let M := b^4 + 4 * b^3 + 6 * b^2 + 4 * b + 1 in
  nat.factors_count M = 5 :=
by
  sorry

end number_of_factors_l54_54343


namespace bottles_more_than_apples_l54_54110

def regular_soda : ℕ := 72
def diet_soda : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := regular_soda + diet_soda

theorem bottles_more_than_apples : total_bottles - apples = 26 := by
  -- Proof will go here
  sorry

end bottles_more_than_apples_l54_54110


namespace medium_size_shoes_initially_stocked_l54_54123

variable {M : ℕ}  -- The number of medium-size shoes initially stocked

noncomputable def initial_pairs_eq (M : ℕ) := 22 + M + 24
noncomputable def shoes_sold (M : ℕ) := initial_pairs_eq M - 13

theorem medium_size_shoes_initially_stocked :
  shoes_sold M = 83 → M = 26 :=
by
  sorry

end medium_size_shoes_initially_stocked_l54_54123


namespace semicircle_radius_l54_54450

noncomputable def π : ℝ := Real.pi

theorem semicircle_radius (P : ℝ) (hP : P = 216) : 
  ∃ r : ℝ, r ≈ 42.01 ∧ P = r * (π + 2) :=
by
  have r := 216 / (π + 2)
  use r
  rw [hP]
  rw [Real.pi]
  simp
  split
  { assume h
    norm_num at h
    norm_num
  {
    sorry -- complete the proof.
  } 

end semicircle_radius_l54_54450


namespace intersection_of_A_and_B_l54_54588

open Set

def A := { x : ℕ | -1 < x ∧ x < 3 }
def B := { x : ℤ | -2 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end intersection_of_A_and_B_l54_54588


namespace lambda_range_l54_54244

noncomputable def arithmetic_seq_a (n : ℕ) : ℝ := 2 * n + 1

def S (n : ℕ) : ℝ := (n * (arithmetic_seq_a 1 + arithmetic_seq_a n)) / 2

def b (n : ℕ) (λ : ℝ) : ℝ := (arithmetic_seq_a n)^2 + λ * (arithmetic_seq_a n)

theorem lambda_range (λ : ℝ) : (∀ n : ℕ, n ∈ ℕ → b (n+1) λ > b n λ) → λ > -4 := sorry

end lambda_range_l54_54244


namespace problem1_problem2_problem3_l54_54906

-- Proof statement for Problem 1
theorem problem1 : 23 * (-5) - (-3) / (3 / 108) = -7 := 
by 
  sorry

-- Proof statement for Problem 2
theorem problem2 : (-7) * (-3) * (-0.5) + (-12) * (-2.6) = 20.7 := 
by 
  sorry

-- Proof statement for Problem 3
theorem problem3 : ((-1 / 2) - (1 / 12) + (3 / 4) - (1 / 6)) * (-48) = 0 := 
by 
  sorry

end problem1_problem2_problem3_l54_54906


namespace quadratic_roots_range_l54_54275

theorem quadratic_roots_range (k : ℝ) : (x^2 - 6*x + k = 0) → k < 9 := 
by
  sorry

end quadratic_roots_range_l54_54275


namespace coefficient_x_term_l54_54033

theorem coefficient_x_term :
  let expr := (x^2 - x - 2)^3 in
  expr.coeff (monomial 1) = -12 :=
by sorry

end coefficient_x_term_l54_54033


namespace exists_nested_rectangles_l54_54921

theorem exists_nested_rectangles (rectangles : ℕ × ℕ → Prop) :
  (∀ n m : ℕ, rectangles (n, m)) → ∃ (n1 m1 n2 m2 : ℕ), n1 ≤ n2 ∧ m1 ≤ m2 ∧ rectangles (n1, m1) ∧ rectangles (n2, m2) :=
by {
  sorry
}

end exists_nested_rectangles_l54_54921


namespace factor_product_modulo_l54_54818

theorem factor_product_modulo (h1 : 2021 % 23 = 21) (h2 : 2022 % 23 = 22) (h3 : 2023 % 23 = 0) (h4 : 2024 % 23 = 1) (h5 : 2025 % 23 = 2) :
  (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 0 :=
by
  sorry

end factor_product_modulo_l54_54818


namespace part1_part2_l54_54778

noncomputable def f (x a : ℝ) := 5 - |x + a| - |x - 2|

theorem part1 : 
  (∀ x, f x 1 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
sorry

theorem part2 :
  (∀ a, (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2)) :=
sorry

end part1_part2_l54_54778


namespace no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l54_54841

theorem no_sequence_of_14_consecutive_divisible_by_some_prime_le_11 :
  ¬ ∃ n : ℕ, ∀ k : ℕ, k < 14 → ∃ p ∈ [2, 3, 5, 7, 11], (n + k) % p = 0 :=
by
  sorry

end no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l54_54841


namespace system1_solution_system2_solution_l54_54019

-- Problem 1
theorem system1_solution (x z : ℤ) (h1 : 3 * x - 5 * z = 6) (h2 : x + 4 * z = -15) : x = -3 ∧ z = -3 :=
by
  sorry

-- Problem 2
theorem system2_solution (x y : ℚ) 
 (h1 : ((2 * x - 1) / 5) + ((3 * y - 2) / 4) = 2) 
 (h2 : ((3 * x + 1) / 5) - ((3 * y + 2) / 4) = 0) : x = 3 ∧ y = 2 :=
by
  sorry

end system1_solution_system2_solution_l54_54019


namespace train_length_is_109_98_l54_54130

-- Definitions based on the conditions
def train_speed_kmh : ℝ := 72
def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
def crossing_time : ℝ := 12.499
def bridge_length : ℝ := 140

-- Theorem statement that length of the train is 109.98 meters
theorem train_length_is_109_98 : 
  let distance_covered := train_speed_ms * crossing_time in
  let train_length := distance_covered - bridge_length in
  train_length = 109.98 :=
by
  -- Proof will be provided here
  sorry

end train_length_is_109_98_l54_54130


namespace possible_remainders_2012_l54_54386

theorem possible_remainders_2012 : 
  ∀ (signs : Fin 2012 → Bool), 
  let expr := (Finset.range 2012).sum (λ i, if signs i then (i + 1 : ℤ) else -(i + 1 : ℤ)) in 
  ∃ (remainders : Finset ℤ), 
  ∀ r ∈ remainders, 0 ≤ r ∧ r < 2012 ∧ r % 2 = 0 ∧ remainders.card = 1006 :=
sorry

end possible_remainders_2012_l54_54386


namespace domain_of_f_eq_A_range_of_m_for_B_subset_A_range_of_m_for_B_empty_l54_54720

def f (x : ℝ) : ℝ := Real.sqrt (2 + x) + Real.log (4 - x)

def A : Set ℝ := { x | -2 ≤ x ∧ x < 4 }

def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

theorem domain_of_f_eq_A : 
  { x | ∃ (y : ℝ), f x = y } = A :=
sorry

theorem range_of_m_for_B_subset_A :
  {m : ℝ | B m ⊆ A ∧ B m ≠ ∅} = set.Iio (5/2) :=
sorry

-- Another theorem for the case where B is empty when m < 2
theorem range_of_m_for_B_empty :
  {m : ℝ | B m = ∅} = set.Iio 2 :=
sorry

end domain_of_f_eq_A_range_of_m_for_B_subset_A_range_of_m_for_B_empty_l54_54720


namespace Alice_wins_no_matter_what_Bob_does_l54_54885

theorem Alice_wins_no_matter_what_Bob_does (a b c : ℝ) :
  (∀ d : ℝ, (b + d) ^ 2 - 4 * (a + d) * (c + d) ≤ 0) → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intro h
  sorry

end Alice_wins_no_matter_what_Bob_does_l54_54885


namespace min_value_f_l54_54924

def f (x : ℝ) : ℝ := 5 * x^2 + 10 * x + 20

theorem min_value_f : ∃ (x : ℝ), f x = 15 :=
by
  sorry

end min_value_f_l54_54924


namespace number_of_perpendicular_plane_pairs_l54_54978

universe u

-- Defining the basic elements involved
variables {Point : Type u}

-- Four points defining the rectangle
variables (A B C D P : Point)

-- Perpendicularity in the context of the problem
def is_perpendicular_to (x y : Plane) : Prop := sorry -- Define perpendicular relationship

-- Points defining rectangle ABCD and point P which is perpendicular to the plane ABCD
def Rectangle_ABCD := sorry
def PD_perpendicular_to_ABCD := sorry

-- Theorem stating the problem
theorem number_of_perpendicular_plane_pairs (h_PD_perpendicular_to_ABCD : PD_perpendicular_to_ABCD) : 
  number_of_pairs_perpendicular := 6 :=
sorry

end number_of_perpendicular_plane_pairs_l54_54978


namespace folding_paper_proves_desired_properties_l54_54009

noncomputable def can_fold_paper_to_align_points (P: Type) (points_on_line : list P) (three_points : list P) : Prop :=
  ∃ fold_strategy : list (P → P), 
    ∀ point ∈ points_on_line ++ three_points,
      point_is_at_piercing_position (apply_folds fold_strategy point) 

-- Auxiliary function to signify that all input points coincide when folded and pierced at a single point.
def point_is_at_piercing_position (point_position : P) : Prop := 
  ∃ common_position : P, point_position = common_position

-- Function to apply a sequence of folds to a point.
noncomputable def apply_folds (folds : list (P → P)) (point : P) : P :=
  folds.foldl (λ acc fold => fold acc) point

-- Main theorem statement
theorem folding_paper_proves_desired_properties (P: Type) (points_on_line : list P) (three_points : list P) :
  (length points_on_line ≥ 2 ∨ length three_points = 3) →
  can_fold_paper_to_align_points P points_on_line three_points :=
by {
  sorry
}

end folding_paper_proves_desired_properties_l54_54009


namespace necessary_but_not_sufficient_condition_for_inequality_l54_54134

theorem necessary_but_not_sufficient_condition_for_inequality 
    {a b c : ℝ} (h : a * c^2 ≥ b * c^2) : ¬(a > b → (a * c^2 < b * c^2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_for_inequality_l54_54134


namespace derivative_of_3x_squared_l54_54384

theorem derivative_of_3x_squared : 
  ∀ (x : ℝ), deriv (λ x : ℝ, 3 * x ^ 2) x = 6 * x := 
by sorry

end derivative_of_3x_squared_l54_54384


namespace cos_neg_30_eq_sqrt3_div_2_l54_54403

-- Given known value: cos 30 degrees
def cos_30_deg : ℝ := Real.cos (Real.pi / 6)

theorem cos_neg_30_eq_sqrt3_div_2 : Real.cos (-Real.pi / 6) = cos_30_deg := by
  -- Skipping the proof intentionally
  sorry

end cos_neg_30_eq_sqrt3_div_2_l54_54403


namespace apples_selling_price_l54_54423

theorem apples_selling_price (total_harvest : ℕ) (juice : ℕ) (restaurant : ℕ) (bag_weight : ℕ) (total_revenue : ℤ) (sold_bags : ℕ) :
  total_harvest = 405 →
  juice = 90 →
  restaurant = 60 →
  bag_weight = 5 →
  total_revenue = 408 →
  sold_bags = (total_harvest - juice - restaurant) / bag_weight →
  total_revenue / sold_bags = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end apples_selling_price_l54_54423


namespace vasya_grades_l54_54676

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧ 
    (grades.nthLe 2 (by linarith) = 4) ∧ 
    (grades.sum = 19) ∧
    (grades.count 5 > List.foldl (fun acc n => if n ≠ 5 then acc + grades.count n else acc) 0 grades) ∧ 
    (grades.sorted = [2, 3, 4, 5, 5]) :=
sorry

end vasya_grades_l54_54676


namespace probability_first_le_second_l54_54463

theorem probability_first_le_second : 
  let cards := {1, 2, 3, 4, 5}
  let outcomes := (cards × cards).filter (λ (xy : ℕ × ℕ), xy.1 <= xy.2)
  (outcomes.size / (cards.size * cards.size) : ℚ) = 3 / 5 := by
  sorry

end probability_first_le_second_l54_54463


namespace odot_property_l54_54513

def odot (x y : ℤ) := 2 * x + y

theorem odot_property (a b : ℤ) (h : odot a (-6 * b) = 4) : odot (a - 5 * b) (a + b) = 6 :=
by
  sorry

end odot_property_l54_54513


namespace repeating_base_k_fraction_representation_l54_54942

theorem repeating_base_k_fraction_representation (k : ℕ) (h_pos : k > 0) :
  (∃ k : ℕ, k > 0 ∧ (∑ n in (Finset.range 1000), (3 * k^n + 1 * k^(n+1) / k^(2*n + 1)) ) = 12 / 65) ↔ k = 17 :=
by
  have geom_series_summation : (∑ n in (Finset.range 1000), (3 / k^(2*n + 1) + 1 / k^(2*n + 2))) = 3 * k / (k^2 -1) + 1 / (k^2 - 1),
  sorry
  have equation_formed := ( 3 * k + 1 ) / (k^2 - 1) = 12 / 65 ,
  sorry
  have solution := 12 * k^2 - 195 * k - 77 = 0,
  sorry
  use [k]
  exact 17,
  sorry

end repeating_base_k_fraction_representation_l54_54942


namespace find_b_c_l54_54045

variable {b c : ℝ}

theorem find_b_c
  (h_bc_pos : (0 < b) ∧ (0 < c))
  (h_prod : ∀ (x1 x2 x3 x4 : ℝ), 
    (x1 * x2 * x3 * x4 = 1) ∧
    (x1 + x2 = -2 * b) ∧ (x1 * x2 = c) ∧
    (x3 + x4 = -2 * c) ∧ (x3 * x4 = b)) :
  b = 1 ∧ c = 1 := 
by
  sorry

end find_b_c_l54_54045


namespace train_length_l54_54877

/-- Definition of the problem conditions --/
def speed_kmph : ℝ := 360
def time_sec : ℝ := 30
def speed_mps : ℝ := (speed_kmph * 1000) / 3600

theorem train_length :
  let speed := speed_mps in
  let distance := speed * time_sec in
  distance = 3000 :=
by
  sorry

end train_length_l54_54877


namespace graph_single_point_c_eq_7_l54_54376

theorem graph_single_point_c_eq_7 (x y : ℝ) (c : ℝ) :
  (∃ p : ℝ × ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * y^2 + 6 * x - 8 * y + c = 0 ↔ (x, y) = p) →
  c = 7 :=
by
  sorry

end graph_single_point_c_eq_7_l54_54376


namespace max_frac_sum_l54_54225

theorem max_frac_sum (n a b c d : ℕ) (hn : 1 < n) (hab : 0 < a) (hcd : 0 < c)
    (hfrac : (a / b) + (c / d) < 1) (hsum : a + c ≤ n) :
    (∃ (b_val : ℕ), 2 ≤ b_val ∧ b_val ≤ n ∧ 
    1 - 1 / (b_val * (b_val * (n + 1 - b_val) + 1)) = 
    1 - 1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1))) :=
sorry

end max_frac_sum_l54_54225


namespace projection_theorem_sine_rule_cosine_rule_l54_54690

variable (a b c : ℝ)
variable (A B C : ℝ)
hypothesis (H_triangle : A + B + C = Real.pi)

theorem projection_theorem (H_cos_A : a * Real.cos B + b * Real.cos A) : 
  c = a * Real.cos B + b * Real.cos A := sorry

theorem sine_rule (H_sine_A : a / Real.sin A) (H_sine_B : b / Real.sin B) : 
  a / Real.sin A = b / Real.sin B := sorry

theorem cosine_rule (H_cosine : c * c) : 
  c * c = a * a + b * b - 2 * a * b * Real.cos C := sorry

end projection_theorem_sine_rule_cosine_rule_l54_54690


namespace no_real_roots_other_than_zero_l54_54206

theorem no_real_roots_other_than_zero (k : ℝ) (h : k ≠ 0):
  ¬(∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 = 0) :=
by
  sorry

end no_real_roots_other_than_zero_l54_54206


namespace alice_distance_regular_hexagon_l54_54863

noncomputable def distance_from_start (hex_side_length : ℝ) (total_distance_walked : ℝ) : ℝ :=
  let π := Real.pi
  let x := -0.5 * hex_side_length in
  let y := -3.5 * (Math.sqrt 3) * hex_side_length in
  (x^2 + y^2).sqrt

theorem alice_distance_regular_hexagon :
  let hex_side_length := 3
  let total_distance_walked := 10
  distance_from_start hex_side_length total_distance_walked = (Math.sqrt 37) :=
by
  -- Assume hex_side_length = 3 and total_distance_walked = 10
  let hex_side_length := 3
  let total_distance_walked := 10

  -- Calculate her distance using the given formula
  let π := Real.pi
  let x := -0.5 * hex_side_length
  let y := -3.5 * (Math.sqrt 3) * hex_side_length
  let distance := (x^2 + y^2).sqrt
  
  -- Show that this distance is equal to sqrt(37)
  show True, from sorry

end alice_distance_regular_hexagon_l54_54863


namespace blocks_leftover_zero_l54_54917

theorem blocks_leftover_zero : 
  ∃ n : ℕ, (∑ k in finset.range (n + 1), k = 36) ∧ 36 - (∑ k in finset.range (n + 1), k) = 0 :=
by
  sorry

end blocks_leftover_zero_l54_54917


namespace square_area_l54_54816

theorem square_area (side_length : ℕ) (h : side_length = 17) : side_length * side_length = 289 :=
by sorry

end square_area_l54_54816


namespace choose_five_representatives_choose_five_specific_girl_choose_five_at_least_two_boys_group_into_three_groups_l54_54408
open Nat

/-- Number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := if h : n < k then 0 else n.factorial / ((n - k).factorial * k.factorial)

/-- How many ways are there to choose 5 representatives from 4 boys and 5 girls --/
theorem choose_five_representatives :
  choose 9 5 = 126 :=
by sorry

/-- How many ways are there to choose 5 representatives with 2 boys, 3 girls, and one specific girl included --/
theorem choose_five_specific_girl :
  (choose 4 2) * (choose 4 2) = 36 :=
by sorry

/-- How many ways are there to choose 5 representatives with at least 2 boys included --/
theorem choose_five_at_least_two_boys :
  (choose 4 2 * choose 5 3) + (choose 4 3 * choose 5 2) + (choose 4 4 * choose 5 1) = 105 :=
by sorry

/-- How many ways are there to group 9 people into three groups with 4, 3, and 2 people respectively --/
theorem group_into_three_groups :
  (choose 9 4) * (choose 5 3) = 1260 :=
by sorry

end choose_five_representatives_choose_five_specific_girl_choose_five_at_least_two_boys_group_into_three_groups_l54_54408


namespace ones_smaller_than_tens_count_divisible_by_5_count_l54_54507

def digits := {0, 1, 2, 3, 4, 5}
def is_three_digit_number (n : ℕ) : Prop :=
  n / 100 ≥ 1 ∧ n / 100 < 10

def no_repeating_digits (n : ℕ) : Prop :=
  let d := [n / 100, (n / 10) % 10, n % 10]
  list.nodup d ∧ list.all d (λ x, x ∈ digits)

def ones_smaller_than_tens (n : ℕ) : Prop :=
  (n % 10) < ((n / 10) % 10)

def divisible_by_5 (n : ℕ) : Prop :=
  (n % 10) = 0 ∨ (n % 10) = 5

theorem ones_smaller_than_tens_count :
  fintype.card {n : ℕ // is_three_digit_number n ∧ no_repeating_digits n ∧ ones_smaller_than_tens n} = 63 :=
by sorry

theorem divisible_by_5_count :
  fintype.card {n : ℕ // is_three_digit_number n ∧ no_repeating_digits n ∧ divisible_by_5 n} = 36 :=
by sorry

end ones_smaller_than_tens_count_divisible_by_5_count_l54_54507


namespace find_M_l54_54919

def grid_conditions :=
  ∃ (M : ℤ), 
  ∀ d1 d2 d3 d4, 
    (d1 = 22) ∧ (d2 = 6) ∧ (d3 = -34 / 6) ∧ (d4 = (8 - M) / 6) ∧
    (10 = 32 - d2) ∧ 
    (16 = 10 + d2) ∧ 
    (-2 = 10 - d2) ∧
    (32 - M = 34 / 6 * 6) ∧ 
    (M = -34 / 6 - (-17 / 3))

theorem find_M : grid_conditions → ∃ M : ℤ, M = 17 :=
by
  intros
  existsi (17 : ℤ) 
  sorry

end find_M_l54_54919


namespace normal_pdf_even_l54_54044

def normal_pdf (mu sigma x : ℝ) : ℝ :=
  - (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(x^2) / 2)

theorem normal_pdf_even (mu sigma : ℝ) (h_mu : mu = 0) (h_sigma : sigma = -1) :
  ∀ x : ℝ, normal_pdf mu sigma x = normal_pdf mu sigma (-x) :=
by
  sorry

end normal_pdf_even_l54_54044


namespace base_six_product_l54_54428

def base_six_to_base_ten (n : List ℕ) : ℕ :=
n.reverse.enum.prod.map (λ (d, i) -> d * 6^i)

def base_ten_to_base_six (n : ℕ) : List ℕ :=
if n = 0 then [0] else
  let rec convert (n : ℕ) :=
    if n = 0 then [] else
    (n % 6) :: convert (n / 6)
  convert n

theorem base_six_product :
  let n1 := base_six_to_base_ten [2, 3, 1]
  let n2 := base_six_to_base_ten [4, 1]
  let product_base_ten := n1 * n2
  let result_base_six := base_ten_to_base_six product_base_ten
  result_base_six.reverse = [2, 3, 3, 2] :=
by {
  sorry
}

end base_six_product_l54_54428


namespace shaded_area_correct_l54_54297

noncomputable def area_shaded_regions : ℝ :=
  36 + 18 * Real.pi

theorem shaded_area_correct :
  ∀ (AB CD : ℝ) (O A B C D : Point)
  (diameter : ∀ X Y, distance X Y = AB)
  (intersections : ∀ X Y, X ≠ Y → ∃ Z, Z ≠ X ∧ Z ≠ Y)
  (right_angle_O : ∠AOB = π / 2)
  (radius : AB / 2 = 6)
  (horizontal : ∀ X ∈ {B, D}, ∃ y, (y, X) ∈ set_of_is_on_diameter)
  (vertical : ∀ X ∈ {A, C}, ∃ y, (y, X) ∈ set_of_is_on_diameter)
  (isosceles_right_triangle : is_isosceles_right_triangle O A C)
  (circle_area_sector : ∀ θ r, area_sector θ r = θ / (2 * π) * (π * r^2)),

  area_shaded_regions = 
    area_triangle O A C + area_triangle O B D +
    area_sector (π / 2) 6 + area_sector (π / 2) 6 := 
sorry

end shaded_area_correct_l54_54297


namespace quadratic_passes_through_constant_point_l54_54641

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - m * x + 2 * m + 1

theorem quadratic_passes_through_constant_point :
  ∀ m : ℝ, f m 2 = 13 :=
by
  intro m
  unfold f
  simp
  rfl

end quadratic_passes_through_constant_point_l54_54641


namespace not_prime_for_large_n_l54_54761

theorem not_prime_for_large_n {n : ℕ} (h : n > 1) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end not_prime_for_large_n_l54_54761


namespace intersection_A_B_l54_54332

-- Define the set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, x + 1) }

-- Define the set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, -2*x + 4) }

-- State the theorem to prove A ∩ B = {(1, 2)}
theorem intersection_A_B : A ∩ B = { (1, 2) } :=
by
  sorry

end intersection_A_B_l54_54332


namespace area_ABE_9_l54_54414

variable {ABC : Type} [triangle ABC]
variable (A B C D E F : point)
variable [on_side A B C D E F]
variable (area_ABC : area ABC = 15)
variable (AD : length A D = 3)
variable (DB : length D B = 2)
variable (equal_areas : area (triangle_set A B E) = area (quadrilateral_set D B E F))

theorem area_ABE_9
  (h₁ : area_ABC)
  (h₂ : AD)
  (h₃ : DB)
  (h₄ : equal_areas) :
  area (triangle_set A B E) = 9 := by
  sorry

end area_ABE_9_l54_54414


namespace vivians_mail_in_august_l54_54815

-- Definitions based on the conditions provided
def mail_july : ℕ := 40
def business_days_august : ℕ := 22
def weekend_days_august : ℕ := 9

-- Lean 4 statement to prove the equivalent proof problem
theorem vivians_mail_in_august :
  let mail_business_days := 2 * mail_july
  let total_mail_business_days := business_days_august * mail_business_days
  let mail_weekend_days := mail_july / 2
  let total_mail_weekend_days := weekend_days_august * mail_weekend_days
  total_mail_business_days + total_mail_weekend_days = 1940 := by
  sorry

end vivians_mail_in_august_l54_54815


namespace problem_solution_l54_54731

noncomputable def proof_problem (x y z : ℝ) (h1 : 5^x = t) (h2 : 9^y = t) (h3 : 225^z = t) : Prop :=
  (1 / z = 2 / x + 1 / y)

-- Now we state the theorem we need to prove
theorem problem_solution (x y z t : ℝ) (h1 : 5^x = t) (h2 : 9^y = t) (h3 : 225^z = t) : 
  1 / z = 2 / x + 1 / y :=
sorry

end problem_solution_l54_54731


namespace problem1_problem2_l54_54842

-- Problem 1: Proving the expression of f for given functional equation
theorem problem1 (f : ℝ → ℝ) (h : ∀ x, f (1 - real.sqrt x) = x) :
  ∀ x, x ≤ 1 → f x = x^2 - 2*x + 1 :=
sorry

-- Problem 2: Proving the expression of a linear f for given functional equation
theorem problem2 (f : ℝ → ℝ) [linear_map ℝ f] (h : ∀ x, f (f x) = 4*x + 3) :
  ∃ k b : ℝ, (f = λ x, k * x + b) ∧ (k = 2 ∧ b = 1 ∨ k = -2 ∧ b = -3) :=
sorry

end problem1_problem2_l54_54842


namespace find_constants_find_min_and_max_values_l54_54991

/-- Define the function -/
def f (x : ℝ) (a b : ℝ) := a * x^3 - 5 * x^2 - b * x

theorem find_constants (a b : ℝ):
  (∀ x : ℝ, 3 * a * x^2 - 10 * x - b = 0 → x = 3) ∧
  (a * 1^3 - 5 * 1^2 - b * 1 = -1)
  → a = 1 ∧ b = -3 :=
begin
  intro h,
  sorry
end

theorem find_min_and_max_values (a b x : ℝ) (h₁ : a = 1) (h₂ : b = -3):
  (f x a b) = x^3 - 5 * x^2 + 3 * x → (f 3 1 (-3) = -9) ∧ (f 4 1 (-3) = 0) :=
begin
  intros,
  sorry
end

end find_constants_find_min_and_max_values_l54_54991


namespace hyperbola_range_of_a_l54_54114

theorem hyperbola_range_of_a (a : ℝ) (h : a > 0) :
  ∃ A B : ℝ×ℝ, (A.1^2 / a^2 - A.2^2 = 1) ∧ (B.1^2 / a^2 - B.2^2 = 1) ∧ 
  ∃ l : ℝ → ℝ, (l (left_focus(a)) = left_focus(a)) ∧ 
  (left_focus(a) = line_a) ∧ (left_focus(a) = line_b) ∧ 
  dist (A,B) = 4 ∧
  (∃! l : ℝ → ℝ, (l (left_focus(a)) = left_focus(a))) →
  (a ∈ (Set.Ioo 0 (1/2)) ∨ a ∈ (Set.Ioi 2)) := sorry

end hyperbola_range_of_a_l54_54114


namespace value_of_x_l54_54823

theorem value_of_x :
  (32^32 + 32^32 + 32^32 + 32^32 + 32^32 + 32^32 = 2^((Real.log 6 / Real.log 2) + 160)) :=
by
  sorry

end value_of_x_l54_54823


namespace tickets_difference_l54_54492

def number_of_tickets_for_toys := 31
def number_of_tickets_for_clothes := 14

theorem tickets_difference : number_of_tickets_for_toys - number_of_tickets_for_clothes = 17 := by
  sorry

end tickets_difference_l54_54492


namespace tangent_to_incircle_l54_54089

variables {R : Type*} [Real R] (x x₀ y y₀ z z₀ : R) (α β γ : ℝ)

def tangent_eq (x₀ y₀ z₀ x y z α β γ : R) : Prop :=
  (x / sqrt x₀) * cos (α / 2) + (y / sqrt y₀) * cos (β / 2) + (z / sqrt z₀) * cos (γ / 2) = 0

theorem tangent_to_incircle 
  (x₀ y₀ z₀: R)
  (h_incircle : -- some conditions here to state that x0, y0, z0 are from an incircle )
  :
  tangent_eq x₀ y₀ z₀ x y z α β γ :=
sorry

end tangent_to_incircle_l54_54089


namespace product_of_ten_numbers_cannot_end_in_1580_l54_54572

-- Define the condition that the sum of any four of the ten numbers is even
def all_four_sum_even (S : Finset ℕ) : Prop :=
  ∀ (a b c d : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S → (a + b + c + d) % 2 = 0

-- Define the main theorem
theorem product_of_ten_numbers_cannot_end_in_1580 (S : Finset ℕ) (h : S.card = 10) (heven : all_four_sum_even S) :
  ∀ (P : ℕ), (P = ∏ i in S, i) → (P % 10000 ≠ 1580) :=
sorry

end product_of_ten_numbers_cannot_end_in_1580_l54_54572


namespace no_valid_arrangement_l54_54316

theorem no_valid_arrangement :
  ∀ (table : Fin 300 → Fin 300 → Int), 
  (∀ i j, table i j = 1 ∨ table i j = -1) →
  abs (∑ i j, table i j) < 30000 →
  (∀ i j, abs (∑ x in Finset.finRange 3, ∑ y in Finset.finRange 5, table (i + x) (j + y)) > 3) →
  (∀ i j, abs (∑ x in Finset.finRange 5, ∑ y in Finset.finRange 3, table (i + x) (j + y)) > 3) →
  False :=
by
  intros table h_entries h_total_sum h_rect_sum_3x5 h_rect_sum_5x3
  sorry

end no_valid_arrangement_l54_54316


namespace circle_equation_and_shortest_chord_l54_54464

-- Definitions based on given conditions
def point_P : ℝ × ℝ := (4, -1)
def line_l1 (x y : ℝ) : Prop := x - 6 * y - 10 = 0
def line_l2 (x y : ℝ) : Prop := 5 * x - 3 * y = 0

-- The circle should be such that it intersects line l1 at point P and its center lies on line l2
theorem circle_equation_and_shortest_chord 
  (C : ℝ × ℝ) (r : ℝ) (hC_l2 : line_l2 C.1 C.2)
  (h_intersect : ∃ (k : ℝ), point_P.1 = (C.1 + k * (C.1 - point_P.1)) ∧ point_P.2 = (C.2 + k * (C.2 - point_P.2))) :
  -- Proving (1): Equation of the circle
  ((C.1 = 3) ∧ (C.2 = 5) ∧ r^2 = 37) ∧
  -- Proving (2): Length of the shortest chord through the origin is 2 * sqrt(3)
  (2 * Real.sqrt 3 = 2 * Real.sqrt (r^2 - ((C.1^2 + C.2^2) - (2 * C.1 * 0 + 2 * C.2 * 0)))) :=
by
  sorry

end circle_equation_and_shortest_chord_l54_54464


namespace exhaust_pipe_leak_time_l54_54102

theorem exhaust_pipe_leak_time : 
  (∃ T : Real, T > 0 ∧ 
                (1 / 10 - 1 / T) = 1 / 59.999999999999964 ∧ 
                T = 12) :=
by
  sorry

end exhaust_pipe_leak_time_l54_54102


namespace angle_C_is_65_l54_54301

theorem angle_C_is_65 (AB CD AD BC : Line) (A B C D O : Point)
  (h_parallel : Parallel AB CD)
  (h_intersect : Intersect AD BC O)
  (angle_A : ∠ (lineThrough A B) (lineThrough A D) = 40)
  (angle_AOB : ∠ (lineThrough A O) (lineThrough O B) = 75) :
  ∠ (lineThrough C D) (lineThrough C D) = 65 := by
  sorry

end angle_C_is_65_l54_54301


namespace graph_transformation_l54_54804

theorem graph_transformation :
  (∀ x, sqrt 3 * cos x ^ 2 + sin x * cos x = sqrt 3 / 2 + sin (2 * x + π / 3)) ↔ 
  ((∀ y, y = sin (2 * x) → ∃ x, y = sin (2 * (x - π / 6) + π / 3) + sqrt 3 / 2) :=
begin
  sorry
end

end graph_transformation_l54_54804


namespace problem_statement_l54_54824

theorem problem_statement (x : ℕ) (h : x = 2016) : (x^2 - x) - (x^2 - 2 * x + 1) = 2015 := by
  sorry

end problem_statement_l54_54824


namespace red_peaches_count_l54_54410

/-- Math problem statement:
There are some red peaches and 16 green peaches in the basket.
There is 1 more red peach than green peaches in the basket.
Prove that the number of red peaches in the basket is 17.
--/

-- Let G be the number of green peaches and R be the number of red peaches.
def G : ℕ := 16
def R : ℕ := G + 1

theorem red_peaches_count : R = 17 := by
  sorry

end red_peaches_count_l54_54410


namespace percentage_of_pushups_l54_54523

-- Problem conditions as definitions
def jumpingJacks := 12
def pushups := 8
def situps := 20
def totalExercises := jumpingJacks + pushups + situps

-- Question and the proof goal
theorem percentage_of_pushups : 
  (pushups / totalExercises : ℝ) * 100 = 20 := by
  sorry

end percentage_of_pushups_l54_54523


namespace students_remaining_l54_54636

noncomputable def students_after_stops 
  (n : ℕ)
  (first_fraction : ℚ)
  (second_fraction : ℚ)
  (third_fraction : ℚ) : ℕ :=
  let after_first := n - (first_fraction * n : ℚ).to_nat
  let after_second := after_first - (second_fraction * after_first : ℚ).to_nat
  let after_third := after_second - (third_fraction * after_second : ℚ).to_nat
  after_third

theorem students_remaining (initial_students : ℕ)
  (first_fraction : ℚ)
  (second_fraction : ℚ)
  (third_fraction : ℚ)
  (h_initial : initial_students = 64)
  (h_first : first_fraction = 1/4)
  (h_second : second_fraction = 1/3)
  (h_third : third_fraction = 1/6) :
  students_after_stops initial_students first_fraction second_fraction third_fraction = 27 := 
by 
  rw [h_initial, h_first, h_second, h_third]
  unfold students_after_stops
  norm_num
  sorry

end students_remaining_l54_54636


namespace Alan_age_is_29_l54_54486

/-- Alan and Chris ages problem -/
theorem Alan_age_is_29
    (A C : ℕ)
    (h1 : A + C = 52)
    (h2 : C = A / 3 + 2 * (A - C)) :
    A = 29 :=
by
  sorry

end Alan_age_is_29_l54_54486


namespace eval_power_l54_54180

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end eval_power_l54_54180


namespace quadratic_equation_unique_solution_l54_54398

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  16 - 4 * a * c = 0 ∧ a + c = 5 ∧ a < c → (a, c) = (1, 4) :=
by
  sorry

end quadratic_equation_unique_solution_l54_54398


namespace max_volume_surface_area_ratio_sine_l54_54984

theorem max_volume_surface_area_ratio_sine :
  ∀ (r : ℝ), (let l := 3 in
         let h := sqrt (l^2 - r^2) in
         let V := (1 / 3) * π * r^2 * h in
         let S := π * r * l in
         let ratio := (V / S) in
         r = 3 * sqrt(2) / 2 →
         sin (atan2 h r) = sqrt(2) / 2) :=
by 
  assume r,
  let l := 3 in
  let h := sqrt (l^2 - r^2) in
  let V := (1 / 3) * π * r^2 * h in
  let S := π * r * l in
  let ratio := V / S in
  assume h_eq : r = 3 * sqrt(2) / 2,
  sorry

end max_volume_surface_area_ratio_sine_l54_54984


namespace correct_choice_l54_54793

-- Define the structures and options
inductive Structure
| Sequential
| Conditional
| Loop
| Module

def option_A : List Structure :=
  [Structure.Sequential, Structure.Module, Structure.Conditional]

def option_B : List Structure :=
  [Structure.Sequential, Structure.Loop, Structure.Module]

def option_C : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

def option_D : List Structure :=
  [Structure.Module, Structure.Conditional, Structure.Loop]

-- Define the correct structures
def basic_structures : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

-- The theorem statement
theorem correct_choice : option_C = basic_structures :=
  by
    sorry  -- Proof would go here

end correct_choice_l54_54793


namespace shorter_side_of_room_l54_54121

theorem shorter_side_of_room
  (P : ℕ) (A : ℕ) (a b : ℕ)
  (perimeter_eq : 2 * a + 2 * b = P)
  (area_eq : a * b = A) (partition_len : ℕ) (partition_cond : partition_len = 5)
  (room_perimeter : P = 60)
  (room_area : A = 200) :
  b = 10 := 
by
  sorry

end shorter_side_of_room_l54_54121


namespace angle_conversion_l54_54161

theorem angle_conversion
  (angle_in_degrees : ℝ)
  (h1 : angle_in_degrees = -1485) :
  ∃ (k : ℤ) (α : ℝ), 0 ≤ α ∧ α < 2 * real.pi ∧ α + 2 * k * real.pi = angle_in_degrees * real.pi / 180 :=
begin
  use [-5, 7 * real.pi / 4],
  split,
  { norm_num },
  split,
  { norm_num,
    apply real.pi_pos },
  {
    conv_rhs { rw [← sub_eq_add_neg, ← sub_eq_add_neg (10 * real.pi)] },
    norm_num,
    field_simp,
    ring,
  },
end

end angle_conversion_l54_54161


namespace lines_parallel_if_perpendicular_to_same_plane_l54_54756

variables {α : Type*} [plane α] (a b : line α)

-- Defining the condition that the lines are perpendicular to the plane α
def perpendicular_to_plane (l : line α) (π : plane α) : Prop :=
  -- Definition for line being perpendicular to a plane (not provided, typically involves angles)
  sorry 

-- Assuming a and b are perpendicular to the same plane α
axiom a_perpendicular_to_plane : perpendicular_to_plane a α
axiom b_perpendicular_to_plane : perpendicular_to_plane b α

-- Proving that lines a and b are parallel
theorem lines_parallel_if_perpendicular_to_same_plane :
  a_perpendicular_to_plane α → b_perpendicular_to_plane α → a ∥ b :=
by
  sorry

end lines_parallel_if_perpendicular_to_same_plane_l54_54756


namespace dice_problem_l54_54759

/-- Rory uses four identical standard dice to build a solid. The numbers on opposite faces of a standard die add up to 7. Whenever two dice touch, the numbers on the touching faces are the same. The numbers on some faces of the solid are shown (4 on the rear right-hand side and 1 on the left-hand side of the front). Prove that the number on the face marked with a question mark is 5. -/
theorem dice_problem : 
  ∀ (d₁ d₂ d₃ d₄ : ℕ), 
    d₁ = 1 ∧ d₂ = 4 ∧ d₃ = 6 ∧ d₄ = 3 → 
    ∀ opposite_sum : ∀ x y, (x + y = 7) → 
    ∃! x, x = 5 := 
sorry

end dice_problem_l54_54759


namespace sum_S_2009_l54_54735

def S_k (k : ℕ) : ℚ := 1 / (2 * k * (k + 1))

def sum_S (n : ℕ) : ℚ :=
  finset.sum (finset.range n) (λ k, S_k (k + 1))

theorem sum_S_2009 : sum_S 2009 = 2009 / 4020 :=
by
  sorry

end sum_S_2009_l54_54735


namespace coeff_x_term_expansion_l54_54032

theorem coeff_x_term_expansion : 
  ∀ (x : ℝ), coeff (expand (3) (polynomial.x^2 - polynomial.C 3 * polynomial.x + polynomial.C 3)) 1 = -81 := by
  sorry

end coeff_x_term_expansion_l54_54032


namespace jordan_total_points_l54_54286

-- Definitions based on conditions in the problem
def jordan_attempts (x y : ℕ) : Prop :=
  x + y = 40

def points_from_three_point_shots (x : ℕ) : ℝ :=
  0.75 * x

def points_from_two_point_shots (y : ℕ) : ℝ :=
  0.8 * y

-- Main theorem to prove the total points scored by Jordan
theorem jordan_total_points (x y : ℕ) 
  (h_attempts : jordan_attempts x y) : 
  points_from_three_point_shots x + points_from_two_point_shots y = 30 := 
by
  sorry

end jordan_total_points_l54_54286


namespace quadratic_roots_range_l54_54274

theorem quadratic_roots_range (k : ℝ) : (x^2 - 6*x + k = 0) → k < 9 := 
by
  sorry

end quadratic_roots_range_l54_54274


namespace sum_of_second_largest_and_smallest_is_1089_l54_54741

theorem sum_of_second_largest_and_smallest_is_1089 :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.
  let three_digit_numbers := {n | ∃ (d1 d2 d3 : ℕ), d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d1 ≠ 0 ∧ n = d1 * 100 + d2 * 10 + d3}.
  let second_smallest := 103
  let second_largest := 986
  (second_smallest + second_largest = 1089) :=
by
  sorry

end sum_of_second_largest_and_smallest_is_1089_l54_54741


namespace percentage_of_pushups_l54_54524

-- Problem conditions as definitions
def jumpingJacks := 12
def pushups := 8
def situps := 20
def totalExercises := jumpingJacks + pushups + situps

-- Question and the proof goal
theorem percentage_of_pushups : 
  (pushups / totalExercises : ℝ) * 100 = 20 := by
  sorry

end percentage_of_pushups_l54_54524


namespace b_50_is_3729_l54_54162

def b : ℕ → ℕ
| 0 := 5
| (n+1) := b n + 3 * n + 1

theorem b_50_is_3729 : b 50 = 3729 := 
sorry

end b_50_is_3729_l54_54162


namespace find_m_collinear_l54_54115

-- Definition of a point in 2D space
structure Point2D where
  x : ℤ
  y : ℤ

-- Predicate to check if three points are collinear 
def collinear_points (p1 p2 p3 : Point2D) : Prop :=
  (p3.x - p2.x) * (p2.y - p1.y) = (p2.x - p1.x) * (p3.y - p2.y)

-- Given points A, B, and C
def A : Point2D := ⟨2, 3⟩
def B (m : ℤ) : Point2D := ⟨-4, m⟩
def C : Point2D := ⟨-12, -1⟩

-- Theorem stating the value of m such that points A, B, and C are collinear
theorem find_m_collinear : ∃ (m : ℤ), collinear_points A (B m) C ∧ m = 9 / 7 := sorry

end find_m_collinear_l54_54115


namespace find_p_plus_q_l54_54592

noncomputable def p (d e : ℝ) (x : ℝ) : ℝ := d * x + e
noncomputable def q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_p_plus_q (d e a b c : ℝ)
  (h1 : p d e 0 / q a b c 0 = 4)
  (h2 : p d e (-1) = -1)
  (h3 : q a b c 1 = 3)
  (e_eq : e = 4 * c):
  (p d e x + q a b c x) = (3*x^2 + 26*x - 30) :=
by
  sorry

end find_p_plus_q_l54_54592


namespace baker_made_cakes_l54_54495

-- Conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- Question and required proof
theorem baker_made_cakes : (cakes_sold + cakes_left = 217) :=
by
  sorry

end baker_made_cakes_l54_54495


namespace insulation_cost_l54_54867

def tank_length : ℕ := 4
def tank_width : ℕ := 5
def tank_height : ℕ := 2
def cost_per_sqft : ℕ := 20

def surface_area (L W H : ℕ) : ℕ := 2 * (L * W + L * H + W * H)
def total_cost (SA cost_per_sqft : ℕ) : ℕ := SA * cost_per_sqft

theorem insulation_cost : 
  total_cost (surface_area tank_length tank_width tank_height) cost_per_sqft = 1520 :=
by
  sorry

end insulation_cost_l54_54867


namespace BretCatchesFrogs_l54_54757

-- Define the number of frogs caught by Alster, Quinn, and Bret.
def AlsterFrogs : Nat := 2
def QuinnFrogs (a : Nat) : Nat := 2 * a
def BretFrogs (q : Nat) : Nat := 3 * q

-- The main theorem to prove
theorem BretCatchesFrogs : BretFrogs (QuinnFrogs AlsterFrogs) = 12 :=
by
  sorry

end BretCatchesFrogs_l54_54757


namespace g_spec_l54_54733

noncomputable def g : ℝ → ℝ := λ x, 3^x - 2^x

theorem g_spec (g : ℝ → ℝ) (h1 : g 1 = 1)
  (h2 : ∀ (x y : ℝ), g (x + y) = 2^y * g x + 3^x * g y) :
  ∀ x, g x = 3^x - 2^x := 
by
  sorry

end g_spec_l54_54733


namespace area_of_rhombus_and_triangle_l54_54435

noncomputable def total_enclosed_area : ℝ :=
  12.24

theorem area_of_rhombus_and_triangle :
  ∀ (x y : ℝ), 
  |4 * x| + |3 * y| = 12 ∧ y ≤ 2 * x - 4 → 
  x ≥ -3 ∧ x ≤ 3 →
  y ≥ -4 ∧ y ≤ 4 →
  ∑ (total_area : ℝ) in set (λ x y, 
    if (x < 3) ∨ (y < 4) then (12.24) else 0),
  total_area = 12.24 :=
sorry

end area_of_rhombus_and_triangle_l54_54435


namespace pi_approx_by_jews_l54_54303

theorem pi_approx_by_jews (S D C : ℝ) (h1 : 4 * S = (5 / 4) * C) (h2 : D = S) (h3 : C = π * D) : π = 3 := by
  sorry

end pi_approx_by_jews_l54_54303


namespace charles_drawn_after_work_l54_54908

-- Conditions
def total_papers : ℕ := 20
def drawn_today : ℕ := 6
def drawn_yesterday_before_work : ℕ := 6
def papers_left : ℕ := 2

-- Question and proof goal
theorem charles_drawn_after_work :
  ∀ (total_papers drawn_today drawn_yesterday_before_work papers_left : ℕ),
  total_papers = 20 →
  drawn_today = 6 →
  drawn_yesterday_before_work = 6 →
  papers_left = 2 →
  (total_papers - drawn_today - drawn_yesterday_before_work - papers_left = 6) :=
by
  intros total_papers drawn_today drawn_yesterday_before_work papers_left
  assume h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact sorry

end charles_drawn_after_work_l54_54908


namespace probability_more_heads_than_tails_l54_54487

theorem probability_more_heads_than_tails :
  let p_heads := 2/3 in
  let p_tails := 1/3 in
  let flips := 5 in
  (∑ k in {3, 4, 5}, (Nat.choose flips k) * (p_heads ^ k) * (p_tails ^ (flips - k))) = 64 / 81 :=
by sorry

end probability_more_heads_than_tails_l54_54487


namespace sum_of_squares_l54_54782

theorem sum_of_squares :
  ∃ (a b c d e f : ℤ),
    (∀ t, 0 ≤ t ∧ t ≤ 1 → (t = 0 → b = 1 ∧ d = 3 ∧ f = 6) ∧
        (t = 1 → a + b = 6 ∧ c + d = -2 ∧ e + f = 12)) ∧
    (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 132) :=
begin
  use [5, 1, -5, 3, 6, 6],
  split,
  {
    intros t h_le h_ub,
    split; intros h_t,
    { cases h_t with b0 d0 f0,
      exact ⟨b0, d0, f0⟩ },
    { cases h_t with ab cd ef,
      split,
      linarith,
      split,
      linarith,
      linarith }
  },
  { simp }
end

end sum_of_squares_l54_54782


namespace toys_bought_l54_54034

-- Define the cost of a single toy
def toy_cost : ℝ := 12.00

-- Define the cost for every second toy (half off)
def half_off_cost : ℝ := toy_cost / 2

-- Define the total amount spent by Samantha
def total_spent : ℝ := 36.00

-- Define the cost of a pair of toys under the "buy one, get one half off" deal
def pair_cost : ℝ := toy_cost + half_off_cost

-- State the theorem to prove that Samantha buys 4 toys
theorem toys_bought : ∃ n : ℤ, n = 4 ∧ total_spent = (n / 2 * pair_cost) ∧ n % 2 = 0 :=
by
  sorry

end toys_bought_l54_54034


namespace geom_seq_conditions_l54_54283

noncomputable def z_sequence (a b : ℝ) : ℕ → ℂ
| 0     := 1
| 1     := a + b * Complex.I
| 2     := b + a * Complex.I
| (n+3) := (a + b * Complex.I) * z_sequence n

theorem geom_seq_conditions (a b : ℝ) (h : a > 0) :
  (z_sequence a b 12).sum = 0 ∧ (z_sequence a b 12).prod = -1 := by
sorry

end geom_seq_conditions_l54_54283


namespace part_I_part_II_l54_54595

section cs_inequalities

variables {a b c m n p : ℝ}

-- Conditions
def condition1 := a^2 + b^2 + c^2 = 1
def condition2 := m^2 + n^2 + p^2 = 1
def condition3 := a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

-- Part (I)
theorem part_I (h1 : condition1) (h2 : condition2) : |a * m + b * n + c * p| ≤ 1 :=
sorry

-- Part (II)
theorem part_II (h1 : condition1) (h2 : condition2) (h3 : condition3) : (m^4 / a^2) + (n^4 / b^2) + (p^4 / c^2) ≥ 1 :=
sorry

end cs_inequalities

end part_I_part_II_l54_54595


namespace no_rational_solution_l54_54361

theorem no_rational_solution :
  ¬ ∃ (x y z : ℚ), 
  x + y + z = 0 ∧ x^2 + y^2 + z^2 = 100 := sorry

end no_rational_solution_l54_54361


namespace students_per_row_l54_54460

theorem students_per_row (x : ℕ) : 45 = 11 * x + 1 → x = 4 :=
by
  intro h
  sorry

end students_per_row_l54_54460


namespace find_k_l54_54601

theorem find_k (k : ℝ) :
  let l1 (x : ℝ) := x,
      l2 (x : ℝ) := k*x - k + 1,
      A := (1,1 : ℝ × ℝ),
      Bx := 4 ∨ Bx := -4,
      B := (Bx, 0 : ℝ × ℝ),
      O := (0, 0 : ℝ × ℝ),
      area_OAB := 2 in
  (k = -1/3 ∨ k = 1/5) ∧
  A = (1, 1) ∧
  (l1 A.fst = A.snd) ∧
  (l2 A.fst = A.snd) ∧
  (B = (4, 0) ∨ B = (-4, 0)) ∧
  (1/2 * (4-0) * 1 = 2) :=
sorry

end find_k_l54_54601


namespace hyperbola_eccentricity_range_l54_54994

/-- Prove that the range of the eccentricity of the given hyperbola is [sqrt(5)/2, sqrt(2)) --/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : 2 * b ≥ a) : 
    (∃ e : ℝ, e = sqrt(1 + (b / a) ^ 2) ∧ e ∈ set.Ico (sqrt 5 / 2) (sqrt 2)) :=
by sorry

end hyperbola_eccentricity_range_l54_54994


namespace vasya_grades_l54_54662

variables
  (grades : List ℕ)
  (length_grades : grades.length = 5)
  (median_grade : grades.nthLe 2 sorry = 4)  -- Assuming 0-based indexing
  (mean_grade : (grades.sum : ℚ) / 5 = 3.8)
  (most_frequent_A : ∀ n ∈ grades, n ≤ 5)

theorem vasya_grades (h : ∀ x ∈ grades, x ≤ 5 ∧ ∃ k, grades.nthLe 3 sorry = 5 ∧ grades.count 5 > grades.count x):
  ∃ g1 g2 g3 g4 g5 : ℕ, grades = [g1, g2, g3, g4, g5] ∧ g1 ≤ g2 ∧ g2 ≤ g3 ∧ g3 ≤ g4 ∧ g4 ≤ g5 ∧ [g1, g2, g3, g4, g5] = [2, 3, 4, 5, 5] :=
sorry

end vasya_grades_l54_54662


namespace remainder_when_divided_l54_54385

theorem remainder_when_divided (L S R : ℕ) (h1: L - S = 1365) (h2: S = 270) (h3: L = 6 * S + R) : 
  R = 15 := 
by 
  sorry

end remainder_when_divided_l54_54385


namespace solve_for_x_l54_54364

theorem solve_for_x (x : ℝ) (h : 8^(4 * x - 6) = (1 / 2)^(x + 5)) : x = 1 :=
by {
  sorry
}

end solve_for_x_l54_54364


namespace charles_pictures_after_work_l54_54914

variable (initial_papers : ℕ)
variable (draw_today : ℕ)
variable (draw_yesterday_morning : ℕ)
variable (papers_left : ℕ)

theorem charles_pictures_after_work :
    initial_papers = 20 →
    draw_today = 6 →
    draw_yesterday_morning = 6 →
    papers_left = 2 →
    initial_papers - (draw_today + draw_yesterday_morning + 6) = papers_left →
    6 = (initial_papers - draw_today - draw_yesterday_morning - papers_left) := 
by
  intros h1 h2 h3 h4 h5
  exact sorry

end charles_pictures_after_work_l54_54914


namespace transmission_time_estimate_l54_54932

theorem transmission_time_estimate :
  let blocks := 60
  let chunks_per_block := 512
  let transmission_rate := 120
  let seconds_per_minute := 60
  let total_chunks := blocks * chunks_per_block
  let time_in_seconds := total_chunks / transmission_rate
  let time_in_minutes := time_in_seconds / seconds_per_minute
  time_in_minutes ≈ 4 := 
by
  sorry

end transmission_time_estimate_l54_54932


namespace distinct_x_intercepts_nonzero_d_l54_54257

noncomputable def Q (a b c d e f : ℝ) (x : ℝ) : ℝ := x^6 + a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f

theorem distinct_x_intercepts_nonzero_d
  (a b c d e f : ℝ) :
  (∃ p q r s t : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0 ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t ∧ x * (x - p) * (x - q) * (x - r) * (x - s) * (x - t) = Q a b c d e 0) →
  d ≠ 0 := by
  sorry

end distinct_x_intercepts_nonzero_d_l54_54257


namespace max_difference_y_coordinates_l54_54940

theorem max_difference_y_coordinates :
  let f x := 5 - x^2 + x^3
  let g x := 1 + x^2 + x^3
  ∃ x1 x2 : ℝ, f x1 = g x1 ∧ f x2 = g x2 ∧ x1 ≠ x2
  ∧ (x1 = real.sqrt 2 ∨ x1 = -real.sqrt 2)
  ∧ (x2 = real.sqrt 2 ∨ x2 = -real.sqrt 2)
  ∧ (abs (f (real.sqrt 2) - f (-real.sqrt 2)) = 4 * real.sqrt 2) :=
by
  let f := fun x => 5 - x^2 + x^3
  let g := fun x => 1 + x^2 + x^3
  have h_f_eq_g : ∀ x : ℝ, f x = g x ↔  4 = 2 * x^2 := by
    sorry -- Proof that intersection points satisfy the equation 4 = 2x^2
  have h_x_val : ∀ x : ℝ, 4 = 2 * x^2 ↔ x = real.sqrt 2 ∨ x = -real.sqrt 2 := by
    sorry -- Proof that the solutions to 4 = 2x^2 are x = ±√2
  have h_y_diff : abs (f (real.sqrt 2) - f (-real.sqrt 2)) = 4 * real.sqrt 2 := by
    sorry -- Proof that the maximum difference between the y-coordinates is 4√2
  exact ⟨real.sqrt 2, -real.sqrt 2, (h_f_eq_g (real.sqrt 2)).mpr rfl, (h_f_eq_g (-real.sqrt 2)).mpr rfl, sorry⟩

end max_difference_y_coordinates_l54_54940


namespace time_to_cross_bridge_l54_54478

/-- Definition of the length of the train. -/
def length_of_train : ℕ := 475

/-- Definition of the speed of the train in km/hr. -/
def speed_of_train_km_per_hr : ℕ := 90

/-- Definition of the length of the bridge. -/
def length_of_bridge : ℕ := 275

/-- Conversion factor from km/hr to m/s. -/
def km_per_hr_to_m_per_s (speed_km_hr : ℕ) : ℕ :=
  speed_km_hr * 1000 / 3600

/-- The time it takes for the train to cross the bridge in seconds. -/
theorem time_to_cross_bridge :
  let total_distance := length_of_train + length_of_bridge,
      speed_m_per_s := km_per_hr_to_m_per_s speed_of_train_km_per_hr
  in total_distance / speed_m_per_s = 30 := by
  let total_distance := length_of_train + length_of_bridge
  let speed_m_per_s := km_per_hr_to_m_per_s speed_of_train_km_per_hr
  sorry

end time_to_cross_bridge_l54_54478


namespace total_rats_l54_54702

variable (Kenia Hunter Elodie : ℕ) -- Number of rats each person has

-- Conditions
-- Elodie has 30 rats
axiom h1 : Elodie = 30
-- Elodie has 10 rats more than Hunter
axiom h2 : Elodie = Hunter + 10
-- Kenia has three times as many rats as Hunter and Elodie have together
axiom h3 : Kenia = 3 * (Hunter + Elodie)

-- Prove that the total number of pets the three have together is 200
theorem total_rats : Kenia + Hunter + Elodie = 200 := 
by 
  sorry

end total_rats_l54_54702


namespace math_problem_l54_54397

noncomputable theory
open_locale classical

/-
  Problem statement: 
  Given the following conditions:
  1. The perimeter of triangle AQM is 180.
  2. The angle QAM is a right angle.
  3. A circle of radius 15 with center O on AQ is tangent to AM and QM.

  Prove that OQ = 75/4, where m = 75, n = 4, and m + n = 79.
-/
def relatively_prime (a b : ℕ) : Prop := nat.gcd a b = 1

variables (A Q M O : Type) [field A] [field Q] [field M] [field O]
variables [metric_space A] [metric_space Q] [metric_space M] [metric_space O]
variables (AQ AM QM : A → A → ℝ) (angle_QAM : ℝ)
variables (circle_center_O : A) (radius_O : ℝ)
variables (OQ : ℝ)

theorem math_problem 
  (perimeter_AQM : AQ + AM + QM = 180)
  (right_angle_QAM : angle_QAM = π / 2)
  (circle_tangent_AM_QM : radius_O = 15 ∧ circle_center_O ∈ AQ)
  (OQ_is_fraction : ∃ m n : ℕ, OQ = m / n ∧ relatively_prime m n) :
  OQ = 75 / 4 :=
sorry

end math_problem_l54_54397


namespace cycloid_area_l54_54503

-- Define the parametric equations
def x (t : ℝ) : ℝ := 2 * (t - sin t)
def y (t : ℝ) : ℝ := 2 * (1 - cos t)

-- Calculate the range of t
def t_range : Set ℝ := {t | 0 ≤ t ∧ t ≤ 2 * Real.pi}

-- Define the integrand
def integrand (t : ℝ) : ℝ := y t * (deriv x t)

-- Prove that the area under one arc of the cycloid is 12π
theorem cycloid_area : (| ∫ t in 0..(2 * Real.pi), integrand t |) = 12 * Real.pi :=
by
  sorry

end cycloid_area_l54_54503


namespace vasya_grades_l54_54681

-- Given conditions
constants (a1 a2 a3 a4 a5 : ℕ)
axiom grade_median : a3 = 4
axiom grade_sum : a1 + a2 + a3 + a4 + a5 = 19
axiom most_A_grades : ∀ (n : ℕ), n ≠ 5 → (∃ m, m > 0 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 5 → (if a1 = n ∨ a2 = n ∨ a3 = n ∨ a4 = n ∨ a5 = n then m > 1 else m = 0))

-- Prove that the grades are (2, 3, 4, 5, 5)
theorem vasya_grades : (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨ 
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) := 
by sorry

end vasya_grades_l54_54681


namespace collinear_points_on_curve_sum_zero_l54_54457

theorem collinear_points_on_curve_sum_zero
  {x1 y1 x2 y2 x3 y3 : ℝ}
  (h_curve1 : y1^2 = x1^3)
  (h_curve2 : y2^2 = x2^3)
  (h_curve3 : y3^2 = x3^3)
  (h_collinear : ∃ (a b c k : ℝ), k ≠ 0 ∧ 
    a * x1 + b * y1 + c = 0 ∧
    a * x2 + b * y2 + c = 0 ∧
    a * x3 + b * y3 + c = 0) :
  x1 / y1 + x2 / y2 + x3 / y3 = 0 :=
sorry

end collinear_points_on_curve_sum_zero_l54_54457


namespace drink_price_is_correct_l54_54740

variable (sandwich_price : ℝ)
variable (coupon_fraction : ℝ)
variable (avocado_upgrade_cost : ℝ)
variable (salad_cost : ℝ)
variable (total_lunch_cost : ℝ)

def discount := sandwich_price * coupon_fraction
def discounted_sandwich_price := sandwich_price - discount
def upgraded_sandwich_price := discounted_sandwich_price + avocado_upgrade_cost
def combined_meal_cost := upgraded_sandwich_price + salad_cost
def drink_cost := total_lunch_cost - combined_meal_cost

theorem drink_price_is_correct
  (h1 : sandwich_price = 8)
  (h2 : coupon_fraction = 1/4)
  (h3 : avocado_upgrade_cost = 1)
  (h4 : salad_cost = 3)
  (h5 : total_lunch_cost = 12) : drink_cost = 2 := by
  sorry

end drink_price_is_correct_l54_54740


namespace value_of_expression_l54_54404

theorem value_of_expression : 8 * (6 - 4) + 2 = 18 := by
  sorry

end value_of_expression_l54_54404


namespace problem_statement_l54_54013

variable {a b c d : ℝ}

theorem problem_statement (h : a * d - b * c = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + c * d ≠ 1 := 
sorry

end problem_statement_l54_54013


namespace original_deck_total_l54_54106

theorem original_deck_total (b y : ℕ) 
    (h1 : (b : ℚ) / (b + y) = 2 / 5)
    (h2 : (b : ℚ) / (b + y + 6) = 5 / 14) :
    b + y = 50 := by
  sorry

end original_deck_total_l54_54106


namespace acute_tris_in_cuboid_l54_54729

theorem acute_tris_in_cuboid (cuboid : Type) [rectangular_cuboid cuboid]
  (vertices : Finset cuboid) (h_vertices : vertices.card = 8)
  (h_tris : ∀ v : cuboid, (vertices.filter (λ t, is_right_angle_triangle t v)).card = 6) :
  (vertices.subsets 3).filter (λ t, is_acute_triangle t).card = 8 := by
  sorry

end acute_tris_in_cuboid_l54_54729


namespace judys_expenditure_l54_54698

theorem judys_expenditure 
  (bananas_price : ℝ := 2) (bananas_count : ℕ := 4)
  (rice_price : ℝ := 6) (rice_count : ℕ := 2)
  (pineapples_price : ℝ := 5) (pineapples_count : ℕ := 3)
  (cake_price : ℝ := 10) (cake_count : ℕ := 1)
  (discount : ℝ := 0.25) (coupon_threshold : ℝ := 30) (coupon_value : ℝ := 10) :
  let discounted_bananas_price := bananas_price * (1 - discount)
  in ((discounted_bananas_price * bananas_count) + (rice_price * rice_count) + 
      (pineapples_price * pineapples_count) + (cake_price * cake_count) - 
      if ((discounted_bananas_price * bananas_count) + (rice_price * rice_count) + 
          (pineapples_price * pineapples_count) + (cake_price * cake_count)) > coupon_threshold 
      then coupon_value else 0) = 33 := 
by
  let discounted_bananas_price := bananas_price * (1 - discount)
  let total_bananas := discounted_bananas_price * bananas_count
  let total_rice := rice_price * rice_count
  let total_pineapples := pineapples_price * pineapples_count
  let total_cake := cake_price * cake_count
  let subtotal := total_bananas + total_rice + total_pineapples + total_cake
  have h1 : total_bananas = 4 * 1.5, by norm_num
  have h2 : total_rice = 2 * 6, by norm_num
  have h3 : total_pineapples = 3 * 5, by norm_num
  have h4 : total_cake = 1 * 10, by norm_num
  have h5 : subtotal = 43, by linarith [h1, h2, h3, h4]
  have h6 : subtotal > coupon_threshold, by norm_num [h5]
  let total := subtotal - coupon_value
  have h7 : total = 33, by linarith [h5]
  exact h7

end judys_expenditure_l54_54698


namespace volume_of_S_ABC_l54_54382

noncomputable def volume_of_tetrahedron (S A B C : ℝ) : ℝ := sorry

theorem volume_of_S_ABC {S A B C : ℝ} 
  (base_equilateral : ∀ (x y z : ℝ), (x = y ∧ y = z ∧ z = x) → (true))
  (A_orthocenter : ∀ (H : ℝ), H = orthocenter_of_triangle B S C)
  (dihedral_angle : ∀ (angle : ℝ), angle = 30)
  (SA_length : real : S - A = 2 * real.sqrt 3) :
  volume_of_tetrahedron S A B C = 9 / 4 * real.sqrt 3 :=
sorry

end volume_of_S_ABC_l54_54382


namespace triangle_def_trig_identity_l54_54350

theorem triangle_def_trig_identity :
  ∀ (D E F : ℝ), 
    (∀ DE DF EF : ℝ, 
      DE = 7 ∧ DF = 8 ∧ EF = 6 → 
      (\frac{\cos (\frac{D - E}{2})}{\sin (\frac{F}{2})} + \frac{\sin (\frac{D - E}{2})}{\cos (\frac{F}{2})}) = \frac{7 * \sqrt{15}}{12}) :=
begin
  intros D E F DE DF EF h,
  sorry
end

end triangle_def_trig_identity_l54_54350


namespace resulting_graph_expression_l54_54780

def translated_and_scaled_sine (x : ℝ) : ℝ :=
  2 * sin (x - (π / 4))

theorem resulting_graph_expression :
  ∀ x, (2 * sin (x - (π / 4))) = translated_and_scaled_sine x :=
by
  intro x
  unfold translated_and_scaled_sine
  rfl

end resulting_graph_expression_l54_54780


namespace complex_number_in_third_quadrant_l54_54042

theorem complex_number_in_third_quadrant :
  (1 - Complex.i)^2 / (1 + Complex.i) = -1 - Complex.i ∧ (-1 < 0 ∧ -1 < 0) :=
by
  sorry

end complex_number_in_third_quadrant_l54_54042


namespace factorize_expr1_factorize_expr2_l54_54191

variable (x y a b : ℝ)

theorem factorize_expr1 : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := sorry

theorem factorize_expr2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := sorry

end factorize_expr1_factorize_expr2_l54_54191


namespace range_of_a_l54_54557

noncomputable def range_a : set ℝ :=
  {a | ∃ x : ℝ, x > 0 ∧ a - 2 * x - abs (Real.log x) ≤ 0}

theorem range_of_a :
  range_a = {a : ℝ | a ≤ 1 + Real.log 2} :=
by
  sorry

end range_of_a_l54_54557


namespace max_value_of_f_l54_54261

def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, 1)
def b (x : ℝ) : ℝ × ℝ := (1 / 2, Real.sqrt 3 * Real.cos x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem max_value_of_f : ∃ x : ℝ, f x = 2 :=
by
  sorry

end max_value_of_f_l54_54261


namespace sequence_10_eq_1_div_2_l54_54687

noncomputable def sequence (n : ℕ) : ℚ :=
match n with
| 0     => 1/2
| (n+1) => 1 - 1/(sequence n)

theorem sequence_10_eq_1_div_2 :
  sequence 10 = 1/2 :=
sorry

end sequence_10_eq_1_div_2_l54_54687


namespace girl_with_more_envelopes_l54_54845

-- Defining the necessary conditions.
def initial_envelopes (girls : List Girl) (initial_count : Nat) : Prop :=
  ∀ g : Girl, g.envelopes = initial_count

def unique_envelopes (girls : List Girl) : Prop :=
  ∀ g1 g2 : Girl, g1 ≠ g2 → g1.envelopes ≠ g2.envelopes

noncomputable def send_envelopes (g : Girl) (n : Nat) : Prop :=
  g.sent_envelopes = n

noncomputable def receive_envelopes (g : Girl) (n : Nat) : Prop :=
  g.received_envelopes = n

-- The main theorem we want to prove.
theorem girl_with_more_envelopes
  (girls : List Girl)
  (h1 : length girls = 10)
  (h2 : initial_envelopes girls 10)
  (h3 : ∀ g, ∃ n, send_envelopes g n)
  (h4 : ∀ g, ∃ n, receive_envelopes g n)
  (h5 : unique_envelopes girls) :
  ∃ g : Girl, g.envelopes > g.sent_envelopes :=
sorry

end girl_with_more_envelopes_l54_54845


namespace correct_statements_l54_54829

theorem correct_statements (data : List ℝ) (c : ℝ) :
  (∀ x, x ∈ data → x - c ∈ data) →
  (is_variance_unchanged : bool) →
  (is_min_regression : bool) →
  (is_zero_sum_residuals : bool) →
  (is_variance_unchanged = true ∧ 
   ¬(abs (correlation_coefficient data) = 1) ∧ 
   is_min_regression = true ∧ 
   is_zero_sum_residuals = true) := by
  sorry

end correct_statements_l54_54829


namespace line_intersects_x_axis_at_point_l54_54494

-- Define the conditions and required proof
theorem line_intersects_x_axis_at_point :
  (∃ x : ℝ, ∃ y : ℝ, 5 * y - 7 * x = 35 ∧ y = 0 ∧ (x, y) = (-5, 0)) :=
by
  -- The proof is omitted according to the steps
  sorry

end line_intersects_x_axis_at_point_l54_54494


namespace repeating_decimal_sum_as_fraction_l54_54534

theorem repeating_decimal_sum_as_fraction :
  let d1 := 1 / 9    -- Representation of 0.\overline{1}
  let d2 := 1 / 99   -- Representation of 0.\overline{01}
  d1 + d2 = (4 : ℚ) / 33 := by
{
  sorry
}

end repeating_decimal_sum_as_fraction_l54_54534


namespace determine_delta_l54_54895

theorem determine_delta (r1 r2 r3 r4 r5 r6 : ℕ) (c1 c2 c3 c4 c5 c6 : ℕ) (O Δ : ℕ) 
  (h_sums_rows : r1 + r2 + r3 + r4 + r5 + r6 = 190)
  (h_row1 : r1 = 29) (h_row2 : r2 = 33) (h_row3 : r3 = 33) 
  (h_row4 : r4 = 32) (h_row5 : r5 = 32) (h_row6 : r6 = 31)
  (h_sums_cols : c1 + c2 + c3 + c4 + c5 + c6 = 190)
  (h_col1 : c1 = 29) (h_col2 : c2 = 33) (h_col3 : c3 = 33) 
  (h_col4 : c4 = 32) (h_col5 : c5 = 32) (h_col6 : c6 = 31)
  (h_O : O = 6) : 
  Δ = 4 :=
by 
  sorry

end determine_delta_l54_54895


namespace find_dividend_l54_54541

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 14) (h2 : quotient = 4) (h3 : dividend = quotient * k) : dividend = 56 :=
by
  sorry

end find_dividend_l54_54541


namespace impossible_arrangement_of_1_and_neg1_in_300_by_300_table_l54_54313

theorem impossible_arrangement_of_1_and_neg1_in_300_by_300_table :
  ¬∃ (table : ℕ → ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ 300 ∧ 1 ≤ j ∧ j ≤ 300 → table i j = 1 ∨ table i j = -1) ∧
    abs (∑ i in finset.range 300, ∑ j in finset.range 300, table i j) < 30000 ∧
    ∀ i j, (1 ≤ i ∧ i ≤ 298) ∧ (1 ≤ j ∧ j ≤ 295) →
      abs (∑ di in finset.range 3, ∑ dj in finset.range 5, table (i + di) (j + dj)) > 3 ∧
    ∀ i j, (1 ≤ i ∧ i ≤ 296) ∧ (1 ≤ j ∧ j ≤ 298) →
      abs (∑ di in finset.range 5, ∑ dj in finset.range 3, table (i + di) (j + dj)) > 3 := sorry

end impossible_arrangement_of_1_and_neg1_in_300_by_300_table_l54_54313


namespace original_selling_price_is_440_l54_54496

variable (P : ℝ)

-- Condition: Bill made a profit of 10% by selling a product.
def original_selling_price := 1.10 * P

-- Condition: He had purchased the product for 10% less.
def new_purchase_price := 0.90 * P

-- Condition: With a 30% profit on the new purchase price, the new selling price.
def new_selling_price := 1.17 * P

-- Condition: The new selling price is $28 more than the original selling price.
def price_difference_condition : Prop := new_selling_price P = original_selling_price P + 28

-- Conclusion: The original selling price was \$440
theorem original_selling_price_is_440
    (h : price_difference_condition P) : original_selling_price P = 440 :=
sorry

end original_selling_price_is_440_l54_54496


namespace probability_sum_equals_age_l54_54744

-- Define probability events for coin flip and die roll
def coin_flip (coin: Fin 2 -> ℤ) : ℚ := 1 / 2
def die_roll (die: Fin 6 -> ℤ) : ℚ := 1 / 6

-- Noah's age
def noah_age : ℤ := 16

-- Define the event that the sum of coin flip and die roll equals Noah's age
def event (coin: Fin 2 -> ℤ) (die: Fin 6 -> ℤ) : Prop :=
  ∃ c d, coin c + die d = noah_age

theorem probability_sum_equals_age
  (coin : Fin 2 -> ℤ)
  (die : Fin 6 -> ℤ) :
  event coin die → (coin 0 = 15) → (die 0 = 1) →
  P coin_flip * P die_roll = 1 / 12 :=
sorry

end probability_sum_equals_age_l54_54744


namespace volume_of_figure_eq_half_l54_54151

-- Define a cube data structure and its properties
structure Cube where
  edge_length : ℝ
  h_el : edge_length = 1

-- Define a function to calculate volume of the figure
noncomputable def volume_of_figure (c : Cube) : ℝ := sorry

-- Example cube
def example_cube : Cube := { edge_length := 1, h_el := rfl }

-- Theorem statement
theorem volume_of_figure_eq_half (c : Cube) : volume_of_figure c = 1 / 2 := by
  sorry

end volume_of_figure_eq_half_l54_54151


namespace solve_equation_l54_54018

theorem solve_equation : 
  (∀ x : ℝ, 
      (7 / (x^2 + x) - 3 / (x - x^2) = 1 + (7 - x^2) / (x^2 - 1)) → x = 1) := 
λ x h, sorry

end solve_equation_l54_54018


namespace most_negative_value_l54_54516

-- Define the polynomial equation
def polynomial (x : ℝ) : ℝ := 6 * x^2 - 37 * x + 6

-- Statement: Prove that the most negative value of x for which polynomial(x) = 0 is 1/6
theorem most_negative_value :
  ∃ x : ℝ, polynomial x = 0 ∧ ∀ y : ℝ, polynomial y = 0 → x ≤ y → x = 1 / 6 :=
by
  -- Provide existence proof
  use (1 / 6),
  -- Placeholder for actual proof
  sorry

end most_negative_value_l54_54516


namespace trigonometric_identity_l54_54951

theorem trigonometric_identity (α : ℝ) :
  sin (π / 6 - α) - cos α = 1 / 3 →
  cos (2 * α + π / 3) = 7 / 9 :=
by sorry

end trigonometric_identity_l54_54951


namespace similar_quadratic_radical_l54_54888

theorem similar_quadratic_radical (a b : ℕ) (x : ℝ) (hx₁ : x = sqrt 18) (hx₂ : a = 2) :
  SimplifyRadical x = 3 * sqrt a ↔ b = 2 := sorry

-- Supporting definition for simplifying the radical
noncomputable def SimplifyRadical (x : ℝ) : ℝ :=
  if x = sqrt 18 then 3 * sqrt 2 else x

end similar_quadratic_radical_l54_54888


namespace probability_of_diamond_king_ace_l54_54411

noncomputable def probability_three_cards : ℚ :=
  (11 / 52) * (4 / 51) * (4 / 50) + 
  (1 / 52) * (3 / 51) * (4 / 50) + 
  (1 / 52) * (4 / 51) * (3 / 50)

theorem probability_of_diamond_king_ace :
  probability_three_cards = 284 / 132600 := 
by
  sorry

end probability_of_diamond_king_ace_l54_54411


namespace actual_time_at_5PM_on_car_clock_l54_54031

theorem actual_time_at_5PM_on_car_clock :
  (∀ (t_real t_car : ℝ), (t_real = 0 ∧ t_car = 0) → (t_real = 15 ∧ t_car = 30) →
  ∀ (t_real_target t_car_target : ℝ), (t_car_target = 300) → t_real_target / t_car_target = 1 / 2 →
  t_real_target + 12 * 60 = 2 * 60 + 30 :=
sorry

end actual_time_at_5PM_on_car_clock_l54_54031


namespace numerator_of_fraction_l54_54959

theorem numerator_of_fraction (p : ℚ) (h : 1 / 7 + (2 * 5 - p) / (2 * 5 + p) = 0.5714285714285714) : p = 4 := 
by 
  -- We know that 0.5714285714285714 is equivalent to 4/7
  have h_eq : 1 / 7 + (10 - p) / (10 + p) = 4 / 7 := by 
    rw h 
    norm_num 
  -- Combining fractions and solving for p
  sorry

end numerator_of_fraction_l54_54959


namespace rosy_has_14_fish_l54_54001

-- Define the number of Lilly's fish
def lilly_fish : ℕ := 10

-- Define the total number of fish
def total_fish : ℕ := 24

-- Define the number of Rosy's fish, which we need to prove equals 14
def rosy_fish : ℕ := total_fish - lilly_fish

-- Prove that Rosy has 14 fish
theorem rosy_has_14_fish : rosy_fish = 14 := by
  sorry

end rosy_has_14_fish_l54_54001


namespace min_distance_sum_coordinates_l54_54585

theorem min_distance_sum_coordinates (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, -1)) :
  ∃ P : ℝ × ℝ, P = (0, 3) ∧ ∀ Q : ℝ × ℝ, Q.1 = 0 → |A.1 - Q.1| + |A.2 - Q.2| + |B.1 - Q.1| + |B.2 - Q.2| ≥ |A.1 - (0 : ℝ)| + |A.2 - (3 : ℝ)| + |B.1 - (0 : ℝ)| + |B.2 - (3 : ℝ)| := 
sorry

end min_distance_sum_coordinates_l54_54585


namespace triangle_concyclic_centroid_midpoints_l54_54573

variable (θ : Real)

-- Stating the main theorem in Lean
theorem triangle_concyclic_centroid_midpoints (hθ : 0 < θ ∧ θ < π) : 
  (θ ≤ π / 3 → ∃! (ABC : Triangle ℝ), ∠ABC.A = θ ∧ ABC.bc.length = 1 ∧ 
    areConcyclic [ABC.A, ABC.centroid, ABC.midpoint_AB, ABC.midpoint_AC]) ∧ 
  (θ > π / 3 → ¬ ∃ (ABC : Triangle ℝ), ∠ABC.A = θ ∧ ABC.bc.length = 1 ∧ 
    areConcyclic [ABC.A, ABC.centroid, ABC.midpoint_AB, ABC.midpoint_AC]) := 
sorry

end triangle_concyclic_centroid_midpoints_l54_54573


namespace part1_part2_l54_54656

/-
Given:
total investment: 120 ten thousand Yuan
minimum investment per city: 40 ten thousand Yuan
P(x): profit function for city A = 3 * sqrt(2 * x) - 6
Q(y): profit function for city B = 1/4 * y + 2
x: investment in city A (in ten thousand Yuan)
y: investment in city B = 120 - x

Functions:
total_profit(x) = P(x) + Q(120 - x) = 3 * sqrt(2 * x) - 6 + 1/4 * (120 - x) + 2
  = 3 * sqrt(2 * x) - 1/4 * x + 26

Derivative of total_profit(x):
total_profit'(x) = (3 * sqrt 2) / (2 * sqrt x) - 1/4

We need to prove:
1. total_profit(50) = 43.5
2. argmax_x (total_profit(x)) = 72 when 40 ≤ x ≤ 80
-/
noncomputable def P (x : ℝ) : ℝ := 3 * Real.sqrt (2 * x) - 6
noncomputable def Q (x : ℝ) : ℝ := 1 / 4 * x + 2
def total_profit (x : ℝ) : ℝ := P x + Q (120 - x)

theorem part1 (h1 : 40 ≤ 50 ∧ 50 ≤ 80) : total_profit 50 = 43.5 := sorry

theorem part2 (h2 : 40 ≤ 72 ∧ 72 ≤ 80) :
  ∀ x, 40 ≤ x ∧ x ≤ 80 → total_profit x ≤ total_profit 72 :=
sorry

end part1_part2_l54_54656


namespace line_through_perpendicular_l54_54543

theorem line_through_perpendicular (x y : ℝ) :
  (∃ (k : ℝ), (2 * x - y + 3 = 0) ∧ k = - 1 / 2) →
  (∃ (a b c : ℝ), (a * (-1) + b * 1 + c = 0) ∧ a = 1 ∧ b = 2 ∧ c = -1) :=
by
  sorry

end line_through_perpendicular_l54_54543


namespace thirtieth_number_sequence_l54_54695

theorem thirtieth_number_sequence :
  let a := 1
  let d := 2
  let n := 30
  in a + (n - 1) * d = 59 :=
by
  let a := 1
  let d := 2
  let n := 30
  show a + (n - 1) * d = 59
  sorry

end thirtieth_number_sequence_l54_54695


namespace minimum_pebbles_l54_54749

def pebble_set {m n : ℕ} (board: matrix (fin m) (fin n) ℕ) (i j : fin m × fin n) : finset ℕ :=
(finset.univ.image (λ a : fin m, board a j.snd)).union (finset.univ.image (λ b : fin n, board i.fst b))

def unique_pebble_set (m n k : ℕ) (board : matrix (fin m) (fin n) ℕ) : Prop :=
∀ i j i' j', i ≠ i' ∨ j ≠ j' → pebble_set board (i, j) ≠ pebble_set board (i', j')

theorem minimum_pebbles (n : ℕ) (board : matrix (fin (2 * n + 1)) (fin (2 * n + 1)) ℕ) :
  unique_pebble_set (2 * n + 1) (2 * n + 1) ((2 * n + 1) * 3 + 1) board :=
by
  have h : n = 1010 := by sorry
  have h_correct : unique_pebble_set (2021) (2021) 3031 board := by sorry
  exact h_correct

end minimum_pebbles_l54_54749


namespace cheaper_store_price_difference_in_cents_l54_54498

theorem cheaper_store_price_difference_in_cents :
  let list_price : ℝ := 59.99
  let discount_budget_buys := list_price * 0.15
  let discount_frugal_finds : ℝ := 20
  let sale_price_budget_buys := list_price - discount_budget_buys
  let sale_price_frugal_finds := list_price - discount_frugal_finds
  let difference_in_price := sale_price_budget_buys - sale_price_frugal_finds
  let difference_in_cents := difference_in_price * 100
  difference_in_cents = 1099.15 :=
by
  sorry

end cheaper_store_price_difference_in_cents_l54_54498


namespace exists_student_knows_one_other_l54_54285

variable (Student : Type) [Fintype Student] (knows : Student → Student → Prop)
variable (h : ∀ s t : Student, s ≠ t → (∃ n : ℕ, (Fintype.card {u | knows s u} = n ∧ Fintype.card {u | knows t u} = n)) → ¬ ∃ u : Student, knows s u ∧ knows t u)

theorem exists_student_knows_one_other :
  ∃ s : Student, Fintype.card {t : Student | knows s t} = 1 :=
begin
  sorry
end

end exists_student_knows_one_other_l54_54285


namespace volume_of_bounded_figure_l54_54153

-- Define the volume of a cube with edge length 1
def volume_of_cube (a : ℝ) : ℝ := a^3

-- Define the edge length of the smaller cubes
def small_cube_edge_length (a : ℝ) : ℝ := a / 2

-- Define the volume of a small cube
def volume_of_small_cube (a : ℝ) : ℝ := volume_of_cube (small_cube_edge_length a)

-- Theorem: Proving the volume of the bounded figure
theorem volume_of_bounded_figure (a : ℝ) : volume_of_cube a = 1 → 
  let V := volume_of_small_cube a in 8 * (V / 2) = 1 / 2 :=
begin
  sorry
end

end volume_of_bounded_figure_l54_54153


namespace f_increasing_l54_54253

def f : ℝ → ℝ :=
λ x, if x < 0 then x - sin x else x^3 + 1

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
sorry

end f_increasing_l54_54253


namespace reinforcement_arrival_l54_54856

theorem reinforcement_arrival (x : ℕ) :
  (2000 * 40) = (2000 * x + 4000 * 10) → x = 20 :=
by
  sorry

end reinforcement_arrival_l54_54856


namespace find_f1_f_minus1_l54_54569

theorem find_f1_f_minus1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f(x + 1) = x^2 + x) : f(1) + f(-1) = 2 := 
sorry

end find_f1_f_minus1_l54_54569


namespace convex_hexagons_count_l54_54535

theorem convex_hexagons_count (n k : ℕ) (h1 : n = 15) (h2 : k = 6) : nat.choose n k = 5005 := by
  rw [h1, h2]
  exact nat.choose_succ_succ 14 5
  sorry

end convex_hexagons_count_l54_54535


namespace smallest_whole_number_larger_than_any_triangle_perimeter_l54_54433

def is_valid_triangle (a b c : ℕ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem smallest_whole_number_larger_than_any_triangle_perimeter : 
  ∀ (s : ℕ), 16 < s ∧ s < 30 → is_valid_triangle 7 23 s → 
    60 = (Nat.succ (7 + 23 + s - 1)) := 
by 
  sorry

end smallest_whole_number_larger_than_any_triangle_perimeter_l54_54433


namespace vector_operation_result_l54_54943

def vector1 : ℕ × ℕ × ℕ := (-3, 2, -1)
def vector2 : ℕ × ℕ × ℕ := (1, 10, -2)
def scalar : ℕ := 2

def scalar_mult (a : ℕ) (v : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  (a * v.1, a * v.2, a * v.3)

def vector_add (v1 v2 : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

theorem vector_operation_result : 
  vector_add (scalar_mult scalar vector1) vector2 = (-5, 14, -4) :=
by 
  sorry

end vector_operation_result_l54_54943


namespace length_EB_l54_54160

-- Definitions of points and segments in an equilateral triangle
variables {A B C D E F : Type} [Semiring A]

-- Length conditions definitions
axiom AD_eq_4 : AD = 4
axiom DE_eq_8 : DE = 8
axiom EF_eq_10 : EF = 10
axiom FA_eq_6 : FA = 6
axiom ABC_equilateral : equilateral_triangle A B C

-- Conclusion to prove
theorem length_EB (AD DE EF FA : ℕ) :
  AD = 4 → DE = 8 → EF = 10 → FA = 6 → 
  (∀ x, x = Length_of_Side ABC) →
  Side_is_equilateral ABC → (some_proof_here ABC 12) → 
  EB = 2 :=
by 
  sorry

end length_EB_l54_54160


namespace solve_f_inequality_l54_54716

theorem solve_f_inequality
  (a : ℝ) (h_a : a < 0)
  (f : ℝ → ℝ) (h_f : ∀ x, f(x) = a*(x - 1)*(x - 3)) :
  (∀ t : ℝ, f(|t| + 8) < f(2 + t^2) ↔ (-3 < t ∧ t < 3 ∧ t ≠ 0)) :=
by
  sorry

end solve_f_inequality_l54_54716


namespace mode_and_median_of_data_set_l54_54772

def data_set : List ℕ := [3, 5, 4, 6, 3, 3, 4]

noncomputable def mode_of_data_set : ℕ :=
  sorry  -- The mode calculation goes here (implementation is skipped)

noncomputable def median_of_data_set : ℕ :=
  sorry  -- The median calculation goes here (implementation is skipped)

theorem mode_and_median_of_data_set :
  mode_of_data_set = 3 ∧ median_of_data_set = 4 :=
  by
    sorry  -- Proof goes here

end mode_and_median_of_data_set_l54_54772


namespace monotonicity_of_f_l54_54167

open Set

def f (x : ℝ) : ℝ := 1 / (x^2 - 1)

theorem monotonicity_of_f : 
  ∀ x1 x2 : ℝ, (1 < x1) ∧ (1 < x2) ∧ (x1 < x2) → f x1 > f x2 :=
by 
  sorry

end monotonicity_of_f_l54_54167


namespace smallest_s_value_l54_54052

-- Define the sequence terms and the condition.

variable {n : ℕ} (x : Fin n → ℝ)

def sum_eq_one : Prop := (∑ i, x i) = 1

def s := Finset.univ.sup (λ i, x i / (1 + ∑ j in Finset.range (i + 1), x j))

theorem smallest_s_value :
  sum_eq_one x →
  s x ≥ 1 - Real.sqrt (1 / 2) :=
sorry

end smallest_s_value_l54_54052


namespace weight_b_calculation_l54_54030

-- Definitions based on conditions
def weight_from_height (h : ℝ) : ℝ := 2 * h^2 + 3 * h - 5

variable (wa wb wc : ℝ)
variable (ha hb hc : ℝ)

-- Given average conditions
def avg_weights : Prop := (wa + wb + wc) / 3 = 45
def avg_weights_a_b : Prop := (wa + wb) / 2 = 40
def avg_weights_b_c : Prop := (wb + wc) / 2 = 43

-- Quadratic relationship condition
def weight_height_relation_a : Prop := wa = weight_from_height ha
def weight_height_relation_b : Prop := wb = weight_from_height hb
def weight_height_relation_c : Prop := wc = weight_from_height hc

-- The theorem we want to prove
theorem weight_b_calculation 
    (h_avg : (ha + hc) / 2 = 155)
    (h_avg_weights : avg_weights)
    (h_avg_weights_a_b : avg_weights_a_b)
    (h_avg_weights_b_c : avg_weights_b_c)
    (h_relation_a : weight_height_relation_a)
    (h_relation_b : weight_height_relation_b)
    (h_relation_c : weight_height_relation_c)
    : wb = 31 := 
  by sorry

end weight_b_calculation_l54_54030


namespace parity_of_f_zero_range_of_m_range_of_a_for_inequality_l54_54251

-- Definitions for the given functions
def f (x a : ℝ) : ℝ := x^2 - a * x - a
def g (x : ℝ) : ℝ := -Real.exp x
def F (x a : ℝ) : ℝ := x * (f x a)

-- Prove parity of f for a = 0
theorem parity_of_f_zero : ∀ x : ℝ, (f x 0) = f (-x) 0 := 
by sorry

-- Prove the range of m for F when a = 1, F has 3 distinct real roots
theorem range_of_m : ∀ m : ℝ, 
  (F 1 1 = x^3 - x^2 - x) ∧ (3.distinct.real.roots { x : ℝ | F x 1 = m) → -1 < m ∧ m < 5/27 :=
by sorry

-- Prove the range of a for the given inequality condition
theorem range_of_a_for_inequality : ∀ a : ℝ, 
  (∀ x1 x2 ∈ Icc 0 (Real.exp 1), x1 > x2 → abs (f x1 a - f x2 a) < abs (g x1 - g x2)) → 
  (2 * Real.log 2 - 2 ≤ a ∧ a ≤ 1) :=
by sorry

end parity_of_f_zero_range_of_m_range_of_a_for_inequality_l54_54251


namespace time_interval_between_trains_l54_54129

def train1_speed := 40 -- speed of the first train in kmph
def train2_speed := 80 -- speed of the second train in kmph
def meeting_distance := 80 -- distance where the trains meet in km

-- The time after the first train leaves, when the second train leaves (in hours)
def time_between_trains : ℚ := 2 / 3

theorem time_interval_between_trains :
  let t := time_between_trains in
  train1_speed * t + train2_speed * t = meeting_distance :=
by
  simp only [train1_speed, train2_speed, meeting_distance, time_between_trains]
  sorry

end time_interval_between_trains_l54_54129


namespace drop_perpendicular_l54_54061

open Classical

-- Definitions for geometrical constructions on the plane
structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(p1 : Point)
(p2 : Point)

-- Condition 1: Drawing a line through two points
def draw_line (A B : Point) : Line := {
  p1 := A,
  p2 := B
}

-- Condition 2: Drawing a perpendicular line through a given point on a line
def draw_perpendicular (l : Line) (P : Point) : Line :=
-- Details of construction skipped, this function should return the perpendicular line
sorry

-- The problem: Given a point A and a line l not passing through A, construct the perpendicular from A to l
theorem drop_perpendicular : 
  ∀ (A : Point) (l : Line), ¬ (A = l.p1 ∨ A = l.p2) → ∃ (P : Point), ∃ (m : Line), (m = draw_perpendicular l P) ∧ (m.p1 = A) :=
by
  intros A l h
  -- Details of theorem-proof skipped, assert the existence of P and m as required
  sorry

end drop_perpendicular_l54_54061


namespace darij_grinberg_inequality_l54_54706

theorem darij_grinberg_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  a + b + c ≤ (bc / (b + c)) + (ca / (c + a)) + (ab / (a + b)) + (1 / 2 * ((bc / a) + (ca / b) + (ab / c))) := 
by sorry

end darij_grinberg_inequality_l54_54706


namespace unique_satisfying_floor_eq_l54_54540

-- Define the function to check the greatest integer less than or equal to s
def floor (s: ℝ) : ℤ := ⌊s⌋

-- The proposition to be proved
theorem unique_satisfying_floor_eq (s : ℝ) (h : floor s + s = 17.5) : s = 8.5 :=
by
  -- Proof goes here
  sorry

end unique_satisfying_floor_eq_l54_54540


namespace seating_arrangements_l54_54797

theorem seating_arrangements : ∃ (n : ℕ), n = 20 ∧ 
  (∀ (seats : Fin 9 → ℕ), 
    (∀ (i : Fin 9), seats i = 0 ∨ seats i = 1) ∧ (∀ (i : Fin 9), seats i = 1 → 
      ((i.val > 0 → seats (⟨i.val - 1, by linarith⟩) = 0) ∧ 
       (i.val < 8 → seats (⟨i.val + 1, by linarith⟩) = 0))) ∧ 
    (seats ⟨4, by linarith⟩ = 1) ∧ 
    ((∃ (b c : Fin 9), b ≠ c ∧ seats b = 1 ∧ seats c = 1) →
      (b.val < 4 → c.val > 4) ∨ (c.val < 4 → b.val > 4))) → false := sorry

end seating_arrangements_l54_54797


namespace find_PS_l54_54304

variables {P Q R S : Type} [real : P ≠ Q] [real : P ≠ R] [real : Q ≠ R]

def foot_of_perpendicular (P : Type) (Q R S: P) :=
-- Placeholder for geometric definition of the perpendicular foot
sorry

theorem find_PS
  (PQ PR QS SR : ℝ)
  (h : ℝ)
  (h_sq : h * h = 117.025)
  (foot_S : foot_of_perpendicular P Q R S)
  (ratio_QS_SR : QS / SR = 3 / 7)
  (PQ_eq : PQ = 13)
  (PR_eq : PR = 20) :
  (PS = sqrt 117.025) :=
begin
  -- Proof placeholder
  sorry
end

end find_PS_l54_54304


namespace num_three_digit_integers_l54_54623

theorem num_three_digit_integers : 
  (set.univ : set ℤ).count (λ n, ∃ a b c, 
    {a, b, c} ⊆ {1, 2, 3, 4, 6, 7} ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    n = a * 100 + b * 10 + c) = 120 := 
sorry

end num_three_digit_integers_l54_54623


namespace rodney_guess_count_l54_54758

theorem rodney_guess_count :
  ∃ n : ℕ, (n = 7) ∧ (∀ m : ℕ, (75 < m ∧ m < 100) → 
  (odd (m / 10) ∧ odd (m % 10)) → n = 7) := 
begin
  sorry
end

end rodney_guess_count_l54_54758


namespace marion_paperclips_correct_l54_54077

noncomputable def yun_initial : ℕ := 20
noncomputable def yun_loss : ℕ := 12
noncomputable def yun_current : ℕ := yun_initial - yun_loss
noncomputable def marion_additional_fraction : ℚ := 1 / 4
noncomputable def marion_additional : ℕ := (marion_additional_fraction * yun_current).to_nat
noncomputable def marion_extra : ℕ := 7

noncomputable def marion_paperclips : ℕ := yun_current + marion_additional + marion_extra

theorem marion_paperclips_correct : marion_paperclips = 17 :=
by
  sorry

end marion_paperclips_correct_l54_54077


namespace projection_of_b_on_a_l54_54599

open RealInnerProductSpace

variable {V : Type} [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

variables (a b : V)

theorem projection_of_b_on_a 
  (hab : (∥a∥ > 0) ∧ (∥b∥ > 0)) 
  (ha : ∥a∥ = 2) 
  (orthogonal : inner a (a + 2 • b) = 0) :
  (inner b a / ∥a∥) = -1 :=
by
  sorry

end projection_of_b_on_a_l54_54599


namespace annual_return_percentage_l54_54900

theorem annual_return_percentage (initial_value final_value gain : ℕ)
    (h1 : initial_value = 8000)
    (h2 : final_value = initial_value + 400)
    (h3 : gain = final_value - initial_value) :
    (gain * 100 / initial_value) = 5 := by
  sorry

end annual_return_percentage_l54_54900


namespace values_of_c_for_one_asymptote_l54_54947

def has_exactly_one_vertical_asymptote (g : ℝ → ℝ) : Prop :=
  ∃ c, (g = λ x, (x^2 - 2*x + c) / (x^2 - x - 12)) ∧ 
       (x = 4 ∨ x = -3) ∧
       ¬((x = 4 ∧ x = -3))

theorem values_of_c_for_one_asymptote :
  ∀ c : ℝ,
    (has_exactly_one_vertical_asymptote (λ x, (x^2 - 2*x + c) / (x^2 - x - 12))) ↔ 
    (c = -8 ∨ c = -3) :=
by sorry

end values_of_c_for_one_asymptote_l54_54947


namespace greatest_integer_floor_div_l54_54429

-- Define the parameters
def a : ℕ := 3^100 + 2^105
def b : ℕ := 3^96 + 2^101

-- Formulate the proof statement
theorem greatest_integer_floor_div (a b : ℕ) : 
  a = 3^100 + 2^105 →
  b = 3^96 + 2^101 →
  (a / b) = 16 := 
by
  intros ha hb
  sorry

end greatest_integer_floor_div_l54_54429


namespace distance_against_stream_l54_54118

noncomputable def rowing_distance (d : ℝ) (v_m : ℝ) (t1 t2 : ℝ) (v_s : ℝ) : Prop :=
  (d = (v_m - v_s) * t1) ∧ (d = (v_m + v_s) * t2)

theorem distance_against_stream :
  ∃ d, rowing_distance d 5 675 450 1 :=
by {
  let d := 2700,
  use d,
  split;
  norm_num
}

end distance_against_stream_l54_54118


namespace evaluate_81_power_5_div_4_l54_54187

-- Define the conditions
def base_factorized : ℕ := 3 ^ 4
def power_rule (b : ℕ) (m n : ℝ) : ℝ := (b : ℝ) ^ m ^ n

-- Define the primary calculation
noncomputable def power_calculation : ℝ := 81 ^ (5 / 4)

-- Prove that the calculation equals 243
theorem evaluate_81_power_5_div_4 : power_calculation = 243 := 
by
  have h1 : base_factorized = 81 := by sorry
  have h2 : power_rule 3 4 (5 / 4) = 3 ^ 5 := by sorry
  have h3 : 3 ^ 5 = 243 := by sorry
  have h4 : power_calculation = power_rule 3 4 (5 / 4) := by sorry
  rw [h1, h2, h3, h4]
  exact h3

end evaluate_81_power_5_div_4_l54_54187


namespace sum_of_digits_201_l54_54958

theorem sum_of_digits_201 
  (a : ℕ → ℕ) 
  (h_length : ∀ k, 1 ≤ k ∧ k ≤ 29 → 0 ≤ a k ∧ a k ≤ 9)
  (h_nonzero : a 1 ≠ 0)
  (h_symmetry : ∀ k, 1 ≤ k ∧ k ≤ 29 → a k = a (30 - k)) 
  : (∑ k in finset.range 29, a (k + 1)) = 201 := 
sorry

end sum_of_digits_201_l54_54958


namespace full_weight_l54_54055

-- conditions
variables (c d : ℝ)

-- Definitions based on conditions
def eq1 (x y : ℝ) := x + (3 / 4) * y = c
def eq2 (x y : ℝ) := x + (1 / 3) * y = d

theorem full_weight (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : x + y = (8 / 5) * c - (3 / 5) * d :=
sorry

end full_weight_l54_54055


namespace thirty_percent_of_x_l54_54449

noncomputable def x : ℝ := 160 / 0.40

theorem thirty_percent_of_x (h : 0.40 * x = 160) : 0.30 * x = 120 :=
sorry

end thirty_percent_of_x_l54_54449


namespace PQ_eq_PT_l54_54734

variables {A B C O I D E F P Q T : Point}
variable {ω : Circle}
variable [noncomputable_ring] -- To handle non-computable aspects of geometry like points and circles

-- Definitions from the conditions
def scalene_triangle (ABC : Triangle) : Prop := ∀ A B C, ¬ (A = B ∨ B = C ∨ C = A)
def circumcenter (ABC : Triangle) (O : Point) : Prop := Circumcenter O ABC
def incenter (ABC : Triangle) (I : Point) : Prop := Incenter I ABC
def incircle (ABC : Triangle) (ω : Circle) (D E F : Point) : Prop := ω = Incircle ABC ∧ ω.TangentAt (D E F) ∧ (ω.TangentAt D BC ∧ ω.TangentAt E CA ∧ ω.TangentAt F AB)
def foot_of_altitude (D Point) (EF : Line) (P : Point) : Prop := FootOfAltitude P D EF
def line_intersect (DP : Line) (ω : Circle) (Q : Point) : Prop := DP.IntersectCircle ω ∧ Q ≠ D ∧ ω.TangentAt Q DP
def line_intersect_altitude (OI : Line) (BC_alt : Altitude A BC) (T : Point) : Prop := OI ∥ BC ∧ OI.IntersectAltitude BC_alt T

-- Statement of the problem
theorem PQ_eq_PT
  (h₁ : scalene_triangle (Triangle A B C))
  (h₂ : circumcenter (Triangle A B C) O)
  (h₃ : incenter (Triangle A B C) I)
  (h₄ : incircle (Triangle A B C) ω D E F)
  (h₅ : foot_of_altitude D (Line E F) P)
  (h₆ : line_intersect (Line D P) ω Q)
  (h₇ : line_intersect_altitude (Line O I) (Altitude A B C) T)
  : PQ = PT :=
sorry

end PQ_eq_PT_l54_54734


namespace negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l54_54972

theorem negation_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem converse_of_p (π : ℝ) (a b c d : ℚ) (h : a = c ∧ b = d) : a * π + b = c * π + d :=
  sorry

theorem inverse_of_p (π : ℝ) (a b c d : ℚ) (h : a * π + b ≠ c * π + d) : a ≠ c ∨ b ≠ d :=
  sorry

theorem contrapositive_of_p (π : ℝ) (a b c d : ℚ) (h : a ≠ c ∨ b ≠ d) : a * π + b ≠ c * π + d :=
  sorry

theorem original_p (π : ℝ) (a b c d : ℚ) (h : a * π + b = c * π + d) : a = c ∧ b = d :=
  sorry

end negation_of_p_converse_of_p_inverse_of_p_contrapositive_of_p_original_p_l54_54972


namespace log_base_9_of_3_l54_54527

theorem log_base_9_of_3 : log 9 3 = 1 / 2 :=
by sorry

end log_base_9_of_3_l54_54527


namespace correct_option_B_l54_54075

-- Definitions based on conditions
def prism (P : Type) : Prop := -- Definition of a prism
sorry

def straight_prism (P : Type) : Prop := -- Definition of a straight prism
sorry

def quadrilateral (P : Type) : Prop := -- Definition of quadrilateral
sorry

def square (Q : Type) : Prop := -- Definition of square
sorry

def regular_prism (P : Type) : Prop := -- Definition of a regular prism
sorry

-- Theorem based on the problem statement and conditions
theorem correct_option_B (P : Type) [prism P] [straight_prism P] [quadrilateral P] [square P] :
  regular_prism P := sorry

end correct_option_B_l54_54075


namespace exists_nonzero_multiple_of_k_with_four_distinct_digits_l54_54765

theorem exists_nonzero_multiple_of_k_with_four_distinct_digits (k : ℤ) (h1 : 1 < k) : 
  ∃ m : ℤ, (m ≠ 0) ∧ (m < k^4) ∧ (∀ d ∈ List.ofDigits 10 m, d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 9) := 
sorry

end exists_nonzero_multiple_of_k_with_four_distinct_digits_l54_54765


namespace zoe_total_spent_l54_54078

theorem zoe_total_spent :
  let cost_of_app := 5
  let monthly_cost := 8
  let months_played := 2
  let total_cost := cost_of_app + (monthly_cost * months_played)
  in total_cost = 21 :=
by
  sorry

end zoe_total_spent_l54_54078


namespace factor_product_modulo_l54_54819

theorem factor_product_modulo (h1 : 2021 % 23 = 21) (h2 : 2022 % 23 = 22) (h3 : 2023 % 23 = 0) (h4 : 2024 % 23 = 1) (h5 : 2025 % 23 = 2) :
  (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 0 :=
by
  sorry

end factor_product_modulo_l54_54819


namespace anthony_transactions_percentage_more_l54_54008

noncomputable def transactions_mabel : ℕ := 90
noncomputable def transactions_jade : ℕ := 81
noncomputable def transactions_more_than_cal : ℕ := 15
noncomputable def factor_cal_anthony : ℚ := 2 / 3

theorem anthony_transactions_percentage_more :
  ∃ (A P : ℚ), 
  let C := transactions_jade - transactions_more_than_cal in
  let A := (C / factor_cal_anthony : ℚ) in
  let P := ((A - transactions_mabel) / transactions_mabel) * 100 in
  P = 10 :=
by
  sorry

end anthony_transactions_percentage_more_l54_54008


namespace fold_paper_to_dodecagon_l54_54226

-- Definitions based on the given conditions
structure PaperSquare :=
  (grid : ℕ × ℕ := (4, 4))
  (allowed_folds : set (ℕ × ℕ × ℕ × ℕ))
  (side_folds : ∀ i j, (i, j) ∈ allowed_folds ∧ (i+1, j) ∈ allowed_folds ∧ (i, j+1) ∈ allowed_folds)
  (diagonal_folds : ∀ i j, (i, j) ∈ allowed_folds ∧ (i+1, j+1) ∈ allowed_folds)

-- Proof problem statement: Is it possible to obtain a 12-sided polygon?
theorem fold_paper_to_dodecagon :
  ∃ dodecagon : set (ℕ × ℕ), 
    let sq : PaperSquare :=
    { grid := (4, 4),  allowed_folds := {v | ∃ (i j : ℕ), (v = (i, j, i+1, j) ∨ v = (i, j, i, j+1) ∨ v = (i, j, i+1, j+1))},
      side_folds := by intros i j; exact ⟨rfl, rfl, rfl⟩, diagonal_folds := by intros i j; exact ⟨rfl, rfl⟩ } in
     (12 : ℕ) ∈ dodecagon.sides :=
sorry

end fold_paper_to_dodecagon_l54_54226


namespace train_length_l54_54876

theorem train_length (time_crossing : ℕ) (speed_kmh : ℕ) (conversion_factor : ℕ) (expected_length : ℕ) :
  time_crossing = 4 ∧ speed_kmh = 144 ∧ conversion_factor = 1000 / 3600 * 144 →
  expected_length = 160 :=
by
  sorry

end train_length_l54_54876


namespace minimum_S_l54_54553

def count_angles (n : ℕ) (points : Fin n → ℝ × ℝ) : ℕ :=
  let angles := {p | ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ n ∧ ∠ points i points j ≤ 2 * π / 3}
  angles.card

theorem minimum_S (n : ℕ) (h : n ≥ 3) (points : Fin n → ℝ × ℝ) :
  let S := count_angles n points
  S ≥ if n % 2 = 1 then (n - 1) * (n - 1) / 4 else n * n / 4 - n / 2 := 
sorry

end minimum_S_l54_54553


namespace min_diff_m_n_l54_54602

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x + 1 / x else -x + 1 / (-x)

lemma even_function (x : ℝ) : f(x) = f(-x) :=
begin
  unfold f,
  split_ifs;
  ring,
end

theorem min_diff_m_n : ∀ (m n : ℝ), 
  (∀ x ∈ Icc (-3 : ℝ) (-1 : ℝ), n ≤ f x ∧ f x ≤ m) → 
  m - n = 4 / 3 :=
begin
  intros m n h,
  have h_neg3 : f (-3) = 10 / 3,
  { unfold f, split_ifs; ring },
  have h_neg1 : f (-1) = 2,
  { unfold f, split_ifs; ring },
  cases h (-3) (by norm_num) with h_nneg3 h_mneg3,
  cases h (-1) (by norm_num) with h_nneg1 h_mneg1,
  have h_m : m = 10 / 3 := by linarith,
  have h_n : n = 2 := by linarith,
  rw [h_m, h_n],
  norm_num,
end

end min_diff_m_n_l54_54602


namespace total_surface_area_l54_54505

def base_layer_area : ℕ := 2 * 3 * 2 - 4
def middle_layer_area : ℕ := 4 * 2 - 2
def top_layer_area : ℕ := 2 * 1
def side_area : ℕ := 2 * 3 + 2 * 3 + 2 * 2 + 2 * 2
  
theorem total_surface_area : base_layer_area + middle_layer_area + top_layer_area + side_area = 36 :=
by
  have h_base_layer : base_layer_area = 8 := rfl
  have h_middle_layer : middle_layer_area = 6 := rfl
  have h_top_layer : top_layer_area = 2 := rfl
  have h_side_area : side_area = 20 := rfl
  rw [h_base_layer, h_middle_layer, h_top_layer, h_side_area]
  norm_num

end total_surface_area_l54_54505


namespace options_correct_l54_54074

theorem options_correct :
    ( (\sin 15 - \cos 15)^2 = 1/2 ) ∧
    ¬ ( \sin^2 22.5 - \cos^2 22.5 = \sqrt(2)/2 ) ∧
    ( \sin 40 * (\tan 10 - \sqrt 3) = -1 ) ∧
    ( \cos 24 * \cos 36 - \cos 66 * \cos 54 = 1/2 ) :=
by
    sorry

end options_correct_l54_54074


namespace negation_exists_to_forall_l54_54393

theorem negation_exists_to_forall (P : ℝ → Prop) (h : ∃ x : ℝ, x^2 + 3 * x + 2 < 0) :
  (¬ (∃ x : ℝ, x^2 + 3 * x + 2 < 0)) ↔ (∀ x : ℝ, x^2 + 3 * x + 2 ≥ 0) := by
sorry

end negation_exists_to_forall_l54_54393


namespace triangle_to_rectangle_ratio_l54_54135

def triangle_perimeter := 60
def rectangle_perimeter := 60

def is_equilateral_triangle (side_length: ℝ) : Prop :=
  3 * side_length = triangle_perimeter

def is_valid_rectangle (length width: ℝ) : Prop :=
  2 * (length + width) = rectangle_perimeter ∧ length = 2 * width

theorem triangle_to_rectangle_ratio (s l w: ℝ) 
  (ht: is_equilateral_triangle s) 
  (hr: is_valid_rectangle l w) : 
  s / w = 2 := by
  sorry

end triangle_to_rectangle_ratio_l54_54135


namespace pulley_problem_l54_54442

variables (m M d θ μ g : ℝ)
noncomputable def t (a : ℝ) :=
  sqrt (2 * d / a)

theorem pulley_problem
  (h_m : m = 1.0)
  (h_M : M = 2.0)
  (h_d : d = 1.0)
  (h_θ : θ = 10 * (Real.pi/180))
  (h_μ : μ = 0.50)
  (h_g : g = 9.8)
  (h_t : t 4.88 = 0.64) : 
  100*t 4.88 = 64 :=
sorry

end pulley_problem_l54_54442


namespace min_value_S_l54_54211

open Real

noncomputable def S (x a : ℝ) : ℝ := (x - a)^2 + (ln x - a)^2

theorem min_value_S (a : ℝ) : ∃ x : ℝ, S x a = 1 / 2 :=
by
  sorry

end min_value_S_l54_54211


namespace half_vectorAB_is_2_1_l54_54233

def point := ℝ × ℝ -- Define a point as a pair of real numbers
def vector := ℝ × ℝ -- Define a vector as a pair of real numbers

def A : point := (-1, 0) -- Define point A
def B : point := (3, 2) -- Define point B

noncomputable def vectorAB : vector := (B.1 - A.1, B.2 - A.2) -- Define vector AB as B - A

noncomputable def half_vectorAB : vector := (1 / 2 * vectorAB.1, 1 / 2 * vectorAB.2) -- Define half of vector AB

theorem half_vectorAB_is_2_1 : half_vectorAB = (2, 1) := by
  -- Sorry is a placeholder for the proof
  sorry

end half_vectorAB_is_2_1_l54_54233


namespace min_rectangles_Vasya_min_rectangle_cover_l54_54814

-- Define the conditions
def conditions :=
  let first_type_corners := 12
  let second_type_corners := 12
  let groups_of_three := 4
  (first_type_corners, second_type_corners, groups_of_three)

-- Define the problem statement theorem
theorem min_rectangles (h : conditions) : Nat :=
  match h with
  | (first_type_corners, second_type_corners, groups_of_three) =>
    -- Prove that the minimum number of rectangles needed is 12
    12

-- Now we state the proof goal
theorem Vasya_min_rectangle_cover : min_rectangles conditions = 12 :=
by
  sorry

end min_rectangles_Vasya_min_rectangle_cover_l54_54814


namespace correct_marks_per_answer_l54_54290

variable {x : ℕ}
variable (totalQuestions : ℕ) (correctQuestions : ℕ) (wrongQuestions : ℕ) (totalMarks : ℕ) (marksPerCorrectAnswer : ℕ)

-- Define the conditions
def contest_conditions := totalQuestions = 60 ∧ correctQuestions = 42 ∧ wrongQuestions = totalQuestions - correctQuestions ∧ totalMarks = 150

-- Define the equation derived from the conditions
def marks_equation := marksPerCorrectAnswer * correctQuestions - wrongQuestions = totalMarks

-- Prove that the number of marks for each correct answer is 4
theorem correct_marks_per_answer : contest_conditions ∧ marks_equation → marksPerCorrectAnswer = 4 := by
  sorry

end correct_marks_per_answer_l54_54290


namespace product_four_integers_sum_to_50_l54_54948

theorem product_four_integers_sum_to_50 (E F G H : ℝ) 
  (h₀ : E + F + G + H = 50)
  (h₁ : E - 3 = F + 3)
  (h₂ : E - 3 = G * 3)
  (h₃ : E - 3 = H / 3) :
  E * F * G * H = 7461.9140625 := 
sorry

end product_four_integers_sum_to_50_l54_54948


namespace find_k_l54_54212

def vector := (ℝ × ℝ)
def a : vector := (1, 2)
def b : vector := (-3, 2)

def parallel (v1 v2 : vector) : Prop := 
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_k (k : ℝ) : 
  let u := (k * a.1 + b.1, k * a.2 + b.2)
  let v := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  parallel u v → k = -1/3 := 
by
  intros u v h_parallel
  rw [a, b] at u v
  sorry

end find_k_l54_54212


namespace quadratic_two_distinct_real_roots_l54_54277

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 - 6 * x + k = 0) ↔ k < 9 :=
by
  sorry

end quadratic_two_distinct_real_roots_l54_54277


namespace find_xy_l54_54997

theorem find_xy (x y : ℝ) (h : {x, y^2, 1} = {1, 2*x, y}) : x = 2 ∧ y = 2 := 
sorry

end find_xy_l54_54997


namespace geometric_seq_product_increasing_imp_nonge_deq_l54_54967

variable {α : Type} [LinearOrderedField α]

/-- 
  Given an infinite geometric sequence {a_n}, the sum and product of the first n terms are 
  denoted as S_n and T_n respectively. If T_n is an increasing sequence, 
  then a_{2022} ≥ a_{2021}.
-/
theorem geometric_seq_product_increasing_imp_nonge_deq
  (a : ℕ → α) (S T : ℕ → α)
  (hS : ∀ n, S n = (finset.range n).sum a)
  (hT : ∀ n, T n = (finset.range n).prod a)
  (h_inc_T : ∀ n, T (n + 1) > T n) :
  a 2022 ≥ a 2021 :=
sorry

end geometric_seq_product_increasing_imp_nonge_deq_l54_54967


namespace johns_average_speed_remaining_duration_l54_54697

noncomputable def average_speed_remaining_duration : ℝ :=
  let total_distance := 150
  let total_time := 3
  let first_hour_speed := 45
  let stop_time := 0.5
  let next_45_minutes_speed := 50
  let next_45_minutes_time := 0.75
  let driving_time := total_time - stop_time
  let distance_first_hour := first_hour_speed * 1
  let distance_next_45_minutes := next_45_minutes_speed * next_45_minutes_time
  let remaining_distance := total_distance - distance_first_hour - distance_next_45_minutes
  let remaining_time := driving_time - (1 + next_45_minutes_time)
  remaining_distance / remaining_time

theorem johns_average_speed_remaining_duration : average_speed_remaining_duration = 90 := by
  sorry

end johns_average_speed_remaining_duration_l54_54697


namespace angle_AFG_is_175_l54_54047
open Real EuclideanGeometry

/-- The quadrilateral ABCD is a square.
    Point E is such that ∠CDE = 100°.
    Point F is on the extension of AD beyond D such that DE = DF.
    Point G is on the extension of AD beyond D.
    Then, ∠AFG = 175°. --/
theorem angle_AFG_is_175 (A B C D E F G : Point)
  (h1 : quadrilateral A B C D)
  (h2 : is_square A B C D)
  (h3 : angle C D E = 100)
  (h4 : collinear D A G)
  (h5 : collinear D F G)
  (h6 : dist D E = dist D F) :
  angle A F G = 175 := 
by
  sorry

end angle_AFG_is_175_l54_54047


namespace min_rp_rq_value_l54_54576

theorem min_rp_rq_value (x0 y0 : ℝ) (hx : x0 > 0) (hy : y0 > 0) 
    (hM : x0^2 - y0^2 = 1) :
    ∃ (R P Q : ℝ × ℝ), 
    let RP := (R.1 - P.1, R.2 - P.2) in
    let RQ := (R.1 - Q.1, R.2 - Q.2) in
    RP.1 * RP.1 + RP.2 * RP.2 * RQ.1 * RQ.1 + RQ.2 * RQ.2 = -1/2 :=
sorry

end min_rp_rq_value_l54_54576


namespace c_pow_a_gt_b_pow_c_l54_54969

noncomputable def a : ℝ := 2^(-1 / Real.exp 1)
noncomputable def b : ℝ := 2^(-1 / 3)
noncomputable def c : ℝ := Real.exp (1 / Real.exp 1)

theorem c_pow_a_gt_b_pow_c : c^a > b^c :=
by
  have h1 : a = 2^(-1 / Real.exp 1) := rfl
  have h2 : b = 2^(-1 / 3) := rfl
  have h3 : c = Real.exp (1 / Real.exp 1) := rfl
  sorry

end c_pow_a_gt_b_pow_c_l54_54969


namespace net_pay_calculation_l54_54278

theorem net_pay_calculation
  (gross_pay : ℝ)
  (taxes_paid : ℝ)
  (net_pay : ℝ) : 
  gross_pay = 450 → 
  taxes_paid = 135 → 
  net_pay = gross_pay - taxes_paid → 
  net_pay = 315 :=
by 
  intros h₁ h₂ h₃ 
  rw [h₁, h₂] at h₃ 
  exact h₃

end net_pay_calculation_l54_54278


namespace quadratic_result_l54_54597

noncomputable def quadratic_has_two_positive_integer_roots (k p : ℕ) : Prop :=
  ∃ x1 x2 : ℕ, x1 > 0 ∧ x2 > 0 ∧ (k - 1) * x1 * x1 - p * x1 + k = 0 ∧ (k - 1) * x2 * x2 - p * x2 + k = 0

theorem quadratic_result (k p : ℕ) (h1 : k = 2) (h2 : quadratic_has_two_positive_integer_roots k p) :
  k^(k*p) * (p^p + k^k) = 1984 :=
by
  sorry

end quadratic_result_l54_54597


namespace find_f_neg2_l54_54717

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - 3 else -(2^(-x) - 3)

theorem find_f_neg2 : f (-2) = -1 :=
sorry

end find_f_neg2_l54_54717


namespace circumradius_inequality_equality_conditions_l54_54346

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  if h : a + b > c ∧ a + c > b ∧ b + c > a then
    sqrt ((a * b * c) / sqrt ((a + b + c) * (a - b + c) * (a + b - c) * (a - b - c)))
  else 0

theorem circumradius_inequality
  {a b c R : ℝ}
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)
  (hR : R = circumradius a b c) :
  R ≥ (a^2 + b^2) / (2 * sqrt(2 * a^2 + 2 * b^2 - c^2)) :=
sorry

theorem equality_conditions
  {a b c R : ℝ}
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a)
  (hR : R = circumradius a b c) :
  R = (a^2 + b^2) / (2 * sqrt(2 * a^2 + 2 * b^2 - c^2)) ↔ (a = b ∨ (a^2 + b^2 = c^2)) :=
sorry

end circumradius_inequality_equality_conditions_l54_54346


namespace last_digit_of_product_l54_54817

theorem last_digit_of_product :
    (3 ^ 65 * 6 ^ 59 * 7 ^ 71) % 10 = 4 := 
  by sorry

end last_digit_of_product_l54_54817


namespace probability_blue_then_red_l54_54462

/--
A box contains 15 balls, of which 5 are blue and 10 are red.
Two balls are drawn sequentially from the box without returning the first ball to the box.
Prove that the probability that the first ball drawn is blue and the second ball is red is 5 / 21.
-/
theorem probability_blue_then_red :
  let total_balls := 15
  let blue_balls := 5
  let red_balls := 10
  let first_is_blue := (blue_balls : ℚ) / total_balls
  let second_is_red_given_blue := (red_balls : ℚ) / (total_balls - 1)
  first_is_blue * second_is_red_given_blue = 5 / 21 := by
  sorry

end probability_blue_then_red_l54_54462


namespace correct_operation_l54_54438

theorem correct_operation :
  (∀ {a : ℝ}, a^6 / a^3 = a^3) = false ∧
  (∀ {a b : ℝ}, (a + b) * (a - b) = a^2 - b^2) ∧
  (∀ {a : ℝ}, (-a^3)^3 = -a^9) = false ∧
  (∀ {a : ℝ}, 2 * a^2 + 3 * a^3 = 5 * a^5) = false :=
by
  sorry

end correct_operation_l54_54438


namespace retailer_profit_percentage_l54_54476

noncomputable def profit_percentage : ℝ :=
  let market_price_per_pen := 1.20
  let total_pens := 150
  let bulk_price_pens := 50
  let discount_1 := 0.10
  let discount_2 := 0.05
  let tax_rate := 0.02
  let discount_price num_pens discount := num_pens * (market_price_per_pen * (1 - discount))
  let no_discount_price num_pens := num_pens * market_price_per_pen
  let tax amount := amount * tax_rate

  let cost := bulk_price_pens * market_price_per_pen
  let revenue1 := discount_price 50 discount_1
  let revenue2 := discount_price 50 discount_2
  let revenue3 := no_discount_price 50
  let total_revenue := revenue1 + revenue2 + revenue3
  let total_tax := tax revenue1 + tax revenue2 + tax revenue3
  let total_received := total_revenue + total_tax
  let profit := total_received - cost

  (profit / cost) * 100

theorem retailer_profit_percentage (market_price_per_pen total_pens cost)
: market_price_per_pen = 1.20 
→ total_pens = 150 
→ cost = 50 * market_price_per_pen 
→ (profit_percentage ≈ 190.7) :=
by {
  intros,
  sorry
}

end retailer_profit_percentage_l54_54476


namespace sleepySquirrelNutsPerDay_l54_54003

def twoBusySquirrelsNutsPerDay : ℕ := 2 * 30
def totalDays : ℕ := 40
def totalNuts : ℕ := 3200

theorem sleepySquirrelNutsPerDay 
  (s  : ℕ) 
  (h₁ : 2 * 30 * totalDays + s * totalDays = totalNuts) 
  : s = 20 := 
  sorry

end sleepySquirrelNutsPerDay_l54_54003


namespace mean_equality_l54_54783

theorem mean_equality (z : ℚ) :
  ((8 + 7 + 28) / 3 : ℚ) = (14 + z) / 2 → z = 44 / 3 :=
by
  sorry

end mean_equality_l54_54783


namespace math_bonanza_2016_8_l54_54707

def f (x : ℕ) := x^2 + x + 1

theorem math_bonanza_2016_8 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : f p = f q + 242) (hpq : p > q) :
  (p, q) = (61, 59) :=
by sorry

end math_bonanza_2016_8_l54_54707


namespace treehouse_paint_cost_l54_54054

theorem treehouse_paint_cost
  (white_paint_oz : Float := 20)
  (green_paint_oz : Float := 15)
  (brown_paint_oz : Float := 34)
  (blue_paint_oz : Float := 12)
  (evaporation_loss_rate : Float := 0.10)
  (white_paint_cost_per_liter : Float := 8.50)
  (green_paint_cost_per_liter : Float := 7.20)
  (brown_paint_cost_per_liter : Float := 6.90)
  (blue_paint_cost_per_liter : Float := 9.00)
  (ounce_to_liter : Float := 33.814) :
  let total_white_paint := white_paint_oz * (1 + evaporation_loss_rate)
  let total_green_paint := green_paint_oz * (1 + evaporation_loss_rate)
  let total_brown_paint := brown_paint_oz * (1 + evaporation_loss_rate)
  let total_blue_paint := blue_paint_oz * (1 + evaporation_loss_rate)
  let white_paint_liters := total_white_paint / ounce_to_liter
  let green_paint_liters := total_green_paint / ounce_to_liter
  let brown_paint_liters := total_brown_paint / ounce_to_liter
  let blue_paint_liters := total_blue_paint / ounce_to_liter
  let white_paint_cost := white_paint_liters * white_paint_cost_per_liter
  let green_paint_cost := green_paint_liters * green_paint_cost_per_liter
  let brown_paint_cost := brown_paint_liters * brown_paint_cost_per_liter
  let blue_paint_cost := blue_paint_liters * blue_paint_cost_per_liter
  let total_cost := white_paint_cost + green_paint_cost + brown_paint_cost + blue_paint_cost in
  total_cost = 20.23 := by
  sorry

end treehouse_paint_cost_l54_54054


namespace perc_freshmen_in_SLA_l54_54896

variables (T : ℕ) (P : ℝ)

-- 60% of students are freshmen
def freshmen (T : ℕ) : ℝ := 0.60 * T

-- 4.8% of students are freshmen psychology majors in the school of liberal arts
def freshmen_psych_majors (T : ℕ) : ℝ := 0.048 * T

-- 20% of freshmen in the school of liberal arts are psychology majors
def perc_fresh_psych (F_LA : ℝ) : ℝ := 0.20 * F_LA

-- Number of freshmen in the school of liberal arts as a percentage P of the total number of freshmen
def fresh_in_SLA_as_perc (T : ℕ) (P : ℝ) : ℝ := P * (0.60 * T)

theorem perc_freshmen_in_SLA (T : ℕ) (P : ℝ) :
  (0.20 * (P * (0.60 * T)) = 0.048 * T) → P = 0.4 :=
sorry

end perc_freshmen_in_SLA_l54_54896


namespace parallel_lines_l54_54584

theorem parallel_lines (a : ℝ) (h : ∀ x y : ℝ, 2*x - a*y - 1 = 0 → a*x - y = 0) : a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
sorry

end parallel_lines_l54_54584


namespace polynomial_divisibility_l54_54726

theorem polynomial_divisibility 
  {P : ℤ[X]} 
  {a b : ℤ} 
  (h_gt : a > b) 
  : (P.eval a - P.eval b) ∣ (a - b) := 
sorry

end polynomial_divisibility_l54_54726


namespace average_of_rest_equals_40_l54_54648

-- Defining the initial conditions
def total_students : ℕ := 20
def high_scorers : ℕ := 2
def low_scorers : ℕ := 3
def class_average : ℚ := 40

-- The target function to calculate the average of the rest of the students
def average_rest_students (total_students high_scorers low_scorers : ℕ) (class_average : ℚ) : ℚ :=
  let total_marks := total_students * class_average
  let high_scorer_marks := 100 * high_scorers
  let low_scorer_marks := 0 * low_scorers
  let rest_marks := total_marks - (high_scorer_marks + low_scorer_marks)
  let rest_students := total_students - high_scorers - low_scorers
  rest_marks / rest_students

-- The theorem to prove that the average of the rest of the students is 40
theorem average_of_rest_equals_40 : average_rest_students total_students high_scorers low_scorers class_average = 40 := 
by
  sorry

end average_of_rest_equals_40_l54_54648


namespace find_s_l54_54063

noncomputable theory

open Real

def A : Point := ⟨0, 10⟩
def B : Point := ⟨3, 0⟩
def C : Point := ⟨9, 0⟩

def y_s (s : ℝ) (P : Point) (Q : Point) : Prop :=
  P = (⟨(3/10) * (10 - s), s⟩ : Point) ∧ Q = (⟨(9/10) * (10 - s), s⟩ : Point) ∧
  (1/2) * ((6/10) * (10 - s)) * (10 - s) = 18

theorem find_s (s : ℝ) (P Q : Point) (h : y_s s P Q) : s = 10 - 2 * sqrt 15 :=
sorry

end find_s_l54_54063


namespace initial_water_amount_l54_54469

open Real

theorem initial_water_amount (W : ℝ)
  (h1 : ∀ (d : ℝ), d = 0.03 * 20)
  (h2 : ∀ (W : ℝ) (d : ℝ), d = 0.06 * W) :
  W = 10 :=
by
  sorry

end initial_water_amount_l54_54469


namespace min_time_to_complete_tasks_l54_54441

-- Define the conditions as individual time durations for each task in minutes
def bed_making_time : ℕ := 3
def teeth_washing_time : ℕ := 4
def water_boiling_time : ℕ := 10
def breakfast_time : ℕ := 7
def dish_washing_time : ℕ := 1
def backpack_organizing_time : ℕ := 2
def milk_making_time : ℕ := 1

-- Define the total minimum time required to complete all tasks
def min_completion_time : ℕ := 18

-- A theorem stating that given the times for each task, the minimum completion time is 18 minutes
theorem min_time_to_complete_tasks :
  bed_making_time + teeth_washing_time + water_boiling_time + 
  breakfast_time + dish_washing_time + backpack_organizing_time + milk_making_time - 
  (bed_making_time + teeth_washing_time + backpack_organizing_time + milk_making_time) <=
  min_completion_time := by
  sorry

end min_time_to_complete_tasks_l54_54441


namespace cube_projection_area_l54_54360

theorem cube_projection_area (a : ℝ) : 
  ∀ {v1 v2 : ℝ^3}, 
    ∃ (plane : ℝ^3 → ℝ) (proj : ℝ^3 → ℝ^2),
    (plane v1 = 0) ∧ (plane v2 = 0) ∧
    (plane ⟨a, a, a⟩ = a^2 * sqrt 3) ∧
    (proj ⟨a, a, a⟩ = a^2 * sqrt 3) →
    (∃ s1 s2 : ℝ^3, plane s1 = 0 ∧ plane s2 = 0 ∧ proj s1 = proj v1 ∧ proj s2 = proj v2) →
    (area_proj : ℝ^2 → ℝ, area_proj (proj (cube a)) = a^2 * sqrt 3) :=
begin
  sorry
end

end cube_projection_area_l54_54360


namespace parity_of_sum_of_primes_l54_54084

-- Define that q is a set of exactly eight distinct prime numbers.
def q (s : Set ℕ) := s.card = 8 ∧ ∀ x ∈ s, Nat.Prime x

-- x is a member of q and the least value of x is 3.
def x_in_q_and_least_value_is_3 (s : Set ℕ) (x : ℕ) := x ∈ s ∧ x = 3

-- Prove that the sum of the integers in q is even.
theorem parity_of_sum_of_primes (s : Set ℕ) (x : ℕ) (hs : q s) (hx : x_in_q_and_least_value_is_3 s x) :
  Even (s.sum id) := by
  sorry

end parity_of_sum_of_primes_l54_54084


namespace general_term_sequence_l54_54985

theorem general_term_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) 
(hS : ∀ n, S n = n ^ 2 + 3 * n + 1) :
(∀ n, a n = if n = 1 then 5 else 2 * n + 2) :=
begin
  sorry
end

end general_term_sequence_l54_54985


namespace probability_anna_wins_l54_54563

-- Define the problem conditions
variable {Anna Juan Carlos Manu : Type}
variable (flip : Anna → Nat)

-- Define the event of winning by two consecutive heads
def wins_with_two_heads (n : Nat) : Prop := (n = 2) -- Assuming '2' represents two consecutive heads

-- Define the probability calculation
def winning_probability : Nat → Prop :=
  λ n, ∃ m : ℕ, Anna = m ∧ (1 / (2 ^ (8 + 4 * (m - 1)))) = 1 / 240

-- Formal statement of the problem
theorem probability_anna_wins : winning_probability = 1 / 240 :=
by {
  sorry -- proof skipped
}

end probability_anna_wins_l54_54563


namespace part1_part2_l54_54691

-- Definitions based on provided conditions
def triangle (A B C : ℝ) (a b c : ℝ) (D : ℝ) (BD DC AD : ℝ) : Prop :=
  b * Real.cos A + a * Real.cos B = 2 * c * Real.cos A ∧
  BD = 3 * DC ∧
  AD = 3

-- Part (1) Proof: A == π / 3
theorem part1 (A B C a b c : ℝ) (h : b * Real.cos A + a * Real.cos B = 2 * c * Real.cos A) : 
  A = Real.pi / 3 := 
  sorry

-- Part (2) Proof: Maximum area is 4 * sqrt 3
theorem part2 (A B C a b c : ℝ) (D BD DC AD : ℝ)
  (h1 : b * Real.cos A + a * Real.cos B = 2 * c * Real.cos A)
  (h2 : BD = 3 * DC)
  (h3 : AD = 3) :
  area a b c = 4 * Real.sqrt 3 :=
  sorry

end part1_part2_l54_54691


namespace sum_eq_24_l54_54609

-- Define the conditions
variables (A B C D X Y Z : ℕ)
variables (AB CD XYZ : ℕ)
variables (xyz_digits : Finset ℕ) (digits : Finset ℕ)

-- Convert the factors into a lean representation
def is_unique_digits (s : Finset ℕ) : Prop :=
  s.card = 7

def increments_by_one (n : ℕ) : Prop :=
  let d0 := n % 10 in
  let d1 := (n / 10) % 10 in
  let d2 := (n / 100) % 10 in
  d1 = d0 + 1 ∧ d2 = d1 + 1

-- The given conditions
axiom ab_condition : AB = 10 * A + B
axiom cd_condition : CD = 10 * C + D
axiom xyz_condition : XYZ = 100 * X + 10 * Y + Z
axiom eq_condition : AB + CD = XYZ
axiom unique_digits_condition : is_unique_digits (digits A ∪ digits B ∪ digits C ∪ digits D ∪ digits X ∪ digits Y ∪ digits Z)
axiom increments_condition : increments_by_one XYZ

-- The proof problem stating the final sum is 24
theorem sum_eq_24
  (hAB : AB = 10 * A + B)
  (hCD : CD = 10 * C + D)
  (hXYZ : XYZ = 100 * X + 10 * Y + Z)
  (hEq : AB + CD = XYZ)
  (hUnique : is_unique_digits (xyz_digits))
  (hIncrements : increments_by_one XYZ):
  A + B + C + D + X + Y + Z = 24 := by
  sorry

end sum_eq_24_l54_54609


namespace spinner_prob_divisible_by_5_l54_54521

def probability_divisible_by_5 := 1 / 3

theorem spinner_prob_divisible_by_5:
  let spinner : list ℕ := [1, 2, 5] in
  let digits := spinner.product spinner.product spinner in
  let total_outcomes := list.length digits in
  let divisible_by_5 := list.filter (λ (d : ℕ × (ℕ × ℕ)), (d.2.2 % 5 = 0)) digits in
  let favourable_outcomes := list.length divisible_by_5 in
  (favourable_outcomes : ℚ) / (total_outcomes : ℚ) = probability_divisible_by_5 :=
sorry

end spinner_prob_divisible_by_5_l54_54521


namespace minimum_A_B_C_value_l54_54443

theorem minimum_A_B_C_value (A B C : ℕ) (hA7 : (nat.factors A).length + 1 = 7)
(hB6 : (nat.factors B).length + 1 = 6) (hC3 : (nat.factors C).length + 1 = 3) 
(hAB24 : (nat.factors (A * B)).length + 1 = 24) (hBC10 : (nat.factors (B * C)).length + 1 = 10) : 
  A + B + C = 91 :=
sorry

end minimum_A_B_C_value_l54_54443


namespace jack_afternoon_emails_l54_54324

theorem jack_afternoon_emails : 
  ∀ (morning_emails afternoon_emails : ℕ), 
  morning_emails = 6 → 
  afternoon_emails = morning_emails + 2 → 
  afternoon_emails = 8 := 
by
  intros morning_emails afternoon_emails hm ha
  rw [hm] at ha
  exact ha

end jack_afternoon_emails_l54_54324


namespace exponentiation_81_5_4_eq_243_l54_54177

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end exponentiation_81_5_4_eq_243_l54_54177


namespace pipe_fills_cistern_l54_54119

theorem pipe_fills_cistern (t : ℕ) (h : t = 5) : 11 * t = 55 :=
by
  sorry

end pipe_fills_cistern_l54_54119


namespace count_mountain_numbers_l54_54511

-- Define the mountain number conditions
def isMountainNumber (a b c d : ℕ) : Prop :=
  b > a ∧ b > c ∧ b > d ∧ a ≠ 0 ∧ d ≠ 0

-- Define the count of mountain numbers
theorem count_mountain_numbers : 
  let count := 
  (∑ b in finset.range 10, -- Loop for b from 1 to 9 (as 10 is exclusive)
  if b = 0 then 0 else
  (∑ a in finset.range 10, -- Loop for a from 1 to 9 (as 10 is exclusive)
  if a = 0 ∨ a = b then 0 else
  (∑ d in finset.range 10, -- Loop for d from 1 to 9 (as 10 is exclusive)
  if d = 0 ∨ d = b ∨ d = a then 0 else
  (∑ c in finset.range 10, -- Loop for c from 0 to 9
  if c = b ∨ c = a ∨ c = d ∨ b ≤ c then 0 else 1))))
  in count = 5184 :=
by
-- The proof would be filled in here
sorry

end count_mountain_numbers_l54_54511


namespace sector_central_angle_l54_54871

theorem sector_central_angle (r θ : ℝ) 
  (h1 : 1 = (1 / 2) * 2 * r) 
  (h2 : 2 = θ * r) : θ = 2 := 
sorry

end sector_central_angle_l54_54871


namespace polygon_area_l54_54920

theorem polygon_area (n : ℕ) (s : ℝ) (perimeter : ℝ) (area : ℝ) 
  (h1 : n = 24) 
  (h2 : n * s = perimeter) 
  (h3 : perimeter = 48) 
  (h4 : s = 2) 
  (h5 : area = n * s^2 / 2) : 
  area = 96 :=
by
  sorry

end polygon_area_l54_54920


namespace cubic_sum_l54_54375

theorem cubic_sum (a b c : ℤ) (h1 : a + b + c = 7) (h2 : a * b + a * c + b * c = 11) (h3 : a * b * c = -6) :
  a^3 + b^3 + c^3 = 223 :=
by
  sorry

end cubic_sum_l54_54375


namespace ferris_wheel_seats_l54_54025

theorem ferris_wheel_seats : ∀ (total_people_per_wheel seats_per_seat : ℕ), 
  total_people_per_wheel = 84 → seats_per_seat = 6 → 
  total_people_per_wheel / seats_per_seat = 14 :=
by
  intros total_people_per_wheel seats_per_seat h1 h2
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul Nat.zero_le (by norm_num) (by norm_num)

end ferris_wheel_seats_l54_54025


namespace partA_maximize_daily_production_partB_maximize_total_production_partC_maximize_revenue_partD_minimum_exhibition_revenue_l54_54835

noncomputable def dailyProduction (L: ℝ) : ℝ := sqrt L
noncomputable def lifespan (L: ℝ) : ℝ := 8 - sqrt L
noncomputable def totalProduction (L: ℝ) : ℝ := (sqrt L) * lifespan L
noncomputable def revenue (L R X: ℝ) : ℝ := (90 * totalProduction L) + (X * (24 - L) * lifespan L)

theorem partA_maximize_daily_production :
  L = 24 → ∀ L', dailyProduction L' ≤ dailyProduction L := sorry

theorem partB_maximize_total_production :
  L = 16 → ∀ L', totalProduction L' ≤ totalProduction L := sorry

theorem partC_maximize_revenue : 
  L = 9 ∧ R = 1650 →
  ∀ L' R', revenue L' 0 4 ≤ revenue L 0 4 ∧ revenue L 0 4 = R := sorry

theorem partD_minimum_exhibition_revenue :
  X = 30 →
  ∀ L x, x >= X -> (revenue 0 L x 4 > revenue L' L x 4) := sorry

end partA_maximize_daily_production_partB_maximize_total_production_partC_maximize_revenue_partD_minimum_exhibition_revenue_l54_54835


namespace monotonic_increasing_interval_find_bc_in_triangle_l54_54618

def f (x : ℝ) : ℝ :=
  (sqrt 3 * (Real.sin (3 * Real.pi + x)) * (Real.cos (Real.pi - x)) +
  (Real.cos (Real.pi / 2 + x))^2)

theorem monotonic_increasing_interval (k : ℤ) :
  ∃ x : ℝ, (k * π - π / 6) ≤ x ∧ x ≤ (k * π + π / 3)  :=
sorry

theorem find_bc_in_triangle (A a b c : ℝ)
  (hA : 0 < A ∧ A < π)
  (h_a : a = 2)
  (h_bc_eq : b + c = 4)
  (h_fA : f(A) = 3 / 2) :
  b = 2 ∧ c = 2 :=
begin
  sorry
end

end monotonic_increasing_interval_find_bc_in_triangle_l54_54618


namespace smallest_value_of_A_l54_54725

-- Define the set A and the properties
constant A : Set ℤ
constant n : ℤ -> ℤ
constant m : ℤ

-- Define the condition for the set A
axiom exists_a_n (m : ℤ) : ∃ (a : ℤ) (n : ℕ), a ∈ A ∧ a^n ≡ m [MOD 100]

-- Theorem stating the smallest possible value of |A|
theorem smallest_value_of_A : |A| = 41 := 
sorry

end smallest_value_of_A_l54_54725


namespace part1_l54_54139

variables {Point Line Plane Poly}

def BDEF_perpendicular_ABCD (poly: Poly) : Prop := sorry
def AB_eq_1 (A B : Point) : Prop := dist A B = 1
def midpoint_AE (M A E : Point) : Prop := sorry
def BM_parallel_EFC (B M E F C : Point) : Prop := sorry

theorem part1 {poly: Poly} {A B C D E F M : Point} :
    BDEF_perpendicular_ABCD poly →
    AB_eq_1 A B →
    midpoint_AE M A E →
    BM_parallel_EFC B M E F C := sorry

end part1_l54_54139


namespace sum_f_a_seq_positive_l54_54736

noncomputable def f (x : ℝ) : ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_monotone_decreasing_nonneg : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → f y ≤ f x
axiom a_seq : ∀ n : ℕ, ℝ
axiom a_arithmetic : ∀ m n k : ℕ, m + k = 2 * n → a_seq m + a_seq k = 2 * a_seq n
axiom a3_neg : a_seq 3 < 0

theorem sum_f_a_seq_positive :
    f (a_seq 1) + 
    f (a_seq 2) + 
    f (a_seq 3) + 
    f (a_seq 4) + 
    f (a_seq 5) > 0 :=
sorry

end sum_f_a_seq_positive_l54_54736


namespace right_angled_triangle_l54_54026

noncomputable def area {a b c : ℝ} (S : ℝ) (h_a h_b h_c : ℝ) := 
  ∃ S a b c : ℝ, 
    a = 2 * S / h_a ∧ 
    b = 2 * S / h_b ∧ 
    c = 2 * S / h_c

theorem right_angled_triangle (h_a h_b h_c : ℝ) (S : ℝ) (a b c : ℝ) :
  h_a = 12 ∧ h_b = 15 ∧ h_c = 20 →
  area S h_a h_b h_c →
  a^2 + b^2 = c^2 :=
by
  sorry

end right_angled_triangle_l54_54026


namespace area_of_EFGH_l54_54825

-- Define vertices as points in the 2D plane
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the specific points E, F, G, and H
def E : Point := ⟨2, -3⟩
def F : Point := ⟨2, 2⟩
def G : Point := ⟨7, 9⟩
def H : Point := ⟨7, 2⟩

-- Calculate the distance between two points
def distance (p1 p2 : Point) : ℤ :=
  abs (p2.y - p1.y)

-- Calculate the height of the parallelogram (difference in x-coordinates)
def height (p1 p2 : Point) : ℤ :=
  abs (p2.x - p1.x)

-- Define the area of the parallelogram
def area_parallelogram (E F G H : Point) : ℤ :=
  let base := distance E F
  let h := height E G
  base * h

-- The theorem to prove the area is 25
theorem area_of_EFGH :
  area_parallelogram E F G H = 25 :=
by
  -- Step to state that we skip the actual proof
  sorry

end area_of_EFGH_l54_54825


namespace find_a_range_empty_solution_set_l54_54255

theorem find_a_range_empty_solution_set :
  ∀ a : ℝ, (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0 → false) ↔ (-2 ≤ a ∧ a < 6 / 5) :=
by sorry

end find_a_range_empty_solution_set_l54_54255


namespace point_on_x_axis_right_of_origin_is_3_units_away_l54_54294

theorem point_on_x_axis_right_of_origin_is_3_units_away :
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧ P.1 > 0 ∧ dist (P.1, P.2) (0, 0) = 3 ∧ P = (3, 0) := 
by
  sorry

end point_on_x_axis_right_of_origin_is_3_units_away_l54_54294


namespace correct_expression_l54_54136

def expr1 := 3 * 6 * 9 / 3
def expr2 := 2 * 6 * 9 / 2

theorem correct_expression :
  expr1 = 54 → expr1 = expr2 :=
by
  intro h
  rw h
  norm_num
  sorry

end correct_expression_l54_54136


namespace least_four_digit_palindrome_divisible_by_5_is_5005_l54_54356

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def is_four_digit_palindrome_divisible_by_5 (n : ℕ) : Prop :=
  is_palindrome n ∧ is_four_digit n ∧ is_divisible_by_5 n

theorem least_four_digit_palindrome_divisible_by_5_is_5005 :
  ∃ n, is_four_digit_palindrome_divisible_by_5 n ∧ ∀ m, is_four_digit_palindrome_divisible_by_5 m → n ≤ m :=
  sorry

end least_four_digit_palindrome_divisible_by_5_is_5005_l54_54356


namespace triangle_area_x_value_l54_54945

theorem triangle_area_x_value (x : ℝ) (h1 : x > 0) (h2 : 1 / 2 * x * (2 * x) = 64) : x = 8 :=
by
  sorry

end triangle_area_x_value_l54_54945


namespace rectangular_area_l54_54027

theorem rectangular_area (length width : ℝ) (h₁ : length = 0.4) (h₂ : width = 0.22) :
  (length * width = 0.088) :=
by sorry

end rectangular_area_l54_54027


namespace num_sets_satisfying_union_l54_54970

open Set

namespace ProofProblem

def A : Set ℕ := {0, 1}
def B : Set ℕ := {0, 1, 2}

theorem num_sets_satisfying_union : 
  { C : Set ℕ | A ∪ C = B }.to_finset.card = 4 :=
by 
  sorry

end ProofProblem

end num_sets_satisfying_union_l54_54970


namespace emails_received_in_afternoon_l54_54323

theorem emails_received_in_afternoon (A : ℕ) 
  (h1 : 4 + (A - 3) = 9) : 
  A = 8 :=
by
  sorry

end emails_received_in_afternoon_l54_54323


namespace solve_equation_l54_54365

noncomputable def equation := 
  (λ x : ℂ, (x^3 + 3 * x^2 * Complex.sqrt 3 + 8 * x + 2 * Complex.sqrt 3))

theorem solve_equation :
  ∀ x : ℂ, equation x = 0 ↔ (x = Complex.sqrt 3 ∨ x = Complex.sqrt 3 + Complex.I * Complex.sqrt 2 ∨ x = Complex.sqrt 3 - Complex.I * Complex.sqrt 2) :=
by
  sorry

end solve_equation_l54_54365


namespace company_total_after_hiring_l54_54087

variable (E : ℝ)

def initial_female_percentage := 0.60
def additional_male_workers := 30
def final_female_percentage := 0.55

theorem company_total_after_hiring :
  initial_female_percentage * E = final_female_percentage * (E + additional_male_workers) → E + additional_male_workers = 360 :=
by
  intro h
  have h_initial : initial_female_percentage * E = 0.60 * E := rfl
  have h_final : final_female_percentage * (E + additional_male_workers) = 0.55 * (E + additional_male_workers) := rfl
  sorry

end company_total_after_hiring_l54_54087


namespace selling_price_of_one_bag_l54_54421

theorem selling_price_of_one_bag :
  ∀ (total_harvested total_juice total_restaurant total_revenue per_bag weight_bags : ℕ),
    total_harvested = 405 →
    total_juice = 90 →
    total_restaurant = 60 →
    total_revenue = 408 →
    per_bag = 5 →
    weight_bags = (total_harvested - (total_juice + total_restaurant)) →
    (weight_bags / per_bag) = 51 →
    (total_revenue / (weight_bags / per_bag)) = 8 :=
by
  intros total_harvested total_juice total_restaurant total_revenue per_bag weight_bags
  assume h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5] at h6
  rw [h6] at h7
  exact h7.symm

end selling_price_of_one_bag_l54_54421


namespace ratio_A_B_l54_54510

noncomputable def A : ℝ := 
  let seq := λ n : ℕ, if n % 4 = 0 then 0 else 1 / (n^2) * (if n % 2 = 0 then -1 else 1)
  ∑' n, seq (2 * n + 1)

noncomputable def B : ℝ :=
  let seq := λ n : ℕ, if n % 4 = 0 then (1 / (n^2)) * (if (n / 4) % 2 = 0 then 1 else -1) else 0
  ∑' n, seq (4 * n + 4)

theorem ratio_A_B : A / B = 17 := 
by 
  sorry

end ratio_A_B_l54_54510


namespace xyz_inequality_l54_54955

noncomputable def x : ℝ := Real.logBase 2 3 - Real.logBase 2 (Real.sqrt 3)
noncomputable def y : ℝ := Real.logBase (1/2) Real.pi
noncomputable def z : ℝ := 0.9 ^ (-1.1)

theorem xyz_inequality : y < x ∧ x < z :=
  by sorry

end xyz_inequality_l54_54955


namespace part1_part2_l54_54223

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x + a - 1) + abs (x - 2 * a)

-- Part (1) of the proof problem
theorem part1 (a : ℝ) : f 1 a < 3 → - (2 : ℝ)/3 < a ∧ a < 4 / 3 := sorry

-- Part (2) of the proof problem
theorem part2 (a x : ℝ) : a ≥ 1 → f x a ≥ 2 := sorry

end part1_part2_l54_54223


namespace work_together_days_l54_54832

-- Define a noncomputable def for calculation purposes
noncomputable def combinedWorkDays (Wp Wq : ℚ) : ℚ :=
  1 / (Wp + Wq)

theorem work_together_days (Wq : ℚ) :
  let Wp := 1.20 * Wq in
  Wp = 1 / 22 →
  Wq = 1 / 26.4 →
  combinedWorkDays Wp Wq = 12 := by
  intros Wp hWp hWq
  -- Sorry will be used to skip the implementation of the proof
  sorry

end work_together_days_l54_54832


namespace determine_K_class_comparison_l54_54810

variables (a b : ℕ) -- number of students in classes A and B respectively
variable (K : ℕ) -- amount that each A student would pay if they covered all cost

-- Conditions from the problem statement
def first_event_total (a b : ℕ) := 5 * a + 3 * b
def second_event_total (a b : ℕ) := 4 * a + 6 * b
def total_balance (a b K : ℕ) := 9 * (a + b) = K * (a + b)

-- Questions to be answered
theorem determine_K : total_balance a b K → K = 9 :=
by
  sorry

theorem class_comparison (a b : ℕ) : 5 * a + 3 * b = 4 * a + 6 * b → b > a :=
by
  sorry

end determine_K_class_comparison_l54_54810


namespace prob_even_l54_54596

-- Given conditions
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)
def even_function (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = g (x)

-- Property to be proven
theorem prob_even (f g : ℝ → ℝ) (hf : odd_function f) (hg : even_function g) : even_function (λ x, f (|x|) * g x) :=
sorry

end prob_even_l54_54596


namespace apple_counts_l54_54355

theorem apple_counts (x y : ℤ) (h1 : y - x = 2) (h2 : y = 3 * x - 4) : x = 3 ∧ y = 5 := 
by
  sorry

end apple_counts_l54_54355


namespace perpendicular_lines_l54_54231

theorem perpendicular_lines (a : ℝ) :
  let l1 := ax + 2 * y + 6 = 0 
  let l2 := x + (a - 1) * y + a^2 - 1 = 0 
  l1 ⊥ l2 ↔ a = 2 / 3 := 
by 
  sorry

end perpendicular_lines_l54_54231


namespace max_distance_point_to_line_l54_54232

theorem max_distance_point_to_line :
  let P := (-5 : ℝ, 0 : ℝ),
      l (m : ℝ) := (1 + 2 * m) * x - (m + 1) * y - 4 * m - 3 = 0
  ∃ (m : ℝ), 
    dist P (some_line_through_l_formula_holds_for_m l m) = 2 * Real.sqrt 10 := 
sorry

end max_distance_point_to_line_l54_54232


namespace parameterized_curve_is_line_l54_54466

theorem parameterized_curve_is_line :
  ∀ (t : ℝ), ∃ (m b : ℝ), y = 5 * ((x - 5) / 3) - 3 → y = (5 * x - 34) / 3 := 
by
  sorry

end parameterized_curve_is_line_l54_54466


namespace sequence_of_sevens_appears_first_l54_54745

theorem sequence_of_sevens_appears_first :
  ∀ (S : ℕ → String), (∀ n, S n = toString n) →
  (∃ i, String.drop i (String.concat (List.map S (List.range (i + 2007)))) = "7777777777777777777777777777777777777777777777777777777777777777" ∧ 
   String.get i (String.concat (List.map S (List.range (i + 2007)))) ≠ '7') → 
  (∃ j, String.drop j (String.concat (List.map S (List.range (j + 2006)))) = "6666666666666666666666666666666666666666666666666666666666666666" ∧ 
   String.get j (String.concat (List.map S (List.range (j + 2006)))) ≠ '6') →
  (∀ i j, (String.drop i (String.concat (List.map S (List.range (i + 2007)))) = "7777777777777777777777777777777777777777777777777777777777777777" ∧ 
   String.get i (String.concat (List.map S (List.range (i + 2007)))) ≠ '7') → 
  (String.drop j (String.concat (List.map S (List.range (j + 2006)))) = "6666666666666666666666666666666666666666666666666666666666666666" ∧ 
   String.get j (String.concat (List.map S (List.range (j + 2006)))) ≠ '6') → 
  i < j).
sorry

end sequence_of_sevens_appears_first_l54_54745


namespace common_point_log_exp_l54_54952

theorem common_point_log_exp (a : ℝ) (h₀ : a > 1) 
  (h₁ : ∃ x, a ^ x = log a x) : (log (log a)).nat_abs = nat_abs (-1) :=
sorry

end common_point_log_exp_l54_54952


namespace sample_size_correctness_l54_54284

-- Given conditions as definitions
def total_students : ℕ := 700 + 500 + 300
def freshmen_students : ℕ := 700
def sophomore_students : ℕ := 500
def senior_students : ℕ := 300
def freshmen_sampled : ℕ := 14

-- The problem statement to be translated into Lean
theorem sample_size_correctness (n : ℕ) 
  (h : freshmen_sampled = 14) 
  (hn : freshman_students = 700) 
  (hs : total_students = 1500) :
  ∃ n, (14:nat) / (n:nat) =  700 / (700 + 500 + 300) ∧ n = 30 :=
by
  sorry

end sample_size_correctness_l54_54284


namespace range_of_a_l54_54977

variable (a : ℝ)

def quadratic_function : ℝ → ℝ :=
  λ x, x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (h_decrease : ∀ x, x ≤ 4 → quadratic_function a x ≤ quadratic_function a (4:ℝ))
                   (h_increase : ∀ x, x ≥ 5 → quadratic_function a x ≥ quadratic_function a (5:ℝ)) :
  -4 ≤ a ∧ a ≤ -3 :=
begin
  sorry
end

end range_of_a_l54_54977


namespace compoundInterestRateProof_l54_54631

noncomputable def annualInterestRate : ℝ :=
   2 * (Real.sqrt 1.0812 - 1)

theorem compoundInterestRateProof :
  let principal : ℝ := 5000
  let simpleInterest : ℝ := 400
  let additionalAmount : ℝ := 6
  -- Total amount from simple interest
  let simpleTotal : ℝ := simpleInterest + principal
  -- Total compound interest to match the problem statement
  let compoundInterest : ℝ := principal * ((1 + annualInterestRate / 2)^2 - 1)
  -- Let the semi-annual compounded rate 
  let totalWithCompound : ℝ := simpleTotal + additionalAmount
  	-- Verifying the relationship between compound interest and the additional amount
  compoundInterest = simpleInterest + additionalAmount :=
by
  let principal : ℝ := 5000
  let simpleInterest : ℝ := 400
  let additionalAmount : ℝ := 6
  let simpleTotal : ℝ := simpleInterest + principal
  let compoundInterest : ℝ := principal * ((1 + (annualInterestRate / 2))^2 - 1)
  let totalWithCompound : ℝ := simpleTotal + additionalAmount
  -- Verifying the relationship between compound interest and the additional amount
  sorry


end compoundInterestRateProof_l54_54631


namespace sum_q_p_values_l54_54993

def p (x : ℝ) : ℝ := abs x - 3
def q (x : ℝ) : ℝ := - abs x

theorem sum_q_p_values : (List.sum (List.map (λ x, q (p x)) [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])) = -15 := 
by sorry

end sum_q_p_values_l54_54993


namespace total_handshakes_l54_54141

theorem total_handshakes (players_team1 players_team2 referees : ℕ) 
  (h1 : players_team1 = 11) (h2 : players_team2 = 11) (h3 : referees = 3) : 
  players_team1 * players_team2 + (players_team1 + players_team2) * referees = 187 := 
by
  sorry

end total_handshakes_l54_54141


namespace hyperbola_eccentricity_l54_54619

theorem hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (chord_len : ∃ (x y : ℝ), (x - 2)^2 + y^2 = 4 ∧ (∃ k : ℝ, y = k * x) ∧ 2 = 2 * sqrt (4 - (4 * b^2) / (a^2 + b^2))) :
  let c := sqrt (a^2 + b^2) in
  (e : ℝ) = (c / a) :=
sorry -- Proof goes here

end hyperbola_eccentricity_l54_54619


namespace find_k_of_quadratic_roots_l54_54549

theorem find_k_of_quadratic_roots :
  (∃ k : ℝ, ∀ x : ℂ, (5 * x^2 - 2 * x + k = 0 ↔ x = (1 + complex.i * real.sqrt 39) / 10 ∨ x = (1 - complex.i * real.sqrt 39) / 10))
  ↔ k = 2.15 :=
sorry

end find_k_of_quadratic_roots_l54_54549


namespace correct_transformation_l54_54806

def transform_sine_function : Prop :=
  ∃ f : ℝ → ℝ, 
    (f = λ x, sin x) ∧ 
    (f = λ x, sin (2 * ((x + π / 3) / 2) - π / 3))
    
theorem correct_transformation : transform_sine_function :=
sorry

end correct_transformation_l54_54806


namespace common_temperature_after_opening_l54_54811

def Room : Type :=
  { length : ℝ
  , width : ℝ
  , height : ℝ
  , temperature : ℝ }

def volume (room : Room) : ℝ :=
  room.length * room.width * room.height

def common_temperature (room1 room2 : Room) : ℝ :=
  ((volume room1 * room1.temperature) + (volume room2 * room2.temperature)) / (volume room1 + volume room2)

theorem common_temperature_after_opening :
  ∀ (room1 room2 : Room), 
    room1.length = 5 → room1.width = 3 → room1.height = 4 → room1.temperature = 22 → 
    room2.length = 6 → room2.width = 5 → room2.height = 4 → room2.temperature = 13 →
    common_temperature room1 room2 = 16 :=
by
  intros room1 room2
  assume h1_length h1_width h1_height h1_temp h2_length h2_width h2_height h2_temp
  sorry

end common_temperature_after_opening_l54_54811


namespace probability_increase_l54_54892

theorem probability_increase:
  let P_win1 := 0.30
  let P_lose1 := 0.70
  let P_win2 := 0.50
  let P_lose2 := 0.50
  let P_win3 := 0.40
  let P_lose3 := 0.60
  let P_win4 := 0.25
  let P_lose4 := 0.75
  let P_win_all := P_win1 * P_win2 * P_win3 * P_win4
  let P_lose_all := P_lose1 * P_lose2 * P_lose3 * P_lose4
  (P_lose_all - P_win_all) / P_win_all = 9.5 :=
by
  sorry

end probability_increase_l54_54892


namespace find_k_l54_54579

noncomputable def sequence {n : ℕ} (a : ℕ → ℝ) : ℝ := (2 / 3) * (a n) - (1 / 3)

theorem find_k (a : ℕ → ℝ) (S : ℕ → ℝ) (h : ∀ n : ℕ, S n = (2 / 3) * (a n) - (1 / 3)) :
  ∃ k : ℕ, k > 0 ∧ -1 < S k ∧ S k < 2 :=
begin
  sorry
end

end find_k_l54_54579


namespace cricket_run_rate_l54_54465

theorem cricket_run_rate (initial_run_rate : ℝ) (initial_overs : ℕ) (target : ℕ) (remaining_overs : ℕ) 
    (run_rate_in_remaining_overs : ℝ)
    (h1 : initial_run_rate = 3.2)
    (h2 : initial_overs = 10)
    (h3 : target = 272)
    (h4 : remaining_overs = 40) :
    run_rate_in_remaining_overs = 6 :=
  sorry

end cricket_run_rate_l54_54465


namespace sculpture_and_base_height_l54_54446

def height_in_inches (feet: ℕ) (inches: ℕ) : ℕ :=
  feet * 12 + inches

theorem sculpture_and_base_height
  (sculpture_feet: ℕ) (sculpture_inches: ℕ) (base_inches: ℕ)
  (hf: sculpture_feet = 2)
  (hi: sculpture_inches = 10)
  (hb: base_inches = 8)
  : height_in_inches sculpture_feet sculpture_inches + base_inches = 42 :=
by
  -- Placeholder for the proof
  sorry

end sculpture_and_base_height_l54_54446


namespace trigonometric_order_l54_54709

-- Definitions
def a : ℝ := Real.sin (5 * Real.pi / 7)
def b : ℝ := Real.cos (2 * Real.pi / 7)
def c : ℝ := Real.tan (2 * Real.pi / 7)

-- Theorem Statement
theorem trigonometric_order : b < a ∧ a < c :=
by
  sorry

end trigonometric_order_l54_54709


namespace no_roots_in_interval_l54_54388

noncomputable def f (x : ℝ) : ℝ := x^4 - 4 * x^3 + 10 * x^2

theorem no_roots_in_interval : ∀ x ∈ set.Icc (1 : ℝ) 2, f x ≠ 0 :=
by
  assume x hx
  have fx_pos : f x > 0 := sorry
  exact ne_of_gt fx_pos

end no_roots_in_interval_l54_54388


namespace age_ratio_in_future_l54_54208

variables (t j x : ℕ)

theorem age_ratio_in_future:
  (t - 4 = 5 * (j - 4)) → 
  (t - 10 = 6 * (j - 10)) →
  (t + x = 3 * (j + x)) →
  x = 26 := 
by {
  sorry
}

end age_ratio_in_future_l54_54208


namespace stone_counting_l54_54377

theorem stone_counting (n : ℕ) (m : ℕ) : 
    10 > 0 ∧  (n ≡ 6 [MOD 20]) ∧ m = 126 → n = 6 := 
by
  sorry

end stone_counting_l54_54377


namespace max_value_expr_l54_54396

theorem max_value_expr (a b c d : ℝ) (ha : -12.5 ≤ a ∧ a ≤ 12.5) (hb : -12.5 ≤ b ∧ b ≤ 12.5) (hc : -12.5 ≤ c ∧ c ≤ 12.5) (hd : -12.5 ≤ d ∧ d ≤ 12.5) :
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 650 :=
sorry

end max_value_expr_l54_54396


namespace domain_of_f_eq_A_range_of_m_for_B_subset_A_range_of_m_for_B_empty_l54_54719

def f (x : ℝ) : ℝ := Real.sqrt (2 + x) + Real.log (4 - x)

def A : Set ℝ := { x | -2 ≤ x ∧ x < 4 }

def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

theorem domain_of_f_eq_A : 
  { x | ∃ (y : ℝ), f x = y } = A :=
sorry

theorem range_of_m_for_B_subset_A :
  {m : ℝ | B m ⊆ A ∧ B m ≠ ∅} = set.Iio (5/2) :=
sorry

-- Another theorem for the case where B is empty when m < 2
theorem range_of_m_for_B_empty :
  {m : ℝ | B m = ∅} = set.Iio 2 :=
sorry

end domain_of_f_eq_A_range_of_m_for_B_subset_A_range_of_m_for_B_empty_l54_54719


namespace find_a_value_l54_54522

-- Define the parametric form for the line
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (-3/5 * t + 2, 4/5 * t)

-- Define the polar equation of the circle
def polar_circle (θ : ℝ) (a : ℝ) : ℝ :=
  a * Real.sin θ

-- Given conditions
axiom nonzero_a (a : ℝ) : a ≠ 0
axiom chord_length_condition (a : ℝ) : (|3 * a / 2 - 8| / 5) = (sqrt 3 / 2 * |a|) / 2

-- The final theorem to prove the values of a
theorem find_a_value (a : ℝ) (h : 2 * |3 * a - 16| = 5 * |a|) : a = 32 ∨ a = 32 / 11 :=
sorry

end find_a_value_l54_54522


namespace eval_power_l54_54182

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end eval_power_l54_54182


namespace fifth_equation_l54_54843

theorem fifth_equation :
  (5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13) = 81 :=
begin
  sorry
end

end fifth_equation_l54_54843


namespace particle_visits_every_point_l54_54755

theorem particle_visits_every_point (P : ℕ → Prop)
  (h0 : P 1)
  (h_step : ∀ n > 0, P n → P (n + 1)) :
  ∀ n, P n :=
begin
  sorry
end

end particle_visits_every_point_l54_54755


namespace option_B_correct_l54_54205

variables {α : Type*} [LinearOrderedField α] 
variables (λ : α) (a : Vector α 3)

theorem option_B_correct (λ : α) (a : Vector α 3) : λ • a = 0 → λ = 0 ∨ a = 0 :=
sorry

end option_B_correct_l54_54205


namespace intersect_three_points_l54_54330

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * x

theorem intersect_three_points (a : ℝ) :
  (∃ (t1 t2 t3 : ℝ), t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    f t1 = g t1 a ∧ f t2 = g t2 a ∧ f t3 = g t3 a) ↔ 
  a ∈ Set.Ioo (2 / (7 * Real.pi)) (2 / (3 * Real.pi)) ∨ a = -2 / (5 * Real.pi) :=
sorry

end intersect_three_points_l54_54330


namespace beta_meets_midpoint_at_3_3_hours_l54_54066

-- Define the essentials.
def distance_between_A_and_B := 66
def speed_alpha := 12
def speed_beta := 10
def speed_gamma := 8
def start_time := 0

-- Define what we want to prove.
theorem beta_meets_midpoint_at_3_3_hours :
  let avg_speed := (speed_alpha + speed_gamma) / 2 in
  let effective_speed := speed_beta + avg_speed in
  distance_between_A_and_B = effective_speed * 3.3 := 
by
  -- Introduce assumptions based on the conditions.
  let avg_speed := (speed_alpha + speed_gamma) / 2
  let effective_speed := speed_beta + avg_speed
  have h1: avg_speed = 10 := by sorry
  have h2: effective_speed = 20 := by sorry

  -- Calculate the correct time.
  show distance_between_A_and_B = effective_speed * 3.3 from
  calc
    distance_between_A_and_B
      = 66                  : by sorry
      ... = 20 * 3.3 : by sorry

end beta_meets_midpoint_at_3_3_hours_l54_54066


namespace tan_ratio_sum_is_one_l54_54249

theorem tan_ratio_sum_is_one (x y : ℝ) 
  (h1 : (sin x / cos y + sin y / cos x = 2)) 
  (h2 : (cos x / sin y + cos y / sin x = 3)) :
  (tan x / tan y + tan y / tan x = 1) :=
sorry

end tan_ratio_sum_is_one_l54_54249


namespace cos_double_alpha_cos_double_beta_l54_54971

theorem cos_double_alpha (α β : ℝ) (h1 : cos α = 3/5) (h2 : cos β = -1/2) (h3 : π < α + β ∧ α + β < 2 * π) (h4 : 0 < α - β ∧ α - β < π) :
  cos (2 * α) = -7/25 :=
sorry

theorem cos_double_beta (α β : ℝ) (h1 : cos α = 3/5) (h2 : cos β = -1/2) (h3 : π < α + β ∧ α + β < 2 * π) (h4 : 0 < α - β ∧ α - β < π) :
  cos (2 * β) = -1/2 :=
sorry

end cos_double_alpha_cos_double_beta_l54_54971


namespace ship_distance_graph_l54_54872

-- Define the conditions
variables {X A B C : Type}
variables (R r : ℝ) (hR : R > 0) (hr : 0 < r) (hRr : r < R)
-- The paths AB and BC are semicircles centered at X with radii R and r respectively
variables (d_AB : ∀ (p : Type), p = X → (∀ (q : Type), q ≠ X → distance p q = R))
variables (d_BC : ∀ (p : Type), p = X → (∀ (q : Type), q ≠ X → distance p q = r))

-- Prove the graph representing the ship's distance from X is a step graph starting at R and stepping down to r
theorem ship_distance_graph : 
  ∀ (p : Type), p = A ∨ p = B ∨ p = C → 
  (∀ q, q = A ∨ q = B → distance q X = R) ∧ 
  (∀ q, q = B ∨ q = C → distance q X = r) :=
begin
  sorry
end

end ship_distance_graph_l54_54872


namespace count_convex_functions_on_D_l54_54986

def second_derivative (f : ℝ → ℝ) := fun x : ℝ => (deriv (deriv f)) x

def is_convex_on (f : ℝ → ℝ) (D : set ℝ) := ∀ x ∈ D, second_derivative f x < 0

def f1 (x : ℝ) := -x^3 + 2*x - 1
def f2 (x : ℝ) := real.log x - 2*x
def f3 (x : ℝ) := real.sin x + real.cos x
def f4 (x : ℝ) := x * real.exp x

def D : set ℝ := set.Ioo 0 (3 * real.pi / 4)

theorem count_convex_functions_on_D : 
  (∃ n, n = 3 ∧
  (n = (if is_convex_on f1 D then 1 else 0) +
        (if is_convex_on f2 D then 1 else 0) +
        (if is_convex_on f3 D then 1 else 0) +
        (if is_convex_on f4 D then 1 else 0))) :=
by
  sorry

end count_convex_functions_on_D_l54_54986


namespace parallel_is_sufficient_for_outside_outside_is_not_necessary_for_parallel_parallel_is_sufficient_but_not_necessary_l54_54839

variables {l : Type} {α : Type}

def is_parallel (l : set α) (α : set α) : Prop := 
  ∀ p1 p2 ∈ l, ∀ q1 q2 ∈ α, p1 ≠ q1 → p2 ≠ q2 → l ∩ α = ∅

def is_outside (l : set α) (α : set α) : Prop := 
  l ∩ α = ∅

theorem parallel_is_sufficient_for_outside (l : set α) (α : set α) :
  is_parallel l α → is_outside l α :=
by
  intro h
  unfold is_outside
  rw is_parallel at h
  apply h; sorry

theorem outside_is_not_necessary_for_parallel (l : set α) (α : set α) :
  is_outside l α → ¬ is_parallel l α :=
by 
  intro h
  unfold is_outside at h
  apply h; sorry

theorem parallel_is_sufficient_but_not_necessary (l : set α) (α : set α) : 
  is_parallel l α ↔ is_outside l α ∧ ¬is_parallel l α := 
by 
  apply and.intro; 
  apply parallel_is_sufficient_for_outside; 
  apply outside_is_not_necessary_for_parallel; sorry

end parallel_is_sufficient_for_outside_outside_is_not_necessary_for_parallel_parallel_is_sufficient_but_not_necessary_l54_54839


namespace brownies_pieces_count_l54_54351

theorem brownies_pieces_count:
  let pan_width := 24
  let pan_length := 15
  let piece_width := 3
  let piece_length := 2
  pan_width * pan_length / (piece_width * piece_length) = 60 := 
by
  sorry

end brownies_pieces_count_l54_54351


namespace problem_statement_l54_54614

-- Definitions based on the conditions
noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 + sqrt 3 * sin (π - x) * cos (π + x) - 1 / 2

def triangle_area (a b C : ℝ) : ℝ := 1 / 2 * a * b * sin C

-- The main statement
theorem problem_statement :
  (∀ x ∈ Icc (0 : ℝ) π, f x ≤ f (x + π / 3)) ∧
  ∃ (A B C a b c : ℝ), 
      (A < π / 2) ∧ 
      (A + B + C = π) ∧ 
      (a = 2) ∧ 
      (b * sin C = a * sin A) ∧ 
      (f A = -1) →
  ∃ (area : ℝ), area = sqrt 3 :=
begin
  sorry
end

end problem_statement_l54_54614


namespace cos_Z_in_right_triangle_l54_54280

theorem cos_Z_in_right_triangle :
  ∀ (X Y Z : Type) [triangle X Y Z], 
    angle X Y Z = 90 ∧ angle Y Z X = 45 ∧ tan (angle Z X Y) = 1 / 2 →
    cos (angle Z X Y) = sqrt(5) / 5 :=
by
  sorry

end cos_Z_in_right_triangle_l54_54280


namespace intersection_AB_l54_54621

def setA : Set ℝ := { x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := { x | x > 1 }
def intersection : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_AB : setA ∩ setB = intersection :=
by
  sorry

end intersection_AB_l54_54621


namespace largest_divisor_of_prime_power_l54_54838

theorem largest_divisor_of_prime_power (p : ℕ) (hp : Nat.Prime p) :
  ∃ d : ℕ, d = 24 ∧ p^d ∣ p^(4!) := by
  sorry

end largest_divisor_of_prime_power_l54_54838


namespace find_k_inv_h_of_10_l54_54374

-- Assuming h and k are functions with appropriate properties
variables (h k : ℝ → ℝ)
variables (h_inv : ℝ → ℝ) (k_inv : ℝ → ℝ)

-- Given condition: h_inv (k(x)) = 4 * x - 5
axiom h_inv_k_eq : ∀ x, h_inv (k x) = 4 * x - 5

-- Statement to prove
theorem find_k_inv_h_of_10 :
  k_inv (h 10) = 15 / 4 := 
sorry

end find_k_inv_h_of_10_l54_54374


namespace centers_form_rectangle_l54_54062

variables (O O1 O2 O3 : Type)
variable [metric_space O]
variables [metric_space O1] [metric_space O2] [metric_space O3]
variables (R R1 R2 R3 : ℝ)
variables (d : ℝ)

-- Following the conditions:
-- Conditions for the distances between the centers
variables (d1 : dist O O1 = R - R1)
variables (d2 : dist O O2 = R - R2)
variables (d3 : dist O O3 = R - R3)
variables (d4 : dist O1 O2 = R1 + R2)
variables (d5 : dist O1 O3 = R1 + R3)
variables (d6 : dist O2 O3 = R2 + R3)

-- The proof statement
theorem centers_form_rectangle 
  (h1 : dist O O1 = R - R1) 
  (h2 : dist O O2 = R - R2)
  (h3 : dist O O3 = R - R3)
  (h4 : dist O1 O2 = R1 + R2)
  (h5 : dist O1 O3 = R1 + R3)
  (h6 : dist O2 O3 = R2 + R3)
  (impossible_to_identify_center: ¬∀ center ∈ {O, O1, O2, O3}, true ∧ ∀ p q ∈ {O, O1, O2, O3}, dist p q = dist q p) :
  dist O O1 + dist O1 O2 + dist O2 O3 + dist O3 O = 0 → 
  is_rectangle O O1 O2 O3 :=
sorry

end centers_form_rectangle_l54_54062


namespace vasya_grades_l54_54672

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧
    (grades.filter (λ x, x = 5)).length > 2 ∧
    List.sorted (≤) grades ∧
    grades.nth 2 = some 4 ∧
    (grades.sum : ℚ) / 5 = 3.8 ∧
    grades = [2, 3, 4, 5, 5] := by
  sorry

end vasya_grades_l54_54672


namespace helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l54_54171

def heights_A : List ℝ := [3.6, -2.4, 2.8, -1.5, 0.9]
def heights_B : List ℝ := [3.8, -2, 4.1, -2.3]

theorem helicopter_A_highest_altitude :
  List.maximum heights_A = some 3.6 :=
by sorry

theorem helicopter_A_final_altitude :
  List.sum heights_A = 3.4 :=
by sorry

theorem helicopter_B_5th_performance :
  ∃ (x : ℝ), List.sum heights_B + x = 3.4 ∧ x = -0.2 :=
by sorry

end helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l54_54171


namespace f_value_l54_54561

def fractional_part (x : ℚ) : ℚ :=
  x - ⌊x⌋

def F (p : ℕ) : ℚ :=
  ∑ k in finset.range (p / 2 + 1), (k : ℚ)^120

noncomputable def f (p : ℕ) : ℚ :=
  1 / 2 - fractional_part (F p / p)

theorem f_value (p : ℕ) [fact (p.prime)] (h : 3 ≤ p) :
  f p = 
  if p ∣ 120 then
    1 / (2 * p)
  else
    1 / 2 :=
sorry

end f_value_l54_54561


namespace maximize_area_DEF_l54_54554

open Real

-- Definitions based on conditions
variables (A B C P D E F : Point)
variables (triangle_ABC : Triangle A B C)
variables (inside_triangle : ∀ P, Point.inside_triangle P triangle_ABC)
variables (intersection_D : Line A P ∩ Line B C = D)
variables (intersection_E : Line B P ∩ Line C A = E)
variables (intersection_F : Line C P ∩ Line A B = F)
variables (area_ABC : ℝ := Triangle.area triangle_ABC)

-- The area of the triangle DEF in terms of variables
noncomputable def area_DEF : ℝ := Triangle.area (Triangle.mk D E F)

-- The target centroid point
variable (centroid_P : Triangle.centroid triangle_ABC = P)

-- Statement of the problem
theorem maximize_area_DEF :
  inside_triangle P triangle_ABC →
  intersection_D →
  intersection_E →
  intersection_F →
  centroid_P →
  area_DEF = (1 / 4) * area_ABC :=
sorry

end maximize_area_DEF_l54_54554


namespace percentage_of_acid_is_18_18_percent_l54_54860

noncomputable def percentage_of_acid_in_original_mixture
  (a w : ℝ) (h1 : (a + 1) / (a + w + 1) = 1 / 4) (h2 : (a + 1) / (a + w + 2) = 1 / 5) : ℝ :=
  a / (a + w) 

theorem percentage_of_acid_is_18_18_percent :
  ∃ (a w : ℝ), (a + 1) / (a + w + 1) = 1 / 4 ∧ (a + 1) / (a + w + 2) = 1 / 5 ∧ percentage_of_acid_in_original_mixture a w (by sorry) (by sorry) = 18.18 := by
  sorry

end percentage_of_acid_is_18_18_percent_l54_54860


namespace distinct_ordered_triple_solutions_l54_54989

theorem distinct_ordered_triple_solutions :
  {a b c : ℕ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 50) → 
  ∃! (n : ℕ), n = 1176 :=
by
  sorry

end distinct_ordered_triple_solutions_l54_54989


namespace product_f_eq_one_l54_54715

def point (ℕ × ℕ)
def S : set point := {p | (p.1 ∈ {0, 1, 2, 3, 4, 5}) ∧ (p.2 ∈ {0, 1, 2, 3, 4, 5}) ∧ (p ≠ (0, 5))}
def is_right_triangle (A B C : point) : Prop := -- Define is_right_triangle conditionally ensuring B has a right angle.
  ((B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0)
def f (A B C : point) (h : is_right_triangle A B C) : ℝ := 
  real.sin (real.angle_between A B C) * real.tan (real.angle_between C A B)
def T : set (point × point × point) := {(A, B, C) | B ∈ S ∧ A ∈ S ∧ C ∈ S ∧ is_right_triangle A B C}

theorem product_f_eq_one : (∏ t in T, f t.1 t.2.1 t.2.2) = 1 :=
by {
  sorry
}

end product_f_eq_one_l54_54715


namespace tangent_line_eq_l54_54982

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

def derivative_curve (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

-- Define the problem as a theorem statement
theorem tangent_line_eq (L : ℝ → ℝ) (hL : ∀ x, L x = 2 * x ∨ L x = - x/4) :
  (∀ x, x = 0 → L x = 0) →
  (∀ x x0, L x = curve x → derivative_curve x0 = derivative_curve 0 → x0 = 0 ∨ x0 = 3/2) →
  (L x = 2 * x - curve x ∨ L x = 4 * x + curve x) :=
by
  sorry

end tangent_line_eq_l54_54982


namespace charles_drawn_after_work_l54_54909

-- Conditions
def total_papers : ℕ := 20
def drawn_today : ℕ := 6
def drawn_yesterday_before_work : ℕ := 6
def papers_left : ℕ := 2

-- Question and proof goal
theorem charles_drawn_after_work :
  ∀ (total_papers drawn_today drawn_yesterday_before_work papers_left : ℕ),
  total_papers = 20 →
  drawn_today = 6 →
  drawn_yesterday_before_work = 6 →
  papers_left = 2 →
  (total_papers - drawn_today - drawn_yesterday_before_work - papers_left = 6) :=
by
  intros total_papers drawn_today drawn_yesterday_before_work papers_left
  assume h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact sorry

end charles_drawn_after_work_l54_54909


namespace sufficient_but_not_necessary_condition_l54_54577

noncomputable def a_n (n : ℕ) (λ : ℝ) : ℝ := n^2 - 2 * λ * n

theorem sufficient_but_not_necessary_condition (λ : ℝ) :
  (∀ n : ℕ, n > 0 → a_n n λ - a_n (n - 1) λ > 0) → (λ < 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l54_54577


namespace ratio_p_r_l54_54399

variables (p q r s : ℚ)

theorem ratio_p_r (h1 : p / q = 5 / 4)
                  (h2 : r / s = 3 / 2)
                  (h3 : s / q = 1 / 5) :
  p / r = 25 / 6 := 
sorry

end ratio_p_r_l54_54399


namespace sum_of_roots_l54_54976

theorem sum_of_roots (x₁ x₂ : ℝ) (h1 : x₁^2 = 2 * x₁ + 1) (h2 : x₂^2 = 2 * x₂ + 1) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l54_54976


namespace possible_years_count_l54_54861
open Nat

def is_prime_digit (n : Nat) : Prop :=
  n = 2 ∨ n = 5

theorem possible_years_count : ∃ (count : Nat), 
  count = ∑ d in [2, 5, 9], 
    if d = 2 then (Nat.factorial 5) / ((Nat.factorial 2) * (Nat.factorial 2)) -- case when year starts with 2
    else if d = 5 then (Nat.factorial 5) / (Nat.factorial 2) -- case when year starts with 5
    else (Nat.factorial 5) / ((Nat.factorial 2) * (Nat.factorial 2)) -- case when year starts with 9
  ∧ count = 120 :=
by
  sorry

end possible_years_count_l54_54861


namespace sum_first_99_natural_numbers_l54_54547

theorem sum_first_99_natural_numbers :
  (∑ i in Finset.range 99.succ, i) = 4950 := by
  sorry

end sum_first_99_natural_numbers_l54_54547


namespace area_of_triangle_ABC_l54_54296

-- Define the sides of the triangle
def AB : ℝ := 12
def BC : ℝ := 9

-- Define the expected area of the triangle
def expectedArea : ℝ := 54

-- Prove the area of the triangle using the given conditions
theorem area_of_triangle_ABC : (1/2) * AB * BC = expectedArea := 
by
  sorry

end area_of_triangle_ABC_l54_54296


namespace exponentiation_81_5_4_eq_243_l54_54179

theorem exponentiation_81_5_4_eq_243 : 81^(5/4) = 243 := by
  sorry

end exponentiation_81_5_4_eq_243_l54_54179


namespace problem1_problem2_l54_54904

variable {a b : ℝ}

-- Proof problem 1
-- Goal: (1)(2a^(2/3)b^(1/2))(-6a^(1/2)b^(1/3)) / (-3a^(1/6)b^(5/6)) = -12a
theorem problem1 (h1 : 0 < a) (h2 : 0 < b) : 
  (1 : ℝ) * (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = -12 * a := 
sorry

-- Proof problem 2
-- Goal: 2(log(sqrt(2)))^2 + log(sqrt(2)) * log(5) + sqrt((log(sqrt(2)))^2 - log(2) + 1) = 1 + (1 / 2) * log(5)
theorem problem2 : 
  2 * (Real.log (Real.sqrt 2))^2 + (Real.log (Real.sqrt 2)) * (Real.log 5) + 
  Real.sqrt ((Real.log (Real.sqrt 2))^2 - Real.log 2 + 1) = 
  1 + 0.5 * (Real.log 5) := 
sorry

end problem1_problem2_l54_54904


namespace arith_progression_possible_values_l54_54790

theorem arith_progression_possible_values :
  ∃ n_set : Finset ℕ, 
    (card n_set = 14 ∧ 
    ∀ n ∈ n_set, 
      1 < n ∧ 
      ∃ a : ℤ, 
        0 < a ∧ 
        2 * 180 = n * (2 * a + (n - 1) * 3)) :=
by { sorry }

end arith_progression_possible_values_l54_54790


namespace ratio_determinable_l54_54766

theorem ratio_determinable (a b c x y : ℝ) (h : a * x^2 + 2 * b * x * y + c * y^2 = 0) :
  y ≠ 0 →
  ( ∃ t : ℝ, t = (-b + sqrt (b^2 - a * c)) / a ∨ t = (-b - sqrt (b^2 - a * c)) / a ∧ t = x / y ) :=
by
  intro hy
  have h_equiv : a * (x / y) ^ 2 + 2 * b * (x / y) + c = 0 := sorry
  use (x / y)
  rw [h_equiv]
  left
  exact sorry

end ratio_determinable_l54_54766


namespace negation_proposition_l54_54392

-- Definitions based on the conditions
def original_proposition : Prop := ∃ x : ℝ, x^2 + 3*x + 2 < 0

-- Theorem requiring proof
theorem negation_proposition : (¬ original_proposition) = ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_proposition_l54_54392


namespace compute_double_binomial_coefficient_l54_54156

-- Define the binomial coefficient.
def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- The theorem we want to prove.
theorem compute_double_binomial_coefficient : 2 * binomial_coefficient 12 3 = 440 :=
by sorry

end compute_double_binomial_coefficient_l54_54156


namespace min_value_fraction_sum_l54_54968

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_collinear : 3 * a + 2 * b = 1)

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_collinear : 3 * a + 2 * b = 1) : 
  (3 / a + 1 / b) = 11 + 6 * Real.sqrt 2 :=
by
  sorry

end min_value_fraction_sum_l54_54968


namespace candy_difference_l54_54490

def given_away : ℕ := 6
def left : ℕ := 5
def difference : ℕ := given_away - left

theorem candy_difference :
  difference = 1 :=
by
  sorry

end candy_difference_l54_54490


namespace value_of_a_2018_l54_54996

noncomputable def sequence (n : ℕ) : ℚ :=
  Nat.recOn n 2 (λ n a_n, (1 + a_n) / (1 - a_n))

theorem value_of_a_2018 : sequence 2018 = -3 :=
by sorry

end value_of_a_2018_l54_54996


namespace min_area_of_triangle_AOB_l54_54570

theorem min_area_of_triangle_AOB (m n : ℝ) 
  (h_chord_length : ∀ (x y : ℝ), (m * x + n * y - 1 = 0) ∧ (x^2 + y^2 = 4) → sqrt(4 - (x + y)^2/4) = 2)
  (h_line : ∀ (x : ℝ), y = 0 → m * x - 1 = 0)
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 = 4) : 
  (∃ A B : ℝ × ℝ, A = (1/m, 0) ∧ B = (0, 1/n) ∧ 
  sqrt(3) = 1 / sqrt(m^2 + n^2) ∧
  m^2 + n^2 = 1 / 3 ∧ 
  1 / (2*abs(m*n)) ≥ 3) :=
sorry

end min_area_of_triangle_AOB_l54_54570


namespace sum_a_eq_2525_l54_54954

variable (a : ℕ → ℕ)

theorem sum_a_eq_2525 (h1 : a 1 + a 2 = 1)
                      (h2 : a 2 + a 3 = 2)
                      (h3 : a 3 + a 4 = 3)
                      ...
                      (h99 : a 99 + a 100 = 99)
                      (h100 : a 100 + a 1 = 100) :
  (Finset.range 100).sum a = 2525 :=
sorry

end sum_a_eq_2525_l54_54954


namespace ellipse_property_isosceles_right_triangle_inscribed_l54_54988

theorem ellipse_property (a b : ℝ) (h : a > b ∧ b > 0)
  (hyp_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (point_P_angle : ∃ P : ℝ × ℝ, P ∈ {XY : ℝ × ℝ | hyp_ellipse XY.fst XY.snd} 
    ∧ ∃ (F₁ F₂ : ℝ × ℝ), ∠(F₁, P, F₂) = 60)
  (triangle_area : S_Δ (PF₁F₂) = sqrt(3) / 3) :
  b = 1 := sorry

theorem isosceles_right_triangle_inscribed (a b : ℝ) (h₁ : a = 2) (h₂ : b = 1) 
  (A : ℝ × ℝ) (hA : A = (0, b))
  (equation_ellipse : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1) :
  ∃ (triangles_count : ℕ), triangles_count = 3 := sorry

end ellipse_property_isosceles_right_triangle_inscribed_l54_54988


namespace students_zero_marks_count_l54_54282

theorem students_zero_marks_count:
  ∃ (z : ℕ), 
    let total_students := 20,
    let students_scored_100 := 2,
    let students_scored_zero := z,
    let students_other := total_students - students_scored_100 - students_scored_zero,
    let avg_other := 40,
    let total_marks_other := avg_other * students_other,
    let total_marks_100 := students_scored_100 * 100,
    let total_marks_zero := 0 * students_scored_zero,
    let avg_total := 40,
    let total_marks_class := avg_total * total_students,
    total_marks_other + total_marks_100 + total_marks_zero = total_marks_class 
and students_scored_zero = 3 := by
  sorry

end students_zero_marks_count_l54_54282


namespace bacteria_growth_l54_54097

theorem bacteria_growth (n : ℕ) (start_bacteria : ℕ) (final_bacteria : ℕ)
  (h_split_rate: ∀ t, bacteria_after t = 2^t) : 
  (start_bacteria = 1 ∧ bacteria_after 64 = 2^64) →
  (start_bacteria = 4 ∧ bacteria_after (62 + 2) = 2^64) :=
begin
  intro h,
  cases h with h_start_one h_final_one,
  have h_start_four : bacteria_after 0 = 4 := by sorry, -- should show bacteria_after 0 with 4 bacteria coincides with 2^2
  have h_final_four : final_bacteria = 2^64 := by sorry,  -- akin to doubling 64 times
  have time_difference : 2 = 64 - 62 := by sorry, -- The effective "head start"
  exact (start_bacteria, bacteria_after),
end

end bacteria_growth_l54_54097


namespace wind_velocity_l54_54786

variable (P A V : Real)
variable (k : Real)
variable (velocity initial_area final_area initial_pressure final_pressure : Real)

theorem wind_velocity :
  (initial_area = 9) →
  (initial_pressure = 4) →
  (velocity = 105) →
  (P = k * A * velocity^2) →
  (final_area = 36) →
  (final_pressure = 64) →
  final_pressure = k * final_area * V^2 →
  V = 70 := 
begin
  intros,
  sorry
end

end wind_velocity_l54_54786


namespace sawing_steel_bar_time_l54_54760

theorem sawing_steel_bar_time (pieces : ℕ) (time_per_cut : ℕ) : 
  pieces = 6 → time_per_cut = 2 → (pieces - 1) * time_per_cut = 10 := 
by
  intros
  sorry

end sawing_steel_bar_time_l54_54760


namespace total_workers_l54_54692

theorem total_workers (P : ℚ) : (P = 0.047619047619047616) → ∀ (N : ℕ), (C(N, 2) = 21) → (N = 7) := by
  sorry

end total_workers_l54_54692


namespace find_number_l54_54096

theorem find_number (x n : ℕ) (h1 : 3 * x + n = 48) (h2 : x = 4) : n = 36 :=
by
  sorry

end find_number_l54_54096


namespace problem1_problem2_problem3_problem4_l54_54979

-- Problem 1: Proving the value of P
theorem problem1 :
  (P = ∑ i in finset.range 99, (1 : ℝ) / ((i + 1) * (i + 2))) → P = 99 / 100 :=
sorry

-- Problem 2: Proving the value of Q
theorem problem2 (P : ℝ) (hP : P = 99 / 100) :
  (99 * Q = P * ∑ i in finset.range 100, (99 / 100) ^ i) → Q = 1 :=
sorry

-- Problem 3: Max value of R
theorem problem3 (x : ℝ) (R : ℝ) (h : ∀ x, (2 * x^2 + 2 * R * x + R) / (4 * x^2 + 6 * x + 3) ≤ 1) :
  R ≤ 3 :=
sorry

-- Problem 4: Value of S
theorem problem4 (R : ℝ) :
  (S = log (144) (2^(1/R)) + log (144) (R^(1/R))) → S = 1/12 :=
sorry

end problem1_problem2_problem3_problem4_l54_54979


namespace similar_triangle_area_l54_54808

noncomputable theory

def triangle_ABC_area (AB AC : ℝ) : ℝ := 
(1 / 2) * AB * AC

def triangle_XYZ_area (k : ℝ) (ABC_area : ℝ) : ℝ := 
k^2 * ABC_area

theorem similar_triangle_area
  (AB AC : ℝ)
  (h_AB : AB = 8)
  (h_AC : AC = 8 * (real.sqrt 3 / 3))
  (ABC_area : ℝ := triangle_ABC_area AB AC)
  (k : ℝ)
  (h_k : k^2 = 1 / 4)
  (XYZ_area : ℝ := triangle_XYZ_area k ABC_area)
  (p q r s t : ℕ)
  (h_XYZ_form : XYZ_area = (p * (real.sqrt q) - r * (real.sqrt s)) / t)
  (h_gcd : nat.gcd t (nat.gcd r p) = 1)
  (h_prime_divisibility : ¬ ∃ (prime : ℕ) (hp : nat.prime prime), prime^2 ∣ q ∨ prime^2 ∣ s) :
  p + q + r + s + t = 15 :=
begin
  sorry
end

end similar_triangle_area_l54_54808


namespace cheesecake_needs_more_eggs_l54_54100

def chocolate_eggs_per_cake := 3
def cheesecake_eggs_per_cake := 8
def num_chocolate_cakes := 5
def num_cheesecakes := 9

theorem cheesecake_needs_more_eggs :
  cheesecake_eggs_per_cake * num_cheesecakes - chocolate_eggs_per_cake * num_chocolate_cakes = 57 :=
by
  sorry

end cheesecake_needs_more_eggs_l54_54100


namespace missing_fraction_is_correct_l54_54051

theorem missing_fraction_is_correct :
  (1 / 3 + 1 / 2 + -5 / 6 + 1 / 5 + -9 / 20 + -9 / 20) = 0.45 - (23 / 20) :=
by
  sorry

end missing_fraction_is_correct_l54_54051


namespace value_range_l54_54794

noncomputable def func (x : ℝ) : ℝ := Math.sin x - Math.sin (abs x)

theorem value_range : Set.range func = Set.Icc (-2 : ℝ) 2 := 
by
  sorry

end value_range_l54_54794


namespace students_history_not_statistics_l54_54651

def num_students := 89
def students_history := 36
def students_statistics := 32
def students_history_or_statistics := 59

theorem students_history_not_statistics : (students_history - (students_history + students_statistics - students_history_or_statistics)) = 27 :=
by 
  -- The ultimate goal lemma to solve the problem using inclusion-exclusion principle
  let num_both := students_history + students_statistics - students_history_or_statistics 
  have h_num_both : num_both = 9 :=
    calc
      num_both = 36 + 32 - 59 : by refl

  exact calc
    (students_history - num_both) = (36 - 9) : by rw h_num_both
                        ... = 27 : by refl

end students_history_not_statistics_l54_54651


namespace g_f_x_not_quadratic_l54_54267

open Real

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_f_x_not_quadratic (h : ∃ x : ℝ, x - f (g x) = 0) :
  ∀ x : ℝ, g (f x) ≠ x^2 + x + 1 / 5 := sorry

end g_f_x_not_quadratic_l54_54267


namespace find_range_of_a_l54_54992

noncomputable def f (a x : ℝ) : ℝ := a / x - Real.exp (-x)

theorem find_range_of_a (p q a : ℝ) (h : 0 < a) (hpq : p < q) :
  (∀ x : ℝ, 0 < x → x ∈ Set.Icc p q → f a x ≤ 0) → 
  (0 < a ∧ a < 1 / Real.exp 1) :=
by
  sorry

end find_range_of_a_l54_54992


namespace evaluate_imaginary_expression_l54_54188

theorem evaluate_imaginary_expression (i : ℂ) (h : i^2 = -1) : i^8 + i^{20} + i^{-40} + 3 = 6 :=
by
  sorry

end evaluate_imaginary_expression_l54_54188


namespace math_proof_statement_l54_54239

variables (a b : ℝ^3)
variables (θ : ℝ)

-- Given conditions
def magnitude_a : ℝ := 2
def magnitude_b : ℝ := 2
def projection_a_on_b : ℝ := -1
def norm_a : ℝ := real.sqrt (a.dot_product a)
def norm_b : ℝ := real.sqrt (b.dot_product b)

-- Definitions to express the conditions in Lean
def norm_a_is_2 : Prop := norm_a = magnitude_a
def norm_b_is_2 : Prop := norm_b = magnitude_b
def projection_condition : Prop := (a.dot_product b) / norm_b = projection_a_on_b

-- To prove the angle θ is 120 degrees
def cos_theta : ℝ := -1 / 2
def angle_condition : Prop := real.cos θ = cos_theta

-- To prove the magnitude of a - 2b
def a_minus_2b : ℝ^3 := a - 2 • b
def norm_a_minus_2b : ℝ := real.sqrt ((a_minus_2b).dot_product (a_minus_2b))
def magnitude_a_minus_2b : ℝ := 2 * real.sqrt 7

def proof_problem : Prop :=
  norm_a_is_2 ∧ norm_b_is_2 ∧ projection_condition → 
  (angle_condition ∧ (norm_a_minus_2b = magnitude_a_minus_2b))

-- Lean statement
theorem math_proof_statement (a b : ℝ^3) (θ : ℝ) :
  proof_problem :=
by sorry

end math_proof_statement_l54_54239


namespace angle_CED_eq_180_degrees_l54_54064

-- Definitions
variable {Point : Type} [MetricSpace Point]

noncomputable def centerA : Point := sorry
noncomputable def centerB : Point := sorry
noncomputable def radiusA : ℝ := 5
noncomputable def radiusB : ℝ := 3
noncomputable def left_intersection_point (A B : Point) : Point := sorry
noncomputable def right_intersection_point (A B : Point) : Point := sorry
noncomputable def intersection_points : Set Point := sorry
noncomputable def common_point : Point := sorry

-- Conditions
axiom condition1 : dist centerA centerB = 2
axiom condition2 : (∀ p ∈ intersection_points, dist p centerA = 5 ∧ dist p centerB = 3)
axiom condition3 : left_intersection_point centerA centerB ∈ intersection_points
axiom condition4 : right_intersection_point centerA centerB ∈ intersection_points
axiom condition5 : common_point ∈ intersection_points

-- Goal
theorem angle_CED_eq_180_degrees :
  angle (left_intersection_point centerA centerB) common_point (right_intersection_point centerA centerB) = 180 :=
sorry

end angle_CED_eq_180_degrees_l54_54064


namespace vasya_grades_l54_54684

-- Given conditions
constants (a1 a2 a3 a4 a5 : ℕ)
axiom grade_median : a3 = 4
axiom grade_sum : a1 + a2 + a3 + a4 + a5 = 19
axiom most_A_grades : ∀ (n : ℕ), n ≠ 5 → (∃ m, m > 0 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 5 → (if a1 = n ∨ a2 = n ∨ a3 = n ∨ a4 = n ∨ a5 = n then m > 1 else m = 0))

-- Prove that the grades are (2, 3, 4, 5, 5)
theorem vasya_grades : (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨ 
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) := 
by sorry

end vasya_grades_l54_54684


namespace pounds_over_minimum_l54_54140

noncomputable def cost_per_pound : ℝ := 3
noncomputable def min_purchase_pounds : ℝ := 15
noncomputable def bulk_discount_rate : ℝ := 0.10
noncomputable def early_bird_discount_rate : ℝ := 0.05
noncomputable def sales_tax_rate : ℝ := 0.08
noncomputable def total_spent : ℝ := 119.88

def bulk_discount_applies (pounds : ℝ) : Prop := pounds ≥ 25
def early_bird_discount_applies (before_10am : Bool) : Prop := before_10am = true
noncomputable def get_total_cost (pounds : ℝ) (before_10am : Bool) : ℝ :=
  let base_cost := pounds * cost_per_pound
  let bulk_discount := if bulk_discount_applies pounds then 1 - bulk_discount_rate else 1
  let early_bird_discount := if early_bird_discount_applies before_10am then 1 - early_bird_discount_rate else 1
  let discounted_cost := base_cost * bulk_discount * early_bird_discount
  discounted_cost * (1 + sales_tax_rate)

theorem pounds_over_minimum (before_10am : Bool) (h1 : get_total_cost 43 true = total_spent) :
  43 - min_purchase_pounds = 28 := sorry

end pounds_over_minimum_l54_54140


namespace sum_geometric_series_l54_54159

noncomputable def S_n (n : ℕ) : ℝ :=
  3 - 3 * ((2 / 3)^n)

theorem sum_geometric_series (a : ℝ) (r : ℝ) (n : ℕ) (h_a : a = 1) (h_r : r = 2 / 3) :
  S_n n = a * (1 - r^n) / (1 - r) :=
by
  sorry

end sum_geometric_series_l54_54159


namespace tap_A_turn_off_time_l54_54067

theorem tap_A_turn_off_time
  (C : ℝ) 
  (tapA_rate : ℝ := C / 12) 
  (tapB_rate : ℝ := C / 18) 
  (total_time_fill_B : ℝ := 8)
  (remaining_fill_by_B : ℝ := total_time_fill_B * tapB_rate)
  (combined_rate : ℝ := tapA_rate + tapB_rate)
  (filled_by_both : ℝ → ℝ := λ t, t * combined_rate) :
  ∀ t : ℝ, filled_by_both t + remaining_fill_by_B = C → t = 4 := 
begin
  intros t h,
  -- transformation to isolate t
  have h₁ : filled_by_both t = C - remaining_fill_by_B, from eq_sub_of_add_eq h,
  have h₂ : t * combined_rate = C - remaining_fill_by_B, from h₁,
  -- use the values of rates
  calc t *
    combined_rate = C - remaining_fill_by_B : (h₂)
    ... = C - total_time_fill_B * tapB_rate : by rw remaining_fill_by_B
    ... = C - 8 * (C / 18) : by rw total_time_fill_B
    ... = C - (8/18) * C : by ring
    ... = C * (1 - 8/18) : by ring
    ... = C * (10/18) : by norm_num
    ... = C * (5/9) : by norm_num
    ... = 5C/9 : by ring_nf
    -- and so on until t is isolated
  sorry  -- complete the proof if necessary

end tap_A_turn_off_time_l54_54067


namespace denali_star_speed_l54_54773

theorem denali_star_speed (length : ℝ := 1 / 6) (time : ℝ := 10) (glacier_speed : ℝ := 70) : 
  ∃ v : ℝ, v = 50 :=
by
  let relative_speed := v + glacier_speed
  let relative_speed_mps := relative_speed / 3600
  let distance_covered := time * relative_speed_mps
  have h_distance : distance_covered = 2 * length := rfl
  have h1 : distance_covered = 10 * (v + 70) / 3600 := rfl
  have h2 : (v + 70) / 3600 = 1 / 3 / 10 := sorry
  have h3 : v + 70 = 120 := sorry
  use v
  have h4 : v = 120 - 70 := sorry
  have h5 : v = 50 := sorry
  exact ⟨v, h5⟩

end denali_star_speed_l54_54773


namespace find_sum_of_squares_l54_54712

-- Define the vertices of the triangle as points in Euclidean space.
variables {V : Type*} [inner_product_space ℝ V] 
variables {A B C G : V}

-- Define the condition that G is the centroid of triangle ABC.
def is_centroid (A B C G : V) : Prop :=
  G = (A + B + C) / 3

-- The main theorem statement
theorem find_sum_of_squares 
  (h_centroid : is_centroid A B C G)
  (h_sum_squares : (∥G - A∥^2 + ∥G - B∥^2 + ∥G - C∥^2) = 88) :
  (∥A - B∥^2 + ∥A - C∥^2 + ∥B - C∥^2) = 396 :=
sorry

end find_sum_of_squares_l54_54712


namespace exists_infinite_set_no_kth_power_sum_l54_54166

open Classical

theorem exists_infinite_set_no_kth_power_sum : 
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ (∀ (T : Set ℕ), T ⊆ S ∧ T.Finite → ∀ k ≥ 2, ¬∃ m, ∑ t in T, t = m^k) := 
by
  sorry

end exists_infinite_set_no_kth_power_sum_l54_54166


namespace male_and_female_solo_artists_15_25_l54_54897

def num_contestants : ℕ := 18
def female_solo_artist_percentage : ℕ := 35
def male_solo_artist_percentage : ℕ := 25
def duet_or_group_percentage : ℕ := 40
def male_15_25_percentage : ℕ := 30
def female_15_25_percentage : ℕ := 20

theorem male_and_female_solo_artists_15_25 :
  let num_female_solo : ℕ := (female_solo_artist_percentage * num_contestants) / 100,
      num_male_solo : ℕ := (male_solo_artist_percentage * num_contestants) / 100,
      num_male_15_25 : ℕ := (male_15_25_percentage * num_male_solo) / 100,
      num_female_15_25 : ℕ := (female_15_25_percentage * num_female_solo) / 100 in
  num_male_15_25 = 1 ∧ num_female_15_25 = 1 :=
by
  sorry

end male_and_female_solo_artists_15_25_l54_54897


namespace vasya_grades_l54_54661

variables
  (grades : List ℕ)
  (length_grades : grades.length = 5)
  (median_grade : grades.nthLe 2 sorry = 4)  -- Assuming 0-based indexing
  (mean_grade : (grades.sum : ℚ) / 5 = 3.8)
  (most_frequent_A : ∀ n ∈ grades, n ≤ 5)

theorem vasya_grades (h : ∀ x ∈ grades, x ≤ 5 ∧ ∃ k, grades.nthLe 3 sorry = 5 ∧ grades.count 5 > grades.count x):
  ∃ g1 g2 g3 g4 g5 : ℕ, grades = [g1, g2, g3, g4, g5] ∧ g1 ≤ g2 ∧ g2 ≤ g3 ∧ g3 ≤ g4 ∧ g4 ≤ g5 ∧ [g1, g2, g3, g4, g5] = [2, 3, 4, 5, 5] :=
sorry

end vasya_grades_l54_54661


namespace daria_credit_card_debt_l54_54509

def discounted_price (original_price : ℝ) (discount_percentage : ℝ) : ℝ :=
  original_price * (1 - discount_percentage / 100)

def total_furniture_cost : ℝ :=
  let couch := discounted_price 750 10
  let table := discounted_price 100 5
  let lamp := 50
  let rug := discounted_price 200 15
  let bookshelf := discounted_price 150 20
  couch + table + lamp + rug + bookshelf

theorem daria_credit_card_debt : total_furniture_cost - 500 = 610 := by
  -- proof here
  sorry

end daria_credit_card_debt_l54_54509


namespace partial_deriv_sum_eq_one_l54_54708

noncomputable def z (x y : ℝ) : ℝ := sorry -- z(x,y) is implicitly defined

def F (x y z : ℝ) := 2 * Real.sin (x + 2 * y - 3 * z) - (x + 2 * y - 3 * z)

def Fx (x y z : ℝ) := 2 * Real.cos (x + 2 * y - 3 * z) - 1
def Fy (x y z : ℝ) := 4 * Real.cos (x + 2 * y - 3 * z) - 2
def Fz (x y z : ℝ) := -6 * Real.cos (x + 2 * y - 3 * z) + 3

theorem partial_deriv_sum_eq_one (x y : ℝ) (hz : F x y (z x y) = 0) :
  (∂ z / ∂ x) + (∂ z / ∂ y) = 1 := by
  sorry

end partial_deriv_sum_eq_one_l54_54708


namespace symmetry_axis_cos_shifted_l54_54792

theorem symmetry_axis_cos_shifted:
  ∃ k : ℤ, ∀ x : ℝ, x = (k * π) / 2 + π / 12 →
  (∀ y : ℝ, y = cos (2 * x - π / 6) → 
    (∃ k' : ℤ, ∀ x' : ℝ, x' = k' * π → y = cos x')) := 
begin
  sorry
end

end symmetry_axis_cos_shifted_l54_54792


namespace find_even_and_increasing_function_l54_54488

-- Define the functions
def f1 (x : ℝ) := x^3
def f2 (x : ℝ) := abs x + 1
def f3 (x : ℝ) := -x^2 + 1
def f4 (x : ℝ) := 1 / x^2

-- Define even functions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Define increasing functions on (0, +∞)
def is_increasing_on_pos (f : ℝ → ℝ) := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- The proof problem statement
theorem find_even_and_increasing_function :
  is_even f2 ∧ is_increasing_on_pos f2 ∧
  (∀ f : ℝ → ℝ, (f = f1 ∨ f = f3 ∨ f = f4) → ¬ (is_even f ∧ is_increasing_on_pos f)) :=
sorry

end find_even_and_increasing_function_l54_54488


namespace triangles_with_area_leq_two_thirds_l54_54796

/-- 
There are n points on a plane such that no three of them are collinear. 
Number of triangles formed by these points, each with an area of 1, 
is at most 2/3 * (n^2 - n).
 -/
theorem triangles_with_area_leq_two_thirds (n : ℕ) (h : ∀ (A B C : Point), 
(A ≠ B ∧ B ≠ C ∧ A ≠ C) → triangle_area A B C ≠ 1) :
  ∃ (m : ℕ), set_of_triangles_with_area_one.card ≤ (2 * (n^2 - n)) / 3 :=
sorry

end triangles_with_area_leq_two_thirds_l54_54796


namespace shortest_distance_l54_54331

-- Define the parabola and line
def parabola (x : ℝ) : ℝ := x^2 - 8 * x + 18
def line (x : ℝ) : ℝ := 2 * x - 8

-- Define the distance function from a point to the line
def distance_to_line (a : ℝ) : ℝ :=
  (abs ((2 * a) - (parabola a) - 8)) / (Real.sqrt 5)

-- Prove that the shortest possible distance is 1 / sqrt(5)
theorem shortest_distance : ∃ a : ℝ, distance_to_line a = 1 / (Real.sqrt 5) :=
by
  sorry

end shortest_distance_l54_54331


namespace all_xi_eq_one_l54_54370

variable (n : ℕ)
variable (x : Fin n → ℝ)

-- Conditions
def sum_eq_n : Prop := ∑ i, x i = n
def sum_squares_eq_n : Prop := ∑ i, (x i) ^ 2 = n
def sum_cubes_eq_n : Prop := ∑ i, (x i) ^ 3 = n

-- All k-th power sums are equal to n
def sum_kth_power_eq_n (k : ℕ) : Prop := ∑ i, (x i) ^ k = n

-- The main theorem to prove given the conditions
theorem all_xi_eq_one (h1 : sum_eq_n x n)
                      (h2 : sum_squares_eq_n x n)
                      (h3 : ∀ k : ℕ, 1 ≤ k → k ≤ n → sum_kth_power_eq_n x n k)
                      : ∀ i, x i = 1 := 
  sorry

end all_xi_eq_one_l54_54370


namespace curve_symmetry_symmetric_y_eq_x_l54_54246

theorem curve_symmetry_symmetric_y_eq_x (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + a^2 * x + (1 - a^2) * y - 4 = 0) → 
  (∀ (x y : ℝ), (x, y) = (-a^2 / 2, -(1 - a^2) / 2) → (-a^2 / 2 = -(1 - a^2) / 2) → a = real.sqrt 2 / 2 ∨ a = -real.sqrt 2 / 2) := 
by
  intros
  sorry

end curve_symmetry_symmetric_y_eq_x_l54_54246


namespace find_value_expression_l54_54192

theorem find_value_expression :
  (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 :=
by 
  have h1 : (-a)^(n : ℕ) = if even n then a^n else -a^n,
  {
    intro a,
    intro n,
    exact if even n then (pow_even a n).symm else (pow_odd a n).symm,
  },
  sorry

end find_value_expression_l54_54192


namespace vasya_grades_l54_54683

-- Given conditions
constants (a1 a2 a3 a4 a5 : ℕ)
axiom grade_median : a3 = 4
axiom grade_sum : a1 + a2 + a3 + a4 + a5 = 19
axiom most_A_grades : ∀ (n : ℕ), n ≠ 5 → (∃ m, m > 0 ∧ ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 5 → (if a1 = n ∨ a2 = n ∨ a3 = n ∨ a4 = n ∨ a5 = n then m > 1 else m = 0))

-- Prove that the grades are (2, 3, 4, 5, 5)
theorem vasya_grades : (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨ 
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) ∨
                       (a1 = 2 ∧ a2 = 3 ∧ a3 = 4 ∧ a4 = 5 ∧ a5 = 5) := 
by sorry

end vasya_grades_l54_54683


namespace B_speed_correct_l54_54125

noncomputable def A_speed : ℝ := 4
noncomputable def B_time : ℝ := 108 / 60
noncomputable def A_initial_time : ℝ := 0.5
noncomputable def A_distance_before_B_starts : ℝ := A_speed * A_initial_time
noncomputable def A_additional_distance : ℝ := A_speed * B_time
noncomputable def total_A_distance : ℝ := A_distance_before_B_starts + A_additional_distance
noncomputable def B_speed : ℝ := total_A_distance / B_time

theorem B_speed_correct : B_speed ≈ 5.11 := 
by 
  sorry

end B_speed_correct_l54_54125


namespace filter_replacement_month_l54_54326

theorem filter_replacement_month (n : ℕ) (h : n = 25) : (7 * (n - 1)) % 12 = 0 → "January" = "January" :=
by
  intros
  sorry

end filter_replacement_month_l54_54326


namespace slope_range_of_line_l54_54222

theorem slope_range_of_line
  (a b : ℝ)
  (h : ∃ C : ℝ × ℝ, C = (2, 2) ∧ (C.1 - 2)^2 + (C.2 - 2)^2 = 18)
  (line : ∀ x y : ℝ, a * x + b * y = 0 → ∥(x, y) - (2, 2)∥ = 2*sqrt 2)
  : 2 - sqrt 3 ≤ -a / b ∧ -a / b ≤ 2 + sqrt 3 :=
by
  sorry

end slope_range_of_line_l54_54222


namespace magnitude_of_b_l54_54260

-- Definitions of vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (x, y)

-- Conditions
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def norm (u : ℝ × ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2)

-- Given conditions
axiom a_dot_b_eq_10 : dot_product a b = 10
axiom norm_a_plus_b_eq_5 : norm (a.1 + b.1, a.2 + b.2) = 5

-- Proof statement
theorem magnitude_of_b : norm b = sqrt 20 := by
  sorry

end magnitude_of_b_l54_54260


namespace eval_exp_l54_54173

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end eval_exp_l54_54173


namespace eval_power_l54_54183

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end eval_power_l54_54183


namespace locus_of_Q_is_ellipse_l54_54837

theorem locus_of_Q_is_ellipse (m n x0 y0 : ℝ) (hm : m > 0) (hn : n > 0)
    (h_inside : m * x0 ^ 2 + n * y0 ^ 2 < 1) (h_not_center : x0 ≠ 0 ∨ y0 ≠ 0) :
    ∃ (x y : ℝ), m * (x - x0) ^ 2 + n * (y - y0) ^ 2 = m * x0 ^ 2 + n * y0 ^ 2 ∧ (m * x ^ 2 + n * y ^ 2 ≤ 1) :=
begin
  sorry
end

end locus_of_Q_is_ellipse_l54_54837


namespace good_number_set_n_is_odd_and_min_n_equal_7_l54_54963

def is_good_number_set (A : Finset ℕ) : Prop :=
  ∀ (a ∈ A), ∃ (B C : Finset ℕ), B ∩ C = ∅ ∧ B ∪ C = A \ {a} ∧ B.sum = C.sum

theorem good_number_set_n_is_odd_and_min_n_equal_7 (A : Finset ℕ) (hA : is_good_number_set A) :
  ∃ (n : ℕ), A.card = n ∧ ¬ even n ∧ n ≥ 7 :=
by
  sorry

end good_number_set_n_is_odd_and_min_n_equal_7_l54_54963


namespace floor_eq_for_odd_n_ge_3_l54_54559

noncomputable def A_n (n : ℕ) : ℝ :=
  ∑ k in finset.range n, real.sqrt (n ^ 2 + (2 * k + 1))

noncomputable def B_n (n : ℕ) : ℝ :=
  ∑ k in finset.range n, real.sqrt (n ^ 2 + 2 * (k + 1))

theorem floor_eq_for_odd_n_ge_3 (n : ℕ) (h : n ≥ 3) (h_odd : odd n) :
  (int.floor (A_n n)) = (int.floor (B_n n)) := 
sorry

end floor_eq_for_odd_n_ge_3_l54_54559


namespace problem_solution_l54_54213

open Real

noncomputable def vector_a (θ : ℝ) : ℝ × ℝ :=
  (cos θ, sin θ)

noncomputable def vector_b (ϕ : ℝ) : ℝ × ℝ :=
  (cos ϕ, sin ϕ)

noncomputable def option_a (θ ϕ : ℝ) : Prop :=
  ‖(vector_a θ).1 + (vector_b ϕ).1, (vector_a θ).2 + (vector_b ϕ).2‖ = 
  ‖(vector_a θ).1 - (vector_b ϕ).1, (vector_a θ).2 - (vector_b ϕ).2‖

noncomputable def option_b (θ ϕ : ℝ) : Prop :=
  ‖(vector_a θ).1 - (vector_b ϕ).1, (vector_a θ).2 - (vector_b ϕ).2‖ = 1

noncomputable def option_c (θ ϕ : ℝ) : Prop :=
  ((vector_a θ).1 + (vector_b ϕ).1, (vector_a θ).2 + (vector_b ϕ).2) • 
  ((vector_a θ).1 - (vector_b ϕ).1, (vector_a θ).2 - (vector_b ϕ).2) = 1

noncomputable def option_d (θ ϕ : ℝ) : Prop :=
  ‖(4 * (vector_a θ).1 - 5 * (vector_b ϕ).1, 4 * (vector_a θ).2 - 5 * (vector_b ϕ).2)‖ = 6

theorem problem_solution (θ ϕ : ℝ) : 
  (option_a θ ϕ ∨ option_b θ ϕ ∨ option_d θ ϕ) ↔ (cos (θ - ϕ) = 0) ∨ (cos (θ - ϕ) = 1 / 2) ∨ (cos (θ - ϕ) = 1 / 8) := 
sorry

end problem_solution_l54_54213


namespace mean_of_data_set_l54_54122

theorem mean_of_data_set (data : List ℝ) (h : data = [2, 4, 5, 6, 5]) :
  (data.sum / data.length.toReal) = 4.4 :=
by
  -- This is the definition of the data and its computed mean, according to the problem.
  sorry

end mean_of_data_set_l54_54122


namespace impossible_arrangement_of_1_and_neg1_in_300_by_300_table_l54_54311

theorem impossible_arrangement_of_1_and_neg1_in_300_by_300_table :
  ¬∃ (table : ℕ → ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ 300 ∧ 1 ≤ j ∧ j ≤ 300 → table i j = 1 ∨ table i j = -1) ∧
    abs (∑ i in finset.range 300, ∑ j in finset.range 300, table i j) < 30000 ∧
    ∀ i j, (1 ≤ i ∧ i ≤ 298) ∧ (1 ≤ j ∧ j ≤ 295) →
      abs (∑ di in finset.range 3, ∑ dj in finset.range 5, table (i + di) (j + dj)) > 3 ∧
    ∀ i j, (1 ≤ i ∧ i ≤ 296) ∧ (1 ≤ j ∧ j ≤ 298) →
      abs (∑ di in finset.range 5, ∑ dj in finset.range 3, table (i + di) (j + dj)) > 3 := sorry

end impossible_arrangement_of_1_and_neg1_in_300_by_300_table_l54_54311


namespace solve_ellipse_problem_l54_54600

open_locale real

-- Problem statement and necessary conditions
variables (origin : ℝ×ℝ := (0, 0)) -- The center of ellipse C1 is at the origin.
def hyperbola_C2 : ℝ → ℝ → Prop := λ x y, x^2 / 2 - y^2 = 1 -- The hyperbola C2
def line_intersects_ellipse (x y : ℝ) : Prop := x + sqrt 2 * y = 0 -- The line intersects ellipse C1 at points (A and B).
def point_A : ℝ×ℝ := (-sqrt 2, 1) -- Point A

-- Define the condition that point Q satisfies for the trajectory
def satisfied_condition (A P Q : ℝ×ℝ) : Prop :=
  let APx := (P.1 + sqrt 2) - A.1 in
  let APy := (P.2 - 1) - A.2 in
  let AQx := (Q.1 + sqrt 2) - A.1 in
  let AQy := (Q.2 - 1) - A.2 in
  let BPx := (P.1 - sqrt 2) - A.1 in
  let BPy := (P.2 + 1) - A.2 in
  let BQx := (Q.1 - sqrt 2) - A.1 in
  let BQy := (Q.2 + 1) - A.2 in
  APx * AQx + APy * AQy = 0 ∧ BPx * BQx + BPy * BQy = 0

-- Colliniarity of points A, B, Q
def not_collinear (A B Q : ℝ×ℝ) : Prop :=
  A.1 * (B.2 - Q.2) + B.1 * (Q.2 - A.2) + Q.1 * (A.2 - B.2) ≠ 0

-- Points A, B
def B := (sqrt 2, -1)
variables {A B}
variables {Q : Type*} [inhabited Q] (Q_coords : Q → ℝ × ℝ)

noncomputable def find_ellipse_equation : Prop :=
  ∃ (a b : ℝ), a = 2 ∧ b = sqrt 2 ∧ ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

noncomputable def find_trajectory : Prop :=
  ∀ (x y : ℝ), 2 * x^2 + y^2 = 5 ∧ (x, y) ≠ (sqrt 2, -1) ∧ (x, y) ≠ (sqrt 2 / 2, -2)

noncomputable def max_area_triangle : Prop :=
  let area := (B.1 - A.1) * (Q_coords Q).2 - (B.2 - A.2) * (Q_coords Q).1 in
  ∃ (Qx Qy : ℝ), (Q_coords Q) = (Qx, Qy) ∧ abs(area / 2) = 5 * sqrt 2 / 2

-- The lean statement tying everything together
theorem solve_ellipse_problem :
  (find_ellipse_equation ∧ find_trajectory ∧ max_area_triangle) :=
by sorry

end solve_ellipse_problem_l54_54600


namespace age_difference_of_siblings_l54_54400

theorem age_difference_of_siblings (x : ℝ) 
  (h1 : 19 * x + 20 = 230) :
  |4 * x - 3 * x| = 210 / 19 := by
    sorry

end age_difference_of_siblings_l54_54400


namespace correct_statements_l54_54489

noncomputable theory

def f (x : ℝ) : ℝ := sorry
def f_inv (x : ℝ) : ℝ := sorry -- Assume f has an inverse function

lemma graph_symmetric_1 : ∀ (x : ℝ), 
  (∃ y : ℝ, y = f x ∧ y ∈ range f) ↔ (∃ y : ℝ, x = f_inv y ∧ y ∈ range f_inv) := sorry

lemma graph_symmetric_2 : ∀ (x : ℝ), 
  (∃ y : ℝ, y = f x ∧ y ∈ range f) ↔ (∃ y : ℝ, x = f y ∧ y ∈ range f) := sorry

lemma graph_not_symmetric_3 : ¬ (∀ (x : ℝ), 
  (∃ y : ℝ, y = f x ∧ y ∈ range f) ↔ (∃ y : ℝ, x = f_inv y ∧ y ∈ range f)) := sorry

lemma graph_same_curve_4 : ∀ (x : ℝ), 
  (∃ y : ℝ, y = f x ∧ y ∈ range f) = (∃ y : ℝ, x = f_inv y ∧ y ∈ range f_inv) := sorry

theorem correct_statements : {1, 2, 4} := by
  split,
  apply graph_symmetric_1,
  apply graph_symmetric_2,
  apply_not graph_not_symmetric_3,
  apply graph_same_curve_4

end correct_statements_l54_54489


namespace domain_g_l54_54515

def g (x : ℝ) : ℝ := Real.sin (Real.arccos x)

theorem domain_g : ∀ x, -1 ≤ x ∧ x ≤ 1 → ∃ y, g x = y :=
by
  assume x,
  assume h : -1 ≤ x ∧ x ≤ 1,
  sorry

end domain_g_l54_54515


namespace area_of_triangle_l54_54878

theorem area_of_triangle :
  let A := (10, 1)
  let B := (15, 8)
  let C := (10, 8)
  ∃ (area : ℝ), 
  area = 17.5 ∧ 
  area = 1 / 2 * (abs (B.1 - C.1)) * (abs (C.2 - A.2)) :=
by
  sorry

end area_of_triangle_l54_54878


namespace num_triangles_in_circle_l54_54415

noncomputable def num_triangles (n : ℕ) : ℕ :=
  n.choose 3

theorem num_triangles_in_circle (n : ℕ) :
  num_triangles n = n.choose 3 :=
by
  sorry

end num_triangles_in_circle_l54_54415


namespace find_first_number_l54_54028

theorem find_first_number
  (avg1 : (20 + 40 + 60) / 3 = 40)
  (avg2 : 40 - 4 = (x + 70 + 28) / 3)
  (sum_eq : x + 70 + 28 = 108) :
  x = 10 :=
by
  sorry

end find_first_number_l54_54028


namespace exists_triangle_same_color_l54_54068

-- Define the Color type
inductive Color
| red : Color
| blue : Color

-- Define a function that assigns a color to each point on the plane
def color (point: ℝ × ℝ) : Color := sorry

-- Write the Lean 4 statement for the proof problem
theorem exists_triangle_same_color (color : ℝ × ℝ → Color) :
  ∃ (A B C : ℝ × ℝ), (centroid A B C : ℝ × ℝ) →
  (color A = color B) ∧ (color B = color C) ∧ (color C = color (centroid A B C))
:= sorry

-- Define the centroid function for a given triangle
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

end exists_triangle_same_color_l54_54068


namespace initial_number_of_students_l54_54493

theorem initial_number_of_students (S : ℕ) (h : S + 6 = 37) : S = 31 :=
sorry

end initial_number_of_students_l54_54493


namespace perfect_number_of_form_l54_54538

theorem perfect_number_of_form (n : ℕ) (hp : prime (2^(n+1) - 1)) : 
  let p := 2^(n+1) - 1 in
  let N := 2^n * p in
  ∃ (N : ℕ), N = 2^n * p ∧ (∀ d : ℕ, d ∣ N ∧ d ≠ N → d + N / d = N) := 
sorry

end perfect_number_of_form_l54_54538


namespace eccentricity_of_conic_section_l54_54975

theorem eccentricity_of_conic_section (m : ℝ) (h : m = 4 ∨ m = -4) : 
  (∃ e : ℝ, e = sqrt 3 / 2 ∨ e = sqrt 5) :=
by
  sorry

end eccentricity_of_conic_section_l54_54975


namespace arithmetic_sequence_properties_l54_54264

theorem arithmetic_sequence_properties (a b c : ℝ) (h1 : ∃ d : ℝ, [2, a, b, c, 9] = [2, 2 + d, 2 + 2 * d, 2 + 3 * d, 2 + 4 * d]) : 
  c - a = 7 / 2 := 
by
  -- We assume the proof here
  sorry

end arithmetic_sequence_properties_l54_54264


namespace total_rats_l54_54700

theorem total_rats (Elodie_rats Hunter_rats Kenia_rats : ℕ) 
  (h1 : Elodie_rats = 30) 
  (h2 : Elodie_rats = Hunter_rats + 10)
  (h3 : Kenia_rats = 3 * (Elodie_rats + Hunter_rats)) :
  Elodie_rats + Hunter_rats + Kenia_rats = 200 :=
by
  sorry

end total_rats_l54_54700


namespace a_2n_is_perfect_square_l54_54727

noncomputable def a_n : ℕ → ℕ 
| 0       := 0
| 1       := 1
| 2       := 1
| 3       := 2
| n       := if n > 4 then a_n (n - 1) + a_n (n - 3) + a_n (n - 4) else 0

theorem a_2n_is_perfect_square (n : ℕ) (h : n > 0) : ∃ k : ℕ, a_n (2 * n) = k * k :=
sorry

end a_2n_is_perfect_square_l54_54727


namespace vector_parallel_l54_54626

-- Define the vectors a, b, and c
def a (x : ℝ) : ℝ × ℝ := (x, 1)
def b (x : ℝ) : ℝ × ℝ := (-x, x^2)
def c : ℝ × ℝ := (0, 1)

-- Prove that a + b is parallel to c
theorem vector_parallel (x : ℝ) :
  let ab := (a x).1 + (b x).1, (a x).2 + (b x).2 in
  ∃ k : ℝ, ab = (k * c.1, k * c.2) :=
by 
  sorry

end vector_parallel_l54_54626


namespace no_intersection_points_l54_54514

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 5

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ := -x^2 + 6 * x - 8

-- The statement asserting that the parabolas do not intersect
theorem no_intersection_points :
  ∀ (x y : ℝ), parabola1 x = y → parabola2 x = y → false :=
by
  -- Introducing x and y as elements of the real numbers
  intros x y h1 h2
  
  -- Since this is only the statement, we use sorry to skip the actual proof
  sorry

end no_intersection_points_l54_54514


namespace christina_rearrangements_l54_54916

-- define the main conditions
def rearrangements (n : Nat) : Nat := Nat.factorial n

def half (n : Nat) : Nat := n / 2

def time_for_first_half (r : Nat) : Nat := r / 12

def time_for_second_half (r : Nat) : Nat := r / 18

def total_time_in_minutes (t1 t2 : Nat) : Nat := t1 + t2

def total_time_in_hours (t : Nat) : Nat := t / 60

-- statement proving that the total time will be 420 hours
theorem christina_rearrangements : 
  rearrangements 9 = 362880 →
  half (rearrangements 9) = 181440 →
  time_for_first_half 181440 = 15120 →
  time_for_second_half 181440 = 10080 →
  total_time_in_minutes 15120 10080 = 25200 →
  total_time_in_hours 25200 = 420 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end christina_rearrangements_l54_54916


namespace log_equality_l54_54198

theorem log_equality {x : ℝ} (h : log 8 (x + 8) = 7 / 3) : x = 120 :=
sorry

end log_equality_l54_54198


namespace min_value_of_a_l54_54953

theorem min_value_of_a (a : ℝ) : 
  (a > 0) ∧ ∀ x : ℝ, x ∈ set.Ici (1 / Real.exp 1) →
    8 * Real.exp (a * x) * (Real.log 2 + (a * x) / 2) - x * Real.log x ≥ 0
  → a ≥ 1 / (4 * Real.exp 1) :=
sorry

end min_value_of_a_l54_54953


namespace measure_DAB_l54_54657

-- Define a regular hexagon and its properties
structure RegularHexagon :=
  (vertices : Fin 6 → Point)
  (interior_angle : ∀ i : Fin 6, angle (vertices i) = 120)

-- Define Point and angle in a way that Lean can understand
-- (Here assuming basic definitions)
-- ...

-- Problem Specification: Given these conditions
variables (hex : RegularHexagon)

-- Define the diagonal AD
noncomputable def diagonal_AD := segment (hex.vertices 0) (hex.vertices 3)

-- The main statement about angle measure
theorem measure_DAB (hex : RegularHexagon) :
  angle (diagonal_AD hex) (segment (hex.vertices 0) (hex.vertices 1)) = 30 := 
sorry

end measure_DAB_l54_54657


namespace integer_part_of_sum_is_2_l54_54402

noncomputable def x : ℕ → ℚ
| 0     := 1 / 3
| (n+1) := x n ^ 2 + x n

theorem integer_part_of_sum_is_2 :
  ⌊∑ i in Finset.range 2015, 1 / (1 + x i)⌋ = 2 :=
sorry

end integer_part_of_sum_is_2_l54_54402


namespace n_sided_polygonal_chain_exists_l54_54653

theorem n_sided_polygonal_chain_exists 
  (n : ℕ) 
  (lines : fin n → set (set (ℝ × ℝ)))
  (h1 : ∀ i j, i ≠ j → ¬ (lines i ∥ lines j)) 
  (h2 : ∀ i j k, set.intersect (set.intersect (lines i) (lines j)) (lines k) = ∅) :
  ∃ polygon_chain : fin (n + 1) → (ℝ × ℝ), 
    (∀ (i : fin n), (line_segment (polygon_chain i) (polygon_chain (i + 1)) ⊆ lines i)) ∧ 
    ¬(∃ i j, i ≠ j ∧ polygon_chain i = polygon_chain j) := 
sorry

end n_sided_polygonal_chain_exists_l54_54653


namespace dividend_percentage_l54_54472

theorem dividend_percentage
    (cost_price_market_value : ℝ)
    (desired_interest_percentage : ℝ)
    (market_value : ℝ)
    (desired_interest_percentage_eq : desired_interest_percentage = 12)
    (cost_price_market_value_eq : cost_price_market_value = 60)
    (market_value_eq : market_value = 45) :
  let interest_per_share := (market_value / 100) * desired_interest_percentage in
  let dividend_percentage := (interest_per_share / cost_price_market_value) * 100 in
  dividend_percentage = 9 := 
by
    sorry

end dividend_percentage_l54_54472


namespace consecutive_numbers_less_than_100_l54_54635

def is_consecutive_number (n : ℕ) : Prop :=
  let f := fun x => List.reverse (x.digits 10) in
  f (n + (n + 1) + (n + 2)) = f n ++ f (n + 1) ++ f (n + 2)

def count_consecutive_numbers (limit : ℕ) : ℕ :=
  (List.range limit).countp is_consecutive_number

theorem consecutive_numbers_less_than_100 : count_consecutive_numbers 100 = 24 :=
  sorry

end consecutive_numbers_less_than_100_l54_54635


namespace deepak_investment_l54_54891

theorem deepak_investment (D : ℝ) (A : ℝ) (P : ℝ) (Dp : ℝ) (Ap : ℝ) 
  (hA : A = 22500)
  (hP : P = 13800)
  (hDp : Dp = 5400)
  (h_ratio : Dp / P = D / (A + D)) :
  D = 15000 := by
  sorry

end deepak_investment_l54_54891


namespace number_of_possible_values_r_l54_54043

open Polynomial

def polynomial := x^4 - 5020*x^3 + p*x^2 + q*x + r

variables {a b u v w x : ℕ}

/-- There exists a polynomial with integer coefficients and four distinct positive zeros -
    Exactly two of these zeros are integers, each being the sum of the other two zeros (pairwise). -/
theorem number_of_possible_values_r :
  (∃ P : polynomial ℕ, (P = x^4 - 5020*x^3 + p*x^2 + q*x + r) ∧
    (a = u + v) ∧ (b = w + x) ∧
    (distinct {a, b, u, v, w, x}) ∧
    (a + b = 2510)) →
    (∃ r, r = 1255^2 * (1255 - u) * u ∧ 1 ≤ u * (1255 - u) ∧ u * (1255 - u) ≤ 1575024) →
    Σ n, n = 1575024 :=
by
  sorry

end number_of_possible_values_r_l54_54043


namespace exists_increasing_sequence_l54_54012

theorem exists_increasing_sequence (M : ℕ) (hM : M > 2) : 
  ∃ (a : ℕ → ℕ), 
  (∀ i, a i > M^i) ∧
  (∀ n : ℤ, n ≠ 0 → ∃ (m : ℕ) (b : Fin m.succ → {-1, 1}), n = ∑ i in Finset.range m.succ, a i * b i) :=
sorry

end exists_increasing_sequence_l54_54012


namespace divides_sum_of_powers_l54_54539

theorem divides_sum_of_powers (n : ℕ) (h : 1 < n) : 
  (n % 2 = 1 ↔ ↑n ∣ (∑ k in Finset.range (n - 1) | (k + 1) ^ n)) :=
begin
  sorry
end

end divides_sum_of_powers_l54_54539


namespace part_a_part_b_l54_54887

-- Assume given points A, B, C, D making an acute-angled tetrahedron ABCD
variables (A B C D : Point)
(hacute : ∀ (X Y Z : Point), triangle ABC ∧ triangle ABD ∧ triangle ACD ∧ triangle BCD → 
  ∀ (angle : Angle), acute angle)

-- Points in the interior segments
variables (X Y Z T : Point)
(hX : X ∈ segment AB)
(hY : Y ∈ segment BC)
(hZ : Z ∈ segment CD)
(hT : T ∈ segment AD)

-- Angle condition for part (a)
variables (hangle_ineq : ∠ DAB + ∠ BCD ≠ ∠ CDA + ∠ ABC)

-- Theorem statement for part (a)
theorem part_a (hX : X ∈ segment AB) 
  (hY : Y ∈ segment BC) 
  (hZ : Z ∈ segment CD) 
  (hT : T ∈ segment AD) : 
  ¬ minimal_length_closed_path X Y Z T :=
sorry

-- Angle condition for part (b)
variables (hangle_eq : ∠ DAB + ∠ BCD = ∠ CDA + ∠ ABC)

-- Theorem statement for part (b)
theorem part_b (hX : X ∈ segment AB) 
  (hY : Y ∈ segment BC) 
  (hZ : Z ∈ segment CD) 
  (hT : T ∈ segment AD) 
  (k : ℝ) 
  (hangle_sum : 2 * k = ∠ BAC + ∠ CAD + ∠ DAB) : 
  (exists AC : Real, infinite_shortest_paths X Y Z T (2 * AC * sin k)) :=
sorry

end part_a_part_b_l54_54887


namespace period_tan_half_l54_54431

noncomputable def period_of_tan_half : Real :=
  2 * Real.pi

theorem period_tan_half (f : Real → Real) (h : ∀ x, f x = Real.tan (x / 2)) :
  ∀ x, f (x + period_of_tan_half) = f x := 
by 
  sorry

end period_tan_half_l54_54431


namespace shyam_weight_increase_l54_54069

theorem shyam_weight_increase 
  (x : ℝ)
  (Ram_weight : ℝ := 6 * x)
  (Shyam_weight : ℝ := 5 * x)
  (Geeta_weight : ℝ := 7 * x)
  (Ram_weight_new : ℝ := Ram_weight * 1.10)
  (Geeta_weight_new : ℝ := Geeta_weight * 1.20)
  (Total_weight_new : ℝ := 105.4)
  (Total_weight_original : ℝ := Total_weight_new / 1.15) 
  (x_value : ℝ := Total_weight_original / 18) :
  let Shyam_weight_new := Shyam_weight * (1 + 14.5 / 100) in
  Ram_weight_new + Shyam_weight_new + Geeta_weight_new = Total_weight_new :=
by
  sorry

end shyam_weight_increase_l54_54069


namespace value_of_a_l54_54269

theorem value_of_a (x a : ℂ) (h1 : a = x^4 + x^(-4)) (h2 : x^2 + x + 1 = 0) : a = -1 := 
by
  sorry

end value_of_a_l54_54269


namespace probability_xi_eq_2_l54_54227

noncomputable def p_of_xi_eq_2 (ξ : ℕ → ℕ → ℝ) (n : ℕ) (p : ℝ) : ℝ :=
  if h : 0 ≤ p ∧ p ≤ 1 then
    (nat.binomial n 2 : ℝ) * (p ^ 2) * ((1 - p) ^ (n - 2))
  else 0

theorem probability_xi_eq_2 {n : ℕ} {p : ℝ}
  (h1 : n * p = 4)
  (h2 : 4 * n * p * (1 - p) = 3.2) :
  p_of_xi_eq_2 binomial n p = 32 / 625 :=
sorry

end probability_xi_eq_2_l54_54227


namespace volume_of_bounded_figure_l54_54152

-- Define the volume of a cube with edge length 1
def volume_of_cube (a : ℝ) : ℝ := a^3

-- Define the edge length of the smaller cubes
def small_cube_edge_length (a : ℝ) : ℝ := a / 2

-- Define the volume of a small cube
def volume_of_small_cube (a : ℝ) : ℝ := volume_of_cube (small_cube_edge_length a)

-- Theorem: Proving the volume of the bounded figure
theorem volume_of_bounded_figure (a : ℝ) : volume_of_cube a = 1 → 
  let V := volume_of_small_cube a in 8 * (V / 2) = 1 / 2 :=
begin
  sorry
end

end volume_of_bounded_figure_l54_54152


namespace stratified_sampling_third_year_l54_54112

theorem stratified_sampling_third_year :
  ∀ (total students_first_year students_second_year sample_size students_third_year sampled_students : ℕ),
  (total = 900) →
  (students_first_year = 240) →
  (students_second_year = 260) →
  (sample_size = 45) →
  (students_third_year = total - students_first_year - students_second_year) →
  (sampled_students = sample_size * students_third_year / total) →
  sampled_students = 20 :=
by
  intros
  sorry

end stratified_sampling_third_year_l54_54112


namespace xiao_hu_ma_speed_l54_54830

theorem xiao_hu_ma_speed (distance_to_school : ℕ) (distance_father_caught_up : ℕ) (time_difference : ℕ) (father_speed_multiplier : ℕ)
  (h1 : distance_to_school = 1800)
  (h2 : distance_father_caught_up = 200)
  (h3 : time_difference = 10)
  (h4 : father_speed_multiplier = 2) :
  ∃ (x : ℕ), x = 80 :=
by
  use 80
  -- Proof would go here, but it is omitted for brevity.
  sorry

end xiao_hu_ma_speed_l54_54830


namespace gino_popsicle_sticks_l54_54949

variable (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ)

def popsicle_sticks_condition (my_sticks : ℕ) (total_sticks : ℕ) (gino_sticks : ℕ) : Prop :=
  my_sticks = 50 ∧ total_sticks = 113

theorem gino_popsicle_sticks
  (h : popsicle_sticks_condition my_sticks total_sticks gino_sticks) :
  gino_sticks = 63 :=
  sorry

end gino_popsicle_sticks_l54_54949


namespace constant_term_when_coef_of_fourth_term_largest_l54_54221

def binomial_expansion (n : ℕ) (x : ℝ) :=
  (x - 1 / sqrt x) ^ n

theorem constant_term_when_coef_of_fourth_term_largest :
  ∀ (x : ℝ), nat.choose 6 4 = 15 := by
  sorry

end constant_term_when_coef_of_fourth_term_largest_l54_54221


namespace solve_system_l54_54371

def eq1 (x y : ℝ) := 2 * x + real.sqrt (2 * x + 3 * y) - 3 * y = 5
def eq2 (x y : ℝ) := 4 * x ^ 2 + 2 * x + 3 * y - 9 * y ^ 2 = 32

theorem solve_system : eq1 (17 / 4) (5 / 2) ∧ eq2 (17 / 4) (5 / 2) := by
  sorry

end solve_system_l54_54371


namespace pushups_percentage_l54_54526

def total_exercises : ℕ := 12 + 8 + 20

def percentage_pushups (total_ex: ℕ) : ℕ := (8 * 100) / total_ex

theorem pushups_percentage (h : total_exercises = 40) : percentage_pushups total_exercises = 20 :=
by
  sorry

end pushups_percentage_l54_54526


namespace find_three_digit_numbers_l54_54194

def is_geometric_progression (x y z : ℕ) : Prop :=
  y^2 = x * z

theorem find_three_digit_numbers :
  ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 →
  let x := 100 * a + 10 * b + c in
  let y := 100 * b + 10 * c + a in
  let z := 100 * c + 10 * a + b in
  is_geometric_progression x y z →
  (a = b ∧ b = c ∨ (a = 2 ∧ b = 4 ∧ c = 3) ∨ (a = 4 ∧ b = 8 ∧ c = 6)) :=
by
  intros a b c h1 x y z hgp.
  let x := 100 * a + 10 * b + c,
  let y := 100 * b + 10 * c + a,
  let z := 100 * c + 10 * a + b,
  sorry

end find_three_digit_numbers_l54_54194


namespace impossible_arrangement_of_1_and_neg1_in_300_by_300_table_l54_54312

theorem impossible_arrangement_of_1_and_neg1_in_300_by_300_table :
  ¬∃ (table : ℕ → ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ 300 ∧ 1 ≤ j ∧ j ≤ 300 → table i j = 1 ∨ table i j = -1) ∧
    abs (∑ i in finset.range 300, ∑ j in finset.range 300, table i j) < 30000 ∧
    ∀ i j, (1 ≤ i ∧ i ≤ 298) ∧ (1 ≤ j ∧ j ≤ 295) →
      abs (∑ di in finset.range 3, ∑ dj in finset.range 5, table (i + di) (j + dj)) > 3 ∧
    ∀ i j, (1 ≤ i ∧ i ≤ 296) ∧ (1 ≤ j ∧ j ≤ 298) →
      abs (∑ di in finset.range 5, ∑ dj in finset.range 3, table (i + di) (j + dj)) > 3 := sorry

end impossible_arrangement_of_1_and_neg1_in_300_by_300_table_l54_54312


namespace count_valid_numbers_l54_54263

-- Define the condition for a 3-digit number whose product of digits equals 30
def is_valid_number (n : ℕ) : Prop :=
  floor (n / 100) ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  floor ((n % 100) / 10) ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (n % 10) ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (n / 100) * floor (((n % 100) / 10).toNat) * (n % 10) = 30

-- Prove the number of 3-digit numbers satisfying the condition equals 12
theorem count_valid_numbers : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ is_valid_number n}.card = 12 := 
by
  sorry

end count_valid_numbers_l54_54263


namespace two_beta_plus_alpha_eq_pi_div_two_l54_54218

theorem two_beta_plus_alpha_eq_pi_div_two
  (α β : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2)
  (hβ1 : 0 < β) (hβ2 : β < π / 2)
  (h : Real.tan α + Real.tan β = 1 / Real.cos α) :
  2 * β + α = π / 2 :=
sorry

end two_beta_plus_alpha_eq_pi_div_two_l54_54218


namespace capital_letter_axisymmetric_l54_54076

-- Define the property of being an axisymmetric figure
def is_axisymmetric (L : Char) : Prop :=
  L = 'A' ∨ L = 'B' ∨ L = 'D' ∨ L = 'E'

-- Statement of the problem in Lean 4
theorem capital_letter_axisymmetric (L : Char) (h : L = 'A' ∨ L = 'B' ∨ L = 'D' ∨ L = 'E') : 
  is_axisymmetric L :=
by
  exact h
sorry

end capital_letter_axisymmetric_l54_54076


namespace set_union_example_l54_54234

open Set

theorem set_union_example :
  let A := ({1, 3, 5, 6} : Set ℤ)
  let B := ({-1, 5, 7} : Set ℤ)
  A ∪ B = ({-1, 1, 3, 5, 6, 7} : Set ℤ) :=
by
  intros
  sorry

end set_union_example_l54_54234


namespace charles_draws_yesterday_after_work_l54_54912

theorem charles_draws_yesterday_after_work :
  ∀ (initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work : ℕ),
    initial_papers = 20 →
    today_drawn = 6 →
    yesterday_drawn_before_work = 6 →
    current_papers_left = 2 →
    (initial_papers - (today_drawn + yesterday_drawn_before_work) - yesterday_drawn_after_work = current_papers_left) →
    yesterday_drawn_after_work = 6 :=
by
  intros initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work
  intro h1 h2 h3 h4 h5
  sorry

end charles_draws_yesterday_after_work_l54_54912


namespace limit_proof_main_theorem_l54_54358

theorem limit_proof (f : ℝ → ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + (1 / 2)| ∧ |x + (1 / 2)| < δ → |f x + 5| < ε) :=
by
  assume ε ε_pos,
  let δ := ε / 6,
  use δ,
  sorry

-- Specific function as given in the problem
noncomputable def f (x : ℝ) : ℝ := (6 * x^2 + x - 1) / (x + (1 / 2))

theorem main_theorem :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x + (1 / 2)| ∧ |x + (1 / 2)| < δ → |f x + 5| < ε :=
by
  assume ε ε_pos,
  -- Here we state the delta value as derived
  let δ := ε / 6,
  use δ,
  -- Provide proof outline
  sorry

end limit_proof_main_theorem_l54_54358


namespace x_plus_y_bound_l54_54337

-- Definitions based on given conditions
variable {x y : ℝ}
variable [x_floor_equality : y = 4 * (⌊x⌋ : ℝ) + 1]
variable [x_minus_1_floor_equality : y = 2 * (⌊x - 1⌋ : ℝ) + 7]
variable [x_not_int : x ∉ ℤ]

-- Problem Statement
theorem x_plus_y_bound : 11 < x + y ∧ x + y < 12 := 
begin
  sorry
end

end x_plus_y_bound_l54_54337


namespace dislike_both_tv_and_books_l54_54748

theorem dislike_both_tv_and_books (n : ℕ) (p_tv_dislike : ℚ) (p_both_dislike : ℚ) (total_people : ℕ) (h_n : n = total_people)
  (h_p_tv_dislike : p_tv_dislike = 0.4) (h_p_both_dislike : p_both_dislike = 0.15)
  (h_total_people : total_people = 1500) : ∃ k : ℕ, k = 90 :=
by
  have tv_dislike := p_tv_dislike * total_people,
  have both_dislike := p_both_dislike * tv_dislike,
  use both_dislike.to_nat,
  sorry

end dislike_both_tv_and_books_l54_54748


namespace sara_gave_dan_limes_l54_54508

theorem sara_gave_dan_limes (initial_limes : ℕ) (final_limes : ℕ) (d : ℕ) 
  (h1: initial_limes = 9) (h2: final_limes = 13) (h3: final_limes = initial_limes + d) : d = 4 := 
by sorry

end sara_gave_dan_limes_l54_54508


namespace cost_per_square_meter_of_mat_l54_54652

theorem cost_per_square_meter_of_mat {L W E : ℝ} : 
  L = 20 → W = 15 → E = 57000 → (E / (L * W)) = 190 :=
by
  intros hL hW hE
  rw [hL, hW, hE]
  sorry

end cost_per_square_meter_of_mat_l54_54652


namespace remainder_of_2_pow_2005_mod_7_l54_54072

theorem remainder_of_2_pow_2005_mod_7 :
  2 ^ 2005 % 7 = 2 :=
sorry

end remainder_of_2_pow_2005_mod_7_l54_54072


namespace vasya_grades_l54_54679

theorem vasya_grades :
  ∃ (grades : List ℕ), 
    grades.length = 5 ∧ 
    (grades.nthLe 2 (by linarith) = 4) ∧ 
    (grades.sum = 19) ∧
    (grades.count 5 > List.foldl (fun acc n => if n ≠ 5 then acc + grades.count n else acc) 0 grades) ∧ 
    (grades.sorted = [2, 3, 4, 5, 5]) :=
sorry

end vasya_grades_l54_54679


namespace range_of_a_l54_54556

noncomputable def inequality_condition (x : ℝ) (a : ℝ) : Prop :=
  a - 2 * x - |Real.log x| ≤ 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → inequality_condition x a) ↔ a ≤ 1 + Real.log 2 :=
sorry

end range_of_a_l54_54556


namespace evaluate_expression_l54_54532

theorem evaluate_expression (x y : ℝ) (P Q : ℝ) 
  (hP : P = x^2 + y^2) 
  (hQ : Q = x - y) : 
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = 4 * x * y / ((x^2 + y^2)^2 - (x - y)^2) :=
by 
  -- Insert proof here
  sorry

end evaluate_expression_l54_54532


namespace min_range_of_seven_observations_l54_54869

theorem min_range_of_seven_observations (x : Fin 7 → ℝ)
  (h_mean : (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6) / 7 = 15)
  (h_median : ∀ (i : Fin 7), x (Fin.mk 3 (by decide)) = 17) :
  ∃ r, r = (Finset.max' (Finset.univ.image x) (by decide) - Finset.min' (Finset.univ.image x) (by decide)) ∧ r = 11 := by
  sorry

end min_range_of_seven_observations_l54_54869


namespace total_votes_cast_l54_54654

theorem total_votes_cast (b_votes c_votes total_votes : ℕ)
  (h1 : b_votes = 48)
  (h2 : c_votes = 35)
  (h3 : b_votes = (4 * total_votes) / 15) :
  total_votes = 180 :=
by
  sorry

end total_votes_cast_l54_54654


namespace evaluate_f_at_3_l54_54266

variable {α : Type*} [Field α]

def f (a b x : α) : α := a * x^2 + b * x + 2

theorem evaluate_f_at_3 (a b : α) (h1 : f a b 1 = 7) (h2 : f a b 2 = 14) : f a b 3 = 23 := by
  have h_eq1 : a + b = 5 := by
    -- From h1: a * 1^2 + b * 1 + 2 = 7
    rw [f] at h1
    simp at h1
    linarith

  have h_eq2 : 2 * a + b = 6 := by
    -- From h2: a * 2^2 + b * 2 + 2 = 14
    rw [f] at h2
    simp at h2
    linarith

  -- Solve the system of linear equations a + b = 5 and 2 * a + b = 6
  have ha : a = 1 := by
    -- 2 * a + b = 6 => b = 6 - 2 * a
    -- a + b = 5 
    -- a + (6 - 2 * a) = 5 => -a + 6 = 5 => a = 1
    linarith [h_eq1, h_eq2]
    
  have hb : b = 4 := by
    -- b = 5 - a = 5 - 1 = 4
    linarith [ha, h_eq1]

  rw [ha, hb]
  -- Finally, f 1 4 3 => 1 * 9 + 4 * 3 + 2 = 23
  rw [f]
  simp
  linarith

end evaluate_f_at_3_l54_54266


namespace Esha_behind_Anusha_l54_54893

/-- Define conditions for the race -/

def Anusha_speed := 100
def Banu_behind_when_Anusha_finishes := 10
def Banu_run_when_Anusha_finishes := Anusha_speed - Banu_behind_when_Anusha_finishes
def Esha_behind_when_Banu_finishes := 10
def Esha_run_when_Banu_finishes := Anusha_speed - Esha_behind_when_Banu_finishes
def Banu_speed_ratio := Banu_run_when_Anusha_finishes / Anusha_speed
def Esha_speed_ratio := Esha_run_when_Banu_finishes / Anusha_speed
def Esha_to_Anusha_speed_ratio := Esha_speed_ratio * Banu_speed_ratio
def Esha_run_when_Anusha_finishes := Anusha_speed * Esha_to_Anusha_speed_ratio

/-- Prove that Esha is 19 meters behind Anusha when Anusha finishes the race -/
theorem Esha_behind_Anusha {V_A V_B V_E : ℝ} :
  (V_B / V_A = 9 / 10) →
  (V_E / V_B = 9 / 10) →
  (Esha_run_when_Anusha_finishes = Anusha_speed * (9 / 10 * 9 / 10)) →
  Anusha_speed - Esha_run_when_Anusha_finishes = 19 := 
by
  intros h1 h2 h3
  sorry

end Esha_behind_Anusha_l54_54893


namespace greatest_possible_y_l54_54022

theorem greatest_possible_y
  (x y : ℤ)
  (h : x * y + 7 * x + 6 * y = -14) : y ≤ 21 :=
sorry

end greatest_possible_y_l54_54022


namespace teamB_can_serve_at_least_half_l54_54868

-- Define the height requirement for serving on the submarine
def max_height := 168

-- Define the teams with their respective height statistics
def teamA_average := 166
def teamB_median := 167
def teamC_tallest := 169
def teamD_mode := 167

-- Formally state the problem in Lean 4
theorem teamB_can_serve_at_least_half :
  ∀ heights : List ℕ, 
  (∀ h, h ∈ heights → h ≤ max_height → HalfOrMoreCanServe heights) →

  -- Team A: Average height condition
  List.average heights = teamA_average →
  ¬(at_least_half_can_serve heights) →

  -- Team B: Median height condition
  List.median heights = teamB_median →
  at_least_half_can_serve heights →

  -- Team C: Tallest sailor condition
  List.maximum heights = teamC_tallest →
  ¬(at_least_half_can_serve heights) →

  -- Team D: Mode height condition
  List.mode heights = teamD_mode →
  ¬(at_least_half_can_serve heights) →

  -- Conclusion: At least half the sailors in team B can serve
  true
:= sorry

end teamB_can_serve_at_least_half_l54_54868


namespace find_incorrect_value_l54_54551

variable (k b : ℝ)

-- Linear function definition
def linear_function (x : ℝ) : ℝ := k * x + b

-- Given points
theorem find_incorrect_value (h₁ : linear_function k b (-1) = 3)
                             (h₂ : linear_function k b 0 = 2)
                             (h₃ : linear_function k b 1 = 1)
                             (h₄ : linear_function k b 2 = 0)
                             (h₅ : linear_function k b 3 = -2) :
                             (∃ x y, linear_function k b x ≠ y) := by
  sorry

end find_incorrect_value_l54_54551


namespace gas_cost_per_gallon_l54_54501

-- Definitions for the conditions
def one_way_commute := 21 -- miles
def car_efficiency := 30 -- miles per gallon
def days_per_week := 5
def weeks_per_month := 4
def monthly_payment_per_person := 14 -- dollars
def number_of_friends := 5

-- The theorem stating that the cost of gas per gallon given the conditions
theorem gas_cost_per_gallon : 
  (monthly_payment_per_person * number_of_friends) / 
  ((one_way_commute * 2 * days_per_week * weeks_per_month) / car_efficiency) = 2.50 :=
by sorry

end gas_cost_per_gallon_l54_54501


namespace number_of_rabbits_l54_54108

theorem number_of_rabbits (C D : ℕ) (hC : C = 49) (hD : D = 37) (h : D + R = C + 9) :
  R = 21 :=
by
    sorry

end number_of_rabbits_l54_54108


namespace chris_wins_l54_54502

noncomputable def chris_heads : ℚ := 1 / 4
noncomputable def drew_heads : ℚ := 1 / 3
noncomputable def both_tails : ℚ := (1 - chris_heads) * (1 - drew_heads)

/-- The probability that Chris wins comparing with relatively prime -/
theorem chris_wins (p q : ℕ) (hpq : Nat.Coprime p q) (hq0 : q ≠ 0) :
  (chris_heads * (1 + both_tails)) = (p : ℚ) / q ∧ (q - p = 1) :=
sorry

end chris_wins_l54_54502


namespace grocer_sales_l54_54109

theorem grocer_sales :
  let first_month := 8435
  let second_month := 8927
  let fourth_month := 9230
  let fifth_month := 8562
  let sixth_month := 6991
  let average_sale := 8500
  let total_sales := average_sale * 6
  let known_sales := first_month + second_month + fourth_month + fifth_month + sixth_month
  total_sales - known_sales = 8855 :=
by
  let first_month := 8435
  let second_month := 8927
  let fourth_month := 9230
  let fifth_month := 8562
  let sixth_month := 6991
  let average_sale := 8500
  let total_sales := average_sale * 6
  let known_sales := first_month + second_month + fourth_month + fifth_month + sixth_month
  show total_sales - known_sales = 8855 by exact sorry

end grocer_sales_l54_54109


namespace sequence_sum_eq_l54_54686

-- Define the sequence {a_n}
def seq (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3 * seq (n - 1)

-- Define the sum of the first n terms of the sequence {a_n}
def sum_seq (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, seq (i + 1))

-- Theorem statement
theorem sequence_sum_eq (n : ℕ) : sum_seq n = 3^n - 1 :=
sorry

end sequence_sum_eq_l54_54686


namespace angle_sum_cyclic_quad_l54_54101

theorem angle_sum_cyclic_quad (A B C D O : Type) 
  (h_circumscribed : ∃ (circle : Circle), circle ∈ A ∧ circle ∈ B ∧ circle ∈ C ∧ circle ∈ D)
  (angle_ACB : ∠ A C B = 50)
  (angle_CAD : ∠ C A D = 40) :
  ∠ B A C + ∠ A D C = 90 :=
by
  sorry

end angle_sum_cyclic_quad_l54_54101


namespace union_M_N_l54_54622

namespace MyMath

def M : Set ℝ := {x | x^2 = 1}
def N : Set ℝ := {1, 2}

theorem union_M_N : M ∪ N = {-1, 1, 2} := sorry

end MyMath

end union_M_N_l54_54622


namespace max_f_value_l54_54390

def f (x : ℝ) : ℝ := cos (2 * x) + 6 * cos (π / 2 - x)

theorem max_f_value : ∃ x : ℝ, f x = 5 :=
by
  sorry

end max_f_value_l54_54390


namespace negation_example_l54_54620

theorem negation_example (p : ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) :
  ∃ x0 : ℝ, x0 > 0 ∧ (x0 + 1) * Real.exp x0 ≤ 1 :=
sorry

end negation_example_l54_54620


namespace area_convex_quadrilateral_l54_54104

theorem area_convex_quadrilateral (x y : ℝ) :
  (x^2 + y^2 = 73 ∧ x * y = 24) →
  -- You can place a formal statement specifying the four vertices here if needed
  ∃ a b c d : ℝ × ℝ,
  a.1^2 + a.2^2 = 73 ∧
  a.1 * a.2 = 24 ∧
  b.1^2 + b.2^2 = 73 ∧
  b.1 * b.2 = 24 ∧
  c.1^2 + c.2^2 = 73 ∧
  c.1 * c.2 = 24 ∧
  d.1^2 + d.2^2 = 73 ∧
  d.1 * d.2 = 24 ∧
  -- Ensure the quadrilateral forms a rectangle (additional conditions here)
  -- Compute the side lengths and area
  -- Specify finally the area and prove it equals 110
  True :=
sorry

end area_convex_quadrilateral_l54_54104


namespace carolyn_sum_is_12_l54_54155

def carolyn_paul_game (n : ℕ) (initial_numbers : Set ℕ) : ℕ :=
  let carolyn_removes := {3, 9} -- Carolyn's moves
  let remaining_after_pauls_turn := initial_numbers \ {1, 2, 4, 5, 6, 7, 8, 10} -- Paul's removals
  carolyn_removes.sum

theorem carolyn_sum_is_12 : carolyn_paul_game 10 ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) = 12 :=
  by
    -- the sum of Carolyn's removed numbers 3 and 9 is 12
    sorry

end carolyn_sum_is_12_l54_54155


namespace complex_equation_solution_l54_54598

theorem complex_equation_solution (x : ℝ) (i : ℂ) (h_imag_unit : i * i = -1) (h_eq : (x + 2 * i) * (x - i) = 6 + 2 * i) : x = 2 :=
by
  sorry

end complex_equation_solution_l54_54598


namespace cos_double_angle_l54_54567

theorem cos_double_angle (k : ℝ) (h : Real.sin (10 * Real.pi / 180) = k) : 
  Real.cos (20 * Real.pi / 180) = 1 - 2 * k^2 := by
  sorry

end cos_double_angle_l54_54567


namespace problem1_problem2_l54_54250

def f (x : ℝ) : ℝ := log 4 (x - 2)

def g (x : ℝ) : ℝ := (2^(2 * x) - 6 * 2^x + 5)

open Set

theorem problem1 (h1 : f 3 = 0) (h2 : f 6 = 1) : 
    f = (λ x, log 4 (x - 2)) := 
by sorry

theorem problem2 : range (λ x, 2^(2 * x) - 6 * 2^x + 5) = Icc (-4 : ℝ) (21 : ℝ) :=
by sorry

end problem1_problem2_l54_54250


namespace b_work_fraction_l54_54444

/-- Definitions of the conditions -/
def combined_work_rate := 1 / 12
def a_work_rate := 1 / 20
def combined_partial_work_rate (x : ℝ) := a_work_rate + x * (1 / 12)

/-- Main theorem stating the problem -/
theorem b_work_fraction :
  ∃ (x : ℝ), combined_partial_work_rate x = 1 / 15 ∧ x = 1 / 5 :=
by {
  sorry -- proof skipped
}

end b_work_fraction_l54_54444


namespace find_f_4_l54_54718

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_4 : f 4 = 5.5 :=
by
  sorry

end find_f_4_l54_54718


namespace total_books_is_120_l54_54902

theorem total_books_is_120 (B : ℕ) (h1 : 0.65 * B = n) (h2 : 18 = g) (h3 : 0.20 * B = c) (h4 : B = n + g + c)
  (h5 : 18 = 0.15 * B) : B = 120 := by
  sorry

end total_books_is_120_l54_54902


namespace total_rats_l54_54703

variable (Kenia Hunter Elodie : ℕ) -- Number of rats each person has

-- Conditions
-- Elodie has 30 rats
axiom h1 : Elodie = 30
-- Elodie has 10 rats more than Hunter
axiom h2 : Elodie = Hunter + 10
-- Kenia has three times as many rats as Hunter and Elodie have together
axiom h3 : Kenia = 3 * (Hunter + Elodie)

-- Prove that the total number of pets the three have together is 200
theorem total_rats : Kenia + Hunter + Elodie = 200 := 
by 
  sorry

end total_rats_l54_54703


namespace angle_XDY_is_45_degrees_l54_54658

theorem angle_XDY_is_45_degrees {A B C D X Y : Type*}
  (hABC: triangle A B C)
  (hRight: ∠BAC = 90)
  (hAX_eq_AD: AX = AD)
  (hCY_eq_CD: CY = CD) :
  ∠XDY = 45 := 
sorry

end angle_XDY_is_45_degrees_l54_54658


namespace part1_part2_l54_54615

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 3 ↔ x ≤ -3 / 2 ∨ x ≥ 3 / 2 := 
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a) ↔ -1 ≤ a ∧ a ≤ 2 :=
  sorry

end part1_part2_l54_54615


namespace correct_ratio_l54_54483

theorem correct_ratio (a b : ℝ) (h : 4 * a = 5 * b) : a / b = 5 / 4 :=
by
  sorry

end correct_ratio_l54_54483


namespace min_T_tetrominoes_l54_54430

-- A chessboard is an 8x8 array.
def Chessboard := Array (Array Bool)

-- A T-tetromino is defined as a set of four cells forming a T shape.
def is_T_tetromino (c : Chessboard) (x y : Nat) : Prop := 
  -- Check positions assuming (x, y) is the center of the T
  -- Simplified representation of various T shapes
  (x < 7 ∧ y < 7 ∧ x > 0 ∧ y > 0) ∧
  ((c[x][y] ∧ c[x][y+1] ∧ c[x-1][y] ∧ c[x+1][y]) ∨ -- T facing up
   (c[x][y] ∧ c[x][y+1] ∧ c[x][y+2] ∧ c[x+1][y+1]) ∨ -- T facing down
   (c[x][y] ∧ c[x-1][y] ∧ c[x+1][y] ∧ c[x][y-1]) ∨ -- T facing left
   (c[x][y] ∧ c[x-1][y+1] ∧ c[x+1][y+1] ∧ c[x][y+1])) -- T facing right

-- Define the problem of placing T-tetrominoes on a chessboard
def count_T_tetrominoes (c : Chessboard) : Nat :=
  -- Count all positions in the chessboard that form T-tetrominoes
  (Array.range 8).sum (λ i => (Array.range 8).sum (λ j => if is_T_tetromino c i j then 1 else 0))

-- The theorem to prove
theorem min_T_tetrominoes : ∀ (b : Chessboard), (count_T_tetrominoes b ≥ 7) → (∃ (b' : Chessboard), count_T_tetrominoes b' = 7) :=
by
  sorry

end min_T_tetrominoes_l54_54430


namespace product_fraction_even_n_l54_54170

theorem product_fraction_even_n (n : ℕ) (hn : n % 2 = 0) (hn_lt_100 : n < 100) :
  (∏ k in Finset.range (n + 1), (1 - 1 / (k + 2))) = 1 / (n + 1) :=
sorry

end product_fraction_even_n_l54_54170


namespace vasya_grades_l54_54664

variables
  (grades : List ℕ)
  (length_grades : grades.length = 5)
  (median_grade : grades.nthLe 2 sorry = 4)  -- Assuming 0-based indexing
  (mean_grade : (grades.sum : ℚ) / 5 = 3.8)
  (most_frequent_A : ∀ n ∈ grades, n ≤ 5)

theorem vasya_grades (h : ∀ x ∈ grades, x ≤ 5 ∧ ∃ k, grades.nthLe 3 sorry = 5 ∧ grades.count 5 > grades.count x):
  ∃ g1 g2 g3 g4 g5 : ℕ, grades = [g1, g2, g3, g4, g5] ∧ g1 ≤ g2 ∧ g2 ≤ g3 ∧ g3 ≤ g4 ∧ g4 ≤ g5 ∧ [g1, g2, g3, g4, g5] = [2, 3, 4, 5, 5] :=
sorry

end vasya_grades_l54_54664


namespace binary_101101_is_45_l54_54091

def binary_to_nat (b : List ℕ) : ℕ :=
  b.foldr (λ (bit acc : ℕ), bit + 2 * acc) 0

theorem binary_101101_is_45 : binary_to_nat [1, 0, 1, 1, 0, 1] = 45 := by
  sorry

end binary_101101_is_45_l54_54091


namespace total_games_played_l54_54058

theorem total_games_played (n : ℕ) (h: n = 12) : ∑ (2 : ℕ) (choose 12 2 = 66 := by {
  sorry
}

end total_games_played_l54_54058


namespace solve_inequality_system_l54_54367

theorem solve_inequality_system (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1)) →
  ((1 / 2) * x - 1 ≤ 7 - (3 / 2) * x) →
  (2 < x ∧ x ≤ 4) :=
by
  intro h1 h2
  sorry

end solve_inequality_system_l54_54367


namespace angle_between_slant_height_and_base_l54_54039

noncomputable def find_angle (R l : ℝ) (h1 : l = 2 * R) : ℝ := 
  let θ := Real.arccos (R / l) in
  θ

theorem angle_between_slant_height_and_base (R l θ : ℝ) 
  (h1 : l = 2 * R) 
  (h2 : θ = find_angle R l h1) :
  θ = Real.pi / 3 := 
begin
  -- h2 is a placeholder for the computed angle
  sorry
end

end angle_between_slant_height_and_base_l54_54039


namespace Cannot_Halve_Triangles_With_Diagonals_l54_54650

structure Polygon where
  vertices : Nat
  edges : Nat

def is_convex (n : Nat) (P : Polygon) : Prop :=
  P.vertices = n ∧ P.edges = n

def non_intersecting_diagonals (P : Polygon) : Prop :=
  -- Assuming a placeholder for the actual non-intersecting diagonals condition
  true

def count_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  P.vertices - 2 -- This is the simplification used for counting triangles

def count_all_diagonals_triangles (P : Polygon) (d : non_intersecting_diagonals P) : Nat :=
  -- Placeholder for function to count triangles formed exclusively by diagonals
  1000

theorem Cannot_Halve_Triangles_With_Diagonals (P : Polygon) (h : is_convex 2002 P) (d : non_intersecting_diagonals P) :
  count_triangles P d = 2000 → ¬ (count_all_diagonals_triangles P d = 1000) :=
by
  intro h1
  sorry

end Cannot_Halve_Triangles_With_Diagonals_l54_54650


namespace least_value_MX_l54_54292

-- Definitions of points and lines
variables (A B C D M P X : ℝ × ℝ)
variables (y : ℝ)

-- Hypotheses based on the conditions
variables (h1 : A = (0, 0))
variables (h2 : B = (33, 0))
variables (h3 : C = (33, 56))
variables (h4 : D = (0, 56))
variables (h5 : M = (33 / 2, 0)) -- M is midpoint of AB
variables (h6 : P = (33, y)) -- P is on BC
variables (hy_range : 0 ≤ y ∧ y ≤ 56) -- y is within the bounds of BC

-- Additional derived hypotheses needed for the proof
variables (h7 : ∃ x, X = (x, sqrt (816.75))) -- X is intersection point on DA

-- The theorem statement
theorem least_value_MX : ∃ y, 0 ≤ y ∧ y ≤ 56 ∧ MX = 33 :=
by
  use 28
  sorry

end least_value_MX_l54_54292


namespace cuboids_painted_l54_54886

-- Let's define the conditions first
def faces_per_cuboid : ℕ := 6
def total_faces_painted : ℕ := 36

-- Now, we state the theorem we want to prove
theorem cuboids_painted (n : ℕ) (h : total_faces_painted = n * faces_per_cuboid) : n = 6 :=
by
  -- Add proof here
  sorry

end cuboids_painted_l54_54886


namespace AgOH_moles_formed_l54_54195

noncomputable def number_of_moles_of_AgOH (n_AgNO3 n_NaOH : ℕ) : ℕ :=
  if n_AgNO3 = n_NaOH then n_AgNO3 else 0

theorem AgOH_moles_formed :
  number_of_moles_of_AgOH 3 3 = 3 := by
  sorry

end AgOH_moles_formed_l54_54195


namespace lemon_heads_distribution_l54_54145

-- Conditions
def total_lemon_heads := 72
def number_of_friends := 6

-- Desired answer
def lemon_heads_per_friend := 12

-- Lean 4 statement
theorem lemon_heads_distribution : total_lemon_heads / number_of_friends = lemon_heads_per_friend := by 
  sorry

end lemon_heads_distribution_l54_54145


namespace borrowed_years_l54_54005

noncomputable def principal : ℚ := 5331.168831168831
noncomputable def rate : ℚ := 0.06
noncomputable def amount_returned : ℚ := 8210
noncomputable def interest_earned : ℚ := amount_returned - principal

theorem borrowed_years :
  ∃ t : ℚ, interest_earned = principal * rate * t ∧ t ≈ 9 :=
by
  -- Insert Lean code for correctness
  sorry

end borrowed_years_l54_54005


namespace find_n_l54_54481

theorem find_n (n : ℕ)
  (h1 : ∃ k : ℕ, k = n^3) -- the cube is cut into n^3 unit cubes
  (h2 : ∃ r : ℕ, r = 4 * n^2) -- 4 faces are painted, each with area n^2
  (h3 : 1 / 3 = r / (6 * k)) -- one-third of the total number of faces are red
  : n = 2 :=
by
  sorry

end find_n_l54_54481


namespace candy_store_l54_54646

theorem candy_store : 
  let percentage_customers_sample_candy : ℝ := 27.5 in
  let uncaught_percentage : ℝ := 20 in
  let caught_percentage : ℝ := 100 - uncaught_percentage in
  let caught_percentage_of_sampling_candy : ℝ := 0.80 * percentage_customers_sample_candy in
  caught_percentage_of_sampling_candy = 22 :=
by
  let percentage_customers_sample_candy := 27.5
  let uncaught_percentage := 20
  let caught_percentage := 100 - uncaught_percentage
  let caught_percentage_of_sampling_candy := 0.80 * percentage_customers_sample_candy
  show caught_percentage_of_sampling_candy = 22
  sorry

end candy_store_l54_54646


namespace vasya_grades_l54_54668

def proves_grades_are_five (grades : List ℕ) : Prop :=
  (grades.length = 5) ∧ (grades.sum = 19) ∧ (grades.sorted.nth 2 = some 4) ∧
  ((grades.count 5) > (grades.count n) for n in (grades.erase_all [5]))

theorem vasya_grades : exists (grades : List ℕ), proves_grades_are_five grades ∧ grades = [2, 3, 4, 5, 5] := 
by
  sorry

end vasya_grades_l54_54668


namespace inclination_angle_of_given_line_l54_54784

noncomputable def inclination_angle: ℝ :=
  let t (x : ℝ) := (x + 1) / Real.sin (50 * Real.pi / 180)
  let y (x : ℝ) := - t x * Real.cos (50 * Real.pi / 180)
  Real.atan (y 0 / (0 + 1)) * 180 / Real.pi

theorem inclination_angle_of_given_line:
  inclination_angle = 140 := 
by 
  -- Import necessary trigonometric identities
  sorry

end inclination_angle_of_given_line_l54_54784


namespace triangle_interior_angle_sum_l54_54436

theorem triangle_interior_angle_sum (ex_angle1 ex_angle2 : ℝ) (h1 : ex_angle1 = 110) (h2 : ex_angle2 = 120) :
  ∃ x y, (x + y = 180) ∧ (y = 130) ∧ (x = 50) := by
  have sum_exterior := 360
  have y := sum_exterior - (ex_angle1 + ex_angle2)
  simp [h1, h2, y]
  use 50
  use 130
  split
  { simp [add_comm] }
  split
  { refl }
  { simp }
sorry

end triangle_interior_angle_sum_l54_54436


namespace m_perp_n_α_perp_β_l54_54732

variables {Plane Line : Type}
variables (α β : Plane) (m n : Line)

def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry

-- Problem 1:
axiom m_perp_α : perpendicular_to_plane m α
axiom n_perp_β : perpendicular_to_plane n β
axiom α_perp_β : perpendicular_planes α β

theorem m_perp_n : perpendicular_lines m n :=
sorry

-- Problem 2:
axiom m_perp_n' : perpendicular_lines m n
axiom m_perp_α' : perpendicular_to_plane m α
axiom n_perp_β' : perpendicular_to_plane n β

theorem α_perp_β' : perpendicular_planes α β :=
sorry

end m_perp_n_α_perp_β_l54_54732


namespace negation_exists_to_forall_l54_54394

theorem negation_exists_to_forall (P : ℝ → Prop) (h : ∃ x : ℝ, x^2 + 3 * x + 2 < 0) :
  (¬ (∃ x : ℝ, x^2 + 3 * x + 2 < 0)) ↔ (∀ x : ℝ, x^2 + 3 * x + 2 ≥ 0) := by
sorry

end negation_exists_to_forall_l54_54394


namespace log_base_9_of_3_l54_54529

theorem log_base_9_of_3 : log 9 3 = 1 / 2 := by
  -- Contextual conditions specified in Lean
  have h1 : 9 = 3 ^ 2 := by norm_num
  have h2 : log 3 9 = 2 := by
    rw [h1, log_pow, log_self]
    norm_num
  rw [log_div_log, h2]
  norm_num

-- helper lemmas
lemma log_pow (a b n : ℝ) (h : 0 < a) (h' : a ≠ 1) : log b (a ^ n) = n * log b a := by
  sorry

lemma log_self (a : ℝ) (h : 0 < a) (h' : a ≠ 1) : log a a = 1 := by
  sorry

lemma log_div_log {a b : ℝ} (h : 0 < a) (h' : a ≠ 1) (h₁ : 0 < b) (h₂ : b ≠ 1) :
   log a b = 1 / log b a := by
  sorry

end log_base_9_of_3_l54_54529


namespace jelly_cost_l54_54931

theorem jelly_cost (N B J : ℕ) (hN : N > 1) (hTotalCost : N * (3 * B + 6 * J) = 306) (hB : B > 0) (hJ : J > 0) :
  (6 * J * N) * 6 / 100 = 2.88 :=
by
  sorry

end jelly_cost_l54_54931


namespace arithmetic_sequence_solution_l54_54965

noncomputable def arithmetic_sequence_problem : Prop :=
exists (S : ℕ → ℝ) (a : ℕ → ℝ), 
  (S 4, S 2, S 3) ∈ {s | ∃ q a1, q ≠ 1 ∧ 
    S n = a1 * n + (q - 1) * (n * (n - 1)) / 2 ∧
    2 * S 2 = S 4 + S 3} ∧ a 2 + a 3 + a 4 = -18 ∧
  ∀ n, S n >= 2016 ↔ n >= 11 ∧ (n % 2 = 1)

theorem arithmetic_sequence_solution : arithmetic_sequence_problem := sorry

end arithmetic_sequence_solution_l54_54965


namespace contrapositive_perpendicularity_original_proposition_true_l54_54957

variables (k1 k2 : ℝ)

def is_perpendicular (k1 k2 : ℝ) : Prop :=
  k1 * k2 = -1

theorem contrapositive_perpendicularity (k1 k2 : ℝ) :
  (¬ is_perpendicular k1 k2) → (¬ (k1 * k2 = -1) → ¬ is_perpendicular k1 k2) :=
  by intros h h'; apply h' h; sorry

theorem original_proposition_true (k1 k2 : ℝ) :
  is_perpendicular k1 k2 ↔ k1 * k2 = -1 :=
  by intros; exact iff.rfl

end contrapositive_perpendicularity_original_proposition_true_l54_54957


namespace mike_total_cards_l54_54004

variable (original_cards : ℕ) (birthday_cards : ℕ)

def initial_cards : ℕ := 64
def received_cards : ℕ := 18

theorem mike_total_cards :
  original_cards = 64 →
  birthday_cards = 18 →
  original_cards + birthday_cards = 82 :=
by
  intros
  sorry

end mike_total_cards_l54_54004


namespace habitable_planets_combinations_l54_54629

open Finset

/-- The number of different combinations of planets that can be occupied is 2478. -/
theorem habitable_planets_combinations : (∑ d in range (18 // 3 + 1), if d % 2 = 0 then choose 8 (6 - d / 3) * choose 7 d else 0) = 2478 := 
by
  sorry

end habitable_planets_combinations_l54_54629


namespace fraction_pure_Fuji_l54_54468

variable (F C T : Nat)
variable (H1 : C = 0.10 * T)
variable (H2 : F + C = 153)
variable (H3 : T = F + 27 + C)

theorem fraction_pure_Fuji : F / T = 3 / 4 :=
by sorry

end fraction_pure_Fuji_l54_54468


namespace impossible_arrangement_l54_54318

theorem impossible_arrangement :
  ∀ (f : Fin 300 → Fin 300 → ℤ),
    (∀ i j, -1 ≤ f i j ∧ f i j ≤ 1) →
    (∀ i j, abs (∑ u : Fin 3, ∑ v : Fin 5, f (i + u) (j + v)) > 3) →
    abs (∑ i : Fin 300, ∑ j : Fin 300, f i j) < 30000 →
    false :=
by
  intros f h_bound h_rect h_sum
  sorry

end impossible_arrangement_l54_54318


namespace trish_price_per_animal_l54_54143

/-- Constants as given in conditions --/
def barbara_stuffed_animals : ℕ := 9
def trish_stuffed_animals : ℕ := 2 * barbara_stuffed_animals
def barbara_price_per_animal : ℕ := 2
def total_donation : ℕ := 45

/-- Hypothesis based on the problem statement --/
def barbara_donation : ℕ := barbara_stuffed_animals * barbara_price_per_animal
def trish_donation : ℕ := total_donation - barbara_donation

/-- Main proof statement --/
theorem trish_price_per_animal :
  ∀ (barbara_stuffed_animals trish_stuffed_animals barbara_price_per_animal total_donation : ℕ),
  barbara_stuffed_animals = 9 →
  trish_stuffed_animals = 2 * barbara_stuffed_animals →
  barbara_price_per_animal = 2 →
  total_donation = 45 →
  trish_donation = 27 →
  (27 / 18 : ℝ) = 1.50 :=
begin
  intros,
  sorry
end

end trish_price_per_animal_l54_54143


namespace length_HP_closest_to_59_l54_54452

theorem length_HP_closest_to_59
  (F G H J K L M N P : Type)
  (edge_length : ℝ)
  (cube : Π (x : Type), ¬x ∈ {F, G, H, J, K, L, M, N} → false)
  (P_on_HG : ∃ x : ℝ, P = x)
  (shortest_distance_GP : ℝ)
  (distance_GP_PFM : shortest_distance_GP = 100)
  (HP_length : ℝ) :
  HP_length = 200 - (100 * real.sqrt 2) :=
begin
  sorry
end

end length_HP_closest_to_59_l54_54452


namespace total_dots_not_visible_l54_54562

theorem total_dots_not_visible :
  ∀ (dice : ℕ) (sum_one_die : ℕ) (visible : list ℕ),
  dice = 4 →
  sum_one_die = 21 →
  visible = [1, 2, 2, 3, 3, 4, 5, 5, 6, 6] →
  sum (repeat sum_one_die dice) - sum visible = 47 :=
by
  intros dice sum_one_die visible hdice hsum hvisible
  rw [hdice, hsum, hvisible]
  simp
  norm_num
  sorry

end total_dots_not_visible_l54_54562


namespace conditional_probability_l54_54412

variable (Ω : Type) [Fintype Ω] [DecidableEq Ω]

namespace conditional_probability_problem

-- Define the sample space for the experiment
def sample_space : Finset (Finset Ω) := 
  {s ∈ (univ : Finset (Finset Ω)) | s.card = 3}

-- Define events A and B
def A (s : Finset Ω) : Prop := s.card = 3
def B (s : Finset Ω) : Prop := ∃ a ∈ s, ∀ b ∈ s, b ≠ a

-- Given the sample space Ω, define the conditional probability
def P (A B : Finset Ω → Prop) : ℝ := 
  (Finset.card (sample_space.filter (λ s, A s ∧ B s))).to_real / 
  (Finset.card (sample_space.filter B)).to_real

theorem conditional_probability :
  P A B = 1 / 2 := by
  sorry

end conditional_probability_problem

end conditional_probability_l54_54412


namespace total_reactions_eq_100_l54_54520

variable (x : ℕ) -- Total number of reactions.
variable (thumbs_up : ℕ) -- Number of "thumbs up" reactions.
variable (thumbs_down : ℕ) -- Number of "thumbs down" reactions.
variable (S : ℕ) -- Net Score.

-- Conditions
axiom thumbs_up_eq_75percent_reactions : thumbs_up = 3 * x / 4
axiom thumbs_down_eq_25percent_reactions : thumbs_down = x / 4
axiom score_definition : S = thumbs_up - thumbs_down
axiom initial_score : S = 50

theorem total_reactions_eq_100 : x = 100 :=
by 
  sorry

end total_reactions_eq_100_l54_54520


namespace vasya_grades_l54_54666

def proves_grades_are_five (grades : List ℕ) : Prop :=
  (grades.length = 5) ∧ (grades.sum = 19) ∧ (grades.sorted.nth 2 = some 4) ∧
  ((grades.count 5) > (grades.count n) for n in (grades.erase_all [5]))

theorem vasya_grades : exists (grades : List ℕ), proves_grades_are_five grades ∧ grades = [2, 3, 4, 5, 5] := 
by
  sorry

end vasya_grades_l54_54666


namespace area_ratios_l54_54713

variables {A B C Q N : Type*}
variables [AddCommGroup A] [AffineSpace A B] [AddCommGroup B] [AffineSpace B C] [AddCommGroup C] [AffineSpace C Q] [AddCommGroup Q] [AffineSpace Q N]
variables {s : ℝ}

-- Midpoint N of BC, Q between B and N, and NA parallel to QC
axiom midpoint_N (BC : line B C) (N : point BC) (mid_BC : midpoint BC N)
axiom point_Q (BC : line B C) (Q : point BC) (between_BN : Q ∈ segment B N)
axiom parallel_NA_QC (NA QC : line) (parallel : parallel NA QC)

-- Area ratios involving triangles AQC and ABC, given the conditions
theorem area_ratios : (∃ s, (s = (area (triangle A Q C) / area (triangle A B C))) ∧ (1/4 ≤ s ∧ s < 1)) :=
sorry

end area_ratios_l54_54713


namespace conditional_probability_l54_54413

variable (Ω : Type) [Fintype Ω] [DecidableEq Ω]

namespace conditional_probability_problem

-- Define the sample space for the experiment
def sample_space : Finset (Finset Ω) := 
  {s ∈ (univ : Finset (Finset Ω)) | s.card = 3}

-- Define events A and B
def A (s : Finset Ω) : Prop := s.card = 3
def B (s : Finset Ω) : Prop := ∃ a ∈ s, ∀ b ∈ s, b ≠ a

-- Given the sample space Ω, define the conditional probability
def P (A B : Finset Ω → Prop) : ℝ := 
  (Finset.card (sample_space.filter (λ s, A s ∧ B s))).to_real / 
  (Finset.card (sample_space.filter B)).to_real

theorem conditional_probability :
  P A B = 1 / 2 := by
  sorry

end conditional_probability_problem

end conditional_probability_l54_54413


namespace ninth_term_geometric_sequence_l54_54387

noncomputable def geometric_seq (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem ninth_term_geometric_sequence (a r : ℝ) (h_positive : ∀ n, 0 < geometric_seq a r n)
  (h_fifth_term : geometric_seq a r 5 = 32)
  (h_eleventh_term : geometric_seq a r 11 = 2) :
  geometric_seq a r 9 = 2 :=
by
{
  sorry
}

end ninth_term_geometric_sequence_l54_54387


namespace impossible_arrangement_l54_54321

theorem impossible_arrangement :
  ∀ (f : Fin 300 → Fin 300 → ℤ),
    (∀ i j, -1 ≤ f i j ∧ f i j ≤ 1) →
    (∀ i j, abs (∑ u : Fin 3, ∑ v : Fin 5, f (i + u) (j + v)) > 3) →
    abs (∑ i : Fin 300, ∑ j : Fin 300, f i j) < 30000 →
    false :=
by
  intros f h_bound h_rect h_sum
  sorry

end impossible_arrangement_l54_54321


namespace minimum_spending_l54_54437

def volume_box : ℝ := 20 * 20 * 15

def total_volume_standard_items : ℝ := 3.06 * 10^6

def reserve_factor : ℝ := 1.03

def cost_per_standard_box : ℝ := 1.30

def cost_per_custom_crate : ℝ := 10

def total_volume_needed : ℝ := total_volume_standard_items * reserve_factor

def number_of_standard_boxes_needed : ℝ := total_volume_needed / volume_box

def cost_of_standard_boxes : ℝ := number_of_standard_boxes_needed.toNat * cost_per_standard_box

def total_minimum_cost : ℝ := cost_of_standard_boxes + cost_per_custom_crate

theorem minimum_spending (h₁ : volume_box = 6000)
                         (h₂ : total_volume_standard_items = 3.06 * 10^6)
                         (h₃ : reserve_factor = 1.03)
                         (h₄ : cost_per_standard_box = 1.30)
                         (h₅ : cost_per_custom_crate = 10)
                         (h₆ : total_volume_needed = total_volume_standard_items * reserve_factor)
                         (h₇ : number_of_standard_boxes_needed = total_volume_needed / volume_box)
                         (h₈ : number_of_standard_boxes_needed.toNat = 526)
                         (h₉ : cost_of_standard_boxes = 526 * cost_per_standard_box)
                         (h₁₀ : cost_of_standard_boxes = 683.80)
                         (h₁₁ : total_minimum_cost = cost_of_standard_boxes + cost_per_custom_crate)
                         (h₁₂ : total_minimum_cost = 693.80) :
  total_minimum_cost = 693.80 := by
  sorry

end minimum_spending_l54_54437


namespace successful_filling_possible_l54_54220

def table_filled_non_unsuccessfully (T : Fin 6 → Fin 3 → ℕ) : Prop :=
  ∀ (i j : Fin 6) (k l : Fin 3), i ≠ j → k ≠ l →
    (T i k = T j k ∧ T i l = T j l) → False

theorem successful_filling_possible :
  ∃ (T : Fin 6 → Fin 3 → ℕ), (∀ i j k l, i ≠ j → k ≠ l →
    (T i k = T j k ∧ T i l = T j l) → False) :=
begin
  use λ i j, match (i, j) with
    | (0, 0) | (1, 0) | (4, 0) => 0
    | (0, 1) | (2, 0) | (5, 1) => 1
    | (1, 1) | (3, 1) | (4, 1) => 1
    | (2, 1) | (5, 0) | (3, 2) => 0
    | (2, 2) | (3, 0) | (5, 2) => 1
    | (4, 2) | (0, 2) | (1, 2) => 1
    | _ => 0
  end,
  sorry
end

end successful_filling_possible_l54_54220


namespace problem1_problem2_l54_54903

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem problem1 : (lg 2)^2 + lg 2 * lg 50 + lg 25 = 2 := 
by sorry

theorem problem2 : (9 / 4)^(3 / 2) + (10^(-1))^(-2) + (3^(-3))^(-1/3) + 2 = 867 / 8 := 
by sorry

end problem1_problem2_l54_54903


namespace eval_exp_l54_54174

theorem eval_exp {a b : ℝ} (h : a = 3^4) : a^(5/4) = 243 :=
by
  sorry

end eval_exp_l54_54174


namespace equation_of_ellipse_x_0_value_exists_l54_54987

-- Define the conditions for the ellipse and the eccentricity
def ellipse (a b x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def eccentricity (a c : ℝ) : ℝ := c / a
def on_ellipse (x y a b : ℝ) : Prop := ellipse a b x y

-- Define the conditions for the problem
variables (a b c : ℝ)
axiom a_gt_b : a > b
axiom b_gt_0 : b > 0
axiom eccentricity_cond : eccentricity a c = sqrt 3 / 2
axiom point_on_ellipse : on_ellipse (sqrt 2) (sqrt 2 / 2) a b
axiom major_minor_relation : a^2 - b^2 = c^2

-- Define the ellipse equation
def ellipse_eq : (ell_x ell_y : ℝ) → Prop := λ x y, (x^2 / a^2 + y^2 / b^2 = 1)

-- Problem statement of (Ⅰ) Find the equation of the ellipse C
theorem equation_of_ellipse : ∃ a b, ellipse_eq (sqrt 2) (sqrt 2 / 2)
  :=
sorry
  
-- Define the line passing through point P(1, 0)
def line_eq (k x : ℝ) : ℝ := k * (x - 1)

-- Problem statement of (Ⅱ) 
theorem x_0_value_exists :
  ∃ x_0 > 2, ∀ (A B P : ℝ × ℝ),
  let d_A := abs (A.1 - x_0),
      d_B := abs (B.1 - x_0),
      PA := (A.1 - P.1)^2 + (A.2 - P.2)^2,
      PB := (B.1 - P.1)^2 + (B.2 - P.2)^2
  in  d_A / d_B = PA / PB :=
sorry

end equation_of_ellipse_x_0_value_exists_l54_54987


namespace grid_cell_A_value_l54_54688

-- Definitions representing the problem conditions.
structure Grid (α : Type) :=
  (cells : list (list α))

-- An operation on the grid involves adding or subtracting 1 on two adjacent cells.
def operation (grid : Grid ℤ) (i j : ℕ) (op : ℤ → ℤ → ℤ) : Grid ℤ :=
sorry -- The actual operation details are omitted, as we only need to state the problem.

-- Main theorem to prove.
theorem grid_cell_A_value (initial : Grid ℤ) (final : Grid ℤ) (A : ℕ × ℕ) 
  (h_preconditions : 
    -- Constraint representing the finite number of operations leading from initial to final.
    exists (ops : list (ℕ × ℕ × (ℤ → ℤ → ℤ))), ∀ (i j : ℕ) (op : ℤ → ℤ → ℤ), 
      (i, j, op) ∈ ops → valid_operation initial final (i, j, op)
  ) : 
  -- The number in cell A in the table on the right is 5
  get_cell_value final A = 5 :=
sorry

end grid_cell_A_value_l54_54688


namespace cost_of_item_D_is_30_usd_l54_54694

noncomputable def cost_of_item_D_in_usd (total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate : ℝ) : ℝ :=
  let total_spent_with_fee := total_spent * (1 + service_fee_rate)
  let item_D_cost_FC := total_spent_with_fee - items_ABC_spent
  item_D_cost_FC * exchange_rate

theorem cost_of_item_D_is_30_usd
  (total_spent : ℝ)
  (items_ABC_spent : ℝ)
  (tax_paid : ℝ)
  (service_fee_rate : ℝ)
  (exchange_rate : ℝ)
  (h_total_spent : total_spent = 500)
  (h_items_ABC_spent : items_ABC_spent = 450)
  (h_tax_paid : tax_paid = 60)
  (h_service_fee_rate : service_fee_rate = 0.02)
  (h_exchange_rate : exchange_rate = 0.5) :
  cost_of_item_D_in_usd total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate = 30 :=
by
  have h1 : total_spent * (1 + service_fee_rate) = 500 * 1.02 := sorry
  have h2 : 500 * 1.02 - 450 = 60 := sorry
  have h3 : 60 * 0.5 = 30 := sorry
  sorry

end cost_of_item_D_is_30_usd_l54_54694


namespace factor_expression_l54_54935

theorem factor_expression (
  x y z : ℝ
) : 
  ( (x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / 
  ( (x - y)^3 + (y - z)^3 + (z - x)^3) = 
  (x + y) * (y + z) * (z + x) := 
sorry

end factor_expression_l54_54935


namespace charles_drawn_after_work_l54_54907

-- Conditions
def total_papers : ℕ := 20
def drawn_today : ℕ := 6
def drawn_yesterday_before_work : ℕ := 6
def papers_left : ℕ := 2

-- Question and proof goal
theorem charles_drawn_after_work :
  ∀ (total_papers drawn_today drawn_yesterday_before_work papers_left : ℕ),
  total_papers = 20 →
  drawn_today = 6 →
  drawn_yesterday_before_work = 6 →
  papers_left = 2 →
  (total_papers - drawn_today - drawn_yesterday_before_work - papers_left = 6) :=
by
  intros total_papers drawn_today drawn_yesterday_before_work papers_left
  assume h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  exact sorry

end charles_drawn_after_work_l54_54907


namespace intersection_M_N_l54_54258

def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {x | x^2 - 4 * x + 3 = 0}

theorem intersection_M_N : M ∩ N = {1, 3} :=
by sorry

end intersection_M_N_l54_54258


namespace quadratic_sum_l54_54046

def quadratic_expression := 8 * (x : ℝ)^2 + 48 * x + 200

theorem quadratic_sum (a b c : ℝ) (h : quadratic_expression = a * (x + b)^2 + c) : a + b + c = 139 :=
sorry

end quadratic_sum_l54_54046


namespace probability_valid_assignment_l54_54485

open Function

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_power_of (a b : ℕ) : Prop := 
  ∃ k : ℕ, b = a ^ k

def perfect_square (n : ℕ) : Prop := 
  ∃ m : ℕ, m * m = n

def valid_assignment (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  c ∈ {2, 3, 5, 7, 11} ∧
  is_power_of c b ∧ 
  perfect_square a ∧ 
  a % b = 0

theorem probability_valid_assignment : 
  let total_assignments := 1320
  let valid_cases := 2
  (valid_cases / total_assignments : ℚ) = 1 / 660 :=
by
  sorry

end probability_valid_assignment_l54_54485


namespace simplify_expression_l54_54017

variable (b c d x y : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * b^2 * y^3 + c^3 * y^3) + dy * (b^2 * x^3 + 3 * c^3 * x^3 + c^3 * y^3)) / (cx + dy) 
  = b^2 * x^3 + 3 * c^2 * xy^3 + c^3 * y^3 :=
by sorry

end simplify_expression_l54_54017


namespace tap_fills_tank_without_leakage_in_12_hours_l54_54270

theorem tap_fills_tank_without_leakage_in_12_hours 
  (R_t R_l : ℝ)
  (h1 : (R_t - R_l) * 18 = 1)
  (h2 : R_l * 36 = 1) :
  1 / R_t = 12 := 
by
  sorry

end tap_fills_tank_without_leakage_in_12_hours_l54_54270


namespace number_of_committees_correct_l54_54142

-- Definition to formalize the problem setup
def departments : List String := ["mathematics", "statistics", "computer science", "data science"]

def professors (d : String) : Type := { P : String // P ∈ ["M1", "M2", "M3", "F1", "F2", "F3"] }

-- Definition of the committee with conditions specified
noncomputable def number_of_committees 
  (D : List String) 
  (male_count_per_dept female_count_per_dept : String -> Nat)
  (total_committee_size men_count women_count : Nat) 
  (required_distribution : Nat) :=
  
  let case1 := (9^4) 
  let case2 := 9 * 81 * 6 * 2
  case1 + case2

-- The theorem statement
theorem number_of_committees_correct :
  number_of_committees departments (λ _, 3) (λ _, 3) 8 4 4 2 = 15309 :=
by
  sorry

end number_of_committees_correct_l54_54142


namespace race_winner_laps_l54_54788

/-- Given:
  * A lap equals 100 meters.
  * Award per hundred meters is $3.5.
  * The winner earned $7 per minute.
  * The race lasted 12 minutes.
  Prove that the number of laps run by the winner is 24.
-/ 
theorem race_winner_laps :
  let lap_distance := 100 -- meters
  let award_per_100meters := 3.5 -- dollars per 100 meters
  let earnings_per_minute := 7 -- dollars per minute
  let race_duration := 12 -- minutes
  let total_earnings := earnings_per_minute * race_duration
  let total_100meters := total_earnings / award_per_100meters
  let laps := total_100meters
  laps = 24 := by
  sorry

end race_winner_laps_l54_54788


namespace proof_problem_l54_54836

-- Variables representing the numbers a, b, and c
variables {a b c : ℝ}

-- Given condition
def given_condition (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (b^2 + c^2) = a / c

-- Required to prove
def to_prove (a b c : ℝ) : Prop :=
  (a / b = b / c) → False

-- Theorem stating that the given condition does not imply the required assertion
theorem proof_problem (a b c : ℝ) (h : given_condition a b c) : to_prove a b c :=
sorry

end proof_problem_l54_54836


namespace apples_selling_price_l54_54424

theorem apples_selling_price (total_harvest : ℕ) (juice : ℕ) (restaurant : ℕ) (bag_weight : ℕ) (total_revenue : ℤ) (sold_bags : ℕ) :
  total_harvest = 405 →
  juice = 90 →
  restaurant = 60 →
  bag_weight = 5 →
  total_revenue = 408 →
  sold_bags = (total_harvest - juice - restaurant) / bag_weight →
  total_revenue / sold_bags = 8 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end apples_selling_price_l54_54424


namespace marble_set_count_l54_54565

theorem marble_set_count :
  ∃ (x : ℕ), 
    (0.10 * 50 + 0.20 * x = 17) ∧ x = 60 :=
by
  existsi 60
  split
  sorry
  sorry

end marble_set_count_l54_54565


namespace tangent_line_at_zero_find_a_minimum_value_l54_54990

-- Proof Problem (1)
theorem tangent_line_at_zero (a : ℝ) (h_a : a = 1) :
  let f := λ x : ℝ, (1 / 3) * x^3 - a * x + 1 in
  ∀ x : ℝ, let f' := (x^2 - a) in
  (f'(0) + (f 0 - 1) = 0) →
  (f'(0) = -1) →
  (f 0 = 1) →
  y = -x + 1 := 
sorry

-- Proof Problem (2)
theorem find_a_minimum_value (f : ℝ → ℝ)
  (a : ℝ) (h_f : ∀ x : ℝ, f x = (1 / 3) * x^3 - a * x + 1)
  (h_min : ∀ x ∈ Set.Icc 0 1, f x ≥ (11 / 12)) :
  a = 1 / 4 := 
sorry

end tangent_line_at_zero_find_a_minimum_value_l54_54990


namespace rows_equal_columns_l54_54219

variable {n : ℕ}
variable {a b : Fin n → ℝ}
variable {M : ℕ → ℕ → ℝ}
variable (prod_eq_col : ∀ j: Fin n, ∃ c: ℝ, (∏ i, M i j) = c)

-- Defining the matrix cell values
def M (i j : Fin n) : ℝ := a i + b j

-- The theorem statement
theorem rows_equal_columns :
  (∀ j: Fin n, ∃ c: ℝ, (∏ i, M i j) = c) →
  (∀ i: Fin n, ∃ c: ℝ, (∏ j, M i j) = c) :=
by
  sorry

end rows_equal_columns_l54_54219


namespace find_a_b_l54_54617

def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

def f_derivative (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_a_b (a b : ℝ) (h1 : f 1 a b = 10) (h2 : f_derivative 1 a b = 0) : a = 4 ∧ b = -11 :=
sorry

end find_a_b_l54_54617


namespace final_price_approximation_l54_54117

noncomputable def final_selling_price(cycle_price initial_helmet_price light_unit_price 
                                      cycle_discount_percent tax_percent 
                                      cycle_loss_percent helmet_profit_percent 
                                      transaction_fee_percent) : ℝ :=
let cycle_price_after_discount := cycle_price * (1 - cycle_discount_percent / 100) in
let total_cost_before_tax := cycle_price_after_discount + initial_helmet_price + 2 * light_unit_price in
let total_cost_after_tax := total_cost_before_tax * (1 + tax_percent / 100) in
let cycle_selling_price := cycle_price_after_discount * (1 - cycle_loss_percent / 100) in
let helmet_selling_price := initial_helmet_price * (1 + helmet_profit_percent / 100) in
let lights_selling_price := 2 * light_unit_price in
let total_selling_price := cycle_selling_price + helmet_selling_price + lights_selling_price in
let transaction_fee := total_selling_price * (transaction_fee_percent / 100) in
total_selling_price - transaction_fee

theorem final_price_approximation : 
final_selling_price 1400 400 200 10 5 12 25 3 = 1949 :=
by
  -- Calculation steps are skipped
  sorry

end final_price_approximation_l54_54117


namespace liquid_level_ratio_l54_54812

noncomputable def volume_of_cone (r h : ℝ) : ℝ := 
  (1/3) * real.pi * r^2 * h

noncomputable def volume_of_sphere (r : ℝ) : ℝ := 
  (4/3) * real.pi * r^3

theorem liquid_level_ratio 
  (h1 h2 : ℝ) 
  (V : ℝ := volume_of_cone 4 h1)
  (r1 r2 : ℝ := 4) 
  (r_sphere : ℝ := 1.5)
  (V_sphere := volume_of_sphere r_sphere)
  (equal_volumes : volume_of_cone 4 h1 = volume_of_cone 8 h2) :
  V + V_sphere = volume_of_cone 4 (h1 + 0.84375) ∧ 
  V + V_sphere = volume_of_cone 8 (h2 + 0.2109375) →
  (h1 + 0.84375) - h1 = 4 * ((h2 + 0.2109375) - h2) :=
sorry

end liquid_level_ratio_l54_54812


namespace find_n_l54_54537

-- Define the function for the number of divisors
def tau (n : ℕ) : ℕ :=
  Finset.card (Finset.filter (λ d, n % d = 0) (Finset.range (n+1)))

-- Define the function for the sum of divisors
def sigma (n : ℕ) : ℕ :=
  Finset.sum (Finset.filter (λ d, n % d = 0) (Finset.range (n+1))) id

-- Define a predicate indicating if a number has exactly two distinct prime divisors
def has_two_prime_divisors (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p^2 * q

-- State the main theorem
theorem find_n (n : ℕ) (h1 : has_two_prime_divisors n) (h2 : tau n = 6) (h3 : sigma n = 28) : n = 12 := 
  sorry

end find_n_l54_54537


namespace evaluate_expression_l54_54531

theorem evaluate_expression 
    (a b c : ℕ) 
    (ha : a = 7)
    (hb : b = 11)
    (hc : c = 13) :
  let numerator := a^3 * (1 / b - 1 / c) + b^3 * (1 / c - 1 / a) + c^3 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  numerator / denominator = 31 := 
by {
  sorry
}

end evaluate_expression_l54_54531


namespace concert_ticket_cost_l54_54325

-- Definitions based on the conditions
def hourlyWage : ℝ := 18
def hoursPerWeek : ℝ := 30
def drinkTicketCost : ℝ := 7
def numberOfDrinkTickets : ℝ := 5
def outingPercentage : ℝ := 0.10
def weeksPerMonth : ℝ := 4

-- Proof statement
theorem concert_ticket_cost (hourlyWage hoursPerWeek drinkTicketCost numberOfDrinkTickets outingPercentage weeksPerMonth : ℝ)
  (monthlySalary := weeksPerMonth * (hoursPerWeek * hourlyWage))
  (outingAmount := outingPercentage * monthlySalary)
  (costOfDrinkTickets := numberOfDrinkTickets * drinkTicketCost)
  (costOfConcertTicket := outingAmount - costOfDrinkTickets)
  : costOfConcertTicket = 181 := 
sorry

end concert_ticket_cost_l54_54325


namespace tangent_circle_properties_l54_54564

-- Definitions of conditions as constants and hypotheses
constant P : EuclideanGeometry.Point
constant O : EuclideanGeometry.Point
constant D : EuclideanGeometry.Point
constant A : EuclideanGeometry.Point
constant B : EuclideanGeometry.Point
constant M : EuclideanGeometry.Point
constant N : EuclideanGeometry.Point
constant circle_center : ∀ (O : EuclideanGeometry.Point), EuclideanGeometry.Circle
constant is_tangent : (P D : EuclideanGeometry.Point) → EuclideanGeometry.Circle = circle_center O → Prop
constant on_diameter : (A B : EuclideanGeometry.Point) → EuclideanGeometry.Circle = circle_center O → Prop
constant perpendicular : (P O : EuclideanGeometry.Point) → (A B : EuclideanGeometry.Point) → Prop
constant intersection : ∀ (D A : EuclideanGeometry.Point) (P O : EuclideanGeometry.Point), EuclideanGeometry.Point

-- Tangent, diameter, and intersection properties
axiom tangent_PD : is_tangent P D (circle_center O)
axiom diameter_AB : on_diameter A B (circle_center O)
axiom perp_DAB_PO : perpendicular P O A B
axiom intersect_M : M = intersection D A P O
axiom intersect_N : N = intersection D B P O

-- Proof objective
theorem tangent_circle_properties (P O D A B M N : EuclideanGeometry.Point)
  (tangent_PD : is_tangent P D (circle_center O))
  (diameter_AB : on_diameter A B (circle_center O))
  (perp_DAB_PO : perpendicular P O A B)
  (intersect_M : M = intersection D A P O)
  (intersect_N : N = intersection D B P O)) :
  (EuclideanGeometry.distance P M = EuclideanGeometry.distance P N) ∧
  (EuclideanGeometry.distance P N = EuclideanGeometry.distance P D) :=
by
  sorry

end tangent_circle_properties_l54_54564


namespace area_of_triangle_DEF_l54_54918

theorem area_of_triangle_DEF :
  let s := 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let radius := s
  let distance_between_centers := 2 * radius
  let side_of_triangle_DEF := distance_between_centers
  let triangle_area := (Real.sqrt 3 / 4) * side_of_triangle_DEF^2
  triangle_area = 4 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_DEF_l54_54918


namespace imag_conj_z_is_minus_one_l54_54383

noncomputable def z : Complex := (-3 + Complex.i) / (2 + Complex.i)
def conj_z : Complex := Complex.conj z
def imag_conj_z : ℂ := Complex.im conj_z

theorem imag_conj_z_is_minus_one : imag_conj_z = -1 :=
sorry

end imag_conj_z_is_minus_one_l54_54383


namespace mid_segment_PQ_l54_54295

variables {A B C A' B' P Q D : EuclideanSpace ℝ}
variables {circumcircle : Circle (triangle ABC)}
variables {AA' BB' BD AD : Line}

-- Condition: Triangle ABC is acute
def acute_triangle (A B C : EuclideanSpace ℝ) : Prop := ∀ angle (ABC), angle < π / 2

-- Condition: AA' and BB' are altitudes
def is_altitude (A' : EuclideanSpace ℝ) (A B C : EuclideanSpace ℝ) : Prop := 
  ∀ (line_from_A' : Line), perpendicular line_from_A' (Line A C)

-- Condition: D is on arc ACB of the circumcircle
def arc_ACB (D : EuclideanSpace ℝ) (circumcircle : Circle (triangle ABC)) : Prop := 
  on_circle D circumcircle ∧ between A C B D

-- Condition: Line AA' intersects BD at P, Line BB' intersects AD at Q
def intersects (l1 l2 : Line) : EuclideanSpace ℝ := ∃ P, lies_on P l1 ∧ lies_on P l2

theorem mid_segment_PQ (ABC : {ABC : EuclideanSpace ℝ // acute_triangle A B C}) 
  (A' B' P Q : EuclideanSpace ℝ) (AA' BB' BD AD : Line)
  (Ha : is_altitude A' A B C) (Hb : is_altitude B' A B C) (Hc : arc_ACB D circumcircle)
  (Hintersect1 : intersects AA' BD = P) (Hintersect2 : intersects BB' AD = Q) : 
  passes_through (line A' B') (midpoint P Q) :=
sorry

end mid_segment_PQ_l54_54295


namespace polynomial_with_root_5_plus_i_and_leading_coeff_three_l54_54197

theorem polynomial_with_root_5_plus_i_and_leading_coeff_three :
  ∃ (p : Polynomial ℝ), p.coeff 2 = 3 ∧ p.coeff 0 = 78 ∧ p.coeff 1 = -30 ∧ p.is_root (5 + Complex.i) :=
by
  sorry

end polynomial_with_root_5_plus_i_and_leading_coeff_three_l54_54197


namespace walking_speed_is_10_km_per_hour_l54_54473

-- Definitions of the given conditions
def time_minutes : ℝ := 3
def time_hours : ℝ := time_minutes / 60
def distance_meters : ℝ := 500
def distance_kilometers : ℝ := distance_meters / 1000

-- Theorem to prove the walking speed
theorem walking_speed_is_10_km_per_hour :
  (distance_kilometers / time_hours = 10) := 
by sorry

end walking_speed_is_10_km_per_hour_l54_54473


namespace find_C_and_D_l54_54937

theorem find_C_and_D :
  ∃ (C D : ℚ), 
  (C = 73/12 ∧ D = 11/12) ∧ 
  (∀ x : ℚ, x ≠ 10 → x ≠ -2 → 
  (7 * x + 3) / (x^2 - 8 * x - 20) = 
  C / (x - 10) + D / (x + 2)) :=
by
  use 73/12, 11/12
  split
  · exact ⟨rfl, rfl⟩
  · intros x hx1 hx2
    calc
      (7 * x + 3) / (x^2 - 8 * x - 20) = (7 * x + 3) / ((x - 10) * (x + 2)) : by rw [sub_mul, add_mul]
      ... = (73 / 12) / (x - 10) + (11 / 12) / (x + 2) : sorry

end find_C_and_D_l54_54937


namespace kim_total_water_intake_l54_54704

def quarts_to_ounces (q : ℝ) : ℝ := q * 32

theorem kim_total_water_intake :
  (quarts_to_ounces 1.5) + 12 = 60 := 
by
  -- proof step 
  sorry

end kim_total_water_intake_l54_54704


namespace tetrahedron_has_7_midplanes_parallelepiped_has_3_midplanes_four_points_form_29_parallelepipeds_l54_54552

def is_midplane (α : Plane) (P : Polyhedron) : Prop := 
  ∀ v ∈ vertices P, distance v α = some constant

-- Part 1: Number of distinct midplanes for a tetrahedron
theorem tetrahedron_has_7_midplanes (T : Tetrahedron) : 
  ∃ midplanes : Finset Plane, midplanes.card = 7 ∧ ∀ α ∈ midplanes, is_midplane α T :=
by sorry

-- Part 2: Number of distinct midplanes for a parallelepiped
theorem parallelepiped_has_3_midplanes (P : Parallelepiped) : 
  ∃ midplanes : Finset Plane, midplanes.card = 3 ∧ ∀ α ∈ midplanes, is_midplane α P :=
by sorry

-- Part 3: Number of distinct parallelepipeds formed by four non-coplanar points
theorem four_points_form_29_parallelepipeds (A B C D : Point) (h : ¬ Coplanar A B C D) :
  ∃ parallelepipeds : Finset Parallelepiped, parallelepipeds.card = 29 ∧ 
  ∀ P ∈ parallelepipeds, ∃ S : Finset Point, S.card = 4 ∧ A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ 
  (S ⊆ vertices P):=
by sorry

end tetrahedron_has_7_midplanes_parallelepiped_has_3_midplanes_four_points_form_29_parallelepipeds_l54_54552


namespace problem_I_problem_II_l54_54024

noncomputable def line_l_condition := ∀ (ρ θ : ℝ), ρ * sin (θ + π / 4) = sqrt 2 / 2

noncomputable def circle_C_equation := ∀ θ : ℝ, 
  (exists (x y : ℝ), x = 2 * cos θ ∧ y = -2 + 2 * sin θ ∧ (x^2 + (y + 2)^2 = 4))

noncomputable def ellipse_equation := ∀ φ : ℝ, 
  (exists (x y : ℝ), x = 2 * cos φ ∧ y = sqrt 3 * sin φ ∧ ((x^2) / 4 + (y^2) / 3 = 1))

theorem problem_I (C : ℝ × ℝ) (r : ℝ) :
  (∀ ρ θ : ℝ, ρ * sin (θ + π / 4) = sqrt 2 / 2) →
  (∀ θ : ℝ, ∃ (x y : ℝ), x = 2 * cos θ ∧ y = -2 + 2 * sin θ ∧ (x^2 + (y + 2)^2 = 4)) →
  dist C (λ p : ℝ × ℝ, p.1 + p.2 + 1 = 0) > r → 
  r = 2 →
  C = (0, -2) →
  ∃ d : ℝ, d = 3 * sqrt 2 / 2 ∧ d > 2 :=
sorry

theorem problem_II (C : ℝ × ℝ) :
  (∀ φ : ℝ, ∃ (x y : ℝ), x = 2 * cos φ ∧ y = sqrt 3 * sin φ ∧ ((x^2) / 4 + (y^2) / 3 = 1)) →
  (∀ t : ℝ, ∃ (x y : ℝ), (x = (sqrt 2 / 2) * t) ∧ (y = -2 + (sqrt 2 / 2) * t)) →
  dist (∃ A B : ℝ × ℝ, ∀ t : ℝ, (λ p : ℝ × ℝ, p.1 = sqrt 2 * p.2  / 2 ∧ p.2 = -2 + sqrt 2 * p.2 / 2)) A B = 12 * sqrt 2 / 7 :=
sorry

end problem_I_problem_II_l54_54024


namespace triangle_equilateral_if_similar_l54_54852

-- Definitions of points and triangle
variables {A B C A1 B1 C1 : Type} [Point A] [Point B] [Point C] [Point A1] [Point B1] [Point C1]
variables (ABC : Triangle A B C) (A1B1C1 : Triangle A1 B1 C1)

-- Definition of inscribed circle and similarity of triangles
def inscribed_circle (T : Triangle A B C) (A1 B1 C1 : Point) : Prop :=
  touches T.sides A1 B1 C1

def similar (T1 T2 : Triangle A B C) : Prop :=
  ∀ a b c : Angle, T1.interior_angles a b c ↔ T2.interior_angles a b c

-- The theorem statement
theorem triangle_equilateral_if_similar (T1 T2 : Triangle A B C) (h₁ : inscribed_circle T1 A1 B1 C1)
(h₂ : similar T1 T2) : T1.equilateral := 
sorry

end triangle_equilateral_if_similar_l54_54852


namespace intersection_A_B_l54_54590

def A : Set ℕ := {x | -1 < x ∧ x < 3}

def B : Set ℤ := {x | -2 ≤ x ∧ x < 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  -- Proof omitted
  sorry

end intersection_A_B_l54_54590


namespace circle_characterization_l54_54939

noncomputable def A : ℝ × ℝ := (-2, -4)
noncomputable def B : ℝ × ℝ := (8, 6)
noncomputable def tangent_line (x y : ℝ) : Prop := x + 3 * y - 26 = 0
noncomputable def circle_eq (x y D E F : ℝ) : Prop := x^2 + y^2 + D * x + E * y + F = 0

theorem circle_characterization :
  ∃ (D E F : ℝ), 
    circle_eq (-2) (-4) D E F ∧ 
    tangent_line 8 6 ∧
    circle_eq 8 6 D E F ∧
    3 * D - E + 36 = 0 ∧ 
    2 * D + 4 * E - F = 20 ∧ 
    8 * D + 6 * E + F = -100 ∧
    x^2 + y^2 + D * x + E * y + F = 0 :=
by 
  have h1 : (3 * (-11) - 3 + 36 = 0), sorry,
  have h2 : (2 * (-11) + 4 * 3 - (-30) = 20), sorry,
  have h3 : (8 * (-11) + 6 * 3 + (-30) = -100), sorry,
  use [-11, 3, -30],
  split; assumption, split; assumption, split; assumption, split; exact h1, split; exact h2, exact h3

end circle_characterization_l54_54939


namespace percentage_proof_l54_54850

/--
Given that 30% of 50% of 5200 is 117, prove that the percentage (P) is 15%.
-/
theorem percentage_proof (n : ℕ) (h₁ : 50% of n = 2600)
                         (h₂ : 30% of 2600 = 780)
                         (h₃ : P * 780 = 11700) :
  P = 15 := by
  sorry

end percentage_proof_l54_54850


namespace find_m_l54_54262

-- Definition of the vectors
def vector_a : ℝ × ℝ := (-2, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (3, m)

-- Definition of perpendicularity in terms of the dot product
def perpendicular (a b : ℝ × ℝ) : Prop := 
  a.1 * b.1 + a.2 * b.2 = 0

-- The statement of the problem
theorem find_m (m : ℝ) (h : perpendicular vector_a (vector_b m)) : m = 2 :=
begin
  sorry
end

end find_m_l54_54262


namespace cos_angle_identity_l54_54593

theorem cos_angle_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 :=
by
  sorry

end cos_angle_identity_l54_54593


namespace disks_rotation_l54_54798

theorem disks_rotation (n : ℕ) (a b : Fin n → ℝ)
  (ha : (∑ i, a i) < 0)
  (hb : (∑ i, b i) < 0) :
  ∃ σ : Fin n → Fin n, (∑ i, a i * b (σ i)) > 0 :=
sorry

end disks_rotation_l54_54798


namespace alex_downhill_time_l54_54884

theorem alex_downhill_time
  (speed_flat : ℝ)
  (time_flat : ℝ)
  (speed_uphill : ℝ)
  (time_uphill : ℝ)
  (speed_downhill : ℝ)
  (distance_walked : ℝ)
  (total_distance : ℝ)
  (h_flat : speed_flat = 20)
  (h_time_flat : time_flat = 4.5)
  (h_uphill : speed_uphill = 12)
  (h_time_uphill : time_uphill = 2.5)
  (h_downhill : speed_downhill = 24)
  (h_walked : distance_walked = 8)
  (h_total : total_distance = 164)
  : (156 - (speed_flat * time_flat + speed_uphill * time_uphill)) / speed_downhill = 1.5 :=
by 
  sorry

end alex_downhill_time_l54_54884


namespace unused_square_is_teal_l54_54363

-- Define the set of colors
inductive Color
| Cyan
| Magenta
| Lime
| Purple
| Teal
| Silver
| Violet

open Color

-- Define the condition that Lime is opposite Purple in the cube
def opposite (a b : Color) : Prop :=
  (a = Lime ∧ b = Purple) ∨ (a = Purple ∧ b = Lime)

-- Define the problem: seven squares are colored and one color remains unused.
def seven_squares_set (hinge : List Color) : Prop :=
  hinge.length = 6 ∧ 
  opposite Lime Purple ∧
  Color.Cyan ∈ hinge ∧
  Color.Magenta ∈ hinge ∧ 
  Color.Lime ∈ hinge ∧ 
  Color.Purple ∈ hinge ∧ 
  Color.Teal ∈ hinge ∧ 
  Color.Silver ∈ hinge ∧ 
  Color.Violet ∈ hinge

theorem unused_square_is_teal :
  ∃ hinge : List Color, seven_squares_set hinge ∧ ¬ (Teal ∈ hinge) := 
by sorry

end unused_square_is_teal_l54_54363


namespace fixed_point_quadratic_l54_54639

theorem fixed_point_quadratic : 
  (∀ m : ℝ, 3 * a ^ 2 - m * a + 2 * m + 1 = b) → (a = 2 ∧ b = 13) := 
by sorry

end fixed_point_quadratic_l54_54639


namespace proof_problem_l54_54281

variables (Z Y X W : Type) 
variables (h1 : ∀ z : Z, Y) -- All Zeefs are Yamps
variables (h2 : ∀ x : X, Y) -- All Xoons are Yamps
variables (h3 : ∀ w : W, Z) -- All Woons are Zeefs
variables (h4 : ∀ x : X, W) -- All Xoons are Woons

theorem proof_problem : (∀ x : X, Z) ∧ (∀ x : X, Y) :=
by {
  sorry
}

end proof_problem_l54_54281


namespace hexagon_diagonals_concurrency_l54_54575

open EuclideanGeometry

variable (A B C D E F : Point)

-- Define ConvexHexagon
def ConvexHexagon (A B C D E F : Point) : Prop :=
  ConvexPolygon 6 [A, B, C, D, E, F]

-- Define Bisection by a diagonal
def DiagonalBisectsArea (A B C D E F : Point) (d1 d2 : Point) : Prop :=
  Area (Polygon [A, B, C, d1, d2]) = (1 / 2) * Area (Polygon [A, B, C, D, E, F])

theorem hexagon_diagonals_concurrency
  (h_convex : ConvexHexagon A B C D E F)
  (h_AD_bisect : DiagonalBisectsArea A B C D E F A D)
  (h_BE_bisect : DiagonalBisectsArea A B C D E F B E)
  (h_CF_bisect : DiagonalBisectsArea A B C D E F C F):
  ∃ P : Point, Collinear [A, P, D] ∧ Collinear [B, P, E] ∧ Collinear [C, P, F] :=
by
  sorry

end hexagon_diagonals_concurrency_l54_54575


namespace curve_meets_line_once_l54_54995

theorem curve_meets_line_once (a : ℝ) (h : a > 0) :
  (∃! P : ℝ × ℝ, (∃ θ : ℝ, P.1 = a + 4 * Real.cos θ ∧ P.2 = 1 + 4 * Real.sin θ)
  ∧ (3 * P.1 + 4 * P.2 = 5)) → a = 7 :=
sorry

end curve_meets_line_once_l54_54995


namespace prob_six_or_more_points_l54_54378

def outcome : Type := {win | draw | lose}

def points (o : outcome) : ℕ :=
  match o with
  | win => 3
  | draw => 1
  | lose => 0

def prob (o : outcome) : ℝ :=
  match o with
  | win => 0.5
  | draw => 0.3
  | lose => 0.2

def scenarios : List (outcome × outcome × outcome) :=
  [(win, win, lose), (win, win, draw), (win, win, win),
   (win, lose, win), (win, draw, win),
   (lose, win, win), (draw, win, win)]

def scenario_prob (s : outcome × outcome × outcome) : ℝ :=
  prob s.1 * prob s.2 * prob s.3

def is_six_or_more_points (s : outcome × outcome × outcome) : Prop :=
  points s.1 + points s.2 + points s.3 ≥ 6

theorem prob_six_or_more_points : 
  (scenarios.filter is_six_or_more_points).sum (λ s, scenario_prob s) = 0.5 :=
by
  sorry

end prob_six_or_more_points_l54_54378


namespace min_value_expression_l54_54574

theorem min_value_expression (a b : ℝ) (h : a > b) (h1 : b > 0) : 
  (a^2 + 1 / (b * (a - b)) >= 5) :=
begin
  sorry
end

end min_value_expression_l54_54574


namespace ratio_HC_JE_l54_54751

theorem ratio_HC_JE (A B C D E F G H J: Type) 
  (h1 : B = A + 2) 
  (h2 : C = B + 2) 
  (h3 : D = C + 3) 
  (h4 : E = D + 3) 
  (h5 : F = E + 4)
  (hG_not_on_AF : ¬ colinear A F G)
  (hH_on_GD : H ∈ line_segment G D)
  (hJ_on_GF : J ∈ line_segment G F)
  (hHC_parallel_AG : parallel (line H C) (line A G))
  (hJE_parallel_AG : parallel (line J E) (line A G))
  : (length_segment H C) / (length_segment J E) = 3 / 2 := 
  sorry

end ratio_HC_JE_l54_54751


namespace opposite_sides_range_a_l54_54271

theorem opposite_sides_range_a (a: ℝ) :
  ((1 - 2 * a + 1) * (a + 4 + 1) < 0) ↔ (a < -5 ∨ a > 1) :=
by
  sorry

end opposite_sides_range_a_l54_54271


namespace max_profit_l54_54133

def fixed_cost : ℝ := 7500
def additional_cost_per_unit : ℝ := 100

def revenue (x : ℝ) : ℝ :=
  if x ≤ 200 then 400 * x - x^2
  else 40000

def total_cost (x : ℝ) : ℝ := fixed_cost + additional_cost_per_unit * x

def profit (x : ℝ) : ℝ :=
  revenue x - total_cost x

theorem max_profit : 
  let f := λ x, if x ≤ 200 then -x^2 + 300 * x - 7500 else -100 * x + 32500 in
  f 150 = 15000 ∧ ∀ x, f x ≤ 15000 :=
by
  sorry

end max_profit_l54_54133


namespace total_number_of_possible_assignment_plans_l54_54799

-- Define the countries and venues
inductive Country
  | China
  | England
  | Sweden

inductive Venue
  | one
  | two
  | three

-- Define the conditions
def umpires : List (Country × Country) :=
  [(Country.China, Country.England), (Country.China, Country.Sweden), (Country.England, Country.Sweden)]

def different_countries (a b : (Country × Country)) : Prop :=
  a.1 ≠ a.2 ∧ b.1 ≠ b.2

-- Define the problem
theorem total_number_of_possible_assignment_plans : 
  (permutations (list.duplicate 1 [Country.China, Country.England, Country.Sweden]).toArray × permutations Venue.toArray).length = 48 := sorry

end total_number_of_possible_assignment_plans_l54_54799


namespace total_rats_l54_54701

theorem total_rats (Elodie_rats Hunter_rats Kenia_rats : ℕ) 
  (h1 : Elodie_rats = 30) 
  (h2 : Elodie_rats = Hunter_rats + 10)
  (h3 : Kenia_rats = 3 * (Elodie_rats + Hunter_rats)) :
  Elodie_rats + Hunter_rats + Kenia_rats = 200 :=
by
  sorry

end total_rats_l54_54701


namespace range_of_a2_over_b_proof_l54_54594

noncomputable def range_of_a2_over_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : Set ℝ :=
{ ∃ y, ∃ d, ∃ e, ∃ x : ℝ, (x + y + a = 0) ∧ ((x - b)^2 + (y - 1)^2 = 2) ∧ d * ((a + b + 1) / sqrt(2)) = sqrt(2) ∧ e * (abs(a + b + 1) = 2) ∧ Set.Ioo 0 +∞ }

theorem range_of_a2_over_b_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  range_of_a2_over_b a b h1 h2 h3 = { x : ℝ | 0 < x }  :=
begin
  sorry
end

end range_of_a2_over_b_proof_l54_54594


namespace find_scalar_m_l54_54333

-- Define the origin O
def O : Point := ⟨0, 0, 0⟩

-- Define the vectors OE, OF, OG, OH for points E, F, G, H
variables {E F G H : Point}

-- Define the coplanar condition for points
def coplanar (E F G H : Point) : Prop :=
  let v1 := (F -ᵥ E)
  let v2 := (G -ᵥ E)
  let v3 := (H -ᵥ E)
  ∃ k1 k2 k3 : ℝ, k1 • v1 + k2 • v2 + k3 • v3 = 0

-- Determine the scalar m such that the points E, F, G, H are coplanar
theorem find_scalar_m
  (m : ℝ) 
  (h : 4 • (E -ᵥ O) - 3 • (F -ᵥ O) + 6 • (G -ᵥ O) + m • (H -ᵥ O) = 0) :
  coplanar E F G H :=
begin
  -- Since the proof is not required, we introduce a placeholder
  sorry
end

end find_scalar_m_l54_54333


namespace partA_maximize_daily_production_partB_maximize_total_production_partC_maximize_revenue_partD_minimum_exhibition_revenue_l54_54834

noncomputable def dailyProduction (L: ℝ) : ℝ := sqrt L
noncomputable def lifespan (L: ℝ) : ℝ := 8 - sqrt L
noncomputable def totalProduction (L: ℝ) : ℝ := (sqrt L) * lifespan L
noncomputable def revenue (L R X: ℝ) : ℝ := (90 * totalProduction L) + (X * (24 - L) * lifespan L)

theorem partA_maximize_daily_production :
  L = 24 → ∀ L', dailyProduction L' ≤ dailyProduction L := sorry

theorem partB_maximize_total_production :
  L = 16 → ∀ L', totalProduction L' ≤ totalProduction L := sorry

theorem partC_maximize_revenue : 
  L = 9 ∧ R = 1650 →
  ∀ L' R', revenue L' 0 4 ≤ revenue L 0 4 ∧ revenue L 0 4 = R := sorry

theorem partD_minimum_exhibition_revenue :
  X = 30 →
  ∀ L x, x >= X -> (revenue 0 L x 4 > revenue L' L x 4) := sorry

end partA_maximize_daily_production_partB_maximize_total_production_partC_maximize_revenue_partD_minimum_exhibition_revenue_l54_54834


namespace vasya_grades_l54_54665

def proves_grades_are_five (grades : List ℕ) : Prop :=
  (grades.length = 5) ∧ (grades.sum = 19) ∧ (grades.sorted.nth 2 = some 4) ∧
  ((grades.count 5) > (grades.count n) for n in (grades.erase_all [5]))

theorem vasya_grades : exists (grades : List ℕ), proves_grades_are_five grades ∧ grades = [2, 3, 4, 5, 5] := 
by
  sorry

end vasya_grades_l54_54665


namespace calvin_bug_collector_l54_54154

theorem calvin_bug_collector :
  ∃ (R : ℕ), 
    let S := 3 in
    let C := R / 2 in
    let K := 2 * S in
    (R + S + C + K = 27 ∧ R = 12) :=
begin
  sorry
end

end calvin_bug_collector_l54_54154


namespace natural_number_solution_l54_54632

noncomputable def find_natural_number (C : ℕ → ℕ) : ℕ :=
  ∃ n : ℕ, (∑ i in finset.range (n - 2), (C (i + 3))^2 = 363) ∧ n = 13

theorem natural_number_solution (C : ℕ → ℕ) :
  (∃ n : ℕ, (∑ i in finset.range (n - 2), (C (i + 3))^2 = 363) ∧ n = 13) :=
sorry

end natural_number_solution_l54_54632


namespace tetrahedron_inradii_inequality_l54_54341

variable {A B C D : Type}
variable {r_A r_B r_C r_D : ℝ}
variable {sum_opposite_sides : A → B → C → D → ℝ}

theorem tetrahedron_inradii_inequality 
    (h_sum_opposite : sum_opposite_sides A B C D = 1) 
    (h_inradii : r_A + r_B + r_C + r_D ≤ (Real.sqrt 3 / 3)) :
    r_A + r_B + r_C + r_D ≤ (Real.sqrt 3 / 3) :=
sorry  -- Proof to be provided.

end tetrahedron_inradii_inequality_l54_54341


namespace sin_double_angle_l54_54566

theorem sin_double_angle (α : ℝ) (h : cos (π / 4 + α) = 2 / 5) : sin (2 * α) = 17 / 25 :=
sorry

end sin_double_angle_l54_54566


namespace beef_weight_before_processing_l54_54477

-- Define the initial weight of the beef.
def W_initial := 1070.5882

-- Define the loss percentages.
def loss1 := 0.20
def loss2 := 0.15
def loss3 := 0.25

-- Define the final weight after all losses.
def W_final := 546.0

-- The main proof goal: show that W_initial results in W_final after considering the weight losses.
theorem beef_weight_before_processing (W_initial W_final : ℝ) (loss1 loss2 loss3 : ℝ) :
  W_final = (1 - loss3) * (1 - loss2) * (1 - loss1) * W_initial :=
by
  sorry

end beef_weight_before_processing_l54_54477


namespace helly_four_convex_sets_helly_n_convex_sets_l54_54458

open Set

variables {α : Type*} [LinearOrderedAddCommGroup α] [TopologicalSpace α] 
variables {C1 C2 C3 C4 : Set α}

/-- Helly's Theorem in the plane for four convex sets -/
theorem helly_four_convex_sets (h1 : Convex ℝ C1) (h2 : Convex ℝ C2) (h3 : Convex ℝ C3) (h4 : Convex ℝ C4) 
  (h123 : (C1 ∩ C2 ∩ C3).Nonempty) (h124 : (C1 ∩ C2 ∩ C4).Nonempty)
  (h134 : (C1 ∩ C3 ∩ C4).Nonempty) (h234 : (C2 ∩ C3 ∩ C4).Nonempty) : 
  (C1 ∩ C2 ∩ C3 ∩ C4).Nonempty :=
sorry

/-- Helly's Theorem in the plane for any n ≥ 4 convex sets -/
theorem helly_n_convex_sets {n : ℕ} (h_conv : ∀ (i : Fin n), Convex ℝ (C i)) 
  (h_inter : ∀ {k : ℕ} (hk : 3 ≤ k . h) [IndexedFamily.PointwiseIntersection (Fin k) (Fin n) Convex ℝ]) : 
  4 ≤ n → (⋂ i, C i).Nonempty :=
sorry

end helly_four_convex_sets_helly_n_convex_sets_l54_54458


namespace minimum_shots_to_hit_ship_l54_54746

theorem minimum_shots_to_hit_ship (grid : Matrix Nat 8 8)
    (ship : Matrix Nat 1 3)
    (shots : Fin 4 → Fin (8 * 2))
    (valid_shot : ∀ (i : Fin 4), ∃ r c : Fin 8, shots i = r ∨ shots i = 8 + c) :
    (∀ (placement : Fin 8 × Fin 8) (orientation : Bool),
        (∃ hit : Fin 4, 
            ∃ (i j : Fin 3),
                if orientation 
                then ship 0 j + placement.snd = ship 0 0 + j + placement.snd ∧ (shots hit = placement.fst ∨ shots hit = 8 + ship 0 j + placement.snd)
                else ship 0 j + placement.fst = ship 0 0 + j + placement.fst ∧ (shots hit = ship 0 j + placement.fst ∨ shots hit = 8 + placement.snd)
        )
    ) :=
sorry

end minimum_shots_to_hit_ship_l54_54746


namespace no_valid_arrangement_l54_54314

theorem no_valid_arrangement :
  ∀ (table : Fin 300 → Fin 300 → Int), 
  (∀ i j, table i j = 1 ∨ table i j = -1) →
  abs (∑ i j, table i j) < 30000 →
  (∀ i j, abs (∑ x in Finset.finRange 3, ∑ y in Finset.finRange 5, table (i + x) (j + y)) > 3) →
  (∀ i j, abs (∑ x in Finset.finRange 5, ∑ y in Finset.finRange 3, table (i + x) (j + y)) > 3) →
  False :=
by
  intros table h_entries h_total_sum h_rect_sum_3x5 h_rect_sum_5x3
  sorry

end no_valid_arrangement_l54_54314


namespace geometric_sequence_sum_l54_54605

theorem geometric_sequence_sum :
  ∀ {a : ℕ → ℝ} (r : ℝ),
    (∀ n, a (n + 1) = r * a n) →
    a 1 + a 2 = 1 →
    a 3 + a 4 = 4 →
    a 5 + a 6 + a 7 + a 8 = 80 :=
by
  intros a r h_geom h_sum_1 h_sum_2
  sorry

end geometric_sequence_sum_l54_54605


namespace clock_diff_at_least_one_minute_51_seconds_at_least_once_clock_diff_never_exactly_two_minutes_l54_54416

/-- Two chess players have played a game with 40 moves each. Both clocks show an equal time of 2 hours and 30 minutes at the end. -/
variable (clock_1 clock_2: ℕ → ℕ)

axiom moves_40_equal_time : clock_1 40 = clock_2 40 ∧ clock_1 40 = 150 ∧ clock_2 40 = 150

/-- There exists a moment during the game when the clock difference was at least 1 minute and 51 seconds. -/
theorem clock_diff_at_least_one_minute_51_seconds_at_least_once :
  ∃ n, n ≤ 40 ∧ |clock_1 n - clock_2 n| ≥ 1 + (51 / 60 : ℝ) := sorry

/-- At no point during the game is the difference in the clock readings exactly 2 minutes. -/
theorem clock_diff_never_exactly_two_minutes :
  ∀ n, n ≤ 40 → |clock_1 n - clock_2 n| ≠ 2 := sorry

end clock_diff_at_least_one_minute_51_seconds_at_least_once_clock_diff_never_exactly_two_minutes_l54_54416


namespace bus_speed_including_stoppages_is_12_l54_54533

variable (speed_excluding_stoppages : ℕ) (stoppage_time_per_hr : ℕ)

def speed_including_stoppages 
  (speed_excluding_stoppages = 48)
  (stoppage_time_per_hr = 45) : ℕ :=
  12

theorem bus_speed_including_stoppages_is_12 :
  speed_including_stoppages 48 45 = 12 := by
  sorry

end bus_speed_including_stoppages_is_12_l54_54533


namespace right_angle_triangle_cond_a_right_angle_triangle_cond_b_right_angle_triangle_cond_c_right_angle_triangle_cond_d_option_a_is_correct_l54_54440

theorem right_angle_triangle_cond_a (a b c : ℕ) :
  ((a = 3^2 ∧ b = 4^2 ∧ c = 5^2) ∧ (a*a + b*b ≠ c*c)) :=
begin
  sorry,
end

theorem right_angle_triangle_cond_b (a b c : ℕ) :
  (a = 5 ∧ b = 12 ∧ c = 13) →
  (a*a + b*b = c*c) :=
begin
  sorry,
end

theorem right_angle_triangle_cond_c (a b c : ℕ) :
  (a = 7 ∧ b = 24 ∧ c = 25) →
  (a*a + b*b = c*c) :=
begin
  sorry,
end

theorem right_angle_triangle_cond_d (a b c : ℕ) :
  (a = 1 ∧ b = 2 ∧ c = 3) →
  (a*a + b*b = c*c) :=
begin
  sorry,
end

theorem option_a_is_correct :
  (∃ (a b c : ℕ), (a = 3^2 ∧ b = 4^2 ∧ c = 5^2) ∧ (a*a + b*b ≠ c*c)) :=
begin
  use [3^2, 4^2, 5^2],
  split,
  { split; refl },
  { exact right_angle_triangle_cond_a 3^2 4^2 5^2 },
end

end right_angle_triangle_cond_a_right_angle_triangle_cond_b_right_angle_triangle_cond_c_right_angle_triangle_cond_d_option_a_is_correct_l54_54440


namespace all_meet_time_is_1pm_l54_54144

def minutes_in_a_day : ℕ := 1440
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

def ben_lap_time : ℕ := 5
def clara_lap_time : ℕ := 9
def david_lap_time : ℕ := 8

def ben_clara_david_lcm : ℕ := lcm ben_lap_time (lcm clara_lap_time david_lap_time)

def start_time_in_minutes : ℕ := 7 * 60 -- 7:00 AM in minutes

def meet_time_in_minutes (start : ℕ) (lcm_val : ℕ) : ℕ :=
  (start + lcm_val) % minutes_in_a_day

theorem all_meet_time_is_1pm :
  meet_time_in_minutes start_time_in_minutes ben_clara_david_lcm = 13 * 60 := 
by
  sorry

end all_meet_time_is_1pm_l54_54144


namespace cost_function_property_l54_54777

variable {R : Type*} [Ring R]

theorem cost_function_property (f : R → R) (h : ∀ b : R, f(2 * b) = 16 * f(b)) : true :=
sorry

end cost_function_property_l54_54777


namespace problem1_problem2_l54_54809

noncomputable def O := (0 : ℝ, 0 : ℝ, 0 : ℝ)
noncomputable def O1 := (0 : ℝ, 1 : ℝ, sqrt 3 : ℝ)
noncomputable def A := (sqrt 3 : ℝ, 0 : ℝ, 0 : ℝ)
noncomputable def A1 := (sqrt 3 : ℝ, 1 : ℝ, sqrt 3 : ℝ)
noncomputable def B := (0 : ℝ, 2 : ℝ, 0 : ℝ)

noncomputable def angle_O1OB : ℝ := 60
noncomputable def angle_AOB : ℝ := 90
noncomputable def OB := (2 : ℝ)
noncomputable def OO1 := (2 : ℝ)
noncomputable def OA := (sqrt 3: ℝ)

def dihedral_angle_O1_AB_O : ℝ := Real.arccos (sqrt 2 / 4)
def distance_A1B_OA : ℝ := sqrt 3

theorem problem1 (cond1 : ℝ := angle_O1OB) 
                (cond2 : ℝ := angle_AOB) 
                (cond3 : ℝ := OB) 
                (cond4 : ℝ := OO1) 
                (cond5 : ℝ := OA) :
                dihedral_angle_O1_AB_O = Real.arccos (sqrt 2 / 4) :=
sorry

theorem problem2 (cond1 : ℝ := angle_O1OB) 
                (cond2 : ℝ := angle_AOB) 
                (cond3 : ℝ := OB) 
                (cond4 : ℝ := OO1) 
                (cond5 : ℝ := OA) :
                distance_A1B_OA = sqrt 3 :=
sorry

end problem1_problem2_l54_54809


namespace estimate_m_l54_54216

noncomputable def m : ℝ := Real.sqrt 4 + Real.sqrt 3

theorem estimate_m : 3 < m ∧ m < 4 :=
by
  have h := Real.sqrt_sub_eq_rpow
  sorry

end estimate_m_l54_54216


namespace probability_eq_two_thirds_l54_54881

noncomputable def probability_real_roots : ℝ :=
  let interval := 2 * Real.pi in
  let endpoints := (Real.pi / 6, 7 * Real.pi / 6) in
  let range_length := (endpoints.2 - endpoints.1) / interval in
  range_length

theorem probability_eq_two_thirds
  (α : ℝ)
  (hα : 0 ≤ α ∧ α ≤ 2 * Real.pi)
  (h : ∃ x:ℝ, x^2 - 4 * x * Real.sin α + 1 = 0) : 
  probability_real_roots = 2 / 3 :=
by
  sorry

end probability_eq_two_thirds_l54_54881


namespace arrangement_A_not_middle_or_ends_arrangement_A_B_ends_arrangement_grouped_arrangement_alternate_arrangement_fixed_sequence_l54_54407

-- Problem 1
theorem arrangement_A_not_middle_or_ends (n : ℕ) (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 5) : 
  let total := boys + girls + 1 in n = 6 * Nat.factorial 8 := 
by
  sorry

-- Problem 2
theorem arrangement_A_B_ends (n : ℕ) (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 5) :
  let total := boys + girls + 1 in n = 2 * Nat.factorial 7 :=
by
  sorry

-- Problem 3
theorem arrangement_grouped (n : ℕ) (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 5) :
  n = 2 * Nat.factorial boys * Nat.factorial girls :=
by
  sorry

-- Problem 4
theorem arrangement_alternate (n : ℕ) (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 5) :
  n = 5 * 4 * 4 * 3 * 3 * 2 * 2 * 1 * 1 :=
by
  sorry

-- Problem 5
theorem arrangement_fixed_sequence (n : ℕ) (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 5) :
  let total := boys + girls + 1 in n = 7 * Nat.factorial 6 :=
by
  sorry

end arrangement_A_not_middle_or_ends_arrangement_A_B_ends_arrangement_grouped_arrangement_alternate_arrangement_fixed_sequence_l54_54407


namespace tan_theta_of_obtuse_angle_l54_54607

noncomputable def theta_expression (θ : Real) : Complex :=
  Complex.mk (3 * Real.sin θ) (Real.cos θ)

theorem tan_theta_of_obtuse_angle {θ : Real} (h_modulus : Complex.abs (theta_expression θ) = Real.sqrt 5) 
  (h_obtuse : π / 2 < θ ∧ θ < π) : Real.tan θ = -1 := 
  sorry

end tan_theta_of_obtuse_angle_l54_54607


namespace running_minutes_l54_54890

theorem running_minutes
  (r w : ℕ)
  (h1 : 10 * r + 4 * w = 450)
  (h2 : r + w = 60) :
  r = 35 :=
begin
  sorry
end

end running_minutes_l54_54890


namespace distribute_problems_l54_54883

theorem distribute_problems :
  (12 ^ 6) = 2985984 := by
  sorry

end distribute_problems_l54_54883


namespace problem_solution_l54_54088

noncomputable def proof_problem (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) : Prop :=
  ((x1^2 - x3 * x5) * (x2^2 - x3 * x5) ≤ 0) ∧
  ((x2^2 - x4 * x1) * (x3^2 - x4 * x1) ≤ 0) ∧
  ((x3^2 - x5 * x2) * (x4^2 - x5 * x2) ≤ 0) ∧
  ((x4^2 - x1 * x3) * (x5^2 - x1 * x3) ≤ 0) ∧
  ((x5^2 - x2 * x4) * (x1^2 - x2 * x4) ≤ 0) → 
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5

theorem problem_solution (x1 x2 x3 x4 x5 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  proof_problem x1 x2 x3 x4 x5 h1 h2 h3 h4 h5 :=
  by
    sorry

end problem_solution_l54_54088


namespace line_parallel_perp_imp_perp_l54_54237

-- Definitions:
variables {Line Plane : Type}
variables (m n : Line) (α : Plane)

-- Conditions:
variable [diff_lines : m ≠ n]
variable [line_parallel : m ∥ n]
variable [line_perp_plane : n ⟂ α]

-- The Proposition to Prove:
theorem line_parallel_perp_imp_perp (m_parallel_n : m ∥ n) (n_perp_α : n ⟂ α) : m ⟂ α := 
by
  sorry

end line_parallel_perp_imp_perp_l54_54237


namespace triangle_sinB_and_area_l54_54580

theorem triangle_sinB_and_area (a b c : ℝ) (A B C : ℝ)
  (triangle : IsTriangle a b c A B C)
  (area_eq : S = (a^2 + b^2 - c^2) / 4)
  (sin_A_eq : sin A = 3 / 5)
  (c_eq : c = 5) :
  sin B = 7 * sqrt 2 / 10 ∧ S = 21 / 2 :=
by
  sorry

end triangle_sinB_and_area_l54_54580


namespace selling_price_of_one_bag_l54_54422

theorem selling_price_of_one_bag :
  ∀ (total_harvested total_juice total_restaurant total_revenue per_bag weight_bags : ℕ),
    total_harvested = 405 →
    total_juice = 90 →
    total_restaurant = 60 →
    total_revenue = 408 →
    per_bag = 5 →
    weight_bags = (total_harvested - (total_juice + total_restaurant)) →
    (weight_bags / per_bag) = 51 →
    (total_revenue / (weight_bags / per_bag)) = 8 :=
by
  intros total_harvested total_juice total_restaurant total_revenue per_bag weight_bags
  assume h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5] at h6
  rw [h6] at h7
  exact h7.symm

end selling_price_of_one_bag_l54_54422


namespace find_n_l54_54272

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem find_n (n : ℤ) (h : ∃ x, n < x ∧ x < n+1 ∧ f x = 0) : n = 2 :=
sorry

end find_n_l54_54272


namespace number_of_positive_integers_solution_count_number_of_positive_integers_solution_l54_54196

theorem number_of_positive_integers_solution (n : ℕ) :
  (∃ x : ℤ, n * x - 30 = 5 * n) ↔ 
  n ∈ {d ∈ finset.range 31 | 30 % d = 0} :=
by {
  -- Proof omitted
  sorry
}

theorem count_number_of_positive_integers_solution :
  finset.card {d ∈ finset.range 31 | 30 % d = 0} = 8 :=
by {
  -- Proof omitted
  sorry
}

end number_of_positive_integers_solution_count_number_of_positive_integers_solution_l54_54196


namespace remainder_when_2n_divided_by_4_l54_54082

theorem remainder_when_2n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (2 * n) % 4 = 2 :=
by
  sorry

end remainder_when_2n_divided_by_4_l54_54082


namespace solve_for_y_l54_54934

-- The given condition as a hypothesis
variables {x y : ℝ}

-- The theorem statement
theorem solve_for_y (h : 3 * x - y + 5 = 0) : y = 3 * x + 5 :=
sorry

end solve_for_y_l54_54934


namespace units_digit_sum_of_factorials_l54_54454

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_sum_of_factorials :
  ones_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
              factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) = 3 := 
sorry

end units_digit_sum_of_factorials_l54_54454


namespace ratio_of_pencils_l54_54926

theorem ratio_of_pencils 
  (M : ℕ) 
  (H1 : ∀ D, D = 3 * M) 
  (H2 : M + 3 * M = 480) 
  (H3 : ∀ C, C = 30 / 0.5) : 
  (M / (30 / 0.5) = 2) := 
by
  sorry

end ratio_of_pencils_l54_54926


namespace volume_of_remaining_solid_l54_54854

noncomputable def volume_cube_with_cylindrical_hole 
  (side_length : ℝ) (hole_diameter : ℝ) (π : ℝ := 3.141592653589793) : ℝ :=
  let V_cube := side_length^3
  let radius := hole_diameter / 2
  let height := side_length
  let V_cylinder := π * radius^2 * height
  V_cube - V_cylinder

theorem volume_of_remaining_solid 
  (side_length : ℝ)
  (hole_diameter : ℝ)
  (h₁ : side_length = 6) 
  (h₂ : hole_diameter = 3)
  (π : ℝ := 3.141592653589793) : 
  abs (volume_cube_with_cylindrical_hole side_length hole_diameter π - 173.59) < 0.01 :=
by
  sorry

end volume_of_remaining_solid_l54_54854


namespace prime_factors_n_l54_54081

open Int

theorem prime_factors_n (n : ℕ) (h1 : n > 0) (h2 : n < 200) (h3 : (14 * n) % 60 = 0) : 
  (nat.factors n).to_finset.card = 3 :=
sorry

end prime_factors_n_l54_54081


namespace curve_intersects_at_point_2_3_l54_54491

open Real

theorem curve_intersects_at_point_2_3 :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
                 (t₁^2 - 4 = t₂^2 - 4) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = t₂^3 - 6 * t₂ + 3) ∧ 
                 (t₁^2 - 4 = 2) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = 3) :=
by
  sorry

end curve_intersects_at_point_2_3_l54_54491


namespace exists_nonzero_multiple_of_k_with_four_distinct_digits_l54_54764

theorem exists_nonzero_multiple_of_k_with_four_distinct_digits (k : ℤ) (h1 : 1 < k) : 
  ∃ m : ℤ, (m ≠ 0) ∧ (m < k^4) ∧ (∀ d ∈ List.ofDigits 10 m, d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 9) := 
sorry

end exists_nonzero_multiple_of_k_with_four_distinct_digits_l54_54764


namespace cyclic_quadrilateral_iff_perpendicular_diagonals_l54_54342

open EuclideanGeometry

noncomputable def is_cyclic_quad (A_1 B_1 C_1 D_1 : Point) : Prop :=
∃ (circ : Circle), circ ∈ [A_1, B_1, C_1, D_1]

theorem cyclic_quadrilateral_iff_perpendicular_diagonals
  (A B C D O : Point)
  (h_quad : quadrilateral A B C D)
  (h_O_is_intersection : ∃ (E : Line), is_intersection E (AC A C) ∧ is_intersection E (BD B D))
  (A' B' C' D' : Point)
  (h_rotate : ∃ θ : Real, rotation_about O θ (quad_to_points h_quad) = quad_to_points' )
  (A_1 B_1 C_1 D_1 : Point)
  (h_intersections : 
    A_1 = line_intersection (A'B') (AB) ∧ 
    B_1 = line_intersection (B'C') (BC) ∧ 
    C_1 = line_intersection (C'D') (CD) ∧ 
    D_1 = line_intersection (D'A') (DA)) : 
  is_cyclic_quad A_1 B_1 C_1 D_1 ↔ perp AC BD :=
sorry -- proof skipped

end cyclic_quadrilateral_iff_perpendicular_diagonals_l54_54342


namespace octahedron_side_length_l54_54880

open Real

theorem octahedron_side_length :
  let A1  : ℝ × ℝ × ℝ := (0, 0, 0),
      A1' : ℝ × ℝ × ℝ := (1, 1, 1),
      A2  : ℝ × ℝ × ℝ := (1, 0, 0),
      A3  : ℝ × ℝ × ℝ := (0, 1, 0),
      A4  : ℝ × ℝ × ℝ := (0, 0, 1),
      A2' : ℝ × ℝ × ℝ := (0, 1, 1),
      A3' : ℝ × ℝ × ℝ := (1, 0, 1),
      A4' : ℝ × ℝ × ℝ := (1, 1, 0),
      -- Vertices of the octahedron
      V1 : ℝ × ℝ × ℝ := (1/3, 0, 0),
      V2 : ℝ × ℝ × ℝ := (0, 1/3, 0),
      V3 : ℝ × ℝ × ℝ := (0, 0, 1/3),
      U1 : ℝ × ℝ × ℝ := (2/3, 2/3, 2/3),
      U2 : ℝ × ℝ × ℝ := (2/3, 2/3, 0),
      U3 : ℝ × ℝ × ℝ := (2/3, 0, 2/3)
  in dist (1/3, 0, 0) (2/3, 2/3, 2/3) = 1 :=
by
  sorry

end octahedron_side_length_l54_54880


namespace fraction_of_reciprocal_l54_54419

theorem fraction_of_reciprocal (x : ℝ) (f : ℝ) (h_pos : 0 < x) (h_eq : (2 / 3) * x = f * (1 / x))
  (h_x : x = 0.4166666666666667) : f = 0.2777777777777778 / 2.4 :=
by {
  sorry,
}

end fraction_of_reciprocal_l54_54419


namespace luca_drink_cost_l54_54738

theorem luca_drink_cost (sandwich_price : ℕ) (coupon_fraction : ℚ) (avocado_extra : ℕ) (salad_cost : ℕ) (total_bill : ℕ) (drink_cost: ℕ) 
  (h_sandwich_price : sandwich_price = 8)
  (h_coupon_fraction : coupon_fraction = 1 / 4)
  (h_avocado_extra : avocado_extra = 1)
  (h_salad_cost : salad_cost = 3)
  (h_total_bill : total_bill = 12)
  (h_drink_cost : drink_cost = 2)
  (h : total_bill = sandwich_price * (1 - coupon_fraction).toNat + avocado_extra + salad_cost + drink_cost) :
  drink_cost = 2 := sorry

end luca_drink_cost_l54_54738


namespace find_n_tan_eq_l54_54545

theorem find_n_tan_eq (n : ℤ) (h₁ : -180 < n) (h₂ : n < 180) 
  (h₃ : Real.tan (n * (Real.pi / 180)) = Real.tan (276 * (Real.pi / 180))) : 
  n = 96 :=
sorry

end find_n_tan_eq_l54_54545


namespace operation_evaluation_l54_54512

def operation (x y : ℕ) : ℕ := x * y - 3 * x

theorem operation_evaluation : (operation 6 4) - (operation 4 6) = -6 := by
  sorry

end operation_evaluation_l54_54512


namespace max_value_sum_of_first_n_terms_l54_54966

noncomputable def max_sum_of_arithmetic_sequence_lt (a : ℕ → ℝ) (d : ℝ) (n : ℕ) : ℝ := 
  n * (a 1) + (n * (n - 1) / 2) * d

theorem max_value_sum_of_first_n_terms
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : 3 * a 7 = 5 * a 13)
  (h2 : cos (a 4)^2 - cos (a 4)^2 * sin (a 7)^2 + 
        sin (a 4)^2 * cos (a 7)^2 - sin (a 4)^2 = - cos (a 5 + a 6))
  (h3 : 0 < d ∧ d < 2) :
  ∃ n : ℕ, max_sum_of_arithmetic_sequence_lt a d n = 77 * π := by
  sorry

end max_value_sum_of_first_n_terms_l54_54966
