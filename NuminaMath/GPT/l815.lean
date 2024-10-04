import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.FactorialRing
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.GeomSeries
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Archimedean
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binom
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.Geometry.Geometry
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.Init.Data.Int.Basic
import Mathlib.LinearAlgebra.EigenVector
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.Order.Floor
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityLemma1
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.SolveByElim
import Mathlib.Topology.Instances.Real

namespace find_x_intercept_l815_815347

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := 4 * x + 7 * y = 28

-- Define the x-intercept point when y = 0
def x_intercept (x : ℝ) : Prop := line_eq x 0

-- Prove that for the x-intercept, when y = 0, x = 7
theorem find_x_intercept : x_intercept 7 :=
by
  -- proof would go here
  sorry

end find_x_intercept_l815_815347


namespace certain_event_triangle_interior_angles_l815_815633

theorem certain_event_triangle_interior_angles (T : Type) [Triangle T] :
  (∀ (t : T), sum_interior_angles t = 180) :=
sorry

end certain_event_triangle_interior_angles_l815_815633


namespace solve_for_x_l815_815148

theorem solve_for_x (x : ℝ) (h1 : 5^(2*x) = real.sqrt 125) : x = 3 / 4 :=
sorry

end solve_for_x_l815_815148


namespace has_maximum_value_l815_815903

def f (x : ℝ) := 2 * x + (1 / x) - 1

theorem has_maximum_value : 
  ∃ M, ∀ x < 0, f x ≤ M :=
sorry

end has_maximum_value_l815_815903


namespace count_valid_duck_words_l815_815488

-- Definition of letters used in duck language
inductive DuckLetter
| q : DuckLetter
| a : DuckLetter
| k : DuckLetter

-- Definition of vowels and consonants
def is_vowel : DuckLetter → Prop
| DuckLetter.a := true
| _ := false

def is_consonant : DuckLetter → Prop
| DuckLetter.q := true
| DuckLetter.k := true
| _ := false

-- Function to check if a list of DuckLetters is a valid duck language word
def is_valid_duck_word (word : List DuckLetter) : Prop :=
  word.length = 4 ∧ ∀ i, i < word.length - 1 → 
  not (is_consonant (word.get ⟨i, (Nat.lt_trans (Nat.lt_add_of_lt_sub_right (nat.lt_of_succ_lt_succ (le_refl (3 : ℕ))) (by decide)) (le_refl 3))⟩) ∧ 
       is_consonant (word.get ⟨i+1, (Nat.succ_lt_succ (Nat.lt_of_succ_lt_succ (nat.lt_of_succ_lt_succ (le_refl 2)))⟩)))

-- Finite set of all possible duck words
def all_duck_words : Finset (List DuckLetter) := 
  (Finset.univ : Finset (List DuckLetter)).filter is_valid_duck_word

-- Theorem to prove the number of valid duck words
theorem count_valid_duck_words : all_duck_words.card = 21 :=
sorry

end count_valid_duck_words_l815_815488


namespace last_two_digits_of_sum_of_factorials_l815_815215

-- Problem statement: Sum of factorials from 1 to 15
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.factorial k)

-- Define the main problem
theorem last_two_digits_of_sum_of_factorials : 
  (sum_factorials 15) % 100 = 13 :=
by 
  sorry

end last_two_digits_of_sum_of_factorials_l815_815215


namespace bicycles_purchased_on_Friday_l815_815136

theorem bicycles_purchased_on_Friday (F : ℕ) : (F - 10) - 4 + 2 = 3 → F = 15 := by
  intro h
  sorry

end bicycles_purchased_on_Friday_l815_815136


namespace find_a_l815_815192

open Real

noncomputable def tangent_line_slope (x : ℝ) : ℝ := 1 + log x

theorem find_a (a : ℝ) (h : tangent_line_slope exp(1) = 2) (h_perp : 2 * (-1 / a) = -1) : a = 2 :=
by
  sorry

end find_a_l815_815192


namespace sum_of_intersections_l815_815171

noncomputable def line_intersects := 
  ∃ (l : Real → Real) (k x1 x2 x3 : ℝ), 
    (k ≠ 0) ∧
    (∀ x, l x = k / x → (x = x1 ∨ x = x2)) ∧
    (l x3 = 0) ∧
    (x1 + x2 = x3)

theorem sum_of_intersections (k x1 x2 x3 : ℝ) (h : line_intersects) :
  k ≠ 0 → (∀ x, (∃ l : Real → Real, l x = k / x → (x = x1 ∨ x = x2)) ∧ l x3 = 0) → 
  x1 + x2 = x3 :=
by 
  sorry

end sum_of_intersections_l815_815171


namespace commutative_star_l815_815721

def star (a b : ℤ) : ℤ := a^2 + b^2

theorem commutative_star (a b : ℤ) : star a b = star b a :=
by sorry

end commutative_star_l815_815721


namespace probability_of_red_card_l815_815842

theorem probability_of_red_card (successful_attempts not_successful_attempts : ℕ) (h : successful_attempts = 5) (h2 : not_successful_attempts = 8) : (successful_attempts / (successful_attempts + not_successful_attempts) : ℚ) = 5 / 13 := by
  sorry

end probability_of_red_card_l815_815842


namespace members_even_and_divisible_l815_815081

structure ClubMember (α : Type) := 
  (friend : α) 
  (enemy : α)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

def can_be_divided_into_two_subclubs (members : List (ClubMember Nat)) : Prop :=
sorry -- Definition of dividing into two subclubs here

theorem members_even_and_divisible (members : List (ClubMember Nat)) :
  is_even members.length ∧ can_be_divided_into_two_subclubs members :=
sorry

end members_even_and_divisible_l815_815081


namespace larger_number_is_23_l815_815985

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l815_815985


namespace log_equation_l815_815047

theorem log_equation (y : ℝ) : log8 (y - 4) = 1.5 → y = 16 * sqrt 2 + 4 :=
by
  sorry

end log_equation_l815_815047


namespace number_of_teams_l815_815847

theorem number_of_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end number_of_teams_l815_815847


namespace grid_becomes_black_l815_815262

noncomputable def prob_black_grid : ℚ := 6561 / 65536

theorem grid_becomes_black :
  let grid := (Array (Array Bool)) in
  let painted_randomly := ∀ (i j : Fin 4), (grid[i][j] = true) ∨ (grid[i][j] = false) in
  let rotated := ∀ (i j : Fin 4), grid[i][j] = grid[3 - i][3 - j] in
  let repaint := ∀ (i j : Fin 4), (grid[i][j] = true) ∨ (grid[i][j] = false) → (grid[i][j] ∨ grid[3 - i][3 - j] = false) in
  (rotated → repaint) → prob_black_grid = 6561 / 65536 :=
begin
  sorry
end

end grid_becomes_black_l815_815262


namespace tangent_line_x_squared_at_one_one_l815_815343

open Real

theorem tangent_line_x_squared_at_one_one :
  ∀ (x y : ℝ), y = x^2 → (x, y) = (1, 1) → (2 * x - y - 1 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_x_squared_at_one_one_l815_815343


namespace am_gm_equality_l815_815520

theorem am_gm_equality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end am_gm_equality_l815_815520


namespace num_possible_route_numbers_l815_815162

def seven_segment_possible_numbers : list (list char) :=
  [['2', '5', '7'],      -- possible transformations for '3'
   ['3', '6', '9'],      -- possible transformations for '5'
   []]                   -- no transformations for the essential '1' segments

def possible_route_numbers (digits : list char) (malfunction_index : ℕ) : list (list char) :=
  sorry  -- Implementation skipped

theorem num_possible_route_numbers : 
  (possible_route_numbers ['3', '5', '1'] 1).length = 5 :=
by
  sorry

end num_possible_route_numbers_l815_815162


namespace probability_even_sum_l815_815938

def tiles := fin 10
def players := fin 3

-- Each player selects 3 tiles from the 10 numbered tiles
def selects (p : players) : set tiles := {x | p = p} -- Placeholder since we're not defining the actual selection explicitly

-- Probability calculation of all three players obtaining an even sum
def prob_even_sum (s1 s2 s3 : set tiles) : ℚ :=
  if (all_even_sum s1 s2 s3)
  then rational.mk 1 28
  else 0

-- Function to determine if all selected sets sum to an even number
def all_even_sum (s1 s2 s3 : set tiles) : Prop :=
  even_sum s1 ∧ even_sum s2 ∧ even_sum s3

-- Function to determine if a set's sum is even
def even_sum (s : set tiles) : Prop :=
  ∑ x in s, x.val % 2 = 0

-- m and n such that the probability is m/n and they are relatively prime positive integers
def m := 1
def n := 28

-- Final statement
theorem probability_even_sum (s1 s2 s3 : set tiles) :
  prob_even_sum s1 s2 s3 = 1 / 28 :=
by sorry

end probability_even_sum_l815_815938


namespace analogical_reasoning_correct_l815_815629

variable (a b c : Real)

theorem analogical_reasoning_correct (h : c ≠ 0) (h_eq : (a + b) * c = a * c + b * c) : 
  (a + b) / c = a / c + b / c :=
  sorry

end analogical_reasoning_correct_l815_815629


namespace expected_value_of_winnings_l815_815678

theorem expected_value_of_winnings : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probabilities := [1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8] -- uniform distribution over an 8-sided die
  let winnings := λ (x : ℕ), if x == 3 then 3
                             else if x == 6 then 6
                             else if x == 2 then -2
                             else if x == 4 then -2
                             else 0 in 
  let expected_value := ∑ i in (finset.range 8).map (function.embedding.mk (λ i, i + 1) sorry),
                           (probabilities[i] * (winnings outcomes[i])) in
  expected_value = 5 / 4 := 
begin
  sorry
end

end expected_value_of_winnings_l815_815678


namespace Jeff_total_vehicles_l815_815880

variable (T : Nat)

theorem Jeff_total_vehicles (h : Jeff has twice as many cars as trucks)
    (k : Jeff has T trucks) :
    total_vehicles = 3 * T := 
  sorry

end Jeff_total_vehicles_l815_815880


namespace cosine_angle_between_planes_l815_815899

noncomputable def normal_vector_plane_1 : ℝ × ℝ × ℝ := (3, 2, -1)
noncomputable def normal_vector_plane_2 : ℝ × ℝ × ℝ := (9, 6, 3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1 * u.1 + u.2 * u.2 + u.3 * u.3)

theorem cosine_angle_between_planes :
  let n1 := normal_vector_plane_1
  let n2 := normal_vector_plane_2
  (dot_product n1 n2) / ((magnitude n1) * (magnitude n2)) = 6 / 7 :=
by
  sorry

end cosine_angle_between_planes_l815_815899


namespace minimum_overlapping_area_of_folded_stripe_l815_815702

open Real

theorem minimum_overlapping_area_of_folded_stripe :
  ∀ (rectangular_stripe_width : ℝ) (overlapping_area : ℝ), rectangular_stripe_width = 3 → 
  overlapping_area = 9 / 2 → ∃ (θ : ℝ), (0 < θ) ∧ (θ < pi / 2) → 
  (∃ tan_θ : ℝ, tan θ = tan_θ ∧ tan_θ + 1/tan_θ = 2) :=
by {
  intros rectangular_stripe_width overlapping_area w3 a45 hθ,
  sorry
}

end minimum_overlapping_area_of_folded_stripe_l815_815702


namespace ratio_expression_l815_815045

theorem ratio_expression (p q s u : ℚ) (h1 : p / q = 3 / 5) (h2 : s / u = 8 / 11) : 
  (4 * p * s - 3 * q * u) / (5 * q * u - 8 * p * s) = -69 / 83 :=
by
  sorry

end ratio_expression_l815_815045


namespace tangent_line_when_a_eq_2_decreasing_interval_implies_a_le_minus2_l815_815819

noncomputable def f (x a : ℝ) := x^3 + a*x^2 - 4*x + 3
noncomputable def f_prime (x a : ℝ) := 3*x^2 + 2*a*x - 4

theorem tangent_line_when_a_eq_2 :
  let a := 2 in
  let f_1 := f 1 a in
  let f_prime_1 := f_prime 1 a in
  3*x - y - 1 = 0 := sorry

theorem decreasing_interval_implies_a_le_minus2 :
  (∀ x : ℝ, 1 < x ∧ x < 2 → f_prime x a ≤ 0) → a ≤ -2 := sorry

end tangent_line_when_a_eq_2_decreasing_interval_implies_a_le_minus2_l815_815819


namespace maggie_total_spent_l815_815909

theorem maggie_total_spent :
  let cost_plants := 20 * 25
  let cost_fish := 7 * 30
  let cost_magazines := 25 * 5
  let total_cost := cost_plants + cost_fish + cost_magazines
  let discount := 0.10 * total_cost
  let final_cost := total_cost - discount
  final_cost = 751.50 :=
by
  let cost_plants := 20 * 25
  let cost_fish := 7 * 30
  let cost_magazines := 25 * 5
  let total_cost := cost_plants + cost_fish + cost_magazines
  let discount := 0.10 * total_cost
  let final_cost := total_cost - discount
  have cost_plants_eq : cost_plants = 500 := by sorry
  have cost_fish_eq : cost_fish = 210 := by sorry
  have cost_magazines_eq : cost_magazines = 125 := by sorry
  have total_cost_eq : total_cost = 835 := by sorry
  have discount_eq : discount = 83.50 := by sorry
  have final_cost_eq : final_cost = 751.50 := by sorry
  exact final_cost_eq

end maggie_total_spent_l815_815909


namespace larger_number_is_23_l815_815981

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l815_815981


namespace max_range_f_plus_2g_l815_815566

noncomputable def max_val_of_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) : ℝ :=
  9

theorem max_range_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) :
  ∃ (a b : ℝ), (-3 ≤ a ∧ a ≤ 5) ∧ (-8 ≤ b ∧ b ≤ 4) ∧ b = 9 := 
sorry

end max_range_f_plus_2g_l815_815566


namespace problem1_center_of_curvature_problem2_center_of_curvature_l815_815339

-- Problem 1
def curve1 : ℝ → ℝ := λ x => 4 * x - x^2

-- Problem 2
def parametric_x (t : ℝ) : ℝ := t - sin t
def parametric_y (t : ℝ) : ℝ := 1 - cos t

theorem problem1_center_of_curvature :
  -- Conditions
  let vertex_x := 2
  let vertex_y := 4
  let second_derivative := -2
  let curvature_center : ℝ × ℝ := (vertex_x - (1 + (0)^2) / second_derivative * 0, vertex_y + 1 / second_derivative)
  -- Question == Answer
  curvature_center = (2, 7 / 2) := sorry

theorem problem2_center_of_curvature :
  -- Conditions
  let t := π / 2
  let dot_x := 1 - cos t
  let dot_y := sin t
  let ddot_x := sin t
  let ddot_y := cos t
  let curvature_center : ℝ × ℝ := (parametric_x t - (dot_x^2 + dot_y^2) / (dot_x * ddot_y - dot_y * ddot_x) * dot_y,
                                    parametric_y t + (dot_x^2 + dot_y^2) / (dot_x * ddot_y - dot_y * ddot_x) * dot_x)
  -- Question == Answer
  curvature_center = (π / 2 + 1, -1) := sorry

end problem1_center_of_curvature_problem2_center_of_curvature_l815_815339


namespace possible_rectangular_arrays_l815_815481

theorem possible_rectangular_arrays (n : ℕ) (h : n = 48) :
  ∃ (m k : ℕ), m * k = n ∧ 2 ≤ m ∧ 2 ≤ k :=
sorry

end possible_rectangular_arrays_l815_815481


namespace larger_number_l815_815972

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l815_815972


namespace larger_number_is_23_l815_815980

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815980


namespace exam_correct_answers_l815_815612

theorem exam_correct_answers : (A_answers : ℕ → Prop) (B_answers : ℕ → Prop) (C_answers : ℕ → Prop) (correct : ℕ → Prop) :
    (∀ n, n ∈ {1, 2, 3, 4, 5} → (A_answers n = correct n ∨ A_answers n ≠ correct n)) ∧
    (A_answers 1 = ff) ∧ (A_answers 2 = ff) ∧ (A_answers 3 = tt) ∧ (A_answers 4 = ff) ∧ (A_answers 5 = ff) ∧
    (B_answers 1 = tt) ∧ (B_answers 2 = ff) ∧ (B_answers 3 = ff) ∧ (B_answers 4 = ff) ∧ (B_answers 5 = ff) ∧
    (C_answers 1 = tt) ∧ (C_answers 2 = tt) ∧ (C_answers 3 = tt) ∧ (C_answers 4 = ff) ∧ (C_answers 5 = ff) ∧
    (∀ student_answers, (student_answers = A_answers ∨ student_answers = B_answers ∨ student_answers = C_answers) →
        (∑ n in {1, 2, 3, 4, 5}, if student_answers n = correct n then 1 else 0) = 4) →
    (correct 1 = tt ∧ correct 2 = tt ∧ correct 3 = tt ∧ correct 4 = ff ∧ correct 5 = ff) :=
by 
  sorry

end exam_correct_answers_l815_815612


namespace general_formula_sum_of_first_n_terms_l815_815387

variables (q : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ)

-- Geometric sequence conditions
def geometric_sequence (q a : ℕ) : Prop :=
q = 2 ∧ a 2 + a 3 = 12 ∧ ∀ n : ℕ, a n = 2 ^ n

-- Arithmetic sequence conditions
def arithmetic_sequence (a b : ℕ → ℕ) : Prop :=
b 3 = a 3 ∧ b 7 = a 4 ∧ (∀ n : ℕ, b n = 4 + (n - 1) * 2)

-- General formula for geometric sequence
theorem general_formula
  (h : geometric_sequence q a) :
  ∀ n : ℕ, a n = 2 ^ n :=
sorry

-- Sum of the first n terms of arithmetic sequence
theorem sum_of_first_n_terms (a b : ℕ → ℕ)
  (h : arithmetic_sequence a b) :
  ∀ n : ℕ, (finset.range n).sum b = n^2 + 3*n :=
sorry

end general_formula_sum_of_first_n_terms_l815_815387


namespace larger_number_is_23_l815_815977

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815977


namespace collinear_intersection_points_l815_815034

noncomputable def are_collinear (P Q R : Point) : Prop := sorry

theorem collinear_intersection_points
  (Γ1 Γ2 Γ3 : Circle) 
  (r1 r2 r3 : ℝ)
  (h_disjoint : Γ1 ∩ Γ2 = ∅ ∧ Γ2 ∩ Γ3 = ∅ ∧ Γ3 ∩ Γ1 = ∅)
  (A12 A23 A31 : Point)
  (h_A12 : A12 = intersection_of_external_tangents Γ1 Γ2)
  (h_A23 : A23 = intersection_of_external_tangents Γ2 Γ3)
  (h_A31 : A31 = intersection_of_external_tangents Γ3 Γ1) :
  are_collinear A12 A23 A31 :=
sorry

end collinear_intersection_points_l815_815034


namespace ratio_student_to_sister_l815_815467

def student_weight : ℕ := 90
def total_weight : ℕ := 132
def weight_loss : ℕ := 6

theorem ratio_student_to_sister : 
  let S := student_weight - weight_loss in
  let R := total_weight - student_weight in
  S / R = 2 :=
by
  let S := student_weight - weight_loss
  let R := total_weight - student_weight
  have h1 : S = 84 := rfl
  have h2 : R = 42 := rfl
  rw [h1, h2]
  norm_num
  sorry

end ratio_student_to_sister_l815_815467


namespace suitable_for_training_l815_815671

def studentA_scores (scores : List ℤ) : Prop :=
  Mode scores = 140 ∧ Median scores = 145

def studentB_scores (scores : List ℤ) : Prop :=
  Median scores = 145 ∧ Range scores = 6

def studentC_scores (scores : List ℤ) : Prop :=
  Mean scores = 143 ∧ scores.contains 145 ∧ Variance scores = 1.6

def scores_suitable (scores : List ℤ) : Prop :=
  ∀ score ∈ scores, score ≥ 140

theorem suitable_for_training (scoresA scoresC : List ℤ) :
  studentA_scores scoresA ∧ 
  studentC_scores scoresC ∧ 
  scores_suitable scoresA ∧ 
  scores_suitable scoresC :=
by
  -- proof goes here
  sorry

end suitable_for_training_l815_815671


namespace James_total_area_l815_815094

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase_dimension : ℕ := 2
def number_of_new_rooms : ℕ := 4
def larger_room_multiplier : ℕ := 2

noncomputable def new_length : ℕ := initial_length + increase_dimension
noncomputable def new_width : ℕ := initial_width + increase_dimension
noncomputable def area_of_one_new_room : ℕ := new_length * new_width
noncomputable def total_area_of_4_rooms : ℕ := area_of_one_new_room * number_of_new_rooms
noncomputable def area_of_larger_room : ℕ := area_of_one_new_room * larger_room_multiplier
noncomputable def total_area : ℕ := total_area_of_4_rooms + area_of_larger_room

theorem James_total_area : total_area = 1800 := 
by
  sorry

end James_total_area_l815_815094


namespace yogurt_cost_production_proof_l815_815614

noncomputable def cost_of_ingredient (quantity : ℕ → ℝ) (price : ℝ) : ℝ :=
λ (n : ℕ), n * quantity n * price

def batch_ingredients_adjusted_price : ℕ → ℝ 
| 1 := 15 
| 2 := 12
| 3 := 0.4
| 4 := 37.5
| 5 := 0.75
| 6 := 20

noncomputable def calculate_total_cost 
  (quantities : ℕ → ℝ) 
  (prices : ℕ → ℝ) 
  (adjustments : ℕ → ℝ) 
  (batches : ℕ) 
  (discount_conditions : ℕ → ℝ → ℝ) : ℝ :=
(∑ i in finset.range 6, quantity i * adjustments i * prices i * 7) - discounts 42 1.5

theorem yogurt_cost_production_proof 
: 7 * (
    15 
  + 12
  + 0.4
  + 37.5 
  + 0.75 
  + 20 
) - discounts 42 1.5 = 578.55 :=
by sorry

end yogurt_cost_production_proof_l815_815614


namespace larger_number_is_23_l815_815986

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l815_815986


namespace prime_square_not_perfect_square_l815_815523
open Nat

theorem prime_square_not_perfect_square 
  (p q : ℕ) (h1 : Prime p) (h2 : Prime q) 
  (h3 : ∃ k : ℕ, p + q^2 = k^2) : 
  ∀ n : ℕ, ∀ m : ℕ, p^2 + q^n ≠ m^2 :=
by
  sorry

end prime_square_not_perfect_square_l815_815523


namespace unique_solution_of_system_l815_815934

noncomputable def solve_system_of_equations (x1 x2 x3 x4 x5 x6 x7 : ℝ) : Prop :=
  10 * x1 + 3 * x2 + 4 * x3 + x4 + x5 = 0 ∧
  11 * x2 + 2 * x3 + 2 * x4 + 3 * x5 + x6 = 0 ∧
  15 * x3 + 4 * x4 + 5 * x5 + 4 * x6 + x7 = 0 ∧
  2 * x1 + x2 - 3 * x3 + 12 * x4 - 3 * x5 + x6 + x7 = 0 ∧
  6 * x1 - 5 * x2 + 3 * x3 - x4 + 17 * x5 + x6 = 0 ∧
  3 * x1 + 2 * x2 - 3 * x3 + 4 * x4 + x5 - 16 * x6 + 2 * x7 = 0 ∧
  4 * x1 - 8 * x2 + x3 + x4 - 3 * x5 + 19 * x7 = 0

theorem unique_solution_of_system :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℝ),
    solve_system_of_equations x1 x2 x3 x4 x5 x6 x7 →
    x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0 ∧ x6 = 0 ∧ x7 = 0 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h
  sorry

end unique_solution_of_system_l815_815934


namespace class_schedule_count_l815_815613

-- Definitions based on the conditions
def is_morning (period : ℕ) : Prop := period < 4
def is_afternoon (period : ℕ) : Prop := period >= 4

-- The count of ways to arrange the schedule under given conditions
theorem class_schedule_count : 
  let lessons := ["Chinese", "Mathematics", "Politics", "English", "Physical Education", "Art"]
  let morning_lessons := filter (λ p, is_morning p) (range 6)
  let afternoon_lessons := filter (λ p, is_afternoon p) (range 6)
  let math_combinations := morning_lessons.length * afternoon_lessons.length
  let remaining_lessons := lessons.erase "Mathematics" |>.erase "Physical Education"
  let remaining_combinations := (remaining_lessons.length).fact
  in math_combinations * remaining_combinations = 192 := 
by
  sorry

end class_schedule_count_l815_815613


namespace sequence_monotonically_increasing_l815_815823

noncomputable def a (n : ℕ) : ℝ := (n - 1 : ℝ) / (n + 1 : ℝ)

theorem sequence_monotonically_increasing : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end sequence_monotonically_increasing_l815_815823


namespace reciprocal_of_subtraction_l815_815626

-- Defining the conditions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 3

-- Defining the main theorem statement
theorem reciprocal_of_subtraction : (1 / (y - x)) = 9 / 5 :=
by
  sorry

end reciprocal_of_subtraction_l815_815626


namespace larger_number_is_23_l815_815979

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815979


namespace center_of_symmetry_and_sum_inequality_solutions_l815_815697

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3 * x - 2

theorem center_of_symmetry_and_sum :
  (∃ x₀ y₀, f'' x₀ = 0 ∧ (x₀, y₀) = (1 / 2, -1) ∧ 
    (Σ i in finset.range 2023, f (i / 2024) = -2023)) :=
sorry

theorem inequality_solutions (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 3 → {x : ℝ | 1 / 3 < x ∧ x < 1 / a}) ∧
  (a = 3 → ∅) ∧
  (a > 3 → {x : ℝ | 1 / a < x ∧ x < 1 / 3}) :=
sorry

#check center_of_symmetry_and_sum
#check inequality_solutions

end center_of_symmetry_and_sum_inequality_solutions_l815_815697


namespace percentage_difference_l815_815458

theorem percentage_difference :
  (0.50 * 56 - 0.30 * 50) = 13 := 
by
  -- sorry is used to skip the actual proof steps
  sorry 

end percentage_difference_l815_815458


namespace constant_term_in_expansion_l815_815012

theorem constant_term_in_expansion 
    (n : ℕ) 
    (h : (∑ i in Finset.range (n + 1), Nat.choose n i) = 64) 
    : ∃ k, (Nat.choose 6 4 = 15) := 
by
  sorry

end constant_term_in_expansion_l815_815012


namespace angle_BDE_eq_10_l815_815076

lean 4 proof:

-- Definitions of the angles and points in the quadrilateral.
variables (A B C D E : Point)
variables (angle_A angle_B angle_C : ℝ)

-- Given conditions.
axiom angle_A_eq_70 : angle_A = 70
axiom angle_B_eq_90 : angle_B = 90
axiom angle_C_eq_40 : angle_C = 40
axiom E_on_extension_CD : collinear E C D ∧ distance E D = distance E C

-- The theorem to prove.
theorem angle_BDE_eq_10 : angle B D E = 10 := sorry

end angle_BDE_eq_10_l815_815076


namespace arith_seq_general_formula_geom_seq_sum_l815_815799

-- Problem 1
theorem arith_seq_general_formula (a : ℕ → ℕ) (d : ℕ) (h_d : d = 3) (h_a1 : a 1 = 4) :
  a n = 3 * n + 1 :=
sorry

-- Problem 2
theorem geom_seq_sum (b : ℕ → ℚ) (S : ℕ → ℚ) (h_b1 : b 1 = 1 / 3) (r : ℚ) (h_r : r = 1 / 3) :
  S n = (1 / 2) * (1 - (1 / 3 ^ n)) :=
sorry

end arith_seq_general_formula_geom_seq_sum_l815_815799


namespace cos_of_angle_in_fourth_quadrant_l815_815380

variable {α : ℝ}

theorem cos_of_angle_in_fourth_quadrant (h1 : α ∈ Icc (3 * π / 2) (2 * π))
                          (h2 : Real.tan α = -5 / 12) : Real.cos α = 12 / 13 :=
by
  sorry

end cos_of_angle_in_fourth_quadrant_l815_815380


namespace find_OC_l815_815867

variable {AB BC AD r PC angle_C cos_angle_C : ℝ}

variable {O : Point} -- Defining a Point type for clarity.

axiom h1 : AB = 25/64 
axiom h2 : BC = 12 + 25/64
axiom h3 : AD = 6 + 1/4
axiom h4 : ∠DAB = α
axiom h5 : ∠ABC = β
axiom h6 : sin α = 3/5
axiom h7 : cos β = -63/65
axiom h8 : Center O is center of circle tangent to BC, CD, and AD
axiom h9 : cos (α + β) = -5/13

theorem find_OC : OC = sqrt(130)/2 :=
by
  sorry

end find_OC_l815_815867


namespace tom_dad_collect_leaves_in_21_5_minutes_l815_815206

theorem tom_dad_collect_leaves_in_21_5_minutes : ∀ (total_leaves : ℕ) (gathered : ℕ) (scattered : ℕ) (cycle_time : ℕ), 
  total_leaves = 45 → gathered = 4 → scattered = 3 → cycle_time = 30 → 
  (21.5 : ℝ) * 60 = total_leaves * cycle_time / (gathered - scattered) → true :=
by
  intros total_leaves gathered scattered cycle_time h1 h2 h3 h4 h5
  exact trivial

end tom_dad_collect_leaves_in_21_5_minutes_l815_815206


namespace proof_l815_815052

def problem (a b : ℝ) : Prop :=
  b = sqrt (3 - a) + sqrt (a - 3) + 8

theorem proof (a b : ℝ) (h1 : problem a b) (ha : a = 3) : sqrt (a * b + 1) = 5 ∨ sqrt (a * b + 1) = -5 := 
by
  sorry

end proof_l815_815052


namespace initial_water_in_hole_l815_815039

theorem initial_water_in_hole (total_needed additional_needed initial : ℕ) (h1 : total_needed = 823) (h2 : additional_needed = 147) :
  initial = total_needed - additional_needed :=
by
  sorry

end initial_water_in_hole_l815_815039


namespace product_eq_sqrt_101_div_200_l815_815734

noncomputable def product_of_sqrt_fractions : ℝ :=
  ∏ n in finset.range 99, (sqrt n * sqrt (n + 2)) / ((n + 1) ^ 2)

theorem product_eq_sqrt_101_div_200 : product_of_sqrt_fractions = sqrt 101 / 200 := 
by
  sorry

end product_eq_sqrt_101_div_200_l815_815734


namespace find_hyperbola_asymptote_l815_815822

def hyperbola_asymptote (a : ℝ) (h : a > 0) (x y : ℝ) :=
  (x^2 / a^2 - y^2 / 3 = 1) → ∃ t: ℝ, (t ≠ 0) ∧ (y = t * x)

theorem find_hyperbola_asymptote:
  ∀ (a : ℝ) (h : a > 0),
    (hyperbola_asymptote a h (-2) 1) →
    ((a = sqrt(3)) →
      (∀ x y : ℝ, (x ± y = 0) ↔ (y = ± x))) :=
by
  sorry

end find_hyperbola_asymptote_l815_815822


namespace remainder_of_concatenated_numbers_from_1_to_39_mod_40_l815_815511

/-- Let M be the number formed by writing integers from 1 to 39 consecutively. 
    Prove that the remainder when M is divided by 40 is 0. -/
theorem remainder_of_concatenated_numbers_from_1_to_39_mod_40 :
  let M := nat_concat (list.range 39).map (λ x, x + 1) 
  in M % 40 = 0 := sorry

end remainder_of_concatenated_numbers_from_1_to_39_mod_40_l815_815511


namespace divide_students_into_groups_l815_815854

-- Define a structure to represent the groups
structure Group :=
  (english : Nat)
  (french : Nat)
  (spanish : Nat)

-- Define a predicate to check if a list of groups satisfies the condition
def satisfies_conditions (groups : List Group) : Prop :=
  groups.length = 5 ∧ 
  (∀ g ∈ groups, g.english = 10 ∧ g.french = 10 ∧ g.spanish = 10)

theorem divide_students_into_groups :
  ∃ (groups : List Group), satisfies_conditions groups :=
sorry

end divide_students_into_groups_l815_815854


namespace range_of_a_l815_815141

open Real

namespace PropositionProof

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0)

theorem range_of_a (a : ℝ) (h : a < 0) :
  (¬ ∀ x, ¬ p a x → ∀ x, ¬ q x) ↔ (a ≤ -4 ∨ -2/3 ≤ a ∧ a < 0) :=
sorry

end PropositionProof

end range_of_a_l815_815141


namespace wedge_volume_correct_l815_815675

-- Given conditions
def diameter : ℝ := 20
def radius : ℝ := diameter / 2
def height : ℝ := diameter
def angle : ℝ := 60

-- Volume of the wedge
def volume_cylinder : ℝ := Real.pi * radius^2 * height
def volume_wedge : ℝ := volume_cylinder / 3

-- Proving the number of cubic inches in the wedge equals 667π
theorem wedge_volume_correct : ∃ m : ℕ, volume_wedge = m * Real.pi ∧ m = 667 := 
by {
  have r_eq : radius = 10 := by norm_num [radius],
  have h_eq : height = 20 := by norm_num [height],
  have v_cylinder_eq : volume_cylinder = 2000 * Real.pi := by 
    simp [volume_cylinder, r_eq, h_eq]; ring_nf, 
  
  have v_wedge_eq : volume_wedge = 2000 / 3 * Real.pi := by 
    simp [volume_wedge, v_cylinder_eq]; ring, 
  
  use 667,
  split,
  { exact v_wedge_eq.symm.trans (by field_simp [eq_refl Real.pi]; norm_num) },
  { norm_num }
}

end wedge_volume_correct_l815_815675


namespace larger_number_is_23_l815_815991

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815991


namespace actual_cost_of_article_l815_815292

noncomputable def article_actual_cost (x : ℝ) : Prop :=
  (0.58 * x = 1050) → x = 1810.34

theorem actual_cost_of_article : ∃ x : ℝ, article_actual_cost x :=
by
  use 1810.34
  sorry

end actual_cost_of_article_l815_815292


namespace negation_proof_l815_815182

theorem negation_proof (x : ℝ) : ¬ (x^2 - x + 3 > 0) ↔ (x^2 - x + 3 ≤ 0) :=
by sorry

end negation_proof_l815_815182


namespace rabbits_initially_bought_l815_815269

theorem rabbits_initially_bought (R : ℕ) (h : ∃ (k : ℕ), R + 6 = 17 * k) : R = 28 :=
sorry

end rabbits_initially_bought_l815_815269


namespace value_of_a_for_perfect_square_trinomial_l815_815464

theorem value_of_a_for_perfect_square_trinomial (a : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 + 2 * a * x + 9 = (x + b)^2) → (a = 3 ∨ a = -3) :=
by
  sorry

end value_of_a_for_perfect_square_trinomial_l815_815464


namespace eccentricity_of_hyperbola_l815_815113

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := 2 * b in
  let e := c / a in
  e

theorem eccentricity_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hyp : ∀ (x y : ℝ), (x / a)^2 - (y / b)^2 = 1) 
  (isosceles_right_triangle : (P : ℝ × ℝ) = (0, 2 * b)) :
  hyperbola_eccentricity a b ha hb = (2 * (real.sqrt 3) / 3) := 
by
  sorry

end eccentricity_of_hyperbola_l815_815113


namespace four_digit_perfect_square_is_1156_l815_815004

theorem four_digit_perfect_square_is_1156 :
  ∃ (N : ℕ), (N ≥ 1000) ∧ (N < 10000) ∧ (∀ a, a ∈ [N / 1000, (N % 1000) / 100, (N % 100) / 10, N % 10] → a < 7) 
              ∧ (∃ n : ℕ, N = n * n) ∧ (∃ m : ℕ, (N + 3333 = m * m)) ∧ (N = 1156) :=
by
  sorry

end four_digit_perfect_square_is_1156_l815_815004


namespace problem_statement_l815_815836

-- Define the conditions
variables (x y : ℝ)
variables (hx1 : x ≠ 0) (hx2 : x ≠ 3)
variables (hy1 : y ≠ 0) (hy2 : y ≠ 7)

-- Define the main hypothesis
def hypothesis := (5 / x) + (4 / y) = 1 / 3

-- Define the goal
def goal := x = 15 * y / (y - 12)

theorem problem_statement : hypothesis x y hx1 hx2 hy1 hy2 → goal x y := by
  sorry

end problem_statement_l815_815836


namespace line_equation_length_AB_l815_815787

-- Given conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 8) + (y^2 / 4) = 1
def midpoint (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := 
  P = (2, 1) ∧ 
  ∃ l : ℝ, line_through (2, 1) (A) l ∧ line_through (2, 1) (B) l ∧
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1

-- Proof problem 1: Line equation
theorem line_equation : 
  ∀ (A B : ℝ × ℝ), 
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ midpoint (2, 1) A B → 
  ∃ (l m : ℝ), l ≠ 0 → 
  ∀ (x y : ℝ), line_through (2, 1) (x, y) l m ↔ x + y - 3 = 0 :=
sorry

-- Proof problem 2: Length of |AB|
theorem length_AB :
  ∀ (A B : ℝ × ℝ), 
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ midpoint (2, 1) A B ∧ 
  (∀ (x y : ℝ), line_through (2, 1) (x, y) 1 (-1) → x + y - 3 = 0) →
  |AB| = (4 / 3) * sqrt 3 :=
sorry

end line_equation_length_AB_l815_815787


namespace num_integers_satisfying_inequality_l815_815430

theorem num_integers_satisfying_inequality : 
  {x : ℤ | (x - 2)^2 ≤ 4}.finite.to_finset.card = 5 :=
by {
  sorry
}

end num_integers_satisfying_inequality_l815_815430


namespace max_height_projectile_l815_815684

theorem max_height_projectile :
  ∃ t : ℝ, h = -16 * t^2 + 64 * t + 36 ∧ (h ≤ 100) ∧ ∀ t', -16 * t'^2 + 64 * t' + 36 ≤ 100 :=
begin
  have vertex_t := 2, -- The time t when the projectile reaches the maximum height
  have h_max : ℝ := 100,
  use vertex_t,
  split,
  { -- h = -16t^2 + 64t + 36 evaluated at t = 2
    calc 
      h = (-16 * 2^2 + 64 * 2 + 36) : by sorry,
  },
  split,
  { -- h <= 100 for all t
    sorry,
  },
  { -- -16t'^2 + 64t' + 36 ≤ 100 for all t'
    intros t',
    sorry,
  }
end

end max_height_projectile_l815_815684


namespace cosine_of_angle_in_second_quadrant_l815_815461

theorem cosine_of_angle_in_second_quadrant 
  (α : ℝ) 
  (h1 : sin α = 5 / 13) 
  (h2 : π / 2 < α ∧ α < π) : cos α = -12 / 13 :=
by 
  sorry

end cosine_of_angle_in_second_quadrant_l815_815461


namespace T_divisible_by_7_pow_ceil_l815_815553

noncomputable def x (i : ℕ) : ℝ := ((2:ℝ) * Real.sin (i * Real.pi / 7))^2
noncomputable def T (n : ℕ) : ℝ := x 1 ^ n + x 2 ^ n + x 3 ^ n

theorem T_divisible_by_7_pow_ceil (n : ℕ) : ∃ k : ℕ, T n = 7^k * T' n :=
sorry

end T_divisible_by_7_pow_ceil_l815_815553


namespace map_distance_to_actual_distance_l815_815576

theorem map_distance_to_actual_distance :
  ∀ (d_map : ℝ) (scale_inch : ℝ) (scale_mile : ℝ), 
    d_map = 15 → scale_inch = 0.25 → scale_mile = 3 →
    (d_map / scale_inch) * scale_mile = 180 :=
by
  intros d_map scale_inch scale_mile h1 h2 h3
  rw [h1, h2, h3]
  sorry

end map_distance_to_actual_distance_l815_815576


namespace solve_triangle_problem_l815_815486

noncomputable def triangle_problem : Prop :=
  ∃ (x : ℝ) (A B C : ℝ),
    A = 45 ∧
    B = 3 * x ∧
    C = 1 / 2 * B ∧
    A + B + C = 180 ∧
    C = 45

theorem solve_triangle_problem : triangle_problem :=
begin
  use 30,
  use 45,
  use 90,
  use 45,
  sorry
end

end solve_triangle_problem_l815_815486


namespace solve_y_l815_815151

theorem solve_y : ∀ y : ℚ, (\(\left(\frac{1}{8}\right)^{3 * y + 9} = 64^{3 * y + 4}\)) → y = -\(\frac{17}{9}\) :=
by
  intro y
  sorry

end solve_y_l815_815151


namespace solve_equation_l815_815191

-- Definition of the greatest integer function [x]
def greatest_integer (x : ℝ) : ℤ :=
  Int.floor x

theorem solve_equation : ∀ (x : ℝ),
  (greatest_integer (3 * x - (29 / 6)) - 2 * x - 1 = 0) → 
  x = 6.5 :=
by
  intro x h
  sorry

end solve_equation_l815_815191


namespace hyperbola_equation_l815_815165

theorem hyperbola_equation (h : ∃ (x y : ℝ), y = 1 / 2 * x) (p : (2, 2) ∈ {p : ℝ × ℝ | ((p.snd)^2 / 3) - ((p.fst)^2 / 12) = 1}) :
  ∀ (x y : ℝ), (y^2 / 3 - x^2 / 12 = 1) ↔ (∃ (a b : ℝ), y = a * x ∧ b * y = x ^ 2) :=
sorry

end hyperbola_equation_l815_815165


namespace problem_statement_l815_815367

noncomputable def collinear (A B C : Point) : Prop :=
  ∃ l : Line, A ∈ l ∧ B ∈ l ∧ C ∈ l

noncomputable def concurrent (l1 l2 l3 : Line) : Prop :=
  ∃ P : Point, P ∈ l1 ∧ P ∈ l2 ∧ P ∈ l3

theorem problem_statement
  (A B C D A₁ B₁ C₁ D₁ M N P E G F : Point)
  (cuboid : RectangularCuboid ABCD-A₁B₁C₁D₁)
  (perps : PerpendicularsFromA A A₁B A₁C A₁D to A₁B₁ A₁C₁ A₁D₁ at M N P feet E G F) :
  collinear M N P ∧ concurrent (lineThrough P E) (lineThrough M F) (lineThrough A N) :=
sorry

end problem_statement_l815_815367


namespace center_of_circle_l815_815266

theorem center_of_circle 
  (C : ℝ × ℝ)
  (H1 : ∃ r : ℝ, (0, 3) ∈ Sphere C r)
  (H2 : Tangent_to_parabola_at C (y = x^2 + 2x) (-1, 1)) :
  C = (-1, 1 + Real.sqrt 5) ∨ C = (-1, 1 - Real.sqrt 5) :=
by sorry

end center_of_circle_l815_815266


namespace rectangle_diagonal_opposite_vertex_l815_815460

theorem rectangle_diagonal_opposite_vertex :
  ∀ (x y : ℝ),
    (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
      (x1, y1) = (5, 10) ∧ (x2, y2) = (15, -6) ∧ (x3, y3) = (11, 2) ∧
      (∃ (mx my : ℝ), mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2 ∧
        mx = (x + x3) / 2 ∧ my = (y + y3) / 2) ∧
      x = 9 ∧ y = 2) :=
by
  sorry

end rectangle_diagonal_opposite_vertex_l815_815460


namespace simplify_expression_l815_815774

theorem simplify_expression : ( (3 + 4 + 5 + 6) / 3 ) + ( (3 * 6 + 9) / 4 ) = 12.75 := by
  sorry

end simplify_expression_l815_815774


namespace ratio_of_chickens_in_run_to_coop_l815_815607

def chickens_in_coop : ℕ := 14
def free_ranging_chickens : ℕ := 52
def run_condition (R : ℕ) : Prop := 2 * R - 4 = 52

theorem ratio_of_chickens_in_run_to_coop (R : ℕ) (hR : run_condition R) :
  R / chickens_in_coop = 2 :=
by
  sorry

end ratio_of_chickens_in_run_to_coop_l815_815607


namespace area_of_given_quadrilateral_l815_815226

def point := ℝ × ℝ

def area_of_quadrilateral (A B C D : point) : ℝ :=
  1 / 2 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2) - 
               (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1))

theorem area_of_given_quadrilateral :
  area_of_quadrilateral (3, 1) (1, 6) (4, 5) (9, 9) = 6.5 :=
by
  sorry

end area_of_given_quadrilateral_l815_815226


namespace number_of_real_roots_of_f_eq_0_is_3_l815_815703

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2012^x + Real.log x / Real.log 2012
else if x = 0 then 0
else - (2012^(-x) + Real.log (-x) / Real.log 2012)

theorem number_of_real_roots_of_f_eq_0_is_3 :
  ∃! x : ℝ, f(x) = 0 :=
sorry

end number_of_real_roots_of_f_eq_0_is_3_l815_815703


namespace flag_combinations_l815_815320

namespace CrestviewFlag

-- Define the number of colors and the stripes in the flag
def numColors : ℕ := 3
def numStripes : ℕ := 4

-- Define the problem as a theorem
theorem flag_combinations (h1 : numColors = 3) (h2 : numStripes = 4) :
  ∃ n, n = 3 * 2 * 2 * 2 ∧ n = 24 :=
by {
  use 24,
  split,
  { sorry },  -- We acknowledge 3 * 2 * 2 * 2 = 24 but defer proof
  { refl }
}

end CrestviewFlag

end flag_combinations_l815_815320


namespace concyclic_points_of_rectangle_l815_815120

noncomputable theory
open_locale classical

variables (A B C D E F G P : Type*)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
          [metric_space E] [metric_space F] [metric_space G] [metric_space P]

-- Define the rectangle ABCD
def rectangle (A B C D : Type*) := ∃ (AB BC CD DA : ℝ), AB = BC / 2 ∧ BC = 2 * AB ∧ CD = AB ∧ DA = BC

-- Define the midpoint E of BC
def midpoint (B C E : Type*) := ∃ (B C E : ℝ), E = (B + C) / 2

-- Define the foot of perpendiculars F from A to BP and G from D to CP
def foot_of_perpendicular_f (A B P F : Type*) := ∃ (A B P F : ℝ), is_perp_to_line A B P F
def foot_of_perpendicular_g (D C P G : Type*) := ∃ (D C P G : ℝ), is_perp_to_line D C P G

-- Define the concyclic property for points E, F, P, and G
def concyclic (E F P G : Type*) := ∃ (circle : Type*), E, F, P, G ∈ circle

theorem concyclic_points_of_rectangle
  (h1 : rectangle A B C D) (h2 : BC = 2 * AB)
  (h3 : midpoint B C E) (h4 : arbitrary_inner_point P AD)
  (h5 : foot_of_perpendicular_f A B P F)
  (h6 : foot_of_perpendicular_g D C P G) :
  concyclic E F P G :=
sorry

end concyclic_points_of_rectangle_l815_815120


namespace count_integers_satisfying_inequality_l815_815449

theorem count_integers_satisfying_inequality : ∃ (count : ℕ), count = 5 ∧ 
  ∀ x : ℤ, (0 ≤ x ∧ x ≤ 4) → ((x - 2) * (x - 2) ≤ 4) :=
begin
  existsi 5, 
  split,
  { refl },
  { intros x hx,
    cases hx with h1 h2,
    have : 0 ≤ (x - 2) ^ 2, from pow_two_nonneg (x - 2),
    exact le_antisymm this (show (x - 2) ^ 2 ≤ 4, by sorry) }
end

end count_integers_satisfying_inequality_l815_815449


namespace spherical_to_rectangular_coords_l815_815319

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), ρ = 5 → θ = (Real.pi / 2) → φ = (Real.pi / 3) →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 0 ∧ y = (5 * Real.sqrt 3 / 2) ∧ z = (5 / 2) :=
by
  intros ρ θ φ h1 h2 h3
  dsimp [*, Real.sin, Real.cos]
  sorry

end spherical_to_rectangular_coords_l815_815319


namespace total_people_in_house_l815_815643

-- Definitions of initial condition in the bedroom and living room
def charlie_susan_in_bedroom : ℕ := 2
def sarah_and_friends_in_bedroom : ℕ := 5
def people_in_living_room : ℕ := 8

-- Prove the total number of people in the house is 15
theorem total_people_in_house : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom + people_in_living_room = 15 :=
by
  -- sum the people in the bedroom (Charlie, Susan, Sarah, 4 friends)
  have bedroom_total : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom = 7 := by sorry
  -- sum the people in the house (bedroom + living room)
  show bedroom_total + people_in_living_room = 15 from sorry

end total_people_in_house_l815_815643


namespace projected_revenue_increase_is_20_percent_l815_815540

noncomputable def projected_percentage_increase_of_revenue (R : ℝ) (actual_revenue : ℝ) (projected_revenue : ℝ) : ℝ :=
  (projected_revenue / R - 1) * 100

theorem projected_revenue_increase_is_20_percent (R : ℝ) (actual_revenue : ℝ) :
  actual_revenue = R * 0.75 →
  actual_revenue = (R * (1 + 20 / 100)) * 0.625 →
  projected_percentage_increase_of_revenue R ((R * (1 + 20 / 100))) = 20 :=
by
  intros h1 h2
  sorry

end projected_revenue_increase_is_20_percent_l815_815540


namespace total_people_in_house_l815_815642

-- Definitions of initial condition in the bedroom and living room
def charlie_susan_in_bedroom : ℕ := 2
def sarah_and_friends_in_bedroom : ℕ := 5
def people_in_living_room : ℕ := 8

-- Prove the total number of people in the house is 15
theorem total_people_in_house : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom + people_in_living_room = 15 :=
by
  -- sum the people in the bedroom (Charlie, Susan, Sarah, 4 friends)
  have bedroom_total : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom = 7 := by sorry
  -- sum the people in the house (bedroom + living room)
  show bedroom_total + people_in_living_room = 15 from sorry

end total_people_in_house_l815_815642


namespace problem_ellipse_circle_tangent_l815_815790

-- Define the problem: Given the ellipse properties and points.
theorem problem_ellipse_circle_tangent 
  (h_center : ∀ (C : ℝ × ℝ → Prop), C = λ p, ∃ x y, p = (x, y) ∧ Origin (0, 0))
  (h_foci : ∀ (C : ℝ × ℝ → Prop), C = λ p, ∃ x y, p = (x, y) ∧ p ∈ x_axis)
  (h_eccentricity : ∀ (C : ℝ × ℝ → Prop), C = λ e, e = 1/2)
  (h_point : ∀ (C : ℝ × ℝ → Prop), C = (1, 3/2))
  (h_area : ∀ (l : ℝ × ℝ → Prop), ∃ A B O, l A ∧ l B ∧ Area (triangle A O B) = 6*sqrt 2 / 7) :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ C = λ p, p = (x, y) → x^2 / a^2 + y^2 / b^2 = 1 ∧ p = (1, 3/2)) ∧ 
  (∃ (r : ℝ), r = sqrt 2 / 2 ∧ Circle (Origin (0, 0)) r ∧ ∃ l, Tangent (l) (Origin (0, 0)) r :=
by sorry

end problem_ellipse_circle_tangent_l815_815790


namespace total_area_correct_l815_815099

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase : ℕ := 2
def num_extra_rooms_same_size : ℕ := 3
def num_double_size_rooms : ℕ := 1

def increased_length : ℕ := initial_length + increase
def increased_width : ℕ := initial_width + increase

def area_of_one_room (length width : ℕ) : ℕ := length * width

def number_of_rooms : ℕ := 1 + num_extra_rooms_same_size

def total_area_same_size_rooms : ℕ := number_of_rooms * area_of_one_room increased_length increased_width

def total_area_extra_size_room : ℕ := num_double_size_rooms * 2 * area_of_one_room increased_length increased_width

def total_area : ℕ := total_area_same_size_rooms + total_area_extra_size_room

theorem total_area_correct : total_area = 1800 := by
  -- Proof omitted
  sorry

end total_area_correct_l815_815099


namespace find_f_lg_lg3_l815_815021

def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin x + b * x^(1/3 : ℝ) + 4

theorem find_f_lg_lg3 (a b : ℝ) (h : f (Real.log (Real.log 10 / Real.log 3) / Real.log 10) a b = 5) :
  f (Real.log (Real.log 3) / Real.log 10) a b = 3 := 
sorry

end find_f_lg_lg3_l815_815021


namespace vector_magnitude_difference_l815_815417

noncomputable def vector_magnitude {α : Type*} [NormedSpace ℝ α] (v : α) : ℝ := ‖v‖

theorem vector_magnitude_difference
  {α : Type*} [NormedSpace ℝ α]
  (a b : α)
  (ha : vector_magnitude a = 1)
  (hb : vector_magnitude b = 2)
  (angle_eq : real.angle a b = real.pi / 3) :
  2 * vector_magnitude (a - b) = 4 := by
  sorry

end vector_magnitude_difference_l815_815417


namespace age_of_teacher_l815_815570

theorem age_of_teacher
    (n_students : ℕ)
    (avg_age_students : ℕ)
    (new_avg_age : ℕ)
    (n_total : ℕ)
    (H1 : n_students = 22)
    (H2 : avg_age_students = 21)
    (H3 : new_avg_age = avg_age_students + 1)
    (H4 : n_total = n_students + 1) :
    ((new_avg_age * n_total) - (avg_age_students * n_students) = 44) :=
by
    sorry

end age_of_teacher_l815_815570


namespace sum_of_fractions_l815_815350

theorem sum_of_fractions (n : ℕ) (h : n ≥ 3) : 
  ∃ (d : Fin n → ℕ), (∀ i j : Fin n, i ≠ j → d i ≠ d j) ∧ (∑ i, 1 / (d i : ℝ) = 1) :=
sorry

end sum_of_fractions_l815_815350


namespace volume_of_omega_eq_l815_815761

noncomputable def volume_of_omega : ℝ :=
  ∫ (φ : ℝ) in 0..2 * Real.pi,
  ∫ (θ : ℝ) in 0..Real.pi / 3,
  ∫ (ρ : ℝ) in 0..6,
    ρ^2 * sin θ

theorem volume_of_omega_eq : volume_of_omega = 72 * Real.pi := by
  sorry

end volume_of_omega_eq_l815_815761


namespace sum_factorials_last_two_digits_l815_815218

/-- Prove that the last two digits of the sum of factorials of the first 15 positive integers equal to 13 --/
theorem sum_factorials_last_two_digits : 
  let f := fun n => Nat.factorial n in
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 + f 11 + f 12 + f 13 + f 14 + f 15) % 100 = 13 :=
by 
  sorry

end sum_factorials_last_two_digits_l815_815218


namespace cyclic_sequence_l815_815690

theorem cyclic_sequence (n : ℕ) (h : n % 6 = 0) : 
  (arrow_seq (530 + n)) % 6 = 2 ∧ 
  (arrow_seq (530 + n + 1)) % 6 = 3 ∧ 
  (arrow_seq (530 + n + 2)) % 6 = 4 ∧ 
  (arrow_seq (530 + n + 3)) % 6 = 5 ∧ 
  (arrow_seq (530 + n + 4)) % 6 = 0 ∧ 
  (arrow_seq (530 + n + 5)) % 6 = 1 ∧ 
  (arrow_seq (530 + n + 6)) % 6 = 2 ∧ 
  (arrow_seq (530 + n + 7)) % 6 = 3 :=
sorry

end cyclic_sequence_l815_815690


namespace three_numbers_less_or_equal_than_3_l815_815201

theorem three_numbers_less_or_equal_than_3 : 
  let a := 0.8
  let b := 0.5
  let c := 0.9
  (a ≤ 3) ∧ (b ≤ 3) ∧ (c ≤ 3) → 
  3 = 3 :=
by
  intros h
  sorry

end three_numbers_less_or_equal_than_3_l815_815201


namespace weather_condition_july4_l815_815707

variable (T : ℝ) -- Temperature in degrees Fahrenheit
variable (R : Prop) -- Raining or not

-- Statement of the problem
theorem weather_condition_july4
  (h1 : ∀ T, ¬ R → (T ≥ 70) → "crowded")
  (h2 : ¬ "crowded") :
  T < 70 ∨ R :=
sorry

end weather_condition_july4_l815_815707


namespace value_of_8b_l815_815377

theorem value_of_8b (a b : ℝ) (h1 : 6 * a + 3 * b = 3) (h2 : b = 2 * a - 3) : 8 * b = -8 := by
  sorry

end value_of_8b_l815_815377


namespace tallest_building_model_height_l815_815742

def height_campus : ℝ := 120
def volume_campus : ℝ := 30000
def volume_model : ℝ := 0.03
def height_model : ℝ := 1.2

theorem tallest_building_model_height :
  (volume_campus / volume_model)^(1/3) = (height_campus / height_model) :=
by
  sorry

end tallest_building_model_height_l815_815742


namespace general_term_eq_smallest_n_l815_815789

-- Define the arithmetic sequence and its sum.
def arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + d * (n - 1)
def sum_arith_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Define the first condition S_10 = 120
axiom S10_eq_120 (a1 d : ℤ) : sum_arith_seq a1 d 10 = 120

-- Define the geometric sequence condition
axiom geom_seq_cond (a1 d : ℤ) : (d * d) * 4 = d * (2 * a1 + d)

-- Define the general term formula of the sequence
def general_term (n : ℕ) : ℤ := 2 * n + 1

-- Define sum of reciprocals of S_n (T_n)
def inv_sum_seq (a1 d : ℤ) (n : ℕ) : ℚ := (1 : ℚ) / sum_arith_seq a1 d n
def cumulative_inv_sum_seq (a1 d : ℤ) (n : ℕ) : ℚ :=
  (finset.range n).sum (λ k, inv_sum_seq a1 d (k + 1))

-- Define the second part of the problem condition T_n > 15/22
axiom Tn_bound (a1 d : ℤ) (n : ℕ) : cumulative_inv_sum_seq a1 d n > (15 : ℚ) / 22

theorem general_term_eq (a1 d : ℤ) (n : ℕ) 
  (h1 : sum_arith_seq a1 d 10 = 120)
  (h2 : (d * d) * 4 = d * (2 * a1 + d)) :
  arith_seq a1 d n = general_term n := sorry

theorem smallest_n (a1 d : ℤ) (n : ℕ)
  (h1 : sum_arith_seq a1 d 10 = 120)
  (h2 : (d * d) * 4 = d * (2 * a1 + d))
  (h3 : cumulative_inv_sum_seq a1 d n > (15 : ℚ) / 22) :
  14 ≤ n := sorry

end general_term_eq_smallest_n_l815_815789


namespace expected_value_computation_l815_815153

open real

noncomputable def expected_value_ceil_log2_floor_log3 : ℚ :=
  -- define intervals for x
  let E_log2 := (1 / 3) * 1 + (2 / 3) * 2,
  -- define intervals for y
  let E_log3 := (1 / 4) * 0 + (3 / 4) * 1,
  -- calculate the expected value
  (E_log2 - E_log3)

theorem expected_value_computation :
  let E_log2 := (1 / 3) * 1 + (2 / 3) * 2,
  let E_log3 := (1 / 4) * 0 + (3 / 4) * 1,
  E_log2 - E_log3 = 11 / 12 :=
by
  let E_log2 := (1 / 3) * 1 + (2 / 3) * 2,
  let E_log3 := (1 / 4) * 0 + (3 / 4) * 1,
  have h : E_log2 - E_log3 = 11 / 12 := by norm_num,
  exact h

lemma final_answer :
  let m := 11,
  let n := 12,
  100 * m + n = 1112 :=
by
  let m := 11,
  let n := 12,
  have h1 : 100 * m + n = 100 * 11 + 12 := by rfl,
  have h2 : 100 * 11 = 1100 := by norm_num,
  have h3 : 1100 + 12 = 1112 := by norm_num,
  rw [h1, h2, h3],
  exact rfl

end expected_value_computation_l815_815153


namespace expression_evaluates_to_one_l815_815719

theorem expression_evaluates_to_one :
  (1 / 3)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (Real.pi / 3) + (Real.pi - 2016)^0 - (8:ℝ)^(1/3) = 1 :=
by
  -- step-by-step simplification skipped, as per requirements
  sorry

end expression_evaluates_to_one_l815_815719


namespace problem_2014_jining_mock_test_l815_815246

theorem problem_2014_jining_mock_test (a : ℝ) (f : ℝ → ℝ)
  (h1 : f = λ x, abs (2 * x + a))
  (h2 : ∀ x y : ℝ, 3 ≤ x → 3 ≤ y → x ≤ y → f x ≤ f y) :
  a = -6 :=
sorry

end problem_2014_jining_mock_test_l815_815246


namespace people_in_house_l815_815646

theorem people_in_house 
  (charlie_and_susan : ℕ := 2)
  (sarah_and_friends : ℕ := 5)
  (living_room_people : ℕ := 8) :
  (charlie_and_susan + sarah_and_friends) + living_room_people = 15 := 
by
  sorry

end people_in_house_l815_815646


namespace omega_value_monotonicity_l815_815401

variable (ω : ℝ) (h_ω_pos : ω > 0)

def f (x : ℝ) : ℝ := 4 * cos (ω * x) * sin (ω * x + π / 4)

theorem omega_value :
  (∀ x, f x = 4 * cos (ω * x) * sin (ω * x + π / 4)) ∧ (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (T = π) 
  → ω = 1 := sorry

theorem monotonicity (x : ℝ) :
  (0 ≤ x ∧ x ≤ π / 2) 
  → (∀ a, (0 ≤ a ∧ a ≤ π / 8) → f' a ≥ 0) ∧ (∀ b, (π / 8 ≤ b ∧ b ≤ π / 2) → f' b ≤ 0) := sorry

end omega_value_monotonicity_l815_815401


namespace geometric_sequence_first_term_l815_815190

theorem geometric_sequence_first_term (a1 q : ℝ) 
  (h1 : (a1 * (1 - q^4)) / (1 - q) = 240)
  (h2 : a1 * q + a1 * q^3 = 180) : 
  a1 = 6 :=
by
  sorry

end geometric_sequence_first_term_l815_815190


namespace base_of_isosceles_triangle_l815_815966

namespace TriangleProblem

def equilateral_triangle_perimeter (s : ℕ) : ℕ := 3 * s
def isosceles_triangle_perimeter (s b : ℕ) : ℕ := 2 * s + b

theorem base_of_isosceles_triangle (s b : ℕ) (h1 : equilateral_triangle_perimeter s = 45) 
    (h2 : isosceles_triangle_perimeter s b = 40) : b = 10 :=
by
  sorry

end TriangleProblem

end base_of_isosceles_triangle_l815_815966


namespace P_on_l_l815_815084

-- Declare point P
def P := (0 : ℝ, real.sqrt 3)

-- Definition of curve C with parametric equations
def C (phi : ℝ) : (ℝ × ℝ) := (real.sqrt 2 * real.cos phi, 2 * real.sin phi)

-- Polar equation of line l
def polar_l (ρ θ : ℝ) : Prop := ρ = (real.sqrt 3 / (2 * real.cos (θ - (real.pi / 6))))

-- Rectangular equation of line l derived
def line_l (x y : ℝ) : Prop := (real.sqrt 3 * x + y = real.sqrt 3)

-- Proving point P lies on line l
theorem P_on_l : line_l (P.1) (P.2) := 
by
  -- Provide the actual proof here
  sorry

end P_on_l_l815_815084


namespace max_balloons_l815_815545

-- Definitions
variable (p : ℝ) -- Regular price of one balloon
variable (m : ℝ) -- Money Orvin has (40 balloons at regular price p means m = 40 * p)

-- Conditions
def regular_price := p
def money_available := 40 * regular_price

-- Proof Problem Statement
theorem max_balloons (h : money_available = m) : m = 40 * p → ∀ s, s = 2 * (m / (3/2 * p)) → s ≤ 52 := by
  intros
  rw [money_available] at h
  exact sorry  -- Skipping the actual proof

end max_balloons_l815_815545


namespace distinct_real_roots_l815_815394

open Real

theorem distinct_real_roots (n : ℕ) (hn : n > 0) (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (2 * n - 1 < x1 ∧ x1 ≤ 2 * n + 1) ∧ 
  (2 * n - 1 < x2 ∧ x2 ≤ 2 * n + 1) ∧ |x1 - 2 * n| = k ∧ |x2 - 2 * n| = k) ↔ (0 < k ∧ k ≤ 1) :=
by
  sorry

end distinct_real_roots_l815_815394


namespace free_throws_count_l815_815183

-- Given conditions:
variables (a b x : ℕ) -- α is an abbreviation for natural numbers

-- Condition: number of points from all shots
axiom points_condition : 2 * a + 3 * b + x = 79
-- Condition: three-point shots are twice the points of two-point shots
axiom three_point_condition : 3 * b = 4 * a
-- Condition: number of free throws is one more than the number of two-point shots
axiom free_throw_condition : x = a + 1

-- Prove that the number of free throws is 12
theorem free_throws_count : x = 12 :=
by {
  sorry
}

end free_throws_count_l815_815183


namespace circle_chord_intersect_l815_815849

theorem circle_chord_intersect (r l d a b c : ℕ) (h₁ : r = 36) (h₂ : l = 90) (h₃ : d = 12)
  (h₄ : ∃ a b c, aπ - b√c = by calculation and c indivisible by prime square) :
  a + b + c = 216 :=
sorry

end circle_chord_intersect_l815_815849


namespace simplify_sqrt_expression_l815_815762

theorem simplify_sqrt_expression (x : ℝ) : 
  Real.sqrt (x^6 + x^4 + 1) = Real.sqrt (x^6 + x^4 + 1) := by
  sorry

end simplify_sqrt_expression_l815_815762


namespace average_salary_l815_815572

theorem average_salary (avg_officer_salary avg_nonofficer_salary num_officers num_nonofficers : ℕ) (total_salary total_employees : ℕ) : 
  avg_officer_salary = 430 → 
  avg_nonofficer_salary = 110 → 
  num_officers = 15 → 
  num_nonofficers = 465 → 
  total_salary = avg_officer_salary * num_officers + avg_nonofficer_salary * num_nonofficers → 
  total_employees = num_officers + num_nonofficers → 
  total_salary / total_employees = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l815_815572


namespace cut_sticks_to_non_triangle_l815_815291

def min_cut_length_to_prevent_triangle : ℕ := 17

theorem cut_sticks_to_non_triangle (x : ℕ) (h9 : 9 - x > 0) (h12 : 12 - x > 0) (h20 : 20 - x > 0) :
  (9 - x) + (12 - x) ≤ 20 - x ∨
  (9 - x) + (20 - x) ≤ 12 - x ∨
  (12 - x) + (20 - x) ≤ 9 - x ↔ x ≥ min_cut_length_to_prevent_triangle :=
begin
  sorry
end

end cut_sticks_to_non_triangle_l815_815291


namespace math_problem_prove_l815_815239

noncomputable def percentile_60th (data : List ℕ) : ℚ :=
  let sorted := data.sort
  let n := sorted.length
  if n = 0 then
    0
  else
    let position := (n * 60 / 100) - 1
    if position < n - 1 then
      (sorted.nth position).getD 0 + ((sorted.nth (position + 1)).getD 0 - (sorted.nth position).getD 0) / 2
    else
      (sorted.nth position).getD 0

def optionA : Bool :=
  percentile_60th [64, 91, 72, 75, 85, 76, 78, 86, 79, 92] = 79

def binomial_prob (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  nat.choose n k * p^k * (1 - p)^(n - k)

def optionB : Bool :=
  binomial_prob 4 (1/2) 3 = 1/4

noncomputable def normal_cdf (μ σ x : ℚ) : ℚ :=
  Real.err (σ * (x - μ))

def optionC : Bool :=
  let μ := 5
  let σ := ? -- missing parameter; assumed given via distribution
  normal_cdf μ σ 2 = 0.1 ∧ 1 - normal_cdf μ σ 2 - normal_cdf μ sigma 8 = 0.8

def optionD : Bool :=
  let students_11 := 400
  let students_12 := 360
  let total_selected := 57
  let selected_11 := 20
  let proportion := selected_11 / students_11.to_float
  let selected_12 := proportion * students_12.to_float
  let selected_13 := total_selected.to_float - selected_11 - selected_12
  abs (selected_13 - 19) < 1e-1

def final_result : Prop :=
  ¬optionA ∧ optionB ∧ optionC ∧ optionD

theorem math_problem_prove : final_result :=
by
  sorry

end math_problem_prove_l815_815239


namespace intersection_P_Q_l815_815889

open Set

-- Definitions for the conditions
def P : Set ℝ := {x | x + 2 ≥ x^2}
def Q : Set ℕ := {x | x ≤ 3}

-- Proof problem statement: Prove P ∩ Q = {0, 1, 2}
theorem intersection_P_Q : P ∩ Q = {0, 1, 2} :=
  sorry

end intersection_P_Q_l815_815889


namespace smallest_period_of_five_cycles_l815_815330

variable {f : ℝ → ℝ}

def periodic (T : ℝ) : Prop :=
  ∀ x, f(x + T) = f(x)

theorem smallest_period_of_five_cycles (h_periodic: ∃ T, periodic T)
  (h_cycles: ∀ x, x ∈ [0, 2*Real.pi] → f(x + (2*Real.pi / 5)) = f(x)) :
  ∃ T, T = (2 * Real.pi / 5) ∧ periodic T :=
sorry

end smallest_period_of_five_cycles_l815_815330


namespace area_of_triangle_TAD_l815_815109

noncomputable theory

open Real

-- Define the problem conditions
def is_convex_pentagon (ABTCD : Pent) (area : ℝ) : Prop := ABTCD.area = area
def equal_lengths (AB CD : ℝ) : Prop := AB = CD
def internally_tangent_circles (T A B C D : Point) : Prop := 
    ∃ γ₁ γ₂ : Circle, γ₁.internally_tangent γ₂ ∧ γ₁.contains T ∧ γ₁.contains A ∧ γ₁.contains B ∧ γ₂.contains T ∧ γ₂.contains C ∧ γ₂.contains D
def right_angle (T A D : Point) : Prop := ∠ ATD = π/2
def specific_angle_120 (T B C : Point) : Prop := ∠ BTC = 2 * π / 3
def segment_length (BT : ℝ) (T B : Point) : Prop := dist T B = BT
def segment_length' (CT : ℝ) (T C : Point) : Prop := dist T C = CT

-- Given points in the plane
variables (A B C D T : Point)
variables (AB CD BT CT : ℝ)

-- The Lean definition
theorem area_of_triangle_TAD :
  ∀ (P : Pent), is_convex_pentagon P 22 → 
  equal_lengths AB CD →
  internally_tangent_circles T A B C D →
  right_angle T A D →
  specific_angle_120 T B C →
  segment_length 4 T B → 
  segment_length' 5 T C → 
  ∃ (area_TAD : ℝ), area_TAD = 128 - 64 * sqrt 3 :=
by
  intros
  sorry

end area_of_triangle_TAD_l815_815109


namespace find_correction_time_l815_815695

-- Define the conditions
def loses_minutes_per_day : ℚ := 2 + 1/2
def initial_time_set : ℚ := 1 * 60 -- 1 PM in minutes
def time_on_march_21 : ℚ := 9 * 60 -- 9 AM in minutes on March 21
def total_minutes_per_day : ℚ := 24 * 60
def days_between : ℚ := 6 - 4/24 -- 6 days minus 4 hours

-- Calculate effective functioning minutes per day
def effective_minutes_per_day : ℚ := total_minutes_per_day - loses_minutes_per_day

-- Calculate the ratio of actual time to the watch's time
def time_ratio : ℚ := total_minutes_per_day / effective_minutes_per_day

-- Calculate the total actual time in minutes between initial set time and the given time showing on the watch
def total_actual_time : ℚ := days_between * total_minutes_per_day + initial_time_set

-- Calculate the actual time according to the ratio
def actual_time_according_to_ratio : ℚ := total_actual_time * time_ratio

-- Calculate the correction required 'n'
def required_minutes_correction : ℚ := actual_time_according_to_ratio - total_actual_time

-- The theorem stating that the required correction is as calculated
theorem find_correction_time : required_minutes_correction = (14 + 14/23) := by
  sorry

end find_correction_time_l815_815695


namespace range_of_a_for_extreme_points_l815_815403

theorem range_of_a_for_extreme_points :
  (∃ (a : ℝ), (∀ (f : ℝ → ℝ), f = (λ x, x * (Real.log x - a * x)) →
    ∃ (g : ℝ → ℝ), g = (λ x, Real.log x + 1 - 2 * a * x) ∧
    ∃ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ g x1 = 0 ∧ g x2 = 0)) ↔ 
    (0 < a ∧ a < 1/2) :=
by
  sorry

end range_of_a_for_extreme_points_l815_815403


namespace larger_number_l815_815974

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l815_815974


namespace julia_kid_hours_total_l815_815503

/-- Julia played with different numbers of kids for various hours over the week. 
We define the total kid-hours for each day and sum them up to find the total kid-hours 
for the week, which should be equal to 114.75 kid-hours. -/
theorem julia_kid_hours_total :
  let monday_kid_hours := 17 * 1.5 in
  let tuesday_kid_hours := 15 * 2.25 in
  let wednesday_kid_hours := 2 * 1.75 in
  let thursday_kid_hours := 12 * 2.5 in
  let friday_kid_hours := 7 * 3 in
  (monday_kid_hours + tuesday_kid_hours + wednesday_kid_hours + thursday_kid_hours + friday_kid_hours) = 114.75 :=
by
  sorry

end julia_kid_hours_total_l815_815503


namespace number_of_tables_l815_815694

-- Define the total number of customers the waiter is serving
def total_customers := 90

-- Define the number of women per table
def women_per_table := 7

-- Define the number of men per table
def men_per_table := 3

-- Define the total number of people per table
def people_per_table : ℕ := women_per_table + men_per_table

-- Statement to prove the number of tables
theorem number_of_tables (T : ℕ) (h : T * people_per_table = total_customers) : T = 9 := by
  sorry

end number_of_tables_l815_815694


namespace chris_age_approx_16_l815_815571

-- Define variables representing the ages of Amy, Ben, and Chris
variables (a b c : ℕ)

-- State the conditions
def age_conditions : Prop :=
  (a + b + c) / 3 = 10 ∧
  c - 5 = 2 * (a - 5) ∧
  b + 5 = (a + 5) / 2

-- State the theorem to be proved
theorem chris_age_approx_16 (hc : age_conditions a b c) : abs (c - 16) ≤ 1 :=
sorry

end chris_age_approx_16_l815_815571


namespace divide_students_into_groups_l815_815853

-- Define a structure to represent the groups
structure Group :=
  (english : Nat)
  (french : Nat)
  (spanish : Nat)

-- Define a predicate to check if a list of groups satisfies the condition
def satisfies_conditions (groups : List Group) : Prop :=
  groups.length = 5 ∧ 
  (∀ g ∈ groups, g.english = 10 ∧ g.french = 10 ∧ g.spanish = 10)

theorem divide_students_into_groups :
  ∃ (groups : List Group), satisfies_conditions groups :=
sorry

end divide_students_into_groups_l815_815853


namespace liam_birthday_next_monday_2018_l815_815906

-- Define year advancement rules
def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

-- Define function to calculate next weekday
def next_weekday (current_day : ℕ) (years_elapsed : ℕ) : ℕ :=
  let advance := (years_elapsed / 4) * 2 + (years_elapsed % 4)
  (current_day + advance) % 7

theorem liam_birthday_next_monday_2018 :
  (next_weekday 4 3 = 0) :=
sorry

end liam_birthday_next_monday_2018_l815_815906


namespace rate_in_still_water_l815_815681

-- Variables declaration
variables (v c : ℝ) 

-- Given conditions
def downstream_rate := v + c = 32
def upstream_rate := v - c = 17
def current_rate := c = 7.5 

-- The final statement we want to prove
theorem rate_in_still_water : downstream_rate v c ∧ upstream_rate v c ∧ current_rate c → v = 24.5 :=
by
  intros
  have h1 : v + c = 32 := ‹v + c = 32›
  have h2 : v - c = 17 := ‹v - c = 17›
  have h3 : c = 7.5 := ‹c = 7.5›
  sorry

end rate_in_still_water_l815_815681


namespace find_f_4_l815_815952

-- Lean code to encapsulate the conditions and the goal
theorem find_f_4 (f : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), x * f y = y * f x)
  (h2 : f 12 = 24) : 
  f 4 = 8 :=
sorry

end find_f_4_l815_815952


namespace Sn_eq_1_over_n_l815_815512

noncomputable def a_seq : ℕ → ℚ
| 0       := 0
| 1       := 1
| (n+2) := - (S_seq (n+1) * S_seq n)

noncomputable def S_seq : ℕ → ℚ
| 0       := 0
| 1       := 1
| (n+1) := S_seq n + a_seq (n+1)

theorem Sn_eq_1_over_n (n : ℕ) (hn : n > 0) : S_seq n = 1 / n :=
by sorry

end Sn_eq_1_over_n_l815_815512


namespace volume_equality_of_tetrahedrons_l815_815364

def Point := ℝ × ℝ × ℝ
def Line := Point × Point

def parallel (p₁ p₂ : Line) : Prop :=
  ∃ k : ℝ, ∀ (t : ℝ), fst p₂ + t * snd p₂ = fst p₁ + k * t * snd p₁

def volume (a b c d : Point) : ℝ :=
  abs (1 / 6 * determinant (
    ![[1, 1, 1, 1],
      [a.fst, b.fst, c.fst, d.fst],
      [a.snd, b.snd, c.snd, d.snd],
      [a.2, b.2, c.2, d.2]]))

variables (A B C D A' B' C' D' : Point)
variables (p : Line)
variables (P P' : Point → Prop)

-- Conditions of the problem
axiom hP : ∀ x, P x → ¬p.fst = x ∧ ¬p.snd = x
axiom hP' : ∀ x, P' x → ¬p.fst = x ∧ ¬p.snd = x
axiom h_parallel : parallel (A, A') (B, B') ∧ parallel (A, A') (C, C') ∧ parallel (A, A') (D, D')

-- Volume equality proof statement
theorem volume_equality_of_tetrahedrons (h_noncollinear : ¬ collinear A B C) :
  volume A B C D' = volume A' B' C' D :=
sorry

end volume_equality_of_tetrahedrons_l815_815364


namespace time_savings_l815_815936

   def distance_per_day : ℝ := 3
   def days : ℕ := 4
   def speed_mon : ℝ := 6
   def speed_tue : ℝ := 4
   def speed_wed : ℝ := 3
   def speed_thu : ℝ := 5
   def constant_speed : ℝ := 5

   theorem time_savings :
     let time_day (distance speed : ℝ) := distance / speed in
     let total_time := 
           time_day distance_per_day speed_mon
         + time_day distance_per_day speed_tue
         + time_day distance_per_day speed_wed
         + time_day distance_per_day speed_thu in
     let constant_total_time := (distance_per_day * days) / constant_speed in
     let time_saved_hrs := total_time - constant_total_time in
     let time_saved_mins := time_saved_hrs * 60 in
     time_saved_mins = 27 :=
   by
     sorry
   
end time_savings_l815_815936


namespace converse_proposition_l815_815575

-- Define the propositions p and q
variables (p q : Prop)

-- State the problem as a theorem
theorem converse_proposition (p q : Prop) : (q → p) ↔ ¬p → ¬q ∧ ¬q → ¬p ∧ (p → q) := 
by 
  sorry

end converse_proposition_l815_815575


namespace little_twelve_conference_games_l815_815160

def teams_in_division : ℕ := 6
def divisions : ℕ :=  2

def games_within_division (t : ℕ) : ℕ := (t * (t - 1)) / 2 * 2

def games_between_divisions (d t : ℕ) : ℕ := t * t

def total_conference_games (d t : ℕ) : ℕ :=
  d * games_within_division t + games_between_divisions d t

theorem little_twelve_conference_games :
  total_conference_games divisions teams_in_division = 96 :=
by
  sorry

end little_twelve_conference_games_l815_815160


namespace cost_price_l815_815297

theorem cost_price (MP SP C : ℝ) (h1 : MP = 112.5) (h2 : SP = 0.95 * MP) (h3 : SP = 1.25 * C) : 
  C = 85.5 :=
by
  sorry

end cost_price_l815_815297


namespace placing_points_in_square_l815_815876

theorem placing_points_in_square (side_len : ℝ) (num_points : ℕ) : 
  side_len = 1 ∧ num_points = 1965 →
  (∀ (rect : set (ℝ × ℝ)), 
    (∃ l w : ℝ, l * w = 1 / 200 ∧ 
      (∀ x₁ y₁ x₂ y₂ : ℝ, 
        x₁ < x₂ ∧ y₁ < y₂ → 
        (l = |x₂ - x₁| ∨ l = |y₂ - y₁| → 
         w = |x₂ - x₁| ∨ w = |y₂ - y₁| →
         rect = set.inter (set.Icc (x₁, y₁)) (set.Icc (x₂, y₂))))) →
    ∃ p ∈ set.finset_univ (finset.range num_points), 
       set.finite (set.inter p rect)) 
→  true :=
begin
  sorry
end

end placing_points_in_square_l815_815876


namespace ellipse_touch_locus_l815_815689

noncomputable def touch_point_locus {AB : Segment} {C : Point} {O : Point} (ellipse_centered_at_O : Ellipse centered_at O) : Locus :=
  let D' := reflect_point (midpoint AB) B in
  circle_with_diameter B D'

theorem ellipse_touch_locus (AB : Segment) (C : Point) (hC : is_perpendicular_bisector C AB) 
  (O : Point) (hO : is_circumcircle_point O ABC C) 
  (ellipse : Ellipse) (hEllipse : is_centered_at ellipse O) 
  (hTouch : touches ellipse AB ∧ touches ellipse BC ∧ touches ellipse CA) :
  touch_point_locus ellipse = circle_with_diameter B (reflect_point (midpoint AB) B) := 
sorry

end ellipse_touch_locus_l815_815689


namespace probability_calculation_l815_815615

noncomputable def population_mean (scores : List ℤ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def sample_means (scores : List ℤ) : List ℚ :=
  (scores.toFinset.powerset.filter (λ s, s.card = 2)).val.map (λ s, ((s.toList.sum : ℚ) / 2))

def count_within_range (means : List ℚ) (mean : ℚ) (range : ℚ) : ℕ :=
  means.count (λ m, |m - mean| ≤ range)
  
def probability_of_within_range (scores : List ℤ) : ℚ :=
  let means := sample_means scores in
  count_within_range means (population_mean scores) 0.5 / means.length

theorem probability_calculation :
  let scores := [5, 6, 7, 8, 9, 10]
  in probability_of_within_range scores = 7/15 := by 
  sorry

end probability_calculation_l815_815615


namespace hyperbola_foci_distance_l815_815341

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 - 4 * x - 9 * y^2 - 18 * y = 56

-- Define the distance between the foci of the hyperbola
def distance_between_foci (d : ℝ) : Prop :=
  d = 2 * Real.sqrt (170 / 3)

-- The theorem stating that the distance between the foci of the given hyperbola
theorem hyperbola_foci_distance :
  ∃ d, hyperbola_eq x y → distance_between_foci d :=
by { sorry }

end hyperbola_foci_distance_l815_815341


namespace find_x_coordinate_l815_815871

theorem find_x_coordinate 
  (x : ℝ)
  (h1 : (0, 0) = (0, 0))
  (h2 : (0, 4) = (0, 4))
  (h3 : (x, 4) = (x, 4))
  (h4 : (x, 0) = (x, 0))
  (h5 : 0.4 * (4 * x) = 8)
  : x = 5 := 
sorry

end find_x_coordinate_l815_815871


namespace max_sum_min_two_sequences_l815_815935
open Nat

theorem max_sum_min_two_sequences :
  ∀ (a b: Fin 20 → Fin 40),
  (∀ n1 : Fin 20, ∀ n2 : Fin 20, n1 ≠ n2 → (a n1) ≠ (a n2) ∧ (b n1) ≠ (b n2) ∧ (a n1) ≠ (b n2)) →
  (∃! v : Fin 40, (∃ (x : Fin 20), (v = a x)) ∨ (∃ (y : Fin 20), (v = b y))) →
  (∑ i in Finset.range 20, ∑ j in Finset.range 20, min (a i) (b j)) ≤ 5530 :=
by
  intros a b h_unique_combined h_all_numbers_once
  sorry

end max_sum_min_two_sequences_l815_815935


namespace find_b_l815_815591

noncomputable def P : Polynomial ℝ := X^3 + a * X^2 + b * X + c

variables (a b c : ℝ)

-- Conditions from the problem
def condition1 : c = 5 := sorry
def meanOfZeros : -a / 3 = -c := sorry
def sumOfCoefficients : 1 + a + b + c = -a / 3 := sorry

-- The main theorem to prove
theorem find_b : c = 5 ∧ -a / 3 = -c ∧ 1 + a + b + c = -a / 3 → b = -26 :=
by
  intros h
  cases h with hc hf_other
  cases hf_other with hf hsum
  sorry

end find_b_l815_815591


namespace length_AC_eq_sqrt6_l815_815092

-- Definitions for the conditions
variables {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (O1 O2 : Type*) [MetricSpace O1] [MetricSpace O2]
variables (R1 R2 : ℝ)
variable (AC : ℝ)

-- Constants given in the problem
constant angle_B_eq_pi_six : ∀ (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C], 
  ∀ (α β : ℝ), β = π / 6 → α + β = π - β → α = π - (π - β)

constant radius_circle1 : R1 = 2
constant radius_circle2 : R2 = 3

open Metric

-- Theorem to prove
theorem length_AC_eq_sqrt6 {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C]
   (O1 O2 : Type*) [MetricSpace O1] [MetricSpace O2] 
    (R1 R2 : ℝ) (AC : ℝ)
  [h1 : radius_circle1 R1]
  [h2 : radius_circle2 R2]
  (h3 : angle_B_eq_pi_six A B C (AC:ℝ) (π / 6)) : 
  AC = sqrt 6 := 
by sorry

end length_AC_eq_sqrt6_l815_815092


namespace hyperbola_eqn_l815_815025

-- Definitions of given conditions
def a := 4
def b := 3
def c := 5

def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Hypotheses derived from conditions
axiom asymptotes : b / a = 3 / 4
axiom right_focus : a^2 + b^2 = c^2

-- Main theorem statement
theorem hyperbola_eqn : (forall x y, hyperbola x y ↔ x^2 / 16 - y^2 / 9 = 1) :=
by
  intros
  sorry

end hyperbola_eqn_l815_815025


namespace calculate_area_hexagon_and_pq_result_l815_815111

noncomputable def coordinates : ℝ×ℝ := (0, 0)

noncomputable def pointB : ℝ×ℝ := sorry

noncomputable def pointF : ℝ×ℝ := sorry

noncomputable def area_hexagon (A B F : ℝ×ℝ) : ℝ := sorry

theorem calculate_area_hexagon_and_pq_result (p q : ℕ) (h1 : q = 3) :
  ∃ (A B F : ℝ×ℝ)
    (C D E : ℝ×ℝ),
  A = (0,0) ∧ B = (sorry, 3) ∧ F = (sorry, 6) ∧
  (hexagon_property_1 A B C D E F) ∧
  (hexagon_property_2 A B C D E F) ∧
  (hexagon_property_3 A B C D E F) ∧
  distinct_y_coordinates [0,3,6,9,12,15] [A.2, B.2, C.2, D.2, E.2, F.2] ∧
  area_hexagon A B F = 144 * Real.sqrt 3 ∧
  p + q = 147 :=
begin
  sorry
end

end calculate_area_hexagon_and_pq_result_l815_815111


namespace obtuse_angles_from_incenter_l815_815922

noncomputable def angle_sum (a b c : ℝ) : ℝ := a + b + c

theorem obtuse_angles_from_incenter
  (A B C : Type*)
  [metric_space A] [metric_space B] [metric_space C]
  (ABC_triangle : is_triangle A B C)
  (O : incenter A B C)
  (angle_BAC angle_ABC angle_ACB : ℝ)
  (angle_AOB angle_BOC angle_COA : ℝ) :
  angle_BOC = 180 - angle_BAC / 2 →
  angle_COA = 180 - angle_ABC / 2 →
  angle_AOB = 180 - angle_ACB / 2 →
  angle_BAC > 0 → 
  angle_BOC > 90 ∧ angle_COA > 90 ∧ angle_AOB > 90 := by
    intros h1 h2 h3 h4
    sorry

end obtuse_angles_from_incenter_l815_815922


namespace coin_heads_probability_l815_815685

theorem coin_heads_probability :
  let p := 0.5
  let q := 1 - p
  let n := 4
  let k := 2
  p = 0.5 ∧ q = 0.5 →
  let binomial_coefficient := Nat.choose n k
  let prob := binomial_coefficient * (p^k) * (q^(n-k))
  prob = 0.375 :=
by
  intros p q n k p_eq q_eq
  have h1: p = 0.5 := p_eq
  have h2: q = 0.5 := q_eq
  have h3 : binomial_coefficient = Nat.choose 4 2 := rfl
  have h4 : prob = 6 * (0.5^2) * (0.5^2) := sorry
  have h5 : prob = 6 * (1 / 16) := sorry
  have h6 : prob = 0.375 := sorry
  exact h6

end coin_heads_probability_l815_815685


namespace max_unique_numbers_in_notebook_l815_815211

theorem max_unique_numbers_in_notebook :
  ∀ (n: ℕ) (N: ℕ) (a: ℕ → set ℕ),
  n = 18 →
  (∀ i, 1 ≤ i ∧ i ≤ n → (a i).card ≥ 10) →
  (∀ i, 1 ≤ i ∧ i ≤ n - 2 → ((a i) ∪ (a (i+1)) ∪ (a (i+2))).card ≤ 20) →
  (N = (finset.range n).sum (λ i, (a i ∪ a (i + 1)).card - (a i ∩ a (i + 1)).card)) →
  N ≤ 190 :=
by
  sorry

end max_unique_numbers_in_notebook_l815_815211


namespace simplify_expression_l815_815770

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end simplify_expression_l815_815770


namespace vacant_seats_calculation_l815_815859

noncomputable def seats_vacant (total_seats : ℕ) (percentage_filled : ℚ) : ℚ := 
  total_seats * (1 - percentage_filled)

theorem vacant_seats_calculation: 
  seats_vacant 600 0.45 = 330 := 
by 
    -- sorry to skip the proof.
    sorry

end vacant_seats_calculation_l815_815859


namespace smallest_angle_l815_815891

-- Let a, b, c be vectors in a Euclidean space.
variables (a b c : EuclideanSpace ℝ)

-- Conditions given in the problem
variables (h1 : ∥a∥ = 1)
variables (h2 : ∥b∥ = 1)
variables (h3 : ∥c∥ = 3)
variables (h4 : a × (a × c) + b + c = 0)

-- Statement of the problem in Lean 4
theorem smallest_angle (a b c : EuclideanSpace ℝ) 
  (h1 : ∥a∥ = 1) 
  (h2 : ∥b∥ = 1) 
  (h3 : ∥c∥ = 3) 
  (h4 : a × (a × c) + b + c = 0) : 
  ∃ θ : ℝ, θ = real.arccos (1 / 3) := 
sorry

end smallest_angle_l815_815891


namespace triangle_AC_length_l815_815496

theorem triangle_AC_length (A B C : Type) [Inhabited C]
  (h1 : right_angle A C B) 
  (h2 : tan_C A B C = 4 / 3)
  (h3 : BC_length B C = 3) : 
  AC_length A C B = 5 := 
sorry

end triangle_AC_length_l815_815496


namespace measure_angle_BDC_l815_815069

theorem measure_angle_BDC {A B C D E : Type*} -- Define the points
  (angle_A : ℝ) (angle_E : ℝ) (angle_C : ℝ)
  (h_angle_A : angle_A = 70) (h_angle_E : angle_E = 50) (h_angle_C : angle_C = 40)
  (h_D_on_BC_extension : ∃ D, D ∉ segment B C)
  (h_E_on_AB_extension : ∃ E, E ∉ segment A B) :
  angle_BDC = 60 := 
by 
  sorry

end measure_angle_BDC_l815_815069


namespace number_of_integers_satisfying_inequality_l815_815441

theorem number_of_integers_satisfying_inequality : 
  (finset.filter (λ x, (x - 2)^2 ≤ 4) (finset.Icc 0 4)).card = 5 := 
  sorry

end number_of_integers_satisfying_inequality_l815_815441


namespace cannot_finish_third_l815_815860

theorem cannot_finish_third (P Q R S T : Type) 
  (P_beats_Q : P ≠ Q → P < Q) 
  (P_beats_R : P ≠ R → P < R) 
  (P_beats_S : P ≠ S → P < S)
  (Q_beats_S : Q ≠ S → Q < S)
  (S_beats_R : S ≠ R → S < R)
  (T_finishes_after_P_before_Q : P < T ∧ T < Q) :
  ¬(P = third) ∧ ¬(R = third) :=
begin
  sorry
end

end cannot_finish_third_l815_815860


namespace andrew_permit_rate_l815_815704

def permits_per_hour (a h_a H T : ℕ) : ℕ :=
  T / (H - (a * h_a))

theorem andrew_permit_rate :
  permits_per_hour 2 3 8 100 = 50 := by
  sorry

end andrew_permit_rate_l815_815704


namespace findFirstCarSpeed_l815_815617

noncomputable def firstCarSpeed (v : ℝ) (blackCarSpeed : ℝ) (initialGap : ℝ) (timeToCatchUp : ℝ) : Prop :=
  blackCarSpeed * timeToCatchUp = initialGap + v * timeToCatchUp → v = 30

theorem findFirstCarSpeed :
  firstCarSpeed 30 50 20 1 :=
by
  sorry

end findFirstCarSpeed_l815_815617


namespace rhombus_area_l815_815951

-- Statement of the proof problem
theorem rhombus_area (a b c d : ℂ) (h : (polynomial.from_roots [a, b, c, d] = λ z, z^4 + 4i*z^3 + (-5 + 5i)*z^2 + (-10 - i)*z + (1 - 6i)))
  (rhombus_form : ∃ (O : ℂ), (O = -i) ∧ is_rhombus ([a, b, c, d] : list ℂ) O ):
  2 * (|a + i|) * (|b + i|) = sqrt 10 := by
sorry

end rhombus_area_l815_815951


namespace integer_solution_count_l815_815455

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end integer_solution_count_l815_815455


namespace distance_point_to_line_eq_l815_815752

noncomputable def distance_from_point_to_line : ℝ :=
  let p : ℝ × ℝ × ℝ := (2, 0, 1)
  let a : ℝ × ℝ × ℝ := (1, 3, 2)
  let b : ℝ × ℝ × ℝ := (2, -1, 0)
  let direction : ℝ × ℝ × ℝ := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let t := 5 / 7
  let v_t : ℝ × ℝ × ℝ := (a.1 + t * direction.1, a.2 + t * direction.2, a.3 + t * direction.3)
  let vector_difference : ℝ × ℝ × ℝ := (p.1 - v_t.1, p.2 - v_t.2, p.3 - v_t.3)
  real.sqrt ((vector_difference.1 * vector_difference.1) + (vector_difference.2 * vector_difference.2) + (vector_difference.3 * vector_difference.3))

theorem distance_point_to_line_eq :
  distance_from_point_to_line = (real.sqrt 6) / 7 :=
sorry

end distance_point_to_line_eq_l815_815752


namespace initial_snails_collected_l815_815038

theorem initial_snails_collected (sea_stars : ℕ) (seashells : ℕ) (total_items : ℕ) (lost_sea_creatures : ℕ) (remaining_items : ℕ) :
    sea_stars = 34 →
    seashells = 21 →
    total_items = 59 →
    lost_sea_creatures = 25 →
    remaining_items = 59 →
    34 + 21 - 25 + (remaining_items - (34 + 21 - 25)) = 29 :=
by
  intros h1 h2 h3 h4 h5
  have h_sum_sea_creatures : 34 + 21 = 55, from rfl
  have h_remaining_sea_creatures : 55 - 25 = 30, from rfl
  have h_remaining_snails : remaining_items - 30 = 29, from rfl
  rw [h1, h2, h3, h4, h5] at h_remaining_snails
  exact h_remaining_snails

end initial_snails_collected_l815_815038


namespace fraction_of_eggs_hatched_l815_815354

theorem fraction_of_eggs_hatched
  (initial_doves : ℕ)
  (eggs_per_dove : ℕ)
  (total_doves : ℕ)
  (initial_doves = 20)
  (eggs_per_dove = 3)
  (total_doves = 65)
  : (45 : ℚ) / (60 : ℚ) = 3 / 4 :=
by
  sorry

end fraction_of_eggs_hatched_l815_815354


namespace probability_X_greater_than_4_l815_815391

noncomputable def pdf (x : ℝ) : ℝ := (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(x - 2)^2 / 2)

-- Condition: The integral of the pdf from 0 to 2 is 1/3
axiom integral_condition : ∫ x in 0..2, pdf x = 1/3

-- Prove that the probability that X is greater than 4 is 1/6
theorem probability_X_greater_than_4 : ∫ x in 4..Real.inf, pdf x = 1/6 :=
by
  -- Ensure usage of the integral_condition and the symmetry of the normal distribution
  -- The proof itself is not required, so we use sorry to skip it
  sorry

end probability_X_greater_than_4_l815_815391


namespace AQI_data_median_is_184_5_l815_815941

open List

/--
The Hefei Environmental Protection Central Station released the Air Quality Index (AQI) data from January 11 to January 20, 2014.
The data are as follows: 153, 203, 268, 166, 157, 164, 268, 407, 335, 119.
-/
def AQI_data : List ℝ := [153, 203, 268, 166, 157, 164, 268, 407, 335, 119]

/--
To find the median of the AQI data set.
-/
def median (l : List ℝ) : ℝ :=
  let sorted_l := sort l
  if (length sorted_l) % 2 = 1 then
    -- If the length of the list is odd, take the middle element
    nth_le sorted_l ((length sorted_l) / 2) (by sorry)
  else
    -- If the length of the list is even, take the average of the two middle elements
    (nth_le sorted_l ((length sorted_l) / 2 - 1) (by sorry) + nth_le sorted_l ((length sorted_l) / 2) (by sorry)) / 2

/--
Proof that the median of the AQI data set is 184.5.
-/
theorem AQI_data_median_is_184_5 : median AQI_data = 184.5 :=
by sorry

end AQI_data_median_is_184_5_l815_815941


namespace unique_f_value_f_2m_l815_815555

def f : ℕ+ → ℕ+
noncomputable def f (n : ℕ+) : ℕ+ :=
  if n = 1 then 1
  else if n = 2 then 1
  else sorry

theorem unique_f (f g : ℕ+ → ℕ+) 
  (h1 : f 1 = 1) (h2 : f 2 = 1)
  (h3 : ∀ n : ℕ+, n ≥ 3 → f n = f (f (n - 1)) + f (n - f (n - 1)))
  (h4 : g 1 = 1) (h5 : g 2 = 1)
  (h6 : ∀ n : ℕ+, n ≥ 3 → g n = g (g (n - 1)) + g (n - g (n - 1))) :
  f = g :=
sorry

theorem value_f_2m (f : ℕ+ → ℕ+)
  (h1 : f 1 = 1) (h2 : f 2 = 1)
  (h3 : ∀ n : ℕ+, n ≥ 3 → f n = f (f (n - 1)) + f (n - f (n - 1)))
  (m : ℕ) (hm : m ≥ 2) :
  f (2 ^ m) = 2 ^ (m - 1) :=
sorry

end unique_f_value_f_2m_l815_815555


namespace susan_structure_blocks_l815_815567

theorem susan_structure_blocks :
  ∃ (length width height floor_thickness wall_thickness : ℕ),
    length = 16 ∧
    width = 12 ∧
    height = 8 ∧
    floor_thickness = 2 ∧
    wall_thickness = 1.5 ∧
    (let total_volume := length * width * height in
     let effective_height := height - 2 * floor_thickness in
     let effective_length := length - 2 * wall_thickness in
     let effective_width := width - 2 * wall_thickness in
     let internal_volume := effective_length * effective_width * effective_height in
     total_volume - internal_volume = 1068) :=
by
  sorry

end susan_structure_blocks_l815_815567


namespace fewer_peppers_bought_l815_815267

-- Define the initial conditions
def peppers_for_very_spicy : ℕ := 3
def peppers_for_spicy : ℕ := 2
def peppers_for_mild : ℕ := 1

def initial_very_spicy_curries : ℕ := 30
def initial_spicy_curries : ℕ := 30
def initial_mild_curries : ℕ := 10

def current_spicy_curries : ℕ := 15
def current_mild_curries : ℕ := 90

-- Define the total peppers bought initially and currently
def total_initial_peppers : ℕ :=
  (initial_very_spicy_curries * peppers_for_very_spicy) +
  (initial_spicy_curries * peppers_for_spicy) +
  (initial_mild_curries * peppers_for_mild)

def total_current_peppers : ℕ :=
  (current_spicy_curries * peppers_for_spicy) +
  (current_mild_curries * peppers_for_mild)

-- Proof statement: The difference in the number of peppers bought is 40
theorem fewer_peppers_bought : total_initial_peppers - total_current_peppers = 40 :=
by 
  unfold total_initial_peppers 
  unfold total_current_peppers 
  unfold initial_very_spicy_curries 
  unfold initial_spicy_curries 
  unfold initial_mild_curries 
  unfold current_spicy_curries 
  unfold current_mild_curries 
  unfold peppers_for_very_spicy 
  unfold peppers_for_spicy 
  unfold peppers_for_mild 
  norm_num
  sorry -- Proof goes here

end fewer_peppers_bought_l815_815267


namespace total_people_in_house_l815_815640

-- Define the number of people in various locations based on the given conditions.
def charlie_and_susan := 2
def sarah_and_friends := 5
def people_in_bedroom := charlie_and_susan + sarah_and_friends
def people_in_living_room := 8

-- Prove the total number of people in the house is 14.
theorem total_people_in_house : people_in_bedroom + people_in_living_room = 14 := by
  -- Here we can use Lean's proof system, but we skip with 'sorry'
  sorry

end total_people_in_house_l815_815640


namespace proof_problem_l815_815413

-- Define f(x) as |log_2 x|
def f (x : ℝ) : ℝ := abs (Real.log x / Real.log 2)

-- Proposition p
def p : Prop := ∀ x > 1, ∃ y ∈ set.Ici 0, f x = y

-- Proposition q
def q : Prop := ∃ m ≥ 0, (2 * π / abs m < (π / 2))

-- The proof statement
theorem proof_problem : ¬ (p ∧ ¬ q) ∧ ¬ (¬ p ∧ q) :=
by 
  sorry

end proof_problem_l815_815413


namespace total_value_of_treats_l815_815724

def hotel_cost_per_night : ℕ := 4000
def number_of_nights : ℕ := 2
def car_cost : ℕ := 30000
def house_multiplier : ℕ := 4

theorem total_value_of_treats : 
  (number_of_nights * hotel_cost_per_night) + car_cost + (house_multiplier * car_cost) = 158000 := 
by
  sorry

end total_value_of_treats_l815_815724


namespace sum_divisor_floor_eq_l815_815552

open Nat

def sigma0 (k : ℕ) : ℕ :=
  (Finset.range k).filter (λ d => d + 1 ∣ k).card

theorem sum_divisor_floor_eq (n : ℕ) :
  (∑ k in Finset.range (2 * n), sigma0 (k + 1)) -
  (∑ k in Finset.range n, (2 * n) / (k + 1)) = n := by
  sorry

end sum_divisor_floor_eq_l815_815552


namespace cuboid_surface_area_l815_815527

variables {a b c : ℝ}

theorem cuboid_surface_area (a b c : ℝ) : 
  let SA := 2 * (a * b) + 2 * (b * c) + 2 * (a * c) 
  in SA = 2 * a * b + 2 * b * c + 2 * a * c := 
by 
  sorry

end cuboid_surface_area_l815_815527


namespace pears_value_equivalence_l815_815155

theorem pears_value_equivalence :
  (3 / 4 * 12 = 9) ∧ (9 * a = 10 * p) ∧ (2 / 3 * 9 * a = 6 * a) ⟹ (6 * (10 / 9) * p = 20 / 3 * p) :=
by
  sorry

end pears_value_equivalence_l815_815155


namespace number_of_integers_satisfying_inequality_l815_815439

theorem number_of_integers_satisfying_inequality : 
  (finset.filter (λ x, (x - 2)^2 ≤ 4) (finset.Icc 0 4)).card = 5 := 
  sorry

end number_of_integers_satisfying_inequality_l815_815439


namespace James_total_area_l815_815095

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase_dimension : ℕ := 2
def number_of_new_rooms : ℕ := 4
def larger_room_multiplier : ℕ := 2

noncomputable def new_length : ℕ := initial_length + increase_dimension
noncomputable def new_width : ℕ := initial_width + increase_dimension
noncomputable def area_of_one_new_room : ℕ := new_length * new_width
noncomputable def total_area_of_4_rooms : ℕ := area_of_one_new_room * number_of_new_rooms
noncomputable def area_of_larger_room : ℕ := area_of_one_new_room * larger_room_multiplier
noncomputable def total_area : ℕ := total_area_of_4_rooms + area_of_larger_room

theorem James_total_area : total_area = 1800 := 
by
  sorry

end James_total_area_l815_815095


namespace probability_of_falling_into_A_l815_815355

theorem probability_of_falling_into_A :
  let Ω := {p : ℝ × ℝ | abs p.1 ≤ 1 ∧ abs p.2 ≤ 1}
  let A := {p : ℝ × ℝ | p.2 = p.1 ∧ (p.1, p.2) ∈ Ω ∨ p.2 = p.1^2 ∧ (p.1, p.2) ∈ Ω}
  let area_A := ∫ x in 0..1, x - x^2
  let area_Ω := (2 : ℝ) * (2 : ℝ)
  area_A / area_Ω = 1 / 24 :=
by
  sorry

end probability_of_falling_into_A_l815_815355


namespace Smiths_Bakery_Sales_l815_815904

noncomputable def M : ℕ := 16
noncomputable def S : ℝ := 1.5 * M * (M^2 - Real.sqrt M)

theorem Smiths_Bakery_Sales : S = 6048 := by
  unfold M S
  sorry

end Smiths_Bakery_Sales_l815_815904


namespace min_value_of_quadratic_l815_815756

theorem min_value_of_quadratic :
  ∃ (x y : ℝ), 2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x - 6 * y + 1 = -3 :=
sorry

end min_value_of_quadratic_l815_815756


namespace intersection_A_B_l815_815826

noncomputable def A : set ℝ := { x | 3 ^ (x * (x - 3)) < 1 }

noncomputable def B : set ℝ := { x | ∃ y, y = real.sqrt (real.log2 (x - 1)) }

/-- The statement in the proof form -/
theorem intersection_A_B :
  (A ∩ B) = { x | 2 ≤ x ∧ x < 3 } := sorry

end intersection_A_B_l815_815826


namespace longest_rod_in_cube_l815_815653

theorem longest_rod_in_cube (s : ℝ) (h : s = 4) : 
  let d := Real.sqrt (s^2 + s^2 + s^2)
  in d = 4 * Real.sqrt 3 := by
  sorry

end longest_rod_in_cube_l815_815653


namespace main_theorem_l815_815816

noncomputable def f (x a : ℝ) : ℝ :=
  (1 / 2) * x^2 - (a - 1) * x - a * Real.log x

variable (a b x₁ x₂ : ℝ)

theorem main_theorem
(hf : f x₁ a = b)
(hf2 : f x₂ a = b)
(hne : x₁ ≠ x₂) :
  f' ((x₁ + x₂) / 2) a > 0 :=
sorry

end main_theorem_l815_815816


namespace fraction_is_square_l815_815376

theorem fraction_is_square (a b : ℕ) (hpos_a : a > 0) (hpos_b : b > 0) 
  (hdiv : (ab + 1) ∣ (a^2 + b^2)) :
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end fraction_is_square_l815_815376


namespace eighth_square_shading_l815_815060

theorem eighth_square_shading :
  ∀ (n : ℕ), n = 8 → (n^2 : ℕ) / (n^2 : ℕ) = (1 : ℚ) :=
by
intros n hn
rw hn
norm_num

end eighth_square_shading_l815_815060


namespace reciprocal_of_negative_half_l815_815185

def n : ℝ := -1 / 2

theorem reciprocal_of_negative_half : 1 / n = -2 :=
by
  -- Proof to be filled in
  sorry

end reciprocal_of_negative_half_l815_815185


namespace triangle_area_parabola_l815_815117

theorem triangle_area_parabola
  (a b : ℝ)  -- These represent |OF| and |PQ| respectively
  (h_a_pos : a > 0)  -- Ensure that distances are positive
  (h_b_pos : b > 0)
  (O F P Q : ℝ × ℝ)  -- Points O, F, P, and Q in the plane
  (h_O_vertex : O = (0, 0))  -- O is the vertex of the parabola
  (h_F_focus : F = (a, 0))  -- F is the focus of the parabola at distance a from O
  (h_PQ_chord_through_F : ∃ θ : ℝ, P = (a * cos θ, a * sin θ) ∧ Q = (a * cos (π + θ), a * sin (π + θ)))  -- PQ is a chord through F
  (h_PQ_length : dist P Q = b) :  -- Length of PQ is b
  area (triangle.mk O P Q) = a * sqrt (a * b) := sorry

end triangle_area_parabola_l815_815117


namespace num_integers_satisfying_inequality_l815_815426

theorem num_integers_satisfying_inequality : 
  {x : ℤ | (x - 2)^2 ≤ 4}.finite.to_finset.card = 5 :=
by {
  sorry
}

end num_integers_satisfying_inequality_l815_815426


namespace rockham_soccer_league_members_count_l815_815132

def cost_per_pair_of_socks : Nat := 4
def additional_cost_per_tshirt : Nat := 5
def cost_per_tshirt : Nat := cost_per_pair_of_socks + additional_cost_per_tshirt

def pairs_of_socks_per_member : Nat := 2
def tshirts_per_member : Nat := 2

def total_cost_per_member : Nat :=
  pairs_of_socks_per_member * cost_per_pair_of_socks + tshirts_per_member * cost_per_tshirt

def total_cost_all_members : Nat := 2366
def total_members : Nat := total_cost_all_members / total_cost_per_member

theorem rockham_soccer_league_members_count : total_members = 91 :=
by
  -- Given steps in the solution, verify each condition and calculation.
  sorry

end rockham_soccer_league_members_count_l815_815132


namespace construct_triangle_inradius_incenter_orthocenter_l815_815317

-- Define the given conditions
variables (r AO AH : ℝ)
-- r is the inradius, AO and AH are the given segment lengths.
-- We assume r > 0, AO > 0, AH > 0 (implicitly understood)

theorem construct_triangle_inradius_incenter_orthocenter
  (h_r_pos : r > 0) (h_AO_pos : AO > 0) (h_AH_pos : AH > 0) :
  ∃ (A B C : Point), triangle_with_inradius_and_height r AO AH A B C := 
sorry

end construct_triangle_inradius_incenter_orthocenter_l815_815317


namespace seventh_place_is_unspecified_l815_815478

noncomputable def charlie_position : ℕ := 5
noncomputable def emily_position : ℕ := charlie_position + 5
noncomputable def dana_position : ℕ := 10
noncomputable def bob_position : ℕ := dana_position - 2
noncomputable def alice_position : ℕ := emily_position + 3

theorem seventh_place_is_unspecified :
  ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 15 ∧ x ≠ charlie_position ∧ x ≠ emily_position ∧
  x ≠ dana_position ∧ x ≠ bob_position ∧ x ≠ alice_position →
  x = 7 → false := 
by
  sorry

end seventh_place_is_unspecified_l815_815478


namespace fraction_of_yard_occupied_l815_815283

noncomputable def area_triangle_flower_bed : ℝ := 
  2 * (0.5 * (10:ℝ) * (10:ℝ))

noncomputable def area_circular_flower_bed : ℝ := 
  Real.pi * (2:ℝ)^2

noncomputable def total_area_flower_beds : ℝ := 
  area_triangle_flower_bed + area_circular_flower_bed

noncomputable def area_yard : ℝ := 
  (40:ℝ) * (10:ℝ)

noncomputable def fraction_occupied := 
  total_area_flower_beds / area_yard

theorem fraction_of_yard_occupied : 
  fraction_occupied = 0.2814 := 
sorry

end fraction_of_yard_occupied_l815_815283


namespace number_of_repeated_ones_divisible_by_9_l815_815559

theorem number_of_repeated_ones_divisible_by_9 (n : ℕ) : 
  let number_of_ones := 3^3^n in 
  (number_of_ones * 111) % 9 = 0 :=
sorry

end number_of_repeated_ones_divisible_by_9_l815_815559


namespace difference_in_lengths_l815_815618

def speed_of_first_train := 60 -- in km/hr
def time_to_cross_pole_first_train := 3 -- in seconds
def speed_of_second_train := 90 -- in km/hr
def time_to_cross_pole_second_train := 2 -- in seconds

noncomputable def length_of_first_train : ℝ := (speed_of_first_train * (5 / 18)) * time_to_cross_pole_first_train
noncomputable def length_of_second_train : ℝ := (speed_of_second_train * (5 / 18)) * time_to_cross_pole_second_train

theorem difference_in_lengths : abs (length_of_second_train - length_of_first_train) = 0.01 :=
by
  -- The full proof would be placed here.
  sorry

end difference_in_lengths_l815_815618


namespace solve_exponent_equation_l815_815150

theorem solve_exponent_equation (x : ℚ) : 5^(2 * x) = 125^(1/2) → x = 3/4 :=
by sorry

end solve_exponent_equation_l815_815150


namespace simplify_expression_l815_815772

theorem simplify_expression : ( (3 + 4 + 5 + 6) / 3 ) + ( (3 * 6 + 9) / 4 ) = 12.75 := by
  sorry

end simplify_expression_l815_815772


namespace trajectory_of_midpoint_l815_815167

-- Define the fixed point P
def pointP : (ℝ × ℝ) := (4, -2)

-- Define the equation of the circle
def circle (x y : ℝ) := x^2 + y^2 = 4

-- Define the equation of the trajectory of the midpoint
def trajectory (x y : ℝ) := (x - 2)^2 + (y + 1)^2 = 1

theorem trajectory_of_midpoint (x y : ℝ) (A : ℝ × ℝ) 
  (hA : circle A.1 A.2)
  (hx : x = (A.1 + pointP.1) / 2) 
  (hy : y = (A.2 + pointP.2) / 2) : 
  trajectory x y :=
sorry

end trajectory_of_midpoint_l815_815167


namespace complex_max_value_l815_815897

theorem complex_max_value (z : ℂ) (h : complex.abs z = real.sqrt 3) : 
  real.sqrt (complex.abs ((z - 1) * (z + 1)^2)) ≤ 3 * real.sqrt 3 :=
sorry

end complex_max_value_l815_815897


namespace combined_colonies_male_red_ants_percentage_l815_815086

theorem combined_colonies_male_red_ants_percentage :
  (let
     colony_a_red := 0.60
     colony_a_female_red := 0.35
     colony_b_red := 0.45
     colony_b_female_red := 0.50
     colony_c_red := 0.70
     colony_c_female_red := 0.40
     male_red_percentage_a := colony_a_red * (1 - colony_a_female_red)
     male_red_percentage_b := colony_b_red * (1 - colony_b_female_red)
     male_red_percentage_c := colony_c_red * (1 - colony_c_female_red)
   in
     (male_red_percentage_a + male_red_percentage_b + male_red_percentage_c) / 3 * 100 = 34.5) :=
  sorry

end combined_colonies_male_red_ants_percentage_l815_815086


namespace John_subtracts_79_from_40sq_to_get_39sq_l815_815999

theorem John_subtracts_79_from_40sq_to_get_39sq :
  ∃ (n : ℤ),
    (41^2 = 40^2 + 81) ∧ (39^2 = 40^2 - n) ∧ (n = 79) :=
by {
  use 79,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { refl, }
}

end John_subtracts_79_from_40sq_to_get_39sq_l815_815999


namespace ellipse_eq_from_hyperbola_l815_815410

noncomputable def hyperbola_eq : Prop :=
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = -1) →
  (x^2 / 4 + y^2 / 16 = 1)

theorem ellipse_eq_from_hyperbola :
  hyperbola_eq :=
by
  sorry

end ellipse_eq_from_hyperbola_l815_815410


namespace problem_statement_l815_815382

theorem problem_statement (a b c : ℝ) (h₀ : 4 * a - 4 * b + c > 0) (h₁ : a + 2 * b + c < 0) : b^2 > a * c :=
sorry

end problem_statement_l815_815382


namespace sum_of_binomial_solutions_sum_of_integer_values_l815_815744

open Nat

theorem sum_of_binomial_solutions (n : ℕ) (h1 : binomial 18 n + binomial 18 8 = binomial 19 9) :
  n = 9 ∨ n = 10 :=
by sorry

theorem sum_of_integer_values :
  (∑ n in {9, 10}, n) = 19 :=
by
  rw [Finset.sum_insert (by decide : 9 ≠ 10)]
  rw [Finset.sum_singleton]
  norm_num

end sum_of_binomial_solutions_sum_of_integer_values_l815_815744


namespace minimize_triangle_sum_l815_815290

variables {α : Type*} [ordered_field α] {A B C D : α}
variables (l : α) (AC_inter : α) (BD_inter : α) 

def is_trapezoid (A B C D : α) : Prop := sorry
def sum_of_areas_min (A B C D : α) (l : α) : Prop := sorry

theorem minimize_triangle_sum (A B C D : α) 
  (trapezoid : is_trapezoid A B C D)
  (AC_inter : α) (BD_inter : α) 
  (intersection : AC_inter = BD_inter) :
  sum_of_areas_min A B C D (AC_inter) :=
sorry

end minimize_triangle_sum_l815_815290


namespace at_least_14_grandchildren_l815_815616

variables (ℕ : Type) (G : ℕ → ℕ → Prop)
variables (child : ℕ) (grandfather : ℕ)
variables (grandchildren : grandfather → finset child)
variables (common_grandfather : ∀ {u v : child}, u ≠ v → ∃ g : grandfather, u ∈ grandchildren g ∧ v ∈ grandchildren g)

theorem at_least_14_grandchildren
  (child_count : finset child) (h_child_count : child_count.card = 20)
  (h_common_grandfather : common_grandfather) :
  ∃ g : grandfather, (grandchildren g).card ≥ 14 :=
begin
  sorry
end

end at_least_14_grandchildren_l815_815616


namespace three_monotonic_intervals_l815_815472

open Real

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := - (4 / 3) * x ^ 3 + (b - 1) * x

noncomputable def f' (b : ℝ) (x : ℝ) : ℝ := -4 * x ^ 2 + (b - 1)

theorem three_monotonic_intervals (b : ℝ) (h : (b - 1) > 0) : b > 1 := 
by
  have discriminant : 16 * (b - 1) > 0 := sorry
  sorry

end three_monotonic_intervals_l815_815472


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l815_815662

-- Problem 1: f(x) = 2x f'(e) + ln(x) implies f'(e) = -1/e
theorem problem_1 (f : ℝ → ℝ) (h1 : ∀ x, deriv f x = 2 * (deriv f e) + 1/x) :
  deriv f e = -1 / real.exp 1 := 
sorry

-- Problem 2: |x + 2| <= 5 solution set is [-7, 3]
theorem problem_2 {x : ℝ} : abs (x + 2) ≤ 5 ↔ (-7 ≤ x ∧ x ≤ 3) := 
sorry

-- Problem 3: Minimum value of x^2 + 2/x for x > 0 is 3
theorem problem_3 {x : ℝ} (h : 0 < x) :
  ∃ y, y = x^2 + 2/x ∧ (∀ z, 0 < z → x^2 + 2/x ≤ z^2 + 2/z ↔ z = 1) := 
sorry

-- Problem 4: Tangent line to y = exp(x) at (1, exp(1)) is y = exp(1) * x
theorem problem_4 : tangent_line (λ x : ℝ, real.exp x) 1 = λ x, real.exp 1 * x :=
sorry

-- Problem 5: f(x) = x^2 + 2x + a ln(x) is monotonically decreasing on (0, 1) implies a <= -4
theorem problem_5 (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x ∈ Ioo 0 1, deriv (λ x, x^2 + 2 * x + a * real.log x) x ≤ 0):
  a ≤ -4 :=
sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l815_815662


namespace count_integers_satisfying_inequality_l815_815448

theorem count_integers_satisfying_inequality : ∃ (count : ℕ), count = 5 ∧ 
  ∀ x : ℤ, (0 ≤ x ∧ x ≤ 4) → ((x - 2) * (x - 2) ≤ 4) :=
begin
  existsi 5, 
  split,
  { refl },
  { intros x hx,
    cases hx with h1 h2,
    have : 0 ≤ (x - 2) ^ 2, from pow_two_nonneg (x - 2),
    exact le_antisymm this (show (x - 2) ^ 2 ≤ 4, by sorry) }
end

end count_integers_satisfying_inequality_l815_815448


namespace least_positive_int_to_next_multiple_l815_815230

theorem least_positive_int_to_next_multiple (x : ℕ) (n : ℕ) (h : x = 365 ∧ n > 0) 
  (hm : (x + n) % 5 = 0) : n = 5 :=
by
  sorry

end least_positive_int_to_next_multiple_l815_815230


namespace vitamins_delivery_l815_815134

theorem vitamins_delivery (total_medicine : ℕ) (boxes_supplements : ℕ) (total_medicine = 760 : Prop) (boxes_supplements = 288 : Prop) : 
  (total_medicine - boxes_supplements = 472) :=
by
  -- proof goes here
  sorry

end vitamins_delivery_l815_815134


namespace Hanna_erasers_count_l815_815831

theorem Hanna_erasers_count :
  ∀ (tanya_total r tanya_red tanya_blue tanya_yellow rachel hannah : ℕ),
    tanya_total = 30 →
    tanya_red = tanya_total / 2 →
    tanya_blue = tanya_total / 3 →
    tanya_yellow = 2 * tanya_blue →
    tanya_red + tanya_blue + tanya_yellow = tanya_total →
    r = tanya_red / 3 - 5 →
    hannah = 3 * r →
    hannah = 0 :=
by
  intros tanya_total r tanya_red tanya_blue tanya_yellow rachel hannah
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, add_assoc, h5, h6, h7]
  sorry

end Hanna_erasers_count_l815_815831


namespace prime_number_9_greater_than_perfect_square_l815_815332

theorem prime_number_9_greater_than_perfect_square :
  ∃ p, (Nat.prime p) ∧ (∃ n m, p = n^2 + 9 ∧ p = m^2 - 8) :=
by
  let p := 73
  have p_prime : Nat.prime p := by
    sorry -- Skipping the proof that 73 is a prime number.
  use p
  split
  exact p_prime
  -- Define n and m such that p = n^2 + 9 and p = m^2 - 8
  use 8
  use 9
  split
  have hn : 8^2 = 64 := by norm_num
  rw [hn]
  exact rfl
  have hm : 9^2 = 81 := by norm_num
  rw [hm]
  exact rfl

end prime_number_9_greater_than_perfect_square_l815_815332


namespace total_pieces_of_pizza_l815_815736

def pieces_per_pizza : ℕ := 6
def pizzas_per_fourthgrader : ℕ := 20
def fourthgraders_count : ℕ := 10

theorem total_pieces_of_pizza :
  fourthgraders_count * (pieces_per_pizza * pizzas_per_fourthgrader) = 1200 :=
by
  /-
  We have:
  - pieces_per_pizza = 6
  - pizzas_per_fourthgrader = 20
  - fourthgraders_count = 10

  Therefore,
  10 * (6 * 20) = 1200
  -/
  sorry

end total_pieces_of_pizza_l815_815736


namespace production_rate_l815_815837

theorem production_rate (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (H : x * x * 2 * x = 2 * x^3) :
  y * y * 3 * y = 3 * y^3 := by
  sorry

end production_rate_l815_815837


namespace tangents_equal_segments_l815_815660

/-- GH and ED are perpendicular diameters of a circle. 
From a point B outside the circle, two tangents are drawn to the circle, intersecting GH at A and C. 
The lines BE and BD intersect GH at F and K, respectively. 
Prove that AF = KC. -/
theorem tangents_equal_segments
  (circle : Type) [metric_space circle] [inner_product_space ℝ circle]
  (CenteredCircle : circle)
  (radius : ℝ) (r_pos : 0 < radius)
  (O G H E D A C F K B : circle)
  (perpendicular_diameters : GH_perpendicular_ED : G ≠ H ∧ E ≠ D)
  (A_tangent : point_tangent A circle radius)
  (C_tangent : point_tangent C circle radius)
  (F_intersection : line_intersects BE GH F)
  (K_intersection : line_intersects BD GH K)
  : distance AF = distance KC := 
sorry

end tangents_equal_segments_l815_815660


namespace circumcircle_equation_l815_815032

def triangle_verts : set (ℝ × ℝ) := {(4,0), (0,3), (0,0)}

theorem circumcircle_equation (x y : ℝ) 
  (hx : (4, 0) ∈ triangle_verts) 
  (hy : (0, 3) ∈ triangle_verts) 
  (hz : (0, 0) ∈ triangle_verts) :
  x^2 + y^2 - 4 * x - 3 * y = 0 :=
sorry

end circumcircle_equation_l815_815032


namespace sequence_periodicity_l815_815374

theorem sequence_periodicity (a : ℕ → ℤ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 = 6) 
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n): 
  a 2015 = -6 := 
sorry

end sequence_periodicity_l815_815374


namespace combined_mpg_l815_815925

-- Variables and conditions
variables (miles_ray gallons_ray mpg_ray : ℕ) (miles_tom gallons_tom mpg_tom : ℕ)

-- Given conditions
def ray_car := mpg_ray = 30
def tom_car := mpg_tom = 20
def ray_distance := miles_ray = 150
def tom_distance := miles_tom = 100
def ray_gallons := gallons_ray = miles_ray / mpg_ray
def tom_gallons := gallons_tom = miles_tom / mpg_tom
def total_distance := miles_ray + miles_tom = 250
def total_gallons := gallons_ray + gallons_tom = 10

-- Theorem statement to prove
theorem combined_mpg : 
  ray_car ∧ tom_car ∧ ray_distance ∧ tom_distance ∧ ray_gallons ∧ tom_gallons ∧ total_distance ∧ total_gallons 
  → (250 / 10 = 25) := 
by
  sorry

end combined_mpg_l815_815925


namespace count_correct_propositions_l815_815041

theorem count_correct_propositions :
  let p := true
  let q := true
  (¬p → q) ∧ ¬(q → ¬p) → (p → ¬q) ∧ ¬(¬q → p) → true :=
begin
  -- Condition 1: If ¬p is a necessary but not sufficient condition for q, then
  -- p is a sufficient but not necessary condition for ¬q
  let cond1 := ¬¬p → q ∧ ¬(q → ¬¬p) → (p → ¬q) ∧ ¬(¬q → p),

  -- Condition 2: The negation of "For every x ∈ ℝ, x² ≥ 0" is
  -- "There exists x₀ ∈ ℝ such that x₀² < 0"
  let cond2 := ∀ x, x ∈ ℝ → x^2 ≥ 0 → ∃ x₀, x₀ ∈ ℝ ∧ x₀^2 < 0,

  -- Condition 3: If p ∧ q is false, then both p and q are false
  let cond3 := (p ∧ q = false) → false = true → p = false ∧ q = false,

  -- We need to check if two out of three conditions are logically consistent
  (2 = ([cond1, cond2, cond3].count true) :=
  sorry -- proof omitted
end

end count_correct_propositions_l815_815041


namespace daqing_oilfield_scientific_notation_l815_815993

theorem daqing_oilfield_scientific_notation (total_production : ℕ) (h : total_production = 45000000) :
  total_production = 4.5 * 10^8 := by
sorry

end daqing_oilfield_scientific_notation_l815_815993


namespace peter_hunts_3_times_more_than_mark_l815_815483

theorem peter_hunts_3_times_more_than_mark : 
  ∀ (Sam Rob Mark Peter : ℕ),
  Sam = 6 →
  Rob = Sam / 2 →
  Mark = (Sam + Rob) / 3 →
  Sam + Rob + Mark + Peter = 21 →
  Peter = 3 * Mark :=
by
  intros Sam Rob Mark Peter h1 h2 h3 h4
  sorry

end peter_hunts_3_times_more_than_mark_l815_815483


namespace least_element_of_T_l815_815114

noncomputable def T : Set ℕ := sorry

theorem least_element_of_T {T : Set ℕ}
    (h1 : T ⊆ {n | 1 ≤ n ∧ n ≤ 15})
    (h2 : 7 ≤ T.card)
    (h3 : ∀ ⦃a b⦄, a ∈ T → b ∈ T → a < b → ¬ (b % a = 0))
    (h4 : 2 ≤ (T ∩ {n | Prime n}).card) :
    ∃ t ∈ T, ∀ t' ∈ T, t ≤ t' ∧ t = 4 := sorry

end least_element_of_T_l815_815114


namespace find_x_for_minimum_value_l815_815580

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := cos (2 * x + φ)

theorem find_x_for_minimum_value (φ : ℝ) (hφ : abs φ < π / 2) :
  let shifted_f := λ x : ℝ, cos (2 * (x + π / 6) + φ)
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 2) → shifted_f x = cos (2 * x + π / 6 + φ) →
  (∀ y, (0 ≤ y ∧ y ≤ π / 2) → shifted_f y ≥ shifted_f (5 * π / 12)) → 
  x = 5 * π / 12 :=
sorry

end find_x_for_minimum_value_l815_815580


namespace smallest_integer_solution_l815_815232

theorem smallest_integer_solution (x : ℤ) (h : 10 - 5 * x < -18) : x = 6 :=
sorry

end smallest_integer_solution_l815_815232


namespace number_of_persons_l815_815946

-- Definitions of the given conditions
def average : ℕ := 15
def average_5 : ℕ := 14
def sum_5 : ℕ := 5 * average_5
def average_9 : ℕ := 16
def sum_9 : ℕ := 9 * average_9
def age_15th : ℕ := 41
def total_sum : ℕ := sum_5 + sum_9 + age_15th

-- The main theorem stating the equivalence
theorem number_of_persons (N : ℕ) (h_average : average * N = total_sum) : N = 17 :=
by
  -- Proof goes here
  sorry

end number_of_persons_l815_815946


namespace count_divisible_by_5_l815_815698

theorem count_divisible_by_5 : (finset.filter (λ n : ℕ, n % 5 = 0) (finset.range 1001)).card = 200 := by
  sorry

end count_divisible_by_5_l815_815698


namespace maura_seashells_l815_815541

theorem maura_seashells (original_seashells given_seashells remaining_seashells : ℕ)
  (h1 : original_seashells = 75) 
  (h2 : remaining_seashells = 57) 
  (h3 : given_seashells = original_seashells - remaining_seashells) :
  given_seashells = 18 := by
  -- Lean will use 'sorry' as a placeholder for the actual proof
  sorry

end maura_seashells_l815_815541


namespace gcd_factorial_l815_815228

-- Definitions and conditions
def a : ℕ := 7!
def b : ℕ := 12! / 5!

-- Theorem statement we need to prove
theorem gcd_factorial : Nat.gcd a b = 5040 := by
  sorry

end gcd_factorial_l815_815228


namespace quadratic_no_real_roots_l815_815353

theorem quadratic_no_real_roots (k : ℝ) (h : k ≠ 0) : 
  ∀ (a b c : ℝ), a = 1 ∧ b = k ∧ c = k^2 → (b^2 - 4 * a * c < 0) :=
begin
  intros a b c habc,
  rcases habc with ⟨ha, hb, hc⟩,
  rw [ha, hb, hc],
  simp,
  linarith,
end

end quadratic_no_real_roots_l815_815353


namespace inequality_true_l815_815050

theorem inequality_true (a b : ℝ) (hab : a < b) (hb : b < 0) (ha : a < 0) : (b / a) < 1 :=
by
  sorry

end inequality_true_l815_815050


namespace number_of_cars_in_group_l815_815852

def group_of_cars (T A R : ℕ) : Prop :=
  (T - A = 47) ∧
  (R ≥ 55) ∧
  (A - 45 ≥ 10)

theorem number_of_cars_in_group (T A R : ℕ) (h : group_of_cars T A R) : T = 102 :=
by
  cases h with h1 h2,
  cases h2 with h3 h4,
  sorry

end number_of_cars_in_group_l815_815852


namespace find_k_real_vectors_l815_815336

theorem find_k_real_vectors :
  ∃ (k : ℝ), 
    (k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6) ∧ 
    ∃ (v : Vector ℝ 2), v ≠ 0 ∧
    (Matrix.of ![![3, 4], ![6, 3]] ⬝ v) = k • v := 
sorry

end find_k_real_vectors_l815_815336


namespace heather_distance_when_meet_l815_815562

theorem heather_distance_when_meet :
  let H := 5
  let S := 6
  let initial_distance := 5
  let delay_in_hours := 0.4
  ∃ t : ℝ, (H * t + S * (t + delay_in_hours) = initial_distance) → (H * t = 1.18) :=
by
  -- Definitions of the rates and initial conditions
  let H := 5     -- Heather's speed in miles per hour
  let S := 6     -- Stacy's speed in miles per hour
  let initial_distance := 5   -- Initial distance between Heather and Stacy in miles
  let delay_in_hours := 0.4   -- Time delay for Heather in hours

  -- Existence of time t when they meet such that the total distance walked is the initial distance
  existsi (2.6 / 11 : ℝ)
  intro h
  rw [←h]
  norm_num
  apply rfl

end heather_distance_when_meet_l815_815562


namespace distance_between_intersections_l815_815584

theorem distance_between_intersections :
  let a := 3
  let b := 2
  let c := -7
  let x1 := (-1 + Real.sqrt 22) / 3
  let x2 := (-1 - Real.sqrt 22) / 3
  let distance := abs (x1 - x2)
  let p := 88  -- 2^2 * 22 = 88
  let q := 9   -- 3^2 = 9
  distance = 2 * Real.sqrt 22 / 3 →
  p - q = 79 :=
by
  sorry

end distance_between_intersections_l815_815584


namespace find_diameter_of_hemisphere_l815_815605

theorem find_diameter_of_hemisphere (r a : ℝ) (hr : r = a / 2) (volume : ℝ) (hV : volume = 18 * Real.pi) : 
  2/3 * Real.pi * r ^ 3 = 18 * Real.pi → a = 6 := by
  intro h
  sorry

end find_diameter_of_hemisphere_l815_815605


namespace calendar_diagonal_difference_l815_815263

noncomputable def calendar_matrix : Matrix (Fin 5) (Fin 5) ℕ :=
  ![[1, 2, 3, 4, 5],
    [8, 9, 10, 11, 12],
    [15, 16, 17, 18, 19],
    [22, 23, 24, 25, 26],
    [29, 30, 31, 32, 33]]

noncomputable def modified_matrix (A : Matrix (Fin 5) (Fin 5) ℕ) : Matrix (Fin 5) (Fin 5) ℕ :=
  A.updateRow 1 (A 1).reverse
   .updateRow 4 (A 4).reverse

noncomputable def main_diagonal_sum (A : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  ∑ i, A i i

noncomputable def anti_diagonal_sum (A : Matrix (Fin 5) (Fin 5) ℕ) : ℕ :=
  ∑ i, A i (Fin.reverse i)

theorem calendar_diagonal_difference : 
  ∃ A : Matrix (Fin 5) (Fin 5) ℕ, 
    A = modified_matrix calendar_matrix ∧
    Int.natAbs (main_diagonal_sum A - anti_diagonal_sum A) = 4 := 
  sorry

end calendar_diagonal_difference_l815_815263


namespace largest_a_plus_b_l815_815519

theorem largest_a_plus_b (a b c : ℕ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0)
    (h1 : 1/a + 1/b = 1/c) (h2 : Int.gcdₓ (Int.gcdₓ a b) c = 1) (h3 : a + b ≤ 2011) :
    a + b ≤ 1936 :=
sorry

end largest_a_plus_b_l815_815519


namespace area_ratio_l815_815548

open Real

-- Define the points and conditions
def A : ℝ × ℝ := (0, 12)
def B : ℝ × ℝ := (16, 12)
def C : ℝ × ℝ := (16, 0)
def D : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (4, 3)  -- as determined in the solution
def N : ℝ × ℝ := (16, -24)  -- as determined in the solution

-- Define the function to calculate the area of a triangle using the Shoelace formula
def triangle_area (P1 P2 P3 : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (P1.1 * (P2.2 - P3.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P1.2 - P2.2))

-- Define the function to calculate the area of a parallelogram given the lengths of its sides
def parallelogram_area (side1 side2 : ℝ) : ℝ :=
  side1 * side2

theorem area_ratio :
  let area_triangle_MND := triangle_area M N D in
  let area_parallelogram_ABCD := parallelogram_area 16 12 in
  area_triangle_MND / area_parallelogram_ABCD = 3 / 8 :=
by
  sorry

end area_ratio_l815_815548


namespace last_person_standing_l815_815563

theorem last_person_standing (students : List String) 
  (initial_positions : List (String × Nat))
  (elimination_rule : Nat → Bool)
  (final_person : String) : 
  final_person = "Dan" :=
by
  let students := ["Arn", "Bob", "Cyd", "Dan", "Eve", "Fon"]
  let initial_positions := [("Arn", 1), ("Bob", 2), ("Cyd", 3), ("Dan", 4), ("Eve", 5), ("Fon", 6)]
  let elimination_rule := λ n, n % 7 = 0 ∨ (n % 10 = 7) ∨ (n / 10 = 7)

  -- The main proof logic would go here
  sorry

end last_person_standing_l815_815563


namespace mark_donates_cans_of_soup_l815_815534

theorem mark_donates_cans_of_soup:
  let n_shelters := 6
  let p_per_shelter := 30
  let c_per_person := 10
  let total_people := n_shelters * p_per_shelter
  let total_cans := total_people * c_per_person
  total_cans = 1800 :=
by sorry

end mark_donates_cans_of_soup_l815_815534


namespace set_equality_power_sum_l815_815462

theorem set_equality_power_sum (a b : ℤ) (h : ({a^2, 0, -1} : set ℤ) = {a, b, 0}) : a^2007 + b^2007 = 0 :=
sorry

end set_equality_power_sum_l815_815462


namespace count_integers_satisfying_inequality_l815_815444

theorem count_integers_satisfying_inequality : ∃ (count : ℕ), count = 5 ∧ 
  ∀ x : ℤ, (0 ≤ x ∧ x ≤ 4) → ((x - 2) * (x - 2) ≤ 4) :=
begin
  existsi 5, 
  split,
  { refl },
  { intros x hx,
    cases hx with h1 h2,
    have : 0 ≤ (x - 2) ^ 2, from pow_two_nonneg (x - 2),
    exact le_antisymm this (show (x - 2) ^ 2 ≤ 4, by sorry) }
end

end count_integers_satisfying_inequality_l815_815444


namespace relationship_2x_3sinx_l815_815043

noncomputable def theta : ℝ := 
  {
    θ | θ ∈ Ioo (Real.arccos (2 / 3)) (Real.pi / 2) ∧
    2 * θ = 3 * Real.sin θ
  }

theorem relationship_2x_3sinx (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) :
  (x < theta → 2 * x < 3 * Real.sin x) ∧ 
  (x = theta → 2 * x = 3 * Real.sin x) ∧ 
  (x > theta → 2 * x > 3 * Real.sin x) :=
sorry

end relationship_2x_3sinx_l815_815043


namespace largest_reciprocal_l815_815635

theorem largest_reciprocal :
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  (1 / b) > (1 / a) ∧ (1 / b) > (1 / c) ∧ (1 / b) > (1 / d) ∧ (1 / b) > (1 / e) :=
by
  let a := -1/2
  let b := 1/4
  let c := 0.5
  let d := 3
  let e := 10
  sorry

end largest_reciprocal_l815_815635


namespace shopkeeper_marked_price_l815_815285

theorem shopkeeper_marked_price 
  (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : C = 0.75 * S)
  (h3 : S = 0.85 * M) :
  M = 1.17647 * L :=
sorry

end shopkeeper_marked_price_l815_815285


namespace min_value_and_period_l815_815782

noncomputable def f (x : ℝ) : ℝ := Math.sin x * Math.cos x

theorem min_value_and_period : 
  (∃ x : ℝ, f x = -1/2) ∧ (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ (∀ x : ℝ, f (x + δ) = f x)) :=
by
  -- Proof skipped
  sorry

end min_value_and_period_l815_815782


namespace brick_wall_problem_l815_815709

theorem brick_wall_problem
  (b : ℕ)
  (rate_ben rate_arya : ℕ → ℕ)
  (combined_rate : ℕ → ℕ → ℕ)
  (work_duration : ℕ)
  (effective_combined_rate : ℕ → ℕ × ℕ → ℕ)
  (rate_ben_def : ∀ (b : ℕ), rate_ben b = b / 12)
  (rate_arya_def : ∀ (b : ℕ), rate_arya b = b / 15)
  (combined_rate_def : ∀ (b : ℕ), combined_rate (rate_ben b) (rate_arya b) = rate_ben b + rate_arya b)
  (effective_combined_rate_def : ∀ (b : ℕ), effective_combined_rate b (rate_ben b, rate_arya b) = combined_rate (rate_ben b) (rate_arya b) - 15)
  (work_duration_def : work_duration = 6)
  (completion_condition : ∀ (b : ℕ), work_duration * effective_combined_rate b (rate_ben b, rate_arya b) = b) :
  b = 900 :=
by
  -- Proof would go here
  sorry

end brick_wall_problem_l815_815709


namespace average_books_read_l815_815324

/-- Determining the average number of books read by each member, rounded to the nearest whole number
given the distribution:
- 1 book: 4 members
- 2 books: 3 members
- 3 books: 6 members
- 4 books: 2 members
- 6 books: 3 members
The answer should be 3.
-/
theorem average_books_read :
  let total_books := 4 * 1 + 3 * 2 + 6 * 3 + 2 * 4 + 3 * 6,
      total_members := 4 + 3 + 6 + 2 + 3,
      average := (total_books : ℚ) / (total_members : ℚ)
  in average.round = 3 :=
by
  let total_books := 4 * 1 + 3 * 2 + 6 * 3 + 2 * 4 + 3 * 6
  let total_members := 4 + 3 + 6 + 2 + 3
  let average := (total_books : ℚ) / (total_members : ℚ)
  have h_average : average = 3 := sorry
  exact h_average.round_eq 3

end average_books_read_l815_815324


namespace driver_travel_distance_per_week_l815_815677

noncomputable def daily_distance := 30 * 3 + 25 * 4 + 40 * 2

noncomputable def total_weekly_distance := daily_distance * 6 + 35 * 5

theorem driver_travel_distance_per_week : total_weekly_distance = 1795 := by
  simp [daily_distance, total_weekly_distance]
  done

end driver_travel_distance_per_week_l815_815677


namespace quadratic_inequality_solution_l815_815061

theorem quadratic_inequality_solution (a b : ℝ)
  (h1 : ∀ x, (x > -1 ∧ x < 2) ↔ ax^2 + x + b > 0) :
  a + b = 1 :=
sorry

end quadratic_inequality_solution_l815_815061


namespace john_bought_two_shirts_l815_815106

/-- The number of shirts John bought, given the conditions:
1. The first shirt costs $6 more than the second shirt.
2. The first shirt costs $15.
3. The total cost of the shirts is $24,
is equal to 2. -/
theorem john_bought_two_shirts
  (S : ℝ) 
  (first_shirt_cost : ℝ := 15)
  (second_shirt_cost : ℝ := S)
  (cost_difference : first_shirt_cost = second_shirt_cost + 6)
  (total_cost : first_shirt_cost + second_shirt_cost = 24)
  : 2 = 2 :=
by
  sorry

end john_bought_two_shirts_l815_815106


namespace number_of_integers_satisfying_inequality_l815_815442

theorem number_of_integers_satisfying_inequality : 
  (finset.filter (λ x, (x - 2)^2 ≤ 4) (finset.Icc 0 4)).card = 5 := 
  sorry

end number_of_integers_satisfying_inequality_l815_815442


namespace distance_B1_to_ABC_l815_815085

-- Define the geometrical structure and the distances involved
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Plane3D :=
  (a : ℝ) -- Coefficient of x
  (b : ℝ) -- Coefficient of y
  (c : ℝ) -- Coefficient of z
  (d : ℝ) -- Constant term

-- Given problem conditions
def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨1, 0, 0⟩
def C : Point3D := ⟨1 / 2, sqrt(3) / 2, 0⟩
def B1 : Point3D := ⟨1, 0, 1⟩
def ABC : Plane3D := -- Plane formed by points A, B and C
  ⟨0, 0, 1, 0⟩ -- Since A, B, C are on xy-plane, z is 0 

-- Function to calculate distance from a point to a plane
def distancePointToPlane (pt : Point3D) (pl : Plane3D) : ℝ :=
  (abs (pl.a * pt.x + pl.b * pt.y + pl.c * pt.z + pl.d)) / 
  (sqrt (pl.a ^ 2 + pl.b ^ 2 + pl.c ^ 2))

theorem distance_B1_to_ABC :
  distancePointToPlane B1 ABC = sqrt(21) / 7 :=
sorry

end distance_B1_to_ABC_l815_815085


namespace problem_l815_815514

variables {a b c d : ℝ}

theorem problem (h1 : c + d = 14 * a) (h2 : c * d = 15 * b) (h3 : a + b = 14 * c) (h4 : a * b = 15 * d) (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) :
  a + b + c + d = 3150 := sorry

end problem_l815_815514


namespace problem_statement_l815_815294

noncomputable def f1 (x : ℝ) : ℝ := x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := 1 - 1 / x
noncomputable def f3 (x : ℝ) : ℝ := x^2 - 5 * x - 6
noncomputable def f4 (x : ℝ) : ℝ := 3 - x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃a b : ℝ⦄, a ∈ I → b ∈ I → a < b → f a < f b

theorem problem_statement :
  is_increasing_on f4 (Ioi (0 : ℝ)) ∧
  ¬ is_increasing_on f1 (Ioi (0 : ℝ)) ∧
  ¬ is_increasing_on f2 (Ioi (0 : ℝ)) ∧
  ¬ is_increasing_on f3 (Ioi (0 : ℝ)) := by
  sorry

end problem_statement_l815_815294


namespace f_decreasing_range_f_on_1_2_range_of_a_l815_815396

noncomputable def f (x : ℝ) : ℝ := - (2^x) / (2^x + 1)

theorem f_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := sorry

theorem range_f_on_1_2 : set.range (λ x : ℝ, f x) ∩ set.Icc 1 2 = set.Icc (-4/5 : ℝ) (-2/3 : ℝ) := sorry

noncomputable def g (a x : ℝ) : ℝ := a / 2 + f x

theorem range_of_a : ∀ x ∈ set.Icc 1 2, g a x ≥ 0 ↔ a ≥ 8 / 5 := sorry

end f_decreasing_range_f_on_1_2_range_of_a_l815_815396


namespace solve_for_x_l815_815147

theorem solve_for_x (x : ℝ) (h1 : 5^(2*x) = real.sqrt 125) : x = 3 / 4 :=
sorry

end solve_for_x_l815_815147


namespace count_integer_solutions_l815_815421

theorem count_integer_solutions :
  {x : ℤ | (x - 2)^2 ≤ 4}.card = 5 :=
by
  sorry

end count_integer_solutions_l815_815421


namespace complete_work_in_12_days_l815_815924

def Ravi_rate_per_day : ℚ := 1 / 24
def Prakash_rate_per_day : ℚ := 1 / 40
def Suresh_rate_per_day : ℚ := 1 / 60
def combined_rate_per_day : ℚ := Ravi_rate_per_day + Prakash_rate_per_day + Suresh_rate_per_day

theorem complete_work_in_12_days : 
  (1 / combined_rate_per_day) = 12 := 
by
  sorry

end complete_work_in_12_days_l815_815924


namespace percentage_voters_for_candidate_A_l815_815067

-- Definitions based on conditions
def percentage_democrats (total_voters : ℕ) : ℕ := 60 * total_voters / 100
def percentage_republicans (total_voters : ℕ) : ℕ := 40 * total_voters / 100

def percentage_votes_democrats (total_democrats : ℕ) : ℕ := 75 * total_democrats / 100
def percentage_votes_republicans (total_republicans : ℕ) : ℕ := 30 * total_republicans / 100

-- Theorem statement to be proven
theorem percentage_voters_for_candidate_A (total_voters : ℕ) :
  let total_democrats := percentage_democrats total_voters,
      total_republicans := percentage_republicans total_voters,
      votes_democrats := percentage_votes_democrats total_democrats,
      votes_republicans := percentage_votes_republicans total_republicans,
      total_votes_for_A := votes_democrats + votes_republicans in
  (total_votes_for_A * 100 / total_voters = 57) :=
sorry

end percentage_voters_for_candidate_A_l815_815067


namespace largest_vertex_sum_l815_815351

def parabola_vertex_sum (a T : ℤ) (hT : T ≠ 0) : ℤ :=
  let x_vertex := T
  let y_vertex := a * T^2 - 2 * a * T^2
  x_vertex + y_vertex

theorem largest_vertex_sum (a T : ℤ) (hT : T ≠ 0)
  (hA : 0 = a * 0^2 + 0 * 0 + 0)
  (hB : 0 = a * (2 * T)^2 + (2 * T) * (2 * -T))
  (hC : 36 = a * (2 * T + 1)^2 + (2 * T - 2 * T * (2 * T + 1)))
  : parabola_vertex_sum a T hT ≤ -14 :=
sorry

end largest_vertex_sum_l815_815351


namespace sum_of_interior_numbers_l815_815713

theorem sum_of_interior_numbers :
  ∑ k in {3, 5, 7, 9}, (2^(k-1) - 2) = 332 :=
by {
  sorry
}

end sum_of_interior_numbers_l815_815713


namespace find_a_l815_815058

noncomputable def curve (x : ℝ) : ℝ :=
  x^(-1/2)

def point (a : ℝ) : ℝ × ℝ :=
  (a, curve a)

def tangent_slope (a : ℝ) : ℝ :=
  - (1 / 2) * a^(-3 / 2)

def tangent_line (a : ℝ) (x : ℝ) : ℝ :=
  curve a + tangent_slope a * (x - a)

def intersection_with_y_axis (a : ℝ) : ℝ :=
  tangent_line a 0

def intersection_with_x_axis (a : ℝ) : ℝ :=
  a - (curve a / tangent_slope a)

def area_of_triangle (a : ℝ) : ℝ :=
  1 / 2 * intersection_with_x_axis a * intersection_with_y_axis a

theorem find_a (a : ℝ) (h : area_of_triangle a = 18) : a = 64 :=
  sorry

end find_a_l815_815058


namespace scooter_price_l815_815107

theorem scooter_price (total_cost: ℝ) (h: 0.20 * total_cost = 240): total_cost = 1200 :=
by
  sorry

end scooter_price_l815_815107


namespace negation_of_p_l815_815962

def p (x : ℝ) : Prop := 2^x + 1 > 0

theorem negation_of_p : (∃ (x : ℝ), 2^x + 1 ≤ 0) ↔ ¬ (∀ (x : ℝ), 2^x + 1 > 0) :=
by
  sorry

end negation_of_p_l815_815962


namespace baking_ratio_total_cups_l815_815066

theorem baking_ratio_total_cups (sugar : ℕ) (b f e : ℕ) 
  (h_ratio : (2 : 7 : 5 : 1) : list ℕ = [b, f, sugar, e])
  (h_sugar : sugar = 10) :
  b + f + sugar + e = 30 := sorry

end baking_ratio_total_cups_l815_815066


namespace stream_speed_l815_815968

def boat_speed_still : ℝ := 30
def distance_downstream : ℝ := 80
def distance_upstream : ℝ := 40

theorem stream_speed (v : ℝ) (h : (distance_downstream / (boat_speed_still + v) = distance_upstream / (boat_speed_still - v))) :
  v = 10 :=
sorry

end stream_speed_l815_815968


namespace f_f_one_eighth_l815_815019

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then real.log x / real.log 2 else (5 : ℝ) ^ x

theorem f_f_one_eighth :
  f (f (1 / 8)) = 1 / 125 :=
by {
  sorry
}

end f_f_one_eighth_l815_815019


namespace good_sequences_l815_815074

theorem good_sequences (n : ℕ) (h : 0 < n) : 
  ∑ i in finset.range(n), 
    if (i.even_digs8) 1 then 1 else 0
  = (1/2) * (10^(n+1) / 9 + 8^(n+1) / 7 - 142 / 63) := sorry

end good_sequences_l815_815074


namespace sin_alpha_value_l815_815796

theorem sin_alpha_value
  (α : ℝ)
  (h₀ : 0 < α)
  (h₁ : α < Real.pi)
  (h₂ : Real.sin (α / 2) = Real.sqrt 3 / 3) :
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

end sin_alpha_value_l815_815796


namespace find_F_and_cos_dihedral_angle_l815_815259

noncomputable def quadrilateral_pyramid (P A B C D E F : Type*) :=
  ∃ (s: ℝ),
  s = 2 ∧
  ∃ (angle_ABC: ℝ),
  angle_ABC = 60 ∧
  ∃ (sine_angle_n_P : ℝ),
  (sine_angle_n_P = √6 / 4) ∧
  (P.perpendicular (plane ABCD)) ∧
  (E.midpoint A B) ∧
  (F.on_line PD) ∧
  parallel (line AF) (plane PEC)

theorem find_F_and_cos_dihedral_angle (P A B C D E F : Type*) (cos_phi : ℝ):
  quadrilateral_pyramid P A B C D E F →
  F.midpoint PD →
  cos_phi = 4 * √31 / 31 :=
begin
  sorry,
end

end find_F_and_cos_dihedral_angle_l815_815259


namespace sin_double_x_condition_l815_815247

theorem sin_double_x_condition (x : ℝ) 
  (h : sin x + cos x + tan x + cot x + sec x + csc x = 9) : 
  sin (2 * x) = 40 - 10 * sqrt 39 :=
sorry

end sin_double_x_condition_l815_815247


namespace simplify_expression_l815_815771

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end simplify_expression_l815_815771


namespace find_a_l815_815112

def A : Set ℤ := {-1, 1, 3}
def B (a : ℤ) : Set ℤ := {a + 1, a^2 + 4}
def intersection (a : ℤ) : Set ℤ := A ∩ B a

theorem find_a : ∃ a : ℤ, intersection a = {3} ∧ a = 2 :=
by
  sorry

end find_a_l815_815112


namespace min_students_wearing_both_l815_815850

theorem min_students_wearing_both (n : ℕ) (H1 : n % 3 = 0) (H2 : n % 6 = 0) (H3 : n = 6) :
  ∃ x : ℕ, x = 1 ∧ 
           (∃ b : ℕ, b = n / 3) ∧
           (∃ r : ℕ, r = 5 * n / 6) ∧
           6 = b + r - x :=
by sorry

end min_students_wearing_both_l815_815850


namespace train_stoppage_time_l815_815745

-- Definitions based on conditions
def speed_excl_stoppages := 30 -- speed in kmph
def speed_incl_stoppages := 21 -- speed in kmph

-- The statement to be proved
theorem train_stoppage_time (S1 S2 : ℝ) (hS1 : S1 = speed_excl_stoppages) (hS2 : S2 = speed_incl_stoppages) :
  ∃ t : ℝ, t = 18 / 60 ∧ t * 60 = 18 :=
begin
  use 18 / 60,
  split,
  { refl },
  { norm_num }
end

end train_stoppage_time_l815_815745


namespace rectangle_area_percentage_change_l815_815256

theorem rectangle_area_percentage_change
  (L W : ℝ) : 
  let L_new := 1.10 * L,
      W_new := 0.90 * W,
      A_initial := L * W,
      A_new := L_new * W_new
  in
  ((A_new - A_initial) / A_initial) * 100 = -1 := by
  sorry

end rectangle_area_percentage_change_l815_815256


namespace Rockham_Soccer_League_Members_l815_815129

-- conditions
def sock_cost : ℕ := 4
def tshirt_cost : ℕ := sock_cost + 5
def total_cost_per_member : ℕ := 2 * sock_cost + 2 * tshirt_cost
def total_cost : ℕ := 2366
def number_of_members (n : ℕ) : Prop := n * total_cost_per_member = total_cost

-- statement to prove
theorem Rockham_Soccer_League_Members : ∃ n : ℕ, number_of_members n ∧ n = 91 :=
by {
  use 91,
  show number_of_members 91,
  sorry
}

end Rockham_Soccer_League_Members_l815_815129


namespace median_length_of_isosceles_triangle_l815_815091

-- Define the variables and constants
variables (D E F M : Point)
variables (DE DF EF : ℝ)
variables (is_isosceles : DE = DF) (length_DE : DE = 13) (length_EF : EF = 14)
variables (median_DM : Segment D M) (bisects_EF : M ∈ Segment E F ∧ Segment E M = Segment M F)

-- Define the result we aim to prove
theorem median_length_of_isosceles_triangle
  (h1 : is_isosceles)
  (h2 : length_DE)
  (h3 : length_EF)
  (h4 : bisects_EF) :
  length(median_DM) = 2 * sqrt 30 :=
sorry

end median_length_of_isosceles_triangle_l815_815091


namespace LiamFinishesOnSaturday_l815_815124

theorem LiamFinishesOnSaturday :
  let n := 20
  let first_day_duration := 2
  let common_diff := 1
  let total_days := (n * (2 * first_day_duration + (n - 1) * common_diff)) / 2
  let remainder := total_days % 7
in remainder = 6 → "Saturday" :=
by
  let n := 20
  let first_day_duration := 2
  let common_diff := 1
  let total_days := (n * (2 * first_day_duration + (n - 1) * common_diff)) / 2
  let remainder := total_days % 7
  have : total_days = 230 := by sorry
  have : remainder = 6 := by sorry
  have : "Saturday" = "Saturday" := rfl
  exact this

end LiamFinishesOnSaturday_l815_815124


namespace graph_of_h_is_C_l815_815408

noncomputable def g (x : ℝ) : ℝ :=
  if x >= -2 ∧ x <= 1 then -x
  else if x > 1 ∧ x <= 3 then Real.sqrt (4 - (x - 1)^2)
  else if x > 3 ∧ x <= 5 then x - 3
  else 0  -- Outside the domain given in the problem, we define it as 0 for completeness.

noncomputable def h (x : ℝ) : ℝ := (1 / 3) * g x - 2

theorem graph_of_h_is_C : 
  (∀ x, h x = (1 / 3) * g x - 2) → 
  "C"
:= sorry

end graph_of_h_is_C_l815_815408


namespace simplify_expr_l815_815928

theorem simplify_expr (a : ℝ) : 2 * a * (3 * a ^ 2 - 4 * a + 3) - 3 * a ^ 2 * (2 * a - 4) = 4 * a ^ 2 + 6 * a :=
by
  sorry

end simplify_expr_l815_815928


namespace f_value_plus_deriv_l815_815407

noncomputable def f : ℝ → ℝ := sorry

-- Define the function f and its derivative at x = 1
axiom f_deriv_at_1 : deriv f 1 = 1 / 2

-- Define the value of the function f at x = 1
axiom f_value_at_1 : f 1 = 5 / 2

-- Prove that f(1) + f'(1) = 3
theorem f_value_plus_deriv : f 1 + deriv f 1 = 3 :=
by
  rw [f_value_at_1, f_deriv_at_1]
  norm_num

end f_value_plus_deriv_l815_815407


namespace larger_number_l815_815973

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l815_815973


namespace perpendicular_condition_l815_815176

theorem perpendicular_condition (m : ℝ) : 
  (2 * (m + 1) * (m - 3) + 2 * (m - 3) = 0) ↔ (m = 3 ∨ m = -3) :=
by
  sorry

end perpendicular_condition_l815_815176


namespace some_number_value_l815_815473

theorem some_number_value (a : ℤ) (x1 x2 : ℤ)
  (h1 : x1 + a = 10) (h2 : x2 + a = -10) (h_sum : x1 + x2 = 20) : a = -10 :=
by
  sorry

end some_number_value_l815_815473


namespace range_of_m_l815_815530

variable {x m : ℝ}
def A := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) := {x | x^2 - (2 * m + 1) * x + 2 * m < 0}

theorem range_of_m (h : ∀ x, x ∈ A ∨ x ∈ B m → x ∈ A) :
  -1 / 2 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l815_815530


namespace integer_solutions_to_inequality_l815_815436

theorem integer_solutions_to_inequality :
  ∃ n : ℕ, n = 5 ∧ (∀ x : ℤ, (x - 2) ^ 2 ≤ 4 ↔ x ∈ {0, 1, 2, 3, 4}) :=
by
  sorry

end integer_solutions_to_inequality_l815_815436


namespace simplify_expression_l815_815768

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by 
  sorry

end simplify_expression_l815_815768


namespace interior_points_in_divided_square_l815_815286

theorem interior_points_in_divided_square :
  ∀ (n : ℕ), 
  (n = 2016) →
  ∃ (k : ℕ), 
  (∀ (t : ℕ), t = 180 * n) → 
  k = 1007 :=
by
  intros n hn
  use 1007
  sorry

end interior_points_in_divided_square_l815_815286


namespace find_quadratic_eq_with_given_roots_l815_815595

theorem find_quadratic_eq_with_given_roots (A z x1 x2 : ℝ) 
  (h1 : A * z * x1^2 + x1 * x1 + x2 = 0) 
  (h2 : A * z * x2^2 + x1 * x2 + x2 = 0) : 
  (A * z * x^2 + x1 * x - x2 = 0) :=
by
  sorry

end find_quadratic_eq_with_given_roots_l815_815595


namespace option_b_is_factorization_l815_815240

theorem option_b_is_factorization (m : ℝ) :
  m^2 - 1 = (m + 1) * (m - 1) :=
sorry

end option_b_is_factorization_l815_815240


namespace apples_remaining_l815_815125

variable (initial_apples : ℕ)
variable (picked_day1 : ℕ)
variable (picked_day2 : ℕ)
variable (picked_day3 : ℕ)

-- Given conditions
def condition1 : initial_apples = 200 := sorry
def condition2 : picked_day1 = initial_apples / 5 := sorry
def condition3 : picked_day2 = 2 * picked_day1 := sorry
def condition4 : picked_day3 = picked_day1 + 20 := sorry

-- Prove the total number of apples remaining is 20
theorem apples_remaining (H1 : initial_apples = 200) 
  (H2 : picked_day1 = initial_apples / 5) 
  (H3 : picked_day2 = 2 * picked_day1)
  (H4 : picked_day3 = picked_day1 + 20) : 
  initial_apples - (picked_day1 + picked_day2 + picked_day3) = 20 := 
by
  sorry

end apples_remaining_l815_815125


namespace bella_eats_six_apples_a_day_l815_815308

variable (A : ℕ) -- Number of apples Bella eats per day
variable (G : ℕ) -- Total number of apples Grace picks in 6 weeks
variable (B : ℕ) -- Total number of apples Bella eats in 6 weeks

-- Definitions for the conditions 
def condition1 := B = 42 * A
def condition2 := B = (1 / 3) * G
def condition3 := (2 / 3) * G = 504

-- Final statement that needs to be proved
theorem bella_eats_six_apples_a_day (A G B : ℕ) 
  (h1 : condition1 A B) 
  (h2 : condition2 G B) 
  (h3 : condition3 G) 
  : A = 6 := by sorry

end bella_eats_six_apples_a_day_l815_815308


namespace imaginary_part_z_l815_815003

open Complex

noncomputable def complex_imaginary_part : ℂ := a + I * b

theorem imaginary_part_z (z : ℂ) (h : z = I * (2 - z)) : z.im = 1 :=
  sorry

end imaginary_part_z_l815_815003


namespace triangle_side_length_squared_l815_815701

-- Define the ellipse equation and represent vertices of the triangle.
def ellipse_eq (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

-- Define the vertex A of the triangle.
def vertex_A : ℝ × ℝ := (0, 1)

-- Define the condition for an equilateral triangle inscribed in the ellipse.
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  A = vertex_A ∧
  (B.1 = -C.1) ∧
  (ellipse_eq B.1 B.2) ∧
  (ellipse_eq C.1 C.2) ∧
  (abs (B.2 - C.2) / abs (B.1 - C.1) = Math.sqrt 3)

-- Define the square of the side length of the triangle as required.
def side_length_squared : ℝ :=
  (18 * Math.sqrt 3 / 14)^2

-- The goal is to prove that the length of each side of the triangle squared is as calculated.
theorem triangle_side_length_squared (A B C : ℝ × ℝ) :
  is_equilateral A B C → (∃ d, d^2 = side_length_squared) :=
by
  sorry

end triangle_side_length_squared_l815_815701


namespace total_cost_one_each_l815_815995

theorem total_cost_one_each (x y z : ℝ)
  (h1 : 3 * x + 7 * y + z = 6.3)
  (h2 : 4 * x + 10 * y + z = 8.4) :
  x + y + z = 2.1 :=
  sorry

end total_cost_one_each_l815_815995


namespace inequality_proof_l815_815793

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  (a / (b^2 * (c + 1))) + (b / (c^2 * (a + 1))) + (c / (a^2 * (b + 1))) ≥ 3 / 2 :=
sorry

end inequality_proof_l815_815793


namespace find_angle_A_find_B_and_b_l815_815863

-- Conditions
variables {a b c : ℝ} [triangle_abc : triangle a b c] (h_cond: b^2 - a^2 - c^2 = ac * (cos (A + C) / (sin A * cos A)))
variable (a_val: a = real.sqrt 2)

-- Prove angle A
theorem find_angle_A (h_angle : ∃ (A B C : ℝ), angle_A B C a b c ∧ acute_triangle A B C a b c ∧ \
  (sin 2 * A = 1 ∧ 2 * A = π / 2)) : A = π / 4 :=
sorry

-- Prove angle B and side b when sin B + cos(7π/12 - C) reaches maximum value
theorem find_B_and_b (h_angle_max : ∀ (B C : ℝ), angle_B_max a b c ∧ (0 < B ∧ B < π / 2 ∧ \
  sinB + cos(7 * π / 12 - C) = sinB + cosB * cos (π / 6) ∧ ∃ sinB ∃ cosB, \
  sin B + cos (7 * π / 12 - C) = sqrt 3 := max ∧ \
  B + π / 6 = π / 2) : B = π / 3 ∧ b = sqrt 3 :=
sorry

end find_angle_A_find_B_and_b_l815_815863


namespace line_intersect_yaxis_at_l815_815679

theorem line_intersect_yaxis_at
  (x1 y1 x2 y2 : ℝ) : (x1 = 3) → (y1 = 19) → (x2 = -7) → (y2 = -1) →
  ∃ y : ℝ, (0, y) = (0, 13) :=
by
  intros h1 h2 h3 h4
  sorry

end line_intersect_yaxis_at_l815_815679


namespace riley_mistakes_l815_815858

theorem riley_mistakes :
  ∃ R O : ℕ, R + O = 17 ∧ O = 35 - ((35 - R) / 2 + 5) ∧ R = 3 := by
  sorry

end riley_mistakes_l815_815858


namespace nested_sqrt_eq_two_l815_815838

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by {
    -- Proof skipped
    sorry
}

end nested_sqrt_eq_two_l815_815838


namespace proof_equiv_expr_l815_815720

noncomputable def my_expr : ℝ :=
  real.sqrt 27 - (-2)^0 + |1 - real.sqrt 3| + 2 * real.cos (30 * real.pi / 180)

theorem proof_equiv_expr :
  my_expr = 5 * real.sqrt 3 - 2 :=
sorry

end proof_equiv_expr_l815_815720


namespace injective_function_n_equal_l815_815748

theorem injective_function_n_equal (f : ℕ → ℕ) (h_inj : function.injective f) :
  (∀ n : ℕ, f(f(n)) ≤ (f(n) + n) / 2) → (∀ n : ℕ, f(n) = n) :=
  by
    intros h_cond n
    have h : f(f(n)) ≤ (f(n) + n) / 2 := h_cond n
    sorry

end injective_function_n_equal_l815_815748


namespace numbers_less_than_reciprocal_l815_815638

theorem numbers_less_than_reciprocal :
  (1 / 3 < 3) ∧ (1 / 2 < 2) ∧ ¬(1 < 1) ∧ ¬(2 < 1 / 2) ∧ ¬(3 < 1 / 3) :=
by
  sorry

end numbers_less_than_reciprocal_l815_815638


namespace cone_equation_l815_815603

noncomputable theory
open Real

theorem cone_equation (x y z ϕ : ℝ) : x^2 + y^2 = z^2 * (tan ϕ)^2 → x^2 + y^2 - z^2 * (tan ϕ)^2 = 0 :=
  by
    intro h
    sorry

end cone_equation_l815_815603


namespace pentagon_area_1170_l815_815712

noncomputable def area_of_pentagon (a b c d e : ℕ) (area_rectangle area_triangle : ℕ) : ℕ :=
a + b + c + d + e + area_rectangle + area_triangle

theorem pentagon_area_1170 :
  ∀ (a b c d e : ℕ), a = 25 → b = 30 → c = 28 → d = 25 → e = 30 →
  let area_rectangle := 25 * 30 in
  let area_triangle := (1 / 2) * 30 * 28 in
  let total_area := area_rectangle + area_triangle in
  total_area = 1170 :=
by {
  intros a b c d e h1 h2 h3 h4 h5,
  let area_rectangle := 25 * 30,
  let area_triangle := (1 / 2) * 30 * 28,
  sorry
}

end pentagon_area_1170_l815_815712


namespace division_correct_l815_815711

theorem division_correct :
  250 / (15 + 13 * 3^2) = 125 / 66 :=
by
  -- The proof steps can be filled in here.
  sorry

end division_correct_l815_815711


namespace inequality_problem_l815_815051

theorem inequality_problem
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  (b^2 / a + a^2 / b) ≥ (a + b) :=
sorry

end inequality_problem_l815_815051


namespace solve_custom_problem_l815_815728

def custom_op (a b c : ℕ) : ℚ := (a + b) / c

theorem solve_custom_problem :
  (custom_op (custom_op 30 45 75) (custom_op 4 2 6) (custom_op 12 18 30) 1) = 2 := by
  sorry

end solve_custom_problem_l815_815728


namespace James_total_area_l815_815096

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase_dimension : ℕ := 2
def number_of_new_rooms : ℕ := 4
def larger_room_multiplier : ℕ := 2

noncomputable def new_length : ℕ := initial_length + increase_dimension
noncomputable def new_width : ℕ := initial_width + increase_dimension
noncomputable def area_of_one_new_room : ℕ := new_length * new_width
noncomputable def total_area_of_4_rooms : ℕ := area_of_one_new_room * number_of_new_rooms
noncomputable def area_of_larger_room : ℕ := area_of_one_new_room * larger_room_multiplier
noncomputable def total_area : ℕ := total_area_of_4_rooms + area_of_larger_room

theorem James_total_area : total_area = 1800 := 
by
  sorry

end James_total_area_l815_815096


namespace orthocenter_of_distinct_triangle_l815_815550

variable {V : Type*} [InnerProductSpace ℝ V]
variables (A B C O : V)

theorem orthocenter_of_distinct_triangle (h1 : ⟪O - A, O - B⟫ = 0)
                                          (h2 : ⟪O - B, O - C⟫ = 0)
                                          (h3 : ⟪O - C, O - A⟫ = 0) : 
  is_orthocenter O A B C := 
sorry

end orthocenter_of_distinct_triangle_l815_815550


namespace relation_between_abc_l815_815358

theorem relation_between_abc (a b c : ℕ) (h₁ : a = 3 ^ 44) (h₂ : b = 4 ^ 33) (h₃ : c = 5 ^ 22) : a > b ∧ b > c :=
by
  -- Proof goes here
  sorry

end relation_between_abc_l815_815358


namespace num_integers_satisfying_inequality_l815_815431

theorem num_integers_satisfying_inequality : 
  {x : ℤ | (x - 2)^2 ≤ 4}.finite.to_finset.card = 5 :=
by {
  sorry
}

end num_integers_satisfying_inequality_l815_815431


namespace north_is_positive_four_l815_815844

-- Define the conditions
def south_movement (meters : ℤ) : Prop := meters < 0

-- Define the question function
def north_movement (meters : ℤ) : ℤ :=
  if south_movement (-meters) then meters else 0

-- The main theorem to be proven
theorem north_is_positive_four : north_movement 4 = 4 :=
by sorry

end north_is_positive_four_l815_815844


namespace polynomial_degree_l815_815926

def polynomial : Expr := -2 * x^2 * y + 3 * x * y - 1

theorem polynomial_degree : polynomial.degree = 3 :=
sorry

end polynomial_degree_l815_815926


namespace find_N_l815_815322

def star (a : ℝ) (h : a ≠ -1) : ℝ := (a - 1) / (a + 1)

def tan_15_deg : ℝ := 2 - Real.sqrt 3

theorem find_N (N : ℝ) (hN : N ≠ -1) (hN1 : (star (star N hN) h (by simp)) = tan_15_deg) : 
    N = -2 - Real.sqrt 3 :=
sorry

end find_N_l815_815322


namespace larger_number_is_23_l815_815992

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815992


namespace solve_for_x_l815_815521

theorem solve_for_x (h : ℝ → ℝ) (H : ∀ x : ℝ, h(3 * x - 2) = 5 * x + 3) :
  (∃ x : ℝ, h(x) = 2 * x) ↔ x = 19 :=
by
  sorry

end solve_for_x_l815_815521


namespace triangle_area_less_than_thirteen_l815_815195

theorem triangle_area_less_than_thirteen
  (points : Fin 13 → (ℝ × ℝ))
  (h_on_circle : ∀ i, (points i).fst^2 + (points i).snd^2 = 13^2)
  : ∃ (i j k : Fin 13), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ 
    let p1 := points i in
    let p2 := points j in
    let p3 := points k in
    let area := 1/2 * abs (p1.fst * (p2.snd - p3.snd) + p2.fst * (p3.snd - p1.snd) + p3.fst * (p1.snd - p2.snd)) in
    area < 13 := 
by 
  sorry

end triangle_area_less_than_thirteen_l815_815195


namespace hyperbola_asymptote_slope_l815_815170

theorem hyperbola_asymptote_slope :
  ∃ m : ℝ, (∀ x y : ℝ, (y^2 / 16 - x^2 / 9 = 1) → (y = m * x ∨ y = -m * x)) ∧ m = 4 / 3 :=
begin
  use 4 / 3,
  split,
  { intros x y hyp,
    sorry },
  { refl }
end

end hyperbola_asymptote_slope_l815_815170


namespace arithmetic_sequence_S12_l815_815492

theorem arithmetic_sequence_S12 (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (hS4 : S 4 = 25) (hS8 : S 8 = 100) : S 12 = 225 :=
by
  sorry

end arithmetic_sequence_S12_l815_815492


namespace count_integers_satisfying_inequality_l815_815445

theorem count_integers_satisfying_inequality : ∃ (count : ℕ), count = 5 ∧ 
  ∀ x : ℤ, (0 ≤ x ∧ x ≤ 4) → ((x - 2) * (x - 2) ≤ 4) :=
begin
  existsi 5, 
  split,
  { refl },
  { intros x hx,
    cases hx with h1 h2,
    have : 0 ≤ (x - 2) ^ 2, from pow_two_nonneg (x - 2),
    exact le_antisymm this (show (x - 2) ^ 2 ≤ 4, by sorry) }
end

end count_integers_satisfying_inequality_l815_815445


namespace cos_neg_cos_sub360_cos_60_cos_neg300_l815_815310

-- Defining the cosine function equivalences and known values based on given conditions
theorem cos_neg {θ : ℝ} : Real.cos (-θ) = Real.cos θ :=
by sorry

theorem cos_sub360 {θ : ℝ} : Real.cos (360 - θ) = Real.cos θ :=
by sorry

theorem cos_60 : Real.cos 60 = 1 / 2 :=
by sorry

-- Proving the main statement
theorem cos_neg300 : Real.cos (-300) = 1 / 2 :=
by
  have h1 : Real.cos (-300) = Real.cos 300 := cos_neg 300
  have h2 : 300 = 360 - 60 := by linarith
  have h3 : Real.cos 300 = Real.cos 60 := cos_sub360 60
  show Real.cos (-300) = 1 / 2, from
    by rw [h1, h3, cos_60]

end cos_neg_cos_sub360_cos_60_cos_neg300_l815_815310


namespace sum_of_undefined_fractions_l815_815627

theorem sum_of_undefined_fractions (x₁ x₂ : ℝ) (h₁ : x₁^2 - 7*x₁ + 12 = 0) (h₂ : x₂^2 - 7*x₂ + 12 = 0) :
  x₁ + x₂ = 7 :=
sorry

end sum_of_undefined_fractions_l815_815627


namespace at_least_one_not_less_than_two_l815_815518

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x + 1/y) ≥ 2 ∨ (y + 1/z) ≥ 2 ∨ (z + 1/x) ≥ 2 :=
by
  sorry

end at_least_one_not_less_than_two_l815_815518


namespace cevas_theorem_intersection_or_parallel_l815_815888

-- Define the points on the sides of the triangle
variable (A B C A1 B1 C1 : Type)

-- Assume the points A1, B1, and C1 lie on sides BC, CA, and AB respectively
axiom A1_on_BC : ∃ t : Type, A1 = B + t * (C - B)
axiom B1_on_CA : ∃ t : Type, B1 = C + t * (A - C)
axiom C1_on_AB : ∃ t : Type, C1 = A + t * (B - A)

-- The main theorem to prove the intersection condition
theorem cevas_theorem_intersection_or_parallel
  (A B C A1 B1 C1 : Type) 
  (A1_on_BC : ∃ t : Type, A1 = B + t * (C - B))
  (B1_on_CA : ∃ t : Type, B1 = C + t * (A - C))
  (C1_on_AB : ∃ t : Type, C1 = A + t * (B - A)) :
  (∃ K : Type, K = (A + A1) ∧ K = (B + B1) ∧ K = (C + C1)) ∨
  (∀ K : Type, (A + A1) ≠ K ∨ (B + B1) ≠ K ∨ (C + C1) ≠ K) ↔ 
  (∃ r1 r2 r3 : ℝ, 
    (r1 = (C - C1) / (C1 - B) ∧ r2 = (A - A1) / (A1 - C) ∧ r3 = (B - B1) / (B1 - A)) ∧
     r1 * r2 * r3 = 1)
:= sorry

end cevas_theorem_intersection_or_parallel_l815_815888


namespace winner_votes_calculation_l815_815655

-- Definitions based on conditions
def total_votes := 1000
def winner_votes := 0.75 * total_votes
def loser_votes := 0.25 * total_votes
def vote_difference := 500

-- Lean statement of the problem
theorem winner_votes_calculation  :
  (winner_votes - loser_votes = vote_difference) →
  winner_votes = 750 :=
by
  sorry

end winner_votes_calculation_l815_815655


namespace janelle_bought_6_bags_of_blue_marbles_l815_815879

def initial_green_marbles := 26
def marbles_per_bag := 10
def gift_green_marbles := 6
def gift_blue_marbles := 8
def marbles_left := 72

theorem janelle_bought_6_bags_of_blue_marbles :
  ∃ (bags_of_blue_marbles : ℕ), bags_of_blue_marbles = 6 ∧
    let total_given_away := gift_green_marbles + gift_blue_marbles,
        total_before_giving := marbles_left + total_given_away,
        initial_total_blue := total_before_giving - initial_green_marbles,
        total_bags := initial_total_blue / marbles_per_bag
    in total_bags = bags_of_blue_marbles :=
by
  sorry

end janelle_bought_6_bags_of_blue_marbles_l815_815879


namespace mark_donates_1800_cans_l815_815537

variable (number_of_shelters people_per_shelter cans_per_person : ℕ)
variable (total_people total_cans_of_soup : ℕ)

-- Given conditions
def number_of_shelters := 6
def people_per_shelter := 30
def cans_per_person := 10

-- Calculations based on conditions
def total_people := number_of_shelters * people_per_shelter
def total_cans_of_soup := total_people * cans_per_person

-- Proof statement
theorem mark_donates_1800_cans : total_cans_of_soup = 1800 := by
  -- stretch sorry proof placeholder for the proof
  sorry

end mark_donates_1800_cans_l815_815537


namespace cameron_gold_tokens_l815_815314

/-- Cameron starts with 90 red tokens and 60 blue tokens. 
  Booth 1 exchange: 3 red tokens for 1 gold token and 2 blue tokens.
  Booth 2 exchange: 2 blue tokens for 1 gold token and 1 red token.
  Cameron stops when fewer than 3 red tokens or 2 blue tokens remain.
  Prove that the number of gold tokens Cameron ends up with is 148.
-/
theorem cameron_gold_tokens :
  ∃ (x y : ℕ), 
    90 - 3 * x + y < 3 ∧
    60 + 2 * x - 2 * y < 2 ∧
    (x + y = 148) :=
  sorry

end cameron_gold_tokens_l815_815314


namespace find_acute_angle_l815_815005

theorem find_acute_angle (α : ℝ) (h : ∃! x : ℝ, 3 * x^2 + 1 = 4 * sin α * x) : α = 60 :=
sorry

end find_acute_angle_l815_815005


namespace baron_munchausen_theorem_l815_815307

theorem baron_munchausen_theorem (n : ℕ) (roots : Fin n → ℕ)
  (a : ℕ) (b : ℕ)
  (h_poly : ∀ x, (∑ i, x ^ i) = x^n - a * x^(n-1) + b * x^(n-2) + ... ) :
  (∑ i, roots i) = a ∧ (∑ (i j : Fin n), i ≠ j → (roots i * roots j)) = b :=
sorry

end baron_munchausen_theorem_l815_815307


namespace max_ab_min_2_pow_a_plus_4_pow_b_b_in_interval_min_inv_a_plus_inv_b_l815_815356

variable (a b : ℝ)
variable (H1 : a > 0)
variable (H2 : b > 0)
variable (H3 : a + 2 * b = 1)

-- Statement for the maximum value of ab
theorem max_ab : a > 0 → b > 0 → a + 2 * b = 1 → (∀ ab, ab ≤ 1/8) := sorry

-- Statement for the minimum value of 2^a + 4^b
theorem min_2_pow_a_plus_4_pow_b : a > 0 → b > 0 → a + 2 * b = 1 → (∀ val, val ≥ 2 * sqrt 2) := sorry

-- Statement for b in (0, 1/2)
theorem b_in_interval : a > 0 → b > 0 → a + 2 * b = 1 → (0 < b ∧ b < 1/2) := sorry

-- Statement for the minimum value of 1/a + 1/b
theorem min_inv_a_plus_inv_b : a > 0 → b > 0 → a + 2 * b = 1 → ¬(∀ val, val = 4 * sqrt 2) := sorry

end max_ab_min_2_pow_a_plus_4_pow_b_b_in_interval_min_inv_a_plus_inv_b_l815_815356


namespace PM_eq_PN_l815_815872

variable (Points : Type) [EuclideanGeometry Points]
open EuclideanGeometry

def trapezoid (A B C D : Points) : Prop :=
  ∃ L M: Line Points, L ∥ M ∧ A ∈ L ∧ B ∈ L ∧ C ∈ M ∧ D ∈ M

theorem PM_eq_PN (A B C D P M N : Points)
  (h_trapezoid : trapezoid A B C D)
  (h_intersect : LineThrough P M ∥ LineThrough C D ∧ LineThrough P N ∥ LineThrough C D)
  (h_AC : intersects (LineThrough A C) (LineThrough B D) P)
  (h_MC : intersects (LineThrough A D) (LineThrough B C) M)
  (h_NB : intersects (LineThrough A D) (LineThrough B C) N) :
  dist P M = dist P N :=
sorry

end PM_eq_PN_l815_815872


namespace distance_between_A_and_B_is_40_l815_815683

def is_distance_between_A_and_B (s v1 v2 : ℝ) : Prop :=
  s / (2 * v1) = 15 / v2 ∧ 24 / v1 = s / (2 * v2) ∧
  (s / v1 = (s - 24) / v2 ∨ s / v2 = (s - 15) / v1)

theorem distance_between_A_and_B_is_40 (v1 v2 : ℝ) : ∃ s : ℝ, is_distance_between_A_and_B s v1 v2 ∧ s = 40 := 
begin
  sorry
end

end distance_between_A_and_B_is_40_l815_815683


namespace shaded_fraction_l815_815223

theorem shaded_fraction (side_length : ℝ) (base : ℝ) (height : ℝ) (H1: side_length = 4) (H2: base = 3) (H3: height = 2):
  ((side_length ^ 2) - 2 * (1 / 2 * base * height)) / (side_length ^ 2) = 5 / 8 := by
  sorry

end shaded_fraction_l815_815223


namespace larger_number_is_23_l815_815982

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l815_815982


namespace binomial_positive_integer_power_terms_l815_815965

noncomputable def binomialExpansionTerm (n k : ℕ) (x : ℝ) : ℝ :=
  (choose n k) * (x ^ (n - k / 2))

def containsPositiveIntegerPowerTerms (n : ℕ) (expr : ℝ → ℝ) : ℕ :=
  (Finset.range n).count (λ k, (n - k / 2) > 0 ∧ (n - k / 2) % 1 = 0)

theorem binomial_positive_integer_power_terms : 
  containsPositiveIntegerPowerTerms 10 (λ x => (sqrt x - 1 / (3 * x))) = 2 :=
sorry

end binomial_positive_integer_power_terms_l815_815965


namespace radius_of_the_sphere_equivalent_l815_815870

-- Definitions:
variables {A B C D : Type} [LinearOrderedField A] {a : A}

-- Conditions:
def AB_perp_CD (P : A) : Prop := 
  ∀ x y z w: A, x * y = 0

def AC_perp_BD (P : A) : Prop :=
  ∀ x y z w: A, x * y = 0

def AC_eq_BD (P : A) : Prop := 
  ∀ x y : A, x = y

def BC_eq_a (b : A) : Prop := 
  ∀ x : A, x = b

-- Prove:
theorem radius_of_the_sphere_equivalent (P : A) (SV : A) (M : A):
  AB_perp_CD P →
  AC_perp_BD P →
  AC_eq_BD P →
  BC_eq_a a →
  ∃ r : A, r = (a * Real.sqrt 2 / 4) :=
sorry

end radius_of_the_sphere_equivalent_l815_815870


namespace complement_A_A_subset_B_A_intersection_B_empty_l815_815532

open Set

noncomputable def A : Set ℝ := { x : ℝ | -x^2 - 3x > 0 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | x < m }
noncomputable def U : Set ℝ := univ
noncomputable def complement_U_A : Set ℝ := { x : ℝ | x ≤ -3 ∨ x ≥ 0 }

theorem complement_A :
  (U \ A) = complement_U_A :=
sorry

theorem A_subset_B (m : ℝ) :
  A ⊆ B m → m ≥ 0 :=
sorry

theorem A_intersection_B_empty (m : ℝ) :
  A ∩ B m = ∅ → m ≤ -3 :=
sorry

end complement_A_A_subset_B_A_intersection_B_empty_l815_815532


namespace typing_time_less_than_20_minutes_l815_815303

theorem typing_time_less_than_20_minutes
  (barbara_max_speed : ℕ)
  (reduction_due_to_syndrome : ℕ)
  (fatigue_reduction_rate : ℕ)
  (fatigue_time_interval : ℕ)
  (jim_speed : ℕ)
  (jim_availability : ℕ)
  (document_length : ℕ)
  (barbara_reduced_speed : barbara_max_speed - reduction_due_to_syndrome = 172)
  (combined_speed : barbara_reduced_speed + jim_speed = 272)
  (words_typed_in_20_minutes : combined_speed * jim_availability = 5440)
  (document_length_less_than_5440 : document_length < 5440)
  (document_length_equals : document_length = 3440) :
  ∃ t : ℕ, t < jim_availability ∧ combined_speed * t = document_length :=
by {
  sorry
}

end typing_time_less_than_20_minutes_l815_815303


namespace sara_survey_total_people_l815_815558

theorem sara_survey_total_people (y : ℕ) (h1 : 0.753 * y ≈ 61) (h2 : 0.602 * 61 ≈ 37) : y = 81 :=
by 
  sorry

end sara_survey_total_people_l815_815558


namespace tangent_line_at_point_l815_815166

noncomputable def f : ℝ → ℝ := λ x, 4 * x + x^2

def point_on_curve (x y : ℝ) : Prop := y = f x

theorem tangent_line_at_point :
  ∀ (x0 y0 : ℝ), point_on_curve x0 y0 →
  ∃ (m b : ℝ), (y0 = m * x0 + b) ∧ ∀ x, f x - y0 = m * (x - x0) + (f x - f x0) :=
begin
  intros x0 y0 h,
  sorry
end

end tangent_line_at_point_l815_815166


namespace rect_coords_of_new_polar_l815_815670

-- Define the given conditions
def r := 10
def θ := Real.arccos (8 / r)
def point := (8, 6)
def polar_coords := (r, θ)

-- Triple angle formulas
def cos_3θ := 4 * (Real.cos θ)^3 - 3 * (Real.cos θ)
def sin_3θ := 3 * (Real.sin θ) - 4 * (Real.sin θ)^3

-- Polar to rectangular coordinates transformation for (r^3, 3θ)
def r_cubed := r ^ 3
def new_point := (r_cubed * cos_3θ, r_cubed * sin_3θ)

theorem rect_coords_of_new_polar :
  new_point = (480 : ℝ, 9360 : ℝ) :=
by
  -- Transformation from the problem, assume necessary correctness of operations
  have hr : r * Real.cos θ = 8 := by sorry
  have hs : r * Real.sin θ = 6 := by sorry

  have hr_def : r = 10 := by sorry
  have hθ_def : Real.cos θ = 0.8 := by sorry 
  have hs_def : Real.sin θ = 0.6 := by sorry 

  -- Using triple angle formulas
  have hcos3θ : cos_3θ = 0.048 := by sorry
  have hsin3θ : sin_3θ = 0.936 := by sorry

  -- Using transformations
  have hx : r_cubed * cos_3θ = 480 := by sorry
  have hy : r_cubed * sin_3θ = 9360 := by sorry

  exact ⟨hx, hy⟩

end rect_coords_of_new_polar_l815_815670


namespace proof_A_div_B_l815_815321

noncomputable def seriesA : ℝ :=
  (∑' n in {1, 5, 7, 11, 13, 17, ...}.filter (λ n, ¬(3 ∣ n ∧ odd n)), 1 / (n ^ 2))

noncomputable def seriesB : ℝ :=
  (∑' n in (finset.range ∞).filter (λ n, 3 ∣ n ∧ odd n), ((-1) ^ (n / 3)) * (1 / (n ^ 2)))

theorem proof_A_div_B : seriesA / seriesB = 10 :=
  sorry

end proof_A_div_B_l815_815321


namespace negation_proposition_l815_815180

theorem negation_proposition (x : ℝ) :
  ¬(∀ x : ℝ, x^2 - x + 3 > 0) ↔ ∃ x : ℝ, x^2 - x + 3 ≤ 0 := 
by { sorry }

end negation_proposition_l815_815180


namespace interior_triangle_area_l815_815013

theorem interior_triangle_area (s1 s2 s3 : ℝ) (hs1 : s1 = 15) (hs2 : s2 = 6) (hs3 : s3 = 15) 
  (a1 a2 a3 : ℝ) (ha1 : a1 = 225) (ha2 : a2 = 36) (ha3 : a3 = 225) 
  (h1 : s1 * s1 = a1) (h2 : s2 * s2 = a2) (h3 : s3 * s3 = a3) :
  (1/2) * s1 * s2 = 45 :=
by
  sorry

end interior_triangle_area_l815_815013


namespace yz_orthogonal_ho_l815_815108

-- Define the geometric setup
variables {A B C H O P Q R Y Z : Type} 
variables [Point A] [Point B] [Point C] [Point H] [Point O]
variables [Point P] [Point Q] [Point R] [Point Y] [Point Z]
variables [Line BC] [Line CA] [Line AB] [Line HO] [Line YZ]

-- Conditions
axiom orthocenter (H : Point) (A B C : Triangle) : Line HO
axiom circumcenter (O : Point) (A B C : Triangle) : Line HO
axiom perpendicular_from_A (P : Point) : AP ⊥ BC
axiom reflection_CA (Q : Point) : reflection P CA Q
axiom reflection_AB (R : Point) : reflection P AB R
axiom projection_R_CA (Y : Point) : projection R CA Y
axiom projection_Q_AB (Z : Point) : projection Q AB Z
axiom H_not_O : H ≠ O
axiom Y_not_Z : Y ≠ Z

-- The main theorem statement
theorem yz_orthogonal_ho : YZ ⊥ HO :=
sorry

end yz_orthogonal_ho_l815_815108


namespace total_boys_and_girls_sum_to_41_l815_815145

theorem total_boys_and_girls_sum_to_41 (Rs : ℕ) (amount_per_boy : ℕ) (amount_per_girl : ℕ) (total_amount : ℕ) (num_boys : ℕ) :
  Rs = 1 ∧ amount_per_boy = 12 * Rs ∧ amount_per_girl = 8 * Rs ∧ total_amount = 460 * Rs ∧ num_boys = 33 →
  ∃ num_girls : ℕ, num_boys + num_girls = 41 :=
by
  sorry

end total_boys_and_girls_sum_to_41_l815_815145


namespace word_is_komputer_l815_815144

-- Define the unique digits mapping with inequalities
def letters := {К : ℕ, О : ℕ, М : ℕ, П : ℕ, Ь : ℕ, Ю : ℕ, Т : ℕ, Е : ℕ, Р : ℕ}

def conditions (d : letters) : Prop :=
  d.К < d.О ∧
  d.К < d.М ∧
  d.К < d.П ∧
  d.К < d.Ь ∧
  d.К < d.Ю ∧
  d.К < d.Т ∧
  d.К < d.Е ∧
  d.К < d.Р ∧
  d.О < d.М ∧
  d.О < d.П ∧
  d.О < d.Ь ∧
  d.О < d.Ю ∧
  d.О < d.Т ∧
  d.О < d.Е ∧
  d.О < d.Р ∧
  d.М < d.П ∧
  d.М < d.Ь ∧
  d.М < d.Ю ∧
  d.М < d.Т ∧
  d.М < d.Е ∧
  d.М < d.Р ∧
  d.П < d.Ь ∧
  d.П < d.Ю ∧
  d.П < d.Т ∧
  d.П < d.Е ∧
  d.П < d.Р ∧
  d.Ь < d.Ю ∧
  d.Ь < d.Т ∧
  d.Ь < d.Е ∧
  d.Ь < d.Р ∧
  d.Ю < d.Т ∧
  d.Ю < d.Е ∧
  d.Ю < d.Р ∧
  d.Т < d.Е ∧
  d.Т < d.Р ∧
  d.Е < d.Р ∧
  -- Ensure we have unique digits from 1 to 9
  {d.К, d.О, d.М, d.П, d.Ь, d.Ю, d.Т, d.Е, d.Р} = {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Main theorem stating that given these conditions, the word is "КОМПЬЮТЕР"
theorem word_is_komputer (d : letters) (h : conditions d) : 
  (d.К, d.О, d.М, d.П, d.Ь, d.Ю, d.Т, d.Е, d.Р) = (1, 2, 3, 4, 5, 6, 7, 8, 9) := sorry

end word_is_komputer_l815_815144


namespace distinct_keychain_arrangements_l815_815864

theorem distinct_keychain_arrangements : 
  let keys := ["house", "car", "bike", "mailbox", "key1", "key2"] in
  let pairs := [["house", "car"], ["bike", "mailbox"]]
  in
  num_distinct_arrangements keys pairs = 24 :=
sorry

end distinct_keychain_arrangements_l815_815864


namespace major_axis_length_l815_815278

theorem major_axis_length (r : ℝ) (h_r : r = 2) (minor_axis : ℝ) (h_minor_axis : minor_axis = 2 * r) (major_axis : ℝ) (h_major_axis : major_axis = minor_axis + 0.2 * minor_axis) :
  major_axis = 4.8 :=
by { 
  have h1 : minor_axis = 4 := by 
    rw [h_minor_axis, h_r]; norm_num, 
  have h2 : major_axis = 4 + 0.8 := by 
    rw [h_major_axis, h1]; norm_num, 
  exact h2 
}

end major_axis_length_l815_815278


namespace certain_event_l815_815632

-- Define the problem conditions as hypotheses
def condition_a := ∀ (△ : Triangle), ∀ (O : Point), IsCircumcenter(△, O) → ∀ (d1 d2 d3 : ℝ), Distance(O, Side1(△)) = d1 ∧ Distance(O, Side2(△)) = d2 ∧ Distance(O, Side3(△)) = d3
def condition_b := ∀ (Shooter : Type) (shot : Event), Probability(hit(Shooter, shot)) > 0
def condition_c := ∀ (△ : Triangle), InteriorAnglesSum(△) = 180
def condition_d := ∀ (coin : Coin), ∃ (toss : Event), Probability(heads(toss)) = 0.5 ∧ Probability(tails(toss)) = 0.5

-- Mathematical statement to prove
theorem certain_event : ∀ (△ : Triangle), InteriorAnglesSum(△) = 180 :=
by
  -- the statement uses condition_c directly
  exact condition_c
  sorry

end certain_event_l815_815632


namespace circle_through_points_intersection_l815_815753

noncomputable def circleEquation (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 9 / 2

theorem circle_through_points_intersection :
  (∃ (x_0 y_0 r : ℝ), circleEquation x_0 y_0 ∧
   (x_0 + 2)^2 + (y_0 - 4)^2 = r^2 ∧
   (x_0 - 3)^2 + (y_0 + 1)^2 = r^2 ∧
   y_0 = -(3 / 2) ∧
   r = 3 * sqrt 2 / 2) :=
by
  -- We prove that these conditions derive the equation of the circle
  sorry

end circle_through_points_intersection_l815_815753


namespace diagonals_in_decagon_l815_815419

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_decagon : number_of_diagonals 10 = 35 := by
  sorry

end diagonals_in_decagon_l815_815419


namespace math_problem_l815_815020

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + (Real.log10 a + 2) * x + Real.log10 b

theorem math_problem
  (a b : ℝ)
  (h1 : f a b (-1) = -2)
  (h2 : ∀ x : ℝ, f a b x ≥ 2 * x) :
  a = 100 ∧ b = 10 ∧ { x : ℝ | f 100 10 x < x + 5 } = { x : ℝ | -4 < x ∧ x < 1 } :=
by
  sorry

end math_problem_l815_815020


namespace highest_power_of_3_in_n_l815_815649

-- Definition of N as described in the problem
def N : ℕ := -- Here you would input the specific function that generates this large number

-- Lean does not easily handle large formed numbers directly, so make assumptions
-- Instead let's use the properties given
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

example : sum_of_digits N = 315 := sorry

theorem highest_power_of_3_in_n :
  ∃ k : ℕ, ∃ m : ℕ, (N = 3^k * m) ∧ (¬ ∃ l, l > k ∧ ∃ m', N = 3^l * m') ∧ k = 2 :=
begin
  -- The proof would proceed here
  sorry
end

end highest_power_of_3_in_n_l815_815649


namespace segment_CM_length_l815_815087

open Classical

-- Define the variables
variables (a b : ℝ) (h : ℝ) (x : ℝ)

-- Define the conditions
def trapezoid (AD BC : ℝ) := AD = a ∧ BC = b

def point_on_extension (M : ℝ) := true -- Placeholder for existence of point M on extension.

def cuts_off_one_fourth_area (area_trap whole_area : ℝ) := area_trap = whole_area / 4

-- Define the area of the trapezoid
def area_trapezoid (a b h : ℝ) : ℝ := (a + b) * h / 2

-- Main theorem statement
theorem segment_CM_length (a b : ℝ) (h : ℝ) (p : point_on_extension x) (t : trapezoid a b) :
  let whole_area := area_trapezoid a b h in 
  let area_trap := (1/4 : ℝ) * whole_area in
  cuts_off_one_fourth_area area_trap whole_area →
  ∃ x, x = a * (3*a - b) / (a + b) ∨ x = a * (a - 3*b) / (3 * (a + b)) :=
begin
  sorry -- Proof omitted
end

end segment_CM_length_l815_815087


namespace LittleRedHeightCorrect_l815_815908

noncomputable def LittleRedHeight : ℝ :=
let LittleMingHeight := 1.3 
let HeightDifference := 0.2 
LittleMingHeight - HeightDifference

theorem LittleRedHeightCorrect : LittleRedHeight = 1.1 := by
  sorry

end LittleRedHeightCorrect_l815_815908


namespace PQRS_is_parallelogram_l815_815574

noncomputable theory
open_locale classical

variables 
{S1 S2 : Type*} [circle S1] [circle S2] 
(A B Q R S P : point)
(tangent1 : tangent_line A S1 Q) 
(tangent2 : tangent_line B S2 S)
(intersect1 : intersect_line B Q S1 R)
(intersect2 : intersect_line A S S2 P)

theorem PQRS_is_parallelogram
  (hTangent1 : tangent1 = tangent_line.mk A S1 Q) 
  (hTangent2 : tangent2 = tangent_line.mk B S2 S)
  (hIntersect1 : intersect1 = intersect_line.mk B Q S1 R) 
  (hIntersect2 : intersect2 = intersect_line.mk A S S2 P) :
  is_parallelogram P Q R S :=
sorry

end PQRS_is_parallelogram_l815_815574


namespace prove_ff_of_half_eq_half_l815_815397

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

-- Proof statement
theorem prove_ff_of_half_eq_half : f (f (1 / 2)) = 1 / 2 := by
  sorry

end prove_ff_of_half_eq_half_l815_815397


namespace count_valid_arrangements_l815_815213

-- Definitions for colors
inductive Color | orange | red | blue | yellow | green | purple

open Color

def valid_arrangements : Finset (List Color) :=
  Finset.filter (λ arrangement,
    (∃ g b, g < b ∧ arrangement.nth g = some green ∧ arrangement.nth b = some blue) ∧
    (∃ o p, o < p ∧ arrangement.nth o = some orange ∧ arrangement.nth p = some purple) ∧
    (∀ g b, arrangement.nth g = some green → arrangement.nth b = some blue → (g + 1 ≠ b ∧ b + 1 ≠ g))
  )
  (Finset.map (Equiv.ListEquiv.apply (Finₙ 5 ↪ Color)) (Finset.List.finPerms 5))

theorem count_valid_arrangements :
  valid_arrangements.card = 7 :=
sorry

end count_valid_arrangements_l815_815213


namespace complex_conjugate_pure_imaginary_l815_815839

-- Declare the necessary variables and types
variables {z : ℂ}

-- Define the conditions and proof statement
theorem complex_conjugate_pure_imaginary (hz_abs : |z| = 3) (hz_pure : ∃ y : ℝ, z + 3 * complex.I = y * complex.I) : z.conj = -3 * complex.I :=
by
  sorry

end complex_conjugate_pure_imaginary_l815_815839


namespace solution_example_l815_815581

def isDigit080 (d : ℕ) : Prop := d = 8 ∨ d = 0

def isDigits080 (n : ℕ) : Prop := ∀ (i : ℕ), (n / 10^i % 10) = 0 ∨ (n / 10^i % 10) = 8

def largestMultipleOf15WithDigits080 : ℕ :=
  nat.find_greatest (λ n, n > 0 ∧ (n % 15 = 0) ∧ isDigits080 n) 9999 -- Upper limit is a large reasonable number assuming constraints.

theorem solution_example : ∃ n, n = largestMultipleOf15WithDigits080 ∧ n / 15 = 592 :=
begin
  use largestMultipleOf15WithDigits080,
  split,
  { refl },
  { sorry } -- Proof that n == 8880, hence n / 15 == 592
end

end solution_example_l815_815581


namespace radius_B_is_correct_l815_815717

-- Define radii of circles A and D
def radius_A : ℝ := 2
def radius_D : ℝ := 3

-- Radii of circles B and C
noncomputable def radius_B : ℝ := 3 / 5

-- Centers of circles, with E, H, F
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- E is the center of circle A
def E : Point := Point.mk 0 0

-- F is the center of circle D
def F : Point := Point.mk 1 0

-- Distance between E and F
axiom distEF : (E.x - F.x)^2 + (E.y - F.y)^2 = (radius_D - radius_A)^2

-- EH = 2 + r
def distEH (r : ℝ) : ℝ := radius_A + r

-- HF = 3 - r
def distHF (r : ℝ) : ℝ := radius_D - r

-- Pythagorean Theorem
axiom pythagorean (r : ℝ) : distEF = distEH r ^ 2 - distHF r ^ 2

-- Define the proof problem
theorem radius_B_is_correct : radius_B = 3 / 5 := by
  intro r h
  sorry

end radius_B_is_correct_l815_815717


namespace trigonometric_identity_l815_815049

theorem trigonometric_identity (α : ℝ) (h : Real.tan (π - α) = -2) :
  (Real.cos (2 * π - α) + 2 * Real.cos (3 * π / 2 - α)) / (Real.sin (π - α) - Real.sin (-π / 2 - α)) = -1 :=
by
  sorry

end trigonometric_identity_l815_815049


namespace book_selection_methods_l815_815328

theorem book_selection_methods {SS_books NS_books : ℕ} (h1 : SS_books = 5) (h2 : NS_books = 4) :
  ∑ (k : ℕ) in {1, 2, 3, 4}, (Nat.choose SS_books (k + 1) * Nat.choose NS_books k) = 121 :=
by
  sorry

end book_selection_methods_l815_815328


namespace certain_event_triangle_interior_angles_l815_815634

theorem certain_event_triangle_interior_angles (T : Type) [Triangle T] :
  (∀ (t : T), sum_interior_angles t = 180) :=
sorry

end certain_event_triangle_interior_angles_l815_815634


namespace sqrt3_binary_has_one_in_interval_l815_815121

theorem sqrt3_binary_has_one_in_interval (n : ℕ) (n_pos: 0 < n)
  (b : ℕ → ℕ) (h : ∀ i, b i ∈ {0, 1})
  (h_sqrt3 : ∑ i in finset.range (2 * n + 1), b i * (2:ℚ)^(-i) = 3^(1/2)) :
  ∃ k, n ≤ k ∧ k ≤ 2 * n ∧ b k = 1 := 
sorry

end sqrt3_binary_has_one_in_interval_l815_815121


namespace nested_root_of_0_001_l815_815311

noncomputable def nested_root (x : ℝ) : ℝ :=
  Float.toDec (sqrt (cbrt x))

theorem nested_root_of_0_001 :
  nested_root 0.001 = 0.3 :=
by
  sorry

end nested_root_of_0_001_l815_815311


namespace unit_digit_12_pow_100_l815_815233

theorem unit_digit_12_pow_100 : 
    let cycle := [2, 4, 8, 6] in
    (cycle[(100 % 4)] = 6)
:=
by
  sorry

end unit_digit_12_pow_100_l815_815233


namespace increasing_function_solve_inequality_l815_815902

noncomputable def f : ℝ → ℝ := sorry
axiom add_property : ∀ m n : ℝ, f (m + n) = f m + f n
axiom pos_property : ∀ x : ℝ, x > 0 → f x > 0

theorem increasing_function :
  ∀ x₁ x₂ : ℝ, x₂ > x₁ → f x₂ > f x₁ := sorry

theorem solve_inequality (h : f 1 = 1) :
  ∀ x : ℝ, f (Real.log 2 (x ^ 2 - x - 2)) < 2 ↔ ((-2 < x ∧ x < -1) ∨ (2 < x ∧ x < 3)) := sorry

end increasing_function_solve_inequality_l815_815902


namespace triangle_inscribed_circle_AC_length_l815_815484

theorem triangle_inscribed_circle_AC_length:
  ∀ (A B C : Type)
    [metric_space A] [metric_space B] [metric_space C]
    [inhabited A] [inhabited B] [inhabited C],
    let r := 4 in
    let B_angle := 45 in
    ∃ (AC : ℝ), AC = 8 :=
by 
  sorry

end triangle_inscribed_circle_AC_length_l815_815484


namespace problem_I_problem_II_l815_815405

theorem problem_I (m : ℝ) (h1 : ∀ x : ℝ, |2 * x - 1| ≤ 2 * m + 1 ↔ x ∈ Icc -2 2) : m = 3 / 2 :=
sorry

theorem problem_II (a : ℝ) (h2 : ∀ x y : ℝ, |2 * x - 1| ≤ 2^y + a / 2^y + |2 * x + 3|) : a ≥ 4 :=
sorry

end problem_I_problem_II_l815_815405


namespace triangle_side_length_condition_l815_815323

theorem triangle_side_length_condition (v : ℝ) (h1 : 3 * v - 2 ≥ 0) (h2 : 3 * v + 2 ≥ 0) (h3 : 6 * v ≥ 0)
  (h_triangle_ineq1 : real.sqrt (3 * v - 2) + real.sqrt (3 * v + 2) > real.sqrt (6 * v))
  (h_triangle_ineq2 : real.sqrt (3 * v - 2) + real.sqrt (6 * v) > real.sqrt (3 * v + 2))
  (h_triangle_ineq3 : real.sqrt (3 * v + 2) + real.sqrt (6 * v) > real.sqrt (3 * v - 2)) :
  ∃ (angle : ℝ), angle = 90 :=
by
  existsi 90
  sorry

end triangle_side_length_condition_l815_815323


namespace sum_of_three_squares_power_l815_815834

theorem sum_of_three_squares_power (n a b c k : ℕ) (h : n = a^2 + b^2 + c^2) (h_pos : n > 0) (k_pos : k > 0) :
  ∃ A B C : ℕ, n^(2*k) = A^2 + B^2 + C^2 :=
by
  sorry

end sum_of_three_squares_power_l815_815834


namespace max_elevation_reached_l815_815682

theorem max_elevation_reached 
  (t : ℝ) 
  (s : ℝ) 
  (h : s = 200 * t - 20 * t^2) : 
  ∃ t_max : ℝ, ∃ s_max : ℝ, t_max = 5 ∧ s_max = 500 ∧ s_max = 200 * t_max - 20 * t_max^2 := sorry

end max_elevation_reached_l815_815682


namespace mark_donates_1800_cans_l815_815538

variable (number_of_shelters people_per_shelter cans_per_person : ℕ)
variable (total_people total_cans_of_soup : ℕ)

-- Given conditions
def number_of_shelters := 6
def people_per_shelter := 30
def cans_per_person := 10

-- Calculations based on conditions
def total_people := number_of_shelters * people_per_shelter
def total_cans_of_soup := total_people * cans_per_person

-- Proof statement
theorem mark_donates_1800_cans : total_cans_of_soup = 1800 := by
  -- stretch sorry proof placeholder for the proof
  sorry

end mark_donates_1800_cans_l815_815538


namespace expected_value_is_150_l815_815913

noncomputable def expected_value_of_winnings : ℝ :=
  let p := (1:ℝ)/8
  let winnings := [0, 2, 3, 5, 7]
  let losses := [4, 6]
  let extra := 5
  let win_sum := (winnings.sum : ℝ)
  let loss_sum := (losses.sum : ℝ)
  let E := p * 0 + p * win_sum - p * loss_sum + p * extra
  E

theorem expected_value_is_150 : expected_value_of_winnings = 1.5 := 
by sorry

end expected_value_is_150_l815_815913


namespace combination_problem_l815_815843

theorem combination_problem (x : ℕ) (hx_pos : 0 < x) (h_comb : Nat.choose 9 x = Nat.choose 9 (2 * x + 3)) : x = 2 :=
by {
  sorry
}

end combination_problem_l815_815843


namespace a_n_eq_n_l815_815385

variable (a : Nat → Nat)
variable {n : Nat}
variable (h1 : ∀ n : Nat, n > 0 → a n > 0) 
variable (h2 : ∀ n : Nat, n > 0 → (∑ j in Finset.range n.succ, a j ^ 3) 
  = (∑ j in Finset.range n.succ, a j) ^ 2)

theorem a_n_eq_n : ∀ n : Nat, n > 0 → a n = n :=
by
  assume n,
  assume hn : n > 0,
  sorry

end a_n_eq_n_l815_815385


namespace polynomial_sum_of_coeffs_l815_815592

theorem polynomial_sum_of_coeffs
  (p q r s : ℝ)
  (h : ∀ x : ℂ, (x^4 + (p * x^3) + (q * x^2) + (r * x) + s = 0) → ((x = 3 * complex.I) ∨ (x = 1 + 2*complex.I) ∨ (x = -3 * complex.I) ∨ (x = 1 - 2*complex.I))) :
  p + q + r + s = 39 := sorry

end polynomial_sum_of_coeffs_l815_815592


namespace luncheon_cost_l815_815947

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + 2 * p = 3.50)
  (h2 : 3 * s + 7 * c + 2 * p = 4.90) :
  s + c + p = 1.00 :=
  sorry

end luncheon_cost_l815_815947


namespace go_stones_perimeter_l815_815459

-- Define the conditions for the problem
def stones_wide : ℕ := 4
def stones_tall : ℕ := 8

-- Define what we want to prove based on the conditions
theorem go_stones_perimeter : 2 * stones_wide + 2 * stones_tall - 4 = 20 :=
by
  -- Proof would normally go here
  sorry

end go_stones_perimeter_l815_815459


namespace james_total_room_area_l815_815100

theorem james_total_room_area :
  let original_length := 13
  let original_width := 18
  let increase := 2
  let new_length := original_length + increase
  let new_width := original_width + increase
  let area_of_one_room := new_length * new_width
  let number_of_rooms := 4
  let area_of_four_rooms := area_of_one_room * number_of_rooms
  let area_of_larger_room := area_of_one_room * 2
  let total_area := area_of_four_rooms + area_of_larger_room
  total_area = 1800
  := sorry

end james_total_room_area_l815_815100


namespace sequence_general_formula_sequence_bn_sum_n_terms_l815_815369

theorem sequence_general_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = 2 * n^2 + 2 * n) :
  (a 1 = 4) ∧ (∀ n, a (n + 1) = S (n + 1) - S n → a (n + 1) = 4 * (n + 1)) :=
sorry

theorem sequence_bn_sum_n_terms (S : ℕ → ℕ) (a b : ℕ → ℕ)
  (hS : ∀ n, S n = 2 * n^2 + 2 * n)
  (ha : ∀ n, a (n + 1) = S (n + 1) - S n)
  (hb : ∀ n, log 2 (b n) = a n) :
  (∀ n, b n = 2 ^ (4 * n)) → (∀ n, ∑ i in range n, b i = (16^n - 1) * 16 / 15) :=
sorry

end sequence_general_formula_sequence_bn_sum_n_terms_l815_815369


namespace incircle_incenters_perpendicular_l815_815300

theorem incircle_incenters_perpendicular (A B C D E F : Point) (incircle_touch_BC_at_D : incircle_touches_side BC A B C D) 
  (E_is_incenter_ABD : incenter_of_tri A B D E) (F_is_incenter_ACD : incenter_of_tri A C D F) : 
  perp EF AD := 
sorry

end incircle_incenters_perpendicular_l815_815300


namespace find_m_eq_mt_sqrt_3_l815_815814

theorem find_m_eq_mt_sqrt_3 
  (m : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + m * x - 1 / 4 = 0)
  (parabola_eq : ∀ x y : ℝ, y = 1 / 4 * x^2)
  (tangent_condition : (m / 2) ^ 2 + (0 + 1) ^ 2 = (1 + m ^ 2) / 4) :
  m = sqrt 3 ∨ m = -sqrt 3 :=
by {
  sorry
}

end find_m_eq_mt_sqrt_3_l815_815814


namespace inequality_solution_l815_815337

theorem inequality_solution :
  {x : ℝ | (x - 3) * (x + 2) ≠ 0 ∧ (x^2 + 1) / ((x - 3) * (x + 2)) ≥ 0} = 
  {x : ℝ | x ≤ -2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end inequality_solution_l815_815337


namespace vanaspati_percentage_after_adding_ghee_l815_815079

theorem vanaspati_percentage_after_adding_ghee :
  ∀ (initial_quantity added_pure_ghee : ℝ), 
    initial_quantity = 10 →
    added_pure_ghee = 10 →
    let pure_ghee_initial := 0.60 * initial_quantity in
    let vanaspati_initial := 0.40 * initial_quantity in
    let total_quantity := initial_quantity + added_pure_ghee in
    let new_percentage_vanaspati := (vanaspati_initial / total_quantity) * 100 in
    new_percentage_vanaspati = 20 := 
by
  intros initial_quantity added_pure_ghee h1 h2 pure_ghee_initial vanaspati_initial total_quantity new_percentage_vanaspati
  /- Given the constraints, it can be shown that the new percentage of vanaspati ghee is 20%. -/
  sorry

end vanaspati_percentage_after_adding_ghee_l815_815079


namespace min_value_of_a_l815_815411

theorem min_value_of_a (x y : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) 
  (h : ∀ x y, 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  4 ≤ a :=
sorry

end min_value_of_a_l815_815411


namespace evaluate_composite_function_l815_815783

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_composite_function : g (h 2) = 5288 := by
  sorry

end evaluate_composite_function_l815_815783


namespace math_problem_prove_l815_815238

noncomputable def percentile_60th (data : List ℕ) : ℚ :=
  let sorted := data.sort
  let n := sorted.length
  if n = 0 then
    0
  else
    let position := (n * 60 / 100) - 1
    if position < n - 1 then
      (sorted.nth position).getD 0 + ((sorted.nth (position + 1)).getD 0 - (sorted.nth position).getD 0) / 2
    else
      (sorted.nth position).getD 0

def optionA : Bool :=
  percentile_60th [64, 91, 72, 75, 85, 76, 78, 86, 79, 92] = 79

def binomial_prob (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  nat.choose n k * p^k * (1 - p)^(n - k)

def optionB : Bool :=
  binomial_prob 4 (1/2) 3 = 1/4

noncomputable def normal_cdf (μ σ x : ℚ) : ℚ :=
  Real.err (σ * (x - μ))

def optionC : Bool :=
  let μ := 5
  let σ := ? -- missing parameter; assumed given via distribution
  normal_cdf μ σ 2 = 0.1 ∧ 1 - normal_cdf μ σ 2 - normal_cdf μ sigma 8 = 0.8

def optionD : Bool :=
  let students_11 := 400
  let students_12 := 360
  let total_selected := 57
  let selected_11 := 20
  let proportion := selected_11 / students_11.to_float
  let selected_12 := proportion * students_12.to_float
  let selected_13 := total_selected.to_float - selected_11 - selected_12
  abs (selected_13 - 19) < 1e-1

def final_result : Prop :=
  ¬optionA ∧ optionB ∧ optionC ∧ optionD

theorem math_problem_prove : final_result :=
by
  sorry

end math_problem_prove_l815_815238


namespace sum_of_squares_of_roots_eq_zero_l815_815325

theorem sum_of_squares_of_roots_eq_zero :
  let p : Polynomial ℂ := Polynomial.X ^ 2020 + 50 * Polynomial.X ^ 2017 + 5 * Polynomial.X ^ 4 + 450 in
  let roots := p.roots in
  (∑ root in roots.toFinset, root ^ 2) = 0 :=
by
  sorry

end sum_of_squares_of_roots_eq_zero_l815_815325


namespace players_even_sum_probability_l815_815939

-- Define the problem in Lean 4
theorem players_even_sum_probability :
  let tiles := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  ∃ (m n : ℕ), gcd m n = 1 ∧
    (let chosen_sets := (finset.powersetLen 3 tiles).to_finset in
    let distributions := chosen_sets × chosen_sets × chosen_sets in
    let valid_distributions := 
      {triple ∈ distributions | 
      let ⟨a, b, c⟩ := triple in
      ((a ∪ b ∪ c).card = 9 ∧ ((a.sum % 2 = 0) ∧ (b.sum % 2 = 0) ∧ (c.sum % 2 = 0)) ∧ (tiles - (a ∪ b ∪ c)).card = 1)} in
    let total_valid := valid_distributions.card in
    let total_distributions := (finset.powersetLen 3 tiles).to_finset.card * ((finset.powersetLen 6 tiles).to_finset.powersetLen 3).card in
    total_valid / total_distributions = ↑m / ↑n ∧ m + n = 29) :=
sorry

end players_even_sum_probability_l815_815939


namespace Heejin_is_oldest_l815_815650

variable (Yoona : ℕ) (Miyoung : ℕ) (Heejin : ℕ)

def oldest (Yoona Miyoung Heejin : ℕ) : Prop :=
  Heejin > Yoona ∧ Heejin > Miyoung

theorem Heejin_is_oldest : oldest 23 22 24 :=
by
  unfold oldest
  constructor
  · exact Nat.lt_trans (Nat.zero_lt_succ 22) (Nat.succ_lt_succ (Nat.zero_lt_succ 22))
  · exact Nat.succ_lt_succ (Nat.zero_lt_succ 21)

end Heejin_is_oldest_l815_815650


namespace ring_worth_l815_815882

theorem ring_worth (R : ℝ) (h1 : (R + 2000 + 2 * R = 14000)) : R = 4000 :=
by 
  sorry

end ring_worth_l815_815882


namespace range_of_abscissa_l815_815373

noncomputable def midpoint (C : ℝ × ℝ) : ℝ × ℝ :=
  ((C.1 / 2), (1 - C.1))

theorem range_of_abscissa (x₀ : ℝ) (hC : 2 * x₀ + (2 - 2 * x₀) - 2 = 0) :
  ∃ (A B : ℝ × ℝ), 
    ((∃ t : ℝ, A = (cos t, sin t) ∧ B = (cos (-t), sin (-t))) ∧ ((midpoint (x₀, 2 - 2 * x₀)).1 ^ 2 + (midpoint (x₀, 2 - 2 * x₀)).2 ^ 2 < 1)) → 
    0 < x₀ ∧ x₀ < 8 / 5 :=
by
  sorry

end range_of_abscissa_l815_815373


namespace area_AOC_l815_815710

-- Definitions based on the conditions
variable (BC AC S_ABC S_ACC1 CO OC1 : ℝ)
variable (h1 : AC / BC = 3 / 2)
variable (h2 : BC = 4)
variable (h3 : S_ABC = 15 * real.sqrt 7 / 4)
variable (h4 : S_ACC1 = 3 / 5 * S_ABC)
variable (h5 : CO / OC1 = 2)

-- The main theorem to prove
theorem area_AOC (h_AC : AC = 6) :
    2 / 3 * S_ACC1 = 3 * real.sqrt 7 / 2 :=
sorry

end area_AOC_l815_815710


namespace S_seq_bounds_l815_815028

def a_seq : ℕ → ℕ
| 0 := 9
| (n+1) := a_seq n + 2 * n + 5

def b_seq : ℕ → ℚ
| 0 := 1 / 4
| (n+1) := (n + 1) / (n + 2) * b_seq n

def S_seq (n : ℕ) : ℚ := 
∑ i in List.range n, b_seq i / Real.sqrt (a_seq i)

theorem S_seq_bounds (n : ℕ) : 1 / 12 ≤ S_seq n ∧ S_seq n < 1 / 4 :=
by
  sorry

end S_seq_bounds_l815_815028


namespace problem_proof_l815_815384

-- Define the given conditions and the target statement
theorem problem_proof (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 10.5) : a^2 + b^2 = 25 := 
by sorry

end problem_proof_l815_815384


namespace condition_C_implies_alpha_parallel_beta_l815_815791

variables {Line Plane : Type}
variable (m n : Line)
variables (α β γ : Plane)

-- Define the parallel and perpendicular relationships as predicates
variable parallel : Line → Line → Prop
variable perp : Line → Plane → Prop
variable p_parallel : Plane → Plane → Prop

axiom line_distinct : m ≠ n
axiom plane_distinct : α ≠ β ∧ β ≠ γ ∧ γ ≠ α

-- The given problem statement
theorem condition_C_implies_alpha_parallel_beta 
  (h1 : parallel n m)
  (h2 : perp n α)
  (h3 : perp m β) 
  : p_parallel α β :=
sorry

end condition_C_implies_alpha_parallel_beta_l815_815791


namespace shaded_region_perimeter_l815_815611

def radius_of_circle (C : ℝ) : ℝ := C / (2 * Real.pi)

def arc_length (r : ℝ) (angle : ℝ) : ℝ := angle/360 * (2 * Real.pi * r)

def perimeter_of_shaded_region (C : ℝ) : ℝ :=
  let r := radius_of_circle C in
  let arc_len := arc_length r 60 in
  3 * arc_len

theorem shaded_region_perimeter (C : ℝ) (hC: C = 30) : (perimeter_of_shaded_region C) = 15 :=
by
  sorry

end shaded_region_perimeter_l815_815611


namespace range_of_a_l815_815776

variable {α : Type*} [DecidableLinearOrder α] -- Assuming α is a type with order properties, suitable for real numbers

def A (x : α) : Prop := 1 ≤ x ∧ x ≤ 5

def B (a x : α) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

def condition_p (x : α) : Prop := A x

def condition_q (a x : α) : Prop := B a x

theorem range_of_a (a : α) :
  (∀ x, condition_q a x → condition_p x) → (∃ x, ¬condition_p x ∧ ¬condition_q a x) → (2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l815_815776


namespace probability_at_least_75_cents_l815_815667

def total_coins : ℕ := 3 + 5 + 4 + 3 -- total number of coins

def pennies : ℕ := 3
def nickels : ℕ := 5
def dimes : ℕ := 4
def quarters : ℕ := 3

def successful_outcomes_case1 : ℕ := (Nat.choose 3 3) * (Nat.choose 12 3)
def successful_outcomes_case2 : ℕ := (Nat.choose 3 2) * (Nat.choose 4 2) * (Nat.choose 5 2)

def total_outcomes : ℕ := Nat.choose 15 6
def successful_outcomes : ℕ := successful_outcomes_case1 + successful_outcomes_case2

def probability : ℚ := successful_outcomes / total_outcomes

theorem probability_at_least_75_cents :
  probability = 400 / 5005 := by
  sorry

end probability_at_least_75_cents_l815_815667


namespace cot_B_plus_cot_C_inequality_l815_815495

theorem cot_B_plus_cot_C_inequality 
  (A B C G D X : Type) [triangle : ∀ {A B C : Type}, triangle.abc A B C] 
  (med_perp : ∀ {A B C : Type}, medians_perpendicular_to_sides A B C) : 
  ∀ (cot_B cot_C : ℝ), 
  (cot_B + cot_C ≥ 2 / 3) :=
by 
  sorry

end cot_B_plus_cot_C_inequality_l815_815495


namespace max_sum_of_sequence_l815_815918

open BigOperators

theorem max_sum_of_sequence (n : ℕ) (a : ℕ → ℤ)
  (h_n : n ≥ 40)
  (h_a : ∀ i, a i = 1 ∨ a i = -1)
  (h_sum_40 : ∀ i, (∑ j in finset.range (i + 40), a j) = 0)
  (h_sum_42 : ∀ i, (∑ j in finset.range (i + 42), a j) ≠ 0) :
  ∑ i in finset.range n, a i ≤ 20 :=
sorry

end max_sum_of_sequence_l815_815918


namespace integer_solution_count_l815_815453

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end integer_solution_count_l815_815453


namespace projection_of_a_in_direction_of_b_l815_815830

noncomputable def vector_projection_in_direction (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / magnitude_b

theorem projection_of_a_in_direction_of_b :
  vector_projection_in_direction (3, 2) (-2, 1) = -4 * Real.sqrt 5 / 5 := 
by
  sorry

end projection_of_a_in_direction_of_b_l815_815830


namespace arc_length_of_regular_octagon_l815_815687

-- Defining a regular octagon inscribed in a circle and its properties
def regular_octagon_side_length := 4

def regular_octagon_circumference (side_length : ℝ) : ℝ :=
  2 * Real.pi * side_length

-- Arc length corresponding to one side of the octagon
def arc_length_one_side (circumference : ℝ) : ℝ :=
  circumference / 8

-- The theorem to be proved
theorem arc_length_of_regular_octagon : 
  arc_length_one_side (regular_octagon_circumference regular_octagon_side_length) = Real.pi :=
by
  sorry

end arc_length_of_regular_octagon_l815_815687


namespace find_middle_number_l815_815533

theorem find_middle_number (x y z : ℕ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 22) (h4 : x + z = 29) (h5 : y + z = 31) (h6 : x = 10) :
  y = 12 :=
sorry

end find_middle_number_l815_815533


namespace projective_plane_coloring_l815_815784

theorem projective_plane_coloring
  (p k n : ℕ)
  (hp : Nat.Prime p)
  (hk : k = p + 1)
  (hn : n = k^2 - k + 1) :
  ∃ color : Fin n × Fin n → Prop,
    (∀ i, (Finset.filter (λ j, color (i, j)) Finset.univ).card = k) ∧
    (∀ j, (Finset.filter (λ i, color (i, j)) Finset.univ).card = k) ∧
    (∀ (i₁ i₂ j₁ j₂ : Fin n), i₁ ≠ i₂ → j₁ ≠ j₂ → ¬(color (i₁, j₁) ∧ color (i₁, j₂) ∧ color (i₂, j₁) ∧ color (i₂, j₂))) :=
sorry

end projective_plane_coloring_l815_815784


namespace larger_number_is_23_l815_815987

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815987


namespace correct_trig_comparison_l815_815630

theorem correct_trig_comparison :
  ¬ (sin (19 * Real.pi / 8) < cos (14 * Real.pi / 9)) ∧
  ¬ (sin (-54 * Real.pi / 7) < sin (-63 * Real.pi / 8)) ∧
  (tan (-13 * Real.pi / 4) > tan (-17 * Real.pi / 5)) ∧
  ¬ (tan 138 -° > tan 143 -°) :=
sorry 

end correct_trig_comparison_l815_815630


namespace standard_eq_of_parabola_l815_815599

-- Conditions:
-- The point (1, -2) lies on the parabola.
def point_on_parabola : Prop := ∃ p : ℝ, (1, -2).2^2 = 2 * p * (1, -2).1 ∨ (1, -2).1^2 = 2 * p * (1, -2).2

-- Question to be proved:
-- The standard equation of the parabola passing through the point (1, -2) is y^2 = 4x or x^2 = - (1/2) y.
theorem standard_eq_of_parabola : point_on_parabola → (y^2 = 4*x ∨ x^2 = -(1/(2:ℝ)) * y) :=
by
  sorry -- proof to be provided

end standard_eq_of_parabola_l815_815599


namespace mans_rate_in_still_water_l815_815248

/-- The man's rowing speed in still water given his rowing speeds with and against the stream. -/
theorem mans_rate_in_still_water (v_with_stream v_against_stream : ℝ) (h1 : v_with_stream = 6) (h2 : v_against_stream = 2) : (v_with_stream + v_against_stream) / 2 = 4 := by
  sorry

end mans_rate_in_still_water_l815_815248


namespace integer_solutions_to_inequality_l815_815435

theorem integer_solutions_to_inequality :
  ∃ n : ℕ, n = 5 ∧ (∀ x : ℤ, (x - 2) ^ 2 ≤ 4 ↔ x ∈ {0, 1, 2, 3, 4}) :=
by
  sorry

end integer_solutions_to_inequality_l815_815435


namespace intersect_at_two_points_l815_815033

theorem intersect_at_two_points (a : ℝ) :
  (∃ p q : ℝ × ℝ, 
    (p.1 - p.2 + 1 = 0) ∧ (2 * p.1 + p.2 - 4 = 0) ∧ (a * p.1 - p.2 + 2 = 0) ∧
    (q.1 - q.2 + 1 = 0) ∧ (2 * q.1 + q.2 - 4 = 0) ∧ (a * q.1 - q.2 + 2 = 0) ∧ p ≠ q) →
  (a = 1 ∨ a = -2) :=
by 
  sorry

end intersect_at_two_points_l815_815033


namespace right_triangle_legs_l815_815861

theorem right_triangle_legs (a b : ℕ) (hypotenuse : ℕ) (h : hypotenuse = 39) : a^2 + b^2 = 39^2 → (a = 15 ∧ b = 36) ∨ (a = 36 ∧ b = 15) :=
by
  sorry

end right_triangle_legs_l815_815861


namespace distance_between_intersections_l815_815585

theorem distance_between_intersections :
  let a := 3
  let b := 2
  let c := -7
  let x1 := (-1 + Real.sqrt 22) / 3
  let x2 := (-1 - Real.sqrt 22) / 3
  let distance := abs (x1 - x2)
  let p := 88  -- 2^2 * 22 = 88
  let q := 9   -- 3^2 = 9
  distance = 2 * Real.sqrt 22 / 3 →
  p - q = 79 :=
by
  sorry

end distance_between_intersections_l815_815585


namespace min_omega_l815_815017

theorem min_omega (ω : ℝ) (hω_pos : ω > 0) :
  (∀ x, f x = sin (ω * x) → g x = sin (ω * (x + π / 6)) = 
  cos (ω * x)) → 
  ∃ k ∈ ℤ, ω = 6 * k + 3 ∧ ∀ n ∈ ℤ, 6 * n + 3 > 0 → ω = 3 :=
sorry

end min_omega_l815_815017


namespace contestant_wins_with_prob_1_over_9_l815_815281

noncomputable def probability_of_winning : ℚ :=
  let prob_correct : ℚ := 1/3 in
  let prob_incorrect : ℚ := 2/3 in
  let prob_all_correct := prob_correct ^ 4 in
  let prob_three_correct := 4 * (prob_correct ^ 3 * prob_incorrect) in
  prob_all_correct + prob_three_correct

theorem contestant_wins_with_prob_1_over_9 :
  probability_of_winning = 1/9 :=
by
  sorry

end contestant_wins_with_prob_1_over_9_l815_815281


namespace train_speed_l815_815249

theorem train_speed (L : ℝ) (T : ℝ) (L_pos : 0 < L) (T_pos : 0 < T) (L_eq : L = 150) (T_eq : T = 3) : L / T = 50 := by
  sorry

end train_speed_l815_815249


namespace integer_solution_count_l815_815452

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end integer_solution_count_l815_815452


namespace pyramid_ratio_l815_815188

theorem pyramid_ratio (a m : ℝ) (h_square : m^2 > 0) (h_base : a > 0) :
  let lateral_area := (3 / 4) * (a * real.sqrt (9 * m^2 - 3 * a^2 + 6 * a * m)) / (a - m)
  let base_area := (real.sqrt 3 / 4) * a^2
  base_area > 0 → lateral_area > 0 → 
  (lateral_area / base_area) = real.sqrt (9 * m^2 - 3 * a^2 + 6 * a * m) / (a - m) :=
begin
  -- Dummy proof; replace with the actual proof
  intros,
  sorry
end

end pyramid_ratio_l815_815188


namespace total_area_correct_l815_815097

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase : ℕ := 2
def num_extra_rooms_same_size : ℕ := 3
def num_double_size_rooms : ℕ := 1

def increased_length : ℕ := initial_length + increase
def increased_width : ℕ := initial_width + increase

def area_of_one_room (length width : ℕ) : ℕ := length * width

def number_of_rooms : ℕ := 1 + num_extra_rooms_same_size

def total_area_same_size_rooms : ℕ := number_of_rooms * area_of_one_room increased_length increased_width

def total_area_extra_size_room : ℕ := num_double_size_rooms * 2 * area_of_one_room increased_length increased_width

def total_area : ℕ := total_area_same_size_rooms + total_area_extra_size_room

theorem total_area_correct : total_area = 1800 := by
  -- Proof omitted
  sorry

end total_area_correct_l815_815097


namespace inequality_solution_l815_815345

open Set

noncomputable def solution_set := { x : ℝ | 5 - x^2 > 4 * x }

theorem inequality_solution :
  solution_set = { x : ℝ | -5 < x ∧ x < 1 } :=
by
  sorry

end inequality_solution_l815_815345


namespace find_b_l815_815335

def has_exactly_one_real_solution (f : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = 0

theorem find_b (b : ℝ) :
  (∃! (x : ℝ), x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) ↔ b < 2 :=
by
  sorry

end find_b_l815_815335


namespace players_even_sum_probability_l815_815940

-- Define the problem in Lean 4
theorem players_even_sum_probability :
  let tiles := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  ∃ (m n : ℕ), gcd m n = 1 ∧
    (let chosen_sets := (finset.powersetLen 3 tiles).to_finset in
    let distributions := chosen_sets × chosen_sets × chosen_sets in
    let valid_distributions := 
      {triple ∈ distributions | 
      let ⟨a, b, c⟩ := triple in
      ((a ∪ b ∪ c).card = 9 ∧ ((a.sum % 2 = 0) ∧ (b.sum % 2 = 0) ∧ (c.sum % 2 = 0)) ∧ (tiles - (a ∪ b ∪ c)).card = 1)} in
    let total_valid := valid_distributions.card in
    let total_distributions := (finset.powersetLen 3 tiles).to_finset.card * ((finset.powersetLen 6 tiles).to_finset.powersetLen 3).card in
    total_valid / total_distributions = ↑m / ↑n ∧ m + n = 29) :=
sorry

end players_even_sum_probability_l815_815940


namespace jason_lost_pokemon_cards_l815_815105

theorem jason_lost_pokemon_cards
  (initial_cards : ℕ)
  (cards_bought : ℕ)
  (final_cards : ℕ)
  (initial_cards = 676)
  (cards_bought = 224)
  (final_cards = 712) :
  initial_cards + cards_bought - final_cards = 188 :=
by sorry

end jason_lost_pokemon_cards_l815_815105


namespace round_to_scientific_notation_l815_815476

theorem round_to_scientific_notation (n : ℕ) (h : n = 26341) : round_to_3_significant_figures_in_scientific_notation n = 2.63 * 10^4 :=
by sorry

-- Assuming the function round_to_3_significant_figures_in_scientific_notation is defined elsewhere
/--
Function to round a given integer to 3 significant figures in scientific notation.
--/
def round_to_3_significant_figures_in_scientific_notation (x : ℕ) : ℝ :=
-- Implementation is assumed to be present and correct.
sorry

end round_to_scientific_notation_l815_815476


namespace tetrahedron_volume_ratio_l815_815073

theorem tetrahedron_volume_ratio (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 1 / 2) :
  (let V := λ e : ℝ, (sqrt 2 / 12) * e ^ 3 in V a / V b = 1 / 8) :=
by
  let V := λ e : ℝ, (sqrt 2 / 12) * e ^ 3
  -- We need to show that (V a / V b) = 1 / 8
  have V_a := V a
  have V_b := V b
  have : V a / V b = ((sqrt 2 / 12) * a ^ 3) / ((sqrt 2 / 12) * b ^ 3),
  {
    apply congr_arg (λ x, x / ((sqrt 2 / 12) * b ^ 3))
    refl
  },
  rw [this], clear this,
  rw [div_mul_eq_mul_div, mul_div_mul_left (sqrt 2) (a ^ 3) (b ^ 3) (sqrt_ne_zero 2), ← div_pow, pow_three],
  -- Given a / b = 1 / 2, we substitute this into (a / b) ^ 3
  rw [h3, div_pow, one_pow, two_pow, one_div],
  norm_num

end tetrahedron_volume_ratio_l815_815073


namespace number_of_integers_satisfying_inequality_l815_815438

theorem number_of_integers_satisfying_inequality : 
  (finset.filter (λ x, (x - 2)^2 ≤ 4) (finset.Icc 0 4)).card = 5 := 
  sorry

end number_of_integers_satisfying_inequality_l815_815438


namespace number_of_shapes_l815_815833

-- Define the conditions and the region boundaries
def region_bounds (x y : ℤ) : Prop :=
  y <= 3 * x ∧ y >= -1 ∧ x <= 6

-- Define the property that checks if squares lie within the region
def in_region_square (x y : ℤ) : Prop :=
  region_bounds x y ∧ region_bounds (x + 1) y ∧ region_bounds x (y + 1) ∧ region_bounds (x + 1) (y + 1)

-- Define the property that checks if right-angled isosceles triangles lie within the region
def in_region_triangle (x y : ℤ) : Prop :=
  region_bounds x y ∧ region_bounds (x + 1) y ∧ region_bounds x (y + 1)

-- Statement of the theorem to be proven
theorem number_of_shapes : 
  (∑ x in Finset.range 6, ∑ y in Finset.range (3 * x + 2), (if in_region_square x y then 1 else 0) + (if in_region_triangle x y then 1 else 0)) = 120 :=
sorry

end number_of_shapes_l815_815833


namespace power_function_value_l815_815409

noncomputable def f (x : ℝ) := x^(1/2)

theorem power_function_value :
  f(1/2) = (Real.sqrt 2) / 2 → f 4 = 2 :=
by {
  intro h,
  have : f x = x^(1/2) := by rfl,
  rw this at *,
  sorry
}

end power_function_value_l815_815409


namespace mary_initial_borrowed_books_l815_815911

-- We first define the initial number of books B.
variable (B : ℕ)

-- Next, we encode the conditions into a final condition of having 12 books.
def final_books (B : ℕ) : ℕ := (B - 3 + 5) - 2 + 7

-- The proof problem is to show that B must be 5.
theorem mary_initial_borrowed_books (B : ℕ) (h : final_books B = 12) : B = 5 :=
by
  sorry

end mary_initial_borrowed_books_l815_815911


namespace change_in_total_berries_l815_815622

theorem change_in_total_berries (B S : ℕ) (hB : B = 20) (hS : S + B = 50) : (S - B) = 10 := by
  sorry

end change_in_total_berries_l815_815622


namespace larger_number_is_23_l815_815989

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815989


namespace a_2016_is_neg2_l815_815414

def sequence (n : ℕ) : ℤ :=
  if n = 1 then 4
  else if n = 2 then 6
  else sequence (n-1) - sequence (n-2)

theorem a_2016_is_neg2 : sequence 2016 = -2 :=
  sorry

end a_2016_is_neg2_l815_815414


namespace events_A_and_B_independent_l815_815542

-- Definitions of the events and sample space
def sample_space : Set (Set Bool) := 
  {{true, true, true}, {true, true, false}, {true, false, true}, {false, true, true},
   {true, false, false}, {false, true, false}, {false, false, true}, {false, false, false}}

-- Event that there are both boys (true) and girls (false)
def event_A : Set (Set Bool) :=
  {{true, true, false}, {true, false, true}, {false, true, true}, 
   {true, false, false}, {false, true, false}, {false, false, true}}

-- Event that there is at most one boy
def event_B : Set (Set Bool) :=
  {{true, false, false}, {false, true, false}, {false, false, true}, {false, false, false}}

-- Probability of an event given uniform probability distribution on the sample space
def probability (e : Set (Set Bool)) : ℝ :=
  (e.card : ℝ) / (sample_space.card : ℝ)

-- Independence of two events
def independent (e1 e2 : Set (Set Bool)) : Prop :=
  probability (e1 ∩ e2) = probability e1 * probability e2

theorem events_A_and_B_independent : independent event_A event_B := by
  sorry

end events_A_and_B_independent_l815_815542


namespace min_distance_between_intersections_range_of_a_l815_815022

variable {a : ℝ}

/-- Given the function f(x) = x^2 - 2ax - 2(a + 1), 
1. Prove that the graph of function f(x) always intersects the x-axis at two distinct points.
2. For all x in the interval (-1, ∞), prove that f(x) + 3 ≥ 0 implies a ≤ sqrt 2 - 1. --/

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 2 * (a + 1)

theorem min_distance_between_intersections (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, (f x₁ a = 0) ∧ (f x₂ a = 0) ∧ (x₁ ≠ x₂) ∧ (dist x₁ x₂ = 2) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x → f x a + 3 ≥ 0) → a ≤ Real.sqrt 2 - 1 := sorry

end min_distance_between_intersections_range_of_a_l815_815022


namespace doubled_cylinder_volume_l815_815271

theorem doubled_cylinder_volume (r h : ℝ) (V : ℝ) (original_volume : V = π * r^2 * h) (V' : ℝ) : (2 * 2 * π * r^2 * h = 40) := 
by 
  have original_volume := 5
  sorry

end doubled_cylinder_volume_l815_815271


namespace axis_of_symmetry_shifted_sine_function_l815_815016

open Real

noncomputable def axisOfSymmetry (k : ℤ) : ℝ := k * π / 2 + π / 6

theorem axis_of_symmetry_shifted_sine_function (x : ℝ) (k : ℤ) :
  ∃ k : ℤ, x = axisOfSymmetry k := by
sorry

end axis_of_symmetry_shifted_sine_function_l815_815016


namespace midpoint_trajectory_l815_815363

def Point (α : Type) := × α

variables (P P' : Point ℝ)

def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

def midpoint (x0 y0 : ℝ) : Point ℝ :=
  (x0, y0 / 2)

def trajectory_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

theorem midpoint_trajectory (x0 y0 : ℝ):
  circle_equation x0 y0 →
  midpoint x0 y0 = (x0, y0 / 2) →
  trajectory_equation x0 (y0 / 2) :=
begin
  intros h1 h2,
  replace h1 : circle_equation x0 y0 := h1,
  rw [midpoint, trajectory_equation],
  sorry
end

end midpoint_trajectory_l815_815363


namespace police_arrangements_l815_815326

theorem police_arrangements (officers : Fin 5) (A B : Fin 5) (intersections : Fin 3) :
  A ≠ B →
  (∃ arrangement : Fin 5 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ off : Fin 5, arrangement off = i ∧ arrangement off = j) ∧
    arrangement A = arrangement B) →
  ∃ arrangements_count : Nat, arrangements_count = 36 :=
by
  sorry

end police_arrangements_l815_815326


namespace regular_tetrahedron_surface_area_l815_815602

open_locale big_operators

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ := (sqrt 3 / 4) * a ^ 2

theorem regular_tetrahedron_surface_area (a : ℝ) :
  let face_area := equilateral_triangle_area a in
  4 * face_area = sqrt 3 * a ^ 2 :=
by
  sorry

end regular_tetrahedron_surface_area_l815_815602


namespace part1_part2_l815_815497

variable {A B C a b c : ℝ}

-- Condition: Triangle with sides and corresponding angles
def in_triangle (A B C a b c : ℝ) : Prop :=
  a = c - b ∧    -- Given equation c - b = 2b cos A
  c - b = 2 * b * Real.cos A

-- Proof Problem (1)
theorem part1 (h : in_triangle A B C a b c) : A = 2 * B :=
sorry

variable {sin cos : ℝ → ℝ}

-- Additional Conditions for Proof Problem (2)
def cos_condition (B : ℝ) := Real.cos B = 3 / 4
def side_condition (c : ℝ) := c = 5

-- Proof Problem (2)
theorem part2 (h : in_triangle A B C a b c) (h_cos : cos_condition B) (h_side : side_condition c) :
  let area := 1 / 2 * b * c * Real.sin A in
  area = 15 / 4 * Real.sqrt 7 :=
sorry

end part1_part2_l815_815497


namespace tan_double_angle_l815_815044

theorem tan_double_angle (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi)
  (h3 : Real.cos α + Real.sin α = -1 / 5) : Real.tan (2 * α) = -24 / 7 :=
by
  sorry

end tan_double_angle_l815_815044


namespace determine_incorrect_propositions_l815_815732

noncomputable def prop1_false : Prop :=
  ∀ {a : ℕ → ℕ} {m n p q : ℕ}, ( ∀ k, a k = a 0 ) ∧ a m + a n = a p + a q → ¬(m + n = p + q)

noncomputable def prop2_false : Prop :=
  ∀ {a : ℕ → ℝ} {n : ℕ}, ( ∃ r : ℝ, r ≠ 0 ∧ ∀ k, a k = a 0 * r^k ) ∧ 
  (let sn := (finset.range (n + 1)).sum (λ i, a i),
  s2n := (finset.range (2*n + 1)).sum (λ i, a i),
  s3n := (finset.range (3*n + 1)).sum (λ i, a i) in
  sn ≠ 0 ∧ ¬(∃ r : ℝ, ∀ k, s2n - sn = sn * r^(k-1) ∧ s3n - s2n = sn * r^(k-1)))

noncomputable def prop3_true : Prop :=
  ∀ {A B C : Type} [decidable_eq C] {a b c : ℝ}, 
  ∀ (R : ℝ), (a < b) → (sin A < sin B)

noncomputable def prop4_false : Prop :=
  ∀ {A B C : Type} [decidable_eq C] {a b c : ℝ},
  a * cos A = b * cos B → ¬(A = B ∨ A + B = π / 2)

noncomputable def prop5_true : Prop :=
  ∀ {a : ℕ → ℝ}, ∀ {q : ℝ}, (q > 0) → (a 4 = 4) ∧ (a 12 = 16) ∧ (q = (a 12 / a 4)^(1/8)) → a 8 = 8

theorem determine_incorrect_propositions : 
  prop1_false ∧ prop2_false ∧ prop3_true ∧ prop4_false ∧ prop5_true :=
by 
  split; 
  try { exact sorry };
  split; 
  try { exact sorry };
  exact sorry

end determine_incorrect_propositions_l815_815732


namespace find_lambda_values_l815_815035

variable (λ : ℝ)
variable (a : ℝ × ℝ × ℝ := (1, λ, 1)) (b : ℝ × ℝ × ℝ := (2, -1, 1))

-- Define the dot product of two 3D vectors
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Define the magnitude of a 3D vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Given cosine of the angle condition
def cosine_angle_condition : Prop :=
  (dot_product a b) / (magnitude a * magnitude b) = 1 / 6

-- Prove that the lambda values satisfying the cosine condition are 2 or 26/5
theorem find_lambda_values (h : cosine_angle_condition) : λ = 2 ∨ λ = 26 / 5 := by {
  sorry
}

end find_lambda_values_l815_815035


namespace inscribed_sphere_radius_l815_815187

/-- 
  Define the terms used in the problem.
-/
def side_length_of_base (a : ℝ) : ℝ := a
def lateral_edge_angle_with_base_plane : ℝ := 45

/-- 
  The proof task is to show that, given the above conditions, the radius r of the inscribed sphere 
  in a regular triangular pyramid (tetrahedron) can be calculated as follows.
-/
theorem inscribed_sphere_radius (a : ℝ) :
  let r := (a * Real.sqrt 3 * (Real.sqrt 5 - 1)) / 12 
  in r = (a * Real.sqrt 3 * (Real.sqrt 5 - 1)) / 12 := by
  -- Proof goes here
  sorry

end inscribed_sphere_radius_l815_815187


namespace intersect_complement_l815_815029

open Set

variable (U : Set ℤ) (A B : Set ℤ)
variable [DecidablePred (· ∈ U)]
variable [DecidablePred (· ∈ A)]
variable [DecidablePred (· ∈ B)]

noncomputable def U_def := {-1, 0, 1, 2, 3}
noncomputable def A_def := {2, 3}
noncomputable def B_def := {0, 1}

theorem intersect_complement :
  (U_def \ A_def) ∩ B_def = {0, 1} := by
  sorry

end intersect_complement_l815_815029


namespace arun_profit_percentage_l815_815706

/-- Arun purchased 30 kg of wheat at 11.50 Rs/kg and 20 kg of wheat at 14.25 Rs/kg.
    He mixed the two and sold the mixture at 15.75 Rs/kg. What is his profit percentage? --/
theorem arun_profit_percentage :
  ∃ profit_percentage, 
    let cost1 := 30 * 11.50,
    cost2 := 20 * 14.25,
    total_cost := cost1 + cost2,
    total_weight := 30 + 20,
    selling_price_per_kg := 15.75,
    total_selling_price := total_weight * selling_price_per_kg,
    profit := total_selling_price - total_cost,
    profit_percentage := (profit / total_cost) * 100
    in profit_percentage = 25 := by
  sorry

end arun_profit_percentage_l815_815706


namespace equation_of_trajectory_collinearity_of_points_l815_815375

variables {F : ℝ × ℝ} {x y : ℝ}
def moving_point (E : ℝ × ℝ) := 
  ∃ F : ℝ × ℝ, F = (0, 1) ∧ 
  let P := (E.1 / 2, (E.2 + 1) / 2) in 
  |P.2| = (real.sqrt (E.1 ^ 2 + (E.2 - 1) ^ 2) / 2)

theorem equation_of_trajectory (E : ℝ × ℝ) (h : moving_point E) : E.1 ^ 2 = 4 * E.2 :=
by sorry

theorem collinearity_of_points (A B : ℝ × ℝ) 
  (hA : moving_point A) (hB : moving_point B)
  (h_perpendicular : (A.2 / 2 * A.1) * (B.2 / 2 * B.1) = -1) :
  ∃ k : ℝ, k * (A.2 - 1) = A.1 ∧ k * (B.2 - 1) = B.1 :=
by sorry

end equation_of_trajectory_collinearity_of_points_l815_815375


namespace at_least_seven_pencils_of_one_color_l815_815197

theorem at_least_seven_pencils_of_one_color :
  ∃ c : Color, 7 ≤ count c in box :=
begin
  -- Definitions and given conditions
  def Box : Type := fin 25
  def Color : Type := finitely_many_colors

  def exists_two_of_same_color (l : list Color) : Prop :=
    ∃ c : Color, list.count c l ≥ 2

  -- Given condition: any five pencils contain at least two pencils of the same color
  axiom condition : ∀ (subset : finset Box), subset.card = 5 → exists_two_of_same_color (subset.to_list)

  -- The theorem stating there is a color occurring at least 7 times
  sorry
end

end at_least_seven_pencils_of_one_color_l815_815197


namespace AE_EB_ratio_l815_815656

theorem AE_EB_ratio
  (AB CD : ℝ)
  (A B C D E F : Point)
  (is_isosceles_trapezoid : IsoscelesTrapezoid A B C D)
  (AB_parallel_CD : AB ∥ CD)
  (AB_eq_2_CD : AB = 2 * CD)
  (angle_A_eq_60 : ∠A = 60)
  (on_base_AB : onBase AB E)
  (FE_eq_FB : distance F E = distance F B)
  (FB_eq_AC : distance F B = distance A C)
  (FA_eq_AB : distance F A = distance A B)
  : (distance A E / distance E B = 1 / 3) := sorry

end AE_EB_ratio_l815_815656


namespace tan_monotonic_increasing_interval_l815_815961

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | 2 * k * Real.pi - (5 * Real.pi) / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3 }

theorem tan_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (y = Real.tan ((x / 2) + (Real.pi / 3))) → 
           x ∈ monotonic_increasing_interval k :=
sorry

end tan_monotonic_increasing_interval_l815_815961


namespace john_tran_probability_2_9_l815_815193

def johnArrivalProbability (train_start train_end john_min john_max: ℕ) : ℚ := 
  let overlap_area := ((train_end - train_start - 15) * 15) / 2 
  let total_area := (john_max - john_min) * (train_end - train_start)
  overlap_area / total_area

theorem john_tran_probability_2_9 :
  johnArrivalProbability 30 90 0 90 = 2 / 9 := by
  sorry

end john_tran_probability_2_9_l815_815193


namespace sufficient_but_not_necessary_condition_l815_815815

-- Define the problem conditions
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

noncomputable def z (a : ℝ) : ℂ :=
  (a^2 - 1 : ℝ) + (a - 2 : ℝ) * complex.I

-- Prove that "a=1" is a sufficient but not necessary condition for "z" to be purely imaginary
theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 → is_purely_imaginary (z a)) ∧
  ¬(∀ a, is_purely_imaginary (z a) → a = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l815_815815


namespace circle_radius_m_eq_l815_815840

theorem circle_radius_m_eq (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 2 * x + 2 * y + m = 0) ∧ (√9 = 3) → m = -7 :=
by
  sorry

end circle_radius_m_eq_l815_815840


namespace baron_munchausen_theorem_l815_815306

theorem baron_munchausen_theorem (n : ℕ) (roots : Fin n → ℕ)
  (a : ℕ) (b : ℕ)
  (h_poly : ∀ x, (∑ i, x ^ i) = x^n - a * x^(n-1) + b * x^(n-2) + ... ) :
  (∑ i, roots i) = a ∧ (∑ (i j : Fin n), i ≠ j → (roots i * roots j)) = b :=
sorry

end baron_munchausen_theorem_l815_815306


namespace parabola_standard_eq_l815_815189

theorem parabola_standard_eq (p p' : ℝ) (h₁ : p > 0) (h₂ : p' > 0) :
  (∀ (x y : ℝ), (x^2 = 2 * p * y ∨ y^2 = -2 * p' * x) → 
  (x = -2 ∧ y = 4 → (x^2 = y ∨ y^2 = -8 * x))) :=
by
  sorry

end parabola_standard_eq_l815_815189


namespace stuffed_animals_total_l815_815243

variable (x y z : ℕ)

theorem stuffed_animals_total :
  let initial := x
  let after_mom := initial + y
  let after_dad := z * after_mom
  let total := after_mom + after_dad
  total = (x + y) * (1 + z) := 
  by 
    let initial := x
    let after_mom := initial + y
    let after_dad := z * after_mom
    let total := after_mom + after_dad
    sorry

end stuffed_animals_total_l815_815243


namespace intersection_distance_eq_l815_815582

theorem intersection_distance_eq (p q : ℕ) (h1 : p = 88) (h2 : q = 9) :
  p - q = 79 :=
by
  sorry

end intersection_distance_eq_l815_815582


namespace general_term_an_sum_bn_Tn_l815_815368

-- Problem 1
theorem general_term_an (a_n S_n : ℕ → ℝ) (h_a1 : a_n 1 = 1/2)
  (h_cond : ∀ n, a_n n + S_n n = 1) :
  ∀ n, a_n n = 1 / 2^n :=
sorry

-- Problem 2
theorem sum_bn_Tn (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h_b_n_def : ∀ n, b_n n = n / 2^n)
  (h_T_n_def : ∀ n, T_n n = ∑ k in finset.range n, b_n (k + 1)) :
  ∀ n, T_n n = 2 - ((n + 2) / 2^n) :=
sorry

end general_term_an_sum_bn_Tn_l815_815368


namespace find_acute_angle_l815_815006

theorem find_acute_angle (α : ℝ) (h : ∃! x : ℝ, 3 * x^2 + 1 = 4 * sin α * x) : α = 60 :=
sorry

end find_acute_angle_l815_815006


namespace farmland_acres_l815_815128

theorem farmland_acres (x y : ℝ) 
  (h1 : x + y = 100) 
  (h2 : 300 * x + (500 / 7) * y = 10000) : 
  true :=
sorry

end farmland_acres_l815_815128


namespace complement_union_empty_l815_815194

open Set

variable (U M N : Set ℕ)

-- Define the specific sets as given in the problem
def U := {1, 2, 3, 4, 5}
def M := {1, 3, 4}
def N := {2, 4, 5}

-- Define the complement function in terms of the universal set
def C_U (A : Set ℕ) := U \ A

theorem complement_union_empty :
  C_U (M ∪ N) = ∅ :=
by
  -- Proof can be filled in here; this statement just establishes the problem.
  sorry

end complement_union_empty_l815_815194


namespace intersection_complement_l815_815827

open Set

theorem intersection_complement 
  (U : Set ℝ) (M : Set ℝ) (N : Set ℝ) 
  (hU : U = univ : Set ℝ) 
  (hM : M = { x : ℝ | x^2 - 4 * x - 5 > 0 }) 
  (hN : N = { x : ℝ | x ≥ 1}) : 
  M ∩ (U \ N) = { x : ℝ | x < -1 } :=
by 
  rw [hU, hM, hN]
  sorry

end intersection_complement_l815_815827


namespace equilateral_triangle_exists_isosceles_right_triangle_exists_l815_815718

theorem equilateral_triangle_exists (red_or_blue : ℝ → ℝ → bool) :
  ∃ (Δ : Triangle), 
    (Δ.is_equilateral ∧ 
    (Δ.side_length = 673 * Real.sqrt 3 ∨ Δ.side_length = 2019) ∧ 
    Δ.is_monochromatic red_or_blue) :=
sorry

theorem isosceles_right_triangle_exists (red_or_blue : ℝ → ℝ → bool) :
  ∃ (Δ : Triangle), 
    (Δ.is_isosceles_right ∧ 
    (Δ.side_length = 1010 * Real.sqrt 2 ∨ Δ.side_length = 2020) ∧ 
    Δ.is_monochromatic red_or_blue) :=
sorry

end equilateral_triangle_exists_isosceles_right_triangle_exists_l815_815718


namespace tangent_line_at_point_l815_815949

noncomputable def f : ℝ → ℝ := λ x => 2 * Real.log x + x^2 

def tangent_line_equation (x y : ℝ) : Prop :=
  4 * x - y - 3 = 0 

theorem tangent_line_at_point {x y : ℝ} (h : f 1 = 1) : 
  tangent_line_equation 1 1 ∧
  y = 4 * (x - 1) + 1 := 
sorry

end tangent_line_at_point_l815_815949


namespace pentagon_triangle_area_percentage_l815_815276

theorem pentagon_triangle_area_percentage (s : ℝ) :
  let h := s * (Real.sqrt 3 / 2)
  ∧ let area_square := (h) ^ 2
  ∧ let area_triangle := (s ^ 2 * Real.sqrt 3) / 4
  ∧ let area_pentagon := area_square + area_triangle
  √3 / (3 + √3) ≈ 0.191 :=
begin
  sorry
end

end pentagon_triangle_area_percentage_l815_815276


namespace f_f_minus1_is_minus1_l815_815388

variable {𝔽 : Type*} [LinearOrderedField 𝔽]

noncomputable def f (x : 𝔽) : 𝔽 :=
if x > 0 then 2^x - 1 else if x < 0 then - (2^(-x) - 1) else 0

theorem f_f_minus1_is_minus1 (x : 𝔽) (hf : ∀ x : 𝔽, f(-x) = -f(x)) (hx_pos : x > 0):
  f (f (-1)) = -1 :=
by {
  have f1_eq_1 : f 1 = 1, from
    by {
      have h1 : 1 > 0 := zero_lt_one,
      exact if_pos h1
    },
  have f_minus1_eq_minus_f1 : f (-1) = -1, from
    by {
      have h_neg1 : -1 < 0 := neg_lt_zero.mpr zero_lt_one,
      rw [hf 1, f1_eq_1],
      exact if_neg h_neg1
    },
  rw f_minus1_eq_minus_f1,
  exact -1
}

end f_f_minus1_is_minus1_l815_815388


namespace area_of_square_on_RS_l815_815209

theorem area_of_square_on_RS (PQ QR PS PS_square PQ_square QR_square : ℝ)
  (hPQ : PQ_square = 25) (hQR : QR_square = 49) (hPS : PS_square = 64)
  (hPQ_eq : PQ_square = PQ^2) (hQR_eq : QR_square = QR^2) (hPS_eq : PS_square = PS^2)
  : ∃ RS_square : ℝ, RS_square = 138 := by
  let PR_square := PQ^2 + QR^2
  let RS_square := PR_square + PS^2
  use RS_square
  sorry

end area_of_square_on_RS_l815_815209


namespace water_usage_l815_815480

noncomputable def litres_per_household_per_month (total_litres : ℕ) (number_of_households : ℕ) : ℕ :=
  total_litres / number_of_households

theorem water_usage : litres_per_household_per_month 2000 10 = 200 :=
by
  sorry

end water_usage_l815_815480


namespace perpendicular_vector_condition_l815_815000

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

noncomputable def dot_product : ℝ := ⟪a, b⟫

theorem perpendicular_vector_condition 
  (h_unit_a : ∥a∥ = 1) 
  (h_unit_b : ∥b∥ = 1) 
  (h_angle : real.angle_of _ a b = real.pi / 3) :
  ⟪2 • a - b, b⟫ = 0 := 
begin
  sorry
end

end perpendicular_vector_condition_l815_815000


namespace trapezoid_area_l815_815485

-- Definitions corresponding to the conditions
def smaller_base := 2
def adjacent_angle := 135
def angle_between_diagonals := 150

-- Statement of the problem
theorem trapezoid_area :
  ∀ (BC : ℝ) (angle_ABC angle_DCB angle_BOC : ℝ),
    BC = smaller_base →
    angle_ABC = adjacent_angle →
    angle_DCB = adjacent_angle →
    angle_BOC = angle_between_diagonals →
    let area_of_trapezoid := 2 in
    area_of_trapezoid = 2 :=
by
  intros BC angle_ABC angle_DCB angle_BOC hBC hABC hDCB hBOC
  let area_of_trapezoid := 2
  sorry

end trapezoid_area_l815_815485


namespace necessary_and_sufficient_condition_for_extreme_value_l815_815175

-- Defining the function f(x) = ax^3 + x + 1
def f (a x : ℝ) : ℝ := a * x^3 + x + 1

-- Defining the condition for f to have an extreme value
def has_extreme_value (a : ℝ) : Prop := ∃ x : ℝ, deriv (f a) x = 0

-- Stating the problem
theorem necessary_and_sufficient_condition_for_extreme_value (a : ℝ) :
  has_extreme_value a ↔ a < 0 := by
  sorry

end necessary_and_sufficient_condition_for_extreme_value_l815_815175


namespace count_integer_solutions_l815_815422

theorem count_integer_solutions :
  {x : ℤ | (x - 2)^2 ≤ 4}.card = 5 :=
by
  sorry

end count_integer_solutions_l815_815422


namespace num_integers_satisfying_inequality_l815_815429

theorem num_integers_satisfying_inequality : 
  {x : ℤ | (x - 2)^2 ≤ 4}.finite.to_finset.card = 5 :=
by {
  sorry
}

end num_integers_satisfying_inequality_l815_815429


namespace dot_product_b1_b2_l815_815829

open Real

variables (e1 e2 : EuclideanSpace ℝ (Fin 2))

-- Conditions
axiom unit_e1 : ‖e1‖ = 1
axiom unit_e2 : ‖e2‖ = 1
axiom angle_e1_e2 : ∀ (θ : ℝ), θ = π / 3 → e1 ⬝ e2 = cos θ

def b1 : EuclideanSpace ℝ (Fin 2) := 2 • e1 - 4 • e2
def b2 : EuclideanSpace ℝ (Fin 2) := 3 • e1 + 4 • e2

theorem dot_product_b1_b2 : b1 ⬝ b2 = -12 := by
  -- Skipping the proof
  sorry

end dot_product_b1_b2_l815_815829


namespace sum_Q_inv_eq_l815_815412

noncomputable def sum_Q_inv (n : ℕ) : ℝ :=
  let nums := {i : ℕ | ∃ k : ℕ, k < n ∧ i = 2^k }
  let all_perms := equiv.Perm.ofFinset (nums.toFinset)
  Finset.sum all_perms.toFinset (λ σ =>
    (Finset.range n).prod (λ k => ∑(i : ℕ) in (Finset.range (k+1)).map (σ.to_fun), (i : ℝ))⁻¹
  )

theorem sum_Q_inv_eq (n : ℕ) : sum_Q_inv n = 2^(-(n * (n-1)) / 2) :=
  sorry

end sum_Q_inv_eq_l815_815412


namespace sin_cos_rational_of_isosceles_triangle_l815_815372

theorem sin_cos_rational_of_isosceles_triangle (BC AD : ℤ) 
  (h1 : ∃ A B C D : ℝ, is_isosceles_triangle A B C D BC AD) :
  ∃ (sinA cosA : ℚ), sin (A : ℝ) = sinA ∧ cos (A : ℝ) = cosA := 
by sorry

end sin_cos_rational_of_isosceles_triangle_l815_815372


namespace zack_group_size_l815_815651

theorem zack_group_size (total_students : Nat) (groups : Nat) (group_size : Nat)
  (H1 : total_students = 70)
  (H2 : groups = 7)
  (H3 : total_students = group_size * groups) :
  group_size = 10 := by
  sorry

end zack_group_size_l815_815651


namespace minimum_k_l815_815601

theorem minimum_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  a 1 = (1/2) ∧ (∀ n, 2 * a (n + 1) + S n = 0) ∧ (∀ n, S n ≤ k) → k = (1/2) :=
sorry

end minimum_k_l815_815601


namespace last_two_digits_of_sum_of_first_15_factorials_is_13_l815_815222

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (k+1) => (k+1) * factorial k

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ :=
  (Nat.fold (fun acc x => acc + factorial x) 0 (List.range n))

theorem last_two_digits_of_sum_of_first_15_factorials_is_13 :
  last_two_digits (sum_of_factorials 15) = 13 :=
by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_is_13_l815_815222


namespace pascal_identity_l815_815927

noncomputable def pascal_sequence (n : ℕ) (i : ℕ) : ℕ :=
  Nat.choose n i

theorem pascal_identity (n : ℕ) (h : n = 3004) :
  2 * ∑ i in Finset.range (n + 1), (pascal_sequence n i : ℝ) / (pascal_sequence (n + 1) i) -
  ∑ i in Finset.range n, (pascal_sequence (n - 1) i : ℝ) / (pascal_sequence n i) = 1503.5 :=
by
  rw h
  sorry

end pascal_identity_l815_815927


namespace find_x_l815_815469

theorem find_x (x : ℝ) : (1 + (1 / (1 + x)) = 2 * (1 / (1 + x))) → x = 0 :=
by
  intro h
  sorry

end find_x_l815_815469


namespace sample_mean_is_an_estimate_of_population_mean_l815_815157

-- Define the sample mean
def sample_mean : Type := ℝ
def population_mean : Type := ℝ

-- Given conditions
variable (x̄ : sample_mean)
variable (μ : population_mean)

-- Statement: The sample mean (x̄) is an estimate of the population mean (μ)
theorem sample_mean_is_an_estimate_of_population_mean : x̄ ≈ μ :=
sorry

end sample_mean_is_an_estimate_of_population_mean_l815_815157


namespace volume_tetrahedron_KLMN_l815_815551

theorem volume_tetrahedron_KLMN :
  (∀ S A B C K L M N : Type, -- The vertices of tetrahedra and their inscribed circle centers
   (AB SC AC SB BC SA : ℝ) -- Side lengths of tetrahedron SABC
   (K_center : ∀ P Q R : Type, K = incenter S A B) -- Inscribed circle centers
   (a_5 : AB = 5)
   (a_5' : SC = 5)
   (b_7 : AC = 7)
   (b_7' : SB = 7)
   (c_8 : BC = 8)
   (c_8' : SA = 8)) → 
  volume K L M N = 0.66 := 
sorry

end volume_tetrahedron_KLMN_l815_815551


namespace exists_monochromatic_equilateral_triangle_l815_815156

def color := ℕ -- Assume two colors represented by 0 and 1

def is_colored (plane : ℝ × ℝ → color) : Prop := 
  ∀ x, plane x ∈ {0, 1}

theorem exists_monochromatic_equilateral_triangle
  (plane : ℝ × ℝ → color)
  (h : is_colored plane) :
  ∃ (A B C : ℝ × ℝ), 
    (dist A B = 1 ∨ dist A B = sqrt 3) ∧ 
    (dist B C = 1 ∨ dist B C = sqrt 3) ∧ 
    (dist C A = 1 ∨ dist C A = sqrt 3) ∧ 
    plane A = plane B ∧ 
    plane B = plane C := 
by sorry

end exists_monochromatic_equilateral_triangle_l815_815156


namespace problem_statement_l815_815817

noncomputable def f (x : ℝ) : ℝ := (Real.cos x + 1 - Real.sin x ^ 2) * Real.tan (x / 2)

theorem problem_statement : 
  (∀ x ∈ Icc (-π / 4) (π / 4), f.deriv x > 0) ∧ (∀ x, f (x + π) = f x) :=
  sorry

end problem_statement_l815_815817


namespace range_of_a_l815_815063

open Real

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, abs(x - a) + abs(x - 1) ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l815_815063


namespace total_people_in_house_l815_815641

-- Define the number of people in various locations based on the given conditions.
def charlie_and_susan := 2
def sarah_and_friends := 5
def people_in_bedroom := charlie_and_susan + sarah_and_friends
def people_in_living_room := 8

-- Prove the total number of people in the house is 14.
theorem total_people_in_house : people_in_bedroom + people_in_living_room = 14 := by
  -- Here we can use Lean's proof system, but we skip with 'sorry'
  sorry

end total_people_in_house_l815_815641


namespace sin_2B_sin_A_sin_C_eq_neg_7_over_8_l815_815475

theorem sin_2B_sin_A_sin_C_eq_neg_7_over_8
    (A B C : ℝ)
    (a b c : ℝ)
    (h1 : (2 * a + c) * Real.cos B + b * Real.cos C = 0)
    (h2 : 1/2 * a * c * Real.sin B = 15 * Real.sqrt 3)
    (h3 : a + b + c = 30) :
    (2 * Real.sin B * Real.cos B) / (Real.sin A + Real.sin C) = -7/8 := 
sorry

end sin_2B_sin_A_sin_C_eq_neg_7_over_8_l815_815475


namespace intersection_point_correct_l815_815172

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Line :=
(p1 : Point3D) (p2 : Point3D)

structure Plane :=
(trace : Line) (point : Point3D)

noncomputable def intersection_point (l : Line) (β : Plane) : Point3D := sorry

theorem intersection_point_correct (l : Line) (β : Plane) (P : Point3D) :
  let res := intersection_point l β
  res = P :=
sorry

end intersection_point_correct_l815_815172


namespace total_treats_value_l815_815727

noncomputable def hotel_per_night := 4000
noncomputable def nights := 2
noncomputable def car_value := 30000
noncomputable def house_value := 4 * car_value
noncomputable def total_value := hotel_per_night * nights + car_value + house_value

theorem total_treats_value : total_value = 158000 :=
by
  sorry

end total_treats_value_l815_815727


namespace least_possible_value_of_x_minus_y_plus_z_l815_815253

theorem least_possible_value_of_x_minus_y_plus_z : 
  ∃ (x y z : ℕ), 3 * x = 4 * y ∧ 4 * y = 7 * z ∧ x - y + z = 19 :=
by
  sorry

end least_possible_value_of_x_minus_y_plus_z_l815_815253


namespace problem_conditions_l815_815797

theorem problem_conditions (a1 : ℤ) (d : ℚ) (n : ℕ) :
  (a1 + 4 * d = -3) ∧ (5 * (2 * a1 + 4 * d) = 0) → 
  (∀ n, a1 + (n - 1) * d = (3 * (3 - n)) / 2) ∧ 
  (∀ n, (a1 + (n - 1) * d) * (n * (2 * a1 + (n - 1) * d)) / 2 < 0 → n = 4) :=
by
  sorry  -- proof details excluded

end problem_conditions_l815_815797


namespace happiness_80th_percentile_l815_815158

def happiness_indices : List ℕ := [7, 3, 5, 6, 7, 4, 8, 9, 5, 10]

def percentile (l : List ℕ) (p : ℕ) : ℚ := 
  let sorted := l.qsort (≤)
  let n := l.length
  let k := (n * p) / 100
  if k < n then (sorted.get! k + sorted.get! (k + 1)) / 2 else sorted.get! k

theorem happiness_80th_percentile :
  percentile happiness_indices 80 = 8.5 := by
  sorry

end happiness_80th_percentile_l815_815158


namespace integers_with_abs_less_than_four_l815_815093

theorem integers_with_abs_less_than_four :
  {x : ℤ | |x| < 4} = {-3, -2, -1, 0, 1, 2, 3} :=
sorry

end integers_with_abs_less_than_four_l815_815093


namespace calculate_minus_one_minus_two_l815_815715

theorem calculate_minus_one_minus_two : -1 - 2 = -3 := by
  sorry

end calculate_minus_one_minus_two_l815_815715


namespace problem_l815_815361

theorem problem (x : ℝ) (y : ℝ) (hp : y > 0)
  (hA : {x^2 + x + 1, -x, -x - 1} = {-y, -y/2, y + 1}) :
  x^2 + y^2 = 5 := 
by
  sorry

end problem_l815_815361


namespace division_of_land_l815_815270

/-- A rectangular plot of land has dimensions 25 m × 36 m, and it needs to be divided into 
  three rectangular sections each of area 300 m². We need to prove:
  1. There are 4 ways to divide the land into such sections.
  2. The configuration with the minimum total internal fence length involves one plot 
     of 25 m × 12 m and two plots of 12.5 m × 24 m with a total internal fence length of 49 m. -/
theorem division_of_land 
  (length width : ℕ) (area : ℕ) 
  (h_length : length = 25) 
  (h_width : width = 36) 
  (h_area : area = length * width) 
  (section_area : ℕ) (h_section_area : section_area = 300) :
  ∃ (ways : ℕ) (min_fence : ℕ),
  ways = 4 ∧ min_fence = 49 ∧
  (∃ l1 w1 l2 w2 l3 w3,
  l1 * w1 = section_area ∧ l2 * w2 = section_area ∧ l3 * w3 = section_area ∧
  ((l1 = 25 ∧ w1 = 12) ∧ (l2 = 12.5 ∧ w2 = 24) ∧ (l3 = 12.5 ∧ w3 = 24))) := sorry

end division_of_land_l815_815270


namespace circle_area_conversion_l815_815231

-- Define the given diameter
def diameter (d : ℝ) := d = 8

-- Define the radius calculation
def radius (r : ℝ) := r = 4

-- Define the formula for the area of the circle in square meters
def area_sq_m (A : ℝ) := A = 16 * Real.pi

-- Define the conversion factor from square meters to square centimeters
def conversion_factor := 10000

-- Define the expected area in square centimeters
def area_sq_cm (A : ℝ) := A = 160000 * Real.pi

-- The theorem to prove
theorem circle_area_conversion (d r A_cm : ℝ) (h1 : diameter d) (h2 : radius r) (h3 : area_sq_cm A_cm) :
  A_cm = 160000 * Real.pi :=
by
  sorry

end circle_area_conversion_l815_815231


namespace pure_imaginary_number_implies_x_eq_1_l815_815261

theorem pure_imaginary_number_implies_x_eq_1 (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x + 1 ≠ 0) : x = 1 :=
sorry

end pure_imaginary_number_implies_x_eq_1_l815_815261


namespace percentage_discount_l815_815547

def cost_per_ball : ℝ := 0.1
def number_of_balls : ℕ := 10000
def amount_paid : ℝ := 700

theorem percentage_discount : (number_of_balls * cost_per_ball - amount_paid) / (number_of_balls * cost_per_ball) * 100 = 30 :=
by
  sorry

end percentage_discount_l815_815547


namespace cubic_polynomial_zero_l815_815279

theorem cubic_polynomial_zero (r α β : ℤ) 
  (h_poly : ∀ x, (x - r) * (x^2 + (α : ℚ) * x + (β : ℚ)) = ∑ i in range 4, (coeff i) * x ^ i) 
  (h_sqrt : -α = 3 ∧ 4 * β - α^2 = 7) :
  ∃ x : ℂ, x = (3 + complex.i * (complex.sqrt 7)) / 2 := sorry

end cubic_polynomial_zero_l815_815279


namespace find_x_l815_815331

-- Given definitions and conditions
def fractional_part (x : ℝ) : ℝ := x - floor x
def integer_part (x : ℝ) : ℝ := floor x

-- Given the condition: x is nonzero, and \{x\}, \lfloor x \rfloor, and x form a geometric sequence
def is_geometric_sequence (x : ℝ) : Prop := 
  x ≠ 0 ∧ 
  (fractional_part x ≠ 0 ∧ 
  integer_part x / fractional_part x = x / integer_part x)

-- The statement to prove
theorem find_x (x : ℝ) (h : is_geometric_sequence x) : 
  x = (5 + Real.sqrt 5) / 4 :=
sorry

end find_x_l815_815331


namespace simplify_sqrt_expression_l815_815758

variable (x : ℝ)

theorem simplify_sqrt_expression : ∀ x : ℝ, sqrt (9 * x ^ 4 + 3 * x ^ 2) = sqrt 3 * abs x * sqrt (3 * x ^ 2 + 1) :=
by 
  sorry

end simplify_sqrt_expression_l815_815758


namespace orvin_max_balloons_l815_815139

theorem orvin_max_balloons : 
  ∃ n : ℤ, 
  let regular_price := 1 in
  let money := 40 in
  let promo_price := regular_price / 2 in
  let full_price_group := 4 * regular_price + promo_price in
  let groups_of_five := money / full_price_group in
  let remainder_money := money - groups_of_five * full_price_group in
  let additional_balloons := remainder_money / regular_price in
  n = groups_of_five * 5 + additional_balloons ∧ 
  n = 44 := 
by
  sorry

end orvin_max_balloons_l815_815139


namespace optimal_distribution_l815_815075

/-- The total number of points is 1989. Every group formed must have a different number of points.
    This theorem states that the optimal distribution of points into 30 groups that maximizes the
    total number of triangles formed by choosing one point from each of any three different groups
    is approximately around the distribution provided. -/
theorem optimal_distribution (n : ℕ → ℕ) (h_sum : (Finset.range 30).sum n = 1989)
  (h_unique : ∀ i j, i ≠ j → n i ≠ n j) :
  let list_ni := [51, 52, 53, ..., 55, 56, 58, 59, ..., 81] in
  is_optimal_distribution n list_ni :=
sorry

/-- Predicate to check whether a given distribution is optimal. -/
def is_optimal_distribution (n : ℕ → ℕ) (list_ni : List ℕ) : Prop :=
  ∃ (perm : List ℕ), perm.permutations list_ni ∧ ∀ k, n k = perm k

end optimal_distribution_l815_815075


namespace domain_of_f_log2_x_l815_815470

theorem domain_of_f_log2_x (f : ℝ → ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (1/2 : ℝ) 2 → f x = f x) →
  ∀ x : ℝ, x ∈ set.Icc (sqrt 2) 4 → f (log 2 x) = f (log 2 x) :=
by
  intros h x hx
  sorry

end domain_of_f_log2_x_l815_815470


namespace sum_difference_even_integers_l815_815055

theorem sum_difference_even_integers : 
  let i := (112 / 2) * (2 + 224) 
  let k := (37 / 2) * (8 + 80)
in i - k = 11028 :=
by 
  let i := (112 / 2) * (2 + 224)
  let k := (37 / 2) * (8 + 80)
  sorry

end sum_difference_even_integers_l815_815055


namespace count_integer_solutions_l815_815420

theorem count_integer_solutions :
  {x : ℤ | (x - 2)^2 ≤ 4}.card = 5 :=
by
  sorry

end count_integer_solutions_l815_815420


namespace ratio_of_areas_l815_815064

variable (X Y Z W : Type) [Point X] [Point Y] [Point Z] [Point W]

-- Definition of the segments
variable (YW WZ : ℝ)
variable (W_on_YZ : Point_on_Side W Y Z)

-- Specific lengths given in the problem
variable (YW_value : YW = 8) (WZ_value : WZ = 12)

-- The goal to prove the ratio of areas is 2:3
theorem ratio_of_areas (hYW : YW = 8) (hWZ : WZ = 12) (hW : Point_on_Side W Y Z) :
  (area_of_triangle X Y W) / (area_of_triangle X W Z) = 2 / 3 := 
sorry

end ratio_of_areas_l815_815064


namespace sin_C_in_right_triangle_l815_815078

theorem sin_C_in_right_triangle (A B C : ℝ)
  (h₁ : A = 90) (h₂ : sin B = 3/5) :
  sin C = 4/5 :=
sorry

end sin_C_in_right_triangle_l815_815078


namespace larger_number_is_23_l815_815990

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815990


namespace negation_proposition_l815_815178

theorem negation_proposition {x : ℝ} : ¬ (x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := sorry

end negation_proposition_l815_815178


namespace shift_g_left_by_2_l815_815023

def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x < -1 then x + 3
  else if -1 ≤ x ∧ x ≤ 2 then -x^2 + 4
  else if 2 < x ∧ x ≤ 4 then x - 2
  else 0

theorem shift_g_left_by_2 (x : ℝ) : 
  g(x + 2) = 
    if -6 ≤ x ∧ x < -3 then x + 5
    else if -3 ≤ x ∧ x ≤ 0 then -x^2 - 4*x
    else if 0 < x ∧ x ≤ 2 then x
    else 0 := sorry

end shift_g_left_by_2_l815_815023


namespace exists_h_l815_815202

-- Define the type of a graph and its average degree
variable (G : Type) [graph G]

-- Define what it means for a graph to have a topological minor
variable (H : Type) [graph H]

-- State the main theorem
theorem exists_h (h : ℕ → ℕ := λ r, 2 ^ nat.choose r 2) :
  ∀ r : ℕ, ∀ G : Type, [graph G] → (average_degree G ≥ h r) → (∃ H : Type, [graph H] ∧ is_topological_minor H G ∧ H = K r) :=
by
  sorry

end exists_h_l815_815202


namespace minimum_k_value_l815_815491

theorem minimum_k_value {k : ℝ} (P Q : ℝ × ℝ)
  (hP : P.2 = k * (P.1 - 3 * real.sqrt 3))
  (hQ : Q.1^2 + (Q.2 - 1)^2 = 1)
  (hOP_OQ : P = (3 * Q.1, 3 * Q.2)) :
  k = -real.sqrt 3 :=
by
  sorry

end minimum_k_value_l815_815491


namespace green_pairs_count_l815_815857

theorem green_pairs_count :
  ∀ (red_students green_students total_students total_pairs red_pairs : ℕ),
    red_students = 63 →
    green_students = 69 →
    total_students = 132 →
    total_pairs = 66 →
    red_pairs = 26 →
    (red_students + green_students = total_students) →
    (total_pairs = total_students / 2) →
    (2 * red_pairs ≤ red_students) →
    (green_students - (red_students - 2 * red_pairs) >= 0) →
    let mixed_pairs := red_students - 2 * red_pairs,
        green_pairs := green_students - mixed_pairs,
        green_pairs_count := green_pairs / 2
    in 
    green_pairs_count = 29 :=
by
  intros
  sorry

end green_pairs_count_l815_815857


namespace larger_number_is_23_l815_815976

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815976


namespace baron_theorem_correct_l815_815304

theorem baron_theorem_correct (n a b : ℕ) (P : Polynomial ℕ)
  (roots : Fin n → ℕ)
  (hP : ∀ x, P.eval x = ∏ i, (x - roots i))
  (h_sum_root: a = (Finset.univ : Finset (Fin n)).sum (λ i, roots i))
  (h_prod_root: b = (Finset.univ : Finset (Fin n)).sum (λ(i_j : Fin n × Fin n), if i_j.1 < i_j.2 then roots i_j.1 * roots i_j.2 else 0)):
  b = (a * (a - 1)) / 2 := 
by
  sorry

end baron_theorem_correct_l815_815304


namespace minimal_curve_length_eq_triangle_division_l815_815340

theorem minimal_curve_length_eq_triangle_division (S : ℝ) (hS : 0 < S) :
  ∃ (L : ℝ), L = sqrt (π * S / 3) ∧
  ∀ (curve : ℝ) (area : ℝ), (curve > 0) → (area = S / 2) → curve ≥ sqrt (π * S / 3) :=
by {
  sorry
}

end minimal_curve_length_eq_triangle_division_l815_815340


namespace unique_geo_seq_l815_815416

theorem unique_geo_seq (m : ℝ) (q : ℝ) (m_pos : 0 < m) :
  (∀ n : ℕ, ∃ a_n b_n, 
    (a_1 = m) ∧ 
    (b_1 = 1 + a_1) ∧ 
    (b_2 = 2 + a_1 * q) ∧ 
    (b_3 = 3 + a_1 * q^2) ∧ 
    ((1 + a_1) * (3 + a_1 * q^2) = (2 + a_1 * q)^2)
  ) → (m = 1/3) :=
by sorry


end unique_geo_seq_l815_815416


namespace find_a_l815_815059

noncomputable def curve (x : ℝ) : ℝ :=
  x^(-1/2)

def point (a : ℝ) : ℝ × ℝ :=
  (a, curve a)

def tangent_slope (a : ℝ) : ℝ :=
  - (1 / 2) * a^(-3 / 2)

def tangent_line (a : ℝ) (x : ℝ) : ℝ :=
  curve a + tangent_slope a * (x - a)

def intersection_with_y_axis (a : ℝ) : ℝ :=
  tangent_line a 0

def intersection_with_x_axis (a : ℝ) : ℝ :=
  a - (curve a / tangent_slope a)

def area_of_triangle (a : ℝ) : ℝ :=
  1 / 2 * intersection_with_x_axis a * intersection_with_y_axis a

theorem find_a (a : ℝ) (h : area_of_triangle a = 18) : a = 64 :=
  sorry

end find_a_l815_815059


namespace ana_can_avoid_perfect_square_l815_815588

theorem ana_can_avoid_perfect_square (numbers : Finset ℕ) (grid_size : ℕ) (k : ℕ)
    (ana_chosen : Finset ℕ) (valid_numbers : ∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 25)
    (ana_chooses_five : ana_chosen = {17, 19, 23, 22, 11}) :
    grid_size = 5 ∧ k = 5 ∧ (∀ row_or_col, 
    let product := row_or_col.prod in ¬ (∃ m, m^2 = product)) :=
by
  sorry

end ana_can_avoid_perfect_square_l815_815588


namespace max_product_xy_l815_815392

open Real

theorem max_product_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y = 2) : 
  xy ≤ 1 / 2 :=
begin
  sorry
end

end max_product_xy_l815_815392


namespace find_x_l815_815806

theorem find_x (x : ℝ) (h : sqrt (5 * x - 1) = 3) : x = 2 := by
  sorry

end find_x_l815_815806


namespace gain_percent_approx_l815_815252

-- Definitions for the conditions
def gain_percent (C S : ℝ) : ℝ :=
  ((S - C) / C) * 100

-- The theorem statement
theorem gain_percent_approx (C S : ℝ) (h : 121 * C = 77 * S) :
  gain_percent C S ≈ 57.14 :=
by
  have hS : S = (121 / 77) * C := eq_of_mul_eq_mul_right (ne_of_gt (show (121 : ℝ) > 0 by norm_num)) h
  calc gain_percent C S
      = ((S - C) / C) * 100 : rfl
  ... = ((121 / 77 * C - C) / C) * 100 : by rw [hS]
  ... = ((121 / 77 - 1) * C / C) * 100 : by ring
  ... = ((121 / 77 - 1) * 100) : by { field_simp [ne_of_gt (show (C : ℝ) > 0 by sorry)], ring }
  ... = (44 / 77) * 100 : by norm_num
  ... = 4400 / 77 : by ring
  ... ≈ 57.14 : by norm_num
  sorry

end gain_percent_approx_l815_815252


namespace max_n_perfect_square_l815_815349

def L (n : ℕ) : ℕ := 4 * (2 ^ (n - 2) - 1)

theorem max_n_perfect_square (n : ℕ) (h : n ≥ 3) : ∃ k, L(n) = k ^ 2 ↔ n = 3 := by
  sorry

end max_n_perfect_square_l815_815349


namespace language_grouping_possible_l815_815856

-- Definitions for the numbers of students speaking different combinations of languages
variables {E F S EF ES FS EFS : ℕ}

-- Conditions given in the problem
def condition1 := E + ES + EF + EFS = 50
def condition2 := F + EF + FS + EFS = 50
def condition3 := S + ES + FS + EFS = 50

-- Theorem statement
theorem language_grouping_possible 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3) 
  : ∃ (group1 group2 group3 group4 group5 : list ℕ), 
      (∀ (group : list ℕ), group ∈ [group1, group2, group3, group4, group5] → 
      list.count group 1 = 10 ∧ 
      list.count group 2 = 10 ∧ 
      list.count group 3 = 10) := sorry

end language_grouping_possible_l815_815856


namespace probability_x_gt_5y_l815_815920

theorem probability_x_gt_5y (x y : ℝ) (h1 : 0 ≤ x ∧ x ≤ 2020) (h2 : 0 ≤ y ∧ y ≤ 2021)
  : ∃ p : ℝ, p = 101 / 1011 ∧ 
    (λ x y, Classical.some (x, y)) = (λ x y, Pr[ x > 5 * y ]) := sorry

end probability_x_gt_5y_l815_815920


namespace find_real_number_a_l815_815798

-- Defining the pure imaginary condition
def is_pure_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = complex.i * b

theorem find_real_number_a (a : ℝ) (ha : is_pure_imaginary ((1 + a * complex.i) / (2 - complex.i))) : a = 2 :=
sorry

end find_real_number_a_l815_815798


namespace find_1450th_digit_l815_815750

theorem find_1450th_digit : 
  let r := 7 / 18 in
  let dec_exp := "0." ++ String.repeat "3" 1450 in
  dec_exp[1451] = '3' :=
by \
  let r := 7 / 18 in \
  let dec_exp := "0." ++ String.repeat "3" 1450 in \
  sorry

end find_1450th_digit_l815_815750


namespace max_sum_of_multiplication_table_l815_815932

-- Define primes and their sums
def primes : List ℕ := [2, 3, 5, 7, 17, 19]

noncomputable def sum_primes := primes.sum -- 2 + 3 + 5 + 7 + 17 + 19 = 53

-- Define two groups of primes to maximize the product of their sums
def group1 : List ℕ := [2, 3, 17]
def group2 : List ℕ := [5, 7, 19]

noncomputable def sum_group1 := group1.sum -- 2 + 3 + 17 = 22
noncomputable def sum_group2 := group2.sum -- 5 + 7 + 19 = 31

-- Formulate the proof problem
theorem max_sum_of_multiplication_table : 
  ∃ a b c d e f : ℕ, 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
    (a ∈ primes ∧ b ∈ primes ∧ c ∈ primes ∧ d ∈ primes ∧ e ∈ primes ∧ f ∈ primes) ∧ 
    (a + b + c = sum_group1 ∨ a + b + c = sum_group2) ∧ 
    (d + e + f = sum_group1 ∨ d + e + f = sum_group2) ∧ 
    (a + b + c) ≠ (d + e + f) ∧ 
    ((a + b + c) * (d + e + f) = 682) := 
by
  use 2, 3, 17, 5, 7, 19
  sorry

end max_sum_of_multiplication_table_l815_815932


namespace motorboat_can_complete_lap_l815_815543

theorem motorboat_can_complete_lap (n : ℕ) (fuel : ℕ) 
  (boats_fuel : Fin n → ℕ) (total_fuel : (Fin n → ℕ) → ℕ) 
  (fuel_enough: total_fuel boats_fuel = fuel) 
  (lap_required_fuel: fuel_needed_for_lap : ℕ) 
  (fuel_needed_for_lap: fuel_needed_for_lap = fuel) : 
    ∃ (i : Fin n), can_complete_lap (boats_fuel : Fin n → ℕ) (i : Fin n) :=
by 
  sorry

end motorboat_can_complete_lap_l815_815543


namespace sqrt_real_domain_l815_815062

theorem sqrt_real_domain (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 := 
sorry

end sqrt_real_domain_l815_815062


namespace find_k_l815_815208

noncomputable def k_value (p q : ℝ^3) (h : p ≠ q) : ℝ :=
  let k := 3 / 5
  in k

theorem find_k (p q : ℝ^3) (h : p ≠ q) :
  ∃ k : ℝ, (k • p + (2 / 5) • q = (p + (2 / 5) • (q - p))) ∧ k = (3 / 5) :=
by
  use 3 / 5
  split
  { simp [smul_add, smul_sub, add_sub_cancel], ring },
  sorry

end find_k_l815_815208


namespace remaining_money_l815_815915

def initial_amount : ℕ := 10
def spent_on_toy_truck : ℕ := 3
def spent_on_pencil_case : ℕ := 2

theorem remaining_money (initial_amount spent_on_toy_truck spent_on_pencil_case : ℕ) : 
  initial_amount - (spent_on_toy_truck + spent_on_pencil_case) = 5 :=
by
  sorry

end remaining_money_l815_815915


namespace locus_of_point_C_is_ellipse_line_OP_bisects_MN_l815_815805

theorem locus_of_point_C_is_ellipse :
  ∀ (E : ℝ × ℝ), 
    (E.1 + sqrt 6) ^ 2 + E.2 ^ 2 = 32 →
    ∃ C : ℝ × ℝ, 
      (C.1 ^ 2 / 8) + (C.2 ^ 2 / 2) = 1 := 
sorry

theorem line_OP_bisects_MN :
  ∀ (A B M N P O : ℝ × ℝ),
    A = (2, 1) →
    B = (-2, -1) →
    ∃ l, 
      is_parallel l (line A B) ∧
      intersects l (locus (λ C, (C.1 ^ 2 / 8) + (C.2 ^ 2 / 2) = 1)) M N ∧
      intersection (line A M) (line B N) = P →
    bisects (line O P) (segment M N) := 
sorry

end locus_of_point_C_is_ellipse_line_OP_bisects_MN_l815_815805


namespace missing_digit_in_97th_rising_number_l815_815284

theorem missing_digit_in_97th_rising_number :
  let rising_numbers := (list.range (10 ^ 5)).filter (λ n, 
    let digits := int.to_digits 10 n in
    digits.length = 5 ∧ ∀ i j : ℕ, i < j → digits.nth i < digits.nth j)
  rising_97 := list.nth rising_numbers 96
  in
  ∀ d : ℕ, d ∈ (int.to_digits 10 rising_97) ↔ d ≠ 5 :=
by
  -- Proof goes here
  sorry

end missing_digit_in_97th_rising_number_l815_815284


namespace find_complex_number_l815_815393

open Complex

theorem find_complex_number (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
sorry

end find_complex_number_l815_815393


namespace number_of_integers_satisfying_inequality_l815_815440

theorem number_of_integers_satisfying_inequality : 
  (finset.filter (λ x, (x - 2)^2 ≤ 4) (finset.Icc 0 4)).card = 5 := 
  sorry

end number_of_integers_satisfying_inequality_l815_815440


namespace inverse_proposition_equivalence_l815_815955

theorem inverse_proposition_equivalence (x y : ℝ) :
  (x = y → abs x = abs y) ↔ (abs x = abs y → x = y) :=
sorry

end inverse_proposition_equivalence_l815_815955


namespace sequence_sum_l815_815370

noncomputable def p : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 4
| 4 := 8
| (n + 5) := p (n + 1)

noncomputable def q : ℕ → ℤ
| 1 := -1
| 2 := -1
| 3 := 1
| (n + 4) := q (n + 1)

def sum_product (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, (p (i+1) * q (i+1))

theorem sequence_sum : sum_product 2016 = -2520 :=
  sorry

end sequence_sum_l815_815370


namespace find_m_l815_815031

theorem find_m (x y m : ℝ) (h1 : 2 * x + y = 1) (h2 : x + 2 * y = 2) (h3 : x + y = 2 * m - 1) : m = 1 :=
by
  sorry

end find_m_l815_815031


namespace existence_of_b_l815_815522

theorem existence_of_b's (n m : ℕ) (h1 : 1 < n) (h2 : 1 < m) 
  (a : Fin m → ℕ) (h3 : ∀ i, 0 < a i ∧ a i ≤ n^m) :
  ∃ b : Fin m → ℕ, (∀ i, 0 < b i ∧ b i ≤ n) ∧ (∀ i, a i + b i < n) :=
by
  sorry

end existence_of_b_l815_815522


namespace probability_even_sum_l815_815937

def tiles := fin 10
def players := fin 3

-- Each player selects 3 tiles from the 10 numbered tiles
def selects (p : players) : set tiles := {x | p = p} -- Placeholder since we're not defining the actual selection explicitly

-- Probability calculation of all three players obtaining an even sum
def prob_even_sum (s1 s2 s3 : set tiles) : ℚ :=
  if (all_even_sum s1 s2 s3)
  then rational.mk 1 28
  else 0

-- Function to determine if all selected sets sum to an even number
def all_even_sum (s1 s2 s3 : set tiles) : Prop :=
  even_sum s1 ∧ even_sum s2 ∧ even_sum s3

-- Function to determine if a set's sum is even
def even_sum (s : set tiles) : Prop :=
  ∑ x in s, x.val % 2 = 0

-- m and n such that the probability is m/n and they are relatively prime positive integers
def m := 1
def n := 28

-- Final statement
theorem probability_even_sum (s1 s2 s3 : set tiles) :
  prob_even_sum s1 s2 s3 = 1 / 28 :=
by sorry

end probability_even_sum_l815_815937


namespace total_cost_is_108_l815_815716

def num_students : ℕ := 30
def index_cards_per_student : ℕ := 10
def num_periods : ℕ := 6
def cost_per_pack : ℕ := 3
def cards_per_pack : ℕ := 50

def total_cards_needed : ℕ := num_students * index_cards_per_student * num_periods
def num_packs_needed : ℕ := total_cards_needed / cards_per_pack
def total_cost : ℕ := num_packs_needed * cost_per_pack

theorem total_cost_is_108 : total_cost = 108 := by
  unfold total_cards_needed
  unfold num_packs_needed
  unfold total_cost
  unfold num_students
  unfold index_cards_per_student
  unfold num_periods
  unfold cost_per_pack
  unfold cards_per_pack
  -- Calculates: total_cards_needed = 30 * 10 * 6 = 1800
  -- Then: num_packs_needed = 1800 / 50 = 36
  -- Finally: total_cost = 36 * 3 = 108
  sorry

end total_cost_is_108_l815_815716


namespace segment_length_of_sphere_cut_l815_815604

def length_of_segment_cut_by_sphere (a : ℝ) (cube : set (ℝ × ℝ × ℝ)) (sphere : set (ℝ × ℝ × ℝ)) 
  (midpoint_AA1 : ℝ × ℝ × ℝ) (midpoint_DD1 : ℝ × ℝ × ℝ) : ℝ := a

theorem segment_length_of_sphere_cut
  (a : ℝ) 
  (cube : set (ℝ × ℝ × ℝ)) 
  (sphere : set (ℝ × ℝ × ℝ))
  (h_cube : ∀ (v : ℝ × ℝ × ℝ), v ∈ cube → v ∈ sphere)
  (E : ℝ × ℝ × ℝ) (F : ℝ × ℝ × ℝ)
  (hE : E = (1/2 : ℝ) • ((1, 0, 0) + (1, 0, a)))
  (hF : F = (1/2 : ℝ) • ((0, 1, 0) + (0, 1, a))) : 
  length_of_segment_cut_by_sphere a cube sphere E F = a := 
sorry

end segment_length_of_sphere_cut_l815_815604


namespace arithmetic_sequence_sum_l815_815866

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₁ : a 1 = -2012)
  (h₂ : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1)))
  (h₃ : (S 12) / 12 - (S 10) / 10 = 2) :
  S 2012 = -2012 := by
  sorry

end arithmetic_sequence_sum_l815_815866


namespace parallel_vectors_l815_815037

noncomputable def a : ℝ × ℝ × ℝ := (2, -1, 3)
noncomputable def b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

theorem parallel_vectors (x : ℝ) :
  (∃ λ : ℝ, b x = (λ * 2, λ * -1, λ * 3)) → x = -6 :=
by
  sorry

end parallel_vectors_l815_815037


namespace find_value_l815_815811

noncomputable def f : ℝ → ℝ := sorry

def tangent_line (x y : ℝ) : Prop := x + 2 * y + 1 = 0

def has_tangent_at (f : ℝ → ℝ) (x0 : ℝ) (L : ℝ → ℝ → Prop) : Prop :=
  L x0 (f x0)

theorem find_value (h : has_tangent_at f 2 tangent_line) :
  f 2 - 2 * (deriv f 2) = -1/2 :=
sorry

end find_value_l815_815811


namespace length_of_bridge_l815_815289

/--
A train 110 meters long running at the speed of 72 km/hr takes 12.299016078713702 seconds to cross a bridge of certain length. Prove that the length of the bridge is 136.98032157427404 meters.
-/
theorem length_of_bridge (length_of_train : ℝ) (train_speed_kmh : ℝ) (time_to_cross_bridge : ℝ) : 
  length_of_train = 110 →
  train_speed_kmh = 72 →
  time_to_cross_bridge = 12.299016078713702 →
  let train_speed_ms := train_speed_kmh * (1000 / 3600) in
  let total_distance := train_speed_ms * time_to_cross_bridge in
  total_distance - length_of_train = 136.98032157427404 :=
begin
  intros h_train_length h_train_speed h_time,
  let train_speed_ms := train_speed_kmh * (1000 / 3600),
  let total_distance := train_speed_ms * time_to_cross_bridge,
  rw [h_train_length, h_train_speed, h_time],
  norm_num,
end

end length_of_bridge_l815_815289


namespace solve_for_x_l815_815046

theorem solve_for_x 
  (y : ℚ) (x : ℚ)
  (h : x / (x - 1) = (y^3 + 2 * y^2 - 2) / (y^3 + 2 * y^2 - 3)) :
  x = (y^3 + 2 * y^2 - 2) / 2 :=
sorry

end solve_for_x_l815_815046


namespace tangent_line_at_0_l815_815578

noncomputable def f (x : ℝ) : ℝ :=
  exp x * (2 * x - 1)

def tangent_line_eq (m b x y: ℝ) : Prop :=
  y = m * x + b

theorem tangent_line_at_0 (x y : ℝ) : 
  let p := (0, f 0)
  let m := 1
  let b := -1
  tangent_line_eq m b x y → p = (0, -1) → y = x - 1 :=
  by
    sorry

end tangent_line_at_0_l815_815578


namespace math_class_problem_l815_815071

theorem math_class_problem
  (x a : ℝ)
  (h_mistaken : (2 * (2 * 4 - 1) + 1 = 5 * (4 + a)))
  (h_original : (2 * x - 1) / 5 + 1 = (x + a) / 2)
  : a = -1 ∧ x = 13 := by
  sorry

end math_class_problem_l815_815071


namespace inf_primes_exists_l815_815887

theorem inf_primes_exists
  (a : ℕ → ℤ) 
  (h_distinct : ∀ i j : ℕ, i ≠ j → a i ≠ a j) :
  ∃^∞ (p : ℕ), ∃ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ p ∣ (a i * a j * a k - 1) :=
sorry

end inf_primes_exists_l815_815887


namespace derivative_y_l815_815751

noncomputable def tg_ln2 : ℝ := Real.tan (Real.log 2)

def y (x : ℝ) : ℝ := (tg_ln2 * (Real.sin (19 * x))^2) / (19 * Real.cos (38 * x))

theorem derivative_y (x : ℝ) :
  deriv y x = (tg_ln2^2 * Real.tan (38 * x)) / Real.cos (38 * x) :=
by
  sorry

end derivative_y_l815_815751


namespace regular_polygon_star_angle_l815_815658

theorem regular_polygon_star_angle (N : ℕ) (h1 : N ≥ 3) (h2 : (∀ x, x ∈ (set.range (λ k : ℕ, 360 / N)) → x = 108)) : N = 10 :=
sorry

end regular_polygon_star_angle_l815_815658


namespace quadratic_equation_nonzero_coefficient_l815_815809

theorem quadratic_equation_nonzero_coefficient (m : ℝ) : 
  m - 1 ≠ 0 ↔ m ≠ 1 :=
by
  sorry

end quadratic_equation_nonzero_coefficient_l815_815809


namespace find_p_from_conditions_l815_815027

variable (p : ℝ) (y x : ℝ)

noncomputable def parabola_eq : Prop := y^2 = 2 * p * x

noncomputable def p_positive : Prop := p > 0

noncomputable def point_on_parabola : Prop := parabola_eq p 1 (p / 4)

theorem find_p_from_conditions (hp : p_positive p) (hpp : point_on_parabola p) : p = Real.sqrt 2 :=
by 
  -- The actual proof goes here
  sorry

end find_p_from_conditions_l815_815027


namespace boat_distance_downstream_l815_815666

-- Definitions
def boat_speed_in_still_water : ℝ := 24
def stream_speed : ℝ := 4
def time_downstream : ℝ := 3

-- Effective speed downstream
def speed_downstream := boat_speed_in_still_water + stream_speed

-- Distance calculation
def distance_downstream := speed_downstream * time_downstream

-- Proof statement
theorem boat_distance_downstream : distance_downstream = 84 := 
by
  -- This is where the proof would go, but we use sorry for now
  sorry

end boat_distance_downstream_l815_815666


namespace negation_of_proposition_l815_815177

theorem negation_of_proposition : 
    (¬ (∀ x : ℝ, x^2 - 2 * |x| ≥ 0)) ↔ (∃ x : ℝ, x^2 - 2 * |x| < 0) :=
by sorry

end negation_of_proposition_l815_815177


namespace max_expression_value_l815_815755

theorem max_expression_value (a b c : ℝ) (hb : b > a) (ha : a > c) (hb_ne : b ≠ 0) :
  ∃ M, M = 27 ∧ (∀ a b c, b > a → a > c → b ≠ 0 → (∃ M, (2*a + 3*b)^2 + (b - c)^2 + (2*c - a)^2 ≤ M * b^2) → M ≤ 27) :=
  sorry

end max_expression_value_l815_815755


namespace max_checkers_convex_polygon_l815_815625

/--
Given an 8x8 chessboard, the maximum number of checkers that can be placed on the board such that they form the vertices of a convex polygon is 13.
-/
theorem max_checkers_convex_polygon (chessboard : grid 8 8) : 
  ∃ (points : set (fin 8 × fin 8)), 
  set.finite points ∧ 
  convex_hull ℝ points ∧ 
  set.card points = 13 :=
sorry

end max_checkers_convex_polygon_l815_815625


namespace sum_of_two_numbers_l815_815544

theorem sum_of_two_numbers (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x = 14) : x + y = 39 :=
by
  sorry

end sum_of_two_numbers_l815_815544


namespace distance_from_y_axis_l815_815164

theorem distance_from_y_axis (dx dy : ℝ) (h1 : dx = 8) (h2 : dx = (1/2) * dy) : dy = 16 :=
by
  sorry

end distance_from_y_axis_l815_815164


namespace probability_wendy_rolls_higher_l815_815138

theorem probability_wendy_rolls_higher (a b : ℕ) (h : Nat.gcd a b = 1) :
  (∃ a b : ℕ, a + b = 17 ∧ a / b = 5 / 12) :=
begin
  -- Conditions: each roll one six-sided die
  have total_outcomes : 6 * 6 = 36 := by norm_num,
  have favorable_outcomes : 1 + 2 + 3 + 4 + 5 = 15 := by norm_num,
  have probability : (15 : ℚ) / 36 = 5 / 12 := by norm_num,
  use [5, 12],
  split,
  { norm_num },
  { exact probability }
end

end probability_wendy_rolls_higher_l815_815138


namespace indeterminate_apothem_relationship_l815_815944

-- Definitions for the rectangle
variables (l w : ℝ) (a_pentagon p_pentagon : ℝ)

-- Conditions for the rectangle's area and perimeter relationship
def rectangle_conditions : Prop :=
  (l * w = 4 * (l + w))

-- Definitions for the pentagon
variables (s : ℝ)

-- Conditions for the pentagon's area and perimeter relationship
def pentagon_conditions : Prop :=
  (s^2 * (1/4) * (Real.sqrt (5 * (5 + 2 * Real.sqrt 5))) = 5 * s)

-- Definitions of apothems
def apothem_rectangle : ℝ :=
  (Real.sqrt (l^2 + w^2)) / 2

def apothem_pentagon : ℝ :=
  (s / (2 * Real.tan (Real.pi / 5)))

-- Main statement proving the indeterminate relationship
theorem indeterminate_apothem_relationship
  (h_rect : rectangle_conditions l w)
  (h_pent : pentagon_conditions s) :
  ∃ (ar ap : ℝ), 
    ar = apothem_rectangle l w ∧
    ap = apothem_pentagon s ∧
    ar ≠ ap ∧
    ar > ap  ∨
    ar < ap :=
by { sorry }

end indeterminate_apothem_relationship_l815_815944


namespace smallest_perfect_cube_l815_815517

theorem smallest_perfect_cube (p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) (n : ℕ) (hn : n = p^2 * q^3 * r^5) : 
  ∃ k : ℕ, k = p^6 * q^9 * r^15 ∧ (∃ m : ℕ, k = m^3 ∧ n ∣ k) := 
sorry

end smallest_perfect_cube_l815_815517


namespace last_two_digits_of_sum_of_first_15_factorials_is_13_l815_815220

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (k+1) => (k+1) * factorial k

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ :=
  (Nat.fold (fun acc x => acc + factorial x) 0 (List.range n))

theorem last_two_digits_of_sum_of_first_15_factorials_is_13 :
  last_two_digits (sum_of_factorials 15) = 13 :=
by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_is_13_l815_815220


namespace solution_set_of_inequality_l815_815516

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f x

theorem solution_set_of_inequality
  (f : R → R)
  (odd_f : odd_function f)
  (h1 : f (-2) = 0)
  (h2 : ∀ (x1 x2 : R), x1 ≠ x2 ∧ 0 < x1 ∧ 0 < x2 → (x2 * f x1 - x1 * f x2) / (x1 - x2) < 0) :
  { x : R | (f x) / x < 0 } = { x : R | x < -2 } ∪ { x : R | x > 2 } := 
sorry

end solution_set_of_inequality_l815_815516


namespace polynomial_bound_l815_815110

noncomputable def P (x : ℝ) : ℝ := sorry  -- Placeholder for the polynomial P(x)

theorem polynomial_bound (n : ℕ) (hP : ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → |P y| ≤ 1) :
  P (-1 / n) ≤ 2^(n + 1) - 1 :=
sorry

end polynomial_bound_l815_815110


namespace last_two_digits_of_sum_of_first_15_factorials_is_13_l815_815221

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | (k+1) => (k+1) * factorial k

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_of_factorials (n : ℕ) : ℕ :=
  (Nat.fold (fun acc x => acc + factorial x) 0 (List.range n))

theorem last_two_digits_of_sum_of_first_15_factorials_is_13 :
  last_two_digits (sum_of_factorials 15) = 13 :=
by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_is_13_l815_815221


namespace max_min_values_of_f_l815_815402

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log (1/4)) ^ 2 - Real.log x / Real.log (1/4) + 5

theorem max_min_values_of_f :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f x ≤ 7) ∧ (∀ x, 2 ≤ x ∧ x ≤ 4 → f x ≥ 23 / 4) :=
begin
  sorry
end

end max_min_values_of_f_l815_815402


namespace total_value_of_treats_l815_815725

def hotel_cost_per_night : ℕ := 4000
def number_of_nights : ℕ := 2
def car_cost : ℕ := 30000
def house_multiplier : ℕ := 4

theorem total_value_of_treats : 
  (number_of_nights * hotel_cost_per_night) + car_cost + (house_multiplier * car_cost) = 158000 := 
by
  sorry

end total_value_of_treats_l815_815725


namespace married_students_percentage_l815_815851

variable (T : ℝ) -- Total number of students
variable (married_percentage : ℝ := 30)

def percentage_of_married_students (total_students married_students : ℝ) : ℝ :=
  (married_students / total_students) * 100

theorem married_students_percentage :
  let male_students := 0.70 * T
  let female_students := 0.30 * T
  let married_male_students := (2 / 7) * male_students
  let married_female_students := (1 / 3) * female_students
  let total_married_students := married_male_students + married_female_students
  percentage_of_married_students T total_married_students = married_percentage :=
by
  sorry

end married_students_percentage_l815_815851


namespace original_fraction_l815_815163

theorem original_fraction (n d : ℤ) (h1 : d = n + 5) (h2 : (n + 1) / (d + 1) = 7 / 12) : (n, d) = (6, 6 + 5) := 
by
  sorry

end original_fraction_l815_815163


namespace find_a_in_triangle_l815_815874

variable (a b c : ℝ) (A B C : ℝ)

-- Given conditions
def condition_c : c = 3 := sorry
def condition_C : C = Real.pi / 3 := sorry
def condition_sinB : Real.sin B = 2 * Real.sin A := sorry

-- Theorem to prove
theorem find_a_in_triangle (hC : condition_C) (hc : condition_c) (hsinB : condition_sinB) :
  a = Real.sqrt 3 :=
sorry

end find_a_in_triangle_l815_815874


namespace club_members_remainder_l815_815587

theorem club_members_remainder (N : ℕ) (h1 : 50 < N) (h2 : N < 80)
  (h3 : N % 5 = 0) (h4 : N % 8 = 0 ∨ N % 7 = 0) :
  N % 9 = 6 ∨ N % 9 = 7 := by
  sorry

end club_members_remainder_l815_815587


namespace negation_is_false_l815_815594

-- Define the proposition and its negation
def proposition (x y : ℝ) : Prop := (x > 2 ∧ y > 3) → (x + y > 5)
def negation_proposition (x y : ℝ) : Prop := ¬ proposition x y

-- The proposition and its negation
theorem negation_is_false : ∀ (x y : ℝ), negation_proposition x y = false :=
by sorry

end negation_is_false_l815_815594


namespace larger_number_is_23_l815_815984

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l815_815984


namespace find_inverse_and_B_l815_815529

-- Define the matrix A
def A : matrix (fin 2) (fin 2) ℤ := ![
  ![1, -2],
  ![3, -7]
]

-- Define the vector B satisfies AB = [3, 1]
def B : vector (fin 2) ℤ := ![
  19,
  8
]

-- Assume that AB = [3, 1]
lemma AB_eq_31 : (A.mul_vec B) = ![
  3,
  1
] := by sorry

-- Define the inverse matrix A_inv
def A_inv : matrix (fin 2) (fin 2) ℤ := ![
  ![7, -2],
  ![3, -1]
]

-- Assume A * A_inv = identity matrix
lemma A_mul_A_inv_is_I : (A * A_inv) = 1 := by sorry

-- State the final theorem to be proven
theorem find_inverse_and_B :
  A_inv = ![
    ![7, -2],
    ![3, -1]
  ] ∧
  B = ![
    19,
    8
  ] := by
  -- Proof will be filled in here
  sorry

end find_inverse_and_B_l815_815529


namespace sum_of_digits_divisible_by_9_l815_815275

theorem sum_of_digits_divisible_by_9 (D E : ℕ) (hD : D < 10) (hE : E < 10) : 
  (D + E + 37) % 9 = 0 → ((D + E = 8) ∨ (D + E = 17)) →
  (8 + 17 = 25) := 
by
  intro h1 h2
  sorry

end sum_of_digits_divisible_by_9_l815_815275


namespace union_M_N_is_R_l815_815030

open Set

/-- Define the sets M and N -/
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x < 3}

/-- Main goal: prove M ∪ N = ℝ -/
theorem union_M_N_is_R : M ∪ N = univ :=
by
  sorry

end union_M_N_is_R_l815_815030


namespace students_in_school_l815_815546

theorem students_in_school (football cricket both neither total : Nat) 
  (h_football : football = 325)
  (h_cricket : cricket = 175)
  (h_both : both = 140)
  (h_neither : neither = 50)
  (h_total : total = football + cricket - both + neither) :
  total = 410 := 
by
  rw [h_total, h_football, h_cricket, h_both, h_neither]
  rfl

end students_in_school_l815_815546


namespace problem_1_problem_2_l815_815824

noncomputable def a : ℕ → ℝ
| 0       := 4
| (n + 1) := real.sqrt ((6 + a n) / 2)

def S (n : ℕ) : ℝ := (finset.range n).sum (λ i, a i)

theorem problem_1 (n : ℕ) (h : n > 0) : a n > a (n + 1) :=
sorry

theorem problem_2 (n : ℕ) (h : n > 0) : 2 ≤ S n - 2 * n ∧ S n - 2 * n < 16 / 7 :=
sorry

end problem_1_problem_2_l815_815824


namespace min_positive_period_of_sinusoidal_l815_815471

theorem min_positive_period_of_sinusoidal (A ω : ℝ) (ϕ : ℝ) (hA : A > 0) (hω : ω > 0)
  (hϕ : |ϕ| < π / 2) (h_symmetry : ∀ x : ℝ, f x = f (-x)) 
  (h_extremum : ∃ x : ℝ, abs x > 0 ∧ x < π / 3 ∧ is_extremum_of x f) : 
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x :=
begin
  sorry
end

noncomputable def f (x : ℝ) := A * sin (ω * x + ϕ)

end min_positive_period_of_sinusoidal_l815_815471


namespace price_reduction_l815_815669

noncomputable def percentageReduction (P R : ℝ) : ℝ := ((P - R) / P) * 100

theorem price_reduction (P : ℝ) (R : ℝ) (hR : R = 56) (h : 800 / R - 800 / P = 5) :
  percentageReduction P R ≈ 34.99 :=
by
  have hR56 : R = 56 := hR
  have hp_eq : 800 / 56 - 800 / P = 5 := h
  -- further computations and steps would follow
  sorry

end price_reduction_l815_815669


namespace mode_of_scores_is_9_6_l815_815477

def scores : List ℝ := [9.6, 9.2, 9.6, 9.7, 9.4]

theorem mode_of_scores_is_9_6 : List.mode scores = 9.6 := by
  sorry

end mode_of_scores_is_9_6_l815_815477


namespace part1_part2_l815_815821

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 1 then 1/4 + Real.logBase 4 x else 2^(-x) - 1/4

theorem part1 (x : ℝ) : f(x) >= 1/4 := sorry

theorem part2 (x0 : ℝ) : f(x0) = 3/4 → x0 = 0 ∨ x0 = 2 := sorry

end part1_part2_l815_815821


namespace main_theorem_l815_815508

-- Define the sequence a_k
def A_seq (t : ℕ) (a : ℕ → ℕ) :=
  (∀ i j k : ℕ, i < j ∧ j < k → ¬ (2 * a j = a i + a k)) ∧
  (∀ k : ℕ, a (k + 1) > a k ∧
  ¬ ∃ i j : ℕ, i < j ∧ 2 * a j = a i + a (k + 1))

-- Define the function A(x) which counts the terms ≤ x
def A_function (a : ℕ → ℕ) (x : ℝ) : ℕ :=
  finset.card (finset.filter (λ n, (a n : ℝ) ≤ x) (finset.range (x.to_nat + 1)))

-- The main theorem
theorem main_theorem (t : ℕ) (a : ℕ → ℕ) (h_seq : A_seq t a) :
  ∃ c K : ℝ, 1 < c ∧ 0 < K ∧ ∀ x : ℝ, K < x → (c * real.sqrt x ≤ (A_function a x : ℝ)) :=
begin
  sorry
end

end main_theorem_l815_815508


namespace least_number_to_subtract_l815_815235

theorem least_number_to_subtract (n : ℕ) (h : n = 427398) : ∃ k : ℕ, (n - k) % 11 = 0 ∧ k = 4 :=
by
  sorry

end least_number_to_subtract_l815_815235


namespace amount_a_receives_l815_815250

theorem amount_a_receives (a b c : ℕ) (h1 : a + b + c = 50000) (h2 : a = b + 4000) (h3 : b = c + 5000) :
  (21000 / 50000) * 36000 = 15120 :=
by
  sorry

end amount_a_receives_l815_815250


namespace proof_problem_l815_815293

def is_fraction (n d : ℕ) : Prop := d ≠ 0

def condition_expressions_as_fractions (x a b c y π : ℝ) : Prop :=
  is_fraction 2 x ∧
  is_fraction (a - b) (4 * c) ∧
  is_fraction 2 (5 + y) ∧
  is_fraction (5 * x^2 * y) (3 * x)

theorem proof_problem (x a b c y π : ℝ) 
  (h1 : is_fraction 2 x) 
  (h2 : is_fraction (a - b) (4 * c)) 
  (h3 : is_fraction 2 (5 + y)) 
  (h4 : is_fraction (5 * x^2 * y) (3 * x)) 
: ∃ n, n = 4 :=
begin
  use 4,
end

end proof_problem_l815_815293


namespace a10_gt_500_l815_815118

theorem a10_gt_500 (a : ℕ → ℕ) (b : Π k, ℕ)
  (h1 : ∀ k, 1 ≤ k → k ≤ 10 → a k > 0)
  (h2 : ∀ k l, 1 ≤ k → k < l → l ≤ 10 → a k < a l)
  (h3 : ∀ k (h₁ : 1 ≤ k) (h₂ : k ≤ 10), b k = Nat.greatest.divisor (a k) (a k))
  (h4 : ∀ k l, 1 ≤ k → k < l → l ≤ 10 → b k > b l) :
  a 10 > 500 :=
sorry

end a10_gt_500_l815_815118


namespace perpendicular_vector_condition_l815_815001

variables {V : Type*} [inner_product_space ℝ V] (a b : V)

noncomputable def dot_product : ℝ := ⟪a, b⟫

theorem perpendicular_vector_condition 
  (h_unit_a : ∥a∥ = 1) 
  (h_unit_b : ∥b∥ = 1) 
  (h_angle : real.angle_of _ a b = real.pi / 3) :
  ⟪2 • a - b, b⟫ = 0 := 
begin
  sorry
end

end perpendicular_vector_condition_l815_815001


namespace seq_ratio_l815_815371

noncomputable def arith_seq (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem seq_ratio (a d : ℝ) (h₁ : d ≠ 0) (h₂ : (arith_seq a d 2)^2 = (arith_seq a d 0) * (arith_seq a d 8)) :
  (arith_seq a d 0 + arith_seq a d 2 + arith_seq a d 4) / (arith_seq a d 1 + arith_seq a d 3 + arith_seq a d 5) = 3 / 4 :=
by
  sorry

end seq_ratio_l815_815371


namespace number_of_integers_satisfying_inequality_l815_815443

theorem number_of_integers_satisfying_inequality : 
  (finset.filter (λ x, (x - 2)^2 ≤ 4) (finset.Icc 0 4)).card = 5 := 
  sorry

end number_of_integers_satisfying_inequality_l815_815443


namespace equivalent_proof_problem_l815_815015

lemma condition_1 (a b : ℝ) (h : b > 0 ∧ 0 > a) : (1 / a) < (1 / b) :=
sorry

lemma condition_2 (a b : ℝ) (h : 0 > a ∧ a > b) : (1 / b) > (1 / a) :=
sorry

lemma condition_4 (a b : ℝ) (h : a > b ∧ b > 0) : (1 / b) > (1 / a) :=
sorry

theorem equivalent_proof_problem (a b : ℝ) :
  (b > 0 ∧ 0 > a → (1 / a) < (1 / b)) ∧
  (0 > a ∧ a > b → (1 / b) > (1 / a)) ∧
  (a > b ∧ b > 0 → (1 / b) > (1 / a)) :=
by {
  exact ⟨condition_1 a b, condition_2 a b, condition_4 a b⟩
}

end equivalent_proof_problem_l815_815015


namespace part1_part2_l815_815406

def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

theorem part1 (x : ℝ) : f x (-1) ≤ 0 ↔ x ≤ -1/3 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2_l815_815406


namespace correct_options_l815_815237

-- Define the dataset for part A
def dataset : List ℕ := [64, 91, 72, 75, 85, 76, 78, 86, 79, 92]

-- Define the binomial distribution parameters for part B
def n : ℕ := 4
def p : ℚ := 1/2

-- Define the normal distribution parameters for part C
def mu : ℚ := 5
def sigma_sq : ℚ := sorry -- will be determined from the context

-- Define the counts for the stratified sampling for part D
def students_grade_11 : ℕ := 400
def students_grade_12 : ℕ := 360
def total_selected : ℕ := 57
def selected_grade_11 : ℕ := 20

-- The main theorem to verify the correctness of the chosen options
theorem correct_options : 
  (¬ (List.nth dataset 6 = some 79) ∧      -- part A
   (n.choose 3 * (p ^ 3) * ((1 - p) ^ (n - 3)) = 1/4) ∧  -- part B
   (P (η ≤ 2) = 0.1 → P (2 < η < 8) = 0.8) ∧ -- part C
   selected_grade_11 = 20 →
   (total_selected - selected_grade_11 - (selected_grade_11 * students_grade_12 / students_grade_11)) = 19) := 
sorry

end correct_options_l815_815237


namespace solution_set_of_inequality_l815_815053

def is_decreasing (f : ℝ → ℝ) := ∀ ⦃x y : ℝ⦄, x < y → f y ≤ f x

theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_decreasing : is_decreasing f)
  (hA : f 0 = 3) (hB : f 3 = -1) :
  { x : ℝ | |f(x + 1) - 1| < 2 } = { x | -1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l815_815053


namespace geometric_sum_eight_terms_l815_815002

theorem geometric_sum_eight_terms (r a : ℝ) (h1 : r = 2) (h2 : a * (1 - r^4) / (1 - r) = 1) :
    a * (1 - r^8) / (1 - r) = 17 := 
by
  sorry

end geometric_sum_eight_terms_l815_815002


namespace apollonian_circle_theorem_l815_815950

def point : Type := (ℝ × ℝ)

def dist (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def circle_eq (p : point) (center : point) (radius : ℝ) : Prop :=
  dist p center = radius^2

noncomputable def tangent_line_angle (a b tang_center : point) (angle : ℝ) : Prop := sorry

noncomputable def intersection_points (a center : point) (radius slope : ℝ) (n : ℕ) : Prop := sorry

noncomputable def points_on_line_with_ratio (d e l : point) (ratio : ℝ) : Prop := sorry

theorem apollonian_circle_theorem :
  let A := (-4:ℝ, 2:ℝ),
      B := (2, 2),
      center := (4, 2),
      radius := 4 in
  (∀ P : point, (dist P A / dist P B = 4) ↔ circle_eq P center radius) ∧
  (tangent_line_angle A center center (π / 3)) ∧
  (¬ intersection_points A center radius (sqrt 15 / 5) 3) ∧
  (∃ D E : point, D ≠ A ∧ D ≠ B ∧ E ≠ A ∧ E ≠ B ∧ D.2 = 2 ∧ E.2 = 2 ∧ points_on_line_with_ratio D E (2, 2) 2) :=
sorry

end apollonian_circle_theorem_l815_815950


namespace number_of_positive_integer_solutions_l815_815457

theorem number_of_positive_integer_solutions :
  let S := {x : ℕ | 15 < -x + 18}
  (S ∩ {x : ℕ | x > 0}).toFinset.card = 2 :=
by
  let S := {x : ℕ | 15 < -x + 18}
  have hs1 : ∀x ∈ S, -x + 18 > 15 := by sorry
  have hs2 : ∀x ∈ S, x < 3 := by sorry
  have hs3 : {x : ℕ | x ∈ S ∧ x > 0} = {1, 2} := by sorry
  show (S ∩ {x : ℕ | x > 0}).toFinset.card = 2,
  from calc
    (S ∩ {x : ℕ | x > 0}).toFinset.card
        = {1, 2}.card : by rw hs3
        = 2 : by norm_num

end number_of_positive_integer_solutions_l815_815457


namespace mark_donates_1800_cans_l815_815539

variable (number_of_shelters people_per_shelter cans_per_person : ℕ)
variable (total_people total_cans_of_soup : ℕ)

-- Given conditions
def number_of_shelters := 6
def people_per_shelter := 30
def cans_per_person := 10

-- Calculations based on conditions
def total_people := number_of_shelters * people_per_shelter
def total_cans_of_soup := total_people * cans_per_person

-- Proof statement
theorem mark_donates_1800_cans : total_cans_of_soup = 1800 := by
  -- stretch sorry proof placeholder for the proof
  sorry

end mark_donates_1800_cans_l815_815539


namespace transmission_time_is_128_l815_815743

def total_time (blocks chunks_per_block rate : ℕ) : ℕ :=
  (blocks * chunks_per_block) / rate

theorem transmission_time_is_128 :
  total_time 80 256 160 = 128 :=
  by
  sorry

end transmission_time_is_128_l815_815743


namespace lineup_count_l815_815489

-- Define five distinct people
inductive Person 
| youngest : Person 
| oldest : Person 
| person1 : Person 
| person2 : Person 
| person3 : Person 

-- Define the total number of people
def numberOfPeople : ℕ := 5

-- Define a function to calculate the number of ways to line up five people with constraints
def lineupWays : ℕ := 3 * 4 * 3 * 2 * 1

-- State the theorem
theorem lineup_count (h₁ : numberOfPeople = 5) (h₂ : ¬ ∃ (p : Person), p = Person.youngest ∨ p = Person.oldest → p = Person.youngest) :
  lineupWays = 72 :=
by
  sorry

end lineup_count_l815_815489


namespace family_e_initial_members_l815_815652

theorem family_e_initial_members 
(a b c d f E : ℕ) 
(h_a : a = 7) 
(h_b : b = 8) 
(h_c : c = 10) 
(h_d : d = 13) 
(h_f : f = 10)
(h_avg : (a - 1 + b - 1 + c - 1 + d - 1 + E - 1 + f - 1) / 6 = 8) : 
E = 6 := 
by 
  sorry

end family_e_initial_members_l815_815652


namespace five_letter_combinations_l815_815104

theorem five_letter_combinations : 
  let letters := 26 in
  let total_combinations := letters * letters * letters in
  total_combinations = 17576 := 
by
  -- Sketch of the mathematical statement
  -- Let letters be the number of choices for any letter from the alphabet, which is 26.
  -- The first and last letters must be the same (1 choice out of 26).
  -- The middle letter must be 'A' (1 fixed choice).
  -- The second and fourth letters can be any letter from the alphabet with 26 choices each.
  sorry

end five_letter_combinations_l815_815104


namespace value_of_f_m_minus_1_pos_l815_815905

variable (a m : ℝ)
variable (f : ℝ → ℝ)
variable (a_pos : a > 0)
variable (fm_neg : f m < 0)
variable (f_def : ∀ x, f x = x^2 - x + a)

theorem value_of_f_m_minus_1_pos : f (m - 1) > 0 :=
by
  sorry

end value_of_f_m_minus_1_pos_l815_815905


namespace pyramid_volume_correct_l815_815282

-- Define points and variables
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

-- Given conditions
def E := Point3D 0 (Real.sqrt 493) 0
def G := Point3D (7 * Real.sqrt 2) 0 0
def H := Point3D (-7 * Real.sqrt 2) 0 0

-- Coordinates of Q, solving from the given conditions
def Q : Point3D := {
  x := 0,
  y := 337 / (2 * Real.sqrt 493),
  z := 115 / Real.sqrt 151
}

-- Area of triangle EFG
def area_EFG : ℝ := 21 * Real.sqrt 151

-- Correct answer (volume of the pyramid)
def volume_pyramid : ℝ := 7 * 115

theorem pyramid_volume_correct :
  volume_pyramid = 805 :=
by
  sorry

end pyramid_volume_correct_l815_815282


namespace painting_time_l815_815505

theorem painting_time (karl_time leo_time : ℝ) (t : ℝ) (break_time : ℝ) : 
  karl_time = 6 → leo_time = 8 → break_time = 0.5 → 
  (1 / karl_time + 1 / leo_time) * (t - break_time) = 1 :=
by
  intros h_karl h_leo h_break
  rw [h_karl, h_leo, h_break]
  -- sorry to skip the proof
  sorry

end painting_time_l815_815505


namespace kayla_total_score_after_ten_levels_l815_815884

def level_score : ℕ → ℕ
| 1       := 2
| 2       := 3
| 3       := 5
| 4       := 8
| 5       := 12
| (n + 1) := level_score n + (level_score n - level_score (n - 1)) + 1

def total_score_after_ten_levels : ℕ :=
  (List.range 10).map (λ n => level_score (n + 1)).sum

theorem kayla_total_score_after_ten_levels :
  total_score_after_ten_levels = 185 :=
sorry

end kayla_total_score_after_ten_levels_l815_815884


namespace geometric_arithmetic_mean_inequality_equality_condition_l815_815845

theorem geometric_arithmetic_mean_inequality
  {a b : ℕ → ℝ}
  (n : ℕ)
  (h_pos_a : ∀ k, 1 ≤ k ∧ k ≤ n → 0 < a k)
  (h_pos_b : ∀ k, 1 ≤ k ∧ k ≤ n → 0 < b k) :
  (Real.geom_mean (Finset.range n) a) + (Real.geom_mean (Finset.range n) b) ≤ Real.geom_mean (Finset.range n) (λ k, a k + b k) :=
sorry

theorem equality_condition
  {a b : ℕ → ℝ}
  (n : ℕ)
  (h_pos_a : ∀ k, 1 ≤ k ∧ k ≤ n → 0 < a k)
  (h_pos_b : ∀ k, 1 ≤ k ∧ k ≤ n → 0 < b k) :
  (∀ k, 1 ≤ k ∧ k ≤ n → a k / b k = a 1 / b 1) ↔
  (Real.geom_mean (Finset.range n) a + Real.geom_mean (Finset.range n) b = Real.geom_mean (Finset.range n) (λ k, a k + b k)) :=
sorry

end geometric_arithmetic_mean_inequality_equality_condition_l815_815845


namespace exists_group_of_10_l815_815068

theorem exists_group_of_10 (members : Finset ℕ) (h : members.card = 30)
  (send_hat : ∀ m ∈ members, {n | n ∈ members ∧ n ≠ m}) :
  ∃ (G : Finset ℕ), G.card = 10 ∧ ∀ a b ∈ G, a ≠ b → 
    (send_hat a).disjoint (send_hat b) :=
sorry

end exists_group_of_10_l815_815068


namespace integer_solutions_to_inequality_l815_815434

theorem integer_solutions_to_inequality :
  ∃ n : ℕ, n = 5 ∧ (∀ x : ℤ, (x - 2) ^ 2 ≤ 4 ↔ x ∈ {0, 1, 2, 3, 4}) :=
by
  sorry

end integer_solutions_to_inequality_l815_815434


namespace parabola_intersects_line_at_one_point_l815_815008

theorem parabola_intersects_line_at_one_point (α : ℝ) (h : ∀ x : ℝ, (3 * x^2 + 1 = 4 * (sin α) * x) → α = 60) : α = 60 :=
  sorry

end parabola_intersects_line_at_one_point_l815_815008


namespace quadratic_condition_l815_815807

theorem quadratic_condition (m : ℝ) (h : (m - 1) ≠ 0) : m ≠ 1 :=
by {
  intro h1,
  rw h1 at h,
  apply h,
  ring,
}

end quadratic_condition_l815_815807


namespace sequence_a10_l815_815869

theorem sequence_a10 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n+1) - a n = 1 / (4 * ↑n^2 - 1)) :
  a 10 = 28 / 19 :=
by
  sorry

end sequence_a10_l815_815869


namespace complement_intersect_eq_l815_815828

-- Define Universal Set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define Set P
def P : Set ℕ := {2, 3, 4}

-- Define Set Q
def Q : Set ℕ := {1, 2}

-- Complement of P in U
def complement_U_P : Set ℕ := U \ P

-- Goal Statement
theorem complement_intersect_eq {U P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4}) 
  (hP : P = {2, 3, 4}) 
  (hQ : Q = {1, 2}) : 
  (complement_U_P ∩ Q) = {1} := 
by
  sorry

end complement_intersect_eq_l815_815828


namespace bread_slices_left_l815_815207

theorem bread_slices_left :
    let monday_slices := 2
    let tuesday_slices := 3
    let wednesday_slices := 4
    let thursday_slices := 1
    let friday_slices := 3
    let saturday_slices := 5
    let sunday_slices := 3
    let total_slices := 22
    let used_slices := monday_slices + tuesday_slices + wednesday_slices + thursday_slices + friday_slices + saturday_slices + sunday_slices
    total_slices - used_slices = 1 := 
by {
    have h1 : used_slices = 21 := by decide,
    rw [h1],
    have h2 : total_slices - 21 = 1 := by decide,
    exact h2,
    sorry
}

end bread_slices_left_l815_815207


namespace a_arithmetic_sum_b_of_first_n_terms_l815_815359

-- Define the function f(x)
def f (x n : ℝ) : ℝ := x^2 - 2*(n + 1)*x + n^2 + 5*n - 7

-- Define the sequence a_n as the y-coordinate of the vertex of f(x)
def a (n : ℕ) : ℝ := 3*n - 8

-- Proving the sequence {a_n} is arithmetic sequence
theorem a_arithmetic (n : ℕ) : a n + 3 = a (n + 1) := by {
  unfold a,
  ring,
}

-- Define b_n as the distance from the vertex of the graph of f(x) to the x-axis
def b (n : ℕ) : ℝ := abs (3*n - 8)

-- Define S_n as the sum of the first n terms of b_n
def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b (i + 1) -- Sum of the first n terms of b(n)

-- We can state the final sum as a theorem which needs to be proved
theorem sum_b_of_first_n_terms (n : ℕ) : S n = sorry := 
  sorry -- We expect the user to fill the actual proof here.


end a_arithmetic_sum_b_of_first_n_terms_l815_815359


namespace find_side_length_a_l815_815873

variable {a b c : ℝ}
variable {B : ℝ}

theorem find_side_length_a (h_b : b = 7) (h_c : c = 5) (h_B : B = 2 * Real.pi / 3) :
  a = 3 :=
sorry

end find_side_length_a_l815_815873


namespace simplify_expression_l815_815767

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by 
  sorry

end simplify_expression_l815_815767


namespace sum_of_squares_series_additional_terms_l815_815142

theorem sum_of_squares_series (n : ℕ) : 
  ∑ i in range (n + 1), i^2 + ∑ i in range n, i^2 = n * (2 * n^2 + 1) / 3 :=
by induction n with k ih
  · simp
  · sorry

theorem additional_terms (k : ℕ) :
  let lhs := (∑ i in range (k + 1), i^2 + ∑ i in range k, i^2)
  let rhs := (∑ i in range (k + 2), i^2 + ∑ i in range (k + 1), i^2)
  rhs - lhs = (k + 1)^2 + k^2 :=
by sorry

end sum_of_squares_series_additional_terms_l815_815142


namespace value_of_v3_at_neg4_l815_815619

def poly (x : ℤ) : ℤ := (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem value_of_v3_at_neg4 : poly (-4) = -49 := 
by
  sorry

end value_of_v3_at_neg4_l815_815619


namespace largest_monochromatic_subgraph_at_least_1012_l815_815606

-- Definitions based on conditions
def complete_graph (n : ℕ) : Type := 
  { G : graph (fin n) // ∀ i j, G.adj i j }

def edge_coloring (G : complete_graph 2024) : fin 2024 → fin 2024 → ℕ :=
  λ i j, sorry -- defines an arbitrary edge coloring with three colors

def monochromatic_subgraph_size (G : complete_graph 2024) (coloring : fin 2024 → fin 2024 → ℕ) (color : ℕ) : ℕ := 
  sorry -- function to determine the size of the largest monochromatic connected subgraph

-- Theorem based on question and correct answer
theorem largest_monochromatic_subgraph_at_least_1012 :
  ∀ (G : complete_graph 2024) (coloring : fin 2024 → fin 2024 → ℕ), 
  ∃ color, monochromatic_subgraph_size G coloring color ≥ 1012 :=
sorry

end largest_monochromatic_subgraph_at_least_1012_l815_815606


namespace conditionally_constrained_functions_l815_815260

-- Definitions of the functions
def f1 (x : ℝ) : ℝ := 4 * x
def f2 (x : ℝ) : ℝ := x ^ 2 + 2
def f3 (x : ℝ) : ℝ := 2 * x / (x ^ 2 - 2 * x + 5)
noncomputable def f4 (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_cond : ∀ x1 x2, f x1 - f x2 ≤ 4 * |x1 - x2|) : Prop :=
  ∀ x, |f x| ≤ 4 * |x|

-- The theorem
theorem conditionally_constrained_functions (ω : ℝ) (h_ω : ω > 0) :
  (∀ x, |f1 x| ≤ ω * |x|) ∧
  (∀ x, ¬ (|f2 x| ≤ ω * |x|)) ∧
  (∀ x, |f3 x| ≤ ω * |x|) ∧
  ∀ (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_cond : ∀ x1 x2, f x1 - f x2 ≤ 4 * |x1 - x2|), (∀ x, |f x| ≤ 4 * |x|) :=
by
  sorry

end conditionally_constrained_functions_l815_815260


namespace solution_set_of_inequality_l815_815760

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x^2 - 3*x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
by sorry

end solution_set_of_inequality_l815_815760


namespace problem_1_problem_2_l815_815901

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 4) + abs (x - 1)

-- Define the function g
def g (x m : ℝ) : ℝ := (2017 * x - 2016) / (f x + 2 * m)

-- The first problem statement: solving the inequality f(x) ≤ 5
theorem problem_1 (x : ℝ) : f x ≤ 5 ↔ 0 ≤ x ∧ x ≤ 5 :=
by sorry

-- The second problem statement: finding the range of m
theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x + 2 * m ≠ 0) ↔ m > -3 / 2 :=
by sorry

end problem_1_problem_2_l815_815901


namespace solution_set_equiv_l815_815746

def solution_set (x : ℝ) : Prop := 2 * x - 6 < 0

theorem solution_set_equiv (x : ℝ) : solution_set x ↔ x < 3 := by
  sorry

end solution_set_equiv_l815_815746


namespace measure_85_liters_l815_815287

theorem measure_85_liters (C1 C2 C3 : ℕ) (capacity : ℕ) : 
  (C1 = 0 ∧ C2 = 0 ∧ C3 = 1 ∧ capacity = 85) → 
  (∃ weighings : ℕ, weighings ≤ 8 ∧ C1 = 85 ∨ C2 = 85 ∨ C3 = 85) :=
by 
  sorry

end measure_85_liters_l815_815287


namespace distinct_values_S_l815_815119

noncomputable def j : ℂ := complex.sqrt 2 * complex.I
noncomputable def S (n : ℤ) : ℂ := j^n + j^(-n)

theorem distinct_values_S : (finset.image (λ n, S n) (finset.range 4)).card = 4 := by
  sorry

end distinct_values_S_l815_815119


namespace hundredth_letter_is_A_l815_815225

def pattern_ABC (n : ℕ) : char :=
  ['A', 'B', 'C'][n % 3]

theorem hundredth_letter_is_A : pattern_ABC 100 = 'A' :=
by
  sorry

end hundredth_letter_is_A_l815_815225


namespace exists_root_between_roots_l815_815507

theorem exists_root_between_roots 
  (a b c : ℝ) 
  (h_a : a ≠ 0) 
  (x₁ x₂ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + c = 0) 
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) 
  (hx : x₁ < x₂) :
  ∃ x₃ : ℝ, x₁ < x₃ ∧ x₃ < x₂ ∧ (a / 2) * x₃^2 + b * x₃ + c = 0 :=
by 
  sorry

end exists_root_between_roots_l815_815507


namespace compute_expression_l815_815896

noncomputable def z : ℂ := complex.exp (2 * complex.pi * complex.I * 4 / 7)

theorem compute_expression : 
  (z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2) :=
sorry

end compute_expression_l815_815896


namespace combination_vs_permutation_problem_l815_815636

theorem combination_vs_permutation_problem :
  (∀ (n : ℕ), (choose 10 2) ≠ 90) *  -- Option A (combination): 10 friends handshake count using combinations
  (choose 9 2 = 36) *  -- Option B (combination): Connecting any two of 9 distinct points
  (∀ (n : ℕ) (h : n ≥ 3), choose n 3 ≥ 1) ∨ (∀ S (H : S = finset.range (n + 1)), (finset.card  S) = 3) *  -- Option C (combination): Subsets of 3 elements
  (∀ (n : ℕ) (h : n=50), (factorial 2) * (choose 50 2)) := -- Option D (permutation): Selecting and assigning roles to 2 students from 50, permutation scenario.
sorry

end combination_vs_permutation_problem_l815_815636


namespace female_hippos_to_total_hippos_ratio_l815_815200

theorem female_hippos_to_total_hippos_ratio
  (initial_elephants : ℕ := 20)
  (initial_hippos : ℕ := 35)
  (total_animals_after_birth : ℕ := 315)
  (newborns_per_female_hippo : ℕ := 5)
  (extra_newborn_elephants : ℕ := 10)
  : ∃ (F : ℕ), ratio (F : ℝ) (initial_hippos + (newborns_per_female_hippo * F) : ℝ) = (5 : ℝ) / (32 : ℝ) := 
begin
  sorry
end

end female_hippos_to_total_hippos_ratio_l815_815200


namespace people_in_house_l815_815645

theorem people_in_house 
  (charlie_and_susan : ℕ := 2)
  (sarah_and_friends : ℕ := 5)
  (living_room_people : ℕ := 8) :
  (charlie_and_susan + sarah_and_friends) + living_room_people = 15 := 
by
  sorry

end people_in_house_l815_815645


namespace storage_capacity_l815_815500

theorem storage_capacity (used_storage : ℕ) (song_storage : ℕ) (num_songs : ℕ) (mb_to_gb : ℕ) : 
  used_storage = 4 → song_storage = 30 → num_songs = 400 → mb_to_gb = 1000 → 
  (used_storage + (num_songs * song_storage) / mb_to_gb) = 16 :=
by
  intros h_used h_song h_num h_mb_to_gb
  rw [h_used, h_song, h_num, h_mb_to_gb]
  simp
  norm_num
  sorry

end storage_capacity_l815_815500


namespace max_area_of_triangle_l815_815346

-- Defining the conditions
def c : ℝ := sqrt 2
def C : ℝ := π / 4
def cosA (A : ℝ) (sinA : ℝ) : ℝ := sinA * sin C

-- Using cosine rule
def cosineRule (a b : ℝ) : ℝ := a^2 + b^2 - 2 * a * b * cos C

-- The theorem to prove maximum area
theorem max_area_of_triangle (a b : ℝ) :
  ∃ A B : ℝ, c^2 = cosineRule a b ∧ 
            C = π / 4 ∧ 
            ( ∃ max_area : ℝ, max_area = (1 + sqrt 2) / 2 ) :=
by
  sorry

end max_area_of_triangle_l815_815346


namespace Y_squared_is_384_l815_815082

-- Given conditions
def r_1 : ℝ := 2
def r_2 := 2 * Real.sqrt 2
def r_3 := 2 * Real.sqrt 3

-- Define the product of radii
def Y : ℝ := r_1 * r_2 * r_3

-- The theorem to prove
theorem Y_squared_is_384 : Y^2 = 384 := by
    sorry

end Y_squared_is_384_l815_815082


namespace find_a2_l815_815812

theorem find_a2 (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + 2)
  (h_geom : (a 1) * (a 5) = (a 2) * (a 2)) : a 2 = 3 :=
by
  -- We are given the conditions and need to prove the statement.
  sorry

end find_a2_l815_815812


namespace cone_surface_area_l815_815959

theorem cone_surface_area (r l : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = 120 * π / 180) : 
  let A_lateral := 1 / 2 * l^2 * θ in
  let base_radius := (l * θ) / (2 * π) in
  let A_base := π * base_radius^2 in
  l = r → 
  r = 6 →
  θ = 2 * π / 3 →
  A_lateral + A_base = 16 * π :=  
by 
  intros r l θ h1 h2
  sorry

end cone_surface_area_l815_815959


namespace simplify_fraction_a_simplify_fraction_b_l815_815931

variables {R : Type*} [CommRing R]

-- Part (a)
theorem simplify_fraction_a (a b : R) (h : a ≠ b) :
    (a + b)^2 * (a^3 - b^3) / (a^2 - b^2)^2 = (a^2 + ab + b^2) / (a - b) :=
    sorry

-- Part (b)
theorem simplify_fraction_b (a b : R) (h : a ≠ b) :
    (6 * a^2 * b^2 - 3 * a^3 * b - 3 * a * b^3) / (a * b^3 - a^3 * b) = 3 * (a - b) / (a + b) :=
    sorry

end simplify_fraction_a_simplify_fraction_b_l815_815931


namespace find_a_g_monotonicity_inequality_proof_l815_815400

-- Problem 1
theorem find_a (a : ℝ) (h : ∃ x, f(x)=1 + ln(2) ∧ f(x) = ax - ln(x)) : a = 2 := sorry

-- Problem 2
theorem g_monotonicity (g f : ℝ → ℝ) (h : ∀ x, f(x) = 2x - ln(x)) (h_g : ∀ x, g(x) = 3x - 3 * ln(x) - 1 - f(x)) : 
  (∀ x ∈ Ioo 0 2, g(x) < g(1)) ∧ (∀ x ∈ Ioo 2 (1 / 0), g(x) > g(1)) := sorry

-- Problem 3
theorem inequality_proof (x1 x2 : ℝ) (h : 0 < x1 ∧ x1 < x2) : 
  (x1 - x2) / (ln(x1) - ln(x2)) < 2 * x2 := sorry

end find_a_g_monotonicity_inequality_proof_l815_815400


namespace negation_proposition_l815_815179

theorem negation_proposition {x : ℝ} : ¬ (x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := sorry

end negation_proposition_l815_815179


namespace point_distance_l815_815921

theorem point_distance (a d : ℝ) (P is_inside_square : Prop)
    (equidistant_from_endpoints_and_opposite_side : Prop) :
    is_inside_square → equidistant_from_endpoints_and_opposite_side → d = (5 * a / 8) :=
begin
    intros h1 h2,
    sorry
end

end point_distance_l815_815921


namespace people_in_house_l815_815647

theorem people_in_house 
  (charlie_and_susan : ℕ := 2)
  (sarah_and_friends : ℕ := 5)
  (living_room_people : ℕ := 8) :
  (charlie_and_susan + sarah_and_friends) + living_room_people = 15 := 
by
  sorry

end people_in_house_l815_815647


namespace certain_event_l815_815631

-- Define the problem conditions as hypotheses
def condition_a := ∀ (△ : Triangle), ∀ (O : Point), IsCircumcenter(△, O) → ∀ (d1 d2 d3 : ℝ), Distance(O, Side1(△)) = d1 ∧ Distance(O, Side2(△)) = d2 ∧ Distance(O, Side3(△)) = d3
def condition_b := ∀ (Shooter : Type) (shot : Event), Probability(hit(Shooter, shot)) > 0
def condition_c := ∀ (△ : Triangle), InteriorAnglesSum(△) = 180
def condition_d := ∀ (coin : Coin), ∃ (toss : Event), Probability(heads(toss)) = 0.5 ∧ Probability(tails(toss)) = 0.5

-- Mathematical statement to prove
theorem certain_event : ∀ (△ : Triangle), InteriorAnglesSum(△) = 180 :=
by
  -- the statement uses condition_c directly
  exact condition_c
  sorry

end certain_event_l815_815631


namespace num_isosceles_points_l815_815083

/-- Definition of a point on a 2D grid -/
structure Point :=
  (x : ℕ) (y : ℕ)

/-- Given points A and B on a 6x6 grid, we want to find the number of points C such that triangle ABC is isosceles. -/
def point_A : Point := ⟨2, 2⟩
def point_B : Point := ⟨5, 2⟩

noncomputable def valid_isosceles_points_count : ℕ :=
  let points := {p : Point // p.x ≤ 6 ∧ p.y ≤ 6}
  let is_isosceles (C : Point) : Prop :=
    let dist (P Q : Point) : ℝ := real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)
    (dist point_A point_B = dist point_A C ∨ dist point_A point_B = dist point_B C ∨ dist point_A C = dist point_B C)
  {C : Point // is_isosceles C ∧ C ≠ point_A ∧ C ≠ point_B}.to_finset.card

theorem num_isosceles_points : valid_isosceles_points_count = 10 := by
  sorry

end num_isosceles_points_l815_815083


namespace mark_donates_cans_of_soup_l815_815535

theorem mark_donates_cans_of_soup:
  let n_shelters := 6
  let p_per_shelter := 30
  let c_per_person := 10
  let total_people := n_shelters * p_per_shelter
  let total_cans := total_people * c_per_person
  total_cans = 1800 :=
by sorry

end mark_donates_cans_of_soup_l815_815535


namespace conic_section_eccentricity_l815_815010

theorem conic_section_eccentricity (m : ℝ) 
  (h : 6^2 = m * (-9)) : let e := sqrt 5 in
  ∃ a b c : ℝ, a^2 = 1 ∧ b^2 = 4 ∧ c^2 = a^2 + b^2 ∧ e = c / a :=
begin
  sorry
end

end conic_section_eccentricity_l815_815010


namespace forty_second_and_fiftieth_shooter_scores_l815_815482

-- Define the sequence of shooters' scores
def scores : ℕ → ℝ
| 0     := 60  -- First shooter
| 1     := 80  -- Second shooter
| (n+2) := (1 / (n+2)) * (scores 0 + scores 1 + ∑ i in finset.range (n + 1), scores i)

theorem forty_second_and_fiftieth_shooter_scores :
  scores 41 = 70 ∧ scores 49 = 70 :=
by
  -- Proof goes here
  sorry

end forty_second_and_fiftieth_shooter_scores_l815_815482


namespace inequality_and_converse_l815_815515

theorem inequality_and_converse
  (n : ℕ)
  (a : Fin n → ℝ)
  (x : Fin n → ℝ)
  (h_nonneg_sum : ∀ i j : Fin n, i < j → a i + a j ≥ 0)
  (h_sum_eq_one : ∑ i, x i = 1)
  (h_nonneg_x : ∀ i, 0 ≤ x i) :
  (∑ i, a i * x i ≥ ∑ i, a i * (x i)^2) ∧ 
  (∑ i, a i * x i ≥ ∑ i, a i * (x i)^2 → ∀ i j : Fin n, i < j → a i + a j ≥ 0) := by
  sorry

end inequality_and_converse_l815_815515


namespace coefficient_of_a_neg1_in_expansion_l815_815493

theorem coefficient_of_a_neg1_in_expansion :
  (∑ k in Finset.range 7, (Nat.choose 6 k) * (a ^ (6 - 3 * k)) * ((-1) ^ k)) .coeff (-1) = 0 :=
sorry

end coefficient_of_a_neg1_in_expansion_l815_815493


namespace problem_1_problem_2_l815_815792

-- Mathematical definitions based on conditions
def point_A : ℝ × ℝ := (1, -2)
def vector_a : ℝ × ℝ := (2, 3)

-- Proof problem statements
theorem problem_1 : 
  ∀ B : ℝ × ℝ, 
  (∃ k: ℝ, B = (1 + 2 * k, -2 + 3 * k)) ∧ ((B.1 - 1)^2 + (B.2 + 2)^2 = 52) → 
  B = (5, 4) := 
sorry

theorem problem_2 : 
  ∀ k : ℝ, 
  (vector_a.1 * (-3) + vector_a.2 * k < 0) ∧ (2 * k + 9 ≠ 0) → 
  k ∈ set.Ioo (-∞) (-9 / 2) ∪ set.Ioo (-9 / 2) 2 := 
sorry

end problem_1_problem_2_l815_815792


namespace license_plate_count_l815_815942

def alphabet := {'A', 'B', 'D', 'G', 'K', 'L', 'M', 'N', 'R'}

def isValidLicensePlate (plate : List Char) : Bool :=
  plate.head? ∈ some 'A' ∨ plate.head? ∈ some 'B' ∧
  plate.getLast? ∈ some 'R' ∧
  'F' ∉ plate ∧
  plate.length = 5 ∧
  plate.nodup

theorem license_plate_count : 
  let valid_plates := {plate | isValidLicensePlate plate}
  finset.card valid_plates = 420 :=
sorry

end license_plate_count_l815_815942


namespace num_integers_satisfying_inequality_l815_815427

theorem num_integers_satisfying_inequality : 
  {x : ℤ | (x - 2)^2 ≤ 4}.finite.to_finset.card = 5 :=
by {
  sorry
}

end num_integers_satisfying_inequality_l815_815427


namespace balls_picking_l815_815186

theorem balls_picking (red_bag blue_bag : ℕ) (h_red : red_bag = 3) (h_blue : blue_bag = 5) : (red_bag * blue_bag = 15) :=
by
  sorry

end balls_picking_l815_815186


namespace negation_proposition_l815_815181

theorem negation_proposition (x : ℝ) :
  ¬(∀ x : ℝ, x^2 - x + 3 > 0) ↔ ∃ x : ℝ, x^2 - x + 3 ≤ 0 := 
by { sorry }

end negation_proposition_l815_815181


namespace median_of_consecutive_integers_l815_815600

theorem median_of_consecutive_integers (n : ℕ) (S : ℤ) (h1 : n = 35) (h2 : S = 1225) : 
  n % 2 = 1 → S / n = 35 := 
sorry

end median_of_consecutive_integers_l815_815600


namespace arithmetic_geometric_product_l815_815487

variable {α : Type*} [LinearOrderedField α]

variables (a : ℕ → α) (b : ℕ → α)
variable (d : α)

-- Arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∃ a1, ∀ n, a n = a1 + n * d

-- Geometric sequence condition
def is_geometric_sequence (b : ℕ → α) : Prop :=
  ∃ b1 r, ∀ n, b n = b1 * (r ^ n)

-- Given conditions
def given_conditions (a d : α) (a_seq : ℕ → α) (b_seq : ℕ → α) : Prop :=
  is_arithmetic_sequence a_seq d ∧ 
  is_geometric_sequence b_seq ∧ 
  2 * a_seq 3 - (a_seq 7)^2 + 2 * a_seq 11 = 0 ∧
  b_seq 7 = a_seq 7

-- Target statement
theorem arithmetic_geometric_product : 
  (∀ (a_seq b_seq : ℕ → α) (d : α), given_conditions α d a_seq b_seq → b_seq 6 * b_seq 8 = 16) :=
by {
  sorry
}

end arithmetic_geometric_product_l815_815487


namespace sin_cos_acute_angle_lt_one_l815_815143

theorem sin_cos_acute_angle_lt_one (α β : ℝ) (a b c : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_triangle : a^2 + b^2 = c^2) (h_nonzero_c : c ≠ 0) :
  (a / c < 1) ∧ (b / c < 1) :=
by 
  sorry

end sin_cos_acute_angle_lt_one_l815_815143


namespace youngest_child_age_l815_815255

theorem youngest_child_age :
  ∃ x : ℕ, x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65 ∧ x = 7 :=
by
  sorry

end youngest_child_age_l815_815255


namespace opposite_of_negative_2020_is_2020_l815_815590

theorem opposite_of_negative_2020_is_2020 :
  ∃ x : ℤ, -2020 + x = 0 :=
by
  use 2020
  sorry

end opposite_of_negative_2020_is_2020_l815_815590


namespace shaded_area_to_circle_area_ratio_l815_815365

noncomputable def AB := 10 -- cm
noncomputable def AC := 6  -- cm
noncomputable def CB := 4  -- cm
noncomputable def r_AB := AB / 2 -- radius of the large semi-circle
noncomputable def r_AC := AC / 2 -- radius of the medium semi-circle
noncomputable def r_CB := CB / 2 -- radius of the small semi-circle
noncomputable def shaded_area := (1/2) * Real.pi * r_AB^2 - (1/2) * Real.pi * r_AC^2 - (1/2) * Real.pi * r_CB^2
noncomputable def area_of_circle_with_CD_as_radius := Real.pi * r_CB^2
noncomputable def ratio := shaded_area / area_of_circle_with_CD_as_radius

theorem shaded_area_to_circle_area_ratio : ratio = 3 / 2 :=
by
  sorry

end shaded_area_to_circle_area_ratio_l815_815365


namespace find_f_37_5_l815_815801

noncomputable def f (x : ℝ) : ℝ := sorry

/--
Given that \( f \) is an odd function defined on \( \mathbb{R} \) and satisfies
\( f(x+2) = -f(x) \). When \( 0 \leqslant x \leqslant 1 \), \( f(x) = x \),
prove that \( f(37.5) = 0.5 \).
-/
theorem find_f_37_5 (f : ℝ → ℝ) (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (periodic_f : ∀ x : ℝ, f (x + 2) = -f x)
  (interval_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) : f 37.5 = 0.5 :=
sorry

end find_f_37_5_l815_815801


namespace fixed_point_is_one_three_l815_815386

noncomputable def fixed_point_of_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : ℝ × ℝ :=
  (1, 3)

theorem fixed_point_is_one_three {a : ℝ} (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  fixed_point_of_function a h_pos h_ne_one = (1, 3) :=
  sorry

end fixed_point_is_one_three_l815_815386


namespace cost_per_meter_l815_815960

-- Defining the parameters and their relationships
def length : ℝ := 58
def breadth : ℝ := length - 16
def total_cost : ℝ := 5300
def perimeter : ℝ := 2 * (length + breadth)

-- Proving the cost per meter of fencing
theorem cost_per_meter : total_cost / perimeter = 26.50 := 
by
  sorry

end cost_per_meter_l815_815960


namespace find_missing_number_l815_815561

theorem find_missing_number (x : ℤ) (h : (4 + 3) + (8 - x - 1) = 11) : x = 3 :=
sorry

end find_missing_number_l815_815561


namespace hexagon_piece_area_l815_815686

theorem hexagon_piece_area (A : ℝ) (n : ℕ) (h1 : A = 21.12) (h2 : n = 6) : 
  A / n = 3.52 :=
by
  -- The proof will go here
  sorry

end hexagon_piece_area_l815_815686


namespace inverse_of_original_l815_815957

-- Definitions based on conditions
def original_proposition : Prop := ∀ (x y : ℝ), x = y → |x| = |y|

def inverse_proposition : Prop := ∀ (x y : ℝ), |x| = |y| → x = y

-- Lean 4 statement
theorem inverse_of_original : original_proposition → inverse_proposition :=
sorry

end inverse_of_original_l815_815957


namespace range_of_a_l815_815786

def f (a x : ℝ) : ℝ := 
  if x < 1/2 then -x + a 
  else Real.log x / Real.log 2

theorem range_of_a (a : ℝ) (f_min : ∀ x : ℝ, f a x ≥ -1) : a ≥ 1/2 :=
begin
  sorry
end

end range_of_a_l815_815786


namespace ratio_is_one_half_l815_815708

noncomputable def ratio_of_females_to_males (f m : ℕ) (avg_female_age avg_male_age avg_total_age : ℕ) : ℚ :=
  (f : ℚ) / (m : ℚ)

theorem ratio_is_one_half (f m : ℕ) (avg_female_age avg_male_age avg_total_age : ℕ)
  (h_female_age : avg_female_age = 45)
  (h_male_age : avg_male_age = 30)
  (h_total_age : avg_total_age = 35)
  (h_total_avg : (45 * f + 30 * m) / (f + m) = 35) :
  ratio_of_females_to_males f m avg_female_age avg_male_age avg_total_age = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l815_815708


namespace time_for_trains_to_cross_l815_815210

noncomputable def time_to_cross_each_other (L1 L2 : ℝ) (v1_kmph v2_kmph : ℝ) : ℝ :=
  let v1 := v1_kmph * (1000 / 3600) in -- convert v1 from km/hr to m/s
  let v2 := v2_kmph * (1000 / 3600) in -- convert v2 from km/hr to m/s
  let relative_speed := v1 + v2 in
  (L1 + L2) / relative_speed

theorem time_for_trains_to_cross :
  time_to_cross_each_other 156.62 100 30 36 ≈ 14.01 :=
by
  sorry

end time_for_trains_to_cross_l815_815210


namespace carries_container_holds_1200_marbles_l815_815862

def container_volume (base_area height : ℕ) : ℕ :=
  base_area * height

theorem carries_container_holds_1200_marbles :
  let base_area := 10 in
  let height := 3 in
  let capacity := 150 in
  container_volume base_area height = 30 →
  container_volume (4 * base_area) (2 * height) = 240 →
  (240 / 30) * capacity = 1200 :=
by
  intros h1 h2
  simp [container_volume, h1, h2]
  sorry

end carries_container_holds_1200_marbles_l815_815862


namespace larger_number_is_23_l815_815983

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end larger_number_is_23_l815_815983


namespace suitable_survey_method_l815_815204

-- Definitions from conditions
def is_destructive (process : Prop) : Prop := process

-- The formal statement for the proof problem
theorem suitable_survey_method 
  (process : Prop) 
  (h : is_destructive process) : 
  (most_suitable_method process = "sampling survey") :=
sorry

end suitable_survey_method_l815_815204


namespace range_of_x_l815_815820

theorem range_of_x (x m t : ℝ) (h1 : x > -1) (h2 : x < Real.sqrt 2) 
    (h3 : 1 ≤ m) (h4 : m ≤ 2) (h5 : π / 6 ≤ t) (h6 : t ≤ π / 2) :
    2 * x^2 + m * x - 2 < m + 2 * x := by 
  have ht : f t = 2 * Real.sin t := by rfl 
  have hf : 1 ≤ 2 * Real.sin t ∧ 2 * Real.sin t ≤ 2 := by
    split
    · apply Real.sin_nonneg_of_nonneg_of_le_pi_div_two; linarith
    · apply Real.sin_le_one_of_le_pi_div_two; linarith
  have hm : 1 ≤ f t ∧ f t ≤ 2 := by
    rw [ht]
    exact hf
  sorry

end range_of_x_l815_815820


namespace miles_walked_l815_815042

/-- 
If a person walks 1 mile every 15 minutes and walked for 45 minutes, then
the number of miles walked is 3.
-/
theorem miles_walked (walk_rate : ℕ) (walk_time : ℕ) (h1 : walk_rate = 15) (h2 : walk_time = 45) :
  walk_time / walk_rate = 3 :=
by
  simp [h1, h2]
  sorry

end miles_walked_l815_815042


namespace moles_of_CO2_formed_l815_815757

-- Definitions based on the conditions provided
def moles_HNO3 := 2
def moles_NaHCO3 := 2
def balanced_eq (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) : Prop :=
  HNO3 = NaHCO3 ∧ NaNO3 = NaHCO3 ∧ CO2 = NaHCO3 ∧ H2O = NaHCO3

-- Lean Proposition: Prove that 2 moles of CO2 are formed
theorem moles_of_CO2_formed :
  balanced_eq moles_HNO3 moles_NaHCO3 moles_HNO3 moles_HNO3 moles_HNO3 →
  ∃ CO2, CO2 = 2 :=
by
  sorry

end moles_of_CO2_formed_l815_815757


namespace correct_function_for_conditions_l815_815295

noncomputable def is_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x < f y

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x, f x = f (x + p)

def abs_sin (x : ℝ) : ℝ :=
| sin x |

def abs_sin_2x (x : ℝ) : ℝ :=
| sin (2 * x) |

def abs_cos (x : ℝ) : ℝ :=
| cos x |

def abs_cos_2x (x : ℝ) : ℝ :=
| cos (2 * x) |

theorem correct_function_for_conditions :
  ∀ (f : ℝ → ℝ),
  (f = abs_sin ∨ f = abs_sin_2x ∨ f = abs_cos ∨ f = abs_cos_2x) →
  (is_increasing f (Set.Ioo 0 (π / 2)) ∧ is_even f ∧ has_period f π) ↔ (f = abs_sin) :=
by
  intros
  sorry

end correct_function_for_conditions_l815_815295


namespace sum_of_fourth_and_sixth_terms_of_sequence_l815_815072

noncomputable def sequence (n : ℕ) : ℚ :=
if n = 1 then 1
else (n / (n - 1))^2

theorem sum_of_fourth_and_sixth_terms_of_sequence :
  sequence 4 + sequence 6 = 724 / 225 :=
by
  sorry

end sum_of_fourth_and_sixth_terms_of_sequence_l815_815072


namespace Sarah_correct_responses_l815_815159

theorem Sarah_correct_responses : ∃ x : ℕ, x ≥ 22 ∧ (7 * x - (26 - x) + 4 ≥ 150) :=
by
  sorry

end Sarah_correct_responses_l815_815159


namespace cube_root_approx_l815_815048

-- Define constants for the approximate values given in the problem
def approx_root_2_37 : ℝ := 1.333
def root_1000 : ℝ := 10

-- Define the main theorem
theorem cube_root_approx : ∀ (x : ℝ), x = 2370 → (x^(1/3)) ≈ 13.33 :=
by
  intro x hx
  have h1 : x = 2.37 * 1000 := by linarith [hx]
  have h2 : (2.37:ℝ)^(1/3) ≈ approx_root_2_37 := by norm_num
  have h3 : (1000:ℝ)^(1/3) = root_1000 := by norm_num
  rw [h1] -- use x = 2.37 * 1000
  rw [Mul.cube_root_eq_mul_cube_root h2 h3] -- use property of cube roots
  linarith -- approximate result using given values
  sorry

end cube_root_approx_l815_815048


namespace probability_of_sum_4_6_8_l815_815674

-- Define the faces of the dice
def die1_faces := [1, 2, 3, 3, 4, 4]
def die2_faces := [1, 3, 5, 5, 6, 7]

-- Calculate the probability that the sum of the top faces is 4, 6, or 8
def favorable_sum_counts : Nat :=
  let sums := [ (d1 + d2) | d1 ← die1_faces, d2 ← die2_faces ]
  (sums.count (λ s => s = 4 || s = 6 || s = 8))

-- Probability that the sum is 4, 6, or 8
def probability_favorable_sum : Rat :=
  favorable_sum_counts / (die1_faces.length * die2_faces.length)

-- Expected result
def expected_probability : Rat := 7 / 18

-- Assertion
theorem probability_of_sum_4_6_8 : probability_favorable_sum = expected_probability := by
  sorry

end probability_of_sum_4_6_8_l815_815674


namespace average_difference_correct_l815_815313

def swimming_sessions_over_six_weeks := {
  Camden: 16,
  Susannah: 24,
  Elijah: 30,
  Mia: 20
}

def weeks := 6

-- Function to calculate the weekly average
def weekly_average (total_sessions : ℕ) (weeks : ℕ) : ℚ :=
  total_sessions / weeks

-- Weekly averages
def Camden_weekly_average := weekly_average swimming_sessions_over_six_weeks.Camden weeks
def Susannah_weekly_average := weekly_average swimming_sessions_over_six_weeks.Susannah weeks
def Elijah_weekly_average := weekly_average swimming_sessions_over_six_weeks.Elijah weeks
def Mia_weekly_average := weekly_average swimming_sessions_over_six_weeks.Mia weeks

def max_weekly_average := max (max Camden_weekly_average Susannah_weekly_average) (max Elijah_weekly_average Mia_weekly_average)
def min_weekly_average := min (min Camden_weekly_average Susannah_weekly_average) (min Elijah_weekly_average Mia_weekly_average)

def average_difference := max_weekly_average - min_weekly_average

theorem average_difference_correct : average_difference = 2.33 := by
  sorry

end average_difference_correct_l815_815313


namespace increasing_function_range_l815_815383

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : ∀ x y : ℝ, x < y → f a x < f a y) : 
  3 / 2 ≤ a ∧ a < 2 := by
  sorry

end increasing_function_range_l815_815383


namespace packs_of_peanuts_l815_815907

noncomputable def price_coloring_books : ℕ := 2 * 4
noncomputable def price_stuffed_animal : ℕ := 11
noncomputable def total_given : ℕ := 25
noncomputable def price_per_pack_of_peanuts : ℚ := 1.5

theorem packs_of_peanuts (price_coloring_books price_stuffed_animal total_given : ℕ) 
    (price_per_pack_of_peanuts : ℚ) : 
    25 - (price_coloring_books + price_stuffed_animal) / price_per_pack_of_peanuts = 4 :=
by
  -- Definitions for easy readability
  let total_cost_coloring_books := price_coloring_books
  let total_cost_stuffed_animal := price_stuffed_animal
  let total_spent_on_peanuts := total_given - (total_cost_coloring_books + total_cost_stuffed_animal)
  let packs_of_peanuts := total_spent_on_peanuts / price_per_pack_of_peanuts

  -- Direct substitution and calculation
  have h₀ : total_cost_coloring_books = 2 * 4 := rfl
  have h₁ : total_cost_stuffed_animal = 11 := rfl
  have h₂ : total_given = 25 := rfl
  have h₃ : price_per_pack_of_peanuts = 1.5 := rfl

  -- Perform the actual calculation
  have h₄ : total_spent_on_peanuts = 25 - (8 + 11) := rfl
  have h₅ : total_spent_on_peanuts = 6 := by simp [h₄]
  have h₆ : packs_of_peanuts = 6 / 1.5 := by simp [total_spent_on_peanuts, h₅, h₃]

  -- Prove the final necessity
  have : 6 / 1.5 = 4 := by norm_num
  exact this

sorry

end packs_of_peanuts_l815_815907


namespace option_C_true_l815_815357

variable {a b : ℝ}

theorem option_C_true (h : a < b) : a / 3 < b / 3 := sorry

end option_C_true_l815_815357


namespace integer_solution_count_l815_815454

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end integer_solution_count_l815_815454


namespace max_value_of_f_l815_815795

theorem max_value_of_f (x : ℝ) (h : 0 < x ∧ x < 1) : ∃ c, c = 1 / 4 ∧ ∀ y, 0 < y ∧ y < 1 → x(1 - x) ≤ c :=
by
    sorry

end max_value_of_f_l815_815795


namespace inclination_angle_range_l815_815184

open Real

noncomputable def inclination_range : Set ℝ :=
  {θ : ℝ | (π / 4) ≤ θ ∧ θ ≤ (3 * π / 4)}

theorem inclination_angle_range (α : ℝ) :
  ∃ θ : ℝ, (x - y * sin α - 3 = 0) -> (θ ∈ inclination_range) :=
sorry

end inclination_angle_range_l815_815184


namespace quadratic_condition_l815_815808

theorem quadratic_condition (m : ℝ) (h : (m - 1) ≠ 0) : m ≠ 1 :=
by {
  intro h1,
  rw h1 at h,
  apply h,
  ring,
}

end quadratic_condition_l815_815808


namespace intersection_of_sets_l815_815525

noncomputable def setM : Set ℝ := { x : ℝ | (x + 6) * (x - 1) < 0 }
noncomputable def setN : Set ℝ := { x : ℝ | 2 ^ x < 1 }

theorem intersection_of_sets :
  setM ∩ setN = { x : ℝ | -6 < x ∧ x < 0 } :=
sorry

end intersection_of_sets_l815_815525


namespace ellipse_equation_l815_815014

theorem ellipse_equation (f1 f2 p : ℝ × ℝ)
  (h_f1 : f1 = (0, -4))
  (h_f2 : f2 = (0, 4))
  (h_p : p = (0, -6)) :
  ∃ a b : ℝ, (a = 6) ∧ (b = real.sqrt 20) ∧ (∀ x y, x^2 / b^2 + y^2 / a^2 = 1) :=
begin
  use [6, real.sqrt 20],
  split,
  { refl },
  split,
  { simp },
  { intro x,
    intro y,
    sorry }
end

end ellipse_equation_l815_815014


namespace intersection_point_l815_815699

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (-1, 3)
def F2 : ℝ × ℝ := (4, 1)

-- Define the known intersection point of the ellipse with y-axis
def known_point : ℝ × ℝ := (0, 1)

-- Define a function to calculate the Euclidean distance between two points
def dist (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the property of the ellipse
def ellipse_property (P : ℝ × ℝ) : Prop :=
  dist P F1 + dist P F2 = dist known_point F1 + dist known_point F2

-- Define the second intersection point of the ellipse with y-axis
def second_point : ℝ × ℝ := (0, -2)

-- Prove that the second intersection point satisfies the ellipse property
theorem intersection_point : ellipse_property second_point :=
  sorry

end intersection_point_l815_815699


namespace juice_profit_eq_l815_815741

theorem juice_profit_eq (x : ℝ) :
  (70 - x) * (160 + 8 * x) = 16000 :=
sorry

end juice_profit_eq_l815_815741


namespace gcd_n7_n_l815_815754

theorem gcd_n7_n (n : ℤ) : ∃ g, is_greatest (n^7 - n) g ∧ g = 42 := sorry

end gcd_n7_n_l815_815754


namespace tangent_of_KT_to_Gamma1_l815_815890

noncomputable def is_tangent (l : Line) (c : Circle) (p : Point) : Prop :=
  -- Definition of the tangent line condition.
  l.is_tangent c p

theorem tangent_of_KT_to_Gamma1
  (Γ Γ1 : Circle) (R S T J A K : Point) (l : Line)
  (hRS_distinct : R ≠ S)
  (hRS_not_diameter : ¬Γ.is_diameter R S)
  (hl_tangent : l.is_tangent Γ R)
  (hS_midpoint : S.is_midpoint R T)
  (hJ_on_minor_arc : Γ.is_on_minor_arc R S J)
  (hAJ_intersect_K : A.J_intersects Γ K)
  (hΓ1_circumcircle : Γ1 = Γ.circumcircle_of_triangle J S T)
  (hΓ1_intersects_l : Π(l : Line), l.is_line_through (Γ1.intersect_line l).fst (Γ1.intersect_line l).snd)
  : is_tangent (Line.mk K T) Γ1 T :=
sorry

end tangent_of_KT_to_Gamma1_l815_815890


namespace total_people_in_house_l815_815644

-- Definitions of initial condition in the bedroom and living room
def charlie_susan_in_bedroom : ℕ := 2
def sarah_and_friends_in_bedroom : ℕ := 5
def people_in_living_room : ℕ := 8

-- Prove the total number of people in the house is 15
theorem total_people_in_house : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom + people_in_living_room = 15 :=
by
  -- sum the people in the bedroom (Charlie, Susan, Sarah, 4 friends)
  have bedroom_total : charlie_susan_in_bedroom + sarah_and_friends_in_bedroom = 7 := by sorry
  -- sum the people in the house (bedroom + living room)
  show bedroom_total + people_in_living_room = 15 from sorry

end total_people_in_house_l815_815644


namespace correct_options_l815_815236

-- Define the dataset for part A
def dataset : List ℕ := [64, 91, 72, 75, 85, 76, 78, 86, 79, 92]

-- Define the binomial distribution parameters for part B
def n : ℕ := 4
def p : ℚ := 1/2

-- Define the normal distribution parameters for part C
def mu : ℚ := 5
def sigma_sq : ℚ := sorry -- will be determined from the context

-- Define the counts for the stratified sampling for part D
def students_grade_11 : ℕ := 400
def students_grade_12 : ℕ := 360
def total_selected : ℕ := 57
def selected_grade_11 : ℕ := 20

-- The main theorem to verify the correctness of the chosen options
theorem correct_options : 
  (¬ (List.nth dataset 6 = some 79) ∧      -- part A
   (n.choose 3 * (p ^ 3) * ((1 - p) ^ (n - 3)) = 1/4) ∧  -- part B
   (P (η ≤ 2) = 0.1 → P (2 < η < 8) = 0.8) ∧ -- part C
   selected_grade_11 = 20 →
   (total_selected - selected_grade_11 - (selected_grade_11 * students_grade_12 / students_grade_11)) = 19) := 
sorry

end correct_options_l815_815236


namespace population_increase_l815_815943

theorem population_increase:
  ∃ (r : ℝ), (r = 10) ∧ (12000 * (1 + r / 100)^2 = 14520) :=
by
  exists 10
  split
  rfl
  norm_num
  sorry

end population_increase_l815_815943


namespace function_passes_through_point_l815_815953

theorem function_passes_through_point
  (a : ℝ)
  (h₀ : 0 < a)
  (h₁ : a ≠ 1) :
  ∃ (x y : ℝ), x = -2 ∧ y = 5 ∧ y = a^(-x-2) + 4 := 
by
  use (-2, 5)
  split
  case left => sorry
  case right =>
    show 5 = a^(-(-2) - 2) + 4, sorry

end function_passes_through_point_l815_815953


namespace last_two_digits_of_sum_of_factorials_l815_815214

-- Problem statement: Sum of factorials from 1 to 15
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.factorial k)

-- Define the main problem
theorem last_two_digits_of_sum_of_factorials : 
  (sum_factorials 15) % 100 = 13 :=
by 
  sorry

end last_two_digits_of_sum_of_factorials_l815_815214


namespace num_integers_satisfying_inequality_l815_815428

theorem num_integers_satisfying_inequality : 
  {x : ℤ | (x - 2)^2 ≤ 4}.finite.to_finset.card = 5 :=
by {
  sorry
}

end num_integers_satisfying_inequality_l815_815428


namespace perpendicular_line_eq_l815_815342

variables {x y : ℝ}

theorem perpendicular_line_eq (A : ℝ × ℝ) (hA : A = (-1, 2))
  (h_perpendicular : ∀ {m' : ℝ}, m' = 3 / 2) :
  ∃ (c : ℝ), ∀ (x y : ℝ), (3 * x - 2 * y + c = 0) ∧ (x, y) = A :=
begin
  sorry
end

end perpendicular_line_eq_l815_815342


namespace staircases_descending_order_l815_815739

theorem staircases_descending_order :
  {s : List (ℕ × ℕ × ℕ × ℕ) | s = [(4, 0, 0, 0), (3, 1, 0, 0), (2, 2, 0, 0), (2, 1, 1, 0), (1, 1, 1, 1)]} ∈
  (all_staircases : List (ℕ × ℕ × ℕ × ℕ)) →
  (∀ s, s ∈ all_staircases → 
    ∃ a b c d, a + b + c + d = 4 ∧ a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ 
               a * b * c * d = 0 ∧ -- ensures only 4 bricks in total
               (4, 0, 0, 0) ≥ s ∧ s ≥ (1, 1, 1, 1)) :=
by
  sorry

end staircases_descending_order_l815_815739


namespace james_total_distance_l815_815103

-- Definitions for the distance formula between two points.
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Conditions: The initial and stopping points.
def point1 : ℝ × ℝ := (2, 3)
def point2 : ℝ × ℝ := (5, 5)
def point3 : ℝ × ℝ := (10, -3)

-- Calculating distances based on the given points.
def distance1 : ℝ := distance point1 point2
def distance2 : ℝ := distance point2 point3

-- Main statement: The total distance James travels.
theorem james_total_distance : distance1 + distance2 = real.sqrt 13 + real.sqrt 89 := 
by
  sorry

end james_total_distance_l815_815103


namespace shift_left_by_pi_over_12_l815_815205

-- Given definitions based only on the problem statement and conditions
def f1 (x : ℝ) : ℝ := sin (2 * x)
def f2 (x : ℝ) : ℝ := cos (2 * x - π / 3)

-- The goal statement to prove
theorem shift_left_by_pi_over_12 : ∀ x : ℝ, f1 (x + π / 12) = f2 x :=
by
  sorry

end shift_left_by_pi_over_12_l815_815205


namespace probability_of_both_defective_l815_815672

-- Definitions based on the given conditions
def total_tubes : ℕ := 20
def defective_tubes : ℕ := 5
def non_replacement : Bool := true

-- Definition of the event for the first draw being defective
def P_first_defective : ℚ := defective_tubes / total_tubes

-- Definition of the event for the second draw being defective given the first was defective
def P_second_defective_given_first_defective : ℚ := (defective_tubes - 1) / (total_tubes - 1)

-- The overall probability that both draws are defective
def P_both_defective : ℚ := P_first_defective * P_second_defective_given_first_defective

-- The theorem to prove
theorem probability_of_both_defective : P_both_defective = 1 / 19 :=
by
  have P_first : ℚ := 5 / 20
  have P_second_given_first : ℚ := 4 / 19
  have P_both : ℚ := P_first * P_second_given_first
  show P_both = 1 / 19
  rw [P_first, P_second_given_first, mul_div_cancel']
  exact div_self 19 sorry

end probability_of_both_defective_l815_815672


namespace divides_22_l815_815241

theorem divides_22 (n : ℕ) : ∀ (n ∈ [3, 4, 5, 7, 11]), (n ∣ 22) ↔ (n = 11) :=
by
  intros n hn
  simp at hn
  fin_cases hn <;> simp [Nat.dvd_iff_mod_eq_zero]

end divides_22_l815_815241


namespace dishonest_dealer_profit_percent_l815_815268

-- Define the dealer's claimed weight and actual weight
def claimed_weight : ℕ := 1000
def actual_weight : ℕ := 780

-- Compute the profit in grams
def profit : ℕ := claimed_weight - actual_weight

-- Compute the profit percent
def profit_percent : ℝ := (profit.toFloat / actual_weight.toFloat) * 100

-- Statement to be proved
theorem dishonest_dealer_profit_percent :
  profit_percent ≈ 28.205 :=
by
  -- Skip the proof
  sorry

end dishonest_dealer_profit_percent_l815_815268


namespace max_base_seven_digit_sum_l815_815229

theorem max_base_seven_digit_sum (n : ℕ) (h : n < 2401) : ∃ m, m < 2401 ∧ (nat.digits 7 m).sum = 12 :=
begin
  sorry
end

end max_base_seven_digit_sum_l815_815229


namespace square_side_length_l815_815244

theorem square_side_length (P : ℝ) (h : P = 100) : ∃ (side_length : ℝ), side_length = 25 := 
by
  -- Given the perimeter P of a square
  -- We know P = 100
  use 25
  sorry

end square_side_length_l815_815244


namespace triangle_properties_l815_815090

theorem triangle_properties
  (b : ℝ) (c : ℝ) (A : ℝ)
  (hb : b = 2) (hc : c = √3) (hA : A = π / 6) :
  (1 / 2 * b * c * Real.sin A = √3 / 2)
  ∧ (Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A) = 1) :=
by
  sorry

end triangle_properties_l815_815090


namespace relationship_among_abc_l815_815779

noncomputable def a : ℝ := Real.logBase 0.5 3
noncomputable def b : ℝ := 2 ^ 0.5
noncomputable def c : ℝ := 0.5 ^ 0.3

theorem relationship_among_abc : a < c ∧ c < b :=
by
  -- use sorry to skip the proof
  sorry

end relationship_among_abc_l815_815779


namespace projection_of_w_l815_815280

open Matrix

-- Define Vectors
def u : Vector ℝ 2 := ![3, -3]
def v : Vector ℝ 2 := ![18 / 5, -18 / 5]
def w : Vector ℝ 2 := ![4, -1]

-- Define projection function
def proj (b : Vector ℝ 2) (a : Vector ℝ 2) : Vector ℝ 2 :=
  (dot_product a b / dot_product b b) • b

-- Theorem to be proved
theorem projection_of_w : proj (![1, -1]) w = ![5 / 2, -5 / 2] := sorry

end projection_of_w_l815_815280


namespace harry_apples_l815_815998

theorem harry_apples (martha_apples : ℕ) (tim_apples : ℕ) (harry_apples : ℕ)
  (h1 : martha_apples = 68)
  (h2 : tim_apples = martha_apples - 30)
  (h3 : harry_apples = tim_apples / 2) :
  harry_apples = 19 := 
by sorry

end harry_apples_l815_815998


namespace trajectory_of_P_min_cos_angle_F1PF2_l815_815024

-- Define the hyperbola and related conditions
def satisfies_hyperbola (x y : ℝ) : Prop := 2 * x^2 - 2 * y^2 = 1

-- Define the conditions for the moving point P
def satisfies_moving_point (PF1 PF2 : ℝ) : Prop := PF1 + PF2 = 4

-- Specify the equation of the ellipse (trajectory of P)
def ellipse_trajectory (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1

-- Problem 1: Prove the trajectory of point P is described by the ellipse equation
theorem trajectory_of_P (x y PF1 PF2 : ℝ) 
  (H1 : satisfies_hyperbola x y)
  (H2 : satisfies_moving_point PF1 PF2) : ellipse_trajectory x y := 
sorry

-- Problem 2: Prove the minimum value of cos(angle(F1PF2)) 
noncomputable def cos_angle_F1PF2_min (m n : ℝ) : ℝ := -- fill this with exact expression needed
sorry

theorem min_cos_angle_F1PF2 
  (m n : ℝ) -- ensure the given conditions are specified here
  (H1 : m > 0) (H2 : n > 0) (H3 : m + n = 4) (H4 : |F1F2| = 2) 
  : ∃ (c : ℝ), c = cos_angle_F1PF2_min m n :=
sorry

end trajectory_of_P_min_cos_angle_F1PF2_l815_815024


namespace exradius_circumradius_inequality_l815_815116

-- define the parameters for the problem
variables {A B C : Type*} [triangle A B C]

def exradius_opposite_vertex (r_a : ℝ) (a b c : ℝ) : Prop :=
  -- radius of the excircle opposite vertex A
  r_a = (triangle.area A B C) / ((a + b + c) / 2 - a)

def circumradius (R : ℝ) (a b c : ℝ) : Prop :=
  -- radius of the circumcircle
  R = (a * b * c) / (4 * (triangle.area A B C))

-- the theorem to prove that r_a >= 3 / 2 * R
theorem exradius_circumradius_inequality (r_a R a b c : ℝ) (h1 : exradius_opposite_vertex r_a a b c) (h2 : circumradius R a b c) :
  r_a >= (3/2) * R :=
sorry

end exradius_circumradius_inequality_l815_815116


namespace number_of_laborers_l815_815700

-- Definitions based on conditions in the problem
def hpd := 140   -- Earnings per day for heavy equipment operators
def gpd := 90    -- Earnings per day for general laborers
def totalPeople := 35  -- Total number of people hired
def totalPayroll := 3950  -- Total payroll in dollars

-- Variables H and L for the number of operators and laborers
variables (H L : ℕ)

-- Conditions provided in mathematical problem
axiom equation1 : H + L = totalPeople
axiom equation2 : hpd * H + gpd * L = totalPayroll

-- Theorem statement: we want to prove that L = 19
theorem number_of_laborers : L = 19 :=
sorry

end number_of_laborers_l815_815700


namespace sum_factorials_last_two_digits_l815_815219

/-- Prove that the last two digits of the sum of factorials of the first 15 positive integers equal to 13 --/
theorem sum_factorials_last_two_digits : 
  let f := fun n => Nat.factorial n in
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 + f 11 + f 12 + f 13 + f 14 + f 15) % 100 = 13 :=
by 
  sorry

end sum_factorials_last_two_digits_l815_815219


namespace garbage_classification_competition_l815_815070

theorem garbage_classification_competition :
  let boy_rate_seventh := 0.4
  let boy_rate_eighth := 0.5
  let girl_rate_seventh := 0.6
  let girl_rate_eighth := 0.7
  let combined_boy_rate := (boy_rate_seventh + boy_rate_eighth) / 2
  let combined_girl_rate := (girl_rate_seventh + girl_rate_eighth) / 2
  boy_rate_seventh < boy_rate_eighth ∧ combined_boy_rate < combined_girl_rate :=
by {
  sorry
}

end garbage_classification_competition_l815_815070


namespace train_length_l815_815693

/-- 
A train running at the speed of 120 km/hr crosses a pole in 12 seconds.
Prove that the length of the train is 400 meters. 
-/
theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (h_speed : speed_km_hr = 120) (h_time : time_sec = 12) : 
  let speed_m_s := speed_km_hr * (1000 / 3600) in -- converting km/hr to m/s
  let length_m := speed_m_s * time_sec in -- calculating the length
  length_m = 400 :=
by
  -- here will be the proof, which is not required in the statement
  sorry

end train_length_l815_815693


namespace radar_placement_coverage_l815_815765

noncomputable def max_distance_radars (r : ℝ) (n : ℕ) : ℝ :=
  r / Real.sin (Real.pi / n)

noncomputable def coverage_ring_area (r : ℝ) (width : ℝ) (n : ℕ) : ℝ :=
  (1440 * Real.pi) / Real.tan (Real.pi / n)

theorem radar_placement_coverage :
  let r := 41
  let width := 18
  let n := 7
  max_distance_radars r n = 40 / Real.sin (Real.pi / 7) ∧
  coverage_ring_area r width n = (1440 * Real.pi) / Real.tan (Real.pi / 7) :=
by
  sorry

end radar_placement_coverage_l815_815765


namespace ilwoong_drive_files_l815_815065

theorem ilwoong_drive_files (folders subfolders_per_folder files_per_subfolder : ℕ) :
  folders = 25 → subfolders_per_folder = 10 → files_per_subfolder = 8 →
  folders * subfolders_per_folder * files_per_subfolder = 2000 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end ilwoong_drive_files_l815_815065


namespace ellipse_semi_focal_range_l815_815800

-- Definitions and conditions from the problem
variables (a b c : ℝ) (h1 : a > b ∧ b > 0) (h2 : a^2 = b^2 + c^2)

-- Statement of the theorem
theorem ellipse_semi_focal_range : 1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 :=
by 
  sorry

end ellipse_semi_focal_range_l815_815800


namespace remainder_of_division_largest_l815_815608
-- Import the necessary library

-- Define the set of numbers
def numbers : set ℕ := {10, 11, 12, 13, 14}

-- Define the largest and the next largest numbers
def largest_number : ℕ := 14
def next_largest_number : ℕ := 13

-- Define the theorem to state the problem
theorem remainder_of_division_largest : (largest_number % next_largest_number) = 1 := by
  sorry

end remainder_of_division_largest_l815_815608


namespace f_comp_eq_two_l815_815778

-- Define the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else log a x

-- Define the main theorem with conditions and the required proof
theorem f_comp_eq_two (a : ℝ) (h : a > 2) : f a (f a 2) = 2 :=
  sorry

end f_comp_eq_two_l815_815778


namespace problem_no_solution_l815_815730

noncomputable def no_solution (x : ℝ) : Prop :=
x > 9 ∧ (sqrt (x - 9) + 3 = sqrt (x + 9) - 3) → false

theorem problem_no_solution : ¬ ∃ x, no_solution x :=
begin
  sorry
end

end problem_no_solution_l815_815730


namespace number_of_classes_schedules_l815_815688

theorem number_of_classes_schedules : 
  let classes := ["Chinese", "Mathematics", "English", "Physics", "Chemistry", "Physical Education"]
  let isValidSchedule (schedule : List String) := 
    (schedule.head ≠ "Physical Education") ∧ (schedule.get! 3 ≠ "Mathematics")
  (card {schedule : List String // schedule ⊆ classes ∧ schedule.nodup ∧ schedule.length = 6 ∧ isValidSchedule schedule}) = 480 := 
  by
  sorry

end number_of_classes_schedules_l815_815688


namespace find_a_l815_815057

noncomputable def tangent_slope (a : ℝ) : ℝ := 
-1 / 2 * a^(-3 / 2)

noncomputable def tangent_line (a x : ℝ) : ℝ := 
a^(-1 / 2) + tangent_slope a * (x - a)

noncomputable def intersection_with_y_axis (a : ℝ) : ℝ := 
tangent_line a 0

noncomputable def intersection_with_x_axis (a : ℝ) : ℝ :=
a + 2 * a^2 

noncomputable def triangle_area (a : ℝ) : ℝ := 
1 / 2 * intersection_with_x_axis a * intersection_with_y_axis a

theorem find_a (a : ℝ) (ha : triangle_area a = 18) : 
a = 64 := by sorry

end find_a_l815_815057


namespace not_all_isosceles_are_equilateral_l815_815637

-- Define an isosceles triangle
def is_isosceles (A B C : Triangle) : Prop :=
  A.side AB = A.side AC ∨ A.side AB = A.side BC ∨ A.side AC = A.side BC

-- Define an equilateral triangle
def is_equilateral (A B C : Triangle) : Prop :=
  (A.side AB = A.side AC) ∧ (A.side AB = A.side BC)

-- Define the false statement
def false_statement : Prop :=
  ∀ (A B C : Triangle), is_isosceles A B C → is_equilateral A B C

theorem not_all_isosceles_are_equilateral :
  ¬ false_statement :=
by
  sorry

end not_all_isosceles_are_equilateral_l815_815637


namespace fruit_basket_ratio_l815_815198

theorem fruit_basket_ratio :
  ∀ (oranges apples bananas peaches : ℕ),
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  oranges + apples + bananas + peaches = 28 →
  peaches / bananas = 1 / 2 := 
by
  intros oranges apples bananas peaches h_oranges h_apples h_bananas h_total,
  sorry

end fruit_basket_ratio_l815_815198


namespace intersection_product_eq_three_l815_815009

-- Define the condition of the polar equation of curve C
def polar_eq (ρ θ : ℝ) := ρ^2 - 4 * ρ * cos θ - 2 * ρ * sin θ = 0

-- Define the conversion to Cartesian coordinates
def cartesian_eq (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 5

-- Define the parametric equation of the line through P(1, 2) with inclination angle π/6
def line_param_eq (t : ℝ) := 
  (1 + (sqrt 3 / 2) * t, 2 + (1 / 2) * t)

-- Define the intersection points and the desired value
theorem intersection_product_eq_three : 
  ∀ (t1 t2 : ℝ), 
    (let (x1, y1) := line_param_eq t1 in cartesian_eq x1 y1) → 
    (let (x2, y2) := line_param_eq t2 in cartesian_eq x2 y2) → 
    abs (t1 * t2) = 3 :=
by
  sorry

end intersection_product_eq_three_l815_815009


namespace smallest_percentage_owning_90_percent_money_l815_815498

theorem smallest_percentage_owning_90_percent_money
  (P M : ℝ)
  (h1 : 0.2 * P = 0.8 * M) :
  (∃ x : ℝ, x = 0.6 * P ∧ 0.9 * M <= (0.2 * P + (x - 0.2 * P))) :=
sorry

end smallest_percentage_owning_90_percent_money_l815_815498


namespace original_number_l815_815274

theorem original_number (x : ℝ) (h : 1.35 * x = 935) : x = 693 := by
  sorry

end original_number_l815_815274


namespace smallest_value_am_hm_l815_815115

theorem smallest_value_am_hm (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * ((1 / (a + b + c)) + (1 / (b + c)) + (1 / (c + a))) ≥ 9 / 2 :=
sorry

end smallest_value_am_hm_l815_815115


namespace negation_of_universal_l815_815362

open Classical

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ∃ x : ℤ, x^3 ≥ 1 :=
by sorry

end negation_of_universal_l815_815362


namespace _l815_815917

noncomputable theorem max_colors_for_points (n : ℕ) (h : ∀ i j : ℕ, i < 2019 → j < 2019 → i ≠ j → ¬ collinear {p i, p j}):
  ∃ m : ℕ, ∀ (K : Fin m → Set Point), (∀ i j : Fin m, i ≠ j → (K i ⊆ K j ∨ K j ⊆ K i)) → (∀ i, ∀ x ∈ K i, ∀ y ∉ K i, color x ≠ color y) → ¬ collinear {p x | x < 2019} → m = 2019 :=
begin
  sorry
end

end _l815_815917


namespace optimal_fence_area_l815_815997

variables {l w : ℝ}

theorem optimal_fence_area
  (h1 : 2 * l + 2 * w = 400) -- Tiffany must use exactly 400 feet of fencing.
  (h2 : l ≥ 100) -- The length must be at least 100 feet.
  (h3 : w ≥ 50) -- The width must be at least 50 feet.
  : l * w ≤ 10000 :=      -- We need to prove that the area is at most 10000 square feet.
by
  sorry

end optimal_fence_area_l815_815997


namespace pairs_sum_to_n_l815_815154

noncomputable def divisors_sum_to_n (n : ℕ) (a1 a2 a3 a4 : ℕ) :=
  a1 % n + a2 % n + a3 % n + a4 % n = 2 * n

theorem pairs_sum_to_n {n a1 a2 a3 a4 : ℕ}
  (h1 : 2 ≤ n)
  (h2 : ∀ i ∈ {a1, a2, a3, a4}, Nat.gcd n i = 1)
  (h3 : ∀ k ∈ Nat.range (n - 1).succ, divisors_sum_to_n n (k * a1) (k * a2) (k * a3) (k * a4)) :
  ∃ r1 r2 r3 r4 : ℕ, 
    {r1, r2, r3, r4} = {a1 % n, a2 % n, a3 % n, a4 % n} ∧
    (r1 + r2 = n ∧ r3 + r4 = n) :=
sorry

end pairs_sum_to_n_l815_815154


namespace integer_solution_count_l815_815450

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end integer_solution_count_l815_815450


namespace probability_of_scoring_l815_815277

theorem probability_of_scoring :
  ∀ (p : ℝ), (p + (1 / 3) * p = 1) → (p = 3 / 4) → (p * (1 - p) = 3 / 16) :=
by
  intros p h1 h2
  sorry

end probability_of_scoring_l815_815277


namespace circle_theorem_l815_815785

noncomputable theory

variables {α : Type*} [MetricSpace α]

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

variables {α}
variable {k : Circle α}
variables {A B E C D : α}

-- Definitions to introduce the conditions:
def is_diameter (k : Circle α) (A B : α) : Prop := dist A B = 2 * k.radius
def is_on_circle (k : Circle α) (X : α) : Prop := dist k.center X = k.radius
def line_intersection_on_circle (k : Circle α) (X E : α) : α := sorry

-- Defining the given conditions
def conditions (k : Circle α) (A B E C D : α) :=
  is_diameter k A B ∧ is_on_circle k A ∧ is_on_circle k B ∧ is_on_circle k C ∧ is_on_circle k D ∧
  line_intersection_on_circle k A E = C ∧ line_intersection_on_circle k B E = D

-- Statement of the theorem to be proved
theorem circle_theorem (k : Circle α) (A B E C D : α)
  (h : conditions k A B E C D) :
  dist A C * dist A E + dist B D * dist B E = dist A B ^ 2 :=
sorry

end circle_theorem_l815_815785


namespace emily_workers_needed_l815_815318

noncomputable def least_workers_needed
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ) :
  ℕ :=
  (remaining_work / remaining_days) / (work_done / initial_days / total_workers) * total_workers

theorem emily_workers_needed 
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ)
  (h1 : total_days = 40)
  (h2 : initial_days = 10)
  (h3 : total_workers = 12)
  (h4 : work_done = 40)
  (h5 : remaining_work = 60)
  (h6 : remaining_days = 30) :
  least_workers_needed total_days initial_days total_workers work_done remaining_work remaining_days = 6 := 
sorry

end emily_workers_needed_l815_815318


namespace total_school_population_220_l815_815664

theorem total_school_population_220 (x B : ℕ) 
  (h1 : 242 = (x * B) / 100) 
  (h2 : B = (50 * x) / 100) : x = 220 := by
  sorry

end total_school_population_220_l815_815664


namespace circle_numbers_l815_815288

def valid_pattern (N : ℚ) (pattern : List ℚ) : Prop :=
  pattern = List.replicate 674 N ++ List.replicate 674 N ++ List.replicate 674 0

theorem circle_numbers (N : ℚ) (numbers : List ℚ) :
  (numbers.sum = 2022) ∧ (∀ i, abs (numbers[i] - numbers[(i+1) % 2022]) = numbers[(i+1) % 2022]) →
  (∀ k, N ≠ 0 → numbers = List.replicate k N ++ List.replicate (674 - k) N ++ List.replicate 674 0) →
  N = 3 / 2 :=
by
  sorry

end circle_numbers_l815_815288


namespace zero_in_interval_l815_815398

theorem zero_in_interval :
  ∃ n : ℤ, 
    (f : ℝ → ℝ := λ x, Real.log x + 3 / 2 * x - 9) ∧ 
    (f 5 < 0) ∧ 
    (f 6 > 0) ∧ 
    Continuous f ∧ 
    StrictMono f ∧ 
    f (5 : ℝ) = 0 := 
sorry

end zero_in_interval_l815_815398


namespace count_integers_satisfying_inequality_l815_815446

theorem count_integers_satisfying_inequality : ∃ (count : ℕ), count = 5 ∧ 
  ∀ x : ℤ, (0 ≤ x ∧ x ≤ 4) → ((x - 2) * (x - 2) ≤ 4) :=
begin
  existsi 5, 
  split,
  { refl },
  { intros x hx,
    cases hx with h1 h2,
    have : 0 ≤ (x - 2) ^ 2, from pow_two_nonneg (x - 2),
    exact le_antisymm this (show (x - 2) ^ 2 ≤ 4, by sorry) }
end

end count_integers_satisfying_inequality_l815_815446


namespace count_coin_toss_sequences_l815_815731

theorem count_coin_toss_sequences : 
  ∃ n, n = (Nat.choose (3 + 5 - 1) (5 - 1)) * (Nat.choose (6 + 6 - 1) (6 - 1)) ∧
    n = 16170 := 
by {
  have num_H_sequences : (Nat.choose (3 + 5 - 1) (5 - 1)) = 35 := by sorry,
  have num_T_sequences : (Nat.choose (6 + 6 - 1) (6 - 1)) = 462 := by sorry,
  have total_sequences : 35 * 462 = 16170 := by {
    simp [num_H_sequences, num_T_sequences],
    done,
  },
  exact ⟨_, total_sequences, rfl⟩,
}

end count_coin_toss_sequences_l815_815731


namespace problem_max_value_of_product_solved_l815_815565

noncomputable def max_value_of_product {d : ℝ} (x : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → (1 / x (n + 1) - 1 / x n = d) ∧ (∑ i in finset.range 20, x (i + 1)) = 200) →
  (∃ x3 x18, x3 = x 3 ∧ x18 = x 18 ∧ x3 * x18 = 100)

-- Define the specific problem as an instance
noncomputable def specific_problem : Prop :=
  max_value_of_product (λ n, if n = 0 then 0 else (1 / (d * (n - 1) + 1)))

-- Example usage (unproved statement)
theorem problem_max_value_of_product_solved : specific_problem :=
  sorry

end problem_max_value_of_product_solved_l815_815565


namespace range_of_k_l815_815395

noncomputable def quadratic_has_real_roots (k : ℝ) :=
  ∃ (x : ℝ), (k - 3) * x^2 - 4 * x + 2 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≤ 5 := 
  sorry

end range_of_k_l815_815395


namespace number_of_odd_blue_faces_cubes_l815_815696

/-
A wooden block is 5 inches long, 5 inches wide, and 1 inch high.
The block is painted blue on all six sides and then cut into twenty-five 1 inch cubes.
Prove that the number of cubes each have a total number of blue faces that is an odd number is 9.
-/

def cubes_with_odd_blue_faces : ℕ :=
  let corner_cubes := 4
  let edge_cubes_not_corners := 16
  let center_cubes := 5
  corner_cubes + center_cubes

theorem number_of_odd_blue_faces_cubes : cubes_with_odd_blue_faces = 9 := by
  have h1 : cubes_with_odd_blue_faces = 4 + 5 := sorry
  have h2 : 4 + 5 = 9 := by norm_num
  exact Eq.trans h1 h2

end number_of_odd_blue_faces_cubes_l815_815696


namespace count_integer_solutions_l815_815423

theorem count_integer_solutions :
  {x : ℤ | (x - 2)^2 ≤ 4}.card = 5 :=
by
  sorry

end count_integer_solutions_l815_815423


namespace james_total_room_area_l815_815101

theorem james_total_room_area :
  let original_length := 13
  let original_width := 18
  let increase := 2
  let new_length := original_length + increase
  let new_width := original_width + increase
  let area_of_one_room := new_length * new_width
  let number_of_rooms := 4
  let area_of_four_rooms := area_of_one_room * number_of_rooms
  let area_of_larger_room := area_of_one_room * 2
  let total_area := area_of_four_rooms + area_of_larger_room
  total_area = 1800
  := sorry

end james_total_room_area_l815_815101


namespace arrange_numbers_l815_815298

theorem arrange_numbers (x y z : ℝ) (h1 : x = 20.8) (h2 : y = 0.82) (h3 : z = Real.log 20.8) : z < y ∧ y < x :=
by
  sorry

end arrange_numbers_l815_815298


namespace simplify_expression_l815_815930

theorem simplify_expression : 
  (1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1) :=
by
  sorry

end simplify_expression_l815_815930


namespace minimum_cost_l815_815996

variable (x y z : ℕ) (W : ℝ) -- Variable definitions
variable (cost_A cost_B cost_C : ℕ) -- Cost per hour for each type of generator

-- Given conditions
def total_generators := x + y + z = 10
def irrigation_requirement := 4 * x + 3 * y + 2 * z = 32
def cost_per_hour := W = cost_A * x + cost_B * y + cost_C * z
def cost_values := cost_A = 130 ∧ cost_B = 120 ∧ cost_C = 100
def generator_counts := x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1

-- Expected relationships
def z_value := z = 10 - x - y
def y_function := y = 12 - 2 * x

-- Final proof statement
theorem minimum_cost (x y z : ℕ) (W : ℝ) (cost_A cost_B cost_C : ℕ)
  (h1 : total_generators x y z)
  (h2 : irrigation_requirement x y z)
  (h3 : cost_per_hour W cost_A cost_B cost_C x y z)
  (h4 : cost_values cost_A cost_B cost_C)
  (h5 : generator_counts x y z)
  : W = 1190 := 
sorry

end minimum_cost_l815_815996


namespace find_product_of_M1_M2_l815_815513

theorem find_product_of_M1_M2 (x M1 M2 : ℝ) 
  (h : (27 * x - 19) / (x^2 - 5 * x + 6) = M1 / (x - 2) + M2 / (x - 3)) : 
  M1 * M2 = -2170 := 
sorry

end find_product_of_M1_M2_l815_815513


namespace sequence_1024_operations_final_number_l815_815316

theorem sequence_1024_operations_final_number :
  ∃ x : ℕ, (∀ n ≥ 0, n = 5 → (initial_sequence_operation n 1024 = [1023])) :=
begin
  sorry
end

end sequence_1024_operations_final_number_l815_815316


namespace TimPrankCombinations_l815_815299

-- Definitions of the conditions in the problem
def MondayChoices : ℕ := 3
def TuesdayChoices : ℕ := 1
def WednesdayChoices : ℕ := 6
def ThursdayChoices : ℕ := 4
def FridayChoices : ℕ := 2

-- The main theorem to prove the total combinations
theorem TimPrankCombinations : 
  MondayChoices * TuesdayChoices * WednesdayChoices * ThursdayChoices * FridayChoices = 144 := 
by
  sorry

end TimPrankCombinations_l815_815299


namespace simplify_complex_fraction_l815_815560

theorem simplify_complex_fraction : 
  ∀ (i : ℂ), 
  i^2 = -1 → 
  (2 - 2 * i) / (3 + 4 * i) = -(2 / 25 : ℝ) - (14 / 25) * i :=
by
  intros
  sorry

end simplify_complex_fraction_l815_815560


namespace find_a_l815_815056

noncomputable def tangent_slope (a : ℝ) : ℝ := 
-1 / 2 * a^(-3 / 2)

noncomputable def tangent_line (a x : ℝ) : ℝ := 
a^(-1 / 2) + tangent_slope a * (x - a)

noncomputable def intersection_with_y_axis (a : ℝ) : ℝ := 
tangent_line a 0

noncomputable def intersection_with_x_axis (a : ℝ) : ℝ :=
a + 2 * a^2 

noncomputable def triangle_area (a : ℝ) : ℝ := 
1 / 2 * intersection_with_x_axis a * intersection_with_y_axis a

theorem find_a (a : ℝ) (ha : triangle_area a = 18) : 
a = 64 := by sorry

end find_a_l815_815056


namespace five_letter_words_with_at_least_one_consonant_l815_815040

-- Definitions
def letters := ['A', 'B', 'C', 'D', 'E']
def consonants := ['B', 'C', 'D']
def vowels := ['A', 'E']

-- Statement
theorem five_letter_words_with_at_least_one_consonant :
  let total_words := 5 ^ 5 in
  let all_vowel_words := 2 ^ 5 in
  total_words - all_vowel_words = 3093 :=
by
  sorry

end five_letter_words_with_at_least_one_consonant_l815_815040


namespace number_of_digits_6_74_16_5_25_in_decimal_l815_815964

theorem number_of_digits_6_74_16_5_25_in_decimal : 
  let n := 6 * 74^16 * 5^25 in
  let num_digits := Nat.log10 n + 1 in
  num_digits = 28 :=
by
  sorry

end number_of_digits_6_74_16_5_25_in_decimal_l815_815964


namespace geometric_arithmetic_sequences_l815_815788

noncomputable def an (n : ℕ) : ℕ := 3 * n
noncomputable def bn (n : ℕ) : ℕ := 3^(n-1)

def Sn (n : ℕ) : ℕ := n * (an n) / 2   -- Since arithmetic series sum is calculated usually.

def c_n (n : ℕ) (k : ℕ) : ℕ := k + (an n) + (Int.log 3 (bn n))

theorem geometric_arithmetic_sequences (k t : ℕ) : 
  ∃ {k t : ℕ}, (1 / c_n 1 k + 1 / c_n t k) * 2 = (1 / c_n 2 k) ∧
  t = 3 + (8 / (k-1)) ∧
  ∀ t, t ≥ 3 →
    ((k = 2 ∧ t = 11) ∨ 
     (k = 3 ∧ t = 7) ∨ 
     (k = 5 ∧ t = 5) ∨ 
     (k = 9 ∧ t = 4)).
Proof by
  sorry

end geometric_arithmetic_sequences_l815_815788


namespace solve_exponent_equation_l815_815149

theorem solve_exponent_equation (x : ℚ) : 5^(2 * x) = 125^(1/2) → x = 3/4 :=
by sorry

end solve_exponent_equation_l815_815149


namespace num_positive_integer_solution_pairs_l815_815733

theorem num_positive_integer_solution_pairs : 
  (∃ (x y : ℕ), (7 * x + 4 * y = 800) ∧ (x > 0) ∧ (y > 0) → 29) := 
by sorry

end num_positive_integer_solution_pairs_l815_815733


namespace area_triangle_A0B0C0_eq_twice_hexagon_AC1BA1CB1_area_triangle_A0B0C0_ge_four_times_triangle_ABC_l815_815381

variable (A B C A1 B1 C1 A0 B0 C0 : Type) [acute_triangle: is_acute_triangle A B C]
variable (circumcircle : circumcircle_triangle A B C) [intersection_A1: intersect_bisector_with_circumcircle A A1]
variable [intersection_B1: intersect_bisector_with_circumcircle B B1] [intersection_C1: intersect_bisector_with_circumcircle C C1]
variable [intersection_ext_A0: intersect_external_bisectors A A1 B C A0]
variable [intersection_ext_B0: intersect_external_bisectors B B1 A C B0] 
variable [intersection_ext_C0: intersect_external_bisectors C C1 A B C0]

theorem area_triangle_A0B0C0_eq_twice_hexagon_AC1BA1CB1 :
  ∃ (area_triangle : ℝ) (area_hexagon : ℝ), area_triangle = 2 * area_hexagon ∧ 
  area_of_triangle A0 B0 C0 = area_triangle ∧ 
  area_of_hexagon A C1 B A1 C B1 = area_hexagon := sorry

theorem area_triangle_A0B0C0_ge_four_times_triangle_ABC :
  ∃ (area_triangle_A0B0C0 : ℝ) (area_triangle_ABC : ℝ), 
  area_triangle_A0B0C0 ≥ 4 * area_triangle_ABC ∧ 
  area_of_triangle A0 B0 C0 = area_triangle_A0B0C0 ∧ 
  area_of_triangle A B C = area_triangle_ABC := sorry

end area_triangle_A0B0C0_eq_twice_hexagon_AC1BA1CB1_area_triangle_A0B0C0_ge_four_times_triangle_ABC_l815_815381


namespace anthony_more_transactions_than_mabel_l815_815916

theorem anthony_more_transactions_than_mabel :
  ∃ P : ℕ, 
  let M : ℕ := 90,
      J : ℕ := 85,
      Diff : ℕ := 19,
      C : ℕ := J - Diff,
      A : ℕ := (C * 3) / 2
  in A = M * (1 + P / 100) ∧ P = 10 :=
by
  sorry

end anthony_more_transactions_than_mabel_l815_815916


namespace julia_played_more_kids_on_monday_l815_815504

def n_monday : ℕ := 6
def n_tuesday : ℕ := 5

theorem julia_played_more_kids_on_monday : n_monday - n_tuesday = 1 := by
  -- Proof goes here
  sorry

end julia_played_more_kids_on_monday_l815_815504


namespace centroid_y_sum_zero_l815_815919

theorem centroid_y_sum_zero
  (x1 x2 x3 y2 y3 : ℝ)
  (h : y2 + y3 = 0) :
  (x1 + x2 + x3) / 3 = (x1 / 3 + x2 / 3 + x3 / 3) ∧ (y2 + y3) / 3 = 0 :=
by
  sorry

end centroid_y_sum_zero_l815_815919


namespace mean_of_two_equals_mean_of_three_l815_815586

theorem mean_of_two_equals_mean_of_three (z : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + z) / 2 → 
  z = 25 / 3 := 
by 
  sorry

end mean_of_two_equals_mean_of_three_l815_815586


namespace trapezoid_area_no_solutions_l815_815945

noncomputable def no_solutions_to_trapezoid_problem : Prop :=
  ∀ (b1 b2 : ℕ), 
    (∃ (m n : ℕ), b1 = 10 * m ∧ b2 = 10 * n) →
    (b1 + b2 = 72) → false

theorem trapezoid_area_no_solutions : no_solutions_to_trapezoid_problem :=
by
  sorry

end trapezoid_area_no_solutions_l815_815945


namespace acute_dihedral_implies_acute_planar_l815_815554

theorem acute_dihedral_implies_acute_planar (T : Tetrahedron) (h : ∀ d ∈ T.dihedralAngles, d < π / 2) :
  ∀ p ∈ T.planarAngles, p < π / 2 := 
sorry

end acute_dihedral_implies_acute_planar_l815_815554


namespace min_value_abs_diff_l815_815011

-- Definitions of conditions
def is_in_interval (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 4

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  (b^2 - a^2 = 2) ∧ (c^2 - b^2 = 2)

-- Main statement
theorem min_value_abs_diff (x y z : ℝ)
  (h1 : is_in_interval x)
  (h2 : is_in_interval y)
  (h3 : is_in_interval z)
  (h4 : is_arithmetic_progression x y z) :
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end min_value_abs_diff_l815_815011


namespace problem_statement_l815_815463

noncomputable def a_value : ℝ := real.cbrt 4

theorem problem_statement
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a ^ b = b ^ a)
  (h4 : b = 4 * a) :
  a = a_value :=
by sorry

end problem_statement_l815_815463


namespace selling_price_is_correct_l815_815245

-- Definitions of the given conditions

def cost_of_string_per_bracelet := 1
def cost_of_beads_per_bracelet := 3
def number_of_bracelets_sold := 25
def total_profit := 50

def cost_of_bracelet := cost_of_string_per_bracelet + cost_of_beads_per_bracelet
def total_cost := cost_of_bracelet * number_of_bracelets_sold
def total_revenue := total_profit + total_cost
def selling_price_per_bracelet := total_revenue / number_of_bracelets_sold

-- Target theorem
theorem selling_price_is_correct : selling_price_per_bracelet = 6 :=
  by
  sorry

end selling_price_is_correct_l815_815245


namespace find_original_number_l815_815054

-- Declare the values and proof goal based on the problem conditions and solution
theorem find_original_number (x : ℝ) : 
  (213 * 16 = 3408) ∧ (1.6 * x = 3.408) → x = 2.13 :=
by
  intro h
  cases h with h213 h16
  sorry

end find_original_number_l815_815054


namespace min_value_pf1_pq_l815_815379

noncomputable def hyperbola := {p : ℝ × ℝ // (p.1 ^ 2) / 2 - p.2 ^ 2 = 1}

def is_asymptote (l : ℝ × ℝ → Prop) :=
  ∃ (m b : ℝ), l = λ p, p.2 = m * p.1 + b ∧ m = 1 / (Real.sqrt 2)

def projection (p : hyperbola) (l : ℝ × ℝ → Prop) : ℝ × ℝ :=
  let m := 1 / (Real.sqrt 2)
  let b := 0
  let x := p.val.1
  let y := p.val.2
  ((x + (m * y)) / (1 + m ^ 2), (m * (x + (m * y))) / (1 + m ^ 2))

def focus_left (h : hyperbola) : ℝ × ℝ := (Real.sqrt 3, 0)

def focus_right (h : hyperbola) : ℝ × ℝ := (Real.sqrt 3, 0)

def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2)

theorem min_value_pf1_pq (p : hyperbola) (l : ℝ × ℝ → Prop) (h : is_asymptote l) (f₁ f₂ : ℝ × ℝ) :
  focus_left p = f₁ → focus_right p = f₂ → 
  let q := projection p l
  distance p.val f₁ + distance p.val q = 2 * Real.sqrt 2 + 1 := sorry

end min_value_pf1_pq_l815_815379


namespace find_m_l815_815036

theorem find_m (m : ℝ) : 
  let a : ℝ × ℝ := (-1, m)
  let b : ℝ × ℝ := (0, 1)
  let angle_ab := Real.pi / 3
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let magnitude_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) 
  in (dot_product / (magnitude_a * magnitude_b) = Real.cos angle_ab) → m = Real.sqrt 3 / 3 :=
by 
  sorry -- proof not required as per the instruction

end find_m_l815_815036


namespace sum_of_fractions_l815_815714

theorem sum_of_fractions :
  (3 / 9) + (6 / 12) = 5 / 6 := by
  sorry

end sum_of_fractions_l815_815714


namespace count_integers_satisfying_inequality_l815_815447

theorem count_integers_satisfying_inequality : ∃ (count : ℕ), count = 5 ∧ 
  ∀ x : ℤ, (0 ≤ x ∧ x ≤ 4) → ((x - 2) * (x - 2) ≤ 4) :=
begin
  existsi 5, 
  split,
  { refl },
  { intros x hx,
    cases hx with h1 h2,
    have : 0 ≤ (x - 2) ^ 2, from pow_two_nonneg (x - 2),
    exact le_antisymm this (show (x - 2) ^ 2 ≤ 4, by sorry) }
end

end count_integers_satisfying_inequality_l815_815447


namespace sum_of_sequence_l815_815494

def closest_int (x : ℝ) : ℕ := 
  if x - x.floor ≤ 0.5 then x.floor.to_nat else x.ceil.to_nat

def a_n (n : ℕ) : ℕ := closest_int (real.sqrt n)

theorem sum_of_sequence :
  (∑ i in finset.range 100 \u0001, \(mathbb{\frac{1}}\mathbb{a_n}(i+1)) = 19 := 
by
  sorry

end sum_of_sequence_l815_815494


namespace steak_and_egg_meal_cost_is_16_l815_815499

noncomputable def steak_and_egg_cost (x : ℝ) := 
  (x + 14) / 2 + 0.20 * (x + 14) = 21

theorem steak_and_egg_meal_cost_is_16 (x : ℝ) (h : steak_and_egg_cost x) : x = 16 := 
by 
  sorry

end steak_and_egg_meal_cost_is_16_l815_815499


namespace simplify_expression_l815_815773

theorem simplify_expression : ( (3 + 4 + 5 + 6) / 3 ) + ( (3 * 6 + 9) / 4 ) = 12.75 := by
  sorry

end simplify_expression_l815_815773


namespace larger_number_is_23_l815_815975

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815975


namespace work_problem_l815_815251

theorem work_problem (x : ℝ) (hx_pos : 0 < x) (h_total : (1/x + 3/x) = 1/30) : x / 3 = 40 :=
by
  have h1 : 4 / x = 1 / 30, by linarith
  have hx : x = 120, by
    field_simp at h1
    linarith
  linarith [hx]

end work_problem_l815_815251


namespace mike_total_spent_l815_815133

def shirt_cost (wallet_cost : ℝ) : ℝ := wallet_cost / 3
def wallet_cost (food_cost : ℝ) : ℝ := food_cost + 60
def discounted_shirt_cost (shirt_cost : ℝ) : ℝ := shirt_cost * 0.8

def food_cost : ℝ := 30

theorem mike_total_spent : 
  let w := wallet_cost food_cost in
  let s := shirt_cost w in
  let ds := discounted_shirt_cost s in
  food_cost + w + ds = 144 :=
by
  sorry

end mike_total_spent_l815_815133


namespace quadratic_inequality_solution_l815_815597

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 5*x + 6 ≤ 0) : 
  28 ≤ x^2 + 7*x + 10 ∧ x^2 + 7*x + 10 ≤ 40 :=
sorry

end quadratic_inequality_solution_l815_815597


namespace exchanged_teaspoons_tablespoons_cost_in_euros_l815_815315

theorem exchanged_teaspoons_tablespoons_cost_in_euros :
  let teaspoon_cost_dollars : ℝ := 1.50
  let tablespoon_cost_dollars : ℝ := 2.25
  let conversion_rate_eur_per_dollar : ℝ := 0.85
  let num_dessert_spoons : ℕ := 7
  let total_cost_dollars := (num_dessert_spoons * teaspoon_cost_dollars) + 
                            (num_dessert_spoons * tablespoon_cost_dollars)
  let total_cost_euros := total_cost_dollars * conversion_rate_eur_per_dollar
  let rounded_total_cost_euros := Real.round (total_cost_euros * 100) / 100
  rounded_total_cost_euros = 22.31 :=
by
  sorry

end exchanged_teaspoons_tablespoons_cost_in_euros_l815_815315


namespace language_grouping_possible_l815_815855

-- Definitions for the numbers of students speaking different combinations of languages
variables {E F S EF ES FS EFS : ℕ}

-- Conditions given in the problem
def condition1 := E + ES + EF + EFS = 50
def condition2 := F + EF + FS + EFS = 50
def condition3 := S + ES + FS + EFS = 50

-- Theorem statement
theorem language_grouping_possible 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3) 
  : ∃ (group1 group2 group3 group4 group5 : list ℕ), 
      (∀ (group : list ℕ), group ∈ [group1, group2, group3, group4, group5] → 
      list.count group 1 = 10 ∧ 
      list.count group 2 = 10 ∧ 
      list.count group 3 = 10) := sorry

end language_grouping_possible_l815_815855


namespace common_divisor_seven_l815_815749

-- Definition of numbers A, B, and C based on given conditions
def A (m n : ℤ) : ℤ := n^2 + 2 * m * n + 3 * m^2 + 2
def B (m n : ℤ) : ℤ := 2 * n^2 + 3 * m * n + m^2 + 2
def C (m n : ℤ) : ℤ := 3 * n^2 + m * n + 2 * m^2 + 1

-- The proof statement ensuring A, B and C have a common divisor of 7
theorem common_divisor_seven (m n : ℤ) : ∃ d : ℤ, d > 1 ∧ d ∣ A m n ∧ d ∣ B m n ∧ d ∣ C m n → d = 7 :=
by
  sorry

end common_divisor_seven_l815_815749


namespace enchilada_taco_cost_l815_815923

variables (e t : ℝ)

theorem enchilada_taco_cost 
  (h1 : 4 * e + 5 * t = 4.00) 
  (h2 : 5 * e + 3 * t = 3.80) 
  (h3 : 7 * e + 6 * t = 6.10) : 
  4 * e + 7 * t = 4.75 := 
sorry

end enchilada_taco_cost_l815_815923


namespace sum_of_digits_of_minimal_N_l815_815309

theorem sum_of_digits_of_minimal_N : 
  ∃ N: ℕ, 0 ≤ N ∧ N ≤ 499 ∧ 8 * N + 450 < 500 ∧ nat.digits 10 N = [6] := sorry

end sum_of_digits_of_minimal_N_l815_815309


namespace range_of_m_l815_815404

noncomputable def f (m x : ℝ) : ℝ := x^3 + m * x^2 + (m + 6) * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, ∃ c : ℝ, (f m x) ≤ c ∧ (f m y) ≥ (f m x) ∧ ∀ z : ℝ, f m z ≥ f m x ∧ f m z ≤ c) ↔ (m < -3 ∨ m > 6) :=
by
  sorry

end range_of_m_l815_815404


namespace find_four_digit_square_l815_815747

noncomputable def is_four_digit_sq_eq_aabb (N : ℕ) : Prop := 
  ∃ (a b : ℕ), N = 1100 * a + 11 * b ∧ N = 7744

theorem find_four_digit_square :
  ∃ (N : ℕ), N = 7744 ∧ is_four_digit_sq_eq_aabb N :=
begin
  sorry
end

end find_four_digit_square_l815_815747


namespace dot_product_of_vectors_l815_815813

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-1, 1) - vector_a

theorem dot_product_of_vectors :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = -4 :=
by
  sorry

end dot_product_of_vectors_l815_815813


namespace total_area_correct_l815_815098

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase : ℕ := 2
def num_extra_rooms_same_size : ℕ := 3
def num_double_size_rooms : ℕ := 1

def increased_length : ℕ := initial_length + increase
def increased_width : ℕ := initial_width + increase

def area_of_one_room (length width : ℕ) : ℕ := length * width

def number_of_rooms : ℕ := 1 + num_extra_rooms_same_size

def total_area_same_size_rooms : ℕ := number_of_rooms * area_of_one_room increased_length increased_width

def total_area_extra_size_room : ℕ := num_double_size_rooms * 2 * area_of_one_room increased_length increased_width

def total_area : ℕ := total_area_same_size_rooms + total_area_extra_size_room

theorem total_area_correct : total_area = 1800 := by
  -- Proof omitted
  sorry

end total_area_correct_l815_815098


namespace bicycle_weight_l815_815569

theorem bicycle_weight (b s : ℕ) (h1 : 10 * b = 5 * s) (h2 : 5 * s = 200) : b = 20 := 
by 
  sorry

end bicycle_weight_l815_815569


namespace simplify_expression_l815_815929

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end simplify_expression_l815_815929


namespace cannot_be_altitudes_of_tetrahedron_l815_815296

-- Define the four sets given in the problem
def setA := {1, (3 * Real.sqrt 2) / 2, (3 * Real.sqrt 2) / 2, (3 * Real.sqrt 2) / 2}
def setB := {4, (25 * Real.sqrt 3) / 3, (25 * Real.sqrt 3) / 3, (25 * Real.sqrt 3) / 3}
def setC := {Real.sqrt 5, (Real.sqrt 30) / 2, (Real.sqrt 30) / 2, (Real.sqrt 30) / 2}
def setD := {2, (6 * Real.sqrt 5) / 5, (6 * Real.sqrt 5) / 5, (6 * Real.sqrt 5) / 5}

-- Define the condition that the product of face area and height is equal for all faces.
axiom face_area_height_product (S1 S2 S3 S4 h1 h2 h3 h4 : ℝ) : (S1 * h1) = (S2 * h2) = (S3 * h3) = (S4 * h4)

-- Define the fact that setB cannot be the altitudes of a tetrahedron
theorem cannot_be_altitudes_of_tetrahedron : ¬(∃ S1 S2 S3 S4 : ℝ,
  face_area_height_product S1 S2 S3 S4
  (4) ((25 * Real.sqrt 3) / 3) ((25 * Real.sqrt 3) / 3) ((25 * Real.sqrt 3) / 3)) :=
by
  sorry

end cannot_be_altitudes_of_tetrahedron_l815_815296


namespace intersection_point_l815_815257

variable (x y z : ℝ) (t : ℝ)

def parametric_line : Prop :=
  x = 1 + t ∧ y = 0 ∧ z = -3 + 2 * t

def plane : Prop :=
  2 * x - y + 4 * z = 0

theorem intersection_point : parametric_line x y z t ∧ plane x y z → 
  (x = 2 ∧ y = 0 ∧ z = -1) :=
by
  intro h
  cases h with h_line h_plane
  sorry

end intersection_point_l815_815257


namespace overall_gain_percentage_l815_815668

theorem overall_gain_percentage
  (SP_gadget : ℝ) (Profit_gadget : ℝ) 
  (SP_book : ℝ) (Profit_book : ℝ) 
  (SP_furniture : ℝ) (Profit_furniture : ℝ) 
  (h_gadget : SP_gadget = 200) (h_profit_gadget : Profit_gadget = 50)
  (h_book : SP_book = 30) (h_profit_book : Profit_book = 10)
  (h_furniture : SP_furniture = 350) (h_profit_furniture : Profit_furniture = 70) :
  ( (SP_gadget - Profit_gadget + SP_book - Profit_book + SP_furniture - Profit_furniture) / 
    (SP_gadget + SP_book + SP_furniture - Profit_gadget - Profit_book - Profit_furniture) * 100 ) 
  ≈ 28.89 := 
begin
  sorry
end

end overall_gain_percentage_l815_815668


namespace non_empty_subsets_satisfying_criteria_l815_815456

theorem non_empty_subsets_satisfying_criteria :
  (∑ m in Finset.range 11 \ Finset.singleton 0, ((20 - 2 * (m - 1)) choose m)) = 2163 :=
by
  sorry

end non_empty_subsets_satisfying_criteria_l815_815456


namespace total_pieces_of_pizza_l815_815735

def pieces_per_pizza : ℕ := 6
def pizzas_per_fourthgrader : ℕ := 20
def fourthgraders_count : ℕ := 10

theorem total_pieces_of_pizza :
  fourthgraders_count * (pieces_per_pizza * pizzas_per_fourthgrader) = 1200 :=
by
  /-
  We have:
  - pieces_per_pizza = 6
  - pizzas_per_fourthgrader = 20
  - fourthgraders_count = 10

  Therefore,
  10 * (6 * 20) = 1200
  -/
  sorry

end total_pieces_of_pizza_l815_815735


namespace annie_hair_decorations_l815_815705

theorem annie_hair_decorations :
  let barrettes := 100
  let scrunchies := 4 * barrettes
  let bobby_pins := 3 * barrettes - 17
  let total_decorations := barrettes + scrunchies + bobby_pins
  let percentage_bobby_pins := (bobby_pins.toReal / total_decorations.toReal) * 100
  percentage_bobby_pins ≈ 36 := 
  -- proof here
  sorry

end annie_hair_decorations_l815_815705


namespace semi_circle_area_for_2_3_inscribed_rectangle_l815_815665

def rectangle_inscribed_semi_circle_area
  (w h : ℕ) (hw : w = 2) (hh : h = 3)
  (longer_side_on_diameter : Prop) : ℝ :=
  let d := sqrt (w^2 + h^2) in
  let r := d / 2 in
  let area_circle := π * (r^2) in
  let area_semi_circle := area_circle / 2 in
  area_semi_circle

theorem semi_circle_area_for_2_3_inscribed_rectangle :
  rectangle_inscribed_semi_circle_area 2 3 (by rfl) (by rfl) True = 13 * π / 8 :=
by
  sorry

end semi_circle_area_for_2_3_inscribed_rectangle_l815_815665


namespace cube_set_closed_under_multiplication_l815_815526

def is_cube (n : ℕ) : Prop := ∃ k : ℕ, k ^ 3 = n

def cube_set : Set ℕ := {n | is_cube n}

theorem cube_set_closed_under_multiplication :
  ∀ a b ∈ cube_set, (a * b) ∈ cube_set :=
by
  intros a ha b hb
  obtain ⟨ka, ha_cube⟩ := ha
  obtain ⟨kb, hb_cube⟩ := hb
  use ka * kb
  rw [← ha_cube, ← hb_cube, mul_pow]
  sorry

end cube_set_closed_under_multiplication_l815_815526


namespace RoqueBikesTwiceAWeek_l815_815878

-- Definitions based on conditions
def hoursWalking := 12 -- 3 walks of 4 hours each
def totalCommuteTime := 16 -- total hours spent commuting

-- Proof statement
theorem RoqueBikesTwiceAWeek (x : ℕ) : 2 * x + hoursWalking = totalCommuteTime → x = 2 :=
by
  intro h,
  sorry

end RoqueBikesTwiceAWeek_l815_815878


namespace total_goals_in_five_matches_is_4_l815_815272

theorem total_goals_in_five_matches_is_4
    (A : ℚ) -- defining the average number of goals before the fifth match as rational
    (h1 : A * 4 + 2 = (A + 0.3) * 5) : -- condition representing total goals equation
    4 = (4 * A + 2) := -- statement that the total number of goals in 5 matches is 4
by
  sorry

end total_goals_in_five_matches_is_4_l815_815272


namespace range_of_m_l815_815825

def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

theorem range_of_m (m : ℝ) (h : (A m) ∩ B ≠ ∅) : m ≤ -1 :=
sorry

end range_of_m_l815_815825


namespace fourth_hexagon_dots_l815_815610

def dots_in_hexagon (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 1 + (12 * (n * (n + 1) / 2))

theorem fourth_hexagon_dots : dots_in_hexagon 4 = 85 :=
by
  unfold dots_in_hexagon
  norm_num
  sorry

end fourth_hexagon_dots_l815_815610


namespace thirteen_pow_five_mod_seven_l815_815564

theorem thirteen_pow_five_mod_seven :
  ∃ m : ℤ, (13^5 ≡ m [MOD 7]) ∧ (0 ≤ m) ∧ (m < 7) ∧ (m = 6) := by
  sorry

end thirteen_pow_five_mod_seven_l815_815564


namespace baron_theorem_correct_l815_815305

theorem baron_theorem_correct (n a b : ℕ) (P : Polynomial ℕ)
  (roots : Fin n → ℕ)
  (hP : ∀ x, P.eval x = ∏ i, (x - roots i))
  (h_sum_root: a = (Finset.univ : Finset (Fin n)).sum (λ i, roots i))
  (h_prod_root: b = (Finset.univ : Finset (Fin n)).sum (λ(i_j : Fin n × Fin n), if i_j.1 < i_j.2 then roots i_j.1 * roots i_j.2 else 0)):
  b = (a * (a - 1)) / 2 := 
by
  sorry

end baron_theorem_correct_l815_815305


namespace cricket_innings_l815_815673

theorem cricket_innings (n : ℕ) 
  (avg_run_inn : n * 36 = n * 36)  -- average runs is 36 (initially true for any n)
  (increase_avg_by_4 : (36 * n + 120) / (n + 1) = 40) : 
  n = 20 := 
sorry

end cricket_innings_l815_815673


namespace red_envelope_scenarios_l815_815848

/-- In a WeChat group of five people (A, B, C, D, E), there are four red envelopes to be grabbed. 
    Each person can grab at most one envelope.
    The four red envelopes consist of two 2-yuan envelopes, one 3-yuan envelope, and one 4-yuan envelope.
    Prove that there are 36 scenarios where both A and B grab a red envelope. --/
theorem red_envelope_scenarios :
  let people := ['A', 'B', 'C', 'D', 'E'] in
  let envelopes := [2, 2, 3, 4] in
  ∃ (n : ℕ), (n = 36)
:=
sorry

end red_envelope_scenarios_l815_815848


namespace max_sum_ac_bc_l815_815474

noncomputable def triangle_ab_bc_sum_max (AB : ℝ) (C : ℝ) : ℝ :=
  if AB = Real.sqrt 6 - Real.sqrt 2 ∧ C = Real.pi / 6 then 4 else 0

theorem max_sum_ac_bc {A B C : ℝ} (h1 : AB = Real.sqrt 6 - Real.sqrt 2) (h2 : C = Real.pi / 6) :
  triangle_ab_bc_sum_max AB C = 4 :=
by {
  sorry
}

end max_sum_ac_bc_l815_815474


namespace integer_solutions_to_inequality_l815_815433

theorem integer_solutions_to_inequality :
  ∃ n : ℕ, n = 5 ∧ (∀ x : ℤ, (x - 2) ^ 2 ≤ 4 ↔ x ∈ {0, 1, 2, 3, 4}) :=
by
  sorry

end integer_solutions_to_inequality_l815_815433


namespace intersections_line_segment_l815_815302

def intersects_count (a b : ℕ) (x y : ℕ) : ℕ :=
  let steps := gcd x y
  2 * (steps + 1)

theorem intersections_line_segment (x y : ℕ) (h_x : x = 501) (h_y : y = 201) :
  intersects_count 1 1 x y = 336 := by
  sorry

end intersections_line_segment_l815_815302


namespace parabola_intersects_line_at_one_point_l815_815007

theorem parabola_intersects_line_at_one_point (α : ℝ) (h : ∀ x : ℝ, (3 * x^2 + 1 = 4 * (sin α) * x) → α = 60) : α = 60 :=
  sorry

end parabola_intersects_line_at_one_point_l815_815007


namespace average_growth_rate_bing_dwen_dwen_l815_815173

noncomputable def sales_growth_rate (v0 v2 : ℕ) (x : ℝ) : Prop :=
  (1 + x) ^ 2 = (v2 : ℝ) / (v0 : ℝ)

theorem average_growth_rate_bing_dwen_dwen :
  ∀ (v0 v2 : ℕ) (x : ℝ),
    v0 = 10000 →
    v2 = 12100 →
    sales_growth_rate v0 v2 x →
    x = 0.1 :=
by
  intros v0 v2 x h₀ h₂ h_growth
  sorry

end average_growth_rate_bing_dwen_dwen_l815_815173


namespace time_to_cross_signal_pole_approx_l815_815264

-- Definitions based on conditions
def L_train : ℝ := 300
def L_platform : ℝ := 535.7142857142857
def T_platform : ℝ := 39

-- Definition for speed of the train
def V_train : ℝ := (L_train + L_platform) / T_platform

-- Target function to calculate time to cross the signal pole
def T_pole : ℝ := L_train / V_train

-- Statement to be proven
theorem time_to_cross_signal_pole_approx : abs (T_pole - 14) < 0.001 := sorry

end time_to_cross_signal_pole_approx_l815_815264


namespace find_line_and_b_coordinates_l815_815415

theorem find_line_and_b_coordinates :
  ∃ (B : ℝ × ℝ) (AC_eqn : ℝ → ℝ → Prop), 
    (AC_eqn = (λ x y, 2 * x + y - 11 = 0)) ∧ 
    (B = (-1, -3)) ∧ 
    (let A : ℝ × ℝ := (5, 1), 
         median_CM_eqn := λ x y, 2 * x - y - 5 = 0, 
         altitude_BH_eqn := λ x y, x - 2 * y - 5 = 0 
     in True) := 
sorry

end find_line_and_b_coordinates_l815_815415


namespace Mark_running_speed_l815_815127

theorem Mark_running_speed
    (x : ℝ)
    (h_biking : 15 / (3 * x + 2))
    (h_running : 3 / x)
    (h_total_time : h_biking + h_running = 1.47) :
    x ≈ 5.05 :=
by
  sorry

end Mark_running_speed_l815_815127


namespace sqrt_inequality_l815_815510

variable (x y z : ℝ)

-- All given conditions
def conditions := x > 1 ∧ y > 1 ∧ z > 1 ∧ (1 / x + 1 / y + 1 / z = 2)

-- The theorem statement to be proved
theorem sqrt_inequality (hx : x > 1) (hy : y > 1) (hz : z > 1) 
(hsum : 1 / x + 1 / y + 1 / z = 2) : 
sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) ≤ sqrt (x + y + z) := by sorry

end sqrt_inequality_l815_815510


namespace quadratic_inequality_solution_sets_l815_815026

theorem quadratic_inequality_solution_sets (a b c : ℝ) (h1 : a < 0) (h2 : b = 3 * a) (h3 : c = 2 * a) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2 ⊆ {x : ℝ | ax^2 - bx + c ≥ 0} →
  {x : ℝ | cx^2 + bx + a ≤ 0} = {x : ℝ | x ≤ -1 ∨ x ≥ -1/2} :=
by
  -- Proof goes here
  sorry

end quadratic_inequality_solution_sets_l815_815026


namespace intersection_unique_point_l815_815506

noncomputable def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 12*x + 20

theorem intersection_unique_point :
  ∃ (c d : ℝ), g(c) = c ∧ d = c ∧ (c, d) = (-4, -4) :=
by
  sorry

end intersection_unique_point_l815_815506


namespace quadratic_equation_nonzero_coefficient_l815_815810

theorem quadratic_equation_nonzero_coefficient (m : ℝ) : 
  m - 1 ≠ 0 ↔ m ≠ 1 :=
by
  sorry

end quadratic_equation_nonzero_coefficient_l815_815810


namespace total_people_in_house_l815_815639

-- Define the number of people in various locations based on the given conditions.
def charlie_and_susan := 2
def sarah_and_friends := 5
def people_in_bedroom := charlie_and_susan + sarah_and_friends
def people_in_living_room := 8

-- Prove the total number of people in the house is 14.
theorem total_people_in_house : people_in_bedroom + people_in_living_room = 14 := by
  -- Here we can use Lean's proof system, but we skip with 'sorry'
  sorry

end total_people_in_house_l815_815639


namespace father_age_three_times_xiaojun_after_years_l815_815609

theorem father_age_three_times_xiaojun_after_years (years_passed : ℕ) (xiaojun_current_age : ℕ) (father_current_age : ℕ) 
  (h1 : xiaojun_current_age = 5) (h2 : father_current_age = 31) (h3 : years_passed = 8) :
  father_current_age + years_passed = 3 * (xiaojun_current_age + years_passed) := by
  sorry

end father_age_three_times_xiaojun_after_years_l815_815609


namespace tank_capacity_l815_815691

variable (x : ℝ) -- Total capacity of the tank

theorem tank_capacity (h1 : x / 8 = 120 / (1 / 2 - 1 / 8)) :
  x = 320 :=
by
  sorry

end tank_capacity_l815_815691


namespace larger_number_is_23_l815_815988

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815988


namespace triangle_area_l815_815227

theorem triangle_area :
  ∀ (a b c : ℝ), a = 6 → b = 4 → c = 5 →
  let s := (a + b + c) / 2 in
  (√(s * (s - a) * (s - b) * (s - c)) = √98.4375) :=
by sorry

end triangle_area_l815_815227


namespace sum_of_numbers_greater_than_or_equal_to_0_1_l815_815199

def num1 : ℝ := 0.8
def num2 : ℝ := 0.5  -- converting 1/2 to 0.5
def num3 : ℝ := 0.6

def is_greater_than_or_equal_to_0_1 (n : ℝ) : Prop :=
  n ≥ 0.1

theorem sum_of_numbers_greater_than_or_equal_to_0_1 :
  is_greater_than_or_equal_to_0_1 num1 ∧ 
  is_greater_than_or_equal_to_0_1 num2 ∧ 
  is_greater_than_or_equal_to_0_1 num3 →
  num1 + num2 + num3 = 1.9 :=
by
  sorry

end sum_of_numbers_greater_than_or_equal_to_0_1_l815_815199


namespace condition_1_condition_2_l815_815894

def f (x : ℝ) : ℝ := Real.sin x

theorem condition_1 (θ : ℝ) (hθ : θ ∈ Set.Ico 0 (2 * Real.pi)) :
  (∀ x : ℝ, f (x + θ) = f (-x + θ)) ↔ (θ = Real.pi / 2 ∨ θ = 3 * Real.pi / 2) :=
sorry

theorem condition_2 : Set.Icc
  (1 - Real.sqrt 3 / 2)
  (1 + Real.sqrt 3 / 2) =
  Set.range (λ x : ℝ, (Real.sin (x + Real.pi / 12))^2 + (Real.sin (x + Real.pi / 4))^2) :=
sorry

end condition_1_condition_2_l815_815894


namespace num_permutations_with_exactly_two_lifters_l815_815344

theorem num_permutations_with_exactly_two_lifters :
  let n := 2013 in
  let permutations_with_two_lifters : ℕ := 3^n - (n + 1) * 2^n + n * (n + 1) / 2 in
  true := permutations_with_two_lifters = 3^2013 - 2014 * 2^2013 + 2013 * 2014 / 2 :=
by
  sorry

end num_permutations_with_exactly_two_lifters_l815_815344


namespace balloons_total_l815_815466

theorem balloons_total (b_curr : ℕ) (b_add : ℕ) 
                       (t_curr : ℕ) (t_add : ℕ) 
                       (t_fraction_pop : ℕ) (b_fraction_giveaway : ℕ)
                       (remove_balloons : ℕ) 
                       (b_curr_eq : b_curr = 25)
                       (b_add_eq : b_add = 22)
                       (t_curr_eq : t_curr = 16)
                       (t_add_eq : t_add = 42)
                       (t_fraction_pop_eq : t_fraction_pop = 2)
                       (b_fraction_giveaway_eq : b_fraction_giveaway = 1)
                       (fraction_pop_base : ℕ) (fraction_giveaway_base : ℕ)
                       (fraction_pop_base_eq : fraction_pop_base = 5)
                       (fraction_giveaway_base_eq : fraction_giveaway_base = 4)
                       (remove_balloons_eq : remove_balloons = 5) :
  let
    b_initial := b_curr + b_add,
    t_initial := t_curr + t_add,
    t_pop := (t_fraction_pop * t_initial) / fraction_pop_base,
    b_giveaway := (b_fraction_giveaway * b_initial) / fraction_giveaway_base,
    t_after_pop := t_initial - t_pop,
    b_after_giveaway := b_initial - b_giveaway,
    t_final := t_after_pop - remove_balloons,
    b_final := b_after_giveaway - remove_balloons
  in
    t_final + b_final = 61 := by sorry

end balloons_total_l815_815466


namespace n_times_s_l815_815892

noncomputable def f (x : ℝ) : ℝ := sorry

theorem n_times_s : (f 0 = 0 ∨ f 0 = 1) ∧
  (∀ (y : ℝ), f 0 = 0 → False) ∧
  (∀ (x y : ℝ), f x * f y - f (x * y) = x^2 + y^2) → 
  let n : ℕ := if f 0 = 0 then 1 else 1
  let s : ℝ := if f 0 = 0 then 0 else 1
  n * s = 1 :=
by
  sorry

end n_times_s_l815_815892


namespace first_year_after_2020_with_digit_sum_7_l815_815846

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2020_with_digit_sum_7 :
  ∀ n : ℕ, n > 2020 → sum_of_digits n = 7 → n = 2023 :=
by
  intro n hn hsum
  have := Classical.dec (n < 2023)
  cases this
  case is_false =>
    have := Classical.dec (n > 2023)
    cases this
    case is_false =>
      contradiction
    case is_true =>
      sorry
  case is_true =>
    sorry

end first_year_after_2020_with_digit_sum_7_l815_815846


namespace Rockham_Soccer_League_Members_l815_815130

-- conditions
def sock_cost : ℕ := 4
def tshirt_cost : ℕ := sock_cost + 5
def total_cost_per_member : ℕ := 2 * sock_cost + 2 * tshirt_cost
def total_cost : ℕ := 2366
def number_of_members (n : ℕ) : Prop := n * total_cost_per_member = total_cost

-- statement to prove
theorem Rockham_Soccer_League_Members : ∃ n : ℕ, number_of_members n ∧ n = 91 :=
by {
  use 91,
  show number_of_members 91,
  sorry
}

end Rockham_Soccer_League_Members_l815_815130


namespace integer_solution_count_l815_815451

theorem integer_solution_count : 
  let condition := ∀ x : ℤ, (x - 2) ^ 2 ≤ 4
  ∃ count : ℕ, count = 5 ∧ (∀ x : ℤ, condition → (0 ≤ x ∧ x ≤ 4)) := 
sorry

end integer_solution_count_l815_815451


namespace nonneg_int_solutions_eq_l815_815657

theorem nonneg_int_solutions_eq (a b : ℕ) : a^2 + b^2 = 841 * (a * b + 1) ↔ (a = 0 ∧ b = 29) ∨ (a = 29 ∧ b = 0) :=
by {
  sorry -- Proof omitted
}

end nonneg_int_solutions_eq_l815_815657


namespace unique_integer_m_l815_815348

theorem unique_integer_m :
  ∃! (m : ℤ), m - ⌊m / (2005 : ℝ)⌋ = 2005 :=
by
  --- Here belongs the proof part, but we leave it with a sorry
  sorry

end unique_integer_m_l815_815348


namespace minimum_value_varphi_a_eq_1_range_of_a_if_varphi_has_zero_point_exp_x_geq_one_add_x_add_one_sixth_x_cubed_l815_815122

noncomputable def varphi (a x : ℝ) : ℝ := exp x - 1 - a * x

-- (I) When a = 1, find the minimum value of function varphi(x)
theorem minimum_value_varphi_a_eq_1 : 
  ∃ x : ℝ, varphi 1 x = 0 := sorry

-- (II) If the function varphi(x) has a zero point in (0, +∞), find the range of the real number a
theorem range_of_a_if_varphi_has_zero_point (a : ℝ) (h : ∃ x ∈ Set.Ioi 0, varphi a x = 0) : 
  1 < a := sorry

-- (III) Prove the inequality e^x ≥ 1 + x + 1/6 * x^3 for x ∈ ℝ
theorem exp_x_geq_one_add_x_add_one_sixth_x_cubed (x : ℝ) : 
  exp x ≥ 1 + x + 1/6 * x^3 := sorry

end minimum_value_varphi_a_eq_1_range_of_a_if_varphi_has_zero_point_exp_x_geq_one_add_x_add_one_sixth_x_cubed_l815_815122


namespace population_zalotis_15000_l815_815329

theorem population_zalotis_15000 :
  ∀ (year : ℕ), (year ≥ 1950 ∧ year ≤ 2125) →
    ∃ (n : ℕ), n ≥ 0 ∧ (year = 1950 + 35 * n ∧ 750 * 2^n ≥ 15000) :=
begin
  sorry
end

end population_zalotis_15000_l815_815329


namespace sequence_mono_iff_b_gt_neg3_l815_815528

theorem sequence_mono_iff_b_gt_neg3 (b : ℝ) : 
  (∀ n : ℕ, 1 ≤ n → (n + 1) ^ 2 + b * (n + 1) > n ^ 2 + b * n) → b > -3 := 
by
  sorry

end sequence_mono_iff_b_gt_neg3_l815_815528


namespace gcd_largest_of_forms_l815_815258

theorem gcd_largest_of_forms (a b : ℕ) (h1 : a ≠ b) (h2 : a < 10) (h3 : b < 10) :
  Nat.gcd (100 * a + 11 * b) (101 * b + 10 * a) = 45 :=
by
  sorry

end gcd_largest_of_forms_l815_815258


namespace bug_return_probability_18_moves_l815_815265

def recurrence (P: ℕ → ℚ) (n: ℕ) : Prop :=
  P 0 = 1 ∧ (∀ n > 0, P n = (1/2) * (1 - P (n - 1)))

theorem bug_return_probability_18_moves (P : ℕ → ℚ) (m n : ℕ) 
  (h : recurrence P 18) 
  (prime : ∀ (a b: ℕ), m = a → n = b → nat.coprime a b) 
  (result : P 18 = (m : ℚ) / n) : 
    m + n = m + n :=
sorry

end bug_return_probability_18_moves_l815_815265


namespace minimum_value_minimum_value_achieved_l815_815624

theorem minimum_value :
  ∀ x y z : ℝ, (x^2 * y - 1)^2 + (x + y + z)^2 ≥ 1 :=
begin
  sorry
end

theorem minimum_value_achieved :
  ∃ x y z : ℝ, (x^2 * y - 1)^2 + (x + y + z)^2 = 1 :=
begin
  use [0, 0, 0],
  simp,
end

end minimum_value_minimum_value_achieved_l815_815624


namespace f_84_equals_997_l815_815168

def f : ℕ → ℕ
| n := if n >= 1000 then n - 3 else f (f (n + 5))

theorem f_84_equals_997 : f 84 = 997 := 
by 
    sorry

end f_84_equals_997_l815_815168


namespace part1_solution_set_part2_range_x_l815_815781

def f (x : ℝ) : ℝ := |2 * x - 1| + |5 * x - 1|

def m (n : ℝ) : ℝ := 2 - n

theorem part1_solution_set :
  {(x : ℝ) | f(x) > x + 1} = set.Iio (1 / 8) ∪ set.Ioi (1 / 2) := sorry

theorem part2_range_x (n : ℝ) (hn : n > 0) (hm : 2 - n > 0) :
  {x : ℝ | (1 / m n) + (4 / n) ≥ f(x) } = set.Ioo (-5 / 14) (13 / 14) := sorry

end part1_solution_set_part2_range_x_l815_815781


namespace value_of_dimes_in_bag_l815_815680

-- Definitions for the conditions
def loonie_value : ℝ := 1.0
def dime_value : ℝ := 0.10
def loonie_mass : ℝ := 1.0
def dime_mass : ℝ := loonie_mass / 4
def bag_of_loonies_value : ℝ := 400.0

-- The proof goal
theorem value_of_dimes_in_bag :
  let number_of_loonies := bag_of_loonies_value / loonie_value in
  let equivalent_dimes := number_of_loonies * (loonie_mass / dime_mass) in
  equivalent_dimes * dime_value = 160.0 :=
by
  sorry

end value_of_dimes_in_bag_l815_815680


namespace inverse_proposition_equivalence_l815_815954

theorem inverse_proposition_equivalence (x y : ℝ) :
  (x = y → abs x = abs y) ↔ (abs x = abs y → x = y) :=
sorry

end inverse_proposition_equivalence_l815_815954


namespace Dmitry_socks_l815_815327

theorem Dmitry_socks :
  ∃ x : ℕ,
    let initial_blue := 6 in
    let initial_black := 18 in
    let initial_white := 12 in
    let initial_total := initial_blue + initial_black + initial_white in
    let new_black := initial_black + x in
    let new_total := initial_total + x in
    (new_black : ℚ) / new_total = 3 / 5 ∧ x = 9 :=
by sorry

end Dmitry_socks_l815_815327


namespace ellipse_standard_equation_l815_815598

theorem ellipse_standard_equation :
  ∃ (a b c : ℝ),
    2 * a = 10 ∧
    c / a = 3 / 5 ∧
    b^2 = a^2 - c^2 ∧
    (∀ x y : ℝ, (x^2 / 16) + (y^2 / 25) = 1) :=
by
  sorry

end ellipse_standard_equation_l815_815598


namespace common_root_for_permutations_of_coeffs_l815_815468

theorem common_root_for_permutations_of_coeffs :
  ∀ (a b c d : ℤ), (a = -7 ∨ a = 4 ∨ a = -3 ∨ a = 6) ∧ 
                   (b = -7 ∨ b = 4 ∨ b = -3 ∨ b = 6) ∧
                   (c = -7 ∨ c = 4 ∨ c = -3 ∨ c = 6) ∧
                   (d = -7 ∨ d = 4 ∨ d = -3 ∨ d = 6) ∧
                   (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (a * 1^3 + b * 1^2 + c * 1 + d = 0) :=
by
  intros a b c d h
  sorry

end common_root_for_permutations_of_coeffs_l815_815468


namespace count_integer_solutions_l815_815425

theorem count_integer_solutions :
  {x : ℤ | (x - 2)^2 ≤ 4}.card = 5 :=
by
  sorry

end count_integer_solutions_l815_815425


namespace ratio_not_necessarily_constant_l815_815596

theorem ratio_not_necessarily_constant (x y : ℝ) : ¬ (∃ k : ℝ, ∀ x y, x / y = k) :=
by
  sorry

end ratio_not_necessarily_constant_l815_815596


namespace max_distinct_numbers_condition_l815_815502

theorem max_distinct_numbers_condition (S : Finset ℝ) (h : ∀ x ∈ S, ∀ y z ∈ S, x ≠ y → x ≠ z → y ≠ z → x^2 > y * z) :
  Finset.card S ≤ 3 ∧ ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (S = {a, b, c}) := sorry

end max_distinct_numbers_condition_l815_815502


namespace angle_sum_BD_l815_815835

/-- Given the conditions that ∠A = 20° and ∠AFG = ∠AGF,
    prove that the sum of ∠B and ∠D is 80°. -/
theorem angle_sum_BD (A F G B D : Point) (h₁ : ∠ A = 20) (h₂ : ∠ AFG = ∠ AGF) : 
  ∠ B + ∠ D = 80° := 
by 
  sorry

end angle_sum_BD_l815_815835


namespace total_pizza_pieces_l815_815737

-- Definitions based on the conditions
def pieces_per_pizza : Nat := 6
def pizzas_per_student : Nat := 20
def number_of_students : Nat := 10

-- Statement of the theorem
theorem total_pizza_pieces :
  pieces_per_pizza * pizzas_per_student * number_of_students = 1200 :=
by
  -- Placeholder for the proof
  sorry

end total_pizza_pieces_l815_815737


namespace parallel_lines_m_condition_l815_815524

theorem parallel_lines_m_condition (m : ℝ) : 
  (∀ (x y : ℝ), (2 * x - m * y - 1 = 0) ↔ ((m - 1) * x - y + 1 = 0)) → m = 2 :=
by
  sorry

end parallel_lines_m_condition_l815_815524


namespace simplify_expression_l815_815895

theorem simplify_expression (x y z : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) (h₂ : z ≠ 0) 
  (h : x^2 + y^2 + z^2 = xy + yz + zx) : 
  (1 / (y^2 + z^2 - x^2)) + (1 / (x^2 + z^2 - y^2)) + (1 / (x^2 + y^2 - z^2)) = 3 / x^2 := 
by
  sorry

end simplify_expression_l815_815895


namespace isosceles_right_triangle_exists_point_on_EC_l815_815301

open EuclideanGeometry Real

variables {A B C D E M : Point}

noncomputable def is_isosceles_right_triangle (A B C : Point) : Prop :=
  (dist A B = dist A C) ∧ (dist B C = dist A C * sqrt 2)

theorem isosceles_right_triangle_exists_point_on_EC
  (c e : ℝ) (h1 : c ≠ e)
  (h2 : dist A C = c)
  (h3 : dist A B = dist B C)
  (h4 : ∃ θ : ℝ, dist (rotate_point θ A E) E = e)
  (h5 : dist (rotate_point θ A D) (point_halfway A E) = sqrt (2 * e^2)) :
  ∃ M : Point, MkSegment A C ∋ M ∧ is_isosceles_right_triangle B M D := sorry

end isosceles_right_triangle_exists_point_on_EC_l815_815301


namespace fran_speed_calculation_l815_815501

noncomputable def fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) : ℝ :=
  joann_speed * joann_time / fran_time

theorem fran_speed_calculation : 
  fran_speed 15 3 2.5 = 18 := 
by
  -- Remember to write down the proof steps if needed, currently we use sorry as placeholder
  sorry

end fran_speed_calculation_l815_815501


namespace solve_sqrt_equation_l815_815338

theorem solve_sqrt_equation :
  (∃ x : ℝ, sqrt x + 3 * sqrt (x^2 + 9 * x) + sqrt (x + 9) = 45 - 3 * x ∧ x = 729 / 144) :=
sorry

end solve_sqrt_equation_l815_815338


namespace sum_of_cubes_l815_815135

theorem sum_of_cubes (n : ℕ) : 
  (∑ k in Finset.range (n + 1), k ^ 3) = (n * (n + 1) / 2) ^ 2 := 
sorry

end sum_of_cubes_l815_815135


namespace sum_of_three_numbers_eq_16_l815_815254

variable {a b c : ℝ}

theorem sum_of_three_numbers_eq_16
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + c * a = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_eq_16_l815_815254


namespace integer_solutions_to_inequality_l815_815437

theorem integer_solutions_to_inequality :
  ∃ n : ℕ, n = 5 ∧ (∀ x : ℤ, (x - 2) ^ 2 ≤ 4 ↔ x ∈ {0, 1, 2, 3, 4}) :=
by
  sorry

end integer_solutions_to_inequality_l815_815437


namespace line_slope_l815_815723

noncomputable def parallelogram_points : list (ℝ × ℝ) :=
  [(15, 70), (15, 210), (45, 280), (45, 140)]

def line_through_origin (m n : ℝ) : linear_map ℝ (ℝ × ℝ) :=
  λ t, (t * n, t * m)

theorem line_slope (h : 15 * 6 =  cum_sum_parallel): 
  let m := 35
  let n := 6
  in (m : ℕ) + (n : ℕ) = 41 :=
by sorry

end line_slope_l815_815723


namespace unit_digit_of_product_l815_815654

theorem unit_digit_of_product :
  let unit_digit (n : Nat) := n % 10 in
  (unit_digit (624 * 708 * 913 * 463) = 8) :=
by
  sorry

end unit_digit_of_product_l815_815654


namespace arithmetic_sequence_terms_l815_815967

theorem arithmetic_sequence_terms (
  a : ℕ → ℚ
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0) -- arithmetic sequence condition
  (even_length : ∃ m : ℕ, ∃ n : ℕ, (2 * n = m) ∧ ((n : ℚ) = 4))
  (sum_odd : ∑ i in finset.range(4) | i%2=1, a i = 24)
  (sum_even : ∑ i in finset.range(4) | i%2=0, a i = 30)
  (last_diff_first : a (4 - 1) - a 0 = (21/2))
) : 2 * 4 = 8 :=
by
  sorry

end arithmetic_sequence_terms_l815_815967


namespace two_digit_number_swap_product_l815_815203

theorem two_digit_number_swap_product 
  (a b : ℕ) (a_ne_b : a ≠ b) (a_nonzero : 1 ≤ a) (a_max : a ≤ 9) 
  (b_nonzero : 1 ≤ b) (b_max : b ≤ 9) :
  let x := 10 * a + b,
      y := 10 * b + a,
      z := x * y in
  (z ≥ 100 ∧ z < 1000) ∧ ((z / 100) = (z % 10)) :=
sorry

end two_digit_number_swap_product_l815_815203


namespace find_a_in_triangle_l815_815875

variable (a b c : ℝ) (A B C : ℝ)

-- Given conditions
def condition_c : c = 3 := sorry
def condition_C : C = Real.pi / 3 := sorry
def condition_sinB : Real.sin B = 2 * Real.sin A := sorry

-- Theorem to prove
theorem find_a_in_triangle (hC : condition_C) (hc : condition_c) (hsinB : condition_sinB) :
  a = Real.sqrt 3 :=
sorry

end find_a_in_triangle_l815_815875


namespace part_one_part_two_range_l815_815418

/-
Definitions based on conditions from the problem:
- Given vectors ax = (\cos x, \sin x), bx = (3, - sqrt(3))
- Domain for x is [0, π]
--
- Prove if a + b is parallel to b, then x = 5π / 6
- Definition of function f(x), and g(x) based on problem requirements.
- Prove the range of g(x) is [-3, sqrt(3)]
-/

/-
Part (1):
Given ax + bx = (cos x + 3, sin x - sqrt(3)) is parallel to bx =  (3, - sqrt(3));
Prove that x = 5π / 6 under x ∈ [0, π].
-/
noncomputable def vector_ax (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_bx : ℝ × ℝ := (3, - Real.sqrt 3)

theorem part_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) 
  (h_parallel : (vector_ax x).1 + vector_bx.1 = (vector_ax x).2 + vector_bx.2) :
  x = 5 * Real.pi / 6 :=
  sorry

/-
Part (2):
Let f(x) = 3 cos x - sqrt(3) sin x.
The function g(x) = -2 sqrt(3) sin(1/2 x - 2π/3) is defined by shifting f(x) right by π/3 and doubling the horizontal coordinate.
Prove the range of g(x) is [-3, sqrt(3)].
-/
noncomputable def f (x : ℝ) := 3 * Real.cos x - Real.sqrt 3 * Real.sin x
noncomputable def g (x : ℝ) := -2 * Real.sqrt 3 * Real.sin (0.5 * x - 2 * Real.pi / 3)

theorem part_two_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  -3 ≤ g x ∧ g x ≤ Real.sqrt 3 :=
  sorry

end part_one_part_two_range_l815_815418


namespace probability_at_least_one_black_ball_l815_815479

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4
def selected_balls : ℕ := 4

theorem probability_at_least_one_black_ball :
  (∃ (p : ℚ), p = 13 / 14 ∧ 
  (number_of_ways_to_choose4_balls_has_at_least_1_black / number_of_ways_to_choose4_balls) = p) :=
by
  sorry

end probability_at_least_one_black_ball_l815_815479


namespace larger_number_l815_815969

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l815_815969


namespace rockham_soccer_league_members_count_l815_815131

def cost_per_pair_of_socks : Nat := 4
def additional_cost_per_tshirt : Nat := 5
def cost_per_tshirt : Nat := cost_per_pair_of_socks + additional_cost_per_tshirt

def pairs_of_socks_per_member : Nat := 2
def tshirts_per_member : Nat := 2

def total_cost_per_member : Nat :=
  pairs_of_socks_per_member * cost_per_pair_of_socks + tshirts_per_member * cost_per_tshirt

def total_cost_all_members : Nat := 2366
def total_members : Nat := total_cost_all_members / total_cost_per_member

theorem rockham_soccer_league_members_count : total_members = 91 :=
by
  -- Given steps in the solution, verify each condition and calculation.
  sorry

end rockham_soccer_league_members_count_l815_815131


namespace max_earnings_by_combining_piles_l815_815196

theorem max_earnings_by_combining_piles :
  ∃ (earnings : ℕ), 
    let card_count := 2018,
        operation_count := card_count - 1,
        max_earnings := (card_count - 1) * card_count / 2
    in earnings = max_earnings ∧ earnings = 2035153 :=
sorry

end max_earnings_by_combining_piles_l815_815196


namespace count_integer_solutions_l815_815424

theorem count_integer_solutions :
  {x : ℤ | (x - 2)^2 ≤ 4}.card = 5 :=
by
  sorry

end count_integer_solutions_l815_815424


namespace hyperbola_equation_focus_and_eccentricity_l815_815804

theorem hyperbola_equation_focus_and_eccentricity (a b : ℝ)
  (h_focus : ∃ c : ℝ, c = 1 ∧ (∃ c_squared : ℝ, c_squared = c ^ 2))
  (h_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 ∧ e = c / a)
  (h_b : b ^ 2 = c ^ 2 - a ^ 2) :
  5 * x^2 - (5 / 4) * y^2 = 1 :=
sorry

end hyperbola_equation_focus_and_eccentricity_l815_815804


namespace part1_part2_part3_l815_815900

-- Part (1): Prove that k = 1 given the function f(x) is odd
theorem part1 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) (h_odd : ∀ x : ℝ, f (a, k, x) = -f (a, k, -x)) :
  k = 1 := sorry

-- Part (2): Prove the range of x for f(x + 2) + f(3 - 2x) > 0 is (5, +∞)
theorem part2 (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) (h_lt1 : 0 < a ∧ a < 1)
  (h_odd : ∀ x : ℝ, f (a, 1, x) = -f (a, 1, -x)) :
  ∀ x : ℝ, f (a, 1, x + 2) + f (a, 1, 3 - 2x) > 0 ↔ x > 5 := sorry

-- Part (3): Prove that m = 25/12 given f(1) = 8/3 and the minimum value condition for g(x) on [1, +∞)
theorem part3 (a : ℝ) (m : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) (h_f1 : f (a, 1, 1) = 8 / 3)
  (h_min_val : ∀ x : ℝ, x ≥ 1 → g (a, m, x) ≥ -2) :
  m = 25 / 12 := sorry

-- Function definition for f
def f (a k x : ℝ) : ℝ := k * a ^ x - (a ^ (-x))

-- Function definition for g
def g (a m x : ℝ) : ℝ := a ^ (2 * x) + a ^ (-2 * x) - 2 * m * f (a, 1, x)

end part1_part2_part3_l815_815900


namespace x_intercept_of_tangent_line_l815_815360

def f (x : ℝ) : ℝ := 3^x + 2*x*((λ x, (Real.log 3)*3 + 2*x) (-3*(Real.log 3)))

theorem x_intercept_of_tangent_line :
  let f' := λ x, (Real.log 3) * 3^x + 2 * (-3 * (Real.log 3))
  let f_0 := f 0
  let f'_0 := f' 0
  let tangent_line := λ x, (λ y, y = -5*(Real.log 3)*x + 1)
  ∃ x, tangent_line x = 0 ∧ x = 1 / (5 * Real.log 3) :=
sorry

end x_intercept_of_tangent_line_l815_815360


namespace find_x_l815_815465

theorem find_x (x : ℝ) (h : 9 / (x + 4) = 1) : x = 5 :=
sorry

end find_x_l815_815465


namespace aluminum_weight_l815_815140

variable {weight_iron : ℝ}
variable {weight_aluminum : ℝ}
variable {difference : ℝ}

def weight_aluminum_is_correct (weight_iron weight_aluminum difference : ℝ) : Prop := 
  weight_iron = weight_aluminum + difference

theorem aluminum_weight 
  (H1 : weight_iron = 11.17)
  (H2 : difference = 10.33)
  (H3 : weight_aluminum_is_correct weight_iron weight_aluminum difference) : 
  weight_aluminum = 0.84 :=
sorry

end aluminum_weight_l815_815140


namespace triangle_construction_exists_l815_815137

theorem triangle_construction_exists 
  (O : Point) 
  (l m n : Line) 
  (M N P : Point) 
  (H1 : O ∈ l ∧ O ∈ m ∧ O ∈ n) -- three lines intersect at point O
  (H2 : ∃ A ∈ l, ∃ B ∈ m, ∃ C ∈ n, ∃ (H: collinear M N P), triangle A B C) -- conditions given M, N, P, and lines intersection
  : ∃ (A ∈ l) (B ∈ m) (C ∈ n), 
      (line_through A B).contains M ∧ 
      (line_through B C).contains N ∧ 
      (line_through C A).contains P :=
sorry

end triangle_construction_exists_l815_815137


namespace train_length_proof_l815_815692

def length_of_train (time_to_cross : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_to_cross

theorem train_length_proof :
  length_of_train 20 125.99999999999999 = 700 :=
by
  -- calculations to convert and multiply values
  sorry

end train_length_proof_l815_815692


namespace tamika_always_wins_l815_815568

-- Define the sets for Tamika and Carlos
def tamika_set : List ℕ := [6, 7, 8]
def carlos_set : List ℕ := [2, 3, 5]

-- Define the function to compute the products of all pairs of distinct elements from a set
def products (s : List ℕ) : List ℕ :=
  s.product s |>.filter (λ p => p.fst ≠ p.snd) |>.map (λ p => p.fst * p.snd)

-- Calculate the products for Tamika and Carlos
def tamika_products : List ℕ := products tamika_set
def carlos_products : List ℕ := products carlos_set

-- Define the proof problem
theorem tamika_always_wins :
  (∀ t ∈ tamika_products, ∀ c ∈ carlos_products, t > c) →
  /- Probability calculation logic leading to -/ 
  (sum (map (λ c, count (λ t, t > c) tamika_products) carlos_products) =
   tamika_products.length * carlos_products.length) :=
by
  intro h
  sorry

end tamika_always_wins_l815_815568


namespace sixth_day_is_wednesday_l815_815877

noncomputable def day_of_week : Type := 
  { d // d ∈ ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] }

def five_fridays_sum_correct (x : ℤ) : Prop :=
  x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 75

def first_is_friday (x : ℤ) : Prop :=
  x = 1

def day_of_6th_is_wednesday (d : day_of_week) : Prop :=
  d.1 = "Wednesday"

theorem sixth_day_is_wednesday (x : ℤ) (d : day_of_week) :
  five_fridays_sum_correct x → first_is_friday x → day_of_6th_is_wednesday d :=
by
  sorry

end sixth_day_is_wednesday_l815_815877


namespace length_BD_l815_815088

/-- In an isosceles triangle ABC with AB = BC, the point D is the midpoint of both segments BC and AC.
We are also given that there is a point E on segment BC such that BE = 9 units and AC = 16 units.
The goal is to prove that the length of segment BD is 8 units. -/
theorem length_BD (A B C D E : Point) (hABC : AB = BC) 
  (hD_mid_BC : midpoint D B C) (hD_mid_AC : midpoint D A C)
  (hBE_9 : dist B E = 9) (hAC_16 : dist A C = 16) :
  dist B D = 8 := 
sorry

end length_BD_l815_815088


namespace new_average_weight_is_27_3_l815_815573

-- Define the given conditions as variables/constants in Lean
noncomputable def original_students : ℕ := 29
noncomputable def original_average_weight : ℝ := 28
noncomputable def new_student_weight : ℝ := 7

-- The total weight of the original students
noncomputable def original_total_weight : ℝ := original_students * original_average_weight
-- The new total number of students
noncomputable def new_total_students : ℕ := original_students + 1
-- The new total weight after new student is added
noncomputable def new_total_weight : ℝ := original_total_weight + new_student_weight

-- The theorem to prove that the new average weight is 27.3 kg
theorem new_average_weight_is_27_3 : (new_total_weight / new_total_students) = 27.3 := 
by
  sorry -- The proof will be provided here

end new_average_weight_is_27_3_l815_815573


namespace solve_for_x_l815_815933

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.8) : x = 71.7647 := 
by 
  sorry

end solve_for_x_l815_815933


namespace polygon_sides_twice_diagonals_l815_815963

theorem polygon_sides_twice_diagonals (n : ℕ) (h1 : n ≥ 3) (h2 : n * (n - 3) / 2 = 2 * n) : n = 7 :=
sorry

end polygon_sides_twice_diagonals_l815_815963


namespace sequence_divisibility_condition_l815_815764

theorem sequence_divisibility_condition (t a b x1 : ℕ) (x : ℕ → ℕ)
  (h1 : a = 1) (h2 : b = t) (h3 : x1 = t) (h4 : x 1 = x1)
  (h5 : ∀ n, n ≥ 2 → x n = a * x (n - 1) + b) :
  (∀ m n, m ∣ n → x m ∣ x n) ↔ (a = 1 ∧ b = t ∧ x1 = t) := sorry

end sequence_divisibility_condition_l815_815764


namespace min_sum_areas_l815_815378

variables (O A B F : ℝ × ℝ)
variable [h_parabola_A : A.snd ^ 2 = A.fst]
variable [h_parabola_B : B.snd ^ 2 = B.fst]
variable [h_opposite_sides : A.snd * B.snd < 0]
variable [h_dot_product_2 : O.fst * A.fst + O.snd * A.snd + O.fst * B.fst + O.snd * B.snd = 2]

theorem min_sum_areas (hab : A.snd + B.snd = 0) :
  let S_abo := 1/2 * (B.fst - A.fst) * A.snd
      S_afo := 1/2 * (F.fst * A.snd - A.fst * F.snd)
      y := A.snd in
  min (S_abo + S_afo) = 3 :=
sorry

end min_sum_areas_l815_815378


namespace simplify_expression_l815_815661

variable {x y z : ℝ}

theorem simplify_expression : [x - (y - z)] - [(x - y) - z] = 2z := 
by
  sorry

end simplify_expression_l815_815661


namespace polynomials_exist_l815_815509

theorem polynomials_exist (p : ℕ) (k : ℕ) (f : Polynomial ℤ)
  (hp : Nat.Prime p) (hk : 0 < k ∧ k ≤ p)
  (hdiv : ∀ x : ℤ, p^k ∣ f.eval x) :
  ∃ (A : Fin (k+1) → Polynomial ℤ),
    f = ∑ i in Finset.range (k+1), (Polynomial.X^p - Polynomial.X)^i * p^(k-i) * A i := 
sorry

end polynomials_exist_l815_815509


namespace intersection_distance_eq_l815_815583

theorem intersection_distance_eq (p q : ℕ) (h1 : p = 88) (h2 : q = 9) :
  p - q = 79 :=
by
  sorry

end intersection_distance_eq_l815_815583


namespace LocusOfCenterP_l815_815390

def CircleN (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16
def PointM : (ℝ × ℝ) := (-1, 0)

theorem LocusOfCenterP :
  (∀ P : ℝ × ℝ, CircleN P.1 P.2 ∧ abs (dist P PointM) = 2 → (x = P.1 ∧ y = P.2)) → 
  ((∃ x y : ℝ, (x = 1) ∧ y^2 = 3 * (1 - x / 4))) →
  (locus_Ω : ℝ × ℝ → Prop) :=
  sorry

end LocusOfCenterP_l815_815390


namespace angle_MAN_eq_angle_BPM_l815_815077

-- Given a rectangle ABCD with points M, N, and P defined as specified
structure Rectangle (α : Type*) [Field α] :=
(A B C D : α × α)
(M : α × α) (N : α × α) (P : α × α)
(mid_BC : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
(mid_CD : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
(inter_DM_BN : ∃ Mx My : α, ∃ Nx Ny : α, ∃ t u : α,
  M = (Mx, My) ∧ N = (Nx, Ny) ∧
  P = (D.1 * (1 - t) + Mx * t, D.2 * (1 - t) + My * t) ∧
  P = (B.1 * (1 - u) + Nx * u, B.2 * (1 - u) + Ny * u))

-- The theorem to prove
theorem angle_MAN_eq_angle_BPM {α : Type*} [Field α] (rect : Rectangle α) :
  ∠(rect.M) (rect.A) (rect.N) = ∠(rect.B) (rect.P) (rect.M) :=
sorry

end angle_MAN_eq_angle_BPM_l815_815077


namespace non_subset_condition_l815_815174

theorem non_subset_condition (M P : Set α) (non_empty : M ≠ ∅) : 
  ¬(M ⊆ P) ↔ ∃ x ∈ M, x ∉ P := 
sorry

end non_subset_condition_l815_815174


namespace min_value_y_at_x_equals_2_l815_815628

noncomputable def y (x : ℝ) : ℝ := real.sqrt ((x - 2)^2 + 4)

theorem min_value_y_at_x_equals_2 : y 2 = 2 :=
by sorry

end min_value_y_at_x_equals_2_l815_815628


namespace find_f_beta_l815_815818

open Real Trigonometry

theorem find_f_beta :
  ∀ (f : ℝ → ℝ) (α β : ℝ),
  (f x = sin (5 / 4 * π - x) - cos (π / 4 + x))
  → (cos (α - β) = 3 / 5)
  → (cos (α + β) = -3 / 5)
  → (0 < α ∧ α < β ∧ β ≤ π / 2)
  → f β = sqrt 2 :=
by
  intros f α β h1 h2 h3 h4
  sorry

end find_f_beta_l815_815818


namespace comic_book_stack_count_l815_815914

theorem comic_book_stack_count :
  let marvel_count := 7
  let dc_count := 6
  let indie_count := 5
  let total_comic_count := marvel_count + dc_count + indie_count
  let marvel_ways := factorial marvel_count
  let dc_ways := factorial dc_count
  let indie_ways := factorial indie_count
  let group_count := 3
  let group_ways := factorial group_count
  marvel_count = 7 ∧ dc_count = 6 ∧ indie_count = 5 →
  marvel_ways * dc_ways * indie_ways * group_ways = 2458982400 :=
sorry

end comic_book_stack_count_l815_815914


namespace maximize_sum_achieved_l815_815663

noncomputable def maximize_sum (a : Fin 10 → ℝ) : Prop :=
  ∀ σ : Equiv.Perm (Fin 10), 
    (∀ i j : Fin 10, i < j → a (σ i) < a (σ j)) →
    (∑ i : Fin 10, (i + 1) * a (σ i)) = (∑ i, (i + 1) * a i)

theorem maximize_sum_achieved (a : Fin 10 → ℝ) (h : ∀ i j : Fin 10, i < j → a i < a j) :
  maximize_sum a :=
by
  sorry

end maximize_sum_achieved_l815_815663


namespace drill_through_cube_without_intersecting_cuboids_l815_815620

-- Define the sizes of the cuboids and the large cube
def small_cuboid_size : ℕ × ℕ × ℕ := (2, 2, 1)
def large_cube_size : ℕ × ℕ × ℕ := (20, 20, 20)

-- Define the number of small cuboids
def number_of_small_cuboids : ℕ := 2000

-- State the problem in Lean 4 statement
theorem drill_through_cube_without_intersecting_cuboids :
  ∀ (assembly : list (ℕ × ℕ × ℕ)), 
  (assembly.length = number_of_small_cuboids) →
  (∀ (xyz : ℕ × ℕ × ℕ), xyz ∈ assembly → xyz.1 < large_cube_size.1 ∧ xyz.2 < large_cube_size.2 ∧ xyz.3 < large_cube_size.3) →
  ∃ (line : ℕ × ℕ), -- line parallel to an edge of the large cube
  (∀ (cuboid : ℕ × ℕ × ℕ), cuboid ∈ assembly → ¬ line_intersects_cuboid_interior line cuboid) :=
begin
  sorry
end

-- Helper function to determine if a line intersects the interior of a cuboid
def line_intersects_cuboid_interior (line : ℕ × ℕ) (cuboid : ℕ × ℕ × ℕ) : Prop :=
  -- Define your condition to check if the line intersects the interior of the cuboid
  sorry

end drill_through_cube_without_intersecting_cuboids_l815_815620


namespace angle_ABC_is_67_5_degrees_l815_815579

theorem angle_ABC_is_67_5_degrees
  (square_octagon: ∀ A B C D E F G H I J: π (λ x, x ∈ RegularPolygon) → (λ y, y ∈ Square) → Prop)
  (n := 8)
  (interior_angle_octagon : ∀ (x : ℝ), x ∈ RegularPolygon n -> x = (n - 2) * 180 / n)
  (interior_angle_square : ∀ (x : ℝ), x ∈ Square -> x = 90)
  (shared_side : ∀ (a b: E), (λ x y, Square x y ∧ Octagon x y))
  (isosceles_triangle : ∀ (x y z: F), (λ a b, EquilateralTriangle a b ∧ Triangle x (a, b) y z))
  : ∀ (angle_ABC: ℝ), angle_ABC = 67.5 :=
begin
  sorry
end

end angle_ABC_is_67_5_degrees_l815_815579


namespace larger_number_l815_815970

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l815_815970


namespace larger_number_is_23_l815_815978

theorem larger_number_is_23 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l815_815978


namespace range_of_m_correct_l815_815352

noncomputable def range_of_m (x : ℝ) (m : ℝ) : Prop :=
  (x + m) / (x - 2) - (2 * m) / (x - 2) = 3 ∧ x > 0 ∧ x ≠ 2

theorem range_of_m_correct (m : ℝ) : 
  (∃ x : ℝ, range_of_m x m) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_correct_l815_815352


namespace maximum_ab_l815_815018

noncomputable theory

def function_positive (a b : ℝ) : Prop :=
  ∀ x : ℝ, exp x - a * x - b ≥ 0

theorem maximum_ab (a b : ℝ) (h : function_positive a b) : 
  a * b ≤ exp 1 / 2 :=
sorry

end maximum_ab_l815_815018


namespace divisible_by_6_l815_815146

theorem divisible_by_6 {n : ℕ} (h2 : 2 ∣ n) (h3 : 3 ∣ n) : 6 ∣ n :=
sorry

end divisible_by_6_l815_815146


namespace find_y_of_rectangle_area_l815_815161

theorem find_y_of_rectangle_area (y : ℝ) (h1 : y > 0) 
(h2 : (0, 0) = (0, 0)) (h3 : (0, 6) = (0, 6)) 
(h4 : (y, 6) = (y, 6)) (h5 : (y, 0) = (y, 0)) 
(h6 : 6 * y = 42) : y = 7 :=
by {
  sorry
}

end find_y_of_rectangle_area_l815_815161


namespace inradius_inequality_l815_815659

/-- Given a point P inside the triangle ABC, where da, db, and dc are the distances from P to the sides BC, CA, and AB respectively,
 and r is the inradius of the triangle ABC, prove the inequality -/
theorem inradius_inequality (a b c da db dc : ℝ) (r : ℝ) 
  (h1 : 0 < da) (h2 : 0 < db) (h3 : 0 < dc)
  (h4 : r = (a * da + b * db + c * dc) / (a + b + c)) :
  2 / (1 / da + 1 / db + 1 / dc) < r ∧ r < (da + db + dc) / 2 :=
  sorry

end inradius_inequality_l815_815659


namespace solve_equation_l815_815152

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x^2 + x + 1) / (x + 2) = x + 1 → x = -1 / 2 := 
by
  intro h1
  sorry

end solve_equation_l815_815152


namespace nuts_needed_for_cookies_l815_815910

-- Given conditions
def total_cookies : Nat := 120
def fraction_nuts : Rat := 1 / 3
def fraction_chocolate : Rat := 0.25
def nuts_per_cookie : Nat := 3

-- Translated conditions as helpful functions
def cookies_with_nuts : Nat := Nat.floor (fraction_nuts * total_cookies)
def cookies_with_chocolate : Nat := Nat.floor (fraction_chocolate * total_cookies)
def cookies_with_both : Nat := total_cookies - cookies_with_nuts - cookies_with_chocolate
def total_cookies_with_nuts : Nat := cookies_with_nuts + cookies_with_both
def total_nuts_needed : Nat := total_cookies_with_nuts * nuts_per_cookie

-- Proof problem: proving that total nuts needed is 270
theorem nuts_needed_for_cookies : total_nuts_needed = 270 :=
by
  sorry

end nuts_needed_for_cookies_l815_815910


namespace probability_of_four_of_a_kind_is_correct_l815_815621

noncomputable def probability_four_of_a_kind: ℚ :=
  let total_ways := Nat.choose 52 5
  let successful_ways := 13 * 1 * 12 * 4
  (successful_ways: ℚ) / (total_ways: ℚ)

theorem probability_of_four_of_a_kind_is_correct :
  probability_four_of_a_kind = 13 / 54145 := 
by
  -- sorry is used because we are only writing the statement, no proof required
  sorry

end probability_of_four_of_a_kind_is_correct_l815_815621


namespace incenter_coordinates_l815_815089

theorem incenter_coordinates (p q r : ℝ) (h₁ : p = 8) (h₂ : q = 6) (h₃ : r = 10) :
  ∃ x y z : ℝ, x + y + z = 1 ∧ x = p / (p + q + r) ∧ y = q / (p + q + r) ∧ z = r / (p + q + r) ∧
  x = 1 / 3 ∧ y = 1 / 4 ∧ z = 5 / 12 :=
by
  sorry

end incenter_coordinates_l815_815089


namespace problem_1_problem_2_problem_3_l815_815399

noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) := 0.5 * x ^ 2 - 2 * x
noncomputable def g' (x : ℝ) := x - 2

theorem problem_1 (m n : ℝ) (a b : ℝ) (h0 : ∀ x, f x = Real.log x) (h1 : ∀ x, g x = 0.5 * x ^ 2 - 2 * x) : 
    f(1) = Real.log 1 ∧ f'(1) = 1 / (1 + m) := by
  intro x
  sorry

theorem problem_2 (t : ℝ) (φ := λ x, Real.log (x + 1) - x + 2) :
    (∃ t, φ' t ≠ 0 ) → - 0.5 < t ∧ t < 0 := by
  sorry

theorem problem_3 (k : ℤ) (h : ∀ x > 1, k * (x - 1) < x * f (x) + 3 * g' (x) + 4) :
    k ≤ 5 := by
  sorry

end problem_1_problem_2_problem_3_l815_815399


namespace total_treats_value_l815_815726

noncomputable def hotel_per_night := 4000
noncomputable def nights := 2
noncomputable def car_value := 30000
noncomputable def house_value := 4 * car_value
noncomputable def total_value := hotel_per_night * nights + car_value + house_value

theorem total_treats_value : total_value = 158000 :=
by
  sorry

end total_treats_value_l815_815726


namespace nth_letter_2023_is_X_l815_815224

def repeating_sequence : string := "XYZZYX"

def nth_repeating_letter (n : ℕ) : char :=
  let seq := repeating_sequence.to_list;
  seq.get! (n % seq.length)

theorem nth_letter_2023_is_X :
  nth_repeating_letter 2023 = 'X' :=
by
  sorry

end nth_letter_2023_is_X_l815_815224


namespace range_of_t_l815_815841

variable {f : ℝ → ℝ}

theorem range_of_t (h₁ : ∀ x y : ℝ, x < y → f x ≥ f y) (h₂ : ∀ t : ℝ, f (t^2) < f t) : 
  ∀ t : ℝ, f (t^2) < f t ↔ (t < 0 ∨ t > 1) := 
by 
  sorry

end range_of_t_l815_815841


namespace largest_positive_integer_solution_l815_815958

theorem largest_positive_integer_solution (x : ℕ) (h₁ : 1 ≤ x) (h₂ : x + 3 ≤ 6) : 
  x = 3 := by
  sorry

end largest_positive_integer_solution_l815_815958


namespace unique_function_solution_l815_815334

theorem unique_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end unique_function_solution_l815_815334


namespace laura_running_speed_l815_815885

noncomputable def running_speed_nearest_hundredth (x : ℝ) (transition_min : ℝ) (bike_distance : ℝ) (bike_speed : ℝ → ℝ) (run_distance : ℝ) (workout_min : ℝ) : ℝ :=
  let total_motion_time := (workout_min - transition_min) / 60
  let bike_time := bike_distance / bike_speed x
  let run_time := run_distance / x
  have h : bike_time + run_time = total_motion_time
  exact x 

theorem laura_running_speed : running_speed_nearest_hundredth 8.38 10 25 (λ x, 2 * x + 2) 7 140 ≈ 8.38 :=
sorry

end laura_running_speed_l815_815885


namespace max_no_T_pieces_l815_815623

-- Define the "T" shaped piece. It occupies 4 cells, specifically, we can define 
-- it in terms of coordinates it occupies on a grid, for simplicity we can represent 
-- its structure but not directly use in our conditions as it inferred from solution steps

noncomputable def T_piece : ℕ := 4

-- Define the grid as a 4 x 5 grid.
structure Grid := 
(width height : ℕ)

-- Define the properties of the given grid.
def myGrid : Grid := {width := 5, height := 4}

-- Define the problem statement with conditions and the answer 4
theorem max_no_T_pieces : 
  ∀ (g : Grid), g = myGrid → ∀ (t : ℕ), t = T_piece → (max_non_overlapping_pieces g t = 4) := 
sorry

end max_no_T_pieces_l815_815623


namespace mike_total_cost_self_correct_l815_815912

-- Definition of the given conditions
def cost_per_rose_bush : ℕ := 75
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def cost_per_tiger_tooth_aloes : ℕ := 100
def total_tiger_tooth_aloes : ℕ := 2

-- Calculate the total cost for Mike's plants
def total_cost_mike_self: ℕ := 
  (total_rose_bushes - friend_rose_bushes) * cost_per_rose_bush + total_tiger_tooth_aloes * cost_per_tiger_tooth_aloes

-- The main proposition to be proved
theorem mike_total_cost_self_correct : total_cost_mike_self = 500 := by
  sorry

end mike_total_cost_self_correct_l815_815912


namespace apples_remaining_l815_815126

variable (initial_apples : ℕ)
variable (picked_day1 : ℕ)
variable (picked_day2 : ℕ)
variable (picked_day3 : ℕ)

-- Given conditions
def condition1 : initial_apples = 200 := sorry
def condition2 : picked_day1 = initial_apples / 5 := sorry
def condition3 : picked_day2 = 2 * picked_day1 := sorry
def condition4 : picked_day3 = picked_day1 + 20 := sorry

-- Prove the total number of apples remaining is 20
theorem apples_remaining (H1 : initial_apples = 200) 
  (H2 : picked_day1 = initial_apples / 5) 
  (H3 : picked_day2 = 2 * picked_day1)
  (H4 : picked_day3 = picked_day1 + 20) : 
  initial_apples - (picked_day1 + picked_day2 + picked_day3) = 20 := 
by
  sorry

end apples_remaining_l815_815126


namespace smallest_solution_l815_815759

open Int

theorem smallest_solution (x : ℝ) (h : ⌊x⌋ = 3 + 50 * (x - ⌊x⌋)) : x = 3.00 :=
by
  sorry

end smallest_solution_l815_815759


namespace point_B_in_second_quadrant_l815_815080

variables (a b : ℝ)

theorem point_B_in_second_quadrant (ha : 0 < a) (hb : b < 0) : (0 < -b) ∧ (-a < 0) :=
by
  split;
  { sorry }

end point_B_in_second_quadrant_l815_815080


namespace james_total_room_area_l815_815102

theorem james_total_room_area :
  let original_length := 13
  let original_width := 18
  let increase := 2
  let new_length := original_length + increase
  let new_width := original_width + increase
  let area_of_one_room := new_length * new_width
  let number_of_rooms := 4
  let area_of_four_rooms := area_of_one_room * number_of_rooms
  let area_of_larger_room := area_of_one_room * 2
  let total_area := area_of_four_rooms + area_of_larger_room
  total_area = 1800
  := sorry

end james_total_room_area_l815_815102


namespace functional_eqn_solution_l815_815333

-- Define the function f : ℝ⁺ → ℝ⁺
noncomputable def f (x : ℝ) : ℝ := 
  if x = 1 then 1 else 1 / x

-- State the theorem
theorem functional_eqn_solution (f : ℝ → ℝ) (x y : ℝ)
  (h1 : ∀ x y : ℝ, x > 0 → y > 0 → f(f(x) + y) * f(x) = f(x * y + 1))
  (h2 : f 1 = 1)
  (h3 : ∀ x : ℝ, x > 1 → f x = 1 / x) : 
  ∀ x : ℝ, x > 0 → f x = (if x = 1 then 1 else 1 / x) :=
begin
  sorry
end

end functional_eqn_solution_l815_815333


namespace simplify_expression_l815_815769

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end simplify_expression_l815_815769


namespace opposite_of_one_over_2023_l815_815577

def one_over_2023 : ℚ := 1 / 2023

theorem opposite_of_one_over_2023 : -one_over_2023 = -1 / 2023 :=
by
  sorry

end opposite_of_one_over_2023_l815_815577


namespace otimes_not_preserve_square_l815_815729

def otimes (x y : ℝ) : ℝ := max x y

theorem otimes_not_preserve_square : ∃ (x y : ℝ), (otimes x y)^2 ≠ otimes (x^2) (y^2) :=
by
  sorry

end otimes_not_preserve_square_l815_815729


namespace polynomial_roots_l815_815593

theorem polynomial_roots (a b c : ℚ) (h1 : (x : ℝ) → x^4 + a * x^2 + b * x + c = 0) :
    (2 - Real.sqrt 5) = -2 ∨ (2 + Real.sqrt 5) = -2 := 
begin
  sorry -- Proof here
end

end polynomial_roots_l815_815593


namespace last_two_digits_of_sum_of_factorials_l815_815216

-- Problem statement: Sum of factorials from 1 to 15
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.factorial k)

-- Define the main problem
theorem last_two_digits_of_sum_of_factorials : 
  (sum_factorials 15) % 100 = 13 :=
by 
  sorry

end last_two_digits_of_sum_of_factorials_l815_815216


namespace tangent_line_at_1_0_l815_815740

noncomputable def tangent_line_equation (x y : ℝ) : ℝ :=
  e^2 * x - y - e^2

theorem tangent_line_at_1_0 :
  let f := λ x : ℝ, exp x in
  tangent_line_equation 1 0 = 0 :=
by 
  sorry

end tangent_line_at_1_0_l815_815740


namespace matrix_and_line_l815_815389

open Matrix

noncomputable def eigen_matrix := λ (M : Matrix (Fin 2) (Fin 2) ℝ), 
  eigenvalues M = [8] ∧ has_eigenvector M 8 ![1, 1] 
  ∧ M.mul_vec ![-1, 2] = ![-2, 4]

noncomputable def mapped_line_eq := 
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ), eigen_matrix M →
  (λ (x y x' y' : ℝ), 
    M.mul_vec ![x, y] = ![x', y'] → (x' - 2 * y' = 4 ↔ x + 3 * y + 2 = 0))

theorem matrix_and_line (M : Matrix (Fin 2) (Fin 2) ℝ):
  eigen_matrix M →
  M = ![![6, 2], ![4, 4]] ∧ mapped_line_eq M :=
by
  sorry

end matrix_and_line_l815_815389


namespace find_speed2_l815_815881

-- Define the constants
def speed1 := 80 -- driving speed in miles/hour for the first segment
def time1 := 6 -- time in hours for the first segment
def speed3 := 40 -- driving speed in miles/hour for the last segment
def time3 := 2 -- time in hours for the last segment
def total_distance := 800 -- total distance traveled in miles

-- Define the unknown speed during the second segment
variable (speed2 : ℕ)

-- Calculate the distance for each segment; known distances for segment 1 and 3
def distance1 := speed1 * time1
def distance3 := speed3 * time3

-- Calculate the distance for the unknown segment 2
def time2 := 4
def distance2 := total_distance - (distance1 + distance3)

-- Lean theorem to prove the required speed
theorem find_speed2 : speed2 * time2 = distance2 → speed2 = 60 :=
by
  intro h
  have : distance2 = 240 := rfl
  rw [mul_comm] at h
  rw [this] at h
  rw [mul_comm speed2] at h
  norm_num at h
  exact h

end find_speed2_l815_815881


namespace acute_angle_between_CD_AF_l815_815549

noncomputable def midpoint (A B C : Type*) [Euclidean_space V P] (B : V) := midpoint ℝ A C = B

noncomputable def square (A B D E : Type*) [Euclidean_space V P] := 
  IsSquare (convex_hull ℝ {A, B, D, E})

noncomputable def equilateral_triangle (B C F : Type*) [Euclidean_space V P] := 
  IsEquilateralTriangle (convex_hull ℝ {B, C, F})

theorem acute_angle_between_CD_AF
  (A B C D E F : Type*) [Euclidean_space V P] 
  (h1 : midpoint A B C)
  (h2 : square A B D E)
  (h3 : equilateral_triangle B C F) : 
  acute_angle_between_lines D C A F = 75 :=
sorry

end acute_angle_between_CD_AF_l815_815549


namespace origin_moves_3sqrt5_under_dilation_l815_815676

/--
Given:
1. The original circle has radius 3 centered at point B(3, 3).
2. The dilated circle has radius 6 centered at point B'(9, 12).

Prove that the distance moved by the origin O(0, 0) under this dilation is 3 * sqrt(5).
-/
theorem origin_moves_3sqrt5_under_dilation:
  let B := (3, 3)
  let B' := (9, 12)
  let radius_B := 3
  let radius_B' := 6
  let dilation_center := (-3, -6)
  let origin := (0, 0)
  let k := radius_B' / radius_B
  let d_0 := Real.sqrt ((-3 : ℝ)^2 + (-6 : ℝ)^2)
  let d_1 := k * d_0
  d_1 - d_0 = 3 * Real.sqrt (5 : ℝ) := by sorry

end origin_moves_3sqrt5_under_dilation_l815_815676


namespace inverse_of_original_l815_815956

-- Definitions based on conditions
def original_proposition : Prop := ∀ (x y : ℝ), x = y → |x| = |y|

def inverse_proposition : Prop := ∀ (x y : ℝ), |x| = |y| → x = y

-- Lean 4 statement
theorem inverse_of_original : original_proposition → inverse_proposition :=
sorry

end inverse_of_original_l815_815956


namespace probability_floor_eq_l815_815556

theorem probability_floor_eq (x y : ℝ) (h₁: 0 < x ∧ x < 1) (h₂: 0 < y ∧ y < 1) : 
  let p := ∫l in 0..ln 3, ∫m in 0..ln 3, (1 : ℝ) in
  p = (ln 3)^2 := 
sorry

end probability_floor_eq_l815_815556


namespace fraction_value_l815_815777

variable (x y : ℝ)

theorem fraction_value (h : 1/x - 1/y = 3) : (2 * x + 3 * x * y - 2 * y) / (x - 2 * x * y - y) = 3 / 5 := 
by sorry

end fraction_value_l815_815777


namespace players_on_team_are_4_l815_815775

noncomputable def number_of_players (score_old_record : ℕ) (rounds : ℕ) (score_first_9_rounds : ℕ) (final_round_diff : ℕ) :=
  let points_needed := score_old_record * rounds
  let points_final_needed := score_old_record - final_round_diff
  let total_points_needed := points_needed * 1
  let final_round_points_needed := total_points_needed - score_first_9_rounds
  let P := final_round_points_needed / points_final_needed
  P

theorem players_on_team_are_4 :
  number_of_players 287 10 10440 27 = 4 :=
by
  sorry

end players_on_team_are_4_l815_815775


namespace relationship_a_c_b_l815_815780

-- Definitions as given in the problem
def a := Real.log 0.3 / Real.log 2
def b := 2 ^ 0.1
def c := 0.2 ^ 1.3

-- The proof statement
theorem relationship_a_c_b : a < c ∧ c < b := 
by
  -- Conditions are not directly used here; we just state the final theorem.
  sorry

end relationship_a_c_b_l815_815780


namespace sally_poems_l815_815557

theorem sally_poems (
  (initial_memorized : ℕ := 20)
  (recite_perfectly : ℕ := 7)
  (remember_stanza : ℕ := 5)
  (mix_up_lines : ℕ := 4)
) : 
  (initial_memorized - (recite_perfectly + remember_stanza + mix_up_lines) = 4)
  ∧ (remember_stanza + mix_up_lines = 9) :=
by {
  have total_remembered := recite_perfectly + remember_stanza + mix_up_lines,
  have total_memorized := initial_memorized,
  have poems_forgotten := total_memorized - total_remembered,
  have poems_partial := remember_stanza + mix_up_lines,
  split,
  {
    exact poems_forgotten,
    sorry  -- skip the proof
  },
  {
    exact poems_partial,
    sorry  -- skip the proof
  }
}

end sally_poems_l815_815557


namespace perfect_iff_induced_subgraph_condition_l815_815273

variables {G : Type*} [graph G]

def is_perfect (G : graph) : Prop :=
  ∀ (H : graph), H ⊆ G → χ(H) = ω(H)

theorem perfect_iff_induced_subgraph_condition (G : graph) :
  (is_perfect G) ↔ (∀ (H : graph), H ⊆ G → |H| ≤ α(H) * ω(H)) :=
sorry

end perfect_iff_induced_subgraph_condition_l815_815273


namespace find_a_odd_function_l815_815802

theorem find_a_odd_function (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h1 : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = log x - a * x)
  (h2 : a > 1 / 2)
  (h_min : ∃ x : ℝ, -2 < x ∧ x < 0 ∧ f x = 1 ∧ ∀ y : ℝ, -2 < y ∧ y < 0 → f y ≥ f x) :
  a = 1 := by
  sorry

end find_a_odd_function_l815_815802


namespace arithmetic_sequence_log_l815_815589

theorem arithmetic_sequence_log {a b : ℝ} (h1 : ∀ n : ℕ, 
  (log (a ^ (n + 5) * b ^ (n + 7)) = log (a ^ (n + 11) * b ^ (n + 17))) -> 
  (log (a ^ (n + 11) * b ^ (n + 17)) = log (a ^ (n + 19) * b ^ (n + 29)))) :

  ∀ (n : ℕ), (10 = n - 1) → log (b ^ n) = 156 :=
begin
  sorry
end

end arithmetic_sequence_log_l815_815589


namespace CA_inter_B_l815_815123

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5, 7}

theorem CA_inter_B :
  (U \ A) ∩ B = {2, 7} := by
  sorry

end CA_inter_B_l815_815123


namespace graph_of_cubic_equation_is_three_lines_l815_815722

theorem graph_of_cubic_equation_is_three_lines (x y : ℝ) :
  (x + y) ^ 3 = x ^ 3 + y ^ 3 →
  (y = -x ∨ x = 0 ∨ y = 0) :=
by
  sorry

end graph_of_cubic_equation_is_three_lines_l815_815722


namespace zeros_of_f_is_pm3_l815_815994

def f (x : ℝ) : ℝ := x^2 - 9

theorem zeros_of_f_is_pm3 :
  ∃ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -3 :=
by sorry

end zeros_of_f_is_pm3_l815_815994


namespace solve_for_y_l815_815803

theorem solve_for_y (y : ℤ) (h : (8 + 12 + 23 + 17 + y) / 5 = 15) : y = 15 :=
by {
  sorry
}

end solve_for_y_l815_815803


namespace simplify_expression_l815_815766

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by 
  sorry

end simplify_expression_l815_815766


namespace sum_factorials_last_two_digits_l815_815217

/-- Prove that the last two digits of the sum of factorials of the first 15 positive integers equal to 13 --/
theorem sum_factorials_last_two_digits : 
  let f := fun n => Nat.factorial n in
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 + f 11 + f 12 + f 13 + f 14 + f 15) % 100 = 13 :=
by 
  sorry

end sum_factorials_last_two_digits_l815_815217


namespace pq_iff_cond_l815_815794

def p (a : ℝ) := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem pq_iff_cond (a : ℝ) : (p a ∧ q a) ↔ (a ≤ -2 ∨ a = 1) := 
by
  sorry

end pq_iff_cond_l815_815794


namespace job_completion_time_l815_815648

theorem job_completion_time 
  (tina_hours : ℕ) (ann_hours : ℕ) (tina_worked_hours : ℕ) (remaining_fraction : ℚ) 
  (Tina_job_completion : tina_hours = 12) 
  (Ann_job_completion : ann_hours = 9) 
  (Tina_worked : tina_worked_hours = 8) 
  (Remaining_job : remaining_fraction = 1 / 3) :
  ∃ (ann_remaining_hours : ℕ), ann_remaining_hours = 3 :=
by 
  -- Steps based on provided problem and solution
  let tina_rate := 1 / 12
  let ann_rate := 1 / 9
  let tina_completed := 8 * tina_rate 
  let remaining := 1 - tina_completed
  have Ann_time : remaining / ann_rate = 3 := sorry
  use 3
  exact Ann_time

end job_completion_time_l815_815648


namespace solution_set_f_pos_min_a2_b2_c2_l815_815893

def f (x : ℝ) : ℝ := |2 * x + 3| - |x - 1|

theorem solution_set_f_pos : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -3 / 2 ∨ -2 / 3 < x } := 
sorry

theorem min_a2_b2_c2 (a b c : ℝ) (h : a + 2 * b + 3 * c = 5) : 
  a^2 + b^2 + c^2 ≥ 25 / 14 :=
sorry

end solution_set_f_pos_min_a2_b2_c2_l815_815893


namespace necklace_count_17_beads_l815_815832

theorem necklace_count_17_beads :
  let n := 17 in
  let circular_permutations := (n - 1)! in
  let reflection_symmetry := 2 in
  circular_permutations / reflection_symmetry = 8 * 15! := 
by
  sorry

end necklace_count_17_beads_l815_815832


namespace bakery_used_0_2_bags_of_wheat_flour_l815_815242

-- Define the conditions
def total_flour := 0.3
def white_flour := 0.1

-- Define the number of bags of wheat flour used
def wheat_flour := total_flour - white_flour

-- The proof statement
theorem bakery_used_0_2_bags_of_wheat_flour : wheat_flour = 0.2 := 
by
  sorry

end bakery_used_0_2_bags_of_wheat_flour_l815_815242


namespace problem_value_l815_815234

theorem problem_value :
  (1 / 3 * 9 * 1 / 27 * 81 * 1 / 243 * 729 * 1 / 2187 * 6561 * 1 / 19683 * 59049) = 243 := 
sorry

end problem_value_l815_815234


namespace exists_real_x_y_l815_815898

-- (Condition)
variables (f : ℝ → ℝ)

-- (Statement to prove)
theorem exists_real_x_y :
  ∃ x y : ℝ, f (x - f y) > y * f x + x :=
begin
  sorry
end

end exists_real_x_y_l815_815898


namespace integer_solutions_to_inequality_l815_815432

theorem integer_solutions_to_inequality :
  ∃ n : ℕ, n = 5 ∧ (∀ x : ℤ, (x - 2) ^ 2 ≤ 4 ↔ x ∈ {0, 1, 2, 3, 4}) :=
by
  sorry

end integer_solutions_to_inequality_l815_815432


namespace total_pizza_pieces_l815_815738

-- Definitions based on the conditions
def pieces_per_pizza : Nat := 6
def pizzas_per_student : Nat := 20
def number_of_students : Nat := 10

-- Statement of the theorem
theorem total_pizza_pieces :
  pieces_per_pizza * pizzas_per_student * number_of_students = 1200 :=
by
  -- Placeholder for the proof
  sorry

end total_pizza_pieces_l815_815738


namespace right_isosceles_triangle_ab_length_l815_815490

theorem right_isosceles_triangle_ab_length (AC : ℝ) (hyp_1 : AC = 8 * real.sqrt 2) :
  let AB := (AC / real.sqrt 2)
  in AC = 8 * real.sqrt 2 → AB = 8 := by sorry

end right_isosceles_triangle_ab_length_l815_815490


namespace mark_donates_cans_of_soup_l815_815536

theorem mark_donates_cans_of_soup:
  let n_shelters := 6
  let p_per_shelter := 30
  let c_per_person := 10
  let total_people := n_shelters * p_per_shelter
  let total_cans := total_people * c_per_person
  total_cans = 1800 :=
by sorry

end mark_donates_cans_of_soup_l815_815536


namespace radical_axis_through_TB_and_TD_l815_815886

variables (A B C D F_A F_B F_C F_D I_A I_B I_C I_D T_B T_D : Type) 
variable [cyclic_quadrilateral A B C D] 

def midpoint_arcs (ω : Type) (AB BC CD DA : Type) : Type := sorry 
def incenters (DAB ABC BCD CDA : Type) : Type := sorry 
def tangent_to (circle1 circle2 : Type) (pt : Type) (line : Type) : Prop := sorry 
def second_intersection (ω : Type) (circle : Type) (pt : Type) : Type := sorry 
def radical_axis (circle1 circle2 : Type) : Type := sorry 
def passes_through (line pt1 pt2 : Type) : Prop := sorry 

theorem radical_axis_through_TB_and_TD : 
    passes_through (radical_axis (tangent_to ω ω F_A CD) (tangent_to ω ω F_C AB)) T_B T_D := 
sorry

end radical_axis_through_TB_and_TD_l815_815886


namespace triangle_lengths_relationship_l815_815531

-- Given data
variables {a b c f_a f_b f_c t_a t_b t_c : ℝ}
-- Conditions/assumptions
variables (h1 : f_a * t_a = b * c)
variables (h2 : f_b * t_b = a * c)
variables (h3 : f_c * t_c = a * b)

-- Theorem to prove
theorem triangle_lengths_relationship :
  a^2 * b^2 * c^2 = f_a * f_b * f_c * t_a * t_b * t_c :=
by sorry

end triangle_lengths_relationship_l815_815531


namespace g_75_value_l815_815169

variable {α : Type} [LinearOrderedField α]

noncomputable def g (x : α) : α := sorry

theorem g_75_value (h1 : ∀ x y : α, (0 < x) → (0 < y) → g (x * y) = g x / y^2)
  (h2 : g (50 : α) = 25) :
  g (75 : α) = (100 / 9 : α) := 
by
sorρι

end g_75_value_l815_815169


namespace find_polar_center_polar_coordinates_find_value_of_r_l815_815865

noncomputable def circle_center_polar_coordinates :=
  (-(Real.sqrt 2) / 2, -(Real.sqrt 2) / 2)

noncomputable def line_polar_equation (theta : ℝ) (rho : ℝ) : Prop :=
  rho * Real.sin (theta + Real.pi / 4) = Real.sqrt 2 / 2

def max_distance_condition (r : ℝ) : Prop :=
  let center := (-(Real.sqrt 2) / 2, -(Real.sqrt 2) / 2)
  let distance_from_center := abs (-(Real.sqrt 2) - 1) / Real.sqrt 2
  distance_from_center + r = 3

theorem find_polar_center_polar_coordinates :
  circle_center_polar_coordinates = (1, 5 * Real.pi / 4) := sorry

theorem find_value_of_r (r > 0) :
  max_distance_condition r ↔ r = 2 - Real.sqrt 2 / 2 := sorry

end find_polar_center_polar_coordinates_find_value_of_r_l815_815865


namespace quintuple_eq_l815_815366

variables (A' B' C' D' A B C D : Type) [AddCommGroup A'] [VectorSpace ℝ A'] 

-- Definitions of the points and conditions
def B_def (A A' : A') : A' := (1/2 : ℝ) • A + (1/2 : ℝ) • A'
def C_def (C' : C') : C' := C'
def D_def (C' D' : D') : D' := (1/2 : ℝ) • C' + (1/2 : ℝ) • D'

-- Proof of the given statement
theorem quintuple_eq (A' B' C' D' : A') 
  (hB : B' = B_def A' A')
  (hC : C' = C_def C')
  (hD : D' = D_def C' D') :
  A = 0 • A' + 0 • B' + (1/4 : ℝ) • C' + (3/4 : ℝ) • D' :=
sorry

end quintuple_eq_l815_815366


namespace correct_line_equation_l815_815948

theorem correct_line_equation :
  ∃ (c : ℝ), (∀ (x y : ℝ), 2 * x - 3 * y + 4 = 0 → 2 * x - 3 * y + c = 0 ∧ 2 * (-1) - 3 * 2 + c = 0) ∧ c = 8 :=
by
  use 8
  sorry

end correct_line_equation_l815_815948


namespace ads_ratio_l815_815212

theorem ads_ratio 
  (first_ads : ℕ := 12)
  (second_ads : ℕ)
  (third_ads := second_ads + 24)
  (fourth_ads := (3 / 4) * second_ads)
  (clicked_ads := 68)
  (total_ads := (3 / 2) * clicked_ads == 102)
  (ads_eq : first_ads + second_ads + third_ads + fourth_ads = total_ads) :
  second_ads / first_ads = 2 :=
by sorry

end ads_ratio_l815_815212


namespace measure_AB_l815_815868

noncomputable def segment_measure (a b : ℝ) : ℝ :=
  a + (2 / 3) * b

theorem measure_AB (a b : ℝ) (parallel_AB_CD : true) (angle_B_three_times_angle_D : true) (measure_AD_eq_a : true) (measure_CD_eq_b : true) :
  segment_measure a b = a + (2 / 3) * b :=
by
  sorry

end measure_AB_l815_815868


namespace count_even_n_factorial_condition_l815_815763

def is_integer (x : ℚ) : Prop := ∃ z : ℤ, x = z

theorem count_even_n_factorial_condition :
  (finset.card (finset.filter (λ n : ℕ, n % 2 = 0 ∧ 1 ≤ n ∧ n ≤ 100 ∧ is_integer (rat.of_int ((factorial (n^2 - 1)) / (rat.of_int (factorial n)^n)))) (finset.range 101))) = 5 := 
by
  sorry

end count_even_n_factorial_condition_l815_815763


namespace calculate_expression_l815_815312

theorem calculate_expression :
  (sqrt 3 * tan (real.pi / 3) - real.cbrt 8 - (-2 : ℝ)⁻² + abs (-sqrt 2)) = (3 / 4 + sqrt 2) :=
by
  sorry

end calculate_expression_l815_815312


namespace john_ate_80_percent_l815_815883

variable (original_steak : ℝ) (eaten_steak : ℝ)

def half_unburned_steak (w : ℝ) : ℝ := w / 2

theorem john_ate_80_percent 
  (h_original_steak : original_steak = 30)
  (h_half_burned : half_unburned_steak original_steak = 15)
  (h_eaten_steak : eaten_steak = 12)
  : (eaten_steak / (half_unburned_steak original_steak)) * 100 = 80 :=
by
  sorry

end john_ate_80_percent_l815_815883


namespace larger_number_l815_815971

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l815_815971
